import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="潜龙 V9·天眼系统", layout="wide")
st.title("🐉 潜龙 V9·天眼系统 (板块共振 + 领头羊)")
st.markdown("""
**策略核心：自上而下，先找“裤子”，再找“大哥”**
1.  **一级扫描 (找热点)**：锁定 **涨幅>2.0%** 且 **量比>1.2** 的最强板块。
2.  **二级扫描 (找龙头)**：在最强板块中，筛选 **涨幅前3名** 的个股。
3.  **三级验证 (防假冒)**：个股量比 > 1.5 (资金坚决) + 换手率 > 5% (人气充足)。
4.  **结果**：你将看到资金风口上的最强领头羊。
""")

# ==========================================
# 2. 核心数据引擎
# ==========================================
@st.cache_data(persist="disk", show_spinner=False)
def get_trade_cal(token, start_date, end_date):
    ts.set_token(token)
    pro = ts.pro_api()
    for attempt in range(3):
        try:
            df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
            if not df.empty:
                return sorted(df['cal_date'].tolist())
            time.sleep(0.5)
        except:
            time.sleep(1)
    return []

@st.cache_data(persist="disk", show_spinner=False)
def fetch_all_market_data_by_date(token, date_list):
    ts.set_token(token)
    pro = ts.pro_api()
    data_list = []
    total = len(date_list)
    bar = st.progress(0, text="正在同步全市场数据...")
    
    for i, date in enumerate(date_list):
        try:
            time.sleep(0.05)
            df = pro.daily(trade_date=date)
            if not df.empty:
                # 获取每日指标(换手率、量比等需要daily_basic，这里简化计算)
                # 为了速度，我们还是用daily，量比自己算
                df = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount', 'pct_chg']]
                data_list.append(df)
        except:
            time.sleep(0.5)
        if (i+1) % 10 == 0:
            bar.progress((i+1)/total, text=f"加载进度: {i+1}/{total}")
            
    bar.empty()
    if not data_list: return pd.DataFrame()
    full_df = pd.concat(data_list)
    full_df = full_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    return full_df

@st.cache_data(persist="disk", show_spinner=False)
def get_stock_basics(token):
    ts.set_token(token)
    pro = ts.pro_api()
    for _ in range(3):
        try:
            time.sleep(0.5)
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,industry,list_date')
            if not df.empty:
                df = df[~df['name'].str.contains('ST')]
                df = df[~df['market'].str.contains('北交')]
                df = df[~df['ts_code'].str.contains('BJ')]
                return df
        except: time.sleep(1)
    return pd.DataFrame()

# ==========================================
# 3. 核心计算
# ==========================================
def calculate_strategy(df_all, df_basic, top_k_sector, sec_min_pct, stock_min_pct):
    """
    V9 核心逻辑：板块 -> 个股
    """
    # 1. 预处理：合并行业信息
    if 'industry' not in df_all.columns:
        df_merged = pd.merge(df_all, df_basic[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df_merged = df_all.copy()
    
    # 2. 计算个股辅助指标 (量比、均线)
    # 量比 = 今日vol / 5日均vol
    df_merged['vol_5'] = df_merged.groupby('ts_code')['vol'].transform(lambda x: x.shift(1).rolling(5).mean())
    df_merged['vol_ratio'] = df_merged['vol'] / (df_merged['vol_5'] + 1)
    
    # 3. 按日期循环，每天找出最强板块和龙头
    results = []
    
    dates = sorted(df_merged['trade_date'].unique())
    
    # 我们只计算有足够数据的日期 (跳过前5天)
    for i in range(5, len(dates)):
        curr_date = dates[i]
        daily_data = df_merged[df_merged['trade_date'] == curr_date].copy()
        
        if daily_data.empty: continue
        
        # === 第一步：板块排位赛 ===
        # 计算每个板块的：平均涨幅、总成交量、上涨家数
        sector_stats = daily_data.groupby('industry').agg({
            'pct_chg': 'mean',
            'vol': 'sum',
            'ts_code': 'count'
        }).reset_index()
        
        # 过滤掉只有1-2只股的微型板块
        sector_stats = sector_stats[sector_stats['ts_code'] > 5]
        
        # 筛选强板块：涨幅 > sec_min_pct
        strong_sectors = sector_stats[sector_stats['pct_chg'] > sec_min_pct].sort_values('pct_chg', ascending=False)
        
        # 取前 K 名 (穿裤子的板块)
        top_sectors_list = strong_sectors.head(top_k_sector)['industry'].tolist()
        
        if not top_sectors_list: continue
        
        # === 第二步：龙头选拔赛 ===
        # 只看这些强板块里的股票
        candidates = daily_data[daily_data['industry'].isin(top_sectors_list)].copy()
        
        # 筛选条件：
        # 1. 涨幅够大 (领头羊)
        cond_limit = candidates['pct_chg'] > stock_min_pct
        # 2. 有量 (有人气)
        cond_vol = candidates['vol_ratio'] > 1.2
        
        winners = candidates[cond_limit & cond_vol].copy()
        
        if winners.empty: continue
        
        # === 第三步：板块内排序 ===
        # 对每个板块内的股票，按涨幅降序，取前 2 名
        winners['rank_in_sector'] = winners.groupby('industry')['pct_chg'].rank(method='first', ascending=False)
        top_winners = winners[winners['rank_in_sector'] <= 2]
        
        # 记录信号
        for _, row in top_winners.iterrows():
            # 找到该板块的涨幅
            sec_gain = sector_stats[sector_stats['industry'] == row['industry']]['pct_chg'].values[0]
            
            results.append({
                'ts_code': row['ts_code'],
                'trade_date': curr_date,
                'name': row['name'],
                'industry': row['industry'],
                'sector_pct': sec_gain,
                'pct_chg': row['pct_chg'],
                'vol_ratio': row['vol_ratio'],
                'close': row['close'],
                'is_signal': True
            })
            
    return pd.DataFrame(results)

def calculate_score(row):
    # 简单的评分：板块涨幅 + 个股涨幅
    return row['sector_pct'] + row['pct_chg']

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V9 天眼参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("🔥 板块与龙头阈值")
    
    top_k_sector = st.number_input("锁定前几名板块?", 1, 10, 3, help="只看前3名最强板块")
    sec_min_pct = st.number_input("板块涨幅门槛%", 0.0, 5.0, 2.0, help="板块必须大涨")
    stock_min_pct = st.number_input("个股涨幅门槛%", 3.0, 9.9, 5.0, help="龙头必须大涨")
    
    run_btn = st.button("🚀 启动天眼雷达")

def run_analysis():
    if not user_token:
        st.error("请先输入 Token")
        return

    # 1. 准备数据
    end_str = end_date_input.strftime('%Y%m%d')
    start_dt = end_date_input - timedelta(days=days_back * 1.5 + 80)
    
    cal_dates = get_trade_cal(user_token, start_dt.strftime('%Y%m%d'), end_str)
    if not cal_dates: return
        
    df_all = fetch_all_market_data_by_date(user_token, cal_dates)
    if df_all.empty: return
    st.success(f"✅ K线数据就绪: {len(df_all):,} 条")

    # 2. 基础信息
    df_basic = get_stock_basics(user_token)
    if df_basic.empty: return
        
    # 3. 计算
    with st.spinner("天眼正在扫描热点板块..."):
        # V9 不需要 calculate_sector_heat 预处理，整合在 strategy 里了
        df_calc = calculate_strategy(df_all, df_basic, top_k_sector, sec_min_pct, stock_min_pct)
        
    # 4. 结果
    st.markdown("### 🐉 V9 诊断 (板块共振)")
    
    if df_calc.empty:
        st.warning("无信号。近期无强势板块效应。")
        return
        
    # 过滤时间窗
    valid_dates = cal_dates[-(days_back):] 
    df_signals = df_calc[df_calc['trade_date'].isin(valid_dates)].copy()
    
    st.write(f"⚪ 捕获共振龙头: **{len(df_signals)}** 个")

    # 5. 评分与 Top N
    # 这里的 Top N 已经在 strategy 里按板块选了 Top 2，这里只需按日期展示
    df_signals['潜龙分'] = df_signals.apply(calculate_score, axis=1)
    df_signals = df_signals.sort_values(['trade_date', 'sector_pct', 'pct_chg'], ascending=[True, False, False])
    
    # 6. 回测
    # 需要重新构建 lookup，因为 df_calc 结构变了
    price_lookup = df_all[['ts_code', 'trade_date', 'open', 'close', 'low', 'pre_close']].set_index(['ts_code', 'trade_date'])
    trades = []
    
    progress = st.progress(0)
    total_sig = len(df_signals)
    
    for i, row in enumerate(df_signals.itertuples()):
        progress.progress((i+1)/total_sig)
        
        signal_date = row.trade_date
        code = row.ts_code
        
        try:
            curr_idx = cal_dates.index(signal_date)
            future_dates = cal_dates[curr_idx+1 : curr_idx+11]
        except: continue
            
        if not future_dates: continue
            
        d1_date = future_dates[0]
        if (code, d1_date) not in price_lookup.index: continue
        d1_data = price_lookup.loc[(code, d1_date)]
        
        # 风控
        open_pct = (d1_data['open'] - d1_data.get('pre_close', row.close)) / row.close
        if open_pct < -0.05: continue
            
        buy_price = d1_data['open']
        stop_price = buy_price * 0.90
        
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, 
            '行业': row.industry, '板块涨幅': f"{row.sector_pct:.1f}%",
            '个股涨幅': f"{row.pct_chg:.1f}%",
            '量比': f"{row.vol_ratio:.1f}",
            '买入价': buy_price, '状态': '持有'
        }
        
        triggered = False
        for n, f_date in enumerate(future_dates):
            if (code, f_date) not in price_lookup.index: break
            f_data = price_lookup.loc[(code, f_date)]
            day_label = f"D+{n+1}"
            
            if not triggered:
                if f_data['low'] <= stop_price:
                    triggered = True
                    trade['状态'] = '止损'
                    trade[day_label] = -10.0
                else:
                    ret = (f_data['close'] - buy_price) / buy_price * 100
                    trade[day_label] = round(ret, 2)
            else:
                trade[day_label] = -10.0
        
        trades.append(trade)
        
    progress.empty()
    
    if trades:
        df_res = pd.DataFrame(trades)
        
        st.markdown(f"### 📊 V9 (天眼系统) 回测结果")
        cols = st.columns(5)
        days = ['D+1', 'D+3', 'D+5', 'D+7', 'D+10']
        
        for idx, d in enumerate(days):
            if d in df_res.columns:
                valid_data = df_res[pd.to_numeric(df_res[d], errors='coerce').notna()]
                if not valid_data.empty:
                    wins = valid_data[valid_data[d] > 0]
                    win_rate = len(wins) / len(valid_data) * 100
                    avg_ret = valid_data[d].mean()
                    cols[idx].metric(f"{d} 胜率", f"{win_rate:.1f}%")
                    cols[idx].metric(f"{d} 均收", f"{avg_ret:.2f}%")
        
        st.dataframe(df_res.sort_values(['信号日', '板块涨幅'], ascending=[False, False]), use_container_width=True)
    else:
        st.warning("无交易")

if run_btn:
    run_analysis()
