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
st.set_page_config(page_title="潜龙 V12·皇冠明珠", layout="wide")
st.title("🐉 潜龙 V12·皇冠明珠 (Top3板块 + 唯一真龙)")
st.markdown("""
**策略核心：回归初心，只做最强**
1.  **极窄赛道**：只扫描当日 **涨幅前 3 名** 的板块 (资金最集中的战场)。
2.  **唯一真龙**：只选板块内 **涨幅第 1 名** 的领头羊 (龙头战法)。
3.  **严格验身**：
    * **换手率 > 5%** (有人气，非一字躺赢)。
    * **量比 > 1.8** (资金主攻)。
    * **拒绝上影** (收盘强劲，无抛压)。
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
            # 获取日线行情
            df = pro.daily(trade_date=date)
            # 获取每日指标(换手率等)
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,volume_ratio')
            
            if not df.empty and not df_basic.empty:
                # 合并行情和指标
                df = pd.merge(df, df_basic, on='ts_code', how='left')
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
def calculate_strategy(df_all, df_basic, top_k_sector):
    """
    V12 核心逻辑: Top Sector -> Top Stock
    """
    # 1. 预处理
    if 'industry' not in df_all.columns:
        df_merged = pd.merge(df_all, df_basic[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df_merged = df_all.copy()
    
    # 2. RSI计算 (辅助判断强度)
    df_merged['up_move'] = np.where(df_merged['pct_chg'] > 0, df_merged['pct_chg'], 0)
    df_merged['down_move'] = np.where(df_merged['pct_chg'] < 0, abs(df_merged['pct_chg']), 0)
    avg_up = df_merged.groupby('ts_code')['up_move'].transform(lambda x: x.rolling(6).mean())
    avg_down = df_merged.groupby('ts_code')['down_move'].transform(lambda x: x.rolling(6).mean())
    df_merged['rsi_6'] = 100 * avg_up / (avg_up + avg_down + 0.0001)

    results = []
    dates = sorted(df_merged['trade_date'].unique())
    
    # 循环每一天
    for i in range(10, len(dates)):
        curr_date = dates[i]
        daily_data = df_merged[df_merged['trade_date'] == curr_date].copy()
        
        if daily_data.empty: continue
        
        # === Step 1: 决出板块前三名 ===
        sector_stats = daily_data.groupby('industry').agg({
            'pct_chg': 'mean',
            'ts_code': 'count',
            'amount': 'sum'
        }).reset_index()
        
        # 过滤微型板块 (股票数>5) 和 成交过小的板块
        sector_stats = sector_stats[(sector_stats['ts_code'] > 5) & (sector_stats['amount'] > 100000)]
        
        # 排序取 Top 3
        top_sectors = sector_stats.sort_values('pct_chg', ascending=False).head(top_k_sector)
        top_sector_names = top_sectors['industry'].tolist()
        
        if not top_sector_names: continue
        
        # === Step 2: 决出板块内的真龙 ===
        for sec_name in top_sector_names:
            sec_data = daily_data[daily_data['industry'] == sec_name].copy()
            sec_gain = top_sectors[top_sectors['industry'] == sec_name]['pct_chg'].values[0]
            
            # 必须是大涨的板块 (>1.5%)，否则没意义
            if sec_gain < 1.5: continue
            
            # 板块内排序
            sec_data = sec_data.sort_values('pct_chg', ascending=False)
            
            # 取第一名 (Leader)
            if sec_data.empty: continue
            leader = sec_data.iloc[0]
            
            # === Step 3: 严格验身 (Strict Filter) ===
            # 1. 涨幅够大 (必须 > 5%)
            if leader['pct_chg'] < 5.0: continue
            
            # 2. 换手率 (Turnover > 5%) - 必须活跃
            # 注意: 如果当天一字板缩量，turnover可能低，我们要抓的是"换手龙"，不是"一字龙"(买不进)
            if leader['turnover_rate'] < 5.0: continue
            
            # 3. 量比 (Volume Ratio > 1.8) - 资金攻击
            if leader['volume_ratio'] < 1.8: continue
            
            # 4. 拒绝长上影 (收盘价接近最高价)
            # (High - Close) / Close < 1%
            upper_shadow = (leader['high'] - leader['close']) / leader['close']
            if upper_shadow > 0.015: continue
            
            # 5. RSI 强度
            if leader['rsi_6'] < 70: continue

            results.append({
                'ts_code': leader['ts_code'],
                'trade_date': curr_date,
                'name': leader['name'],
                'industry': sec_name,
                'sector_pct': sec_gain,
                'pct_chg': leader['pct_chg'],
                'turnover': leader['turnover_rate'],
                'vol_ratio': leader['volume_ratio'],
                'close': leader['close'],
                'is_signal': True
            })
            
    return pd.DataFrame(results)

def calculate_score(row):
    # 简单的双重得分
    return row['sector_pct'] + row['pct_chg']

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V12 皇冠明珠参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("🔥 筛选标准")
    top_k_sector = st.number_input("锁定板块前几名?", 1, 5, 3)
    
    run_btn = st.button("🚀 启动 V12 回测")

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
    with st.spinner("正在加冕皇冠明珠..."):
        df_calc = calculate_strategy(df_all, df_basic, top_k_sector)
        
    # 4. 结果
    st.markdown("### 🐉 V12 诊断 (Top3板块+真龙)")
    
    if df_calc.empty:
        st.warning("无信号。近期缺乏板块效应。")
        return
        
    # 过滤时间窗
    valid_dates = cal_dates[-(days_back):] 
    df_signals = df_calc[df_calc['trade_date'].isin(valid_dates)].copy()
    
    st.write(f"⚪ 捕获明珠: **{len(df_signals)}** 个")

    # 5. 评分与 Top N
    df_signals['潜龙分'] = df_signals.apply(calculate_score, axis=1)
    df_signals = df_signals.sort_values(['trade_date', 'sector_pct'], ascending=[True, False])
    
    # 6. 回测
    # 需要重新构建 lookup
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
            '换手%': f"{row.turnover:.1f}",
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
        
        st.markdown(f"### 📊 V12 (皇冠明珠) 回测结果")
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
        
        st.dataframe(df_res.sort_values(['信号日'], ascending=False), use_container_width=True)
    else:
        st.warning("无交易")

if run_btn:
    run_analysis()
