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
st.set_page_config(page_title="潜龙 V10·神龙摆尾", layout="wide")
st.title("🐉 潜龙 V10·神龙摆尾 (龙头首阴/N字战法)")
st.markdown("""
**策略核心：不追高，只做“龙回头”**
1.  **妖股基因**：过去 10 天内，必须有过 **涨停 (涨幅 > 9.5%)**。
2.  **缩量回调**：今日量能 < 涨停日量能的 **70%** (主力锁仓洗盘)。
3.  **黄金支撑**：股价回踩 **10日线 (MA10)** 附近获得支撑。
4.  **拒绝破位**：收盘价必须站稳 MA20 (趋势未坏)。
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
def calculate_sector_heat(df_daily, df_basic):
    if 'industry' not in df_daily.columns:
        df_merged = pd.merge(df_daily, df_basic[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df_merged = df_daily.copy()
    
    valid_df = df_merged[df_merged['pct_chg'] != 0]
    sector_stats = valid_df.groupby(['trade_date', 'industry'])['pct_chg'].mean().reset_index()
    sector_stats.rename(columns={'pct_chg': 'sector_pct'}, inplace=True)
    df_final = pd.merge(df_merged, sector_stats, on=['trade_date', 'industry'], how='left')
    return df_final

def calculate_strategy(df, days_lookback):
    """
    V10 核心逻辑: 龙回头
    """
    # 1. 均线系统
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    
    # 2. 寻找过去的涨停板
    # 标记涨停 (>9.5%)
    df['is_limit_up'] = df['pct_chg'] > 9.5
    
    # 滑动窗口判断过去 N 天是否有涨停 (不含今天)
    # rolling sum on boolean gives count
    df['limit_up_count'] = df.groupby('ts_code')['is_limit_up'].transform(
        lambda x: x.shift(1).rolling(days_lookback).sum()
    )
    
    # 3. 获取涨停那天的成交量 (取最近一次涨停的 vol)
    # 这是一个近似处理，为了速度，我们对比 10日最大量
    df['max_vol_10'] = df.groupby('ts_code')['vol'].transform(
        lambda x: x.shift(1).rolling(days_lookback).max()
    )
    
    # === 信号判定 ===
    
    # A. 妖股基因: 过去10天至少有1个涨停
    cond_gene = df['limit_up_count'] >= 1
    
    # B. 缩量回调: 今日成交量 < 10日最大量 * 0.7 (且越小越好)
    cond_vol = df['vol'] < (df['max_vol_10'] * 0.7)
    
    # C. 黄金支撑: 
    # 回踩 MA10: 最低价 <= MA10 * 1.02 (触及或跌破)
    # 也就是股价离 MA10 很近了
    cond_support = (df['low'] <= df['ma10'] * 1.02) & (df['close'] > df['ma20'])
    
    # D. 拒绝大跌: 今日跌幅不能太惨 (最好是小阴小阳)
    # 跌幅不能超过 -5%，如果是红盘更好
    cond_shape = df['pct_chg'] > -5.0
    
    # E. 趋势未坏: MA20 向上 (这里简单用收盘价 > MA20)
    cond_trend = df['close'] > df['ma20']
    
    # F. 流动性
    cond_mv = (df['amount'] > 50000) & (df['amount'] < 5000000)
    
    df['is_signal'] = cond_gene & cond_vol & cond_support & cond_shape & cond_trend & cond_mv
    
    return df

def calculate_score(row):
    score = 60
    
    # 离 MA10 越近越好 (性价比高)
    dist_ma10 = abs(row['close'] - row['ma10']) / row['ma10']
    if dist_ma10 < 0.02: score += 20
    
    # 缩量越极致越好
    if row['vol'] < row['max_vol_10'] * 0.5: score += 10
    
    # 板块如果今天也是绿的(洗盘)，反而好
    if -2.0 < row['sector_pct'] < 1.0: score += 10
        
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V10 神龙摆尾参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    days_lookback = st.slider("追溯几天内的涨停?", 3, 15, 10)
    top_n = st.number_input("每日优选 (Top N)", 1, 20, 2)
    
    run_btn = st.button("🚀 启动龙回头回测")

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
    with st.spinner("正在寻找回踩到位的真龙..."):
        df_sector = calculate_sector_heat(df_all, df_basic)
        df_calc = calculate_strategy(df_sector, days_lookback)
        
    # 4. 结果
    st.markdown("### 🐉 V10 诊断")
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    df_signals = df_window[df_window['is_signal']].copy()
    st.write(f"⚪ 涨停回调标的: **{len(df_signals)}** 个")
    
    if df_signals.empty:
        st.warning("无信号。")
        return

    # 5. 评分与 Top N
    df_signals['潜龙分'] = df_signals.apply(calculate_score, axis=1)
    df_signals = df_signals.sort_values(['trade_date', '潜龙分'], ascending=[True, False])
    df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
    
    df_top = df_signals[df_signals['排名'] <= top_n].copy()
    
    # 6. 回测
    price_lookup = df_calc[['ts_code', 'trade_date', 'open', 'close', 'low']].set_index(['ts_code', 'trade_date'])
    trades = []
    
    progress = st.progress(0)
    total_sig = len(df_top)
    
    for i, row in enumerate(df_top.itertuples()):
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
            '信号日': signal_date, '代码': code, '名称': row.name, '排名': row.排名,
            '行业': row.industry, '板块涨幅': f"{row.sector_pct:.1f}%",
            '收盘价': row.close,
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
        
        st.markdown(f"### 📊 V10 (神龙摆尾) 回测结果 (Top {top_n})")
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
