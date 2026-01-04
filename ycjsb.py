# -*- coding: utf-8 -*-
"""
选股王 · 核武器版 (V30.25 STAR Only)
核心策略：MACD Rank 1 + 纯血科创板 (688)
目标：极致的动量，极致的赔率。拥抱波动，拥抱 20cm。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V30.25 核武器版 (科创专攻)", layout="wide")
st.title("🚀 V30.25 核武器版 (Only STAR 688)")
st.markdown("""
**💀 核心纪律：**
1.  **板块：** 仅限 **科创板 (688)**。创业板、主板一律不看。
2.  **门槛：** 最低股价 **20元**。
3.  **买入：** D+1 开盘价买入 (若开盘涨幅 > 1.5% 且非一字板)。
4.  **持有：**
    * **D+3 观察：** 浮盈 > 0，死拿到 D+5；浮盈 < 0，坚决止损。
    * **目标：** 捕捉单笔 +20% ~ +50% 的主升浪。
""")

# ---------------------------
# 全局缓存 & 工具
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 5)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns: return []
    return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()

# ----------------------------------------------------------------------
# 数据下载
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d") 
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    st.info(f"⏳ 正在下载 {start_date} 到 {end_date} 全市场数据...")

    adj_list, daily_list = [], []
    bar = st.progress(0)
    
    for i, date in enumerate(all_dates):
        try:
            cached = fetch_and_cache_daily_data(date)
            if not cached['adj'].empty: adj_list.append(cached['adj'])
            if not cached['daily'].empty: daily_list.append(cached['daily'])
            if i % 10 == 0: bar.progress((i+1)/len(all_dates))
        except: continue 
    bar.empty()

    if not adj_list or not daily_list: return False
        
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw = pd.concat(daily_list)
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'vol']:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    
    latest_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_date:
        GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_date), 'adj_factor'].droplevel(1).to_dict()
    
    return True

def get_qfq_data(ts_code, start_date, end_date):
    base_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code)
    if not base_adj: return pd.DataFrame()

    try:
        daily = GLOBAL_DAILY_RAW.loc[(ts_code, slice(start_date, end_date)), :]
        adj = GLOBAL_ADJ_FACTOR.loc[(ts_code, slice(start_date, end_date)), 'adj_factor']
    except: return pd.DataFrame()
    
    if daily.empty or adj.empty: return pd.DataFrame()
    
    df = daily.join(adj, how='left').dropna(subset=['adj_factor'])
    factor = df['adj_factor'] / base_adj
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns: df[col] = df[col] * factor
    
    df = df.reset_index()
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    return df.set_index('trade_date').sort_index()

# ----------------------------------------------------------------------
# 选股逻辑
# ----------------------------------------------------------------------
def compute_score(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data(ts_code, start_date, end_date)
    if df.empty or len(df) < 26: return 0
    
    close = df['close']
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd_val = (diff - dea) * 2
    
    score = (macd_val.iloc[-1] / close.iloc[-1]) * 100000
    if pd.isna(score): score = 0
    return score

# ----------------------------------------------------------------------
# 回测执行
# ----------------------------------------------------------------------
def run_backtest_on_date(date, min_price):
    daily = safe_get('daily', trade_date=date)
    if daily.empty: return None
    
    # --- 终极过滤：只看科创板 (688) ---
    pool = daily[daily['close'] >= min_price]
    pool = pool[pool['ts_code'].str.startswith('688')] # 纯血科创
    
    if pool.empty: return None
    
    pool = pool[pool['pct_chg'] > 0].sort_values('pct_chg', ascending=False)
    if len(pool) > 100: pool = pool.head(100)
    
    best_score = -1
    rank1_code = None
    rank1_close = 0
    
    for row in pool.itertuples():
        score = compute_score(row.ts_code, date)
        if score > best_score:
            best_score = score
            rank1_code = row.ts_code
            rank1_close = row.close
            
    if not rank1_code: return None
    
    # 模拟交易
    d0 = datetime.strptime(date, "%Y%m%d")
    start_fut = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_fut = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    hist = get_qfq_data(rank1_code, start_fut, end_fut)
    
    ret_d1, ret_d3, ret_d5 = np.nan, np.nan, np.nan
    
    if len(hist) >= 1:
        # D+1 开盘买入判定
        d1_row = hist.iloc[0]
        try:
            d1_raw = GLOBAL_DAILY_RAW.loc[(rank1_code, d1_row.name.strftime("%Y%m%d"))]
            if isinstance(d1_raw, pd.Series):
                open_pct = (d1_raw['open'] / d1_raw['pre_close'] - 1) * 100
            else:
                open_pct = 0 
        except:
            open_pct = 0
            
        if open_pct > 1.5:
            buy_price = d1_row['open']
            
            # D+1 收益
            ret_d1 = (d1_row['close'] / buy_price - 1) * 100
            
            # D+3 收益
            if len(hist) >= 3:
                ret_d3 = (hist.iloc[2]['close'] / buy_price - 1) * 100
            
            # D+5 收益
            if len(hist) >= 5:
                ret_d5 = (hist.iloc[4]['close'] / buy_price - 1) * 100
            elif len(hist) > 0:
                ret_d5 = (hist.iloc[-1]['close'] / buy_price - 1) * 100
        else:
            pass # 没买入
            
    return {
        'Trade_Date': date,
        'ts_code': rank1_code,
        'Close': rank1_close,
        'Score': best_score,
        'Return_D1': ret_d1,
        'Return_D3': ret_d3,
        'Return_D5': ret_d5
    }

# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测参数")
    end_date = st.date_input("结束日期", value=datetime.now().date())
    days_back = int(st.number_input("回测天数", value=50))
    MIN_PRICE = st.number_input("最低股价 (元)", value=20.0)
    
    st.markdown("---")
    TS_TOKEN = st.text_input("Tushare Token", type="password")

if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ---------------------------
# 主程序
# ---------------------------
if st.button("🚀 启动核武器 (Only 688)"):
    dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    if not dates: st.stop()
    if not get_all_historical_data(dates): st.stop()
    
    st.success(f"✅ 目标锁定：科创板 (688) | 价格 > {MIN_PRICE} | Rank 1")
    
    results = []
    bar = st.progress(0)
    
    for i, date in enumerate(dates):
        res = run_backtest_on_date(date, MIN_PRICE)
        if res:
            results.append(res)
        bar.progress((i+1)/len(dates))
    
    bar.empty()
    
    if not results:
        st.warning("没有选出符合条件的科创板股票。")
        st.stop()
        
    df_res = pd.DataFrame(results)
    valid_trades = df_res.dropna(subset=['Return_D1'])
    
    st.header("📊 核武器版 (Only 688) 最终报告")
    st.caption(f"回测区间: {dates[-1]} 至 {dates[0]} | 有效交易: {len(valid_trades)} 笔")
    
    col1, col2, col3 = st.columns(3)
    
    def get_metrics(col):
        if valid_trades.empty: return 0, 0
        avg = valid_trades[col].mean()
        win = (valid_trades[col] > 0).mean() * 100
        return avg, win
        
    d1_avg, d1_win = get_metrics('Return_D1')
    d3_avg, d3_win = get_metrics('Return_D3')
    d5_avg, d5_win = get_metrics('Return_D5')
    
    col1.metric("D+1 收益 / 胜率", f"{d1_avg:.2f}% / {d1_win:.1f}%")
    col2.metric("D+3 收益 / 胜率", f"{d3_avg:.2f}% / {d3_win:.1f}%")
    col3.metric("D+5 收益 / 胜率", f"{d5_avg:.2f}% / {d5_win:.1f}%")
    
    # 模拟 Hybrid 资金曲线
    if not valid_trades.empty:
        valid_trades['Return_Hybrid'] = np.where(valid_trades['Return_D3']>0, valid_trades['Return_D5'], valid_trades['Return_D3'])
        equity = (1 + valid_trades['Return_Hybrid']/100).cumprod()
        total_ret = (equity.iloc[-1] - 1) * 100
        st.metric("Hybrid 策略 (D3止盈止损) 总回报", f"{total_ret:.2f}%")
        st.line_chart(equity)
    
    st.subheader("📋 交易明细")
    st.dataframe(df_res, use_container_width=True)
