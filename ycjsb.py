# -*- coding: utf-8 -*-
"""
选股王 · V30.25 终极实战版 (突破买入法)
策略：双创组合 (688 + 300)
买入修正：
1. 必须高开 (Open > Pre_Close)。
2. 盘中必须涨幅达到开盘价的 1.5% (Price >= Open * 1.015) 才触发买入。
3. 买入价格按 Open * 1.015 计算。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V30.25 突破买入版", layout="wide")
st.title("🛡️ V30.25 突破买入版 (确认强势再上车)")
st.markdown("""
**核心逻辑修正：**
* **观察：** 竞价必须高开 (Open > Pre_Close)。
* **买入：** 盘中价格突破 **开盘价 + 1.5%** 时触发买入。
* **目的：** 过滤“高开低走”的骗线，只做真突破。
""")

# ---------------------------
# 全局缓存
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
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3)).strftime("%Y%m%d")
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
    
    if not daily_df.empty:
        daily_df = daily_df[daily_df['ts_code'].str.startswith(('30', '688'))]
    if not adj_df.empty:
        adj_df = adj_df[adj_df['ts_code'].str.startswith(('30', '688'))]
        
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d") 
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    st.info(f"⏳ 正在拉取数据 ({start_date} ~ {end_date})...")

    adj_list, daily_list = [], []
    bar = st.progress(0)
    
    for i, date in enumerate(all_dates):
        try:
            cached = fetch_and_cache_daily_data(date)
            if not cached['adj'].empty: adj_list.append(cached['adj'])
            if not cached['daily'].empty: daily_list.append(cached['daily'])
            if i % 20 == 0: bar.progress((i+1)/len(all_dates))
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
# 评分逻辑
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
# 回测主逻辑 (修正版)
# ----------------------------------------------------------------------
def run_backtest_on_date(date, min_price):
    try:
        daily = GLOBAL_DAILY_RAW.xs(date, level='trade_date')
    except KeyError: return None
    if daily.empty: return None
    
    pool = daily[daily['close'] >= min_price]
    if pool.empty: return None
    
    pool = pool[pool['pct_chg'] > 0].sort_values('pct_chg', ascending=False)
    if len(pool) > 150: pool = pool.head(150)
    
    best_score = -1
    rank1_code = None
    rank1_close = 0
    
    for row in pool.itertuples():
        score = compute_score(row.Index, date)
        if score > best_score:
            best_score = score
            rank1_code = row.Index
            rank1_close = row.close
            
    if not rank1_code: return None
    
    d0 = datetime.strptime(date, "%Y%m%d")
    start_fut = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_fut = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    hist = get_qfq_data(rank1_code, start_fut, end_fut)
    
    ret_d1, ret_d3, ret_d5 = np.nan, np.nan, np.nan
    buy_triggered = False
    
    if len(hist) >= 1:
        d1_row = hist.iloc[0]
        
        # 获取 D1 当天的原始行情 (Open, High, Pre_Close)
        try:
            d1_raw = GLOBAL_DAILY_RAW.loc[(rank1_code, d1_row.name.strftime("%Y%m%d"))]
            if isinstance(d1_raw, pd.Series):
                d1_open = d1_raw['open']
                d1_high = d1_raw['high']
                d1_pre = d1_raw['pre_close']
                d1_close = d1_raw['close']
                
                # --- 买入条件判定 ---
                # 1. 竞价必须高开
                if d1_open > d1_pre:
                    # 2. 盘中必须触及 Open * 1.015
                    target_buy_price_raw = d1_open * 1.015
                    
                    if d1_high >= target_buy_price_raw:
                        # 触发买入！
                        buy_triggered = True
                        
                        # 计算复权后的买入成本
                        # 注意：hist 数据是复权后的，我们要按比例换算买入价
                        # 复权因子 = hist_open / raw_open
                        adj_ratio = d1_row['open'] / d1_open
                        buy_price_adj = target_buy_price_raw * adj_ratio
                        
                        # 3. 计算收益 (相对于买入成本)
                        ret_d1 = (d1_row['close'] / buy_price_adj - 1) * 100
                        
                        if len(hist) >= 3:
                            ret_d3 = (hist.iloc[2]['close'] / buy_price_adj - 1) * 100
                        if len(hist) >= 5:
                            ret_d5 = (hist.iloc[4]['close'] / buy_price_adj - 1) * 100
                        elif len(hist) > 0:
                            ret_d5 = (hist.iloc[-1]['close'] / buy_price_adj - 1) * 100
            else:
                pass
        except:
            pass
            
    if not buy_triggered:
        # 如果没触发买入，返回 None (或者记录为“空仓”)
        return None 
    
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
    st.header("1. 回测设置")
    end_date = st.date_input("结束日期", value=datetime.now().date())
    days_back = int(st.number_input("回测天数", value=200))
    
    st.markdown("---")
    st.header("2. 策略参数")
    MIN_PRICE = st.number_input("最低股价 (元)", value=20.0)
    
    st.markdown("---")
    TS_TOKEN = st.text_input("Tushare Token", type="password")

if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ---------------------------
# 主程序
# ---------------------------
if st.button("🚀 运行 (突破买入版)"):
    dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    if not dates: st.stop()
    if not get_all_historical_data(dates): st.stop()
    
    st.success(f"✅ 策略：高开且盘中上涨 1.5% 买入")
    
    results = []
    bar = st.progress(0)
    
    for i, date in enumerate(dates):
        res = run_backtest_on_date(date, MIN_PRICE)
        if res:
            results.append(res)
        bar.progress((i+1)/len(dates))
    
    bar.empty()
    
    if not results:
        st.warning("没有触发买入条件的交易。")
        st.stop()
        
    df_res = pd.DataFrame(results)
    valid_trades = df_res.dropna(subset=['Return_D1'])
    
    st.header("📊 V30.25 实战报告 (突破买入)")
    st.caption(f"区间: {dates[-1]} ~ {dates[0]} | 触发交易: {len(valid_trades)}")
    
    col1, col2, col3 = st.columns(3)
    def get_m(col):
        if valid_trades.empty: return 0, 0
        return valid_trades[col].mean(), (valid_trades[col]>0).mean()*100
    
    d1_a, d1_w = get_m('Return_D1')
    d3_a, d3_w = get_m('Return_D3')
    d5_a, d5_w = get_m('Return_D5')
    
    col1.metric("D+1 收益/胜率", f"{d1_a:.2f}% / {d1_w:.1f}%")
    col2.metric("D+3 收益/胜率", f"{d3_a:.2f}% / {d3_w:.1f}%")
    col3.metric("D+5 收益/胜率", f"{d5_a:.2f}% / {d5_w:.1f}%")
    
    st.subheader("📋 交易明细")
    st.dataframe(df_res.round(2), use_container_width=True)
    
    csv = df_res.to_csv().encode('utf-8')
    st.download_button("📥 下载 CSV", csv, "v30.25_breakout_export.csv", "text/csv")
