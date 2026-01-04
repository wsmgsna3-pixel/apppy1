# -*- coding: utf-8 -*-
"""
选股王 · V30.25 终极监控版 (选股+回测二合一)
策略：双创组合 (688 + 300)
特性：
1. 全天候监控：无论是否触发买入，均显示当日 Rank 1。
2. 智能信号：明确区分“✅ 买入”、“👀 观望(太弱)”、“⚠️ 观望(太强)”。
3. 自动寻历：自动匹配最近交易日。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V30.25 选股监控台", layout="wide")
st.title("🔭 V30.25 选股监控台 (含未成交记录)")
st.markdown("""
**策略逻辑 (Rank 1 + 黄金区间):**
* **选股：** 每日选出 **Score 第一名** 的双创股票。
* **决策：** * ✅ **买入**：竞价高开 **[+2.0%, +7.5%]**。
    * 👀 **观望**：高开不足或过高 (防止大面)。
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
    # 向前多找一些日子，防止长假导致天数不够
    start_search = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=max(num_days * 3, 30))).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_search, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns: return []
    # 取最近的 num_days 个交易日
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
    
    # 稍微多拉取一点数据用于计算指标
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d") 
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    st.info(f"⏳ 正在扫描市场数据 ({start_date} ~ {end_date})...")

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
# 回测主逻辑 (含监控)
# ----------------------------------------------------------------------
def run_backtest_on_date(date, min_price):
    try:
        daily = GLOBAL_DAILY_RAW.xs(date, level='trade_date')
    except KeyError: return None
    if daily.empty: return None
    
    # 选股逻辑
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
    
    # 判断买入条件
    open_pct = 0.0
    signal_type = "未知"
    is_buy = False
    
    try:
        d1_raw = GLOBAL_DAILY_RAW.loc[(rank1_code, date)]
        if isinstance(d1_raw, pd.Series):
            open_pct = (d1_raw['open'] / d1_raw['pre_close'] - 1) * 100
            
            # --- 核心判断 ---
            if 2.0 <= open_pct <= 7.5:
                is_buy = True
                signal_type = "✅ BUY (触发买入)"
            elif open_pct < 2.0:
                signal_type = "👀 WATCH (高开不足)"
            else:
                signal_type = "⚠️ WATCH (高开过高)"
    except:
        pass

    # 计算收益 (仅当 is_buy = True 时计算，否则为 None)
    ret_strategy = None
    
    if is_buy:
        d0 = datetime.strptime(date, "%Y%m%d")
        start_fut = (d0 + timedelta(days=1)).strftime("%Y%m%d")
        end_fut = (d0 + timedelta(days=20)).strftime("%Y%m%d")
        
        hist_d1 = get_qfq_data(rank1_code, date, date)
        hist_fut = get_qfq_data(rank1_code, start_fut, end_fut)
        
        if not hist_d1.empty:
            buy_price = hist_d1.iloc[0]['open']
            
            # 策略收益
            if rank1_code.startswith('30'): # 创业板 D2 Open 跑
                if len(hist_fut) >= 1:
                    sell_price = hist_fut.iloc[0]['open']
                    ret_strategy = (sell_price / buy_price - 1) * 100
            elif rank1_code.startswith('688'): # 科创板 D5 Close 跑
                if len(hist_fut) >= 4:
                    sell_price = hist_fut.iloc[3]['close']
                    ret_strategy = (sell_price / buy_price - 1) * 100
                elif len(hist_fut) > 0:
                    sell_price = hist_fut.iloc[-1]['close']
                    ret_strategy = (sell_price / buy_price - 1) * 100

    return {
        'Trade_Date': date,
        'ts_code': rank1_code,
        'Name': '加载中...', 
        'Signal': signal_type,
        'Open_Pct': open_pct,
        'Close': rank1_close,
        'Score': best_score,
        'Return_Strategy': ret_strategy
    }

# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 选股/回测设置")
    # 自动设置为今天 (如果是盘后)
    default_date = datetime.now().date()
    end_date = st.date_input("结束日期 (自动定位最近交易日)", value=default_date)
    days_back = int(st.number_input("回测天数 (1=今日选股)", value=1))
    
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
if st.button("🚀 开始扫描"):
    # 1. 获取交易日历
    dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    
    if not dates: 
        st.error(f"❌ 在 {end_date} 之前找不到交易日数据，请检查日期设置。")
        st.stop()
        
    st.info(f"📅 正在分析: {dates[-1]} ~ {dates[0]} (共 {len(dates)} 个交易日)")
    
    # 2. 拉取数据
    if not get_all_historical_data(dates): st.stop()
    
    # 3. 逐日分析
    results = []
    bar = st.progress(0)
    
    for i, date in enumerate(dates):
        res = run_backtest_on_date(date, MIN_PRICE)
        if res:
            results.append(res)
        bar.progress((i+1)/len(dates))
    
    bar.empty()
    
    if not results:
        st.warning("⚠️ 数据不足，无法计算。")
        st.stop()
        
    df_res = pd.DataFrame(results)
    
    # 4. 展示结果
    
    # A. 核心统计 (只统计实际买入的)
    executed_trades = df_res[df_res['Signal'].str.contains('BUY')]
    
    if not executed_trades.empty:
        st.header("💰 实战收益统计 (仅含已成交)")
        col1, col2, col3 = st.columns(3)
        avg_ret = executed_trades['Return_Strategy'].mean()
        win_rate = (executed_trades['Return_Strategy'] > 0).mean() * 100
        count = len(executed_trades)
        
        col1.metric("成交笔数", f"{count}")
        col2.metric("策略平均收益", f"{avg_ret:.2f}%")
        col3.metric("策略胜率", f"{win_rate:.1f}%")
    else:
        st.info("💡 选定区间内无【符合买入条件】的股票。")

    # B. 每日选股监控 (含观望)
    st.header("📋 每日选股监控 (含未成交)")
    
    # 颜色高亮函数
    def highlight_signal(val):
        if 'BUY' in str(val):
            return 'color: red; font-weight: bold'
        elif 'WATCH' in str(val):
            return 'color: gray'
        return ''

    # 安全的格式化函数 (防止空值报错)
    def safe_format(val):
        if val is None or pd.isna(val):
            return "-"
        return f"{val:.2f}%"

    st.dataframe(
        df_res[['Trade_Date', 'ts_code', 'Signal', 'Open_Pct', 'Return_Strategy', 'Close', 'Score']]
        .style
        .map(highlight_signal, subset=['Signal'])
        .format({
            'Open_Pct': safe_format,
            'Return_Strategy': safe_format,
            'Score': '{:.0f}'
        }),
        use_container_width=True
    )
    
    csv = df_res.to_csv().encode('utf-8')
    st.download_button("📥 下载完整监控表 CSV", csv, "v30.25_monitor_export.csv", "text/csv")
