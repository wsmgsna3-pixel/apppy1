# -*- coding: utf-8 -*-
"""
选股王 · V30.25 经典复刻版 (内核V30.24 + 经典仪表盘)
核心理念：
1. [内核] 沿用 V30.24 的全市场扫描、纯粹评分(MACD/Price)、大盘风控、去一字板。
2. [界面] 恢复 V30.22 的经典指标卡设计 (D+1/D+3/D+5)。
3. [灵活] 侧边栏增加"每日持仓数量"控制，由用户决定只买第一名还是前三名。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import time

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="选股王 · V30.25 经典复刻版", layout="wide")
st.title("选股王 · V30.25 经典复刻版 (🛡️ 风控内核 + 📊 经典报表)")
st.markdown("""
**📝 版本说明：**
* **内核：** 保持 V30.24 的最强逻辑 (全扫描 + 纯评分 + 大盘风控)。
* **交互：** 侧边栏可调整 **Top K** (建议设为 1)。
* **展示：** 恢复经典的收益率/胜率仪表盘。
""")

# ---------------------------
# 全局缓存
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_INDEX_DATA = pd.DataFrame() 

# ---------------------------
# 基础工具函数
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        if kwargs.get('is_index'): df = pro.index_daily(**kwargs)
        else: df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 5)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历。")
        return []
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
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_INDEX_DATA
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 1. 下载大盘数据 (上证指数) 用于风控
    start_date_idx = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
    end_date_idx = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=10)).strftime("%Y%m%d")
    
    with st.spinner("正在获取大盘指数数据..."):
        GLOBAL_INDEX_DATA = safe_get('index_daily', ts_code='000001.SH', start_date=start_date_idx, end_date=end_date_idx)
        if not GLOBAL_INDEX_DATA.empty:
            GLOBAL_INDEX_DATA = GLOBAL_INDEX_DATA.sort_values('trade_date').set_index('trade_date')
            GLOBAL_INDEX_DATA['ma20'] = GLOBAL_INDEX_DATA['close'].rolling(window=20).mean()

    # 2. 下载个股数据
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d") 
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    st.info(f"⏳ 正在按日期下载 {start_date} 到 {end_date} 全市场数据...")

    adj_list, daily_list = [], []
    download_progress = st.progress(0, text="下载进度...")
    
    total_dates = len(all_dates)
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty: adj_list.append(cached_data['adj'])
            if not cached_data['daily'].empty: daily_list.append(cached_data['daily'])
            if i % 5 == 0: 
                download_progress.progress((i + 1) / total_dates)
        except: continue 
    download_progress.empty()

    if not adj_list or not daily_list:
        st.error("无法获取历史数据。")
        return False
        
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    cols_to_keep = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol']
    valid_cols = [c for c in cols_to_keep if c in daily_list[0].columns]
    daily_raw = pd.concat(daily_list)[valid_cols]
    
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'vol']:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        try:
            latest_adj = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj.droplevel(1).to_dict()
        except: GLOBAL_QFQ_BASE_FACTORS = {}
    
    return True

# ----------------------------------------------------------------------
# 复权数据提取
# ----------------------------------------------------------------------
def get_qfq_data_v4(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty: return pd.DataFrame()
        
    base_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(base_adj) or base_adj < 1e-9: return pd.DataFrame() 

    try:
        daily = GLOBAL_DAILY_RAW.loc[(ts_code, slice(start_date, end_date)), :]
        adj = GLOBAL_ADJ_FACTOR.loc[(ts_code, slice(start_date, end_date)), 'adj_factor']
    except KeyError: return pd.DataFrame()
    
    if daily.empty or adj.empty: return pd.DataFrame()
    
    df = daily.join(adj, how='left').dropna(subset=['adj_factor'])
    
    factor = df['adj_factor'] / base_adj
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns: df[col] = df[col] * factor
    
    df = df.reset_index()
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    return df.set_index('trade_date').sort_index()[['open', 'high', 'low', 'close', 'pre_close', 'vol']]

# ----------------------------------------------------------------------
# 核心指标计算
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data_v4(ts_code, start_date=start_date, end_date=end_date)
    res = {}
    if df.empty or len(df) < 26: return res
         
    close = df['close']
    vol = df['vol']
    
    # 暴力 MACD (8, 17, 5)
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd_val = (diff - dea) * 2
    
    res['macd_val'] = macd_val.iloc[-1]
    
    ma20 = close.rolling(window=20).mean()
    ma5_vol = vol.rolling(window=5).mean()
    
    res['close_current'] = close.iloc[-1]
    res['ma20_current'] = ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else 0
    res['vol_current'] = vol.iloc[-1]
    res['ma5_vol_current'] = ma5_vol.iloc[-1] if not pd.isna(ma5_vol.iloc[-1]) else 0
    
    return res

# ----------------------------------------------------------------------
# 未来收益计算
# ----------------------------------------------------------------------
def get_future_returns(ts_code, selection_date, buy_threshold_pct=1.5):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    hist = get_qfq_data_v4(ts_code, start_date=start_future, end_date=end_future)
    results = {'Return_D1': np.nan, 'Return_D3': np.nan, 'Return_D5': np.nan}

    if hist.empty: return results
    d1_data = hist.iloc[0]
    
    # 实战模拟：拒绝低开
    if d1_data['open'] <= d1_data['pre_close']: return results 
    
    # 实战模拟：盘中必须触及 +1.5% 才能成交
    buy_price = d1_data['open'] * (1 + buy_threshold_pct / 100.0)
    if d1_data['high'] < buy_price: return results 

    # 计算收益
    for n in [1, 3, 5]:
        idx = n - 1
        if len(hist) > idx:
            results[f'Return_D{n}'] = (hist.iloc[idx]['close'] / buy_price - 1) * 100
            
    return results

# ----------------------------------------------------
# 侧边栏设置
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测参数")
    backtest_date_end = st.date_input("结束日期", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("回测天数", value=50, step=1))
    
    st.markdown("---")
    st.header("2. 策略控制")
    # [恢复] 每日持仓数量调整
    TOP_BACKTEST = int(st.number_input("每日持仓数量 (Top K)", value=1, min_value=1, max_value=10, help="建议设为1，只买最强的那只"))
    BUY_THRESHOLD = st.number_input("买入触发涨幅(%)", value=1.5)

    st.markdown("---")
    st.header("3. 选股门槛")
    MIN_PRICE = st.number_input("最低股价", value=20.0, step=1.0) 
    MAX_PRICE = st.number_input("最高股价", value=300.0, step=5.0)
    MIN_CIRC_MV = st.number_input("最低流通市值(亿)", value=30.0, step=5.0) 

    st.markdown("---")
    st.info("⚠️ 已启用大盘风控 (MA20)。")

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ----------------------------------------------------------------------
# V30.25 核心逻辑：全扫描 + 纯粹评分 + Top K截取
# ----------------------------------------------------------------------
def run_backtest_daily(date_str, top_k):
    # 1. 大盘风控
    if not GLOBAL_INDEX_DATA.empty and date_str in GLOBAL_INDEX_DATA.index:
        idx_today = GLOBAL_INDEX_DATA.loc[date_str]
        if idx_today['close'] < idx_today['ma20']:
            return pd.DataFrame(), "🛡️ 大盘破位，系统空仓"
    
    # 2. 获取数据
    daily = safe_get('daily', trade_date=date_str)
    if daily.empty: return pd.DataFrame(), "数据缺失"
    
    # 3. 基础过滤
    pool = daily.copy()
    pool['close'] = pd.to_numeric(pool['close'], errors='coerce')
    d_basic = safe_get('daily_basic', trade_date=date_str, fields='ts_code,circ_mv,turnover_rate')
    if d_basic.empty: return pd.DataFrame(), "基础数据缺失"
    pool = pool.merge(d_basic, on='ts_code', how='inner')
    
    # 3.1 价格/市值/板块过滤
    pool = pool[(pool['close'] >= MIN_PRICE) & (pool['close'] <= MAX_PRICE)]
    pool['circ_mv_billion'] = pool['circ_mv'] / 10000 
    pool = pool[pool['circ_mv_billion'] >= MIN_CIRC_MV]
    pool = pool[~pool['ts_code'].str.startswith(('8', '4', '92'))] 
    
    # 3.2 去一字板
    pool = pool[~((pool['high'] == pool['low']) & (pool['pct_chg'] > 9.0))]

    # 3.3 全扫描 (涨幅>0)
    candidates = pool[pool['pct_chg'] > 0]
    
    if len(candidates) > 400:
        candidates = candidates.sort_values('pct_chg', ascending=False).head(400)
    
    if candidates.empty: return pd.DataFrame(), "无初选股票"

    # 4. 计算 MACD
    records = []
    for row in candidates.itertuples():
        ind = compute_indicators(row.ts_code, date_str)
        
        if ind.get('close_current', 0) <= ind.get('ma20_current', 0): continue
        if ind.get('vol_current', 0) <= ind.get('ma5_vol_current', 0) * 1.2: continue
        if pd.isna(ind.get('macd_val')) or ind.get('macd_val') <= 0: continue
        
        future = get_future_returns(row.ts_code, date_str, buy_threshold_pct=BUY_THRESHOLD)
        
        # 纯评分
        score = (ind['macd_val'] / row.close) * 100000
        
        records.append({
            'ts_code': row.ts_code,
            'Close': row.close,
            'Pct_Chg': row.pct_chg,
            'MACD': ind['macd_val'],
            'Score': score,
            'Return_D1': future['Return_D1'],
            'Return_D3': future['Return_D3'],
            'Return_D5': future['Return_D5']
        })
    
    if not records: return pd.DataFrame(), "无达标股票"
    
    # 5. [恢复] 根据用户设定的 Top K 进行截取
    df_res = pd.DataFrame(records)
    df_res = df_res.sort_values('Score', ascending=False).head(top_k)
    
    return df_res, "Success"

# ---------------------------
# 主程序
# ---------------------------
if st.button(f"🚀 运行 V30.25 回测 (Top {TOP_BACKTEST})"):
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days: st.stop()
    if not get_all_historical_data(trade_days): st.stop()
    
    st.success(f"✅ V30.25 启动 | 每日持仓: Top {TOP_BACKTEST} | 风控: 开启")
    results = []
    bar = st.progress(0)
    status_text = st.empty()
    
    for i, date in enumerate(trade_days):
        status_text.text(f"正在分析: {date} ...")
        try:
            # 传入 TOP_BACKTEST 参数
            df, msg = run_backtest_daily(date, TOP_BACKTEST)
            if not df.empty:
                df['Trade_Date'] = date
                results.append(df)
        except Exception: pass
        bar.progress((i + 1) / len(trade_days))
    
    bar.empty()
    status_text.text("回测完成！")
    
    if not results:
        st.warning("区间内无交易或大盘一直处于避险状态。")
        st.stop()
        
    all_res = pd.concat(results)
    if all_res['Trade_Date'].dtype != 'object': all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
    
    # [恢复] 经典的指标卡样式
    st.header(f"📊 V30.25 回测报告 (Top {TOP_BACKTEST})")
    
    # 统计逻辑：计算所有入选股票的平均表现
    # 如果 Top K = 1，就是第一名的表现
    # 如果 Top K = 3，就是前三名的平均表现
    valid_days = all_res['Trade_Date'].nunique()
    total_trades = len(all_res.dropna(subset=['Return_D1']))
    
    st.markdown(f"**有效交易天数：** {valid_days} 天 | **总成交笔数：** {total_trades} 笔")

    cols = st.columns(3)
    for idx, n in enumerate([1, 3, 5]):
        col = f'Return_D{n}'
        valid = all_res.dropna(subset=[col])
        if not valid.empty:
            avg_ret = valid[col].mean()
            hit_rate = (valid[col] > 0).sum() / len(valid) * 100
            count = len(valid)
        else: avg_ret, hit_rate, count = 0, 0, 0
        with cols[idx]:
            st.metric(f"D+{n} 收益 / 胜率", f"{avg_ret:.2f}% / {hit_rate:.1f}%", help=f"样本数：{count}")

    st.header("📋 每日成交明细")
    st.dataframe(all_res.sort_values('Trade_Date', ascending=False), use_container_width=True)
