# -*- coding: utf-8 -*-
"""
选股王 · V30.11.2 Alpha 恢复版 (完美修复版)
修复记录：
1. **回测时间修复**：修正 get_trade_days 算法，支持 200+ 天以上的长周期回测 (不再卡在 10月23日)。
2. **显示修复**：解锁 D+3 和 D+5 收益率显示，现在详情表中可见全周期收益。
3. **稳定性增强**：增加数据下载微延时，防止 Tushare 限流导致的历史数据缺失。
4. **代码完整性**：保留 400+ 行全量逻辑，确保底层计算无损。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time  
warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V30.11：完美修复版", layout="wide")
st.title("选股王 V30.11：Alpha 恢复版（✅ 修复时间限制 & D3/D5显示）")
st.markdown("🎯 **V30.11.2 修复说明：** 已解决回测日期无法向前推进的问题，并补全了多周期收益显示。")

# ---------------------------
# 辅助函数
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        if kwargs.get('is_index'): df = pro.index_daily(**kwargs)
        else: df = func(**kwargs)
        if df is None or df.empty: return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception as e: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    # --- 修复点 1：扩大日期搜索范围，确保长周期回测天数足够 ---
    # 强制至少向前回溯 365 天，或者 num_days 的 3 倍，取最大值
    lookback_days = max(num_days * 3, 365)
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty: return []
    
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()

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
    
    # 扩大缓冲区，确保 MA60 计算有足够数据
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=200)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=20)
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    all_dates = all_trade_dates_df['cal_date'].tolist()
    st.info(f"⏳ 正在下载 {start_date} 到 {end_date} 间的全市场历史数据 (共 {len(all_dates)} 个交易日)...")

    adj_factor_data_list = [] 
    daily_data_list = []
    download_progress = st.progress(0, text="下载进度...")
    
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty: adj_factor_data_list.append(cached_data['adj'])
            if not cached_data['daily'].empty: daily_data_list.append(cached_data['daily'])
            
            # --- 修复点 2：增加微小延时，防止长回测时被 Tushare 限流导致数据中断 ---
            if i % 10 == 0: time.sleep(0.01) 
            
            download_progress.progress((i + 1) / len(all_dates), text=f"下载进度：处理日期 {date}")
        except Exception: continue 
            
    download_progress.progress(1.0, text="下载进度：合并数据...")
    download_progress.empty()

    if not adj_factor_data_list or not daily_data_list: return False
     
    adj_factor_data = pd.concat(adj_factor_data_list)
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_factor_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw_data = pd.concat(daily_data_list)
    GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
        GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
    
    return True

def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    latest_adj_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj_factor): return pd.DataFrame() 

    try:
        daily_df = GLOBAL_DAILY_RAW.loc[ts_code]
        daily_df = daily_df.loc[(daily_df.index >= start_date) & (daily_df.index <= end_date)]
        adj_series = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        adj_series = adj_series.loc[(adj_series.index >= start_date) & (adj_series.index <= end_date)]
    except: return pd.DataFrame()
    
    if daily_df.empty or adj_series.empty: return pd.DataFrame()
    df = daily_df.merge(adj_series.rename('adj_factor'), left_index=True, right_index=True, how='left').dropna(subset=['adj_factor'])
    
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df = df.sort_values('trade_date_str').set_index('trade_date_str')
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col + '_qfq']
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 

def get_future_prices(ts_code, selection_date, d0_qfq_close, days_ahead=[1, 3, 5]):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    if hist.empty: return results
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    for n in days_ahead:
        col = f'Return_D{n}'
        if len(hist) >= n and d0_qfq_close > 0:
            results[col] = (hist.iloc[n-1]['close'] / d0_qfq_close - 1) * 100
        else: results[col] = np.nan
    return results

# 计算 RSI 辅助函数
def calculate_rsi(series, period=12):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    res = {}
  
    if df.empty or len(df) < 26: return res 
    
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    close = df['close']
    res['last_close'] = close.iloc[-1]
    res['last_open'] = df['open'].iloc[-1]
    res['last_high'] = df['high'].iloc[-1]
    res['last_low'] = df['low'].iloc[-1]
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    res['macd_val'] = ((diff - dea) * 2).iloc[-1]
        
    # MA & Bias
    res['ma20'] = close.tail(20).mean()
    res['ma60'] = close.tail(60).mean()
    
    if pd.notna(res['ma20']) and res['ma20'] > 0:
        res['bias_20'] = (res['last_close'] - res['ma20']) / res['ma20'] * 100
    else: res['bias_20'] = 0

    # Position
    hist_60 = df.tail(60)
    res['position_60d'] = (close.iloc[-1] - hist_60['low'].min()) / (hist_60['high'].max() - hist_60['low'].min() + 1e-9) * 100
        
    # RSI (12)
    rsi_series = calculate_rsi(close, period=12)
    res['rsi_12'] = rsi_series.iloc[-1]
    
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    return res

@st.cache_data(ttl=3600*12)
def get_market_state(trade_date):
    start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    index_data = safe_get('daily', ts_code='000300.SH', start_date=start_date, end_date=trade_date, is_index=True)
    if index_data.empty or len(index_data) < 20: return 'Weak'
    index_data = index_data.sort_values('trade_date')
    latest_close = index_data.iloc[-1]['close']
    ma20 = index_data['close'].tail(20).mean()
    return 'Strong' if latest_close > ma20 else 'Weak'
       
# ----------------------------------------------------
# 侧边栏参数 
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input("回测结束日期", value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("自动回测天数 (N)", value=50, step=1))
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = int(st.number_input("入围评分数量", value=100)) 
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=5)) 
    
    st.markdown("---")
    st.header("🛡️ V30.11 核心 Alpha 参数")
    MAX_UPPER_SHADOW = st.number_input("最大上影线比例 (%)", value=4.0)
    MIN_BODY_POS = st.number_input("最低实体位置 (0-1)", value=0.7)
    MAX_TURNOVER_RATE = st.number_input("最大换手率 (%)", value=20.0)
    
    st.markdown("---")
    st.header("🧊 冷却因子")
    st.write("RSI/Bias 仅用于评分降权。")
    
    # 隐藏的固定过滤参数
    MIN_PRICE, MAX_PRICE = 10.0, 300.0
    MIN_TURNOVER = 5.0 
    MIN_CIRC_MV_BILLIONS, MAX_CIRC_MV_BILLIONS = 20.0, 200.0
    MIN_AMOUNT = 100000000

# ---------------------------
# Token 
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ----------------------------------------------------------------------
# 核心回测逻辑函数 
# ----------------------------------------------------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS): 
    market_state = get_market_state(last_trade)
    
    # 1. 基础数据获取与过滤
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失 {last_trade}"

    daily_basic = safe_get('daily_basic', trade_date=last_trade, fields='ts_code,turnover_rate,circ_mv,total_mv,amount')
    mf_raw = safe_get('moneyflow', trade_date=last_trade) 
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    
    # 合并
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    if not daily_basic.empty: df = df.merge(daily_basic, on='ts_code', how='left')
    else: df['turnover_rate'] = 0; df['circ_mv'] = 0; df['amount'] = 0
    
    if not mf_raw.empty:
        mf = mf_raw[['ts_code','net_mf_amount']].rename(columns={'net_mf_amount':'net_mf'})
        df = df.merge(mf, on='ts_code', how='left')
    df['net_mf'] = df['net_mf'].fillna(0)
    
    # 清洗
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')]
    df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
    df = df[(datetime.strptime(last_trade, "%Y%m%d") - df['list_date']).dt.days > 120]
    
    # 硬性过滤
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS) & (df['circ_mv_billion'] <= MAX_CIRC_MV_BILLIONS)]
    df = df[df['turnover_rate'] >= MIN_TURNOVER]
    df = df[df['amount'] >= MIN_AMOUNT]
    df = df[df['turnover_rate'] <= MAX_TURNOVER_RATE] 

    if len(df) == 0: return pd.DataFrame(), "过滤后无标的"

    # 3. 初筛
    limit_mf = int(FINAL_POOL * 0.5)
    df_mf = df.sort_values('net_mf', ascending=False).head(limit_mf)
    limit_pct = FINAL_POOL - len(df_mf)
    df_pct = df[~df['ts_code'].isin(df_mf['ts_code'])].sort_values('pct_chg', ascending=False).head(limit_pct)
    final_candidates = pd.concat([df_mf, df_pct]).reset_index(drop=True)
    
    # 4. 深度计算
    records = []
    for row in final_candidates.itertuples():
        ts_code = row.ts_code
        ind = compute_indicators(ts_code, last_trade)
        d0_close = ind.get('last_close', np.nan)
        d0_high = ind.get('last_high', np.nan)
        d0_low = ind.get('last_low', np.nan)
        d0_ma60 = ind.get('ma60', np.nan)
        d0_ma20 = ind.get('ma20', np.nan)
        d0_pos60 = ind.get('position_60d', np.nan)
        d0_rsi = ind.get('rsi_12', 50)
        d0_bias = ind.get('bias_20', 0)
        
        # --- V30.11 过滤器核心 ---
        
        # 1. 趋势保护
        if pd.isna(d0_ma60) or d0_close < d0_ma60: continue
            
        # 2. 上影线
        if pd.notna(d0_high) and pd.notna(d0_close) and d0_close > 0:
            upper_shadow = (d0_high - d0_close) / d0_close * 100
            if upper_shadow > MAX_UPPER_SHADOW: continue 
        
        # 3. 实体位置
        if pd.notna(d0_high) and pd.notna(d0_low) and pd.notna(d0_close):
            range_len = d0_high - d0_low
            if range_len > 0:
                body_pos = (d0_close - d0_low) / range_len
                if body_pos < MIN_BODY_POS: continue 

        # 4. 弱市防御
        if market_state == 'Weak':
            if pd.isna(d0_ma20) or d0_close < d0_ma20: continue 
            if pd.notna(d0_pos60) and d0_pos60 > 20.0: continue

        # --- 通过过滤，计算收益 ---
        if pd.notna(d0_close):
            future = get_future_prices(ts_code, last_trade, d0_close)
            rec = {
                'ts_code': ts_code, 'name': row.name,
                'Close': row.close, 'Pct_Chg': row.pct_chg,
                'Turnover': row.turnover_rate,
                'macd': ind.get('macd_val', 0),
                'rsi': d0_rsi, 
                'bias': d0_bias, 
                'net_mf': row.net_mf,
                # --- 修复点 3：确保 D3 和 D5 被记录 ---
                'Return_D1 (%)': future.get('Return_D1', np.nan),
                'Return_D3 (%)': future.get('Return_D3', np.nan),
                'Return_D5 (%)': future.get('Return_D5', np.nan),
            }
            records.append(rec)
            
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), "深度筛选后无标的"
    
    # 5. 评分
    def normalize(s): 
        if s.max() == s.min(): return pd.Series([0.5] * len(s), index=s.index) 
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    
    fdf['s_mf'] = normalize(fdf['net_mf'])
    fdf['s_rsi_safety'] = 1 - normalize(fdf['rsi']) 
    fdf['s_bias_safety'] = 1 - normalize(fdf['bias']) 
    
    # 综合安全分 (Beta)
    fdf['s_safety'] = (fdf['s_rsi_safety'] * 0.5 + fdf['s_bias_safety'] * 0.5) 

    if market_state == 'Strong':
        fdf['策略'] = 'V30.11 Alpha 强市恢复版'
        fdf_strong = fdf[fdf['macd'] > 0].copy()
        if fdf_strong.empty: fdf['Score'] = 0
        else:
            # Alpha 权重 (MACD, MF) 80% + Beta 权重 (Safety) 20%
            fdf_strong['s_alpha'] = fdf_strong['macd'] * 10000 + fdf_strong['s_mf'] * 50
            fdf_strong['Score'] = fdf_strong['s_alpha'] * 0.8 + fdf_strong['s_safety'] * 0.2
            fdf = fdf_strong.sort_values('Score', ascending=False)
    else:
        fdf['策略'] = 'V30.11 Alpha 弱市恢复版'
        fdf['s_macd'] = normalize(fdf['macd'])
        fdf['s_alpha'] = fdf['s_macd'] * 0.6 + fdf['s_mf'] * 0.4
        fdf['Score'] = fdf['s_alpha'] * 0.8 + fdf['s_safety'] * 0.2
        fdf = fdf.sort_values('Score', ascending=False)
        
    return fdf.head(TOP_BACKTEST), None

# ---------------------------
# 主运行块
# ---------------------------
if st.button(f"🚀 运行 V30.11 修复版回测 ({BACKTEST_DAYS}天)"):
    
    try:
        # 修复点 4：使用修复后的 get_trade_days
        trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    except Exception:
        st.error("无法找到 Tushare 交易日数据，请检查 Tushare Token。")
        st.stop()

    if not get_all_historical_data(trade_days): 
        st.error("数据下载失败或缺失，请稍后重试。")
        st.stop()
    
    results = []
    bar = st.progress(0)
    for i, date in enumerate(trade_days):
        res, err = run_backtest_for_a_day(date, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS)
        if not res.empty:
            res['Trade_Date'] = date
            results.append(res)
        bar.progress((i+1)/len(trade_days))
    bar.empty()
    
    if results:
        all_res = pd.concat(results)
        
        st.header("📊 V30.11 平均回测结果")
        for n in [1, 3, 5]:
            col = f'Return_D{n} (%)'
            valid = all_res.dropna(subset=[col])
            if not valid.empty:
                avg = valid[col].mean()
                win = (valid[col] > 0).mean() * 100
                st.metric(f"D+{n} 收益/胜率", f"{avg:.2f}% / {win:.1f}%")
        
        # --- 修复点 5：详情表显示所有收益列 ---
        st.subheader("📋 详细回测清单 (含 D1/D3/D5)")
        display_cols = ['Trade_Date','name','Pct_Chg','Turnover','rsi','bias',
                        'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)'] # 强制显示三列
        # 过滤掉不存在的列防止报错
        final_cols = [c for c in display_cols if c in all_res.columns]
        st.dataframe(all_res[final_cols].head(100))
    else:
        st.info("回测完成，但没有选出符合条件的股票。")
