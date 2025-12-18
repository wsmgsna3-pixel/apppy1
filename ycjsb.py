# -*- coding: utf-8 -*-
"""
选股王 · V30.12.2 安全增强完整版
基于 V30.11.3 稳定回测版缝合修改。

核心功能升级：
1. **[span_0](start_span)[span_1](start_span)RSI 拦截器**：在深度筛选环节自动拦截 RSI > 80 的票，防止高位闷杀[span_0](end_span)[span_1](end_span)。
2. **[span_2](start_span)[span_3](start_span)Bias 离群索居保护**：在深度筛选环节自动拦截 Bias > 25 的票，防止均值回归大跌[span_2](end_span)[span_3](end_span)。
3. **[span_4](start_span)API 限流保护**：完整保留原版 0.2s 延时，确保回测长周期不断流[span_4](end_span)。
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
st.set_page_config(page_title="选股王 V30.12：安全版", layout="wide")
st.title("选股王 V30.12：安全增强完整版（✅ 已集成 RSI/Bias 风控）")
st.markdown("""
**版本更新说明 (V30.12.2)：**
1. [span_5](start_span)🛡️ **双重风控**：已在评分逻辑前强制过滤 RSI > 80 和 Bias > 25 的个股[span_5](end_span)。
2. [span_6](start_span)[span_7](start_span)🐢 **稳定至上**：保留原版限流逻辑，解决 API 频控导致的回测中断问题[span_6](end_span)[span_7](end_span)。
3. [span_8](start_span)[span_9](start_span)👁️ **全景收益**：详情表完整保留 D1 / D3 / D5 收益率显示[span_8](end_span)[span_9](end_span)。
""")

# ---------------------------
# 辅助函数 (完整保留 V30.11 逻辑)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    [span_10](start_span)if pro is None: return pd.DataFrame(columns=['ts_code'])[span_10](end_span)
    func = getattr(pro, func_name) 
    try:
        if kwargs.get('is_index'): df = pro.index_daily(**kwargs)
        else: df = func(**kwargs)
        if df is None or df.empty: return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception as e: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    lookback_days = max(num_days * 3, 365) 
    [span_11](start_span)start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")[span_11](end_span)
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
    [span_12](start_span)global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS[span_12](end_span)
    if not trade_days_list: return False
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=200)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=30)
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    all_dates = all_trade_dates_df['cal_date'].tolist()
    st.info(f"⏳ 正在预加载 {start_date} 到 {end_date} 的全市场数据...")
    [span_13](start_span)adj_factor_data_list = [][span_13](end_span)
    daily_data_list = []
    my_bar = st.progress(0, text="数据下载中...")
    total_steps = len(all_dates)
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty: adj_factor_data_list.append(cached_data['adj'])
            [span_14](start_span)if not cached_data['daily'].empty: daily_data_list.append(cached_data['daily'])[span_14](end_span)
            if i % 20 == 0: time.sleep(0.05)
            [span_15](start_span)if i % 5 == 0: my_bar.progress((i + 1) / total_steps, text=f"正在下载: {date}")[span_15](end_span)
        except Exception: continue 
    my_bar.empty()
    if not adj_factor_data_list or not daily_data_list: return False
    adj_factor_data = pd.concat(adj_factor_data_list)
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(0)
    [span_16](start_span)GLOBAL_ADJ_FACTOR = adj_factor_data.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])[span_16](end_span)
    daily_raw_data = pd.concat(daily_data_list)
    GLOBAL_DAILY_RAW = daily_raw_data.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        try:
            latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
        [span_17](start_span)except: GLOBAL_QFQ_BASE_FACTORS = {}[span_17](end_span)
    return True

def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    latest_adj_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj_factor): return pd.DataFrame() 
    try:
        daily_df = GLOBAL_DAILY_RAW.loc[ts_code]
        [span_18](start_span)daily_df = daily_df.loc[(daily_df.index >= start_date) & (daily_df.index <= end_date)][span_18](end_span)
        adj_series = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        adj_series = adj_series.loc[(adj_series.index >= start_date) & (adj_series.index <= end_date)]
    except KeyError: return pd.DataFrame()
    if daily_df.empty or adj_series.empty: return pd.DataFrame()
    df = daily_df.merge(adj_series.rename('adj_factor'), left_index=True, right_index=True, how='left').dropna(subset=['adj_factor'])
    [span_19](start_span)for col in ['open', 'high', 'low', 'close', 'pre_close']:[span_19](end_span)
        if col in df.columns: df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df = df.sort_values('trade_date_str').set_index('trade_date_str')
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col + '_qfq']
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 

def get_future_prices(ts_code, selection_date, d0_qfq_close, days_ahead=[1, 3, 5]):
    [span_20](start_span)d0 = datetime.strptime(selection_date, "%Y%m%d")[span_20](end_span)
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    if hist.empty: return results
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    for n in days_ahead:
        col = f'Return_D{n}'
        [span_21](start_span)if len(hist) >= n and d0_qfq_close > 0: results[col] = (hist.iloc[n-1]['close'] / d0_qfq_close - 1) * 100[span_21](end_span)
        else: results[col] = np.nan
    return results

def calculate_rsi(series, period=12):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    [span_22](start_span)start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")[span_22](end_span)
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    res = {}
    if df.empty or len(df) < 26: return res 
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    close = df['close']
    res['last_close'] = close.iloc[-1]
    res['last_open'] = df['open'].iloc[-1]
    res['last_high'] = df['high'].iloc[-1]
    res['last_low'] = df['low'].iloc[-1]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    [span_23](start_span)diff = ema12 - ema26[span_23](end_span)
    dea = diff.ewm(span=9, adjust=False).mean()
    res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    res['ma20'] = close.tail(20).mean()
    res['ma60'] = close.tail(60).mean()
    if pd.notna(res['ma20']) and res['ma20'] > 0: res['bias_20'] = (res['last_close'] - res['ma20']) / res['ma20'] * 100
    else: res['bias_20'] = 0
    hist_60 = df.tail(60)
    [span_24](start_span)res['position_60d'] = (close.iloc[-1] - hist_60['low'].min()) / (hist_60['high'].max() - hist_60['low'].min() + 1e-9) * 100[span_24](end_span)
    rsi_series = calculate_rsi(close, period=12)
    res['rsi_12'] = rsi_series.iloc[-1]
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    return res

@st.cache_data(ttl=3600*12)
def get_market_state(trade_date):
    start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    index_data = safe_get('daily', ts_code='000300.SH', start_date=start_date, end_date=trade_date, is_index=True)
    if index_data.empty or len(index_data) < 20: return 'Weak'
    [span_25](start_span)index_data = index_data.sort_values('trade_date')[span_25](end_span)
    latest_close = index_data.iloc[-1]['close']
    ma20 = index_data['close'].tail(20).mean()
    return 'Strong' if latest_close > ma20 else 'Weak'

# ---------------------------
# 侧边栏及主逻辑输入
# ---------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input("回测结束日期", value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("自动回测天数 (N)", value=50, step=1))
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = int(st.number_input("入围评分数量", value=100)) 
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=5)) 
    st.markdown("---")
    [span_26](start_span)st.header("🛡️ V30.12 核心风控参数")[span_26](end_span)
    MAX_UPPER_SHADOW = st.number_input("最大上影线比例 (%)", value=4.0)
    MIN_BODY_POS = st.number_input("最低实体位置 (0-1)", value=0.7)
    MAX_TURNOVER_RATE = st.number_input("最大换手率 (%)", value=20.0)
    # 增加风控阈值
    RSI_LIMIT = st.number_input("RSI 拦截上限", value=80.0)
    BIAS_LIMIT = st.number_input("Bias 拦截上限", value=25.0)
    
    MIN_PRICE, MAX_PRICE = 10.0, 300.0
    MIN_TURNOVER = 5.0 
    MIN_CIRC_MV_BILLIONS, MAX_CIRC_MV_BILLIONS = 20.0, 200.0
    MIN_AMOUNT = 100000000

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ----------------------------------------------------------------------
# 核心选股与评分函数 (V30.12 核心缝合点)
# ----------------------------------------------------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS): 
    [span_27](start_span)market_state = get_market_state(last_trade)[span_27](end_span)
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失 {last_trade}"
    daily_basic = safe_get('daily_basic', trade_date=last_trade, fields='ts_code,turnover_rate,circ_mv,total_mv,amount')
    mf_raw = safe_get('moneyflow', trade_date=last_trade) 
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    if not daily_basic.empty: df = df.merge(daily_basic, on='ts_code', how='left')
    [span_28](start_span)else: df['turnover_rate'] = 0; df['circ_mv'] = 0; df['amount'] = 0[span_28](end_span)
    if not mf_raw.empty:
        mf = mf_raw[['ts_code','net_mf_amount']].rename(columns={'net_mf_amount':'net_mf'})
        df = df.merge(mf, on='ts_code', how='left')
    df['net_mf'] = df['net_mf'].fillna(0)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000
    [span_29](start_span)df = df[~df['name'].str.contains('ST|退', na=False)][span_29](end_span)
    df = df[~df['ts_code'].str.startswith('92')]
    df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
    df = df[(datetime.strptime(last_trade, "%Y%m%d") - df['list_date']).dt.days > 120]
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS) & (df['circ_mv_billion'] <= MAX_CIRC_MV_BILLIONS)]
    df = df[df['turnover_rate'] >= MIN_TURNOVER]
    df = df[df['amount'] >= MIN_AMOUNT]
    df = df[df['turnover_rate'] <= MAX_TURNOVER_RATE] 

    if len(df) == 0: return pd.DataFrame(), "过滤后无标的"

    [span_30](start_span)limit_mf = int(FINAL_POOL * 0.5)[span_30](end_span)
    df_mf = df.sort_values('net_mf', ascending=False).head(limit_mf)
    limit_pct = FINAL_POOL - len(df_mf)
    df_pct = df[~df['ts_code'].isin(df_mf['ts_code'])].sort_values('pct_chg', ascending=False).head(limit_pct)
    final_candidates = pd.concat([df_mf, df_pct]).reset_index(drop=True)
    
    records = []
    for row in final_candidates.itertuples():
        ts_code = row.ts_code
        ind = compute_indicators(ts_code, last_trade)
        d0_close = ind.get('last_close', np.nan)
        [span_31](start_span)d0_high = ind.get('last_high', np.nan)[span_31](end_span)
        d0_low = ind.get('last_low', np.nan)
        d0_ma60 = ind.get('ma60', np.nan)
        d0_ma20 = ind.get('ma20', np.nan)
        d0_pos60 = ind.get('position_60d', np.nan)
        d0_rsi = ind.get('rsi_12', 50)
        d0_bias = ind.get('bias_20', 0)
        
        # --- V30.12 硬拦截逻辑 ---
        [span_32](start_span)if d0_rsi > RSI_LIMIT: continue # RSI 动能透支拦截[span_32](end_span)
        [span_33](start_span)if d0_bias > BIAS_LIMIT: continue # Bias 乖离回归拦截[span_33](end_span)
        
        [span_34](start_span)if pd.isna(d0_ma60) or d0_close < d0_ma60: continue[span_34](end_span)
        if pd.notna(d0_high) and pd.notna(d0_close) and d0_close > 0:
            upper_shadow = (d0_high - d0_close) / d0_close * 100
            if upper_shadow > MAX_UPPER_SHADOW: continue 
        if pd.notna(d0_high) and pd.notna(d0_low) and pd.notna(d0_close):
            [span_35](start_span)range_len = d0_high - d0_low[span_35](end_span)
            if range_len > 0:
                body_pos = (d0_close - d0_low) / range_len
                if body_pos < MIN_BODY_POS: continue 
        if market_state == 'Weak':
            if pd.isna(d0_ma20) or d0_close < d0_ma20: continue 
            [span_36](start_span)if pd.notna(d0_pos60) and d0_pos60 > 20.0: continue[span_36](end_span)

        if pd.notna(d0_close):
            future = get_future_prices(ts_code, last_trade, d0_close)
            rec = {
                'ts_code': ts_code, 'name': row.name,
                'Close': row.close, 'Pct_Chg': row.pct_chg,
                [span_37](start_span)'Turnover': row.turnover_rate,[span_37](end_span)
                'macd': ind.get('macd_val', 0),
                'rsi': d0_rsi, 'bias': d0_bias, 'net_mf': row.net_mf,
                [span_38](start_span)'Return_D1 (%)': future.get('Return_D1', np.nan),[span_38](end_span)
                'Return_D3 (%)': future.get('Return_D3', np.nan),
                'Return_D5 (%)': future.get('Return_D5', np.nan),
            }
            records.append(rec)
            
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), "深度筛选后无标的"
    
    [span_39](start_span)def normalize(s):[span_39](end_span)
        if s.max() == s.min(): return pd.Series([0.5] * len(s), index=s.index) 
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    
    fdf['s_mf'] = normalize(fdf['net_mf'])
    # RSI 在 60-75 区间奖励加分
    fdf['s_rsi_safety'] = fdf['rsi'].apply(lambda x: 1.2 if 60 <= x <= 75 else 0.8) 
    fdf['s_bias_safety'] = 1 - normalize(fdf['bias']) 
    fdf['s_safety'] = (fdf['s_rsi_safety'] * 0.5 + fdf['s_bias_safety'] * 0.5) 

    if market_state == 'Strong':
        [span_40](start_span)fdf['策略'] = 'V30.12 Alpha 强市'[span_40](end_span)
        fdf_strong = fdf[fdf['macd'] > 0].copy()
        if fdf_strong.empty: fdf['Score'] = 0
        else:
            fdf_strong['s_alpha'] = fdf_strong['macd'] * 10000 + fdf_strong['s_mf'] * 50
            fdf_strong['Score'] = fdf_strong['s_alpha'] * 0.8 + fdf_strong['s_safety'] * 0.2
            fdf = fdf_strong.sort_values('Score', ascending=False)
    else:
        [span_41](start_span)fdf['策略'] = 'V30.12 Alpha 弱市'[span_41](end_span)
        fdf['s_macd'] = normalize(fdf['macd'])
        fdf['s_alpha'] = fdf['s_macd'] * 0.6 + fdf['s_mf'] * 0.4
        fdf['Score'] = fdf['s_alpha'] * 0.8 + fdf['s_safety'] * 0.2
        fdf = fdf.sort_values('Score', ascending=False)
        
    return fdf.head(TOP_BACKTEST), None

# ---------------------------
# 主运行块 (完整保留限流保护)
# ---------------------------
if st.button(f"🚀 运行 V30.12 增强版回测 ({BACKTEST_DAYS}天)"):
    try:
        trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
        [span_42](start_span)st.info(f"📅 计划回测交易日数量: {len(trade_days)} 天")[span_42](end_span)
    except Exception: st.stop()

    if not get_all_historical_data(trade_days): st.stop()
    
    results = []
    bar = st.progress(0, text="开始分析...")
    
    [span_43](start_span)for i, date in enumerate(trade_days):[span_43](end_span)
        res, err = run_backtest_for_a_day(date, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS)
        if not res.empty:
            res['Trade_Date'] = date
            results.append(res)
        
        # --- 核心限流保护 ---
        [span_44](start_span)time.sleep(0.2)[span_44](end_span)
        bar.progress((i+1)/len(trade_days), text=f"正在分析: {date}")
        
    bar.empty()
    if results:
        all_res = pd.concat(results)
        st.header("📊 V30.12 回测统计")
        cols = st.columns(3)
        for idx, n in enumerate([1, 3, 5]):
            [span_45](start_span)col_name = f'Return_D{n} (%)'[span_45](end_span)
            valid = all_res.dropna(subset=[col_name])
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                cols[idx].metric(f"D+{n} 收益/胜率", f"{avg:.2f}% / {win:.1f}%")
        
        [span_46](start_span)st.subheader("📋 选股清单 (含 RSI/Bias)")[span_46](end_span)
        display_cols = ['Trade_Date','name','ts_code','Close','Pct_Chg',
                        'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)',
                        'rsi','bias','策略','Score']
        [span_47](start_span)final_cols = [c for c in display_cols if c in all_res.columns][span_47](end_span)
        st.dataframe(all_res[final_cols], use_container_width=True)
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载完整回测结果 CSV", csv, "backtest_results.csv", "text/csv")
