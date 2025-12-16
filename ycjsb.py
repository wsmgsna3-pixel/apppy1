# -*- coding: utf-8 -*-
"""
选股王 · V30.7 修正版：拒绝分歧与拥挤 (Alpha 复合框架)
V30.7.0 更新：
1. **回滚**：删除了 V30.6 导致的性能下降的“高位”和“高涨幅”硬性过滤。
2. **新增防御 1**：上影线过滤 (Upper Shadow)。排除上影线 > 2.0% 的股票（避免拉高出货）。
3. **新增防御 2**：高换手过滤 (Max Turnover)。排除换手率 > 20% 的股票（避免筹码松动）。
4. 保留 V30.4/V30.5 的所有优秀机制（资金流、MACD评分、趋势过滤）。
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
st.set_page_config(page_title="选股王 V30.7：拒绝分歧修正版", layout="wide")
st.title("选股王 V30.7：拒绝分歧修正版（🛡️ 排除长上影线 & 极高换手）")
st.markdown("🎯 **V30.7 策略核心：** 承认 V30.6 的失败，**取消对高位/高涨幅的限制**（找回爆发力）。转而通过 **排除长上影线（>2%）** 和 **排除极高换手（>20%）** 来规避真正的顶部出货风险。")


# ---------------------------
# 辅助函数 (API调用和数据获取)
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
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 2)).strftime("%Y%m%d")
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
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=20)
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    all_dates = all_trade_dates_df['cal_date'].tolist()
    st.info(f"⏳ 正在下载 {start_date} 到 {end_date} 间的全市场历史数据...")

    adj_factor_data_list = []
    daily_data_list = []
    download_progress = st.progress(0, text="下载进度...")
    
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty: adj_factor_data_list.append(cached_data['adj'])
            if not cached_data['daily'].empty: daily_data_list.append(cached_data['daily'])
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

@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    res = {}
    if df.empty or len(df) < 3: return res
    
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    close = df['close']
    res['last_close'] = close.iloc[-1]
    res['last_open'] = df['open'].iloc[-1]
    res['last_high'] = df['high'].iloc[-1]
    
    # MACD
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    else: res['macd_val'] = np.nan
        
    # MA / Volatility / Position
    if len(close) >= 20: res['ma20'] = close.tail(20).mean()
    else: res['ma20'] = np.nan
    
    if len(close) >= 60:
        res['ma60'] = close.tail(60).mean()
        hist_60 = df.tail(60)
        res['position_60d'] = (close.iloc[-1] - hist_60['low'].min()) / (hist_60['high'].max() - hist_60['low'].min() + 1e-9) * 100
    else: 
        res['ma60'] = np.nan
        res['position_60d'] = np.nan
        
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
    st.header("🛡️ V30.7 风险过滤器 (修正版)")
    MAX_UPPER_SHADOW = st.number_input("最大上影线比例 (%)", value=2.0, step=0.1, help="排除上影线长度超过此值的股票。")
    MAX_TURNOVER_RATE = st.number_input("最大换手率 (%)", value=20.0, step=1.0, help="排除换手率过高（筹码松动）的股票。")
    
    # 隐藏的固定过滤参数 (保持 V30.5 标准)
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
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE): 
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
    
    # 基础清洗
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')]
    df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
    df = df[(datetime.strptime(last_trade, "%Y%m%d") - df['list_date']).dt.days > 120]
    
    # 硬性过滤 (V30.5 标准)
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS) & (df['circ_mv_billion'] <= MAX_CIRC_MV_BILLIONS)]
    df = df[df['turnover_rate'] >= MIN_TURNOVER]
    df = df[df['amount'] >= MIN_AMOUNT]

    # V30.7 新增：换手率天花板过滤 (避免拥挤)
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
        # 指标计算
        ind = compute_indicators(ts_code, last_trade)
        d0_close = ind.get('last_close', np.nan)
        d0_open = ind.get('last_open', np.nan)
        d0_high = ind.get('last_high', np.nan)
        d0_ma60 = ind.get('ma60', np.nan)
        d0_ma20 = ind.get('ma20', np.nan)
        d0_pos60 = ind.get('position_60d', np.nan)
        
        # --- V30.7 过滤器核心 ---
        
        # 1. 趋势保护 (MA60)
        if pd.isna(d0_ma60) or d0_close < d0_ma60: continue
            
        # 2. 上影线过滤 (V30.7 新增)
        # 逻辑：(最高价 - 收盘价) / 收盘价 > 2% -> 视为抛压过大，排除
        if pd.notna(d0_high) and pd.notna(d0_close) and d0_close > 0:
            upper_shadow = (d0_high - d0_close) / d0_close * 100
            if upper_shadow > MAX_UPPER_SHADOW:
                continue 
        
        # 3. 弱市防御 (保持 V30.4 逻辑)
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
                'volatility': ind.get('volatility', 0),
                'net_mf': row.net_mf,
                'position_60d': d0_pos60,
                'Return_D1 (%)': future.get('Return_D1', np.nan),
                'Return_D3 (%)': future.get('Return_D3', np.nan),
                'Return_D5 (%)': future.get('Return_D5', np.nan),
            }
            records.append(rec)
            
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), "深度筛选后无标的"
    
    # 5. 评分
    def normalize(s): 
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    
    fdf['s_mf'] = normalize(fdf['net_mf'])
    fdf['s_vol'] = normalize(fdf['volatility'])
    
    if market_state == 'Strong':
        fdf['策略'] = 'V30.7 强势MACD修正'
        # 允许高MACD，但通过上影线和换手率已过滤掉风险
        fdf_strong = fdf[fdf['macd'] > 0].copy()
        if fdf_strong.empty: fdf['Score'] = 0
        else:
            fdf_strong['Score'] = fdf_strong['macd'] * 10000 + \
                                  (fdf_strong['s_vol'].rsub(1) * 0.3 + fdf_strong['s_mf'] * 0.7)
            fdf = fdf_strong.sort_values('Score', ascending=False)
    else:
        fdf['策略'] = 'V30.7 弱势反弹修正'
        fdf['s_macd'] = normalize(fdf['macd'])
        fdf['Score'] = fdf['s_vol'].rsub(1) * 0.45 + fdf['s_macd'] * 0.45 + fdf['s_mf'] * 0.1
        fdf = fdf.sort_values('Score', ascending=False)
        
    return fdf.head(TOP_BACKTEST), None

# ---------------------------
# 主运行块
# ---------------------------
if st.button(f"🚀 运行 V30.7 修正版回测 ({BACKTEST_DAYS}天)"):
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not get_all_historical_data(trade_days): st.stop()
    
    results = []
    bar = st.progress(0)
    for i, date in enumerate(trade_days):
        res, err = run_backtest_for_a_day(date, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE)
        if not res.empty:
            res['Trade_Date'] = date
            results.append(res)
        bar.progress((i+1)/len(trade_days))
    bar.empty()
    
    if results:
        all_res = pd.concat(results)
        st.header("📊 V30.7 平均回测结果")
        for n in [1, 3, 5]:
            col = f'Return_D{n} (%)'
            valid = all_res.dropna(subset=[col])
            if not valid.empty:
                avg = valid[col].mean()
                win = (valid[col] > 0).mean() * 100
                st.metric(f"D+{n} 收益/胜率", f"{avg:.2f}% / {win:.1f}%")
                
        st.dataframe(all_res[['Trade_Date','name','Pct_Chg','Turnover','Return_D1 (%)','Return_D3 (%)']].head(100))
