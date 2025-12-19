# -*- coding: utf-8 -*-
"""
选股王 · V30.11.7 智能风控全量版
核心承诺：
1. **零简化**：100%保留 V30.11 源代码中的数据预加载、全局缓存、复权及 API 限流逻辑。
2. **风险拦截**：精准嵌入 RSI(12)>80 与 Bias(20)>25% 逻辑（弱市拦截，强市评分惩罚）。
3. **统计修复**：仪表盘 D1/D3/D5 采用 .dropna() 稳健统计。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time  # 引入时间模块用于限流
warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量初始化 (严禁删减)
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V30.11：全量增强版", layout="wide")
st.title("选股王 V30.11：全量增强版（✅ 逻辑零删减 & 智能拦截）")
st.markdown("""
**版本更新说明 (V30.11.7)：**
1. 🛡️ **智能风控**：弱市环境下 RSI>80 或 Bias>25% 强制拦截；强市环境下通过 Score 扣分限制。
2. 🐢 **API 限流保护**：完整保留 0.2s 延迟及批量缓存逻辑，防止长周期回测崩溃。
3. 📊 **仪表盘修复**：优化 D1/D3/D5 统计函数，自动忽略未来日期的空值，确保胜率真实。
""")

# ---------------------------
# 基础 API 函数 (完整保留 V30.11 逻辑)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: 
        return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        if kwargs.get('is_index'):
            df = pro.index_daily(**kwargs)
        else:
            df = func(**kwargs)
        if df is None or df.empty:
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception as e:
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
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
    
    # 获取回测周期前后的数据以计算指标和未来收益
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=200)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=30)
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    st.info(f"⏳ 正在预加载 {start_date} 到 {end_date} 的全市场数据 (V30.11 底层架构)...")

    adj_factor_data_list = [] 
    daily_data_list = []
    
    progress_text = "数据下载中，请稍候..."
    my_bar = st.progress(0, text=progress_text)
    total_steps = len(all_dates)
    
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty:
                adj_factor_data_list.append(cached_data['adj'])
            if not cached_data['daily'].empty:
                daily_data_list.append(cached_data['daily'])
            
            # 严格保留 API 限流保护逻辑
            if i % 20 == 0: time.sleep(0.05)
            if i % 5 == 0:
                my_bar.progress((i + 1) / total_steps, text=f"正在并行下载并缓存: {date}")
        except Exception:
            continue 
            
    my_bar.empty()
    if not adj_factor_data_list or not daily_data_list:
        return False
     
    adj_factor_data = pd.concat(adj_factor_data_list)
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_factor_data.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw_data = pd.concat(daily_data_list)
    GLOBAL_DAILY_RAW = daily_raw_data.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        try:
            latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
        except:
            GLOBAL_QFQ_BASE_FACTORS = {}
            
    return True

# ---------------------------
# 复权计算核心逻辑 (100% 保留 V30.11 函数)
# ---------------------------
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
    except KeyError:
        return pd.DataFrame()
    
    if daily_df.empty or adj_series.empty:
        return pd.DataFrame()
    
    df = daily_df.merge(adj_series.rename('adj_factor'), left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df = df.sort_values('trade_date_str').set_index('trade_date_str')
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col + '_qfq']
        
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
        else:
            results[col] = np.nan
    return results

# ---------------------------
# 技术指标计算 (含 RSI 与 Bias)
# ---------------------------
def calculate_rsi(series, period=12):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
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
    
    # MACD 完整计算
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    
    # 均线与偏离度
    res['ma20'] = close.tail(20).mean()
    res['ma60'] = close.tail(60).mean()
    
    # Bias(20) 精准计算
    if pd.notna(res['ma20']) and res['ma20'] > 0:
        res['bias_20'] = (res['last_close'] - res['ma20']) / res['ma20'] * 100
    else: res['bias_20'] = 0

    # RSI(12) 计算
    rsi_series = calculate_rsi(close, period=12)
    res['rsi_12'] = rsi_series.iloc[-1]
    
    # 60日位置与波动率
    hist_60 = df.tail(60)
    res['position_60d'] = (close.iloc[-1] - hist_60['low'].min()) / (hist_60['high'].max() - hist_60['low'].min() + 1e-9) * 100
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

# ---------------------------
# 核心回测逻辑函数 (480行版核心，精准嵌入)
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS, RSI_LIMIT, BIAS_LIMIT):
    market_state = get_market_state(last_trade)
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失 {last_trade}"

    daily_basic = safe_get('daily_basic', trade_date=last_trade)
    mf_raw = safe_get('moneyflow', trade_date=last_trade) 
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    
    # 数据合并
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    if not daily_basic.empty:
        # 修复 KeyError：动态检查列是否存在
        cols_to_use = ['ts_code', 'turnover_rate', 'circ_mv', 'amount']
        existing_cols = [c for c in cols_to_use if c in daily_basic.columns]
        df = df.merge(daily_basic[existing_cols], on='ts_code', how='left')
    
    # 缺失列补齐
    for col in ['turnover_rate', 'circ_mv', 'amount']:
        if col not in df.columns: df[col] = 0
    
    if not mf_raw.empty:
        mf = mf_raw[['ts_code','net_mf_amount']].rename(columns={'net_mf_amount':'net_mf'})
        df = df.merge(mf, on='ts_code', how='left')
    df['net_mf'] = df['net_mf'].fillna(0)
    
    # 基础清洗流程 (V30.11 全量逻辑)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['circ_mv_billion'] = df['circ_mv'] / 10000
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')]
    
    # 过滤：股价、市值、换手率
    df = df[(df['close'] >= 10.0) & (df['close'] <= 300.0)]
    df = df[(df['circ_mv_billion'] >= 20.0) & (df['circ_mv_billion'] <= 200.0)]
    df = df[df['turnover_rate'] <= MAX_TURNOVER_RATE] 

    if len(df) == 0: return pd.DataFrame(), "过滤后无标的"

    # 根据动能初筛入围池
    candidates = df.sort_values('pct_chg', ascending=False).head(FINAL_POOL)
    
    records = []
    for row in candidates.itertuples():
        ind = compute_indicators(row.ts_code, last_trade)
        if not ind: continue
        
        d0_close = ind['last_close']
        d0_rsi = ind.get('rsi_12', 50)
        d0_bias = ind.get('bias_20', 0)
        
        # --- 嵌入拦截逻辑 ---
        if market_state == 'Weak':
            # 弱市硬拦截
            if d0_rsi > RSI_LIMIT or d0_bias > BIAS_LIMIT: continue
            if d0_close < ind['ma20'] or ind['position_60d'] > 20.0: continue
        
        # 普适硬性指标 (MA60 以上，影线控制)
        if d0_close < ind['ma60']: continue
        
        upper_shadow = (ind['last_high'] - d0_close) / d0_close * 100
        if upper_shadow > MAX_UPPER_SHADOW: continue 
        
        range_len = ind['last_high'] - ind['last_low']
        if range_len > 0:
            body_pos = (d0_close - ind['last_low']) / range_len
            if body_pos < MIN_BODY_POS: continue 

        # 核心：计算未来收益以便仪表盘统计
        future = get_future_prices(row.ts_code, last_trade, d0_close)
        
        records.append({
            'ts_code': row.ts_code, 'name': row.name,
            'Close': row.close, 'Pct_Chg': row.pct_chg,
            'rsi': d0_rsi, 'bias': d0_bias, 'macd': ind['macd_val'], 'net_mf': row.net_mf,
            'Return_D1 (%)': future.get('Return_D1', np.nan),
            'Return_D3 (%)': future.get('Return_D3', np.nan),
            'Return_D5 (%)': future.get('Return_D5', np.nan),
            'market_state': market_state
        })
            
    if not records: return pd.DataFrame(), "深度筛选后无标的"
    fdf = pd.DataFrame(records)
    
    # --- 评分系统 (强市惩罚逻辑) ---
    def final_scoring(r):
        # 基础分 = MACD 动能 + 资金流
        base = r['macd'] * 1000 + (r['net_mf'] / 10000)
        if r['market_state'] == 'Strong':
            # 强市下，如果超买，不拦截但大幅扣分，让排名靠后
            penalty = 0
            if r['rsi'] > RSI_LIMIT: penalty += 500
            if r['bias'] > BIAS_LIMIT: penalty += 500
            return base - penalty
        return base

    fdf['Score'] = fdf.apply(final_scoring, axis=1)
    return fdf.sort_values('Score', ascending=False).head(TOP_BACKTEST), None

# ---------------------------
# 主程序与 UI (V30.11 原汁原味)
# ---------------------------
with st.sidebar:
    st.header("V30.11.7 系统参数")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析交易日数", value=30, step=1)
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=5)
    st.markdown("---")
    RSI_LIMIT = st.number_input("RSI 拦截线", value=80.0)
    BIAS_LIMIT = st.number_input("Bias(20) 拦截线", value=25.0)
    MAX_UPPER_SHADOW = st.number_input("最大上影线 (%)", value=4.0)
    MIN_BODY_POS = st.number_input("最低实体位置", value=0.7)
    MAX_TURNOVER_RATE = st.number_input("最大换手率 (%)", value=20.0)

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🚀 运行 V30.11.7 全量回测引擎"):
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    
    # 执行 V30.11 庞大的预加载逻辑
    if not get_all_historical_data(trade_days):
        st.error("数据加载失败，请检查网络或 Token 额度")
        st.stop()
        
    results = []
    bar = st.progress(0, text="回测引擎启动...")
    
    for i, date in enumerate(trade_days):
        res, err = run_backtest_for_a_day(date, int(TOP_BACKTEST), 100, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS, RSI_LIMIT, BIAS_LIMIT)
        if not res.empty:
            res['Trade_Date'] = date
            results.append(res)
        
        # --- 严格保留 V30.11 核心限流逻辑 ---
        time.sleep(0.2) 
        bar.progress((i+1)/len(trade_days), text=f"正在分析第 {i+1} 天: {date}")
        
    bar.empty()
    
    if results:
        all_res = pd.concat(results)
        
        # --- 仪表盘稳健统计 ---
        st.header("📊 选股王 V30.11.7 统计大屏")
        cols = st.columns(3)
        for idx, n in enumerate([1, 3, 5]):
            col_name = f'Return_D{n} (%)'
            valid = all_res.dropna(subset=[col_name]) # 修复统计：忽略未来空值
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                cols[idx].metric(f"D+{n} 收益/胜率", f"{avg:.2f}% / {win:.1f}%")
            else:
                cols[idx].metric(f"D+{n} 收益/胜率", "计算中...")
        
        st.subheader("📋 回测明细表")
        display_cols = ['Trade_Date','name','ts_code','Close','Pct_Chg',
                        'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)',
                        'rsi','bias']
        st.dataframe(all_res[display_cols].sort_values('Trade_Date', ascending=False), use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载完整回测数据 (CSV)", csv, "backtest_v30_11_7.csv", "text/csv")
