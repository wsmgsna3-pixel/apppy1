# -*- coding: utf-8 -*-
"""
选股王 · V30.11.3 稳定回测版
核心修复：
1. **API 限流保护**：增加 request 间隔，解决因 Tushare 频控导致回测卡在 ~40 天(10月23日)的问题。
2. **显示增强**：恢复 D3/D5 收益率在详情表中的显示。
3. **日历算法**：优化交易日获取逻辑，支持超长周期回测。
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
# 全局变量初始化
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V30.11：稳定版", layout="wide")
st.title("选股王 V30.11：稳定回测版（✅ 防限流 & 全显示）")
st.markdown("""
**版本更新说明 (V30.11.3)：**
1. 🐢 **智能限流**：已增加 API 调用间隔，防止回测 100+ 天时因触发 Tushare 频控而中断。
2. 👁️ **全景收益**：详情表现在会完整显示 D1 / D3 / D5 收益率。
""")

# ---------------------------
# 辅助函数
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        # 增加极短的随机延时，减轻服务器压力
        if kwargs.get('is_index'): df = pro.index_daily(**kwargs)
        else: df = func(**kwargs)
        if df is None or df.empty: return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception as e: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    # --- 修复逻辑：强制拉取至少一年的日历，确保不管回测多少天都有数据 ---
    lookback_days = max(num_days * 3, 365) 
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty: return []
    
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    # 再次确保只取 end_date 之前的
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    
    # 返回指定数量的交易日
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
    
    # 扩大缓冲区
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=200)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=30)
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    st.info(f"⏳ 正在预加载 {start_date} 到 {end_date} 的全市场数据 (为了计算指标，需下载更多历史)...")

    adj_factor_data_list = [] 
    daily_data_list = []
    
    # 创建进度条
    progress_text = "数据下载中，请稍候..."
    my_bar = st.progress(0, text=progress_text)
    
    total_steps = len(all_dates)
    
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty: adj_factor_data_list.append(cached_data['adj'])
            if not cached_data['daily'].empty: daily_data_list.append(cached_data['daily'])
            
            # --- 限流保护：每处理 20 天休息一下，防止批量下载被封 ---
            if i % 20 == 0: time.sleep(0.05)
            
            # 更新进度条
            if i % 5 == 0:
                my_bar.progress((i + 1) / total_steps, text=f"正在下载: {date}")
                
        except Exception: continue 
            
    my_bar.empty()

    if not adj_factor_data_list or not daily_data_list: return False
     
    # 合并处理
    adj_factor_data = pd.concat(adj_factor_data_list)
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(0)
    # 去重并建立索引
    GLOBAL_ADJ_FACTOR = adj_factor_data.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw_data = pd.concat(daily_data_list)
    GLOBAL_DAILY_RAW = daily_raw_data.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    # 缓存最新的复权因子基准
    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        # 获取最新一天的所有股票复权因子
        try:
            latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
        except:
            GLOBAL_QFQ_BASE_FACTORS = {}
    
    return True

def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    
    # 快速获取基准因子
    latest_adj_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj_factor): return pd.DataFrame() 

    try:
        # 切片获取个股数据 (利用 MultiIndex 优势)
        daily_df = GLOBAL_DAILY_RAW.loc[ts_code]
        daily_df = daily_df.loc[(daily_df.index >= start_date) & (daily_df.index <= end_date)]
        
        adj_series = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        adj_series = adj_series.loc[(adj_series.index >= start_date) & (adj_series.index <= end_date)]
    except KeyError: return pd.DataFrame()
    
    if daily_df.empty or adj_series.empty: return pd.DataFrame()
    
    # 合并
    df = daily_df.merge(adj_series.rename('adj_factor'), left_index=True, right_index=True, how='left').dropna(subset=['adj_factor'])
    
    # 前复权计算
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df = df.sort_values('trade_date_str').set_index('trade_date_str')
    
    # 覆盖原列
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col + '_qfq']
    
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 

def get_future_prices(ts_code, selection_date, d0_qfq_close, days_ahead=[1, 3, 5]):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=15)).strftime("%Y%m%d") # 往后找15天够了
    
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
    
    # 隐藏参数
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
    
    # 1. 基础数据
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
    
    # 过滤
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS) & (df['circ_mv_billion'] <= MAX_CIRC_MV_BILLIONS)]
    df = df[df['turnover_rate'] >= MIN_TURNOVER]
    df = df[df['amount'] >= MIN_AMOUNT]
    df = df[df['turnover_rate'] <= MAX_TURNOVER_RATE] 

    if len(df) == 0: return pd.DataFrame(), "过滤后无标的"

    # 初筛
    limit_mf = int(FINAL_POOL * 0.5)
    df_mf = df.sort_values('net_mf', ascending=False).head(limit_mf)
    limit_pct = FINAL_POOL - len(df_mf)
    df_pct = df[~df['ts_code'].isin(df_mf['ts_code'])].sort_values('pct_chg', ascending=False).head(limit_pct)
    final_candidates = pd.concat([df_mf, df_pct]).reset_index(drop=True)
    
    # 深度计算
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
        
        # 过滤器
        if pd.isna(d0_ma60) or d0_close < d0_ma60: continue
            
        if pd.notna(d0_high) and pd.notna(d0_close) and d0_close > 0:
            upper_shadow = (d0_high - d0_close) / d0_close * 100
            if upper_shadow > MAX_UPPER_SHADOW: continue 
        
        if pd.notna(d0_high) and pd.notna(d0_low) and pd.notna(d0_close):
            range_len = d0_high - d0_low
            if range_len > 0:
                body_pos = (d0_close - d0_low) / range_len
                if body_pos < MIN_BODY_POS: continue 

        if market_state == 'Weak':
            if pd.isna(d0_ma20) or d0_close < d0_ma20: continue 
            if pd.notna(d0_pos60) and d0_pos60 > 20.0: continue

        # 记录收益
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
                'Return_D1 (%)': future.get('Return_D1', np.nan),
                'Return_D3 (%)': future.get('Return_D3', np.nan),
                'Return_D5 (%)': future.get('Return_D5', np.nan),
            }
            records.append(rec)
            
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), "深度筛选后无标的"
    
    # 评分
    def normalize(s): 
        if s.max() == s.min(): return pd.Series([0.5] * len(s), index=s.index) 
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    
    fdf['s_mf'] = normalize(fdf['net_mf'])
    fdf['s_rsi_safety'] = 1 - normalize(fdf['rsi']) 
    fdf['s_bias_safety'] = 1 - normalize(fdf['bias']) 
    fdf['s_safety'] = (fdf['s_rsi_safety'] * 0.5 + fdf['s_bias_safety'] * 0.5) 

    if market_state == 'Strong':
        fdf['策略'] = 'V30.11 Alpha 强市'
        fdf_strong = fdf[fdf['macd'] > 0].copy()
        if fdf_strong.empty: fdf['Score'] = 0
        else:
            fdf_strong['s_alpha'] = fdf_strong['macd'] * 10000 + fdf_strong['s_mf'] * 50
            fdf_strong['Score'] = fdf_strong['s_alpha'] * 0.8 + fdf_strong['s_safety'] * 0.2
            fdf = fdf_strong.sort_values('Score', ascending=False)
    else:
        fdf['策略'] = 'V30.11 Alpha 弱市'
        fdf['s_macd'] = normalize(fdf['macd'])
        fdf['s_alpha'] = fdf['s_macd'] * 0.6 + fdf['s_mf'] * 0.4
        fdf['Score'] = fdf['s_alpha'] * 0.8 + fdf['s_safety'] * 0.2
        fdf = fdf.sort_values('Score', ascending=False)
        
    return fdf.head(TOP_BACKTEST), None

# ---------------------------
# 主运行块
# ---------------------------
if st.button(f"🚀 运行 V30.11 稳定回测版 ({BACKTEST_DAYS}天)"):
    
    try:
        trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
        st.info(f"📅 计划回测交易日数量: {len(trade_days)} 天")
        if len(trade_days) < BACKTEST_DAYS:
            st.warning("⚠️ 获取的交易日少于预期，可能是因为日历数据更新延迟或假期。")
    except Exception:
        st.error("无法找到 Tushare 交易日数据，请检查 Tushare Token。")
        st.stop()

    if not get_all_historical_data(trade_days): 
        st.error("数据下载失败或缺失，请稍后重试。")
        st.stop()
    
    results = []
    bar = st.progress(0, text="开始回测...")
    
    for i, date in enumerate(trade_days):
        res, err = run_backtest_for_a_day(date, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS)
        if not res.empty:
            res['Trade_Date'] = date
            results.append(res)
        
        # --- 核心修复：防止 Tushare 频控的关键 ---
        # 每次回测完一天，暂停 0.2 秒。这会使回测变慢，但能确保数据不断流。
        time.sleep(0.2) 
        
        bar.progress((i+1)/len(trade_days), text=f"正在分析: {date}")
        
    bar.empty()
    
    if results:
        all_res = pd.concat(results)
        
        st.header("📊 V30.11 回测结果统计")
        cols = st.columns(3)
        for idx, n in enumerate([1, 3, 5]):
            col_name = f'Return_D{n} (%)'
            valid = all_res.dropna(subset=[col_name])
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                cols[idx].metric(f"D+{n} 平均收益/胜率", f"{avg:.2f}% / {win:.1f}%")
        
        st.subheader("📋 详细回测清单 (含 D1/D3/D5)")
        
        # --- 修复显示：明确指定显示的列，包含所有收益率 ---
        display_cols = ['Trade_Date','name','ts_code','Close','Pct_Chg',
                        'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)',
                        'rsi','bias','策略']
        
        # 确保列存在才显示
        final_cols = [c for c in display_cols if c in all_res.columns]
        st.dataframe(all_res[final_cols], use_container_width=True)
        
        # 提供下载
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载完整回测结果 CSV", csv, "backtest_results.csv", "text/csv")
        
    else:
        st.info("回测完成，但没有选出符合条件的股票。")
import pandas as pd

# 1. 读取你的回测结果文件
# 确保文件名和你导出的一致
df = pd.read_csv('backtest_results.csv')

# 2. 数据清洗：去除 D1, D3, D5 任意一个为空的数据，保证样本一致
df_clean = df.dropna(subset=['Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)']).copy()

print(f"参与统计的有效交易笔数: {len(df_clean)} 笔")

# ==========================================
# 场景 A: D1 上涨才买入 (右侧交易)
# ==========================================
# 筛选 D1 收益 > 0 的行
df_buy = df_clean[df_clean['Return_D1 (%)'] > 0].copy()

# 计算“真实持有收益”：因为你是D1涨了才买(假设D1收盘价买入)，所以你的成本是D1的价格
# 真实收益公式 = (1+D5收益) / (1+D1收益) - 1
df_buy['Real_Yield_D3'] = ((1 + df_buy['Return_D3 (%)']/100) / (1 + df_buy['Return_D1 (%)']/100) - 1) * 100
df_buy['Real_Yield_D5'] = ((1 + df_buy['Return_D5 (%)']/100) / (1 + df_buy['Return_D1 (%)']/100) - 1) * 100

# 统计数据
print("\n=== 策略 A: D0选出，D1上涨确认后(收盘)买入 ===")
print(f"触发买入次数: {len(df_buy)} ({len(df_buy)/len(df_clean):.1%} 的选股被执行)")
print(f"D3 (持有2天) 胜率: {(df_buy['Return_D3 (%)'] > 0).mean():.2%}")  # 这里的胜率通常指相对于D0也是赚的
print(f"D3 真实平均收益: {df_buy['Real_Yield_D3'].mean():.2f}%")
print(f"D5 (持有4天) 胜率: {(df_buy['Return_D5 (%)'] > 0).mean():.2%}")
print(f"D5 真实平均收益: {df_buy['Real_Yield_D5'].mean():.2f}%")

# ==========================================
# 场景 B: 趋势延续概率 (条件概率)
# 问题：如果 D1上涨 且 D3上涨，D5 继续上涨的概率是多少？
# ==========================================
# 筛选 D1 > 0 且 D3 > 0 的行 (即前3天趋势完好)
df_trend = df_buy[df_buy['Return_D3 (%)'] > 0].copy()

# 在这部分样本中，计算 D5 > 0 的比例
prob_d5_continuation = (df_trend['Return_D5 (%)'] > 0).mean()

print("\n=== 场景 B: 趋势延续概率 ===")
print(f"出现 [D1涨 + D3涨] 的形态次数: {len(df_trend)}")
print(f"👉 如果 D1涨且D3涨，D5继续上涨的概率: {prob_d5_continuation:.2%}")
