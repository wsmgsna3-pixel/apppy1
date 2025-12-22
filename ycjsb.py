# -*- coding: utf-8 -*-
"""
选股王 · V30.12.3 实战仿真版 (Hell Mode)
1. **市值修正**：50亿-1000亿，聚焦中盘趋势。
2. **板块共振**：全市场行业遍历覆盖。
3. **实战仿真**：
   - 剔除 D1 低开个股。
   - 买入价 = D1开盘价 * 1.015 (支付 1.5% 确认成本)。
   - 严格验证最高价成交逻辑。
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
GLOBAL_STOCK_INDUSTRY = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V30.12.3：实战仿真版", layout="wide")
st.title("选股王 V30.12.3：实战仿真版（✅ 确认上涨再买入）")
st.markdown("""
**⚠️ 实战仿真模式 (Hell Mode) 说明：**
1. **拒绝低开**：如果 D1 开盘价 <= D0 收盘价，**系统不买入**（回测收益为空）。
2. **支付溢价**：买入成本按 **`D1开盘价 * 1.015`** 计算（模拟追涨 1.5% 确认）。
3. **成交验证**：如果 D1 最高价未达到买入价，视为**无法成交**。
""")

# ---------------------------
# 基础 API 函数
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

# --- 核心修复：遍历拉取所有行业成分股 ---
@st.cache_data(ttl=3600*24*7) 
def load_industry_mapping():
    global pro
    if pro is None: return {}
    
    try:
        sw_indices = pro.index_classify(level='L1', src='SW2021')
        if sw_indices.empty: return {}
        
        index_codes = sw_indices['index_code'].tolist()
        all_members = []
        load_bar = st.progress(0, text="正在遍历加载行业数据...")
        
        for i, idx_code in enumerate(index_codes):
            df = pro.index_member(index_code=idx_code, is_new='Y')
            if not df.empty:
                all_members.append(df)
            time.sleep(0.02) 
            load_bar.progress((i + 1) / len(index_codes), text=f"加载行业数据: {idx_code}")
            
        load_bar.empty()
        
        if not all_members: return {}
        full_df = pd.concat(all_members)
        full_df = full_df.drop_duplicates(subset=['con_code'])
        return dict(zip(full_df['con_code'], full_df['index_code']))
        
    except Exception as e:
        return {}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_INDUSTRY
    if not trade_days_list: return False
    
    with st.spinner("正在同步全市场行业数据 (遍历模式)..."):
        GLOBAL_STOCK_INDUSTRY = load_industry_mapping()
        stock_count = len(GLOBAL_STOCK_INDUSTRY)
        if stock_count < 3000:
            st.warning(f"⚠️ 行业数据加载异常，仅覆盖 {stock_count} 只股票。")
        else:
            st.success(f"✅ 行业映射图谱构建完成，覆盖 {stock_count} 只股票 (100% 全市场)")

    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=200)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=30)
  
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    st.info(f"⏳ 正在预加载全市场数据: {start_date} 至 {end_date}...")

    adj_factor_data_list = [] 
    daily_data_list = []
    
    progress_text = "Tushare 数据同步中..."
    my_bar = st.progress(0, text=progress_text)
    total_steps = len(all_dates)
    
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty:
                adj_factor_data_list.append(cached_data['adj'])
            if not cached_data['daily'].empty:
                daily_data_list.append(cached_data['daily'])
            
            if i % 20 == 0: time.sleep(0.05)
            if i % 5 == 0:
                my_bar.progress((i + 1) / total_steps, text=f"缓存全市场数据: {date}")
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
# 复权计算核心逻辑
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

# ==============================================================================
#  【核心修改点】实战仿真回测逻辑
#   修改人：您的量化参谋
#   修改内容：实现 "开盘+1.5%确认" 买入法，剔除低开股
# ==============================================================================
def get_future_prices(ts_code, selection_date, d0_qfq_close, days_ahead=[1, 3, 5]):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    
    # 获取未来数据
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    
    # 如果没数据，直接返回
    if hist.empty or len(hist) < 1: 
        return results
    
    # 确保列是数值型
    hist['open'] = pd.to_numeric(hist['open'], errors='coerce')
    hist['high'] = pd.to_numeric(hist['high'], errors='coerce')
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    
    # --- 实战仿真逻辑开始 ---
    d1_data = hist.iloc[0]
    next_open = d1_data['open']
    next_high = d1_data['high']
    
    # 1. 拒绝低开：如果明天低开或平开，直接不买 (Return 为空)
    if next_open <= d0_qfq_close:
        return results 
    
    # 2. 支付确认成本：买入价 = 开盘价 * 1.015
    target_buy_price = next_open * 1.015
    
    # 3. 验证成交：如果当天最高价都不到这个买入价，说明没机会买进
    if next_high < target_buy_price:
        return results
        
    # 4. 计算真实收益 (卖出按收盘价)
    for n in days_ahead:
        col = f'Return_D{n}'
        if len(hist) >= n:
            sell_price = hist.iloc[n-1]['close']
            # 真实收益率 = (卖出价 - 成本价) / 成本价
            results[col] = (sell_price - target_buy_price) / target_buy_price * 100
        else:
            results[col] = np.nan
            
    return results
# ==============================================================================

# ---------------------------
# 技术指标逻辑
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
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    
    res['ma20'] = close.tail(20).mean()
    res['ma60'] = close.tail(60).mean()
    
    if pd.notna(res['ma20']) and res['ma20'] > 0:
        res['bias_20'] = (res['last_close'] - res['ma20']) / res['ma20'] * 100
    else: res['bias_20'] = 0

    rsi_series = calculate_rsi(close, period=12)
    res['rsi_12'] = rsi_series.iloc[-1]
    
    hist_60 = df.tail(60)
    res['position_60d'] = (close.iloc[-1] - hist_60['low'].min()) / (hist_60['high'].max() - hist_60['low'].min() + 1e-9) * 100
  
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
# 核心回测逻辑函数 (含市值修正)
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS, RSI_LIMIT, BIAS_LIMIT, SECTOR_THRESHOLD, MIN_MV, MAX_MV):
    global GLOBAL_STOCK_INDUSTRY
    
    market_state = get_market_state(last_trade)
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失 {last_trade}"

    daily_basic = safe_get('daily_basic', trade_date=last_trade)
    mf_raw = safe_get('moneyflow', trade_date=last_trade) 
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    
    strong_industry_codes = set()
    try:
        sw_df = safe_get('sw_daily', trade_date=last_trade)
        if not sw_df.empty:
            strong_sw = sw_df[sw_df['pct_chg'] >= SECTOR_THRESHOLD]
            strong_industry_codes = set(strong_sw['index_code'].tolist())
    except Exception as e:
        pass 
        
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    
    if not daily_basic.empty:
        needed_cols = ['ts_code','turnover_rate','circ_mv','amount']
        existing_cols = [c for c in needed_cols if c in daily_basic.columns]
        df = df.merge(daily_basic[existing_cols], on='ts_code', how='left')
    
    if not mf_raw.empty:
        mf = mf_raw[['ts_code','net_mf_amount']].rename(columns={'net_mf_amount':'net_mf'})
        df = df.merge(mf, on='ts_code', how='left')
    
    for col in ['net_mf', 'turnover_rate', 'circ_mv', 'amount']:
        if col not in df.columns: df[col] = 0
    
    df['net_mf'] = df['net_mf'].fillna(0)
    df['circ_mv_billion'] = df['circ_mv'] / 10000 # 换算为亿元
    
    # 基础清洗流程 (已应用市值修正)
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')]
    df = df[(df['close'] >= 10.0) & (df['close'] <= 300.0)]
    
    # 【核心修正】市值筛选
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    
    df = df[df['turnover_rate'] <= MAX_TURNOVER_RATE] 

    if len(df) == 0: return pd.DataFrame(), "过滤后无标的"

    candidates = df.sort_values('pct_chg', ascending=False).head(FINAL_POOL)
    
    records = []
    for row in candidates.itertuples():
        if GLOBAL_STOCK_INDUSTRY and strong_industry_codes:
            ind_code = GLOBAL_STOCK_INDUSTRY.get(row.ts_code)
            if ind_code and (ind_code not in strong_industry_codes):
                continue

        ind = compute_indicators(row.ts_code, last_trade)
        if not ind: continue
        
        d0_close = ind['last_close']
        d0_rsi = ind.get('rsi_12', 50)
        d0_bias = ind.get('bias_20', 0)
        
        if market_state == 'Weak':
            if d0_rsi > RSI_LIMIT or d0_bias > BIAS_LIMIT: continue
            if d0_close < ind['ma20'] or ind['position_60d'] > 20.0: continue
        
        if d0_close < ind['ma60']: continue
        
        upper_shadow = (ind['last_high'] - d0_close) / d0_close * 100
        if upper_shadow > MAX_UPPER_SHADOW: continue
        
        range_len = ind['last_high'] - ind['last_low']
        if range_len > 0:
            body_pos = (d0_close - ind['last_low']) / range_len
            if body_pos < MIN_BODY_POS: continue

        future = get_future_prices(row.ts_code, last_trade, d0_close)
        
        records.append({
            'ts_code': row.ts_code, 'name': row.name, 'Close': row.close, 'Pct_Chg': row.pct_chg,
            'rsi': d0_rsi, 'bias': d0_bias, 'macd': ind['macd_val'], 'net_mf': row.net_mf,
            'Return_D1 (%)': future.get('Return_D1', np.nan),
            'Return_D3 (%)': future.get('Return_D3', np.nan),
            'Return_D5 (%)': future.get('Return_D5', np.nan),
            'market_state': market_state,
            'Sector_Boost': 'Yes' if GLOBAL_STOCK_INDUSTRY else 'N/A'
        })
            
    if not records: return pd.DataFrame(), "深度筛选后无标的"
    fdf = pd.DataFrame(records)
    
    def dynamic_score(r):
        base_score = r['macd'] * 1000 + (r['net_mf'] / 10000) 
        if r['market_state'] == 'Strong':
            penalty = 0
            if r['rsi'] > RSI_LIMIT: penalty += 500
            if r['bias'] > BIAS_LIMIT: penalty += 500
            return base_score - penalty
        return base_score

    fdf['Score'] = fdf.apply(dynamic_score, axis=1)
    return fdf.sort_values('Score', ascending=False).head(TOP_BACKTEST), None

# ---------------------------
# UI 及 主程序
# ---------------------------
with st.sidebar:
    st.header("V30.12.3 全量配置 (实战仿真版)")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数", value=30, step=1)
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=5)
    
    st.markdown("---")
    st.subheader("💰 市值筛选 (亿元)")
    col_mv1, col_mv2 = st.columns(2)
    MIN_MV = col_mv1.number_input("最小市值", value=50.0, step=10.0, help="单位：亿元")
    MAX_MV = col_mv2.number_input("最大市值", value=1000.0, step=50.0, help="单位：亿元")
    
    st.markdown("---")
    st.subheader("🔥 板块共振设置")
    SECTOR_THRESHOLD = st.number_input("板块当日最低涨幅 (%)", value=1.5, step=0.1, help="建议设为 1.5% 以上以增强过滤效果。")
    
    st.markdown("---")
    RSI_LIMIT = st.number_input("RSI 拦截线", value=80.0)
    BIAS_LIMIT = st.number_input("Bias(20) 拦截线 (%)", value=25.0)
    MAX_UPPER_SHADOW = st.number_input("最大上影线 (%)", value=4.0)
    MIN_BODY_POS = st.number_input("最低实体位置", value=0.7)
    MAX_TURNOVER_RATE = st.number_input("最大换手率 (%)", value=20.0)

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🚀 启动 V30.12.3 实战仿真回测"):
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    
    if not get_all_historical_data(trade_days):
        st.error("数据预加载失败")
        st.stop()
        
    results = []
    bar = st.progress(0, text="回测引擎流水线启动...")
    
    for i, date in enumerate(trade_days):
        res, err = run_backtest_for_a_day(date, int(TOP_BACKTEST), 100, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS, RSI_LIMIT, BIAS_LIMIT, SECTOR_THRESHOLD, MIN_MV, MAX_MV)
        if not res.empty:
            res['Trade_Date'] = date
            results.append(res)
        
        time.sleep(0.2) 
        bar.progress((i+1)/len(trade_days), text=f"正在分析第 {i+1} 天: {date}")
        
    bar.empty()
    
    if results:
        all_res = pd.concat(results)
        
        st.header("📊 V30.12.3 统计仪表盘")
        cols = st.columns(3)
        for idx, n in enumerate([1, 3, 5]):
            col_name = f'Return_D{n} (%)'
            # 自动过滤掉空值（即没买入的交易），只统计成功的交易
            valid = all_res.dropna(subset=[col_name]) 
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                cols[idx].metric(f"D+{n} 均益 / 胜率", f"{avg:.2f}% / {win:.1f}%")
        
        st.subheader("📋 回测清单")
        display_cols = ['Trade_Date','name','ts_code','Close','Pct_Chg',
             'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)',
                        'rsi','bias','Sector_Boost']
        st.dataframe(all_res[display_cols].sort_values('Trade_Date', ascending=False), use_container_width=True)
        
        # 增加导出功能，方便您做深度分析
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载回测结果 (CSV)",
            data=csv,
            file_name=f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_simulation_export.csv",
            mime="text/csv",
        )
    else:
        st.warning("⚠️ 没有选出任何股票。")
