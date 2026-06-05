# -*- coding: utf-8 -*-
"""
选股王 · V35.14 胜率强化版 (深蹲均线收敛)
------------------------------------------------
修改记录:
1. [胜率过滤] 跌幅过滤：昨日跌幅限制在 -4% ~ 0%，拒绝深度破位死鱼，只做温和深蹲。
2. [分值重塑] 对获利盘(Winner_Rate)实施钟形惩罚机制：仅奖励获利盘在 30%-55% 的“金发姑娘”区。
3. [乖离锁定] 增加均线收敛过滤，强制股价偏离 20日线不得超过 8%，只抓趋势起爆初期。
4. [T+1核心] 继承严苛的 T+1 双向结算法则（10%止盈，6%止损）及右侧 1.5% 动能确认买点。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import concurrent.futures 
import os
import pickle

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
st.set_page_config(page_title="选股王 V35.14 胜率强化版", layout="wide")
st.title("选股王 V35.14：胜率强化与乖离收敛")

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
        for _ in range(3):
            try:
                if kwargs.get('is_index'):
                    df = pro.index_daily(**kwargs)
                else:
                    df = func(**kwargs)
                
                if df is not None and not df.empty:
                    return df
                time.sleep(0.5)
            except:
                time.sleep(1)
                continue
        return pd.DataFrame(columns=['ts_code']) 
    except Exception as e:
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    lookback_days = max(num_days * 3, 365) 
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    
    if cal.empty or 'cal_date' not in cal.columns:
        return []
        
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()

@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    return {'adj': adj_df, 'daily': daily_df}

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
            if not df.empty: all_members.append(df)
            time.sleep(0.02) 
            load_bar.progress((i + 1) / len(index_codes), text=f"加载行业数据: {idx_code}")
        load_bar.empty()
        if not all_members: return {}
        full_df = pd.concat(all_members)
        full_df = full_df.drop_duplicates(subset=['con_code'])
        return dict(zip(full_df['con_code'], full_df['index_code']))
    except Exception as e:
        return {}

# ---------------------------
# 数据获取核心 (复用超高速缓存)
# ---------------------------
CACHE_FILE_NAME = "market_data_cache_v35_10.pkl" 

def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_INDUSTRY
    if not trade_days_list: return False
    
    with st.spinner("正在同步全市场行业数据..."):
        GLOBAL_STOCK_INDUSTRY = load_industry_mapping()

    if use_cache and os.path.exists(CACHE_FILE_NAME):
        st.success(f"⚡ 发现本地行情缓存 ({CACHE_FILE_NAME})，正在极速加载...")
        try:
            with open(CACHE_FILE_NAME, 'rb') as f:
                cached_data = pickle.load(f)
                GLOBAL_ADJ_FACTOR = cached_data['adj']
                GLOBAL_DAILY_RAW = cached_data['daily']
                
            latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
            if latest_global_date:
                try:
                    latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
                    GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
                except: GLOBAL_QFQ_BASE_FACTORS = {}
            
            st.info("✅ 本地缓存加载成功！")
            return True
        except Exception as e:
            st.warning(f"缓存文件损坏，将重新下载: {e}")
            os.remove(CACHE_FILE_NAME)

    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=200)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=30)
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    
    if all_trade_dates_df.empty or 'cal_date' not in all_trade_dates_df.columns:
        st.error("❌ 无法获取交易日历数据。")
        return False
        
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    st.info(f"📡 [首次运行] 正在下载复权行情数据: {start_date} 至 {end_date} (下载后将自动缓存)...")

    adj_factor_data_list = [] 
    daily_data_list = []

    def fetch_worker(date):
        return fetch_and_cache_daily_data(date)

    progress_text = "Tushare 数据下载中..."
    my_bar = st.progress(0, text=progress_text)
    total_steps = len(all_dates)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_date = {executor.submit(fetch_worker, date): date for date in all_dates}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_date)):
            try:
                data = future.result()
                if not data['adj'].empty: adj_factor_data_list.append(data['adj'])
                if not data['daily'].empty: daily_data_list.append(data['daily'])
            except Exception as exc: pass
            
            if i % 5 == 0 or i == total_steps - 1:
                my_bar.progress((i + 1) / total_steps, text=f"下载中: {i+1}/{total_steps}")

    my_bar.empty()
    
    if not daily_data_list:
        st.error("❌ 数据同步失败，请检查网络或休息片刻再试。")
        return False
   
    with st.spinner("正在构建索引并保存缓存..."):
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
            except: GLOBAL_QFQ_BASE_FACTORS = {}
        
        try:
            with open(CACHE_FILE_NAME, 'wb') as f:
                pickle.dump({'adj': GLOBAL_ADJ_FACTOR, 'daily': GLOBAL_DAILY_RAW}, f)
            st.success("💾 行情数据已缓存到硬盘，下次重启将秒开！")
        except Exception as e:
            st.warning(f"缓存写入失败: {e}")
            
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
    except KeyError: return pd.DataFrame()
    
    if daily_df.empty or adj_series.empty: return pd.DataFrame()
    
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

# ---------------------------
# 🌟实战 T+1 仿真与双向结算 
# ---------------------------
def get_future_prices(ts_code, selection_date, d0_qfq_close, days_ahead=[2, 3, 5], stop_loss=6.0, take_profit=10.0):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    
    if hist.empty or len(hist) < 1: return results
    
    hist['open'] = pd.to_numeric(hist['open'], errors='coerce')
    hist['high'] = pd.to_numeric(hist['high'], errors='coerce')
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    hist['low'] = pd.to_numeric(hist['low'], errors='coerce') 
    
    d1_data = hist.iloc[0]
    next_open = d1_data['open']
    next_high = d1_data['high']
    
    if next_open < d0_qfq_close * 0.985: return results 
    if next_open > d0_qfq_close * 1.03: return results 

    # 🌟动能确认：日内突破开盘价 1.5%
    target_buy_price = next_open * 1.015
    if next_high < target_buy_price: return results
    
    stop_loss_price = target_buy_price * (1 - stop_loss / 100.0)
    take_profit_price = target_buy_price * (1 + take_profit / 100.0)
        
    for n in days_ahead:
        col = f'Return_T{n-1} (%)' 
        if len(hist) >= n:
            period_data = hist.iloc[0:n]
            final_return = np.nan
            
            for i_day, (_, row) in enumerate(period_data.iterrows()):
                if i_day == 0:
                    continue # T+0 绝对锁定
                    
                # T+1 结算逻辑
                if row['open'] <= stop_loss_price:
                    final_return = (row['open'] - target_buy_price) / target_buy_price * 100
                    break
                elif row['open'] >= take_profit_price:
                    final_return = (row['open'] - target_buy_price) / target_buy_price * 100
                    break
                    
                if row['low'] <= stop_loss_price:
                    final_return = -stop_loss
                    break
                elif row['high'] >= take_profit_price:
                    final_return = take_profit
                    break
                        
            if pd.isna(final_return):
                sell_price = hist.iloc[n-1]['close']
                final_return = (sell_price - target_buy_price) / target_buy_price * 100
                
            results[col] = final_return
        else:
            results[col] = np.nan
    return results

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
    
    rsi_series = calculate_rsi(close, period=12)
    res['rsi_12'] = rsi_series.iloc[-1]
    
    hist_60 = df.tail(60)
    res['position_60d'] = (close.iloc[-1] - hist_60['low'].min()) / (hist_60['high'].max() - hist_60['low'].min() + 1e-9) * 100

    res['has_limit_up_gene'] = (df['pct_chg'].tail(15).max() >= 9.5)
    
    if len(df) >= 2:
        res['is_vol_shrink'] = df['vol'].iloc[-1] <= df['vol'].iloc[-2] * 1.05 
    else:
        res['is_vol_shrink'] = False

    df.index = pd.to_datetime(df.index)
    weekly_df = df.resample('W').agg({'close': 'last'}).dropna()
    if len(weekly_df) >= 4:
        weekly_ma4 = weekly_df['close'].tail(4).mean() 
        res['is_weekly_uptrend'] = weekly_df['close'].iloc[-1] >= weekly_ma4
    else:
        res['is_weekly_uptrend'] = False
  
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
# 核心回测逻辑函数 
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS, RSI_LIMIT, CHIP_MIN_WIN_RATE, CHIP_MAX_WIN_RATE, SECTOR_THRESHOLD, MIN_MV, MAX_MV, MIN_PREV_PCT, MAX_PREV_PCT, MIN_PRICE, STOP_LOSS_PCT, TAKE_PROFIT_PCT):
    global GLOBAL_STOCK_INDUSTRY
    
    market_state = get_market_state(last_trade)
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失 {last_trade}"

    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    if stock_basic.empty or 'name' not in stock_basic.columns:
        stock_basic = safe_get('stock_basic', list_status='L')
    
    chip_dict = {}
    try:
        chip_df = safe_get('cyq_perf', trade_date=last_trade)
        if not chip_df.empty:
            chip_dict = dict(zip(chip_df['ts_code'], chip_df['winner_rate']))
    except: pass 
    
    strong_industry_codes = set()
    try:
        sw_df = safe_get('sw_daily', trade_date=last_trade)
        if not sw_df.empty:
            strong_sw = sw_df[sw_df['pct_chg'] >= SECTOR_THRESHOLD]
            strong_industry_codes = set(strong_sw['index_code'].tolist())
    except: pass 
        
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    if 'name' not in df.columns: df['name'] = ''

    daily_basic = safe_get('daily_basic', trade_date=last_trade)
    if not daily_basic.empty:
        needed_cols = ['ts_code','turnover_rate','circ_mv','amount']
        existing_cols = [c for c in needed_cols if c in daily_basic.columns]
        df = df.merge(daily_basic[existing_cols], on='ts_code', how='left')
    
    mf_raw = safe_get('moneyflow', trade_date=last_trade)
    if not mf_raw.empty:
        mf = mf_raw[['ts_code','net_mf_amount']].rename(columns={'net_mf_amount':'net_mf'})
        df = df.merge(mf, on='ts_code', how='left')
    else:
        df['net_mf'] = 0 
    
    for col in ['net_mf', 'turnover_rate', 'circ_mv', 'amount']:
        if col not in df.columns: df[col] = 0
    df['net_mf'] = df['net_mf'].fillna(0)
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')] 
    
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= 2000.0)]
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    df = df[df['turnover_rate'] <= MAX_TURNOVER_RATE] 

    # 🌟【纯绿盘锁死】：只吃跌幅不超过4%的健康洗盘阴线
    df = df[(df['pct_chg'] >= MIN_PREV_PCT) & (df['pct_chg'] <= MAX_PREV_PCT)]
    
    if len(df) == 0: return pd.DataFrame(), "过滤后无标的"

    df['activity_score'] = df['turnover_rate']
    candidates = df.sort_values('activity_score', ascending=False).head(FINAL_POOL)
    
    records = []
    for row in candidates.itertuples():
        if GLOBAL_STOCK_INDUSTRY and strong_industry_codes:
            ind_code = GLOBAL_STOCK_INDUSTRY.get(row.ts_code)
            if ind_code and (ind_code not in strong_industry_codes): continue

        ind = compute_indicators(row.ts_code, last_trade)
        if not ind: continue
        
        if not ind.get('has_limit_up_gene', False): continue
        if not ind.get('is_vol_shrink', False): continue
        
        d0_close = ind['last_close']
        d0_rsi = ind.get('rsi_12', 50)
        
        if d0_rsi < 50: continue 
        if d0_rsi > 85: continue 
        
        if market_state == 'Weak':
            if d0_rsi > RSI_LIMIT: continue
            if d0_close < ind['ma20']: continue 
            
        if d0_close < ind['ma60']: continue
        
        if not ind.get('is_weekly_uptrend', False): continue
        
        # 🌟【乖离收缩锁】：股价偏离 MA20 不能超过 8%，防止高位被套
        if ind['ma20'] > 0 and abs(d0_close - ind['ma20']) / ind['ma20'] > 0.08: 
            continue
        
        win_rate = chip_dict.get(row.ts_code, 50) 
        if win_rate < CHIP_MIN_WIN_RATE or win_rate > CHIP_MAX_WIN_RATE: continue

        future = get_future_prices(row.ts_code, last_trade, d0_close, [2, 3, 5], STOP_LOSS_PCT, TAKE_PROFIT_PCT)
        records.append({
            'ts_code': row.ts_code, 'name': row.name, 'Close': row.close, 'Pct_Chg': row.pct_chg,
            'rsi': d0_rsi, 'winner_rate': win_rate, 
            'macd': ind['macd_val'], 'net_mf': row.net_mf,
            'circ_mv': row.circ_mv, 
            'Return_T1 (%)': future.get('Return_T1 (%)', np.nan),
            'Return_T2 (%)': future.get('Return_T2 (%)', np.nan),
            'Return_T4 (%)': future.get('Return_T4 (%)', np.nan),
            'market_state': market_state,
            'Sector_Boost': 'Yes' if GLOBAL_STOCK_INDUSTRY else 'N/A'
        })
            
    if not records: return pd.DataFrame(), "深度筛选后无标的"
    fdf = pd.DataFrame(records)
    
    # 🌟【高胜率打分】：资金流动能 + 获利盘钟形惩罚
    def dynamic_score(r):
        mf_ratio = r['net_mf'] / (r['circ_mv'] * 10000 + 1) if r['circ_mv'] > 0 else 0
        base_score = r['macd'] * 1000 
        base_score += min(max(mf_ratio * 10000, -500), 1000) 
        
        wr = r['winner_rate']
        if 30 <= wr <= 55: base_score += 1500  # 核心金发姑娘区，重奖
        elif wr > 65: base_score -= 1000       # 浮筹太多，重罚
        elif wr < 20: base_score -= 500        # 无人问津，轻罚
        
        if 55 < r['rsi'] < 80: base_score += 500
        
        return base_score

    fdf['Score'] = fdf.apply(dynamic_score, axis=1)
    
    final_df = fdf.sort_values('Score', ascending=False).head(TOP_BACKTEST).copy()
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    
    return final_df, None

# ---------------------------
# UI 及 主程序
# ---------------------------
with st.sidebar:
    st.header("V35.14 胜率强化版")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数", value=30, step=1, help="建议30-50天")
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=4, help="实盘重点看 Rank 1 和 2")
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 清除行情缓存"):
        if os.path.exists(CACHE_FILE_NAME):
            os.remove(CACHE_FILE_NAME)
            st.success("缓存已清除，下次运行将重新下载最新数据。")
    CHECKPOINT_FILE = "backtest_checkpoint_v35_14.csv" 
    
    st.markdown("---")
    st.subheader("⚔️ 实战双向边界 (止盈/止损)")
    TAKE_PROFIT_PCT = st.number_input("动态止盈线 (%)", value=10.0)
    STOP_LOSS_PCT = st.number_input("硬性止损线 (%)", value=6.0)
    
    st.markdown("---")
    st.subheader("💰 基础过滤")
    col1, col2 = st.columns(2)
    MIN_PRICE = col1.number_input("最低股价", value=15.0) 
    MIN_MV = col2.number_input("最小市值(亿)", value=30.0) 
    MAX_MV = st.number_input("最大市值(亿)", value=500.0)
    
    st.markdown("---")
    st.subheader("⚔️ 核心印钞参数 (绝杀锁)")
    CHIP_MIN_WIN_RATE = st.number_input("最低获利盘 (%)", value=20.0)
    CHIP_MAX_WIN_RATE = st.number_input("最大获利盘 (%)", value=65.0, help="最高放宽至 65%，但打分系统倾向于挑选 30-55% 的标的")
    # 🌟强制约束为 -4.0 到 0.0！
    MAX_PREV_PCT = st.number_input("昨日最大涨幅 (%)", value=0.0, help="【深蹲锁】强制最高涨幅为0，绝对禁止买红盘接力！")
    MIN_PREV_PCT = st.number_input("昨日最大跌幅 (%)", value=-4.0, help="【防破位】限制最大跌幅不得低于 -4%，剔除死鱼")
    RSI_LIMIT = st.number_input("弱势拦截线", value=100.0)
    
    st.markdown("---")
    st.subheader("📊 形态参数")
    SECTOR_THRESHOLD = st.number_input("板块涨幅 (%)", value=1.0)
    MAX_UPPER_SHADOW = st.number_input("上影线 (%)", value=6.0) 
    MIN_BODY_POS = st.number_input("实体位置", value=0.5) 
    MAX_TURNOVER_RATE = st.number_input("换手率 (%)", value=20.0)

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🚀 启动 V35.14 胜率引擎"):
    processed_dates = set()
    results = []
    
    if RESUME_CHECKPOINT and os.path.exists(CHECKPOINT_FILE):
        try:
            existing_df = pd.read_csv(CHECKPOINT_FILE)
            existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
            processed_dates = set(existing_df['Trade_Date'].unique())
            results.append(existing_df)
            st.success(f"✅ 检测到断点存档，跳过 {len(processed_dates)} 个交易日...")
        except:
            if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    else:
        if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    
    trade_days_list = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_list: st.stop()
        
    dates_to_run = [d for d in trade_days_list if d not in processed_dates]
    
    if not dates_to_run:
        st.success("🎉 所有日期已计算完毕！")
    else:
        if not get_all_historical_data(trade_days_list, use_cache=True):
            st.stop()
            
        bar = st.progress(0, text="回测引擎启动...")
        
        for i, date in enumerate(dates_to_run):
            res, err = run_backtest_for_a_day(date, int(TOP_BACKTEST), 100, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS, RSI_LIMIT, CHIP_MIN_WIN_RATE, CHIP_MAX_WIN_RATE, SECTOR_THRESHOLD, MIN_MV, MAX_MV, MIN_PREV_PCT, MAX_PREV_PCT, MIN_PRICE, STOP_LOSS_PCT, TAKE_PROFIT_PCT)
            if not res.empty:
                res['Trade_Date'] = date
                is_first = not os.path.exists(CHECKPOINT_FILE)
                res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                results.append(res)
            
            bar.progress((i+1)/len(dates_to_run), text=f"分析中: {date}")
        
        bar.empty()
    
    if results:
        all_res = pd.concat(results)
        all_res = all_res[all_res['Rank'] <= int(TOP_BACKTEST)]
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
        st.header(f"📊 V35.14 胜率强化 (深蹲均线收敛)")
        cols = st.columns(3)
        for idx, n in enumerate([2, 3, 5]):
            col_name = f'Return_T{n-1} (%)'
            valid = all_res.dropna(subset=[col_name]) 
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                cols[idx].metric(f"T+{n-1} 均益 / 胜率", f"{avg:.2f}% / {win:.1f}%")
 
        st.subheader("📋 回测清单 (优先锁定 Rank 1 和 2)")
        
        show_cols = ['Rank', 'Trade_Date','name','ts_code','Close','Pct_Chg',
             'Return_T1 (%)', 'Return_T2 (%)', 'Return_T4 (%)',
                        'rsi','winner_rate','Sector_Boost']
        final_cols = [c for c in show_cols if c in all_res.columns]
    
        display_df = all_res[final_cols].sort_values(['Trade_Date', 'Rank'], ascending=[False, True])
        st.dataframe(display_df, use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载结果 (CSV)", csv, f"export_v35_14.csv", "text/csv")
    else:
        st.warning("⚠️ 没有结果。")
