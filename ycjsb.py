# -*- coding: utf-8 -*-
"""
选股王 · V39.6 主板一字板过滤版
------------------------------------------------
核心改进 (基于V39.5):
1. [周线松绑] 移除周线 MA10 > MA20 的硬性限制，允许底部爆量反转的大牛股进场。
2. [防高位陷阱] 增加周线乖离率与高位上影线风控，专门绞杀高位 B 浪诱多派发。
3. [日线爆量点火] 保持日线 20 日线爆量反包的极致灵敏度。
4. [去未来函数] 买入价为信号日次日(T+1)开盘价，而非信号当天收盘价。
5. [入场信号收紧] 突破幅度：收盘价需高出MA20至少2%。
6. [止损止盈-三层简化] 固定止损-8%/-12% → 浮盈满10%保本 → 浮盈满20%后回撤15%移动止盈，
   全程用最高价追踪浮盈峰值，固定止损全程兜底生效，保本/移动止盈延迟到次日开盘结算。
7. [301止损档位/量比基准] 已修复。

【★V39.6新增：主板一字涨停过滤】
   主板股票(非创业板300/301、非科创板688/689)信号次日如果开盘=最高=最低(全天封在
   同一价位、没有任何成交空间)，说明是一字板，实盘根本买不进去。原来的回测把这种情况
   当成"能以开盘价成交"，会虚增存活率/胜率。现在检测到这种情况会把这笔交易整个从统计
   中剔除(Return_W1~W8全部留空，不计入存活/胜率)，Exit_Reason标注"一字板无法买入(剔除)"，
   方便你直接看出信号里有多少比例是实盘根本没法执行的。
   双创板(创业板/科创板)按实盘经验不会出现一字板，不做此项检查。
   顺带修复了表格颜色标注函数(color_exit)——它里面匹配的还是V39.2之前"一档/二档/假突破"
   这些早就废弃的旧名称，一直没起作用，现在改成匹配现有的固定止损/保本止盈/移动止盈/
   一字板/周期结束平仓这几个真实存在的分类。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import concurrent.futures 
import os
import pickle

warnings.filterwarnings("ignore")

CACHE_FILE_NAME = "market_data_cache_v39_1.pkl"  # 原始行情数据缓存，未变更，可复用

# ---------------------------
# 全局变量与探针
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_INDUSTRY = {} 
SINA_STATUS = {'success': 0, 'fail': 0} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V39.6 主板一字板过滤版", layout="wide")
st.title("选股王 V39.6：周线温和过滤 + 日线爆量共振（新增主板一字板过滤）")

# ---------------------------
# 新浪实时行情引擎
# ---------------------------
def get_sina_realtime_kline(ts_code):
    global SINA_STATUS
    code_split = ts_code.split('.')
    if len(code_split) != 2: return None
    sina_code = code_split[1].lower() + code_split[0]
    
    url = f"http://hq.sinajs.cn/list={sina_code}"
    headers = {'Referer': 'https://finance.sina.com.cn'}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = 'gbk'
        data_str = response.text.split('="')[1].split('";')[0]
        if not data_str: 
            SINA_STATUS['fail'] += 1
            return None
        data_list = data_str.split(',')
        
        SINA_STATUS['success'] += 1
        return {
            'trade_date_str': datetime.now().strftime('%Y%m%d'),
            'open': float(data_list[1]),
            'pre_close': float(data_list[2]),
            'close': float(data_list[3]),
            'high': float(data_list[4]),
            'low': float(data_list[5]),
            'vol': (float(data_list[8]) / 100) * (240 / 225) 
        }
    except Exception:
        SINA_STATUS['fail'] += 1
        return None

# ---------------------------
# 基础 API 函数
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        for _ in range(3):
            try:
                if func_name == 'index_daily': 
                    df = pro.index_daily(**kwargs)
                else: 
                    df = func(**kwargs)
                if df is not None and not df.empty: return df
                time.sleep(0.5)
            except: time.sleep(1); continue
        return pd.DataFrame(columns=['ts_code']) 
    except Exception as e: return pd.DataFrame(columns=['ts_code'])

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

@st.cache_data(ttl=3600*24*7) 
def load_industry_mapping():
    global pro
    if pro is None: return {}
    try:
        sw_indices = pro.index_classify(level='L1', src='SW2021')
        if sw_indices.empty: return {}
        white_list_names = ['电子', '计算机', '通信', '医药生物', '国防军工', '机械设备']
        target_indices = sw_indices[sw_indices['industry_name'].isin(white_list_names)]
        index_codes = target_indices['index_code'].tolist()
        
        all_members = []
        load_bar = st.progress(0, text="正在加载硬科技白名单赛道数据...")
        for i, idx_code in enumerate(index_codes):
            df = pro.index_member(index_code=idx_code, is_new='Y')
            if not df.empty: 
                df['industry_code'] = idx_code
                all_members.append(df)
            time.sleep(0.05) 
            load_bar.progress((i + 1) / len(index_codes))
        load_bar.empty()
        
        if not all_members: return {}
        full_df = pd.concat(all_members).drop_duplicates(subset=['con_code'])
        return dict(zip(full_df['con_code'], full_df['industry_code']))
    except Exception:
        return {}

# ---------------------------
# 数据获取与复权引擎
# ---------------------------
def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_INDUSTRY
    if not trade_days_list: return False
    
    with st.spinner("正在同步全市场行业数据 (白名单)..."):
        GLOBAL_STOCK_INDUSTRY = load_industry_mapping()

    if use_cache and os.path.exists(CACHE_FILE_NAME):
        st.success(f"⚡ 发现本地行情缓存，极速加载中...")
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
            return True
        except Exception:
            os.remove(CACHE_FILE_NAME)

    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=150)).strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty: return False
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    st.info(f"📡 [首次运行] 正在下载复权行情数据...")
    adj_factor_data_list, daily_data_list = [], []

    my_bar = st.progress(0, text="Tushare 数据下载中...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_date = {executor.submit(fetch_and_cache_daily_data, date): date for date in all_dates}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_date)):
            try:
                data = future.result()
                if not data['adj'].empty: adj_factor_data_list.append(data['adj'])
                if not data['daily'].empty: daily_data_list.append(data['daily'])
            except Exception: pass
            if i % 5 == 0 or i == len(all_dates) - 1:
                my_bar.progress((i + 1) / len(all_dates), text=f"下载中: {i+1}/{len(all_dates)}")
    my_bar.empty()
    
    if not daily_data_list: return False
   
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
        except Exception: pass
            
    return True

def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date, use_sina=False):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    
    latest_adj_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj_factor): return pd.DataFrame() 

    try:
        daily_df = GLOBAL_DAILY_RAW.loc[ts_code]
        daily_df = daily_df.loc[(daily_df.index >= start_date) & (daily_df.index <= end_date)].copy()
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
        
    final_df = df[['open', 'high', 'low', 'close', 'pre_close', 'vol']].copy() 

    if use_sina:
        today_str = datetime.now().strftime('%Y%m%d')
        if end_date == today_str:
            sina_data = get_sina_realtime_kline(ts_code)
            if sina_data and sina_data['close'] > 0:
                sina_row = pd.DataFrame([sina_data]).set_index('trade_date_str')
                if today_str in final_df.index:
                    final_df.loc[today_str] = sina_row.iloc[0]
                else:
                    final_df = pd.concat([final_df, sina_row])
                    
    return final_df

# ---------------------------
# 核心指标计算 (温和周线过滤)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def compute_trend_indicators(ts_code, end_date, use_sina=False, _run_id=None):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date, end_date, use_sina=use_sina)
    res = {}
    if df.empty or len(df) < 120: return res 
    
    # 1. 日线指标计算
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['ma120'] = df['close'].rolling(120).mean()
    df['ma5_vol'] = df['vol'].shift(1).rolling(5).mean()  # 【V39.5修复漏洞③】前5天均量(不含当天)，避免自己拉高自己的基准线
    
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    df_calc = df.dropna().copy().reset_index()
    if len(df_calc) < 20: return res

    # 2. 周线重采样合成
    df_calc['dt'] = pd.to_datetime(df_calc['trade_date_str'])
    iso_cal = df_calc['dt'].dt.isocalendar()
    df_calc['year_week'] = iso_cal.year.astype(str) + "_" + iso_cal.week.astype(str).str.zfill(2)
    
    weekly_df = df_calc.groupby('year_week', as_index=False).agg({
        'trade_date_str': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum'
    }).sort_values('trade_date_str').reset_index(drop=True)
    
    if len(weekly_df) < 10: return res
    
    weekly_df['w_ma20'] = weekly_df['close'].rolling(20).mean()
    
    w_curr = weekly_df.iloc[-1]
    w_prev = weekly_df.iloc[-2] if len(weekly_df) >= 2 else w_curr
    
    # 【温和周线过滤 1】：防高位过度偏离（股价不能高于 20周线 45% 以上，防止高位派发接盘）
    w_bias_safe = True
    if not pd.isna(w_curr['w_ma20']) and w_curr['w_ma20'] > 0:
        w_bias = (w_curr['close'] - w_curr['w_ma20']) / w_curr['w_ma20']
        if w_bias > 0.45:
            w_bias_safe = False
            
    # 【温和周线过滤 2】：防高位长上影线（上一周上影线不能占总振幅 60% 以上，防止墓碑线陷阱）
    w_shadow_safe = True
    w_prev_range = w_prev['high'] - w_prev['low']
    w_prev_upper_shadow = w_prev['high'] - max(w_prev['open'], w_prev['close'])
    if w_prev_range > 0 and (w_prev_upper_shadow / w_prev_range) >= 0.60:
        w_shadow_safe = False

    is_weekly_safe = w_bias_safe and w_shadow_safe

    # 3. 日线突破点火信号
    row = df_calc.iloc[-1]
    prev_row = df_calc.iloc[-2]
    prev2_row = df_calc.iloc[-3]
    
    is_daily_trend_up = row['ma60'] > row['ma120']
    is_daily_pulled_back = (prev2_row['close'] < prev2_row['ma20']) and (prev_row['close'] < prev_row['ma20'])

    # 【V39.3改动①】突破幅度收紧：不再是"微弱越线"，要求收盘价至少高出MA20 2%，
    # 排除只是贴着均线蹭一下的弱突破
    is_daily_breakout = row['close'] > row['ma20'] * 1.02

    is_daily_ma20_healthy = row['ma20'] >= prev_row['ma20']
    is_daily_vol_strong = row['vol'] > (1.2 * row['ma5_vol'])
    
    candle_range = row['high'] - row['low']
    candle_body = row['close'] - row['open']
    is_solid_yang = (row['close'] > row['open']) and (candle_body >= candle_range * 0.6 if candle_range > 0 else True)
    is_macd_healthy = (row['dif'] > 0) and (row['macd'] > prev_row['macd'])

    # 【V39.4改动】去掉V39.3新加的"近期新高确认"(③)：
    # 它和已有的 is_daily_pulled_back(要求最近2天在MA20下方，即"先回调过")天然矛盾——
    # 稍微像样一点的回调(比如从高点跌8%~10%再企稳反包，典型主升浪走法)，
    # 20日最高点会明显高于当前价，③会把这类票直接挡在外面，导致信号数量骤减、
    # 光迅科技这类真正的主升浪票反而被误杀。保留①(突破幅度收紧到MA20*1.02)，
    # 因为①和"先回调"不矛盾，两者可以共存。
    
    # 【综合决策】：温和周线风控 + 日线强攻击信号 + 突破幅度确认(①)
    res['is_v38_buy_signal'] = (is_weekly_safe and 
                                is_daily_trend_up and is_daily_pulled_back and 
                                is_daily_breakout and is_daily_ma20_healthy and 
                                is_daily_vol_strong and is_solid_yang and is_macd_healthy)
    
    if res['is_v38_buy_signal']:
        res['vol_ratio'] = row['vol'] / row['ma5_vol']  
        res['pre_close'] = prev_row['close']            
        
    res['last_close'] = row['close']
    res['bottom_line'] = row['low'] 
    res['ma20'] = row['ma20']
    
    return res

# ---------------------------
# V39.3 核心大脑：三层简化止盈止损系统 (T+1开盘买入，不依赖均线)
# ---------------------------
def get_medium_term_future(ts_code, selection_date, signal_close, bottom_line, hold_weeks=8, use_sina=False):
    # 注意：signal_close 是信号当天的收盘价，仅作为参考(计算Gap_pct用)，
    # 不再作为买入价——买入价改为下面从 hist_future 里取的 T+1 开盘价。
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_fetch = (d0 - timedelta(days=60)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=150)).strftime("%Y%m%d") 
    
    hist_full = get_qfq_data_v4_optimized_final(ts_code, start_date=start_fetch, end_date=end_future, use_sina=use_sina)
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    results['Exit_Reason'] = "持仓中"
    results['Buy_Price'] = np.nan
    results['Gap_pct (%)'] = np.nan
    
    if hist_full.empty or len(hist_full) < 30: return results
    
    hist_full['open'] = pd.to_numeric(hist_full['open'], errors='coerce')
    hist_full['high'] = pd.to_numeric(hist_full['high'], errors='coerce')
    hist_full['low'] = pd.to_numeric(hist_full['low'], errors='coerce')
    hist_full['close'] = pd.to_numeric(hist_full['close'], errors='coerce')
    
    hist_full['ma10'] = hist_full['close'].rolling(10).mean()
    hist_full['ma20'] = hist_full['close'].rolling(20).mean()
    
    hist_future = hist_full[hist_full.index > selection_date]

    # 【去未来函数】没有次日数据（比如信号刚好是数据里最后一天）就无法确定买入价，直接返回
    if hist_future.empty:
        return results

    next_row = hist_future.iloc[0]

    # 【V39.6新增】主板一字涨停过滤：主板(非创业板301/300、非科创板688/689)信号次日
    # 如果开盘=最高=最低(全天封在同一价位、没有任何成交空间)，说明是一字板，实际根本买不进去。
    # 双创板(创业板/科创板)按用户实盘经验不会出现这种情况，不做这项检查。
    is_main_board = not (ts_code.startswith('300') or ts_code.startswith('301') 
                          or ts_code.startswith('688') or ts_code.startswith('689'))
    is_one_word_limit = (is_main_board and pd.notna(next_row['open']) and pd.notna(next_row['high']) 
                          and pd.notna(next_row['low']) and next_row['open'] == next_row['high'] == next_row['low'])
    if is_one_word_limit:
        results['Exit_Reason'] = "一字板无法买入(剔除)"
        results['Buy_Price'] = round(next_row['open'], 2)  # 仅作参考，实际没有成交机会
        # Return_W1~W8 全部保持NaN，不计入存活/胜率统计——因为这笔交易实盘根本成交不了
        return results

    buy_price = next_row['open']
    if pd.isna(buy_price) or buy_price <= 0:
        return results

    results['Buy_Price'] = round(buy_price, 2)
    if signal_close and signal_close > 0:
        results['Gap_pct (%)'] = round((buy_price - signal_close) / signal_close * 100, 2)

    # 【V39.3设计，V39.5修复①④⑤】止损止盈三层，全程只看价格与买入价/最高价的关系，不再依赖均线：
    #   第0层：固定止损 -8%/-12%（创业板/科创板/301新股放宽），全程生效的兜底防线，不因升档而失效
    #   第1层：浮盈(按盘中最高价计)达到10% → 止损线上移到买入价（保本，留0.5%缓冲）
    #   第2层：浮盈(按盘中最高价计)达到20% → 切换为"从最高价回撤15%止盈"
    #   保本/移动止盈的退出条件基于收盘价判断，但成交价延迟到次日开盘价结算(去未来函数)；
    #   固定止损是盘中触发的止损单，当天用低点判断、以开盘价与止损价的较差者结算，维持原有处理方式。
    exit_triggered = False
    tier = 0  # 0=固定止损期, 1=保本期, 2=移动止盈期
    peak_close = buy_price
    pending_exit_reason = None  # 【V39.5修复④】昨日收盘触发的保本/止盈，今日开盘执行
    
    # 【V39.5修复漏洞②】301开头(创业板注册制新股)同样是20%涨跌停板，之前只认300/688漏判了301
    is_20cm = ts_code.startswith('300') or ts_code.startswith('301') or ts_code.startswith('688')
    hard_stop_limit = -0.12 if is_20cm else -0.08
    
    for i in range(len(hist_future)):
        if i >= hold_weeks * 5: break 
            
        row = hist_future.iloc[i]
        day_count = i + 1
        current_week = ((day_count - 1) // 5) + 1 
        
        curr_open = row['open']
        curr_close = row['close']
        curr_high = row['high']
        curr_low = row['low']
        
        final_return = np.nan
        
        # 【V39.5修复④】昨天收盘已经触发保本/移动止盈，今天用开盘价结算离场
        if pending_exit_reason is not None:
            final_return = (curr_open - buy_price) / buy_price * 100.0
            exit_triggered = True
            results['Exit_Reason'] = pending_exit_reason
            results[f'Return_W{current_week} (%)'] = final_return
            break
        
        # 【V39.5修复⑤】峰值用盘中最高价追踪，不再用收盘价（避免低估真实触及过的浮盈）
        peak_close = max(peak_close, curr_high)
        peak_profit_pct = (peak_close - buy_price) / buy_price
        
        # 【V39.5修复①】固定止损兜底：不管当前处于哪一档，跌破-8%/-12%都立刻离场
        if (curr_low - buy_price) / buy_price <= hard_stop_limit:
            final_return = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
            exit_triggered = True
            results['Exit_Reason'] = f"固定止损(破{int(hard_stop_limit*100)}%)"
            results[f'Return_W{current_week} (%)'] = final_return
            break
        
        if tier == 0 and peak_profit_pct >= 0.10:
            tier = 1
                
        if tier == 1:
            # 保本层：浮盈到过10%后，跌破买入价*0.995就走（留0.5%缓冲）——次日开盘结算
            if curr_close < buy_price * 0.995:
                pending_exit_reason = "保本止盈"
            elif peak_profit_pct >= 0.20:
                tier = 2
                
        if tier == 2:
            # 移动止盈层：浮盈(按最高价)到过20%后，从最高价回撤15%就走——次日开盘结算
            giveback = (peak_close - curr_close) / peak_close
            if giveback >= 0.15:
                pending_exit_reason = "移动止盈(回撤15%)"
            
        if day_count % 5 == 0:
            results[f'Return_W{current_week} (%)'] = (curr_close - buy_price) / buy_price * 100.0
            
    if not exit_triggered and len(hist_future) >= hold_weeks * 5:
        last_price = hist_future.iloc[hold_weeks * 5 - 1]['close']
        results[f'Return_W{hold_weeks} (%)'] = (last_price - buy_price) / buy_price * 100.0
        results['Exit_Reason'] = "周期结束平仓"
        
    return results

# ---------------------------
# 核心回测循环
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, MIN_MV, MAX_MV, MIN_PRICE, use_sina=False, run_timestamp=None):
    global GLOBAL_STOCK_INDUSTRY

    query_date = last_trade
    daily_all = safe_get('daily', trade_date=query_date) 
    
    if use_sina and daily_all.empty:
        for i in range(1, 10):
            temp_date = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=i)).strftime("%Y%m%d")
            daily_all = safe_get('daily', trade_date=temp_date)
            if not daily_all.empty:
                query_date = temp_date
                break
                
    if daily_all.empty: return pd.DataFrame(), "数据缺失"

    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name')
    if stock_basic.empty: stock_basic = safe_get('stock_basic', list_status='L')
        
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    
    daily_basic = safe_get('daily_basic', trade_date=query_date)
    if not daily_basic.empty:
        df = df.merge(daily_basic[['ts_code','circ_mv']], on='ts_code', how='left')
    else: 
        return pd.DataFrame(), "市值数据缺失"
    
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')] 
    
    df = df[(df['close'] >= MIN_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    
    records = []
    for row in df.itertuples():
        if GLOBAL_STOCK_INDUSTRY and row.ts_code not in GLOBAL_STOCK_INDUSTRY: 
            continue
            
        ind = compute_trend_indicators(row.ts_code, last_trade, use_sina=use_sina, _run_id=run_timestamp)
        if not ind or not ind.get('is_v38_buy_signal'): 
            continue
            
        if use_sina and ind['last_close'] < MIN_PRICE:
            continue
            
        pct_chg = (ind['last_close'] - ind['pre_close']) / ind['pre_close'] * 100
        score_breakout = pct_chg * 10 
        
        score_vol = ind['vol_ratio'] * 10
        total_score = score_breakout + score_vol
        
        # 注意：ind['last_close'] 是信号日收盘价，现在只作为 Gap_pct 的参考基准传入，
        # 真正的买入价在 get_medium_term_future 内部用 T+1 开盘价重新确定。
        future_returns = get_medium_term_future(row.ts_code, last_trade, ind['last_close'], ind['bottom_line'], hold_weeks=8, use_sina=use_sina)
        
        record_dict = {
            'ts_code': row.ts_code, 'name': row.name, 'Signal_Close': ind['last_close'], 
            'circ_mv': row.circ_mv_billion,
            'Total_Score': round(total_score, 1),
            'Breakout_S': round(score_breakout, 1),
            'Volume_S': round(score_vol, 1)
        }
        record_dict.update(future_returns)
        records.append(record_dict)
            
    if not records: return pd.DataFrame(), "无标的"
    
    fdf = pd.DataFrame(records)
    final_df = fdf.sort_values('Total_Score', ascending=False).head(TOP_BACKTEST).copy()
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    return final_df, None

# ---------------------------
# UI 及 主程序
# ---------------------------
with st.sidebar:
    st.header("V39.6 主板一字板过滤版")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数 (设为 1 即启动实盘雷达)", value=100, step=1)
    
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=3)
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 清除行情缓存"):
        if os.path.exists(CACHE_FILE_NAME):
            os.remove(CACHE_FILE_NAME)
            st.success("缓存已清除，下次运行将重新下载最新数据。")
    CHECKPOINT_FILE = "backtest_checkpoint_v39_6_oneword.csv" 
    if st.button("🗑️ 清除断点记录 (重新回测)"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            st.success("旧进度已清理！")
            
    st.markdown("---")
    st.subheader("💰 核心护城河门槛")
    MIN_PRICE = st.number_input("最低股价 (元)", value=20.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=200.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🚀 启动 V39.6 追踪(一字板过滤)"):
    SINA_STATUS = {'success': 0, 'fail': 0}
    processed_dates = set()
    results = []
    
    if RESUME_CHECKPOINT and os.path.exists(CHECKPOINT_FILE):
        try:
            existing_df = pd.read_csv(CHECKPOINT_FILE)
            existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
            processed_dates = set(existing_df['Trade_Date'].unique())
            results.append(existing_df)
        except:
            if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    else:
        if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
        
    trade_days_list = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_list: st.stop()
    
    if not get_all_historical_data(trade_days_list, use_cache=True): st.stop()
            
    dates_to_run = [d for d in trade_days_list if d not in processed_dates]
    if not dates_to_run:
        st.success("🎉 扫描已全部完毕！")
    else:
        bar = st.progress(0, text="温和周线风控与爆量点火计算中...")
        for i, date in enumerate(dates_to_run):
            
            is_realtime_radar = (int(BACKTEST_DAYS) == 1 and date == datetime.now().strftime("%Y%m%d"))
            run_timestamp = time.time() if is_realtime_radar else None
            
            res, err = run_backtest_for_a_day(
                date, int(TOP_BACKTEST), MIN_MV, MAX_MV, MIN_PRICE,
                use_sina=is_realtime_radar, run_timestamp=run_timestamp
            )
            
            if not res.empty:
                res['Trade_Date'] = date
                is_first = not os.path.exists(CHECKPOINT_FILE)
                res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                results.append(res)
            bar.progress((i+1)/len(dates_to_run), text=f"分析中: {date}")
        bar.empty()
    
    if int(BACKTEST_DAYS) == 1:
        st.markdown("---")
        if SINA_STATUS['success'] > 0:
            st.success(f"✅ **盘中实时探针响应正常**：成功接入新浪底层数据 {SINA_STATUS['success']} 次，行情已接管。")
        elif SINA_STATUS['fail'] > 0:
            st.error(f"❌ **盘中实时探针警告**：新浪数据抓取失败 {SINA_STATUS['fail']} 次。请确认当前是否在交易时间。")
        else:
            st.info("ℹ️ 实时探针未触发（可能由于基础选股条件未通过）。")
        st.markdown("---")
    
    if results:
        all_res = pd.concat(results)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
        st.header(f"📊 V39.6 主板一字板过滤版")
        st.subheader("🗓️ 周度生存与收益切片")
        cols_row1 = st.columns(4)
        cols_row2 = st.columns(4)
        
        for w in range(1, 9):
            col_name = f'Return_W{w} (%)'
            if col_name in all_res.columns:
                valid = all_res.dropna(subset=[col_name]) 
                target_col = cols_row1[w-1] if w <= 4 else cols_row2[w-5]
                with target_col:
                    if not valid.empty:
                        avg = valid[col_name].mean()
                        win = (valid[col_name] > 0).mean() * 100
                        st.metric(f"W{w} 均益/胜率 (存活{len(valid)}只)", f"{avg:.2f}% / {win:.1f}%")
                    else:
                        st.metric(f"W{w} 无持仓", "N/A")
                        
        st.subheader("📋 优等生清单")
        display_cols = [
            'Rank', 'Trade_Date', 'name', 'ts_code', 'Signal_Close', 'Buy_Price', 'Gap_pct (%)',
            'Total_Score', 'Breakout_S', 'Volume_S', 'circ_mv', 'Exit_Reason'
        ] + [f'Return_W{w} (%)' for w in range(1, 9)]
        final_cols = [c for c in display_cols if c in all_res.columns]
    
        display_df = all_res[final_cols].sort_values(['Trade_Date', 'Rank'], ascending=[False, True]).reset_index(drop=True)
        
        def color_exit(val):
            if isinstance(val, str):
                if '固定止损' in val: return 'color: white; background-color: darkred'
                elif '一字板' in val: return 'color: white; background-color: gray'
                elif '保本止盈' in val: return 'color: orange'
                elif '移动止盈' in val: return 'color: green'
                elif '周期结束平仓' in val: return 'color: blue'
            return ''
        
        if 'Exit_Reason' in display_df.columns:
            try:
                st.dataframe(display_df.style.map(color_exit, subset=['Exit_Reason']), use_container_width=True)
            except AttributeError:
                st.dataframe(display_df.style.applymap(color_exit, subset=['Exit_Reason']), use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载完整轨迹 (CSV)", csv, f"export_v39_6_oneword.csv", "text/csv")
    else:
        st.warning("⚠️ 暂无符合条件的标的。")
