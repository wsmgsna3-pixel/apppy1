# -*- coding: utf-8 -*-
"""
选股王 · V41.0 资金接力版 (换手率排序)
战略转型：
1. 核心排序：使用 [换手率] (人气/资金) 替代 技术指标。
   - 逻辑：只做市场资金关注度最高的“换手龙”。
2. 硬性门槛：量比 > 1.5 (拒绝缩量假涨)。
3. 基础风控：涨幅 6-16% + 站上20日线。
4. 买入：[-1%, +3%] 顺势接力。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import os
import gc

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V41.0 资金接力战法", layout="wide")
st.title("🚀 V41.0 资金接力监控台 (换手率排序)")

# ---------------------------
# 全局设置
# ---------------------------
SCORE_DB_FILE = "v41_turnover_db.csv" # 新数据库
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_CALENDAR = [] 

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
    start_search = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=max(num_days * 5, 120))).strftime("%Y%m%d")
    end_search = (datetime.strptime(end_date_str, "%Y%m%d") + timedelta(days=60)).strftime("%Y%m%d")
    
    cal = safe_get('trade_cal', start_date=start_search, end_date=end_search)
    if cal.empty or 'is_open' not in cal.columns: return []
    
    global GLOBAL_CALENDAR
    open_cal = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=True)
    GLOBAL_CALENDAR = open_cal['cal_date'].tolist()
    
    past_days = open_cal[open_cal['cal_date'] <= end_date_str]['cal_date'].tolist()
    return past_days[-num_days:]

# ----------------------------------------------------------------------
# 数据下载
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    daily_df = safe_get('daily', trade_date=date)
    adj_df = safe_get('adj_factor', trade_date=date)
    # 必须获取量比和换手率
    basic_df = safe_get('daily_basic', trade_date=date, fields='ts_code,circ_mv,turnover_rate,volume_ratio')
    name_df = safe_get('stock_basic', fields='ts_code,name')
    
    if not daily_df.empty:
        daily_df = daily_df[daily_df['ts_code'].str.startswith(('30', '688'))]
        if not basic_df.empty: daily_df = daily_df.merge(basic_df, on='ts_code', how='left')
        if not name_df.empty: daily_df = daily_df.merge(name_df, on='ts_code', how='left')

    if not adj_df.empty:
        adj_df = adj_df[adj_df['ts_code'].str.startswith(('30', '688'))]
        
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(select_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_CALENDAR
    if not select_days_list: return False
    
    first_select_date = min(select_days_list)
    last_select_date = max(select_days_list)
    
    try:
        last_idx = GLOBAL_CALENDAR.index(last_select_date)
        end_fetch_idx = min(last_idx + 20, len(GLOBAL_CALENDAR) - 1)
        end_fetch_date = GLOBAL_CALENDAR[end_fetch_idx]
    except:
        end_fetch_date = (datetime.now() + timedelta(days=20)).strftime("%Y%m%d")

    start_fetch_date = (datetime.strptime(first_select_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    cal_range = safe_get('trade_cal', start_date=start_fetch_date, end_date=end_fetch_date, is_open='1')
    all_dates = cal_range['cal_date'].tolist()
    
    st.info(f"⏳ 正在拉取数据 ({start_fetch_date} ~ {end_fetch_date})...")

    adj_list, daily_list = [], []
    bar = st.progress(0)
    
    total_steps = len(all_dates)
    for i, date in enumerate(all_dates):
        try:
            cached = fetch_and_cache_daily_data(date)
            if not cached['adj'].empty: adj_list.append(cached['adj'])
            if not cached['daily'].empty: daily_list.append(cached['daily'])
            if i % 20 == 0: bar.progress((i+1)/total_steps)
        except: continue 
    bar.empty()

    if not adj_list or not daily_list: return False
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw = pd.concat(daily_list)
    cols_to_float = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'circ_mv', 'pct_chg', 'turnover_rate', 'volume_ratio']
    for col in cols_to_float:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    latest_date_in_data = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_date_in_data:
        GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_date_in_data), 'adj_factor'].droplevel(1).to_dict()
    
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
    return df.reset_index().sort_values('trade_date')

# ----------------------------------------------------------------------
# 核心算法：趋势检查 + 换手率排序
# ----------------------------------------------------------------------
def check_trend_and_get_score(ts_code, current_date):
    """
    1. 检查是否站上20日线 (趋势护城河)。
    2. 检查 MACD 是否大于0 (基本多头)。
    3. 返回 Score = 换手率 (Turnover)。
    """
    try:
        start_date = (datetime.strptime(current_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
        df = get_qfq_data(ts_code, start_date, current_date)
        
        if df.empty or len(df) < 30: return None
        
        last_row = df.iloc[-1]
        last_date_val = last_row['trade_date']
        last_date_str = last_date_val.strftime('%Y%m%d') if hasattr(last_date_val, 'strftime') else str(last_date_val)
        
        if last_date_str != current_date: return None

        close = df['close']
        
        # 计算 20日均线
        ma20 = close.rolling(window=20).mean()
        
        # 计算 MACD (辅助检查)
        ema_fast = close.ewm(span=8, adjust=False).mean()
        ema_slow = close.ewm(span=17, adjust=False).mean()
        diff = ema_fast - ema_slow
        dea = diff.ewm(span=5, adjust=False).mean()
        macd = (diff - dea) * 2
        
        # 条件A: 趋势向上 (收盘价 > 20日线)
        trend_ok = close.iloc[-1] > ma20.iloc[-1]
        
        # 条件B: 多头区域 (MACD > 0)
        macd_ok = macd.iloc[-1] > 0
        
        if trend_ok and macd_ok:
            # 如果满足趋势要求，返回 None，分数由外层 Turnover 决定
            return True
            
        return None
    except Exception:
        return None

def batch_compute_scores(date):
    try:
        daily_t = GLOBAL_DAILY_RAW.xs(date, level='trade_date')
    except KeyError: return []

    mask = (daily_t['vol'] > 0) & (daily_t['close'] >= 2.0)
    pool = daily_t[mask]
    if pool.empty: return []

    results = []
    candidates = pool.index.tolist()
    
    for code in candidates:
        # 先做趋势检查
        if check_trend_and_get_score(code, date):
            row = pool.loc[code]
            
            turnover = float(row['turnover_rate']) if 'turnover_rate' in row else 0.0
            vol_ratio = float(row['volume_ratio']) if 'volume_ratio' in row else 0.0
            
            # 核心：Score 直接等于 换手率
            score = turnover 
            
            results.append({
                'Select_Date': date,
                'Code': code,
                'Score': score, # 换手率
                'Name': row['name'] if 'name' in row else code,
                'Close': float(row['close']),
                'Pct_Chg': float(row['pct_chg']) if 'pct_chg' in row else 0.0,
                'Circ_Mv': float(row['circ_mv']) if 'circ_mv' in row else 0.0,
                'Turnover': turnover,
                'Vol_Ratio': vol_ratio
            })
    return results

# ----------------------------------------------------------------------
# 动态筛选与回测
# ----------------------------------------------------------------------
def apply_strategy_and_backtest(df_scores, top_n, min_mv_yi, min_pct, max_pct, min_vol_ratio, buy_open_min, buy_open_max, stop_loss_pct):
    min_mv_val = min_mv_yi * 10000
    
    # 筛选：市值、涨幅、量比
    mask = (df_scores['Circ_Mv'] >= min_mv_val) & \
           (df_scores['Pct_Chg'] >= min_pct) & \
           (df_scores['Pct_Chg'] <= max_pct) & \
           (df_scores['Vol_Ratio'] >= min_vol_ratio) # 新增量比过滤

    filtered_df = df_scores[mask].copy()
    if filtered_df.empty: return []
    
    # 排序：按换手率 (Score) 从高到低
    filtered_df = filtered_df.sort_values('Score', ascending=False).head(top_n)
    
    select_date = str(filtered_df.iloc[0]['Select_Date'])
    try:
        t_idx = GLOBAL_CALENDAR.index(select_date)
        if t_idx < len(GLOBAL_CALENDAR) - 1:
            buy_date = GLOBAL_CALENDAR[t_idx + 1]
        else:
            buy_date = None
    except: buy_date = None
    
    final_results = []
    
    for rank, (idx, row) in enumerate(filtered_df.iterrows(), 1):
        code = row['Code']
        signal = "⏳"
        open_pct = np.nan
        is_buy = False
        ret_d3, ret_d5 = np.nan, np.nan
        status = "-"
        
        if buy_date:
            try:
                d1_raw = GLOBAL_DAILY_RAW.loc[(code, buy_date)]
                if isinstance(d1_raw, pd.DataFrame): d1_raw = d1_raw.iloc[0]

                daily_buy_open = float(d1_raw['open'])
                daily_buy_pre = float(d1_raw['pre_close'])
                open_pct = (daily_buy_open / daily_buy_pre - 1) * 100
                
                # 买入条件
                if buy_open_min <= open_pct <= buy_open_max:
                    is_buy = True
                    signal = "✅ BUY"
                else:
                    signal = "👀 观望"
                
                if is_buy:
                    future_df = get_qfq_data(code, buy_date, "20991231")
                    if not future_df.empty:
                        buy_price = future_df.iloc[0]['open']
                        stop_price = buy_price * (1 - abs(stop_loss_pct)/100)
                        is_stopped = False
                        
                        # 风控：D1盘中 or D2开盘
                        if future_df.iloc[0]['low'] <= stop_price: is_stopped = True
                        if not is_stopped and len(future_df) >= 2:
                            if future_df.iloc[1]['open'] <= stop_price: is_stopped = True
                        
                        if is_stopped:
                            if len(future_df) >= 2:
                                sell_price = future_df.iloc[1]['open']
                                ret_d3 = (sell_price / buy_price - 1) * 100
                                ret_d5 = ret_d3 
                                status = "📉 止损(D2开)"
                            else:
                                sell_price = future_df.iloc[0]['close']
                                ret_d3 = (sell_price / buy_price - 1) * 100
                                ret_d5 = ret_d3
                                status = "📉 止损(无法卖)"
                        else:
                            status = "💰 持有"
                            if len(future_df) >= 3:
                                ret_d3 = (future_df.iloc[2]['close'] / buy_price - 1) * 100
                            else:
                                ret_d3 = (future_df.iloc[-1]['close'] / buy_price - 1) * 100
                            
                            if len(future_df) >= 5:
                                ret_d5 = (future_df.iloc[4]['close'] / buy_price - 1) * 100
                            else:
                                ret_d5 = (future_df.iloc[-1]['close'] / buy_price - 1) * 100
            except Exception:
                signal = "❌ 数据Err"
        
        final_results.append({
            'Select_Date': select_date,
            'Trade_Date': buy_date if buy_date else "-",
            'Rank': rank,
            'Code': code,
            'Name': row['Name'],
            'Signal': signal,
            'Open_Pct': open_pct,
            'Vol_Ratio': row['Vol_Ratio'],
            'Score': row['Score'],
            'Ret_D3': ret_d3,
            'Ret_D5': ret_d5,
            'Status': status
        })
        
    return final_results

# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测设置")
    default_date = datetime.now().date()
    end_date = st.date_input("选股截止日期", value=default_date)
    days_back = int(st.number_input("回测天数", value=5))
    
    st.markdown("---")
    st.header("2. 选股 (资金接力)")
    st.info("🎯 核心排序: **换手率** (人气/资金)")
    TOP_N = 3
    
    MIN_MV_YI = st.number_input("最低市值 (亿)", 10, 500, 30, 10)
    
    col_pct1, col_pct2 = st.columns(2)
    with col_pct1: MIN_PCT = st.number_input("涨幅下限%", 0, 20, 6, 1)
    with col_pct2: MAX_PCT = st.number_input("涨幅上限%", 0, 20, 16, 1)
        
    MIN_VOL_RATIO = st.number_input("最低量比", 0.0, 10.0, 1.5, 0.1, help="必须放量")
    
    st.markdown("---")
    st.header("3. 交易 (接力)")
    
    st.caption("🟢 **买入区间**")
    col1, col2 = st.columns(2)
    with col1: BUY_MIN = st.number_input("开盘Min%", -10.0, 10.0, -1.0, 0.5)
    with col2: BUY_MAX = st.number_input("开盘Max%", -10.0, 10.0, 3.0, 0.5)
    
    st.caption("🛡️ **风控**")
    STOP_LOSS = st.number_input("累计跌幅止损%", 1, 20, 5, 1)

    st.markdown("---")
    if st.button("🚨 删库重跑"):
        if os.path.exists(SCORE_DB_FILE): os.remove(SCORE_DB_FILE)
        st.toast("缓存已清空", icon="🗑️")

# ---------------------------
# 主程序
# ---------------------------
col_token, col_btn = st.columns([3, 1])
with col_token:
    TS_TOKEN = st.text_input("🔑 Token", type="password")
with col_btn:
    start_btn = st.button("🚀 启动V41.0 (资金版)", type="primary", use_container_width=True)

if start_btn:
    if not TS_TOKEN: st.stop()
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    
    select_dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    if not select_dates: st.stop()
    
    if not get_all_historical_data(select_dates): st.stop()

    existing_dates = []
    if os.path.exists(SCORE_DB_FILE):
        try:
            df_dates = pd.read_csv(SCORE_DB_FILE, usecols=['Select_Date'])
            existing_dates = df_dates['Select_Date'].astype(str).unique().tolist()
        except: pass
    
    dates_to_compute = [d for d in select_dates if str(d) not in existing_dates]
    
    if dates_to_compute:
        st.write(f"🔄 计算换手率/量比...")
        bar = st.progress(0)
        for i, date in enumerate(dates_to_compute):
            scores = batch_compute_scores(date)
            if scores:
                df_chunk = pd.DataFrame(scores)
                need_header = not os.path.exists(SCORE_DB_FILE)
                df_chunk.to_csv(SCORE_DB_FILE, mode='a', header=need_header, index=False)
            if i % 10 == 0: gc.collect()
            bar.progress((i+1)/len(dates_to_compute))
        bar.empty()
    
    if os.path.exists(SCORE_DB_FILE):
        df_all = pd.read_csv(SCORE_DB_FILE)
        df_all['Select_Date'] = df_all['Select_Date'].astype(str)
        
        final_report = []
        for date in select_dates:
            df_daily = df_all[df_all['Select_Date'] == str(date)]
            if df_daily.empty: continue
            
            res = apply_strategy_and_backtest(
                df_daily, TOP_N, MIN_MV_YI, MIN_PCT, MAX_PCT, MIN_VOL_RATIO, BUY_MIN, BUY_MAX, STOP_LOSS
            )
            if res: final_report.extend(res)
        
        if final_report:
            df_res = pd.DataFrame(final_report)
            trades = df_res[df_res['Signal'].str.contains('BUY', na=False)]
            
            st.markdown(f"### 📊 策略表现 (换手率排序 | 买入[{BUY_MIN}%, {BUY_MAX}%])")
            
            cols = st.columns(3)
            for i, r in enumerate([1, 2, 3]):
                rank_trades = trades[trades['Rank'] == r]
                count = len(rank_trades)
                if count > 0:
                    ret_d3 = rank_trades['Ret_D3'].mean()
                    ret_d5 = rank_trades['Ret_D5'].mean()
                    win_d5 = (rank_trades['Ret_D5'] > 0).mean() * 100
                    color = "red" if ret_d5 > 0 else "green"
                    cols[i].markdown(f"#### 🥇 Rank {r}\n- 交易数: **{count}**\n- D3均收: {ret_d3:.2f}%\n- **D5均收: :{color}[{ret_d5:.2f}%]**\n- D5胜率: {win_d5:.1f}%")
                else:
                    cols[i].markdown(f"#### 🥇 Rank {r}\n- 无交易")

            st.dataframe(df_res, use_container_width=True)
        else:
            st.warning("无符合条件的交易")
