# -*- coding: utf-8 -*-
"""
选股王 · V46.0 RSRS趋势锁定版 (吃鱼身战法)
核心逻辑：
1. 战略目标：右侧交易，买在趋势起步或1/3处，拒绝山顶。
2. 核心指标：
   - RSRS斜率 (Slope)：计算高低点的回归斜率，确认处于上升通道。
   - 乖离率 (Bias)：股价距离20日线 < 15%，防止买在山顶。
   - 黄金通道：换手[18,26] + 量比[1.5,3.5] (动力保障)。
3. 买入战术：红盘确认 [0%, 4%]。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import os
import gc
import time
from scipy.stats import linregress # 引入科学计算库做回归分析

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V46.0 RSRS趋势战法", layout="wide")
st.title("🚀 V46.0 RSRS趋势监控 (吃鱼身战法)")

# ---------------------------
# 全局设置
# ---------------------------
SCORE_DB_FILE = "v46_rsrs_trend_db.csv" # 新数据库
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
    for attempt in range(3):
        try:
            df = func(**kwargs)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return pd.DataFrame(columns=['ts_code']) 
            return df
        except Exception:
            if attempt < 2: time.sleep(1); continue
            else: return pd.DataFrame(columns=['ts_code'])

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
    try:
        daily_df = safe_get('daily', trade_date=date)
        adj_df = safe_get('adj_factor', trade_date=date)
        basic_df = safe_get('daily_basic', trade_date=date, fields='ts_code,circ_mv,turnover_rate,volume_ratio')
        name_df = safe_get('stock_basic', fields='ts_code,name')
        
        if not daily_df.empty:
            daily_df = daily_df[daily_df['ts_code'].str.startswith(('30', '688'))]
            if not basic_df.empty: daily_df = daily_df.merge(basic_df, on='ts_code', how='left')
            if not name_df.empty: daily_df = daily_df.merge(name_df, on='ts_code', how='left')

        if not adj_df.empty:
            adj_df = adj_df[adj_df['ts_code'].str.startswith(('30', '688'))]
            
        return {'adj': adj_df, 'daily': daily_df}
    except: return {'adj': pd.DataFrame(), 'daily': pd.DataFrame()}

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
# 核心算法：RSRS 趋势 + 乖离率检查
# ----------------------------------------------------------------------
def analyze_rsrs_trend(ts_code, current_date, max_bias_pct):
    """
    1. 计算 RSRS 斜率：确认处于上升通道 (右侧)。
    2. 计算 乖离率 (Bias)：股价距离20日线不能太远 (拒绝山顶)。
    """
    try:
        # 取18天数据做回归 (经典参数)
        start_date = (datetime.strptime(current_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
        df = get_qfq_data(ts_code, start_date, current_date)
        if df.empty or len(df) < 20: return None
        
        last_row = df.iloc[-1]
        if last_row['trade_date'].strftime('%Y%m%d') != current_date: return None

        close = df['close']
        high = df['high']
        low = df['low']
        
        # 1. 计算 20日均线 & 乖离率 (Bias)
        ma20 = close.rolling(window=20).mean()
        current_ma20 = ma20.iloc[-1]
        current_close = close.iloc[-1]
        
        # Bias = (收盘 - 均线) / 均线
        bias_pct = ((current_close - current_ma20) / current_ma20) * 100
        
        # 核心过滤：如果乖离率太大(>15%)，说明在山顶，不要买
        if bias_pct > max_bias_pct: 
            return None
        
        # 核心过滤：必须站上20日线 (右侧基础)
        if current_close < current_ma20:
            return None

        # 2. 计算 RSRS 斜率 (简化版：只看 High 的斜率，代表阻力位的变化)
        # 取最近 18 天
        recent_df = df.iloc[-18:]
        x = np.arange(len(recent_df))
        y_high = recent_df['high'].values
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = linregress(x, y_high)
        
        # 条件：斜率必须 > 0 (上升通道) 且 R方 > 0.3 (趋势比较明显)
        # R方在这里不是必须的，只要斜率正即可，放宽一点
        if slope > 0:
            return slope # 返回斜率作为强度分数
            
        return None
    except Exception: return None

def batch_compute_scores(date, max_bias):
    try:
        try:
            daily_t = GLOBAL_DAILY_RAW.xs(date, level='trade_date')
        except KeyError: return []

        mask = (daily_t['vol'] > 0) & (daily_t['close'] >= 2.0)
        pool = daily_t[mask]
        if pool.empty: return []

        results = []
        candidates = pool.index.tolist()
        
        for code in candidates:
            # 传入 max_bias 参数
            rsrs_slope = analyze_rsrs_trend(code, date, max_bias)
            
            if rsrs_slope is not None:
                row = pool.loc[code]
                turnover = float(row['turnover_rate']) if 'turnover_rate' in row else 0.0
                vol_ratio = float(row['volume_ratio']) if 'volume_ratio' in row else 0.0
                
                # 依然按换手率排序，因为这是短线活力的根本
                # RSRS 只是门槛 (Pass/Fail)
                score = turnover 
                
                results.append({
                    'Select_Date': date,
                    'Code': code,
                    'Score': score,
                    'Name': row['name'] if 'name' in row else code,
                    'Close': float(row['close']),
                    'Pct_Chg': float(row['pct_chg']) if 'pct_chg' in row else 0.0,
                    'Circ_Mv': float(row['circ_mv']) if 'circ_mv' in row else 0.0,
                    'Turnover': turnover,
                    'Vol_Ratio': vol_ratio,
                    'RSRS_Slope': rsrs_slope
                })
        return results
    except Exception: return []

# ----------------------------------------------------------------------
# 动态筛选与回测
# ----------------------------------------------------------------------
def apply_strategy_and_backtest(df_scores, top_n, min_mv_yi, min_pct, max_pct, min_turnover, max_turnover, min_vol, max_vol, buy_open_min, buy_open_max, stop_loss_pct):
    min_mv_val = min_mv_yi * 10000
    
    # 黄金通道过滤
    mask = (df_scores['Circ_Mv'] >= min_mv_val) & \
           (df_scores['Pct_Chg'] >= min_pct) & \
           (df_scores['Pct_Chg'] <= max_pct) & \
           (df_scores['Turnover'] >= min_turnover) & \
           (df_scores['Turnover'] <= max_turnover) & \
           (df_scores['Vol_Ratio'] >= min_vol) & \
           (df_scores['Vol_Ratio'] <= max_vol)

    filtered_df = df_scores[mask].copy()
    if filtered_df.empty: return []
    
    # 排序：在符合RSRS趋势的股票里，按换手率降序
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
            'Turnover': row['Turnover'],
            'RSRS_Slope': row['RSRS_Slope'],
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
    st.header("2. 选股 (RSRS趋势 + 黄金通道)")
    st.info("📈 **RSRS斜率: 必须向上**")
    
    TOP_N = 3
    MIN_MV_YI = st.number_input("最低市值 (亿)", 10, 500, 30, 10)
    MAX_BIAS = st.number_input("乖离率上限% (拒接高位)", 5, 50, 15, 1, help="价格距离20日线太远视为山顶")
    
    col_pct1, col_pct2 = st.columns(2)
    with col_pct1: MIN_PCT = st.number_input("涨幅下限%", 0, 20, 6, 1)
    with col_pct2: MAX_PCT = st.number_input("涨幅上限%", 0, 20, 16, 1)
        
    st.caption("🔥 **黄金通道**")
    col_t1, col_t2 = st.columns(2)
    with col_t1: MIN_TURNOVER = st.number_input("换手Min%", 0.0, 50.0, 18.0, 1.0)
    with col_t2: MAX_TURNOVER = st.number_input("换手Max%", 0.0, 50.0, 26.0, 1.0)
    
    col_v1, col_v2 = st.columns(2)
    with col_v1: MIN_VOL = st.number_input("量比Min", 0.0, 10.0, 1.5, 0.1)
    with col_v2: MAX_VOL = st.number_input("量比Max", 0.0, 10.0, 3.5, 0.1)

    st.markdown("---")
    st.header("3. 交易 (红盘确认)")
    
    col1, col2 = st.columns(2)
    with col1: BUY_MIN = st.number_input("开盘Min%", -10.0, 10.0, 0.0, 0.5)
    with col2: BUY_MAX = st.number_input("开盘Max%", -10.0, 10.0, 4.0, 0.5)
    
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
    start_btn = st.button("🚀 启动V46.0 (RSRS版)", type="primary", use_container_width=True)

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
            st.success(f"📂 跳过 {len(existing_dates)} 天")
        except: pass
    
    dates_to_compute = [d for d in select_dates if str(d) not in existing_dates]
    
    if dates_to_compute:
        st.write(f"🔄 补全数据...")
        bar = st.progress(0)
        for i, date in enumerate(dates_to_compute):
            # 传入 MAX_BIAS 参数
            scores = batch_compute_scores(date, MAX_BIAS)
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
                df_daily, TOP_N, MIN_MV_YI, MIN_PCT, MAX_PCT, MIN_TURNOVER, MAX_TURNOVER, MIN_VOL, MAX_VOL, BUY_MIN, BUY_MAX, STOP_LOSS
            )
            if res: final_report.extend(res)
        
        if final_report:
            df_res = pd.DataFrame(final_report)
            trades = df_res[df_res['Signal'].str.contains('BUY', na=False)]
            
            st.markdown(f"### 📊 策略表现 (RSRS趋势 | 拒接高位>{MAX_BIAS}%)")
            
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
