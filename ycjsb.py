# -*- coding: utf-8 -*-
"""
选股王 · V34.0 动态调参版
策略：双创组合 (688 + 300)
更新：
1. 侧边栏支持动态调整【最低流通市值】(10亿起，步长10亿)。
2. 侧边栏支持动态调整【最低涨幅】(-5%起，步长1%)。
3. 旨在通过参数扫描寻找最佳获利区间。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import os

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V34.0 动态调参台", layout="wide")
st.title("🎛️ V34.0 动态调参监控台 (寻找最优解)")

# ---------------------------
# 全局缓存
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_CALENDAR = [] 
CHECKPOINT_FILE = "v34_checkpoint.csv" 

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
    # 获取市值 circ_mv (单位:万元)
    basic_df = safe_get('daily_basic', trade_date=date, fields='ts_code,circ_mv')
    name_df = safe_get('stock_basic', fields='ts_code,name')
    
    if not daily_df.empty:
        daily_df = daily_df[daily_df['ts_code'].str.startswith(('30', '688'))]
        if not basic_df.empty:
            daily_df = daily_df.merge(basic_df, on='ts_code', how='left')
        if not name_df.empty:
            daily_df = daily_df.merge(name_df, on='ts_code', how='left')

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
        end_fetch_idx = min(last_idx + 15, len(GLOBAL_CALENDAR) - 1)
        end_fetch_date = GLOBAL_CALENDAR[end_fetch_idx]
    except:
        end_fetch_date = (datetime.now() + timedelta(days=20)).strftime("%Y%m%d")

    start_fetch_date = (datetime.strptime(first_select_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    
    cal_range = safe_get('trade_cal', start_date=start_fetch_date, end_date=end_fetch_date, is_open='1')
    all_dates = cal_range['cal_date'].tolist()
    
    st.info(f"⏳ 正在预加载数据 ({start_fetch_date} ~ {end_fetch_date})...")

    adj_list, daily_list = [], []
    bar = st.progress(0)
    
    total_steps = len(all_dates)
    for i, date in enumerate(all_dates):
        try:
            cached = fetch_and_cache_daily_data(date)
            if not cached['adj'].empty: adj_list.append(cached['adj'])
            if not cached['daily'].empty: daily_list.append(cached['daily'])
            if i % 10 == 0: bar.progress((i+1)/total_steps)
        except: continue 
    bar.empty()

    if not adj_list or not daily_list: return False
     
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw = pd.concat(daily_list)
    cols_to_float = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'circ_mv', 'pct_chg']
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
    
    df = df.reset_index()
    return df.sort_values('trade_date')

# ----------------------------------------------------------------------
# 评分逻辑
# ----------------------------------------------------------------------
def compute_score(ts_code, current_date):
    start_date = (datetime.strptime(current_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data(ts_code, start_date, current_date)
    
    if df.empty or len(df) < 30: return -1
    
    last_date = df.iloc[-1]['trade_date']
    if hasattr(last_date, 'strftime'):
        last_date_str = last_date.strftime('%Y%m%d')
    else:
        last_date_str = str(last_date)
        
    if last_date_str != current_date: return -1

    close = df['close']
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd_val = (diff - dea) * 2
    
    score = (macd_val.iloc[-1] / close.iloc[-1]) * 100000
    if pd.isna(score): score = -1
    return score

# ----------------------------------------------------------------------
# 核心逻辑 (接收动态参数)
# ----------------------------------------------------------------------
def run_strategy_step(select_date, top_n_limit, min_mv_yi, min_pct):
    try:
        daily_t = GLOBAL_DAILY_RAW.xs(select_date, level='trade_date')
    except KeyError: return []
    
    # ---------------------------
    # 动态漏斗筛选
    # ---------------------------
    # 1. 基础条件
    mask = (daily_t['vol'] > 0)
    # 2. 价格条件 (固定)
    mask &= (daily_t['close'] >= 20.0) & (daily_t['close'] <= 300.0)
    
    # 3. 市值条件 (动态) -> 转换单位：亿 -> 万元
    min_mv_wanyuan = min_mv_yi * 10000 
    if 'circ_mv' in daily_t.columns:
        mask &= (daily_t['circ_mv'] >= min_mv_wanyuan) & (daily_t['circ_mv'] <= 8000000)
    
    # 4. 涨幅条件 (动态)
    if 'pct_chg' in daily_t.columns:
        mask &= (daily_t['pct_chg'] >= min_pct)
        
    pool = daily_t[mask]
    if pool.empty: return []

    # 评分与排名
    candidates = pool.index.tolist()
    scores = []
    
    for code in candidates:
        s = compute_score(code, select_date)
        if s > 0:
            scores.append((code, s))
            
    scores.sort(key=lambda x: x[1], reverse=True)
    final_candidates = scores[:top_n_limit]
    
    # 买入判定
    try:
        t_idx = GLOBAL_CALENDAR.index(select_date)
        if t_idx < len(GLOBAL_CALENDAR) - 1:
            buy_date = GLOBAL_CALENDAR[t_idx + 1]
        else:
            buy_date = None 
    except ValueError: buy_date = None

    results = []
    for rank, (code, score) in enumerate(final_candidates, 1):
        name = pool.loc[code, 'name'] if 'name' in pool.columns else code
        
        signal = "⏳ 等待开盘"
        open_pct = np.nan
        is_buy = False
        ret_d5 = np.nan # 仅展示D5简化
        
        if buy_date:
            try:
                d1_raw = GLOBAL_DAILY_RAW.loc[(code, buy_date)]
                if isinstance(d1_raw, pd.DataFrame): d1_raw = d1_raw.iloc[0]

                daily_buy_open = float(d1_raw['open'])
                daily_buy_pre = float(d1_raw['pre_close'])
                open_pct = (daily_buy_open / daily_buy_pre - 1) * 100
                
                if 2.0 <= open_pct <= 7.5:
                    is_buy = True
                    signal = "✅ BUY"
                elif open_pct < 2.0:
                    signal = "👀 弱"
                else:
                    signal = "⚠️ 强"
                    
                if is_buy:
                    future_df = get_qfq_data(code, buy_date, "20991231")
                    if not future_df.empty and len(future_df) >= 5:
                         buy_price = future_df.iloc[0]['open']
                         ret_d5 = (future_df.iloc[4]['close'] / buy_price - 1) * 100
                    elif not future_df.empty: # 数据不足5天，按最新
                         buy_price = future_df.iloc[0]['open']
                         ret_d5 = (future_df.iloc[-1]['close'] / buy_price - 1) * 100

            except (KeyError, TypeError):
                signal = "❌ 无数据"

        results.append({
            'Select_Date': select_date,
            'Trade_Date': buy_date if buy_date else "未来",
            'Rank': rank,
            'Code': code,
            'Name': name,
            'Signal': signal,
            'Open_Pct': open_pct,
            'Ret_D5': ret_d5
        })

    return results

# ----------------------------------------------------
# 侧边栏 (调参核心区)
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测参数")
    default_date = datetime.now().date()
    end_date = st.date_input("选股截止日期", value=default_date)
    days_back = int(st.number_input("回测天数", value=5))
    
    st.markdown("---")
    st.header("2. 策略核心参数调节")
    
    # 2.1 股票数量
    TOP_N = st.slider("最大持仓 (Top N)", 1, 5, 1, help="只做Rank 1选1，做组合选3-5")
    
    # 2.2 流通市值调节 (10亿起，步长10亿)
    st.caption("📉 **市值过滤**")
    MIN_MV_YI = st.number_input(
        "最低流通市值 (亿)", 
        min_value=10, 
        max_value=500, 
        value=30, 
        step=10,
        help="调小(如10亿)可捕捉微盘妖股，调大(如50亿)增加稳健性。"
    )
    
    # 2.3 涨幅调节 (-5%起，步长1%)
    st.caption("📈 **趋势过滤**")
    MIN_PCT = st.number_input(
        "最低当日涨幅 (%)",
        min_value=-5,
        max_value=5,
        value=0,
        step=1,
        help="设为 0 代表只做红盘；设为 -2 代表允许绿盘潜伏。"
    )

    st.markdown("---")
    # 清空缓存按钮
    if st.button("🗑️ 清空历史缓存 (调参必点)", type="secondary"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            st.toast("缓存已清空，请点击开始扫描使用新参数！", icon="🧹")
        else:
            st.toast("无需清空", icon="ℹ️")

# ---------------------------
# 主界面
# ---------------------------
col_token, col_btn = st.columns([3, 1])
with col_token:
    TS_TOKEN = st.text_input("🔑 Token", type="password", label_visibility="collapsed", placeholder="输入 Tushare Token")

with col_btn:
    start_btn = st.button("🚀 开始扫描", type="primary", use_container_width=True)

st.markdown(f"> **当前参数:** Top {TOP_N} | 市值 > {MIN_MV_YI}亿 | 涨幅 > {MIN_PCT}%")

if start_btn:
    if not TS_TOKEN:
        st.error("请输入 Token")
        st.stop()
        
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    
    select_dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    
    if not select_dates: st.stop()
    
    # 读取断点
    processed_dates = []
    if os.path.exists(CHECKPOINT_FILE):
        try:
            cached_df = pd.read_csv(CHECKPOINT_FILE)
            if 'Select_Date' in cached_df.columns:
                processed_dates = cached_df['Select_Date'].astype(str).unique().tolist()
        except: pass
    
    todo_dates = [d for d in select_dates if str(d) not in processed_dates]
    
    if todo_dates:
        if not get_all_historical_data(select_dates): st.stop()
        
        bar = st.progress(0)
        status = st.empty()
        
        for i, date in enumerate(todo_dates):
            status.text(f"计算中: {date} (市值>{MIN_MV_YI}亿, 涨幅>{MIN_PCT}%)")
            # 传入动态参数
            res_list = run_strategy_step(date, TOP_N, MIN_MV_YI, MIN_PCT)
            
            if res_list:
                df_chunk = pd.DataFrame(res_list)
                need_header = not os.path.exists(CHECKPOINT_FILE)
                df_chunk.to_csv(CHECKPOINT_FILE, mode='a', header=need_header, index=False)
            
            bar.progress((i+1)/len(todo_dates))
        
        bar.empty()
        status.empty()

    # 展示结果
    if os.path.exists(CHECKPOINT_FILE):
        full_df = pd.read_csv(CHECKPOINT_FILE)
        mask = full_df['Select_Date'].astype(str).isin([str(d) for d in select_dates])
        df_show = full_df[mask].copy()
        
        # 再次按 Top N 过滤显示
        df_show = df_show[df_show['Rank'] <= TOP_N]
        
        if not df_show.empty:
            trades = df_show[df_show['Signal'].str.contains('BUY', na=False)]
            
            st.markdown(f"### 📊 策略表现 (Top {TOP_N})")
            c1, c2, c3, c4 = st.columns(4)
            
            avg_ret = trades['Ret_D5'].mean()
            win_rate = (trades['Ret_D5'] > 0).mean() * 100
            
            c1.metric("入围", f"{len(df_show)}")
            c2.metric("交易", f"{len(trades)}")
            c3.metric("收益 (D5)", f"{avg_ret:.2f}%")
            c4.metric("胜率 (D5)", f"{win_rate:.1f}%")

            st.dataframe(
                df_show[['Trade_Date', 'Code', 'Name', 'Signal', 'Open_Pct', 'Rank', 'Ret_D5']]
                .style.applymap(lambda x: 'background-color: #ff4b4b; color: white' if 'BUY' in str(x) else '', subset=['Signal'])
                .format({'Ret_D5': '{:.2f}%', 'Open_Pct': '{:.2f}%'}),
                use_container_width=True
            )
            
            csv = df_show.to_csv().encode('utf-8')
            st.download_button("📥 下载结果", csv, "v34_param_test.csv", "text/csv")
