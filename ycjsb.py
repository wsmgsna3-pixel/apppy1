# -*- coding: utf-8 -*-
"""
选股王 · V35.0 极速调参版 (算分与筛选分离)
解决痛点：
1. 调整参数(市值/涨幅)无需清除缓存，秒级响应。
2. 即使崩溃，重启后可直接利用已算好的分数数据，不浪费时间。
3. 架构：先建立【全量分数库】，再进行【动态策略筛选】。
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
st.set_page_config(page_title="V35.0 极速调参台", layout="wide")
st.title("⚡ V35.0 极速调参监控台 (一次算分，无限调参)")

# ---------------------------
# 全局设置与文件路径
# ---------------------------
SCORE_DB_FILE = "v35_score_database.csv" # 存放全量算分数据的"大数据库"
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
    # 基础行情
    daily_df = safe_get('daily', trade_date=date)
    # 复权因子
    adj_df = safe_get('adj_factor', trade_date=date)
    # 每日指标 (市值 circ_mv, 换手率等)
    basic_df = safe_get('daily_basic', trade_date=date, fields='ts_code,circ_mv')
    # 股票名称
    name_df = safe_get('stock_basic', fields='ts_code,name')
    
    if not daily_df.empty:
        # 仅保留双创 (30/688)
        daily_df = daily_df[daily_df['ts_code'].str.startswith(('30', '688'))]
        
        # 合并市值 (注意: circ_mv 单位是万元)
        if not basic_df.empty:
            daily_df = daily_df.merge(basic_df, on='ts_code', how='left')
        
        # 合并名称
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
    
    # 预加载数据范围：最早选股日前150天 ~ 最晚选股日后20天
    try:
        last_idx = GLOBAL_CALENDAR.index(last_select_date)
        end_fetch_idx = min(last_idx + 20, len(GLOBAL_CALENDAR) - 1)
        end_fetch_date = GLOBAL_CALENDAR[end_fetch_idx]
    except:
        end_fetch_date = (datetime.now() + timedelta(days=20)).strftime("%Y%m%d")

    start_fetch_date = (datetime.strptime(first_select_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    
    # 获取日期列表
    cal_range = safe_get('trade_cal', start_date=start_fetch_date, end_date=end_fetch_date, is_open='1')
    all_dates = cal_range['cal_date'].tolist()
    
    st.info(f"⏳ 正在预加载全量行情数据 ({start_fetch_date} ~ {end_fetch_date})...")

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
     
    # 合并数据
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw = pd.concat(daily_list)
    cols_to_float = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'circ_mv', 'pct_chg']
    for col in cols_to_float:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    
    # 获取最新复权因子
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
# 算分逻辑 (计算 MACD)
# ----------------------------------------------------------------------
def compute_score_for_stock(ts_code, current_date):
    start_date = (datetime.strptime(current_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data(ts_code, start_date, current_date)
    
    if df.empty or len(df) < 30: return -1
    
    last_date = df.iloc[-1]['trade_date']
    last_date_str = last_date.strftime('%Y%m%d') if hasattr(last_date, 'strftime') else str(last_date)
    
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
# 阶段一：全量算分与存档 (这是"苦力活"，只做一次)
# ----------------------------------------------------------------------
def batch_compute_scores(date):
    try:
        daily_t = GLOBAL_DAILY_RAW.xs(date, level='trade_date')
    except KeyError: return []

    # 1. 宽泛初筛 (只做最基本的过滤，保留尽可能多的数据以便后续调参)
    # 只要有成交量且价格>20即可，暂不卡市值和涨幅
    mask = (daily_t['vol'] > 0) & (daily_t['close'] >= 20.0) & (daily_t['close'] <= 350.0)
    pool = daily_t[mask]
    
    if pool.empty: return []

    results = []
    candidates = pool.index.tolist()
    
    # 遍历计算 MACD
    for code in candidates:
        s = compute_score_for_stock(code, date)
        if s > 0:
            # 将后续调参需要用到的因子全部存下来
            row = pool.loc[code]
            results.append({
                'Select_Date': date,
                'Code': code,
                'Score': s,
                'Name': row['name'] if 'name' in row else code,
                'Close': float(row['close']),
                'Pct_Chg': float(row['pct_chg']) if 'pct_chg' in row else 0.0,
                'Circ_Mv': float(row['circ_mv']) if 'circ_mv' in row else 0.0 # 单位: 万元
            })
    
    return results

# ----------------------------------------------------------------------
# 阶段二：动态筛选与回测 (这是"指挥官"，秒级响应)
# ----------------------------------------------------------------------
def apply_strategy_and_backtest(df_scores, top_n, min_mv_yi, min_pct):
    # df_scores 已经是某一日期的所有备选股数据了
    
    # 1. 动态过滤
    # 市值过滤: 输入是亿，数据是万元 -> min_mv_yi * 10000
    min_mv_val = min_mv_yi * 10000
    mask = (df_scores['Circ_Mv'] >= min_mv_val) & (df_scores['Pct_Chg'] >= min_pct)
    
    filtered_df = df_scores[mask].copy()
    
    if filtered_df.empty: return []
    
    # 2. 排序取 Top N
    filtered_df = filtered_df.sort_values('Score', ascending=False).head(top_n)
    
    # 3. 回测 T+1 表现 (需要查 GLOBAL_DAILY_RAW)
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
        signal = "⏳ 等待开盘"
        open_pct = np.nan
        is_buy = False
        ret_d5 = np.nan
        
        if buy_date:
            try:
                # 从内存中的行情数据查 T+1 开盘
                d1_raw = GLOBAL_DAILY_RAW.loc[(code, buy_date)]
                if isinstance(d1_raw, pd.DataFrame): d1_raw = d1_raw.iloc[0]

                daily_buy_open = float(d1_raw['open'])
                daily_buy_pre = float(d1_raw['pre_close'])
                open_pct = (daily_buy_open / daily_buy_pre - 1) * 100
                
                # 买入逻辑 (固定不变)
                if 2.0 <= open_pct <= 7.5:
                    is_buy = True
                    signal = "✅ BUY"
                elif open_pct < 2.0:
                    signal = "👀 弱"
                else:
                    signal = "⚠️ 强"
                
                # 计算 D5 收益
                if is_buy:
                     future_df = get_qfq_data(code, buy_date, "20991231")
                     if not future_df.empty:
                         buy_price = future_df.iloc[0]['open']
                         # 尝试取 D5，不够就取最新的
                         idx_sell = 4 if len(future_df) >= 5 else -1
                         ret_d5 = (future_df.iloc[idx_sell]['close'] / buy_price - 1) * 100

            except:
                signal = "❌ 无数据"
        
        final_results.append({
            'Select_Date': select_date,
            'Trade_Date': buy_date if buy_date else "-",
            'Rank': rank,
            'Code': code,
            'Name': row['Name'],
            'Signal': signal,
            'Open_Pct': open_pct,
            'Ret_D5': ret_d5,
            'Raw_Score': row['Score']
        })
        
    return final_results

# ----------------------------------------------------
# 侧边栏 (参数调节)
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 基础设置")
    default_date = datetime.now().date()
    end_date = st.date_input("选股截止日期", value=default_date)
    days_back = int(st.number_input("回测天数", value=5))
    
    st.markdown("---")
    st.header("2. 动态调参 (实时生效)")
    
    TOP_N = st.slider("Top N", 1, 5, 1)
    
    MIN_MV_YI = st.number_input(
        "最低流通市值 (亿)", 
        min_value=10, max_value=500, value=30, step=10
    )
    
    MIN_PCT = st.number_input(
        "最低当日涨幅 (%)",
        min_value=-5, max_value=5, value=0, step=1
    )

    st.markdown("---")
    st.info("💡 说明: 第一次运行需要算分(较慢)。算分完成后，调整上方参数无需清空缓存，结果秒出。")
    
    # 仅当想彻底重置所有算分数据时才用
    if st.button("🚨 彻底清空所有数据 (慎点)"):
        if os.path.exists(SCORE_DB_FILE):
            os.remove(SCORE_DB_FILE)
            st.toast("数据库已删除，下次运行将重新全量算分。", icon="🗑️")

# ---------------------------
# 主程序
# ---------------------------
col_token, col_btn = st.columns([3, 1])
with col_token:
    TS_TOKEN = st.text_input("🔑 Token", type="password", placeholder="输入 Tushare Token")
with col_btn:
    start_btn = st.button("🚀 执行策略", type="primary", use_container_width=True)

if start_btn:
    if not TS_TOKEN: st.stop()
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    
    # 1. 获取日期
    select_dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    if not select_dates: st.stop()
    
    # 2. 预加载行情数据 (必须步骤，用于算分和查T+1)
    if not get_all_historical_data(select_dates): st.stop()

    # 3. 检查/建立分数组库 (SCORE_DB_FILE)
    #    逻辑: 看看 SCORE_DB_FILE 里有哪些日期已经算过了
    existing_dates = []
    if os.path.exists(SCORE_DB_FILE):
        try:
            # 只读 Select_Date 列，加快速度
            df_dates = pd.read_csv(SCORE_DB_FILE, usecols=['Select_Date'])
            existing_dates = df_dates['Select_Date'].astype(str).unique().tolist()
        except: pass
    
    # 找出哪些日期还没算分
    dates_to_compute = [d for d in select_dates if str(d) not in existing_dates]
    
    # 3.1 补全缺失日期的算分 (苦力活)
    if dates_to_compute:
        st.write(f"🔄 发现 {len(dates_to_compute)} 个新日期需要算分，正在处理... (完成后将永久缓存)")
        bar = st.progress(0)
        
        for i, date in enumerate(dates_to_compute):
            scores = batch_compute_scores(date)
            if scores:
                df_chunk = pd.DataFrame(scores)
                # 追加模式写入 CSV
                need_header = not os.path.exists(SCORE_DB_FILE)
                df_chunk.to_csv(SCORE_DB_FILE, mode='a', header=need_header, index=False)
            
            # 内存清理，防崩溃
            if i % 5 == 0: gc.collect()
            bar.progress((i+1)/len(dates_to_compute))
        bar.empty()
    
    # 4. 核心：读取全量分数库，进行动态筛选 (极速)
    st.write("⚡ 正在应用策略参数进行极速回测...")
    
    # 一次性读取所需日期的数据
    if os.path.exists(SCORE_DB_FILE):
        df_all_scores = pd.read_csv(SCORE_DB_FILE)
        # 转换日期格式以便筛选
        df_all_scores['Select_Date'] = df_all_scores['Select_Date'].astype(str)
        
        final_report = []
        
        for date in select_dates:
            # 从内存中切片
            df_daily = df_all_scores[df_all_scores['Select_Date'] == str(date)]
            if df_daily.empty: continue
            
            # 传入参数进行筛选
            res = apply_strategy_and_backtest(df_daily, TOP_N, MIN_MV_YI, MIN_PCT)
            if res:
                final_report.extend(res)
        
        # 5. 展示结果
        if final_report:
            df_res = pd.DataFrame(final_report)
            
            # Dashboard
            trades = df_res[df_res['Signal'].str.contains('BUY', na=False)]
            
            st.markdown(f"### 📊 策略表现 (Top {TOP_N} | 市值>{MIN_MV_YI}亿 | 涨幅>{MIN_PCT}%)")
            c1, c2, c3, c4 = st.columns(4)
            
            avg_ret = trades['Ret_D5'].mean()
            win_rate = (trades['Ret_D5'] > 0).mean() * 100
            
            c1.metric("总入围", f"{len(df_res)}")
            c2.metric("交易次数", f"{len(trades)}")
            c3.metric("D5 均收", f"{avg_ret:.2f}%")
            c4.metric("D5 胜率", f"{win_rate:.1f}%")
            
            # 样式
            st.dataframe(
                df_res[['Trade_Date', 'Code', 'Name', 'Signal', 'Open_Pct', 'Rank', 'Ret_D5', 'Raw_Score']]
                .style.applymap(lambda x: 'background-color: #ff4b4b; color: white' if 'BUY' in str(x) else '', subset=['Signal'])
                .format({'Ret_D5': '{:.2f}%', 'Open_Pct': '{:.2f}%', 'Raw_Score': '{:.0f}'}),
                use_container_width=True
            )
        else:
            st.warning("⚠️ 当前筛选条件下无符合结果，请尝试放宽参数。")
            
    else:
        st.error("❌ 数据库创建失败，请检查 Tushare 权限。")
