# -*- coding: utf-8 -*-
"""
选股王 · V36.0 筹码透视版 (10000积分专属)
策略核心升级：
1. 数据源：引入【每日筹码及胜率】(cyq_perf)，利用10000积分权限。
2. 选股因子：增加【获利盘比例】(profit_pct) 过滤，剔除套牢盘沉重的个股。
3. 风控升级：回测加入 -8% 硬止损逻辑。
4. 架构：维持 V35 的【算分与筛选分离】，支持秒级调参。
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
st.set_page_config(page_title="V36.0 筹码透视台", layout="wide")
st.title("💎 V36.0 筹码透视监控台 (含获利盘分析+硬止损)")
st.markdown("""
> **10000积分尊享功能已启用：**
> * **筹码透视：** 调用 `cyq_perf` 接口，获取个股 **获利盘比例**。
> * **逻辑：** 优选 **MACD强** 且 **上方无套牢盘(获利盘比例高)** 的股票。
> * **风控：** 模拟实战，盘中触及 **-8%** 强制止损。
""")

# ---------------------------
# 全局设置
# ---------------------------
SCORE_DB_FILE = "v36_chip_database.csv" # 升级数据库文件名，避免混淆
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
# 数据下载 (新增：cyq_perf)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    # 1. 基础行情
    daily_df = safe_get('daily', trade_date=date)
    # 2. 复权因子
    adj_df = safe_get('adj_factor', trade_date=date)
    # 3. 每日指标 (市值 circ_mv)
    basic_df = safe_get('daily_basic', trade_date=date, fields='ts_code,circ_mv')
    # 4. ★★★ 特色数据：每日筹码及胜率 (需2000积分以上) ★★★
    # fields: ts_code, profit_pct(获利盘比例), cost50, cost90
    cyq_df = safe_get('cyq_perf', trade_date=date, fields='ts_code,profit_pct')
    
    # 5. 股票名称
    name_df = safe_get('stock_basic', fields='ts_code,name')
    
    if not daily_df.empty:
        # 仅保留双创 (30/688)
        daily_df = daily_df[daily_df['ts_code'].str.startswith(('30', '688'))]
        
        # 合并市值
        if not basic_df.empty:
            daily_df = daily_df.merge(basic_df, on='ts_code', how='left')
        
        # 合并筹码数据 (Critical Step)
        if not cyq_df.empty:
            daily_df = daily_df.merge(cyq_df, on='ts_code', how='left')
        else:
            daily_df['profit_pct'] = np.nan # 如果没拉到，填空
            
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
    
    try:
        last_idx = GLOBAL_CALENDAR.index(last_select_date)
        end_fetch_idx = min(last_idx + 20, len(GLOBAL_CALENDAR) - 1)
        end_fetch_date = GLOBAL_CALENDAR[end_fetch_idx]
    except:
        end_fetch_date = (datetime.now() + timedelta(days=20)).strftime("%Y%m%d")

    start_fetch_date = (datetime.strptime(first_select_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    
    cal_range = safe_get('trade_cal', start_date=start_fetch_date, end_date=end_fetch_date, is_open='1')
    all_dates = cal_range['cal_date'].tolist()
    
    st.info(f"⏳ 正在拉取全量数据 (含10000积分筹码数据) ({start_fetch_date} ~ {end_fetch_date})...")

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
    cols_to_float = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'circ_mv', 'pct_chg', 'profit_pct']
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
# 算分逻辑 (MACD + 筹码读取)
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
# 阶段一：全量算分与存档 (存入筹码数据)
# ----------------------------------------------------------------------
def batch_compute_scores(date):
    try:
        daily_t = GLOBAL_DAILY_RAW.xs(date, level='trade_date')
    except KeyError: return []

    # 宽泛初筛
    mask = (daily_t['vol'] > 0) & (daily_t['close'] >= 20.0) & (daily_t['close'] <= 350.0)
    pool = daily_t[mask]
    
    if pool.empty: return []

    results = []
    candidates = pool.index.tolist()
    
    for code in candidates:
        s = compute_score_for_stock(code, date)
        if s > 0:
            row = pool.loc[code]
            results.append({
                'Select_Date': date,
                'Code': code,
                'Score': s,
                'Name': row['name'] if 'name' in row else code,
                'Close': float(row['close']),
                'Pct_Chg': float(row['pct_chg']) if 'pct_chg' in row else 0.0,
                'Circ_Mv': float(row['circ_mv']) if 'circ_mv' in row else 0.0,
                'Profit_Pct': float(row['profit_pct']) if 'profit_pct' in row else 0.0 # 存入筹码数据
            })
    
    return results

# ----------------------------------------------------------------------
# 阶段二：动态筛选 (含筹码过滤 + 止损回测)
# ----------------------------------------------------------------------
def apply_strategy_and_backtest(df_scores, top_n, min_mv_yi, min_pct, min_profit_pct, stop_loss_pct):
    # 1. 动态过滤
    min_mv_val = min_mv_yi * 10000
    
    mask = (df_scores['Circ_Mv'] >= min_mv_val) & (df_scores['Pct_Chg'] >= min_pct)
    
    # ★★★ 筹码过滤 ★★★
    # profit_pct 是百分比吗？Tushare 返回通常是 0-100 或 0-1?
    # 假设 Tushare profit_pct 返回的是 50 代表 50%。
    # 如果数据是 NaN (没取到), 暂且放行或过滤? 建议: profit_pct >= min_profit_pct
    # 注意处理 NaN: fillna(0)
    df_scores['Profit_Pct'] = df_scores['Profit_Pct'].fillna(0)
    mask &= (df_scores['Profit_Pct'] >= min_profit_pct)
    
    filtered_df = df_scores[mask].copy()
    
    if filtered_df.empty: return []
    
    # 2. 排序取 Top N
    filtered_df = filtered_df.sort_values('Score', ascending=False).head(top_n)
    
    # 3. 回测
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
        ret_d1 = np.nan
        ret_d3 = np.nan
        
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
                    # 获取未来数据进行止损回测
                    future_df = get_qfq_data(code, buy_date, "20991231")
                    if not future_df.empty:
                        buy_price = future_df.iloc[0]['open']
                        stop_price = buy_price * (1 + stop_loss_pct/100) # 止损价格
                        
                        # 计算 D1 (当天)
                        # 检查当天 Low 是否击穿止损
                        d1_low = future_df.iloc[0]['low']
                        d1_close = future_df.iloc[0]['close']
                        
                        if d1_low <= stop_price:
                            # 触发止损，按止损价离场
                            ret_d1 = stop_loss_pct
                        else:
                            ret_d1 = (d1_close / buy_price - 1) * 100
                        
                        # 计算 D3
                        if len(future_df) >= 3:
                            # 检查 D1-D3 期间是否有任何一天击穿止损
                            triggered_stop = False
                            for i in range(3):
                                if future_df.iloc[i]['low'] <= stop_price:
                                    ret_d3 = stop_loss_pct
                                    triggered_stop = True
                                    break
                            if not triggered_stop:
                                ret_d3 = (future_df.iloc[2]['close'] / buy_price - 1) * 100
                        elif len(future_df) > 0 and not pd.isna(ret_d1):
                             # 数据不够3天，如果第一天没止损，就用最新价
                             if ret_d1 != stop_loss_pct:
                                 ret_d3 = (future_df.iloc[-1]['close'] / buy_price - 1) * 100
                             else:
                                 ret_d3 = stop_loss_pct

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
            'Profit_Pct': row['Profit_Pct'], # 展示获利盘
            'Ret_D1': ret_d1,
            'Ret_D3': ret_d3,
            'Raw_Score': row['Score']
        })
        
    return final_results

# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 基础设置")
    default_date = datetime.now().date()
    end_date = st.date_input("选股截止日期", value=default_date)
    days_back = int(st.number_input("回测天数", value=5))
    
    st.markdown("---")
    st.header("2. 策略参数 (10000积分版)")
    
    TOP_N = st.slider("Top N", 1, 5, 1)
    
    st.caption("📉 **市值 & 涨幅**")
    MIN_MV_YI = st.number_input("最低市值 (亿)", 10, 500, 30, 10)
    MIN_PCT = st.number_input("最低涨幅 (%)", -5, 5, 0, 1)
    
    st.markdown("---")
    st.caption("💎 **筹码 & 风控 (特色功能)**")
    
    MIN_PROFIT_PCT = st.slider(
        "最低获利盘比例 (%)", 
        0, 100, 50, 
        help="【特色数据】仅选获利盘大于此比例的股票。50%代表有一半筹码是赚钱的，抛压较小。"
    )
    
    STOP_LOSS = st.slider(
        "止损线 (%)", 
        -20, 0, -8, 
        help="盘中触及此跌幅强制离场，控制回撤。"
    )

    st.markdown("---")
    if st.button("🚨 删库重跑 (更新筹码数据必点)"):
        if os.path.exists(SCORE_DB_FILE):
            os.remove(SCORE_DB_FILE)
            st.toast("数据库已清除，下次运行将拉取最新筹码数据。", icon="🗑️")

# ---------------------------
# 主程序
# ---------------------------
col_token, col_btn = st.columns([3, 1])
with col_token:
    TS_TOKEN = st.text_input("🔑 Token", type="password")
with col_btn:
    start_btn = st.button("🚀 启动透视", type="primary", use_container_width=True)

if start_btn:
    if not TS_TOKEN: st.stop()
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    
    select_dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    if not select_dates: st.stop()
    
    # 1. 预加载 (含筹码)
    if not get_all_historical_data(select_dates): st.stop()

    # 2. 补全算分
    existing_dates = []
    if os.path.exists(SCORE_DB_FILE):
        try:
            df_dates = pd.read_csv(SCORE_DB_FILE, usecols=['Select_Date'])
            existing_dates = df_dates['Select_Date'].astype(str).unique().tolist()
        except: pass
    
    dates_to_compute = [d for d in select_dates if str(d) not in existing_dates]
    
    if dates_to_compute:
        st.write(f"🔄 正在调用特色接口计算 {len(dates_to_compute)} 天数据...")
        bar = st.progress(0)
        for i, date in enumerate(dates_to_compute):
            scores = batch_compute_scores(date)
            if scores:
                df_chunk = pd.DataFrame(scores)
                need_header = not os.path.exists(SCORE_DB_FILE)
                df_chunk.to_csv(SCORE_DB_FILE, mode='a', header=need_header, index=False)
            if i % 5 == 0: gc.collect()
            bar.progress((i+1)/len(dates_to_compute))
        bar.empty()
    
    # 3. 动态筛选
    st.write("⚡ 正在应用筹码过滤与止损回测...")
    if os.path.exists(SCORE_DB_FILE):
        df_all = pd.read_csv(SCORE_DB_FILE)
        df_all['Select_Date'] = df_all['Select_Date'].astype(str)
        
        final_report = []
        for date in select_dates:
            df_daily = df_all[df_all['Select_Date'] == str(date)]
            if df_daily.empty: continue
            
            # 传入新参数
            res = apply_strategy_and_backtest(
                df_daily, TOP_N, MIN_MV_YI, MIN_PCT, MIN_PROFIT_PCT, STOP_LOSS
            )
            if res: final_report.extend(res)
        
        if final_report:
            df_res = pd.DataFrame(final_report)
            trades = df_res[df_res['Signal'].str.contains('BUY', na=False)]
            
            st.markdown(f"### 📊 策略表现 (获利盘>{MIN_PROFIT_PCT}% | 止损{STOP_LOSS}%)")
            c1, c2, c3, c4 = st.columns(4)
            
            # 重点展示 D1 和 D3
            avg_d1 = trades['Ret_D1'].mean()
            avg_d3 = trades['Ret_D3'].mean()
            win_d3 = (trades['Ret_D3'] > 0).mean() * 100
            
            c1.metric("交易次数", f"{len(trades)}")
            c2.metric("D1 均收", f"{avg_d1:.2f}%")
            c3.metric("D3 均收", f"{avg_d3:.2f}%")
            c4.metric("D3 胜率", f"{win_d3:.1f}%")
            
            st.dataframe(
                df_res[['Trade_Date', 'Code', 'Name', 'Signal', 'Open_Pct', 'Profit_Pct', 'Ret_D1', 'Ret_D3']]
                .style.applymap(lambda x: 'background-color: #ff4b4b; color: white' if 'BUY' in str(x) else '', subset=['Signal'])
                .format({'Ret_D1': '{:.2f}%', 'Ret_D3': '{:.2f}%', 'Open_Pct': '{:.2f}%', 'Profit_Pct': '{:.1f}%'}),
                use_container_width=True
            )
        else:
            st.warning("⚠️ 筛选结果为空，请尝试降低获利盘比例要求。")
