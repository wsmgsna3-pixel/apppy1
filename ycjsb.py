# -*- coding: utf-8 -*-
"""
选股王 · V46.2 RSRS趋势锁定版 (Numpy稳定内核 + 断点续传)
逻辑架构 (已对齐):
1. 【大势确认】RSRS斜率 > 0 (Numpy计算)。确保处于右侧上升通道。
2. 【位置安全】乖离率 < 15%。拒绝山顶，只吃鱼身。
3. 【动力诊断】换手[18,26] + 量比[1.5,3.5]。剔除死鱼与炸弹。
4. 【交易确认】开盘[0, 4]。主力红盘表态。
5. 【工程保障】本地数据仓库技术，崩溃后重启可秒续传。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import os
import gc
import time
import pickle # 用于数据仓库

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V46.2 吃鱼身战法", layout="wide")
st.title("🚀 V46.2 RSRS趋势监控 (Numpy稳定版)")

# ---------------------------
# 全局设置
# ---------------------------
SCORE_DB_FILE = "v46_rsrs_trend_db.csv"   # 结果数据库
CACHE_DIR = "daily_data_cache"            # 本地行情仓库

# 自动创建缓存目录
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

pro = None 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_ADJ_FACTOR = pd.DataFrame()
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
    # 多取60天用于RSRS预热计算
    start_search = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=max(num_days * 5, 150))).strftime("%Y%m%d")
    end_search = (datetime.strptime(end_date_str, "%Y%m%d") + timedelta(days=60)).strftime("%Y%m%d")
    
    cal = safe_get('trade_cal', start_date=start_search, end_date=end_search)
    if cal.empty or 'is_open' not in cal.columns: return []
    
    global GLOBAL_CALENDAR
    open_cal = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=True)
    GLOBAL_CALENDAR = open_cal['cal_date'].tolist()
    
    past_days = open_cal[open_cal['cal_date'] <= end_date_str]['cal_date'].tolist()
    return past_days[-(num_days + 60):] # 返回包含预热期的时间段

# ----------------------------------------------------------------------
# 数据下载引擎 (断点续传核心)
# ----------------------------------------------------------------------
def fetch_single_day_data(date):
    """下载单日数据"""
    try:
        daily_df = safe_get('daily', trade_date=date)
        adj_df = safe_get('adj_factor', trade_date=date)
        # 获取换手率和量比
        basic_df = safe_get('daily_basic', trade_date=date, fields='ts_code,circ_mv,turnover_rate,volume_ratio')
        name_df = safe_get('stock_basic', fields='ts_code,name')
        
        if not daily_df.empty:
            daily_df = daily_df[daily_df['ts_code'].str.startswith(('30', '688'))]
            if not basic_df.empty: daily_df = daily_df.merge(basic_df, on='ts_code', how='left')
            if not name_df.empty: daily_df = daily_df.merge(name_df, on='ts_code', how='left')
        
        if not adj_df.empty:
            adj_df = adj_df[adj_df['ts_code'].str.startswith(('30', '688'))]

        return {'adj': adj_df, 'daily': daily_df}
    except:
        return None

def load_or_fetch_data(date_list):
    """
    智能加载：优先读本地硬盘缓存，没有才下载并存盘。
    """
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    
    adj_list = []
    daily_list = []
    
    st.info(f"📂 正在准备数据 ({len(date_list)} 天) - 支持断点续传...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, date in enumerate(date_list):
        cache_path = os.path.join(CACHE_DIR, f"{date}.pkl")
        data_packet = None
        
        # 1. 读本地
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data_packet = pickle.load(f)
            except: pass
        
        # 2. 下云端 (并存本地)
        if data_packet is None:
            status_text.text(f"📥 下载并缓存: {date}")
            data_packet = fetch_single_day_data(date)
            if data_packet:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data_packet, f)
            else:
                continue # 下载失败跳过
        
        # 3. 汇总
        if not data_packet['adj'].empty: adj_list.append(data_packet['adj'])
        if not data_packet['daily'].empty: daily_list.append(data_packet['daily'])
        
        if i % 10 == 0: progress_bar.progress((i + 1) / len(date_list))
    
    progress_bar.empty()
    status_text.text("✅ 数据准备完毕，正在构建复权因子...")
    
    if not adj_list or not daily_list: return False
    
    # 合并
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw = pd.concat(daily_list)
    cols_to_float = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'circ_mv', 'pct_chg', 'turnover_rate', 'volume_ratio']
    for col in cols_to_float:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    
    # 基准因子
    latest_date_in_data = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_date_in_data:
        GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_date_in_data), 'adj_factor'].droplevel(1).to_dict()
        
    status_text.empty()
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
# 核心算法：RSRS + 乖离率 (Numpy原生版)
# ----------------------------------------------------------------------
def analyze_rsrs_trend_numpy(ts_code, current_date, max_bias_pct):
    try:
        # 取60天数据
        start_date = (datetime.strptime(current_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
        df = get_qfq_data(ts_code, start_date, current_date)
        if df.empty or len(df) < 20: return None
        
        last_row = df.iloc[-1]
        if last_row['trade_date'].strftime('%Y%m%d') != current_date: return None

        close = df['close']
        
        # 1. 位置安全检查 (乖离率)
        ma20 = close.rolling(window=20).mean()
        current_ma20 = ma20.iloc[-1]
        current_close = close.iloc[-1]
        
        if pd.isna(current_ma20) or current_ma20 == 0: return None
        
        # Bias = (收盘 - 20日线) / 20日线
        bias_pct = ((current_close - current_ma20) / current_ma20) * 100
        
        # 逻辑：拒绝山顶
        if bias_pct > max_bias_pct: return None
        # 逻辑：必须在生命线上方 (右侧基础)
        if current_close < current_ma20: return None

        # 2. RSRS 趋势确认 (Numpy Polyfit)
        # 用最近18天的 High 数据拟合斜率
        recent_df = df.iloc[-18:]
        if len(recent_df) < 18: return None
        
        y_high = recent_df['high'].values
        x = np.arange(len(y_high))
        
        # 核心数学：一元线性回归 (deg=1)
        # 效果等同于 scipy.stats.linregress
        slope, intercept = np.polyfit(x, y_high, 1)
        
        # 逻辑：斜率必须向上 (趋势起步)
        if slope > 0:
            return slope * 10 # 放大系数，方便显示
            
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
            # 这里的 RSRS_Slope 是通过 Numpy 算出来的
            rsrs_slope = analyze_rsrs_trend_numpy(code, date, max_bias)
            
            if rsrs_slope is not None:
                row = pool.loc[code]
                turnover = float(row['turnover_rate']) if 'turnover_rate' in row else 0.0
                vol_ratio = float(row['volume_ratio']) if 'volume_ratio' in row else 0.0
                
                # 双黄金通道核心：RSRS确认趋势后，依然按[换手率]排序，寻找活跃资金
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
# 回测执行
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
    
    # 排序：按换手率降序 (人气优先)
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
                
                # 交易确认：红盘买入
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
                        
                        # 止损逻辑
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
    st.header("2. RSRS战法参数")
    st.info("📈 **趋势核心: 斜率>0 + 拒绝山顶**")
    
    TOP_N = 3
    MIN_MV_YI = st.number_input("最低市值 (亿)", 10, 500, 30, 10)
    MAX_BIAS = st.number_input("乖离率上限% (山顶线)", 5, 50, 15, 1, help="高于此值视为鱼尾，不接")
    
    col_pct1, col_pct2 = st.columns(2)
    with col_pct1: MIN_PCT = st.number_input("涨幅下限%", 0, 20, 6, 1)
    with col_pct2: MAX_PCT = st.number_input("涨幅上限%", 0, 20, 16, 1)
        
    st.caption("🔥 **双黄金通道**")
    col_t1, col_t2 = st.columns(2)
    with col_t1: MIN_TURNOVER = st.number_input("换手Min%", 0.0, 50.0, 18.0, 1.0)
    with col_t2: MAX_TURNOVER = st.number_input("换手Max%", 0.0, 50.0, 26.0, 1.0)
    
    col_v1, col_v2 = st.columns(2)
    with col_v1: MIN_VOL = st.number_input("量比Min", 0.0, 10.0, 1.5, 0.1)
    with col_v2: MAX_VOL = st.number_input("量比Max", 0.0, 10.0, 3.5, 0.1)

    st.markdown("---")
    st.header("3. 交易执行")
    
    col1, col2 = st.columns(2)
    with col1: BUY_MIN = st.number_input("开盘Min%", -10.0, 10.0, 0.0, 0.5)
    with col2: BUY_MAX = st.number_input("开盘Max%", -10.0, 10.0, 4.0, 0.5)
    
    STOP_LOSS = st.number_input("累计跌幅止损%", 1, 20, 5, 1)

    st.markdown("---")
    col_del1, col_del2 = st.columns(2)
    with col_del1:
        if st.button("🗑️ 清空结果"):
            if os.path.exists(SCORE_DB_FILE): os.remove(SCORE_DB_FILE)
            st.toast("已清除计算结果", icon="🧹")
    with col_del2:
        if st.button("💣 清空缓存"):
            import shutil
            if os.path.exists(CACHE_DIR): shutil.rmtree(CACHE_DIR)
            st.toast("本地数据已删除", icon="💥")

# ---------------------------
# 主程序
# ---------------------------
col_token, col_btn = st.columns([3, 1])
with col_token:
    TS_TOKEN = st.text_input("🔑 Token", type="password")
with col_btn:
    start_btn = st.button("🚀 启动V46.2", type="primary", use_container_width=True)

if start_btn:
    if not TS_TOKEN: st.stop()
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    
    # 1. 计算日期
    target_dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    if not target_dates: st.stop()
    
    # 2. 智能下载 (带断点保护)
    if not load_or_fetch_data(target_dates): st.stop()
    
    # 3. 智能计算 (带结果去重)
    existing_dates = []
    if os.path.exists(SCORE_DB_FILE):
        try:
            df_dates = pd.read_csv(SCORE_DB_FILE, usecols=['Select_Date'])
            existing_dates = df_dates['Select_Date'].astype(str).unique().tolist()
        except: pass
    
    # 只计算真正回测的那几天
    backtest_dates = target_dates[-days_back:]
    dates_to_compute = [d for d in backtest_dates if str(d) not in existing_dates]
    
    if dates_to_compute:
        st.write(f"🔄 正在计算 ({len(dates_to_compute)}天)...")
        bar = st.progress(0)
        for i, date in enumerate(dates_to_compute):
            # 核心调用
            scores = batch_compute_scores(date, MAX_BIAS)
            if scores:
                df_chunk = pd.DataFrame(scores)
                need_header = not os.path.exists(SCORE_DB_FILE)
                df_chunk.to_csv(SCORE_DB_FILE, mode='a', header=need_header, index=False)
            if i % 10 == 0: gc.collect()
            bar.progress((i+1)/len(dates_to_compute))
        bar.empty()
    
    # 4. 生成报表
    if os.path.exists(SCORE_DB_FILE):
        df_all = pd.read_csv(SCORE_DB_FILE)
        df_all['Select_Date'] = df_all['Select_Date'].astype(str)
        
        final_report = []
        for date in backtest_dates:
            df_daily = df_all[df_all['Select_Date'] == str(date)]
            if df_daily.empty: continue
            
            res = apply_strategy_and_backtest(
                df_daily, TOP_N, MIN_MV_YI, MIN_PCT, MAX_PCT, MIN_TURNOVER, MAX_TURNOVER, MIN_VOL, MAX_VOL, BUY_MIN, BUY_MAX, STOP_LOSS
            )
            if res: final_report.extend(res)
        
        if final_report:
            df_res = pd.DataFrame(final_report)
            trades = df_res[df_res['Signal'].str.contains('BUY', na=False)]
            
            st.markdown(f"### 📊 RSRS稳健表现 (Numpy内核)")
            
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
