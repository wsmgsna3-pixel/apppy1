# -*- coding: utf-8 -*-
"""
选股王 · V32.0 强力监控版 (Top5 + 趋势透视)
策略：双创组合 (688 + 300)
更新：
1. 选股池扩大：监控 MACD Score 前 5 名。
2. 趋势透视：新增 D1/D3/D5 收益列，辅助判断持仓。
3. 界面复刻：恢复经典仪表盘与红绿表格。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import time
import os

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V32.0 强力监控台", layout="wide")
st.title("🔭 V32.0 强力监控台 (Top 5 入围 + 趋势透视)")
st.markdown("""
**策略逻辑 (No Future Function):**
* **选股 (T日盘后):** 计算 MACD 强度，选取 **Top 5** 候选股。
* **决策 (T+1日开盘):** * ✅ **买入**: 竞价高开 **[+2.0%, +7.5%]**。
    * 👀 **观望**: 开盘太弱或太强。
* **趋势列说明:** * **D1/D3/D5**: 分别代表买入后持有 1天、3天、5天的收益率，助您判断去留。
""")

# ---------------------------
# 全局缓存
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_CALENDAR = [] 
CHECKPOINT_FILE = "v32_checkpoint.csv" # 升级存档文件

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
    # 多取一些日子用于计算未来收益
    start_search = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=max(num_days * 5, 100))).strftime("%Y%m%d")
    end_search = (datetime.strptime(end_date_str, "%Y%m%d") + timedelta(days=60)).strftime("%Y%m%d")
    
    cal = safe_get('trade_cal', start_date=start_search, end_date=end_search)
    if cal.empty or 'is_open' not in cal.columns: return []
    
    global GLOBAL_CALENDAR
    open_cal = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=True)
    GLOBAL_CALENDAR = open_cal['cal_date'].tolist()
    
    # 截止到 end_date 的过去 num_days
    past_days = open_cal[open_cal['cal_date'] <= end_date_str]['cal_date'].tolist()
    return past_days[-num_days:]

# ----------------------------------------------------------------------
# 数据下载
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    
    # 增加 name 数据用于显示
    basic = safe_get('stock_basic', fields='ts_code,name')
    
    if not daily_df.empty:
        daily_df = daily_df[daily_df['ts_code'].str.startswith(('30', '688'))]
        if not basic.empty:
            daily_df = daily_df.merge(basic, on='ts_code', how='left')
            
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
        # 往后多取 15 天以计算 D5 收益
        end_fetch_idx = min(last_idx + 15, len(GLOBAL_CALENDAR) - 1)
        end_fetch_date = GLOBAL_CALENDAR[end_fetch_idx]
    except:
        end_fetch_date = (datetime.now() + timedelta(days=20)).strftime("%Y%m%d")

    start_fetch_date = (datetime.strptime(first_select_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    
    cal_range = safe_get('trade_cal', start_date=start_fetch_date, end_date=end_fetch_date, is_open='1')
    all_dates = cal_range['cal_date'].tolist()
    
    st.info(f"⏳ 正在预加载全量数据 ({start_fetch_date} ~ {end_fetch_date})...")

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
    cols_to_float = ['open', 'high', 'low', 'close', 'pre_close', 'vol']
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
    # 往前找足够的数据计算 MACD
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
# 核心逻辑 (Top 5 扫描)
# ----------------------------------------------------------------------
def run_strategy_step(select_date, min_price):
    try:
        daily_t = GLOBAL_DAILY_RAW.xs(select_date, level='trade_date')
    except KeyError: return []
    
    pool = daily_t[(daily_t['close'] >= min_price) & (daily_t['vol'] > 0)]
    if pool.empty: return []

    candidates = pool.index.tolist()
    scores = []
    
    # 1. 计算所有股票分数
    for code in candidates:
        s = compute_score(code, select_date)
        if s > 0:
            scores.append((code, s))
            
    # 2. 排序取 Top 5
    scores.sort(key=lambda x: x[1], reverse=True)
    top_5 = scores[:5]
    
    # 3. 获取 T+1 日期
    try:
        t_idx = GLOBAL_CALENDAR.index(select_date)
        if t_idx < len(GLOBAL_CALENDAR) - 1:
            buy_date = GLOBAL_CALENDAR[t_idx + 1]
        else:
            buy_date = None 
    except ValueError: buy_date = None

    results = []
    
    # 4. 对 Top 5 分别判定
    for rank, (code, score) in enumerate(top_5, 1):
        name = pool.loc[code, 'name'] if 'name' in pool.columns else code
        
        signal = "⏳ 等待开盘"
        open_pct = np.nan
        is_buy = False
        
        ret_d1 = np.nan
        ret_d3 = np.nan
        ret_d5 = np.nan
        
        if buy_date:
            try:
                d1_raw = GLOBAL_DAILY_RAW.loc[(code, buy_date)]
                if isinstance(d1_raw, pd.DataFrame): d1_raw = d1_raw.iloc[0]

                daily_buy_open = float(d1_raw['open'])
                daily_buy_pre = float(d1_raw['pre_close'])
                open_pct = (daily_buy_open / daily_buy_pre - 1) * 100
                
                # 买入逻辑
                if 2.0 <= open_pct <= 7.5:
                    is_buy = True
                    signal = "✅ BUY"
                elif open_pct < 2.0:
                    signal = "👀 弱"
                else:
                    signal = "⚠️ 强"
                    
                # 如果触发买入，计算未来趋势 (D1, D3, D5)
                if is_buy:
                    # 获取未来数据 (含 T+1 当天)
                    future_df = get_qfq_data(code, buy_date, "20991231")
                    if not future_df.empty:
                        buy_price = future_df.iloc[0]['open'] # 假设开盘买入
                        
                        # D1 (当天收盘)
                        if len(future_df) >= 1:
                            ret_d1 = (future_df.iloc[0]['close'] / buy_price - 1) * 100
                        
                        # D3 (第3个交易日收盘)
                        if len(future_df) >= 3:
                            ret_d3 = (future_df.iloc[2]['close'] / buy_price - 1) * 100
                            
                        # D5 (第5个交易日收盘)
                        if len(future_df) >= 5:
                            ret_d5 = (future_df.iloc[4]['close'] / buy_price - 1) * 100

            except (KeyError, TypeError):
                signal = "❌ 停牌"

        results.append({
            'Select_Date': select_date,
            'Trade_Date': buy_date if buy_date else "未来",
            'Rank': rank,
            'Code': code,
            'Name': name,
            'Signal': signal,
            'Open_Pct': open_pct,
            'Score': score,
            'Ret_D1': ret_d1,
            'Ret_D3': ret_d3,
            'Ret_D5': ret_d5
        })

    return results

# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测设置")
    default_date = datetime.now().date()
    end_date = st.date_input("选股截止日期", value=default_date)
    days_back = int(st.number_input("回测天数", value=5))
    
    st.markdown("---")
    st.header("2. 策略参数")
    MIN_PRICE = st.number_input("最低股价 (元)", value=20.0)
    
    st.markdown("---")
    if st.button("🗑️ 清空历史缓存"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            st.toast("缓存已清空", icon="🧹")

    st.markdown("---")
    TS_TOKEN = st.text_input("Tushare Token", type="password")

if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ---------------------------
# 主程序
# ---------------------------
if st.button("🚀 启动 Top5 扫描"):
    select_dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    
    if not select_dates: 
        st.error("❌ 日期获取失败")
        st.stop()
        
    st.info(f"📅 扫描区间: {select_dates[0]} ~ {select_dates[-1]}")
    
    # 读取断点
    processed_dates = []
    if os.path.exists(CHECKPOINT_FILE):
        try:
            cached_df = pd.read_csv(CHECKPOINT_FILE)
            if 'Select_Date' in cached_df.columns:
                processed_dates = cached_df['Select_Date'].astype(str).unique().tolist()
                st.success(f"📂 已加载历史记录，跳过 {len(processed_dates)} 天")
        except: pass
    
    todo_dates = [d for d in select_dates if str(d) not in processed_dates]
    
    # 只有新任务时才拉取数据
    if todo_dates:
        if not get_all_historical_data(select_dates): st.stop()
        
        status_bar = st.progress(0)
        status_text = st.empty()
        
        for i, date in enumerate(todo_dates):
            status_text.text(f"正在计算: {date} (Top 5 寻优中...)")
            res_list = run_strategy_step(date, MIN_PRICE)
            
            if res_list:
                df_chunk = pd.DataFrame(res_list)
                need_header = not os.path.exists(CHECKPOINT_FILE)
                df_chunk.to_csv(CHECKPOINT_FILE, mode='a', header=need_header, index=False)
            
            status_bar.progress((i+1)/len(todo_dates))
        
        status_bar.empty()
        status_text.empty()

    # ---------------------------
    # 结果展示 (恢复经典 Dashboard)
    # ---------------------------
    if os.path.exists(CHECKPOINT_FILE):
        full_df = pd.read_csv(CHECKPOINT_FILE)
        # 过滤当前请求日期
        mask = full_df['Select_Date'].astype(str).isin([str(d) for d in select_dates])
        df_show = full_df[mask].copy()
        
        if df_show.empty:
            st.warning("暂无数据")
        else:
            # 1. 核心指标 Dashboard
            trades = df_show[df_show['Signal'].str.contains('BUY', na=False)]
            
            st.markdown("### 📊 策略表现 (D5持有基准)")
            col1, col2, col3, col4 = st.columns(4)
            
            total_buy = len(trades)
            # 使用 D5 收益作为胜率基准，如果没有 D5 用 D1 顶替
            final_ret = trades['Ret_D5'].fillna(trades['Ret_D1'])
            
            avg_ret = final_ret.mean()
            win_rate = (final_ret > 0).mean() * 100
            
            col1.metric("入围股票数", f"{len(df_show)}")
            col2.metric("触发交易", f"{total_buy}", delta="Top 5 贡献")
            col3.metric("平均收益 (D5)", f"{avg_ret:.2f}%")
            col4.metric("胜率 (D5)", f"{win_rate:.1f}%")

            # 2. 详细表格 (恢复高亮样式)
            st.markdown("### 📋 每日 Top 5 监控明细")
            
            def color_signal(val):
                if 'BUY' in str(val): return 'background-color: #ff4b4b; color: white'
                if '弱' in str(val): return 'color: #808080'
                if '强' in str(val): return 'color: #ffaa00'
                return ''
            
            def color_ret(val):
                if pd.isna(val): return ''
                if val > 0: return 'color: red'
                if val < 0: return 'color: green'
                return ''

            st.dataframe(
                df_show[['Trade_Date', 'Code', 'Name', 'Signal', 'Open_Pct', 'Rank', 'Ret_D1', 'Ret_D3', 'Ret_D5']]
                .style
                .map(color_signal, subset=['Signal'])
                .map(color_ret, subset=['Ret_D1', 'Ret_D3', 'Ret_D5'])
                .format({
                    'Open_Pct': '{:.2f}%',
                    'Ret_D1': '{:.2f}%',
                    'Ret_D3': '{:.2f}%',
                    'Ret_D5': '{:.2f}%'
                }),
                use_container_width=True
            )
            
            csv = df_show.to_csv().encode('utf-8')
            st.download_button("📥 下载完整报表", csv, "v32_top5_report.csv", "text/csv")
