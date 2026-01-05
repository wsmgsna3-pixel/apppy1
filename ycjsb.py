# -*- coding: utf-8 -*-
"""
选股王 · V31.0 真实波段版 (无未来函数)
策略：双创组合 (688 + 300)
逻辑修正：
1. 选股时间：T日收盘后 (使用T日及过去数据选出 Rank 1)。
2. 买入时间：T+1日开盘 (使用T+1日开盘数据决策)。
3. 选股池：全市场扫描，不再局限于涨幅前150名。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import time

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V31.0 真实选股台", layout="wide")
st.title("🛡️ V31.0 真实选股监控台 (去伪存真版)")
st.markdown("""
**策略逻辑 (Swing Trading):**
* **选股 (T日):** 盘后计算全市场 MACD Score，选出 **Rank 1**。
* **买入 (T+1日):** * 竞价高开 **[+2.0%, +7.5%]** -> ✅ 买入。
    * 否则 -> 👀 观望。
* **卖出:** * 创业板 (30): T+2 开盘卖出 (持仓1天)。
    * 科创板 (688): T+6 收盘卖出 (持仓5天)。
""")

# ---------------------------
# 全局缓存
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_CALENDAR = [] # 存储交易日历列表

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
    # 获取足够长的日历，包含未来几天以便计算收益
    start_search = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=max(num_days * 5, 60))).strftime("%Y%m%d")
    end_search = (datetime.strptime(end_date_str, "%Y%m%d") + timedelta(days=30)).strftime("%Y%m%d")
    
    cal = safe_get('trade_cal', start_date=start_search, end_date=end_search)
    if cal.empty or 'is_open' not in cal.columns: return []
    
    # 全局日历缓存，用于查找 "下一交易日"
    global GLOBAL_CALENDAR
    open_cal = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=True)
    GLOBAL_CALENDAR = open_cal['cal_date'].tolist()
    
    # 返回用于"选股"的日子 (截止到 end_date 之前的 num_days 个)
    # 过滤掉 end_date 之后的日子用于回测选股
    past_days = open_cal[open_cal['cal_date'] <= end_date_str]['cal_date'].tolist()
    return past_days[-num_days:]

# ----------------------------------------------------------------------
# 数据下载
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    
    if not daily_df.empty:
        daily_df = daily_df[daily_df['ts_code'].str.startswith(('30', '688'))]
    if not adj_df.empty:
        adj_df = adj_df[adj_df['ts_code'].str.startswith(('30', '688'))]
        
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(select_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_CALENDAR
    if not select_days_list: return False
    
    # 确定数据拉取范围：
    # 开始：最早选股日往前推 150 天 (计算指标用)
    # 结束：最晚选股日往后推 20 天 (计算 T+1 买入和 T+N 卖出用)
    
    first_select_date = min(select_days_list)
    last_select_date = max(select_days_list)
    
    # 在全局日历中找到 last_select_date 的索引，往后多取 15 个交易日
    try:
        last_idx = GLOBAL_CALENDAR.index(last_select_date)
        end_fetch_idx = min(last_idx + 15, len(GLOBAL_CALENDAR) - 1)
        end_fetch_date = GLOBAL_CALENDAR[end_fetch_idx]
    except:
        end_fetch_date = (datetime.now() + timedelta(days=10)).strftime("%Y%m%d")

    start_fetch_date = (datetime.strptime(first_select_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    
    # 获取范围内所有交易日
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
    cols_to_float = ['open', 'high', 'low', 'close', 'pre_close', 'vol']
    for col in cols_to_float:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    
    # 缓存最新的复权因子作为基准
    latest_date_in_data = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_date_in_data:
        GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_date_in_data), 'adj_factor'].droplevel(1).to_dict()
    
    return True

def get_qfq_data(ts_code, start_date, end_date):
    base_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code)
    if not base_adj: return pd.DataFrame()

    try:
        # 切片获取区间数据
        daily = GLOBAL_DAILY_RAW.loc[(ts_code, slice(start_date, end_date)), :]
        adj = GLOBAL_ADJ_FACTOR.loc[(ts_code, slice(start_date, end_date)), 'adj_factor']
    except: return pd.DataFrame()
    
    if daily.empty or adj.empty: return pd.DataFrame()
    
    # 对齐数据
    df = daily.join(adj, how='left').dropna(subset=['adj_factor'])
    factor = df['adj_factor'] / base_adj
    
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns: df[col] = df[col] * factor
    
    df = df.reset_index()
    return df.sort_values('trade_date')

# ----------------------------------------------------------------------
# 评分逻辑 (不变，确保用的是 end_date 之前的数据)
# ----------------------------------------------------------------------
def compute_score(ts_code, current_date):
    # 取当前日期前120天的数据计算指标
    start_date = (datetime.strptime(current_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data(ts_code, start_date, current_date)
    
    if df.empty or len(df) < 30: return -1
    
    # 确保最后一行数据的日期就是 current_date (防止停牌股混入)
    if df.iloc[-1]['trade_date'].strftime('%Y%m%d') != current_date:
        return -1

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
# 核心逻辑：T日选股，T+1日交互
# ----------------------------------------------------------------------
def run_strategy_step(select_date, min_price):
    # 1. 获取 T 日所有符合价格条件的股票
    try:
        daily_t = GLOBAL_DAILY_RAW.xs(select_date, level='trade_date')
    except KeyError: return None
    
    # 筛选基础池：价格达标 & 必须有成交量(非停牌)
    pool = daily_t[(daily_t['close'] >= min_price) & (daily_t['vol'] > 0)]
    if pool.empty: return None

    # 2. 全市场扫描 Score (不再局限于前150名)
    # 为了速度，我们使用简单的进度显示
    best_score = -9999
    rank1_code = None
    rank1_close_t = 0
    
    # 候选列表：只计算涨幅非负的？用户要求扩大池子，我们计算全部
    # 但为了性能，可以剔除跌停板的，保留一点理性
    # 这里严格按照用户要求：全市场扫描修正
    candidates = pool.index.tolist()
    
    # 简单的批处理循环
    for code in candidates:
        s = compute_score(code, select_date)
        if s > best_score:
            best_score = s
            rank1_code = code
            rank1_close_t = pool.loc[code, 'close']

    if not rank1_code: return None
    
    # 3. 进入 T+1 日买入决策
    # 找到 select_date 的下一个交易日
    try:
        t_idx = GLOBAL_CALENDAR.index(select_date)
        if t_idx < len(GLOBAL_CALENDAR) - 1:
            buy_date = GLOBAL_CALENDAR[t_idx + 1]
        else:
            buy_date = None # 已经是最新数据，无法回测明天
    except ValueError:
        buy_date = None

    signal_type = "⏳ 等待次日开盘"
    open_pct = None
    ret_strategy = None
    is_buy = False
    
    # 如果有 T+1 数据，进行买入判定
    if buy_date:
        try:
            # 获取 T+1 日数据
            d1_raw = GLOBAL_DAILY_RAW.loc[(rank1_code, buy_date)]
            # 计算开盘涨幅 (相对于 T 日收盘价，或 T+1 的 PreClose)
            # T+1 的 pre_close 理论上等于 T 的 close
            # 使用真实数据计算：
            daily_buy_open = d1_raw['open']
            daily_buy_pre = d1_raw['pre_close']
            
            open_pct = (daily_buy_open / daily_buy_pre - 1) * 100
            
            # --- 修正后的买入逻辑 ---
            if 2.0 <= open_pct <= 7.5:
                is_buy = True
                signal_type = f"✅ BUY (T+1日 {buy_date})"
            elif open_pct < 2.0:
                signal_type = "👀 观望 (T+1开盘太弱)"
            else:
                signal_type = "⚠️ 观望 (T+1开盘太强)"
                
        except KeyError:
            signal_type = "❌ 数据缺失 (T+1停牌?)"

    # 4. 如果买入，计算收益 (Swing 模式)
    if is_buy and buy_date:
        # 获取买入后的数据用于卖出
        # 30系：T+2 开盘卖
        # 688系：T+6 收盘卖 (策略原意是持仓5天，这里简化逻辑：30系隔日，688持多日)
        
        # 获取未来数据流
        future_df = get_qfq_data(rank1_code, buy_date, "20991231")
        
        if not future_df.empty:
            buy_price_real = future_df.iloc[0]['open'] # T+1 Open
            
            sell_price = None
            
            if rank1_code.startswith('30'):
                # 30系：T+2 (索引1) Open 卖出
                if len(future_df) >= 2:
                    sell_price = future_df.iloc[1]['open']
                elif len(future_df) == 1:
                    # 还没到 T+2，用当前收盘估算
                    sell_price = future_df.iloc[0]['close'] 
            
            elif rank1_code.startswith('688'):
                # 688系：T+6 (索引5) Close 卖出 (持仓5天)
                hold_days = 5
                if len(future_df) >= (hold_days + 1):
                    sell_price = future_df.iloc[hold_days]['close']
                else:
                    sell_price = future_df.iloc[-1]['close'] # 拿最新的算
            
            if sell_price:
                ret_strategy = (sell_price / buy_price_real - 1) * 100

    return {
        'Select_Date': select_date,
        'Buy_Date': buy_date if buy_date else "未来",
        'ts_code': rank1_code,
        'Signal': signal_type,
        'T_Close': rank1_close_t,
        'T+1_Open_Pct': open_pct,
        'Score': best_score,
        'Return_Strategy': ret_strategy
    }

# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 真实回测设置")
    default_date = datetime.now().date()
    end_date = st.date_input("选股截止日期", value=default_date)
    days_back = int(st.number_input("回测天数", value=5))
    
    st.markdown("---")
    st.header("2. 策略参数")
    MIN_PRICE = st.number_input("最低股价 (元)", value=20.0)
    
    st.markdown("---")
    TS_TOKEN = st.text_input("Tushare Token", type="password")

if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ---------------------------
# 主程序
# ---------------------------
if st.button("🚀 开始真实扫描"):
    # 1. 获取选股日期列表
    select_dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    
    if not select_dates: 
        st.error(f"❌ 无法获取交易日历。")
        st.stop()
        
    st.info(f"📅 选股日期范围: {select_dates[0]} ~ {select_dates[-1]}")
    
    # 2. 拉取数据 (包含未来数据以便计算 T+1)
    if not get_all_historical_data(select_dates): st.stop()
    
    # 3. 逐日回测
    results = []
    status_text = st.empty()
    bar = st.progress(0)
    
    for i, date in enumerate(select_dates):
        status_text.text(f"正在全市场扫描选股: {date} ... (速度较慢请耐心等待)")
        res = run_strategy_step(date, MIN_PRICE)
        if res:
            results.append(res)
        bar.progress((i+1)/len(select_dates))
    
    bar.empty()
    status_text.empty()
    
    if not results:
        st.warning("⚠️ 无结果。")
        st.stop()
        
    df_res = pd.DataFrame(results)
    
    # 4. 展示结果
    
    # A. 核心统计 (只统计实际买入的)
    executed_trades = df_res[df_res['Signal'].str.contains('BUY', na=False)]
    
    st.markdown("### 📊 真实波段统计")
    if not executed_trades.empty:
        col1, col2, col3 = st.columns(3)
        avg_ret = executed_trades['Return_Strategy'].mean()
        win_rate = (executed_trades['Return_Strategy'] > 0).mean() * 100
        count = len(executed_trades)
        
        col1.metric("触发买入次数", f"{count}")
        col2.metric("平均收益", f"{avg_ret:.2f}%")
        col3.metric("胜率", f"{win_rate:.1f}%")
    else:
        st.info("💡 选定区间内虽然选出了Rank1，但次日开盘均未满足【+2%~+7.5%】的买入条件。")

    # B. 每日明细
    st.markdown("### 📋 每日交易明细")
    
    def highlight_signal(val):
        if 'BUY' in str(val): return 'color: #ff4b4b; font-weight: bold' # Red
        if '观望' in str(val): return 'color: #808080' # Grey
        return ''

    def safe_fmt(val):
        return f"{val:.2f}%" if pd.notnull(val) else "-"

    st.dataframe(
        df_res[['Select_Date', 'Buy_Date', 'ts_code', 'Signal', 'T+1_Open_Pct', 'Return_Strategy', 'Score']]
        .style
        .map(highlight_signal, subset=['Signal'])
        .format({
            'T+1_Open_Pct': safe_fmt,
            'Return_Strategy': safe_fmt,
            'Score': '{:.0f}'
        }),
        use_container_width=True
    )
    
    csv = df_res.to_csv().encode('utf-8')
    st.download_button("📥 下载回测结果 CSV", csv, "v31_real_backtest.csv", "text/csv")
