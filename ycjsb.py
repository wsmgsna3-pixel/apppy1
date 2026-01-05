# -*- coding: utf-8 -*-
"""
选股王 · V31.2 真实波段版 (断点续传 + 修复版)
策略：双创组合 (688 + 300)
特性：
1. 去除未来函数：T日盘后选股，T+1日开盘买入。
2. 断点续传：每跑完一天自动保存，崩溃后重启可自动接着跑。
3. 全市场扫描：不再局限于前150名。
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
st.set_page_config(page_title="V31.2 真实选股台", layout="wide")
st.title("🛡️ V31.2 真实选股监控台 (含断点续传)")
st.markdown("""
**策略逻辑 (Swing Trading):**
* **选股 (T日):** 盘后计算全市场 MACD Score，选出 **Rank 1**。
* **买入 (T+1日):** * 竞价高开 **[+2.0%, +7.5%]** -> ✅ 买入。
    * 否则 -> 👀 观望。
* **卖出:** * 创业板 (30): T+2 开盘卖出。
    * 科创板 (688): T+6 收盘卖出。
""")

# ---------------------------
# 全局缓存
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_CALENDAR = [] 
CHECKPOINT_FILE = "v31_checkpoint_data.csv" # 断点续传存档文件

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
    # 获取足够长的日历
    start_search = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=max(num_days * 5, 60))).strftime("%Y%m%d")
    end_search = (datetime.strptime(end_date_str, "%Y%m%d") + timedelta(days=30)).strftime("%Y%m%d")
    
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
    
    first_select_date = min(select_days_list)
    last_select_date = max(select_days_list)
    
    try:
        last_idx = GLOBAL_CALENDAR.index(last_select_date)
        end_fetch_idx = min(last_idx + 15, len(GLOBAL_CALENDAR) - 1)
        end_fetch_date = GLOBAL_CALENDAR[end_fetch_idx]
    except:
        end_fetch_date = (datetime.now() + timedelta(days=10)).strftime("%Y%m%d")

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
    start_date = (datetime.strptime(current_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data(ts_code, start_date, current_date)
    
    if df.empty or len(df) < 30: return -1
    
    last_date = df.iloc[-1]['trade_date']
    if hasattr(last_date, 'strftime'):
        last_date_str = last_date.strftime('%Y%m%d')
    else:
        last_date_str = str(last_date)
        
    if last_date_str != current_date:
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
# 核心逻辑
# ----------------------------------------------------------------------
def run_strategy_step(select_date, min_price):
    # 返回默认结构，防止 None 导致无法记录断点
    default_res = {
        'Select_Date': select_date,
        'Buy_Date': "-",
        'ts_code': "-",
        'Signal': "无符合股票",
        'T_Close': 0,
        'T+1_Open_Pct': 0,
        'Score': 0,
        'Return_Strategy': None
    }

    try:
        daily_t = GLOBAL_DAILY_RAW.xs(select_date, level='trade_date')
    except KeyError: return default_res
    
    pool = daily_t[(daily_t['close'] >= min_price) & (daily_t['vol'] > 0)]
    if pool.empty: return default_res

    best_score = -9999
    rank1_code = None
    rank1_close_t = 0
    
    candidates = pool.index.tolist()
    
    for code in candidates:
        s = compute_score(code, select_date)
        if s > best_score:
            best_score = s
            rank1_code = code
            rank1_close_t = pool.loc[code, 'close']

    if not rank1_code: return default_res
    
    # 获取买入日期
    try:
        t_idx = GLOBAL_CALENDAR.index(select_date)
        if t_idx < len(GLOBAL_CALENDAR) - 1:
            buy_date = GLOBAL_CALENDAR[t_idx + 1]
        else:
            buy_date = None 
    except ValueError:
        buy_date = None

    signal_type = "⏳ 等待次日开盘"
    open_pct = None
    ret_strategy = None
    is_buy = False
    
    if buy_date:
        try:
            d1_raw = GLOBAL_DAILY_RAW.loc[(rank1_code, buy_date)]
            if isinstance(d1_raw, pd.DataFrame):
                d1_raw = d1_raw.iloc[0]

            daily_buy_open = float(d1_raw['open'])
            daily_buy_pre = float(d1_raw['pre_close'])
            open_pct = (daily_buy_open / daily_buy_pre - 1) * 100
            
            if 2.0 <= open_pct <= 7.5:
                is_buy = True
                signal_type = f"✅ BUY (T+1日 {buy_date})"
            elif open_pct < 2.0:
                signal_type = "👀 观望 (T+1开盘太弱)"
            else:
                signal_type = "⚠️ 观望 (T+1开盘太强)"
                
        except (KeyError, TypeError):
            signal_type = "❌ 数据缺失 (T+1停牌?)"

    if is_buy and buy_date:
        future_df = get_qfq_data(rank1_code, buy_date, "20991231")
        if not future_df.empty:
            buy_price_real = future_df.iloc[0]['open']
            sell_price = None
            
            if rank1_code.startswith('30'):
                if len(future_df) >= 2:
                    sell_price = future_df.iloc[1]['open']
                elif len(future_df) == 1:
                    sell_price = future_df.iloc[0]['close'] 
            elif rank1_code.startswith('688'):
                hold_days = 5
                if len(future_df) >= (hold_days + 1):
                    sell_price = future_df.iloc[hold_days]['close']
                else:
                    sell_price = future_df.iloc[-1]['close'] 
            
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
    st.header("3. 系统控制")
    # 清空缓存按钮
    if st.button("🗑️ 清空缓存重新运行"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            st.toast("已清空历史缓存，请点击下方【开始/继续】", icon="🧹")
        else:
            st.toast("暂无缓存文件", icon="ℹ️")

    st.markdown("---")
    TS_TOKEN = st.text_input("Tushare Token", type="password")

if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ---------------------------
# 主程序
# ---------------------------
if st.button("🚀 开始 / 继续扫描"):
    # 1. 获取选股日期列表
    select_dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    
    if not select_dates: 
        st.error(f"❌ 无法获取交易日历，请检查 Token 或网络。")
        st.stop()
        
    st.info(f"📅 目标选股日期范围: {select_dates[0]} ~ {select_dates[-1]}")
    
    # 2. 检查断点 (Checkpoint Check)
    processed_dates = []
    if os.path.exists(CHECKPOINT_FILE):
        try:
            cached_df = pd.read_csv(CHECKPOINT_FILE)
            if 'Select_Date' in cached_df.columns:
                # 转换成字符串确保匹配
                processed_dates = cached_df['Select_Date'].astype(str).tolist()
                st.success(f"📂 检测到断点存档，已跳过 {len(processed_dates)} 个交易日。")
        except Exception as e:
            st.warning(f"存档文件读取失败，将重新运行: {e}")
    
    # 过滤掉已经跑过的日期
    todo_dates = [d for d in select_dates if str(d) not in processed_dates]
    
    if not todo_dates and len(processed_dates) > 0:
        st.info("✅ 所有日期均已处理完毕，直接展示结果。")
    
    else:
        # 3. 只有当有新任务时才拉取数据
        if todo_dates:
            if not get_all_historical_data(select_dates): st.stop() # 拉取数据范围还是整体的，保证计算指标连续性
            
            st.write(f"🚀 开始处理剩余 {len(todo_dates)} 个交易日...")
            
            status_text = st.empty()
            bar = st.progress(0)
            
            for i, date in enumerate(todo_dates):
                status_text.text(f"正在分析: {date} ...")
                
                # 运行策略
                res = run_strategy_step(date, MIN_PRICE)
                
                # --- 断点保存 (Append Mode) ---
                if res:
                    df_single = pd.DataFrame([res])
                    # 如果文件不存在，写入表头；如果存在，追加不写表头
                    need_header = not os.path.exists(CHECKPOINT_FILE)
                    df_single.to_csv(CHECKPOINT_FILE, mode='a', header=need_header, index=False)
                
                bar.progress((i+1)/len(todo_dates))
            
            bar.empty()
            status_text.empty()

    # 4. 展示最终合并结果
    if os.path.exists(CHECKPOINT_FILE):
        full_results = pd.read_csv(CHECKPOINT_FILE)
        
        # 过滤只显示本次请求时间范围内的数据 (防止CSV里堆积了去年的数据)
        mask = full_results['Select_Date'].astype(str).isin([str(d) for d in select_dates])
        df_display = full_results[mask].copy()
        
        if df_display.empty:
            st.warning("⚠️ 结果为空 (可能所有日期都无符合条件股票)")
        else:
             # A. 核心统计
            executed_trades = df_display[df_display['Signal'].str.contains('BUY', na=False)]
            
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
                st.info("💡 选定区间内次日开盘均未满足【+2%~+7.5%】的买入条件。")

            # B. 每日明细
            st.markdown("### 📋 每日交易明细")
            
            def highlight_signal(val):
                if 'BUY' in str(val): return 'color: #ff4b4b; font-weight: bold' 
                if '观望' in str(val): return 'color: #808080' 
                return ''

            def safe_fmt(val):
                return f"{val:.2f}%" if pd.notnull(val) else "-"

            st.dataframe(
                df_display[['Select_Date', 'Buy_Date', 'ts_code', 'Signal', 'T+1_Open_Pct', 'Return_Strategy', 'Score']]
                .style
                .map(highlight_signal, subset=['Signal'])
                .format({
                    'T+1_Open_Pct': safe_fmt,
                    'Return_Strategy': safe_fmt,
                    'Score': '{:.0f}'
                }),
                use_container_width=True
            )
            
            csv = df_display.to_csv().encode('utf-8')
            st.download_button("📥 下载完整回测结果 CSV", csv, "v31.2_checkpoint_result.csv", "text/csv")
    else:
        st.warning("还没有产生任何数据。")
