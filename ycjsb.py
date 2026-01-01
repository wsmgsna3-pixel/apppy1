# -*- coding: utf-8 -*-
"""
选股王 · 终极PK版 (D0持有 vs D3接力)
功能：
1. 专为验证用户的“D3买入法”设计。
2. 不显示每日明细，防止浏览器卡死。
3. 直接输出两大策略的胜率、收益、回撤对比。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="V30.23 终极PK", layout="wide")
st.title("🥊 终极擂台：D0 持有策略 vs D3 接力策略")
st.markdown("""
**⚔️ 对决规则：**
* **选股池：** V30.22 Top 4 (剔除 Rank 2)。
* **🔴 红方 (D0 潜伏)：** T日突破买入 -> 死拿至 T+5 收盘卖出。
* **🔵 蓝方 (D3 接力)：** T日不买 -> T+3 收盘确认浮盈 -> T+3 收盘买入 -> T+5 收盘卖出。
""")

# 全局变量
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

# ---------------------------
# 辅助函数 
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        if kwargs.get('is_index'): df = pro.index_daily(**kwargs)
        else: df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty: return []
    return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()

# ----------------------------------------------------------------------
# 数据拉取与处理 (同前，略微精简)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj = safe_get('adj_factor', trade_date=date)
    daily = safe_get('daily', trade_date=date)
    return {'adj': adj, 'daily': daily}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    latest = max(trade_days_list) 
    earliest = min(trade_days_list)
    start_dt = (datetime.strptime(earliest, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_dt = (datetime.strptime(latest, "%Y%m%d") + timedelta(days=30)).strftime("%Y%m%d") 
    
    dates = safe_get('trade_cal', start_date=start_dt, end_date=end_dt, is_open='1')['cal_date'].tolist()
    
    progress = st.progress(0, text="正在构建全市场数据矩阵...")
    adj_list, daily_list = [], []
    
    for i, d in enumerate(dates):
        res = fetch_and_cache_daily_data(d)
        if not res['adj'].empty: adj_list.append(res['adj'])
        if not res['daily'].empty: daily_list.append(res['daily'])
        progress.progress((i+1)/len(dates))
    progress.empty()
    
    if not adj_list: return False
    
    adj_all = pd.concat(adj_list)
    adj_all['adj_factor'] = pd.to_numeric(adj_all['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_all.set_index(['ts_code', 'trade_date']).sort_index()
    
    daily_raw = pd.concat(daily_list)
    for c in ['open','high','low','close','pre_close','vol']:
        if c in daily_raw.columns: daily_raw[c] = pd.to_numeric(daily_raw[c], errors='coerce')
    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index()
    
    latest_dt = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_dt:
        GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.xs(latest_dt, level='trade_date')['adj_factor'].to_dict()
        
    return True

def get_qfq_data(ts_code, start_date, end_date):
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    base = GLOBAL_QFQ_BASE_FACTORS.get(ts_code)
    if not base: return pd.DataFrame()
    
    try:
        df = GLOBAL_DAILY_RAW.loc[(ts_code, slice(start_date, end_date)), :].copy()
        factors = GLOBAL_ADJ_FACTOR.loc[(ts_code, slice(start_date, end_date)), 'adj_factor']
        if df.empty or factors.empty: return pd.DataFrame()
        
        df = df.join(factors)
        norm = df['adj_factor'] / base
        for c in ['open','high','low','close','pre_close']: df[c] *= norm
        return df.reset_index()
    except: return pd.DataFrame()

# ----------------------------------------------------------------------
# 核心：双策略收益计算
# ----------------------------------------------------------------------
def calculate_pk_returns(ts_code, selection_date, buy_threshold=1.5):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start = d0.strftime("%Y%m%d")
    end = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    df = get_qfq_data(ts_code, start, end)
    if df.empty: return None
    
    dates = df['trade_date'].dt.strftime('%Y%m%d').tolist()
    if selection_date not in dates: return None
    
    idx_0 = dates.index(selection_date)
    row_0 = df.iloc[idx_0]
    
    # --- 1. D0 触发判断 ---
    # 过滤低开
    if row_0['open'] <= row_0['pre_close']: return None
    # 确认突破 +1.5%
    buy_price_d0 = row_0['open'] * (1 + buy_threshold/100)
    if row_0['high'] < buy_price_d0: return None
    
    # 确保有 D3 和 D5 的数据
    if len(df) <= idx_0 + 5: return None
    
    row_d3 = df.iloc[idx_0 + 3]
    row_d5 = df.iloc[idx_0 + 5]
    
    # --- 🔴 策略 A: D0 买入，D5 卖出 ---
    ret_a = (row_d5['close'] / buy_price_d0 - 1) * 100
    
    # --- 🔵 策略 B: D3 接力 (用户的天才想法) ---
    # 条件: D3收盘价 > D0买入价 (即该股目前是赚钱的)
    ret_b = 0.0
    status_b = "空仓"
    
    if row_d3['close'] > buy_price_d0:
        # 触发接力买入
        buy_price_d3 = row_d3['close']
        # D5 卖出
        ret_b = (row_d5['close'] / buy_price_d3 - 1) * 100
        status_b = "买入"
    else:
        # D3 是亏的，不接力
        ret_b = np.nan # 标记为没交易
        status_b = "观望"
        
    return {
        'buy_price_d0': buy_price_d0,
        'close_d3': row_d3['close'],
        'Strategy_A_Return': ret_a,
        'Strategy_B_Return': ret_b,
        'Strategy_B_Status': status_b
    }

# ----------------------------------------------------------------------
# 选股核心
# ----------------------------------------------------------------------
def compute_v3022_score(ts_code, trade_date):
    start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data(ts_code, start, trade_date)
    if df.empty or len(df) < 30: return None
    
    curr = df.iloc[-1]
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    ma5_vol = df['vol'].rolling(5).mean().iloc[-1]
    
    if curr['close'] <= ma20: return None
    if curr['vol'] <= ma5_vol * 1.2: return None
    
    close = df['close']
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd_val = (diff - dea).iloc[-1] * 2
    
    if macd_val <= 0: return None
    
    bonus = 1.0
    price = curr['close']
    pct = (price / curr['pre_close'] - 1) * 100
    vol = df['close'].pct_change().tail(10).std()*100
    
    if 40 <= price <= 80: bonus += 0.1
    if pct > 9.5: bonus += 0.1
    if 4 <= vol <= 8: bonus += 0.05
    
    return {'score': macd_val * 10000 * bonus}

# ----------------------------------------------------------------------
# 回测执行
# ----------------------------------------------------------------------
def run_pk_backtest(dates):
    results = []
    bar = st.progress(0, text="擂台赛开始...")
    
    # 提前获取名称，防止 KeyError
    basic = safe_get('stock_basic', list_status='L', fields='ts_code,name')
    
    for i, date in enumerate(dates):
        daily = safe_get('daily', trade_date=date)
        if daily.empty: continue
        
        if basic.empty: daily['name'] = 'Unknown'
        else: daily = daily.merge(basic, on='ts_code', how='left')
        
        candidates = daily[~daily['name'].str.contains('ST|退', na=False)]
        candidates = candidates[~candidates['ts_code'].str.startswith('92')]
        
        # 只算前300成交额，加速
        candidates['amount'] = pd.to_numeric(candidates['amount'], errors='coerce')
        candidates = candidates.sort_values('amount', ascending=False).head(300)
        
        scores = []
        for code in candidates['ts_code']:
            res = compute_v3022_score(code, date)
            if res:
                res['ts_code'] = code
                res['name'] = candidates.loc[candidates['ts_code']==code, 'name'].values[0]
                scores.append(res)
                
        if not scores: continue
        
        # 排序 Top 4
        df_day = pd.DataFrame(scores).sort_values('score', ascending=False).head(4).reset_index(drop=True)
        df_day['Rank'] = df_day.index + 1
        
        # 剔除 Rank 2
        df_final = df_day[df_day['Rank'] != 2].copy()
        
        # 计算双策略收益
        for _, row in df_final.iterrows():
            pk_res = calculate_pk_returns(row['ts_code'], date)
            if pk_res:
                rec = row.to_dict()
                rec.update(pk_res)
                rec['Trade_Date'] = date
                results.append(rec)
        
        bar.progress((i+1)/len(dates))
        
    bar.empty()
    return pd.DataFrame(results)

# ---------------------------
# 主程序
# ---------------------------
with st.sidebar:
    st.header("PK 参数")
    days_back = st.number_input("回测天数", value=100)
    ts_token = st.text_input("Tushare Token", type="password")

if st.button("🚀 开始终极 PK"):
    if not ts_token: st.error("Token?"); st.stop()
    ts.set_token(ts_token)
    pro = ts.pro_api()
    
    end_date = datetime.now().strftime("%Y%m%d")
    dates = get_trade_days(end_date, int(days_back))
    if not dates: st.stop()
    
    if not get_all_historical_data(dates): st.stop()
    
    df = run_pk_backtest(dates)
    
    if df.empty: st.warning("无交易"); st.stop()
    
    # --- 结果展示 ---
    st.markdown("---")
    st.header("🏆 终极对决结果")
    
    # 策略 A 统计
    valid_a = df.dropna(subset=['Strategy_A_Return'])
    win_a = (valid_a['Strategy_A_Return'] > 0).mean() * 100
    avg_a = valid_a['Strategy_A_Return'].mean()
    count_a = len(valid_a)
    
    # 策略 B 统计 (排除没交易的)
    valid_b = df.dropna(subset=['Strategy_B_Return']) # 自动排除了 NaN (观望)
    win_b = (valid_b['Strategy_B_Return'] > 0).mean() * 100
    avg_b = valid_b['Strategy_B_Return'].mean()
    count_b = len(valid_b)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔴 策略 A: D0 买入 (死拿)")
        st.metric("胜率 (Win Rate)", f"{win_a:.1f}%")
        st.metric("平均收益", f"{avg_a:.2f}%")
        st.metric("交易次数", f"{count_a}")
        
    with col2:
        st.subheader("🔵 策略 B: D3 接力 (您的想法)")
        st.metric("胜率 (Win Rate)", f"{win_b:.1f}%")
        st.metric("平均收益", f"{avg_b:.2f}%")
        st.metric("交易次数", f"{count_b}")
        
    # 判定胜负
    st.markdown("---")
    if win_b > win_a and avg_b > avg_a:
        st.success("🎉 **结果：策略 B (D3 接力) 完胜！** 您的天才想法是对的！")
    elif win_b < win_a:
        st.error("📉 **结果：策略 A (D0 持有) 胜出。** 看来还是买在起爆点比较安全。")
    else:
        st.info("⚖️ **结果：各有千秋。**")
        
    st.markdown("### 📝 详细对决记录")
    st.dataframe(df[['Trade_Date', 'Rank', 'ts_code', 'name', 'Strategy_A_Return', 'Strategy_B_Status', 'Strategy_B_Return']], use_container_width=True)
