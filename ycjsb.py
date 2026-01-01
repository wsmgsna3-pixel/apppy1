# -*- coding: utf-8 -*-
"""
选股王 · V30.23 验证版 (Relay 接力策略)
用户逻辑：
1. D0选出 Top 4，剔除 Rank 2。
2. 观察期：不买入，等 D3。
3. 买入点：如果 D3收盘价 > D0买入价 (即强者恒强)，则在 D3 收盘买入。
4. 卖出点：D5 收盘卖出。
5. 目的：验证是否可以通过“跳过洗盘”来提高胜率和盈亏比。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 全局设置
# ---------------------------
st.set_page_config(page_title="V30.23 接力策略验证", layout="wide")
st.title("🛡️ V30.23 验证版 · 接力策略 (Relay Strategy)")
st.markdown("""
**⚔️ 策略逻辑 (用户天才设想版)：**
1. **初筛：** V30.22 (暴力MACD + 黄金形态)，取 **Top 4**。
2. **清洗：** 🚫 **剔除第 2 名** (保留 Rank 1, 3, 4)。
3. **接力信号 (D3)：** - 只有当 **D3 收盘价 > D0 买入成本** (即这只票抗住了洗盘且盈利) 时，才买入！
   - 否则 **空仓观望**。
4. **持有期：** D3 买入 -> D5 卖出 (只吃主升浪)。
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
        return df if df is not None and not df.empty else pd.DataFrame(columns=['ts_code'])
    except: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty: return []
    return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()

# ----------------------------------------------------------------------
# 数据核心
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj = safe_get('adj_factor', trade_date=date)
    daily = safe_get('daily', trade_date=date)
    return {'adj': adj, 'daily': daily}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    # 稍微多拉几天，确保能取到 D5 的数据
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
    
    # 1. 处理复权因子
    adj_all = pd.concat(adj_list)
    adj_all['adj_factor'] = pd.to_numeric(adj_all['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_all.set_index(['ts_code', 'trade_date']).sort_index()
    
    # 2. 处理日线
    daily_raw = pd.concat(daily_list)
    for c in ['open','high','low','close','pre_close','vol']:
        if c in daily_raw.columns: daily_raw[c] = pd.to_numeric(daily_raw[c], errors='coerce')
    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index()
    
    # 3. 基准复权因子
    latest_dt = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_dt:
        GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.xs(latest_dt, level='trade_date')['adj_factor'].to_dict()
        
    return True

def get_qfq_data(ts_code, start_date, end_date):
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    base = GLOBAL_QFQ_BASE_FACTORS.get(ts_code)
    if not base: return pd.DataFrame()
    
    try:
        # 切片获取数据
        df = GLOBAL_DAILY_RAW.loc[(ts_code, slice(start_date, end_date)), :].copy()
        factors = GLOBAL_ADJ_FACTOR.loc[(ts_code, slice(start_date, end_date)), 'adj_factor']
        if df.empty or factors.empty: return pd.DataFrame()
        
        df = df.join(factors)
        norm = df['adj_factor'] / base
        for c in ['open','high','low','close','pre_close']: df[c] *= norm
        return df.reset_index()
    except: return pd.DataFrame()

# ----------------------------------------------------------------------
# 核心逻辑：接力买入计算
# ----------------------------------------------------------------------
def calculate_relay_trade(ts_code, signal_date, buy_threshold_pct=1.5):
    """
    计算接力策略的收益：
    1. D0: 确认是否触发买入信号 (Open + 1.5%) -> 确定 成本价。
    2. D3: 确认收盘价 > 成本价 -> 触发 接力买入。
    3. D5: 卖出。
    """
    # 获取未来数据 (从 D0 到 D10，足够覆盖)
    d0 = datetime.strptime(signal_date, "%Y%m%d")
    start = d0.strftime("%Y%m%d")
    end = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    df = get_qfq_data(ts_code, start, end)
    if df.empty: return None
    
    # 找到 D0, D1, D2, D3, D5 的数据行
    # 注意：df 包含了 signal_date 当天
    trade_dates = df['trade_date'].tolist()
    if signal_date not in trade_dates: return None
    
    idx_d0 = trade_dates.index(signal_date)
    # 需要至少有 D5 的数据 (即 index + 5)
    if len(df) <= idx_d0 + 5: return None 
    
    row_d0 = df.iloc[idx_d0]
    row_d3 = df.iloc[idx_d0 + 3] # T+3
    row_d5 = df.iloc[idx_d0 + 5] # T+5 (用户逻辑是持有到D5)
    
    # --- 1. D0 原始买入判断 ---
    # 如果低开，直接 Pass
    if row_d0['open'] <= row_d0['pre_close']: return None
    
    # 确认突破 +1.5%
    buy_price_d0 = row_d0['open'] * (1 + buy_threshold_pct/100)
    if row_d0['high'] < buy_price_d0: return None
    
    # --- 2. D3 接力判定 (关键逻辑) ---
    # 条件：D3 收盘价 > D0 买入成本 (即 D0买入者是赚钱的)
    d3_close = row_d3['close']
    
    status = "观望 (D3浮亏)"
    relay_ret = 0.0
    
    if d3_close > buy_price_d0:
        status = "✅ 接力买入"
        # 买入价 = D3 收盘价
        # 卖出价 = D5 收盘价
        relay_ret = (row_d5['close'] / d3_close - 1) * 100
    else:
        status = "❌ 放弃 (D3亏损)"
        relay_ret = 0.0 # 没买，收益为0
        
    return {
        'buy_price_d0': buy_price_d0,
        'd3_close': d3_close,
        'd5_close': row_d5['close'],
        'status': status,
        'relay_return': relay_ret
    }

# ----------------------------------------------------------------------
# V30.22 选股核心
# ----------------------------------------------------------------------
def compute_v3022_score(ts_code, trade_date):
    # 用之前120天数据计算指标
    start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data(ts_code, start, trade_date)
    if df.empty or len(df) < 30: return None
    
    # 1. 硬门槛
    curr = df.iloc[-1]
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    ma5_vol = df['vol'].rolling(5).mean().iloc[-1]
    
    if curr['close'] <= ma20: return None # 没站上20日线
    if curr['vol'] <= ma5_vol * 1.2: return None # 没放量
    
    # 2. MACD (8,17,5)
    close = df['close']
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd = (diff - dea) * 2
    macd_val = macd.iloc[-1]
    
    if macd_val <= 0: return None
    
    # 3. 黄金形态加分
    bonus = 1.0
    price = curr['close']
    pct = (price / curr['pre_close'] - 1) * 100
    volatility = df['close'].pct_change().tail(10).std()*100
    
    tags = []
    if 40 <= price <= 80: 
        bonus += 0.1
        tags.append("价佳")
    if pct > 9.5: 
        bonus += 0.1
        tags.append("板")
    if 4 <= volatility <= 8: 
        bonus += 0.05
        tags.append("波稳")
        
    score = macd_val * 10000 * bonus
    return {'score': score, 'macd': macd_val, 'bonus_tags': "+".join(tags), 'close': price}

# ----------------------------------------------------------------------
# 回测主循环
# ----------------------------------------------------------------------
def run_relay_backtest(trade_days, top_n=4):
    results = []
    progress = st.progress(0, text="正在验证接力策略...")
    
    for i, date in enumerate(trade_days):
        # 1. 基础池
        daily = safe_get('daily', trade_date=date)
        if daily.empty: continue
        
        # 简单过滤
        candidates = daily[~daily['ts_code'].str.startswith('92')] # 排除B股等
        candidates = candidates[~candidates['name'].str.contains('ST|退')]
        
        # 2. 计算分数
        scored_list = []
        # 为了速度，只取成交量大的前200进行计算，或者全算（这里简化全算太慢，取Top）
        # 实盘是全算，回测为了速度取Top200成交额
        candidates['amount'] = pd.to_numeric(candidates['amount'], errors='coerce')
        candidates = candidates.sort_values('amount', ascending=False).head(300)
        
        for ts_code in candidates['ts_code']:
            res = compute_v3022_score(ts_code, date)
            if res:
                res['ts_code'] = ts_code
                res['name'] = candidates.loc[candidates['ts_code']==ts_code, 'name'].values[0]
                scored_list.append(res)
        
        if not scored_list: continue
        
        # 3. 排序 & 截取 Top 4
        df_scored = pd.DataFrame(scored_list)
        df_scored = df_scored.sort_values('score', ascending=False).head(top_n).reset_index(drop=True)
        df_scored['Rank'] = df_scored.index + 1
        
        # 4. 【关键步骤】剔除 Rank 2
        df_final = df_scored[df_scored['Rank'] != 2].copy()
        
        # 5. 接力交易模拟
        for idx, row in df_final.iterrows():
            trade_res = calculate_relay_trade(row['ts_code'], date)
            if trade_res:
                results.append({
                    'Signal_Date': date,
                    'Rank': row['Rank'],
                    'ts_code': row['ts_code'],
                    'Name': row['name'],
                    'D0_Buy_Price': trade_res['buy_price_d0'],
                    'D3_Close': trade_res['d3_close'],
                    'Status': trade_res['status'],
                    'Relay_Return (%)': trade_res['relay_return']
                })
        
        progress.progress((i+1)/len(trade_days))
        
    progress.empty()
    return pd.DataFrame(results)

# ---------------------------
# 侧边栏 & 运行
# ---------------------------
with st.sidebar:
    st.header("参数设置")
    days_back = st.number_input("回测天数", value=100)
    ts_token = st.text_input("Tushare Token", type="password")

if st.button("🚀 开始验证天才想法 (接力策略)"):
    if not ts_token: st.error("请输入 Token"); st.stop()
    ts.set_token(ts_token)
    pro = ts.pro_api()
    
    # 1. 获取日期
    end_date = datetime.now().strftime("%Y%m%d")
    dates = get_trade_days(end_date, int(days_back))
    if not dates: st.stop()
    
    # 2. 准备数据
    if not get_all_historical_data(dates): st.stop()
    
    # 3. 跑回测
    df_res = run_relay_backtest(dates)
    
    if df_res.empty:
        st.warning("没有产生交易信号。")
        st.stop()
        
    # 4. 分析结果
    st.markdown("---")
    st.header("📊 接力策略 (Relay Strategy) 最终战报")
    
    # 区分“买了的”和“没买的”
    df_traded = df_res[df_res['Status'] == '✅ 接力买入']
    df_skipped = df_res[df_res['Status'] != '✅ 接力买入']
    
    col1, col2, col3 = st.columns(3)
    
    # 指标 1: 接力机会占比
    total_signals = len(df_res)
    actual_trades = len(df_traded)
    ratio = actual_trades / total_signals * 100 if total_signals > 0 else 0
    col1.metric("接力开仓率", f"{ratio:.1f}%", f"{actual_trades}/{total_signals} 次信号触发")
    
    # 指标 2: 接力胜率 (D3买->D5卖)
    if actual_trades > 0:
        win_rate = (df_traded['Relay_Return (%)'] > 0).mean() * 100
        avg_ret = df_traded['Relay_Return (%)'].mean()
    else:
        win_rate, avg_ret = 0, 0
    
    col2.metric("接力胜率 (Win Rate)", f"{win_rate:.1f}%", "目标 > 50%")
    col3.metric("接力平均收益", f"{avg_ret:.2f}%", "扣费前")
    
    st.markdown("### 📝 详细交易记录")
    st.dataframe(df_res, use_container_width=True)
    
    st.markdown("""
    **结果解读：**
    * 如果 **接力胜率 < 45%**：说明 D3 追涨是接盘侠，证明“我的朋友”之前的担心是对的。
    * 如果 **接力胜率 > 55%**：说明 D3 确认是黄金机会，你是对的！
    """)
