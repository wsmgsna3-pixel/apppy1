# -*- coding: utf-8 -*-
"""
选股王 · D3 接力策略 (纯净独立版)
功能：单独验证“D3买入法” (D3赚钱才买，D5卖出)。
修复：解决了日期格式导致的 AttributeError 报错。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="D3 接力策略独立验证", layout="wide")
st.title("🧪 D3 接力策略独立验证 (D3买 -> D5卖)")
st.markdown("""
**📝 策略逻辑 (您的天才想法)：**
1. **D0 (选股日)：** 选中 Top 4 (剔除 Rank 2)，记录 **模拟买入价** (开盘+1.5%)。
2. **D3 (决策日)：** 观察收盘价。
   - 🔴 如果 **D3收盘价 > D0模拟买入价** (说明是赢家) -> **买入**。
   - ⚪ 如果 **D3收盘价 <= D0模拟买入价** (说明是输家) -> **放弃**。
3. **D5 (卖出日)：** 收盘卖出。
""")

# 全局变量
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

# ---------------------------
# 基础工具
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
    # 多取一些日子以防假期
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 5)).strftime("%Y%m%d")
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
    
    latest = max(trade_days_list) 
    earliest = min(trade_days_list)
    # 往后多取25天，确保能拿到 D5 的数据
    start_dt = (datetime.strptime(earliest, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_dt = (datetime.strptime(latest, "%Y%m%d") + timedelta(days=40)).strftime("%Y%m%d") 
    
    dates = safe_get('trade_cal', start_date=start_dt, end_date=end_dt, is_open='1')['cal_date'].tolist()
    
    progress = st.progress(0, text="正在加载数据 (请耐心等待)...")
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
    for c in ['open','high','low','close','pre_close','vol','amount']:
        if c in daily_raw.columns: daily_raw[c] = pd.to_numeric(daily_raw[c], errors='coerce')
    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index()
    
    latest_dt = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_dt:
        try:
            GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.xs(latest_dt, level='trade_date')['adj_factor'].to_dict()
        except: GLOBAL_QFQ_BASE_FACTORS = {}
        
    return True

def get_qfq_data(ts_code, start_date, end_date):
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    base = GLOBAL_QFQ_BASE_FACTORS.get(ts_code)
    if not base: return pd.DataFrame()
    
    try:
        # 这里的切片依赖索引排序
        df = GLOBAL_DAILY_RAW.loc[(ts_code, slice(start_date, end_date)), :].copy()
        factors = GLOBAL_ADJ_FACTOR.loc[(ts_code, slice(start_date, end_date)), 'adj_factor']
        if df.empty or factors.empty: return pd.DataFrame()
        
        df = df.join(factors)
        norm = df['adj_factor'] / base
        for c in ['open','high','low','close','pre_close']: df[c] *= norm
        
        # 修复：直接重置索引，trade_date 变成字符串列
        return df.reset_index() 
    except: return pd.DataFrame()

# ----------------------------------------------------------------------
# 核心逻辑：D3 接力计算 (修复日期Bug版)
# ----------------------------------------------------------------------
def calculate_d3_relay(ts_code, selection_date, buy_threshold=1.5):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start = d0.strftime("%Y%m%d")
    end = (d0 + timedelta(days=25)).strftime("%Y%m%d") # 足够覆盖 D5
    
    df = get_qfq_data(ts_code, start, end)
    if df.empty: return None
    
    # 获取日期列表 (已经是字符串格式，无需 .dt.strftime)
    trade_dates = df['trade_date'].tolist()
    if selection_date not in trade_dates: return None
    
    idx_0 = trade_dates.index(selection_date)
    
    # 检查是否有足够的未来数据 (至少要有 D5, 即 idx+5)
    if len(df) <= idx_0 + 5: return None
    
    row_d0 = df.iloc[idx_0]
    row_d3 = df.iloc[idx_0 + 3]
    row_d5 = df.iloc[idx_0 + 5]
    
    # 1. 计算 D0 模拟买入价 (门槛)
    if row_d0['open'] <= row_d0['pre_close']: return None # 低开过滤
    
    buy_price_d0 = row_d0['open'] * (1 + buy_threshold/100)
    if row_d0['high'] < buy_price_d0: return None # 没触发
    
    # 2. D3 接力判定
    # 只有当 D3收盘价 > D0买入价 (即目前是赚钱的) 才买
    if row_d3['close'] > buy_price_d0:
        status = "买入"
        buy_price_d3 = row_d3['close']
        sell_price_d5 = row_d5['close']
        
        # 收益率 = (D5卖出 / D3买入 - 1)
        profit = (sell_price_d5 / buy_price_d3 - 1) * 100
        
        return {
            'Status': status,
            'D3_Buy_Price': buy_price_d3,
            'D5_Sell_Price': sell_price_d5,
            'Relay_Return': profit,
            'D0_Simulated_Cost': buy_price_d0
        }
    else:
        # D3 亏损，不接力
        return {
            'Status': '观望',
            'D3_Buy_Price': np.nan,
            'D5_Sell_Price': np.nan,
            'Relay_Return': np.nan,
            'D0_Simulated_Cost': buy_price_d0
        }

# ----------------------------------------------------------------------
# 选股
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
# 主回测循环
# ----------------------------------------------------------------------
def run_solo_backtest(dates):
    results = []
    bar = st.progress(0, text="正在验证您的天才想法...")
    
    # 获取股票名称 (一次性获取，防止循环中调用慢)
    basic = safe_get('stock_basic', list_status='L', fields='ts_code,name')
    
    for i, date in enumerate(dates):
        daily = safe_get('daily', trade_date=date)
        if daily.empty: continue
        
        # 合并名称
        if basic.empty: daily['name'] = 'Unknown'
        else: daily = daily.merge(basic, on='ts_code', how='left')
        
        # 简单过滤
        candidates = daily[~daily['name'].str.contains('ST|退', na=False)]
        candidates = candidates[~candidates['ts_code'].str.startswith('92')]
        
        # 仅计算 Top 300 活跃股，加速
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
        
        # 选出 Top 4
        df_day = pd.DataFrame(scores).sort_values('score', ascending=False).head(4).reset_index(drop=True)
        df_day['Rank'] = df_day.index + 1
        
        # 剔除 Rank 2
        df_final = df_day[df_day['Rank'] != 2].copy()
        
        # 计算接力收益
        for _, row in df_final.iterrows():
            res = calculate_d3_relay(row['ts_code'], date)
            if res:
                rec = row.to_dict()
                rec.update(res)
                rec['Signal_Date'] = date
                results.append(rec)
        
        bar.progress((i+1)/len(dates))
        
    bar.empty()
    return pd.DataFrame(results)

# ---------------------------
# 侧边栏 & 启动
# ---------------------------
with st.sidebar:
    st.header("参数设置")
    days_back = st.number_input("回测天数", value=100)
    ts_token = st.text_input("Tushare Token", type="password")

if st.button("🚀 开始验证 (D3接力法)"):
    if not ts_token: st.error("请填入 Token"); st.stop()
    ts.set_token(ts_token)
    pro = ts.pro_api()
    
    end_date = datetime.now().strftime("%Y%m%d")
    dates = get_trade_days(end_date, int(days_back))
    if not dates: st.stop()
    
    if not get_all_historical_data(dates): st.stop()
    
    df = run_solo_backtest(dates)
    
    if df.empty: st.warning("没有产生信号"); st.stop()
    
    # --- 统计结果 ---
    st.markdown("---")
    st.header("🧪 D3 接力策略 · 实测结果")
    
    # 筛选出真正买入的交易 (Status == '买入')
    trades = df[df['Status'] == '买入'].copy()
    
    total_signals = len(df) # 总共触发选股次数 (包括观望的)
    executed_trades = len(trades) # 实际 D3 接力次数
    
    col1, col2, col3 = st.columns(3)
    
    # 1. 胜率
    win_rate = 0
    if executed_trades > 0:
        win_rate = (trades['Relay_Return'] > 0).mean() * 100
        avg_ret = trades['Relay_Return'].mean()
        
        # 简单年化计算
        daily_ret = trades.groupby('Signal_Date')['Relay_Return'].mean()
        dates_idx = pd.to_datetime(daily_ret.index)
        days_span = (dates_idx.max() - dates_idx.min()).days
        if days_span > 0:
            cagr = ((1 + avg_ret/100 * executed_trades/days_span) ** 250 - 1) * 100 # 粗略估算
        else: cagr = 0
    else:
        avg_ret = 0
        cagr = 0
        
    col1.metric("接力胜率 (Win Rate)", f"{win_rate:.1f}%", f"基准线: 50%")
    col2.metric("接力平均收益 (每笔)", f"{avg_ret:.2f}%", "D3买->D5卖")
    col3.metric("接力开仓率", f"{executed_trades}/{total_signals}", "符合接力条件的比例")
    
    st.info(f"""
    **结果解读：**
    * 您原本的选股产生了 {total_signals} 次机会。
    * 其中有 {executed_trades} 次在 D3 确认盈利，触发了您的接力买入。
    * 这 {executed_trades} 次接力操作，最终只有 {win_rate:.1f}% 是赚钱出来的。
    """)
    
    st.markdown("### 📋 详细交易记录")
    st.dataframe(df[['Signal_Date', 'Rank', 'ts_code', 'name', 'Status', 'D0_Simulated_Cost', 'D3_Buy_Price', 'Relay_Return']], use_container_width=True)
