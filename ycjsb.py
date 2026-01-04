# -*- coding: utf-8 -*-
"""
选股王 · V30.25 长期压力测试版 (Long-Term Stress Test)
核心改动：仅下载双创数据 (30/688)，大幅减少数据量，从而支持 3-5 年长周期回测。
目标：验证策略穿越牛熊的稳定性。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V30.25 长期压力测试 (双创版)", layout="wide")
st.title("🛡️ V30.25 长期压力测试 (Only 300/688)")
st.markdown("""
**🎯 回测目标：**
* **范围：** 仅限 **创业板 (30)** 和 **科创板 (688)**。
* **周期：** 建议测试 **1000天 (约4年)**，穿越牛熊周期。
* **目的：** 用几百次交易的大样本，验证策略的真实胜率和生存能力。
""")

# ---------------------------
# 全局缓存
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

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
    # 多取一些缓冲天数用于计算指标
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 2)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns: return []
    return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()

# ----------------------------------------------------------------------
# 优化版数据下载 (只下双创)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data_optimized(date):
    # 分别获取，减少无效数据传输
    # Tushare 没有直接按板块取行情的接口，但我们可以取全市场后过滤，或者分板取。
    # 为了兼容性，我们还是取全市场 daily，但在内存中立刻过滤，减少后续处理压力。
    # 更好的方式：daily 接口不支持按板块，但 stock_basic 支持。
    # 这里为了代码简单，我们在下载后立刻 drop 掉非双创的行。
    
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    
    if not daily_df.empty:
        # 只保留 30 和 688 开头
        daily_df = daily_df[daily_df['ts_code'].str.startswith(('30', '688'))]
        
    if not adj_df.empty:
        # 同样过滤复权因子，节省内存
        adj_df = adj_df[adj_df['ts_code'].str.startswith(('30', '688'))]
        
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 向前多取 150 天用于 MACD 计算
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    # 向后多取 20 天用于计算未来收益
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d") 
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    st.info(f"⏳ 正在下载 {start_date} 到 {end_date} 的双创数据 (可能需要几分钟)...")

    adj_list, daily_list = [], []
    bar = st.progress(0)
    
    # 批量下载优化
    # 由于要下载 1000+ 天，按天循环太慢。
    # 我们可以尝试按月或季度下载吗？Tushare daily 接口一次最多 4000-5000 行。
    # 5000只股票一天就超了。所以按天是必须的。
    # 但我们现在只关心双创，约 1500 只股票。
    
    for i, date in enumerate(all_dates):
        try:
            cached = fetch_and_cache_daily_data_optimized(date)
            if not cached['adj'].empty: adj_list.append(cached['adj'])
            if not cached['daily'].empty: daily_list.append(cached['daily'])
            if i % 50 == 0: bar.progress((i+1)/len(all_dates)) # 减少刷新频率
        except: continue 
    bar.empty()

    if not adj_list or not daily_list: return False
        
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    # 建立索引
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw = pd.concat(daily_list)
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'vol']:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32') # 降精度省内存

    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    
    latest_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_date:
        GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_date), 'adj_factor'].droplevel(1).to_dict()
    
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
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    return df.set_index('trade_date').sort_index()

# ----------------------------------------------------------------------
# 核心指标
# ----------------------------------------------------------------------
def compute_score(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data(ts_code, start_date, end_date)
    if df.empty or len(df) < 26: return 0
    
    close = df['close']
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd_val = (diff - dea) * 2
    
    score = (macd_val.iloc[-1] / close.iloc[-1]) * 100000
    if pd.isna(score): score = 0
    return score

# ----------------------------------------------------------------------
# 回测逻辑
# ----------------------------------------------------------------------
def run_backtest_on_date(date, min_price):
    try:
        # 获取当日数据 (已过滤为双创)
        daily = GLOBAL_DAILY_RAW.xs(date, level='trade_date')
    except KeyError:
        return None
        
    if daily.empty: return None
    
    # 1. 价格过滤
    pool = daily[daily['close'] >= min_price]
    # (板块过滤在下载时已做)
    
    if pool.empty: return None
    
    # 2. 粗筛
    pool = pool[pool['pct_chg'] > 0].sort_values('pct_chg', ascending=False)
    if len(pool) > 100: pool = pool.head(100)
    
    best_score = -1
    rank1_code = None
    rank1_close = 0
    
    # 3. 评分
    for row in pool.itertuples():
        # row.Index 是 ts_code (因为 xs(date) 后只剩 ts_code 索引)
        score = compute_score(row.Index, date)
        if score > best_score:
            best_score = score
            rank1_code = row.Index
            rank1_close = row.close
            
    if not rank1_code: return None
    
    # 4. 模拟交易
    d0 = datetime.strptime(date, "%Y%m%d")
    start_fut = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_fut = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    hist = get_qfq_data(rank1_code, start_fut, end_fut)
    
    ret_d1, ret_d3, ret_d5 = np.nan, np.nan, np.nan
    
    if len(hist) >= 1:
        d1_row = hist.iloc[0]
        # 判断 D+1 开盘涨幅
        try:
            d1_raw = GLOBAL_DAILY_RAW.loc[(rank1_code, d1_row.name.strftime("%Y%m%d"))]
            if isinstance(d1_raw, pd.Series):
                open_pct = (d1_raw['open'] / d1_raw['pre_close'] - 1) * 100
            else:
                open_pct = 0
        except:
            open_pct = 0
            
        if open_pct > 1.5:
            buy_price = d1_row['open']
            
            ret_d1 = (d1_row['close'] / buy_price - 1) * 100
            if len(hist) >= 3:
                ret_d3 = (hist.iloc[2]['close'] / buy_price - 1) * 100
            if len(hist) >= 5:
                ret_d5 = (hist.iloc[4]['close'] / buy_price - 1) * 100
            elif len(hist) > 0:
                ret_d5 = (hist.iloc[-1]['close'] / buy_price - 1) * 100
    
    return {
        'Trade_Date': date,
        'ts_code': rank1_code,
        'Return_D1': ret_d1,
        'Return_D3': ret_d3,
        'Return_D5': ret_d5
    }

# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测参数")
    end_date = st.date_input("结束日期", value=datetime.now().date())
    days_back = int(st.number_input("回测天数", value=1000, help="建议输入1000，约4年数据"))
    MIN_PRICE = st.number_input("最低股价 (元)", value=20.0)
    
    st.markdown("---")
    TS_TOKEN = st.text_input("Tushare Token", type="password")

if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ---------------------------
# 主程序
# ---------------------------
if st.button("🚀 启动长期压力测试 (1000天)"):
    dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    if not dates: st.stop()
    # 下载数据
    if not get_all_historical_data(dates): st.stop()
    
    st.success(f"✅ 数据就绪：双创版 (300/688) | 周期: {len(dates)} 天")
    
    results = []
    bar = st.progress(0)
    
    for i, date in enumerate(dates):
        res = run_backtest_on_date(date, MIN_PRICE)
        if res:
            results.append(res)
        if i % 10 == 0: bar.progress((i+1)/len(dates))
    
    bar.empty()
    
    if not results:
        st.warning("无交易记录。")
        st.stop()
        
    df_res = pd.DataFrame(results)
    valid_trades = df_res.dropna(subset=['Return_D1'])
    
    # ---------------------------
    # 长期分析报告
    # ---------------------------
    st.header("📊 V30.25 长期生存报告 (Only 300+688)")
    st.caption(f"回测区间: {dates[-1]} 至 {dates[0]} | 交易次数: {len(valid_trades)}")
    
    col1, col2, col3 = st.columns(3)
    
    def get_metrics(col):
        if valid_trades.empty: return 0, 0
        avg = valid_trades[col].mean()
        win = (valid_trades[col] > 0).mean() * 100
        return avg, win
    
    d1_avg, d1_win = get_metrics('Return_D1')
    d3_avg, d3_win = get_metrics('Return_D3')
    d5_avg, d5_win = get_metrics('Return_D5')
    
    col1.metric("D+1 收益/胜率", f"{d1_avg:.2f}% / {d1_win:.1f}%")
    col2.metric("D+3 收益/胜率", f"{d3_avg:.2f}% / {d3_win:.1f}%")
    col3.metric("D+5 收益/胜率", f"{d5_avg:.2f}% / {d5_win:.1f}%")
    
    # 分年度统计 (看穿越牛熊能力)
    valid_trades['Year'] = pd.to_datetime(valid_trades['Trade_Date']).dt.year
    year_stats = valid_trades.groupby('Year')[['Return_D1', 'Return_D5']].agg(['count', 'mean', lambda x: (x>0).mean()*100])
    st.subheader("📅 分年度表现 (穿越牛熊验证)")
    st.dataframe(year_stats)
    
    # 模拟 Hybrid 资金曲线
    if not valid_trades.empty:
        valid_trades['Return_Hybrid'] = np.where(valid_trades['Return_D3']>0, valid_trades['Return_D5'], valid_trades['Return_D3'])
        equity = (1 + valid_trades['Return_Hybrid']/100).cumprod()
        st.subheader("📈 长期资金曲线 (Hybrid 策略)")
        st.line_chart(equity)

    csv = df_res.to_csv().encode('utf-8')
    st.download_button("📥 下载完整回测数据", csv, "v30.25_long_term_export.csv", "text/csv")
