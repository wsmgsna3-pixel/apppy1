# -*- coding: utf-8 -*-
"""
选股王 · V30.25 核武器长期版 (Only STAR 688 Long-Term)
目标：仅拉取科创板数据，进行 2-3 年的长周期回测。
新增功能：内置“评分(Score)有效性分析”，验证高分是否对应高胜率。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V30.25 核武器长期版", layout="wide")
st.title("🚀 V30.25 核武器长期版 (Only STAR 688)")
st.markdown("""
**🎯 回测目标：**
* **范围：** 仅限 **科创板 (688)**。
* **周期：** 建议 **750天 (约3年)**。
* **核心验证：** 在大样本下，Rank 1 的评分 (Score) 是否是胜率的分水岭？
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
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 2)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns: return []
    return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()

# ----------------------------------------------------------------------
# 极速数据下载 (Only 688)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    
    # 核心优化：只保留 688，极大幅度减少数据量
    if not daily_df.empty:
        daily_df = daily_df[daily_df['ts_code'].str.startswith('688')]
    if not adj_df.empty:
        adj_df = adj_df[adj_df['ts_code'].str.startswith('688')]
        
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d") 
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    st.info(f"⏳ 正在拉取科创板全历史数据 ({start_date} ~ {end_date})...")

    adj_list, daily_list = [], []
    bar = st.progress(0)
    
    for i, date in enumerate(all_dates):
        try:
            cached = fetch_and_cache_daily_data(date)
            if not cached['adj'].empty: adj_list.append(cached['adj'])
            if not cached['daily'].empty: daily_list.append(cached['daily'])
            # 688数据量小，可以刷新快一点
            if i % 50 == 0: bar.progress((i+1)/len(all_dates))
        except: continue 
    bar.empty()

    if not adj_list or not daily_list: return False
        
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw = pd.concat(daily_list)
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'vol']:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

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
# 评分逻辑
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
# 回测主逻辑
# ----------------------------------------------------------------------
def run_backtest_on_date(date, min_price):
    try:
        daily = GLOBAL_DAILY_RAW.xs(date, level='trade_date')
    except KeyError: return None
    if daily.empty: return None
    
    # 1. 价格过滤
    pool = daily[daily['close'] >= min_price]
    # (数据源本身已是纯血科创，无需再过滤板块)
    
    if pool.empty: return None
    
    # 2. 粗筛
    pool = pool[pool['pct_chg'] > 0].sort_values('pct_chg', ascending=False)
    if len(pool) > 80: pool = pool.head(80)
    
    best_score = -1
    rank1_code = None
    rank1_close = 0
    
    # 3. 计算 Rank 1
    for row in pool.itertuples():
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
        try:
            d1_raw = GLOBAL_DAILY_RAW.loc[(rank1_code, d1_row.name.strftime("%Y%m%d"))]
            if isinstance(d1_raw, pd.Series):
                open_pct = (d1_raw['open'] / d1_raw['pre_close'] - 1) * 100
            else: open_pct = 0
        except: open_pct = 0
            
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
        'Close': rank1_close,
        'Score': best_score,
        'Return_D1': ret_d1,
        'Return_D3': ret_d3,
        'Return_D5': ret_d5
    }

# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测设置")
    end_date = st.date_input("结束日期", value=datetime.now().date())
    days_back = int(st.number_input("回测天数", value=750, help="750天约等于3年")) 
    
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
if st.button("🚀 启动科创板长期压力测试"):
    dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    if not dates: st.stop()
    if not get_all_historical_data(dates): st.stop()
    
    st.success(f"✅ 数据就绪：Only 688 | 周期: {len(dates)} 天")
    
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
    # 结果分析
    # ---------------------------
    st.header("📊 科创板长期生存报告 (Only 688)")
    st.caption(f"回测区间: {dates[-1]} 至 {dates[0]} | 交易次数: {len(valid_trades)}")
    
    col1, col2, col3 = st.columns(3)
    def get_m(col):
        if valid_trades.empty: return 0, 0
        return valid_trades[col].mean(), (valid_trades[col]>0).mean()*100
    
    d1_a, d1_w = get_m('Return_D1')
    d3_a, d3_w = get_m('Return_D3')
    d5_a, d5_w = get_m('Return_D5')
    
    col1.metric("D+1 收益/胜率", f"{d1_a:.2f}% / {d1_w:.1f}%")
    col2.metric("D+3 收益/胜率", f"{d3_a:.2f}% / {d3_w:.1f}%")
    col3.metric("D+5 收益/胜率", f"{d5_a:.2f}% / {d5_w:.1f}%")
    
    # --- 评分有效性分析 (核心功能) ---
    st.markdown("---")
    st.subheader("🔍 评分(Score)有效性验证")
    st.markdown("将所有交易按分数分为 4 组 (Q1低分 -> Q4高分)，查看高分是否真的对应高胜率：")
    
    if len(valid_trades) >= 4:
        valid_trades['Score_Group'] = pd.qcut(valid_trades['Score'], 4, labels=['Q1 (低分区)', 'Q2 (中低区)', 'Q3 (中高区)', 'Q4 (高分区)'])
        score_stats = valid_trades.groupby('Score_Group')[['Return_D1', 'Return_D5']].agg(['count', 'mean', lambda x: (x>0).mean()*100])
        score_stats.columns = ['交易次数', 'D1平均收益', 'D1胜率', 'D5交易次数', 'D5平均收益', 'D5胜率']
        # 简化展示
        score_show = score_stats[['交易次数', 'D1平均收益', 'D1胜率', 'D5平均收益', 'D5胜率']]
        st.dataframe(score_show.style.format("{:.2f}"))
        
        # 寻找最佳阈值
        median_score = valid_trades['Score'].median()
        high_score_trades = valid_trades[valid_trades['Score'] > median_score]
        high_win = (high_score_trades['Return_D1'] > 0).mean() * 100
        high_ret = high_score_trades['Return_D1'].mean()
        
        st.info(f"💡 **数据洞察：** 如果只做分数高于中位数 ({median_score:.0f}) 的交易：\n"
                f"- D+1 胜率将变为 **{high_win:.1f}%**\n"
                f"- D+1 平均收益将变为 **{high_ret:.2f}%**")
    else:
        st.warning("交易样本太少，无法进行分组分析。")
        
    st.subheader("📋 详细交易单")
    # 格式化
    display_df = df_res.copy()
    cols_to_round = ['Close', 'Score', 'Return_D1', 'Return_D3', 'Return_D5']
    display_df[cols_to_round] = display_df[cols_to_round].round(2)
    st.dataframe(display_df, use_container_width=True)
    
    csv = df_res.to_csv().encode('utf-8')
    st.download_button("📥 下载完整数据", csv, "star_long_term_export.csv", "text/csv")
