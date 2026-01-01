# -*- coding: utf-8 -*-
"""
选股王 · V30.22 实战狙击版 (Rank 1/3/4 战法)
核心逻辑：
1. [基石] MACD(8,17,5) + 放量(>1.2倍) + 趋势(>MA20)。
2. [加分] 价格舒适区(40-80) / 涨停确认 / 波动率适中。
3. [实战] 
   - 只看 Top 4。
   - 坚决剔除 Rank 2 (胜率低) 和 Rank 5 (垃圾)。
   - 买入信号：突破 (开盘价 + 1.5%)。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V30.22 实战狙击版", layout="wide")
st.title("🏹 选股王 · V30.22 实战狙击版 (Rank 1/3/4 战法)")
st.markdown("""
**⚔️ 今日实战纪律：**
1. **目标：** 只看下方列表中的 **Rank 1, 3, 4** (已自动剔除 Rank 2 和 5)。
2. **清洗：** 9:25 竞价后，**删除所有低开 (<昨收)** 的股票。
3. **买入：** 盘中价格突破 **【狙击价】(开盘价+1.5%)** 时，果断买入。
4. **持仓：** - D2: 低开直接跑，高开持有。
   - D3: 收盘前浮盈>0则死拿(博D5)，浮亏则清仓。
""")

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
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历。")
        return []
    return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()

# ----------------------------------------------------------------------
# 数据拉取
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d") 
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    
    adj_list, daily_list = [], []
    download_progress = st.progress(0, text="正在加载历史数据...")
    
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty: adj_list.append(cached_data['adj'])
            if not cached_data['daily'].empty: daily_list.append(cached_data['daily'])
            download_progress.progress((i + 1) / len(all_dates))
        except: continue 
    download_progress.empty()

    if not adj_list or not daily_list:
        st.error("无法获取历史数据。")
        return False
        
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw = pd.concat(daily_list)
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'vol']:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        try:
            latest_adj = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj.droplevel(1).to_dict()
        except: GLOBAL_QFQ_BASE_FACTORS = {}
    
    return True

# ----------------------------------------------------------------------
# 数据处理
# ----------------------------------------------------------------------
def get_qfq_data(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty: return pd.DataFrame()
        
    base_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(base_adj) or base_adj < 1e-9: return pd.DataFrame() 

    try:
        # 使用切片加速
        daily = GLOBAL_DAILY_RAW.loc[(ts_code, slice(start_date, end_date)), :]
        adj = GLOBAL_ADJ_FACTOR.loc[(ts_code, slice(start_date, end_date)), 'adj_factor']
    except KeyError: return pd.DataFrame()
    
    if daily.empty or adj.empty: return pd.DataFrame()
    
    # 既然已经按索引对齐，可以直接join
    df = daily.join(adj)
    df = df.dropna(subset=['adj_factor'])
    
    factor = df['adj_factor'] / base_adj
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns: df[col] = df[col] * factor
    
    df = df.reset_index()
    df['trade_date_str'] = df['trade_date']
    df['trade_date'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')
    return df.sort_values('trade_date').set_index('trade_date_str')[['open', 'high', 'low', 'close', 'pre_close', 'vol']]

# ----------------------------------------------------------------------
# 核心买入计算 (实盘严选)
# ----------------------------------------------------------------------
def get_future_prices_real_combat(ts_code, selection_date, days_ahead=[1, 3, 5], buy_threshold_pct=1.5):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=25)).strftime("%Y%m%d")
    
    hist = get_qfq_data(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    for n in days_ahead: results[f'Return_D{n}'] = np.nan

    if hist.empty: return results
    d1_data = hist.iloc[0]
    
    # 1. 拒绝低开
    if d1_data['open'] <= d1_data['pre_close']: return results 
    
    # 2. 确认 +1.5%
    buy_price_threshold = d1_data['open'] * (1 + buy_threshold_pct / 100.0)
    if d1_data['high'] < buy_price_threshold: return results 

    for n in days_ahead:
        idx = n - 1
        if len(hist) > idx:
            results[f'Return_D{n}'] = (hist.iloc[idx]['close'] / buy_price_threshold - 1) * 100
            
    return results

# ----------------------------------------------------------------------
# 指标计算 (V30.22 核心)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data(ts_code, start_date=start_date, end_date=end_date)
    res = {}
    if df.empty or len(df) < 30: return res
         
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    close = df['close']
    vol = df['vol']
    
    # 1. 改进版 MACD (8, 17, 5)
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd_val = (diff - dea) * 2
    res['macd_val'] = macd_val.iloc[-1]
    
    # 2. 均线/量能/其他特征
    ma20 = close.rolling(window=20).mean()
    ma5_vol = vol.rolling(window=5).mean()
    
    res['close_current'] = close.iloc[-1]
    res['ma20_current'] = ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else 0
    res['vol_current'] = vol.iloc[-1]
    res['ma5_vol_current'] = ma5_vol.iloc[-1] if not pd.isna(ma5_vol.iloc[-1]) else 0
    res['pct_chg_current'] = df['pct_chg'].iloc[-1]
    res['pre_close'] = df['pre_close'].iloc[-1] # 用于计算明天的涨停价
    
    # 波动率
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    return res

@st.cache_data(ttl=3600*12)
def get_market_state(trade_date):
    start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    index_data = safe_get('daily', ts_code='000300.SH', start_date=start_date, end_date=trade_date, is_index=True)
    if index_data.empty or len(index_data) < 20: return 'Weak'
    index_data = index_data.sort_values('trade_date')
    return 'Strong' if index_data.iloc[-1]['close'] > index_data['close'].tail(20).mean() else 'Weak'
      
# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测设置")
    backtest_date_end = st.date_input("回测结束日期", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("**回测天数 (N)**", value=1, step=1)) # 默认为1，方便每天看最新信号
    
    st.markdown("---")
    st.header("2. 实战参数")
    BUY_THRESHOLD_PCT = 1.5
    st.info(f"买入阈值: 开盘价 + {BUY_THRESHOLD_PCT}%")
    
    st.markdown("---")
    st.header("3. 基础过滤")
    FINAL_POOL = 100
    TOP_BACKTEST = 4 # 只看 Top 4
    MIN_PRICE = st.number_input("最低股价", value=10.0, step=0.5) 
    MAX_PRICE = st.number_input("最高股价", value=300.0, step=5.0)
    MIN_TURNOVER = st.number_input("最低换手 (%)", value=3.0) 
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿)", value=20.0)
    MIN_AMOUNT = st.number_input("最低成交额 (亿)", value=1.0) * 100000000 

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ----------------------------------------------------------------------
# 核心逻辑 (V30.22 终极形态)
# ----------------------------------------------------------------------
def run_backtest_for_a_day(last_trade, top_n, pool_size, buy_threshold):
    # 1. 弱市熔断
    market_state = get_market_state(last_trade)
    if market_state == 'Weak': return pd.DataFrame(), f"⚠️ 弱市避险 (指数跌破20日线)"

    # 2. 拉取数据
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失"
    
    # 提前获取 name，防止 KeyError
    basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    if basic.empty:
        pool = daily_all.copy()
        pool['name'] = 'Unknown'
    else:
        pool = daily_all.merge(basic, on='ts_code', how='left')
    
    d_basic = safe_get('daily_basic', trade_date=last_trade, fields='ts_code,turnover_rate,circ_mv')
    if not d_basic.empty: pool = pool.merge(d_basic, on='ts_code', how='left')
    
    # 资金流
    mf = safe_get('moneyflow', trade_date=last_trade)
    if not mf.empty and 'net_mf' in mf.columns:
        mf = mf[['ts_code', 'net_mf']].fillna(0)
        pool = pool.merge(mf, on='ts_code', how='left')
    
    for c in ['turnover_rate','circ_mv','net_mf']: 
        if c not in pool.columns: pool[c] = 0.0
    
    # 3. 基础过滤
    df = pool.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce') 
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000 
    df = df[~df['name'].str.contains('ST|退', case=False, na=False)]
    df = df[~df['ts_code'].str.startswith('92')]
    if 'list_date' in df.columns:
        df['days_listed'] = (datetime.strptime(last_trade, "%Y%m%d") - pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')).dt.days
        df = df[df['days_listed'] >= 120]
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE) & 
        (df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS) &
        (df['turnover_rate'] >= MIN_TURNOVER) & (df['turnover_rate'] <= 25.0) &
        (df['amount'] * 1000 >= MIN_AMOUNT)]
    
    if len(df) == 0: return pd.DataFrame(), f"过滤后无股票"

    limit_mf = int(pool_size * 0.5)
    df_mf = df.sort_values('net_mf', ascending=False).head(limit_mf)
    df_pct = df[~df['ts_code'].isin(df_mf['ts_code'])].sort_values('pct_chg', ascending=False).head(pool_size - len(df_mf))
    candidates = pd.concat([df_mf, df_pct]).reset_index(drop=True)

    # 4. 深度计算
    records = []
    for row in candidates.itertuples():
        ind = compute_indicators(row.ts_code, last_trade) 
        if not ind: continue

        # 硬门槛
        if ind.get('close_current', 0) <= ind.get('ma20_current', 0): continue
        if ind.get('vol_current', 0) <= ind.get('ma5_vol_current', 0) * 1.2: continue
        if pd.isna(ind.get('macd_val')) or ind.get('macd_val') <= 0: continue
        
        future = get_future_prices_real_combat(row.ts_code, last_trade, buy_threshold_pct=buy_threshold)
        
        records.append({
            'ts_code': row.ts_code, 
            '名称': getattr(row, 'name', row.ts_code),
            '收盘价': row.close, 
            '涨幅 (%)': getattr(row, 'pct_chg', 0),
            'MACD值': ind['macd_val'], 
            'volatility': ind['volatility'],
            'Return_D1': future.get('Return_D1'), 
            'Return_D3': future.get('Return_D3'),
            'Return_D5': future.get('Return_D5')
        })
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), "无符合条件的股票"

    # 5. 终极评分 (V30.22)
    base_score = fdf['MACD值'] * 10000 
    
    def calculate_smart_bonus(row):
        bonus = 1.0
        tags = []
        # A. 价格舒适区
        if 40 <= row['收盘价'] <= 80:
            bonus += 0.1
            tags.append('价佳')
        # B. 涨停确认
        if row['涨幅 (%)'] >= 9.5:
            bonus += 0.1
            tags.append('板')
        # C. 波动适中
        if 4.0 <= row['volatility'] <= 8.0:
            bonus += 0.05
            tags.append('波稳')
            
        return bonus, "+".join(tags)

    fdf[['bonus', '加分项']] = fdf.apply(lambda x: pd.Series(calculate_smart_bonus(x)), axis=1)
    fdf['综合评分'] = base_score * fdf['bonus']
    
    # 截取前4名，并标记Rank
    fdf = fdf.sort_values('综合评分', ascending=False).head(4).reset_index(drop=True)
    fdf['Rank'] = fdf.index + 1
    
    return fdf, None

# ---------------------------
# 主程序
# ---------------------------
if st.button(f"🚀 生成实战狙击名单 (Rank 1/3/4)"):
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days: st.stop()
    if not get_all_historical_data(trade_days): st.stop()
    
    st.success(f"✅ V30.22 实战计算完成！日期：{trade_days[0]}")
    
    for i, date in enumerate(trade_days):
        df, msg = run_backtest_for_a_day(date, TOP_BACKTEST, FINAL_POOL, BUY_THRESHOLD_PCT)
        
        if not df.empty:
            st.markdown(f"### 📅 {date} 狙击名单")
            
            # --- 实战增强显示 ---
            # 1. 高亮显示 Rank 2 (警告)
            def highlight_rows(row):
                if row['Rank'] == 2:
                    return ['background-color: #ffe6e6'] * len(row) # 红色警告背景
                elif row['Rank'] in [1, 3, 4]:
                    return ['background-color: #e6fffa'] * len(row) # 绿色推荐背景
                return [''] * len(row)

            # 2. 计算狙击价 (明天开盘价预估 = 今天收盘价，实战中看开盘价)
            # 这里显示的是“基于今日收盘的参考狙击价”
            df['参考狙击价(+1.5%)'] = df['收盘价'] * 1.015
            
            # 3. 整理列顺序
            cols = ['Rank', 'ts_code', '名称', '收盘价', '涨幅 (%)', '加分项', '参考狙击价(+1.5%)', 'Return_D1', 'Return_D3']
            df_display = df[cols].copy()
            
            # 格式化
            st.dataframe(df_display.style.apply(highlight_rows, axis=1).format({
                '收盘价': '{:.2f}', 
                '涨幅 (%)': '{:.2f}', 
                '参考狙击价(+1.5%)': '{:.2f}',
                'Return_D1': '{:.2f}%',
                'Return_D3': '{:.2f}%'
            }), use_container_width=True)
            
            st.warning("🚨 **注意：** 表格中 **红色背景 (Rank 2)** 的股票请**坚决剔除**！只关注 **绿色背景 (Rank 1, 3, 4)** 的股票！")
            
        else:
            st.info(f"{date}: {msg}")
