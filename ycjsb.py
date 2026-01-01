# -*- coding: utf-8 -*-
"""
选股王 · V30.22 终极修正版 (MACD归一化 + 资金共振)
核心逻辑优化：
1. [评分修正] 使用 MACD/Price 归一化评分，消除高价股偏差。
2. [实战过滤] 剔除一字涨停无法买入的情况。
3. [黄金形态] 引入量比(>1.5)和换手率(5-15%)作为核心加分项。
4. [范围放宽] 不再强制锁定 40-80 元，全市场扫描。
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
st.set_page_config(page_title="选股王 · V30.22 终极修正版", layout="wide")
st.title("选股王 · V30.22 终极修正版（⚖️ 评分修正 + 💰 资金共振）")
st.markdown("""
**🛠️ 策略逻辑升级：**
1. **公平评分：** 使用 **MACD / 股价** 进行评分，低价妖股也能上榜。
2. **买入风控：** 自动剔除 **一字涨停** (买不进) 的标的。
3. **资金共振 (利用10000积分数据)：**
    * ✅ **量比 > 1.5** (主力异动)
    * ✅ **换手 5%~15%** (黄金活跃区)
    * ✅ **涨停确认** (涨幅>9.5%)
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
        # 增加重试机制，防止网络抖动
        for _ in range(3):
            try:
                if kwargs.get('is_index'): df = pro.index_daily(**kwargs)
                else: df = func(**kwargs)
                if df is not None and not df.empty:
                    return df
            except:
                continue
        return pd.DataFrame(columns=['ts_code'])
    except Exception: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    # 多取一些天数以防假期
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3 + 30)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历，请检查 Token 或网络。")
        return []
    return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()

# ----------------------------------------------------------------------
# 数据拉取
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    # 拉取行情
    daily_df = safe_get('daily', trade_date=date)
    # 拉取复权因子
    adj_df = safe_get('adj_factor', trade_date=date)
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    # 预留足够的前置数据用于计算MACD (至少120天)
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d") 
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    st.info(f"⏳ 正在拉取全市场数据 ({start_date} ~ {end_date})，请耐心等待...")

    adj_list, daily_list = [], []
    download_progress = st.progress(0, text="数据下载进度...")
    
    # 批量或循环拉取
    total_dates = len(all_dates)
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty: adj_list.append(cached_data['adj'])
            if not cached_data['daily'].empty: daily_list.append(cached_data['daily'])
            
            # 更新进度条
            if i % 5 == 0:
                download_progress.progress((i + 1) / total_dates)
        except: continue 
    download_progress.empty()

    if not adj_list or not daily_list:
        st.error("❌ 无法获取历史数据，请检查您的 Token 权限。")
        return False
        
    # 合并数据
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    # 去重并建立索引
    GLOBAL_ADJ_FACTOR = adj_data.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    cols_to_keep = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol']
    daily_raw = pd.concat(daily_list)
    valid_cols = [c for c in cols_to_keep if c in daily_raw.columns]
    daily_raw = daily_raw[valid_cols]
    
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'vol']:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    # 缓存最新的复权基准
    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        try:
            latest_adj = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj.droplevel(1).to_dict()
        except: GLOBAL_QFQ_BASE_FACTORS = {}
    
    return True

# ----------------------------------------------------------------------
# 数据处理 (前复权)
# ----------------------------------------------------------------------
def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty: return pd.DataFrame()
        
    base_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(base_adj) or base_adj < 1e-9: return pd.DataFrame() 

    try:
        # 利用索引切片加速
        daily = GLOBAL_DAILY_RAW.loc[(ts_code, slice(start_date, end_date)), :]
        adj = GLOBAL_ADJ_FACTOR.loc[(ts_code, slice(start_date, end_date)), 'adj_factor']
    except KeyError: return pd.DataFrame()
    
    if daily.empty or adj.empty: return pd.DataFrame()
    
    # 对齐索引
    df = daily.merge(adj.rename('adj_factor'), left_index=True, right_index=True, how='left').dropna(subset=['adj_factor'])
    
    factor = df['adj_factor'] / base_adj
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns: df[col] = df[col] * factor
    
    df = df.reset_index() # 恢复 ts_code, trade_date 为列
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
    
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    for n in days_ahead: results[f'Return_D{n}'] = np.nan

    if hist.empty: return results
    d1_data = hist.iloc[0]
    
    # 1. 拒绝低开
    if d1_data['open'] <= d1_data['pre_close']: return results 
    
    # 2. [新增] 拒绝一字涨停 (买不进)
    # 判定标准：开盘价 >= 昨收 * 1.095 (科创板需更宽，此处通用保守处理) 且 最低价 == 开盘价
    limit_up_price = d1_data['pre_close'] * 1.095
    if d1_data['open'] >= limit_up_price and d1_data['low'] >= d1_data['open']:
        return results # 一字板，放弃

    # 3. 确认突破阈值 (如 +1.5%)
    buy_price_threshold = d1_data['open'] * (1 + buy_threshold_pct / 100.0)
    
    # 如果最高价都没摸到阈值，没成交
    if d1_data['high'] < buy_price_threshold: return results 

    # 计算收益
    for n in days_ahead:
        idx = n - 1
        if len(hist) > idx:
            # 收益率 = (N日收盘价 / 买入价 - 1) * 100
            results[f'Return_D{n}'] = (hist.iloc[idx]['close'] / buy_price_threshold - 1) * 100
            
    return results

# ----------------------------------------------------------------------
# 指标计算 
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    res = {}
    if df.empty or len(df) < 26: return res
         
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
    
    return res

@st.cache_data(ttl=3600*12)
def get_market_state(trade_date):
    # 简单的大盘状态判断，若大盘跌破20日线则为弱势
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
    BACKTEST_DAYS = int(st.number_input("**回测天数 (N)**", value=30, step=1))
    
    st.markdown("---")
    st.header("2. 实战参数 (V30.22)")
    BUY_THRESHOLD_PCT = st.number_input("买入确认阈值 (%)", value=1.5, step=0.1)
    
    st.markdown("---")
    st.header("3. 基础过滤")
    FINAL_POOL = int(st.number_input("入围数量", value=100)) 
    TOP_BACKTEST = int(st.number_input("Top K", value=5))
    # [修改] 放宽价格限制，避免价格歧视
    MIN_PRICE = st.number_input("最低股价", value=4.0, step=0.5) 
    MAX_PRICE = st.number_input("最高股价", value=600.0, step=10.0)
    MIN_TURNOVER = st.number_input("最低换手 (%)", value=3.0) 
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿)", value=20.0)
    MIN_AMOUNT = st.number_input("最低成交额 (亿)", value=1.0) * 100000000 

TS_TOKEN = st.text_input("Tushare Token (需10000积分)", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ----------------------------------------------------------------------
# 核心逻辑 (V30.22 终极形态)
# ----------------------------------------------------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, buy_threshold):
    # 1. 弱市熔断
    market_state = get_market_state(last_trade)
    if market_state == 'Weak': return pd.DataFrame(), f"弱市避险"

    # 2. 拉取数据
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失"
    pool = daily_all.reset_index(drop=True)
    
    basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date') 
    if not basic.empty: pool = pool.merge(basic, on='ts_code', how='left')
    
    # [关键] 获取每日指标：换手率、量比 (Volume Ratio)
    # volume_ratio 需要 10000 积分或足够权限
    d_basic = safe_get('daily_basic', trade_date=last_trade, fields='ts_code,turnover_rate,circ_mv,total_mv,volume_ratio')
    if not d_basic.empty: pool = pool.merge(d_basic, on='ts_code', how='left')
    
    # 资金流 (可选，辅助)
    mf = safe_get('moneyflow', trade_date=last_trade)
    if not mf.empty and 'net_mf' in mf.columns:
        mf = mf[['ts_code', 'net_mf']].fillna(0)
        pool = pool.merge(mf, on='ts_code', how='left')
    
    # 填充缺失值
    for c in ['turnover_rate','circ_mv','net_mf', 'volume_ratio']: 
        if c not in pool.columns: pool[c] = 0.0
    
    # 3. 基础过滤
    df = pool.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce') 
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000 
    
    # 过滤 ST、退市、北交所(92开头)
    df = df[~df['name'].str.contains('ST|退', case=False, na=False)]
    df = df[~df['ts_code'].str.startswith('92')] # 过滤北交所
    df = df[~df['ts_code'].str.startswith('688')] if False else df # 科创板可选保留
    
    if 'list_date' in df.columns:
        df['days_listed'] = (datetime.strptime(last_trade, "%Y%m%d") - pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')).dt.days
        df = df[df['days_listed'] >= 120]
        
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE) & 
        (df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS) &
        (df['turnover_rate'] >= MIN_TURNOVER) & (df['turnover_rate'] <= 25.0) &
        (df['amount'] * 1000 >= MIN_AMOUNT)]
        
    if len(df) == 0: return pd.DataFrame(), f"过滤后无股票"

    # 初选池逻辑：一半选资金流强，一半选涨幅强
    limit_mf = int(FINAL_POOL * 0.5)
    df_mf = df.sort_values('net_mf', ascending=False).head(limit_mf)
    df_pct = df[~df['ts_code'].isin(df_mf['ts_code'])].sort_values('pct_chg', ascending=False).head(FINAL_POOL - len(df_mf))
    candidates = pd.concat([df_mf, df_pct]).reset_index(drop=True)
    
    # 确保有历史数据
    if not GLOBAL_DAILY_RAW.empty:
        try:
            available = GLOBAL_DAILY_RAW.loc[(slice(None), last_trade), :].index.get_level_values('ts_code').unique()
            candidates = candidates[candidates['ts_code'].isin(available)]
        except: return pd.DataFrame(), "缓存缺失"

    # 4. 深度计算 (硬门槛)
    records = []
    for row in candidates.itertuples():
        ind = compute_indicators(row.ts_code, last_trade) 
        
        # [硬门槛]
        # 必须是上升趋势 (收盘价 > 20日线)
        if ind.get('close_current', 0) <= ind.get('ma20_current', 0): continue
        # 必须放量 (当前量 > 5日均量 * 1.2)
        if ind.get('vol_current', 0) <= ind.get('ma5_vol_current', 0) * 1.2: continue
        # MACD 必须为正
        if pd.isna(ind.get('macd_val')) or ind.get('macd_val') <= 0: continue
        
        # 计算实战买入结果
        future = get_future_prices_real_combat(row.ts_code, last_trade, buy_threshold_pct=buy_threshold)
        
        records.append({
            'ts_code': row.ts_code, 
            'name': getattr(row, 'name', row.ts_code),
            'Close': row.close, 
            'Pct_Chg (%)': getattr(row, 'pct_chg', 0),
            'macd': ind['macd_val'], 
            'volume_ratio': getattr(row, 'volume_ratio', 0),
            'turnover_rate': getattr(row, 'turnover_rate', 0),
            'Return_D1 (%)': future.get('Return_D1'), 
            'Return_D3 (%)': future.get('Return_D3'),
            'Return_D5 (%)': future.get('Return_D5')
        })
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), "无优质放量股票"

    # 5. [终极评分系统]
    
    # A. 基础分修正：(MACD / Close) * 1000000
    # 解决了高价股 MACD 绝对值过大的问题
    fdf['macd_ratio'] = fdf['macd'] / fdf['Close']
    base_score = fdf['macd_ratio'] * 1000000 
    
    # B. 动态加分 (资金共振)
    def calculate_smart_bonus(row):
        bonus = 1.0
        tags = []
        
        # [加分 1] 量能确认：量比 > 1.5 (说明主力资金介入明显)
        if row['volume_ratio'] > 1.5:
            bonus += 0.1
            tags.append('量比佳')
            
        # [加分 2] 换手率黄金区间：5% ~ 15% (活跃但不过热)
        # 替代了原先难用的“波动率”指标
        if 5.0 <= row['turnover_rate'] <= 15.0:
            bonus += 0.1
            tags.append('换手佳')
            
        # [加分 3] 涨停板确认 (>9.5%) -> +10%
        if row['Pct_Chg (%)'] >= 9.5:
            bonus += 0.1
            tags.append('板确认')
            
        return bonus, "+".join(tags)

    fdf[['bonus', '加分项']] = fdf.apply(lambda x: pd.Series(calculate_smart_bonus(x)), axis=1)
    fdf['综合评分'] = base_score * fdf['bonus']
    
    fdf = fdf.sort_values('综合评分', ascending=False).head(TOP_BACKTEST)
    return fdf.reset_index(drop=True), None

# ---------------------------
# 主程序
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日 V30.22 终极回测"):
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days: st.stop()
    if not get_all_historical_data(trade_days): st.stop()
    
    st.success("✅ V30.22 (终极修正版) 启动... 正在全市场扫描")
    results = []
    bar = st.progress(0)
    
    for i, date in enumerate(trade_days):
        try:
            df, msg = run_backtest_for_a_day(date, TOP_BACKTEST, FINAL_POOL, BUY_THRESHOLD_PCT)
            if not df.empty:
                df['Trade_Date'] = date
                results.append(df)
        except Exception: pass
        bar.progress((i + 1) / len(trade_days))
    bar.empty()
    
    if not results:
        st.error("无结果。")
        st.stop()
        
    all_res = pd.concat(results)
    if all_res['Trade_Date'].dtype != 'object': all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
    
    # ---------------------------
    # 结果展示
    # ---------------------------
    st.header(f"📊 V30.22 回测报告 (MACD修正 + 资金共振)")
    st.markdown(f"**有效交易天数：** {all_res['Trade_Date'].nunique()} 天")

    cols = st.columns(3)
    for idx, n in enumerate([1, 3, 5]):
        col = f'Return_D{n} (%)' 
        valid = all_res.dropna(subset=[col])
        if not valid.empty:
            avg_ret = valid[col].mean()
            hit_rate = (valid[col] > 0).sum() / len(valid) * 100
            count = len(valid)
        else: avg_ret, hit_rate, count = 0, 0, 0
        with cols[idx]:
            st.metric(f"D+{n} 收益 / 胜率", f"{avg_ret:.2f}% / {hit_rate:.1f}%", help=f"成交：{count} 笔")
            
    # 下载按钮
    csv = all_res.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 下载完整回测数据 (CSV)", csv, "v30_22_export.csv", "text/csv")

    st.header("📋 每日成交明细")
    st.dataframe(all_res.sort_values('Trade_Date', ascending=False), use_container_width=True)
