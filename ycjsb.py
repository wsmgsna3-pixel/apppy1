# -*- coding: utf-8 -*-
"""
选股王 · V9.0 极限防御版 (批量获取 + 极度偏爱流动性与低波动)
说明：
1. 【性能保持】保留 V5.0 批量获取数据逻辑，确保回测速度在秒级/分钟低端。
2. 【策略更新】实施 V9.0 极限防御策略：
    - 保持 V6.0 严格风控参数 (波动率、换手率等)。
    - **核心调整：** 将换手率和低波动率权重推向极致 (0.35, 0.25)，极度弱化所有动量因子，以求在弱势市场中找到绝对安全的股票。
3. 【稳定性】使用 Joblib 磁盘缓存 + Streamlit Session State 断点续传。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import joblib 
import os

warnings.filterwarnings("ignore")

# ---------------------------
# 外部缓存配置 (用于历史数据)
# ---------------------------
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
# joblib 封装 Tushare 接口，实现磁盘持久化缓存
memory = joblib.Memory(CACHE_DIR, verbose=0)

# ---------------------------
# 页面设置 (UI 空间最大化)
# ---------------------------
st.set_page_config(page_title="选股王（V9.0 极限防御版）", layout="wide")
st.markdown("### 选股王（V9.0 极限防御版）") 

# ---------------------------
# 侧边栏参数（V9.0 策略：沿用 V6.0 的严格防御默认值）
# ---------------------------
with st.sidebar:
    st.header("可调参数（实时）")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=500, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    
    # V6.0/V9.0 调整：严格收紧流动性要求 (3.5%)
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=3.5, step=0.5)) 
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=150_000_000.0, step=50_000_000.0))
    
    # V6.0/V9.0 调整：严格收紧风险过滤
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.4, step=0.1)) 
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=6.0, step=0.5)) 
    
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    
    BACKTEST_DAYS = int(st.number_input("回测：最近 N 个交易日", value=10, step=1))
    st.markdown("---")
    st.caption("提示：策略已调整至 '极限防御' 模式。")

# ---------------------------
# Token 输入
# ---------------------------
st.markdown("请输入 Tushare Token。")
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password", label_visibility="collapsed")

if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

# 初始化 tushare
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 依赖函数：数据安全获取和交易日查找
# ---------------------------
def safe_get(func, **kwargs):
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def find_last_trade_day(max_days=20):
    today = datetime.now().date()
    for i in range(max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        df = safe_get(pro.daily, trade_date=ds)
        if not df.empty:
            return ds
    return None

last_trade = find_last_trade_day()
if not last_trade:
    st.error("无法找到最近交易日，检查网络或 Token 权限。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")


# ----------------------------------------------------
# V5.0/V9.0 核心加速函数：批量获取历史数据
# ----------------------------------------------------
@memory.cache
def bulk_fetch_daily(ts_codes, start_date, end_date):
    """批量获取多支股票在一段时间内的日线数据"""
    if not ts_codes or not start_date or not end_date:
        return pd.DataFrame()
        
    all_data = []
    # 限制每次查询的股票数量，避免 URL 过长或查询超时
    chunk_size = 50 
    
    # 显示加载进度条
    fetch_bar = st.progress(0.0, text=f"正在批量获取 {len(ts_codes)} 支股票历史数据...")
    
    for i in range(0, len(ts_codes), chunk_size):
        chunk_codes = ts_codes[i:i + chunk_size]
        ts_code_list = ','.join(chunk_codes)
        
        # 使用 pro.daily 接口的 ts_code 参数批量查询
        df = safe_get(pro.daily, ts_code=ts_code_list, start_date=start_date, end_date=end_date)
        if not df.empty:
            all_data.append(df)
            
        fetch_bar.progress((i + len(chunk_codes)) / len(ts_codes), text=f"批量获取进度：{i + len(chunk_codes)}/{len(ts_codes)}...")

    if not all_data:
        fetch_bar.empty()
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    fetch_bar.empty()
    return final_df

# ----------------------------------------------------
# 按钮控制模块 (Session State 断点续传)
# ----------------------------------------------------
if 'run_selection' not in st.session_state: st.session_state['run_selection'] = False
if 'run_backtest' not in st.session_state: st.session_state['run_backtest'] = False
if 'backtest_status' not in st.session_state: 
    st.session_state['backtest_status'] = {'progress': 0.0, 'results': [], 'current_index': 0, 'total_days': 0, 'bulk_data': None}

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 运行当日选股", use_container_width=True):
        st.session_state['run_selection'] = True
        st.session_state['run_backtest'] = False
        st.session_state['backtest_status'] = {'progress': 0.0, 'results': [], 'current_index': 0, 'total_days': 0, 'bulk_data': None}
        st.rerun()

with col2:
    if st.button(f"✅ 运行历史回测 ({BACKTEST_DAYS} 日)", use_container_width=True):
        st.session_state['run_backtest'] = True
        st.session_state['run_selection'] = False
        if st.session_state['backtest_status']['progress'] == 1.0 or st.session_state['backtest_status']['total_days'] == 0:
             st.session_state['backtest_status'] = {'progress': 0.0, 'results': [], 'current_index': 0, 'total_days': 0, 'bulk_data': None}
        st.rerun()

st.markdown("---")

# ---------------------------
# 指标计算和归一化
# ---------------------------
def compute_indicators(df):
    res = {}
    if df.empty or len(df) < 3: return res
    close = df['close'].astype(float); high = df['high'].astype(float); low = df['low'].astype(float)
    try: res['last_close'] = close.iloc[-1]
    except: res['last_close'] = np.nan
    for n in (5,10,20):
        if len(close) >= n: res[f'ma{n}'] = close.rolling(window=n).mean().iloc[-1]
        else: res[f'ma{n}'] = np.nan
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_val = (diff - dea) * 2
        res['macd'] = macd_val.iloc[-1]; res['diff'] = diff.iloc[-1]; res['dea'] = dea.iloc[-1]
    else: res['macd'] = res['diff'] = res['dea'] = np.nan
    n = 9
    if len(close) >= n:
        low_n = low.rolling(window=n).min()
        high_n = high.rolling(window=n).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
        rsv = rsv.fillna(50)
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3*k - 2*d
        res['k'] = k.iloc[-1]; res['d'] = d.iloc[-1]; res['j'] = j.iloc[-1]
    else: res['k'] = res['d'] = res['j'] = np.nan
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        avg_prev5 = np.mean(vols[-6:-1])
        res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9)
        res['vol_last'] = vols[-1]; res['vol_ma5'] = avg_prev5
    else: res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan
    if len(close) >= 10: res['10d_return'] = close.iloc[-1] / close.iloc[-10] - 1
    else: res['10d_return'] = np.nan
    if 'pct_chg' in df.columns and len(df) >= 4:
        try: res['prev3_sum'] = df['pct_chg'].astype(float).iloc[-4:-1].sum()
        except: res['prev3_sum'] = np.nan
    else: res['prev3_sum'] = np.nan
    try:
        if 'pct_chg' in df.columns and len(df) >= 10:
            res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()
        else: res['volatility_10'] = np.nan
    except: res['volatility_10'] = np.nan
    return res

def safe_merge_pool(pool_df, other_df, cols):
    pool = pool_df.set_index('ts_code').copy()
    if other_df is None or other_df.empty:
        for c in cols: pool[c] = np.nan
        return pool.reset_index()
    if 'ts_code' not in other_df.columns:
        try: other_df = other_df.reset_index()
        except:
            for c in cols: pool[c] = np.nan
            return pool.reset_index()
    for c in cols:
        if c not in other_df.columns: other_df[c] = np.nan
    try: joined = pool.join(other_df.set_index('ts_code')[cols], how='left')
    except Exception:
        for c in cols: pool[c] = np.nan
        return pool.reset_index()
    for c in cols:
        if c not in joined.columns: joined[c] = np.nan
    return joined.reset_index()

def norm_col(s):
    s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    mn = s.min(); mx = s.max()
    if mx - mn < 1e-9: return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)

# ----------------------------------------------------
# 核心评分函数 (V9.0：使用批量数据进行评分，并调整权重)
# ----------------------------------------------------
def run_scoring_for_date(trade_date, all_daily_data, params):
    """
    V9.0 评分函数：从内存中的 all_daily_data 中切片获取历史数据，不再调用 API。
    """
    
    # 解包参数
    initial_top_n, final_pool_limit, min_price, max_price, min_turnover, min_amount, vol_spike_mult, volatility_max, high_pct_threshold = \
        params['INITIAL_TOP_N'], params['FINAL_POOL'], params['MIN_PRICE'], params['MAX_PRICE'], \
        params['MIN_TURNOVER'], params['MIN_AMOUNT'], params['VOL_SPIKE_MULT'], \
        params['VOLATILITY_MAX'], params['HIGH_PCT_THRESHOLD']
    
    # 1. 拉取当日涨幅榜初筛 (需要 API，但只调用一次)
    daily_all = safe_get(pro.daily, trade_date=trade_date)
    daily_basic = safe_get(pro.daily_basic, trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    mf_raw = safe_get(pro.moneyflow, trade_date=trade_date)
    
    if daily_all.empty: return pd.DataFrame()
    daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
    pool0 = daily_all.head(int(initial_top_n)).copy().reset_index(drop=True)

    # 2. 合并高级接口数据 (逻辑不变)
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,total_mv,circ_mv')
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in','net_mf_out']
        col = next((c for c in possible if c in mf_raw.columns), None)
        if col: moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)
    
    if not stock_basic.empty:
        keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_basic.columns]
        try: pool0 = pool0.merge(stock_basic[keep], on='ts_code', how='left')
        except Exception: pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
    else: pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
        
    pool_merged = safe_merge_pool(pool0, daily_basic, ['turnover_rate','amount','total_mv','circ_mv'])
    
    if moneyflow.empty: moneyflow = pd.DataFrame({'ts_code': pool_merged['ts_code'].tolist(), 'net_mf': [0.0]*len(pool_merged)})
    try: pool_merged = pool_merged.set_index('ts_code').join(moneyflow.set_index('ts_code'), how='left').reset_index()
    except: pool_merged['net_mf'] = 0.0
    pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0.0)
    
    # 3. 清洗 (使用 V6.0 严格参数)
    clean_list = []
    for r in pool_merged.itertuples():
        ts_code = getattr(r, 'ts_code')
        vol = getattr(r, 'vol', np.nan)
        close = getattr(r, 'close', np.nan)
        open_p = getattr(r, 'open', np.nan)
        pre_close = getattr(r, 'pre_close', np.nan)
        pct = getattr(r, 'pct_chg', np.nan)
        amount = getattr(r, 'amount', np.nan)
        turnover = getattr(r, 'turnover_rate', np.nan)
        total_mv = getattr(r, 'total_mv', np.nan)
        name = getattr(r, 'name', ts_code)

        if (pd.isna(vol) or vol == 0) and (pd.isna(amount) or amount == 0): continue
        if pd.isna(close) or (close < min_price) or (close > max_price): continue
        if isinstance(name, str) and (('ST' in name.upper()) or ('退' in name)): continue
        try:
            high = getattr(r, 'high', np.nan); low = getattr(r, 'low', np.nan)
            if (not pd.isna(open_p) and not pd.isna(high) and not pd.isna(low) and not pd.isna(pre_close)) and (open_p == high == low == pre_close): continue
        except: pass
        try:
            tv = total_mv; tv_yuan = tv * 10000.0 if not pd.isna(tv) and tv > 1e6 else tv;
            if not pd.isna(tv_yuan) and tv_yuan > 2000 * 1e8: continue
        except: pass
        # V6.0/V9.0 使用调整后的严格参数
        if not pd.isna(turnover) and float(turnover) < min_turnover: continue
        if not pd.isna(amount):
            amt = amount;
            if amt > 0 and amt < 1e5: amt = amt * 10000.0
            if amt < min_amount: continue
        if not pd.isna(pct) and float(pct) < 0: continue
        
        clean_list.append(r)

    
    clean_df = pd.DataFrame([dict(zip(r._fields, r)) for r in clean_list])
    if clean_df.empty: return pd.DataFrame()

    score_pool_n = min(int(final_pool_limit), 300)
    clean_df = clean_df.sort_values('pct_chg', ascending=False).head(score_pool_n).reset_index(drop=True)
    
    # 4. 指标计算与评分
    records = []
    # 找到当前日期的数据范围，只取 trade_date <= 当前日期的
    current_hist_data = all_daily_data[all_daily_data['trade_date'] <= trade_date]

    for row in clean_df.itertuples():
        ts_code = getattr(row, 'ts_code'); pct_chg = getattr(row, 'pct_chg', 0.0);
        turnover_rate = getattr(row, 'turnover_rate', np.nan); net_mf = float(getattr(row, 'net_mf', 0.0));
        amount_raw = getattr(row, 'amount', np.nan)
        amount = amount_raw * 10000.0 if not pd.isna(amount_raw) and amount_raw > 0 and amount_raw < 1e5 else amount_raw
        amount = amount if not pd.isna(amount) else 0.0
        name = getattr(row, 'name', ts_code)

        # 关键：从内存中获取历史数据
        hist = current_hist_data[current_hist_data['ts_code'] == ts_code].tail(60).copy()
        
        ind = compute_indicators(hist)

        vol_ratio, ten_return, macd, k, d, j, vol_last, vol_ma5, prev3_sum, volatility_10, ma20, last_close = \
            ind.get('vol_ratio', np.nan), ind.get('10d_return', np.nan), ind.get('macd', np.nan), \
            ind.get('k', np.nan), ind.get('d', np.nan), ind.get('j', np.nan), \
            ind.get('vol_last', np.nan), ind.get('vol_ma5', np.nan), ind.get('prev3_sum', np.nan), ind.get('volatility_10', np.nan), ind.get('ma20', np.nan), ind.get('last_close', np.nan)

        try: proxy_money = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
        except: proxy_money = 0.0

        rec = {'ts_code': ts_code, 'pct_chg': pct_chg, 'turnover_rate': turnover_rate, 'net_mf': net_mf, 'amount': amount,
               'vol_ratio': vol_ratio, '10d_return': ten_return, 'macd': macd, 'k': k, 'd': d, 'j': j,
               'vol_last': vol_last, 'vol_ma5': vol_ma5, 'prev3_sum': prev3_sum, 'volatility_10': volatility_10,
               'proxy_money': proxy_money, 'name': name,
               'last_close': last_close, 'ma20': ma20}
        records.append(rec)
        
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame()

    # 5. 风险过滤 (使用 V6.0 严格参数)
    if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):
        fdf = fdf[~((fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > high_pct_threshold))]
    if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
        fdf = fdf[~((fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > high_pct_threshold))]
    # V6.0/V9.0 使用调整后的严格参数：vol_spike_mult
    if all(c in fdf.columns for c in ['vol_last','vol_ma5']):
        fdf = fdf[~((fdf['vol_last'] > (fdf['vol_ma5'] * vol_spike_mult)))]
    # V6.0/V9.0 使用调整后的严格参数：volatility_max
    if 'volatility_10' in fdf.columns:
        fdf = fdf[~(fdf['volatility_10'] > volatility_max)]

    if fdf.empty: return pd.DataFrame()

    # 6. RSL & 归一化 (不变)
    if '10d_return' in fdf.columns:
        try:
            market_mean_10d = fdf['10d_return'].replace([np.inf,-np.inf], np.nan).dropna().mean()
            fdf['rsl'] = fdf['10d_return'] / (market_mean_10d if abs(market_mean_10d) >= 1e-9 else 1e-9)
        except: fdf['rsl'] = 1.0
    else: fdf['rsl'] = 1.0

    fdf['s_pct'] = norm_col(fdf.get('pct_chg', pd.Series([0]*len(fdf))))
    fdf['s_volratio'] = norm_col(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))
    fdf['s_turn'] = norm_col(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))
    fdf['s_money'] = norm_col(fdf.get('net_mf', pd.Series([0]*len(fdf)))) if fdf['net_mf'].abs().sum() > 0 else norm_col(fdf.get('proxy_money', pd.Series([0]*len(fdf))))
    fdf['s_amount'] = norm_col(fdf.get('amount', pd.Series([0]*len(fdf))))
    fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))
    fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
    fdf['s_rsl'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))
    fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))

    # 7. 综合评分 (V9.0 权重调整：极限防御：w_turn (0.35), w_volatility (0.25))
    # V9.0 权重: w_pct (0.05), w_volratio (0.10), w_turn (0.35), w_money (0.10), w_10d (0.05), w_macd (0.10), w_rsl (0.05), w_volatility (0.25)
    w_pct, w_volratio, w_turn, w_money, w_10d, w_macd, w_rsl, w_volatility = 0.05, 0.10, 0.35, 0.10, 0.05, 0.10, 0.05, 0.25
    
    fdf['综合评分'] = (fdf['s_pct'] * w_pct + fdf['s_volratio'] * w_volratio + fdf['s_turn'] * w_turn + fdf['s_money'] * w_money + fdf['s_10d'] * w_10d + fdf['s_macd'] * w_macd + fdf['s_rsl'] * w_rsl + fdf['s_volatility'] * w_volatility)
    
    return fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)


# ----------------------------------------------------
# 简易回测模块 (V9.0：使用批量数据进行回测)
# ----------------------------------------------------
def run_simple_backtest(days, params):
    status = st.session_state['backtest_status']
    
    container = st.empty()
    with container.container():
        st.subheader("📈 简易历史回测结果")
        
        # 1. 获取交易日历
        trade_dates_df = safe_get(pro.trade_cal, exchange='SSE', is_open='1', end_date=find_last_trade_day(), fields='cal_date')
        if trade_dates_df.empty:
            st.error("无法获取历史交易日历。")
            return

        trade_dates = trade_dates_df['cal_date'].sort_values(ascending=False).head(days + 1).tolist()
        trade_dates.reverse() 
        total_iterations = len(trade_dates) - 1
        
        if total_iterations < 1:
            st.warning("交易日不足，无法进行回测。")
            return
            
        status['total_days'] = total_iterations
        
        # 2. V5.0 预加载所有数据 (仅在 bulk_data 为 None 时运行)
        if status['bulk_data'] is None:
            st.warning("V9.0 终极加速中：正在预加载所有所需历史数据。请耐心等待，只需运行一次。")
            
            # a) 确定所有股票代码
            stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code')
            if stock_basic.empty:
                st.error("无法获取股票列表。")
                return

            all_codes = stock_basic['ts_code'].tolist()
            
            # b) 确定历史时间范围 (回测起始日 - 60 天)
            earliest_select_date = trade_dates[0]
            start_dt = datetime.strptime(earliest_select_date, "%Y%m%d") - timedelta(days=60 * 1.5) # 冗余60天
            start_date_hist = start_dt.strftime("%Y%m%d")
            
            # c) 批量获取数据并缓存
            final_end_date = trade_dates[-2]
            status['bulk_data'] = bulk_fetch_daily(all_codes, start_date_hist, final_end_date)
            
            if status['bulk_data'].empty:
                st.error("批量历史数据获取失败，请检查 Tushare Token 权限。")
                return
            
            st.success(f"数据预加载完成！共获取 {len(all_codes)} 支股票在 {start_date_hist} 至 {final_end_date} 间的数据。")
            st.rerun() # 预加载完成后强制刷新，进入正式回测阶段

        # 3. 正式回测流程
        bulk_data = status['bulk_data']
        start_index = status['current_index']
        
        if start_index >= total_iterations:
             st.success(f"回测已完成。累计收益率请查看下方。")
        else:
             st.info(f"回测周期：**{trade_dates[0]}** 至 **{trade_dates[-2]}**。正在从第 {start_index+1} 天继续...")

        pbar = st.progress(status['progress'], text=f"回测进度：[{status['current_index']}/{status['total_days']}]...")
        
        # 4. 参数打包
        params_dict = {
            'INITIAL_TOP_N': params['INITIAL_TOP_N'], 'FINAL_POOL': params['FINAL_POOL'], 'MIN_PRICE': params['MIN_PRICE'], 
            'MAX_PRICE': params['MAX_PRICE'], 'MIN_TURNOVER': params['MIN_TURNOVER'], 'MIN_AMOUNT': params['MIN_AMOUNT'], 
            'VOL_SPIKE_MULT': params['VOL_SPIKE_MULT'], 'VOLATILITY_MAX': params['VOLATILITY_MAX'], 
            'HIGH_PCT_THRESHOLD': params['HIGH_PCT_THRESHOLD']
        }
        
        for i in range(start_index, total_iterations):
            select_date = trade_dates[i]
            next_trade_date = trade_dates[i+1]
            
            # 核心步骤：调用 V9.0 评分函数，传入内存中的 bulk_data
            select_df_full = run_scoring_for_date(select_date, bulk_data, params_dict) 

            # T+1 收益计算逻辑 (不变)
            return_pct = 0.0
            buy_price, sell_price = np.nan, np.nan

            if select_df_full.empty:
                result = {'选股日': select_date, '股票': '无符合条件', 'T+1 收益率': 0.0, '买入价 (T+1 开盘)': np.nan, '卖出价 (T+1 收盘)': np.nan, '评分': np.nan}
            else:
                top_pick = select_df_full.iloc[0] 
                ts_code = top_pick['ts_code']
                
                next_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=next_trade_date)
                

                if not next_day_data.empty and 'open' in next_day_data.columns and 'close' in next_day_data.columns:
                    buy_price = next_day_data.iloc[0]['open']
                    sell_price = next_day_data.iloc[0]['close']
                    
                    if buy_price > 0 and not pd.isna(sell_price):
                        return_pct = (sell_price / buy_price) - 1.0

                result = {
                    '选股日': select_date,
                    '股票': f"{top_pick.get('name', 'N/A')}({ts_code})",
                    'T+1 收益率': return_pct * 100,
                    '买入价 (T+1 开盘)': buy_price,
                    '卖出价 (T+1 收盘)': sell_price,
                    '评分': top_pick['综合评分']
                }

            # 5. 更新状态和进度条
            status['results'].append(result)
            status['current_index'] = i + 1
            status['progress'] = (i + 1) / total_iterations
            
            pbar.progress(status['progress'], text=f"正在回测 {select_date}... [{i+1}/{total_iterations}]")
            
            # 强制 Rerun 确保进度条更新
            if (i+1) % 2 == 0: 
                 st.rerun() 
        
        # 循环结束，标记完成
        status['progress'] = 1.0
        status['current_index'] = total_iterations
        pbar.progress(1.0, text="回测完成。")
        
        # 6. 结果展示 (不变)
        results_df = pd.DataFrame(status['results'])
        
        if results_df.empty:
            st.warning("回测结果为空。")
            return
            
        results_df['T+1 收益率'] = results_df['T+1 收益率'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        cumulative_return = (results_df['T+1 收益率'] / 100 + 1).product() - 1
        wins = (results_df['T+1 收益率'] > 0).sum()
        total_trades = len(results_df)
        win_rate = wins / total_trades if total_trades > 0 else 0

        st.markdown("---")
        st.subheader("💡 最终回测指标")
        colA, colB, colC = st.columns(3)
        colA.metric("累计收益率 (T+1)", f"{cumulative_return*100:.2f}%")
        colB.metric("胜率", f"{win_rate*100:.2f}%")
        colC.metric("交易次数", f"{total_trades}")
        
        st.subheader("📋 每日交易记录")
        st.dataframe(results_df, use_container_width=True)

# ----------------------------------------------------
# 实时选股模块 (V9.0：使用批量数据进行实时评分)
# ----------------------------------------------------
def run_live_selection(last_trade, params):
    st.write(f"正在运行实时选股（最近交易日：{last_trade}）...")
    
    # V9.0 实时选股也需要历史数据，但不再逐个获取
    st.warning("正在预加载历史数据...")
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code')
    all_codes = stock_basic['ts_code'].tolist()
    start_dt = datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=60 * 1.5) 
    start_date_hist = start_dt.strftime("%Y%m%d")
    
    # 批量获取数据并缓存
    bulk_data = bulk_fetch_daily(all_codes, start_date_hist, last_trade)
    if bulk_data.empty:
        st.error("实时选股所需历史数据获取失败。")
        return
    
    # 5. 传入 bulk_data 进行评分
    params_dict = {
        'INITIAL_TOP_N': params['INITIAL_TOP_N'], 'FINAL_POOL': params['FINAL_POOL'], 'MIN_PRICE': params['MIN_PRICE'], 
        'MAX_PRICE': params['MAX_PRICE'], 'MIN_TURNOVER': params['MIN_TURNOVER'], 'MIN_AMOUNT': params['MIN_AMOUNT'], 
        'VOL_SPIKE_MULT': params['VOL_SPIKE_MULT'], 'VOLATILITY_MAX': params['VOLATILITY_MAX'], 
        'HIGH_PCT_THRESHOLD': params['HIGH_PCT_THRESHOLD']
    }
    fdf_full = run_scoring_for_date(last_trade, bulk_data, params_dict)

    if fdf_full.empty:
        st.error("清洗和评分后没有候选，建议放宽条件或检查接口权限。")
        st.stop()

    fdf = fdf_full.head(params['TOP_DISPLAY']).copy()
    fdf.index = fdf.index + 1

    st.success(f"评分完成：总候选 {len(fdf_full)} 支，显示 Top {min(params['TOP_DISPLAY'], len(fdf))}。")
    display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','proxy_money','amount','10d_return','macd','k','d','j','rsl','volatility_10']
    for c in display_cols:
        if c not in fdf.columns: fdf[c] = np.nan

    st.dataframe(fdf[display_cols], use_container_width=True)

    out_csv = fdf_full[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

    st.markdown("### 小结与操作提示（简洁）")
    st.markdown("""
- **【策略风格】** 本版本为 **极限防御模式**，极度偏爱高换手率和低波动率，是当前弱势行情下的最安全策略。
- **【风控提示】** 已启用最严格的风控参数。实战中，请遵循交易纪律，及时止盈止损。
- **【重要纪律】** 9:40 前不买 → 观察 9:40-10:05 的量价节奏 → 10:05 后择优介入。
""")


# ----------------------------------------------------
# 主程序控制逻辑
# ----------------------------------------------------
params = {
    'INITIAL_TOP_N': INITIAL_TOP_N, 'FINAL_POOL': FINAL_POOL, 'TOP_DISPLAY': TOP_DISPLAY,
    'MIN_PRICE': MIN_PRICE, 'MAX_PRICE': MAX_PRICE, 'MIN_TURNOVER': MIN_TURNOVER,
    'MIN_AMOUNT': MIN_AMOUNT, 'VOL_SPIKE_MULT': VOL_SPIKE_MULT, 'VOLATILITY_MAX': VOLATILITY_MAX,
    'HIGH_PCT_THRESHOLD': HIGH_PCT_THRESHOLD
}

if st.session_state.get('run_backtest', False):
    run_simple_backtest(BACKTEST_DAYS, params)
    
elif st.session_state.get('run_selection', False):
    run_live_selection(last_trade, params)
    
else:
    st.info("请点击上方的按钮开始运行。")
