# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（回测日期修正终极版）
说明：
- 目标：**激进短线爆发 (B) + 妖股捕捉 (C)**
- 【2025-11-23 最终修复】：
    - 修复成交额单位（已解决选股成功）
    - 增强回测数据鲁棒性（已解决数据碎片化）
    - **修复回测日期起始点错误，解决“回测仅覆盖 1 天”和“交易次数 0”的问题。**
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
st.set_page_config(page_title="选股王 · 10000旗舰（日期修正）", layout="wide")
st.title("选股王 · 10000 积分旗舰（回测日期修正终极版）")
st.markdown("输入你的 Tushare Token（仅本次运行使用）。若有权限缺失，脚本会自动降级并继续运行。")

# ---------------------------
# 侧边栏参数（实时可改）
# ---------------------------
with st.sidebar:
    st.header("可调参数（实时）")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=300, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    # 参数默认值：调至极低，确保通过
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=3.0, step=0.5)) 
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=0.5, step=0.5)) 
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=20_000_000.0, step=10_000_000.0)) # 2000 万
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=8.0, step=0.5))
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    MIN_MARKET_CAP = float(st.number_input("最低市值 (元)", value=2000000000.0, step=100000000.0))  
    MAX_MARKET_CAP = float(st.number_input("最高市值 (元)", value=50000000000.0, step=1000000000.0))  
    st.markdown("---")
    # --- 新增回测参数 ---
    st.header("历史回测参数")
    BACKTEST_DAYS = int(st.number_input("回测交易日天数", value=60, min_value=10, max_value=250))
    HOLD_DAYS_OPTIONS = st.multiselect("回测持股天数", options=[1, 3, 5, 10, 20], default=[1, 3, 5])
    # ---
    st.caption("提示：**回测日期已修正，应能覆盖足够的天数。**")

# ---------------------------
# Token 输入（主区）
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

# 初始化 tushare
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 安全调用 & 缓存辅助
# ---------------------------
def safe_get(func, **kwargs):
    """Call API and return DataFrame or empty df on any error."""
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_trade_cal(start_date, end_date):
    """获取交易日历并缓存"""
    try:
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
        return df[df.is_open == 1]['cal_date'].tolist()
    except Exception:
        return []

@st.cache_data(ttl=6000)
def get_bulk_daily_data(ts_codes, start_date, end_date):
    """批量获取指定股票和日期的 daily 数据 (仅用于实时评分)"""
    all_data = []
    trade_dates = get_trade_cal(start_date, end_date)
    
    st.write(f"正在批量加载 {len(trade_dates)} 个交易日的 daily 数据 (用于指标计算)...")
    pbar = st.progress(0)
    for i, date in enumerate(trade_dates):
        daily_df = safe_get(pro.daily, trade_date=date)
        if not daily_df.empty:
            all_data.append(daily_df)
        pbar.progress((i + 1) / len(trade_dates))
    pbar.progress(1.0)
    
    if not all_data:
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)
    if ts_codes:
        full_df = full_df[full_df['ts_code'].isin(ts_codes)]
    
    return full_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

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

# ---------------------------
# 拉当日涨幅榜初筛
# ---------------------------
st.write("正在拉取当日 daily（涨幅榜）作为初筛...")
daily_all = safe_get(pro.daily, trade_date=last_trade)
if daily_all.empty:
    st.error("无法获取当日 daily 数据（Tushare 返回空）。请确认 Token 权限。")
    st.stop()

daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
st.write(f"当日记录：{len(daily_all)}，取涨幅前 {INITIAL_TOP_N} 作为初筛。")
pool0 = daily_all.head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

# ---------------------------
# 尝试加载高级接口（有权限时启用）
# ---------------------------
st.write("尝试加载 stock_basic / daily_basic / moneyflow 等高级接口（若权限允许）...")
stock_list = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')
daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)

# moneyflow 预处理
if not mf_raw.empty:
    possible = ['net_mf','net_mf_amount','net_mf_in','net_mf_out']
    col = None
    for c in possible:
        if c in mf_raw.columns:
            col = c; break
    if col is None:
        numeric_cols = [c for c in mf_raw.columns if c != 'ts_code' and pd.api.types.is_numeric_dtype(mf_raw[c])]
        col = numeric_cols[0] if numeric_cols else None
    if col:
        moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)
    else:
        moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
else:
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    st.warning("moneyflow 未获取到，将把主力流向因子置为 0。")

# ---------------------------
# 合并基本信息（safe）
# ---------------------------
def safe_merge_pool(pool_df, other_df, cols):
    pool = pool_df.copy()
    if other_df is None or other_df.empty or 'ts_code' not in other_df.columns:
        return pool
    cols_to_merge = [c for c in cols if c in other_df.columns]
    if not cols_to_merge:
        return pool
    try:
        joined = pool.merge(other_df[['ts_code'] + cols_to_merge], on='ts_code', how='left')
        return joined
    except Exception:
        return pool

# merge stock_basic
if not stock_list.empty:
    keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_list.columns]
    try:
        pool0 = pool0.merge(stock_list[keep], on='ts_code', how='left')
    except Exception:
        pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
else:
    pool0['name'] = pool0['ts_code']; pool0['industry'] = ''

# merge daily_basic
pool_merged = safe_merge_pool(pool0, daily_basic, ['turnover_rate','amount','total_mv','circ_mv'])

# merge moneyflow robustly
if not moneyflow.empty:
    pool_merged = pool_merged.merge(moneyflow, on='ts_code', how='left')
if 'net_mf' not in pool_merged.columns:
    pool_merged['net_mf'] = 0.0
pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0.0)

# ---------------------------
# 基本清洗（ST / 停牌 / 价格区间 / 一字板 / 换手 / 成交额 / 市值）
# ---------------------------
st.write("对初筛池进行清洗（ST/停牌/价格/一字板/换手/成交额等）...")

pool_merged['total_mv_yuan'] = pool_merged['total_mv'].fillna(0) * 10000

clean_df = pool_merged.copy()

# 1. 过滤 ST / 退市 / 北交所
clean_df = clean_df[~clean_df['name'].str.contains('ST|退|N', na=False, case=False)]
clean_df = clean_df[~clean_df['ts_code'].str.startswith('4', na=False)]
clean_df = clean_df[~clean_df['ts_code'].str.startswith('8', na=False)]

# 2. 价格过滤
clean_df = clean_df[(clean_df['close'] >= MIN_PRICE) & (clean_df['close'] <= MAX_PRICE)]

# 3. 市值过滤
clean_df = clean_df[(clean_df['total_mv_yuan'] >= MIN_MARKET_CAP) & (clean_df['total_mv_yuan'] <= MAX_MARKET_CAP)]

# 4. 换手率过滤
if 'turnover_rate' in clean_df.columns:
    clean_df['turnover_rate'] = clean_df['turnover_rate'].fillna(0)
    clean_df = clean_df[clean_df['turnover_rate'] >= MIN_TURNOVER]
else:
    st.warning("daily_basic 接口缺失，跳过换手率过滤。")

# 5. 【关键修正：成交额单位转换】
daily_amount = daily_all[['ts_code', 'amount']].copy()
daily_amount['amount_actual_yuan'] = daily_amount['amount'].astype(float) * 1000.0 

clean_df = clean_df.merge(daily_amount[['ts_code', 'amount_actual_yuan']], on='ts_code', how='left')
clean_df['amount_actual_yuan'] = clean_df['amount_actual_yuan'].fillna(0) 

# 过滤：成交额(元) >= 最低成交额(元)
clean_df = clean_df[clean_df['amount_actual_yuan'] >= MIN_AMOUNT]

# 6. 过滤停牌/无成交
clean_df = clean_df[(clean_df['vol'] > 0) & (clean_df['amount_actual_yuan'] > 0)]

# 7. 过滤一字涨停板 (使用 open == high)
clean_df['is_zt'] = (clean_df['open'] == clean_df['high']) & (clean_df['pct_chg'] > 9.5)
clean_df = clean_df[~clean_df['is_zt']]


st.write(f"清洗后候选数量：{len(clean_df)} （将从中取涨幅前 {FINAL_POOL} 进入评分阶段）")
if len(clean_df) == 0:
    st.error("清洗后没有候选，这通常是 Tushare Token 接口权限缺失，无法获取到基本的 daily/daily_basic 数据导致。")
    st.stop()

# ---------------------------
# 取涨幅前 FINAL_POOL 进入评分池
# ---------------------------
clean_df = clean_df.sort_values('pct_chg', ascending=False).head(int(FINAL_POOL)).reset_index(drop=True)
st.write(f"用于评分的池子大小：{len(clean_df)}")

# ---------------------------
# 批量获取 K 线历史数据 (仅用于实时评分，数据可能会不全)
# ---------------------------
latest_date = last_trade
max_hist_days = 60 
start_date_hist = (datetime.strptime(latest_date, "%Y%m%d") - timedelta(days=max_hist_days * 2)).strftime("%Y%m%d")
GLOBAL_KLINE_DATA = get_bulk_daily_data(clean_df['ts_code'].unique().tolist(), start_date_hist, latest_date)

def get_hist_cached_bulk(ts_code, end_date, days=60):
    """从全局缓存中获取历史 K 线数据"""
    if GLOBAL_KLINE_DATA.empty:
        return pd.DataFrame()
        
    hist_df = GLOBAL_KLINE_DATA[GLOBAL_KLINE_DATA['ts_code'] == ts_code].copy()
    
    if hist_df.empty:
        return pd.DataFrame()
    
    hist_df = hist_df[hist_df['trade_date'] <= end_date]
    hist_df = hist_df.tail(days * 2) 
    
    return hist_df.sort_values('trade_date').reset_index(drop=True)

# ---------------------------
# 指标计算（使用 bulk 缓存）
# ---------------------------
def compute_indicators(df):
    res = {}
    if df.empty or len(df) < 3: return res
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    try: res['last_close'] = close.iloc[-1]
    except: res['last_close'] = np.nan

    # MA
    for n in (5,10,20):
        if len(close) >= n:
            res[f'ma{n}'] = close.rolling(window=n).mean().iloc[-1]
        else:
            res[f'ma{n}'] = np.nan

    # MACD 
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_val = (diff - dea) * 2
        res['macd'] = macd_val.iloc[-1]; res['diff'] = diff.iloc[-1]; res['dea'] = dea.iloc[-1]
    else:
        res['macd'] = res['diff'] = res['dea'] = np.nan

    # KDJ
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
    else:
        res['k'] = res['d'] = res['j'] = np.nan

    # vol ratio and metrics
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        avg_prev5 = np.mean(vols[-6:-1])
        res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9)
        res['vol_last'] = vols[-1]
        res['vol_ma5'] = avg_prev5
    else:
        res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan

    # 10d return
    if len(close) >= 10:
        res['10d_return'] = close.iloc[-1] / close.iloc[-10] - 1
    else:
        res['10d_return'] = np.nan

    # prev3_sum for down-then-bounce detection
    if 'pct_chg' in df.columns and len(df) >= 4:
        try:
            pct = df['pct_chg'].astype(float)
            res['prev3_sum'] = pct.iloc[-4:-1].sum()
        except:
            res['prev3_sum'] = np.nan
    else:
        res['prev3_sum'] = np.nan

    # volatility (std of last 10 pct_chg)
    try:
        if 'pct_chg' in df.columns and len(df) >= 10:
            res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()
        else:
            res['volatility_10'] = np.nan
    except:
        res['volatility_10'] = np.nan

    # recent 20-day high for breakout detection
    try:
        if len(high) >= 20:
            res['recent20_high'] = float(high.tail(20).max())
        else:
            res['recent20_high'] = float(high.max()) if len(high)>0 else np.nan
    except:
        res['recent20_high'] = np.nan

    # 阳线实体强度（今天）
    try:
        today_open = df['open'].astype(float).iloc[-1]
        today_close = df['close'].astype(float).iloc[-1]
        today_high = df['high'].astype(float).iloc[-1]
        today_low = df['low'].astype(float).iloc[-1]
        body = abs(today_close - today_open)
        rng = max(today_high - today_low, 1e-9)
        res['yang_body_strength'] = body / rng
    except:
        res['yang_body_strength'] = 0.0

    return res

# 评分计算
st.write("为评分池逐票计算指标...")
records = []
pbar2 = st.progress(0)
for idx, row in enumerate(clean_df.itertuples()):
    ts_code = getattr(row, 'ts_code')
    name = getattr(row, 'name', ts_code)
    pct_chg = getattr(row, 'pct_chg', 0.0)
    amount = getattr(row, 'amount_actual_yuan', 0.0) 
    turnover_rate = getattr(row, 'turnover_rate', np.nan)
    net_mf = float(getattr(row, 'net_mf', 0.0))

    hist = get_hist_cached_bulk(ts_code, last_trade, days=60)
    ind = compute_indicators(hist)

    vol_ratio = ind.get('vol_ratio', np.nan)
    ten_return = ind.get('10d_return', np.nan)
    ma5 = ind.get('ma5', np.nan)
    ma10 = ind.get('ma10', np.nan)
    ma20 = ind.get('ma20', np.nan)
    macd = ind.get('macd', np.nan)
    diff = ind.get('diff', np.nan)
    dea = ind.get('dea', np.nan)
    k, d, j = ind.get('k', np.nan), ind.get('d', np.nan), ind.get('j', np.nan)
    last_close = ind.get('last_close', np.nan)
    vol_last = ind.get('vol_last', np.nan)
    vol_ma5 = ind.get('vol_ma5', np.nan)
    prev3_sum = ind.get('prev3_sum', np.nan)
    volatility_10 = ind.get('volatility_10', np.nan)
    recent20_high = ind.get('recent20_high', np.nan)
    yang_body_strength = ind.get('yang_body_strength', 0.0)

    # 资金强度代理
    try:
        proxy_money = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
    except:
        proxy_money = 0.0

    rec = {
        'ts_code': ts_code, 'name': name, 'pct_chg': pct_chg,
        'amount': amount,
        'turnover_rate': turnover_rate if not pd.isna(turnover_rate) else np.nan,
        'net_mf': net_mf,
        'vol_ratio': vol_ratio if not pd.isna(vol_ratio) else np.nan,
        '10d_return': ten_return if not pd.isna(ten_return) else np.nan,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
        'macd': macd, 'diff': diff, 'dea': dea, 'k': k, 'd': k, 'j': j,
        'last_close': last_close, 'vol_last': vol_last, 'vol_ma5': vol_ma5, 'recent20_high': recent20_high, 'yang_body_strength': yang_body_strength,
        'prev3_sum': prev3_sum, 'volatility_10': volatility_10,
        'proxy_money': proxy_money
    }

    records.append(rec)
    pbar2.progress((idx+1)/len(clean_df))

pbar2.progress(1.0)
fdf = pd.DataFrame(records)
if fdf.empty:
    st.error("评分计算失败或无数据，请检查 Token 权限与接口。")
    st.stop()


# ---------------------------
# 风险过滤
# ---------------------------
st.write("执行风险过滤：下跌途中大阳 / 高位大阳 ...")
try:
    before_cnt = len(fdf)
    # A: 高位大阳线
    HIGH_PCT_THRESHOLD_VAL = float(HIGH_PCT_THRESHOLD) # 使用参数
    if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):
        mask_high_big = (fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD_VAL)
        fdf = fdf[~mask_high_big]

    # B: 下跌途中反抽
    if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
        mask_down_rebound = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD_VAL)
        fdf = fdf[~mask_down_rebound]

    after_cnt = len(fdf)
    st.write(f"风险过滤：{before_cnt} -> {after_cnt}（仅保留追高风险）")
except Exception as e:
    st.warning(f"风险过滤模块异常，跳过过滤。错误：{e}")

# ---------------------------
# RSL（相对强弱）
# ---------------------------
if '10d_return' in fdf.columns:
    try:
        market_mean_10d = fdf['10d_return'].replace([np.inf,-np.inf], np.nan).dropna().mean()
        if np.isnan(market_mean_10d) or abs(market_mean_10d) < 1e-9:
            market_mean_10d = 1e-9
        fdf['rsl'] = fdf['10d_return'] / market_mean_10d
    except:
        fdf['rsl'] = 1.0
else:
    fdf['rsl'] = 1.0

# ---------------------------
# 子指标归一化（稳健）
# ---------------------------
def norm_col(s):
    s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    mn = s.min(); mx = s.max()
    if mx - mn < 1e-9:
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)

fdf['s_pct'] = norm_col(fdf.get('pct_chg', pd.Series([0]*len(fdf))))
fdf['s_volratio'] = norm_col(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))
fdf['s_turn'] = norm_col(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))
if 'net_mf' in fdf.columns and fdf['net_mf'].abs().sum() > 0:
    fdf['s_money'] = norm_col(fdf.get('net_mf', pd.Series([0]*len(fdf))))
else:
    fdf['s_money'] = norm_col(fdf.get('proxy_money', pd.Series([0]*len(fdf))))
fdf['s_amount'] = norm_col(fdf.get('amount', pd.Series([0]*len(fdf))))
fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))
fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
fdf['s_rsl'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))
fdf['s_volatility'] = norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))

# ---------------------------
# 趋势因子与强化评分
# ---------------------------
fdf['ma_trend_flag'] = ((fdf.get('ma5', pd.Series([])) > fdf.get('ma10', pd.Series([]))) & (fdf.get('ma10', pd.Series([])) > fdf.get('ma20', pd.Series([])))).fillna(False)
fdf['macd_golden_flag'] = (fdf.get('diff', 0) > fdf.get('dea', 0)).fillna(False)
fdf['vol_price_up_flag'] = (fdf.get('vol_last', 0) > fdf.get('vol_ma5', 0)).fillna(False)
fdf['break_high_flag'] = (fdf.get('last_close', 0) > fdf.get('recent20_high', 0)).fillna(False)
fdf['yang_body_strength'] = fdf.get('yang_body_strength', 0.0).fillna(0.0)

# 组合成趋势原始分
fdf['trend_score_raw'] = (
    fdf['ma_trend_flag'].astype(float) * 1.5 +  
    fdf['macd_golden_flag'].astype(float) * 1.3 +
    fdf['vol_price_up_flag'].astype(float) * 1.0 +
    fdf['break_high_flag'].astype(float) * 1.3 +
    fdf['yang_body_strength'].astype(float) * 0.8
)

# 归一化趋势分
fdf['trend_score'] = norm_col(fdf['trend_score_raw'])

# ---------------------------
# 最终综合评分
# ---------------------------
fdf['综合评分'] = (
    fdf['trend_score'] * 0.40 +      
    fdf.get('s_10d', 0)*0.10 +       
    fdf.get('s_rsl', 0)*0.08 +       
    fdf.get('s_volratio', 0)*0.10 +  
    fdf.get('s_turn', 0)*0.10 +      
    fdf.get('s_money', 0)*0.10 +     
    fdf.get('s_pct', 0)*0.05 +       
    fdf.get('s_volatility', 0)*0.07  
)

# ---------------------------
# 最终排序与展示
# ---------------------------
fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
fdf.index = fdf.index + 1

st.success(f"评分完成：总候选 {len(fdf)} 支，显示 Top {min(TOP_DISPLAY, len(fdf))}。")
display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','proxy_money','amount','10d_return','macd','diff','dea','k','d','j','rsl','volatility_10']
for c in display_cols:
    if c not in fdf.columns:
        fdf[c] = np.nan

st.dataframe(fdf[display_cols].head(TOP_DISPLAY), use_container_width=True)

# 下载
out_csv = fdf[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

# ---------------------------
# 历史回测部分（数据鲁棒性增强 & 日期修正）
# ---------------------------
@st.cache_data(ttl=6000)
def run_backtest(start_date, end_date, hold_days, top_k):
    # start_date 和 end_date 之间应该有足够多的交易日
    trade_dates = get_trade_cal(start_date, end_date)
    
    if not trade_dates:
        return {h: {'returns': [], 'wins': 0, 'total': 0, 'win_rate': 0.0, 'avg_return': 0.0} for h in hold_days}

    results = {h: {'returns': [], 'wins': 0, 'total': 0, 'win_rate': 0.0, 'avg_return': 0.0} for h in hold_days}
    
    # 确定回测实际的起始日（回溯 x 天）
    bt_start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=BACKTEST_DAYS * 1.5)).strftime("%Y%m%d")
    
    # 确保只回测 BACKTEST_DAYS 个交易日
    backtest_dates = [d for d in trade_dates if d >= bt_start and d <= end_date]
    if len(backtest_dates) < BACKTEST_DAYS:
        st.warning(f"由于数据或交易日限制，回测仅能覆盖 {len(backtest_dates)} 天。")
    
    # 取最近的 BACKTEST_DAYS 个交易日作为买入日期池
    backtest_dates = backtest_dates[-BACKTEST_DAYS:]
    
    st.write(f"正在模拟 {len(backtest_dates)} 个交易日的选股回测...")
    pbar_bt = st.progress(0)
    
    for i, buy_date in enumerate(backtest_dates):
        # 模拟当日选股：直接调用 API 获取当日数据，更稳定
        daily_df = safe_get(pro.daily, trade_date=buy_date)
        
        if daily_df.empty:
            pbar_bt.progress((i+1)/len(backtest_dates)); continue

        # 模拟当日的筛选逻辑 (简化版)
        daily_df = daily_df.sort_values("pct_chg", ascending=False).head(INITIAL_TOP_N).copy()
        
        # 1. 价格、成交额过滤 (回测中修正单位)
        daily_df['amount_yuan'] = daily_df['amount'].fillna(0) * 1000.0 # **回测中修正单位**
        daily_df = daily_df[(daily_df['close'] >= MIN_PRICE) & (daily_df['close'] <= MAX_PRICE)]
        daily_df = daily_df[daily_df['amount_yuan'] >= MIN_AMOUNT]
        
        # 2. 过滤停牌/无成交
        daily_df = daily_df[(daily_df['vol'] > 0) & (daily_df['amount_yuan'] > 0)]

        # 3. 过滤一字涨停板 (使用 open == high)
        daily_df['is_zt'] = (daily_df['open'] == daily_df['high']) & (daily_df['pct_chg'] > 9.5)
        daily_df = daily_df[~daily_df['is_zt']]

        # 模拟评分：简化为取当日涨幅榜前 top_k
        scored_stocks = daily_df.head(top_k).copy()
        
        for _, row in scored_stocks.iterrows():
            ts_code = row['ts_code']
            buy_price = float(row['close']) 
            
            if pd.isna(buy_price) or buy_price <= 0: continue

            for h in hold_days:
                try:
                    # 确定卖出日期在 trade_dates 中的位置
                    current_index = trade_dates.index(buy_date)
                    sell_date = trade_dates[current_index + h]
                except (ValueError, IndexError):
                    continue
                
                # 获取卖出价格 - 直接调用 API，增加鲁棒性
                sell_price_df = safe_get(pro.daily, trade_date=sell_date, ts_code=ts_code)
                sell_price = sell_price_df['close'].iloc[0] if not sell_price_df.empty else np.nan
                
                if pd.isna(sell_price) or sell_price <= 0: continue
                
                ret = (sell_price / buy_price) - 1.0
                results[h]['total'] += 1
                results[h]['returns'].append(ret)
                if ret > 0:
                    results[h]['wins'] += 1

        pbar_bt.progress((i+1)/len(backtest_dates))

    pbar_bt.progress(1.0)
    
    final_results = {}
    for h, res in results.items():
        total = res['total']
        if total > 0:
            avg_return = np.mean(res['returns']) * 100.0
            win_rate = (res['wins'] / total) * 100.0
        else:
            avg_return = 0.0
            win_rate = 0.0
            
        final_results[h] = {
            '平均收益率 (%)': f"{avg_return:.2f}",
            '胜率 (%)': f"{win_rate:.2f}",
            '总交易次数': total
        }
        
    return final_results

# ---------------------------
# 回测执行
# ---------------------------
if st.checkbox("✅ 运行历史回测 (使用 Top K)", value=False):
    if not HOLD_DAYS_OPTIONS:
        st.warning("请至少选择一个回测持股天数。")
    else:
        st.header("📈 历史回测结果（买入收盘价 / 卖出收盘价）")
        
        # 【核心修复】：计算一个足够远的起始日期
        try:
            # 往回推 200 个日历日，确保有足够的交易日被包含
            start_date_for_cal = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
        except:
            start_date_for_cal = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            
        backtest_result = run_backtest(
            start_date=start_date_for_cal, # 传入一个足够早的日期
            end_date=last_trade,
            hold_days=HOLD_DAYS_OPTIONS,
            top_k=TOP_DISPLAY
        )

        bt_df = pd.DataFrame(backtest_result).T
        bt_df.index.name = "持股天数"
        bt_df = bt_df.reset_index()
        bt_df['持股天数'] = bt_df['持股天数'].astype(str) + ' 天'
        
        st.dataframe(bt_df, use_container_width=True, hide_index=True)
        st.success("回测完成！")
        
        export_df = bt_df.copy()
        export_df.columns = ['HoldDays', 'AvgReturn', 'WinRate', 'TotalTrades']
        out_csv_bt = export_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "下载回测结果 CSV", 
            data=out_csv_bt, 
            file_name=f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
            mime="text/csv"
        )


# ---------------------------
# 小结与建议（简洁）
# ---------------------------
st.markdown("### 小结与操作提示（回测日期修正版）")
st.markdown("""
- **当前代码：** **回测日期修正终极版**，已修复回测起始日期错误。
- **操作：** 请重新运行脚本，并勾选底部的 **“✅ 运行历史回测”** 选项。
""")
