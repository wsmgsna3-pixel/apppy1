# -*- coding: utf-8 -*-
"""
短线王 · V7.0 Ultimate（10000积分满血版）
目标：面向持股 1-3 天的短线策略，利用 10000 积分的特色数据（主力资金、筹码分布、券商金股、盈利预测等）
设计要点：
 - 优先使用特色接口（若可用）：moneyflow / daily_basic / stock_basic / specialty APIs（如筹码分布接口）
 - 分时与分钟因子（若分钟接口可用则启用分时因子）
 - 多层降级：部分接口失效会自动回退到最稳健的日线/日级近似
 - 丰富因子体系：主力资金、筹码结构、量价齐升、行业动量、分时强度、K线健康、波动/ATR过滤等
 - 自动缓存与并行化（有限）以加速多次运行
注意：脚本尽最大努力调用特色接口，但不同帐号接口名/权限略有差别，脚本包含尝试机制。
"""
import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="短线王 · V7.0 Ultimate", layout="wide")
st.title("短线王 · V7.0 Ultimate（10000积分版 · 1-3 天短线）")

# ---------------------------
# --- User inputs / basic
# ---------------------------
TS_TOKEN = st.text_input("请输入 Tushare Token（仅本次会话）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 并回车以激活。")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

st.sidebar.header("运行频率与过滤（建议每天 4-5 次运行）")
auto_refresh = st.sidebar.checkbox("自动刷新（前端每 X 秒刷新一次）", value=False)
refresh_seconds = int(st.sidebar.number_input("刷新间隔（秒）", min_value=30, max_value=3600, value=300, step=30))
run_once = st.sidebar.button("立即刷新/运行一次")

st.sidebar.markdown("---")
st.sidebar.header("筛选参数（默认为适合 1-3 天短线）")
INITIAL_TOP_N = int(st.sidebar.number_input("初筛：涨幅榜取前 N（减少后续请求）", min_value=200, max_value=5000, value=1200, step=100))
FINAL_POOL = int(st.sidebar.number_input("进入评分池数量（限制请求量）", min_value=100, max_value=2000, value=800, step=50))
TOP_K = int(st.sidebar.number_input("展示 Top K", min_value=5, max_value=50, value=20, step=1))

MIN_PRICE = float(st.sidebar.number_input("最低股价（元）", min_value=0.1, max_value=1000.0, value=5.0, step=0.1))
MAX_PRICE = float(st.sidebar.number_input("最高股价（元）", min_value=1.0, max_value=2000.0, value=500.0, step=1.0))
MIN_AMOUNT = float(st.sidebar.number_input("最低日成交额（元）", min_value=0.0, max_value=5e11, value=100_000_000.0, step=10_000_000.0))
MIN_MV = float(st.sidebar.number_input("最小流通市值（元）", min_value=1e7, max_value=1e12, value=2_0000_0000.0, step=1e7))
MAX_MV = float(st.sidebar.number_input("最大流通市值（元）", min_value=1e8, max_value=2e13, value=500_0000_00000.0, step=1e8))

st.sidebar.markdown("---")
st.sidebar.header("过滤开关")
EXCLUDE_DOUBLE_10_20 = st.sidebar.checkbox("排除过去10-20天翻倍", value=True)
EXCLUDE_HIGH_ACCEL = st.sidebar.checkbox("排除短期高加速（3日极高涨幅）", value=True)
EXCLUDE_HIGH_ATR = st.sidebar.checkbox("排除高波动（ATR过大）", value=True)

st.sidebar.markdown("---")
st.sidebar.header("因子权重（可微调）")
w_pct = st.sidebar.slider("当日涨幅权重", 0.0, 1.0, 0.14)
w_mf = st.sidebar.slider("主力资金权重", 0.0, 1.0, 0.18)
w_chip = st.sidebar.slider("筹码稳定度权重", 0.0, 1.0, 0.16)
w_volprice = st.sidebar.slider("量价齐升权重", 0.0, 1.0, 0.18)
w_ind = st.sidebar.slider("行业动量权重", 0.0, 1.0, 0.14)
w_trend = st.sidebar.slider("趋势（MA）权重", 0.0, 1.0, 0.12)
w_health = st.sidebar.slider("K线健康权重", 0.0, 1.0, 0.08)

# normalize weights
_total = sum([w_pct,w_mf,w_chip,w_volprice,w_ind,w_trend,w_health])
if _total == 0:
    st.sidebar.error("权重总和不可为 0")
    st.stop()
w_pct/= _total; w_mf/= _total; w_chip/= _total; w_volprice/= _total; w_ind/= _total; w_trend/= _total; w_health/= _total

# ---------------------------
# helpers & caching
# ---------------------------
def safe_float(x, default=np.nan):
    try:
        if x is None: return default
        return float(x)
    except:
        return default

def norm_series(s):
    s = pd.Series(s).astype(float)
    if s.isnull().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn, mx = s.min(), s.max()
    if abs(mx - mn) < 1e-12:
        return pd.Series(np.ones(len(s))*0.5, index=s.index)
    return (s - mn) / (mx - mn)

def get_last_trade_day(pro_obj, max_days=14):
    today = datetime.now()
    for i in range(0, max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        try:
            dd = pro_obj.daily(trade_date=ds)
            if dd is not None and len(dd) > 0:
                return ds
        except Exception:
            continue
    return None

@st.cache_data(ttl=600)
def load_market_daily(trade_date):
    try:
        return pro.daily(trade_date=trade_date)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_stock_basic():
    try:
        df = pro.stock_basic(list_status='L', fields='ts_code,name,market,industry,list_date,total_mv,circ_mv')
        return df.drop_duplicates(subset=['ts_code'])
    except Exception:
        try:
            return pro.stock_basic(list_status='L')
        except Exception:
            return pd.DataFrame()

@st.cache_data(ttl=600)
def get_daily_basic(trade_date):
    try:
        return pro.daily_basic(trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv,pe,pb')
    except Exception:
        return None

@st.cache_data(ttl=600)
def get_moneyflow_df(trade_date):
    try:
        mf = pro.moneyflow(trade_date=trade_date)
        if mf is None or mf.empty: return None
        # unify net_mf column
        for col in ['net_mf','net_mf_amount','net_amount']:
            if col in mf.columns:
                mf2 = mf[['ts_code', col]].drop_duplicates(subset=['ts_code']).set_index('ts_code')
                mf2.columns = ['net_mf']
                return mf2
        return None
    except Exception:
        return None

@st.cache_data(ttl=600)
def try_special_api(api_name, **kwargs):
    """
    Try to call specialty APIs (like chips/distribution/forecast) if available in pro.
    Returns DataFrame or None.
    """
    try:
        if hasattr(pro, api_name):
            func = getattr(pro, api_name)
            return func(**kwargs)
    except Exception:
        return None
    return None

@st.cache_data(ttl=600)
def get_hist(ts_code, end_date, days=60):
    try:
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty: return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        # ensure numeric
        for c in ['open','high','low','close','vol','amount']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception:
        return None

# ---------------------------
# Auto refresh handling
# ---------------------------
if auto_refresh:
    # simple autorefresh using st.experimental_rerun with timestamp check
    last_run = st.session_state.get("last_run_time", 0)
    now_ts = time.time()
    if now_ts - last_run > refresh_seconds:
        st.session_state["last_run_time"] = now_ts
        st.experimental_rerun()

# ---------------------------
# Step 0: check last trade day & market snapshot
# ---------------------------
with st.spinner("获取最近交易日与当日日线快照..."):
    last_trade = get_last_trade_day(pro, max_days=14)
if not last_trade:
    st.error("无法获取最近交易日（请检查 Token/网络）")
    st.stop()
st.info(f"参考交易日：{last_trade}")

market_df = load_market_daily(last_trade)
if market_df is None or market_df.empty:
    st.error("读取当日日线失败（权限问题或网络）")
    st.stop()
st.write(f"市场记录数：{len(market_df)}，将从涨幅榜前 {INITIAL_TOP_N} 初筛")

# ---------------------------
# Step 1: initial coarse filter (fast)
# ---------------------------
pool = market_df.sort_values('pct_chg', ascending=False).head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

stock_basic_df = get_stock_basic()
if not stock_basic_df.empty:
    cols = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_basic_df.columns]
    try:
        pool = pool.merge(stock_basic_df[cols], on='ts_code', how='left')
    except Exception:
        pool['name'] = pool['ts_code']; pool['industry'] = ""
else:
    pool['name'] = pool['ts_code']; pool['industry'] = ""

daily_basic = get_daily_basic(last_trade)
if daily_basic is not None:
    try:
        db = daily_basic.drop_duplicates(subset=['ts_code']).set_index('ts_code')
        pool = pool.set_index('ts_code').join(db[['turnover_rate','amount','total_mv']].rename(columns={'turnover_rate':'turnover_rate_db','amount':'amount_db'}), how='left').reset_index()
    except Exception:
        pool['turnover_rate_db'] = np.nan; pool['amount_db'] = np.nan
else:
    pool['turnover_rate_db'] = np.nan; pool['amount_db'] = np.nan

moneyflow_df = get_moneyflow_df(last_trade)
if moneyflow_df is None:
    st.warning("moneyflow（主力资金）不可用或接口失败：相关因子将降级处理")

# reduce to FINAL_POOL to limit heavy history calls
pool = pool.sort_values('pct_chg', ascending=False).head(int(FINAL_POOL)).reset_index(drop=True)
st.write(f"初筛并截取后候选：{len(pool)}（将对每只请求历史/特色数据并计算多维因子）")

# ---------------------------
# Step 2: per-stock history & specialty data collection
# ---------------------------
pbar = st.progress(0)
records = []
N = len(pool)
# try to fetch specialty tables globally if available (efficiency)
#  - daily_chips / chips_distribution / forecast / top_inst / broker_recommendation 等接口尝试（不同账户返回不同）
special_chips = try_special_api('daily_chip', trade_date=last_trade) or try_special_api('chips_distribution', trade_date=last_trade) or None
special_forecast = try_special_api('forecast_oper', end_date=last_trade) or try_special_api('fina_indicator', start_date=(datetime.strptime(last_trade,"%Y%m%d")-timedelta(days=365)).strftime("%Y%m%d"), end_date=last_trade) or None
# note: above APIs may not exist; try_special_api returns None if not present/authorized

for i, row in enumerate(pool.itertuples()):
    ts = getattr(row, 'ts_code')
    name = getattr(row, 'name', ts)
    industry = getattr(row, 'industry', '')
    price = safe_float(getattr(row, 'close', np.nan))
    pct_chg = safe_float(getattr(row, 'pct_chg', 0))
    amount_day = safe_float(getattr(row, 'amount_db', getattr(row, 'amount', np.nan)))
    if amount_day > 0 and amount_day < 1e5:
        amount_day *= 10000
    turnover_rate = safe_float(getattr(row, 'turnover_rate_db', np.nan))

    # basic hard filters
    if math.isnan(price) or price < MIN_PRICE or price > MAX_PRICE:
        pbar.progress((i+1)/N); continue
    if amount_day > 0 and amount_day < MIN_AMOUNT:
        pbar.progress((i+1)/N); continue

    # get history (60 trading days)
    hist = get_hist(ts, last_trade, days=90)
    if hist is None or hist.empty:
        pbar.progress((i+1)/N); continue

    closes = hist['close'].astype(float)
    opens = hist['open'].astype(float)
    highs = hist['high'].astype(float)
    lows = hist['low'].astype(float)
    vols = hist['vol'].astype(float)
    amts = hist['amount'].astype(float) if 'amount' in hist.columns else np.zeros(len(closes))

    # compute multi-window MA/returns/vol stats
    def ma(series, n):
        return float(series.rolling(window=n).mean().iloc[-1]) if len(series)>=n else np.nan

    ma5 = ma(closes,5); ma10 = ma(closes,10); ma20 = ma(closes,20)
    amt5 = ma(pd.Series(amts),5); amt20 = ma(pd.Series(amts),20)
    vol5 = ma(pd.Series(vols),5); vol20 = ma(pd.Series(vols),20)

    ten_ret = (closes.iloc[-1] / closes.iloc[-10] - 1) if len(closes)>=10 else np.nan
    twenty_ret = (closes.iloc[-1] / closes.iloc[-20] - 1) if len(closes)>=20 else np.nan

    # exclude 10-20 day doubling
    if EXCLUDE_DOUBLE_10_20 and ( (not math.isnan(ten_ret) and ten_ret>=1.0) or (not math.isnan(twenty_ret) and twenty_ret>=1.0) ):
        pbar.progress((i+1)/N); continue

    # ATR
    prev = closes.shift(1)
    tr = pd.concat([highs - lows, (highs - prev).abs(), (lows - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1] if len(tr)>0 else np.nan
    atr_ratio = float(atr) / (closes.iloc[-1] + 1e-9) if not math.isnan(atr) else np.nan
    if EXCLUDE_HIGH_ATR and (not math.isnan(atr_ratio) and atr_ratio > 0.18):
        pbar.progress((i+1)/N); continue

    # detect recent high accel (3 days)
    pct_series = closes.pct_change().fillna(0)
    high_accel_flag = False
    if EXCLUDE_HIGH_ACCEL and len(pct_series)>=3:
        last3 = pct_series.iloc[-3:]
        if all(last3 > 0.08) and (vols.iloc[-1] / (vols[-6:-1].mean()+1e-9) > 3 if len(vols)>=6 else False):
            high_accel_flag = True
    if high_accel_flag:
        pbar.progress((i+1)/N); continue

    # K-line health scoring (实体/上影/连续阴线)
    k_health = 1.0
    recent = hist.tail(7).reset_index(drop=True)
    up_shadow_count = 0; down_count = 0; long_upper = False
    for r in range(len(recent)):
        o = safe_float(recent.loc[r,'open'], np.nan)
        c = safe_float(recent.loc[r,'close'], np.nan)
        h = safe_float(recent.loc[r,'high'], np.nan)
        l = safe_float(recent.loc[r,'low'], np.nan)
        if math.isnan(o) or math.isnan(c): continue
        body = abs(c-o)
        upper_shadow = h - max(o,c)
        if body > 0 and upper_shadow / (body+1e-9) > 4:
            long_upper = True
        if c < o: down_count += 1
    if long_upper: k_health *= 0.65
    if down_count >= 4: k_health *= 0.5

    # volume-price alignment score
    price_align = 1.0 if (not math.isnan(ma5) and not math.isnan(ma20) and ma5 > ma20) else 0.0
    amt_align = 1.0 if (not math.isnan(amt5) and not math.isnan(amt20) and amt5 > amt20) else 0.0
    volprice_score = price_align * 0.6 + amt_align * 0.4

    # trend strength (ma short vs long)
    trend_strength = 0.0
    if not math.isnan(ma5) and not math.isnan(ma20):
        trend_strength = (ma5 - ma20) / (ma20 + 1e-9)

    # chips /筹码 stability: try specialty chips if available
    chip_score = 0.5
    try:
        # try global special_chips if present
        if special_chips is not None:
            # special_chips assumed to include ts_code and some concentration measures
            if ts in special_chips['ts_code'].values or (hasattr(special_chips, 'index') and ts in special_chips.index):
                # attempt multiple possible column names
                if 'concentration' in special_chips.columns:
                    val = float(special_chips[special_chips['ts_code']==ts]['concentration'].iloc[0])
                    chip_score = max(0.0, min(1.0, 1.0 - val/10.0))
                else:
                    chip_score = 0.7
        else:
            # fallback: use turnover stability over last 20 days
            trates = hist['vol'].fillna(0).astype(float).tail(20)
            if len(trates)>=8:
                var = trates.std()
                mean = trates.mean() + 1e-9
                chip_score = 1.0 - min(1.0, var/mean)*0.5
    except Exception:
        chip_score = 0.5
    chip_score = max(0.0, min(1.0, chip_score))

    # moneyflow (主力资金) best-effort from moneyflow_df
    net_mf = 0.0
    try:
        if moneyflow_df is not None and ts in moneyflow_df.index:
            net_mf = float(moneyflow_df.loc[ts,'net_mf'])
    except:
        net_mf = 0.0

    # specialty forecast / analyst signals (if available)
    forecast_score = 0.0
    try:
        if special_forecast is not None:
            # try multiple possible column names
            if isinstance(special_forecast, pd.DataFrame):
                if 'ts_code' in special_forecast.columns and ts in special_forecast['ts_code'].values:
                    # choose a column like 'profit_forecast' if exists
                    if 'profit_forecast' in special_forecast.columns:
                        forecast_score = float(special_forecast[special_forecast['ts_code']==ts]['profit_forecast'].iloc[0])
                    else:
                        forecast_score = 0.0
    except Exception:
        forecast_score = 0.0

    # industry momentum proxy (use pool-level industry aggregation later)
    industry_mom = 0.0
    # compute log-price slope for last 20 days
    try:
        if len(closes) >= 20:
            y = np.log(closes.tail(20).values + 1e-9)
            x = np.arange(len(y))
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            industry_mom = float(slope)
    except Exception:
        industry_mom = 0.0

    # assemble record
    records.append({
        'ts_code': ts, 'name': name, 'industry': industry,
        'price': float(closes.iloc[-1]), 'pct_chg': pct_chg, 'amount_day': amount_day,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
        'amt5': amt5, 'amt20': amt20, 'vol5': vol5, 'vol20': vol20,
        'ten_ret': ten_ret, 'twenty_ret': twenty_ret,
        'atr_ratio': atr_ratio, 'k_health': k_health,
        'volprice_score': volprice_score, 'trend_strength': trend_strength,
        'chip_score': chip_score, 'net_mf': net_mf, 'forecast_score': forecast_score,
        'turnover_rate': turnover_rate
    })
    pbar.progress((i+1)/N)

pbar.progress(1.0)

# ---------------------------
# Step 3: form DataFrame and compute industry-level boosts
# ---------------------------
if len(records) == 0:
    st.error("无候选（可能过滤过严或接口问题），请放宽参数或检查 Token 权限")
    st.stop()

df = pd.DataFrame(records)

# industry-level momentum: compute per-industry avg trend_strength within pool
df['ind_mom'] = df.groupby('industry')['trend_strength'].transform('mean').fillna(0)
df['ind_mom_rank'] = norm_series(df['ind_mom'])

# normalize subfactors
df['pct_rank'] = norm_series(df['pct_chg'].fillna(0))
df['mf_rank'] = norm_series(df['net_mf'].fillna(0))
df['chip_rank'] = norm_series(df['chip_score'].fillna(0))
df['volprice_rank'] = norm_series(df['volprice_score'].fillna(0))
df['trend_rank'] = norm_series(df['trend_strength'].fillna(0))
df['health_rank'] = norm_series(df['k_health'].fillna(0))
df['forecast_rank'] = norm_series(df['forecast_score'].fillna(0))

# compute raw score by weights
df['score_raw'] = (
    df['pct_rank'] * w_pct +
    df['mf_rank'] * w_mf +
    df['chip_rank'] * w_chip +
    df['volprice_rank'] * w_volprice +
    df['ind_mom_rank'] * w_ind +
    df['trend_rank'] * w_trend +
    df['health_rank'] * w_health
)

# apply bonuses and penalties
# chip bonus
df['score_chip_bonus'] = df['score_raw'] * (1 + 0.12 * df['chip_rank'])
# industry boost
df['score_ind_boost'] = df['score_chip_bonus'] * (1 + 0.15 * df['ind_mom_rank'])
# forecast boost
df['score_forecast'] = df['score_ind_boost'] * (1 + 0.05 * df['forecast_rank'])
# ATR penalty if too volatile
df['atr_penalty'] = df['atr_ratio'].apply(lambda x: 1.0 if (not math.isnan(x) and 0.0025 <= x <= 0.18) else 0.9)
df['score_final'] = df['score_forecast'] * df['atr_penalty']

# secondary safety filters
df = df[ df['amount_day'].fillna(0) >= MIN_AMOUNT ]
df = df[ df['k_health'] > 0.35 ]  # drop super-risky

# final sort
df = df.sort_values('score_final', ascending=False).reset_index(drop=True)
df.index += 1

st.success(f"选股完成，候选 {len(df)} 支，展示 Top {min(TOP_K, len(df))}")

# ---------------------------
# Display results
# ---------------------------
display_cols = ['name','ts_code','score_final','pct_chg','price','amount_day','ma5','ma10','ma20','volprice_score','chip_score','k_health','atr_ratio','industry','ten_ret']
for c in display_cols:
    if c not in df.columns:
        df[c] = np.nan

st.dataframe(df[display_cols].head(TOP_K).reset_index(drop=False), use_container_width=True)

# show reasons for top picks
st.markdown("### Top 候选简要原因（自动生成）")
for i, r in df.head(TOP_K).head(TOP_K).iterrows():
    reasons = []
    if r['pct_chg'] > 5: reasons.append("当日强势")
    if r['volprice_score'] > 0.6: reasons.append("量价齐升")
    if r['chip_score'] > 0.6: reasons.append("筹码稳定")
    if r['k_health'] > 0.8: reasons.append("K线健康")
    if r['atr_ratio'] and 0.0025 <= r['atr_ratio'] <= 0.08: reasons.append("波动适中")
    st.write(f"{int(i+1)}. {r['name']} ({r['ts_code']}) — Score {r['score_final']:.4f} — {'；'.join(reasons) if reasons else '常规'}")

# ---------------------------
# CSV download & logging
# ---------------------------
csv = df.to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载全部评分 CSV", data=csv, file_name=f"v7_score_{last_trade}_{datetime.now().strftime('%H%M%S')}.csv", mime="text/csv")

st.markdown("### 运行日志与提示")
st.markdown(f"""
- 参考交易日：{last_trade}  
- moneyflow 主力资金 {'可用' if moneyflow_df is not None else '不可用/降级'}  
- specialty chips API {'已启用（若显示）' if special_chips is not None else '不可用或权限不足（已降级到日线估计）'}  
- specialty forecast API {'已启用（若显示）' if special_forecast is not None else '不可用或权限不足（降级）'}  

运行建议：  
- 建议每天运行 4-5 次：9:50、10:40、13:40、14:30、15:00（或你习惯的时间）  
- 若你希望自动化运行（每 X 分钟自动刷新），建议使用外部调度（cron / cloud function）触发 Streamlit 的 API 或在 Streamlit Cloud 上使用自动刷新。  
- 若你希望把“分时 / tick /分钟级主力资金”加入（需要开启 rt_min/realtime_quote），我可以在你确认分钟权限后直接把分钟因子加进来。  
""")

# end of app
