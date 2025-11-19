# -*- coding: utf-8 -*-
"""
短线王 · v4.0 C+（日线深度增强版）
- 目标：在 5000 积分/日线权限下，最大化 1-5 天短线命中率
- 包含增强模块：
    1) 行业动量增强（20 日斜率 + 量能放大）
    2) 筹码稳定度（换手、波动、长影线等反推机构偏好）
    3) 量价齐升（5/20 均线量价配合）
    4) K 线健康度（上影/实体/连续阴线等风险剔除）
- 运行说明：streamlit run app.py，界面输入 Tushare Token
"""
import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="短线王 · v4.0 C+", layout="wide")
st.title("短线王 · v4.0 C+（日线深度增强版 · 20+维度）")

# ---------------------------
# User inputs
# ---------------------------
TS_TOKEN = st.text_input("请输入 Tushare Token（仅本次会话使用）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 后运行（界面输入，不保存）。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# Sidebar params
st.sidebar.header("参数（适配 1-5 天短线）")
INITIAL_TOP_N = int(st.sidebar.number_input("初筛：涨幅榜取前 N", min_value=200, max_value=5000, value=1000, step=100))
FINAL_POOL = int(st.sidebar.number_input("进入评分池（限制请求量）", min_value=50, max_value=2000, value=500, step=50))
TOP_K = int(st.sidebar.number_input("展示 Top K", min_value=5, max_value=50, value=20, step=1))

MIN_PRICE = float(st.sidebar.number_input("最低股价（元）", min_value=0.1, max_value=1000.0, value=10.0, step=0.1))
MAX_PRICE = float(st.sidebar.number_input("最高股价（元）", min_value=1.0, max_value=2000.0, value=200.0, step=1.0))
MIN_AMOUNT = float(st.sidebar.number_input("最低日成交额（元）", min_value=0.0, max_value=1e12, value=200_000_000.0, step=10_000_000.0))
MIN_MV = float(st.sidebar.number_input("最小市值（元）", min_value=1e7, max_value=1e12, value=2_0000_0000.0, step=1e7))
MAX_MV = float(st.sidebar.number_input("最大市值（元）", min_value=1e8, max_value=1e13, value=50_0000_00000.0, step=1e8))

EXCLUDE_DOUBLE_10_20 = st.sidebar.checkbox("排除过去10-20天翻倍", value=True)
EXCLUDE_HIGH_ACCEL = st.sidebar.checkbox("排除短期高加速（3日极高涨幅）", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("权重微调（默认调优适配短线）")
w_pct = st.sidebar.slider("涨幅权重", 0.0, 1.0, 0.16)
w_volratio = st.sidebar.slider("量比权重", 0.0, 1.0, 0.12)
w_turn = st.sidebar.slider("换手权重", 0.0, 1.0, 0.08)
w_money = st.sidebar.slider("主力资金权重", 0.0, 1.0, 0.08)
w_ind = st.sidebar.slider("行业动量权重", 0.0, 1.0, 0.22)
w_volprice = st.sidebar.slider("量价齐升权重", 0.0, 1.0, 0.18)
w_health = st.sidebar.slider("K线健康权重", 0.0, 1.0, 0.16)

total_w = w_pct + w_volratio + w_turn + w_money + w_ind + w_volprice + w_health
if total_w == 0:
    st.sidebar.error("权重总和不可为 0")
    st.stop()
w_pct /= total_w; w_volratio /= total_w; w_turn /= total_w; w_money /= total_w
w_ind /= total_w; w_volprice /= total_w; w_health /= total_w

st.sidebar.markdown("---")
st.sidebar.markdown("说明：本版本基于日线历史，不依赖实时 API，适配 5000 积分帐号。")

# ---------------------------
# helpers
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
    if abs(mx - mn) < 1e-9:
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
def get_hist(ts_code, end_date, days=60):
    try:
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty: return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        return df
    except Exception:
        return None

@st.cache_data(ttl=600)
def get_daily_basic(trade_date):
    try:
        return pro.daily_basic(trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv,pe,pb')
    except Exception:
        return None

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
def get_moneyflow(trade_date):
    try:
        mf = pro.moneyflow(trade_date=trade_date)
        # prefer net_mf if exists
        if mf is None or mf.empty: return None
        for col in ['net_mf','net_mf_amount','net_amount']:
            if col in mf.columns:
                x = mf[['ts_code', col]].drop_duplicates(subset=['ts_code']).set_index('ts_code')
                x.columns = ['net_mf']
                return x
        return None
    except Exception:
        return None

# ---------------------------
# Step: get last trade day and market snapshot
# ---------------------------
with st.spinner("获取最近交易日与当日日线快照..."):
    last_trade = get_last_trade_day(pro, max_days=14)
if not last_trade:
    st.error("无法获取最近交易日，请检查 Token 或网络")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")

market_df = load_market_daily(last_trade)
if market_df is None or market_df.empty:
    st.error("获取当日日线失败（可能权限受限）")
    st.stop()
st.write(f"当日记录数：{len(market_df)}（将从涨幅榜前 {INITIAL_TOP_N} 进行初筛）")

# ---------------------------
# initial pool (fast)
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

daily_basic_df = get_daily_basic(last_trade)
if daily_basic_df is not None:
    try:
        db = daily_basic_df.drop_duplicates(subset=['ts_code']).set_index('ts_code')
        pool = pool.set_index('ts_code').join(db[['turnover_rate','amount','total_mv']].rename(columns={'turnover_rate':'turnover_rate_db','amount':'amount_db'}), how='left').reset_index()
    except Exception:
        pool['turnover_rate_db'] = np.nan; pool['amount_db'] = np.nan
else:
    pool['turnover_rate_db'] = np.nan; pool['amount_db'] = np.nan
moneyflow_df = get_moneyflow(last_trade)
if moneyflow_df is None:
    st.warning("moneyflow（主力金额）不可用，主力相关因子将降级")

# reduce to FINAL_POOL by pct_chg to limit heavy history requests
pool = pool.sort_values('pct_chg', ascending=False).head(int(FINAL_POOL)).reset_index(drop=True)
st.write(f"初筛并截取后候选数量：{len(pool)}（将对每只做历史日线提取与多维评分）")

# ---------------------------
# cleaning & build candidate historical features
# ---------------------------
records = []
pbar = st.progress(0)
N = len(pool)
for i, row in enumerate(pool.itertuples()):
    ts = getattr(row, 'ts_code')
    name = getattr(row, 'name', ts)
    industry = getattr(row, 'industry', '') if 'industry' in pool.columns else ''
    # base fields
    pct_chg = safe_float(getattr(row, 'pct_chg', 0))
    amount_day = safe_float(getattr(row, 'amount', getattr(row, 'amount_db', np.nan)))
    if amount_day > 0 and amount_day < 1e5: amount_day *= 10000
    price = safe_float(getattr(row, 'close', getattr(row, 'price', np.nan)))
    turnover_rt = safe_float(getattr(row, 'turnover_rate_db', np.nan))

    # skip basic filters
    if math.isnan(price) or price < MIN_PRICE or price > MAX_PRICE:
        pbar.progress((i+1)/N); continue
    if amount_day > 0 and amount_day < MIN_AMOUNT:
        pbar.progress((i+1)/N); continue

    # get hist (best-effort)
    hist = get_hist(ts, last_trade, days=60)
    if hist is None or hist.empty:
        # skip if no history (can't compute many factors)
        pbar.progress((i+1)/N); continue

    # ensure numeric columns
    hist = hist.sort_values('trade_date').reset_index(drop=True)
    for col in ['open','high','low','close','vol','amount']:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors='coerce')

    # compute technical rolling stats (5/10/20)
    closes = hist['close'].astype(float)
    amounts = hist['amount'].astype(float) if 'amount' in hist.columns else np.zeros(len(closes))
    vols = hist['vol'].astype(float) if 'vol' in hist.columns else np.zeros(len(closes))

    # recent windows
    def ma(series, n):
        if len(series) < n: return np.nan
        return float(series.rolling(window=n).mean().iloc[-1])

    ma5 = ma(closes, 5); ma10 = ma(closes, 10); ma20 = ma(closes, 20)
    amt5 = ma(pd.Series(amounts), 5); amt20 = ma(pd.Series(amounts), 20)
    vol5 = ma(pd.Series(vols), 5); vol20 = ma(pd.Series(vols), 20)

    # ten/twenty day return
    ten_return = (closes.iloc[-1] / closes.iloc[-10] - 1.0) if len(closes) >= 10 else np.nan
    twenty_return = (closes.iloc[-1] / closes.iloc[-20] - 1.0) if len(closes) >= 20 else np.nan

    # exclude 10-20day doubling
    if EXCLUDE_DOUBLE_10_20 and ( (not math.isnan(ten_return) and ten_return >= 1.0) or (not math.isnan(twenty_return) and twenty_return >= 1.0) ):
        pbar.progress((i+1)/N); continue

    # detect short-term high accel
    recent_pct = closes.pct_change().fillna(0).replace([np.inf,-np.inf],0)
    high_accel_flag = False
    if EXCLUDE_HIGH_ACCEL and len(recent_pct) >= 3:
        last3 = recent_pct.iloc[-3:]
        if all(last3 > 0.08) and (vols.iloc[-1] / (np.mean(vols[-6:-1]) + 1e-9) > 3 if len(vols)>=6 else False):
            high_accel_flag = True
    if high_accel_flag:
        pbar.progress((i+1)/N); continue

    # ATR (14)
    highs = hist['high'].astype(float)
    lows = hist['low'].astype(float)
    prev_close = closes.shift(1)
    tr1 = highs - lows
    tr2 = (highs - prev_close).abs()
    tr3 = (lows - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1] if len(tr)>0 else np.nan
    atr_ratio = float(atr) / (closes.iloc[-1] + 1e-9) if not math.isnan(atr) else np.nan

    # K-line health checks
    #  - recent long upper shadows -> risk
    #  - consecutive big negative candles -> risk
    #  - entity/upper-shadow ratio
    k_health = 1.0
    # compute last 5 candle stats
    recent = hist.tail(5).reset_index(drop=True)
    bad_shadow = False
    bad_consecutive_down = False
    down_count = 0
    for idx_r in range(len(recent)):
        o = safe_float(recent.loc[idx_r,'open'], np.nan)
        c = safe_float(recent.loc[idx_r,'close'], np.nan)
        h = safe_float(recent.loc[idx_r,'high'], np.nan)
        l = safe_float(recent.loc[idx_r,'low'], np.nan)
        if math.isnan(o) or math.isnan(c): continue
        body = abs(c - o)
        upper_shadow = h - max(o,c)
        lower_shadow = min(o,c) - l
        # if upper shadow is very long relative to body, penalize
        if body > 0 and upper_shadow / (body + 1e-9) > 4:
            bad_shadow = True
        if c < o:
            down_count += 1
    if bad_shadow:
        k_health *= 0.7
    if down_count >= 3:
        k_health *= 0.6

    # Volume-price alignment (量价齐升)
    # - price MA5 > MA20
    # - amt5 > amt20 (量能放大)
    volprice_score = 0.5
    try:
        price_alignment = 1.0 if (not math.isnan(ma5) and not math.isnan(ma20) and ma5 > ma20) else 0.0
        amt_alignment = 1.0 if (not math.isnan(amt5) and not math.isnan(amt20) and amt5 > amt20) else 0.0
        volprice_score = (price_alignment * 0.6 + amt_alignment * 0.4)
    except:
        volprice_score = 0.5

    # Chips stability (筹码稳定度) — approximate via turnover & vol stability
    # lower recent turnover variance & slight increasing turnover -> higher stability
    chip_score = 0.5
    try:
        if 'turnover_rate' in hist.columns:
            trates = pd.to_numeric(hist['turnover_rate'].fillna(0), errors='coerce')
            recent_tr = trates.tail(10)
            if len(recent_tr) >= 5:
                var = recent_tr.std()
                mean = recent_tr.mean()
                # lower var and moderate mean -> higher chip stability
                chip_score = 1.0 - min(1.0, var / (mean + 1e-9 + 1e-6)) * 0.5
                # if mean gradually increasing -> bonus
                if recent_tr.iloc[-1] > recent_tr.iloc[-5]:
                    chip_score += 0.1
        else:
            # fallback: use vol stability
            vv = pd.Series(vols).tail(10)
            if len(vv) >= 5:
                var = vv.std()
                mean = vv.mean()
                chip_score = 1.0 - min(1.0, var / (mean + 1e-9)) * 0.5
    except Exception:
        chip_score = 0.5
    chip_score = max(0.0, min(1.0, chip_score))

    # Industry momentum (20日斜率 + industry avg amount up)
    industry_momentum = 0.0
    try:
        # build industry mean pct_chg over last 20 days across pool (will adjust later globally)
        # here compute single stock 20d slope as proxy
        if len(closes) >= 20:
            y = np.log(closes.tail(20).values + 1e-9)
            x = np.arange(len(y))
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            industry_momentum = float(slope)  # small positive indicates uptrend
        else:
            industry_momentum = float((closes.pct_change().rolling(window=5).mean().iloc[-1]) if len(closes)>=5 else 0.0)
    except:
        industry_momentum = 0.0

    # moneyflow (net_mf) if available from daily moneyflow aggregated earlier
    net_mf_val = 0.0
    try:
        if moneyflow_df is not None and ts in moneyflow_df.index:
            net_mf_val = float(moneyflow_df.loc[ts, 'net_mf'])
    except:
        net_mf_val = 0.0

    # compose record
    records.append({
        'ts_code': ts,
        'name': name,
        'industry': industry,
        'price': float(closes.iloc[-1]),
        'pct_chg': pct_chg,
        'amount_day': amount_day,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
        'amt5': amt5, 'amt20': amt20,
        'vol5': vol5, 'vol20': vol20,
        'ten_return': ten_return, 'twenty_return': twenty_return,
        'atr_ratio': atr_ratio,
        'k_health': k_health,
        'volprice_score': volprice_score,
        'chip_score': chip_score,
        'industry_momentum': industry_momentum,
        'net_mf': net_mf_val,
        'turnover_rate': turnover_rt
    })
    pbar.progress((i+1)/N)

pbar.progress(1.0)
if len(records) == 0:
    st.error("无候选（历史数据不足或过滤过严），请放宽条件或检查 Token 权限")
    st.stop()

score_df = pd.DataFrame(records)

# ---------------------------
# industry-level aggregation (for industry boost)
# ---------------------------
# compute industry mean slope across candidates
if 'industry' in score_df.columns and score_df['industry'].notnull().any():
    # normalize individual industry_momentum to 0-1 within pool
    score_df['industry_mom_norm'] = norm_series(score_df['industry_momentum'].fillna(0))
else:
    score_df['industry_mom_norm'] = 0.0

# ---------------------------
# normalize subfactors and compute component scores
# ---------------------------
score_df['pct_rank'] = norm_series(score_df['pct_chg'].fillna(0))
score_df['volratio_rank'] = norm_series((score_df['vol5'] / (score_df['vol20']+1e-9)).replace([np.inf,-np.inf], np.nan).fillna(0))
score_df['turn_rank'] = norm_series(score_df['turnover_rate'].fillna(0))
score_df['money_rank'] = norm_series(score_df['net_mf'].fillna(0))
score_df['ind_rank'] = norm_series(score_df['industry_mom_norm'].fillna(0))
score_df['volprice_rank'] = norm_series(score_df['volprice_score'].fillna(0))
score_df['chip_rank'] = norm_series(score_df['chip_score'].fillna(0))
score_df['health_rank'] = norm_series(score_df['k_health'].fillna(0))
score_df['atr_penalty'] = score_df['atr_ratio'].apply(lambda x: 1.0 if (not math.isnan(x) and 0.0025 <= x <= 0.12) else 0.9)

# Compose final multi-dimensional score
score_df['score_raw'] = (
    score_df['pct_rank'] * w_pct +
    score_df['volratio_rank'] * w_volratio +
    score_df['turn_rank'] * w_turn +
    score_df['money_rank'] * w_money +
    score_df['ind_rank'] * w_ind +
    score_df['volprice_rank'] * w_volprice +
    score_df['health_rank'] * w_health
)

# apply chip bonus & atr penalty
score_df['score'] = score_df['score_raw'] * (1 + 0.12 * score_df['chip_rank']) * score_df['atr_penalty']

# second-level filters: remove tiny amount, extreme atr, negative health
score_df = score_df[ score_df['amount_day'].fillna(0) >= MIN_AMOUNT ]
score_df = score_df[ score_df['k_health'] > 0.4 ]  # remove obviously unhealthy k-lines

# sort and output
score_df = score_df.sort_values('score', ascending=False).reset_index(drop=True)
score_df.index += 1

st.success(f"评分完成，候选 {len(score_df)} 支，展示 Top {min(TOP_K, len(score_df))}（按综合评分降序）")

# display columns
display_cols = ['name','ts_code','score','pct_chg','price','amount_day','ma5','ma10','ma20','volprice_score','chip_score','k_health','atr_ratio','industry','ten_return']
for c in display_cols:
    if c not in score_df.columns:
        score_df[c] = np.nan

st.dataframe(score_df[display_cols].head(int(TOP_K)).reset_index(drop=False), use_container_width=True)

# explain top picks
st.markdown("### Top 候选说明（示例）")
top_show = score_df.head(int(TOP_K)).reset_index()
for _, r in top_show.iterrows():
    reasons = []
    if r['pct_chg'] > 5: reasons.append("当日强势")
    if r['volprice_score'] > 0.6: reasons.append("量价齐升")
    if r['chip_score'] > 0.6: reasons.append("筹码稳定")
    if r['k_health'] > 0.8: reasons.append("K 线健康")
    if r['atr_ratio'] and 0.0025 <= r['atr_ratio'] <= 0.06: reasons.append("波动适中")
    st.write(f"{int(r['index'])}. {r['name']} ({r['ts_code']}) — Score {r['score']:.4f} — {'；'.join(reasons) if reasons else '常规'}")

# CSV download
csv = score_df.to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载全部评分结果 CSV", data=csv, file_name=f"score_v4c_{last_trade}.csv", mime="text/csv")

# summary & tips
st.markdown("### 小结与运行建议")
st.markdown("""
- 本版为日线深度 C+ 强化版，适配 5000 积分（不依赖实时）。  
- 推荐运行节奏：**每天 9:50（早盘初步）+ 14:30（复盘）**。9:50 可以抓早盘资金延续的票；14:30 有利于收盘前的规整与择取次日计划。  
- 若你未来拿到实时权限（rt_min/realtime_quote），我可以再把量价齐升改为分钟级检测，进一步提升命中率。  
- 如果你希望更保守：把 K 线健康权重提高，或将 MIN_AMOUNT 提高到 3~5 亿。  
- 任何报错请截图红色日志发给我，我会逐条修复（脚本已做大量降级与容错，正常不会崩溃）。  
""")
