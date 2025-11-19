# -*- coding: utf-8 -*-
"""
短线王 · v4.0（盘中实时版）
说明：
- 优先使用 tushare 的实时接口： realtime_quote（单只实时快照） 和 rt_min（分钟线）
- 若实时接口不可用（无权限或异常），自动回退为日线历史并提示
- 策略：先用“昨收/涨幅榜”做粗筛 → 对候选池请求实时/分钟数据 → 应用更严格的短线过滤并评分
- 目标：在盘中任意时间点产生不同候选（9:40、10:00、10:05 等）
"""
import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import math
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="短线王 · v4.0（盘中实时）", layout="wide")
st.title("短线王 · v4.0（盘中实时） — 任意时刻可变结果")

# ---------------------------
# User inputs
# ---------------------------
TS_TOKEN = st.text_input("请输入你的 Tushare Token（仅本次会话）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 并回车以激活。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

st.sidebar.header("快速参数（适合 1-5 天短线）")
INITIAL_TOP_N = int(st.sidebar.number_input("初筛：涨幅榜取前 N（减少实时请求）", min_value=200, max_value=5000, value=1000, step=100))
FINAL_POOL = int(st.sidebar.number_input("进入实时评分池数量（最终候选）", min_value=50, max_value=2000, value=500, step=50))
TOP_K = int(st.sidebar.number_input("展示 Top K", min_value=5, max_value=50, value=20, step=1))

MIN_PRICE = float(st.sidebar.number_input("最低股价（元）", min_value=0.1, max_value=1000.0, value=10.0, step=0.1))
MAX_PRICE = float(st.sidebar.number_input("最高股价（元）", min_value=1.0, max_value=2000.0, value=200.0, step=1.0))
MIN_AMOUNT = float(st.sidebar.number_input("最低日成交额（元）", min_value=0.0, max_value=1e11, value=200_000_000.0, step=10_000_000.0))
MIN_MV = float(st.sidebar.number_input("最小市值（元）", min_value=1e7, max_value=1e12, value=20_0000_0000.0, step=1e7))
MAX_MV = float(st.sidebar.number_input("最大市值（元）", min_value=1e8, max_value=1e13, value=500_0000_00000.0, step=1e8))

EXCLUDE_DOUBLE_10_20 = st.sidebar.checkbox("排除过去10-20天翻倍", value=True)
EXCLUDE_IMMEDIATE_CHASE = st.sidebar.checkbox("排除今日短期暴拉（3日内连续大幅）", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("实时模式说明：若实时接口不可用，脚本会自动用日线回退并在界面提示（结果会变成日线快照）。")

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
    if s.isnull().all(): return pd.Series(np.zeros(len(s)), index=s.index)
    mn, mx = s.min(), s.max()
    if abs(mx - mn) < 1e-9: return pd.Series(np.ones(len(s))*0.5, index=s.index)
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

# try realtime_quote (single snapshot) for a ts_code
def try_realtime_quote(pro_obj, ts_code, src='sina'):
    # realtime_quote is a爬虫接口 in tushare; some accounts may not have it or the network may block
    try:
        df = pro_obj.realtime_quote(ts_code=ts_code, src=src)
        # df expected to contain fields like 'price','change','percent','volume','amount' depending on src
        return df
    except Exception as e:
        return None

# try minute bars for multiple codes at once (rt_min)
def try_rt_min(pro_obj, codes, freq='1MIN'):
    try:
        # rt_min may accept comma-separated ts_code; limit may apply
        df = pro_obj.rt_min(freq=freq, ts_code=",".join(codes))
        return df
    except Exception:
        return None

# fallback to pro.daily for a single code history
def try_hist_daily(pro_obj, ts_code, start_date, end_date):
    try:
        df = pro_obj.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty: return None
        return df.sort_values('trade_date').reset_index(drop=True)
    except Exception:
        return None

# ---------------------------
# Step 0: get last trade day and today's market snapshot (daily)
# ---------------------------
with st.spinner("获取最近交易日与当日日线快照（用于粗筛）..."):
    last_trade = get_last_trade_day(pro, max_days=14)
if not last_trade:
    st.error("无法获取最近交易日，请检查 Token 或网络")
    st.stop()
st.info(f"参考交易日（用于初筛）：{last_trade}")

# load market daily snapshot (this is a single API call)
@st.cache_data(ttl=60)
def load_market_daily(trade_date):
    try:
        return pro.daily(trade_date=trade_date)
    except Exception:
        return pd.DataFrame()

market_df = load_market_daily(last_trade)
if market_df is None or market_df.empty:
    st.error("读取当日日线失败（可能权限不足）——脚本无法继续")
    st.stop()
st.write(f"市场日线记录数（参考）：{len(market_df)}，将从涨幅榜前 {INITIAL_TOP_N} 初筛以减少实时请求")

# ---------------------------
# Step 1: initial top N from day snapshot (fast)
# ---------------------------
initial_pool = market_df.sort_values('pct_chg', ascending=False).head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

# attach basic info if available (stock_basic)
def try_get_stock_basic():
    try:
        df = pro.stock_basic(list_status='L', fields='ts_code,name,market,industry,list_date,total_mv,circ_mv')
        return df.drop_duplicates(subset=['ts_code'])
    except Exception:
        try:
            df = pro.stock_basic(list_status='L')
            return df.drop_duplicates(subset=['ts_code'])
        except Exception:
            return pd.DataFrame()

stock_basic_df = try_get_stock_basic()
if not stock_basic_df.empty:
    # merge safely
    cols = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_basic_df.columns]
    initial_pool = initial_pool.merge(stock_basic_df[cols], on='ts_code', how='left')
else:
    initial_pool['name'] = initial_pool['ts_code']
    initial_pool['industry'] = ""

# limit to FINAL_POOL by pct_chg to reduce realtime calls further
initial_pool = initial_pool.sort_values('pct_chg', ascending=False).head(int(FINAL_POOL)).reset_index(drop=True)
st.write(f"初筛并限制后候选数量（将做实时/分钟更新）：{len(initial_pool)}")

# ---------------------------
# Step 2: attempt to fetch realtime snapshots for candidates
# ---------------------------
use_realtime = True
realtime_available = True
realtime_fail_reasons = []
realtime_records = []

# We'll try two strategies:
# 1) Use pro.rt_min to fetch 1-minute bars for chunks (preferred for multi-code)
# 2) If rt_min unavailable, fall back to per-code realtime_quote (slower)
codes = initial_pool['ts_code'].astype(str).tolist()

st.info("开始获取盘中/分钟级数据（若权限不够将回退到日线）...")
progress = st.progress(0)
chunk_size = 80  # rt_min may limit; smaller chunk if errors
all_minute_frames = {}

try:
    # first try batch rt_min for all codes (may fail if no permission)
    # we fetch 1MIN for each code in chunks to avoid too large requests
    for i in range(0, len(codes), chunk_size):
        chunk = codes[i:i+chunk_size]
        try:
            rt = try_rt_min(pro, chunk, freq='1MIN')
            if rt is None or rt.empty:
                raise RuntimeError("rt_min returned empty or None")
            # rt likely contains multi-code minute lines. Ensure structure.
            # Save per-code tail minute frame
            for code in chunk:
                sub = rt[rt['ts_code'] == code].sort_values('trade_time').reset_index(drop=True) if 'ts_code' in rt.columns else pd.DataFrame()
                all_minute_frames[code] = sub
        except Exception:
            # if any chunk fails, flag and break to fallback
            realtime_available = False
            realtime_fail_reasons.append("rt_min 批量请求失败或无权限")
            break
except Exception:
    realtime_available = False
    realtime_fail_reasons.append("rt_min 全局异常")

if not realtime_available:
    # fallback to per-code realtime_quote (slower, but sometimes available)
    per_code_frames = {}
    try:
        for idx, code in enumerate(codes):
            dfrt = try_realtime_quote(pro, code, src='sina')
            # realtime_quote returns one-row snapshot; we wrap it as a tiny DataFrame with timestamp
            if dfrt is not None and not dfrt.empty:
                per_code_frames[code] = dfrt.reset_index(drop=True)
            # update progress
            progress.progress((idx+1)/len(codes))
            time.sleep(0.05)  # small sleep to avoid throttle
        if len(per_code_frames) == 0:
            realtime_available = False
            realtime_fail_reasons.append("realtime_quote 单票接口不可用或返回空")
        else:
            realtime_available = True
            all_minute_frames = {}  # we will rely on per_code_frames
    except Exception:
        realtime_available = False
        realtime_fail_reasons.append("realtime_quote 逐票请求失败")
else:
    progress.progress(1.0)

# If still not realtime_available, we will fall back to daily history per code
if not realtime_available:
    st.warning("实时 API 不可用或权限受限： " + "; ".join(realtime_fail_reasons))
    st.info("脚本将使用日线历史做近似盘中判断（结果会与真实盘中不同）")

# ---------------------------
# Step 3: build candidate records using best-available data
# ---------------------------
records = []
N = len(initial_pool)
pbar = st.progress(0)
for i, row in enumerate(initial_pool.itertuples()):
    ts = getattr(row, 'ts_code')
    name = getattr(row, 'name', ts)
    industry = getattr(row, 'industry', '')
    # base daily fields
    pct_chg = safe_float(getattr(row, 'pct_chg', 0))
    amount_day = safe_float(getattr(row, 'amount', 0))
    if amount_day > 0 and amount_day < 1e5: amount_day *= 10000

    # try to obtain minute-level info if available
    vol_ratio = np.nan
    recent_price = np.nan
    turnover_now = np.nan
    atr_ratio = np.nan
    ma_short = np.nan
    ma_long = np.nan
    trend_ok = False
    ten_return = np.nan
    net_mf = 0.0

    # preferred: minute bars from rt_min
    if realtime_available and ts in all_minute_frames and not all_minute_frames[ts].empty:
        mdf = all_minute_frames[ts].copy()
        # try fields: 'close','vol','amount' or 'trade' names depending on rt_min output
        # normalize column names
        if 'close' not in mdf.columns and 'price' in mdf.columns:
            mdf = mdf.rename(columns={'price':'close'})
        if 'vol' not in mdf.columns and 'volume' in mdf.columns:
            mdf = mdf.rename(columns={'volume':'vol'})
        if 'amount' not in mdf.columns and 'money' in mdf.columns:
            mdf = mdf.rename(columns={'money':'amount'})

        if 'close' in mdf.columns and len(mdf) >= 3:
            recent_price = safe_float(mdf['close'].iloc[-1])
            # compute simple MA on minute closes (short 5, long 15)
            closes = mdf['close'].astype(float)
            if len(closes) >= 5: ma_short = float(closes.rolling(window=5).mean().iloc[-1])
            if len(closes) >= 15: ma_long = float(closes.rolling(window=15).mean().iloc[-1])
            # ATR on minute
            try:
                highs = mdf['high'].astype(float) if 'high' in mdf.columns else closes
                lows = mdf['low'].astype(float) if 'low' in mdf.columns else closes
                prev = closes.shift(1)
                tr = pd.concat([highs - lows, (highs - prev).abs(), (lows - prev).abs()], axis=1).max(axis=1)
                atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]
                if recent_price > 0: atr_ratio = float(atr) / recent_price
            except Exception:
                atr_ratio = np.nan
            # vol ratio: current minute vol vs prev 5-min avg (if vol column exists)
            if 'vol' in mdf.columns:
                vols = mdf['vol'].astype(float).tolist()
                if len(vols) >= 6:
                    avg5 = np.mean(vols[-6:-1])
                    vol_ratio = vols[-1] / (avg5 + 1e-9)
            # trend: last 5 lows vs previous 5 lows
            if 'low' in mdf.columns:
                lows = mdf['low'].astype(float).tolist()
                if len(lows) >= 10:
                    if np.nanmean(lows[-5:]) > np.nanmean(lows[-10:-5]): trend_ok = True
        # attempt to get minute-based pct change if available
        if 'pct_chg' in mdf.columns:
            pct_chg = safe_float(mdf['pct_chg'].iloc[-1])
    else:
        # fallback: try per-code realtime_quote snapshot
        try:
            r = try_realtime_quote(pro, ts, src='sina')
            if r is not None and not r.empty:
                # try to parse price/pct/vol/amount
                srow = r.iloc[0]
                # different sources have different field names; try common keys
                recent_price = safe_float(srow.get('price', srow.get('now', srow.get('last_price', np.nan))))
                pct_chg = safe_float(srow.get('percent', srow.get('pct_chg', pct_chg)))
                # volume/amount fields may be named differently
                vol = safe_float(srow.get('volume', srow.get('vol', np.nan)))
                amount_now = safe_float(srow.get('amount', srow.get('money', np.nan)))
                # we can't compute minute ATR/MA from single snapshot; keep as NaN
            else:
                # no realtime available at all — use historical daily-derived approximations
                recent_price = safe_float(getattr(row, 'close', np.nan))
        except Exception:
            recent_price = safe_float(getattr(row, 'close', np.nan))

    # if we still don't have price, use daily close
    if math.isnan(recent_price):
        recent_price = safe_float(getattr(row, 'close', np.nan))

    # compute ten-day return using historical daily if available (best-effort)
    try:
        hist = try_hist_daily(pro, ts, (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d"), last_trade)
        if hist is not None and len(hist) >= 10:
            hclose = hist.sort_values('trade_date')['close'].astype(float).reset_index(drop=True)
            ten_return = float(hclose.iloc[-1]) / float(hclose.iloc[-10]) - 1.0
    except Exception:
        ten_return = np.nan

    # net_mf (moneyflow) if available from daily moneyflow (fallback)
    try:
        mf = pro.moneyflow(trade_date=last_trade) if 'pro' in globals() else None
        if mf is not None and 'net_mf' in mf.columns:
            mf_sub = mf[mf['ts_code'] == ts]
            if not mf_sub.empty:
                net_mf = safe_float(mf_sub.iloc[0]['net_mf'])
    except Exception:
        # ignore moneyflow failure (permissions)
        net_mf = 0.0

    # price bounds and amount/day checks (hard filters)
    if recent_price < MIN_PRICE or recent_price > MAX_PRICE:
        pbar.progress((i+1)/N)
        continue
    if amount_day > 0 and amount_day < MIN_AMOUNT:
        pbar.progress((i+1)/N)
        continue

    # exclude 10-20 day doubling if requested
    if EXCLUDE_DOUBLE_10_20 and not math.isnan(ten_return) and ten_return >= 1.0:
        pbar.progress((i+1)/N)
        continue

    # ATR-based filters (if available)
    if not math.isnan(atr_ratio):
        if atr_ratio > 0.12 or atr_ratio < 0.0025:
            pbar.progress((i+1)/N)
            continue

    # immediate high-accel exclusion: if today's pct_chg > 8% and vol_ratio huge and EXCLUDE_IMMEDIATE_CHASE
    if EXCLUDE_IMMEDIATE_CHASE and pct_chg is not None and pct_chg > 8 and (not math.isnan(vol_ratio) and vol_ratio > 3):
        pbar.progress((i+1)/N)
        continue

    records.append({
        'ts_code': ts,
        'name': name,
        'industry': industry,
        'price': recent_price,
        'pct_chg': pct_chg,
        'vol_ratio': vol_ratio if not math.isnan(vol_ratio) else np.nan,
        'turnover_rate': turnover_now,
        'net_mf': net_mf,
        'amount_day': amount_day,
        'ma_short': ma_short,
        'ma_long': ma_long,
        'atr_ratio': atr_ratio,
        'ten_return': ten_return,
        'trend_ok': trend_ok
    })
    pbar.progress((i+1)/N)

pbar.progress(1.0)
if len(records) == 0:
    st.error("实时/回退筛选后无候选，请放宽条件或确认接口权限")
    st.stop()

score_df = pd.DataFrame(records)

# ---------------------------
# Step 4: scoring (real-time aware)
# ---------------------------
# normalize subfactors (fillna with conservative defaults)
score_df['pct_rank'] = norm_series(score_df['pct_chg'].fillna(0))
score_df['volrank'] = norm_series(score_df['vol_ratio'].fillna(0))
score_df['amt_rank'] = norm_series(score_df['amount_day'].fillna(0))
score_df['money_rank'] = norm_series(score_df['net_mf'].fillna(0))
score_df['trend_rank'] = norm_series((score_df['ma_short'].fillna(0) - score_df['ma_long'].fillna(0)).fillna(0))
score_df['atr_penalty'] = score_df['atr_ratio'].fillna(0).apply(lambda x: 1.0 if 0.0025 <= x <= 0.12 else 0.85)

# weights tuned for intraday shortline
w_pct = 0.20
w_vol = 0.18
w_amt = 0.15
w_money = 0.10
w_trend = 0.22
w_extra = 0.15  # safety / atr / trend_ok multiplier

# compute raw score
score_df['score_raw'] = (
    score_df['pct_rank'] * w_pct +
    score_df['volrank'] * w_vol +
    score_df['amt_rank'] * w_amt +
    score_df['money_rank'] * w_money +
    score_df['trend_rank'] * w_trend
)

# apply multipliers
score_df['trend_mult'] = score_df['trend_ok'].apply(lambda x: 1.0 if x else 0.8)
score_df['score'] = score_df['score_raw'] * score_df['trend_mult'] * score_df['atr_penalty']

# industry boost: compute industry mean pct in current pool and boost top industries slightly
ind_mean = score_df.groupby('industry')['pct_chg'].transform('mean').fillna(0)
score_df['industry_rank'] = norm_series(ind_mean)
score_df['score'] = score_df['score'] * (1 + 0.15 * score_df['industry_rank'])

# final sort & display
score_df = score_df.sort_values('score', ascending=False).reset_index(drop=True)
score_df.index += 1

st.success(f"实时评分完成（模式：{'实时' if realtime_available else '日线回退'}），候选 {len(score_df)} 支，展示 Top {min(TOP_K, len(score_df))}")
display_cols = ['name','ts_code','score','pct_chg','vol_ratio','amount_day','price','ma_short','ma_long','atr_ratio','industry','trend_ok']
for c in display_cols:
    if c not in score_df.columns:
        score_df[c] = np.nan

st.dataframe(score_df[display_cols].head(int(TOP_K)).reset_index(drop=False), use_container_width=True)

# Reasons / quick notes for top picks
st.markdown("### Top 候选说明（示例）")
for idx, r in score_df.head(int(TOP_K)).reset_index().iterrows():
    reasons = []
    if r['pct_chg'] and r['pct_chg'] > 5: reasons.append("当日强势")
    if not math.isnan(r['vol_ratio']) and r['vol_ratio'] > 1.5: reasons.append("当前分钟放量")
    if r['ma_short'] and r['price'] > r['ma_short']: reasons.append("站上短均（分钟）")
    if r['atr_ratio'] and 0.0025 <= r['atr_ratio'] <= 0.06: reasons.append("波动适中")
    st.write(f"{int(r['index'])}. {r['name']} ({r['ts_code']}) — Score {r['score']:.4f} — {'；'.join(reasons) if reasons else '常规'}")

# CSV download
csv = score_df.to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载全部评分结果 CSV", data=csv, file_name=f"realtime_score_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

# final note
st.markdown("### 说明与提示")
st.markdown("""
- 本版优先使用实时接口（rt_min / realtime_quote）。若你看到页面顶部警告，说明实时接口不可用并已回退为日线近似。  
- 实时接口权限受 Tushare 积分/分钟频次限制影响。若频次受限，建议把 INITIAL_TOP_N 减小到 300 或 200，以降低实时请求量。  
- 推荐场景：**9:40 - 10:10** 之间运行可获得最佳短线候选（既避开开盘噪声，又抓到真实早盘资金方向）。  
- 若你希望我把结果自动每 X 秒刷新并高亮新入榜票（自动监控模式），告诉我我会给你一个可配置的自动刷新版本（注意频次受限）。  
""")
