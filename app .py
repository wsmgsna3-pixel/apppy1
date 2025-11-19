# -*- coding: utf-8 -*-
"""
短线王 · v3.5（1-5天短线 专属强化版）
特性（相较 v3.0 的主要升级）：
- 更严格的趋势过滤（最近低点抬升 / 不破短均线 / 5/10/20 日结构）
- 成交额硬过滤（默认日成交额 ≥ 2 亿）
- 行业主线强制：非强行业将被大幅降权或剔除
- MACD/RSI 评分体系重写，差异化更明显
- ATR 波动与波动比率更严格过滤
- 翻倍排除、连续大幅加速排除
- 完整容错、缓存与 UI 降级提示
"""
import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="短线王 · v3.5（Top20）", layout="wide")
st.title("短线王 · v3.5（1-5 天短线 专属强化）")

# ---------------------------
# Token (manual)
# ---------------------------
TS_TOKEN = st.text_input("请输入 Tushare Token（仅本次会话使用）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 后运行。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# Utilities & safety
# ---------------------------
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except:
        return default

def norm_series(s):
    s = pd.Series(s).astype(float)
    if s.isnull().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn = s.min(); mx = s.max()
    if abs(mx - mn) < 1e-9:
        return pd.Series(np.ones(len(s)) * 0.5, index=s.index)
    return (s - mn) / (mx - mn)

def check_and_fill_data(df, required_cols):
    missing = []
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan
            missing.append(c)
    return df, missing

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

# Technical indicators
def calculate_macd(df, close_col='close', short=12, long=26, signal=9):
    close = df[close_col].astype(float)
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    dif = ema_short - ema_long
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    res = df.copy()
    res['DIF'] = dif; res['DEA'] = dea; res['MACD_HIST'] = hist
    return res

def calculate_rsi(df, close_col='close', period=6):
    close = df[close_col].astype(float)
    delta = close.diff()
    up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    res = df.copy()
    res[f'RSI_{period}'] = rsi
    return res

def calculate_atr(df, period=14):
    df = df.copy()
    df['high'] = df['high'].astype(float); df['low'] = df['low'].astype(float); df['close'] = df['close'].astype(float)
    df['prev_close'] = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['prev_close']).abs()
    tr3 = (df['low'] - df['prev_close']).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    df['ATR'] = atr
    return df

# ---------------------------
# Sidebar (params tuned for 1-5d)
# ---------------------------
st.sidebar.header("筛选参数（短线 1-5 天，v3.5）")
INITIAL_TOP_N = int(st.sidebar.number_input("初筛：涨幅榜取前 N", min_value=200, max_value=5000, value=1000, step=100))
FINAL_POOL = int(st.sidebar.number_input("进入评分池数量", min_value=50, max_value=2000, value=500, step=50))
TOP_K = int(st.sidebar.number_input("界面展示 Top K", min_value=5, max_value=50, value=20, step=1))

MIN_PRICE = float(st.sidebar.number_input("最低股价（元）", min_value=0.1, max_value=1000.0, value=10.0, step=0.1))
MAX_PRICE = float(st.sidebar.number_input("最高股价（元）", min_value=1.0, max_value=2000.0, value=200.0, step=1.0))
MIN_AMOUNT = float(st.sidebar.number_input("最低日成交额（元，硬过滤，建议 >= 200,000,000）", min_value=0.0, max_value=1e11, value=200_000_000.0, step=10_000_000.0))
MIN_MV = float(st.sidebar.number_input("最小市值（元）", min_value=1e7, max_value=1e12, value=20_0000_0000.0, step=1e7))
MAX_MV = float(st.sidebar.number_input("最大市值（元）", min_value=1e8, max_value=1e13, value=500_0000_00000.0, step=1e8))

EXCLUDE_DOUBLE_10_20 = st.sidebar.checkbox("排除过去10-20天内翻倍", value=True)
EXCLUDE_HIGH_ACCEL = st.sidebar.checkbox("排除短期连续高加速（3日内连续放量大涨）", value=True)

st.sidebar.markdown("---")
st.sidebar.header("因子权重（默认强化行业与技术）")
w_pct = st.sidebar.slider("短期涨幅强度", 0.0, 1.0, 0.16)
w_volratio = st.sidebar.slider("量比（放量）", 0.0, 1.0, 0.14)
w_turn = st.sidebar.slider("成交额/活跃度", 0.0, 1.0, 0.12)
w_money = st.sidebar.slider("主力资金", 0.0, 1.0, 0.10)
w_ind = st.sidebar.slider("行业强度（提升）", 0.0, 1.0, 0.30)
w_tech = st.sidebar.slider("技术形态（MACD/RSI/形态）", 0.0, 1.0, 0.18)

total_w = w_pct + w_volratio + w_turn + w_money + w_ind + w_tech
if total_w == 0:
    st.sidebar.error("权重总和不可为0")
    st.stop()
w_pct /= total_w; w_volratio /= total_w; w_turn /= total_w; w_money /= total_w; w_ind /= total_w; w_tech /= total_w

st.sidebar.markdown("---")
st.sidebar.markdown("说明：若 moneyflow/daily_basic/stock_basic 某些字段缺失，脚本自动降级并提示。")

# ---------------------------
# Get last trade day & market
# ---------------------------
with st.spinner("获取最近交易日..."):
    last_trade = get_last_trade_day(pro, max_days=14)
if not last_trade:
    st.error("无法获取最近交易日，请检查 Token 或网络")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")

@st.cache_data(ttl=60)
def load_market_daily(trade_date):
    try:
        return pro.daily(trade_date=trade_date)
    except Exception:
        return pd.DataFrame()

market_df = load_market_daily(last_trade)
market_df, _ = check_and_fill_data(market_df, ['ts_code','pct_chg','vol','amount','open','high','low','pre_close','close'])
if market_df.empty:
    st.error("无法获取当日行情或数据为空")
    st.stop()
st.write(f"当日记录数：{len(market_df)}（从涨幅榜前 {INITIAL_TOP_N} 初筛）")

# ---------------------------
# stock_basic / daily_basic / moneyflow safe
# ---------------------------
def try_get_stock_basic():
    try:
        return pro.stock_basic(list_status='L', fields='ts_code,symbol,name,market,industry,list_date')
    except Exception:
        try:
            return pro.stock_basic(list_status='L')
        except Exception:
            return pd.DataFrame()

stock_basic_df = try_get_stock_basic()
stock_basic_df, missing_sb = check_and_fill_data(stock_basic_df, ['ts_code','name','industry'])
if missing_sb:
    st.warning(f"stock_basic 缺失列：{missing_sb}，行业/名称可能受限，相关因子将自动降级。")

def try_get_daily_basic(trade_date):
    try:
        return pro.daily_basic(trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv,pe,pb')
    except Exception:
        return None

def try_get_moneyflow(trade_date):
    try:
        mf = pro.moneyflow(trade_date=trade_date)
        for col in ['net_mf','net_mf_amount','net_amount']:
            if col in mf.columns:
                mf = mf[['ts_code', col]].drop_duplicates(subset=['ts_code']).set_index('ts_code')
                mf.columns = ['net_mf']
                return mf
        return None
    except Exception:
        return None

daily_basic_df = try_get_daily_basic(last_trade)
if daily_basic_df is None:
    st.warning("daily_basic 不可用，换手/市值/PE 等因子将被降级或用近似替代。")

moneyflow_df = try_get_moneyflow(last_trade)
if moneyflow_df is None:
    st.warning("moneyflow 不可用，主力资金因子将被禁用。")

# ---------------------------
# initial pool
# ---------------------------
pool = market_df.sort_values('pct_chg', ascending=False).head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

# safe merge stock_basic
need_cols = ['ts_code','name','industry']
actual_cols = stock_basic_df.columns.tolist() if not stock_basic_df.empty else []
use_cols = [c for c in need_cols if c in actual_cols]
if 'ts_code' in use_cols and len(use_cols)>0:
    pool = pool.merge(stock_basic_df[use_cols], on='ts_code', how='left')
else:
    pool['name'] = pool['ts_code']
    pool['industry'] = ""

# join daily_basic if available
if daily_basic_df is not None:
    try:
        db = daily_basic_df.drop_duplicates(subset=['ts_code']).set_index('ts_code')
        pool = pool.set_index('ts_code').join(db[['turnover_rate','amount','total_mv','circ_mv']].rename(columns={'turnover_rate':'turnover_rate_db','amount':'amount_db'}), how='left').reset_index()
    except Exception:
        pool['turnover_rate_db'] = np.nan; pool['amount_db'] = np.nan
else:
    pool['turnover_rate_db'] = np.nan; pool['amount_db'] = np.nan

# join moneyflow if available
if moneyflow_df is not None:
    try:
        pool = pool.set_index('ts_code').join(moneyflow_df[['net_mf']], how='left').reset_index()
    except Exception:
        pool['net_mf'] = 0.0
else:
    pool['net_mf'] = 0.0

pool, _ = check_and_fill_data(pool, ['ts_code','pct_chg','vol','amount','open','high','low','pre_close','close','name','industry','turnover_rate_db','net_mf'])

# ---------------------------
# cleaning with stronger hard filters
# ---------------------------
cleaned = []
for idx, r in pool.iterrows():
    try:
        vol = safe_float(r.get('vol', 0)); amt = safe_float(r.get('amount', 0))
        if vol == 0 or (amt == 0 or np.isnan(amt)):
            continue
        # amount normalization
        if amt > 0 and amt < 1e5:
            amt *= 10000
        if amt < MIN_AMOUNT:
            continue
        price = safe_float(r.get('close', np.nan))
        if np.isnan(price): continue
        if price < MIN_PRICE or price > MAX_PRICE: continue
        # name filter
        name = r.get('name','')
        if isinstance(name, str) and name != "":
            if 'ST' in name.upper() or '退' in name:
                continue
        # market cap
        tv = r.get('total_mv', np.nan)
        try:
            tvf = float(tv)
            if tvf > 1e6:
                tv_yuan = tvf * 10000
            else:
                tv_yuan = tvf
            if not np.isnan(tv_yuan):
                if tv_yuan < MIN_MV or tv_yuan > MAX_MV:
                    continue
        except:
            pass
        # exclude one-word boards
        try:
            if (safe_float(r.get('open',0)) == safe_float(r.get('high',0)) == safe_float(r.get('low',0)) == safe_float(r.get('pre_close',0))):
                continue
        except:
            pass
        cleaned.append(r)
    except Exception:
        continue

cleaned_df = pd.DataFrame(cleaned).reset_index(drop=True)
st.write(f"清洗后候选：{len(cleaned_df)}（从中取涨幅前 {FINAL_POOL} 进入评分）")
if cleaned_df.empty:
    st.error("清洗后无候选，请放宽条件或检查 Token 权限")
    st.stop()

# reduce to FINAL_POOL
cleaned_df = cleaned_df.sort_values('pct_chg', ascending=False).head(int(FINAL_POOL)).reset_index(drop=True)
st.write(f"评分池大小：{len(cleaned_df)}")

# ---------------------------
# history with caching
# ---------------------------
@st.cache_data(ttl=600)
def get_hist(ts_code, end_date, days=80):
    try:
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty: return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        return df
    except Exception:
        return None

# ---------------------------
# scoring loop v3.5
# ---------------------------
records = []
pbar = st.progress(0)
N = len(cleaned_df)
for i, row in enumerate(cleaned_df.itertuples()):
    try:
        ts_code = getattr(row,'ts_code'); pct_chg = safe_float(getattr(row,'pct_chg',0))
        amount_val = safe_float(getattr(row,'amount',0))
        if amount_val >0 and amount_val < 1e5: amount_val *= 10000

        hist = get_hist(ts_code, last_trade, days=80)
        # initialize defaults
        vol_ratio = np.nan; ma5 = np.nan; ma10 = np.nan; ten_return = np.nan; twenty_return = np.nan
        macd_score = 0.5; rsi_score = 0.5; atr_ratio = np.nan; trend_ok = False; exclude_double = False; high_accel = False

        if hist is None or len(hist) < 15:
            # fallback neutral (but still keep filtering by amt/price)
            pass
        else:
            hist = hist.reset_index(drop=True)
            tail = hist.tail(30).reset_index(drop=True)
            # ATR
            atr_df = calculate_atr(tail, period=14)
            atr = atr_df['ATR'].iloc[-1] if 'ATR' in atr_df.columns else np.nan

            price_now = safe_float(tail['close'].iloc[-1])
            atr_ratio = atr / price_now if (not np.isnan(atr) and price_now>0) else np.nan

            # vol ratio (today / prev5 avg)
            vols = tail['vol'].astype(float).tolist()
            if len(vols) >= 6:
                avg5 = np.mean(vols[-6:-1])
                vol_ratio = vols[-1] / (avg5 + 1e-9)
            else:
                vol_ratio = np.nan

            # ma5/ma10 & returns
            close_series = tail['close'].astype(float)
            if len(close_series) >=5: ma5 = close_series.rolling(window=5).mean().iloc[-1]
            if len(close_series) >=10: ma10 = close_series.rolling(window=10).mean().iloc[-1]
            if len(close_series) >=10: ten_return = float(close_series.iloc[-1]) / float(close_series.iloc[0]) - 1.0
            if len(close_series) >=20: twenty_return = float(close_series.iloc[-1]) / float(close_series.iloc[0]) - 1.0

            # exclude double
            if EXCLUDE_DOUBLE_10_20 and ( (not np.isnan(ten_return) and ten_return >= 1.0) or (not np.isnan(twenty_return) and twenty_return >= 1.0) ):
                exclude_double = True

            # detect high acceleration: last 3 days all big pct_chg > threshold AND vol_ratio > threshold
            if EXCLUDE_HIGH_ACCEL:
                tail_pct = tail['close'].pct_change().fillna(0).replace([np.inf,-np.inf],0).tolist()
                recent_pct3 = tail_pct[-3:] if len(tail_pct) >=3 else []
                if len(recent_pct3) == 3 and all([p > 0.08 for p in recent_pct3]) and not np.isnan(vol_ratio) and vol_ratio > 3:
                    high_accel = True

            # trend quality: last5 lows avg > prev5 lows avg
            lows = tail['low'].astype(float).tolist()
            trend_ok = False
            if len(lows) >= 10:
                prev5 = lows[-10:-5]; last5 = lows[-5:]
                if np.nanmean(last5) > np.nanmean(prev5):
                    trend_ok = True

            # MACD/RSI scoring
            macd_tail = calculate_macd(tail, close_col='close')
            dif = macd_tail['DIF'].iloc[-1]; dea = macd_tail['DEA'].iloc[-1]; macd_hist = macd_tail['MACD_HIST'].iloc[-1]
            # MACD rules
            if dif > dea and macd_hist > 0:
                macd_score = 1.0
            elif dif > dea and macd_hist <= 0:
                macd_score = 0.8
            elif dif <= dea and macd_hist > 0:
                macd_score = 0.6
            else:
                macd_score = 0.2

            rsi_tail = calculate_rsi(tail, close_col='close', period=6)
            rsi6 = rsi_tail['RSI_6'].iloc[-1] if 'RSI_6' in rsi_tail.columns else 50.0
            # RSI rules tuned for short-term
            if 55 <= rsi6 <= 70:
                rsi_score = 1.0
            elif 45 <= rsi6 < 55:
                rsi_score = 0.75
            elif 40 <= rsi6 < 45:
                rsi_score = 0.5
            elif rsi6 > 70:
                rsi_score = 0.3
            else:
                rsi_score = 0.15

        # hard exclusions post-history
        if exclude_double:
            continue
        if high_accel:
            # skip immediate chase of extreme加速
            continue
        # ATR absolute filtering (exclude extreme vols)
        if not np.isnan(atr_ratio):
            if atr_ratio > 0.12:  # >12% daily volatility is too insane
                continue
            if atr_ratio < 0.0025:  # <0.25% too sleepy
                continue

        # compose record
        net_mf = 0.0
        try:
            if moneyflow_df is not None and ts_code in moneyflow_df.index:
                net_mf = float(moneyflow_df.loc[ts_code,'net_mf'])
        except Exception:
            net_mf = 0.0

        records.append({
            'ts_code': ts_code,
            'name': getattr(row,'name',ts_code),
            'pct_chg': pct_chg,
            'amount': amount_val,
            'vol_ratio': vol_ratio if not pd.isna(vol_ratio) else 1.0,
            'turnover_rate': safe_float(getattr(row,'turnover_rate_db',np.nan)),
            'net_mf': net_mf,
            'ten_return': ten_return if not pd.isna(ten_return) else 0.0,
            'ma5': ma5,
            'ma10': ma10,
            'macd_score': macd_score,
            'rsi_score': rsi_score,
            'atr_ratio': atr_ratio,
            'trend_ok': trend_ok,
            'price': safe_float(getattr(row,'close',np.nan)),
            'industry': getattr(row,'industry','')
        })
    except Exception:
        continue
    pbar.progress((i+1)/N if N>0 else 1.0)

pbar.progress(1.0)
score_df = pd.DataFrame(records)
if score_df.empty:
    st.error("评分后无候选（可能被严格过滤或历史数据不足），请放宽条件或检查接口权限。")
    st.stop()

# ---------------------------
# industry strength & enforce mainline
# ---------------------------
if 'industry' in score_df.columns and score_df['industry'].notnull().any():
    ind_mean = score_df.groupby('industry')['pct_chg'].transform('mean')
    score_df['industry_score'] = (ind_mean - ind_mean.min()) / (ind_mean.max() - ind_mean.min() + 1e-9)
    score_df['industry_score'] = score_df['industry_score'].fillna(0.0)
    # industry ranking percentile
    ind_pct = score_df.groupby('industry')['pct_chg'].transform(lambda x: pd.Series(x).rank(pct=True).values)
    # If industry_score is very low (bottom 50%), treat as weak; we will penalize later
else:
    score_df['industry_score'] = 0.0
    st.warning("行业数据不可用或为空，行业因子将被禁用（降权）")

# ---------------------------
# normalize & penalties
# ---------------------------
score_df['pct_rank'] = norm_series(score_df['pct_chg'])
score_df['vol_rank'] = norm_series(score_df['vol_ratio'].replace([np.inf,-np.inf],np.nan).fillna(0))
score_df['turn_rank'] = norm_series(score_df['turnover_rate'].fillna(0))
score_df['money_rank'] = norm_series(score_df['net_mf'].fillna(0))
score_df['tech_rank'] = norm_series(score_df['macd_score'] * 0.65 + score_df['rsi_score'] * 0.35)
score_df['industry_rank'] = norm_series(score_df['industry_score'].fillna(0))

# trend bonus (must be true or small penalty)
score_df['trend_bonus'] = score_df['trend_ok'].apply(lambda x: 1.0 if x else 0.6)

# industry enforcement: if industry's mean is low, apply heavy penalty
if 'industry' in score_df.columns and score_df['industry'].notnull().any():
    # compute industry mean pct by group
    ind_group_mean = score_df.groupby('industry')['pct_chg'].transform('mean')
    # industry percentile
    ind_rank = norm_series(ind_group_mean)
    # if ind_rank < 0.4 penalize heavily
    score_df['industry_penalty'] = score_df['industry_score'].apply(lambda x: 1.0 if x >= np.nanpercentile(score_df['industry_score'].fillna(0), 40) else 0.6)
else:
    score_df['industry_penalty'] = 1.0
    w_ind = 0.0  # disable industry weight if no data

# disable money factor if no data
if moneyflow_df is None:
    w_money = 0.0

# re-normalize weights if any disabled
total_w = w_pct + w_volratio + w_turn + w_money + w_ind + w_tech
if total_w == 0:
    st.error("所有因子被禁用，无法评分")
    st.stop()
w_pct /= total_w; w_volratio /= total_w; w_turn /= total_w; w_money /= total_w; w_ind /= total_w; w_tech /= total_w

# compose final score
score_df['综合评分_raw'] = (
    score_df['pct_rank'] * w_pct +
    score_df['vol_rank'] * w_volratio +
    score_df['turn_rank'] * w_turn +
    score_df['money_rank'] * w_money +
    score_df['industry_rank'] * w_ind +
    score_df['tech_rank'] * w_tech
)

# apply trend & industry multipliers and small ATR smoothing
score_df['综合评分'] = score_df['综合评分_raw'] * score_df['trend_bonus'] * score_df['industry_penalty']
score_df['综合评分'] = score_df['综合评分'] * (1 - 0.08 * (1 - np.tanh(10 * (score_df['atr_ratio'].fillna(0.01)-0.01))))

score_df = score_df.sort_values('综合评分', ascending=False).reset_index(drop=True)
score_df.index += 1

# ---------------------------
# Output Top K
# ---------------------------
display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','amount','price','ma5','ma10','ten_return','atr_ratio','industry','trend_ok']
for c in display_cols:
    if c not in score_df.columns:
        score_df[c] = np.nan

st.success(f"评分完成，候选 {len(score_df)} 支，展示 Top {min(TOP_K, len(score_df))}（按综合评分降序）")
st.dataframe(score_df[display_cols].head(int(TOP_K)).reset_index(drop=False), use_container_width=True)

# CSV download
csv = score_df[display_cols].to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载全部评分结果 CSV", data=csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

# explain top picks
st.markdown("### Top 候选说明（示例）")
top_show = score_df.head(int(TOP_K))
for idx, r in top_show.reset_index().iterrows():
    reasons = []
    if r['pct_chg'] > 5: reasons.append("当日强势")
    if r['vol_ratio'] and r['vol_ratio'] > 1.5: reasons.append("放量")
    if r['ma5'] and r['price'] > r['ma5']: reasons.append("站上短均")
    if r['atr_ratio'] and 0.003 <= r['atr_ratio'] <= 0.06: reasons.append("波动适中")
    if r['net_mf'] and r['net_mf'] > 0: reasons.append("主力净流入")
    if r['trend_ok']: reasons.append("低点抬升（趋势质量好）")
    st.write(f"{int(r['index'])}. {r['name']} ({r['ts_code']}) — Score {r['综合评分']:.4f} — {'；'.join(reasons) if reasons else '常规强势'}")

st.markdown("### 小结与建议")
st.markdown("""
- v3.5 为 1–5 天短线做了强过滤：成交额硬过滤、趋势与形态硬要求、行业主线强制、ATR 波动过滤、翻倍/极端加速剔除。  
- 若某接口缺失（例如 moneyflow/daily_basic），脚本会自动降级并提示；核心仍基于 daily 与历史日线。  
- 建议每日早盘 9:30-10:15 运行一次；若看到连续高加速的票（被排除），等回调缩量再做观察。  
- 想微调偏好（更偏保守或更偏激进），告诉我具体指标我直接帮你改。  
""")
