# -*- coding: utf-8 -*-
"""
短线王 · V5.0 双模型自动切换（Ultimate 满血版，适配 10000 积分）
说明：修复了 Streamlit 缓存相关 UnhashableParamError（不再把 pro 作为缓存参数）。
运行：streamlit run app.py
"""
import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import time
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# APP CONFIG
# ---------------------------
st.set_page_config(page_title="短线王 · V5.0 双模型", layout="wide")
st.title("短线王 · V5.0 双模型自动切换（Ultimate · 10000积分）")

# ---------------------------
# User inputs
# ---------------------------
TS_TOKEN = st.text_input("请输入 Tushare Token（本次会话使用）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 后运行。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

st.sidebar.header("运行与频率设置（建议每天 4-5 次）")
auto_refresh = st.sidebar.checkbox("启用自动前端刷新（仅页面自动刷新，不替代外部调度）", value=False)
refresh_seconds = int(st.sidebar.number_input("自动刷新间隔（秒）", min_value=30, max_value=3600, value=600, step=30))
run_now = st.sidebar.button("立即运行/刷新一次")

st.sidebar.markdown("---")
st.sidebar.header("样本与筛选（控制请求量）")
INITIAL_TOP_N = int(st.sidebar.number_input("初筛：涨幅榜取前 N", min_value=200, max_value=5000, value=1200, step=100))
FINAL_POOL = int(st.sidebar.number_input("进入评分池数量（历史请求限制）", min_value=100, max_value=2000, value=800, step=50))
TOP_K = int(st.sidebar.number_input("展示 Top K", min_value=5, max_value=50, value=20, step=1))

st.sidebar.markdown("---")
st.sidebar.header("资金/市值/价格过滤（建议保守）")
MIN_PRICE = float(st.sidebar.number_input("最低股价（元）", min_value=0.1, max_value=1000.0, value=5.0, step=0.1))
MAX_PRICE = float(st.sidebar.number_input("最高股价（元）", min_value=1.0, max_value=5000.0, value=500.0, step=1.0))
MIN_AMOUNT = float(st.sidebar.number_input("最低日成交额（元）", min_value=0.0, max_value=1e12, value=100_000_000.0, step=10_000_000.0))
MIN_CIRC_MV = float(st.sidebar.number_input("最小流通市值（元）", min_value=1e7, max_value=1e12, value=2_0000_0000.0, step=1e7))
MAX_CIRC_MV = float(st.sidebar.number_input("最大流通市值（元）", min_value=1e8, max_value=2e13, value=50_0000_00000.0, step=1e8))

st.sidebar.markdown("---")
st.sidebar.header("模型行为开关")
allow_weak_model = st.sidebar.checkbox("允许弱势反转模型在盘面差时启用（会降低选股强度）", value=True)
exclude_10_20_double = st.sidebar.checkbox("排除近 10-20 天翻倍股票", value=True)
exclude_recent_accel = st.sidebar.checkbox("排除极端短期暴涨（3日>8% 且量放）", value=True)

st.sidebar.markdown("---")
st.sidebar.header("因子权重（可微调）")
weights = {
    'pct': st.sidebar.slider("当日涨幅权重", 0.0, 1.0, 0.12),
    'mf': st.sidebar.slider("主力资金权重", 0.0, 1.0, 0.18),
    'chip': st.sidebar.slider("筹码稳定权重", 0.0, 1.0, 0.14),
    'volprice': st.sidebar.slider("量价齐升权重", 0.0, 1.0, 0.16),
    'ind': st.sidebar.slider("行业动量权重", 0.0, 1.0, 0.14),
    'trend': st.sidebar.slider("趋势（MA）权重", 0.0, 1.0, 0.12),
    'health': st.sidebar.slider("K线健康权重", 0.0, 1.0, 0.14)
}
_wsum = sum(weights.values())
if _wsum == 0:
    st.sidebar.error("权重总和不可为 0")
    st.stop()
for k in weights:
    weights[k] /= _wsum

# ---------------------------
# Helpers & caching
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

# --- IMPORTANT: do NOT pass 'pro' into cached functions (pro is unhashable) ---
@st.cache_data(ttl=600)
def get_last_trade_day(max_days=14):
    """
    Use global pro inside. Returns a string like 'YYYYMMDD' or None.
    """
    today = datetime.now()
    for i in range(0, max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        try:
            dd = pro.daily(trade_date=ds)
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
def get_moneyflow(trade_date):
    try:
        mf = pro.moneyflow(trade_date=trade_date)
        if mf is None or mf.empty:
            return None
        for col in ['net_mf','net_mf_amount','net_amount']:
            if col in mf.columns:
                tmp = mf[['ts_code', col]].drop_duplicates(subset=['ts_code']).set_index('ts_code')
                tmp.columns = ['net_mf']
                return tmp
        return None
    except Exception:
        return None

# remove caching from special api call to avoid hashing kwargs issues
def try_special_api(api_name, **kwargs):
    """
    Try specialty APIs (like chips/distribution/forecast) if available in pro.
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
def get_hist(ts_code, end_date, days=90):
    try:
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        for c in ['open','high','low','close','vol','amount']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception:
        return None

# ---------------------------
# Auto-refresh simple mechanism
# ---------------------------
if auto_refresh:
    last_run = st.session_state.get("_v5_last_run", 0.0)
    if time.time() - last_run > refresh_seconds:
        st.session_state["_v5_last_run"] = time.time()
        st.experimental_rerun()

# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline():
    # 0. last trade day & market snapshot
    last_trade = get_last_trade_day(max_days=14)
    if not last_trade:
        st.error("无法获取最近交易日，请检查 Token 或网络。")
        return
    st.info(f"参考交易日：{last_trade}")

    market_df = load_market_daily(last_trade)
    if market_df is None or market_df.empty:
        st.error("获取当日日线失败（权限或网络问题）")
        return
    st.write(f"当日市场记录数：{len(market_df)}（将从涨幅榜前 {INITIAL_TOP_N} 进行初筛）")

    # 0.5 determine market state (for switching)
    market_index_pct = None
    try:
        idx = pro.index_daily(ts_code='000001.SH', trade_date=last_trade)
        if idx is not None and len(idx) > 0:
            market_index_pct = float(idx.iloc[0].get('pct_chg', 0.0))
    except Exception:
        market_index_pct = None

    try:
        up_count = (market_df['pct_chg'] > 0).sum()
        down_count = (market_df['pct_chg'] < 0).sum()
        breadth = (up_count - down_count) / (up_count + down_count + 1e-9)
    except Exception:
        breadth = 0.0

    trend_mode = True
    if market_index_pct is not None:
        if market_index_pct < -0.3 or breadth < -0.15:
            trend_mode = False
    else:
        if breadth < -0.15:
            trend_mode = False

    st.markdown(f"**当前模型模式：** {'右侧趋势模式（Trend Mode）' if trend_mode else '弱势反转模式（Weak/Reversal Mode）'}")
    if not trend_mode and not allow_weak_model:
        st.warning("检测到弱势盘面且你关闭了弱势模型，系统将保持空仓并不返回候选。")
        return

    # 1. initial pool (top N by pct_chg)
    pool = market_df.sort_values('pct_chg', ascending=False).head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

    stock_basic = get_stock_basic()
    if not stock_basic.empty:
        cols = [c for c in ['ts_code','name','industry','circ_mv','total_mv'] if c in stock_basic.columns]
        try:
            pool = pool.merge(stock_basic[cols], on='ts_code', how='left')
        except Exception:
            pool['name'] = pool['ts_code']; pool['industry'] = ""
    else:
        pool['name'] = pool['ts_code']; pool['industry'] = ""

    daily_basic = get_daily_basic(last_trade)
    if daily_basic is not None:
        try:
            db = daily_basic.drop_duplicates(subset=['ts_code']).set_index('ts_code')
            join_cols = []
            if 'turnover_rate' in db.columns: join_cols.append('turnover_rate')
            if 'amount' in db.columns: join_cols.append('amount')
            if 'circ_mv' in db.columns: join_cols.append('circ_mv')
            if join_cols:
                pool = pool.set_index('ts_code').join(db[join_cols].rename(columns={'turnover_rate':'turnover_rate_db','amount':'amount_db'}), how='left').reset_index()
            else:
                pool['turnover_rate_db'] = np.nan; pool['amount_db'] = np.nan
        except Exception:
            pool['turnover_rate_db'] = np.nan; pool['amount_db'] = np.nan
    else:
        pool['turnover_rate_db'] = np.nan; pool['amount_db'] = np.nan

    moneyflow_df = get_moneyflow(last_trade)
    if moneyflow_df is None:
        st.warning("moneyflow 主力资金接口不可用或返回空（将降级）。")

    # limit to FINAL_POOL to reduce heavy history calls
    pool = pool.sort_values('pct_chg', ascending=False).head(int(FINAL_POOL)).reset_index(drop=True)
    st.write(f"初筛后候选：{len(pool)}（接下来将为每只请求历史与特色数据并计算因子）")

    # try special APIs once (best-effort)
    special_chips = try_special_api('daily_chip', trade_date=last_trade) or try_special_api('chips_distribution', trade_date=last_trade) or None
    special_forecast = try_special_api('fina_indicator', start_date=(datetime.strptime(last_trade,"%Y%m%d")-timedelta(days=365)).strftime("%Y%m%d"), end_date=last_trade) or None

    # iterate and compute features
    records = []
    pbar = st.progress(0)
    N = len(pool)
    for i, r in enumerate(pool.itertuples()):
        ts = getattr(r, 'ts_code')
        name = getattr(r, 'name', ts)
        industry = getattr(r, 'industry', '') if 'industry' in pool.columns else ''
        pct_chg = safe_float(getattr(r, 'pct_chg', 0))
        amount_day = safe_float(getattr(r, 'amount_db', getattr(r, 'amount', np.nan)))
        # normalize amount from 万元 if necessary
        if amount_day > 0 and amount_day < 1e5:
            amount_day *= 10000
        price = safe_float(getattr(r, 'close', np.nan))
        turnover_rt = safe_float(getattr(r, 'turnover_rate_db', np.nan))
        circ_mv = safe_float(getattr(r, 'circ_mv', np.nan))

        # basic filters
        if math.isnan(price) or price < MIN_PRICE or price > MAX_PRICE:
            pbar.progress((i+1)/N); continue
        if amount_day > 0 and amount_day < MIN_AMOUNT:
            pbar.progress((i+1)/N); continue
        if not math.isnan(circ_mv) and (circ_mv < MIN_CIRC_MV or circ_mv > MAX_CIRC_MV):
            pbar.progress((i+1)/N); continue

        hist = get_hist(ts, last_trade, days=90)
        if hist is None or hist.empty:
            pbar.progress((i+1)/N); continue

        closes = hist['close'].astype(float)
        opens = hist['open'].astype(float)
        highs = hist['high'].astype(float)
        lows = hist['low'].astype(float)
        vols = hist['vol'].astype(float)
        amts = hist['amount'].astype(float) if 'amount' in hist.columns else np.zeros(len(closes))

        # moving averages & volume stats
        def ma(series, n):
            return float(series.rolling(window=n).mean().iloc[-1]) if len(series) >= n else np.nan
        ma5 = ma(closes,5); ma10 = ma(closes,10); ma20 = ma(closes,20)
        amt5 = ma(pd.Series(amts),5); amt20 = ma(pd.Series(amts),20)
        vol5 = ma(pd.Series(vols),5); vol20 = ma(pd.Series(vols),20)

        ten_ret = (closes.iloc[-1]/closes.iloc[-10]-1) if len(closes) >= 10 else np.nan
        twenty_ret = (closes.iloc[-1]/closes.iloc[-20]-1) if len(closes) >= 20 else np.nan

        # exclude 10-20 day doubling
        if exclude_10_20_double and ( (not math.isnan(ten_ret) and ten_ret >= 1.0) or (not math.isnan(twenty_ret) and twenty_ret >= 1.0) ):
            pbar.progress((i+1)/N); continue

        # exclude recent accel
        if exclude_recent_accel and len(closes) >= 6:
            pct_recent = closes.pct_change().fillna(0)
            if len(pct_recent) >= 3:
                last3 = pct_recent.iloc[-3:]
                if all(last3 > 0.08) and (vols.iloc[-1] / (vols.iloc[-6:-1].mean() + 1e-9) > 3):
                    pbar.progress((i+1)/N); continue

        # ATR
        prev = closes.shift(1)
        tr = pd.concat([highs - lows, (highs - prev).abs(), (lows - prev).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1] if len(tr) > 0 else np.nan
        atr_ratio = float(atr) / (closes.iloc[-1] + 1e-9) if not math.isnan(atr) else np.nan

        # K-line health
        recent = hist.tail(7).reset_index(drop=True)
        k_health = 1.0
        long_upper = False; consec_down = 0
        for idx_r in range(len(recent)):
            o = safe_float(recent.loc[idx_r,'open'], np.nan)
            c = safe_float(recent.loc[idx_r,'close'], np.nan)
            h = safe_float(recent.loc[idx_r,'high'], np.nan)
            l = safe_float(recent.loc[idx_r,'low'], np.nan)
            if math.isnan(o) or math.isnan(c): continue
            body = abs(c-o)
            upper = h - max(o,c)
            if body > 0 and upper/(body+1e-9) > 4:
                long_upper = True
            if c < o:
                consec_down += 1
        if long_upper: k_health *= 0.65
        if consec_down >= 4: k_health *= 0.5

        # volume-price alignment
        price_align = 1.0 if (not math.isnan(ma5) and not math.isnan(ma20) and ma5 > ma20) else 0.0
        amt_align = 1.0 if (not math.isnan(amt5) and not math.isnan(amt20) and amt5 > amt20) else 0.0
        volprice = price_align*0.6 + amt_align*0.4

        # trend strength
        trend_strength = 0.0
        if not math.isnan(ma5) and not math.isnan(ma20):
            trend_strength = (ma5 - ma20) / (ma20 + 1e-9)

        # chip_score: try special chips else use vol stability
        chip_score = 0.5
        try:
            if special_chips is not None and isinstance(special_chips, pd.DataFrame):
                if 'ts_code' in special_chips.columns and ts in special_chips['ts_code'].values:
                    for col in ['concentration','chip_concentration','chipratio','peak_density']:
                        if col in special_chips.columns:
                            val = float(special_chips[special_chips['ts_code']==ts][col].iloc[0])
                            chip_score = max(0.0, min(1.0, 1 - val/10.0))
                            break
                    else:
                        chip_score = 0.7
            else:
                vv = vols.tail(20).astype(float)
                if len(vv) >= 8:
                    var = vv.std(); mean = vv.mean() + 1e-9
                    chip_score = 1.0 - min(1.0, var/mean) * 0.5
        except Exception:
            chip_score = 0.5
        chip_score = max(0.0, min(1.0, chip_score))

        # moneyflow
        net_mf = 0.0
        try:
            if moneyflow_df is not None and ts in moneyflow_df.index:
                net_mf = float(moneyflow_df.loc[ts,'net_mf'])
        except Exception:
            net_mf = 0.0

        # forecast score (if available)
        forecast_score = 0.0
        try:
            if special_forecast is not None and isinstance(special_forecast, pd.DataFrame):
                if 'ts_code' in special_forecast.columns and ts in special_forecast['ts_code'].values:
                    for col in ['profit_forecast','inc_netprofit','netprofit_ratio']:
                        if col in special_forecast.columns:
                            forecast_score = safe_float(special_forecast[special_forecast['ts_code']==ts][col].iloc[0], 0.0)
                            break
        except Exception:
            forecast_score = 0.0

        records.append({
            'ts_code': ts, 'name': name, 'industry': industry, 'price': float(closes.iloc[-1]),
            'pct_chg': pct_chg, 'amount_day': amount_day, 'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
            'amt5': amt5, 'amt20': amt20, 'vol5': vol5, 'vol20': vol20,
            'ten_ret': ten_ret, 'twenty_ret': twenty_ret,
            'atr_ratio': atr_ratio, 'k_health': k_health,
            'volprice': volprice, 'trend_strength': trend_strength,
            'chip_score': chip_score, 'net_mf': net_mf, 'forecast_score': forecast_score,
            'turnover_rate': turnover_rt, 'circ_mv': circ_mv
        })
        pbar.progress((i+1)/N)

    pbar.progress(1.0)
    if len(records) == 0:
        st.warning("最终候选为 0（历史数据不足或过滤过严）。可放宽过滤条件后重试。")
        return

    df = pd.DataFrame(records)

    # industry-level boost
    df['ind_mom'] = df.groupby('industry')['trend_strength'].transform('mean').fillna(0)
    df['ind_mom_rank'] = norm_series(df['ind_mom'])

    # normalize factors
    df['pct_rank'] = norm_series(df['pct_chg'].fillna(0))
    df['mf_rank'] = norm_series(df['net_mf'].fillna(0))
    df['chip_rank'] = norm_series(df['chip_score'].fillna(0))
    df['volprice_rank'] = norm_series(df['volprice'].fillna(0))
    df['trend_rank'] = norm_series(df['trend_strength'].fillna(0))
    df['health_rank'] = norm_series(df['k_health'].fillna(0))
    df['forecast_rank'] = norm_series(df['forecast_score'].fillna(0))

    # choose scoring template depending on mode
    if trend_mode:
        df['score_raw'] = (
            df['pct_rank'] * weights['pct'] +
            df['mf_rank'] * weights['mf'] +
            df['chip_rank'] * weights['chip'] +
            df['volprice_rank'] * weights['volprice'] +
            df['ind_mom_rank'] * weights['ind'] +
            df['trend_rank'] * weights['trend'] +
            df['health_rank'] * weights['health']
        )
    else:
        df['score_raw'] = (
            df['chip_rank'] * (weights['chip'] + 0.12) +
            df['volprice_rank'] * (weights['volprice'] + 0.06) +
            df['mf_rank'] * (weights['mf'] * 0.7) +
            df['pct_rank'] * (weights['pct'] * 0.7) +
            df['ind_mom_rank'] * (weights['ind'] * 0.5) +
            df['health_rank'] * weights['health']
        )

    # apply bonuses & penalties
    df['score_chip_boost'] = df['score_raw'] * (1 + 0.12 * df['chip_rank'])
    df['score_ind_boost'] = df['score_chip_boost'] * (1 + 0.12 * df['ind_mom_rank'])
    df['atr_penalty'] = df['atr_ratio'].apply(lambda x: 1.0 if (not math.isnan(x) and 0.002 <= x <= 0.18) else 0.9)
    df['score_final'] = df['score_ind_boost'] * df['atr_penalty']

    # safety filters
    df = df[ df['amount_day'].fillna(0) >= MIN_AMOUNT ]
    df = df[ df['k_health'] > 0.35 ]
    df = df.sort_values('score_final', ascending=False).reset_index(drop=True)
    df.index += 1

    st.success(f"评分完成，候选 {len(df)} 支，展示 Top {min(TOP_K, len(df))}")

    display_cols = ['name','ts_code','score_final','pct_chg','price','amount_day','ma5','ma10','ma20','volprice','chip_score','k_health','atr_ratio','ind_mom','ten_ret']
    for c in display_cols:
        if c not in df.columns:
            df[c] = np.nan

    st.dataframe(df[display_cols].head(TOP_K).reset_index(drop=False), use_container_width=True)

    # top reasons
    st.markdown("### Top 候选简要原因（自动生成）")
    for idx, row in df.head(TOP_K).iterrows():
        reasons = []
        if row['pct_chg'] > 5: reasons.append("当日强势")
        if row['volprice'] > 0.6: reasons.append("量价齐升")
        if row['chip_score'] > 0.6: reasons.append("筹码稳定")
        if row['k_health'] > 0.8: reasons.append("K线健康")
        if 0.002 <= safe_float(row['atr_ratio'], np.nan) <= 0.08: reasons.append("波动适中")
        st.write(f"{int(idx+0)}. {row['name']} ({row['ts_code']}) — Score {row['score_final']:.4f} — {'；'.join(reasons) if reasons else '常规'}")

    # CSV download
    csv = df.to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载全部评分 CSV", data=csv, file_name=f"v5_score_{last_trade}_{datetime.now().strftime('%H%M%S')}.csv", mime="text/csv")

    # run summary
    st.markdown("### 运行日志与说明")
    st.markdown(f"""
- 参考交易日：{last_trade}  
- 当前模式：{'右侧趋势模式（Trend）' if trend_mode else '弱势反转模式（Weak/Reversal）'}  
- moneyflow 主力资金：{'可用' if moneyflow_df is not None else '不可用/降级'}  
- specialty chips API：{'可用' if special_chips is not None else '不可用/降级（用日线推估筹码）'}  
- specialty forecast API：{'可用' if special_forecast is not None else '不可用/降级'}  

运行建议：  
- 建议每天运行 4-5 次（例如 9:50、10:40、13:40、14:30、15:00）以捕捉盘中分时延续与午后发力。  
- 若你希望完全自动化（无人值守），请部署在云端并用 cron/GitHub Actions 调用（Streamlit 本身不保证后台定时运行）。  
- 若需要我把“买入价 / 止损 / 目标”自动生成规则加入（基于 ATR 与风险承受），回复我“加交易建议”。  
""")

# ---------------------------
# Execute on initial load or on demand
# ---------------------------
if run_now:
    st.session_state["_v5_last_run"] = time.time()
    run_pipeline()
else:
    if "_v5_has_run" not in st.session_state:
        st.session_state["_v5_has_run"] = True
        st.session_state["_v5_last_run"] = time.time()
        run_pipeline()
    else:
        st.info("点击左侧“立即运行/刷新一次”可强制刷新结果。")
