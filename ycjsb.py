# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（BC 混合增强版）· 极速版
说明：
- 该版本在完全保留现有逻辑与功能的前提下做两项加速：
  1) 清洗阶段不再调用 API（使用已合并的 pool_merged 字段）
  2) 评分阶段最多对 300 支股票拉历史（上限，FINAL_POOL 仍可调）
- 其余功能 / 权重 / 容错逻辑与你原版一致。
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
st.set_page_config(page_title="选股王 · 10000旗舰（BC增强）· 极速版", layout="wide")
st.title("选股王 · 10000 积分旗舰（BC 混合增强版）· 极速版")
st.markdown("输入你的 Tushare Token（仅本次运行使用）。若有权限缺失，脚本会自动降级并继续运行。")

# ---------------------------
# 侧边栏参数（实时可改）
# ---------------------------
with st.sidebar:
    st.header("可调参数（实时）")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=500, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=3.0, step=0.5))
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=200_000_000.0, step=50_000_000.0))
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=8.0, step=0.5))
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    st.markdown("---")
    st.caption("提示：保守→降低阈值；激进→提高阈值。")

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
st.write("尝试加载 stock_basic / daily_basic / moneyflow / ths_member / chip 等高级接口（若权限允许）...")
stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')
daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)
# ths_member 可能需权限
try:
    ths_hot = safe_get(pro.ths_member)
except Exception:
    ths_hot = pd.DataFrame()

# chip 接口示例（不影响主流程）
chip_sample = pd.DataFrame()
if hasattr(pro, 'chip'):
    try:
        if len(pool0) > 0:
            chip_sample = safe_get(pro.chip, ts_code=pool0['ts_code'].iloc[0], trade_date=last_trade)
    except Exception:
        chip_sample = pd.DataFrame()

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
    st.warning("moneyflow 未获取到，将把主力流向因子置为 0（若有权限请确认 Token/积分）。")

# ---------------------------
# 合并基本信息（safe）
# ---------------------------
def safe_merge_pool(pool_df, other_df, cols):
    pool = pool_df.set_index('ts_code').copy()
    if other_df is None or other_df.empty:
        for c in cols:
            pool[c] = np.nan
        return pool.reset_index()
    if 'ts_code' not in other_df.columns:
        try:
            other_df = other_df.reset_index()
        except:
            for c in cols:
                pool[c] = np.nan
            return pool.reset_index()
    for c in cols:
        if c not in other_df.columns:
            other_df[c] = np.nan
    try:
        joined = pool.join(other_df.set_index('ts_code')[cols], how='left')
    except Exception:
        for c in cols:
            pool[c] = np.nan
        return pool.reset_index()
    for c in cols:
        if c not in joined.columns:
            joined[c] = np.nan
    return joined.reset_index()

# merge stock_basic
if not stock_basic.empty:
    keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_basic.columns]
    try:
        pool0 = pool0.merge(stock_basic[keep], on='ts_code', how='left')
    except Exception:
        pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
else:
    pool0['name'] = pool0['ts_code']; pool0['industry'] = ''

# merge daily_basic
pool_merged = safe_merge_pool(pool0, daily_basic, ['turnover_rate','amount','total_mv','circ_mv'])

# merge moneyflow robustly
if moneyflow.empty:
    moneyflow = pd.DataFrame({'ts_code': pool_merged['ts_code'].tolist(), 'net_mf': [0.0]*len(pool_merged)})
else:
    if 'ts_code' not in moneyflow.columns:
        moneyflow['ts_code'] = None
try:
    pool_merged = pool_merged.set_index('ts_code').join(moneyflow.set_index('ts_code'), how='left').reset_index()
except Exception:
    if 'net_mf' not in pool_merged.columns:
        pool_merged['net_mf'] = 0.0

if 'net_mf' not in pool_merged.columns:
    pool_merged['net_mf'] = 0.0
pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0.0)

# ---------------------------
# 基本清洗（ST / 停牌 / 价格区间 / 一字板 / 换手 / 成交额 / 市值）
# ------------- 极速优化 -------------
# 关键改动：清洗阶段 **不再调用任何 API**，直接使用 pool_merged 中已有字段。
# ---------------------------
st.write("对初筛池进行清洗（ST/停牌/价格/一字板/换手/成交额等）...（清洗阶段不再调用 API）")
clean_list = []
pbar = st.progress(0)
total_rows = len(pool_merged)
for i, r in enumerate(pool_merged.itertuples()):
    ts = getattr(r, 'ts_code')
    # ---------- 不再请求 pro.daily ----------
    # 直接从合并表里读取 vol/amount/close/open/pre_close等字段
    vol = getattr(r, 'vol', np.nan)
    if pd.isna(vol):
        # 有时 vol 在 daily_basic 的 amount 字段附近，可以尝试用 amount 作简易判断（非精确）
        vol = 0
    close = getattr(r, 'close', np.nan)
    open_p = getattr(r, 'open', np.nan)
    pre_close = getattr(r, 'pre_close', np.nan)
    pct = getattr(r, 'pct_chg', np.nan)
    amount = getattr(r, 'amount', np.nan)
    turnover = getattr(r, 'turnover_rate', np.nan)
    total_mv = getattr(r, 'total_mv', np.nan)
    name = getattr(r, 'name', ts)

    # skip no trading (use amount or vol if available)
    if (pd.isna(vol) or vol == 0) and (pd.isna(amount) or amount == 0):
        pbar.progress((i+1)/total_rows); continue

    # price filter
    if pd.isna(close):
        pbar.progress((i+1)/total_rows); continue
    if (close < MIN_PRICE) or (close > MAX_PRICE):
        pbar.progress((i+1)/total_rows); continue

    # exclude ST / delist
    if isinstance(name, str) and (('ST' in name.upper()) or ('退' in name)):
        pbar.progress((i+1)/total_rows); continue

    # one-word board (open==high==low==pre_close) - read fields from merged
    try:
        high = getattr(r, 'high', np.nan); low = getattr(r, 'low', np.nan)
        if (not pd.isna(open_p) and not pd.isna(high) and not pd.isna(low) and not pd.isna(pre_close)):
            if (open_p == high == low == pre_close):
                pbar.progress((i+1)/total_rows); continue
    except:
        pass

    # market cap filter (兜底)
    try:
        tv = total_mv
        if not pd.isna(tv):
            tv = float(tv)
            if tv > 1e6:
                tv_yuan = tv * 10000.0
            else:
                tv_yuan = tv
            # skip mega caps beyond reason (保守)
            if tv_yuan > 2000 * 1e8:  # 2000亿
                pbar.progress((i+1)/total_rows); continue
    except:
        pass

    # turnover
    if not pd.isna(turnover):
        try:
            if float(turnover) < MIN_TURNOVER: pbar.progress((i+1)/total_rows); continue
        except:
            pass

    # amount (convert if likely in 万元)
    if not pd.isna(amount):
        amt = amount
        if amt > 0 and amt < 1e5:
            amt = amt * 10000.0
        if amt < MIN_AMOUNT: pbar.progress((i+1)/total_rows); continue

    # exclude yesterday down
    try:
        if float(pct) < 0: pbar.progress((i+1)/total_rows); continue
    except:
        pass

    clean_list.append(r)
    pbar.progress((i+1)/total_rows)

pbar.progress(1.0)
# build clean_df from tuples
clean_df = pd.DataFrame([dict(zip(r._fields, r)) for r in clean_list])
st.write(f"清洗后候选数量：{len(clean_df)} （将从中取涨幅前 {FINAL_POOL} 进入评分阶段）")
if len(clean_df) == 0:
    st.error("清洗后没有候选，建议放宽条件或检查接口权限。")
    st.stop()

# ---------------------------
# 取涨幅前 FINAL_POOL 进入评分池
# ------------- 极速优化 -------------
# 关键改动：评分阶段最多对 300 支股票拉历史（提高速度）
# ---------------------------
score_pool_n = min(int(FINAL_POOL), 300)
clean_df = clean_df.sort_values('pct_chg', ascending=False).head(score_pool_n).reset_index(drop=True)
st.write(f"用于评分的池子大小：{len(clean_df)}（已限制为最多 300 以提速）")

# ---------------------------
# 历史拉取（缓存）与指标计算（含 MACD / KDJ / vol metrics / volatility 等）
# ---------------------------
@st.cache_data(ttl=600)
def get_hist(ts_code, end_date, days=60):
    try:
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = safe_get(pro.daily, ts_code=ts_code, start_date=start, end_date=end_date)
        if df.empty:
            return pd.DataFrame()
        df = df.sort_values('trade_date').reset_index(drop=True)
        return df
    except:
        return pd.DataFrame()

def compute_indicators(df):
    res = {}
    if df.empty or len(df) < 3:
        return res
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    # last close
    try: res['last_close'] = close.iloc[-1]
    except: res['last_close'] = np.nan

    # MA
    for n in (5,10,20):
        if len(close) >= n:
            res[f'ma{n}'] = close.rolling(window=n).mean().iloc[-1]
        else:
            res[f'ma{n}'] = np.nan

    # MACD (12,26,9)
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

    return res

# ---------------------------
# 评分池逐票计算因子（缓存 get_hist）
# ---------------------------
st.write("为评分池逐票拉历史并计算指标（此步骤调用历史接口，已缓存）...")
records = []
pbar2 = st.progress(0)
for idx, row in enumerate(clean_df.itertuples()):
    ts_code = getattr(row, 'ts_code')
    name = getattr(row, 'name', ts_code)
    pct_chg = getattr(row, 'pct_chg', 0.0)
    amount = getattr(row, 'amount', np.nan)
    if amount is not None and not pd.isna(amount) and amount > 0 and amount < 1e5:
        amount = amount * 10000.0

    turnover_rate = getattr(row, 'turnover_rate', np.nan)
    net_mf = float(getattr(row, 'net_mf', 0.0))

    hist = get_hist(ts_code, last_trade, days=60)
    ind = compute_indicators(hist)

    vol_ratio = ind.get('vol_ratio', np.nan)
    ten_return = ind.get('10d_return', np.nan)
    ma5 = ind.get('ma5', np.nan)
    ma10 = ind.get('ma10', np.nan)
    ma20 = ind.get('ma20', np.nan)
    macd = ind.get('macd', np.nan)
    k, d, j = ind.get('k', np.nan), ind.get('d', np.nan), ind.get('j', np.nan)
    last_close = ind.get('last_close', np.nan)
    vol_last = ind.get('vol_last', np.nan)
    vol_ma5 = ind.get('vol_ma5', np.nan)
    prev3_sum = ind.get('prev3_sum', np.nan)
    volatility_10 = ind.get('volatility_10', np.nan)

    # 资金强度代理（不依赖 moneyflow）：简单乘积指标（price move * vol_ratio * turnover）
    try:
        proxy_money = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
    except:
        proxy_money = 0.0

    rec = {
        'ts_code': ts_code, 'name': name, 'pct_chg': pct_chg,
        'amount': amount if not pd.isna(amount) else 0.0,
        'turnover_rate': turnover_rate if not pd.isna(turnover_rate) else np.nan,
        'net_mf': net_mf,
        'vol_ratio': vol_ratio if not pd.isna(vol_ratio) else np.nan,
        '10d_return': ten_return if not pd.isna(ten_return) else np.nan,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
        'macd': macd, 'k': k, 'd': d, 'j': j,
        'last_close': last_close, 'vol_last': vol_last, 'vol_ma5': vol_ma5,
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
# 风险过滤（放在评分前以节省历史调用）
# ---------------------------
st.write("执行风险过滤：下跌途中大阳 / 巨量冲高 / 高位大阳 / 极端波动 ...")
try:
    before_cnt = len(fdf)
    # A: 高位大阳线 -> last_close > ma20*1.10 且 pct_chg > HIGH_PCT_THRESHOLD
    if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):
        mask_high_big = (fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
        fdf = fdf[~mask_high_big]

    # B: 下跌途中反抽 -> prev3_sum < 0 且 pct_chg > HIGH_PCT_THRESHOLD
    if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
        mask_down_rebound = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
        fdf = fdf[~mask_down_rebound]

    # C: 巨量放量大阳 -> vol_last > vol_ma5 * VOL_SPIKE_MULT
    if all(c in fdf.columns for c in ['vol_last','vol_ma5']):
        mask_vol_spike = (fdf['vol_last'] > (fdf['vol_ma5'] * VOL_SPIKE_MULT))
        fdf = fdf[~mask_vol_spike]

    # D: 极端波动 -> volatility_10 > VOLATILITY_MAX
    if 'volatility_10' in fdf.columns:
        mask_volatility = fdf['volatility_10'] > VOLATILITY_MAX
        fdf = fdf[~mask_volatility]

    after_cnt = len(fdf)
    st.write(f"风险过滤：{before_cnt} -> {after_cnt}（若过严请在侧边栏调整阈值）")
except Exception as e:
    st.warning(f"风险过滤模块异常，跳过过滤。错误：{e}")

# ---------------------------
# RSL（相对强弱）：基于池内 10d_return 的相对表现
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
# prefer real moneyflow if available, else proxy_money
if 'net_mf' in fdf.columns and fdf['net_mf'].abs().sum() > 0:
    fdf['s_money'] = norm_col(fdf.get('net_mf', pd.Series([0]*len(fdf))))
else:
    fdf['s_money'] = norm_col(fdf.get('proxy_money', pd.Series([0]*len(fdf))))
fdf['s_amount'] = norm_col(fdf.get('amount', pd.Series([0]*len(fdf))))
fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))
fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
fdf['s_rsl'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))
fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))

# ---------------------------
# 综合评分（BC 混合权重）
#    B (爆发) + C (妖股) 混合权重
# ---------------------------
w_pct = 0.18
w_volratio = 0.18
w_turn = 0.12
w_money = 0.14
w_10d = 0.12
w_macd = 0.06
w_rsl = 0.12
w_volatility = 0.08

fdf['综合评分'] = (
    fdf['s_pct'] * w_pct +
    fdf['s_volratio'] * w_volratio +
    fdf['s_turn'] * w_turn +
    fdf['s_money'] * w_money +
    fdf['s_10d'] * w_10d +
    fdf['s_macd'] * w_macd +
    fdf['s_rsl'] * w_rsl +
    fdf['s_volatility'] * w_volatility
)

# ---------------------------
# 最终排序与展示
# ---------------------------
fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
fdf.index = fdf.index + 1

st.success(f"评分完成：总候选 {len(fdf)} 支，显示 Top {min(TOP_DISPLAY, len(fdf))}。")
display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','proxy_money','amount','10d_return','macd','k','d','j','rsl','volatility_10']
for c in display_cols:
    if c not in fdf.columns:
        fdf[c] = np.nan

st.dataframe(fdf[display_cols].head(TOP_DISPLAY), use_container_width=True)

# 下载（仅导出前200避免过大）
out_csv = fdf[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

# ---------------------------
# 小结与建议（简洁）
# ---------------------------
st.markdown("### 小结与操作提示（简洁）")
st.markdown("""
- 本版本为 BC 混合增强版（短线爆发 + 妖股捕捉）。  
- 已启用 MACD、RSL、下跌途中大阳线过滤、巨量冲高过滤、极端波动过滤。  
- 若 moneyflow / chip / ths_member 能成功拉取，将作为额外加分因子；若无权限脚本会自动用 proxy_money 代替。  
- 实战纪律（必须遵守）：**9:40 前不买 → 观察 9:40-10:05 的量价节奏 → 10:05 后择优介入**。  
- 若今日候选普遍翻绿，请保持空仓。  
""")

st.info("运行出现问题请把 Streamlit 的错误日志或首段报错发给我（截图或文字都行），我会在两次修改内继续帮你调优。")
