# signal_builder.py
# 构建选股信号：MACD/RSI/量价/趋势/真突破/妖股剔除/资金流（优雅降级）

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts
from typing import Dict, Any

# Global pro will be set by caller
pro = None

def set_pro(api):
    global pro
    pro = api

def safe_get(func, **kwargs):
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

def get_hist(ts_code, start_date=None, end_date=None):
    """拉取日线历史数据（降级容错）"""
    if pro is None:
        raise RuntimeError("Tushare pro not set. Call set_pro(pro) first.")
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    df = safe_get(pro.daily, ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty:
        return df
    df = df.sort_values("trade_date").reset_index(drop=True)
    # ensure numeric columns
    for col in ["open","high","low","close","vol","amount","pre_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        return d
    d['close'] = pd.to_numeric(d['close'], errors='coerce')
    d['vol'] = pd.to_numeric(d['vol'], errors='coerce')
    # EMA for MACD
    d['ema12'] = d['close'].ewm(span=12, adjust=False).mean()
    d['ema26'] = d['close'].ewm(span=26, adjust=False).mean()
    d['macd'] = d['ema12'] - d['ema26']
    d['signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['hist'] = d['macd'] - d['signal']
    # RSI
    diff = d['close'].diff()
    up = diff.where(diff > 0, 0)
    down = -diff.where(diff < 0, 0)
    d['rsi'] = 100 - 100 / (1 + (up.rolling(14).mean() / down.rolling(14).mean()))
    # MA
    d['ma5'] = d['close'].rolling(5).mean()
    d['ma10'] = d['close'].rolling(10).mean()
    d['ma20'] = d['close'].rolling(20).mean()
    d['vol_ma5'] = d['vol'].rolling(5).mean()
    d['pct5'] = (d['close'] / d['close'].shift(5) - 1) * 100
    return d

def is_true_breakthrough(df: pd.DataFrame, vol_ratio_thresh=1.2, turnover_thresh=3.0, macd_thresh=0.0, rsi_max=75):
    """
    判定“真突破”：当天收盘位于历史若干区间新高，并且量价/动能支持
    返回 True/False
    """
    if df.empty or len(df) < 30:
        return False
    d = compute_indicators(df)
    last = d.iloc[-1]
    # 近20日最高
    high20 = d['high'].rolling(20).max().iloc[-1]
    is_new_high = last['close'] >= high20
    vol_ratio = last['vol'] / (d['vol_ma5'].iloc[-1] if not np.isnan(d['vol_ma5'].iloc[-1]) and d['vol_ma5'].iloc[-1] > 0 else 1)
    turnover = None
    if 'turnover_rate' in d.columns:
        turnover = d.get('turnover_rate', np.nan)
    cond1 = (vol_ratio >= vol_ratio_thresh)
    cond2 = (last['macd'] >= macd_thresh)
    cond3 = (last['rsi'] <= rsi_max)
    # require at least two of volume/macd/rsi to be good OR high volume + macd positive
    score = sum([is_new_high, cond1, cond2, cond3])
    return score >= 3 or (is_new_high and cond1 and cond2)

def basic_filters(ts_code: str, last_trade: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行一系列基本过滤并返回指标与标记字典
    params 包括：MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, VOL_RATIO_MIN, RSI_MAX, MACD_MIN, MAX_5D_PCT
    """
    out = {"ts_code": ts_code, "ok": False}
    hist = get_hist(ts_code, start_date=params.get("start_date", None), end_date=last_trade)
    if hist.empty or len(hist) < 30:
        return out
    d = compute_indicators(hist)
    last = d.iloc[-1]
    # price
    if last['close'] < params['MIN_PRICE'] or last['close'] > params['MAX_PRICE']:
        return out
    # turnover and amount if available in the day's row from external merge, those can be added by caller
    if 'turnover_rate' in last and not np.isnan(last['turnover_rate']):
        if float(last['turnover_rate']) < params['MIN_TURNOVER']:
            return out
    # amount field may be in historical df as 'amount' (in 元)
    if 'amount' in last and not np.isnan(last['amount']):
        if float(last['amount']) < params['MIN_AMOUNT']:
            return out
    # trend: close > ma10 > ma20 (温和要求)
    if not (last['close'] > last['ma10'] and last['ma10'] > last['ma20']):
        return out
    # MA20 upward
    ma20_now = d['ma20'].iloc[-1]
    ma20_prev = d['ma20'].iloc[-6]  # 5 days ago
    if ma20_now <= ma20_prev:
        return out
    # avoid high volatility / high-level choppy
    max20 = d['close'].rolling(20).max().iloc[-1]
    min20 = d['close'].rolling(20).min().iloc[-1]
    if (max20 - min20) / (min20 + 1e-9) > params.get('MAX_20_RANGE', 0.4):
        # Too volatile in 20 days (default allow quite large)
        return out
    # 5d pct constraint
    if d['pct5'].iloc[-1] > params.get('MAX_5D_PCT', 40):
        return out
    # RSI constraint (avoid extreme overbought)
    if last['rsi'] > params.get('RSI_MAX', 75):
        return out
    # macd check
    if last['macd'] < params.get('MACD_MIN', -0.5):
        return out
    # vol ratio
    vol_ratio = last['vol'] / (d['vol_ma5'].iloc[-1] if not np.isnan(d['vol_ma5'].iloc[-1]) and d['vol_ma5'].iloc[-1] > 0 else 1)
    if vol_ratio < params.get('VOL_RATIO_MIN', 1.0):
        return out
    # True breakthrough check:
    tb = is_true_breakthrough(d,
                              vol_ratio_thresh=params.get('VOL_RATIO_MIN', 1.2),
                              turnover_thresh=params.get('MIN_TURNOVER',3.0),
                              macd_thresh=params.get('MACD_MIN',0.0),
                              rsi_max=params.get('RSI_MAX',75))
    # We will accept tb OR strong trend (to allow innovation high)
    if not (tb or (last['macd'] > 0 and last['close'] > last['ma10'])):
        return out

    out.update({
        "ok": True,
        "close": last['close'],
        "macd": last['macd'],
        "rsi": last['rsi'],
        "vol_ratio": vol_ratio,
        "pct5": d['pct5'].iloc[-1],
        "ma10": last['ma10'],
        "ma20": last['ma20'],
        "hist": d
    })
    return out

# Moneyflow usage helper (optional)
def get_moneyflow(ts_code: str, trade_date: str):
    """Try to fetch moneyflow; returns dict or None"""
    global pro
    if pro is None:
        return None
    try:
        mf = safe_get(pro.moneyflow, ts_code=ts_code, trade_date=trade_date)
        if mf.empty:
            return None
        # pick aggregated amounts
        row = mf.iloc[0].to_dict()
        return row
    except Exception:
        return None
