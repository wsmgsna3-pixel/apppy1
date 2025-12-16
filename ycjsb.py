# ===============================
# V30.11 可信回测内核（Final）
# ===============================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------- 安全数据接口 ----------

def safe_daily(pro, trade_date):
    df = pro.daily(trade_date=trade_date)
    if df is None or df.empty:
        return None
    need = {"ts_code", "open", "high", "low", "close", "trade_date"}
    if not need.issubset(df.columns):
        return None
    return df.copy()

def safe_adj(pro, ts_code, start, end):
    df = pro.adj_factor(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return None
    return df.sort_values("trade_date")

def get_qfq_safe(pro, ts_code, start, end):
    daily = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    adj = safe_adj(pro, ts_code, start, end)
    if daily is None or daily.empty or adj is None:
        return None

    df = daily.merge(adj, on=["ts_code", "trade_date"], how="inner")
    if df.empty:
        return None

    df = df.sort_values("trade_date")
    base = df["adj_factor"].iloc[-1]
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c] * df["adj_factor"] / base

    return df

# ---------- 指标计算（只用历史） ----------

def calc_v30_indicators(df):
    if df is None or len(df) < 60:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ma20 = close.tail(20).mean()
    ma60 = close.tail(60).mean()

    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    ).iloc[-1]

    # RSI 12（安全版）
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(12).mean()
    loss = -diff.clip(upper=0).rolling(12).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-9))
    rsi_val = rsi.iloc[-1]

    pos60 = (close.iloc[-1] - low.tail(60).min()) / (
        high.tail(60).max() - low.tail(60).min() + 1e-9
    ) * 100

    return {
        "close": close.iloc[-1],
        "high": high.iloc[-1],
        "low": low.iloc[-1],
        "ma20": ma20,
        "ma60": ma60,
        "macd": macd,
        "rsi": rsi_val,
        "pos60": pos60
    }

# ---------- 核心选股（逐日） ----------

def run_v30_for_day(pro, trade_date, TOP_K=5):
    daily = safe_daily(pro, trade_date)
    if daily is None:
        return pd.DataFrame()

    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.dropna(subset=["close"])
    daily = daily[daily["close"] > 10]

    picks = []

    for row in daily.itertuples():
        start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
        hist = get_qfq_safe(pro, row.ts_code, start, trade_date)
        ind = calc_v30_indicators(hist)
        if not ind:
            continue

        # ---- V30.11 核心过滤 ----
        if ind["close"] < ind["ma60"]:
            continue

        upper_shadow = (ind["high"] - ind["close"]) / ind["close"] * 100
        if upper_shadow > 4:
            continue

        body_pos = (ind["close"] - ind["low"]) / (ind["high"] - ind["low"] + 1e-9)
        if body_pos < 0.7:
            continue

        picks.append({
            "trade_date": trade_date,
            "ts_code": row.ts_code,
            "close": ind["close"],
            "macd": ind["macd"],
            "rsi": ind["rsi"]
        })

    if not picks:
        return pd.DataFrame()

    df = pd.DataFrame(picks)
    return df.sort_values("macd", ascending=False).head(TOP_K)
