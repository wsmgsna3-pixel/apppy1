# -*- coding: utf-8 -*-
"""
选股王 V30.11
可信回测 + 实盘共用版本（最终推荐）
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 页面初始化
# ===============================
st.set_page_config(page_title="选股王 V30.11（可信回测）", layout="wide")
st.title("选股王 V30.11 · 可信回测版本")

# ===============================
# 侧边栏
# ===============================
with st.sidebar:
    st.header("基础设置")
    TS_TOKEN = st.text_input("Tushare Token", type="password")
    BACKTEST_DAYS = st.number_input("回测天数", value=100, step=20)
    TOP_K = st.number_input("每日选股数量", value=5)
    run_btn = st.button("🚀 运行可信回测")

if not TS_TOKEN:
    st.info("请先在左侧输入 Tushare Token")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ===============================
# 工具函数（可信）
# ===============================
def safe_daily(trade_date):
    df = pro.daily(trade_date=trade_date)
    if df is None or df.empty:
        return None
    need = {"ts_code", "open", "high", "low", "close", "trade_date"}
    if not need.issubset(df.columns):
        return None
    return df.copy()

def get_qfq(ts_code, start, end):
    daily = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    adj = pro.adj_factor(ts_code=ts_code, start_date=start, end_date=end)
    if daily is None or adj is None or daily.empty or adj.empty:
        return None

    df = daily.merge(adj, on=["ts_code", "trade_date"], how="inner")
    if df.empty:
        return None

    df = df.sort_values("trade_date")
    base = df["adj_factor"].iloc[-1]
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c] * df["adj_factor"] / base

    return df

def calc_indicators(df):
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

# ===============================
# 回测主逻辑
# ===============================
if run_btn:
    end_date = datetime.now().strftime("%Y%m%d")
    cal = pro.trade_cal(end_date=end_date, is_open='1')
    trade_days = cal.sort_values("cal_date", ascending=False)["cal_date"].head(BACKTEST_DAYS).tolist()

    st.write(f"📅 实际回测区间：{min(trade_days)} ~ {max(trade_days)}")

    results = []
    progress = st.progress(0.0)

    for i, day in enumerate(trade_days):
        daily = safe_daily(day)
        if daily is None:
            continue

        daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
        daily = daily.dropna(subset=["close"])
        daily = daily[daily["close"] > 10]

        picks = []

        for row in daily.itertuples():
            start = (datetime.strptime(day, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
            hist = get_qfq(row.ts_code, start, day)
            ind = calc_indicators(hist)
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
                "交易日": day,
                "股票代码": row.ts_code,
                "收盘价": ind["close"],
                "MACD": ind["macd"],
                "RSI": ind["rsi"]
            })

        if picks:
            df = pd.DataFrame(picks).sort_values("MACD", ascending=False).head(TOP_K)
            results.append(df)

        progress.progress((i + 1) / len(trade_days))

    progress.empty()

    if results:
        all_res = pd.concat(results)
        st.subheader("📊 回测选股结果（可信）")
        st.dataframe(all_res.head(100))
    else:
        st.warning("回测完成，但未选出股票")
