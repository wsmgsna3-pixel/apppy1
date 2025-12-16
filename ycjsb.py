# -*- coding: utf-8 -*-
"""
选股王 V30.11
可信回测 + 提速版（最终推荐）
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
st.set_page_config(page_title="选股王 V30.11（可信·提速）", layout="wide")
st.title("选股王 V30.11 · 可信回测（提速版）")

# ===============================
# 侧边栏
# ===============================
with st.sidebar:
    st.header("基础设置")
    TS_TOKEN = st.text_input("Tushare Token", type="password")
    BACKTEST_DAYS = st.number_input("回测天数", value=60, step=20)
    TOP_K = st.number_input("每日选股数量", value=5)
    run_btn = st.button("🚀 运行可信回测（提速）")

if not TS_TOKEN:
    st.info("请先在左侧输入 Tushare Token")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ===============================
# 批量行情 + 复权（提速核心）
# ===============================
@st.cache_data(ttl=3600)
def get_daily_qfq_by_date(trade_date, lookback=150):
    start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=lookback)).strftime("%Y%m%d")

    daily = pro.daily(start_date=start, end_date=trade_date)
    adj = pro.adj_factor(start_date=start, end_date=trade_date)

    if daily is None or adj is None or daily.empty or adj.empty:
        return None

    need = {"ts_code", "trade_date", "open", "high", "low", "close"}
    if not need.issubset(daily.columns):
        return None

    df = daily.merge(adj, on=["ts_code", "trade_date"], how="inner")
    if df.empty:
        return None

    df = df.sort_values(["ts_code", "trade_date"])

    # ⚠️ 每只股票：用“该回测日”为复权基准（不看未来）
    def qfq_one_stock(x):
        base = x["adj_factor"].iloc[-1]
        for c in ["open", "high", "low", "close"]:
            x[c] = x[c] * x["adj_factor"] / base
        return x

    df = df.groupby("ts_code", group_keys=False).apply(qfq_one_stock)
    return df

# ===============================
# 指标计算（仅用历史）
# ===============================
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
    bar = st.progress(0.0)

    for i, day in enumerate(trade_days):
        hist_all = get_daily_qfq_by_date(day)
        if hist_all is None:
            continue

        picks = []

        for ts_code, hist in hist_all.groupby("ts_code"):
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

            if ind["close"] <= 10:
                continue

            picks.append({
                "交易日": day,
                "股票代码": ts_code,
                "收盘价": round(ind["close"], 2),
                "MACD": round(ind["macd"], 3),
                "RSI": round(ind["rsi"], 1)
            })

        if picks:
            df = pd.DataFrame(picks).sort_values("MACD", ascending=False).head(TOP_K)
            results.append(df)

        bar.progress((i + 1) / len(trade_days))

    bar.empty()

    if results:
        all_res = pd.concat(results)
        st.subheader("📊 回测选股结果（可信 · 提速）")
        st.dataframe(all_res.head(100))
    else:
        st.warning("回测完成，但未选出股票")
