# -*- coding: utf-8 -*-
"""
选股王 V30.11
可信回测 · 新手友好 · 抗 Tushare 超时稳定版
"""

import streamlit as st
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 页面初始化
# ===============================
st.set_page_config(
    page_title="选股王 V30.11（可信·稳定版）",
    layout="wide"
)
st.title("📈 选股王 V30.11 · 可信回测（稳定抗超时）")

# ===============================
# 侧边栏
# ===============================
with st.sidebar:
    st.header("🔐 基础设置")
    TS_TOKEN = st.text_input("Tushare Token", type="password")

    st.markdown("---")
    BACKTEST_DAYS = st.number_input("回测天数", value=30, step=10)
    TOP_K = st.number_input("每日选股数量", value=5)

    st.markdown("---")
    st.markdown("📌 **V30.11 固定选股条件（已调优）**")
    RUN_BTN = st.button("🚀 运行可信回测")

if not TS_TOKEN:
    st.info("👈 请先输入 Tushare Token")
    st.stop()

# ===============================
# Tushare 初始化（加长超时）
# ===============================
ts.set_token(TS_TOKEN)
pro = ts.pro_api(timeout=60)

# ===============================
# 安全请求封装（核心稳定器）
# ===============================
def safe_query(func, **kwargs):
    for _ in range(3):  # 最多重试 3 次
        try:
            return func(**kwargs)
        except Exception:
            time.sleep(2)
    return None

# ===============================
# 批量行情 + 前复权（安全版）
# ===============================
@st.cache_data(ttl=3600)
def get_daily_qfq_by_date(trade_date, lookback=150):
    start = (
        datetime.strptime(trade_date, "%Y%m%d")
        - timedelta(days=lookback)
    ).strftime("%Y%m%d")

    daily = safe_query(pro.daily, start_date=start, end_date=trade_date)
    adj = safe_query(pro.adj_factor, start_date=start, end_date=trade_date)

    if daily is None or adj is None or daily.empty or adj.empty:
        return None

    df = daily.merge(adj, on=["ts_code", "trade_date"], how="inner")
    df = df.sort_values(["ts_code", "trade_date"])

    def qfq_one(x):
        base = x["adj_factor"].iloc[-1]
        for c in ["open", "high", "low", "close"]:
            x[c] = x[c] * x["adj_factor"] / base
        return x

    time.sleep(1)  # 🔒 限速，防止接口被打爆
    return df.groupby("ts_code", group_keys=False).apply(qfq_one)

# ===============================
# 技术指标
# ===============================
def calc_indicators(df):
    if df is None or len(df) < 60:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ma60 = close.tail(60).mean()
    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    ).iloc[-1]

    body_pos = (close.iloc[-1] - low.iloc[-1]) / (
        high.iloc[-1] - low.iloc[-1] + 1e-9
    )

    upper_shadow = (high.iloc[-1] - close.iloc[-1]) / close.iloc[-1] * 100

    return ma60, macd, body_pos, upper_shadow, close.iloc[-1]

# ===============================
# 回测主逻辑
# ===============================
if RUN_BTN:
    today = datetime.now().strftime("%Y%m%d")
    cal = safe_query(pro.trade_cal, end_date=today, is_open="1")
    trade_days = (
        cal.sort_values("cal_date", ascending=False)
        .head(BACKTEST_DAYS)["cal_date"]
        .tolist()
    )

    st.success(f"📅 实际回测区间：{min(trade_days)} ~ {max(trade_days)}")

    results = []
    bar = st.progress(0.0)

    for i, day in enumerate(trade_days):

        daily = safe_query(pro.daily, trade_date=day)
        basic = safe_query(
            pro.daily_basic,
            trade_date=day,
            fields="ts_code,turnover_rate,circ_mv,amount"
        )

        if daily is None or basic is None:
            continue

        df0 = daily.merge(basic, on="ts_code", how="inner")
        df0["circ_mv_b"] = df0["circ_mv"] / 10000
        df0["amount"] = df0["amount"] * 1000

        df0 = df0[
            (df0["close"] >= 10) &
            (df0["close"] <= 200) &
            (df0["circ_mv_b"] >= 20) &
            (df0["circ_mv_b"] <= 500) &
            (df0["turnover_rate"] >= 3) &
            (df0["amount"] >= 1e8)
        ]

        ts_list = df0["ts_code"].tolist()
        if not ts_list:
            continue

        hist_all = get_daily_qfq_by_date(day)
        if hist_all is None:
            continue

        hist_all = hist_all[hist_all["ts_code"].isin(ts_list)]

        picks = []

        for ts_code, hist in hist_all.groupby("ts_code"):
            ind = calc_indicators(hist)
            if not ind:
                continue

            ma60, macd, body_pos, upper_shadow, close = ind

            if close < ma60:
                continue
            if upper_shadow > 4:
                continue
            if body_pos < 0.6:
                continue

            picks.append({
                "交易日": day,
                "股票代码": ts_code,
                "收盘价": round(close, 2),
                "MACD": round(macd, 3)
            })

        if picks:
            results.append(
                pd.DataFrame(picks)
                .sort_values("MACD", ascending=False)
                .head(TOP_K)
            )

        bar.progress((i + 1) / len(trade_days))

    bar.empty()

    if results:
        st.subheader("📊 回测选股结果（可信·稳定）")
        st.dataframe(pd.concat(results, ignore_index=True))
    else:
        st.warning("⚠️ 回测完成：该区间未出现符合条件的股票")
