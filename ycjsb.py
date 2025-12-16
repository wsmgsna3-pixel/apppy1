# -*- coding: utf-8 -*-
"""
选股王 V30.11
可信回测 + 提速 + 完整硬条件版
（新手可直接覆盖运行）
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
st.set_page_config(
    page_title="选股王 V30.11（可信·提速·完整版）",
    layout="wide"
)
st.title("📈 选股王 V30.11 · 可信回测（提速完整版）")

# ===============================
# 侧边栏（你只需要填这里）
# ===============================
with st.sidebar:
    st.header("🔐 基础设置")
    TS_TOKEN = st.text_input("Tushare Token", type="password")

    st.markdown("---")
    st.header("📅 回测参数")
    BACKTEST_DAYS = st.number_input("回测天数", value=30, step=10)
    TOP_K = st.number_input("每日选股数量", value=5)

    st.markdown("---")
    st.header("📌 核心硬条件（V30.11）")
    MIN_PRICE = st.number_input("最低股价", value=10.0)
    MAX_PRICE = st.number_input("最高股价", value=200.0)

    MIN_CIRC_MV = st.number_input("最小流通市值（亿）", value=20.0)
    MAX_CIRC_MV = st.number_input("最大流通市值（亿）", value=500.0)

    MIN_TURNOVER = st.number_input("最低换手率（%）", value=5.0)
    MIN_AMOUNT = st.number_input("最低成交额（亿元）", value=1.0)

    st.markdown("---")
    RUN_BTN = st.button("🚀 运行可信回测")

if not TS_TOKEN:
    st.info("👈 请先在左侧输入你的 Tushare Token")
    st.stop()

# ===============================
# Tushare 初始化
# ===============================
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ===============================
# 批量行情 + 前复权（提速核心）
# ===============================
@st.cache_data(ttl=3600)
def get_daily_qfq_by_date(trade_date, lookback=150):
    start = (
        datetime.strptime(trade_date, "%Y%m%d")
        - timedelta(days=lookback)
    ).strftime("%Y%m%d")

    daily = pro.daily(start_date=start, end_date=trade_date)
    adj = pro.adj_factor(start_date=start, end_date=trade_date)

    if daily is None or adj is None or daily.empty or adj.empty:
        return None

    need_cols = {"ts_code", "trade_date", "open", "high", "low", "close"}
    if not need_cols.issubset(daily.columns):
        return None

    df = daily.merge(adj, on=["ts_code", "trade_date"], how="inner")
    if df.empty:
        return None

    df = df.sort_values(["ts_code", "trade_date"])

    # ⚠️ 每只股票用“当日”作为复权基准（不看未来）
    def qfq_one_stock(x):
        base = x["adj_factor"].iloc[-1]
        for c in ["open", "high", "low", "close"]:
            x[c] = x[c] * x["adj_factor"] / base
        return x

    df = df.groupby("ts_code", group_keys=False).apply(qfq_one_stock)
    return df

# ===============================
# 技术指标（只用历史）
# ===============================
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
        "rsi": rsi.iloc[-1],
        "pos60": pos60
    }

# ===============================
# 回测主逻辑
# ===============================
if RUN_BTN:
    today = datetime.now().strftime("%Y%m%d")
    cal = pro.trade_cal(end_date=today, is_open="1")
    trade_days = (
        cal.sort_values("cal_date", ascending=False)
        .head(BACKTEST_DAYS)["cal_date"]
        .tolist()
    )

    st.success(
        f"📅 实际回测区间：{min(trade_days)} ~ {max(trade_days)}"
    )

    results = []
    progress = st.progress(0.0)

    for i, day in enumerate(trade_days):

        # ---------- 第一步：当日粗筛 ----------
        daily = pro.daily(trade_date=day)
        basic = pro.daily_basic(
            trade_date=day,
            fields="ts_code,turnover_rate,circ_mv,amount"
        )

        if daily is None or basic is None or daily.empty or basic.empty:
            continue

        df0 = daily.merge(basic, on="ts_code", how="inner")

        df0["close"] = pd.to_numeric(df0["close"], errors="coerce")
        df0["circ_mv_billion"] = df0["circ_mv"] / 10000
        df0["amount"] = df0["amount"] * 1000

        df0 = df0[
            (df0["close"] >= MIN_PRICE) &
            (df0["close"] <= MAX_PRICE) &
            (df0["circ_mv_billion"] >= MIN_CIRC_MV) &
            (df0["circ_mv_billion"] <= MAX_CIRC_MV) &
            (df0["turnover_rate"] >= MIN_TURNOVER) &
            (df0["amount"] >= MIN_AMOUNT * 1e8)
        ]

        if df0.empty:
            continue

        ts_list = df0["ts_code"].unique().tolist()

        # ---------- 第二步：批量取历史 + 复权 ----------
        hist_all = get_daily_qfq_by_date(day)
        if hist_all is None:
            continue

        hist_all = hist_all[hist_all["ts_code"].isin(ts_list)]

        picks = []

        for ts_code, hist in hist_all.groupby("ts_code"):
            ind = calc_indicators(hist)
            if not ind:
                continue

            # ---- V30.11 Alpha 核心过滤 ----
            if ind["close"] < ind["ma60"]:
                continue

            upper_shadow = (ind["high"] - ind["close"]) / ind["close"] * 100
            if upper_shadow > 4:
                continue

            body_pos = (ind["close"] - ind["low"]) / (
                ind["high"] - ind["low"] + 1e-9
            )
            if body_pos < 0.7:
                continue

            picks.append({
                "交易日": day,
                "股票代码": ts_code,
                "收盘价": round(ind["close"], 2),
                "MACD": round(ind["macd"], 3),
                "RSI": round(ind["rsi"], 1),
                "流通市值(亿)": round(
                    df0[df0["ts_code"] == ts_code]["circ_mv_billion"].iloc[0],
                    1
                )
            })

        if picks:
            day_df = (
                pd.DataFrame(picks)
                .sort_values("MACD", ascending=False)
                .head(TOP_K)
            )
            results.append(day_df)

        progress.progress((i + 1) / len(trade_days))

    progress.empty()

    if results:
        final_df = pd.concat(results, ignore_index=True)
        st.subheader("📊 V30.11 回测选股结果（可信）")
        st.dataframe(final_df, use_container_width=True)
    else:
        st.warning("⚠️ 回测完成，但在该区间未选出符合条件的股票")
