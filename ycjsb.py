# -*- coding: utf-8 -*-
"""
选股王 V30.11
可信回测 · 新手友好版（完整覆盖运行）
"""

import streamlit as st
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 页面初始化
# ===============================
st.set_page_config(
    page_title="选股王 V30.11（可信·新手版）",
    layout="wide"
)
st.title("📈 选股王 V30.11 · 可信回测（新手友好版）")

# ===============================
# 侧边栏（只填这里）
# ===============================
with st.sidebar:
    st.header("🔐 基础设置")
    TS_TOKEN = st.text_input("Tushare Token", type="password")

    st.markdown("---")
    st.header("📅 回测参数")
    BACKTEST_DAYS = st.number_input("回测天数", value=30, step=10)
    TOP_K = st.number_input("每日选股数量", value=5)

    st.markdown("---")
    st.header("📌 V30.11 硬条件（已调优）")
    MIN_PRICE = 10.0
    MAX_PRICE = 200.0

    MIN_CIRC_MV = 20.0     # 亿
    MAX_CIRC_MV = 500.0    # 亿

    MIN_TURNOVER = 3.0     # 🔧 从 5 → 3
    MIN_AMOUNT = 1.0       # 亿

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
# 批量行情 + 前复权（不看未来）
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

    df = daily.merge(adj, on=["ts_code", "trade_date"], how="inner")
    df = df.sort_values(["ts_code", "trade_date"])

    # ⚠️ 复权基准 = 当日（安全）
    def qfq_one(x):
        base = x["adj_factor"].iloc[-1]
        for c in ["open", "high", "low", "close"]:
            x[c] = x[c] * x["adj_factor"] / base
        return x

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

    return {
        "close": close.iloc[-1],
        "ma60": ma60,
        "macd": macd,
        "body_pos": body_pos,
        "upper_shadow": upper_shadow
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
    bar = st.progress(0.0)

    for i, day in enumerate(trade_days):

        # --------- 当日粗筛 ---------
        daily = pro.daily(trade_date=day)
        basic = pro.daily_basic(
            trade_date=day,
            fields="ts_code,turnover_rate,circ_mv,amount"
        )

        if daily is None or basic is None:
            continue

        df0 = daily.merge(basic, on="ts_code", how="inner")

        df0["circ_mv_b"] = df0["circ_mv"] / 10000
        df0["amount"] = df0["amount"] * 1000

        df0 = df0[
            (df0["close"] >= MIN_PRICE) &
            (df0["close"] <= MAX_PRICE) &
            (df0["circ_mv_b"] >= MIN_CIRC_MV) &
            (df0["circ_mv_b"] <= MAX_CIRC_MV) &
            (df0["turnover_rate"] >= MIN_TURNOVER) &
            (df0["amount"] >= MIN_AMOUNT * 1e8)
        ]

        if df0.empty:
            continue

        ts_list = df0["ts_code"].tolist()

        # --------- 历史行情 ---------
        hist_all = get_daily_qfq_by_date(day)
        if hist_all is None:
            continue

        hist_all = hist_all[hist_all["ts_code"].isin(ts_list)]

        picks = []

        for ts_code, hist in hist_all.groupby("ts_code"):
            ind = calc_indicators(hist)
            if not ind:
                continue

            # ===== V30.11 Alpha（已松绑）=====
            if ind["close"] < ind["ma60"]:
                continue

            if ind["upper_shadow"] > 4:
                continue

            if ind["body_pos"] < 0.6:   # 🔧 0.7 → 0.6
                continue

            picks.append({
                "交易日": day,
                "股票代码": ts_code,
                "收盘价": round(ind["close"], 2),
                "MACD": round(ind["macd"], 3),
                "换手率": round(
                    df0[df0["ts_code"] == ts_code]["turnover_rate"].iloc[0], 1
                )
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
        final_df = pd.concat(results, ignore_index=True)
        st.subheader("📊 V30.11 回测选股结果（可信·新手版）")
        st.dataframe(final_df, use_container_width=True)
    else:
        st.warning("⚠️ 回测完成：该区间未出现符合条件的强趋势股票")
