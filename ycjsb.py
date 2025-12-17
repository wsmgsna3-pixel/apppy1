# -*- coding: utf-8 -*-
"""
选股王 V30.11
可信回测 · 可用版（解决长期 0 命中）
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
st.set_page_config(page_title="选股王 V30.11（可信·可用版）", layout="wide")
st.title("📈 选股王 V30.11 · 可信回测（可用版）")

# ===============================
# 侧边栏
# ===============================
with st.sidebar:
    TS_TOKEN = st.text_input("Tushare Token", type="password")
    BACKTEST_DAYS = st.number_input("回测天数", value=100, step=20)
    TOP_K = st.number_input("每日选股数量", value=5)
    RUN = st.button("🚀 开始回测")

if not TS_TOKEN:
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api(timeout=60)

def safe(func, **kwargs):
    for _ in range(3):
        try:
            return func(**kwargs)
        except:
            time.sleep(2)
    return None

@st.cache_data(ttl=3600)
def get_hist(day, lookback=120):
    start = (datetime.strptime(day, "%Y%m%d") - timedelta(days=lookback)).strftime("%Y%m%d")
    d = safe(pro.daily, start_date=start, end_date=day)
    a = safe(pro.adj_factor, start_date=start, end_date=day)
    if d is None or a is None:
        return None
    df = d.merge(a, on=["ts_code", "trade_date"])
    df = df.sort_values(["ts_code", "trade_date"])

    def qfq(x):
        base = x["adj_factor"].iloc[-1]
        for c in ["open","high","low","close"]:
            x[c] = x[c] * x["adj_factor"] / base
        return x

    return df.groupby("ts_code", group_keys=False).apply(qfq)

def indicators(df):
    if len(df) < 60:
        return None
    c = df["close"]
    h = df["high"]
    l = df["low"]
    ma60 = c.tail(60).mean()
    body_pos = (c.iloc[-1] - l.iloc[-1]) / (h.iloc[-1] - l.iloc[-1] + 1e-9)
    upper = (h.iloc[-1] - c.iloc[-1]) / c.iloc[-1] * 100
    macd = (c.ewm(span=12,adjust=False).mean() -
            c.ewm(span=26,adjust=False).mean()).iloc[-1]
    return c.iloc[-1], ma60, body_pos, upper, macd

# ===============================
# 回测
# ===============================
if RUN:
    cal = safe(pro.trade_cal, is_open="1")
    days = cal.sort_values("cal_date", ascending=False).head(BACKTEST_DAYS)["cal_date"].tolist()
    st.success(f"📅 回测区间：{min(days)} ~ {max(days)}")

    res = []
    bar = st.progress(0.0)

    for i, day in enumerate(days):
        d = safe(pro.daily, trade_date=day)
        b = safe(pro.daily_basic, trade_date=day,
                 fields="ts_code,turnover_rate,circ_mv,amount")
        if d is None or b is None:
            continue

        df0 = d.merge(b, on="ts_code")
        df0["circ_mv"] /= 10000
        df0["amount"] *= 1000

        df0 = df0[
            (df0["close"] >= 10) &
            (df0["close"] <= 200) &
            (df0["circ_mv"] >= 20) &
            (df0["circ_mv"] <= 500) &
            (df0["turnover_rate"] >= 3) &
            (df0["amount"] >= 1e8)
        ]

        if df0.empty:
            continue

        hist = get_hist(day)
        if hist is None:
            continue

        hist = hist[hist["ts_code"].isin(df0["ts_code"])]

        picks = []
        for code, hdf in hist.groupby("ts_code"):
            ind = indicators(hdf)
            if not ind:
                continue
            close, ma60, body_pos, upper, macd = ind

            # ===== 可用版 Alpha =====
            if close < ma60 * 0.97:
                continue
            if body_pos < 0.5:
                continue
            if upper > 6:
                continue

            picks.append({
                "交易日": day,
                "股票": code,
                "收盘价": round(close,2),
                "MACD": round(macd,3)
            })

        if picks:
            res.append(pd.DataFrame(picks).sort_values("MACD", ascending=False).head(TOP_K))

        bar.progress((i+1)/len(days))

    bar.empty()

    if res:
        st.dataframe(pd.concat(res, ignore_index=True))
    else:
        st.warning("⚠️ 回测完成：该区间没有出现满足条件的股票")
