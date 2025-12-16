# -*- coding: utf-8 -*-
"""
选股王 V30.11 - 回测内核【干净重构版】
核心目标：
- 不预加载全市场
- 不使用全局复权基准
- 回测天数完全可控
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 基础设置
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("选股王 V30.11 · 干净重构回测内核")

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN:
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# -------------------------------------------------
# 工具函数
# -------------------------------------------------
@st.cache_data(ttl=3600)
def get_trade_days(end_date, n):
    cal = pro.trade_cal(
        start_date=(datetime.strptime(end_date, "%Y%m%d") - timedelta(days=n*2)).strftime("%Y%m%d"),
        end_date=end_date,
        is_open='1'
    )
    cal = cal.sort_values("cal_date", ascending=False)
    return cal["cal_date"].head(n).tolist()

@st.cache_data(ttl=3600)
def get_qfq(ts_code, start_date, end_date):
    """按股票、按区间获取前复权行情"""
    df = pro.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    if df is None or df.empty:
        return pd.DataFrame()

    adj = pro.adj_factor(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    if adj is None or adj.empty:
        return pd.DataFrame()

    df = df.merge(adj, on=["ts_code", "trade_date"], how="inner")
    df = df.sort_values("trade_date")

    latest_adj = df["adj_factor"].iloc[-1]
    for c in ["open", "high", "low", "close", "pre_close"]:
        df[c] = df[c] * df["adj_factor"] / latest_adj

    return df

def calc_indicators(df):
    if len(df) < 60:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ma20 = close.tail(20).mean()
    ma60 = close.tail(60).mean()

    macd = (
        close.ewm(span=12).mean()
        - close.ewm(span=26).mean()
    ).iloc[-1]

    rsi = 100 - 100 / (
        1 + close.diff().clip(lower=0).rolling(12).mean()
        / (-close.diff().clip(upper=0).rolling(12).mean())
    ).iloc[-1]

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
        "rsi": rsi,
        "pos60": pos60
    }

def future_return(ts_code, d0, close0, n):
    df = get_qfq(
        ts_code,
        (datetime.strptime(d0, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d"),
        (datetime.strptime(d0, "%Y%m%d") + timedelta(days=15)).strftime("%Y%m%d")
    )
    if len(df) >= n:
        return (df.iloc[n-1]["close"] / close0 - 1) * 100
    return np.nan

# -------------------------------------------------
# 回测主逻辑
# -------------------------------------------------
with st.sidebar:
    end_date = st.date_input("回测结束日期", datetime.now().date())
    BACKTEST_DAYS = st.number_input("回测天数", value=100, step=10)
    TOP_K = st.number_input("每日选股数", value=5)

if st.button("🚀 运行回测"):
    trade_days = get_trade_days(end_date.strftime("%Y%m%d"), BACKTEST_DAYS)
    st.write(f"真实回测区间：{min(trade_days)} ~ {max(trade_days)}")

    results = []

    bar = st.progress(0.0)
    for i, day in enumerate(trade_days):
        daily = pro.daily(trade_date=day)
        basic = pro.daily_basic(trade_date=day)

        if daily is None or daily.empty:
            continue

        df = daily.merge(basic, on="ts_code", how="left")
        df = df[df["close"] > 10]

        picks = []

        for row in df.itertuples():
            hist = get_qfq(
                row.ts_code,
                (datetime.strptime(day, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d"),
                day
            )
            ind = calc_indicators(hist)
            if not ind:
                continue

            if ind["close"] < ind["ma60"]:
                continue

            ret5 = future_return(row.ts_code, day, ind["close"], 5)

            picks.append({
                "trade_date": day,
                "ts_code": row.ts_code,
                "close": ind["close"],
                "macd": ind["macd"],
                "rsi": ind["rsi"],
                "Return_D5": ret5
            })

        if picks:
            res = pd.DataFrame(picks).sort_values("macd", ascending=False).head(TOP_K)
            results.append(res)

        bar.progress((i+1)/len(trade_days))

    bar.empty()

    if results:
        all_res = pd.concat(results)
        st.dataframe(all_res.head(50))
        st.metric(
            "D+5 平均收益",
            f"{all_res['Return_D5'].mean():.2f}%"
        )
    else:
        st.warning("无回测结果")
