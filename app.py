# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（稳定版 · 含评分 · 无回测）
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 页面设置
# =========================================================
st.set_page_config(page_title="选股王 · 稳定评分版", layout="wide")
st.title("选股王 · 10000 积分旗舰（稳定版）")
st.markdown("本版本取消回测，但保留 MACD/RSI/量价/评分系统，速度更快更稳定。")

# =========================================================
# 参数设置
# =========================================================
with st.sidebar:
    st.header("过滤参数")
    INITIAL_TOP_N = int(st.number_input("涨幅榜前 N", value=800, step=100))
    MIN_PRICE = float(st.number_input("最低价格", value=10.0))
    MAX_PRICE = float(st.number_input("最高价格", value=200.0))
    MIN_TURNOVER = float(st.number_input("最低换手率%", value=3.0))
    MIN_AMOUNT = float(st.number_input("最低成交额（元）", value=2e8))
    VOL_RATIO_MIN = float(st.number_input("成交量放大倍数 vol/ma5", value=1.3))
    RSI_MAX = float(st.number_input("RSI 上限", value=70))
    MACD_MIN = float(st.number_input("MACD 最低值", value=-0.2))

# =========================================================
# Token 输入
# =========================================================
TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN:
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# =========================================================
# 安全 API 调用
# =========================================================
def safe_get(func, **kwargs):
    try:
        df = func(**kwargs)
        return df if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

# =========================================================
# 找最近交易日
# =========================================================
@st.cache_data(ttl=600)
def get_last_trade():
    today = datetime.now().date()
    for i in range(15):
        d = (today - timedelta(days=i)).strftime("%Y%m%d")
        if not safe_get(pro.daily, trade_date=d).empty:
            return d
    return None

last_trade = get_last_trade()
st.info(f"最近交易日：{last_trade}")

# =========================================================
# 初筛：涨幅榜
# =========================================================
daily = safe_get(pro.daily, trade_date=last_trade)
daily = daily.sort_values("pct_chg", ascending=False).head(INITIAL_TOP_N)

# 基础数据
base = safe_get(pro.stock_basic, list_status="L", fields="ts_code,name,total_mv")
basic = safe_get(pro.daily_basic, trade_date=last_trade, fields="ts_code,turnover_rate,amount,total_mv")

pool = daily.merge(base, on="ts_code", how="left")
pool = pool.merge(basic, on="ts_code", how="left")

# =========================================================
# 清洗 + 指标计算 + 评分
# =========================================================
results = []
bar = st.progress(0)
total = len(pool)

for idx, row in pool.iterrows():
    ts_code = row["ts_code"]
    name = row["name"]

    # --- 基础过滤 ---
    close = row["close"]
    if close < MIN_PRICE or close > MAX_PRICE:
        bar.progress((idx+1)/total); continue

    if "ST" in name.upper() or "退" in name:
        bar.progress((idx+1)/total); continue

    turn = row["turnover_rate"]
    if turn < MIN_TURNOVER:
        bar.progress((idx+1)/total); continue

    amt = row["amount"]
    if amt < MIN_AMOUNT:
        bar.progress((idx+1)/total); continue

    mv = row["total_mv"]
    if mv > 2000e8:   # 避免大盘股
        bar.progress((idx+1)/total); continue

    # --- 拉历史数据（约 90 天） ---
    hist = safe_get(
        pro.daily,
        ts_code=ts_code,
        start_date=(datetime.now() - timedelta(days=180)).strftime("%Y%m%d"),
        end_date=last_trade
    )
    if len(hist) < 35:
        bar.progress((idx+1)/total); continue

    hist = hist.sort_values("trade_date")

    # ========== MACD ==========
    hist["ema12"] = hist["close"].ewm(span=12).mean()
    hist["ema26"] = hist["close"].ewm(span=26).mean()
    hist["macd"] = hist["ema12"] - hist["ema26"]
    macd = hist["macd"].iloc[-1]
    if macd < MACD_MIN:
        bar.progress((idx+1)/total); continue

    # ========== RSI ==========
    delta = hist["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    rsi_last = rsi.iloc[-1]
    if rsi_last > RSI_MAX:
        bar.progress((idx+1)/total); continue

    # ========== 成交量放大 ==========
    hist["vol_ma5"] = hist["vol"].rolling(5).mean()
    vol_ratio = hist["vol"].iloc[-1] / hist["vol_ma5"].iloc[-1]
    if vol_ratio < VOL_RATIO_MIN:
        bar.progress((idx+1)/total); continue

    # =========================================================
    # ⭐⭐⭐ 评分系统（0–100 分）⭐⭐⭐
    # =========================================================

    score = 0

    # 趋势评分（MACD 越大越好：0–40）
    score += min(max((macd + 0.5) * 40, 0), 40)

    # RSI 评分（越低越安全：0–25）
    if rsi_last < 30:
        score += 25
    elif rsi_last < 50:
        score += 18
    elif rsi_last < 60:
        score += 10
    else:
        score += 3

    # 成交量动能评分：0–20
    score += min(vol_ratio * 8, 20)

    # 换手评分：0–10
    score += min(turn / 2, 10)

    # 当日涨幅确认：0–5
    pct = row["pct_chg"]
    score += min(max(pct / 2, 0), 5)

    # 最终记录
    results.append({
        "ts_code": ts_code,
        "name": name,
        "close": close,
        "pct_chg": pct,
        "macd": round(macd, 3),
        "rsi": round(rsi_last, 2),
        "vol_ratio": round(vol_ratio, 2),
        "score": round(score, 1)
    })

    bar.progress((idx+1)/total)

# ========== 输出结果 ==========
df = pd.DataFrame(results)

st.subheader("最终结果（评分排序）")
st.dataframe(df.sort_values("score", ascending=False), use_container_width=True)
