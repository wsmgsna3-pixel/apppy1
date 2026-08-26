# -*- coding: utf-8 -*-
"""
周线 SKDJ 每日状态与三路径买点系统 V14.6
============================================================
核心约定
1. 保留原 V14.5 周线 SKDJ：周线 K 上穿 25 且 K>D 为正式信号。
2. 每日状态模式使用当周未完成周线，信号分为：预备上穿、正式上穿、回踩止跌。
3. 三条路径独立模拟，只用于比较买点；同一股票实盘不可重复买三次。
4. 普通 A 股按 T+1：买入日不允许卖出，次一交易日起止损有效。
5. 首周第 5 个交易日按收盘价近似 14:50 卖出；暂不计交易成本。
6. 历史回测只用每周最后一个交易日，避免未完成周线污染历史样本。
7. 所有持仓每次运行都重新计算，旧“持仓中”记录会随新行情更新。
8. 日常界面只提醒最近5个交易日的变化，但未结束持仓始终保留并继续更新。
============================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts

warnings.filterwarnings("ignore")

APP_VERSION = "V14.6"
MARKET_CACHE_FILE = "skdj_v14_6_market_data_master.pkl"

st.set_page_config(page_title=f"SKDJ {APP_VERSION} 每日状态系统", layout="wide")
st.title(f"🔬 周线 SKDJ 每日状态与三路径买点系统 ({APP_VERSION})")
st.markdown(
    "**每日状态**使用当周动态周线；**历史回测**只使用完成周线。"
    "三种买点为平行研究路径，不代表同一股票重复买入。"
)


# -----------------------------------------------------------------------------
# 通用工具
# -----------------------------------------------------------------------------
def clean_token_str(raw_token: str) -> str:
    if not raw_token:
        return ""
    return re.sub(r"[\s\u3000\ufeff\xa0\r\n]+", "", str(raw_token)).strip()


def verify_token_connection(token_str: str):
    if not token_str:
        return False, "Token 为空，请在侧边栏填入 Token。"
    try:
        ts.set_token(token_str)
        pro = ts.pro_api(token_str)
        end_d = datetime.now().strftime("%Y%m%d")
        start_d = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
        test_df = pro.trade_cal(exchange="SSE", start_date=start_d, end_date=end_d)
        if test_df is not None and not test_df.empty:
            return True, "验证通过"
        return False, "Token 校验未返回数据，请检查网络连接。"
    except Exception as exc:
        msg = str(exc)
        if "token不对" in msg or "-40001" in msg:
            return False, "Token 不正确，请检查复制内容。"
        return False, f"接口校验失败: {msg}"


def safe_tushare_call(func, max_retries=3, sleep_time=0.8, **kwargs):
    for attempt in range(max_retries):
        try:
            df = func(**kwargs)
            if df is not None and not df.empty:
                return df
            time.sleep(sleep_time)
        except Exception:
            time.sleep(sleep_time * (attempt + 1))
    return pd.DataFrame()


def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def atomic_write_csv(df: pd.DataFrame, path: str):
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, path)


def parse_yyyymmdd(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\.0$", "", str(value)).replace("-", "")
    return text if re.fullmatch(r"\d{8}", text) else None


def make_config_id(min_price, min_mv, max_mv, top_n, pullback_window=None):
    raw = f"{APP_VERSION}|p={min_price:.2f}|mv={min_mv:.2f}-{max_mv:.2f}|n={int(top_n)}"
    if pullback_window is not None:
        raw += f"|pullback={int(pullback_window)}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{APP_VERSION.replace('.', '_')}_{digest}"


def is_20cm_stock(ts_code: str) -> bool:
    return any(str(ts_code).startswith(prefix) for prefix in ("300", "301", "688", "689"))


def is_one_price_limit_down(row: pd.Series, ts_code: str) -> bool:
    pre_close = row.get("pre_close", np.nan)
    if pd.isna(pre_close) or pre_close <= 0:
        return False
    limit_rate = 0.195 if is_20cm_stock(ts_code) else 0.095
    one_price = np.isclose(row["open"], row["high"]) and np.isclose(row["high"], row["low"])
    return bool(one_price and (row["close"] - pre_close) / pre_close <= -limit_rate)


# -----------------------------------------------------------------------------
# 股票池与行情缓存
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600 * 24 * 7, show_spinner=False)
def load_custom_tech_whitelist(token):
    token_c = clean_token_str(token)
    if not token_c:
        return set(), {}

    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    stock_basic = safe_tushare_call(
        pro.stock_basic,
        list_status="L",
        fields="ts_code,symbol,name,industry,market,list_date",
    )
    if stock_basic.empty:
        return set(), {}

    valid = stock_basic[stock_basic["market"].isin(("主板", "创业板", "科创板"))].copy()
    valid = valid[~valid["name"].str.contains("ST|退", na=False)]
    valid = valid[~valid["ts_code"].str.startswith("92")]

    core_l1 = {"电子", "计算机", "通信", "国防军工"}
    extended_l1 = {"机械设备", "电力设备", "医药生物", "汽车", "基础化工", "有色金属", "建筑材料"}
    keywords = {
        "半导体", "电子元件", "元件", "光学光电子", "消费电子", "电子化学品",
        "计算机设备", "软件开发", "IT服务", "通信设备", "军工电子", "航空装备",
        "航天装备", "自动化设备", "机器人", "激光设备", "工控设备", "仪器仪表",
        "电池", "光伏设备", "风电设备", "电网设备", "电机", "医疗器械",
        "生物制品", "汽车电子", "金属新材料", "非金属材料", "膜材料", "碳纤维",
    }

    sw_indices = safe_tushare_call(pro.index_classify, level="L1", src="SW2021")
    target_sw = (
        sw_indices[sw_indices["industry_name"].isin(core_l1 | extended_l1)]
        if not sw_indices.empty
        else pd.DataFrame()
    )
    stock_sw_map = {}
    if not target_sw.empty:
        for _, row in target_sw.iterrows():
            member = safe_tushare_call(pro.index_member, index_code=row["index_code"], is_new="Y")
            if not member.empty:
                for code in member["con_code"]:
                    stock_sw_map[code] = row["industry_name"]
            time.sleep(0.03)

    whitelist = set()
    name_map = dict(zip(stock_basic["ts_code"], stock_basic["name"]))
    for _, row in valid.iterrows():
        code = row["ts_code"]
        basic_industry = str(row["industry"]) if pd.notna(row["industry"]) else ""
        sw_l1 = stock_sw_map.get(code, "")
        if sw_l1 in core_l1:
            whitelist.add(code)
            continue
        if sw_l1 in extended_l1:
            if (
                any(word in basic_industry for word in keywords)
                or basic_industry == ""
                or sw_l1 in {"机械设备", "电力设备", "医药生物"}
            ):
                whitelist.add(code)
                continue
        if any(word in basic_industry for word in keywords):
            whitelist.add(code)

    return whitelist, name_map


def sync_market_data_incrementally(start_date, end_date, token, whitelist_set):
    token_c = clean_token_str(token)
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    cal = safe_tushare_call(pro.trade_cal, exchange="SSE", start_date=start_date, end_date=end_date)
    if cal.empty:
        return {"daily": [], "adj": [], "daily_basic": [], "fetched_dates": set()}

    all_dates = cal[cal["is_open"] == 1].sort_values("cal_date")["cal_date"].astype(str).tolist()
    today_str = datetime.now().strftime("%Y%m%d")
    valid_dates = [d for d in all_dates if d <= today_str]

    cache = {"daily": [], "adj": [], "daily_basic": [], "fetched_dates": set()}
    if os.path.exists(MARKET_CACHE_FILE):
        try:
            with open(MARKET_CACHE_FILE, "rb") as file_obj:
                loaded = pickle.load(file_obj)
            if isinstance(loaded, dict):
                cache.update(loaded)
                cache.setdefault("daily_basic", [])
                cache.setdefault("fetched_dates", set())
        except Exception:
            pass

    missing = [d for d in valid_dates if d not in cache["fetched_dates"]]
    if missing:
        bar = st.progress(0, text=f"📥 同步 {len(missing)} 个交易日行情...")
        for idx, trade_date in enumerate(missing):
            daily = safe_tushare_call(pro.daily, trade_date=trade_date)
            adj = safe_tushare_call(pro.adj_factor, trade_date=trade_date)
            basic = safe_tushare_call(
                pro.daily_basic,
                trade_date=trade_date,
                fields="ts_code,trade_date,circ_mv",
            )
            if whitelist_set:
                if not daily.empty:
                    daily = daily[daily["ts_code"].isin(whitelist_set)]
                if not adj.empty:
                    adj = adj[adj["ts_code"].isin(whitelist_set)]
                if not basic.empty:
                    basic = basic[basic["ts_code"].isin(whitelist_set)]

            # 当天日线尚未发布时不标记成功，下次运行会继续补。
            if not daily.empty and not adj.empty:
                cache["daily"].append(daily)
                cache["adj"].append(adj)
                if not basic.empty:
                    cache["daily_basic"].append(basic)
                cache["fetched_dates"].add(trade_date)

            if (idx + 1) % 10 == 0 or idx == len(missing) - 1:
                bar.progress((idx + 1) / len(missing), text=f"📥 行情同步 {idx + 1}/{len(missing)}")
                try:
                    tmp = MARKET_CACHE_FILE + ".tmp"
                    with open(tmp, "wb") as file_obj:
                        pickle.dump(cache, file_obj)
                    os.replace(tmp, MARKET_CACHE_FILE)
                except Exception:
                    pass
            time.sleep(0.2)
        bar.empty()
    return cache


@st.cache_data(ttl=300, show_spinner=False)
def load_optimized_market_data(start_date, end_date, token, whitelist_keys, cache_stamp):
    del cache_stamp
    whitelist = set(whitelist_keys)
    cache = sync_market_data_incrementally(start_date, end_date, token, whitelist)
    daily_raw = pd.concat(cache.get("daily", []), ignore_index=True) if cache.get("daily") else pd.DataFrame()
    adj_raw = pd.concat(cache.get("adj", []), ignore_index=True) if cache.get("adj") else pd.DataFrame()
    basic_raw = (
        pd.concat(cache.get("daily_basic", []), ignore_index=True)
        if cache.get("daily_basic")
        else pd.DataFrame()
    )
    if daily_raw.empty or adj_raw.empty:
        return {}, pd.DataFrame()

    daily_raw = daily_raw[daily_raw["ts_code"].isin(whitelist)]
    adj_raw = adj_raw[adj_raw["ts_code"].isin(whitelist)]
    if not basic_raw.empty:
        basic_raw = basic_raw[basic_raw["ts_code"].isin(whitelist)]

    merged = daily_raw.merge(
        adj_raw[["ts_code", "trade_date", "adj_factor"]],
        on=["ts_code", "trade_date"],
        how="inner",
    )
    merged["trade_date_str"] = merged["trade_date"].astype(str)
    merged = merged.drop_duplicates(["ts_code", "trade_date_str"], keep="last")
    merged = merged.sort_values(["ts_code", "trade_date_str"])

    stock_dict = {}
    for code, group in merged.groupby("ts_code"):
        frame = group.copy()
        for col in ("open", "high", "low", "close", "pre_close"):
            if col in frame.columns:
                frame[f"raw_{col}"] = frame[col]
        latest_adj = frame["adj_factor"].iloc[-1]
        if pd.notna(latest_adj) and latest_adj > 0:
            for col in ("open", "high", "low", "close", "pre_close"):
                if col in frame.columns:
                    frame[col] = frame[col] * frame["adj_factor"] / latest_adj
        frame = frame.set_index("trade_date_str").sort_index()
        stock_dict[code] = frame

    if not basic_raw.empty:
        basic_raw["trade_date"] = basic_raw["trade_date"].astype(str)
        basic_indexed = (
            basic_raw.drop_duplicates(["trade_date", "ts_code"], keep="last")
            .set_index(["trade_date", "ts_code"])
            .sort_index()
        )
    else:
        basic_indexed = pd.DataFrame()
    return stock_dict, basic_indexed


def get_circ_mv_billion(basic_indexed, trade_date, ts_code):
    if basic_indexed.empty:
        return np.nan
    key = (str(trade_date), str(ts_code))
    if key not in basic_indexed.index:
        return np.nan
    value = basic_indexed.loc[key, "circ_mv"]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return float(value) / 10000.0 if pd.notna(value) else np.nan


# -----------------------------------------------------------------------------
# 周线指标、三类信号与排序
# -----------------------------------------------------------------------------
def build_weekly_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    frame = df_daily.copy().reset_index()
    frame["dt"] = pd.to_datetime(frame["trade_date_str"])
    frame["year_week"] = frame["dt"].dt.strftime("%G_%V")
    weekly = (
        frame.groupby("year_week", as_index=False)
        .agg(
            trade_date_str=("trade_date_str", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            vol=("vol", "sum"),
        )
        .sort_values("trade_date_str")
        .reset_index(drop=True)
    )
    weekly["lowv"] = weekly["low"].rolling(6).min()
    weekly["highv"] = weekly["high"].rolling(6).max()
    diff = (weekly["highv"] - weekly["lowv"]).replace(0, 0.001)
    weekly["rsv"] = ((weekly["close"] - weekly["lowv"]) / diff * 100).ewm(span=3, adjust=False).mean()
    weekly["k"] = weekly["rsv"].ewm(span=3, adjust=False).mean()
    weekly["d"] = weekly["k"].rolling(3).mean()
    weekly["ma5_vol"] = weekly["vol"].shift(1).rolling(5).mean()
    weekly["ma20"] = weekly["close"].rolling(20).mean()
    return weekly


def compute_signal_snapshot(ts_code, end_date, stock_dict):
    if ts_code not in stock_dict:
        return {}
    full = stock_dict[ts_code]
    daily = full[full.index <= str(end_date)].copy()
    if len(daily) < 100:
        return {}

    last = daily.iloc[-1]
    pre_close = last.get("pre_close", np.nan)
    if pd.isna(pre_close) or pre_close <= 0:
        pre_close = daily.iloc[-2]["close"]
    limit_rate = 0.195 if is_20cm_stock(ts_code) else 0.095
    one_price_up = (
        np.isclose(last["high"], last["low"])
        and (last["close"] - pre_close) / pre_close >= limit_rate
    )
    if one_price_up:
        return {}

    weekly = build_weekly_frame(daily)
    if len(weekly) < 21:
        return {}
    curr, prev = weekly.iloc[-1], weekly.iloc[-2]
    if pd.isna(curr["k"]) or pd.isna(curr["d"]) or pd.isna(prev["k"]) or pd.isna(prev["d"]):
        return {}

    prior = weekly.tail(15).iloc[:-1]
    recent_k_min = float(prior["k"].min())
    weeks_under = int((prior["k"] < 25.0).sum())
    ma20 = curr["ma20"] if pd.notna(curr["ma20"]) else curr["close"]
    vol_ratio = curr["vol"] / curr["ma5_vol"] if pd.notna(curr["ma5_vol"]) and curr["ma5_vol"] > 0 else 1.0

    components = {}
    components["Trend_Score"] = 20.0 if curr["close"] >= ma20 else -5.0
    if 22.0 <= recent_k_min <= 25.0:
        components["KMin_Score"] = 30.0
    elif 15.0 <= recent_k_min < 22.0:
        components["KMin_Score"] = 15.0
    elif 5.0 <= recent_k_min < 15.0:
        components["KMin_Score"] = -10.0
    else:
        components["KMin_Score"] = -25.0

    if 1 <= weeks_under <= 2:
        components["Under25_Score"] = 30.0
    elif 3 <= weeks_under <= 5:
        components["Under25_Score"] = 15.0
    elif 6 <= weeks_under <= 9:
        components["Under25_Score"] = -5.0
    else:
        components["Under25_Score"] = -20.0

    if 25.0 < curr["k"] <= 32.0:
        components["CurrentK_Score"] = 10.0
    elif curr["k"] > 38.0:
        components["CurrentK_Score"] = -10.0
    else:
        components["CurrentK_Score"] = 0.0

    if 1.0 <= vol_ratio <= 2.5:
        components["Volume_Score"] = 10.0
    elif vol_ratio > 4.0:
        components["Volume_Score"] = -15.0
    else:
        components["Volume_Score"] = 0.0

    k_gap = float(curr["k"] - curr["d"])
    prev_gap = float(prev["k"] - prev["d"])
    is_breakout = bool(curr["k"] > 25.0 and prev["k"] <= 25.0 and curr["k"] > curr["d"])
    is_watch = bool(
        22.0 <= curr["k"] <= 25.0
        and curr["k"] > prev["k"]
        and (curr["k"] > curr["d"] or k_gap > prev_gap)
    )

    closes = daily["close"]
    highs = daily["high"]
    pre_ret = {}
    for days in (5, 10, 15):
        pre_ret[f"Pre_Return_{days}D"] = (
            (closes.iloc[-1] / closes.iloc[-days - 1] - 1) * 100 if len(closes) > days else np.nan
        )
    tail20 = daily.tail(20)
    tail10 = daily.tail(10)
    up_days_10 = int((closes.diff().tail(10) > 0).sum())
    low_20_position = int(np.argmin(tail20["low"].to_numpy()))
    days_since_20d_low = len(tail20) - 1 - low_20_position
    daily_vol_base = daily["vol"].shift(1).tail(5).mean()
    daily_vol_ratio = last["vol"] / daily_vol_base if pd.notna(daily_vol_base) and daily_vol_base > 0 else np.nan
    rise_from_20d_low = (closes.iloc[-1] / tail20["low"].min() - 1) * 100
    last_3d_return = (closes.iloc[-1] / closes.iloc[-4] - 1) * 100 if len(closes) >= 4 else np.nan
    timing_flags = []
    if days_since_20d_low >= 8 and rise_from_20d_low >= 8.0:
        timing_flags.append(f"上涨已延续{days_since_20d_low}日")
    if pd.notna(pre_ret["Pre_Return_10D"]) and pre_ret["Pre_Return_10D"] >= 10.0:
        timing_flags.append("10日涨幅偏大")
    if pd.notna(last_3d_return) and last_3d_return < 0:
        timing_flags.append("近3日转弱")

    result = {
        "Signal_Date": str(end_date),
        "Setup_Week": pd.to_datetime(str(end_date)).strftime("%G_%V"),
        "is_watch": is_watch,
        "is_breakout": is_breakout,
        "SKDJ_K": round(float(curr["k"]), 2),
        "SKDJ_D": round(float(curr["d"]), 2),
        "Prev_K": round(float(prev["k"]), 2),
        "K_Slope": round(float(curr["k"] - prev["k"]), 2),
        "KD_Gap_Improve": round(k_gap - prev_gap, 2),
        "K_Min_14W": round(recent_k_min, 2),
        "Weeks_Under_25_14W": weeks_under,
        "Trend_Type": "均线上方" if curr["close"] >= ma20 else "均线下方(超跌)",
        "Weekly_Vol_Ratio": round(float(vol_ratio), 2),
        "Signal_Close": float(curr["close"]),
        "Signal_Raw_Close": float(last.get("raw_close", last["close"])),
        "Total_Score": round(sum(components.values()), 1),
        "Rise_From_20D_Low": round(rise_from_20d_low, 2),
        "Days_Since_20D_Low": days_since_20d_low,
        "Below_10D_High": round((closes.iloc[-1] / tail10["high"].max() - 1) * 100, 2),
        "Below_20D_High": round((closes.iloc[-1] / tail20["high"].max() - 1) * 100, 2),
        "Up_Days_10D": up_days_10,
        "Last_3D_Return": round(last_3d_return, 2) if pd.notna(last_3d_return) else np.nan,
        "Timing_Flag": "；".join(timing_flags) if timing_flags else "正常",
        "Daily_Vol_Ratio": round(float(daily_vol_ratio), 2) if pd.notna(daily_vol_ratio) else np.nan,
    }
    result.update({k: round(v, 1) for k, v in components.items()})
    result.update({k: round(float(v), 2) if pd.notna(v) else np.nan for k, v in pre_ret.items()})
    return result


def deterministic_rank(records):
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).copy()
    frame["_trend"] = (frame["Trend_Type"] == "均线上方").astype(int)
    frame["_kmin_dist"] = (frame["K_Min_14W"] - 23.5).abs()
    frame["_k_dist"] = (frame["SKDJ_K"] - 28.5).abs()
    frame["_vol_dist"] = (frame["Weekly_Vol_Ratio"] - 1.5).abs()
    frame = frame.sort_values(
        ["Total_Score", "_trend", "_kmin_dist", "_k_dist", "Weeks_Under_25_14W", "_vol_dist", "ts_code"],
        ascending=[False, False, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["Raw_Rank"] = np.arange(1, len(frame) + 1)
    return frame.drop(columns=["_trend", "_kmin_dist", "_k_dist", "_vol_dist"])


def build_candidate_records(date, signal_kind, whitelist_keys, name_map, stock_dict, basic_indexed,
                            min_price, min_mv, max_mv):
    records = []
    for code in whitelist_keys:
        if code not in stock_dict or str(date) not in stock_dict[code].index:
            continue
        row = stock_dict[code].loc[str(date)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        raw_close = float(row.get("raw_close", row["close"]))
        if raw_close < min_price:
            continue
        mv = get_circ_mv_billion(basic_indexed, date, code)
        if pd.notna(mv) and (mv < min_mv or mv > max_mv):
            continue
        snap = compute_signal_snapshot(code, date, stock_dict)
        if not snap:
            continue
        if signal_kind == "预备上穿" and not snap["is_watch"]:
            continue
        if signal_kind == "正式上穿25" and not snap["is_breakout"]:
            continue
        snap.update(
            {
                "ts_code": code,
                "name": name_map.get(code, code),
                "circ_mv": round(mv, 2) if pd.notna(mv) else np.nan,
                "MV_Data_Missing": bool(pd.isna(mv)),
                "Signal_Type": signal_kind,
            }
        )
        records.append(snap)
    return deterministic_rank(records)


# -----------------------------------------------------------------------------
# 买入检查、T+1 生命周期与 W8/W10/W12 并行结果
# -----------------------------------------------------------------------------
def entry_check(ts_code, signal_date, signal_close, stock_dict):
    default = {
        "Entry_Status": "WAIT_BUY",
        "Buy_Date": None,
        "Buy_Price": np.nan,
        "Gap_pct": np.nan,
        "Entry_Reason": "等待下一交易日开盘",
    }
    if ts_code not in stock_dict:
        return default
    future = stock_dict[ts_code][stock_dict[ts_code].index > str(signal_date)]
    if future.empty:
        return default
    row = future.iloc[0]
    buy_price = float(row["open"])
    gap = (buy_price - float(signal_close)) / float(signal_close) * 100
    result = {
        "Entry_Status": "READY",
        "Buy_Date": future.index[0],
        "Buy_Price": round(buy_price, 4),
        "Gap_pct": round(gap, 2),
        "Entry_Reason": "下一交易日开盘买入",
    }
    is20 = is_20cm_stock(ts_code)
    limit_pct = 19.0 if is20 else 9.5
    one_price_up = (
        np.isclose(row["open"], row["high"])
        and np.isclose(row["high"], row["low"])
        and gap >= limit_pct
    )
    if one_price_up:
        result.update(Entry_Status="SKIPPED", Entry_Reason=f"一字板无法买入({gap:.1f}%)")
    elif is20 and gap > 8.0:
        result.update(Entry_Status="SKIPPED", Entry_Reason=f"双创高开过大({gap:.2f}%)")
    elif (not is20) and gap > 5.0:
        result.update(Entry_Status="SKIPPED", Entry_Reason=f"主板高开过大({gap:.2f}%)")
    elif gap < -4.0:
        result.update(Entry_Status="SKIPPED", Entry_Reason=f"恶劣低开({gap:.2f}%)")
    return result


def _exit_payload(reason, date, price, buy_price, day_count):
    ret = (price - buy_price) / buy_price * 100
    return {
        "Status": "CLOSED",
        "Exit_Reason": reason,
        "Exit_Date": date,
        "Exit_Price": round(float(price), 4),
        "Final_Return (%)": round(float(ret), 2),
        "Hold_Days": int(day_count),
    }


def simulate_lifecycle(event, stock_dict, asof_date, max_weeks=12):
    ts_code = str(event["ts_code"])
    signal_date = parse_yyyymmdd(event.get("Signal_Date"))
    signal_close = float(event.get("Signal_Close", np.nan))
    result = {f"Return_W{w} (%)": np.nan for w in range(1, 13)}
    result.update(
        {
            "Status": "WAIT_BUY",
            "Buy_Date": None,
            "Buy_Price": np.nan,
            "Gap_pct": np.nan,
            "Entry_Reason": "等待下一交易日开盘",
            "Exit_Reason": "等待买入",
            "Exit_Date": None,
            "Exit_Price": np.nan,
            "Final_Return (%)": np.nan,
            "Hold_Days": 0,
            "Followup_Days": 0,
            "Current_Return (%)": np.nan,
            "Peak_Return (%)": np.nan,
            "Drawdown_From_Peak (%)": np.nan,
            "Stop_Stage": "未买入",
            "Stop_Level": np.nan,
            "Current_Week": 0,
            "Horizon_8W_Return (%)": np.nan,
            "Horizon_10W_Return (%)": np.nan,
            "Horizon_12W_Return (%)": np.nan,
        }
    )
    if not signal_date or ts_code not in stock_dict or not np.isfinite(signal_close):
        return result

    full = stock_dict[ts_code]
    future_all = full[(full.index > signal_date) & (full.index <= str(asof_date))]
    result["Followup_Days"] = int(len(future_all))
    entry = entry_check(ts_code, signal_date, signal_close, stock_dict)
    result.update(entry)
    if entry["Entry_Status"] == "WAIT_BUY":
        return result
    if entry["Entry_Status"] == "SKIPPED":
        result.update(Status="SKIPPED", Exit_Reason=entry["Entry_Reason"])
        return result

    buy_date = str(entry["Buy_Date"])
    buy_price = float(entry["Buy_Price"])
    future = full[(full.index >= buy_date) & (full.index <= str(asof_date))]
    if future.empty:
        return result

    result.update(Status="HOLDING", Exit_Reason="持仓中", Stop_Stage="基础-10%止损")
    peak = buy_price
    tier = 0
    exit_data = None
    max_days = int(max_weeks) * 5
    forced_exit_pending_reason = None

    for idx in range(len(future)):
        row = future.iloc[idx]
        date = future.index[idx]
        day_count = idx + 1
        open_p, high_p, low_p, close_p = map(float, (row["open"], row["high"], row["low"], row["close"]))

        one_price_down_today = is_one_price_limit_down(row, ts_code)

        # 一字跌停无法成交时保留退出指令，顺延到首次可成交日。
        if forced_exit_pending_reason is not None:
            if is_one_price_limit_down(row, ts_code):
                continue
            exit_data = _exit_payload(
                f"{forced_exit_pending_reason}(流动性顺延)", date, open_p, buy_price, day_count
            )
            result[f"Return_W{min(12, (day_count - 1) // 5 + 1)} (%)"] = exit_data["Final_Return (%)"]
            break

        # T+1：买入日不允许卖出；从第二个交易日起启用所有保护价。
        if day_count >= 2:
            stop_level = buy_price * 0.90
            stop_reason = "认栽出局(破-10%)"
            if tier >= 1 and buy_price * 1.02 > stop_level:
                stop_level = buy_price * 1.02
                stop_reason = "盈利保护(+2%线)"
            if tier >= 2 and peak * 0.85 > stop_level:
                stop_level = peak * 0.85
                stop_reason = "移动止盈(距峰值15%)"

            if open_p <= stop_level or low_p <= stop_level:
                if one_price_down_today:
                    forced_exit_pending_reason = stop_reason
                    if day_count % 5 == 0 and day_count <= 60:
                        result[f"Return_W{day_count // 5} (%)"] = round(
                            (close_p - buy_price) / buy_price * 100, 2
                        )
                    continue
                exit_price = open_p if open_p <= stop_level else stop_level
                exit_data = _exit_payload(stop_reason, date, exit_price, buy_price, day_count)
                result[f"Return_W{min(12, (day_count - 1) // 5 + 1)} (%)"] = exit_data["Final_Return (%)"]
                break

        # 当天创新高后，新的保护层级从下一交易日生效，避免日线高低价先后顺序偏差。
        peak = max(peak, high_p)
        peak_profit = (peak - buy_price) / buy_price
        if peak_profit >= 0.20:
            tier = 2
        elif peak_profit >= 0.10:
            tier = max(tier, 1)

        # 用户确认：首周第5个交易日按收盘价近似14:50执行。
        if day_count == 5:
            w1_ret = (close_p - buy_price) / buy_price * 100
            if w1_ret <= -3.0:
                reason = f"首周不及预期截断({w1_ret:.1f}%)"
                if one_price_down_today:
                    forced_exit_pending_reason = reason
                else:
                    exit_data = _exit_payload(reason, date, close_p, buy_price, day_count)
                    result["Return_W1 (%)"] = exit_data["Final_Return (%)"]
                    break

        if day_count % 5 == 0 and day_count <= 60:
            week_no = day_count // 5
            result[f"Return_W{week_no} (%)"] = round((close_p - buy_price) / buy_price * 100, 2)

        if day_count == max_days:
            if one_price_down_today:
                forced_exit_pending_reason = "12周期到期"
            else:
                exit_data = _exit_payload("12周期满平仓", date, close_p, buy_price, day_count)
                result["Return_W12 (%)"] = exit_data["Final_Return (%)"]
                break

    observed_days = min(len(future), max_days if forced_exit_pending_reason is None else len(future))
    result["Hold_Days"] = int(observed_days)

    # 相同交易路径下的8/10/12周上限反事实结果。
    for weeks in (8, 10, 12):
        horizon_days = weeks * 5
        col = f"Horizon_{weeks}W_Return (%)"
        if exit_data is not None and exit_data["Hold_Days"] <= horizon_days:
            result[col] = exit_data["Final_Return (%)"]
        elif len(future) >= horizon_days:
            price = float(future.iloc[horizon_days - 1]["close"])
            result[col] = round((price - buy_price) / buy_price * 100, 2)

    if exit_data is not None:
        result.update(exit_data)
        final_peak_return = (peak - buy_price) / buy_price * 100
        result["Peak_Return (%)"] = round(final_peak_return, 2)
        result["Drawdown_From_Peak (%)"] = round((peak - float(exit_data["Exit_Price"])) / peak * 100, 2)
        result["Current_Return (%)"] = result["Final_Return (%)"]
        result["Current_Week"] = min(12, (int(result["Hold_Days"]) - 1) // 5 + 1)
        result["Stop_Stage"] = "已退出"
        result["Stop_Level"] = np.nan
        return result

    last_row = future.iloc[-1]
    current_close = float(last_row["close"])
    current_ret = (current_close - buy_price) / buy_price * 100
    peak_ret = (peak - buy_price) / buy_price * 100
    if tier >= 2:
        stage = "+20%后移动止盈"
        level = max(buy_price * 1.02, peak * 0.85)
    elif tier >= 1:
        stage = "+10%后盈利保护"
        level = buy_price * 1.02
    else:
        stage = "T+1锁定" if len(future) == 1 else "基础-10%止损"
        level = buy_price * 0.90
    if forced_exit_pending_reason is not None:
        stage = "一字跌停排队卖出"
    result.update(
        {
            "Status": "HOLDING",
            "Exit_Reason": (
                f"{forced_exit_pending_reason}(等待可成交)"
                if forced_exit_pending_reason is not None
                else "持仓中"
            ),
            "Hold_Days": int(len(future)),
            "Current_Return (%)": round(current_ret, 2),
            "Peak_Return (%)": round(peak_ret, 2),
            "Drawdown_From_Peak (%)": round((peak - current_close) / peak * 100, 2),
            "Stop_Stage": stage,
            "Stop_Level": round(level, 4),
            "Current_Week": min(12, (len(future) - 1) // 5 + 1),
        }
    )
    return result


# -----------------------------------------------------------------------------
# 每日状态：事件登记、回踩识别、所有旧持仓刷新
# -----------------------------------------------------------------------------
SIGNAL_BASE_COLUMNS = [
    "Event_ID", "Signal_Date", "Setup_Week", "Signal_Type", "ts_code", "name",
    "Raw_Rank", "Rank", "Signal_Close", "Signal_Raw_Close", "Total_Score",
    "SKDJ_K", "SKDJ_D", "Prev_K", "K_Slope", "KD_Gap_Improve", "K_Min_14W",
    "Weeks_Under_25_14W", "Trend_Type", "Weekly_Vol_Ratio", "circ_mv",
    "Pre_Return_5D", "Pre_Return_10D", "Pre_Return_15D", "Rise_From_20D_Low",
    "Days_Since_20D_Low", "Timing_Flag",
    "Below_10D_High", "Below_20D_High", "Up_Days_10D", "Last_3D_Return",
    "Daily_Vol_Ratio", "Trend_Score", "KMin_Score", "Under25_Score",
    "CurrentK_Score", "Volume_Score", "Parent_Event_ID", "Signal_Note", "Config_ID",
]


def event_id_for(row, signal_type, parent_event_id=""):
    setup = parent_event_id or row.get("Setup_Week", row.get("Signal_Date", ""))
    raw = f"{row['ts_code']}|{signal_type}|{setup}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def candidates_to_events(frame, signal_type, config_id, parent_col=None):
    if frame.empty:
        return pd.DataFrame(columns=SIGNAL_BASE_COLUMNS)
    events = frame.copy()
    events["Signal_Type"] = signal_type
    events["Rank"] = np.arange(1, len(events) + 1)
    if "Parent_Event_ID" not in events.columns:
        events["Parent_Event_ID"] = ""
    events["Event_ID"] = events.apply(
        lambda row: event_id_for(row, signal_type, str(row.get(parent_col, "")) if parent_col else ""),
        axis=1,
    )
    events["Signal_Note"] = events.get("Signal_Note", "首次状态变化")
    events["Config_ID"] = config_id
    for col in SIGNAL_BASE_COLUMNS:
        if col not in events.columns:
            events[col] = np.nan if col not in {"Parent_Event_ID", "Signal_Note"} else ""
    return events[SIGNAL_BASE_COLUMNS]


def merge_new_events(existing, additions):
    frames = [frame for frame in (existing, *additions) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=SIGNAL_BASE_COLUMNS)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["Signal_Date"] = merged["Signal_Date"].map(parse_yyyymmdd)
    merged["Event_ID"] = merged["Event_ID"].astype(str)
    merged["Signal_Type"] = merged["Signal_Type"].astype(str)
    merged["Rank"] = pd.to_numeric(merged["Rank"], errors="coerce")
    merged = merged.drop_duplicates("Event_ID", keep="first")
    return merged.sort_values(["Signal_Date", "Signal_Type", "Rank"]).reset_index(drop=True)


def build_recent_formal_sources(dates, whitelist_keys, name_map, stock_dict, basic_indexed,
                                min_price, min_mv, max_mv, top_n, config_id):
    """回补最近若干交易日的Top N正式突破，仅作为首次运行时识别回踩的父信号。"""
    source_events = []
    for date in dates:
        candidates = build_candidate_records(
            date, "正式上穿25", whitelist_keys, name_map, stock_dict, basic_indexed,
            min_price, min_mv, max_mv,
        ).head(int(top_n))
        if candidates.empty:
            continue
        events = candidates_to_events(candidates, "正式上穿25", config_id)
        events["Signal_Note"] = "最近交易日自动回补，仅用于识别回踩来源"
        source_events.append(events)
    return merge_new_events(pd.DataFrame(), source_events)


def build_pullback_candidates(signal_history, current_date, stock_dict, name_map, pullback_window):
    if signal_history.empty:
        return pd.DataFrame()
    formal = signal_history[signal_history["Signal_Type"] == "正式上穿25"].copy()
    if formal.empty:
        return pd.DataFrame()
    existing_parents = set(
        signal_history.loc[signal_history["Signal_Type"] == "回踩止跌", "Parent_Event_ID"].dropna().astype(str)
    )
    records = []
    for _, event in formal.iterrows():
        parent_id = str(event["Event_ID"])
        if parent_id in existing_parents:
            continue
        code = str(event["ts_code"])
        signal_date = parse_yyyymmdd(event["Signal_Date"])
        if not signal_date or code not in stock_dict:
            continue
        frame = stock_dict[code]
        post = frame[(frame.index > signal_date) & (frame.index <= str(current_date))]
        if len(post) < 2 or len(post) > int(pullback_window):
            continue
        snap = compute_signal_snapshot(code, current_date, stock_dict)
        if not snap or snap["SKDJ_K"] < 25.0 or snap["SKDJ_K"] <= snap["SKDJ_D"]:
            continue
        high_since = float(post["high"].max())
        current = post.iloc[-1]
        previous = post.iloc[-2]
        drawdown = (high_since - float(current["close"])) / high_since * 100
        prior_vol = post["vol"].shift(1).tail(5).mean()
        pullback_vol_ratio = float(current["vol"] / prior_vol) if pd.notna(prior_vol) and prior_vol > 0 else np.nan
        stopped_falling = float(current["close"]) >= float(previous["close"])
        shrinking = pd.isna(pullback_vol_ratio) or pullback_vol_ratio <= 1.0
        if not (3.0 <= drawdown <= 10.0 and stopped_falling and shrinking):
            continue
        snap.update(
            {
                "ts_code": code,
                "name": name_map.get(code, event.get("name", code)),
                "circ_mv": event.get("circ_mv", np.nan),
                "Signal_Type": "回踩止跌",
                "Signal_Close": float(current["close"]),
                "Signal_Raw_Close": float(current.get("raw_close", current["close"])),
                "Signal_Date": str(current_date),
                "Setup_Week": pd.to_datetime(str(current_date)).strftime("%G_%V"),
                "Parent_Event_ID": parent_id,
                "Pullback_Drawdown (%)": round(drawdown, 2),
                "Pullback_Days": int(len(post)),
                "Signal_Note": f"突破后第{len(post)}日，距峰值回撤{drawdown:.1f}%",
            }
        )
        records.append(snap)
    return deterministic_rank(records)


def refresh_all_positions(signal_history, stock_dict, asof_date):
    if signal_history.empty:
        return pd.DataFrame()
    rows = []
    progress = st.progress(0, text="🔄 更新所有历史信号与持仓状态...")
    for idx, (_, event) in enumerate(signal_history.iterrows()):
        lifecycle = simulate_lifecycle(event, stock_dict, asof_date, max_weeks=12)
        combined = event.to_dict()
        combined.update(lifecycle)
        rows.append(combined)
        if (idx + 1) % 20 == 0 or idx == len(signal_history) - 1:
            progress.progress((idx + 1) / len(signal_history), text=f"🔄 状态更新 {idx + 1}/{len(signal_history)}")
    progress.empty()
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 报告：正确区分本周有记录、周初存活、周末存活与固定队列收益
# -----------------------------------------------------------------------------
def display_lifecycle_report(position_df, title, recent_trade_dates=None):
    st.header(title)
    if position_df.empty:
        st.info("暂无可分析记录。")
        return

    valid = position_df[position_df["Status"] != "SKIPPED"].copy()
    completed = valid[valid["Status"] == "CLOSED"].copy()
    if "Final_Return (%)" in completed.columns:
        completed["Final_Return (%)"] = pd.to_numeric(completed["Final_Return (%)"], errors="coerce")
        completed = completed.dropna(subset=["Final_Return (%)"])

    cols = st.columns(5)
    cols[0].metric("有效信号", len(valid))
    cols[1].metric("当前持仓", int((valid["Status"] == "HOLDING").sum()))
    cols[2].metric("等待买入", int((valid["Status"] == "WAIT_BUY").sum()))
    cols[3].metric("已结束", len(completed))
    if not completed.empty:
        cols[4].metric("已结束胜率", f"{(completed['Final_Return (%)'] > 0).mean() * 100:.1f}%")

    if not completed.empty:
        mean_ret = completed["Final_Return (%)"].mean()
        median_ret = completed["Final_Return (%)"].median()
        st.caption(f"已结束样本：平均收益 {mean_ret:.2f}%｜中位数 {median_ret:.2f}%｜暂不计交易成本")

    st.subheader("🗓️ 条件表现与真实存活率")
    survival_rows = []
    for week in range(1, 13):
        w_col = f"Return_W{week} (%)"
        w_values = pd.to_numeric(valid.get(w_col, pd.Series(dtype=float)), errors="coerce").dropna()
        mature = valid[pd.to_numeric(valid["Followup_Days"], errors="coerce").fillna(0) >= week * 5]
        start_alive = mature[pd.to_numeric(mature["Hold_Days"], errors="coerce").fillna(0) > (week - 1) * 5]
        mature_days = pd.to_numeric(mature["Hold_Days"], errors="coerce").fillna(0)
        end_alive = mature[
            (mature_days > week * 5)
            | ((mature["Status"] == "HOLDING") & (mature_days >= week * 5))
        ]
        survival_rows.append(
            {
                "周期": f"W{week}",
                "本周有记录": len(w_values),
                "本周条件均益": w_values.mean() if len(w_values) else np.nan,
                "本周条件胜率": (w_values > 0).mean() * 100 if len(w_values) else np.nan,
                "成熟队列": len(mature),
                "周初仍持有": len(start_alive),
                "周末仍持有": len(end_alive),
                "累计周末存活率": len(end_alive) / len(mature) * 100 if len(mature) else np.nan,
            }
        )
    survival = pd.DataFrame(survival_rows)
    st.dataframe(
        survival.style.format(
            {
                "本周条件均益": "{:.2f}%",
                "本周条件胜率": "{:.1f}%",
                "累计周末存活率": "{:.1f}%",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.subheader("⏳ 相同交易路径的最大持有周期比较")
    horizon_rows = []
    for weeks in (8, 10, 12):
        col = f"Horizon_{weeks}W_Return (%)"
        values = pd.to_numeric(valid.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        horizon_rows.append(
            {
                "最大持有": f"W{weeks}",
                "成熟样本": len(values),
                "平均收益": values.mean() if len(values) else np.nan,
                "中位数": values.median() if len(values) else np.nan,
                "胜率": (values > 0).mean() * 100 if len(values) else np.nan,
            }
        )
    horizon = pd.DataFrame(horizon_rows)
    st.dataframe(
        horizon.style.format({"平均收益": "{:.2f}%", "中位数": "{:.2f}%", "胜率": "{:.1f}%"}, na_rep="—"),
        width="stretch",
    )

    if not completed.empty:
        st.subheader("🏆 排名与买点路径")
        rank_stats = (
            completed.groupby(["Signal_Type", "Rank"], observed=False)
            .agg(
                样本数=("Final_Return (%)", "count"),
                胜率=("Final_Return (%)", lambda x: (x > 0).mean() * 100),
                均益=("Final_Return (%)", "mean"),
                中位数=("Final_Return (%)", "median"),
                平均分=("Total_Score", "mean"),
            )
            .reset_index()
        )
        st.dataframe(
            rank_stats.style.format({"胜率": "{:.1f}%", "均益": "{:.2f}%", "中位数": "{:.2f}%", "平均分": "{:.1f}"}),
            width="stretch",
        )

        ordered = completed.sort_values("Final_Return (%)", ascending=False)
        sensitivity = []
        for remove_n in (0, 1, 3, 5, 10):
            sample = ordered.iloc[remove_n:] if remove_n else ordered
            sensitivity.append(
                {
                    "剔除最高收益": remove_n,
                    "剩余样本": len(sample),
                    "平均收益": sample["Final_Return (%)"].mean(),
                    "中位数": sample["Final_Return (%)"].median(),
                    "胜率": (sample["Final_Return (%)"] > 0).mean() * 100,
                }
            )
        st.subheader("🧪 大牛股依赖敏感性")
        st.dataframe(
            pd.DataFrame(sensitivity).style.format({"平均收益": "{:.2f}%", "中位数": "{:.2f}%", "胜率": "{:.1f}%"}),
            width="stretch",
        )

    detail = valid.copy()
    if recent_trade_dates:
        recent_set = {str(date) for date in recent_trade_dates}
        active_mask = detail["Status"].isin(["WAIT_BUY", "HOLDING"])
        recent_mask = pd.Series(False, index=detail.index)
        for date_col in ("Signal_Date", "Buy_Date", "Exit_Date"):
            if date_col in detail.columns:
                recent_mask |= detail[date_col].map(parse_yyyymmdd).isin(recent_set)
        detail = detail[active_mask | recent_mask].copy()
        st.subheader("📋 最近5个交易日状态提醒")
        st.caption("只显示最近5个交易日发生变化的记录；尚在持仓、待买入或排队卖出的股票不受5日限制。")
    else:
        st.subheader("📋 最新状态")
    display_cols = [
        "Signal_Date", "Signal_Type", "Rank", "name", "ts_code", "Total_Score",
        "Pre_Return_10D", "Days_Since_20D_Low", "Timing_Flag", "Status",
        "Buy_Date", "Buy_Price", "Gap_pct", "Hold_Days",
        "Current_Week", "Current_Return (%)", "Peak_Return (%)", "Drawdown_From_Peak (%)",
        "Stop_Stage", "Stop_Level", "Exit_Date", "Exit_Reason", "Final_Return (%)",
    ]
    display_cols = [col for col in display_cols if col in detail.columns]
    if detail.empty:
        st.info("最近5个交易日没有新状态，且当前没有未结束持仓。")
    else:
        st.dataframe(
            detail[display_cols].sort_values(
                ["Signal_Date", "Signal_Type", "Rank"], ascending=[False, True, True]
            ),
            width="stretch",
        )


# -----------------------------------------------------------------------------
# 侧边栏
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 运行配置")
    RUN_MODE = st.radio("运行模式", ["每日状态更新", "历史周末回测"], index=0)
    end_date = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("历史回测交易日数", value=250, min_value=30, step=30)
    MAX_TOP_N = st.number_input("每类信号最多跟踪数", value=5, min_value=1, max_value=20, step=1)
    PULLBACK_WINDOW = st.number_input("突破后等待回踩天数", value=10, min_value=3, max_value=20, step=1)

    st.markdown("---")
    st.subheader("💰 股票池")
    MIN_PRICE = st.number_input("最低股价(元)", value=10.0, min_value=0.0)
    c1, c2 = st.columns(2)
    MIN_MV = c1.number_input("最小流通市值(亿)", value=50.0, min_value=0.0)
    MAX_MV = c2.number_input("最大流通市值(亿)", value=1000.0, min_value=0.0)

    config_id = make_config_id(MIN_PRICE, MIN_MV, MAX_MV, MAX_TOP_N, PULLBACK_WINDOW)
    DAILY_SIGNAL_FILE = f"skdj_v14_6_daily_signal_history_{config_id}.csv"
    DAILY_POSITION_FILE = f"skdj_v14_6_daily_position_state_{config_id}.csv"
    HISTORY_FILE = f"skdj_v14_6_history_{config_id}.csv"
    HISTORY_LEDGER_FILE = f"skdj_v14_6_scanned_dates_{config_id}.csv"

    st.markdown("---")
    if st.button("🗑️ 清空行情缓存"):
        if os.path.exists(MARKET_CACHE_FILE):
            os.remove(MARKET_CACHE_FILE)
        st.cache_data.clear()
        st.success("行情缓存已清理。")
    if st.button("🗑️ 清空V14.6每日状态"):
        for path in (DAILY_SIGNAL_FILE, DAILY_POSITION_FILE):
            if os.path.exists(path):
                os.remove(path)
        st.success("每日信号与持仓状态已清理。")

    try:
        secret_token = st.secrets.get("TUSHARE_TOKEN", "") if hasattr(st, "secrets") else ""
    except Exception:
        secret_token = ""
    TOKEN_INPUT = st.text_input("🔑 Tushare Token", value=secret_token, type="password")

token_clean = clean_token_str(TOKEN_INPUT)


# -----------------------------------------------------------------------------
# 主运行流程
# -----------------------------------------------------------------------------
if st.button("🚀 运行V14.6"):
    valid_token, token_message = verify_token_connection(token_clean)
    if not valid_token:
        st.error(f"❌ Token预检失败：{token_message}")
    else:
        try:
            ts.set_token(token_clean)
            pro = ts.pro_api(token_clean)
            with st.spinner("构建科技扩展池..."):
                whitelist, name_map = load_custom_tech_whitelist(token_clean)
                whitelist_keys = tuple(sorted(whitelist))
            if not whitelist_keys:
                st.error("未获取到股票池，请检查Token权限或网络。")
                st.stop()
            st.info(f"股票池共 {len(whitelist_keys)} 只。当前池包含科技、高端制造及部分医药生物。")

            end_str = end_date.strftime("%Y%m%d")
            lookback_calendar_days = max(int(BACKTEST_DAYS) * 3, 950)
            cal_start = (datetime.strptime(end_str, "%Y%m%d") - timedelta(days=lookback_calendar_days)).strftime("%Y%m%d")
            cal_extended_end = (datetime.strptime(end_str, "%Y%m%d") + timedelta(days=15)).strftime("%Y%m%d")
            cal = safe_tushare_call(pro.trade_cal, exchange="SSE", start_date=cal_start, end_date=cal_extended_end)
            if cal.empty:
                st.error("无法获取交易日历。")
                st.stop()
            open_days_all = cal[cal["is_open"] == 1].sort_values("cal_date")["cal_date"].astype(str).tolist()
            available_cal_days = [d for d in open_days_all if d <= end_str]
            if not available_cal_days:
                st.error("截止日期前没有有效交易日。")
                st.stop()

            fetch_start = (datetime.strptime(available_cal_days[0], "%Y%m%d") - timedelta(days=320)).strftime("%Y%m%d")
            fetch_end = min(
                (datetime.strptime(end_str, "%Y%m%d") + timedelta(days=5)).strftime("%Y%m%d"),
                datetime.now().strftime("%Y%m%d"),
            )
            cache_stamp = int(os.path.getmtime(MARKET_CACHE_FILE)) if os.path.exists(MARKET_CACHE_FILE) else 0
            stock_dict, basic_indexed = load_optimized_market_data(
                fetch_start, fetch_end, token_clean, whitelist_keys, cache_stamp
            )
            if not stock_dict:
                st.error("没有加载到行情数据，请重试。")
                st.stop()

            market_dates = [frame.index.max() for frame in stock_dict.values() if not frame.empty]
            data_date = min(end_str, max(market_dates))
            if data_date < end_str:
                st.warning(f"截止日行情尚未完整发布，本次使用最新可用日线：{data_date}。盘中14:50需另接实时/分钟行情。")
            else:
                st.success(f"本次使用行情日期：{data_date}")

            if RUN_MODE == "每日状态更新":
                existing_signals = read_csv_safe(DAILY_SIGNAL_FILE)
                if not existing_signals.empty and "Config_ID" in existing_signals.columns:
                    existing_signals = existing_signals[existing_signals["Config_ID"].astype(str) == config_id]

                with st.spinner("扫描预备上穿与正式上穿信号..."):
                    watch_all = build_candidate_records(
                        data_date, "预备上穿", whitelist_keys, name_map, stock_dict, basic_indexed,
                        MIN_PRICE, MIN_MV, MAX_MV,
                    )
                    formal_all = build_candidate_records(
                        data_date, "正式上穿25", whitelist_keys, name_map, stock_dict, basic_indexed,
                        MIN_PRICE, MIN_MV, MAX_MV,
                    )

                watch_top = watch_all.head(int(MAX_TOP_N)).copy()
                formal_top = formal_all.head(int(MAX_TOP_N)).copy()
                watch_events = candidates_to_events(watch_top, "预备上穿", config_id)
                formal_events = candidates_to_events(formal_top, "正式上穿25", config_id)
                stage_one_history = merge_new_events(existing_signals, [watch_events, formal_events])

                # 首次运行也能识别“几天前突破、今天回踩止跌”，不必等应用先连续运行十天。
                parent_dates = [
                    date for date in available_cal_days if date < data_date
                ][-int(PULLBACK_WINDOW):]
                with st.spinner("回补近期正式突破来源并扫描回踩止跌..."):
                    recent_formal_sources = build_recent_formal_sources(
                        parent_dates, whitelist_keys, name_map, stock_dict, basic_indexed,
                        MIN_PRICE, MIN_MV, MAX_MV, MAX_TOP_N, config_id,
                    )
                    pullback_source_history = merge_new_events(
                        stage_one_history, [recent_formal_sources]
                    )
                    pullback_all = build_pullback_candidates(
                        pullback_source_history, data_date, stock_dict, name_map, int(PULLBACK_WINDOW)
                    )
                pullback_top = pullback_all.head(int(MAX_TOP_N)).copy()
                pullback_events = candidates_to_events(
                    pullback_top, "回踩止跌", config_id, parent_col="Parent_Event_ID"
                )
                signal_history = merge_new_events(stage_one_history, [pullback_events])
                atomic_write_csv(signal_history, DAILY_SIGNAL_FILE)

                st.subheader("📡 今日三路径信号")
                tabs = st.tabs(["预备上穿", "正式上穿25", "回踩止跌"])
                for tab, frame, label in zip(
                    tabs,
                    (watch_top, formal_top, pullback_top),
                    ("预备信号：周中可能变化", "正式突破：周五后确认", "突破后日线回踩止跌"),
                ):
                    with tab:
                        st.caption(label)
                        if frame.empty:
                            st.info("今日没有新进入该状态的候选；既有股票仍会继续更新。")
                        else:
                            show_cols = [
                                "Raw_Rank", "name", "ts_code", "Total_Score", "SKDJ_K", "SKDJ_D",
                                "Pre_Return_5D", "Pre_Return_10D", "Pre_Return_15D",
                                "Rise_From_20D_Low", "Days_Since_20D_Low", "Below_10D_High",
                                "Up_Days_10D", "Last_3D_Return", "Timing_Flag",
                            ]
                            show_cols += [c for c in ("Pullback_Drawdown (%)", "Pullback_Days") if c in frame.columns]
                            st.dataframe(frame[[c for c in show_cols if c in frame.columns]], width="stretch")

                positions = refresh_all_positions(signal_history, stock_dict, data_date)
                atomic_write_csv(positions, DAILY_POSITION_FILE)
                recent_status_dates = [
                    date for date in available_cal_days if date <= data_date
                ][-5:]
                display_lifecycle_report(
                    positions, "📈 每日信号与持仓状态", recent_trade_dates=recent_status_dates
                )
                st.download_button(
                    "📥 下载每日信号历史",
                    signal_history.to_csv(index=False).encode("utf-8-sig"),
                    file_name="skdj_v14_6_daily_signal_history.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "📥 下载每日持仓状态",
                    positions.to_csv(index=False).encode("utf-8-sig"),
                    file_name="skdj_v14_6_daily_position_state.csv",
                    mime="text/csv",
                )

            else:
                cal_df = pd.DataFrame({"cal_date": open_days_all})
                cal_df["dt"] = pd.to_datetime(cal_df["cal_date"])
                cal_df["year_week"] = cal_df["dt"].dt.strftime("%G_%V")
                completed_week_ends = set(cal_df.groupby("year_week")["cal_date"].max().tolist())
                recent_days = available_cal_days[-int(BACKTEST_DAYS):]
                # 历史样本必须已经出现下一交易日开盘，才能执行高低开过滤并在被剔除后递补。
                # 最新完成周若尚未开出下一交易日，留给每日状态模式跟踪，历史账本暂不锁定。
                scan_dates = [d for d in recent_days if d in completed_week_ends and d < data_date]

                ledger = read_csv_safe(HISTORY_LEDGER_FILE)
                processed = set(ledger["Trade_Date"].astype(str)) if not ledger.empty and "Trade_Date" in ledger.columns else set()
                new_scan_dates = [d for d in scan_dates if d not in processed]
                history = read_csv_safe(HISTORY_FILE)

                if new_scan_dates:
                    bar = st.progress(0, text="执行完成周线历史扫描...")
                    new_events = []
                    for idx, date in enumerate(new_scan_dates):
                        formal_all = build_candidate_records(
                            date, "正式上穿25", whitelist_keys, name_map, stock_dict, basic_indexed,
                            MIN_PRICE, MIN_MV, MAX_MV,
                        )
                        selected = []
                        if not formal_all.empty:
                            for _, candidate in formal_all.iterrows():
                                check = entry_check(candidate["ts_code"], date, candidate["Signal_Close"], stock_dict)
                                if check["Entry_Status"] == "SKIPPED":
                                    continue
                                selected.append(candidate.to_dict())
                                if len(selected) >= int(MAX_TOP_N):
                                    break
                        selected_df = pd.DataFrame(selected)
                        if not selected_df.empty:
                            selected_df["Rank"] = np.arange(1, len(selected_df) + 1)
                            events = candidates_to_events(selected_df, "正式上穿25", config_id)
                            new_events.append(events)

                        ledger_row = pd.DataFrame(
                            [{"Trade_Date": date, "Raw_Signal_Count": len(formal_all), "Selected_Count": len(selected), "Config_ID": config_id}]
                        )
                        ledger = pd.concat([ledger, ledger_row], ignore_index=True)
                        bar.progress((idx + 1) / len(new_scan_dates), text=f"历史扫描 {idx + 1}/{len(new_scan_dates)}：{date}")
                    bar.empty()
                    if new_events:
                        history = merge_new_events(history, new_events)
                    atomic_write_csv(ledger.drop_duplicates(["Trade_Date", "Config_ID"], keep="last"), HISTORY_LEDGER_FILE)

                if history.empty:
                    st.info("当前历史区间没有信号记录。")
                else:
                    # 不重新选股，但每次运行都用最新行情刷新全部旧持仓。
                    refreshed = refresh_all_positions(history, stock_dict, data_date)
                    atomic_write_csv(refreshed, HISTORY_FILE)
                    display_lifecycle_report(refreshed, "📈 完成周线历史回测")
                    st.download_button(
                        "📥 下载V14.6历史流水单",
                        refreshed.to_csv(index=False).encode("utf-8-sig"),
                        file_name="skdj_v14_6_history_export.csv",
                        mime="text/csv",
                    )

        except Exception as exc:
            st.exception(exc)


# 页面底部始终展示已保存的最近一次每日状态，避免必须重新运行才能查看。
if RUN_MODE == "每日状态更新" and os.path.exists(DAILY_POSITION_FILE):
    saved_positions = read_csv_safe(DAILY_POSITION_FILE)
    if not saved_positions.empty:
        st.markdown("---")
        st.caption("以下为磁盘中最近一次成功保存的每日状态。点击运行后会用最新行情重算。")
        active_saved = saved_positions[saved_positions["Status"].isin(["WAIT_BUY", "HOLDING"])].copy()
        if not active_saved.empty:
            cols = [
                "Signal_Date", "Signal_Type", "Rank", "name", "Status", "Buy_Date", "Buy_Price",
                "Hold_Days", "Current_Week", "Current_Return (%)", "Peak_Return (%)",
                "Drawdown_From_Peak (%)", "Stop_Stage", "Stop_Level",
            ]
            st.subheader("📌 最近保存的待买入/持仓中股票")
            st.dataframe(active_saved[[c for c in cols if c in active_saved.columns]], width="stretch")
