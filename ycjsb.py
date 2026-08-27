# -*- coding: utf-8 -*-
"""
周线 SKDJ 双向 ETA10/ETA15 提前买点验证系统 V14.7
============================================================
核心约定
1. 保留原 V14.5 周线 SKDJ：周线 K 上穿 25 且 K>D 为正式信号。
2. 冻结原 V14.5 正式信号与评分；每周 Top 5 不受 ETA 条件影响。
3. 同一正式 Top 5 母样本并行验证：正式上穿、下跌 ETA10/15、上升 ETA10/15。
4. ETA 表示按最近5个交易日动态周线K速度，预计还需多少交易日触及25；
   首次进入10/15日阈值时登记，下一交易日开盘模拟买入。
5. 普通 A 股按 T+1；首周第5个交易日收盘截断；暂不计交易成本。
6. ETA 对照属于“以后来成为正式 Top 5 为条件”的配对诊断，不冒充独立实盘胜率。
7. 行情按交易日分片原子保存；历史结果先保存、扫描账本后提交，崩溃可断点续跑。
8. 下载按钮与回测按钮解耦，页面重跑后仍展示最近一次完整/部分结果。
============================================================
"""

from __future__ import annotations

import hashlib
import json
import gc
import os
import pickle
import re
import shutil
import tempfile
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts

warnings.filterwarnings("ignore")

APP_VERSION = "V14.7"
MARKET_CACHE_ROOT = "skdj_v14_7_market_cache"
ETA_BRANCHES = ("正式上穿25", "下跌ETA10", "下跌ETA15", "上升ETA10", "上升ETA15")

st.set_page_config(page_title=f"SKDJ {APP_VERSION} ETA验证系统", layout="wide")
st.title(f"🔬 周线 SKDJ 双向 ETA10/ETA15 提前买点验证系统 ({APP_VERSION})")
st.markdown(
    "**正式 Top 5 与 V14.5 规则保持独立**；ETA 只改变同一母样本的买入时点。"
    "下跌组是用户假设，上升组是对照假设，结果由回测决定。"
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
    except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError, OSError):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def atomic_write_csv(df: pd.DataFrame, path: str):
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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


def _pool_cache_dir(whitelist_set):
    digest = hashlib.sha1("|".join(sorted(whitelist_set)).encode("utf-8")).hexdigest()[:12]
    path = os.path.join(MARKET_CACHE_ROOT, digest)
    os.makedirs(path, exist_ok=True)
    return path, digest


def _valid_market_partition(payload, trade_date, pool_hash):
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != 2 or payload.get("trade_date") != str(trade_date):
        return False
    if payload.get("pool_hash") != pool_hash:
        return False
    daily, adj, basic = payload.get("daily"), payload.get("adj"), payload.get("daily_basic")
    if not isinstance(daily, pd.DataFrame) or not isinstance(adj, pd.DataFrame) or not isinstance(basic, pd.DataFrame):
        return False
    # Tushare按交易日应返回全市场数千行；过小的非空响应视为被截断，不永久写入缓存。
    if min(int(payload.get("raw_daily_count", 0)), int(payload.get("raw_adj_count", 0)), int(payload.get("raw_basic_count", 0))) < 1000:
        return False
    if daily.empty or adj.empty or basic.empty:
        return False
    required_daily = {"ts_code", "trade_date", "open", "high", "low", "close", "vol"}
    return (
        required_daily.issubset(daily.columns)
        and {"ts_code", "trade_date", "adj_factor"}.issubset(adj.columns)
        and {"ts_code", "trade_date", "circ_mv"}.issubset(basic.columns)
    )


def _atomic_write_pickle(payload, path):
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as file_obj:
            pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _read_market_partition(path, trade_date, pool_hash):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as file_obj:
            payload = pickle.load(file_obj)
        return payload if _valid_market_partition(payload, trade_date, pool_hash) else None
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        return None


def sync_market_data_incrementally(start_date, end_date, token, whitelist_set):
    """按交易日分片保存，任何单日损坏都不会拖垮整个两小时缓存。"""
    token_c = clean_token_str(token)
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    cal = safe_tushare_call(pro.trade_cal, exchange="SSE", start_date=start_date, end_date=end_date)
    if cal.empty:
        return [], "", ""

    today_str = datetime.now().strftime("%Y%m%d")
    valid_dates = (
        cal[(cal["is_open"] == 1) & (cal["cal_date"].astype(str) <= today_str)]
        .sort_values("cal_date")["cal_date"].astype(str).tolist()
    )
    cache_dir, pool_hash = _pool_cache_dir(whitelist_set)
    missing = []
    for date in valid_dates:
        path = os.path.join(cache_dir, f"{date}.pkl")
        if _read_market_partition(path, date, pool_hash) is None:
            missing.append(date)

    if missing:
        bar = st.progress(0, text=f"📥 断点同步 {len(missing)} 个交易日行情...")
        for idx, trade_date in enumerate(missing):
            # 先取全市场，再过滤。只有日线与复权因子均完整返回才提交该日期。
            daily_all = safe_tushare_call(pro.daily, trade_date=trade_date)
            adj_all = safe_tushare_call(pro.adj_factor, trade_date=trade_date)
            basic_all = safe_tushare_call(
                pro.daily_basic, trade_date=trade_date, fields="ts_code,trade_date,circ_mv"
            )
            if not daily_all.empty and not adj_all.empty:
                common = set(daily_all["ts_code"].astype(str)) & set(adj_all["ts_code"].astype(str))
                daily = daily_all[daily_all["ts_code"].isin(whitelist_set & common)].copy()
                adj = adj_all[adj_all["ts_code"].isin(whitelist_set & common)].copy()
                basic = (
                    basic_all[basic_all["ts_code"].isin(whitelist_set)].copy()
                    if not basic_all.empty else pd.DataFrame()
                )
                payload = {
                    "version": 2,
                    "trade_date": trade_date,
                    "pool_hash": pool_hash,
                    "raw_daily_count": int(len(daily_all)),
                    "raw_adj_count": int(len(adj_all)),
                    "raw_basic_count": int(len(basic_all)),
                    "daily": daily,
                    "adj": adj,
                    "daily_basic": basic,
                }
                if _valid_market_partition(payload, trade_date, pool_hash):
                    _atomic_write_pickle(payload, os.path.join(cache_dir, f"{trade_date}.pkl"))
            bar.progress((idx + 1) / len(missing), text=f"📥 行情同步 {idx + 1}/{len(missing)}：{trade_date}")
            time.sleep(0.12)
        bar.empty()
    return valid_dates, cache_dir, pool_hash


def load_optimized_market_data(start_date, end_date, token, whitelist_keys, cache_stamp=None):
    del cache_stamp
    whitelist = set(whitelist_keys)
    valid_dates, cache_dir, pool_hash = sync_market_data_incrementally(start_date, end_date, token, whitelist)
    daily_parts, adj_parts, basic_parts = [], [], []
    for trade_date in valid_dates:
        payload = _read_market_partition(os.path.join(cache_dir, f"{trade_date}.pkl"), trade_date, pool_hash)
        if payload is None:
            continue
        daily_parts.append(payload["daily"])
        adj_parts.append(payload["adj"])
        basic = payload.get("daily_basic")
        if isinstance(basic, pd.DataFrame) and not basic.empty:
            basic_parts.append(basic)
    daily_raw = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    adj_raw = pd.concat(adj_parts, ignore_index=True) if adj_parts else pd.DataFrame()
    basic_raw = pd.concat(basic_parts, ignore_index=True) if basic_parts else pd.DataFrame()
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
    del daily_raw, adj_raw, daily_parts, adj_parts
    gc.collect()

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
    del merged, basic_raw, basic_parts
    gc.collect()
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


def build_dynamic_skdj_series(df_daily: pd.DataFrame) -> pd.DataFrame:
    """逐交易日重建“截至当天”的动态周线K/D，不把未来周内数据泄漏到过去。"""
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
    frame = df_daily.copy().reset_index()
    frame["trade_date_str"] = frame["trade_date_str"].astype(str)
    frame = frame.sort_values("trade_date_str").reset_index(drop=True)
    frame["dt"] = pd.to_datetime(frame["trade_date_str"])
    frame["year_week"] = frame["dt"].dt.strftime("%G_%V")

    completed_lows, completed_highs, completed_ks = [], [], []
    previous_rsv_ewm = np.nan
    previous_k = np.nan
    records = []
    alpha = 0.5  # pandas ewm(span=3, adjust=False)

    for _, week in frame.groupby("year_week", sort=False):
        week = week.sort_values("trade_date_str")
        running_high = -np.inf
        running_low = np.inf
        week_last_rsv = np.nan
        week_last_k = np.nan
        for _, row in week.iterrows():
            running_high = max(running_high, float(row["high"]))
            running_low = min(running_low, float(row["low"]))
            lows = completed_lows[-5:] + [running_low]
            highs = completed_highs[-5:] + [running_high]
            raw_rsv = np.nan
            rsv_ewm = np.nan
            dynamic_k = np.nan
            dynamic_d = np.nan
            if len(lows) == 6:
                lowv, highv = min(lows), max(highs)
                raw_rsv = (float(row["close"]) - lowv) / max(highv - lowv, 0.001) * 100.0
                rsv_ewm = raw_rsv if not np.isfinite(previous_rsv_ewm) else (
                    alpha * raw_rsv + (1.0 - alpha) * previous_rsv_ewm
                )
                dynamic_k = rsv_ewm if not np.isfinite(previous_k) else (
                    alpha * rsv_ewm + (1.0 - alpha) * previous_k
                )
                prior_two = [value for value in completed_ks[-2:] if np.isfinite(value)]
                if len(prior_two) == 2:
                    dynamic_d = float(np.mean(prior_two + [dynamic_k]))
                week_last_rsv = rsv_ewm
                week_last_k = dynamic_k
            records.append(
                {
                    "trade_date_str": str(row["trade_date_str"]),
                    "Dynamic_K": dynamic_k,
                    "Dynamic_D": dynamic_d,
                    "Close": float(row["close"]),
                }
            )
        completed_lows.append(float(week["low"].min()))
        completed_highs.append(float(week["high"].max()))
        completed_ks.append(week_last_k)
        if np.isfinite(week_last_rsv):
            previous_rsv_ewm = week_last_rsv
        if np.isfinite(week_last_k):
            previous_k = week_last_k

    result = pd.DataFrame(records).set_index("trade_date_str")
    result["K_Velocity_5D"] = (result["Dynamic_K"] - result["Dynamic_K"].shift(4)) / 4.0
    result["Direction"] = np.where(
        (result["Dynamic_K"] > 25.0) & (result["K_Velocity_5D"] < 0), "DOWN",
        np.where((result["Dynamic_K"] < 25.0) & (result["K_Velocity_5D"] > 0), "UP", "NONE"),
    )
    result["ETA_Days"] = np.where(
        result["Direction"] == "DOWN",
        (result["Dynamic_K"] - 25.0) / result["K_Velocity_5D"].abs(),
        np.where(
            result["Direction"] == "UP",
            (25.0 - result["Dynamic_K"]) / result["K_Velocity_5D"],
            np.nan,
        ),
    )
    result.loc[(result["ETA_Days"] < 0) | (result["ETA_Days"] > 120), "ETA_Days"] = np.nan
    return result


def _eta_entered_horizon(dynamic, pos, direction, horizon, tolerance):
    row = dynamic.iloc[pos]
    if row.get("Direction") != direction or not np.isfinite(row.get("ETA_Days", np.nan)):
        return False, ""
    eta = float(row["ETA_Days"])
    prev = dynamic.iloc[pos - 1] if pos > 0 else None
    in_window = abs(eta - float(horizon)) <= float(tolerance)
    prev_in_window = bool(
        prev is not None
        and prev.get("Direction") == direction
        and np.isfinite(prev.get("ETA_Days", np.nan))
        and abs(float(prev["ETA_Days"]) - float(horizon)) <= float(tolerance)
    )
    if in_window and not prev_in_window:
        return True, "首次进入容差窗口"
    if pos <= 0:
        return False, ""
    if (
        prev.get("Direction") == direction
        and np.isfinite(prev.get("ETA_Days", np.nan))
        and float(prev["ETA_Days"]) > float(horizon) + float(tolerance)
        and eta < float(horizon) - float(tolerance)
    ):
        return True, "速度突变跨越窗口"
    return False, ""


def find_eta_trigger(dynamic, formal_date, direction, horizon, tolerance=2.0, lookback_days=60):
    if dynamic is None or dynamic.empty:
        return None
    prior = dynamic[dynamic.index < str(formal_date)].tail(int(lookback_days)).copy()
    if prior.empty:
        return None
    hits = []
    for pos in range(len(prior)):
        entered, trigger_note = _eta_entered_horizon(prior, pos, direction, horizon, tolerance)
        if entered:
            row = prior.iloc[pos]
            hits.append(
                {
                    "Prediction_Date": prior.index[pos],
                    "Prediction_K": round(float(row["Dynamic_K"]), 4),
                    "Prediction_D": round(float(row["Dynamic_D"]), 4) if np.isfinite(row["Dynamic_D"]) else np.nan,
                    "K_Velocity_5D": round(float(row["K_Velocity_5D"]), 4),
                    "Predicted_ETA_Days": round(float(row["ETA_Days"]), 2),
                    "ETA_Trigger_Note": trigger_note,
                    "Signal_Close": float(row["Close"]),
                }
            )
    # 一次正式信号只配对最近一轮靠近25的预测，避免把更早的无关波段错配进来。
    return hits[-1] if hits else None


def build_eta_comparison_events(formal_history, stock_dict, tolerance=2.0, lookback_days=60):
    if formal_history is None or formal_history.empty:
        return pd.DataFrame()
    dynamic_cache = {}
    rows = []
    branch_specs = [
        ("下跌ETA10", "DOWN", 10),
        ("下跌ETA15", "DOWN", 15),
        ("上升ETA10", "UP", 10),
        ("上升ETA15", "UP", 15),
    ]
    ordered = formal_history.sort_values(["Signal_Date", "Rank"]).reset_index(drop=True)
    for _, mother in ordered.iterrows():
        code = str(mother["ts_code"])
        parent_id = str(mother["Event_ID"])
        formal_date = parse_yyyymmdd(mother["Signal_Date"])
        if code not in dynamic_cache:
            dynamic_cache[code] = build_dynamic_skdj_series(stock_dict.get(code, pd.DataFrame()))

        baseline = mother.to_dict()
        baseline.update(
            {
                "Parent_Event_ID": parent_id,
                "Formal_Signal_Date": formal_date,
                "Entry_Path": "正式上穿25",
                "Has_ETA_Signal": True,
                "Prediction_Date": formal_date,
                "Prediction_K": mother.get("SKDJ_K", np.nan),
                "Prediction_D": mother.get("SKDJ_D", np.nan),
                "Predicted_ETA_Days": 0.0,
                "ETA_Trigger_Note": "原V14.5正式买点",
                "Lead_Trading_Days": 0,
                "Event_ID": hashlib.sha1(f"{parent_id}|正式上穿25".encode()).hexdigest()[:20],
            }
        )
        rows.append(baseline)

        dynamic = dynamic_cache[code]
        dates_before = list(dynamic.index[dynamic.index < formal_date]) if not dynamic.empty else []
        for branch, direction, horizon in branch_specs:
            trigger = find_eta_trigger(
                dynamic, formal_date, direction, horizon,
                tolerance=float(tolerance), lookback_days=int(lookback_days),
            )
            item = mother.to_dict()
            item.update(
                {
                    "Parent_Event_ID": parent_id,
                    "Formal_Signal_Date": formal_date,
                    "Entry_Path": branch,
                    "Signal_Type": branch,
                    "Event_ID": hashlib.sha1(f"{parent_id}|{branch}".encode()).hexdigest()[:20],
                    "Has_ETA_Signal": trigger is not None,
                }
            )
            if trigger is None:
                item.update(
                    {
                        "Signal_Date": None,
                        "Prediction_Date": None,
                        "Prediction_K": np.nan,
                        "Prediction_D": np.nan,
                        "K_Velocity_5D": np.nan,
                        "Predicted_ETA_Days": np.nan,
                        "ETA_Trigger_Note": "此前未出现该方向/期限预测",
                        "Lead_Trading_Days": np.nan,
                    }
                )
            else:
                prediction_date = str(trigger["Prediction_Date"])
                lead = sum(prediction_date < date <= formal_date for date in dates_before + [formal_date])
                item.update(trigger)
                item.update(
                    {
                        "Signal_Date": prediction_date,
                        "Signal_Raw_Close": np.nan,
                        "Lead_Trading_Days": int(lead),
                    }
                )
            rows.append(item)
    return pd.DataFrame(rows)


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


def build_live_eta_candidates(date, branch, whitelist_keys, name_map, stock_dict, basic_indexed,
                              min_price, min_mv, max_mv, tolerance):
    specs = {
        "下跌ETA10": ("DOWN", 10), "下跌ETA15": ("DOWN", 15),
        "上升ETA10": ("UP", 10), "上升ETA15": ("UP", 15),
    }
    if branch not in specs:
        return pd.DataFrame()
    direction, horizon = specs[branch]
    records = []
    for code in whitelist_keys:
        if code not in stock_dict or str(date) not in stock_dict[code].index:
            continue
        market_row = stock_dict[code].loc[str(date)]
        if isinstance(market_row, pd.DataFrame):
            market_row = market_row.iloc[-1]
        raw_close = float(market_row.get("raw_close", market_row["close"]))
        if raw_close < min_price:
            continue
        mv = get_circ_mv_billion(basic_indexed, date, code)
        if pd.notna(mv) and (mv < min_mv or mv > max_mv):
            continue
        dynamic = build_dynamic_skdj_series(stock_dict[code][stock_dict[code].index <= str(date)])
        if dynamic.empty:
            continue
        entered, note = _eta_entered_horizon(dynamic, len(dynamic) - 1, direction, horizon, tolerance)
        if not entered:
            continue
        current = dynamic.iloc[-1]
        snap = compute_signal_snapshot(code, date, stock_dict)
        if not snap:
            continue
        snap.update(
            {
                "ts_code": code, "name": name_map.get(code, code),
                "circ_mv": round(mv, 2) if pd.notna(mv) else np.nan,
                "MV_Data_Missing": bool(pd.isna(mv)), "Signal_Type": branch,
                "Entry_Path": branch, "Has_ETA_Signal": True,
                "Prediction_Date": str(date), "Formal_Signal_Date": None,
                "Prediction_K": round(float(current["Dynamic_K"]), 4),
                "Prediction_D": round(float(current["Dynamic_D"]), 4) if np.isfinite(current["Dynamic_D"]) else np.nan,
                "K_Velocity_5D": round(float(current["K_Velocity_5D"]), 4),
                "Predicted_ETA_Days": round(float(current["ETA_Days"]), 2),
                "ETA_Trigger_Note": note,
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


def compute_macd_audit(full, buy_date, asof_date):
    result = {
        "MACD_Hist_Buy": np.nan,
        "MACD_Hist_Day3": np.nan,
        "MACD_Rapid_Shrink_Date": None,
        "MACD_Rapid_Shrink_Days": np.nan,
        "MACD_Rapid_Shrink": False,
    }
    sample = full[full.index <= str(asof_date)].copy()
    if sample.empty or str(buy_date) not in sample.index:
        return result
    close = pd.to_numeric(sample["close"], errors="coerce")
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2.0 * (dif - dea)
    post = hist[hist.index >= str(buy_date)].dropna()
    if post.empty:
        return result
    result["MACD_Hist_Buy"] = round(float(post.iloc[0]), 6)
    if len(post) >= 3:
        result["MACD_Hist_Day3"] = round(float(post.iloc[2]), 6)
    for pos in range(1, len(post)):
        current, previous = float(post.iloc[pos]), float(post.iloc[pos - 1])
        one_day_fast = previous > 0 and current < previous and (previous - current) / max(abs(previous), 1e-9) >= 0.30
        three_day_fast = False
        if pos >= 2:
            base = float(post.iloc[pos - 2])
            three_day_fast = base > 0 and current < base and (base - current) / max(abs(base), 1e-9) >= 0.50
        if one_day_fast or three_day_fast:
            result.update(
                {
                    "MACD_Rapid_Shrink_Date": post.index[pos],
                    "MACD_Rapid_Shrink_Days": int(pos + 1),
                    "MACD_Rapid_Shrink": True,
                }
            )
            break
    return result


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
            "Max_Adverse_Excursion (%)": np.nan,
            "Drawdown_From_Peak (%)": np.nan,
            "Stop_Stage": "未买入",
            "Stop_Level": np.nan,
            "Current_Week": 0,
            "Horizon_8W_Return (%)": np.nan,
            "Horizon_10W_Return (%)": np.nan,
            "Horizon_12W_Return (%)": np.nan,
            "MACD_Hist_Buy": np.nan,
            "MACD_Hist_Day3": np.nan,
            "MACD_Rapid_Shrink_Date": None,
            "MACD_Rapid_Shrink_Days": np.nan,
            "MACD_Rapid_Shrink": False,
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
    trough = buy_price
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
        trough = min(trough, low_p)

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
    result["Max_Adverse_Excursion (%)"] = round((trough - buy_price) / buy_price * 100, 2)
    result.update(compute_macd_audit(full, buy_date, asof_date))

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
    "Entry_Path", "Has_ETA_Signal", "Prediction_Date", "Formal_Signal_Date",
    "Prediction_K", "Prediction_D", "K_Velocity_5D", "Predicted_ETA_Days", "ETA_Trigger_Note",
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


def refresh_eta_positions(comparison_events, stock_dict, asof_date):
    if comparison_events is None or comparison_events.empty:
        return pd.DataFrame()
    rows = []
    progress = st.progress(0, text="🔄 重算五条买入路径...")
    for idx, (_, event) in enumerate(comparison_events.iterrows()):
        combined = event.to_dict()
        if not bool(event.get("Has_ETA_Signal", False)):
            lifecycle = {f"Return_W{w} (%)": np.nan for w in range(1, 13)}
            lifecycle.update(
                {
                    "Status": "NO_SIGNAL", "Entry_Status": "NO_SIGNAL",
                    "Entry_Reason": "该母样本此前未出现此ETA预测", "Exit_Reason": "无ETA信号",
                    "Buy_Date": None, "Buy_Price": np.nan, "Gap_pct": np.nan,
                    "Exit_Date": None, "Exit_Price": np.nan, "Final_Return (%)": np.nan,
                    "Hold_Days": 0, "Followup_Days": 0, "Current_Return (%)": np.nan,
                    "Peak_Return (%)": np.nan, "Max_Adverse_Excursion (%)": np.nan,
                    "Drawdown_From_Peak (%)": np.nan, "MACD_Rapid_Shrink": False,
                }
            )
        else:
            lifecycle = simulate_lifecycle(event, stock_dict, asof_date, max_weeks=12)
        combined.update(lifecycle)
        rows.append(combined)
        if (idx + 1) % 20 == 0 or idx == len(comparison_events) - 1:
            progress.progress(
                (idx + 1) / len(comparison_events),
                text=f"🔄 五路径更新 {idx + 1}/{len(comparison_events)}",
            )
    progress.empty()
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 报告：正确区分本周有记录、周初存活、周末存活与固定队列收益
# -----------------------------------------------------------------------------
def display_eta_validation_report(position_df):
    if position_df.empty or "Entry_Path" not in position_df.columns:
        return
    st.header("🧭 双向 ETA10/ETA15 配对验证")
    st.warning(
        "四个ETA组均以‘后来进入正式Top 5’为母样本，因此用于回答同一股票能否提前买，"
        "不等于全市场独立ETA策略的实盘胜率。未出现ETA信号的母样本仍计入覆盖率。"
    )
    total_mothers = position_df["Parent_Event_ID"].astype(str).nunique()
    rows = []
    for branch in ETA_BRANCHES:
        group = position_df[position_df["Entry_Path"] == branch].copy()
        has_signal = (
            group["Has_ETA_Signal"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
            if "Has_ETA_Signal" in group else pd.Series(False, index=group.index)
        )
        signaled = group[has_signal]
        executable = signaled[~signaled["Status"].isin(["SKIPPED", "NO_SIGNAL", "WAIT_BUY"])]
        closed = executable[executable["Status"] == "CLOSED"].copy()
        final_ret = pd.to_numeric(closed.get("Final_Return (%)"), errors="coerce").dropna()
        w1 = pd.to_numeric(executable.get("Return_W1 (%)"), errors="coerce").dropna()
        mae = pd.to_numeric(executable.get("Max_Adverse_Excursion (%)"), errors="coerce").dropna()
        lead = pd.to_numeric(signaled.get("Lead_Trading_Days"), errors="coerce").dropna()
        stop_mask = closed.get("Exit_Reason", pd.Series("", index=closed.index)).astype(str).str.contains("破-10%")
        rows.append(
            {
                "买入路径": branch,
                "母样本": total_mothers,
                "出现预测": int(has_signal.sum()),
                "预测覆盖率": has_signal.mean() * 100 if len(group) else np.nan,
                "平均提前交易日": lead.mean() if len(lead) else np.nan,
                "可执行样本": len(executable),
                "W1均益": w1.mean() if len(w1) else np.nan,
                "W1胜率": (w1 > 0).mean() * 100 if len(w1) else np.nan,
                "已结束": len(final_ret),
                "最终均益": final_ret.mean() if len(final_ret) else np.nan,
                "最终中位数": final_ret.median() if len(final_ret) else np.nan,
                "最终胜率": (final_ret > 0).mean() * 100 if len(final_ret) else np.nan,
                "平均最大不利回撤": mae.mean() if len(mae) else np.nan,
                "最差盘中回撤": mae.min() if len(mae) else np.nan,
                "-10%止损率": stop_mask.mean() * 100 if len(closed) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    pct_cols = ["预测覆盖率", "W1均益", "W1胜率", "最终均益", "最终中位数", "最终胜率", "平均最大不利回撤", "最差盘中回撤", "-10%止损率"]
    st.dataframe(summary.style.format({col: "{:.2f}%" for col in pct_cols}, na_rep="—"), width="stretch")

    closed = position_df[position_df["Status"] == "CLOSED"].copy()
    if not closed.empty:
        closed["Final_Return (%)"] = pd.to_numeric(closed["Final_Return (%)"], errors="coerce")
        pivot = closed.pivot_table(index="Parent_Event_ID", columns="Entry_Path", values="Final_Return (%)", aggfunc="first")
        if "正式上穿25" in pivot.columns:
            paired_rows = []
            for branch in ETA_BRANCHES[1:]:
                pair = pivot[["正式上穿25", branch]].dropna() if branch in pivot.columns else pd.DataFrame()
                delta = pair[branch] - pair["正式上穿25"] if not pair.empty else pd.Series(dtype=float)
                paired_rows.append(
                    {
                        "提前路径": branch,
                        "双方均结束样本": len(pair),
                        "相对原买点平均改善": delta.mean() if len(delta) else np.nan,
                        "相对原买点中位改善": delta.median() if len(delta) else np.nan,
                        "收益优于原买点比例": (delta > 0).mean() * 100 if len(delta) else np.nan,
                    }
                )
            st.subheader("同一股票相对正式买点的收益差")
            st.dataframe(
                pd.DataFrame(paired_rows).style.format(
                    {"相对原买点平均改善": "{:.2f}%", "相对原买点中位改善": "{:.2f}%", "收益优于原买点比例": "{:.1f}%"},
                    na_rep="—",
                ), width="stretch",
            )

        if "MACD_Rapid_Shrink" in closed.columns:
            audit = (
                closed.groupby(["Entry_Path", "MACD_Rapid_Shrink"], observed=False)["Final_Return (%)"]
                .agg([("样本数", "count"), ("最终均益", "mean"), ("最终胜率", lambda x: (x > 0).mean() * 100)])
                .reset_index()
            )
            st.subheader("日线MACD红柱快速缩短审计（只观察，不参与退出）")
            st.caption("快速缩短定义：单日缩短≥30%，或三日内缩短≥50%；后续可根据回测再决定是否写入卖出规则。")
            st.dataframe(audit.style.format({"最终均益": "{:.2f}%", "最终胜率": "{:.1f}%"}), width="stretch")


def display_lifecycle_report(position_df, title, recent_trade_dates=None):
    st.header(title)
    if position_df.empty:
        st.info("暂无可分析记录。")
        return

    valid = position_df[~position_df["Status"].isin(["SKIPPED", "NO_SIGNAL"])].copy()
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
        "Formal_Signal_Date", "Signal_Date", "Entry_Path", "Signal_Type", "Rank", "name", "ts_code", "Total_Score",
        "Prediction_K", "K_Velocity_5D", "Predicted_ETA_Days", "Lead_Trading_Days",
        "Pre_Return_10D", "Days_Since_20D_Low", "Timing_Flag", "Status",
        "Buy_Date", "Buy_Price", "Gap_pct", "Hold_Days",
        "Current_Week", "Current_Return (%)", "Peak_Return (%)", "Max_Adverse_Excursion (%)", "Drawdown_From_Peak (%)",
        "MACD_Rapid_Shrink", "MACD_Rapid_Shrink_Date",
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
    RUN_MODE = st.radio("运行模式", ["每日ETA状态更新", "历史五路径回测"], index=1)
    end_date = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("历史回测交易日数", value=250, min_value=30, step=30)
    MAX_TOP_N = st.number_input("每类信号最多跟踪数", value=5, min_value=1, max_value=20, step=1)
    ETA_TOLERANCE = st.number_input("ETA允许误差(交易日)", value=2.0, min_value=0.5, max_value=5.0, step=0.5)
    ETA_LOOKBACK = st.number_input("正式信号前ETA搜索天数", value=60, min_value=20, max_value=120, step=10)

    st.markdown("---")
    st.subheader("💰 股票池")
    MIN_PRICE = st.number_input("最低股价(元)", value=10.0, min_value=0.0)
    c1, c2 = st.columns(2)
    MIN_MV = c1.number_input("最小流通市值(亿)", value=50.0, min_value=0.0)
    MAX_MV = c2.number_input("最大流通市值(亿)", value=1000.0, min_value=0.0)

    config_id = make_config_id(MIN_PRICE, MIN_MV, MAX_MV, MAX_TOP_N, ETA_LOOKBACK)
    config_id += f"_tol{str(ETA_TOLERANCE).replace('.', '_')}"
    DAILY_SIGNAL_FILE = f"skdj_v14_7_daily_signal_history_{config_id}.csv"
    DAILY_POSITION_FILE = f"skdj_v14_7_daily_position_state_{config_id}.csv"
    FORMAL_HISTORY_FILE = f"skdj_v14_7_formal_mothers_{config_id}.csv"
    HISTORY_FILE = f"skdj_v14_7_eta_comparison_{config_id}.csv"
    HISTORY_LEDGER_FILE = f"skdj_v14_7_scanned_dates_{config_id}.csv"

    st.markdown("---")
    if st.button("🗑️ 清空行情缓存"):
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.cache_data.clear()
        st.success("行情缓存已清理。")
    if st.button("🗑️ 清空V14.7本配置结果"):
        for path in (DAILY_SIGNAL_FILE, DAILY_POSITION_FILE):
            if os.path.exists(path):
                os.remove(path)
        for path in (FORMAL_HISTORY_FILE, HISTORY_FILE, HISTORY_LEDGER_FILE):
            if os.path.exists(path):
                os.remove(path)
        st.success("当前参数对应的信号、回测和断点账本已清理；行情分片仍保留。")

    try:
        secret_token = st.secrets.get("TUSHARE_TOKEN", "") if hasattr(st, "secrets") else ""
    except Exception:
        secret_token = ""
    TOKEN_INPUT = st.text_input("🔑 Tushare Token", value=secret_token, type="password")

token_clean = clean_token_str(TOKEN_INPUT)


# -----------------------------------------------------------------------------
# V14.7 主流程：轻量下载范围、断点扫描、五路径对照
# -----------------------------------------------------------------------------
def prepare_runtime_market(token_value, analysis_end, backtest_days, whitelist_keys):
    pro = ts.pro_api(token_value)
    end_str = analysis_end.strftime("%Y%m%d")
    warmup_trade_days = 180
    calendar_days = max(260, int((int(backtest_days) + warmup_trade_days) * 1.60))
    cal_start = (datetime.strptime(end_str, "%Y%m%d") - timedelta(days=calendar_days)).strftime("%Y%m%d")
    cal_end = (datetime.strptime(end_str, "%Y%m%d") + timedelta(days=15)).strftime("%Y%m%d")
    cal = safe_tushare_call(pro.trade_cal, exchange="SSE", start_date=cal_start, end_date=cal_end)
    if cal.empty:
        raise RuntimeError("无法获取交易日历。")
    open_days_all = cal[cal["is_open"] == 1].sort_values("cal_date")["cal_date"].astype(str).tolist()
    available = [date for date in open_days_all if date <= end_str]
    if len(available) < 120:
        raise RuntimeError("可用交易日不足120天，无法稳定计算20周均线与ETA。")
    required_count = min(len(available), int(backtest_days) + warmup_trade_days)
    fetch_start = available[-required_count]
    fetch_end = min(end_str, datetime.now().strftime("%Y%m%d"))
    stock_dict, basic_indexed = load_optimized_market_data(
        fetch_start, fetch_end, token_value, tuple(whitelist_keys)
    )
    if not stock_dict:
        raise RuntimeError("没有加载到有效行情；未成功的单日分片会在下次运行自动重试。")
    market_dates = [frame.index.max() for frame in stock_dict.values() if not frame.empty]
    data_date = min(end_str, max(market_dates))
    return stock_dict, basic_indexed, open_days_all, available, data_date


run_clicked = st.button("🚀 运行V14.7验证", type="primary")
if run_clicked:
    valid_token, token_message = verify_token_connection(token_clean)
    if not valid_token:
        st.error(f"❌ Token预检失败：{token_message}")
    else:
        try:
            ts.set_token(token_clean)
            with st.spinner("构建冻结股票池并检查断点..."):
                whitelist, name_map = load_custom_tech_whitelist(token_clean)
                whitelist_keys = tuple(sorted(whitelist))
            if not whitelist_keys:
                raise RuntimeError("未获取到股票池，请检查Token权限或网络。")
            st.info(f"股票池 {len(whitelist_keys)} 只；ETA不会改变正式Top {int(MAX_TOP_N)}的入选数量。")

            stock_dict, basic_indexed, open_days_all, available_cal_days, data_date = prepare_runtime_market(
                token_clean, end_date, int(BACKTEST_DAYS), whitelist_keys
            )
            if data_date < end_date.strftime("%Y%m%d"):
                st.warning(f"截止日行情尚未完整发布，本次使用最新可用日线：{data_date}。")
            else:
                st.success(f"行情已更新至：{data_date}")

            if RUN_MODE == "每日ETA状态更新":
                existing = read_csv_safe(DAILY_SIGNAL_FILE)
                frames = []
                formal = build_candidate_records(
                    data_date, "正式上穿25", whitelist_keys, name_map, stock_dict, basic_indexed,
                    MIN_PRICE, MIN_MV, MAX_MV,
                ).head(int(MAX_TOP_N)).copy()
                if not formal.empty:
                    formal["Entry_Path"] = "正式上穿25"
                    formal["Has_ETA_Signal"] = True
                    formal["Prediction_Date"] = data_date
                    formal["Formal_Signal_Date"] = data_date
                    frames.append(candidates_to_events(formal, "正式上穿25", config_id))

                live_tables = {"正式上穿25": formal}
                for branch in ETA_BRANCHES[1:]:
                    eta_frame = build_live_eta_candidates(
                        data_date, branch, whitelist_keys, name_map, stock_dict, basic_indexed,
                        MIN_PRICE, MIN_MV, MAX_MV, ETA_TOLERANCE,
                    ).head(int(MAX_TOP_N)).copy()
                    live_tables[branch] = eta_frame
                    if not eta_frame.empty:
                        frames.append(candidates_to_events(eta_frame, branch, config_id))

                signals = merge_new_events(existing, frames)
                atomic_write_csv(signals, DAILY_SIGNAL_FILE)
                positions = refresh_all_positions(signals, stock_dict, data_date)
                atomic_write_csv(positions, DAILY_POSITION_FILE)
                counts = "｜".join(f"{name} {len(frame)}只" for name, frame in live_tables.items())
                st.success(f"今日首次进入阈值：{counts}")

            else:
                cal_df = pd.DataFrame({"cal_date": open_days_all})
                cal_df["dt"] = pd.to_datetime(cal_df["cal_date"])
                cal_df["year_week"] = cal_df["dt"].dt.strftime("%G_%V")
                completed_week_ends = set(cal_df.groupby("year_week")["cal_date"].max().tolist())
                recent_days = available_cal_days[-int(BACKTEST_DAYS):]
                scan_dates = [date for date in recent_days if date in completed_week_ends and date < data_date]

                ledger = read_csv_safe(HISTORY_LEDGER_FILE)
                if not ledger.empty and "Config_ID" in ledger.columns:
                    ledger = ledger[ledger["Config_ID"].astype(str) == config_id].copy()
                processed = (
                    set(ledger["Trade_Date"].map(parse_yyyymmdd).dropna())
                    if not ledger.empty and "Trade_Date" in ledger.columns else set()
                )
                mothers = read_csv_safe(FORMAL_HISTORY_FILE)
                new_scan_dates = [date for date in scan_dates if date not in processed]
                if new_scan_dates:
                    bar = st.progress(0, text="扫描冻结的V14.5正式Top 5...")
                    for idx, date in enumerate(new_scan_dates):
                        formal_all = build_candidate_records(
                            date, "正式上穿25", whitelist_keys, name_map, stock_dict, basic_indexed,
                            MIN_PRICE, MIN_MV, MAX_MV,
                        )
                        selected = []
                        for _, candidate in formal_all.iterrows():
                            check = entry_check(candidate["ts_code"], date, candidate["Signal_Close"], stock_dict)
                            if check["Entry_Status"] == "SKIPPED":
                                continue
                            selected.append(candidate.to_dict())
                            if len(selected) >= int(MAX_TOP_N):
                                break
                        if selected:
                            selected_df = pd.DataFrame(selected)
                            selected_df["Rank"] = np.arange(1, len(selected_df) + 1)
                            events = candidates_to_events(selected_df, "正式上穿25", config_id)
                            mothers = merge_new_events(mothers, [events])
                            # 先提交结果；若随后崩溃，日期尚未入账，重跑会按Event_ID安全去重。
                            atomic_write_csv(mothers, FORMAL_HISTORY_FILE)

                        ledger_row = pd.DataFrame(
                            [{
                                "Trade_Date": date, "Raw_Signal_Count": len(formal_all),
                                "Selected_Count": len(selected), "Missing_To_TopN": max(0, int(MAX_TOP_N) - len(selected)),
                                "Config_ID": config_id,
                            }]
                        )
                        ledger = pd.concat([ledger, ledger_row], ignore_index=True)
                        ledger = ledger.drop_duplicates(["Trade_Date", "Config_ID"], keep="last")
                        # 日期账本最后提交，消除“已标记完成但结果尚未保存”的永久漏周。
                        atomic_write_csv(ledger, HISTORY_LEDGER_FILE)
                        bar.progress(
                            (idx + 1) / len(new_scan_dates),
                            text=f"断点扫描 {idx + 1}/{len(new_scan_dates)}：{date}（选出{len(selected)}只）",
                        )
                    bar.empty()

                if mothers.empty:
                    st.warning("本区间没有正式上穿母样本；ETA没有作为硬过滤条件。")
                else:
                    comparison_events = build_eta_comparison_events(
                        mothers, stock_dict, tolerance=float(ETA_TOLERANCE), lookback_days=int(ETA_LOOKBACK)
                    )
                    result = refresh_eta_positions(comparison_events, stock_dict, data_date)
                    atomic_write_csv(result, HISTORY_FILE)
                    st.success(
                        f"已保存 {mothers['Event_ID'].nunique()} 个正式母样本、{len(result)} 条五路径记录。"
                        "下载或刷新页面不会清空结果。"
                    )
        except Exception as exc:
            st.error("本次运行中断。已完成的单日行情分片和已提交周次会保留，下次直接续跑。")
            st.exception(exc)


# 报告和下载始终从最后一次原子保存的文件加载，不依赖本次按钮状态。
if RUN_MODE == "历史五路径回测":
    saved_ledger = read_csv_safe(HISTORY_LEDGER_FILE)
    if not saved_ledger.empty:
        st.subheader("📦 正式Top 5选出数量审计")
        short = saved_ledger[pd.to_numeric(saved_ledger.get("Selected_Count"), errors="coerce") < int(MAX_TOP_N)]
        c1, c2, c3 = st.columns(3)
        c1.metric("已扫描完整周", len(saved_ledger))
        c2.metric(f"选满Top {int(MAX_TOP_N)}周", len(saved_ledger) - len(short))
        c3.metric("不足周数", len(short))
        if not short.empty:
            st.caption("不足来自正式信号总数或次日不可成交，不是ETA过滤；明细保留Raw_Signal_Count。")
            st.dataframe(short.sort_values("Trade_Date"), width="stretch")
    saved_result = read_csv_safe(HISTORY_FILE)
    if not saved_result.empty:
        st.markdown("---")
        st.caption("以下报告来自磁盘中最近一次成功原子保存的结果；点击下载触发页面重跑后仍会保留。")
        display_eta_validation_report(saved_result)
        display_lifecycle_report(saved_result, "📈 五路径生命周期明细")
        st.download_button(
            "📥 下载V14.7五路径回测数据",
            saved_result.to_csv(index=False).encode("utf-8-sig"),
            file_name="skdj_v14_7_eta_validation_export.csv",
            mime="text/csv",
            key="download_v147_history",
        )
    elif os.path.exists(FORMAL_HISTORY_FILE):
        mothers_partial = read_csv_safe(FORMAL_HISTORY_FILE)
        st.info(f"已保存 {len(mothers_partial)} 条正式母样本，五路径结果尚未完成；再次点击运行会从此断点继续。")
else:
    saved_daily = read_csv_safe(DAILY_POSITION_FILE)
    if not saved_daily.empty:
        st.markdown("---")
        recent_dates = sorted(
            {date for col in ("Signal_Date", "Buy_Date", "Exit_Date") if col in saved_daily.columns
             for date in saved_daily[col].map(parse_yyyymmdd).dropna()}
        )[-5:]
        display_lifecycle_report(saved_daily, "📡 最近5个交易日ETA状态", recent_trade_dates=recent_dates)
        st.download_button(
            "📥 下载V14.7每日ETA状态",
            saved_daily.to_csv(index=False).encode("utf-8-sig"),
            file_name="skdj_v14_7_daily_eta_state.csv",
            mime="text/csv",
            key="download_v147_daily",
        )
