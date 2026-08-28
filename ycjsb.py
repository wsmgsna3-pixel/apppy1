# -*- coding: utf-8 -*-
"""
周线 SKDJ 信号后日线MACD确认验证系统 V14.10
============================================================
核心约定
1. 保留原 V14.5 周线 SKDJ：周线 K 上穿 25 且 K>D 为正式信号。
2. 先冻结原始可成交 Top 5；所有日线路径只研究这五只，拒绝后保持空缺，不再低排名递补。
3. 并行验证原始买入、信号日健康过滤、MACD一日确认、MACD二日确认和买后风控五条路径。
4. 所有判断只使用当日收盘前已经发生的数据；触发后统一在下一交易日开盘执行。
5. 普通 A 股按 T+1；首周第5个交易日收盘截断；暂不计交易成本。
6. 红柱重新扩张与绿柱翻红分别打标签；报告计算过滤得失、买点延迟、误杀利润与空窗周。
7. 行情按交易日分片原子保存；历史结果先保存、扫描账本后提交，崩溃可断点续跑。
8. 持久化“任务运行中”标记；网络重连或页面重跑后自动从断点继续，可手动停止。
============================================================
"""

from __future__ import annotations

import hashlib
import json
import gc
import io
import os
import pickle
import re
import shutil
import tempfile
import time
import warnings
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts

warnings.filterwarnings("ignore")

APP_VERSION = "V14.10"
MARKET_CACHE_ROOT = "skdj_v14_7_market_cache"
STATE_PATHS = (
    "原始次日买入",
    "信号日健康买入",
    "MACD一日确认买入",
    "MACD二日确认买入",
    "买后MACD风控",
)

st.set_page_config(page_title=f"SKDJ {APP_VERSION} 日线MACD确认系统", layout="wide")
st.title(f"🔬 周线 SKDJ 信号后日线MACD确认验证系统 ({APP_VERSION})")
st.markdown(
    "**正式 Top 5、周线信号和评分保持冻结**；日线路径不得用第6名以后递补。"
    "原始路径始终保留，用于衡量一日确认、二日确认、静态过滤和买后风控的真实增减。"
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


def read_json_safe(path: str):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            value = json.load(file_obj)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def atomic_write_json(value, path: str):
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
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
            if (idx + 1) % 10 == 0 or idx == len(missing) - 1:
                bar.progress(
                    (idx + 1) / len(missing),
                    text=f"📥 行情同步 {idx + 1}/{len(missing)}：{trade_date}",
                )
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


def compute_daily_state(ts_code, end_date, stock_dict, macd_shrink_limit=30.0,
                        overextension_10d=15.0, overextension_20d=20.0):
    """只使用end_date及以前的日线，给正式周线信号做可复核状态分类。"""
    if ts_code not in stock_dict:
        return {}
    daily = stock_dict[ts_code][stock_dict[ts_code].index <= str(end_date)].copy()
    if len(daily) < 30 or str(end_date) not in daily.index:
        return {}
    close = pd.to_numeric(daily["close"], errors="coerce")
    volume = pd.to_numeric(daily["vol"], errors="coerce")
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2.0 * (dif - dea)

    last_close = float(close.iloc[-1])
    last_ma5 = float(ma5.iloc[-1])
    last_ma10 = float(ma10.iloc[-1])
    ma5_slope = (last_ma5 / float(ma5.iloc[-4]) - 1.0) * 100 if len(ma5.dropna()) >= 4 else np.nan
    ret3 = (last_close / float(close.iloc[-4]) - 1.0) * 100
    ret5 = (last_close / float(close.iloc[-6]) - 1.0) * 100
    ret10 = (last_close / float(close.iloc[-11]) - 1.0) * 100
    tail20 = daily.tail(20)
    rise20 = (last_close / float(tail20["low"].min()) - 1.0) * 100
    below20_high = (last_close / float(tail20["high"].max()) - 1.0) * 100

    hist_now = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2])
    hist_three = float(hist.iloc[-3])
    shrink_pct = (
        (hist_three - hist_now) / max(abs(hist_three), 1e-9) * 100.0
        if hist_three > 0 and hist_now < hist_three else 0.0
    )
    rapid_shrink = bool(hist_three > 0 and shrink_pct >= float(macd_shrink_limit))
    macd_growing = bool(hist_now > hist_prev)
    green_expanding = bool(hist_now < 0 and hist_now < hist_prev)
    prior_vol = volume.shift(1).tail(5).mean()
    vol_ratio = float(volume.iloc[-1] / prior_vol) if pd.notna(prior_vol) and prior_vol > 0 else np.nan

    overextended = bool(
        ret10 >= float(overextension_10d)
        or (rise20 >= float(overextension_20d) and below20_high >= -3.0)
    )
    price_weak = bool(last_close < last_ma5 and ret3 < 0)
    exhaustion = bool(overextended and (rapid_shrink or price_weak or hist_now < hist_prev))
    falling = bool(
        not exhaustion
        and (
            (last_close < last_ma5 and ma5_slope < 0 and ret3 < 0)
            or (rapid_shrink and last_close < last_ma5)
            or (green_expanding and ret3 < 0)
        )
    )
    healthy = bool(
        not exhaustion and not falling
        and last_close >= last_ma5 >= last_ma10
        and ma5_slope > 0 and ret3 >= 0
        and hist_now > 0 and not rapid_shrink
    )

    if exhaustion:
        state = "上涨衰竭"
    elif falling:
        state = "下跌转弱"
    elif healthy:
        state = "健康上涨"
    else:
        state = "震荡蓄势"

    score = 0
    score += 2 if last_close >= last_ma5 else -2
    score += 1 if last_ma5 >= last_ma10 else -1
    score += 1 if ma5_slope > 0 else -1
    score += 1 if ret3 >= 0 else -1
    score += 2 if hist_now > 0 else -2
    score += 1 if macd_growing else 0
    score -= 2 if rapid_shrink else 0
    score -= 2 if green_expanding else 0
    score -= 2 if exhaustion else 0

    reasons = [
        f"收盘{'≥' if last_close >= last_ma5 else '<'}MA5",
        f"MA5斜率{ma5_slope:.2f}%",
        f"3日{ret3:.2f}%",
        f"MACD柱{hist_now:.4f}",
    ]
    if rapid_shrink:
        reasons.append(f"MACD三柱缩短{shrink_pct:.1f}%")
    if overextended:
        reasons.append("短期涨幅偏大")
    if green_expanding:
        reasons.append("绿柱扩大")

    return {
        "State_AsOf_Date": str(end_date),
        "Daily_State": state,
        "State_Score": int(score),
        "Daily_Close": round(last_close, 4),
        "Daily_MA5": round(last_ma5, 4),
        "Daily_MA10": round(last_ma10, 4),
        "MA5_Slope_3D (%)": round(float(ma5_slope), 2),
        "Daily_Return_3D (%)": round(float(ret3), 2),
        "Daily_Return_5D (%)": round(float(ret5), 2),
        "Daily_Return_10D (%)": round(float(ret10), 2),
        "Rise_From_20D_Low_State (%)": round(float(rise20), 2),
        "Below_20D_High_State (%)": round(float(below20_high), 2),
        "MACD_Hist_Signal": round(hist_now, 6),
        "MACD_Hist_Prev": round(hist_prev, 6),
        "MACD_Hist_3Bars_Ago": round(hist_three, 6),
        "MACD_Shrink_3Bars (%)": round(float(shrink_pct), 2),
        "MACD_Rapid_Shrink_Signal": rapid_shrink,
        "MACD_Growing_Signal": macd_growing,
        "MACD_Green_Expanding": green_expanding,
        "Daily_Vol_Ratio_State": round(vol_ratio, 2) if np.isfinite(vol_ratio) else np.nan,
        "Overextended_Flag": overextended,
        "State_Reason": "；".join(reasons),
    }


def find_macd_confirmations(event, stock_dict, asof_date, max_wait_days,
                            macd_shrink_limit, overextension_10d, overextension_20d):
    """一次扫描同时返回MACD一日与二日确认；所有判断仅使用当日及以前数据。"""
    code = str(event["ts_code"])
    formal_date = parse_yyyymmdd(event.get("Signal_Date"))
    if not formal_date or code not in stock_dict:
        missing = {"Decision_Status": "REJECT", "Decision_Reason": "缺少MACD确认行情"}
        return {1: missing.copy(), 2: missing.copy()}
    future_dates = list(
        stock_dict[code][
            (stock_dict[code].index > formal_date) & (stock_dict[code].index <= str(asof_date))
        ].index
    )
    observed = future_dates[: int(max_wait_days)]
    decisions = {1: None, 2: None}
    previous_expand = False
    previous_cross = False

    for pos, date in enumerate(observed, start=1):
        state = compute_daily_state(
            code, date, stock_dict, macd_shrink_limit, overextension_10d, overextension_20d
        )
        hist_now = float(state.get("MACD_Hist_Signal", np.nan))
        hist_prev = float(state.get("MACD_Hist_Prev", np.nan))
        red_expand = bool(
            np.isfinite(hist_now) and np.isfinite(hist_prev)
            and hist_now > 0 and hist_now > hist_prev
        )
        green_to_red = bool(red_expand and hist_prev <= 0)

        def confirmation_payload(confirm_bars, confirm_type):
            result = {
                "Decision_Status": "BUY",
                "Decision_Reason": f"信号后第{pos}日收盘确认{confirm_type}",
                "Confirmation_Date": date,
                "Confirmation_Days": pos,
                "Execution_State": state.get("Daily_State"),
                "Signal_Close": float(stock_dict[code].loc[date]["close"]),
                "MACD_Confirm_Type": confirm_type,
                "MACD_Confirm_Bars": int(confirm_bars),
                "MACD_Confirm_Hist": round(hist_now, 6),
                "MACD_Confirm_Prev_Hist": round(hist_prev, 6),
            }
            result.update({f"Execution_{key}": value for key, value in state.items()})
            return result

        if red_expand and decisions[1] is None:
            one_type = "绿柱翻红1日" if green_to_red else "红柱扩张1日"
            decisions[1] = confirmation_payload(1, one_type)

        if red_expand and previous_expand and decisions[2] is None:
            two_type = "绿柱翻红后扩张2日" if previous_cross else "红柱连续扩张2日"
            decisions[2] = confirmation_payload(2, two_type)

        previous_expand = red_expand
        previous_cross = green_to_red

    for confirm_bars in (1, 2):
        if decisions[confirm_bars] is not None:
            continue
        if len(future_dates) < int(max_wait_days):
            decisions[confirm_bars] = {
                "Decision_Status": "PENDING",
                "Decision_Reason": (
                    f"已观察{len(future_dates)}/{int(max_wait_days)}日，"
                    f"尚未形成MACD连续{confirm_bars}日确认"
                ),
                "Confirmation_Days": len(future_dates),
                "Execution_State": "等待MACD确认",
                "MACD_Confirm_Bars": int(confirm_bars),
            }
        else:
            decisions[confirm_bars] = {
                "Decision_Status": "REJECT",
                "Decision_Reason": f"信号后{int(max_wait_days)}日内未形成MACD连续{confirm_bars}日确认",
                "Confirmation_Days": int(max_wait_days),
                "Execution_State": "MACD确认期满",
                "MACD_Confirm_Bars": int(confirm_bars),
            }
    return decisions


def build_state_comparison_events(formal_history, stock_dict, asof_date, max_wait_days=10,
                                  macd_shrink_limit=30.0, overextension_10d=15.0,
                                  overextension_20d=20.0, top_n=5,
                                  macd_exit_window_days=5, macd_exit_one_day_pct=30.0,
                                  macd_exit_three_day_pct=50.0):
    if formal_history is None or formal_history.empty:
        return pd.DataFrame()
    rows = []
    ordered = formal_history.sort_values(["Signal_Date", "Rank"]).reset_index(drop=True)

    # 先按原始信号排名和次日可成交性冻结Top N，避免给大量低排名候选重复计算日线状态。
    frozen_parent_ids = set()
    for _, week_group in ordered.groupby("Signal_Date", sort=False):
        selected_count = 0
        for _, candidate in week_group.sort_values(["Rank", "Raw_Rank"], kind="mergesort").iterrows():
            check = entry_check(
                str(candidate["ts_code"]), parse_yyyymmdd(candidate["Signal_Date"]),
                candidate.get("Signal_Close", np.nan), stock_dict,
            )
            if check.get("Entry_Status") == "SKIPPED":
                continue
            frozen_parent_ids.add(str(candidate["Event_ID"]))
            selected_count += 1
            if selected_count >= int(top_n):
                break
    ordered = ordered[ordered["Event_ID"].astype(str).isin(frozen_parent_ids)].copy()

    for _, mother in ordered.iterrows():
        code = str(mother["ts_code"])
        parent_id = str(mother["Event_ID"])
        formal_date = parse_yyyymmdd(mother["Signal_Date"])
        signal_state = compute_daily_state(
            code, formal_date, stock_dict, macd_shrink_limit, overextension_10d, overextension_20d
        )
        state_name = signal_state.get("Daily_State", "数据不足")

        common = mother.to_dict()
        common.update(signal_state)
        common.update(
            {
                "Parent_Event_ID": parent_id,
                "Original_Signal_Date": formal_date,
                "Signal_Daily_State": state_name,
            }
        )

        baseline = common.copy()
        baseline.update(
            {
                "Entry_Path": "原始次日买入",
                "Has_Entry_Signal": True,
                "Decision_Status": "BUY",
                "Decision_Reason": "原始正式信号，不做日线过滤",
                "Confirmation_Date": formal_date,
                "Confirmation_Days": 0,
                "Execution_State": state_name,
                "Lifecycle_Mode": "STANDARD",
                "Event_ID": hashlib.sha1(f"{parent_id}|原始次日买入".encode()).hexdigest()[:20],
            }
        )
        rows.append(baseline)

        immediate = common.copy()
        immediate_buy = state_name == "健康上涨"
        immediate.update(
            {
                "Entry_Path": "信号日健康买入",
                "Has_Entry_Signal": immediate_buy,
                "Decision_Status": "BUY" if immediate_buy else "REJECT",
                "Decision_Reason": "信号日健康上涨" if immediate_buy else f"信号日为{state_name}",
                "Confirmation_Date": formal_date if immediate_buy else None,
                "Confirmation_Days": 0,
                "Execution_State": state_name,
                "Lifecycle_Mode": "STANDARD",
                "Event_ID": hashlib.sha1(f"{parent_id}|信号日健康买入".encode()).hexdigest()[:20],
            }
        )
        if not immediate_buy:
            immediate["Signal_Date"] = None
        rows.append(immediate)

        macd_decisions = find_macd_confirmations(
            mother, stock_dict, asof_date, max_wait_days,
            macd_shrink_limit, overextension_10d, overextension_20d,
        )
        for confirm_bars, path_name in ((1, "MACD一日确认买入"), (2, "MACD二日确认买入")):
            confirmed = common.copy()
            decision = macd_decisions[confirm_bars]
            confirmed.update(decision)
            has_signal = decision.get("Decision_Status") == "BUY"
            confirmed.update(
                {
                    "Entry_Path": path_name,
                    "Has_Entry_Signal": has_signal,
                    "Lifecycle_Mode": "STANDARD",
                    "Event_ID": hashlib.sha1(f"{parent_id}|{path_name}".encode()).hexdigest()[:20],
                }
            )
            if has_signal:
                confirmed["Signal_Date"] = decision.get("Confirmation_Date")
                confirmed["Signal_Close"] = decision.get("Signal_Close", np.nan)
            else:
                confirmed["Signal_Date"] = None
            rows.append(confirmed)

        macd_exit = common.copy()
        macd_exit.update(
            {
                "Entry_Path": "买后MACD风控",
                "Has_Entry_Signal": True,
                "Decision_Status": "BUY",
                "Decision_Reason": "原始买点；前5日MACD快速缩短则下一交易日退出",
                "Confirmation_Date": formal_date,
                "Confirmation_Days": 0,
                "Execution_State": state_name,
                "Lifecycle_Mode": "MACD_EARLY_EXIT",
                "MACD_Exit_Window_Days": int(macd_exit_window_days),
                "MACD_Exit_OneDay_Pct": float(macd_exit_one_day_pct),
                "MACD_Exit_ThreeDay_Pct": float(macd_exit_three_day_pct),
                "Event_ID": hashlib.sha1(f"{parent_id}|买后MACD风控".encode()).hexdigest()[:20],
            }
        )
        rows.append(macd_exit)
    frame = pd.DataFrame(rows)
    frame["Classifier_Passed"] = frame["Decision_Status"].eq("BUY")
    frame["Selected_In_Path"] = False
    frame["Has_Entry_Signal"] = False
    frame["Selection_Reason"] = frame["Decision_Reason"]
    frame["Entry_Check_Status_At_Selection"] = "NOT_CHECKED"

    # 第一步：完全按原策略和下一日可成交性，先冻结每周原始Top N。
    baseline_rows = frame[frame["Entry_Path"] == "原始次日买入"]
    for _, group in baseline_rows.groupby("Original_Signal_Date", sort=False):
        eligible = group.sort_values(["Rank", "Raw_Rank"], kind="mergesort")
        selected_count = 0
        for row_index, candidate in eligible.iterrows():
            check = entry_check(
                str(candidate["ts_code"]), parse_yyyymmdd(candidate["Signal_Date"]),
                candidate.get("Signal_Close", np.nan), stock_dict,
            )
            frame.at[row_index, "Entry_Check_Status_At_Selection"] = check.get("Entry_Status", "WAIT_BUY")
            if check.get("Entry_Status") == "SKIPPED":
                frame.at[row_index, "Decision_Status"] = "ENTRY_SKIPPED"
                frame.at[row_index, "Selection_Reason"] = check.get("Entry_Reason", "次日不可买")
                continue
            if selected_count >= int(top_n):
                frame.at[row_index, "Decision_Status"] = "NOT_SELECTED_CAPACITY"
                frame.at[row_index, "Selection_Reason"] = f"本路径已选满Top {int(top_n)}"
                continue
            frame.at[row_index, "Selected_In_Path"] = True
            frame.at[row_index, "Has_Entry_Signal"] = True
            frame.at[row_index, "Selection_Reason"] = candidate.get("Decision_Reason", "通过")
            selected_count += 1

    baseline_ids = set(
        frame.loc[
            (frame["Entry_Path"] == "原始次日买入") & frame["Selected_In_Path"],
            "Parent_Event_ID",
        ].astype(str)
    )
    frame["Baseline_Selected"] = frame["Parent_Event_ID"].astype(str).isin(baseline_ids)

    # 第二步：其他路径只能研究已经冻结的Top N；拒绝或不可成交后保持空缺。
    other_rows = frame[frame["Entry_Path"] != "原始次日买入"]
    for row_index, candidate in other_rows.iterrows():
        parent_id = str(candidate.get("Parent_Event_ID", ""))
        if parent_id not in baseline_ids:
            frame.at[row_index, "Decision_Status"] = "NOT_IN_BASELINE_TOPN"
            frame.at[row_index, "Selection_Reason"] = "不属于冻结的原始Top N，禁止递补"
            continue
        if not bool(candidate.get("Classifier_Passed", False)):
            continue
        check = entry_check(
            str(candidate["ts_code"]), parse_yyyymmdd(candidate["Signal_Date"]),
            candidate.get("Signal_Close", np.nan), stock_dict,
        )
        frame.at[row_index, "Entry_Check_Status_At_Selection"] = check.get("Entry_Status", "WAIT_BUY")
        if check.get("Entry_Status") == "SKIPPED":
            frame.at[row_index, "Decision_Status"] = "ENTRY_SKIPPED"
            frame.at[row_index, "Selection_Reason"] = check.get("Entry_Reason", "次日不可买")
            continue
        frame.at[row_index, "Selected_In_Path"] = True
        frame.at[row_index, "Has_Entry_Signal"] = True
        frame.at[row_index, "Selection_Reason"] = candidate.get("Decision_Reason", "通过")
    return frame


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


def macd_histogram(close):
    close = pd.to_numeric(close, errors="coerce")
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    return 2.0 * (dif - dea)


def detect_early_macd_shrink(post_hist, position, one_day_pct=30.0, three_day_pct=50.0):
    """只比较买入日起已经收盘的MACD柱；返回当日可确认的触发类型。"""
    if position <= 0 or position >= len(post_hist):
        return None
    current = float(post_hist.iloc[position])
    previous = float(post_hist.iloc[position - 1])
    if previous > 0 and current < previous:
        shrink = (previous - current) / max(abs(previous), 1e-9) * 100.0
        if shrink >= float(one_day_pct):
            return f"单日缩短{shrink:.1f}%"
    if position >= 2:
        base = float(post_hist.iloc[position - 2])
        if base > 0 and current < base:
            shrink = (base - current) / max(abs(base), 1e-9) * 100.0
            if shrink >= float(three_day_pct):
                return f"三日缩短{shrink:.1f}%"
    return None


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
            "MACD_Exit_Trigger_Date": None,
            "MACD_Exit_Trigger_Days": np.nan,
            "MACD_Exit_Trigger_Type": None,
            "MACD_Exit_Executed": False,
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
    lifecycle_mode = str(event.get("Lifecycle_Mode", "STANDARD"))
    macd_exit_enabled = lifecycle_mode == "MACD_EARLY_EXIT"

    def event_number(name, default):
        try:
            value = float(event.get(name, default))
            return value if np.isfinite(value) else float(default)
        except (TypeError, ValueError):
            return float(default)

    macd_exit_window = int(event_number("MACD_Exit_Window_Days", 5))
    macd_one_day_pct = event_number("MACD_Exit_OneDay_Pct", 30.0)
    macd_three_day_pct = event_number("MACD_Exit_ThreeDay_Pct", 50.0)
    if macd_exit_enabled:
        hist_full = macd_histogram(full[full.index <= str(asof_date)]["close"])
        post_hist = hist_full[hist_full.index >= buy_date].dropna()
    else:
        post_hist = pd.Series(dtype=float)
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
            pending_reason = forced_exit_pending_reason
            executed_reason = (
                pending_reason
                if str(pending_reason).startswith("MACD前")
                else f"{pending_reason}(流动性顺延)"
            )
            exit_data = _exit_payload(
                executed_reason, date, open_p, buy_price, day_count
            )
            if str(pending_reason).startswith("MACD前"):
                result["MACD_Exit_Executed"] = True
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

        # 收盘后才能确认MACD柱缩短；最早在下一交易日开盘退出，严格遵守T+1。
        if macd_exit_enabled and day_count <= macd_exit_window and forced_exit_pending_reason is None:
            hist_position = post_hist.index.get_loc(date) if date in post_hist.index else None
            trigger_type = (
                detect_early_macd_shrink(
                    post_hist, int(hist_position), macd_one_day_pct, macd_three_day_pct
                )
                if hist_position is not None else None
            )
            if trigger_type:
                result.update(
                    {
                        "MACD_Exit_Trigger_Date": date,
                        "MACD_Exit_Trigger_Days": int(day_count),
                        "MACD_Exit_Trigger_Type": trigger_type,
                    }
                )
                forced_exit_pending_reason = f"MACD前{macd_exit_window}日快速缩短退出({trigger_type})"

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
    audit_end_date = exit_data["Exit_Date"] if exit_data is not None else asof_date
    result.update(compute_macd_audit(full, buy_date, audit_end_date))

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
        stage = "等待下一交易日执行退出"
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
# 正式信号母样本：冻结选股结果，执行路径每次可用最新数据重算
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


def refresh_state_positions(comparison_events, stock_dict, asof_date):
    if comparison_events is None or comparison_events.empty:
        return pd.DataFrame()
    rows = []
    progress = st.progress(0, text="🔄 重算五条日线执行路径...")
    for idx, (_, event) in enumerate(comparison_events.iterrows()):
        combined = event.to_dict()
        has_signal = str(event.get("Has_Entry_Signal", False)).lower() in {"true", "1", "yes"}
        decision = str(event.get("Decision_Status", "REJECT"))
        if not has_signal:
            lifecycle = {f"Return_W{w} (%)": np.nan for w in range(1, 13)}
            status = "PENDING_CONFIRMATION" if decision == "PENDING" else "REJECTED"
            lifecycle.update(
                {
                    "Status": status, "Entry_Status": status,
                    "Entry_Reason": event.get("Decision_Reason", "日线状态未通过"),
                    "Exit_Reason": "等待确认" if decision == "PENDING" else "日线状态放弃",
                    "Buy_Date": None, "Buy_Price": np.nan, "Gap_pct": np.nan,
                    "Exit_Date": None, "Exit_Price": np.nan, "Final_Return (%)": np.nan,
                    "Hold_Days": 0, "Followup_Days": 0, "Current_Return (%)": np.nan,
                    "Peak_Return (%)": np.nan, "Max_Adverse_Excursion (%)": np.nan,
                    "Drawdown_From_Peak (%)": np.nan, "MACD_Rapid_Shrink": False,
                    "MACD_Exit_Trigger_Date": None, "MACD_Exit_Trigger_Days": np.nan,
                    "MACD_Exit_Trigger_Type": None, "MACD_Exit_Executed": False,
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
def _true_mask(series):
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def display_state_validation_report(position_df):
    if position_df.empty or "Entry_Path" not in position_df.columns:
        return
    st.header("🧭 信号后日线MACD五路径验证")
    st.caption(
        "先冻结原始可成交Top N，其他路径不允许低排名递补；所有确认和退出均在当日收盘后登记，下一交易日开盘执行。"
    )
    total_mothers = position_df["Parent_Event_ID"].astype(str).nunique()
    all_weeks = sorted(position_df["Original_Signal_Date"].map(parse_yyyymmdd).dropna().unique())
    rows = []
    for path in STATE_PATHS:
        group = position_df[position_df["Entry_Path"] == path].copy()
        selected = group[_true_mask(group["Selected_In_Path"])].copy()
        frozen_count = int(_true_mask(group["Baseline_Selected"]).sum())
        followup = pd.to_numeric(selected.get("Followup_Days"), errors="coerce").fillna(0)
        w1_values = pd.to_numeric(selected.get("Return_W1 (%)"), errors="coerce")
        w1 = w1_values[(followup >= 5) & w1_values.notna()]
        h12_values = pd.to_numeric(selected.get("Horizon_12W_Return (%)"), errors="coerce")
        mature = selected[(followup >= 60) & h12_values.notna()].copy()
        h12 = pd.to_numeric(mature.get("Horizon_12W_Return (%)"), errors="coerce").dropna()
        mae = pd.to_numeric(mature.get("Max_Adverse_Excursion (%)"), errors="coerce").dropna()
        wait = pd.to_numeric(selected.get("Confirmation_Days"), errors="coerce").dropna()
        per_week = (
            selected.assign(_week=selected["Original_Signal_Date"].map(parse_yyyymmdd))
            .groupby("_week").size().reindex(all_weeks, fill_value=0)
        )
        rows.append(
            {
                "执行路径": path,
                "冻结母样本": total_mothers,
                "冻结TopN": frozen_count,
                "实际买入": len(selected),
                "接受率": len(selected) / frozen_count * 100 if frozen_count else np.nan,
                "平均等待日": wait.mean() if len(wait) else np.nan,
                "平均每周买入": per_week.mean() if len(per_week) else np.nan,
                "选满TopN周": int((per_week >= int(MAX_TOP_N)).sum()) if len(per_week) else 0,
                "空窗周": int((per_week == 0).sum()) if len(per_week) else 0,
                "W1样本": len(w1),
                "W1均益": w1.mean() if len(w1) else np.nan,
                "W1胜率": (w1 > 0).mean() * 100 if len(w1) else np.nan,
                "W12成熟样本": len(h12),
                "W12均益": h12.mean() if len(h12) else np.nan,
                "W12中位数": h12.median() if len(h12) else np.nan,
                "W12胜率": (h12 > 0).mean() * 100 if len(h12) else np.nan,
                "平均最大不利回撤": mae.mean() if len(mae) else np.nan,
                "-10%止损率": (
                    mature["Exit_Reason"].astype(str).str.contains("破-10%").mean() * 100
                    if len(mature) else np.nan
                ),
                "MACD风控退出": int(_true_mask(mature.get(
                    "MACD_Exit_Executed", pd.Series(False, index=mature.index)
                )).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    pct_cols = [
        "接受率", "W1均益", "W1胜率", "W12均益", "W12中位数", "W12胜率",
        "平均最大不利回撤", "-10%止损率",
    ]
    formats = {column: "{:.2f}%" for column in pct_cols}
    formats.update({"平均等待日": "{:.2f}", "平均每周买入": "{:.2f}"})
    st.dataframe(summary.style.format(formats, na_rep="—"), width="stretch")

    baseline = position_df[
        (position_df["Entry_Path"] == "原始次日买入")
        & _true_mask(position_df["Selected_In_Path"])
    ].copy()
    baseline["Horizon_12W_Return (%)"] = pd.to_numeric(
        baseline["Horizon_12W_Return (%)"], errors="coerce"
    )
    baseline["Followup_Days"] = pd.to_numeric(baseline["Followup_Days"], errors="coerce")
    baseline["Return_W1 (%)"] = pd.to_numeric(baseline["Return_W1 (%)"], errors="coerce")
    baseline["Buy_Price"] = pd.to_numeric(baseline["Buy_Price"], errors="coerce")
    mature_baseline = baseline[
        (baseline["Followup_Days"] >= 60) & baseline["Horizon_12W_Return (%)"].notna()
    ].copy()
    state_rows = []
    for state, group in mature_baseline.groupby("Signal_Daily_State", dropna=False):
        h12 = group["Horizon_12W_Return (%)"].dropna()
        w1 = group["Return_W1 (%)"].dropna()
        mae = pd.to_numeric(group["Max_Adverse_Excursion (%)"], errors="coerce").dropna()
        state_rows.append(
            {
                "信号日日线状态": state,
                "母样本": len(group),
                "W1样本": len(w1),
                "W1均益": w1.mean() if len(w1) else np.nan,
                "W1胜率": (w1 > 0).mean() * 100 if len(w1) else np.nan,
                "W12样本": len(h12),
                "W12均益": h12.mean() if len(h12) else np.nan,
                "W12中位数": h12.median() if len(h12) else np.nan,
                "W12胜率": (h12 > 0).mean() * 100 if len(h12) else np.nan,
                "平均最大不利回撤": mae.mean() if len(mae) else np.nan,
            }
        )
    st.subheader("信号日状态本身的历史辨别力")
    st.dataframe(
        pd.DataFrame(state_rows).style.format(
            {
                "W1均益": "{:.2f}%", "W1胜率": "{:.1f}%", "W12均益": "{:.2f}%",
                "W12中位数": "{:.2f}%", "W12胜率": "{:.1f}%", "平均最大不利回撤": "{:.2f}%",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    mature_baseline = mature_baseline.set_index("Parent_Event_ID")
    selection_rows = []
    for path in ("信号日健康买入", "MACD一日确认买入", "MACD二日确认买入"):
        path_rows = position_df[position_df["Entry_Path"] == path].set_index("Parent_Event_ID")
        common_ids = mature_baseline.index.intersection(path_rows.index)
        audit = path_rows.loc[common_ids].copy()
        base_ret = mature_baseline.loc[common_ids, "Horizon_12W_Return (%)"]
        rejected = ~_true_mask(audit["Selected_In_Path"])
        rejected_ret = base_ret[rejected]
        avoided_losers = rejected_ret[rejected_ret <= 0]
        missed_winners = rejected_ret[rejected_ret > 0]
        selection_rows.append(
            {
                "分类路径": path,
                "可判断基准样本": len(common_ids),
                "放弃样本": len(rejected_ret),
                "放弃中亏损股": len(avoided_losers),
                "放弃准确率": len(avoided_losers) / len(rejected_ret) * 100 if len(rejected_ret) else np.nan,
                "避免亏损合计": -avoided_losers.sum(),
                "误杀盈利股": len(missed_winners),
                "错失利润合计": missed_winners.sum(),
                "净筛选价值": -rejected_ret.sum(),
            }
        )
    st.subheader("避免亏损与误杀利润")
    st.caption("“净筛选价值”＝被放弃亏损的绝对值－被放弃盈利股的利润；正数才说明过滤总体有价值。")
    st.dataframe(
        pd.DataFrame(selection_rows).style.format(
            {
                "放弃准确率": "{:.1f}%", "避免亏损合计": "{:.2f}%", "错失利润合计": "{:.2f}%",
                "净筛选价值": "{:.2f}%",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    execution_rows = []
    for path in ("MACD一日确认买入", "MACD二日确认买入", "买后MACD风控"):
        path_rows = position_df[
            (position_df["Entry_Path"] == path) & _true_mask(position_df["Selected_In_Path"])
        ].copy()
        path_rows["Followup_Days"] = pd.to_numeric(path_rows["Followup_Days"], errors="coerce")
        path_rows["Horizon_12W_Return (%)"] = pd.to_numeric(
            path_rows["Horizon_12W_Return (%)"], errors="coerce"
        )
        path_rows["Buy_Price"] = pd.to_numeric(path_rows["Buy_Price"], errors="coerce")
        path_rows = path_rows[
            (path_rows["Followup_Days"] >= 60) & path_rows["Horizon_12W_Return (%)"].notna()
        ].set_index("Parent_Event_ID")
        paired_ids = mature_baseline.index.intersection(path_rows.index)
        base_ret = mature_baseline.loc[paired_ids, "Horizon_12W_Return (%)"]
        path_ret = path_rows.loc[paired_ids, "Horizon_12W_Return (%)"]
        delta = (path_ret - base_ret).dropna()
        paired_base_buy = mature_baseline.loc[paired_ids, "Buy_Price"]
        paired_path_buy = path_rows.loc[paired_ids, "Buy_Price"]
        execution_rows.append(
            {
                "执行路径": path,
                "成对成熟样本": len(delta),
                "基准均益": base_ret.loc[delta.index].mean() if len(delta) else np.nan,
                "路径均益": path_ret.loc[delta.index].mean() if len(delta) else np.nan,
                "平均改善": delta.mean() if len(delta) else np.nan,
                "优于基准比例": (delta > 0).mean() * 100 if len(delta) else np.nan,
                "差于基准比例": (delta < 0).mean() * 100 if len(delta) else np.nan,
                "平均等待日": pd.to_numeric(
                    path_rows.loc[paired_ids, "Confirmation_Days"], errors="coerce"
                ).mean() if len(paired_ids) else np.nan,
                "买得更高比例": (paired_path_buy > paired_base_buy).mean() * 100 if len(paired_ids) else np.nan,
                "买得更低比例": (paired_path_buy < paired_base_buy).mean() * 100 if len(paired_ids) else np.nan,
            }
        )
    st.subheader("动态执行相对同一只股票原买点的变化")
    st.dataframe(
        pd.DataFrame(execution_rows).style.format(
            {
                "基准均益": "{:.2f}%", "路径均益": "{:.2f}%", "平均改善": "{:.2f}%",
                "优于基准比例": "{:.1f}%", "差于基准比例": "{:.1f}%", "平均等待日": "{:.2f}",
                "买得更高比例": "{:.1f}%", "买得更低比例": "{:.1f}%",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    confirmation_rows = []
    if "MACD_Confirm_Type" in position_df.columns:
        for path in ("MACD一日确认买入", "MACD二日确认买入"):
            path_rows = position_df[
                (position_df["Entry_Path"] == path) & _true_mask(position_df["Selected_In_Path"])
            ].copy()
            path_rows["Followup_Days"] = pd.to_numeric(path_rows["Followup_Days"], errors="coerce")
            path_rows["Horizon_12W_Return (%)"] = pd.to_numeric(
                path_rows["Horizon_12W_Return (%)"], errors="coerce"
            )
            path_rows = path_rows[
                (path_rows["Followup_Days"] >= 60) & path_rows["Horizon_12W_Return (%)"].notna()
            ]
            for confirm_type, group in path_rows.groupby("MACD_Confirm_Type", dropna=False):
                returns = group["Horizon_12W_Return (%)"].dropna()
                mae = pd.to_numeric(group["Max_Adverse_Excursion (%)"], errors="coerce").dropna()
                confirmation_rows.append(
                    {
                        "确认路径": path,
                        "MACD确认类型": confirm_type,
                        "成熟样本": len(returns),
                        "平均等待日": pd.to_numeric(group["Confirmation_Days"], errors="coerce").mean(),
                        "W12均益": returns.mean() if len(returns) else np.nan,
                        "W12中位数": returns.median() if len(returns) else np.nan,
                        "W12胜率": (returns > 0).mean() * 100 if len(returns) else np.nan,
                        "平均最大不利回撤": mae.mean() if len(mae) else np.nan,
                    }
                )
    if confirmation_rows:
        st.subheader("红柱扩张与绿柱翻红分组")
        st.dataframe(
            pd.DataFrame(confirmation_rows).style.format(
                {
                    "平均等待日": "{:.2f}", "W12均益": "{:.2f}%", "W12中位数": "{:.2f}%",
                    "W12胜率": "{:.1f}%", "平均最大不利回撤": "{:.2f}%",
                },
                na_rep="—",
            ),
            width="stretch",
        )


def prepare_lifecycle_detail_view(detail, display_cols):
    """按现有字段稳定排序，再截取展示列；兼容旧版或不同路径CSV。"""
    available_display_cols = [col for col in display_cols if col in detail.columns]
    sort_preferences = [
        ("Original_Signal_Date", False),
        ("Signal_Date", False),
        ("Entry_Path", True),
        ("Rank", True),
    ]
    sort_cols = [col for col, _ in sort_preferences if col in detail.columns]
    ascending = [direction for col, direction in sort_preferences if col in detail.columns]
    if sort_cols:
        detail = detail.sort_values(
            sort_cols,
            ascending=ascending,
            na_position="last",
            kind="mergesort",
        )
    return detail[available_display_cols]


def display_lifecycle_report(position_df, title, recent_trade_dates=None):
    st.header(title)
    if position_df.empty:
        st.info("暂无可分析记录。")
        return

    valid = position_df[
        ~position_df["Status"].isin(["SKIPPED", "NO_SIGNAL", "REJECTED", "PENDING_CONFIRMATION"])
    ].copy()
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
            completed.groupby(["Entry_Path", "Rank"], observed=False)
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
        active_mask = detail["Status"].isin(["WAIT_BUY", "HOLDING", "PENDING_CONFIRMATION"])
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
        "Original_Signal_Date", "Signal_Date", "Entry_Path", "Rank", "name", "ts_code", "Total_Score",
        "Lifecycle_Mode", "Baseline_Selected", "Selected_In_Path",
        "Signal_Daily_State", "Execution_State", "Decision_Status", "Decision_Reason",
        "Confirmation_Date", "Confirmation_Days", "MACD_Confirm_Type", "MACD_Confirm_Bars",
        "MACD_Confirm_Hist", "MACD_Confirm_Prev_Hist",
        "Daily_Return_3D (%)", "Daily_Return_10D (%)", "MA5_Slope_3D (%)",
        "MACD_Hist_Signal", "MACD_Shrink_3Bars (%)", "Overextended_Flag", "State_Reason",
        "Pre_Return_10D", "Days_Since_20D_Low", "Timing_Flag", "Status",
        "Buy_Date", "Buy_Price", "Gap_pct", "Hold_Days",
        "Current_Week", "Current_Return (%)", "Peak_Return (%)", "Max_Adverse_Excursion (%)", "Drawdown_From_Peak (%)",
        "MACD_Rapid_Shrink", "MACD_Rapid_Shrink_Date",
        "MACD_Exit_Trigger_Date", "MACD_Exit_Trigger_Days", "MACD_Exit_Trigger_Type", "MACD_Exit_Executed",
        "Stop_Stage", "Stop_Level", "Exit_Date", "Exit_Reason", "Final_Return (%)",
    ]
    if detail.empty:
        st.info("最近5个交易日没有新状态，且当前没有未结束持仓。")
    else:
        st.dataframe(
            prepare_lifecycle_detail_view(detail, display_cols),
            width="stretch",
        )


# -----------------------------------------------------------------------------
# 侧边栏
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 运行配置")
    RUN_MODE = st.radio("运行模式", ["每日状态更新", "历史MACD路径回测"], index=1)
    end_date = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("历史回测交易日数", value=250, min_value=30, step=30)
    MAX_TOP_N = st.number_input("每周正式母样本数", value=5, min_value=1, max_value=20, step=1)
    MAX_CONFIRM_DAYS = st.number_input(
        "MACD确认最长观察天数", value=10, min_value=2, max_value=20, step=1,
        help="一日或二日扩张在收盘后确认，下一交易日开盘买入；期满未确认则放弃。",
    )

    st.markdown("---")
    st.subheader("📐 日线状态参数")
    MACD_SHRINK_LIMIT = st.number_input(
        "MACD三柱快速缩短阈值(%)", value=30.0, min_value=10.0, max_value=80.0, step=5.0
    )
    MACD_EXIT_WINDOW_DAYS = st.number_input(
        "买后MACD风控窗口(交易日)", value=5, min_value=2, max_value=10, step=1
    )
    MACD_EXIT_ONE_DAY_PCT = st.number_input(
        "MACD单日快速缩短退出阈值(%)", value=30.0, min_value=10.0, max_value=90.0, step=5.0
    )
    MACD_EXIT_THREE_DAY_PCT = st.number_input(
        "MACD三日累计缩短退出阈值(%)", value=50.0, min_value=20.0, max_value=100.0, step=5.0
    )
    OVEREXTENSION_10D = st.number_input(
        "10日涨幅偏大阈值(%)", value=15.0, min_value=5.0, max_value=40.0, step=1.0
    )
    OVEREXTENSION_20D = st.number_input(
        "距20日低点涨幅阈值(%)", value=20.0, min_value=10.0, max_value=60.0, step=2.0
    )

    st.markdown("---")
    st.subheader("💰 股票池")
    MIN_PRICE = st.number_input("最低股价(元)", value=10.0, min_value=0.0)
    c1, c2 = st.columns(2)
    MIN_MV = c1.number_input("最小流通市值(亿)", value=50.0, min_value=0.0)
    MAX_MV = c2.number_input("最大流通市值(亿)", value=1000.0, min_value=0.0)

    st.markdown("---")
    st.subheader("🔁 自动回测")
    AUTO_RESUME = st.checkbox("网络重连后自动续跑", value=True)
    HISTORY_BATCH_WEEKS = st.number_input(
        "每批扫描周数", value=5, min_value=1, max_value=20, step=1,
        help="每批原子保存后自动进入下一批，缩短单次页面连接时间。",
    )

    config_seed = (
        f"confirm={MAX_CONFIRM_DAYS}|state_shrink={MACD_SHRINK_LIMIT:.1f}|"
        f"exit_window={MACD_EXIT_WINDOW_DAYS}|exit1={MACD_EXIT_ONE_DAY_PCT:.1f}|"
        f"exit3={MACD_EXIT_THREE_DAY_PCT:.1f}|over10={OVEREXTENSION_10D:.1f}|"
        f"over20={OVEREXTENSION_20D:.1f}"
    )
    config_id = make_config_id(MIN_PRICE, MIN_MV, MAX_MV, MAX_TOP_N, MAX_CONFIRM_DAYS)
    config_id += "_" + hashlib.sha1(config_seed.encode("utf-8")).hexdigest()[:8]
    DAILY_MOTHER_FILE = f"skdj_v14_10_daily_mothers_{config_id}.csv"
    DAILY_POSITION_FILE = f"skdj_v14_10_daily_paths_{config_id}.csv"
    FORMAL_HISTORY_FILE = f"skdj_v14_10_formal_mothers_{config_id}.csv"
    HISTORY_FILE = f"skdj_v14_10_path_comparison_{config_id}.csv"
    HISTORY_LEDGER_FILE = f"skdj_v14_10_scanned_dates_{config_id}.csv"
    RUN_TASK_FILE = f"skdj_v14_10_running_task_{config_id}.json"

    task_snapshot = read_json_safe(RUN_TASK_FILE)
    if task_snapshot:
        st.caption(
            f"任务状态：{task_snapshot.get('state', '未知')}｜"
            f"已完成周次：{task_snapshot.get('completed_weeks', 0)}"
        )
    stop_auto = st.button("⏹️ 停止自动续跑")
    if stop_auto and os.path.exists(RUN_TASK_FILE):
        os.remove(RUN_TASK_FILE)
        st.success("已停止自动续跑；已保存的行情、母样本和周次不会删除。")

    if st.button("🗑️ 清空行情缓存"):
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.cache_data.clear()
        st.success("行情缓存已清理。")
    if st.button("🗑️ 清空V14.10本配置结果"):
        for path in (
            DAILY_MOTHER_FILE, DAILY_POSITION_FILE, FORMAL_HISTORY_FILE,
            HISTORY_FILE, HISTORY_LEDGER_FILE, RUN_TASK_FILE,
        ):
            if os.path.exists(path):
                os.remove(path)
        st.success("当前参数对应的结果和任务标记已清理；分片行情仍保留。")

    try:
        secret_token = st.secrets.get("TUSHARE_TOKEN", "") if hasattr(st, "secrets") else ""
    except Exception:
        secret_token = ""
    TOKEN_INPUT = st.text_input("🔑 Tushare Token", value=secret_token, type="password")
    if AUTO_RESUME and not secret_token:
        st.caption("若页面形成新会话，自动续跑需要在Secrets中配置TUSHARE_TOKEN。")

token_clean = clean_token_str(TOKEN_INPUT)


# -----------------------------------------------------------------------------
# V14.10 主流程：持久化任务标记、小批次续跑、冻结Top N后的五路径验证
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
        raise RuntimeError("可用交易日不足120天，无法稳定计算20周均线与日线状态。")
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


def start_task(mode, analysis_end, backtest_days):
    task = {
        "version": APP_VERSION,
        "config_id": config_id,
        "mode": mode,
        "end_date": analysis_end.strftime("%Y%m%d"),
        "backtest_days": int(backtest_days),
        "state": "RUNNING",
        "error_count": 0,
        "completed_weeks": 0,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(task, RUN_TASK_FILE)
    return task


def complete_task():
    if os.path.exists(RUN_TASK_FILE):
        os.remove(RUN_TASK_FILE)


def build_all_download_zip(result, mothers, ledger):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "skdj_v14_10_path_validation_export.csv",
            result.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            "skdj_v14_10_formal_mothers.csv",
            mothers.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            "skdj_v14_10_scanned_dates.csv",
            ledger.to_csv(index=False).encode("utf-8-sig"),
        )
    return buffer.getvalue()


run_clicked = st.button("🚀 运行V14.10 MACD确认验证", type="primary")
task = read_json_safe(RUN_TASK_FILE)
auto_resume = bool(
    AUTO_RESUME
    and task
    and task.get("state") == "RUNNING"
    and task.get("config_id") == config_id
    and task.get("mode") == RUN_MODE
)
if run_clicked:
    task = start_task(RUN_MODE, end_date, BACKTEST_DAYS)
    auto_resume = False

should_run = bool(run_clicked or auto_resume)
if task and task.get("state") == "PAUSED_ERROR" and not run_clicked:
    st.warning("自动续跑连续失败3次后已暂停。网络恢复后点击运行，会从现有断点继续。")
if auto_resume:
    st.info("检测到未完成任务，正在自动从最近断点续跑。")

if should_run:
    if not token_clean:
        st.error("任务断点仍在，但当前会话没有Token。请填写Token；建议在Secrets配置TUSHARE_TOKEN。")
    else:
        try:
            valid_token, token_message = verify_token_connection(token_clean)
            if not valid_token:
                raise RuntimeError(f"Token预检失败：{token_message}")

            task = read_json_safe(RUN_TASK_FILE) or start_task(RUN_MODE, end_date, BACKTEST_DAYS)
            effective_end = datetime.strptime(task.get("end_date"), "%Y%m%d").date()
            effective_days = int(task.get("backtest_days", BACKTEST_DAYS))

            ts.set_token(token_clean)
            with st.spinner("构建冻结股票池并读取断点..."):
                whitelist, name_map = load_custom_tech_whitelist(token_clean)
                whitelist_keys = tuple(sorted(whitelist))
            if not whitelist_keys:
                raise RuntimeError("未获取到股票池，请检查Token权限或网络。")
            st.info(f"股票池 {len(whitelist_keys)} 只；周线信号、原评分和正式Top {int(MAX_TOP_N)}保持不变。")

            stock_dict, basic_indexed, open_days_all, available_cal_days, data_date = prepare_runtime_market(
                token_clean, effective_end, effective_days, whitelist_keys
            )
            if data_date < effective_end.strftime("%Y%m%d"):
                st.warning(f"截止日行情尚未完整发布，本次使用最新可用日线：{data_date}。")
            else:
                st.success(f"行情已更新至：{data_date}")

            if RUN_MODE == "每日状态更新":
                mothers = read_csv_safe(DAILY_MOTHER_FILE)
                formal_all = build_candidate_records(
                    data_date, "正式上穿25", whitelist_keys, name_map, stock_dict, basic_indexed,
                    MIN_PRICE, MIN_MV, MAX_MV,
                )
                if not formal_all.empty:
                    formal_all = formal_all.copy()
                    formal_all["Rank"] = np.arange(1, len(formal_all) + 1)
                    events = candidates_to_events(formal_all, "正式上穿25", config_id)
                    mothers = merge_new_events(mothers, [events])
                    atomic_write_csv(mothers, DAILY_MOTHER_FILE)

                comparison = build_state_comparison_events(
                    mothers, stock_dict, data_date, int(MAX_CONFIRM_DAYS),
                    float(MACD_SHRINK_LIMIT), float(OVEREXTENSION_10D), float(OVEREXTENSION_20D),
                    int(MAX_TOP_N),
                    int(MACD_EXIT_WINDOW_DAYS), float(MACD_EXIT_ONE_DAY_PCT),
                    float(MACD_EXIT_THREE_DAY_PCT),
                )
                positions = refresh_state_positions(comparison, stock_dict, data_date)
                atomic_write_csv(positions, DAILY_POSITION_FILE)
                complete_task()
                state_counts = (
                    positions[positions["Entry_Path"] == "原始次日买入"]["Signal_Daily_State"]
                    .value_counts().to_dict()
                )
                st.success(f"每日状态更新完成：{state_counts}")

            else:
                cal_df = pd.DataFrame({"cal_date": open_days_all})
                cal_df["dt"] = pd.to_datetime(cal_df["cal_date"])
                cal_df["year_week"] = cal_df["dt"].dt.strftime("%G_%V")
                completed_week_ends = set(cal_df.groupby("year_week")["cal_date"].max().tolist())
                recent_days = available_cal_days[-int(effective_days):]
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
                batch_dates = new_scan_dates[: int(HISTORY_BATCH_WEEKS)]

                if batch_dates:
                    bar = st.progress(0, text="分批扫描冻结的V14.5正式Top 5...")
                    for idx, date in enumerate(batch_dates):
                        coverage_count = sum(
                            1 for frame in stock_dict.values() if not frame.empty and str(date) in frame.index
                        )
                        coverage_rate = coverage_count / max(len(stock_dict), 1)
                        if coverage_rate < 0.60:
                            ledger_row = pd.DataFrame(
                                [{
                                    "Trade_Date": date, "Raw_Signal_Count": np.nan,
                                    "Selected_Count": np.nan, "Missing_To_TopN": np.nan,
                                    "Market_Coverage_Count": coverage_count,
                                    "Market_Coverage_Rate": round(coverage_rate * 100, 2),
                                    "Scan_Status": "SKIPPED_INCOMPLETE",
                                    "Config_ID": config_id,
                                }]
                            )
                            ledger = pd.concat([ledger, ledger_row], ignore_index=True)
                            ledger = ledger.drop_duplicates(["Trade_Date", "Config_ID"], keep="last")
                            atomic_write_csv(ledger, HISTORY_LEDGER_FILE)
                            bar.progress(
                                (idx + 1) / len(batch_dates),
                                text=f"本批 {idx + 1}/{len(batch_dates)}：{date}行情不完整，已跳过",
                            )
                            continue
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
                        if not formal_all.empty:
                            formal_store = formal_all.copy()
                            formal_store["Rank"] = np.arange(1, len(formal_store) + 1)
                            events = candidates_to_events(formal_store, "正式上穿25", config_id)
                            mothers = merge_new_events(mothers, [events])
                            atomic_write_csv(mothers, FORMAL_HISTORY_FILE)

                        ledger_row = pd.DataFrame(
                            [{
                                "Trade_Date": date, "Raw_Signal_Count": len(formal_all),
                                "Selected_Count": len(selected),
                                "Stored_Candidate_Count": len(formal_all),
                                "Missing_To_TopN": max(0, int(MAX_TOP_N) - len(selected)),
                                "Market_Coverage_Count": coverage_count,
                                "Market_Coverage_Rate": round(coverage_rate * 100, 2),
                                "Scan_Status": "COMPLETED",
                                "Config_ID": config_id,
                            }]
                        )
                        ledger = pd.concat([ledger, ledger_row], ignore_index=True)
                        ledger = ledger.drop_duplicates(["Trade_Date", "Config_ID"], keep="last")
                        atomic_write_csv(ledger, HISTORY_LEDGER_FILE)
                        bar.progress(
                            (idx + 1) / len(batch_dates),
                            text=f"本批 {idx + 1}/{len(batch_dates)}：{date}（选出{len(selected)}只）",
                        )
                    bar.empty()

                    remaining = len(new_scan_dates) - len(batch_dates)
                    task.update(
                        {
                            "state": "RUNNING", "error_count": 0,
                            "completed_weeks": len(ledger),
                            "remaining_weeks": max(0, remaining),
                            "updated_at": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    atomic_write_json(task, RUN_TASK_FILE)
                    if remaining > 0 and AUTO_RESUME:
                        st.info(f"本批已原子保存，剩余{remaining}个周次，页面将自动继续。")
                        time.sleep(0.8)
                        st.rerun()
                    elif remaining > 0:
                        st.warning(f"本批已保存，剩余{remaining}个周次；再次点击运行可续跑。")

                processed_after = (
                    set(ledger["Trade_Date"].map(parse_yyyymmdd).dropna())
                    if not ledger.empty and "Trade_Date" in ledger.columns else set()
                )
                remaining_dates = [date for date in scan_dates if date not in processed_after]
                if not remaining_dates:
                    if mothers.empty:
                        st.warning("本区间没有正式上穿母样本；日线分类没有参与周线母样本筛选。")
                        complete_task()
                    else:
                        comparison = build_state_comparison_events(
                            mothers, stock_dict, data_date, int(MAX_CONFIRM_DAYS),
                            float(MACD_SHRINK_LIMIT), float(OVEREXTENSION_10D), float(OVEREXTENSION_20D),
                            int(MAX_TOP_N),
                            int(MACD_EXIT_WINDOW_DAYS), float(MACD_EXIT_ONE_DAY_PCT),
                            float(MACD_EXIT_THREE_DAY_PCT),
                        )
                        result = refresh_state_positions(comparison, stock_dict, data_date)
                        atomic_write_csv(result, HISTORY_FILE)
                        complete_task()
                        st.success(
                            f"已保存{mothers['Event_ID'].nunique()}个正式母样本、{len(result)}条五路径记录。"
                            "下载或页面重跑不会清空结果。"
                        )
        except Exception as exc:
            current_task = read_json_safe(RUN_TASK_FILE)
            failures = int(current_task.get("error_count", 0)) + 1
            current_task.update(
                {
                    "state": "PAUSED_ERROR" if failures >= 3 else "RUNNING",
                    "error_count": failures,
                    "last_error": str(exc)[:500],
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            atomic_write_json(current_task, RUN_TASK_FILE)
            st.error("本次连接或运行中断；已完成分片和周次均已保存。")
            st.exception(exc)
            if AUTO_RESUME and failures < 3:
                st.info(f"自动重试 {failures}/3，将从断点继续。")
                time.sleep(2.0)
                st.rerun()


# 报告与下载始终从磁盘加载，和运行按钮、自动重跑状态完全解耦。
if RUN_MODE == "历史MACD路径回测":
    saved_ledger = read_csv_safe(HISTORY_LEDGER_FILE)
    saved_mothers = read_csv_safe(FORMAL_HISTORY_FILE)
    if not saved_ledger.empty:
        st.subheader("📦 正式Top 5选出数量审计")
        scan_status = saved_ledger.get(
            "Scan_Status", pd.Series("COMPLETED", index=saved_ledger.index)
        ).fillna("COMPLETED")
        complete_ledger = saved_ledger[scan_status == "COMPLETED"]
        skipped_ledger = saved_ledger[scan_status == "SKIPPED_INCOMPLETE"]
        selected_count = pd.to_numeric(complete_ledger.get("Selected_Count"), errors="coerce")
        short = complete_ledger[selected_count < int(MAX_TOP_N)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("已扫描完整周", len(complete_ledger))
        c2.metric(f"选满Top {int(MAX_TOP_N)}周", len(complete_ledger) - len(short))
        c3.metric("正式信号不足周", len(short))
        c4.metric("行情不完整跳过", len(skipped_ledger))
        if not short.empty:
            st.caption("不足来自正式信号总数或次日不可成交；日线路径只在冻结Top N内部研究，不改变母样本。")
            st.dataframe(short.sort_values("Trade_Date"), width="stretch")
        if not skipped_ledger.empty:
            st.warning("下列周次因行情覆盖不足被跳过，没有被错误记成零信号。")
            st.dataframe(skipped_ledger.sort_values("Trade_Date"), width="stretch")

    saved_result = read_csv_safe(HISTORY_FILE)
    if not saved_result.empty:
        st.markdown("---")
        st.caption("以下报告来自最近一次成功原子保存的结果；下载触发页面重跑后仍会保留。")
        display_state_validation_report(saved_result)
        display_lifecycle_report(saved_result, "📈 五路径生命周期明细")
        st.download_button(
            "📥 一键下载V14.10全部回测数据",
            build_all_download_zip(saved_result, saved_mothers, saved_ledger),
            file_name="skdj_v14_10_all_backtest_data.zip",
            mime="application/zip",
            key="download_v1410_all",
        )
        st.download_button(
            "📥 仅下载五路径流水CSV",
            saved_result.to_csv(index=False).encode("utf-8-sig"),
            file_name="skdj_v14_10_path_validation_export.csv",
            mime="text/csv",
            key="download_v1410_history",
        )
    elif not saved_mothers.empty:
        st.info(f"已保存{len(saved_mothers)}条正式母样本；自动任务会继续生成五路径结果。")
else:
    saved_daily = read_csv_safe(DAILY_POSITION_FILE)
    if not saved_daily.empty:
        st.markdown("---")
        recent_dates = sorted(
            {
                date
                for column in ("Original_Signal_Date", "Signal_Date", "Buy_Date", "Exit_Date")
                if column in saved_daily.columns
                for date in saved_daily[column].map(parse_yyyymmdd).dropna()
            }
        )[-5:]
        display_state_validation_report(saved_daily)
        display_lifecycle_report(saved_daily, "📡 最近5个交易日状态", recent_trade_dates=recent_dates)
        st.download_button(
            "📥 下载V14.10每日状态",
            saved_daily.to_csv(index=False).encode("utf-8-sig"),
            file_name="skdj_v14_10_daily_paths.csv",
            mime="text/csv",
            key="download_v1410_daily",
        )
