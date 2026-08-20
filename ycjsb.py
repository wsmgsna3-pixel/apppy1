# -*- coding: utf-8 -*-
"""周线SKDJ N=6/7 K线上穿25快速审计 V4.8。

本版只研究K线从25下方向上穿过25，不要求此前低位金叉，也不要求K>D。
同一份行情同时计算N=6和N=7，暂停机器学习、评分、TopK与三仓组合。
正式信号窗口默认250个交易日，开始前只保留30周指标预热，信号后观察W1-W8。
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import math
import os
import pickle
import shutil
import time
import zipfile
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts

TITLE = "周线SKDJ N=6/7上穿25快速审计 V4.8"
VERSION = "V4.8-WEEKLY-SKDJ-N6-N7-K-CROSS-25-NO-ML"
UI_PATCH = "V4.8.4-RUNTIME-STABILITY"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# 沿用旧行情缓存目录，以便直接复用V4.7已经下载的更长历史数据。
PRICE_CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")
CHECKPOINT_DIR = os.path.join(APP_DIR, "weekly_skdj_v4_8_checkpoints")
RESULT_DIR = os.path.join(APP_DIR, "weekly_skdj_v4_8_results")
JOB_DIR = os.path.join(APP_DIR, "weekly_skdj_v4_8_jobs")

SKDJ_NS = (6, 7)
SKDJ_M = 3
SKDJ_BOTTOM = 25.0
WARMUP_WEEKS = 30
AUDIT_WEEKS = 8
UI_HEARTBEAT_SECONDS = 5.0
FIRST_HIT_LEVELS = (10.0, 15.0, 20.0)

CORE_TECH_L1 = {"电子", "计算机", "通信", "国防军工"}
EXTENDED_TECH_L1 = {"机械设备", "电力设备", "医药生物", "汽车", "基础化工", "有色金属", "建筑材料"}
TECH_INDUSTRY_KEYWORDS = {
    "半导体", "电子元件", "元件", "光学光电子", "消费电子", "电子化学品",
    "计算机设备", "软件开发", "IT服务", "通信设备", "军工电子", "航空装备",
    "航天装备", "自动化设备", "机器人", "激光设备", "工控设备", "仪器仪表",
    "电池", "光伏设备", "风电设备", "电网设备", "电机", "医疗器械",
    "生物制品", "汽车电子", "金属新材料", "非金属材料", "膜材料", "碳纤维",
}
BOARDS = ("主板", "创业板", "科创板")

pro = None
API_ERRORS: list[str] = []


def normalize_date(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip().replace("-", "").replace("/", "")
    if text.endswith(".0"):
        text = text[:-2]
    return text[:8] if len(text) >= 8 and text[:8].isdigit() else default


def finite_num(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def true_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(to_bool)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def record_error(message: str) -> None:
    if len(API_ERRORS) < 500:
        API_ERRORS.append(message)


def safe_get(func_name: str, retries: int = 3, required: bool = False, **kwargs) -> pd.DataFrame:
    global pro
    if pro is None:
        if required:
            raise RuntimeError("Tushare尚未初始化")
        return pd.DataFrame()
    try:
        func = getattr(pro, func_name)
    except AttributeError as exc:
        if required:
            raise RuntimeError(f"当前Tushare SDK不支持{func_name}") from exc
        record_error(f"缺少接口 {func_name}")
        return pd.DataFrame()
    last_error = None
    for attempt in range(retries):
        try:
            result = func(**kwargs)
            return pd.DataFrame() if result is None else result
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    message = f"{func_name}失败: {last_error}"
    record_error(message)
    if required:
        raise RuntimeError(message)
    return pd.DataFrame()


def atomic_pickle(payload: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    with open(temp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def atomic_bytes(payload: bytes, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    with open(temp, "wb") as handle:
        handle.write(payload)
    os.replace(temp, path)


def stable_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def configured_tushare_token() -> str:
    try:
        return str(st.secrets.get("TUSHARE_TOKEN", "")).strip()
    except Exception:
        return ""


def active_job_path(signature: str) -> str:
    return os.path.join(JOB_DIR, f"{signature}.active")


def mark_job_active(signature: str) -> None:
    atomic_bytes(json.dumps({
        "signature": signature, "version": VERSION,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }, ensure_ascii=False).encode("utf-8"), active_job_path(signature))


def clear_job_active(signature: str) -> None:
    path = active_job_path(signature)
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        record_error(f"任务标记清除失败: {exc}")


def is_job_active(signature: str) -> bool:
    return os.path.exists(active_job_path(signature))


def checkpoint_path(signature: str, ts_code: str) -> str:
    return os.path.join(
        CHECKPOINT_DIR, signature, f"{str(ts_code).replace('.', '_')}.pkl")


def load_checkpoint(signature: str, ts_code: str) -> dict[str, Any] | None:
    path = checkpoint_path(signature, ts_code)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if (not isinstance(payload, dict)
                or payload.get("signature") != signature
                or payload.get("ts_code") != str(ts_code)
                or "events" not in payload or "rejects" not in payload):
            return None
        return payload
    except Exception as exc:
        record_error(f"检查点损坏 {ts_code}: {exc}")
        return None


def save_checkpoint(signature: str, ts_code: str,
                    events: list[dict[str, Any]], rejects: dict[str, int]) -> None:
    atomic_pickle({
        "signature": signature, "ts_code": str(ts_code),
        "events": events, "rejects": rejects,
    }, checkpoint_path(signature, ts_code))


def merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(files: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in files.items():
            archive.writestr(name, csv_bytes(frame))
    return buffer.getvalue()


def render_plain_table(frame: pd.DataFrame, max_rows: int = 200) -> None:
    """Render without st.dataframe's separately loaded frontend module."""
    if frame.empty:
        st.caption("无数据")
        return
    shown = frame.head(int(max_rows)).copy()
    table_html = shown.to_html(
        index=False, border=0, na_rep="", float_format=lambda value: f"{value:.4f}")
    st.markdown(
        "<div style='overflow-x:auto;font-size:0.88rem'>"
        + table_html + "</div>", unsafe_allow_html=True)
    if len(frame) > len(shown):
        st.caption(f"页面仅显示前{len(shown)}行；完整内容在结果ZIP中。")


def render_download(payload: bytes, filename: str, key: str) -> None:
    """Use Streamlit's proven in-memory download path; no static config needed."""
    st.download_button(
        "一键下载全部研究结果（ZIP）",
        data=payload,
        file_name=filename,
        mime="application/zip",
        type="primary",
        key=key,
        on_click="ignore",
    )
    st.caption(f"结果大小：{len(payload) / 1024 / 1024:.2f} MB。")


@st.cache_data(ttl=24 * 3600)
def load_trade_calendar(start_date: str, end_date: str) -> list[str]:
    frame = safe_get(
        "trade_cal", required=True, exchange="SSE",
        start_date=start_date, end_date=end_date)
    if frame.empty:
        raise RuntimeError("交易日历为空")
    return sorted(frame.loc[frame["is_open"].eq(1), "cal_date"].astype(str).tolist())


@st.cache_data(ttl=24 * 3600)
def load_stock_basic() -> pd.DataFrame:
    frames = []
    fields = "ts_code,symbol,name,market,exchange,list_status,list_date,delist_date"
    for status in ("L", "P", "D"):
        frame = safe_get("stock_basic", list_status=status, fields=fields)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("stock_basic加载失败")
    result = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
    result = result[
        result["market"].isin(BOARDS)
        & result["exchange"].ne("BSE")
        & ~result["ts_code"].astype(str).str.endswith(".BJ", na=False)
        & ~result["name"].astype(str).str.contains("ST|退", na=False)
    ].copy()
    result["list_date"] = result["list_date"].map(lambda x: normalize_date(x, "19000101"))
    result["delist_date"] = result["delist_date"].map(lambda x: normalize_date(x, "99991231"))
    return result


def is_tech_industry(row: pd.Series) -> bool:
    l1 = str(row.get("l1_name", ""))
    l2 = str(row.get("l2_name", ""))
    l3 = str(row.get("l3_name", ""))
    if l1 in CORE_TECH_L1:
        return True
    return l1 in EXTENDED_TECH_L1 and any(
        keyword in f"{l2}|{l3}" for keyword in TECH_INDUSTRY_KEYWORDS)


@st.cache_data(ttl=7 * 24 * 3600)
def load_tech_memberships(api_pause: float) -> pd.DataFrame:
    levels = safe_get("index_classify", required=True, level="L1", src="SW2021")
    targets = levels[levels["industry_name"].isin(CORE_TECH_L1 | EXTENDED_TECH_L1)]
    if targets.empty:
        raise RuntimeError("未找到申万2021目标行业")
    frames = []
    jobs = [(str(row.index_code), flag)
            for row in targets.itertuples(index=False) for flag in ("Y", "N")]
    for code, flag in jobs:
        frame = safe_get("index_member_all", l1_code=code, is_new=flag)
        if not frame.empty:
            if "ts_code" not in frame.columns and "con_code" in frame.columns:
                frame = frame.rename(columns={"con_code": "ts_code"})
            frames.append(frame)
        time.sleep(api_pause)
    if not frames:
        raise RuntimeError("index_member_all未返回数据")
    result = pd.concat(frames, ignore_index=True)
    for column in ("ts_code", "l1_name", "l2_name", "l3_name", "in_date", "out_date"):
        if column not in result.columns:
            result[column] = ""
    result = result[result.apply(is_tech_industry, axis=1)].copy()
    result["in_date"] = result["in_date"].map(lambda x: normalize_date(x, "19000101"))
    result["out_date"] = result["out_date"].map(lambda x: normalize_date(x, "99991231"))
    return result.drop_duplicates(
        ["ts_code", "l1_name", "l2_name", "l3_name", "in_date", "out_date"])


def build_period_index(memberships: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in memberships.itertuples(index=False):
        result.setdefault(str(row.ts_code), []).append({
            "in_date": str(row.in_date), "out_date": str(row.out_date),
            "l1": str(row.l1_name), "l2": str(row.l2_name), "l3": str(row.l3_name),
        })
    return result


def membership_on_date(periods: list[dict[str, str]], trade_date: str) -> dict[str, str] | None:
    for period in periods:
        if period["in_date"] <= trade_date < period["out_date"]:
            return period
    return None


def periods_overlap(periods: list[dict[str, str]], start_date: str, end_date: str) -> bool:
    return any(period["in_date"] <= end_date and period["out_date"] > start_date
               for period in periods)


def trailing_signal_start(open_dates: list[str], signal_end: str, days: int) -> str:
    eligible = [item for item in open_dates if item <= signal_end]
    if len(eligible) < int(days):
        raise ValueError(f"交易日历不足{int(days)}日")
    return eligible[-int(days)]


def complete_week_last_dates(open_dates: list[str]) -> dict[pd.Timestamp, str]:
    frame = pd.DataFrame({"trade_date": open_dates})
    frame["dt"] = pd.to_datetime(frame["trade_date"])
    frame["week_label"] = frame["dt"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    return frame.groupby("week_label")["trade_date"].max().to_dict()


def market_week_sequence(open_dates: list[str]) -> list[tuple[pd.Period, str]]:
    frame = pd.DataFrame({"trade_date": open_dates})
    frame["period"] = pd.to_datetime(frame["trade_date"]).dt.to_period("W-FRI")
    return [(period, str(group["trade_date"].max()))
            for period, group in frame.groupby("period", sort=True)]


def cache_path(ts_code: str, start_date: str, end_date: str) -> str:
    return os.path.join(
        PRICE_CACHE_DIR,
        f"{str(ts_code).replace('.', '_')}_{start_date}_{end_date}.pkl")


def normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    for column in ("open", "high", "low", "close", "vol"):
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    return (work.dropna(subset=["trade_date", "open", "high", "low", "close"])
            .drop_duplicates("trade_date", keep="last")
            .sort_values("trade_date").reset_index(drop=True))


def normalize_basic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    for column in ("close", "circ_mv", "turnover_rate"):
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    return work.drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)


def load_covering_cache(ts_code: str, start_date: str, end_date: str
                        ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    prefix = str(ts_code).replace(".", "_")
    exact = cache_path(ts_code, start_date, end_date)
    candidates = [exact] + sorted(
        glob.glob(os.path.join(PRICE_CACHE_DIR, f"{prefix}_*.pkl")), reverse=True)
    seen: set[str] = set()
    for path in candidates:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        name = os.path.basename(path)
        stem = name[:-4] if name.endswith(".pkl") else name
        tail = stem[len(prefix) + 1:]
        pieces = tail.rsplit("_", 1)
        if len(pieces) != 2:
            continue
        cached_start, cached_end = pieces
        if cached_start > start_date or cached_end < end_date:
            continue
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            daily = normalize_price_frame(payload.get("daily", pd.DataFrame()))
            basic = normalize_basic_frame(payload.get("basic", pd.DataFrame()))
            daily = daily[daily["trade_date"].between(start_date, end_date)].copy()
            if not basic.empty:
                basic = basic[basic["trade_date"].between(start_date, end_date)].copy()
            if not daily.empty:
                return daily, basic, path
        except Exception as exc:
            record_error(f"行情缓存损坏 {ts_code}: {exc}")
    return pd.DataFrame(), pd.DataFrame(), exact


def fetch_price(ts_code: str, start_date: str, end_date: str,
                use_cache: bool, api_pause: float
                ) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    if use_cache:
        daily, basic, source_path = load_covering_cache(ts_code, start_date, end_date)
        if not daily.empty:
            return daily, basic, source_path, True
    last_error = None
    daily = pd.DataFrame()
    for attempt in range(3):
        try:
            result = ts.pro_bar(
                api=pro, ts_code=ts_code, start_date=start_date, end_date=end_date,
                adj="qfq", freq="D", factors=["tor"])
            daily = pd.DataFrame() if result is None else result
            break
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    if daily.empty:
        record_error(f"pro_bar {ts_code}失败: {last_error}")
        return pd.DataFrame(), pd.DataFrame(), cache_path(ts_code, start_date, end_date), False
    daily = normalize_price_frame(daily)
    time.sleep(api_pause)
    path = cache_path(ts_code, start_date, end_date)
    if use_cache and not daily.empty:
        atomic_pickle({"daily": daily, "basic": pd.DataFrame()}, path)
    return daily, pd.DataFrame(), path, False


def ensure_daily_basic(ts_code: str, start_date: str, end_date: str,
                       daily: pd.DataFrame, cached_basic: pd.DataFrame,
                       storage_path: str, use_cache: bool, api_pause: float
                       ) -> pd.DataFrame:
    if not cached_basic.empty:
        return cached_basic
    basic = safe_get(
        "daily_basic", ts_code=ts_code, start_date=start_date, end_date=end_date,
        fields="ts_code,trade_date,close,circ_mv,turnover_rate")
    basic = normalize_basic_frame(basic)
    time.sleep(api_pause)
    if use_cache and not daily.empty and not basic.empty:
        # 写入本版精确区间缓存，不覆盖可能来自旧版的长区间文件。
        atomic_pickle({"daily": daily, "basic": basic}, cache_path(ts_code, start_date, end_date))
    return basic


def aggregate_complete_weekly(daily: pd.DataFrame,
                              week_last_map: dict[pd.Timestamp, str]) -> pd.DataFrame:
    work = daily.copy()
    work["dt"] = pd.to_datetime(work["trade_date"])
    weekly = (work.set_index("dt").resample("W-FRI").agg({
        "trade_date": "last", "open": "first", "high": "max", "low": "min",
        "close": "last", "vol": "sum",
    }).dropna(subset=["close"]).reset_index().rename(columns={"dt": "week_label"}))
    weekly["calendar_week_last"] = weekly["week_label"].map(week_last_map)
    return weekly[
        weekly["calendar_week_last"].notna()
        & weekly["trade_date"].astype(str).eq(weekly["calendar_week_last"].astype(str))
    ].copy().reset_index(drop=True)


def add_skdj(weekly: pd.DataFrame, n: int) -> pd.DataFrame:
    work = weekly.copy()
    lowv = work["low"].rolling(int(n), min_periods=int(n)).min()
    highv = work["high"].rolling(int(n), min_periods=int(n)).max()
    raw = (work["close"] - lowv) / (highv - lowv).replace(0, np.nan) * 100.0
    rsv = raw.ewm(span=SKDJ_M, adjust=False, min_periods=1).mean()
    work["K"] = rsv.ewm(span=SKDJ_M, adjust=False, min_periods=1).mean()
    work["D"] = work["K"].rolling(SKDJ_M, min_periods=SKDJ_M).mean()
    work["K_Change_1W"] = work["K"].diff()
    work["KD_Spread"] = work["K"] - work["D"]
    work["K_Cross_25"] = work["K"].ge(SKDJ_BOTTOM) & work["K"].shift(1).lt(SKDJ_BOTTOM)
    return work


def market_snapshot(basic: pd.DataFrame, trade_date: str) -> dict[str, float]:
    history = basic[basic["trade_date"].astype(str).le(trade_date)]
    if history.empty:
        return {"Raw_Close": np.nan, "Circ_MV_Billion": np.nan, "Turnover_Rate": np.nan}
    row = history.iloc[-1]
    return {
        "Raw_Close": finite_num(row.get("close")),
        "Circ_MV_Billion": finite_num(row.get("circ_mv")) / 10000.0,
        "Turnover_Rate": finite_num(row.get("turnover_rate")),
    }


def first_hit_label(path: pd.DataFrame, entry: float, target_pct: float) -> str:
    upper, lower = entry * (1 + target_pct / 100.0), entry * 0.90
    for row in path.itertuples(index=False):
        hit_up = finite_num(getattr(row, "high", np.nan)) >= upper
        hit_down = finite_num(getattr(row, "low", np.nan)) <= lower
        if hit_up and hit_down:
            return "同日同时触发_保守按-10%先"
        if hit_down:
            return "先到-10%"
        if hit_up:
            return f"先到+{int(target_pct)}%"
    return "均未触发"


def entry_outcomes(daily: pd.DataFrame, signal_date: str, ts_code: str,
                   open_dates: list[str], open_pos: dict[str, int],
                   market_weeks: list[tuple[pd.Period, str]], config: dict[str, Any]
                   ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "Tradable": False, "Date": "", "Raw_Open": np.nan,
        "Reason": "",
    }
    for week in range(1, AUDIT_WEEKS + 1):
        out.update({
            f"Has_W{week}": False, f"W{week}_End_Date": "",
            f"W{week}_MFE_Net_pct": np.nan, f"W{week}_MAE_Raw_pct": np.nan,
            f"W{week}_Close_Return_Net_pct": np.nan,
        })
    if signal_date not in open_pos or open_pos[signal_date] + 1 >= len(open_dates):
        out["Reason"] = "未来市场交易日不足"
        return out
    entry_date = open_dates[open_pos[signal_date] + 1]
    row = daily[daily["trade_date"].astype(str).eq(entry_date)]
    if row.empty:
        out["Reason"] = "下一市场交易日停牌"
        return out
    first = row.iloc[-1]
    raw_entry = finite_num(first.get("open"))
    if not math.isfinite(raw_entry) or raw_entry <= 0:
        out["Reason"] = "开盘价无效"
        return out
    if str(ts_code).startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        if float(first["open"]) == float(first["high"]) == float(first["low"]):
            out["Reason"] = "主板下一交易日一字板"
            return out
    buy_cost = (config["commission_pct"] + config["transfer_fee_pct"]) / 100.0
    sell_cost = (config["commission_pct"] + config["transfer_fee_pct"] + config["stamp_duty_pct"]) / 100.0
    net_entry = raw_entry * (1 + config["buy_slippage_pct"] / 100.0) * (1 + buy_cost)
    sell_factor = (1 - config["sell_slippage_pct"] / 100.0) * (1 - sell_cost)
    out.update({"Tradable": True, "Date": entry_date, "Raw_Open": raw_entry})
    entry_period = pd.Timestamp(entry_date).to_period("W-FRI")
    future_weeks = [(period, end) for period, end in market_weeks if period >= entry_period]
    for week in range(1, AUDIT_WEEKS + 1):
        if len(future_weeks) < week:
            continue
        end_date = future_weeks[week - 1][1]
        path = daily[daily["trade_date"].astype(str).between(entry_date, end_date)]
        if path.empty:
            continue
        high = finite_num(path["high"].max())
        low = finite_num(path["low"].min())
        close = finite_num(path.iloc[-1]["close"])
        out.update({
            f"Has_W{week}": True, f"W{week}_End_Date": end_date,
            f"W{week}_MFE_Net_pct": (high * sell_factor / net_entry - 1) * 100,
            f"W{week}_MAE_Raw_pct": (low / raw_entry - 1) * 100,
            f"W{week}_Close_Return_Net_pct": (close * sell_factor / net_entry - 1) * 100,
        })
        if week == AUDIT_WEEKS:
            for level in FIRST_HIT_LEVELS:
                out[f"First_Hit_{int(level)}_vs_Minus10_W8"] = first_hit_label(
                    path, raw_entry, level)
    return out


def analyze_stock(stock: pd.Series, periods: list[dict[str, str]], daily: pd.DataFrame,
                  cached_basic: pd.DataFrame, storage_path: str,
                  week_last_map: dict[pd.Timestamp, str], open_dates: list[str],
                  open_pos: dict[str, int], market_weeks: list[tuple[pd.Period, str]],
                  config: dict[str, Any], use_cache: bool, api_pause: float
                  ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rejects: dict[str, int] = {}
    weekly_base = aggregate_complete_weekly(daily, week_last_map)
    crosses: list[tuple[int, pd.Series]] = []
    for n in SKDJ_NS:
        indicator = add_skdj(weekly_base, n)
        selected = indicator[
            indicator["K_Cross_25"]
            & indicator["trade_date"].astype(str).between(
                config["signal_start"], config["signal_end"])
        ]
        crosses.extend((n, row) for _, row in selected.iterrows())
    if not crosses:
        return [], rejects

    code = str(stock["ts_code"])
    basic = ensure_daily_basic(
        code, config["data_start"], config["market_end"], daily, cached_basic,
        storage_path, use_cache, api_pause)
    if basic.empty:
        rejects["出现K上穿25但daily_basic缺失"] = len(crosses)
        return [], rejects

    rows: list[dict[str, Any]] = []
    for n, signal in crosses:
        signal_date = str(signal["trade_date"])
        membership = membership_on_date(periods, signal_date)
        snapshot = market_snapshot(basic, signal_date)
        reason = ""
        if membership is None:
            reason = "信号日不在历史科技池"
        elif not (str(stock["list_date"]) <= signal_date < str(stock["delist_date"])):
            reason = "信号日上市状态无效"
        elif not math.isfinite(snapshot["Raw_Close"]) or snapshot["Raw_Close"] < config["min_price"]:
            reason = "信号日股价不足"
        elif (not math.isfinite(snapshot["Circ_MV_Billion"])
              or snapshot["Circ_MV_Billion"] < config["min_mv"]):
            reason = "信号日流通市值不足"
        if reason:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        outcome = entry_outcomes(
            daily, signal_date, code, open_dates, open_pos, market_weeks, config)
        row = {
            "ts_code": code, "name": str(stock["name"]),
            "SKDJ_N": n, "SKDJ_M": SKDJ_M, "Signal_Date": signal_date,
            "Signal_K": finite_num(signal["K"]), "Signal_D": finite_num(signal["D"]),
            "Signal_KD_Spread": finite_num(signal["KD_Spread"]),
            "Signal_K_Change_1W": finite_num(signal["K_Change_1W"]),
            "Signal_K_Above_D": finite_num(signal["K"]) > finite_num(signal["D"]),
            "SW_L1": membership["l1"], "SW_L2": membership["l2"], "SW_L3": membership["l3"],
            **snapshot,
        }
        row.update({f"Entry_{key}": value for key, value in outcome.items()})
        rows.append(row)
    return rows, rejects


def mature_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events[
        true_mask(events, "Entry_Tradable")
        & true_mask(events, f"Entry_Has_W{AUDIT_WEEKS}")
    ].copy()


def max_empty_run(series: pd.Series) -> int:
    longest = current = 0
    for value in series.tolist():
        current = current + 1 if int(value) == 0 else 0
        longest = max(longest, current)
    return longest


def signal_calendar(open_dates: list[str], start_date: str, end_date: str,
                    events: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"trade_date": [d for d in open_dates if start_date <= d <= end_date]})
    frame["week"] = pd.to_datetime(frame["trade_date"]).dt.to_period("W-FRI")
    calendar = frame.groupby("week")["trade_date"].max().rename("Week_End").reset_index()
    event_work = events.copy()
    event_work["week"] = pd.to_datetime(
        event_work["Signal_Date"].astype(str), format="%Y%m%d").dt.to_period("W-FRI")
    for n in SKDJ_NS:
        counts = event_work[event_work["SKDJ_N"].eq(n)].groupby("week").size()
        calendar[f"N{n}_Signals"] = calendar["week"].map(counts).fillna(0).astype(int)
    calendar["Either_N_Signals"] = calendar[[f"N{n}_Signals" for n in SKDJ_NS]].max(axis=1)
    calendar["week"] = calendar["week"].astype(str)
    return calendar


def behavior_audit(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base = events[events["SKDJ_N"].eq(n)]
        groups = [
            ("全部K上穿25", base),
            ("上穿时K>D", base[true_mask(base, "Signal_K_Above_D")]),
            ("上穿时K≤D", base[~true_mask(base, "Signal_K_Above_D")]),
        ]
        for group_name, selected in groups:
            row: dict[str, Any] = {
                "SKDJ_N": n, "分组": group_name, "事件数": len(selected),
                "不同股票": selected["ts_code"].nunique() if not selected.empty else 0,
                "信号日": selected["Signal_Date"].nunique() if not selected.empty else 0,
            }
            for week in (1, 2, 3, 5, 8):
                mfe = numeric(selected, f"Entry_W{week}_MFE_Net_pct")
                mae = numeric(selected, f"Entry_W{week}_MAE_Raw_pct")
                close_ret = numeric(selected, f"Entry_W{week}_Close_Return_Net_pct")
                row.update({
                    f"W{week}最大浮盈均值%": mfe.mean(),
                    f"W{week}最大浮盈中位%": mfe.median(),
                    f"W{week}达到10%比例%": mfe.ge(10).mean() * 100,
                    f"W{week}达到20%比例%": mfe.ge(20).mean() * 100,
                    f"W{week}收盘平均净收益%": close_ret.mean(),
                    f"W{week}收盘中位净收益%": close_ret.median(),
                    f"W{week}收盘胜率%": close_ret.gt(0).mean() * 100,
                    f"W{week}触及-10%比例%": mae.le(-10).mean() * 100,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def first_hit_audit(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in SKDJ_NS:
        group = events[events["SKDJ_N"].eq(n)]
        row: dict[str, Any] = {"SKDJ_N": n, "事件数": len(group)}
        for level in FIRST_HIT_LEVELS:
            values = group.get(
                f"Entry_First_Hit_{int(level)}_vs_Minus10_W8",
                pd.Series(index=group.index, dtype=str)).astype(str)
            row[f"先到+{int(level)}比例%"] = values.eq(f"先到+{int(level)}%").mean() * 100
            row[f"先到-10比例%_对比{int(level)}"] = values.eq("先到-10%").mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def pair_signals(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n6 = events[events["SKDJ_N"].eq(6)].copy()
    n7 = events[events["SKDJ_N"].eq(7)].copy()
    n7_groups = {code: group.copy() for code, group in n7.groupby("ts_code")}
    details = []
    for row in n6.itertuples(index=False):
        candidates = n7_groups.get(str(row.ts_code))
        if candidates is None or candidates.empty:
            continue
        date6 = pd.to_datetime(str(row.Signal_Date), format="%Y%m%d")
        dates7 = pd.to_datetime(candidates["Signal_Date"].astype(str), format="%Y%m%d")
        gaps = (dates7 - date6).dt.days
        eligible = gaps.abs().le(56)
        if not eligible.any():
            continue
        nearest_index = gaps[eligible].abs().idxmin()
        nearest = candidates.loc[nearest_index]
        details.append({
            "ts_code": row.ts_code, "name": row.name,
            "N6_Signal_Date": str(row.Signal_Date),
            "N7_Signal_Date": str(nearest["Signal_Date"]),
            "N7_Minus_N6_Calendar_Days": int(gaps.loc[nearest_index]),
            "Exact_Same_Date": int(gaps.loc[nearest_index]) == 0,
        })
    detail = pd.DataFrame(details)
    summary = pd.DataFrame([{
        "N6事件": len(n6), "N7事件": len(n7),
        "N6与N7同股票8周内匹配": len(detail),
        "完全同日": int(true_mask(detail, "Exact_Same_Date").sum()) if not detail.empty else 0,
        "N6更早比例%": numeric(detail, "N7_Minus_N6_Calendar_Days").gt(0).mean() * 100 if not detail.empty else np.nan,
        "N7更早比例%": numeric(detail, "N7_Minus_N6_Calendar_Days").lt(0).mean() * 100 if not detail.empty else np.nan,
        "N7减N6天数中位数": numeric(detail, "N7_Minus_N6_Calendar_Days").median() if not detail.empty else np.nan,
    }])
    return summary, detail


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    streamlit_version = str(getattr(st, "__version__", "unknown"))
    st.caption(
        f"{UI_PATCH}｜只验证K线上穿25；N=6和N=7同场计算；"
        f"机器学习、评分、TopK和三仓已暂停。｜Streamlit {streamlit_version}")
    if streamlit_version.startswith("1.62"):
        st.error(
            "检测到Streamlit 1.62.x。该环境与本次约32秒断线重连日志一致；"
            "请同时使用本版requirements.txt锁定到1.61.0后再运行。")
    with st.expander("本版口径", expanded=True):
        st.markdown(f"""
- **信号**：上一完整周K＜25，本完整周K≥25；不要求低位金叉，不要求K>D。
- **重复信号**：同一股票以后再次跌回25下方并重新上穿25，会再次计为新事件。
- **买入**：信号完整周结束后的下一市场交易日开盘。
- **数据长度**：默认正式窗口250个交易日（约52周）＋开始前{WARMUP_WEEKS}周预热＋截止后W1-W8观察；通常约90～100周，而不是120周或290周。
- **过滤**：每个历史信号日分别检查当时科技行业归属、股价≥10元、流通市值≥100亿元，避免使用今天状态回看历史。
- **比较**：同一次运行同时计算N=6和N=7，比较信号覆盖、W1-W8爆发力、持续性、回撤和先到止盈/止损。
""")
    with st.sidebar:
        st.header("运行参数")
        backtest_days = st.number_input(
            "回测交易日数", 100, 1000, 250, 50, key="v48_days")
        signal_end_date = st.date_input(
            "买入信号截止", date(2026, 6, 5), key="v48_signal_end")
        market_end_date = st.date_input(
            "行情观察截止", date.today(), key="v48_market_end")
        pause = st.number_input(
            "接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v48_pause")
        use_cache = st.checkbox("复用行情缓存", True, key="v48_cache")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005,
            format="%.3f", key="v48_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01,
            format="%.3f", key="v48_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001,
            format="%.3f", key="v48_transfer")
        if st.button("清除V4.8检查点和结果", key="v48_clear"):
            shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
            shutil.rmtree(RESULT_DIR, ignore_errors=True)
            shutil.rmtree(JOB_DIR, ignore_errors=True)
            st.success("V4.8检查点和结果已清除；旧行情缓存保留")

    request_payload = {
        "version": VERSION, "days": int(backtest_days),
        "signal_end": signal_end_date.strftime("%Y%m%d"),
        "market_end": market_end_date.strftime("%Y%m%d"),
        "commission": float(commission_pct), "stamp": float(stamp_duty_pct),
        "transfer": float(transfer_fee_pct),
    }
    request_signature = stable_signature(request_payload)
    result_path = os.path.join(RESULT_DIR, f"{request_signature}.zip")
    result_name = f"weekly_skdj_n6_n7_k_cross25_fast_v4_8_{int(backtest_days)}d.zip"
    completed_available = False
    if os.path.exists(result_path):
        try:
            with open(result_path, "rb") as handle:
                saved_result = handle.read()
            completed_available = True
            clear_job_active(request_signature)
            st.success("发现相同参数的已完成结果，可直接下载。")
            render_download(
                saved_result, result_name, f"v48_saved_{request_signature}")
        except Exception as exc:
            st.warning(f"旧结果读取失败：{exc}")

    secret_token = configured_tushare_token()
    if secret_token:
        token = secret_token
        st.caption("已从Streamlit Secrets读取TUSHARE_TOKEN。")
    else:
        token = st.text_input("Tushare Token", type="password", key="v48_token")

    job_active = is_job_active(request_signature)
    left, right = st.columns(2)
    with left:
        start_clicked = st.button("开始/重新运行V4.8", type="primary", key="v48_run")
    with right:
        stop_clicked = st.button("停止自动续跑", disabled=not job_active, key="v48_stop")
    if stop_clicked:
        clear_job_active(request_signature)
        st.success("已停止；逐股票检查点保留。")
        return
    if start_clicked:
        if market_end_date <= signal_end_date:
            st.error("行情观察截止必须晚于信号截止")
            return
        mark_job_active(request_signature)
        job_active = True
    if not token:
        st.info("请输入Token；若任务已经开始，重新输入后会自动续跑。")
        return
    if not job_active:
        st.caption("首次点击开始后，页面重连会自动续跑。" if not completed_available else "如需覆盖结果请点击重新运行。")
        return

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    signal_end = signal_end_date.strftime("%Y%m%d")
    market_end = market_end_date.strftime("%Y%m%d")
    try:
        probe_start = signal_end_date - timedelta(days=int(backtest_days) * 2 + 120)
        probe_dates = load_trade_calendar(probe_start.strftime("%Y%m%d"), signal_end)
        signal_start = trailing_signal_start(probe_dates, signal_end, int(backtest_days))
        signal_start_date = pd.Timestamp(signal_start).date()
    except Exception as exc:
        st.error(f"确定250日窗口失败：{exc}")
        return
    data_start_date = signal_start_date - timedelta(weeks=WARMUP_WEEKS, days=7)
    data_start = data_start_date.strftime("%Y%m%d")
    config = {
        "signal_start": signal_start, "signal_end": signal_end,
        "data_start": data_start, "market_end": market_end,
        "min_price": 10.0, "min_mv": 100.0,
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct),
        "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct),
    }
    run_signature = stable_signature({"version": VERSION, **config})

    try:
        with st.spinner("加载交易日历和历史科技池..."):
            open_dates = load_trade_calendar(data_start, market_end)
            extended_end = (market_end_date + timedelta(days=7)).strftime("%Y%m%d")
            week_last_map = complete_week_last_dates(
                load_trade_calendar(data_start, extended_end))
            market_weeks = market_week_sequence(open_dates)
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return

    period_index = build_period_index(memberships)
    active_codes = {
        code for code, periods in period_index.items()
        if periods_overlap(periods, signal_start, signal_end)
    }
    stocks = stock_basic[stock_basic["ts_code"].isin(active_codes)].copy()
    stocks = stocks[
        ~stocks["list_date"].gt(signal_end)
        & ~stocks["delist_date"].lt(data_start)
    ].sort_values("ts_code").reset_index(drop=True)
    open_pos = {day: position for position, day in enumerate(open_dates)}

    event_rows: list[dict[str, Any]] = []
    rejects: dict[str, int] = {}
    checkpoint_hits = price_cache_hits = failures = 0
    progress, status = st.progress(0.0), st.empty()
    last_update = 0.0
    stopped = False
    for number, stock in stocks.iterrows():
        if not is_job_active(request_signature):
            stopped = True
            break
        code = str(stock["ts_code"])
        checkpoint = load_checkpoint(run_signature, code)
        if checkpoint is not None:
            event_rows.extend(checkpoint["events"])
            merge_counts(rejects, checkpoint["rejects"])
            checkpoint_hits += 1
        else:
            daily, cached_basic, storage_path, cache_hit = fetch_price(
                code, data_start, market_end, bool(use_cache), float(pause))
            price_cache_hits += int(cache_hit)
            if daily.empty:
                failures += 1
            else:
                try:
                    rows, stock_rejects = analyze_stock(
                        stock, period_index.get(code, []), daily, cached_basic,
                        storage_path, week_last_map, open_dates, open_pos,
                        market_weeks, config, bool(use_cache), float(pause))
                    event_rows.extend(rows)
                    merge_counts(rejects, stock_rejects)
                    save_checkpoint(run_signature, code, rows, stock_rejects)
                except Exception as exc:
                    failures += 1
                    record_error(f"逐股票分析失败 {code}: {exc}")
        now = time.monotonic()
        processed = number + 1
        if processed == 1 or now - last_update >= UI_HEARTBEAT_SECONDS or processed == len(stocks):
            progress.progress(
                processed / max(len(stocks), 1),
                text=f"已处理{processed}/{len(stocks)}只股票，最近{code}")
            status.caption(
                f"事件{len(event_rows)}；检查点{checkpoint_hits}；行情缓存{price_cache_hits}；失败{failures}")
            last_update = now
    progress.empty()
    status.empty()
    if stopped:
        st.warning("任务已停止，检查点已保留。")
        return

    events_all = pd.DataFrame(event_rows)
    if events_all.empty:
        st.error("本区间没有生成通过历史时点过滤的K上穿25事件。")
        return
    events_all = events_all.sort_values(["Signal_Date", "SKDJ_N", "ts_code"]).reset_index(drop=True)
    mature = mature_events(events_all)
    if mature.empty:
        st.error("存在信号，但没有未来完整W8的成熟事件。")
        return

    calendar = signal_calendar(open_dates, signal_start, signal_end, events_all)
    behavior = behavior_audit(mature)
    first_hits = first_hit_audit(mature)
    pair_summary, pair_details = pair_signals(mature)
    summary_rows = []
    for n in SKDJ_NS:
        all_n = events_all[events_all["SKDJ_N"].eq(n)]
        mature_n = mature[mature["SKDJ_N"].eq(n)]
        counts = calendar[f"N{n}_Signals"]
        summary_rows.append({
            "SKDJ_N": n, "SKDJ_M": SKDJ_M,
            "全部通过过滤事件": len(all_n), "W8成熟事件": len(mature_n),
            "不同股票": mature_n["ts_code"].nunique(),
            "有信号周": int(counts.gt(0).sum()), "空窗周": int(counts.eq(0).sum()),
            "最长连续空窗周": max_empty_run(counts),
            "每周信号均值": counts.mean(), "单周最多": counts.max(),
        })
    run_summary = pd.DataFrame(summary_rows)
    run_summary.insert(0, "程序版本", VERSION)
    run_summary["正式信号开始"] = signal_start
    run_summary["正式信号截止"] = signal_end
    run_summary["实际行情开始"] = data_start
    run_summary["行情观察截止"] = market_end
    run_summary["处理股票数"] = len(stocks)
    run_summary["检查点恢复"] = checkpoint_hits
    run_summary["行情缓存命中"] = price_cache_hits
    run_summary["失败股票"] = failures

    metadata = pd.DataFrame([
        ("信号", "上一完整周K<25，本完整周K>=25；不要求低位金叉，不要求K>D"),
        ("重复信号", "以后跌回25下方再上穿时重新计为新事件"),
        ("参数", "同一次运行分别计算N=6、N=7；M固定为3"),
        ("数据窗口", f"正式{int(backtest_days)}个交易日；开始前{WARMUP_WEEKS}周预热；截止后观察W1-W8"),
        ("暂停功能", "机器学习、评分、TopK、三仓组合全部暂停"),
        ("历史过滤", "每个信号日使用当时科技行业归属、股价和流通市值"),
        ("买入", "信号完整周结束后的下一市场交易日开盘"),
        ("成本", "买卖0.2%滑点、佣金、过户费；卖出另计印花税"),
        ("运行环境", f"Streamlit {streamlit_version}；运行稳定版要求requirements锁定1.61.0"),
    ], columns=["项目", "说明"])
    files = {
        "01_run_summary_v4_8.csv": run_summary,
        "02_n6_n7_behavior_w1_w8_v4_8.csv": behavior,
        "03_first_hit_profit_vs_stop_v4_8.csv": first_hits,
        "04_weekly_signal_calendar_v4_8.csv": calendar,
        "05_n6_n7_pair_summary_v4_8.csv": pair_summary,
        "06_n6_n7_pair_details_v4_8.csv": pair_details,
        "07_all_mature_events_v4_8.csv": mature,
        "08_all_filtered_events_including_immature_v4_8.csv": events_all,
        "09_rejection_audit_v4_8.csv": pd.DataFrame(
            [{"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]),
        "10_api_errors_v4_8.csv": pd.DataFrame({"错误": API_ERRORS}),
        "11_metadata_v4_8.csv": metadata,
    }
    result_zip = make_zip(files)
    try:
        atomic_bytes(result_zip, result_path)
        clear_job_active(request_signature)
        persisted = True
    except Exception as exc:
        persisted = False
        st.warning(f"结果未能持久保存，但当前页面仍可下载：{exc}")

    st.success(
        f"完成：N=6成熟{len(mature[mature['SKDJ_N'].eq(6)])}个，"
        f"N=7成熟{len(mature[mature['SKDJ_N'].eq(7)])}个；"
        f"实际行情约{round((market_end_date - data_start_date).days / 7, 1)}周；"
        f"结果{'已保存' if persisted else '仅当前页面可下载'}。")
    st.subheader("N=6 / N=7运行摘要")
    render_plain_table(run_summary)
    st.subheader("爆发力与持续性")
    render_plain_table(behavior)
    st.subheader("止盈与-10%先后顺序")
    render_plain_table(first_hits)
    st.subheader("N=6与N=7信号领先关系")
    render_plain_table(pair_summary)
    render_download(result_zip, result_name, f"v48_current_{request_signature}")


if __name__ == "__main__":
    main()
