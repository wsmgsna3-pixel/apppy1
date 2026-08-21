# -*- coding: utf-8 -*-
"""周线SKDJ历史完整度与爆发力排名审计 V5.8。

V5.7已经证明双因子能较稳定地识别优秀前20%，但前三名排序仍然分段不稳。
复核发现，V5.4历史H2需要最近3次已完成K/D金叉周期，而旧版正式窗口前只预热
30周，早期事件可能把“历史不足”误当成“历史失败”。

本版冻结V5.7候选池、N=6/7/9、K上穿25、25线下1至5周硬条件、三套旧排名
和四套V5.7排名，不增加新指标、不调权。唯一基础变更是把预热扩展到200周，
并明确导出历史有效周期0/1/2/3次。另增加两套透明对照：完整历史H2，以及
双因子层内完整历史H2；有效周期不足3次者进入PX层，不再按失败次数为0处理。
判卷继续以S级、A/S、前三至少两只A/S和W8最大浮盈为主。
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

TITLE = "周线SKDJ历史完整度与爆发力排名审计 V5.8"
VERSION = "V5.8-WEEKLY-SKDJ-HISTORY-COMPLETENESS-AUDIT"
UI_PATCH = "V5.8-200W-HISTORY-COMPLETE-H2"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# 沿用旧行情缓存目录，以便直接复用V4.7已经下载的更长历史数据。
PRICE_CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")
# 200周预热会改变历史周期字段，不能复用30周预热的逐股票检查点。
CHECKPOINT_DIR = os.path.join(APP_DIR, "weekly_skdj_v5_8_checkpoints")
RESULT_DIR = os.path.join(APP_DIR, "weekly_skdj_v5_8_results")
JOB_DIR = os.path.join(APP_DIR, "weekly_skdj_v5_8_jobs")

SKDJ_NS = (6, 7, 9)
SKDJ_M = 3
SKDJ_BOTTOM = 25.0
WARMUP_WEEKS = 200
RANKING_WEEKS = 8
AUDIT_WEEKS = 12
LIFECYCLE_WEEKS = tuple(range(1, AUDIT_WEEKS + 1))
UI_HEARTBEAT_SECONDS = 5.0
FIRST_HIT_LEVELS = (10.0, 15.0, 20.0, 30.0)
FIRST_HIT_AUDIT_WEEKS = (RANKING_WEEKS, AUDIT_WEEKS)
PAIR_MAX_CALENDAR_DAYS = 84
MAX_BOTTOM_STREAK = 5
RANDOM_DRAWS = 500
RANDOM_SEED = 20260820
LEGACY_MIN_MV = 100.0
ALL_TECH_GROUP = "__ALL_TECH__"
HISTORY_EVENTS = 3
FEATURE_MIN_WEEK_SIZE = 5
FEATURE_PRIORITY_FRACTION = 0.20

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
                or "events" not in payload or "rejects" not in payload
                or "breadth" not in payload):
            return None
        return payload
    except Exception as exc:
        record_error(f"检查点损坏 {ts_code}: {exc}")
        return None


def save_checkpoint(signature: str, ts_code: str,
                    events: list[dict[str, Any]], rejects: dict[str, int],
                    breadth: dict[str, dict[str, float]]) -> None:
    atomic_pickle({
        "signature": signature, "ts_code": str(ts_code),
        "events": events, "rejects": rejects, "breadth": breadth,
    }, checkpoint_path(signature, ts_code))


def merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


BREADTH_SUM_FIELDS = (
    "Constituent_Count", "K_Valid", "K_Rising_Count", "K_Above25_Count",
    "MA20_Valid", "Above_MA20_Count", "MA20_Slope_Valid",
    "MA20_Rising_Count", "Return4W_Valid", "Positive_Return4W_Count",
    "Return4W_Sum", "VolumeRatio_Valid", "VolumeExpand_Count",
)


def merge_breadth(target: dict[str, dict[str, float]],
                  source: dict[str, dict[str, float]]) -> None:
    for key, values in source.items():
        bucket = target.setdefault(str(key), {
            field: 0.0 for field in BREADTH_SUM_FIELDS})
        for field in BREADTH_SUM_FIELDS:
            bucket[field] = float(bucket.get(field, 0.0)) + float(
                values.get(field, 0.0))


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
    for column in ("open", "high", "low", "close", "vol", "amount"):
        if column not in work.columns:
            work[column] = np.nan
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
        "close": "last", "vol": "sum", "amount": "sum",
    }).dropna(subset=["close"]).reset_index().rename(columns={"dt": "week_label"}))
    weekly["calendar_week_last"] = weekly["week_label"].map(week_last_map)
    return weekly[
        weekly["calendar_week_last"].notna()
        & weekly["trade_date"].astype(str).eq(weekly["calendar_week_last"].astype(str))
    ].copy().reset_index(drop=True)


def add_completed_golden_cross_history(work: pd.DataFrame) -> pd.DataFrame:
    """Add peaks of the latest three *completed* K/D golden-cross cycles.

    A cycle starts when K crosses above D and completes at the next K-below-D
    death cross.  For row i only cycles whose death cross is before i are used,
    so an unfinished current cycle and all future K values are excluded.
    """
    result = work.copy()
    k = numeric(result, "K")
    d = numeric(result, "D")
    golden = k.gt(d) & k.shift(1).le(d.shift(1))
    death = k.lt(d) & k.shift(1).ge(d.shift(1))
    golden_positions = np.flatnonzero(golden.fillna(False).to_numpy())
    death_positions = np.flatnonzero(death.fillna(False).to_numpy())
    completed: list[tuple[int, int, float, str, str]] = []
    for start in golden_positions:
        later_deaths = death_positions[death_positions > start]
        if len(later_deaths) == 0:
            continue
        end = int(later_deaths[0])
        peak = finite_num(k.iloc[start:end + 1].max())
        completed.append((
            int(start), end, peak,
            str(result.iloc[start].get("trade_date", "")),
            str(result.iloc[end].get("trade_date", "")),
        ))

    peak1: list[float] = []
    peak2: list[float] = []
    peak3: list[float] = []
    max3: list[float] = []
    reached75: list[float] = []
    valid_count: list[int] = []
    latest_start: list[str] = []
    latest_end: list[str] = []
    for position in range(len(result)):
        prior = [cycle for cycle in completed if cycle[1] < position][-3:]
        newest_first = list(reversed(prior))
        peaks = [cycle[2] for cycle in newest_first]
        padded = peaks + [np.nan] * (3 - len(peaks))
        peak1.append(padded[0])
        peak2.append(padded[1])
        peak3.append(padded[2])
        finite_peaks = [value for value in peaks if math.isfinite(value)]
        max3.append(max(finite_peaks) if finite_peaks else np.nan)
        reached75.append(float(sum(value >= 75.0 for value in finite_peaks)))
        valid_count.append(len(finite_peaks))
        latest_start.append(newest_first[0][3] if newest_first else "")
        latest_end.append(newest_first[0][4] if newest_first else "")
    result["Prior_GC1_Peak_K"] = peak1
    result["Prior_GC2_Peak_K"] = peak2
    result["Prior_GC3_Peak_K"] = peak3
    result["Prior_GC_MaxPeak_K_Last3"] = max3
    result["Prior_GC_Reached75_Count_Last3"] = reached75
    result["Prior_GC_Valid_Count_Last3"] = valid_count
    result["Prior_GC1_Start_Date"] = latest_start
    result["Prior_GC1_Death_Date"] = latest_end
    return result


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
    work = add_completed_golden_cross_history(work)

    # 只使用信号当周及此前完整周；所有“过去窗口”都先shift(1)，不把信号周混入历史。
    work["MA20"] = work["close"].rolling(20, min_periods=20).mean()
    work["Close_to_MA20_pct"] = (work["close"] / work["MA20"] - 1.0) * 100.0
    work["MA20_Slope_4W_pct"] = (work["MA20"] / work["MA20"].shift(4) - 1.0) * 100.0
    work["Prior15_Min_K"] = work["K"].shift(1).rolling(15, min_periods=15).min()
    work["Prior15_Max_K"] = work["K"].shift(1).rolling(15, min_periods=15).max()
    work["Prior15_Below25_Weeks"] = (
        work["K"].shift(1).lt(SKDJ_BOTTOM).astype(float)
        .rolling(15, min_periods=15).sum()
    )
    below_streak: list[int] = []
    current_streak = 0
    for is_below in work["K"].lt(SKDJ_BOTTOM).fillna(False).tolist():
        current_streak = current_streak + 1 if bool(is_below) else 0
        below_streak.append(current_streak)
    work["Prior_Below25_Streak"] = pd.Series(
        below_streak, index=work.index, dtype=float).shift(1)

    prior_volume_ma5 = work["vol"].shift(1).rolling(5, min_periods=5).mean()
    work["Volume_Ratio_5W"] = work["vol"] / prior_volume_ma5.replace(0, np.nan)
    work["Signal_Week_Return_pct"] = work["close"].pct_change() * 100.0
    work["Return_4W_pct"] = work["close"].pct_change(4, fill_method=None) * 100.0
    work["Return_8W_pct"] = work["close"].pct_change(8, fill_method=None) * 100.0
    work["Return_12W_pct"] = work["close"].pct_change(12, fill_method=None) * 100.0
    price_range = (work["high"] - work["low"]).replace(0, np.nan)
    work["Signal_Close_Location_pct"] = (
        (work["close"] - work["low"]) / price_range * 100.0)
    work["Signal_Upper_Shadow_pct"] = (
        (work["high"] - work[["open", "close"]].max(axis=1))
        / price_range * 100.0)
    return work


def trend_state(signal: pd.Series) -> str:
    distance = finite_num(signal.get("Close_to_MA20_pct"))
    slope = finite_num(signal.get("MA20_Slope_4W_pct"))
    if not math.isfinite(distance) or not math.isfinite(slope):
        return "数据不足"
    if distance >= 0 and slope > 0:
        return "站上MA20且MA20向上"
    if distance >= 0:
        return "站上MA20但MA20未向上"
    if slope > 0:
        return "MA20下方但MA20向上"
    return "MA20下方且MA20未向上"


def volume_price_state(signal: pd.Series) -> str:
    ratio = finite_num(signal.get("Volume_Ratio_5W"))
    weekly_return = finite_num(signal.get("Signal_Week_Return_pct"))
    close_location = finite_num(signal.get("Signal_Close_Location_pct"))
    upper_shadow = finite_num(signal.get("Signal_Upper_Shadow_pct"))
    if not all(math.isfinite(value) for value in (
            ratio, weekly_return, close_location, upper_shadow)):
        return "数据不足"
    strong_close = weekly_return > 0 and close_location >= 60 and upper_shadow < 35
    if ratio < 1.0:
        return "缩量强收" if strong_close else "缩量一般"
    if ratio <= 2.5:
        return "温和放量强收" if strong_close else "温和放量一般"
    if ratio <= 4.0:
        return "明显放量强收" if strong_close else "明显放量一般"
    return "爆量强收" if strong_close else "爆量弱收"


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


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return np.nan
    return numerator / denominator


def signal_week_daily_features(daily: pd.DataFrame, basic: pd.DataFrame,
                               signal_date: str) -> dict[str, Any]:
    """Describe the completed signal week's daily path with no next-week data."""
    names = [
        "Trading_Days", "Up_Days", "Down_Days", "Up_Day_Pct",
        "Signed_Path_Efficiency_Pct", "One_Day_Gain_Share_Pct",
        "Up_Down_Volume_Ratio", "Up_Volume_Share_Pct",
        "Up_Down_Turnover_Ratio", "Last2_Return_Pct",
        "High_Day_Position_Pct", "Low_Before_High",
        "Amount_Ratio_vs_Prior20D", "Turnover20_CV",
        "Abs_Return_Per_Amount_20D",
    ]
    out: dict[str, Any] = {name: np.nan for name in names}
    signal_ts = pd.Timestamp(signal_date)
    period = signal_ts.to_period("W-FRI")
    start_date = period.start_time.strftime("%Y%m%d")
    history = daily[daily["trade_date"].astype(str).le(signal_date)].copy()
    week = history[history["trade_date"].astype(str).between(
        start_date, signal_date)].copy().sort_values("trade_date")
    before = history[history["trade_date"].astype(str).lt(start_date)]
    if week.empty or before.empty:
        return out
    previous_close = finite_num(before.iloc[-1].get("close"))
    if not math.isfinite(previous_close) or previous_close <= 0:
        return out
    closes = pd.to_numeric(week["close"], errors="coerce")
    previous = closes.shift(1)
    previous.iloc[0] = previous_close
    returns = (closes / previous.replace(0, np.nan) - 1.0) * 100.0
    valid_returns = returns.dropna()
    up = returns.gt(0)
    down = returns.lt(0)
    volumes = pd.to_numeric(week.get("vol"), errors="coerce")
    amounts = pd.to_numeric(week.get("amount"), errors="coerce")
    absolute_path = valid_returns.abs().sum()
    net_return = (finite_num(closes.iloc[-1]) / previous_close - 1.0) * 100.0
    positive_sum = valid_returns[valid_returns.gt(0)].sum()
    out.update({
        "Trading_Days": int(len(week)),
        "Up_Days": int(up.sum()),
        "Down_Days": int(down.sum()),
        "Up_Day_Pct": up.mean() * 100.0,
        "Signed_Path_Efficiency_Pct": safe_ratio(net_return, absolute_path) * 100.0,
        "One_Day_Gain_Share_Pct": safe_ratio(
            finite_num(valid_returns[valid_returns.gt(0)].max()),
            finite_num(positive_sum)) * 100.0,
        "Up_Down_Volume_Ratio": safe_ratio(
            finite_num(volumes[up].sum()), finite_num(volumes[down].sum())),
        "Up_Volume_Share_Pct": safe_ratio(
            finite_num(volumes[up].sum()), finite_num(volumes.sum())) * 100.0,
        "Last2_Return_Pct": (
            (finite_num(closes.iloc[-1]) / finite_num(previous.iloc[max(0, len(week) - 2)]) - 1.0)
            * 100.0 if len(week) >= 2 and finite_num(
                previous.iloc[max(0, len(week) - 2)]) > 0 else net_return),
    })
    highs = pd.to_numeric(week["high"], errors="coerce")
    lows = pd.to_numeric(week["low"], errors="coerce")
    if highs.notna().any() and lows.notna().any():
        high_position = int(np.flatnonzero(highs.eq(highs.max()).to_numpy())[-1])
        low_position = int(np.flatnonzero(lows.eq(lows.min()).to_numpy())[0])
        out["High_Day_Position_Pct"] = (
            (high_position + 1) / max(len(week), 1) * 100.0)
        out["Low_Before_High"] = bool(low_position <= high_position)

    turnover = pd.Series(np.nan, index=week.index, dtype=float)
    if not basic.empty:
        lookup = basic.set_index(basic["trade_date"].astype(str))["turnover_rate"]
        turnover = week["trade_date"].astype(str).map(lookup)
        out["Up_Down_Turnover_Ratio"] = safe_ratio(
            finite_num(turnover[up].sum()), finite_num(turnover[down].sum()))
        basic20 = basic[basic["trade_date"].astype(str).le(signal_date)].tail(20)
        turn20 = pd.to_numeric(basic20.get("turnover_rate"), errors="coerce").dropna()
        if len(turn20) >= 5 and finite_num(turn20.mean()) > 0:
            out["Turnover20_CV"] = finite_num(turn20.std(ddof=0) / turn20.mean())

    prior20 = before.tail(20)
    prior_amount = pd.to_numeric(prior20.get("amount"), errors="coerce")
    if amounts.notna().any() and prior_amount.notna().sum() >= 5:
        out["Amount_Ratio_vs_Prior20D"] = safe_ratio(
            finite_num(amounts.mean()), finite_num(prior_amount.mean()))
    recent20 = history.tail(20).copy()
    recent_close = pd.to_numeric(recent20.get("close"), errors="coerce")
    recent_abs_return = recent_close.pct_change(fill_method=None).abs().sum() * 100.0
    recent_amount = pd.to_numeric(recent20.get("amount"), errors="coerce").sum()
    if len(recent20) >= 10:
        out["Abs_Return_Per_Amount_20D"] = safe_ratio(
            finite_num(recent_abs_return), finite_num(recent_amount)) * 1_000_000.0
    return out


def breadth_key(n: int, trade_date: str, industry: str) -> str:
    return f"{int(n)}|{str(trade_date)}|{str(industry)}"


def add_breadth_observation(store: dict[str, dict[str, float]], n: int,
                            trade_date: str, industry: str,
                            signal: pd.Series) -> None:
    key = breadth_key(n, trade_date, industry)
    bucket = store.setdefault(key, {field: 0.0 for field in BREADTH_SUM_FIELDS})
    bucket["Constituent_Count"] += 1.0
    k = finite_num(signal.get("K"))
    k_change = finite_num(signal.get("K_Change_1W"))
    if math.isfinite(k):
        bucket["K_Valid"] += 1.0
        bucket["K_Above25_Count"] += float(k >= SKDJ_BOTTOM)
        bucket["K_Rising_Count"] += float(math.isfinite(k_change) and k_change > 0)
    ma20_distance = finite_num(signal.get("Close_to_MA20_pct"))
    if math.isfinite(ma20_distance):
        bucket["MA20_Valid"] += 1.0
        bucket["Above_MA20_Count"] += float(ma20_distance >= 0)
    ma20_slope = finite_num(signal.get("MA20_Slope_4W_pct"))
    if math.isfinite(ma20_slope):
        bucket["MA20_Slope_Valid"] += 1.0
        bucket["MA20_Rising_Count"] += float(ma20_slope > 0)
    return4 = finite_num(signal.get("Return_4W_pct"))
    if math.isfinite(return4):
        bucket["Return4W_Valid"] += 1.0
        bucket["Positive_Return4W_Count"] += float(return4 > 0)
        bucket["Return4W_Sum"] += return4
    volume_ratio = finite_num(signal.get("Volume_Ratio_5W"))
    if math.isfinite(volume_ratio):
        bucket["VolumeRatio_Valid"] += 1.0
        bucket["VolumeExpand_Count"] += float(volume_ratio >= 1.0)


def build_industry_breadth_frame(
        store: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, values in store.items():
        try:
            n_text, trade_date, industry = str(key).split("|", 2)
        except ValueError:
            continue
        rows.append({
            "SKDJ_N": int(n_text), "Signal_Date": trade_date,
            "Breadth_Industry": industry,
            "Breadth_Constituent_Count": int(values.get("Constituent_Count", 0)),
            "Breadth_K_Rising_Pct": safe_ratio(
                values.get("K_Rising_Count", 0), values.get("K_Valid", 0)) * 100.0,
            "Breadth_K_Above25_Pct": safe_ratio(
                values.get("K_Above25_Count", 0), values.get("K_Valid", 0)) * 100.0,
            "Breadth_Above_MA20_Pct": safe_ratio(
                values.get("Above_MA20_Count", 0), values.get("MA20_Valid", 0)) * 100.0,
            "Breadth_MA20_Rising_Pct": safe_ratio(
                values.get("MA20_Rising_Count", 0),
                values.get("MA20_Slope_Valid", 0)) * 100.0,
            "Breadth_Positive_Return4W_Pct": safe_ratio(
                values.get("Positive_Return4W_Count", 0),
                values.get("Return4W_Valid", 0)) * 100.0,
            "Breadth_Return4W_Mean_Pct": safe_ratio(
                values.get("Return4W_Sum", 0), values.get("Return4W_Valid", 0)),
            "Breadth_Volume_Expand_Pct": safe_ratio(
                values.get("VolumeExpand_Count", 0),
                values.get("VolumeRatio_Valid", 0)) * 100.0,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    tech = frame[frame["Breadth_Industry"].eq(ALL_TECH_GROUP)].copy()
    tech = tech.rename(columns={
        "Breadth_Constituent_Count": "Tech_Breadth_Constituent_Count",
        "Breadth_K_Rising_Pct": "Tech_Breadth_K_Rising_Pct",
        "Breadth_K_Above25_Pct": "Tech_Breadth_K_Above25_Pct",
        "Breadth_Above_MA20_Pct": "Tech_Breadth_Above_MA20_Pct",
        "Breadth_MA20_Rising_Pct": "Tech_Breadth_MA20_Rising_Pct",
        "Breadth_Positive_Return4W_Pct": "Tech_Breadth_Positive_Return4W_Pct",
        "Breadth_Return4W_Mean_Pct": "Tech_Breadth_Return4W_Mean_Pct",
        "Breadth_Volume_Expand_Pct": "Tech_Breadth_Volume_Expand_Pct",
    }).drop(columns=["Breadth_Industry"])
    industry = frame[~frame["Breadth_Industry"].eq(ALL_TECH_GROUP)].copy()
    result = industry.merge(tech, on=["SKDJ_N", "Signal_Date"], how="left")
    result["Industry_Relative_Tech_Return4W_Pct"] = (
        numeric(result, "Breadth_Return4W_Mean_Pct")
        - numeric(result, "Tech_Breadth_Return4W_Mean_Pct"))
    result["Industry_Relative_Tech_K_Rising_Pct"] = (
        numeric(result, "Breadth_K_Rising_Pct")
        - numeric(result, "Tech_Breadth_K_Rising_Pct"))
    result["Industry_Relative_Tech_Above_MA20_Pct"] = (
        numeric(result, "Breadth_Above_MA20_Pct")
        - numeric(result, "Tech_Breadth_Above_MA20_Pct"))
    return result


def first_hit_detail(path: pd.DataFrame, entry: float, target_pct: float,
                     entry_period: pd.Period) -> tuple[str, str, float]:
    upper, lower = entry * (1 + target_pct / 100.0), entry * 0.90
    for row in path.itertuples(index=False):
        hit_up = finite_num(getattr(row, "high", np.nan)) >= upper
        hit_down = finite_num(getattr(row, "low", np.nan)) <= lower
        hit_date = normalize_date(getattr(row, "trade_date", ""))
        hit_period = pd.Timestamp(hit_date).to_period("W-FRI") if hit_date else None
        hit_week = (
            float(hit_period.ordinal - entry_period.ordinal + 1)
            if hit_period is not None else np.nan)
        if hit_up and hit_down:
            return "同日同时触发_保守按-10%先", hit_date, hit_week
        if hit_down:
            return "先到-10%", hit_date, hit_week
        if hit_up:
            return f"先到+{int(target_pct)}%", hit_date, hit_week
    return "均未触发", "", np.nan


def first_hit_label(path: pd.DataFrame, entry: float, target_pct: float) -> str:
    """Backward-compatible label helper used by older audit functions."""
    if path.empty:
        return "均未触发"
    first_date = normalize_date(path.iloc[0].get("trade_date", ""))
    entry_period = pd.Timestamp(first_date).to_period("W-FRI")
    return first_hit_detail(path, entry, target_pct, entry_period)[0]


def entry_outcomes(daily: pd.DataFrame, signal_date: str, ts_code: str,
                   open_dates: list[str], open_pos: dict[str, int],
                   market_weeks: list[tuple[pd.Period, str]], config: dict[str, Any]
                   ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "Tradable": False, "Date": "", "Raw_Open": np.nan,
        "Reason": "", "Peak_Date_W12": "", "Peak_Week_W12": np.nan,
    }
    for week in range(1, AUDIT_WEEKS + 1):
        out.update({
            f"Has_W{week}": False, f"W{week}_End_Date": "",
            f"W{week}_MFE_Net_pct": np.nan, f"W{week}_MAE_Raw_pct": np.nan,
            f"W{week}_Close_Return_Net_pct": np.nan,
        })
    for audit_week in FIRST_HIT_AUDIT_WEEKS:
        for level in FIRST_HIT_LEVELS:
            prefix = f"First_Hit_{int(level)}_vs_Minus10_W{audit_week}"
            out[prefix] = ""
            out[f"{prefix}_Date"] = ""
            out[f"{prefix}_Week"] = np.nan
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
        if week in FIRST_HIT_AUDIT_WEEKS:
            for level in FIRST_HIT_LEVELS:
                prefix = f"First_Hit_{int(level)}_vs_Minus10_W{week}"
                label, hit_date, hit_week = first_hit_detail(
                    path, raw_entry, level, entry_period)
                out[prefix] = label
                out[f"{prefix}_Date"] = hit_date
                out[f"{prefix}_Week"] = hit_week
        if week == AUDIT_WEEKS:
            highs = pd.to_numeric(path["high"], errors="coerce")
            if highs.notna().any():
                peak_row = path.loc[highs.idxmax()]
                peak_date = normalize_date(peak_row.get("trade_date", ""))
                peak_period = pd.Timestamp(peak_date).to_period("W-FRI")
                out["Peak_Date_W12"] = peak_date
                out["Peak_Week_W12"] = float(
                    peak_period.ordinal - entry_period.ordinal + 1)
    return out


def historical_same_signal_features(
        indicator: pd.DataFrame, current_signal_date: str, daily: pd.DataFrame,
        ts_code: str, open_dates: list[str], open_pos: dict[str, int],
        market_weeks: list[tuple[pd.Period, str]], config: dict[str, Any],
        outcome_cache: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Use only prior K-cross-25 events whose W8 ended before this signal."""
    prefix = "Hist_SameSignal"
    out = {
        f"{prefix}_Valid_Count_Last3": 0.0,
        f"{prefix}_Last1_W4_MFE_Net_pct": np.nan,
        f"{prefix}_Last1_W8_MFE_Net_pct": np.nan,
        f"{prefix}_W2_MFE_Median_Last3_pct": np.nan,
        f"{prefix}_W4_MFE_Median_Last3_pct": np.nan,
        f"{prefix}_W8_MFE_Median_Last3_pct": np.nan,
        f"{prefix}_W8_Close_Median_Last3_pct": np.nan,
        f"{prefix}_W8_MAE_Median_Last3_pct": np.nan,
        f"{prefix}_MFE_to_AbsMAE_Median_Last3": np.nan,
        f"{prefix}_Hit10_Rate_Last3_pct": np.nan,
        f"{prefix}_Hit20_Rate_Last3_pct": np.nan,
        f"{prefix}_Hit30_Rate_Last3_pct": np.nan,
    }
    prior = indicator[
        true_mask(indicator, "K_Cross_25")
        & numeric(indicator, "Prior_Below25_Streak").between(
            1, MAX_BOTTOM_STREAK, inclusive="both")
        & indicator["trade_date"].astype(str).lt(current_signal_date)
    ].sort_values("trade_date", ascending=False)
    completed: list[dict[str, Any]] = []
    for signal in prior.itertuples(index=False):
        date_text = str(getattr(signal, "trade_date"))
        if date_text not in outcome_cache:
            outcome_cache[date_text] = entry_outcomes(
                daily, date_text, ts_code, open_dates, open_pos,
                market_weeks, config)
        prior_outcome = outcome_cache[date_text]
        if (not to_bool(prior_outcome.get("Tradable"))
                or not to_bool(prior_outcome.get("Has_W8"))
                or str(prior_outcome.get("W8_End_Date", "")) >= current_signal_date):
            continue
        completed.append(prior_outcome)
        if len(completed) >= HISTORY_EVENTS:
            break
    if not completed:
        return out

    frame = pd.DataFrame(completed)
    count = len(frame)
    out[f"{prefix}_Valid_Count_Last3"] = float(count)
    out[f"{prefix}_Last1_W4_MFE_Net_pct"] = finite_num(
        frame.iloc[0].get("W4_MFE_Net_pct"))
    out[f"{prefix}_Last1_W8_MFE_Net_pct"] = finite_num(
        frame.iloc[0].get("W8_MFE_Net_pct"))
    for horizon in (2, 4, 8):
        out[f"{prefix}_W{horizon}_MFE_Median_Last3_pct"] = finite_num(
            pd.to_numeric(frame.get(f"W{horizon}_MFE_Net_pct"),
                          errors="coerce").median())
    out[f"{prefix}_W8_Close_Median_Last3_pct"] = finite_num(
        pd.to_numeric(frame.get("W8_Close_Return_Net_pct"), errors="coerce").median())
    out[f"{prefix}_W8_MAE_Median_Last3_pct"] = finite_num(
        pd.to_numeric(frame.get("W8_MAE_Raw_pct"), errors="coerce").median())
    mfe = pd.to_numeric(frame.get("W8_MFE_Net_pct"), errors="coerce")
    mae = pd.to_numeric(frame.get("W8_MAE_Raw_pct"), errors="coerce").abs()
    out[f"{prefix}_MFE_to_AbsMAE_Median_Last3"] = finite_num(
        (mfe / mae.replace(0, np.nan)).median())
    for level in (10, 20, 30):
        labels = frame.get(
            f"First_Hit_{level}_vs_Minus10_W8",
            pd.Series(index=frame.index, dtype=str)).astype(str)
        out[f"{prefix}_Hit{level}_Rate_Last3_pct"] = (
            labels.eq(f"先到+{level}%").mean() * 100.0)
    return out


def analyze_stock(stock: pd.Series, periods: list[dict[str, str]], daily: pd.DataFrame,
                  cached_basic: pd.DataFrame, storage_path: str,
                  week_last_map: dict[pd.Timestamp, str], open_dates: list[str],
                  open_pos: dict[str, int], market_weeks: list[tuple[pd.Period, str]],
                  config: dict[str, Any], use_cache: bool, api_pause: float
                  ) -> tuple[list[dict[str, Any]], dict[str, int],
                             dict[str, dict[str, float]]]:
    rejects: dict[str, int] = {}
    breadth: dict[str, dict[str, float]] = {}
    weekly_base = aggregate_complete_weekly(daily, week_last_map)
    crosses: list[tuple[int, pd.Series]] = []
    indicators: dict[int, pd.DataFrame] = {}
    for n in SKDJ_NS:
        indicator = add_skdj(weekly_base, n)
        indicators[n] = indicator
        breadth_weeks = indicator[
            indicator["trade_date"].astype(str).between(
                config["signal_start"], config["signal_end"])]
        for _, week_signal in breadth_weeks.iterrows():
            week_date = str(week_signal["trade_date"])
            membership = membership_on_date(periods, week_date)
            if membership is None:
                continue
            if not (str(stock["list_date"]) <= week_date < str(stock["delist_date"])):
                continue
            add_breadth_observation(
                breadth, n, week_date, membership["l1"], week_signal)
            add_breadth_observation(
                breadth, n, week_date, ALL_TECH_GROUP, week_signal)
        selected = indicator[
            indicator["K_Cross_25"]
            & indicator["trade_date"].astype(str).between(
                config["signal_start"], config["signal_end"])
        ]
        crosses.extend((n, row) for _, row in selected.iterrows())
    if not crosses:
        return [], rejects, breadth

    code = str(stock["ts_code"])
    basic = ensure_daily_basic(
        code, config["data_start"], config["market_end"], daily, cached_basic,
        storage_path, use_cache, api_pause)
    if basic.empty:
        rejects["出现K上穿25但daily_basic缺失"] = len(crosses)
        return [], rejects, breadth

    rows: list[dict[str, Any]] = []
    historical_outcome_cache: dict[int, dict[str, dict[str, Any]]] = {
        n: {} for n in SKDJ_NS}
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
        historical = historical_same_signal_features(
            indicators[n], signal_date, daily, code, open_dates, open_pos,
            market_weeks, config, historical_outcome_cache[n])
        daily_path = signal_week_daily_features(daily, basic, signal_date)
        k_change = finite_num(signal.get("K_Change_1W"))
        week_return = finite_num(signal.get("Signal_Week_Return_pct"))
        row = {
            "ts_code": code, "name": str(stock["name"]),
            "SKDJ_N": n, "SKDJ_M": SKDJ_M, "Signal_Date": signal_date,
            "Signal_K": finite_num(signal["K"]), "Signal_D": finite_num(signal["D"]),
            "Signal_KD_Spread": finite_num(signal["KD_Spread"]),
            "Signal_K_Change_1W": finite_num(signal["K_Change_1W"]),
            "Signal_K_Above_D": finite_num(signal["K"]) > finite_num(signal["D"]),
            "Signal_MA20": finite_num(signal.get("MA20")),
            "Signal_Close_to_MA20_pct": finite_num(signal.get("Close_to_MA20_pct")),
            "Signal_MA20_Slope_4W_pct": finite_num(signal.get("MA20_Slope_4W_pct")),
            "Signal_Trend_State": trend_state(signal),
            "Signal_Prior15_Min_K": finite_num(signal.get("Prior15_Min_K")),
            "Signal_Prior15_Max_K": finite_num(signal.get("Prior15_Max_K")),
            "Signal_Prior15_Below25_Weeks": finite_num(
                signal.get("Prior15_Below25_Weeks")),
            "Signal_Prior_Below25_Streak": finite_num(
                signal.get("Prior_Below25_Streak")),
            "Signal_Volume_Ratio_5W": finite_num(signal.get("Volume_Ratio_5W")),
            "Signal_Week_Return_pct": finite_num(signal.get("Signal_Week_Return_pct")),
            "Signal_Return_4W_pct": finite_num(signal.get("Return_4W_pct")),
            "Signal_Return_8W_pct": finite_num(signal.get("Return_8W_pct")),
            "Signal_Return_12W_pct": finite_num(signal.get("Return_12W_pct")),
            "Signal_Close_Location_pct": finite_num(
                signal.get("Signal_Close_Location_pct")),
            "Signal_Upper_Shadow_pct": finite_num(
                signal.get("Signal_Upper_Shadow_pct")),
            "Signal_Volume_Price_State": volume_price_state(signal),
            "Signal_Prior_GC1_Peak_K": finite_num(signal.get("Prior_GC1_Peak_K")),
            "Signal_Prior_GC2_Peak_K": finite_num(signal.get("Prior_GC2_Peak_K")),
            "Signal_Prior_GC3_Peak_K": finite_num(signal.get("Prior_GC3_Peak_K")),
            "Signal_Prior_GC_MaxPeak_K_Last3": finite_num(
                signal.get("Prior_GC_MaxPeak_K_Last3")),
            "Signal_Prior_GC_Reached75_Count_Last3": finite_num(
                signal.get("Prior_GC_Reached75_Count_Last3")),
            "Signal_Prior_GC_Valid_Count_Last3": finite_num(
                signal.get("Prior_GC_Valid_Count_Last3")),
            "Signal_Prior_GC1_Start_Date": str(
                signal.get("Prior_GC1_Start_Date", "")),
            "Signal_Prior_GC1_Death_Date": str(
                signal.get("Prior_GC1_Death_Date", "")),
            "SW_L1": membership["l1"], "SW_L2": membership["l2"], "SW_L3": membership["l3"],
            "Signal_K_Thrust_per_AbsWeekReturn": safe_ratio(
                k_change, abs(week_return)),
            **snapshot,
            **historical,
            **{f"SignalWeek_{key}": value for key, value in daily_path.items()},
        }
        row.update({f"Entry_{key}": value for key, value in outcome.items()})
        rows.append(row)
    return rows, rejects, breadth


def mature_events(events: pd.DataFrame, weeks: int = RANKING_WEEKS) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events[
        true_mask(events, "Entry_Tradable")
        & true_mask(events, f"Entry_Has_W{int(weeks)}")
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
            conservative_stop = values.eq("先到-10%") | values.str.startswith("同日同时触发")
            row[f"先到-10比例%_对比{int(level)}"] = conservative_stop.mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def outcome_audit_row(n: int, factor: str, group_name: str,
                      selected: pd.DataFrame, base_count: int,
                      total_weeks: int) -> dict[str, Any]:
    w1_mfe = numeric(selected, "Entry_W1_MFE_Net_pct")
    w3_mfe = numeric(selected, "Entry_W3_MFE_Net_pct")
    w8_mfe = numeric(selected, "Entry_W8_MFE_Net_pct")
    w8_mae = numeric(selected, "Entry_W8_MAE_Raw_pct")
    w8_close = numeric(selected, "Entry_W8_Close_Return_Net_pct")
    hit10 = selected.get(
        "Entry_First_Hit_10_vs_Minus10_W8",
        pd.Series(index=selected.index, dtype=str)).astype(str)
    hit20 = selected.get(
        "Entry_First_Hit_20_vs_Minus10_W8",
        pd.Series(index=selected.index, dtype=str)).astype(str)
    stop10 = hit10.eq("先到-10%") | hit10.str.startswith("同日同时触发")
    signal_weeks = (
        pd.to_datetime(selected["Signal_Date"].astype(str), format="%Y%m%d")
        .dt.to_period("W-FRI").nunique()
        if not selected.empty else 0)
    return {
        "SKDJ_N": n, "因子": factor, "分组": group_name,
        "事件数": len(selected),
        "保留事件比例%": len(selected) / base_count * 100 if base_count else np.nan,
        "不同股票": selected["ts_code"].nunique() if not selected.empty else 0,
        "有信号周": signal_weeks,
        "空窗周": max(total_weeks - signal_weeks, 0),
        "W1最大浮盈中位%": w1_mfe.median(),
        "W3最大浮盈中位%": w3_mfe.median(),
        "W8最大浮盈均值%": w8_mfe.mean(),
        "W8最大浮盈中位%": w8_mfe.median(),
        "W8收盘平均净收益%": w8_close.mean(),
        "W8收盘中位净收益%": w8_close.median(),
        "W8收盘胜率%": w8_close.gt(0).mean() * 100,
        "W8最大回撤中位%": w8_mae.median(),
        "W8触及-10%比例%": w8_mae.le(-10).mean() * 100,
        "先到+10比例%": hit10.eq("先到+10%").mean() * 100,
        "先到-10比例%_对比10": stop10.mean() * 100,
        "先到+20比例%": hit20.eq("先到+20%").mean() * 100,
    }


def feature_bucket_audit(events: pd.DataFrame, total_weeks: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric_specs = [
        ("突破K值", "Signal_K", [-np.inf, 28, 32, 38, np.inf],
         ["≤28", "28～32", "32～38", ">38"]),
        ("收盘相对MA20", "Signal_Close_to_MA20_pct", [-np.inf, 0, 5, 15, np.inf],
         ["MA20下方", "上方0～5%", "上方5～15%", "上方>15%"]),
        ("MA20近4周斜率", "Signal_MA20_Slope_4W_pct", [-np.inf, -3, 0, 3, np.inf],
         ["<-3%", "-3%～0", "0～3%", ">3%"]),
        ("此前15周最低K", "Signal_Prior15_Min_K", [-np.inf, 5, 15, 22, 25, np.inf],
         ["<5", "5～15", "15～22", "22～25", "≥25"]),
        ("连续处于25下方周数", "Signal_Prior_Below25_Streak",
         [0, 2, 5, 9, np.inf], ["1～2周", "3～5周", "6～9周", "≥10周"]),
        ("当周量比/此前5周", "Signal_Volume_Ratio_5W",
         [-np.inf, 1, 2.5, 4, np.inf], ["<1", "1～2.5", "2.5～4", ">4"]),
        ("信号周涨幅", "Signal_Week_Return_pct", [-np.inf, 0, 5, 10, np.inf],
         ["≤0", "0～5%", "5～10%", ">10%"]),
        ("信号周收盘位置", "Signal_Close_Location_pct", [-np.inf, 40, 60, 80, np.inf],
         ["≤40", "40～60", "60～80", ">80"]),
        ("信号周上影占振幅", "Signal_Upper_Shadow_pct", [-np.inf, 10, 25, 40, np.inf],
         ["≤10", "10～25", "25～40", ">40"]),
    ]
    categorical_specs = [
        ("趋势组合状态", "Signal_Trend_State"),
        ("量价组合状态", "Signal_Volume_Price_State"),
    ]
    for n in SKDJ_NS:
        base = events[events["SKDJ_N"].eq(n)].copy()
        base_count = len(base)
        rows.append(outcome_audit_row(
            n, "基准", "全部事件", base, base_count, total_weeks))
        for factor, column, bins, labels in numeric_specs:
            values = numeric(base, column)
            groups = pd.cut(values, bins=bins, labels=labels, right=True)
            for label in labels:
                selected = base[groups.eq(label)]
                rows.append(outcome_audit_row(
                    n, factor, label, selected, base_count, total_weeks))
            missing = base[values.isna()]
            if not missing.empty:
                rows.append(outcome_audit_row(
                    n, factor, "数据不足", missing, base_count, total_weeks))
        for factor, column in categorical_specs:
            for label in base[column].fillna("数据不足").drop_duplicates().tolist():
                selected = base[base[column].fillna("数据不足").eq(label)]
                rows.append(outcome_audit_row(
                    n, factor, str(label), selected, base_count, total_weeks))
    return pd.DataFrame(rows)


def combination_audit(events: pd.DataFrame, total_weeks: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base = events[events["SKDJ_N"].eq(n)].copy()
        base_count = len(base)
        distance = numeric(base, "Signal_Close_to_MA20_pct")
        slope = numeric(base, "Signal_MA20_Slope_4W_pct")
        k_value = numeric(base, "Signal_K")
        prior_min = numeric(base, "Signal_Prior15_Min_K")
        duration = numeric(base, "Signal_Prior_Below25_Streak")
        volume_ratio = numeric(base, "Signal_Volume_Ratio_5W")
        week_return = numeric(base, "Signal_Week_Return_pct")
        close_location = numeric(base, "Signal_Close_Location_pct")
        upper_shadow = numeric(base, "Signal_Upper_Shadow_pct")
        trend_confirmed = distance.ge(0) & slope.gt(0)
        strong_close = week_return.gt(0) & close_location.ge(60) & upper_shadow.lt(35)
        gentle_volume = volume_ratio.between(1.0, 2.5, inclusive="both")
        definitions = [
            ("基准", "全部事件", pd.Series(True, index=base.index)),
            ("趋势", "站上MA20且MA20向上", trend_confirmed),
            ("K位置", "K在28～32", k_value.gt(28) & k_value.le(32)),
            ("K位置", "K>38_不预设为坏", k_value.gt(38)),
            ("此前最低K", "最低K在22～25", prior_min.ge(22) & prior_min.lt(25)),
            ("此前最低K", "最低K<15", prior_min.lt(15)),
            ("潜伏时间", "连续1～2周", duration.between(1, 2, inclusive="both")),
            ("潜伏时间", "连续≥10周", duration.ge(10)),
            ("量能", "量比1～2.5", gentle_volume),
            ("量价", "温和放量且强收盘", gentle_volume & strong_close),
            ("量价", "爆量但强收盘", volume_ratio.gt(4) & strong_close),
            ("量价", "爆量且弱收盘", volume_ratio.gt(4) & ~strong_close),
            ("组合", "趋势确认+温和放量强收盘",
             trend_confirmed & gentle_volume & strong_close),
            ("组合", "趋势确认+任意量能强收盘", trend_confirmed & strong_close),
        ]
        for factor, label, mask in definitions:
            rows.append(outcome_audit_row(
                n, factor, label, base[mask.fillna(False)], base_count, total_weeks))
    return pd.DataFrame(rows)


def winner_loser_feature_audit(events: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        ("突破K", "Signal_K"),
        ("突破D", "Signal_D"),
        ("当周KD差", "Signal_KD_Spread"),
        ("K一周变化", "Signal_K_Change_1W"),
        ("收盘相对MA20%", "Signal_Close_to_MA20_pct"),
        ("MA20近4周斜率%", "Signal_MA20_Slope_4W_pct"),
        ("此前15周最低K", "Signal_Prior15_Min_K"),
        ("此前15周最高K", "Signal_Prior15_Max_K"),
        ("此前15周低于25周数", "Signal_Prior15_Below25_Weeks"),
        ("连续低于25周数", "Signal_Prior_Below25_Streak"),
        ("当周量比/此前5周", "Signal_Volume_Ratio_5W"),
        ("信号周涨幅%", "Signal_Week_Return_pct"),
        ("信号周收盘位置%", "Signal_Close_Location_pct"),
        ("信号周上影占振幅%", "Signal_Upper_Shadow_pct"),
        ("信号日股价", "Raw_Close"),
        ("信号日流通市值亿元", "Circ_MV_Billion"),
        ("信号日换手率%", "Turnover_Rate"),
    ]
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base = events[events["SKDJ_N"].eq(n)].copy()
        w8_close = numeric(base, "Entry_W8_Close_Return_Net_pct")
        w8_mfe = numeric(base, "Entry_W8_MFE_Net_pct")
        hit10 = base.get(
            "Entry_First_Hit_10_vs_Minus10_W8",
            pd.Series(index=base.index, dtype=str)).astype(str)
        outcome_groups = [
            ("W8收盘盈亏", "盈利", w8_close.gt(0)),
            ("W8收盘盈亏", "亏损或持平", w8_close.le(0)),
            ("+10与-10先后", "先到+10", hit10.eq("先到+10%")),
            ("+10与-10先后", "先到-10_含同日保守",
             hit10.eq("先到-10%") | hit10.str.startswith("同日同时触发")),
            ("W8是否达到20%", "达到20%", w8_mfe.ge(20)),
            ("W8是否达到20%", "未达到20%", w8_mfe.lt(20)),
        ]
        for outcome, label, mask in outcome_groups:
            selected = base[mask.fillna(False)]
            for feature_name, column in feature_columns:
                values = numeric(selected, column).dropna()
                rows.append({
                    "SKDJ_N": n, "结果定义": outcome, "结果分组": label,
                    "事件数": len(selected), "特征": feature_name,
                    "有效样本": len(values), "均值": values.mean(),
                    "中位数": values.median(), "P25": values.quantile(0.25),
                    "P75": values.quantile(0.75),
                })
    return pd.DataFrame(rows)


def lifecycle_cohort_specs() -> list[tuple[str, str]]:
    return [
        ("全部硬条件通过", ""),
        ("原评分周内前3", "Top3"),
        ("V5.4历史二层周内前20%", "H2_Top20Pct"),
        ("V5.4历史二层周内前3", "H2_Top3"),
    ]


def select_lifecycle_cohort(frame: pd.DataFrame, flag: str) -> pd.DataFrame:
    return frame if not flag else frame[true_mask(frame, flag)].copy()


def lifecycle_horizon_audit(eligible_w12: pd.DataFrame,
                            periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Trace cumulative returns without pretending every N has one fixed exit week."""
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible_w12[eligible_w12["SKDJ_N"].eq(n)]
        for period in periods:
            period_base = select_period(base_n, period)
            for cohort_name, flag in lifecycle_cohort_specs():
                selected = select_lifecycle_cohort(period_base, flag)
                previous_close = pd.Series(0.0, index=selected.index, dtype=float)
                previous_mfe = pd.Series(0.0, index=selected.index, dtype=float)
                for week in LIFECYCLE_WEEKS:
                    mfe = numeric(selected, f"Entry_W{week}_MFE_Net_pct")
                    mae = numeric(selected, f"Entry_W{week}_MAE_Raw_pct")
                    close_ret = numeric(selected, f"Entry_W{week}_Close_Return_Net_pct")
                    close_increment = close_ret - previous_close
                    mfe_increment = mfe - previous_mfe
                    weekly_equal = selected.assign(_value=close_ret).groupby(
                        "Signal_Week")["_value"].mean()
                    rows.append({
                        "SKDJ_N": n, "时间分段": period["name"],
                        "排名组": cohort_name, "观察周": week,
                        "事件数": len(selected),
                        "不同股票": selected["ts_code"].nunique() if not selected.empty else 0,
                        "信号周": selected["Signal_Week"].nunique() if not selected.empty else 0,
                        "累计最大浮盈均值%": mfe.mean(),
                        "累计最大浮盈中位%": mfe.median(),
                        "累计达到10%比例%": mfe.ge(10).mean() * 100,
                        "累计达到15%比例%": mfe.ge(15).mean() * 100,
                        "累计达到20%比例%": mfe.ge(20).mean() * 100,
                        "累计最大不利波动中位%": mae.median(),
                        "累计触及-10%比例%": mae.le(-10).mean() * 100,
                        "周末收盘平均净收益%": close_ret.mean(),
                        "周末收盘中位净收益%": close_ret.median(),
                        "周末收盘胜率%": close_ret.gt(0).mean() * 100,
                        "按信号周等权平均净收益%": weekly_equal.mean(),
                        "相对前一周收盘增量均值百分点": close_increment.mean(),
                        "相对前一周收盘增量中位百分点": close_increment.median(),
                        "本周新增最大浮盈均值百分点": mfe_increment.mean(),
                        "本周新增最大浮盈中位百分点": mfe_increment.median(),
                    })
                    previous_close = close_ret
                    previous_mfe = mfe
    return pd.DataFrame(rows)


def _first_reach_week(values: pd.Series, target: float) -> float:
    if not math.isfinite(target):
        return np.nan
    reached = values[numeric(pd.DataFrame({"v": values}), "v").ge(target)]
    return float(reached.index.min()) if not reached.empty else np.nan


def holding_window_summary(eligible_w12: pd.DataFrame,
                           lifecycle: pd.DataFrame,
                           periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible_w12[eligible_w12["SKDJ_N"].eq(n)]
        for period in periods:
            period_base = select_period(base_n, period)
            for cohort_name, flag in lifecycle_cohort_specs():
                selected = select_lifecycle_cohort(period_base, flag)
                curve = lifecycle[
                    lifecycle["SKDJ_N"].eq(n)
                    & lifecycle["时间分段"].eq(period["name"])
                    & lifecycle["排名组"].eq(cohort_name)
                ].set_index("观察周")
                mean_close = numeric(curve, "周末收盘平均净收益%")
                median_close = numeric(curve, "周末收盘中位净收益%")
                median_mfe = numeric(curve, "累计最大浮盈中位%")
                w12_mfe = finite_num(median_mfe.get(AUDIT_WEEKS, np.nan))
                maturity_target = w12_mfe * 0.90 if math.isfinite(w12_mfe) and w12_mfe > 0 else np.nan
                hit_weeks: dict[int, float] = {}
                for level in FIRST_HIT_LEVELS:
                    prefix = f"Entry_First_Hit_{int(level)}_vs_Minus10_W12"
                    labels = selected.get(
                        prefix, pd.Series(index=selected.index, dtype=str)).astype(str)
                    weeks = numeric(selected, f"{prefix}_Week")
                    hit_weeks[int(level)] = weeks[labels.eq(
                        f"先到+{int(level)}%")].median()
                hit10 = selected.get(
                    "Entry_First_Hit_10_vs_Minus10_W12",
                    pd.Series(index=selected.index, dtype=str)).astype(str)
                rows.append({
                    "SKDJ_N": n, "时间分段": period["name"], "排名组": cohort_name,
                    "事件数": len(selected),
                    "平均收盘收益最高周": float(mean_close.idxmax()) if mean_close.notna().any() else np.nan,
                    "中位收盘收益最高周": float(median_close.idxmax()) if median_close.notna().any() else np.nan,
                    "最大浮盈中位达到W12的90%首周": _first_reach_week(
                        median_mfe, maturity_target),
                    "W12较W8平均收盘变化百分点": (
                        numeric(selected, "Entry_W12_Close_Return_Net_pct")
                        - numeric(selected, "Entry_W8_Close_Return_Net_pct")).mean(),
                    "W12较W8中位收盘变化百分点": (
                        numeric(selected, "Entry_W12_Close_Return_Net_pct")
                        - numeric(selected, "Entry_W8_Close_Return_Net_pct")).median(),
                    "W12较W8新增最大浮盈中位百分点": (
                        numeric(selected, "Entry_W12_MFE_Net_pct")
                        - numeric(selected, "Entry_W8_MFE_Net_pct")).median(),
                    "W12最高价出现周中位": numeric(
                        selected, "Entry_Peak_Week_W12").median(),
                    "W12最高价出现周P25": numeric(
                        selected, "Entry_Peak_Week_W12").quantile(0.25),
                    "W12最高价出现周P75": numeric(
                        selected, "Entry_Peak_Week_W12").quantile(0.75),
                    "先到+10事件的触发周中位": hit_weeks[10],
                    "先到+15事件的触发周中位": hit_weeks[15],
                    "先到+20事件的触发周中位": hit_weeks[20],
                    "W12先到+10比例%": hit10.eq("先到+10%").mean() * 100,
                    "W12先到-10比例%_对比10": (
                        hit10.eq("先到-10%")
                        | hit10.str.startswith("同日同时触发")).mean() * 100,
                })
    return pd.DataFrame(rows)


def early_failure_audit(eligible_w12: pd.DataFrame,
                        periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Describe early divergence; this is an audit, not a fitted exit rule."""
    rows: list[dict[str, Any]] = []
    audit_periods = [p for p in periods if p["name"] in {
        "全部区间", "前段观察", "后段冻结检验"}]
    for n in SKDJ_NS:
        base_n = eligible_w12[eligible_w12["SKDJ_N"].eq(n)]
        for period in audit_periods:
            base = select_period(base_n, period)
            hit10 = base.get(
                "Entry_First_Hit_10_vs_Minus10_W12",
                pd.Series(index=base.index, dtype=str)).astype(str)
            outcome_specs = [
                ("W12最大浮盈≥20", numeric(base, "Entry_W12_MFE_Net_pct").ge(20)),
                ("W12先到+10", hit10.eq("先到+10%")),
                ("W12先到-10_含同日保守", hit10.eq("先到-10%") | hit10.str.startswith("同日同时触发")),
                ("W12收盘亏损或持平", numeric(base, "Entry_W12_Close_Return_Net_pct").le(0)),
            ]
            for outcome_name, mask in outcome_specs:
                selected = base[mask.fillna(False)]
                row: dict[str, Any] = {
                    "SKDJ_N": n, "时间分段": period["name"],
                    "W12结果组": outcome_name, "事件数": len(selected),
                    "不同股票": selected["ts_code"].nunique() if not selected.empty else 0,
                }
                for week in (1, 2, 3, 4):
                    close_ret = numeric(selected, f"Entry_W{week}_Close_Return_Net_pct")
                    mfe = numeric(selected, f"Entry_W{week}_MFE_Net_pct")
                    mae = numeric(selected, f"Entry_W{week}_MAE_Raw_pct")
                    row.update({
                        f"W{week}收盘中位%": close_ret.median(),
                        f"W{week}收盘≤0比例%": close_ret.le(0).mean() * 100,
                        f"W{week}最大浮盈中位%": mfe.median(),
                        f"W{week}最大浮盈<5比例%": mfe.lt(5).mean() * 100,
                        f"W{week}最大不利波动中位%": mae.median(),
                    })
                rows.append(row)
    return pd.DataFrame(rows)


def pair_parameter_signals(events: pd.DataFrame, fast_n: int, slow_n: int,
                           max_days: int = PAIR_MAX_CALENDAR_DAYS) -> pd.DataFrame:
    fast = events[events["SKDJ_N"].eq(fast_n)].reset_index(drop=True)
    slow = events[events["SKDJ_N"].eq(slow_n)].reset_index(drop=True)
    slow_groups = {str(code): group.reset_index(drop=True)
                   for code, group in slow.groupby("ts_code")}
    details: list[dict[str, Any]] = []
    for code, fast_group in fast.groupby("ts_code"):
        slow_group = slow_groups.get(str(code))
        if slow_group is None or slow_group.empty:
            continue
        candidates: list[tuple[int, int, int, int]] = []
        fast_group = fast_group.reset_index(drop=True)
        for fast_pos, fast_row in fast_group.iterrows():
            fast_date = pd.to_datetime(str(fast_row["Signal_Date"]), format="%Y%m%d")
            for slow_pos, slow_row in slow_group.iterrows():
                slow_date = pd.to_datetime(str(slow_row["Signal_Date"]), format="%Y%m%d")
                gap = int((slow_date - fast_date).days)
                if abs(gap) <= int(max_days):
                    candidates.append((abs(gap), gap, int(fast_pos), int(slow_pos)))
        used_fast: set[int] = set()
        used_slow: set[int] = set()
        for _, gap, fast_pos, slow_pos in sorted(candidates):
            if fast_pos in used_fast or slow_pos in used_slow:
                continue
            used_fast.add(fast_pos)
            used_slow.add(slow_pos)
            fast_row = fast_group.iloc[fast_pos]
            slow_row = slow_group.iloc[slow_pos]
            fast_price = finite_num(fast_row.get("Entry_Raw_Open"))
            slow_price = finite_num(slow_row.get("Entry_Raw_Open"))
            fast_peak = str(fast_row.get("Entry_Peak_Date_W12", ""))
            slow_peak = str(slow_row.get("Entry_Peak_Date_W12", ""))
            peak_gap = (
                abs((pd.to_datetime(slow_peak) - pd.to_datetime(fast_peak)).days)
                if normalize_date(fast_peak) and normalize_date(slow_peak) else np.nan)
            detail = {
                "参数对": f"N{fast_n}-N{slow_n}", "快参数": fast_n,
                "慢参数": slow_n, "ts_code": str(code),
                "name": str(fast_row.get("name", "")),
                "快参数信号日": str(fast_row["Signal_Date"]),
                "慢参数信号日": str(slow_row["Signal_Date"]),
                "慢减快日历天": gap, "完全同日": gap == 0,
                "快参数更早": gap > 0,
                "快参数买入日": str(fast_row.get("Entry_Date", "")),
                "慢参数买入日": str(slow_row.get("Entry_Date", "")),
                "快参数买入开盘价": fast_price,
                "慢参数买入开盘价": slow_price,
                "慢参数相对快参数买价差%": (
                    (slow_price / fast_price - 1) * 100
                    if math.isfinite(fast_price) and fast_price > 0
                    and math.isfinite(slow_price) else np.nan),
                "快参数W12最大浮盈%": finite_num(
                    fast_row.get("Entry_W12_MFE_Net_pct")),
                "慢参数W12最大浮盈%": finite_num(
                    slow_row.get("Entry_W12_MFE_Net_pct")),
                "慢减快W12最大浮盈百分点": (
                    finite_num(slow_row.get("Entry_W12_MFE_Net_pct"))
                    - finite_num(fast_row.get("Entry_W12_MFE_Net_pct"))),
                "快参数W12收盘净收益%": finite_num(
                    fast_row.get("Entry_W12_Close_Return_Net_pct")),
                "慢参数W12收盘净收益%": finite_num(
                    slow_row.get("Entry_W12_Close_Return_Net_pct")),
                "慢减快W12收盘百分点": (
                    finite_num(slow_row.get("Entry_W12_Close_Return_Net_pct"))
                    - finite_num(fast_row.get("Entry_W12_Close_Return_Net_pct"))),
                "快参数W12最高价日期": fast_peak,
                "慢参数W12最高价日期": slow_peak,
                "两参数W12最高价日期相差天": peak_gap,
                "两参数最高价日期在14天内": bool(
                    math.isfinite(peak_gap) and peak_gap <= 14),
            }
            details.append(detail)
    return pd.DataFrame(details)


def parameter_pair_audit(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_frames = [
        pair_parameter_signals(events, 6, 7),
        pair_parameter_signals(events, 6, 9),
        pair_parameter_signals(events, 7, 9),
    ]
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for label in ("N6-N7", "N6-N9", "N7-N9"):
        selected = detail[detail["参数对"].eq(label)] if not detail.empty else pd.DataFrame()
        summary_rows.append({
            "参数对": label, "匹配事件": len(selected),
            "不同股票": selected["ts_code"].nunique() if not selected.empty else 0,
            "完全同日比例%": true_mask(selected, "完全同日").mean() * 100 if not selected.empty else np.nan,
            "快参数更早比例%": true_mask(selected, "快参数更早").mean() * 100 if not selected.empty else np.nan,
            "慢减快日历天中位": numeric(selected, "慢减快日历天").median(),
            "慢减快日历天P75": numeric(selected, "慢减快日历天").quantile(0.75),
            "慢参数相对快参数买价差中位%": numeric(
                selected, "慢参数相对快参数买价差%").median(),
            "慢减快W12最大浮盈中位百分点": numeric(
                selected, "慢减快W12最大浮盈百分点").median(),
            "慢减快W12收盘中位百分点": numeric(
                selected, "慢减快W12收盘百分点").median(),
            "最高价日期在14天内比例%": true_mask(
                selected, "两参数最高价日期在14天内").mean() * 100 if not selected.empty else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    n69 = detail[detail["参数对"].eq("N6-N9")].copy()
    n79 = detail[detail["参数对"].eq("N7-N9")].copy()
    if n69.empty or n79.empty:
        triple = pd.DataFrame()
    else:
        triple = n69.merge(
            n79, on=["ts_code", "慢参数信号日"], suffixes=("_N6N9", "_N7N9"))
        keep = [
            "ts_code", "name_N6N9", "快参数信号日_N6N9",
            "快参数信号日_N7N9", "慢参数信号日",
            "慢减快日历天_N6N9", "慢减快日历天_N7N9",
            "慢参数相对快参数买价差%_N6N9",
            "慢参数相对快参数买价差%_N7N9",
            "两参数最高价日期在14天内_N6N9",
            "两参数最高价日期在14天内_N7N9",
        ]
        triple = triple[[column for column in keep if column in triple.columns]].copy()
        triple = triple.rename(columns={
            "name_N6N9": "name", "快参数信号日_N6N9": "N6信号日",
            "快参数信号日_N7N9": "N7信号日", "慢参数信号日": "N9信号日",
            "慢减快日历天_N6N9": "N9减N6日历天",
            "慢减快日历天_N7N9": "N9减N7日历天",
            "慢参数相对快参数买价差%_N6N9": "N9相对N6买价差%",
            "慢参数相对快参数买价差%_N7N9": "N9相对N7买价差%",
            "两参数最高价日期在14天内_N6N9": "N6与N9最高价日期14天内",
            "两参数最高价日期在14天内_N7N9": "N7与N9最高价日期14天内",
        })
    return summary, detail, triple


def frozen_score_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        ("硬条件", "25线下连续周数", "1～5周通过；≥6周或数据不足不进入排名", "不计分"),
        ("SKDJ重置质量", "此前15周最低K", "15≤最低K<25", "+35"),
        ("SKDJ重置质量", "此前15周最低K", "其他区间", "+0"),
        ("量能质量", "信号周量/此前5周均量", "1.0～2.5", "+20"),
        ("量能质量", "信号周量/此前5周均量", "<1.0", "+8"),
        ("量能质量", "信号周量/此前5周均量", "2.5～4.0", "+0"),
        ("量能质量", "信号周量/此前5周均量", ">4.0", "-10"),
        ("周K线结构", "收盘位置", "60～80", "+10"),
        ("周K线结构", "上影线占振幅", "10～40", "+10"),
        ("中线趋势", "MA20近4周斜率", "≥0", "+8"),
        ("中线趋势", "收盘相对MA20", "上方5%～15%", "+7"),
        ("中线趋势", "收盘相对MA20", "上方0～5%", "+4"),
        ("中线趋势", "收盘相对MA20", "上方超过15%", "+2"),
        ("价格市值", "同参数同周候选内部百分位", "价格百分位×5＋流通市值百分位×5", "0～10"),
        ("总分", "冻结透明分", "各项相加后限制在0～100；突破当周K值不计分", "0～100"),
        ("排名", "同参数同周", "总分降序；分项和代码仅用于确定性破同分", "第1名起"),
        ("独立审计", "同行业4/8/12周相对强度", "同周同申万一级行业候选内比较", "不计分"),
        ("独立审计", "板块共振", "同周同行业候选数量和占比", "不计分"),
        ("独立审计", "此前金叉峰值", "最近3个已完成K/D金叉周期最高K", "不计分"),
    ], columns=["层级", "字段", "条件", "分值"])


def challenger_score_definitions() -> pd.DataFrame:
    """V5.2 control and V5.4 pre-registered rules; no future return enters them."""
    return pd.DataFrame([
        ("共同硬条件", "25线下连续周数", "1～5周通过；≥6周或数据不足不排名", "不计分"),
        ("V5.2旧方案对照", "S级", "达到75次数≥2且板块共振40%～60%", "最高优先级"),
        ("V5.2旧方案对照", "A/B/C级", "沿用V5.2共振S/A/B/C定义", "依次降低"),
        ("V5.4历史二层", "优先层", "最近3次已完成金叉至少2次达到75；2次和3次不再区分", "最高优先级"),
        ("V5.4历史二层", "普通层", "达到75次数少于2次或无有效记录", "第二优先级"),
        ("V5.4同层排序", "V5.1.1冻结100分", "同层先按原冻结总分降序", "主要顺序"),
        ("V5.4同层破同分", "最近一次已完成金叉最高K", "仅在原冻结总分相同时降序", "破同分"),
        ("板块共振", "同行业信号占当周候选比例", "只分档审计，不参与个股名次", "不计分"),
        ("每周候选数量", "1～4 / 5～20 / >20", "低/中/高仓位置信度，只审计不剔除", "不计分"),
    ], columns=["层级", "字段", "条件", "分值或优先级"])


def ranking_scheme_specs() -> list[dict[str, str]]:
    return [
        {"name": "原冻结100分", "rank": "Weekly_Rank", "rank_pct": "Weekly_Rank_Pct",
         "top3": "Top3", "top5": "Top5", "top20": "Top20Pct",
         "bottom20": "Bottom20Pct", "score": "Score_Total_100", "tier": ""},
        {"name": "V5.2共振SABC对照", "rank": "V52_Tier_Weekly_Rank",
         "rank_pct": "V52_Tier_Weekly_Rank_Pct", "top3": "V52_Tier_Top3",
         "top5": "V52_Tier_Top5", "top20": "V52_Tier_Top20Pct",
         "bottom20": "V52_Tier_Bottom20Pct", "score": "V52_Tier_Order",
         "tier": "V52_Tier_Level"},
        {"name": "V5.4历史≥2次优先", "rank": "H2_Weekly_Rank",
         "rank_pct": "H2_Weekly_Rank_Pct", "top3": "H2_Top3",
         "top5": "H2_Top5", "top20": "H2_Top20Pct",
         "bottom20": "H2_Bottom20Pct", "score": "H2_Tier_Order",
         "tier": "H2_Tier_Level"},
    ]


def v57_ranking_scheme_specs() -> list[dict[str, str]]:
    """Pre-registered V5.7 rankings; none uses post-entry outcomes."""
    return [
        {"name": "V5.7-K效率低优先", "rank": "V57_K_Weekly_Rank",
         "rank_pct": "V57_K_Weekly_Rank_Pct", "top3": "V57_K_Top3",
         "top5": "V57_K_Top5", "top20": "V57_K_Top20Pct",
         "bottom20": "V57_K_Bottom20Pct",
         "score": "V57_K_Efficiency_Weekly_Pct", "tier": ""},
        {"name": "V5.7-行业MA20广度高优先", "rank": "V57_Breadth_Weekly_Rank",
         "rank_pct": "V57_Breadth_Weekly_Rank_Pct",
         "top3": "V57_Breadth_Top3", "top5": "V57_Breadth_Top5",
         "top20": "V57_Breadth_Top20Pct",
         "bottom20": "V57_Breadth_Bottom20Pct",
         "score": "V57_Industry_MA20_Weekly_Pct", "tier": ""},
        {"name": "V5.7-双因子分层", "rank": "V57_Dual_Weekly_Rank",
         "rank_pct": "V57_Dual_Weekly_Rank_Pct", "top3": "V57_Dual_Top3",
         "top5": "V57_Dual_Top5", "top20": "V57_Dual_Top20Pct",
         "bottom20": "V57_Dual_Bottom20Pct",
         "score": "V57_Factor_Tier_Order", "tier": "V57_Factor_Tier"},
        {"name": "V5.7-双因子层内H2优先", "rank": "V57_DualH2_Weekly_Rank",
         "rank_pct": "V57_DualH2_Weekly_Rank_Pct",
         "top3": "V57_DualH2_Top3", "top5": "V57_DualH2_Top5",
         "top20": "V57_DualH2_Top20Pct",
         "bottom20": "V57_DualH2_Bottom20Pct",
         "score": "V57_Factor_Tier_Order", "tier": "V57_Factor_Tier"},
    ]


def v58_ranking_scheme_specs() -> list[dict[str, str]]:
    """History-completeness controls added after the V5.7 result was frozen."""
    return [
        {"name": "V5.8完整历史H2优先", "rank": "H2C_Weekly_Rank",
         "rank_pct": "H2C_Weekly_Rank_Pct", "top3": "H2C_Top3",
         "top5": "H2C_Top5", "top20": "H2C_Top20Pct",
         "bottom20": "H2C_Bottom20Pct", "score": "H2C_Tier_Order",
         "tier": "H2C_Tier_Level"},
        {"name": "V5.8双因子层内完整H2", "rank": "V58_DualH2C_Weekly_Rank",
         "rank_pct": "V58_DualH2C_Weekly_Rank_Pct",
         "top3": "V58_DualH2C_Top3", "top5": "V58_DualH2C_Top5",
         "top20": "V58_DualH2C_Top20Pct",
         "bottom20": "V58_DualH2C_Bottom20Pct",
         "score": "V57_Factor_Tier_Order", "tier": "V57_Factor_Tier"},
    ]


def all_ranking_scheme_specs() -> list[dict[str, str]]:
    return (ranking_scheme_specs() + v57_ranking_scheme_specs()
            + v58_ranking_scheme_specs())


def v57_ranking_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        ("共同候选", "不变", "科技池、10元、50亿元、K上穿25、25线下1～5周硬条件全部冻结"),
        ("K效率排名", "主因子", "K动能相对股价扩张效率从低到高；原100分仅破同值"),
        ("行业广度排名", "主因子", "所属行业MA20上升比例从高到低；原100分仅破同值"),
        ("双因子P1", "最高层", "同周K效率处于优秀20%，且行业MA20上升比例处于优秀20%"),
        ("双因子P2", "第二层", "两个特征中只有一个进入同周优秀20%"),
        ("双因子P3", "第三层", "两个特征均未进入同周优秀20%"),
        ("双因子PX", "最后层", "任一特征缺失；不会因缺失获得优先级"),
        ("双因子分层", "层内排序", "先P1/P2/P3/PX，再按两因子周内百分位和，最后用原100分破同值"),
        ("双因子层内H2", "层内排序", "先P1/P2/P3/PX；同层先V5.4历史H2，再按两因子百分位和，最后原100分"),
        ("判卷", "主目标", "S级、A/S比例、前三至少两只A/S周、W8最大浮盈；B级和回撤只辅助"),
    ], columns=["方案", "层级", "预先冻结规则"])


def v58_history_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        ("历史窗口", "200周", "正式信号开始前读取200个日历周，仅用于形成指标和历史周期，不增加正式信号"),
        ("历史完整", "完整3次", "当前信号以前至少存在3个已经死叉结束的K/D金叉周期"),
        ("历史不足", "部分2次", "只找到2个已完成周期；单独标记，不按第三次失败处理"),
        ("历史不足", "部分1次", "只找到1个已完成周期；单独标记，不按其余两次失败处理"),
        ("历史不足", "无有效周期", "没有已完成周期；单独标记，不等同于三次均未到75"),
        ("完整H2优先层", "S", "历史完整3次，且其中至少2次最高K达到75"),
        ("完整H2普通层", "C", "历史完整3次，但达到75不足2次"),
        ("完整H2数据不足层", "PX", "有效历史周期不足3次；排在完整S/C之后，但不删除事件"),
        ("冻结对照", "V5.4/V5.7", "七套旧排名完全保留；新增两套排名只检验历史完整度处理方式"),
    ], columns=["审计项目", "状态", "定义"])


def score_and_rank_events(events: pd.DataFrame, split_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = events.copy()
    work["Signal_Date"] = work["Signal_Date"].astype(str)
    work["Signal_Week"] = (
        pd.to_datetime(work["Signal_Date"], format="%Y%m%d")
        .dt.to_period("W-FRI").astype(str))
    work["Signal_Year"] = work["Signal_Date"].str[:4].astype(int)
    work["Validation_Period"] = np.where(
        work["Signal_Date"].le(str(split_end)), "前段观察", "后段冻结检验")
    streak = numeric(work, "Signal_Prior_Below25_Streak")
    work["Hard_Pass"] = streak.between(1, MAX_BOTTOM_STREAK, inclusive="both")
    work["Hard_Reject_Reason"] = np.where(
        streak.isna(), "连续处于25下方周数缺失",
        np.where(streak.gt(MAX_BOTTOM_STREAK), "连续处于25下方超过5周", ""))

    score_columns = [
        "Score_Reset_35", "Score_Volume_20", "Score_Candle_20",
        "Score_Trend_15", "Score_Price_Size_10", "Score_Total_100",
        "Weekly_Rank", "Candidates_This_Week", "Weekly_Rank_Pct",
    ]
    for column in score_columns:
        work[column] = np.nan
    for column in ("Top3", "Top5", "Top20Pct", "Bottom20Pct"):
        work[column] = False

    eligible = work[true_mask(work, "Hard_Pass")].copy()
    minimum_k = numeric(eligible, "Signal_Prior15_Min_K")
    volume_ratio = numeric(eligible, "Signal_Volume_Ratio_5W")
    close_location = numeric(eligible, "Signal_Close_Location_pct")
    upper_shadow = numeric(eligible, "Signal_Upper_Shadow_pct")
    ma20_slope = numeric(eligible, "Signal_MA20_Slope_4W_pct")
    ma20_distance = numeric(eligible, "Signal_Close_to_MA20_pct")

    eligible["Score_Reset_35"] = np.where(
        minimum_k.ge(15) & minimum_k.lt(25), 35.0, 0.0)
    eligible["Score_Volume_20"] = np.select(
        [volume_ratio.between(1.0, 2.5, inclusive="both"),
         volume_ratio.lt(1.0),
         volume_ratio.gt(4.0)],
        [20.0, 8.0, -10.0], default=0.0)
    eligible["Score_Candle_20"] = (
        close_location.between(60.0, 80.0, inclusive="both").astype(float) * 10.0
        + upper_shadow.between(10.0, 40.0, inclusive="both").astype(float) * 10.0)
    distance_score = np.select(
        [ma20_distance.between(5.0, 15.0, inclusive="both"),
         ma20_distance.ge(0) & ma20_distance.lt(5.0),
         ma20_distance.gt(15.0)],
        [7.0, 4.0, 2.0], default=0.0)
    eligible["Score_Trend_15"] = ma20_slope.ge(0).astype(float) * 8.0 + distance_score

    group_keys = ["SKDJ_N", "Signal_Week"]
    price_pct = eligible.groupby(group_keys)["Raw_Close"].rank(
        method="average", pct=True, ascending=True)
    size_pct = eligible.groupby(group_keys)["Circ_MV_Billion"].rank(
        method="average", pct=True, ascending=True)
    eligible["Score_Price_Size_10"] = price_pct.fillna(0.5) * 5.0 + size_pct.fillna(0.5) * 5.0
    eligible["Score_Total_100"] = (
        eligible["Score_Reset_35"] + eligible["Score_Volume_20"]
        + eligible["Score_Candle_20"] + eligible["Score_Trend_15"]
        + eligible["Score_Price_Size_10"]
    ).clip(lower=0.0, upper=100.0)

    eligible = eligible.sort_values([
        "SKDJ_N", "Signal_Week", "Score_Total_100", "Score_Reset_35",
        "Score_Volume_20", "Score_Candle_20", "Score_Trend_15",
        "Score_Price_Size_10", "ts_code",
    ], ascending=[True, True, False, False, False, False, False, False, True])
    eligible["Weekly_Rank"] = eligible.groupby(group_keys).cumcount() + 1
    eligible["Candidates_This_Week"] = eligible.groupby(group_keys)["ts_code"].transform("size")
    eligible["Weekly_Rank_Pct"] = (
        eligible["Weekly_Rank"] / eligible["Candidates_This_Week"])
    top20_count = np.ceil(eligible["Candidates_This_Week"] * 0.20).clip(lower=1)
    eligible["Top3"] = eligible["Weekly_Rank"].le(3)
    eligible["Top5"] = eligible["Weekly_Rank"].le(5)
    eligible["Top20Pct"] = eligible["Weekly_Rank"].le(top20_count)
    eligible["Bottom20Pct"] = eligible["Weekly_Rank"].gt(
        eligible["Candidates_This_Week"] - top20_count)
    eligible["Score_Band"] = pd.cut(
        eligible["Score_Total_100"], [-0.001, 40, 60, 75, 90, 100],
        labels=["0～40", "40～60", "60～75", "75～90", "90～100"],
        include_lowest=True).astype(str)

    update_columns = [
        "Hard_Pass", "Hard_Reject_Reason", *score_columns,
        "Top3", "Top5", "Top20Pct", "Bottom20Pct", "Score_Band",
    ]
    for column in update_columns:
        if column in eligible.columns:
            work.loc[eligible.index, column] = eligible[column]
    work = work.sort_values(["Signal_Date", "SKDJ_N", "Weekly_Rank", "ts_code"],
                            na_position="last").reset_index(drop=True)
    eligible = work[true_mask(work, "Hard_Pass")].copy()
    return work, eligible


def add_independent_candidate_features(eligible: pd.DataFrame) -> pd.DataFrame:
    """Add signal-time audit features without changing the frozen score.

    Industry-relative strength is explicitly relative to other eligible SKDJ
    candidates in the same N/week/SW_L1 group, not to a proprietary full-market
    industry index.  Groups with only one candidate are left missing rather than
    calling a one-stock group "strong".
    """
    if eligible.empty:
        return eligible.copy()
    work = eligible.copy()
    industry_keys = ["SKDJ_N", "Signal_Week", "SW_L1"]
    week_keys = ["SKDJ_N", "Signal_Week"]
    industry_counts = (
        work.groupby(industry_keys, dropna=False).size()
        .rename("Industry_Signal_Count").reset_index())
    industry_counts["Industry_Resonance_Rank_Pct"] = (
        industry_counts.groupby(week_keys)["Industry_Signal_Count"]
        .rank(method="average", pct=True, ascending=True) * 100.0)
    work = work.merge(industry_counts, on=industry_keys, how="left")
    work["Weekly_Eligible_Count"] = work.groupby(week_keys)["ts_code"].transform("size")
    work["Industry_Resonance_Pct"] = (
        numeric(work, "Industry_Signal_Count")
        / numeric(work, "Weekly_Eligible_Count").replace(0, np.nan) * 100.0)

    valid_industry = numeric(work, "Industry_Signal_Count").ge(2)
    for horizon in (4, 8, 12):
        source = f"Signal_Return_{horizon}W_pct"
        median_column = f"Industry_Candidate_Median_Return_{horizon}W_pct"
        relative_column = f"Signal_Relative_Industry_{horizon}W_pct"
        percentile_column = f"Signal_Relative_Industry_{horizon}W_Pctile"
        work[median_column] = work.groupby(industry_keys)[source].transform("median")
        work[relative_column] = (
            numeric(work, source) - numeric(work, median_column)).where(valid_industry)
        work[percentile_column] = (
            work.groupby(industry_keys)[source]
            .rank(method="average", pct=True, ascending=True) * 100.0
        ).where(valid_industry)
    return work.sort_values(
        ["Signal_Date", "SKDJ_N", "Weekly_Rank", "ts_code"],
        na_position="last").reset_index(drop=True)


def add_industry_breadth_features(
        eligible: pd.DataFrame, breadth_frame: pd.DataFrame) -> pd.DataFrame:
    if eligible.empty or breadth_frame.empty:
        return eligible.copy()
    work = eligible.copy()
    work["Signal_Date"] = work["Signal_Date"].astype(str)
    breadth = breadth_frame.copy()
    breadth["Signal_Date"] = breadth["Signal_Date"].astype(str)
    result = work.merge(
        breadth,
        left_on=["SKDJ_N", "Signal_Date", "SW_L1"],
        right_on=["SKDJ_N", "Signal_Date", "Breadth_Industry"],
        how="left", validate="many_to_one")
    return result.drop(columns=["Breadth_Industry"], errors="ignore").sort_values(
        ["Signal_Date", "SKDJ_N", "Weekly_Rank", "ts_code"],
        na_position="last").reset_index(drop=True)


def _assign_challenger_rank(
        frame: pd.DataFrame, prefix: str, sort_columns: list[str],
        ascending: list[bool]) -> pd.DataFrame:
    """Assign one complete weekly ranking while preserving original row identity."""
    work = frame.copy()
    keys = ["SKDJ_N", "Signal_Week"]
    ranked = work.sort_values(
        ["SKDJ_N", "Signal_Week", *sort_columns, "ts_code"],
        ascending=[True, True, *ascending, True])
    rank_column = f"{prefix}_Weekly_Rank"
    pct_column = f"{prefix}_Weekly_Rank_Pct"
    ranked[rank_column] = ranked.groupby(keys).cumcount() + 1
    rank_map = ranked[rank_column]
    work.loc[rank_map.index, rank_column] = rank_map
    candidate_count = numeric(work, "Candidates_This_Week").replace(0, np.nan)
    work[pct_column] = numeric(work, rank_column) / candidate_count
    top20_count = np.ceil(candidate_count * 0.20).clip(lower=1)
    work[f"{prefix}_Top3"] = numeric(work, rank_column).le(3)
    work[f"{prefix}_Top5"] = numeric(work, rank_column).le(5)
    work[f"{prefix}_Top20Pct"] = numeric(work, rank_column).le(top20_count)
    work[f"{prefix}_Bottom20Pct"] = numeric(work, rank_column).gt(
        candidate_count - top20_count)
    return work


def add_challenger_rankings(eligible: pd.DataFrame) -> pd.DataFrame:
    """Add the V5.2 control and V5.4 history-at-least-two tier ranking.

    Every input is known by the signal-week close.  Industry resonance and
    weekly candidate breadth receive labels for state/confidence audits but do
    not enter the V5.4 stock order.
    """
    if eligible.empty:
        return eligible.copy()
    work = eligible.copy()
    reached75 = numeric(
        work, "Signal_Prior_GC_Reached75_Count_Last3").fillna(0).clip(0, 3)
    valid_count = numeric(
        work, "Signal_Prior_GC_Valid_Count_Last3").fillna(0).clip(0, 3)
    history_complete = valid_count.ge(3)
    work["H2_History_Valid_Count"] = valid_count.astype(float)
    work["H2_History_Complete"] = history_complete
    work["H2_History_State"] = np.select(
        [valid_count.ge(3), valid_count.eq(2), valid_count.eq(1)],
        ["完整3次", "部分2次", "部分1次"], default="无有效周期")
    latest_peak = numeric(work, "Signal_Prior_GC1_Peak_K")
    resonance = numeric(work, "Industry_Resonance_Pct")
    resonance_40_60 = resonance.between(40.0, 60.0, inclusive="both")

    # Preserve V5.2's tier result as a direct control, without its discarded
    # 50+30+20 linear score.
    v52_s = reached75.ge(2) & resonance_40_60
    v52_a = ~v52_s & (reached75.ge(2) | latest_peak.ge(75.0))
    v52_b = ~v52_s & ~v52_a & (reached75.ge(1) | latest_peak.ge(50.0))
    work["V52_Tier_Level"] = np.select(
        [v52_s, v52_a, v52_b], ["S", "A", "B"], default="C")
    work["V52_Tier_Order"] = work["V52_Tier_Level"].map(
        {"S": 1.0, "A": 2.0, "B": 3.0, "C": 4.0}).astype(float)
    v52_reached_tie = np.select(
        [reached75.ge(3), reached75.eq(2), reached75.eq(1)],
        [40.0, 30.0, 10.0], default=0.0)
    v52_peak_tie = np.select(
        [latest_peak.ge(75.0), latest_peak.ge(50.0)],
        [10.0, 5.0], default=0.0)
    work["V52_History_Tie"] = np.minimum(v52_reached_tie + v52_peak_tie, 50.0)
    work["V52_Resonance_Tie"] = resonance_40_60.astype(float) * 30.0
    work = _assign_challenger_rank(
        work, "V52_Tier",
        ["V52_Tier_Order", "Score_Total_100",
         "V52_History_Tie", "V52_Resonance_Tie"],
        [True, False, False, False])

    # V5.4: reached 75 at least twice is one preferred tier.  Two and three
    # successes are deliberately equal; the frozen score owns the order inside
    # both tiers.  The latest completed-cycle peak can only break a frozen-score tie.
    work["H2_Tier_Level"] = np.where(reached75.ge(2), "S", "C")
    work["H2_Tier_Order"] = work["H2_Tier_Level"].map(
        {"S": 1.0, "C": 2.0}).astype(float)
    work = _assign_challenger_rank(
        work, "H2",
        ["H2_Tier_Order", "Score_Total_100", "Signal_Prior_GC1_Peak_K"],
        [True, False, False])

    # V5.8 completeness correction: incomplete history is neither success nor
    # failure.  It remains tradable and auditable, but is ranked behind stocks
    # with three completed historical cycles.
    h2c_s = history_complete & reached75.ge(2)
    work["H2C_Tier_Level"] = np.select(
        [h2c_s, history_complete], ["S", "C"], default="PX")
    work["H2C_Tier_Order"] = work["H2C_Tier_Level"].map(
        {"S": 1.0, "C": 2.0, "PX": 3.0}).astype(float)
    work = _assign_challenger_rank(
        work, "H2C",
        ["H2C_Tier_Order", "Score_Total_100", "Signal_Prior_GC1_Peak_K"],
        [True, False, False])

    candidate_count = numeric(work, "Candidates_This_Week")
    work["Week_Breadth_State"] = np.select(
        [candidate_count.le(4), candidate_count.le(20)],
        ["1～4只", "5～20只"], default=">20只")
    work["Week_Position_Confidence"] = np.select(
        [candidate_count.le(4), candidate_count.le(20)],
        ["低", "中"], default="高")
    work["Industry_Resonance_State"] = pd.cut(
        resonance, [-np.inf, 20.0, 40.0, 60.0, 80.0, np.inf],
        right=False, labels=["<20%", "20%～40%", "40%～60%", "60%～80%", "≥80%"]
    ).astype(str)
    return work.sort_values(
        ["Signal_Date", "SKDJ_N", "Weekly_Rank", "ts_code"],
        na_position="last").reset_index(drop=True)


def add_v57_explosion_rankings(eligible: pd.DataFrame) -> pd.DataFrame:
    """Apply four frozen rankings from the two V5.6.1 discoveries."""
    if eligible.empty:
        return eligible.copy()
    work = eligible.copy()
    keys = ["SKDJ_N", "Signal_Week"]
    k_efficiency = numeric(work, "Signal_K_Thrust_per_AbsWeekReturn")
    industry_ma20 = numeric(work, "Breadth_MA20_Rising_Pct")
    work["V57_K_Efficiency_Weekly_Pct"] = (
        work.assign(_feature=k_efficiency).groupby(keys)["_feature"]
        .rank(method="average", pct=True, ascending=True))
    work["V57_Industry_MA20_Weekly_Pct"] = (
        work.assign(_feature=industry_ma20).groupby(keys)["_feature"]
        .rank(method="average", pct=True, ascending=False))
    valid = k_efficiency.notna() & industry_ma20.notna()
    work["V57_K_Favorable20"] = (
        valid & numeric(work, "V57_K_Efficiency_Weekly_Pct").le(
            FEATURE_PRIORITY_FRACTION))
    work["V57_Breadth_Favorable20"] = (
        valid & numeric(work, "V57_Industry_MA20_Weekly_Pct").le(
            FEATURE_PRIORITY_FRACTION))
    favorable_count = (
        work[["V57_K_Favorable20", "V57_Breadth_Favorable20"]]
        .astype(int).sum(axis=1))
    work["V57_Favorable_Count"] = favorable_count.where(valid, np.nan)
    work["V57_Feature_Data_Valid"] = valid
    work["V57_Factor_Tier"] = np.select(
        [~valid, favorable_count.eq(2), favorable_count.eq(1)],
        ["PX", "P1", "P2"], default="P3")
    work["V57_Factor_Tier_Order"] = work["V57_Factor_Tier"].map(
        {"P1": 1.0, "P2": 2.0, "P3": 3.0, "PX": 4.0}).astype(float)
    work["V57_TwoFactor_Rank_Sum"] = (
        numeric(work, "V57_K_Efficiency_Weekly_Pct").fillna(1.0)
        + numeric(work, "V57_Industry_MA20_Weekly_Pct").fillna(1.0))

    work = _assign_challenger_rank(
        work, "V57_K",
        ["Signal_K_Thrust_per_AbsWeekReturn", "Score_Total_100"],
        [True, False])
    work = _assign_challenger_rank(
        work, "V57_Breadth",
        ["Breadth_MA20_Rising_Pct", "Score_Total_100"],
        [False, False])
    work = _assign_challenger_rank(
        work, "V57_Dual",
        ["V57_Factor_Tier_Order", "V57_TwoFactor_Rank_Sum", "Score_Total_100"],
        [True, True, False])
    work = _assign_challenger_rank(
        work, "V57_DualH2",
        ["V57_Factor_Tier_Order", "H2_Tier_Order",
         "V57_TwoFactor_Rank_Sum", "Score_Total_100"],
        [True, True, True, False])
    work = _assign_challenger_rank(
        work, "V58_DualH2C",
        ["V57_Factor_Tier_Order", "H2C_Tier_Order",
         "V57_TwoFactor_Rank_Sum", "Score_Total_100"],
        [True, True, True, False])
    return work.sort_values(
        ["Signal_Date", "SKDJ_N", "V58_DualH2C_Weekly_Rank", "ts_code"],
        na_position="last").reset_index(drop=True)


def build_periods(calendar: pd.DataFrame, split_ratio: float
                  ) -> tuple[str, list[dict[str, Any]]]:
    week_ends = calendar["Week_End"].astype(str).sort_values().drop_duplicates().tolist()
    if len(week_ends) < 2:
        raise ValueError("正式窗口不足2个市场周，无法时间分段")
    split_position = max(0, min(len(week_ends) - 2, math.ceil(len(week_ends) * split_ratio) - 1))
    split_end = week_ends[split_position]
    periods: list[dict[str, Any]] = [{
        "name": "全部区间", "start": week_ends[0], "end": week_ends[-1],
        "total_weeks": len(week_ends),
    }, {
        "name": "前段观察", "start": week_ends[0], "end": split_end,
        "total_weeks": split_position + 1,
    }, {
        "name": "后段冻结检验", "start": week_ends[split_position + 1], "end": week_ends[-1],
        "total_weeks": len(week_ends) - split_position - 1,
    }]
    for year in sorted({item[:4] for item in week_ends}):
        in_year = [item for item in week_ends if item.startswith(year)]
        periods.append({
            "name": f"年度{year}", "start": in_year[0], "end": in_year[-1],
            "total_weeks": len(in_year),
        })
    return split_end, periods


def select_period(frame: pd.DataFrame, period: dict[str, Any]) -> pd.DataFrame:
    dates = frame["Signal_Date"].astype(str)
    return frame[dates.between(str(period["start"]), str(period["end"]))].copy()


def cohort_metric_row(n: int, period_name: str, cohort_name: str,
                      selected: pd.DataFrame, total_weeks: int) -> dict[str, Any]:
    close_ret = numeric(selected, "Entry_W8_Close_Return_Net_pct")
    mfe1 = numeric(selected, "Entry_W1_MFE_Net_pct")
    mfe3 = numeric(selected, "Entry_W3_MFE_Net_pct")
    mfe8 = numeric(selected, "Entry_W8_MFE_Net_pct")
    mae8 = numeric(selected, "Entry_W8_MAE_Raw_pct")
    hit10 = selected.get(
        "Entry_First_Hit_10_vs_Minus10_W8",
        pd.Series(index=selected.index, dtype=str)).astype(str)
    hit20 = selected.get(
        "Entry_First_Hit_20_vs_Minus10_W8",
        pd.Series(index=selected.index, dtype=str)).astype(str)
    weekly_close = selected.assign(_ret=close_ret).groupby("Signal_Week")["_ret"].mean()
    week_counts = selected.groupby("Signal_Week").size()
    signal_weeks = selected["Signal_Week"].nunique() if not selected.empty else 0
    return {
        "SKDJ_N": n, "时间分段": period_name, "排名组": cohort_name,
        "事件数": len(selected),
        "不同股票": selected["ts_code"].nunique() if not selected.empty else 0,
        "有信号周": signal_weeks, "空窗周": max(total_weeks - signal_weeks, 0),
        "平均每个信号周事件": len(selected) / signal_weeks if signal_weeks else np.nan,
        "单周最多事件": week_counts.max() if not week_counts.empty else 0,
        "评分均值": numeric(selected, "Score_Total_100").mean(),
        "评分中位": numeric(selected, "Score_Total_100").median(),
        "W1最大浮盈中位%": mfe1.median(),
        "W3最大浮盈中位%": mfe3.median(),
        "W8最大浮盈中位%": mfe8.median(),
        "W8收盘平均净收益%": close_ret.mean(),
        "W8收盘中位净收益%": close_ret.median(),
        "W8每周等权平均净收益%": weekly_close.mean(),
        "W8每周等权净收益中位%": weekly_close.median(),
        "W8收盘胜率%": close_ret.gt(0).mean() * 100,
        "W8最大回撤中位%": mae8.median(),
        "W8触及-10%比例%": mae8.le(-10).mean() * 100,
        "先到+10比例%": hit10.eq("先到+10%").mean() * 100,
        "先到-10比例%_对比10": (
            hit10.eq("先到-10%") | hit10.str.startswith("同日同时触发")).mean() * 100,
        "先到+20比例%": hit20.eq("先到+20%").mean() * 100,
    }


def scheme_rank_cohort_audit(eligible: pd.DataFrame,
                             periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Compare all three old and four V5.7 rankings on identical candidates."""
    rows: list[dict[str, Any]] = []
    for spec in all_ranking_scheme_specs():
        for n in SKDJ_NS:
            base_n = eligible[eligible["SKDJ_N"].eq(n)]
            for period in periods:
                base = select_period(base_n, period)
                groups = [
                    ("全部硬条件通过", base),
                    ("周内前20%", base[true_mask(base, spec["top20"])]),
                    ("周内前5", base[true_mask(base, spec["top5"])]),
                    ("周内前3", base[true_mask(base, spec["top3"])]),
                    ("周内第4名以后", base[numeric(base, spec["rank"]).gt(3)]),
                    ("周内后20%", base[true_mask(base, spec["bottom20"])]),
                ]
                for label, selected in groups:
                    row = cohort_metric_row(
                        n, str(period["name"]), label, selected,
                        int(period["total_weeks"]))
                    row["排序方案"] = spec["name"]
                    row["方案排序值均值"] = numeric(selected, spec["score"]).mean()
                    tier_column = spec.get("tier", "")
                    row["S级占比%"] = (
                        selected.get(tier_column, pd.Series(
                            index=selected.index, dtype=str)).astype(str).eq("S").mean() * 100
                        if len(selected) and tier_column else np.nan)
                    rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    compare_metrics = [
        "W8收盘平均净收益%", "W8收盘中位净收益%", "W8每周等权平均净收益%",
        "W8收盘胜率%", "W8最大浮盈中位%", "W8最大回撤中位%",
        "W8触及-10%比例%", "先到+10比例%", "先到+20比例%",
    ]
    keys = ["SKDJ_N", "时间分段", "排名组"]
    baseline = result[result["排序方案"].eq("原冻结100分")][keys + compare_metrics].copy()
    baseline = baseline.rename(columns={metric: f"_baseline_{metric}" for metric in compare_metrics})
    result = result.merge(baseline, on=keys, how="left")
    for metric in compare_metrics:
        result[f"相对原评分_{metric}"] = (
            numeric(result, metric) - numeric(result, f"_baseline_{metric}"))
    v52 = result[result["排序方案"].eq("V5.2共振SABC对照")][
        keys + compare_metrics].copy()
    v52 = v52.rename(columns={metric: f"_v52_{metric}" for metric in compare_metrics})
    result = result.merge(v52, on=keys, how="left")
    for metric in compare_metrics:
        result[f"相对V5.2_{metric}"] = (
            numeric(result, metric) - numeric(result, f"_v52_{metric}"))
    hidden = ([f"_baseline_{metric}" for metric in compare_metrics]
              + [f"_v52_{metric}" for metric in compare_metrics])
    return result.drop(columns=hidden)


def scheme_rank_overlap_audit(eligible: pd.DataFrame,
                              periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            baseline_top3 = base[true_mask(base, "Top3")]
            baseline_top20 = base[true_mask(base, "Top20Pct")]
            for spec in ranking_scheme_specs()[1:]:
                challenger_top3 = base[true_mask(base, spec["top3"])]
                challenger_top20 = base[true_mask(base, spec["top20"])]
                rows.append({
                    "SKDJ_N": n, "时间分段": period["name"], "挑战方案": spec["name"],
                    "原评分前3事件": len(baseline_top3), "挑战方案前3事件": len(challenger_top3),
                    "前3重合事件": int(baseline_top3.index.isin(challenger_top3.index).sum()),
                    "前3重合率%": (
                        baseline_top3.index.isin(challenger_top3.index).mean() * 100
                        if len(baseline_top3) else np.nan),
                    "原评分前20%事件": len(baseline_top20),
                    "挑战方案前20%事件": len(challenger_top20),
                    "前20%重合事件": int(baseline_top20.index.isin(challenger_top20.index).sum()),
                    "前20%重合率%": (
                        baseline_top20.index.isin(challenger_top20.index).mean() * 100
                        if len(baseline_top20) else np.nan),
                })
    return pd.DataFrame(rows)


def scheme_acceptance_audit(rank_outcomes: pd.DataFrame) -> pd.DataFrame:
    """Pre-declared material-improvement check; equality across years is not required."""
    rows: list[dict[str, Any]] = []
    focus = rank_outcomes[
        ~rank_outcomes["排序方案"].eq("原冻结100分")
        & rank_outcomes["时间分段"].isin(["前段观察", "后段冻结检验"])
        & rank_outcomes["排名组"].isin(["周内前20%", "周内前3"])].copy()
    for _, row in focus.iterrows():
        mean_delta = finite_num(row.get("相对原评分_W8收盘平均净收益%"))
        median_delta = finite_num(row.get("相对原评分_W8收盘中位净收益%"))
        win_delta = finite_num(row.get("相对原评分_W8收盘胜率%"))
        stop_delta = finite_num(row.get("相对原评分_W8触及-10%比例%"))
        checks = {
            "平均收益至少提高2个百分点": bool(mean_delta >= 2.0),
            "中位收益至少提高2个百分点": bool(median_delta >= 2.0),
            "胜率至少提高3个百分点": bool(win_delta >= 3.0),
            "触及负10比例至少下降3个百分点": bool(stop_delta <= -3.0),
        }
        rows.append({
            "SKDJ_N": row["SKDJ_N"], "排序方案": row["排序方案"],
            "排名组": row["排名组"], "时间分段": row["时间分段"],
            "平均收益变化百分点": mean_delta,
            "中位收益变化百分点": median_delta,
            "胜率变化百分点": win_delta,
            "触及负10比例变化百分点": stop_delta,
            **checks,
            "明显改善项数_共4项": sum(checks.values()),
            "本阶段至少3项明显改善": sum(checks.values()) >= 3,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        keys = ["SKDJ_N", "排序方案", "排名组"]
        both = result.groupby(keys)["本阶段至少3项明显改善"].transform("all")
        count = result.groupby(keys)["时间分段"].transform("nunique")
        result["前后段均通过"] = both & count.eq(2)
    return result


def h2_vs_v52_acceptance_audit(rank_outcomes: pd.DataFrame) -> pd.DataFrame:
    """Apply the same frozen thresholds to V5.4 versus the V5.2 tier control."""
    rows: list[dict[str, Any]] = []
    focus = rank_outcomes[
        rank_outcomes["排序方案"].eq("V5.4历史≥2次优先")
        & rank_outcomes["时间分段"].isin(["前段观察", "后段冻结检验"])
        & rank_outcomes["排名组"].isin(["周内前20%", "周内前3"])].copy()
    for _, row in focus.iterrows():
        mean_delta = finite_num(row.get("相对V5.2_W8收盘平均净收益%"))
        median_delta = finite_num(row.get("相对V5.2_W8收盘中位净收益%"))
        win_delta = finite_num(row.get("相对V5.2_W8收盘胜率%"))
        stop_delta = finite_num(row.get("相对V5.2_W8触及-10%比例%"))
        checks = {
            "平均收益至少提高2个百分点": bool(mean_delta >= 2.0),
            "中位收益至少提高2个百分点": bool(median_delta >= 2.0),
            "胜率至少提高3个百分点": bool(win_delta >= 3.0),
            "触及负10比例至少下降3个百分点": bool(stop_delta <= -3.0),
        }
        rows.append({
            "SKDJ_N": row["SKDJ_N"], "排名组": row["排名组"],
            "时间分段": row["时间分段"],
            "平均收益变化百分点": mean_delta,
            "中位收益变化百分点": median_delta,
            "胜率变化百分点": win_delta,
            "触及负10比例变化百分点": stop_delta,
            **checks,
            "明显改善项数_共4项": sum(checks.values()),
            "本阶段至少3项明显改善": sum(checks.values()) >= 3,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        keys = ["SKDJ_N", "排名组"]
        both = result.groupby(keys)["本阶段至少3项明显改善"].transform("all")
        count = result.groupby(keys)["时间分段"].transform("nunique")
        result["前后段均超过V5.2"] = both & count.eq(2)
    return result


def rank_cohort_audit(eligible: pd.DataFrame,
                      periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            bottom_count = np.ceil(numeric(base, "Candidates_This_Week") * 0.20)
            groups = [
                ("全部硬条件通过", base),
                ("周内前20%", base[true_mask(base, "Top20Pct")]),
                ("周内前5", base[true_mask(base, "Top5")]),
                ("周内前3", base[true_mask(base, "Top3")]),
                ("周内第4名以后", base[numeric(base, "Weekly_Rank").gt(3)]),
                ("周内后20%", base[numeric(base, "Weekly_Rank").gt(
                    numeric(base, "Candidates_This_Week") - bottom_count)]),
            ]
            for label, selected in groups:
                rows.append(cohort_metric_row(
                    n, str(period["name"]), label, selected,
                    int(period["total_weeks"])))
    return pd.DataFrame(rows)


def weekly_rank_calendar(calendar: pd.DataFrame, all_events: pd.DataFrame,
                         eligible: pd.DataFrame, split_end: str) -> pd.DataFrame:
    result = calendar.copy()
    result["Week_End"] = result["Week_End"].astype(str)
    result["时间分段"] = np.where(
        result["Week_End"].le(split_end), "前段观察", "后段冻结检验")
    result["年份"] = result["Week_End"].str[:4]
    all_work = all_events.copy()
    # 事件只来自完整周，Signal_Date本身就是该市场周最后交易日。
    all_work["Week_End"] = all_work["Signal_Date"].astype(str)
    eligible_work = eligible.copy()
    eligible_work["Week_End"] = eligible_work["Signal_Date"].astype(str)
    for n in SKDJ_NS:
        raw_counts = all_work[all_work["SKDJ_N"].eq(n)].groupby("Week_End").size()
        group = eligible_work[eligible_work["SKDJ_N"].eq(n)]
        pass_counts = group.groupby("Week_End").size()
        top3_counts = group[true_mask(group, "Top3")].groupby("Week_End").size()
        top5_counts = group[true_mask(group, "Top5")].groupby("Week_End").size()
        v52_top3_counts = group[true_mask(
            group, "V52_Tier_Top3")].groupby("Week_End").size()
        h2_top3_counts = group[true_mask(
            group, "H2_Top3")].groupby("Week_End").size()
        h2_s_counts = group[group.get(
            "H2_Tier_Level", pd.Series(index=group.index, dtype=str))
            .astype(str).eq("S")].groupby("Week_End").size()
        h2c_top3_counts = group[true_mask(
            group, "H2C_Top3")].groupby("Week_End").size()
        h2_complete_counts = group[true_mask(
            group, "H2_History_Complete")].groupby("Week_End").size()
        result[f"N{n}_全部原始事件"] = result["Week_End"].map(raw_counts).fillna(0).astype(int)
        result[f"N{n}_硬条件通过"] = result["Week_End"].map(pass_counts).fillna(0).astype(int)
        result[f"N{n}_前3"] = result["Week_End"].map(top3_counts).fillna(0).astype(int)
        result[f"N{n}_前5"] = result["Week_End"].map(top5_counts).fillna(0).astype(int)
        result[f"N{n}_V5.2分层前3"] = result["Week_End"].map(
            v52_top3_counts).fillna(0).astype(int)
        result[f"N{n}_V5.4历史二层前3"] = result["Week_End"].map(
            h2_top3_counts).fillna(0).astype(int)
        result[f"N{n}_V5.4历史优先层事件"] = result["Week_End"].map(
            h2_s_counts).fillna(0).astype(int)
        result[f"N{n}_历史完整3次事件"] = result["Week_End"].map(
            h2_complete_counts).fillna(0).astype(int)
        result[f"N{n}_V5.8完整H2前3"] = result["Week_End"].map(
            h2c_top3_counts).fillna(0).astype(int)
        p1_counts = group[group.get(
            "V57_Factor_Tier", pd.Series(index=group.index, dtype=str))
            .astype(str).eq("P1")].groupby("Week_End").size()
        result[f"N{n}_V5.7双优P1事件"] = result["Week_End"].map(
            p1_counts).fillna(0).astype(int)
        for spec in v57_ranking_scheme_specs() + v58_ranking_scheme_specs():
            counts = group[true_mask(group, spec["top3"])].groupby("Week_End").size()
            result[f"N{n}_{spec['name']}前3"] = result["Week_End"].map(
                counts).fillna(0).astype(int)
        score_stats = group.groupby("Week_End")["Score_Total_100"].agg(["max", "median", "min"])
        for stat, cn in (("max", "最高分"), ("median", "中位分"), ("min", "最低分")):
            result[f"N{n}_{cn}"] = result["Week_End"].map(score_stats[stat])
    return result


def confidence_state_audit(eligible: pd.DataFrame,
                           periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Audit breadth and resonance as state labels, never as stock ranks."""
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            for dimension, column, ordered_values in [
                ("每周候选数量", "Week_Breadth_State", ["1～4只", "5～20只", ">20只"]),
                ("板块共振占比", "Industry_Resonance_State",
                 ["<20%", "20%～40%", "40%～60%", "60%～80%", "≥80%"]),
            ]:
                for state in ordered_values:
                    state_base = base[base[column].astype(str).eq(state)]
                    if state_base.empty:
                        continue
                    for spec in ranking_scheme_specs():
                        for group_name, selected in [
                            ("全部状态内事件", state_base),
                            ("该方案周内前20%", state_base[true_mask(state_base, spec["top20"])]),
                            ("该方案周内前3", state_base[true_mask(state_base, spec["top3"])]),
                        ]:
                            row = cohort_metric_row(
                                n, str(period["name"]), group_name, selected,
                                int(period["total_weeks"]))
                            row["置信度维度"] = dimension
                            row["状态分组"] = state
                            row["排序方案"] = spec["name"]
                            rows.append(row)
    return pd.DataFrame(rows)


def within_week_rank_ic_audit(eligible: pd.DataFrame,
                              periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Measure whether a feature can rank stocks inside the same signal week."""
    features = [
        ("原冻结100分", "Score_Total_100"),
        ("历史达到75次数", "Signal_Prior_GC_Reached75_Count_Last3"),
        ("最近一次金叉峰值", "Signal_Prior_GC1_Peak_K"),
        ("板块共振占比", "Industry_Resonance_Pct"),
        ("个股相对行业12周强度", "Signal_Relative_Industry_12W_pct"),
    ]
    rows: list[dict[str, Any]] = []
    return_column = "Entry_W8_Close_Return_Net_pct"
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            for feature_name, feature_column in features:
                weekly_ics: list[float] = []
                skipped = 0
                for _, group in base.groupby("Signal_Week"):
                    x = numeric(group, feature_column)
                    y = numeric(group, return_column)
                    valid = x.notna() & y.notna()
                    if (valid.sum() < 5 or x[valid].nunique() < 2
                            or y[valid].nunique() < 2):
                        skipped += 1
                        continue
                    weekly_ics.append(float(x[valid].corr(y[valid], method="spearman")))
                values = pd.Series(weekly_ics, dtype=float).dropna()
                rows.append({
                    "SKDJ_N": n, "时间分段": period["name"],
                    "周内排序特征": feature_name,
                    "有效周_至少5只且特征非恒定": len(values),
                    "跳过周": skipped,
                    "周内Spearman均值": values.mean(),
                    "周内Spearman中位": values.median(),
                    "周内Spearman为正比例%": values.gt(0).mean() * 100
                    if len(values) else np.nan,
                })
    return pd.DataFrame(rows)


def random_metric_values(selected: pd.DataFrame) -> dict[str, float]:
    close_ret = numeric(selected, "Entry_W8_Close_Return_Net_pct")
    weekly_close = selected.assign(_ret=close_ret).groupby("Signal_Week")["_ret"].mean()
    hit10 = selected.get(
        "Entry_First_Hit_10_vs_Minus10_W8",
        pd.Series(index=selected.index, dtype=str)).astype(str)
    return {
        "W8收盘平均净收益%": close_ret.mean(),
        "W8每周等权平均净收益%": weekly_close.mean(),
        "W8收盘胜率%": close_ret.gt(0).mean() * 100,
        "W8最大浮盈中位%": numeric(selected, "Entry_W8_MFE_Net_pct").median(),
        "W8触及-10%比例%": numeric(selected, "Entry_W8_MAE_Raw_pct").le(-10).mean() * 100,
        "先到+10比例%": hit10.eq("先到+10%").mean() * 100,
    }


def random_top3_benchmark(eligible: pd.DataFrame,
                          periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lower_is_better = {"W8触及-10%比例%"}
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period_number, period in enumerate(periods):
            base = select_period(base_n, period)
            if base.empty:
                continue
            top3 = base[true_mask(base, "Top3")]
            actual = random_metric_values(top3)
            weekly_indices = [group.index.to_numpy() for _, group in base.groupby("Signal_Week")]
            rng = np.random.default_rng(RANDOM_SEED + n * 1000 + period_number)
            draws: dict[str, list[float]] = {key: [] for key in actual}
            for _ in range(RANDOM_DRAWS):
                picked = np.concatenate([
                    rng.choice(indices, size=min(3, len(indices)), replace=False)
                    for indices in weekly_indices
                ])
                values = random_metric_values(base.loc[picked])
                for key, value in values.items():
                    draws[key].append(value)
            for metric, actual_value in actual.items():
                distribution = pd.Series(draws[metric], dtype=float).dropna()
                if metric in lower_is_better:
                    percentile = distribution.ge(actual_value).mean() * 100
                else:
                    percentile = distribution.le(actual_value).mean() * 100
                rows.append({
                    "SKDJ_N": n, "时间分段": period["name"], "指标": metric,
                    "正式前3": actual_value,
                    "随机3只P05": distribution.quantile(0.05),
                    "随机3只中位": distribution.median(),
                    "随机3只P95": distribution.quantile(0.95),
                    "正式前3优于随机百分位%": percentile,
                    "随机次数": RANDOM_DRAWS,
                })
    return pd.DataFrame(rows)


def scheme_random_top3_benchmark(eligible: pd.DataFrame,
                                 periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Use the same random draws for every scheme in one N/period comparison."""
    rows: list[dict[str, Any]] = []
    lower_is_better = {"W8触及-10%比例%"}
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period_number, period in enumerate(periods):
            base = select_period(base_n, period)
            if base.empty:
                continue
            weekly_indices = [
                group.index.to_numpy() for _, group in base.groupby("Signal_Week")]
            rng = np.random.default_rng(RANDOM_SEED + n * 1000 + period_number)
            sample_metrics = random_metric_values(base.iloc[0:0])
            draws: dict[str, list[float]] = {key: [] for key in sample_metrics}
            for _ in range(RANDOM_DRAWS):
                picked = np.concatenate([
                    rng.choice(indices, size=min(3, len(indices)), replace=False)
                    for indices in weekly_indices
                ])
                values = random_metric_values(base.loc[picked])
                for key, value in values.items():
                    draws[key].append(value)
            distributions = {
                key: pd.Series(values, dtype=float).dropna()
                for key, values in draws.items()}
            for spec in ranking_scheme_specs():
                selected = base[true_mask(base, spec["top3"])]
                actual = random_metric_values(selected)
                for metric, actual_value in actual.items():
                    distribution = distributions[metric]
                    percentile = (
                        distribution.ge(actual_value).mean() * 100
                        if metric in lower_is_better
                        else distribution.le(actual_value).mean() * 100)
                    rows.append({
                        "SKDJ_N": n, "时间分段": period["name"],
                        "排序方案": spec["name"], "指标": metric,
                        "方案前3": actual_value,
                        "随机3只P05": distribution.quantile(0.05),
                        "随机3只中位": distribution.median(),
                        "随机3只P95": distribution.quantile(0.95),
                        "方案前3优于随机百分位%": percentile,
                        "随机次数": RANDOM_DRAWS,
                    })
    return pd.DataFrame(rows)


def concentration_stress(eligible: pd.DataFrame,
                         periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_periods = [period for period in periods
                        if period["name"] in {"全部区间", "后段冻结检验"}
                        or str(period["name"]).startswith("年度")]
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in selected_periods:
            top3 = select_period(base_n, period)
            top3 = top3[true_mask(top3, "Top3")].copy()
            if top3.empty:
                continue
            scenarios: list[tuple[str, str, pd.DataFrame]] = [("原始", "", top3)]
            for count in (1, 3, 5):
                best_events = top3.nlargest(min(count, len(top3)), "Entry_W8_Close_Return_Net_pct")
                scenarios.append((
                    f"剔除收益最高{count}个事件",
                    ",".join(best_events["ts_code"].astype(str) + "@" + best_events["Signal_Date"].astype(str)),
                    top3.drop(index=best_events.index)))
                stock_contribution = top3.groupby("ts_code")["Entry_W8_Close_Return_Net_pct"].sum()
                best_stocks = stock_contribution.nlargest(min(count, len(stock_contribution))).index.astype(str)
                scenarios.append((
                    f"剔除贡献最高{count}只股票", ",".join(best_stocks),
                    top3[~top3["ts_code"].astype(str).isin(best_stocks)]))
                week_contribution = top3.groupby("Signal_Week")["Entry_W8_Close_Return_Net_pct"].mean()
                best_weeks = week_contribution.nlargest(min(count, len(week_contribution))).index.astype(str)
                scenarios.append((
                    f"剔除表现最好{count}个信号周", ",".join(best_weeks),
                    top3[~top3["Signal_Week"].astype(str).isin(best_weeks)]))
            for scenario, removed, selected in scenarios:
                row = cohort_metric_row(
                    n, str(period["name"]), scenario, selected,
                    int(period["total_weeks"]))
                row["剔除对象"] = removed
                rows.append(row)
    return pd.DataFrame(rows)


def scheme_concentration_stress(eligible: pd.DataFrame,
                                periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_periods = [
        period for period in periods
        if period["name"] in {"全部区间", "后段冻结检验"}
        or str(period["name"]).startswith("年度")]
    for spec in ranking_scheme_specs():
        for n in SKDJ_NS:
            base_n = eligible[eligible["SKDJ_N"].eq(n)]
            for period in selected_periods:
                top3 = select_period(base_n, period)
                top3 = top3[true_mask(top3, spec["top3"])].copy()
                if top3.empty:
                    continue
                scenarios: list[tuple[str, str, pd.DataFrame]] = [("原始", "", top3)]
                for count in (1, 3, 5):
                    best_events = top3.nlargest(
                        min(count, len(top3)), "Entry_W8_Close_Return_Net_pct")
                    scenarios.append((
                        f"剔除收益最高{count}个事件",
                        ",".join(best_events["ts_code"].astype(str)
                                 + "@" + best_events["Signal_Date"].astype(str)),
                        top3.drop(index=best_events.index)))
                    stock_contribution = top3.groupby(
                        "ts_code")["Entry_W8_Close_Return_Net_pct"].sum()
                    best_stocks = stock_contribution.nlargest(
                        min(count, len(stock_contribution))).index.astype(str)
                    scenarios.append((
                        f"剔除贡献最高{count}只股票", ",".join(best_stocks),
                        top3[~top3["ts_code"].astype(str).isin(best_stocks)]))
                    week_contribution = top3.groupby(
                        "Signal_Week")["Entry_W8_Close_Return_Net_pct"].mean()
                    best_weeks = week_contribution.nlargest(
                        min(count, len(week_contribution))).index.astype(str)
                    scenarios.append((
                        f"剔除表现最好{count}个信号周", ",".join(best_weeks),
                        top3[~top3["Signal_Week"].astype(str).isin(best_weeks)]))
                for scenario, removed, selected in scenarios:
                    row = cohort_metric_row(
                        n, str(period["name"]), scenario, selected,
                        int(period["total_weeks"]))
                    row["排序方案"] = spec["name"]
                    row["剔除对象"] = removed
                    rows.append(row)
    return pd.DataFrame(rows)


def parameter_agreement(eligible: pd.DataFrame,
                        periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in periods:
        frame = select_period(eligible, period)
        n6 = frame[frame["SKDJ_N"].eq(6)].copy()
        n7 = frame[frame["SKDJ_N"].eq(7)].copy()
        merged = n6.merge(
            n7, on=["ts_code", "Signal_Week"], suffixes=("_N6", "_N7"))
        top6 = n6[true_mask(n6, "Top3")][["ts_code", "Signal_Week"]].drop_duplicates()
        top7 = n7[true_mask(n7, "Top3")][["ts_code", "Signal_Week"]].drop_duplicates()
        overlap = top6.merge(top7, on=["ts_code", "Signal_Week"])
        denominator = min(len(top6), len(top7))
        rows.append({
            "时间分段": period["name"],
            "共同硬条件事件": len(merged),
            "共同事件评分Spearman": numeric(merged, "Score_Total_100_N6").corr(
                numeric(merged, "Score_Total_100_N7"), method="spearman") if len(merged) > 1 else np.nan,
            "N6前3事件": len(top6), "N7前3事件": len(top7),
            "前3相同股票同周": len(overlap),
            "前3重合率%": len(overlap) / denominator * 100 if denominator else np.nan,
        })
    return pd.DataFrame(rows)


def score_ablation_top3(eligible: pd.DataFrame,
                        periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Compare frozen score with component removals; all variants remain signal-time only."""
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)].copy()
        base_n["_Score_Full100"] = numeric(base_n, "Score_Total_100")
        base_n["_Score_NoPriceSize90"] = (
            numeric(base_n, "Score_Total_100")
            - numeric(base_n, "Score_Price_Size_10"))
        base_n["_Score_Structure75"] = (
            numeric(base_n, "Score_Reset_35")
            + numeric(base_n, "Score_Volume_20")
            + numeric(base_n, "Score_Candle_20"))
        base_n["_Score_EqualFive"] = (
            numeric(base_n, "Signal_Prior15_Min_K").ge(15).astype(int)
            + numeric(base_n, "Signal_Volume_Ratio_5W").between(1, 2.5).astype(int)
            + numeric(base_n, "Signal_MA20_Slope_4W_pct").ge(0).astype(int)
            + numeric(base_n, "Signal_Close_Location_pct").between(60, 80).astype(int)
            + numeric(base_n, "Signal_Upper_Shadow_pct").between(10, 40).astype(int))
        variants = [
            ("冻结完整100分", "_Score_Full100"),
            ("去掉价格市值_90分", "_Score_NoPriceSize90"),
            ("仅重置量能K线_75分", "_Score_Structure75"),
            ("五项等权计数_5分", "_Score_EqualFive"),
        ]
        for variant, column in variants:
            ranked = base_n.sort_values(
                ["Signal_Week", column, "Score_Total_100", "ts_code"],
                ascending=[True, False, False, True]).copy()
            ranked["_VariantRank"] = ranked.groupby("Signal_Week").cumcount() + 1
            ranked = ranked[ranked["_VariantRank"].le(3)]
            for period in periods:
                selected = select_period(ranked, period)
                row = cohort_metric_row(
                    n, str(period["name"]), "周内前3", selected,
                    int(period["total_weeks"]))
                row["评分变体"] = variant
                rows.append(row)
    return pd.DataFrame(rows)


def add_legacy_100b_ranks(eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the frozen V5.0 ranking inside the >=100bn-yuan legacy pool."""
    work = eligible.copy()
    legacy_columns = [
        "Legacy100_Score_Price_Size_10", "Legacy100_Score_Total_100",
        "Legacy100_Weekly_Rank", "Legacy100_Candidates_This_Week",
        "Legacy100_Weekly_Rank_Pct",
    ]
    for column in legacy_columns:
        work[column] = np.nan
    for column in ("Legacy100_Top3", "Legacy100_Top5", "Legacy100_Top20Pct"):
        work[column] = False

    legacy = work[numeric(work, "Circ_MV_Billion").ge(LEGACY_MIN_MV)].copy()
    if legacy.empty:
        return work, legacy
    keys = ["SKDJ_N", "Signal_Week"]
    price_pct = legacy.groupby(keys)["Raw_Close"].rank(
        method="average", pct=True, ascending=True)
    size_pct = legacy.groupby(keys)["Circ_MV_Billion"].rank(
        method="average", pct=True, ascending=True)
    legacy["Legacy100_Score_Price_Size_10"] = (
        price_pct.fillna(0.5) * 5.0 + size_pct.fillna(0.5) * 5.0)
    legacy["Legacy100_Score_Total_100"] = (
        numeric(legacy, "Score_Reset_35")
        + numeric(legacy, "Score_Volume_20")
        + numeric(legacy, "Score_Candle_20")
        + numeric(legacy, "Score_Trend_15")
        + numeric(legacy, "Legacy100_Score_Price_Size_10")
    ).clip(lower=0.0, upper=100.0)
    legacy = legacy.sort_values([
        "SKDJ_N", "Signal_Week", "Legacy100_Score_Total_100",
        "Score_Reset_35", "Score_Volume_20", "Score_Candle_20",
        "Score_Trend_15", "Legacy100_Score_Price_Size_10", "ts_code",
    ], ascending=[True, True, False, False, False, False, False, False, True])
    legacy["Legacy100_Weekly_Rank"] = legacy.groupby(keys).cumcount() + 1
    legacy["Legacy100_Candidates_This_Week"] = legacy.groupby(keys)["ts_code"].transform("size")
    legacy["Legacy100_Weekly_Rank_Pct"] = (
        legacy["Legacy100_Weekly_Rank"] / legacy["Legacy100_Candidates_This_Week"])
    top20_count = np.ceil(
        legacy["Legacy100_Candidates_This_Week"] * 0.20).clip(lower=1)
    legacy["Legacy100_Top3"] = legacy["Legacy100_Weekly_Rank"].le(3)
    legacy["Legacy100_Top5"] = legacy["Legacy100_Weekly_Rank"].le(5)
    legacy["Legacy100_Top20Pct"] = legacy["Legacy100_Weekly_Rank"].le(top20_count)
    for column in legacy_columns + [
            "Legacy100_Top3", "Legacy100_Top5", "Legacy100_Top20Pct"]:
        work.loc[legacy.index, column] = legacy[column]
    return work, work[numeric(work, "Circ_MV_Billion").ge(LEGACY_MIN_MV)].copy()


def legacy_rank_retention_audit(legacy: pd.DataFrame,
                                periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = legacy[legacy["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            old_top3 = base[true_mask(base, "Legacy100_Top3")]
            old_top20 = base[true_mask(base, "Legacy100_Top20Pct")]
            rows.append({
                "SKDJ_N": n, "时间分段": period["name"],
                "100亿旧池事件": len(base),
                "旧池前3事件": len(old_top3),
                "旧池前3仍在50亿前3": int(true_mask(old_top3, "Top3").sum()),
                "旧池前3仍在50亿前3比例%": (
                    true_mask(old_top3, "Top3").mean() * 100 if len(old_top3) else np.nan),
                "旧池前3进入50亿前5比例%": (
                    true_mask(old_top3, "Top5").mean() * 100 if len(old_top3) else np.nan),
                "旧池前3进入50亿前20%比例%": (
                    true_mask(old_top3, "Top20Pct").mean() * 100 if len(old_top3) else np.nan),
                "旧池前20%事件": len(old_top20),
                "旧池前20%仍在50亿前20%": int(true_mask(old_top20, "Top20Pct").sum()),
                "旧池前20%仍在50亿前20%比例%": (
                    true_mask(old_top20, "Top20Pct").mean() * 100 if len(old_top20) else np.nan),
                "旧池前20%进入50亿前5比例%": (
                    true_mask(old_top20, "Top5").mean() * 100 if len(old_top20) else np.nan),
                "旧池前20%进入50亿前3比例%": (
                    true_mask(old_top20, "Top3").mean() * 100 if len(old_top20) else np.nan),
            })
    return pd.DataFrame(rows)


def market_cap_cohort_audit(eligible: pd.DataFrame,
                            periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    size = numeric(eligible, "Circ_MV_Billion")
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            base_size = numeric(base, "Circ_MV_Billion")
            size_groups = [
                ("全部50亿以上", base),
                ("新增50～100亿", base[base_size.ge(50) & base_size.lt(100)]),
                ("原100亿以上", base[base_size.ge(100)]),
            ]
            for size_label, size_frame in size_groups:
                cohorts = [
                    (f"{size_label}_全部", size_frame),
                    (f"{size_label}_扩池前20%", size_frame[true_mask(size_frame, "Top20Pct")]),
                    (f"{size_label}_扩池前5", size_frame[true_mask(size_frame, "Top5")]),
                    (f"{size_label}_扩池前3", size_frame[true_mask(size_frame, "Top3")]),
                ]
                for label, selected in cohorts:
                    rows.append(cohort_metric_row(
                        n, str(period["name"]), label, selected,
                        int(period["total_weeks"])))
    return pd.DataFrame(rows)


def excellent_event_capture_audit(eligible: pd.DataFrame,
                                  periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = [
        ("W8收盘净收益>0", lambda frame: numeric(frame, "Entry_W8_Close_Return_Net_pct").gt(0)),
        ("W8收盘净收益≥10%", lambda frame: numeric(frame, "Entry_W8_Close_Return_Net_pct").ge(10)),
        ("W8收盘净收益≥20%", lambda frame: numeric(frame, "Entry_W8_Close_Return_Net_pct").ge(20)),
        ("W8最大浮盈≥20%", lambda frame: numeric(frame, "Entry_W8_MFE_Net_pct").ge(20)),
        ("先到+10%而非-10%", lambda frame: frame.get(
            "Entry_First_Hit_10_vs_Minus10_W8",
            pd.Series(index=frame.index, dtype=str)).astype(str).eq("先到+10%")),
    ]
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            period_base = select_period(base_n, period)
            period_size = numeric(period_base, "Circ_MV_Billion")
            size_groups = [
                ("全部50亿以上", period_base),
                ("新增50～100亿", period_base[
                    period_size.ge(50) & period_size.lt(100)]),
                ("原100亿以上", period_base[period_size.ge(100)]),
            ]
            for size_label, base in size_groups:
                for definition, mask_builder in definitions:
                    excellent = base[mask_builder(base)]
                    count = len(excellent)
                    rows.append({
                        "SKDJ_N": n, "时间分段": period["name"],
                        "市值组": size_label, "优秀定义": definition,
                        "优秀事件数": count,
                        "进入扩池前3": int(true_mask(excellent, "Top3").sum()),
                        "进入扩池前3比例%": (
                            true_mask(excellent, "Top3").mean() * 100 if count else np.nan),
                        "进入扩池前5": int(true_mask(excellent, "Top5").sum()),
                        "进入扩池前5比例%": (
                            true_mask(excellent, "Top5").mean() * 100 if count else np.nan),
                        "进入扩池前20%": int(true_mask(excellent, "Top20Pct").sum()),
                        "进入扩池前20%比例%": (
                            true_mask(excellent, "Top20Pct").mean() * 100 if count else np.nan),
                        "优秀事件评分均值": numeric(excellent, "Score_Total_100").mean(),
                        "优秀事件周内排名中位": numeric(excellent, "Weekly_Rank").median(),
                    })
    return pd.DataFrame(rows)


def scheme_excellent_capture_audit(eligible: pd.DataFrame,
                                   periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = [
        ("W8收盘净收益>0", lambda frame: numeric(
            frame, "Entry_W8_Close_Return_Net_pct").gt(0)),
        ("W8收盘净收益≥10%", lambda frame: numeric(
            frame, "Entry_W8_Close_Return_Net_pct").ge(10)),
        ("W8收盘净收益≥20%", lambda frame: numeric(
            frame, "Entry_W8_Close_Return_Net_pct").ge(20)),
        ("W8最大浮盈≥20%", lambda frame: numeric(
            frame, "Entry_W8_MFE_Net_pct").ge(20)),
        ("先到+10%而非-10%", lambda frame: frame.get(
            "Entry_First_Hit_10_vs_Minus10_W8",
            pd.Series(index=frame.index, dtype=str)).astype(str).eq("先到+10%")),
    ]
    for spec in ranking_scheme_specs():
        for n in SKDJ_NS:
            base_n = eligible[eligible["SKDJ_N"].eq(n)]
            for period in periods:
                period_base = select_period(base_n, period)
                period_size = numeric(period_base, "Circ_MV_Billion")
                size_groups = [
                    ("全部50亿以上", period_base),
                    ("新增50～100亿", period_base[
                        period_size.ge(50) & period_size.lt(100)]),
                    ("原100亿以上", period_base[period_size.ge(100)]),
                ]
                for size_label, base in size_groups:
                    for definition, mask_builder in definitions:
                        excellent = base[mask_builder(base)]
                        count = len(excellent)
                        rows.append({
                            "排序方案": spec["name"], "SKDJ_N": n,
                            "时间分段": period["name"], "市值组": size_label,
                            "优秀定义": definition, "优秀事件数": count,
                            "进入前3": int(true_mask(excellent, spec["top3"]).sum()),
                            "进入前3比例%": (
                                true_mask(excellent, spec["top3"]).mean() * 100
                                if count else np.nan),
                            "进入前5": int(true_mask(excellent, spec["top5"]).sum()),
                            "进入前5比例%": (
                                true_mask(excellent, spec["top5"]).mean() * 100
                                if count else np.nan),
                            "进入前20%": int(true_mask(excellent, spec["top20"]).sum()),
                            "进入前20%比例%": (
                                true_mask(excellent, spec["top20"]).mean() * 100
                                if count else np.nan),
                            "优秀事件方案排名中位": numeric(
                                excellent, spec["rank"]).median(),
                        })
    return pd.DataFrame(rows)


def independent_feature_audit(eligible: pd.DataFrame,
                              periods: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bucket_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    specs: list[tuple[str, list[float], list[str]]] = [
        ("Signal_Relative_Industry_4W_pct", [-np.inf, -10, 0, 10, np.inf],
         ["≤-10", "-10～0", "0～10", ">10"]),
        ("Signal_Relative_Industry_8W_pct", [-np.inf, -10, 0, 10, np.inf],
         ["≤-10", "-10～0", "0～10", ">10"]),
        ("Signal_Relative_Industry_12W_pct", [-np.inf, -15, 0, 15, np.inf],
         ["≤-15", "-15～0", "0～15", ">15"]),
        ("Industry_Signal_Count", [-np.inf, 1, 3, 7, np.inf],
         ["1", "2～3", "4～7", "≥8"]),
        ("Industry_Resonance_Pct", [-np.inf, 20, 40, 60, np.inf],
         ["≤20%", "20%～40%", "40%～60%", ">60%"]),
        ("Signal_Prior_GC1_Peak_K", [-np.inf, 50, 75, np.inf],
         ["<50", "50～75", "≥75"]),
        ("Signal_Prior_GC_MaxPeak_K_Last3", [-np.inf, 50, 75, np.inf],
         ["<50", "50～75", "≥75"]),
    ]
    audit_periods = [period for period in periods
                     if period["name"] in {"全部区间", "后段冻结检验"}
                     or str(period["name"]).startswith("年度")]
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in audit_periods:
            base = select_period(base_n, period)
            for feature, bins, labels in specs:
                values = numeric(base, feature)
                groups = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
                for label in labels:
                    selected = base[groups.astype(str).eq(label)]
                    row = cohort_metric_row(
                        n, str(period["name"]), str(label), selected,
                        int(period["total_weeks"]))
                    row["审计特征"] = feature
                    row["分组口径"] = "固定区间"
                    bucket_rows.append(row)
                valid = values.notna() & numeric(
                    base, "Entry_W8_Close_Return_Net_pct").notna()
                correlation_rows.append({
                    "SKDJ_N": n, "时间分段": period["name"], "审计特征": feature,
                    "有效事件": int(valid.sum()),
                    "与W8收盘收益Spearman": values[valid].corr(
                        numeric(base, "Entry_W8_Close_Return_Net_pct")[valid],
                        method="spearman") if valid.sum() > 2 else np.nan,
                    "与W8最大浮盈Spearman": values[valid].corr(
                        numeric(base, "Entry_W8_MFE_Net_pct")[valid],
                        method="spearman") if valid.sum() > 2 else np.nan,
                })

            count_values = numeric(base, "Signal_Prior_GC_Reached75_Count_Last3")
            for count in (0, 1, 2, 3):
                selected = base[count_values.eq(count)]
                row = cohort_metric_row(
                    n, str(period["name"]), str(count), selected,
                    int(period["total_weeks"]))
                row["审计特征"] = "Signal_Prior_GC_Reached75_Count_Last3"
                row["分组口径"] = "最近3次已完成金叉周期达到75的次数"
                bucket_rows.append(row)
    return pd.DataFrame(bucket_rows), pd.DataFrame(correlation_rows)


def discovery_feature_specs() -> list[dict[str, str]]:
    return [
        {"家族": "历史真实反应", "字段": "Hist_SameSignal_Valid_Count_Last3",
         "名称": "此前同类信号有效次数", "方向假设": "只表示置信度，不预设越多越好"},
        {"家族": "历史真实反应", "字段": "Hist_SameSignal_Last1_W8_MFE_Net_pct",
         "名称": "最近一次同类信号W8最大浮盈", "方向假设": "高可能更好"},
        {"家族": "历史真实反应", "字段": "Hist_SameSignal_W4_MFE_Median_Last3_pct",
         "名称": "最近三次同类信号W4最大浮盈中位", "方向假设": "高可能更好"},
        {"家族": "历史真实反应", "字段": "Hist_SameSignal_W8_MFE_Median_Last3_pct",
         "名称": "最近三次同类信号W8最大浮盈中位", "方向假设": "高可能更好"},
        {"家族": "历史真实反应", "字段": "Hist_SameSignal_W8_Close_Median_Last3_pct",
         "名称": "最近三次同类信号W8收盘收益中位", "方向假设": "高可能更好"},
        {"家族": "历史真实反应", "字段": "Hist_SameSignal_MFE_to_AbsMAE_Median_Last3",
         "名称": "最近三次同类信号浮盈回撤比中位", "方向假设": "高可能更好"},
        {"家族": "历史真实反应", "字段": "Hist_SameSignal_Hit10_Rate_Last3_pct",
         "名称": "最近三次同类信号先到+10比例", "方向假设": "高可能更好"},
        {"家族": "历史真实反应", "字段": "Hist_SameSignal_Hit20_Rate_Last3_pct",
         "名称": "最近三次同类信号先到+20比例", "方向假设": "高可能更好"},
        {"家族": "历史真实反应", "字段": "Hist_SameSignal_Hit30_Rate_Last3_pct",
         "名称": "最近三次同类信号先到+30比例", "方向假设": "高可能更好"},

        {"家族": "信号周日线路径", "字段": "SignalWeek_Up_Day_Pct",
         "名称": "信号周上涨天数比例", "方向假设": "待验证"},
        {"家族": "信号周日线路径", "字段": "SignalWeek_Signed_Path_Efficiency_Pct",
         "名称": "信号周有符号路径效率", "方向假设": "高可能更好"},
        {"家族": "信号周日线路径", "字段": "SignalWeek_One_Day_Gain_Share_Pct",
         "名称": "单日上涨贡献集中度", "方向假设": "过高可能较差"},
        {"家族": "信号周日线路径", "字段": "SignalWeek_Last2_Return_Pct",
         "名称": "信号周最后两日涨幅", "方向假设": "待验证持续还是追高"},
        {"家族": "信号周日线路径", "字段": "SignalWeek_High_Day_Position_Pct",
         "名称": "周内最高价出现位置", "方向假设": "靠后可能更强"},
        {"家族": "信号周日线路径", "字段": "SignalWeek_Low_Before_High",
         "名称": "周内低点先于高点", "方向假设": "真可能更好"},
        {"家族": "信号周日线路径", "字段": "Signal_K_Thrust_per_AbsWeekReturn",
         "名称": "K动能相对股价扩张效率", "方向假设": "高可能更早期"},

        {"家族": "全行业周状态", "字段": "Breadth_K_Rising_Pct",
         "名称": "行业K上升比例", "方向假设": "高可能更好"},
        {"家族": "全行业周状态", "字段": "Breadth_K_Above25_Pct",
         "名称": "行业K高于25比例", "方向假设": "待验证启动与拥挤"},
        {"家族": "全行业周状态", "字段": "Breadth_Above_MA20_Pct",
         "名称": "行业站上MA20比例", "方向假设": "高可能更好"},
        {"家族": "全行业周状态", "字段": "Breadth_MA20_Rising_Pct",
         "名称": "行业MA20上升比例", "方向假设": "高可能更好"},
        {"家族": "全行业周状态", "字段": "Breadth_Positive_Return4W_Pct",
         "名称": "行业近4周上涨比例", "方向假设": "高可能更好"},
        {"家族": "全行业周状态", "字段": "Breadth_Volume_Expand_Pct",
         "名称": "行业量能扩张比例", "方向假设": "待验证"},
        {"家族": "全行业周状态", "字段": "Industry_Relative_Tech_Return4W_Pct",
         "名称": "行业相对科技池4周强度", "方向假设": "高可能更好"},
        {"家族": "全行业周状态", "字段": "Industry_Relative_Tech_K_Rising_Pct",
         "名称": "行业相对科技池K上升广度", "方向假设": "高可能更好"},
        {"家族": "全行业周状态", "字段": "Industry_Relative_Tech_Above_MA20_Pct",
         "名称": "行业相对科技池MA20广度", "方向假设": "高可能更好"},

        {"家族": "方向性量价效率", "字段": "SignalWeek_Up_Down_Volume_Ratio",
         "名称": "上涨日与下跌日成交量比", "方向假设": "高可能更好"},
        {"家族": "方向性量价效率", "字段": "SignalWeek_Up_Volume_Share_Pct",
         "名称": "上涨日成交量占比", "方向假设": "高可能更好"},
        {"家族": "方向性量价效率", "字段": "SignalWeek_Up_Down_Turnover_Ratio",
         "名称": "上涨日与下跌日换手比", "方向假设": "高可能更好"},
        {"家族": "方向性量价效率", "字段": "SignalWeek_Amount_Ratio_vs_Prior20D",
         "名称": "信号周成交额相对前20日", "方向假设": "中等可能更好"},
        {"家族": "方向性量价效率", "字段": "SignalWeek_Turnover20_CV",
         "名称": "近20日换手波动系数", "方向假设": "过高可能较差"},
        {"家族": "方向性量价效率", "字段": "SignalWeek_Abs_Return_Per_Amount_20D",
         "名称": "近20日单位成交额价格弹性", "方向假设": "待验证"},
    ]


def discovery_feature_definitions() -> pd.DataFrame:
    frame = pd.DataFrame(discovery_feature_specs())
    frame["是否计分"] = "否，仅审计"
    frame["防未来数据"] = "仅使用信号日收盘时已经可知的数据"
    return frame


def breadth_bucket_masks(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    counts = numeric(frame, "Candidates_This_Week")
    return [
        ("全部宽度", pd.Series(True, index=frame.index)),
        ("1～5只", counts.between(1, 5, inclusive="both")),
        ("6～15只", counts.between(6, 15, inclusive="both")),
        ("16～25只", counts.between(16, 25, inclusive="both")),
        (">25只", counts.gt(25)),
    ]


def add_explosion_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add mutually exclusive S/A/B/F labels without changing any ranking.

    The labels are research outcomes, not exit rules: S means +30% was reached
    before -10% inside the horizon; A means +20% but not S; B means +10% but
    not A/S; F contains every other mature path. Immature W12 rows stay blank.
    """
    work = frame.copy()
    for horizon in FIRST_HIT_AUDIT_WEEKS:
        mature = true_mask(work, f"Entry_Has_W{horizon}")
        hit10 = work.get(
            f"Entry_First_Hit_10_vs_Minus10_W{horizon}",
            pd.Series(index=work.index, dtype=str)).astype(str).eq("先到+10%")
        hit20 = work.get(
            f"Entry_First_Hit_20_vs_Minus10_W{horizon}",
            pd.Series(index=work.index, dtype=str)).astype(str).eq("先到+20%")
        hit30 = work.get(
            f"Entry_First_Hit_30_vs_Minus10_W{horizon}",
            pd.Series(index=work.index, dtype=str)).astype(str).eq("先到+30%")
        labels = pd.Series("", index=work.index, dtype=object)
        labels.loc[mature] = "F"
        labels.loc[mature & hit10] = "B"
        labels.loc[mature & hit20] = "A"
        labels.loc[mature & hit30] = "S"
        work[f"Explosion_Class_W{horizon}"] = labels
        work[f"Explosion_Grade_W{horizon}"] = labels.map(
            {"F": 0.0, "B": 1.0, "A": 2.0, "S": 3.0})
        work[f"Explosion_B_or_Better_W{horizon}"] = mature & hit10
        work[f"Explosion_A_or_S_W{horizon}"] = mature & hit20
        work[f"Explosion_S_W{horizon}"] = mature & hit30
    return work


def discovery_target_series(frame: pd.DataFrame, target: str) -> pd.Series:
    if target == "W8爆发等级":
        return numeric(frame, "Explosion_Grade_W8")
    if target == "W8收盘净收益":
        return numeric(frame, "Entry_W8_Close_Return_Net_pct")
    if target == "W8最大浮盈":
        return numeric(frame, "Entry_W8_MFE_Net_pct")
    if target == "先到+10而非-10":
        labels = frame.get(
            "Entry_First_Hit_10_vs_Minus10_W8",
            pd.Series(index=frame.index, dtype=str)).astype(str)
        return labels.eq("先到+10%").astype(float)
    if target == "先到+20而非-10":
        labels = frame.get(
            "Entry_First_Hit_20_vs_Minus10_W8",
            pd.Series(index=frame.index, dtype=str)).astype(str)
        return labels.eq("先到+20%").astype(float)
    if target == "先到+30而非-10":
        labels = frame.get(
            "Entry_First_Hit_30_vs_Minus10_W8",
            pd.Series(index=frame.index, dtype=str)).astype(str)
        return labels.eq("先到+30%").astype(float)
    if target == "较小W8不利波动":
        return -numeric(frame, "Entry_W8_MAE_Raw_pct").abs()
    if target == "W12最大浮盈_补充":
        result = numeric(frame, "Entry_W12_MFE_Net_pct")
        return result.where(true_mask(frame, "Entry_Has_W12"))
    if target in {"W12先到+20_补充", "W12先到+30_补充"}:
        level = 20 if "+20" in target else 30
        labels = frame.get(
            f"Entry_First_Hit_{level}_vs_Minus10_W12",
            pd.Series(index=frame.index, dtype=str)).astype(str)
        result = labels.eq(f"先到+{level}%").astype(float)
        return result.where(true_mask(frame, "Entry_Has_W12"))
    raise KeyError(target)


PRIMARY_DISCOVERY_TARGETS = (
    "W8爆发等级", "W8最大浮盈",
    "先到+20而非-10", "先到+30而非-10",
)
SUPPLEMENTARY_DISCOVERY_TARGETS = (
    "W8收盘净收益", "先到+10而非-10", "较小W8不利波动",
)
DISCOVERY_TARGETS = PRIMARY_DISCOVERY_TARGETS + SUPPLEMENTARY_DISCOVERY_TARGETS


def weekly_spearman_values(weeks: pd.Series, x: pd.Series,
                           y: pd.Series) -> tuple[pd.Series, int]:
    """Vectorized equal-week Spearman correlations."""
    frame = pd.DataFrame({"week": weeks.astype(str), "x": x, "y": y}).dropna()
    total_weeks = int(weeks.astype(str).nunique())
    if frame.empty:
        return pd.Series(dtype=float), total_weeks
    grouped = frame.groupby("week", sort=False)
    stats = grouped.agg(count=("x", "size"), x_unique=("x", "nunique"),
                        y_unique=("y", "nunique"))
    valid_weeks = stats.index[
        stats["count"].ge(FEATURE_MIN_WEEK_SIZE)
        & stats["x_unique"].ge(2) & stats["y_unique"].ge(2)]
    frame = frame[frame["week"].isin(valid_weeks)].copy()
    if frame.empty:
        return pd.Series(dtype=float), total_weeks
    frame["rx"] = frame.groupby("week")["x"].rank(method="average")
    frame["ry"] = frame.groupby("week")["y"].rank(method="average")
    frame["dx"] = frame["rx"] - frame.groupby("week")["rx"].transform("mean")
    frame["dy"] = frame["ry"] - frame.groupby("week")["ry"].transform("mean")
    frame["cross"] = frame["dx"] * frame["dy"]
    frame["x2"] = frame["dx"] ** 2
    frame["y2"] = frame["dy"] ** 2
    moments = frame.groupby("week")[["cross", "x2", "y2"]].sum()
    correlations = moments["cross"] / np.sqrt(
        moments["x2"] * moments["y2"]).replace(0, np.nan)
    correlations = correlations.replace([np.inf, -np.inf], np.nan).dropna()
    return correlations.astype(float), total_weeks - len(correlations)


def new_feature_rank_ic_audit(eligible: pd.DataFrame,
                              periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            period_base = select_period(base_n, period)
            for width, mask in breadth_bucket_masks(period_base):
                base = period_base[mask]
                if base.empty:
                    continue
                for spec in discovery_feature_specs():
                    x_all = numeric(base, spec["字段"])
                    for target in DISCOVERY_TARGETS:
                        y_all = discovery_target_series(base, target)
                        values, skipped = weekly_spearman_values(
                            base["Signal_Week"], x_all, y_all)
                        rows.append({
                            "SKDJ_N": n, "时间分段": period["name"],
                            "候选宽度": width, "特征家族": spec["家族"],
                            "特征": spec["名称"], "字段": spec["字段"],
                            "评价目标": target,
                            "目标地位": (
                                "主目标：爆发力" if target in PRIMARY_DISCOVERY_TARGETS
                                else "辅助观察：不参与入选判定"),
                            "有效周": int(len(values)), "跳过周": int(skipped),
                            "周内Spearman均值": values.mean(),
                            "周内Spearman中位": values.median(),
                            "Spearman为正周比例%": (
                                values.gt(0).mean() * 100 if len(values) else np.nan),
                        })
    return pd.DataFrame(rows)


def discovery_outcome_metrics(frame: pd.DataFrame) -> dict[str, float]:
    close = numeric(frame, "Entry_W8_Close_Return_Net_pct")
    mfe = numeric(frame, "Entry_W8_MFE_Net_pct")
    mae = numeric(frame, "Entry_W8_MAE_Raw_pct")
    hit10 = discovery_target_series(frame, "先到+10而非-10")
    hit20 = discovery_target_series(frame, "先到+20而非-10")
    hit30 = discovery_target_series(frame, "先到+30而非-10")
    classes = frame.get(
        "Explosion_Class_W8", pd.Series(index=frame.index, dtype=str)).astype(str)
    return {
        "事件数": float(len(frame)),
        "W8平均净收益%": close.mean(),
        "W8中位净收益%": close.median(),
        "W8胜率%": close.gt(0).mean() * 100 if len(close) else np.nan,
        "W8最大浮盈均值%": mfe.mean(),
        "W8最大回撤均值%": mae.mean(),
        "先到+10比例%": hit10.mean() * 100 if len(hit10) else np.nan,
        "先到+20比例%": hit20.mean() * 100 if len(hit20) else np.nan,
        "先到+30比例%": hit30.mean() * 100 if len(hit30) else np.nan,
        "S级比例%": classes.eq("S").mean() * 100 if len(classes) else np.nan,
        "A或S比例%": classes.isin(["A", "S"]).mean() * 100 if len(classes) else np.nan,
        "B及以上比例%": classes.isin(["B", "A", "S"]).mean() * 100 if len(classes) else np.nan,
        "F级比例%": classes.eq("F").mean() * 100 if len(classes) else np.nan,
    }


def new_feature_quintile_spread_audit(
        eligible: pd.DataFrame, periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = [
        "W8平均净收益%", "W8中位净收益%", "W8胜率%",
        "W8最大浮盈均值%", "W8最大回撤均值%",
        "先到+10比例%", "先到+20比例%", "先到+30比例%",
        "S级比例%", "A或S比例%", "B及以上比例%", "F级比例%",
    ]
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            period_base = select_period(base_n, period)
            for width, mask in breadth_bucket_masks(period_base):
                base = period_base[mask]
                if base.empty:
                    continue
                for spec in discovery_feature_specs():
                    high_indices: list[Any] = []
                    low_indices: list[Any] = []
                    valid_weeks = 0
                    for _, group in base.groupby("Signal_Week"):
                        values = numeric(group, spec["字段"]).dropna()
                        if len(values) < FEATURE_MIN_WEEK_SIZE or values.nunique() < 2:
                            continue
                        count = max(1, int(math.ceil(len(values) * 0.20)))
                        ordered = values.sort_values(kind="mergesort")
                        low_indices.extend(ordered.head(count).index.tolist())
                        high_indices.extend(ordered.tail(count).index.tolist())
                        valid_weeks += 1
                    high = base.loc[base.index.intersection(high_indices)]
                    low = base.loc[base.index.intersection(low_indices)]
                    high_metrics = discovery_outcome_metrics(high)
                    low_metrics = discovery_outcome_metrics(low)
                    row: dict[str, Any] = {
                        "SKDJ_N": n, "时间分段": period["name"],
                        "候选宽度": width, "特征家族": spec["家族"],
                        "特征": spec["名称"], "字段": spec["字段"],
                        "方向假设": spec["方向假设"], "有效周": valid_weeks,
                        "高20%特征值中位": numeric(high, spec["字段"]).median(),
                        "低20%特征值中位": numeric(low, spec["字段"]).median(),
                    }
                    row["高20%事件数"] = int(high_metrics["事件数"])
                    row["低20%事件数"] = int(low_metrics["事件数"])
                    for metric in metric_names:
                        row[f"高20%_{metric}"] = high_metrics[metric]
                        row[f"低20%_{metric}"] = low_metrics[metric]
                        row[f"高减低_{metric}"] = high_metrics[metric] - low_metrics[metric]
                    rows.append(row)
    return pd.DataFrame(rows)


def new_feature_redundancy_audit(
        eligible: pd.DataFrame, periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baselines = [
        ("原冻结100分", "Score_Total_100"),
        ("历史达到75次数", "Signal_Prior_GC_Reached75_Count_Last3"),
    ]
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            if base.empty:
                continue
            for spec in discovery_feature_specs():
                for baseline_name, baseline_field in baselines:
                    values, _ = weekly_spearman_values(
                        base["Signal_Week"], numeric(base, spec["字段"]),
                        numeric(base, baseline_field))
                    rows.append({
                        "SKDJ_N": n, "时间分段": period["name"],
                        "特征家族": spec["家族"], "特征": spec["名称"],
                        "字段": spec["字段"], "对照": baseline_name,
                        "有效周": len(values), "周内相关均值": values.mean(),
                        "周内相关绝对值均值": values.abs().mean(),
                        "周内相关中位": values.median(),
                    })
    return pd.DataFrame(rows)


def new_feature_similar_pair_audit(
        eligible: pd.DataFrame, periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Compare similar frozen-rank stocks using explosion outcomes first."""
    rows: list[dict[str, Any]] = []
    audit_periods = [period for period in periods if period["name"] in {
        "全部区间", "前段观察", "后段冻结检验"}]
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in audit_periods:
            period_base = select_period(base_n, period)
            for width, mask in breadth_bucket_masks(period_base):
                base = period_base[mask]
                if base.empty:
                    continue
                adjacent_pairs: list[tuple[str, Any, Any]] = []
                for week_name, week in base.groupby("Signal_Week"):
                    for _, tier in week.groupby("H2_Tier_Level", dropna=False):
                        ordered = tier.sort_values(
                            ["Score_Total_100", "ts_code"],
                            ascending=[False, True])
                        ordered_indices = ordered.index.tolist()
                        for position in range(len(ordered_indices) - 1):
                            left_index = ordered_indices[position]
                            right_index = ordered_indices[position + 1]
                            left_score = finite_num(base.loc[
                                left_index, "Score_Total_100"])
                            right_score = finite_num(base.loc[
                                right_index, "Score_Total_100"])
                            if (not math.isfinite(left_score)
                                    or not math.isfinite(right_score)
                                    or abs(left_score - right_score) > 5.0):
                                continue
                            adjacent_pairs.append(
                                (str(week_name), left_index, right_index))
                for spec in discovery_feature_specs():
                    mfe_by_week: dict[str, list[float]] = {}
                    close_by_week: dict[str, list[float]] = {}
                    hit20_by_week: dict[str, list[float]] = {}
                    hit30_by_week: dict[str, list[float]] = {}
                    for week_name, left_index, right_index in adjacent_pairs:
                        left, right = base.loc[left_index], base.loc[right_index]
                        x_left = finite_num(left.get(spec["字段"]))
                        x_right = finite_num(right.get(spec["字段"]))
                        mfe_left = finite_num(left.get("Entry_W8_MFE_Net_pct"))
                        mfe_right = finite_num(right.get("Entry_W8_MFE_Net_pct"))
                        if (math.isfinite(x_left) and math.isfinite(x_right)
                                and x_left != x_right and math.isfinite(mfe_left)
                                and math.isfinite(mfe_right) and mfe_left != mfe_right):
                            mfe_by_week.setdefault(week_name, []).append(float(
                                (x_left - x_right) * (mfe_left - mfe_right) > 0))
                        y_left = finite_num(left.get("Entry_W8_Close_Return_Net_pct"))
                        y_right = finite_num(right.get("Entry_W8_Close_Return_Net_pct"))
                        if (math.isfinite(x_left) and math.isfinite(x_right)
                                and x_left != x_right and math.isfinite(y_left)
                                and math.isfinite(y_right) and y_left != y_right):
                            close_by_week.setdefault(week_name, []).append(float(
                                (x_left - x_right) * (y_left - y_right) > 0))
                        for level, store in ((20, hit20_by_week), (30, hit30_by_week)):
                            h_left = str(left.get(
                                f"Entry_First_Hit_{level}_vs_Minus10_W8", "")) == f"先到+{level}%"
                            h_right = str(right.get(
                                f"Entry_First_Hit_{level}_vs_Minus10_W8", "")) == f"先到+{level}%"
                            if (math.isfinite(x_left) and math.isfinite(x_right)
                                    and x_left != x_right and h_left != h_right):
                                store.setdefault(week_name, []).append(float(
                                    (x_left > x_right) == (h_left and not h_right)))
                    weekly_mfe_accuracy = [
                        float(np.mean(values)) for values in mfe_by_week.values()]
                    weekly_close_accuracy = [
                        float(np.mean(values)) for values in close_by_week.values()]
                    weekly_hit20_accuracy = [
                        float(np.mean(values)) for values in hit20_by_week.values()]
                    weekly_hit30_accuracy = [
                        float(np.mean(values)) for values in hit30_by_week.values()]
                    rows.append({
                        "SKDJ_N": n, "时间分段": period["name"],
                        "候选宽度": width, "特征家族": spec["家族"],
                        "特征": spec["名称"], "字段": spec["字段"],
                        "相似股票浮盈有效周": len(weekly_mfe_accuracy),
                        "相似股票浮盈有效对数": sum(
                            len(values) for values in mfe_by_week.values()),
                        "较高特征值选对更高W8最大浮盈_周均比例%": (
                            np.mean(weekly_mfe_accuracy) * 100
                            if weekly_mfe_accuracy else np.nan),
                        "相似股票期末收益有效周_辅助": len(weekly_close_accuracy),
                        "相似股票期末收益有效对数_辅助": sum(
                            len(values) for values in close_by_week.values()),
                        "较高特征值选对更高W8期末收益_辅助%": (
                            np.mean(weekly_close_accuracy) * 100
                            if weekly_close_accuracy else np.nan),
                        "相似股票+20有效周": len(weekly_hit20_accuracy),
                        "相似股票+20有效对数": sum(
                            len(values) for values in hit20_by_week.values()),
                        "较高特征值选对先到+20_周均比例%": (
                            np.mean(weekly_hit20_accuracy) * 100
                            if weekly_hit20_accuracy else np.nan),
                        "相似股票+30有效周": len(weekly_hit30_accuracy),
                        "相似股票+30有效对数": sum(
                            len(values) for values in hit30_by_week.values()),
                        "较高特征值选对先到+30_周均比例%": (
                            np.mean(weekly_hit30_accuracy) * 100
                            if weekly_hit30_accuracy else np.nan),
                    })
    return pd.DataFrame(rows)


def discovery_shortlist(ic: pd.DataFrame, spreads: pd.DataFrame,
                        redundancy: pd.DataFrame,
                        pair_audit: pd.DataFrame) -> pd.DataFrame:
    """Select factors by explosion power, never by smoothness or low volatility."""
    rows: list[dict[str, Any]] = []
    for spec in discovery_feature_specs():
        field = spec["字段"]
        base = ic[
            ic["SKDJ_N"].eq(6)
            & ic["候选宽度"].eq("全部宽度")
            & ic["字段"].eq(field)]
        def lookup(period: str, target: str, column: str) -> float:
            selected = base[
                base["时间分段"].eq(period)
                & base["评价目标"].eq(target)]
            return finite_num(selected.iloc[0].get(column)) if len(selected) else np.nan

        full_grade = lookup("全部区间", "W8爆发等级", "周内Spearman均值")
        full_mfe = lookup("全部区间", "W8最大浮盈", "周内Spearman均值")
        full_hit20 = lookup("全部区间", "先到+20而非-10", "周内Spearman均值")
        full_hit30 = lookup("全部区间", "先到+30而非-10", "周内Spearman均值")
        front_grade = lookup("前段观察", "W8爆发等级", "周内Spearman均值")
        back_grade = lookup("后段冻结检验", "W8爆发等级", "周内Spearman均值")
        front_mfe = lookup("前段观察", "W8最大浮盈", "周内Spearman均值")
        back_mfe = lookup("后段冻结检验", "W8最大浮盈", "周内Spearman均值")
        front_hit20 = lookup("前段观察", "先到+20而非-10", "周内Spearman均值")
        back_hit20 = lookup("后段冻结检验", "先到+20而非-10", "周内Spearman均值")
        front_hit30 = lookup("前段观察", "先到+30而非-10", "周内Spearman均值")
        back_hit30 = lookup("后段冻结检验", "先到+30而非-10", "周内Spearman均值")
        front_week_values = [value for value in (
            lookup("前段观察", "W8爆发等级", "有效周"),
            lookup("前段观察", "W8最大浮盈", "有效周")) if math.isfinite(value)]
        back_week_values = [value for value in (
            lookup("后段冻结检验", "W8爆发等级", "有效周"),
            lookup("后段冻结检验", "W8最大浮盈", "有效周")) if math.isfinite(value)]
        front_weeks = max(front_week_values) if front_week_values else np.nan
        back_weeks = max(back_week_values) if back_week_values else np.nan
        outcome_values = [value for value in (
            full_grade, full_mfe, full_hit20, full_hit30) if math.isfinite(value)]
        combined = float(np.mean(outcome_values)) if outcome_values else np.nan
        direction = 1 if math.isfinite(combined) and combined > 0 else -1
        front_values = [value for value in (
            front_grade, front_mfe, front_hit20, front_hit30) if math.isfinite(value)]
        back_values = [value for value in (
            back_grade, back_mfe, back_hit20, back_hit30) if math.isfinite(value)]
        front_combined = float(np.mean(front_values)) if front_values else np.nan
        back_combined = float(np.mean(back_values)) if back_values else np.nan
        same_period_direction = (
            math.isfinite(front_combined) and math.isfinite(back_combined)
            and front_combined * back_combined > 0)
        same_objective_direction = (
            math.isfinite(full_grade) and math.isfinite(full_mfe)
            and full_grade * full_mfe > 0
            and (not math.isfinite(full_hit20) or full_hit20 * direction > 0))
        enough = (math.isfinite(front_weeks) and math.isfinite(back_weeks)
                  and front_weeks >= 15 and back_weeks >= 15)
        meaningful = math.isfinite(combined) and abs(combined) >= 0.08

        spread_row = spreads[
            spreads["SKDJ_N"].eq(6)
            & spreads["时间分段"].eq("全部区间")
            & spreads["候选宽度"].eq("全部宽度")
            & spreads["字段"].eq(field)]
        spread_mfe = finite_num(
            spread_row.iloc[0].get("高减低_W8最大浮盈均值%")) if len(spread_row) else np.nan
        spread_hit20 = finite_num(
            spread_row.iloc[0].get("高减低_先到+20比例%")) if len(spread_row) else np.nan
        spread_hit30 = finite_num(
            spread_row.iloc[0].get("高减低_先到+30比例%")) if len(spread_row) else np.nan
        spread_agrees = (
            math.isfinite(spread_mfe) and spread_mfe * direction > 0
            and (not math.isfinite(spread_hit20) or spread_hit20 * direction > 0))

        redundant_row = redundancy[
            redundancy["SKDJ_N"].eq(6)
            & redundancy["时间分段"].eq("全部区间")
            & redundancy["字段"].eq(field)
            & redundancy["对照"].eq("原冻结100分")]
        old_score_abs_corr = (
            finite_num(redundant_row.iloc[0].get("周内相关绝对值均值"))
            if len(redundant_row) else np.nan)

        pair_row = pair_audit[
            pair_audit["SKDJ_N"].eq(6)
            & pair_audit["时间分段"].eq("全部区间")
            & pair_audit["候选宽度"].eq("全部宽度")
            & pair_audit["字段"].eq(field)]
        pair_accuracy = finite_num(pair_row.iloc[0].get(
            "较高特征值选对更高W8最大浮盈_周均比例%")) if len(pair_row) else np.nan
        if not enough:
            verdict = "样本不足"
        elif not same_period_direction:
            verdict = "前后段方向不稳"
        elif not same_objective_direction:
            verdict = "爆发等级与最大浮盈方向冲突"
        elif not meaningful:
            verdict = "增量偏弱"
        elif not spread_agrees:
            verdict = "IC与高低分位结果冲突"
        else:
            verdict = "进入爆发力下一轮候选"
        rows.append({
            "优先顺序": 0 if verdict == "进入爆发力下一轮候选" else 1,
            "特征家族": spec["家族"], "特征": spec["名称"], "字段": field,
            "综合IC方向": "高值更好" if direction > 0 else "低值更好",
            "全部_爆发等级IC": full_grade, "全部_W8浮盈IC": full_mfe,
            "全部_先到20IC": full_hit20, "全部_先到30IC": full_hit30,
            "前段_爆发综合IC": front_combined,
            "后段_爆发综合IC": back_combined,
            "前段_爆发等级IC": front_grade, "后段_爆发等级IC": back_grade,
            "前段_W8浮盈IC": front_mfe, "后段_W8浮盈IC": back_mfe,
            "前段有效周": front_weeks, "后段有效周": back_weeks,
            "高20减低20_W8最大浮盈百分点": spread_mfe,
            "高20减低20_先到20百分点": spread_hit20,
            "高20减低20_先到30百分点": spread_hit30,
            "与原100分周内相关绝对值均值": old_score_abs_corr,
            "相似股票_高值选对更高最大浮盈比例%": pair_accuracy,
            "爆发力预设结论": verdict,
        })
    return pd.DataFrame(rows).sort_values(
        ["优先顺序", "全部_爆发等级IC"], ascending=[True, False])


def explosion_class_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        ("S", 3, "W8内先到+30%，且在此之前未先到-10%", "首选"),
        ("A", 2, "W8内先到+20%但未达到S级，且未先到-10%", "次选"),
        ("B", 1, "W8内先到+10%但未达到A/S级，且未先到-10%", "仅在没有S/A时补位"),
        ("F", 0, "其余完整W8路径，包括先到-10%或未先到+10%", "失败/不优先"),
    ], columns=["爆发等级", "等级数值", "定义", "实战优先级"])


def explosion_class_audit(eligible: pd.DataFrame,
                          periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            total = len(base)
            for label in ("S", "A", "B", "F"):
                selected = base[base["Explosion_Class_W8"].astype(str).eq(label)]
                signal_weeks = selected["Signal_Week"].nunique() if len(selected) else 0
                rows.append({
                    "SKDJ_N": n, "时间分段": period["name"],
                    "爆发等级": label, "事件数": len(selected),
                    "占全部事件%": len(selected) / total * 100 if total else np.nan,
                    "不同股票": selected["ts_code"].nunique() if len(selected) else 0,
                    "覆盖信号周": signal_weeks,
                    "W8最大浮盈均值%": numeric(
                        selected, "Entry_W8_MFE_Net_pct").mean(),
                    "W8最大浮盈中位%": numeric(
                        selected, "Entry_W8_MFE_Net_pct").median(),
                    "W8期末净收益均值%_辅助": numeric(
                        selected, "Entry_W8_Close_Return_Net_pct").mean(),
                    "W8期末净收益中位%_辅助": numeric(
                        selected, "Entry_W8_Close_Return_Net_pct").median(),
                    "W8最大回撤均值%_风险观察": numeric(
                        selected, "Entry_W8_MAE_Raw_pct").mean(),
                })
    return pd.DataFrame(rows)


def top3_explosion_portfolio_audit(
        eligible: pd.DataFrame, periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Evaluate whether every old/new Top3 finds two A/S or B+ stocks."""
    rows: list[dict[str, Any]] = []
    for spec in all_ranking_scheme_specs():
        for n in SKDJ_NS:
            base_n = eligible[eligible["SKDJ_N"].eq(n)]
            for period in periods:
                base = select_period(base_n, period)
                if base.empty:
                    continue
                selected = base[true_mask(base, spec["top3"])]
                pool_week = base.groupby("Signal_Week").agg(
                    候选数=("ts_code", "size"),
                    候选A或S数=("Explosion_A_or_S_W8", "sum"),
                    候选S数=("Explosion_S_W8", "sum"),
                    候选B及以上数=("Explosion_B_or_Better_W8", "sum"),
                )
                top_week = selected.groupby("Signal_Week").agg(
                    前3实际买入数=("ts_code", "size"),
                    前3A或S数=("Explosion_A_or_S_W8", "sum"),
                    前3S数=("Explosion_S_W8", "sum"),
                    前3B及以上数=("Explosion_B_or_Better_W8", "sum"),
                )
                weekly = pool_week.join(top_week, how="left").fillna(0)
                classes = selected.get(
                    "Explosion_Class_W8",
                    pd.Series(index=selected.index, dtype=str)).astype(str)
                pool_as = int(true_mask(base, "Explosion_A_or_S_W8").sum())
                pool_s = int(true_mask(base, "Explosion_S_W8").sum())
                selected_as = int(true_mask(selected, "Explosion_A_or_S_W8").sum())
                selected_s = int(true_mask(selected, "Explosion_S_W8").sum())
                available_two_as = weekly["候选A或S数"].ge(2)
                selected_two_as = weekly["前3A或S数"].ge(2)
                available_two_b = weekly["候选B及以上数"].ge(2)
                selected_two_b = weekly["前3B及以上数"].ge(2)
                at_least_two_positions = weekly["前3实际买入数"].ge(2)
                rows.append({
                    "排序方案": spec["name"], "SKDJ_N": n,
                    "时间分段": period["name"],
                    "候选池信号周": len(weekly),
                    "可买满三仓周": int(weekly["候选数"].ge(3).sum()),
                    "前3平均实际股票数": weekly["前3实际买入数"].mean(),
                    "前3事件数": len(selected),
                    "前3_S级数": int(classes.eq("S").sum()),
                    "前3_A级数": int(classes.eq("A").sum()),
                    "前3_B级数": int(classes.eq("B").sum()),
                    "前3_F级数": int(classes.eq("F").sum()),
                    "前3_S级比例%": classes.eq("S").mean() * 100 if len(classes) else np.nan,
                    "前3_A或S比例%": classes.isin(["A", "S"]).mean() * 100 if len(classes) else np.nan,
                    "前3_B及以上比例%": classes.isin(["B", "A", "S"]).mean() * 100 if len(classes) else np.nan,
                    "前3_W8最大浮盈均值%": numeric(
                        selected, "Entry_W8_MFE_Net_pct").mean(),
                    "候选池至少2只A或S周": int(available_two_as.sum()),
                    "前3至少2只A或S周": int(selected_two_as.sum()),
                    "前3至少2只A或S占全部信号周%": (
                        selected_two_as.mean() * 100 if len(weekly) else np.nan),
                    "前3至少2只A或S占至少买2只周%": (
                        selected_two_as[at_least_two_positions].mean() * 100
                        if at_least_two_positions.any() else np.nan),
                    "候选池本可选至少2只A或S时_前3命中率%": (
                        selected_two_as[available_two_as].mean() * 100
                        if available_two_as.any() else np.nan),
                    "候选池至少2只B及以上周": int(available_two_b.sum()),
                    "前3至少2只B及以上周": int(selected_two_b.sum()),
                    "前3至少2只B及以上占全部信号周%": (
                        selected_two_b.mean() * 100 if len(weekly) else np.nan),
                    "候选池本可选至少2只B及以上时_前3命中率%": (
                        selected_two_b[available_two_b].mean() * 100
                        if available_two_b.any() else np.nan),
                    "A或S事件捕获率%": (
                        selected_as / pool_as * 100 if pool_as else np.nan),
                    "S事件捕获率%": selected_s / pool_s * 100 if pool_s else np.nan,
                })
    return pd.DataFrame(rows)


def v57_feature_tier_audit(
        eligible: pd.DataFrame, periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Show whether P1/P2/P3 tiers actually separate explosion outcomes."""
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            groups = [("全部候选", base)] + [
                (f"{tier}_{label}", base[base["V57_Factor_Tier"].astype(str).eq(tier)])
                for tier, label in (
                    ("P1", "双优"), ("P2", "单优"),
                    ("P3", "无优"), ("PX", "数据不足"))]
            for tier_name, selected in groups:
                classes = selected.get(
                    "Explosion_Class_W8",
                    pd.Series(index=selected.index, dtype=str)).astype(str)
                rows.append({
                    "SKDJ_N": n, "时间分段": period["name"],
                    "特征层": tier_name, "事件数": len(selected),
                    "覆盖信号周": selected["Signal_Week"].nunique()
                    if len(selected) else 0,
                    "S级比例%": classes.eq("S").mean() * 100
                    if len(classes) else np.nan,
                    "A或S比例%": classes.isin(["A", "S"]).mean() * 100
                    if len(classes) else np.nan,
                    "B及以上比例%": classes.isin(["B", "A", "S"]).mean() * 100
                    if len(classes) else np.nan,
                    "W8最大浮盈均值%": numeric(
                        selected, "Entry_W8_MFE_Net_pct").mean(),
                    "W8最大浮盈中位%": numeric(
                        selected, "Entry_W8_MFE_Net_pct").median(),
                    "W8期末净收益均值%_辅助": numeric(
                        selected, "Entry_W8_Close_Return_Net_pct").mean(),
                    "W8最大回撤均值%_风险": numeric(
                        selected, "Entry_W8_MAE_Raw_pct").mean(),
                })
    return pd.DataFrame(rows)


def v58_history_completeness_audit(
        eligible: pd.DataFrame, periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Separate insufficient history from three-cycle completed history."""
    rows: list[dict[str, Any]] = []
    ordered_states = ("完整3次", "部分2次", "部分1次", "无有效周期")
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            for state in ordered_states:
                selected = base[base.get(
                    "H2_History_State",
                    pd.Series(index=base.index, dtype=str)).astype(str).eq(state)]
                classes = selected.get(
                    "Explosion_Class_W8",
                    pd.Series(index=selected.index, dtype=str)).astype(str)
                rows.append({
                    "SKDJ_N": n, "时间分段": period["name"],
                    "历史完整度": state, "事件数": len(selected),
                    "占本段候选%": len(selected) / len(base) * 100
                    if len(base) else np.nan,
                    "不同股票": selected["ts_code"].nunique()
                    if len(selected) else 0,
                    "覆盖信号周": selected["Signal_Week"].nunique()
                    if len(selected) else 0,
                    "达到75次数均值": numeric(
                        selected, "Signal_Prior_GC_Reached75_Count_Last3").mean(),
                    "旧V5.4优先层比例%": selected.get(
                        "H2_Tier_Level",
                        pd.Series(index=selected.index, dtype=str)
                    ).astype(str).eq("S").mean() * 100 if len(selected) else np.nan,
                    "S级比例%": classes.eq("S").mean() * 100
                    if len(classes) else np.nan,
                    "A或S比例%": classes.isin(["A", "S"]).mean() * 100
                    if len(classes) else np.nan,
                    "B及以上比例%": classes.isin(["B", "A", "S"]).mean() * 100
                    if len(classes) else np.nan,
                    "W8最大浮盈均值%": numeric(
                        selected, "Entry_W8_MFE_Net_pct").mean(),
                    "W8最大浮盈中位%": numeric(
                        selected, "Entry_W8_MFE_Net_pct").median(),
                    "W8期末净收益均值%_辅助": numeric(
                        selected, "Entry_W8_Close_Return_Net_pct").mean(),
                    "W8最大回撤均值%_风险": numeric(
                        selected, "Entry_W8_MAE_Raw_pct").mean(),
                })
    return pd.DataFrame(rows)


def v58_history_rank_acceptance_audit(
        top3_audit: pd.DataFrame) -> pd.DataFrame:
    """Compare each completeness correction only with its frozen parent."""
    comparisons = (
        ("V5.8完整历史H2优先", "V5.4历史≥2次优先"),
        ("V5.8双因子层内完整H2", "V5.7-双因子层内H2优先"),
    )
    periods = ("全部区间", "前段观察", "后段冻结检验")
    metrics = (
        ("前3_S级比例%", "S级"),
        ("前3_A或S比例%", "A或S"),
        ("前3至少2只A或S占全部信号周%", "至少2只A或S周"),
        ("前3_W8最大浮盈均值%", "最大浮盈"),
    )

    def one(scheme: str, period: str) -> pd.Series | None:
        selected = top3_audit[
            top3_audit["排序方案"].eq(scheme)
            & top3_audit["SKDJ_N"].eq(6)
            & top3_audit["时间分段"].eq(period)]
        return selected.iloc[0] if len(selected) else None

    rows: list[dict[str, Any]] = []
    for scheme, baseline_name in comparisons:
        row: dict[str, Any] = {"排序方案": scheme, "冻结基准": baseline_name}
        improvements: dict[str, int] = {}
        for period in periods:
            current, baseline = one(scheme, period), one(baseline_name, period)
            key = {"全部区间": "全部", "前段观察": "前段",
                   "后段冻结检验": "后段"}[period]
            deltas: list[float] = []
            for metric, short in metrics:
                delta = (
                    finite_num(current.get(metric)) - finite_num(baseline.get(metric))
                    if current is not None and baseline is not None else np.nan)
                row[f"{key}_{short}差"] = delta
                if math.isfinite(delta):
                    deltas.append(delta)
            improvements[key] = sum(value > 0 for value in deltas)
            row[f"{key}_四项改善数"] = improvements[key]
            row[f"{key}_B及以上差"] = (
                finite_num(current.get("前3_B及以上比例%"))
                - finite_num(baseline.get("前3_B及以上比例%"))
                if current is not None and baseline is not None else np.nan)
        if (improvements.get("全部", 0) >= 3
                and improvements.get("前段", 0) >= 2
                and improvements.get("后段", 0) >= 2):
            verdict = "支持：历史完整度修正前后段均改善"
        elif improvements.get("全部", 0) >= 3:
            verdict = "观察：总体改善但分段不一致"
        else:
            verdict = "不支持：完整度修正未改善主要目标"
        row["历史完整度修正结论"] = verdict
        rows.append(row)
    return pd.DataFrame(rows)


def v57_rank_acceptance_audit(top3_audit: pd.DataFrame) -> pd.DataFrame:
    """Apply predeclared acceptance rules versus V5.4, focused on N=6."""
    rows: list[dict[str, Any]] = []
    baseline_name = "V5.4历史≥2次优先"
    periods = ("全部区间", "前段观察", "后段冻结检验")
    primary = (
        "前3_S级比例%", "前3_A或S比例%",
        "前3至少2只A或S占全部信号周%", "前3_W8最大浮盈均值%")

    def selected_row(scheme: str, period: str) -> pd.Series | None:
        selected = top3_audit[
            top3_audit["排序方案"].eq(scheme)
            & top3_audit["SKDJ_N"].eq(6)
            & top3_audit["时间分段"].eq(period)]
        return selected.iloc[0] if len(selected) else None

    for spec in v57_ranking_scheme_specs():
        scheme = spec["name"]
        row: dict[str, Any] = {"排序方案": scheme, "基准": baseline_name}
        improvement_counts: dict[str, int] = {}
        for period in periods:
            current = selected_row(scheme, period)
            baseline = selected_row(baseline_name, period)
            period_key = {"全部区间": "全部", "前段观察": "前段", "后段冻结检验": "后段"}[period]
            deltas: list[float] = []
            for metric in primary:
                delta = (
                    finite_num(current.get(metric)) - finite_num(baseline.get(metric))
                    if current is not None and baseline is not None else np.nan)
                short_name = {
                    "前3_S级比例%": "S级",
                    "前3_A或S比例%": "A或S",
                    "前3至少2只A或S占全部信号周%": "至少2只A或S周",
                    "前3_W8最大浮盈均值%": "最大浮盈",
                }[metric]
                row[f"{period_key}_{short_name}差"] = delta
                if math.isfinite(delta):
                    deltas.append(delta)
            improvement_counts[period_key] = sum(value > 0 for value in deltas)
            row[f"{period_key}_四项改善数"] = improvement_counts[period_key]
            for metric, short_name in (
                    ("前3_B及以上比例%", "B及以上"),
                    ("候选池本可选至少2只A或S时_前3命中率%", "有货时命中")):
                row[f"{period_key}_{short_name}差"] = (
                    finite_num(current.get(metric)) - finite_num(baseline.get(metric))
                    if current is not None and baseline is not None else np.nan)

        full_pass = (
            improvement_counts.get("全部", 0) >= 3
            and finite_num(row.get("全部_至少2只A或S周差")) >= 3.0
            and finite_num(row.get("全部_B及以上差")) >= -3.0)
        segment_pass = (
            improvement_counts.get("前段", 0) >= 2
            and improvement_counts.get("后段", 0) >= 2
            and min(finite_num(row.get("前段_A或S差")),
                    finite_num(row.get("后段_A或S差"))) >= -5.0
            and min(finite_num(row.get("前段_至少2只A或S周差")),
                    finite_num(row.get("后段_至少2只A或S周差"))) >= -5.0)
        if full_pass and segment_pass:
            verdict = "通过：进入下一版冻结候选"
        elif improvement_counts.get("全部", 0) >= 3:
            verdict = "观察：总体改善但分段不稳"
        else:
            verdict = "淘汰：总体爆发力改善不足"
        row["预设验收结论"] = verdict
        rows.append(row)
    return pd.DataFrame(rows)


def slim_discovery_candidates(eligible: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "ts_code", "name", "SKDJ_N", "Signal_Date", "Signal_Week",
        "SW_L1", "SW_L2", "Raw_Close", "Circ_MV_Billion",
        "Signal_K", "Signal_D", "Signal_Prior_Below25_Streak",
        "Score_Total_100", "Weekly_Rank", "Candidates_This_Week",
        "Top3", "Top20Pct", "V52_Tier_Level", "V52_Tier_Weekly_Rank",
        "V52_Tier_Top3", "V52_Tier_Top20Pct",
        "H2_Tier_Level", "H2_Weekly_Rank", "H2_Top3", "H2_Top20Pct",
        "H2_History_Valid_Count", "H2_History_State", "H2_History_Complete",
        "H2C_Tier_Level", "H2C_Weekly_Rank", "H2C_Top3", "H2C_Top20Pct",
        "V57_K_Weekly_Rank", "V57_K_Top3", "V57_K_Top20Pct",
        "V57_Breadth_Weekly_Rank", "V57_Breadth_Top3", "V57_Breadth_Top20Pct",
        "V57_Dual_Weekly_Rank", "V57_Dual_Top3", "V57_Dual_Top20Pct",
        "V57_DualH2_Weekly_Rank", "V57_DualH2_Top3", "V57_DualH2_Top20Pct",
        "V58_DualH2C_Weekly_Rank", "V58_DualH2C_Top3",
        "V58_DualH2C_Top20Pct",
        "Validation_Period",
    ]
    feature_columns = [
        "Signal_K_Thrust_per_AbsWeekReturn", "Breadth_MA20_Rising_Pct",
        "V57_K_Efficiency_Weekly_Pct", "V57_Industry_MA20_Weekly_Pct",
        "V57_K_Favorable20", "V57_Breadth_Favorable20",
        "V57_Favorable_Count", "V57_Feature_Data_Valid",
        "V57_Factor_Tier", "V57_Factor_Tier_Order",
        "V57_TwoFactor_Rank_Sum",
        "Signal_Prior_GC_Valid_Count_Last3",
        "Signal_Prior_GC_Reached75_Count_Last3", "Signal_Prior_GC1_Peak_K",
    ]
    outcomes = [
        "Entry_Date", "Entry_W4_MFE_Net_pct", "Entry_W4_MAE_Raw_pct",
        "Entry_W4_Close_Return_Net_pct", "Entry_W8_MFE_Net_pct",
        "Entry_W8_MAE_Raw_pct", "Entry_W8_Close_Return_Net_pct",
        "Entry_First_Hit_10_vs_Minus10_W8",
        "Entry_First_Hit_20_vs_Minus10_W8",
        "Entry_First_Hit_30_vs_Minus10_W8",
        "Explosion_Class_W8", "Explosion_Grade_W8",
        "Explosion_B_or_Better_W8", "Explosion_A_or_S_W8", "Explosion_S_W8",
        "Explosion_Class_W12", "Explosion_Grade_W12",
    ]
    columns = [column for column in identity + feature_columns + outcomes
               if column in eligible.columns]
    return eligible[columns].copy()


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    streamlit_version = str(getattr(st, "__version__", "unknown"))
    st.caption(
        f"{UI_PATCH}｜冻结V5.7七套排名；新增两套历史完整度对照。"
        f"｜Streamlit {streamlit_version}")
    if streamlit_version.startswith("1.62"):
        st.error(
            "检测到Streamlit 1.62.x。该环境与本次约32秒断线重连日志一致；"
            "请同时使用本版requirements.txt锁定到1.61.0后再运行。")
    with st.expander("本版口径", expanded=True):
        st.markdown(f"""
- **信号**：上一完整周K＜25，本完整周K≥25；不要求低位金叉，不要求K>D。
- **参数**：同一次运行分别计算N=6、N=7和默认N=9，M固定为3；三者使用完全相同的科技池、价格市值和交易成本。
- **重复信号**：同一股票以后再次跌回25下方并重新上穿25，会再次计为新事件。
- **买入**：信号完整周结束后的下一市场交易日开盘。
- **数据长度**：默认正式窗口500个交易日；开始前预热扩展为{WARMUP_WEEKS}周，截止后观察W1-W12。
- **预热用途**：200周只用于形成指标和读取已完成历史金叉，不增加正式信号；因此候选买点窗口仍是500个交易日。
- **过滤**：每个历史信号日分别检查当时科技行业归属；最低股价默认10元、最低流通市值默认50亿元，侧边栏可切换，避免使用今天状态回看历史。
- **共同硬条件**：信号前连续处于25下方1～{MAX_BOTTOM_STREAK}周；超过5周不进入九套排名，但仍保留剔除计数。
- **原评分基准**：SKDJ重置35分、量能20分、周K线结构20分、MA20趋势15分、同周价格/市值相对排名10分。
- **V5.2/V5.4对照**：三套旧排名完整保留，新增特征不改变任何股票名次。
- **V5.4历史二层**：最近3次已完成金叉至少2次达到75线进入优先层；2次和3次不再区分，其余进入普通层。
- **V5.4同层顺序**：先按原100分降序；最近一次已完成金叉峰值仅在原分相同时破同分。
- **V5.6.1确认的特征一**：K动能相对股价扩张效率越低越好，表示K上穿25伴随更真实的股价响应。
- **V5.6.1确认的特征二**：所属行业MA20上升比例越高越好，表示个股处在更健康的行业中期趋势中。
- **单因子排名**：分别测试K效率低优先、行业MA20广度高优先；原100分只破同值。
- **双因子分层**：同周两个特征都进入优秀20%为P1、一个进入为P2、均未进入为P3、数据不足为PX。
- **双因子对照**：一套按P1/P2/P3直接排序；另一套在特征层内让V5.4历史H2优先。
- **历史完整度**：分别标记有效历史周期0/1/2/3次；只有3次完整历史才能判定“至少2次达到75”。
- **V5.8完整H2**：完整3次且至少2次达到75为S层；完整但不足2次为C层；历史不足为PX层，事件不删除。
- **V5.8双因子完整H2**：保持P1/P2/P3/PX顺序，在同一特征层内使用完整历史H2，而不是把历史不足当失败。
- **爆发等级**：S级=先到+30%，A级=先到+20%但未到S，B级=先到+10%但未到A/S，其余为F；全部与-10%比较且同日冲突按-10%先。
- **主目标**：爆发等级、W8最大浮盈、+20先于-10、+30先于-10；S优先、A其次、B只在无S/A时补位。
- **辅助目标**：W8期末收益、+10和较小回撤仅供风险解释，不参与特征入选判定，也不要求W1～W12平滑一致。
- **三仓审计**：同时统计前三名至少两只A/S和至少两只B级以上；A/S是主目标，B只作没有S/A时的补位。
- **预设验收**：四套V5.7排名继续与V5.4比较；两套V5.8完整度排名只与各自冻结父方案比较。
- **防泄漏**：全部特征只使用信号周收盘时已经知道的数据；买入后结果只用于判卷。
- **时间分段**：评分规则在代码中预先冻结；前段和后段只用于分别报告，不读取后段收益重新调权。
- **本版边界**：不加入新指标、不调权、不修改候选硬条件、不重新寻找31项特征、不研究止损止盈。
""")
    with st.sidebar:
        st.header("运行参数")
        backtest_days = st.number_input(
            "回测交易日数", 100, 1000, 500, 50, key="v58_days")
        st.caption("默认500日；首次运行需读取额外200周历史，数据量会大于V5.7。")
        # 沿用本页已经稳定加载的number_input，避免部分iOS/Safari会话在
        # 首次加载selectbox前端分块时出现“Importing a module script failed”。
        min_price = st.number_input(
            "最低股价（元）", min_value=10.0, max_value=20.0,
            value=10.0, step=10.0, format="%.0f", key="v58_min_price")
        min_mv = st.number_input(
            "最低流通市值（亿元）", min_value=50.0, max_value=100.0,
            value=50.0, step=50.0, format="%.0f", key="v58_min_mv")
        st.caption(f"本次历史过滤：股价≥{min_price:.0f}元，流通市值≥{min_mv:.0f}亿元")
        signal_end_date = st.date_input(
            "买入信号截止", date(2026, 6, 5), key="v58_signal_end")
        market_end_date = st.date_input(
            "行情观察截止", date.today(), key="v58_market_end")
        split_ratio_pct = st.number_input(
            "前段观察占正式周比例(%)", 50, 80, 60, 5, key="v58_split")
        pause = st.number_input(
            "接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v58_pause")
        use_cache = st.checkbox("复用行情缓存", True, key="v58_cache")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005,
            format="%.3f", key="v58_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01,
            format="%.3f", key="v58_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001,
            format="%.3f", key="v58_transfer")
        if st.button("清除V5.8结果和运行状态", key="v58_clear"):
            shutil.rmtree(RESULT_DIR, ignore_errors=True)
            shutil.rmtree(JOB_DIR, ignore_errors=True)
            st.success("V5.8结果和检查点已清除；通用行情缓存保留")

    request_payload = {
        "version": VERSION, "days": int(backtest_days),
        "signal_end": signal_end_date.strftime("%Y%m%d"),
        "market_end": market_end_date.strftime("%Y%m%d"),
        "split_ratio_pct": int(split_ratio_pct),
        "min_price": float(min_price), "min_mv": float(min_mv),
        "commission": float(commission_pct), "stamp": float(stamp_duty_pct),
        "transfer": float(transfer_fee_pct),
    }
    request_signature = stable_signature(request_payload)
    result_path = os.path.join(RESULT_DIR, f"{request_signature}.zip")
    result_name = (
        f"weekly_skdj_history_completeness_audit_v5_8_{int(backtest_days)}d_"
        f"p{int(min_price)}_mv{int(min_mv)}.zip")
    completed_available = False
    if os.path.exists(result_path):
        try:
            with open(result_path, "rb") as handle:
                saved_result = handle.read()
            completed_available = True
            clear_job_active(request_signature)
            st.success("发现相同参数的已完成结果，可直接下载。")
            render_download(
                saved_result, result_name, f"v58_saved_{request_signature}")
        except Exception as exc:
            st.warning(f"旧结果读取失败：{exc}")

    secret_token = configured_tushare_token()
    if secret_token:
        token = secret_token
        st.caption("已从Streamlit Secrets读取TUSHARE_TOKEN。")
    else:
        token = st.text_input("Tushare Token", type="password", key="v58_token")

    job_active = is_job_active(request_signature)
    left, right = st.columns(2)
    with left:
        start_clicked = st.button("开始/重新运行V5.8", type="primary", key="v58_run")
    with right:
        stop_clicked = st.button("停止自动续跑", disabled=not job_active, key="v58_stop")
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
        st.error(f"确定{int(backtest_days)}日窗口失败：{exc}")
        return
    data_start_date = signal_start_date - timedelta(weeks=WARMUP_WEEKS, days=7)
    data_start = data_start_date.strftime("%Y%m%d")
    config = {
        "signal_start": signal_start, "signal_end": signal_end,
        "data_start": data_start, "market_end": market_end,
        "min_price": float(min_price), "min_mv": float(min_mv),
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct),
        "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct),
    }
    # 200周预热会改变历史周期完整度，必须使用V5.8独立检查点。
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
    breadth_totals: dict[str, dict[str, float]] = {}
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
            merge_breadth(breadth_totals, checkpoint["breadth"])
            checkpoint_hits += 1
        else:
            daily, cached_basic, storage_path, cache_hit = fetch_price(
                code, data_start, market_end, bool(use_cache), float(pause))
            price_cache_hits += int(cache_hit)
            if daily.empty:
                failures += 1
            else:
                try:
                    rows, stock_rejects, stock_breadth = analyze_stock(
                        stock, period_index.get(code, []), daily, cached_basic,
                        storage_path, week_last_map, open_dates, open_pos,
                        market_weeks, config, bool(use_cache), float(pause))
                    event_rows.extend(rows)
                    merge_counts(rejects, stock_rejects)
                    merge_breadth(breadth_totals, stock_breadth)
                    save_checkpoint(
                        run_signature, code, rows, stock_rejects, stock_breadth)
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
    breadth_frame = build_industry_breadth_frame(breadth_totals)
    mature_w8 = mature_events(events_all, RANKING_WEEKS)
    mature_w12 = mature_events(events_all, AUDIT_WEEKS)
    if mature_w8.empty:
        st.error("存在信号，但没有未来完整W8的成熟事件。")
        return

    calendar = signal_calendar(open_dates, signal_start, signal_end, events_all)
    split_end, periods = build_periods(calendar, float(split_ratio_pct) / 100.0)
    scored_all, eligible = score_and_rank_events(mature_w8, split_end)
    if eligible.empty:
        st.error("存在成熟事件，但没有股票通过‘连续处于25下方不超过5周’硬条件。")
        return
    eligible = add_independent_candidate_features(eligible)
    eligible = add_industry_breadth_features(eligible, breadth_frame)
    eligible = add_challenger_rankings(eligible)
    eligible = add_v57_explosion_rankings(eligible)
    eligible = add_explosion_labels(eligible)
    feature_status = st.empty()
    with st.spinner("冻结V5.7七套排名，并执行历史完整度审计..."):
        feature_status.caption("1/8 复现共同候选池与V5.7七套排名...")
        score_rules = frozen_score_definitions()
        challenger_rules = challenger_score_definitions()
        v57_rules = v57_ranking_definitions()
        v58_rules = v58_history_definitions()
        explosion_definitions = explosion_class_definitions()
        feature_status.caption("2/8 生成两套历史完整度修正排名...")
        rank_cohorts = scheme_rank_cohort_audit(eligible, periods)
        feature_status.caption("3/8 汇总历史有效周期0/1/2/3次...")
        history_completeness = v58_history_completeness_audit(eligible, periods)
        feature_status.caption("4/8 汇总P1/P2/P3/PX特征层...")
        feature_tiers = v57_feature_tier_audit(eligible, periods)
        feature_status.caption("5/8 检验九套排名的三仓表现...")
        top3_explosion = top3_explosion_portfolio_audit(eligible, periods)
        feature_status.caption("6/8 复核V5.7冻结验收...")
        acceptance = v57_rank_acceptance_audit(top3_explosion)
        feature_status.caption("7/8 比较完整度修正与冻结父方案...")
        history_acceptance = v58_history_rank_acceptance_audit(top3_explosion)
        explosion_classes = explosion_class_audit(eligible, periods)
        feature_status.caption("8/8 生成周历和精简候选明细...")
        ranked_calendar = weekly_rank_calendar(
            calendar, scored_all, eligible, split_end)
        slim_candidates = slim_discovery_candidates(eligible)
    feature_status.empty()
    summary_rows = []
    for n in SKDJ_NS:
        all_n = events_all[events_all["SKDJ_N"].eq(n)]
        mature_n = mature_w8[mature_w8["SKDJ_N"].eq(n)]
        mature_w12_n = mature_w12[mature_w12["SKDJ_N"].eq(n)]
        eligible_n = eligible[eligible["SKDJ_N"].eq(n)]
        pass_by_date = eligible_n.groupby(eligible_n["Signal_Date"].astype(str)).size()
        pass_counts = calendar["Week_End"].astype(str).map(pass_by_date).fillna(0).astype(int)
        summary_rows.append({
            "SKDJ_N": n, "SKDJ_M": SKDJ_M,
            "全部通过过滤事件": len(all_n), "W8成熟事件": len(mature_n),
            "W12成熟事件": len(mature_w12_n),
            "硬条件通过事件": len(eligible_n),
            "硬条件剔除事件": len(mature_n) - len(eligible_n),
            "硬条件通过不同股票": eligible_n["ts_code"].nunique(),
            "硬条件通过有信号周": int(pass_counts.gt(0).sum()),
            "硬条件后空窗周": int(pass_counts.eq(0).sum()),
            "硬条件后最长连续空窗周": max_empty_run(pass_counts),
            "硬条件后每周信号均值": pass_counts.mean(),
            "硬条件后单周最多": pass_counts.max(),
            "硬条件后1至5只候选周": int(pass_counts.between(1, 5).sum()),
            "硬条件后6至15只候选周": int(pass_counts.between(6, 15).sum()),
            "硬条件后16至25只候选周": int(pass_counts.between(16, 25).sum()),
            "硬条件后超过25只候选周": int(pass_counts.gt(25).sum()),
            "新增50至100亿事件": int(numeric(
                eligible_n, "Circ_MV_Billion").between(50, 100, inclusive="left").sum()),
            "原100亿以上事件": int(numeric(
                eligible_n, "Circ_MV_Billion").ge(100).sum()),
            "原评分前3事件": int(true_mask(eligible_n, "Top3").sum()),
            "V5.2共振分层前3事件": int(true_mask(
                eligible_n, "V52_Tier_Top3").sum()),
            "V5.4历史二层前3事件": int(true_mask(
                eligible_n, "H2_Top3").sum()),
            "V5.4历史优先层事件": int(
                eligible_n["H2_Tier_Level"].astype(str).eq("S").sum()),
            "历史完整3次事件": int(numeric(
                eligible_n, "H2_History_Valid_Count").ge(3).sum()),
            "历史部分2次事件": int(numeric(
                eligible_n, "H2_History_Valid_Count").eq(2).sum()),
            "历史部分1次事件": int(numeric(
                eligible_n, "H2_History_Valid_Count").eq(1).sum()),
            "历史无有效周期事件": int(numeric(
                eligible_n, "H2_History_Valid_Count").eq(0).sum()),
            "V5.8完整H2优先层事件": int(
                eligible_n["H2C_Tier_Level"].astype(str).eq("S").sum()),
            "V5.8完整H2前3事件": int(true_mask(
                eligible_n, "H2C_Top3").sum()),
            "V5.7双优P1事件": int(
                eligible_n["V57_Factor_Tier"].astype(str).eq("P1").sum()),
            "V5.7单优P2事件": int(
                eligible_n["V57_Factor_Tier"].astype(str).eq("P2").sum()),
            "V5.7无优P3事件": int(
                eligible_n["V57_Factor_Tier"].astype(str).eq("P3").sum()),
            "V5.7数据不足PX事件": int(
                eligible_n["V57_Factor_Tier"].astype(str).eq("PX").sum()),
            "V5.7_K效率前3事件": int(true_mask(
                eligible_n, "V57_K_Top3").sum()),
            "V5.7_行业广度前3事件": int(true_mask(
                eligible_n, "V57_Breadth_Top3").sum()),
            "V5.7_双因子前3事件": int(true_mask(
                eligible_n, "V57_Dual_Top3").sum()),
            "V5.7_双因子H2前3事件": int(true_mask(
                eligible_n, "V57_DualH2_Top3").sum()),
            "V5.8_双因子完整H2前3事件": int(true_mask(
                eligible_n, "V58_DualH2C_Top3").sum()),
            "爆发S级事件": int(eligible_n[
                "Explosion_Class_W8"].astype(str).eq("S").sum()),
            "爆发A级事件": int(eligible_n[
                "Explosion_Class_W8"].astype(str).eq("A").sum()),
            "爆发B级事件": int(eligible_n[
                "Explosion_Class_W8"].astype(str).eq("B").sum()),
            "爆发F级事件": int(eligible_n[
                "Explosion_Class_W8"].astype(str).eq("F").sum()),
            "有至少1次成熟历史同类信号": int(numeric(
                eligible_n, "Hist_SameSignal_Valid_Count_Last3").ge(1).sum()),
            "信号周日线路径有效": int(numeric(
                eligible_n, "SignalWeek_Trading_Days").ge(2).sum()),
            "全行业广度匹配有效": int(numeric(
                eligible_n, "Breadth_Constituent_Count").ge(2).sum()),
        })
    run_summary = pd.DataFrame(summary_rows)
    run_summary.insert(0, "程序版本", VERSION)
    run_summary["正式信号开始"] = signal_start
    run_summary["正式信号截止"] = signal_end
    run_summary["实际行情开始"] = data_start
    run_summary["行情观察截止"] = market_end
    run_summary["前段观察截止"] = split_end
    run_summary["前段观察比例%"] = int(split_ratio_pct)
    run_summary["最低股价元"] = float(min_price)
    run_summary["最低流通市值亿元"] = float(min_mv)
    run_summary["处理股票数"] = len(stocks)
    run_summary["检查点恢复"] = checkpoint_hits
    run_summary["行情缓存命中"] = price_cache_hits
    run_summary["失败股票"] = failures

    hard_rejections = []
    for n in SKDJ_NS:
        group = scored_all[scored_all["SKDJ_N"].eq(n)]
        rejected = group[~true_mask(group, "Hard_Pass")]
        for reason, count in rejected["Hard_Reject_Reason"].fillna("未知").value_counts().items():
            hard_rejections.append({
                "层级": "V5.8冻结共同硬条件", "SKDJ_N": n,
                "剔除原因": reason, "次数": int(count)})
    historical_rejections = [{
        "层级": "历史时点基础过滤", "SKDJ_N": "全部",
        "剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]
    rejection_audit = pd.DataFrame(historical_rejections + hard_rejections)

    metadata = pd.DataFrame([
        ("信号", "上一完整周K<25，本完整周K>=25；不要求低位金叉，不要求K>D"),
        ("重复信号", "以后跌回25下方再上穿时重新计为新事件"),
        ("参数", "同一次运行分别计算N=6、N=7和默认N=9；M固定为3"),
        ("数据窗口", f"正式{int(backtest_days)}个交易日；开始前{WARMUP_WEEKS}周预热；截止后观察W1-W12"),
        ("成熟样本", "旧排名和爆发力主审计使用完整W8样本；W12只作补充生命周期观察，不要求与W8表现一致"),
        ("历史价格市值过滤", f"信号日股价≥{float(min_price):g}元、流通市值≥{float(min_mv):g}亿元"),
        ("九方案共同硬条件", f"信号前连续处于25下方1～{MAX_BOTTOM_STREAK}周；超过5周不进入排名"),
        ("事件检查点", "200周预热改变历史周期完整度，V5.8使用独立逐股票检查点；覆盖足够长区间的通用行情缓存仍可复用"),
        ("原评分基准", "V5.1.1冻结100分原样保留；新单因子排序仅用它破同值，新双因子排序仅在更高优先规则都相同时使用"),
        ("原评分结构", "SKDJ重置35、量能20、周K线结构20、MA20趋势15、同周价格市值百分位10"),
        ("V5.2对照", "保留上一版共振S/A/B/C定义和同层原评分排序；已删除V5.2强权重线性100分"),
        ("V5.4历史二层", "最近3次已完成金叉至少2次达到75线进入优先层；2次和3次同级，其余进入普通层；板块共振不参与等级"),
        ("V5.4同层规则", "同一层内先按V5.1.1冻结100分降序；最近一次金叉峰值只用于原分相同时破同分"),
        ("历史有效周期", "只统计当前信号以前已经出现死叉、完整结束的K/D金叉周期；分别标记0/1/2/3次"),
        ("V5.8完整历史H2", "只有有效周期=3且其中至少2次达到75才进入S层；完整但不足2次为C层；历史不足为PX层"),
        ("V5.8双因子完整H2", "保持V5.7的P1/P2/P3/PX顺序；同一特征层内依次按完整历史H2、两因子百分位和、原100分排序"),
        ("V5.7确认特征一", "K动能相对股价扩张效率越低越好；只在同参数同信号周内部比较，优秀20%为有利状态"),
        ("V5.7确认特征二", "所属行业MA20上升比例越高越好；只在同参数同信号周内部比较，优秀20%为有利状态"),
        ("V5.7单因子排名", "分别按K效率由低到高、行业MA20广度由高到低；冻结100分只破同值，不做温和加分"),
        ("V5.7双因子层", "P1=两项均进入同周优秀20%，P2=一项，P3=均未进入，PX=任一数据缺失；优先级P1>P2>P3>PX"),
        ("V5.7双因子排序", "方案一层内按两因子周内百分位和排序；方案二层内先让V5.4历史H2优先，再按百分位和；最后才用原100分"),
        ("当前K值", "突破当周K值不计分"),
        ("时间分段", f"前段观察截止{split_end}；规则预先写死，后段不重新调权；不要求前后段绝对收益相同"),
        ("全行业广度", "按历史申万一级归属累计全部可识别科技成分股；行业MA20上升比例只用信号当时可知数据"),
        ("策略目标", "科技股高爆发优先；允许约三分之一失败，目标是三仓中尽量有两只达到A/S，而不是挑低波动小涨股票"),
        ("爆发等级", "S=W8先到+30%；A=先到+20%但未到S；B=先到+10%但未到A/S；F=其余完整路径；均与-10%比较"),
        ("主评价目标", "S级比例、A/S比例、前三至少两只A/S周比例、W8最大浮盈；四项共同判卷，不用期末小幅稳定盈利替代爆发力"),
        ("辅助评价目标", "W8期末收益、+10、较小不利波动和W12结果只作解释，不参与入选判定；不要求W1至W12平滑或前后段收益相同"),
        ("候选宽度", "全部宽度、1至5、6至15、16至25、超过25只分别报告；不改变候选和排名"),
        ("V5.7预设验收", "四套V5.7排名继续统一与V5.4比较，不因延长历史后出现的新结果改变门槛"),
        ("V5.8完整度验收", "完整H2与V5.4比较；双因子完整H2与V5.7双因子H2比较；总体至少改善3项且前后段各至少改善2项才支持"),
        ("三仓验收", "同时报告前三名个股A/S比例、每周至少两只A/S比例，以及候选池本来存在至少两只A/S时的命中率；不以单一平均胜率代替组合检验"),
        ("防未来数据", "买入后结果只用于判卷；全部特征在信号周收盘时已经可知"),
        ("历史过滤", "每个信号日使用当时科技行业归属、股价和流通市值"),
        ("买入", "信号完整周结束后的下一市场交易日开盘"),
        ("成本", "买卖0.2%滑点、佣金、过户费；卖出另计印花税"),
        ("先后顺序", "同一天同时触及止盈和-10%时保守计为-10%先；W8和W12分别独立计算"),
        ("本版边界", "不加入基本面或新指标、不修改硬条件、不重新搜索31项特征、不调旧排名权重、不研究止损止盈"),
        ("运行环境", f"Streamlit {streamlit_version}；运行稳定版要求requirements锁定1.61.0"),
    ], columns=["项目", "说明"])
    score_rules_export = pd.concat([
        score_rules.assign(规则组="V5.1.1冻结100分"),
        challenger_rules.assign(规则组="V5.2/V5.4冻结分层"),
    ], ignore_index=True, sort=False)
    files = {
        "01_run_summary_v5_8.csv": run_summary,
        "02_history_completeness_outcomes_v5_8.csv": history_completeness,
        "03_history_complete_rank_acceptance_v5_8.csv": history_acceptance,
        "04_history_completeness_definitions_v5_8.csv": v58_rules,
        "05_v57_predeclared_acceptance_recheck_v5_8.csv": acceptance,
        "06_v57_rank_definitions_frozen_v5_8.csv": v57_rules,
        "07_v57_feature_tier_outcomes_v5_8.csv": feature_tiers,
        "08_all_nine_rank_top3_two_winner_audit_v5_8.csv": top3_explosion,
        "09_all_nine_rank_cohort_outcomes_v5_8.csv": rank_cohorts,
        "10_explosion_class_definitions_v5_8.csv": explosion_definitions,
        "11_explosion_class_outcomes_v5_8.csv": explosion_classes,
        "12_weekly_candidate_calendar_v5_8.csv": ranked_calendar,
        "13_slim_w8_candidates_with_history_completeness_v5_8.csv": slim_candidates,
        "14_frozen_old_rules_v5_8.csv": score_rules_export,
        "15_rejection_audit_v5_8.csv": rejection_audit,
        "16_metadata_v5_8.csv": metadata,
        "17_api_errors_v5_8.csv": pd.DataFrame({"错误": API_ERRORS}),
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
        f"完成：N=6/7/9的W8候选分别为"
        f"{len(eligible[eligible['SKDJ_N'].eq(6)])}/"
        f"{len(eligible[eligible['SKDJ_N'].eq(7)])}/"
        f"{len(eligible[eligible['SKDJ_N'].eq(9)])}个；"
        f"实际行情约{round((market_end_date - data_start_date).days / 7, 1)}周；"
        f"结果{'已保存' if persisted else '仅当前页面可下载'}。")
    st.subheader("N=6 / N=7 / N=9运行摘要")
    render_plain_table(run_summary)
    st.subheader("N=6历史完整度分布与表现")
    render_plain_table(history_completeness[
        history_completeness["SKDJ_N"].eq(6)
        & history_completeness["时间分段"].isin(
            ["全部区间", "前段观察", "后段冻结检验"])], 30)
    st.subheader("N=6两套历史完整度修正验收")
    render_plain_table(history_acceptance, 20)
    st.subheader("N=6四套V5.7排名的冻结验收复核")
    render_plain_table(acceptance, 20)
    st.subheader("N=6双因子P1/P2/P3分层表现")
    render_plain_table(feature_tiers[
        feature_tiers["SKDJ_N"].eq(6)
        & feature_tiers["时间分段"].isin(
            ["全部区间", "前段观察", "后段冻结检验"])], 30)
    st.subheader("N=6九套排名的三仓爆发力")
    render_plain_table(top3_explosion[
        top3_explosion["SKDJ_N"].eq(6)
        & top3_explosion["时间分段"].isin(
            ["全部区间", "前段观察", "后段冻结检验"])], 30)
    st.caption("结果ZIP保留17个审计文件；核心新增是历史完整度分组和两套修正排名，不重复31项特征发现。")
    render_download(result_zip, result_name, f"v58_current_{request_signature}")


if __name__ == "__main__":
    main()
