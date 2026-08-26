# -*- coding: utf-8 -*-
"""周线SKDJ N6股票池、日线MACD红2与Top5生命周期回测 V6.14。

每天使用截至当日收盘可见的未完成周线生成N=6、M=3的SKDJ准备池，再由
日线MACD第2根红柱确认主买点，下一交易日开盘模拟成交。每天最多选择评分
前5名并为每个事件提供独立虚拟资金，不运行三仓资金组合；红1和红3只保留
为相同规则下的影子买点对照。

持仓不再固定40日。收盘跌破买入价5%、或跌破买入以来截至上一交易日最高
价回撤15%的保护线时，于下一可交易日开盘退出；挑战方案在最高收盘达到
+20%后，再保护最高收盘浮盈的一半。以下保留V6.13及更早版本的函数，最后
定义的V6.14 ``main`` 是唯一入口，以降低重写历史数据、行业池、缓存和交易
成本基础设施带来的回归风险。
"""
from __future__ import annotations

import bisect
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
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts

TITLE = "周线SKDJ观察池与日线强度确认买点审计 V6.2"
VERSION = "V6.2-WEEKLY-SKDJ-DAILY-STRENGTH-CONFIRMATION"
UI_PATCH = "V6.2-STRENGTH-3-5-8-10-AND-SAME-CYCLE-AGE-AUDIT"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# 沿用旧行情缓存目录，以便直接复用V4.7已经下载的更长历史数据。
PRICE_CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")
# 200周预热会改变历史周期字段，不能复用30周预热的逐股票检查点。
CHECKPOINT_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_2_checkpoints")
RESULT_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_2_results")
JOB_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_2_jobs")

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
SWING_LOOKBACK_WEEKS = 52
SWING_REVERSAL_PCT = 15.0
SWING_LEVELS = (30.0, 50.0, 100.0)
NEGATIVE_ABOVE_MA20_PCT = 70.0
NEGATIVE_VOLUME_EXPAND_PCT = 35.0
NEGATIVE_K_RISING_PCT = 60.0

# Pairwise LTR uses only signal-time fields.  All source columns are transformed
# into within-N/within-week percentile ranks before fitting, so different market
# regimes and units are comparable.  Feature count is intentionally small
# because the effective independent sample is the number of signal weeks.
LTR_FEATURES = (
    "Signal_K_Change_1W",
    "Signal_KD_Spread",
    "Signal_Prior_Below25_Streak",
    "Signal_Volume_Ratio_5W",
    "Signal_Week_Return_pct",
    "Signal_Return_4W_pct",
    "Signal_Return_12W_pct",
    "Signal_Relative_Industry_12W_pct",
    "Breadth_MA20_Rising_Pct",
    "Industry_Resonance_Pct",
    "Signal_MA20_Slope_4W_pct",
    "Signal_K_Thrust_per_AbsWeekReturn",
    "Swing52_Close_to_High_pct",
    "Signal_VCP_Range_4W_vs_12W",
    "Signal_Prior_GC_Reached75_Count_Last3",
)
LTR_MIN_TRAIN_WEEKS = 20
LTR_MIN_TRAIN_ROWS = 160
LTR_MAX_PAIRS_PER_WEEK = 80
LTR_C = 0.35
LTR_NEWTON_MAX_ITER = 100
LTR_NEWTON_TOL = 1e-8
LTR_PRIMARY_N = 6

# V6.1 timing hypotheses.  The threshold sweep is deliberately exported so
# that one attractive result cannot silently become the rule.
EARLY_PRIMARY_N = 6
EARLY_WEEKLY_K_MIN = 15.0
EARLY_WEEKLY_K_MAX = 25.0
EARLY_RED_AGE_MIN = 2
EARLY_RED_AGE_MAX = 5
DAILY_AUDIT_DAYS = 40
DAILY_EXTENDED_DAYS = 60
MACD_REMAINING_THRESHOLDS = (10.0, 20.0, 30.0)
MACD_HEALTHY_REMAINING_PCT = 35.0
MACD_HEALTHY_RETENTION_PCT = 75.0
FUTURE_WEEKLY_CROSS_DAYS = 42
CACHE_TTL_SECONDS = 72 * 3600
STRENGTH_THRESHOLDS = (3.0, 5.0, 8.0, 10.0)
COHORT_RED_AGES = (2, 3, 4, 5)
STRENGTH_MIN_REMAINING_PCT = 75.0
STRENGTH_MAX_PRIOR_RALLY_PCT = 30.0

# V6.12 full-slot early-F exit audit.  Internal v63-v611 helpers
# are retained to minimize regression risk; the final ``main`` is the only
# entry point.
# The stock-level event rows are byte-compatible with V6.4, so its checkpoint
# signature and directory are deliberately reused.  Results/jobs use new V6.12
# directories and cannot collide with a completed V6.4 result.
V63_TITLE = "周线SKDJ全额三仓与F级早期失败退出回测 V6.12"
V63_VERSION = "V6.12-SKDJ-FULL-SLOT-EARLY-F-EXIT"
V63_UI_PATCH = "V6.12-FULL100-TOP3-D3-D5-D7-HARD-FAILURE"
V65_EVENT_ENGINE_VERSION = "V6.4-SKDJ-STATE-CONFIRMATION-SELECTIVE-REENTRY"
V63_CHECKPOINT_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_4_checkpoints")
V63_RESULT_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_12_results")
V63_JOB_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_12_jobs")
V63_CONFIRM_DEADLINES = (7, 14)
V63_REENTRY_WINDOW_DAYS = 42
V63_STRENGTH_THRESHOLD = 3.0
V63_SCORE_K_WEIGHT = 2
V63_SCORE_AGE_WEIGHT = 2
V63_SCORE_KD_WEIGHT = 1
V64_PRIMARY_CONFIRM_DAYS = 14
V64_CROSS_HIGH = "高质量确认_红柱扩张且已涨10至30"
V64_CROSS_ORDINARY = "普通确认_不加仓"
V64_CROSS_OVERHEATED = "已涨超过30_早仓保护_禁止新追"
V64_CROSS_NONE = "42日内未确认"
V64_POST_CROSS_REMAINING = (10.0, 20.0, 30.0)
V65_TRIAL_WEIGHTS = (1.0 / 3.0, 1.0 / 2.0)
V66_INITIAL_CAPITAL = 300_000.0
V66_MAX_STOCKS = 3
V66_FULL_SLOT_CAPITAL = V66_INITIAL_CAPITAL / V66_MAX_STOCKS
V66_BOARD_LOT = 100
V66_DEFAULT_DRAWS = 300
V66_RANDOM_SEED = 20260827
V68_ORDINARY_STOP_PCT = -10.0
V68_RISK_WINDOWS = (0, 5, 10)
V611_FEATURES = (
    "Signal_K", "Signal_D", "Signal_KD_Spread", "Signal_K_Change_1W",
    "Signal_Prior_Below25_Streak", "Signal_MA20_Slope_4W_pct",
    "Signal_Volume_Ratio_5W", "Signal_Week_Return_pct",
    "Daily_MACD_Red_Age", "Daily_MACD_Remaining_pct",
    "Daily_MACD_Retention_pct", "Signal_Rally_From_Red_Start_pct",
    "Turnover_Rate", "V611_MACD_Hist_to_Price",
)
V611_MIN_TRAIN_DATES = 20
V611_MIN_TRAIN_ROWS = 160
V611_MAX_PAIRS_PER_DATE = 80
V612_EARLY_FAILURE_DAYS = (3, 5, 7)
V612_PRICE_FAILURE_PCT = -5.0
V612_MACD_REMAINING_PCT = 10.0

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


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_trade_calendar(start_date: str, end_date: str) -> list[str]:
    frame = safe_get(
        "trade_cal", required=True, exchange="SSE",
        start_date=start_date, end_date=end_date)
    if frame.empty:
        raise RuntimeError("交易日历为空")
    return sorted(frame.loc[frame["is_open"].eq(1), "cal_date"].astype(str).tolist())


@st.cache_data(ttl=CACHE_TTL_SECONDS)
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


def swing_character_features(weekly: pd.DataFrame, signal_date: str) -> dict[str, Any]:
    """Describe non-overlapping pre-signal rallies without future leakage.

    Only the 52 complete weeks strictly before ``signal_date`` are examined.
    A trough-to-peak leg becomes completed only after a later weekly low has
    fallen at least 15% from that peak.  The last unconfirmed leg is exported
    separately, so one long trend is not counted repeatedly as nested rallies.
    """
    prefix = "Swing52_"
    out: dict[str, Any] = {
        f"{prefix}History_Weeks": 0.0,
        f"{prefix}Completed_Legs": 0.0,
        f"{prefix}Count_30": 0.0,
        f"{prefix}Count_50": 0.0,
        f"{prefix}Count_100": 0.0,
        f"{prefix}Count_30_Including_Ongoing": 0.0,
        f"{prefix}Count_50_Including_Ongoing": 0.0,
        f"{prefix}Count_100_Including_Ongoing": 0.0,
        f"{prefix}Max_Completed_Rally_pct": np.nan,
        f"{prefix}Median_Completed_Rally_pct": np.nan,
        f"{prefix}Last_Completed_Rally_pct": np.nan,
        f"{prefix}Weeks_Since_Last_Completed_Peak": np.nan,
        f"{prefix}Ongoing_Max_Rally_pct": np.nan,
        f"{prefix}Ongoing_Close_Rally_pct": np.nan,
        f"{prefix}Ongoing_Drawdown_From_Peak_pct": np.nan,
        f"{prefix}Range_pct": np.nan,
        f"{prefix}Current_Position_pct": np.nan,
        f"{prefix}Close_to_High_pct": np.nan,
        f"{prefix}Higher_High_Ratio_pct": np.nan,
        f"{prefix}Higher_Low_Ratio_pct": np.nan,
        f"{prefix}Structure_State": "数据不足",
        f"{prefix}Activity_Class": "数据不足",
    }
    if weekly.empty:
        return out
    history = weekly[
        weekly["trade_date"].astype(str).lt(str(signal_date))
    ].tail(SWING_LOOKBACK_WEEKS).copy().reset_index(drop=True)
    history = history.dropna(subset=["high", "low", "close"])
    out[f"{prefix}History_Weeks"] = float(len(history))
    if len(history) < 8:
        return out

    lows = pd.to_numeric(history["low"], errors="coerce")
    highs = pd.to_numeric(history["high"], errors="coerce")
    closes = pd.to_numeric(history["close"], errors="coerce")
    dates = history["trade_date"].astype(str).tolist()
    completed: list[dict[str, Any]] = []
    trough_price = finite_num(lows.iloc[0])
    trough_pos = 0
    # Do not assume whether the first week's high occurred before or after its
    # low.  A valid peak must come from a later complete week.
    peak_price = trough_price
    peak_pos = 0

    for position in range(1, len(history)):
        low = finite_num(lows.iloc[position])
        high = finite_num(highs.iloc[position])
        if not math.isfinite(low) or not math.isfinite(high):
            continue
        # Confirm a drawdown only against a peak from an earlier week.  The
        # order of this week's high and low is unknown in weekly OHLC, so this
        # week's high must not confirm a reversal with this week's low.
        gain = (peak_price / trough_price - 1.0) * 100.0
        drawdown = (low / peak_price - 1.0) * 100.0 if peak_price > 0 else np.nan
        if gain > 0 and drawdown <= -SWING_REVERSAL_PCT:
            completed.append({
                "trough": trough_price, "trough_pos": trough_pos,
                "peak": peak_price, "peak_pos": peak_pos,
                "gain": gain, "trough_date": dates[trough_pos],
                "peak_date": dates[peak_pos],
            })
            # The reversal week confirms the old peak.  Its low may become the
            # next trough, but its high is not reused because within-week order
            # is unknown; this deliberately avoids double counting.
            trough_price, trough_pos = low, position
            peak_price, peak_pos = low, position
            continue
        if not math.isfinite(trough_price) or low < trough_price:
            trough_price, trough_pos = low, position
            peak_price, peak_pos = low, position
            continue
        if high > peak_price:
            peak_price, peak_pos = high, position

    gains = [finite_num(leg["gain"]) for leg in completed]
    finite_gains = [value for value in gains if math.isfinite(value)]
    ongoing_max_raw = (
        (peak_price / trough_price - 1.0) * 100.0
        if trough_price > 0 and math.isfinite(peak_price) else np.nan)
    latest_close = finite_num(closes.iloc[-1])
    ongoing_close = (
        (latest_close / trough_price - 1.0) * 100.0
        if trough_price > 0 and math.isfinite(latest_close) else np.nan)
    ongoing_max = (
        ongoing_max_raw if peak_pos > trough_pos
        else max(ongoing_close, 0.0) if math.isfinite(ongoing_close) else np.nan)
    ongoing_drawdown = (
        (latest_close / peak_price - 1.0) * 100.0
        if peak_price > 0 and math.isfinite(latest_close) else np.nan)

    out[f"{prefix}Completed_Legs"] = float(len(completed))
    out[f"{prefix}Max_Completed_Rally_pct"] = (
        max(finite_gains) if finite_gains else np.nan)
    out[f"{prefix}Median_Completed_Rally_pct"] = (
        finite_num(pd.Series(finite_gains).median()) if finite_gains else np.nan)
    out[f"{prefix}Last_Completed_Rally_pct"] = (
        finite_gains[-1] if finite_gains else np.nan)
    out[f"{prefix}Weeks_Since_Last_Completed_Peak"] = (
        float(len(history) - 1 - completed[-1]["peak_pos"])
        if completed else np.nan)
    out[f"{prefix}Ongoing_Max_Rally_pct"] = ongoing_max
    out[f"{prefix}Ongoing_Close_Rally_pct"] = ongoing_close
    out[f"{prefix}Ongoing_Drawdown_From_Peak_pct"] = ongoing_drawdown

    for level in SWING_LEVELS:
        completed_count = sum(value >= level for value in finite_gains)
        ongoing_count = int(math.isfinite(ongoing_max) and ongoing_max >= level)
        out[f"{prefix}Count_{int(level)}"] = float(completed_count)
        out[f"{prefix}Count_{int(level)}_Including_Ongoing"] = float(
            completed_count + ongoing_count)

    year_low, year_high = finite_num(lows.min()), finite_num(highs.max())
    out[f"{prefix}Range_pct"] = (
        (year_high / year_low - 1.0) * 100.0 if year_low > 0 else np.nan)
    out[f"{prefix}Current_Position_pct"] = (
        (latest_close - year_low) / (year_high - year_low) * 100.0
        if year_high > year_low and math.isfinite(latest_close) else np.nan)
    out[f"{prefix}Close_to_High_pct"] = (
        (latest_close / year_high - 1.0) * 100.0
        if year_high > 0 and math.isfinite(latest_close) else np.nan)

    peaks = [finite_num(leg["peak"]) for leg in completed]
    troughs = [finite_num(leg["trough"]) for leg in completed]
    # A current leg that has already risen 30% is observable at the signal
    # time and can be used once in the higher-high/higher-low structure test.
    if math.isfinite(ongoing_max) and ongoing_max >= 30.0:
        peaks.append(peak_price)
        troughs.append(trough_price)
    if len(peaks) >= 2:
        high_flags = [peaks[i] > peaks[i - 1] for i in range(1, len(peaks))]
        low_flags = [troughs[i] > troughs[i - 1] for i in range(1, len(troughs))]
        high_ratio = float(np.mean(high_flags) * 100.0)
        low_ratio = float(np.mean(low_flags) * 100.0)
        out[f"{prefix}Higher_High_Ratio_pct"] = high_ratio
        out[f"{prefix}Higher_Low_Ratio_pct"] = low_ratio
        if high_ratio >= 50.0 and low_ratio >= 50.0:
            structure = "高低点共同抬高"
        elif high_ratio >= 50.0:
            structure = "仅高点抬高"
        elif low_ratio >= 50.0:
            structure = "仅低点抬高"
        else:
            structure = "高低点均未抬高"
    else:
        structure = "周期不足"
    out[f"{prefix}Structure_State"] = structure

    completed30 = int(out[f"{prefix}Count_30"])
    repeated30 = int(out[f"{prefix}Count_30_Including_Ongoing"])
    if repeated30 >= 2:
        activity = "反复爆发型"
    elif (completed30 == 0 and math.isfinite(ongoing_max)
          and ongoing_max >= 30.0 and math.isfinite(ongoing_drawdown)
          and ongoing_drawdown > -SWING_REVERSAL_PCT):
        activity = "持续趋势型"
    elif repeated30 == 1:
        activity = "一次爆发型"
    else:
        activity = "低活跃型"
    out[f"{prefix}Activity_Class"] = activity
    return out


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
    weekly_range_pct = (
        (work["high"] - work["low"])
        / work["close"].shift(1).replace(0, np.nan) * 100.0)
    prior4_range = weekly_range_pct.shift(1).rolling(4, min_periods=4).mean()
    prior12_range = weekly_range_pct.shift(1).rolling(12, min_periods=8).mean()
    work["VCP_Range_4W_vs_12W"] = (
        prior4_range / prior12_range.replace(0, np.nan))
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
    event_signal_end = str(config.get("event_signal_end", config["signal_end"]))
    crosses: list[tuple[int, pd.Series]] = []
    indicators: dict[int, pd.DataFrame] = {}
    for n in SKDJ_NS:
        indicator = add_skdj(weekly_base, n)
        indicators[n] = indicator
        breadth_weeks = indicator[
            indicator["trade_date"].astype(str).between(
                config["signal_start"], event_signal_end)]
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
                config["signal_start"], event_signal_end)
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
        swing_character = swing_character_features(weekly_base, signal_date)
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
            "Signal_VCP_Range_4W_vs_12W": finite_num(
                signal.get("VCP_Range_4W_vs_12W")),
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
            **swing_character,
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


def add_v59_audit_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add transparent V5.9 labels; do not modify any frozen ranking."""
    work = frame.copy()
    activity = work.get(
        "Swing52_Activity_Class",
        pd.Series("数据不足", index=work.index)).astype(str)
    structure = work.get(
        "Swing52_Structure_State",
        pd.Series("数据不足", index=work.index)).astype(str)
    relative_12w = numeric(work, "Signal_Relative_Industry_12W_pct")
    current_position = numeric(work, "Swing52_Current_Position_pct")
    strong_structure = structure.eq("高低点共同抬高")
    repeated = activity.eq("反复爆发型")
    persistent = activity.eq("持续趋势型")
    work["V59_Swing_Leader_Candidate"] = (
        (repeated & strong_structure | persistent)
        & relative_12w.gt(0)
        & current_position.ge(40.0))
    work["V59_Stock_Character"] = np.select(
        [work["V59_Swing_Leader_Candidate"],
         repeated,
         persistent,
         activity.eq("一次爆发型"),
         activity.eq("低活跃型")],
        ["结构强活跃候选", "反复波动但结构未确认", "持续趋势但相对强度未确认",
         "仅一次大波段", "低活跃"],
        default="数据不足")

    crowded = numeric(work, "Breadth_Above_MA20_Pct").ge(
        NEGATIVE_ABOVE_MA20_PCT)
    volume_fade = numeric(work, "Breadth_Volume_Expand_Pct").le(
        NEGATIVE_VOLUME_EXPAND_PCT)
    k_fade = numeric(work, "Breadth_K_Rising_Pct").le(
        NEGATIVE_K_RISING_PCT)
    work["V59_Negative_OnlyOneCandidate"] = numeric(
        work, "Candidates_This_Week").eq(1)
    work["V59_Negative_CrowdedVolumeFade"] = crowded & volume_fade
    work["V59_Negative_CrowdedKFade"] = crowded & k_fade
    work["V59_Negative_CrowdedBothFade"] = crowded & volume_fade & k_fade
    work["V59_Negative_LowActivity"] = (
        numeric(work, "Swing52_Count_30_Including_Ongoing").eq(0)
        & numeric(work, "Swing52_Range_pct").lt(30.0))
    return work


def v59_swing_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        ("观察窗口", "信号前52个完整周", "严格排除信号周和买入后行情"),
        ("独立上涨段", "低点到高点后回撤≥15%才结束", "互不重叠；同一轮上涨不会重复计数"),
        ("当前上涨段", "尚未出现15%回撤确认", "单独导出，不冒充已完成波段"),
        ("反复爆发型", "含当前段在内≥2次达到30%", "检验多次活跃是否比一次异动更可重复"),
        ("持续趋势型", "无已完成30%波段，但当前段已≥30%", "保留不靠反复回撤的趋势龙头"),
        ("一次爆发型", "含当前段仅1次达到30%", "与反复爆发分开"),
        ("低活跃型", "过去52周没有30%独立上涨段", "只作审计，不直接删除"),
        ("结构强候选", "反复爆发且高低点共同抬高，或持续趋势；同时行业相对12周强度>0且年内位置≥40%", "只作分组，不进入V5.8九套排名"),
        ("主要判卷", "W8 S/A/B/F、MFE及50/70/100%命中", "寻找翻倍股和高爆发股的区别，不追求低波动平滑"),
    ], columns=["项目", "定义", "说明"])


def v59_negative_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        ("当周仅1只候选", "Candidates_This_Week=1", "用户已同意可放弃该周，不改变其他周"),
        ("行业拥挤但量能不扩张",
         f"行业站上MA20比例≥{NEGATIVE_ABOVE_MA20_PCT:g}%，且放量比例≤{NEGATIVE_VOLUME_EXPAND_PCT:g}%",
         "前版探索中F级集中，V5.9正式核验误杀率"),
        ("行业拥挤但K动能衰减",
         f"行业站上MA20比例≥{NEGATIVE_ABOVE_MA20_PCT:g}%，且K上升比例≤{NEGATIVE_K_RISING_PCT:g}%",
         "与量能衰减分开验证"),
        ("行业拥挤且量价双衰减", "同时满足前两条", "更严格、预计剔除更少"),
        ("52周低活跃", "没有30%独立上涨段且全年高低波幅<30%", "检验是否可清除稳定但缺乏爆发潜力的股票"),
        ("验收口径", "高F捕获、低A/S误杀、前后段方向一致、不过度增加空窗周", "本版只审计，不自动成为硬条件"),
    ], columns=["负面规则", "信号时点定义", "用途"])


def _v59_outcome_metrics(selected: pd.DataFrame) -> dict[str, Any]:
    classes = selected.get(
        "Explosion_Class_W8",
        pd.Series(index=selected.index, dtype=str)).astype(str)
    mfe = numeric(selected, "Entry_W8_MFE_Net_pct")
    return {
        "事件数": len(selected),
        "不同股票": selected["ts_code"].nunique() if len(selected) else 0,
        "覆盖信号周": selected["Signal_Week"].nunique() if len(selected) else 0,
        "S级比例%": classes.eq("S").mean() * 100 if len(selected) else np.nan,
        "A或S比例%": classes.isin(["A", "S"]).mean() * 100 if len(selected) else np.nan,
        "B及以上比例%": classes.isin(["B", "A", "S"]).mean() * 100 if len(selected) else np.nan,
        "F级比例%": classes.eq("F").mean() * 100 if len(selected) else np.nan,
        "W8最大浮盈均值%": mfe.mean(),
        "W8最大浮盈中位%": mfe.median(),
        "W8达到50%比例%": mfe.ge(50.0).mean() * 100 if len(selected) else np.nan,
        "W8达到70%比例%": mfe.ge(70.0).mean() * 100 if len(selected) else np.nan,
        "W8达到100%比例%": mfe.ge(100.0).mean() * 100 if len(selected) else np.nan,
        "W8最大不利波动均值%": numeric(
            selected, "Entry_W8_MAE_Raw_pct").mean(),
    }


def v59_swing_outcome_audit(
        eligible: pd.DataFrame, periods: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            dimensions: list[tuple[str, list[tuple[str, pd.Series]]]] = [
                ("股性分类", [(name, base["V59_Stock_Character"].astype(str).eq(name))
                           for name in ("结构强活跃候选", "反复波动但结构未确认",
                                        "持续趋势但相对强度未确认", "仅一次大波段", "低活跃", "数据不足")]),
                ("30%波段次数(含当前)", [
                    ("0次", numeric(base, "Swing52_Count_30_Including_Ongoing").eq(0)),
                    ("1次", numeric(base, "Swing52_Count_30_Including_Ongoing").eq(1)),
                    ("2次", numeric(base, "Swing52_Count_30_Including_Ongoing").eq(2)),
                    ("≥3次", numeric(base, "Swing52_Count_30_Including_Ongoing").ge(3))]),
                ("50%波段次数(含当前)", [
                    ("0次", numeric(base, "Swing52_Count_50_Including_Ongoing").eq(0)),
                    ("1次", numeric(base, "Swing52_Count_50_Including_Ongoing").eq(1)),
                    ("≥2次", numeric(base, "Swing52_Count_50_Including_Ongoing").ge(2))]),
                ("100%波段次数(含当前)", [
                    ("0次", numeric(base, "Swing52_Count_100_Including_Ongoing").eq(0)),
                    ("1次", numeric(base, "Swing52_Count_100_Including_Ongoing").eq(1)),
                    ("≥2次", numeric(base, "Swing52_Count_100_Including_Ongoing").ge(2))]),
                ("52周总波幅", [
                    ("<30%", numeric(base, "Swing52_Range_pct").lt(30)),
                    ("30%～60%", numeric(base, "Swing52_Range_pct").between(30, 60, inclusive="left")),
                    ("60%～100%", numeric(base, "Swing52_Range_pct").between(60, 100, inclusive="left")),
                    ("≥100%", numeric(base, "Swing52_Range_pct").ge(100))]),
            ]
            for dimension, groups in dimensions:
                for label, mask in groups:
                    selected = base[mask.fillna(False)]
                    rows.append({"SKDJ_N": n, "时间分段": period["name"],
                                 "审计维度": dimension, "分组": label,
                                 **_v59_outcome_metrics(selected)})
    return pd.DataFrame(rows)


def v59_negative_filter_audit(
        eligible: pd.DataFrame, periods: list[dict[str, Any]]) -> pd.DataFrame:
    rules = (
        ("不剔除_冻结基准", None),
        ("当周仅1只候选", "V59_Negative_OnlyOneCandidate"),
        ("行业拥挤但量能不扩张", "V59_Negative_CrowdedVolumeFade"),
        ("行业拥挤但K动能衰减", "V59_Negative_CrowdedKFade"),
        ("行业拥挤且量价双衰减", "V59_Negative_CrowdedBothFade"),
        ("52周低活跃", "V59_Negative_LowActivity"),
    )
    universes = (("全部候选", None), ("V5.7双因子前20%", "V57_DualH2_Top20Pct"))
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            period_base = select_period(base_n, period)
            for universe_name, universe_flag in universes:
                base = (period_base if universe_flag is None
                        else period_base[true_mask(period_base, universe_flag)])
                base_classes = base.get(
                    "Explosion_Class_W8",
                    pd.Series(index=base.index, dtype=str)).astype(str)
                base_f = int(base_classes.eq("F").sum())
                base_as = int(base_classes.isin(["A", "S"]).sum())
                for rule_name, rule_column in rules:
                    remove = (pd.Series(False, index=base.index) if rule_column is None
                              else true_mask(base, rule_column))
                    removed = base[remove]
                    retained = base[~remove]
                    removed_classes = removed.get(
                        "Explosion_Class_W8",
                        pd.Series(index=removed.index, dtype=str)).astype(str)
                    retained_metrics = _v59_outcome_metrics(retained)
                    rows.append({
                        "SKDJ_N": n, "时间分段": period["name"],
                        "候选范围": universe_name, "负面规则": rule_name,
                        "剔除事件": len(removed),
                        "剔除比例%": len(removed) / len(base) * 100 if len(base) else np.nan,
                        "剔除中F级比例%": removed_classes.eq("F").mean() * 100 if len(removed) else np.nan,
                        "剔除中A或S比例%": removed_classes.isin(["A", "S"]).mean() * 100 if len(removed) else np.nan,
                        "F级捕获率%": removed_classes.eq("F").sum() / base_f * 100 if base_f else np.nan,
                        "A或S误杀率%": removed_classes.isin(["A", "S"]).sum() / base_as * 100 if base_as else np.nan,
                        "新增空窗周": max(
                            int(base["Signal_Week"].nunique())
                            - int(retained["Signal_Week"].nunique()), 0),
                        **{f"保留_{key}": value for key, value in retained_metrics.items()},
                    })
    return pd.DataFrame(rows)


def v59_negative_frozen_top3_audit(
        eligible: pd.DataFrame, periods: list[dict[str, Any]]) -> pd.DataFrame:
    """Remove each negative group, then keep the frozen V5.7 order."""
    rules = (
        ("不剔除_冻结基准", None),
        ("当周仅1只候选", "V59_Negative_OnlyOneCandidate"),
        ("行业拥挤但量能不扩张", "V59_Negative_CrowdedVolumeFade"),
        ("行业拥挤但K动能衰减", "V59_Negative_CrowdedKFade"),
        ("行业拥挤且量价双衰减", "V59_Negative_CrowdedBothFade"),
        ("52周低活跃", "V59_Negative_LowActivity"),
    )
    rows: list[dict[str, Any]] = []
    for n in SKDJ_NS:
        base_n = eligible[eligible["SKDJ_N"].eq(n)]
        for period in periods:
            base = select_period(base_n, period)
            for rule_name, rule_column in rules:
                retained = (base if rule_column is None
                            else base[~true_mask(base, rule_column)])
                selected = (retained.sort_values(
                    ["Signal_Week", "V57_DualH2_Weekly_Rank", "ts_code"])
                    .groupby("Signal_Week", sort=False).head(3))
                metrics = _v59_outcome_metrics(selected)
                weekly_as = (selected.assign(
                    _as=selected.get("Explosion_Class_W8", "").astype(str).isin(["A", "S"]))
                    .groupby("Signal_Week")["_as"].sum()) if len(selected) else pd.Series(dtype=float)
                rows.append({
                    "SKDJ_N": n, "时间分段": period["name"],
                    "冻结排序": "V5.7双因子层内H2", "负面规则": rule_name,
                    **metrics,
                    "前三至少2只A或S周": int(weekly_as.ge(2).sum()),
                    "前三至少2只A或S占所选周%": weekly_as.ge(2).mean() * 100 if len(weekly_as) else np.nan,
                })
    return pd.DataFrame(rows)


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
        "Breadth_Above_MA20_Pct", "Breadth_Volume_Expand_Pct",
        "Breadth_K_Rising_Pct",
        "V57_K_Efficiency_Weekly_Pct", "V57_Industry_MA20_Weekly_Pct",
        "V57_K_Favorable20", "V57_Breadth_Favorable20",
        "V57_Favorable_Count", "V57_Feature_Data_Valid",
        "V57_Factor_Tier", "V57_Factor_Tier_Order",
        "V57_TwoFactor_Rank_Sum",
        "Signal_Prior_GC_Valid_Count_Last3",
        "Signal_Prior_GC_Reached75_Count_Last3", "Signal_Prior_GC1_Peak_K",
        "Signal_Relative_Industry_12W_pct",
        "Swing52_History_Weeks", "Swing52_Completed_Legs",
        "Swing52_Count_30", "Swing52_Count_50", "Swing52_Count_100",
        "Swing52_Count_30_Including_Ongoing",
        "Swing52_Count_50_Including_Ongoing",
        "Swing52_Count_100_Including_Ongoing",
        "Swing52_Max_Completed_Rally_pct", "Swing52_Median_Completed_Rally_pct",
        "Swing52_Last_Completed_Rally_pct",
        "Swing52_Weeks_Since_Last_Completed_Peak",
        "Swing52_Ongoing_Max_Rally_pct", "Swing52_Ongoing_Close_Rally_pct",
        "Swing52_Ongoing_Drawdown_From_Peak_pct", "Swing52_Range_pct",
        "Swing52_Current_Position_pct", "Swing52_Higher_High_Ratio_pct",
        "Swing52_Higher_Low_Ratio_pct", "Swing52_Structure_State",
        "Swing52_Activity_Class", "V59_Stock_Character",
        "V59_Swing_Leader_Candidate", "V59_Negative_OnlyOneCandidate",
        "V59_Negative_CrowdedVolumeFade", "V59_Negative_CrowdedKFade",
        "V59_Negative_CrowdedBothFade", "V59_Negative_LowActivity",
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


def ltr_definitions() -> pd.DataFrame:
    rows = [
        ("候选池", "保持宽池", "科技股、股价/市值过滤、K上穿25、25线下1～5周；不增加52周股性硬过滤"),
        ("分组", "Signal_Week", "只比较同一个N、同一个信号周内的候选股票"),
        ("标签", "F/B/A/S=0/1/2/3", "W8内分别未先到+10、先到+10、先到+20、先到+30；全部与-10%比较"),
        ("等级收益", "0/1/3/7", "成对训练按等级收益差加权，S对F的比较权重最高"),
        ("模型", "成对逻辑回归", "高等级减低等级作为正样本，反向差值作为负样本；L2正则，不新增部署依赖"),
        ("特征尺度", "周内百分位", "每项特征先转换为同N同周百分位；缺失值中性填充0.5"),
        ("防泄漏", "W8_End_Date < 当前预测日", "滚动预测时，只使用在当前周以前已经完整揭晓W8结果的事件"),
        ("最小训练", f"{LTR_MIN_TRAIN_WEEKS}周/{LTR_MIN_TRAIN_ROWS}行", "不足时不伪造LTR历史名次；实盘使用截止日前全部成熟历史"),
        ("周权重", "每周总权重相等", "每周最多抽取固定数量的有效等级对，避免信号扎堆周支配模型"),
        ("验收", "Top3爆发力", "与精确随机Top3、原100分和V5.7双因子H2在完全相同OOS周比较"),
        ("实盘观察", "最近完整交易周", "只显示状态和模型排名；不生成W8标签、不计入历史成绩"),
    ]
    return pd.DataFrame(rows, columns=["环节", "规则", "说明"])


def add_ltr_week_percentiles(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Transform signal-time features to comparable within-week percentiles."""
    work = frame.copy()
    keys = ["SKDJ_N", "Signal_Week"]
    columns: list[str] = []
    for source in LTR_FEATURES:
        target = f"LTRP_{source}"
        values = numeric(work, source)
        work[target] = (
            work.assign(_ltr_value=values).groupby(keys)["_ltr_value"]
            .rank(method="average", pct=True, ascending=True))
        columns.append(target)
    return work, columns


def _ltr_seed(text_value: str) -> int:
    payload = f"{RANDOM_SEED}|{text_value}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _fit_weighted_pairwise_logit(
        matrix: np.ndarray, labels: np.ndarray,
        sample_weights: np.ndarray) -> np.ndarray:
    """Solve a no-intercept L2 logistic model with deterministic Newton steps.

    Keeping this tiny solver in app.py avoids adding a new deployment package.
    Pair observations are symmetric (+difference and -difference), so an
    intercept is neither needed nor desirable for a relative ranking model.
    """
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=float)
    weights = np.asarray(sample_weights, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(weights):
        raise ValueError("成对LTR训练矩阵尺寸不一致")
    coefficients = np.zeros(x.shape[1], dtype=float)
    ridge = 1.0 / max(float(LTR_C), 1e-9)
    identity = np.eye(x.shape[1], dtype=float)
    for _ in range(LTR_NEWTON_MAX_ITER):
        logits = np.clip(x @ coefficients, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = x.T @ (weights * (probabilities - y)) + ridge * coefficients
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = x.T @ (x * curvature[:, None]) + ridge * identity
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        coefficients -= step
        if float(np.max(np.abs(step))) < LTR_NEWTON_TOL:
            break
    if not np.isfinite(coefficients).all():
        raise RuntimeError("成对LTR求解产生非有限系数")
    return coefficients


def fit_pairwise_ltr(
        train: pd.DataFrame, feature_columns: list[str]
        ) -> dict[str, Any] | None:
    """Fit a deterministic RankNet-style linear pairwise ranker.

    Pairs are constructed only within a signal week.  Every week has total
    sample weight one, while F/B/A/S gain differences 0/1/3/7 emphasize
    comparisons involving the desired explosive classes.
    """
    if train.empty:
        return None
    work = train.copy()
    grade = numeric(work, "Explosion_Grade_W8")
    work = work[grade.notna()].copy()
    work["_grade"] = numeric(work, "Explosion_Grade_W8")
    if (len(work) < LTR_MIN_TRAIN_ROWS
            or work["Signal_Week"].nunique() < LTR_MIN_TRAIN_WEEKS
            or work["_grade"].nunique() < 2):
        return None

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    weight_rows: list[float] = []
    pair_count = 0
    gain_map = {0: 0.0, 1: 1.0, 2: 3.0, 3: 7.0}
    for week, group in work.groupby("Signal_Week", sort=True):
        group = group.reset_index(drop=True)
        matrix = group[feature_columns].apply(
            pd.to_numeric, errors="coerce").fillna(0.5).to_numpy(dtype=float)
        labels = pd.to_numeric(group["_grade"], errors="coerce").to_numpy(dtype=float)
        possible: list[tuple[int, int]] = []
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                if not math.isfinite(labels[left]) or not math.isfinite(labels[right]):
                    continue
                if labels[left] == labels[right]:
                    continue
                high, low = ((left, right) if labels[left] > labels[right]
                             else (right, left))
                possible.append((high, low))
        if not possible:
            continue
        if len(possible) > LTR_MAX_PAIRS_PER_WEEK:
            rng = np.random.default_rng(_ltr_seed(str(week)))
            chosen = rng.choice(
                len(possible), size=LTR_MAX_PAIRS_PER_WEEK, replace=False)
            pairs = [possible[int(position)] for position in sorted(chosen.tolist())]
        else:
            pairs = possible
        raw_weights = np.array([
            gain_map[int(labels[high])] - gain_map[int(labels[low])]
            for high, low in pairs], dtype=float)
        raw_weights = raw_weights / max(float(raw_weights.sum()), 1.0)
        for (high, low), pair_weight in zip(pairs, raw_weights):
            difference = matrix[high] - matrix[low]
            x_rows.extend([difference, -difference])
            y_rows.extend([1, 0])
            # Positive and reverse observations together contribute one week.
            weight_rows.extend([float(pair_weight) * 0.5,
                                float(pair_weight) * 0.5])
        pair_count += len(pairs)
    if pair_count == 0 or len(set(y_rows)) < 2:
        return None
    coefficients = _fit_weighted_pairwise_logit(
        np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=int),
        np.asarray(weight_rows, dtype=float))
    return {
        "coefficients": coefficients,
        "feature_columns": list(feature_columns),
        "train_rows": int(len(work)),
        "train_weeks": int(work["Signal_Week"].nunique()),
        "pair_count": int(pair_count),
        "train_start": str(work["Signal_Date"].astype(str).min()),
        "train_end": str(work["Signal_Date"].astype(str).max()),
    }


def predict_pairwise_ltr(
        frame: pd.DataFrame, bundle: dict[str, Any] | None
        ) -> pd.Series:
    if frame.empty or bundle is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    columns = list(bundle["feature_columns"])
    matrix = frame[columns].apply(
        pd.to_numeric, errors="coerce").fillna(0.5).to_numpy(dtype=float)
    coefficients = np.asarray(bundle["coefficients"], dtype=float)
    return pd.Series(matrix @ coefficients, index=frame.index, dtype=float)


def _assign_ltr_flags(work: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keys = ["SKDJ_N", "Signal_Week"]
    score_column = f"{prefix}_Score"
    rank_column = f"{prefix}_Weekly_Rank"
    valid = numeric(work, score_column).notna()
    ranked = work[valid].sort_values(
        ["SKDJ_N", "Signal_Week", score_column, "V57_DualH2_Weekly_Rank",
         "Score_Total_100", "ts_code"],
        ascending=[True, True, False, True, False, True]).copy()
    ranked[rank_column] = ranked.groupby(keys).cumcount() + 1
    work.loc[ranked.index, rank_column] = ranked[rank_column]
    candidate_count = numeric(work, "Candidates_This_Week").replace(0, np.nan)
    top20_count = np.ceil(candidate_count * 0.20).clip(lower=1)
    work[f"{prefix}_Top3"] = numeric(work, rank_column).le(3)
    work[f"{prefix}_Top20Pct"] = numeric(work, rank_column).le(top20_count)
    return work


def add_walk_forward_ltr(eligible: pd.DataFrame) -> pd.DataFrame:
    """Generate strictly walk-forward OOS ranks for N=6 historical events."""
    work, feature_columns = add_ltr_week_percentiles(eligible)
    work["LTR_OOS_Score"] = np.nan
    work["LTR_OOS_Weekly_Rank"] = np.nan
    work["LTR_OOS_Top3"] = False
    work["LTR_OOS_Top20Pct"] = False
    work["LTR_OOS_Available"] = False
    work["LTR_OOS_Train_Rows"] = 0
    work["LTR_OOS_Train_Weeks"] = 0
    primary = work[work["SKDJ_N"].eq(LTR_PRIMARY_N)].copy()
    for week, group in primary.groupby("Signal_Week", sort=True):
        prediction_date = str(group["Signal_Date"].astype(str).max())
        revealed = (
            primary.get("Entry_W8_End_Date", pd.Series(
                "", index=primary.index, dtype=str)).astype(str)
            .lt(prediction_date))
        train = primary[
            revealed & numeric(primary, "Explosion_Grade_W8").notna()].copy()
        bundle = fit_pairwise_ltr(train, feature_columns)
        if bundle is None:
            continue
        work.loc[group.index, "LTR_OOS_Score"] = predict_pairwise_ltr(
            work.loc[group.index], bundle)
        work.loc[group.index, "LTR_OOS_Available"] = True
        work.loc[group.index, "LTR_OOS_Train_Rows"] = bundle["train_rows"]
        work.loc[group.index, "LTR_OOS_Train_Weeks"] = bundle["train_weeks"]
    work = _assign_ltr_flags(work, "LTR_OOS")
    return work.sort_values(
        ["Signal_Date", "SKDJ_N", "LTR_OOS_Weekly_Rank", "ts_code"],
        na_position="last").reset_index(drop=True)


def fit_full_history_ltr(eligible: pd.DataFrame) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    prepared, feature_columns = add_ltr_week_percentiles(eligible)
    primary = prepared[
        prepared["SKDJ_N"].eq(LTR_PRIMARY_N)
        & numeric(prepared, "Explosion_Grade_W8").notna()].copy()
    return fit_pairwise_ltr(primary, feature_columns), prepared


def score_live_candidates(
        live: pd.DataFrame, history: pd.DataFrame,
        bundle: dict[str, Any] | None
        ) -> pd.DataFrame:
    if live.empty:
        return live.copy()
    work, feature_columns = add_ltr_week_percentiles(live)
    # A full-history model uses the same fixed feature list as live candidates.
    if bundle is not None and feature_columns != list(bundle["feature_columns"]):
        raise RuntimeError("实盘LTR特征列与历史训练不一致")
    work["LTR_Live_Score"] = np.nan
    primary = work["SKDJ_N"].eq(LTR_PRIMARY_N)
    work.loc[primary, "LTR_Live_Score"] = predict_pairwise_ltr(
        work.loc[primary], bundle)
    work["LTR_Live_Weekly_Rank"] = np.nan
    work["LTR_Live_Top3"] = False
    work["LTR_Live_Top20Pct"] = False
    work = _assign_ltr_flags(work, "LTR_Live")
    resonance_keys = ["Signal_Week", "ts_code"]
    confirmations = (work.groupby(resonance_keys)["SKDJ_N"]
                     .agg(lambda values: "/".join(
                         str(int(value)) for value in sorted(set(values))))
                     .rename("参数共振N").reset_index())
    counts = (work.groupby(resonance_keys)["SKDJ_N"].nunique()
              .rename("参数共振数量").reset_index())
    work = work.merge(confirmations, on=resonance_keys, how="left")
    work = work.merge(counts, on=resonance_keys, how="left")
    work["历史训练事件"] = int(bundle["train_rows"]) if bundle else 0
    work["历史训练周"] = int(bundle["train_weeks"]) if bundle else 0
    return work.sort_values(
        ["SKDJ_N", "LTR_Live_Weekly_Rank", "V57_DualH2_Weekly_Rank",
         "ts_code"], na_position="last").reset_index(drop=True)


def _top3_metrics(pool: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    signal_weeks = int(pool["Signal_Week"].nunique())
    classes = selected.get(
        "Explosion_Class_W8", pd.Series(index=selected.index, dtype=str)).astype(str)
    as_flag = classes.isin(["A", "S"])
    weekly_as = selected.assign(_as=as_flag.astype(int)).groupby(
        "Signal_Week")["_as"].sum()
    return {
        "事件数": int(len(selected)),
        "覆盖信号周": int(selected["Signal_Week"].nunique()),
        "S级比例%": classes.eq("S").mean() * 100 if len(selected) else np.nan,
        "A或S比例%": as_flag.mean() * 100 if len(selected) else np.nan,
        "B级以上比例%": classes.isin(["B", "A", "S"]).mean() * 100 if len(selected) else np.nan,
        "W8最大浮盈均值%": numeric(selected, "Entry_W8_MFE_Net_pct").mean(),
        "每周至少2只A或S比例%": (
            weekly_as.ge(2).sum() / signal_weeks * 100 if signal_weeks else np.nan),
    }


def _exact_random_top3_metrics(pool: pd.DataFrame) -> dict[str, Any]:
    selected_count = 0
    expected_s = expected_as = expected_b = expected_mfe = 0.0
    expected_two_as_weeks = 0.0
    signal_weeks = int(pool["Signal_Week"].nunique())
    for _, group in pool.groupby("Signal_Week"):
        n = len(group)
        k = min(3, n)
        if n <= 0:
            continue
        selected_count += k
        classes = group["Explosion_Class_W8"].astype(str)
        expected_s += k * classes.eq("S").mean()
        expected_as += k * classes.isin(["A", "S"]).mean()
        expected_b += k * classes.isin(["B", "A", "S"]).mean()
        expected_mfe += k * numeric(group, "Entry_W8_MFE_Net_pct").mean()
        successes = int(classes.isin(["A", "S"]).sum())
        denominator = math.comb(n, k)
        probability = 0.0
        for count in range(2, k + 1):
            if count <= successes and k - count <= n - successes:
                probability += (
                    math.comb(successes, count)
                    * math.comb(n - successes, k - count) / denominator)
        expected_two_as_weeks += probability
    denominator = max(selected_count, 1)
    return {
        "事件数": selected_count,
        "覆盖信号周": signal_weeks,
        "S级比例%": expected_s / denominator * 100,
        "A或S比例%": expected_as / denominator * 100,
        "B级以上比例%": expected_b / denominator * 100,
        "W8最大浮盈均值%": expected_mfe / denominator,
        "每周至少2只A或S比例%": (
            expected_two_as_weeks / signal_weeks * 100 if signal_weeks else np.nan),
    }


def ltr_oos_comparison(eligible: pd.DataFrame) -> pd.DataFrame:
    primary = eligible[
        eligible["SKDJ_N"].eq(LTR_PRIMARY_N)
        & true_mask(eligible, "LTR_OOS_Available")].copy()
    if primary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    period_masks = [
        ("全部OOS", pd.Series(True, index=primary.index)),
        ("前段OOS", primary["Validation_Period"].astype(str).eq("前段观察")),
        ("后段OOS", primary["Validation_Period"].astype(str).eq("后段冻结检验")),
    ]
    for period_name, mask in period_masks:
        pool = primary[mask].copy()
        if pool.empty:
            continue
        schemes = [
            ("精确随机Top3期望", None),
            ("原冻结100分Top3", "Top3"),
            ("V5.7双因子H2 Top3", "V57_DualH2_Top3"),
            ("V6.0成对LTR Top3", "LTR_OOS_Top3"),
            ("V6.0成对LTR前20%", "LTR_OOS_Top20Pct"),
        ]
        for scheme, flag in schemes:
            metrics = (_exact_random_top3_metrics(pool) if flag is None
                       else _top3_metrics(pool, pool[true_mask(pool, flag)]))
            rows.append({"时间分段": period_name, "方案": scheme, **metrics})
    return pd.DataFrame(rows)


def ltr_weekly_detail(eligible: pd.DataFrame) -> pd.DataFrame:
    primary = eligible[
        eligible["SKDJ_N"].eq(LTR_PRIMARY_N)
        & true_mask(eligible, "LTR_OOS_Available")].copy()
    rows = []
    for week, group in primary.groupby("Signal_Week", sort=True):
        selected = group[true_mask(group, "LTR_OOS_Top3")].sort_values(
            "LTR_OOS_Weekly_Rank")
        classes = selected["Explosion_Class_W8"].astype(str)
        rows.append({
            "Signal_Week": week,
            "Signal_Date": str(group["Signal_Date"].astype(str).max()),
            "候选数": len(group),
            "训练事件": int(numeric(group, "LTR_OOS_Train_Rows").max()),
            "训练周": int(numeric(group, "LTR_OOS_Train_Weeks").max()),
            "前三股票": "、".join(selected["name"].astype(str).tolist()),
            "前三爆发等级": "/".join(classes.tolist()),
            "前三A或S数量": int(classes.isin(["A", "S"]).sum()),
            "前三S数量": int(classes.eq("S").sum()),
            "前三W8最大浮盈均值%": numeric(
                selected, "Entry_W8_MFE_Net_pct").mean(),
        })
    return pd.DataFrame(rows)


def ltr_feature_coefficients(bundle: dict[str, Any] | None) -> pd.DataFrame:
    if bundle is None:
        return pd.DataFrame(columns=["特征", "周内百分位系数", "模型方向"])
    coefficients = np.asarray(bundle["coefficients"], dtype=float)
    rows = []
    for source, transformed, coefficient in zip(
            LTR_FEATURES, bundle["feature_columns"], coefficients):
        rows.append({
            "特征": source, "模型字段": transformed,
            "周内百分位系数": float(coefficient),
            "模型方向": "数值较高有利" if coefficient > 0 else "数值较低有利",
            "绝对影响": abs(float(coefficient)),
            "历史训练事件": int(bundle["train_rows"]),
            "历史训练周": int(bundle["train_weeks"]),
            "训练成对样本": int(bundle["pair_count"]),
        })
    return pd.DataFrame(rows).sort_values("绝对影响", ascending=False).reset_index(drop=True)


def slim_ltr_candidates(eligible: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ts_code", "name", "SKDJ_N", "Signal_Date", "Signal_Week",
        "SW_L1", "SW_L2", "Raw_Close", "Circ_MV_Billion",
        "Signal_K", "Signal_D", "Signal_KD_Spread",
        "Signal_Prior_Below25_Streak", "Signal_Volume_Ratio_5W",
        "Signal_Week_Return_pct", "Signal_Return_4W_pct",
        "Signal_Return_12W_pct", "Signal_Relative_Industry_12W_pct",
        "Breadth_MA20_Rising_Pct", "Industry_Resonance_Pct",
        "Signal_MA20_Slope_4W_pct", "Signal_K_Thrust_per_AbsWeekReturn",
        "Swing52_Close_to_High_pct", "Signal_VCP_Range_4W_vs_12W",
        "Signal_Prior_GC_Reached75_Count_Last3",
        "Score_Total_100", "Weekly_Rank", "V57_DualH2_Weekly_Rank",
        "LTR_OOS_Score", "LTR_OOS_Weekly_Rank", "LTR_OOS_Top3",
        "LTR_OOS_Top20Pct", "LTR_OOS_Available",
        "LTR_OOS_Train_Rows", "LTR_OOS_Train_Weeks",
        "Explosion_Class_W8", "Explosion_Grade_W8",
        "Entry_W8_MFE_Net_pct", "Entry_W8_MAE_Raw_pct",
        "Entry_W8_Close_Return_Net_pct", "Validation_Period",
    ]
    return eligible[[column for column in columns if column in eligible.columns]].copy()


# ---------------------------------------------------------------------------
# V6.1 timing audit.  These definitions intentionally override the V6.0 stock
# analyzer below while leaving the proven data/cache infrastructure untouched.
# ---------------------------------------------------------------------------

def add_daily_macd(daily: pd.DataFrame) -> pd.DataFrame:
    """Add observable daily MACD-cycle features without looking ahead."""
    work = normalize_price_frame(daily).sort_values("trade_date").reset_index(drop=True)
    close = pd.to_numeric(work["close"], errors="coerce")
    dif = close.ewm(span=12, adjust=False, min_periods=1).mean() - close.ewm(
        span=26, adjust=False, min_periods=1).mean()
    dea = dif.ewm(span=9, adjust=False, min_periods=1).mean()
    hist = dif - dea
    positive = hist.gt(0)
    cycle_start = positive & ~positive.shift(1, fill_value=False)
    cycle_number = cycle_start.cumsum().where(positive, 0).astype(int)
    red_age = pd.Series(0, index=work.index, dtype=int)
    positive_index = work.index[positive]
    if len(positive_index):
        red_age.loc[positive_index] = (
            work.loc[positive_index].groupby(cycle_number.loc[positive_index]).cumcount() + 1
        ).astype(int)
    cycle_peak = pd.Series(np.nan, index=work.index, dtype=float)
    cycle_start_close = pd.Series(np.nan, index=work.index, dtype=float)
    if len(positive_index):
        positive_cycles = cycle_number.loc[positive_index]
        cycle_peak.loc[positive_index] = hist.loc[positive_index].groupby(
            positive_cycles).cummax()
        cycle_start_close.loc[positive_index] = close.loc[positive_index].groupby(
            positive_cycles).transform("first")
    previous_hist = hist.shift(1)
    remaining = hist / cycle_peak.replace(0, np.nan) * 100.0
    retention = (hist / previous_hist.replace(0, np.nan) * 100.0).where(
        positive & previous_hist.gt(0))

    state = pd.Series("已翻绿", index=work.index, dtype=object)
    state.loc[positive & red_age.eq(1)] = "红柱首日"
    state.loc[positive & retention.ge(100.0)] = "红柱扩张"
    state.loc[
        positive & retention.lt(100.0) & remaining.le(20.0)
    ] = "临近翻绿"
    state.loc[
        positive & retention.lt(100.0)
        & remaining.gt(20.0)
        & remaining.ge(MACD_HEALTHY_REMAINING_PCT)
        & retention.ge(MACD_HEALTHY_RETENTION_PCT)
    ] = "健康缩短"
    state.loc[
        positive & retention.lt(100.0)
        & ~state.isin(["临近翻绿", "健康缩短"])
    ] = "明显衰减"

    work["Daily_DIF"] = dif
    work["Daily_DEA"] = dea
    work["Daily_MACD_Hist"] = hist
    work["Daily_MACD_Positive"] = positive
    work["Daily_MACD_Cycle"] = cycle_number
    work["Daily_MACD_Red_Age"] = red_age
    work["Daily_MACD_Cycle_Peak"] = cycle_peak
    work["Daily_MACD_Remaining_pct"] = remaining
    work["Daily_MACD_Retention_pct"] = retention
    work["Daily_MACD_State"] = state
    work["Daily_MACD_Cycle_Start_Close"] = cycle_start_close
    work["Daily_Return_Since_Red_Start_pct"] = (
        close / cycle_start_close.replace(0, np.nan) - 1.0) * 100.0
    return work


def attach_latest_completed_weekly(
        daily_macd: pd.DataFrame, indicator: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest complete N=6 weekly row known at each daily close."""
    daily_work = daily_macd.copy()
    daily_work["_dt"] = pd.to_datetime(daily_work["trade_date"], format="%Y%m%d")
    weekly = indicator.copy()
    weekly["_dt"] = pd.to_datetime(weekly["trade_date"], format="%Y%m%d")
    weekly = weekly[[
        "_dt", "trade_date", "K", "D", "K_Change_1W", "KD_Spread",
        "Prior_Below25_Streak", "Close_to_MA20_pct", "MA20_Slope_4W_pct",
        "Volume_Ratio_5W", "Signal_Week_Return_pct",
    ]].rename(columns={
        "trade_date": "Setup_Weekly_Date",
        "K": "Setup_Weekly_K", "D": "Setup_Weekly_D",
        "K_Change_1W": "Setup_Weekly_K_Change_1W",
        "KD_Spread": "Setup_Weekly_KD_Spread",
        "Prior_Below25_Streak": "Setup_Prior_Below25_Streak",
        "Close_to_MA20_pct": "Setup_Close_to_MA20_pct",
        "MA20_Slope_4W_pct": "Setup_MA20_Slope_4W_pct",
        "Volume_Ratio_5W": "Setup_Volume_Ratio_5W",
        "Signal_Week_Return_pct": "Setup_Week_Return_pct",
    }).sort_values("_dt")
    result = pd.merge_asof(
        daily_work.sort_values("_dt"), weekly, on="_dt", direction="backward")
    return result.drop(columns=["_dt"]).sort_values("trade_date").reset_index(drop=True)


def macd_snapshot(row: pd.Series) -> dict[str, Any]:
    cycle_value = finite_num(row.get("Daily_MACD_Cycle"))
    age_value = finite_num(row.get("Daily_MACD_Red_Age"))
    return {
        "Daily_DIF": finite_num(row.get("Daily_DIF")),
        "Daily_DEA": finite_num(row.get("Daily_DEA")),
        "Daily_MACD_Hist": finite_num(row.get("Daily_MACD_Hist")),
        "Daily_MACD_Positive": to_bool(row.get("Daily_MACD_Positive")),
        "Daily_MACD_Cycle": int(cycle_value) if math.isfinite(cycle_value) else 0,
        "Daily_MACD_Red_Age": int(age_value) if math.isfinite(age_value) else 0,
        "Daily_MACD_Remaining_pct": finite_num(row.get("Daily_MACD_Remaining_pct")),
        "Daily_MACD_Retention_pct": finite_num(row.get("Daily_MACD_Retention_pct")),
        "Daily_MACD_State": str(row.get("Daily_MACD_State", "数据不足")),
        "Daily_Return_Since_Red_Start_pct": finite_num(
            row.get("Daily_Return_Since_Red_Start_pct")),
    }


def _cost_factors(config: dict[str, Any]) -> tuple[float, float]:
    buy_cost = (config["commission_pct"] + config["transfer_fee_pct"]) / 100.0
    sell_cost = (
        config["commission_pct"] + config["transfer_fee_pct"]
        + config["stamp_duty_pct"]) / 100.0
    buy_factor = (1 + config["buy_slippage_pct"] / 100.0) * (1 + buy_cost)
    sell_factor = (1 - config["sell_slippage_pct"] / 100.0) * (1 - sell_cost)
    return buy_factor, sell_factor


def _next_stock_open(
        daily: pd.DataFrame, after_date: str, latest_date: str
        ) -> tuple[str, float]:
    future = daily[
        daily["trade_date"].astype(str).gt(after_date)
        & daily["trade_date"].astype(str).le(latest_date)
    ].sort_values("trade_date")
    if future.empty:
        return "", np.nan
    row = future.iloc[0]
    return str(row["trade_date"]), finite_num(row.get("open"))


def daily_timing_outcomes(
        daily_macd: pd.DataFrame, signal_date: str, ts_code: str,
        open_dates: list[str], open_pos: dict[str, int], config: dict[str, Any]
        ) -> dict[str, Any]:
    """Evaluate identical 40-session outcomes and observable MACD exits."""
    out: dict[str, Any] = {
        "Tradable": False, "Reason": "", "Date": "", "Raw_Open": np.nan,
        "Has_40D": False, "End_Date_40D": "", "MFE_Net_pct": np.nan,
        "MAE_Raw_pct": np.nan, "Close_Return_Net_pct": np.nan,
        "Peak_Date_40D": "", "Peak_Market_Day_40D": np.nan,
    }
    for level in (10, 20, 30):
        out[f"First_Hit_{level}_vs_Minus10_40D"] = ""
        out[f"First_Hit_{level}_vs_Minus10_40D_Date"] = ""
    for threshold in MACD_REMAINING_THRESHOLDS:
        prefix = f"Exit_Remaining{int(threshold)}"
        out.update({
            f"{prefix}_Date": "", f"{prefix}_Reason": "",
            f"{prefix}_Return_Net_pct": np.nan,
            f"{prefix}_Hold_Market_Days": np.nan,
            f"{prefix}_MAE_Raw_pct": np.nan,
        })
    if signal_date not in open_pos or open_pos[signal_date] + 1 >= len(open_dates):
        out["Reason"] = "未来市场交易日不足"
        return out
    entry_pos = open_pos[signal_date] + 1
    entry_date = open_dates[entry_pos]
    entry_row = daily_macd[daily_macd["trade_date"].astype(str).eq(entry_date)]
    if entry_row.empty:
        out["Reason"] = "下一市场交易日停牌"
        return out
    first = entry_row.iloc[-1]
    raw_entry = finite_num(first.get("open"))
    if not math.isfinite(raw_entry) or raw_entry <= 0:
        out["Reason"] = "开盘价无效"
        return out
    if str(ts_code).startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        if float(first["open"]) == float(first["high"]) == float(first["low"]):
            out["Reason"] = "主板下一交易日一字板"
            return out
    out.update({"Tradable": True, "Date": entry_date, "Raw_Open": raw_entry})
    if entry_pos + DAILY_AUDIT_DAYS - 1 >= len(open_dates):
        out["Reason"] = "未来40个市场交易日不足"
        return out
    end_date = open_dates[entry_pos + DAILY_AUDIT_DAYS - 1]
    path = daily_macd[
        daily_macd["trade_date"].astype(str).between(entry_date, end_date)
    ].sort_values("trade_date").copy()
    if path.empty:
        out["Reason"] = "未来40日个股行情缺失"
        return out
    buy_factor, sell_factor = _cost_factors(config)
    net_entry = raw_entry * buy_factor
    high = finite_num(pd.to_numeric(path["high"], errors="coerce").max())
    low = finite_num(pd.to_numeric(path["low"], errors="coerce").min())
    close = finite_num(path.iloc[-1].get("close"))
    highs = pd.to_numeric(path["high"], errors="coerce")
    peak_date = ""
    peak_day = np.nan
    if highs.notna().any():
        peak_index = highs.idxmax()
        peak_date = str(path.loc[peak_index, "trade_date"])
        peak_day = float(open_pos.get(peak_date, entry_pos) - entry_pos + 1)
    out.update({
        "Has_40D": True, "End_Date_40D": end_date,
        "MFE_Net_pct": (high * sell_factor / net_entry - 1.0) * 100.0,
        "MAE_Raw_pct": (low / raw_entry - 1.0) * 100.0,
        "Close_Return_Net_pct": (close * sell_factor / net_entry - 1.0) * 100.0,
        "Peak_Date_40D": peak_date, "Peak_Market_Day_40D": peak_day,
    })
    entry_period = pd.Timestamp(entry_date).to_period("W-FRI")
    for level in (10, 20, 30):
        label, hit_date, _ = first_hit_detail(path, raw_entry, float(level), entry_period)
        out[f"First_Hit_{level}_vs_Minus10_40D"] = label
        out[f"First_Hit_{level}_vs_Minus10_40D_Date"] = hit_date

    # A close signal can only be acted on at a later open.  If the stock is
    # suspended, use its first later available open, still inside the horizon.
    decision_path = path[path["trade_date"].astype(str).ge(entry_date)].copy()
    for threshold in MACD_REMAINING_THRESHOLDS:
        prefix = f"Exit_Remaining{int(threshold)}"
        hist = numeric(decision_path, "Daily_MACD_Hist")
        remaining = numeric(decision_path, "Daily_MACD_Remaining_pct")
        retention = numeric(decision_path, "Daily_MACD_Retention_pct")
        green = hist.le(0)
        near = hist.gt(0) & remaining.le(threshold) & retention.lt(100.0)
        decisions = decision_path[green | near]
        if decisions.empty:
            exit_date, raw_exit, reason = end_date, close, "40日仍未触发_期末收盘"
        else:
            decision = decisions.iloc[0]
            decision_date = str(decision["trade_date"])
            reason = "红柱翻绿" if finite_num(decision.get("Daily_MACD_Hist")) <= 0 else f"红柱剩余≤{int(threshold)}%"
            exit_date, raw_exit = _next_stock_open(daily_macd, decision_date, end_date)
            if not exit_date or not math.isfinite(raw_exit):
                exit_date, raw_exit, reason = end_date, close, f"{reason}_无后续开盘_期末收盘"
        exit_path = path[path["trade_date"].astype(str).between(entry_date, exit_date)]
        exit_low = finite_num(pd.to_numeric(exit_path["low"], errors="coerce").min())
        out.update({
            f"{prefix}_Date": exit_date,
            f"{prefix}_Reason": reason,
            f"{prefix}_Return_Net_pct": (raw_exit * sell_factor / net_entry - 1.0) * 100.0,
            f"{prefix}_Hold_Market_Days": float(
                open_pos.get(exit_date, entry_pos) - entry_pos + 1),
            f"{prefix}_MAE_Raw_pct": (exit_low / raw_entry - 1.0) * 100.0,
        })
    return out


def outcome_explosion_class(row: pd.Series) -> str:
    if not to_bool(row.get("Entry_Has_40D")):
        return "未成熟"
    if str(row.get("Entry_First_Hit_30_vs_Minus10_40D", "")) == "先到+30%":
        return "S"
    if str(row.get("Entry_First_Hit_20_vs_Minus10_40D", "")) == "先到+20%":
        return "A"
    if str(row.get("Entry_First_Hit_10_vs_Minus10_40D", "")) == "先到+10%":
        return "B"
    return "F"


def _event_base_row(
        stock: pd.Series, membership: dict[str, str], signal_date: str,
        snapshot: dict[str, float], event_type: str) -> dict[str, Any]:
    return {
        "Event_Type": event_type, "ts_code": str(stock["ts_code"]),
        "name": str(stock["name"]), "SKDJ_N": EARLY_PRIMARY_N,
        "SKDJ_M": SKDJ_M, "Signal_Date": signal_date,
        "Signal_Week": str(pd.Timestamp(signal_date).to_period("W-FRI")),
        "SW_L1": membership["l1"], "SW_L2": membership["l2"],
        "SW_L3": membership["l3"], **snapshot,
    }


def analyze_stock(stock: pd.Series, periods: list[dict[str, str]], daily: pd.DataFrame,
                  cached_basic: pd.DataFrame, storage_path: str,
                  week_last_map: dict[pd.Timestamp, str], open_dates: list[str],
                  open_pos: dict[str, int], market_weeks: list[tuple[pd.Period, str]],
                  config: dict[str, Any], use_cache: bool, api_pause: float
                  ) -> tuple[list[dict[str, Any]], dict[str, int],
                             dict[str, dict[str, float]]]:
    """Build independent early events plus late weekly-cross controls."""
    del market_weeks  # V6.1 uses exact market-session horizons, not partial weeks.
    rejects: dict[str, int] = {}
    weekly_base = aggregate_complete_weekly(daily, week_last_map)
    if weekly_base.empty:
        return [], rejects, {}
    n6 = add_skdj(weekly_base, EARLY_PRIMARY_N)
    daily_macd = attach_latest_completed_weekly(add_daily_macd(daily), n6)
    event_end = str(config.get("event_signal_end", config["signal_end"]))

    early_mask = (
        daily_macd["trade_date"].astype(str).between(config["signal_start"], event_end)
        & numeric(daily_macd, "Setup_Weekly_K").between(
            EARLY_WEEKLY_K_MIN, EARLY_WEEKLY_K_MAX, inclusive="both")
        & numeric(daily_macd, "Daily_MACD_Red_Age").between(
            EARLY_RED_AGE_MIN, EARLY_RED_AGE_MAX, inclusive="both")
        & true_mask(daily_macd, "Daily_MACD_Positive")
    )
    early_candidates = daily_macd[early_mask].copy()
    if not early_candidates.empty:
        # One buy candidate per positive MACD cycle; if weekly K enters the
        # setup range on day 4, day 4 is selected rather than inventing day 2.
        early_candidates = early_candidates.sort_values("trade_date").groupby(
            "Daily_MACD_Cycle", as_index=False, sort=False).first()
    weekly_crosses = n6[
        true_mask(n6, "K_Cross_25")
        & n6["trade_date"].astype(str).between(config["signal_start"], event_end)
    ].copy()
    if early_candidates.empty and weekly_crosses.empty:
        return [], rejects, {}

    code = str(stock["ts_code"])
    basic = ensure_daily_basic(
        code, config["data_start"], config["market_end"], daily, cached_basic,
        storage_path, use_cache, api_pause)
    if basic.empty:
        rejects["存在时序信号但daily_basic缺失"] = len(early_candidates) + len(weekly_crosses)
        return [], rejects, {}

    cross_dates = weekly_crosses["trade_date"].astype(str).sort_values().tolist()
    rows: list[dict[str, Any]] = []

    def valid_context(signal_date: str) -> tuple[dict[str, str] | None, dict[str, float], str]:
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
        return membership, snapshot, reason

    for _, signal in early_candidates.iterrows():
        signal_date = str(signal["trade_date"])
        membership, snapshot, reason = valid_context(signal_date)
        if reason or membership is None:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        outcome = daily_timing_outcomes(
            daily_macd, signal_date, code, open_dates, open_pos, config)
        future_cross = ""
        for cross_date in cross_dates:
            if cross_date >= signal_date and (
                    pd.Timestamp(cross_date) - pd.Timestamp(signal_date)).days <= FUTURE_WEEKLY_CROSS_DAYS:
                future_cross = cross_date
                break
        future_open_date = ""
        future_open = np.nan
        if future_cross:
            future_open_date, future_open = _next_stock_open(
                daily_macd, future_cross, config["market_end"])
        early_open = finite_num(outcome.get("Raw_Open"))
        price_advantage = (
            (future_open / early_open - 1.0) * 100.0
            if math.isfinite(early_open) and early_open > 0 and math.isfinite(future_open)
            else np.nan)
        row = _event_base_row(
            stock, membership, signal_date, snapshot, "EARLY_DAILY_MACD")
        row.update({
            "Setup_Weekly_Date": str(signal.get("Setup_Weekly_Date", "")),
            "Signal_K": finite_num(signal.get("Setup_Weekly_K")),
            "Signal_D": finite_num(signal.get("Setup_Weekly_D")),
            "Signal_KD_Spread": finite_num(signal.get("Setup_Weekly_KD_Spread")),
            "Signal_K_Change_1W": finite_num(signal.get("Setup_Weekly_K_Change_1W")),
            "Signal_Prior_Below25_Streak": finite_num(
                signal.get("Setup_Prior_Below25_Streak")),
            "Signal_Close_to_MA20_pct": finite_num(
                signal.get("Setup_Close_to_MA20_pct")),
            "Signal_MA20_Slope_4W_pct": finite_num(
                signal.get("Setup_MA20_Slope_4W_pct")),
            "Signal_Volume_Ratio_5W": finite_num(
                signal.get("Setup_Volume_Ratio_5W")),
            "Signal_Week_Return_pct": finite_num(
                signal.get("Setup_Week_Return_pct")),
            **macd_snapshot(signal),
            "Future_Weekly_Cross25_Within42D": bool(future_cross),
            "Future_Weekly_Cross25_Date": future_cross,
            "Lead_Calendar_Days_to_Weekly_Cross": (
                float((pd.Timestamp(future_cross) - pd.Timestamp(signal_date)).days)
                if future_cross else np.nan),
            "Future_Weekly_Entry_Date": future_open_date,
            "Future_Weekly_Raw_Open": future_open,
            "Early_Price_Advantage_vs_Weekly_pct": price_advantage,
        })
        row.update({f"Entry_{key}": value for key, value in outcome.items()})
        rows.append(row)

    macd_lookup = daily_macd.set_index(daily_macd["trade_date"].astype(str))
    for _, signal in weekly_crosses.iterrows():
        signal_date = str(signal["trade_date"])
        membership, snapshot, reason = valid_context(signal_date)
        if reason or membership is None:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        if signal_date in macd_lookup.index:
            daily_signal = macd_lookup.loc[signal_date]
            if isinstance(daily_signal, pd.DataFrame):
                daily_signal = daily_signal.iloc[-1]
        else:
            before = daily_macd[daily_macd["trade_date"].astype(str).le(signal_date)]
            daily_signal = before.iloc[-1] if not before.empty else pd.Series(dtype=object)
        outcome = daily_timing_outcomes(
            daily_macd, signal_date, code, open_dates, open_pos, config)
        row = _event_base_row(
            stock, membership, signal_date, snapshot, "WEEKLY_CROSS25")
        row.update({
            "Setup_Weekly_Date": signal_date,
            "Signal_K": finite_num(signal.get("K")),
            "Signal_D": finite_num(signal.get("D")),
            "Signal_KD_Spread": finite_num(signal.get("KD_Spread")),
            "Signal_K_Change_1W": finite_num(signal.get("K_Change_1W")),
            "Signal_Prior_Below25_Streak": finite_num(
                signal.get("Prior_Below25_Streak")),
            "Signal_Close_to_MA20_pct": finite_num(signal.get("Close_to_MA20_pct")),
            "Signal_MA20_Slope_4W_pct": finite_num(signal.get("MA20_Slope_4W_pct")),
            "Signal_Volume_Ratio_5W": finite_num(signal.get("Volume_Ratio_5W")),
            "Signal_Week_Return_pct": finite_num(signal.get("Signal_Week_Return_pct")),
            **macd_snapshot(daily_signal),
        })
        row.update({f"Entry_{key}": value for key, value in outcome.items()})
        rows.append(row)
    return rows, rejects, {}


def add_timing_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        result["Explosion_Class_40D"] = pd.Series(dtype=str)
        return result
    result["Explosion_Class_40D"] = result.apply(outcome_explosion_class, axis=1)
    result["Explosion_Grade_40D"] = result["Explosion_Class_40D"].map(
        {"F": 0, "B": 1, "A": 2, "S": 3})
    return result


def timing_outcome_summary(
        frame: pd.DataFrame, group_columns: list[str], scheme: str = ""
        ) -> pd.DataFrame:
    columns = group_columns if group_columns else ["_all"]
    work = frame.copy()
    if not group_columns:
        work["_all"] = "全部"
    rows: list[dict[str, Any]] = []
    grouped = work.groupby(columns[0] if len(columns) == 1 else columns, dropna=False)
    for key, group in grouped:
        keys = (key,) if len(columns) == 1 else tuple(key)
        classes = group["Explosion_Class_40D"].astype(str)
        row = {column: value for column, value in zip(columns, keys)}
        row.update({
            "方案": scheme,
            "事件数": len(group), "不同股票": group["ts_code"].nunique(),
            "信号日": group["Signal_Date"].nunique(),
            "信号周": group["Signal_Week"].nunique(),
            "S级比例%": classes.eq("S").mean() * 100.0,
            "A或S比例%": classes.isin(["A", "S"]).mean() * 100.0,
            "B级以上比例%": classes.isin(["B", "A", "S"]).mean() * 100.0,
            "F级比例%": classes.eq("F").mean() * 100.0,
            "40日最大浮盈均值%": numeric(group, "Entry_MFE_Net_pct").mean(),
            "40日最大浮盈中位%": numeric(group, "Entry_MFE_Net_pct").median(),
            "40日最大回撤均值%": numeric(group, "Entry_MAE_Raw_pct").mean(),
            "40日期末净收益均值%": numeric(group, "Entry_Close_Return_Net_pct").mean(),
            "40日期末净收益中位%": numeric(group, "Entry_Close_Return_Net_pct").median(),
            "40日期末胜率%": numeric(group, "Entry_Close_Return_Net_pct").gt(0).mean() * 100.0,
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.drop(columns=["_all"], errors="ignore")


def timing_calendar(
        open_dates: list[str], start_date: str, end_date: str,
        early: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    days = [value for value in open_dates if start_date <= value <= end_date]
    calendar = pd.DataFrame({"trade_date": days})
    calendar["Signal_Week"] = pd.to_datetime(
        calendar["trade_date"], format="%Y%m%d").dt.to_period("W-FRI").astype(str)
    weeks = calendar.groupby("Signal_Week")["trade_date"].max().rename(
        "Week_Last_Trading_Date").reset_index()
    for name, source in (("提前日线信号", early), ("周线确认信号", weekly)):
        counts = source.groupby("Signal_Week").size() if not source.empty else pd.Series(dtype=int)
        weeks[name] = weeks["Signal_Week"].map(counts).fillna(0).astype(int)
    return weeks


def early_exit_strategy_audit(early: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if early.empty:
        return pd.DataFrame()
    rows.append({
        "退出方案": "固定40个市场交易日期末收盘", "事件数": len(early),
        "净收益均值%": numeric(early, "Entry_Close_Return_Net_pct").mean(),
        "净收益中位%": numeric(early, "Entry_Close_Return_Net_pct").median(),
        "盈利比例%": numeric(early, "Entry_Close_Return_Net_pct").gt(0).mean() * 100.0,
        "持有市场日均值": 40.0,
        "持有期最大回撤均值%": numeric(early, "Entry_MAE_Raw_pct").mean(),
    })
    for threshold in MACD_REMAINING_THRESHOLDS:
        prefix = f"Entry_Exit_Remaining{int(threshold)}"
        returns = numeric(early, f"{prefix}_Return_Net_pct")
        rows.append({
            "退出方案": f"红柱剩余≤{int(threshold)}%或翻绿，次日开盘",
            "事件数": int(returns.notna().sum()),
            "净收益均值%": returns.mean(), "净收益中位%": returns.median(),
            "盈利比例%": returns.gt(0).mean() * 100.0,
            "持有市场日均值": numeric(early, f"{prefix}_Hold_Market_Days").mean(),
            "持有期最大回撤均值%": numeric(early, f"{prefix}_MAE_Raw_pct").mean(),
        })
    return pd.DataFrame(rows)


def legacy_main_v60() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    streamlit_version = str(getattr(st, "__version__", "unknown"))
    st.caption(
        f"{UI_PATCH}｜历史W8判卷与最近完整周候选分离；N=6为主模型。"
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
- **历史时间线**：默认回测500个交易日，买入信号严格截止2026-06-05；只有次周可成交且已经走完W8的事件参与训练、排名验收和收益统计。
- **实盘时间线**：行情读取到侧边栏“实盘观察截止”；只显示最近一个已经完整结束的市场周候选，未成熟事件绝不进入历史成绩。
- **预热用途**：开始前{WARMUP_WEEKS}周只用于形成指标、历史金叉和52周状态，不增加正式历史信号。
- **过滤**：每个历史信号日分别检查当时科技行业归属；最低股价默认10元、最低流通市值默认50亿元，侧边栏可切换，避免使用今天状态回看历史。
- **唯一共同硬条件**：信号前连续处于25下方1～{MAX_BOTTOM_STREAK}周；不再把52周活跃度、当周候选数量或板块状态升级为硬过滤。
- **爆发等级**：S级=先到+30%，A级=先到+20%但未到S，B级=先到+10%但未到A/S，其余为F；全部与-10%比较且同日冲突按-10%先。
- **LTR排序**：每个信号周是一个独立组，只比较同周股票；F/B/A/S按0/1/2/3并用0/1/3/7收益差训练成对逻辑模型。
- **时间外预测**：预测某一历史周时，训练集只允许包含W8结束日早于该周信号日的事件；至少{LTR_MIN_TRAIN_WEEKS}个成熟历史周才出LTR名次。
- **主验收**：同一批OOS周比较LTR、精确随机Top3、原100分和V5.7双因子H2；看S、A/S、B以上、W8最大浮盈和每周至少两只A/S。
- **防过拟合**：只使用15项预先声明的信号时点技术特征、L2正则、每周等权和固定成对抽样；不因结果不佳增加硬条件。
- **实盘排名**：N=6使用截至历史信号截止日的全部成熟W8样本训练；N=7/9只显示参数共振与旧排名，不混成独立训练样本。
- **本版边界**：不研究止损止盈、不要求W1～W12平滑、不把实盘候选的未来走势写回本次历史模型。
""")
    with st.sidebar:
        st.header("运行参数")
        backtest_days = st.number_input(
            "回测交易日数", 100, 1000, 500, 50, key="v60_days")
        st.caption("默认500日；只决定2026-06-05以前的历史信号起点。")
        # 沿用本页已经稳定加载的number_input，避免部分iOS/Safari会话在
        # 首次加载selectbox前端分块时出现“Importing a module script failed”。
        min_price = st.number_input(
            "最低股价（元）", min_value=10.0, max_value=20.0,
            value=10.0, step=10.0, format="%.0f", key="v60_min_price")
        min_mv = st.number_input(
            "最低流通市值（亿元）", min_value=50.0, max_value=100.0,
            value=50.0, step=50.0, format="%.0f", key="v60_min_mv")
        st.caption(f"本次历史过滤：股价≥{min_price:.0f}元，流通市值≥{min_mv:.0f}亿元")
        signal_end_date = st.date_input(
            "历史买入信号截止（W8判卷）", date(2026, 6, 5),
            key="v60_signal_end")
        market_end_date = st.date_input(
            "实盘候选观察截止（默认今天）", date.today(),
            key="v60_market_end")
        st.caption(
            "历史成绩只接收信号日≤历史截止且W8成熟的事件；"
            "观察截止仅用于寻找最近完整周候选。")
        split_ratio_pct = st.number_input(
            "前段观察占正式周比例(%)", 50, 80, 60, 5, key="v60_split")
        pause = st.number_input(
            "接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v60_pause")
        use_cache = st.checkbox("复用行情缓存", True, key="v60_cache")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005,
            format="%.3f", key="v60_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01,
            format="%.3f", key="v60_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001,
            format="%.3f", key="v60_transfer")
        if st.button("清除V6.0结果和运行状态", key="v60_clear"):
            shutil.rmtree(RESULT_DIR, ignore_errors=True)
            shutil.rmtree(JOB_DIR, ignore_errors=True)
            st.success("V6.0结果和检查点已清除；通用行情缓存保留")

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
        f"weekly_skdj_pairwise_ltr_live_v6_0_{int(backtest_days)}d_"
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
                saved_result, result_name, f"v60_saved_{request_signature}")
        except Exception as exc:
            st.warning(f"旧结果读取失败：{exc}")

    secret_token = configured_tushare_token()
    if secret_token:
        token = secret_token
        st.caption("已从Streamlit Secrets读取TUSHARE_TOKEN。")
    else:
        token = st.text_input("Tushare Token", type="password", key="v60_token")

    job_active = is_job_active(request_signature)
    left, right = st.columns(2)
    with left:
        start_clicked = st.button("开始/重新运行V6.0", type="primary", key="v60_run")
    with right:
        stop_clicked = st.button("停止自动续跑", disabled=not job_active, key="v60_stop")
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
        # event_signal_end extends event generation for the live observation
        # lane.  Historical training is filtered back to signal_end below.
        "event_signal_end": market_end,
        "data_start": data_start, "market_end": market_end,
        "min_price": float(min_price), "min_mv": float(min_mv),
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct),
        "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct),
    }
    # Live event generation changes the event range, so V6.0 uses its own
    # checkpoints while retaining the common price cache.
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

    open_date_set = set(open_dates)
    completed_week_ends = sorted({
        str(value) for value in week_last_map.values()
        if str(value) <= market_end and str(value) in open_date_set
    })
    if not completed_week_ends:
        st.error("实盘观察截止以前没有可识别的完整市场周")
        return
    live_week_end = completed_week_ends[-1]
    period_index = build_period_index(memberships)
    active_codes = {
        code for code, periods in period_index.items()
        if periods_overlap(periods, signal_start, market_end)
    }
    stocks = stock_basic[stock_basic["ts_code"].isin(active_codes)].copy()
    stocks = stocks[
        ~stocks["list_date"].gt(market_end)
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
    # Strict separation: post-cutoff events may be shown, but only events at or
    # before the historical cutoff can mature into labels or enter training.
    history_events = events_all[
        events_all["Signal_Date"].astype(str).le(signal_end)].copy()
    observation_events = events_all[
        events_all["Signal_Date"].astype(str).gt(signal_end)
        & events_all["Signal_Date"].astype(str).le(live_week_end)].copy()
    if (history_events["Signal_Date"].astype(str).gt(signal_end).any()
            or observation_events["Signal_Date"].astype(str).le(signal_end).any()):
        raise RuntimeError("历史训练与实盘观察时间线分离失败")
    mature_w8 = mature_events(history_events, RANKING_WEEKS)
    mature_w12 = mature_events(history_events, AUDIT_WEEKS)
    if mature_w8.empty:
        st.error("存在信号，但没有未来完整W8的成熟事件。")
        return
    if mature_w8["Signal_Date"].astype(str).gt(signal_end).any():
        raise RuntimeError("W8历史样本混入了历史截止日以后的候选")

    calendar = signal_calendar(open_dates, signal_start, signal_end, history_events)
    split_end, periods = build_periods(calendar, float(split_ratio_pct) / 100.0)
    scored_all, eligible = score_and_rank_events(mature_w8, split_end)
    if eligible.empty:
        st.error("存在成熟事件，但没有股票通过‘连续处于25下方不超过5周’硬条件。")
        return
    if eligible["Signal_Date"].astype(str).gt(signal_end).any():
        raise RuntimeError("LTR历史候选混入了实盘观察事件")
    eligible = add_independent_candidate_features(eligible)
    eligible = add_industry_breadth_features(eligible, breadth_frame)
    eligible = add_challenger_rankings(eligible)
    eligible = add_v57_explosion_rankings(eligible)
    eligible = add_explosion_labels(eligible)
    eligible = add_walk_forward_ltr(eligible)
    live_model, _ = fit_full_history_ltr(eligible)

    # Score every post-cutoff observation week with the frozen full-history
    # model, then display the exact latest completed market week.  No row here
    # is passed to add_explosion_labels or any historical metric function.
    live_scored_all = pd.DataFrame()
    live_ranked_all = pd.DataFrame()
    if not observation_events.empty:
        live_scored_all, live_ranked_all = score_and_rank_events(
            observation_events, split_end)
        if not live_ranked_all.empty:
            live_ranked_all = add_independent_candidate_features(live_ranked_all)
            live_ranked_all = add_industry_breadth_features(
                live_ranked_all, breadth_frame)
            live_ranked_all = add_challenger_rankings(live_ranked_all)
            live_ranked_all = add_v57_explosion_rankings(live_ranked_all)
            live_ranked_all = score_live_candidates(
                live_ranked_all, eligible, live_model)
    live_candidates = live_ranked_all[
        live_ranked_all.get(
            "Signal_Date", pd.Series(index=live_ranked_all.index, dtype=str)
        ).astype(str).eq(live_week_end)].copy()
    recent_candidate_week = ""
    recent_reference_candidates = pd.DataFrame()
    if not live_ranked_all.empty:
        recent_candidate_week = str(
            live_ranked_all["Signal_Date"].astype(str).max())
        recent_reference_candidates = live_ranked_all[
            live_ranked_all["Signal_Date"].astype(str).eq(
                recent_candidate_week)].copy()

    feature_status = st.empty()
    with st.spinner("执行V6.0周内LTR时间外验证并生成最近完整周候选..."):
        feature_status.caption("1/6 复现宽候选池、原100分和V5.7对照排名...")
        score_rules = frozen_score_definitions()
        v57_rules = v57_ranking_definitions()
        ltr_rules = ltr_definitions()
        explosion_definitions = explosion_class_definitions()
        feature_status.caption("2/6 生成严格滚动的N=6成对LTR时间外名次...")
        ltr_comparison = ltr_oos_comparison(eligible)
        ltr_week_detail = ltr_weekly_detail(eligible)
        feature_status.caption("3/6 训练截至历史截止日的实盘模型...")
        ltr_coefficients = ltr_feature_coefficients(live_model)
        feature_status.caption("4/6 汇总历史爆发等级和基准排名...")
        explosion_classes = explosion_class_audit(eligible, periods)
        feature_status.caption("5/6 生成历史周历和精简LTR候选明细...")
        ranked_calendar = weekly_rank_calendar(
            calendar, scored_all, eligible, split_end)
        slim_candidates = slim_ltr_candidates(eligible)
        feature_status.caption("6/6 整理最近完整周实盘观察表...")
        live_columns = [
            "ts_code", "name", "SKDJ_N", "Signal_Date", "Signal_Week",
            "SW_L1", "SW_L2", "Raw_Close", "Circ_MV_Billion",
            "Signal_K", "Signal_D", "Signal_KD_Spread",
            "Signal_Prior_Below25_Streak", "Signal_Volume_Ratio_5W",
            "Signal_Week_Return_pct", "Signal_Return_4W_pct",
            "Signal_Return_12W_pct", "Signal_Relative_Industry_12W_pct",
            "Breadth_MA20_Rising_Pct", "Industry_Resonance_Pct",
            "Swing52_Close_to_High_pct", "Signal_VCP_Range_4W_vs_12W",
            "Signal_Prior_GC_Reached75_Count_Last3",
            "Score_Total_100", "Weekly_Rank", "V57_DualH2_Weekly_Rank",
            "LTR_Live_Score", "LTR_Live_Weekly_Rank", "LTR_Live_Top3",
            "LTR_Live_Top20Pct", "参数共振N", "参数共振数量",
            "历史训练事件", "历史训练周",
        ]
        live_candidates_export = live_candidates[[
            column for column in live_columns if column in live_candidates.columns
        ]].copy()
    feature_status.empty()
    summary_rows = []
    for n in SKDJ_NS:
        all_n = history_events[history_events["SKDJ_N"].eq(n)]
        mature_n = mature_w8[mature_w8["SKDJ_N"].eq(n)]
        mature_w12_n = mature_w12[mature_w12["SKDJ_N"].eq(n)]
        eligible_n = eligible[eligible["SKDJ_N"].eq(n)]
        observation_n = live_ranked_all[
            live_ranked_all.get(
                "SKDJ_N", pd.Series(index=live_ranked_all.index, dtype=float)
            ).eq(n)].copy()
        live_n = live_candidates[
            live_candidates.get(
                "SKDJ_N", pd.Series(index=live_candidates.index, dtype=float)
            ).eq(n)].copy()
        pass_by_date = eligible_n.groupby(eligible_n["Signal_Date"].astype(str)).size()
        pass_counts = calendar["Week_End"].astype(str).map(pass_by_date).fillna(0).astype(int)
        oos_available = true_mask(eligible_n, "LTR_OOS_Available")
        oos_dates = eligible_n.loc[oos_available, "Signal_Date"].astype(str)
        summary_rows.append({
            "SKDJ_N": n, "SKDJ_M": SKDJ_M,
            "历史基础过滤事件": len(all_n), "W8成熟事件": len(mature_n),
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
            "V5.7_双因子H2前3事件": int(true_mask(
                eligible_n, "V57_DualH2_Top3").sum()),
            "LTR可严格OOS事件": int(oos_available.sum()),
            "LTR可严格OOS信号周": int(
                eligible_n.loc[oos_available, "Signal_Week"].nunique()),
            "LTR严格OOS开始": oos_dates.min() if not oos_dates.empty else "",
            "LTR严格OOS前3事件": int(true_mask(
                eligible_n, "LTR_OOS_Top3").sum()),
            "爆发S级事件": int(eligible_n[
                "Explosion_Class_W8"].astype(str).eq("S").sum()),
            "爆发A级事件": int(eligible_n[
                "Explosion_Class_W8"].astype(str).eq("A").sum()),
            "爆发B级事件": int(eligible_n[
                "Explosion_Class_W8"].astype(str).eq("B").sum()),
            "爆发F级事件": int(eligible_n[
                "Explosion_Class_W8"].astype(str).eq("F").sum()),
            "历史截止后观察事件": len(observation_n),
            "最近完整周候选": len(live_n),
            "最近完整周LTR前3": int(true_mask(live_n, "LTR_Live_Top3").sum()),
        })
    run_summary = pd.DataFrame(summary_rows)
    run_summary.insert(0, "程序版本", VERSION)
    run_summary["正式信号开始"] = signal_start
    run_summary["正式信号截止"] = signal_end
    run_summary["最近完整市场周"] = live_week_end
    run_summary["最近有候选观察周"] = recent_candidate_week
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
                "层级": "V6.0唯一共同硬条件", "SKDJ_N": n,
                "剔除原因": reason, "次数": int(count)})
    historical_rejections = [{
        "层级": "历史时点基础过滤", "SKDJ_N": "全部",
        "剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]
    rejection_audit = pd.DataFrame(historical_rejections + hard_rejections)

    metadata = pd.DataFrame([
        ("信号", "上一完整周K<25，本完整周K>=25；不要求低位金叉，不要求K>D"),
        ("重复信号", "以后跌回25下方再上穿时重新计为新事件"),
        ("参数", "同一次运行分别计算N=6、N=7和默认N=9；M固定为3"),
        ("历史数据窗口", f"正式{int(backtest_days)}个交易日；历史信号{signal_start}至{signal_end}；开始前{WARMUP_WEEKS}周预热"),
        ("历史成熟样本", "只有Signal_Date不晚于历史截止、次周可成交且Entry_Has_W8=True的事件才能训练和判卷"),
        ("实盘观察窗口", f"行情读取至{market_end}；最近完整市场周为{live_week_end}；该时间线不进入历史标签和收益"),
        ("历史价格市值过滤", f"信号日股价≥{float(min_price):g}元、流通市值≥{float(min_mv):g}亿元"),
        ("唯一共同硬条件", f"信号前连续处于25下方1～{MAX_BOTTOM_STREAK}周；不增加52周股性或候选数量硬过滤"),
        ("原评分基准", "V5.1.1冻结100分仅作失效基准，不作为V6.0主模型"),
        ("原评分结构", "SKDJ重置35、量能20、周K线结构20、MA20趋势15、同周价格市值百分位10"),
        ("V5.7对照", "双因子H2完整保留为当前最好人工基准，与随机和V6.0在相同OOS周比较"),
        ("爆发等级", "S=先到+30；A=先到+20但未到S；B=先到+10但未到A/S；F=其余；全部在W8内与-10比较"),
        ("LTR分组", "同一个SKDJ_N和Signal_Week构成一个查询组；只学习组内相对次序"),
        ("LTR标签收益", "F/B/A/S标签为0/1/2/3，成对权重使用0/1/3/7的等级收益差，S优先于稳定小涨"),
        ("LTR特征", "15项信号时点技术特征先转换为同N同周百分位；缺失按中性0.5；不使用买入后数据"),
        ("LTR模型", f"NumPy成对逻辑回归，L2正则C={LTR_C:g}；每周最多{LTR_MAX_PAIRS_PER_WEEK}个等级不同股票对且每周总权重相同"),
        ("LTR最小训练", f"至少{LTR_MIN_TRAIN_WEEKS}个已成熟信号周且{LTR_MIN_TRAIN_ROWS}个事件；不足时历史OOS名次留空而非伪造"),
        ("LTR防泄漏", "预测某历史周时，训练事件必须满足Entry_W8_End_Date严格早于当前Signal_Date"),
        ("LTR实盘模型", "只用历史截止日以前全部W8成熟N=6事件训练；N=7和N=9只作参数共振观察"),
        ("VCP", "信号前4周平均周振幅/前12周平均周振幅作为收缩特征；只参与LTR，不作硬条件"),
        ("52周位置", "使用信号前52周最近收盘距离52周最高价，不用全年总波幅作硬过滤"),
        ("时间分段", f"前段观察截止{split_end}；规则预先写死，后段不重新调权；不要求前后段绝对收益相同"),
        ("策略目标", "科技股高爆发优先；允许约三分之一失败，目标是三仓中尽量有两只达到A/S，而不是挑低波动小涨股票"),
        ("主评价目标", "S比例、A/S比例、B以上比例、W8最大浮盈、每周前三至少两只A/S；不把期末平滑收益作为主目标"),
        ("随机基准", "按每周候选数精确计算随机抽取最多3只的事件比例和超几何两只A/S概率，不依赖抽样运气"),
        ("买入", "信号完整周结束后的下一市场交易日开盘"),
        ("成本", "买卖0.2%滑点、佣金、过户费；卖出另计印花税"),
        ("先后顺序", "同一天同时触及止盈和-10%时保守计为-10%先；W8和W12分别独立计算"),
        ("本版边界", "不加入基本面、不增加新硬过滤、不研究止损止盈；实盘观察行不写入历史模型"),
        ("运行环境", f"Streamlit {streamlit_version}；运行稳定版要求requirements锁定1.61.0"),
    ], columns=["项目", "说明"])
    model_status = pd.DataFrame([{
        "历史信号开始": signal_start,
        "历史信号截止": signal_end,
        "行情观察截止": market_end,
        "最近完整市场周": live_week_end,
        "最近完整周候选": len(live_candidates),
        "最近有候选观察周": recent_candidate_week,
        "N6训练事件": int(live_model["train_rows"]) if live_model else 0,
        "N6训练周": int(live_model["train_weeks"]) if live_model else 0,
        "N6训练成对样本": int(live_model["pair_count"]) if live_model else 0,
        "实盘模型可用": bool(live_model is not None),
    }])
    files = {
        "01_run_summary_v6_0.csv": run_summary,
        "02_ltr_definitions_v6_0.csv": ltr_rules,
        "03_ltr_oos_top3_comparison_v6_0.csv": ltr_comparison,
        "04_ltr_oos_weekly_detail_v6_0.csv": ltr_week_detail,
        "05_ltr_full_history_coefficients_v6_0.csv": ltr_coefficients,
        "06_live_latest_complete_week_candidates_v6_0.csv": live_candidates_export,
        "07_live_model_status_v6_0.csv": model_status,
        "08_historical_explosion_class_outcomes_v6_0.csv": explosion_classes,
        "09_historical_candidate_calendar_v6_0.csv": ranked_calendar,
        "10_historical_w8_ltr_candidates_v6_0.csv": slim_candidates,
        "11_legacy_score_definitions_v6_0.csv": score_rules,
        "12_v57_baseline_definitions_v6_0.csv": v57_rules,
        "13_rejection_audit_v6_0.csv": rejection_audit,
        "14_metadata_v6_0.csv": metadata,
        "15_api_errors_v6_0.csv": pd.DataFrame({"错误": API_ERRORS}),
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
        f"完成：历史N=6/7/9的W8宽池候选分别为"
        f"{len(eligible[eligible['SKDJ_N'].eq(6)])}/"
        f"{len(eligible[eligible['SKDJ_N'].eq(7)])}/"
        f"{len(eligible[eligible['SKDJ_N'].eq(9)])}个；"
        f"最近完整市场周{live_week_end}共有{len(live_candidates)}个跨参数候选；"
        f"结果{'已保存' if persisted else '仅当前页面可下载'}。")
    st.subheader(f"最近完整市场周候选：{live_week_end}")
    st.caption(
        f"历史训练截止{signal_end}；行情观察截止{market_end}。下表仅供实盘观察，"
        "没有W8标签，也没有进入本次历史成绩。N=6的LTR名次是主排序。")
    primary_live = live_candidates_export[
        live_candidates_export.get(
            "SKDJ_N", pd.Series(index=live_candidates_export.index, dtype=float)
        ).eq(LTR_PRIMARY_N)].copy()
    if not primary_live.empty:
        render_plain_table(primary_live.sort_values("LTR_Live_Weekly_Rank"), 100)
    elif not live_candidates_export.empty:
        st.warning("最近完整周没有N=6候选；以下仅为N=7/9参数观察，不是主模型买入名单。")
        render_plain_table(live_candidates_export, 100)
    else:
        st.warning("最近完整周没有股票通过当前宽池与1～5周硬条件，本周观察名单为空。")
        if (recent_candidate_week and recent_candidate_week != live_week_end
                and not recent_reference_candidates.empty):
            st.caption(f"为便于查看形态，下面另列最近一次有候选的观察周：{recent_candidate_week}；它不是本周信号。")
            reference_export = recent_reference_candidates[[
                column for column in live_columns
                if column in recent_reference_candidates.columns]].copy()
            render_plain_table(reference_export, 100)
    st.subheader("N=6严格时间外：LTR与随机/旧排名同周比较")
    render_plain_table(ltr_comparison, 30)
    st.subheader("N=6全历史实盘模型特征方向")
    render_plain_table(ltr_coefficients, 30)
    st.subheader("N=6 / N=7 / N=9运行摘要")
    render_plain_table(run_summary)
    st.caption("结果ZIP含15个文件；最近周候选与历史W8训练/验收在代码中严格分流。")
    render_download(result_zip, result_name, f"v60_current_{request_signature}")


def legacy_main_v61() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    streamlit_version = str(getattr(st, "__version__", "unknown"))
    st.caption(
        f"{UI_PATCH}｜先验证买点时序，不训练评分模型。｜Streamlit {streamlit_version}")
    with st.expander("本版验证的两个方案", expanded=True):
        st.markdown(f"""
- **方案一（提前买入）**：以最近已经完成的周线为准，N=6周线SKDJ的K处于{EARLY_WEEKLY_K_MIN:g}～{EARLY_WEEKLY_K_MAX:g}，日线MACD进入第{EARLY_RED_AGE_MIN}～{EARLY_RED_AGE_MAX}根红柱；信号收盘确认，下一市场交易日开盘买入。
- **关键防泄漏**：方案一独立扫描全部历史科技池。后来是否上穿周线25只作为结果中的配对字段，**不参与信号生成**；从未在六周内上穿25的失败案例同样保留。
- **方案二（周线确认过滤）**：继续在N=6周线K由25下方上穿25后，检查当日日线MACD是红柱扩张、健康缩短、明显衰减、临近翻绿还是已经翻绿。
- **红柱状态**：`剩余强度=当日红柱/本轮最高红柱`；`单日保留=当日红柱/上一日红柱`。健康缩短暂记为剩余≥{MACD_HEALTHY_REMAINING_PCT:g}%且单日保留≥{MACD_HEALTHY_RETENTION_PCT:g}%；临近翻绿同时审计10%、20%、30%三道阈值，不预先挑最好的一条。
- **历史判卷**：信号日不晚于历史截止、下一日可成交并已走完{DAILY_AUDIT_DAYS}个市场交易日；S/A/B/F从新买入价重新计算，分别代表先到+30/+20/+10或均未达到，全部与-10%比较。
- **退出只是审计**：方案一同时比较固定40日，以及红柱剩余≤10/20/30%或翻绿后下一可交易日开盘退出；本版不会据结果自动修改规则。
- **实时观察**：历史截止以后的事件单独列出，不进入历史收益率、等级和阈值比较。
""")

    with st.sidebar:
        st.header("运行参数")
        backtest_days = st.number_input(
            "回测交易日数", 100, 1000, 500, 50, key="v61_days")
        min_price = st.number_input(
            "最低股价（元）", 10.0, 20.0, 10.0, 10.0,
            format="%.0f", key="v61_min_price")
        min_mv = st.number_input(
            "最低流通市值（亿元）", 50.0, 100.0, 50.0, 50.0,
            format="%.0f", key="v61_min_mv")
        signal_end_date = st.date_input(
            "历史买入信号截止（40日判卷）", date(2026, 6, 5),
            key="v61_signal_end")
        market_end_date = st.date_input(
            "最新信号观察截止（默认今天）", date.today(),
            key="v61_market_end")
        st.caption(
            "历史统计只用信号日≤历史截止的成熟事件；最新观察名单不进入历史结论。")
        pause = st.number_input(
            "接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v61_pause")
        use_cache = st.checkbox("复用行情缓存", True, key="v61_cache")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005,
            format="%.3f", key="v61_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01,
            format="%.3f", key="v61_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001,
            format="%.3f", key="v61_transfer")
        if st.button("清除V6.1结果和运行状态", key="v61_clear"):
            shutil.rmtree(RESULT_DIR, ignore_errors=True)
            shutil.rmtree(JOB_DIR, ignore_errors=True)
            st.success("V6.1结果和运行状态已清除；逐股票检查点及通用行情缓存保留。")

    request_payload = {
        "version": VERSION, "days": int(backtest_days),
        "signal_end": signal_end_date.strftime("%Y%m%d"),
        "market_end": market_end_date.strftime("%Y%m%d"),
        "min_price": float(min_price), "min_mv": float(min_mv),
        "commission": float(commission_pct), "stamp": float(stamp_duty_pct),
        "transfer": float(transfer_fee_pct),
        "early_k_min": EARLY_WEEKLY_K_MIN, "early_k_max": EARLY_WEEKLY_K_MAX,
        "red_age": [EARLY_RED_AGE_MIN, EARLY_RED_AGE_MAX],
    }
    request_signature = stable_signature(request_payload)
    result_path = os.path.join(RESULT_DIR, f"{request_signature}.zip")
    result_name = (
        f"weekly_skdj_daily_macd_timing_audit_v6_1_{int(backtest_days)}d_"
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
                saved_result, result_name, f"v61_saved_{request_signature}")
        except Exception as exc:
            st.warning(f"已保存结果读取失败：{exc}")

    secret_token = configured_tushare_token()
    if secret_token:
        token = secret_token
        st.caption("已从Streamlit Secrets读取TUSHARE_TOKEN。")
    else:
        token = st.text_input("Tushare Token", type="password", key="v61_token")

    job_active = is_job_active(request_signature)
    left, right = st.columns(2)
    with left:
        start_clicked = st.button(
            "开始/重新运行V6.1", type="primary", key="v61_run")
    with right:
        stop_clicked = st.button(
            "停止自动续跑", disabled=not job_active, key="v61_stop")
    if stop_clicked:
        clear_job_active(request_signature)
        st.success("已停止；已完成股票的检查点保留。")
        return
    if start_clicked:
        if market_end_date <= signal_end_date:
            st.error("最新信号观察截止必须晚于历史信号截止")
            return
        mark_job_active(request_signature)
        job_active = True
    if not token:
        st.info("请输入Token；任务启动后如页面重连，会从逐股票检查点自动续跑。")
        return
    if not job_active:
        st.caption(
            "点击开始运行。" if not completed_available
            else "相同参数结果已可下载；如需覆盖，请点击重新运行。")
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
    except Exception as exc:
        st.error(f"确定{int(backtest_days)}个交易日窗口失败：{exc}")
        return
    data_start_date = pd.Timestamp(signal_start).date() - timedelta(
        weeks=WARMUP_WEEKS, days=7)
    data_start = data_start_date.strftime("%Y%m%d")
    config = {
        "signal_start": signal_start, "signal_end": signal_end,
        "event_signal_end": market_end, "data_start": data_start,
        "market_end": market_end, "min_price": float(min_price),
        "min_mv": float(min_mv), "buy_slippage_pct": 0.20,
        "sell_slippage_pct": 0.20, "commission_pct": float(commission_pct),
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
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return
    if not open_dates:
        st.error("区间内没有市场交易日")
        return
    latest_market_date = max(value for value in open_dates if value <= market_end)
    market_weeks = market_week_sequence(open_dates)
    open_pos = {day: position for position, day in enumerate(open_dates)}
    period_index = build_period_index(memberships)
    active_codes = {
        code for code, code_periods in period_index.items()
        if periods_overlap(code_periods, signal_start, market_end)
    }
    stocks = stock_basic[stock_basic["ts_code"].isin(active_codes)].copy()
    stocks = stocks[
        ~stocks["list_date"].gt(market_end)
        & ~stocks["delist_date"].lt(data_start)
    ].sort_values("ts_code").reset_index(drop=True)

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
                    rows, stock_rejects, stock_breadth = analyze_stock(
                        stock, period_index.get(code, []), daily, cached_basic,
                        storage_path, week_last_map, open_dates, open_pos,
                        market_weeks, config, bool(use_cache), float(pause))
                    event_rows.extend(rows)
                    merge_counts(rejects, stock_rejects)
                    save_checkpoint(
                        run_signature, code, rows, stock_rejects, stock_breadth)
                except Exception as exc:
                    failures += 1
                    record_error(f"逐股票分析失败 {code}: {exc}")
        processed = number + 1
        now = time.monotonic()
        if (processed == 1 or now - last_update >= UI_HEARTBEAT_SECONDS
                or processed == len(stocks)):
            progress.progress(
                processed / max(len(stocks), 1),
                text=f"已处理{processed}/{len(stocks)}只股票，最近{code}")
            status.caption(
                f"时序事件{len(event_rows)}；检查点{checkpoint_hits}；"
                f"行情缓存{price_cache_hits}；失败{failures}")
            last_update = now
    progress.empty()
    status.empty()
    if stopped:
        st.warning("任务已停止，逐股票检查点已保留。")
        return

    events_all = pd.DataFrame(event_rows)
    if events_all.empty:
        st.error("本区间没有生成通过历史时点过滤的提前或周线确认事件。")
        return
    events_all = events_all.sort_values(
        ["Signal_Date", "Event_Type", "ts_code"]).reset_index(drop=True)
    history = events_all[events_all["Signal_Date"].astype(str).le(signal_end)].copy()
    observation = events_all[
        events_all["Signal_Date"].astype(str).gt(signal_end)
        & events_all["Signal_Date"].astype(str).le(latest_market_date)
    ].copy()
    if history["Signal_Date"].astype(str).gt(signal_end).any():
        raise RuntimeError("历史时序样本混入了观察截止后的事件")
    mature = history[
        true_mask(history, "Entry_Tradable")
        & true_mask(history, "Entry_Has_40D")
    ].copy()
    if mature.empty:
        st.error("存在时序信号，但没有可成交且已走完40个市场交易日的成熟事件。")
        return
    mature = add_timing_labels(mature)
    early = mature[mature["Event_Type"].eq("EARLY_DAILY_MACD")].copy()
    weekly = mature[mature["Event_Type"].eq("WEEKLY_CROSS25")].copy()
    if early.empty or weekly.empty:
        st.error("提前事件或周线确认对照为空，无法完成双方案比较。")
        return

    weekly["Weekly_MACD_Filter_Group"] = "中间状态_单独观察"
    weekly.loc[
        weekly["Daily_MACD_State"].isin(["红柱首日", "红柱扩张", "健康缩短"]),
        "Weekly_MACD_Filter_Group"] = "方案二保留_扩张或健康缩短"
    weekly.loc[
        weekly["Daily_MACD_State"].isin(["临近翻绿", "已翻绿"]),
        "Weekly_MACD_Filter_Group"] = "方案二剔除_临近或已经翻绿"

    early["Weekly_K_Band"] = pd.cut(
        numeric(early, "Signal_K"), [15, 20, 25, 30, 35],
        labels=["15～20", "20～25", "25～30", "30～35"],
        include_lowest=True, right=True).astype(str)
    early["Daily_Red_Age_Group"] = numeric(
        early, "Daily_MACD_Red_Age").round().astype("Int64").astype(str)
    weekly["Daily_MACD_Remaining_Band"] = pd.cut(
        numeric(weekly, "Daily_MACD_Remaining_pct"),
        [-np.inf, 0, 10, 20, 30, 50, 75, 100, np.inf],
        labels=["已翻绿/≤0", "0～10", "10～20", "20～30", "30～50",
                "50～75", "75～100", ">100"], right=True).astype(str)
    weekly["Daily_Red_Age_Band"] = pd.cut(
        numeric(weekly, "Daily_MACD_Red_Age"),
        [-np.inf, 0, 5, 10, 15, 20, np.inf],
        labels=["已翻绿/0", "1～5", "6～10", "11～15", "16～20", ">20"],
        right=True).astype(str)
    weekly["Pre_Cross_Rally_Band"] = pd.cut(
        numeric(weekly, "Daily_Return_Since_Red_Start_pct"),
        [-np.inf, 0, 10, 20, 30, np.inf],
        labels=["<0", "0～10", "10～20", "20～30", "≥30"],
        right=True).astype(str)

    overall_parts = [
        timing_outcome_summary(early, [], "方案一：提前日线MACD第2～5红柱"),
        timing_outcome_summary(weekly, [], "旧买点：周线K上穿25全部"),
        timing_outcome_summary(
            weekly[weekly["Weekly_MACD_Filter_Group"].eq(
                "方案二保留_扩张或健康缩短")], [],
            "方案二：周线确认且日线MACD健康"),
        timing_outcome_summary(
            weekly[weekly["Weekly_MACD_Filter_Group"].eq(
                "方案二剔除_临近或已经翻绿")], [],
            "方案二被剔除组"),
    ]
    overall_comparison = pd.concat(overall_parts, ignore_index=True)
    weekly_state_audit = timing_outcome_summary(
        weekly, ["Daily_MACD_State"], "周线确认当日日线状态")
    weekly_remaining_audit = timing_outcome_summary(
        weekly, ["Daily_MACD_Remaining_Band"], "周线确认红柱剩余强度")
    weekly_red_age_audit = timing_outcome_summary(
        weekly, ["Daily_Red_Age_Band"], "周线确认时红柱已经持续多久")
    weekly_pre_rally_audit = timing_outcome_summary(
        weekly, ["Pre_Cross_Rally_Band"], "周线确认前本轮日线上涨幅度")
    weekly_lateness_overview = pd.DataFrame([{
        "周线确认成熟事件": len(weekly),
        "确认日MACD仍为红柱比例%": true_mask(
            weekly, "Daily_MACD_Positive").mean() * 100.0,
        "确认日红柱日龄均值": numeric(weekly, "Daily_MACD_Red_Age").mean(),
        "确认日红柱日龄中位": numeric(weekly, "Daily_MACD_Red_Age").median(),
        "确认时已持续11至15日比例%": numeric(
            weekly, "Daily_MACD_Red_Age").between(11, 15).mean() * 100.0,
        "确认前本轮日线上涨均值%": numeric(
            weekly, "Daily_Return_Since_Red_Start_pct").mean(),
        "确认前本轮日线上涨中位%": numeric(
            weekly, "Daily_Return_Since_Red_Start_pct").median(),
        "确认前已上涨10至30比例%": numeric(
            weekly, "Daily_Return_Since_Red_Start_pct").between(10, 30).mean() * 100.0,
        "确认日临近或已经翻绿比例%": weekly["Daily_MACD_State"].isin(
            ["临近翻绿", "已翻绿"]).mean() * 100.0,
    }])
    scheme2_audit = timing_outcome_summary(
        weekly, ["Weekly_MACD_Filter_Group"], "方案二三分组")
    early_k_audit = timing_outcome_summary(
        early, ["Weekly_K_Band"], "方案一周线准备区")
    early_age_audit = timing_outcome_summary(
        early, ["Daily_Red_Age_Group"], "方案一买在第几根红柱")
    early_pair = early.copy()
    early_pair["Future_Cross_Group"] = np.where(
        true_mask(early_pair, "Future_Weekly_Cross25_Within42D"),
        "六周内后来上穿25", "六周内没有上穿25")
    early_pair_audit = timing_outcome_summary(
        early_pair, ["Future_Cross_Group"], "方案一未来周线确认仅作审计")
    pair_price_audit = (early_pair.groupby("Future_Cross_Group", dropna=False).agg(
        事件数=("ts_code", "size"),
        配对周线确认数=("Future_Weekly_Cross25_Date", lambda values: values.astype(str).ne("").sum()),
        领先自然日均值=("Lead_Calendar_Days_to_Weekly_Cross", "mean"),
        领先自然日中位=("Lead_Calendar_Days_to_Weekly_Cross", "median"),
        周线确认买价相对提前买价涨幅均值_pct=("Early_Price_Advantage_vs_Weekly_pct", "mean"),
        周线确认买价相对提前买价涨幅中位_pct=("Early_Price_Advantage_vs_Weekly_pct", "median"),
    ).reset_index())
    paired_events = early_pair[
        true_mask(early_pair, "Future_Weekly_Cross25_Within42D")
    ].merge(
        weekly, left_on=["ts_code", "Future_Weekly_Cross25_Date"],
        right_on=["ts_code", "Signal_Date"], how="inner",
        suffixes=("_Early", "_Weekly"))
    if paired_events.empty:
        paired_timing_comparison = pd.DataFrame()
        paired_event_export = pd.DataFrame()
    else:
        early_grade = paired_events["Explosion_Class_40D_Early"].map(
            {"F": 0, "B": 1, "A": 2, "S": 3})
        weekly_grade = paired_events["Explosion_Class_40D_Weekly"].map(
            {"F": 0, "B": 1, "A": 2, "S": 3})
        paired_timing_comparison = pd.DataFrame([{
            "同股同轮成熟配对数": len(paired_events),
            "提前买价相对优势均值%": numeric(
                paired_events, "Early_Price_Advantage_vs_Weekly_pct_Early").mean(),
            "提前买价相对优势中位%": numeric(
                paired_events, "Early_Price_Advantage_vs_Weekly_pct_Early").median(),
            "提前买S比例%": paired_events[
                "Explosion_Class_40D_Early"].astype(str).eq("S").mean() * 100.0,
            "周线确认买S比例%": paired_events[
                "Explosion_Class_40D_Weekly"].astype(str).eq("S").mean() * 100.0,
            "提前买A或S比例%": paired_events[
                "Explosion_Class_40D_Early"].astype(str).isin(["A", "S"]).mean() * 100.0,
            "周线确认买A或S比例%": paired_events[
                "Explosion_Class_40D_Weekly"].astype(str).isin(["A", "S"]).mean() * 100.0,
            "提前买F比例%": paired_events[
                "Explosion_Class_40D_Early"].astype(str).eq("F").mean() * 100.0,
            "周线确认买F比例%": paired_events[
                "Explosion_Class_40D_Weekly"].astype(str).eq("F").mean() * 100.0,
            "提前买等级高于周线确认比例%": early_grade.gt(weekly_grade).mean() * 100.0,
            "提前买等级低于周线确认比例%": early_grade.lt(weekly_grade).mean() * 100.0,
            "提前买40日最大浮盈均值%": numeric(
                paired_events, "Entry_MFE_Net_pct_Early").mean(),
            "周线确认买40日最大浮盈均值%": numeric(
                paired_events, "Entry_MFE_Net_pct_Weekly").mean(),
            "提前买40日最大回撤均值%": numeric(
                paired_events, "Entry_MAE_Raw_pct_Early").mean(),
            "周线确认买40日最大回撤均值%": numeric(
                paired_events, "Entry_MAE_Raw_pct_Weekly").mean(),
        }])
        paired_columns = [
            "ts_code", "name_Early", "Signal_Date_Early",
            "Future_Weekly_Cross25_Date_Early", "Signal_Date_Weekly",
            "Entry_Date_Early", "Entry_Raw_Open_Early",
            "Entry_Date_Weekly", "Entry_Raw_Open_Weekly",
            "Early_Price_Advantage_vs_Weekly_pct_Early",
            "Explosion_Class_40D_Early", "Explosion_Class_40D_Weekly",
            "Entry_MFE_Net_pct_Early", "Entry_MFE_Net_pct_Weekly",
            "Entry_MAE_Raw_pct_Early", "Entry_MAE_Raw_pct_Weekly",
            "Daily_MACD_State_Weekly", "Daily_MACD_Red_Age_Weekly",
            "Daily_MACD_Remaining_pct_Weekly",
        ]
        paired_event_export = paired_events[[
            column for column in paired_columns if column in paired_events.columns]].copy()
    exit_audit = early_exit_strategy_audit(early)
    calendar = timing_calendar(
        open_dates, signal_start, signal_end, early, weekly)

    latest_early_date = ""
    latest_early = pd.DataFrame()
    observation_early = observation[
        observation["Event_Type"].eq("EARLY_DAILY_MACD")].copy()
    if not observation_early.empty:
        latest_early_date = str(observation_early["Signal_Date"].astype(str).max())
        latest_early = observation_early[
            observation_early["Signal_Date"].astype(str).eq(latest_early_date)].copy()
    latest_weekly_date = ""
    latest_weekly = pd.DataFrame()
    observation_weekly = observation[
        observation["Event_Type"].eq("WEEKLY_CROSS25")].copy()
    if not observation_weekly.empty:
        latest_weekly_date = str(observation_weekly["Signal_Date"].astype(str).max())
        latest_weekly = observation_weekly[
            observation_weekly["Signal_Date"].astype(str).eq(latest_weekly_date)].copy()

    run_summary = pd.DataFrame([{
        "程序版本": VERSION, "正式信号开始": signal_start,
        "历史信号截止": signal_end, "行情观察截止": market_end,
        "最新市场交易日": latest_market_date,
        "方案一全部历史事件": len(history[history["Event_Type"].eq("EARLY_DAILY_MACD")]),
        "方案一40日成熟事件": len(early),
        "方案一不同股票": early["ts_code"].nunique(),
        "方案一有信号周": early["Signal_Week"].nunique(),
        "方案一空窗周": int(calendar["提前日线信号"].eq(0).sum()),
        "方案一单周最多": int(calendar["提前日线信号"].max()),
        "周线确认40日成熟事件": len(weekly),
        "周线确认不同股票": weekly["ts_code"].nunique(),
        "周线确认有信号周": weekly["Signal_Week"].nunique(),
        "周线确认空窗周": int(calendar["周线确认信号"].eq(0).sum()),
        "方案二保留事件": int(weekly["Weekly_MACD_Filter_Group"].eq(
            "方案二保留_扩张或健康缩短").sum()),
        "方案二剔除事件": int(weekly["Weekly_MACD_Filter_Group"].eq(
            "方案二剔除_临近或已经翻绿").sum()),
        "六周内后来上穿25的提前事件": int(true_mask(
            early, "Future_Weekly_Cross25_Within42D").sum()),
        "六周内未上穿25的提前事件": int((~true_mask(
            early, "Future_Weekly_Cross25_Within42D")).sum()),
        "历史截止后提前观察事件": len(observation_early),
        "最近提前信号日期": latest_early_date,
        "最近提前信号股票数": len(latest_early),
        "处理股票数": len(stocks), "检查点恢复": checkpoint_hits,
        "行情缓存命中": price_cache_hits, "失败股票": failures,
    }])

    definitions = pd.DataFrame([
        ("方案一候选池", f"历史科技池、股价≥{min_price:g}元、流通市值≥{min_mv:g}亿元；不要求未来周线突破"),
        ("方案一周线准备区", f"当日日线收盘时可见的最近完整周N=6 SKDJ K在{EARLY_WEEKLY_K_MIN:g}～{EARLY_WEEKLY_K_MAX:g}"),
        ("方案一日线触发", f"日线MACD正柱连续第{EARLY_RED_AGE_MIN}～{EARLY_RED_AGE_MAX}日；同一轮红柱只取首次满足日"),
        ("方案二原信号", "N=6上一完整周K<25、本完整周K≥25"),
        ("方案二暂定保留", f"红柱首日/扩张，或缩短但剩余≥{MACD_HEALTHY_REMAINING_PCT:g}%且单日保留≥{MACD_HEALTHY_RETENTION_PCT:g}%"),
        ("方案二暂定剔除", "周线确认日已经翻绿，或红柱缩短且仅剩本轮峰值20%以下"),
        ("买入", "信号收盘确认后的下一市场交易日开盘；停牌和主板一字板不成交"),
        ("成熟", f"买入后已有完整{DAILY_AUDIT_DAYS}个市场交易日；不把当前未成熟候选写入历史收益"),
        ("等级", "S/A/B分别为40日内先到+30/+20/+10且先于-10；否则F；同日冲突保守按-10先"),
        ("未来周线配对", f"只审计提前信号后{FUTURE_WEEKLY_CROSS_DAYS}自然日内是否K上穿25，绝不反向决定提前事件是否入池"),
        ("MACD口径", "DIF=EMA12-EMA26，DEA=EMA(DIF,9)，柱=DIF-DEA；是否乘2不影响剩余强度和状态"),
        ("防未来函数", "周线状态只用当日日线收盘前已经完成的周数据；日线退出信号收盘确认后下一可交易日开盘执行"),
    ], columns=["项目", "定义"])
    rejection_audit = pd.DataFrame([
        {"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())])
    metadata = pd.DataFrame([{
        "程序版本": VERSION, "SKDJ_N": EARLY_PRIMARY_N, "SKDJ_M": SKDJ_M,
        "回测交易日": int(backtest_days), "预热周": WARMUP_WEEKS,
        "历史信号开始": signal_start, "历史信号截止": signal_end,
        "行情截止": market_end, "最低股价": float(min_price),
        "最低流通市值亿元": float(min_mv),
        "日线审计市场日": DAILY_AUDIT_DAYS,
        "Streamlit": streamlit_version,
    }])

    event_columns = [
        "Event_Type", "ts_code", "name", "Signal_Date", "Signal_Week",
        "SW_L1", "SW_L2", "Raw_Close", "Circ_MV_Billion",
        "Setup_Weekly_Date", "Signal_K", "Signal_D", "Signal_KD_Spread",
        "Daily_MACD_State", "Daily_MACD_Red_Age",
        "Daily_MACD_Remaining_pct", "Daily_MACD_Retention_pct",
        "Daily_Return_Since_Red_Start_pct",
        "Future_Weekly_Cross25_Within42D", "Future_Weekly_Cross25_Date",
        "Lead_Calendar_Days_to_Weekly_Cross",
        "Early_Price_Advantage_vs_Weekly_pct", "Entry_Date", "Entry_Raw_Open",
        "Entry_MFE_Net_pct", "Entry_MAE_Raw_pct",
        "Entry_Close_Return_Net_pct", "Explosion_Class_40D",
        "Entry_First_Hit_10_vs_Minus10_40D",
        "Entry_First_Hit_20_vs_Minus10_40D",
        "Entry_First_Hit_30_vs_Minus10_40D",
        "Entry_Exit_Remaining10_Date", "Entry_Exit_Remaining10_Reason",
        "Entry_Exit_Remaining10_Return_Net_pct",
        "Entry_Exit_Remaining20_Date", "Entry_Exit_Remaining20_Reason",
        "Entry_Exit_Remaining20_Return_Net_pct",
        "Entry_Exit_Remaining30_Date", "Entry_Exit_Remaining30_Reason",
        "Entry_Exit_Remaining30_Return_Net_pct",
    ]
    history_early_export = early[[
        column for column in event_columns if column in early.columns]].copy()
    history_weekly_export = weekly[[
        column for column in event_columns + [
            "Weekly_MACD_Filter_Group", "Daily_MACD_Remaining_Band"]
        if column in weekly.columns]].copy()
    live_columns = [
        "Event_Type", "ts_code", "name", "Signal_Date", "Signal_Week",
        "SW_L1", "SW_L2", "Raw_Close", "Circ_MV_Billion",
        "Setup_Weekly_Date", "Signal_K", "Signal_D", "Signal_KD_Spread",
        "Daily_MACD_State", "Daily_MACD_Red_Age",
        "Daily_MACD_Remaining_pct", "Daily_MACD_Retention_pct",
        "Daily_Return_Since_Red_Start_pct",
    ]
    latest_early_export = latest_early[[
        column for column in live_columns if column in latest_early.columns]].copy()
    latest_weekly_export = latest_weekly[[
        column for column in live_columns if column in latest_weekly.columns]].copy()
    files = {
        "01_run_summary_v6_1.csv": run_summary,
        "02_experiment_definitions_v6_1.csv": definitions,
        "03_early_vs_weekly_timing_outcomes_v6_1.csv": overall_comparison,
        "04_weekly_cross_daily_macd_state_outcomes_v6_1.csv": weekly_state_audit,
        "05_weekly_cross_remaining_strength_outcomes_v6_1.csv": weekly_remaining_audit,
        "06_weekly_cross_lateness_overview_v6_1.csv": weekly_lateness_overview,
        "07_weekly_cross_red_age_outcomes_v6_1.csv": weekly_red_age_audit,
        "08_weekly_cross_pre_rally_outcomes_v6_1.csv": weekly_pre_rally_audit,
        "09_scheme2_filter_audit_v6_1.csv": scheme2_audit,
        "10_early_weekly_k_band_outcomes_v6_1.csv": early_k_audit,
        "11_early_red_age_outcomes_v6_1.csv": early_age_audit,
        "12_early_future_weekly_cross_outcomes_v6_1.csv": early_pair_audit,
        "13_early_lead_and_price_advantage_v6_1.csv": pair_price_audit,
        "14_same_stock_same_cycle_timing_comparison_v6_1.csv": paired_timing_comparison,
        "15_same_stock_same_cycle_event_detail_v6_1.csv": paired_event_export,
        "16_early_macd_exit_strategy_audit_v6_1.csv": exit_audit,
        "17_weekly_signal_coverage_v6_1.csv": calendar,
        "18_historical_early_events_v6_1.csv": history_early_export,
        "19_historical_weekly_cross_events_v6_1.csv": history_weekly_export,
        "20_latest_early_candidates_v6_1.csv": latest_early_export,
        "21_latest_weekly_cross_candidates_v6_1.csv": latest_weekly_export,
        "22_rejection_audit_v6_1.csv": rejection_audit,
        "23_metadata_v6_1.csv": metadata,
        "24_api_errors_v6_1.csv": pd.DataFrame({"错误": API_ERRORS}),
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
        f"完成：方案一提前买点{len(early)}个成熟事件，周线确认对照{len(weekly)}个；"
        f"方案一中{int(true_mask(early, 'Future_Weekly_Cross25_Within42D').sum())}个后来六周内上穿25，"
        f"{int((~true_mask(early, 'Future_Weekly_Cross25_Within42D')).sum())}个未上穿；"
        f"结果{'已保存' if persisted else '仅当前页面可下载'}。")
    st.subheader("方案一、旧买点与方案二过滤的同口径比较")
    render_plain_table(overall_comparison, 20)
    st.subheader("周线K上穿25当日：日线MACD状态与后续等级")
    render_plain_table(weekly_lateness_overview, 5)
    render_plain_table(weekly_state_audit, 20)
    st.subheader("方案一：周线K位置与日线红柱日龄")
    render_plain_table(early_k_audit, 20)
    render_plain_table(early_age_audit, 20)
    st.subheader("提前信号是否后来得到周线25确认")
    render_plain_table(early_pair_audit, 10)
    render_plain_table(pair_price_audit, 10)
    st.subheader("同一只股票同一轮：提前买与周线确认买")
    render_plain_table(paired_timing_comparison, 5)
    st.subheader(f"最近一次提前观察信号：{latest_early_date or '无'}")
    st.caption(
        f"最新市场交易日为{latest_market_date}；这里只展示历史截止{signal_end}之后最近一次有信号的日期，"
        "它不是历史成绩，也没有未来等级。")
    render_plain_table(latest_early_export, 100)
    st.subheader("运行摘要")
    render_plain_table(run_summary, 10)
    st.caption("结果ZIP共24个CSV；历史成熟事件与最新观察事件严格分开。")
    render_download(result_zip, result_name, f"v61_current_{request_signature}")


def _v62_signal_fields(signal: pd.Series) -> dict[str, Any]:
    """Signal-time fields shared by every V6.2 event."""
    return {
        "Setup_Weekly_Date": str(signal.get("Setup_Weekly_Date", "")),
        "Signal_K": finite_num(signal.get("Setup_Weekly_K", signal.get("K"))),
        "Signal_D": finite_num(signal.get("Setup_Weekly_D", signal.get("D"))),
        "Signal_KD_Spread": finite_num(
            signal.get("Setup_Weekly_KD_Spread", signal.get("KD_Spread"))),
        "Signal_K_Change_1W": finite_num(
            signal.get("Setup_Weekly_K_Change_1W", signal.get("K_Change_1W"))),
        "Signal_Prior_Below25_Streak": finite_num(
            signal.get("Setup_Prior_Below25_Streak", signal.get("Prior_Below25_Streak"))),
        "Signal_Close_to_MA20_pct": finite_num(
            signal.get("Setup_Close_to_MA20_pct", signal.get("Close_to_MA20_pct"))),
        "Signal_MA20_Slope_4W_pct": finite_num(
            signal.get("Setup_MA20_Slope_4W_pct", signal.get("MA20_Slope_4W_pct"))),
        "Signal_Volume_Ratio_5W": finite_num(
            signal.get("Setup_Volume_Ratio_5W", signal.get("Volume_Ratio_5W"))),
        "Signal_Week_Return_pct": finite_num(
            signal.get("Setup_Week_Return_pct", signal.get("Signal_Week_Return_pct"))),
        **macd_snapshot(signal),
    }


def _v62_next_threshold(value: float) -> str:
    if not math.isfinite(value):
        return "等待日线红柱"
    for threshold in STRENGTH_THRESHOLDS:
        if value < threshold:
            return f"等待达到{int(threshold)}%"
    if value <= STRENGTH_MAX_PRIOR_RALLY_PCT:
        return "已达到10%"
    return "已超过30%风险线"


def analyze_stock_v62(
        stock: pd.Series, periods: list[dict[str, str]], daily: pd.DataFrame,
        cached_basic: pd.DataFrame, storage_path: str,
        week_last_map: dict[pd.Timestamp, str], open_dates: list[str],
        open_pos: dict[str, int], config: dict[str, Any],
        use_cache: bool, api_pause: float
        ) -> tuple[list[dict[str, Any]], dict[str, int],
                   dict[str, dict[str, float]]]:
    """Generate strength-confirmation, common-cohort and weekly controls."""
    rejects: dict[str, int] = {}
    weekly_base = aggregate_complete_weekly(daily, week_last_map)
    if weekly_base.empty:
        return [], rejects, {}
    n6 = add_skdj(weekly_base, EARLY_PRIMARY_N)
    daily_macd = attach_latest_completed_weekly(add_daily_macd(daily), n6)
    event_end = str(config.get("event_signal_end", config["signal_end"]))
    date_series = daily_macd["trade_date"].astype(str)
    formal_date = date_series.between(config["signal_start"], event_end)
    watch_k = numeric(daily_macd, "Setup_Weekly_K").between(
        EARLY_WEEKLY_K_MIN, EARLY_WEEKLY_K_MAX, inclusive="both")
    positive = true_mask(daily_macd, "Daily_MACD_Positive")
    red_age = numeric(daily_macd, "Daily_MACD_Red_Age")
    rally = numeric(daily_macd, "Daily_Return_Since_Red_Start_pct")
    remaining = numeric(daily_macd, "Daily_MACD_Remaining_pct")
    quality = (
        daily_macd["Daily_MACD_State"].astype(str).eq("红柱扩张")
        | remaining.ge(STRENGTH_MIN_REMAINING_PCT))

    # Real-time age-2 baseline: no future survival requirement.
    age2_candidates = daily_macd[
        formal_date & watch_k & positive & red_age.eq(2)
    ].copy().sort_values("trade_date")
    if not age2_candidates.empty:
        age2_candidates = age2_candidates.groupby(
            "Daily_MACD_Cycle", as_index=False, sort=False).first()

    # Four independent real-time strategies.  A threshold is recorded on the
    # first observable close in that cycle satisfying all conditions.  Later
    # weekly cross information never affects inclusion.
    strength_candidates: list[tuple[float, pd.Series]] = []
    for threshold in STRENGTH_THRESHOLDS:
        selected = daily_macd[
            formal_date & watch_k & positive & quality & rally.ge(threshold)
        ].copy().sort_values("trade_date")
        if not selected.empty:
            selected = selected.groupby(
                "Daily_MACD_Cycle", as_index=False, sort=False).first()
            strength_candidates.extend(
                (float(threshold), row) for _, row in selected.iterrows())

    # Diagnostic common cohort.  Inclusion requires knowledge that the red
    # cycle survives through day 5; these rows are explicitly never live rules.
    cohort_candidates: list[tuple[int, str, pd.Series]] = []
    for _, base in age2_candidates.iterrows():
        cycle = int(finite_num(base.get("Daily_MACD_Cycle")))
        cycle_rows = daily_macd[
            numeric(daily_macd, "Daily_MACD_Cycle").eq(cycle)
            & date_series.le(event_end)
        ].copy()
        by_age = {
            int(finite_num(row.get("Daily_MACD_Red_Age"))): row
            for _, row in cycle_rows.iterrows()
            if int(finite_num(row.get("Daily_MACD_Red_Age"))) in COHORT_RED_AGES
        }
        if not all(age in by_age for age in COHORT_RED_AGES):
            continue
        cohort_id = f"{str(stock['ts_code'])}|{cycle}|{str(base['trade_date'])}"
        cohort_candidates.extend(
            (age, cohort_id, by_age[age]) for age in COHORT_RED_AGES)

    weekly_crosses = n6[
        true_mask(n6, "K_Cross_25")
        & n6["trade_date"].astype(str).between(config["signal_start"], event_end)
    ].copy()

    latest_watch = pd.DataFrame()
    latest_market_date = str(config.get("latest_market_date", config["market_end"]))
    exact_latest = daily_macd[date_series.eq(latest_market_date) & watch_k].copy()
    if not exact_latest.empty:
        latest_watch = exact_latest.tail(1)

    estimated_events = (
        len(age2_candidates) + len(strength_candidates)
        + len(cohort_candidates) + len(weekly_crosses) + len(latest_watch))
    if estimated_events == 0:
        return [], rejects, {}

    code = str(stock["ts_code"])
    basic = ensure_daily_basic(
        code, config["data_start"], config["market_end"], daily,
        cached_basic, storage_path, use_cache, api_pause)
    if basic.empty:
        rejects["存在V6.2信号但daily_basic缺失"] = estimated_events
        return [], rejects, {}

    cross_dates = weekly_crosses["trade_date"].astype(str).sort_values().tolist()
    outcome_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    def valid_context(signal_date: str) -> tuple[dict[str, str] | None, dict[str, float], str]:
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
        return membership, snapshot, reason

    def append_event(
            signal: pd.Series, event_type: str, strategy_label: str,
            strength_threshold: float = np.nan, cohort_age: float = np.nan,
            cohort_id: str = "", diagnostic_future_condition: bool = False,
            calculate_outcome: bool = True) -> None:
        signal_date = str(signal["trade_date"])
        membership, snapshot, reason = valid_context(signal_date)
        if reason or membership is None:
            rejects[reason] = rejects.get(reason, 0) + 1
            return
        outcome: dict[str, Any] = {}
        if calculate_outcome:
            if signal_date not in outcome_cache:
                outcome_cache[signal_date] = daily_timing_outcomes(
                    daily_macd, signal_date, code, open_dates, open_pos, config)
            outcome = outcome_cache[signal_date]
        future_cross = ""
        if event_type.startswith("STRENGTH_") or event_type == "AGE2_BASELINE":
            for cross_date in cross_dates:
                if (cross_date >= signal_date and
                        (pd.Timestamp(cross_date) - pd.Timestamp(signal_date)).days
                        <= FUTURE_WEEKLY_CROSS_DAYS):
                    future_cross = cross_date
                    break
        future_open_date, future_open = "", np.nan
        if future_cross:
            future_open_date, future_open = _next_stock_open(
                daily_macd, future_cross, config["market_end"])
        entry_open = finite_num(outcome.get("Raw_Open"))
        price_advantage = (
            (future_open / entry_open - 1.0) * 100.0
            if math.isfinite(entry_open) and entry_open > 0
            and math.isfinite(future_open) else np.nan)
        row = _event_base_row(
            stock, membership, signal_date, snapshot, event_type)
        row.update({
            "Strategy_Label": strategy_label,
            "Strength_Threshold_pct": strength_threshold,
            "Cohort_Entry_Red_Age": cohort_age,
            "Common_Cohort_ID": cohort_id,
            "Diagnostic_Uses_Future_Day5_Survival": diagnostic_future_condition,
            **_v62_signal_fields(signal),
            "Signal_Rally_From_Red_Start_pct": finite_num(
                signal.get("Daily_Return_Since_Red_Start_pct")),
            "Signal_Rally_Above30": finite_num(
                signal.get("Daily_Return_Since_Red_Start_pct"))
                > STRENGTH_MAX_PRIOR_RALLY_PCT,
            "Strength_Quality_Pass": bool(
                str(signal.get("Daily_MACD_State", "")) == "红柱扩张"
                or finite_num(signal.get("Daily_MACD_Remaining_pct"))
                >= STRENGTH_MIN_REMAINING_PCT),
            "Future_Weekly_Cross25_Within42D": bool(future_cross),
            "Future_Weekly_Cross25_Date": future_cross,
            "Lead_Calendar_Days_to_Weekly_Cross": (
                float((pd.Timestamp(future_cross) - pd.Timestamp(signal_date)).days)
                if future_cross else np.nan),
            "Future_Weekly_Entry_Date": future_open_date,
            "Future_Weekly_Raw_Open": future_open,
            "Entry_Price_Advantage_vs_Weekly_pct": price_advantage,
            "Watch_Next_Strength_Threshold": _v62_next_threshold(
                finite_num(signal.get("Daily_Return_Since_Red_Start_pct"))),
        })
        if calculate_outcome:
            row.update({f"Entry_{key}": value for key, value in outcome.items()})
        rows.append(row)

    for _, signal in age2_candidates.iterrows():
        append_event(signal, "AGE2_BASELINE", "观察池内红柱第2日直接买")
    for threshold, signal in strength_candidates:
        append_event(
            signal, f"STRENGTH_{int(threshold)}",
            f"本轮日线上涨首次达到{int(threshold)}%", threshold)
    for age, cohort_id, signal in cohort_candidates:
        append_event(
            signal, f"COHORT_AGE{int(age)}", f"共同周期红柱第{int(age)}日",
            cohort_age=float(age), cohort_id=cohort_id,
            diagnostic_future_condition=True)

    macd_lookup = daily_macd.set_index(date_series)
    for _, signal in weekly_crosses.iterrows():
        signal_date = str(signal["trade_date"])
        if signal_date in macd_lookup.index:
            daily_signal = macd_lookup.loc[signal_date]
            if isinstance(daily_signal, pd.DataFrame):
                daily_signal = daily_signal.iloc[-1]
        else:
            before = daily_macd[date_series.le(signal_date)]
            daily_signal = before.iloc[-1] if not before.empty else pd.Series(dtype=object)
        weekly_signal = daily_signal.copy()
        weekly_signal["Setup_Weekly_Date"] = signal_date
        weekly_signal["Setup_Weekly_K"] = signal.get("K")
        weekly_signal["Setup_Weekly_D"] = signal.get("D")
        weekly_signal["Setup_Weekly_KD_Spread"] = signal.get("KD_Spread")
        weekly_signal["Setup_Weekly_K_Change_1W"] = signal.get("K_Change_1W")
        weekly_signal["Setup_Prior_Below25_Streak"] = signal.get("Prior_Below25_Streak")
        weekly_signal["Setup_Close_to_MA20_pct"] = signal.get("Close_to_MA20_pct")
        weekly_signal["Setup_MA20_Slope_4W_pct"] = signal.get("MA20_Slope_4W_pct")
        weekly_signal["Setup_Volume_Ratio_5W"] = signal.get("Volume_Ratio_5W")
        weekly_signal["Setup_Week_Return_pct"] = signal.get("Signal_Week_Return_pct")
        weekly_signal["trade_date"] = signal_date
        append_event(weekly_signal, "WEEKLY_CROSS25", "周线K上穿25旧买点")

    for _, signal in latest_watch.iterrows():
        append_event(
            signal, "LIVE_WATCH", "最新交易日周线K15～25观察池",
            calculate_outcome=False)
    return rows, rejects, {}


def v62_strategy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    order = [
        "AGE2_BASELINE", "STRENGTH_3", "STRENGTH_5", "STRENGTH_8",
        "STRENGTH_10", "WEEKLY_CROSS25"]
    work = frame[frame["Event_Type"].isin(order)].copy()
    if work.empty:
        return pd.DataFrame()
    result = timing_outcome_summary(
        work, ["Event_Type", "Strategy_Label"], "V6.2可执行策略")
    if result.empty:
        return result
    result["_order"] = result["Event_Type"].map({name: i for i, name in enumerate(order)})
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def v62_coverage_calendar(
        open_dates: list[str], start_date: str, end_date: str,
        mature: pd.DataFrame) -> pd.DataFrame:
    dates = [value for value in open_dates if start_date <= value <= end_date]
    calendar = pd.DataFrame({"trade_date": dates})
    calendar["Signal_Week"] = pd.to_datetime(
        calendar["trade_date"], format="%Y%m%d").dt.to_period("W-FRI").astype(str)
    weeks = calendar.groupby("Signal_Week")["trade_date"].max().rename(
        "Week_Last_Trading_Date").reset_index()
    event_types = [
        "AGE2_BASELINE", "STRENGTH_3", "STRENGTH_5", "STRENGTH_8",
        "STRENGTH_10", "WEEKLY_CROSS25"]
    for event_type in event_types:
        counts = mature[mature["Event_Type"].eq(event_type)].groupby(
            "Signal_Week").size()
        weeks[event_type] = weeks["Signal_Week"].map(counts).fillna(0).astype(int)
    return weeks


def v62_coverage_summary(calendar: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_type in [column for column in calendar.columns if column not in (
            "Signal_Week", "Week_Last_Trading_Date")]:
        counts = numeric(calendar, event_type)
        rows.append({
            "Event_Type": event_type, "总事件": int(counts.sum()),
            "有信号周": int(counts.gt(0).sum()),
            "空窗周": int(counts.eq(0).sum()),
            "最长连续空窗周": max_empty_run(counts),
            "每周信号均值": counts.mean(), "每周信号中位": counts.median(),
            "单周最多": int(counts.max()),
            "1至5只候选周": int(counts.between(1, 5).sum()),
            "6至20只候选周": int(counts.between(6, 20).sum()),
            "超过20只候选周": int(counts.gt(20).sum()),
        })
    return pd.DataFrame(rows)


def v62_common_cycle_summary(
        cohort_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cohort_history.empty:
        return pd.DataFrame(), pd.DataFrame()
    valid = cohort_history[
        true_mask(cohort_history, "Entry_Tradable")
        & true_mask(cohort_history, "Entry_Has_40D")].copy()
    counts = valid.groupby("Common_Cohort_ID")["Cohort_Entry_Red_Age"].nunique()
    ids = counts[counts.eq(len(COHORT_RED_AGES))].index
    common = add_timing_labels(valid[valid["Common_Cohort_ID"].isin(ids)].copy())
    if common.empty:
        return pd.DataFrame(), common
    summary = timing_outcome_summary(
        common, ["Cohort_Entry_Red_Age", "Strategy_Label"],
        "仅作诊断：同一批持续到第5日的红柱周期")
    if summary.empty:
        return summary, common
    return summary.sort_values("Cohort_Entry_Red_Age"), common


def v62_all_threshold_common_summary(
        strength: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strength.empty:
        return pd.DataFrame(), pd.DataFrame()
    strength = strength.copy()
    strength["Strength_Cycle_ID"] = (
        strength["ts_code"].astype(str) + "|"
        + numeric(strength, "Daily_MACD_Cycle").fillna(-1).astype(int).astype(str))
    counts = strength.groupby("Strength_Cycle_ID")["Strength_Threshold_pct"].nunique()
    ids = counts[counts.eq(len(STRENGTH_THRESHOLDS))].index
    common = strength[strength["Strength_Cycle_ID"].isin(ids)].copy()
    if common.empty:
        return pd.DataFrame(), common
    summary = timing_outcome_summary(
        common, ["Strength_Threshold_pct", "Strategy_Label"],
        "同一批最终达到10%的周期")
    if summary.empty:
        return summary, common
    return summary.sort_values("Strength_Threshold_pct"), common


def legacy_main_v62() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    streamlit_version = str(getattr(st, "__version__", "unknown"))
    st.caption(
        f"{UI_PATCH}｜可执行策略与未来条件诊断严格分离。｜Streamlit {streamlit_version}")
    with st.expander("V6.2验证口径", expanded=True):
        st.markdown(f"""
- **周线观察池**：当日日线收盘时可见的最近完整周，N=6、M=3的SKDJ K位于{EARLY_WEEKLY_K_MIN:g}～{EARLY_WEEKLY_K_MAX:g}。
- **强度确认买点**：日线MACD仍为红柱，且红柱正在扩张或剩余强度≥{STRENGTH_MIN_REMAINING_PCT:g}%；分别在本轮股价首次达到+3%、+5%、+8%、+10%的收盘日发出信号，下一市场交易日开盘买入。
- **实时基准**：观察池内日线MACD第2根红柱直接买，不要求以后持续到第5根，也不要求以后周线突破。
- **公平等待实验**：仅取第2根红柱时已在观察池、并且同一红柱周期实际持续到第5日的共同样本，分别模拟第2/3/4/5日买入。该实验使用未来存活信息，**只用于研究，不是策略**。
- **周线对照**：保留周线K上穿25旧买点，并继续输出“红柱扩张＋本轮已上涨10%～30%”高优先级特征。
- **防泄漏**：未来是否上穿周线25只作为结果字段；不会决定任何强度确认事件是否入池。历史结果与历史截止日以后的观察名单完全分开。
- **判卷**：买入后40个市场交易日；S/A/B分别为先到+30/+20/+10且先于-10，其余为F；同日冲突保守按-10先。
- **缓存**：交易日历和股票基础资料缓存72小时；行业成员缓存7天；逐股票行情文件不设自动过期。应用重启、休眠迁移或重新部署仍可能清空Streamlit实例的内存/临时磁盘。
""")

    with st.sidebar:
        st.header("运行参数")
        backtest_days = st.number_input(
            "回测交易日数", 100, 1000, 500, 50, key="v62_days")
        min_price = st.number_input(
            "最低股价（元）", 10.0, 20.0, 10.0, 10.0,
            format="%.0f", key="v62_min_price")
        min_mv = st.number_input(
            "最低流通市值（亿元）", 50.0, 100.0, 50.0, 50.0,
            format="%.0f", key="v62_min_mv")
        signal_end_date = st.date_input(
            "历史买入信号截止（40日判卷）", date(2026, 6, 5),
            key="v62_signal_end")
        market_end_date = st.date_input(
            "最新信号观察截止（默认今天）", date.today(),
            key="v62_market_end")
        pause = st.number_input(
            "接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v62_pause")
        use_cache = st.checkbox("复用行情和72小时基础缓存", True, key="v62_cache")
        st.caption(
            "72小时是TTL上限，不是在线保证；应用被平台重启后内存缓存会消失。"
            "逐股票行情文件和检查点没有TTL。")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005,
            format="%.3f", key="v62_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01,
            format="%.3f", key="v62_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001,
            format="%.3f", key="v62_transfer")
        if st.button("清除V6.2结果和运行状态", key="v62_clear"):
            shutil.rmtree(RESULT_DIR, ignore_errors=True)
            shutil.rmtree(JOB_DIR, ignore_errors=True)
            st.success("结果和运行状态已清除；逐股票检查点与行情缓存保留。")

    request_payload = {
        "version": VERSION, "days": int(backtest_days),
        "signal_end": signal_end_date.strftime("%Y%m%d"),
        "market_end": market_end_date.strftime("%Y%m%d"),
        "min_price": float(min_price), "min_mv": float(min_mv),
        "commission": float(commission_pct), "stamp": float(stamp_duty_pct),
        "transfer": float(transfer_fee_pct),
        "watch_k": [EARLY_WEEKLY_K_MIN, EARLY_WEEKLY_K_MAX],
        "strength_thresholds": list(STRENGTH_THRESHOLDS),
        "strength_remaining": STRENGTH_MIN_REMAINING_PCT,
    }
    request_signature = stable_signature(request_payload)
    result_path = os.path.join(RESULT_DIR, f"{request_signature}.zip")
    result_name = (
        f"weekly_skdj_strength_confirmation_audit_v6_2_"
        f"{int(backtest_days)}d_p{int(min_price)}_mv{int(min_mv)}.zip")
    completed_available = False
    if os.path.exists(result_path):
        try:
            with open(result_path, "rb") as handle:
                saved_result = handle.read()
            completed_available = True
            clear_job_active(request_signature)
            st.success("发现相同参数的已完成结果，可直接下载。")
            render_download(saved_result, result_name, f"v62_saved_{request_signature}")
        except Exception as exc:
            st.warning(f"已保存结果读取失败：{exc}")

    secret_token = configured_tushare_token()
    if secret_token:
        token = secret_token
        st.caption("已从Streamlit Secrets读取TUSHARE_TOKEN。")
    else:
        token = st.text_input("Tushare Token", type="password", key="v62_token")
    job_active = is_job_active(request_signature)
    left, right = st.columns(2)
    with left:
        start_clicked = st.button(
            "开始/重新运行V6.2", type="primary", key="v62_run")
    with right:
        stop_clicked = st.button(
            "停止自动续跑", disabled=not job_active, key="v62_stop")
    if stop_clicked:
        clear_job_active(request_signature)
        st.success("已停止；逐股票检查点保留。")
        return
    if start_clicked:
        if market_end_date <= signal_end_date:
            st.error("最新观察截止必须晚于历史信号截止")
            return
        mark_job_active(request_signature)
        job_active = True
    if not token:
        st.info("请输入Token；任务启动后若页面重连，会从逐股票检查点自动续跑。")
        return
    if not job_active:
        st.caption(
            "点击开始运行。" if not completed_available
            else "相同参数结果已可下载；如需覆盖请点击重新运行。")
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
    except Exception as exc:
        st.error(f"确定{int(backtest_days)}个交易日窗口失败：{exc}")
        return
    data_start = (
        pd.Timestamp(signal_start).date() - timedelta(weeks=WARMUP_WEEKS, days=7)
    ).strftime("%Y%m%d")
    try:
        with st.spinner("加载交易日历和历史科技池..."):
            open_dates = load_trade_calendar(data_start, market_end)
            extended_end = (market_end_date + timedelta(days=7)).strftime("%Y%m%d")
            week_last_map = complete_week_last_dates(
                load_trade_calendar(data_start, extended_end))
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return
    if not open_dates:
        st.error("区间内没有市场交易日")
        return
    latest_market_date = max(value for value in open_dates if value <= market_end)
    open_pos = {day: position for position, day in enumerate(open_dates)}
    config = {
        "signal_start": signal_start, "signal_end": signal_end,
        "event_signal_end": market_end, "data_start": data_start,
        "market_end": market_end, "latest_market_date": latest_market_date,
        "min_price": float(min_price), "min_mv": float(min_mv),
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct),
        "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct),
    }
    run_signature = stable_signature({"version": VERSION, **config})
    period_index = build_period_index(memberships)
    active_codes = {
        code for code, code_periods in period_index.items()
        if periods_overlap(code_periods, signal_start, market_end)}
    stocks = stock_basic[stock_basic["ts_code"].isin(active_codes)].copy()
    stocks = stocks[
        ~stocks["list_date"].gt(market_end)
        & ~stocks["delist_date"].lt(data_start)
    ].sort_values("ts_code").reset_index(drop=True)

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
                    rows, stock_rejects, stock_breadth = analyze_stock_v62(
                        stock, period_index.get(code, []), daily, cached_basic,
                        storage_path, week_last_map, open_dates, open_pos,
                        config, bool(use_cache), float(pause))
                    event_rows.extend(rows)
                    merge_counts(rejects, stock_rejects)
                    save_checkpoint(
                        run_signature, code, rows, stock_rejects, stock_breadth)
                except Exception as exc:
                    failures += 1
                    record_error(f"逐股票分析失败 {code}: {exc}")
        processed = number + 1
        now = time.monotonic()
        if (processed == 1 or now - last_update >= UI_HEARTBEAT_SECONDS
                or processed == len(stocks)):
            progress.progress(
                processed / max(len(stocks), 1),
                text=f"已处理{processed}/{len(stocks)}只股票，最近{code}")
            status.caption(
                f"事件{len(event_rows)}；检查点{checkpoint_hits}；"
                f"行情缓存{price_cache_hits}；失败{failures}")
            last_update = now
    progress.empty()
    status.empty()
    if stopped:
        st.warning("任务已停止，逐股票检查点已保留。")
        return

    events_all = pd.DataFrame(event_rows)
    if events_all.empty:
        st.error("本区间没有生成V6.2事件。")
        return
    events_all = events_all.sort_values(
        ["Signal_Date", "Event_Type", "ts_code"]).reset_index(drop=True)
    live_watch = events_all[events_all["Event_Type"].eq("LIVE_WATCH")].copy()
    research_events = events_all[~events_all["Event_Type"].eq("LIVE_WATCH")].copy()
    history = research_events[
        research_events["Signal_Date"].astype(str).le(signal_end)].copy()
    observation = research_events[
        research_events["Signal_Date"].astype(str).gt(signal_end)
        & research_events["Signal_Date"].astype(str).le(latest_market_date)
    ].copy()
    mature = history[
        true_mask(history, "Entry_Tradable")
        & true_mask(history, "Entry_Has_40D")].copy()
    if mature.empty:
        st.error("存在信号，但没有可成交且走完40个市场交易日的成熟事件。")
        return
    mature = add_timing_labels(mature)
    strength = mature[mature["Event_Type"].astype(str).str.startswith(
        "STRENGTH_")].copy()
    weekly = mature[mature["Event_Type"].eq("WEEKLY_CROSS25")].copy()
    if strength.empty or weekly.empty:
        st.error("强度确认事件或周线旧买点对照为空。")
        return

    strategy_summary = v62_strategy_summary(mature)
    calendar = v62_coverage_calendar(
        open_dates, signal_start, signal_end, mature)
    coverage_summary = v62_coverage_summary(calendar)
    common_strength_summary, common_strength = v62_all_threshold_common_summary(
        strength)
    cohort_history = history[history["Event_Type"].astype(str).str.startswith(
        "COHORT_AGE")].copy()
    common_age_summary, common_age = v62_common_cycle_summary(cohort_history)

    strength["Signal_Red_Age_Band"] = pd.cut(
        numeric(strength, "Daily_MACD_Red_Age"),
        [0, 2, 3, 4, 5, 7, 10, 15, np.inf],
        labels=["1～2", "第3日", "第4日", "第5日", "6～7", "8～10", "11～15", ">15"],
        right=True).astype(str)
    strength["Signal_K_Band"] = pd.cut(
        numeric(strength, "Signal_K"), [15, 20, 25],
        labels=["15～20", "20～25"], include_lowest=True).astype(str)
    strength["Future_Cross_Group"] = np.where(
        true_mask(strength, "Future_Weekly_Cross25_Within42D"),
        "六周内后来上穿25", "六周内没有上穿25")
    strength["Above30_Group"] = np.where(
        true_mask(strength, "Signal_Rally_Above30"),
        "触发日已超过30%", "触发日不超过30%")
    strength_age_audit = timing_outcome_summary(
        strength, ["Strength_Threshold_pct", "Signal_Red_Age_Band"],
        "强度阈值与触发日龄")
    strength_k_audit = timing_outcome_summary(
        strength, ["Strength_Threshold_pct", "Signal_K_Band"],
        "强度阈值与周线K位置")
    future_cross_audit = timing_outcome_summary(
        strength, ["Strength_Threshold_pct", "Future_Cross_Group"],
        "未来周线确认仅作审计")
    above30_audit = timing_outcome_summary(
        strength, ["Strength_Threshold_pct", "Above30_Group"],
        "触发日是否已超过30%")

    weekly["Weekly_Momentum_Tier"] = "普通周线确认"
    weekly.loc[
        weekly["Daily_MACD_State"].astype(str).eq("红柱扩张")
        & numeric(weekly, "Daily_Return_Since_Red_Start_pct").between(10, 30),
        "Weekly_Momentum_Tier"] = "高优先级_红柱扩张且已涨10至30"
    weekly.loc[
        numeric(weekly, "Daily_Return_Since_Red_Start_pct").gt(30),
        "Weekly_Momentum_Tier"] = "风险组_已涨超过30"
    weekly_momentum_audit = timing_outcome_summary(
        weekly, ["Weekly_Momentum_Tier"], "V6.1发现复核")

    mature["Calendar_Year"] = mature["Signal_Date"].astype(str).str[:4]
    year_audit = timing_outcome_summary(
        mature[mature["Event_Type"].isin([
            "AGE2_BASELINE", "STRENGTH_3", "STRENGTH_5", "STRENGTH_8",
            "STRENGTH_10", "WEEKLY_CROSS25"])],
        ["Event_Type", "Calendar_Year"], "逐年稳定性")

    recent_start = (pd.Timestamp(latest_market_date) - pd.Timedelta(days=14)).strftime("%Y%m%d")
    recent_strength = observation[
        observation["Event_Type"].astype(str).str.startswith("STRENGTH_")
        & observation["Signal_Date"].astype(str).between(recent_start, latest_market_date)
    ].copy()
    recent_age2 = observation[
        observation["Event_Type"].eq("AGE2_BASELINE")
        & observation["Signal_Date"].astype(str).between(recent_start, latest_market_date)
    ].copy()
    recent_weekly = observation[
        observation["Event_Type"].eq("WEEKLY_CROSS25")
        & observation["Signal_Date"].astype(str).between(recent_start, latest_market_date)
    ].copy()

    run_summary = pd.DataFrame([{
        "程序版本": VERSION, "正式信号开始": signal_start,
        "历史信号截止": signal_end, "行情观察截止": market_end,
        "最新市场交易日": latest_market_date,
        "成熟强度确认事件": len(strength),
        "成熟红柱第2日基准": int(mature["Event_Type"].eq("AGE2_BASELINE").sum()),
        "成熟周线确认事件": len(weekly),
        "共同达到全部强度阈值周期": common_strength.get(
            "Strength_Cycle_ID", pd.Series(dtype=str)).nunique(),
        "共同持续至第5红柱周期": common_age.get(
            "Common_Cohort_ID", pd.Series(dtype=str)).nunique(),
        "最新观察池股票": len(live_watch),
        "最近14日强度信号": len(recent_strength),
        "最近14日红柱第2日信号": len(recent_age2),
        "最近14日周线确认信号": len(recent_weekly),
        "处理股票数": len(stocks), "检查点恢复": checkpoint_hits,
        "行情缓存命中": price_cache_hits, "失败股票": failures,
    }])
    definitions = pd.DataFrame([
        ("观察池", f"最近完整周N=6 SKDJ K在{EARLY_WEEKLY_K_MIN:g}～{EARLY_WEEKLY_K_MAX:g}"),
        ("强度确认", "日线红柱扩张或剩余≥75%，本轮涨幅首次达到3/5/8/10%"),
        ("基准", "观察池内日线MACD第2根红柱，次日开盘买入"),
        ("共同周期诊断", "仅取实际持续到第5红柱的相同周期比较第2/3/4/5日；含未来条件，不可实盘"),
        ("周线对照", "N=6周线K由25下方上穿25，下一市场交易日开盘"),
        ("30%风险", "触发日涨幅超过30%只分组审计，本版不提前删除样本"),
        ("判卷", "40个市场交易日；S/A/B为先到+30/+20/+10且先于-10，否则F"),
        ("内存缓存", "交易日历和股票基础资料TTL=72小时；行业成员TTL=7天"),
        ("磁盘缓存", "逐股票行情文件与逐股票检查点不设TTL；实例重启/重新部署可能清空临时磁盘"),
    ], columns=["项目", "定义"])
    cache_policy = pd.DataFrame([
        ("交易日历", "Streamlit内存", "72小时", "进程/实例重启可能提前丢失"),
        ("股票基础资料", "Streamlit内存", "72小时", "进程/实例重启可能提前丢失"),
        ("申万科技行业成员", "Streamlit内存", "7天", "进程/实例重启可能提前丢失"),
        ("逐股票日线与daily_basic", "应用临时磁盘", "不自动过期", "重新部署或平台迁移可能清空"),
        ("逐股票分析检查点", "应用临时磁盘", "不自动过期", "版本/参数签名变化会使用新检查点"),
        ("已完成结果ZIP", "应用临时磁盘", "不自动过期", "重新部署或平台迁移可能清空"),
    ], columns=["对象", "位置", "设定时长", "实际边界"])
    rejection_audit = pd.DataFrame([
        {"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())])
    metadata = pd.DataFrame([{
        "程序版本": VERSION, "SKDJ_N": EARLY_PRIMARY_N, "SKDJ_M": SKDJ_M,
        "回测交易日": int(backtest_days), "预热周": WARMUP_WEEKS,
        "历史信号开始": signal_start, "历史信号截止": signal_end,
        "行情截止": market_end, "最低股价": float(min_price),
        "最低流通市值亿元": float(min_mv),
        "观察池K下限": EARLY_WEEKLY_K_MIN,
        "观察池K上限": EARLY_WEEKLY_K_MAX,
        "强度阈值": "/".join(str(int(v)) for v in STRENGTH_THRESHOLDS),
        "MACD最低剩余强度%": STRENGTH_MIN_REMAINING_PCT,
        "内存缓存小时": CACHE_TTL_SECONDS / 3600,
        "Streamlit": streamlit_version,
    }])

    detail_columns = [
        "Event_Type", "Strategy_Label", "ts_code", "name", "Signal_Date",
        "Signal_Week", "SW_L1", "SW_L2", "Raw_Close", "Circ_MV_Billion",
        "Setup_Weekly_Date", "Signal_K", "Signal_D", "Signal_KD_Spread",
        "Daily_MACD_State", "Daily_MACD_Red_Age",
        "Daily_MACD_Remaining_pct", "Daily_MACD_Retention_pct",
        "Signal_Rally_From_Red_Start_pct", "Signal_Rally_Above30",
        "Strength_Threshold_pct", "Strength_Quality_Pass",
        "Cohort_Entry_Red_Age", "Common_Cohort_ID",
        "Diagnostic_Uses_Future_Day5_Survival",
        "Future_Weekly_Cross25_Within42D", "Future_Weekly_Cross25_Date",
        "Lead_Calendar_Days_to_Weekly_Cross",
        "Entry_Price_Advantage_vs_Weekly_pct", "Entry_Date", "Entry_Raw_Open",
        "Entry_MFE_Net_pct", "Entry_MAE_Raw_pct",
        "Entry_Close_Return_Net_pct", "Explosion_Class_40D",
        "Entry_First_Hit_10_vs_Minus10_40D",
        "Entry_First_Hit_20_vs_Minus10_40D",
        "Entry_First_Hit_30_vs_Minus10_40D",
    ]
    live_columns = [
        "Event_Type", "Strategy_Label", "ts_code", "name", "Signal_Date",
        "Signal_Week", "SW_L1", "SW_L2", "Raw_Close", "Circ_MV_Billion",
        "Setup_Weekly_Date", "Signal_K", "Signal_D", "Signal_KD_Spread",
        "Daily_MACD_State", "Daily_MACD_Red_Age",
        "Daily_MACD_Remaining_pct", "Daily_MACD_Retention_pct",
        "Signal_Rally_From_Red_Start_pct", "Strength_Threshold_pct",
        "Watch_Next_Strength_Threshold",
    ]
    history_strength_export = strength[[
        column for column in detail_columns if column in strength.columns]].copy()
    history_weekly_export = weekly[[
        column for column in detail_columns + ["Weekly_Momentum_Tier"]
        if column in weekly.columns]].copy()
    common_strength_export = common_strength[[
        column for column in detail_columns + ["Strength_Cycle_ID"]
        if column in common_strength.columns]].copy()
    common_age_export = common_age[[
        column for column in detail_columns if column in common_age.columns]].copy()
    recent_strength_export = recent_strength[[
        column for column in live_columns if column in recent_strength.columns]].copy()
    recent_age2_export = recent_age2[[
        column for column in live_columns if column in recent_age2.columns]].copy()
    recent_weekly_export = recent_weekly[[
        column for column in live_columns if column in recent_weekly.columns]].copy()
    live_watch_export = live_watch[[
        column for column in live_columns if column in live_watch.columns]].copy()
    files = {
        "01_run_summary_v6_2.csv": run_summary,
        "02_experiment_definitions_v6_2.csv": definitions,
        "03_executable_strategy_outcomes_v6_2.csv": strategy_summary,
        "04_strategy_coverage_summary_v6_2.csv": coverage_summary,
        "05_weekly_signal_calendar_v6_2.csv": calendar,
        "06_same_cycles_reaching_all_thresholds_v6_2.csv": common_strength_summary,
        "07_same_cycles_day2_to_day5_v6_2.csv": common_age_summary,
        "08_strength_threshold_red_age_audit_v6_2.csv": strength_age_audit,
        "09_strength_threshold_weekly_k_audit_v6_2.csv": strength_k_audit,
        "10_strength_future_weekly_cross_audit_v6_2.csv": future_cross_audit,
        "11_strength_above30_risk_audit_v6_2.csv": above30_audit,
        "12_weekly_momentum_priority_recheck_v6_2.csv": weekly_momentum_audit,
        "13_year_stability_audit_v6_2.csv": year_audit,
        "14_historical_strength_event_detail_v6_2.csv": history_strength_export,
        "15_common_threshold_cycle_detail_v6_2.csv": common_strength_export,
        "16_common_day2_to_day5_cycle_detail_v6_2.csv": common_age_export,
        "17_historical_weekly_control_detail_v6_2.csv": history_weekly_export,
        "18_recent_14d_strength_candidates_v6_2.csv": recent_strength_export,
        "19_recent_14d_age2_candidates_v6_2.csv": recent_age2_export,
        "20_recent_14d_weekly_cross_candidates_v6_2.csv": recent_weekly_export,
        "21_latest_market_day_watch_pool_v6_2.csv": live_watch_export,
        "22_cache_policy_v6_2.csv": cache_policy,
        "23_rejection_audit_v6_2.csv": rejection_audit,
        "24_metadata_v6_2.csv": metadata,
        "25_api_errors_v6_2.csv": pd.DataFrame({"错误": API_ERRORS}),
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
        f"完成：成熟强度确认事件{len(strength)}个；"
        f"共同达到3/5/8/10%的可配对周期"
        f"{common_strength.get('Strength_Cycle_ID', pd.Series(dtype=str)).nunique()}个；"
        f"共同持续到第5红柱的诊断周期"
        f"{common_age.get('Common_Cohort_ID', pd.Series(dtype=str)).nunique()}个；"
        f"结果{'已保存' if persisted else '仅当前页面可下载'}。")
    st.subheader("可执行策略：第2红柱、强度3/5/8/10与周线确认")
    render_plain_table(strategy_summary, 20)
    st.subheader("同一批最终达到10%的周期：等待更高强度的真实代价")
    render_plain_table(common_strength_summary, 20)
    st.subheader("严格同周期诊断：第2/3/4/5红柱")
    st.warning("本表以未来能持续到第5日作为共同样本条件，只能研究，不能作为实盘成绩。")
    render_plain_table(common_age_summary, 20)
    st.subheader("覆盖率与信号拥挤度")
    render_plain_table(coverage_summary, 20)
    st.subheader(f"最新交易日观察池：{latest_market_date}")
    st.caption("这里只表示周线K处于15～25；下一强度阈值字段说明日线还在等待什么。")
    render_plain_table(live_watch_export.sort_values(
        ["Daily_MACD_State", "Signal_Rally_From_Red_Start_pct"],
        ascending=[True, False]), 200)
    st.subheader("最近14日真实强度确认信号")
    render_plain_table(recent_strength_export.sort_values(
        ["Signal_Date", "Strength_Threshold_pct"], ascending=[False, False]), 200)
    st.subheader("运行摘要与缓存口径")
    render_plain_table(run_summary, 10)
    render_plain_table(cache_policy, 10)
    st.caption("结果ZIP共25个CSV；实时候选不含未来等级，共同周期诊断不进入可执行策略。")
    render_download(result_zip, result_name, f"v62_current_{request_signature}")


def _v63_class_from_outcome(outcome: dict[str, Any]) -> str:
    row = pd.Series({f"Entry_{key}": value for key, value in outcome.items()})
    return outcome_explosion_class(row)


def _v63_deadline_market_date(
        signal_date: str, deadline_days: int, open_dates: list[str],
        market_end: str) -> str:
    target = (pd.Timestamp(signal_date) + pd.Timedelta(
        days=int(deadline_days))).strftime("%Y%m%d")
    position = bisect.bisect_left(open_dates, target)
    if position >= len(open_dates):
        return ""
    value = str(open_dates[position])
    return value if value <= market_end else ""


def v64_cross_state(signal: pd.Series) -> str:
    """Classify only information observable at the weekly-cross close."""
    rally = finite_num(signal.get("Daily_Return_Since_Red_Start_pct"))
    state = str(signal.get("Daily_MACD_State", ""))
    if math.isfinite(rally) and rally > 30.0:
        return V64_CROSS_OVERHEATED
    if (state == "红柱扩张" and math.isfinite(rally)
            and 10.0 <= rally <= 30.0):
        return V64_CROSS_HIGH
    return V64_CROSS_ORDINARY


def v64_original_position_exit_after_cross(
        daily_macd: pd.DataFrame, entry_outcome: dict[str, Any],
        cross_date: str, open_pos: dict[str, int], config: dict[str, Any],
        remaining_threshold: float | None = None) -> dict[str, Any]:
    """Exit the original early position using decisions known after cross.

    ``remaining_threshold=None`` means exit at the first open after the weekly
    confirmation.  Otherwise, start observing at the cross close and exit at
    the next open after MACD turns green or remaining red-column strength falls
    to the requested threshold.  If neither happens, use the original 40-day
    endpoint.  No pre-confirmation daily state is allowed to trigger an exit.
    """
    out: dict[str, Any] = {
        "Has_Result": False, "Decision_Date": "", "Exit_Date": "",
        "Reason": "", "Return_Net_pct": np.nan, "MFE_Net_pct": np.nan,
        "MAE_Raw_pct": np.nan, "Hold_Market_Days": np.nan,
    }
    if not to_bool(entry_outcome.get("Has_40D")):
        out["Reason"] = "原提前买点没有40日结果"
        return out
    entry_date = str(entry_outcome.get("Date", ""))
    end_date = str(entry_outcome.get("End_Date_40D", ""))
    raw_entry = finite_num(entry_outcome.get("Raw_Open"))
    if (not entry_date or not end_date or not cross_date
            or cross_date > end_date or not math.isfinite(raw_entry)
            or raw_entry <= 0):
        out["Reason"] = "确认日或原买入数据无效"
        return out
    decision_path = daily_macd[
        daily_macd["trade_date"].astype(str).between(cross_date, end_date)
    ].copy().sort_values("trade_date")
    if decision_path.empty:
        out["Reason"] = "确认后没有日线数据"
        return out
    if remaining_threshold is None:
        decision_date = cross_date
        reason = "周线确认后下一开盘退出"
    else:
        hist = numeric(decision_path, "Daily_MACD_Hist")
        remaining = numeric(decision_path, "Daily_MACD_Remaining_pct")
        retention = numeric(decision_path, "Daily_MACD_Retention_pct")
        triggered = decision_path[
            hist.le(0)
            | (hist.gt(0) & remaining.le(float(remaining_threshold))
               & retention.lt(100.0))
        ]
        if triggered.empty:
            decision_date = end_date
            reason = "确认后未触发MACD退出_40日期末"
        else:
            decision = triggered.iloc[0]
            decision_date = str(decision["trade_date"])
            reason = (
                "确认后红柱翻绿" if finite_num(
                    decision.get("Daily_MACD_Hist")) <= 0
                else f"确认后红柱剩余≤{int(remaining_threshold)}%")
    if decision_date == end_date:
        end_row = daily_macd[
            daily_macd["trade_date"].astype(str).eq(end_date)]
        if end_row.empty:
            out["Reason"] = "40日期末行情缺失"
            return out
        exit_date = end_date
        raw_exit = finite_num(end_row.iloc[-1].get("close"))
    else:
        exit_date, raw_exit = _next_stock_open(
            daily_macd, decision_date, end_date)
        if not exit_date or not math.isfinite(raw_exit):
            end_row = daily_macd[
                daily_macd["trade_date"].astype(str).eq(end_date)]
            if end_row.empty:
                out["Reason"] = "退出后没有可交易开盘"
                return out
            exit_date = end_date
            raw_exit = finite_num(end_row.iloc[-1].get("close"))
            reason = f"{reason}_无后续开盘_40日期末"
    path = daily_macd[
        daily_macd["trade_date"].astype(str).between(entry_date, exit_date)
    ].copy().sort_values("trade_date")
    if path.empty or not math.isfinite(raw_exit):
        out["Reason"] = "原仓位退出路径不足"
        return out
    buy_factor, sell_factor = _cost_factors(config)
    net_entry = raw_entry * buy_factor
    high = finite_num(pd.to_numeric(path.get("high"), errors="coerce").max())
    low = finite_num(pd.to_numeric(path.get("low"), errors="coerce").min())
    out.update({
        "Has_Result": True, "Decision_Date": decision_date,
        "Exit_Date": exit_date, "Reason": reason,
        "Return_Net_pct": (raw_exit * sell_factor / net_entry - 1.0) * 100.0,
        "MFE_Net_pct": (
            (high * sell_factor / net_entry - 1.0) * 100.0
            if math.isfinite(high) else np.nan),
        "MAE_Raw_pct": (
            (low / raw_entry - 1.0) * 100.0
            if math.isfinite(low) else np.nan),
        "Hold_Market_Days": float(
            open_pos.get(exit_date, 0) - open_pos.get(entry_date, 0) + 1),
    })
    return out


def v63_confirmation_outcomes(
        daily_macd: pd.DataFrame, signal_date: str, ts_code: str,
        cross_dates: list[str], entry_outcome: dict[str, Any],
        deadline_days: int, open_dates: list[str], open_pos: dict[str, int],
        config: dict[str, Any]) -> dict[str, Any]:
    """Observable confirm-or-exit lifecycle after an early strength signal."""
    out: dict[str, Any] = {
        "Status": "数据不足", "Decision_Date": "", "Confirmed": False,
        "Cross_Date": "", "Cross_Delay_Calendar_Days": np.nan,
        "Exit_Date": "", "Exit_Reason": "", "Strategy_Has_Result": False,
        "Strategy_Return_Net_pct": np.nan, "Strategy_MFE_Net_pct": np.nan,
        "Strategy_MAE_Raw_pct": np.nan, "Strategy_Hold_Market_Days": np.nan,
        "Reentry_Signal_Date": "", "Reentry_Entry_Date": "",
        "Reentry_Tradable": False, "Reentry_Has_40D": False,
        "Reentry_Class_40D": "", "Reentry_MFE_Net_pct": np.nan,
        "Reentry_MAE_Raw_pct": np.nan,
        "Reentry_Close_Return_Net_pct": np.nan,
        "Combined_Has_Result": False, "Combined_Return_Net_pct": np.nan,
    }
    if not to_bool(entry_outcome.get("Tradable")):
        out["Status"] = "提前买点不可成交"
        return out
    decision_date = _v63_deadline_market_date(
        signal_date, deadline_days, open_dates, config["market_end"])
    if not decision_date:
        out["Status"] = "确认期限尚未到达"
        return out
    out["Decision_Date"] = decision_date
    confirmed_cross = next((
        value for value in cross_dates
        if signal_date <= value <= decision_date), "")
    if confirmed_cross:
        out.update({
            "Status": f"{int(deadline_days)}日内周线确认_继续持有",
            "Confirmed": True, "Cross_Date": confirmed_cross,
            "Cross_Delay_Calendar_Days": float(
                (pd.Timestamp(confirmed_cross) - pd.Timestamp(signal_date)).days),
            "Exit_Date": str(entry_outcome.get("End_Date_40D", "")),
            "Exit_Reason": "确认后持有至40日观察终点",
            "Strategy_Has_Result": to_bool(entry_outcome.get("Has_40D")),
            "Strategy_Return_Net_pct": finite_num(
                entry_outcome.get("Close_Return_Net_pct")),
            "Strategy_MFE_Net_pct": finite_num(entry_outcome.get("MFE_Net_pct")),
            "Strategy_MAE_Raw_pct": finite_num(entry_outcome.get("MAE_Raw_pct")),
            "Strategy_Hold_Market_Days": (
                40.0 if to_bool(entry_outcome.get("Has_40D")) else np.nan),
        })
        return out

    entry_date = str(entry_outcome.get("Date", ""))
    raw_entry = finite_num(entry_outcome.get("Raw_Open"))
    exit_date, raw_exit = _next_stock_open(
        daily_macd, decision_date, config["market_end"])
    if not exit_date or not math.isfinite(raw_exit):
        out["Status"] = "期限已到但没有后续可交易开盘"
        return out
    buy_factor, sell_factor = _cost_factors(config)
    net_entry = raw_entry * buy_factor
    path = daily_macd[
        daily_macd["trade_date"].astype(str).ge(entry_date)
        & daily_macd["trade_date"].astype(str).lt(exit_date)
    ].copy().sort_values("trade_date")
    high = finite_num(pd.to_numeric(path.get("high"), errors="coerce").max())
    low = finite_num(pd.to_numeric(path.get("low"), errors="coerce").min())
    early_return = (raw_exit * sell_factor / net_entry - 1.0) * 100.0
    out.update({
        "Status": f"{int(deadline_days)}日未确认_下一开盘退出",
        "Exit_Date": exit_date, "Exit_Reason": "周线K未在期限内上穿25",
        "Strategy_Has_Result": True,
        "Strategy_Return_Net_pct": early_return,
        "Strategy_MFE_Net_pct": (
            (high * sell_factor / net_entry - 1.0) * 100.0
            if math.isfinite(high) else np.nan),
        "Strategy_MAE_Raw_pct": (
            (low / raw_entry - 1.0) * 100.0
            if math.isfinite(low) and math.isfinite(raw_entry) else np.nan),
        "Strategy_Hold_Market_Days": float(
            open_pos.get(exit_date, 0) - open_pos.get(entry_date, 0) + 1),
    })

    last_reentry_date = (pd.Timestamp(signal_date) + pd.Timedelta(
        days=V63_REENTRY_WINDOW_DAYS)).strftime("%Y%m%d")
    reentry_signal = next((
        value for value in cross_dates
        if decision_date < value <= last_reentry_date), "")
    if not reentry_signal:
        return out
    reentry = daily_timing_outcomes(
        daily_macd, reentry_signal, ts_code, open_dates, open_pos, config)
    reentry_return = finite_num(reentry.get("Close_Return_Net_pct"))
    combined = (
        ((1.0 + early_return / 100.0) * (1.0 + reentry_return / 100.0) - 1.0)
        * 100.0
        if to_bool(reentry.get("Has_40D")) and math.isfinite(reentry_return)
        else np.nan)
    out.update({
        "Reentry_Signal_Date": reentry_signal,
        "Reentry_Entry_Date": str(reentry.get("Date", "")),
        "Reentry_Tradable": to_bool(reentry.get("Tradable")),
        "Reentry_Has_40D": to_bool(reentry.get("Has_40D")),
        "Reentry_Class_40D": _v63_class_from_outcome(reentry),
        "Reentry_MFE_Net_pct": finite_num(reentry.get("MFE_Net_pct")),
        "Reentry_MAE_Raw_pct": finite_num(reentry.get("MAE_Raw_pct")),
        "Reentry_Close_Return_Net_pct": reentry_return,
        "Combined_Has_Result": math.isfinite(combined),
        "Combined_Return_Net_pct": combined,
    })
    return out


def analyze_stock_v63(
        stock: pd.Series, periods: list[dict[str, str]], daily: pd.DataFrame,
        cached_basic: pd.DataFrame, storage_path: str,
        week_last_map: dict[pd.Timestamp, str], open_dates: list[str],
        open_pos: dict[str, int], config: dict[str, Any], use_cache: bool,
        api_pause: float
        ) -> tuple[list[dict[str, Any]], dict[str, int],
                   dict[str, dict[str, float]]]:
    """Generate only 3% early entries, weekly controls and the latest watch pool."""
    rejects: dict[str, int] = {}
    weekly_base = aggregate_complete_weekly(daily, week_last_map)
    if weekly_base.empty:
        return [], rejects, {}
    n6 = add_skdj(weekly_base, EARLY_PRIMARY_N)
    daily_macd = attach_latest_completed_weekly(add_daily_macd(daily), n6)
    event_end = str(config.get("event_signal_end", config["signal_end"]))
    dates = daily_macd["trade_date"].astype(str)
    watch_k = numeric(daily_macd, "Setup_Weekly_K").between(
        EARLY_WEEKLY_K_MIN, EARLY_WEEKLY_K_MAX, inclusive="both")
    positive = true_mask(daily_macd, "Daily_MACD_Positive")
    quality = (
        daily_macd["Daily_MACD_State"].astype(str).eq("红柱扩张")
        | numeric(daily_macd, "Daily_MACD_Remaining_pct").ge(
            STRENGTH_MIN_REMAINING_PCT))
    strength3 = daily_macd[
        dates.between(config["signal_start"], event_end)
        & watch_k & positive & quality
        & numeric(daily_macd, "Daily_Return_Since_Red_Start_pct").ge(
            V63_STRENGTH_THRESHOLD)
    ].copy().sort_values("trade_date")
    if not strength3.empty:
        strength3 = strength3.groupby(
            "Daily_MACD_Cycle", as_index=False, sort=False).first()
    weekly_crosses = n6[
        true_mask(n6, "K_Cross_25")
        & n6["trade_date"].astype(str).between(
            config["signal_start"], event_end)
    ].copy()
    latest_market_date = str(config.get("latest_market_date", config["market_end"]))
    latest_watch = daily_macd[
        dates.eq(latest_market_date) & watch_k
    ].copy().tail(1)
    estimated = len(strength3) + len(weekly_crosses) + len(latest_watch)
    if estimated == 0:
        return [], rejects, {}

    code = str(stock["ts_code"])
    basic = ensure_daily_basic(
        code, config["data_start"], config["market_end"], daily,
        cached_basic, storage_path, use_cache, api_pause)
    if basic.empty:
        rejects["存在提前信号但daily_basic缺失"] = estimated
        return [], rejects, {}
    cross_dates = weekly_crosses["trade_date"].astype(str).sort_values().tolist()
    outcome_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    def valid_context(signal_date: str) -> tuple[dict[str, str] | None, dict[str, float], str]:
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
        return membership, snapshot, reason

    def append_event(
            signal: pd.Series, event_type: str, label: str,
            calculate_outcome: bool = True) -> None:
        signal_date = str(signal["trade_date"])
        membership, snapshot, reason = valid_context(signal_date)
        if reason or membership is None:
            rejects[reason] = rejects.get(reason, 0) + 1
            return
        outcome: dict[str, Any] = {}
        if calculate_outcome:
            if signal_date not in outcome_cache:
                outcome_cache[signal_date] = daily_timing_outcomes(
                    daily_macd, signal_date, code, open_dates, open_pos, config)
            outcome = outcome_cache[signal_date]
        future_cross = ""
        if event_type == "STRENGTH_3":
            future_cross = next((
                value for value in cross_dates
                if value >= signal_date and (
                    pd.Timestamp(value) - pd.Timestamp(signal_date)).days
                <= FUTURE_WEEKLY_CROSS_DAYS), "")
        future_open_date, future_open = "", np.nan
        cross_state = V64_CROSS_NONE
        cross_macd_state = ""
        cross_rally = np.nan
        cross_outcome: dict[str, Any] = {}
        if future_cross:
            future_open_date, future_open = _next_stock_open(
                daily_macd, future_cross, config["market_end"])
            cross_frame = daily_macd[
                daily_macd["trade_date"].astype(str).eq(future_cross)]
            if cross_frame.empty:
                cross_frame = daily_macd[
                    daily_macd["trade_date"].astype(str).le(future_cross)]
            if not cross_frame.empty:
                cross_signal = cross_frame.iloc[-1]
                cross_state = v64_cross_state(cross_signal)
                cross_macd_state = str(
                    cross_signal.get("Daily_MACD_State", ""))
                cross_rally = finite_num(
                    cross_signal.get("Daily_Return_Since_Red_Start_pct"))
            if calculate_outcome:
                if future_cross not in outcome_cache:
                    outcome_cache[future_cross] = daily_timing_outcomes(
                        daily_macd, future_cross, code, open_dates, open_pos,
                        config)
                cross_outcome = outcome_cache[future_cross]
        entry_open = finite_num(outcome.get("Raw_Open"))
        price_advantage = (
            (future_open / entry_open - 1.0) * 100.0
            if math.isfinite(entry_open) and entry_open > 0
            and math.isfinite(future_open) else np.nan)
        row = _event_base_row(stock, membership, signal_date, snapshot, event_type)
        row.update({
            "Strategy_Label": label, **_v62_signal_fields(signal),
            "Signal_Rally_From_Red_Start_pct": finite_num(
                signal.get("Daily_Return_Since_Red_Start_pct")),
            "Future_Weekly_Cross25_Within42D": bool(future_cross),
            "Future_Weekly_Cross25_Date": future_cross,
            "Lead_Calendar_Days_to_Weekly_Cross": (
                float((pd.Timestamp(future_cross) - pd.Timestamp(signal_date)).days)
                if future_cross else np.nan),
            "Future_Weekly_Entry_Date": future_open_date,
            "Future_Weekly_Raw_Open": future_open,
            "Entry_Price_Advantage_vs_Weekly_pct": price_advantage,
            "Future_Cross_State": cross_state,
            "Future_Cross_Daily_MACD_State": cross_macd_state,
            "Future_Cross_Rally_From_Red_Start_pct": cross_rally,
            "Watch_Next_Strength_Threshold": _v62_next_threshold(
                finite_num(signal.get("Daily_Return_Since_Red_Start_pct"))),
        })
        if calculate_outcome:
            row.update({f"Entry_{key}": value for key, value in outcome.items()})
        if event_type == "STRENGTH_3" and calculate_outcome:
            if future_cross:
                row.update({
                    "CrossEntry_Date": str(cross_outcome.get("Date", "")),
                    "CrossEntry_Tradable": to_bool(
                        cross_outcome.get("Tradable")),
                    "CrossEntry_Has_40D": to_bool(
                        cross_outcome.get("Has_40D")),
                    "CrossEntry_Class_40D": _v63_class_from_outcome(
                        cross_outcome),
                    "CrossEntry_MFE_Net_pct": finite_num(
                        cross_outcome.get("MFE_Net_pct")),
                    "CrossEntry_MAE_Raw_pct": finite_num(
                        cross_outcome.get("MAE_Raw_pct")),
                    "CrossEntry_Close_Return_Net_pct": finite_num(
                        cross_outcome.get("Close_Return_Net_pct")),
                })
                immediate = v64_original_position_exit_after_cross(
                    daily_macd, outcome, future_cross, open_pos, config)
                row.update({
                    f"CrossImmediate_{key}": value
                    for key, value in immediate.items()})
                for threshold in V64_POST_CROSS_REMAINING:
                    exit_outcome = v64_original_position_exit_after_cross(
                        daily_macd, outcome, future_cross, open_pos, config,
                        remaining_threshold=threshold)
                    row.update({
                        f"CrossRemaining{int(threshold)}_{key}": value
                        for key, value in exit_outcome.items()})
            for deadline in V63_CONFIRM_DEADLINES:
                gate = v63_confirmation_outcomes(
                    daily_macd, signal_date, code, cross_dates, outcome,
                    deadline, open_dates, open_pos, config)
                row.update({f"Confirm{int(deadline)}_{key}": value
                            for key, value in gate.items()})
            confirm14 = {
                key.removeprefix("Confirm14_"): value
                for key, value in row.items() if key.startswith("Confirm14_")
            }
            timeout14 = (
                not to_bool(confirm14.get("Confirmed"))
                and "未确认_下一开盘退出" in str(
                    confirm14.get("Status", "")))
            selective = bool(
                timeout14 and future_cross
                and cross_state == V64_CROSS_HIGH)
            early_exit_return = finite_num(
                confirm14.get("Strategy_Return_Net_pct"))
            reentry_return = finite_num(
                cross_outcome.get("Close_Return_Net_pct"))
            selective_combined = (
                ((1.0 + early_exit_return / 100.0)
                 * (1.0 + reentry_return / 100.0) - 1.0) * 100.0
                if selective and to_bool(cross_outcome.get("Has_40D"))
                and math.isfinite(early_exit_return)
                and math.isfinite(reentry_return) else np.nan)
            row.update({
                "Selective_Reentry14_Qualified": selective,
                "Selective_Reentry14_Signal_Date": (
                    future_cross if selective else ""),
                "Selective_Reentry14_Entry_Date": (
                    str(cross_outcome.get("Date", "")) if selective else ""),
                "Selective_Reentry14_Has_40D": bool(
                    selective and to_bool(cross_outcome.get("Has_40D"))),
                "Selective_Reentry14_Class_40D": (
                    _v63_class_from_outcome(cross_outcome)
                    if selective else ""),
                "Selective_Reentry14_MFE_Net_pct": (
                    finite_num(cross_outcome.get("MFE_Net_pct"))
                    if selective else np.nan),
                "Selective_Reentry14_MAE_Raw_pct": (
                    finite_num(cross_outcome.get("MAE_Raw_pct"))
                    if selective else np.nan),
                "Selective_Reentry14_Close_Return_Net_pct": (
                    reentry_return if selective else np.nan),
                "Selective_Reentry14_Combined_Return_Net_pct": (
                    selective_combined),
            })
        rows.append(row)

    for _, signal in strength3.iterrows():
        append_event(signal, "STRENGTH_3", "日线上涨首次达到3%")

    macd_lookup = daily_macd.set_index(dates)
    for _, signal in weekly_crosses.iterrows():
        signal_date = str(signal["trade_date"])
        if signal_date in macd_lookup.index:
            daily_signal = macd_lookup.loc[signal_date]
            if isinstance(daily_signal, pd.DataFrame):
                daily_signal = daily_signal.iloc[-1]
        else:
            before = daily_macd[dates.le(signal_date)]
            daily_signal = before.iloc[-1] if not before.empty else pd.Series(dtype=object)
        weekly_signal = daily_signal.copy()
        weekly_signal["trade_date"] = signal_date
        weekly_signal["Setup_Weekly_Date"] = signal_date
        for target, source in (
                ("Setup_Weekly_K", "K"), ("Setup_Weekly_D", "D"),
                ("Setup_Weekly_KD_Spread", "KD_Spread"),
                ("Setup_Weekly_K_Change_1W", "K_Change_1W"),
                ("Setup_Prior_Below25_Streak", "Prior_Below25_Streak"),
                ("Setup_Close_to_MA20_pct", "Close_to_MA20_pct"),
                ("Setup_MA20_Slope_4W_pct", "MA20_Slope_4W_pct"),
                ("Setup_Volume_Ratio_5W", "Volume_Ratio_5W"),
                ("Setup_Week_Return_pct", "Signal_Week_Return_pct")):
            weekly_signal[target] = signal.get(source)
        append_event(weekly_signal, "WEEKLY_CROSS25", "周线K上穿25旧买点")
    for _, signal in latest_watch.iterrows():
        append_event(
            signal, "LIVE_WATCH", "最新交易日周线K15至25观察池",
            calculate_outcome=False)
    return rows, rejects, {}


def v63_attach_breadth_and_rank(
        events: pd.DataFrame, open_dates: list[str]) -> pd.DataFrame:
    result = events.copy()
    strength_mask = result["Event_Type"].astype(str).eq("STRENGTH_3")
    strength = result[strength_mask].copy()
    if strength.empty:
        return result
    date_counts = strength.groupby("Signal_Date").size().sort_index()
    full_week_counts = strength.groupby("Signal_Week").size()
    daily_counts = pd.DataFrame({"Signal_Day_Count": date_counts})
    daily_counts["Signal_Week"] = pd.to_datetime(
        daily_counts.index.astype(str), format="%Y%m%d").to_period("W-FRI").astype(str)
    daily_counts["Week_Cumulative_Count"] = daily_counts.groupby(
        "Signal_Week")["Signal_Day_Count"].cumsum()
    open_count = pd.Series(0.0, index=pd.Index(open_dates, dtype=str))
    common_dates = open_count.index.intersection(date_counts.index.astype(str))
    open_count.loc[common_dates] = date_counts.reindex(common_dates).fillna(0).astype(float)
    trailing5 = open_count.rolling(5, min_periods=1).sum()
    result.loc[strength_mask, "Signal_Day_Strength3_Count"] = result.loc[
        strength_mask, "Signal_Date"].map(date_counts)
    result.loc[strength_mask, "Signal_Week_Cumulative_Strength3_Count"] = result.loc[
        strength_mask, "Signal_Date"].map(daily_counts["Week_Cumulative_Count"])
    result.loc[strength_mask, "Trailing5D_Strength3_Count"] = result.loc[
        strength_mask, "Signal_Date"].map(trailing5)
    result.loc[strength_mask, "Future_Full_Week_Strength3_Count"] = result.loc[
        strength_mask, "Signal_Week"].map(full_week_counts)

    def bucket(series: pd.Series) -> pd.Series:
        return pd.cut(
            numeric(pd.DataFrame({"value": series}), "value"),
            [0, 1, 3, 5, 10, 20, np.inf],
            labels=["1", "2至3", "4至5", "6至10", "11至20", ">20"],
            include_lowest=True).astype(str)

    for source, target in (
            ("Signal_Day_Strength3_Count", "Signal_Day_Breadth_Group"),
            ("Signal_Week_Cumulative_Strength3_Count", "Week_Cumulative_Breadth_Group"),
            ("Trailing5D_Strength3_Count", "Trailing5D_Breadth_Group"),
            ("Future_Full_Week_Strength3_Count", "Future_Full_Week_Breadth_Group")):
        result.loc[strength_mask, target] = bucket(result.loc[strength_mask, source]).values

    k_score = numeric(result, "Signal_K").between(15, 20).astype(int)
    age_score = numeric(result, "Daily_MACD_Red_Age").between(3, 5).astype(int)
    kd_score = numeric(result, "Signal_KD_Spread").lt(0).astype(int)
    result.loc[strength_mask, "Timing_K15_20_Pass"] = k_score[strength_mask].astype(bool)
    result.loc[strength_mask, "Timing_RedAge3_5_Pass"] = age_score[strength_mask].astype(bool)
    result.loc[strength_mask, "Timing_WeeklyK_BelowD_Pass"] = kd_score[strength_mask].astype(bool)
    result.loc[strength_mask, "Timing_Score_221"] = (
        V63_SCORE_K_WEIGHT * k_score
        + V63_SCORE_AGE_WEIGHT * age_score
        + V63_SCORE_KD_WEIGHT * kd_score)[strength_mask]
    result.loc[strength_mask, "Timing_Three_Factor_Count"] = (
        k_score + age_score + kd_score)[strength_mask]

    def assign_probabilities(group: pd.DataFrame, slots: int) -> pd.Series:
        probability = pd.Series(0.0, index=group.index)
        remaining_slots = min(int(slots), len(group))
        for score in sorted(numeric(group, "Timing_Score_221").unique(), reverse=True):
            indexes = group.index[numeric(group, "Timing_Score_221").eq(score)]
            selected = min(remaining_slots, len(indexes))
            if selected > 0:
                probability.loc[indexes] = selected / len(indexes)
                remaining_slots -= selected
            if remaining_slots <= 0:
                break
        return probability

    for signal_date, group in result[strength_mask].groupby("Signal_Date"):
        day_size = len(group)
        top3 = assign_probabilities(group, 3)
        top20_slots = max(1, int(math.ceil(day_size * 0.20)))
        top20 = assign_probabilities(group, top20_slots)
        result.loc[group.index, "Score_Top3_Expected_Weight"] = top3
        result.loc[group.index, "Random_Top3_Expected_Weight"] = min(3, day_size) / day_size
        result.loc[group.index, "Score_Top20_Expected_Weight"] = top20
        result.loc[group.index, "Random_Top20_Expected_Weight"] = top20_slots / day_size
    return result


def v63_gate_summary(early: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if early.empty:
        return pd.DataFrame()
    original_class = early["Explosion_Class_40D"].astype(str)
    for deadline in V63_CONFIRM_DEADLINES:
        prefix = f"Confirm{int(deadline)}_"
        confirmed = true_mask(early, f"{prefix}Confirmed")
        has_result = true_mask(early, f"{prefix}Strategy_Has_Result")
        timeout = early[f"{prefix}Status"].astype(str).str.contains("未确认_下一开盘退出")
        strategy_return = numeric(early, f"{prefix}Strategy_Return_Net_pct")
        rows.append({
            "确认期限自然日": int(deadline), "提前买入事件": len(early),
            "期限内确认": int(confirmed.sum()),
            "期限内确认比例%": confirmed.mean() * 100.0,
            "超时退出": int(timeout.sum()),
            "原F被超时退出比例%": (
                timeout[original_class.eq("F")].mean() * 100.0
                if original_class.eq("F").any() else np.nan),
            "原S被误退出比例%": (
                timeout[original_class.eq("S")].mean() * 100.0
                if original_class.eq("S").any() else np.nan),
            "原B级以上被误退出比例%": (
                timeout[original_class.isin(["S", "A", "B"])].mean() * 100.0
                if original_class.isin(["S", "A", "B"]).any() else np.nan),
            "策略有结果事件": int(has_result.sum()),
            "确认退出策略收益均值%": strategy_return[has_result].mean(),
            "确认退出策略收益中位%": strategy_return[has_result].median(),
            "确认退出策略盈利比例%": strategy_return[has_result].gt(0).mean() * 100.0,
            "确认退出策略最大浮盈均值%": numeric(
                early.loc[has_result], f"{prefix}Strategy_MFE_Net_pct").mean(),
            "确认退出策略最大回撤均值%": numeric(
                early.loc[has_result], f"{prefix}Strategy_MAE_Raw_pct").mean(),
            "超时退出收益均值%": strategy_return[timeout].mean(),
            "超时退出收益中位%": strategy_return[timeout].median(),
            "超时退出盈利比例%": strategy_return[timeout].gt(0).mean() * 100.0,
        })
    return pd.DataFrame(rows)


def v63_reentry_summary(early: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for deadline in V63_CONFIRM_DEADLINES:
        prefix = f"Confirm{int(deadline)}_"
        has_reentry = early[f"{prefix}Reentry_Signal_Date"].map(
            normalize_date).str.len().gt(0)
        # V6.4 did not export a Confirm*_Reentry_Has_40D field.  A finite
        # 40-day close return is the authoritative maturity flag.
        mature = numeric(
            early, f"{prefix}Reentry_Close_Return_Net_pct").notna()
        classes = early[f"{prefix}Reentry_Class_40D"].astype(str)
        combined = numeric(early, f"{prefix}Combined_Return_Net_pct")
        rows.append({
            "原确认期限自然日": int(deadline),
            "超时后后来六周内上穿": int(has_reentry.sum()),
            "重新买入已有40日结果": int(mature.sum()),
            "重入S级比例%": classes[mature].eq("S").mean() * 100.0,
            "重入A或S比例%": classes[mature].isin(["S", "A"]).mean() * 100.0,
            "重入B级以上比例%": classes[mature].isin(["S", "A", "B"]).mean() * 100.0,
            "重入F级比例%": classes[mature].eq("F").mean() * 100.0,
            "重入最大浮盈中位%": numeric(
                early.loc[mature], f"{prefix}Reentry_MFE_Net_pct").median(),
            "重入最大回撤均值%": numeric(
                early.loc[mature], f"{prefix}Reentry_MAE_Raw_pct").mean(),
            "重入40日期末收益中位%": numeric(
                early.loc[mature], f"{prefix}Reentry_Close_Return_Net_pct").median(),
            "退出再重入复合收益均值%": combined[mature].mean(),
            "退出再重入复合收益中位%": combined[mature].median(),
            "退出再重入复合盈利比例%": combined[mature].gt(0).mean() * 100.0,
        })
    return pd.DataFrame(rows)


def v64_add_state_lifecycle_columns(early: pd.DataFrame) -> pd.DataFrame:
    """Build executable strategy variants without using the final grade."""
    result = early.copy()
    crossed = true_mask(result, "Future_Weekly_Cross25_Within42D")
    confirm7 = true_mask(result, "Confirm7_Confirmed")
    confirm14 = true_mask(result, "Confirm14_Confirmed")
    result["Cross_Delay_Group"] = V64_CROSS_NONE
    result.loc[crossed & confirm7, "Cross_Delay_Group"] = "7日内确认"
    result.loc[crossed & ~confirm7 & confirm14,
               "Cross_Delay_Group"] = "8至14日确认"
    result.loc[crossed & ~confirm14,
               "Cross_Delay_Group"] = "15至42日确认"
    result["Future_Cross_State"] = result.get(
        "Future_Cross_State", pd.Series(V64_CROSS_NONE, index=result.index)
    ).fillna(V64_CROSS_NONE).astype(str)

    base_return = numeric(result, "Entry_Close_Return_Net_pct")
    gate_return = numeric(result, "Confirm14_Strategy_Return_Net_pct")
    gate_hold = numeric(result, "Confirm14_Strategy_Hold_Market_Days")
    result["Lifecycle_Baseline40_Return_Net_pct"] = base_return
    result["Lifecycle_Baseline40_Hold_Market_Days"] = 40.0
    result["Lifecycle_Baseline40_Round_Trips"] = 1.0
    result["Lifecycle_Confirm14_Return_Net_pct"] = gate_return
    result["Lifecycle_Confirm14_Hold_Market_Days"] = gate_hold
    result["Lifecycle_Confirm14_Round_Trips"] = 1.0

    any_reentry = numeric(result, "Confirm14_Combined_Return_Net_pct")
    any_lifecycle = gate_return.copy()
    any_used = ~confirm14 & any_reentry.notna()
    any_lifecycle.loc[any_used] = any_reentry.loc[any_used]
    result["Lifecycle_AnyReentry14_Return_Net_pct"] = any_lifecycle
    result["Lifecycle_AnyReentry14_Round_Trips"] = 1.0 + any_used.astype(float)

    selective_reentry = numeric(
        result, "Selective_Reentry14_Combined_Return_Net_pct")
    selective_lifecycle = gate_return.copy()
    selective_used = ~confirm14 & selective_reentry.notna()
    selective_lifecycle.loc[selective_used] = selective_reentry.loc[
        selective_used]
    result["Lifecycle_HighOnlyReentry14_Return_Net_pct"] = selective_lifecycle
    result["Lifecycle_HighOnlyReentry14_Round_Trips"] = (
        1.0 + selective_used.astype(float))

    high = result["Future_Cross_State"].eq(V64_CROSS_HIGH)
    ordinary = result["Future_Cross_State"].eq(V64_CROSS_ORDINARY)
    overheated = result["Future_Cross_State"].eq(V64_CROSS_OVERHEATED)

    def mixed_strategy(
            return_column: str, hold_column: str,
            protect_overheated: bool = False) -> tuple[pd.Series, pd.Series]:
        strategy_return = gate_return.copy()
        strategy_hold = gate_hold.copy()
        confirmed_strong = confirm14 & (high | overheated)
        strategy_return.loc[confirmed_strong] = base_return.loc[
            confirmed_strong]
        strategy_hold.loc[confirmed_strong] = 40.0
        ordinary_return = numeric(result, return_column)
        ordinary_hold = numeric(result, hold_column)
        use_ordinary = confirm14 & ordinary & ordinary_return.notna()
        strategy_return.loc[use_ordinary] = ordinary_return.loc[use_ordinary]
        strategy_hold.loc[use_ordinary] = ordinary_hold.loc[use_ordinary]
        if protect_overheated:
            immediate_return = numeric(
                result, "CrossImmediate_Return_Net_pct")
            immediate_hold = numeric(
                result, "CrossImmediate_Hold_Market_Days")
            protect = confirm14 & overheated & immediate_return.notna()
            strategy_return.loc[protect] = immediate_return.loc[protect]
            strategy_hold.loc[protect] = immediate_hold.loc[protect]
        return strategy_return, strategy_hold

    immediate_return, immediate_hold = mixed_strategy(
        "CrossImmediate_Return_Net_pct",
        "CrossImmediate_Hold_Market_Days")
    result["Lifecycle_State14_OrdinaryImmediate_Return_Net_pct"] = immediate_return
    result["Lifecycle_State14_OrdinaryImmediate_Hold_Market_Days"] = immediate_hold
    result["Lifecycle_State14_OrdinaryImmediate_Round_Trips"] = 1.0

    for threshold in V64_POST_CROSS_REMAINING:
        number = int(threshold)
        strategy_return, strategy_hold = mixed_strategy(
            f"CrossRemaining{number}_Return_Net_pct",
            f"CrossRemaining{number}_Hold_Market_Days")
        result[
            f"Lifecycle_State14_OrdinaryMACD{number}_Return_Net_pct"
        ] = strategy_return
        result[
            f"Lifecycle_State14_OrdinaryMACD{number}_Hold_Market_Days"
        ] = strategy_hold
        result[
            f"Lifecycle_State14_OrdinaryMACD{number}_Round_Trips"
        ] = 1.0

    protect_return, protect_hold = mixed_strategy(
        "CrossRemaining20_Return_Net_pct",
        "CrossRemaining20_Hold_Market_Days", protect_overheated=True)
    result[
        "Lifecycle_State14_MACD20_ProtectOver30_Return_Net_pct"
    ] = protect_return
    result[
        "Lifecycle_State14_MACD20_ProtectOver30_Hold_Market_Days"
    ] = protect_hold
    result[
        "Lifecycle_State14_MACD20_ProtectOver30_Round_Trips"
    ] = 1.0
    return result


def v65_add_staged_position_columns(early: pd.DataFrame) -> pd.DataFrame:
    """Build capital-level trial/add/reentry paths from observable states.

    The first leg uses only ``trial_weight`` of starting capital.  An on-time
    high-quality confirmation adds the unused capital at the next tradable
    open and measures that added leg over its own 40-market-day window.  A
    14-day timeout closes only the trial leg; if a late high-quality
    confirmation arrives, the then-current whole account is re-entered.  This
    keeps parallel adds additive and sequential reentries compounded.
    """
    result = early.copy()
    confirm14 = true_mask(result, "Confirm14_Confirmed")
    state = result["Future_Cross_State"].fillna(V64_CROSS_NONE).astype(str)
    high = state.eq(V64_CROSS_HIGH)
    ordinary = state.eq(V64_CROSS_ORDINARY)
    overheated = state.eq(V64_CROSS_OVERHEATED)
    no_cross = state.eq(V64_CROSS_NONE)
    on_time_high = confirm14 & high
    late_high = ~confirm14 & high

    action = pd.Series("", index=result.index, dtype=object)
    action.loc[confirm14 & high] = "14日内高质量确认_试仓升级"
    action.loc[confirm14 & ordinary] = "14日内普通确认_只保留试仓"
    action.loc[confirm14 & overheated] = "14日内已涨超30_保护试仓不追买"
    action.loc[~confirm14 & late_high] = "14日退出_迟到高质量确认可重入"
    action.loc[~confirm14 & ordinary] = "14日退出_迟到普通确认不重入"
    action.loc[~confirm14 & overheated] = "14日退出_迟到且已涨超30不追买"
    action.loc[~confirm14 & no_cross] = "14日退出_42日仍未确认"
    action.loc[action.eq("")] = "其他_仅保留试仓"
    result["V65_Lifecycle_Action"] = action

    base_return = numeric(result, "Entry_Close_Return_Net_pct")
    gate_return = numeric(result, "Confirm14_Strategy_Return_Net_pct")
    gate_hold = numeric(result, "Confirm14_Strategy_Hold_Market_Days")
    add_return = numeric(result, "CrossEntry_Close_Return_Net_pct")
    late_reentry_return = numeric(
        result, "Selective_Reentry14_Close_Return_Net_pct")

    result["Lifecycle_V65_FullUpfront_Return_Net_pct"] = base_return
    result["Lifecycle_V65_FullUpfront_Capital_Exposure_Days"] = 40.0
    result["Lifecycle_V65_FullUpfront_Peak_Capital_Weight"] = 1.0
    result["Lifecycle_V65_FullUpfront_Round_Trips"] = 1.0
    result["Lifecycle_V65_FullUpfront_High_Upgrade_Used"] = False
    result["Lifecycle_V65_FullUpfront_Late_High_Reentry_Used"] = False

    def add_variant(key: str, trial_weight: float,
                    allow_late_high_reentry: bool) -> None:
        returns = pd.Series(np.nan, index=result.index, dtype=float)
        exposure = pd.Series(np.nan, index=result.index, dtype=float)
        peak_weight = pd.Series(np.nan, index=result.index, dtype=float)
        round_trips = pd.Series(np.nan, index=result.index, dtype=float)

        confirmed_base = confirm14 & base_return.notna()
        timeout_base = ~confirm14 & gate_return.notna()
        returns.loc[confirmed_base] = (
            float(trial_weight) * base_return.loc[confirmed_base])
        returns.loc[timeout_base] = (
            float(trial_weight) * gate_return.loc[timeout_base])
        exposure.loc[confirmed_base] = float(trial_weight) * 40.0
        exposure.loc[timeout_base] = (
            float(trial_weight) * gate_hold.loc[timeout_base])
        initial_valid = confirmed_base | timeout_base
        peak_weight.loc[initial_valid] = float(trial_weight)
        round_trips.loc[initial_valid] = 1.0

        upgrade_used = (
            on_time_high & base_return.notna() & add_return.notna())
        returns.loc[upgrade_used] = (
            float(trial_weight) * base_return.loc[upgrade_used]
            + (1.0 - float(trial_weight)) * add_return.loc[upgrade_used])
        exposure.loc[upgrade_used] = 40.0
        peak_weight.loc[upgrade_used] = 1.0
        round_trips.loc[upgrade_used] = 2.0

        reentry_used = pd.Series(False, index=result.index, dtype=bool)
        if allow_late_high_reentry:
            reentry_used = (
                late_high & gate_return.notna() & late_reentry_return.notna())
            pre_reentry_multiplier = (
                1.0 + float(trial_weight)
                * gate_return.loc[reentry_used] / 100.0)
            returns.loc[reentry_used] = (
                pre_reentry_multiplier
                * (1.0 + late_reentry_return.loc[reentry_used] / 100.0)
                - 1.0) * 100.0
            exposure.loc[reentry_used] = (
                float(trial_weight) * gate_hold.loc[reentry_used] + 40.0)
            peak_weight.loc[reentry_used] = 1.0
            round_trips.loc[reentry_used] = 2.0

        prefix = f"Lifecycle_V65_{key}_"
        result[f"{prefix}Return_Net_pct"] = returns
        result[f"{prefix}Capital_Exposure_Days"] = exposure
        result[f"{prefix}Peak_Capital_Weight"] = peak_weight
        result[f"{prefix}Round_Trips"] = round_trips
        result[f"{prefix}High_Upgrade_Used"] = upgrade_used
        result[f"{prefix}Late_High_Reentry_Used"] = reentry_used

    for weight in V65_TRIAL_WEIGHTS:
        number = int(round(weight * 100.0))
        add_variant(f"Trial{number}UpgradeNoReentry", weight, False)
        add_variant(f"Trial{number}UpgradeLateHigh", weight, True)
    return result


V65_STAGED_SCHEMES = (
    ("100%提前买入固定40日", "FullUpfront"),
    ("1/3试仓_高质量补满_不重入", "Trial33UpgradeNoReentry"),
    ("1/3试仓_高质量补满_迟到高质量满仓重入",
     "Trial33UpgradeLateHigh"),
    ("1/2试仓_高质量补满_不重入", "Trial50UpgradeNoReentry"),
    ("1/2试仓_高质量补满_迟到高质量满仓重入",
     "Trial50UpgradeLateHigh"),
)


def v65_staged_position_summary(early: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, key in V65_STAGED_SCHEMES:
        prefix = f"Lifecycle_V65_{key}_"
        returns = numeric(early, f"{prefix}Return_Net_pct").dropna()
        if returns.empty:
            continue
        source = early.loc[returns.index]
        exposure = numeric(source, f"{prefix}Capital_Exposure_Days")
        peak = numeric(source, f"{prefix}Peak_Capital_Weight")
        trips = numeric(source, f"{prefix}Round_Trips")
        upgrades = true_mask(source, f"{prefix}High_Upgrade_Used")
        reentries = true_mask(source, f"{prefix}Late_High_Reentry_Used")
        mean_exposure = exposure.mean()
        equivalent_40d_return = (
            returns.mean() * 40.0 / mean_exposure
            if math.isfinite(mean_exposure) and mean_exposure > 0 else np.nan)
        rows.append({
            "资金生命周期方案": label,
            "有结果事件": len(returns),
            "实际收益均值%": returns.mean(),
            "实际收益中位%": returns.median(),
            "实际盈利比例%": returns.gt(0).mean() * 100.0,
            "实际达到10%比例%": returns.ge(10).mean() * 100.0,
            "实际达到20%比例%": returns.ge(20).mean() * 100.0,
            "实际达到30%比例%": returns.ge(30).mean() * 100.0,
            "实际亏损10%以上比例%": returns.le(-10).mean() * 100.0,
            "收益第10分位%": returns.quantile(0.10),
            "最差10%平均收益%": returns[returns.le(
                returns.quantile(0.10))].mean(),
            "收益标准差%": returns.std(ddof=0),
            "平均峰值资金占用%": peak.mean() * 100.0,
            "平均资本暴露日": mean_exposure,
            "粗略40日等效资本收益%": equivalent_40d_return,
            "高质量确认补仓事件": int(upgrades.sum()),
            "迟到高质量满仓重入事件": int(reentries.sum()),
            "平均完整交易腿数": trips.mean(),
        })
    return pd.DataFrame(rows)


def v65_action_outcome_summary(early: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for action, group in early.groupby("V65_Lifecycle_Action", dropna=False):
        classes = group["Explosion_Class_40D"].astype(str)
        early_return = numeric(group, "Entry_Close_Return_Net_pct")
        row: dict[str, Any] = {
            "生命周期动作": action,
            "事件数": len(group),
            "不同股票": group["ts_code"].nunique(),
            "信号周": group["Signal_Week"].nunique(),
            "原早仓S级比例%": classes.eq("S").mean() * 100.0,
            "原早仓A或S比例%": classes.isin(["S", "A"]).mean() * 100.0,
            "原早仓B级以上比例%": classes.isin(
                ["S", "A", "B"]).mean() * 100.0,
            "原早仓F级比例%": classes.eq("F").mean() * 100.0,
            "原早仓40日收益均值%": early_return.mean(),
            "原早仓40日收益中位%": early_return.median(),
            "原早仓最大浮盈中位%": numeric(
                group, "Entry_MFE_Net_pct").median(),
            "原早仓最大回撤均值%": numeric(
                group, "Entry_MAE_Raw_pct").mean(),
        }
        for label, key in V65_STAGED_SCHEMES[1:]:
            values = numeric(
                group, f"Lifecycle_V65_{key}_Return_Net_pct")
            short = "33%含重入" if key == "Trial33UpgradeLateHigh" else (
                "33%不重入" if key == "Trial33UpgradeNoReentry" else (
                    "50%含重入" if key == "Trial50UpgradeLateHigh"
                    else "50%不重入"))
            row[f"{short}收益均值%"] = values.mean()
            row[f"{short}收益中位%"] = values.median()
        rows.append(row)
    return pd.DataFrame(rows)


def v65_staged_year_summary(early: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    years = early["Signal_Date"].astype(str).str[:4]
    for year, group in early.groupby(years):
        part = v65_staged_position_summary(group)
        if not part.empty:
            part.insert(0, "信号年度", year)
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def v611_add_feature_percentiles(
        frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build signal-time, same-entry-date comparable model features."""
    work = frame.copy()
    raw_close = numeric(work, "Raw_Close").replace(0, np.nan)
    work["V611_MACD_Hist_to_Price"] = (
        numeric(work, "Daily_MACD_Hist") / raw_close * 10_000.0)
    if "Entry_Date" not in work:
        work["Entry_Date"] = ""
    work["Entry_Date"] = work["Entry_Date"].map(normalize_date)
    feature_columns: list[str] = []
    for source in V611_FEATURES:
        target = f"V611P_{source}"
        values = numeric(work, source)
        work[target] = (
            work.assign(_v611_value=values).groupby("Entry_Date")[
                "_v611_value"].rank(method="average", pct=True))
        feature_columns.append(target)
    return work, feature_columns


def v611_fit_pairwise_ordinal(
        train: pd.DataFrame, feature_columns: list[str]
        ) -> dict[str, Any] | None:
    """Fit an ordinal S>A>B>F pairwise model on already revealed dates."""
    if train.empty:
        return None
    work = train.copy()
    grade_map = {"F": 0.0, "B": 1.0, "A": 2.0, "S": 3.0}
    work["_V611_Grade"] = work["Explosion_Class_40D"].astype(str).map(
        grade_map)
    work = work[work["_V611_Grade"].notna()].copy()
    if (len(work) < V611_MIN_TRAIN_ROWS
            or work["Entry_Date"].nunique() < V611_MIN_TRAIN_DATES
            or work["_V611_Grade"].nunique() < 2):
        return None

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    weight_rows: list[float] = []
    pair_count = 0
    gain_map = {0: 0.0, 1: 1.0, 2: 3.0, 3: 7.0}
    for entry_date, group in work.groupby("Entry_Date", sort=True):
        group = group.reset_index(drop=True)
        matrix = group[feature_columns].apply(
            pd.to_numeric, errors="coerce").fillna(0.5).to_numpy(dtype=float)
        labels = pd.to_numeric(
            group["_V611_Grade"], errors="coerce").to_numpy(dtype=float)
        possible: list[tuple[int, int]] = []
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                if labels[left] == labels[right]:
                    continue
                high, low = ((left, right) if labels[left] > labels[right]
                             else (right, left))
                possible.append((high, low))
        if not possible:
            continue
        if len(possible) > V611_MAX_PAIRS_PER_DATE:
            rng = np.random.default_rng(_ltr_seed(f"V611|{entry_date}"))
            chosen = rng.choice(
                len(possible), size=V611_MAX_PAIRS_PER_DATE, replace=False)
            pairs = [possible[int(position)]
                     for position in sorted(chosen.tolist())]
        else:
            pairs = possible
        raw_weights = np.array([
            gain_map[int(labels[high])] - gain_map[int(labels[low])]
            for high, low in pairs], dtype=float)
        raw_weights /= max(float(raw_weights.sum()), 1.0)
        for (high, low), pair_weight in zip(pairs, raw_weights):
            difference = matrix[high] - matrix[low]
            x_rows.extend([difference, -difference])
            y_rows.extend([1, 0])
            weight_rows.extend([
                float(pair_weight) * 0.5, float(pair_weight) * 0.5])
        pair_count += len(pairs)
    if pair_count == 0:
        return None
    coefficients = _fit_weighted_pairwise_logit(
        np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=int),
        np.asarray(weight_rows, dtype=float))
    return {
        "coefficients": coefficients,
        "feature_columns": list(feature_columns),
        "train_rows": int(len(work)),
        "train_dates": int(work["Entry_Date"].nunique()),
        "train_weeks": int(work["Signal_Week"].nunique()),
        "pair_count": int(pair_count),
        "train_start": str(work["Signal_Date"].astype(str).min()),
        "train_end": str(work["Signal_Date"].astype(str).max()),
        "max_maturity_date": str(
            work["Entry_End_Date_40D"].astype(str).max()),
    }


def v611_predict_ordinal(
        frame: pd.DataFrame, bundle: dict[str, Any] | None) -> pd.Series:
    if frame.empty or bundle is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    columns = list(bundle["feature_columns"])
    matrix = frame[columns].apply(
        pd.to_numeric, errors="coerce").fillna(0.5).to_numpy(dtype=float)
    coefficients = np.asarray(bundle["coefficients"], dtype=float)
    return pd.Series(matrix @ coefficients, index=frame.index, dtype=float)


def v611_add_walk_forward_ordinal(
        early: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    """Generate strict 40-day-maturity walk-forward scores per entry date."""
    work, feature_columns = v611_add_feature_percentiles(early)
    work["V611_OOS_Score"] = np.nan
    work["V611_OOS_Rank"] = np.nan
    work["V611_Model_Available"] = False
    work["V611_Train_Rows"] = 0
    work["V611_Train_Dates"] = 0
    work["V611_Train_Weeks"] = 0
    work["V611_Train_Max_Maturity_Date"] = ""
    maturity = work["Entry_End_Date_40D"].map(normalize_date)
    audit_rows: list[dict[str, Any]] = []
    for entry_date, group in work.groupby("Entry_Date", sort=True):
        revealed = maturity.lt(str(entry_date))
        train = work[revealed].copy()
        bundle = v611_fit_pairwise_ordinal(train, feature_columns)
        max_maturity = str(maturity[revealed].max()) if revealed.any() else ""
        audit_row = {
            "预测入场日": str(entry_date), "当日候选": len(group),
            "已揭晓训练事件": int(revealed.sum()),
            "已揭晓训练入场日": int(train["Entry_Date"].nunique()),
            "训练最大40日结束日期": max_maturity,
            "严格早于预测日": bool(
                not max_maturity or max_maturity < str(entry_date)),
            "模型可用": bundle is not None,
        }
        if bundle is not None:
            work.loc[group.index, "V611_OOS_Score"] = v611_predict_ordinal(
                work.loc[group.index], bundle)
            work.loc[group.index, "V611_Model_Available"] = True
            work.loc[group.index, "V611_Train_Rows"] = bundle["train_rows"]
            work.loc[group.index, "V611_Train_Dates"] = bundle["train_dates"]
            work.loc[group.index, "V611_Train_Weeks"] = bundle["train_weeks"]
            work.loc[group.index, "V611_Train_Max_Maturity_Date"] = (
                bundle["max_maturity_date"])
            audit_row.update({
                "实际训练事件": bundle["train_rows"],
                "实际训练入场日": bundle["train_dates"],
                "实际训练信号周": bundle["train_weeks"],
                "训练股票对": bundle["pair_count"],
            })
        else:
            audit_row.update({
                "实际训练事件": 0, "实际训练入场日": 0,
                "实际训练信号周": 0, "训练股票对": 0,
            })
        audit_rows.append(audit_row)

    valid = true_mask(work, "V611_Model_Available")
    ranked = work[valid].sort_values(
        ["Entry_Date", "V611_OOS_Score", "Timing_Score_221",
         "Daily_MACD_Hist", "ts_code"],
        ascending=[True, False, False, True, True]).copy()
    ranked["V611_OOS_Rank"] = ranked.groupby("Entry_Date").cumcount() + 1
    work.loc[ranked.index, "V611_OOS_Rank"] = ranked["V611_OOS_Rank"]
    full_bundle = v611_fit_pairwise_ordinal(work, feature_columns)
    return work, pd.DataFrame(audit_rows), full_bundle


def v611_score_live_candidates(
        live: pd.DataFrame, bundle: dict[str, Any] | None) -> pd.DataFrame:
    if live.empty:
        return live.copy()
    work, feature_columns = v611_add_feature_percentiles(live)
    work["V611_Live_Score"] = np.nan
    work["V611_Live_Rank"] = np.nan
    work["V611_Live_Model_Available"] = bundle is not None
    if bundle is None:
        return work
    if feature_columns != list(bundle["feature_columns"]):
        raise RuntimeError("V6.11实时特征列与历史训练不一致")
    work["V611_Live_Score"] = v611_predict_ordinal(work, bundle)
    ranked = work.sort_values(
        ["Entry_Date", "V611_Live_Score", "Timing_Score_221",
         "Daily_MACD_Hist", "ts_code"],
        ascending=[True, False, False, True, True]).copy()
    ranked["V611_Live_Rank"] = ranked.groupby("Entry_Date").cumcount() + 1
    work.loc[ranked.index, "V611_Live_Rank"] = ranked["V611_Live_Rank"]
    work["V611_Live_Train_Rows"] = int(bundle["train_rows"])
    work["V611_Live_Train_Dates"] = int(bundle["train_dates"])
    work["V611_Live_Train_Weeks"] = int(bundle["train_weeks"])
    return work


def v611_feature_coefficients(
        bundle: dict[str, Any] | None) -> pd.DataFrame:
    if bundle is None:
        return pd.DataFrame(columns=["特征", "同日百分位系数", "模型方向"])
    rows: list[dict[str, Any]] = []
    for source, transformed, coefficient in zip(
            V611_FEATURES, bundle["feature_columns"],
            np.asarray(bundle["coefficients"], dtype=float)):
        rows.append({
            "特征": source, "模型字段": transformed,
            "同日百分位系数": float(coefficient),
            "模型方向": "数值较高有利" if coefficient > 0 else "数值较低有利",
            "绝对影响": abs(float(coefficient)),
            "训练事件": int(bundle["train_rows"]),
            "训练入场日": int(bundle["train_dates"]),
            "训练信号周": int(bundle["train_weeks"]),
            "训练股票对": int(bundle["pair_count"]),
        })
    return pd.DataFrame(rows).sort_values(
        "绝对影响", ascending=False).reset_index(drop=True)


def v611_oos_grade_audit(early: pd.DataFrame) -> pd.DataFrame:
    """Compare Top3 grades only on dates where the walk-forward model exists."""
    pool = early[true_mask(early, "V611_Model_Available")].copy()
    if pool.empty:
        return pd.DataFrame()
    pool["_Year"] = pool["Entry_Date"].astype(str).str[:4]

    def deterministic_metrics(frame: pd.DataFrame, mode: str) -> dict[str, Any]:
        chosen_parts: list[pd.DataFrame] = []
        three_group_hits: list[bool] = []
        for _, group in frame.groupby("Entry_Date", sort=True):
            if mode == "WalkForwardSABF":
                ordered = group.sort_values(
                    ["V611_OOS_Score", "Timing_Score_221",
                     "Daily_MACD_Hist", "ts_code"],
                    ascending=[False, False, True, True])
            else:
                ordered = group.sort_values(
                    ["Timing_Score_221", "Daily_MACD_Hist", "ts_code"],
                    ascending=[False, True, True])
            chosen = ordered.head(3)
            chosen_parts.append(chosen)
            if len(group) >= 3:
                three_group_hits.append(
                    bool(numeric(chosen, "Entry_Close_Return_Net_pct").gt(0).sum() >= 2))
        selected = pd.concat(chosen_parts, ignore_index=True)
        classes = selected["Explosion_Class_40D"].astype(str)
        returns = numeric(selected, "Entry_Close_Return_Net_pct")
        return {
            "入选事件": len(selected),
            "S级比例%": classes.eq("S").mean() * 100.0,
            "A或S比例%": classes.isin(["S", "A"]).mean() * 100.0,
            "B级以上比例%": classes.isin(["S", "A", "B"]).mean() * 100.0,
            "F级比例%": classes.eq("F").mean() * 100.0,
            "固定40日盈利比例%": returns.gt(0).mean() * 100.0,
            "固定40日收益均值%": returns.mean(),
            "三只候选组": len(three_group_hits),
            "三只中至少两只盈利组比例%": (
                np.mean(three_group_hits) * 100.0 if three_group_hits else np.nan),
        }

    def random_metrics(frame: pd.DataFrame) -> dict[str, Any]:
        selected_count = 0
        expected = defaultdict(float)
        three_groups = 0
        expected_two_wins = 0.0
        for _, group in frame.groupby("Entry_Date", sort=True):
            n, k = len(group), min(3, len(group))
            if not n:
                continue
            selected_count += k
            classes = group["Explosion_Class_40D"].astype(str)
            returns = numeric(group, "Entry_Close_Return_Net_pct")
            expected["S"] += k * classes.eq("S").mean()
            expected["AS"] += k * classes.isin(["S", "A"]).mean()
            expected["B"] += k * classes.isin(["S", "A", "B"]).mean()
            expected["F"] += k * classes.eq("F").mean()
            expected["Win"] += k * returns.gt(0).mean()
            expected["Return"] += k * returns.mean()
            if n >= 3:
                three_groups += 1
                wins = int(returns.gt(0).sum())
                denominator = math.comb(n, 3)
                probability = 0.0
                for count in (2, 3):
                    if count <= wins and 3 - count <= n - wins:
                        probability += (
                            math.comb(wins, count)
                            * math.comb(n - wins, 3 - count) / denominator)
                expected_two_wins += probability
        denominator = max(selected_count, 1)
        return {
            "入选事件": selected_count,
            "S级比例%": expected["S"] / denominator * 100.0,
            "A或S比例%": expected["AS"] / denominator * 100.0,
            "B级以上比例%": expected["B"] / denominator * 100.0,
            "F级比例%": expected["F"] / denominator * 100.0,
            "固定40日盈利比例%": expected["Win"] / denominator * 100.0,
            "固定40日收益均值%": expected["Return"] / denominator,
            "三只候选组": three_groups,
            "三只中至少两只盈利组比例%": (
                expected_two_wins / three_groups * 100.0
                if three_groups else np.nan),
        }

    rows: list[dict[str, Any]] = []
    periods = [("全部OOS", pool)] + [
        (str(year), group) for year, group in pool.groupby("_Year", sort=True)]
    for period, frame in periods:
        common = {
            "统计期": period, "模型可用入场日": frame["Entry_Date"].nunique(),
            "覆盖信号周": frame["Signal_Week"].nunique(),
        }
        rows.append({**common, "排序方案": "随机Top3精确期望",
                     **random_metrics(frame)})
        rows.append({**common, "排序方案": "原2-2-1主分_MACD同分",
                     **deterministic_metrics(frame, "Score221")})
        rows.append({**common, "排序方案": "走步S>A>B>F主排序",
                     **deterministic_metrics(frame, "WalkForwardSABF")})
    return pd.DataFrame(rows)


V66_PORTFOLIO_SCHEMES = (
    {
        "key": "Full100Top3NoEarlyFailure",
        "label": "全额三仓_Top3_无早退_14日生命周期基线",
        "trial_weight": 1.0, "staged": True,
        "allow_late_reentry": False, "ordinary_keep_min_score": 5.0,
        "max_signal_day_rank": 3,
        "primary_rank_mode": "Score221MACDHistAsc",
        "tie_break_mode": "DailyMACDHistAsc",
        "ordinary_risk_days": 0, "ordinary_stop_pct": V68_ORDINARY_STOP_PCT,
        "early_failure_day": 0,
        "early_price_stop_pct": V612_PRICE_FAILURE_PCT,
        "early_macd_remaining_pct": V612_MACD_REMAINING_PCT,
    },
    {
        "key": "Full100Top3EarlyFailureD3",
        "label": "全额三仓_Top3_第3日硬失败退出",
        "trial_weight": 1.0, "staged": True,
        "allow_late_reentry": False, "ordinary_keep_min_score": 5.0,
        "max_signal_day_rank": 3,
        "primary_rank_mode": "Score221MACDHistAsc",
        "tie_break_mode": "DailyMACDHistAsc",
        "ordinary_risk_days": 0, "ordinary_stop_pct": V68_ORDINARY_STOP_PCT,
        "early_failure_day": 3,
        "early_price_stop_pct": V612_PRICE_FAILURE_PCT,
        "early_macd_remaining_pct": V612_MACD_REMAINING_PCT,
    },
    {
        "key": "Full100Top3EarlyFailureD5",
        "label": "全额三仓_Top3_第5日硬失败退出",
        "trial_weight": 1.0, "staged": True,
        "allow_late_reentry": False, "ordinary_keep_min_score": 5.0,
        "max_signal_day_rank": 3,
        "primary_rank_mode": "Score221MACDHistAsc",
        "tie_break_mode": "DailyMACDHistAsc",
        "ordinary_risk_days": 0, "ordinary_stop_pct": V68_ORDINARY_STOP_PCT,
        "early_failure_day": 5,
        "early_price_stop_pct": V612_PRICE_FAILURE_PCT,
        "early_macd_remaining_pct": V612_MACD_REMAINING_PCT,
    },
    {
        "key": "Full100Top3EarlyFailureD7",
        "label": "全额三仓_Top3_第7日硬失败退出",
        "trial_weight": 1.0, "staged": True,
        "allow_late_reentry": False, "ordinary_keep_min_score": 5.0,
        "max_signal_day_rank": 3,
        "primary_rank_mode": "Score221MACDHistAsc",
        "tie_break_mode": "DailyMACDHistAsc",
        "ordinary_risk_days": 0, "ordinary_stop_pct": V68_ORDINARY_STOP_PCT,
        "early_failure_day": 7,
        "early_price_stop_pct": V612_PRICE_FAILURE_PCT,
        "early_macd_remaining_pct": V612_MACD_REMAINING_PCT,
    },
)


def v66_market_date_offset(
        value: Any, offset: int, open_dates: list[str],
        open_pos: dict[str, int]) -> str:
    date_value = normalize_date(value)
    if not date_value or date_value not in open_pos:
        return ""
    position = open_pos[date_value] + int(offset)
    return open_dates[position] if 0 <= position < len(open_dates) else ""


def v66_prepare_portfolio_events(
        early: pd.DataFrame, open_dates: list[str]) -> pd.DataFrame:
    """Keep only signal-time ranking fields and observable execution dates."""
    result = early.copy().reset_index(drop=True)
    open_pos = {value: position for position, value in enumerate(open_dates)}
    date_columns = (
        "Signal_Date", "Entry_Date", "Entry_End_Date_40D",
        "CrossEntry_Date", "Confirm14_Exit_Date",
        "CrossImmediate_Decision_Date", "Future_Weekly_Cross25_Date",
        "CrossImmediate_Exit_Date", "Selective_Reentry14_Entry_Date",
    )
    for column in date_columns:
        if column in result:
            result[column] = result[column].map(normalize_date)
        else:
            result[column] = ""
    result["_Event_ID"] = [
        f"{number}:{row.get('ts_code', '')}:{row.get('Signal_Date', '')}"
        for number, row in result.iterrows()
    ]
    result["_Score_221"] = numeric(result, "Timing_Score_221").fillna(-999.0)
    result["_Base_Exit_Date"] = result["Entry_End_Date_40D"]
    missing_base = result["_Base_Exit_Date"].eq("")
    result.loc[missing_base, "_Base_Exit_Date"] = result.loc[
        missing_base, "Entry_Date"].map(
            lambda value: v66_market_date_offset(
                value, DAILY_AUDIT_DAYS - 1, open_dates, open_pos))
    result["_Add_Exit_Date"] = result["CrossEntry_Date"].map(
        lambda value: v66_market_date_offset(
            value, DAILY_AUDIT_DAYS - 1, open_dates, open_pos))
    result["_Reentry_Exit_Date"] = result[
        "Selective_Reentry14_Entry_Date"].map(
            lambda value: v66_market_date_offset(
                value, DAILY_AUDIT_DAYS - 1, open_dates, open_pos))
    result = result[
        result["Entry_Date"].str.len().eq(8)
        & result["_Base_Exit_Date"].str.len().eq(8)
    ].copy()
    return result.sort_values(
        ["Entry_Date", "Signal_Date", "ts_code", "_Event_ID"]
    ).reset_index(drop=True)


def v66_load_price_book(
        events: pd.DataFrame, data_start: str, market_end: str
        ) -> tuple[dict[str, dict[str, Any]], int, list[str]]:
    """Load already-cached qfq daily prices without making new API calls."""
    book: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    cache_hits = 0
    for code in sorted(events["ts_code"].astype(str).unique()):
        daily, _, _ = load_covering_cache(code, data_start, market_end)
        if daily.empty:
            missing.append(code)
            continue
        daily = add_daily_macd(normalize_price_frame(daily))
        dates = daily["trade_date"].astype(str).tolist()
        book[code] = {
            "dates": dates,
            "open": dict(zip(dates, numeric(daily, "open"))),
            "close": dict(zip(dates, numeric(daily, "close"))),
            "high": dict(zip(dates, numeric(daily, "high"))),
            "low": dict(zip(dates, numeric(daily, "low"))),
            "macd_hist": dict(zip(
                dates, numeric(daily, "Daily_MACD_Hist"))),
            "macd_remaining": dict(zip(
                dates, numeric(daily, "Daily_MACD_Remaining_pct"))),
            "macd_retention": dict(zip(
                dates, numeric(daily, "Daily_MACD_Retention_pct"))),
        }
        cache_hits += 1
    return book, cache_hits, missing


def v66_run_portfolio_path(
        events: pd.DataFrame, price_book: dict[str, dict[str, Any]],
        open_dates: list[str], config: dict[str, Any], scheme: dict[str, Any],
        selection_mode: str, seed: int, keep_details: bool = False
        ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Chronological three-stock portfolio simulation with no future ranking.

    Every executable scheme buys one complete 100,000-yuan slot.  The frozen
    ranking is 2-2-1 descending with smaller MACD histogram for exact ties.
    Early-failure challengers inspect only the configured D3/D5/D7 close and
    execute at the next tradable open.  Future state is consulted only after
    its actual confirmation date has arrived.
    """
    rng = np.random.default_rng(int(seed))
    buy_factor, sell_factor = _cost_factors(config)
    initial_capital = float(V66_INITIAL_CAPITAL)
    slot_capital = float(V66_FULL_SLOT_CAPITAL)
    trial_weight = float(scheme["trial_weight"])
    staged = bool(scheme["staged"])
    allow_late = bool(scheme["allow_late_reentry"])
    max_signal_day_rank = max(
        int(scheme.get("max_signal_day_rank", V66_MAX_STOCKS)), 1)
    primary_rank_mode = str(
        scheme.get("primary_rank_mode", "Score221MACDHistAsc"))
    tie_break_mode = str(scheme.get("tie_break_mode", "Random"))
    ordinary_keep_min_score = scheme.get("ordinary_keep_min_score")
    if ordinary_keep_min_score is not None:
        ordinary_keep_min_score = float(ordinary_keep_min_score)
    ordinary_risk_days = max(int(scheme.get("ordinary_risk_days", 0)), 0)
    ordinary_stop_pct = float(
        scheme.get("ordinary_stop_pct", V68_ORDINARY_STOP_PCT))
    early_failure_day = max(int(scheme.get("early_failure_day", 0)), 0)
    early_price_stop_pct = float(scheme.get(
        "early_price_stop_pct", V612_PRICE_FAILURE_PCT))
    early_macd_remaining_pct = float(scheme.get(
        "early_macd_remaining_pct", V612_MACD_REMAINING_PCT))
    open_pos = {value: position for position, value in enumerate(open_dates)}

    records = events.to_dict("records")
    event_by_id = {str(row["_Event_ID"]): row for row in records}
    candidates_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        code = str(row.get("ts_code", ""))
        entry_date = normalize_date(row.get("Entry_Date"))
        if code in price_book and entry_date:
            candidates_by_date[entry_date].append(row)
    if not candidates_by_date:
        return {}, pd.DataFrame(), pd.DataFrame()
    first_date = min(candidates_by_date)
    planned_columns = ["_Base_Exit_Date"]
    if staged and trial_weight < 1.0 - 1e-9:
        planned_columns.append("_Add_Exit_Date")
    if allow_late:
        planned_columns.append("_Reentry_Exit_Date")
    last_planned = max(
        [normalize_date(value) for value in events[
            planned_columns].to_numpy().ravel() if normalize_date(value)]
        + [first_date])
    market_end = str(config["market_end"])
    core_end = min(max(last_planned, str(config["signal_end"])), market_end)
    # Keep a short liquidation buffer only for a stock suspended on its
    # planned exit date.  Performance is later trimmed back to the common
    # core horizon unless an actual delayed exit needs the extra dates.
    buffer_end = v66_market_date_offset(core_end, 20, open_dates, open_pos)
    last_date = min(buffer_end or core_end, market_end)
    simulation_dates = [
        value for value in open_dates if first_date <= value <= last_date]
    if not simulation_dates:
        return {}, pd.DataFrame(), pd.DataFrame()

    cash = initial_capital
    active_legs: dict[str, dict[str, Any]] = {}
    code_legs: dict[str, set[str]] = defaultdict(set)
    selected_events: set[str] = set()
    open_exit_schedule: dict[str, list[str]] = defaultdict(list)
    open_exit_reason_by_leg: dict[str, str] = {}
    add_schedule: dict[str, list[str]] = defaultdict(list)
    reentry_schedule: dict[str, list[str]] = defaultdict(list)
    early_failure_schedule: dict[str, list[str]] = defaultdict(list)
    close_exit_schedule: dict[str, list[str]] = defaultdict(list)
    ordinary_risk_rules: dict[str, dict[str, Any]] = {}
    selected_rank_by_event: dict[str, int] = {}
    last_close: dict[str, float] = {}
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    leg_number = 0
    counters: dict[str, int] = defaultdict(int)

    def exact_price(code: str, trade_date: str, field: str,
                    fallback: Any = np.nan) -> float:
        value = finite_num(price_book.get(code, {}).get(field, {}).get(
            trade_date, fallback))
        return value

    def next_stock_date(code: str, trade_date: str) -> str:
        dates = price_book.get(code, {}).get("dates", [])
        position = bisect.bisect_right(dates, trade_date)
        return dates[position] if position < len(dates) else ""

    def buy_leg(row: dict[str, Any], trade_date: str, kind: str,
                desired_cash: float, fallback_open: Any = np.nan,
                selection_rank: int | None = None
                ) -> str:
        nonlocal cash, leg_number
        code = str(row.get("ts_code", ""))
        raw_open = exact_price(code, trade_date, "open", fallback_open)
        budget = min(float(desired_cash), max(cash, 0.0))
        if not math.isfinite(raw_open) or raw_open <= 0 or budget <= 0:
            return ""
        effective = raw_open * buy_factor
        shares = math.floor(budget / effective / V66_BOARD_LOT) * V66_BOARD_LOT
        if shares < V66_BOARD_LOT:
            return ""
        cost = float(shares) * effective
        if cost > cash + 1e-6:
            return ""
        cash -= cost
        leg_number += 1
        leg_id = f"L{leg_number}"
        event_id = str(row["_Event_ID"])
        active_legs[leg_id] = {
            "Leg_ID": leg_id, "Event_ID": event_id, "ts_code": code,
            "name": str(row.get("name", "")), "Kind": kind,
            "Entry_Date": trade_date, "Raw_Entry": raw_open,
            "Shares": float(shares), "Entry_Cost": cost,
            "Signal_Day_Selection_Rank": (
                int(selection_rank) if selection_rank is not None else np.nan),
            "Selection_Rank_Mode": selection_mode,
        }
        code_legs[code].add(leg_id)
        return leg_id

    def sell_leg(leg_id: str, trade_date: str, field: str,
                 reason: str, allow_stale_close: bool = False) -> bool:
        nonlocal cash
        leg = active_legs.get(leg_id)
        if leg is None:
            return False
        code = str(leg["ts_code"])
        raw_exit = exact_price(code, trade_date, field)
        if ((not math.isfinite(raw_exit) or raw_exit <= 0)
                and field == "close" and allow_stale_close):
            raw_exit = finite_num(last_close.get(code))
        if not math.isfinite(raw_exit) or raw_exit <= 0:
            return False
        proceeds = float(leg["Shares"]) * raw_exit * sell_factor
        cash += proceeds
        pnl = proceeds - float(leg["Entry_Cost"])
        trades.append({
            "方案": str(scheme["label"]),
            "选择方式": selection_mode,
            "Seed": int(seed),
            **leg,
            "Exit_Date": trade_date, "Raw_Exit": raw_exit,
            "Exit_Reason": reason, "Net_Proceeds": proceeds,
            "Net_PnL": pnl,
            "Net_Return_pct": (
                proceeds / float(leg["Entry_Cost"]) - 1.0) * 100.0,
        })
        del active_legs[leg_id]
        ordinary_risk_rules.pop(leg_id, None)
        open_exit_reason_by_leg.pop(leg_id, None)
        code_legs[code].discard(leg_id)
        if not code_legs[code]:
            del code_legs[code]
        return True

    def schedule_selected_event(row: dict[str, Any], trial_leg: str) -> None:
        event_id = str(row["_Event_ID"])
        if not staged:
            exit_date = normalize_date(row.get("_Base_Exit_Date"))
            if exit_date:
                close_exit_schedule[exit_date].append(trial_leg)
            return
        action = str(row.get("V65_Lifecycle_Action", ""))
        if action == "14日内高质量确认_试仓升级":
            exit_date = normalize_date(row.get("_Base_Exit_Date"))
            if exit_date:
                close_exit_schedule[exit_date].append(trial_leg)
            add_date = normalize_date(row.get("CrossEntry_Date"))
            if (trial_weight < 1.0 - 1e-9
                    and add_date and to_bool(row.get("CrossEntry_Has_40D"))):
                add_schedule[add_date].append(event_id)
                counters["高质量补仓资格"] += 1
            elif add_date:
                counters["高质量确认全额持仓继续持有"] += 1
            return
        if action == "14日内普通确认_只保留试仓":
            counters["已买入的普通确认事件"] += 1
            score_value = finite_num(row.get("_Score_221"))
            keep_trial = (
                ordinary_keep_min_score is not None
                and math.isfinite(score_value)
                and score_value >= ordinary_keep_min_score)
            if keep_trial:
                counters["普通确认评分达标保留"] += 1
                exit_date = normalize_date(row.get("_Base_Exit_Date"))
                if exit_date:
                    close_exit_schedule[exit_date].append(trial_leg)
                if ordinary_risk_days > 0:
                    risk_start = (
                        normalize_date(row.get("CrossImmediate_Decision_Date"))
                        or normalize_date(row.get("Future_Weekly_Cross25_Date")))
                    risk_end = v66_market_date_offset(
                        risk_start, ordinary_risk_days - 1,
                        open_dates, open_pos)
                    if risk_start and risk_end:
                        ordinary_risk_rules[trial_leg] = {
                            "start": risk_start,
                            "end": min(risk_end, exit_date) if exit_date else risk_end,
                            "stop_pct": ordinary_stop_pct,
                            "risk_days": ordinary_risk_days,
                        }
                        counters["普通5分风控监控资格"] += 1
            else:
                counters["普通确认评分未达标腾位"] += 1
                exit_date = normalize_date(row.get("CrossImmediate_Exit_Date"))
                if exit_date:
                    open_exit_schedule[exit_date].append(trial_leg)
                else:
                    close_exit_schedule[normalize_date(
                        row.get("_Base_Exit_Date"))].append(trial_leg)
            return
        if action == "14日内已涨超30_保护试仓不追买":
            exit_date = normalize_date(row.get("_Base_Exit_Date"))
            if exit_date:
                close_exit_schedule[exit_date].append(trial_leg)
            return
        timeout_exit = normalize_date(row.get("Confirm14_Exit_Date"))
        if timeout_exit:
            open_exit_schedule[timeout_exit].append(trial_leg)
        else:
            close_exit_schedule[normalize_date(
                row.get("_Base_Exit_Date"))].append(trial_leg)
        if (allow_late
                and action == "14日退出_迟到高质量确认可重入"
                and to_bool(row.get("Selective_Reentry14_Has_40D"))):
            reentry_date = normalize_date(
                row.get("Selective_Reentry14_Entry_Date"))
            if reentry_date:
                reentry_schedule[reentry_date].append(event_id)
                counters["迟到高质量重入资格"] += 1

    for trade_date in simulation_dates:
        # Decisions made earlier execute at this open.  Exits release both a
        # stock-name slot and cash before adds, reentries and new trials.
        for leg_id in list(open_exit_schedule.pop(trade_date, [])):
            leg = active_legs.get(leg_id)
            if leg is None:
                continue
            code = str(leg["ts_code"])
            exit_reason = open_exit_reason_by_leg.get(
                leg_id, "规则退出_下一开盘")
            if sell_leg(leg_id, trade_date, "open", exit_reason):
                if exit_reason.startswith("普通5分"):
                    counters["普通5分风控退出执行"] += 1
                if exit_reason.startswith("第") and "硬失败" in exit_reason:
                    counters["早期硬失败退出执行"] += 1
            else:
                delayed = next_stock_date(code, trade_date)
                if delayed and delayed > trade_date and delayed <= last_date:
                    open_exit_schedule[delayed].append(leg_id)
                    open_exit_reason_by_leg[leg_id] = exit_reason

        for event_id in list(add_schedule.pop(trade_date, [])):
            row = event_by_id.get(event_id)
            if row is None:
                continue
            code = str(row.get("ts_code", ""))
            has_original_leg = any(
                active_legs[leg_id]["Event_ID"] == event_id
                and active_legs[leg_id]["Kind"] == "试仓"
                for leg_id in code_legs.get(code, set()))
            if not has_original_leg:
                counters["高质量补仓未执行_原仓已无"] += 1
                continue
            desired = slot_capital * (1.0 - trial_weight)
            leg_id = buy_leg(
                row, trade_date, "高质量确认补仓", desired,
                selection_rank=selected_rank_by_event.get(event_id))
            if not leg_id:
                counters["高质量补仓未执行_现金或整手不足"] += 1
                continue
            counters["高质量补仓执行"] += 1
            exit_date = normalize_date(row.get("_Add_Exit_Date"))
            if exit_date:
                close_exit_schedule[exit_date].append(leg_id)

        for event_id in list(reentry_schedule.pop(trade_date, [])):
            row = event_by_id.get(event_id)
            if row is None:
                continue
            code = str(row.get("ts_code", ""))
            if code in code_legs:
                counters["迟到高质量重入未执行_同股已持有"] += 1
                continue
            if len(code_legs) >= V66_MAX_STOCKS:
                counters["迟到高质量重入未执行_无股票名额"] += 1
                continue
            leg_id = buy_leg(
                row, trade_date, "迟到高质量满仓重入", slot_capital,
                selection_rank=selected_rank_by_event.get(event_id))
            if not leg_id:
                counters["迟到高质量重入未执行_现金或整手不足"] += 1
                continue
            counters["迟到高质量重入执行"] += 1
            exit_date = normalize_date(row.get("_Reentry_Exit_Date"))
            if exit_date:
                close_exit_schedule[exit_date].append(leg_id)

        candidates = list(candidates_by_date.get(trade_date, []))
        counters["候选事件"] += len(candidates)
        counters["候选中的高质量确认事件"] += sum(
            str(row.get("V65_Lifecycle_Action", ""))
            == "14日内高质量确认_试仓升级" for row in candidates)
        if selection_mode == "规则排序":
            tie = {str(row["_Event_ID"]): float(rng.random())
                   for row in candidates}

            def macd_key(row: dict[str, Any]) -> tuple[int, float]:
                value = finite_num(row.get("Daily_MACD_Hist"))
                return ((0, value) if math.isfinite(value) else (1, 0.0))

            if (primary_rank_mode == "WalkForwardSABF"
                    and candidates
                    and all(to_bool(row.get("V611_Model_Available"))
                            and math.isfinite(finite_num(
                                row.get("V611_OOS_Score")))
                            for row in candidates)):
                candidates.sort(key=lambda row: (
                    -finite_num(row.get("V611_OOS_Score")),
                    -finite_num(row.get("_Score_221")),
                    *macd_key(row), tie[str(row["_Event_ID"])]))
                counters["走步模型排序候选日"] += 1
            else:
                candidates.sort(key=lambda row: (
                    -finite_num(row.get("_Score_221")),
                    *macd_key(row), tie[str(row["_Event_ID"])]))
                if primary_rank_mode == "WalkForwardSABF" and candidates:
                    counters["训练不足回退基线候选日"] += 1
        else:
            rng.shuffle(candidates)
        for selection_rank, row in enumerate(candidates, start=1):
            code = str(row.get("ts_code", ""))
            is_high = str(row.get("V65_Lifecycle_Action", "")) == (
                "14日内高质量确认_试仓升级")
            # The Top3 gate is based on the original same-day candidate order.
            # A blocked/held higher-ranked stock never promotes a lower-ranked
            # candidate; this admission rule is frozen across both schemes.
            if selection_rank > max_signal_day_rank:
                counters["新试仓未执行_超过方案顺位"] += 1
                if is_high:
                    counters["错过高质量确认_超过方案顺位"] += 1
                continue
            if code in code_legs:
                counters["新试仓未执行_同股已持有"] += 1
                if is_high:
                    counters["错过高质量确认_同股已持有"] += 1
                continue
            if len(code_legs) >= V66_MAX_STOCKS:
                counters["新试仓未执行_无股票名额"] += 1
                if is_high:
                    counters["错过高质量确认_无股票名额"] += 1
                continue
            desired = slot_capital * trial_weight
            entry_kind = (
                "全额提前买入" if trial_weight >= 1.0 - 1e-9
                else ("试仓" if staged else "满仓提前买入"))
            leg_id = buy_leg(
                row, trade_date, entry_kind,
                desired, row.get("Entry_Raw_Open"),
                selection_rank=selection_rank)
            if not leg_id:
                counters["新试仓未执行_现金或整手不足"] += 1
                if is_high:
                    counters["错过高质量确认_现金或整手不足"] += 1
                continue
            event_id = str(row["_Event_ID"])
            selected_events.add(event_id)
            selected_rank_by_event[event_id] = int(selection_rank)
            counters["实际初始买入事件"] += 1
            if (primary_rank_mode == "WalkForwardSABF"
                    and to_bool(row.get("V611_Model_Available"))):
                counters["走步模型实际买入事件"] += 1
            elif primary_rank_mode == "WalkForwardSABF":
                counters["训练不足回退实际买入事件"] += 1
            if is_high:
                counters["实际买入后高质量确认事件"] += 1
            schedule_selected_event(row, leg_id)
            if early_failure_day > 0:
                check_date = v66_market_date_offset(
                    trade_date, early_failure_day - 1,
                    open_dates, open_pos)
                if check_date and check_date <= last_date:
                    early_failure_schedule[check_date].append(leg_id)
                    counters["早期硬失败检查资格"] += 1

        # Update mark-to-market closes, then execute fixed-horizon close sales.
        for code in list(code_legs):
            close_value = exact_price(code, trade_date, "close")
            if math.isfinite(close_value) and close_value > 0:
                last_close[code] = close_value

        # V6.12 hard-failure milestone.  Entry day is D1; the configured D3,
        # D5 or D7 close is inspected once and any decision executes at the
        # next available open.  A confirmation protects the holding only when
        # that cross has actually occurred by this close.
        for leg_id in list(early_failure_schedule.pop(trade_date, [])):
            leg = active_legs.get(leg_id)
            if leg is None:
                continue
            row = event_by_id.get(str(leg.get("Event_ID")))
            if row is None:
                continue
            action = str(row.get("V65_Lifecycle_Action", ""))
            cross_date = normalize_date(row.get("Future_Weekly_Cross25_Date"))
            score_value = finite_num(row.get("_Score_221"))
            cross_observed = bool(cross_date and cross_date <= trade_date)
            protected = bool(cross_observed and (
                action in {
                    "14日内高质量确认_试仓升级",
                    "14日内已涨超30_保护试仓不追买",
                }
                or (action == "14日内普通确认_只保留试仓"
                    and ordinary_keep_min_score is not None
                    and math.isfinite(score_value)
                    and score_value >= ordinary_keep_min_score)))
            if protected:
                counters["早期硬失败检查_确认后受保护"] += 1
                continue
            code = str(leg.get("ts_code", ""))
            raw_close = exact_price(code, trade_date, "close")
            raw_entry = finite_num(leg.get("Raw_Entry"))
            hist = exact_price(code, trade_date, "macd_hist")
            remaining = exact_price(code, trade_date, "macd_remaining")
            retention = exact_price(code, trade_date, "macd_retention")
            raw_return = (
                (raw_close / raw_entry - 1.0) * 100.0
                if (math.isfinite(raw_close) and raw_close > 0
                    and math.isfinite(raw_entry) and raw_entry > 0)
                else np.nan)
            macd_failure = bool(
                math.isfinite(hist) and (
                    hist <= 0
                    or (hist > 0 and math.isfinite(remaining)
                        and remaining <= early_macd_remaining_pct
                        and math.isfinite(retention) and retention < 100.0)))
            price_failure = bool(
                math.isfinite(raw_return)
                and raw_return <= early_price_stop_pct)
            if not (macd_failure or price_failure):
                counters["早期硬失败检查_通过"] += 1
                continue
            next_date = next_stock_date(code, trade_date)
            if not next_date or next_date > last_date:
                counters["早期硬失败触发但无后续开盘"] += 1
                continue
            cause = (
                "MACD及跌幅" if macd_failure and price_failure
                else ("MACD" if macd_failure else "跌幅"))
            open_exit_schedule[next_date].append(leg_id)
            open_exit_reason_by_leg[leg_id] = (
                f"第{early_failure_day}日硬失败_{cause}_下一开盘")
            counters["早期硬失败触发"] += 1
            counters[f"早期硬失败触发_{cause}"] += 1

        for leg_id in list(close_exit_schedule.pop(trade_date, [])):
            leg = active_legs.get(leg_id)
            if leg is None:
                continue
            code = str(leg["ts_code"])
            if not sell_leg(
                    leg_id, trade_date, "close", "固定观察期收盘退出"):
                delayed = next_stock_date(code, trade_date)
                if delayed and delayed <= last_date:
                    close_exit_schedule[delayed].append(leg_id)

        # Only after an ordinary confirmation is actually known do the W1/W2
        # controls start observing the retained score-5 trial.  The decision
        # uses that day's close and always executes at the next tradable open.
        for leg_id, rule in list(ordinary_risk_rules.items()):
            leg = active_legs.get(leg_id)
            if leg is None:
                ordinary_risk_rules.pop(leg_id, None)
                continue
            if not (str(rule["start"]) <= trade_date <= str(rule["end"])):
                continue
            code = str(leg["ts_code"])
            raw_close = exact_price(code, trade_date, "close")
            raw_entry = finite_num(leg.get("Raw_Entry"))
            if (not math.isfinite(raw_close) or raw_close <= 0
                    or not math.isfinite(raw_entry) or raw_entry <= 0):
                continue
            raw_return = (raw_close / raw_entry - 1.0) * 100.0
            if raw_return > float(rule["stop_pct"]):
                continue
            next_date = next_stock_date(code, trade_date)
            if not next_date or next_date > last_date:
                counters["普通5分风控触发但无后续开盘"] += 1
                continue
            window_label = "W1" if int(rule["risk_days"]) <= 5 else "W1W2"
            open_exit_schedule[next_date].append(leg_id)
            open_exit_reason_by_leg[leg_id] = (
                f"普通5分确认后{window_label}跌破"
                f"{abs(float(rule['stop_pct'])):g}%_下一开盘")
            ordinary_risk_rules.pop(leg_id, None)
            counters["普通5分风控触发"] += 1

        invested = 0.0
        for leg in active_legs.values():
            close_value = finite_num(last_close.get(str(leg["ts_code"])))
            if math.isfinite(close_value) and close_value > 0:
                invested += float(leg["Shares"]) * close_value * sell_factor
        equity = cash + invested
        equity_rows.append({
            "trade_date": trade_date, "Equity": equity, "Cash": cash,
            "Invested_Liquidation_Value": invested,
            "Exposure_pct": invested / equity * 100.0 if equity > 0 else np.nan,
            "Held_Stocks": len(code_legs), "Active_Legs": len(active_legs),
        })

    # A mature result should normally have no position left.  Force-close only
    # as a transparent safety net if a suspension or edge date left one open.
    final_date = simulation_dates[-1]
    for leg_id in list(active_legs):
        sell_leg(
            leg_id, final_date, "close", "观察截止强制退出",
            allow_stale_close=True)
        counters["观察截止强制退出腿"] += 1
    if equity_rows:
        equity_rows[-1]["Equity"] = cash
        equity_rows[-1]["Cash"] = cash
        equity_rows[-1]["Invested_Liquidation_Value"] = 0.0
        equity_rows[-1]["Exposure_pct"] = 0.0
        equity_rows[-1]["Held_Stocks"] = 0
        equity_rows[-1]["Active_Legs"] = 0

    trade_frame = pd.DataFrame(trades)
    actual_exit_dates = (
        trade_frame["Exit_Date"].astype(str).tolist()
        if not trade_frame.empty and "Exit_Date" in trade_frame else [])
    performance_end = max([core_end] + actual_exit_dates)
    equity_frame = pd.DataFrame(equity_rows)
    equity_frame = equity_frame[
        equity_frame["trade_date"].astype(str).le(performance_end)
    ].reset_index(drop=True)
    final_date = performance_end
    equity = numeric(equity_frame, "Equity")
    running_peak = equity.cummax()
    drawdown = (equity / running_peak - 1.0) * 100.0
    daily_return = equity.pct_change().replace([np.inf, -np.inf], np.nan)
    elapsed_years = max(
        (pd.Timestamp(final_date) - pd.Timestamp(first_date)).days / 365.25,
        1.0 / 365.25)
    total_return = (cash / initial_capital - 1.0) * 100.0
    annualized = ((cash / initial_capital) ** (1.0 / elapsed_years) - 1.0) * 100.0
    trade_returns = numeric(trade_frame, "Net_Return_pct")
    rank_metrics: dict[str, Any] = {}
    if not trade_frame.empty:
        event_trade = trade_frame.groupby("Event_ID", as_index=False).agg(
            Event_Net_PnL=("Net_PnL", "sum"),
            Event_Entry_Cost=("Entry_Cost", "sum"),
            Selection_Rank=("Signal_Day_Selection_Rank", "min"),
        )
        event_trade["Event_Net_Return_pct"] = np.where(
            numeric(event_trade, "Event_Entry_Cost").gt(0),
            numeric(event_trade, "Event_Net_PnL")
            / numeric(event_trade, "Event_Entry_Cost") * 100.0,
            np.nan)
        for rank in (1, 2, 3):
            group = event_trade[numeric(
                event_trade, "Selection_Rank").eq(rank)]
            rank_metrics.update({
                f"入场顺位{rank}事件": len(group),
                f"入场顺位{rank}事件胜率%": numeric(
                    group, "Event_Net_Return_pct").gt(0).mean() * 100.0
                if not group.empty else np.nan,
                f"入场顺位{rank}事件平均收益%": numeric(
                    group, "Event_Net_Return_pct").mean(),
                f"入场顺位{rank}净利润": numeric(
                    group, "Event_Net_PnL").sum(),
                f"入场顺位{rank}收益贡献百分点": numeric(
                    group, "Event_Net_PnL").sum() / initial_capital * 100.0,
            })
        stock_profit = trade_frame.groupby(
            ["ts_code", "name"], as_index=False, dropna=False
        )["Net_PnL"].sum().sort_values("Net_PnL", ascending=False)
        positive_profit = numeric(stock_profit, "Net_PnL").clip(lower=0.0)
        top1_profit = positive_profit.head(1).sum()
        top3_profit = positive_profit.head(3).sum()
    else:
        top1_profit = top3_profit = 0.0
    net_profit = cash - initial_capital
    concentration_metrics = {
        "最大盈利股票净利润": top1_profit,
        "前三盈利股票净利润": top3_profit,
        "最大盈利股票占总净利润%": (
            top1_profit / net_profit * 100.0 if net_profit > 0 else np.nan),
        "前三盈利股票占总净利润%": (
            top3_profit / net_profit * 100.0 if net_profit > 0 else np.nan),
        "静态剔除最大盈利股票后总收益%": (
            (cash - top1_profit) / initial_capital - 1.0) * 100.0,
        "静态剔除前三盈利股票后总收益%": (
            (cash - top3_profit) / initial_capital - 1.0) * 100.0,
    }
    high_total = int(counters["候选中的高质量确认事件"])
    high_selected = int(counters["实际买入后高质量确认事件"])
    metrics = {
        "方案Key": str(scheme["key"]), "组合方案": str(scheme["label"]),
        "选择方式": selection_mode, "Seed": int(seed),
        "同日入场顺位上限": max_signal_day_rank,
        "主排序方式": primary_rank_mode,
        "同分二级排序": tie_break_mode,
        "首次买入仓位比例": trial_weight,
        "早期硬失败检查日": early_failure_day,
        "早期跌幅阈值%": (
            early_price_stop_pct if early_failure_day > 0 else np.nan),
        "早期MACD剩余阈值%": (
            early_macd_remaining_pct if early_failure_day > 0 else np.nan),
        "初始资金": initial_capital, "期末权益": cash,
        "总收益率%": total_return, "年化收益率%": annualized,
        "最大回撤%": drawdown.min(),
        "日收益年化波动%": daily_return.std(ddof=0) * math.sqrt(244) * 100.0,
        "收益回撤比": total_return / abs(drawdown.min())
        if math.isfinite(drawdown.min()) and drawdown.min() < 0 else np.nan,
        "平均资金暴露%": numeric(equity_frame, "Exposure_pct").mean(),
        "平均持股数": numeric(equity_frame, "Held_Stocks").mean(),
        "三只满额持股日比例%": numeric(
            equity_frame, "Held_Stocks").eq(V66_MAX_STOCKS).mean() * 100.0,
        "空仓日比例%": numeric(
            equity_frame, "Held_Stocks").eq(0).mean() * 100.0,
        "候选事件": int(counters["候选事件"]),
        "顺位过滤事件": int(counters["新试仓未执行_超过方案顺位"]),
        "顺位过滤高质量确认": int(
            counters["错过高质量确认_超过方案顺位"]),
        "实际初始买入事件": int(counters["实际初始买入事件"]),
        "走步模型排序候选日": int(counters["走步模型排序候选日"]),
        "训练不足回退基线候选日": int(
            counters["训练不足回退基线候选日"]),
        "走步模型实际买入事件": int(counters["走步模型实际买入事件"]),
        "训练不足回退实际买入事件": int(
            counters["训练不足回退实际买入事件"]),
        "高质量确认候选": high_total,
        "买入后高质量确认": high_selected,
        "错过高质量确认": max(high_total - high_selected, 0),
        "高质量确认捕获率%": high_selected / high_total * 100.0
        if high_total else np.nan,
        "高质量补仓资格": int(counters["高质量补仓资格"]),
        "高质量补仓执行": int(counters["高质量补仓执行"]),
        "迟到高质量重入资格": int(counters["迟到高质量重入资格"]),
        "迟到高质量重入执行": int(counters["迟到高质量重入执行"]),
        "允许迟到高质量重入": allow_late,
        "普通确认风控窗口市场日": ordinary_risk_days,
        "普通确认风控阈值%": (
            ordinary_stop_pct if ordinary_risk_days > 0 else np.nan),
        "普通确认规则": (
            (f"评分={int(ordinary_keep_min_score)}保留"
             if ordinary_keep_min_score == 5.0
             else f"评分≥{int(ordinary_keep_min_score)}保留")
            if ordinary_keep_min_score is not None else "全部腾位"),
        "普通确认保留最低分": (
            ordinary_keep_min_score
            if ordinary_keep_min_score is not None else np.nan),
        "已买入的普通确认事件": int(
            counters["已买入的普通确认事件"]),
        "普通确认评分达标保留": int(
            counters["普通确认评分达标保留"]),
        "普通确认评分未达标腾位": int(
            counters["普通确认评分未达标腾位"]),
        "普通5分风控监控资格": int(
            counters["普通5分风控监控资格"]),
        "普通5分风控触发": int(counters["普通5分风控触发"]),
        "普通5分风控退出执行": int(
            counters["普通5分风控退出执行"]),
        "早期硬失败检查资格": int(counters["早期硬失败检查资格"]),
        "早期硬失败检查_确认后受保护": int(
            counters["早期硬失败检查_确认后受保护"]),
        "早期硬失败检查_通过": int(
            counters["早期硬失败检查_通过"]),
        "早期硬失败触发": int(counters["早期硬失败触发"]),
        "早期硬失败触发_MACD": int(
            counters["早期硬失败触发_MACD"]),
        "早期硬失败触发_跌幅": int(
            counters["早期硬失败触发_跌幅"]),
        "早期硬失败触发_MACD及跌幅": int(
            counters["早期硬失败触发_MACD及跌幅"]),
        "早期硬失败退出执行": int(
            counters["早期硬失败退出执行"]),
        "高质量确认全额持仓继续持有": int(
            counters["高质量确认全额持仓继续持有"]),
        "完成交易腿": len(trade_frame),
        "交易腿胜率%": trade_returns.gt(0).mean() * 100.0
        if not trade_returns.empty else np.nan,
        "最差单腿收益%": trade_returns.min(),
        "观察截止强制退出腿": int(counters["观察截止强制退出腿"]),
        **rank_metrics,
        **concentration_metrics,
    }
    # Store every path's calendar-year result directly in its metrics row.
    # This adds only a few columns per year and avoids retaining millions of
    # daily-equity rows merely to judge year stability.
    annual_frame = equity_frame.copy()
    annual_frame["_Year"] = annual_frame["trade_date"].astype(str).str[:4]
    prior_year_end = initial_capital
    for year, year_group in annual_frame.groupby("_Year", sort=True):
        year_equity = numeric(year_group, "Equity").dropna()
        if year_equity.empty:
            continue
        year_end_equity = float(year_equity.iloc[-1])
        drawdown_values = pd.concat([
            pd.Series([prior_year_end]), year_equity.reset_index(drop=True)
        ], ignore_index=True)
        year_drawdown = (
            drawdown_values / drawdown_values.cummax() - 1.0) * 100.0
        metrics[f"Year_{year}_Return_pct"] = (
            year_end_equity / prior_year_end - 1.0) * 100.0
        metrics[f"Year_{year}_Max_Drawdown_pct"] = year_drawdown.min()
        metrics[f"Year_{year}_Exposure_pct"] = numeric(
            year_group, "Exposure_pct").mean()
        metrics[f"Year_{year}_Average_Held_Stocks"] = numeric(
            year_group, "Held_Stocks").mean()
        prior_year_end = year_end_equity
    if keep_details:
        equity_frame.insert(0, "组合方案", str(scheme["label"]))
        equity_frame.insert(1, "选择方式", selection_mode)
        equity_frame.insert(2, "Seed", int(seed))
    else:
        equity_frame = pd.DataFrame()
        trade_frame = pd.DataFrame()
    return metrics, equity_frame, trade_frame


def v66_ensemble_summary(paths: pd.DataFrame) -> pd.DataFrame:
    def clean_median(frame: pd.DataFrame, column: str) -> float:
        values = numeric(frame, column).dropna()
        return float(values.median()) if not values.empty else np.nan

    rows: list[dict[str, Any]] = []
    for keys, group in paths.groupby(
            ["方案Key", "组合方案", "选择方式"], dropna=False):
        key, label, mode = keys
        returns = numeric(group, "总收益率%")
        drawdown = numeric(group, "最大回撤%")
        rows.append({
            "方案Key": key, "组合方案": label, "选择方式": mode,
            "主排序方式": str(group["主排序方式"].iloc[0]),
            "同分二级排序": str(group["同分二级排序"].iloc[0]),
            "首次买入仓位比例": clean_median(
                group, "首次买入仓位比例"),
            "早期硬失败检查日": clean_median(
                group, "早期硬失败检查日"),
            "路径数": len(group),
            "期末权益中位": clean_median(group, "期末权益"),
            "总收益中位%": returns.median(),
            "总收益第10分位%": returns.quantile(0.10),
            "总收益第90分位%": returns.quantile(0.90),
            "盈利路径比例%": returns.gt(0).mean() * 100.0,
            "最大回撤中位%": drawdown.median(),
            "最差10%路径最大回撤均值%": drawdown[drawdown.le(
                drawdown.quantile(0.10))].mean(),
            "年化收益中位%": clean_median(group, "年化收益率%"),
            "收益回撤比中位": clean_median(group, "收益回撤比"),
            "平均资金暴露中位%": clean_median(group, "平均资金暴露%"),
            "平均持股数中位": clean_median(group, "平均持股数"),
            "空仓日比例中位%": clean_median(group, "空仓日比例%"),
            "同日入场顺位上限": clean_median(
                group, "同日入场顺位上限"),
            "顺位过滤事件中位": clean_median(group, "顺位过滤事件"),
            "顺位过滤高质量确认中位": clean_median(
                group, "顺位过滤高质量确认"),
            "初始买入事件中位": clean_median(group, "实际初始买入事件"),
            "高质量确认捕获率中位%": clean_median(
                group, "高质量确认捕获率%"),
            "高质量确认全额继续持有中位": clean_median(
                group, "高质量确认全额持仓继续持有"),
            "普通确认保留中位": clean_median(
                group, "普通确认评分达标保留"),
            "普通确认腾位中位": clean_median(
                group, "普通确认评分未达标腾位"),
            "普通5分风控触发中位": clean_median(
                group, "普通5分风控触发"),
            "普通5分风控退出中位": clean_median(
                group, "普通5分风控退出执行"),
            "早期硬失败触发中位": clean_median(
                group, "早期硬失败触发"),
            "早期硬失败退出中位": clean_median(
                group, "早期硬失败退出执行"),
            "早期检查受保护中位": clean_median(
                group, "早期硬失败检查_确认后受保护"),
            "早期检查通过中位": clean_median(
                group, "早期硬失败检查_通过"),
            "交易腿胜率中位%": clean_median(group, "交易腿胜率%"),
            "静态剔除最大盈利股票后总收益中位%": clean_median(
                group, "静态剔除最大盈利股票后总收益%"),
            "静态剔除前三盈利股票后总收益中位%": clean_median(
                group, "静态剔除前三盈利股票后总收益%"),
            "最大盈利股票占总净利润中位%": clean_median(
                group, "最大盈利股票占总净利润%"),
            "前三盈利股票占总净利润中位%": clean_median(
                group, "前三盈利股票占总净利润%"),
        })
    return pd.DataFrame(rows)


def v66_score_vs_random_summary(paths: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in paths.groupby(["方案Key", "组合方案"], dropna=False):
        key, label = keys
        fields = (
            "总收益率%", "高质量确认捕获率%", "最大回撤%")
        score = group[group["选择方式"].eq("规则排序")][
            ["Path_No", "Seed", *fields]].copy()
        random = group[group["选择方式"].eq("完全随机选择")][
            ["Path_No", "Seed", *fields]].copy()
        paired = score.merge(
            random, on=["Path_No", "Seed"], how="inner",
            suffixes=("_规则", "_随机"))
        if paired.empty:
            continue
        score_return = numeric(paired, "总收益率%_规则")
        random_return = numeric(paired, "总收益率%_随机")
        delta = score_return - random_return
        rows.append({
            "方案Key": key, "组合方案": label, "配对路径": len(paired),
            "规则总收益中位%": score_return.median(),
            "随机总收益中位%": random_return.median(),
            "规则减随机收益均值百分点": delta.mean(),
            "规则减随机收益中位百分点": delta.median(),
            "规则优于随机比例%": delta.gt(0).mean() * 100.0,
            "规则差值第10分位": delta.quantile(0.10),
            "规则差值第90分位": delta.quantile(0.90),
            "规则高质量捕获率中位%": numeric(
                paired, "高质量确认捕获率%_规则").median(),
            "随机高质量捕获率中位%": numeric(
                paired, "高质量确认捕获率%_随机").median(),
            "规则最大回撤中位%": numeric(
                paired, "最大回撤%_规则").median(),
            "随机最大回撤中位%": numeric(
                paired, "最大回撤%_随机").median(),
        })
    return pd.DataFrame(rows)


def v612_paired_early_failure_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """Compare each D3/D5/D7 exit with the full-slot no-early-exit baseline."""
    baseline_key = "Full100Top3NoEarlyFailure"
    baseline = paths[paths["方案Key"].eq(baseline_key)].copy()
    if baseline.empty:
        return pd.DataFrame()
    metrics = (
        "总收益率%", "最大回撤%", "平均资金暴露%", "高质量确认捕获率%",
        "实际初始买入事件", "空仓日比例%", "静态剔除前三盈利股票后总收益%",
        "早期硬失败退出执行",
    )
    rows: list[dict[str, Any]] = []
    for scheme in V66_PORTFOLIO_SCHEMES:
        if scheme["key"] == baseline_key:
            continue
        challenger = paths[paths["方案Key"].eq(scheme["key"])].copy()
        for mode in ("规则排序", "完全随机选择"):
            base_mode = baseline[baseline["选择方式"].eq(mode)][
                ["Path_No", "Seed", *metrics]].copy()
            new_mode = challenger[challenger["选择方式"].eq(mode)][
                ["Path_No", "Seed", *metrics]].copy()
            paired = base_mode.merge(
                new_mode, on=["Path_No", "Seed"], how="inner",
                suffixes=("_基线", "_方案"))
            if paired.empty:
                continue
            base_return = numeric(paired, "总收益率%_基线")
            new_return = numeric(paired, "总收益率%_方案")
            return_delta = new_return - base_return
            drawdown_delta = (
                numeric(paired, "最大回撤%_方案")
                - numeric(paired, "最大回撤%_基线"))
            rows.append({
                "基线方案Key": baseline_key,
                "对比方案Key": scheme["key"],
                "对比方案": scheme["label"],
                "选择方式": mode,
                "配对路径": len(paired),
                "种子完全一致": bool(
                    len(paired) == len(base_mode) == len(new_mode)
                    and paired["Path_No"].nunique() == len(paired)),
                "基线总收益中位%": base_return.median(),
                "对比总收益中位%": new_return.median(),
                "对比总收益第10分位%": new_return.quantile(0.10),
                "对比总收益第90分位%": new_return.quantile(0.90),
                "对比减基线收益均值百分点": return_delta.mean(),
                "对比减基线收益中位百分点": return_delta.median(),
                "收益差第10分位": return_delta.quantile(0.10),
                "收益差第90分位": return_delta.quantile(0.90),
                "对比优于基线比例%": return_delta.gt(0).mean() * 100.0,
                "最大回撤改善中位百分点": drawdown_delta.median(),
                "资金暴露变化中位百分点": (
                    numeric(paired, "平均资金暴露%_方案")
                    - numeric(paired, "平均资金暴露%_基线")).median(),
                "高质量捕获率变化中位百分点": (
                    numeric(paired, "高质量确认捕获率%_方案")
                    - numeric(paired, "高质量确认捕获率%_基线")).median(),
                "初始买入事件变化中位": (
                    numeric(paired, "实际初始买入事件_方案")
                    - numeric(paired, "实际初始买入事件_基线")).median(),
                "早期硬失败退出增加中位": (
                    numeric(paired, "早期硬失败退出执行_方案")
                    - numeric(paired, "早期硬失败退出执行_基线")).median(),
                "空仓日比例变化中位百分点": (
                    numeric(paired, "空仓日比例%_方案")
                    - numeric(paired, "空仓日比例%_基线")).median(),
                "剔除前三盈利股后收益变化中位百分点": (
                    numeric(paired, "静态剔除前三盈利股票后总收益%_方案")
                    - numeric(paired, "静态剔除前三盈利股票后总收益%_基线")).median(),
            })
    return pd.DataFrame(rows)


def v611_seed_pairing_audit(paths: pd.DataFrame) -> pd.DataFrame:
    """Verify every mode/Path_No reuses one seed across lifecycle schemes."""
    if paths.empty:
        return pd.DataFrame()
    expected = len(V66_PORTFOLIO_SCHEMES)
    grouped = paths.groupby(["选择方式", "Path_No"], as_index=False).agg(
        方案数=("方案Key", "nunique"),
        种子数=("Seed", "nunique"),
        Seed=("Seed", "min"),
    )
    rows: list[dict[str, Any]] = []
    for mode, group in grouped.groupby("选择方式", dropna=False):
        rows.append({
            "选择方式": mode,
            "Path_No数": group["Path_No"].nunique(),
            "应有方案数": expected,
            "每路径最少方案数": int(group["方案数"].min()),
            "每路径最多方案数": int(group["方案数"].max()),
            "方案缺失路径": int(group["方案数"].ne(expected).sum()),
            "种子不一致路径": int(group["种子数"].ne(1).sum()),
            "首个Seed": int(group["Seed"].min()),
            "最后Seed": int(group["Seed"].max()),
            "公平配对通过": bool(
                group["方案数"].eq(expected).all()
                and group["种子数"].eq(1).all()),
        })
    return pd.DataFrame(rows)


def v610_same_score_tie_break_audit(early: pd.DataFrame) -> pd.DataFrame:
    """Event-level diagnostic for exact score ties; never an entry label."""
    required = {
        "Entry_Date", "Timing_Score_221", "ts_code", "Signal_Date",
        "Entry_Close_Return_Net_pct", "Explosion_Class_40D",
    }
    if early.empty or not required.issubset(early.columns):
        return pd.DataFrame()
    frame = early.copy()
    frame["_Tie_Score"] = numeric(frame, "Timing_Score_221")
    frame["_Tie_Return"] = numeric(frame, "Entry_Close_Return_Net_pct")
    frame["_Tie_Year"] = frame["Entry_Date"].astype(str).str[:4]
    frame = frame[
        frame["Entry_Date"].astype(str).str.len().eq(8)
        & frame["_Tie_Score"].notna()
        & frame["_Tie_Return"].notna()
    ].copy()
    if frame.empty:
        return pd.DataFrame()

    modes = (
        ("Random", "同分随机期望", ""),
        ("DailyMACDHistAsc", "同分MACD柱较小优先", "Daily_MACD_Hist"),
        ("KChangeDesc", "同分周K变化较大优先", "Signal_K_Change_1W"),
        ("VolumeRatioDesc", "同分量比较大优先", "Signal_Volume_Ratio_5W"),
    )
    records: list[dict[str, Any]] = []
    for (entry_date, score), group in frame.groupby(
            ["Entry_Date", "_Tie_Score"], dropna=False, sort=True):
        if len(group) < 2:
            continue
        random_return = float(group["_Tie_Return"].mean())
        random_win = float(group["_Tie_Return"].gt(0).mean())
        classes = group["Explosion_Class_40D"].astype(str).str.upper()
        random_class = {
            letter: float(classes.str.startswith(letter).mean())
            for letter in ("S", "A", "B")
        }
        for mode, label, field in modes:
            usable = True
            if mode == "Random":
                chosen_return = random_return
                chosen_win = random_win
                chosen_class = random_class
            else:
                values = numeric(group, field)
                usable = bool(values.notna().any())
                ordered = group.assign(_Secondary=values)
                ascending = mode == "DailyMACDHistAsc"
                ordered = ordered.sort_values(
                    ["_Secondary", "ts_code"],
                    ascending=[ascending, True], na_position="last")
                chosen = ordered.iloc[0]
                chosen_return = float(chosen["_Tie_Return"])
                chosen_win = float(chosen_return > 0)
                chosen_label = str(chosen.get("Explosion_Class_40D", "")).upper()
                chosen_class = {
                    letter: float(chosen_label.startswith(letter))
                    for letter in ("S", "A", "B")
                }
            records.append({
                "同分二级排序": mode, "方案": label,
                "入场日": str(entry_date), "年度": str(entry_date)[:4],
                "2-2-1分": float(score), "同分候选数": len(group),
                "因子可用": usable, "选中固定40日收益%": chosen_return,
                "选中盈利": chosen_win,
                "选中S": chosen_class["S"], "选中A": chosen_class["A"],
                "选中B": chosen_class["B"],
                "随机期望固定40日收益%": random_return,
                "随机期望盈利": random_win,
                "随机期望S": random_class["S"],
                "随机期望A": random_class["A"],
                "随机期望B": random_class["B"],
            })
    detail = pd.DataFrame(records)
    if detail.empty:
        return detail

    rows: list[dict[str, Any]] = []
    periods = [("全部", detail)] + [
        (str(year), part) for year, part in detail.groupby("年度", sort=True)]
    for period, period_frame in periods:
        for (mode, label), group in period_frame.groupby(
                ["同分二级排序", "方案"], sort=False):
            chosen_return = numeric(group, "选中固定40日收益%")
            random_return = numeric(group, "随机期望固定40日收益%")
            rows.append({
                "统计期": period, "同分二级排序": mode, "方案": label,
                "同分组数": len(group),
                "平均每组候选数": numeric(group, "同分候选数").mean(),
                "因子可用组数": int(true_mask(group, "因子可用").sum()),
                "选中固定40日收益均值%": chosen_return.mean(),
                "选中固定40日收益中位%": chosen_return.median(),
                "选中固定40日收益第10分位%": chosen_return.quantile(0.10),
                "选中盈利比例%": numeric(group, "选中盈利").mean() * 100.0,
                "选中S比例%": numeric(group, "选中S").mean() * 100.0,
                "选中A比例%": numeric(group, "选中A").mean() * 100.0,
                "选中B比例%": numeric(group, "选中B").mean() * 100.0,
                "随机期望固定40日收益均值%": random_return.mean(),
                "相对随机期望收益改善百分点": (
                    chosen_return - random_return).mean(),
                "随机期望盈利比例%": numeric(
                    group, "随机期望盈利").mean() * 100.0,
                "随机期望S比例%": numeric(group, "随机期望S").mean() * 100.0,
                "随机期望A比例%": numeric(group, "随机期望A").mean() * 100.0,
                "随机期望B比例%": numeric(group, "随机期望B").mean() * 100.0,
            })
    return pd.DataFrame(rows)


def v66_representative_annual_returns(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    frame = equity.copy()
    frame["年度"] = frame["trade_date"].astype(str).str[:4]
    rows: list[dict[str, Any]] = []
    for keys, path in frame.groupby(
            ["组合方案", "选择方式", "Seed"], dropna=False):
        label, mode, seed = keys
        path = path.sort_values("trade_date")
        prior_close_equity = float(V66_INITIAL_CAPITAL)
        for year, group in path.groupby("年度", sort=True):
            values = numeric(group, "Equity").dropna()
            if values.empty:
                continue
            start_equity = prior_close_equity
            end_equity = float(values.iloc[-1])
            drawdown_values = pd.concat([
                pd.Series([start_equity]), values.reset_index(drop=True)],
                ignore_index=True)
            running_peak = drawdown_values.cummax()
            drawdown = (drawdown_values / running_peak - 1.0) * 100.0
            rows.append({
                "组合方案": label, "选择方式": mode, "Seed": int(seed),
                "年度": year, "年初权益": start_equity, "年末权益": end_equity,
                "年度收益%": (end_equity / start_equity - 1.0) * 100.0,
                "年度最大回撤%": drawdown.min(),
                "年度平均资金暴露%": numeric(group, "Exposure_pct").mean(),
                "年度平均持股数": numeric(group, "Held_Stocks").mean(),
            })
            prior_close_equity = end_equity
    return pd.DataFrame(rows)


def v67_all_path_year_detail(paths: pd.DataFrame) -> pd.DataFrame:
    """Convert compact Year_YYYY metrics into an auditable long table."""
    years = sorted({
        column.split("_")[1]
        for column in paths.columns
        if column.startswith("Year_") and column.endswith("_Return_pct")
    })
    rows: list[dict[str, Any]] = []
    for row in paths.to_dict("records"):
        for year in years:
            return_value = finite_num(row.get(f"Year_{year}_Return_pct"))
            if not math.isfinite(return_value):
                continue
            rows.append({
                "方案Key": row.get("方案Key"),
                "组合方案": row.get("组合方案"),
                "选择方式": row.get("选择方式"),
                "Seed": row.get("Seed"), "Path_No": row.get("Path_No"),
                "年度": year, "年度收益%": return_value,
                "年度最大回撤%": finite_num(
                    row.get(f"Year_{year}_Max_Drawdown_pct")),
                "年度平均资金暴露%": finite_num(
                    row.get(f"Year_{year}_Exposure_pct")),
                "年度平均持股数": finite_num(
                    row.get(f"Year_{year}_Average_Held_Stocks")),
            })
    return pd.DataFrame(rows)


def v67_all_path_year_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in detail.groupby(
            ["方案Key", "组合方案", "选择方式", "年度"],
            dropna=False):
        key, label, mode, year = keys
        returns = numeric(group, "年度收益%").dropna()
        drawdowns = numeric(group, "年度最大回撤%").dropna()
        rows.append({
            "方案Key": key, "组合方案": label,
            "选择方式": mode, "年度": year,
            "路径数": len(group),
            "年度收益均值%": returns.mean(),
            "年度收益中位%": returns.median(),
            "年度收益第10分位%": returns.quantile(0.10),
            "年度收益第90分位%": returns.quantile(0.90),
            "年度盈利路径比例%": returns.gt(0).mean() * 100.0,
            "年度最大回撤中位%": (
                drawdowns.median() if not drawdowns.empty else np.nan),
            "年度平均资金暴露中位%": numeric(
                group, "年度平均资金暴露%").median(),
            "年度平均持股数中位": numeric(
                group, "年度平均持股数").median(),
        })
    return pd.DataFrame(rows)


def v67_action_score_audit(
        early: pd.DataFrame, action: str, label: str) -> pd.DataFrame:
    frame = early[early["V65_Lifecycle_Action"].astype(str).eq(action)].copy()
    rows: list[dict[str, Any]] = []
    for score, group in frame.groupby("Timing_Score_221", dropna=False):
        classes = group["Explosion_Class_40D"].astype(str)
        early_return = numeric(group, "Entry_Close_Return_Net_pct")
        immediate_return = numeric(group, "CrossImmediate_Return_Net_pct")
        upgraded_return = numeric(
            group, "Lifecycle_V65_Trial50UpgradeLateHigh_Return_Net_pct")
        rows.append({
            "动作类型": label, "2-2-1分": score,
            "事件数": len(group), "不同股票": group["ts_code"].nunique(),
            "信号周": group["Signal_Week"].nunique(),
            "S级比例%": classes.eq("S").mean() * 100.0,
            "A或S比例%": classes.isin(["S", "A"]).mean() * 100.0,
            "B级以上比例%": classes.isin(
                ["S", "A", "B"]).mean() * 100.0,
            "早仓40日收益均值%": early_return.mean(),
            "早仓40日收益中位%": early_return.median(),
            "早仓40日胜率%": early_return.gt(0).mean() * 100.0,
            "早仓最大浮盈中位%": numeric(
                group, "Entry_MFE_Net_pct").median(),
            "早仓最大回撤均值%": numeric(
                group, "Entry_MAE_Raw_pct").mean(),
            "立即腾位收益均值%": immediate_return.mean(),
            "立即腾位收益中位%": immediate_return.median(),
            "1/2试仓升级收益均值%": upgraded_return.mean(),
            "1/2试仓升级收益中位%": upgraded_return.median(),
        })
    return pd.DataFrame(rows).sort_values("2-2-1分", ascending=False)


def v67_ordinary_threshold_year_audit(early: pd.DataFrame) -> pd.DataFrame:
    ordinary = early[early["V65_Lifecycle_Action"].astype(str).eq(
        "14日内普通确认_只保留试仓")].copy()
    ordinary["_Year"] = ordinary["Signal_Date"].astype(str).str[:4]
    rows: list[dict[str, Any]] = []
    year_groups: list[tuple[str, pd.DataFrame]] = [("全部", ordinary)]
    year_groups.extend((str(year), group) for year, group in ordinary.groupby("_Year"))
    for year, year_frame in year_groups:
        for threshold in (5, 4, 3):
            score = numeric(year_frame, "Timing_Score_221")
            for decision, mask in (
                    ("达标保留", score.ge(threshold)),
                    ("未达标腾位", score.lt(threshold))):
                group = year_frame[mask].copy()
                if group.empty:
                    continue
                if decision == "达标保留":
                    decision_return = numeric(group, "Entry_Close_Return_Net_pct")
                    hold_days = pd.Series(40.0, index=group.index)
                else:
                    decision_return = numeric(
                        group, "CrossImmediate_Return_Net_pct")
                    hold_days = numeric(group, "CrossImmediate_Hold_Market_Days")
                classes = group["Explosion_Class_40D"].astype(str)
                rows.append({
                    "年度": year, "保留阈值": f"评分≥{threshold}",
                    "决策": decision, "事件数": len(group),
                    "不同股票": group["ts_code"].nunique(),
                    "S级比例%": classes.eq("S").mean() * 100.0,
                    "B级以上比例%": classes.isin(
                        ["S", "A", "B"]).mean() * 100.0,
                    "决策后收益均值%": decision_return.mean(),
                    "决策后收益中位%": decision_return.median(),
                    "决策后盈利比例%": decision_return.gt(0).mean() * 100.0,
                    "决策后收益第10分位%": decision_return.quantile(0.10),
                    "平均持有市场日": hold_days.mean(),
                })
    return pd.DataFrame(rows)


def v68_exact_factor_combo_audit(early: pd.DataFrame) -> pd.DataFrame:
    """Expose all eight observable timing-factor combinations separately."""
    ordinary = early[early["V65_Lifecycle_Action"].astype(str).eq(
        "14日内普通确认_只保留试仓")].copy()
    if ordinary.empty:
        return pd.DataFrame()
    ordinary["_K"] = true_mask(ordinary, "Timing_K15_20_Pass").astype(int)
    ordinary["_Age"] = true_mask(
        ordinary, "Timing_RedAge3_5_Pass").astype(int)
    ordinary["_KD"] = true_mask(
        ordinary, "Timing_WeeklyK_BelowD_Pass").astype(int)
    ordinary["_Year"] = ordinary["Signal_Date"].astype(str).str[:4]
    rows: list[dict[str, Any]] = []
    year_groups: list[tuple[str, pd.DataFrame]] = [("全部", ordinary)]
    year_groups.extend(
        (str(year), group) for year, group in ordinary.groupby("_Year"))
    for year, year_frame in year_groups:
        for keys, group in year_frame.groupby(["_K", "_Age", "_KD"]):
            k_pass, age_pass, kd_pass = (int(value) for value in keys)
            returns = numeric(group, "Entry_Close_Return_Net_pct")
            immediate = numeric(group, "CrossImmediate_Return_Net_pct")
            classes = group["Explosion_Class_40D"].astype(str)
            rows.append({
                "年度": year,
                "精确组合": (
                    f"K15至20={'是' if k_pass else '否'}｜"
                    f"红柱3至5日={'是' if age_pass else '否'}｜"
                    f"周K低于D={'是' if kd_pass else '否'}"),
                "K15至20": bool(k_pass),
                "红柱3至5日": bool(age_pass),
                "周K低于D": bool(kd_pass),
                "2-2-1分": (
                    V63_SCORE_K_WEIGHT * k_pass
                    + V63_SCORE_AGE_WEIGHT * age_pass
                    + V63_SCORE_KD_WEIGHT * kd_pass),
                "事件数": len(group),
                "不同股票": group["ts_code"].nunique(),
                "信号周": group["Signal_Week"].nunique(),
                "S级比例%": classes.eq("S").mean() * 100.0,
                "B级以上比例%": classes.isin(
                    ["S", "A", "B"]).mean() * 100.0,
                "固定40日收益均值%": returns.mean(),
                "固定40日收益中位%": returns.median(),
                "固定40日盈利比例%": returns.gt(0).mean() * 100.0,
                "固定40日收益第10分位%": returns.quantile(0.10),
                "最大浮盈中位%": numeric(
                    group, "Entry_MFE_Net_pct").median(),
                "最大回撤均值%": numeric(
                    group, "Entry_MAE_Raw_pct").mean(),
                "普通确认立即退出收益中位%": immediate.median(),
            })
    return pd.DataFrame(rows).sort_values(
        ["年度", "2-2-1分", "精确组合"],
        ascending=[True, False, True]).reset_index(drop=True)


def v68_score_rank_path_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """Summarize actual rule-order ranks among executed initial entries."""
    score_paths = paths[paths["选择方式"].eq("规则排序")].copy()
    rows: list[dict[str, Any]] = []

    def median_or_nan(values: pd.Series) -> float:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        return float(clean.median()) if not clean.empty else np.nan

    def quantile_or_nan(values: pd.Series, q: float) -> float:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        return float(clean.quantile(q)) if not clean.empty else np.nan

    for keys, group in score_paths.groupby(
            ["方案Key", "组合方案"], dropna=False):
        key, label = keys
        for rank in (1, 2, 3):
            count = numeric(group, f"入场顺位{rank}事件")
            win = numeric(group, f"入场顺位{rank}事件胜率%")
            event_return = numeric(
                group, f"入场顺位{rank}事件平均收益%")
            pnl = numeric(group, f"入场顺位{rank}净利润")
            contribution = numeric(
                group, f"入场顺位{rank}收益贡献百分点")
            rows.append({
                "方案Key": key, "组合方案": label,
                "规则入场顺位": rank, "路径数": len(group),
                "实际买入事件中位": median_or_nan(count),
                "事件胜率中位%": median_or_nan(win),
                "事件平均收益中位%": median_or_nan(event_return),
                "净利润中位": median_or_nan(pnl),
                "账户收益贡献中位百分点": median_or_nan(contribution),
                "账户收益贡献第10分位": quantile_or_nan(
                    contribution, 0.10),
                "账户收益贡献第90分位": quantile_or_nan(
                    contribution, 0.90),
            })
    return pd.DataFrame(rows)


def v68_profit_concentration_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """Static PnL ablation; it does not pretend to reallocate freed slots."""
    rows: list[dict[str, Any]] = []

    def median_or_nan(values: pd.Series) -> float:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        return float(clean.median()) if not clean.empty else np.nan

    for keys, group in paths.groupby(
            ["方案Key", "组合方案", "选择方式"], dropna=False):
        key, label, mode = keys
        total_return = numeric(group, "总收益率%")
        ex_top1 = numeric(group, "静态剔除最大盈利股票后总收益%")
        ex_top3 = numeric(group, "静态剔除前三盈利股票后总收益%")
        rows.append({
            "方案Key": key, "组合方案": label, "选择方式": mode,
            "路径数": len(group),
            "原总收益中位%": median_or_nan(total_return),
            "静态剔除最大盈利股票后收益中位%": median_or_nan(ex_top1),
            "静态剔除前三盈利股票后收益中位%": median_or_nan(ex_top3),
            "剔除最大盈利股票后盈利路径比例%": (
                ex_top1.gt(0).mean() * 100.0),
            "剔除前三盈利股票后盈利路径比例%": (
                ex_top3.gt(0).mean() * 100.0),
            "最大盈利股票占净利润中位%": median_or_nan(numeric(
                group, "最大盈利股票占总净利润%")),
            "前三盈利股票占净利润中位%": median_or_nan(numeric(
                group, "前三盈利股票占总净利润%")),
        })
    return pd.DataFrame(rows)


def v68_representative_stock_profit(
        trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    grouped = trades.groupby(
        ["方案", "选择方式", "Seed", "ts_code", "name"],
        as_index=False, dropna=False).agg(
            交易腿=("Leg_ID", "count"),
            股票净利润=("Net_PnL", "sum"),
            总投入成本=("Entry_Cost", "sum"),
        )
    grouped["股票净收益率%"] = np.where(
        numeric(grouped, "总投入成本").gt(0),
        numeric(grouped, "股票净利润")
        / numeric(grouped, "总投入成本") * 100.0,
        np.nan)
    grouped["路径总净利润"] = grouped.groupby(
        ["方案", "选择方式", "Seed"])["股票净利润"].transform("sum")
    grouped["占路径总净利润%"] = np.where(
        numeric(grouped, "路径总净利润").gt(0),
        numeric(grouped, "股票净利润")
        / numeric(grouped, "路径总净利润") * 100.0,
        np.nan)
    grouped["盈利贡献排名"] = grouped.groupby(
        ["方案", "选择方式", "Seed"])["股票净利润"].rank(
            method="first", ascending=False).astype(int)
    return grouped.sort_values(
        ["方案", "选择方式", "Seed", "盈利贡献排名"]
    ).reset_index(drop=True)


def v611_actual_selected_grade_audit(
        trades: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Audit actual bought events against the 66.7% and S/A/B objectives."""
    if trades.empty or events.empty or "_Event_ID" not in events:
        return pd.DataFrame()
    lookup_columns = [
        "_Event_ID", "Explosion_Class_40D", "V65_Lifecycle_Action",
        "Timing_Score_221", "Signal_Date", "Entry_Date",
    ]
    lookup = events[[
        column for column in lookup_columns if column in events.columns
    ]].drop_duplicates("_Event_ID").rename(columns={"_Event_ID": "Event_ID"})
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(
            ["方案", "选择方式", "Seed"], dropna=False):
        label, mode, seed = keys
        event = group.groupby("Event_ID", as_index=False).agg(
            实际净利润=("Net_PnL", "sum"),
            实际投入成本=("Entry_Cost", "sum"),
            实际买入日=("Entry_Date", "min"),
            实际入场顺位=("Signal_Day_Selection_Rank", "min"),
        ).merge(lookup, on="Event_ID", how="left")
        event["实际事件收益%"] = np.where(
            numeric(event, "实际投入成本").gt(0),
            numeric(event, "实际净利润")
            / numeric(event, "实际投入成本") * 100.0,
            np.nan)
        returns = numeric(event, "实际事件收益%")
        classes = event["Explosion_Class_40D"].astype(str)
        high = event["V65_Lifecycle_Action"].astype(str).eq(
            "14日内高质量确认_试仓升级")
        ordinary5 = (
            event["V65_Lifecycle_Action"].astype(str).eq(
                "14日内普通确认_只保留试仓")
            & numeric(event, "Timing_Score_221").eq(5.0))
        ordered = event.sort_values(
            ["实际买入日", "实际入场顺位", "Event_ID"]
        ).reset_index(drop=True)
        complete_three = len(ordered) // 3
        triple_passes = 0
        if complete_three:
            triple_frame = ordered.iloc[:complete_three * 3].copy()
            triple_frame["_Triple"] = np.arange(len(triple_frame)) // 3
            triple_passes = int(triple_frame.groupby("_Triple")[
                "实际事件收益%"].apply(
                    lambda values: numeric(
                        pd.DataFrame({"value": values}), "value"
                    ).gt(0).sum() >= 2).sum())
        same_day_groups = 0
        same_day_passes = 0
        for _, day_group in ordered.groupby("实际买入日", sort=True):
            if len(day_group) < 3:
                continue
            day_three = day_group.head(3)
            same_day_groups += 1
            same_day_passes += int(
                numeric(day_three, "实际事件收益%").gt(0).sum() >= 2)
        rows.append({
            "组合方案": label, "选择方式": mode, "Seed": int(seed),
            "实际买入事件": len(event),
            "盈利事件": int(returns.gt(0).sum()),
            "实际事件胜率%": returns.gt(0).mean() * 100.0,
            "距三分之二目标百分点": returns.gt(0).mean() * 100.0 - 66.666667,
            "连续每3笔完整组": complete_three,
            "连续3笔中至少2笔盈利组": triple_passes,
            "连续3笔中至少2笔盈利比例%": (
                triple_passes / complete_three * 100.0
                if complete_three else np.nan),
            "同日买满3只批次": same_day_groups,
            "同日3只中至少2只盈利批次": same_day_passes,
            "同日3只中至少2只盈利比例%": (
                same_day_passes / same_day_groups * 100.0
                if same_day_groups else np.nan),
            "S级比例%": classes.eq("S").mean() * 100.0,
            "A或S比例%": classes.isin(["S", "A"]).mean() * 100.0,
            "B级以上比例%": classes.isin(["S", "A", "B"]).mean() * 100.0,
            "F级比例%": classes.eq("F").mean() * 100.0,
            "高质量确认买入事件": int(high.sum()),
            "高质量确认实际胜率%": returns[high].gt(0).mean() * 100.0
            if high.any() else np.nan,
            "普通确认5分买入事件": int(ordinary5.sum()),
            "普通确认5分实际胜率%": returns[ordinary5].gt(0).mean() * 100.0
            if ordinary5.any() else np.nan,
            "总净利润": numeric(event, "实际净利润").sum(),
        })
    return pd.DataFrame(rows)


def v612_early_failure_event_audit(
        events: pd.DataFrame, price_book: dict[str, dict[str, Any]],
        open_dates: list[str], config: dict[str, Any]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit D3/D5/D7 hard failures using only each milestone close."""
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    open_pos = {value: position for position, value in enumerate(open_dates)}
    buy_factor, sell_factor = _cost_factors(config)
    detail_rows: list[dict[str, Any]] = []
    for row in events.to_dict("records"):
        code = str(row.get("ts_code", ""))
        book = price_book.get(code, {})
        stock_dates = list(book.get("dates", []))
        entry_date = normalize_date(row.get("Entry_Date"))
        raw_entry = finite_num(row.get("Entry_Raw_Open"))
        if (not stock_dates or not entry_date or entry_date not in open_pos
                or not math.isfinite(raw_entry) or raw_entry <= 0):
            continue
        action = str(row.get("V65_Lifecycle_Action", ""))
        cross_date = normalize_date(row.get("Future_Weekly_Cross25_Date"))
        score_value = finite_num(row.get("_Score_221"))
        grade = str(row.get("Explosion_Class_40D", ""))
        for day in V612_EARLY_FAILURE_DAYS:
            check_date = v66_market_date_offset(
                entry_date, day - 1, open_dates, open_pos)
            if not check_date:
                continue
            cross_observed = bool(cross_date and cross_date <= check_date)
            protected = bool(cross_observed and (
                action in {
                    "14日内高质量确认_试仓升级",
                    "14日内已涨超30_保护试仓不追买",
                }
                or (action == "14日内普通确认_只保留试仓"
                    and math.isfinite(score_value) and score_value >= 5.0)))
            raw_close = finite_num(book.get("close", {}).get(check_date))
            hist = finite_num(book.get("macd_hist", {}).get(check_date))
            remaining = finite_num(
                book.get("macd_remaining", {}).get(check_date))
            retention = finite_num(
                book.get("macd_retention", {}).get(check_date))
            raw_return = (
                (raw_close / raw_entry - 1.0) * 100.0
                if math.isfinite(raw_close) and raw_close > 0 else np.nan)
            macd_failure = bool(
                math.isfinite(hist) and (
                    hist <= 0
                    or (hist > 0 and math.isfinite(remaining)
                        and remaining <= V612_MACD_REMAINING_PCT
                        and math.isfinite(retention) and retention < 100.0)))
            price_failure = bool(
                math.isfinite(raw_return)
                and raw_return <= V612_PRICE_FAILURE_PCT)
            triggered = bool(not protected and (
                macd_failure or price_failure))
            next_position = bisect.bisect_right(stock_dates, check_date)
            exit_date = (
                stock_dates[next_position]
                if triggered and next_position < len(stock_dates) else "")
            raw_exit = finite_num(book.get("open", {}).get(exit_date))
            exit_return = (
                (raw_exit * sell_factor / (raw_entry * buy_factor) - 1.0)
                * 100.0
                if (exit_date and math.isfinite(raw_exit) and raw_exit > 0)
                else np.nan)
            cause = (
                "MACD及跌幅" if macd_failure and price_failure
                else ("MACD" if macd_failure else (
                    "跌幅" if price_failure else "未触发")))
            detail_rows.append({
                "Event_ID": row.get("_Event_ID"), "ts_code": code,
                "name": row.get("name"), "Signal_Date": row.get("Signal_Date"),
                "Entry_Date": entry_date, "检查市场日": int(day),
                "检查日期": check_date, "当日收盘相对买入开盘%": raw_return,
                "MACD柱": hist, "MACD剩余强度%": remaining,
                "MACD较前日保留%": retention,
                "此前确认受保护": protected,
                "MACD硬失败": macd_failure, "跌幅硬失败": price_failure,
                "硬失败触发": triggered, "触发原因": cause,
                "模拟退出日": exit_date, "模拟净收益%": exit_return,
                "40日等级": grade,
                "是否S或A": grade in {"S", "A"}, "是否F": grade == "F",
                "40日期末是否盈利": finite_num(
                    row.get("Entry_Close_Return_Net_pct")) > 0,
            })
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return pd.DataFrame(), detail
    rows: list[dict[str, Any]] = []
    for day, group in detail.groupby("检查市场日", sort=True):
        triggered = group[true_mask(group, "硬失败触发")].copy()
        all_f = true_mask(group, "是否F")
        all_sa = true_mask(group, "是否S或A")
        trigger_f = true_mask(triggered, "是否F")
        trigger_sa = true_mask(triggered, "是否S或A")
        rows.append({
            "检查市场日": int(day), "可评估事件": len(group),
            "此前确认受保护": int(true_mask(
                group, "此前确认受保护").sum()),
            "硬失败触发事件": len(triggered),
            "触发比例%": len(triggered) / len(group) * 100.0,
            "仅MACD触发": int((
                true_mask(group, "MACD硬失败")
                & ~true_mask(group, "跌幅硬失败")
                & true_mask(group, "硬失败触发")).sum()),
            "仅跌幅触发": int((
                ~true_mask(group, "MACD硬失败")
                & true_mask(group, "跌幅硬失败")
                & true_mask(group, "硬失败触发")).sum()),
            "双重触发": int((
                true_mask(group, "MACD硬失败")
                & true_mask(group, "跌幅硬失败")
                & true_mask(group, "硬失败触发")).sum()),
            "触发中F级比例%": (
                trigger_f.mean() * 100.0 if len(triggered) else np.nan),
            "触发中S或A比例%": (
                trigger_sa.mean() * 100.0 if len(triggered) else np.nan),
            "清除全部F级比例%": (
                trigger_f.sum() / all_f.sum() * 100.0
                if all_f.sum() else np.nan),
            "误杀全部S或A比例%": (
                trigger_sa.sum() / all_sa.sum() * 100.0
                if all_sa.sum() else np.nan),
            "触发组40日期末盈利比例%": (
                true_mask(triggered, "40日期末是否盈利").mean() * 100.0
                if len(triggered) else np.nan),
            "模拟退出收益均值%": numeric(
                triggered, "模拟净收益%").mean(),
            "模拟退出收益中位%": numeric(
                triggered, "模拟净收益%").median(),
        })
    return pd.DataFrame(rows), detail


def v612_actual_f_slot_audit(
        trades: pd.DataFrame, events: pd.DataFrame,
        open_dates: list[str]) -> pd.DataFrame:
    """Measure F slot-days and S/A false exits on representative paths."""
    if trades.empty or events.empty or "_Event_ID" not in events:
        return pd.DataFrame()
    open_pos = {value: position for position, value in enumerate(open_dates)}
    lookup = events[[
        "_Event_ID", "Explosion_Class_40D", "V65_Lifecycle_Action",
        "Timing_Score_221", "Signal_Date",
    ]].drop_duplicates("_Event_ID").rename(columns={"_Event_ID": "Event_ID"})
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(
            ["方案", "选择方式", "Seed"], dropna=False):
        label, mode, seed = keys
        event = group.groupby("Event_ID", as_index=False).agg(
            实际净利润=("Net_PnL", "sum"),
            实际投入成本=("Entry_Cost", "sum"),
            买入日=("Entry_Date", "min"), 退出日=("Exit_Date", "max"),
            退出原因=("Exit_Reason", lambda values: "|".join(
                sorted(set(str(value) for value in values)))),
        ).merge(lookup, on="Event_ID", how="left")
        event["实际事件收益%"] = np.where(
            numeric(event, "实际投入成本").gt(0),
            numeric(event, "实际净利润")
            / numeric(event, "实际投入成本") * 100.0, np.nan)
        event["持有市场日"] = [
            (open_pos.get(normalize_date(exit_date), -1)
             - open_pos.get(normalize_date(entry_date), -1) + 1)
            if (normalize_date(entry_date) in open_pos
                and normalize_date(exit_date) in open_pos) else np.nan
            for entry_date, exit_date in zip(event["买入日"], event["退出日"])
        ]
        classes = event["Explosion_Class_40D"].astype(str)
        f_mask = classes.eq("F")
        sa_mask = classes.isin(["S", "A"])
        early_mask = event["退出原因"].astype(str).str.contains(
            "硬失败", regex=False)
        early = event[early_mask]
        early_f = early["Explosion_Class_40D"].astype(str).eq("F")
        early_sa = early["Explosion_Class_40D"].astype(str).isin(["S", "A"])
        rows.append({
            "组合方案": label, "选择方式": mode, "Seed": int(seed),
            "实际买入事件": len(event),
            "实际事件胜率%": numeric(
                event, "实际事件收益%").gt(0).mean() * 100.0,
            "S级比例%": classes.eq("S").mean() * 100.0,
            "A或S比例%": sa_mask.mean() * 100.0,
            "F级比例%": f_mask.mean() * 100.0,
            "F级事件": int(f_mask.sum()),
            "F级平均持有市场日": numeric(
                event[f_mask], "持有市场日").mean(),
            "F级股票名额占用日": numeric(
                event[f_mask], "持有市场日").sum(),
            "F级净利润": numeric(event[f_mask], "实际净利润").sum(),
            "S或A平均持有市场日": numeric(
                event[sa_mask], "持有市场日").mean(),
            "早期硬失败退出事件": int(early_mask.sum()),
            "早退中F级事件": int(early_f.sum()),
            "早退中F级比例%": (
                early_f.mean() * 100.0 if len(early) else np.nan),
            "买入F级被早退比例%": (
                early_f.sum() / f_mask.sum() * 100.0
                if f_mask.sum() else np.nan),
            "早退中S或A事件": int(early_sa.sum()),
            "买入S或A被早退比例%": (
                early_sa.sum() / sa_mask.sum() * 100.0
                if sa_mask.sum() else np.nan),
            "早退收益均值%": numeric(
                early, "实际事件收益%").mean(),
            "总净利润": numeric(event, "实际净利润").sum(),
        })
    return pd.DataFrame(rows)


def v68_ordinary_score5_risk_event_audit(
        early: pd.DataFrame, price_book: dict[str, dict[str, Any]],
        open_dates: list[str], config: dict[str, Any]) -> pd.DataFrame:
    """Event-level no-future audit matching the portfolio W1/W2 stop rule."""
    events = v66_prepare_portfolio_events(early, open_dates)
    ordinary = events[
        events["V65_Lifecycle_Action"].astype(str).eq(
            "14日内普通确认_只保留试仓")
        & numeric(events, "_Score_221").eq(5.0)
    ].copy()
    if ordinary.empty:
        return pd.DataFrame()
    open_pos = {value: position for position, value in enumerate(open_dates)}
    buy_factor, sell_factor = _cost_factors(config)
    rows: list[dict[str, Any]] = []
    for event in ordinary.to_dict("records"):
        code = str(event.get("ts_code", ""))
        book = price_book.get(code, {})
        stock_dates = list(book.get("dates", []))
        entry_date = normalize_date(event.get("Entry_Date"))
        base_exit = normalize_date(event.get("_Base_Exit_Date"))
        confirm_date = (
            normalize_date(event.get("CrossImmediate_Decision_Date"))
            or normalize_date(event.get("Future_Weekly_Cross25_Date")))
        raw_entry = finite_num(book.get("open", {}).get(
            entry_date, event.get("Entry_Raw_Open")))
        baseline_return = finite_num(event.get("Entry_Close_Return_Net_pct"))
        if (not stock_dates or not entry_date or not base_exit
                or not confirm_date or not math.isfinite(raw_entry)
                or raw_entry <= 0):
            continue
        for risk_days in V68_RISK_WINDOWS:
            trigger_date = ""
            exit_date = base_exit
            exit_field = "close"
            strategy = "固定40日" if risk_days == 0 else (
                "确认后W1跌破-10%" if risk_days == 5
                else "确认后W1W2跌破-10%")
            if risk_days > 0:
                risk_end = v66_market_date_offset(
                    confirm_date, risk_days - 1, open_dates, open_pos)
                if risk_end:
                    decision_end = min(risk_end, base_exit)
                    for decision_date in stock_dates:
                        if not (confirm_date <= decision_date <= decision_end):
                            continue
                        raw_close = finite_num(
                            book.get("close", {}).get(decision_date))
                        if (not math.isfinite(raw_close) or raw_close <= 0
                                or (raw_close / raw_entry - 1.0) * 100.0
                                > V68_ORDINARY_STOP_PCT):
                            continue
                        position = bisect.bisect_right(
                            stock_dates, decision_date)
                        candidate_exit = (
                            stock_dates[position]
                            if position < len(stock_dates) else "")
                        if candidate_exit and candidate_exit <= base_exit:
                            trigger_date = decision_date
                            exit_date = candidate_exit
                            exit_field = "open"
                        break
            raw_exit = finite_num(
                book.get(exit_field, {}).get(exit_date))
            if not math.isfinite(raw_exit) or raw_exit <= 0:
                continue
            strategy_return = (
                raw_exit * sell_factor / (raw_entry * buy_factor) - 1.0
            ) * 100.0
            rows.append({
                "Event_ID": event.get("_Event_ID"),
                "ts_code": code, "name": event.get("name"),
                "信号日": event.get("Signal_Date"),
                "买入日": entry_date, "普通确认日": confirm_date,
                "方案": strategy, "风控窗口市场日": int(risk_days),
                "止损阈值%": (
                    V68_ORDINARY_STOP_PCT if risk_days else np.nan),
                "是否触发": bool(trigger_date),
                "触发收盘日": trigger_date, "退出日": exit_date,
                "退出价格字段": exit_field,
                "策略净收益%": strategy_return,
                "固定40日净收益%": baseline_return,
                "相对固定40日改善百分点": strategy_return - baseline_return,
                "固定40日等级": event.get("Explosion_Class_40D"),
                "固定40日最大回撤%": event.get("Entry_MAE_Raw_pct"),
            })
    return pd.DataFrame(rows)


def v66_run_portfolio_ensemble(
        early: pd.DataFrame, price_book: dict[str, dict[str, Any]],
        open_dates: list[str], config: dict[str, Any], draws: int
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame]:
    events = v66_prepare_portfolio_events(early, open_dates)
    path_rows: list[dict[str, Any]] = []
    draw_count = max(int(draws), 1)
    for scheme in V66_PORTFOLIO_SCHEMES:
        for mode in ("规则排序", "完全随机选择"):
            for path_no in range(draw_count):
                # Common random numbers: the same Path_No reuses the exact
                # same seed across lifecycle schemes and selection modes.
                # Exits change slot availability, so realized later trades
                # can legitimately differ even under the random control.
                seed = V66_RANDOM_SEED + path_no
                metrics, _, _ = v66_run_portfolio_path(
                    events, price_book, open_dates, config, scheme, mode, seed)
                if metrics:
                    metrics["Path_No"] = path_no
                    path_rows.append(metrics)
    paths = pd.DataFrame(path_rows)
    summary = v66_ensemble_summary(paths)
    score_vs_random = v66_score_vs_random_summary(paths)

    representative_equity: list[pd.DataFrame] = []
    representative_trades: list[pd.DataFrame] = []
    if not paths.empty:
        for keys, group in paths.groupby(
                ["方案Key", "组合方案", "选择方式"], dropna=False):
            key, _, mode = keys
            target = numeric(group, "总收益率%").median()
            chosen_index = (numeric(group, "总收益率%") - target).abs().idxmin()
            chosen = group.loc[chosen_index]
            scheme = next(item for item in V66_PORTFOLIO_SCHEMES
                          if item["key"] == key)
            _, equity, trades = v66_run_portfolio_path(
                events, price_book, open_dates, config, scheme, str(mode),
                int(chosen["Seed"]), keep_details=True)
            if not equity.empty:
                representative_equity.append(equity)
            if not trades.empty:
                representative_trades.append(trades)
    equity_export = pd.concat(
        representative_equity, ignore_index=True
    ) if representative_equity else pd.DataFrame()
    trades_export = pd.concat(
        representative_trades, ignore_index=True
    ) if representative_trades else pd.DataFrame()
    annual = v66_representative_annual_returns(equity_export)
    path_year_detail = v67_all_path_year_detail(paths)
    path_year_summary = v67_all_path_year_summary(path_year_detail)
    return (paths, summary, score_vs_random, equity_export, trades_export,
            annual, path_year_detail, path_year_summary)


def v64_realized_strategy_summary(early: pd.DataFrame) -> pd.DataFrame:
    labels = (
        ("提前买入固定40日", "Baseline40"),
        ("14日任意确认_未确认退出", "Confirm14"),
        ("14日退出_任意金叉重入", "AnyReentry14"),
        ("14日退出_只在高质量确认重入", "HighOnlyReentry14"),
        ("状态管理_普通确认立即退出", "State14_OrdinaryImmediate"),
        ("状态管理_普通确认MACD剩余10退出", "State14_OrdinaryMACD10"),
        ("状态管理_普通确认MACD剩余20退出", "State14_OrdinaryMACD20"),
        ("状态管理_普通确认MACD剩余30退出", "State14_OrdinaryMACD30"),
        ("状态管理_MACD20且涨超30立即保护", "State14_MACD20_ProtectOver30"),
    )
    rows: list[dict[str, Any]] = []
    for label, key in labels:
        return_column = f"Lifecycle_{key}_Return_Net_pct"
        if return_column not in early:
            continue
        returns = numeric(early, return_column).dropna()
        if returns.empty:
            continue
        hold_column = f"Lifecycle_{key}_Hold_Market_Days"
        trips_column = f"Lifecycle_{key}_Round_Trips"
        rows.append({
            "可执行方案": label, "有结果事件": len(returns),
            "实际收益均值%": returns.mean(),
            "实际收益中位%": returns.median(),
            "实际盈利比例%": returns.gt(0).mean() * 100.0,
            "实际达到10%比例%": returns.ge(10).mean() * 100.0,
            "实际达到20%比例%": returns.ge(20).mean() * 100.0,
            "实际达到30%比例%": returns.ge(30).mean() * 100.0,
            "实际亏损10%以上比例%": returns.le(-10).mean() * 100.0,
            "收益第10分位%": returns.quantile(0.10),
            "最差10%平均收益%": returns[returns.le(
                returns.quantile(0.10))].mean(),
            "平均持有市场日": (
                numeric(early.loc[returns.index], hold_column).mean()
                if hold_column in early else np.nan),
            "平均完整交易次数": (
                numeric(early.loc[returns.index], trips_column).mean()
                if trips_column in early else np.nan),
        })
    return pd.DataFrame(rows)


def v64_cross_entry_summary(early: pd.DataFrame) -> pd.DataFrame:
    mature = early[true_mask(early, "CrossEntry_Has_40D")].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in mature.groupby(
            ["Cross_Delay_Group", "Future_Cross_State"], dropna=False):
        delay, state = keys
        classes = group["CrossEntry_Class_40D"].astype(str)
        returns = numeric(group, "CrossEntry_Close_Return_Net_pct")
        rows.append({
            "确认速度": delay, "确认日状态": state,
            "确认后新买成熟事件": len(group),
            "不同股票": group["ts_code"].nunique(),
            "确认周": pd.to_datetime(
                group["Future_Weekly_Cross25_Date"].astype(str),
                format="%Y%m%d", errors="coerce").dt.to_period(
                    "W-FRI").nunique(),
            "新买S级比例%": classes.eq("S").mean() * 100.0,
            "新买A或S比例%": classes.isin(["S", "A"]).mean() * 100.0,
            "新买B级以上比例%": classes.isin(
                ["S", "A", "B"]).mean() * 100.0,
            "新买F级比例%": classes.eq("F").mean() * 100.0,
            "新买最大浮盈中位%": numeric(
                group, "CrossEntry_MFE_Net_pct").median(),
            "新买最大回撤均值%": numeric(
                group, "CrossEntry_MAE_Raw_pct").mean(),
            "新买40日收益均值%": returns.mean(),
            "新买40日收益中位%": returns.median(),
            "新买40日盈利比例%": returns.gt(0).mean() * 100.0,
        })
    return pd.DataFrame(rows)


def v64_selective_reentry_summary(early: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    schemes = (
        ("任意后来上穿25都重入", numeric(
            early, "Confirm14_Reentry_Close_Return_Net_pct").notna(),
         "Confirm14_Reentry_", "Confirm14_Combined_Return_Net_pct"),
        ("只在高质量确认时重入", true_mask(
            early, "Selective_Reentry14_Has_40D"),
         "Selective_Reentry14_",
         "Selective_Reentry14_Combined_Return_Net_pct"),
    )
    for label, mask, prefix, combined_column in schemes:
        group = early[mask].copy()
        classes = group[f"{prefix}Class_40D"].astype(str)
        combined = numeric(group, combined_column)
        signal_dates = group[f"{prefix}Signal_Date"].map(normalize_date)
        rows.append({
            "重入方案": label, "成熟重入事件": len(group),
            "不同股票": group["ts_code"].nunique(),
            "重入信号日": signal_dates[signal_dates.str.len().eq(8)].nunique(),
            "重入信号周": pd.to_datetime(
                signal_dates, format="%Y%m%d", errors="coerce").dt.to_period(
                    "W-FRI").nunique(),
            "重入S级比例%": classes.eq("S").mean() * 100.0,
            "重入A或S比例%": classes.isin(["S", "A"]).mean() * 100.0,
            "重入B级以上比例%": classes.isin(
                ["S", "A", "B"]).mean() * 100.0,
            "重入F级比例%": classes.eq("F").mean() * 100.0,
            "重入最大浮盈中位%": numeric(
                group, f"{prefix}MFE_Net_pct").median(),
            "重入最大回撤均值%": numeric(
                group, f"{prefix}MAE_Raw_pct").mean(),
            "重入40日收益均值%": numeric(
                group, f"{prefix}Close_Return_Net_pct").mean(),
            "重入40日收益中位%": numeric(
                group, f"{prefix}Close_Return_Net_pct").median(),
            "退出再重入复合收益均值%": combined.mean(),
            "退出再重入复合收益中位%": combined.median(),
            "退出再重入复合盈利比例%": combined.gt(0).mean() * 100.0,
        })
    return pd.DataFrame(rows)


def v64_reentry_week_coverage(
        early: pd.DataFrame, calendar: pd.DataFrame,
        start_date: str, end_date: str) -> pd.DataFrame:
    all_weeks = set(calendar["Signal_Week"].astype(str))
    base_weeks = set(early["Signal_Week"].astype(str)) & all_weeks
    rows: list[dict[str, Any]] = []
    schemes = (
        ("原3%提前信号", None),
        ("3%提前信号+任意金叉重入", "Confirm14_Reentry_Signal_Date"),
        ("3%提前信号+高质量确认重入", "Selective_Reentry14_Signal_Date"),
    )
    for label, column in schemes:
        if column is None:
            reentry_events = 0
            reentry_weeks: set[str] = set()
        else:
            dates = early[column].map(normalize_date)
            dates = dates[
                dates.str.len().eq(8) & dates.between(start_date, end_date)]
            reentry_events = len(dates)
            reentry_weeks = set(pd.to_datetime(
                dates, format="%Y%m%d", errors="coerce").dt.to_period(
                    "W-FRI").dropna().astype(str)) & all_weeks
        combined_weeks = base_weeks | reentry_weeks
        rows.append({
            "覆盖方案": label, "原3%信号周": len(base_weeks),
            "区间内重入事件": reentry_events,
            "重入信号周": len(reentry_weeks),
            "与原信号周重合": len(reentry_weeks & base_weeks),
            "实际填补原空窗周": len(reentry_weeks - base_weeks),
            "合并后有交易信号周": len(combined_weeks),
            "合并后空窗周": len(all_weeks - combined_weeks),
        })
    return pd.DataFrame(rows)


def v64_live_lifecycle_status(
        frame: pd.DataFrame, latest_market_date: str) -> pd.Series:
    statuses = pd.Series("", index=frame.index, dtype=object)
    for index, row in frame.iterrows():
        cross_date = normalize_date(row.get("Future_Weekly_Cross25_Date"))
        if cross_date:
            signal_date = normalize_date(row.get("Signal_Date"))
            delay = (
                (pd.Timestamp(cross_date) - pd.Timestamp(signal_date)).days
                if signal_date else 999)
            state = str(row.get(
                "Future_Cross_State", V64_CROSS_ORDINARY))
            on_time = delay <= V64_PRIMARY_CONFIRM_DAYS or to_bool(
                row.get("Confirm14_Confirmed"))
            if on_time and state == V64_CROSS_HIGH:
                statuses.loc[index] = "14日内高质量确认_试仓升级满仓"
            elif on_time and state == V64_CROSS_OVERHEATED:
                statuses.loc[index] = "14日内已涨超30_保护试仓_禁止追买"
            elif on_time:
                statuses.loc[index] = "14日内普通确认_只保留试仓"
            elif state == V64_CROSS_HIGH:
                statuses.loc[index] = "14日退出后高质量确认_允许满仓重入"
            else:
                statuses.loc[index] = "14日退出后非高质量确认_禁止重入"
            continue
        signal_date = normalize_date(row.get("Signal_Date"))
        elapsed = (
            (pd.Timestamp(latest_market_date) - pd.Timestamp(signal_date)).days
            if signal_date else 999)
        statuses.loc[index] = (
            "等待14日周线确认" if elapsed <= V64_PRIMARY_CONFIRM_DAYS
            else "14日仍未确认_退出观察")
    return statuses


def v63_weighted_rank_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    schemes = (
        ("2-2-1评分同日Top3", "Score_Top3_Expected_Weight"),
        ("同日随机Top3", "Random_Top3_Expected_Weight"),
        ("2-2-1评分同日前20%", "Score_Top20_Expected_Weight"),
        ("同日随机前20%", "Random_Top20_Expected_Weight"),
    )
    for label, weight_column in schemes:
        if weight_column not in frame:
            continue
        weights = numeric(frame, weight_column).fillna(0.0)
        total = float(weights.sum())
        if total <= 0:
            continue
        classes = frame["Explosion_Class_40D"].astype(str)

        def weighted(column: str) -> float:
            values = numeric(frame, column)
            valid = values.notna() & weights.gt(0)
            denominator = float(weights[valid].sum())
            return float((values[valid] * weights[valid]).sum() / denominator) if denominator else np.nan

        rows.append({
            "选择方案": label, "期望入选事件": total,
            "覆盖信号日": int(frame.loc[weights.gt(0), "Signal_Date"].nunique()),
            "S级比例%": float((classes.eq("S") * weights).sum() / total * 100.0),
            "A或S比例%": float((classes.isin(["S", "A"]) * weights).sum() / total * 100.0),
            "B级以上比例%": float((classes.isin(["S", "A", "B"]) * weights).sum() / total * 100.0),
            "F级比例%": float((classes.eq("F") * weights).sum() / total * 100.0),
            "40日最大浮盈均值%": weighted("Entry_MFE_Net_pct"),
            "40日最大回撤均值%": weighted("Entry_MAE_Raw_pct"),
            "40日期末收益均值%": weighted("Entry_Close_Return_Net_pct"),
            "40日期末盈利比例%": float((
                numeric(frame, "Entry_Close_Return_Net_pct").gt(0) * weights
            ).sum() / total * 100.0),
        })
    return pd.DataFrame(rows)


def v63_week_calendar(
        open_dates: list[str], start_date: str, end_date: str,
        mature: pd.DataFrame) -> pd.DataFrame:
    days = [value for value in open_dates if start_date <= value <= end_date]
    calendar = pd.DataFrame({"trade_date": days})
    calendar["Signal_Week"] = pd.to_datetime(
        calendar["trade_date"], format="%Y%m%d").dt.to_period("W-FRI").astype(str)
    weeks = calendar.groupby("Signal_Week")["trade_date"].max().rename(
        "Week_Last_Trading_Date").reset_index()
    for event_type in ("STRENGTH_3", "WEEKLY_CROSS25"):
        counts = mature[mature["Event_Type"].eq(event_type)].groupby("Signal_Week").size()
        weeks[event_type] = weeks["Signal_Week"].map(counts).fillna(0).astype(int)
    return weeks


def v63_coverage_summary(calendar: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_type in ("STRENGTH_3", "WEEKLY_CROSS25"):
        counts = numeric(calendar, event_type)
        rows.append({
            "Event_Type": event_type, "总事件": int(counts.sum()),
            "有信号周": int(counts.gt(0).sum()), "空窗周": int(counts.eq(0).sum()),
            "最长连续空窗周": max_empty_run(counts),
            "每周信号均值": counts.mean(), "每周信号中位": counts.median(),
            "单周最多": int(counts.max()),
            "只有1只的周": int(counts.eq(1).sum()),
            "1至3只的周": int(counts.between(1, 3).sum()),
            "超过20只的周": int(counts.gt(20).sum()),
        })
    return pd.DataFrame(rows)


def v63_active_job_path(signature: str) -> str:
    return os.path.join(V63_JOB_DIR, f"{signature}.active")


def v63_mark_job_active(signature: str) -> None:
    atomic_bytes(json.dumps({
        "signature": signature, "version": V63_VERSION,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }, ensure_ascii=False).encode("utf-8"), v63_active_job_path(signature))


def v63_clear_job_active(signature: str) -> None:
    path = v63_active_job_path(signature)
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        record_error(f"V6.12任务标记清除失败: {exc}")


def v63_is_job_active(signature: str) -> bool:
    return os.path.exists(v63_active_job_path(signature))


def v63_checkpoint_path(signature: str, ts_code: str) -> str:
    return os.path.join(
        V63_CHECKPOINT_DIR, signature,
        f"{str(ts_code).replace('.', '_')}.pkl")


def v63_load_checkpoint(signature: str, ts_code: str) -> dict[str, Any] | None:
    path = v63_checkpoint_path(signature, ts_code)
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
        record_error(f"V6.12兼容检查点损坏 {ts_code}: {exc}")
        return None


def v63_save_checkpoint(
        signature: str, ts_code: str, events: list[dict[str, Any]],
        rejects: dict[str, int]) -> None:
    atomic_pickle({
        "signature": signature, "ts_code": str(ts_code),
        "events": events, "rejects": rejects,
    }, v63_checkpoint_path(signature, ts_code))


def _v611_main_legacy() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=V63_TITLE, layout="wide")
    st.title(V63_TITLE)
    streamlit_version = str(getattr(st, "__version__", "unknown"))
    st.caption(f"{V63_UI_PATCH}｜Streamlit {streamlit_version}")
    with st.expander("V6.11验证口径", expanded=True):
        st.markdown(f"""
- **唯一提前买点**：最近完整周N=6、M=3的SKDJ K位于{EARLY_WEEKLY_K_MIN:g}～{EARLY_WEEKLY_K_MAX:g}；日线MACD红柱扩张或剩余强度≥{STRENGTH_MIN_REMAINING_PCT:g}%；本轮股价首次达到+{V63_STRENGTH_THRESHOLD:g}%后，下一市场交易日开盘试仓。
- **真实三仓**：初始资金30万元，最多同时持有3只股票，每只满仓10万元；试仓也占用1个股票名额，买入数量按100股整手且计入交易成本。
- **冻结Top3**：全部固定1/2试仓、普通确认恰好5分保留40日、禁止迟到重入并保留同日前3；V6.9已否定缩到前2或仅第1，本版不再改变顺位上限。
- **两组主排序对照**：基线为2-2-1主分、完全同分时MACD柱较小优先；挑战组为走步S>A>B>F主排序，模型同分时再依次使用2-2-1和MACD柱较小。
- **严格走步防泄漏**：预测某个入场日时，只允许使用40日结果结束日期严格早于该入场日的事件；至少{V611_MIN_TRAIN_ROWS}个成熟事件和{V611_MIN_TRAIN_DATES}个成熟入场日才启用模型，训练不足自动回退基线。
- **排序目标**：F/B/A/S映射为0/1/2/3，成对权重使用0/1/3/7，S相对F的排序权重最高；2-2-1只是模型特征所表达结构的一部分，不再是挑战组的硬主键。
- **14日内高质量确认**：周线K上穿25当天，日线红柱扩张且本轮累计上涨10～30%，下一开盘用剩余资金补足至满仓；早仓和补仓腿分别观察40个市场日。
- **14日内普通确认**：确认发生后才查看买入时已知的2-2-1分；只有5分保留原试仓且不补仓，其余下一可交易开盘退出腾位。
- **顺位资格不递补**：顺位按当天全部候选的原始排序确定；较高顺位因已持仓、满仓或现金不足未成交时，较低顺位不会自动晋级。
- **已涨超过30%**：只保护已有试仓，不补仓、不把周线确认解释为新追买点。
- **14日未确认**：期限后的下一只可交易开盘退出试仓；无论以后出现何种确认，本版组合都不重入。
- **日内顺序**：先执行既定退出释放名额和现金，再处理高质量补仓，最后才处理当日新试仓；收盘触发的风险退出一律等到下一开盘执行。
- **公平配对**：同一Path_No在两种主排序方案及规则/随机选择中复用完全相同的Seed；完全随机选择忽略主排序，两组随机控制必须逐路径完全一致。
- **随机对照**：随机组只随机排列同日全部候选，买入后的确认分层、固定40日和Top3上限与评分组完全相同。
- **新增诊断**：直接导出模型可用日期上的Top3 S/A/B/F、三只中至少两只盈利比例、实际三仓买入事件胜率、逐日训练成熟边界和最终模型系数。
- **逐年验收**：每一条路径都单独记录各年收益、回撤、资金暴露和持股数；中位代表路径只作交易明细示例。
- **判卷**：S/A/B仍为40日内先到+30/+20/+10且先于-10，其余为F；所有退出与补仓均按当时可见数据和下一可交易开盘执行并计入成本。
- **缓存**：交易日历和股票基础资料72小时，行业成员7天，逐股票行情与检查点不主动过期；应用重启或重新部署仍可能清空实例内存和临时磁盘。
""")

    with st.sidebar:
        st.header("运行参数")
        backtest_days = st.number_input(
            "回测交易日数", 100, 1000, 500, 50, key="v611_days")
        min_price = st.number_input(
            "最低股价（元）", 10.0, 20.0, 10.0, 10.0,
            format="%.0f", key="v611_min_price")
        min_mv = st.number_input(
            "最低流通市值（亿元）", 50.0, 100.0, 50.0, 50.0,
            format="%.0f", key="v611_min_mv")
        signal_end_date = st.date_input(
            "历史买入信号截止（40日判卷）", date(2026, 6, 5),
            key="v611_signal_end")
        market_end_date = st.date_input(
            "最新信号观察截止（默认今天）", date.today(),
            key="v611_market_end")
        pause = st.number_input(
            "接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v611_pause")
        use_cache = st.checkbox(
            "复用行情和72小时基础缓存", True, key="v611_cache")
        st.caption("逐股票行情不设TTL；本版兼容并复用V6.4逐股票检查点。")
        portfolio_draws = st.number_input(
            "每组随机路径数", 100, 1000, V66_DEFAULT_DRAWS, 100,
            key="v611_draws")
        st.caption("共2个主排序方案×2种选择方式；300条时共1200次组合模拟。")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005,
            format="%.3f", key="v611_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01,
            format="%.3f", key="v611_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001,
            format="%.3f", key="v611_transfer")
        if st.button("清除V6.11结果和运行状态", key="v611_clear"):
            shutil.rmtree(V63_RESULT_DIR, ignore_errors=True)
            shutil.rmtree(V63_JOB_DIR, ignore_errors=True)
            st.success("结果和运行状态已清除；逐股票检查点与行情缓存保留。")

    request_payload = {
        "version": V63_VERSION, "days": int(backtest_days),
        "signal_end": signal_end_date.strftime("%Y%m%d"),
        "market_end": market_end_date.strftime("%Y%m%d"),
        "min_price": float(min_price), "min_mv": float(min_mv),
        "commission": float(commission_pct), "stamp": float(stamp_duty_pct),
        "transfer": float(transfer_fee_pct),
        "watch_k": [EARLY_WEEKLY_K_MIN, EARLY_WEEKLY_K_MAX],
        "strength": V63_STRENGTH_THRESHOLD,
        "deadlines": list(V63_CONFIRM_DEADLINES),
        "reentry_window": V63_REENTRY_WINDOW_DAYS,
        "cross_states": [V64_CROSS_HIGH, V64_CROSS_ORDINARY,
                         V64_CROSS_OVERHEATED],
        "post_cross_remaining": list(V64_POST_CROSS_REMAINING),
        "trial_weights": [0.5],
        "initial_capital": V66_INITIAL_CAPITAL,
        "max_stocks": V66_MAX_STOCKS,
        "full_slot_capital": V66_FULL_SLOT_CAPITAL,
        "portfolio_schemes": [item["key"] for item in V66_PORTFOLIO_SCHEMES],
        "entry_rank_caps": [3],
        "primary_rank_modes": [
            str(item["primary_rank_mode"]) for item in V66_PORTFOLIO_SCHEMES],
        "same_score_tie_breaker": "DailyMACDHistAsc",
        "walk_forward_features": list(V611_FEATURES),
        "walk_forward_min_train_rows": V611_MIN_TRAIN_ROWS,
        "walk_forward_min_train_dates": V611_MIN_TRAIN_DATES,
        "walk_forward_label_gain": {"F": 0, "B": 1, "A": 3, "S": 7},
        "portfolio_draws": int(portfolio_draws),
        "common_random_seed": V66_RANDOM_SEED,
        "paired_seed_across_schemes_and_modes": True,
        "upgrade_rule": "14d_high_confirm_add_to_full",
        "ordinary_rule": "keep_exact_score5_to_fixed40",
        "rank_gate_rule": "frozen_top3_original_same_day_rank_no_fallback",
        "primary_rank_rule": "score221_macd_vs_strict_mature40d_walk_forward_sabf",
        "timeout_rule": "exit_trial_and_never_reenter",
        "score_weights": [V63_SCORE_K_WEIGHT, V63_SCORE_AGE_WEIGHT,
                          V63_SCORE_KD_WEIGHT],
    }
    request_signature = stable_signature(request_payload)
    result_path = os.path.join(V63_RESULT_DIR, f"{request_signature}.zip")
    result_name = (
        f"weekly_skdj_walk_forward_sabf_rank_v6_11_"
        f"{int(backtest_days)}d_p{int(min_price)}_mv{int(min_mv)}.zip")
    completed_available = False
    if os.path.exists(result_path):
        try:
            with open(result_path, "rb") as handle:
                saved_result = handle.read()
            completed_available = True
            v63_clear_job_active(request_signature)
            st.success("发现相同参数的已完成结果，可直接下载。")
            render_download(
                saved_result, result_name, f"v611_saved_{request_signature}")
        except Exception as exc:
            st.warning(f"已保存结果读取失败：{exc}")

    secret_token = configured_tushare_token()
    if secret_token:
        token = secret_token
        st.caption("已从Streamlit Secrets读取TUSHARE_TOKEN。")
    else:
        token = st.text_input("Tushare Token", type="password", key="v611_token")
    job_active = v63_is_job_active(request_signature)
    left, right = st.columns(2)
    with left:
        start_clicked = st.button(
            "开始/重新运行V6.11", type="primary", key="v611_run")
    with right:
        stop_clicked = st.button(
            "停止自动续跑", disabled=not job_active, key="v611_stop")
    if stop_clicked:
        v63_clear_job_active(request_signature)
        st.success("已停止；逐股票检查点保留。")
        return
    if start_clicked:
        if market_end_date <= signal_end_date:
            st.error("最新观察截止必须晚于历史信号截止")
            return
        v63_mark_job_active(request_signature)
        job_active = True
    if not token:
        st.info("请输入Token；任务启动后若页面重连，会从逐股票检查点自动续跑。")
        return
    if not job_active:
        st.caption(
            "点击开始运行。" if not completed_available
            else "相同参数结果已可下载；如需覆盖请点击重新运行。")
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
    except Exception as exc:
        st.error(f"确定{int(backtest_days)}个交易日窗口失败：{exc}")
        return
    data_start = (
        pd.Timestamp(signal_start).date() - timedelta(weeks=WARMUP_WEEKS, days=7)
    ).strftime("%Y%m%d")
    try:
        with st.spinner("加载交易日历和历史科技池..."):
            open_dates = load_trade_calendar(data_start, market_end)
            extended_end = (market_end_date + timedelta(days=7)).strftime("%Y%m%d")
            week_last_map = complete_week_last_dates(
                load_trade_calendar(data_start, extended_end))
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return
    if not open_dates:
        st.error("区间内没有市场交易日")
        return
    latest_market_date = max(value for value in open_dates if value <= market_end)
    open_pos = {value: position for position, value in enumerate(open_dates)}
    config = {
        "signal_start": signal_start, "signal_end": signal_end,
        "event_signal_end": market_end, "data_start": data_start,
        "market_end": market_end, "latest_market_date": latest_market_date,
        "min_price": float(min_price), "min_mv": float(min_mv),
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct),
        "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct),
    }
    # Reuse V6.4 stock-level checkpoints: event generation is unchanged and
    # V6.11继续复用V6.4事件行，只在组合层增加严格走步主排序。
    run_signature = stable_signature({
        "version": V65_EVENT_ENGINE_VERSION, **config})
    period_index = build_period_index(memberships)
    active_codes = {
        code for code, code_periods in period_index.items()
        if periods_overlap(code_periods, signal_start, market_end)}
    stocks = stock_basic[stock_basic["ts_code"].isin(active_codes)].copy()
    stocks = stocks[
        ~stocks["list_date"].gt(market_end)
        & ~stocks["delist_date"].lt(data_start)
    ].sort_values("ts_code").reset_index(drop=True)

    event_rows: list[dict[str, Any]] = []
    rejects: dict[str, int] = {}
    checkpoint_hits = price_cache_hits = failures = 0
    progress, status = st.progress(0.0), st.empty()
    last_update = 0.0
    stopped = False
    for number, stock in stocks.iterrows():
        if not v63_is_job_active(request_signature):
            stopped = True
            break
        code = str(stock["ts_code"])
        checkpoint = v63_load_checkpoint(run_signature, code)
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
                    rows, stock_rejects, _ = analyze_stock_v63(
                        stock, period_index.get(code, []), daily, cached_basic,
                        storage_path, week_last_map, open_dates, open_pos,
                        config, bool(use_cache), float(pause))
                    event_rows.extend(rows)
                    merge_counts(rejects, stock_rejects)
                    v63_save_checkpoint(
                        run_signature, code, rows, stock_rejects)
                except Exception as exc:
                    failures += 1
                    record_error(f"V6.11逐股票分析失败 {code}: {exc}")
        processed = number + 1
        now = time.monotonic()
        if (processed == 1 or now - last_update >= UI_HEARTBEAT_SECONDS
                or processed == len(stocks)):
            progress.progress(
                processed / max(len(stocks), 1),
                text=f"已处理{processed}/{len(stocks)}只股票，最近{code}")
            status.caption(
                f"事件{len(event_rows)}；检查点{checkpoint_hits}；"
                f"行情缓存{price_cache_hits}；失败{failures}")
            last_update = now
    progress.empty()
    status.empty()
    if stopped:
        st.warning("任务已停止，逐股票检查点已保留。")
        return

    events_all = pd.DataFrame(event_rows)
    if events_all.empty:
        st.error("本区间没有生成V6.11可用事件。")
        return
    events_all = v63_attach_breadth_and_rank(events_all, open_dates)
    events_all = events_all.sort_values(
        ["Signal_Date", "Event_Type", "ts_code"]).reset_index(drop=True)
    live_watch = events_all[events_all["Event_Type"].eq("LIVE_WATCH")].copy()
    research = events_all[~events_all["Event_Type"].eq("LIVE_WATCH")].copy()
    history = research[research["Signal_Date"].astype(str).le(signal_end)].copy()
    observation = research[
        research["Signal_Date"].astype(str).gt(signal_end)
        & research["Signal_Date"].astype(str).le(latest_market_date)
    ].copy()
    mature = history[
        true_mask(history, "Entry_Tradable")
        & true_mask(history, "Entry_Has_40D")].copy()
    if mature.empty:
        st.error("存在信号，但没有可成交且走完40个市场交易日的成熟事件。")
        return
    mature = add_timing_labels(mature)
    early = mature[mature["Event_Type"].eq("STRENGTH_3")].copy()
    weekly = mature[mature["Event_Type"].eq("WEEKLY_CROSS25")].copy()
    if early.empty or weekly.empty:
        st.error("3%提前买点或周线确认对照为空。")
        return

    weekly["Weekly_Momentum_Tier"] = "普通周线确认"
    weekly.loc[
        weekly["Daily_MACD_State"].astype(str).eq("红柱扩张")
        & numeric(weekly, "Daily_Return_Since_Red_Start_pct").between(10, 30),
        "Weekly_Momentum_Tier"] = "高优先级_红柱扩张且已涨10至30"
    weekly.loc[
        numeric(weekly, "Daily_Return_Since_Red_Start_pct").gt(30),
        "Weekly_Momentum_Tier"] = "风险组_已涨超过30"

    early = v64_add_state_lifecycle_columns(early)
    early = v65_add_staged_position_columns(early)
    with st.spinner("逐入场日构建严格成熟窗S>A>B>F走步排序..."):
        early, walk_forward_training_audit, v611_full_bundle = (
            v611_add_walk_forward_ordinal(early))
        walk_forward_oos_audit = v611_oos_grade_audit(early)
        walk_forward_coefficients = v611_feature_coefficients(
            v611_full_bundle)

    with st.spinner("复用本地行情缓存，构建30万元三仓撮合价格簿..."):
        price_book, portfolio_cache_hits, portfolio_missing = v66_load_price_book(
            early, data_start, market_end)
        # A compatible event checkpoint can occasionally outlive one stock's
        # price pickle.  Re-download only those missing prices, never the full
        # universe, so a portfolio audit is not silently biased by omission.
        recovered_price_files = 0
        still_missing: list[str] = []
        for code in portfolio_missing:
            daily, _, _, _ = fetch_price(
                code, data_start, market_end, True, float(pause))
            if daily.empty:
                still_missing.append(code)
                continue
            daily = normalize_price_frame(daily)
            dates = daily["trade_date"].astype(str).tolist()
            price_book[code] = {
                "dates": dates,
                "open": dict(zip(dates, numeric(daily, "open"))),
                "close": dict(zip(dates, numeric(daily, "close"))),
            }
            recovered_price_files += 1
        portfolio_missing = still_missing
    if portfolio_missing:
        st.error(
            f"三仓组合回测缺少{len(portfolio_missing)}只股票的完整行情，"
            "为避免样本偏差本次不继续。请保留检查点后重新运行。")
        st.dataframe(pd.DataFrame({"缺失股票": portfolio_missing}),
                     use_container_width=True, hide_index=True)
        return

    with st.spinner(
            f"运行真实三仓组合：2个主排序方案×2种选择×"
            f"{int(portfolio_draws)}条路径..."):
        (portfolio_paths, portfolio_summary, score_vs_random,
         representative_equity, representative_trades,
         representative_annual, all_path_year_detail,
         all_path_year_summary) = v66_run_portfolio_ensemble(
            early, price_book, open_dates, config, int(portfolio_draws))
    if portfolio_paths.empty:
        st.error("成熟事件存在，但三仓组合未能生成可执行路径。")
        return

    early["Calendar_Year"] = early["Signal_Date"].astype(str).str[:4]
    paired_primary_rank = v611_paired_primary_rank_summary(portfolio_paths)
    seed_pairing_audit = v611_seed_pairing_audit(portfolio_paths)
    score_rank_summary = v68_score_rank_path_summary(portfolio_paths)
    profit_concentration = v68_profit_concentration_summary(portfolio_paths)
    representative_stock_profit = v68_representative_stock_profit(
        representative_trades)
    portfolio_event_audit = v66_prepare_portfolio_events(early, open_dates)
    actual_selected_grade_audit = v611_actual_selected_grade_audit(
        representative_trades, portfolio_event_audit)
    ordinary_factor_combo = v68_exact_factor_combo_audit(early)
    gate_summary = v63_gate_summary(early)
    realized_strategy = v64_realized_strategy_summary(early)
    ordinary_score_audit = v67_action_score_audit(
        early, "14日内普通确认_只保留试仓", "普通确认")
    high_score_audit = v67_action_score_audit(
        early, "14日内高质量确认_试仓升级", "高质量确认")
    cross_state_outcomes = timing_outcome_summary(
        early, ["Cross_Delay_Group", "Future_Cross_State"],
        "提前买入后按确认速度和确认日状态分组")
    cross_entry_outcomes = v64_cross_entry_summary(early)
    selective_reentry = v64_selective_reentry_summary(early)
    cross_state_year = timing_outcome_summary(
        early.assign(
            Calendar_Year=lambda x: x["Signal_Date"].astype(str).str[:4]),
        ["Calendar_Year", "Future_Cross_State"],
        "确认日状态逐年稳定性")
    gate_status_parts: list[pd.DataFrame] = []
    for deadline in V63_CONFIRM_DEADLINES:
        part = timing_outcome_summary(
            early, [f"Confirm{int(deadline)}_Status"],
            f"原40日等级按{int(deadline)}日确认状态分组")
        part.insert(0, "确认期限自然日", int(deadline))
        gate_status_parts.append(part)
    gate_status_audit = pd.concat(gate_status_parts, ignore_index=True)

    day_breadth = timing_outcome_summary(
        early, ["Signal_Day_Breadth_Group"], "实盘可见_信号当日广度")
    cumulative_breadth = timing_outcome_summary(
        early, ["Week_Cumulative_Breadth_Group"], "实盘可见_当周截至当日累计")
    trailing_breadth = timing_outcome_summary(
        early, ["Trailing5D_Breadth_Group"], "实盘可见_过去5个市场日")
    future_full_week = timing_outcome_summary(
        early, ["Future_Full_Week_Breadth_Group"],
        "仅事后审计_完整周总数不可用于周中买点")
    weekly_momentum = timing_outcome_summary(
        weekly, ["Weekly_Momentum_Tier"], "周线确认优先级复核")
    year_audit = timing_outcome_summary(
        mature[mature["Event_Type"].isin(["STRENGTH_3", "WEEKLY_CROSS25"])].assign(
            Calendar_Year=lambda x: x["Signal_Date"].astype(str).str[:4]),
        ["Event_Type", "Calendar_Year"], "逐年稳定性")
    calendar = v63_week_calendar(open_dates, signal_start, signal_end, mature)
    coverage = v63_coverage_summary(calendar)
    reentry_week_coverage = v64_reentry_week_coverage(
        early, calendar, signal_start, signal_end)

    recent_start = (
        pd.Timestamp(latest_market_date) - pd.Timedelta(days=14)
    ).strftime("%Y%m%d")
    recent_early = observation[
        observation["Event_Type"].eq("STRENGTH_3")
        & observation["Signal_Date"].astype(str).between(
            recent_start, latest_market_date)].copy()
    recent_weekly = observation[
        observation["Event_Type"].eq("WEEKLY_CROSS25")
        & observation["Signal_Date"].astype(str).between(
            recent_start, latest_market_date)].copy()
    if not recent_early.empty:
        recent_early["Current_Lifecycle_Status"] = v64_live_lifecycle_status(
            recent_early, latest_market_date)
        recent_early = v611_score_live_candidates(
            recent_early, v611_full_bundle)
    high_priority_weekly = weekly[
        weekly["Weekly_Momentum_Tier"].eq(
            "高优先级_红柱扩张且已涨10至30")].copy()

    run_summary = pd.DataFrame([{
        "程序版本": V63_VERSION, "正式信号开始": signal_start,
        "历史信号截止": signal_end, "行情观察截止": market_end,
        "最新市场交易日": latest_market_date,
        "成熟3%提前事件": len(early), "成熟周线确认事件": len(weekly),
        "高优先级周线事件": len(high_priority_weekly),
        "提前事件后来高质量确认": int(
            early["Future_Cross_State"].eq(V64_CROSS_HIGH).sum()),
        "14日内高质量补仓资格": int(early[
            "V65_Lifecycle_Action"].eq("14日内高质量确认_试仓升级").sum()),
        "14日内普通确认5分事件": int((
            early["V65_Lifecycle_Action"].eq(
                "14日内普通确认_只保留试仓")
            & numeric(early, "Timing_Score_221").eq(5.0)).sum()),
        "14日超时": int((~true_mask(early, "Confirm14_Confirmed")).sum()),
        "组合迟到重入方案": 0,
        "同日入场顺位方案": "固定前3",
        "主排序方案": "2-2-1主分/严格走步S>A>B>F",
        "同分二级排序": "固定MACD柱较小优先",
        "走步模型可用事件": int(true_mask(
            early, "V611_Model_Available").sum()),
        "走步模型可用入场日": int(early.loc[true_mask(
            early, "V611_Model_Available"), "Entry_Date"].nunique()),
        "训练不足回退入场日": int(early.loc[~true_mask(
            early, "V611_Model_Available"), "Entry_Date"].nunique()),
        "最终模型训练事件": int(
            v611_full_bundle["train_rows"] if v611_full_bundle else 0),
        "最终模型训练入场日": int(
            v611_full_bundle["train_dates"] if v611_full_bundle else 0),
        "共用随机种子起点": V66_RANDOM_SEED,
        "公平配对是否全部通过": bool(
            not seed_pairing_audit.empty
            and true_mask(seed_pairing_audit, "公平配对通过").all()),
        "3%提前信号周": early["Signal_Week"].nunique(),
        "最近14日3%提前信号": len(recent_early),
        "最近14日周线确认信号": len(recent_weekly),
        "最新观察池股票": len(live_watch),
        "组合初始资金": V66_INITIAL_CAPITAL,
        "最大持股数": V66_MAX_STOCKS,
        "单股满仓额": V66_FULL_SLOT_CAPITAL,
        "每组随机路径": int(portfolio_draws),
        "组合行情缓存命中": portfolio_cache_hits,
        "组合行情补下载": recovered_price_files,
        "组合路径总数": len(portfolio_paths),
        "处理股票数": len(stocks), "检查点恢复": checkpoint_hits,
        "行情缓存命中": price_cache_hits, "失败股票": failures,
    }])
    definitions = pd.DataFrame([
        ("提前试仓", "周线K15至25；日线MACD质量合格；本轮首次上涨3%；次日开盘"),
        ("账户规模", "初始30万元；最多3只股票；每只满仓10万元"),
        ("名额规则", "试仓也占1个股票名额；同股试仓和补仓只占1个名额"),
        ("试仓比例", "所有V6.11组合统一使用1/2试仓"),
        ("主确认期限", "14自然日内上穿25视为按时确认；7日只作速度对照"),
        ("高质量确认", "上穿日红柱扩张且本轮累计上涨10至30%；下一开盘补足满仓"),
        ("普通确认规则", "两个主排序方案均只保留买入时2-2-1恰好5分的原试仓"),
        ("普通5分固定方案", "确认后不再加仓，继续持有原1/2试仓至初始买入后的第40市场日"),
        ("同日顺位方案", "固定只允许原始同日候选顺位前3进入买入判断"),
        ("顺位不递补", "高顺位因已持仓、满仓或现金不足未成交时，低顺位不自动晋级"),
        ("已涨超过30", "只保护已有试仓；不补仓、不追买"),
        ("未确认退出", "14日仍未上穿25，下一可交易开盘退出试仓"),
        ("迟到确认", "无论迟到高质量或普通确认，本版组合一律不重入"),
        ("资本核算", "日序模拟现金、整手股数和持仓市值；所有交易按净值复合"),
        ("当日优先级", "开盘退出→高质量补仓→新试仓；固定期限到期按收盘退出"),
        ("基线主排序", "2-2-1分降序；完全同分时MACD柱较小优先"),
        ("挑战主排序", "严格走步S>A>B>F模型分降序；同分再用2-2-1和MACD柱较小"),
        ("模型标签", "等级序位F/B/A/S=0/1/2/3；成对权重按收益0/1/3/7，优先学习S相对F"),
        ("模型特征", "14项信号时字段转为同入场日百分位；MACD柱先除以股价再比较"),
        ("防泄漏", "每个预测入场日只训练40日结束日期严格早于该日的事件"),
        ("训练不足回退", f"不足{V611_MIN_TRAIN_ROWS}事件或{V611_MIN_TRAIN_DATES}入场日时使用基线，不删候选"),
        ("公平配对", "同一Path_No在两个主排序方案和规则/随机选择中复用同一Seed；随机控制必须一致"),
        ("规则顺位审计", "按每条规则路径的真实候选顺序分别统计第1、第2、第3顺位交易贡献"),
        ("S/A/B用途", "历史已成熟标签只训练未来日期；当前及未来事件不得读取自身40日标签"),
        ("精确因子组合", "K15至20、红柱3至5日、周K低于D拆成全部八种布尔组合，不再只看总分"),
        ("牛股依赖", "静态扣除每条路径最大和前三盈利股票的已实现净利润；不虚构释放仓位后的再投资"),
        ("全路径逐年验收", "每条路径保留各年收益、回撤、资金暴露和持股数"),
        ("历史与实时", "历史只到默认20260605并要求40日成熟；以后只展示观察名单"),
        ("缓存", "基础缓存72小时；行情不设TTL；逐股票事件复用V6.4检查点"),
    ], columns=["项目", "定义"])
    cache_policy = pd.DataFrame([
        ("交易日历", "Streamlit内存", "72小时", "实例重启可能提前消失"),
        ("股票基础资料", "Streamlit内存", "72小时", "实例重启可能提前消失"),
        ("申万科技行业成员", "Streamlit内存", "7天", "实例重启可能提前消失"),
        ("逐股票日线与daily_basic", "应用临时磁盘", "不自动过期", "重新部署可能清空"),
        ("V6.4兼容逐股票检查点", "应用临时磁盘", "不自动过期", "V6.11直接复用相同事件"),
    ], columns=["对象", "位置", "设定时长", "实际边界"])
    rejection_audit = pd.DataFrame([
        {"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())])
    metadata = pd.DataFrame([{
        "程序版本": V63_VERSION, "SKDJ_N": EARLY_PRIMARY_N, "SKDJ_M": SKDJ_M,
        "回测交易日": int(backtest_days), "预热周": WARMUP_WEEKS,
        "历史信号开始": signal_start, "历史信号截止": signal_end,
        "行情截止": market_end, "最低股价": float(min_price),
        "最低流通市值亿元": float(min_mv), "提前涨幅阈值%": V63_STRENGTH_THRESHOLD,
        "确认期限自然日": "/".join(str(v) for v in V63_CONFIRM_DEADLINES),
        "高质量确认": "红柱扩张且累计上涨10至30%",
        "试仓比例": "0.5000",
        "高质量补仓": "补足至100%",
        "普通确认": "两个主排序方案均为恰好5分固定40日",
        "普通确认保留评分": "恰好5分",
        "同日入场顺位上限": "固定3",
        "主排序": "Score221MACDHistAsc/WalkForwardSABF",
        "同分二级排序": "DailyMACDHistAsc",
        "走步最小训练事件": V611_MIN_TRAIN_ROWS,
        "走步最小训练入场日": V611_MIN_TRAIN_DATES,
        "走步特征": "/".join(V611_FEATURES),
        "走步标签收益": "F0/B1/A3/S7",
        "顺位递补": "禁止",
        "共用随机种子起点": V66_RANDOM_SEED,
        "跨方案与选择方式共用Seed": True,
        "迟到确认重入": "全部禁止",
        "评分权重": f"{V63_SCORE_K_WEIGHT}-{V63_SCORE_AGE_WEIGHT}-{V63_SCORE_KD_WEIGHT}",
        "组合初始资金": V66_INITIAL_CAPITAL,
        "组合股票名额": V66_MAX_STOCKS,
        "单股满仓额": V66_FULL_SLOT_CAPITAL,
        "整手股数": V66_BOARD_LOT,
        "组合方案数": len(V66_PORTFOLIO_SCHEMES),
        "每组选择方式路径数": int(portfolio_draws),
        "内存缓存小时": CACHE_TTL_SECONDS / 3600,
        "Streamlit": streamlit_version,
    }])

    detail_columns = [
        "Event_Type", "Strategy_Label", "ts_code", "name", "Signal_Date",
        "Signal_Week", "SW_L1", "SW_L2", "Raw_Close", "Circ_MV_Billion",
        "Setup_Weekly_Date", "Signal_K", "Signal_D", "Signal_KD_Spread",
        "Daily_MACD_State", "Daily_MACD_Red_Age",
        "Daily_MACD_Remaining_pct", "Daily_MACD_Retention_pct",
        "Signal_Rally_From_Red_Start_pct",
        "Signal_Day_Strength3_Count", "Signal_Week_Cumulative_Strength3_Count",
        "Trailing5D_Strength3_Count", "Future_Full_Week_Strength3_Count",
        "Signal_Day_Breadth_Group", "Week_Cumulative_Breadth_Group",
        "Trailing5D_Breadth_Group", "Future_Full_Week_Breadth_Group",
        "Timing_K15_20_Pass", "Timing_RedAge3_5_Pass",
        "Timing_WeeklyK_BelowD_Pass", "Timing_Score_221",
        "Timing_Three_Factor_Count", "Score_Top3_Expected_Weight",
        "Random_Top3_Expected_Weight", "Score_Top20_Expected_Weight",
        "Random_Top20_Expected_Weight",
        "Future_Weekly_Cross25_Within42D", "Future_Weekly_Cross25_Date",
        "Lead_Calendar_Days_to_Weekly_Cross", "Entry_Date", "Entry_Raw_Open",
        "Future_Cross_State", "Future_Cross_Daily_MACD_State",
        "Future_Cross_Rally_From_Red_Start_pct", "Cross_Delay_Group",
        "V65_Lifecycle_Action",
        "CrossEntry_Date", "CrossEntry_Tradable", "CrossEntry_Has_40D",
        "CrossEntry_Class_40D", "CrossEntry_MFE_Net_pct",
        "CrossEntry_MAE_Raw_pct", "CrossEntry_Close_Return_Net_pct",
        "CrossImmediate_Decision_Date", "CrossImmediate_Exit_Date",
        "CrossImmediate_Reason", "CrossImmediate_Return_Net_pct",
        "CrossImmediate_MFE_Net_pct", "CrossImmediate_MAE_Raw_pct",
        "CrossImmediate_Hold_Market_Days",
        "Entry_MFE_Net_pct", "Entry_MAE_Raw_pct",
        "Entry_Close_Return_Net_pct", "Explosion_Class_40D",
    ]
    for threshold in V64_POST_CROSS_REMAINING:
        prefix = f"CrossRemaining{int(threshold)}_"
        detail_columns.extend([
            f"{prefix}Decision_Date", f"{prefix}Exit_Date",
            f"{prefix}Reason", f"{prefix}Return_Net_pct",
            f"{prefix}MFE_Net_pct", f"{prefix}MAE_Raw_pct",
            f"{prefix}Hold_Market_Days",
        ])
    for deadline in V63_CONFIRM_DEADLINES:
        prefix = f"Confirm{int(deadline)}_"
        detail_columns.extend([
            f"{prefix}Status", f"{prefix}Decision_Date", f"{prefix}Confirmed",
            f"{prefix}Cross_Date", f"{prefix}Cross_Delay_Calendar_Days",
            f"{prefix}Exit_Date", f"{prefix}Exit_Reason",
            f"{prefix}Strategy_Return_Net_pct", f"{prefix}Strategy_MFE_Net_pct",
            f"{prefix}Strategy_MAE_Raw_pct", f"{prefix}Strategy_Hold_Market_Days",
            f"{prefix}Reentry_Signal_Date", f"{prefix}Reentry_Entry_Date",
            f"{prefix}Reentry_Class_40D", f"{prefix}Reentry_MFE_Net_pct",
            f"{prefix}Reentry_MAE_Raw_pct",
            f"{prefix}Reentry_Close_Return_Net_pct",
            f"{prefix}Combined_Return_Net_pct",
        ])
    detail_columns.extend([
        "Selective_Reentry14_Qualified",
        "Selective_Reentry14_Signal_Date",
        "Selective_Reentry14_Entry_Date",
        "Selective_Reentry14_Has_40D",
        "Selective_Reentry14_Class_40D",
        "Selective_Reentry14_MFE_Net_pct",
        "Selective_Reentry14_MAE_Raw_pct",
        "Selective_Reentry14_Close_Return_Net_pct",
        "Selective_Reentry14_Combined_Return_Net_pct",
    ])
    detail_columns.extend([
        column for column in early.columns
        if column.startswith("Lifecycle_") and column not in detail_columns
    ])
    detail_columns.extend([
        column for column in early.columns
        if column.startswith("V611_") and column not in detail_columns
    ])
    live_columns = [
        "Event_Type", "Strategy_Label", "ts_code", "name", "Signal_Date",
        "Signal_Week", "SW_L1", "SW_L2", "Raw_Close", "Circ_MV_Billion",
        "Setup_Weekly_Date", "Signal_K", "Signal_D", "Signal_KD_Spread",
        "Daily_MACD_State", "Daily_MACD_Red_Age",
        "Daily_MACD_Remaining_pct", "Signal_Rally_From_Red_Start_pct",
        "Signal_Day_Strength3_Count", "Signal_Week_Cumulative_Strength3_Count",
        "Trailing5D_Strength3_Count", "Timing_Score_221",
        "Timing_Three_Factor_Count", "Score_Top3_Expected_Weight",
        "Future_Cross_State", "Future_Weekly_Cross25_Date",
        "Current_Lifecycle_Status",
        "Watch_Next_Strength_Threshold",
        "V611_Live_Score", "V611_Live_Rank",
        "V611_Live_Model_Available", "V611_Live_Train_Rows",
        "V611_Live_Train_Dates", "V611_Live_Train_Weeks",
    ]
    history_early_export = early[[
        column for column in detail_columns if column in early.columns]].copy()
    history_weekly_export = weekly[[
        column for column in detail_columns + ["Weekly_Momentum_Tier"]
        if column in weekly.columns]].copy()
    recent_early_export = recent_early[[
        column for column in live_columns if column in recent_early.columns]].copy()
    recent_weekly_export = recent_weekly[[
        column for column in live_columns if column in recent_weekly.columns]].copy()
    live_watch_export = live_watch[[
        column for column in live_columns if column in live_watch.columns]].copy()
    portfolio_event_audit = v66_prepare_portfolio_events(early, open_dates)
    files = {
        "01_run_summary_v6_11.csv": run_summary,
        "02_experiment_definitions_v6_11.csv": definitions,
        "03_primary_rank_ensemble_summary_v6_11.csv": portfolio_summary,
        "04_paired_primary_rank_comparison_v6_11.csv": paired_primary_rank,
        "05_rule_vs_random_portfolio_v6_11.csv": score_vs_random,
        "06_all_portfolio_paths_v6_11.csv": portfolio_paths,
        "07_all_path_year_summary_v6_11.csv": all_path_year_summary,
        "08_all_path_year_detail_v6_11.csv": all_path_year_detail,
        "09_representative_path_annual_v6_11.csv": representative_annual,
        "10_representative_path_equity_v6_11.csv": representative_equity,
        "11_representative_path_trades_v6_11.csv": representative_trades,
        "12_actual_selected_grade_and_win_audit_v6_11.csv": actual_selected_grade_audit,
        "13_rule_rank_1_2_3_path_summary_v6_11.csv": score_rank_summary,
        "14_profit_concentration_ablation_v6_11.csv": profit_concentration,
        "15_representative_stock_profit_contribution_v6_11.csv": representative_stock_profit,
        "16_walk_forward_oos_grade_audit_v6_11.csv": walk_forward_oos_audit,
        "17_walk_forward_training_leakage_audit_v6_11.csv": walk_forward_training_audit,
        "18_walk_forward_final_coefficients_v6_11.csv": walk_forward_coefficients,
        "19_common_seed_pairing_audit_v6_11.csv": seed_pairing_audit,
        "20_portfolio_event_audit_v6_11.csv": portfolio_event_audit,
        "21_ordinary_exact_factor_combination_v6_11.csv": ordinary_factor_combo,
        "22_ordinary_confirmation_by_exact_score_v6_11.csv": ordinary_score_audit,
        "23_high_confirmation_by_exact_score_v6_11.csv": high_score_audit,
        "24_early_outcomes_by_cross_state_v6_11.csv": cross_state_outcomes,
        "25_fresh_cross_entry_by_state_v6_11.csv": cross_entry_outcomes,
        "26_weekly_coverage_summary_v6_11.csv": coverage,
        "27_weekly_signal_calendar_v6_11.csv": calendar,
        "28_historical_weekly_control_detail_v6_11.csv": history_weekly_export,
        "29_recent_14d_strength3_candidates_v6_11.csv": recent_early_export,
        "30_recent_14d_weekly_candidates_v6_11.csv": recent_weekly_export,
        "31_latest_market_day_watch_pool_v6_11.csv": live_watch_export,
        "32_cache_policy_v6_11.csv": cache_policy,
        "33_rejection_audit_v6_11.csv": rejection_audit,
        "34_metadata_v6_11.csv": metadata,
        "35_api_errors_v6_11.csv": pd.DataFrame({"错误": API_ERRORS}),
    }
    result_zip = make_zip(files)
    try:
        atomic_bytes(result_zip, result_path)
        v63_clear_job_active(request_signature)
        persisted = True
    except Exception as exc:
        persisted = False
        st.warning(f"结果未能持久保存，但当前页面仍可下载：{exc}")

    st.success(
        f"完成：成熟3%提前事件{len(early)}个，周线确认对照{len(weekly)}个；"
        f"已运行{len(portfolio_paths)}条30万元三仓组合路径；"
        f"14日内高质量补仓资格"
        f"{int(early['V65_Lifecycle_Action'].eq('14日内高质量确认_试仓升级').sum())}个，"
        f"普通确认5分"
        f"{int((early['V65_Lifecycle_Action'].eq('14日内普通确认_只保留试仓') & numeric(early, 'Timing_Score_221').eq(5.0)).sum())}个；"
        f"最近14日3%信号{len(recent_early)}个，最新观察池{len(live_watch)}只；"
        f"结果{'已保存' if persisted else '仅当前页面可下载'}。")
    st.subheader("结论一：原2-2-1与走步S>A>B>F主排序的真实三仓结果")
    render_plain_table(portfolio_summary, 40)
    st.caption(
        "这是日序账户净值，不是事件收益平均。每组的中位、10%和90%分位"
        "同时列出静态剔除最大、前三盈利股票后的结果。")
    st.subheader("结论二：走步主排序相对原2-2-1基线的逐路径配对")
    render_plain_table(paired_primary_rank, 20)
    st.caption(
        "两个方案复用同一Path_No和Seed；完全随机选择下两组路径必须一致，"
        "规则路径差异只能来自走步主排序，不能来自资金、生命周期或候选删减。")
    st.subheader("结论三：实际买入是否达到三分之二盈利与S/A/B目标")
    render_plain_table(actual_selected_grade_audit, 20)
    st.caption("这张表按实际成交事件合并试仓和补仓腿，是66.7%目标的正式验收口径。")
    st.subheader("结论四：模型可用日期上的Top3爆发等级与三只中两只盈利")
    render_plain_table(walk_forward_oos_audit, 30)
    st.caption(
        "只比较走步模型已经具备训练历史的相同入场日；随机行为精确期望，"
        "不删除单股或双股日期，三只中两只指标只在当日候选不少于3只时计算。")
    st.subheader("结论五：两种规则排序是否优于完全随机")
    render_plain_table(score_vs_random, 30)
    st.caption(
        "验收核心是‘规则优于随机比例’和收益差的10%～90%分位；"
        "如果仅中位收益好看但规则优于随机约50%，该排序仍然没有通过。")
    st.subheader("结论六：规则排序实际入场第1/2/3顺位")
    render_plain_table(score_rank_summary, 30)
    st.caption(
        "顺位来自每条规则路径当日的真实候选排序；补仓利润归回原始入场事件，"
        "因此可以直接判断第一、第二、第三名各自贡献。")
    st.subheader("结论七：剔除头部盈利股票后的静态稳健性")
    render_plain_table(profit_concentration, 30)
    st.caption(
        "这里只扣除已实现净利润，不假设腾出的仓位会自动买到下一只股票；"
        "这是保守的利润集中度审计，不是重新撮合后的反事实回测。")
    st.subheader("结论八：全部路径的逐年稳定性")
    render_plain_table(all_path_year_summary, 100)
    st.caption(
        "这张表使用每一条路径，而不是只挑总收益接近中位数的一条路径。"
        "要求有效阈值在各年的收益中位、第10分位和盈利路径比例都不恶化。")
    with st.expander("中位代表路径逐年表（仅供查交易明细）", expanded=False):
        render_plain_table(representative_annual, 50)
    with st.expander("共用随机种子公平性审计", expanded=False):
        render_plain_table(seed_pairing_audit, 20)
    with st.expander("逐入场日训练成熟边界与防泄漏审计", expanded=False):
        render_plain_table(walk_forward_training_audit, 500)
        st.caption(
            "每个模型可用日的训练最大40日结束日期必须严格早于预测入场日；"
            "任何一行不通过，都应判定本版结果无效。")
    with st.expander("最终历史模型特征系数（仅用于最近实时候选）", expanded=False):
        render_plain_table(walk_forward_coefficients, 50)
    with st.expander("三因子精确组合与原分数审计", expanded=False):
        render_plain_table(ordinary_factor_combo, 100)
        render_plain_table(ordinary_score_audit, 20)
        render_plain_table(high_score_audit, 20)
        st.caption(
            "精确组合用于拆开相同总分下的不同条件结构，不能代替真实三仓结果；"
            "V6.11以上方真实现金、名额和持仓净值为准。")
    st.subheader("确认速度与确认日状态")
    render_plain_table(cross_state_outcomes, 30)
    st.caption("上表从3%提前买入者视角判卷；下表从确认后才新买入者视角判卷。")
    render_plain_table(cross_entry_outcomes, 30)
    st.subheader("最近14日3%提前候选")
    render_plain_table(recent_early_export.sort_values(
        [column for column in ["Signal_Date", "Current_Lifecycle_Status",
                               "Signal_Rally_From_Red_Start_pct"]
         if column in recent_early_export.columns],
        ascending=[False, True, False][:
            len([column for column in ["Signal_Date", "Current_Lifecycle_Status",
                                       "Signal_Rally_From_Red_Start_pct"]
                 if column in recent_early_export.columns])]), 300)
    st.subheader(f"最新交易日周线K15至25观察池：{latest_market_date}")
    render_plain_table(live_watch_export.sort_values(
        [column for column in ["Signal_K", "Signal_KD_Spread"]
         if column in live_watch_export.columns]), 300)
    st.subheader("运行摘要与覆盖率")
    render_plain_table(run_summary, 10)
    render_plain_table(coverage, 10)
    st.caption(
        f"结果ZIP共{len(files)}个CSV；历史成熟事件、组合路径、"
        "最近候选和最新观察池严格分开。")
    render_download(result_zip, result_name, f"v611_current_{request_signature}")


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=V63_TITLE, layout="wide")
    st.title(V63_TITLE)
    streamlit_version = str(getattr(st, "__version__", "unknown"))
    st.caption(f"{V63_UI_PATCH}｜Streamlit {streamlit_version}")
    with st.expander("V6.12验证口径", expanded=True):
        st.markdown(f"""
- **唯一提前买点**：最近完整周N=6、M=3的SKDJ K位于{EARLY_WEEKLY_K_MIN:g}～{EARLY_WEEKLY_K_MAX:g}；日线MACD红柱扩张或剩余强度≥{STRENGTH_MIN_REMAINING_PCT:g}%；本轮股价首次达到+{V63_STRENGTH_THRESHOLD:g}%后，下一市场交易日开盘买入。
- **全额等分三仓**：初始资金30万元，最多同时持有3只股票；每个名额一次性投入约10万元，按100股整手并计入成本。只有1～2只合格候选时，其余名额保持现金。
- **不试仓、不加仓**：买入时已经使用完整单股名额；高质量确认只代表继续持有，不再产生第二笔买入。
- **冻结Top3排序**：2-2-1分降序；分数完全相同时，日线MACD柱较小者优先；只允许当天原始顺位前3进入买入判断，顺位不递补。
- **四组生命周期**：无早退基线，以及买入后的第3、第5、第7个市场日收盘检查组；四组的买点、排序、资金和14日确认规则完全相同。
- **早期硬失败**：检查日尚未得到保护，且日线MACD柱≤0；或红柱虽为正但剩余强度≤{V612_MACD_REMAINING_PCT:g}%且较前日保留率<100%；或收盘相对买入开盘≤{V612_PRICE_FAILURE_PCT:g}%。触发后在该股下一可交易日开盘退出。
- **确认保护**：检查日之前已经观察到高质量确认、已涨超30%的确认，或普通确认且买入时2-2-1恰好5分，不执行早期硬失败退出。
- **14日生命周期**：高质量确认继续持有；普通确认只有恰好5分继续持有；其他普通确认及14日未确认者在下一可交易日开盘退出；禁止迟到重入。
- **持有上限**：被保留的股票持有至初始买入后的第40个市场日收盘；早期失败只缩短失败股占用名额的时间，不延长任何股票。
- **日内顺序**：开盘先执行退出释放现金和名额，再处理当天新候选；因此早退释放的约10万元可在同日买入新的完整名额。
- **判卷不参与交易**：S/A/B/F仍按买入后40日内先到+30/+20/+10且先于-10判定，只用于事后检验；交易时只读取当时已经出现的数据。
- **验收重点**：比较D3/D5/D7能否减少F级持有日和亏损，同时不明显误杀S/A，并检查真实三仓总收益、回撤、覆盖、每三只至少两只盈利及头部股票依赖。
- **随机对照与配对**：相同Path_No复用相同Seed；各生命周期方案因退出后名额可用时间不同，后续随机成交可以不同，这是策略路径的真实结果。
""")

    with st.sidebar:
        st.header("运行参数")
        backtest_days = st.number_input(
            "回测交易日数", 100, 1000, 500, 50, key="v612_days")
        min_price = st.number_input(
            "最低股价（元）", 10.0, 20.0, 10.0, 10.0,
            format="%.0f", key="v612_min_price")
        min_mv = st.number_input(
            "最低流通市值（亿元）", 50.0, 100.0, 50.0, 50.0,
            format="%.0f", key="v612_min_mv")
        signal_end_date = st.date_input(
            "历史买入信号截止（40日判卷）", date(2026, 6, 5),
            key="v612_signal_end")
        market_end_date = st.date_input(
            "最新信号观察截止（默认今天）", date.today(),
            key="v612_market_end")
        pause = st.number_input(
            "接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v612_pause")
        use_cache = st.checkbox(
            "复用行情和72小时基础缓存", True, key="v612_cache")
        st.caption("逐股票行情不设TTL；继续复用兼容的逐股票事件检查点。")
        portfolio_draws = st.number_input(
            "每组随机路径数", 100, 1000, V66_DEFAULT_DRAWS, 100,
            key="v612_draws")
        st.caption("共4个生命周期方案×2种选择方式；300条时共2400次组合模拟。")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005,
            format="%.3f", key="v612_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01,
            format="%.3f", key="v612_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001,
            format="%.3f", key="v612_transfer")
        if st.button("清除V6.12结果和运行状态", key="v612_clear"):
            shutil.rmtree(V63_RESULT_DIR, ignore_errors=True)
            shutil.rmtree(V63_JOB_DIR, ignore_errors=True)
            st.success("结果和运行状态已清除；逐股票检查点与行情缓存保留。")

    request_payload = {
        "version": V63_VERSION,
        "days": int(backtest_days),
        "signal_end": signal_end_date.strftime("%Y%m%d"),
        "market_end": market_end_date.strftime("%Y%m%d"),
        "min_price": float(min_price), "min_mv": float(min_mv),
        "commission": float(commission_pct),
        "stamp": float(stamp_duty_pct),
        "transfer": float(transfer_fee_pct),
        "watch_k": [EARLY_WEEKLY_K_MIN, EARLY_WEEKLY_K_MAX],
        "strength": V63_STRENGTH_THRESHOLD,
        "confirm_deadline_days": 14,
        "initial_position_ratio": 1.0,
        "add_position": False,
        "initial_capital": V66_INITIAL_CAPITAL,
        "max_stocks": V66_MAX_STOCKS,
        "full_slot_capital": V66_FULL_SLOT_CAPITAL,
        "portfolio_schemes": [item["key"] for item in V66_PORTFOLIO_SCHEMES],
        "early_failure_days": list(V612_EARLY_FAILURE_DAYS),
        "early_price_failure_pct": V612_PRICE_FAILURE_PCT,
        "early_macd_remaining_pct": V612_MACD_REMAINING_PCT,
        "entry_rank_cap": 3,
        "primary_rank_rule": "Score221Desc_DailyMACDHistAsc_ExactTieSeed",
        "ordinary_rule": "keep_exact_score5_to_fixed40",
        "late_reentry": False,
        "portfolio_draws": int(portfolio_draws),
        "common_random_seed": V66_RANDOM_SEED,
    }
    request_signature = stable_signature(request_payload)
    result_path = os.path.join(V63_RESULT_DIR, f"{request_signature}.zip")
    result_name = (
        f"weekly_skdj_full_slot_early_f_exit_v6_12_"
        f"{int(backtest_days)}d_p{int(min_price)}_mv{int(min_mv)}.zip")
    completed_available = False
    if os.path.exists(result_path):
        try:
            with open(result_path, "rb") as handle:
                saved_result = handle.read()
            completed_available = True
            v63_clear_job_active(request_signature)
            st.success("发现相同参数的已完成结果，可直接下载。")
            render_download(
                saved_result, result_name, f"v612_saved_{request_signature}")
        except Exception as exc:
            st.warning(f"已保存结果读取失败：{exc}")

    secret_token = configured_tushare_token()
    if secret_token:
        token = secret_token
        st.caption("已从Streamlit Secrets读取TUSHARE_TOKEN。")
    else:
        token = st.text_input("Tushare Token", type="password", key="v612_token")
    job_active = v63_is_job_active(request_signature)
    left, right = st.columns(2)
    with left:
        start_clicked = st.button(
            "开始/重新运行V6.12", type="primary", key="v612_run")
    with right:
        stop_clicked = st.button(
            "停止自动续跑", disabled=not job_active, key="v612_stop")
    if stop_clicked:
        v63_clear_job_active(request_signature)
        st.success("已停止；逐股票检查点保留。")
        return
    if start_clicked:
        if market_end_date <= signal_end_date:
            st.error("最新观察截止必须晚于历史信号截止")
            return
        v63_mark_job_active(request_signature)
        job_active = True
    if not token:
        st.info("请输入Token；任务启动后若页面重连，会从逐股票检查点自动续跑。")
        return
    if not job_active:
        st.caption(
            "点击开始运行。" if not completed_available
            else "相同参数结果已可下载；如需覆盖请点击重新运行。")
        return

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    signal_end = signal_end_date.strftime("%Y%m%d")
    market_end = market_end_date.strftime("%Y%m%d")
    try:
        probe_start = signal_end_date - timedelta(
            days=int(backtest_days) * 2 + 120)
        probe_dates = load_trade_calendar(
            probe_start.strftime("%Y%m%d"), signal_end)
        signal_start = trailing_signal_start(
            probe_dates, signal_end, int(backtest_days))
    except Exception as exc:
        st.error(f"确定{int(backtest_days)}个交易日窗口失败：{exc}")
        return
    data_start = (
        pd.Timestamp(signal_start).date()
        - timedelta(weeks=WARMUP_WEEKS, days=7)
    ).strftime("%Y%m%d")
    try:
        with st.spinner("加载交易日历和历史科技池..."):
            open_dates = load_trade_calendar(data_start, market_end)
            extended_end = (
                market_end_date + timedelta(days=7)).strftime("%Y%m%d")
            week_last_map = complete_week_last_dates(
                load_trade_calendar(data_start, extended_end))
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return
    if not open_dates:
        st.error("区间内没有市场交易日")
        return
    eligible_market_dates = [
        value for value in open_dates if value <= market_end]
    if not eligible_market_dates:
        st.error("行情截止日前没有可用市场交易日")
        return
    latest_market_date = max(eligible_market_dates)
    open_pos = {value: position for position, value in enumerate(open_dates)}
    config = {
        "signal_start": signal_start, "signal_end": signal_end,
        "event_signal_end": market_end, "data_start": data_start,
        "market_end": market_end, "latest_market_date": latest_market_date,
        "min_price": float(min_price), "min_mv": float(min_mv),
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct),
        "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct),
    }

    # Event generation is unchanged, so the compatible V6.4 stock checkpoints
    # remain valid. V6.12 changes only the portfolio lifecycle and audits.
    run_signature = stable_signature({
        "version": V65_EVENT_ENGINE_VERSION, **config})
    period_index = build_period_index(memberships)
    active_codes = {
        code for code, code_periods in period_index.items()
        if periods_overlap(code_periods, signal_start, market_end)}
    stocks = stock_basic[stock_basic["ts_code"].isin(active_codes)].copy()
    stocks = stocks[
        ~stocks["list_date"].gt(market_end)
        & ~stocks["delist_date"].lt(data_start)
    ].sort_values("ts_code").reset_index(drop=True)

    event_rows: list[dict[str, Any]] = []
    rejects: dict[str, int] = {}
    checkpoint_hits = price_cache_hits = failures = 0
    progress, status = st.progress(0.0), st.empty()
    last_update = 0.0
    stopped = False
    for number, stock in stocks.iterrows():
        if not v63_is_job_active(request_signature):
            stopped = True
            break
        code = str(stock["ts_code"])
        checkpoint = v63_load_checkpoint(run_signature, code)
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
                    rows, stock_rejects, _ = analyze_stock_v63(
                        stock, period_index.get(code, []), daily,
                        cached_basic, storage_path, week_last_map, open_dates,
                        open_pos, config, bool(use_cache), float(pause))
                    event_rows.extend(rows)
                    merge_counts(rejects, stock_rejects)
                    v63_save_checkpoint(
                        run_signature, code, rows, stock_rejects)
                except Exception as exc:
                    failures += 1
                    record_error(f"V6.12逐股票分析失败 {code}: {exc}")
        processed = number + 1
        now = time.monotonic()
        if (processed == 1 or now - last_update >= UI_HEARTBEAT_SECONDS
                or processed == len(stocks)):
            progress.progress(
                processed / max(len(stocks), 1),
                text=f"已处理{processed}/{len(stocks)}只股票，最近{code}")
            status.caption(
                f"事件{len(event_rows)}；检查点{checkpoint_hits}；"
                f"行情缓存{price_cache_hits}；失败{failures}")
            last_update = now
    progress.empty()
    status.empty()
    if stopped:
        st.warning("任务已停止，逐股票检查点已保留。")
        return

    events_all = pd.DataFrame(event_rows)
    if events_all.empty:
        st.error("本区间没有生成V6.12可用事件。")
        return
    events_all = v63_attach_breadth_and_rank(events_all, open_dates)
    events_all = events_all.sort_values(
        ["Signal_Date", "Event_Type", "ts_code"]).reset_index(drop=True)
    live_watch = events_all[events_all["Event_Type"].eq("LIVE_WATCH")].copy()
    research = events_all[~events_all["Event_Type"].eq("LIVE_WATCH")].copy()
    history = research[
        research["Signal_Date"].astype(str).le(signal_end)].copy()
    observation = research[
        research["Signal_Date"].astype(str).gt(signal_end)
        & research["Signal_Date"].astype(str).le(latest_market_date)
    ].copy()
    mature = history[
        true_mask(history, "Entry_Tradable")
        & true_mask(history, "Entry_Has_40D")].copy()
    if mature.empty:
        st.error("存在信号，但没有可成交且走完40个市场交易日的成熟事件。")
        return
    mature = add_timing_labels(mature)
    early = mature[mature["Event_Type"].eq("STRENGTH_3")].copy()
    weekly = mature[mature["Event_Type"].eq("WEEKLY_CROSS25")].copy()
    if early.empty or weekly.empty:
        st.error("3%提前买点或周线确认对照为空。")
        return

    weekly["Weekly_Momentum_Tier"] = "普通周线确认"
    weekly.loc[
        weekly["Daily_MACD_State"].astype(str).eq("红柱扩张")
        & numeric(weekly, "Daily_Return_Since_Red_Start_pct").between(10, 30),
        "Weekly_Momentum_Tier"] = "高优先级_红柱扩张且已涨10至30"
    weekly.loc[
        numeric(weekly, "Daily_Return_Since_Red_Start_pct").gt(30),
        "Weekly_Momentum_Tier"] = "风险组_已涨超过30"
    early = v64_add_state_lifecycle_columns(early)
    early = v65_add_staged_position_columns(early)

    with st.spinner("复用本地行情缓存，构建全额三仓撮合与D3/D5/D7检查价格簿..."):
        price_book, portfolio_cache_hits, portfolio_missing = (
            v66_load_price_book(early, data_start, market_end))
        recovered_price_files = 0
        still_missing: list[str] = []
        for code in portfolio_missing:
            daily, _, _, _ = fetch_price(
                code, data_start, market_end, True, float(pause))
            if daily.empty:
                still_missing.append(code)
                continue
            daily = add_daily_macd(normalize_price_frame(daily))
            dates = daily["trade_date"].astype(str).tolist()
            price_book[code] = {
                "dates": dates,
                "open": dict(zip(dates, numeric(daily, "open"))),
                "close": dict(zip(dates, numeric(daily, "close"))),
                "high": dict(zip(dates, numeric(daily, "high"))),
                "low": dict(zip(dates, numeric(daily, "low"))),
                "macd_hist": dict(zip(
                    dates, numeric(daily, "Daily_MACD_Hist"))),
                "macd_remaining": dict(zip(
                    dates, numeric(daily, "Daily_MACD_Remaining_pct"))),
                "macd_retention": dict(zip(
                    dates, numeric(daily, "Daily_MACD_Retention_pct"))),
            }
            recovered_price_files += 1
        portfolio_missing = still_missing
    if portfolio_missing:
        st.error(
            f"三仓组合回测缺少{len(portfolio_missing)}只股票的完整行情，"
            "为避免样本偏差本次不继续。请保留检查点后重新运行。")
        st.dataframe(pd.DataFrame({"缺失股票": portfolio_missing}),
                     use_container_width=True, hide_index=True)
        return

    portfolio_event_audit = v66_prepare_portfolio_events(early, open_dates)
    early_failure_event_summary, early_failure_event_detail = (
        v612_early_failure_event_audit(
            portfolio_event_audit, price_book, open_dates, config))
    with st.spinner(
            f"运行真实全额三仓：4个生命周期方案×2种选择×"
            f"{int(portfolio_draws)}条路径..."):
        (portfolio_paths, portfolio_summary, score_vs_random,
         representative_equity, representative_trades,
         representative_annual, all_path_year_detail,
         all_path_year_summary) = v66_run_portfolio_ensemble(
            early, price_book, open_dates, config, int(portfolio_draws))
    if portfolio_paths.empty:
        st.error("成熟事件存在，但全额三仓组合未能生成可执行路径。")
        return

    early["Calendar_Year"] = early["Signal_Date"].astype(str).str[:4]
    paired_early_failure = v612_paired_early_failure_summary(portfolio_paths)
    seed_pairing_audit = v611_seed_pairing_audit(portfolio_paths)
    score_rank_summary = v68_score_rank_path_summary(portfolio_paths)
    profit_concentration = v68_profit_concentration_summary(portfolio_paths)
    representative_stock_profit = v68_representative_stock_profit(
        representative_trades)
    actual_selected_grade_audit = v611_actual_selected_grade_audit(
        representative_trades, portfolio_event_audit)
    actual_f_slot_audit = v612_actual_f_slot_audit(
        representative_trades, portfolio_event_audit, open_dates)
    ordinary_factor_combo = v68_exact_factor_combo_audit(early)
    ordinary_score_audit = v67_action_score_audit(
        early, "14日内普通确认_只保留试仓", "普通确认")
    high_score_audit = v67_action_score_audit(
        early, "14日内高质量确认_试仓升级", "高质量确认")
    cross_state_outcomes = timing_outcome_summary(
        early, ["Cross_Delay_Group", "Future_Cross_State"],
        "提前买入后按确认速度和确认日状态分组")
    cross_entry_outcomes = v64_cross_entry_summary(early)
    calendar = v63_week_calendar(
        open_dates, signal_start, signal_end, mature)
    coverage = v63_coverage_summary(calendar)

    recent_start = (
        pd.Timestamp(latest_market_date) - pd.Timedelta(days=14)
    ).strftime("%Y%m%d")
    recent_early = observation[
        observation["Event_Type"].eq("STRENGTH_3")
        & observation["Signal_Date"].astype(str).between(
            recent_start, latest_market_date)].copy()
    recent_weekly = observation[
        observation["Event_Type"].eq("WEEKLY_CROSS25")
        & observation["Signal_Date"].astype(str).between(
            recent_start, latest_market_date)].copy()
    if not recent_early.empty:
        recent_early["Current_Lifecycle_Status"] = v64_live_lifecycle_status(
            recent_early, latest_market_date)
    high_priority_weekly = weekly[
        weekly["Weekly_Momentum_Tier"].eq(
            "高优先级_红柱扩张且已涨10至30")].copy()

    run_summary = pd.DataFrame([{
        "程序版本": V63_VERSION,
        "正式信号开始": signal_start,
        "历史信号截止": signal_end,
        "行情观察截止": market_end,
        "最新市场交易日": latest_market_date,
        "成熟3%提前事件": len(early),
        "成熟周线确认事件": len(weekly),
        "高优先级周线事件": len(high_priority_weekly),
        "提前事件后来高质量确认": int(
            early["Future_Cross_State"].eq(V64_CROSS_HIGH).sum()),
        "14日内高质量确认继续持有资格": int(
            early["V65_Lifecycle_Action"].eq(
                "14日内高质量确认_试仓升级").sum()),
        "14日内普通确认5分继续持有": int((
            early["V65_Lifecycle_Action"].eq(
                "14日内普通确认_只保留试仓")
            & numeric(early, "Timing_Score_221").eq(5.0)).sum()),
        "14日超时": int((~true_mask(early, "Confirm14_Confirmed")).sum()),
        "首次买入仓位比例": "100%",
        "高质量确认是否加仓": False,
        "迟到确认是否重入": False,
        "同日入场顺位": "固定原始Top3、不递补",
        "主排序": "2-2-1降序；完全同分MACD柱较小优先",
        "早期硬失败检查日": "D3/D5/D7",
        "早期价格失败阈值%": V612_PRICE_FAILURE_PCT,
        "早期MACD剩余阈值%": V612_MACD_REMAINING_PCT,
        "共用随机种子起点": V66_RANDOM_SEED,
        "公平配对是否全部通过": bool(
            not seed_pairing_audit.empty
            and true_mask(seed_pairing_audit, "公平配对通过").all()),
        "3%提前信号周": early["Signal_Week"].nunique(),
        "最近14日3%提前信号": len(recent_early),
        "最近14日周线确认信号": len(recent_weekly),
        "最新观察池股票": len(live_watch),
        "组合初始资金": V66_INITIAL_CAPITAL,
        "最大持股数": V66_MAX_STOCKS,
        "单股目标投入": V66_FULL_SLOT_CAPITAL,
        "每组随机路径": int(portfolio_draws),
        "组合方案数": len(V66_PORTFOLIO_SCHEMES),
        "组合路径总数": len(portfolio_paths),
        "组合行情缓存命中": portfolio_cache_hits,
        "组合行情补下载": recovered_price_files,
        "处理股票数": len(stocks),
        "检查点恢复": checkpoint_hits,
        "行情缓存命中": price_cache_hits,
        "失败股票": failures,
    }])
    definitions = pd.DataFrame([
        ("提前买入", "周线K15至25；日线MACD质量合格；本轮首次上涨3%；次日开盘"),
        ("账户规模", "初始30万元；最多3只；每个名额约10万元"),
        ("首次仓位", "每只一次性使用完整名额；整手买入；剩余零钱留在现金"),
        ("候选不足", "只有1至2只合格候选时只买1至2只，其余名额保持现金"),
        ("加仓", "禁止；高质量确认只表示原全额仓位继续持有"),
        ("确认期限", "买入后14自然日内观察周线K是否上穿25"),
        ("普通确认", "只有买入时2-2-1恰好5分继续持有；其余下一开盘退出"),
        ("高质量确认", "红柱扩张且本轮累计上涨10至30%；继续持有，不加仓"),
        ("已涨超过30", "保护已有全额仓位继续持有；不追买、不加仓"),
        ("未确认退出", "14日仍未确认，下一可交易开盘退出"),
        ("迟到确认", "退出后不重入"),
        ("固定上限", "所有被保留股票持有至初始买入后的第40市场日收盘"),
        ("基线", "不使用早期硬失败退出，其余规则完全相同"),
        ("D3/D5/D7", "分别在买入后的第3/5/7市场日收盘检查，下一可交易开盘退出"),
        ("MACD硬失败", f"柱≤0，或红柱剩余强度≤{V612_MACD_REMAINING_PCT:g}%且保留率<100%"),
        ("价格硬失败", f"检查日收盘相对买入开盘≤{V612_PRICE_FAILURE_PCT:g}%"),
        ("早退保护", "检查日前已观察到高质量、超涨确认，或普通确认且原始评分恰好5分"),
        ("可见性", "早退只读取检查日收盘及此前确认；40日等级不参与交易"),
        ("主排序", "2-2-1分降序；完全同分时日线MACD柱较小优先"),
        ("同日Top3", "固定原始顺位前3；高顺位未成交时低顺位不晋级"),
        ("日内顺序", "开盘退出→当日新候选全额买入；同日释放的名额可以再用"),
        ("资本核算", "逐日现金、整手股数、持仓市值和交易成本真实复合"),
        ("公平配对", "相同Path_No跨生命周期方案共用Seed；退出改变后续名额属于真实路径差异"),
        ("F级目标", "同时审计F占比、F平均持有日、F名额占用日、F净利润与早退召回率"),
        ("S/A保护", "同时审计早退中S/A比例与全部已买入S/A的误杀率"),
        ("三股目标", "按实际成交事件验收盈利比例及每三个实际买入中至少两个盈利"),
        ("牛股依赖", "静态扣除最大和前三盈利股票利润，不虚构释放名额后的反事实交易"),
        ("缓存", "基础缓存72小时；行情和兼容逐股票检查点不主动过期"),
    ], columns=["项目", "定义"])
    cache_policy = pd.DataFrame([
        ("交易日历", "Streamlit内存", "72小时", "实例重启可能提前消失"),
        ("股票基础资料", "Streamlit内存", "72小时", "实例重启可能提前消失"),
        ("申万科技行业成员", "Streamlit内存", "7天", "实例重启可能提前消失"),
        ("逐股票日线与daily_basic", "应用临时磁盘", "不自动过期", "重新部署可能清空"),
        ("兼容逐股票事件检查点", "应用临时磁盘", "不自动过期", "V6.12复用相同事件"),
    ], columns=["对象", "位置", "设定时长", "实际边界"])
    rejection_audit = pd.DataFrame([
        {"剔除原因": key, "次数": value}
        for key, value in sorted(rejects.items())])
    metadata = pd.DataFrame([{
        "程序版本": V63_VERSION,
        "SKDJ_N": EARLY_PRIMARY_N, "SKDJ_M": SKDJ_M,
        "回测交易日": int(backtest_days), "预热周": WARMUP_WEEKS,
        "历史信号开始": signal_start, "历史信号截止": signal_end,
        "行情截止": market_end,
        "最低股价": float(min_price),
        "最低流通市值亿元": float(min_mv),
        "提前涨幅阈值%": V63_STRENGTH_THRESHOLD,
        "确认期限自然日": 14,
        "首次买入比例": "1.0000",
        "买入后加仓": "禁止",
        "高质量确认": "继续持有原全额仓位",
        "普通确认": "原始2-2-1恰好5分固定40日",
        "迟到确认重入": "禁止",
        "同日入场顺位上限": 3,
        "顺位递补": "禁止",
        "主排序": "Score221Desc/DailyMACDHistAsc",
        "早期失败检查市场日": "3/5/7",
        "早期价格失败阈值%": V612_PRICE_FAILURE_PCT,
        "早期MACD剩余阈值%": V612_MACD_REMAINING_PCT,
        "组合初始资金": V66_INITIAL_CAPITAL,
        "组合股票名额": V66_MAX_STOCKS,
        "单股目标投入": V66_FULL_SLOT_CAPITAL,
        "整手股数": V66_BOARD_LOT,
        "组合方案数": len(V66_PORTFOLIO_SCHEMES),
        "每组选择方式路径数": int(portfolio_draws),
        "共用随机种子起点": V66_RANDOM_SEED,
        "内存缓存小时": CACHE_TTL_SECONDS / 3600,
        "Streamlit": streamlit_version,
    }])

    detail_columns = [
        "Event_Type", "Strategy_Label", "ts_code", "name", "Signal_Date",
        "Signal_Week", "SW_L1", "SW_L2", "Raw_Close", "Circ_MV_Billion",
        "Setup_Weekly_Date", "Signal_K", "Signal_D", "Signal_KD_Spread",
        "Daily_MACD_State", "Daily_MACD_Red_Age", "Daily_MACD_Hist",
        "Daily_MACD_Remaining_pct", "Daily_MACD_Retention_pct",
        "Signal_Rally_From_Red_Start_pct",
        "Signal_Day_Strength3_Count", "Signal_Week_Cumulative_Strength3_Count",
        "Trailing5D_Strength3_Count", "Future_Full_Week_Strength3_Count",
        "Signal_Day_Breadth_Group", "Week_Cumulative_Breadth_Group",
        "Trailing5D_Breadth_Group", "Future_Full_Week_Breadth_Group",
        "Timing_K15_20_Pass", "Timing_RedAge3_5_Pass",
        "Timing_WeeklyK_BelowD_Pass", "Timing_Score_221",
        "Timing_Three_Factor_Count", "Entry_Date", "Entry_Raw_Open",
        "Entry_End_Date_40D", "Entry_MFE_Net_pct", "Entry_MAE_Raw_pct",
        "Entry_Close_Return_Net_pct", "Explosion_Class_40D",
        "Future_Weekly_Cross25_Date", "Future_Cross_State",
        "Future_Cross_Daily_MACD_State",
        "Future_Cross_Rally_From_Red_Start_pct", "Cross_Delay_Group",
        "V65_Lifecycle_Action", "Confirm14_Confirmed", "Confirm14_Exit_Date",
        "CrossImmediate_Decision_Date", "CrossImmediate_Exit_Date",
        "CrossEntry_Date", "CrossEntry_Tradable", "CrossEntry_Has_40D",
    ]
    live_columns = [
        "Event_Type", "Strategy_Label", "ts_code", "name", "Signal_Date",
        "Signal_Week", "SW_L1", "SW_L2", "Raw_Close", "Circ_MV_Billion",
        "Setup_Weekly_Date", "Signal_K", "Signal_D", "Signal_KD_Spread",
        "Daily_MACD_State", "Daily_MACD_Red_Age", "Daily_MACD_Hist",
        "Daily_MACD_Remaining_pct", "Daily_MACD_Retention_pct",
        "Signal_Rally_From_Red_Start_pct", "Timing_Score_221",
        "Timing_Three_Factor_Count", "Future_Cross_State",
        "Future_Weekly_Cross25_Date", "Current_Lifecycle_Status",
        "Watch_Next_Strength_Threshold",
    ]
    history_weekly_export = weekly[[
        column for column in detail_columns + ["Weekly_Momentum_Tier"]
        if column in weekly.columns]].copy()
    recent_early_export = recent_early[[
        column for column in live_columns if column in recent_early.columns]].copy()
    recent_weekly_export = recent_weekly[[
        column for column in live_columns if column in recent_weekly.columns]].copy()
    live_watch_export = live_watch[[
        column for column in live_columns if column in live_watch.columns]].copy()

    files = {
        "01_run_summary_v6_12.csv": run_summary,
        "02_experiment_definitions_v6_12.csv": definitions,
        "03_early_failure_ensemble_summary_v6_12.csv": portfolio_summary,
        "04_paired_early_failure_comparison_v6_12.csv": paired_early_failure,
        "05_rule_vs_random_portfolio_v6_12.csv": score_vs_random,
        "06_all_portfolio_paths_v6_12.csv": portfolio_paths,
        "07_all_path_year_summary_v6_12.csv": all_path_year_summary,
        "08_all_path_year_detail_v6_12.csv": all_path_year_detail,
        "09_representative_path_annual_v6_12.csv": representative_annual,
        "10_representative_path_equity_v6_12.csv": representative_equity,
        "11_representative_path_trades_v6_12.csv": representative_trades,
        "12_actual_f_slot_and_sa_false_exit_audit_v6_12.csv": actual_f_slot_audit,
        "13_actual_selected_grade_and_win_audit_v6_12.csv": actual_selected_grade_audit,
        "14_early_failure_event_summary_v6_12.csv": early_failure_event_summary,
        "15_early_failure_event_detail_v6_12.csv": early_failure_event_detail,
        "16_rule_rank_1_2_3_path_summary_v6_12.csv": score_rank_summary,
        "17_profit_concentration_ablation_v6_12.csv": profit_concentration,
        "18_representative_stock_profit_contribution_v6_12.csv": representative_stock_profit,
        "19_common_seed_pairing_audit_v6_12.csv": seed_pairing_audit,
        "20_portfolio_event_audit_v6_12.csv": portfolio_event_audit,
        "21_ordinary_exact_factor_combination_v6_12.csv": ordinary_factor_combo,
        "22_ordinary_confirmation_by_exact_score_v6_12.csv": ordinary_score_audit,
        "23_high_confirmation_by_exact_score_v6_12.csv": high_score_audit,
        "24_early_outcomes_by_cross_state_v6_12.csv": cross_state_outcomes,
        "25_fresh_cross_entry_by_state_v6_12.csv": cross_entry_outcomes,
        "26_weekly_coverage_summary_v6_12.csv": coverage,
        "27_weekly_signal_calendar_v6_12.csv": calendar,
        "28_historical_weekly_control_detail_v6_12.csv": history_weekly_export,
        "29_recent_14d_strength3_candidates_v6_12.csv": recent_early_export,
        "30_recent_14d_weekly_candidates_v6_12.csv": recent_weekly_export,
        "31_latest_market_day_watch_pool_v6_12.csv": live_watch_export,
        "32_cache_policy_v6_12.csv": cache_policy,
        "33_rejection_audit_v6_12.csv": rejection_audit,
        "34_metadata_v6_12.csv": metadata,
        "35_api_errors_v6_12.csv": pd.DataFrame({"错误": API_ERRORS}),
    }
    result_zip = make_zip(files)
    try:
        atomic_bytes(result_zip, result_path)
        v63_clear_job_active(request_signature)
        persisted = True
    except Exception as exc:
        persisted = False
        st.warning(f"结果未能持久保存，但当前页面仍可下载：{exc}")

    st.success(
        f"完成：成熟3%提前事件{len(early)}个，周线确认对照{len(weekly)}个；"
        f"已运行{len(portfolio_paths)}条30万元全额三仓路径；"
        f"14日内高质量确认继续持有资格"
        f"{int(early['V65_Lifecycle_Action'].eq('14日内高质量确认_试仓升级').sum())}个，"
        f"普通确认5分继续持有"
        f"{int((early['V65_Lifecycle_Action'].eq('14日内普通确认_只保留试仓') & numeric(early, 'Timing_Score_221').eq(5.0)).sum())}个；"
        f"最近14日3%信号{len(recent_early)}个，最新观察池{len(live_watch)}只；"
        f"结果{'已保存' if persisted else '仅当前页面可下载'}。")

    st.subheader("结论一：全额三仓基线与D3/D5/D7早期硬失败退出")
    render_plain_table(portfolio_summary, 40)
    st.caption("这是逐日现金、整手股数、持仓市值和成本复合后的账户结果，不是事件收益平均。")
    st.subheader("结论二：D3/D5/D7相对无早退基线的逐路径配对")
    render_plain_table(paired_early_failure, 30)
    st.caption("重点看收益差分布、最大回撤、资金暴露、高质量确认捕获率和早退数量，而不是只看一条中位路径。")
    st.subheader("结论三：实际F级占用名额与S/A误杀")
    render_plain_table(actual_f_slot_audit, 30)
    st.caption("按各方案中位代表路径的实际成交事件统计；F级平均持有日和名额占用日直接对应资金周转效率。")
    st.subheader("结论四：D3/D5/D7硬失败条件本身的识别质量")
    render_plain_table(early_failure_event_summary, 20)
    st.caption("这是全部成熟事件的独立诊断：触发中F比例衡量精度，清除全部F比例衡量召回，误杀全部S/A比例衡量代价。")
    st.subheader("结论五：实际买入是否维持S/A/B追求和三股盈利目标")
    render_plain_table(actual_selected_grade_audit, 30)
    st.caption("正式验收口径是实际成交事件，而不是候选池平均；同时查看F比例、A/S比例、事件胜率和每三只至少两只盈利。")
    st.subheader("结论六：固定排序是否优于完全随机")
    render_plain_table(score_vs_random, 30)
    st.caption("若规则优于随机比例长期只在50%左右，说明排序并未提供稳定优势。")
    st.subheader("结论七：实际入场第1/2/3顺位")
    render_plain_table(score_rank_summary, 40)
    st.subheader("结论八：剔除头部盈利股票后的静态稳健性")
    render_plain_table(profit_concentration, 40)
    st.subheader("结论九：全部路径的逐年稳定性")
    render_plain_table(all_path_year_summary, 100)
    with st.expander("中位代表路径逐年表与共用Seed审计", expanded=False):
        render_plain_table(representative_annual, 80)
        render_plain_table(seed_pairing_audit, 20)
    with st.expander("三因子精确组合与确认评分审计", expanded=False):
        render_plain_table(ordinary_factor_combo, 100)
        render_plain_table(ordinary_score_audit, 30)
        render_plain_table(high_score_audit, 30)
    with st.expander("D3/D5/D7逐事件触发明细", expanded=False):
        render_plain_table(early_failure_event_detail, 500)
    st.subheader("确认速度与确认日状态")
    render_plain_table(cross_state_outcomes, 30)
    render_plain_table(cross_entry_outcomes, 30)
    st.subheader("最近14日3%提前候选")
    recent_sort = [
        column for column in ["Signal_Date", "Current_Lifecycle_Status",
                              "Signal_Rally_From_Red_Start_pct"]
        if column in recent_early_export.columns]
    if recent_sort:
        recent_early_export = recent_early_export.sort_values(
            recent_sort,
            ascending=[False, True, False][:len(recent_sort)])
    render_plain_table(recent_early_export, 300)
    st.subheader(f"最新交易日周线K15至25观察池：{latest_market_date}")
    watch_sort = [
        column for column in ["Signal_K", "Signal_KD_Spread"]
        if column in live_watch_export.columns]
    if watch_sort:
        live_watch_export = live_watch_export.sort_values(watch_sort)
    render_plain_table(live_watch_export, 300)
    st.subheader("运行摘要与覆盖率")
    render_plain_table(run_summary, 10)
    render_plain_table(coverage, 10)
    st.caption(
        f"结果ZIP共{len(files)}个CSV；历史成熟事件、组合路径、"
        "最近候选和最新观察池严格分开。")
    render_download(result_zip, result_name, f"v612_current_{request_signature}")


# ---------------------------------------------------------------------------
# V6.13 SKDJ calibration and MACD-red-day-2 entry audit.
# This block deliberately overrides the V6.12 ``main`` above while reusing its
# data, cache, universe, cost and 40-session outcome infrastructure.
# ---------------------------------------------------------------------------

V613_TITLE = "周线SKDJ参数对齐与日线MACD红柱第2日买点回测 V6.13"
V613_VERSION = "V6.13-SKDJ-CALIBRATION-N9-N6-RED2-ENTRY"
V613_UI_PATCH = "V6.13-CALIBRATE-N2-60-M2-10-N9PRIMARY-N6CONTROL"
V613_RESULT_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_13_results")
V613_JOB_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_13_jobs")
V613_CHECKPOINT_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_13_checkpoints")
V613_EVENT_ENGINE_VERSION = "V6.13-DYNAMIC-WEEKLY-RED2-EVENTS-R1"
V613_PRIMARY_N = 9
V613_CONTROL_N = 6
V613_M = 3
V613_APPROACH_K_MIN = 10.0
V613_APPROACH_K_MAX = 25.0
V613_PARAMETER_N_RANGE = tuple(range(2, 61))
V613_PARAMETER_M_RANGE = tuple(range(2, 11))
V613_CALIBRATION_TOLERANCE = 0.10
V613_RANDOM_SEED = 20260828
V613_PORTFOLIO_EXITS = (
    ("Fixed40", "固定40日"),
    ("EarlyFailureD3", "第3市场日硬失败否则40日"),
)
V613_STRATEGIES = (
    {
        "key": "OLD_N6_STRENGTH3",
        "label": "旧基准_N6_完整周K15至25_上涨3%",
        "n": 6,
        "weekly_mode": "最近完整周",
        "entry_mode": "首次上涨3%",
    },
    {
        "key": "N6_DYNAMIC_RED2",
        "label": "对照_N6_动态周线接近25_红柱第2日",
        "n": 6,
        "weekly_mode": "截至当日动态周线",
        "entry_mode": "MACD红柱第2日",
    },
    {
        "key": "N9_DYNAMIC_RED2",
        "label": "主假设_N9_动态周线接近25_红柱第2日",
        "n": 9,
        "weekly_mode": "截至当日动态周线",
        "entry_mode": "MACD红柱第2日",
    },
)
V613_CALIBRATION_REFERENCES = (
    {
        "ts_code": "688361.SH", "name": "中科飞测", "period": "W",
        "date": "20250411", "target_k": 25.10, "target_d": 25.78,
    },
    {
        "ts_code": "603290.SH", "name": "斯达半导", "period": "D",
        "date": "20260721", "target_k": 7.43, "target_d": 5.32,
    },
    {
        "ts_code": "688783.SH", "name": "西安奕材", "period": "D",
        "date": "20260624", "target_k": 60.80, "target_d": 70.26,
    },
    {
        "ts_code": "688249.SH", "name": "晶合集成", "period": "D",
        "date": "20260806", "target_k": 26.05, "target_d": 17.75,
    },
)


def v613_skdj_values(frame: pd.DataFrame, n: int, m: int) -> pd.DataFrame:
    """Implement the supplied Tonghuashun EMA-EMA-MA formula exactly."""
    work = normalize_price_frame(frame).sort_values(
        "trade_date").reset_index(drop=True)
    lowv = numeric(work, "low").rolling(int(n), min_periods=int(n)).min()
    highv = numeric(work, "high").rolling(int(n), min_periods=int(n)).max()
    raw = (
        (numeric(work, "close") - lowv)
        / (highv - lowv).replace(0, np.nan) * 100.0
    )
    rsv = raw.ewm(span=int(m), adjust=False, min_periods=1).mean()
    k_value = rsv.ewm(span=int(m), adjust=False, min_periods=1).mean()
    d_value = k_value.rolling(int(m), min_periods=int(m)).mean()
    work["SKDJ_Raw"] = raw
    work["SKDJ_RSV_EMA"] = rsv
    work["SKDJ_K"] = k_value
    work["SKDJ_D"] = d_value
    work["SKDJ_KD_Spread"] = k_value - d_value
    return work


def v613_dynamic_weekly_skdj(
        daily: pd.DataFrame, weekly_base: pd.DataFrame, n: int,
        m: int = V613_M) -> pd.DataFrame:
    """Calculate an observable partial-week SKDJ at every daily close.

    Previous weeks are complete bars.  The current week uses only highs, lows
    and the close observed through that day.  Friday therefore equals the
    completed weekly value, while Monday through Thursday never use future
    prices from the rest of the week.
    """
    daily_work = normalize_price_frame(daily).sort_values(
        "trade_date").reset_index(drop=True)
    if daily_work.empty or weekly_base.empty:
        return pd.DataFrame(columns=[
            "trade_date", "Dynamic_Weekly_K", "Dynamic_Weekly_D",
            "Dynamic_Weekly_KD_Spread", "Dynamic_Weekly_K_Change",
            "Dynamic_Prior_Weekly_Date"])
    full = v613_skdj_values(weekly_base, int(n), int(m))
    full["_period"] = pd.to_datetime(
        full["trade_date"], format="%Y%m%d").dt.to_period("W-FRI")
    daily_work["_period"] = pd.to_datetime(
        daily_work["trade_date"], format="%Y%m%d").dt.to_period("W-FRI")
    alpha = 2.0 / (float(m) + 1.0)
    rows: list[dict[str, Any]] = []
    for period, group in daily_work.groupby("_period", sort=True):
        prior = full[full["_period"].lt(period)].sort_values("_period")
        prior_valid = prior[
            numeric(prior, "SKDJ_RSV_EMA").notna()
            & numeric(prior, "SKDJ_K").notna()
        ]
        prior_range = prior.tail(max(int(n) - 1, 0))
        enough_range = len(prior_range) >= max(int(n) - 1, 0)
        enough_smooth = len(prior_valid) >= max(int(m) - 1, 1)
        previous_rsv = (
            finite_num(prior_valid.iloc[-1].get("SKDJ_RSV_EMA"))
            if enough_smooth else np.nan)
        previous_k = (
            finite_num(prior_valid.iloc[-1].get("SKDJ_K"))
            if enough_smooth else np.nan)
        previous_k_values = (
            numeric(prior_valid.tail(max(int(m) - 1, 0)), "SKDJ_K").tolist()
            if enough_smooth else [])
        historical_low = (
            finite_num(numeric(prior_range, "low").min())
            if enough_range and len(prior_range) else np.nan)
        historical_high = (
            finite_num(numeric(prior_range, "high").max())
            if enough_range and len(prior_range) else np.nan)
        prior_date = (
            str(prior.iloc[-1]["trade_date"]) if not prior.empty else "")
        partial_low = numeric(group, "low").cummin()
        partial_high = numeric(group, "high").cummax()
        for position, (_, row) in enumerate(group.iterrows()):
            if int(n) == 1:
                lowv = finite_num(partial_low.iloc[position])
                highv = finite_num(partial_high.iloc[position])
            elif enough_range:
                lowv = min(
                    historical_low, finite_num(partial_low.iloc[position]))
                highv = max(
                    historical_high, finite_num(partial_high.iloc[position]))
            else:
                lowv = highv = np.nan
            close = finite_num(row.get("close"))
            raw = (
                (close - lowv) / (highv - lowv) * 100.0
                if math.isfinite(lowv) and math.isfinite(highv)
                and highv > lowv and math.isfinite(close) else np.nan)
            rsv = (
                alpha * raw + (1.0 - alpha) * previous_rsv
                if math.isfinite(raw) and math.isfinite(previous_rsv)
                else np.nan)
            k_value = (
                alpha * rsv + (1.0 - alpha) * previous_k
                if math.isfinite(rsv) and math.isfinite(previous_k)
                else np.nan)
            d_values = previous_k_values + [k_value]
            d_value = (
                float(np.mean(d_values[-int(m):]))
                if len(d_values) >= int(m)
                and all(math.isfinite(value) for value in d_values[-int(m):])
                else np.nan)
            rows.append({
                "trade_date": str(row["trade_date"]),
                "Dynamic_Weekly_K": k_value,
                "Dynamic_Weekly_D": d_value,
                "Dynamic_Weekly_KD_Spread": k_value - d_value,
                "Dynamic_Weekly_K_Change": k_value - previous_k,
                "Dynamic_Weekly_Raw": raw,
                "Dynamic_Prior_Weekly_Date": prior_date,
            })
    return pd.DataFrame(rows)


def v613_k_band(value: Any) -> str:
    number = finite_num(value)
    if not math.isfinite(number):
        return "数据不足"
    if 10.0 <= number < 15.0:
        return "10至15"
    if 15.0 <= number < 20.0:
        return "15至20"
    if 20.0 <= number <= 25.0:
        return "20至25"
    return "区间外"


def v613_checkpoint_path(signature: str, ts_code: str) -> str:
    return os.path.join(
        V613_CHECKPOINT_DIR, signature,
        f"{str(ts_code).replace('.', '_')}.pkl")


def v613_load_checkpoint(
        signature: str, ts_code: str) -> dict[str, Any] | None:
    path = v613_checkpoint_path(signature, ts_code)
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
        record_error(f"V6.13检查点损坏 {ts_code}: {exc}")
        return None


def v613_save_checkpoint(
        signature: str, ts_code: str, events: list[dict[str, Any]],
        rejects: dict[str, int]) -> None:
    atomic_pickle({
        "signature": signature, "ts_code": str(ts_code),
        "events": events, "rejects": rejects,
    }, v613_checkpoint_path(signature, ts_code))


def v613_active_job_path(signature: str) -> str:
    return os.path.join(V613_JOB_DIR, f"{signature}.active")


def v613_mark_job_active(signature: str) -> None:
    atomic_bytes(json.dumps({
        "signature": signature, "version": V613_VERSION,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }, ensure_ascii=False).encode("utf-8"), v613_active_job_path(signature))


def v613_clear_job_active(signature: str) -> None:
    path = v613_active_job_path(signature)
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        record_error(f"V6.13任务标记清除失败: {exc}")


def v613_is_job_active(signature: str) -> bool:
    return os.path.exists(v613_active_job_path(signature))


def v613_daily_indicator_frame(
        daily: pd.DataFrame, weekly_base: pd.DataFrame) -> pd.DataFrame:
    base = add_daily_macd(normalize_price_frame(daily))
    for n in (V613_CONTROL_N, V613_PRIMARY_N):
        daily_skdj = v613_skdj_values(base, n, V613_M)[[
            "trade_date", "SKDJ_K", "SKDJ_D", "SKDJ_KD_Spread"]]
        daily_skdj = daily_skdj.rename(columns={
            "SKDJ_K": f"Daily_SKDJ_N{n}_K",
            "SKDJ_D": f"Daily_SKDJ_N{n}_D",
            "SKDJ_KD_Spread": f"Daily_SKDJ_N{n}_KD_Spread",
        })
        base = base.merge(daily_skdj, on="trade_date", how="left")
        dynamic = v613_dynamic_weekly_skdj(
            base, weekly_base, n, V613_M).rename(columns={
                "Dynamic_Weekly_K": f"Dynamic_N{n}_Weekly_K",
                "Dynamic_Weekly_D": f"Dynamic_N{n}_Weekly_D",
                "Dynamic_Weekly_KD_Spread": f"Dynamic_N{n}_Weekly_KD_Spread",
                "Dynamic_Weekly_K_Change": f"Dynamic_N{n}_Weekly_K_Change",
                "Dynamic_Weekly_Raw": f"Dynamic_N{n}_Weekly_Raw",
                "Dynamic_Prior_Weekly_Date": f"Dynamic_N{n}_Prior_Weekly_Date",
            })
        base = base.merge(dynamic, on="trade_date", how="left")
    return base


def v613_analyze_stock(
        stock: pd.Series, periods: list[dict[str, str]], daily: pd.DataFrame,
        cached_basic: pd.DataFrame, storage_path: str,
        week_last_map: dict[pd.Timestamp, str], open_dates: list[str],
        open_pos: dict[str, int], config: dict[str, Any], use_cache: bool,
        api_pause: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rejects: dict[str, int] = {}
    weekly_base = aggregate_complete_weekly(daily, week_last_map)
    if weekly_base.empty:
        return [], rejects
    daily_features = v613_daily_indicator_frame(daily, weekly_base)
    weekly_by_n = {
        n: add_skdj(weekly_base, n)
        for n in (V613_CONTROL_N, V613_PRIMARY_N)
    }
    attached = {
        n: attach_latest_completed_weekly(daily_features, weekly_by_n[n])
        for n in (V613_CONTROL_N, V613_PRIMARY_N)
    }
    event_end = str(config.get("event_signal_end", config["signal_end"]))
    specifications: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for spec in V613_STRATEGIES:
        n = int(spec["n"])
        frame = attached[n].copy()
        dates = frame["trade_date"].astype(str)
        formal = dates.between(config["signal_start"], event_end)
        positive = true_mask(frame, "Daily_MACD_Positive")
        if spec["entry_mode"] == "首次上涨3%":
            quality = (
                frame["Daily_MACD_State"].astype(str).eq("红柱扩张")
                | numeric(frame, "Daily_MACD_Remaining_pct").ge(
                    STRENGTH_MIN_REMAINING_PCT))
            mask = (
                formal & positive & quality
                & numeric(frame, "Setup_Weekly_K").between(
                    EARLY_WEEKLY_K_MIN, EARLY_WEEKLY_K_MAX,
                    inclusive="both")
                & numeric(frame, "Daily_Return_Since_Red_Start_pct").ge(
                    V63_STRENGTH_THRESHOLD))
        else:
            dynamic_k = numeric(frame, f"Dynamic_N{n}_Weekly_K")
            dynamic_change = numeric(
                frame, f"Dynamic_N{n}_Weekly_K_Change")
            mask = (
                formal & positive
                & numeric(frame, "Daily_MACD_Red_Age").eq(2)
                & dynamic_k.between(
                    V613_APPROACH_K_MIN, V613_APPROACH_K_MAX,
                    inclusive="both")
                & dynamic_change.gt(0))
        selected = frame[mask].copy().sort_values("trade_date")
        if not selected.empty:
            selected = selected.groupby(
                "Daily_MACD_Cycle", as_index=False, sort=False).first()
        specifications.append((spec, selected))

    latest_date = str(config.get("latest_market_date", config["market_end"]))
    watch_specs: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for n in (V613_CONTROL_N, V613_PRIMARY_N):
        frame = attached[n]
        dynamic_k = numeric(frame, f"Dynamic_N{n}_Weekly_K")
        dynamic_change = numeric(frame, f"Dynamic_N{n}_Weekly_K_Change")
        watch = frame[
            frame["trade_date"].astype(str).eq(latest_date)
            & dynamic_k.between(
                V613_APPROACH_K_MIN, V613_APPROACH_K_MAX,
                inclusive="both")
            & dynamic_change.gt(0)
        ].copy().tail(1)
        watch_specs.append(({
            "key": f"N{n}_LIVE_WATCH",
            "label": f"N{n}_动态周线接近25观察池",
            "n": n, "weekly_mode": "截至当日动态周线",
            "entry_mode": "仅观察",
        }, watch))

    estimated = sum(len(frame) for _, frame in specifications + watch_specs)
    if estimated == 0:
        return [], rejects
    code = str(stock["ts_code"])
    basic = ensure_daily_basic(
        code, config["data_start"], config["market_end"], daily,
        cached_basic, storage_path, use_cache, api_pause)
    if basic.empty:
        rejects["存在V6.13信号但daily_basic缺失"] = estimated
        return [], rejects
    rows: list[dict[str, Any]] = []
    outcome_cache: dict[str, dict[str, Any]] = {}
    daily_lookup = daily_features.set_index(
        daily_features["trade_date"].astype(str))

    def append_signal(
            spec: dict[str, Any], signal: pd.Series,
            is_watch: bool = False) -> None:
        signal_date = str(signal["trade_date"])
        membership = membership_on_date(periods, signal_date)
        snapshot = market_snapshot(basic, signal_date)
        reason = ""
        if membership is None:
            reason = "信号日不在历史科技池"
        elif not (str(stock["list_date"]) <= signal_date
                  < str(stock["delist_date"])):
            reason = "信号日上市状态无效"
        elif (not math.isfinite(snapshot["Raw_Close"])
              or snapshot["Raw_Close"] < config["min_price"]):
            reason = "信号日股价不足"
        elif (not math.isfinite(snapshot["Circ_MV_Billion"])
              or snapshot["Circ_MV_Billion"] < config["min_mv"]):
            reason = "信号日流通市值不足"
        if reason or membership is None:
            rejects[reason] = rejects.get(reason, 0) + 1
            return
        n = int(spec["n"])
        if spec["weekly_mode"] == "最近完整周":
            signal_k = finite_num(signal.get("Setup_Weekly_K"))
            signal_d = finite_num(signal.get("Setup_Weekly_D"))
            signal_k_change = finite_num(
                signal.get("Setup_Weekly_K_Change_1W"))
            setup_date = str(signal.get("Setup_Weekly_Date", ""))
        else:
            signal_k = finite_num(signal.get(f"Dynamic_N{n}_Weekly_K"))
            signal_d = finite_num(signal.get(f"Dynamic_N{n}_Weekly_D"))
            signal_k_change = finite_num(
                signal.get(f"Dynamic_N{n}_Weekly_K_Change"))
            setup_date = str(
                signal.get(f"Dynamic_N{n}_Prior_Weekly_Date", ""))
        score_k = int(math.isfinite(signal_k) and 15.0 <= signal_k <= 20.0)
        red_age = int(finite_num(signal.get("Daily_MACD_Red_Age")))
        score_age = int(3 <= red_age <= 5)
        score_kd = int(
            math.isfinite(signal_k) and math.isfinite(signal_d)
            and signal_k < signal_d)
        row: dict[str, Any] = {
            "Event_Type": "LIVE_WATCH" if is_watch else "ENTRY_SIGNAL",
            "Strategy_Key": str(spec["key"]),
            "Strategy_Label": str(spec["label"]),
            "Weekly_SKDJ_N": n, "SKDJ_M": V613_M,
            "Weekly_Data_Mode": str(spec["weekly_mode"]),
            "Entry_Trigger_Mode": str(spec["entry_mode"]),
            "ts_code": code, "name": str(stock["name"]),
            "Signal_Date": signal_date,
            "Signal_Week": str(
                pd.Timestamp(signal_date).to_period("W-FRI")),
            "SW_L1": membership["l1"], "SW_L2": membership["l2"],
            "SW_L3": membership["l3"], **snapshot,
            "Setup_Weekly_Date": setup_date,
            "Signal_K": signal_k, "Signal_D": signal_d,
            "Signal_KD_Spread": signal_k - signal_d,
            "Signal_K_Change_1W": signal_k_change,
            "Weekly_K_Approach_Band": v613_k_band(signal_k),
            **macd_snapshot(signal),
            "Signal_Rally_From_Red_Start_pct": finite_num(
                signal.get("Daily_Return_Since_Red_Start_pct")),
            "Signal_Daily_SKDJ_N6_K": finite_num(
                signal.get("Daily_SKDJ_N6_K")),
            "Signal_Daily_SKDJ_N6_D": finite_num(
                signal.get("Daily_SKDJ_N6_D")),
            "Signal_Daily_SKDJ_N9_K": finite_num(
                signal.get("Daily_SKDJ_N9_K")),
            "Signal_Daily_SKDJ_N9_D": finite_num(
                signal.get("Daily_SKDJ_N9_D")),
            "Timing_K15_20_Pass": bool(score_k),
            "Timing_RedAge3_5_Pass": bool(score_age),
            "Timing_WeeklyK_BelowD_Pass": bool(score_kd),
            "Timing_Score_221": (
                V63_SCORE_K_WEIGHT * score_k
                + V63_SCORE_AGE_WEIGHT * score_age
                + V63_SCORE_KD_WEIGHT * score_kd),
            "Timing_Three_Factor_Count": score_k + score_age + score_kd,
        }
        if not is_watch:
            if signal_date not in outcome_cache:
                outcome_cache[signal_date] = daily_timing_outcomes(
                    daily_features, signal_date, code, open_dates, open_pos,
                    config)
            outcome = outcome_cache[signal_date]
            row.update({f"Entry_{key}": value for key, value in outcome.items()})
            row["Explosion_Class_40D"] = _v63_class_from_outcome(outcome)
            row["Explosion_Grade_40D"] = {
                "F": 0, "B": 1, "A": 2, "S": 3,
            }.get(row["Explosion_Class_40D"], np.nan)
            entry_date = normalize_date(outcome.get("Date"))
            entry_frame = (
                daily_lookup.loc[[entry_date]] if entry_date in daily_lookup.index
                else pd.DataFrame())
            if not entry_frame.empty:
                entry_row = entry_frame.iloc[-1]
                row.update({
                    "Entry_Close_MACD_Red_Age_Audit": int(finite_num(
                        entry_row.get("Daily_MACD_Red_Age"))),
                    "Entry_Close_MACD_State_Audit": str(
                        entry_row.get("Daily_MACD_State", "")),
                    "Entry_Close_Daily_SKDJ_N6_K_Audit": finite_num(
                        entry_row.get("Daily_SKDJ_N6_K")),
                    "Entry_Close_Daily_SKDJ_N6_D_Audit": finite_num(
                        entry_row.get("Daily_SKDJ_N6_D")),
                    "Entry_Close_Daily_SKDJ_N9_K_Audit": finite_num(
                        entry_row.get("Daily_SKDJ_N9_K")),
                    "Entry_Close_Daily_SKDJ_N9_D_Audit": finite_num(
                        entry_row.get("Daily_SKDJ_N9_D")),
                })
        rows.append(row)

    for spec, selected in specifications:
        for _, signal in selected.iterrows():
            append_signal(spec, signal, False)
    for spec, watch in watch_specs:
        for _, signal in watch.iterrows():
            append_signal(spec, signal, True)
    return rows, rejects


def v613_calibration_audit(
        data_start: str, market_end: str,
        week_last_map: dict[pd.Timestamp, str], use_cache: bool,
        api_pause: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_frames: list[tuple[dict[str, Any], pd.DataFrame]] = []
    missing_rows: list[dict[str, Any]] = []
    for reference in V613_CALIBRATION_REFERENCES:
        daily, _, _, _ = fetch_price(
            str(reference["ts_code"]), data_start, market_end,
            use_cache, api_pause)
        if daily.empty:
            missing_rows.append({
                **reference, "状态": "行情缺失"})
            continue
        frame = (
            aggregate_complete_weekly(daily, week_last_map)
            if reference["period"] == "W"
            else normalize_price_frame(daily))
        if frame.empty or not frame["trade_date"].astype(str).eq(
                str(reference["date"])).any():
            missing_rows.append({
                **reference, "状态": "校准日期行情缺失"})
            continue
        reference_frames.append((reference, frame))
    detail_rows: list[dict[str, Any]] = []
    for n in V613_PARAMETER_N_RANGE:
        for m in V613_PARAMETER_M_RANGE:
            for reference, frame in reference_frames:
                calculated = v613_skdj_values(frame, n, m)
                selected = calculated[
                    calculated["trade_date"].astype(str).eq(
                        str(reference["date"]))]
                k_value = (
                    finite_num(selected.iloc[-1].get("SKDJ_K"))
                    if not selected.empty else np.nan)
                d_value = (
                    finite_num(selected.iloc[-1].get("SKDJ_D"))
                    if not selected.empty else np.nan)
                detail_rows.append({
                    "N": n, "M": m,
                    "股票代码": reference["ts_code"],
                    "股票名称": reference["name"],
                    "周期": "周线" if reference["period"] == "W" else "日线",
                    "日期": reference["date"],
                    "目标K": reference["target_k"],
                    "目标D": reference["target_d"],
                    "计算K": k_value, "计算D": d_value,
                    "K绝对误差": abs(k_value - float(reference["target_k"])),
                    "D绝对误差": abs(d_value - float(reference["target_d"])),
                })
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return pd.DataFrame(missing_rows), detail
    detail["单点总绝对误差"] = (
        numeric(detail, "K绝对误差") + numeric(detail, "D绝对误差"))
    summary_rows: list[dict[str, Any]] = []
    expected_points = len(V613_CALIBRATION_REFERENCES)
    for (n, m), group in detail.groupby(["N", "M"], sort=True):
        errors = pd.concat([
            numeric(group, "K绝对误差"), numeric(group, "D绝对误差")
        ], ignore_index=True).dropna()
        summary_rows.append({
            "N": int(n), "M": int(m),
            "成功校准点": len(group), "要求校准点": expected_points,
            "K与D平均绝对误差": errors.mean(),
            "K与D最大绝对误差": errors.max(),
            "四点总绝对误差": numeric(
                group, "单点总绝对误差").sum(),
            "全部点误差不超过0.10": bool(
                len(group) == expected_points
                and errors.le(V613_CALIBRATION_TOLERANCE).all()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["成功校准点", "四点总绝对误差", "K与D最大绝对误差"],
        ascending=[False, True, True]).reset_index(drop=True)
    summary["误差排名"] = np.arange(1, len(summary) + 1)
    summary["参数身份"] = "其他候选"
    summary.loc[
        numeric(summary, "N").eq(9) & numeric(summary, "M").eq(3),
        "参数身份"] = "推测默认_N9_M3"
    summary.loc[
        numeric(summary, "N").eq(6) & numeric(summary, "M").eq(3),
        "参数身份"] = "旧代码_N6_M3"
    if missing_rows:
        missing = pd.DataFrame(missing_rows)
        missing["N"] = np.nan
        missing["M"] = np.nan
    return summary, detail


def v613_event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["Strategy_Key", "Strategy_Label"]
    for keys, group in events.groupby(group_columns, dropna=False):
        strategy_key, label = keys
        classes = group["Explosion_Class_40D"].astype(str)
        returns = numeric(group, "Entry_Close_Return_Net_pct")
        rows.append({
            "Strategy_Key": strategy_key, "策略": label,
            "成熟事件": len(group), "不同股票": group["ts_code"].nunique(),
            "信号日": group["Signal_Date"].nunique(),
            "信号周": group["Signal_Week"].nunique(),
            "红柱日龄均值": numeric(group, "Daily_MACD_Red_Age").mean(),
            "S级比例%": classes.eq("S").mean() * 100.0,
            "A或S比例%": classes.isin(["A", "S"]).mean() * 100.0,
            "B级以上比例%": classes.isin(["B", "A", "S"]).mean() * 100.0,
            "F级比例%": classes.eq("F").mean() * 100.0,
            "40日盈利比例%": returns.gt(0).mean() * 100.0,
            "40日收益均值%": returns.mean(),
            "40日收益中位%": returns.median(),
            "最大浮盈中位%": numeric(group, "Entry_MFE_Net_pct").median(),
            "最大回撤均值%": numeric(group, "Entry_MAE_Raw_pct").mean(),
        })
    return pd.DataFrame(rows)


def v613_band_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in events.groupby(
            ["Strategy_Label", "Weekly_K_Approach_Band"], dropna=False):
        label, band = keys
        classes = group["Explosion_Class_40D"].astype(str)
        returns = numeric(group, "Entry_Close_Return_Net_pct")
        rows.append({
            "策略": label, "周线K区间": band, "事件": len(group),
            "S或A比例%": classes.isin(["S", "A"]).mean() * 100.0,
            "F级比例%": classes.eq("F").mean() * 100.0,
            "40日盈利比例%": returns.gt(0).mean() * 100.0,
            "40日收益均值%": returns.mean(),
            "40日收益中位%": returns.median(),
        })
    return pd.DataFrame(rows)


def v613_matched_entry_audit(
        events: pd.DataFrame, open_dates: list[str]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    old = events[events["Strategy_Key"].eq("OLD_N6_STRENGTH3")].copy()
    new = events[events["Strategy_Key"].isin(
        ["N6_DYNAMIC_RED2", "N9_DYNAMIC_RED2"])].copy()
    if old.empty or new.empty:
        return pd.DataFrame(), pd.DataFrame()
    match_columns = ["ts_code", "Daily_MACD_Cycle"]
    old = old.sort_values("Signal_Date").drop_duplicates(match_columns)
    old = old[match_columns + [
        "Signal_Date", "Entry_Date", "Entry_Raw_Open",
        "Entry_Close_Return_Net_pct", "Explosion_Class_40D",
    ]].rename(columns={
        "Signal_Date": "Old_Signal_Date",
        "Entry_Date": "Old_Entry_Date",
        "Entry_Raw_Open": "Old_Entry_Raw_Open",
        "Entry_Close_Return_Net_pct": "Old_40D_Return_pct",
        "Explosion_Class_40D": "Old_Explosion_Class",
    })
    detail = new.merge(old, on=match_columns, how="left")
    open_pos = {value: position for position, value in enumerate(open_dates)}
    detail["Matched_Old_Strength3"] = detail["Old_Entry_Date"].map(
        normalize_date).str.len().eq(8)
    detail["Entry_Market_Days_Earlier"] = [
        (open_pos.get(normalize_date(old_date), np.nan)
         - open_pos.get(normalize_date(new_date), np.nan))
        for new_date, old_date in zip(
            detail["Entry_Date"], detail["Old_Entry_Date"])
    ]
    detail["Earlier_Entry_Price_Advantage_pct"] = np.where(
        numeric(detail, "Entry_Raw_Open").gt(0)
        & numeric(detail, "Old_Entry_Raw_Open").gt(0),
        (numeric(detail, "Old_Entry_Raw_Open")
         / numeric(detail, "Entry_Raw_Open") - 1.0) * 100.0,
        np.nan)
    rows: list[dict[str, Any]] = []
    for label, group in detail.groupby("Strategy_Label", dropna=False):
        matched = group[true_mask(group, "Matched_Old_Strength3")]
        rows.append({
            "新策略": label, "全部红2事件": len(group),
            "后来形成旧3%信号": len(matched),
            "后来形成旧3%信号比例%": len(matched) / len(group) * 100.0,
            "未形成旧3%信号的真实早期机会": len(group) - len(matched),
            "匹配样本提前市场日中位": numeric(
                matched, "Entry_Market_Days_Earlier").median(),
            "匹配样本价格优势中位%": numeric(
                matched, "Earlier_Entry_Price_Advantage_pct").median(),
            "匹配样本价格优势5至15比例%": numeric(
                matched, "Earlier_Entry_Price_Advantage_pct").between(
                    5, 15).mean() * 100.0,
        })
    return pd.DataFrame(rows), detail


def v613_weekly_coverage(
        events: pd.DataFrame, open_dates: list[str], start_date: str,
        end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = [value for value in open_dates if start_date <= value <= end_date]
    calendar = pd.DataFrame({"trade_date": days})
    calendar["Signal_Week"] = pd.to_datetime(
        calendar["trade_date"], format="%Y%m%d").dt.to_period(
            "W-FRI").astype(str)
    weeks = calendar.groupby("Signal_Week")["trade_date"].max().rename(
        "Week_Last_Trading_Date").reset_index()
    rows: list[dict[str, Any]] = []
    for spec in V613_STRATEGIES:
        group = events[events["Strategy_Key"].eq(spec["key"])]
        counts = group.groupby("Signal_Week").size()
        column = str(spec["key"])
        weeks[column] = weeks["Signal_Week"].map(counts).fillna(0).astype(int)
        values = numeric(weeks, column)
        rows.append({
            "Strategy_Key": column, "策略": spec["label"],
            "总事件": int(values.sum()),
            "有信号周": int(values.gt(0).sum()),
            "空窗周": int(values.eq(0).sum()),
            "覆盖率%": values.gt(0).mean() * 100.0,
            "最长连续空窗周": max_empty_run(values),
            "每周信号均值": values.mean(), "单周最多": int(values.max()),
        })
    return pd.DataFrame(rows), weeks


def v613_price_asof(book: dict[str, Any], field: str, value: str) -> float:
    direct = finite_num(book.get(field, {}).get(value))
    if math.isfinite(direct) and direct > 0:
        return direct
    dates = list(book.get("dates", []))
    position = bisect.bisect_right(dates, value) - 1
    while position >= 0:
        price = finite_num(book.get(field, {}).get(dates[position]))
        if math.isfinite(price) and price > 0:
            return price
        position -= 1
    return np.nan


def v613_next_stock_date(book: dict[str, Any], after_date: str) -> str:
    dates = list(book.get("dates", []))
    position = bisect.bisect_right(dates, after_date)
    return str(dates[position]) if position < len(dates) else ""


def v613_run_portfolio_path(
        strategy_events: pd.DataFrame,
        price_book: dict[str, dict[str, Any]], open_dates: list[str],
        config: dict[str, Any], exit_mode: str, selection_mode: str,
        seed: int, keep_details: bool = False
        ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    events = strategy_events.copy()
    if events.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    for column in ("Signal_Date", "Entry_Date", "Entry_End_Date_40D"):
        events[column] = events[column].map(normalize_date)
    events = events[
        events["Entry_Date"].str.len().eq(8)
        & events["Entry_End_Date_40D"].str.len().eq(8)
        & events["ts_code"].astype(str).isin(price_book)
    ].copy().reset_index(drop=True)
    if events.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    events["_Event_ID"] = [
        f"{row.Strategy_Key}:{row.ts_code}:{row.Signal_Date}:{number}"
        for number, row in events.iterrows()
    ]
    candidates_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in events.to_dict("records"):
        candidates_by_date[str(record["Entry_Date"])].append(record)
    first_date = min(candidates_by_date)
    last_date = max(events["Entry_End_Date_40D"].astype(str))
    simulation_dates = [
        value for value in open_dates if first_date <= value <= last_date]
    if not simulation_dates:
        return {}, pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    buy_factor, sell_factor = _cost_factors(config)
    open_pos = {value: position for position, value in enumerate(open_dates)}
    cash = float(V66_INITIAL_CAPITAL)
    positions: dict[str, dict[str, Any]] = {}
    pending_open_exits: dict[str, list[tuple[str, str]]] = defaultdict(list)
    selected_event_ids: set[str] = set()
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    leg_number = 0

    def close_position(
            code: str, exit_date: str, field: str, reason: str) -> bool:
        nonlocal cash
        position = positions.get(code)
        if position is None:
            return False
        book = price_book.get(code, {})
        raw_exit = finite_num(book.get(field, {}).get(exit_date))
        if field == "close" and (not math.isfinite(raw_exit) or raw_exit <= 0):
            # Match the 40-day outcome engine: if the stock is suspended on
            # the common market end date, value the planned close at its last
            # observable close inside the horizon rather than leaving an
            # unclosed position out of the trade audit.
            raw_exit = v613_price_asof(book, "close", exit_date)
        if not math.isfinite(raw_exit) or raw_exit <= 0:
            return False
        proceeds = float(position["shares"]) * raw_exit * sell_factor
        cash += proceeds
        entry_cost = float(position["entry_cost"])
        pnl = proceeds - entry_cost
        event = position["event"]
        entry_date = str(position["entry_date"])
        hold_days = (
            open_pos.get(exit_date, -1) - open_pos.get(entry_date, -1) + 1
            if entry_date in open_pos and exit_date in open_pos else np.nan)
        trade_rows.append({
            "Strategy_Key": event.get("Strategy_Key"),
            "策略": event.get("Strategy_Label"),
            "退出方案": exit_mode, "选择方式": selection_mode,
            "Seed": int(seed), "Leg_ID": position["leg_id"],
            "Event_ID": event.get("_Event_ID"),
            "ts_code": code, "name": event.get("name"),
            "信号日": event.get("Signal_Date"),
            "买入日": entry_date, "买入开盘": position["raw_entry"],
            "股数": position["shares"], "投入成本": entry_cost,
            "信号日原始顺位": position["selection_rank"],
            "退出日": exit_date, "退出价格": raw_exit,
            "退出价格字段": field, "退出原因": reason,
            "净回收": proceeds, "净利润": pnl,
            "实际交易收益%": pnl / entry_cost * 100.0,
            "持有市场日": hold_days,
            "Explosion_Class_40D": event.get("Explosion_Class_40D"),
            "Entry_Close_Return_Net_pct": event.get(
                "Entry_Close_Return_Net_pct"),
            "Entry_MFE_Net_pct": event.get("Entry_MFE_Net_pct"),
            "Entry_MAE_Raw_pct": event.get("Entry_MAE_Raw_pct"),
            "Weekly_K_Approach_Band": event.get("Weekly_K_Approach_Band"),
        })
        positions.pop(code, None)
        return True

    for trade_date in simulation_dates:
        # A close signal from a previous day can only be executed now.
        for code, reason in pending_open_exits.pop(trade_date, []):
            if not close_position(code, trade_date, "open", reason):
                next_date = v613_next_stock_date(
                    price_book.get(code, {}), trade_date)
                position = positions.get(code)
                if (next_date and position is not None
                        and next_date <= str(position["base_exit_date"])):
                    pending_open_exits[next_date].append((code, reason))

        day_candidates = candidates_by_date.get(trade_date, [])
        if day_candidates:
            day = pd.DataFrame(day_candidates).drop_duplicates(
                "ts_code", keep="first")
            if selection_mode == "规则排序":
                day = day.sort_values(
                    ["Timing_Score_221", "Daily_MACD_Hist", "ts_code"],
                    ascending=[False, True, True])
            else:
                order = rng.permutation(len(day))
                day = day.iloc[order]
            day = day.head(V66_MAX_STOCKS).copy()
            day["_Selection_Rank"] = np.arange(1, len(day) + 1)
            # No fallback: only the original daily Top3 may be considered.
            for event in day.to_dict("records"):
                event_id = str(event.get("_Event_ID", ""))
                code = str(event.get("ts_code", ""))
                if (event_id in selected_event_ids or code in positions
                        or len(positions) >= V66_MAX_STOCKS):
                    continue
                book = price_book.get(code, {})
                raw_open = finite_num(book.get("open", {}).get(trade_date))
                if not math.isfinite(raw_open) or raw_open <= 0:
                    continue
                budget = min(float(V66_FULL_SLOT_CAPITAL), cash)
                shares = math.floor(
                    budget / (raw_open * buy_factor * V66_BOARD_LOT)
                ) * V66_BOARD_LOT
                if shares < V66_BOARD_LOT:
                    continue
                entry_cost = float(shares) * raw_open * buy_factor
                if entry_cost > cash + 1e-7:
                    continue
                cash -= entry_cost
                leg_number += 1
                base_exit = normalize_date(event.get("Entry_End_Date_40D"))
                d3_date = v66_market_date_offset(
                    trade_date, 2, open_dates, open_pos)
                positions[code] = {
                    "event": event, "entry_date": trade_date,
                    "raw_entry": raw_open, "shares": int(shares),
                    "entry_cost": entry_cost, "base_exit_date": base_exit,
                    "d3_date": d3_date, "d3_scheduled": False,
                    "selection_rank": int(event.get("_Selection_Rank", 0)),
                    "leg_id": f"L{leg_number}",
                }
                selected_event_ids.add(event_id)

        # D3 hard failure is evaluated at the close and traded later.
        if exit_mode == "EarlyFailureD3":
            for code, position in list(positions.items()):
                if (position["d3_scheduled"]
                        or str(position["d3_date"]) != trade_date):
                    continue
                book = price_book.get(code, {})
                close = finite_num(book.get("close", {}).get(trade_date))
                hist = finite_num(book.get("macd_hist", {}).get(trade_date))
                remaining = finite_num(
                    book.get("macd_remaining", {}).get(trade_date))
                retention = finite_num(
                    book.get("macd_retention", {}).get(trade_date))
                price_fail = (
                    math.isfinite(close) and close > 0
                    and (close / float(position["raw_entry"]) - 1.0) * 100.0
                    <= V612_PRICE_FAILURE_PCT)
                green = math.isfinite(hist) and hist <= 0
                exhausted = (
                    math.isfinite(hist) and hist > 0
                    and math.isfinite(remaining)
                    and remaining <= V612_MACD_REMAINING_PCT
                    and math.isfinite(retention) and retention < 100.0)
                reasons = []
                if green:
                    reasons.append("MACD翻绿")
                if exhausted:
                    reasons.append("红柱剩余≤10且缩短")
                if price_fail:
                    reasons.append("相对买入≤-5%")
                if reasons:
                    next_date = v613_next_stock_date(book, trade_date)
                    if (next_date and next_date
                            <= str(position["base_exit_date"])):
                        pending_open_exits[next_date].append((
                            code, "D3硬失败_" + "+".join(reasons)))
                        position["d3_scheduled"] = True

        # Fixed-horizon exits occur at that day's close.
        for code, position in list(positions.items()):
            if str(position["base_exit_date"]) == trade_date:
                close_position(code, trade_date, "close", "固定40日收盘")

        market_value = 0.0
        for code, position in positions.items():
            close = v613_price_asof(
                price_book.get(code, {}), "close", trade_date)
            if math.isfinite(close):
                market_value += float(position["shares"]) * close
        equity = cash + market_value
        equity_rows.append({
            "Strategy_Key": events.iloc[0]["Strategy_Key"],
            "策略": events.iloc[0]["Strategy_Label"],
            "退出方案": exit_mode, "选择方式": selection_mode,
            "Seed": int(seed), "trade_date": trade_date,
            "Cash": cash, "Market_Value": market_value,
            "Equity": equity, "Positions": len(positions),
            "Exposure_pct": (
                market_value / equity * 100.0 if equity > 0 else np.nan),
        })

    equity_frame = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    if equity_frame.empty:
        return {}, equity_frame, trades
    equity_values = numeric(equity_frame, "Equity")
    drawdown = (equity_values / equity_values.cummax() - 1.0) * 100.0
    if trades.empty:
        classes = pd.Series(dtype=str)
        actual_returns = pd.Series(dtype=float)
    else:
        classes = trades["Explosion_Class_40D"].astype(str)
        actual_returns = numeric(trades, "实际交易收益%")

    sequential_triples: list[bool] = []
    same_day_triples: list[bool] = []
    if len(trades) >= 3:
        ordered = trades.sort_values(
            ["买入日", "信号日原始顺位", "Leg_ID"]).reset_index(drop=True)
        for start in range(0, len(ordered) - 2, 3):
            group = ordered.iloc[start:start + 3]
            if len(group) == 3:
                sequential_triples.append(
                    bool(numeric(group, "实际交易收益%").gt(0).sum() >= 2))
        for _, group in trades.groupby("买入日", sort=True):
            if len(group) == 3:
                same_day_triples.append(
                    bool(numeric(group, "Entry_Close_Return_Net_pct")
                         .gt(0).sum() >= 2))
    f_mask = classes.eq("F")
    sa_mask = classes.isin(["S", "A"])
    metrics = {
        "Strategy_Key": events.iloc[0]["Strategy_Key"],
        "策略": events.iloc[0]["Strategy_Label"],
        "退出方案Key": exit_mode,
        "退出方案": dict(V613_PORTFOLIO_EXITS).get(exit_mode, exit_mode),
        "选择方式": selection_mode, "Seed": int(seed),
        "期初资金": V66_INITIAL_CAPITAL,
        "期末权益": float(equity_values.iloc[-1]),
        "总收益率%": (
            equity_values.iloc[-1] / V66_INITIAL_CAPITAL - 1.0) * 100.0,
        "最大回撤%": drawdown.min(),
        "平均资金暴露%": numeric(
            equity_frame, "Exposure_pct").mean(),
        "平均持股数": numeric(equity_frame, "Positions").mean(),
        "最大持股数": int(numeric(equity_frame, "Positions").max()),
        "实际买入事件": len(trades),
        "实际交易胜率%": (
            actual_returns.gt(0).mean() * 100.0 if len(trades) else np.nan),
        "S级比例%": (
            classes.eq("S").mean() * 100.0 if len(trades) else np.nan),
        "A或S比例%": (
            sa_mask.mean() * 100.0 if len(trades) else np.nan),
        "F级比例%": (
            f_mask.mean() * 100.0 if len(trades) else np.nan),
        "F级名额占用市场日": (
            numeric(trades[f_mask], "持有市场日").sum()
            if len(trades) else 0.0),
        "S或A平均持有市场日": (
            numeric(trades[sa_mask], "持有市场日").mean()
            if len(trades) else np.nan),
        "实际买入覆盖周": (
            pd.to_datetime(trades["买入日"], format="%Y%m%d")
            .dt.to_period("W-FRI").nunique() if len(trades) else 0),
        "连续每3笔组数": len(sequential_triples),
        "连续3笔至少2笔盈利比例%": (
            np.mean(sequential_triples) * 100.0
            if sequential_triples else np.nan),
        "同日正好买3只组数": len(same_day_triples),
        "同日3只至少2只40日盈利比例%": (
            np.mean(same_day_triples) * 100.0
            if same_day_triples else np.nan),
    }
    return (
        metrics,
        equity_frame if keep_details else pd.DataFrame(),
        trades if keep_details else pd.DataFrame(),
    )


def v613_run_portfolio_ensemble(
        events: pd.DataFrame, price_book: dict[str, dict[str, Any]],
        open_dates: list[str], config: dict[str, Any], draws: int
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path_rows: list[dict[str, Any]] = []
    equity_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    for spec in V613_STRATEGIES:
        strategy = events[events["Strategy_Key"].eq(spec["key"])].copy()
        if strategy.empty:
            continue
        for exit_key, _ in V613_PORTFOLIO_EXITS:
            metrics, equity, trades = v613_run_portfolio_path(
                strategy, price_book, open_dates, config, exit_key,
                "规则排序", V613_RANDOM_SEED, keep_details=True)
            if metrics:
                metrics["Path_No"] = 0
                path_rows.append(metrics)
            if not equity.empty:
                equity_parts.append(equity)
            if not trades.empty:
                trade_parts.append(trades)
            for path_no in range(max(int(draws), 1)):
                seed = V613_RANDOM_SEED + path_no
                random_metrics, _, _ = v613_run_portfolio_path(
                    strategy, price_book, open_dates, config, exit_key,
                    "完全随机选择", seed, keep_details=False)
                if random_metrics:
                    random_metrics["Path_No"] = path_no
                    path_rows.append(random_metrics)
    paths = pd.DataFrame(path_rows)
    summary_rows: list[dict[str, Any]] = []
    metrics_columns = [
        "期末权益", "总收益率%", "最大回撤%", "平均资金暴露%",
        "平均持股数", "实际买入事件", "实际交易胜率%", "S级比例%",
        "A或S比例%", "F级比例%", "F级名额占用市场日",
        "实际买入覆盖周", "连续3笔至少2笔盈利比例%",
        "同日3只至少2只40日盈利比例%",
    ]
    if not paths.empty:
        for keys, group in paths.groupby(
                ["Strategy_Key", "策略", "退出方案Key", "退出方案",
                 "选择方式"], dropna=False):
            strategy_key, label, exit_key, exit_label, mode = keys
            row: dict[str, Any] = {
                "Strategy_Key": strategy_key, "策略": label,
                "退出方案Key": exit_key, "退出方案": exit_label,
                "选择方式": mode, "路径数": len(group),
            }
            for column in metrics_columns:
                values = numeric(group, column)
                row[column] = values.median()
                if mode == "完全随机选择":
                    row[f"{column}_P10"] = values.quantile(0.10)
                    row[f"{column}_P90"] = values.quantile(0.90)
            summary_rows.append(row)
    return (
        paths,
        pd.DataFrame(summary_rows),
        pd.concat(equity_parts, ignore_index=True)
        if equity_parts else pd.DataFrame(),
        pd.concat(trade_parts, ignore_index=True)
        if trade_parts else pd.DataFrame(),
    )


def v613_main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=V613_TITLE, layout="wide")
    st.title(V613_TITLE)
    streamlit_version = str(getattr(st, "__version__", "unknown"))
    st.caption(f"{V613_UI_PATCH}｜Streamlit {streamlit_version}")
    with st.expander("V6.13验证口径", expanded=True):
        st.markdown(f"""
- **先校准再下结论**：按用户提供的EMA—EMA—MA公式，扫描N=2～60、M=2～10，同时匹配1个周线和3个日线K/D参照点；N=9、M=3只是主假设，不预设为正确答案。
- **旧基准**：最近完整周N=6、K在15～25，日线MACD质量合格且本轮首次上涨3%，收盘确认后下一市场交易日开盘买入。
- **新主方案**：截至每个信号日收盘可见的动态周线N=9、K在{V613_APPROACH_K_MIN:g}～{V613_APPROACH_K_MAX:g}且高于上一完整周K；日线MACD第2根红柱收盘确认，下一市场交易日开盘买入。实际买入日回看通常已是第3根红柱。
- **N=6对照**：与新主方案完全相同，只把周线N改为6，用于判断N=9的平滑是否真正减少F，而不是凭默认参数猜测。
- **动态周线无未来数据**：周一至周四只使用当周截至当天的最高、最低和收盘；周五收盘值等于完整周线。程序同时导出最近完整周日期和动态K/D。
- **全额等分三仓**：初始资金30万元，最多3只，每只一次性投入约10万元，100股整手，计入滑点、佣金、过户费和印花税；不试仓、不加仓。
- **冻结排序**：继续使用2-2-1分降序、完全同分时MACD柱较小优先；同日只认原始Top3，不因已持仓或现金不足向下递补。红柱第2日方案的日龄项恒为0，因此不会伪造新的评分优势。
- **两个退出口径**：固定40日用于纯粹比较买点；D3组只在第3市场日收盘出现MACD翻绿、红柱剩余≤10%且缩短、或相对买入≤-5%时于下一可交易开盘退出，否则仍持有40日。
- **完整早期机会**：红柱第2日策略纳入所有当时合格周期，包括后来从未上涨3%的失败周期；匹配旧信号的价格优势单独标注为幸存者样本，不能替代全样本结果。
- **判卷**：S/A/B/F从各自新的买入价重新计算；同时验收F比例、S/A比例、真实三仓收益与回撤、覆盖周、F名额占用日以及每3笔至少2笔盈利。
""")

    with st.sidebar:
        st.header("运行参数")
        backtest_days = st.number_input(
            "回测交易日数", 100, 1000, 500, 50, key="v613_days")
        min_price = st.number_input(
            "最低股价（元）", 10.0, 20.0, 10.0, 10.0,
            format="%.0f", key="v613_min_price")
        min_mv = st.number_input(
            "最低流通市值（亿元）", 50.0, 100.0, 50.0, 50.0,
            format="%.0f", key="v613_min_mv")
        signal_end_date = st.date_input(
            "历史买入信号截止（40日判卷）", date(2026, 6, 5),
            key="v613_signal_end")
        market_end_date = st.date_input(
            "最新信号观察截止（默认今天）", date.today(),
            key="v613_market_end")
        pause = st.number_input(
            "接口间隔(秒)", 0.0, 2.0, 0.12, 0.02,
            key="v613_pause")
        use_cache = st.checkbox(
            "复用行情和72小时基础缓存", True, key="v613_cache")
        portfolio_draws = st.number_input(
            "每组随机对照路径数", 50, 500, 100, 50,
            key="v613_draws")
        st.caption("3套买点×2种退出；规则路径各1条，随机路径用于判断Top3排序是否提供优势。")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005,
            format="%.3f", key="v613_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01,
            format="%.3f", key="v613_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001,
            format="%.3f", key="v613_transfer")
        if st.button("清除V6.13结果和运行状态", key="v613_clear"):
            shutil.rmtree(V613_RESULT_DIR, ignore_errors=True)
            shutil.rmtree(V613_JOB_DIR, ignore_errors=True)
            st.success("结果和运行状态已清除；行情缓存与逐股票检查点保留。")

    request_payload = {
        "version": V613_VERSION, "days": int(backtest_days),
        "signal_end": signal_end_date.strftime("%Y%m%d"),
        "market_end": market_end_date.strftime("%Y%m%d"),
        "min_price": float(min_price), "min_mv": float(min_mv),
        "commission": float(commission_pct),
        "stamp": float(stamp_duty_pct),
        "transfer": float(transfer_fee_pct),
        "calibration_n": list(V613_PARAMETER_N_RANGE),
        "calibration_m": list(V613_PARAMETER_M_RANGE),
        "calibration_references": list(V613_CALIBRATION_REFERENCES),
        "strategies": list(V613_STRATEGIES),
        "approach_k": [V613_APPROACH_K_MIN, V613_APPROACH_K_MAX],
        "red_age": 2, "initial_capital": V66_INITIAL_CAPITAL,
        "max_stocks": V66_MAX_STOCKS,
        "slot_capital": V66_FULL_SLOT_CAPITAL,
        "exit_modes": [value[0] for value in V613_PORTFOLIO_EXITS],
        "portfolio_draws": int(portfolio_draws),
    }
    request_signature = stable_signature(request_payload)
    result_path = os.path.join(
        V613_RESULT_DIR, f"{request_signature}.zip")
    result_name = (
        f"weekly_skdj_n9_red2_entry_timing_v6_13_"
        f"{int(backtest_days)}d_p{int(min_price)}_mv{int(min_mv)}.zip")
    completed_available = False
    if os.path.exists(result_path):
        try:
            with open(result_path, "rb") as handle:
                saved_result = handle.read()
            completed_available = True
            v613_clear_job_active(request_signature)
            st.success("发现相同参数的V6.13已完成结果，可直接下载。")
            render_download(
                saved_result, result_name,
                f"v613_saved_{request_signature}")
        except Exception as exc:
            st.warning(f"已保存结果读取失败：{exc}")

    secret_token = configured_tushare_token()
    if secret_token:
        token = secret_token
        st.caption("已从Streamlit Secrets读取TUSHARE_TOKEN。")
    else:
        token = st.text_input(
            "Tushare Token", type="password", key="v613_token")
    job_active = v613_is_job_active(request_signature)
    left, right = st.columns(2)
    with left:
        start_clicked = st.button(
            "开始/重新运行V6.13", type="primary", key="v613_run")
    with right:
        stop_clicked = st.button(
            "停止自动续跑", disabled=not job_active, key="v613_stop")
    if stop_clicked:
        v613_clear_job_active(request_signature)
        st.success("已停止；逐股票检查点保留。")
        return
    if start_clicked:
        if market_end_date <= signal_end_date:
            st.error("最新观察截止必须晚于历史信号截止")
            return
        v613_mark_job_active(request_signature)
        job_active = True
    if not token:
        st.info("请输入Token；启动后页面重连会从逐股票检查点续跑。")
        return
    if not job_active:
        st.caption(
            "点击开始运行。" if not completed_available
            else "相同参数结果已可下载；如需覆盖请点击重新运行。")
        return

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    signal_end = signal_end_date.strftime("%Y%m%d")
    market_end = market_end_date.strftime("%Y%m%d")
    try:
        probe_start = signal_end_date - timedelta(
            days=int(backtest_days) * 2 + 120)
        probe_dates = load_trade_calendar(
            probe_start.strftime("%Y%m%d"), signal_end)
        signal_start = trailing_signal_start(
            probe_dates, signal_end, int(backtest_days))
    except Exception as exc:
        st.error(f"确定{int(backtest_days)}个交易日窗口失败：{exc}")
        return
    data_start = (
        pd.Timestamp(signal_start).date()
        - timedelta(weeks=WARMUP_WEEKS, days=7)
    ).strftime("%Y%m%d")
    try:
        with st.spinner("加载交易日历、历史科技池和校准行情..."):
            open_dates = load_trade_calendar(data_start, market_end)
            extended_end = (
                market_end_date + timedelta(days=7)).strftime("%Y%m%d")
            week_last_map = complete_week_last_dates(
                load_trade_calendar(data_start, extended_end))
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
            calibration_summary, calibration_detail = (
                v613_calibration_audit(
                    data_start, market_end, week_last_map,
                    bool(use_cache), float(pause)))
    except Exception as exc:
        st.error(f"基础数据或参数校准加载失败：{exc}")
        return
    eligible_market_dates = [
        value for value in open_dates if value <= market_end]
    if not eligible_market_dates:
        st.error("行情截止日前没有可用市场交易日")
        return
    latest_market_date = max(eligible_market_dates)
    open_pos = {value: position for position, value in enumerate(open_dates)}
    config = {
        "signal_start": signal_start, "signal_end": signal_end,
        "event_signal_end": market_end, "data_start": data_start,
        "market_end": market_end, "latest_market_date": latest_market_date,
        "min_price": float(min_price), "min_mv": float(min_mv),
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct),
        "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct),
    }
    run_signature = stable_signature({
        "version": V613_EVENT_ENGINE_VERSION, **config,
        "primary_n": V613_PRIMARY_N, "control_n": V613_CONTROL_N,
        "m": V613_M, "approach_k": [
            V613_APPROACH_K_MIN, V613_APPROACH_K_MAX],
    })
    period_index = build_period_index(memberships)
    active_codes = {
        code for code, code_periods in period_index.items()
        if periods_overlap(code_periods, signal_start, market_end)}
    stocks = stock_basic[stock_basic["ts_code"].isin(active_codes)].copy()
    stocks = stocks[
        ~stocks["list_date"].gt(market_end)
        & ~stocks["delist_date"].lt(data_start)
    ].sort_values("ts_code").reset_index(drop=True)

    event_rows: list[dict[str, Any]] = []
    rejects: dict[str, int] = {}
    checkpoint_hits = price_cache_hits = failures = 0
    progress, status = st.progress(0.0), st.empty()
    last_update = 0.0
    stopped = False
    for number, stock in stocks.iterrows():
        if not v613_is_job_active(request_signature):
            stopped = True
            break
        code = str(stock["ts_code"])
        checkpoint = (
            v613_load_checkpoint(run_signature, code)
            if bool(use_cache) else None)
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
                    rows, stock_rejects = v613_analyze_stock(
                        stock, period_index.get(code, []), daily,
                        cached_basic, storage_path, week_last_map,
                        open_dates, open_pos, config, bool(use_cache),
                        float(pause))
                    event_rows.extend(rows)
                    merge_counts(rejects, stock_rejects)
                    v613_save_checkpoint(
                        run_signature, code, rows, stock_rejects)
                except Exception as exc:
                    failures += 1
                    record_error(f"V6.13逐股票分析失败 {code}: {exc}")
        processed = number + 1
        now = time.monotonic()
        if (processed == 1 or now - last_update >= UI_HEARTBEAT_SECONDS
                or processed == len(stocks)):
            progress.progress(
                processed / max(len(stocks), 1),
                text=f"已处理{processed}/{len(stocks)}只股票，最近{code}")
            status.caption(
                f"事件/观察{len(event_rows)}；检查点{checkpoint_hits}；"
                f"行情缓存{price_cache_hits}；失败{failures}")
            last_update = now
    progress.empty()
    status.empty()
    if stopped:
        st.warning("任务已停止，逐股票检查点已保留。")
        return
    events_all = pd.DataFrame(event_rows)
    if events_all.empty:
        st.error("本区间没有生成V6.13事件或观察池。")
        return
    events_all = events_all.sort_values(
        ["Signal_Date", "Event_Type", "Strategy_Key", "ts_code"]
    ).reset_index(drop=True)
    live_watch = events_all[events_all["Event_Type"].eq("LIVE_WATCH")].copy()
    signals = events_all[events_all["Event_Type"].eq("ENTRY_SIGNAL")].copy()
    history = signals[signals["Signal_Date"].astype(str).le(signal_end)].copy()
    observation = signals[
        signals["Signal_Date"].astype(str).gt(signal_end)
        & signals["Signal_Date"].astype(str).le(latest_market_date)
    ].copy()
    mature = history[
        true_mask(history, "Entry_Tradable")
        & true_mask(history, "Entry_Has_40D")].copy()
    if mature.empty:
        st.error("存在信号，但没有走完40个市场日的成熟事件。")
        return
    event_summary = v613_event_summary(mature)
    band_summary = v613_band_summary(mature)
    matched_summary, matched_detail = v613_matched_entry_audit(
        mature, open_dates)
    coverage_summary, coverage_calendar = v613_weekly_coverage(
        mature, open_dates, signal_start, signal_end)

    with st.spinner("复用本地行情，构建三仓价格簿..."):
        price_book, portfolio_cache_hits, portfolio_missing = (
            v66_load_price_book(mature, data_start, market_end))
        recovered = 0
        still_missing: list[str] = []
        for code in portfolio_missing:
            daily, _, _, _ = fetch_price(
                code, data_start, market_end, True, float(pause))
            if daily.empty:
                still_missing.append(code)
                continue
            daily = add_daily_macd(normalize_price_frame(daily))
            dates = daily["trade_date"].astype(str).tolist()
            price_book[code] = {
                "dates": dates,
                "open": dict(zip(dates, numeric(daily, "open"))),
                "close": dict(zip(dates, numeric(daily, "close"))),
                "high": dict(zip(dates, numeric(daily, "high"))),
                "low": dict(zip(dates, numeric(daily, "low"))),
                "macd_hist": dict(zip(
                    dates, numeric(daily, "Daily_MACD_Hist"))),
                "macd_remaining": dict(zip(
                    dates, numeric(daily, "Daily_MACD_Remaining_pct"))),
                "macd_retention": dict(zip(
                    dates, numeric(daily, "Daily_MACD_Retention_pct"))),
            }
            recovered += 1
        portfolio_missing = still_missing
    if portfolio_missing:
        st.error(
            f"三仓回测缺少{len(portfolio_missing)}只股票行情，"
            "为避免样本偏差本次停止；保留检查点后重新运行即可补齐。")
        st.dataframe(pd.DataFrame({"缺失股票": portfolio_missing}),
                     use_container_width=True, hide_index=True)
        return
    with st.spinner(
            f"运行3套买点×2种退出的真实三仓及{int(portfolio_draws)}条随机对照..."):
        portfolio_paths, portfolio_summary, deterministic_equity, (
            deterministic_trades) = v613_run_portfolio_ensemble(
                mature, price_book, open_dates, config,
                int(portfolio_draws))
    if portfolio_paths.empty:
        st.error("成熟事件存在，但三仓撮合没有生成可执行路径。")
        return

    recent_start = (
        pd.Timestamp(latest_market_date) - pd.Timedelta(days=30)
    ).strftime("%Y%m%d")
    recent_candidates = observation[
        observation["Signal_Date"].astype(str).between(
            recent_start, latest_market_date)].copy()
    calibration_top = calibration_summary.head(30).copy()
    n6n9_calibration = calibration_summary[
        numeric(calibration_summary, "N").isin([6, 9])
        & numeric(calibration_summary, "M").eq(3)].copy()
    best_calibrated = (
        calibration_summary.iloc[0].to_dict()
        if not calibration_summary.empty else {})
    n9_row = n6n9_calibration[
        numeric(n6n9_calibration, "N").eq(9)]
    n9_confirmed = bool(
        not n9_row.empty
        and to_bool(n9_row.iloc[0].get("全部点误差不超过0.10")))
    run_summary = pd.DataFrame([{
        "程序版本": V613_VERSION,
        "历史信号开始": signal_start, "历史信号截止": signal_end,
        "行情观察截止": market_end,
        "最新市场交易日": latest_market_date,
        "校准最佳N": best_calibrated.get("N", np.nan),
        "校准最佳M": best_calibrated.get("M", np.nan),
        "校准最佳总绝对误差": best_calibrated.get(
            "四点总绝对误差", np.nan),
        "N9_M3是否四点误差均不超过0.10": n9_confirmed,
        "成熟事件总数": len(mature),
        "旧N6上涨3%成熟事件": int(
            mature["Strategy_Key"].eq("OLD_N6_STRENGTH3").sum()),
        "N6红2成熟事件": int(
            mature["Strategy_Key"].eq("N6_DYNAMIC_RED2").sum()),
        "N9红2成熟事件": int(
            mature["Strategy_Key"].eq("N9_DYNAMIC_RED2").sum()),
        "最近30日候选": len(recent_candidates),
        "最新观察池": len(live_watch),
        "组合路径": len(portfolio_paths),
        "组合行情缓存命中": portfolio_cache_hits,
        "组合行情补下载": recovered,
        "处理股票": len(stocks), "检查点恢复": checkpoint_hits,
        "行情缓存命中": price_cache_hits, "失败股票": failures,
    }])
    definitions = pd.DataFrame([
        ("参数校准", "N2至60、M2至10；四个用户图表K/D点同时最小误差"),
        ("N9身份", "主假设；只有四点误差均≤0.10才标记校准通过"),
        ("旧基准", "完整周N6 K15至25；日线上涨3%；下一开盘"),
        ("新主方案", "动态周线N9 K10至25且上升；MACD红2；下一开盘"),
        ("参数对照", "动态周线N6 K10至25且上升；MACD红2；下一开盘"),
        ("仓位", "30万元；最多3只；每只约10万元；整手；不加仓"),
        ("固定40日", "隔离买点差异；第40市场日收盘退出"),
        ("D3硬失败", "D3收盘MACD翻绿/红柱耗尽/跌5%；下一开盘退出"),
        ("等级", "从各自买入价重新计算S/A/B/F；不沿用旧等级"),
        ("价格优势", "只在同股同MACD周期且后来形成旧3%信号的配对样本计算"),
        ("防幸存偏差", "红2全量事件包含后来从未达到上涨3%的周期"),
    ], columns=["项目", "定义"])
    rejection_audit = pd.DataFrame([
        {"剔除原因": key, "次数": value}
        for key, value in sorted(rejects.items())])
    cache_policy = pd.DataFrame([
        ("交易日历/基础资料", "Streamlit内存", "72小时"),
        ("行业成员", "Streamlit内存", "7天"),
        ("逐股票前复权行情", "应用临时磁盘", "不主动过期"),
        ("V6.13逐股票事件检查点", "应用临时磁盘", "不主动过期"),
    ], columns=["对象", "位置", "设定时长"])
    detail_columns = [
        "Strategy_Key", "Strategy_Label", "ts_code", "name",
        "Signal_Date", "Signal_Week", "Entry_Date", "Entry_Raw_Open",
        "Weekly_SKDJ_N", "SKDJ_M", "Weekly_Data_Mode",
        "Entry_Trigger_Mode", "Setup_Weekly_Date", "Signal_K", "Signal_D",
        "Signal_KD_Spread", "Signal_K_Change_1W",
        "Weekly_K_Approach_Band", "Daily_MACD_Red_Age",
        "Daily_MACD_State", "Daily_MACD_Hist",
        "Daily_MACD_Remaining_pct", "Daily_MACD_Retention_pct",
        "Signal_Rally_From_Red_Start_pct", "Signal_Daily_SKDJ_N6_K",
        "Signal_Daily_SKDJ_N6_D", "Signal_Daily_SKDJ_N9_K",
        "Signal_Daily_SKDJ_N9_D", "Entry_Close_MACD_Red_Age_Audit",
        "Entry_Close_MACD_State_Audit",
        "Entry_Close_Daily_SKDJ_N6_K_Audit",
        "Entry_Close_Daily_SKDJ_N6_D_Audit",
        "Entry_Close_Daily_SKDJ_N9_K_Audit",
        "Entry_Close_Daily_SKDJ_N9_D_Audit", "Timing_Score_221",
        "Entry_End_Date_40D", "Entry_MFE_Net_pct", "Entry_MAE_Raw_pct",
        "Entry_Close_Return_Net_pct", "Explosion_Class_40D",
        "SW_L1", "SW_L2", "Circ_MV_Billion", "Raw_Close",
    ]
    history_export = mature[[
        column for column in detail_columns if column in mature.columns]].copy()
    recent_export = recent_candidates[[
        column for column in detail_columns
        if column in recent_candidates.columns]].copy()
    watch_columns = [
        column for column in detail_columns
        if column in live_watch.columns and not column.startswith("Entry_")]
    live_watch_export = live_watch[watch_columns].copy()
    files = {
        "01_run_summary_v6_13.csv": run_summary,
        "02_experiment_definitions_v6_13.csv": definitions,
        "03_skdj_parameter_calibration_rank_v6_13.csv": calibration_summary,
        "04_skdj_parameter_calibration_detail_v6_13.csv": calibration_detail,
        "05_n6_n9_calibration_comparison_v6_13.csv": n6n9_calibration,
        "06_entry_event_summary_v6_13.csv": event_summary,
        "07_weekly_k_band_outcomes_v6_13.csv": band_summary,
        "08_red2_vs_old3_matched_summary_v6_13.csv": matched_summary,
        "09_red2_vs_old3_matched_detail_v6_13.csv": matched_detail,
        "10_weekly_coverage_summary_v6_13.csv": coverage_summary,
        "11_weekly_signal_calendar_v6_13.csv": coverage_calendar,
        "12_full_slot_portfolio_summary_v6_13.csv": portfolio_summary,
        "13_all_portfolio_paths_v6_13.csv": portfolio_paths,
        "14_rule_path_trades_v6_13.csv": deterministic_trades,
        "15_rule_path_equity_v6_13.csv": deterministic_equity,
        "16_historical_mature_event_detail_v6_13.csv": history_export,
        "17_recent_30d_candidates_v6_13.csv": recent_export,
        "18_latest_dynamic_weekly_watch_pool_v6_13.csv": live_watch_export,
        "19_rejection_audit_v6_13.csv": rejection_audit,
        "20_cache_policy_v6_13.csv": cache_policy,
        "21_api_errors_v6_13.csv": pd.DataFrame({"错误": API_ERRORS}),
    }
    result_zip = make_zip(files)
    try:
        atomic_bytes(result_zip, result_path)
        v613_clear_job_active(request_signature)
        persisted = True
    except Exception as exc:
        persisted = False
        st.warning(f"结果未能持久保存，但当前页面仍可下载：{exc}")

    st.success(
        f"完成：校准最佳参数N={best_calibrated.get('N', 'NA')}、"
        f"M={best_calibrated.get('M', 'NA')}；"
        f"N9/M3校准{'通过' if n9_confirmed else '未通过或证据不足'}。"
        f"成熟事件{len(mature)}个，组合路径{len(portfolio_paths)}条，"
        f"结果{'已保存' if persisted else '仅当前页面可下载'}。")
    st.subheader("结论一：SKDJ隐藏参数反推")
    render_plain_table(calibration_top, 30)
    st.caption("先看最佳参数和N9/M3、N6/M3的误差；只有四个点同时吻合才算对齐。")
    st.subheader("结论二：三套买点的全量事件质量")
    render_plain_table(event_summary, 20)
    st.caption("红2方案包含后来未达到上涨3%的失败周期，这是实盘可见的完整机会集。")
    st.subheader("结论三：真实全额三仓")
    render_plain_table(portfolio_summary, 40)
    st.caption("规则排序各1条真实路径；随机组显示中位及P10/P90，便于判断Top3排序是否有效。")
    st.subheader("结论四：匹配旧3%信号后的提前天数与价格优势")
    render_plain_table(matched_summary, 20)
    st.caption("这里只回答同一股票同一红柱周期提前买便宜多少，不能代替红2全样本F比例。")
    st.subheader("结论五：周线K接近25的分段结果")
    render_plain_table(band_summary, 30)
    st.subheader("结论六：信号覆盖")
    render_plain_table(coverage_summary, 20)
    st.subheader("最近30日候选")
    render_plain_table(recent_export.sort_values(
        [column for column in ["Signal_Date", "Strategy_Key", "Timing_Score_221"]
         if column in recent_export.columns],
        ascending=[False, True, False][:
            len([column for column in ["Signal_Date", "Strategy_Key", "Timing_Score_221"]
                 if column in recent_export.columns])]), 300)
    st.subheader(f"最新交易日动态周线观察池：{latest_market_date}")
    render_plain_table(live_watch_export, 300)
    st.subheader("运行摘要")
    render_plain_table(run_summary, 10)
    st.caption(f"结果ZIP共{len(files)}个CSV；参数校准、事件质量和三仓路径分开导出。")
    render_download(
        result_zip, result_name, f"v613_current_{request_signature}")


# ---------------------------------------------------------------------------
# V6.14 daily N6/red2 Top5 lifecycle monitor.
#
# The V6.13 calibration and portfolio code above remains frozen for audit.  The
# new entry point below reuses its observable partial-week calculation, cache,
# universe and transaction-cost infrastructure, but deliberately removes the
# three-slot capital path and fixed-40-day liquidation from the primary test.
# ---------------------------------------------------------------------------

V614_TITLE = "周线SKDJ N6＋日线MACD红2 Top5生命周期回测 V6.14"
V614_VERSION = "V6.14-N6-RED2-DAILY-TOP5-LIFECYCLE"
V614_UI_PATCH = "V6.14-PARTIAL-WEEK-N6-RED123-SHADOW-HARD5-TRAIL15-AHALF"
V614_RESULT_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_14_results")
V614_JOB_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_14_jobs")
V614_CHECKPOINT_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_14_checkpoints")
V614_EVENT_ENGINE_VERSION = "V6.14-N6-PARTIAL-WEEK-RED123-EVENTS-R1"
V614_PRIMARY_N = 6
V614_M = 3
V614_TOP_N = 5
V614_MAIN_RED_AGE = 2
V614_SHADOW_RED_AGES = (1, 2, 3)
V614_K_MIN = 10.0
V614_K_MAX = 25.0
V614_HARD_STOP_PCT = 5.0
V614_TRAIL_DRAWDOWN_PCT = 15.0
V614_PROFIT_FLOOR_ACTIVATION_PCT = 20.0
V614_PROFIT_KEEP_RATIO = 0.50
V614_EXIT_MODES = (
    ("Hard5Trail15", "-5%硬止损＋截至上一日最高价回撤15%"),
    (
        "Hard5Trail15AHalf",
        "-5%硬止损＋回撤15%＋收盘达到20%后保护一半最高收盘浮盈",
    ),
)


def v614_checkpoint_path(signature: str, ts_code: str) -> str:
    return os.path.join(
        V614_CHECKPOINT_DIR, signature,
        f"{str(ts_code).replace('.', '_')}.pkl")


def v614_load_checkpoint(
        signature: str, ts_code: str) -> dict[str, Any] | None:
    path = v614_checkpoint_path(signature, ts_code)
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
        record_error(f"V6.14检查点损坏 {ts_code}: {exc}")
        return None


def v614_save_checkpoint(
        signature: str, ts_code: str, events: list[dict[str, Any]],
        rejects: dict[str, int]) -> None:
    atomic_pickle({
        "signature": signature, "ts_code": str(ts_code),
        "events": events, "rejects": rejects,
    }, v614_checkpoint_path(signature, ts_code))


def v614_active_job_path(signature: str) -> str:
    return os.path.join(V614_JOB_DIR, f"{signature}.active")


def v614_mark_job_active(signature: str) -> None:
    atomic_bytes(json.dumps({
        "signature": signature, "version": V614_VERSION,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }, ensure_ascii=False).encode("utf-8"), v614_active_job_path(signature))


def v614_clear_job_active(signature: str) -> None:
    path = v614_active_job_path(signature)
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        record_error(f"V6.14任务标记清除失败: {exc}")


def v614_is_job_active(signature: str) -> bool:
    return os.path.exists(v614_active_job_path(signature))


def v614_week_bar_state(
        signal_date: str,
        week_last_map: dict[pd.Timestamp, str]) -> tuple[str, str]:
    period = pd.Timestamp(signal_date).to_period("W-FRI")
    week_label = period.end_time.normalize()
    week_last = str(week_last_map.get(week_label, ""))
    state = "本周已完成" if week_last == signal_date else "本周未完成"
    return state, week_last


def v614_analyze_stock(
        stock: pd.Series, periods: list[dict[str, str]], daily: pd.DataFrame,
        cached_basic: pd.DataFrame, storage_path: str,
        week_last_map: dict[pd.Timestamp, str], open_dates: list[str],
        open_pos: dict[str, int], config: dict[str, Any], use_cache: bool,
        api_pause: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Create N6 partial-week red1/red2/red3 events using only as-of data."""
    rejects: dict[str, int] = {}
    weekly_base = aggregate_complete_weekly(daily, week_last_map)
    if weekly_base.empty:
        return [], rejects
    frame = v613_daily_indicator_frame(daily, weekly_base)
    dates = frame["trade_date"].astype(str)
    event_end = str(config.get("event_signal_end", config["market_end"]))
    formal = dates.between(config["signal_start"], event_end)
    positive = true_mask(frame, "Daily_MACD_Positive")
    dynamic_k = numeric(frame, "Dynamic_N6_Weekly_K")
    dynamic_change = numeric(frame, "Dynamic_N6_Weekly_K_Change")
    base_mask = (
        formal & positive
        & dynamic_k.between(V614_K_MIN, V614_K_MAX, inclusive="both")
        & dynamic_change.gt(0)
    )
    selected_by_age: dict[int, pd.DataFrame] = {}
    for red_age in V614_SHADOW_RED_AGES:
        selected = frame[
            base_mask
            & numeric(frame, "Daily_MACD_Red_Age").eq(red_age)
        ].copy().sort_values("trade_date")
        if not selected.empty:
            selected = selected.groupby(
                "Daily_MACD_Cycle", as_index=False, sort=False).first()
        selected_by_age[int(red_age)] = selected

    latest_date = str(config.get("latest_market_date", config["market_end"]))
    latest_watch = frame[
        dates.eq(latest_date)
        & dynamic_k.between(V614_K_MIN, V614_K_MAX, inclusive="both")
        & dynamic_change.gt(0)
    ].copy().tail(1)
    estimated = sum(len(value) for value in selected_by_age.values()) + len(
        latest_watch)
    if estimated == 0:
        return [], rejects

    code = str(stock["ts_code"])
    basic = ensure_daily_basic(
        code, config["data_start"], config["market_end"], daily,
        cached_basic, storage_path, use_cache, api_pause)
    if basic.empty:
        rejects["存在V6.14信号但daily_basic缺失"] = estimated
        return [], rejects

    rows: list[dict[str, Any]] = []
    outcome_cache: dict[str, dict[str, Any]] = {}
    frame_period = pd.to_datetime(
        frame["trade_date"], format="%Y%m%d").dt.to_period("W-FRI")

    def append_row(
            signal: pd.Series, red_age: int,
            is_watch: bool = False) -> None:
        signal_date = str(signal["trade_date"])
        membership = membership_on_date(periods, signal_date)
        snapshot = market_snapshot(basic, signal_date)
        reason = ""
        if membership is None:
            reason = "信号日不在历史科技池"
        elif not (str(stock["list_date"]) <= signal_date
                  < str(stock["delist_date"])):
            reason = "信号日上市状态无效"
        elif (not math.isfinite(snapshot["Raw_Close"])
              or snapshot["Raw_Close"] < config["min_price"]):
            reason = "信号日股价不足"
        elif (not math.isfinite(snapshot["Circ_MV_Billion"])
              or snapshot["Circ_MV_Billion"] < config["min_mv"]):
            reason = "信号日流通市值不足"
        if reason or membership is None:
            rejects[reason] = rejects.get(reason, 0) + 1
            return

        signal_k = finite_num(signal.get("Dynamic_N6_Weekly_K"))
        signal_d = finite_num(signal.get("Dynamic_N6_Weekly_D"))
        signal_change = finite_num(
            signal.get("Dynamic_N6_Weekly_K_Change"))
        legacy_k = int(
            math.isfinite(signal_k) and 15.0 <= signal_k <= 20.0)
        legacy_age = int(3 <= int(red_age) <= 5)
        legacy_kd = int(
            math.isfinite(signal_k) and math.isfinite(signal_d)
            and signal_k < signal_d)
        week_state, week_last = v614_week_bar_state(
            signal_date, week_last_map)
        signal_period = pd.Timestamp(signal_date).to_period("W-FRI")
        final_week = frame[frame_period.eq(signal_period)].sort_values(
            "trade_date")
        completed_available = bool(
            week_last and week_last <= str(config["latest_market_date"])
            and not final_week.empty
            and str(final_week.iloc[-1]["trade_date"]) == week_last)
        if completed_available:
            final_row = final_week.iloc[-1]
            final_k = finite_num(final_row.get("Dynamic_N6_Weekly_K"))
            final_change = finite_num(
                final_row.get("Dynamic_N6_Weekly_K_Change"))
            later_confirmed: Any = bool(
                math.isfinite(final_k)
                and V614_K_MIN <= final_k <= V614_K_MAX
                and math.isfinite(final_change) and final_change > 0)
        else:
            final_k = np.nan
            final_change = np.nan
            later_confirmed = "尚未完成"
        row: dict[str, Any] = {
            "Event_Type": "LIVE_WATCH" if is_watch else "ENTRY_SIGNAL",
            "Strategy_Key": (
                "N6_LIVE_WATCH" if is_watch else f"N6_DYNAMIC_RED{red_age}"),
            "Strategy_Label": (
                "N6动态周线观察池" if is_watch
                else f"N6动态周线＋日线MACD红柱第{red_age}日"),
            "Entry_Red_Age_Hypothesis": int(red_age),
            "Weekly_SKDJ_N": V614_PRIMARY_N, "SKDJ_M": V614_M,
            "Weekly_Data_Mode": "截至当日动态周线",
            "Weekly_Bar_State": week_state,
            "Week_Last_Trading_Date": week_last,
            "Later_Week_Close_Confirmed": later_confirmed,
            "Later_Week_Close_N6_K": final_k,
            "Later_Week_Close_N6_K_Change": final_change,
            "Entry_Trigger_Mode": (
                "仅观察" if is_watch else f"MACD红柱第{red_age}日"),
            "ts_code": code, "name": str(stock["name"]),
            "Signal_Date": signal_date,
            "Signal_Weekday": pd.Timestamp(signal_date).day_name(),
            "Signal_Week": str(signal_period),
            "SW_L1": membership["l1"], "SW_L2": membership["l2"],
            "SW_L3": membership["l3"], **snapshot,
            "Setup_Weekly_Date": str(
                signal.get("Dynamic_N6_Prior_Weekly_Date", "")),
            "Signal_K": signal_k, "Signal_D": signal_d,
            "Signal_KD_Spread": signal_k - signal_d,
            "Signal_K_Change_1W": signal_change,
            "Weekly_K_Approach_Band": v613_k_band(signal_k),
            **macd_snapshot(signal),
            "Signal_Rally_From_Red_Start_pct": finite_num(
                signal.get("Daily_Return_Since_Red_Start_pct")),
            "Signal_Daily_SKDJ_N6_K": finite_num(
                signal.get("Daily_SKDJ_N6_K")),
            "Signal_Daily_SKDJ_N6_D": finite_num(
                signal.get("Daily_SKDJ_N6_D")),
            "Signal_Daily_SKDJ_N9_K": finite_num(
                signal.get("Daily_SKDJ_N9_K")),
            "Signal_Daily_SKDJ_N9_D": finite_num(
                signal.get("Daily_SKDJ_N9_D")),
            "Legacy_Timing_Score_221": (
                V63_SCORE_K_WEIGHT * legacy_k
                + V63_SCORE_AGE_WEIGHT * legacy_age
                + V63_SCORE_KD_WEIGHT * legacy_kd),
        }
        if not is_watch:
            if signal_date not in outcome_cache:
                outcome_cache[signal_date] = daily_timing_outcomes(
                    frame, signal_date, code, open_dates, open_pos, config)
            outcome = outcome_cache[signal_date]
            row.update({f"Entry_{key}": value for key, value in outcome.items()})
        rows.append(row)

    for red_age, selected in selected_by_age.items():
        for _, signal in selected.iterrows():
            append_row(signal, red_age, False)
    for _, signal in latest_watch.iterrows():
        append_row(signal, V614_MAIN_RED_AGE, True)
    return rows, rejects


def v614_rank_daily_events(events: pd.DataFrame) -> pd.DataFrame:
    """Transparent red-entry ranking; every component is visible at signal close."""
    if events.empty:
        return events.copy()
    work = events.copy()
    work["V614_K_Band_Score"] = np.select(
        [
            numeric(work, "Signal_K").between(10.0, 20.0, inclusive="both"),
            numeric(work, "Signal_K").gt(20.0)
            & numeric(work, "Signal_K").le(25.0),
        ], [40.0, 20.0], default=0.0)
    group_keys = ["Entry_Red_Age_Hypothesis", "Signal_Date"]
    work["V614_K_Change_RankPct"] = work.groupby(
        group_keys, dropna=False)["Signal_K_Change_1W"].rank(
            pct=True, method="average", na_option="bottom")
    work["V614_MACD_Retention_RankPct"] = work.groupby(
        group_keys, dropna=False)["Daily_MACD_Retention_pct"].rank(
            pct=True, method="average", na_option="bottom")
    work["V614_K_Change_Score"] = (
        numeric(work, "V614_K_Change_RankPct").fillna(0.0) * 30.0)
    work["V614_MACD_Retention_Score"] = (
        numeric(work, "V614_MACD_Retention_RankPct").fillna(0.0) * 30.0)
    work["V614_Rank_Score_100"] = (
        numeric(work, "V614_K_Band_Score")
        + numeric(work, "V614_K_Change_Score")
        + numeric(work, "V614_MACD_Retention_Score"))
    work = work.sort_values(
        ["Entry_Red_Age_Hypothesis", "Signal_Date",
         "V614_Rank_Score_100", "Signal_K", "ts_code"],
        ascending=[True, True, False, True, True]).reset_index(drop=True)
    work["Daily_Rank"] = work.groupby(
        group_keys, dropna=False).cumcount() + 1
    work["Selected_Top5"] = numeric(work, "Daily_Rank").le(V614_TOP_N)
    work["Pick_ID"] = [
        f"R{int(age)}:{code}:{signal_date}"
        for age, code, signal_date in zip(
            numeric(work, "Entry_Red_Age_Hypothesis").fillna(0),
            work["ts_code"].astype(str), work["Signal_Date"].astype(str))
    ]
    return work


def v614_selection_calendar(
        ranked: pd.DataFrame, open_dates: list[str], start_date: str,
        end_date: str) -> pd.DataFrame:
    days = [value for value in open_dates if start_date <= value <= end_date]
    calendar = pd.DataFrame({"trade_date": days})
    for red_age in V614_SHADOW_RED_AGES:
        group = ranked[
            numeric(ranked, "Entry_Red_Age_Hypothesis").eq(red_age)]
        raw_counts = group.groupby("Signal_Date").size()
        selected_counts = group[true_mask(
            group, "Selected_Top5")].groupby("Signal_Date").size()
        calendar[f"红{red_age}合格候选"] = calendar["trade_date"].map(
            raw_counts).fillna(0).astype(int)
        calendar[f"红{red_age}模拟买入"] = calendar["trade_date"].map(
            selected_counts).fillna(0).astype(int)
    calendar["主方案是否空窗"] = calendar["红2模拟买入"].eq(0)
    calendar["Signal_Week"] = pd.to_datetime(
        calendar["trade_date"], format="%Y%m%d").dt.to_period(
            "W-FRI").astype(str)
    return calendar


def v614_dynamic_grade(mfe_net_pct: Any, is_closed: bool) -> str:
    mfe = finite_num(mfe_net_pct)
    if math.isfinite(mfe) and mfe >= 30.0:
        return "S"
    if math.isfinite(mfe) and mfe >= 20.0:
        return "A"
    if math.isfinite(mfe) and mfe >= 10.0:
        return "B"
    return "F" if is_closed else "未定级"


def v614_simulate_pick(
        event: dict[str, Any], book: dict[str, Any],
        open_dates: list[str], config: dict[str, Any],
        exit_mode: str) -> dict[str, Any]:
    """Follow one Top5 pick without capital constraints or a fixed horizon."""
    signal_date = normalize_date(event.get("Signal_Date"))
    latest_date = normalize_date(config.get("latest_market_date"))
    entry_date = normalize_date(event.get("Entry_Date"))
    entry_reason = str(event.get("Entry_Reason", ""))
    base = {
        **event,
        "Exit_Mode_Key": exit_mode,
        "Exit_Mode": dict(V614_EXIT_MODES).get(exit_mode, exit_mode),
        "Entry_Status_Reason": entry_reason,
        "Entry_Execution_Date": entry_date,
        "Entry_Raw_Open": np.nan,
        "Entry_Trade_Price": np.nan,
        "Lifecycle_Status": "",
        "Decision_Date": "", "Exit_Date": "", "Exit_Reason": "",
        "Exit_Raw_Open": np.nan, "Exit_Trade_Price": np.nan,
        "Realized_Return_Net_pct": np.nan,
        "Current_Return_Net_pct": np.nan,
        "Current_or_Exit_Return_Net_pct": np.nan,
        "Hold_Market_Days": np.nan, "Exit_Week": np.nan,
        "Highest_High_Through_Status": np.nan,
        "Highest_Close_Through_Status": np.nan,
        "MFE_Net_pct": np.nan, "MAE_Raw_pct": np.nan,
        "Decision_Hard_Stop_Line": np.nan,
        "Decision_Trail15_Line": np.nan,
        "Decision_AHalf_Profit_Line": np.nan,
        "Decision_Effective_Line": np.nan,
        "Next_Day_Hard_Stop_Line": np.nan,
        "Next_Day_Trail15_Line": np.nan,
        "Next_Day_AHalf_Profit_Line": np.nan,
        "Next_Day_Effective_Line": np.nan,
        "Dynamic_Grade": "未买入",
    }
    if not entry_date:
        next_market = ""
        if signal_date in open_dates:
            signal_position = open_dates.index(signal_date)
            if signal_position + 1 < len(open_dates):
                next_market = str(open_dates[signal_position + 1])
        if not next_market or next_market > latest_date:
            base["Lifecycle_Status"] = "待次日开盘买入"
        else:
            base["Lifecycle_Status"] = "未成交"
        return base
    if entry_date > latest_date:
        base["Lifecycle_Status"] = "待次日开盘买入"
        return base

    raw_entry = finite_num(book.get("open", {}).get(entry_date))
    if not math.isfinite(raw_entry) or raw_entry <= 0:
        base["Lifecycle_Status"] = "未成交"
        base["Entry_Status_Reason"] = (
            entry_reason or "模拟成交日开盘价缺失")
        return base
    buy_factor, sell_factor = _cost_factors(config)
    buy_slippage = float(config.get("buy_slippage_pct", 0.0)) / 100.0
    sell_slippage = float(config.get("sell_slippage_pct", 0.0)) / 100.0
    entry_trade_price = raw_entry * (1.0 + buy_slippage)
    net_entry = raw_entry * buy_factor
    hard_line = entry_trade_price * (1.0 - V614_HARD_STOP_PCT / 100.0)
    base.update({
        "Entry_Raw_Open": raw_entry,
        "Entry_Trade_Price": entry_trade_price,
        "Next_Day_Hard_Stop_Line": hard_line,
    })

    stock_dates = [
        str(value) for value in book.get("dates", [])
        if entry_date <= str(value) <= latest_date]
    if not stock_dates:
        base["Lifecycle_Status"] = "未成交"
        base["Entry_Status_Reason"] = "成交后无个股行情"
        return base
    open_pos = {value: position for position, value in enumerate(open_dates)}
    prior_high = entry_trade_price
    prior_max_close = entry_trade_price
    observed_high = entry_trade_price
    observed_low = entry_trade_price
    last_close = np.nan
    last_date = entry_date
    decision_date = ""
    decision_reason = ""
    pending_exit = False
    exit_date = ""
    exit_raw_open = np.nan
    decision_lines: dict[str, float] = {}

    for trade_date in stock_dates:
        close = finite_num(book.get("close", {}).get(trade_date))
        high = finite_num(book.get("high", {}).get(trade_date))
        low = finite_num(book.get("low", {}).get(trade_date))
        if not math.isfinite(close) or close <= 0:
            continue
        trail_line = prior_high * (
            1.0 - V614_TRAIL_DRAWDOWN_PCT / 100.0)
        profit_floor = np.nan
        if (exit_mode == "Hard5Trail15AHalf"
                and prior_max_close >= entry_trade_price * (
                    1.0 + V614_PROFIT_FLOOR_ACTIVATION_PCT / 100.0)):
            profit_floor = (
                entry_trade_price
                + V614_PROFIT_KEEP_RATIO
                * (prior_max_close - entry_trade_price))
        effective_candidates = [hard_line, trail_line]
        if math.isfinite(profit_floor):
            effective_candidates.append(profit_floor)
        effective_line = max(effective_candidates)
        triggered = close <= effective_line
        if triggered:
            labels = []
            tolerance = max(abs(effective_line), 1.0) * 1e-10
            if abs(hard_line - effective_line) <= tolerance:
                labels.append("收盘跌破买入价5%硬止损")
            if abs(trail_line - effective_line) <= tolerance:
                labels.append("收盘跌破截至上一日最高价回撤15%线")
            if (math.isfinite(profit_floor)
                    and abs(profit_floor - effective_line) <= tolerance):
                labels.append("收盘跌破A类半浮盈保护线")
            decision_date = trade_date
            decision_reason = "+".join(labels) or "收盘跌破有效退出线"
            decision_lines = {
                "Decision_Hard_Stop_Line": hard_line,
                "Decision_Trail15_Line": trail_line,
                "Decision_AHalf_Profit_Line": profit_floor,
                "Decision_Effective_Line": effective_line,
            }

        # Today's high and close become references only for the next trading
        # day.  They never raise the line used to judge today's close.
        if math.isfinite(high) and high > 0:
            prior_high = max(prior_high, high)
            observed_high = max(observed_high, high)
        prior_max_close = max(prior_max_close, close)
        if math.isfinite(low) and low > 0:
            observed_low = min(observed_low, low)
        last_close = close
        last_date = trade_date

        if triggered:
            next_date = v613_next_stock_date(book, trade_date)
            if next_date and next_date <= latest_date:
                candidate_open = finite_num(book.get("open", {}).get(next_date))
                if math.isfinite(candidate_open) and candidate_open > 0:
                    exit_date = next_date
                    exit_raw_open = candidate_open
                else:
                    pending_exit = True
            else:
                pending_exit = True
            break

    next_trail = prior_high * (
        1.0 - V614_TRAIL_DRAWDOWN_PCT / 100.0)
    next_floor = np.nan
    if (exit_mode == "Hard5Trail15AHalf"
            and prior_max_close >= entry_trade_price * (
                1.0 + V614_PROFIT_FLOOR_ACTIVATION_PCT / 100.0)):
        next_floor = (
            entry_trade_price
            + V614_PROFIT_KEEP_RATIO
            * (prior_max_close - entry_trade_price))
    next_effective = max(
        [hard_line, next_trail]
        + ([next_floor] if math.isfinite(next_floor) else []))
    mfe_net = (observed_high * sell_factor / net_entry - 1.0) * 100.0
    mae_raw = (observed_low / entry_trade_price - 1.0) * 100.0
    base.update({
        **decision_lines,
        "Decision_Date": decision_date,
        "Exit_Reason": decision_reason,
        "Highest_High_Through_Status": prior_high,
        "Highest_Close_Through_Status": prior_max_close,
        "MFE_Net_pct": mfe_net,
        "MAE_Raw_pct": mae_raw,
        "Next_Day_Hard_Stop_Line": hard_line,
        "Next_Day_Trail15_Line": next_trail,
        "Next_Day_AHalf_Profit_Line": next_floor,
        "Next_Day_Effective_Line": next_effective,
    })
    is_closed = bool(exit_date)
    if is_closed:
        exit_trade_price = exit_raw_open * (1.0 - sell_slippage)
        realized = (
            exit_raw_open * sell_factor / net_entry - 1.0) * 100.0
        hold_days = (
            open_pos.get(exit_date, -1) - open_pos.get(entry_date, -1) + 1
            if entry_date in open_pos and exit_date in open_pos else np.nan)
        if "硬止损" in decision_reason:
            status = "已止损"
        elif "半浮盈" in decision_reason:
            status = "已利润保护退出"
        else:
            status = "已移动保护退出"
        base.update({
            "Lifecycle_Status": status,
            "Exit_Date": exit_date,
            "Exit_Raw_Open": exit_raw_open,
            "Exit_Trade_Price": exit_trade_price,
            "Realized_Return_Net_pct": realized,
            "Current_or_Exit_Return_Net_pct": realized,
            "Hold_Market_Days": hold_days,
            "Exit_Week": (
                math.ceil(float(hold_days) / 5.0)
                if math.isfinite(finite_num(hold_days)) else np.nan),
            "Dynamic_Grade": v614_dynamic_grade(mfe_net, True),
        })
    else:
        current_return = (
            (last_close * sell_factor / net_entry - 1.0) * 100.0
            if math.isfinite(last_close) and last_close > 0 else np.nan)
        hold_days = (
            open_pos.get(last_date, -1) - open_pos.get(entry_date, -1) + 1
            if entry_date in open_pos and last_date in open_pos else np.nan)
        base.update({
            "Lifecycle_Status": (
                "待次日开盘卖出" if pending_exit else "持有"),
            "Current_Return_Net_pct": current_return,
            "Current_or_Exit_Return_Net_pct": current_return,
            "Hold_Market_Days": hold_days,
            "Exit_Week": (
                math.ceil(float(hold_days) / 5.0)
                if math.isfinite(finite_num(hold_days)) else np.nan),
            "Dynamic_Grade": v614_dynamic_grade(mfe_net, False),
        })
    return base


def v614_run_lifecycles(
        selected: pd.DataFrame,
        price_book: dict[str, dict[str, Any]], open_dates: list[str],
        config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if selected.empty:
        return pd.DataFrame()
    for event in selected.to_dict("records"):
        code = str(event.get("ts_code", ""))
        book = price_book.get(code, {})
        for exit_key, _ in V614_EXIT_MODES:
            rows.append(v614_simulate_pick(
                event, book, open_dates, config, exit_key))
    return pd.DataFrame(rows)


def v614_lifecycle_summary(lifecycle: pd.DataFrame) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    cohorts = (("第1名", 1), ("前3名", 3), ("前5名", 5))
    for exit_key, exit_group in lifecycle.groupby("Exit_Mode_Key"):
        for red_age, age_group in exit_group.groupby(
                "Entry_Red_Age_Hypothesis"):
            for cohort, max_rank in cohorts:
                group = age_group[numeric(
                    age_group, "Daily_Rank").le(max_rank)]
                if group.empty:
                    continue
                status = group["Lifecycle_Status"].astype(str)
                closed = status.str.startswith("已")
                graded = group["Dynamic_Grade"].astype(str)
                realized = numeric(
                    group.loc[closed], "Realized_Return_Net_pct")
                rows.append({
                    "Exit_Mode_Key": exit_key,
                    "退出方案": dict(V614_EXIT_MODES).get(exit_key, exit_key),
                    "红柱买点": f"红{int(red_age)}",
                    "排名组": cohort,
                    "模拟入选": len(group),
                    "已成交": int(~status.isin([
                        "待次日开盘买入", "未成交"]).sum()),
                    "仍持有": int(status.eq("持有").sum()),
                    "待卖出": int(status.eq("待次日开盘卖出").sum()),
                    "已退出": int(closed.sum()),
                    "退出交易盈利比例%": (
                        realized.gt(0).mean() * 100.0
                        if len(realized) else np.nan),
                    "退出收益均值%": realized.mean(),
                    "退出收益中位%": realized.median(),
                    "S或A比例%": graded.isin(["S", "A"]).mean() * 100.0,
                    "F级比例_全部入选%": graded.eq("F").mean() * 100.0,
                    "未定级比例%": graded.eq("未定级").mean() * 100.0,
                    "持有日中位": numeric(
                        group, "Hold_Market_Days").median(),
                    "最大浮盈中位%": numeric(
                        group, "MFE_Net_pct").median(),
                    "最大浮亏中位%": numeric(
                        group, "MAE_Raw_pct").median(),
                })
    return pd.DataFrame(rows)


def v614_paired_exit_audit(
        lifecycle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lifecycle.empty:
        return pd.DataFrame(), pd.DataFrame()
    identifiers = [
        "Pick_ID", "ts_code", "name", "Signal_Date",
        "Entry_Red_Age_Hypothesis", "Daily_Rank"]
    metrics = [
        "Lifecycle_Status", "Decision_Date", "Exit_Date", "Exit_Reason",
        "Realized_Return_Net_pct", "Current_or_Exit_Return_Net_pct",
        "Hold_Market_Days", "Exit_Week", "Dynamic_Grade",
        "MFE_Net_pct", "MAE_Raw_pct"]
    base = lifecycle[lifecycle["Exit_Mode_Key"].eq(
        "Hard5Trail15")][identifiers + metrics].copy()
    protected = lifecycle[lifecycle["Exit_Mode_Key"].eq(
        "Hard5Trail15AHalf")][identifiers + metrics].copy()
    detail = base.merge(
        protected, on=identifiers, how="outer",
        suffixes=("_基础15", "_A半浮盈"))
    detail["A保护减基础_当前或退出收益百分点"] = (
        numeric(detail, "Current_or_Exit_Return_Net_pct_A半浮盈")
        - numeric(detail, "Current_or_Exit_Return_Net_pct_基础15"))
    detail["A保护更早退出"] = (
        detail["Exit_Date_A半浮盈"].map(normalize_date).str.len().eq(8)
        & (
            ~detail["Exit_Date_基础15"].map(normalize_date).str.len().eq(8)
            | detail["Exit_Date_A半浮盈"].astype(str).lt(
                detail["Exit_Date_基础15"].astype(str))))
    rows: list[dict[str, Any]] = []
    for red_age, group in detail.groupby(
            "Entry_Red_Age_Hypothesis", dropna=False):
        paired_closed = group[
            group["Exit_Date_基础15"].map(normalize_date).str.len().eq(8)
            & group["Exit_Date_A半浮盈"].map(normalize_date).str.len().eq(8)]
        delta = numeric(
            paired_closed, "A保护减基础_当前或退出收益百分点")
        rows.append({
            "红柱买点": f"红{int(red_age)}",
            "全部配对事件": len(group),
            "两方案均已退出": len(paired_closed),
            "A保护更早退出": int(true_mask(group, "A保护更早退出").sum()),
            "A保护收益更高比例%": (
                delta.gt(0).mean() * 100.0 if len(delta) else np.nan),
            "A保护减基础收益均值百分点": delta.mean(),
            "A保护减基础收益中位百分点": delta.median(),
        })
    return pd.DataFrame(rows), detail


def v614_exit_week_summary(lifecycle: pd.DataFrame) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    closed = lifecycle[
        lifecycle["Lifecycle_Status"].astype(str).str.startswith("已")].copy()
    if closed.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in closed.groupby(
            ["Exit_Mode_Key", "Entry_Red_Age_Hypothesis", "Exit_Week"],
            dropna=False):
        exit_key, red_age, week = keys
        returns = numeric(group, "Realized_Return_Net_pct")
        rows.append({
            "退出方案": dict(V614_EXIT_MODES).get(exit_key, exit_key),
            "红柱买点": f"红{int(red_age)}",
            "退出周": week, "事件": len(group),
            "盈利比例%": returns.gt(0).mean() * 100.0,
            "收益均值%": returns.mean(), "收益中位%": returns.median(),
            "硬止损退出": group["Lifecycle_Status"].eq("已止损").sum(),
            "15%移动退出": group["Lifecycle_Status"].eq(
                "已移动保护退出").sum(),
            "A半浮盈退出": group["Lifecycle_Status"].eq(
                "已利润保护退出").sum(),
        })
    return pd.DataFrame(rows)


def v614_top3_cohort_audit(lifecycle: pd.DataFrame) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    work = lifecycle[
        lifecycle["Exit_Mode_Key"].eq("Hard5Trail15AHalf")
        & numeric(lifecycle, "Entry_Red_Age_Hypothesis").eq(
            V614_MAIN_RED_AGE)
        & numeric(lifecycle, "Daily_Rank").le(3)
    ].copy()
    rows: list[dict[str, Any]] = []
    for signal_date, group in work.groupby("Signal_Date", sort=True):
        closed = group[
            group["Lifecycle_Status"].astype(str).str.startswith("已")]
        returns = numeric(closed, "Realized_Return_Net_pct")
        rows.append({
            "Signal_Date": signal_date,
            "当日前3实际数量": len(group),
            "已退出数量": len(closed),
            "仍持有或待处理": len(group) - len(closed),
            "已退出盈利数量": int(returns.gt(0).sum()),
            "完整三只且全部退出": bool(len(group) == 3 and len(closed) == 3),
            "完整三只至少两只盈利": bool(
                len(group) == 3 and len(closed) == 3
                and returns.gt(0).sum() >= 2),
            "三只等权退出收益均值%": (
                returns.mean() if len(group) == 3 and len(closed) == 3
                else np.nan),
        })
    return pd.DataFrame(rows)


def v614_candidate_distribution(calendar: pd.DataFrame) -> pd.DataFrame:
    if calendar.empty:
        return pd.DataFrame()
    counts = numeric(calendar, "红2合格候选").fillna(0)
    groups = (
        ("0只", counts.eq(0)),
        ("1只", counts.eq(1)),
        ("2至3只", counts.between(2, 3)),
        ("4至5只", counts.between(4, 5)),
        ("6至10只", counts.between(6, 10)),
        ("11至20只", counts.between(11, 20)),
        ("21至50只", counts.between(21, 50)),
        ("51只以上", counts.ge(51)),
    )
    return pd.DataFrame([{
        "当日红2候选数量组": label,
        "交易日数": int(mask.sum()),
        "交易日占比%": mask.mean() * 100.0,
        "该组候选总数": int(counts[mask].sum()),
    } for label, mask in groups])


def v614_partial_week_audit(lifecycle: pd.DataFrame) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    work = lifecycle[
        lifecycle["Exit_Mode_Key"].eq("Hard5Trail15AHalf")
        & numeric(lifecycle, "Entry_Red_Age_Hypothesis").eq(
            V614_MAIN_RED_AGE)
    ].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(
            ["Weekly_Bar_State", "Later_Week_Close_Confirmed"],
            dropna=False):
        state, confirmed = keys
        grades = group["Dynamic_Grade"].astype(str)
        values = numeric(group, "Current_or_Exit_Return_Net_pct")
        rows.append({
            "信号周线状态": state,
            "后来周收盘仍确认": confirmed,
            "Top5事件": len(group),
            "S或A比例%": grades.isin(["S", "A"]).mean() * 100.0,
            "F级比例%": grades.eq("F").mean() * 100.0,
            "当前或退出盈利比例%": values.gt(0).mean() * 100.0,
            "当前或退出收益中位%": values.median(),
        })
    return pd.DataFrame(rows)


def v614_main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=V614_TITLE, layout="wide")
    st.title(V614_TITLE)
    streamlit_version = str(getattr(st, "__version__", "unknown"))
    st.caption(f"{V614_UI_PATCH}｜Streamlit {streamlit_version}")
    with st.expander("V6.14执行口径", expanded=True):
        st.markdown(f"""
- **周线池**：每天收盘后，以当周截至当天的开、高、低、收重建未完成周K，使用改良SKDJ N={V614_PRIMARY_N}、M={V614_M}；动态K位于{V614_K_MIN:g}～{V614_K_MAX:g}且高于上一完整周K才进入准备池。周一至周四不使用后来数据，周五收盘自然成为完整周线。
- **日线买点**：主方案只认MACD第2根红柱；信号在当天收盘确认，下一市场交易日开盘模拟买入，因此最新信号可能显示“待次日开盘买入”。红1和红3只做同规则影子对照，不进入主列表。
- **每日Top5**：按当天可见的N6位置、N6上升速度和MACD红柱保持率组成透明100分观察排序；每天最多取5只，不足5只按实际数量，0只绝不补弱信号。每个Top5事件使用独立虚拟资金，不运行三仓、不受现金和持仓名额影响。
- **基础退出**：买入价-5%硬止损与“买入以来、截至上一交易日最高价回撤{V614_TRAIL_DRAWDOWN_PCT:g}%”取较高者；当天收盘跌破，下一可交易日开盘退出。
- **利润保护挑战**：最高收盘价达到买入价+{V614_PROFIT_FLOOR_ACTIVATION_PCT:g}%后，再加入“保护最高收盘浮盈的{V614_PROFIT_KEEP_RATIO * 100:g}%”；三条线永不下移，仍按收盘确认、下一开盘执行。
- **没有固定40日**：未触发退出就一直显示持有。S/A/B按持有期间最大净浮盈30%/20%/10%逐级记录；已经退出但未达到B记为F，尚在持有且未达到B记为未定级。
- **排序说明**：本版排名是可审计的红2专用工作假设，不把它冒充已经成熟的评分模型；输出第1名、前3名、前5名结果，用数据继续验收排序。
""")

    with st.sidebar:
        st.header("运行参数")
        backtest_days = st.number_input(
            "每日选股回测交易日数", 100, 1000, 500, 50,
            key="v614_days")
        min_price = st.number_input(
            "最低股价（元）", 10.0, 20.0, 10.0, 10.0,
            format="%.0f", key="v614_min_price")
        min_mv = st.number_input(
            "最低流通市值（亿元）", 50.0, 100.0, 50.0, 50.0,
            format="%.0f", key="v614_min_mv")
        market_end_date = st.date_input(
            "回测及每日选股截止（默认今天）", date.today(),
            key="v614_market_end")
        pause = st.number_input(
            "接口间隔(秒)", 0.0, 2.0, 0.12, 0.02,
            key="v614_pause")
        use_cache = st.checkbox(
            "复用行情和72小时基础缓存", True, key="v614_cache")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005,
            format="%.3f", key="v614_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01,
            format="%.3f", key="v614_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001,
            format="%.3f", key="v614_transfer")
        if st.button("清除V6.14结果和运行状态", key="v614_clear"):
            shutil.rmtree(V614_RESULT_DIR, ignore_errors=True)
            shutil.rmtree(V614_JOB_DIR, ignore_errors=True)
            st.success("结果和运行状态已清除；行情缓存与逐股票检查点保留。")

    request_payload = {
        "version": V614_VERSION, "days": int(backtest_days),
        "market_end": market_end_date.strftime("%Y%m%d"),
        "min_price": float(min_price), "min_mv": float(min_mv),
        "commission": float(commission_pct),
        "stamp": float(stamp_duty_pct),
        "transfer": float(transfer_fee_pct),
        "n": V614_PRIMARY_N, "m": V614_M,
        "k_band": [V614_K_MIN, V614_K_MAX],
        "red_ages": list(V614_SHADOW_RED_AGES),
        "main_red_age": V614_MAIN_RED_AGE, "top_n": V614_TOP_N,
        "hard_stop": V614_HARD_STOP_PCT,
        "trail_drawdown": V614_TRAIL_DRAWDOWN_PCT,
        "profit_activation": V614_PROFIT_FLOOR_ACTIVATION_PCT,
        "profit_keep": V614_PROFIT_KEEP_RATIO,
        "exit_modes": [value[0] for value in V614_EXIT_MODES],
    }
    request_signature = stable_signature(request_payload)
    result_path = os.path.join(
        V614_RESULT_DIR, f"{request_signature}.zip")
    result_name = (
        f"weekly_skdj_n6_red2_top5_lifecycle_v6_14_"
        f"{int(backtest_days)}d_p{int(min_price)}_mv{int(min_mv)}.zip")
    completed_available = False
    if os.path.exists(result_path):
        try:
            with open(result_path, "rb") as handle:
                saved_result = handle.read()
            completed_available = True
            v614_clear_job_active(request_signature)
            st.success("发现相同参数的V6.14已完成结果，可直接下载。")
            render_download(
                saved_result, result_name,
                f"v614_saved_{request_signature}")
        except Exception as exc:
            st.warning(f"已保存结果读取失败：{exc}")

    secret_token = configured_tushare_token()
    if secret_token:
        token = secret_token
        st.caption("已从Streamlit Secrets读取TUSHARE_TOKEN。")
    else:
        token = st.text_input(
            "Tushare Token", type="password", key="v614_token")
    job_active = v614_is_job_active(request_signature)
    left, right = st.columns(2)
    with left:
        start_clicked = st.button(
            "开始/重新运行V6.14", type="primary", key="v614_run")
    with right:
        stop_clicked = st.button(
            "停止自动续跑", disabled=not job_active, key="v614_stop")
    if stop_clicked:
        v614_clear_job_active(request_signature)
        st.success("已停止；逐股票检查点保留。")
        return
    if start_clicked:
        v614_mark_job_active(request_signature)
        job_active = True
    if not token:
        st.info("请输入Token；启动后页面重连会从逐股票检查点续跑。")
        return
    if not job_active:
        st.caption(
            "点击开始运行。" if not completed_available
            else "相同参数结果已可下载；如需覆盖请点击重新运行。")
        return

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    requested_end = market_end_date.strftime("%Y%m%d")
    try:
        probe_start = market_end_date - timedelta(
            days=int(backtest_days) * 2 + 120)
        probe_dates = load_trade_calendar(
            probe_start.strftime("%Y%m%d"), requested_end)
        signal_start = trailing_signal_start(
            probe_dates, requested_end, int(backtest_days))
    except Exception as exc:
        st.error(f"确定{int(backtest_days)}个交易日窗口失败：{exc}")
        return
    data_start = (
        pd.Timestamp(signal_start).date()
        - timedelta(weeks=WARMUP_WEEKS, days=7)
    ).strftime("%Y%m%d")
    try:
        with st.spinner("加载交易日历和历史科技池..."):
            open_dates = load_trade_calendar(data_start, requested_end)
            extended_end = (
                market_end_date + timedelta(days=7)).strftime("%Y%m%d")
            week_last_map = complete_week_last_dates(
                load_trade_calendar(data_start, extended_end))
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return
    eligible_market_dates = [
        value for value in open_dates if value <= requested_end]
    if not eligible_market_dates:
        st.error("截止日前没有可用市场交易日")
        return
    latest_market_date = max(eligible_market_dates)
    open_pos = {value: position for position, value in enumerate(open_dates)}
    config = {
        "signal_start": signal_start, "signal_end": latest_market_date,
        "event_signal_end": latest_market_date,
        "data_start": data_start, "market_end": latest_market_date,
        "latest_market_date": latest_market_date,
        "min_price": float(min_price), "min_mv": float(min_mv),
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct),
        "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct),
    }
    run_signature = stable_signature({
        "version": V614_EVENT_ENGINE_VERSION, **config,
        "n": V614_PRIMARY_N, "m": V614_M,
        "k_band": [V614_K_MIN, V614_K_MAX],
        "red_ages": list(V614_SHADOW_RED_AGES),
    })
    period_index = build_period_index(memberships)
    active_codes = {
        code for code, code_periods in period_index.items()
        if periods_overlap(code_periods, signal_start, latest_market_date)}
    stocks = stock_basic[stock_basic["ts_code"].isin(active_codes)].copy()
    stocks = stocks[
        ~stocks["list_date"].gt(latest_market_date)
        & ~stocks["delist_date"].lt(data_start)
    ].sort_values("ts_code").reset_index(drop=True)

    event_rows: list[dict[str, Any]] = []
    rejects: dict[str, int] = {}
    checkpoint_hits = price_cache_hits = failures = 0
    progress, status = st.progress(0.0), st.empty()
    last_update = 0.0
    stopped = False
    for number, stock in stocks.iterrows():
        if not v614_is_job_active(request_signature):
            stopped = True
            break
        code = str(stock["ts_code"])
        checkpoint = (
            v614_load_checkpoint(run_signature, code)
            if bool(use_cache) else None)
        if checkpoint is not None:
            event_rows.extend(checkpoint["events"])
            merge_counts(rejects, checkpoint["rejects"])
            checkpoint_hits += 1
        else:
            daily, cached_basic, storage_path, cache_hit = fetch_price(
                code, data_start, latest_market_date,
                bool(use_cache), float(pause))
            price_cache_hits += int(cache_hit)
            if daily.empty:
                failures += 1
            else:
                try:
                    rows, stock_rejects = v614_analyze_stock(
                        stock, period_index.get(code, []), daily,
                        cached_basic, storage_path, week_last_map,
                        open_dates, open_pos, config, bool(use_cache),
                        float(pause))
                    event_rows.extend(rows)
                    merge_counts(rejects, stock_rejects)
                    v614_save_checkpoint(
                        run_signature, code, rows, stock_rejects)
                except Exception as exc:
                    failures += 1
                    record_error(f"V6.14逐股票分析失败 {code}: {exc}")
        processed = number + 1
        now = time.monotonic()
        if (processed == 1 or now - last_update >= UI_HEARTBEAT_SECONDS
                or processed == len(stocks)):
            progress.progress(
                processed / max(len(stocks), 1),
                text=f"已处理{processed}/{len(stocks)}只股票，最近{code}")
            status.caption(
                f"事件/观察{len(event_rows)}；检查点{checkpoint_hits}；"
                f"行情缓存{price_cache_hits}；失败{failures}")
            last_update = now
    progress.empty()
    status.empty()
    if stopped:
        st.warning("任务已停止，逐股票检查点已保留。")
        return

    events_all = pd.DataFrame(event_rows)
    if events_all.empty:
        signals = pd.DataFrame()
        live_watch = pd.DataFrame()
    else:
        events_all = events_all.sort_values(
            ["Signal_Date", "Event_Type", "Strategy_Key", "ts_code"]
        ).reset_index(drop=True)
        signals = events_all[
            events_all["Event_Type"].eq("ENTRY_SIGNAL")].copy()
        live_watch = events_all[
            events_all["Event_Type"].eq("LIVE_WATCH")].copy()
    ranked = v614_rank_daily_events(signals)
    selected = ranked[true_mask(ranked, "Selected_Top5")].copy()
    calendar = v614_selection_calendar(
        ranked, open_dates, signal_start, latest_market_date)
    candidate_distribution = v614_candidate_distribution(calendar)

    price_book: dict[str, dict[str, Any]] = {}
    book_cache_hits = recovered = 0
    portfolio_missing: list[str] = []
    if not selected.empty:
        with st.spinner("复用本地行情，构建Top5生命周期价格簿..."):
            price_book, book_cache_hits, portfolio_missing = (
                v66_load_price_book(
                    selected, data_start, latest_market_date))
            still_missing: list[str] = []
            for code in portfolio_missing:
                daily, _, _, _ = fetch_price(
                    code, data_start, latest_market_date, True, float(pause))
                if daily.empty:
                    still_missing.append(code)
                    continue
                daily = add_daily_macd(normalize_price_frame(daily))
                dates = daily["trade_date"].astype(str).tolist()
                price_book[code] = {
                    "dates": dates,
                    "open": dict(zip(dates, numeric(daily, "open"))),
                    "close": dict(zip(dates, numeric(daily, "close"))),
                    "high": dict(zip(dates, numeric(daily, "high"))),
                    "low": dict(zip(dates, numeric(daily, "low"))),
                    "macd_hist": dict(zip(
                        dates, numeric(daily, "Daily_MACD_Hist"))),
                    "macd_remaining": dict(zip(
                        dates, numeric(daily, "Daily_MACD_Remaining_pct"))),
                    "macd_retention": dict(zip(
                        dates, numeric(daily, "Daily_MACD_Retention_pct"))),
                }
                recovered += 1
            portfolio_missing = still_missing
    lifecycle = v614_run_lifecycles(
        selected, price_book, open_dates, config)
    lifecycle_summary = v614_lifecycle_summary(lifecycle)
    paired_summary, paired_detail = v614_paired_exit_audit(lifecycle)
    exit_week_summary = v614_exit_week_summary(lifecycle)
    top3_cohort = v614_top3_cohort_audit(lifecycle)
    partial_week_audit = v614_partial_week_audit(lifecycle)

    main_selected = selected[
        numeric(selected, "Entry_Red_Age_Hypothesis").eq(
            V614_MAIN_RED_AGE)].copy()
    recommended = lifecycle[
        lifecycle.get("Exit_Mode_Key", pd.Series(dtype=str)).eq(
            "Hard5Trail15AHalf")
        & numeric(lifecycle, "Entry_Red_Age_Hypothesis").eq(
            V614_MAIN_RED_AGE)
    ].copy() if not lifecycle.empty else pd.DataFrame()
    latest_top5 = main_selected[
        main_selected["Signal_Date"].astype(str).eq(
            latest_market_date)].copy()
    active_statuses = {
        "待次日开盘买入", "持有", "待次日开盘卖出"}
    active = recommended[
        recommended["Lifecycle_Status"].astype(str).isin(
            active_statuses)].copy() if not recommended.empty else pd.DataFrame()
    recent_start = (
        pd.Timestamp(latest_market_date) - pd.Timedelta(days=30)
    ).strftime("%Y%m%d")
    recent_selected = main_selected[
        main_selected["Signal_Date"].astype(str).between(
            recent_start, latest_market_date)].copy()
    recent_exits = recommended[
        recommended["Exit_Date"].astype(str).between(
            recent_start, latest_market_date)].copy(
        ) if not recommended.empty else pd.DataFrame()

    completed_top3 = top3_cohort[true_mask(
        top3_cohort, "完整三只且全部退出")] if not top3_cohort.empty else pd.DataFrame()
    top3_two_win_rate = (
        true_mask(completed_top3, "完整三只至少两只盈利").mean() * 100.0
        if not completed_top3.empty else np.nan)
    run_summary = pd.DataFrame([{
        "程序版本": V614_VERSION,
        "信号开始": signal_start,
        "信号及行情截止": latest_market_date,
        "周线SKDJ参数": f"N={V614_PRIMARY_N},M={V614_M}",
        "主买点": "日线MACD第2根红柱收盘确认_下一开盘",
        "主方案全部合格候选": len(ranked[
            numeric(ranked, "Entry_Red_Age_Hypothesis").eq(2)]),
        "主方案Top5模拟入选": len(main_selected),
        "主方案有入选交易日": main_selected["Signal_Date"].nunique(
            ) if not main_selected.empty else 0,
        "主方案空窗交易日": int(calendar["红2模拟买入"].eq(0).sum()),
        "当前持有或待处理": len(active),
        "最近30日主方案入选": len(recent_selected),
        "完整同日三只组": len(completed_top3),
        "同日三只至少两只盈利比例%": top3_two_win_rate,
        "价格簿缓存命中": book_cache_hits,
        "价格簿补下载": recovered,
        "价格簿仍缺失股票": len(portfolio_missing),
        "处理股票": len(stocks), "检查点恢复": checkpoint_hits,
        "行情缓存命中": price_cache_hits, "失败股票": failures,
    }])
    definitions = pd.DataFrame([
        ("周线池", "N6/M3；每天以截至当天数据重建未完成周线；K10至25且较上一完整周K上升"),
        ("主买点", "日线MACD第2根红柱收盘确认；下一市场交易日开盘模拟买入"),
        ("影子买点", "红1和红3使用相同周线池、排序和退出，仅作对照"),
        ("每日排名", "K10至20得40分、K20至25得20分；当日K升速分位30分；MACD保持率分位30分"),
        ("Top5", "每天最多5只；不足按实际数量；零只不补；每个事件使用独立虚拟资金"),
        ("待买入", "收盘确认信号但下一交易日开盘尚未发生"),
        ("硬止损", "收盘跌破含滑点买入价5%；下一可交易开盘退出"),
        ("移动保护", "收盘跌破买入以来截至上一交易日最高价的85%；下一可交易开盘退出"),
        ("A半浮盈", "截至上一日最高收盘达到买入价120%后，至少保护最高收盘浮盈的一半"),
        ("实际退出线", "硬止损、15%移动保护、A半浮盈三线取最高；只升不降"),
        ("持仓期限", "不设固定40日；没有触发退出就持续持有"),
        ("动态等级", "最大净浮盈≥30/20/10对应S/A/B；已退出且不足10为F；持有不足10为未定级"),
    ], columns=["项目", "定义"])
    rejection_audit = pd.DataFrame([
        {"剔除原因": key, "次数": value}
        for key, value in sorted(rejects.items())])
    missing_price_audit = pd.DataFrame({
        "价格簿缺失股票": portfolio_missing})
    cache_policy = pd.DataFrame([
        ("交易日历/基础资料", "Streamlit内存", "72小时"),
        ("行业成员", "Streamlit内存", "7天"),
        ("逐股票前复权行情", "应用临时磁盘", "不主动过期"),
        ("V6.14逐股票事件检查点", "应用临时磁盘", "不主动过期"),
    ], columns=["对象", "位置", "设定时长"])

    selection_columns = [
        "Pick_ID", "Entry_Red_Age_Hypothesis", "Signal_Date",
        "Signal_Weekday", "Daily_Rank", "V614_Rank_Score_100",
        "V614_K_Band_Score", "V614_K_Change_Score",
        "V614_MACD_Retention_Score", "ts_code", "name", "SW_L1",
        "SW_L2", "Weekly_Bar_State", "Week_Last_Trading_Date",
        "Later_Week_Close_Confirmed", "Signal_K", "Signal_D",
        "Signal_KD_Spread", "Signal_K_Change_1W",
        "Daily_MACD_Red_Age", "Daily_MACD_Hist",
        "Daily_MACD_Retention_pct", "Daily_MACD_Remaining_pct",
        "Signal_Daily_SKDJ_N6_K", "Signal_Daily_SKDJ_N6_D",
        "Signal_Daily_SKDJ_N9_K", "Signal_Daily_SKDJ_N9_D",
        "Raw_Close", "Circ_MV_Billion", "Entry_Tradable",
        "Entry_Reason", "Entry_Date", "Entry_Raw_Open",
    ]
    lifecycle_columns = selection_columns + [
        "Exit_Mode_Key", "Exit_Mode", "Entry_Execution_Date",
        "Entry_Trade_Price", "Lifecycle_Status", "Decision_Date",
        "Exit_Date", "Exit_Reason", "Exit_Raw_Open", "Exit_Trade_Price",
        "Realized_Return_Net_pct", "Current_Return_Net_pct",
        "Current_or_Exit_Return_Net_pct", "Hold_Market_Days", "Exit_Week",
        "Highest_High_Through_Status", "Highest_Close_Through_Status",
        "MFE_Net_pct", "MAE_Raw_pct", "Dynamic_Grade",
        "Decision_Hard_Stop_Line", "Decision_Trail15_Line",
        "Decision_AHalf_Profit_Line", "Decision_Effective_Line",
        "Next_Day_Hard_Stop_Line", "Next_Day_Trail15_Line",
        "Next_Day_AHalf_Profit_Line", "Next_Day_Effective_Line",
    ]
    selection_export = selected[[
        column for column in selection_columns if column in selected.columns
    ]].copy()
    lifecycle_export = lifecycle[[
        column for column in lifecycle_columns if column in lifecycle.columns
    ]].copy()
    recommended_export = recommended[[
        column for column in lifecycle_columns if column in recommended.columns
    ]].copy() if not recommended.empty else pd.DataFrame()
    active_export = active[[
        column for column in lifecycle_columns if column in active.columns
    ]].copy() if not active.empty else pd.DataFrame()
    recent_exit_export = recent_exits[[
        column for column in lifecycle_columns if column in recent_exits.columns
    ]].copy() if not recent_exits.empty else pd.DataFrame()
    watch_columns = [
        "Signal_Date", "ts_code", "name", "SW_L1", "SW_L2",
        "Weekly_Bar_State", "Signal_K", "Signal_D", "Signal_KD_Spread",
        "Signal_K_Change_1W", "Daily_MACD_State", "Daily_MACD_Red_Age",
        "Daily_MACD_Hist", "Raw_Close", "Circ_MV_Billion",
    ]
    live_watch_export = live_watch[[
        column for column in watch_columns if column in live_watch.columns
    ]].copy() if not live_watch.empty else pd.DataFrame()
    latest_top5_export = latest_top5[[
        column for column in selection_columns if column in latest_top5.columns
    ]].copy() if not latest_top5.empty else pd.DataFrame()

    files = {
        "01_run_summary_v6_14.csv": run_summary,
        "02_experiment_definitions_v6_14.csv": definitions,
        "03_daily_candidate_calendar_v6_14.csv": calendar,
        "04_daily_candidate_count_distribution_v6_14.csv": candidate_distribution,
        "05_all_red123_ranked_candidates_v6_14.csv": ranked,
        "06_all_red123_top5_selections_v6_14.csv": selection_export,
        "07_all_exit_modes_lifecycle_v6_14.csv": lifecycle_export,
        "08_recommended_red2_lifecycle_v6_14.csv": recommended_export,
        "09_rank1_top3_top5_lifecycle_summary_v6_14.csv": lifecycle_summary,
        "10_exit_mode_paired_summary_v6_14.csv": paired_summary,
        "11_exit_mode_paired_detail_v6_14.csv": paired_detail,
        "12_exit_week_distribution_v6_14.csv": exit_week_summary,
        "13_red2_top3_daily_cohort_audit_v6_14.csv": top3_cohort,
        "14_partial_week_confirmation_audit_v6_14.csv": partial_week_audit,
        "15_latest_day_red2_top5_v6_14.csv": latest_top5_export,
        "16_current_open_and_pending_v6_14.csv": active_export,
        "17_recent_30d_exits_v6_14.csv": recent_exit_export,
        "18_latest_n6_dynamic_weekly_pool_v6_14.csv": live_watch_export,
        "19_rejection_audit_v6_14.csv": rejection_audit,
        "20_missing_price_book_v6_14.csv": missing_price_audit,
        "21_cache_policy_v6_14.csv": cache_policy,
        "22_api_errors_v6_14.csv": pd.DataFrame({"错误": API_ERRORS}),
    }
    result_zip = make_zip(files)
    try:
        atomic_bytes(result_zip, result_path)
        v614_clear_job_active(request_signature)
        persisted = True
    except Exception as exc:
        persisted = False
        st.warning(f"结果未能持久保存，但当前页面仍可下载：{exc}")

    st.success(
        f"完成：N6红2合格候选{len(ranked[numeric(ranked, 'Entry_Red_Age_Hypothesis').eq(2)])}个，"
        f"每日Top5模拟入选{len(main_selected)}个；"
        f"当前持有或待处理{len(active)}个；"
        f"结果{'已保存' if persisted else '仅当前页面可下载'}。")
    if portfolio_missing:
        st.warning(
            f"仍有{len(portfolio_missing)}只入选股票价格簿缺失；"
            "已在20号文件列出，相关未成交状态不能用于评价退出规则。")

    st.subheader(f"今天的红2 Top5：{latest_market_date}")
    if latest_top5_export.empty:
        st.info("今天没有符合周线N6＋日线MACD红2条件的股票，不补弱信号。")
    else:
        render_plain_table(latest_top5_export, V614_TOP_N)
    st.subheader("当前仍持有、待买入或待卖出")
    render_plain_table(active_export, 500)
    st.subheader("第1名、前3名、前5名生命周期结果")
    render_plain_table(lifecycle_summary, 100)
    st.caption("这是无限虚拟资金的独立选股质量，不是三仓资金收益。")
    st.subheader("基础15%回撤与A类半浮盈保护的逐事件配对")
    render_plain_table(paired_summary, 20)
    st.subheader("红2同日前3质量审计")
    render_plain_table(top3_cohort, 300)
    st.caption("只有同日实际入选3只且3只均已退出的日期才进入“至少两只盈利”验收。")
    st.subheader("未完成周线后来是否在周收盘继续确认")
    render_plain_table(partial_week_audit, 30)
    st.subheader("红2每日候选数量分布")
    render_plain_table(candidate_distribution, 20)
    with st.expander("最近30日退出与退出周分布", expanded=False):
        render_plain_table(recent_exit_export, 300)
        render_plain_table(exit_week_summary, 200)
    with st.expander("最近交易日N6动态周线观察池", expanded=False):
        render_plain_table(live_watch_export, 500)
    st.subheader("运行摘要")
    render_plain_table(run_summary, 10)
    st.caption(
        f"结果ZIP共{len(files)}个CSV；每日候选、Top5、两套退出、"
        "当前持有和最近退出均分开导出。")
    render_download(
        result_zip, result_name, f"v614_current_{request_signature}")


def main() -> None:
    v614_main()


if __name__ == "__main__":
    main()
