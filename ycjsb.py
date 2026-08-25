# -*- coding: utf-8 -*-
"""周线SKDJ提前买入、状态确认与选择性重入审计 V6.4。

本版根据V6.1结果，把周线N=6 SKDJ K位于15～25定义为观察池，并独立扫描
日线MACD红柱周期内股价自红柱起点首次上涨3%、5%、8%和10%的四个强度确认
买点。触发时还必须满足红柱扩张或本轮红柱剩余强度不低于75%。是否后来上穿
周线25只作为事后审计字段，绝不参与买点生成。

另设严格诊断组：只取红柱第2日已进入观察池且同一红柱周期确实持续到第5日的
共同样本，分别模拟第2、3、4、5日买入。该组使用了未来“能持续到第5日”的
信息，只用于公平比较等待代价，绝不作为实盘规则或实时候选。

V6.4沿用3%提前买点，但不再把任何一次周线K上穿25都视为同等确认。程序在
确认日按日线MACD状态和本轮累计涨幅分为高质量确认、普通确认与已涨超30%的
早仓利润保护组；14日超时退出后，只允许后来出现“红柱扩张且已涨10～30%”
的高质量确认重新买入。同时比较普通确认立即退出、红柱剩余10/20/30%退出，
并统计选择性重入究竟填补了多少原本没有3%新信号的星期。
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

# V6.4 state-aware lifecycle audit.  Internal v63 helper names are retained to
# minimize regression risk; the final ``main`` is the only entry point.
V63_TITLE = "周线SKDJ提前买入、状态确认与选择性重入审计 V6.4"
V63_VERSION = "V6.4-SKDJ-STATE-CONFIRMATION-SELECTIVE-REENTRY"
V63_UI_PATCH = "V6.4-3PCT-ENTRY-STATE-CONFIRM-SELECTIVE-REENTRY-MACD-EXIT"
V63_CHECKPOINT_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_4_checkpoints")
V63_RESULT_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_4_results")
V63_JOB_DIR = os.path.join(APP_DIR, "weekly_skdj_v6_4_jobs")
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
        rejects["存在V6.4信号但daily_basic缺失"] = estimated
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
        has_reentry = early[f"{prefix}Reentry_Signal_Date"].astype(str).str.len().gt(0)
        mature = true_mask(early, f"{prefix}Reentry_Has_40D")
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
        ("任意后来上穿25都重入", true_mask(
            early, "Confirm14_Reentry_Has_40D"), "Confirm14_Reentry_"),
        ("只在高质量确认时重入", true_mask(
            early, "Selective_Reentry14_Has_40D"),
         "Selective_Reentry14_"),
    )
    for label, mask, prefix in schemes:
        group = early[mask].copy()
        classes = group[f"{prefix}Class_40D"].astype(str)
        combined = numeric(group, f"{prefix}Combined_Return_Net_pct")
        signal_dates = group[f"{prefix}Signal_Date"].astype(str)
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
            dates = early[column].astype(str)
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
                statuses.loc[index] = "14日内高质量确认_继续持有或加仓"
            elif on_time and state == V64_CROSS_OVERHEATED:
                statuses.loc[index] = "14日内已涨超30_早仓保护_禁止追买"
            elif on_time:
                statuses.loc[index] = "14日内普通确认_不加仓"
            elif state == V64_CROSS_HIGH:
                statuses.loc[index] = "14日退出后高质量确认_允许选择性重入"
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
        record_error(f"V6.4任务标记清除失败: {exc}")


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
        record_error(f"V6.4检查点损坏 {ts_code}: {exc}")
        return None


def v63_save_checkpoint(
        signature: str, ts_code: str, events: list[dict[str, Any]],
        rejects: dict[str, int]) -> None:
    atomic_pickle({
        "signature": signature, "ts_code": str(ts_code),
        "events": events, "rejects": rejects,
    }, v63_checkpoint_path(signature, ts_code))


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=V63_TITLE, layout="wide")
    st.title(V63_TITLE)
    streamlit_version = str(getattr(st, "__version__", "unknown"))
    st.caption(f"{V63_UI_PATCH}｜Streamlit {streamlit_version}")
    with st.expander("V6.4验证口径", expanded=True):
        st.markdown(f"""
- **唯一提前买点**：最近完整周N=6、M=3的SKDJ K位于{EARLY_WEEKLY_K_MIN:g}～{EARLY_WEEKLY_K_MAX:g}；日线MACD红柱扩张或剩余强度≥{STRENGTH_MIN_REMAINING_PCT:g}%；本轮股价首次达到+{V63_STRENGTH_THRESHOLD:g}%后，下一市场交易日开盘买入。
- **主确认期限**：提前买入后观察{V64_PRIMARY_CONFIRM_DAYS}自然日。未上穿25则在期限后的下一只可交易开盘退出；7日结果只保留为速度对照。
- **确认状态**：上穿25当天，红柱扩张且本轮累计上涨10～30%为高质量确认；累计上涨超过30%属于已有早仓的利润保护组，禁止把它解释为新的追高买点；其余为普通确认。
- **普通确认退出审计**：分别测试确认后立即退出，以及从确认日开始等待日线MACD红柱翻绿或剩余强度≤10/20/30%再退出。确认前出现过的日线变化不能触发退出。
- **选择性重入**：14日超时退出后，原信号起{V63_REENTRY_WINDOW_DAYS}自然日内只有后来出现高质量确认才允许重入；“任何上穿25都重入”仅作为失败对照。
- **空窗检验**：分别统计任意重入和高质量重入真正填补了多少原3%信号空窗周，不把重入伪装成新的候选股票。
- **判卷**：S/A/B仍为40日内先到+30/+20/+10且先于-10，其余为F；所有退出与重入均按当时可见数据和下一可交易开盘执行并计入成本。
- **缓存**：交易日历和股票基础资料72小时，行业成员7天，逐股票行情与检查点不主动过期；应用重启或重新部署仍可能清空实例内存和临时磁盘。
""")

    with st.sidebar:
        st.header("运行参数")
        backtest_days = st.number_input(
            "回测交易日数", 100, 1000, 500, 50, key="v64_days")
        min_price = st.number_input(
            "最低股价（元）", 10.0, 20.0, 10.0, 10.0,
            format="%.0f", key="v64_min_price")
        min_mv = st.number_input(
            "最低流通市值（亿元）", 50.0, 100.0, 50.0, 50.0,
            format="%.0f", key="v64_min_mv")
        signal_end_date = st.date_input(
            "历史买入信号截止（40日判卷）", date(2026, 6, 5),
            key="v64_signal_end")
        market_end_date = st.date_input(
            "最新信号观察截止（默认今天）", date.today(),
            key="v64_market_end")
        pause = st.number_input(
            "接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v64_pause")
        use_cache = st.checkbox(
            "复用行情和72小时基础缓存", True, key="v64_cache")
        st.caption("逐股票行情和检查点不设TTL；同参数页面重连会自动续跑。")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005,
            format="%.3f", key="v64_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01,
            format="%.3f", key="v64_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001,
            format="%.3f", key="v64_transfer")
        if st.button("清除V6.4结果和运行状态", key="v64_clear"):
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
        "score_weights": [V63_SCORE_K_WEIGHT, V63_SCORE_AGE_WEIGHT,
                          V63_SCORE_KD_WEIGHT],
    }
    request_signature = stable_signature(request_payload)
    result_path = os.path.join(V63_RESULT_DIR, f"{request_signature}.zip")
    result_name = (
        f"weekly_skdj_state_confirmation_reentry_audit_v6_4_"
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
                saved_result, result_name, f"v64_saved_{request_signature}")
        except Exception as exc:
            st.warning(f"已保存结果读取失败：{exc}")

    secret_token = configured_tushare_token()
    if secret_token:
        token = secret_token
        st.caption("已从Streamlit Secrets读取TUSHARE_TOKEN。")
    else:
        token = st.text_input("Tushare Token", type="password", key="v64_token")
    job_active = v63_is_job_active(request_signature)
    left, right = st.columns(2)
    with left:
        start_clicked = st.button(
            "开始/重新运行V6.4", type="primary", key="v64_run")
    with right:
        stop_clicked = st.button(
            "停止自动续跑", disabled=not job_active, key="v64_stop")
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
    run_signature = stable_signature({"version": V63_VERSION, **config})
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
                    record_error(f"V6.4逐股票分析失败 {code}: {exc}")
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
        st.error("本区间没有生成V6.4事件。")
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

    overall = timing_outcome_summary(
        mature[mature["Event_Type"].isin(["STRENGTH_3", "WEEKLY_CROSS25"])],
        ["Event_Type", "Strategy_Label"], "V6.4基础买点")
    score_audit = timing_outcome_summary(
        early, ["Timing_Score_221"], "2-2-1结构分档")
    three_factor_audit = timing_outcome_summary(
        early, ["Timing_Three_Factor_Count"], "三个结构条件命中数")
    rank_summary = v63_weighted_rank_summary(early)
    early["Calendar_Year"] = early["Signal_Date"].astype(str).str[:4]
    rank_year_rows: list[pd.DataFrame] = []
    for year, group in early.groupby("Calendar_Year"):
        part = v63_weighted_rank_summary(group)
        if not part.empty:
            part.insert(0, "年度", year)
            rank_year_rows.append(part)
    rank_year = pd.concat(rank_year_rows, ignore_index=True) if rank_year_rows else pd.DataFrame()
    gate_summary = v63_gate_summary(early)
    reentry_summary = v63_reentry_summary(early)
    realized_strategy = v64_realized_strategy_summary(early)
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
        "14日超时": int((~true_mask(early, "Confirm14_Confirmed")).sum()),
        "任意金叉成熟重入": int(true_mask(
            early, "Confirm14_Reentry_Has_40D").sum()),
        "高质量确认成熟重入": int(true_mask(
            early, "Selective_Reentry14_Has_40D").sum()),
        "3%提前信号周": early["Signal_Week"].nunique(),
        "最近14日3%提前信号": len(recent_early),
        "最近14日周线确认信号": len(recent_weekly),
        "最新观察池股票": len(live_watch),
        "处理股票数": len(stocks), "检查点恢复": checkpoint_hits,
        "行情缓存命中": price_cache_hits, "失败股票": failures,
    }])
    definitions = pd.DataFrame([
        ("提前买点", "周线K15至25；日线MACD质量合格；本轮首次上涨3%；次日开盘"),
        ("主确认期限", "14自然日内上穿25视为按时确认；7日只作速度对照"),
        ("高质量确认", "上穿日红柱扩张且本轮累计上涨10至30%"),
        ("普通确认", "不满足高质量且累计上涨未超过30%；不自动加仓"),
        ("已涨超过30", "已有早仓测试持有或保护；确认后禁止作为新买点追入"),
        ("普通确认退出", "从确认日开始测试立即退出或MACD剩余10/20/30%退出"),
        ("选择性重入", "14日退出后只在42日窗口内出现高质量确认才重新买入"),
        ("失败对照", "14日退出后任何上穿25都重入"),
        ("空窗统计", "重入与原3%新信号分开，统计实际填补空窗周"),
        ("历史与实时", "历史只到默认20260605并要求40日成熟；以后只展示观察名单"),
        ("缓存", "交易日历与股票基础资料72小时；行业7天；行情和检查点不设TTL"),
    ], columns=["项目", "定义"])
    cache_policy = pd.DataFrame([
        ("交易日历", "Streamlit内存", "72小时", "实例重启可能提前消失"),
        ("股票基础资料", "Streamlit内存", "72小时", "实例重启可能提前消失"),
        ("申万科技行业成员", "Streamlit内存", "7天", "实例重启可能提前消失"),
        ("逐股票日线与daily_basic", "应用临时磁盘", "不自动过期", "重新部署可能清空"),
        ("V6.4逐股票检查点", "应用临时磁盘", "不自动过期", "参数变化使用新签名"),
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
        "延迟重入窗口自然日": V63_REENTRY_WINDOW_DAYS,
        "高质量确认": "红柱扩张且累计上涨10至30%",
        "普通确认退出阈值": "/".join(
            str(int(v)) for v in V64_POST_CROSS_REMAINING),
        "选择性重入": "只允许高质量确认",
        "评分权重": f"{V63_SCORE_K_WEIGHT}-{V63_SCORE_AGE_WEIGHT}-{V63_SCORE_KD_WEIGHT}",
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
    files = {
        "01_run_summary_v6_4.csv": run_summary,
        "02_experiment_definitions_v6_4.csv": definitions,
        "03_strength3_vs_weekly_outcomes_v6_4.csv": overall,
        "04_confirmation_exit_7d_14d_v6_4.csv": gate_summary,
        "05_early_outcomes_by_cross_state_v6_4.csv": cross_state_outcomes,
        "06_fresh_cross_entry_by_state_v6_4.csv": cross_entry_outcomes,
        "07_cross_state_by_year_v6_4.csv": cross_state_year,
        "08_realized_lifecycle_strategy_v6_4.csv": realized_strategy,
        "09_any_vs_selective_reentry_v6_4.csv": selective_reentry,
        "10_reentry_week_coverage_v6_4.csv": reentry_week_coverage,
        "11_confirmation_status_original_grade_v6_4.csv": gate_status_audit,
        "12_weekly_momentum_priority_v6_4.csv": weekly_momentum,
        "13_year_stability_v6_4.csv": year_audit,
        "14_weekly_coverage_summary_v6_4.csv": coverage,
        "15_weekly_signal_calendar_v6_4.csv": calendar,
        "16_historical_strength3_lifecycle_detail_v6_4.csv": history_early_export,
        "17_historical_weekly_control_detail_v6_4.csv": history_weekly_export,
        "18_recent_14d_strength3_candidates_v6_4.csv": recent_early_export,
        "19_recent_14d_weekly_candidates_v6_4.csv": recent_weekly_export,
        "20_latest_market_day_watch_pool_v6_4.csv": live_watch_export,
        "21_cache_policy_v6_4.csv": cache_policy,
        "22_rejection_audit_v6_4.csv": rejection_audit,
        "23_metadata_v6_4.csv": metadata,
        "24_api_errors_v6_4.csv": pd.DataFrame({"错误": API_ERRORS}),
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
        f"高质量确认{int(early['Future_Cross_State'].eq(V64_CROSS_HIGH).sum())}个，"
        f"高质量成熟重入{int(true_mask(early, 'Selective_Reentry14_Has_40D').sum())}个；"
        f"最近14日3%信号{len(recent_early)}个，最新观察池{len(live_watch)}只；"
        f"结果{'已保存' if persisted else '仅当前页面可下载'}。")
    st.subheader("结论一：各生命周期方案的真实落袋收益")
    render_plain_table(realized_strategy, 20)
    st.subheader("结论二：确认速度与确认日状态")
    render_plain_table(cross_state_outcomes, 30)
    st.caption("上表从3%提前买入者视角判卷；下表从确认后才新买入者视角判卷。")
    render_plain_table(cross_entry_outcomes, 30)
    st.subheader("结论三：任何金叉重入与高质量确认重入")
    render_plain_table(selective_reentry, 10)
    render_plain_table(reentry_week_coverage, 10)
    st.subheader("7日与14日未确认退出对照")
    render_plain_table(gate_summary, 10)
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
    st.caption("结果ZIP共24个CSV；历史成熟事件、最近候选和最新观察池严格分开。")
    render_download(result_zip, result_name, f"v64_current_{request_signature}")


if __name__ == "__main__":
    main()
