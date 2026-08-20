# -*- coding: utf-8 -*-
"""周线SKDJ N=6/7/9 买入提前量与W1-W12生命周期审计 V5.5。

V5.4的信号、50亿元科技池、25线下连续1至5周硬条件和三套排序全部冻结，
本版不再调分。新增默认参数N=9作为慢速对照，并把入场后的累计最大浮盈、
最大不利波动和收盘净收益扩展到W1-W12。排名仍只使用W8成熟样本；W12只用于
持仓生命周期审计，避免为了观察更久而改变V5.4的排名样本。

另以同一股票、相邻信号轮次做N6/N7/N9一对一匹配，直接测量提前多少天、
早买的价格差以及是否仍捕捉同一轮W12高点。目标不是强迫前段和后段收益相同，
而是分别识别失败信号应多早退出、成功信号的利润通常在哪一周趋于成熟。
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

TITLE = "周线SKDJ N=6/7/9 买入提前量与W1-W12生命周期审计 V5.5"
VERSION = "V5.5-WEEKLY-SKDJ-N6-N7-N9-LIFECYCLE-AUDIT"
UI_PATCH = "V5.5-N6-N7-N9-W12-MATCHED-LIFECYCLE"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# 沿用旧行情缓存目录，以便直接复用V4.7已经下载的更长历史数据。
PRICE_CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")
CHECKPOINT_DIR = os.path.join(APP_DIR, "weekly_skdj_v5_5_checkpoints")
RESULT_DIR = os.path.join(APP_DIR, "weekly_skdj_v5_5_results")
JOB_DIR = os.path.join(APP_DIR, "weekly_skdj_v5_5_jobs")

SKDJ_NS = (6, 7, 9)
SKDJ_M = 3
SKDJ_BOTTOM = 25.0
WARMUP_WEEKS = 30
RANKING_WEEKS = 8
AUDIT_WEEKS = 12
LIFECYCLE_WEEKS = tuple(range(1, AUDIT_WEEKS + 1))
UI_HEARTBEAT_SECONDS = 5.0
FIRST_HIT_LEVELS = (10.0, 15.0, 20.0)
FIRST_HIT_AUDIT_WEEKS = (RANKING_WEEKS, AUDIT_WEEKS)
PAIR_MAX_CALENDAR_DAYS = 84
MAX_BOTTOM_STREAK = 5
RANDOM_DRAWS = 500
RANDOM_SEED = 20260820
LEGACY_MIN_MV = 100.0

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
            **snapshot,
        }
        row.update({f"Entry_{key}": value for key, value in outcome.items()})
        rows.append(row)
    return rows, rejects


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
    """Compare all three rankings on identical candidates and periods."""
    rows: list[dict[str, Any]] = []
    for spec in ranking_scheme_specs():
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


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    streamlit_version = str(getattr(st, "__version__", "unknown"))
    st.caption(
        f"{UI_PATCH}｜冻结V5.4选股与排序；新增N=9、W1～W12和同股同轮匹配。"
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
- **数据长度**：默认正式窗口500个交易日；另加开始前{WARMUP_WEEKS}周预热和截止后W1-W12观察。
- **过滤**：每个历史信号日分别检查当时科技行业归属；最低股价默认10元、最低流通市值默认50亿元，侧边栏可切换，避免使用今天状态回看历史。
- **共同硬条件**：信号前连续处于25下方1～{MAX_BOTTOM_STREAK}周；超过5周不进入三套排名，但仍保留剔除计数。
- **原评分基准**：SKDJ重置35分、量能20分、周K线结构20分、MA20趋势15分、同周价格/市值相对排名10分。
- **V5.2对照**：保留上一版共振S/A/B/C结果，作为新方案必须超过的直接对照。
- **V5.4历史二层**：最近3次已完成金叉至少2次达到75线进入优先层；2次和3次不再区分，其余进入普通层。
- **V5.4同层顺序**：先按原100分降序；最近一次已完成金叉峰值仅在原分相同时破同分。
- **板块共振**：退出个股排名，只报告不同共振区间的表现和周内排序相关性。
- **候选宽度**：1～4只、5～20只、超过20只分别标记低/中/高仓位置信度；只审计，不剔除、不改变名次。
- **相对行业强度**：4/8/12周相对强度继续只审计，不计分。
- **周内比较**：三套排名分别比较全部候选、前20%、前5、前3、第4名以后及后20%。
- **排名口径不变**：排名仍使用完整W8成熟样本；W12成熟样本仅分析持仓生命周期，避免因要求多4周未来数据而改写V5.4名次。
- **生命周期**：逐周报告累计最大浮盈、最大不利波动、收盘收益和相对前一周的增量；不预设N越小就必须越早卖。
- **失败与赢家**：分别观察W12先到-10%、先到+10%、最大浮盈≥20%和最终亏损事件在W1～W4何时开始分化。
- **同股同轮**：在同一股票84个日历日内一对一匹配N6/N7/N9信号，比较领先天数、实际下一日开盘买价和W12高点日期。
- **随机基准**：每个有候选的星期随机选择最多3只，重复{RANDOM_DRAWS}次，判断正式前3是否只是运气。
- **压力测试**：分别剔除收益最高事件、贡献最高股票和表现最好信号周，检查成绩是否依赖少数牛股。
- **防泄漏**：历史最低K和5周均量均只使用信号周以前的完整周；信号当周只使用本周已经收盘的数据。
- **时间分段**：评分规则在代码中预先冻结；前段和后段只用于分别报告，不读取后段收益重新调权。
- **买卖口径**：下一交易日开盘买入，W1～W12仅做观察；本版不根据结果自动生成止盈止损参数。
""")
    with st.sidebar:
        st.header("运行参数")
        backtest_days = st.number_input(
            "回测交易日数", 100, 1000, 500, 50, key="v55_days")
        st.caption("本版默认500日，才能较可靠地比较N6/N7/N9的持仓生命周期。")
        # 沿用本页已经稳定加载的number_input，避免部分iOS/Safari会话在
        # 首次加载selectbox前端分块时出现“Importing a module script failed”。
        min_price = st.number_input(
            "最低股价（元）", min_value=10.0, max_value=20.0,
            value=10.0, step=10.0, format="%.0f", key="v55_min_price")
        min_mv = st.number_input(
            "最低流通市值（亿元）", min_value=50.0, max_value=100.0,
            value=50.0, step=50.0, format="%.0f", key="v55_min_mv")
        st.caption(f"本次历史过滤：股价≥{min_price:.0f}元，流通市值≥{min_mv:.0f}亿元")
        signal_end_date = st.date_input(
            "买入信号截止", date(2026, 6, 5), key="v55_signal_end")
        market_end_date = st.date_input(
            "行情观察截止", date.today(), key="v55_market_end")
        split_ratio_pct = st.number_input(
            "前段观察占正式周比例(%)", 50, 80, 60, 5, key="v55_split")
        pause = st.number_input(
            "接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v55_pause")
        use_cache = st.checkbox("复用行情缓存", True, key="v55_cache")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005,
            format="%.3f", key="v55_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01,
            format="%.3f", key="v55_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001,
            format="%.3f", key="v55_transfer")
        if st.button("清除V5.5检查点和结果", key="v55_clear"):
            shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
            shutil.rmtree(RESULT_DIR, ignore_errors=True)
            shutil.rmtree(JOB_DIR, ignore_errors=True)
            st.success("V5.5检查点和结果已清除；旧行情缓存保留")

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
        f"weekly_skdj_n6_n7_n9_lifecycle_v5_5_{int(backtest_days)}d_"
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
                saved_result, result_name, f"v55_saved_{request_signature}")
        except Exception as exc:
            st.warning(f"旧结果读取失败：{exc}")

    secret_token = configured_tushare_token()
    if secret_token:
        token = secret_token
        st.caption("已从Streamlit Secrets读取TUSHARE_TOKEN。")
    else:
        token = st.text_input("Tushare Token", type="password", key="v55_token")

    job_active = is_job_active(request_signature)
    left, right = st.columns(2)
    with left:
        start_clicked = st.button("开始/重新运行V5.5", type="primary", key="v55_run")
    with right:
        stop_clicked = st.button("停止自动续跑", disabled=not job_active, key="v55_stop")
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
    eligible = add_challenger_rankings(eligible)
    lifecycle_eligible = eligible[true_mask(eligible, "Entry_Has_W12")].copy()
    if lifecycle_eligible.empty:
        st.error("存在W8排名样本，但没有未来完整W12的生命周期样本；请延后行情观察截止。")
        return
    with st.spinner("复现V5.4排名，并审计N6/N7/N9的W1～W12生命周期..."):
        score_rules = frozen_score_definitions()
        challenger_rules = challenger_score_definitions()
        rank_cohorts = scheme_rank_cohort_audit(eligible, periods)
        rank_overlap = scheme_rank_overlap_audit(eligible, periods)
        random_benchmark = scheme_random_top3_benchmark(eligible, periods)
        concentration = scheme_concentration_stress(eligible, periods)
        ranked_calendar = weekly_rank_calendar(
            calendar, scored_all, eligible, split_end)
        within_week_ic = within_week_rank_ic_audit(eligible, periods)
        lifecycle = lifecycle_horizon_audit(lifecycle_eligible, periods)
        holding_summary = holding_window_summary(
            lifecycle_eligible, lifecycle, periods)
        early_failure = early_failure_audit(lifecycle_eligible, periods)
        pair_summary, pair_detail, triple_detail = parameter_pair_audit(
            lifecycle_eligible)
    summary_rows = []
    for n in SKDJ_NS:
        all_n = events_all[events_all["SKDJ_N"].eq(n)]
        mature_n = mature_w8[mature_w8["SKDJ_N"].eq(n)]
        mature_w12_n = mature_w12[mature_w12["SKDJ_N"].eq(n)]
        eligible_n = eligible[eligible["SKDJ_N"].eq(n)]
        lifecycle_n = lifecycle_eligible[lifecycle_eligible["SKDJ_N"].eq(n)]
        pass_by_date = eligible_n.groupby(eligible_n["Signal_Date"].astype(str)).size()
        pass_counts = calendar["Week_End"].astype(str).map(pass_by_date).fillna(0).astype(int)
        summary_rows.append({
            "SKDJ_N": n, "SKDJ_M": SKDJ_M,
            "全部通过过滤事件": len(all_n), "W8成熟事件": len(mature_n),
            "W12成熟事件": len(mature_w12_n),
            "硬条件通过事件": len(eligible_n),
            "硬条件通过且W12成熟事件": len(lifecycle_n),
            "硬条件剔除事件": len(mature_n) - len(eligible_n),
            "硬条件通过不同股票": eligible_n["ts_code"].nunique(),
            "硬条件通过有信号周": int(pass_counts.gt(0).sum()),
            "硬条件后空窗周": int(pass_counts.eq(0).sum()),
            "硬条件后最长连续空窗周": max_empty_run(pass_counts),
            "硬条件后每周信号均值": pass_counts.mean(),
            "硬条件后单周最多": pass_counts.max(),
            "硬条件后1至4只候选周": int(pass_counts.between(1, 4).sum()),
            "硬条件后5至20只候选周": int(pass_counts.between(5, 20).sum()),
            "硬条件后超过20只候选周": int(pass_counts.gt(20).sum()),
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
        })
    run_summary = pd.DataFrame(summary_rows)
    run_summary.insert(0, "程序版本", VERSION)
    run_summary["正式信号开始"] = signal_start
    run_summary["正式信号截止"] = signal_end
    run_summary["实际行情开始"] = data_start
    run_summary["行情观察截止"] = market_end
    run_summary["前段观察截止"] = split_end
    run_summary["前段观察比例%"] = int(split_ratio_pct)
    run_summary["随机3只重复次数"] = RANDOM_DRAWS
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
                "层级": "V5.5沿用V5.4共同硬条件", "SKDJ_N": n,
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
        ("双成熟样本", "V5.4排名继续使用完整W8样本；生命周期只使用完整W12子样本，不让延长观察期改变原排名"),
        ("历史价格市值过滤", f"信号日股价≥{float(min_price):g}元、流通市值≥{float(min_mv):g}亿元"),
        ("三方案共同硬条件", f"信号前连续处于25下方1～{MAX_BOTTOM_STREAK}周；超过5周不进入排名"),
        ("原评分基准", "V5.1.1冻结100分原样保留，既是基准也是各分层内部的主要排序依据"),
        ("原评分结构", "SKDJ重置35、量能20、周K线结构20、MA20趋势15、同周价格市值百分位10"),
        ("V5.2对照", "保留上一版共振S/A/B/C定义和同层原评分排序；已删除V5.2强权重线性100分"),
        ("V5.4历史二层", "最近3次已完成金叉至少2次达到75线进入优先层；2次和3次同级，其余进入普通层；板块共振不参与等级"),
        ("V5.4同层规则", "同一层内先按V5.1.1冻结100分降序；最近一次金叉峰值只用于原分相同时破同分"),
        ("当前K值", "突破当周K值不计分"),
        ("时间分段", f"前段观察截止{split_end}；规则预先写死，后段不重新调权；不要求前后段绝对收益相同"),
        ("验收重点", "前3为主验收、前20%为辅助验收；V5.4同时对比原评分和V5.2共振分层，比较收益、胜率、回撤与随机百分位"),
        ("明显改善预设", "平均/中位收益至少提高2个百分点、胜率至少提高3个百分点、触及-10%至少下降3个百分点；每阶段4项中至少3项通过，且前后段均通过"),
        ("随机基准", f"每周随机选择最多3只，固定随机种子，重复{RANDOM_DRAWS}次"),
        ("集中度压力", "事后剔除最佳事件、贡献最高股票和最佳信号周，仅用于检查牛股依赖，不属于交易规则"),
        ("行业相对强度", "个股4/8/12周收益减去同N、同周、同申万一级行业且通过硬条件候选的收益中位数；同行业仅1只时留空"),
        ("板块共振", "同N同周同行业候选数占当周候选比例；只分档报告表现和周内IC，不再计分或改变个股名次"),
        ("候选宽度置信度", "同N同周1～4只标记低、5～20只标记中、超过20只标记高；只审计，不剔除、不改变名次"),
        ("此前金叉峰值", "只使用信号前已经完成死叉的最近1至3个K上穿D周期；记录各周期最高K及达到75的次数，未完成当前周期不参与"),
        ("周内IC", "每个至少5只候选且特征非恒定的信号周分别计算特征与W8收益Spearman，再报告周均值、中位和为正比例"),
        ("防未来数据", "历史窗口均shift(1)排除信号周；买入及W1-W12结果不参与信号、硬条件或排名定义"),
        ("历史过滤", "每个信号日使用当时科技行业归属、股价和流通市值"),
        ("买入", "信号完整周结束后的下一市场交易日开盘"),
        ("成本", "买卖0.2%滑点、佣金、过户费；卖出另计印花税"),
        ("生命周期", "逐周报告累计最大浮盈、最大不利波动、收盘净收益及相对前一周增量；不预设N越小必须越早卖"),
        ("同股同轮匹配", f"同股票、信号相差不超过{PAIR_MAX_CALENDAR_DAYS}个日历日，按日期最近贪心一对一匹配；禁止一条慢参数信号重复配对"),
        ("匹配买价", "各参数均以自身信号完整周后的下一市场交易日开盘价比较，不使用信号周收盘价替代"),
        ("高点周", "在各自买入日起W12路径内寻找最高日线high，并换算为第1至12持有周；高点不是实际卖出价"),
        ("先后顺序", "同一天同时触及止盈和-10%时保守计为-10%先；W8和W12分别独立计算"),
        ("本版边界", "只识别赢家/失败者通常何时分化，不在本版根据回测结果自动拟合或执行卖点"),
        ("运行环境", f"Streamlit {streamlit_version}；运行稳定版要求requirements锁定1.61.0"),
    ], columns=["项目", "说明"])
    files = {
        "01_run_summary_v5_5.csv": run_summary,
        "02_w1_w12_lifecycle_curve_v5_5.csv": lifecycle,
        "03_holding_window_summary_v5_5.csv": holding_summary,
        "04_early_winner_failure_divergence_v5_5.csv": early_failure,
        "05_n6_n7_n9_matched_pair_summary_v5_5.csv": pair_summary,
        "06_n6_n7_n9_matched_pair_detail_v5_5.csv": pair_detail,
        "07_n6_n7_n9_same_cycle_triples_v5_5.csv": triple_detail,
        "08_v54_frozen_three_scheme_rank_outcomes_v5_5.csv": rank_cohorts,
        "09_v54_three_scheme_random_top3_v5_5.csv": random_benchmark,
        "10_v54_three_scheme_concentration_stress_v5_5.csv": concentration,
        "11_v54_three_scheme_rank_overlap_v5_5.csv": rank_overlap,
        "12_weekly_rank_calendar_v5_5.csv": ranked_calendar,
        "13_within_week_rank_ic_v5_5.csv": within_week_ic,
        "14_v54_original_score_rules_frozen.csv": score_rules,
        "15_v54_challenger_rules_frozen.csv": challenger_rules,
        "16_all_w8_ranking_candidates_v5_5.csv": eligible,
        "17_all_w12_lifecycle_candidates_v5_5.csv": lifecycle_eligible,
        "18_rejection_audit_v5_5.csv": rejection_audit,
        "19_metadata_v5_5.csv": metadata,
        "20_api_errors_v5_5.csv": pd.DataFrame({"错误": API_ERRORS}),
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
        f"完成：N=6/7/9的W12生命周期样本分别为"
        f"{len(lifecycle_eligible[lifecycle_eligible['SKDJ_N'].eq(6)])}/"
        f"{len(lifecycle_eligible[lifecycle_eligible['SKDJ_N'].eq(7)])}/"
        f"{len(lifecycle_eligible[lifecycle_eligible['SKDJ_N'].eq(9)])}个；"
        f"实际行情约{round((market_end_date - data_start_date).days / 7, 1)}周；"
        f"结果{'已保存' if persisted else '仅当前页面可下载'}。")
    st.subheader("N=6 / N=7 / N=9运行与双成熟样本摘要")
    render_plain_table(run_summary)
    st.subheader("W1～W12生命周期摘要（全部区间）")
    render_plain_table(holding_summary[
        holding_summary["时间分段"].eq("全部区间")])
    st.subheader("同股同轮：快参数相对慢参数究竟提前多少")
    render_plain_table(pair_summary)
    st.subheader("赢家与失败者在W1～W4的早期差异")
    render_plain_table(early_failure[
        early_failure["时间分段"].eq("全部区间")])
    st.caption("完整逐周曲线、前后段、年度、匹配明细及冻结的V5.4排名复现均在ZIP中。")
    render_download(result_zip, result_name, f"v55_current_{request_signature}")


if __name__ == "__main__":
    main()
