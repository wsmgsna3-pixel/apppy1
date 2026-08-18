# -*- coding: utf-8 -*-
"""科技股周线SKDJ上穿25直接收益排序与三仓审计 V4.6。

本版恢复较早的可执行买点：周线K首次从25下方上穿25且K>D、K上升，
下一市场交易日开盘买入。模型不再把下一周是否继续强分离当作最终目标，
而是直接预测买入后固定W5净收益（训练标签缩尾至-15%～+20%），并辅以
同周收益百分位和W5盈利概率。每个目标年度只使用此前三年已经完整发生的
历史结果训练，再验证Top1/3/5、随机基准、稳健性和真实三仓占位组合。
"""
from __future__ import annotations

import io
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
TITLE = "科技股周线SKDJ直接收益排序与真实三仓审计 V4.6"
VERSION = "V4.6-WEEKLY-SKDJ-DIRECT-RETURN-RANK"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")

SKDJ_N = 9
SKDJ_M = 3
SKDJ_BOTTOM = 25.0
INDICATOR_WARMUP_WEEKS = 40
AUDIT_WEEKS = 8
HISTORY_PEAK_LEVELS = (65.0, 70.0, 75.0)
FIRST_HIT_PROFIT_LEVELS = (10.0, 15.0, 20.0)
STOP_LOSS_PCT = 10.0
TAKE_PROFITS = (10.0, 15.0, 20.0)
ACTIVATED_TRAILS = ((10.0, 10.0), (15.0, 10.0))
TOP_K = 3
TOP_KS = (1, 3, 5)
MODEL_LOOKBACK_YEARS = 3
MODEL_MIN_TRAIN = 240
MODEL_L2 = 2.0
RETURN_MODEL_L2 = 8.0
RETURN_TARGET_FLOOR = -15.0
RETURN_TARGET_CAP = 20.0
RANDOM_TRIALS = 500
PORTFOLIO_RANDOM_TRIALS = 300
MODEL_FEATURES = (
    "Preconfirm_K", "Preconfirm_D", "Preconfirm_KD_Spread",
    "Preconfirm_K_Change_1W", "Preconfirm_D_Change_1W",
    "Wait_Weeks_To_Preconfirm", "Low_Cross_Gap_To25", "Bottom_Min_Level",
    "Weeks_Both_Below25", "Bottom_Golden_Cross_Count",
    "Prior_Swings_Available", "Prior_3_Peak_K_Mean", "Prior_3_Peak_K_Min",
    "Prior_3_Count_Peak_GE65", "Prior_3_Count_Peak_GE70",
    "Prior_3_Count_Peak_GE75", "Weekly_MA20_Bias_pct",
    "Weekly_MA20_Slope_4W_pct", "Weekly_Return_12W_pct",
    "Daily_MA60_Bias_pct", "Circ_MV_Billion", "Turnover_Rate",
    "Price_Change_Cross_to_Preconfirm_pct",
)
DIRECT_MODEL_FEATURES = MODEL_FEATURES + ("Model_Predicted_W1_Feature",)

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


def record_error(message: str) -> None:
    if len(API_ERRORS) < 300:
        API_ERRORS.append(message)


def validate_dates(signal_start: date, signal_end: date, market_end: date) -> str:
    if signal_start >= signal_end:
        return "信号开始日期必须早于信号截止日期"
    if market_end <= signal_end:
        return "行情观察截止日期必须晚于信号截止日期"
    return ""


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
    temp = f"{path}.tmp"
    with open(temp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(files: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in files.items():
            archive.writestr(name, csv_bytes(frame))
    return buffer.getvalue()


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
    l1, l2, l3 = str(row.get("l1_name", "")), str(row.get("l2_name", "")), str(row.get("l3_name", ""))
    if l1 in CORE_TECH_L1:
        return True
    return l1 in EXTENDED_TECH_L1 and any(word in f"{l2}|{l3}" for word in TECH_INDUSTRY_KEYWORDS)


@st.cache_data(ttl=7 * 24 * 3600)
def load_tech_memberships(api_pause: float) -> pd.DataFrame:
    levels = safe_get("index_classify", required=True, level="L1", src="SW2021")
    targets = levels[levels["industry_name"].isin(CORE_TECH_L1 | EXTENDED_TECH_L1)]
    if targets.empty:
        raise RuntimeError("未找到申万2021目标行业")
    frames = []
    jobs = [(str(row.index_code), str(row.industry_name), flag)
            for row in targets.itertuples(index=False) for flag in ("Y", "N")]
    progress = st.progress(0.0, text="构建申万历史科技池...")
    for number, (code, name, flag) in enumerate(jobs, start=1):
        frame = safe_get("index_member_all", l1_code=code, is_new=flag)
        if not frame.empty:
            if "ts_code" not in frame.columns and "con_code" in frame.columns:
                frame = frame.rename(columns={"con_code": "ts_code"})
            frames.append(frame)
        progress.progress(number / max(len(jobs), 1), text=f"行业池：{name} {flag}")
        time.sleep(api_pause)
    progress.empty()
    if not frames:
        raise RuntimeError("index_member_all未返回数据，请检查权限与SDK版本")
    result = pd.concat(frames, ignore_index=True)
    for column in ("ts_code", "l1_name", "l2_name", "l3_name", "in_date", "out_date"):
        if column not in result.columns:
            result[column] = ""
    result = result[result.apply(is_tech_industry, axis=1)].copy()
    result["in_date"] = result["in_date"].map(lambda x: normalize_date(x, "19000101"))
    result["out_date"] = result["out_date"].map(lambda x: normalize_date(x, "99991231"))
    return result.drop_duplicates(["ts_code", "l1_name", "l2_name", "l3_name", "in_date", "out_date"])


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


def sample_board(row: pd.Series) -> str:
    market = str(row.get("market", ""))
    if market in BOARDS:
        return market
    code = str(row.get("ts_code", ""))
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith(("688", "689")):
        return "科创板"
    return "主板"


@st.cache_data(ttl=24 * 3600)
def load_trade_calendar(start_date: str, end_date: str) -> list[str]:
    frame = safe_get("trade_cal", required=True, exchange="SSE", start_date=start_date, end_date=end_date)
    if frame.empty:
        raise RuntimeError("交易日历为空")
    return sorted(frame.loc[frame["is_open"].eq(1), "cal_date"].astype(str).tolist())


@st.cache_data(ttl=24 * 3600)
def stock_cache_path(ts_code: str, start_date: str, end_date: str) -> str:
    return os.path.join(CACHE_DIR, f"{ts_code.replace('.', '_')}_{start_date}_{end_date}.pkl")


def fetch_pro_bar(ts_code: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    last_error = None
    for attempt in range(retries):
        try:
            frame = ts.pro_bar(api=pro, ts_code=ts_code, start_date=start_date, end_date=end_date,
                               adj="qfq", freq="D", factors=["tor"])
            return pd.DataFrame() if frame is None else frame
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    record_error(f"pro_bar {ts_code}失败: {last_error}")
    return pd.DataFrame()


def fetch_stock_history(ts_code: str, start_date: str, end_date: str,
                        use_cache: bool, api_pause: float) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    path = stock_cache_path(ts_code, start_date, end_date)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            return payload.get("daily", pd.DataFrame()), payload.get("basic", pd.DataFrame()), True
        except Exception as exc:
            record_error(f"缓存损坏 {ts_code}: {exc}")
    daily = fetch_pro_bar(ts_code, start_date, end_date)
    time.sleep(api_pause)
    basic = safe_get("daily_basic", ts_code=ts_code, start_date=start_date, end_date=end_date,
                     fields="ts_code,trade_date,close,circ_mv,turnover_rate")
    time.sleep(api_pause)
    if not daily.empty:
        for column in ("open", "high", "low", "close", "vol"):
            daily[column] = pd.to_numeric(daily.get(column), errors="coerce")
        daily["trade_date"] = daily["trade_date"].astype(str)
        daily = daily.dropna(subset=["trade_date", "open", "high", "low", "close"])
        daily = daily.drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)
    if not basic.empty:
        basic["trade_date"] = basic["trade_date"].astype(str)
        for column in ("close", "circ_mv", "turnover_rate"):
            basic[column] = pd.to_numeric(basic.get(column), errors="coerce")
        basic = basic.drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)
    if use_cache and not daily.empty:
        atomic_pickle({"daily": daily, "basic": basic}, path)
    return daily, basic, False


def complete_week_last_dates(open_dates: list[str]) -> dict[pd.Timestamp, str]:
    frame = pd.DataFrame({"trade_date": open_dates})
    frame["dt"] = pd.to_datetime(frame["trade_date"])
    frame["week_label"] = frame["dt"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    return frame.groupby("week_label")["trade_date"].max().to_dict()


def add_skdj(frame: pd.DataFrame, n: int = SKDJ_N, m: int = SKDJ_M) -> pd.DataFrame:
    work = frame.copy()
    lowv = work["low"].rolling(int(n), min_periods=int(n)).min()
    highv = work["high"].rolling(int(n), min_periods=int(n)).max()
    raw = (work["close"] - lowv) / (highv - lowv).replace(0, np.nan) * 100.0
    rsv = raw.ewm(span=int(m), adjust=False, min_periods=1).mean()
    work["SKDJ_K"] = rsv.ewm(span=int(m), adjust=False, min_periods=1).mean()
    work["SKDJ_D"] = work["SKDJ_K"].rolling(int(m), min_periods=int(m)).mean()
    work["SKDJ_Golden_Cross"] = work["SKDJ_K"].gt(work["SKDJ_D"]) & work["SKDJ_K"].shift(1).le(work["SKDJ_D"].shift(1))
    work["SKDJ_Death_Cross"] = work["SKDJ_K"].lt(work["SKDJ_D"]) & work["SKDJ_K"].shift(1).ge(work["SKDJ_D"].shift(1))
    work["SKDJ_Level"] = (work["SKDJ_K"] + work["SKDJ_D"]) / 2.0
    work["SKDJ_KD_Spread"] = work["SKDJ_K"] - work["SKDJ_D"]
    return work


def aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["dt"] = pd.to_datetime(work["trade_date"])
    return work.set_index("dt").resample("W-FRI").agg({
        "trade_date": "last", "open": "first", "high": "max", "low": "min",
        "close": "last", "vol": "sum",
    }).dropna(subset=["close"]).reset_index().rename(columns={"dt": "week_label"})


def build_complete_weekly(daily: pd.DataFrame, week_last_map: dict[pd.Timestamp, str]) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    weekly = aggregate_weekly(daily)
    weekly["calendar_week_last"] = weekly["week_label"].map(week_last_map)
    weekly = weekly[
        weekly["calendar_week_last"].notna()
        & weekly["trade_date"].astype(str).eq(weekly["calendar_week_last"].astype(str))
    ].copy().reset_index(drop=True)
    return add_skdj(weekly) if not weekly.empty else weekly


def add_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy().sort_values("trade_date").reset_index(drop=True)
    work["D_MA60_Bias_pct"] = (work["close"] / work["close"].rolling(60).mean() - 1.0) * 100.0
    return work


def market_snapshot(basic: pd.DataFrame, signal_date: str) -> dict[str, float]:
    row = basic[basic["trade_date"].astype(str).eq(signal_date)] if not basic.empty else pd.DataFrame()
    if row.empty:
        return {"Raw_Close": np.nan, "Circ_MV_Billion": np.nan, "Turnover_Rate": np.nan}
    item = row.iloc[-1]
    return {
        "Raw_Close": finite_num(item.get("close")),
        "Circ_MV_Billion": finite_num(item.get("circ_mv")) / 10000.0,
        "Turnover_Rate": finite_num(item.get("turnover_rate")),
    }


def signal_filter(snapshot: dict[str, float], min_price: float, min_mv: float) -> tuple[bool, str]:
    if not math.isfinite(snapshot["Raw_Close"]):
        return False, "缺少信号日原始收盘价"
    if snapshot["Raw_Close"] < min_price:
        return False, "低于最低股价"
    if not math.isfinite(snapshot["Circ_MV_Billion"]):
        return False, "缺少历史流通市值"
    if snapshot["Circ_MV_Billion"] < min_mv:
        return False, "低于最低流通市值"
    return True, ""


def is_main_board(ts_code: str) -> bool:
    return not str(ts_code).startswith(("300", "301", "688", "689"))


def market_week_sequence(open_dates: list[str]) -> list[tuple[pd.Period, str]]:
    frame = pd.DataFrame({"trade_date": open_dates})
    frame["period"] = pd.to_datetime(frame["trade_date"]).dt.to_period("W-FRI")
    return [(period, str(group["trade_date"].max())) for period, group in frame.groupby("period", sort=True)]


def prefix_keys(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def first_hit_label(path: pd.DataFrame, raw_entry: float, profit_pct: float) -> tuple[str, str]:
    upper, lower = raw_entry * (1.0 + profit_pct / 100.0), raw_entry * 0.90
    for row in path.itertuples(index=False):
        hit_up = finite_num(getattr(row, "high", np.nan)) >= upper
        hit_down = finite_num(getattr(row, "low", np.nan)) <= lower
        if hit_up and hit_down:
            return f"同日同时触发_保守按-10%先", str(getattr(row, "trade_date", ""))
        if hit_down:
            return "先到-10%", str(getattr(row, "trade_date", ""))
        if hit_up:
            return f"先到+{int(profit_pct)}%", str(getattr(row, "trade_date", ""))
    return f"W{AUDIT_WEEKS}内均未触发", ""


def entry_outcomes(daily: pd.DataFrame, signal_date: str, ts_code: str,
                   open_dates: list[str], open_pos: dict[str, int],
                   market_weeks: list[tuple[pd.Period, str]],
                   config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "Tradable": False, "Reason": "", "Entry_Date": "", "Raw_Entry_Open": np.nan,
        "Net_Entry_Price": np.nan,
        "First_Hit_10_vs_Minus10_W8": "", "First_Hit_10_Date_W8": "",
        "First_Hit_15_vs_Minus10_W8": "", "First_Hit_15_Date_W8": "",
        "First_Hit_20_vs_Minus10_W8": "", "First_Hit_20_Date_W8": "",
    }
    for week in range(1, AUDIT_WEEKS + 1):
        out.update({
            f"Has_W{week}": False, f"W{week}_End_Date": "",
            f"W{week}_Cum_Max_High_Raw_pct": np.nan,
            f"W{week}_Cum_MFE_Net_pct": np.nan,
            f"W{week}_Cum_MAE_Raw_pct": np.nan,
            f"W{week}_Close_Return_Net_pct": np.nan,
        })
    if signal_date not in open_pos or open_pos[signal_date] + 1 >= len(open_dates):
        out["Reason"] = "未来市场交易日不足"
        return out
    entry_market_pos = open_pos[signal_date] + 1
    entry_date = open_dates[entry_market_pos]
    out["Entry_Date"] = entry_date
    rows = daily[daily["trade_date"].astype(str).eq(entry_date)]
    if rows.empty:
        out["Reason"] = "下一市场交易日停牌或无行情"
        return out
    first = rows.iloc[-1]
    if is_main_board(ts_code) and float(first["open"]) == float(first["high"]) == float(first["low"]):
        out["Reason"] = "主板下一交易日一字板"
        return out
    raw_entry = float(first["open"])
    if not math.isfinite(raw_entry) or raw_entry <= 0:
        out["Reason"] = "开盘价无效"
        return out
    buy_cost = (config["commission_pct"] + config["transfer_fee_pct"]) / 100.0
    sell_cost = (config["commission_pct"] + config["transfer_fee_pct"] + config["stamp_duty_pct"]) / 100.0
    buy_factor = (1 + config["buy_slippage_pct"] / 100.0) * (1 + buy_cost)
    sell_factor = (1 - config["sell_slippage_pct"] / 100.0) * (1 - sell_cost)
    net_entry = raw_entry * buy_factor
    out.update({"Tradable": True, "Raw_Entry_Open": raw_entry, "Net_Entry_Price": net_entry})

    entry_period = pd.Timestamp(entry_date).to_period("W-FRI")
    future_weeks = [(period, end_date) for period, end_date in market_weeks if period >= entry_period]
    for week in range(1, AUDIT_WEEKS + 1):
        if len(future_weeks) < week:
            continue
        end_date = future_weeks[week - 1][1]
        path = daily[daily["trade_date"].astype(str).between(entry_date, end_date)].sort_values("trade_date")
        if path.empty:
            continue
        high, low, close = float(path["high"].max()), float(path["low"].min()), float(path.iloc[-1]["close"])
        out.update({
            f"Has_W{week}": True, f"W{week}_End_Date": end_date,
            f"W{week}_Cum_Max_High_Raw_pct": (high / raw_entry - 1.0) * 100.0,
            f"W{week}_Cum_MFE_Net_pct": (high * sell_factor / net_entry - 1.0) * 100.0,
            f"W{week}_Cum_MAE_Raw_pct": (low / raw_entry - 1.0) * 100.0,
            f"W{week}_Close_Return_Net_pct": (close * sell_factor / net_entry - 1.0) * 100.0,
        })
        if week == AUDIT_WEEKS:
            for profit_pct in FIRST_HIT_PROFIT_LEVELS:
                label, hit_date = first_hit_label(path, raw_entry, profit_pct)
                key = int(profit_pct)
                out[f"First_Hit_{key}_vs_Minus10_W8"] = label
                out[f"First_Hit_{key}_Date_W8"] = hit_date

    if not out[f"Has_W{AUDIT_WEEKS}"]:
        out["Reason"] = f"可买但未来不足{AUDIT_WEEKS}个完整市场周"
    return out


def trade_factors(config: dict[str, Any]) -> tuple[float, float]:
    """Return buy/sell multipliers including slippage and explicit fees."""
    buy_cost = (config["commission_pct"] + config["transfer_fee_pct"]) / 100.0
    sell_cost = (
        config["commission_pct"] + config["transfer_fee_pct"]
        + config["stamp_duty_pct"]
    ) / 100.0
    return (
        (1.0 + config["buy_slippage_pct"] / 100.0) * (1.0 + buy_cost),
        (1.0 - config["sell_slippage_pct"] / 100.0) * (1.0 - sell_cost),
    )


def exit_fields(path: pd.DataFrame, raw_entry: float, raw_exit: float,
                exit_date: str, trigger: str, config: dict[str, Any]) -> dict[str, Any]:
    buy_factor, sell_factor = trade_factors(config)
    net_return = (raw_exit * sell_factor / (raw_entry * buy_factor) - 1.0) * 100.0
    holding_days = 0
    if not path.empty:
        dates = path["trade_date"].astype(str).tolist()
        holding_days = dates.index(exit_date) + 1 if exit_date in dates else len(dates)
    return {
        "Available": True, "Exit_Date": exit_date, "Raw_Exit_Price": raw_exit,
        "Trigger": trigger, "Holding_Trading_Days": holding_days,
        "Net_Return_pct": net_return,
    }


def unavailable_exit(reason: str) -> dict[str, Any]:
    return {
        "Available": False, "Exit_Date": "", "Raw_Exit_Price": np.nan,
        "Trigger": reason, "Holding_Trading_Days": np.nan,
        "Net_Return_pct": np.nan,
    }


def simulate_bracket(path: pd.DataFrame, raw_entry: float, take_profit: float,
                     config: dict[str, Any]) -> dict[str, Any]:
    """Fixed -10% stop and take-profit; same-day double hit is conservatively a stop."""
    if path.empty:
        return unavailable_exit("无W8路径")
    stop_price = raw_entry * (1.0 - STOP_LOSS_PCT / 100.0)
    target_price = raw_entry * (1.0 + take_profit / 100.0)
    for row in path.itertuples(index=False):
        trade_date = str(row.trade_date)
        day_open, day_low, day_high = float(row.open), float(row.low), float(row.high)
        stop_hit, target_hit = day_low <= stop_price, day_high >= target_price
        if stop_hit:
            raw_exit = day_open if day_open < stop_price else stop_price
            label = (
                f"同日双触发_保守止损-{int(STOP_LOSS_PCT)}%"
                if target_hit else f"止损-{int(STOP_LOSS_PCT)}%"
            )
            return exit_fields(path, raw_entry, raw_exit, trade_date, label, config)
        if target_hit:
            raw_exit = day_open if day_open > target_price else target_price
            return exit_fields(
                path, raw_entry, raw_exit, trade_date,
                f"止盈+{int(take_profit)}%", config)
    last = path.iloc[-1]
    return exit_fields(
        path, raw_entry, float(last["close"]), str(last["trade_date"]),
        f"W{AUDIT_WEEKS}期末", config)


def simulate_activated_trail(path: pd.DataFrame, raw_entry: float, activation: float,
                             trail: float, config: dict[str, Any]) -> dict[str, Any]:
    """Use a fixed -10% stop until activated, then trail from prior-day peak only."""
    if path.empty:
        return unavailable_exit("无W8路径")
    fixed_stop = raw_entry * (1.0 - STOP_LOSS_PCT / 100.0)
    activation_price = raw_entry * (1.0 + activation / 100.0)
    prior_peak = raw_entry
    armed = False
    for row in path.itertuples(index=False):
        trade_date = str(row.trade_date)
        day_open, day_low, day_high = float(row.open), float(row.low), float(row.high)
        if armed:
            stop_price = max(raw_entry, prior_peak * (1.0 - trail / 100.0))
        else:
            stop_price = fixed_stop
        if day_low <= stop_price:
            raw_exit = day_open if day_open < stop_price else stop_price
            trigger = (
                f"激活+{int(activation)}%后回撤{int(trail)}%"
                if armed else f"激活前止损-{int(STOP_LOSS_PCT)}%"
            )
            return exit_fields(path, raw_entry, raw_exit, trade_date, trigger, config)
        prior_peak = max(prior_peak, day_high)
        if prior_peak >= activation_price:
            armed = True
    last = path.iloc[-1]
    return exit_fields(
        path, raw_entry, float(last["close"]), str(last["trade_date"]),
        f"W{AUDIT_WEEKS}期末", config)


def simulate_exit_policies(daily: pd.DataFrame, outcome: dict[str, Any],
                           config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not to_bool(outcome.get("Tradable")):
        for policy in (
            "Fixed_W5", "Fixed_W8", "SL10_TP10", "SL10_TP15", "SL10_TP20",
            "Activate10_Trail10", "Activate15_Trail10",
        ):
            result.update(prefix_keys(unavailable_exit("不可交易"), policy))
        return result
    raw_entry = finite_num(outcome.get("Raw_Entry_Open"))
    entry_date = str(outcome.get("Entry_Date", ""))
    for week in (5, AUDIT_WEEKS):
        policy = f"Fixed_W{week}"
        if not to_bool(outcome.get(f"Has_W{week}")):
            result.update(prefix_keys(unavailable_exit(f"未来不足W{week}"), policy))
            continue
        end_date = str(outcome[f"W{week}_End_Date"])
        path = daily[daily["trade_date"].astype(str).between(entry_date, end_date)].sort_values("trade_date")
        last = path.iloc[-1]
        result.update(prefix_keys(exit_fields(
            path, raw_entry, float(last["close"]), str(last["trade_date"]),
            f"固定W{week}期末", config), policy))
    if not to_bool(outcome.get(f"Has_W{AUDIT_WEEKS}")):
        for policy in (
            "SL10_TP10", "SL10_TP15", "SL10_TP20",
            "Activate10_Trail10", "Activate15_Trail10",
        ):
            result.update(prefix_keys(unavailable_exit(f"未来不足W{AUDIT_WEEKS}"), policy))
        return result
    end_date = str(outcome[f"W{AUDIT_WEEKS}_End_Date"])
    path = daily[daily["trade_date"].astype(str).between(entry_date, end_date)].sort_values("trade_date")
    for take_profit in TAKE_PROFITS:
        policy = f"SL10_TP{int(take_profit)}"
        result.update(prefix_keys(
            simulate_bracket(path, raw_entry, take_profit, config), policy))
    for activation, trail in ACTIVATED_TRAILS:
        policy = f"Activate{int(activation)}_Trail{int(trail)}"
        result.update(prefix_keys(
            simulate_activated_trail(path, raw_entry, activation, trail, config), policy))
    return result


def completed_golden_swings(weekly: pd.DataFrame) -> list[dict[str, Any]]:
    """Return completed ordinary golden-cross swings using only cross-to-death data."""
    swings: list[dict[str, Any]] = []
    active_position: int | None = None
    for position in range(INDICATOR_WARMUP_WEEKS, len(weekly)):
        row = weekly.iloc[position]
        if active_position is None and to_bool(row.get("SKDJ_Golden_Cross")):
            active_position = position
            continue
        if active_position is None or not to_bool(row.get("SKDJ_Death_Cross")):
            continue
        segment = weekly.iloc[active_position:position + 1]
        peak_index = int(pd.to_numeric(segment["SKDJ_K"], errors="coerce").idxmax())
        swings.append({
            "Golden_Position": active_position,
            "Golden_Date": str(weekly.iloc[active_position]["trade_date"]),
            "Death_Position": position,
            "Death_Date": str(row["trade_date"]),
            "Peak_K": finite_num(segment["SKDJ_K"].max()),
            "Peak_D": finite_num(segment["SKDJ_D"].max()),
            "Peak_K_Date": str(weekly.loc[peak_index, "trade_date"]),
            "Swing_Weeks": int(position - active_position),
        })
        active_position = None
    return swings


def build_bottom_cycles(weekly: pd.DataFrame) -> list[dict[str, Any]]:
    """Collapse repeated crosses below 25 into one observable bottom cycle."""
    cycles: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for position in range(INDICATOR_WARMUP_WEEKS, len(weekly)):
        row = weekly.iloc[position]
        k, d = finite_num(row.get("SKDJ_K")), finite_num(row.get("SKDJ_D"))
        if not (math.isfinite(k) and math.isfinite(d)):
            continue
        low_cross = to_bool(row.get("SKDJ_Golden_Cross")) and k <= SKDJ_BOTTOM and d <= SKDJ_BOTTOM
        if active is None:
            if not low_cross:
                continue
            active = {
                "Anchor_Position": position,
                "Trigger_Position": None,
                "Bottom_Min_K": k,
                "Bottom_Min_D": d,
                "Bottom_Min_Level": (k + d) / 2.0,
                "Weeks_Both_Below25": 1,
                "Bottom_Golden_Cross_Count": 1,
            }
            continue

        active["Bottom_Min_K"] = min(float(active["Bottom_Min_K"]), k)
        active["Bottom_Min_D"] = min(float(active["Bottom_Min_D"]), d)
        active["Bottom_Min_Level"] = min(float(active["Bottom_Min_Level"]), (k + d) / 2.0)
        if k <= SKDJ_BOTTOM and d <= SKDJ_BOTTOM:
            active["Weeks_Both_Below25"] = int(active["Weeks_Both_Below25"]) + 1
        if low_cross:
            active["Bottom_Golden_Cross_Count"] = int(active["Bottom_Golden_Cross_Count"]) + 1

        previous = weekly.iloc[position - 1]
        previous_k = finite_num(previous.get("SKDJ_K"))
        crossed_25 = math.isfinite(previous_k) and previous_k < SKDJ_BOTTOM <= k
        confirmed = k > d and k > previous_k
        if crossed_25 and confirmed:
            active["Trigger_Position"] = position
            cycles.append(active)
            active = None

    if active is not None:
        cycles.append(active)
    return cycles


def prior_swing_features(swings: list[dict[str, Any]], anchor_position: int) -> dict[str, Any]:
    known = [swing for swing in swings if int(swing["Death_Position"]) < anchor_position]
    recent = list(reversed(known[-3:]))
    result: dict[str, Any] = {"Prior_Swings_Available": len(recent)}
    peaks: list[float] = []
    for number in range(1, 4):
        if number <= len(recent):
            swing = recent[number - 1]
            peak = finite_num(swing["Peak_K"])
            peaks.append(peak)
            result.update({
                f"Prev_Swing{number}_Golden_Date": swing["Golden_Date"],
                f"Prev_Swing{number}_Death_Date": swing["Death_Date"],
                f"Prev_Swing{number}_Peak_K": peak,
                f"Prev_Swing{number}_Peak_K_Date": swing["Peak_K_Date"],
                f"Prev_Swing{number}_Weeks": swing["Swing_Weeks"],
            })
        else:
            result.update({
                f"Prev_Swing{number}_Golden_Date": "",
                f"Prev_Swing{number}_Death_Date": "",
                f"Prev_Swing{number}_Peak_K": np.nan,
                f"Prev_Swing{number}_Peak_K_Date": "",
                f"Prev_Swing{number}_Weeks": np.nan,
            })
    valid_peaks = [peak for peak in peaks if math.isfinite(peak)]
    result["Prior_3_Peak_K_Mean"] = float(np.mean(valid_peaks)) if valid_peaks else np.nan
    result["Prior_3_Peak_K_Min"] = float(np.min(valid_peaks)) if valid_peaks else np.nan
    result["Prior_3_Peak_K_Max"] = float(np.max(valid_peaks)) if valid_peaks else np.nan
    for level in HISTORY_PEAK_LEVELS:
        result[f"Prior_3_Count_Peak_GE{int(level)}"] = sum(peak >= level for peak in valid_peaks)
        result[f"Prev_Swing1_Peak_GE{int(level)}"] = (
            bool(valid_peaks and valid_peaks[0] >= level) if recent else False)
    return result


def post_confirmation_features(weekly: pd.DataFrame, trigger_position: int) -> dict[str, Any]:
    """Build the next complete week's training label; never use it in live ranking."""
    result: dict[str, Any] = {}
    previous_position = trigger_position
    for offset in (1,):
        prefix = f"Post_Confirm_W{offset}"
        position = trigger_position + offset
        if position >= len(weekly):
            result.update({
                f"{prefix}_Available": False, f"{prefix}_Date": "",
                f"{prefix}_K": np.nan, f"{prefix}_D": np.nan,
                f"{prefix}_KD_Spread": np.nan, f"{prefix}_Spread_Change": np.nan,
                f"{prefix}_K_Change": np.nan, f"{prefix}_D_Change": np.nan,
                f"{prefix}_K_Above25": False, f"{prefix}_K_Above_D": False,
                f"{prefix}_Both_Rising": False, f"{prefix}_Spread_Widening": False,
                f"{prefix}_Strong_Separation": False,
            })
            continue
        current, previous = weekly.iloc[position], weekly.iloc[previous_position]
        k, d = finite_num(current["SKDJ_K"]), finite_num(current["SKDJ_D"])
        prior_k, prior_d = finite_num(previous["SKDJ_K"]), finite_num(previous["SKDJ_D"])
        spread, prior_spread = k - d, prior_k - prior_d
        both_rising = k > prior_k and d > prior_d
        widening = spread > prior_spread
        result.update({
            f"{prefix}_Available": True, f"{prefix}_Date": str(current["trade_date"]),
            f"{prefix}_K": k, f"{prefix}_D": d,
            f"{prefix}_KD_Spread": spread, f"{prefix}_Spread_Change": spread - prior_spread,
            f"{prefix}_K_Change": k - prior_k, f"{prefix}_D_Change": d - prior_d,
            f"{prefix}_K_Above25": k >= SKDJ_BOTTOM, f"{prefix}_K_Above_D": k > d,
            f"{prefix}_Both_Rising": both_rising, f"{prefix}_Spread_Widening": widening,
            f"{prefix}_Strong_Separation": bool(k >= SKDJ_BOTTOM and k > d and both_rising and widening),
        })
        previous_position = position
    return result


def stock_trend(weekly: pd.DataFrame, position: int, daily: pd.DataFrame, signal_date: str) -> dict[str, Any]:
    close = weekly["close"]
    ma20 = close.rolling(20).mean()
    bias = finite_num((close / ma20 - 1.0).iloc[position] * 100.0)
    slope = finite_num((ma20 / ma20.shift(4) - 1.0).iloc[position] * 100.0)
    ret12 = finite_num(close.pct_change(12, fill_method=None).iloc[position] * 100.0)
    history = daily[daily["trade_date"].astype(str).le(signal_date)]
    daily_bias = finite_num(history.iloc[-1].get("D_MA60_Bias_pct")) if not history.empty else np.nan
    values = [bias, slope, ret12, daily_bias]
    if all(math.isfinite(value) and value > 0 for value in values):
        state = "上涨"
    elif all(math.isfinite(value) and value < 0 for value in values):
        state = "下跌"
    else:
        state = "震荡/过渡"
    return {
        "Individual_Trend": state, "Weekly_MA20_Bias_pct": bias,
        "Weekly_MA20_Slope_4W_pct": slope, "Weekly_Return_12W_pct": ret12,
        "Daily_MA60_Bias_pct": daily_bias,
    }


def analyze_stock(stock: pd.Series, periods: list[dict[str, str]], daily_raw: pd.DataFrame,
                  daily_basic: pd.DataFrame, week_last_map: dict[pd.Timestamp, str],
                  open_dates: list[str], open_pos: dict[str, int],
                  market_weeks: list[tuple[pd.Period, str]], config: dict[str, Any]
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    weekly = build_complete_weekly(daily_raw, week_last_map)
    if len(weekly) < INDICATOR_WARMUP_WEEKS:
        config["rejects"]["周线不足"] = config["rejects"].get("周线不足", 0) + 1
        return [], [], []
    daily = add_daily_features(daily_raw)
    cycles = build_bottom_cycles(weekly)
    swings = completed_golden_swings(weekly)
    cycle_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    live_events: list[dict[str, Any]] = []
    code, board = str(stock["ts_code"]), sample_board(stock)

    for cycle_number, cycle in enumerate(cycles, start=1):
        anchor_position = int(cycle["Anchor_Position"])
        trigger_position = cycle.get("Trigger_Position")
        anchor = weekly.iloc[anchor_position]
        anchor_date = str(anchor["trade_date"])
        if anchor_date > config["signal_end"]:
            continue
        anchor_snapshot = market_snapshot(daily_basic, anchor_date)
        anchor_membership = membership_on_date(periods, anchor_date)
        anchor_passed, anchor_reason = signal_filter(
            anchor_snapshot, config["min_price"], config["min_mv"])
        anchor_listed = str(stock["list_date"]) <= anchor_date < str(stock["delist_date"])
        anchor_eligible = bool(anchor_membership is not None and anchor_listed and anchor_passed)
        reject_reason = ""
        if not anchor_eligible:
            reject_reason = anchor_reason or (
                "当时不在历史科技池" if anchor_membership is None else "当时未上市或已退市")
        history_features = prior_swing_features(swings, anchor_position)
        low_level = (float(anchor["SKDJ_K"]) + float(anchor["SKDJ_D"])) / 2.0
        trigger_date = (
            str(weekly.iloc[int(trigger_position)]["trade_date"])
            if trigger_position is not None else "")
        wait_weeks = (
            int(trigger_position) - anchor_position
            if trigger_position is not None else np.nan)
        post = (
            post_confirmation_features(weekly, int(trigger_position))
            if trigger_position is not None else {})
        separation_date = str(post.get("Post_Confirm_W1_Date", ""))

        if config["model_start"] <= anchor_date <= config["signal_end"]:
            cycle_rows.append({
                "ts_code": code, "name": str(stock["name"]), "Sample_Board": board,
                "Bottom_Cycle_Number": cycle_number,
                "Low_Cross_Date": anchor_date, "Low_Cross_K": float(anchor["SKDJ_K"]),
                "Low_Cross_D": float(anchor["SKDJ_D"]), "Low_Cross_Level": low_level,
                "Low_Cross_Gap_To25": SKDJ_BOTTOM - low_level,
                "Bottom_Min_K": cycle["Bottom_Min_K"], "Bottom_Min_D": cycle["Bottom_Min_D"],
                "Bottom_Min_Level": cycle["Bottom_Min_Level"],
                "Bottom_Max_Depth_From25": SKDJ_BOTTOM - float(cycle["Bottom_Min_Level"]),
                "Weeks_Both_Below25": cycle["Weeks_Both_Below25"],
                "Bottom_Golden_Cross_Count": cycle["Bottom_Golden_Cross_Count"],
                "Preconfirmed": trigger_position is not None,
                "Preconfirm_Date": trigger_date, "Wait_Weeks_To_Preconfirm": wait_weeks,
                "Future_W1_Check_Date": separation_date,
                "Future_W1_Strong_Separation": to_bool(
                    post.get("Post_Confirm_W1_Strong_Separation")),
                "Observation_Ended_At": str(weekly.iloc[-1]["trade_date"]),
                "Eligible_Low_Cross_Pool": anchor_eligible,
                "Low_Cross_Filter_Reason": reject_reason,
                **anchor_snapshot, **history_features,
            })

        if trigger_position is None or not separation_date:
            continue
        if not (config["model_start"] <= trigger_date <= config["signal_end"]):
            continue
        trigger_position_int = int(trigger_position)
        trigger = weekly.iloc[trigger_position_int]
        previous = weekly.iloc[trigger_position_int - 1]
        trigger_snapshot = market_snapshot(daily_basic, trigger_date)
        trigger_membership = membership_on_date(periods, trigger_date)
        trigger_passed, trigger_reason = signal_filter(
            trigger_snapshot, config["min_price"], config["min_mv"])
        trigger_listed = str(stock["list_date"]) <= trigger_date < str(stock["delist_date"])
        eligible = bool(
            anchor_eligible and trigger_membership is not None and trigger_listed and trigger_passed)
        filter_reason = ""
        if not eligible:
            filter_reason = reject_reason or trigger_reason or (
                "上穿25日不在历史科技池" if trigger_membership is None else "上穿25日上市状态无效")
        trend_features = stock_trend(
            weekly, trigger_position_int, daily, trigger_date)
        price_change = (
            (trigger_snapshot["Raw_Close"] / anchor_snapshot["Raw_Close"] - 1.0) * 100.0
            if anchor_snapshot["Raw_Close"] > 0 else np.nan)
        row: dict[str, Any] = {
            "ts_code": code, "name": str(stock["name"]), "Sample_Board": board,
            "SW_L1": trigger_membership["l1"] if trigger_membership else "",
            "SW_L2": trigger_membership["l2"] if trigger_membership else "",
            "SW_L3": trigger_membership["l3"] if trigger_membership else "",
            "Signal_Date": trigger_date, "Preconfirm_Date": trigger_date,
            "Low_Cross_Date": anchor_date, "Future_W1_Check_Date": separation_date,
            "Wait_Weeks_To_Preconfirm": wait_weeks,
            "Low_Cross_K": float(anchor["SKDJ_K"]), "Low_Cross_D": float(anchor["SKDJ_D"]),
            "Low_Cross_Level": low_level, "Low_Cross_Gap_To25": SKDJ_BOTTOM - low_level,
            "Bottom_Min_K": cycle["Bottom_Min_K"], "Bottom_Min_D": cycle["Bottom_Min_D"],
            "Bottom_Min_Level": cycle["Bottom_Min_Level"],
            "Weeks_Both_Below25": cycle["Weeks_Both_Below25"],
            "Bottom_Golden_Cross_Count": cycle["Bottom_Golden_Cross_Count"],
            "Preconfirm_K": float(trigger["SKDJ_K"]),
            "Preconfirm_D": float(trigger["SKDJ_D"]),
            "Preconfirm_KD_Spread": float(trigger["SKDJ_KD_Spread"]),
            "Preconfirm_K_Change_1W": float(trigger["SKDJ_K"] - previous["SKDJ_K"]),
            "Preconfirm_D_Change_1W": float(trigger["SKDJ_D"] - previous["SKDJ_D"]),
            "Low_Cross_Raw_Close": anchor_snapshot["Raw_Close"],
            "Signal_Raw_Close": trigger_snapshot["Raw_Close"],
            "Circ_MV_Billion": trigger_snapshot["Circ_MV_Billion"],
            "Turnover_Rate": trigger_snapshot["Turnover_Rate"],
            "Price_Change_Cross_to_Preconfirm_pct": price_change,
            "Eligible_Preliminary": eligible,
            "Preliminary_Filter_Reason": filter_reason,
            "Future_W1_Available": to_bool(post.get("Post_Confirm_W1_Available")),
            "Future_W1_Strong_Separation": to_bool(
                post.get("Post_Confirm_W1_Strong_Separation")),
            "Future_W1_K": post.get("Post_Confirm_W1_K", np.nan),
            "Future_W1_D": post.get("Post_Confirm_W1_D", np.nan),
            "Future_W1_KD_Spread": post.get("Post_Confirm_W1_KD_Spread", np.nan),
            "Future_W1_K_Change": post.get("Post_Confirm_W1_K_Change", np.nan),
            "Future_W1_D_Change": post.get("Post_Confirm_W1_D_Change", np.nan),
            "Future_W1_Spread_Change": post.get("Post_Confirm_W1_Spread_Change", np.nan),
            **history_features, **trend_features,
        }
        # 历史训练事件也必须使用与目标期完全相同的下一交易日开盘买入口径。
        # 这里只计算当时已经发生的收益路径；年度训练时还会用W5结束日再次
        # 截断，保证目标年度看不到任何跨年后的结果。
        outcome: dict[str, Any] = {}
        if eligible:
            outcome = entry_outcomes(
                daily, trigger_date, code, open_dates, open_pos, market_weeks, config)
            row.update(prefix_keys(outcome, "Entry"))
        model_rows.append(row.copy())
        if not eligible:
            if config["signal_start"] <= trigger_date <= config["signal_end"]:
                key = f"上穿25:{filter_reason}"
                config["rejects"][key] = config["rejects"].get(key, 0) + 1
            continue
        if not (config["signal_start"] <= trigger_date <= config["signal_end"]):
            continue
        event = {
            **row,
            "Rule": "低位金叉后首次上穿25，次周开盘买",
            "Signal_Year": trigger_date[:4],
            "Period_Group": (
                "2025-06以后" if trigger_date >= config["split_date"] else "2025-06以前"),
        }
        event.update(prefix_keys(simulate_exit_policies(daily, outcome, config), "Entry"))
        live_events.append(event)
    return cycle_rows, model_rows, live_events


def max_empty_run(counts: pd.Series) -> int:
    longest = current = 0
    for value in counts.tolist():
        current = current + 1 if int(value) == 0 else 0
        longest = max(longest, current)
    return longest


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def true_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(to_bool)


def mature_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events[
        true_mask(events, "Entry_Tradable")
        & true_mask(events, f"Entry_Has_W{AUDIT_WEEKS}")
    ].copy()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def fit_logistic_model(train: pd.DataFrame, target_column: str,
                       features: tuple[str, ...] = MODEL_FEATURES) -> dict[str, Any]:
    raw = train.loc[:, features].apply(pd.to_numeric, errors="coerce")
    medians = raw.median(axis=0).fillna(0.0)
    filled = raw.fillna(medians)
    lower = filled.quantile(0.01).fillna(medians)
    upper = filled.quantile(0.99).fillna(medians)
    clipped = filled.clip(lower=lower, upper=upper, axis=1)
    means = clipped.mean(axis=0).fillna(0.0)
    stds = clipped.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    scaled = (clipped - means) / stds
    x = np.column_stack([np.ones(len(scaled)), scaled.to_numpy(dtype=float)])
    y = true_mask(train, target_column).astype(float).to_numpy()
    beta = np.zeros(x.shape[1], dtype=float)
    beta[0] = math.log((y.mean() + 1e-4) / (1.0 - y.mean() + 1e-4))
    penalty = np.eye(x.shape[1], dtype=float) * MODEL_L2
    penalty[0, 0] = 0.0
    for _ in range(60):
        probability = sigmoid(x @ beta)
        weight = np.maximum(probability * (1.0 - probability), 1e-6)
        gradient = x.T @ (probability - y) + penalty @ beta
        hessian = x.T @ (x * weight[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return {
        "beta": beta, "medians": medians, "lower": lower, "upper": upper,
        "means": means, "stds": stds, "base_rate": float(y.mean()),
        "features": features,
    }


def predict_logistic(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    raw = frame.loc[:, model["features"]].apply(pd.to_numeric, errors="coerce")
    filled = raw.fillna(model["medians"])
    clipped = filled.clip(lower=model["lower"], upper=model["upper"], axis=1)
    scaled = (clipped - model["means"]) / model["stds"]
    x = np.column_stack([np.ones(len(scaled)), scaled.to_numpy(dtype=float)])
    return sigmoid(x @ model["beta"])


def fit_ridge_model(train: pd.DataFrame, target_column: str,
                    features: tuple[str, ...] = DIRECT_MODEL_FEATURES) -> dict[str, Any]:
    raw = train.loc[:, features].apply(pd.to_numeric, errors="coerce")
    medians = raw.median(axis=0).fillna(0.0)
    filled = raw.fillna(medians)
    lower = filled.quantile(0.01).fillna(medians)
    upper = filled.quantile(0.99).fillna(medians)
    clipped = filled.clip(lower=lower, upper=upper, axis=1)
    means = clipped.mean(axis=0).fillna(0.0)
    stds = clipped.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    scaled = (clipped - means) / stds
    x = np.column_stack([np.ones(len(scaled)), scaled.to_numpy(dtype=float)])
    y = numeric(train, target_column).to_numpy(dtype=float)
    penalty = np.eye(x.shape[1], dtype=float) * RETURN_MODEL_L2
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(x.T @ x + penalty) @ (x.T @ y)
    return {
        "beta": beta, "medians": medians, "lower": lower, "upper": upper,
        "means": means, "stds": stds, "target_mean": float(np.mean(y)),
        "features": features,
    }


def predict_ridge(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    raw = frame.loc[:, model["features"]].apply(pd.to_numeric, errors="coerce")
    filled = raw.fillna(model["medians"])
    clipped = filled.clip(lower=model["lower"], upper=model["upper"], axis=1)
    scaled = (clipped - model["means"]) / model["stds"]
    x = np.column_stack([np.ones(len(scaled)), scaled.to_numpy(dtype=float)])
    return x @ model["beta"]


def add_direct_training_targets(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    actual = numeric(work, "Entry_W5_Close_Return_Net_pct")
    work["Direct_Target_Clipped_W5_pct"] = actual.clip(
        RETURN_TARGET_FLOOR, RETURN_TARGET_CAP)
    work["Direct_Target_W5_Positive"] = actual.gt(0)

    def percentile(values: pd.Series) -> pd.Series:
        valid = values.notna()
        result = pd.Series(np.nan, index=values.index, dtype=float)
        count = int(valid.sum())
        if count == 1:
            result.loc[valid] = 0.5
        elif count > 1:
            ranks = values.loc[valid].rank(method="average")
            result.loc[valid] = (ranks - 1.0) / (count - 1.0)
        return result

    work["Direct_Target_Week_Percentile"] = work.groupby(
        "Signal_Date", sort=False)["Direct_Target_Clipped_W5_pct"].transform(percentile)
    return work


def auc_score(labels: pd.Series, scores: pd.Series) -> float:
    data = pd.DataFrame({"y": labels.map(to_bool).astype(int), "s": pd.to_numeric(scores, errors="coerce")}).dropna()
    positives, negatives = int(data["y"].sum()), int((1 - data["y"]).sum())
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = data["s"].rank(method="average")
    return float((ranks[data["y"].eq(1)].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def apply_annual_oos_models(history: pd.DataFrame, events: pd.DataFrame
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = events.copy()
    work["OOS_Predicted_W1_Probability"] = np.nan
    work["OOS_Predicted_Clipped_W5_Return_pct"] = np.nan
    work["OOS_Predicted_Week_Percentile"] = np.nan
    work["OOS_Predicted_W5_Positive_Probability"] = np.nan
    work["OOS_Direct_Return_Score"] = np.nan
    work["Model_Train_Start"] = ""
    work["Model_Train_End"] = ""
    work["Model_Train_N"] = 0
    work["Model_Train_Base_Rate"] = np.nan
    work["Model_Train_W5_Mean"] = np.nan
    work["Model_Status"] = "未训练"
    history_targets = add_direct_training_targets(history)
    eligible_history = history_targets[
        true_mask(history_targets, "Eligible_Preliminary")
        & true_mask(history_targets, "Entry_Tradable")
        & true_mask(history_targets, "Entry_Has_W5")
        & numeric(history_targets, "Direct_Target_Clipped_W5_pct").notna()
    ].copy()
    separation_history = history_targets[
        true_mask(history_targets, "Eligible_Preliminary")
        & true_mask(history_targets, "Future_W1_Available")
    ].copy()
    coefficient_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for year in sorted(work["Signal_Year"].astype(str).unique()):
        year_number = int(year)
        train_start = f"{year_number - MODEL_LOOKBACK_YEARS}0101"
        train_end = f"{year_number - 1}1231"
        separation_train = separation_history[
            separation_history["Future_W1_Check_Date"].astype(str).between(
                train_start, train_end)
        ].copy()
        # 直接收益标签必须在训练截止日前已经完整结束；不能只凭信号日期截断。
        train = eligible_history[
            eligible_history["Entry_W5_End_Date"].astype(str).between(train_start, train_end)
        ].copy()
        target_index = work.index[work["Signal_Year"].astype(str).eq(year)]
        target_frame = work.loc[target_index].copy()

        if (len(separation_train) >= MODEL_MIN_TRAIN
                and true_mask(separation_train, "Future_W1_Strong_Separation").nunique() >= 2):
            separation_model = fit_logistic_model(
                separation_train, "Future_W1_Strong_Separation", MODEL_FEATURES)
            target_w1_probability = predict_logistic(target_frame, separation_model)
            train_w1_probability = predict_logistic(train, separation_model) if len(train) else np.array([])
            separation_status = "年度样本外Logistic"
            for feature, coefficient in zip(("截距",) + MODEL_FEATURES,
                                            separation_model["beta"]):
                coefficient_rows.append({
                    "目标年度": year, "模型": "辅助_W1持续分离Logistic",
                    "训练开始": train_start, "训练截止": train_end,
                    "训练样本": len(separation_train), "特征": feature,
                    "标准化系数": float(coefficient),
                })
            separation_base = separation_model["base_rate"]
        else:
            separation_base = (
                float(true_mask(separation_train, "Future_W1_Strong_Separation").mean())
                if len(separation_train) else 0.5)
            target_w1_probability = np.full(len(target_frame), separation_base)
            train_w1_probability = np.full(len(train), separation_base)
            separation_status = "W1样本不足_历史基准率"

        work.loc[target_index, "OOS_Predicted_W1_Probability"] = target_w1_probability
        target_frame["Model_Predicted_W1_Feature"] = target_w1_probability
        train["Model_Predicted_W1_Feature"] = train_w1_probability

        enough_return_samples = (
            len(train) >= MODEL_MIN_TRAIN
            and true_mask(train, "Direct_Target_W5_Positive").nunique() >= 2)
        if enough_return_samples:
            return_model = fit_ridge_model(
                train, "Direct_Target_Clipped_W5_pct", DIRECT_MODEL_FEATURES)
            percentile_model = fit_ridge_model(
                train, "Direct_Target_Week_Percentile", DIRECT_MODEL_FEATURES)
            positive_model = fit_logistic_model(
                train, "Direct_Target_W5_Positive", DIRECT_MODEL_FEATURES)
            predicted_return = np.clip(
                predict_ridge(target_frame, return_model),
                RETURN_TARGET_FLOOR, RETURN_TARGET_CAP)
            predicted_percentile = np.clip(
                predict_ridge(target_frame, percentile_model), 0.0, 1.0)
            predicted_positive = predict_logistic(target_frame, positive_model)
            status = "年度样本外_直接收益Ridge+盈利Logistic"
            for model_name, fitted in (
                ("主模型_W5缩尾收益Ridge", return_model),
                ("辅助_同周W5百分位Ridge", percentile_model),
                ("辅助_W5盈利Logistic", positive_model),
            ):
                for feature, coefficient in zip(("截距",) + DIRECT_MODEL_FEATURES,
                                                fitted["beta"]):
                    coefficient_rows.append({
                        "目标年度": year, "模型": model_name,
                        "训练开始": train_start, "训练截止": train_end,
                        "训练样本": len(train), "特征": feature,
                        "标准化系数": float(coefficient),
                    })
        else:
            predicted_return = np.full(
                len(target_frame), numeric(train, "Direct_Target_Clipped_W5_pct").mean()
                if len(train) else 0.0)
            predicted_percentile = np.full(
                len(target_frame), numeric(train, "Direct_Target_Week_Percentile").mean()
                if len(train) else 0.5)
            predicted_positive = np.full(
                len(target_frame), true_mask(train, "Direct_Target_W5_Positive").mean()
                if len(train) else 0.5)
            status = "收益样本不足_仅历史均值"

        work.loc[target_index, "OOS_Predicted_Clipped_W5_Return_pct"] = predicted_return
        work.loc[target_index, "OOS_Predicted_Week_Percentile"] = predicted_percentile
        work.loc[target_index, "OOS_Predicted_W5_Positive_Probability"] = predicted_positive
        work.loc[target_index, "OOS_Direct_Return_Score"] = predicted_return
        work.loc[target_index, "Model_Train_Start"] = train_start
        work.loc[target_index, "Model_Train_End"] = train_end
        work.loc[target_index, "Model_Train_N"] = len(train)
        work.loc[target_index, "Model_Train_Base_Rate"] = separation_base
        work.loc[target_index, "Model_Train_W5_Mean"] = numeric(
            train, "Direct_Target_Clipped_W5_pct").mean()
        work.loc[target_index, "Model_Status"] = status
        model_rows.append({
            "目标年度": year, "训练开始": train_start, "训练截止": train_end,
            "实际最早W5结束日": str(train["Entry_W5_End_Date"].min()) if len(train) else "",
            "实际最晚W5结束日": str(train["Entry_W5_End_Date"].max()) if len(train) else "",
            "直接收益训练样本": len(train),
            "历史W5缩尾平均收益%": numeric(train, "Direct_Target_Clipped_W5_pct").mean(),
            "历史W5盈利率%": true_mask(train, "Direct_Target_W5_Positive").mean() * 100 if len(train) else np.nan,
            "W1辅助训练样本": len(separation_train),
            "历史持续分离率%": separation_base * 100,
            "W1辅助模型状态": separation_status, "模型状态": status,
        })
    return work, pd.DataFrame(model_rows), pd.DataFrame(coefficient_rows)


def rank_same_week(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    trend_rank = work["Individual_Trend"].map({"上涨": 2, "震荡/过渡": 1, "下跌": 0}).fillna(0)
    work["Observable_Trend_Rank"] = trend_rank.astype(int)
    work = work.sort_values([
        "Signal_Date", "OOS_Direct_Return_Score",
        "OOS_Predicted_Week_Percentile",
        "OOS_Predicted_W5_Positive_Probability",
        "OOS_Predicted_W1_Probability",
        "Prior_3_Count_Peak_GE75", "Observable_Trend_Rank",
        "Prior_3_Count_Peak_GE70", "Prior_3_Count_Peak_GE65",
        "Preconfirm_KD_Spread", "ts_code",
    ], ascending=[True, False, False, False, False, False, False, False, False,
                  False, True], kind="mergesort")
    work["Same_Week_Rank"] = work.groupby("Signal_Date", sort=False).cumcount() + 1
    for top_k in TOP_KS:
        work[f"Selected_Top{top_k}"] = work["Same_Week_Rank"].le(top_k)
    return work.sort_values(["Signal_Date", "Same_Week_Rank", "ts_code"]).reset_index(drop=True)


EXIT_POLICIES = {
    "Fixed_W5": "固定持有5周", "Fixed_W8": "固定持有8周",
    "SL10_TP10": "止损10%+止盈10%", "SL10_TP15": "止损10%+止盈15%",
    "SL10_TP20": "止损10%+止盈20%",
    "Activate10_Trail10": "浮盈10%后回撤10%",
    "Activate15_Trail10": "浮盈15%后回撤10%",
}


def signal_week_calendar(open_dates: list[str], start: str, end: str,
                         events: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"trade_date": [day for day in open_dates if start <= day <= end]})
    frame["period"] = pd.to_datetime(frame["trade_date"]).dt.to_period("W-FRI")
    calendar = frame.groupby("period", as_index=False)["trade_date"].max().rename(
        columns={"trade_date": "Week_Last_Trade_Date"})
    for name, selected in [("All_Candidates", events)] + [
        (f"Top{top_k}", events[true_mask(events, f"Selected_Top{top_k}")])
        for top_k in TOP_KS
    ]:
        counts = selected.groupby("Signal_Date").size() if not selected.empty else pd.Series(dtype=int)
        calendar[name] = calendar["Week_Last_Trade_Date"].map(counts).fillna(0).astype(int)
    return calendar


def selection_metrics(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    w5 = numeric(frame, "Entry_Fixed_W5_Net_Return_pct")
    w8 = numeric(frame, "Entry_Fixed_W8_Net_Return_pct")
    mfe5 = numeric(frame, "Entry_W5_Cum_MFE_Net_pct")
    mfe8 = numeric(frame, "Entry_W8_Cum_MFE_Net_pct")
    hit10 = frame["Entry_First_Hit_10_vs_Minus10_W8"].astype(str)
    target = true_mask(frame, "Future_W1_Strong_Separation")
    weekly_w5 = frame.assign(_return=w5).groupby("Signal_Date", sort=True)["_return"].mean()
    return {
        "选择组": label, "事件数": len(frame), "不同股票": frame["ts_code"].nunique(),
        "信号周": frame["Signal_Date"].nunique(),
        "预测W5缩尾收益均值%": numeric(
            frame, "OOS_Predicted_Clipped_W5_Return_pct").mean(),
        "预测W5盈利概率均值%": numeric(
            frame, "OOS_Predicted_W5_Positive_Probability").mean() * 100,
        "未来W1持续分离率%": target.mean() * 100 if len(target) else np.nan,
        "固定W5平均净收益%": w5.mean(), "固定W5中位净收益%": w5.median(),
        "固定W5胜率%": w5.gt(0).mean() * 100 if len(w5) else np.nan,
        "W5达到10%比例%": mfe5.ge(10).mean() * 100 if len(mfe5) else np.nan,
        "固定W8平均净收益%": w8.mean(), "固定W8中位净收益%": w8.median(),
        "固定W8胜率%": w8.gt(0).mean() * 100 if len(w8) else np.nan,
        "W8达到20%比例%": mfe8.ge(20).mean() * 100 if len(mfe8) else np.nan,
        "W8先到+10比例%": hit10.str.contains("先到+10%", regex=False).mean() * 100 if len(hit10) else np.nan,
        "W8先到-10比例%": hit10.str.contains("-10%", regex=False).mean() * 100 if len(hit10) else np.nan,
        "等权周W5平均净收益%": weekly_w5.mean(),
        "等权周W5中位净收益%": weekly_w5.median(),
        "盈利周比例%": weekly_w5.gt(0).mean() * 100 if len(weekly_w5) else np.nan,
    }


def topk_selection_audit(events: pd.DataFrame) -> pd.DataFrame:
    rows = [selection_metrics(events, "全部候选")]
    for top_k in TOP_KS:
        selected = events[true_mask(events, f"Selected_Top{top_k}")].copy()
        rows.append(selection_metrics(selected, f"同周Top{top_k}"))
    return pd.DataFrame(rows)


def future_label_value_audit(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [("全部", events)] + list(events.groupby("Signal_Year", sort=True))
    for period, group in groups:
        label_mask = true_mask(group, "Future_W1_Strong_Separation")
        for label, selected in [("未来W1持续分离", group[label_mask]),
                                ("未来W1未持续分离", group[~label_mask])]:
            row = selection_metrics(selected, label)
            row["分组"] = period
            rows.append(row)
    return pd.DataFrame(rows)


def classification_audit(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [("全部样本外", events)] + list(events.groupby("Signal_Year", sort=True))
    for group_name, group in groups:
        valid = group[true_mask(group, "Future_W1_Available")].copy()
        y = true_mask(valid, "Future_W1_Strong_Separation").astype(int)
        p = numeric(valid, "OOS_Predicted_W1_Probability").clip(1e-6, 1 - 1e-6)
        predicted = p.ge(0.5)
        rows.append({
            "分组": group_name, "样本": len(valid), "实际持续分离率%": y.mean() * 100,
            "预测概率均值%": p.mean() * 100, "AUC": auc_score(y, p),
            "Brier": ((p - y) ** 2).mean(),
            "LogLoss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
            "阈值0.5准确率%": predicted.eq(y.astype(bool)).mean() * 100,
            "阈值0.5精确率%": y[predicted].mean() * 100 if predicted.any() else np.nan,
            "阈值0.5召回率%": predicted[y.eq(1)].mean() * 100 if y.eq(1).any() else np.nan,
        })
    return pd.DataFrame(rows)


def direct_prediction_quality_audit(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [("全部样本外", events)] + list(events.groupby("Signal_Year", sort=True))
    for group_name, group in groups:
        actual_raw = numeric(group, "Entry_Fixed_W5_Net_Return_pct")
        actual = actual_raw.clip(RETURN_TARGET_FLOOR, RETURN_TARGET_CAP)
        predicted = numeric(group, "OOS_Predicted_Clipped_W5_Return_pct")
        predicted_positive = numeric(
            group, "OOS_Predicted_W5_Positive_Probability").clip(1e-6, 1 - 1e-6)
        valid = actual.notna() & predicted.notna()
        y_positive = actual_raw.gt(0)
        error = predicted[valid] - actual[valid]
        rows.append({
            "分组": group_name, "样本": int(valid.sum()),
            "实际W5缩尾平均收益%": actual[valid].mean(),
            "预测W5缩尾平均收益%": predicted[valid].mean(),
            "MAE": error.abs().mean(), "RMSE": np.sqrt((error ** 2).mean()),
            "Pearson": predicted[valid].corr(actual[valid], method="pearson"),
            "Spearman": predicted[valid].corr(actual[valid], method="spearman"),
            "实际W5盈利率%": y_positive[valid].mean() * 100,
            "预测W5盈利概率均值%": predicted_positive[valid].mean() * 100,
            "W5盈利概率AUC": auc_score(y_positive[valid], predicted_positive[valid]),
        })
    return pd.DataFrame(rows)


def score_decile_audit(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    ranks = numeric(work, "OOS_Direct_Return_Score").rank(method="first")
    work["OOS_Score_Decile"] = pd.qcut(
        ranks, 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
    rows = []
    for decile, group in work.groupby("OOS_Score_Decile", observed=True, sort=True):
        row = selection_metrics(group, str(decile))
        row["预测W5缩尾收益均值%"] = numeric(group, "OOS_Direct_Return_Score").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def rank_position_audit(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["Rank_Group"] = np.select([
        numeric(work, "Same_Week_Rank").eq(1), numeric(work, "Same_Week_Rank").eq(2),
        numeric(work, "Same_Week_Rank").eq(3), numeric(work, "Same_Week_Rank").between(4, 5),
        numeric(work, "Same_Week_Rank").between(6, 10),
    ], ["第1名", "第2名", "第3名", "第4–5名", "第6–10名"], default="第11名以后")
    order = ["第1名", "第2名", "第3名", "第4–5名", "第6–10名", "第11名以后"]
    return pd.DataFrame([
        selection_metrics(work[work["Rank_Group"].eq(label)], label)
        for label in order if work["Rank_Group"].eq(label).any()
    ])


def random_topk_audit(events: pd.DataFrame, candidate_universe: pd.DataFrame | None = None,
                      trials: int = RANDOM_TRIALS,
                      seed: int = 20260814) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    universe = events if candidate_universe is None else candidate_universe
    groups = [group.index.to_numpy() for _, group in universe.groupby("Signal_Date", sort=True)]
    summary_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for top_k in TOP_KS:
        actual = events[true_mask(events, f"Selected_Top{top_k}")].copy()
        actual_metrics = selection_metrics(actual, f"实际Top{top_k}")
        current_trials: list[dict[str, Any]] = []
        for trial in range(1, trials + 1):
            chosen: list[Any] = []
            for indices in groups:
                chosen.extend(rng.choice(indices, size=min(top_k, len(indices)), replace=False).tolist())
            selected_indices = events.index.intersection(chosen)
            selected = events.loc[selected_indices].copy()
            metrics = selection_metrics(selected, f"随机Top{top_k}")
            metrics.update({"TopK": top_k, "试验编号": trial})
            current_trials.append(metrics)
        trial_frame = pd.DataFrame(current_trials)
        trial_rows.extend(current_trials)
        for metric in (
            "未来W1持续分离率%", "固定W5平均净收益%", "固定W5中位净收益%",
            "固定W5胜率%", "W8先到+10比例%", "固定W8平均净收益%",
            "等权周W5平均净收益%",
        ):
            values = numeric(trial_frame, metric)
            actual_value = finite_num(actual_metrics.get(metric))
            summary_rows.append({
                "TopK": top_k, "指标": metric, "实际值": actual_value,
                "随机P05": values.quantile(0.05), "随机中位数": values.median(),
                "随机P95": values.quantile(0.95),
                "随机达到或超过实际的比例%": values.ge(actual_value).mean() * 100,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(trial_rows)


def weekly_lift_audit(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    details: list[dict[str, Any]] = []
    for top_k in TOP_KS:
        for signal_date, group in events.groupby("Signal_Date", sort=True):
            top = group[numeric(group, "Same_Week_Rank").le(top_k)]
            rest = group[numeric(group, "Same_Week_Rank").gt(top_k)]
            if top.empty or rest.empty:
                continue
            details.append({
                "TopK": top_k, "Signal_Date": signal_date,
                "Top事件": len(top), "其余事件": len(rest),
                "W1持续分离率差_百分点": (
                    true_mask(top, "Future_W1_Strong_Separation").mean()
                    - true_mask(rest, "Future_W1_Strong_Separation").mean()) * 100,
                "W5平均收益差_百分点": (
                    numeric(top, "Entry_Fixed_W5_Net_Return_pct").mean()
                    - numeric(rest, "Entry_Fixed_W5_Net_Return_pct").mean()),
                "W8平均收益差_百分点": (
                    numeric(top, "Entry_Fixed_W8_Net_Return_pct").mean()
                    - numeric(rest, "Entry_Fixed_W8_Net_Return_pct").mean()),
            })
    detail_frame = pd.DataFrame(details)
    rows = []
    for top_k, group in detail_frame.groupby("TopK", sort=True):
        rows.append({
            "TopK": top_k, "可比较周": len(group),
            "持续分离率差均值_百分点": group["W1持续分离率差_百分点"].mean(),
            "Top胜过其余_持续分离周比例%": group["W1持续分离率差_百分点"].gt(0).mean() * 100,
            "W5收益差均值_百分点": group["W5平均收益差_百分点"].mean(),
            "Top胜过其余_W5周比例%": group["W5平均收益差_百分点"].gt(0).mean() * 100,
            "W8收益差均值_百分点": group["W8平均收益差_百分点"].mean(),
            "Top胜过其余_W8周比例%": group["W8平均收益差_百分点"].gt(0).mean() * 100,
        })
    return pd.DataFrame(rows), detail_frame


def robustness_audit(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selections = [("全部候选", events)] + [
        (f"Top{top_k}", events[true_mask(events, f"Selected_Top{top_k}")].copy())
        for top_k in TOP_KS
    ]
    for selection_name, selected in selections:
        for policy, policy_name in EXIT_POLICIES.items():
            column = f"Entry_{policy}_Net_Return_pct"
            values = numeric(selected, column).dropna()
            ordered = values.sort_values(ascending=False)
            trim_n = int(len(ordered) * 0.05)
            trimmed = ordered.iloc[trim_n:len(ordered) - trim_n] if trim_n else ordered
            winsor = values.clip(values.quantile(0.05), values.quantile(0.95)) if len(values) else values
            first_per_stock = selected.sort_values("Signal_Date").drop_duplicates("ts_code", keep="first")
            row = {
                "选择组": selection_name, "退出规则": policy_name,
                "事件数": len(values), "不同股票": selected["ts_code"].nunique(),
                "平均净收益%": values.mean(), "中位净收益%": values.median(),
                "胜率%": values.gt(0).mean() * 100 if len(values) else np.nan,
                "两端各剔除5%后平均%": trimmed.mean(), "5%缩尾平均%": winsor.mean(),
                "每股仅首次信号平均%": numeric(first_per_stock, column).mean(),
                "排除2025年平均%": numeric(selected[selected["Signal_Year"].astype(str).ne("2025")], column).mean(),
            }
            for remove_n in (1, 3, 5, 10):
                row[f"去掉前{remove_n}赢家后平均%"] = ordered.iloc[remove_n:].mean() if len(ordered) > remove_n else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def period_selection_audit(events: pd.DataFrame, period_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, group in events.groupby(period_column, sort=True):
        for selection_name, selected in [("全部", group)] + [
            (f"Top{top_k}", group[true_mask(group, f"Selected_Top{top_k}")])
            for top_k in TOP_KS
        ]:
            row = selection_metrics(selected, selection_name)
            row[period_column] = period
            rows.append(row)
    return pd.DataFrame(rows)


def exit_policy_audit(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for selection_name, selected in [("全部候选", events)] + [
        (f"Top{top_k}", events[true_mask(events, f"Selected_Top{top_k}")])
        for top_k in TOP_KS
    ]:
        for policy, policy_name in EXIT_POLICIES.items():
            values = numeric(selected, f"Entry_{policy}_Net_Return_pct")
            triggers = selected[f"Entry_{policy}_Trigger"].astype(str)
            rows.append({
                "选择组": selection_name, "退出规则": policy_name, "样本": len(selected),
                "平均净收益%": values.mean(), "中位净收益%": values.median(),
                "胜率%": values.gt(0).mean() * 100,
                "亏损10%以上比例%": values.le(-10).mean() * 100,
                "平均持有交易日": numeric(selected, f"Entry_{policy}_Holding_Trading_Days").mean(),
                "止损触发比例%": triggers.str.contains("止损", regex=False).mean() * 100,
            })
    return pd.DataFrame(rows)


def build_portfolio_records(events: pd.DataFrame) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for _, group in events.groupby("Signal_Date", sort=True):
        records = []
        for row in group.sort_values(["Same_Week_Rank", "ts_code"]).itertuples(index=False):
            policies = {
                policy: (
                    str(getattr(row, f"Entry_{policy}_Exit_Date")),
                    finite_num(getattr(row, f"Entry_{policy}_Net_Return_pct")),
                ) for policy in EXIT_POLICIES
            }
            records.append({
                "Signal_Date": str(row.Signal_Date), "Entry_Date": str(row.Entry_Entry_Date),
                "ts_code": str(row.ts_code), "name": str(row.name),
                "Rank": int(row.Same_Week_Rank), "Year": str(row.Signal_Year),
                "policies": policies,
            })
        groups.append(records)
    return groups


def simulate_three_slots(record_groups: list[list[dict[str, Any]]], policy: str,
                         randomize: bool = False, rng: np.random.Generator | None = None,
                         keep_details: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    slots = [{"equity": 100000.0, "exit": "00000000", "code": ""} for _ in range(3)]
    details: list[dict[str, Any]] = []
    skipped_full = skipped_duplicate = 0
    for records in record_groups:
        if randomize and rng is not None:
            order = rng.permutation(len(records)).tolist()
            candidates = [records[index] for index in order]
        else:
            candidates = records
        for position, record in enumerate(candidates):
            exit_date, trade_return = record["policies"][policy]
            if not exit_date or not math.isfinite(trade_return):
                continue
            entry_date = record["Entry_Date"]
            active_codes = {
                slot["code"] for slot in slots if slot["exit"] >= entry_date and slot["code"]}
            if record["ts_code"] in active_codes:
                skipped_duplicate += 1
                continue
            free = [index for index, slot in enumerate(slots) if slot["exit"] < entry_date]
            if not free:
                skipped_full += len(candidates) - position
                break
            slot_index = free[0]
            before = slots[slot_index]["equity"]
            after = before * (1.0 + trade_return / 100.0)
            slots[slot_index] = {
                "equity": after, "exit": exit_date, "code": record["ts_code"]}
            if keep_details:
                details.append({
                    "退出规则": EXIT_POLICIES[policy], "槽位": slot_index + 1,
                    "Signal_Date": record["Signal_Date"], "Entry_Date": entry_date,
                    "Exit_Date": exit_date, "ts_code": record["ts_code"],
                    "name": record["name"], "Same_Week_Rank": record["Rank"],
                    "交易净收益%": trade_return, "交易前槽位权益": before,
                    "交易后槽位权益": after, "Signal_Year": record["Year"],
                })
    returns = [row["交易净收益%"] for row in details] if keep_details else []
    result = {
        "退出规则": EXIT_POLICIES[policy],
        "初始资金": 300000.0, "期末权益": sum(slot["equity"] for slot in slots),
        "总收益率%": (sum(slot["equity"] for slot in slots) / 300000.0 - 1.0) * 100.0,
        "成交笔数": len(details) if keep_details else np.nan,
        "交易平均收益%": np.mean(returns) if returns else np.nan,
        "交易中位收益%": np.median(returns) if returns else np.nan,
        "交易胜率%": np.mean(np.array(returns) > 0) * 100 if returns else np.nan,
        "平均买入排名": np.mean([row["Same_Week_Rank"] for row in details]) if details else np.nan,
        "仓位满跳过候选": skipped_full, "重复持仓跳过": skipped_duplicate,
        "槽位1期末权益": slots[0]["equity"], "槽位2期末权益": slots[1]["equity"],
        "槽位3期末权益": slots[2]["equity"],
    }
    return result, details


def three_slot_portfolio_audit(events: pd.DataFrame
                               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = build_portfolio_records(events)
    summary_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for policy_number, policy in enumerate(EXIT_POLICIES):
        actual, details = simulate_three_slots(records, policy, keep_details=True)
        summary_rows.append(actual)
        trade_rows.extend(details)
        random_returns = []
        for trial in range(1, PORTFOLIO_RANDOM_TRIALS + 1):
            random_result, _ = simulate_three_slots(
                records, policy, randomize=True,
                rng=np.random.default_rng(20260814 + policy_number * 10000 + trial),
                keep_details=False)
            random_returns.append(random_result["总收益率%"])
        random_series = pd.Series(random_returns, dtype=float)
        random_rows.append({
            "退出规则": EXIT_POLICIES[policy], "实际评分三仓总收益率%": actual["总收益率%"],
            "随机排名三仓P05%": random_series.quantile(0.05),
            "随机排名三仓中位数%": random_series.median(),
            "随机排名三仓P95%": random_series.quantile(0.95),
            "随机达到或超过实际比例%": random_series.ge(actual["总收益率%"]).mean() * 100,
            "随机试验次数": PORTFOLIO_RANDOM_TRIALS,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(trade_rows), pd.DataFrame(random_rows)


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption("上穿25后立即执行；按历史真实W5收益逐年样本外训练；直接收益排序后再进行真实三仓占位。")
    with st.expander("V4.6验证框架", expanded=True):
        st.markdown(f"""
- **可执行买点**：周线低位金叉进入观察；K首次从25下方上穿25、K>D且K上升，该完整周结束后的下一市场交易日开盘买入。
- **主训练目标**：直接预测固定W5净收益，训练标签限制在{RETURN_TARGET_FLOOR:.0f}%～+{RETURN_TARGET_CAP:.0f}%，降低少数牛股对模型的支配。
- **辅助目标**：同时预测同周W5收益百分位、W5盈利概率以及未来W1持续分离概率；未来字段只属于过去训练样本，绝不作为当期已知事实。
- **严格样本外**：每个目标年度只使用此前{MODEL_LOOKBACK_YEARS}个完整年度、并且W5观察期已经结束的样本；不使用本年度结果重新拟合本年度。
- **排序**：预测W5缩尾收益为第一顺序；预测同周百分位、盈利概率、W1分离概率及历史波段特征只负责破同分。
- **准确性审计**：同时报告Top1、Top3、Top5、全部候选、同周随机选择、得分十分位和排名位置，不用三仓总收益代替评分准确率。
- **稳健性审计**：报告中位数、截尾/缩尾、去掉前1/3/5/10赢家、每股仅首次信号、排除2025年。
- **真实三仓**：总资金30万元、三个独立10万元槽位；仓位未退出时不接收新信号，同一股票持仓期间不重复买入；按同周排名依次补空位。
- **三仓边界**：交易级结果可以计算期末权益，但没有逐日组合净值，因此本版不伪造组合最大回撤。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("买入信号开始", date(2023, 6, 5), key="v46_start")
        signal_end_date = st.date_input("买入信号截止", date(2026, 6, 5), key="v46_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v46_market_end")
        split_date_value = st.date_input("近期行情分界", date(2025, 6, 1), key="v46_split")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v46_pause")
        use_cache = st.checkbox("复用逐股票缓存", True, key="v46_cache")
        st.divider()
        commission_pct = st.number_input("佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f", key="v46_commission")
        stamp_duty_pct = st.number_input("卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f", key="v46_stamp")
        transfer_fee_pct = st.number_input("过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f", key="v46_transfer")
        if st.button("清除本程序行情缓存", key="v46_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("缓存已清除")

    token = st.text_input("Tushare Token", type="password", key="v46_token")
    session_key = "weekly_skdj_direct_return_rank_v46_zip"
    result_name = "weekly_skdj_direct_return_rank_v4_6_all_results.zip"
    if not token:
        st.info("请输入Tushare Token；本版没有增加新的Python依赖。")
        return
    if not st.button("开始V4.6直接收益排序审计", type="primary", key="v46_run"):
        if session_key in st.session_state:
            st.download_button("下载上一次结果ZIP", st.session_state[session_key],
                               file_name=result_name, mime="application/zip", on_click="ignore")
        return
    error = validate_dates(signal_start_date, signal_end_date, market_end_date)
    if error:
        st.error(error)
        return
    if (market_end_date - signal_end_date).days < 70:
        st.warning("观察截止日距离信号截止日不足70天，末端事件可能没有完整W8；成熟样本会单独处理。")

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    signal_start = signal_start_date.strftime("%Y%m%d")
    signal_end = signal_end_date.strftime("%Y%m%d")
    market_end = market_end_date.strftime("%Y%m%d")
    # Keep the same history window/cache key as V4.4. The 40-week indicator warm-up
    # naturally shortens the first target year's usable training history.
    model_start_date = signal_start_date - timedelta(days=MODEL_LOOKBACK_YEARS * 365)
    model_start = model_start_date.strftime("%Y%m%d")
    rejects: dict[str, int] = {}
    config = {
        "signal_start": signal_start, "signal_end": signal_end, "market_end": market_end,
        "model_start": model_start, "split_date": split_date_value.strftime("%Y%m%d"),
        "min_price": 10.0, "min_mv": 100.0,
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct), "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct), "rejects": rejects,
    }
    try:
        with st.spinner("加载交易日历、申万历史科技池和训练期数据..."):
            open_dates = load_trade_calendar(model_start, market_end)
            extended_end = (market_end_date + timedelta(days=7)).strftime("%Y%m%d")
            week_last_map = complete_week_last_dates(load_trade_calendar(model_start, extended_end))
            market_weeks = market_week_sequence(open_dates)
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return

    period_index = build_period_index(memberships)
    codes = sorted(set(period_index) & set(stock_basic["ts_code"].astype(str)))
    stocks = stock_basic[stock_basic["ts_code"].isin(codes)].copy()
    stocks = stocks[~stocks["list_date"].gt(signal_end) & ~stocks["delist_date"].lt(model_start)].copy()
    stocks["Sample_Board"] = stocks.apply(sample_board, axis=1)
    stocks = stocks.sort_values("ts_code").reset_index(drop=True)
    population = stocks.groupby("Sample_Board").size().reindex(
        BOARDS, fill_value=0).rename("股票数").reset_index()
    open_pos = {day: position for position, day in enumerate(open_dates)}

    cycle_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    cache_hits = data_failures = 0
    progress, status = st.progress(0.0), st.empty()
    for number, stock in stocks.iterrows():
        code = str(stock["ts_code"])
        progress.progress((number + 1) / max(len(stocks), 1), text=f"{number + 1}/{len(stocks)} {code}")
        status.caption(
            f"底部周期 {len(cycle_rows)}；模型历史 {len(model_rows)}；目标事件 {len(event_rows)}；"
            f"缓存 {cache_hits}；失败 {data_failures}")
        daily, daily_basic, cache_hit = fetch_stock_history(
            code, model_start, market_end, bool(use_cache), float(pause))
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        cycle_part, model_part, event_part = analyze_stock(
            stock, period_index.get(code, []), daily, daily_basic, week_last_map,
            open_dates, open_pos, market_weeks, config)
        cycle_rows.extend(cycle_part)
        model_rows.extend(model_part)
        event_rows.extend(event_part)
    progress.empty()
    status.empty()

    cycles = pd.DataFrame(cycle_rows)
    model_history = pd.DataFrame(model_rows)
    events_all = pd.DataFrame(event_rows)
    if events_all.empty:
        st.error("研究区间没有生成符合股票池的上穿25可交易事件。")
        return
    dt = pd.to_datetime(events_all["Signal_Date"].astype(str), format="%Y%m%d", errors="coerce")
    events_all["Signal_Year"] = events_all["Signal_Date"].astype(str).str[:4]
    events_all["Signal_Half_Year"] = events_all["Signal_Year"] + "H" + np.where(dt.dt.month.le(6), "1", "2")
    events_all, model_summary, coefficients = apply_annual_oos_models(model_history, events_all)
    events_all = rank_same_week(events_all)
    events = mature_events(events_all)
    if events.empty:
        st.error("有上穿25事件，但没有未来完整W8的成熟样本；请延后行情观察截止日。")
        return

    classification = classification_audit(events_all)
    direct_quality = direct_prediction_quality_audit(events)
    label_value = future_label_value_audit(events)
    topk = topk_selection_audit(events)
    deciles = score_decile_audit(events)
    rank_positions = rank_position_audit(events)
    random_summary, random_trials = random_topk_audit(events, events_all)
    weekly_lift, weekly_lift_details = weekly_lift_audit(events)
    robustness = robustness_audit(events)
    exits = exit_policy_audit(events)
    yearly = period_selection_audit(events, "Signal_Year")
    half_yearly = period_selection_audit(events, "Signal_Half_Year")
    portfolio, portfolio_trades, portfolio_random = three_slot_portfolio_audit(events)
    calendar = signal_week_calendar(open_dates, signal_start, signal_end, events_all)
    counts = calendar["All_Candidates"]

    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "买入信号开始": signal_start,
        "买入信号截止": signal_end, "观察截止": market_end,
        "模型历史开始": model_start, "底部周期": len(cycles),
        "历史模型事件": len(model_history), "全部买入信号": len(events_all),
        "W8成熟买入事件": len(events),
        "不同股票": events["ts_code"].nunique(), "自然周": len(calendar),
        "有信号周": int(counts.gt(0).sum()), "空窗周": int(counts.eq(0).sum()),
        "最长连续空窗周": max_empty_run(counts), "每周候选均值": counts.mean(),
        "每周候选中位数": counts.median(), "单周最多": counts.max(),
        "模型样本不足年度数": int(events["Model_Status"].ne(
            "年度样本外_直接收益Ridge+盈利Logistic").groupby(
                events["Signal_Year"]).any().sum()),
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    metadata = pd.DataFrame([
        ("买点", "低位金叉观察后，K首次从25下方上穿25且K>D、K上升；下一市场交易日开盘买"),
        ("主训练目标", f"固定W5净收益缩尾至{RETURN_TARGET_FLOOR:.0f}%～+{RETURN_TARGET_CAP:.0f}%；直接预测可交易结果"),
        ("辅助训练目标", "同周W5收益百分位、W5盈利概率、未来W1持续分离概率"),
        ("年度样本外", f"目标年度仅使用此前{MODEL_LOOKBACK_YEARS}个完整年度，且W5结束日不晚于训练截止日的样本"),
        ("预测模型", f"纯NumPy L2 Ridge+Logistic；特征1%/99%训练期缩尾；最少训练样本{MODEL_MIN_TRAIN}"),
        ("排序", "预测W5缩尾收益优先；同周百分位、盈利概率、W1分离概率及历史波段特征依次破同分"),
        ("随机检验", f"同周随机Top1/3/5各{RANDOM_TRIALS}次；三仓随机排名各退出规则{PORTFOLIO_RANDOM_TRIALS}次"),
        ("真实三仓", "三个10万元独立槽位；退出日当天保守地不能在开盘释放槽位；仓位满则跳过后续候选"),
        ("组合限制", "只输出交易级复利期末权益；没有逐日组合净值，不报告最大回撤"),
        ("成本", "买卖均计0.2%滑点、佣金和过户费，卖出另计印花税"),
        ("股票池", "申万历史科技行业；主板/创业板/科创板；低位金叉及上穿25日股价≥10元、流通市值≥100亿元"),
        ("严禁使用", "目标年度未来W1字段、未来收益、最高价及本年度训练结果均不进入当期评分"),
    ], columns=["项目", "说明"])

    files = {
        "01_run_summary_v4_6.csv": run_summary,
        "02_oos_model_summary_v4_6.csv": model_summary,
        "03_oos_model_coefficients_v4_6.csv": coefficients,
        "04_oos_direct_return_quality_v4_6.csv": direct_quality,
        "05_oos_w1_auxiliary_quality_v4_6.csv": classification,
        "06_future_w1_label_value_v4_6.csv": label_value,
        "07_top1_top3_top5_selection_quality_v4_6.csv": topk,
        "08_direct_score_decile_monotonicity_v4_6.csv": deciles,
        "09_same_week_rank_position_v4_6.csv": rank_positions,
        "10_topk_vs_random_summary_v4_6.csv": random_summary,
        "11_topk_random_trials_v4_6.csv": random_trials,
        "12_weekly_topk_lift_summary_v4_6.csv": weekly_lift,
        "13_weekly_topk_lift_details_v4_6.csv": weekly_lift_details,
        "14_robustness_remove_winners_v4_6.csv": robustness,
        "15_exit_policy_by_selection_v4_6.csv": exits,
        "16_year_oos_stability_v4_6.csv": yearly,
        "17_half_year_oos_stability_v4_6.csv": half_yearly,
        "18_true_three_slot_portfolio_v4_6.csv": portfolio,
        "19_true_three_slot_trades_v4_6.csv": portfolio_trades,
        "20_three_slot_vs_random_rank_v4_6.csv": portfolio_random,
        "21_weekly_signal_calendar_v4_6.csv": calendar,
        "22_all_ranked_signal_events_v4_6.csv": events_all,
        "23_all_ranked_mature_events_v4_6.csv": events,
        "24_all_model_history_events_v4_6.csv": model_history,
        "25_all_bottom_cycles_v4_6.csv": cycles,
        "26_full_tech_universe_v4_6.csv": stocks,
        "27_board_population_v4_6.csv": population,
        "28_rejection_audit_v4_6.csv": pd.DataFrame(
            [{"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]),
        "29_api_errors_v4_6.csv": pd.DataFrame({"错误": API_ERRORS}),
        "30_metadata_v4_6.csv": metadata,
    }
    result_zip = make_zip(files)
    st.session_state[session_key] = result_zip
    st.success(
        f"完成：成熟候选{len(events)}个；有信号周{int(counts.gt(0).sum())}，"
        f"空窗{int(counts.eq(0).sum())}周；已完成样本外TopK和真实三仓审计。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("成熟候选", len(events))
    c2.metric("有信号周", int(counts.gt(0).sum()))
    c3.metric("空窗周", int(counts.eq(0).sum()))
    c4.metric("单周最多", int(counts.max()))
    st.subheader("样本外直接收益预测质量")
    st.dataframe(direct_quality, use_container_width=True, hide_index=True)
    st.subheader("Top1 / Top3 / Top5准确性")
    st.dataframe(topk, use_container_width=True, hide_index=True)
    st.subheader("TopK与同周随机选择")
    st.dataframe(random_summary, use_container_width=True, hide_index=True)
    st.subheader("真实三仓占位")
    st.dataframe(portfolio, use_container_width=True, hide_index=True)
    st.subheader("真实三仓与随机排名")
    st.dataframe(portfolio_random, use_container_width=True, hide_index=True)
    st.download_button("下载V4.6全部结果ZIP", result_zip, file_name=result_name,
                       mime="application/zip", type="primary", key="v46_download", on_click="ignore")
    st.info("先看04判断直接收益预测；07–14判断TopK、随机基准和牛股依赖；最后看18–20判断真实三仓是否稳定优于随机。")


if __name__ == "__main__":
    main()
