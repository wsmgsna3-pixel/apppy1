from __future__ import annotations

import hashlib
import io
import math
import os
import pickle
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts


TITLE = "周线MACD × V40.6：30万元三仓组合验证器 V4.7"
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "portfolio_v47_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 300_000.0
MAX_POSITIONS = 3
POSITION_BUDGET = 100_000.0
LOT_SIZE = 100

# V4.7取消两项硬门槛：Wave_2_5_Pass、Solid_Yang_Pass。
# 实体质量信息没有丢弃，仍保留在Body_Quality_Score和New_Score_100中。
V47_HARD_GATES = [
    "Data_Available",
    "True_First_Red_Audit",
    "Weekly_Bias_Pass",
    "Weekly_Shadow_Pass",
    "Daily_Trend_Pass",
    "Box_Breakout_Pass",
    "Daily_Breakout_Pass",
    "MA20_Healthy_Pass",
    "Volume_Pass",
    "Daily_MACD_Pass",
    "One_Word_Pass",
    "Gap_Pass",
]

REQUIRED_COLUMNS = {
    "ts_code",
    "name",
    "Sample_Board",
    "Signal_Date",
    "Entry_Date",
    "Entry_Price",
    "Exit_T30_Date",
    "Exit_T30_Price",
    "Exit_T30_Return_pct",
    "Exit_T30_Reason",
    "New_Score_100",
    "Original_V406_Score",
    "Board_RS_13W_pct",
    *V47_HARD_GATES,
}


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "是"}


def num(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def date8(value: Any) -> str:
    if pd.isna(value):
        return ""
    digits = "".join(ch for ch in str(value).split(".")[0] if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"CSV无法读取：{last_error}")


def load_v46_audit(uploaded: Any) -> tuple[pd.DataFrame, str]:
    raw = uploaded.getvalue()
    if uploaded.name.lower().endswith(".csv"):
        return read_csv_bytes(raw), uploaded.name
    if not uploaded.name.lower().endswith(".zip"):
        raise ValueError("请上传V4.6全部结果ZIP，或其中的01_hybrid_event_audit.csv。")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        preferred = [
            name for name in names
            if Path(name).name.lower() == "01_hybrid_event_audit.csv"
        ]
        if not preferred:
            preferred = [
                name for name in names
                if name.lower().endswith(".csv")
                and "event_audit" in Path(name).name.lower()
            ]
        if not preferred:
            raise ValueError("ZIP中没有找到01_hybrid_event_audit.csv。请上传V4.6全部结果ZIP。")
        target = sorted(preferred, key=len)[0]
        return read_csv_bytes(archive.read(target)), f"{uploaded.name}/{target}"


def validate_audit(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "输入文件不是完整的V4.6事件审计文件，缺少字段：" + "、".join(missing)
        )


def prepare_v47(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = raw.copy()
    if "Event_Type" in frame.columns:
        frame = frame[frame["Event_Type"].astype(str).eq("第一根红柱")].copy()
    if "Cycle_ID" in frame.columns:
        frame = frame.drop_duplicates("Cycle_ID", keep="last")

    for column in ("Signal_Date", "Entry_Date", "Exit_T30_Date"):
        frame[column] = frame[column].map(date8)
    frame = frame[
        frame["Signal_Date"].str.len().eq(8)
        & frame["Entry_Date"].str.len().eq(8)
        & frame["Exit_T30_Date"].str.len().eq(8)
    ].copy()

    numeric_columns = [
        "Entry_Price", "Exit_T30_Price", "Exit_T30_Return_pct",
        "Exit_T30_Holding_Days", "New_Score_100", "Original_V406_Score",
        "Board_RS_13W_pct", "Body_Ratio", "Body_Quality_Score",
        "Wave_Count_True_Weekly", "MFE_8W_pct", "MAE_8W_pct",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in V47_HARD_GATES:
        frame[column] = frame[column].map(to_bool)
    for column in ("Wave_2_5_Pass", "Solid_Yang_Pass", "Strong_Sustained", "Weak_Rebound", "Stop_8W", "Target30_Before_Stop"):
        if column in frame.columns:
            frame[column] = frame[column].map(to_bool)

    frame["V47_Wave_Hard_Gate_Removed"] = True
    frame["V47_Solid_Yang_Hard_Gate_Removed"] = True
    frame["V47_Candidate_Pass"] = frame[V47_HARD_GATES].all(axis=1)
    candidates = frame[frame["V47_Candidate_Pass"]].copy()

    # 分数相同时用原V40.6分数、板块相对强度和代码作固定裁决，保证重复运行一致。
    candidates = candidates.sort_values(
        [
            "Signal_Date", "New_Score_100", "Original_V406_Score",
            "Board_RS_13W_pct", "ts_code",
        ],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    candidates["V47_Weekly_Rank"] = candidates.groupby("Signal_Date").cumcount() + 1
    candidates["V47_Selected"] = candidates["V47_Weekly_Rank"].eq(1)
    selected = candidates[candidates["V47_Selected"]].copy().reset_index(drop=True)
    frame = frame.merge(
        candidates[["Cycle_ID", "V47_Weekly_Rank", "V47_Selected"]]
        if "Cycle_ID" in frame.columns
        else candidates[["ts_code", "Signal_Date", "V47_Weekly_Rank", "V47_Selected"]],
        on="Cycle_ID" if "Cycle_ID" in frame.columns else ["ts_code", "Signal_Date"],
        how="left",
    )
    frame["V47_Selected"] = frame["V47_Selected"].map(to_bool)
    return frame.sort_values(["Signal_Date", "ts_code"]), candidates, selected


def cache_path(code: str, start: str, end: str) -> Path:
    key = hashlib.sha1(f"{code}|{start}|{end}|qfq".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{code.replace('.', '_')}_{key}.pkl"


def atomic_pickle(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def fetch_daily(
    pro: Any,
    code: str,
    start: str,
    end: str,
    use_cache: bool,
    pause: float,
    retries: int = 3,
) -> tuple[pd.DataFrame, bool, str]:
    path = cache_path(code, start, end)
    if use_cache and path.exists():
        try:
            with path.open("rb") as handle:
                cached = pickle.load(handle)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached, True, ""
        except Exception:
            pass

    last_error = ""
    result = pd.DataFrame()
    for attempt in range(retries):
        try:
            data = ts.pro_bar(
                api=pro,
                ts_code=code,
                start_date=start,
                end_date=end,
                adj="qfq",
                freq="D",
            )
            result = pd.DataFrame() if data is None else pd.DataFrame(data).copy()
            if not result.empty:
                break
            last_error = "返回空行情"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.8 * (attempt + 1))
    time.sleep(pause)

    if result.empty:
        return result, False, last_error or "返回空行情"
    result["trade_date"] = result["trade_date"].astype(str)
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = (
        result.dropna(subset=["trade_date", "open", "close"])
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    if use_cache and not result.empty:
        atomic_pickle(result, path)
    return result, False, ""


def fetch_trade_days(pro: Any, start: str, end: str, histories: dict[str, pd.DataFrame]) -> list[str]:
    try:
        calendar = pro.trade_cal(
            exchange="SSE", start_date=start, end_date=end, is_open="1",
            fields="cal_date,is_open",
        )
        calendar = pd.DataFrame(calendar)
        if not calendar.empty:
            return sorted(calendar["cal_date"].astype(str).unique())
    except Exception:
        pass
    dates: set[str] = set()
    for history in histories.values():
        if not history.empty:
            dates.update(history["trade_date"].astype(str))
    return sorted(date for date in dates if start <= date <= end)


def close_on_or_before(history: pd.DataFrame, trade_date: str) -> float:
    if history.empty:
        return np.nan
    subset = history[history["trade_date"].le(trade_date)]
    return num(subset.iloc[-1]["close"]) if not subset.empty else np.nan


def open_on_date(history: pd.DataFrame, trade_date: str) -> float:
    if history.empty:
        return np.nan
    subset = history[history["trade_date"].eq(trade_date)]
    return num(subset.iloc[-1]["open"]) if not subset.empty else np.nan


def simulate_portfolio(
    selected: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    trade_days: list[str],
    interval_start: str,
    interval_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = selected.copy()
    for column in ("Signal_Date", "Entry_Date", "Exit_T30_Date"):
        work[column] = work[column].map(date8)
    for column in (
        "Entry_Price", "Exit_T30_Price", "Exit_T30_Return_pct",
        "New_Score_100", "Original_V406_Score",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.sort_values(
        ["Entry_Date", "Signal_Date", "New_Score_100", "ts_code"],
        ascending=[True, True, False, True], kind="mergesort",
    ).reset_index(drop=True)

    last_exit = work["Exit_T30_Date"].max()
    days = sorted({
        date for date in trade_days
        if interval_start <= date <= last_exit
    } | set(work["Entry_Date"]) | set(work["Exit_T30_Date"]))
    entry_groups = {
        date: group.sort_values(
            ["New_Score_100", "Original_V406_Score", "Board_RS_13W_pct", "ts_code"],
            ascending=[False, False, False, True], kind="mergesort",
        )
        for date, group in work.groupby("Entry_Date", sort=True)
    }

    cash = INITIAL_CAPITAL
    active: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    full_reason = f"{MAX_POSITIONS}个仓位已满"

    def audit(row: pd.Series, action: str, reason: str, positions_before: int) -> None:
        orders.append({
            "Signal_Date": row.get("Signal_Date", ""),
            "Entry_Date": row.get("Entry_Date", ""),
            "ts_code": row.get("ts_code", ""),
            "name": row.get("name", ""),
            "Sample_Board": row.get("Sample_Board", ""),
            "V47_Weekly_Rank": row.get("V47_Weekly_Rank", np.nan),
            "New_Score_100": row.get("New_Score_100", np.nan),
            "Portfolio_Action": action,
            "Portfolio_Reason": reason,
            "Positions_Before": positions_before,
            "Prospective_Exit_Date": row.get("Exit_T30_Date", ""),
            "Prospective_Return_pct": row.get("Exit_T30_Return_pct", np.nan),
            "Prospective_Exit_Reason": row.get("Exit_T30_Reason", ""),
        })

    for trade_date in days:
        # 与V40.4/V40.6组合口径一致：当日开盘先尝试买入，旧仓当日稍后卖出。
        # 因此当日退出的旧仓不会提前释放开盘时的仓位。
        for _, row in entry_groups.get(trade_date, pd.DataFrame()).iterrows():
            code = str(row["ts_code"])
            positions_before = len(active)
            if code in active:
                audit(row, "未买入", "同一股票已在持仓", positions_before)
                continue
            if len(active) >= MAX_POSITIONS:
                audit(row, "未买入", full_reason, positions_before)
                continue
            entry_price = num(row["Entry_Price"])
            exit_price = num(row["Exit_T30_Price"])
            if not math.isfinite(entry_price) or entry_price <= 0:
                audit(row, "未买入", "买入价无效", positions_before)
                continue
            if not math.isfinite(exit_price) or exit_price <= 0:
                audit(row, "未买入", "退出价无效", positions_before)
                continue
            budget = min(POSITION_BUDGET, cash)
            shares = int(math.floor(budget / entry_price / LOT_SIZE) * LOT_SIZE)
            if shares < LOT_SIZE:
                audit(row, "未买入", "可用现金不足一手", positions_before)
                continue

            entry_cost = shares * entry_price
            cash -= entry_cost
            raw_entry_open = open_on_date(histories.get(code, pd.DataFrame()), trade_date)
            price_scale = entry_price / raw_entry_open if raw_entry_open > 0 else 1.0
            trade = {
                "Signal_Date": row["Signal_Date"],
                "Entry_Date": trade_date,
                "ts_code": code,
                "name": row.get("name", ""),
                "Sample_Board": row.get("Sample_Board", ""),
                "New_Score_100": row.get("New_Score_100", np.nan),
                "Original_V406_Score": row.get("Original_V406_Score", np.nan),
                "Body_Ratio": row.get("Body_Ratio", np.nan),
                "Wave_Count_True_Weekly": row.get("Wave_Count_True_Weekly", np.nan),
                "Entry_Price": entry_price,
                "Shares": shares,
                "Entry_Cost": entry_cost,
                "Planned_Exit_Date": row["Exit_T30_Date"],
                "Actual_Exit_Date": "",
                "Net_Exit_Price": np.nan,
                "Exit_Proceeds": np.nan,
                "PnL": np.nan,
                "Portfolio_Return_pct": np.nan,
                "Exit_Reason": row.get("Exit_T30_Reason", ""),
                "Portfolio_Status": "持仓中",
                "_price_scale": price_scale,
                "_last_mark": entry_price,
            }
            active[code] = trade
            trades.append(trade)
            audit(row, "已买入", f"买入{shares}股，使用资金{entry_cost:,.2f}元", positions_before)

        exiting: list[str] = []
        for code, trade in active.items():
            if trade["Planned_Exit_Date"] != trade_date:
                continue
            matching = work[
                work["Signal_Date"].eq(trade["Signal_Date"])
                & work["ts_code"].astype(str).eq(code)
            ]
            if matching.empty:
                continue
            exit_price = num(matching.iloc[-1]["Exit_T30_Price"])
            proceeds = trade["Shares"] * exit_price
            cash += proceeds
            pnl = proceeds - trade["Entry_Cost"]
            trade.update({
                "Actual_Exit_Date": trade_date,
                "Net_Exit_Price": exit_price,
                "Exit_Proceeds": proceeds,
                "PnL": pnl,
                "Portfolio_Return_pct": pnl / trade["Entry_Cost"] * 100.0,
                "Portfolio_Status": "已平仓",
            })
            exiting.append(code)
        for code in exiting:
            active.pop(code, None)

        market_value = 0.0
        for code, trade in active.items():
            raw_close = close_on_or_before(histories.get(code, pd.DataFrame()), trade_date)
            mark = raw_close * trade["_price_scale"] if raw_close > 0 else trade["_last_mark"]
            trade["_last_mark"] = mark
            market_value += trade["Shares"] * mark
        equity = cash + market_value
        curve_rows.append({
            "Trade_Date": trade_date,
            "Cash": cash,
            "Market_Value": market_value,
            "Equity": equity,
            "Positions": len(active),
            "Exposure_pct": market_value / equity * 100.0 if equity > 0 else np.nan,
            "Is_Empty": len(active) == 0,
        })

    curve = pd.DataFrame(curve_rows)
    curve["Daily_Return_pct"] = curve["Equity"].pct_change().fillna(
        curve["Equity"].iloc[0] / INITIAL_CAPITAL - 1.0
    ) * 100.0
    curve["Drawdown_pct"] = (
        curve["Equity"] / curve["Equity"].cummax().clip(lower=INITIAL_CAPITAL) - 1.0
    ) * 100.0

    ledger = pd.DataFrame(trades)
    if not ledger.empty:
        ledger = ledger.drop(columns=[column for column in ledger.columns if column.startswith("_")])
        for column in (
            "Entry_Price", "Entry_Cost", "Net_Exit_Price", "Exit_Proceeds", "PnL",
            "Portfolio_Return_pct",
        ):
            ledger[column] = pd.to_numeric(ledger[column], errors="coerce").round(4)
    orders_frame = pd.DataFrame(orders)
    missed = orders_frame[orders_frame["Portfolio_Action"].eq("未买入")].copy()
    closed = ledger[ledger["Portfolio_Status"].eq("已平仓")].copy()

    gross_profit = closed.loc[closed["PnL"].gt(0), "PnL"].sum()
    gross_loss = -closed.loc[closed["PnL"].lt(0), "PnL"].sum()
    net_pnl = closed["PnL"].sum()
    positive_pnl = closed.loc[closed["PnL"].gt(0), "PnL"].sort_values(ascending=False)
    top1 = positive_pnl.head(1).sum()
    top3 = positive_pnl.head(3).sum()
    trade_count = max(1, len(curve))
    final_equity = float(curve.iloc[-1]["Equity"])
    annualized = (
        (final_equity / INITIAL_CAPITAL) ** (252.0 / trade_count) - 1.0
    ) * 100.0 if final_equity > 0 else np.nan
    summary = {
        "程序": TITLE,
        "初始资金": INITIAL_CAPITAL,
        "单仓目标资金": POSITION_BUDGET,
        "最多持仓": MAX_POSITIONS,
        "评分第一名信号": len(work),
        "实际买入": len(ledger),
        "完成交易": len(closed),
        "仓位满错过": int(missed["Portfolio_Reason"].eq(full_reason).sum()) if not missed.empty else 0,
        "其他原因错过": int((~missed["Portfolio_Reason"].eq(full_reason)).sum()) if not missed.empty else 0,
        "期末权益": final_equity,
        "总收益率(%)": (final_equity / INITIAL_CAPITAL - 1.0) * 100.0,
        "年化收益率(%)": annualized,
        "最大回撤(%)": float(curve["Drawdown_pct"].min()),
        "有持仓交易日": int((~curve["Is_Empty"]).sum()),
        "空仓交易日比例(%)": float(curve["Is_Empty"].mean() * 100.0),
        "平均持仓数": float(curve["Positions"].mean()),
        "平均资金暴露(%)": float(curve["Exposure_pct"].mean()),
        "交易胜率(%)": float(closed["PnL"].gt(0).mean() * 100.0) if len(closed) else np.nan,
        "止损率(%)": float(closed["Exit_Reason"].astype(str).str.contains("止损").mean() * 100.0) if len(closed) else np.nan,
        "盈利因子": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "已实现净利润": net_pnl,
        "最大盈利一笔占全部正利润(%)": top1 / gross_profit * 100.0 if gross_profit > 0 else np.nan,
        "前三笔占全部正利润(%)": top3 / gross_profit * 100.0 if gross_profit > 0 else np.nan,
        "扣除最大一笔后的净利润": net_pnl - top1,
        "扣除前三笔后的净利润": net_pnl - top3,
    }
    return curve, ledger, orders_frame, missed, summary


def weekly_coverage(
    audit: pd.DataFrame,
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = pd.to_datetime(audit["Signal_Date"].min(), format="%Y%m%d")
    end = pd.to_datetime(audit["Signal_Date"].max(), format="%Y%m%d")
    weeks = pd.date_range(start=start, end=end, freq="W-FRI")
    accepted = ledger.copy()
    accepted["Entry_dt"] = pd.to_datetime(accepted["Entry_Date"], format="%Y%m%d", errors="coerce")
    accepted["Exit_dt"] = pd.to_datetime(accepted["Actual_Exit_Date"], format="%Y%m%d", errors="coerce")
    rows: list[dict[str, Any]] = []
    for week_end in weeks:
        week_start = week_end - pd.Timedelta(days=6)
        held = accepted[
            accepted["Entry_dt"].le(week_end)
            & accepted["Exit_dt"].ge(week_start)
        ]
        rows.append({
            "Week_End": week_end.strftime("%Y%m%d"),
            "Covered": not held.empty,
            "Trades_Touching_Week": len(held),
            "Stocks": "|".join(held["name"].astype(str)),
        })
    report = pd.DataFrame(rows)
    covered = int(report["Covered"].sum()) if len(report) else 0
    summary = {
        "区间周数": len(report),
        "实际持仓覆盖周": covered,
        "完全空仓周": len(report) - covered,
        "持仓周覆盖率(%)": covered / len(report) * 100.0 if len(report) else np.nan,
    }
    return report, summary


def trade_year_report(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    work = ledger.copy()
    work["Entry_Year"] = work["Entry_Date"].astype(str).str[:4]
    rows: list[dict[str, Any]] = []
    for year, group in work.groupby("Entry_Year", sort=True):
        returns = pd.to_numeric(group["Portfolio_Return_pct"], errors="coerce")
        pnl = pd.to_numeric(group["PnL"], errors="coerce")
        rows.append({
            "Entry_Year": year,
            "交易数": len(group),
            "平均单笔收益(%)": returns.mean(),
            "中位单笔收益(%)": returns.median(),
            "胜率(%)": returns.gt(0).mean() * 100.0,
            "止损率(%)": group["Exit_Reason"].astype(str).str.contains("止损").mean() * 100.0,
            "已实现利润": pnl.sum(),
        })
    return pd.DataFrame(rows)


def calendar_year_report(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    work = curve.copy()
    work["Year"] = work["Trade_Date"].astype(str).str[:4]
    rows: list[dict[str, Any]] = []
    previous_equity = INITIAL_CAPITAL
    for year, group in work.groupby("Year", sort=True):
        end_equity = float(group.iloc[-1]["Equity"])
        rows.append({
            "Year": year,
            "年初权益": previous_equity,
            "年末权益": end_equity,
            "年度收益率(%)": (end_equity / previous_equity - 1.0) * 100.0,
            "年度最大回撤(%)": float(group["Drawdown_pct"].min()),
            "平均持仓数": float(group["Positions"].mean()),
            "空仓交易日比例(%)": float(group["Is_Empty"].mean() * 100.0),
        })
        previous_equity = end_equity
    return pd.DataFrame(rows)


def candidate_report(audit: pd.DataFrame, candidates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, frame in (
        ("全部第一根红柱", audit),
        ("V4.7候选池（已取消波浪和实体阳线硬门槛）", candidates),
        ("每周评分第一名", selected),
    ):
        returns = pd.to_numeric(frame.get("Exit_T30_Return_pct", pd.Series(dtype=float)), errors="coerce")
        rows.append({
            "阶段": name,
            "事件数": len(frame),
            "有信号周": frame["Signal_Date"].nunique() if len(frame) else 0,
            "平均收益(%)": returns.mean(),
            "中位收益(%)": returns.median(),
            "胜率(%)": returns.gt(0).mean() * 100.0 if len(returns) else np.nan,
            "止损率(%)": frame["Exit_T30_Reason"].astype(str).str.contains("止损").mean() * 100.0 if len(frame) else np.nan,
        })
    return pd.DataFrame(rows)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def make_result(
    audit: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    trade_days: list[str],
    source: str,
    fetch_failures: pd.DataFrame,
    cache_hits: int,
) -> dict[str, Any]:
    interval_start = audit["Signal_Date"].min()
    interval_end = audit["Signal_Date"].max()
    curve, ledger, orders, missed, summary = simulate_portfolio(
        selected, histories, trade_days, interval_start, interval_end
    )
    coverage, coverage_summary = weekly_coverage(audit, ledger)
    summary.update(coverage_summary)
    summary_frame = pd.DataFrame([summary])
    candidates_summary = candidate_report(audit, candidates, selected)
    trade_year = trade_year_report(ledger)
    calendar_year = calendar_year_report(curve)

    metadata = pd.DataFrame([
        {"项目": "程序", "值": TITLE},
        {"项目": "输入", "值": source},
        {"项目": "生成时间", "值": datetime.now().isoformat(timespec="seconds")},
        {"项目": "初始资金", "值": INITIAL_CAPITAL},
        {"项目": "仓位", "值": "最多3只；每只目标10万元；100股整数倍"},
        {"项目": "正式信号", "值": "V4.7候选池内每周New_Score_100第一名"},
        {"项目": "取消硬门槛", "值": "真正周线2—5波浪、实体阳线≥60%"},
        {"项目": "仍保留信息", "值": "波浪数保留审计；实体质量保留评分"},
        {"项目": "退出", "值": "沿用V4.6事件的+30%止盈、-10%止损、最长八周及交易成本"},
        {"项目": "同日顺序", "值": "开盘先尝试买入，当日退出的旧仓不提前释放仓位"},
        {"项目": "行情缓存命中", "值": cache_hits},
        {"项目": "行情下载失败", "值": len(fetch_failures)},
    ])
    files = {
        "01_portfolio_summary_v4_7.csv": csv_bytes(summary_frame),
        "02_portfolio_curve_v4_7.csv": csv_bytes(curve),
        "03_portfolio_ledger_v4_7.csv": csv_bytes(ledger),
        "04_portfolio_orders_v4_7.csv": csv_bytes(orders),
        "05_missed_signals_v4_7.csv": csv_bytes(missed),
        "06_selected_top1_v4_7.csv": csv_bytes(selected),
        "07_candidate_pool_v4_7.csv": csv_bytes(candidates),
        "08_weekly_coverage_v4_7.csv": csv_bytes(coverage),
        "09_trade_year_report_v4_7.csv": csv_bytes(trade_year),
        "10_calendar_year_report_v4_7.csv": csv_bytes(calendar_year),
        "11_candidate_summary_v4_7.csv": csv_bytes(candidates_summary),
        "12_fetch_failures_v4_7.csv": csv_bytes(fetch_failures),
        "13_metadata_v4_7.csv": csv_bytes(metadata),
    }
    return {
        "summary": summary,
        "summary_frame": summary_frame,
        "curve": curve,
        "ledger": ledger,
        "orders": orders,
        "missed": missed,
        "selected": selected,
        "candidates": candidates,
        "coverage": coverage,
        "trade_year": trade_year,
        "calendar_year": calendar_year,
        "candidates_summary": candidates_summary,
        "fetch_failures": fetch_failures,
        "metadata": metadata,
        "files": files,
        "zip": zip_bytes(files),
    }


def show(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("本表没有记录。")
        return
    formats = {
        column: "{:.2f}"
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and any(word in column for word in ("(%)", "收益", "回撤", "权益", "资金", "利润", "暴露", "因子", "平均", "中位"))
    }
    st.dataframe(frame.style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)


def render(result: dict[str, Any]) -> None:
    summary = result["summary"]
    row1 = st.columns(5)
    row1[0].metric("评分第一名信号", f"{summary['评分第一名信号']:,}")
    row1[1].metric("实际买入", f"{summary['实际买入']:,}")
    row1[2].metric("仓位满错过", f"{summary['仓位满错过']:,}")
    row1[3].metric("期末权益", f"¥{summary['期末权益']:,.0f}")
    row1[4].metric("总收益", f"{summary['总收益率(%)']:.2f}%")
    row2 = st.columns(5)
    row2[0].metric("最大回撤", f"{summary['最大回撤(%)']:.2f}%")
    row2[1].metric("交易胜率", f"{summary['交易胜率(%)']:.2f}%")
    row2[2].metric("止损率", f"{summary['止损率(%)']:.2f}%")
    row2[3].metric("持仓周覆盖", f"{summary['实际持仓覆盖周']}/{summary['区间周数']}")
    row2[4].metric("完全空仓周", f"{summary['完全空仓周']}")

    st.subheader("组合资金曲线")
    chart = result["curve"][["Trade_Date", "Equity"]].copy()
    chart["Trade_Date"] = pd.to_datetime(chart["Trade_Date"], format="%Y%m%d")
    st.line_chart(chart.set_index("Trade_Date"))

    st.subheader("组合总表")
    show(result["summary_frame"])
    st.subheader("候选池与第一名")
    show(result["candidates_summary"])
    st.subheader("分年度表现")
    show(result["calendar_year"])
    show(result["trade_year"])

    with st.expander("交易、错过信号和持仓覆盖明细", expanded=False):
        st.markdown("**实际成交账本**")
        show(result["ledger"])
        st.markdown("**仓位满或其他原因错过的信号**")
        show(result["missed"])
        st.markdown("**每周持仓覆盖**")
        show(result["coverage"])

    st.subheader("下载结果")
    st.download_button(
        "下载全部结果ZIP",
        result["zip"],
        file_name="weekly_macd_portfolio_v4_7_all_results.zip",
        mime="application/zip",
        type="primary",
        key="v47_all_results",
        on_click="ignore",
    )
    labels = [
        "1号：组合总表", "2号：资金曲线", "3号：实际成交账本", "4号：全部下单审计",
        "5号：错过信号", "6号：评分第一名", "7号：V4.7候选池", "8号：持仓周覆盖",
        "9号：按交易年份", "10号：按自然年份", "11号：候选池摘要", "12号：行情失败",
        "13号：运行信息",
    ]
    columns = st.columns(4)
    for index, (name, data) in enumerate(result["files"].items()):
        with columns[index % 4]:
            st.download_button(
                labels[index], data, file_name=name, mime="text/csv",
                key=f"v47_{name}", on_click="ignore",
            )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.info(
        "上传V4.6全部结果ZIP。本版不会重新扫描全部股票，只从V4.6事件审计中生成V4.7的37个左右评分第一名，"
        "再下载这些入选股票的日线，进行30万元、最多三仓的真实资金占用模拟。"
    )
    with st.expander("V4.7固定规则", expanded=False):
        st.markdown(
            """
            - 取消硬门槛：真正周线2—5次波浪、实体阳线≥60%。
            - 保留硬门槛：真正第一根红柱、周线乖离和上影、日线趋势、箱体突破、MA20、成交量、日线MACD及T+1开盘限制。
            - 实体质量仍参与`New_Score_100`，波浪数量仍保留在审计文件中。
            - 每周只买评分第一名；30万元；最多3只；每只目标10万元；100股整数倍。
            - 买入、退出价格和交易成本沿用V4.6的事件口径：+30%止盈、-10%止损、最长八周。
            - 与V40.4/V40.6一致：同一交易日先在开盘尝试新买入，再处理旧仓退出。
            """
        )

    with st.form("v47_form"):
        uploaded = st.file_uploader(
            "上传V4.6全部结果ZIP或01_hybrid_event_audit.csv",
            type=["zip", "csv"],
        )
        token = st.text_input("Tushare Token", type="password")
        c1, c2 = st.columns(2)
        use_cache = c1.checkbox("使用入选股票日线缓存", value=True)
        pause = c2.number_input(
            "每只股票请求后暂停(秒)", min_value=0.05, max_value=2.0,
            value=0.12, step=0.05,
        )
        submitted = st.form_submit_button("开始V4.7三仓组合回测", type="primary")

    if submitted:
        if uploaded is None:
            st.error("请上传V4.6全部结果ZIP或事件审计CSV。")
        elif not token.strip():
            st.error("请输入Tushare Token，用于下载入选股票的日线并计算真实资金曲线。")
        else:
            try:
                raw, source = load_v46_audit(uploaded)
                validate_audit(raw)
                audit, candidates, selected = prepare_v47(raw)
                if selected.empty:
                    raise ValueError("取消波浪和实体阳线硬门槛后仍没有评分第一名信号。")

                start = min(audit["Signal_Date"].min(), selected["Entry_Date"].min())
                end = selected["Exit_T30_Date"].max()
                codes = sorted(selected["ts_code"].astype(str).unique())
                ts.set_token(token.strip())
                pro = ts.pro_api(token.strip())
                histories: dict[str, pd.DataFrame] = {}
                failures: list[dict[str, str]] = []
                cache_hits = 0
                progress = st.progress(0.0, text="准备入选股票日线...")
                for index, code in enumerate(codes, start=1):
                    history, cache_hit, error = fetch_daily(
                        pro, code, start, end, bool(use_cache), float(pause)
                    )
                    histories[code] = history
                    cache_hits += int(cache_hit)
                    if history.empty:
                        failures.append({"ts_code": code, "错误": error})
                    progress.progress(
                        index / len(codes),
                        text=f"入选股票日线：{index}/{len(codes)}；缓存命中{cache_hits}",
                    )
                progress.empty()
                if failures:
                    failed_codes = {row["ts_code"] for row in failures}
                    selected_codes = set(selected["ts_code"].astype(str))
                    if failed_codes & selected_codes:
                        raise ValueError(
                            "有入选股票行情下载失败，不能用不完整行情计算回撤。请保留缓存后重试。"
                        )

                trade_days = fetch_trade_days(pro, start, end, histories)
                if not trade_days:
                    raise ValueError("无法取得交易日历。")
                with st.spinner("正在执行30万元、三仓、评分第一名组合模拟..."):
                    failure_frame = pd.DataFrame(failures, columns=["ts_code", "错误"])
                    result = make_result(
                        audit, candidates, selected, histories, trade_days,
                        source, failure_frame, cache_hits,
                    )
                    st.session_state["v47_result"] = result
                st.success(
                    f"完成：V4.7候选{len(candidates)}个，评分第一名{len(selected)}个，"
                    f"组合实际买入{result['summary']['实际买入']}个。"
                )
            except Exception as exc:
                st.exception(exc)

    if "v47_result" in st.session_state:
        render(st.session_state["v47_result"])
    else:
        st.caption("下载按钮不会清空结果。重复运行时，入选股票日线缓存会显著缩短等待时间。")


if __name__ == "__main__":
    main()
