# -*- coding: utf-8 -*-
"""
选股王 V39.8 · 市场环境与动态门槛增强版
============================================================
主要变化
1. 股票池：主板/创业板/科创板，明确剔除北交所；保留流通市值200~1000亿元参数。
2. 科技池：使用申万2021 L2/L3历史成分，不再只用六个过粗的L1行业。
3. 概念池：THS科技概念仅作为可选的实时雷达补充；历史回测默认关闭，避免未来信息。
4. 信号：核心条件为硬门槛，其余条件改为质量评分，减少多个AND条件造成的误杀。
5. 排名：取消对单日涨幅的线性追涨奖励，增加突破质量、量能、K线、MACD、相对强度评分。
6. 风控：使用信号低点/MA20/ATR构造结构止损，并保留最大风险上限。
7. 成交：T+1开盘买入，加入滑点、佣金和卖出税费；过滤真正的一字涨停与过度高开。
8. 统计：退出收益向后延续，另设Eligible/Held字段，修复周度幸存者偏差。
9. 数据：缓存检查日期覆盖并增量补齐；行业/API失败不再静默放行。
10. 诊断：输出逐日筛选漏斗、退出周、持仓天数、MFE/MAE等字段。
11. 门槛：当天80分以上核心信号达到4只时门槛为80分，否则提高到82分，但不禁止开仓。
12. 环境：创业板指与科创50同时跌破下降中的MA20时，暂停次日新开仓。
13. 保本：浮盈首次达到10%后，从下一交易日启用成本上方0.3%的保护止损。
14. 审计：另行导出全部核心候选及未入选原因，便于核查动态门槛和市场过滤。

说明
- 本程序用于研究和回测，不构成投资建议。
- 默认申万接口需要Tushare 2000积分；THS概念接口需要6000积分，默认关闭。
- 第一次运行或扩大回测区间时需要下载数据，之后会增量使用缓存。
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import pickle
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import tushare as ts

warnings.filterwarnings("ignore")

VERSION = "V39.8"
# 行情结构与V39.7兼容，沿用原缓存可避免重新下载数百个交易日。
CACHE_FILE_NAME = "market_data_cache_v39_7.pkl"
MAX_FETCH_WORKERS = 1  # 避免触发Tushare频率限制

pro = None
GLOBAL_ADJ_FACTOR = pd.DataFrame()
GLOBAL_DAILY_RAW = pd.DataFrame()
GLOBAL_QFQ_BASE_FACTORS: dict[str, float] = {}
GLOBAL_STOCK_BASIC = pd.DataFrame()
GLOBAL_TECH_PERIODS: dict[str, list[dict]] = {}
GLOBAL_THS_TECH_CODES: set[str] = set()
GLOBAL_BENCHMARK = pd.DataFrame()
GLOBAL_REGIME_INDICES: dict[str, pd.DataFrame] = {}
API_ERRORS: list[str] = []
SINA_STATUS = {"success": 0, "fail": 0}


# -----------------------------------------------------------------------------
# 科技行业配置：核心L1全部保留；扩展L1仅保留命中L2/L3关键词的公司
# -----------------------------------------------------------------------------
CORE_TECH_L1 = {"电子", "计算机", "通信", "国防军工"}

EXTENDED_TECH_L1 = {
    "机械设备",
    "电力设备",
    "医药生物",
    "汽车",
    "基础化工",
    "有色金属",
    "建筑材料",
}

TECH_INDUSTRY_KEYWORDS = {
    "半导体",
    "电子元件",
    "元件",
    "光学光电子",
    "消费电子",
    "电子化学品",
    "计算机设备",
    "软件开发",
    "IT服务",
    "通信设备",
    "军工电子",
    "航空装备",
    "航天装备",
    "自动化设备",
    "机器人",
    "激光设备",
    "工控设备",
    "仪器仪表",
    "电池",
    "光伏设备",
    "风电设备",
    "电网设备",
    "电机",
    "医疗器械",
    "生物制品",
    "汽车电子",
    "金属新材料",
    "非金属材料",
    "膜材料",
    "碳纤维",
}

DEFAULT_THS_KEYWORDS = (
    "人工智能,算力,服务器,数据中心,机器人,机器视觉,半导体,芯片,先进封装,"
    "存储,CPO,光模块,卫星互联网,6G,低空经济,无人机,商业航天,固态电池,"
    "汽车电子,创新药,基因,脑机接口"
)


# -----------------------------------------------------------------------------
# 页面
# -----------------------------------------------------------------------------
st.set_page_config(page_title=f"选股王 {VERSION} 科技细分增强版", layout="wide")
st.title(f"选股王 {VERSION}：动态评分 + 双指数环境过滤 + 次日保护止损")


# -----------------------------------------------------------------------------
# 通用工具
# -----------------------------------------------------------------------------
def record_api_error(message: str) -> None:
    if len(API_ERRORS) < 300:
        API_ERRORS.append(message)


def safe_get(func_name: str, required: bool = False, retries: int = 3, **kwargs) -> pd.DataFrame:
    global pro
    if pro is None:
        msg = f"{func_name}: Tushare尚未初始化"
        record_api_error(msg)
        if required:
            raise RuntimeError(msg)
        return pd.DataFrame()

    try:
        func = getattr(pro, func_name)
    except Exception as exc:
        msg = f"当前Tushare SDK不支持接口 {func_name}: {exc}"
        record_api_error(msg)
        if required:
            raise RuntimeError(msg) from exc
        return pd.DataFrame()

    last_error = None
    for attempt in range(retries):
        try:
            result = func(**kwargs)
            if result is not None and not result.empty:
                return result
            last_error = RuntimeError("接口返回空数据")
        except Exception as exc:
            last_error = exc
        time.sleep(0.5 + attempt * 0.6)

    msg = f"{func_name}({kwargs})失败: {last_error}"
    record_api_error(msg)
    if required:
        raise RuntimeError(msg) from last_error
    return pd.DataFrame()


def normalize_date(value, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip().replace(".0", "")
    return text if len(text) == 8 and text.isdigit() else default


def parse_keywords(text: str) -> list[str]:
    normalized = text.replace("，", ",").replace("、", ",").replace("\n", ",")
    return sorted({x.strip() for x in normalized.split(",") if x.strip()})


def get_trade_days(end_date_str: str, num_days: int) -> list[str]:
    lookback_days = max(num_days * 3, 365)
    start_date = (
        datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)
    ).strftime("%Y%m%d")
    cal = safe_get(
        "trade_cal",
        required=True,
        exchange="SSE",
        start_date=start_date,
        end_date=end_date_str,
    )
    cal = cal[(cal["is_open"] == 1) & (cal["cal_date"] <= end_date_str)]
    return cal.sort_values("cal_date", ascending=False)["cal_date"].head(num_days).tolist()


def get_open_dates(start_date: str, end_date: str) -> list[str]:
    cal = safe_get(
        "trade_cal",
        required=True,
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
    )
    return sorted(cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).tolist())


def get_sina_realtime_kline(ts_code: str):
    global SINA_STATUS
    code_split = ts_code.split(".")
    if len(code_split) != 2:
        return None
    sina_code = code_split[1].lower() + code_split[0]
    url = f"http://hq.sinajs.cn/list={sina_code}"
    headers = {"Referer": "https://finance.sina.com.cn"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = "gbk"
        data_str = response.text.split('="')[1].split('";')[0]
        if not data_str:
            SINA_STATUS["fail"] += 1
            return None
        data = data_str.split(",")
        now = datetime.now()
        elapsed = 0
        morning_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        morning_end = now.replace(hour=11, minute=30, second=0, microsecond=0)
        afternoon_start = now.replace(hour=13, minute=0, second=0, microsecond=0)
        afternoon_end = now.replace(hour=15, minute=0, second=0, microsecond=0)
        if now <= morning_start:
            elapsed = 1
        elif now <= morning_end:
            elapsed = int((now - morning_start).total_seconds() / 60)
        elif now <= afternoon_start:
            elapsed = 120
        elif now <= afternoon_end:
            elapsed = 120 + int((now - afternoon_start).total_seconds() / 60)
        else:
            elapsed = 240
        elapsed = max(1, min(elapsed, 240))
        projected_vol = (float(data[8]) / 100.0) * (240.0 / elapsed)
        SINA_STATUS["success"] += 1
        return {
            "trade_date_str": now.strftime("%Y%m%d"),
            "open": float(data[1]),
            "pre_close": float(data[2]),
            "close": float(data[3]),
            "high": float(data[4]),
            "low": float(data[5]),
            "vol": projected_vol,
        }
    except Exception as exc:
        SINA_STATUS["fail"] += 1
        record_api_error(f"新浪实时行情 {ts_code} 失败: {exc}")
        return None


# -----------------------------------------------------------------------------
# 股票基础信息、行业池和概念池
# -----------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600)
def load_stock_basic() -> pd.DataFrame:
    frames = []
    fields = "ts_code,symbol,name,market,exchange,list_status,list_date,delist_date"
    for status in ["L", "P", "D"]:
        df = safe_get("stock_basic", list_status=status, fields=fields)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("stock_basic加载失败，无法确认主板/双创/北交所范围")
    result = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
    result = result[
        result["market"].isin(["主板", "创业板", "科创板"])
        & result["exchange"].ne("BSE")
        & ~result["ts_code"].str.endswith(".BJ", na=False)
    ].copy()
    return result


def industry_row_is_tech(row: pd.Series) -> bool:
    l1 = str(row.get("l1_name", ""))
    l2 = str(row.get("l2_name", ""))
    l3 = str(row.get("l3_name", ""))
    if l1 in CORE_TECH_L1:
        return True
    if l1 not in EXTENDED_TECH_L1:
        return False
    combined = f"{l2}|{l3}"
    return any(keyword in combined for keyword in TECH_INDUSTRY_KEYWORDS)


@st.cache_data(ttl=7 * 24 * 3600)
def load_sw_tech_memberships() -> pd.DataFrame:
    l1_df = safe_get("index_classify", required=True, level="L1", src="SW2021")
    if l1_df.empty:
        raise RuntimeError("申万2021一级行业目录为空")

    target_names = CORE_TECH_L1 | EXTENDED_TECH_L1
    target_l1 = l1_df[l1_df["industry_name"].isin(target_names)]
    if target_l1.empty:
        raise RuntimeError("未找到目标申万科技行业，请检查Tushare数据版本")

    frames = []
    progress = st.progress(0, text="正在构建申万L2/L3科技历史成分池...")
    targets = target_l1[["index_code", "industry_name"]].to_dict("records")
    total_calls = len(targets) * 2
    call_no = 0
    for item in targets:
        for flag in ["Y", "N"]:
            df = safe_get("index_member_all", l1_code=item["index_code"], is_new=flag)
            call_no += 1
            progress.progress(
                call_no / max(total_calls, 1),
                text=f"申万科技池: {item['industry_name']} ({flag})",
            )
            if not df.empty:
                frames.append(df)
            time.sleep(0.05)
    progress.empty()

    if not frames:
        raise RuntimeError(
            "index_member_all未返回数据。请确认Tushare SDK已更新且账户具有2000积分权限。"
        )

    result = pd.concat(frames, ignore_index=True)
    for col in ["l1_name", "l2_name", "l3_name", "in_date", "out_date", "is_new"]:
        if col not in result.columns:
            result[col] = ""
    result = result[result.apply(industry_row_is_tech, axis=1)].copy()
    result["in_date"] = result["in_date"].apply(lambda x: normalize_date(x, "19000101"))
    result["out_date"] = result["out_date"].apply(lambda x: normalize_date(x, "99991231"))
    result = result.drop_duplicates(
        ["ts_code", "l1_name", "l2_name", "l3_name", "in_date", "out_date"]
    )
    if result.empty:
        raise RuntimeError("科技行业关键词过滤后成分为空，请检查行业名称")
    return result


def build_tech_period_index(memberships: pd.DataFrame) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for row in memberships.itertuples(index=False):
        code = str(row.ts_code)
        index.setdefault(code, []).append(
            {
                "in_date": str(row.in_date),
                "out_date": str(row.out_date),
                "l1": str(row.l1_name),
                "l2": str(row.l2_name),
                "l3": str(row.l3_name),
            }
        )
    return index


def get_tech_membership(ts_code: str, trade_date: str):
    for period in GLOBAL_TECH_PERIODS.get(ts_code, []):
        if period["in_date"] <= trade_date < period["out_date"]:
            return period
    if ts_code in GLOBAL_THS_TECH_CODES:
        return {"in_date": trade_date, "out_date": "99991231", "l1": "THS概念", "l2": "", "l3": ""}
    return None


@st.cache_data(ttl=24 * 3600)
def load_ths_current_tech_pool(keywords_tuple: tuple[str, ...]) -> tuple[set[str], pd.DataFrame]:
    index_frames = []
    for index_type in ["N", "TH"]:
        df = safe_get("ths_index", exchange="A", type=index_type)
        if not df.empty:
            index_frames.append(df)
    if not index_frames:
        raise RuntimeError("THS概念接口不可用；请确认账户具有6000积分权限")
    indices = pd.concat(index_frames, ignore_index=True).drop_duplicates("ts_code")
    mask = indices["name"].astype(str).apply(
        lambda name: any(keyword in name for keyword in keywords_tuple)
    )
    selected = indices[mask].copy().head(40)
    if selected.empty:
        raise RuntimeError("未找到匹配的THS科技概念")
    codes: set[str] = set()
    for row in selected.itertuples():
        members = safe_get("ths_member", ts_code=row.ts_code)
        if not members.empty and "con_code" in members.columns:
            codes.update(members["con_code"].dropna().astype(str).tolist())
    return codes, selected[["ts_code", "name", "count"]]


# -----------------------------------------------------------------------------
# 行情缓存与复权
# -----------------------------------------------------------------------------
def fetch_daily_bundle(date: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    daily = safe_get("daily", trade_date=date)
    adj = safe_get("adj_factor", trade_date=date)
    return date, daily, adj


def standardize_market_frames(daily: pd.DataFrame, adj: pd.DataFrame):
    if not daily.empty:
        daily = daily.copy()
        daily["trade_date"] = daily["trade_date"].astype(str)
        daily = daily.drop_duplicates(["ts_code", "trade_date"])
        daily = daily.set_index(["ts_code", "trade_date"]).sort_index()
    if not adj.empty:
        adj = adj.copy()
        adj["trade_date"] = adj["trade_date"].astype(str)
        adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
        adj = adj.dropna(subset=["adj_factor"])
        adj = adj.drop_duplicates(["ts_code", "trade_date"])
        adj = adj.set_index(["ts_code", "trade_date"]).sort_index()
    return daily, adj


def load_market_data(required_dates: list[str], use_cache: bool = True) -> None:
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS

    cached_daily = pd.DataFrame()
    cached_adj = pd.DataFrame()
    if use_cache and os.path.exists(CACHE_FILE_NAME):
        try:
            with open(CACHE_FILE_NAME, "rb") as file:
                cached = pickle.load(file)
            cached_daily = cached.get("daily", pd.DataFrame())
            cached_adj = cached.get("adj", pd.DataFrame())
            if not isinstance(cached_daily.index, pd.MultiIndex):
                cached_daily, cached_adj = standardize_market_frames(cached_daily, cached_adj)
        except Exception as exc:
            record_api_error(f"旧缓存读取失败，将重新下载: {exc}")
            cached_daily = pd.DataFrame()
            cached_adj = pd.DataFrame()

    cached_dates = set()
    if not cached_daily.empty and isinstance(cached_daily.index, pd.MultiIndex):
        cached_dates = set(cached_daily.index.get_level_values("trade_date").astype(str))
    missing_dates = [d for d in required_dates if d not in cached_dates]

    new_daily_frames = []
    new_adj_frames = []
    failed_dates = []
    if missing_dates:
        progress = st.progress(0, text="正在增量下载行情和复权因子...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as executor:
            futures = {executor.submit(fetch_daily_bundle, d): d for d in missing_dates}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                date = futures[future]
                try:
                    _, daily, adj = future.result()
                    if daily.empty or adj.empty:
                        failed_dates.append(date)
                    else:
                        new_daily_frames.append(daily)
                        new_adj_frames.append(adj)
                except Exception as exc:
                    failed_dates.append(date)
                    record_api_error(f"下载交易日 {date} 失败: {exc}")
                progress.progress((i + 1) / len(missing_dates), text=f"行情下载 {i+1}/{len(missing_dates)}")
        progress.empty()

    new_daily = pd.concat(new_daily_frames, ignore_index=True) if new_daily_frames else pd.DataFrame()
    new_adj = pd.concat(new_adj_frames, ignore_index=True) if new_adj_frames else pd.DataFrame()
    new_daily, new_adj = standardize_market_frames(new_daily, new_adj)

    if cached_daily.empty:
        GLOBAL_DAILY_RAW = new_daily
    elif new_daily.empty:
        GLOBAL_DAILY_RAW = cached_daily
    else:
        GLOBAL_DAILY_RAW = pd.concat([cached_daily, new_daily])
        GLOBAL_DAILY_RAW = GLOBAL_DAILY_RAW[~GLOBAL_DAILY_RAW.index.duplicated(keep="last")].sort_index()

    if cached_adj.empty:
        GLOBAL_ADJ_FACTOR = new_adj
    elif new_adj.empty:
        GLOBAL_ADJ_FACTOR = cached_adj
    else:
        GLOBAL_ADJ_FACTOR = pd.concat([cached_adj, new_adj])
        GLOBAL_ADJ_FACTOR = GLOBAL_ADJ_FACTOR[~GLOBAL_ADJ_FACTOR.index.duplicated(keep="last")].sort_index()

    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty:
        raise RuntimeError("行情或复权数据为空，无法回测")

    # 每只股票使用自身最后一个可用复权因子，避免要求所有股票在同一最新日期都有记录
    adj_reset = GLOBAL_ADJ_FACTOR.reset_index().sort_values(["ts_code", "trade_date"])
    GLOBAL_QFQ_BASE_FACTORS = (
        adj_reset.groupby("ts_code", sort=False).tail(1).set_index("ts_code")["adj_factor"].to_dict()
    )

    try:
        with open(CACHE_FILE_NAME, "wb") as file:
            pickle.dump(
                {
                    "version": VERSION,
                    "saved_at": datetime.now().isoformat(),
                    "daily": GLOBAL_DAILY_RAW,
                    "adj": GLOBAL_ADJ_FACTOR,
                },
                file,
            )
    except Exception as exc:
        record_api_error(f"缓存保存失败: {exc}")

    if failed_dates:
        st.warning(f"有 {len(failed_dates)} 个交易日下载失败；示例: {failed_dates[:8]}")


def load_benchmark(start_date: str, end_date: str) -> None:
    global GLOBAL_BENCHMARK
    df = safe_get(
        "index_daily",
        ts_code="000300.SH",
        start_date=start_date,
        end_date=end_date,
    )
    if df.empty:
        GLOBAL_BENCHMARK = pd.DataFrame()
        record_api_error("沪深300指数数据为空，相对强度分将按中性值处理")
        return
    df["trade_date"] = df["trade_date"].astype(str)
    GLOBAL_BENCHMARK = df.sort_values("trade_date").set_index("trade_date")


def load_market_regime_indices(start_date: str, end_date: str) -> None:
    """一次性载入创业板指和科创50，所有判断只使用信号日及以前的数据。"""
    global GLOBAL_REGIME_INDICES
    GLOBAL_REGIME_INDICES = {}
    index_names = {
        "399006.SZ": "创业板指",
        "000688.SH": "科创50",
    }
    for ts_code, name in index_names.items():
        df = safe_get(
            "index_daily",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if df.empty:
            record_api_error(f"{name}({ts_code})数据为空；市场环境过滤对该指数按未知处理")
            continue
        df["trade_date"] = df["trade_date"].astype(str)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.sort_values("trade_date").dropna(subset=["close"]).copy()
        df["ma20"] = df["close"].rolling(20, min_periods=20).mean()
        df["ma20_prev"] = df["ma20"].shift(1)
        GLOBAL_REGIME_INDICES[ts_code] = df.set_index("trade_date")


def market_regime_on_date(trade_date: str, enabled: bool = True) -> dict:
    """双指数同时处于弱势才阻止开仓；单指数弱或数据不足均不阻止。"""
    result = {
        "Market_Filter_Enabled": bool(enabled),
        "Market_Weak_Block": False,
        "Market_Regime": "过滤关闭" if not enabled else "数据不足-允许开仓",
        "CYB_Close": np.nan,
        "CYB_MA20": np.nan,
        "CYB_MA20_Falling": False,
        "STAR50_Close": np.nan,
        "STAR50_MA20": np.nan,
        "STAR50_MA20_Falling": False,
    }
    if not enabled:
        return result

    weak_flags = []
    labels = {
        "399006.SZ": ("CYB", "创业板指"),
        "000688.SH": ("STAR50", "科创50"),
    }
    descriptions = []
    for ts_code, (prefix, name) in labels.items():
        df = GLOBAL_REGIME_INDICES.get(ts_code, pd.DataFrame())
        subset = df[df.index <= trade_date] if not df.empty else pd.DataFrame()
        if subset.empty:
            descriptions.append(f"{name}缺数据")
            continue
        row = subset.iloc[-1]
        close = float(row["close"])
        ma20 = float(row["ma20"]) if pd.notna(row["ma20"]) else np.nan
        ma20_prev = float(row["ma20_prev"]) if pd.notna(row["ma20_prev"]) else np.nan
        falling = bool(pd.notna(ma20) and pd.notna(ma20_prev) and ma20 < ma20_prev)
        below = bool(pd.notna(ma20) and close < ma20)
        weak = below and falling
        result[f"{prefix}_Close"] = round(close, 3)
        result[f"{prefix}_MA20"] = round(ma20, 3) if pd.notna(ma20) else np.nan
        result[f"{prefix}_MA20_Falling"] = falling
        weak_flags.append(weak)
        descriptions.append(f"{name}{'弱' if weak else '非弱'}")

    if len(weak_flags) == 2:
        result["Market_Weak_Block"] = bool(all(weak_flags))
        result["Market_Regime"] = "双指数弱势-暂停开仓" if all(weak_flags) else "允许开仓(" + "、".join(descriptions) + ")"
    else:
        result["Market_Regime"] = "数据不足-允许开仓(" + "、".join(descriptions) + ")"
    return result


def benchmark_return_20d(end_date: str) -> float:
    if GLOBAL_BENCHMARK.empty:
        return 0.0
    subset = GLOBAL_BENCHMARK[GLOBAL_BENCHMARK.index <= end_date]
    if len(subset) < 21:
        return 0.0
    return float(subset.iloc[-1]["close"] / subset.iloc[-21]["close"] - 1.0)


def get_qfq_data(ts_code: str, start_date: str, end_date: str, use_sina: bool = False) -> pd.DataFrame:
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty:
        return pd.DataFrame()
    base_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code)
    if base_factor is None or pd.isna(base_factor) or base_factor <= 0:
        return pd.DataFrame()
    try:
        daily = GLOBAL_DAILY_RAW.loc[ts_code]
        daily = daily.loc[(daily.index >= start_date) & (daily.index <= end_date)].copy()
        adj = GLOBAL_ADJ_FACTOR.loc[ts_code]["adj_factor"]
        adj = adj.loc[(adj.index >= start_date) & (adj.index <= end_date)]
    except KeyError:
        return pd.DataFrame()
    if daily.empty or adj.empty:
        return pd.DataFrame()
    df = daily.merge(adj.rename("adj_factor"), left_index=True, right_index=True, how="inner")
    for col in ["open", "high", "low", "close", "pre_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * df["adj_factor"] / base_factor
    df = df.reset_index().rename(columns={"trade_date": "trade_date_str"})
    df["trade_date_str"] = df["trade_date_str"].astype(str)
    df = df.sort_values("trade_date_str").set_index("trade_date_str")
    result = df[["open", "high", "low", "close", "pre_close", "vol"]].copy()

    if use_sina and end_date == datetime.now().strftime("%Y%m%d"):
        realtime = get_sina_realtime_kline(ts_code)
        if realtime and realtime["close"] > 0:
            today = realtime.pop("trade_date_str")
            # 最新复权基准下，非除权日实时原始价与前复权末端价格一致；除权日建议收盘后复跑
            result.loc[today, list(realtime.keys())] = list(realtime.values())
            result = result.sort_index()
    return result.dropna(subset=["open", "high", "low", "close", "vol"])


# -----------------------------------------------------------------------------
# 指标与评分
# -----------------------------------------------------------------------------
def quality_score(condition: bool, points: float) -> float:
    return float(points if condition else 0.0)


def calculate_volume_score(vol_ratio: float) -> float:
    if vol_ratio < 1.10:
        return 0.0
    if vol_ratio <= 2.20:
        return min(15.0, 4.0 + (vol_ratio - 1.10) / 1.10 * 11.0)
    if vol_ratio <= 3.50:
        return max(8.0, 15.0 - (vol_ratio - 2.20) / 1.30 * 7.0)
    return max(0.0, 8.0 - (vol_ratio - 3.50) * 3.0)


def calculate_breakout_score(bias: float) -> float:
    # bias为相对MA20偏离率；0.5%~5%为较健康突破，8%以上不允许入场
    if bias < 0.005:
        return 0.0
    if bias <= 0.05:
        return min(20.0, 8.0 + (bias - 0.005) / 0.045 * 12.0)
    return max(5.0, 20.0 - (bias - 0.05) / 0.03 * 15.0)


def compute_trend_indicators(
    ts_code: str,
    end_date: str,
    min_atr_pct: float,
    min_vol_ratio: float,
    max_bias_pct: float,
    use_sina: bool = False,
) -> dict:
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=450)).strftime("%Y%m%d")
    df = get_qfq_data(ts_code, start_date, end_date, use_sina=use_sina)
    result = {"valid": False}
    if df.empty or len(df) < 140:
        result["reject_reason"] = "历史K线不足140根"
        return result

    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma120"] = df["close"].rolling(120).mean()
    df["ma5_vol"] = df["vol"].shift(1).rolling(5).mean()
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = df["ema12"] - df["ema26"]
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = (df["dif"] - df["dea"]) * 2
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = true_range.rolling(14).mean()

    clean = df.dropna().copy()
    if len(clean) < 5:
        result["reject_reason"] = "指标预热不足"
        return result
    row = clean.iloc[-1]
    prev = clean.iloc[-2]
    recent_prev = clean.iloc[-4:-1]

    # 周线直接从原始日线合成，不再从MA120预热后的df_calc合成
    weekly = df.reset_index().copy()
    weekly["dt"] = pd.to_datetime(weekly["trade_date_str"])
    weekly = weekly.set_index("dt").resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"}
    )
    weekly = weekly.dropna(subset=["close"]).reset_index()
    weekly["w_ma20"] = weekly["close"].rolling(20).mean()
    if len(weekly) < 21 or pd.isna(weekly.iloc[-1]["w_ma20"]):
        result["reject_reason"] = "有效周线不足20周"
        return result
    w_curr = weekly.iloc[-1]
    w_prev = weekly.iloc[-2]
    weekly_bias = (w_curr["close"] - w_curr["w_ma20"]) / w_curr["w_ma20"]
    w_range = w_prev["high"] - w_prev["low"]
    w_upper = w_prev["high"] - max(w_prev["open"], w_prev["close"])
    shadow_ratio = w_upper / w_range if w_range > 0 else 0.0

    weekly_safe = weekly_bias <= 0.45 and shadow_ratio < 0.60
    trend_up = row["ma60"] > row["ma120"]
    pulled_back = any(recent_prev["close"] <= recent_prev["ma20"] * 1.01)
    bias = row["close"] / row["ma20"] - 1.0
    above_ma20 = bias >= 0.005
    not_overextended = bias <= max_bias_pct / 100.0
    ma20_healthy = row["ma20"] >= prev["ma20"] * 0.995
    vol_ratio = row["vol"] / row["ma5_vol"] if row["ma5_vol"] > 0 else 0.0
    volume_ok = vol_ratio >= min_vol_ratio
    candle_range = row["high"] - row["low"]
    body_ratio = max(0.0, row["close"] - row["open"]) / candle_range if candle_range > 0 else 0.0
    close_location = (row["close"] - row["low"]) / candle_range if candle_range > 0 else 0.5
    solid_candle = row["close"] > row["open"] and body_ratio >= 0.40 and close_location >= 0.65
    macd_improving = row["dif"] > prev["dif"] and row["macd"] > prev["macd"]
    atr_pct = row["atr14"] / row["close"] * 100.0
    atr_ok = atr_pct >= min_atr_pct
    day_gain_pct = (row["close"] / prev["close"] - 1.0) * 100.0
    positive_day = day_gain_pct > 0 and row["close"] > row["open"]
    stock_return_20 = row["close"] / clean.iloc[-21]["close"] - 1.0 if len(clean) >= 21 else 0.0
    rs20 = stock_return_20 - benchmark_return_20d(end_date)

    score_breakout = calculate_breakout_score(bias)
    score_volume = calculate_volume_score(vol_ratio)
    score_pullback = quality_score(pulled_back, 15)
    score_candle = min(15.0, max(0.0, body_ratio * 9.0 + close_location * 6.0))
    score_macd = quality_score(macd_improving, 10)
    score_trend = 8.0 + quality_score(ma20_healthy, 4) + min(
        3.0, max(0.0, (row["ma60"] / row["ma120"] - 1.0) * 100)
    )
    score_rs = min(10.0, max(0.0, 5.0 + rs20 * 50.0))
    chase_penalty = min(20.0, max(0.0, day_gain_pct - 10.0) * 2.0)
    total_score = (
        score_breakout
        + score_volume
        + score_pullback
        + score_candle
        + score_macd
        + score_trend
        + score_rs
        - chase_penalty
    )

    core_signal = (
        weekly_safe
        and trend_up
        and positive_day
        and above_ma20
        and not_overextended
        and volume_ok
        and atr_ok
    )
    result.update(
        {
            "valid": True,
            "is_buy_signal": core_signal,
            "weekly_safe": weekly_safe,
            "trend_up": trend_up,
            "positive_day": positive_day,
            "above_ma20": above_ma20,
            "not_overextended": not_overextended,
            "volume_ok": volume_ok,
            "atr_ok": atr_ok,
            "pulled_back": pulled_back,
            "ma20_healthy": ma20_healthy,
            "solid_candle": solid_candle,
            "macd_improving": macd_improving,
            "last_close": float(row["close"]),
            "pre_close": float(prev["close"]),
            "signal_low": float(row["low"]),
            "ma20": float(row["ma20"]),
            "atr14": float(row["atr14"]),
            "atr_pct": float(atr_pct),
            "vol_ratio": float(vol_ratio),
            "body_ratio": float(body_ratio),
            "close_location": float(close_location),
            "bias_pct": float(bias * 100.0),
            "day_gain_pct": float(day_gain_pct),
            "rs20_pct": float(rs20 * 100.0),
            "total_score": float(total_score),
            "score_breakout": float(score_breakout),
            "score_volume": float(score_volume),
            "score_pullback": float(score_pullback),
            "score_candle": float(score_candle),
            "score_macd": float(score_macd),
            "score_trend": float(score_trend),
            "score_rs": float(score_rs),
            "chase_penalty": float(chase_penalty),
        }
    )
    return result


# -----------------------------------------------------------------------------
# T+1成交、结构止损、收益轨迹
# -----------------------------------------------------------------------------
def board_parameters(market: str) -> tuple[float, float, float]:
    if market in ["创业板", "科创板"]:
        return 0.20, 0.07, 0.13  # 涨停幅度、最小止损、最大止损
    return 0.10, 0.05, 0.10


def future_result_template(hold_weeks: int = 8) -> dict:
    result = {}
    for week in range(1, hold_weeks + 1):
        result[f"Return_W{week} (%)"] = np.nan
        result[f"Eligible_W{week}"] = False
        result[f"Held_W{week}"] = False
    result.update(
        {
            "Exit_Reason": "等待T+1数据",
            "Exit_Week": np.nan,
            "Holding_Days": 0,
            "Buy_Price": np.nan,
            "Gap_pct (%)": np.nan,
            "Stop_pct (%)": np.nan,
            "Protection_Trigger_Day": np.nan,
            "Protection_Stop_Price": np.nan,
            "MFE_pct (%)": np.nan,
            "MAE_pct (%)": np.nan,
            "Tradable": True,
        }
    )
    return result


def get_medium_term_future(
    ts_code: str,
    market: str,
    selection_date: str,
    signal_close: float,
    signal_low: float,
    signal_ma20: float,
    atr14: float,
    hold_weeks: int,
    max_gap_pct: float,
    buy_slippage_pct: float,
    sell_slippage_pct: float,
    commission_pct: float,
    sell_tax_pct: float,
    protection_trigger_pct: float,
    protection_buffer_pct: float,
    use_sina: bool = False,
) -> dict:
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_date = (d0 - timedelta(days=70)).strftime("%Y%m%d")
    end_date = (d0 + timedelta(days=110)).strftime("%Y%m%d")
    hist = get_qfq_data(ts_code, start_date, end_date, use_sina=use_sina)
    result = future_result_template(hold_weeks)
    if hist.empty:
        result["Exit_Reason"] = "未来行情缺失"
        return result
    future = hist[hist.index > selection_date].copy()
    if future.empty:
        return result

    next_row = future.iloc[0]
    limit_pct, min_stop_pct, max_stop_pct = board_parameters(market)
    raw_open = float(next_row["open"])
    gap_pct = (raw_open / signal_close - 1.0) * 100.0 if signal_close > 0 else np.nan
    result["Gap_pct (%)"] = round(gap_pct, 2)

    same_price_all_day = (
        np.isclose(next_row["open"], next_row["high"])
        and np.isclose(next_row["open"], next_row["low"])
    )
    at_limit_up = signal_close > 0 and raw_open / signal_close - 1.0 >= limit_pct - 0.005
    if same_price_all_day and at_limit_up:
        result["Exit_Reason"] = "一字涨停无法买入(剔除)"
        result["Tradable"] = False
        return result
    if pd.notna(gap_pct) and gap_pct > max_gap_pct:
        result["Exit_Reason"] = f"T+1高开>{max_gap_pct:.1f}%(剔除)"
        result["Tradable"] = False
        return result
    if raw_open <= 0 or pd.isna(raw_open):
        result["Exit_Reason"] = "T+1开盘价无效"
        result["Tradable"] = False
        return result

    buy_price = raw_open * (1.0 + buy_slippage_pct / 100.0) * (1.0 + commission_pct / 100.0)
    result["Buy_Price"] = round(buy_price, 3)

    raw_structural_stop = min(signal_low, signal_ma20) - 0.5 * atr14
    raw_risk_pct = (buy_price - raw_structural_stop) / buy_price
    risk_pct = float(np.clip(raw_risk_pct, min_stop_pct, max_stop_pct))
    stop_price = buy_price * (1.0 - risk_pct)
    result["Stop_pct (%)"] = round(-risk_pct * 100.0, 2)
    protection_stop_price = buy_price * (1.0 + protection_buffer_pct / 100.0)
    protection_active_from_day = None

    def net_return(raw_sell_price: float) -> float:
        net_sell = raw_sell_price * (1.0 - sell_slippage_pct / 100.0)
        net_sell *= 1.0 - (commission_pct + sell_tax_pct) / 100.0
        return (net_sell / buy_price - 1.0) * 100.0

    def finalize_exit(raw_sell_price: float, week: int, day_count: int, reason: str):
        exit_return = net_return(raw_sell_price)
        result["Exit_Reason"] = reason
        result["Exit_Week"] = week
        result["Holding_Days"] = day_count
        # 退出后的实现收益延续到后续周，修复幸存者偏差
        for w in range(week, hold_weeks + 1):
            result[f"Return_W{w} (%)"] = exit_return
            result[f"Eligible_W{w}"] = True
            result[f"Held_W{w}"] = False
        return exit_return

    tier = 0
    peak_high = buy_price
    trough_low = buy_price
    pending_reason = None
    exited = False
    observed_days = min(len(future), hold_weeks * 5)

    for i in range(observed_days):
        row = future.iloc[i]
        day_count = i + 1
        week = (day_count - 1) // 5 + 1
        curr_open = float(row["open"])
        curr_high = float(row["high"])
        curr_low = float(row["low"])
        curr_close = float(row["close"])
        peak_high = max(peak_high, curr_high)
        trough_low = min(trough_low, curr_low)

        if pending_reason is not None:
            finalize_exit(curr_open, week, day_count, pending_reason)
            exited = True
            break

        protection_active = (
            protection_active_from_day is not None
            and day_count >= protection_active_from_day
        )
        active_stop = max(stop_price, protection_stop_price) if protection_active else stop_price
        if curr_low <= active_stop:
            raw_exit = curr_open if curr_open < active_stop else active_stop
            if protection_active and protection_stop_price >= stop_price:
                reason = f"成本保护止损(+{protection_buffer_pct:.1f}%原始价)"
            else:
                reason = f"结构止损({-risk_pct*100:.1f}%)"
            finalize_exit(raw_exit, week, day_count, reason)
            exited = True
            break

        peak_profit = peak_high / buy_price - 1.0
        if tier == 0 and peak_profit >= protection_trigger_pct / 100.0:
            tier = 1
            protection_active_from_day = day_count + 1
            result["Protection_Trigger_Day"] = day_count
            result["Protection_Stop_Price"] = round(protection_stop_price, 3)
        if tier == 1:
            if peak_profit >= 0.20:
                tier = 2
        if tier == 2 and (peak_high - curr_close) / peak_high >= 0.15:
            pending_reason = "移动止盈(峰值回撤15%)-次日开盘退出"

        if day_count % 5 == 0:
            result[f"Return_W{week} (%)"] = net_return(curr_close)
            result[f"Eligible_W{week}"] = True
            result[f"Held_W{week}"] = True

    result["MFE_pct (%)"] = round((peak_high / buy_price - 1.0) * 100.0, 2)
    result["MAE_pct (%)"] = round((trough_low / buy_price - 1.0) * 100.0, 2)

    if not exited:
        result["Holding_Days"] = observed_days
        if observed_days >= hold_weeks * 5:
            last_close = float(future.iloc[hold_weeks * 5 - 1]["close"])
            finalize_exit(last_close, hold_weeks, hold_weeks * 5, "周期结束平仓")
            # 周期末强制平仓代表已经完整存活到W8末，不应计为W8前退出。
            result[f"Held_W{hold_weeks}"] = True
        else:
            result["Exit_Reason"] = "持仓中/观察期未满"
    return result


# -----------------------------------------------------------------------------
# 单日回测与筛选漏斗
# -----------------------------------------------------------------------------
def stock_active_on_date(row, trade_date: str) -> bool:
    list_date = normalize_date(getattr(row, "list_date", ""), "19000101")
    delist_date = normalize_date(getattr(row, "delist_date", ""), "99991231")
    return list_date <= trade_date < delist_date


def run_backtest_for_day(
    trade_date: str,
    top_k: int,
    min_mv: float,
    max_mv: float,
    min_price: float,
    min_atr_pct: float,
    min_vol_ratio: float,
    max_bias_pct: float,
    max_gap_pct: float,
    buy_slippage_pct: float,
    sell_slippage_pct: float,
    commission_pct: float,
    sell_tax_pct: float,
    min_total_score: float,
    scarce_total_score: float,
    breadth_threshold: int,
    enable_market_filter: bool,
    protection_trigger_pct: float,
    protection_buffer_pct: float,
    use_sina: bool = False,
):
    query_date = trade_date
    daily = safe_get("daily", trade_date=query_date)
    # 盘中daily/daily_basic通常尚未发布：用最近完整交易日构建基础股票池，
    # 个股技术指标仍由新浪实时行补到trade_date。
    if use_sina and daily.empty:
        for offset in range(1, 10):
            candidate_date = (
                datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=offset)
            ).strftime("%Y%m%d")
            daily = safe_get("daily", trade_date=candidate_date)
            if not daily.empty:
                query_date = candidate_date
                break
    daily_basic = safe_get(
        "daily_basic",
        trade_date=query_date,
        fields="ts_code,trade_date,turnover_rate,turnover_rate_f,circ_mv,total_mv",
    )
    if daily.empty or daily_basic.empty:
        return (
            pd.DataFrame(),
            {"Trade_Date": trade_date, "Error": "daily/daily_basic缺失"},
            pd.DataFrame(),
        )

    stock_basic = GLOBAL_STOCK_BASIC.copy()
    df = daily.merge(stock_basic, on="ts_code", how="inner")
    df = df.merge(daily_basic, on="ts_code", how="left", suffixes=("", "_basic"))
    df["circ_mv_billion"] = pd.to_numeric(df["circ_mv"], errors="coerce") / 10000.0
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    regime = market_regime_on_date(trade_date, enabled=enable_market_filter)
    funnel = {
        "Trade_Date": trade_date,
        "Market_Rows": len(df),
        "Market_Regime": regime["Market_Regime"],
        "Market_Weak_Block": int(regime["Market_Weak_Block"]),
        "CYB_Close": regime["CYB_Close"],
        "CYB_MA20": regime["CYB_MA20"],
        "STAR50_Close": regime["STAR50_Close"],
        "STAR50_MA20": regime["STAR50_MA20"],
    }
    df = df[df.apply(lambda r: stock_active_on_date(r, trade_date), axis=1)]
    df = df[~df["name"].str.contains("ST|退", na=False)]
    funnel["After_ST_List_Filter"] = len(df)
    df = df[df["close"] >= min_price]
    funnel["After_Price"] = len(df)
    df = df[df["circ_mv_billion"].between(min_mv, max_mv, inclusive="both")]
    funnel["After_MarketCap"] = len(df)

    tech_rows = []
    for row in df.itertuples(index=False):
        membership = get_tech_membership(row.ts_code, trade_date)
        if membership is not None:
            tech_rows.append((row, membership))
    funnel["After_Tech_Pool"] = len(tech_rows)

    for key in [
        "Indicator_Valid",
        "Pass_Weekly",
        "Pass_Trend",
        "Pass_Positive_Day",
        "Pass_ATR",
        "Pass_Above_MA20",
        "Pass_Not_Overextended",
        "Pass_Volume",
        "Core_Signal",
        "Base_Score_Pass",
        "Dynamic_Score_Pass",
        "After_Market_Filter",
        "Execution_Checked",
        "Final_Signal",
        "Tradable_Signal",
        "Selected_TopK",
    ]:
        funnel[key] = 0

    records = []
    for row, membership in tech_rows:
        indicators = compute_trend_indicators(
            row.ts_code,
            trade_date,
            min_atr_pct=min_atr_pct,
            min_vol_ratio=min_vol_ratio,
            max_bias_pct=max_bias_pct,
            use_sina=use_sina,
        )
        if not indicators.get("valid"):
            continue
        funnel["Indicator_Valid"] += 1
        funnel["Pass_Weekly"] += int(indicators["weekly_safe"])
        funnel["Pass_Trend"] += int(indicators["trend_up"])
        funnel["Pass_Positive_Day"] += int(indicators["positive_day"])
        funnel["Pass_ATR"] += int(indicators["atr_ok"])
        funnel["Pass_Above_MA20"] += int(indicators["above_ma20"])
        funnel["Pass_Not_Overextended"] += int(indicators["not_overextended"])
        funnel["Pass_Volume"] += int(indicators["volume_ok"])
        if not indicators["is_buy_signal"]:
            continue
        funnel["Core_Signal"] += 1

        record = {
            "Trade_Date": trade_date,
            "ts_code": row.ts_code,
            "name": row.name,
            "market": row.market,
            "SW_L1": membership["l1"],
            "SW_L2": membership["l2"],
            "SW_L3": membership["l3"],
            "Signal_Close": round(indicators["last_close"], 3),
            "circ_mv": round(row.circ_mv_billion, 2),
            "turnover_rate": round(float(row.turnover_rate), 2) if pd.notna(row.turnover_rate) else np.nan,
            "Total_Score": round(indicators["total_score"], 2),
            "Breakout_S": round(indicators["score_breakout"], 2),
            "Volume_S": round(indicators["score_volume"], 2),
            "Pullback_S": round(indicators["score_pullback"], 2),
            "Candle_S": round(indicators["score_candle"], 2),
            "MACD_S": round(indicators["score_macd"], 2),
            "Trend_S": round(indicators["score_trend"], 2),
            "RS20_S": round(indicators["score_rs"], 2),
            "Chase_Penalty": round(indicators["chase_penalty"], 2),
            "Day_Gain_pct": round(indicators["day_gain_pct"], 2),
            "MA20_Bias_pct": round(indicators["bias_pct"], 2),
            "ATR_pct": round(indicators["atr_pct"], 2),
            "Vol_Ratio": round(indicators["vol_ratio"], 2),
            "RS20_pct": round(indicators["rs20_pct"], 2),
            "Pulled_Back": indicators["pulled_back"],
            "MACD_Improving": indicators["macd_improving"],
            "Solid_Candle": indicators["solid_candle"],
            # 下列字段用于通过门槛后的成交回测，候选导出前会移除前导下划线。
            "_Signal_Low": indicators["signal_low"],
            "_Signal_MA20": indicators["ma20"],
            "_ATR14": indicators["atr14"],
        }
        records.append(record)

    if not records:
        funnel["Day_Base_Signal_Count"] = 0
        funnel["Required_Score"] = scarce_total_score
        return pd.DataFrame(), funnel, pd.DataFrame()

    all_candidates = pd.DataFrame(records)
    # 广度只能使用D0收盘已经知道的信息：80分以上核心信号数。
    base_count = int((all_candidates["Total_Score"] >= min_total_score).sum())
    required_score = min_total_score if base_count >= breadth_threshold else scarce_total_score
    all_candidates["Day_Base_Signal_Count"] = base_count
    all_candidates["Required_Score"] = required_score
    all_candidates["Score_Pass"] = all_candidates["Total_Score"] >= required_score
    all_candidates["Market_Regime"] = regime["Market_Regime"]
    all_candidates["Market_Weak_Block"] = bool(regime["Market_Weak_Block"])
    all_candidates["CYB_Close"] = regime["CYB_Close"]
    all_candidates["CYB_MA20"] = regime["CYB_MA20"]
    all_candidates["CYB_MA20_Falling"] = regime["CYB_MA20_Falling"]
    all_candidates["STAR50_Close"] = regime["STAR50_Close"]
    all_candidates["STAR50_MA20"] = regime["STAR50_MA20"]
    all_candidates["STAR50_MA20_Falling"] = regime["STAR50_MA20_Falling"]
    # 固定候选导出结构，避免不同交易日追加CSV时列数不一致。
    for key in future_result_template(8):
        if key not in all_candidates.columns:
            if key.startswith("Eligible_W") or key.startswith("Held_W") or key == "Tradable":
                all_candidates[key] = pd.Series(pd.NA, index=all_candidates.index, dtype="boolean")
            elif key == "Exit_Reason":
                all_candidates[key] = pd.Series(None, index=all_candidates.index, dtype="object")
            else:
                all_candidates[key] = np.nan
    all_candidates["Execution_Checked"] = False
    all_candidates["Selection_Status"] = np.where(
        all_candidates["Score_Pass"], "通过动态分数门槛", "未达动态分数门槛"
    )

    funnel["Day_Base_Signal_Count"] = base_count
    funnel["Required_Score"] = required_score
    funnel["Base_Score_Pass"] = base_count
    funnel["Dynamic_Score_Pass"] = int(all_candidates["Score_Pass"].sum())
    funnel["Final_Signal"] = funnel["Dynamic_Score_Pass"]

    if regime["Market_Weak_Block"]:
        all_candidates.loc[all_candidates["Score_Pass"], "Selection_Status"] = "双指数弱势-暂停开仓"
    else:
        eligible_indices = all_candidates.index[all_candidates["Score_Pass"]].tolist()
        funnel["After_Market_Filter"] = len(eligible_indices)
        for idx in eligible_indices:
            candidate = all_candidates.loc[idx]
            future = get_medium_term_future(
                candidate["ts_code"],
                candidate["market"],
                trade_date,
                float(candidate["Signal_Close"]),
                float(candidate["_Signal_Low"]),
                float(candidate["_Signal_MA20"]),
                float(candidate["_ATR14"]),
                hold_weeks=8,
                max_gap_pct=max_gap_pct,
                buy_slippage_pct=buy_slippage_pct,
                sell_slippage_pct=sell_slippage_pct,
                commission_pct=commission_pct,
                sell_tax_pct=sell_tax_pct,
                protection_trigger_pct=protection_trigger_pct,
                protection_buffer_pct=protection_buffer_pct,
                use_sina=use_sina,
            )
            for key, value in future.items():
                all_candidates.loc[idx, key] = value
            all_candidates.loc[idx, "Execution_Checked"] = True
            if future["Tradable"]:
                all_candidates.loc[idx, "Selection_Status"] = "可成交-等待TopK排序"
            else:
                all_candidates.loc[idx, "Selection_Status"] = str(future["Exit_Reason"])

    checked_mask = all_candidates["Execution_Checked"].fillna(False).astype(bool)
    tradable_mask = checked_mask & all_candidates["Tradable"].eq(True).fillna(False)
    funnel["Execution_Checked"] = int(checked_mask.sum())
    funnel["Tradable_Signal"] = int(tradable_mask.sum())

    tradable = all_candidates[tradable_mask].copy()
    final_indices = (
        tradable.sort_values(["Total_Score", "ATR_pct"], ascending=[False, False])
        .head(top_k)
        .index
        .tolist()
    )
    all_candidates.loc[tradable.index, "Selection_Status"] = "未进入TopK"
    all_candidates.loc[final_indices, "Selection_Status"] = "入选TopK"
    all_candidates["Selected"] = all_candidates.index.isin(final_indices)
    funnel["Selected_TopK"] = len(final_indices)

    internal_cols = ["_Signal_Low", "_Signal_MA20", "_ATR14"]
    candidate_export = all_candidates.drop(columns=internal_cols, errors="ignore").copy()
    if not final_indices:
        return pd.DataFrame(), funnel, candidate_export
    final = candidate_export.loc[final_indices].copy()
    final = final.sort_values(["Total_Score", "ATR_pct"], ascending=[False, False])
    final.insert(1, "Rank", range(1, len(final) + 1))
    return final, funnel, candidate_export


# -----------------------------------------------------------------------------
# 报表
# -----------------------------------------------------------------------------
def show_weekly_report(all_results: pd.DataFrame) -> None:
    st.subheader("🗓️ 全样本周度收益与真实持仓生存")
    row1 = st.columns(4)
    row2 = st.columns(4)
    for week in range(1, 9):
        ret_col = f"Return_W{week} (%)"
        eligible_col = f"Eligible_W{week}"
        held_col = f"Held_W{week}"
        target = row1[week - 1] if week <= 4 else row2[week - 5]
        eligible = all_results[all_results[eligible_col].fillna(False).astype(bool)].copy()
        with target:
            if eligible.empty:
                st.metric(f"W{week}", "尚无成熟样本")
                continue
            returns = pd.to_numeric(eligible[ret_col], errors="coerce").dropna()
            held = eligible[held_col].fillna(False).astype(bool).sum()
            survival = held / len(eligible) * 100.0
            avg_return = returns.mean() if not returns.empty else np.nan
            win_rate = (returns > 0).mean() * 100.0 if not returns.empty else np.nan
            st.metric(
                f"W{week} 生存{held}/{len(eligible)} ({survival:.1f}%)",
                f"{avg_return:.2f}% / 胜率{win_rate:.1f}%",
            )


def show_exit_report(all_results: pd.DataFrame) -> None:
    st.subheader("🚪 退出原因分布")
    summary = (
        all_results.groupby("Exit_Reason", dropna=False)
        .agg(
            数量=("ts_code", "size"),
            平均持仓天数=("Holding_Days", "mean"),
            平均MFE=("MFE_pct (%)", "mean"),
            平均MAE=("MAE_pct (%)", "mean"),
        )
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def style_exit(value):
    if not isinstance(value, str):
        return ""
    if "止损" in value:
        return "color: white; background-color: darkred"
    if "剔除" in value:
        return "color: white; background-color: gray"
    if "保本" in value or "成本保护" in value:
        return "color: orange"
    if "移动止盈" in value:
        return "color: green"
    if "周期结束" in value:
        return "color: blue"
    return ""


# -----------------------------------------------------------------------------
# UI参数
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header(f"{VERSION} 回测参数")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析交易日数", min_value=1, value=100, step=10)
    TOP_BACKTEST = st.number_input("每日优选 TopK", min_value=1, max_value=20, value=5, step=1)

    st.subheader("股票池")
    MIN_PRICE = st.number_input("最低股价(元)", min_value=1.0, value=20.0, step=1.0)
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小流通市值(亿)", min_value=1.0, value=200.0, step=10.0)
    MAX_MV = col2.number_input("最大流通市值(亿)", min_value=10.0, value=1000.0, step=50.0)
    ENABLE_THS = st.checkbox("实时雷达启用THS概念补充(需6000积分)", value=False)
    THS_KEYWORDS_TEXT = st.text_area("THS科技概念关键词", value=DEFAULT_THS_KEYWORDS, height=100)

    st.subheader("信号与动态评分门槛")
    MIN_ATR_PCT = st.number_input("ATR14/股价最低(%)", min_value=0.0, value=2.5, step=0.1)
    MIN_VOL_RATIO = st.number_input("最低量比", min_value=0.5, value=1.10, step=0.05)
    MAX_BIAS_PCT = st.number_input("相对MA20最大偏离(%)", min_value=1.0, value=8.0, step=0.5)
    MAX_GAP_PCT = st.number_input("T+1最大允许高开(%)", min_value=0.0, value=8.0, step=0.5)
    MIN_TOTAL_SCORE = st.number_input("常规最低总分", min_value=0.0, value=80.0, step=1.0)
    SCARCE_TOTAL_SCORE = st.number_input("信号稀少时最低总分", min_value=0.0, value=82.0, step=1.0)
    BREADTH_THRESHOLD = st.number_input("常规门槛所需信号数", min_value=1, value=4, step=1)

    st.subheader("市场环境")
    ENABLE_MARKET_FILTER = st.checkbox(
        "双指数同时跌破下降MA20时暂停开仓",
        value=True,
        help="同时检查创业板指399006.SZ和科创50 000688.SH；数据不足时默认允许开仓。",
    )

    st.subheader("成交成本与保护止损")
    BUY_SLIPPAGE = st.number_input("买入滑点(%)", min_value=0.0, value=0.20, step=0.05)
    SELL_SLIPPAGE = st.number_input("卖出滑点(%)", min_value=0.0, value=0.20, step=0.05)
    COMMISSION = st.number_input("单边佣金(%)", min_value=0.0, value=0.03, step=0.01)
    SELL_TAX = st.number_input("卖出税费(%)", min_value=0.0, value=0.05, step=0.01)
    PROTECTION_TRIGGER = st.number_input("保护止损触发浮盈(%)", min_value=1.0, value=10.0, step=1.0)
    PROTECTION_BUFFER = st.number_input("保护止损原始价高于成本(%)", min_value=0.0, value=0.30, step=0.10)

    RESUME_CHECKPOINT = st.checkbox("开启参数隔离的断点续传", value=True)
    USE_CACHE = st.checkbox("使用并增量更新行情缓存", value=True)
    if st.button("清除V39.7/V39.8共享行情缓存"):
        if os.path.exists(CACHE_FILE_NAME):
            os.remove(CACHE_FILE_NAME)
            st.success("缓存已清除")


TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN:
    st.info("请输入Tushare Token后开始运行。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()


if st.button(f"🚀 启动 {VERSION} 科技增强回测"):
    API_ERRORS.clear()
    SINA_STATUS = {"success": 0, "fail": 0}
    is_realtime = int(BACKTEST_DAYS) == 1
    try:
        if SCARCE_TOTAL_SCORE < MIN_TOTAL_SCORE:
            raise ValueError("信号稀少时最低总分不能低于常规最低总分")
        GLOBAL_STOCK_BASIC = load_stock_basic()
        memberships = load_sw_tech_memberships()
        GLOBAL_TECH_PERIODS = build_tech_period_index(memberships)

        if ENABLE_THS:
            if not is_realtime:
                st.warning("THS当前概念成分不用于历史回测，以免产生未来信息；本次只使用申万历史科技池。")
                GLOBAL_THS_TECH_CODES = set()
            else:
                keywords = tuple(parse_keywords(THS_KEYWORDS_TEXT))
                GLOBAL_THS_TECH_CODES, selected_concepts = load_ths_current_tech_pool(keywords)
                st.info(
                    f"THS概念补充命中 {len(GLOBAL_THS_TECH_CODES)} 只股票、"
                    f"{len(selected_concepts)} 个概念板块。"
                )
        else:
            GLOBAL_THS_TECH_CODES = set()

        end_str = backtest_date_end.strftime("%Y%m%d")
        trade_days = get_trade_days(end_str, int(BACKTEST_DAYS))
        if not trade_days:
            raise RuntimeError("未取得回测交易日")
        earliest_signal = min(trade_days)
        latest_signal = max(trade_days)
        fetch_start = (datetime.strptime(earliest_signal, "%Y%m%d") - timedelta(days=460)).strftime("%Y%m%d")
        theoretical_end = datetime.strptime(latest_signal, "%Y%m%d") + timedelta(days=110)
        fetch_end = min(theoretical_end, datetime.now()).strftime("%Y%m%d")
        required_dates = get_open_dates(fetch_start, fetch_end)
        load_market_data(required_dates, use_cache=USE_CACHE)
        load_benchmark(fetch_start, fetch_end)
        load_market_regime_indices(fetch_start, fetch_end)

        config = {
            "version": VERSION,
            "end": end_str,
            "days": int(BACKTEST_DAYS),
            "topk": int(TOP_BACKTEST),
            "min_price": MIN_PRICE,
            "min_mv": MIN_MV,
            "max_mv": MAX_MV,
            "min_atr": MIN_ATR_PCT,
            "min_vol": MIN_VOL_RATIO,
            "max_bias": MAX_BIAS_PCT,
            "max_gap": MAX_GAP_PCT,
            "min_score": MIN_TOTAL_SCORE,
            "scarce_score": SCARCE_TOTAL_SCORE,
            "breadth_threshold": int(BREADTH_THRESHOLD),
            "market_filter": bool(ENABLE_MARKET_FILTER),
            "buy_slip": BUY_SLIPPAGE,
            "sell_slip": SELL_SLIPPAGE,
            "commission": COMMISSION,
            "sell_tax": SELL_TAX,
            "protection_trigger": PROTECTION_TRIGGER,
            "protection_buffer": PROTECTION_BUFFER,
        }
        config_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        result_file = f"backtest_{VERSION.replace('.', '_')}_{config_hash}.csv"
        funnel_file = f"funnel_{VERSION.replace('.', '_')}_{config_hash}.csv"
        candidate_file = f"candidates_{VERSION.replace('.', '_')}_{config_hash}.csv"
        state_file = f"state_{VERSION.replace('.', '_')}_{config_hash}.json"

        processed = set()
        result_frames = []
        candidate_frames = []
        funnel_rows = []
        if RESUME_CHECKPOINT and os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as file:
                    state = json.load(file)
                processed = set(state.get("processed_dates", []))
                if os.path.exists(result_file):
                    result_frames.append(pd.read_csv(result_file, encoding="utf-8-sig"))
                if os.path.exists(funnel_file):
                    funnel_rows.extend(pd.read_csv(funnel_file, encoding="utf-8-sig").to_dict("records"))
                if os.path.exists(candidate_file):
                    candidate_frames.append(pd.read_csv(candidate_file, encoding="utf-8-sig"))
            except Exception as exc:
                record_api_error(f"断点读取失败，将从头运行: {exc}")
                processed = set()

        dates_to_run = [d for d in trade_days if d not in processed]
        progress = st.progress(0, text="开始逐日计算科技信号...")
        for i, trade_date in enumerate(dates_to_run):
            use_sina = is_realtime and trade_date == datetime.now().strftime("%Y%m%d")
            day_result, funnel, day_candidates = run_backtest_for_day(
                trade_date=trade_date,
                top_k=int(TOP_BACKTEST),
                min_mv=float(MIN_MV),
                max_mv=float(MAX_MV),
                min_price=float(MIN_PRICE),
                min_atr_pct=float(MIN_ATR_PCT),
                min_vol_ratio=float(MIN_VOL_RATIO),
                max_bias_pct=float(MAX_BIAS_PCT),
                max_gap_pct=float(MAX_GAP_PCT),
                buy_slippage_pct=float(BUY_SLIPPAGE),
                sell_slippage_pct=float(SELL_SLIPPAGE),
                commission_pct=float(COMMISSION),
                sell_tax_pct=float(SELL_TAX),
                min_total_score=float(MIN_TOTAL_SCORE),
                scarce_total_score=float(SCARCE_TOTAL_SCORE),
                breadth_threshold=int(BREADTH_THRESHOLD),
                enable_market_filter=bool(ENABLE_MARKET_FILTER),
                protection_trigger_pct=float(PROTECTION_TRIGGER),
                protection_buffer_pct=float(PROTECTION_BUFFER),
                use_sina=use_sina,
            )
            if not day_result.empty:
                result_frames.append(day_result)
                day_result.to_csv(
                    result_file,
                    mode="a",
                    index=False,
                    header=not os.path.exists(result_file),
                    encoding="utf-8-sig",
                )
            if not day_candidates.empty:
                candidate_frames.append(day_candidates)
                day_candidates.to_csv(
                    candidate_file,
                    mode="a",
                    index=False,
                    header=not os.path.exists(candidate_file),
                    encoding="utf-8-sig",
                )
            funnel_rows.append(funnel)
            pd.DataFrame([funnel]).to_csv(
                funnel_file,
                mode="a",
                index=False,
                header=not os.path.exists(funnel_file),
                encoding="utf-8-sig",
            )
            processed.add(trade_date)
            with open(state_file, "w", encoding="utf-8") as file:
                json.dump(
                    {"config": config, "processed_dates": sorted(processed)},
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            progress.progress((i + 1) / max(len(dates_to_run), 1), text=f"分析 {trade_date}")
        progress.empty()

        st.header(f"📊 {VERSION} 回测结果")
        funnel_df = pd.DataFrame(funnel_rows).drop_duplicates("Trade_Date", keep="last")
        all_candidates_export = pd.DataFrame()
        if candidate_frames:
            all_candidates_export = pd.concat(candidate_frames, ignore_index=True)
            all_candidates_export = all_candidates_export.drop_duplicates(
                ["Trade_Date", "ts_code"], keep="last"
            )

        st.subheader("🌦️ 市场环境与动态门槛")
        if not funnel_df.empty:
            funnel_defaults = {
                "Market_Weak_Block": 0,
                "Market_Regime": "数据缺失",
                "Day_Base_Signal_Count": 0,
                "Required_Score": np.nan,
                "Dynamic_Score_Pass": 0,
                "Tradable_Signal": 0,
                "Selected_TopK": 0,
                "CYB_Close": np.nan,
                "CYB_MA20": np.nan,
                "STAR50_Close": np.nan,
                "STAR50_MA20": np.nan,
            }
            for column, default_value in funnel_defaults.items():
                if column not in funnel_df.columns:
                    funnel_df[column] = default_value
            blocked_days = int(pd.to_numeric(funnel_df["Market_Weak_Block"], errors="coerce").fillna(0).sum())
            score_pass_total = int(pd.to_numeric(funnel_df["Dynamic_Score_Pass"], errors="coerce").fillna(0).sum())
            selected_total = int(pd.to_numeric(funnel_df["Selected_TopK"], errors="coerce").fillna(0).sum())
            env_cols = st.columns(3)
            env_cols[0].metric("双指数弱势暂停日", blocked_days)
            env_cols[1].metric("通过动态分数门槛", score_pass_total)
            env_cols[2].metric("最终入选", selected_total)
            st.dataframe(
                funnel_df[
                    [
                        "Trade_Date", "Market_Regime", "Day_Base_Signal_Count", "Required_Score",
                        "Dynamic_Score_Pass", "Tradable_Signal", "Selected_TopK",
                        "CYB_Close", "CYB_MA20", "STAR50_Close", "STAR50_MA20",
                    ]
                ].sort_values("Trade_Date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        if result_frames:
            all_results = pd.concat(result_frames, ignore_index=True)
            all_results = all_results.drop_duplicates(["Trade_Date", "ts_code"], keep="last")
            for week in range(1, 9):
                for prefix in ["Eligible_W", "Held_W"]:
                    col = f"{prefix}{week}"
                    if all_results[col].dtype == object:
                        all_results[col] = all_results[col].astype(str).str.lower().map({"true": True, "false": False})
            show_weekly_report(all_results)
            show_exit_report(all_results)

            st.subheader("🔬 筛选漏斗（逐日合计）")
            numeric_cols = [c for c in funnel_df.columns if c not in ["Trade_Date", "Error"]]
            numeric_funnel = funnel_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            numeric_funnel = numeric_funnel.dropna(axis=1, how="all")
            totals = numeric_funnel.sum().to_frame("累计/求和")
            st.dataframe(totals, use_container_width=True)

            st.subheader("📋 优选记录")
            display = all_results.sort_values(["Trade_Date", "Rank"], ascending=[False, True])
            try:
                st.dataframe(
                    display.style.map(style_exit, subset=["Exit_Reason"]),
                    use_container_width=True,
                    height=650,
                )
            except AttributeError:
                st.dataframe(
                    display.style.applymap(style_exit, subset=["Exit_Reason"]),
                    use_container_width=True,
                    height=650,
                )

            export_csv = all_results.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 下载V39.8入选股票完整轨迹CSV",
                export_csv,
                f"export_v39_8_{config_hash}.csv",
                "text/csv",
            )
        else:
            st.warning("本次没有产生可交易信号。请先查看筛选漏斗确定主要淘汰环节。")

        if not all_candidates_export.empty:
            candidate_csv = all_candidates_export.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 下载V39.8全部核心候选及未入选原因CSV",
                candidate_csv,
                f"candidates_v39_8_{config_hash}.csv",
                "text/csv",
            )
        if not funnel_df.empty:
            funnel_csv = funnel_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 下载V39.8逐日筛选漏斗CSV",
                funnel_csv,
                f"funnel_v39_8_{config_hash}.csv",
                "text/csv",
            )

        if API_ERRORS:
            with st.expander(f"⚠️ API/数据警告（{len(API_ERRORS)}条）"):
                st.code("\n".join(API_ERRORS[:100]))
        if is_realtime:
            if SINA_STATUS["success"]:
                st.success(f"新浪实时探针成功 {SINA_STATUS['success']} 次；盘中信号仅为临时信号。")
            elif SINA_STATUS["fail"]:
                st.warning(f"新浪实时探针失败 {SINA_STATUS['fail']} 次。")

    except Exception as exc:
        st.error(f"运行终止：{exc}")
        if API_ERRORS:
            st.code("\n".join(API_ERRORS[-30:]))
