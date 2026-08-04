# -*- coding: utf-8 -*-
"""
选股王 V40.2 · 周线质量优先版
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
12. 环境：取消创业板指/科创50的一刀切暂停开仓，避免误杀弱市中的强势股。
13. 保本：浮盈首次达到10%后，从下一交易日启用成本上方0.3%的保护止损。
14. 审计：另行导出全部核心候选及未入选原因，便于核查质量门槛。
15. 固化：股价不低于20元、流通市值200~1000亿元、关闭大盘硬过滤。
16. 广度：T+1成交检查后可交易股票少于2只时，全部仅观察而不开仓。
17. 组合：30万元初始资金、最多3只持仓、单只约10万元，输出组合曲线与真实空仓率。
18. 断点：改用主键去重和原子覆盖保存，防止续跑时重复记录。
19. 周线波浪：仅用已完成周K计算周线MACD，标记绿柱缩短、首次翻红、
    红柱扩张/再加速/衰减和绿柱弱势，本版只诊断、不改变入选与买卖。
20. 假反弹：记录T+1收盘是否延续，并将“10个交易日内结构止损且最大浮盈
    不足5%”标记为早期假反弹，用于检验周线波浪假设。
21. 双周线：W_字段只用完成周K；P_W_字段使用截至信号日收盘的当周临时K线，
    两者并列导出，既防止把未来周五数据泄漏到回测，又能观察“本周正在翻红”。
22. 替代入场：主组合仍保持T+1开盘买入；另行诊断T+1收盘延续后T+2开盘买入，
    以及50% T+1 + 50% T+2的近似分批收益，不影响主组合权益。
23. 质量优先：固定82分底线，默认要求信号日涨幅≥3%、高于日MA20≥3%、
    信号日临时周线偏离MA20≤25%，优先G2绿柱缩短和R1首周翻红。
24. 热点保留：不限制每日新开仓数，不限制同一申万二级行业持仓；
    仍允许热点板块联动时同日买入多只，总持仓上限3只不变。
25. 提前退出：增加“T+1未延续时T+2开盘退出”独立开关，默认关闭；
    便于先验证入选质量，再单独比较退出策略，避免混淆W1生存率。

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
import traceback
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import tushare as ts

warnings.filterwarnings("ignore")

VERSION = "V40.2"
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

# 经多轮A/B回测确认后固化的基准参数。
FIXED_MIN_PRICE = 20.0
FIXED_MIN_MV = 200.0
FIXED_MAX_MV = 1000.0
FIXED_MARKET_FILTER = False
MIN_TRADABLE_SIGNALS = 2

# V40.2单股质量默认值；侧边栏可调，方便做严格/宽松对照。
DEFAULT_MIN_DAY_GAIN_PCT = 3.0
DEFAULT_MIN_DAILY_BIAS_PCT = 3.0
DEFAULT_MAX_WEEKLY_BIAS_PCT = 25.0
DEFAULT_FIXED_SCORE_FLOOR = 82.0

# 只改变同日候选的优先顺序，不把任何周线阶段一刀切删除。
STAGE_PRIORITY = {
    "G2_绿柱缩短": (0, "A1_G2绿柱缩短"),
    "R1_首周翻红": (1, "A2_R1首周翻红"),
    "G3_绿柱扩大/弱势": (2, "B_中性阶段"),
    "R3_红柱再加速": (2, "B_中性阶段"),
    "R4_红柱扩张": (2, "B_中性阶段"),
    "G1_绿柱连续缩短": (3, "C_降级阶段"),
    "R5_红柱衰减": (3, "C_降级阶段"),
}

# 真实组合约束。
INITIAL_CAPITAL = 300_000.0
MAX_PORTFOLIO_POSITIONS = 3
POSITION_BUDGET = INITIAL_CAPITAL / MAX_PORTFOLIO_POSITIONS
LOT_SIZE = 100


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
st.set_page_config(page_title=f"选股王 {VERSION} 周线质量优先版", layout="wide")
st.title(f"选股王 {VERSION}：周线阶段优先 + 弱反弹过滤")


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


def canonicalize_records(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """统一断点文件的主键类型并去重。

    Streamlit表格下载可能带入 Unnamed 索引列；CSV重读还可能把
    Trade_Date 变成数字。先规范化再去重，避免“看起来相同、实际类型不同”。
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    clean = df.copy()
    clean = clean.drop(columns=[c for c in clean.columns if str(c).startswith("Unnamed:")], errors="ignore")
    if "Trade_Date" in clean.columns:
        clean["Trade_Date"] = clean["Trade_Date"].map(normalize_date)
        clean = clean[clean["Trade_Date"] != ""]
    if "ts_code" in clean.columns:
        clean["ts_code"] = clean["ts_code"].astype(str).str.strip()
        clean = clean[clean["ts_code"] != ""]
    existing_keys = [key for key in keys if key in clean.columns]
    if existing_keys:
        clean = clean.drop_duplicates(existing_keys, keep="last")
    return clean.reset_index(drop=True)


def merge_records(current: pd.DataFrame, new_rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    frames = [frame for frame in [current, new_rows] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return canonicalize_records(pd.concat(frames, ignore_index=True), keys)


def replace_trade_date_records(
    current: pd.DataFrame,
    new_rows: pd.DataFrame,
    trade_date: str,
    keys: list[str],
) -> pd.DataFrame:
    """断点重跑某日时先删除该日旧快照，再写入新快照。"""
    base = canonicalize_records(current, keys)
    if not base.empty and "Trade_Date" in base.columns:
        base = base[base["Trade_Date"] != normalize_date(trade_date)].copy()
    return merge_records(base, new_rows, keys)


def read_checkpoint_csv(path: str, keys: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    key_dtypes = {key: str for key in keys}
    try:
        loaded = pd.read_csv(path, encoding="utf-8-sig", dtype=key_dtypes)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=keys)
    return canonicalize_records(loaded, keys)


def atomic_write_csv(df: pd.DataFrame, path: str, keys: list[str]) -> pd.DataFrame:
    """先写临时文件再原子替换；每次都写入已去重的完整快照。"""
    clean = canonicalize_records(df, keys)
    if len(clean.columns) == 0:
        clean = pd.DataFrame(columns=keys)
    temp_path = f"{path}.tmp"
    clean.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(temp_path, path)
    return clean


def atomic_write_json(payload: dict, path: str) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


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
    # bias为相对MA20偏离率；评分偏好有力但不过度的突破。
    if bias < 0.005:
        return 0.0
    if bias <= 0.05:
        return min(20.0, 8.0 + (bias - 0.005) / 0.045 * 12.0)
    return max(5.0, 20.0 - (bias - 0.05) / 0.03 * 15.0)


WEEKLY_STAGE_ORDER = {
    "R1_首周翻红": 1,
    "R2_第二根红柱扩张": 2,
    "R3_红柱再加速": 3,
    "R4_红柱扩张": 4,
    "G1_绿柱连续缩短": 5,
    "G2_绿柱缩短": 6,
    "R5_红柱衰减": 7,
    "G3_绿柱扩大/弱势": 8,
    "W0_周线数据不足": 9,
}


def classify_completed_weekly_macd(weekly: pd.DataFrame, signal_date: str) -> dict:
    """
    只用信号日当时已完成的周K进行诊断。

    resample("W-FRI")会在周一至周四生成一根以未来周五为标签的部分周K；
    这根K线盘中会反复变化，因此本函数把其排除，避免周线MACD“重绘”。
    交易逻辑仍使用V39.9原weekly_safe，本函数只产生诊断字段。
    """
    empty = {
        "weekly_completed_date": "",
        "weekly_macd_stage": "W0_周线数据不足",
        "weekly_macd_hist_pct": np.nan,
        "weekly_macd_hist_prev_pct": np.nan,
        "weekly_dif_pct": np.nan,
        "weekly_dea_pct": np.nan,
        "weekly_close": np.nan,
        "weekly_ma20": np.nan,
        "weekly_ma20_slope_pct": np.nan,
        "weekly_bias_pct": np.nan,
        "weekly_dif_rising": False,
        "weekly_dea_rising": False,
        "weekly_ma20_rising": False,
        "weekly_close_above_ma20": False,
        "weekly_trend_confirmed": False,
        "weekly_wave_candidate": False,
    }
    if weekly is None or weekly.empty or "dt" not in weekly.columns:
        return empty

    signal_ts = pd.Timestamp(datetime.strptime(normalize_date(signal_date), "%Y%m%d"))
    completed = weekly[pd.to_datetime(weekly["dt"]) <= signal_ts].copy()
    if len(completed) < 30:
        return empty

    completed["w_ema12"] = completed["close"].ewm(span=12, adjust=False).mean()
    completed["w_ema26"] = completed["close"].ewm(span=26, adjust=False).mean()
    completed["w_dif"] = completed["w_ema12"] - completed["w_ema26"]
    completed["w_dea"] = completed["w_dif"].ewm(span=9, adjust=False).mean()
    completed["w_macd"] = (completed["w_dif"] - completed["w_dea"]) * 2.0
    if "w_ma20" not in completed.columns:
        completed["w_ma20"] = completed["close"].rolling(20).mean()
    if len(completed) < 3 or pd.isna(completed.iloc[-1]["w_ma20"]):
        return empty

    curr, prev, prev2 = completed.iloc[-1], completed.iloc[-2], completed.iloc[-3]
    hist, hist_prev, hist_prev2 = float(curr["w_macd"]), float(prev["w_macd"]), float(prev2["w_macd"])
    dif_rising = float(curr["w_dif"]) > float(prev["w_dif"])
    dea_rising = float(curr["w_dea"]) >= float(prev["w_dea"])
    ma20_rising = float(curr["w_ma20"]) >= float(prev["w_ma20"])
    close_above_ma20 = float(curr["close"]) >= float(curr["w_ma20"])

    if hist > 0:
        if hist_prev <= 0:
            stage = "R1_首周翻红"
        elif hist_prev > 0 and hist_prev2 <= 0 and hist >= hist_prev:
            stage = "R2_第二根红柱扩张"
        elif hist > hist_prev and hist_prev < hist_prev2:
            stage = "R3_红柱再加速"
        elif hist >= hist_prev:
            stage = "R4_红柱扩张"
        else:
            stage = "R5_红柱衰减"
    else:
        if hist > hist_prev and hist_prev > hist_prev2 and dif_rising:
            stage = "G1_绿柱连续缩短"
        elif hist > hist_prev and dif_rising:
            stage = "G2_绿柱缩短"
        else:
            stage = "G3_绿柱扩大/弱势"

    close = float(curr["close"])
    ma20 = float(curr["w_ma20"])
    prev_ma20 = float(prev["w_ma20"])
    trend_confirmed = bool(dif_rising and ma20_rising and close_above_ma20)
    wave_candidate = bool(
        stage in {
            "R1_首周翻红", "R2_第二根红柱扩张", "R3_红柱再加速",
            "R4_红柱扩张", "G1_绿柱连续缩短",
        }
        and dif_rising
    )
    scale = close if close > 0 else np.nan
    return {
        "weekly_completed_date": pd.Timestamp(curr["dt"]).strftime("%Y%m%d"),
        "weekly_macd_stage": stage,
        "weekly_macd_hist_pct": hist / scale * 100.0,
        "weekly_macd_hist_prev_pct": hist_prev / float(prev["close"]) * 100.0,
        "weekly_dif_pct": float(curr["w_dif"]) / scale * 100.0,
        "weekly_dea_pct": float(curr["w_dea"]) / scale * 100.0,
        "weekly_close": close,
        "weekly_ma20": ma20,
        "weekly_ma20_slope_pct": (ma20 / prev_ma20 - 1.0) * 100.0 if prev_ma20 > 0 else np.nan,
        "weekly_bias_pct": (close / ma20 - 1.0) * 100.0 if ma20 > 0 else np.nan,
        "weekly_dif_rising": bool(dif_rising),
        "weekly_dea_rising": bool(dea_rising),
        "weekly_ma20_rising": bool(ma20_rising),
        "weekly_close_above_ma20": bool(close_above_ma20),
        "weekly_trend_confirmed": trend_confirmed,
        "weekly_wave_candidate": wave_candidate,
    }


def classify_provisional_weekly_macd(weekly: pd.DataFrame, signal_date: str) -> dict:
    """
    使用截至D0收盘已知的当周数据。周一至周四时，W-FRI标签会晚于D0，
    因此把最后一根部分周K的标签临时改为D0再调用同一分类器。
    这不使用D0之后的价格，但临时周柱可能在本周剩余交易日内改变。
    """
    provisional = weekly.copy()
    if provisional.empty or "dt" not in provisional.columns:
        base = classify_completed_weekly_macd(provisional, signal_date)
    else:
        signal_ts = pd.Timestamp(datetime.strptime(normalize_date(signal_date), "%Y%m%d"))
        last_idx = provisional.index[-1]
        if pd.Timestamp(provisional.loc[last_idx, "dt"]) > signal_ts:
            provisional.loc[last_idx, "dt"] = signal_ts
        base = classify_completed_weekly_macd(provisional, signal_date)

    return {
        "provisional_asof_date": normalize_date(signal_date),
        "provisional_macd_stage": base["weekly_macd_stage"],
        "provisional_macd_hist_pct": base["weekly_macd_hist_pct"],
        "provisional_macd_hist_prev_pct": base["weekly_macd_hist_prev_pct"],
        "provisional_dif_pct": base["weekly_dif_pct"],
        "provisional_dea_pct": base["weekly_dea_pct"],
        "provisional_close": base["weekly_close"],
        "provisional_ma20": base["weekly_ma20"],
        "provisional_ma20_slope_pct": base["weekly_ma20_slope_pct"],
        "provisional_bias_pct": base["weekly_bias_pct"],
        "provisional_dif_rising": base["weekly_dif_rising"],
        "provisional_dea_rising": base["weekly_dea_rising"],
        "provisional_ma20_rising": base["weekly_ma20_rising"],
        "provisional_close_above_ma20": base["weekly_close_above_ma20"],
        "provisional_trend_confirmed": base["weekly_trend_confirmed"],
        "provisional_wave_candidate": base["weekly_wave_candidate"],
    }


def compute_trend_indicators(
    ts_code: str,
    end_date: str,
    min_atr_pct: float,
    min_vol_ratio: float,
    min_day_gain_pct: float,
    min_daily_bias_pct: float,
    max_bias_pct: float,
    max_weekly_bias_pct: float,
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
    weekly_wave = classify_completed_weekly_macd(weekly, end_date)
    provisional_wave = classify_provisional_weekly_macd(weekly, end_date)
    provisional_wave["provisional_changed_from_completed"] = bool(
        provisional_wave["provisional_macd_stage"] != weekly_wave["weekly_macd_stage"]
    )
    w_curr = weekly.iloc[-1]
    w_prev = weekly.iloc[-2]
    weekly_bias = (w_curr["close"] - w_curr["w_ma20"]) / w_curr["w_ma20"]
    w_range = w_prev["high"] - w_prev["low"]
    w_upper = w_prev["high"] - max(w_prev["open"], w_prev["close"])
    shadow_ratio = w_upper / w_range if w_range > 0 else 0.0

    weekly_safe = (
        weekly_bias <= max_weekly_bias_pct / 100.0
        and shadow_ratio < 0.60
    )
    trend_up = row["ma60"] > row["ma120"]
    pulled_back = any(recent_prev["close"] <= recent_prev["ma20"] * 1.01)
    bias = row["close"] / row["ma20"] - 1.0
    above_ma20 = bias >= min_daily_bias_pct / 100.0
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
    positive_day = (
        day_gain_pct >= min_day_gain_pct
        and row["close"] > row["open"]
    )
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
            **weekly_wave,
            **provisional_wave,
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
            "Entry_Date": "",
            "Exit_Date": "",
            "Buy_Price": np.nan,
            "Raw_Exit_Price": np.nan,
            "Net_Exit_Price": np.nan,
            "Realized_Return (%)": np.nan,
            "Gap_pct (%)": np.nan,
            "Stop_pct (%)": np.nan,
            "Protection_Trigger_Day": np.nan,
            "Protection_Stop_Price": np.nan,
            "MFE_pct (%)": np.nan,
            "MAE_pct (%)": np.nan,
            "T1_Close_Price": np.nan,
            "T1_Close_Return_pct": np.nan,
            "T1_Close_vs_Signal_pct": np.nan,
            "T1_Follow_Through": False,
            "Early_Structural_Stop_10D": False,
            "False_Rebound_10D": False,
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
    exit_non_follow_t2: bool = False,
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
    result["Entry_Date"] = normalize_date(next_row.name)
    result["Buy_Price"] = round(buy_price, 3)
    next_close = float(next_row["close"])
    result["T1_Close_Price"] = round(next_close, 3)
    result["T1_Close_Return_pct"] = round((next_close / buy_price - 1.0) * 100.0, 4)
    result["T1_Close_vs_Signal_pct"] = round(
        (next_close / signal_close - 1.0) * 100.0 if signal_close > 0 else np.nan,
        4,
    )
    result["T1_Follow_Through"] = bool(next_close >= signal_close and next_close >= raw_open)

    raw_structural_stop = min(signal_low, signal_ma20) - 0.5 * atr14
    raw_risk_pct = (buy_price - raw_structural_stop) / buy_price
    risk_pct = float(np.clip(raw_risk_pct, min_stop_pct, max_stop_pct))
    stop_price = buy_price * (1.0 - risk_pct)
    result["Stop_pct (%)"] = round(-risk_pct * 100.0, 2)
    protection_stop_price = buy_price * (1.0 + protection_buffer_pct / 100.0)
    protection_active_from_day = None

    def net_exit_price(raw_sell_price: float) -> float:
        net_sell = raw_sell_price * (1.0 - sell_slippage_pct / 100.0)
        net_sell *= 1.0 - (commission_pct + sell_tax_pct) / 100.0
        return net_sell

    def net_return(raw_sell_price: float) -> float:
        return (net_exit_price(raw_sell_price) / buy_price - 1.0) * 100.0

    def finalize_exit(
        raw_sell_price: float,
        week: int,
        day_count: int,
        reason: str,
        exit_date: str,
    ):
        net_sell = net_exit_price(raw_sell_price)
        exit_return = (net_sell / buy_price - 1.0) * 100.0
        result["Exit_Reason"] = reason
        result["Exit_Week"] = week
        result["Holding_Days"] = day_count
        result["Exit_Date"] = normalize_date(exit_date)
        result["Raw_Exit_Price"] = round(raw_sell_price, 3)
        result["Net_Exit_Price"] = round(net_sell, 3)
        result["Realized_Return (%)"] = round(exit_return, 4)
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
            finalize_exit(curr_open, week, day_count, pending_reason, row.name)
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
            finalize_exit(raw_exit, week, day_count, reason, row.name)
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

        # 买入日收盘已知道是否延续；若开启该独立对照，
        # 只在T+1收盘后挂出退出计划，T+2开盘执行，不偷看T+2数据。
        if (
            day_count == 1
            and exit_non_follow_t2
            and not bool(result["T1_Follow_Through"])
            and pending_reason is None
        ):
            pending_reason = "T+1未延续-T+2开盘退出"

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
            cycle_exit_date = future.iloc[hold_weeks * 5 - 1].name
            finalize_exit(
                last_close,
                hold_weeks,
                hold_weeks * 5,
                "周期结束平仓",
                cycle_exit_date,
            )
            # 周期末强制平仓代表已经完整存活到W8末，不应计为W8前退出。
            result[f"Held_W{hold_weeks}"] = True
        else:
            result["Exit_Reason"] = "持仓中/观察期未满"
    structural_stop = str(result.get("Exit_Reason", "")).startswith("结构止损")
    holding_days = pd.to_numeric(pd.Series([result.get("Holding_Days", np.nan)]), errors="coerce").iloc[0]
    mfe = pd.to_numeric(pd.Series([result.get("MFE_pct (%)", np.nan)]), errors="coerce").iloc[0]
    early_stop = bool(structural_stop and pd.notna(holding_days) and holding_days <= 10)
    result["Early_Structural_Stop_10D"] = early_stop
    result["False_Rebound_10D"] = bool(early_stop and pd.notna(mfe) and mfe < 5.0)
    return result


def alternative_entry_template() -> dict:
    return {
        "T2_Eligible": False,
        "T2_Tradable": False,
        "T2_Skip_Reason": "",
        "T2_Entry_Date": "",
        "T2_Exit_Date": "",
        "T2_Buy_Price": np.nan,
        "T2_Net_Exit_Price": np.nan,
        "T2_Realized_Return_pct": np.nan,
        "T2_Exit_Reason": "",
        "T2_Holding_Days": np.nan,
        "T2_MFE_pct": np.nan,
        "T2_MAE_pct": np.nan,
        "T2_Structural_Stop": False,
        "Split_Invested_pct": np.nan,
        "Split_Realized_Return_pct": np.nan,
    }


def build_alternative_entry_diagnostics(
    main_result: dict,
    ts_code: str,
    market: str,
    signal_low: float,
    signal_ma20: float,
    atr14: float,
    max_gap_pct: float,
    buy_slippage_pct: float,
    sell_slippage_pct: float,
    commission_pct: float,
    sell_tax_pct: float,
    protection_trigger_pct: float,
    protection_buffer_pct: float,
    use_sina: bool = False,
) -> dict:
    """
    主组合仍使用T+1开盘买入。这里只产生两个并行诊断：
    1) T+1收盘延续后，T+2开盘独立买入；
    2) 50%资金按原T+1结果 + 50%资金按T+2结果。若T+1未延续，
       第二半资金保持现金，分批收益为0.5 * 原T+1收益。

    分批结果是两个独立子仓退出的资金加权近似，不是新的真实组合曲线。
    """
    out = alternative_entry_template()
    main_ret = pd.to_numeric(
        pd.Series([main_result.get("Realized_Return (%)", np.nan)]), errors="coerce"
    ).iloc[0]
    follow = bool(main_result.get("T1_Follow_Through", False))
    out["T2_Eligible"] = follow

    if not follow:
        out["T2_Skip_Reason"] = "T+1收盘未延续"
        if pd.notna(main_ret):
            out["Split_Invested_pct"] = 50.0
            out["Split_Realized_Return_pct"] = round(float(main_ret) * 0.5, 4)
        return out

    t1_date = normalize_date(main_result.get("Entry_Date", ""))
    t1_close = pd.to_numeric(
        pd.Series([main_result.get("T1_Close_Price", np.nan)]), errors="coerce"
    ).iloc[0]
    if not t1_date or pd.isna(t1_close) or t1_close <= 0:
        out["T2_Skip_Reason"] = "T+1日期或收盘价缺失"
        if pd.notna(main_ret):
            out["Split_Invested_pct"] = 50.0
            out["Split_Realized_Return_pct"] = round(float(main_ret) * 0.5, 4)
        return out

    delayed = get_medium_term_future(
        ts_code=ts_code,
        market=market,
        selection_date=t1_date,
        signal_close=float(t1_close),
        signal_low=signal_low,
        signal_ma20=signal_ma20,
        atr14=atr14,
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
    out["T2_Tradable"] = bool(delayed.get("Tradable", False))
    out["T2_Entry_Date"] = delayed.get("Entry_Date", "")
    out["T2_Exit_Date"] = delayed.get("Exit_Date", "")
    out["T2_Buy_Price"] = delayed.get("Buy_Price", np.nan)
    out["T2_Net_Exit_Price"] = delayed.get("Net_Exit_Price", np.nan)
    out["T2_Realized_Return_pct"] = delayed.get("Realized_Return (%)", np.nan)
    out["T2_Exit_Reason"] = delayed.get("Exit_Reason", "")
    out["T2_Holding_Days"] = delayed.get("Holding_Days", np.nan)
    out["T2_MFE_pct"] = delayed.get("MFE_pct (%)", np.nan)
    out["T2_MAE_pct"] = delayed.get("MAE_pct (%)", np.nan)
    out["T2_Structural_Stop"] = str(delayed.get("Exit_Reason", "")).startswith("结构止损")
    if not out["T2_Tradable"]:
        out["T2_Skip_Reason"] = str(delayed.get("Exit_Reason", "T+2无法成交"))

    t2_ret = pd.to_numeric(
        pd.Series([out["T2_Realized_Return_pct"]]), errors="coerce"
    ).iloc[0]
    if pd.notna(main_ret):
        if out["T2_Tradable"] and pd.notna(t2_ret):
            out["Split_Invested_pct"] = 100.0
            out["Split_Realized_Return_pct"] = round(
                float(main_ret) * 0.5 + float(t2_ret) * 0.5, 4
            )
        else:
            out["Split_Invested_pct"] = 50.0
            out["Split_Realized_Return_pct"] = round(float(main_ret) * 0.5, 4)
    return out


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
    min_day_gain_pct: float,
    min_daily_bias_pct: float,
    max_bias_pct: float,
    max_weekly_bias_pct: float,
    max_gap_pct: float,
    buy_slippage_pct: float,
    sell_slippage_pct: float,
    commission_pct: float,
    sell_tax_pct: float,
    min_total_score: float,
    scarce_total_score: float,
    breadth_threshold: int,
    min_tradable_signals: int,
    enable_market_filter: bool,
    protection_trigger_pct: float,
    protection_buffer_pct: float,
    exit_non_follow_t2: bool,
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
        "Breadth_Confirmed",
        "Scarce_Tradable_Skipped",
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
            min_day_gain_pct=min_day_gain_pct,
            min_daily_bias_pct=min_daily_bias_pct,
            max_bias_pct=max_bias_pct,
            max_weekly_bias_pct=max_weekly_bias_pct,
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

        stage_priority, stage_priority_label = STAGE_PRIORITY.get(
            indicators["provisional_macd_stage"], (2, "B_中性阶段")
        )

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
            "W_Completed_Date": indicators["weekly_completed_date"],
            "W_MACD_Stage": indicators["weekly_macd_stage"],
            "W_MACD_Hist_pct": round(indicators["weekly_macd_hist_pct"], 4)
            if pd.notna(indicators["weekly_macd_hist_pct"]) else np.nan,
            "W_MACD_Hist_Prev_pct": round(indicators["weekly_macd_hist_prev_pct"], 4)
            if pd.notna(indicators["weekly_macd_hist_prev_pct"]) else np.nan,
            "W_DIF_pct": round(indicators["weekly_dif_pct"], 4)
            if pd.notna(indicators["weekly_dif_pct"]) else np.nan,
            "W_DEA_pct": round(indicators["weekly_dea_pct"], 4)
            if pd.notna(indicators["weekly_dea_pct"]) else np.nan,
            "W_Close": round(indicators["weekly_close"], 3)
            if pd.notna(indicators["weekly_close"]) else np.nan,
            "W_MA20": round(indicators["weekly_ma20"], 3)
            if pd.notna(indicators["weekly_ma20"]) else np.nan,
            "W_MA20_Slope_pct": round(indicators["weekly_ma20_slope_pct"], 4)
            if pd.notna(indicators["weekly_ma20_slope_pct"]) else np.nan,
            "W_Bias_pct": round(indicators["weekly_bias_pct"], 3)
            if pd.notna(indicators["weekly_bias_pct"]) else np.nan,
            "W_DIF_Rising": indicators["weekly_dif_rising"],
            "W_DEA_Rising": indicators["weekly_dea_rising"],
            "W_MA20_Rising": indicators["weekly_ma20_rising"],
            "W_Close_Above_MA20": indicators["weekly_close_above_ma20"],
            "W_Trend_Confirmed": indicators["weekly_trend_confirmed"],
            "W_Wave_Candidate": indicators["weekly_wave_candidate"],
            "P_W_AsOf_Date": indicators["provisional_asof_date"],
            "P_W_MACD_Stage": indicators["provisional_macd_stage"],
            "P_W_MACD_Hist_pct": round(indicators["provisional_macd_hist_pct"], 4)
            if pd.notna(indicators["provisional_macd_hist_pct"]) else np.nan,
            "P_W_MACD_Hist_Prev_pct": round(indicators["provisional_macd_hist_prev_pct"], 4)
            if pd.notna(indicators["provisional_macd_hist_prev_pct"]) else np.nan,
            "P_W_DIF_pct": round(indicators["provisional_dif_pct"], 4)
            if pd.notna(indicators["provisional_dif_pct"]) else np.nan,
            "P_W_DEA_pct": round(indicators["provisional_dea_pct"], 4)
            if pd.notna(indicators["provisional_dea_pct"]) else np.nan,
            "P_W_Close": round(indicators["provisional_close"], 3)
            if pd.notna(indicators["provisional_close"]) else np.nan,
            "P_W_MA20": round(indicators["provisional_ma20"], 3)
            if pd.notna(indicators["provisional_ma20"]) else np.nan,
            "P_W_MA20_Slope_pct": round(indicators["provisional_ma20_slope_pct"], 4)
            if pd.notna(indicators["provisional_ma20_slope_pct"]) else np.nan,
            "P_W_Bias_pct": round(indicators["provisional_bias_pct"], 3)
            if pd.notna(indicators["provisional_bias_pct"]) else np.nan,
            "P_W_DIF_Rising": indicators["provisional_dif_rising"],
            "P_W_DEA_Rising": indicators["provisional_dea_rising"],
            "P_W_MA20_Rising": indicators["provisional_ma20_rising"],
            "P_W_Close_Above_MA20": indicators["provisional_close_above_ma20"],
            "P_W_Trend_Confirmed": indicators["provisional_trend_confirmed"],
            "P_W_Wave_Candidate": indicators["provisional_wave_candidate"],
            "P_W_Changed_From_Completed": indicators["provisional_changed_from_completed"],
            "Stage_Priority": stage_priority,
            "Stage_Priority_Label": stage_priority_label,
            # 下列字段用于通过门槛后的成交回测，候选导出前会移除前导下划线。
            "_Signal_Low": indicators["signal_low"],
            "_Signal_MA20": indicators["ma20"],
            "_ATR14": indicators["atr14"],
        }
        records.append(record)

    if not records:
        funnel["Day_Base_Signal_Count"] = 0
        funnel["Required_Score"] = max(min_total_score, scarce_total_score)
        return pd.DataFrame(), funnel, pd.DataFrame()

    all_candidates = pd.DataFrame(records)
    # V40.2不再因为当日信号多就降低单股质量标准。
    # 保留两个入参只为兼容旧断点结构，实际统一取两者较高值。
    required_score = max(min_total_score, scarce_total_score)
    base_count = int((all_candidates["Total_Score"] >= required_score).sum())
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
            if (
                key.startswith("Eligible_W")
                or key.startswith("Held_W")
                or key in {
                    "Tradable", "T1_Follow_Through",
                    "Early_Structural_Stop_10D", "False_Rebound_10D",
                }
            ):
                all_candidates[key] = pd.Series(pd.NA, index=all_candidates.index, dtype="boolean")
            elif key in ["Exit_Reason", "Entry_Date", "Exit_Date"]:
                all_candidates[key] = pd.Series(None, index=all_candidates.index, dtype="object")
            else:
                all_candidates[key] = np.nan
    for key in alternative_entry_template():
        if key in all_candidates.columns:
            continue
        if key in {"T2_Eligible", "T2_Tradable", "T2_Structural_Stop"}:
            all_candidates[key] = pd.Series(pd.NA, index=all_candidates.index, dtype="boolean")
        elif key in {"T2_Skip_Reason", "T2_Entry_Date", "T2_Exit_Date", "T2_Exit_Reason"}:
            all_candidates[key] = pd.Series(None, index=all_candidates.index, dtype="object")
        else:
            all_candidates[key] = np.nan
    all_candidates["Execution_Checked"] = False
    all_candidates["Selection_Status"] = np.where(
        all_candidates["Score_Pass"], "通过固定质量分数", "未达固定质量分数"
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
                exit_non_follow_t2=exit_non_follow_t2,
                use_sina=use_sina,
            )
            for key, value in future.items():
                all_candidates.loc[idx, key] = value
            alternative = build_alternative_entry_diagnostics(
                main_result=future,
                ts_code=str(candidate["ts_code"]),
                market=str(candidate["market"]),
                signal_low=float(candidate["_Signal_Low"]),
                signal_ma20=float(candidate["_Signal_MA20"]),
                atr14=float(candidate["_ATR14"]),
                max_gap_pct=max_gap_pct,
                buy_slippage_pct=buy_slippage_pct,
                sell_slippage_pct=sell_slippage_pct,
                commission_pct=commission_pct,
                sell_tax_pct=sell_tax_pct,
                protection_trigger_pct=protection_trigger_pct,
                protection_buffer_pct=protection_buffer_pct,
                use_sina=use_sina,
            )
            for key, value in alternative.items():
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
    tradable_count = len(tradable)
    breadth_confirmed = tradable_count >= int(min_tradable_signals)
    all_candidates["Day_Tradable_Signal_Count"] = tradable_count
    all_candidates["Breadth_Confirmed"] = breadth_confirmed
    funnel["Breadth_Confirmed"] = int(breadth_confirmed)
    funnel["Scarce_Tradable_Skipped"] = tradable_count if 0 < tradable_count < int(min_tradable_signals) else 0

    if breadth_confirmed:
        final_indices = (
            tradable.sort_values(
                ["Stage_Priority", "Total_Score", "ATR_pct"],
                ascending=[True, False, False],
            )
            .head(top_k)
            .index
            .tolist()
        )
        all_candidates.loc[tradable.index, "Selection_Status"] = "广度确认-未进入TopK"
        all_candidates.loc[final_indices, "Selection_Status"] = "广度确认-入选TopK"
    else:
        final_indices = []
        if tradable_count > 0:
            all_candidates.loc[tradable.index, "Selection_Status"] = (
                f"实际可成交仅{tradable_count}只<"
                f"{int(min_tradable_signals)}只，观察不买"
            )
    all_candidates["Selected"] = all_candidates.index.isin(final_indices)
    funnel["Selected_TopK"] = len(final_indices)

    internal_cols = ["_Signal_Low", "_Signal_MA20", "_ATR14"]
    candidate_export = all_candidates.drop(columns=internal_cols, errors="ignore").copy()
    if not final_indices:
        return pd.DataFrame(), funnel, candidate_export
    final = candidate_export.loc[final_indices].copy()
    final = final.sort_values(
        ["Stage_Priority", "Total_Score", "ATR_pct"],
        ascending=[True, False, False],
    )
    final.insert(1, "Rank", range(1, len(final) + 1))
    return final, funnel, candidate_export


# -----------------------------------------------------------------------------
# 30万元、最多3只的真实组合回测
# -----------------------------------------------------------------------------
def build_portfolio_backtest(
    signals: pd.DataFrame,
    trade_days: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    max_positions: int = MAX_PORTFOLIO_POSITIONS,
    position_budget: float = POSITION_BUDGET,
    lot_size: int = LOT_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    把独立信号回测转换为有现金、仓位和重复持仓约束的组合回测。

    执行时序：
    - D0收盘产生信号，Entry_Date(D1)开盘按Rank买入。
    - 当日开盘买入时，旧持仓仍占用仓位；当日盘中/收盘退出后才释放。
      这样不使用当日未知的止损结果提前腾仓，是偏保守的模拟。
    - A股数量按100股整手取整；Buy_Price/Net_Exit_Price已包含滑点和费用。
    """
    days = sorted({normalize_date(day) for day in trade_days if normalize_date(day)})
    empty_curve_columns = [
        "Trade_Date", "Cash", "Market_Value", "Equity", "Positions",
        "Exposure_pct", "Daily_Return_pct", "Drawdown_pct", "Is_Empty",
    ]
    if signals is None or signals.empty or not days:
        curve = pd.DataFrame(columns=empty_curve_columns)
        summary = {
            "Initial_Capital": float(initial_capital),
            "Final_Equity": float(initial_capital),
            "Total_Return_pct": 0.0,
            "Max_Drawdown_pct": 0.0,
            "Total_Days": len(days),
            "Empty_Days": len(days),
            "Empty_Ratio_pct": 100.0 if days else np.nan,
            "Invested_Days": 0,
            "Invested_Ratio_pct": 0.0 if days else np.nan,
            "Average_Positions": 0.0,
            "Average_Exposure_pct": 0.0,
            "Executed_Entries": 0,
            "Closed_Trades": 0,
            "Open_Positions": 0,
            "Closed_Win_Rate_pct": np.nan,
            "Structural_Stop_Rate_pct": np.nan,
            "Rejected_Full": 0,
            "Rejected_Duplicate": 0,
        }
        return curve, pd.DataFrame(), pd.DataFrame(), summary

    work = canonicalize_records(signals, ["Trade_Date", "ts_code"])
    for date_col in ["Trade_Date", "Entry_Date", "Exit_Date"]:
        if date_col not in work.columns:
            work[date_col] = ""
        work[date_col] = work[date_col].map(normalize_date)
    for number_col in ["Rank", "Total_Score", "Buy_Price", "Net_Exit_Price", "Realized_Return (%)"]:
        if number_col not in work.columns:
            work[number_col] = np.nan
        work[number_col] = pd.to_numeric(work[number_col], errors="coerce")
    work = work.sort_values(["Entry_Date", "Rank", "Total_Score"], ascending=[True, True, False])
    first_day, last_day = days[0], days[-1]

    # 组合每日盯市使用前复权收盘价，与单笔买入/退出价口径一致。
    price_series: dict[str, pd.Series] = {}
    price_start = (datetime.strptime(first_day, "%Y%m%d") - timedelta(days=15)).strftime("%Y%m%d")
    for ts_code in sorted(work["ts_code"].dropna().astype(str).unique()):
        hist = get_qfq_data(ts_code, price_start, last_day, use_sina=False)
        if hist.empty or "close" not in hist.columns:
            price_series[ts_code] = pd.Series(dtype=float)
        else:
            closes = pd.to_numeric(hist["close"], errors="coerce").dropna()
            closes.index = closes.index.astype(str)
            price_series[ts_code] = closes.sort_index()

    def close_on_or_before(ts_code: str, trade_date: str, fallback: float) -> float:
        series = price_series.get(ts_code, pd.Series(dtype=float))
        subset = series[series.index <= trade_date]
        if subset.empty:
            return float(fallback)
        return float(subset.iloc[-1])

    entry_groups = {
        date: group.sort_values(["Rank", "Total_Score"], ascending=[True, False])
        for date, group in work.groupby("Entry_Date", dropna=False)
        if date
    }
    cash = float(initial_capital)
    active: dict[str, dict] = {}
    executed_trades: list[dict] = []
    order_audit: list[dict] = []
    curve_rows: list[dict] = []

    def audit_order(row: pd.Series, action: str, reason: str, positions_before: int) -> None:
        order_audit.append(
            {
                "Signal_Date": row.get("Trade_Date", ""),
                "Entry_Date": row.get("Entry_Date", ""),
                "Rank": row.get("Rank", np.nan),
                "ts_code": row.get("ts_code", ""),
                "name": row.get("name", ""),
                "Total_Score": row.get("Total_Score", np.nan),
                "W_MACD_Stage": row.get("W_MACD_Stage", ""),
                "W_Trend_Confirmed": row.get("W_Trend_Confirmed", False),
                "P_W_MACD_Stage": row.get("P_W_MACD_Stage", ""),
                "P_W_Trend_Confirmed": row.get("P_W_Trend_Confirmed", False),
                "Stage_Priority": row.get("Stage_Priority", np.nan),
                "Stage_Priority_Label": row.get("Stage_Priority_Label", ""),
                "Day_Gain_pct": row.get("Day_Gain_pct", np.nan),
                "MA20_Bias_pct": row.get("MA20_Bias_pct", np.nan),
                "T1_Follow_Through": row.get("T1_Follow_Through", False),
                "False_Rebound_10D": row.get("False_Rebound_10D", False),
                "T2_Eligible": row.get("T2_Eligible", False),
                "T2_Tradable": row.get("T2_Tradable", False),
                "T2_Realized_Return_pct": row.get("T2_Realized_Return_pct", np.nan),
                "Split_Realized_Return_pct": row.get("Split_Realized_Return_pct", np.nan),
                "Portfolio_Action": action,
                "Portfolio_Reason": reason,
                "Positions_Before": positions_before,
            }
        )

    day_set = set(days)
    for _, outside_row in work[~work["Entry_Date"].isin(day_set)].iterrows():
        audit_order(outside_row, "未计入组合", "T+1买入日不在本次组合窗口", 0)

    for trade_date in days:
        # 先在开盘处理D1候选；当日将退出的旧持仓不提前释放仓位。
        for _, row in entry_groups.get(trade_date, pd.DataFrame()).iterrows():
            ts_code = str(row["ts_code"])
            positions_before = len(active)
            if ts_code in active:
                audit_order(row, "未买入", "同一股票已在持仓", positions_before)
                continue
            if len(active) >= int(max_positions):
                audit_order(row, "未买入", "3个仓位已满", positions_before)
                continue
            buy_price = float(row["Buy_Price"]) if pd.notna(row["Buy_Price"]) else np.nan
            if not np.isfinite(buy_price) or buy_price <= 0:
                audit_order(row, "未买入", "买入价无效", positions_before)
                continue
            budget = min(float(position_budget), cash)
            shares = int(np.floor(budget / buy_price / int(lot_size)) * int(lot_size))
            if shares < int(lot_size):
                audit_order(row, "未买入", "可用现金不足一手", positions_before)
                continue
            cost = shares * buy_price
            cash -= cost
            trade = {
                "Signal_Date": row.get("Trade_Date", ""),
                "Entry_Date": trade_date,
                "Rank": row.get("Rank", np.nan),
                "ts_code": ts_code,
                "name": row.get("name", ""),
                "W_MACD_Stage": row.get("W_MACD_Stage", ""),
                "W_Trend_Confirmed": row.get("W_Trend_Confirmed", False),
                "P_W_MACD_Stage": row.get("P_W_MACD_Stage", ""),
                "P_W_Trend_Confirmed": row.get("P_W_Trend_Confirmed", False),
                "Stage_Priority": row.get("Stage_Priority", np.nan),
                "Stage_Priority_Label": row.get("Stage_Priority_Label", ""),
                "Day_Gain_pct": row.get("Day_Gain_pct", np.nan),
                "MA20_Bias_pct": row.get("MA20_Bias_pct", np.nan),
                "T1_Follow_Through": row.get("T1_Follow_Through", False),
                "False_Rebound_10D": row.get("False_Rebound_10D", False),
                "T2_Eligible": row.get("T2_Eligible", False),
                "T2_Tradable": row.get("T2_Tradable", False),
                "T2_Realized_Return_pct": row.get("T2_Realized_Return_pct", np.nan),
                "Split_Realized_Return_pct": row.get("Split_Realized_Return_pct", np.nan),
                "Buy_Price": round(buy_price, 3),
                "Shares": shares,
                "Entry_Cost": round(cost, 2),
                "Planned_Exit_Date": row.get("Exit_Date", ""),
                "Actual_Exit_Date": "",
                "Net_Exit_Price": np.nan,
                "Exit_Proceeds": np.nan,
                "PnL": np.nan,
                "Portfolio_Return (%)": np.nan,
                "Exit_Reason": row.get("Exit_Reason", ""),
                "Portfolio_Status": "持仓中",
                "_fallback_price": buy_price,
            }
            executed_trades.append(trade)
            active[ts_code] = trade
            audit_order(row, "已买入", f"买入{shares}股", positions_before)

        # 当日退出在日末入账，不为当日开盘的新交易提前腾仓。
        exiting_codes = []
        for ts_code, trade in active.items():
            exit_date = normalize_date(trade.get("Planned_Exit_Date", ""))
            if exit_date != trade_date:
                continue
            net_exit_price = work.loc[
                (work["Trade_Date"] == trade["Signal_Date"]) & (work["ts_code"] == ts_code),
                "Net_Exit_Price",
            ]
            net_exit_price = float(net_exit_price.iloc[-1]) if not net_exit_price.empty else np.nan
            if not np.isfinite(net_exit_price) or net_exit_price <= 0:
                # 理论上已完整退出的记录必有净卖价；缺失时保守按当日收盘估值。
                net_exit_price = close_on_or_before(ts_code, trade_date, trade["Buy_Price"])
            proceeds = trade["Shares"] * net_exit_price
            cash += proceeds
            pnl = proceeds - trade["Entry_Cost"]
            trade["Actual_Exit_Date"] = trade_date
            trade["Net_Exit_Price"] = round(net_exit_price, 3)
            trade["Exit_Proceeds"] = round(proceeds, 2)
            trade["PnL"] = round(pnl, 2)
            trade["Portfolio_Return (%)"] = round(pnl / trade["Entry_Cost"] * 100.0, 4)
            trade["Portfolio_Status"] = "已平仓"
            exiting_codes.append(ts_code)
        for ts_code in exiting_codes:
            active.pop(ts_code, None)

        market_value = 0.0
        for ts_code, trade in active.items():
            mark = close_on_or_before(ts_code, trade_date, trade["_fallback_price"])
            market_value += trade["Shares"] * mark
            trade["_last_mark"] = mark
        equity = cash + market_value
        positions = len(active)
        exposure = market_value / equity * 100.0 if equity > 0 else 0.0
        curve_rows.append(
            {
                "Trade_Date": trade_date,
                "Cash": round(cash, 2),
                "Market_Value": round(market_value, 2),
                "Equity": round(equity, 2),
                "Positions": positions,
                "Exposure_pct": round(exposure, 2),
                "Is_Empty": positions == 0,
            }
        )

    curve = pd.DataFrame(curve_rows)
    curve["Daily_Return_pct"] = curve["Equity"].pct_change().fillna(
        curve["Equity"].iloc[0] / float(initial_capital) - 1.0
    ) * 100.0
    running_peak = curve["Equity"].cummax().clip(lower=float(initial_capital))
    curve["Drawdown_pct"] = (curve["Equity"] / running_peak - 1.0) * 100.0

    # 给回测结束时仍持有的仓位补充未实现盈亏。
    for trade in executed_trades:
        if trade["Portfolio_Status"] == "持仓中":
            mark = float(trade.get("_last_mark", trade["Buy_Price"]))
            market_value = trade["Shares"] * mark
            pnl = market_value - trade["Entry_Cost"]
            trade["Mark_Date"] = last_day
            trade["Mark_Price"] = round(mark, 3)
            trade["Market_Value"] = round(market_value, 2)
            trade["PnL"] = round(pnl, 2)
            trade["Portfolio_Return (%)"] = round(pnl / trade["Entry_Cost"] * 100.0, 4)
        trade.pop("_fallback_price", None)
        trade.pop("_last_mark", None)

    ledger = pd.DataFrame(executed_trades)
    orders = pd.DataFrame(order_audit)
    empty_days = int(curve["Is_Empty"].sum())
    invested_days = len(curve) - empty_days
    closed = ledger[ledger["Portfolio_Status"] == "已平仓"].copy() if not ledger.empty else pd.DataFrame()
    final_equity = float(curve.iloc[-1]["Equity"])
    summary = {
        "Initial_Capital": float(initial_capital),
        "Final_Equity": final_equity,
        "Total_Return_pct": (final_equity / float(initial_capital) - 1.0) * 100.0,
        "Max_Drawdown_pct": float(curve["Drawdown_pct"].min()),
        "Total_Days": len(curve),
        "Empty_Days": empty_days,
        "Empty_Ratio_pct": empty_days / len(curve) * 100.0,
        "Invested_Days": invested_days,
        "Invested_Ratio_pct": invested_days / len(curve) * 100.0,
        "Average_Positions": float(curve["Positions"].mean()),
        "Average_Exposure_pct": float(curve["Exposure_pct"].mean()),
        "Executed_Entries": len(ledger),
        "Closed_Trades": len(closed),
        "Open_Positions": int((ledger["Portfolio_Status"] == "持仓中").sum()) if not ledger.empty else 0,
        "Closed_Win_Rate_pct": (
            float((closed["Portfolio_Return (%)"] > 0).mean() * 100.0) if not closed.empty else np.nan
        ),
        "Structural_Stop_Rate_pct": (
            float(closed["Exit_Reason"].fillna("").str.startswith("结构止损").mean() * 100.0)
            if not closed.empty else np.nan
        ),
        "Rejected_Full": int((orders.get("Portfolio_Reason", pd.Series(dtype=str)) == "3个仓位已满").sum()),
        "Rejected_Duplicate": int((orders.get("Portfolio_Reason", pd.Series(dtype=str)) == "同一股票已在持仓").sum()),
    }
    return curve, ledger, orders, summary


# -----------------------------------------------------------------------------
# 报表
# -----------------------------------------------------------------------------
def show_portfolio_report(
    curve: pd.DataFrame,
    ledger: pd.DataFrame,
    orders: pd.DataFrame,
    summary: dict,
) -> None:
    st.subheader("💼 30万元·最多3只真实组合")
    first = st.columns(4)
    first[0].metric("组合期末权益", f"¥{summary['Final_Equity']:,.0f}")
    first[1].metric("组合总收益", f"{summary['Total_Return_pct']:.2f}%")
    first[2].metric("最大回撤", f"{summary['Max_Drawdown_pct']:.2f}%")
    first[3].metric(
        "真实空仓率",
        f"{summary['Empty_Ratio_pct']:.1f}%",
        help="按回测交易日每日收盘持仓数为0计算。",
    )
    second = st.columns(4)
    second[0].metric(
        "有持仓日",
        f"{summary['Invested_Days']}/{summary['Total_Days']} ({summary['Invested_Ratio_pct']:.1f}%)",
    )
    second[1].metric("平均持仓数", f"{summary['Average_Positions']:.2f}/3")
    second[2].metric("平均资金暴露", f"{summary['Average_Exposure_pct']:.1f}%")
    second[3].metric(
        "已执行/已平仓/在持",
        f"{summary['Executed_Entries']}/{summary['Closed_Trades']}/{summary['Open_Positions']}",
    )
    third = st.columns(4)
    closed_win = summary["Closed_Win_Rate_pct"]
    structural = summary["Structural_Stop_Rate_pct"]
    third[0].metric("已平仓胜率", "--" if pd.isna(closed_win) else f"{closed_win:.1f}%")
    third[1].metric("亏损性结构止损率", "--" if pd.isna(structural) else f"{structural:.1f}%")
    third[2].metric("因仓位已满未买", f"{summary['Rejected_Full']}笔")
    third[3].metric("因重复持仓未买", f"{summary['Rejected_Duplicate']}笔")

    if not curve.empty:
        chart = curve.set_index(pd.to_datetime(curve["Trade_Date"], format="%Y%m%d"))[["Equity"]]
        st.line_chart(chart, use_container_width=True)
        position_distribution = (
            curve["Positions"].value_counts().reindex(range(MAX_PORTFOLIO_POSITIONS + 1), fill_value=0)
            .rename_axis("收盘持仓数").reset_index(name="交易日数")
        )
        position_distribution["占比(%)"] = (
            position_distribution["交易日数"] / len(curve) * 100.0
        ).round(1)
        st.dataframe(position_distribution, use_container_width=True, hide_index=True)

    if not ledger.empty:
        st.caption("组合成交账本（持仓中的收益按回测截止日收盘估值）")
        st.dataframe(ledger, use_container_width=True, hide_index=True, height=420)
    if not orders.empty:
        with st.expander("查看每个信号的组合执行/拒绝原因"):
            st.dataframe(orders, use_container_width=True, hide_index=True, height=360)


def show_quality_priority_report(all_results: pd.DataFrame) -> None:
    st.subheader("🎯 V40.2周线阶段优先级成效")
    required = {
        "Stage_Priority_Label", "Realized_Return (%)", "Exit_Reason", "Held_W1"
    }
    if all_results.empty or not required.issubset(all_results.columns):
        st.info("暂无周线优先级诊断数据。")
        return
    data = all_results.copy()
    data["_ret"] = pd.to_numeric(data["Realized_Return (%)"], errors="coerce")
    data = data[data["_ret"].notna()].copy()
    if data.empty:
        st.info("当前样本尚未成熟。")
        return
    data["_结构止损"] = data["Exit_Reason"].fillna("").astype(str).str.startswith("结构止损")
    data["_W1生存"] = bool_series(data["Held_W1"])
    rows = []
    for label, group in data.groupby("Stage_Priority_Label", dropna=False):
        rows.append(
            {
                "周线优先级": label,
                "样本": len(group),
                "W1生存率(%)": round(group["_W1生存"].mean() * 100.0, 1),
                "结构止损率(%)": round(group["_结构止损"].mean() * 100.0, 1),
                "实质盈利率(%)": round((group["_ret"] >= 0.1).mean() * 100.0, 1),
                "平均收益(%)": round(group["_ret"].mean(), 2),
                "中位收益(%)": round(group["_ret"].median(), 2),
            }
        )
    st.dataframe(
        pd.DataFrame(rows).sort_values("周线优先级"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "A1=G2绿柱缩短，A2=R1首周翻红；B=中性阶段；"
        "C=G1绿柱连续缩短或R5红柱衰减。C类仅降级，没有硬删除。"
    )


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


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False).astype(bool)
    )


def show_weekly_wave_report(all_results: pd.DataFrame) -> None:
    st.subheader("🌊 周线MACD波浪阶段验证（只诊断，未参与选股）")
    required = {"W_MACD_Stage", "Realized_Return (%)", "Exit_Reason"}
    if all_results.empty or not required.issubset(all_results.columns):
        st.info("暂无周线波浪诊断样本。")
        return

    data = all_results.copy()
    data["_ret"] = pd.to_numeric(data["Realized_Return (%)"], errors="coerce")
    data = data[data["_ret"].notna()].copy()
    if data.empty:
        st.info("当前信号尚未形成完整退出收益。")
        return

    data["_实质盈利"] = data["_ret"] >= 0.1
    data["_近似保本"] = (data["_ret"] > 0) & (data["_ret"] < 0.1)
    data["_亏损"] = data["_ret"] < 0
    data["_结构止损"] = data["Exit_Reason"].fillna("").astype(str).str.startswith("结构止损")
    data["_假反弹"] = (
        bool_series(data["False_Rebound_10D"])
        if "False_Rebound_10D" in data.columns else False
    )
    data["_周趋势确认"] = (
        bool_series(data["W_Trend_Confirmed"])
        if "W_Trend_Confirmed" in data.columns else False
    )

    rows = []
    for stage, group in data.groupby("W_MACD_Stage", dropna=False):
        n = len(group)
        rows.append(
            {
                "周线阶段": stage,
                "样本": n,
                "实质盈利率(%)": round(group["_实质盈利"].mean() * 100.0, 1),
                "保本率(%)": round(group["_近似保本"].mean() * 100.0, 1),
                "亏损率(%)": round(group["_亏损"].mean() * 100.0, 1),
                "结构止损率(%)": round(group["_结构止损"].mean() * 100.0, 1),
                "10日假反弹率(%)": round(group["_假反弹"].mean() * 100.0, 1),
                "平均收益(%)": round(group["_ret"].mean(), 2),
                "中位收益(%)": round(group["_ret"].median(), 2),
                "周趋势确认(%)": round(group["_周趋势确认"].mean() * 100.0, 1),
                "_order": WEEKLY_STAGE_ORDER.get(str(stage), 99),
            }
        )
    stage_table = pd.DataFrame(rows).sort_values(["_order", "样本"], ascending=[True, False])
    st.dataframe(stage_table.drop(columns=["_order"]), use_container_width=True, hide_index=True)

    headline = st.columns(4)
    wave_mask = data["W_MACD_Stage"].isin(
        ["R1_首周翻红", "R2_第二根红柱扩张", "R3_红柱再加速", "R4_红柱扩张", "G1_绿柱连续缩短"]
    )
    wave = data[wave_mask]
    weak = data[data["W_MACD_Stage"].isin(["R5_红柱衰减", "G3_绿柱扩大/弱势"])]
    headline[0].metric("波浪候选样本", len(wave))
    headline[1].metric(
        "候选结构止损率",
        "--" if wave.empty else f"{wave['_结构止损'].mean()*100:.1f}%",
    )
    headline[2].metric("衰减/弱势样本", len(weak))
    headline[3].metric(
        "衰减/弱势止损率",
        "--" if weak.empty else f"{weak['_结构止损'].mean()*100:.1f}%",
    )

    if "T1_Follow_Through" in data.columns:
        data["T1状态"] = np.where(bool_series(data["T1_Follow_Through"]), "T+1延续", "T+1未延续")
        follow_rows = []
        for label, group in data.groupby("T1状态"):
            follow_rows.append(
                {
                    "T+1状态": label,
                    "样本": len(group),
                    "实质盈利率(%)": round(group["_实质盈利"].mean() * 100.0, 1),
                    "亏损率(%)": round(group["_亏损"].mean() * 100.0, 1),
                    "结构止损率(%)": round(group["_结构止损"].mean() * 100.0, 1),
                    "10日假反弹率(%)": round(group["_假反弹"].mean() * 100.0, 1),
                    "平均收益(%)": round(group["_ret"].mean(), 2),
                }
            )
        st.caption("T+1延续：买入日收盘同时不低于信号日收盘和买入日开盘。")
        st.dataframe(pd.DataFrame(follow_rows), use_container_width=True, hide_index=True)


def show_provisional_weekly_report(all_results: pd.DataFrame) -> None:
    st.subheader("🌫️ 信号日临时周MACD（P_W_字段）")
    required = {"P_W_MACD_Stage", "W_MACD_Stage", "Realized_Return (%)", "Exit_Reason"}
    if all_results.empty or not required.issubset(all_results.columns):
        st.info("暂无临时周MACD诊断样本。")
        return
    data = all_results.copy()
    data["_ret"] = pd.to_numeric(data["Realized_Return (%)"], errors="coerce")
    data = data[data["_ret"].notna()].copy()
    if data.empty:
        st.info("当前样本尚未成熟。")
        return
    data["_实质盈利"] = data["_ret"] >= 0.1
    data["_保本"] = (data["_ret"] > 0) & (data["_ret"] < 0.1)
    data["_亏损"] = data["_ret"] < 0
    data["_结构止损"] = data["Exit_Reason"].fillna("").astype(str).str.startswith("结构止损")
    data["_假反弹"] = (
        bool_series(data["False_Rebound_10D"])
        if "False_Rebound_10D" in data.columns else False
    )
    data["_临时趋势确认"] = (
        bool_series(data["P_W_Trend_Confirmed"])
        if "P_W_Trend_Confirmed" in data.columns else False
    )
    rows = []
    for stage, group in data.groupby("P_W_MACD_Stage", dropna=False):
        rows.append(
            {
                "临时周阶段": stage,
                "样本": len(group),
                "实质盈利率(%)": round(group["_实质盈利"].mean() * 100.0, 1),
                "保本率(%)": round(group["_保本"].mean() * 100.0, 1),
                "亏损率(%)": round(group["_亏损"].mean() * 100.0, 1),
                "结构止损率(%)": round(group["_结构止损"].mean() * 100.0, 1),
                "10日假反弹率(%)": round(group["_假反弹"].mean() * 100.0, 1),
                "平均收益(%)": round(group["_ret"].mean(), 2),
                "中位收益(%)": round(group["_ret"].median(), 2),
                "临时趋势确认(%)": round(group["_临时趋势确认"].mean() * 100.0, 1),
                "_order": WEEKLY_STAGE_ORDER.get(str(stage), 99),
            }
        )
    table = pd.DataFrame(rows).sort_values(["_order", "样本"], ascending=[True, False])
    st.dataframe(table.drop(columns=["_order"]), use_container_width=True, hide_index=True)

    changed = data["P_W_MACD_Stage"].astype(str) != data["W_MACD_Stage"].astype(str)
    metrics = st.columns(3)
    metrics[0].metric("完成周→临时周发生阶段变化", f"{changed.sum()}/{len(data)} ({changed.mean()*100:.1f}%)")
    provisional_wave = data["P_W_MACD_Stage"].isin(
        ["R1_首周翻红", "R2_第二根红柱扩张", "R3_红柱再加速", "R4_红柱扩张", "G1_绿柱连续缩短"]
    )
    wave = data[provisional_wave]
    metrics[1].metric("临时周波浪候选", len(wave))
    metrics[2].metric(
        "临时候选结构止损率",
        "--" if wave.empty else f"{wave['_结构止损'].mean()*100:.1f}%",
    )
    with st.expander("查看完成周阶段→临时周阶段转移矩阵"):
        cross = pd.crosstab(data["W_MACD_Stage"], data["P_W_MACD_Stage"], margins=True)
        st.dataframe(cross, use_container_width=True)


def show_alternative_entry_report(all_results: pd.DataFrame) -> None:
    st.subheader("⏱️ T+1原入场 vs T+2确认 vs 50/50分批（诊断）")
    needed = {"Realized_Return (%)", "T2_Realized_Return_pct", "Split_Realized_Return_pct"}
    if all_results.empty or not needed.issubset(all_results.columns):
        st.info("暂无替代入场诊断数据。")
        return
    data = all_results.copy()
    data["_main"] = pd.to_numeric(data["Realized_Return (%)"], errors="coerce")
    data["_t2"] = pd.to_numeric(data["T2_Realized_Return_pct"], errors="coerce")
    data["_split"] = pd.to_numeric(data["Split_Realized_Return_pct"], errors="coerce")
    t2_tradable = bool_series(data["T2_Tradable"]) if "T2_Tradable" in data.columns else pd.Series(False, index=data.index)
    t2_mature = t2_tradable & data["_t2"].notna()

    def metric_row(label: str, returns: pd.Series, structural: pd.Series | None = None) -> dict:
        values = pd.to_numeric(returns, errors="coerce").dropna()
        if values.empty:
            return {"方案": label, "成熟样本": 0}
        row = {
            "方案": label,
            "成熟样本": len(values),
            "实质盈利率(%)": round((values >= 0.1).mean() * 100.0, 1),
            "保本率(%)": round(((values > 0) & (values < 0.1)).mean() * 100.0, 1),
            "亏损率(%)": round((values < 0).mean() * 100.0, 1),
            "平均收益(%)": round(values.mean(), 2),
            "中位收益(%)": round(values.median(), 2),
        }
        if structural is not None:
            aligned = bool_series(structural.reindex(values.index))
            row["结构止损率(%)"] = round(aligned.mean() * 100.0, 1)
        else:
            row["结构止损率(%)"] = np.nan
        return row

    main_struct = data["Exit_Reason"].fillna("").astype(str).str.startswith("结构止损")
    t2_struct = bool_series(data["T2_Structural_Stop"]) if "T2_Structural_Stop" in data.columns else pd.Series(False, index=data.index)
    rows = [
        metric_row("原T+1全部信号", data["_main"], main_struct),
        metric_row("原T+1（仅T+2可成交的延续样本）", data.loc[t2_mature, "_main"], main_struct.loc[t2_mature]),
        metric_row("T+2确认买入（同一批样本）", data.loc[t2_mature, "_t2"], t2_struct.loc[t2_mature]),
        metric_row("50/50分批近似（全部信号）", data["_split"], None),
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    eligible = bool_series(data["T2_Eligible"]) if "T2_Eligible" in data.columns else pd.Series(False, index=data.index)
    info = st.columns(4)
    info[0].metric("T+1延续样本", f"{eligible.sum()}/{len(data)}")
    info[1].metric("T+2可成交且成熟", f"{t2_mature.sum()}/{len(data)}")
    info[2].metric(
        "分批平均资金投入",
        "--" if "Split_Invested_pct" not in data.columns else f"{pd.to_numeric(data['Split_Invested_pct'], errors='coerce').mean():.1f}%",
    )
    info[3].metric("T+2不可成交/未成熟", int(eligible.sum() - t2_mature.sum()))
    st.caption(
        "50/50分批为两个独立子仓收益的资金加权近似；"
        "本版主组合仍完全按V39.9/V40.0的T+1开盘买入执行。"
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
    TOP_BACKTEST = st.number_input(
        "每日优选 TopK", min_value=2, max_value=20, value=5, step=1,
        help="TopK用于保留候选广度；真实同时持仓仍由组合引擎限制为3只。",
    )

    st.subheader("已固化的股票池与交易约束")
    MIN_PRICE = FIXED_MIN_PRICE
    MIN_MV = FIXED_MIN_MV
    MAX_MV = FIXED_MAX_MV
    ENABLE_MARKET_FILTER = FIXED_MARKET_FILTER
    st.info(
        "股价≥20元；流通市值200~1000亿元；大盘硬过滤关闭；"
        "T+1实际可成交不足2只时观察不买。"
    )
    st.caption("组合：初始30万元，最多3只，单只目标约10万元。")
    st.info(
        "V40.2只收紧单股质量并改变同日排名；"
        "不限制每日新开仓数，不限制同一申万行业，保留热点板块联动。"
    )
    ENABLE_THS = st.checkbox("实时雷达启用THS概念补充(需6000积分)", value=False)
    THS_KEYWORDS_TEXT = st.text_area("THS科技概念关键词", value=DEFAULT_THS_KEYWORDS, height=100)

    st.subheader("单股质量与固定评分门槛")
    MIN_ATR_PCT = st.number_input("ATR14/股价最低(%)", min_value=0.0, value=2.5, step=0.1)
    MIN_VOL_RATIO = st.number_input("最低量比", min_value=0.5, value=1.10, step=0.05)
    MIN_DAY_GAIN_PCT = st.number_input(
        "信号日最低涨幅(%)", min_value=0.0,
        value=DEFAULT_MIN_DAY_GAIN_PCT, step=0.5,
    )
    MIN_DAILY_BIAS_PCT = st.number_input(
        "信号日高于日MA20最低(%)", min_value=0.0,
        value=DEFAULT_MIN_DAILY_BIAS_PCT, step=0.5,
    )
    MAX_BIAS_PCT = st.number_input("相对MA20最大偏离(%)", min_value=1.0, value=8.0, step=0.5)
    MAX_WEEKLY_BIAS_PCT = st.number_input(
        "信号日临时周线高于周MA20最大(%)", min_value=5.0,
        value=DEFAULT_MAX_WEEKLY_BIAS_PCT, step=5.0,
    )
    MAX_GAP_PCT = st.number_input("T+1最大允许高开(%)", min_value=0.0, value=8.0, step=0.5)
    FIXED_SCORE_FLOOR = st.number_input(
        "固定最低总分", min_value=0.0,
        value=DEFAULT_FIXED_SCORE_FLOOR, step=1.0,
        help="不再因为当日信号较多而从82分降到80分。",
    )
    MIN_TOTAL_SCORE = FIXED_SCORE_FLOOR
    SCARCE_TOTAL_SCORE = FIXED_SCORE_FLOOR
    BREADTH_THRESHOLD = 4  # 仅为兼容旧函数入参，V40.2不再用它降分。

    st.subheader("市场环境")
    st.caption("双指数仅保留为诊断字段，不再一刀切暂停开仓。")

    st.subheader("成交成本与保护止损")
    BUY_SLIPPAGE = st.number_input("买入滑点(%)", min_value=0.0, value=0.20, step=0.05)
    SELL_SLIPPAGE = st.number_input("卖出滑点(%)", min_value=0.0, value=0.20, step=0.05)
    COMMISSION = st.number_input("单边佣金(%)", min_value=0.0, value=0.03, step=0.01)
    SELL_TAX = st.number_input("卖出税费(%)", min_value=0.0, value=0.05, step=0.01)
    PROTECTION_TRIGGER = st.number_input("保护止损触发浮盈(%)", min_value=1.0, value=10.0, step=1.0)
    PROTECTION_BUFFER = st.number_input("保护止损原始价高于成本(%)", min_value=0.0, value=0.30, step=0.10)
    EXIT_NON_FOLLOW_T2 = st.checkbox(
        "T+1未延续时T+2开盘提前退出（独立对照）",
        value=False,
        help=(
            "第一轮建议保持关闭，先测试新入选条件；"
            "第二轮只打开此项，检验提前退出是否降低亏损。"
        ),
    )

    RESUME_CHECKPOINT = st.checkbox("开启参数隔离的断点续传", value=True)
    USE_CACHE = st.checkbox("使用并增量更新行情缓存", value=True)
    if st.button("清除V39.7/V39.9/V40.x共享行情缓存"):
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
        if MIN_DAILY_BIAS_PCT >= MAX_BIAS_PCT:
            raise ValueError("日MA20最低偏离必须小于最大偏离")
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
        # 市场硬过滤已经固化关闭，不再额外请求双指数数据。
        GLOBAL_REGIME_INDICES = {}

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
            "min_day_gain": MIN_DAY_GAIN_PCT,
            "min_daily_bias": MIN_DAILY_BIAS_PCT,
            "max_bias": MAX_BIAS_PCT,
            "max_weekly_bias": MAX_WEEKLY_BIAS_PCT,
            "max_gap": MAX_GAP_PCT,
            "min_score": MIN_TOTAL_SCORE,
            "scarce_score": SCARCE_TOTAL_SCORE,
            "breadth_threshold": int(BREADTH_THRESHOLD),
            "min_tradable_signals": MIN_TRADABLE_SIGNALS,
            "market_filter": bool(ENABLE_MARKET_FILTER),
            "buy_slip": BUY_SLIPPAGE,
            "sell_slip": SELL_SLIPPAGE,
            "commission": COMMISSION,
            "sell_tax": SELL_TAX,
            "protection_trigger": PROTECTION_TRIGGER,
            "protection_buffer": PROTECTION_BUFFER,
            "exit_non_follow_t2": bool(EXIT_NON_FOLLOW_T2),
            "initial_capital": INITIAL_CAPITAL,
            "max_positions": MAX_PORTFOLIO_POSITIONS,
            "position_budget": POSITION_BUDGET,
        }
        config_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        result_file = f"backtest_{VERSION.replace('.', '_')}_{config_hash}.csv"
        funnel_file = f"funnel_{VERSION.replace('.', '_')}_{config_hash}.csv"
        candidate_file = f"candidates_{VERSION.replace('.', '_')}_{config_hash}.csv"
        state_file = f"state_{VERSION.replace('.', '_')}_{config_hash}.json"

        processed: set[str] = set()
        results_store = pd.DataFrame()
        candidates_store = pd.DataFrame()
        funnel_store = pd.DataFrame()
        if RESUME_CHECKPOINT and os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as file:
                    state = json.load(file)
                state_processed = {normalize_date(day) for day in state.get("processed_dates", [])}
                state_processed.discard("")
                results_store = read_checkpoint_csv(result_file, ["Trade_Date", "ts_code"])
                candidates_store = read_checkpoint_csv(candidate_file, ["Trade_Date", "ts_code"])
                funnel_store = read_checkpoint_csv(funnel_file, ["Trade_Date"])
                # 每个交易日必须有一条漏斗记录；只有state而没有漏斗快照的日期重跑。
                funnel_dates = set(funnel_store.get("Trade_Date", pd.Series(dtype=str)).astype(str))
                processed = state_processed & funnel_dates

                # 读取旧断点后立即紧缩为唯一主键快照，从源头消除历史重复。
                if not results_store.empty:
                    results_store = atomic_write_csv(results_store, result_file, ["Trade_Date", "ts_code"])
                if not candidates_store.empty:
                    candidates_store = atomic_write_csv(candidates_store, candidate_file, ["Trade_Date", "ts_code"])
                if not funnel_store.empty:
                    funnel_store = atomic_write_csv(funnel_store, funnel_file, ["Trade_Date"])
            except Exception as exc:
                record_api_error(f"断点读取失败，将从头运行: {exc}")
                processed = set()
                results_store = pd.DataFrame()
                candidates_store = pd.DataFrame()
                funnel_store = pd.DataFrame()
        elif not RESUME_CHECKPOINT:
            # 用户明确关闭续跑时，本次使用同一配置的文件从头覆盖。
            for exact_path in [result_file, candidate_file, funnel_file, state_file]:
                if os.path.exists(exact_path):
                    os.remove(exact_path)

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
                min_day_gain_pct=float(MIN_DAY_GAIN_PCT),
                min_daily_bias_pct=float(MIN_DAILY_BIAS_PCT),
                max_bias_pct=float(MAX_BIAS_PCT),
                max_weekly_bias_pct=float(MAX_WEEKLY_BIAS_PCT),
                max_gap_pct=float(MAX_GAP_PCT),
                buy_slippage_pct=float(BUY_SLIPPAGE),
                sell_slippage_pct=float(SELL_SLIPPAGE),
                commission_pct=float(COMMISSION),
                sell_tax_pct=float(SELL_TAX),
                min_total_score=float(MIN_TOTAL_SCORE),
                scarce_total_score=float(SCARCE_TOTAL_SCORE),
                breadth_threshold=int(BREADTH_THRESHOLD),
                min_tradable_signals=MIN_TRADABLE_SIGNALS,
                enable_market_filter=bool(ENABLE_MARKET_FILTER),
                protection_trigger_pct=float(PROTECTION_TRIGGER),
                protection_buffer_pct=float(PROTECTION_BUFFER),
                exit_non_follow_t2=bool(EXIT_NON_FOLLOW_T2),
                use_sina=use_sina,
            )
            results_store = replace_trade_date_records(
                results_store, day_result, trade_date, ["Trade_Date", "ts_code"]
            )
            candidates_store = replace_trade_date_records(
                candidates_store, day_candidates, trade_date, ["Trade_Date", "ts_code"]
            )
            funnel_store = replace_trade_date_records(
                funnel_store, pd.DataFrame([funnel]), trade_date, ["Trade_Date"]
            )

            # 小数据集每日写完整快照；不再append，因此断点重跑也不会双倍。
            results_store = atomic_write_csv(results_store, result_file, ["Trade_Date", "ts_code"])
            candidates_store = atomic_write_csv(candidates_store, candidate_file, ["Trade_Date", "ts_code"])
            funnel_store = atomic_write_csv(funnel_store, funnel_file, ["Trade_Date"])
            processed.add(trade_date)
            atomic_write_json(
                {"config": config, "processed_dates": sorted(processed)},
                state_file,
            )
            progress.progress((i + 1) / max(len(dates_to_run), 1), text=f"分析 {trade_date}")
        progress.empty()

        st.header(f"📊 {VERSION} 回测结果")
        funnel_df = canonicalize_records(funnel_store, ["Trade_Date"])
        all_candidates_export = canonicalize_records(
            candidates_store, ["Trade_Date", "ts_code"]
        )

        st.subheader("🧭 固定质量门槛与多信号确认")
        if not funnel_df.empty:
            funnel_defaults = {
                "Market_Weak_Block": 0,
                "Market_Regime": "过滤关闭",
                "Day_Base_Signal_Count": 0,
                "Required_Score": np.nan,
                "Dynamic_Score_Pass": 0,
                "Tradable_Signal": 0,
                "Breadth_Confirmed": 0,
                "Scarce_Tradable_Skipped": 0,
                "Selected_TopK": 0,
                "CYB_Close": np.nan,
                "CYB_MA20": np.nan,
                "STAR50_Close": np.nan,
                "STAR50_MA20": np.nan,
            }
            for column, default_value in funnel_defaults.items():
                if column not in funnel_df.columns:
                    funnel_df[column] = default_value
            score_pass_total = int(pd.to_numeric(funnel_df["Dynamic_Score_Pass"], errors="coerce").fillna(0).sum())
            selected_total = int(pd.to_numeric(funnel_df["Selected_TopK"], errors="coerce").fillna(0).sum())
            breadth_days = int(pd.to_numeric(
                funnel_df["Breadth_Confirmed"], errors="coerce"
            ).fillna(0).sum())
            scarce_days = int((pd.to_numeric(
                funnel_df["Scarce_Tradable_Skipped"], errors="coerce"
            ).fillna(0) > 0).sum())
            env_cols = st.columns(4)
            env_cols[0].metric("通过固定质量分数", score_pass_total)
            env_cols[1].metric("至少2只可成交的日期", breadth_days)
            env_cols[2].metric("仅1只而放弃的日期", scarce_days)
            env_cols[3].metric("最终入选样本", selected_total)
            st.dataframe(
                funnel_df[
                    [
                        "Trade_Date", "Market_Regime", "Day_Base_Signal_Count", "Required_Score",
                        "Dynamic_Score_Pass", "Tradable_Signal", "Breadth_Confirmed",
                        "Scarce_Tradable_Skipped", "Selected_TopK",
                    ]
                ].sort_values("Trade_Date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        if not results_store.empty:
            all_results = canonicalize_records(results_store, ["Trade_Date", "ts_code"])
            for week in range(1, 9):
                for prefix in ["Eligible_W", "Held_W"]:
                    col = f"{prefix}{week}"
                    if col in all_results.columns and all_results[col].dtype == object:
                        all_results[col] = (
                            all_results[col].astype(str).str.lower()
                            .map({"true": True, "false": False})
                            .fillna(False)
                        )

            portfolio_curve, portfolio_ledger, portfolio_orders, portfolio_summary = (
                build_portfolio_backtest(
                    all_results,
                    trade_days,
                    initial_capital=INITIAL_CAPITAL,
                    max_positions=MAX_PORTFOLIO_POSITIONS,
                    position_budget=POSITION_BUDGET,
                    lot_size=LOT_SIZE,
                )
            )
            show_portfolio_report(
                portfolio_curve, portfolio_ledger, portfolio_orders, portfolio_summary
            )
            show_quality_priority_report(all_results)
            show_weekly_report(all_results)
            show_weekly_wave_report(all_results)
            show_provisional_weekly_report(all_results)
            show_alternative_entry_report(all_results)
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
                "📥 下载V40.2周线质量优先轨迹CSV",
                export_csv,
                f"export_v40_2_{config_hash}.csv",
                "text/csv",
            )
            if not portfolio_curve.empty:
                st.download_button(
                    "📥 下载V40.2主组合每日权益CSV",
                    portfolio_curve.to_csv(index=False).encode("utf-8-sig"),
                    f"portfolio_curve_v40_2_{config_hash}.csv",
                    "text/csv",
                )
            if not portfolio_ledger.empty:
                st.download_button(
                    "📥 下载V40.2主组合成交账本CSV",
                    portfolio_ledger.to_csv(index=False).encode("utf-8-sig"),
                    f"portfolio_ledger_v40_2_{config_hash}.csv",
                    "text/csv",
                )
            if not portfolio_orders.empty:
                st.download_button(
                    "📥 下载V40.2信号执行审计CSV",
                    portfolio_orders.to_csv(index=False).encode("utf-8-sig"),
                    f"portfolio_orders_v40_2_{config_hash}.csv",
                    "text/csv",
                )
        else:
            empty_curve, empty_ledger, empty_orders, empty_summary = build_portfolio_backtest(
                pd.DataFrame(), trade_days
            )
            show_portfolio_report(empty_curve, empty_ledger, empty_orders, empty_summary)
            st.warning("本次没有产生可交易信号。请先查看筛选漏斗确定主要淘汰环节。")

        if not all_candidates_export.empty:
            candidate_csv = all_candidates_export.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 下载V40.2全部核心候选及诊断CSV",
                candidate_csv,
                f"candidates_v40_2_{config_hash}.csv",
                "text/csv",
            )
        if not funnel_df.empty:
            funnel_csv = funnel_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 下载V40.2逐日筛选漏斗CSV",
                funnel_csv,
                f"funnel_v40_2_{config_hash}.csv",
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
        with st.expander("查看详细错误位置"):
            st.code(traceback.format_exc())
        if API_ERRORS:
            st.code("\n".join(API_ERRORS[-30:]))
