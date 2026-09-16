# -*- coding: utf-8 -*-
"""
周线 SKDJ 分级补位选股系统 (V22)
------------------------------------------------
V22 新增（板块跟踪规则不变，已记录的跟踪数据继续有效）：
1. 【板块内选股方法对比】回测中，在每周前3强申万二级板块里，按 7 种事先固定的方法
   各挑 2 只，与“前3强板块全部成分”（相当于在板块里随机挑）同周对照：
   12周最强 / 12周最弱 / 市值最大 / 市值最小 / 20日波动最低 / 近1周跌最多 / 近1周涨最多。
2. 报告新增“板块内选股方法对比”表：1/2/4/12 周相对板块平均的超额与 t 值、
   分界前后两段分别统计、牛股率差与熊股率差。事先定好的判定标准写在表下方。
因新增指标，回测记录文件改名，需要重扫（行情缓存直接复用）。

V21 新增：
1. 【板块动量前瞻跟踪】新模式“🧭 板块动量跟踪”（默认模式）。规则在代码中冻结，
   从 2026-09-18 这一周起，每周自动记录前3强申万二级板块及其成分股，
   按次日开盘买入分别在 1/2/4 周后自动结算超额（相对同周股票池）。
   名单一经记录即冻结，之后只更新收益；错过的周下次打开时按同一规则补记并标注。
2. 跟踪结果一键导出 ZIP；缓存备份同时包含跟踪记录。
3. 股票池筛选代码抽成公共函数，回测与跟踪共用，保证口径一致（回测结果与 V20 相同）。

V20 新增（选股规则不变）：
1. 【短周期对照】每个对照组增加 1 周（5日）、2 周（10日）固定持有超额，
   总览表同时给出 1/2/4/12 周超额和各自的 t 值，用于检验 3~15 天持有的效果。
2. 【样本外分段】侧边栏可填“样本外分界日期”，总览表自动分成“分界前 / 分界后”两段，
   同一份记录一次看清样本内外是否一致。
3. 修正：入选组名称按“每周选股数量”显示（之前固定写成“入选5只”）。
因新增指标，回测记录文件改名，需要重扫一次（行情缓存直接复用，不用重新下载）。

V19 修复（选股与回测逻辑与 V18 完全一致，回测记录可继续沿用）：
1. 【内存】行情改为按年压缩存储、只保留股票池内股票、逐年读取后直接转成 numpy 数组，
   不再把全市场多年日线拼成大 DataFrame。回测 5 年内存占用从 2GB 以上降到几百 MB，
   避免 Streamlit Cloud 超内存重启。
2. 【缓存不丢】侧边栏“缓存备份与恢复”：一键把行情仓库和回测记录打包下载到电脑，
   重启后上传即可恢复，不用重新下载。
3. 【断点续传】下载每满 40 天写盘一次，中途崩溃最多损失 40 天。
4. 【自动迁移】旧版逐日缓存首次运行自动转入新仓库并删除旧文件，释放磁盘。
5. 股票池新增股票时只按只补历史，不重下全部日期。

V18 新增：
1. 【下行尾部指标】每个对照组增加 熊股率（60个交易日内最大回撤≤-20%的比例）、
   平均最大回撤、12周收益中位数。牛股率高但熊股率同样高，说明只是波动大，不是选股能力。
2. 【报告区间筛选】侧边栏可设定报告起止日期，同一份回测记录可以分别查看
   样本外区间（如 2018~2022）与原回测区间，互不混合。
3. 保留 V17 的四路并发下载与一键导出（ZIP 中附带报告区间）。

V17 新增（选股规则仍不变）：
1. 【四路并发下载】行情同步默认 4 线程并发（侧边栏可调 1~8），遇到 Tushare
   限流自动退避重试；失败的日期下次运行自动补下。
2. 【一键导出】回测报告顶部一个按钮，把全部报告表格、交割流水、每周明细和
   回测参数打包成 ZIP 下载。
3. 【两个新对照组】事先固定定义、不做参数优化，只用于和股票池同周对照：
   - 强势股前10%：过筛股票池中，截至上周的 12 周涨幅排名前 10%
     （跳过最近一周，避免把短期反转混进来）；
   - 强势板块：按申万二级行业分组（组内≥5只），用成分股 12 周涨幅均值排名，
     取前 3 个板块 → “前3强板块全部成分” 与 “前3强板块各取最强2只”。
4. 【候选组总览】每组给出 12 周超额、重叠修正后的 t 值、跑赢池的半年数、
   牛股率差，一张表判断哪个方向值得继续。

V16 新增（不改变任何选股规则，只增加“对照组”用于识别行情依赖与过拟合）：
A. 【同周对照】每个回测周同时计算三组的固定持有表现（次日开盘买入，不设止损）：
   入选股票 / 当周全部 A 级候选 / 当周全部过筛股票池。
   入选≈A候选 → 评分排序没有价值；A候选≈股票池 → 信号本身没有超额，赚的是行情。
B. 【牛股率】12 周内最高涨幅 ≥30% 的比例，直接衡量“捕捉牛股”的能力，与出场规则无关。
C. 【市场宽度】过筛股票池中周收盘站上 20 周线的比例，只做诊断与展示，不参与选股。
D. 【分时期报告】按半年、按宽度分组给出上述对照，每周等权，避免同周股票同涨同跌放大样本。
E. 未走完 12 周的近期周每次回测自动重扫结算。

V15 改动：
1. 【分级补位】A 级 = 原信号（周K上穿25且K>D）；A 级不足时依次用
   B 级（低位金叉 K≤30）、C 级（强势股回踩后周线再金叉）补满每周名额。
2. 【漏斗诊断】选股时显示股票池逐层剩余数量，0 只时能看到卡在哪一步。
3. 【数据未更新回退】当天日线未发布（Tushare 收盘后才更新）时自动改用最近
   一个有数据的交易日，并明确提示，不再静默返回空结果。
4. 【空周统计】回测记录每一周（包括 0 只的周），按年统计空周数。
5. 【参数隔离】回测记录按参数组分文件保存，改参数后不会和旧结果混在一起。
6. 【回测偏差修正】
   - W1~W12 表改为全样本口径（已出场的单子按出场收益计入），不再只统计幸存者；
   - “保本离场”按次日真实开盘价计算，不再固定记 +2%；
   - 最低股价用未复权价格判断；
   - “持仓中”的单子每次回测时用最新行情重新结算。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta, timezone
import warnings
import time
import os
import re
import json
import hashlib
import pickle
import gzip
import tempfile
import shutil
import gc
import traceback
import threading
import random
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    fcntl = None

warnings.filterwarnings("ignore")

VERSION = "V22"
LOGIC_VERSION = "V22"   # 选股/回测逻辑版本，决定参数组编号与记录文件名

SKDJ_N, SKDJ_M = 6, 3
HOLD_WEEKS = 12
FWD_HORIZONS = {"W1": 5, "W2": 10, "W4": 20, "W8": 40, "W12": 60}
BIG_WINNER_PCT = 30.0
BIG_LOSER_PCT = -20.0
RS_LOOKBACK_WEEKS = 12
RS_TOP_PCT = 10.0
SECTOR_MIN_MEMBERS = 5
SECTOR_TOP_K = 3
SECTOR_STOCKS_EACH = 2

# ---- 前瞻跟踪（规则冻结，勿改；改动会让跟踪失去样本外意义）
TRACK_START = "20260918"
TRACK_CFG = {"min_price": 10.0, "min_mv": 50.0, "max_mv": 1000.0}
TRACK_HORIZONS = {"W1": 5, "W2": 10, "W4": 20}
TRACK_EXPECT_W2 = 0.36     # 2013~2026 回测：前3强板块全部成分 2周超额均值
TRACK_STOP_W2 = -1.0       # 满52周后平均2周超额低于此值 → 判定失效
TRACK_WEEKS_FILE = "skdj_track_weeks.csv"
TRACK_STOCKS_FILE = "skdj_track_stocks.csv"
TRACK_POOL_FILE = "skdj_track_pool.csv"

# 板块内选股方法（事先固定，勿按结果调整）：键, 名称, 排序特征, 是否降序
METHOD_GROUPS = [
    ("SECRS", "12周最强2只", "rs", True),
    ("SECWK", "12周最弱2只", "rs", False),
    ("SECBIG", "市值最大2只", "mv", True),
    ("SECSML", "市值最小2只", "mv", False),
    ("SECLV", "20日波动最低2只", "vol20", False),
    ("SECDIP", "近1周跌最多2只", "ret1w", False),
    ("SECHOT", "近1周涨最多2只", "ret1w", True),
]

BENCH_GROUPS = [
    ("Pick", "入选5只"),
    ("A", "A级候选"),
    ("RS", "强势股前10%"),
    ("SEC", "前3强板块全部成分"),
    ("SECRS", "前3强板块各取最强2只"),
]

TIER_ORDER = {"A": 0, "B": 1, "C": 2}
TIER_LABEL = {"A": "A 标准上穿25", "B": "B 低位金叉", "C": "C 趋势回踩金叉"}

st.set_page_config(page_title="板块动量跟踪 · SKDJ 回测系统 V22", layout="wide")
st.title("🔬 板块动量跟踪 · SKDJ 回测系统 (V22)")
st.markdown("板块动量前瞻跟踪 · 同周对照回测 · 四路并发下载 · 一键导出 · 缓存备份")


# ---------------------------
# Token 与安全请求
# ---------------------------
def clean_token_str(raw_token):
    if not raw_token:
        return ""
    return re.sub(r'[\s\u3000\ufeff\xa0\r\n]+', '', str(raw_token)).strip()


def verify_token_connection(token_str):
    if not token_str:
        return False, "Token 为空，请在侧边栏填入 Token。"
    try:
        ts.set_token(token_str)
        pro = ts.pro_api(token_str)
        test_df = pro.trade_cal(exchange='SSE', start_date='20260801', end_date='20260805')
        if test_df is not None and not test_df.empty:
            return True, "验证通过"
        return False, "Token 校验未返回数据，请检查网络连接。"
    except Exception as e:
        err_msg = str(e)
        if "token不对" in err_msg or "-40001" in err_msg:
            return False, "您的 Token 不正确，请检查复制内容。"
        return False, f"接口校验失败: {err_msg}"


RATE_LIMIT_HINTS = ("每分钟", "频率", "最多访问", "too many", "Too Many", "rate limit")


def safe_tushare_call(func, max_retries=3, sleep_time=0.8, **kwargs):
    for attempt in range(max_retries):
        try:
            df = func(**kwargs)
            if df is not None and not df.empty:
                return df
            time.sleep(sleep_time)
        except Exception as e:
            if any(h in str(e) for h in RATE_LIMIT_HINTS):
                time.sleep(min(60.0, 8.0 * (attempt + 1)) + random.uniform(0, 2))  # 限流退避
            else:
                time.sleep(sleep_time * (attempt + 1))
    return pd.DataFrame()


_thread_local = threading.local()


def _thread_pro(token):
    """每个下载线程各用一个 pro 实例。"""
    pro = getattr(_thread_local, "pro", None)
    if pro is None or getattr(_thread_local, "token", None) != token:
        pro = ts.pro_api(token)
        _thread_local.pro = pro
        _thread_local.token = token
    return pro


# ---------------------------
# 原子读写与文件锁
# ---------------------------
def _atomic_replace_bytes(write_callback, target_path):
    target_dir = os.path.dirname(os.path.abspath(target_path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=os.path.basename(target_path) + ".", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        write_callback(temp_path)
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _atomic_write_csv(dataframe, target_path):
    def writer(temp_path):
        dataframe.to_csv(temp_path, index=False, encoding="utf-8-sig")
        with open(temp_path, "rb") as file_obj:
            os.fsync(file_obj.fileno())

    if os.path.exists(target_path):
        try:
            shutil.copy2(target_path, target_path + ".bak")
        except OSError:
            pass
    _atomic_replace_bytes(writer, target_path)


def _read_csv_safely(target_path):
    for candidate in (target_path, target_path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            df = pd.read_csv(candidate, encoding="utf-8-sig", low_memory=False,
                             dtype={"Trade_Date": str, "ts_code": str, "Exit_Date": str, "Signal_Date": str})
            for col in ("Trade_Date", "Exit_Date", "Signal_Date"):
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True).replace({"nan": "", "None": ""})
            return df
        except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
    return pd.DataFrame()


@contextmanager
def _file_lock(path):
    handle = open(path + ".lock", "a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _append_rows_safely(path, new_rows, key_cols, sort_cols):
    with _file_lock(path):
        existing = _read_csv_safely(path)
        combined = pd.concat([existing, new_rows], ignore_index=True, sort=False) if not existing.empty else new_rows.copy()
        combined["Trade_Date"] = combined["Trade_Date"].astype(str).str.replace(r"\.0$", "", regex=True)
        if set(key_cols).issubset(combined.columns):
            combined = combined.drop_duplicates(key_cols, keep="last")
        sort_cols = [c for c in sort_cols if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols, kind="mergesort")
        _atomic_write_csv(combined.reset_index(drop=True), path)


def _replace_date_rows(path, date, new_rows, sort_cols, key_col="Trade_Date"):
    """删除该日期的旧记录后写入新记录（用于重扫未结算的周）。"""
    with _file_lock(path):
        existing = _read_csv_safely(path)
        if not existing.empty and key_col in existing.columns:
            existing = existing[existing[key_col].astype(str) != str(date)]
        parts = [p for p in (existing, new_rows) if p is not None and not p.empty]
        if parts:
            combined = pd.concat(parts, ignore_index=True, sort=False)
            combined[key_col] = combined[key_col].astype(str).str.replace(r"\.0$", "", regex=True)
            sort_cols = [c for c in sort_cols if c in combined.columns]
            if sort_cols:
                combined = combined.sort_values(sort_cols, kind="mergesort")
        else:
            combined = existing
        _atomic_write_csv(combined.reset_index(drop=True), path)


# ---------------------------
# 科技白名单池（保持原逻辑不变）
# ---------------------------
@st.cache_data(ttl=3600 * 24 * 7, show_spinner=False)
def load_custom_tech_whitelist(token):
    token_c = clean_token_str(token)
    if not token_c:
        return set(), {}
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)

    stock_basic = safe_tushare_call(pro.stock_basic, list_status='L', fields='ts_code,symbol,name,industry,market,list_date')
    if stock_basic.empty:
        return set(), {}

    BOARDS = ("主板", "创业板", "科创板")
    valid_stocks = stock_basic[stock_basic['market'].isin(BOARDS)].copy()
    valid_stocks = valid_stocks[~valid_stocks['name'].str.contains('ST|退', na=False)]
    valid_stocks = valid_stocks[~valid_stocks['ts_code'].str.startswith('92')]

    CORE_TECH_L1 = {"电子", "计算机", "通信", "国防军工"}
    EXTENDED_TECH_L1 = {"机械设备", "电力设备", "医药生物", "汽车", "基础化工", "有色金属", "建筑材料"}
    TECH_INDUSTRY_KEYWORDS = {
        "半导体", "电子元件", "元件", "光学光电子", "消费电子", "电子化学品",
        "计算机设备", "软件开发", "IT服务", "通信设备", "军工电子", "航空装备",
        "航天装备", "自动化设备", "机器人", "激光设备", "工控设备", "仪器仪表",
        "电池", "光伏设备", "风电设备", "电网设备", "电机", "医疗器械",
        "生物制品", "汽车电子", "金属新材料", "非金属材料", "膜材料", "碳纤维",
    }

    sw_indices = safe_tushare_call(pro.index_classify, level='L1', src='SW2021')
    tech_l1_names = CORE_TECH_L1.union(EXTENDED_TECH_L1)
    target_sw = sw_indices[sw_indices['industry_name'].isin(tech_l1_names)] if not sw_indices.empty else pd.DataFrame()

    stock_sw_map = {}
    if not target_sw.empty:
        for _, s_row in target_sw.iterrows():
            m_df = safe_tushare_call(pro.index_member, index_code=s_row['index_code'], is_new='Y')
            if not m_df.empty:
                for c_code in m_df['con_code']:
                    stock_sw_map[c_code] = s_row['industry_name']
            time.sleep(0.03)

    whitelist_set = set()
    name_map = dict(zip(stock_basic['ts_code'], stock_basic['name']))
    for _, row in valid_stocks.iterrows():
        code = row['ts_code']
        ind_basic = str(row['industry']) if pd.notna(row['industry']) else ""
        sw_l1 = stock_sw_map.get(code, "")
        if sw_l1 in CORE_TECH_L1:
            whitelist_set.add(code)
            continue
        if sw_l1 in EXTENDED_TECH_L1:
            if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS) or ind_basic == "" or sw_l1 in {"机械设备", "电力设备", "医药生物"}:
                whitelist_set.add(code)
                continue
        if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS):
            whitelist_set.add(code)
    return whitelist_set, name_map


# ---------------------------
# 申万二级行业映射（仅用于“强势板块”对照组）
# ---------------------------
@st.cache_data(ttl=3600 * 24 * 7, show_spinner=False)
def load_sw_l2_map(token):
    """返回 {ts_code: (l2_code, l2_name)}。优先 index_member_all，失败时逐个二级行业调 index_member。
    注意：使用当前行业归属回看历史，存在少量行业变更偏差。"""
    token_c = clean_token_str(token)
    if not token_c:
        return {}
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    mapping = {}
    l1 = safe_tushare_call(pro.index_classify, level='L1', src='SW2021')
    if not l1.empty:
        try:
            probe = safe_tushare_call(pro.index_member_all, max_retries=1, l1_code=l1['index_code'].iloc[0], is_new='Y')
            if not probe.empty and {'ts_code', 'l2_code', 'l2_name'}.issubset(probe.columns):
                for code in l1['index_code']:
                    df = probe if code == l1['index_code'].iloc[0] else safe_tushare_call(pro.index_member_all, l1_code=code, is_new='Y')
                    if not df.empty and {'ts_code', 'l2_code', 'l2_name'}.issubset(df.columns):
                        for r in df[['ts_code', 'l2_code', 'l2_name']].itertuples(index=False):
                            mapping[r.ts_code] = (r.l2_code, r.l2_name)
                    time.sleep(0.05)
        except Exception:
            pass
    if len(mapping) < 1000:
        l2 = safe_tushare_call(pro.index_classify, level='L2', src='SW2021')
        if not l2.empty:
            for r in l2[['index_code', 'industry_name']].itertuples(index=False):
                m_df = safe_tushare_call(pro.index_member, max_retries=2, index_code=r.index_code, is_new='Y')
                if not m_df.empty and 'con_code' in m_df.columns:
                    for c_code in m_df['con_code']:
                        mapping.setdefault(c_code, (r.index_code, r.industry_name))
                time.sleep(0.05)
    return mapping


# ---------------------------
# 行情仓库（V19）：按年压缩存储、只存股票池内股票、逐年读取，内存占用约为 V18 的 1/10
# ---------------------------
STORE_DIR = "skdj_market_store"
LEGACY_CACHE_DIR = "skdj_market_data_daily_cache"   # V14.5~V18 的逐日缓存，首次运行自动迁移
PRICE_FIELDS = ("open", "high", "low", "close", "pre_close")   # float32 存储，读取时还原两位小数
F64_FIELDS = ("vol", "adj_factor", "circ_mv")                    # float64 存储，保证与 V18 结果一致
STORE_FIELDS = PRICE_FIELDS + F64_FIELDS
FLUSH_EVERY_DAYS = 40


def _year_path(year):
    return os.path.join(STORE_DIR, f"{int(year)}.npz")


def _store_years():
    if not os.path.isdir(STORE_DIR):
        return []
    years = []
    for fn in os.listdir(STORE_DIR):
        m = re.fullmatch(r"(\d{4})\.npz", fn)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def load_year_meta(year):
    path = _year_path(year)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return {'dates': z['dates'].astype(np.int64), 'codes': z['codes'].astype(str)}
    except Exception:
        return None


def load_year(year):
    path = _year_path(year)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            data = {'dates': z['dates'].astype(np.int64), 'codes': z['codes'].astype(str)}
            shape = (len(data['dates']), len(data['codes']))
            for f in STORE_FIELDS:
                arr = z[f]
                if arr.shape != shape:
                    raise ValueError(f"{f} 形状不符")
                data[f] = arr
        return data
    except Exception:
        try:
            os.replace(path, path + ".broken")  # 损坏文件改名保留，缺失日期会重新下载
        except OSError:
            pass
        return None


def save_year(year, data):
    os.makedirs(STORE_DIR, exist_ok=True)

    def writer(temp_path):
        with open(temp_path, "wb") as fh:
            payload = {f: np.asarray(data[f], dtype=np.float32 if f in PRICE_FIELDS else np.float64) for f in STORE_FIELDS}
            np.savez_compressed(fh, dates=np.asarray(data['dates'], dtype=np.int32),
                                codes=np.asarray(data['codes'], dtype='<U12'), **payload)
            fh.flush()
            os.fsync(fh.fileno())
    _atomic_replace_bytes(writer, _year_path(year))


def merge_year(existing, day_frames=None, backfill=None, keep_codes=()):
    """
    day_frames: {date_int: DataFrame(ts_code + 字段)}，整日下载的数据。
      年文件已有日期时，只能写入年文件已覆盖的代码（新代码必须先补历史，避免把旧日期误当成停牌）。
    backfill:   {ts_code: DataFrame(trade_date + 字段)}，单只股票补历史，只填年文件已有日期。
    """
    old_dates = existing['dates'] if existing is not None else np.array([], dtype=np.int64)
    old_codes = existing['codes'] if existing is not None else np.array([], dtype=str)
    has_old = len(old_dates) > 0
    day_frames = dict(day_frames or {})
    backfill = dict(backfill or {})

    if day_frames:
        allowed = set(old_codes.tolist()) if has_old else set(keep_codes)
        for d in list(day_frames):
            df = day_frames[d]
            if not has_old:
                allowed.update(df['ts_code'].tolist())
            day_frames[d] = df[df['ts_code'].isin(allowed)]
        if not has_old:
            code_set = allowed
        else:
            code_set = set(old_codes.tolist())
    else:
        code_set = set(old_codes.tolist())
    code_set = code_set | set(backfill)

    new_dates = np.array(sorted(int(d) for d in day_frames), dtype=np.int64)
    dates = np.union1d(old_dates, new_dates).astype(np.int64)
    codes = np.array(sorted(code_set), dtype='<U12')
    n, m = len(dates), len(codes)
    out = {'dates': dates, 'codes': codes}
    r_old = np.searchsorted(dates, old_dates)
    c_old = np.searchsorted(codes, old_codes)
    for f in STORE_FIELDS:
        mat = np.full((n, m), np.nan, dtype=np.float32 if f in PRICE_FIELDS else np.float64)
        if has_old and len(old_codes):
            mat[np.ix_(r_old, c_old)] = existing[f]
        out[f] = mat

    if day_frames:
        for d, df in day_frames.items():
            if df.empty:
                continue
            r = int(np.searchsorted(dates, int(d)))
            cidx = np.searchsorted(codes, df['ts_code'].to_numpy(dtype=str))
            for f in STORE_FIELDS:
                if f in df.columns:
                    out[f][r, cidx] = pd.to_numeric(df[f], errors='coerce').to_numpy(dtype=float)

    if backfill and n:
        for code, df in backfill.items():
            if df is None or df.empty:
                continue
            c = int(np.searchsorted(codes, code))
            td = pd.to_numeric(df['trade_date'], errors='coerce').to_numpy(dtype=float)
            ridx = np.searchsorted(dates, td)
            ok = (ridx < n) & (dates[np.clip(ridx, 0, n - 1)] == td)
            if not ok.any():
                continue
            for f in STORE_FIELDS:
                if f in df.columns:
                    out[f][ridx[ok], c] = pd.to_numeric(df[f], errors='coerce').to_numpy(dtype=float)[ok]
    return out


def store_summary():
    years = _store_years()
    n_dates, codes, size = 0, set(), 0
    first = last = None
    for y in years:
        meta = load_year_meta(y)
        if meta is None or len(meta['dates']) == 0:
            continue
        n_dates += len(meta['dates'])
        codes.update(meta['codes'].tolist())
        first = int(meta['dates'].min()) if first is None else min(first, int(meta['dates'].min()))
        last = int(meta['dates'].max()) if last is None else max(last, int(meta['dates'].max()))
        size += os.path.getsize(_year_path(y))
    return {'years': years, 'n_dates': n_dates, 'n_codes': len(codes), 'first': first, 'last': last,
            'size_mb': size / 1024 / 1024}


# ---- 旧版逐日缓存（只读，用于迁移）
def _legacy_partition_path(trade_date):
    return os.path.join(LEGACY_CACHE_DIR, f"{trade_date}.pkl.gz")


def _legacy_dates():
    if not os.path.isdir(LEGACY_CACHE_DIR):
        return []
    out = []
    for fn in os.listdir(LEGACY_CACHE_DIR):
        m = re.fullmatch(r"(\d{8})\.pkl\.gz", fn)
        if m:
            out.append(m.group(1))
    return sorted(out)


def _valid_market_partition(payload, trade_date):
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != 1 or str(payload.get("trade_date")) != str(trade_date):
        return False
    daily, adj, basic = payload.get("daily"), payload.get("adj"), payload.get("daily_basic")
    if not all(isinstance(frame, pd.DataFrame) for frame in (daily, adj, basic)):
        return False
    if daily.empty or adj.empty:
        return False
    if not {"ts_code", "trade_date", "open", "high", "low", "close", "vol"}.issubset(daily.columns):
        return False
    if not {"ts_code", "trade_date", "adj_factor"}.issubset(adj.columns):
        return False
    if int(payload.get("daily_count", 0)) < 1000 or int(payload.get("adj_count", 0)) < 1000:
        return False
    return True


def _read_legacy_partition(trade_date):
    path = _legacy_partition_path(trade_date)
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rb") as file_obj:
            payload = pickle.load(file_obj)
        if _valid_market_partition(payload, trade_date):
            return payload
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        pass
    return None


def _compact_day_frame(df_d, df_a, df_b, keep_codes):
    """全市场当日数据 → 只保留需要的股票和字段。"""
    df_d = df_d[df_d['ts_code'].isin(keep_codes)]
    cols = ['ts_code'] + [c for c in ('open', 'high', 'low', 'close', 'pre_close', 'vol') if c in df_d.columns]
    out = df_d[cols].drop_duplicates('ts_code', keep='last')
    adj = df_a[df_a['ts_code'].isin(keep_codes)][['ts_code', 'adj_factor']].drop_duplicates('ts_code', keep='last')
    out = out.merge(adj, on='ts_code', how='left')
    if df_b is not None and not df_b.empty and 'circ_mv' in df_b.columns:
        basic = df_b[df_b['ts_code'].isin(keep_codes)][['ts_code', 'circ_mv']].drop_duplicates('ts_code', keep='last')
        out = out.merge(basic, on='ts_code', how='left')
    else:
        out['circ_mv'] = np.nan
    return out.reset_index(drop=True)


def _fetch_market_day(trade_date, token, keep_codes):
    """线程中运行，不调用 streamlit。优先读旧缓存，没有才下载。返回 (日期, 精简数据或None, 来源)。"""
    payload = _read_legacy_partition(trade_date)
    if payload is not None:
        return trade_date, _compact_day_frame(payload['daily'], payload['adj'], payload['daily_basic'], keep_codes), "legacy"
    pro = _thread_pro(token)
    df_d = safe_tushare_call(pro.daily, max_retries=5, trade_date=trade_date)
    df_a = safe_tushare_call(pro.adj_factor, max_retries=5, trade_date=trade_date)
    df_b = safe_tushare_call(pro.daily_basic, max_retries=5, trade_date=trade_date, fields='ts_code,trade_date,circ_mv')
    time.sleep(0.15)
    if len(df_d) < 1000 or len(df_a) < 1000:
        return trade_date, None, "api"
    return trade_date, _compact_day_frame(df_d, df_a, df_b, keep_codes), "api"


def _tushare_fetch_strict(func, max_retries=5, **kwargs):
    """区分“接口正常但无数据”(返回空表) 与 “请求失败”(返回 None)。"""
    for attempt in range(max_retries):
        try:
            df = func(**kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            if any(h in str(e) for h in RATE_LIMIT_HINTS):
                time.sleep(min(60.0, 8.0 * (attempt + 1)) + random.uniform(0, 2))
            else:
                time.sleep(0.8 * (attempt + 1))
    return None


def _fetch_stock_history(ts_code, start_date, end_date, token):
    pro = _thread_pro(token)
    d = _tushare_fetch_strict(pro.daily, ts_code=ts_code, start_date=start_date, end_date=end_date)
    a = _tushare_fetch_strict(pro.adj_factor, ts_code=ts_code, start_date=start_date, end_date=end_date)
    b = _tushare_fetch_strict(pro.daily_basic, ts_code=ts_code, start_date=start_date, end_date=end_date,
                              fields='ts_code,trade_date,circ_mv')
    time.sleep(0.1)
    if d is None or a is None or b is None:
        return ts_code, None
    if d.empty:
        return ts_code, pd.DataFrame(columns=['trade_date'])
    cols = ['trade_date'] + [c for c in ('open', 'high', 'low', 'close', 'pre_close', 'vol') if c in d.columns]
    out = d[cols].drop_duplicates('trade_date', keep='last')
    if not a.empty:
        out = out.merge(a[['trade_date', 'adj_factor']].drop_duplicates('trade_date'), on='trade_date', how='left')
    else:
        out['adj_factor'] = np.nan
    if not b.empty and 'circ_mv' in b.columns:
        out = out.merge(b[['trade_date', 'circ_mv']].drop_duplicates('trade_date'), on='trade_date', how='left')
    else:
        out['circ_mv'] = np.nan
    return ts_code, out


def sync_market_store(start_date, end_date, token, whitelist_keys, n_workers=4):
    token_c = clean_token_str(token)
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_date, end_date=end_date)
    if cal_raw.empty:
        return []
    all_dates = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date')['cal_date'].astype(str).tolist()
    today_str = datetime.now().strftime("%Y%m%d")
    valid_dates = [d for d in all_dates if d <= today_str]
    whitelist = set(whitelist_keys)
    workers = int(max(1, min(8, n_workers)))
    window_years = sorted({int(d[:4]) for d in valid_dates})
    if os.path.isdir(STORE_DIR):  # 清理上次崩溃留下的半截临时文件
        for fn in os.listdir(STORE_DIR):
            if fn.endswith(".tmp") or fn.endswith(".restore"):
                try:
                    os.remove(os.path.join(STORE_DIR, fn))
                except OSError:
                    pass

    # 1) 股票池新增的股票：先按只补齐仓库里已有日期的历史
    need = {}
    for y in window_years:
        meta = load_year_meta(y)
        if meta is None or len(meta['dates']) == 0:
            continue
        lo, hi = int(meta['dates'].min()), int(meta['dates'].max())
        for c in whitelist - set(meta['codes'].tolist()):
            r = need.setdefault(c, [lo, hi])
            r[0], r[1] = min(r[0], lo), max(r[1], hi)
    if need:
        bar = st.progress(0, text=f"📥 股票池新增 {len(need)} 只，补齐历史（{workers} 线程）...")
        fetched, failed = {}, 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_fetch_stock_history, c, str(r[0]), str(r[1]), token_c) for c, r in need.items()]
            for i, fut in enumerate(as_completed(futures), start=1):
                try:
                    code, df = fut.result()
                except Exception:
                    code, df = None, None
                if df is None:
                    failed += 1
                else:
                    fetched[code] = df
                bar.progress(i / len(futures), text=f"📥 补齐历史: {i}/{len(futures)}")
        bar.empty()
        for y in window_years:
            existing = load_year(y)
            if existing is None or len(existing['dates']) == 0:
                continue
            have = set(existing['codes'].tolist())
            bf = {c: df for c, df in fetched.items() if c not in have}
            if bf:
                save_year(y, merge_year(existing, backfill=bf))
            del existing
        gc.collect()
        if failed:
            st.warning(f"⚠️ {failed} 只股票历史补齐失败（多为限流），下次运行会自动重试。")

    # 2) 缺失的交易日：旧缓存有就迁移，没有就下载
    store_dates = set()
    for y in _store_years():
        meta = load_year_meta(y)
        if meta is not None:
            store_dates.update(int(d) for d in meta['dates'])
    todo = sorted({d for d in valid_dates if int(d) not in store_dates} |
                  {d for d in _legacy_dates() if int(d) not in store_dates})
    if todo:
        keep_by_year = {}
        for y in sorted({int(d[:4]) for d in todo}):
            meta = load_year_meta(y)
            keep_by_year[y] = frozenset(whitelist | (set(meta['codes'].tolist()) if meta is not None else set()))
        n_legacy = len([d for d in todo if os.path.exists(_legacy_partition_path(d))])
        bar = st.progress(0, text=f"📥 需要处理 {len(todo)} 个交易日（旧缓存迁移 {n_legacy}，下载 {len(todo) - n_legacy}，{workers} 线程）...")
        buffer, failed = {}, []

        def flush():
            for y, frames in buffer.items():
                if not frames:
                    continue
                merged = merge_year(load_year(y), day_frames=frames, keep_codes=keep_by_year.get(y, whitelist))
                save_year(y, merged)
                del merged
                for d in frames:
                    try:
                        os.remove(_legacy_partition_path(str(d)))
                    except OSError:
                        pass
            buffer.clear()
            gc.collect()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_fetch_market_day, d, token_c, keep_by_year[int(d[:4])]) for d in todo]
            n_buf = 0
            for i, fut in enumerate(as_completed(futures), start=1):
                try:
                    d, frame, _src = fut.result()
                except Exception:
                    d, frame = None, None
                if frame is None:
                    if d is not None:
                        failed.append(d)
                else:
                    buffer.setdefault(int(d[:4]), {})[int(d)] = frame
                    n_buf += 1
                if n_buf >= FLUSH_EVERY_DAYS:
                    flush()
                    n_buf = 0
                bar.progress(i / len(futures), text=f"📥 行情同步中（{workers} 线程）: {i}/{len(futures)}")
            flush()
        bar.empty()
        if os.path.isdir(LEGACY_CACHE_DIR) and not _legacy_dates():
            shutil.rmtree(LEGACY_CACHE_DIR, ignore_errors=True)
        if failed:
            st.warning(f"⚠️ {len(failed)} 天行情未下载成功（限流或当天数据尚未发布），下次运行会自动补下："
                       f"{', '.join(sorted(failed)[:8])}{' ...' if len(failed) > 8 else ''}")
    return valid_dates


class StockSeries:
    """单只股票的前复权日线（numpy 数组，按日期升序）。"""
    __slots__ = ("dates", "yw", "open", "high", "low", "close", "pre_close", "vol", "close_raw")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def __len__(self):
        return len(self.dates)

    def pos(self, date_int):
        i = int(np.searchsorted(self.dates, date_int))
        return i if i < len(self.dates) and self.dates[i] == date_int else -1


class MarketData:
    def __init__(self, stocks, dates, mv, code_col):
        self.stocks = stocks
        self.dates = dates
        self.mv = mv
        self.code_col = code_col
        self.date_row = {int(d): i for i, d in enumerate(dates)}

    def __bool__(self):
        return bool(self.stocks)

    def has_date(self, date):
        return int(date) in self.date_row

    def mv_row(self, date):
        r = self.date_row.get(int(date))
        return None if r is None else self.mv[r]


@st.cache_resource(max_entries=1, ttl=3600 * 12, show_spinner=False)
def _build_market(start_int, end_int, codes, cache_stamp):
    del cache_stamp
    codes_arr = np.array(codes, dtype='<U12')
    date_parts, parts = [], {f: [] for f in STORE_FIELDS}
    for y in _store_years():
        if y < start_int // 10000 or y > end_int // 10000:
            continue
        data = load_year(y)
        if data is None or len(data['dates']) == 0 or len(data['codes']) == 0:
            continue
        dmask = (data['dates'] >= start_int) & (data['dates'] <= end_int)
        if not dmask.any():
            continue
        src = data['codes']
        idx = np.clip(np.searchsorted(src, codes_arr), 0, len(src) - 1)
        present = src[idx] == codes_arr
        date_parts.append(data['dates'][dmask])
        for f in STORE_FIELDS:
            sub = data[f][dmask][:, idx].astype(np.float64)
            sub[:, ~present] = np.nan
            parts[f].append(sub)
        del data
    if not date_parts:
        return MarketData({}, np.array([], dtype=np.int64), np.zeros((0, len(codes))), {})

    dates = np.concatenate(date_parts)
    F = {}
    for f in STORE_FIELDS:
        F[f] = np.vstack(parts[f])
        parts[f] = None
    for f in PRICE_FIELDS:
        F[f] = np.round(F[f], 2)
    iso = pd.to_datetime(pd.Index(dates.astype(str)), format='%Y%m%d').isocalendar()
    yw_all = iso['year'].to_numpy(dtype='int64') * 100 + iso['week'].to_numpy(dtype='int64')

    stocks = {}
    for j, code in enumerate(codes):
        close = F['close'][:, j]
        adj = F['adj_factor'][:, j]
        rows = np.flatnonzero(np.isfinite(close) & np.isfinite(adj))
        if rows.size == 0:
            continue
        a = adj[rows]
        latest = a[-1]

        def q(field):
            raw = F[field][rows, j]
            return raw * a / latest if latest > 0 else raw

        stocks[code] = StockSeries(
            dates=dates[rows], yw=yw_all[rows], open=q('open'), high=q('high'), low=q('low'),
            close=q('close'), pre_close=q('pre_close'), vol=F['vol'][rows, j].copy(), close_raw=close[rows].copy(),
        )
    mv = F['circ_mv']
    del F
    gc.collect()
    return MarketData(stocks, dates, mv, {c: j for j, c in enumerate(codes)})


def load_optimized_market_data(start_date, end_date, token, whitelist_keys):
    token_c = clean_token_str(token)
    valid_dates = sync_market_store(start_date, end_date, token_c, whitelist_keys, DOWNLOAD_WORKERS)
    if not valid_dates:
        return MarketData({}, np.array([], dtype=np.int64), np.zeros((0, 0)), {})
    stamp = tuple((y, os.path.getmtime(_year_path(y)), os.path.getsize(_year_path(y))) for y in _store_years())
    with st.spinner("正在构建前复权行情索引..."):
        return _build_market(int(valid_dates[0]), int(valid_dates[-1]), tuple(sorted(whitelist_keys)), stamp)


# ---- 缓存备份与恢复（Streamlit Cloud 重启会清空磁盘，靠它保住已下载的数据和回测记录）
def build_cache_backup_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as zf:  # npz 已压缩，不再二次压缩
        for y in _store_years():
            zf.write(_year_path(y), arcname=f"market_store/{y}.npz")
        for fn in sorted(os.listdir(".")):
            if re.fullmatch(r"skdj_v(1[5-9]|2[0-9])_[0-9a-f]{8}_(trades|weeks)\.csv", fn) or \
                    re.fullmatch(r"skdj_track_(weeks|stocks|pool)\.csv", fn):
                zf.write(fn, arcname=f"records/{fn}")
    return buf.getvalue()


def restore_cache_backup(file_bytes):
    """行情年文件：本地没有、或备份覆盖的日期更多时才替换；回测记录：本地没有才恢复。"""
    installed, skipped, records = [], [], []
    track_backup = {}
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in zf.namelist():
            m = re.fullmatch(r"market_store/(\d{4})\.npz", name)
            if m:
                year = int(m.group(1))
                tmp_path = _year_path(year) + ".restore"
                os.makedirs(STORE_DIR, exist_ok=True)
                with open(tmp_path, "wb") as fh:
                    fh.write(zf.read(name))
                try:
                    with np.load(tmp_path, allow_pickle=False) as z:
                        new_n = len(z['dates'])
                        shape_ok = all(z[f].shape == (len(z['dates']), len(z['codes'])) for f in STORE_FIELDS)
                    if not shape_ok:
                        raise ValueError
                except Exception:
                    os.remove(tmp_path)
                    skipped.append(f"{year}(文件损坏)")
                    continue
                meta = load_year_meta(year)
                if meta is None or new_n > len(meta['dates']):
                    os.replace(tmp_path, _year_path(year))
                    installed.append(year)
                else:
                    os.remove(tmp_path)
                    skipped.append(f"{year}(本地更全)")
                continue
            m = re.fullmatch(r"records/(skdj_v(?:1[5-9]|2[0-9])_[0-9a-f]{8}_(?:trades|weeks)\.csv|skdj_track_(?:weeks|stocks|pool)\.csv)", name)
            if m and not os.path.exists(m.group(1)):
                with open(m.group(1), "wb") as fh:
                    fh.write(zf.read(name))
                records.append(m.group(1))
            elif m and m.group(1).startswith("skdj_track_"):
                track_backup[m.group(1)] = zf.read(name)
    if len(track_backup) == 3:
        merged = merge_track_backup(track_backup)
        if merged:
            records.append(f"跟踪记录合并 {merged} 周")
    return installed, skipped, records


def merge_track_backup(track_backup):
    """本地已有跟踪记录时：同一信号日保留“最早记录”的那一份（原始前瞻记录优先），两边独有的周都保留。"""
    def read_bytes(b):
        df = pd.read_csv(io.BytesIO(b), encoding="utf-8-sig", dtype={"Signal_Date": str, "ts_code": str})
        df["Signal_Date"] = df["Signal_Date"].astype(str).str.replace(r"\.0$", "", regex=True)
        return df
    bk = {k: read_bytes(v) for k, v in track_backup.items()}
    lc = {k: _read_csv_safely(k) for k in track_backup}
    bw, lw = bk[TRACK_WEEKS_FILE], lc[TRACK_WEEKS_FILE]
    if bw.empty:
        return 0
    use_backup = set()
    local_rec = dict(zip(lw["Signal_Date"], lw["Recorded_At"].astype(str))) if not lw.empty else {}
    for d, rec in zip(bw["Signal_Date"], bw["Recorded_At"].astype(str)):
        if d not in local_rec or rec < local_rec[d]:
            use_backup.add(d)
    if not use_backup:
        return 0
    for fname in (TRACK_WEEKS_FILE, TRACK_STOCKS_FILE, TRACK_POOL_FILE):
        local = lc[fname]
        keep_local = local[~local["Signal_Date"].isin(use_backup)] if not local.empty else local
        from_bk = bk[fname][bk[fname]["Signal_Date"].isin(use_backup)]
        combined = pd.concat([keep_local, from_bk], ignore_index=True, sort=False).sort_values("Signal_Date", kind="mergesort")
        with _file_lock(fname):
            _atomic_write_csv(combined, fname)
    return len(use_backup)


# ---------------------------
# 周线指标
# ---------------------------
def build_weekly_arrays(s, upto=None):
    """把日线合成为 ISO 周线并计算 SKDJ（同花顺公式，N=6, M=3）。所有指标只用过去数据。"""
    if s is None or len(s) == 0:
        return None
    m = len(s) if upto is None else int(upto) + 1
    if m <= 0:
        return None
    yw = s.yw[:m]
    starts = np.r_[0, np.flatnonzero(np.diff(yw)) + 1]
    ends = np.r_[starts[1:] - 1, m - 1]
    wk = pd.DataFrame({
        'date': s.dates[ends].astype(np.int64),
        'high': np.fmax.reduceat(s.high[:m], starts),
        'low': np.fmin.reduceat(s.low[:m], starts),
        'close': s.close[ends],
        'vol': np.add.reduceat(np.nan_to_num(s.vol[:m], nan=0.0), starts),
    })

    lowv = wk['low'].rolling(SKDJ_N).min()
    highv = wk['high'].rolling(SKDJ_N).max()
    diff = (highv - lowv).replace(0, 0.001)
    rsv = ((wk['close'] - lowv) / diff * 100).ewm(span=SKDJ_M, adjust=False).mean()
    k = rsv.ewm(span=SKDJ_M, adjust=False).mean()
    d = k.rolling(SKDJ_M).mean()
    return {
        'date': wk['date'].to_numpy(dtype='int64'),
        'close': wk['close'].to_numpy(dtype=float),
        'vol': wk['vol'].to_numpy(dtype=float),
        'k': k.to_numpy(dtype=float),
        'd': d.to_numpy(dtype=float),
        'ma5_vol': wk['vol'].shift(1).rolling(5).mean().to_numpy(dtype=float),
        'ma20': wk['close'].rolling(20).mean().to_numpy(dtype=float),
    }


def get_weekly_view(ts_code, date_int, s, pos, weekly_cache, is_week_end):
    """周末日期用整段缓存（指标因果，不含未来）；周中日期截断后现算，避免用到本周后几天。"""
    if is_week_end:
        wa = weekly_cache.get(ts_code)
        if wa is None:
            wa = build_weekly_arrays(s)
            weekly_cache[ts_code] = wa
        if wa is None:
            return None, -1
        wpos = int(np.searchsorted(wa['date'], date_int, side='right')) - 1
        if wpos < 0 or wa['date'][wpos] != date_int:
            return None, -1
        return wa, wpos
    wa = build_weekly_arrays(s, upto=pos)
    if wa is None:
        return None, -1
    return wa, len(wa['date']) - 1


def score_oversold(close, ma20, recent_k_min, weeks_under, k, vol_ratio):
    """原 V14.5 打分规则，A/B 级沿用。"""
    score = 20.0 if close >= ma20 else -5.0
    if 22.0 <= recent_k_min <= 25.0:
        score += 30.0
    elif 15.0 <= recent_k_min < 22.0:
        score += 15.0
    elif 5.0 <= recent_k_min < 15.0:
        score -= 10.0
    else:
        score -= 25.0
    if 1 <= weeks_under <= 2:
        score += 30.0
    elif 3 <= weeks_under <= 5:
        score += 15.0
    elif 6 <= weeks_under <= 9:
        score -= 5.0
    else:
        score -= 20.0
    if 25.0 < k <= 32.0:
        score += 10.0
    elif k > 38.0:
        score -= 10.0
    if 1.0 <= vol_ratio <= 2.5:
        score += 10.0
    elif vol_ratio > 4.0:
        score -= 15.0
    return score


def score_pullback(ext, slope, k_min6, vol_ratio):
    score = 0.0
    if 0 <= ext <= 0.10:
        score += 20.0
    elif ext <= 0.18:
        score += 10.0
    if slope >= 0.03:
        score += 15.0
    elif slope > 0:
        score += 5.0
    if 35 <= k_min6 <= 50:
        score += 15.0
    elif 25 <= k_min6 <= 60:
        score += 5.0
    if 1.0 <= vol_ratio <= 2.5:
        score += 10.0
    elif vol_ratio > 4.0:
        score -= 15.0
    return score


def evaluate_signal(wa, i):
    """
    A 标准上穿25：上周K≤25，本周K>25，且K>D（原信号）
    B 低位金叉：本周K上穿D，且K≤30（未满足A）
    C 趋势回踩金叉：周收盘在20周均线上方且均线上行，近12周K曾≥70，
      近6周K回落到25~60，本周K重新上穿D，K≤70，收盘偏离均线不超过25%
    """
    if wa is None or i < SKDJ_N + 15:
        return None
    k_arr, d_arr = wa['k'], wa['d']
    k, pk, d, pdv = k_arr[i], k_arr[i - 1], d_arr[i], d_arr[i - 1]
    if not np.all(np.isfinite([k, pk, d, pdv])):
        return None

    close = wa['close'][i]
    ma20 = wa['ma20'][i] if np.isfinite(wa['ma20'][i]) else close
    ma20_prev4 = wa['ma20'][i - 4]
    hist14 = k_arr[i - 14:i]
    hist14 = hist14[np.isfinite(hist14)]
    hist6 = k_arr[i - 6:i]
    hist12 = k_arr[i - 12:i]
    if hist14.size == 0 or not np.isfinite(hist6).any() or not np.isfinite(hist12).any():
        return None
    recent_k_min = float(hist14.min())
    weeks_under = int((hist14 < 25.0).sum())
    k_min6 = float(np.nanmin(hist6))
    k_max12 = float(np.nanmax(hist12))
    mv5 = wa['ma5_vol'][i]
    vol_ratio = float(wa['vol'][i] / mv5) if (np.isfinite(mv5) and mv5 > 0) else 1.0
    ext = close / ma20 - 1 if ma20 > 0 else 0.0
    slope = ma20 / ma20_prev4 - 1 if (np.isfinite(ma20_prev4) and ma20_prev4 > 0) else np.nan
    golden = (pk <= pdv) and (k > d)

    if pk <= 25.0 < k and k > d:
        tier = "A"
        score = score_oversold(close, ma20, recent_k_min, weeks_under, k, vol_ratio)
    elif golden and k <= 30.0:
        tier = "B"
        score = score_oversold(close, ma20, recent_k_min, weeks_under, k, vol_ratio)
    elif (golden and close >= ma20 and np.isfinite(slope) and slope > 0
          and 25.0 <= k_min6 <= 60.0 and k_max12 >= 70.0 and k <= 70.0 and ext <= 0.25):
        tier = "C"
        score = score_pullback(ext, slope, k_min6, vol_ratio)
    else:
        return None

    return {
        'tier': tier, 'score': round(score, 1), 'k': round(float(k), 2), 'd': round(float(d), 2),
        'recent_k_min': round(recent_k_min, 2), 'weeks_under': weeks_under,
        'signal_close': float(close), 'vol_ratio': round(vol_ratio, 2),
        'trend_type': "均线上方" if close >= ma20 else "均线下方(超跌)",
        'ma20_ext_pct': round(ext * 100, 1),
    }


# ---------------------------
# 买入可行性与固定持有指标（入选、A候选、股票池三组共用同一套规则）
# ---------------------------
def entry_block_reason(ts_code, next_open, next_high, next_low, signal_close):
    """返回 (剔除原因或 None, 跳空幅度%)。与 V14.5 的开盘剔除规则一致。"""
    if not (pd.notna(next_open) and next_open > 0 and pd.notna(signal_close) and signal_close > 0):
        return "无有效开盘价", np.nan
    is_20cm = ts_code.startswith(('300', '301', '688', '689'))
    limit_rate_pct = 19.0 if is_20cm else 9.5
    gap_pct = (next_open - signal_close) / signal_close * 100.0
    if (next_open == next_high == next_low) and gap_pct >= limit_rate_pct:
        return f"一字板无法买入(剔除: {round(gap_pct, 1)}%)", gap_pct
    if is_20cm and gap_pct > 8.0:
        return f"双创高开过大(剔除: {round(gap_pct, 2)}%)", gap_pct
    if not is_20cm and gap_pct > 5.0:
        return f"主板高开过大(剔除: {round(gap_pct, 2)}%)", gap_pct
    if gap_pct < -4.0:
        return f"恶劣低开(剔除: {round(gap_pct, 2)}%)", gap_pct
    return None, gap_pct


def forward_metrics(ts_code, s, pos):
    """次日开盘买入、不设止损的固定持有表现。None=无法买入；complete=False=后续数据不足60个交易日。"""
    n = len(s)
    b = pos + 1
    if b >= n:
        return {'complete': False}
    reason, _ = entry_block_reason(ts_code, s.open[b], s.high[b], s.low[b], s.close[pos])
    if reason:
        return None
    last = FWD_HORIZONS["W12"]
    if b + last - 1 >= n:
        return {'complete': False}
    buy = s.open[b]
    out = {'complete': True}
    for key, days in FWD_HORIZONS.items():
        out[key] = (s.close[b + days - 1] / buy - 1.0) * 100.0
    out['MaxGain'] = (np.nanmax(s.high[b:b + last]) / buy - 1.0) * 100.0
    out['MaxDD'] = (np.nanmin(s.low[b:b + last]) / buy - 1.0) * 100.0
    return out


def summarize_fwd(items):
    comp = [x for x in items if x and x.get('complete')]
    if not comp:
        return {'N': 0, 'W1': np.nan, 'W2': np.nan, 'W4': np.nan, 'W12': np.nan, 'Med12': np.nan, 'MaxGain': np.nan,
                'MaxDD': np.nan, 'Big30': np.nan, 'Bear20': np.nan}
    df = pd.DataFrame(comp)
    return {
        'N': len(df), 'W1': round(df['W1'].mean(), 3), 'W2': round(df['W2'].mean(), 3),
        'W4': round(df['W4'].mean(), 3), 'W12': round(df['W12'].mean(), 3),
        'Med12': round(df['W12'].median(), 3),
        'MaxGain': round(df['MaxGain'].mean(), 3), 'MaxDD': round(df['MaxDD'].mean(), 3),
        'Big30': round((df['MaxGain'] >= BIG_WINNER_PCT).mean() * 100.0, 2),
        'Bear20': round((df['MaxDD'] <= BIG_LOSER_PCT).mean() * 100.0, 2),
    }


# ---------------------------
# 单日扫描（选股与回测共用）
# ---------------------------
def relative_strength(wa, wpos):
    """截至上周的 12 周涨幅（%），跳过最近一周。"""
    if wa is None or wpos - 1 - RS_LOOKBACK_WEEKS < 0:
        return np.nan
    c_end, c_start = wa['close'][wpos - 1], wa['close'][wpos - 1 - RS_LOOKBACK_WEEKS]
    if not (np.isfinite(c_end) and np.isfinite(c_start) and c_start > 0):
        return np.nan
    return (c_end / c_start - 1.0) * 100.0


def rank_sectors(valid, sector_map):
    """valid: [(ts_code, rs)]，rs 为有限值。返回 (前3板块统计, 前3板块成分) 或 (None, None)。"""
    if not sector_map or not valid:
        return None, None
    df = pd.DataFrame([(c, r, sector_map.get(c, (None, None))[0], sector_map.get(c, (None, None))[1])
                       for c, r in valid], columns=['code', 'rs', 'l2', 'l2_name'])
    df = df[df['l2'].notna()]
    if df.empty:
        return None, None
    stats = df.groupby('l2').agg(n=('rs', 'size'), rs=('rs', 'mean'), name=('l2_name', 'first'))
    stats = stats[stats['n'] >= SECTOR_MIN_MEMBERS].sort_values('rs', ascending=False)
    top = stats.head(SECTOR_TOP_K)
    if top.empty:
        return None, None
    return top, df[df['l2'].isin(top.index)]


def build_strength_groups(pool_items, sector_map):
    """pool_items: [(ts_code, fwd, rs, feat)]。返回 (RS组, SEC组, SECRS组, 前3板块名称, 板块内方法组dict)。"""
    valid = [(c, f, r, ft) for c, f, r, ft in pool_items if np.isfinite(r)]
    methods = {key: [] for key, _, _, _ in METHOD_GROUPS if key != "SECRS"}
    if not valid:
        return [], [], [], "", methods
    rs_vals = np.array([r for _, _, r, _ in valid])
    cut = np.percentile(rs_vals, 100.0 - RS_TOP_PCT)
    rs_group = [f for _, f, r, _ in valid if r >= cut]

    sec_group, secrs_group, top_names = [], [], ""
    fwd_map = {c: f for c, f, _, _ in valid}
    feat_map = {c: ft for c, _, _, ft in valid}
    top, in_top = rank_sectors([(c, r) for c, _, r, _ in valid], sector_map)
    if top is not None:
        sec_group = [fwd_map[c] for c in in_top['code']]
        secrs_codes = (in_top.sort_values('rs', ascending=False)
                       .groupby('l2', sort=False).head(SECTOR_STOCKS_EACH)['code'])
        secrs_group = [fwd_map[c] for c in secrs_codes]
        top_names = "、".join(f"{nm}({rs:.0f}%)" for nm, rs in zip(top['name'], top['rs']))
        feat = in_top[['code', 'l2', 'rs']].copy()
        for name in ('mv', 'vol20', 'ret1w'):
            feat[name] = [feat_map[c].get(name, np.nan) for c in feat['code']]
        for key, _, col, desc in METHOD_GROUPS:
            if key == "SECRS":
                continue
            picked = (feat.sort_values([col, 'code'], ascending=[not desc, True], na_position='last', kind='mergesort')
                      .groupby('l2', sort=False).head(SECTOR_STOCKS_EACH))
            picked = picked[picked[col].notna()]
            methods[key] = [fwd_map[c] for c in picked['code']]
    return rs_group, sec_group, secrs_group, top_names, methods


def stock_features(s, pos, wa, wpos, circ_mv):
    """板块内选股方法用到的特征：流通市值、20日收益波动、近1周涨幅。"""
    vol20 = np.nan
    if pos >= 20:
        c = s.close[pos - 20:pos + 1]
        r = c[1:] / c[:-1] - 1.0
        if np.isfinite(r).all():
            vol20 = float(np.std(r, ddof=1))
    ret1w = np.nan
    if wa is not None and wpos >= 1 and wa['close'][wpos - 1] > 0:
        ret1w = float(wa['close'][wpos] / wa['close'][wpos - 1] - 1.0) * 100.0
    return {'mv': circ_mv, 'vol20': vol20, 'ret1w': ret1w}


def iter_pool(date_int, whitelist_keys, market, cfg, funnel):
    """股票池逐层筛选（股价、流通市值、上市≥100天、非一字涨停）。回测与跟踪共用。"""
    mv_row = market.mv_row(date_int)
    for ts_code in whitelist_keys:
        s = market.stocks.get(ts_code)
        if s is None or len(s) == 0:
            continue
        pos = s.pos(date_int)
        if pos < 0:
            continue
        funnel["当日有行情"] += 1

        close_raw = float(s.close_raw[pos])
        if close_raw < cfg['min_price']:
            continue
        funnel["股价达标"] += 1

        col = market.code_col.get(ts_code)
        circ_mv = float(mv_row[col]) if (mv_row is not None and col is not None) else np.nan
        circ_mv = circ_mv / 10000.0 if pd.notna(circ_mv) else np.nan
        if pd.notna(circ_mv):
            if circ_mv < cfg['min_mv'] or circ_mv > cfg['max_mv']:
                continue
        else:
            funnel["市值缺失(未过滤)"] += 1
        funnel["市值达标"] += 1

        if pos < 99:
            continue
        funnel["历史≥100天"] += 1

        high, low, close_q = s.high[pos], s.low[pos], s.close[pos]
        pre = s.pre_close[pos]
        if not (pd.notna(pre) and pre > 0):
            pre = s.close[pos - 1]
        limit_rate = 0.195 if ts_code.startswith(('300', '301', '688', '689')) else 0.095
        if high == low and (close_q - pre) / pre >= limit_rate:
            continue
        funnel["非一字涨停"] += 1
        yield ts_code, s, pos, close_raw, circ_mv


def new_funnel(pool_size):
    return {
        "股票池": pool_size, "当日有行情": 0, "股价达标": 0, "市值达标": 0,
        "历史≥100天": 0, "非一字涨停": 0, "A级": 0, "B级": 0, "C级": 0, "市值缺失(未过滤)": 0,
    }


def scan_date(date, whitelist_keys, market, name_map, weekly_cache, is_week_end, cfg,
              with_benchmark=False, sector_map=None):
    funnel = new_funnel(len(whitelist_keys))
    date = str(date)
    date_int = int(date)

    cands = []
    pool_items, a_fwd = [], []
    breadth_up, breadth_total = 0, 0
    for ts_code, s, pos, close_raw, circ_mv in iter_pool(date_int, whitelist_keys, market, cfg, funnel):
        wa, wpos = get_weekly_view(ts_code, date_int, s, pos, weekly_cache, is_week_end)
        if wa is not None and wpos >= 0 and np.isfinite(wa['ma20'][wpos]):
            breadth_total += 1
            breadth_up += int(wa['close'][wpos] >= wa['ma20'][wpos])

        fwd = forward_metrics(ts_code, s, pos) if with_benchmark else None
        if with_benchmark:
            pool_items.append((ts_code, fwd, relative_strength(wa, wpos), stock_features(s, pos, wa, wpos, circ_mv)))

        sig = evaluate_signal(wa, wpos)
        if not sig:
            continue
        funnel[f"{sig['tier']}级"] += 1
        if sig['tier'] == "A" and with_benchmark:
            a_fwd.append(fwd)
        if sig['tier'] not in cfg['tiers']:
            continue
        rec = {
            'Tier': sig['tier'], 'Tier_Label': TIER_LABEL[sig['tier']],
            'ts_code': ts_code, 'name': name_map.get(ts_code, ts_code),
            'Total_Score': sig['score'], 'SKDJ_K': sig['k'], 'SKDJ_D': sig['d'],
            'K_Min_14W': sig['recent_k_min'], 'Weeks_Under': sig['weeks_under'],
            'Signal_Close': sig['signal_close'], 'Close_Raw': round(close_raw, 2),
            'Trend_Type': sig['trend_type'], 'MA20_Ext (%)': sig['ma20_ext_pct'],
            'vol_ratio': sig['vol_ratio'], 'circ_mv': round(circ_mv, 2) if pd.notna(circ_mv) else np.nan,
        }
        if with_benchmark:
            rec['_fwd'] = fwd
            if fwd and fwd.get('complete'):
                rec.update({'Fwd_W4 (%)': round(fwd['W4'], 2), 'Fwd_W12 (%)': round(fwd['W12'], 2),
                            'Fwd_MaxGain (%)': round(fwd['MaxGain'], 2), 'Fwd_MaxDD (%)': round(fwd['MaxDD'], 2)})
        cands.append(rec)

    bench = {
        'Breadth': round(breadth_up / breadth_total * 100.0, 1) if breadth_total else np.nan,
        'Breadth_Total': breadth_total,
    }
    picks, all_cands = pd.DataFrame(), pd.DataFrame()
    if cands:
        all_cands = pd.DataFrame(cands)
        all_cands['_order'] = all_cands['Tier'].map(TIER_ORDER)
        all_cands = all_cands.sort_values(['_order', 'Total_Score'], ascending=[True, False], kind='mergesort').drop(columns='_order').reset_index(drop=True)
        picks = all_cands.head(int(cfg['top_n'])).copy()
        picks.insert(0, 'Rank', range(1, len(picks) + 1))
        picks['Trade_Date'] = date

    if with_benchmark:
        rs_group, sec_group, secrs_group, top_names, method_groups = build_strength_groups(pool_items, sector_map)
        bench['Top_Sectors'] = top_names
        groups = {
            'Pick': picks['_fwd'].tolist() if not picks.empty else [],
            'A': a_fwd,
            'RS': rs_group,
            'SEC': sec_group,
            'SECRS': secrs_group,
            'Pool': [f for _, f, _, _ in pool_items],
            **method_groups,
        }
        for g, items in groups.items():
            for k, v in summarize_fwd(items).items():
                bench[f'{g}_{k}'] = v
    for frame in (picks, all_cands):
        if not frame.empty and '_fwd' in frame.columns:
            frame.drop(columns='_fwd', inplace=True)
    return picks, all_cands, funnel, bench


# ---------------------------
# 回测出场模拟（规则出场，仅用于参考；固定持有对照见 forward_metrics）
# ---------------------------
def track_future_performance(ts_code, selection_date, signal_close, market, hold_weeks=HOLD_WEEKS):
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    results.update({'Exit_Reason': '持仓中', 'Buy_Price': np.nan, 'Gap_pct (%)': np.nan,
                    'Exit_Date': None, 'Final_Return (%)': np.nan, 'Hold_Days': 0})
    s = market.stocks.get(ts_code)
    if s is None:
        return results
    start = int(np.searchsorted(s.dates, int(selection_date), side='right'))
    n_future = len(s) - start
    if n_future <= 0:
        return results

    buy_price = s.open[start]
    if pd.isna(buy_price) or buy_price <= 0 or not signal_close:
        return results
    reason, gap_pct = entry_block_reason(ts_code, buy_price, s.high[start], s.low[start], signal_close)
    results['Buy_Price'] = round(buy_price, 2)
    results['Gap_pct (%)'] = round(gap_pct, 2) if pd.notna(gap_pct) else np.nan
    if reason:
        results['Exit_Reason'] = reason
        return results

    tier = 0
    peak_price = buy_price
    pending_exit_reason = None
    hard_stop_limit = -0.10
    max_days = hold_weeks * 5

    def close_out(reason_text, ret, date, days, week):
        results['Exit_Reason'] = reason_text
        results['Final_Return (%)'] = round(ret, 2)
        results['Exit_Date'] = date
        results['Hold_Days'] = days
        results[f'Return_W{week} (%)'] = round(ret, 2)

    for i in range(min(n_future, max_days)):
        j = start + i
        day_count = i + 1
        current_week = (day_count - 1) // 5 + 1
        curr_open, curr_close, curr_high, curr_low = s.open[j], s.close[j], s.high[j], s.low[j]
        curr_date = str(int(s.dates[j]))

        if pending_exit_reason is not None and day_count >= 2:
            close_out(pending_exit_reason, (curr_open - buy_price) / buy_price * 100.0, curr_date, day_count, current_week)
            return results

        peak_price = max(peak_price, curr_high)
        peak_profit_pct = (peak_price - buy_price) / buy_price

        if day_count >= 2 and (curr_low - buy_price) / buy_price <= hard_stop_limit:
            ret = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
            close_out("认栽出局(破-10%)", ret, curr_date, day_count, current_week)
            return results

        if tier == 0 and peak_profit_pct >= 0.10:
            tier = 1
        if tier == 1:
            if curr_close <= buy_price * 1.02:
                pending_exit_reason = "保本离场(+2%触发)"
            elif peak_profit_pct >= 0.20:
                tier = 2
        if tier == 2 and (peak_price - curr_close) / peak_price >= 0.15:
            pending_exit_reason = "移动止盈(回撤15%)"

        if day_count == 5 and pending_exit_reason is None:
            w1_ret = (curr_close - buy_price) / buy_price * 100.0
            if w1_ret <= -3.0:
                close_out(f"首周不及预期截断({round(w1_ret, 1)}%)", w1_ret, curr_date, 5, 1)
                return results

        if day_count % 5 == 0:
            results[f'Return_W{current_week} (%)'] = round((curr_close - buy_price) / buy_price * 100.0, 2)

    if n_future >= max_days:
        last_price = s.close[start + max_days - 1]
        close_out(f"{hold_weeks}周期满平仓", (last_price - buy_price) / buy_price * 100.0,
                  str(int(s.dates[start + max_days - 1])), max_days, hold_weeks)
    return results


# ---------------------------
# 板块动量前瞻跟踪
# ---------------------------
def sector_snapshot(date, whitelist_keys, market, sector_map, weekly_cache, is_week_end, name_map):
    """按冻结规则生成某日的前3强板块名单。返回 (周汇总dict, 成分股DataFrame, 股票池代码列表)。"""
    date = str(date)
    date_int = int(date)
    funnel = new_funnel(len(whitelist_keys))
    pool, info = [], {}
    breadth_up, breadth_total = 0, 0
    for ts_code, s, pos, close_raw, circ_mv in iter_pool(date_int, whitelist_keys, market, TRACK_CFG, funnel):
        wa, wpos = get_weekly_view(ts_code, date_int, s, pos, weekly_cache, is_week_end)
        if wa is not None and wpos >= 0 and np.isfinite(wa['ma20'][wpos]):
            breadth_total += 1
            breadth_up += int(wa['close'][wpos] >= wa['ma20'][wpos])
        pool.append((ts_code, relative_strength(wa, wpos)))
        info[ts_code] = (close_raw, circ_mv)

    top, in_top = rank_sectors([(c, r) for c, r in pool if np.isfinite(r)], sector_map)
    rows, top_names, n_top2 = [], "", 0
    if top is not None:
        rank_of = {l2: i + 1 for i, l2 in enumerate(top.index)}
        top2 = set(in_top.sort_values('rs', ascending=False).groupby('l2', sort=False)
                   .head(SECTOR_STOCKS_EACH)['code'])
        n_top2 = len(top2)
        top_names = "、".join(f"{nm}({rs:.0f}%)" for nm, rs in zip(top['name'], top['rs']))
        for r in in_top.itertuples(index=False):
            close_raw, circ_mv = info[r.code]
            rows.append({
                'Signal_Date': date, 'Sector_Rank': rank_of[r.l2], 'Sector': r.l2_name,
                'Sector_RS (%)': round(float(top.loc[r.l2, 'rs']), 2), 'Sector_Members': int(top.loc[r.l2, 'n']),
                'ts_code': r.code, 'name': name_map.get(r.code, r.code), 'RS_12W (%)': round(float(r.rs), 2),
                'Top2': bool(r.code in top2), 'Close_Raw': round(close_raw, 2),
                'circ_mv': round(circ_mv, 2) if pd.notna(circ_mv) else np.nan,
            })
    stocks = pd.DataFrame(rows)
    if not stocks.empty:
        stocks = stocks.sort_values(['Sector_Rank', 'RS_12W (%)'], ascending=[True, False], kind='mergesort').reset_index(drop=True)
    meta = {
        'Signal_Date': date,
        'Breadth': round(breadth_up / breadth_total * 100.0, 1) if breadth_total else np.nan,
        'Pool_N': len(pool), 'Top_Sectors': top_names, 'SEC_N': len(rows), 'SECRS_N': n_top2,
    }
    return meta, stocks, [c for c, _ in pool]


def short_returns(ts_code, s, pos):
    """次日开盘买入，1/2/4 周分别计算（与回测 forward_metrics 同一口径）。"""
    b = pos + 1
    n = len(s)
    if b >= n:
        return {'status': '待买入'}
    reason, _ = entry_block_reason(ts_code, s.open[b], s.high[b], s.low[b], s.close[pos])
    if reason:
        return {'status': reason}
    out = {'status': '已买入', 'buy': float(s.open[b])}
    for key, days in TRACK_HORIZONS.items():
        idx = b + days - 1
        out[key] = (s.close[idx] / s.open[b] - 1.0) * 100.0 if idx < n else np.nan
    return out


def settle_week(date, stock_codes, top2_codes, pool_codes, market):
    """按冻结名单计算各持有期收益。全市场交易日数达到持有期才结算该持有期。"""
    date_int = int(date)
    n_after = int((market.dates > date_int).sum())
    stock_set, top2_set, pool_set = set(stock_codes), set(top2_codes), set(pool_codes)
    per_stock = {}
    rets = {g: {k: [] for k in TRACK_HORIZONS} for g in ('Pool', 'SEC', 'SECRS')}
    for code in pool_set | stock_set:
        s = market.stocks.get(code)
        pos = s.pos(date_int) if s is not None else -1
        if pos < 0:
            if code in stock_set:
                per_stock[code] = {'status': '无行情'}
            continue
        r = short_returns(code, s, pos)
        if code in stock_set:
            per_stock[code] = r
        if r.get('status') != '已买入':
            continue
        for k in TRACK_HORIZONS:
            v = r.get(k, np.nan)
            if not np.isfinite(v):
                continue
            if code in pool_set:
                rets['Pool'][k].append(v)
            if code in stock_set:
                rets['SEC'][k].append(v)
            if code in top2_set:
                rets['SECRS'][k].append(v)
    row = {'Trade_Days_After': n_after}
    for k, days in TRACK_HORIZONS.items():
        settled = n_after >= days
        row[f'Settled_{k}'] = settled
        means = {g: (float(np.mean(rets[g][k])) if (settled and rets[g][k]) else np.nan) for g in rets}
        for g in rets:
            row[f'{g}_{k} (%)'] = round(means[g], 3) if np.isfinite(means[g]) else np.nan
        for g in ('SEC', 'SECRS'):
            ok = np.isfinite(means[g]) and np.isfinite(means['Pool'])
            row[f'{g}_Excess_{k} (%)'] = round(means[g] - means['Pool'], 3) if ok else np.nan
    if n_after == 0:
        row['Status'] = '待买入'
    elif row['Settled_W4']:
        row['Status'] = '已全部结算'
    elif row['Settled_W2']:
        row['Status'] = '2周已结算'
    elif row['Settled_W1']:
        row['Status'] = '1周已结算'
    else:
        row['Status'] = '持有中'
    return row, per_stock


# ---------------------------
# 侧边栏
# ---------------------------
with st.sidebar:
    st.header("⚙️ 设置")
    MODE = st.radio("运行模式", ["🧭 板块动量跟踪", "📌 SKDJ今日选股", "📊 历史回测"], index=0)
    is_tracking_mode = MODE.startswith("🧭")
    is_picking_mode = MODE.startswith("📌")
    is_backtest_mode = MODE.startswith("📊")

    if is_tracking_mode:
        backtest_date_end = datetime.now().date()
        BACKTEST_WEEKS = 0
        st.caption(f"跟踪规则已冻结：从 {TRACK_START[:4]}-{TRACK_START[4:6]}-{TRACK_START[6:]} 这一周起，"
                   "每周末按申万二级板块的12周涨幅（跳过最近一周）取前3强，记录全部成分股，"
                   "次日开盘买入后分别在1/2/4周结算相对股票池的超额。股票池门槛固定为股价≥10元、流通市值50~1000亿。")
    elif is_picking_mode:
        backtest_date_end = st.date_input("选股日期（默认今天）", value=datetime.now().date())
        BACKTEST_WEEKS = 0
    else:
        BACKTEST_WEEKS = int(st.number_input("回测周数（52≈1年）", value=52, min_value=4, max_value=520, step=4))
        backtest_date_end = st.date_input("回测截止日期", value=datetime.now().date())
        st.caption("报告区间（只影响下方报告显示，不影响回测扫描）")
        REPORT_START = st.date_input("报告起始日期", value=datetime(2015, 1, 1).date())
        REPORT_END = st.date_input("报告截止日期", value=datetime.now().date())
        OOS_SPLIT = st.date_input("样本外分界日期（总览表分两段显示）", value=datetime(2022, 8, 12).date())

    if is_tracking_mode:
        TOP_N, USE_B, USE_C = 3, True, True
        MIN_PRICE, MIN_MV, MAX_MV = TRACK_CFG["min_price"], TRACK_CFG["min_mv"], TRACK_CFG["max_mv"]
    else:
        TOP_N = int(st.number_input("每周选股数量", value=3, min_value=1, max_value=10, step=1))

        st.markdown("---")
        st.subheader("补位层级")
        st.caption("A 级（周K上穿25）始终启用。A 级不足名额时，依次用 B、C 补位。")
        USE_B = st.checkbox("B 低位金叉（K≤30 时 K 上穿 D）", value=True)
        USE_C = st.checkbox("C 趋势回踩金叉（强势股回调后周线再金叉）", value=True)

        st.markdown("---")
        st.subheader("💰 股票池门槛")
        MIN_PRICE = float(st.number_input("最低股价 (元，未复权)", value=10.0))
        col1, col2 = st.columns(2)
        MIN_MV = float(col1.number_input("最小流通市值(亿)", value=50.0))
        MAX_MV = float(col2.number_input("最大流通市值(亿)", value=1000.0))

    st.markdown("---")
    secret_token = ""
    try:
        secret_token = st.secrets.get("TUSHARE_TOKEN", "")
    except Exception:
        secret_token = ""
    TS_TOKEN_INPUT = st.text_input("🔑 Tushare Token", value=secret_token, type="password")
    DOWNLOAD_WORKERS = int(st.number_input("行情下载并发线程数", value=4, min_value=1, max_value=8, step=1,
                                           help="积分较低、频繁限流时调小"))

    tiers_enabled = ["A"] + (["B"] if USE_B else []) + (["C"] if USE_C else [])
    CFG = {"v": LOGIC_VERSION, "top_n": TOP_N, "tiers": tiers_enabled,
           "min_price": MIN_PRICE, "min_mv": MIN_MV, "max_mv": MAX_MV}
    CFG_SIG = hashlib.md5(json.dumps(CFG, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    TRADES_FILE = f"skdj_v22_{CFG_SIG}_trades.csv"
    WEEKS_FILE = f"skdj_v22_{CFG_SIG}_weeks.csv"

    with st.expander("🧹 缓存与记录维护"):
        st.caption(f"当前参数组编号：{CFG_SIG}（改任何参数都会自动使用独立的回测记录）")
        if st.button("清除【当前参数组】回测记录"):
            for p in (TRADES_FILE, WEEKS_FILE):
                for suffix in ("", ".bak", ".lock"):
                    if os.path.exists(p + suffix):
                        os.remove(p + suffix)
            st.success("当前参数组的回测记录已清除。")
        if st.button("清空行情缓存（需重新下载）"):
            for cache_dir in (STORE_DIR, LEGACY_CACHE_DIR):
                if os.path.isdir(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("行情缓存已清理。")

    with st.expander("💾 缓存备份与恢复（重启后免重新下载）", expanded=False):
        info = store_summary()
        if info['n_dates']:
            st.caption(f"行情仓库：{info['first']} ~ {info['last']}，{info['n_dates']} 个交易日，"
                       f"{info['n_codes']} 只股票，{info['size_mb']:.1f} MB")
        else:
            st.caption("行情仓库目前为空。")
        st.caption("Streamlit Cloud 重启或长时间休眠后磁盘会被清空。下载完大段行情或跑完回测后，"
                   "生成一次备份存到电脑；重启后上传即可恢复行情和回测记录。")
        if st.button("生成备份文件"):
            with st.spinner("正在打包..."):
                st.session_state['cache_backup'] = build_cache_backup_zip()
                st.session_state['cache_backup_name'] = f"skdj_cache_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        if st.session_state.get('cache_backup'):
            st.download_button(
                label=f"⬇️ 下载备份 ZIP（{len(st.session_state['cache_backup']) / 1024 / 1024:.1f} MB）",
                data=st.session_state['cache_backup'],
                file_name=st.session_state.get('cache_backup_name', 'skdj_cache_backup.zip'),
                mime="application/zip", key="download_cache_backup",
            )
        uploaded_backup = st.file_uploader("上传备份 ZIP 恢复", type=["zip"], key="cache_restore_uploader")
        if uploaded_backup is not None:
            upload_id = f"{uploaded_backup.name}-{uploaded_backup.size}"
            if st.session_state.get('restored_upload_id') != upload_id:
                try:
                    installed, skipped, records = restore_cache_backup(uploaded_backup.getvalue())
                    st.session_state['restored_upload_id'] = upload_id
                    st.cache_resource.clear()
                    st.success(f"已恢复行情年份：{installed or '无'}；回测记录 {len(records)} 个。"
                               + (f" 跳过：{'、'.join(skipped)}" if skipped else ""))
                except zipfile.BadZipFile:
                    st.error("上传的不是有效的备份 ZIP。")

token_clean = clean_token_str(TS_TOKEN_INPUT)


# ---------------------------
# 运行流程
# ---------------------------
def load_calendar(pro, end_date_obj, lookback_days):
    end_str = end_date_obj.strftime("%Y%m%d")
    today_str = datetime.now().strftime("%Y%m%d")
    start_cal = (end_date_obj - timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_ext = (end_date_obj + timedelta(days=15)).strftime("%Y%m%d")
    cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_cal, end_date=end_ext)
    if cal_raw.empty:
        return [], set()
    all_days = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date')['cal_date'].astype(str).tolist()
    iso = pd.to_datetime(pd.Index(all_days), format='%Y%m%d').isocalendar()
    td = pd.DataFrame({'d': all_days, 'yw': iso['year'].to_numpy(dtype='int64') * 100 + iso['week'].to_numpy(dtype='int64')})
    week_end_set = set(td.groupby('yw')['d'].max().tolist())
    trade_days = [d for d in all_days if d <= end_str and d <= today_str]
    return trade_days, week_end_set


def show_df(df):
    st.dataframe(df, width='stretch', hide_index=True)


PICK_COLS_CN = {
    'Rank': '排名', 'Tier_Label': '层级', 'name': '名称', 'ts_code': '代码', 'Total_Score': '评分',
    'SKDJ_K': '周K', 'SKDJ_D': '周D', 'K_Min_14W': '前14周K最低', 'Weeks_Under': 'K<25周数',
    'Close_Raw': '收盘价', 'MA20_Ext (%)': '偏离20周线%', 'vol_ratio': '周量比', 'circ_mv': '流通市值(亿)',
    'Trend_Type': '位置',
}


def run_picking(pro, whitelist_keys, name_map):
    trade_days, week_end_set = load_calendar(pro, backtest_date_end, 60)
    if not trade_days:
        st.error("❌ 未获取到交易日历。")
        return
    last_day = trade_days[-1]
    fetch_start = (datetime.strptime(last_day, "%Y%m%d") - timedelta(days=300)).strftime("%Y%m%d")
    market = load_optimized_market_data(fetch_start, last_day, token_clean, whitelist_keys)
    if not market:
        st.warning("⚠️ 未能加载到行情数据，请重试。")
        return

    scan_day = next((d for d in reversed(trade_days) if market.has_date(d)), None)
    if scan_day is None:
        st.error("❌ 最近交易日都没有可用行情。")
        return
    if scan_day != last_day:
        st.warning(f"⚠️ {last_day} 的日线 Tushare 尚未发布（通常收盘后 15:30~17:00 更新），本次改用 **{scan_day}** 的数据选股。")
    is_week_end = scan_day in week_end_set
    if not is_week_end:
        st.info(f"ℹ️ {scan_day} 不是本周最后一个交易日，本周K线尚未收完，结果为临时值，周五收盘后可能变化。")

    picks, all_cands, funnel, bench = scan_date(scan_day, whitelist_keys, market,
                                                name_map, {}, is_week_end, CFG)

    st.subheader(f"🎯 选股结果 [{scan_day}]")
    if pd.notna(bench.get('Breadth')):
        st.metric("市场宽度：过筛股票中周线站上20周线的比例（仅供参考，不参与选股）", f"{bench['Breadth']:.0f}%")
    if picks.empty:
        st.error("本次没有任何股票满足已启用的层级。请看下方漏斗，找出卡在哪一步。")
    else:
        cols = [c for c in PICK_COLS_CN if c in picks.columns]
        show_df(picks[cols].rename(columns=PICK_COLS_CN))
        tier_counts = picks['Tier'].value_counts().to_dict()
        if tier_counts.get("A", 0) < len(picks):
            st.caption("本周 A 级不足名额，已用补位层级填充。B/C 级的历史表现请以回测中的“分层级统计”为准。")

    st.markdown("#### 🔍 筛选漏斗")
    show_df(pd.DataFrame([{"步骤": k, "数量": v} for k, v in funnel.items()]))
    if not all_cands.empty and len(all_cands) > len(picks):
        with st.expander(f"查看全部 {len(all_cands)} 只候选"):
            cols = [c for c in PICK_COLS_CN if c in all_cands.columns and c != 'Rank']
            show_df(all_cands[cols].rename(columns=PICK_COLS_CN))
    st.success("选股完成（选股模式不写入回测记录）。")


def _truthy(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def run_backtest(pro, whitelist_keys, name_map, sector_map=None):
    trade_days, week_end_set = load_calendar(pro, backtest_date_end, BACKTEST_WEEKS * 7 + 60)
    if not trade_days:
        st.error("❌ 未获取到交易日历。")
        return
    trade_day_set = set(trade_days)
    week_ends = sorted(d for d in week_end_set if d in trade_day_set)
    target_dates = week_ends[-BACKTEST_WEEKS:]

    weeks_log = _read_csv_safely(WEEKS_FILE)
    settled = set()
    if not weeks_log.empty and 'Bench_Complete' in weeks_log.columns:
        settled = set(weeks_log.loc[_truthy(weeks_log['Bench_Complete']), 'Trade_Date'].astype(str))
    dates_to_run = [d for d in target_dates if d not in settled]
    if not dates_to_run:
        st.success("🎉 该区间已全部回测并结算完毕。")
        return

    n_new = len([d for d in dates_to_run if weeks_log.empty or d not in set(weeks_log['Trade_Date'].astype(str))])
    st.info(f"本次扫描 {len(dates_to_run)} 周（新周 {n_new}，未满12周需重新结算 {len(dates_to_run) - n_new}）。")

    today_str = datetime.now().strftime("%Y%m%d")
    fetch_start = (datetime.strptime(min(dates_to_run), "%Y%m%d") - timedelta(days=300)).strftime("%Y%m%d")
    fetch_end = min(today_str, (datetime.strptime(max(dates_to_run), "%Y%m%d") + timedelta(days=130)).strftime("%Y%m%d"))

    market = load_optimized_market_data(fetch_start, fetch_end, token_clean, whitelist_keys)
    if not market:
        st.warning("⚠️ 未能加载到行情数据，请重试。")
        return

    weekly_cache = {}
    skipped = 0
    bar = st.progress(0, text="回测扫描中...")
    for i, date in enumerate(dates_to_run):
        if not market.has_date(date):
            skipped += 1
            continue
        picks, _, funnel, bench = scan_date(date, whitelist_keys, market,
                                            name_map, weekly_cache, True, CFG,
                                            with_benchmark=True, sector_map=sector_map)
        has_open = False
        if not picks.empty:
            future = [track_future_performance(r.ts_code, date, r.Signal_Close, market)
                      for r in picks.itertuples(index=False)]
            picks = pd.concat([picks.reset_index(drop=True), pd.DataFrame(future)], axis=1)
            has_open = bool((picks['Exit_Reason'].astype(str) == '持仓中').any())
        _replace_date_rows(TRADES_FILE, date, picks, ["Trade_Date", "Rank"])

        week_row = {
            'Trade_Date': date, 'A_Count': funnel["A级"], 'B_Count': funnel["B级"], 'C_Count': funnel["C级"],
            'Pool_After_Filter': funnel["非一字涨停"], 'Picks': len(picks),
            'Pick_Tiers': "".join(picks['Tier'].tolist()) if not picks.empty else "",
            'Pick_Names': "、".join(picks['name'].astype(str).tolist()) if not picks.empty else "",
        }
        week_row.update(bench)
        week_row['Bench_Complete'] = bool(bench.get('Pool_N', 0) > 0 and not has_open)
        _replace_date_rows(WEEKS_FILE, date, pd.DataFrame([week_row]), ["Trade_Date"])
        bar.progress((i + 1) / len(dates_to_run),
                     text=f"扫描 {date}：A{funnel['A级']} / B{funnel['B级']} / C{funnel['C级']}，入选 {len(picks)} 只")
    bar.empty()
    if skipped:
        st.warning(f"有 {skipped} 个周末交易日缺少行情数据被跳过，下次运行会自动补扫。")
    st.success("🎉 回测更新完毕，请查看下方报告。")


def _bj_now():
    return datetime.now(timezone(timedelta(hours=8)))


def show_track_list(stocks, title):
    st.markdown(title)
    if stocks is None or stocks.empty:
        st.info("暂无名单。")
        return
    sec = (stocks.groupby(['Sector_Rank', 'Sector'], as_index=False)
           .agg(板块12周涨幅=('Sector_RS (%)', 'first'), 入选成分股=('ts_code', 'size')))
    show_df(sec.rename(columns={'Sector_Rank': '排名', 'Sector': '板块'}))
    cols = {'Sector_Rank': '板块排名', 'Sector': '板块', 'name': '名称', 'ts_code': '代码',
            'RS_12W (%)': '个股12周涨幅%', 'Close_Raw': '收盘价', 'circ_mv': '流通市值(亿)', 'Top2': '板块最强2只'}
    extra = {'Entry_Status': '买入状态', 'Ret_W1 (%)': '1周%', 'Ret_W2 (%)': '2周%', 'Ret_W4 (%)': '4周%'}
    show = stocks.copy()
    if 'Top2' in show.columns:
        show['Top2'] = np.where(_truthy(show['Top2']), '是', '')
    use = {**cols, **{k: v for k, v in extra.items() if k in show.columns}}
    with st.expander(f"查看全部 {len(show)} 只成分股"):
        show_df(show[[c for c in use if c in show.columns]].rename(columns=use))


def run_tracking(pro, whitelist_keys, name_map, sector_map):
    today = datetime.now().date()
    start_dt = datetime.strptime(TRACK_START, "%Y%m%d").date()
    lookback = max(60, (today - start_dt).days + 60)
    cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE',
                                start_date=(today - timedelta(days=lookback)).strftime("%Y%m%d"),
                                end_date=(today + timedelta(days=20)).strftime("%Y%m%d"))
    if cal_raw.empty:
        st.error("❌ 未获取到交易日历。")
        return
    open_days = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date')['cal_date'].astype(str).tolist()
    iso = pd.to_datetime(pd.Index(open_days), format='%Y%m%d').isocalendar()
    yw = iso['year'].to_numpy(dtype='int64') * 100 + iso['week'].to_numpy(dtype='int64')
    week_end_set = set(pd.Series(open_days).groupby(yw).max().tolist())
    today_str = today.strftime("%Y%m%d")
    trade_days = [d for d in open_days if d <= today_str]
    if not trade_days:
        st.error("❌ 没有可用交易日。")
        return

    weeks = _read_csv_safely(TRACK_WEEKS_FILE)
    done = set()
    if not weeks.empty and 'Settled_W4' in weeks.columns:
        done = set(weeks.loc[_truthy(weeks['Settled_W4']), 'Signal_Date'].astype(str))
    signal_dates = [d for d in trade_days if d in week_end_set and d >= TRACK_START]
    todo = [d for d in signal_dates if d not in done]

    last_day = trade_days[-1]
    fetch_start = (datetime.strptime(min(todo + [last_day]), "%Y%m%d") - timedelta(days=300)).strftime("%Y%m%d")
    market = load_optimized_market_data(fetch_start, last_day, token_clean, whitelist_keys)
    if not market:
        st.warning("⚠️ 未能加载到行情数据，请重试。")
        return

    weekly_cache = {}
    n_new, n_update, n_wait = 0, 0, 0
    for d in todo:
        if not market.has_date(d):
            n_wait += 1
            continue
        weeks = _read_csv_safely(TRACK_WEEKS_FILE)
        existing_row = weeks[weeks['Signal_Date'].astype(str) == d] if not weeks.empty else pd.DataFrame()
        if existing_row.empty:
            meta, stocks, pool_codes = sector_snapshot(d, whitelist_keys, market, sector_map, weekly_cache, True, name_map)
            if stocks.empty:
                st.warning(f"⚠️ {d} 未能生成板块名单（申万行业映射不足），本周暂不记录，下次运行重试。")
                continue
            now_bj = _bj_now()
            next_day = next((x for x in open_days if x > d), None)
            backfilled = next_day is None or now_bj.strftime("%Y%m%d%H%M") >= f"{next_day}0925"
            meta.update({'Recorded_At': now_bj.strftime("%Y-%m-%d %H:%M"), 'Backfilled': bool(backfilled)})
            _replace_date_rows(TRACK_POOL_FILE, d, pd.DataFrame([{'Signal_Date': d, 'Pool_Codes': ",".join(pool_codes)}]),
                               ["Signal_Date"], key_col="Signal_Date")
            n_new += 1
        else:
            r0 = existing_row.iloc[0]
            meta = {k: r0[k] for k in ('Signal_Date', 'Breadth', 'Pool_N', 'Top_Sectors', 'SEC_N', 'SECRS_N',
                                       'Recorded_At', 'Backfilled') if k in existing_row.columns}
            all_stocks = _read_csv_safely(TRACK_STOCKS_FILE)
            stocks = all_stocks[all_stocks['Signal_Date'].astype(str) == d].copy()
            pool_df = _read_csv_safely(TRACK_POOL_FILE)
            pr = pool_df[pool_df['Signal_Date'].astype(str) == d]
            pool_codes = str(pr['Pool_Codes'].iloc[0]).split(",") if not pr.empty else []
            n_update += 1

        top2_mask = _truthy(stocks['Top2']) if stocks['Top2'].dtype != bool else stocks['Top2']
        row, per_stock = settle_week(d, stocks['ts_code'].tolist(), stocks.loc[top2_mask, 'ts_code'].tolist(),
                                     pool_codes, market)
        stocks = stocks.drop(columns=[c for c in ('Entry_Status', 'Buy_Price', 'Ret_W1 (%)', 'Ret_W2 (%)', 'Ret_W4 (%)')
                                      if c in stocks.columns])
        stocks['Entry_Status'] = [per_stock.get(c, {}).get('status', '') for c in stocks['ts_code']]
        stocks['Buy_Price'] = [round(per_stock.get(c, {}).get('buy', np.nan), 2) for c in stocks['ts_code']]
        for k in TRACK_HORIZONS:
            vals = [per_stock.get(c, {}).get(k, np.nan) for c in stocks['ts_code']]
            stocks[f'Ret_{k} (%)'] = [round(v, 2) if (row[f'Settled_{k}'] and np.isfinite(v)) else np.nan for v in vals]
        _replace_date_rows(TRACK_STOCKS_FILE, d, stocks, ["Signal_Date", "Sector_Rank"], key_col="Signal_Date")
        _replace_date_rows(TRACK_WEEKS_FILE, d, pd.DataFrame([{**meta, **row}]), ["Signal_Date"], key_col="Signal_Date")

    msg = f"跟踪已更新：新记录 {n_new} 周，更新收益 {n_update} 周。"
    if n_wait:
        msg += f" {n_wait} 周行情尚未发布，下次运行补上。"
    st.success(msg)

    latest = next((d for d in reversed(trade_days) if market.has_date(d)), None)
    recorded = set(_read_csv_safely(TRACK_WEEKS_FILE).get('Signal_Date', pd.Series(dtype=str)).astype(str))
    if latest and latest not in recorded:
        _, preview, _ = sector_snapshot(latest, whitelist_keys, market, sector_map, {}, latest in week_end_set, name_map)
        note = "（周中数据，周五收盘后可能变化）" if latest not in week_end_set else ""
        tag = "跟踪开始前的预览，不计入跟踪" if latest < TRACK_START else "预览，不计入跟踪"
        st.session_state['track_preview'] = (latest, preview, f"{tag}{note}")
    else:
        st.session_state.pop('track_preview', None)


if st.button({"🧭": "🚀 更新板块跟踪", "📌": "🚀 开始选股", "📊": "🚀 开始回测"}[MODE[:1]], type="primary"):
    is_valid, msg = verify_token_connection(token_clean)
    if not is_valid:
        st.error(f"❌ Token 预检失败：{msg}")
    else:
        try:
            st.cache_resource.clear()
            ts.set_token(token_clean)
            pro_api = ts.pro_api(token_clean)
            with st.spinner("正在加载科技股票池..."):
                wl_set, basic_name_map = load_custom_tech_whitelist(token_clean)
            wl_keys = tuple(sorted(wl_set))
            if not wl_keys:
                st.error("❌ 未能获取股票池，请检查 Token 积分或网络。")
            else:
                st.info(f"💡 股票池共 **{len(wl_keys)}** 只（筛股价、市值前）。")
                if is_picking_mode:
                    run_picking(pro_api, wl_keys, basic_name_map)
                else:
                    with st.spinner("正在加载申万二级行业映射（每周只下载一次）..."):
                        sw_map = load_sw_l2_map(token_clean)
                    covered = sum(1 for c in wl_keys if c in sw_map)
                    if covered < len(wl_keys) * 0.5:
                        st.warning(f"⚠️ 申万二级行业只覆盖 {covered}/{len(wl_keys)} 只，“强势板块”对照组结果不可靠"
                                   "（可能是 Tushare 积分不足以调用行业成分接口）。")
                    else:
                        st.info(f"申万二级行业覆盖 {covered}/{len(wl_keys)} 只。")
                    if is_tracking_mode:
                        run_tracking(pro_api, wl_keys, basic_name_map, sw_map)
                    else:
                        run_backtest(pro_api, wl_keys, basic_name_map, sw_map)
        except Exception as e:
            st.error(f"❌ 运行异常：{e}")
            with st.expander("错误详情"):
                st.code(traceback.format_exc())


# ---------------------------
# 回测报告
# ---------------------------
def weekly_full_sample_table(executed):
    hold_days = pd.to_numeric(executed.get('Hold_Days'), errors='coerce').fillna(0)
    exit_week = np.ceil(hold_days / 5.0)
    is_closed = executed['Exit_Reason'].astype(str) != '持仓中'
    final = pd.to_numeric(executed.get('Final_Return (%)'), errors='coerce')
    rows = []
    for w in range(1, HOLD_WEEKS + 1):
        col_name = f'Return_W{w} (%)'
        if col_name not in executed.columns:
            continue
        held = pd.to_numeric(executed[col_name], errors='coerce')
        locked = final.where(is_closed & (exit_week <= w))
        vals = held.where(held.notna(), locked).dropna()
        if vals.empty:
            continue
        rows.append({'周': f"W{w}", '样本数': len(vals), '平均收益%': round(vals.mean(), 2),
                     '胜率%': round((vals > 0).mean() * 100, 1)})
    return pd.DataFrame(rows)


BENCH_METRICS = ('N', 'W1', 'W2', 'W4', 'W12', 'Med12', 'MaxGain', 'MaxDD', 'Big30', 'Bear20')


def _half_label(date_series):
    s = date_series.astype(str)
    return s.str[:4] + np.where(pd.to_numeric(s.str[4:6], errors='coerce') <= 6, 'H1', 'H2')


def nw_tstat(series, lag=HOLD_WEEKS):
    """Newey-West t 值：相邻周的 12 周持有期重叠，普通 t 值会高估显著性。"""
    x = pd.to_numeric(series, errors='coerce').dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 8:
        return (float(np.mean(x)) if n else np.nan), np.nan, n
    e = x - x.mean()
    var = (e * e).sum() / n
    for L in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - L / (lag + 1.0)) * (e[L:] * e[:-L]).sum() / n
    t = x.mean() / np.sqrt(var / n) if var > 0 else np.nan
    return float(x.mean()), float(t), n


def prepare_bench_weeks(weeks_log):
    if weeks_log.empty or 'Bench_Complete' not in weeks_log.columns:
        return pd.DataFrame()
    wl = weeks_log.copy()
    for grp in [g for g, _ in BENCH_GROUPS] + [k for k, _, _, _ in METHOD_GROUPS] + ['Pool']:
        for m in BENCH_METRICS:
            col = f'{grp}_{m}'
            wl[col] = pd.to_numeric(wl[col], errors='coerce') if col in wl.columns else np.nan
    wl['Breadth'] = pd.to_numeric(wl['Breadth'], errors='coerce') if 'Breadth' in wl.columns else np.nan
    comp = wl[_truthy(wl['Bench_Complete'])].copy().sort_values('Trade_Date')
    if comp.empty:
        return comp
    comp['时期'] = _half_label(comp['Trade_Date'])
    comp['宽度分组'] = pd.cut(comp['Breadth'], [-0.1, 30, 50, 70, 100.1], labels=['<30%', '30~50%', '50~70%', '≥70%'])
    return comp


def benchmark_report_tables(comp, closed):
    tables = {}
    if comp.empty:
        return tables
    active = [(g, (f"入选{TOP_N}只" if g == "Pick" else label)) for g, label in BENCH_GROUPS
              if comp[f'{g}_W12'].notna().any()]

    # 1) 候选组总览（全部 + 样本外分界前/后）
    split = OOS_SPLIT.strftime("%Y%m%d")
    segments = [("全部", comp),
                (f"≤{split}", comp[comp['Trade_Date'].astype(str) <= split]),
                (f">{split}", comp[comp['Trade_Date'].astype(str) > split])]
    rows = []
    for seg_name, seg in segments:
        if seg.empty:
            continue
        for g, label in active:
            row = {'区间': seg_name, '候选组': label, '有效周数': int(seg[f'{g}_W12'].notna().sum()),
                   '平均每周股数': round(seg[f'{g}_N'].mean(), 1)}
            for h, lag in (("W1", 1), ("W2", 2), ("W4", 4), ("W12", 12)):
                ex_h = seg[f'{g}_{h}'] - seg[f'Pool_{h}']
                m_h, t_h, _ = nw_tstat(ex_h, lag=lag)
                label_h = h.replace("W", "") + "周"
                row[f'{label_h}超额%'] = round(m_h, 2) if pd.notna(m_h) else np.nan
                row[f'{label_h}t值'] = round(t_h, 2) if pd.notna(t_h) else np.nan
            ex = seg[f'{g}_W12'] - seg['Pool_W12']
            half = ex.groupby(seg['时期']).mean().dropna()
            bm, bt, _ = nw_tstat(seg[f'{g}_Big30'] - seg['Pool_Big30'])
            brm, brt, _ = nw_tstat(seg[f'{g}_Bear20'] - seg['Pool_Bear20'])
            row.update({
                '12周跑赢池的半年': f"{int((half > 0).sum())}/{len(half)}",
                '牛股率差(百分点)': round(bm, 1) if pd.notna(bm) else np.nan,
                '熊股率差(百分点)': round(brm, 1) if pd.notna(brm) else np.nan,
            })
            rows.append(row)
    tables['候选组总览'] = pd.DataFrame(rows)

    # 2) 分时期超额
    cl = closed.copy() if closed is not None and len(closed) else pd.DataFrame(columns=['Trade_Date', 'Final_Return (%)'])
    cl['Trade_Date'] = cl['Trade_Date'].astype(str)
    cl = cl[cl['Trade_Date'].isin(set(comp['Trade_Date']))]
    cl['时期'] = _half_label(cl['Trade_Date']) if len(cl) else pd.Series(dtype=str)
    cl = cl.merge(comp[['Trade_Date', '宽度分组']], on='Trade_Date', how='left')

    def excess_rows(key_col, closed_key_col):
        out = []
        keys = list(comp.groupby(key_col, observed=True, sort=True).groups.keys()) + ['全部']
        for key in keys:
            g_df = comp if key == '全部' else comp[comp[key_col] == key]
            c_df = cl if key == '全部' else cl[cl[closed_key_col] == key]
            row = {key_col: str(key), '周数': len(g_df), '宽度%': round(g_df['Breadth'].mean(), 0),
                   '股票池12周%': round(g_df['Pool_W12'].mean(), 2),
                   '入选规则出场均益%': round(c_df['Final_Return (%)'].mean(), 2) if len(c_df) else np.nan}
            for g, label in active:
                row[f'{label}−池'] = round((g_df[f'{g}_W12'] - g_df['Pool_W12']).mean(), 2)
            out.append(row)
        return pd.DataFrame(out)

    tables['分时期_12周超额'] = excess_rows('时期', '时期')

    # 板块内选股方法对比（基准=前3强板块全部成分）
    m_rows = []
    for seg_name, seg in segments:
        if seg.empty:
            continue
        for key, label, _, _ in METHOD_GROUPS:
            if f'{key}_W2' not in seg.columns or seg[f'{key}_W2'].notna().sum() == 0:
                continue
            row = {'区间': seg_name, '方法': label, '有效周数': int(seg[f'{key}_W2'].notna().sum()),
                   '平均每周股数': round(seg[f'{key}_N'].mean(), 1)}
            for h, lag in (("W1", 1), ("W2", 2), ("W4", 4), ("W12", 12)):
                m_h, t_h, _ = nw_tstat(seg[f'{key}_{h}'] - seg[f'SEC_{h}'], lag=lag)
                wk = h.replace("W", "") + "周"
                row[f'{wk}超额vs板块%'] = round(m_h, 2) if pd.notna(m_h) else np.nan
                row[f'{wk}t值'] = round(t_h, 2) if pd.notna(t_h) else np.nan
            ex2 = seg[f'{key}_W2'] - seg['SEC_W2']
            half = ex2.groupby(seg['时期']).mean().dropna()
            bm, _, _ = nw_tstat(seg[f'{key}_Big30'] - seg['SEC_Big30'])
            brm, _, _ = nw_tstat(seg[f'{key}_Bear20'] - seg['SEC_Bear20'])
            m2p, _, _ = nw_tstat(seg[f'{key}_W2'] - seg['Pool_W2'], lag=2)
            row.update({
                '2周跑赢板块平均的半年': f"{int((half > 0).sum())}/{len(half)}",
                '2周超额vs池%': round(m2p, 2) if pd.notna(m2p) else np.nan,
                '牛股率差vs板块(百分点)': round(bm, 1) if pd.notna(bm) else np.nan,
                '熊股率差vs板块(百分点)': round(brm, 1) if pd.notna(brm) else np.nan,
            })
            m_rows.append(row)
    if m_rows:
        tables['板块内选股方法对比'] = pd.DataFrame(m_rows)

    w2_rows = []
    for key in list(comp.groupby('时期', sort=True).groups.keys()) + ['全部']:
        g_df = comp if key == '全部' else comp[comp['时期'] == key]
        row = {'时期': key, '周数': len(g_df), '股票池2周%': round(g_df['Pool_W2'].mean(), 2)}
        for g, label in active:
            row[f'{label}−池'] = round((g_df[f'{g}_W2'] - g_df['Pool_W2']).mean(), 2)
        w2_rows.append(row)
    tables['分时期_2周超额'] = pd.DataFrame(w2_rows)

    big_rows = []
    for key in list(comp.groupby('时期', sort=True).groups.keys()) + ['全部']:
        g_df = comp if key == '全部' else comp[comp['时期'] == key]
        row = {'时期': key, '股票池': round(g_df['Pool_Big30'].mean(), 1)}
        for g, label in active:
            row[label] = round(g_df[f'{g}_Big30'].mean(), 1)
        big_rows.append(row)
    tables['分时期_牛股率'] = pd.DataFrame(big_rows)

    bear_rows = []
    for key in list(comp.groupby('时期', sort=True).groups.keys()) + ['全部']:
        g_df = comp if key == '全部' else comp[comp['时期'] == key]
        row = {'时期': key, '股票池': round(g_df['Pool_Bear20'].mean(), 1)}
        for g, label in active:
            row[label] = round(g_df[f'{g}_Bear20'].mean(), 1)
        bear_rows.append(row)
    tables['分时期_熊股率'] = pd.DataFrame(bear_rows)

    tables['按市场宽度_12周超额'] = excess_rows('宽度分组', '宽度分组')

    if 'Top_Sectors' in comp.columns:
        sec = comp[['Trade_Date', 'Breadth', 'Top_Sectors', 'Pool_W12', 'SEC_W12', 'SECRS_W12']].copy()
        sec['板块−池'] = (sec['SEC_W12'] - sec['Pool_W12']).round(2)
        tables['每周强势板块'] = sec.sort_values('Trade_Date', ascending=False).rename(columns={
            'Trade_Date': '周末日期', 'Breadth': '宽度%', 'Top_Sectors': '前3强板块(12周涨幅)',
            'Pool_W12': '股票池12周%', 'SEC_W12': '板块全部12周%', 'SECRS_W12': '板块最强2只12周%'})
    return tables


def group_stats(closed, key):
    g = closed.groupby(key, observed=True)
    out = g.agg(
        笔数=('Final_Return (%)', 'count'),
        胜率=('Final_Return (%)', lambda x: round((x > 0).mean() * 100, 1)),
        平均收益=('Final_Return (%)', lambda x: round(x.mean(), 2)),
        中位收益=('Final_Return (%)', lambda x: round(x.median(), 2)),
        止损占比=('Exit_Reason', lambda x: round(x.astype(str).str.contains('破-10%').mean() * 100, 1)),
        平均持有天数=('Hold_Days', lambda x: round(pd.to_numeric(x, errors='coerce').mean(), 1)),
    ).reset_index()
    if 'Fwd_W12 (%)' in closed.columns:
        fixed = closed.groupby(key, observed=True)['Fwd_W12 (%)'].apply(
            lambda x: round(pd.to_numeric(x, errors='coerce').mean(), 2)).reset_index(name='固定持有12周均益')
        out = out.merge(fixed, on=key, how='left')
    return out


def build_export_zip(tables, meta):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('00_回测参数.json', json.dumps(meta, ensure_ascii=False, indent=2))
        for i, (name, df) in enumerate(tables.items(), start=1):
            if df is None or df.empty:
                continue
            zf.writestr(f'{i:02d}_{name}.csv', df.to_csv(index=False).encode('utf-8-sig'))
    return buf.getvalue()


if is_backtest_mode and (os.path.exists(TRADES_FILE) or os.path.exists(WEEKS_FILE)):
    st.markdown("---")
    st.header(f"📈 回测报告（参数组 {CFG_SIG}：每周{TOP_N}只，层级 {'+'.join(tiers_enabled)}）")
    export_slot = st.container()
    export_tables = {}
    try:
        weeks_log = _read_csv_safely(WEEKS_FILE)
        trades = _read_csv_safely(TRADES_FILE)
        rs_str, re_str = REPORT_START.strftime("%Y%m%d"), REPORT_END.strftime("%Y%m%d")
        if not weeks_log.empty:
            weeks_log = weeks_log[weeks_log['Trade_Date'].astype(str).between(rs_str, re_str)].copy()
        if not trades.empty:
            trades = trades[trades['Trade_Date'].astype(str).between(rs_str, re_str)].copy()
        if not weeks_log.empty:
            st.caption(f"报告区间：{weeks_log['Trade_Date'].min()} ~ {weeks_log['Trade_Date'].max()}（共 {len(weeks_log)} 周）")
        else:
            st.info("所选报告区间内没有回测记录。")

        # ---- 空周统计
        if not weeks_log.empty:
            weeks_log['Picks'] = pd.to_numeric(weeks_log['Picks'], errors='coerce').fillna(0).astype(int)
            n_weeks = len(weeks_log)
            n_empty = int((weeks_log['Picks'] == 0).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("已扫描周数", n_weeks)
            c2.metric("空周数（0只）", n_empty)
            c3.metric("折合每年空周", f"{n_empty / n_weeks * 52:.1f}" if n_weeks else "-")
            wl_year = weeks_log.assign(
                年份=weeks_log['Trade_Date'].astype(str).str[:4],
                全A级=weeks_log['Pick_Tiers'].fillna("").astype(str).apply(lambda s: len(s) > 0 and set(s) == {"A"}),
            )
            by_year = wl_year.groupby('年份').agg(
                扫描周数=('Picks', 'size'),
                空周数=('Picks', lambda x: int((x == 0).sum())),
                名额全为A级的周=('全A级', 'sum'),
            ).reset_index()
            export_tables['按年空周统计'] = by_year

        # ---- 交易数据准备
        closed, executed, excluded = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if not trades.empty and 'Exit_Reason' in trades.columns:
            trades['Final_Return (%)'] = pd.to_numeric(trades['Final_Return (%)'], errors='coerce')
            if 'Tier' not in trades.columns:
                trades['Tier'] = 'A'
            excluded = trades[trades['Exit_Reason'].astype(str).str.contains('剔除', na=False)]
            executed = trades[~trades.index.isin(excluded.index)].copy()
            closed = executed[executed['Exit_Reason'].astype(str) != '持仓中'].copy()

        # ---- 对照组
        comp = prepare_bench_weeks(weeks_log)
        bench_tables = benchmark_report_tables(comp, closed)
        if bench_tables:
            st.markdown("#### 🧪 候选组总览（与同周股票池对照，只含已满12周的周）")
            st.caption(
                "12周超额=该组次日开盘买入、不设止损持有60个交易日的平均收益减去同周股票池；每周等权。"
                "每个持有期的 t 值都按各自的重叠长度修正，绝对值≥2 才算显著。一个方向值得继续，至少要：超额为正、t值≥2、"
                "多数半年跑赢池子（包括 2022~2024 年），牛股率差也不为负。"
            )
            show_df(bench_tables['候选组总览'])
            if '板块内选股方法对比' in bench_tables:
                st.markdown("#### 🎯 板块内选股方法对比（基准：前3强板块全部成分，相当于板块里随机挑）")
                st.caption(
                    "事先定好的判定标准：主看 2 周。某方法相对板块平均的 2 周超额，在全部区间 t≥2，"
                    "且样本外分界前、后两段都为正，熊股率差不高于 +5 个百分点，才算有效。"
                    "7 种方法同时比较，偶然有一种 t≥2 的概率不低，所以“两段都为正”是硬条件。"
                )
                show_df(bench_tables['板块内选股方法对比'])
            st.markdown("#### 🗓️ 分时期：2周超额（相对股票池）")
            show_df(bench_tables['分时期_2周超额'])
            st.markdown("#### 🗓️ 分时期：12周超额（相对股票池）")
            show_df(bench_tables['分时期_12周超额'])
            st.markdown("#### 🐂 分时期：牛股率%（60个交易日内最高涨幅≥30%的比例）")
            show_df(bench_tables['分时期_牛股率'])
            st.markdown("#### 🐻 分时期：熊股率%（60个交易日内最大回撤≤-20%的比例）")
            st.caption("牛股率和熊股率同时偏高，说明该组只是波动更大，不代表更会选股。")
            show_df(bench_tables['分时期_熊股率'])
            st.markdown("#### 🌡️ 按市场宽度：12周超额")
            st.caption("宽度=过筛股票中周收盘站上20周线的比例，分组边界固定为30/50/70。")
            show_df(bench_tables['按市场宽度_12周超额'])
            if '每周强势板块' in bench_tables:
                with st.expander("查看每周前3强板块"):
                    show_df(bench_tables['每周强势板块'])
            export_tables.update(bench_tables)

        if '按年空周统计' in export_tables:
            st.markdown("#### 🗓️ 按年空周统计（SKDJ 入选）")
            show_df(export_tables['按年空周统计'])

        # ---- SKDJ 入选交易（规则出场）
        if not trades.empty and 'Exit_Reason' in trades.columns:
            st.markdown("#### 💼 SKDJ 入选交易总览（规则出场，已出场）")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("已出场笔数", len(closed))
            summary = {'已出场笔数': len(closed), '开盘剔除': len(excluded), '持仓中': len(executed) - len(closed)}
            if len(closed):
                win = (closed['Final_Return (%)'] > 0).mean() * 100
                avg = closed['Final_Return (%)'].mean()
                c2.metric("胜率", f"{win:.1f}%")
                c3.metric("平均单笔收益", f"{avg:.2f}%")
                summary.update({'胜率%': round(win, 1), '平均单笔收益%': round(avg, 2)})
            c4.metric("开盘剔除 / 持仓中", f"{len(excluded)} / {len(executed) - len(closed)}")
            export_tables['交易总览'] = pd.DataFrame([summary])

            if len(closed):
                tier_tbl = group_stats(closed, 'Tier')
                tier_tbl['Tier'] = tier_tbl['Tier'].map(TIER_LABEL).fillna(tier_tbl['Tier'])
                tier_tbl = tier_tbl.rename(columns={'Tier': '层级'})
                rank_tbl = group_stats(closed, 'Rank').rename(columns={'Rank': '排名'})
                reasons = closed['Exit_Reason'].astype(str).str.replace(r'\(.*\)', '', regex=True)
                reason_tbl = reasons.value_counts().rename_axis('出场原因').reset_index(name='笔数')
                st.markdown("#### 🧱 分层级统计")
                show_df(tier_tbl)
                st.markdown("#### 🏅 按排名统计")
                show_df(rank_tbl)
                st.markdown("#### 🚪 出场原因分布")
                show_df(reason_tbl)
                export_tables.update({'分层级统计': tier_tbl, '按排名统计': rank_tbl, '出场原因分布': reason_tbl})

            st.markdown("#### 📅 W1~W12 全样本收益（已出场按出场收益计入）")
            wtbl = weekly_full_sample_table(executed)
            if not wtbl.empty:
                show_df(wtbl)
                export_tables['W1-W12全样本收益'] = wtbl

        # ---- 明细
        if not weeks_log.empty:
            with st.expander("🗂️ 每周扫描明细"):
                wl_disp = weeks_log.sort_values('Trade_Date', ascending=False)[
                    ['Trade_Date', 'A_Count', 'B_Count', 'C_Count', 'Picks', 'Pick_Tiers', 'Pick_Names']
                ].rename(columns={'Trade_Date': '周末日期', 'A_Count': 'A级候选', 'B_Count': 'B级候选',
                                  'C_Count': 'C级候选', 'Picks': '入选数', 'Pick_Tiers': '入选层级', 'Pick_Names': '入选股票'})
                show_df(wl_disp)
            export_tables['每周明细_原始数据'] = weeks_log

        if not trades.empty:
            with st.expander("📋 交割流水"):
                disp_cols = ['Trade_Date', 'Rank', 'Tier_Label', 'name', 'ts_code', 'Total_Score', 'SKDJ_K',
                             'Close_Raw', 'Buy_Price', 'Gap_pct (%)', 'Exit_Date', 'Hold_Days', 'Exit_Reason', 'Final_Return (%)',
                             'Fwd_W12 (%)', 'Fwd_MaxGain (%)', 'Fwd_MaxDD (%)']
                disp_cols = [c for c in disp_cols if c in trades.columns]
                show_df(trades[disp_cols].sort_values(['Trade_Date', 'Rank'], ascending=[False, True]))
            export_tables['交割流水_原始数据'] = trades

    except Exception as report_error:
        st.warning(f"回测记录已保留，但部分报告暂时无法显示：{report_error}")

    if export_tables:
        meta = {
            '版本': VERSION, '参数组': CFG_SIG, '导出时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '回测截止日期': backtest_date_end.strftime('%Y-%m-%d'), '回测周数设置': BACKTEST_WEEKS,
            '报告区间': [REPORT_START.strftime('%Y-%m-%d'), REPORT_END.strftime('%Y-%m-%d')],
            '样本外分界日期': OOS_SPLIT.strftime('%Y-%m-%d'),
            '每周选股数': TOP_N, '启用层级': tiers_enabled, '最低股价': MIN_PRICE,
            '流通市值范围(亿)': [MIN_MV, MAX_MV],
            '对照组定义': {
                '强势股前10%': f'截至上周的{RS_LOOKBACK_WEEKS}周涨幅排名前{RS_TOP_PCT:.0f}%（跳过最近一周）',
                '强势板块': f'申万二级、组内≥{SECTOR_MIN_MEMBERS}只、成分股{RS_LOOKBACK_WEEKS}周涨幅均值前{SECTOR_TOP_K}',
                '板块内强势股': f'前{SECTOR_TOP_K}强板块各取{RS_LOOKBACK_WEEKS}周涨幅最高{SECTOR_STOCKS_EACH}只',
                '固定持有': '次日开盘买入，不设止损，持有60个交易日；三组使用同一套开盘剔除规则',
            },
        }
        export_slot.download_button(
            label="📦 一键导出全部回测结果 (ZIP)",
            data=build_export_zip(export_tables, meta),
            file_name=f"skdj_{VERSION}_{CFG_SIG}_{rs_str}-{re_str}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            key="download_all_zip",
            type="primary",
        )


# ---------------------------
# 板块动量跟踪页面
# ---------------------------
def track_summary_table(weeks):
    rows = []
    for g, label in (("SEC", "前3强板块全部成分"), ("SECRS", "前3强板块各取最强2只")):
        row = {'组别': label}
        for k, lag in (("W1", 1), ("W2", 2), ("W4", 4)):
            settled = weeks[_truthy(weeks[f'Settled_{k}'])] if f'Settled_{k}' in weeks.columns else weeks.iloc[0:0]
            ex = pd.to_numeric(settled.get(f'{g}_Excess_{k} (%)', pd.Series(dtype=float)), errors='coerce').dropna()
            m, t, n = nw_tstat(ex, lag=lag) if len(ex) else (np.nan, np.nan, 0)
            wk = k.replace("W", "") + "周"
            row[f'{wk}已结算周数'] = n
            row[f'{wk}平均超额%'] = round(m, 2) if pd.notna(m) else np.nan
            row[f'{wk}跑赢池的周%'] = round((ex > 0).mean() * 100, 0) if len(ex) else np.nan
            row[f'{wk}t值'] = round(t, 2) if pd.notna(t) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


if is_tracking_mode:
    st.markdown("---")
    track_weeks = _read_csv_safely(TRACK_WEEKS_FILE)
    track_stocks = _read_csv_safely(TRACK_STOCKS_FILE)
    track_pool = _read_csv_safely(TRACK_POOL_FILE)
    export_slot_track = st.container()

    preview = st.session_state.get('track_preview')
    if not track_weeks.empty:
        last_sig = track_weeks['Signal_Date'].astype(str).max()
        show_track_list(track_stocks[track_stocks['Signal_Date'].astype(str) == last_sig],
                        f"### 🧭 最新记录名单（信号日 {last_sig}，下一个交易日开盘买入）")
    if preview is not None:
        p_date, p_stocks, p_note = preview
        show_track_list(p_stocks, f"### 👀 {p_date} 名单（{p_note}）")
    if track_weeks.empty and preview is None:
        st.info(f"还没有跟踪记录。跟踪从 {TRACK_START} 这一周开始；点击上方按钮可以先看当前名单预览。")

    if not track_weeks.empty:
        st.markdown("### 📈 跟踪成绩（相对同周股票池）")
        summary = track_summary_table(track_weeks)
        show_df(summary)
        n2 = int(summary.loc[0, '2周已结算周数']) if not summary.empty else 0
        avg2 = summary.loc[0, '2周平均超额%'] if not summary.empty else np.nan
        st.caption(
            f"回测预期：前3强板块全部成分 2周超额约 +{TRACK_EXPECT_W2:.2f}%。每周超额波动约 ±3.5 个百分点，"
            "一年只有约 26 个不重叠的 2 周样本，所以一年内无法证明有效，只能判断是否明显失效。"
            f"事先定好的停止规则：满 52 周后，若 2 周平均超额低于 {TRACK_STOP_W2:.1f}%，判定失效、停止使用。"
        )
        if n2 >= 52 and pd.notna(avg2):
            if avg2 < TRACK_STOP_W2:
                st.error(f"已满 {n2} 周，2周平均超额 {avg2:.2f}% 低于停止线 {TRACK_STOP_W2:.1f}%：判定失效。")
            else:
                st.success(f"已满 {n2} 周，2周平均超额 {avg2:.2f}%，未触发停止线，可继续作为选股过滤条件。")
        else:
            st.info(f"已结算 2 周的周数：{n2}/52，继续跟踪。")

        disp = track_weeks.sort_values('Signal_Date', ascending=False).copy()
        disp['记录方式'] = np.where(_truthy(disp['Backfilled']), '补记', '前瞻') if 'Backfilled' in disp.columns else ''
        cols = {'Signal_Date': '信号日', 'Top_Sectors': '前3强板块(12周涨幅)', 'SEC_N': '成分股数', 'Breadth': '宽度%',
                'Pool_W2 (%)': '股票池2周%', 'SEC_W2 (%)': '板块2周%', 'SEC_Excess_W2 (%)': '板块2周超额%',
                'SECRS_Excess_W2 (%)': '最强2只2周超额%', 'SEC_Excess_W1 (%)': '板块1周超额%',
                'SEC_Excess_W4 (%)': '板块4周超额%', 'Status': '状态', 'Recorded_At': '记录时间(北京)', '记录方式': '记录方式'}
        with st.expander("每周跟踪明细", expanded=True):
            show_df(disp[[c for c in cols if c in disp.columns]].rename(columns=cols))

        track_meta = {
            '版本': VERSION, '导出时间(北京)': _bj_now().strftime('%Y-%m-%d %H:%M'), '跟踪起点': TRACK_START,
            '规则': {
                '股票池': '科技白名单；未复权股价≥10元；流通市值50~1000亿；上市≥100个交易日；信号日非一字涨停',
                '板块排名': f'申万二级，组内≥{SECTOR_MIN_MEMBERS}只，成分股截至上周的{RS_LOOKBACK_WEEKS}周涨幅均值前{SECTOR_TOP_K}',
                '最强2只': f'前{SECTOR_TOP_K}强板块内各取{RS_LOOKBACK_WEEKS}周涨幅最高{SECTOR_STOCKS_EACH}只',
                '收益': '信号日为每周最后一个交易日；次日开盘买入；5/10/20个交易日收盘计算；开盘剔除规则与回测一致',
                '停止规则': f'满52周后2周平均超额低于{TRACK_STOP_W2}%判定失效',
            },
        }
        export_slot_track.download_button(
            label="📦 一键导出跟踪记录 (ZIP)",
            data=build_export_zip({'跟踪成绩汇总': summary, '每周跟踪明细': track_weeks,
                                   '成分股明细': track_stocks, '股票池代码': track_pool}, track_meta),
            file_name=f"skdj_track_{_bj_now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip", key="download_track_zip", type="primary",
        )
