# -*- coding: utf-8 -*-
"""板块领涨惯性 M1.6 — 单一效率过滤对照 / 等机会审计 / 持仓约束重放

依赖：pandas >= 2.0, numpy >= 1.24, streamlit >= 1.32, tushare >= 1.4。
离线逻辑验算：python app.py --self-test
离线演示审计：python app.py --demo --output ./momentum_demo
接口字段依据：https://tushare.pro/document/2?doc_id=27 / 32 / 183 / 335。
本版本是未验证盈利的研究实现，日线撮合不是精确的分钟成交回放。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import inspect
import json
import math
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

VERSION = "M1.6"
CORE = {"电子", "计算机", "通信", "国防军工"}
TECH_WORDS = ("自动化", "机器人", "仪器仪表", "半导体", "光伏设备", "风电设备",
              "电池", "电网设备", "医疗器械", "电子", "金属新材料")
BAR_COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
            "vol", "amount", "circ_mv", "turnover_rate", "adj_factor", "up_limit", "down_limit"]
FIELDS = {
    "daily": "ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
    "daily_basic": "ts_code,trade_date,circ_mv,turnover_rate",
    "adj_factor": "ts_code,trade_date,adj_factor",
    "stk_limit": "ts_code,trade_date,up_limit,down_limit",
}
GROUP_MAIN, GROUP_BASE, GROUP_HOT = "综合动量", "昨日涨幅对照", "过热观察_不买"
GROUP_OBSERVE = "前十独立观察_非买入组合"
TOP10_MODELS = (("T10_PRE3", "前十_低前涨幅", "pre3", True),
                ("T10_ACCEL", "前十_相对强度提升", "rs_accel", False))
EXPLORED_START, EXPLORED_END, FREEZE_DAY = "20250920", "20260918", "20260921"
FROZEN_PARAMS = dict(scope="科技行业", min_price=10.0, min_mv=50.0, max_mv=1000.0,
    top_n=3, heat_limit=.50, max_gap=.03, hold_days=5, stop_loss=.10,
    buy_fee=.0003, sell_fee=.0003, slippage=.001, min_sector_n=5)
VALIDATION_GROUPS = (GROUP_BASE, TOP10_MODELS[0][1])
EFF_EXPERIMENT, EFF_GROUP = 'T10_PRE3_EFF', '低前涨幅_剔除高效率'
EFF_CUTOFF, EFF_FREEZE_DAY = 200/3, '20260922'


@dataclass(frozen=True)
class Config:
    start: str = "20240920"
    end: str = "20250919"
    scope: str = "科技行业"
    min_price: float = 10.0
    min_mv: float = 50.0
    max_mv: float = 1000.0
    top_n: int = 3
    heat_limit: float = 0.50
    max_gap: float = 0.03
    growth_gap: float = 0.06
    research: bool = True
    filter_research: bool = False
    legacy_research: bool = False
    validation_only: bool = True
    efficiency_research: bool = True
    hold_days: int = 5
    stop_loss: float = 0.10
    buy_fee: float = 0.0003
    sell_fee: float = 0.0003
    slippage: float = 0.001
    min_sector_n: int = 5

    def validate(self):
        if ds(self.start) > ds(self.end):
            raise ValueError("开始日期不能晚于结束日期")
        if self.min_mv > self.max_mv or self.min_price < 0 or self.min_mv < 0:
            raise ValueError("价格或市值范围无效")
        if not 2 <= self.hold_days <= 5 or not 1 <= self.top_n <= 3:
            raise ValueError("持有上限应为2—5个市场交易日（含买入日），每日最多3只")
        if not 0 < self.stop_loss < 1 or not 0 < self.heat_limit <= 2:
            raise ValueError("止损或过热阈值无效")
        if any(x < 0 or x >= 0.1 for x in [self.buy_fee, self.sell_fee, self.slippage]):
            raise ValueError("费用参数无效")
        if not 0 <= self.max_gap <= self.growth_gap <= .30:
            raise ValueError("买入溢价须满足 0≤统一/主板上限≤双创上限≤30%")


SCORE_COLS = ("strength_score", "efficiency_score", "volume_score", "position_score")


def board_name(code):
    code = str(code)
    return "创业板" if code.startswith("30") else "科创板" if code.startswith("68") else "主板"


def legacy_experiments(cfg):
    """预先固定的22组，不按回测表现搜索或自动选出最佳参数。"""
    designs = [
        ("B0", "等权基准", GROUP_MAIN, (25, 25, 25, 25), 0, "score", False),
        ("W_RS", "单维加权", "相对强度40", (40, 20, 20, 20), 0, "score", False),
        ("W_EFF", "单维加权", "上涨效率40", (20, 40, 20, 20), 0, "score", False),
        ("W_VOL", "单维加权", "量价配合40", (20, 20, 40, 20), 0, "score", False),
        ("W_POS", "单维加权", "收盘位置40", (20, 20, 20, 40), 0, "score", False),
    ]
    designs += [(f"RS{n}", "连续相对强度", f"连续{n}日跑赢", (25, 25, 25, 25), n, "score", False) for n in [1, 2, 3, 5]]
    designs += [("RETURN", "涨幅对照", GROUP_BASE, (25, 25, 25, 25), 0, "ret1", False),
                ("HOT", "过热观察", GROUP_HOT, (25, 25, 25, 25), 0, "score", True)]
    result = []
    for policy in ["统一上限", "分板块上限"]:
        for key, family, label, weights, days, sort, hot in designs:
            if not cfg.research and (policy != "统一上限" or key not in ["B0", "RETURN", "HOT"]):
                continue
            result.append(dict(experiment=key, family=family, group=label if policy == "统一上限" else "分板块_"+label,
                               gap_policy=policy, weights=weights, rs_days=days, sort=sort, hot=hot))
    return result


FILTERS = (("F_RS", "相对强度", "strength_score"), ("F_EFF", "上涨效率", "efficiency_score"),
           ("F_VOL", "量价配合", "volume_score"), ("F_POS", "收盘位置", "position_score"))


def experiments(cfg):
    specs = legacy_experiments(replace(cfg, research=cfg.legacy_research))
    for spec in specs:
        spec.update(filter_col="", filter_label="无", filter_cutoff=np.nan)
    if cfg.filter_research:
        for key, label, col in FILTERS:
            specs.append(dict(experiment=key, family="涨幅前三过滤", group="涨幅前三_过滤"+label,
                              gap_policy="统一上限", weights=(25, 25, 25, 25), rs_days=0,
                              sort="ret1", hot=False, filter_col=col, filter_label=label, filter_cutoff=25.0))
    if cfg.research or cfg.validation_only or cfg.efficiency_research:
        for key, label, col, ascending in TOP10_MODELS:
            if not (cfg.research or cfg.validation_only) and key != 'T10_PRE3':
                continue
            specs.append(dict(experiment=key, family="前十选三假设", group=label,
                              gap_policy="统一上限", weights=(25, 25, 25, 25), rs_days=0,
                              sort=col, ascending=ascending, top10=True, hot=False,
                              filter_col="", filter_label="无", filter_cutoff=np.nan))
    if cfg.efficiency_research:
        spec = next(s.copy() for s in specs if s['experiment'] == 'T10_PRE3')
        spec.update(experiment=EFF_EXPERIMENT, group=EFF_GROUP, family='效率过滤对照', efficiency_guard=True)
        specs.append(spec)
    if cfg.validation_only:
        return [s for s in specs if s["experiment"] in ["RETURN", "T10_PRE3", EFF_EXPERIMENT] and s["gap_policy"] == "统一上限"]
    return specs


def add_filter_ranks(candidates):
    """只用当日基础合格且不过热的池；先固定分位，再过滤涨幅前三。"""
    f = candidates.copy()
    valid = f.base_reason.fillna("").eq("") & ~f.overheat.eq(True)
    f["filter_pool_n"] = 0
    g = f.loc[valid]
    f.loc[valid, "filter_pool_n"] = g.groupby("date").ts_code.transform("size")
    for col in SCORE_COLS:
        f[col+"_filter_pct"] = np.nan
        grouped = g.groupby("date")[col]
        f.loc[valid, col+"_filter_pct"] = 100*(grouped.rank(method="average")-.5)/grouped.transform("count")
    return f


def apply_filter_rule(frame, spec):
    f = frame.copy()
    if not spec["filter_col"]:
        f["filter_pass"], f["filter_reason"], f["filter_pct"] = True, "未启用过滤", np.nan
        return f
    f["filter_pct"] = f[spec["filter_col"]+"_filter_pct"]
    small = f.filter_pool_n.lt(4)
    f["filter_pass"] = small | f.filter_pct.ge(spec["filter_cutoff"])
    f["filter_reason"] = np.select(
        [small, f.filter_pct.isna(), f.filter_pass],
        ["合格池不足4只，保留且标记不可评估", "过滤分位缺失，拒绝", "通过"],
        default=spec["filter_label"]+"处于当日合格池末25%，拒绝")
    return f


def screening_decisions(candidates, cfg):
    f = add_filter_ranks(candidates)
    f = f[f.base_reason.fillna("").eq("") & ~f.overheat.eq(True)].copy()
    f = f.sort_values(["date", "ret1", "ts_code"], ascending=[True, False, True])
    f["rank"] = f.groupby("date").cumcount()+1
    f = f[f["rank"].le(cfg.top_n)]
    parts = []
    for spec in experiments(cfg):
        if not spec["filter_col"]:
            continue
        part = apply_filter_rule(f, spec)
        for key in ["group", "experiment", "filter_col", "filter_label", "filter_cutoff"]:
            part[key] = spec[key]
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def buy_gap(code, policy, cfg):
    return cfg.growth_gap if policy == "分板块上限" and board_name(code) != "主板" else cfg.max_gap


def top10_candidates(candidates, cfg):
    """冻结收盘时合格池的涨幅前十；不按未来是否能成交替换。"""
    f = candidates[candidates.base_reason.fillna("").eq("") & ~candidates.overheat.eq(True)].copy()
    f = f.sort_values(["date", "ret1", "ts_code"], ascending=[True, False, True])
    f["eligible_pool_n"] = f.groupby("date").ts_code.transform("size")
    f["return_rank"] = f.groupby("date").cumcount()+1
    f = f[f.return_rank.le(10)].copy()
    f["top10_pool_n"] = f.groupby("date").ts_code.transform("size")
    for col in ["pre3", "rs_accel", "signal_limit_up", "signal_limit_ratio"]:
        if col not in f:
            f[col] = np.nan  # 旧报告缺字段时不反推、不伪造。
    f["rank_band"] = np.select([f.return_rank.le(3), f.return_rank.le(6)], ["1—3名", "4—6名"], default="7—10名")
    f["pre3_band"] = np.select([f.pre3.isna(), f.pre3.le(0), f.pre3.le(.10)],
                              ["缺失", "≤0%", "0—10%"], default=">10%")
    f["signal_gain_band"] = np.where(f.ret1.gt(.15+1e-12), ">15%", "≤15%")
    f["signal_limit_state"] = np.select([f.signal_limit_up.isna(), f.signal_limit_up.eq(True)],
                                        ["未知", "收盘涨停"], default="未收盘涨停")
    f["board"] = f.ts_code.map(board_name)
    f["year"] = pd.to_datetime(f.date).dt.year
    f["gap_limit"] = cfg.max_gap
    f["max_buy_reference"] = f.close*(1+cfg.max_gap)
    return f


def top10_observation(candidates, cfg):
    f = top10_candidates(candidates, cfg)
    if not cfg.research or cfg.validation_only:
        return f.iloc[:0]
    f = f.assign(group=GROUP_OBSERVE, experiment="OBS_TOP10", family="前十独立观察",
                 gap_policy="统一上限", rs_days=0)
    f["rank"] = f.return_rank
    return f


def consecutive_rs(excess, sector, n):
    """截至信号日连续n个市场交易日每天跑赢同一板块；缺失/平局不通过。"""
    wins = excess.gt(1e-12).astype(float).where(excess.notna())
    same = pd.Series(True, index=excess.index)
    for lag in range(1, n):
        same &= sector.eq(sector.shift(lag)) & sector.notna()
    return wins.rolling(n, min_periods=n).sum().eq(n) & same


def ds(x):
    return pd.Timestamp(str(x)).strftime("%Y%m%d")


def dates(values):
    s = pd.Series(values, dtype="string").str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(s, errors="coerce", format="mixed")


def ready_day():
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return pd.Timestamp(now.date() - timedelta(days=int(now.hour < 18)))


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    frame.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def read_csv(path):
    return pd.read_csv(path, compression="gzip", dtype={c: str for c in
        ["ts_code", "trade_date", "in_date", "out_date", "start_date", "end_date",
         "ann_date", "list_date", "delist_date", "cal_date"]})


class DataError(RuntimeError):
    pass


class MarketStore:
    """全A行情按日原子提交，按股票读取，避免把多年全市场行情放进内存。"""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=30)
        self.db.execute("PRAGMA cache_size=-4096")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("CREATE TABLE IF NOT EXISTS bars (ts_code TEXT, trade_date TEXT, " +
                        ",".join(c + " REAL" for c in BAR_COLS[2:]) +
                        ", PRIMARY KEY(ts_code,trade_date)) WITHOUT ROWID")
        self.db.execute("CREATE INDEX IF NOT EXISTS bars_date ON bars(trade_date)")
        self.db.execute("CREATE TABLE IF NOT EXISTS days (day TEXT PRIMARY KEY, digest TEXT)")
        self.db.commit()

    def put(self, day, frame):
        f = frame.reindex(columns=BAR_COLS).sort_values("ts_code")
        digest = hashlib.sha256(f.to_csv(index=False).encode()).hexdigest()
        rows = f.astype(object).where(f.notna(), None).itertuples(index=False, name=None)
        with self.db:
            self.db.execute("DELETE FROM bars WHERE trade_date=?", (day,))
            self.db.executemany("INSERT INTO bars VALUES (" + ",".join("?" for _ in BAR_COLS) + ")", rows)
            self.db.execute("INSERT OR REPLACE INTO days VALUES (?,?)", (day, digest))

    def completed(self):
        return {r[0] for r in self.db.execute("SELECT day FROM days")}

    def stock(self, code, start, end):
        f = pd.read_sql_query("SELECT * FROM bars WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
                              self.db, params=(code, ds(start), ds(end)))
        f.index = pd.DatetimeIndex(pd.to_datetime(f.trade_date, format="%Y%m%d"), name="date")
        return f

    def fingerprint(self, start, end):
        rows = list(self.db.execute("SELECT day,digest FROM days WHERE day BETWEEN ? AND ? ORDER BY day", (ds(start), ds(end))))
        return hashlib.sha256(json.dumps(rows).encode()).hexdigest()

    def close(self):
        self.db.close()


class DataClient:
    def __init__(self, token, root, progress=lambda s: None):
        import tushare as ts
        self.pro = ts.pro_api(token.strip(), timeout=25)
        self.root = Path(root)
        self.progress = progress
        self.lock = threading.Lock()
        self.states = {}

    def query(self, endpoint, **kwargs):
        with self.lock:
            state = self.states.setdefault(endpoint, [threading.Lock(), 0.0, 0.36])
        for attempt in range(3):
            with state[0]:
                pause = max(0, state[1] - time.monotonic())
                if pause:
                    time.sleep(pause)
                state[1] = time.monotonic() + state[2]
            try:
                f = self.pro.query(endpoint, **kwargs)
                if not isinstance(f, pd.DataFrame):
                    raise DataError(f"{endpoint} 返回类型异常")
                return f
            except Exception as exc:
                message = str(exc).lower()
                if any(x in message for x in ["每分钟", "频次", "429", "rate limit"]):
                    with state[0]:
                        state[2] = min(2.0, state[2] * 2)
                        state[1] = time.monotonic() + 30
                elif any(x in message for x in ["token", "权限", "积分"]):
                    raise DataError(f"{endpoint} 认证或接口权限不足；请检查Tushare Token和积分") from None
                if attempt == 2:
                    raise DataError(f"{endpoint} 请求失败（{type(exc).__name__}）；缓存已保留") from None
                time.sleep(0.5 * (attempt + 1))

    def paged(self, endpoint, **kwargs):
        parts, hashes = [], set()
        for offset in range(0, 200000, 1000):
            part = self.query(endpoint, limit=1000, offset=offset, **kwargs)
            if part.empty:
                break
            digest = hashlib.sha256(part.to_csv(index=False).encode()).hexdigest()
            if digest in hashes:
                raise DataError(f"{endpoint} 分页重复，无法确认完整性")
            hashes.add(digest)
            parts.append(part)
            if len(part) < 1000:
                break
        else:
            raise DataError(f"{endpoint} 超过分页上限")
        return pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame()

    def metadata(self, name, endpoint, **kwargs):
        path = self.root / "metadata" / (name + ".csv.gz")
        if path.exists() and time.time() - path.stat().st_mtime < 86400:
            try:
                return read_csv(path)
            except (OSError, ValueError):
                pass
        f = self.paged(endpoint, **kwargs)
        if not f.empty:
            atomic_csv(f, path)
        return f

    def universe(self, cfg):
        basic = []
        for status in ["L", "D", "P"]:
            self.progress(f"读取股票名单及退市记录：{status}")
            f = self.metadata("basic_" + status, "stock_basic", list_status=status,
                              fields="ts_code,name,list_date,delist_date")
            if status in ["L", "D"] and f.empty:
                raise DataError("上市或退市名单为空，不能确认历史股票池")
            basic.append(f)
        basic = pd.concat(basic, ignore_index=True).drop_duplicates("ts_code")
        basic = basic[basic.ts_code.str.match(r"^(60|68|00|30)\d{4}\.(SH|SZ)$")].copy()
        classes = self.metadata("sw2021_l1", "index_classify", level="L1", src="SW2021")
        if classes.empty or "index_code" not in classes:
            raise DataError("申万行业目录不可用")
        parts = []
        for row in classes.itertuples():
            for current in ["Y", "N"]:
                self.progress(f"历史行业归属：{row.industry_name} / {current}")
                f = self.metadata(f"members_{row.index_code}_{current}", "index_member_all",
                                  l1_code=row.index_code, is_new=current)
                if current == "Y" and f.empty:
                    raise DataError(f"{row.industry_name} 行业成分缺失")
                parts.append(f)
        member = pd.concat(parts, ignore_index=True).drop_duplicates()
        member = member[member.ts_code.isin(basic.ts_code)].copy()
        if cfg.scope == "科技行业":
            # 同一二级行业整组纳入，板块指数不受个股价格、市值筛选影响。
            labels = member.l2_name.fillna("") + " " + member.l3_name.fillna("")
            mask = member.l1_name.isin(CORE) | labels.str.contains("|".join(TECH_WORDS))
            member = member[member.l2_code.isin(member.loc[mask, "l2_code"].unique())].copy()
        member = normalize_members(member)
        codes = set(member.ts_code)
        self.progress("读取历史股票名称，按当时日期识别ST，不用当前名称过滤过去")
        names = self.metadata("name_history", "namechange",
                              fields="ts_code,name,start_date,end_date,ann_date")
        if names.empty or not {"ts_code", "name", "start_date", "end_date"}.issubset(names):
            raise DataError("历史名称接口不可用，无法可靠排除历史ST股票")
        names = names[names.ts_code.isin(codes)].copy()
        names["start_date"] = dates(names.start_date).values
        names["end_date"] = dates(names.end_date).values
        if names.start_date.isna().any():
            raise DataError("股票历史名称存在缺失生效日")
        return basic[basic.ts_code.isin(codes)].copy(), member, names

    def calendar(self, start, end):
        f = self.query("trade_cal", exchange="SSE", start_date=ds(start), end_date=ds(end), is_open="1")
        if f.empty or "cal_date" not in f:
            raise DataError("交易日历为空")
        return pd.DatetimeIndex(pd.to_datetime(f.cal_date, format="%Y%m%d")).sort_values().unique()

    def endpoint_day(self, endpoint, day):
        path = self.root / "pending" / endpoint / (day + ".csv.gz")
        required = set(FIELDS[endpoint].split(","))
        f = pd.DataFrame()
        if path.exists():
            try:
                f = read_csv(path)
                if not required.issubset(f) or not f.trade_date.astype(str).eq(day).all() or f.ts_code.duplicated().any():
                    f = pd.DataFrame()
            except (OSError, ValueError, EOFError):
                pass
        if f.empty:
            f = self.query(endpoint, trade_date=day, fields=FIELDS[endpoint])
            cap = 5800 if endpoint == "stk_limit" else 6000
            if len(f) >= cap:
                f = self.paged(endpoint, trade_date=day, fields=FIELDS[endpoint])
        if f.empty or not required.issubset(f):
            raise DataError(f"{day} {endpoint} 空表或缺少字段")
        f["trade_date"] = f.trade_date.astype(str)
        if not f.trade_date.eq(day).all() or f.ts_code.duplicated().any():
            raise DataError(f"{day} {endpoint} 日期或主键异常")
        atomic_csv(f, path)
        return f

    def fetch_day(self, day):
        frames = {e: self.endpoint_day(e, day) for e in FIELDS}
        f = frames["daily"]
        f = f[f.ts_code.str.match(r"^(60|68|00|30)\d{4}\.(SH|SZ)$")].copy()
        for e in ["daily_basic", "adj_factor", "stk_limit"]:
            f = f.merge(frames[e], on=["ts_code", "trade_date"], how="left", validate="one_to_one")
        numeric = BAR_COLS[2:]
        f[numeric] = f[numeric].apply(pd.to_numeric, errors="coerce")
        prices = f[["open", "high", "low", "close", "pre_close", "adj_factor"]]
        if f.empty or not np.isfinite(prices).all().all() or not prices.gt(0).all().all():
            for endpoint in FIELDS:
                (self.root / "pending" / endpoint / (day + ".csv.gz")).unlink(missing_ok=True)
            raise DataError(f"{day} 日行情/复权因子缺失或无效，停止而非跳过此日；重试将重新下载")
        # 新股无涨跌停价可能合理；市值/涨跌停价缺失者逐股留痕并禁止买入。
        return f

    def download(self, calendar):
        store = MarketStore(self.root / "market.sqlite")
        todo = [ds(d) for d in calendar if ds(d) not in store.completed()]
        count = len(calendar) - len(todo)
        self.progress(f"行情共{len(calendar)}日，缓存{count}日，待下载{len(todo)}日")
        iterator = iter(todo)
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                pending = {}
                def submit():
                    d = next(iterator, None)
                    if d is not None:
                        pending[pool.submit(self.fetch_day, d)] = d
                for _ in range(4):
                    submit()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        day = pending.pop(future)
                        store.put(day, future.result())
                        for e in FIELDS:
                            (self.root / "pending" / e / (day + ".csv.gz")).unlink(missing_ok=True)
                        count += 1
                        self.progress(f"四路行情下载 {count}/{len(calendar)}，已完成日期可断点续用")
                        submit()
            return store
        except BaseException:
            store.close()
            raise


def normalize_members(raw):
    need = {"ts_code", "l2_code", "l2_name", "in_date", "out_date"}
    if raw.empty or not need.issubset(raw):
        raise DataError("历史行业区间缺失；本程序不退回当前成分回测")
    f = raw.copy()
    f["in_date"] = dates(f.in_date).values
    f["out_date"] = dates(f.out_date).values
    if f.in_date.isna().any() or f.l2_code.isna().any():
        raise DataError("行业纳入日期或二级行业代码缺失")
    f = f.rename(columns={"l2_code": "sector", "l2_name": "sector_name"})
    return f[["ts_code", "sector", "sector_name", "in_date", "out_date"]].drop_duplicates()


def sector_panel(store, member, calendar, cfg):
    """按历史归属构造等权行业日收益；不按当日价格或市值筛选行业成分。"""
    m = member.copy()
    m["in_date"] = m.in_date.dt.strftime("%Y%m%d")
    m["out_date"] = m.out_date.dt.strftime("%Y%m%d").fillna("99991231")
    store.db.execute("DROP TABLE IF EXISTS temp.membership")
    store.db.execute("CREATE TEMP TABLE membership (ts_code TEXT,sector TEXT,sector_name TEXT,in_date TEXT,out_date TEXT)")
    store.db.executemany("INSERT INTO membership VALUES (?,?,?,?,?)", m[["ts_code", "sector", "sector_name", "in_date", "out_date"]].itertuples(index=False, name=None))
    store.db.execute("CREATE INDEX temp.member_key ON membership(ts_code,in_date,out_date)")
    query = """SELECT b.trade_date,b.ts_code,MIN(m.sector) AS sector,
               MIN(m.sector_name) AS sector_name,COUNT(DISTINCT m.sector) AS n_sector,
               b.close/b.pre_close-1 AS ret
               FROM bars b JOIN membership m ON b.ts_code=m.ts_code
               AND b.trade_date>=m.in_date AND b.trade_date<m.out_date
               WHERE b.trade_date BETWEEN ? AND ? AND b.pre_close>0 AND b.vol>0
               GROUP BY b.trade_date,b.ts_code"""
    parts = []
    for f in pd.read_sql_query(query, store.db, params=(ds(calendar[0]), ds(calendar[-1])), chunksize=100000):
        if f.n_sector.gt(1).any():
            raise DataError("同一股票同一天存在冲突行业归属，不能自动选择一个")
        f["up"] = f.ret.gt(0).astype(int)
        parts.append(f.groupby(["trade_date", "sector", "sector_name"], observed=True)
                     .agg(ret_sum=("ret", "sum"), members=("ret", "size"), up_count=("up", "sum")).reset_index())
    if not parts:
        raise DataError("所选股票池没有行业行情")
    p = pd.concat(parts).groupby(["trade_date", "sector", "sector_name"], as_index=False).sum()
    p["date"] = pd.to_datetime(p.trade_date, format="%Y%m%d")
    p["sector_ret1"] = p.ret_sum / p.members
    p["breadth"] = p.up_count / p.members
    out = []
    for sector, f in p.groupby("sector", sort=False):
        f = f.set_index("date").reindex(calendar)
        f.index.name = "date"
        f["sector"] = sector
        f["sector_name"] = f.sector_name.ffill().bfill()  # 名称仅用于显示，不进入评分。
        # 缺行业整日不能填0、不能跨缺失日累计收益。
        logret = np.log1p(f.sector_ret1)
        for n in [2, 3, 5]:
            f[f"sector_ret{n}"] = np.expm1(logret.rolling(n, min_periods=n).sum())
        f["sector_pre3"] = f.sector_ret3.shift(1)
        out.append(f.reset_index())
    p = pd.concat(out, ignore_index=True)
    p["sector_rank"] = np.nan
    valid = p.members.ge(cfg.min_sector_n) & p.sector_ret1.notna()
    ranked = p[valid].sort_values(["date", "sector_ret1", "sector"], ascending=[True, False, True])
    p.loc[ranked.index, "sector_rank"] = ranked.groupby("date").cumcount() + 1
    return p


def max_runup(high, low, window=5):
    """只比较先发生的低价与随后交易日高价，不假定同日高低点顺序。"""
    pairs = [(high / low.shift(lag) - 1).rolling(window + 1 - lag,
             min_periods=window + 1 - lag).max() for lag in range(1, window + 1)]
    return pd.concat(pairs, axis=1).max(axis=1, skipna=False)


def features(stock, calendar):
    f = stock.reindex(calendar).copy()
    f.index.name = "date"
    for x in ["open", "high", "low", "close"]:
        f["a_" + x] = f[x] * f.adj_factor
    c, h, l = f.a_close, f.a_high, f.a_low
    r = c.pct_change(fill_method=None)
    f["ret1"] = f.close / f.pre_close - 1
    f["ret3"], f["ret5"] = c / c.shift(3) - 1, c / c.shift(5) - 1
    f["ret2"] = c / c.shift(2) - 1
    f["pre3"] = (c / c.shift(3) - 1).shift(1)
    f["history_ok"] = c.rolling(21).count().eq(21)
    logr = np.log(c / c.shift(1))
    f["efficiency"] = logr.rolling(5).sum() / logr.abs().rolling(5).sum().replace(0, np.nan)
    draw = [(c / c.shift(k) - 1).rolling(6-k).min() for k in range(1, 6)]
    f["drawdown5"] = pd.concat(draw, axis=1).min(axis=1, skipna=False).clip(upper=0)
    f["low_step"] = (l > l.shift(1)).astype(float).where(l.notna() & l.shift(1).notna()).rolling(5).mean()
    span = (h-l).replace(0, np.nan)
    clv = ((c-l)/span).where(span.notna(), 0.5).where(c.notna()).clip(0, 1)
    f["clv3"] = clv.rolling(3).mean()
    f["near_high"] = c / h.rolling(20).max()
    amount = f.amount.where(f.amount.gt(0))
    f["flow5"] = ((2*clv-1)*amount).rolling(5).sum()/amount.rolling(5).sum()
    up = amount.where(r.gt(0)).rolling(5, min_periods=1).mean()
    down = amount.where(r.lt(0)).rolling(5, min_periods=1).mean()
    f["up_down_amount"] = (up/down).replace([np.inf, -np.inf], np.nan).fillna(1)
    f["amount_ratio"] = amount / amount.shift(1).rolling(20).mean()
    f["volume_support"] = np.log1p(f.amount_ratio) * (2*clv-1)
    f["bias10"] = c/c.rolling(10).mean()-1
    f["runup5"] = max_runup(h, l, 5)
    f["volatility5"] = r.rolling(5).std()
    return f


def assign_sector(index, intervals):
    values = pd.Series(None, index=index, dtype=object)
    for row in intervals.itertuples():
        mask = (index >= row.in_date) & (index < row.out_date if pd.notna(row.out_date) else True)
        if (values.loc[mask].notna() & values.loc[mask].ne(row.sector)).any():
            raise DataError("股票行业区间重叠且分类冲突")
        values.loc[mask] = row.sector
    return values


def name_state(index, names, display_name):
    labels = pd.Series(display_name, index=index, dtype=object)
    known = pd.Series(False, index=index)
    for row in names.sort_values("start_date").itertuples():
        mask = (index >= row.start_date) & (index <= row.end_date if pd.notna(row.end_date) else True)
        labels.loc[mask] = row.name
        known.loc[mask] = True
    bad = labels.astype(str).str.contains(r"ST|退", case=False, regex=True)
    return labels, bad, known


def build_candidates(store, basic, member, names, panel, calendar, cfg, progress=lambda s: None):
    top = panel[panel.sector_rank.eq(1) & panel.date.between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))].copy()
    top_by_day = top.set_index("date").sector
    sector_tables = {s: f.set_index("date") for s, f in panel.groupby("sector", sort=False)}
    groups = {s: f for s, f in member.groupby("ts_code", sort=False)}
    name_groups = {s: f for s, f in names.groupby("ts_code", sort=False)}
    rows = []
    for n, row in enumerate(basic.itertuples(), 1):
        if row.ts_code not in groups:
            continue
        assignment = assign_sector(calendar, groups[row.ts_code])
        mask = assignment.eq(top_by_day.reindex(calendar)) & assignment.notna()
        if not mask.any():
            continue
        stock = store.stock(row.ts_code, calendar[0], calendar[-1])
        if stock.empty:
            continue
        f = features(stock, calendar)
        f["sector"] = assignment
        for col in ["sector_ret1", "sector_ret2", "sector_ret3", "sector_ret5", "sector_pre3", "breadth", "sector_name"]:
            f[col] = np.nan if col != "sector_name" else ""
        for sector in assignment.dropna().unique():
            idx = assignment.index[assignment.eq(sector)]
            p = sector_tables[sector].reindex(idx)
            for col in ["sector_ret1", "sector_ret2", "sector_ret3", "sector_ret5", "sector_pre3", "breadth", "sector_name"]:
                f.loc[idx, col] = p[col].values
        f["rs3"], f["rs5"] = f.ret3-f.sector_ret3, f.ret5-f.sector_ret5
        f["lead_pre3"] = f.pre3-f.sector_pre3
        excess = f.ret1-f.sector_ret1
        f["persistence"] = excess.gt(0).astype(float).where(excess.notna()).rolling(5).mean()
        f["rs1"], f["rs2"] = excess, f.ret2-f.sector_ret2
        for days in [1, 2, 3, 5]:
            f[f"rs_all_{days}"] = consecutive_rs(excess, assignment, days)
        for lag in range(5):
            f[f"excess_d{lag}"] = excess.shift(lag)
        # 信号日超额减去此前3日平均超额；要求四日属于同一行业，防止分类切换伪造提升。
        same_sector = pd.Series(True, index=f.index)
        for lag in [1, 2, 3]:
            same_sector &= assignment.eq(assignment.shift(lag)) & assignment.notna()
        f["rs_accel"] = (excess-excess.shift(1).rolling(3, min_periods=3).mean()).where(same_sector)
        f["signal_limit_ratio"] = f.up_limit/f.pre_close-1
        f["signal_limit_up"] = f.close.ge(f.up_limit-.005).where(f.up_limit.gt(0) & f.close.notna())
        label, st, known = name_state(calendar, name_groups.get(row.ts_code, pd.DataFrame(columns=names.columns)), row.name)
        f["name"], f["is_st"], f["name_known"] = label, st, known
        f["ts_code"] = row.ts_code
        f["board"] = board_name(row.ts_code)
        f["mv_yi"] = f.circ_mv/10000
        f["overheat"] = f.ret5.ge(cfg.heat_limit-1e-12) | f.runup5.ge(cfg.heat_limit-1e-12)
        f["direction_ok"] = f.ret5.gt(0) & (f.rs3.gt(0) | f.rs5.gt(0))
        f["heat_penalty"] = ((f.ret5-0.25)/0.25).clip(0, 1)*10 + ((f.bias10-0.10)/0.15).clip(0, 1)*10
        # 这是风险扣分而非上涨概率，所有常数事前固定、不在回测内训练。
        f = f.loc[mask].copy()
        reasons = pd.Series("", index=f.index)
        gates = [
            (~f.history_ok, "不足21根连续日线"),
            (~f.name_known, "历史名称缺失"),
            (f.is_st, "当时为ST/退市整理"),
            (~f.close.ge(cfg.min_price), "价格不足或缺失"),
            (~f.mv_yi.between(cfg.min_mv, cfg.max_mv), "流通市值越界或缺失"),
            (~f.up_limit.gt(0) | ~f.down_limit.gt(0), "涨跌停价缺失"),
            (~f.sector_ret1.gt(0), "第一板块未上涨"),
            (~f.direction_ok, "近期未上涨或未跑赢板块"),
        ]
        for invalid, reason in gates:
            reasons.loc[invalid] += reason + "；"
        f["base_reason"] = reasons
        f["exclude_reason"] = reasons + np.where(f.overheat, "近5日暴涨达到阈值；", "")
        keep = ["ts_code", "name", "sector", "sector_name", "close", "mv_yi", "ret1", "ret3", "ret5",
                "rs3", "rs5", "lead_pre3", "persistence", "efficiency", "drawdown5", "low_step", "flow5",
                "up_down_amount", "volume_support", "amount_ratio", "clv3", "near_high", "bias10", "runup5",
                "volatility5", "sector_ret1", "breadth", "turnover_rate", "heat_penalty", "overheat",
                "base_reason", "exclude_reason", "board", "ret2", "rs1", "rs2"]
        keep += [f"rs_all_{n}" for n in [1, 2, 3, 5]] + [f"excess_d{n}" for n in range(5)]
        keep += ["pre3", "rs_accel", "signal_limit_up", "signal_limit_ratio"]
        rows.append(f[keep].reset_index())
        if n % 50 == 0:
            progress(f"计算四维动量 {n}/{len(basic)} 只，按股票读取行情")
    if not rows:
        raise DataError("所选日期没有可审计的候选记录，请检查行业归属和数据范围")
    candidates = pd.concat(rows, ignore_index=True)
    return score_candidates(candidates)


def score_candidates(candidates):
    f = candidates.copy()
    eligible = f.base_reason.eq("")
    g = f.loc[eligible].copy()
    dimensions = {
        "strength_score": ["rs3", "rs5", "lead_pre3", "persistence"],
        "efficiency_score": ["efficiency", "drawdown5", "low_step"],
        "volume_score": ["flow5", "up_down_amount", "volume_support"],
        "position_score": ["clv3", "near_high"],
    }
    for target, inputs in dimensions.items():
        ranks = []
        for col in inputs:
            grouped = g.groupby("date")[col]
            ranks.append(100*(grouped.rank(method="average")-0.5)/grouped.transform("count"))
        g[target] = pd.concat(ranks, axis=1).mean(axis=1, skipna=False)
        f[target] = np.nan
        f.loc[g.index, target] = g[target]
    f["score"] = f[list(dimensions)].mean(axis=1, skipna=False)-f.heat_penalty
    invalid = eligible & f.score.isna()
    f.loc[invalid, "base_reason"] += "评分字段缺失；"
    f.loc[invalid, "exclude_reason"] += "评分字段缺失；"
    f["max_buy_reference"] = np.nan  # 报告时按固定配置填写；不是对未来开盘的预测。
    return f


def efficiency_annotations(candidates):
    c = candidates
    valid = c.base_reason.fillna('').eq('') & c.overheat.eq(False)
    values = pd.to_numeric(c.efficiency_score, errors='coerce')
    values = values.where(valid & np.isfinite(values))
    g = values.groupby(c.date)
    n = g.transform('count')
    spread = g.transform('max')-g.transform('min')
    pct = (g.rank(method='average')-.5)/n*100
    pct = pct.where(n.ge(4) & spread.gt(1e-12))
    keep = pct.isna() | pct.le(EFF_CUTOFF)
    reason = np.select([pct.isna(), keep], ['分位不可评估，保留', '未超过最高三分之一，保留'],
                       default='效率分位处于最高三分之一，拒绝且不补位')
    return pd.DataFrame(dict(eff_pool_count=n, eff_pct=pct, eff_keep=keep, eff_reason=reason), index=c.index)


def efficiency_decisions(candidates, selected):
    base = selected[selected.group.eq(TOP10_MODELS[0][1])].copy()
    keys = ['date', 'ts_code']
    annotations = candidates[keys].join(efficiency_annotations(candidates))
    annotations['date'] = pd.to_datetime(annotations.date)
    base['date'] = pd.to_datetime(base.date)
    base = base.drop(columns=['eff_pool_count', 'eff_pct', 'eff_keep', 'eff_reason'], errors='ignore')
    if base.duplicated(keys).any() or annotations.duplicated(keys).any():
        raise DataError('效率过滤的股票日记录重复')
    base = base.merge(annotations, on=keys, how='left', validate='one_to_one', indicator=True)
    if base._merge.ne('both').any():
        raise DataError('效率过滤缺少对应候选特征')
    return base.drop(columns='_merge')


def efficiency_sample_phase(values, mode=''):
    d = pd.to_datetime(values)
    if '演示' in mode:
        return pd.Series('合成演示_不是验证', index=d.index)
    return pd.Series(np.select([d.between(pd.Timestamp('20240920'), pd.Timestamp('20260918')),
        d.gt(pd.Timestamp(EFF_FREEZE_DAY)), d.lt(pd.Timestamp('20240920'))],
        ['M16已参与假设形成_探索', 'M16冻结后日期_须核实事前留存', '更早历史_须排除既往调参'],
        default='M16冻结边界_不作独立验证'), index=d.index)


def efficiency_protocol(cfg):
    logic = '\n'.join(inspect.getsource(f) for f in [efficiency_annotations, efficiency_decisions,
        select_candidates, simulate_trade, run_trades, replay_independent_paths])
    return dict(protocol_id='M16_PRE3_DROP_HIGH_EFF', frozen_on=EFF_FREEZE_DAY,
        base_group=TOP10_MODELS[0][1], trial_group=EFF_GROUP, primary_target=.10, sensitivity_target=.05,
        threshold='eff_pct > 200/3', minimum_finite_pool=4, missing_percentile='保留', ties='平均名次',
        no_replacement=True, rank_preserved=True, scope='当日基础合格且不过热的完整候选池',
        params={k: getattr(cfg, k) for k in FROZEN_PARAMS},
        logic_sha256=hashlib.sha256(logic.encode()).hexdigest(),
        explored_signal_start='20240920', explored_signal_end='20260918',
        interval_labels_are_not_out_of_sample_proof=True, automatic_best_model_selection=False,
        note='先按原低前涨幅规则选最多3只，再过滤；原名次保留，不加仓其他股票。')


REPLAY_BASE_COLUMNS = ['signal_date', 'ts_code', 'name', 'sector', 'sector_name', 'group',
    'take_profit', 'rank', 'score', 'rs5', 'runup5', 'experiment', 'family', 'gap_policy',
    'rs_days', 'board', 'gap_limit', 'return_rank']


def replay_independent_paths(events):
    """复用各信号原成交路径，按股票重放持仓阻塞；不重撮合价格，也不忽略原来被阻塞的信号。"""
    if events.empty:
        return events.copy()
    f = events.copy()
    for col in ['signal_date', 'entry_date', 'exit_date']:
        f[col] = pd.to_datetime(f[col])
    keys = ['group', 'ts_code', 'take_profit', 'signal_date']
    if f.duplicated(keys).any():
        raise DataError('持仓约束重放存在重复独立路径')
    if not f.status.isin(['已平仓', '未平仓', '未买入', '待下一交易日']).all():
        raise DataError('必须使用完整独立路径，不能拿已跳过的实际成交记录代替')
    records = []
    for _, block in f.groupby(['group', 'ts_code', 'take_profit'], sort=False):
        busy_until = pd.Timestamp.min
        for row in block.sort_values('signal_date').to_dict('records'):
            entry = row['entry_date']
            if row['status'] in ['已平仓', '未平仓'] and pd.isna(entry):
                raise DataError('持仓路径缺少买入日期')
            if row['status'] == '已平仓' and (pd.isna(row['exit_date']) or row['exit_date'] < entry):
                raise DataError('已平仓路径的退出日期无效')
            if pd.notna(entry) and entry <= busy_until:
                result = {k: row[k] for k in REPLAY_BASE_COLUMNS if k in row}
                result.update(status='重复持仓跳过', entry_date=entry, exit_date=pd.NaT,
                    net_return=np.nan, reason='同组同止盈版本仍持有；退出当日不重复开仓')
            else:
                result = {k: v for k, v in row.items() if k != 'event_mode'}
                if row['status'] in ['已平仓', '未平仓']:
                    busy_until = row['exit_date'] if row['status'] == '已平仓' else pd.Timestamp.max
            records.append(result)
    return pd.DataFrame(records)


def assert_replay_matches(original, replayed):
    keys = ['group', 'ts_code', 'take_profit', 'signal_date']
    cols = keys+['status', 'entry_date', 'exit_date', 'reason', 'net_return']
    frames = []
    for source in [original, replayed]:
        f = source.reindex(columns=cols).copy()
        for col in ['signal_date', 'entry_date', 'exit_date']:
            f[col] = pd.to_datetime(f[col])
        frames.append(f.sort_values(keys).reset_index(drop=True))
    try:
        pd.testing.assert_frame_equal(*frames, check_dtype=False, atol=1e-10, rtol=1e-10)
    except AssertionError as exc:
        raise DataError('独立路径不能复现报告中的原持仓记录，停止效率过滤对照') from exc


def efficiency_execution_changes(base, trial, decisions):
    keys = ['signal_date', 'ts_code', 'take_profit']
    def entered(f):
        return f[f.status.isin(['已平仓', '未平仓'])].copy()
    a, b = entered(base), entered(trial)
    before = {tuple(r[k] for k in keys): r for r in a.to_dict('records')}
    after = {tuple(r[k] for k in keys): r for r in b.to_dict('records')}
    keep = {(pd.Timestamp(r.date), r.ts_code): bool(r.eff_keep) for r in decisions.itertuples()}
    rows = []
    for key in sorted(set(before) | set(after)):
        if key in before and key in after:
            category, source, side = '共同成交', after[key], '过滤组'
        elif key in after:
            category, source, side = '持仓路径新增', after[key], '过滤组'
        else:
            category = '持仓路径减少' if keep[(pd.Timestamp(key[0]), key[1])] else '直接过滤'
            source, side = before[key], '原低前涨幅'
        rows.append({**source, 'change_type': category, 'reference_side': side})
    return pd.DataFrame(rows)


def efficiency_trial(candidates, selected, events, trades, cfg, mode=''):
    if not cfg.efficiency_research or selected.empty or events.empty:
        return {}
    decisions = efficiency_decisions(candidates, selected)
    if decisions.empty:
        return {}
    base_group = TOP10_MODELS[0][1]
    e = events[events.group.eq(base_group)].copy()
    if e.empty:
        raise DataError('效率过滤缺少原低前涨幅组的独立路径')
    for col in ['signal_date', 'entry_date', 'exit_date']:
        e[col] = pd.to_datetime(e[col])
    d = decisions.rename(columns={'date': 'signal_date'})
    keys = ['signal_date', 'ts_code', 'take_profit']
    expected = d[['signal_date', 'ts_code']].merge(pd.DataFrame({'take_profit': [.05, .10]}), how='cross')
    if e.duplicated(keys).any() or set(map(tuple, e[keys].to_numpy())) != set(map(tuple, expected[keys].to_numpy())):
        raise DataError('效率过滤独立路径不完整，不能把缺记录当空仓')
    audit = e.merge(d[['signal_date', 'ts_code', 'eff_pool_count', 'eff_pct', 'eff_keep', 'eff_reason']],
                    on=['signal_date', 'ts_code'], how='left', validate='many_to_one')
    trial_events = audit[audit.eff_keep].copy()
    trial_events['group'], trial_events['experiment'], trial_events['family'] = EFF_GROUP, EFF_EXPERIMENT, '效率过滤对照'
    base_actual = replay_independent_paths(e)
    original = trades[trades.group.eq(base_group)].copy()
    assert_replay_matches(original, base_actual)
    trial_actual = replay_independent_paths(trial_events)
    if trades.group.eq(EFF_GROUP).any():
        assert_replay_matches(trades[trades.group.eq(EFF_GROUP)], trial_actual)
    actual = pd.concat([base_actual, trial_actual], ignore_index=True)
    audit['closed_known'] = audit.status.eq('已平仓') & ~audit.data_issue.eq(True) & audit.net_return.notna()
    summaries = []
    for tp, block in audit.groupby('take_profit'):
        known = block[block.closed_known]
        retained = known[known.eff_keep]; rejected = known[~known.eff_keep]
        n = len(known)
        avoided = -rejected.net_return.clip(upper=0).sum()/n*100 if n else np.nan
        missed = rejected.net_return.clip(lower=0).sum()/n*100 if n else np.nan
        summaries.append(dict(group=EFF_GROUP, take_profit=tp, **{
            '原始信号数': len(block), '过滤信号数': int((~block.eff_keep).sum()),
            '分位缺失保留数': int(block.eff_pct.isna().sum()), '原始有效平仓机会数': n,
            '未买入数': int((block.status.eq('未买入') & ~block.data_issue.eq(True)).sum()),
            '未知或数据问题数': int((~block.closed_known & ~(block.status.eq('未买入') & ~block.data_issue.eq(True))).sum()),
            '保留平仓数': len(retained), '过滤平仓数': len(rejected),
            '避免亏损笔数': int(rejected.net_return.lt(0).sum()), '错过盈利笔数': int(rejected.net_return.gt(0).sum()),
            '原始机会净均益%': known.net_return.mean()*100, '保留单笔净均益%': retained.net_return.mean()*100,
            '过滤后等机会均益%': retained.net_return.sum()/n*100 if n else np.nan,
            '等机会改善(百分点)': avoided-missed, '避免亏损贡献(百分点)': avoided,
            '错过盈利贡献(百分点)': missed}))
    temporary = pd.concat([e.assign(group=GROUP_BASE), trial_events], ignore_index=True)
    paired, _ = paired_top10_days(temporary, top10_candidates(candidates, cfg), cfg, groups=[GROUP_BASE, EFF_GROUP])
    paired['sample_phase'] = efficiency_sample_phase(paired.signal_date, mode)
    actual_summary = summarize(actual, ['group', 'take_profit'])
    grid = pd.MultiIndex.from_product([[base_group, EFF_GROUP], [.05, .10]], names=['group', 'take_profit']).to_frame(index=False)
    actual_summary = grid.merge(actual_summary, how='left', on=['group', 'take_profit'])
    for col in ['信号数', '成交数', '有效平仓数', '未平仓数', '数据问题数', '双触及笔数']:
        actual_summary[col] = actual_summary[col].fillna(0).astype(int)
    tables = {'efficiency_decisions': decisions, 'efficiency_opportunities': audit,
        'efficiency_opportunity_summary': pd.DataFrame(summaries), 'efficiency_actual_trades': actual,
        'efficiency_actual_summary': actual_summary,
        'efficiency_paired_days': paired}
    for key, value in concentration_tables(paired).items():
        tables[key.replace('validation_', 'efficiency_')] = value
    actual['month'] = pd.to_datetime(actual.signal_date).dt.strftime('%Y-%m')
    tables['efficiency_actual_monthly'] = summarize(actual, ['group', 'take_profit', 'month'])
    changes = efficiency_execution_changes(base_actual, trial_actual, decisions)
    tables['efficiency_execution_changes'] = changes
    tables['efficiency_execution_summary'] = summarize(changes, ['change_type', 'reference_side', 'take_profit']) if not changes.empty else pd.DataFrame()
    return tables


def select_candidates(candidates, cfg):
    candidates = add_filter_ranks(candidates)
    efficiency = efficiency_annotations(candidates) if cfg.efficiency_research else None
    choices = []
    for spec in experiments(cfg):
        mask = candidates.base_reason.eq("") & candidates.overheat.eq(spec["hot"])
        if spec["rs_days"]:
            mask &= candidates[f"rs_all_{spec['rs_days']}"]
        f = candidates[mask].copy()
        f["base_score"] = f.score
        if spec["weights"] != (25, 25, 25, 25):
            f["score"] = sum(f[col]*weight/100 for col, weight in zip(SCORE_COLS, spec["weights"]))-f.heat_penalty
        # 先在同一个基础候选池评分，再加RS条件；不在筛选后重新计算百分位。
        f["eligible_pool_n"] = f.groupby("date").ts_code.transform("size")
        raw = f.sort_values(["date", "ret1", "ts_code"], ascending=[True, False, True])
        f["return_rank"] = (raw.groupby("date").cumcount()+1).reindex(f.index)
        if spec.get("top10"):
            f = f[f.return_rank.le(10)].copy()
            f["top10_pool_n"] = f.groupby("date").ts_code.transform("size")
            if spec["sort"] not in f:
                f[spec["sort"]] = np.nan
            f = f[np.isfinite(pd.to_numeric(f[spec["sort"]], errors="coerce"))].copy()
            # 两条假设各自独立；相同指标时用原涨幅排名，避免混入另一评分。
            f = f.sort_values(["date", spec["sort"], "return_rank", "ts_code"],
                              ascending=[True, spec["ascending"], True, True])
        else:
            f = f.sort_values(["date", spec["sort"], "ts_code"], ascending=[True, False, True])
        f["rank"] = f.groupby("date").cumcount()+1
        f = f[f["rank"].le(cfg.top_n)].copy()
        f = apply_filter_rule(f, spec)
        f = f[f.filter_pass].copy()  # 排名后过滤，不补位、不把第4名提到前三。
        if spec.get('efficiency_guard'):
            f = f.join(efficiency)
            f = f[f.eff_keep].copy()
        for key in ["group", "experiment", "family", "gap_policy", "rs_days", "filter_col", "filter_label", "filter_cutoff"]:
            f[key] = spec[key]
        # 空候选时，pandas 3的字符串map保留字符串dtype，显式转换保证后续价格计算有效。
        f["gap_limit"] = f.ts_code.map(lambda code: buy_gap(code, spec["gap_policy"], cfg)).astype(float)
        f["max_buy_reference"] = f.close*(1+f.gap_limit)
        choices.append(f)
    nonempty = [f for f in choices if not f.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else choices[0].iloc[:0].copy()


def finite_positive(*values):
    return all(pd.notna(x) and np.isfinite(float(x)) and float(x) > 0 for x in values)


def stamp_tax(day):
    # 卖出印花税历史口径；其他手续费以用户输入的双边费率计。
    return 0.0005 if pd.Timestamp(day) >= pd.Timestamp("2023-08-28") else 0.001


def simulate_trade(stock, calendar, signal, take_profit, cfg):
    """保守日线撮合。未卖出的记录不以0收益伪装成已完成交易。"""
    signal = pd.Timestamp(signal)
    pos = int(calendar.searchsorted(signal, side="right"))
    result = dict(status="待下一交易日", entry_date=pd.NaT, exit_date=pd.NaT,
                  entry_price=np.nan, exit_price=np.nan, gap=np.nan, net_return=np.nan,
                  gross_return=np.nan, mark_return=np.nan, mark_date=pd.NaT,
                  hold_days=np.nan, reason="", ambiguous=False, deferred=False,
                  data_issue=False, horizon_complete=False, buy_day_hit5=False,
                  sellable_mfe5=np.nan, mae5=np.nan)
    for n in range(1, 6):
        result[f"close_return_d{n}"] = np.nan
    if pos >= len(calendar):
        return result
    entry_day = calendar[pos]
    result["entry_date"] = entry_day
    if entry_day not in stock.index:
        result.update(status="未买入", reason="次日停牌或无开盘行情")
        return result
    b = stock.loc[entry_day]
    if not finite_positive(b.open, b.adj_factor, b.pre_close, b.up_limit, b.down_limit):
        result.update(status="未买入", reason="开盘或涨跌停数据缺失", data_issue=True)
        return result
    gap = float(b.open/b.pre_close-1)
    result["gap"] = gap
    # 仅依据开盘时已知价格判断。不能用当日最终高低价决定能否在开盘买入。
    if b.open >= b.up_limit-0.005 or b.open <= b.down_limit+0.005:
        result.update(status="未买入", reason="开盘处于涨跌停，跳过竞价排队")
        return result
    entry_raw = float(b.open*(1+cfg.slippage))
    if entry_raw > b.pre_close*(1+cfg.max_gap)+1e-9:
        result.update(status="未买入", reason="含滑点买价超过预设最高买价")
        return result
    if entry_raw >= b.up_limit-0.005:
        result.update(status="未买入", reason="含滑点买价触及涨停")
        return result
    entry = entry_raw*float(b.adj_factor)
    result.update(status="未平仓", entry_price=entry_raw)
    upper, lower = entry*(1+take_profit), entry*(1-cfg.stop_loss)
    eps = entry*1e-12  # 消除恰好触及目标价时的浮点误差，不扩大一个价格档位。
    pending = ""
    path = stock.reindex(calendar[pos:pos+5])
    adjusted = path[["open", "high", "low", "close"]].mul(path.adj_factor, axis=0)
    result["horizon_complete"] = len(path) == 5 and adjusted.notna().all().all()
    result["buy_day_hit5"] = bool(adjusted.high.iloc[0]/entry-1 >= 0.05-1e-12)
    if len(path) > 1 and adjusted.high.iloc[1:].notna().any():
        # 只是一段完整观察窗口内的日线最高价，不是策略兑现收益。
        result["sellable_mfe5"] = float(adjusted.high.iloc[1:].max()/entry-1)
    if adjusted.low.notna().any():
        result["mae5"] = float(adjusted.low.min()/entry-1)
    for i, value in enumerate(adjusted.close, 1):
        if pd.notna(value):
            result[f"close_return_d{i}"] = float(value/entry-1)

    def finish(day, raw_price, factor, reason, held):
        # 比例费用模型；默认买卖手续费各万3，滑点各千1，卖出另计历史印花税。
        gross = raw_price*factor/entry-1
        net = (1+gross)*(1-cfg.sell_fee-stamp_tax(day))/(1+cfg.buy_fee)-1
        result.update(status="已平仓", exit_date=day, exit_price=float(raw_price),
                      net_return=float(net), gross_return=float(gross), hold_days=held, reason=reason)

    for k, day in enumerate(calendar[pos:]):
        held = k+1
        if day not in stock.index:
            if held >= cfg.hold_days:
                pending = pending or "到期延迟退出"
                result["deferred"] = True
            continue
        bar = stock.loc[day]
        if finite_positive(bar.close, bar.adj_factor):
            result["mark_return"] = float(bar.close*bar.adj_factor/entry-1)
            result["mark_date"] = day
        if k == 0:
            continue  # T+1：买入当天任何止盈或止损触发都不可卖出。
        if not finite_positive(bar.open, bar.high, bar.low, bar.close, bar.adj_factor, bar.down_limit):
            result["data_issue"] = True
            # 行情缺失不能推断当天是否触发；该笔最终结果仍标记数据不完整。
            if held >= cfg.hold_days:
                pending = pending or "到期延迟退出"
            continue
        factor = float(bar.adj_factor)
        o, h, l = bar.open*factor, bar.high*factor, bar.low*factor
        if bar.open <= bar.down_limit+0.005:
            # 日线无法证明开盘跌停排队成交，即使后来打开也不回填开盘成交。
            if o <= lower+eps:
                pending = "止损延迟退出"
            elif held >= cfg.hold_days:
                pending = pending or "到期延迟退出"
            result["deferred"] = True
            continue
        if pending:
            sell = bar.open*(1-cfg.slippage)
            if sell <= bar.down_limit+0.005:
                result["deferred"] = True
                continue
            finish(day, sell, factor, pending, held)
            break
        if o <= lower+eps or o >= upper-eps:
            reason = "跳空止损" if o <= lower+eps else "开盘止盈"
            sell = bar.open*(1-cfg.slippage)
        else:
            stop_hit, profit_hit = l <= lower+eps, h >= upper-eps
            if stop_hit and profit_hit:
                result["ambiguous"] = True
            if stop_hit:
                sell, reason = lower/factor*(1-cfg.slippage), "止损"
            elif profit_hit:
                # 止盈按盘中目标价减滑点；真实限价委托可能有不同成交结果。
                sell, reason = upper/factor*(1-cfg.slippage), "止盈"
            elif held >= cfg.hold_days:
                sell, reason = bar.close*(1-cfg.slippage), "到期退出"
            else:
                continue
        if sell <= bar.down_limit+0.005:
            pending = "止损延迟退出" if "止损" in reason else "到期延迟退出"
            result["deferred"] = True
            continue
        finish(day, sell, factor, reason, held)
        break
    if result["status"] == "未平仓":
        result["reason"] = pending or "观察截止时尚未满足退出条件"
        result["hold_days"] = len(calendar)-pos
    return result


def run_trades(store, selected, calendar, cfg, progress=lambda s: None, event_records=None, observations=None):
    records = []
    inputs = selected
    if observations is not None and not observations.empty:
        inputs = pd.concat([selected, observations], ignore_index=True)
    if inputs.empty:
        return pd.DataFrame()
    for n, (code, f) in enumerate(inputs.groupby("ts_code", sort=False), 1):
        stock = store.stock(code, calendar[0], calendar[-1])
        cached_paths = {}
        for group, signals in f.groupby("group", sort=False):
            for tp in [0.05, 0.10]:
                busy_until = pd.Timestamp.min
                for row in signals.sort_values("date").itertuples():
                    entry_pos = calendar.searchsorted(row.date, side="right")
                    planned = calendar[entry_pos] if entry_pos < len(calendar) else pd.NaT
                    base = dict(signal_date=row.date, ts_code=code, name=row.name, sector=row.sector,
                                sector_name=row.sector_name, group=group, take_profit=tp, rank=row.rank,
                                score=row.score, rs5=row.rs5, runup5=row.runup5,
                                experiment=row.experiment, family=row.family, gap_policy=row.gap_policy,
                                rs_days=row.rs_days, board=board_name(code), gap_limit=row.gap_limit)
                    base["return_rank"] = getattr(row, "return_rank", np.nan)
                    cache_key = (row.date, tp, row.gap_limit)
                    # 相同股票、信号日、买价上限和止盈只撮合一次，各实验独立管理持仓。
                    if cache_key not in cached_paths:
                        cached_paths[cache_key] = simulate_trade(stock, calendar, row.date, tp,
                                                               replace(cfg, max_gap=row.gap_limit))
                    path = cached_paths[cache_key]
                    if event_records is not None:
                        # 用全部候选信号的假设入场检查选股惯性，不受早先持仓阻塞影响。
                        event = {**base, **path}
                        event["event_mode"] = "独立信号观察_允许重叠_不是组合收益"
                        event_records.append(event)
                    if row.experiment == "OBS_TOP10":
                        continue  # 前十是完整事件观察，绝不伪装成买十只的持仓策略。
                    if pd.notna(planned) and planned <= busy_until:
                        result = dict(status="重复持仓跳过", entry_date=planned, exit_date=pd.NaT,
                                      net_return=np.nan, reason="同组同止盈版本仍持有；退出当日不重复开仓")
                    else:
                        result = path.copy()
                        if result["status"] in ["已平仓", "未平仓"]:
                            busy_until = result["exit_date"] if result["status"] == "已平仓" else pd.Timestamp.max
                    records.append({**base, **result})
        if n % 50 == 0:
            progress(f"回放成交路径 {n}/{inputs.ts_code.nunique()} 只")
    return pd.DataFrame(records)


def summarize(trades, keys):
    columns = keys + ["信号数", "成交数", "有效平仓数", "未平仓数", "数据问题数", "胜率%", "净均益%",
                      "净中位数%", "平均盈利%", "平均亏损%", "盈亏金额比", "止盈占比%", "止损占比%",
                      "到期占比%", "平均持有日", "双触及笔数", "最差单笔%", "剔除最大5笔后净均益%"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for key, f in trades.groupby(keys, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        issue = f.get("data_issue", pd.Series(False, index=f.index)).eq(True)
        closed = f[f.status.eq("已平仓") & ~issue]
        r = pd.to_numeric(closed.net_return, errors="coerce").dropna()
        wins, losses = r[r > 0], r[r < 0]
        count = len(r)
        row = dict(zip(keys, key))
        row.update({"信号数": len(f), "成交数": int(f.status.isin(["已平仓", "未平仓"]).sum()),
                    "有效平仓数": count, "未平仓数": int(f.status.eq("未平仓").sum()), "数据问题数": int(issue.sum()),
                    "胜率%": float((r > 0).mean()*100) if count else np.nan,
                    "净均益%": r.mean()*100, "净中位数%": r.median()*100,
                    "平均盈利%": wins.mean()*100, "平均亏损%": losses.mean()*100,
                    "盈亏金额比": wins.sum()/(-losses.sum()) if len(losses) else np.nan,
                    "止盈占比%": closed.reason.str.contains("止盈").mean()*100,
                    "止损占比%": closed.reason.str.contains("止损").mean()*100,
                    "到期占比%": closed.reason.str.contains("到期").mean()*100,
                    "平均持有日": closed.hold_days.mean(),
                    "双触及笔数": int(closed.ambiguous.eq(True).sum()),
                    "最差单笔%": r.min()*100,
                    "剔除最大5笔后净均益%": r.sort_values().iloc[:-5].mean()*100 if count > 5 else np.nan})
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def experiment_design(cfg):
    rows = []
    for spec in experiments(cfg):
        row = {k: spec[k] for k in ["group", "experiment", "family", "gap_policy", "rs_days", "filter_col", "filter_label", "filter_cutoff"]}
        row.update(dict(zip(["相对强度权重", "上涨效率权重", "量价配合权重", "收盘位置权重"], spec["weights"])))
        row.update({"排名后效率过滤": bool(spec.get("efficiency_guard")),
                    "候选范围": "基础合格且不过热的涨幅前十" if spec.get("top10") else "原基础池",
                    "排序指标": spec["sort"], "排序方向": "由小到大" if spec.get("ascending") else "由大到小",
                    "综合评分参与排序": spec["sort"] == "score"})
        row.update({"主板买价上限%": cfg.max_gap*100,
                    "双创买价上限%": (cfg.growth_gap if spec["gap_policy"] == "分板块上限" else cfg.max_gap)*100})
        rows.append(row)
    return pd.DataFrame(rows)


def experiment_summary(trades, selected, calendar, cfg):
    design = experiment_design(cfg)
    days = int(((calendar >= pd.Timestamp(cfg.start)) & (calendar <= pd.Timestamp(cfg.end))).sum())
    coverage = []
    for spec in experiments(cfg):
        f = selected[selected.group.eq(spec["group"])]
        coverage.append(dict(group=spec["group"], **{"区间交易日": days, "选股日数": f.date.nunique(),
            "选股覆盖%": 100*f.date.nunique()/days if days else np.nan,
            "合格候选股日数": f.drop_duplicates("date").eligible_pool_n.sum(), "入选信号数": len(f)}))
    grid = design.merge(pd.DataFrame(coverage), on="group")
    grid = grid.merge(pd.DataFrame({"take_profit": [.05, .10]}), how="cross")
    summary = summarize(trades, ["group", "take_profit"])
    grid = grid.merge(summary, on=["group", "take_profit"], how="left")
    for col in ["信号数", "成交数", "有效平仓数", "未平仓数", "数据问题数"]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce").fillna(0).astype(int)
    base = grid[grid.experiment.eq("B0")][["gap_policy", "take_profit", "净均益%"]].rename(columns={"净均益%": "同买价等权净均益%"})
    grid = grid.merge(base, on=["gap_policy", "take_profit"], how="left")
    grid["相对等权均益差(百分点)"] = grid["净均益%"]-grid["同买价等权净均益%"]
    simple = grid[grid.experiment.eq("RETURN")][["gap_policy", "take_profit", "净均益%"]].rename(columns={"净均益%": "同买价涨幅基准净均益%"})
    grid = grid.merge(simple, on=["gap_policy", "take_profit"], how="left")
    grid["相对涨幅均益差(百分点)"] = grid["净均益%"]-grid["同买价涨幅基准净均益%"]
    return grid


def gap_comparison(trades):
    """同一信号在两种买价规则下的新增、共同与持仓路径变化，禁止只展示新增赢家。"""
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["experiment", "signal_date", "ts_code", "take_profit"]
    a = trades[trades.gap_policy.eq("统一上限")].set_index(keys)
    b = trades[trades.gap_policy.eq("分板块上限")].set_index(keys)
    common = a.index.intersection(b.index)
    rows = []
    for key in common:
        old, new = a.loc[key], b.loc[key]
        old_fill, new_fill = old.status in ["已平仓", "未平仓"], new.status in ["已平仓", "未平仓"]
        if old_fill and new_fill:
            label, source = "共同成交", new
        elif new_fill:
            direct = old.status == "未买入" and old.reason == "含滑点买价超过预设最高买价"
            label, source = ("放宽新增_原因高开" if direct else "持仓路径变化新增"), new
        elif old_fill:
            label, source = "持仓路径变化减少", old
        else:
            continue
        row = {**source.to_dict(), **dict(zip(keys, key)), "increment_type": label,
               "uniform_status": old.status, "split_status": new.status,
               "uniform_reason": old.reason, "split_reason": new.reason}
        rows.append(row)
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        return ledger, pd.DataFrame()
    return ledger, summarize(ledger, ["experiment", "increment_type", "board", "take_profit"])


def path_summary(events):
    """不去重的独立信号5日路径，仅衡量入场后的惯性，不能解读为可实现利润。"""
    if events.empty:
        return pd.DataFrame()
    # 两档止盈拥有相同固定观察路径，只取5%版本一次，防止把同一信号重复统计。
    events = events[events.take_profit.eq(.05)]
    rows = []
    for group, f in events.groupby("group", sort=False):
        entered = f[f.entry_price.notna()]
        full = entered[entered.horizon_complete.eq(True) & ~entered.data_issue.eq(True)]
        valid = len(full)
        row = dict(group=group, **{"独立信号数": len(f), "可模拟入场数": len(entered), "完整5日观察数": valid,
                   "观察不完整或数据问题数": len(entered)-valid,
                   "可卖日期曾达5%比例": full.sellable_mfe5.ge(.05-1e-12).mean()*100 if valid else np.nan,
                   "可卖日期曾达10%比例": full.sellable_mfe5.ge(.10-1e-12).mean()*100 if valid else np.nan,
                   "仅买入当天达5%比例": (full.buy_day_hit5.eq(True)&full.sellable_mfe5.lt(.05-1e-12)).mean()*100 if valid else np.nan,
                   "平均窗口最大下探%": full.mae5.mean()*100,
                   "第2日收盘平均涨幅%": full.close_return_d2.mean()*100,
                   "第3日收盘平均涨幅%": full.close_return_d3.mean()*100,
                   "第5日收盘平均涨幅%": full.close_return_d5.mean()*100})
        rows.append(row)
    return pd.DataFrame(rows)


def filter_impact_summary(ledger, by=()):
    """固定基准已平仓机会集合；被过滤票据贡献0，未完成结果仍是未知。"""
    if ledger.empty:
        return pd.DataFrame()
    keys = ["group", "take_profit"]+list(by)
    rows = []
    for key, f in ledger.groupby(keys, sort=False, dropna=False):
        issue = f.data_issue.eq(True)
        closed = f[f.status.eq("已平仓") & ~issue & f.net_return.notna()]
        kept, veto = closed[closed.filter_pass], closed[~closed.filter_pass]
        n = len(closed)
        avoided = -veto.loc[veto.net_return.lt(0), "net_return"].sum()
        missed = veto.loc[veto.net_return.gt(0), "net_return"].sum()
        row = dict(zip(keys, key))
        row.update({"原始信号数": len(f), "规则保留数": int(f.filter_pass.sum()),
                    "规则拒绝数": int((~f.filter_pass).sum()), "信号保留率%": f.filter_pass.mean()*100,
                    "小候选池保留数": int((f.filter_pool_n.lt(4)&f.filter_pass).sum()),
                    "未成交数": int(f.status.eq("未买入").sum()),
                    "待买入或未平仓数": int(f.status.isin(["待下一交易日", "未平仓"]).sum()),
                    "数据问题数": int(issue.sum()), "有效平仓机会数": n,
                    "保留平仓数": len(kept), "拒绝平仓数": len(veto),
                    "避免亏损笔数": int(veto.net_return.lt(0).sum()),
                    "错过盈利笔数": int(veto.net_return.gt(0).sum()),
                    "保留单笔均益%": kept.net_return.mean()*100, "拒绝单笔均益%": veto.net_return.mean()*100,
                    "基准机会均益%": closed.net_return.mean()*100,
                    "过滤后等机会均益%": kept.net_return.sum()/n*100 if n else np.nan,
                    "等机会改善(百分点)": -veto.net_return.sum()/n*100 if n else np.nan,
                    "避免亏损贡献(百分点)": avoided/n*100 if n else np.nan,
                    "错过盈利贡献(百分点)": missed/n*100 if n else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def filter_audit(decisions, events):
    if decisions.empty or events.empty:
        return pd.DataFrame()
    d = decisions.rename(columns={"date": "signal_date", "rank": "baseline_rank"}).copy()
    d["signal_date"] = pd.to_datetime(d.signal_date)
    base = events[events.group.eq(GROUP_BASE)].copy()
    base["signal_date"] = pd.to_datetime(base.signal_date)
    keys = ["signal_date", "ts_code"]
    if base.duplicated(keys+["take_profit"]).any() or d.duplicated(keys+["group"]).any():
        raise DataError("过滤审计存在重复信号，不能重复计入收益")
    base = base[[col for col in base if col not in d.columns or col in keys]]
    out = d.merge(base, on=keys, how="inner")
    out["year"] = out.signal_date.dt.year
    out["outcome_source"] = "涨幅基准独立信号_允许重叠_不是账户收益"
    return out


def selection_comparison(events, candidates):
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["signal_date", "ts_code", "take_profit"]
    e = events.copy(); e["signal_date"] = pd.to_datetime(e.signal_date)
    a = e[e.group.eq(GROUP_MAIN)].set_index(keys)
    b = e[e.group.eq(GROUP_BASE)].set_index(keys)
    common = a.index.intersection(b.index)
    parts = []
    for label, f in [("两种排序共同选中", b.loc[common]),
                     ("仅涨幅排序选中", b.loc[b.index.difference(a.index)]),
                     ("仅综合评分选中", a.loc[a.index.difference(b.index)])]:
        f = f.reset_index().copy(); f["cohort"] = label; parts.append(f)
    ledger = pd.concat([f for f in parts if not f.empty], ignore_index=True) if any(not f.empty for f in parts) else pd.DataFrame()
    if ledger.empty:
        return ledger, pd.DataFrame()
    features = candidates.rename(columns={"date": "signal_date"}).copy()
    features["signal_date"] = pd.to_datetime(features.signal_date)
    feature_cols = ["ret1", "ret5"]+list(SCORE_COLS)
    take = [col for col in feature_cols if col not in ledger]
    ledger = ledger.merge(features[["signal_date", "ts_code"]+take], on=["signal_date", "ts_code"], how="left", validate="many_to_one")
    summary = summarize(ledger, ["cohort", "take_profit"])
    means = ledger.groupby(["cohort", "take_profit"], as_index=False)[feature_cols].mean()
    means = means.rename(columns={col: "信号均值_"+col for col in feature_cols})
    return ledger, summary.merge(means, on=["cohort", "take_profit"], how="left")


def factor_bucket_summary(events, candidates):
    if events.empty:
        return pd.DataFrame()
    base = events[events.group.eq(GROUP_BASE)].copy()
    base["signal_date"] = pd.to_datetime(base.signal_date)
    f = add_filter_ranks(candidates).rename(columns={"date": "signal_date"})
    f["signal_date"] = pd.to_datetime(f.signal_date)
    cols = ["signal_date", "ts_code", "filter_pool_n"]+[col+"_filter_pct" for col in SCORE_COLS]
    base = base.merge(f[cols], on=["signal_date", "ts_code"], how="left", validate="many_to_one")
    parts = []
    for _, label, col in FILTERS:
        block = base.copy(); pct = block[col+"_filter_pct"]
        block["dimension"] = label
        block["factor_bucket"] = np.select(
            [block.filter_pool_n.lt(4)|pct.isna(), pct.lt(25), pct.lt(50), pct.lt(75)],
            ["不可评估：池小或缺分位", "0—25分位", "25—50分位", "50—75分位"], default="75—100分位")
        parts.append(block)
    return summarize(pd.concat(parts, ignore_index=True), ["dimension", "factor_bucket", "take_profit"])


def filter_execution_audit(trades, decisions):
    if trades.empty or decisions.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["signal_date", "ts_code", "take_profit"]
    t = trades.copy(); t["signal_date"] = pd.to_datetime(t.signal_date)
    base = t[t.group.eq(GROUP_BASE)].set_index(keys)
    d = decisions.copy(); d["date"] = pd.to_datetime(d.date)
    rows = []
    for group, signals in d.groupby("group", sort=False):
        decision = signals.set_index(["date", "ts_code"])
        filtered = t[t.group.eq(group)].set_index(keys)
        for key, old in base.iterrows():
            keep = bool(decision.loc[key[:2], "filter_pass"])
            new = filtered.loc[key] if key in filtered.index else None
            if keep and new is None:
                raise DataError("通过过滤的信号缺少成交审计记录")
            old_fill = old.status in ["已平仓", "未平仓"]
            new_fill = new is not None and new.status in ["已平仓", "未平仓"]
            if not keep and old_fill:
                label, source, side = "直接过滤基准成交", old, "基准成交结果（已被过滤）"
            elif old_fill and new_fill:
                label, source, side = "共同成交", new, "过滤组实际模拟"
            elif new_fill:
                label, source, side = "持仓路径变化新增", new, "过滤组实际模拟"
            elif old_fill:
                label, source, side = "持仓路径变化减少", old, "基准成交结果（过滤组未成交）"
            else:
                continue
            rows.append({**source.to_dict(), **dict(zip(keys, key)), "group": group,
                         "change_type": label, "reference_side": side, "baseline_status": old.status,
                         "filtered_status": new.status if new is not None else "规则拒绝"})
    ledger = pd.DataFrame(rows)
    return (ledger, summarize(ledger, ["group", "change_type", "take_profit"])) if not ledger.empty else (ledger, pd.DataFrame())


def top10_event_ledger(events, pool):
    if events.empty or pool.empty:
        return pd.DataFrame()
    e = events[events.experiment.eq("OBS_TOP10")].copy()
    if e.empty:
        return pd.DataFrame()
    e["signal_date"] = pd.to_datetime(e.signal_date)
    p = pool.rename(columns={"date": "signal_date"}).copy()
    p["signal_date"] = pd.to_datetime(p.signal_date)
    keys = ["signal_date", "ts_code"]
    if e.duplicated(keys+["take_profit"]).any() or p.duplicated(keys).any():
        raise DataError("前十独立观察存在重复股票日")
    if len(e) != 2*len(p) or set(e.take_profit) != {.05, .10}:
        raise DataError("前十观察缺少完整两档止盈路径，不能用已入选股票替代全体前十")
    out = e.merge(p[keys+[c for c in p if c not in e]], on=keys, validate="many_to_one")
    if len(out) != len(e):
        raise DataError("前十观察的候选与路径不一致")
    return out


def top10_path_stats(events, keys):
    """路径统计只取一档止盈一次；完整5日是诊断窗口，并非可兑现收益。"""
    if events.empty:
        return pd.DataFrame()
    rows = []
    f = events[events.take_profit.eq(.05)]
    for key, block in f.groupby(keys, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        metrics = path_summary(block.assign(group="前十观察")).iloc[0].drop("group").to_dict()
        entered = block[block.entry_price.notna() & ~block.data_issue.eq(True)]
        first = entered[entered.close_return_d1.notna()]
        metrics.update({"可观察买入当日数": len(first),
                        "买入当日收盘平均涨幅%": first.close_return_d1.mean()*100,
                        "买入当日曾达5%比例": first.buy_day_hit5.eq(True).mean()*100 if len(first) else np.nan})
        rows.append({**dict(zip(keys, key)), **metrics})
    return pd.DataFrame(rows)


def top10_selection_changes(events, pool):
    if events.empty or pool.empty:
        return pd.DataFrame()
    e = events.copy(); e["signal_date"] = pd.to_datetime(e.signal_date)
    keys = ["signal_date", "ts_code", "take_profit"]
    base = e[e.group.eq(GROUP_BASE)].set_index(keys)
    parts = []
    for experiment, label, _, _ in TOP10_MODELS:
        new = e[e.experiment.eq(experiment)].set_index(keys)
        for cohort, block in [("共同选中", new.loc[new.index.intersection(base.index)]),
                              ("新方案换入", new.loc[new.index.difference(base.index)]),
                              ("原前三被换出", base.loc[base.index.difference(new.index)])]:
            block = block.reset_index().copy()
            block["comparison"] = label; block["cohort"] = cohort
            parts.append(block)
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        return out
    p = pool.rename(columns={"date": "signal_date"}).copy()
    p["signal_date"] = pd.to_datetime(p.signal_date)
    join = ["signal_date", "ts_code"]
    return out.merge(p[join+[c for c in p if c not in out]], on=join, how="left", validate="many_to_one")


def paired_top10_days(events, pool, cfg, groups=None):
    """按共同信号日、固定3个名额比较独立事件；未知整日剔除，不归零。"""
    if events.empty or pool.empty:
        return pd.DataFrame(), pd.DataFrame()
    e = events.copy(); e["signal_date"] = pd.to_datetime(e.signal_date)
    groups = list(groups) if groups is not None else [GROUP_BASE]+[x[1] for x in TOP10_MODELS]
    e = e[e.group.isin(groups)].copy()
    issue = e.data_issue.eq(True)
    closed = e.status.eq("已平仓") & ~issue & e.net_return.notna()
    skipped = e.status.eq("未买入") & ~issue
    e["known"] = closed | skipped
    e["contribution"] = np.where(closed, e.net_return, np.where(skipped, 0.0, np.nan))
    lookup = {(g, d, tp): f for (g, d, tp), f in e.groupby(["group", "signal_date", "take_profit"])}
    rows = []
    for day in sorted(pd.to_datetime(pool.date).unique()):
        for tp in [.05, .10]:
            for group in groups:
                f = lookup.get((group, pd.Timestamp(day), tp), e.iloc[:0])
                if len(f) > cfg.top_n:
                    raise DataError("十选三固定名额审计发现超额选股")
                unknown = int((~f.known).sum())
                rows.append(dict(signal_date=pd.Timestamp(day), year=pd.Timestamp(day).year, group=group,
                    take_profit=tp, selected_count=len(f), unknown_count=unknown,
                    cash_slots=cfg.top_n-len(f)+int((f.status.eq("未买入") & f.known).sum()),
                    slot_mean=np.nan if unknown else float(f.contribution.sum()/cfg.top_n)))
    daily = pd.DataFrame(rows)
    base = daily[daily.group.eq(GROUP_BASE)][["signal_date", "take_profit", "slot_mean"]].rename(columns={"slot_mean": "baseline_slot_mean"})
    paired = daily[daily.group.ne(GROUP_BASE)].merge(base, on=["signal_date", "take_profit"], validate="many_to_one")
    paired["pair_complete"] = paired.slot_mean.notna() & paired.baseline_slot_mean.notna()
    paired["paired_delta"] = paired.slot_mean-paired.baseline_slot_mean
    reports = []
    for period, frame in [("全区间", paired)]+[(str(y), f) for y, f in paired.groupby("year")]:
        for (group, tp), f in frame.groupby(["group", "take_profit"]):
            valid = f[f.pair_complete]
            reports.append(dict(period=period, group=group, take_profit=tp,
                **{"信号日数": len(f), "共同结果已知日数": len(valid), "未知剔除日数": len(f)-len(valid),
                   "基准每名额均益%": valid.baseline_slot_mean.mean()*100,
                   "方案每名额均益%": valid.slot_mean.mean()*100,
                   "配对改善(百分点)": valid.paired_delta.mean()*100,
                   "改善日占比%": valid.paired_delta.gt(1e-12).mean()*100 if len(valid) else np.nan}))
    return paired, pd.DataFrame(reports)


def top10_reports(events, candidates, cfg):
    pool = top10_candidates(candidates, cfg)
    out = {"top10_candidates": pool if cfg.research else pool.iloc[:0]}
    if not cfg.research or cfg.validation_only:
        return out
    ledger = top10_event_ledger(events, pool)
    out["top10_events"] = ledger
    if ledger.empty:
        return out
    for name, keys in {
        "rank": ["return_rank"], "band": ["rank_band"],
        "board_rank": ["board", "rank_band"], "year_rank": ["year", "rank_band"],
        "pre3": ["board", "pre3_band"], "gain15": ["board", "signal_gain_band"],
        "gain15_year": ["year", "board", "signal_gain_band"],
        "gain_pre3": ["board", "signal_gain_band", "pre3_band"],
        "limit_up": ["board", "signal_limit_state"],
    }.items():
        summary = summarize(ledger, keys+["take_profit"])
        means = ledger.groupby(keys+["take_profit"], as_index=False, dropna=False).agg(
            signal_days=("signal_date", "nunique"), avg_pre3=("pre3", "mean"),
            avg_ret1=("ret1", "mean"), avg_ret5=("ret5", "mean"))
        out["top10_"+name+"_summary"] = summary.merge(means, on=keys+["take_profit"])
    for name, keys in {"rank": ["return_rank"], "board_rank": ["board", "rank_band"],
                       "gain15": ["board", "signal_gain_band"]}.items():
        out["top10_"+name+"_paths"] = top10_path_stats(ledger, keys)
    changes = top10_selection_changes(events, pool)
    out["top10_selection_changes"] = changes
    out["top10_changes_summary"] = summarize(changes, ["comparison", "cohort", "take_profit"])
    out["top10_changes_by_board"] = summarize(changes, ["comparison", "cohort", "board", "take_profit"])
    out["top10_paired_days"], out["top10_paired_summary"] = paired_top10_days(events, pool, cfg)
    day = pool.groupby("date", as_index=False).agg(eligible_pool_n=("eligible_pool_n", "first"),
        top10_pool_n=("ts_code", "size"), pre3_known=("pre3", "count"), accel_known=("rs_accel", "count"))
    out["top10_coverage"] = day
    return out


def sample_phase(values, mode=""):
    d = pd.to_datetime(values)
    if "演示" in mode:
        return pd.Series("合成演示_不是验证", index=d.index)
    return pd.Series(np.select(
        [d.between(pd.Timestamp(EXPLORED_START), pd.Timestamp(EXPLORED_END)),
         d.lt(pd.Timestamp(EXPLORED_START)), d.gt(pd.Timestamp(FREEZE_DAY))],
        ["已研究区间", "历史待验证_须排除既往调参", "冻结后日期_须核实事前留存"],
        default="冻结边界日期_不作独立验证"), index=d.index)


def validation_protocol(cfg, mode=""):
    actual = {k: getattr(cfg, k) for k in FROZEN_PARAMS}
    differences = {k: {"standard": v, "actual": actual[k]} for k, v in FROZEN_PARAMS.items()
                   if actual[k] != v and not (isinstance(v, (int, float)) and np.isclose(actual[k], v, rtol=0, atol=1e-12))}
    # 代码与参数双重留痕；日期不纳入协议指纹，便于不同区间核对是否用了同一套规则。
    logic = "\n".join(inspect.getsource(f) for f in [normalize_members, sector_panel,
        max_runup, features, assign_sector, name_state, consecutive_rs, build_candidates,
        score_candidates, add_filter_ranks, apply_filter_rule, top10_candidates,
        select_candidates, board_name, buy_gap, finite_positive, simulate_trade,
        run_trades, stamp_tax, paired_top10_days])
    rules = dict(core=sorted(CORE), tech_words=TECH_WORDS,
        models=[s for s in experiments(cfg) if s["group"] in VALIDATION_GROUPS])
    logic += json.dumps(rules, sort_keys=True, ensure_ascii=False)
    contract = dict(models=["RETURN", "T10_PRE3"], primary_target=.10, sensitivity_target=.05,
                    params=actual, trading_logic_sha256=hashlib.sha256(logic.encode()).hexdigest())
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return dict(protocol_id="M14_FIXED_RETURN_PRE3", protocol_sha256=digest,
        frozen_on=FREEZE_DAY, explored_signal_start=EXPLORED_START, explored_signal_end=EXPLORED_END,
        **contract, standard_parameters_match=not differences, parameter_differences=differences,
        mode="合成演示" if "演示" in mode else "固定两组" if cfg.validation_only else "附加探索对照",
        date_labels_are_not_proof_of_out_of_sample=True, automatic_best_model_selection=False,
        leave_one_month_out="仅剔除该月信号的原始独立事件；不重选股、不重放改变持仓路径",
        overlap_note="不同信号日交易会重叠；固定名额均益不是账户收益，不能复利或年化")


def paired_metrics(f):
    valid = f[f.pair_complete.eq(True)]
    delta = valid.paired_delta
    return {"信号日数": len(f), "共同结果已知日数": len(valid), "未知剔除日数": len(f)-len(valid),
            "基准每名额均益%": valid.baseline_slot_mean.mean()*100,
            "方案每名额均益%": valid.slot_mean.mean()*100,
            "配对改善(百分点)": delta.mean()*100,
            "改善日数": int(delta.gt(1e-12).sum()), "落后日数": int(delta.lt(-1e-12).sum()),
            "持平日数": int(delta.abs().le(1e-12).sum())}


def concentration_tables(paired):
    """按同日固定名额审计增量集中度，不拟合阈值或反向删除交易。"""
    if paired.empty:
        return {}
    f = paired.copy()
    f["month"] = pd.to_datetime(f.signal_date).dt.strftime("%Y-%m")
    summaries, monthly, leave_out, concentration, samples, days = [], [], [], [], [], []
    for (group, tp), block in f.groupby(["group", "take_profit"], sort=False):
        base = dict(group=group, take_profit=tp)
        summaries.append({**base, **paired_metrics(block)})
        complete = block[block.pair_complete.eq(True)]
        n = len(complete); total = complete.paired_delta.sum() if n else np.nan
        month_sums = complete.groupby("month").paired_delta.sum()
        positive_month_sum = month_sums[month_sums.gt(0)].sum()
        loo_means = []
        for month, m in block.groupby("month", sort=True):
            known = m[m.pair_complete.eq(True)]
            value = known.paired_delta.sum() if len(known) else np.nan
            monthly.append({**base, "month": month, **paired_metrics(m),
                "对全期改善贡献(百分点)": value/n*100 if n else np.nan,
                "占全期净改善%": value/total*100 if n and total > 1e-12 else np.nan})
            remaining = block[block.month.ne(month)]
            metrics = paired_metrics(remaining)
            leave_out.append({**base, "excluded_month": month, "剔除已知日数": len(known), **metrics})
            loo_means.append(metrics["配对改善(百分点)"])
        if n:
            order = complete.sort_values(["paired_delta", "signal_date"], ascending=[False, True])
            remove_n = min(5, n)
            remaining = order.iloc[remove_n:]
            biggest = month_sums.idxmax()
            best_sum = month_sums.max()
            positive_share = best_sum/positive_month_sum*100 if best_sum > 0 and positive_month_sum > 0 else np.nan
            net_share = best_sum/total*100 if total > 1e-12 else np.nan
        else:
            order, remaining, biggest, remove_n, positive_share, net_share = complete, complete, "无完整结果", 0, np.nan, np.nan
        concentration.append({**base, "完整信号日数": n, "有完整结果月份数": len(month_sums),
            "全期配对改善(百分点)": total/n*100 if n else np.nan,
            "贡献最大月份": biggest, "最大月占净改善%": net_share,
            "最大月占正贡献月份%": positive_share,
            "逐月剔除后最小改善(百分点)": min([x for x in loo_means if pd.notna(x)], default=np.nan),
            "实际剔除最大改善日数": remove_n,
            "剔除最大5个改善日后(百分点)": remaining.paired_delta.mean()*100})
        if n:
            best = order.head(5).copy(); best["diagnostic_tail"] = "改善最大5日"
            worst = order.tail(5).sort_values("paired_delta").copy(); worst["diagnostic_tail"] = "落后最大5日"
            days.extend([best, worst])
        for phase, m in block.groupby("sample_phase", sort=False):
            samples.append({**base, "sample_phase": phase, **paired_metrics(m)})
    return {"validation_summary": pd.DataFrame(summaries), "validation_monthly": pd.DataFrame(monthly),
            "validation_leave_month_out": pd.DataFrame(leave_out), "validation_concentration": pd.DataFrame(concentration),
            "validation_by_sample": pd.DataFrame(samples),
            "validation_extreme_days": pd.concat(days, ignore_index=True) if days else pd.DataFrame()}


def validation_reports(events, selected, candidates, trades, cfg, mode):
    pool = top10_candidates(candidates, cfg)
    if events.empty or pool.empty:
        return {}
    configured = {s["group"] for s in experiments(cfg)}
    if not set(VALIDATION_GROUPS).issubset(configured):
        return {}
    wanted = selected[selected.group.isin(VALIDATION_GROUPS)].copy()
    wanted["signal_date"] = pd.to_datetime(wanted.date)
    e = events[events.group.isin(VALIDATION_GROUPS)].copy()
    e["signal_date"] = pd.to_datetime(e.signal_date)
    keys = ["group", "signal_date", "ts_code", "take_profit"]
    expected = wanted[["group", "signal_date", "ts_code"]].merge(pd.DataFrame({"take_profit": [.05, .10]}), how="cross")
    if e.duplicated(keys).any() or set(map(tuple, e[keys].to_numpy())) != set(map(tuple, expected[keys].to_numpy())):
        raise DataError("固定规则验证缺少已选信号的独立回放，不能把缺记录当作空仓")
    paired, _ = paired_top10_days(e, pool, cfg, groups=VALIDATION_GROUPS)
    paired["sample_phase"] = sample_phase(paired.signal_date, mode)
    out = {"validation_paired_days": paired, **concentration_tables(paired)}
    counts = pool[["date"]].drop_duplicates().copy()
    counts["sample_phase"] = sample_phase(counts.date, mode)
    out["validation_sample_dates"] = counts
    tr = trades[trades.group.isin(VALIDATION_GROUPS)].copy()
    if not tr.empty:
        tr["month"] = pd.to_datetime(tr.signal_date).dt.strftime("%Y-%m")
        tr["sample_phase"] = sample_phase(tr.signal_date, mode)
        out["validation_monthly_trades"] = summarize(tr, ["group", "take_profit", "month"])
        out["validation_trades_by_sample"] = summarize(tr, ["group", "take_profit", "sample_phase"])
    return out


# 仅用于事后归因的固定特征名单；不含次日开盘、后续最高价或任何收益标签。
ATTR_FEATURES = (
    ('ret1', '信号日涨幅%', 100.), ('pre3', '信号前3日涨幅%', 100.),
    ('rs3', '3日相对板块超额(百分点)', 100.), ('rs5', '5日相对板块超额(百分点)', 100.),
    ('efficiency', '5日上涨效率', 1.), ('amount_ratio', '成交额/此前20日均额', 1.),
    ('turnover_rate', '换手率%', 1.), ('clv3', '3日平均收盘位置', 1.),
    ('bias10', '距MA10%', 100.), ('volatility5', '5日波动率%', 100.),
    ('sector_ret1', '板块当日涨幅%', 100.), ('breadth', '板块上涨家数比例%', 100.),
    ('strength_score', '相对强度得分', 1.), ('efficiency_score', '上涨效率得分', 1.),
    ('volume_score', '量价配合得分', 1.), ('position_score', '收盘位置得分', 1.),
)
ATTR_CONTRASTS = ('未达5%到期亏损', '达5%后到期亏损', '止损退出')
ATTR_BASES = ('独立信号', '持仓回放')
ATTR_NOTE = ('全部标签在交易结束后定义，仅作研究，不能用于当天选股。归因主要分析10%止盈。'
    '独立信号允许重叠；持仓回放会跳过同股重复持仓；两者都不是账户收益。'
    '止损后才出现的最高价不能认定为卖早了，止损单独列出。'
    '两段历史用于寻找新特征后，都属于该新假设的探索样本，不能再称为独立验证。'
    '固定16项特征同时观察；同向差异不代表统计显著、因果关系或可交易优势。'
    '不会自动筛选最佳特征、门槛、权重或修改买卖规则。')


def attribution_features(candidates):
    """特征只取信号收盘时的候选表；当日分位参照完整合格、不过热池。"""
    c = candidates.copy()
    c['signal_date'] = pd.to_datetime(c['date'])
    if c.duplicated(['signal_date', 'ts_code']).any():
        raise DataError('归因候选表存在重复股票日')
    eligible = c.base_reason.fillna('').eq('') & c.overheat.eq(False)
    out = c[['signal_date', 'ts_code']].copy()
    for key, _, _ in ATTR_FEATURES:
        value = pd.to_numeric(c[key], errors='coerce') if key in c else pd.Series(np.nan, index=c.index)
        value = value.where(np.isfinite(value))
        out['signal_'+key] = value
        g = value.where(eligible).groupby(c.signal_date)
        count = g.transform('count')
        # 池不足4只、同日没有横截面差异时不强行生成区分信息。
        spread = g.transform('max')-g.transform('min')
        out['pool_pct_'+key] = ((g.rank(method='average')-.5)/count*100).where(count.ge(4) & spread.gt(1e-12))
    return out


def classify_outcomes(records, hold_days=5):
    f = records.copy()
    label = pd.Series('未完成/无法分类', index=f.index, dtype=object)
    good = ~f.data_issue.eq(True)
    closed = f.status.eq('已平仓') & good & f.net_return.notna()
    label.loc[f.status.eq('未买入') & good] = '未买入'
    label.loc[f.status.eq('重复持仓跳过')] = '重复持仓跳过'
    label.loc[closed] = '延期/其他退出'
    # 延期交易与完整5日内的结果分开，避免把止损后的反弹叫作冲高回落。
    timely = closed & pd.to_numeric(f.hold_days, errors='coerce').le(hold_days)
    reason = f.reason.fillna('')
    label.loc[timely & reason.str.contains('止盈')] = '顺利止盈'
    label.loc[closed & reason.str.contains('止损')] = '止损退出'
    expiry = timely & reason.eq('到期退出')
    label.loc[expiry] = '到期路径不完整'
    complete = expiry & f.horizon_complete.eq(True) & f.sellable_mfe5.notna()
    complete &= pd.to_numeric(f.hold_days, errors='coerce').eq(5) & (hold_days == 5)
    reached = f.sellable_mfe5.ge(.05-1e-12)
    loss = f.net_return.lt(0)
    label.loc[complete & ~reached & loss] = '未达5%到期亏损'
    label.loc[complete & ~reached & ~loss] = '未达5%到期非亏损'
    label.loc[complete & reached & loss] = '达5%后到期亏损'
    label.loc[complete & reached & ~loss] = '达5%后到期非亏损'
    label.loc[f.data_issue.eq(True)] = '数据问题'
    return label


def attribution_reports(events, candidates, trades, cfg):
    if cfg.hold_days != 5 or candidates.empty:
        return {}
    signals = attribution_features(candidates)
    ledgers = []
    for basis, records in zip(ATTR_BASES, [events, trades]):
        if records is None or records.empty:
            continue
        f = records[records.group.isin(VALIDATION_GROUPS) & records.take_profit.eq(.10)].copy()
        if f.empty:
            continue
        f['signal_date'] = pd.to_datetime(f.signal_date)
        if f.duplicated(['group', 'signal_date', 'ts_code']).any():
            raise DataError('归因成交表存在重复信号')
        for col, default in [('data_issue', False), ('horizon_complete', False), ('hold_days', np.nan),
                             ('sellable_mfe5', np.nan), ('net_return', np.nan), ('reason', '')]:
            if col not in f:
                f[col] = default
        f['outcome'] = classify_outcomes(f, cfg.hold_days)
        f['basis'] = basis
        f = f.merge(signals, on=['signal_date', 'ts_code'], how='left', validate='many_to_one', indicator=True)
        if f._merge.ne('both').any():
            raise DataError('归因记录缺少对应信号日特征，不能用未来日期补齐')
        ledgers.append(f.drop(columns='_merge'))
    if not ledgers:
        return {}
    ledger = pd.concat(ledgers, ignore_index=True)
    summary, contrasts, buckets = [], [], []
    for (basis, group), block in ledger.groupby(['basis', 'group'], sort=False):
        base = dict(basis=basis, group=group)
        known = block.status.eq('已平仓') & ~block.data_issue.eq(True) & block.net_return.notna()
        closed_count = int(known.sum())
        for outcome, b in block.groupby('outcome', sort=False):
            valid = b[b.status.eq('已平仓') & ~b.data_issue.eq(True) & b.net_return.notna()]
            summary.append({**base, 'outcome': outcome, '记录数': len(b), '占全部信号%': len(b)/len(block)*100,
                '有效平仓数': len(valid), '净均益%': valid.net_return.mean()*100,
                '净中位数%': valid.net_return.median()*100,
                '对全部平仓均益贡献(百分点)': valid.net_return.sum()/closed_count*100 if closed_count and len(valid) else np.nan})
        success = block[block.outcome.eq('顺利止盈')]
        for contrast in ATTR_CONTRASTS:
            other = block[block.outcome.eq(contrast)]
            for feature, label, scale in ATTR_FEATURES:
                a, b = success['signal_'+feature].dropna(), other['signal_'+feature].dropna()
                pa, pb = success['pool_pct_'+feature].dropna(), other['pool_pct_'+feature].dropna()
                contrasts.append({**base, 'contrast': contrast, 'feature': feature, '特征': label,
                    '止盈有效数': len(a), '对照有效数': len(b),
                    '止盈中位数': a.median()*scale, '对照中位数': b.median()*scale,
                    '中位数差': (a.median()-b.median())*scale,
                    '止盈池内分位有效数': len(pa), '对照池内分位有效数': len(pb),
                    '止盈平均池内分位': pa.mean(), '对照平均池内分位': pb.mean(),
                    '池内分位差': pa.mean()-pb.mean()})
        # 固定三等分检视全部已选信号，避免仅对比输赢两端遗漏其他交易。
        for feature, label, _ in ATTR_FEATURES:
            pct = block['pool_pct_'+feature]
            band = pd.cut(pct, [-np.inf, 100/3, 200/3, np.inf], labels=['低三分之一', '中三分之一', '高三分之一'])
            for name in ['低三分之一', '中三分之一', '高三分之一']:
                b = block[band.eq(name)]; valid = b[b.status.eq('已平仓') & ~b.data_issue.eq(True) & b.net_return.notna()]
                full = b[b.horizon_complete.eq(True) & ~b.data_issue.eq(True) & b.entry_price.notna()]
                buckets.append({**base, 'feature': feature, '特征': label, '分位段': name,
                    '记录数': len(b), '分位缺失记录数': int(pct.isna().sum()), '有效平仓数': len(valid),
                    '净均益%': valid.net_return.mean()*100,
                    '止盈占已平仓%': valid.outcome.eq('顺利止盈').mean()*100 if len(valid) else np.nan,
                    '完整路径数': len(full),
                    '可卖日期曾达5%比例': full.sellable_mfe5.ge(.05-1e-12).mean()*100 if len(full) else np.nan,
                    '可卖日期曾达10%比例': full.sellable_mfe5.ge(.10-1e-12).mean()*100 if len(full) else np.nan})
    return {'attribution_ledger': ledger, 'attribution_summary': pd.DataFrame(summary),
            'attribution_features': pd.DataFrame(contrasts), 'attribution_buckets': pd.DataFrame(buckets)}


def load_attribution_archive(raw, name='审计.zip'):
    """只读取受限ZIP内的表格，不解压文件、不运行其中代码、不调用行情接口。"""
    if len(raw) > 100*1024*1024:
        raise DataError('审计ZIP超过100MB，请使用单次回测的审计包')
    needed = ['manifest.json', 'candidates.csv', 'selected.csv', 'trades.csv', 'independent_signal_paths.csv']
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            if any(names.count(k) != 1 for k in needed):
                raise DataError('请上传M1.3或之后的完整回测审计ZIP，缺表或同名重复表不能归因')
            if sum(i.file_size for i in z.infolist()) > 300*1024*1024 or any(z.getinfo(k).file_size > 100*1024*1024 for k in needed):
                raise DataError('审计包解压后过大，拒绝读取')
            manifest = json.loads(z.read('manifest.json'))
            if manifest.get('version') not in ['M1.3', 'M1.4', 'M1.5', 'M1.6']:
                raise DataError('当前导入支持M1.3至M1.6的完整回测审计包，不支持二次归因包')
            frames = {}
            for key in needed[1:]:
                try:
                    frames[key[:-4]] = pd.read_csv(z.open(key), dtype={'ts_code': str, 'sector': str}, float_precision='round_trip')
                except pd.errors.EmptyDataError:
                    frames[key[:-4]] = pd.DataFrame()
    except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DataError('审计ZIP或表格损坏，无法读取') from exc
    config = manifest.get('config', {})
    if not isinstance(config, dict) or not {'start', 'end'}.issubset(config):
        raise DataError('审计缺少有效配置和信号区间')
    cfg = Config(**{k: v for k, v in config.items() if k in Config.__dataclass_fields__})
    cfg.validate()
    if cfg.hold_days != 5:
        raise DataError('五日归因只适用于持有上限5日的报告')
    c, selected, events, trades = [frames[k] for k in ['candidates', 'selected', 'independent_signal_paths', 'trades']]
    if c.empty:
        raise DataError('本报告没有候选特征，无法归因')
    required = [
        (c, ['date', 'ts_code', 'base_reason', 'exclude_reason', 'overheat', 'ret1', 'score', 'close']),
        (selected, ['date', 'ts_code', 'group', 'rank', 'return_rank', 'gap_limit']),
        (events, ['signal_date', 'ts_code', 'group', 'take_profit', 'status', 'net_return',
                  'data_issue', 'horizon_complete', 'sellable_mfe5', 'entry_price', 'hold_days', 'reason']),
        (trades, ['signal_date', 'ts_code', 'group', 'take_profit', 'status', 'net_return']),
    ]
    for frame, columns in required:
        missing = sorted(set(columns)-set(frame.columns))
        if missing:
            raise DataError('审计缺少必要字段：'+', '.join(missing))
    for key in ['base_reason', 'exclude_reason']:
        c[key] = c[key].fillna('')
    c['date'] = pd.to_datetime(c.date)
    if not c.date.between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)).all():
        raise DataError('候选信号日期超出报告声明区间')
    if selected.empty and events.empty:
        raise DataError('本报告没有已选信号，无法进行结果归因')
    keys = ['group', 'signal_date', 'ts_code', 'take_profit']
    chosen = selected[selected.group.isin(VALIDATION_GROUPS)].copy()
    chosen['signal_date'] = pd.to_datetime(chosen.date)
    expected = chosen[['group', 'signal_date', 'ts_code']].merge(pd.DataFrame({'take_profit': [.05, .10]}), how='cross')
    for records in [events, trades]:
        records['signal_date'] = pd.to_datetime(records.signal_date)
        e = records[records.group.isin(VALIDATION_GROUPS)]
        if e.duplicated(keys).any() or set(map(tuple, e[keys].to_numpy())) != set(map(tuple, expected[keys].to_numpy())):
            raise DataError('已选信号与独立回放/持仓记录不齐全或重复，不能将缺记录当作未成交')
    # 检查当前固定规则能否重建导出选择；这里只复算选股，不声称重新撮合行情。
    rebuilt = select_candidates(c, replace(cfg, validation_only=True, efficiency_research=False))
    cols = ['group', 'date', 'ts_code', 'rank', 'return_rank', 'gap_limit']
    original = chosen.copy(); original['date'] = pd.to_datetime(original.date)
    order = ['group', 'date', 'ts_code']
    try:
        pd.testing.assert_frame_equal(original[cols].sort_values(order).reset_index(drop=True),
            rebuilt[cols].sort_values(order).reset_index(drop=True), check_dtype=False, atol=1e-10, rtol=1e-10)
    except AssertionError as exc:
        raise DataError('导出选股不能由当前固定规则重建，不能直接按同一规则比较') from exc
    known_source = manifest.get('source_sha256') in {
        'a4f8dbbaf706cbb034a2693a85bddc38fb93b56505b412aea1f95e778176e202',
        'e861bf410d717fcdf32c14965ce9a800ff87b3dbc2800215983e0b4da43c114e',
        '92ee79774bbe469777aaea7ed6a38dd3db9bec045e6ca145f1d7439832edbcc5'}
    logic = manifest.get('validation_protocol', {}).get('trading_logic_sha256')
    logic_match = known_source or logic == validation_protocol(cfg)['trading_logic_sha256']
    meta = dict(name=Path(name).name, version=manifest['version'], start=cfg.start, end=cfg.end,
        source_sha256=manifest.get('source_sha256'), zip_sha256=hashlib.sha256(raw).hexdigest(),
        mode=manifest.get('mode', ''), params={k: getattr(cfg, k) for k in FROZEN_PARAMS},
        selection_rebuilt=True, price_execution_replayed=False, declared_logic_recognized=logic_match,
        origin='合成演示' if '演示' in manifest.get('mode', '') else '历史探索_新特征未验证')
    tables = attribution_reports(events, c, trades, cfg)
    if logic_match:
        tables.update(efficiency_trial(c, selected, events, trades, replace(cfg, efficiency_research=True), manifest.get('mode', '')))
    meta['efficiency_replay_completed'] = 'efficiency_actual_trades' in tables
    meta['efficiency_protocol'] = efficiency_protocol(cfg)
    return dict(meta=meta, tables=tables)


def attribution_consistency(a, b):
    keys = ['basis', 'group', 'contrast', 'feature', '特征']
    joined = a.merge(b, on=keys, how='outer', suffixes=('_区间1', '_区间2'), validate='one_to_one')
    for axis, source in [('原值', '中位数差'), ('池内分位', '池内分位差')]:
        x, y = joined[source+'_区间1'], joined[source+'_区间2']
        count_cols = ['止盈有效数', '对照有效数'] if axis == '原值' else ['止盈池内分位有效数', '对照池内分位有效数']
        enough = pd.Series(True, index=joined.index)
        for col in count_cols:
            enough &= joined[col+'_区间1'].ge(30) & joined[col+'_区间2'].ge(30)
        direction = np.select([x.gt(1e-12) & y.gt(1e-12), x.lt(-1e-12) & y.lt(-1e-12)],
                              ['止盈组两段都较高', '止盈组两段都较低'], default='方向不一致或无差异')
        joined[axis+'方向检查'] = np.where(x.isna() | y.isna(), '数据不足',
            np.where(enough, direction, '样本不足30_仅列数值'))
    return joined


def review_archives(files):
    if not 1 <= len(files) <= 2:
        raise DataError('请上传一份审计包，或两份不同区间的审计包')
    reports = [load_attribution_archive(raw, name) for name, raw in files]
    reports.sort(key=lambda r: (r['meta']['start'], r['meta']['end']))
    tables = {}
    for i, report in enumerate(reports, 1):
        for key, frame in report['tables'].items():
            f = frame.copy(); f.insert(0, 'source_period', f"区间{i} {report['meta']['start']}—{report['meta']['end']}")
            tables.setdefault(key, []).append(f)
    tables = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
    warnings = []; comparable = False
    if len(reports) == 2:
        a, b = [r['meta'] for r in reports]
        if a['end'] >= b['start']:
            warnings.append('信号区间重叠，不能当作两段独立样本；不生成方向一致性表。')
        elif a['params'] != b['params']:
            warnings.append('两份报告的股票池或交易参数不同；不生成方向一致性表。')
        elif not a['declared_logic_recognized'] or not b['declared_logic_recognized']:
            warnings.append('至少一份报告无法核对交易逻辑来源；仅分别展示。')
        elif '演示' in a['origin'] or '演示' in b['origin']:
            warnings.append('合成演示不能作为历史一致性证据；仅分别展示。')
        else:
            comparable = True
            tables['attribution_consistency'] = attribution_consistency(
                reports[0]['tables']['attribution_features'], reports[1]['tables']['attribution_features'])
    meta = dict(version=VERSION, report_type='既有价格路径复用_新增过滤并重放持仓约束_未重撮合原始行情', sources=[r['meta'] for r in reports],
        comparable=comparable, warnings=warnings, feature_count=len(ATTR_FEATURES), base_rules_changed=False,
        new_trial=EFF_EXPERIMENT, efficiency_replay='复用原独立路径并重放同股持仓约束，不重撮合行情',
        note=ATTR_NOTE, comparison_minimum_count=30,
        minimum_count_is_not_significance_test=True, sorted_periods='区间1为更早信号区间；区间2为更晚区间')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for key, f in tables.items():
            z.writestr(key+'.csv', f.to_csv(index=False).encode('utf-8-sig'))
        z.writestr('manifest.json', json.dumps(meta, ensure_ascii=False, indent=2))
        z.writestr('归因说明.txt', ATTR_NOTE)
    return dict(meta=meta, tables=tables, archive=stream.getvalue())


RULES = """板块领涨惯性 M1.6：单一效率过滤对照

默认三组：原涨幅前三、低前涨幅前三、低前涨幅剔除高效率。
新组先完成低前涨幅前三选择，再排除效率分位严格大于200/3的股票，不补位且保留原始名次。
分位参照当天完整基础合格且不过热池；平均名次处理并列，至少4个有限值，分位缺失保留。
只增加此一个条件，交易价格、止盈、止损、持有期与原规则相同。
10%止盈为主要目标，5%为敏感性对照。2024-09-20至2026-09-18均参与了新假设形成。
新规则冻结于2026-09-22；冻结后日期标签仍不证明信号事前留存。
旧ZIP导入复用独立信号的完整价格路径，先复现原组持仓记录，再重放过滤组持仓约束。
新增持仓路径可能包含原来被重复持仓挡住的信号；不能只删除原成交表中的被过滤交易。
无信号、未买入、未知结果分开；等机会均益保留原有效平仓机会分母，过滤记空仓0。
逐月和极端日剔除仅作集中度诊断，不是可提前执行的条件。

以下保留M1.5及更早审计口径：

M1.5新增结果归因与两份旧审计ZIP对照，不改变选股排序、买价、止盈止损。
只用信号日收盘已知的固定16项特征，未知和缺失不补为0。
止损与冲高回落分开；五日最高价不能证明出现在止损之前。
归因中新特征的两段历史均为探索，不自动筛选门槛，不回填当天选股。


默认只运行原涨幅前三与前十低前涨幅两个方案。10%止盈为主要目标，5%止盈为敏感性对照。
两套选股、交易规则全部继承M1.3；不新增门槛，不把主板/双创拼成事后最优规则。
默认科技池、10元、50—1000亿、含滑点买价上限3%、止损10%、最长5个市场交易日、最多3个名额。
标准手续费与滑点沿用M1.3；参数变更必须在报告中列明，不能当成相同协议验证。
已研究信号区间：2025-09-20至2026-09-18。早于该区间仅标记历史待验证，不能证明从未用于调参。
冻结日期：2026-09-21。信号日严格晚于该日标记冻结后日期；日期标签不证明信号已事前留存。
跨区间报告分别标记样本来源；合成演示不属于历史或前向验证。所有结果仍标记未验证盈利。
固定协议指纹包括交易逻辑与有效参数，不含回测日期；用于检查跨区间是否沿用同一套规则。
逐月配对：同一信号日、固定3个名额，未选/明确未成交为空仓，未平仓与数据问题仍未知。
逐月剔除：逐一移除每个月份的原始独立信号结果，重算其余月份；不改变当时选股和持仓路径。
剔除最大5个改善日、逐月贡献都是事后敏感性诊断，不能用来构造可交易规则或保证稳健。
净改善为正时才报告月份占净改善比例；因负贡献抵消可能超过100%，它不是盈利交易占比。
同时报告最大月占正贡献月份比例，避免总净改善接近0时只看一个夸张百分比。
不要求每月、每年都盈利；不自动按任何门槛宣布通过，不自动推荐历史最优实验。
各月按信号日归属，交易可能跨月、跨样本边界；配对差不是账户日收益，不能累乘或年化。
默认跳过全体前十的影子回放以减少重复研究，只回放两个模型的已选信号；行情四路下载与缓存复用不变。
可关闭固定验证模式复查旧研究；附加探索对照不是新的独立证据。

以下保留M1.3和旧对照定义供复查：

M1.3默认主研究：保留原涨幅前三、综合评分、过热观察；新增两条独立十选三假设，共5组。
另对同日第一板块基础合格且不过热的涨幅前十全部做独立事件回放，观察组不是十只持仓组合。
候选不足十只时按实际数量；并列涨幅按股票代码排序。所有名次均在信号收盘时固定。
T10_PRE3：前十内按信号日前3个市场交易日累计涨幅由小到大选最多三只，不包含信号日。
pre3=复权收盘(t-1)/复权收盘(t-4)-1；可以为负，不把低前涨幅直接命名为有效启动。
T10_ACCEL：前十内按当日个股超额收益减去此前3日平均超额收益，由大到小选最多三只。
超额=个股日涨幅-所在行业日涨幅；四日要求同一行业，字段缺失不参选，不能用第11名替补。
两种排序并列时按原涨幅名次、代码排序，不组合、不调权重、不用未来路径挑赢家。
3%买价上限、近5日50%过热排除、持有和退出规则继承；15%仅用于分板诊断，不作排除线。
观察报告分1—3/4—6/7—10名和单独每一名；按主板/创业板/科创板、年份、信号日>15%分别统计。
前3日涨幅仅按≤0%、0—10%、>10%作描述分组，不据此自动优化买入阈值。
收盘涨停由信号日实际涨停价比较，不把主板涨9.5%或双创涨15%当作涨停。
15%与排名分析均条件于现有科技池和基础门槛，交易结果还条件于可成交，不代表全市场次日规律。
路径诊断列买入当日收盘/冲高，以及买入后可卖日期5%/10%到达率；买入当天冲高不能兑现。
十选三替换审计分别列共同选中、原前三被换出、新方案换入，不只展示换入赢家。
配对比较固定同一批前十候选信号日，每天各cfg.top_n个等额名额。未选/明确未成交按空仓0；
任一已选信号未完成或行情有问题则该组该日未知，只在双方结果已知的共同日比较，并披露剔除数。
每日名额是独立信号的标准化分母，不是账户每日收益；持有期可重叠，不能复利或年化。
2025-09至2026-09已用于提出新假设；本版无独立样本验证，不自动采用最佳模型。
M1.2四维过滤默认关闭，可手动附加复查。最新候选仍默认显示原涨幅前三。

可选M1.2对照：昨日涨幅排序先取前三，再分别检查四个维度的末25%分位；四个条件各自独立，不组合、不调权重。
过滤分位在信号日基础合格且不过热的完整池内计算，按对应维度原始得分作中位秩百分位；不是只在前三内排名。
分位<25拒绝，恰好25保留；并列平均排名。合格池少于4只时不作末25%判定，保留并列出不可评估数量。
拒绝后不补位，保留原始涨幅名次。过滤只改变是否入选，不改变评分、买价、止盈止损或持有期。
默认统一3%含滑点买价上限；开启M1.2追加4组，M1.1完整对照再追加19组，行情均复用。
最新候选默认显示涨幅基准。所有过滤都是待验证假设，不自动选用历史最优过滤，更不声称实盘盈利。
误杀审计：以涨幅基准每个独立信号的假设成交为参照，允许同股重叠，分别记录5%与10%止盈；不是账户回报。
等机会比较使用同一组基准有效平仓机会：被过滤机会按空仓0计，分母固定；未平仓/数据问题仍未知，不当作0。
等机会改善=-被过滤机会收益之和/基准有效平仓机会数；避免亏损贡献减错过盈利贡献等于该改善。
保留均益变高并不代表等机会收益改善；还需同时看拒绝组收益、信号覆盖、各年表现和实际持仓路径变化。
共同/独有选股分析采用独立信号，不受之前持仓阻塞影响，其数量与去重后的实际模拟成交数可能不同。
默认不自动加载旧报告做新策略实际回放；旧M1.1报告只有5%独立信号数据时，只能核对相应5%过滤机会审计。

以下为继承的交易口径及可选M1.1对照：

新增固定对照：统一买价上限（默认均为昨收+3%）与分板块上限（默认主板3%、创业板/科创板6%）。
上限约束含滑点的买价，不是昨天涨幅；除权日使用当日除权参考昨收。创业板以30开头，科创板以68开头。
交易所真实涨跌停价仍来自stk_limit接口；分板块买价上限不替代交易规则，不放宽止损或过热过滤。
两种买价各运行11组：原版四维等权、逐一将某维提高到40且其余各20、连续RS1/2/3/5日、昨日涨幅对照、过热观察。
连续RS严格指最近N个市场交易日每天都跑赢同一板块，包含信号日；不是N日累计涨幅跑赢，缺数据或中途换行业不通过。
RS条件叠加在原版股票池和方向门槛上；各组共用原始百分位评分，只改变一个权重或额外RS条件，过滤后不重算百分位。
权重对照保留原版相对强度维度内部的3/5日与持续性定义；RS窗口对照保留25/25/25/25，不同时改权重。
所有实验均保持最多3只、既定持有期、5%/10%止盈和10%止损。实验之间各自管理同股持仓，结果不能相加。
分板块报告区分共同成交、放宽后因高开新增、后续持仓路径变化新增/减少；新增只数并不代表新增有效盈利。
独立信号路径把每条入选信号单独假设买入，允许同股重叠；仅完整5日窗口用于惯性占比，不是账户收益。
路径窗口固定为买入起5个市场交易日，最高涨幅可能出现在实际止盈/止损之后；可卖最高涨幅不含买入当天。
各实验报告信号覆盖、未完成样本、年度、名次和交易板块；高收益但极少样本不能自动胜出。
M1.1旧对照不自动采用历史最优实验。22组是预先固定的假设对照，不进行参数寻优。
2025-09至2026-09样本已用于提出这些假设，应标记为探索样本；其他未参与调参区间/前向记录才用于验证。

这是全新独立策略，未经过参数寻优，也未证明盈利。无自动下单功能。
每日收盘选股，下一市场交易日开盘尝试买入。每日最多3个名额，因高开等原因未成交不补位。
板块口径：申万2021二级行业，以供应商历史纳入/剔除区间匹配当日成分；剔除日不再纳入。
板块涨幅是当日有交易成分股的等权涨幅，不是软件中的市值加权行业指数，也不是概念榜。
至少5只有行情成分才参与排名。只选昨日涨幅第一板块；其涨幅<=0则空仓，不改选第二板块。
科技池：电子/计算机/通信/国防军工，以及包含自动化、机器人、仪器仪表、半导体、光伏、风电、
电池、电网、医疗器械、电子或金属新材料的二级行业整组；可切全A沪深股票。
基础池默认收盘>=10元、当日流通市值50—1000亿元。不是用今天市值过滤过去。
历史ST/退市整理依历史名称区间排除，名称缺失者禁止入选；当下名称不反向筛选历史。
需要21根连续市场交易日日线，缺行情不前向填充；5日涨幅>0，且3日或5日至少一个跑赢行业。
四维各25%：相对强度=rs3/rs5/前一日为止的3日超额/5日跑赢天数；效率=5日有向效率/回撤/低点抬高比例；
量价=5日成交额加权收盘位置/上涨日与下跌日平均成交额之比/当日相对20日成交额与收盘位置配合；
收盘位置=近3日收盘位置/距20日高点。每项在当日第一板块的基础合格股中取居中百分位后平均。
无历史最优权重、无训练模型、无上涨概率；评分只用于同日排序。
近5日涨幅超过25%、偏离MA10超过10%开始各最多扣10分，扣分线是固定研究假设。
近5日累计涨幅>=50%，或近6根K线中先发生的低点至随后交易日高点涨幅>=50%，禁止入选。
不拿同日最低和最高价臆造日内先后顺序；过热组仅做观察，绝不是买入清单。
综合动量与昨日涨幅排序共用相同基础池/禁入条件，分别独立选前三名；并列按股票代码排序。
预设最高买价=下一日除权参考昨收*(1+该实验对应买价上限)；含买入滑点后仍须不超该价。不是事后按开盘选完再假装开盘成交。
开盘涨停或跌停一律不买；既定候选中未成交不以当日走势补位。
T+1：买入当日不能止盈、不能止损，买入当日达到5%只作诊断。
止盈5%与10%是两个独立版本，止损10%；持有默认5个市场交易日含买入日，第5日收盘到期退出。
停牌、跌停无法成交时可能超过5日，10%不是最大亏损保证。观察期不足则保留未平仓，不计作零收益。
同一股票在每个排名组/止盈版本内不重复加仓，原持仓退出当日也不再次开仓；不强制总账户三仓。
日线撮合：先判断开盘跳空；日内同时触及止盈止损，按止损先发生并标记歧义；需要分钟数据才能消除歧义。
开盘跌停，即使日后打开也不臆造开盘卖出，延迟至后续可成交日。触及跌停的止损/到期退出也延迟。
默认买卖手续费各万3（综合佣金/过户等比例假设），滑点各千1；卖出另计历史印花税：2023-08-28前千1、之后万5。
未模拟券商每笔最低佣金或盘口成交容量；实际小额交易成本可能更高。
指标用价格*当日复权因子的比值；交易触发用等价复权价，涨跌停比较用原始价；跨除权收益为复权收益近似。
逐笔数据完整的平仓交易才进入净收益汇总；未平仓和数据缺口单独统计，不可只看平仓赢家。
最高涨幅/最低跌幅是固定观察窗口的路径描述，包含实际策略可能已退出后的行情，不是可兑现利润。
所有收益是等额单股交易审计，无总资金约束；不可当作账户收益、复利或年化。
包含年度、排名、同日一篮子审计、盈亏金额比及去除最大5笔收益；同日和相邻日交易高度相关。
行业历史分类本身仍依赖供应商记录，SW2021历史重构不等于原始时点快照；不保证无数据修订偏差。
应冻结参数再用未参与调参的年份/前向记录验证，不能只看最近一段结果。
"""


def save_report(root, cfg, candidates, selected, trades, panel, calendar, data_hash, mode, events=None):
    root = Path(root)
    folder = Path(tempfile.mkdtemp(prefix="M16_", dir=root))
    candidates = add_filter_ranks(candidates)
    tables = {"candidates": candidates, "selected": selected, "trades": trades,
              "sector_daily": panel[panel.date.between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))].copy()}
    tables["summary"] = summarize(trades, ["group", "take_profit"])
    tables["experiment_design"] = experiment_design(cfg)
    tables["experiment_summary"] = experiment_summary(trades, selected, calendar, cfg)
    tables["gap_incremental_trades"], tables["gap_incremental_summary"] = gap_comparison(trades)
    decisions = screening_decisions(candidates, cfg)
    tables["filter_decisions"] = decisions
    tables["filter_execution_changes"], tables["filter_execution_summary"] = filter_execution_audit(trades, decisions)
    if events is not None:
        tables.update(efficiency_trial(candidates, selected, events, trades, cfg, mode))
        tables.update(attribution_reports(events, candidates, trades, cfg))
        tables.update(validation_reports(events, selected, candidates, trades, cfg, mode))
        tables.update(top10_reports(events, candidates, cfg))
        tables["independent_signal_paths"] = events
        tables["path_summary"] = path_summary(events)
        ledger = filter_audit(decisions, events)
        tables["filter_signal_outcomes"] = ledger
        tables["filter_impact"] = filter_impact_summary(ledger)
        tables["filter_impact_by_year"] = filter_impact_summary(ledger, ["year"])
        tables["filter_impact_by_rank"] = filter_impact_summary(ledger, ["baseline_rank"])
        if not cfg.validation_only:
            tables["selection_overlap_signals"], tables["selection_overlap_summary"] = selection_comparison(events, candidates)
            tables["factor_buckets"] = factor_bucket_summary(events, candidates)
    if not trades.empty:
        tr = trades.copy()
        tr["year"] = pd.to_datetime(tr.signal_date).dt.year
        tables["by_year"] = summarize(tr, ["group", "take_profit", "year"])
        tables["by_rank"] = summarize(tr, ["group", "take_profit", "rank"])
        tables["by_board"] = summarize(tr, ["group", "take_profit", "board"])
        closed = tr[tr.status.eq("已平仓") & ~tr.data_issue.eq(True)]
        tables["daily_cohorts"] = closed.groupby(["group", "take_profit", "signal_date"], as_index=False).agg(
            closed_count=("net_return", "size"), mean_net=("net_return", "mean"),
            min_net=("net_return", "min"), win_count=("net_return", lambda x: int((x > 0).sum())))
        tables["execution_reasons"] = tr.groupby(["group", "take_profit", "status", "reason"], dropna=False).size().rename("count").reset_index()
    latest = panel.loc[panel.date.le(pd.Timestamp(cfg.end)), "date"].max()
    tables["latest_candidates"] = candidates[candidates.date.eq(latest)].copy()
    tables["latest_selected"] = selected[selected.date.eq(latest)].copy()
    top_pool = top10_candidates(candidates, cfg)
    tables["latest_top10"] = top_pool[top_pool.date.eq(latest)].copy()
    reasons = candidates[["date", "exclude_reason"]].copy()
    reasons["reason"] = reasons.exclude_reason.str.split("；")
    reasons = reasons.explode("reason")
    tables["exclusions"] = reasons[reasons.reason.ne("")].groupby("reason").size().rename("count").reset_index()
    for key, f in tables.items():
        f.to_csv(folder/(key+".csv"), index=False, encoding="utf-8-sig")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = dict(version=VERSION, mode=mode, config=asdict(cfg), latest_signal=ds(latest),
                    data_start=ds(calendar[0]), data_end=ds(calendar[-1]), data_hash=data_hash,
                    source_sha256=source_hash, created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    verified_profitability=False, execution="保守日线近似", tables=list(tables),
                    research_design="M1.6：低前涨幅选后剔除效率最高三分之一；等机会与持仓约束重放；逐月贡献",
                    experiment_count=len(experiments(cfg)),
                    tuning_performed=False, rs_definition="最近N个市场交易日每天收益严格高于同一板块；含信号日；不是累计跑赢",
                    independent_paths="允许重叠的独立信号5日观察；路径可能发生于止盈/止损之后，不是兑现利润")
    manifest["filter_protocol"] = dict(cutoff_percentile=25, minimum_pool=4, replacement=False,
        rank_preserved=True, filters_combined=False, threshold_tuned=False,
        same_opportunity_denominator="基准独立信号有效平仓数；未知不归零", available_event_targets=sorted(events.take_profit.unique().tolist()) if events is not None and not events.empty else [])
    manifest["validation_protocol"] = validation_protocol(cfg, mode)
    manifest["efficiency_protocol"] = efficiency_protocol(cfg)
    manifest["attribution_protocol"] = dict(note=ATTR_NOTE, features=[k for k, _, _ in ATTR_FEATURES],
        primary_target=.10, selection_rules_changed=False, automatic_feature_selection=False,
        eligible_daily_pool_minimum=4, cross_period_minimum_count=30)
    manifest["top10_protocol"] = dict(enabled=cfg.research and not cfg.validation_only, observation_size=10, select_n=cfg.top_n,
        observation_is_portfolio=False, signal_day_excluded_from_pre3=True,
        models=[dict(experiment=k, group=g, sort=c, ascending=a) for k, g, c, a in TOP10_MODELS],
        gain15_is_filter=False, selection_before_next_open=True, execution_backfill=False,
        paired_denominator="同一信号日固定名额；未选/明确未成交为空仓；任一已选未知则整日未知",
        sample_role="2025-09至2026-09为已参与假设形成的探索样本")
    (folder/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder/"规则与口径.txt").write_text(RULES, encoding="utf-8")
    archive = folder/"回测审计.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in list(tables):
            z.write(folder/(name+".csv"), name+".csv")
        for name in ["manifest.json", "规则与口径.txt"]:
            z.write(folder/name, name)
    return str(folder)


def run_online(token, root, cfg, scan=False, progress=lambda s: None):
    cfg.validate()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    end = min(pd.Timestamp(cfg.end), ready_day())
    if end < pd.Timestamp(cfg.start) and not scan:
        raise ValueError("所选开始日尚无完整收盘数据")
    data_start = (end if scan else pd.Timestamp(cfg.start))-pd.Timedelta(days=180)
    data_end = end if scan else min(end+pd.Timedelta(days=45), ready_day())
    client = DataClient(token, root, progress)
    calendar = client.calendar(data_start, data_end)
    valid = calendar[calendar <= end]
    if not len(valid):
        raise DataError("截止日前没有交易日")
    actual_end = valid[-1]
    cfg = replace(cfg, start=ds(actual_end) if scan else cfg.start, end=ds(actual_end))
    basic, member, names = client.universe(cfg)
    store = client.download(calendar)
    try:
        progress("构造历史行业排名")
        panel = sector_panel(store, member, calendar, cfg)
        candidates = build_candidates(store, basic, member, names, panel, calendar, cfg, progress)
        selected = select_candidates(candidates, cfg)
        progress("选股完成，回放两档止盈及同池对照")
        events = []
        trades = run_trades(store, selected, calendar, cfg, progress, event_records=events,
                            observations=top10_observation(candidates, cfg))
        return save_report(root, cfg, candidates, selected, trades, panel, calendar,
                           store.fingerprint(calendar[0], calendar[-1]), "正式数据_未验证盈利", pd.DataFrame(events))
    finally:
        store.close()


def demo_data(root):
    """仅为可重复软件演示生成的合成行情，不含任何真实股票收益。"""
    rng = np.random.default_rng(620926)
    calendar = pd.bdate_range("2024-01-02", periods=130)
    member, basic, names, bars = [], [], [], []
    for sec in range(6):
        wave = .008*np.sin(np.arange(len(calendar))/7+sec) + rng.normal(0, .006, len(calendar))
        for i in range(8):
            code = (f"{600000+sec*100+i}.SH" if i%3 == 0 else
                    f"{300000+sec*100+i}.SZ" if i%3 == 1 else f"{688000+sec*100+i}.SH")
            name = f"演示{sec+1}-{i+1}"
            member.append(dict(ts_code=code, l2_code=f"DEMO{sec}", l2_name=f"演示行业{sec+1}",
                               in_date="20200101", out_date=None))
            basic.append(dict(ts_code=code, name=name, list_date="20100101", delist_date=None))
            names.append(dict(ts_code=code, name=name, start_date=pd.Timestamp("2010-01-01"), end_date=pd.NaT))
            prev = 20.0 + i*2
            for t, d in enumerate(calendar):
                gap = float(np.clip(rng.normal(.001, .006), -.05, .07))
                if board_name(code) != "主板" and (t+i)%17 == 0:
                    gap = .045  # 合成数据故意覆盖3%—6%开盘区间。
                ret = float(np.clip(wave[t]+.0006*(7-i)+rng.normal(0, .013), -.085, .085))
                op, cl = prev*(1+gap), prev*(1+ret)
                limit = .10 if board_name(code) == "主板" else .20
                up_limit, down_limit = round(prev*(1+limit), 2), round(prev*(1-limit), 2)
                high = min(up_limit, max(op, cl)*(1+abs(rng.normal(0, .008))))
                low = max(down_limit, min(op, cl)*(1-abs(rng.normal(0, .008))))
                bars.append(dict(ts_code=code, trade_date=ds(d), open=op, high=high, low=low, close=cl,
                                 pre_close=prev, vol=100000, amount=float(rng.lognormal(12, .3)),
                                 circ_mv=2000000, turnover_rate=2.0, adj_factor=1.0,
                                 up_limit=up_limit, down_limit=down_limit))
                prev = cl
    store = MarketStore(Path(root)/"demo.sqlite")
    for day, f in pd.DataFrame(bars).groupby("trade_date"):
        store.put(day, f)
    return store, pd.DataFrame(basic), normalize_members(pd.DataFrame(member)), pd.DataFrame(names), calendar


def run_demo(root, options=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    store, basic, member, names, calendar = demo_data(root)
    cfg = replace(options or Config(), start=ds(calendar[35]), end=ds(calendar[-12]), scope="演示合成数据")
    try:
        panel = sector_panel(store, member, calendar, cfg)
        candidates = build_candidates(store, basic, member, names, panel, calendar, cfg)
        selected = select_candidates(candidates, cfg)
        events = []
        trades = run_trades(store, selected, calendar, cfg, event_records=events,
                            observations=top10_observation(candidates, cfg))
        return save_report(root, cfg, candidates, selected, trades, panel, calendar,
                           store.fingerprint(calendar[0], calendar[-1]), "演示合成数据_禁止解读为真实收益", pd.DataFrame(events))
    finally:
        store.close()


DISPLAY = {
    "source_period": "来源区间", "basis": "归因口径", "outcome": "交易结果", "contrast": "对照结果",
    "month": "信号月份", "excluded_month": "剔除的信号月份", "sample_phase": "样本属性",
    "diagnostic_tail": "诊断类别",
    "eff_pct": "效率池内分位", "eff_pool_count": "效率有效候选数", "eff_keep": "是否保留", "eff_reason": "效率过滤判定",
    "return_rank": "原涨幅名次", "rank_band": "原涨幅排名段", "pre3": "信号前3日涨幅%",
    "rs_accel": "当日相对强度提升(百分点)", "pre3_band": "信号前3日涨幅区间",
    "signal_gain_band": "信号日涨幅区间", "signal_limit_up": "信号日收盘涨停",
    "signal_limit_state": "信号日收盘涨停状态", "signal_limit_ratio": "信号日实际涨停幅度%",
    "top10_pool_n": "前十实际候选数", "pre3_known": "前涨幅完整数", "accel_known": "提升指标完整数",
    "signal_days": "信号日数", "avg_pre3": "平均信号前3日涨幅%", "avg_ret1": "平均信号日涨幅%",
    "avg_ret5": "平均5日涨幅%", "comparison": "十选三方案", "period": "统计区间",
    "selected_count": "已选名额数", "unknown_count": "结果未知数", "cash_slots": "明确空仓名额数",
    "slot_mean": "方案每名额收益%", "baseline_slot_mean": "基准每名额收益%",
    "paired_delta": "配对差(百分点)", "pair_complete": "双方结果已知",
    "filter_label": "过滤维度", "filter_cutoff": "拒绝分位界限", "filter_pass": "是否保留",
    "filter_reason": "过滤判定", "filter_pct": "当日池内分位", "filter_pool_n": "当日合格池数量",
    "baseline_rank": "原始涨幅名次", "cohort": "选股交集", "dimension": "维度", "factor_bucket": "当日分位区间",
    "change_type": "成交变化", "reference_side": "收益参照来源",
    "信号均值_ret1": "入选时昨日涨幅均值%", "信号均值_ret5": "入选时5日涨幅均值%",
    "experiment": "实验编号", "family": "实验类型", "gap_policy": "买价规则", "rs_days": "连续跑赢日数",
    "board": "交易板块", "gap_limit": "含滑点买价溢价上限%", "base_score": "原版等权分",
    "rs1": "1日超额(百分点)", "rs2": "2日累计超额(百分点)", "ret2": "2日涨幅%",
    "increment_type": "放宽影响", "eligible_pool_n": "当日合格候选数",
    "date": "信号日", "signal_date": "信号日", "ts_code": "代码", "name": "名称",
    "sector_name": "板块", "sector": "板块代码", "sector_rank": "板块名次", "members": "交易成分数",
    "rank": "名次", "group": "组别", "take_profit": "止盈目标%", "close": "收盘价", "mv_yi": "流通市值(亿)",
    "score": "综合分", "strength_score": "相对强度百分位分", "efficiency_score": "上涨效率百分位分",
    "volume_score": "量价百分位分", "position_score": "收盘位置百分位分", "heat_penalty": "过热扣分",
    "ret1": "当日涨幅%", "ret3": "3日涨幅%", "ret5": "5日涨幅%", "rs3": "3日超额(百分点)",
    "rs5": "5日超额(百分点)", "lead_pre3": "启动领先(百分点)", "persistence": "5日跑赢比例%",
    "runup5": "5日最大先低后高涨幅%", "sector_ret1": "板块当日涨幅%", "breadth": "板块上涨家数比例%",
    "bias10": "距MA10%", "drawdown5": "5日最大回撤%", "volatility5": "5日波动率%",
    "turnover_rate": "换手率%", "amount_ratio": "成交额比", "max_buy_reference": "最高买价参考",
    "exclude_reason": "排除原因", "base_reason": "基础排除原因", "overheat": "暴涨禁入",
    "entry_date": "计划/实际买入日", "exit_date": "卖出日", "entry_price": "含滑点买价", "exit_price": "含滑点卖价",
    "net_return": "净收益%", "gross_return": "毛收益%", "mark_return": "未扣费估值涨幅%",
    "mark_date": "估值日", "gap": "开盘缺口%", "hold_days": "市场持有日", "status": "状态", "reason": "原因",
    "ambiguous": "同日双触及", "deferred": "曾延迟卖出", "data_issue": "行情缺口", "horizon_complete": "观察窗口完整",
    "sellable_mfe5": "可卖日期最高涨幅%(非兑现)", "mae5": "窗口最大下探%", "buy_day_hit5": "买入当天曾达5%(不可卖)",
    "year": "年份", "count": "次数", "closed_count": "已平仓数量", "win_count": "盈利数量",
    "mean_net": "该信号日已平仓均益%", "min_net": "该信号日最差收益%",
}
PERCENT_COLUMNS = {"take_profit", "ret1", "ret3", "ret5", "rs3", "rs5", "lead_pre3", "persistence",
    "runup5", "sector_ret1", "breadth", "bias10", "drawdown5", "volatility5", "net_return",
    "gross_return", "mark_return", "gap", "sellable_mfe5", "mae5", "mean_net", "min_net"}
PERCENT_COLUMNS.update({"rs1", "rs2", "ret2", "gap_limit"})
PERCENT_COLUMNS.update({"pre3", "rs_accel", "signal_limit_ratio", "avg_pre3", "avg_ret1", "avg_ret5",
                        "slot_mean", "baseline_slot_mean", "paired_delta"})
PERCENT_COLUMNS.update({"信号均值_ret1", "信号均值_ret5"})
DISPLAY.update({f"rs_all_{n}": f"连续{n}日跑赢" for n in [1, 2, 3, 5]})
DISPLAY.update({'signal_'+key: '信号日特征：'+label for key, label, _ in ATTR_FEATURES})
DISPLAY.update({'pool_pct_'+key: '当日池内分位：'+label for key, label, _ in ATTR_FEATURES})
PERCENT_COLUMNS.update({'signal_'+key for key, _, scale in ATTR_FEATURES if scale == 100.})


def show_table(st, frame):
    # 新版Streamlit改用width；保留旧项目1.32兼容性。
    version = tuple(int(x) for x in st.__version__.split(".")[:2])
    width = {"width": "stretch"} if version >= (1, 49) else {"use_container_width": True}
    st.dataframe(display_frame(frame), hide_index=True, **width)


def show_research(st, folder):
    grid = report_table(folder, "experiment_summary")
    if grid.empty:
        st.info("此报告没有实验对照结果，请重新运行。")
        return
    st.subheader("全部实验结果")
    st.caption("这是探索性对照，不自动推荐历史收益最高的一组。净均益是独立交易均值，非账户收益；检查各年、样本量和信号覆盖。")
    family = st.selectbox("查看实验类型", ["全部", "效率过滤对照", "前十选三假设", "涨幅前三过滤", "等权基准", "单维加权", "连续相对强度", "涨幅对照", "过热观察"], key="research_family")
    policy = st.radio("买入上限规则", ["两种都看", "统一上限", "分板块上限"], horizontal=True, key="research_policy")
    target = st.radio("止盈对照", ["5%", "10%"], horizontal=True, key="research_target")
    f = grid[grid.take_profit.eq(.05 if target == "5%" else .10)]
    if family != "全部":
        f = f[f.family.eq(family)]
    if policy != "两种都看":
        f = f[f.gap_policy.eq(policy)]
    cols = ["group", "gap_policy", "选股日数", "选股覆盖%", "合格候选股日数", "有效平仓数", "未平仓数",
            "胜率%", "净均益%", "相对涨幅均益差(百分点)", "相对等权均益差(百分点)", "止盈占比%", "最差单笔%", "剔除最大5笔后净均益%"]
    show_table(st, f.reindex(columns=cols))
    st.subheader("放宽后新增的交易表现")
    increment = report_table(folder, "gap_incremental_summary")
    if not increment.empty:
        inc = increment[increment.experiment.eq("B0") & increment.take_profit.eq(.05 if target == "5%" else .10)]
        show_table(st, inc.reindex(columns=["increment_type", "board", "成交数", "有效平仓数", "未平仓数", "胜率%", "净均益%", "最差单笔%"]))
        st.caption("这里展示等权基准。共同成交也列出；路径变化减少表示原版可买、放宽组被已有持仓阻塞。各实验完整明细可下载。")
    else:
        st.info("尚无可比较成交，或本次未启用完整对照。")
    st.subheader("上涨惯性：独立信号的完整5日路径")
    paths = report_table(folder, "path_summary")
    if not paths.empty:
        show_table(st, paths[paths.group.isin(f.group)])
    st.caption("每条信号独立假设入场，允许持仓重叠；可卖日期指买入后的第2—5个市场交易日。曾达目标不等于策略获利，可能先止损后反弹。此表不因上方5%/10%切换改变路径。")
    with st.expander("查看固定实验设计和买入上限"):
        st.caption("综合分仅在综合评分排序组起作用；涨幅与十选三组不按旧综合分排序。")
        show_table(st, report_table(folder, "experiment_design"))


def show_screening(st, folder):
    st.subheader("昨日涨幅前三：过滤是否真正改善结果")
    st.caption("四个过滤独立运行：剔除当日合格池内对应维度末25%的原始前三候选；不补位，池不足4只不判定。默认不组合、不调权重。")
    tp = .05 if st.radio("过滤验证止盈", ["5%", "10%"], horizontal=True, key="filter_target") == "5%" else .10
    grid = report_table(folder, "experiment_summary")
    if not grid.empty:
        view = grid[grid.gap_policy.eq("统一上限") & grid.take_profit.eq(tp) &
                    grid.family.isin(["涨幅对照", "等权基准", "涨幅前三过滤"])]
        st.caption("实际模拟：同组同股不重复持仓。此表的交易集合可能因过滤后的持仓路径而改变。")
        show_table(st, view.reindex(columns=["group", "选股日数", "选股覆盖%", "有效平仓数", "未平仓数",
                                            "胜率%", "净均益%", "相对涨幅均益差(百分点)", "剔除最大5笔后净均益%", "最差单笔%"]))
    impact = report_table(folder, "filter_impact")
    if impact.empty:
        st.info("本次没有可用过滤审计。请启用M1.2过滤验证并运行；无候选时不会用0收益伪装成有效结果。")
        return
    st.subheader("避免亏损与误杀盈利：相同机会集合")
    st.info("以下把每条涨幅基准信号独立假设买入，允许重叠。分母固定为基准有效平仓机会数，被过滤机会按空仓0处理；未完成结果不归零。这不是账户收益。")
    part = impact[impact.take_profit.eq(tp)]
    show_table(st, part.reindex(columns=["group", "原始信号数", "规则拒绝数", "有效平仓机会数", "拒绝平仓数",
        "避免亏损笔数", "错过盈利笔数", "保留单笔均益%", "拒绝单笔均益%", "基准机会均益%",
        "过滤后等机会均益%", "等机会改善(百分点)"]))
    st.caption("等机会改善为正，表示被过滤的这批机会合计亏损；为负，表示错过的盈利超过避免的亏损。仅看保留均益上升不够。")
    with st.expander("分年度、原始名次及未完成机会"):
        for name in ["filter_impact_by_year", "filter_impact_by_rank"]:
            table = report_table(folder, name)
            if not table.empty:
                show_table(st, table[table.take_profit.eq(tp)])
        show_table(st, part.reindex(columns=["group", "原始信号数", "信号保留率%", "小候选池保留数",
            "未成交数", "待买入或未平仓数", "数据问题数", "避免亏损贡献(百分点)", "错过盈利贡献(百分点)"]))
    st.subheader("综合评分与涨幅排序：共同选中与各自独有")
    overlap = report_table(folder, "selection_overlap_summary")
    if not overlap.empty:
        show_table(st, overlap[overlap.take_profit.eq(tp)].reindex(columns=["cohort", "信号数", "有效平仓数",
            "未平仓数", "胜率%", "净均益%", "信号均值_ret1", "信号均值_ret5", "最差单笔%"]))
    st.caption("这也是独立信号审计，不受之前持仓阻塞影响，数量可能与实际成交表不同。完整特征与逐笔记录在下载中。")
    with st.expander("四个维度的固定分位区间表现"):
        buckets = report_table(folder, "factor_buckets")
        if not buckets.empty:
            show_table(st, buckets[buckets.take_profit.eq(tp)].reindex(columns=["dimension", "factor_bucket", "信号数",
                "有效平仓数", "未平仓数", "胜率%", "净均益%", "最差单笔%"]))
        st.caption("分位在当日完整合格池内计算；这里仅观察涨幅前三落入各区间后的表现，不据此自动寻找最佳门槛。")
    with st.expander("过滤引起的持仓路径变化"):
        changes = report_table(folder, "filter_execution_summary")
        if not changes.empty:
            show_table(st, changes[changes.take_profit.eq(tp)])
        st.caption("直接过滤/路径减少显示基准中被放弃交易的结果；共同/新增显示过滤组结果。各行不能相加当作组合收益。")
    with st.expander("最新信号的过滤判定"):
        decisions = report_table(folder, "filter_decisions")
        if not decisions.empty:
            manifest = json.loads((Path(folder)/"manifest.json").read_text(encoding="utf-8"))
            decisions = decisions[pd.to_datetime(decisions.date).eq(pd.Timestamp(manifest["latest_signal"]))]
            show_table(st, decisions.reindex(columns=["date", "group", "rank", "ts_code", "name", "ret1",
                "filter_pool_n", "filter_pct", "filter_pass", "filter_reason"]))


def show_top10(st, folder):
    ledger = report_table(folder, "top10_events")
    if ledger.empty:
        st.info("本报告没有前十完整回放。开启M1.3验证后重新运行；旧报告不能替代第4—10名的后续行情。")
        return
    st.subheader("前十候选与十选三")
    st.caption("原涨幅前三保留为基准；低前涨幅、相对强度提升是两条独立假设。15%只分组观察，不自动排除股票。")
    coverage = report_table(folder, "top10_coverage")
    a, b, c = st.columns(3)
    a.metric("有合格候选日", len(coverage))
    b.metric("可观察股票日", len(ledger)//2)
    c.metric("不足十只的日期", int(coverage.top10_pool_n.lt(10).sum()) if not coverage.empty else 0)
    tp = st.selectbox("前十验证止盈目标", [.05, .10], format_func=lambda v: f"{v:.0%}", key="top10_tp")
    grid = report_table(folder, "experiment_summary")
    grid = grid[grid.experiment.isin(["RETURN"]+[x[0] for x in TOP10_MODELS]) & grid.take_profit.eq(tp)]
    st.write("持仓回放：每日最多三只，仍持有的同股不重复买入")
    show_table(st, grid.reindex(columns=["group", "选股日数", "入选信号数", "有效平仓数", "未平仓数",
        "胜率%", "净均益%", "相对涨幅均益差(百分点)", "剔除最大5笔后净均益%", "最差单笔%"]))
    st.write("相同信号日、固定名额的配对比较")
    st.caption("每批最多3个等额名额；未选/明确未成交记空仓，未知结果剔除整对日期并列出数量。不同批次持有期重叠，不能当账户收益或复利。")
    paired = report_table(folder, "top10_paired_summary")
    show_table(st, paired[paired.take_profit.eq(tp)] if not paired.empty else paired)
    st.write("第1—10名各自表现：全部独立信号，可重叠")
    ranks = report_table(folder, "top10_rank_summary")
    show_table(st, ranks[ranks.take_profit.eq(tp)])
    views = {
        "1—3 / 4—6 / 7—10名": "top10_band_summary",
        "各交易板块的排名段": "top10_board_rank_summary",
        "各年份的排名段": "top10_year_rank_summary",
        "信号前3日累计涨幅": "top10_pre3_summary",
        "信号日是否超过15%": "top10_gain15_summary",
        "超过15%：分年份与交易板块": "top10_gain15_year_summary",
        "当日涨幅与此前涨幅交叉": "top10_gain_pre3_summary",
        "主板涨停与双创涨停分别观察": "top10_limit_up_summary",
        "换入、换出与共同选中": "top10_changes_summary",
        "换入换出：分交易板块": "top10_changes_by_board",
    }
    label = st.selectbox("查看诊断明细", list(views), key="top10_diagnostic")
    frame = report_table(folder, views[label])
    show_table(st, frame[frame.take_profit.eq(tp)] if "take_profit" in frame else frame)
    st.caption("以上仅描述本股票池；并非全市场统计。不同排名和板块分布、样本数量可能不同，不能看到某组高收益就直接采用。")
    with st.expander("上涨惯性路径：买入当天与后续可卖日期分开"):
        st.caption("只有完整5日窗口进入后续到达率；最高价可能出现在策略已止损之后，不等于能兑现的收益。买入当日冲高受T+1约束。")
        show_table(st, report_table(folder, "top10_gain15_paths"))
        show_table(st, report_table(folder, "top10_board_rank_paths"))
    with st.expander("最近前十信号与全部审计字段"):
        show_table(st, ledger[ledger.take_profit.eq(tp)].sort_values(["signal_date", "return_rank"], ascending=[False, True]).head(200))


def show_efficiency(st, tables, prefix='eff'):
    st.subheader('低前涨幅选出前三后，剔除效率最高三分之一')
    summary = tables.get('efficiency_opportunity_summary', pd.DataFrame())
    if summary.empty:
        st.info('暂无效率过滤对照。可启用M1.6重新计算，或导入M1.3至M1.6完整回测审计ZIP；没有原低前涨幅信号时不生成结果。')
        return
    st.caption('只增加这一条过滤，不补位，不重新排名。参照当日完整合格且不过热池；分位超过66.6667才拒绝，至少4个有限评分，并列用平均名次，分位不可评估时保留。')
    period = None
    if 'source_period' in summary:
        period = st.selectbox('效率对照区间', ['全部区间']+summary.source_period.unique().tolist(), key=prefix+'_period')
    tp = st.selectbox('效率对照止盈目标', [.10, .05], format_func=lambda x: '10% · 主要目标' if x == .10 else '5% · 敏感性对照', key=prefix+'_target')
    def part(name):
        f = tables.get(name, pd.DataFrame())
        if not f.empty:
            if 'take_profit' in f:
                f = f[f.take_profit.eq(tp)]
            if period and period != '全部区间' and 'source_period' in f:
                f = f[f.source_period.eq(period)]
        return f
    st.info('2024-09-20至2026-09-18两段数据已参与新规则形成，均为探索样本。新规则于2026-09-22固定，不能把本页改善当作独立验证通过。')
    st.write('完整持仓回放：是否真正改善原方案')
    show_table(st, part('efficiency_actual_summary').reindex(columns=['source_period', 'group', 'take_profit',
        '信号数', '有效平仓数', '未平仓数', '数据问题数', '胜率%', '净均益%', '净中位数%',
        '剔除最大5笔后净均益%', '最差单笔%']))
    st.caption('每组分别重放同股持仓占用；退出当日不重复开仓。过滤后原来被持仓挡住的信号可能变为成交，所以新组不是简单删除原成交记录。仍未施加账户总资金限制。')
    st.write('避免亏损与错过盈利：固定原机会分母')
    cols = ['source_period', '原始信号数', '过滤信号数', '分位缺失保留数', '原始有效平仓机会数',
        '避免亏损笔数', '错过盈利笔数', '原始机会净均益%', '保留单笔净均益%', '过滤后等机会均益%',
        '等机会改善(百分点)', '避免亏损贡献(百分点)', '错过盈利贡献(百分点)', '未知或数据问题数']
    show_table(st, part('efficiency_opportunity_summary').reindex(columns=cols))
    st.caption('独立信号允许重叠。等机会分母固定为原方案有效平仓机会数，被过滤记空仓0，未知不归零；改善=避免亏损贡献−错过盈利贡献。保留单笔均益上升本身不足以证明过滤有效。')
    st.write('同日固定名额比较与集中度')
    show_table(st, part('efficiency_summary'))
    show_table(st, part('efficiency_concentration'))
    st.caption('此处基准专指原低前涨幅。每天最多三个固定名额，任一原结果未知则整日剔除，两组都用同一日期集合；不同批次会重叠，不能复利或年化。')
    with st.expander('逐月贡献与逐一剔除月份'):
        show_table(st, part('efficiency_monthly'))
        show_table(st, part('efficiency_leave_month_out'))
        st.caption('逐月和极端日剔除是事后敏感性检查，不是可执行的交易条件。')
    with st.expander('哪些交易因持仓路径发生变化'):
        show_table(st, part('efficiency_execution_summary'))
        st.caption('直接过滤/路径减少展示原组交易；共同成交/路径新增展示过滤组交易。不同来源不能混加成组合收益。')
        f = part('efficiency_execution_changes')
        if not f.empty:
            show_table(st, f.reindex(columns=['source_period', 'signal_date', 'ts_code', 'name',
                'change_type', 'reference_side', 'status', 'entry_date', 'exit_date', 'net_return', 'reason']).tail(150))
    with st.expander('过滤判定明细（最近100条，完整记录见下载）'):
        d = part('efficiency_decisions')
        if not d.empty:
            show_table(st, d.sort_values('date', ascending=False).reindex(columns=['source_period', 'date',
                'ts_code', 'name', 'rank', 'return_rank', 'efficiency_score', 'eff_pool_count',
                'eff_pct', 'eff_keep', 'eff_reason']).head(100))
    with st.expander('逐月成交表现与样本属性'):
        show_table(st, part('efficiency_actual_monthly'))
        show_table(st, part('efficiency_by_sample'))


EFFICIENCY_TABLES = ['efficiency_decisions', 'efficiency_opportunities', 'efficiency_opportunity_summary',
    'efficiency_actual_trades', 'efficiency_actual_summary', 'efficiency_paired_days', 'efficiency_summary',
    'efficiency_monthly', 'efficiency_leave_month_out', 'efficiency_concentration', 'efficiency_by_sample',
    'efficiency_extreme_days', 'efficiency_actual_monthly', 'efficiency_execution_changes', 'efficiency_execution_summary']


def show_attribution(st, tables, prefix='attr'):
    summary = tables.get('attribution_summary', pd.DataFrame())
    st.subheader('交易结果归因：哪些股票没有形成上涨惯性')
    if summary.empty:
        st.info('本报告暂无归因表。持有上限需为5日；M1.3/M1.4旧ZIP可在“已有结果归因”直接导入。')
        return
    st.caption('仅分析10%止盈目标。结果标签在交易结束后确定；买入前特征仅来自信号日收盘，不参与修改选股。')
    col1, col2 = st.columns(2)
    bases = [x for x in ATTR_BASES if x in summary.basis.unique()]
    basis = col1.selectbox('归因口径', bases, key=prefix+'_basis')
    groups = [x for x in VALIDATION_GROUPS if x in summary.group.unique()]
    group = col2.selectbox('归因方案', groups, index=groups.index(TOP10_MODELS[0][1]) if TOP10_MODELS[0][1] in groups else 0, key=prefix+'_group')
    def subset(name):
        f = tables.get(name, pd.DataFrame())
        if not f.empty:
            f = f[f.basis.eq(basis) & f.group.eq(group)]
        return f
    st.caption('独立信号用于研究选股，允许同股信号重叠；持仓回放跳过仍持有的同股。两种口径的数量不同，都不是账户收益。')
    show_table(st, subset('attribution_summary'))
    st.caption('占比的分母是全部信号，包含未买入和未完成；均益仅用有效平仓。各结果的均益贡献相加等于全部有效平仓均益。')
    st.caption('止损单独列出，五日最高价可能在止损之后；“达5%后到期亏损”仅认定完整五日、正常到期且净亏损的交易。')
    contrast = st.selectbox('止盈交易与哪类结果比较', list(ATTR_CONTRASTS), key=prefix+'_contrast')
    consistency = subset('attribution_consistency')
    if not consistency.empty:
        st.write('两段历史的差异方向')
        f = consistency[consistency.contrast.eq(contrast)]
        columns = ['特征', '中位数差_区间1', '中位数差_区间2', '原值方向检查',
                   '池内分位差_区间1', '池内分位差_区间2', '池内分位方向检查']
        show_table(st, f.reindex(columns=columns))
        st.caption('差值均为“止盈组减对照组”；区间1较早、区间2较晚。每组至少30条有效记录才显示方向描述，这不是显著性检验。')
    else:
        st.write('买入前的特征差异')
    features = subset('attribution_features')
    if not features.empty:
        f = features[features.contrast.eq(contrast)]
        if consistency.empty:
            show_table(st, f.drop(columns=['basis', 'group', 'contrast', 'feature'], errors='ignore'))
        else:
            with st.expander('查看两个区间的样本数与原始中位数'):
                show_table(st, f.drop(columns=['basis', 'group', 'contrast', 'feature'], errors='ignore'))
    st.caption('池内分位按同日完整合格且不过热候选池计算；小于4只或无横截面差异时留空。板块涨幅/广度同日相同，不计算池内分位。原值差异可能受市场日期和主板/双创构成影响。')
    st.caption('16项固定特征均展示，不按结果寻找最佳门槛。同向差异仍可能是偶然或板块构成影响，不能当作预测结论。')
    feature = st.selectbox('查看固定三等分表现', [k for k, _, _ in ATTR_FEATURES],
        format_func=lambda k: next(label for key, label, _ in ATTR_FEATURES if key == k), key=prefix+'_feature')
    buckets = subset('attribution_buckets')
    if not buckets.empty:
        show_table(st, buckets[buckets.feature.eq(feature)].drop(columns=['basis', 'group', 'feature'], errors='ignore'))
    st.caption('三等分界限固定在当日合格池的1/3和2/3，不是事后优化阈值；这里统计全部已选信号，包含输赢两端以外的交易。缺失记录不分档，最高价到达不等于兑现收益。')
    with st.expander('逐笔核对结果与买入前特征'):
        ledger = subset('attribution_ledger')
        if not ledger.empty:
            outcomes = sorted(ledger.outcome.unique())
            selected = st.multiselect('查看结果类型', outcomes, default=[x for x in ATTR_CONTRASTS if x in outcomes], key=prefix+'_outcomes')
            columns = ['source_period', 'signal_date', 'ts_code', 'name', 'outcome', 'net_return', 'reason',
                       'hold_days', 'sellable_mfe5', 'signal_'+feature, 'pool_pct_'+feature]
            show_table(st, ledger[ledger.outcome.isin(selected)].sort_values('signal_date', ascending=False).reindex(columns=columns).head(200))
            st.caption('页面显示最近200条，完整记录在下载的归因审计中；signal_字段只取信号日，pool_pct_字段为该日合格池分位。')


def show_archive_review(st):
    st.write('复用已有独立信号路径，重放效率过滤后的持仓约束')
    st.caption('上传M1.3的2026-09-18和M1.4的2025-09-19完整回测审计ZIP。无需Token或下载行情；M1.5二次归因ZIP不包含完整两档路径，不能代替原回测包。')
    uploads = st.file_uploader('上传一至两份回测审计ZIP', type=['zip'], accept_multiple_files=True, key='attribution_uploads')
    if st.button('分析已有结果', type='primary'):
        try:
            with st.spinner('正在核对信号、分组与特征差异…'):
                review = review_archives([(f.name, f.getvalue()) for f in uploads])
            st.session_state['attribution_review'] = review
        except Exception as exc:
            st.error('本次归因未完成：'+str(exc))
    review = st.session_state.get('attribution_review')
    if review is None:
        st.info('导入后可直接查看两个区间的结果分组、买入前特征差异和固定三等分表现。')
        return
    st.caption('以下为上一次已完成的归因；更换文件后请重新点击分析。')
    meta = review['meta']
    for i, source in enumerate(meta['sources'], 1):
        st.write(f"区间{i}：{source['start']}—{source['end']} · {source['name']} · {source['origin']}")
    for warning in meta['warnings']:
        st.warning(warning)
    st.info('两段历史现在用于寻找新特征，都属于新假设的探索样本。同向差异还需要未参与研究的数据验证。')
    if all(s.get('efficiency_replay_completed', False) for s in meta['sources']):
        st.caption('已用独立路径复现原组持仓记录，并对新过滤组重放持仓约束。原始成交价格路径复用报告记录，没有重新撮合原始行情。')
    else:
        st.warning('至少一份报告尚未完成效率回放或无法核对逻辑来源，仅显示已完成区间。更换版本或文件后请重新点击分析。')
    tabs = st.tabs(['效率过滤对照', '原规则归因'])
    with tabs[0]:
        show_efficiency(st, review['tables'], prefix='import_eff')
    with tabs[1]:
        show_attribution(st, review['tables'], prefix='import_attr')
    st.download_button('下载效率对照与归因审计', review['archive'], file_name='momentum_M1.6_efficiency.zip', mime='application/zip')
    with st.expander('归因口径与文件来源'):
        st.write(ATTR_NOTE)
        st.json(meta)


def show_validation(st, folder, manifest):
    protocol = manifest.get("validation_protocol")
    st.subheader("固定两套规则，检查收益改善能否重复出现")
    if not protocol:
        st.info("此报告没有M1.4固定验证审计。请重新运行；旧报告仍可在其他页查看。")
        return
    st.caption("原涨幅前三与前十低前涨幅保持同样交易设置。主要目标为10%止盈，5%仅作敏感性对照。程序不会自动宣布哪组已经通过验证。")
    if not protocol["standard_parameters_match"] and "演示" not in manifest["mode"]:
        st.warning("本次参数与固定标准不同，不能作为同一协议的直接验证。")
        st.json(protocol["parameter_differences"])
    labels = report_table(folder, "validation_sample_dates")
    if not labels.empty:
        st.info("本次信号包含："+"、".join(labels.sample_phase.unique()))
    st.caption("历史待验证表示日期早于已研究区间，需确认未用于本策略调参；冻结后日期也不自动证明信号曾在买入前留存。")
    tp = st.selectbox("固定验证止盈目标", [.10, .05],
        format_func=lambda v: "10% · 主要目标" if v == .10 else "5% · 敏感性对照", key="validation_tp")
    summary = report_table(folder, "validation_summary")
    if summary.empty:
        st.info("本区间没有可比较的两组信号。无候选、未完成与零收益分开处理。")
        return
    def target(name):
        f = report_table(folder, name)
        return f[f.take_profit.eq(tp)] if "take_profit" in f else f
    st.write("持仓回放：已平仓交易表现")
    grid = target("experiment_summary")
    grid = grid[grid.group.isin(VALIDATION_GROUPS)]
    show_table(st, grid.reindex(columns=["group", "选股日数", "有效平仓数", "未平仓数", "数据问题数",
        "胜率%", "净均益%", "平均盈利%", "平均亏损%", "平均持有日", "剔除最大5笔后净均益%"]))
    st.write("同一信号日、固定三个名额的配对比较")
    show_table(st, target("validation_summary"))
    st.caption("未选或明确未成交按空仓0；任一已选结果未知则整对日期剔除。信号日之间会有持仓重叠，以上不是账户日收益，不能复利或年化。")
    st.write("增量收益是否集中")
    show_table(st, target("validation_concentration"))
    st.caption("最大月占净改善可能超过100%，因为其他月份会抵消它；净改善非正时不计算该占比。另列最大月占正贡献月份比例。")
    st.write("逐月对照")
    show_table(st, target("validation_monthly"))
    st.write("逐一剔除每个月后，其余信号表现")
    show_table(st, target("validation_leave_month_out"))
    st.caption("逐月剔除及删除最大改善日仅用于事后检查集中度，不能当成可提前执行的交易规则；无需每月、每年都盈利。")
    with st.expander("分样本来源与逐月持仓回放"):
        show_table(st, target("validation_by_sample"))
        show_table(st, target("validation_monthly_trades"))
    with st.expander("改善最大与落后最大的信号日"):
        cols = ["signal_date", "diagnostic_tail", "baseline_slot_mean", "slot_mean", "paired_delta", "sample_phase"]
        show_table(st, target("validation_extreme_days").reindex(columns=cols))
    with st.expander("固定规则记录"):
        st.caption("相同协议指纹用于核对两个区间是否沿用了相同交易逻辑和参数，不能证明盈利。")
        st.json(protocol)


def display_frame(frame):
    f = frame.copy()
    for col in PERCENT_COLUMNS.intersection(f):
        f[col] = pd.to_numeric(f[col], errors="coerce")*100
    for n in range(1, 6):
        col = f"close_return_d{n}"
        if col in f:
            f[col] = pd.to_numeric(f[col], errors="coerce")*100
    f = f.rename(columns={**DISPLAY, **{f"close_return_d{n}": f"第{n}日收盘涨幅%(非策略收益)" for n in range(1, 6)}})
    numeric = f.select_dtypes(include='number').columns
    f[numeric] = f[numeric].round(3)
    return f


def report_table(folder, name):
    p = Path(folder)/(name+".csv")
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, dtype={"ts_code": str, "sector": str}, float_precision="round_trip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def show_report(st, folder):
    folder = Path(folder)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if "演示" in manifest["mode"]:
        st.warning("以下全部是合成数据演示，仅用于检查程序，不能用于选股或判断盈利。")
    st.caption(f"版本 {manifest['version']} · 信号截止 {manifest['latest_signal']} · 行情截止 {manifest['data_end']}")
    st.info("研究版：评分不是上涨概率。回测是等额单股审计，未施加账户总资金上限，不能当作账户收益。")
    labels = ["效率过滤", "结果归因", "固定验证", "最新候选", "回测结果", "排除与成交审计", "本次规则"]
    if not manifest["config"].get("validation_only", False):
        labels += ["前十观察", "旧过滤验证", "完整对照"]
    tabs = dict(zip(labels, st.tabs(labels)))
    with tabs["效率过滤"]:
        show_efficiency(st, {k: report_table(folder, k) for k in EFFICIENCY_TABLES})
    with tabs["结果归因"]:
        show_attribution(st, {k: report_table(folder, k) for k in ["attribution_ledger", "attribution_summary", "attribution_features", "attribution_buckets"]})
    with tabs["固定验证"]:
        show_validation(st, folder, manifest)
    if "前十观察" in tabs:
        with tabs["前十观察"]:
            show_top10(st, folder)
        with tabs["旧过滤验证"]:
            show_screening(st, folder)
        with tabs["完整对照"]:
            show_research(st, folder)
    with tabs["最新候选"]:
        sectors = report_table(folder, "sector_daily")
        last_day = sectors.date.max() if not sectors.empty else None
        daily = sectors[sectors.date.eq(last_day)].sort_values("sector_rank").head(10) if last_day else pd.DataFrame()
        if not daily.empty:
            st.subheader("板块排名")
            show_table(st, daily[["sector_rank", "sector_name", "sector_ret1", "breadth", "members"]])
        picks = report_table(folder, "latest_selected")
        st.subheader("下一交易日候选 · 最多三只")
        designs = report_table(folder, "experiment_design")
        groups = designs.loc[~designs.experiment.eq("HOT"), "group"].tolist() if not designs.empty else [GROUP_MAIN]
        if st.session_state.get("daily_experiment") not in groups:
            st.session_state["daily_experiment"] = GROUP_BASE if GROUP_BASE in groups else groups[0]
        chosen = st.selectbox("展示候选规则（默认涨幅基准，新方案未验证）", groups, key="daily_experiment")
        if not picks.empty:
            picks = picks[picks.group.eq(chosen)]
        st.caption("已持仓不重复加仓；高开超过最高买价或开盘涨跌停则跳过，不补位。除权日请按除权参考价调整最高买价。")
        st.caption("涨幅基准按昨日涨幅排序，综合分仅作审计；过滤组保留原始涨幅名次，空出的名额不补位。")
        cols = ["rank", "return_rank", "pre3", "rs_accel", "ts_code", "name", "board", "sector_name", "score", "gap_limit", "close", "max_buy_reference", "mv_yi",
                "ret1", "ret5", "rs5", "runup5", "strength_score", "efficiency_score", "volume_score", "position_score", "heat_penalty", "eff_pct", "eff_keep", "eff_reason"]
        if picks.empty:
            st.info("这一天没有合格候选。下方可查看排除原因；不会改选第二板块凑数。")
        else:
            show_table(st, picks.reindex(columns=cols))
        with st.expander("查看当日涨幅前十候选（观察，不是买十只）"):
            show_table(st, report_table(folder, "latest_top10").reindex(columns=["return_rank", "ts_code", "name", "board", "ret1", "pre3", "rs_accel", "ret5", "signal_limit_up", "max_buy_reference"]))
        pool = report_table(folder, "latest_candidates")
        with st.expander("查看第一板块所有候选与排除原因"):
            if not pool.empty:
                cols = ["ts_code", "name", "score", "ret5", "rs5", "runup5", "mv_yi", "exclude_reason"]
                show_table(st, pool.sort_values("score", ascending=False)[cols])
    with tabs["回测结果"]:
        summary = report_table(folder, "summary")
        st.subheader("两档止盈与排序对照")
        st.caption("净收益含比例手续费、历史印花税和双边滑点。过热观察组不属于买入策略。未平仓与数据不完整记录不计入平仓均益，数量会列出。")
        show_table(st, summary)
        for title, name in [("逐年表现", "by_year"), ("第一、第二、第三名", "by_rank"), ("主板、创业板、科创板", "by_board")]:
            f = report_table(folder, name)
            if not f.empty:
                st.subheader(title)
                show_table(st, f)
        with st.expander("按信号日查看同一批股票"):
            f = report_table(folder, "daily_cohorts")
            st.caption("仅汇总已平仓记录，未完成批次不是完整的一篮子收益；各交易有时间重叠。")
            show_table(st, f)
    with tabs["排除与成交审计"]:
        st.subheader("为什么没入选、没买到或没卖出")
        show_table(st, report_table(folder, "exclusions"))
        show_table(st, report_table(folder, "execution_reasons"))
        ledger = report_table(folder, "trades")
        if not ledger.empty:
            options = ledger.group.unique().tolist()
            if "ledger_groups" in st.session_state:
                prior = st.session_state["ledger_groups"]
                clean = [x for x in prior if x in options]
                if clean != prior:
                    st.session_state["ledger_groups"] = clean
            else:
                st.session_state["ledger_groups"] = [GROUP_BASE] if GROUP_BASE in options else []
            groups = st.multiselect("显示交易组", options, key="ledger_groups")
            sub = ledger[ledger.group.isin(groups)]
            st.caption("页面最多显示最近500条；完整记录在审计下载中。最高涨幅只是路径信息，不等于实际获利。")
            cols = ["signal_date", "group", "take_profit", "rank", "ts_code", "name", "status", "entry_date",
                    "entry_price", "exit_date", "exit_price", "net_return", "hold_days", "reason", "ambiguous",
                    "deferred", "data_issue", "mark_date", "mark_return", "horizon_complete", "buy_day_hit5", "sellable_mfe5", "mae5"]
            show_table(st, sub.sort_values("signal_date", ascending=False).reindex(columns=cols).head(500))
    with tabs["本次规则"]:
        st.json(manifest["config"])
        st.text(RULES)
        st.caption(f"代码指纹：{manifest['source_sha256'][:16]} · 数据指纹：{manifest['data_hash'][:16]}")
    with (folder/"回测审计.zip").open("rb") as f:
        st.download_button("下载完整审计结果", f, file_name=f"momentum_{manifest['version']}_{manifest['latest_signal']}.zip", mime="application/zip")


def main():
    import streamlit as st
    st.set_page_config(page_title="板块领涨惯性", page_icon="📈", layout="wide")
    st.title("板块领涨惯性")
    st.caption("M1.6 · 单一效率过滤 · 等机会审计 · 持仓约束重放")
    mode = st.radio("运行方式", ["每日选股", "区间回测", "离线演示", "已有结果归因"], horizontal=True, index=3)
    if mode == "已有结果归因":
        show_archive_review(st)
        return
    with st.sidebar:
        st.header("数据与股票池")
        default_token = os.environ.get("TUSHARE_TOKEN", "")
        if not default_token:
            try:
                default_token = str(st.secrets.get("TUSHARE_TOKEN", st.secrets.get("tushare_token", "")))
            except Exception:
                default_token = ""
        token = st.text_input("Tushare Token", value=default_token, type="password", help="仅用于请求官方数据；不写入报告。需要日线、每日指标、复权因子、涨跌停价和历史行业成分接口权限。")
        validation_only = st.checkbox("固定规则验证（推荐）", value=True,
            help="固定原涨幅前三与低前涨幅基准；默认另加M1.6单一效率过滤。关闭后可修改参数或附加旧实验。")
        efficiency_research = st.checkbox("启用M1.6效率过滤对照", value=True, help="先选低前涨幅前三，再剔除效率最高三分之一，不补位。")
        if validation_only:
            scope, min_price, min_mv, max_mv = "科技行业", 10.0, 50.0, 1000.0
            max_gap, growth_gap, hold_days, fee, slip = .03, .06, 5, .0003, .001
            research, filter_research, legacy_research = True, False, False
            st.caption("科技池 · 股价≥10元 · 流通市值50—1000亿元；买价上限3%，最长5个交易日，止损10%。")
            st.caption("手续费买卖各万3，滑点各0.1%，卖出另计历史印花税。参数与M1.3保持一致。")
        else:
            scope = st.selectbox("股票池", ["科技行业", "全A沪深"])
            min_price = st.number_input("最低股价（元）", min_value=0.0, value=10.0, step=1.0)
            min_mv = st.number_input("最低流通市值（亿元）", min_value=0.0, value=50.0, step=10.0)
            max_mv = st.number_input("最高流通市值（亿元）", min_value=1.0, value=1000.0, step=100.0)
            with st.expander("交易设置"):
                max_gap = st.number_input("统一组/主板买价溢价上限（%）", min_value=0.0, max_value=20.0, value=3.0, step=0.5)/100
                growth_gap = st.number_input("可选旧对照：创业板/科创板上限（%）", min_value=0.0, max_value=30.0, value=6.0, step=0.5,
                                             help="仅附加M1.1完整对照时使用，前十观察和十选三统一沿用上方买价上限。")/100
                hold_days = st.selectbox("最多持有市场交易日（含买入日）", [2, 3, 4, 5], index=3)
                fee = st.number_input("单边综合手续费（万分之）", min_value=0.0, max_value=50.0, value=3.0, step=1.0)/10000
                slip = st.number_input("单边滑点（%）", min_value=0.0, max_value=2.0, value=0.1, step=0.05)/100
                st.caption("卖出另计历史印花税。止盈分别5%/10%，止损10%。默认最高买价含滑点；日线同日双触及按止损先发生。")
            research = st.checkbox("运行M1.3前十验证（默认5组＋完整前十观察）", value=True,
                                   help="保留原基准，新增低前涨幅与相对强度提升两条十选三；完整观察原涨幅1—10名。")
            filter_research = st.checkbox("附加M1.2四维过滤对照", value=False, help="追加四组旧过滤，默认关闭。")
            legacy_research = st.checkbox("附加M1.1完整对照", value=False,
                                          help="仅复查买价、权重或连续RS时打开；追加19组，行情仍只下载一次。")
        with st.expander("运行说明"):
            st.code("streamlit run app.py", language="bash")
            st.caption("依赖 pandas、numpy、streamlit、tushare。沿用原项目依赖即可。四路下载、逐日缓存；首次跨年运行需要下载较多数据。")
        cache_root = str(Path(os.environ.get("MOMENTUM_CACHE_DIR", "momentum_leader_cache")).resolve())
    default_end = ready_day().date()
    if mode == "区间回测":
        preset = st.selectbox("验证区间", ["探索区间1：2024-09-20至2025-09-19", "探索区间2：2025-09-20至2026-09-18", "自定义区间"])
        if preset.startswith("探索区间1"):
            start, end = pd.Timestamp("2024-09-20").date(), pd.Timestamp("2025-09-19").date()
        elif preset.startswith("探索区间2"):
            start, end = pd.Timestamp(EXPLORED_START).date(), pd.Timestamp(EXPLORED_END).date()
        else:
            a, b = st.columns(2)
            start = a.date_input("信号开始日期", value=(pd.Timestamp(default_end)-pd.Timedelta(days=365)).date(), max_value=default_end)
            end = b.date_input("信号结束日期", value=default_end, max_value=default_end)
        st.caption(f"信号日期 {start} 至 {end}。两段预设区间都已参与M1.6假设形成；其他日期仍须核实未参与调参或事前留存。")
        st.caption("必要时读取结束日后最多45个自然日观察退出；月份按信号日归属，未平仓仍单列。")
    elif mode == "每日选股":
        end = st.date_input("收盘信号日期", value=default_end, max_value=default_end)
        start = end
        st.caption("非交易日自动使用此前最近交易日。北京时间18点前默认使用上一自然日，再按交易日历定位。")
    else:
        start = end = default_end
        st.warning("离线演示使用合成行情，检查界面和规则，不产生真实选股建议。")
    cfg = Config(start=ds(start), end=ds(end), scope=scope, min_price=min_price, min_mv=min_mv, max_mv=max_mv,
                 max_gap=max_gap, growth_gap=growth_gap, research=research, filter_research=filter_research, legacy_research=legacy_research, validation_only=validation_only, efficiency_research=efficiency_research,
                 hold_days=hold_days, buy_fee=fee, sell_fee=fee, slippage=slip)
    if st.button("开始选股 / 回测" if mode != "离线演示" else "运行离线演示", type="primary"):
        try:
            cfg.validate()
            if mode != "离线演示" and not token.strip():
                st.error("请填写Tushare Token，或在Streamlit Secrets中配置 TUSHARE_TOKEN。")
            else:
                status = st.empty()
                with st.spinner("正在计算；已下载日期会保留，下次可继续使用。"):
                    folder = run_demo(Path(cache_root)/"demo", cfg) if mode == "离线演示" else run_online(token, cache_root, cfg, scan=mode == "每日选股", progress=status.info)
                st.session_state["momentum_report"] = folder
                status.success("计算完成。请先检查未平仓、数据问题与无法成交数量。")
        except Exception as exc:
            message = str(exc).replace(token, "***") if token else str(exc)
            st.error(f"本次未完成：{message}")
            st.info("成功下载的日期已保留，修复接口或参数后可以重试。不会静默跳过失败日期生成漂亮结果。")
    folder = st.session_state.get("momentum_report")
    if folder and Path(folder).is_dir():
        st.divider()
        st.caption("以下展示上一次已完成运行的结果；修改设置后需要重新运行才生效。")
        show_report(st, folder)
    else:
        st.info("先运行每日选股或区间回测。首次使用也可用离线演示检查界面。")
        with st.expander("查看全部规则"):
            st.text(RULES)


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2024-01-02", periods=10)
        self.cfg = Config(buy_fee=0, sell_fee=0, slippage=0, research=False, filter_research=True, validation_only=False, efficiency_research=False)

    def bars(self):
        return pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0,
                             "pre_close": 100.0, "adj_factor": 1.0, "up_limit": 110.0,
                             "down_limit": 90.0, "vol": 10000.0, "amount": 10000.0,
                             "circ_mv": 1000000.0, "turnover_rate": 2.0}, index=self.calendar)

    def test_efficiency_guard_is_after_top3_no_replacement_and_preserves_baselines(self):
        c = self.filter_candidates(9)
        c['pre3'] = np.arange(9)*.01
        c['efficiency_score'] = [80., 0., 70., 10., 20., 30., 40., 50., 60.]
        cfg = Config()
        old = select_candidates(c, replace(cfg, efficiency_research=False))
        new = select_candidates(c, cfg)
        self.assertEqual([s['experiment'] for s in experiments(cfg)], ['RETURN', 'T10_PRE3', EFF_EXPERIMENT])
        pd.testing.assert_frame_equal(old.reset_index(drop=True), new[new.group.isin(VALIDATION_GROUPS)][old.columns].reset_index(drop=True))
        guard = new[new.group.eq(EFF_GROUP)]
        self.assertEqual(guard.ts_code.tolist(), ['600002.SH'])
        self.assertEqual(guard['rank'].tolist(), [2])
        self.assertEqual(guard.return_rank.tolist(), [2])
        self.assertEqual(guard.eff_keep.tolist(), [True])
        d = efficiency_decisions(c, new)
        self.assertEqual(d.eff_keep.tolist(), [False, True, False])
        # 拒绝决定不依赖下一交易日或未来收益。
        future = c.copy(); future['date'] = self.calendar[1]; future['efficiency_score'] = 999.
        joined = pd.concat([c, future], ignore_index=True)
        pd.testing.assert_frame_equal(efficiency_annotations(c), efficiency_annotations(joined).iloc[:len(c)])
        small = c.iloc[:3]
        self.assertTrue(efficiency_annotations(small).eff_keep.all())
        flat = c.assign(efficiency_score=50.)
        self.assertTrue(efficiency_annotations(flat).eff_pct.isna().all())
        self.assertEqual(len(select_candidates(flat, cfg).query('group == @EFF_GROUP')), 3)
        missing = c.copy(); missing.loc[0, 'efficiency_score'] = np.nan
        self.assertTrue(efficiency_annotations(missing).iloc[0].eff_keep)
        ties = c.assign(efficiency_score=[1., 2., 3., 4., 5., 6., 6., 8., 9.])
        a = efficiency_annotations(ties)
        self.assertAlmostEqual(a.iloc[5].eff_pct, EFF_CUTOFF)
        self.assertTrue(a.iloc[5].eff_keep)
        self.assertFalse(a.iloc[7].eff_keep)

    def test_efficiency_replay_restores_blocked_signals_and_keeps_exit_day_blocked(self):
        dates = pd.date_range('2024-01-01', periods=10)
        rows = []
        for i, exit_i in [(0, 4), (1, 3), (2, 7), (3, 5)]:
            rows.append(dict(group=TOP10_MODELS[0][1], ts_code='600001.SH', take_profit=.10,
                signal_date=dates[i], entry_date=dates[i+1], exit_date=dates[exit_i], status='已平仓',
                reason='到期退出', net_return=-.03, entry_price=20., data_issue=False))
        events = pd.DataFrame(rows)
        original = replay_independent_paths(events)
        self.assertEqual(original.status.tolist(), ['已平仓', '重复持仓跳过', '重复持仓跳过', '重复持仓跳过'])
        retained = events.iloc[1:].copy(); retained['group'] = EFF_GROUP
        updated = replay_independent_paths(retained)
        self.assertEqual(updated.status.tolist(), ['已平仓', '重复持仓跳过', '已平仓'])
        self.assertEqual(updated.net_return.dropna().tolist(), [-.03, -.03])
        unfinished = events.copy(); unfinished.loc[0, ['status', 'exit_date', 'net_return']] = ['未平仓', pd.NaT, np.nan]
        self.assertTrue(replay_independent_paths(unfinished).status.iloc[1:].eq('重复持仓跳过').all())
        delayed = events.copy(); delayed.loc[0, 'exit_date'] = dates[9]
        self.assertTrue(replay_independent_paths(delayed).status.iloc[1:].eq('重复持仓跳过').all())
        bad = events.copy(); bad.loc[0, 'status'] = '重复持仓跳过'
        with self.assertRaises(DataError):
            replay_independent_paths(bad)

    def test_efficiency_empty_trial_keeps_unknown_as_unknown(self):
        c = self.filter_candidates(6)
        c['pre3'] = np.arange(6)*.01
        c['efficiency_score'] = [100., 0., 0., 0., 0., 0.]
        cfg = replace(Config(), top_n=1)
        selected = select_candidates(c, cfg)
        self.assertTrue(selected[selected.group.eq(EFF_GROUP)].empty)
        signal = selected[selected.group.eq(TOP10_MODELS[0][1])].iloc[0]
        e = pd.DataFrame([dict(signal_date=signal.date, ts_code=signal.ts_code, group=signal.group,
            take_profit=tp, status='未平仓', entry_date=self.calendar[1], exit_date=pd.NaT,
            net_return=np.nan, data_issue=False, reason='观察截止时尚未满足退出条件',
            entry_price=20., hold_days=1, ambiguous=False) for tp in [.05, .10]])
        original = replay_independent_paths(e)
        out = efficiency_trial(c, selected, e, original, cfg)
        self.assertTrue(out['efficiency_opportunity_summary']['过滤后等机会均益%'].isna().all())
        self.assertTrue(out['efficiency_paired_days'].paired_delta.isna().all())
        self.assertTrue(out['efficiency_paired_days'].slot_mean.eq(0).all())
        self.assertTrue(out['efficiency_paired_days'].baseline_slot_mean.isna().all())
        self.assertEqual(out['efficiency_actual_trades'].group.unique().tolist(), [TOP10_MODELS[0][1]])

    def test_efficiency_online_replay_audit_and_archive_have_identical_results(self):
        with tempfile.TemporaryDirectory() as root:
            store, basic, member, names, cal = demo_data(root)
            cfg = replace(Config(), start=ds(cal[35]), end=ds(cal[60]))
            panel = sector_panel(store, member, cal, cfg)
            c = build_candidates(store, basic, member, names, panel, cal, cfg)
            selected = select_candidates(c, cfg); events = []
            trades = run_trades(store, selected, cal, cfg, event_records=events)
            e = pd.DataFrame(events)
            out = efficiency_trial(c, selected, e, trades, cfg, '演示测试')
            self.assertFalse(out['efficiency_decisions'].empty)
            self.assertTrue((~out['efficiency_decisions'].eff_keep).any())
            actual = out['efficiency_actual_trades']
            for group in [TOP10_MODELS[0][1], EFF_GROUP]:
                assert_replay_matches(trades[trades.group.eq(group)], actual[actual.group.eq(group)])
            for r in out['efficiency_opportunity_summary'].to_dict('records'):
                self.assertAlmostEqual(r['等机会改善(百分点)'], r['避免亏损贡献(百分点)']-r['错过盈利贡献(百分点)'])
                self.assertAlmostEqual(r['等机会改善(百分点)'], r['过滤后等机会均益%']-r['原始机会净均益%'])
            for r in out['efficiency_summary'].to_dict('records'):
                monthly = out['efficiency_monthly']; monthly = monthly[monthly.take_profit.eq(r['take_profit'])]
                self.assertAlmostEqual(r['配对改善(百分点)'], monthly['对全期改善贡献(百分点)'].sum())
            for tp, block in actual.groupby('take_profit'):
                changes = out['efficiency_execution_changes']
                changes = changes[changes.take_profit.eq(tp)]
                n_original = changes.change_type.isin(['共同成交', '直接过滤', '持仓路径减少']).sum()
                n_new = changes.change_type.isin(['共同成交', '持仓路径新增']).sum()
                self.assertEqual(n_original, block[block.group.eq(TOP10_MODELS[0][1])].status.isin(['已平仓', '未平仓']).sum())
                self.assertEqual(n_new, block[block.group.eq(EFF_GROUP)].status.isin(['已平仓', '未平仓']).sum())
            damaged = e.drop(e[e.group.eq(TOP10_MODELS[0][1])].index[0])
            with self.assertRaisesRegex(DataError, '独立路径不完整'):
                efficiency_trial(c, selected, damaged, trades, cfg)
            bad = trades.copy()
            idx = bad[bad.group.eq(TOP10_MODELS[0][1]) & bad.status.eq('已平仓')].index[0]
            bad.loc[idx, 'net_return'] += .01
            with self.assertRaisesRegex(DataError, '不能复现'):
                efficiency_trial(c, selected, e, bad, cfg)
            folder = save_report(root, cfg, c, selected, trades, panel, cal, 'test', '演示测试', e)
            imported = load_attribution_archive((Path(folder)/'回测审计.zip').read_bytes())
            self.assertTrue(imported['meta']['efficiency_replay_completed'])
            pd.testing.assert_frame_equal(out['efficiency_actual_summary'], imported['tables']['efficiency_actual_summary'], check_dtype=False)
            pd.testing.assert_frame_equal(out['efficiency_opportunity_summary'], imported['tables']['efficiency_opportunity_summary'], check_dtype=False)
            self.assertEqual(len(report_table(folder, 'experiment_summary')), 6)
            self.assertTrue(out['efficiency_by_sample'].sample_phase.eq('合成演示_不是验证').all())
            store.close()

    def test_outcome_classification_does_not_put_post_stop_peak_before_exit(self):
        base = dict(status='已平仓', data_issue=False, net_return=-.03, hold_days=5,
                    reason='到期退出', horizon_complete=True, sellable_mfe5=.06)
        changes = [{}, {'sellable_mfe5': .04}, {'net_return': .02},
                   {'net_return': .02, 'sellable_mfe5': .04},
                   {'reason': '止损', 'sellable_mfe5': .20, 'hold_days': 2},
                   {'reason': '止盈', 'net_return': .098, 'hold_days': 2, 'horizon_complete': False},
                   {'horizon_complete': False}, {'data_issue': True},
                   {'status': '未平仓', 'net_return': np.nan},
                   {'reason': '到期延迟退出', 'hold_days': 7},
                   {'status': '未买入', 'net_return': np.nan},
                   {'status': '重复持仓跳过', 'net_return': np.nan}]
        result = classify_outcomes(pd.DataFrame([{**base, **change} for change in changes]))
        self.assertEqual(result.tolist(), ['达5%后到期亏损', '未达5%到期亏损', '达5%后到期非亏损',
            '未达5%到期非亏损', '止损退出', '顺利止盈', '到期路径不完整', '数据问题',
            '未完成/无法分类', '延期/其他退出', '未买入', '重复持仓跳过'])

    def test_attribution_uses_signal_features_and_reconciles_all_records(self):
        candidates = self.filter_candidates()
        candidates['entry_price'] = -999  # 非白名单字段不能进入预测特征。
        candidates['net_return'] = 999
        records = []
        for i, c in candidates.iterrows():
            result = dict(status='已平仓', data_issue=False, net_return=-.02, hold_days=5,
                reason='到期退出', horizon_complete=True, sellable_mfe5=.02, entry_price=20.)
            if i < 2:
                result.update(reason='止盈', net_return=.098)
            if i == 2:
                result.update(reason='止损', net_return=-.102, sellable_mfe5=.2)
            if i == 7:
                result.update(status='未平仓', net_return=np.nan, horizon_complete=False)
            records.append(dict(**result, signal_date=c.date, ts_code=c.ts_code,
                group=GROUP_BASE, take_profit=.10, ret1=777))
        events = pd.DataFrame(records)
        out = attribution_reports(events, candidates, events, Config())
        ledger = out['attribution_ledger']
        self.assertNotIn('signal_net_return', ledger)
        self.assertNotIn('signal_entry_price', ledger)
        np.testing.assert_allclose(ledger[ledger.basis.eq('独立信号')].signal_ret1, candidates.ret1)
        summary = out['attribution_summary']
        for basis, f in summary.groupby('basis'):
            self.assertEqual(f['记录数'].sum(), len(events))
            self.assertEqual(f['有效平仓数'].sum(), 7)
            self.assertAlmostEqual(f['对全部平仓均益贡献(百分点)'].sum(), events.net_return.mean()*100)
            self.assertTrue(f[f.outcome.eq('未完成/无法分类')]['净均益%'].isna().all())
        # 修改后续收益只能改变结果类别，不能改变信号特征或当日分位。
        changed = events.copy(); changed['net_return'] = .05
        rebuilt = attribution_reports(changed, candidates, changed, Config())['attribution_ledger']
        cols = [x for x in ledger if x.startswith(('signal_', 'pool_pct_'))]
        pd.testing.assert_frame_equal(ledger[cols], rebuilt[cols])
        # 只截断未来候选，不改变此前股票日的特征。
        future = candidates.copy(); future['date'] = self.calendar[1]; future['ret1'] = 100
        a = attribution_features(candidates)
        b = attribution_features(pd.concat([candidates, future], ignore_index=True))
        pd.testing.assert_frame_equal(a, b[b.signal_date.eq(self.calendar[0])])
        constant = candidates.copy(); constant['sector_ret1'] = .03
        self.assertTrue(attribution_features(constant).pool_pct_sector_ret1.isna().all())
        self.assertEqual(attribution_reports(pd.DataFrame(), candidates, pd.DataFrame(), Config()), {})

    def test_attribution_consistency_requires_both_counts_and_direction(self):
        row = dict(basis='独立信号', group=GROUP_BASE, contrast='止损退出', feature='ret1', 特征='涨幅',
            止盈有效数=40, 对照有效数=40, 中位数差=1., 止盈池内分位有效数=40, 对照池内分位有效数=40, 池内分位差=2.)
        a = pd.DataFrame([row]); b = pd.DataFrame([{**row, '池内分位差': -2.}])
        out = attribution_consistency(a, b).iloc[0]
        self.assertEqual(out['原值方向检查'], '止盈组两段都较高')
        self.assertEqual(out['池内分位方向检查'], '方向不一致或无差异')
        b['对照有效数'] = 29
        self.assertEqual(attribution_consistency(a, b).iloc[0]['原值方向检查'], '样本不足30_仅列数值')
        b['中位数差'] = np.nan
        self.assertEqual(attribution_consistency(a, b).iloc[0]['原值方向检查'], '数据不足')

    def test_attribution_archive_roundtrip_rejects_missing_events_and_overlap(self):
        with tempfile.TemporaryDirectory() as root:
            store, basic, member, names, cal = demo_data(root)
            cfg = replace(Config(), start=ds(cal[35]), end=ds(cal[50]), scope='全A沪深', min_mv=0, max_mv=10000)
            panel = sector_panel(store, member, cal, cfg)
            c = build_candidates(store, basic, member, names, panel, cal, cfg)
            selected = select_candidates(c, cfg); events = []
            trades = run_trades(store, selected, cal, cfg, event_records=events)
            folder = save_report(root, cfg, c, selected, trades, panel, cal, 'test', '演示测试', pd.DataFrame(events))
            raw = (Path(folder)/'回测审计.zip').read_bytes()
            report = load_attribution_archive(raw)
            self.assertTrue(report['meta']['selection_rebuilt'])
            self.assertFalse(report['meta']['price_execution_replayed'])
            self.assertIn('attribution_ledger', report['tables'])
            review = review_archives([('first.zip', raw), ('second.zip', raw)])
            self.assertFalse(review['meta']['comparable'])
            self.assertIn('重叠', review['meta']['warnings'][0])
            self.assertNotIn('attribution_consistency', review['tables'])
            with zipfile.ZipFile(io.BytesIO(review['archive'])) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('attribution_features.csv', z.namelist())
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                contents = {n: z.read(n) for n in z.namelist()}
            def altered(changes):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
                    for n, v in {**contents, **changes}.items():
                        z.writestr(n, v)
                return buf.getvalue()
            incomplete = pd.DataFrame(events).iloc[1:].to_csv(index=False).encode()
            with self.assertRaisesRegex(DataError, '不齐全'):
                load_attribution_archive(altered({'independent_signal_paths.csv': incomplete}))
            bad = c.copy(); bad.loc[bad.index[0], 'date'] = pd.Timestamp('2030-01-01')
            with self.assertRaisesRegex(DataError, '日期超出'):
                load_attribution_archive(altered({'candidates.csv': bad.to_csv(index=False).encode()}))
            manifest = json.loads(contents['manifest.json']); manifest['config']['buy_fee'] = .001
            other = altered({'manifest.json': json.dumps(manifest).encode()})
            # 另一合法、不重叠区间用于检测参数不匹配。
            other_report = load_attribution_archive(other)
            self.assertNotEqual(other_report['meta']['params'], report['meta']['params'])
            with self.assertRaises(DataError):
                load_attribution_archive(b'not a zip')
            with self.assertRaises(DataError):
                review_archives([])
            store.close()

    def test_fixed_protocol_dates_parameters_and_model_scope(self):
        cfg = Config(efficiency_research=False)
        self.assertEqual([s["experiment"] for s in experiments(cfg)], ["RETURN", "T10_PRE3"])
        self.assertTrue(top10_observation(self.filter_candidates(), cfg).empty)
        a = validation_protocol(cfg); b = validation_protocol(replace(cfg, start="20200101", end="20210101"))
        self.assertEqual(a["protocol_sha256"], b["protocol_sha256"])
        self.assertTrue(a["standard_parameters_match"])
        changed = validation_protocol(replace(cfg, max_gap=.04))
        self.assertFalse(changed["standard_parameters_match"])
        self.assertIn("max_gap", changed["parameter_differences"])
        self.assertNotEqual(a["protocol_sha256"], changed["protocol_sha256"])
        dates_to_check = pd.Series(["2025-09-19", "2025-09-20", "2026-09-18", "2026-09-19", "2026-09-21", "2026-09-22"])
        phases = sample_phase(dates_to_check)
        self.assertTrue(phases.iloc[0].startswith("历史待验证"))
        self.assertTrue(phases.iloc[[1, 2]].eq("已研究区间").all())
        self.assertTrue(phases.iloc[[3, 4]].str.startswith("冻结边界").all())
        self.assertTrue(phases.iloc[5].startswith("冻结后日期"))
        self.assertTrue(sample_phase(dates_to_check, "演示合成数据").eq("合成演示_不是验证").all())

    def test_month_contribution_reconciles_and_unknown_is_not_zero(self):
        f = pd.DataFrame({"group": [TOP10_MODELS[0][1]]*4, "take_profit": [.10]*4,
            "signal_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-02-03", "2025-02-04"]),
            "baseline_slot_mean": [0., 0., 0., 0.], "slot_mean": [.12, -.02, -.01, np.nan],
            "paired_delta": [.12, -.02, -.01, np.nan], "pair_complete": [True, True, True, False],
            "sample_phase": ["历史待验证_须排除既往调参"]*4})
        tables = concentration_tables(f)
        summary = tables["validation_summary"].iloc[0]
        self.assertEqual(summary["共同结果已知日数"], 3)
        self.assertEqual(summary["未知剔除日数"], 1)
        self.assertAlmostEqual(summary["配对改善(百分点)"], 3.)
        months = tables["validation_monthly"]
        self.assertAlmostEqual(months["对全期改善贡献(百分点)"].sum(), 3.)
        loo = tables["validation_leave_month_out"].set_index("excluded_month")
        self.assertAlmostEqual(loo.loc["2025-01", "配对改善(百分点)"], -1.)
        self.assertEqual(loo.loc["2025-01", "未知剔除日数"], 1)
        conc = tables["validation_concentration"].iloc[0]
        self.assertGreater(conc["最大月占净改善%"], 100.)
        self.assertEqual(conc["最大月占正贡献月份%"], 100.)
        self.assertTrue(pd.isna(conc["剔除最大5个改善日后(百分点)"]))
        f.loc[:, "paired_delta"] = np.nan
        f.loc[:, "pair_complete"] = False
        empty = concentration_tables(f)["validation_concentration"].iloc[0]
        self.assertEqual(empty["完整信号日数"], 0)
        self.assertTrue(pd.isna(empty["全期配对改善(百分点)"]))

    def test_frozen_run_preserves_m13_models_and_requires_all_events(self):
        with tempfile.TemporaryDirectory() as root:
            store, basic, member, names, cal = demo_data(root)
            cfg = Config(start=ds(cal[35]), end=ds(cal[60]), efficiency_research=False)
            panel = sector_panel(store, member, cal, cfg)
            c = build_candidates(store, basic, member, names, panel, cal, cfg)
            selected = select_candidates(c, cfg)
            prior_selected = select_candidates(c, replace(cfg, validation_only=False))
            compare = prior_selected[prior_selected.group.isin(VALIDATION_GROUPS)].reset_index(drop=True)
            pd.testing.assert_frame_equal(selected.reset_index(drop=True), compare)
            events = []; trades = run_trades(store, selected, cal, cfg, event_records=events)
            prior = run_trades(store, compare, cal, cfg)
            pd.testing.assert_frame_equal(trades, prior)
            reports = validation_reports(pd.DataFrame(events), selected, c, trades, cfg, "演示测试")
            self.assertEqual(len(reports["validation_summary"]), 2)
            for row in reports["validation_summary"].to_dict("records"):
                m = reports["validation_monthly"]
                self.assertAlmostEqual(row["配对改善(百分点)"], m[m.take_profit.eq(row["take_profit"])]["对全期改善贡献(百分点)"].sum())
            with self.assertRaises(DataError):
                validation_reports(pd.DataFrame(events).iloc[1:], selected, c, trades, cfg, "演示测试")
            folder = save_report(root, cfg, c, selected, trades, panel, cal, "test", "演示测试", pd.DataFrame(events))
            self.assertEqual(len(report_table(folder, "experiment_summary")), 4)
            self.assertEqual(len(report_table(folder, "independent_signal_paths")), 2*len(selected))
            self.assertTrue(report_table(folder, "top10_events").empty)
            manifest = json.loads((Path(folder)/"manifest.json").read_text())
            self.assertEqual(manifest["validation_protocol"]["primary_target"], .10)
            self.assertFalse(manifest["verified_profitability"])
            # 固定模式筛空时依然可以交付报告，不虚构零收益或空仓胜率。
            empty = c.copy(); empty["base_reason"] = empty["exclude_reason"] = "测试排除；"
            no_picks = select_candidates(empty, cfg)
            empty_folder = save_report(root, cfg, empty, no_picks, pd.DataFrame(), panel, cal, "test", "演示测试", pd.DataFrame())
            self.assertTrue(report_table(empty_folder, "validation_summary").empty)
            store.close()

    def test_top10_reselection_boundary_missing_features_and_ties(self):
        cfg = replace(self.cfg, research=True, filter_research=False)
        c = self.filter_candidates(12)
        c["pre3"] = [.3, .2, .1, -.2, -.1, 0., .15, .16, .17, .18, -5., -6.]
        c["rs_accel"] = [0., 0., 0., .10, .09, .08, .01, .01, .01, .01, 5., 6.]
        selected = select_candidates(c, cfg)
        for name in [x[0] for x in TOP10_MODELS]:
            f = selected[selected.experiment.eq(name)]
            self.assertEqual(f.return_rank.tolist(), [4, 5, 6])
            self.assertEqual(f["rank"].tolist(), [1, 2, 3])
        self.assertEqual(len(top10_observation(c, cfg)), 10)
        self.assertEqual(len(experiments(cfg)), 5)
        self.assertEqual(len(experiments(replace(cfg, filter_research=True))), 9)
        self.assertEqual(len(experiments(replace(cfg, filter_research=True, legacy_research=True))), 28)
        c.loc[:9, "pre3"] = np.nan
        c.loc[:9, "rs_accel"] = 0.
        selected = select_candidates(c, cfg)
        self.assertTrue(selected[selected.experiment.eq("T10_PRE3")].empty)
        self.assertEqual(selected[selected.experiment.eq("T10_ACCEL")].return_rank.tolist(), [1, 2, 3])
        self.assertEqual(selected[selected.experiment.eq("RETURN")].return_rank.tolist(), [1, 2, 3])
        self.assertEqual(len(top10_observation(c.iloc[:2], cfg)), 2)

    def test_pre3_excludes_signal_day_and_does_not_use_future(self):
        f = self.bars()
        f["close"] = [100, 105, 110, 115, 130, 200, 250, 280, 300, 350]
        out = features(f, self.calendar)
        self.assertAlmostEqual(out.pre3.iloc[4], .15)
        changed = f.copy(); changed.loc[self.calendar[4]:, "close"] *= 10
        self.assertAlmostEqual(features(changed, self.calendar).pre3.iloc[4], .15)

    def test_top10_paired_days_keep_cash_and_exclude_unknown(self):
        rows = []
        for tp in [.05, .10]:
            for day in self.calendar[:2]:
                for group in [GROUP_BASE]+[x[1] for x in TOP10_MODELS]:
                    status = "未平仓" if day == self.calendar[1] and group == GROUP_BASE else "已平仓"
                    ret = -.05 if group == GROUP_BASE else .10
                    rows.append(dict(group=group, signal_date=day, take_profit=tp, status=status,
                                     data_issue=False, net_return=ret if status == "已平仓" else np.nan))
                    rows.append(dict(group=group, signal_date=day, take_profit=tp, status="未买入",
                                     data_issue=False, net_return=np.nan))
        pool = pd.DataFrame({"date": self.calendar[:2]})
        daily, summary = paired_top10_days(pd.DataFrame(rows), pool, self.cfg)
        full = summary[summary.period.eq("全区间")]
        self.assertTrue(full["共同结果已知日数"].eq(1).all())
        self.assertTrue(full["未知剔除日数"].eq(1).all())
        np.testing.assert_allclose(full["配对改善(百分点)"], 5.)
        self.assertTrue(daily.cash_slots.eq(2).all())
        self.assertTrue(daily[daily.signal_date.eq(self.calendar[1])].paired_delta.isna().all())

    def test_top10_full_replay_and_baseline_preservation(self):
        with tempfile.TemporaryDirectory() as root:
            store, basic, member, names, cal = demo_data(root)
            cfg = replace(self.cfg, start=ds(cal[35]), end=ds(cal[60]), research=True, filter_research=False)
            panel = sector_panel(store, member, cal, cfg)
            c = build_candidates(store, basic, member, names, panel, cal, cfg)
            selected = select_candidates(c, cfg)
            short = build_candidates(store, basic, member, names, sector_panel(store, member, cal[:61], cfg), cal[:61], cfg)
            pd.testing.assert_frame_equal(c[["date", "ts_code", "pre3", "rs_accel"]], short[["date", "ts_code", "pre3", "rs_accel"]])
            pd.testing.assert_frame_equal(selected, select_candidates(short, cfg))
            events = []
            obs = top10_observation(c, cfg)
            tr = run_trades(store, selected, cal, cfg, event_records=events, observations=obs)
            self.assertFalse(tr.experiment.eq("OBS_TOP10").any())
            self.assertEqual(len(events), 2*(len(selected)+len(obs)))
            old_selected = select_candidates(c, replace(cfg, research=False))
            old_trades = run_trades(store, old_selected, cal, cfg)
            keys = ["group", "take_profit", "signal_date", "ts_code"]
            now = tr[tr.experiment.isin(["B0", "RETURN", "HOT"])].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(now, old_trades.sort_values(keys).reset_index(drop=True))
            folder = save_report(root, cfg, c, selected, tr, panel, cal, "test", "演示测试", pd.DataFrame(events))
            ledger = report_table(folder, "top10_events")
            self.assertEqual(len(ledger), 2*len(obs))
            self.assertFalse(ledger.duplicated(["signal_date", "ts_code", "take_profit"]).any())
            paths = report_table(folder, "top10_rank_paths")
            self.assertEqual(paths["独立信号数"].sum(), len(obs))
            actual = report_table(folder, "experiment_summary")
            self.assertEqual(len(actual), 10)
            changes = report_table(folder, "top10_selection_changes")
            for label in [x[1] for x in TOP10_MODELS]:
                q = changes[changes.comparison.eq(label)]
                for tp in [.05, .10]:
                    self.assertEqual(len(q[q.take_profit.eq(tp) & q.cohort.isin(["共同选中", "新方案换入"])]),
                                     len(selected[selected.group.eq(label)]))
            # 截止仅余1—2日时，未平仓不会成为固定名额中的零收益。
            partial = []
            short_tr = run_trades(store, selected, cal[:62], cfg, event_records=partial, observations=obs)
            tables = top10_reports(pd.DataFrame(partial), c, cfg)
            self.assertGreater(tables["top10_paired_summary"]["未知剔除日数"].sum(), 0)
            with self.assertRaises(DataError):
                top10_event_ledger(pd.DataFrame(events)[lambda x: ~((x.experiment == "OBS_TOP10") & (x.take_profit == .10))], top10_candidates(c, cfg))
            store.close()

    def test_t_plus_one_ignores_buy_day_target(self):
        f = self.bars()
        f.loc[self.calendar[1], "high"] = 109
        out = simulate_trade(f, self.calendar, self.calendar[0], .05, self.cfg)
        self.assertEqual(out["exit_date"], self.calendar[5])
        self.assertEqual(out["reason"], "到期退出")
        self.assertTrue(out["buy_day_hit5"])

    def test_target_and_cost(self):
        f = self.bars()
        f.loc[self.calendar[2], "high"] = 106
        cfg = replace(self.cfg, buy_fee=.0003, sell_fee=.0003)
        out = simulate_trade(f, self.calendar, self.calendar[0], .05, cfg)
        self.assertAlmostEqual(out["net_return"], 1.05*(1-.0003-.0005)/(1+.0003)-1)

    def test_exact_price_touch_triggers_target(self):
        for target, price in [(.05, 105.0), (.10, 110.0)]:
            f = self.bars()
            f.loc[self.calendar[2], "high"] = price
            out = simulate_trade(f, self.calendar, self.calendar[0], target, self.cfg)
            self.assertEqual(out["reason"], "止盈")
            self.assertEqual(out["exit_date"], self.calendar[2])

    def test_board_gap_changes_only_growth_entry(self):
        f = self.bars()
        f.loc[self.calendar[1], ["open", "high", "low", "close"]] = [105, 107, 103, 105]
        for code in ["600001.SH", "002001.SZ", "300001.SZ", "301001.SZ", "688001.SH"]:
            uniform = replace(self.cfg, max_gap=buy_gap(code, "统一上限", self.cfg))
            split = replace(self.cfg, max_gap=buy_gap(code, "分板块上限", self.cfg))
            a = simulate_trade(f, self.calendar, self.calendar[0], .05, uniform)
            b = simulate_trade(f, self.calendar, self.calendar[0], .05, split)
            self.assertEqual(a["status"], "未买入")
            self.assertEqual(b["status"] == "已平仓", board_name(code) != "主板")
        # 上限包含滑点，开盘恰好6%不代表含滑点后还能成交。
        f.loc[self.calendar[1], "open"] = 106
        out = simulate_trade(f, self.calendar, self.calendar[0], .05, replace(self.cfg, max_gap=.06, slippage=.001))
        self.assertEqual(out["status"], "未买入")

    def test_continuous_rs_is_not_cumulative_rs(self):
        sector = pd.Series("A", index=self.calendar)
        e = pd.Series([.01, .01, .01, -.001, .03, .01, .01, .01, .01, .01], index=self.calendar)
        self.assertGreater(e.iloc[:5].sum(), 0)
        self.assertFalse(consecutive_rs(e, sector, 5).iloc[4])
        self.assertTrue(consecutive_rs(e, sector, 1).iloc[4])
        self.assertTrue(consecutive_rs(e, sector, 5).iloc[8])
        for n in [1, 2, 3, 5]:
            a = consecutive_rs(e, sector, n)
            changed = e.copy(); changed.iloc[7:] = -1
            pd.testing.assert_series_equal(a.iloc[:7], consecutive_rs(changed, sector, n).iloc[:7])

    def test_gap_incremental_report_includes_new_losers(self):
        rows = []
        for code in ["600001.SH", "300001.SZ", "688001.SH"]:
            f = self.bars()
            if board_name(code) != "主板":
                f.loc[self.calendar[1], ["open", "high", "low", "close"]] = [105, 107, 103, 105]
            for policy in ["统一上限", "分板块上限"]:
                result = simulate_trade(f, self.calendar, self.calendar[0], .05,
                                        replace(self.cfg, max_gap=buy_gap(code, policy, self.cfg)))
                rows.append(dict(**result, experiment="B0", signal_date=self.calendar[0], ts_code=code,
                                 group=policy, gap_policy=policy, board=board_name(code), take_profit=.05))
        ledger, report = gap_comparison(pd.DataFrame(rows))
        extra = ledger[ledger.increment_type.eq("放宽新增_原因高开")]
        self.assertEqual(len(extra), 2)
        self.assertTrue(extra.net_return.lt(0).all())
        self.assertFalse(extra.board.eq("主板").any())
        self.assertEqual(len(ledger[ledger.increment_type.eq("共同成交")]), 1)
        self.assertEqual(report["有效平仓数"].sum(), 3)

    def test_rs_missing_tie_or_sector_switch_fails(self):
        sector = pd.Series("A", index=self.calendar)
        e = pd.Series(.01, index=self.calendar)
        for value in [np.nan, 0.0, -.001]:
            f = e.copy(); f.iloc[3] = value
            self.assertFalse(consecutive_rs(f, sector, 5).iloc[5])
        sector.iloc[4:] = "B"
        self.assertFalse(consecutive_rs(e, sector, 3).iloc[5])
        self.assertTrue(consecutive_rs(e, sector, 3).iloc[6])

    def test_weight_experiment_can_change_ranking_without_rescoring(self):
        rows = []
        for code, scores in [("600001.SH", [90, 40, 40, 40]), ("300001.SZ", [20, 80, 60, 60])]:
            row = dict(date=self.calendar[0], ts_code=code, board=board_name(code), close=20,
                       base_reason="", overheat=False, heat_penalty=0, ret1=.02,
                       score=sum(scores)/4, **dict(zip(SCORE_COLS, scores)))
            row.update({f"rs_all_{n}": True for n in [1, 2, 3, 5]})
            rows.append(row)
        c = pd.DataFrame(rows)
        selected = select_candidates(c, replace(self.cfg, top_n=1, legacy_research=True))
        self.assertEqual(selected[selected.group.eq(GROUP_MAIN)].iloc[0].ts_code, "300001.SZ")
        strong = selected[selected.group.eq("相对强度40")].iloc[0]
        self.assertEqual(strong.ts_code, "600001.SH")
        self.assertEqual(strong.strength_score, 90)
        self.assertEqual(strong.score, 60)
        self.assertEqual(len(experiments(self.cfg)), 7)
        self.assertEqual(len(experiments(replace(self.cfg, legacy_research=True))), 26)

    def test_path_summary_excludes_incomplete_and_buy_day_only_hit(self):
        f = self.bars()
        f.loc[self.calendar[1], "high"] = 106
        result = simulate_trade(f, self.calendar, self.calendar[0], .05, self.cfg)
        partial = simulate_trade(f.iloc[:3], self.calendar[:3], self.calendar[0], .05, self.cfg)
        report = path_summary(pd.DataFrame([{**result, "group": "test", "take_profit": .05},
                                           {**partial, "group": "test", "take_profit": .05},
                                           {**result, "group": "test", "take_profit": .10}]))
        self.assertEqual(report.iloc[0]["完整5日观察数"], 1)
        self.assertEqual(report.iloc[0]["观察不完整或数据问题数"], 1)
        self.assertEqual(report.iloc[0]["可卖日期曾达5%比例"], 0)
        self.assertEqual(report.iloc[0]["仅买入当天达5%比例"], 100)

    def filter_candidates(self, count=8):
        rows = []
        for i in range(count):
            row = dict(date=self.calendar[0], ts_code=f"{600001+i}.SH", name=f"测试{i}", close=20,
                       base_reason="", exclude_reason="", overheat=False, ret1=.08-i*.005,
                       ret5=.1, rs5=.05, runup5=.15, sector="A", sector_name="测试板块",
                       strength_score=i*10., efficiency_score=50., volume_score=50., position_score=50.,
                       heat_penalty=0.)
            row["score"] = np.mean([row[c] for c in SCORE_COLS]); rows.append(row)
        return pd.DataFrame(rows)

    def test_filter_after_ranking_never_backfills(self):
        candidates = self.filter_candidates()
        selected = select_candidates(candidates, self.cfg)
        filtered = selected[selected.experiment.eq("F_RS")]
        self.assertEqual(filtered.ts_code.tolist(), ["600003.SH"])
        self.assertEqual(filtered["rank"].tolist(), [3])
        decisions = screening_decisions(candidates, self.cfg)
        d = decisions[decisions.experiment.eq("F_RS")]
        self.assertEqual(d["rank"].tolist(), [1, 2, 3])
        self.assertEqual(d.filter_pass.tolist(), [False, False, True])

    def test_filter_small_pool_and_ties_are_not_arbitrary_vetoes(self):
        for candidates in [self.filter_candidates(3), self.filter_candidates().assign(strength_score=10.)]:
            selected = select_candidates(candidates, self.cfg)
            self.assertEqual(len(selected[selected.experiment.eq("F_RS")]), 3)
        f = self.filter_candidates()
        before = screening_decisions(f, self.cfg)
        future = f.copy(); future["date"] = self.calendar[1]; future["strength_score"] = 10000.
        after = screening_decisions(pd.concat([f, future], ignore_index=True), self.cfg)
        pd.testing.assert_frame_equal(before, after[after.date.eq(self.calendar[0])].reset_index(drop=True))

    def test_filter_equal_opportunity_detects_false_average_improvement(self):
        rows = [dict(group="filter", take_profit=.05, status="已平仓", data_issue=False, filter_pool_n=8,
                     filter_pass=keep, net_return=r) for r, keep in [(.10, True), (.02, False), (-.05, True)]]
        rows.append(dict(group="filter", take_profit=.05, status="未平仓", data_issue=False, filter_pool_n=8,
                         filter_pass=False, net_return=np.nan))
        result = filter_impact_summary(pd.DataFrame(rows)).iloc[0]
        self.assertGreater(result["保留单笔均益%"], result["基准机会均益%"])
        self.assertLess(result["过滤后等机会均益%"], result["基准机会均益%"])
        self.assertAlmostEqual(result["等机会改善(百分点)"], -.02/3*100)
        self.assertEqual(result["有效平仓机会数"], 3)
        self.assertEqual(result["待买入或未平仓数"], 1)
        self.assertEqual(result["错过盈利笔数"], 1)
        self.assertAlmostEqual(result["避免亏损贡献(百分点)"]-result["错过盈利贡献(百分点)"], result["等机会改善(百分点)"])

    def test_dual_touch_is_stop_first(self):
        f = self.bars()
        f.loc[self.calendar[2], ["high", "low", "down_limit"]] = [107, 88, 80]
        out = simulate_trade(f, self.calendar, self.calendar[0], .05, self.cfg)
        self.assertTrue(out["ambiguous"])
        self.assertEqual(out["reason"], "止损")
        self.assertAlmostEqual(out["gross_return"], -.10)

    def test_gap_stop_fills_below_threshold(self):
        f = self.bars()
        f.loc[self.calendar[2], ["open", "high", "low", "close", "down_limit"]] = [85, 88, 82, 86, 80]
        out = simulate_trade(f, self.calendar, self.calendar[0], .05, self.cfg)
        self.assertEqual(out["reason"], "跳空止损")
        self.assertAlmostEqual(out["gross_return"], -.15)

    def test_limit_down_defers_exit(self):
        f = self.bars()
        f.loc[self.calendar[2], ["open", "high", "low", "close"]] = 90
        f.loc[self.calendar[3], ["open", "high", "low", "close", "down_limit"]] = [85, 87, 83, 84, 81]
        out = simulate_trade(f, self.calendar, self.calendar[0], .05, self.cfg)
        self.assertEqual(out["exit_date"], self.calendar[3])
        self.assertTrue(out["deferred"])
        self.assertLess(out["gross_return"], -.10)

    def test_suspension_does_not_shift_entry(self):
        f = self.bars().drop(self.calendar[1])
        out = simulate_trade(f, self.calendar, self.calendar[0], .05, self.cfg)
        self.assertEqual(out["status"], "未买入")

    def test_market_calendar_holding_and_delayed_expiry(self):
        f = self.bars().drop(self.calendar[[3, 4, 5]])
        out = simulate_trade(f, self.calendar, self.calendar[0], .05, self.cfg)
        self.assertEqual(out["exit_date"], self.calendar[6])
        self.assertEqual(out["hold_days"], 6)

    def test_unfinished_trade_is_not_zero(self):
        cal = self.calendar[:3]
        out = simulate_trade(self.bars().loc[cal], cal, cal[0], .05, self.cfg)
        self.assertEqual(out["status"], "未平仓")
        self.assertTrue(np.isnan(out["net_return"]))

    def test_high_open_rejected(self):
        f = self.bars()
        f.loc[self.calendar[1], "open"] = 104
        self.assertEqual(simulate_trade(f, self.calendar, self.calendar[0], .05, self.cfg)["status"], "未买入")

    def test_upper_limit_rejected_even_if_later_unlocks(self):
        f = self.bars()
        f.loc[self.calendar[1], "open"] = 110
        out = simulate_trade(f, self.calendar, self.calendar[0], .05, replace(self.cfg, max_gap=.20))
        self.assertEqual(out["status"], "未买入")

    def test_split_is_not_loss(self):
        f = self.bars()
        f.loc[self.calendar[2]:, ["open", "high", "low", "close", "pre_close", "up_limit", "down_limit"]] /= 2
        f.loc[self.calendar[2]:, "adj_factor"] = 2
        out = simulate_trade(f, self.calendar, self.calendar[0], .05, self.cfg)
        self.assertAlmostEqual(out["gross_return"], 0)
        self.assertEqual(out["reason"], "到期退出")

    def test_runup_respects_time_order(self):
        h = pd.Series([160, 103, 103, 103, 103, 103], dtype=float)
        l = pd.Series([150, 100, 100, 100, 100, 100], dtype=float)
        self.assertLess(max_runup(h, l).iloc[-1], .05)
        h = pd.Series([101, 110, 160, 150, 145, 140], dtype=float)
        l = pd.Series([100, 105, 155, 145, 140, 135], dtype=float)
        self.assertGreater(max_runup(h, l).iloc[-1], .50)

    def test_membership_changes_on_exit_date(self):
        member = pd.DataFrame([dict(sector="A", in_date=self.calendar[0], out_date=self.calendar[3]),
                               dict(sector="B", in_date=self.calendar[3], out_date=pd.NaT)])
        result = assign_sector(self.calendar, member)
        self.assertEqual(result.iloc[2], "A")
        self.assertEqual(result.iloc[3], "B")

    def test_future_prices_do_not_change_features(self):
        cal = pd.bdate_range("2024-01-01", periods=50)
        f = self.bars().iloc[:1].reindex(cal).ffill().bfill()
        f["close"] = np.linspace(100, 130, 50)
        f["high"], f["low"] = f.close+2, f.close-2
        before = features(f, cal).iloc[:35]
        f.loc[cal[35]:, ["open", "high", "low", "close", "amount"]] *= 4
        after = features(f, cal).iloc[:35]
        pd.testing.assert_frame_equal(before, after)

    def test_data_adapter_merges_fields_and_reuses_day_cache(self):
        with tempfile.TemporaryDirectory() as root:
            client = object.__new__(DataClient)
            client.root, client.progress = Path(root), lambda _: None
            row = self.bars().iloc[[0]].reset_index(drop=True)
            row["ts_code"], row["trade_date"] = "600001.SH", ds(self.calendar[0])
            calls = []
            def query(endpoint, **kwargs):
                calls.append(endpoint)
                return row[FIELDS[endpoint].split(",")].copy()
            client.query = query
            damaged = Path(root)/"pending"/"daily"/(ds(self.calendar[0])+".csv.gz")
            damaged.parent.mkdir(parents=True)
            damaged.write_bytes(b"incomplete download")
            store = client.download(self.calendar[:1])
            self.assertEqual(len(calls), 4)
            self.assertEqual(store.stock("600001.SH", self.calendar[0], self.calendar[0]).iloc[0].circ_mv, 1000000)
            store.close()
            cached = client.download(self.calendar[:1])
            self.assertEqual(len(calls), 4)
            cached.close()

    def test_incomplete_data_day_is_not_committed(self):
        with tempfile.TemporaryDirectory() as root:
            client = object.__new__(DataClient)
            client.root, client.progress = Path(root), lambda _: None
            row = self.bars().iloc[[0]].reset_index(drop=True)
            row["ts_code"], row["trade_date"] = "600001.SH", ds(self.calendar[0])
            row["adj_factor"] = np.nan
            client.query = lambda endpoint, **kwargs: row[FIELDS[endpoint].split(",")].copy()
            with self.assertRaises(DataError):
                client.download(self.calendar[:1])
            store = MarketStore(Path(root)/"market.sqlite")
            self.assertEqual(store.completed(), set())
            store.close()

    def test_demo_pipeline_and_selection_has_no_future_leakage(self):
        with tempfile.TemporaryDirectory() as root:
            store, basic, member, names, cal = demo_data(root)
            cfg = replace(self.cfg, start=ds(cal[35]), end=ds(cal[60]))
            panel = sector_panel(store, member, cal, cfg)
            c = build_candidates(store, basic, member, names, panel, cal, cfg)
            selected = select_candidates(c, cfg)
            self.assertGreater(len(selected[selected.group.eq(GROUP_MAIN)]), 0)
            self.assertFalse(selected[selected.group.eq(GROUP_MAIN)].overheat.any())
            self.assertLessEqual(selected.groupby(["date", "group"]).size().max(), 3)
            # 真正截断未来行情重算行业、评分和选择，而不只检查单股指标。
            short_cal = cal[:61]
            short_panel = sector_panel(store, member, short_cal, cfg)
            short = build_candidates(store, basic, member, names, short_panel, short_cal, cfg)
            cols = ["date", "ts_code", "score", "exclude_reason", "rs1", "rs2"] + [f"rs_all_{n}" for n in [1, 2, 3, 5]]
            pd.testing.assert_frame_equal(c[cols].reset_index(drop=True), short[cols].reset_index(drop=True))
            chosen_cols = ["date", "ts_code", "group", "rank", "filter_pass", "filter_pct"]
            pd.testing.assert_frame_equal(selected[chosen_cols], select_candidates(short, cfg)[chosen_cols])
            events = []
            tr = run_trades(store, selected, cal, cfg, event_records=events)
            self.assertFalse(tr.empty)
            for _, f in tr[tr.status.eq("已平仓")].groupby(["group", "take_profit", "ts_code"]):
                f = f.sort_values("entry_date")
                self.assertTrue((f.entry_date.iloc[1:].values > f.exit_date.iloc[:-1].values).all())
            ledger, increment = gap_comparison(tr)
            self.assertTrue(ledger.empty)  # 默认不启用6%旧对照。
            self.assertEqual(len(events), 2*len(selected))
            folder = save_report(root, cfg, c, selected, tr, panel, cal, "test", "演示测试", pd.DataFrame(events))
            grid = report_table(folder, "experiment_summary")
            self.assertEqual(len(grid), 14)
            self.assertTrue(grid.loc[grid.experiment.eq("B0"), "相对等权均益差(百分点)"].dropna().eq(0).all())
            audit = report_table(folder, "filter_impact")
            self.assertEqual(len(audit), 8)
            self.assertTrue((audit["保留平仓数"]+audit["拒绝平仓数"]).eq(audit["有效平仓机会数"]).all())
            np.testing.assert_allclose(audit["过滤后等机会均益%"]-audit["基准机会均益%"], audit["等机会改善(百分点)"], atol=1e-12)
            with zipfile.ZipFile(Path(folder)/"回测审计.zip") as z:
                self.assertIsNone(z.testzip())
                self.assertIn("trades.csv", z.namelist())
            # 严格条件筛空时仍应生成有效报告，不能报页面错误或凑足三只。
            empty = c.copy()
            empty["base_reason"] = empty["exclude_reason"] = "测试排除；"
            no_picks = select_candidates(empty, cfg)
            self.assertTrue(no_picks.empty)
            folder = save_report(root, cfg, empty, no_picks, run_trades(store, no_picks, cal, cfg), panel, cal, "test", "演示测试")
            self.assertTrue(report_table(folder, "latest_selected").empty)
            empty_grid = report_table(folder, "experiment_summary")
            self.assertTrue(empty_grid["有效平仓数"].eq(0).all())
            self.assertTrue(empty_grid["净均益%"].isna().all())
            store.close()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(StrategyTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    elif "--demo" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--demo", action="store_true")
        parser.add_argument("--output", default="momentum_demo")
        args = parser.parse_args()
        print(run_demo(args.output))
    else:
        main()
