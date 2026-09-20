# -*- coding: utf-8 -*-
"""板块领涨惯性 M1.1 — 分板块买价 / 权重 / 连续相对强度固定对照

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

VERSION = "M1.1"
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


@dataclass(frozen=True)
class Config:
    start: str = "20250101"
    end: str = "20260918"
    scope: str = "科技行业"
    min_price: float = 10.0
    min_mv: float = 50.0
    max_mv: float = 1000.0
    top_n: int = 3
    heat_limit: float = 0.50
    max_gap: float = 0.03
    growth_gap: float = 0.06
    research: bool = True
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


def experiments(cfg):
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


def buy_gap(code, policy, cfg):
    return cfg.growth_gap if policy == "分板块上限" and board_name(code) != "主板" else cfg.max_gap


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


def select_candidates(candidates, cfg):
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
        f = f.sort_values(["date", spec["sort"], "ts_code"], ascending=[True, False, True])
        f["rank"] = f.groupby("date").cumcount()+1
        f = f[f["rank"].le(cfg.top_n)].copy()
        for key in ["group", "experiment", "family", "gap_policy", "rs_days"]:
            f[key] = spec[key]
        f["gap_limit"] = f.ts_code.map(lambda code: buy_gap(code, spec["gap_policy"], cfg))
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


def run_trades(store, selected, calendar, cfg, progress=lambda s: None, event_records=None):
    records = []
    if selected.empty:
        return pd.DataFrame()
    for n, (code, f) in enumerate(selected.groupby("ts_code", sort=False), 1):
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
                    cache_key = (row.date, tp, row.gap_limit)
                    # 相同股票、信号日、买价上限和止盈只撮合一次，各实验独立管理持仓。
                    if cache_key not in cached_paths:
                        cached_paths[cache_key] = simulate_trade(stock, calendar, row.date, tp,
                                                               replace(cfg, max_gap=row.gap_limit))
                    path = cached_paths[cache_key]
                    if event_records is not None and tp == .05:
                        # 用全部候选信号的假设入场检查选股惯性，不受早先持仓阻塞影响。
                        event = {**base, **path}
                        event["event_mode"] = "独立信号观察_允许重叠_不是组合收益"
                        event_records.append(event)
                    if pd.notna(planned) and planned <= busy_until:
                        result = dict(status="重复持仓跳过", entry_date=planned, exit_date=pd.NaT,
                                      net_return=np.nan, reason="同组同止盈版本仍持有；退出当日不重复开仓")
                    else:
                        result = path.copy()
                        if result["status"] in ["已平仓", "未平仓"]:
                            busy_until = result["exit_date"] if result["status"] == "已平仓" else pd.Timestamp.max
                    records.append({**base, **result})
        if n % 50 == 0:
            progress(f"回放成交路径 {n}/{selected.ts_code.nunique()} 只")
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
        row = {k: spec[k] for k in ["group", "experiment", "family", "gap_policy", "rs_days"]}
        row.update(dict(zip(["相对强度权重", "上涨效率权重", "量价配合权重", "收盘位置权重"], spec["weights"])))
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


RULES = """板块领涨惯性 M1.1：研究规则与边界

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
默认每日候选仍展示原版等权统一上限，不自动采用历史最优实验。22组是预先固定的假设对照，不进行参数寻优。
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
    folder = Path(tempfile.mkdtemp(prefix="M11_", dir=root))
    tables = {"candidates": candidates, "selected": selected, "trades": trades,
              "sector_daily": panel[panel.date.between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))].copy()}
    tables["summary"] = summarize(trades, ["group", "take_profit"])
    tables["experiment_design"] = experiment_design(cfg)
    tables["experiment_summary"] = experiment_summary(trades, selected, calendar, cfg)
    tables["gap_incremental_trades"], tables["gap_incremental_summary"] = gap_comparison(trades)
    if events is not None:
        tables["independent_signal_paths"] = events
        tables["path_summary"] = path_summary(events)
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
                    research_design="固定22组：两种买价×等权/四种40-20-20-20/四种连续RS/涨幅/过热",
                    experiment_count=len(experiments(cfg)),
                    tuning_performed=False, rs_definition="最近N个市场交易日每天收益严格高于同一板块；含信号日；不是累计跑赢",
                    independent_paths="允许重叠的独立信号5日观察；路径可能发生于止盈/止损之后，不是兑现利润")
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
        trades = run_trades(store, selected, calendar, cfg, progress, event_records=events)
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


def run_demo(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    store, basic, member, names, calendar = demo_data(root)
    cfg = Config(start=ds(calendar[35]), end=ds(calendar[-12]), scope="演示合成数据")
    try:
        panel = sector_panel(store, member, calendar, cfg)
        candidates = build_candidates(store, basic, member, names, panel, calendar, cfg)
        selected = select_candidates(candidates, cfg)
        events = []
        trades = run_trades(store, selected, calendar, cfg, event_records=events)
        return save_report(root, cfg, candidates, selected, trades, panel, calendar,
                           store.fingerprint(calendar[0], calendar[-1]), "演示合成数据_禁止解读为真实收益", pd.DataFrame(events))
    finally:
        store.close()


DISPLAY = {
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
DISPLAY.update({f"rs_all_{n}": f"连续{n}日跑赢" for n in [1, 2, 3, 5]})


def show_table(st, frame):
    # 新版Streamlit改用width；保留旧项目1.32兼容性。
    version = tuple(int(x) for x in st.__version__.split(".")[:2])
    width = {"width": "stretch"} if version >= (1, 49) else {"use_container_width": True}
    st.dataframe(display_frame(frame), hide_index=True, **width)


def show_research(st, folder):
    grid = report_table(folder, "experiment_summary")
    if grid.empty:
        st.info("此报告没有M1.1对照结果，请重新运行。")
        return
    st.subheader("固定对照：买价上限、四维权重、连续跑赢天数")
    st.caption("这是探索性对照，不自动推荐历史收益最高的一组。净均益是独立交易均值，非账户收益；检查各年、样本量和信号覆盖。")
    family = st.selectbox("查看实验类型", ["全部", "等权基准", "单维加权", "连续相对强度", "涨幅对照", "过热观察"], key="research_family")
    policy = st.radio("买入上限规则", ["两种都看", "统一上限", "分板块上限"], horizontal=True, key="research_policy")
    target = st.radio("止盈对照", ["5%", "10%"], horizontal=True, key="research_target")
    f = grid[grid.take_profit.eq(.05 if target == "5%" else .10)]
    if family != "全部":
        f = f[f.family.eq(family)]
    if policy != "两种都看":
        f = f[f.gap_policy.eq(policy)]
    cols = ["group", "gap_policy", "选股日数", "选股覆盖%", "合格候选股日数", "有效平仓数", "未平仓数",
            "胜率%", "净均益%", "相对等权均益差(百分点)", "止盈占比%", "最差单笔%", "剔除最大5笔后净均益%"]
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
    with st.expander("查看固定权重和买入上限"):
        show_table(st, report_table(folder, "experiment_design"))


def display_frame(frame):
    f = frame.copy()
    for col in PERCENT_COLUMNS.intersection(f):
        f[col] = pd.to_numeric(f[col], errors="coerce")*100
    for n in range(1, 6):
        col = f"close_return_d{n}"
        if col in f:
            f[col] = pd.to_numeric(f[col], errors="coerce")*100
    f = f.rename(columns={**DISPLAY, **{f"close_return_d{n}": f"第{n}日收盘涨幅%(非策略收益)" for n in range(1, 6)}})
    return f.round(3)


def report_table(folder, name):
    p = Path(folder)/(name+".csv")
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, dtype={"ts_code": str, "sector": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def show_report(st, folder):
    folder = Path(folder)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if "演示" in manifest["mode"]:
        st.warning("以下全部是合成数据演示，仅用于检查程序，不能用于选股或判断盈利。")
    st.caption(f"版本 {manifest['version']} · 信号截止 {manifest['latest_signal']} · 行情截止 {manifest['data_end']}")
    st.info("研究版：评分不是上涨概率。回测是等额单股审计，未施加账户总资金上限，不能当作账户收益。")
    tabs = st.tabs(["对照验证", "最新候选", "回测结果", "排除与成交审计", "本次规则"])
    with tabs[0]:
        show_research(st, folder)
    with tabs[1]:
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
            st.session_state["daily_experiment"] = GROUP_MAIN
        chosen = st.selectbox("展示候选规则（实验尚未验证，默认保留原版）", groups, key="daily_experiment")
        if not picks.empty:
            picks = picks[picks.group.eq(chosen)]
        st.caption("已持仓不重复加仓；高开超过最高买价或开盘涨跌停则跳过，不补位。除权日请按除权参考价调整最高买价。")
        cols = ["rank", "ts_code", "name", "board", "sector_name", "score", "gap_limit", "close", "max_buy_reference", "mv_yi",
                "ret1", "ret5", "rs5", "runup5", "strength_score", "efficiency_score", "volume_score", "position_score", "heat_penalty"]
        if picks.empty:
            st.info("这一天没有合格候选。下方可查看排除原因；不会改选第二板块凑数。")
        else:
            show_table(st, picks.reindex(columns=cols))
        pool = report_table(folder, "latest_candidates")
        with st.expander("查看第一板块所有候选与排除原因"):
            if not pool.empty:
                cols = ["ts_code", "name", "score", "ret5", "rs5", "runup5", "mv_yi", "exclude_reason"]
                show_table(st, pool.sort_values("score", ascending=False)[cols])
    with tabs[2]:
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
    with tabs[3]:
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
                st.session_state["ledger_groups"] = [GROUP_MAIN] if GROUP_MAIN in options else []
            groups = st.multiselect("显示交易组", options, key="ledger_groups")
            sub = ledger[ledger.group.isin(groups)]
            st.caption("页面最多显示最近500条；完整记录在审计下载中。最高涨幅只是路径信息，不等于实际获利。")
            cols = ["signal_date", "group", "take_profit", "rank", "ts_code", "name", "status", "entry_date",
                    "entry_price", "exit_date", "exit_price", "net_return", "hold_days", "reason", "ambiguous",
                    "deferred", "data_issue", "mark_date", "mark_return", "horizon_complete", "buy_day_hit5", "sellable_mfe5", "mae5"]
            show_table(st, sub.sort_values("signal_date", ascending=False).reindex(columns=cols).head(500))
    with tabs[4]:
        st.json(manifest["config"])
        st.text(RULES)
        st.caption(f"代码指纹：{manifest['source_sha256'][:16]} · 数据指纹：{manifest['data_hash'][:16]}")
    with (folder/"回测审计.zip").open("rb") as f:
        st.download_button("下载完整审计结果", f, file_name=f"momentum_{manifest['version']}_{manifest['latest_signal']}.zip", mime="application/zip")


def main():
    import streamlit as st
    st.set_page_config(page_title="板块领涨惯性", page_icon="📈", layout="wide")
    st.title("板块领涨惯性")
    st.caption("M1.1 · 双创买价3%/6%对照 · 四维权重对照 · 连续1/2/3/5日跑赢板块")
    with st.sidebar:
        st.header("数据与股票池")
        default_token = os.environ.get("TUSHARE_TOKEN", "")
        if not default_token:
            try:
                default_token = str(st.secrets.get("TUSHARE_TOKEN", st.secrets.get("tushare_token", "")))
            except Exception:
                default_token = ""
        token = st.text_input("Tushare Token", value=default_token, type="password", help="仅用于请求官方数据；不写入报告。需要日线、每日指标、复权因子、涨跌停价和历史行业成分接口权限。")
        scope = st.selectbox("股票池", ["科技行业", "全A沪深"])
        min_price = st.number_input("最低股价（元）", min_value=0.0, value=10.0, step=1.0)
        min_mv = st.number_input("最低流通市值（亿元）", min_value=0.0, value=50.0, step=10.0)
        max_mv = st.number_input("最高流通市值（亿元）", min_value=1.0, value=1000.0, step=100.0)
        with st.expander("交易设置"):
            max_gap = st.number_input("统一组/主板买价溢价上限（%）", min_value=0.0, max_value=20.0, value=3.0, step=0.5)/100
            growth_gap = st.number_input("分板块组：创业板/科创板上限（%）", min_value=0.0, max_value=30.0, value=6.0, step=0.5)/100
            hold_days = st.selectbox("最多持有市场交易日（含买入日）", [2, 3, 4, 5], index=3)
            fee = st.number_input("单边综合手续费（万分之）", min_value=0.0, max_value=50.0, value=3.0, step=1.0)/10000
            slip = st.number_input("单边滑点（%）", min_value=0.0, max_value=2.0, value=0.1, step=0.05)/100
            st.caption("卖出另计历史印花税。止盈分别5%/10%，止损10%。默认最高买价含滑点；日线同日双触及按止损先发生。")
        research = st.checkbox("运行固定对照验证（22组）", value=True,
                               help="两种买价规则，各比较等权、四种单维40分、四种连续跑赢天数及两个对照。行情只下载一次。关闭后仅运行原版三组。")
        with st.expander("运行说明"):
            st.code("streamlit run app.py", language="bash")
            st.caption("依赖 pandas、numpy、streamlit、tushare。沿用原项目依赖即可。四路下载、逐日缓存；首次跨年运行需要下载较多数据。")
        cache_root = str(Path(os.environ.get("MOMENTUM_CACHE_DIR", "momentum_leader_cache")).resolve())
    mode = st.radio("运行方式", ["每日选股", "区间回测", "离线演示"], horizontal=True)
    default_end = ready_day().date()
    if mode == "区间回测":
        a, b = st.columns(2)
        start = a.date_input("信号开始日期", value=(pd.Timestamp(default_end)-pd.Timedelta(days=365)).date(), max_value=default_end)
        end = b.date_input("信号结束日期", value=default_end, max_value=default_end)
        st.caption("回测按信号日期筛选；必要时读取结束日后最多45个自然日，以观察退出。截止时尚未卖出的股票单列。")
    elif mode == "每日选股":
        end = st.date_input("收盘信号日期", value=default_end, max_value=default_end)
        start = end
        st.caption("非交易日自动使用此前最近交易日。北京时间18点前默认使用上一自然日，再按交易日历定位。")
    else:
        start = end = default_end
        st.warning("离线演示使用合成行情，检查界面和规则，不产生真实选股建议。")
    cfg = Config(start=ds(start), end=ds(end), scope=scope, min_price=min_price, min_mv=min_mv, max_mv=max_mv,
                 max_gap=max_gap, growth_gap=growth_gap, research=research, hold_days=hold_days, buy_fee=fee, sell_fee=fee, slippage=slip)
    if st.button("开始选股 / 回测" if mode != "离线演示" else "运行离线演示", type="primary"):
        try:
            cfg.validate()
            if mode != "离线演示" and not token.strip():
                st.error("请填写Tushare Token，或在Streamlit Secrets中配置 TUSHARE_TOKEN。")
            else:
                status = st.empty()
                with st.spinner("正在计算；已下载日期会保留，下次可继续使用。"):
                    folder = run_demo(Path(cache_root)/"demo") if mode == "离线演示" else run_online(token, cache_root, cfg, scan=mode == "每日选股", progress=status.info)
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
        self.cfg = Config(buy_fee=0, sell_fee=0, slippage=0)

    def bars(self):
        return pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0,
                             "pre_close": 100.0, "adj_factor": 1.0, "up_limit": 110.0,
                             "down_limit": 90.0, "vol": 10000.0, "amount": 10000.0,
                             "circ_mv": 1000000.0, "turnover_rate": 2.0}, index=self.calendar)

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
        selected = select_candidates(c, replace(self.cfg, top_n=1))
        self.assertEqual(selected[selected.group.eq(GROUP_MAIN)].iloc[0].ts_code, "300001.SZ")
        strong = selected[selected.group.eq("相对强度40")].iloc[0]
        self.assertEqual(strong.ts_code, "600001.SH")
        self.assertEqual(strong.strength_score, 90)
        self.assertEqual(strong.score, 60)
        self.assertEqual(len(experiments(self.cfg)), 22)

    def test_path_summary_excludes_incomplete_and_buy_day_only_hit(self):
        f = self.bars()
        f.loc[self.calendar[1], "high"] = 106
        result = simulate_trade(f, self.calendar, self.calendar[0], .05, self.cfg)
        partial = simulate_trade(f.iloc[:3], self.calendar[:3], self.calendar[0], .05, self.cfg)
        report = path_summary(pd.DataFrame([{**result, "group": "test"}, {**partial, "group": "test"}]))
        self.assertEqual(report.iloc[0]["完整5日观察数"], 1)
        self.assertEqual(report.iloc[0]["观察不完整或数据问题数"], 1)
        self.assertEqual(report.iloc[0]["可卖日期曾达5%比例"], 0)
        self.assertEqual(report.iloc[0]["仅买入当天达5%比例"], 100)

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
            events = []
            tr = run_trades(store, selected, cal, cfg, event_records=events)
            self.assertFalse(tr.empty)
            for _, f in tr[tr.status.eq("已平仓")].groupby(["group", "take_profit", "ts_code"]):
                f = f.sort_values("entry_date")
                self.assertTrue((f.entry_date.iloc[1:].values > f.exit_date.iloc[:-1].values).all())
            ledger, increment = gap_comparison(tr)
            self.assertFalse(ledger.empty)
            self.assertEqual(len(events), len(selected))
            folder = save_report(root, cfg, c, selected, tr, panel, cal, "test", "演示测试", pd.DataFrame(events))
            grid = report_table(folder, "experiment_summary")
            self.assertEqual(len(grid), 44)
            self.assertTrue(grid.loc[grid.experiment.eq("B0"), "相对等权均益差(百分点)"].dropna().eq(0).all())
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
