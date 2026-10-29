# -*- coding: utf-8 -*-
"""科技类 1-5 天选股 T1.0：只有两个模式，每日选股 / 区间回测。

设计要点（也是和旧 M1.7 最大的区别）：
- 界面只有两个模式。每日选股跑完会自动把首次留档写进
  data/forward_log.csv，前向跟踪不是一个单独模式，而是每日选股的副产品。
- 只用日线数据，不用分钟数据。Tushare 调用按天缓存到本地 SQLite，
  第二次跑同一天不再重复下载；首次回补 30 天左右，之后每天只补新的一天。
- 两个子策略分开记、分开评，不混成一个总收益：
  * 补涨型 CATCHUP：领涨板块内昨日涨幅前十、信号前 3 日涨幅偏低，持有 1-2 天。
  * 领涨延续型 LEAD：连续跑赢板块、收盘位置偏高，持有 3-5 天。
- 科创板、创业板、主板分板块算百分位、分板块出结果，不混排。
- 退出、成本、未成交、停牌、跌停卖不出都要留痕，不能算成 0 收益。

这只是研究代码，不是投资建议。任何参数在真正使用前，都要走一轮
冻结后的前向跟踪，不能只看回测。

运行：
  python app.py --self-test          # 不用 token，用合成演示数据验算链路
  streamlit run app.py               # 手机浏览器打开，选每日选股或区间回测
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

VERSION = "T1.0.1"
CORE_SECTORS = ["电子", "计算机", "通信", "国防军工"]
# 申万一级行业名在 Tushare index_classify 里就是上面这几个中文名。
BAR_COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
            "vol", "amount", "circ_mv", "turnover_rate", "adj_factor",
            "up_limit", "down_limit"]


def ds(x) -> str:
    return pd.Timestamp(str(x)).strftime("%Y%m%d")


def board_name(code: str) -> str:
    code = str(code)
    if code.startswith("30"):
        return "创业板"
    if code.startswith("68"):
        return "科创板"
    return "主板"


def gap_limit(code: str, cfg: "Config") -> float:
    # 主板 10% 涨跌停，创业板、科创板 20%，买入溢价上限也按板块分开。
    return cfg.max_gap_growth if board_name(code) != "主板" else cfg.max_gap_main


def stamp_tax(day) -> float:
    # 卖出印花税历史口径：2023-08-28 起 0.05%，之前 0.1%。
    return 0.0005 if pd.Timestamp(day) >= pd.Timestamp("2023-08-28") else 0.001


@dataclass(frozen=True)
class Config:
    scope: str = "科技行业"
    min_price: float = 10.0
    min_mv: float = 50.0      # 流通市值，亿元
    max_mv: float = 1000.0    # 流通市值，亿元
    top_n: int = 3            # 每个子策略每天最多 3 只
    min_sector_n: int = 5
    heat_limit: float = 0.50
    min_turnover_5d: float = 0.003  # 近 5 日平均换手率下限，0.3%
    hold_catchup: int = 2
    hold_lead: int = 5
    tp_catchup: float = 0.05
    tp_lead: float = 0.10
    stop_loss: float = 0.10
    buy_fee: float = 0.0003
    sell_fee: float = 0.0003
    slippage: float = 0.001
    max_gap_main: float = 0.03
    max_gap_growth: float = 0.06

    def validate(self):
        if self.min_mv > self.max_mv or self.min_price < 0:
            raise ValueError("价格或市值范围无效")
        if not 1 <= self.top_n <= 3:
            raise ValueError("每个子策略每天最多 3 只")
        if not 1 <= self.hold_catchup <= 2 or not 3 <= self.hold_lead <= 5:
            raise ValueError("补涨持有 1-2 天，领涨持有 3-5 天")
        if not 0 < self.stop_loss < 1:
            raise ValueError("止损无效")


# ---------------------------------------------------------------- 数据层
class MarketStore:
    """按天缓存日线到本地 SQLite，同一天不重复下载。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=30)
        cols = ",".join(c + " REAL" for c in BAR_COLS[2:])
        self.db.execute(
            f"CREATE TABLE IF NOT EXISTS bars (ts_code TEXT, trade_date TEXT, {cols}, "
            "sector TEXT, sector_name TEXT, name TEXT, is_st INTEGER, "
            "PRIMARY KEY(ts_code, trade_date)) WITHOUT ROWID")
        self.db.execute("CREATE TABLE IF NOT EXISTS days (day TEXT PRIMARY KEY, digest TEXT)")
        self.db.commit()

    def completed(self):
        return {r[0] for r in self.db.execute("SELECT day FROM days")}

    def put(self, day: str, frame: pd.DataFrame):
        f = frame.copy()
        digest = hashlib.sha256(f.to_csv(index=False).encode()).hexdigest()
        rows = f.astype(object).where(f.notna(), None).itertuples(index=False, name=None)
        with self.db:
            self.db.execute("DELETE FROM bars WHERE trade_date=?", (day,))
            cols = BAR_COLS + ["sector", "sector_name", "name", "is_st"]
            self.db.executemany(
                f"INSERT INTO bars VALUES ({','.join('?' for _ in cols)})", rows)
            self.db.execute("INSERT OR REPLACE INTO days VALUES (?,?)", (day, digest))

    def stock(self, code: str, start, end) -> pd.DataFrame:
        f = pd.read_sql_query(
            "SELECT * FROM bars WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
            self.db, params=(code, ds(start), ds(end)))
        if f.empty:
            return f
        f.index = pd.DatetimeIndex(pd.to_datetime(f.trade_date, format="%Y%m%d"), name="date")
        return f

    def all_bars(self, start, end) -> pd.DataFrame:
        f = pd.read_sql_query(
            "SELECT * FROM bars WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date, ts_code",
            self.db, params=(ds(start), ds(end)))
        if not f.empty:
            f["date"] = pd.to_datetime(f.trade_date, format="%Y%m%d")
        return f

    def close(self):
        self.db.close()


class DataClient:
    """Tushare 日线客户端。tushare 只在真正需要时才 import，没有 token 时不调用。

    省积分的做法：
    - 行业成分只取电子、计算机、通信、国防军工 4 个一级行业，元数据缓存 7 天。
    - 行情按天取，一天 4 个接口（daily / daily_basic / adj_factor / stk_limit），
      取完写 SQLite，下次同一天直接读缓存，不再花积分。
    - 回测一年大约是 240 多个交易日，首次回补会比较重，之后每天只补新的一天。
    """

    def __init__(self, token: str, root: Path, progress=lambda s: None):
        if not token or not token.strip():
            raise ValueError("需要 Tushare token 才能拉真实数据；没有 token 请用演示模式")
        import tushare as ts
        self.pro = ts.pro_api(token.strip(), timeout=25)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.progress = progress
        self._last = 0.0

    def _query(self, endpoint: str, **kwargs) -> pd.DataFrame:
        # 简单节流：同一接口两次调用至少间隔 0.4 秒，避免触发频次限制。
        pause = max(0.0, 0.4 - (time.monotonic() - self._last))
        if pause:
            time.sleep(pause)
        f = self.pro.query(endpoint, **kwargs)
        self._last = time.monotonic()
        if not isinstance(f, pd.DataFrame):
            raise RuntimeError(f"{endpoint} 返回类型异常")
        return f

    def _meta(self, name: str, endpoint: str, max_age_days: int = 7, **kwargs) -> pd.DataFrame:
        path = self.root / "metadata" / (name + ".csv.gz")
        if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
            try:
                return pd.read_csv(path, compression="gzip", dtype=str)
            except Exception:
                pass
        parts, offset = [], 0
        while True:
            part = self._query(endpoint, limit=5000, offset=offset, **kwargs)
            if part.empty:
                break
            parts.append(part)
            if len(part) < 5000:
                break
            offset += 5000
        f = pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame()
        if not f.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            f.to_csv(path, index=False, compression="gzip")
        return f

    def universe(self):
        """只取核心科技一级行业的成分，返回 basic / member / names 三张表。"""
        basic = self._meta("basic_L", "stock_basic", list_status="L",
                           fields="ts_code,name,list_date,delist_date")
        if basic.empty:
            raise RuntimeError("股票名单为空")
        basic = basic[basic.ts_code.str.match(r"^(60|68|00|30)\d{4}\.(SH|SZ)$")].copy()
        classes = self._meta("sw2021_l1", "index_classify", level="L1", src="SW2021")
        if classes.empty or "index_code" not in classes:
            raise RuntimeError("申万行业目录不可用")
        classes = classes[classes.industry_name.isin(CORE_SECTORS)]
        parts = []
        for row in classes.itertuples():
            for cur in ["Y", "N"]:
                f = self._meta(f"members_{row.index_code}_{cur}", "index_member_all",
                               l1_code=row.index_code, is_new=cur)
                parts.append(f)
        member = pd.concat(parts, ignore_index=True).drop_duplicates()
        member = member[member.ts_code.isin(basic.ts_code)].copy()
        # 只保留二级行业代码和名称、纳入/剔除日期，用于历史归属。
        member = member.rename(columns={"l2_code": "sector", "l2_name": "sector_name"})
        for c in ["in_date", "out_date"]:
            member[c] = pd.to_datetime(member[c], errors="coerce")
        names = self._meta("name_history", "namechange", max_age_days=7,
                           fields="ts_code,name,start_date,end_date,ann_date")
        return basic, member, names

    def calendar(self, start, end) -> pd.DatetimeIndex:
        f = self._query("trade_cal", exchange="SSE", start_date=ds(start),
                        end_date=ds(end), is_open="1")
        if f.empty or "cal_date" not in f:
            raise RuntimeError("交易日历为空")
        return pd.DatetimeIndex(pd.to_datetime(f.cal_date, format="%Y%m%d")).sort_values().unique()

    def fetch_day(self, day: str) -> pd.DataFrame:
        fields = {
            "daily": "ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
            "daily_basic": "ts_code,trade_date,circ_mv,turnover_rate",
            "adj_factor": "ts_code,trade_date,adj_factor",
            "stk_limit": "ts_code,trade_date,up_limit,down_limit",
        }
        frames = {}
        for ep, fd in fields.items():
            path = self.root / "pending" / ep / (day + ".csv.gz")
            f = pd.DataFrame()
            if path.exists():
                try:
                    f = pd.read_csv(path, compression="gzip", dtype=str)
                    if not set(fd.split(",")).issubset(f.columns):
                        f = pd.DataFrame()
                except Exception:
                    f = pd.DataFrame()
            if f.empty:
                f = self._query(ep, trade_date=day, fields=fd)
                if f.empty:
                    raise RuntimeError(f"{day} {ep} 为空")
                path.parent.mkdir(parents=True, exist_ok=True)
                f.to_csv(path, index=False, compression="gzip")
            frames[ep] = f
        f = frames["daily"].copy()
        f = f[f.ts_code.str.match(r"^(60|68|00|30)\d{4}\.(SH|SZ)$")].copy()
        for ep in ["daily_basic", "adj_factor", "stk_limit"]:
            f = f.merge(frames[ep], on=["ts_code", "trade_date"], how="left", validate="one_to_one")
        for c in BAR_COLS[2:]:
            f[c] = pd.to_numeric(f[c], errors="coerce")
        return f

    def download(self, calendar, store: MarketStore, basic, member, names):
        """把行情和历史归属拼成带 sector / name / is_st 的日线，写进 SQLite。"""
        todo = [ds(d) for d in calendar if ds(d) not in store.completed()]
        self.progress(f"行情共 {len(calendar)} 日，缓存 {len(calendar)-len(todo)} 日，待下载 {len(todo)} 日")
        # 历史名称：按当时日期判断 ST，不用当前名称过滤过去。
        name_map = {}
        if not names.empty:
            for code, g in names.groupby("ts_code"):
                name_map[code] = g
        for i, day in enumerate(todo, 1):
            raw = self.fetch_day(day)
            rows = []
            for row in raw.itertuples():
                m = member[(member.ts_code == row.ts_code)
                           & (member.in_date <= pd.Timestamp(day))
                           & ((member.out_date.isna()) | (member.out_date > pd.Timestamp(day)))]
                if m.empty:
                    continue
                sec = m.iloc[0]
                nm, is_st, known = row.ts_code, False, False
                if row.ts_code in name_map:
                    g = name_map[row.ts_code]
                    hit = g[(pd.to_datetime(g.start_date, errors="coerce") <= pd.Timestamp(day))
                            & ((pd.to_datetime(g.end_date, errors="coerce").isna())
                               | (pd.to_datetime(g.end_date, errors="coerce") >= pd.Timestamp(day)))]
                    if not hit.empty:
                        nm = hit.iloc[0]["name"]
                        is_st = bool("ST" in str(nm) or "退" in str(nm))
                        known = True
                rows.append({**{c: getattr(row, c) for c in BAR_COLS},
                             "sector": sec["sector"], "sector_name": sec["sector_name"],
                             "name": nm, "is_st": int(is_st)})
            frame = pd.DataFrame(rows)
            if not frame.empty:
                store.put(day, frame)
            for ep in ["daily", "daily_basic", "adj_factor", "stk_limit"]:
                (self.root / "pending" / ep / (day + ".csv.gz")).unlink(missing_ok=True)
            if i % 10 == 0 or i == len(todo):
                self.progress(f"行情下载 {i}/{len(todo)}")
        return store


# ---------------------------------------------------------------- 特征与选股
def sector_panel(bars: pd.DataFrame, calendar, cfg: Config) -> pd.DataFrame:
    """等权板块日收益 + 中位数 + 上涨家数占比。缺行业整日不填 0。"""
    b = bars.copy()
    b["ret"] = b.close / b.pre_close - 1
    b = b[b.vol > 0]
    g = b.groupby(["trade_date", "sector", "sector_name"], observed=True).agg(
        ret_sum=("ret", "sum"), members=("ret", "size"),
        up_count=("ret", lambda s: int((s > 0).sum())),
        median_ret=("ret", "median")).reset_index()
    g["sector_ret1"] = g.ret_sum / g.members
    g["breadth"] = g.up_count / g.members
    g["date"] = pd.to_datetime(g.trade_date, format="%Y%m%d")
    out = []
    for sec, f in g.groupby("sector", sort=False):
        f = f.set_index("date").reindex(calendar)
        f.index.name = "date"
        f["sector"] = sec
        f["sector_name"] = f.sector_name.ffill().bfill()
        logr = np.log1p(f.sector_ret1)
        for n in [2, 3, 5]:
            f[f"sector_ret{n}"] = np.expm1(logr.rolling(n, min_periods=n).sum())
        f["sector_pre3"] = f.sector_ret3.shift(1)
        out.append(f.reset_index())
    p = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if p.empty:
        return p
    p["sector_rank"] = np.nan
    valid = p.members.ge(cfg.min_sector_n) & p.sector_ret1.notna()
    ranked = p[valid].sort_values(["date", "sector_ret1", "sector"], ascending=[True, False, True])
    p.loc[ranked.index, "sector_rank"] = ranked.groupby("date").cumcount() + 1
    return p


def _max_runup(high, low, window=5):
    pairs = [(high / low.shift(lag) - 1).rolling(window + 1 - lag,
             min_periods=window + 1 - lag).max() for lag in range(1, window + 1)]
    return pd.concat(pairs, axis=1).max(axis=1, skipna=False)


def features(stock: pd.DataFrame, calendar) -> pd.DataFrame:
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
    draw = [(c / c.shift(k) - 1).rolling(6 - k).min() for k in range(1, 6)]
    f["drawdown5"] = pd.concat(draw, axis=1).min(axis=1, skipna=False).clip(upper=0)
    f["low_step"] = (l > l.shift(1)).astype(float).where(l.notna() & l.shift(1).notna()).rolling(5).mean()
    span = (h - l).replace(0, np.nan)
    clv = ((c - l) / span).where(span.notna(), 0.5).where(c.notna()).clip(0, 1)
    f["clv3"] = clv.rolling(3).mean()
    f["near_high"] = c / h.rolling(20).max()
    amount = f.amount.where(f.amount.gt(0))
    f["flow5"] = ((2 * clv - 1) * amount).rolling(5).sum() / amount.rolling(5).sum()
    f["amount_ratio"] = amount / amount.shift(1).rolling(20).mean()
    f["bias10"] = c / c.rolling(10).mean() - 1
    f["runup5"] = _max_runup(h, l, 5)
    f["turnover_5d"] = f.turnover_rate.rolling(5).mean()
    f["volatility5"] = r.rolling(5).std()
    return f


def _consecutive_rs(excess, sector, n: int):
    wins = excess.gt(1e-12).astype(float).where(excess.notna())
    same = pd.Series(True, index=excess.index)
    for lag in range(1, n):
        same &= sector.eq(sector.shift(lag)) & sector.notna()
    return wins.rolling(n, min_periods=n).sum().eq(n) & same


def build_candidates(store: MarketStore, panel: pd.DataFrame, calendar, cfg: Config,
                     progress=lambda s: None) -> pd.DataFrame:
    """只取当日板块第一名的股票，再算特征和门槛。"""
    top = panel[panel.sector_rank.eq(1)].copy()
    top_by_day = top.set_index("date").sector
    sector_tables = {s: f.set_index("date") for s, f in panel.groupby("sector", sort=False)}
    bars = store.all_bars(calendar[0], calendar[-1])
    if bars.empty:
        raise RuntimeError("没有行情数据，请先下载或用演示模式")
    rows = []
    codes = sorted(bars.ts_code.unique())
    for n, code in enumerate(codes, 1):
        sub = bars[bars.ts_code == code].copy()
        if sub.empty:
            continue
        stock = sub.set_index("date").sort_index()
        # 该股在哪些日期属于当日第一板块
        assignment = stock.sector if "sector" in stock else pd.Series(dtype=object)
        mask = assignment.eq(top_by_day.reindex(calendar)) & assignment.notna()
        if not mask.any():
            continue
        f = features(stock, calendar)
        f["sector"] = assignment
        for col in ["sector_ret1", "sector_ret2", "sector_ret3", "sector_ret5",
                    "sector_pre3", "breadth", "sector_name", "median_ret"]:
            f[col] = np.nan if col != "sector_name" else ""
        for sec in assignment.dropna().unique():
            idx = assignment.index[assignment.eq(sec)]
            p = sector_tables[sec].reindex(idx)
            for col in ["sector_ret1", "sector_ret2", "sector_ret3", "sector_ret5",
                        "sector_pre3", "breadth", "sector_name", "median_ret"]:
                f.loc[idx, col] = p[col].values
        f["rs3"], f["rs5"] = f.ret3 - f.sector_ret3, f.ret5 - f.sector_ret5
        f["lead_pre3"] = f.pre3 - f.sector_pre3
        excess = f.ret1 - f.sector_ret1
        f["persistence"] = excess.gt(0).astype(float).where(excess.notna()).rolling(5).mean()
        f["rs_accel"] = (excess - excess.shift(1).rolling(3, min_periods=3).mean())
        for days in [1, 2, 3, 5]:
            f[f"rs_all_{days}"] = _consecutive_rs(excess, assignment, days)
        f["signal_limit_ratio"] = f.up_limit / f.pre_close - 1
        f["signal_limit_up"] = f.close.ge(f.up_limit - 0.005).where(f.up_limit.gt(0) & f.close.notna())
        f["ts_code"] = code
        f["board"] = board_name(code)
        f["mv_yi"] = f.circ_mv / 10000
        f["overheat"] = f.ret5.ge(cfg.heat_limit - 1e-12) | f.runup5.ge(cfg.heat_limit - 1e-12)
        f["direction_ok"] = f.ret5.gt(0) & (f.rs3.gt(0) | f.rs5.gt(0))
        f["heat_penalty"] = ((f.ret5 - 0.25) / 0.25).clip(0, 1) * 10 + ((f.bias10 - 0.10) / 0.15).clip(0, 1) * 10
        f = f.loc[mask].copy()
        reasons = pd.Series("", index=f.index)
        gates = [
            (~f.history_ok, "不足21根连续日线"),
            (f.is_st.fillna(0).astype(bool) if "is_st" in f else False, "当时为ST"),
            (~f.close.ge(cfg.min_price), "价格不足或缺失"),
            (~f.mv_yi.between(cfg.min_mv, cfg.max_mv), "流通市值越界或缺失"),
            (~f.up_limit.gt(0) | ~f.down_limit.gt(0), "涨跌停价缺失"),
            (~f.turnover_5d.ge(cfg.min_turnover_5d), "近5日换手不足"),
            (~f.sector_ret1.gt(0), "第一板块未上涨"),
            (~f.breadth.ge(0.5), "板块上涨家数不足一半"),
            (~f.median_ret.gt(0), "板块中位数未上涨"),
            (~f.direction_ok, "近期未上涨或未跑赢板块"),
        ]
        for invalid, reason in gates:
            if isinstance(invalid, pd.Series):
                reasons.loc[invalid] += reason + "；"
        f["base_reason"] = reasons
        f["exclude_reason"] = reasons + np.where(f.overheat, "近5日暴涨达到阈值；", "")
        keep = ["ts_code", "name", "sector", "sector_name", "close", "mv_yi", "ret1", "ret3", "ret5",
                "rs3", "rs5", "lead_pre3", "persistence", "efficiency", "drawdown5", "low_step", "flow5",
                "amount_ratio", "clv3", "near_high", "bias10", "runup5", "volatility5",
                "sector_ret1", "breadth", "median_ret", "turnover_rate", "turnover_5d",
                "heat_penalty", "overheat", "base_reason", "exclude_reason", "board",
                "ret2", "pre3", "rs_accel", "signal_limit_up", "signal_limit_ratio"]
        keep += [f"rs_all_{d}" for d in [1, 2, 3, 5]]
        rows.append(f[keep].reset_index())
        if n % 50 == 0:
            progress(f"计算特征 {n}/{len(codes)} 只")
    if not rows:
        raise RuntimeError("所选日期没有可审计的候选记录")
    return score_candidates(pd.concat(rows, ignore_index=True))


def score_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    f = candidates.copy()
    eligible = f.base_reason.eq("")
    g = f.loc[eligible].copy()
    dims = {
        "strength_score": ["rs3", "rs5", "lead_pre3", "persistence"],
        "efficiency_score": ["efficiency", "drawdown5", "low_step"],
        "volume_score": ["flow5", "amount_ratio"],
        "position_score": ["clv3", "near_high"],
    }
    for target, inputs in dims.items():
        ranks = []
        for col in inputs:
            grouped = g.groupby("date")[col]
            ranks.append(100 * (grouped.rank(method="average") - 0.5) / grouped.transform("count"))
        g[target] = pd.concat(ranks, axis=1).mean(axis=1, skipna=False)
    for target in dims:
        f[target] = np.nan
        f.loc[g.index, target] = g[target]
    f["score"] = f[list(dims)].mean(axis=1, skipna=False) - f.heat_penalty
    invalid = eligible & f.score.isna()
    f.loc[invalid, "base_reason"] += "评分字段缺失；"
    f.loc[invalid, "exclude_reason"] += "评分字段缺失；"
    return f


def select_candidates(candidates: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """两个子策略分开选，各最多 top_n 只，过滤后不补位。"""
    base = candidates[candidates.base_reason.eq("") & ~candidates.overheat.eq(True)].copy()
    if base.empty:
        return base.iloc[:0].copy()
    base = base.sort_values(["date", "ret1", "ts_code"], ascending=[True, False, True])
    base["return_rank"] = base.groupby("date").cumcount() + 1
    top10 = base[base.return_rank.le(10)].copy()
    out = []
    # 补涨型：前十里信号前 3 日涨幅最低，持有 1-2 天。
    c = top10.sort_values(["date", "pre3", "return_rank", "ts_code"],
                          ascending=[True, True, True, True])
    c["rank"] = c.groupby("date").cumcount() + 1
    c = c[c["rank"].le(cfg.top_n)].copy()
    c["strategy"], c["hold_days"], c["take_profit"] = "补涨型", cfg.hold_catchup, cfg.tp_catchup
    out.append(c)
    # 领涨延续型：综合评分最高，且连续 3 日跑赢板块或 5 日跑赢占比高。
    lead = base[(base.rs_all_3.eq(True)) | (base.persistence.ge(0.6))].copy()
    lead = lead.sort_values(["date", "score", "return_rank", "ts_code"],
                            ascending=[True, False, True, True])
    lead["rank"] = lead.groupby("date").cumcount() + 1
    lead = lead[lead["rank"].le(cfg.top_n)].copy()
    lead["strategy"], lead["hold_days"], lead["take_profit"] = "领涨延续型", cfg.hold_lead, cfg.tp_lead
    out.append(lead)
    # 昨日涨幅对照：只按昨日涨幅排，不用任何打分，当基准。
    b = base.sort_values(["date", "ret1", "ts_code"], ascending=[True, False, True])
    b["rank"] = b.groupby("date").cumcount() + 1
    b = b[b["rank"].le(cfg.top_n)].copy()
    b["strategy"], b["hold_days"], b["take_profit"] = "昨日涨幅对照", cfg.hold_lead, cfg.tp_lead
    out.append(b)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 撮合与统计
def simulate_trade(stock: pd.DataFrame, calendar, signal, take_profit: float,
                   hold_days: int, cfg: Config) -> dict:
    """保守日线撮合：信号日收盘算，T+1 日收盘买，之后按收盘价止盈/止损/到期。

    未买入、停牌、跌停卖不出都不算成 0 收益，状态会单独留痕。
    """
    signal = pd.Timestamp(signal)
    pos = int(calendar.searchsorted(signal, side="right"))
    res = dict(status="待下一交易日", entry_date=pd.NaT, exit_date=pd.NaT,
               entry_price=np.nan, exit_price=np.nan, gap=np.nan,
               net_return=np.nan, gross_return=np.nan, reason="")
    if pos >= len(calendar):
        res["reason"] = "信号后无交易日"
        return res
    entry_day = calendar[pos]
    row = stock.loc[entry_day] if entry_day in stock.index else None
    if row is None or pd.isna(row.close):
        res.update(status="未买入", reason="买入日行情缺失或停牌")
        return res
    # 涨停买不进：收盘接近涨停，或收盘价超过信号收盘价*(1+溢价上限)。
    sig_close = stock.loc[signal, "close"] if signal in stock.index else np.nan
    gap_lim = gap_limit(stock.ts_code.iloc[0] if "ts_code" in stock else "", cfg) \
        if len(stock) else cfg.max_gap_main
    # 上面 ts_code 在 features 里不在 index 上，改用 stock 本身的第一行代码列（若有）
    if "ts_code" in stock.columns and len(stock):
        gap_lim = gap_limit(str(stock["ts_code"].dropna().iloc[0]), cfg)
    if row.close >= (row.up_limit - 0.005) if pd.notna(row.up_limit) else False:
        res.update(status="未买入", reason="买入日收盘涨停")
        return res
    if pd.notna(sig_close) and row.close > sig_close * (1 + gap_lim):
        res.update(status="未买入", reason="买入溢价超上限", gap=row.close / sig_close - 1)
        return res
    entry = float(row.close)
    res.update(status="未平仓", entry_date=entry_day, entry_price=entry)
    # 持有日含买入日：从买入日当天收盘后算起，最多 hold_days 个交易日。
    for i in range(hold_days):
        day = calendar[pos + i] if pos + i < len(calendar) else None
        if day is None:
            break
        r = stock.loc[day] if day in stock.index else None
        if r is None or pd.isna(r.close):
            continue  # 停牌顺延，不算平仓
        close = float(r.close)
        # 跌停卖不出：收盘接近跌停且不是买入日当天，顺延。
        if i > 0 and pd.notna(r.down_limit) and close <= r.down_limit + 0.005:
            continue
        gross_tp = close / entry - 1
        if gross_tp >= take_profit:
            exit_price, reason, status = close, "止盈", "已平仓"
        elif gross_tp <= -cfg.stop_loss:
            exit_price, reason, status = close, "止损", "已平仓"
        elif i == hold_days - 1:
            exit_price, reason, status = close, "到期", "已平仓"
        else:
            continue
        net = (exit_price * (1 - cfg.sell_fee - cfg.slippage - stamp_tax(day))
               / (entry * (1 + cfg.buy_fee + cfg.slippage)) - 1)
        res.update(status=status, exit_date=day, exit_price=exit_price,
                   gross_return=gross_tp, net_return=float(net), reason=reason)
        return res
    res["reason"] = "持有期结束仍未平仓（停牌或跌停顺延）"
    return res


def fixed_horizon(stock: pd.DataFrame, calendar, signal, entry_price: float) -> dict:
    """固定持有期收盘收益（不加止盈止损），用来看 1-5 天衰减。"""
    signal = pd.Timestamp(signal)
    pos = int(calendar.searchsorted(signal, side="right"))
    out = {}
    if pos >= len(calendar) or pd.isna(entry_price):
        return out
    # entry_price 是 T+1 日收盘价，h=1 就是再过 1 个交易日。
    for h in [1, 2, 3, 5]:
        day = calendar[pos + h] if pos + h < len(calendar) else None
        if day is None or day not in stock.index or pd.isna(stock.loc[day, "close"]):
            out[f"h{h}"] = np.nan
        else:
            out[f"h{h}"] = float(stock.loc[day, "close"] / entry_price - 1)
    return out


def run_trades(candidates: pd.DataFrame, store: MarketStore, calendar, cfg: Config):
    """对已选信号逐只撮合，返回逐笔表和固定持有期表。"""
    rows, horizons = [], []
    for row in candidates.itertuples():
        stock = store.stock(row.ts_code, calendar[0], calendar[-1])
        if stock.empty:
            continue
        tr = simulate_trade(stock, calendar, row.date, row.take_profit, int(row.hold_days), cfg)
        rows.append({**{k: getattr(row, k) for k in
                        ["date", "ts_code", "name", "sector", "sector_name", "board",
                         "strategy", "rank", "return_rank", "pre3", "ret1", "score",
                         "efficiency_score", "close"]},
                     **tr})
        if tr["status"] in ["已平仓", "未平仓"] and pd.notna(tr["entry_price"]):
            hz = fixed_horizon(stock, calendar, row.date, tr["entry_price"])
            horizons.append({"date": row.date, "ts_code": row.ts_code,
                             "strategy": row.strategy, **hz})
    return pd.DataFrame(rows), pd.DataFrame(horizons)


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    def agg(g):
        closed = g[g.status.eq("已平仓")]
        win = closed[closed.net_return.gt(0)]
        loss = closed[closed.net_return.le(0)]
        top5 = closed.nlargest(5, "net_return")["net_return"].sum() if len(closed) else 0.0
        total = closed["net_return"].sum() if len(closed) else 0.0
        trim = (total - top5) / max(len(closed), 1)
        return pd.Series({
            "信号数": len(g), "成交数": int(g.status.isin(["已平仓", "未平仓"]).sum()),
            "有效平仓数": len(closed), "未平仓数": int(g.status.eq("未平仓").sum()),
            "胜率%": float(win.shape[0] / max(len(closed), 1) * 100),
            "净均益%": float(closed.net_return.mean() * 100) if len(closed) else np.nan,
            "净中位数%": float(closed.net_return.median() * 100) if len(closed) else np.nan,
            "平均盈利%": float(win.net_return.mean() * 100) if len(win) else np.nan,
            "平均亏损%": float(loss.net_return.mean() * 100) if len(loss) else np.nan,
            "止盈占比%": float((closed.reason.eq("止盈").sum() / max(len(closed), 1)) * 100),
            "止损占比%": float((closed.reason.eq("止损").sum() / max(len(closed), 1)) * 100),
            "到期占比%": float((closed.reason.eq("到期").sum() / max(len(closed), 1)) * 100),
            "最差单笔%": float(closed.net_return.min() * 100) if len(closed) else np.nan,
            "剔除最大5笔后净均益%": float(trim * 100),
        })
    return trades.groupby("strategy").apply(agg).reset_index()


def save_forward_log(candidates: pd.DataFrame, root: Path, cfg: Config) -> str:
    """每日选股首次留档：同一信号日、同一策略只写第一次，之后不覆盖。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "forward_log.csv"
    cfg_hash = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True, default=str).encode()).hexdigest()[:12]
    cols = ["signal_date", "ts_code", "name", "sector", "strategy", "rank",
            "pre3", "ret1", "score", "close", "config_hash", "saved_at"]
    new = []
    existing = set()
    if path.exists():
        try:
            old = pd.read_csv(path, dtype=str)
            existing = set(zip(old.signal_date, old.ts_code, old.strategy))
        except Exception:
            existing = set()
    for row in candidates.itertuples():
        key = (pd.Timestamp(row.date).strftime("%Y-%m-%d"), str(row.ts_code), str(row.strategy))
        if key in existing:
            continue
        new.append({**{"signal_date": key[0], "ts_code": key[1], "strategy": key[2],
                        "name": getattr(row, "name", ""), "sector": getattr(row, "sector", ""),
                        "rank": getattr(row, "rank", ""), "pre3": getattr(row, "pre3", ""),
                        "ret1": getattr(row, "ret1", ""), "score": getattr(row, "score", ""),
                        "close": getattr(row, "close", "")},
                    "config_hash": cfg_hash,
                    "saved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")})
        existing.add(key)
    if new:
        pd.DataFrame(new, columns=cols).to_csv(path, mode="a", header=not path.exists(), index=False)
        return f"首次留档新增 {len(new)} 条，写入 {path}"
    return "该信号日已留档，本次不覆盖"


# ---------------------------------------------------------------- 演示数据
def demo_bars(seed: int = 42, n_stocks: int = 30, n_days: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range("2024-01-02", periods=n_days)
    sectors = ["电子", "计算机", "通信", "国防军工"]
    rows = []
    for i in range(n_stocks):
        code = f"{'30' if i % 3 == 0 else '68' if i % 3 == 1 else '00'}{100000 + i:06d}"[-8:]
        # 上面只是演示代码，统一成 ts_code 格式：用 6 位数字 + 后缀。
        num = 300000 + i if i % 3 == 0 else 688000 + i if i % 3 == 1 else 600000 + i
        suffix = "SZ" if i % 3 != 1 else "SH"
        if i % 3 == 1:
            suffix = "SH"
        ts = f"{num:06d}.{suffix}"
        sec = sectors[i % 4]
        price = 20.0 + (i % 10)
        closes = []
        for _ in range(n_days):
            price = price * (1 + rng.normal(0.001, 0.02))
            closes.append(price)
        closes = np.array(closes)
        for j, day in enumerate(calendar):
            pre = closes[j - 1] if j else closes[j] / 1.01
            rows.append({
                "ts_code": ts, "trade_date": day.strftime("%Y%m%d"),
                "open": closes[j] * 0.99, "high": closes[j] * 1.02, "low": closes[j] * 0.98,
                "close": closes[j], "pre_close": pre, "vol": 1_000_000,
                "amount": closes[j] * 1_000_000, "circ_mv": 500_000,
                "turnover_rate": 2.0, "adj_factor": 1.0,
                "up_limit": closes[j] * 1.2, "down_limit": closes[j] * 0.8,
                "sector": sec, "sector_name": sec, "name": f"演示{i}",
                "is_st": 0, "date": day,
            })
    return pd.DataFrame(rows)


class DemoStore:
    """自测用的内存 Store，接口和 MarketStore 的 stock / all_bars 一致。"""

    def __init__(self, bars: pd.DataFrame):
        self.bars = bars.copy()

    def stock(self, code, start, end):
        f = self.bars[self.bars.ts_code == code].copy()
        f = f[(f.date >= pd.Timestamp(start)) & (f.date <= pd.Timestamp(end))]
        return f.set_index("date").sort_index()

    def all_bars(self, start, end):
        f = self.bars[(self.bars.date >= pd.Timestamp(start)) & (self.bars.date <= pd.Timestamp(end))]
        return f.copy()


def self_test() -> str:
    cfg = Config()
    cfg.validate()
    bars = demo_bars()
    store = DemoStore(bars)
    calendar = pd.DatetimeIndex(sorted(bars.date.unique()))
    # 演示数据需要一个带 sector 的 panel：直接用 bars 构造。
    panel = sector_panel(bars.rename(columns={"date": "date"}), calendar, cfg) \
        if False else None
    # sector_panel 期望 trade_date 列，这里 bars 已有 trade_date，直接用。
    panel = sector_panel(bars, calendar, cfg)
    assert not panel.empty and panel.sector_rank.notna().any(), "板块面板为空"
    # build_candidates 期望 store.all_bars 返回带 date 列的长表。
    cand = build_candidates(store, panel, calendar, cfg)
    assert not cand.empty, "候选为空"
    sel = select_candidates(cand, cfg)
    assert not sel.empty and set(sel.strategy.unique()) <= {"补涨型", "领涨延续型", "昨日涨幅对照"}, sel.strategy.unique()
    trades, horizons = run_trades(sel, store, calendar, cfg)
    assert not trades.empty, "撮合为空"
    summ = summarize(trades)
    assert not summ.empty
    assert horizons.empty or set(horizons.columns) >= {"h1", "h2", "h3", "h5"}
    return (f"SELF_TEST_OK version={VERSION} calendar={len(calendar)} "
            f"candidates={len(cand)} selected={len(sel)} trades={len(trades)} "
            f"strategies={sorted(sel.strategy.unique())}")


# ---------------------------------------------------------------- 界面（只有两个模式）
def main_ui():
    import streamlit as st
    st.set_page_config(page_title="科技选股 T1.0", layout="wide")
    st.title("科技选股 T1.0")
    st.caption("只有两个模式：每日选股 / 区间回测。每日选股会自动首次留档，前向跟踪不是单独模式。")
    cfg = Config()
    mode = st.radio("运行方式", ["每日选股", "区间回测"], horizontal=True)
    token = st.text_input("Tushare token（只用于本次拉数据，不会写进聊天记录）",
                          type="password", help="没有 token 时只能跑演示数据")
    if mode == "每日选股":
        sig = st.date_input("收盘信号日期", value=datetime.now(ZoneInfo("Asia/Shanghai")).date())
        start = sig - timedelta(days=60)
        end = sig
    else:
        c1, c2 = st.columns(2)
        start = c1.date_input("信号开始日期", value=datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=365))
        end = c2.date_input("信号结束日期", value=datetime.now(ZoneInfo("Asia/Shanghai")).date())
    with st.expander("高级参数（默认就是冻结值，改动请先记下来）"):
        st.write(asdict(cfg))
    if st.button("开始选股 / 回测", type="primary"):
        try:
            cfg.validate()
            if token.strip():
                root = Path(__file__).parent / "data"
                client = DataClient(token, root, progress=st.write)
                cal = client.calendar(start, end)
                # 每日选股至少要 60 天历史做特征，回测则按区间取日历。
                hist_start = cal[0] - timedelta(days=90) if mode == "每日选股" else cal[0]
                cal_full = client.calendar(hist_start, end)
                store = MarketStore(root / "market.sqlite")
                basic, member, names = client.universe()
                client.download(cal_full, store, basic, member, names)
                panel = sector_panel(store.all_bars(cal_full[0], cal_full[-1]), cal_full, cfg)
                cand = build_candidates(store, panel, cal_full, cfg, progress=st.write)
                store.close()
            else:
                st.warning("没有 token，正在用合成演示数据跑流程，结果不是实盘信号。")
                bars = demo_bars()
                store = DemoStore(bars)
                cal_full = pd.DatetimeIndex(sorted(bars.date.unique()))
                panel = sector_panel(bars, cal_full, cfg)
                cand = build_candidates(store, panel, cal_full, cfg)
            sel = select_candidates(cand, cfg)
            if mode == "每日选股":
                day = pd.Timestamp(end)
                # 非交易日自动用此前最近交易日（和旧界面一致）。
                day = cal_full[cal_full <= day].max() if (cal_full <= day).any() else cal_full[0]
                today = sel[sel.date == day]
                st.subheader(f"下一交易日候选 · {day.date()} 信号")
                st.dataframe(today)
                st.info(save_forward_log(today, Path(__file__).parent / "data", cfg))
            else:
                trades, horizons = run_trades(sel, store, cal_full, cfg)
                st.subheader("汇总（按笔，不是组合净值）")
                st.dataframe(summarize(trades))
                st.subheader("固定持有期收益（不加止盈止损，用来看 1-5 天衰减）")
                if not horizons.empty:
                    st.dataframe(horizons.groupby("strategy")[["h1", "h2", "h3", "h5"]].mean())
                st.subheader("逐笔（节选）")
                st.dataframe(trades.head(200))
        except Exception as exc:
            st.error(f"运行失败：{type(exc).__name__}：{exc}")


def _in_streamlit_runtime() -> bool:
    """判断是不是正在被 Streamlit 运行（而不是 python app.py 直接运行）。"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        pass
    try:
        import streamlit as st
        return bool(st.runtime.exists())
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        print(self_test())
        return 0
    if _in_streamlit_runtime():
        # Streamlit 下必须调 main_ui() 才会渲染界面；
        # 之前版本这里只 print 了一句话，所以页面是空白的。
        main_ui()
        return 0
    print("Streamlit 界面请用：streamlit run app.py；自测请用：python app.py --self-test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
