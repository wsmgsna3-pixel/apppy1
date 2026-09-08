# -*- coding: utf-8 -*-
"""科技波段研究 T1.0 — streamlit run app.py

单文件；依赖 pandas、numpy、streamlit、tushare。python app.py --self-test 可离线验算。
策略阈值不是回测寻优结果。历史统计不构成策略有效或实盘合格证明。
官方数据字段：https://tushare.pro/document/2?doc_id=32 / 183 / 335
"""
from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

VERSION = "T1.0-FROZEN-20260908"
CACHE_SCHEMA = "t1_data_v1"
CORE = {"电子", "计算机", "通信", "国防军工"}
EXTENDED = {"机械设备", "电力设备", "医药生物", "汽车", "基础化工", "有色金属"}
TECH_WORDS = ("自动化", "机器人", "仪器仪表", "半导体", "光伏设备", "风电设备",
              "电池", "电网设备", "医疗器械", "电子", "金属新材料")
FALLBACK_WORDS = ("半导体", "元器件", "元件", "软件", "电脑", "通信", "电器仪表",
                  "航空", "专用机械", "电气设备", "医疗保健", "新型电力", "汽车配件")
RULES = {
    "A": "收盘突破此前20交易日最高价；此前20日最高/最低-1不超过20%",
    "B": "收盘在60日均线上且均线高于10日前；此前10日曾收于20日均线下；收盘突破此前5日最高价",
    "去重": "同股触发后锁定；至少5交易日后且连续两日收于10日均线下才重新待机",
    "排序": "20日相对收益、此前5/20日ATR收缩、至10日低点距离；当日候选百分位等权",
    "成交": "次交易日开盘；高开超过信号收盘5%、开盘涨停、缺价格/涨跌停数据时取消该次买单",
    "止损": "初始保护价=max(信号日前10日最低价,买入成交价×92%)；收盘跌破→次日开盘卖出",
    "跟踪": "每日收盘将保护价提高至max(原保护价,买入后最高收盘价−3×当日ATR14)，次日生效",
    "期限": "买入满8个自然周的首个交易日开盘退出；跌停/停牌可延迟并计入超期",
    "费用": "买入综合费用0.10%、卖出0.20%，每笔最低5元；另每边滑点0.10%；为保守统一假设",
    "仓位": "30万元起、最多3只；每个新仓目标为上日账户权益的1/3；不强制卖旧换新",
    "价格": "筛池/成交限制用不复权价；形态/收益用原价×当日复权因子",
    "公司行动": "按复权总收益折算持仓，近似红利再投资；不是逐笔派息送转税费账本",
    "周指标": "仅观察；MACD(12,26,9)；SKDJ=RSV9→EMA3→EMA3(K)→MA3(D)；周中只合成截至当日",
}


@dataclass(frozen=True)
class Config:
    start: str = "20220101"
    end: str = "20260908"
    min_price: float = 10.0
    min_mv: float = 50.0
    max_mv: float = 1000.0
    capital: float = 300000.0
    slots: int = 3
    random_runs: int = 100
    seed: int = 20260908


def stamp(x):
    return pd.Timestamp(str(x))


def ds(x):
    return pd.Timestamp(x).strftime("%Y%m%d")


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + str(threading.get_ident()) + ".tmp")
    frame.to_csv(temp, index=False, compression="gzip")
    os.replace(temp, path)


def atomic_bytes(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def load_csv(path):
    return pd.read_csv(path, compression="gzip", dtype={"ts_code": str, "trade_date": str,
                        "in_date": str, "out_date": str, "list_date": str, "delist_date": str})


class DataClient:
    """全市场逐日分端点缓存；成功立即落盘，失败和空表不缓存。"""
    def __init__(self, token, root, progress=lambda text: None):
        import tushare as ts
        self.pro = ts.pro_api(token.strip(), timeout=25)
        self.root = Path(root) / CACHE_SCHEMA
        self.progress = progress
        self.lock = threading.Lock()
        self.next_call = 0.0

    def query(self, endpoint, **kwargs):
        error = None
        for attempt in range(3):
            with self.lock:
                wait = max(0.0, self.next_call - time.monotonic())
                if wait:
                    time.sleep(wait)
                self.next_call = time.monotonic() + 0.36
            try:
                result = self.pro.query(endpoint, **kwargs)
                return result if isinstance(result, pd.DataFrame) else pd.DataFrame()
            except Exception as exc:
                error = exc
                # 权限/Token问题不反复消耗额度，也不在界面输出可能包含凭据的异常全文。
                if any(word in str(exc).lower() for word in ("token", "权限", "积分")):
                    raise RuntimeError(f"{endpoint} 接口认证或积分权限不足") from None
                time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"{endpoint} 三次请求失败（{type(error).__name__}）")

    def paged(self, endpoint, **kwargs):
        parts, seen = [], set()
        for offset in range(0, 200000, 1000):
            part = self.query(endpoint, limit=1000, offset=offset, **kwargs)
            if part.empty:
                break
            signature = hashlib.sha256(part.to_csv(index=False).encode()).hexdigest()
            if signature in seen:
                raise RuntimeError(f"{endpoint} 分页重复，无法证明已下载完整")
            seen.add(signature)
            parts.append(part)
            if len(part) < 1000:
                break
        else:
            raise RuntimeError(f"{endpoint} 超过分页上限")
        return pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame()

    def metadata(self, name, endpoint, ttl_days=7, **kwargs):
        path = self.root / "metadata" / (name + ".csv.gz")
        if path.exists() and time.time() - path.stat().st_mtime < ttl_days * 86400:
            try:
                return load_csv(path)
            except Exception:
                pass
        frame = self.paged(endpoint, **kwargs)
        if not frame.empty:
            atomic_csv(frame, path)
        return frame

    def calendar(self, start, end):
        frame = self.metadata(f"calendar_{start}_{end}", "trade_cal", exchange="SSE",
                              start_date=start, end_date=end, is_open="1", ttl_days=1)
        if frame.empty or "cal_date" not in frame:
            raise RuntimeError("未取得交易日历，不能构造真实回测日期")
        return pd.DatetimeIndex(sorted(pd.to_datetime(frame.cal_date.astype(str)).unique()))

    def universe(self):
        parts = []
        for status in ("L", "D", "P"):
            self.progress(f"读取股票名单：{status}")
            part = self.metadata("stocks_" + status, "stock_basic", list_status=status,
                                 fields="ts_code,name,industry,market,list_date,delist_date")
            if status == "L" and part.empty:
                raise RuntimeError("上市股票名单为空")
            if not part.empty:
                parts.append(part)
        basic = pd.concat(parts, ignore_index=True).drop_duplicates("ts_code")
        basic = basic[basic.ts_code.str.match(r"^(60|68|00|30)\d{4}\.(SH|SZ)$")].copy()
        warnings = []
        try:
            classes = self.metadata("sw2021_l1", "index_classify", level="L1", src="SW2021")
            targets = classes[classes.industry_name.isin(CORE | EXTENDED)]
            if set(targets.industry_name) != CORE | EXTENDED:
                raise RuntimeError("目标行业目录不完整")
            intervals = []
            for row in targets.itertuples():
                for current in ("Y", "N"):
                    self.progress(f"读取历史行业：{row.industry_name} / {current}")
                    part = self.metadata(f"members_{row.index_code}_{current}", "index_member_all",
                                         l1_code=row.index_code, is_new=current)
                    if not part.empty:
                        intervals.append(part)
                    elif current == "Y":
                        raise RuntimeError("当前行业成分缺失")
            member = pd.concat(intervals, ignore_index=True).drop_duplicates()
            labels = member.l2_name.fillna("") + " " + member.l3_name.fillna("")
            member = member[member.l1_name.isin(CORE) | labels.str.contains("|".join(TECH_WORDS))].copy()
            member = member[member.ts_code.isin(basic.ts_code)]
            member["in_date"] = pd.to_datetime(member.in_date, errors="coerce")
            member["out_date"] = pd.to_datetime(member.out_date, errors="coerce")
            if member.in_date.isna().any() or member.empty:
                raise RuntimeError("行业纳入日期缺失")
            if not (member.is_new == "N").any():
                warnings.append("历史退出成分为空，行业区间完整性未证实")
            mode = "历史行业区间（供应商历史记录，仍须核验完整性）"
        except Exception as exc:
            warnings.append(f"历史行业不可用：{exc}；退回基础行业快照，不能用于严格跨年结论")
            include = basic.industry.fillna("").str.contains("|".join(FALLBACK_WORDS))
            selected = basic[include]
            member = pd.DataFrame({"ts_code": selected.ts_code, "name": selected.name,
                "l1_name": selected.industry, "l2_name": "", "l3_name": "",
                "in_date": pd.to_datetime(selected.list_date, errors="coerce"), "out_date": pd.NaT})
            mode = "当前基础行业快照（含接口返回的退市股票，非历史归属）"
        codes = set(member.ts_code)
        if not codes:
            raise RuntimeError("科技股票池为空")
        return basic[basic.ts_code.isin(codes)].copy(), member, mode, warnings

    def day_endpoint(self, endpoint, day):
        fields = {
            "daily": "ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
            "daily_basic": "ts_code,trade_date,circ_mv,turnover_rate",
            "adj_factor": "ts_code,trade_date,adj_factor",
            "stk_limit": "ts_code,trade_date,up_limit,down_limit",
        }
        path = self.root / endpoint / (day + ".csv.gz")
        required = set(fields[endpoint].split(","))
        if path.exists():
            try:
                frame = load_csv(path)
                if required.issubset(frame) and frame.trade_date.astype(str).eq(day).all() and len(frame):
                    return frame
            except Exception:
                pass
        # 首次请求覆盖常见单日数据量；达到端点上限则显式分页，避免截断。
        frame = self.query(endpoint, trade_date=day, fields=fields[endpoint])
        cap = {"daily": 6000, "daily_basic": 6000, "adj_factor": 6000, "stk_limit": 5800}[endpoint]
        if len(frame) >= cap:
            frame = self.paged(endpoint, trade_date=day, fields=fields[endpoint])
        if frame.empty or not required.issubset(frame):
            raise RuntimeError(f"{endpoint} {day} 返回空表或缺字段")
        if not frame.trade_date.astype(str).eq(day).all() or frame.ts_code.duplicated().any():
            raise RuntimeError(f"{endpoint} {day} 日期或主键异常")
        atomic_csv(frame, path)
        return frame

    def download(self, calendar, codes):
        # 先检查必需接口权限，避免权限不足时仍向几千个历史日期重复请求。
        for endpoint in ("daily", "daily_basic", "adj_factor", "stk_limit"):
            self.progress(f"检查数据接口：{endpoint}")
            try:
                self.day_endpoint(endpoint, ds(calendar[-1]))
            except RuntimeError as exc:
                if "认证" in str(exc) or "权限" in str(exc):
                    raise

        def fetch(day):
            frames, issues = {}, []
            for endpoint in ("daily", "daily_basic", "adj_factor", "stk_limit"):
                try:
                    frame = self.day_endpoint(endpoint, day)
                    frames[endpoint] = frame[frame.ts_code.isin(codes)].copy()
                except Exception as exc:
                    issues.append({"date": day, "endpoint": endpoint, "problem": str(exc)})
            if "daily" not in frames:
                return pd.DataFrame(), issues
            merged = frames["daily"]
            for endpoint, cols in (("daily_basic", ["circ_mv", "turnover_rate"]),
                                    ("adj_factor", ["adj_factor"]),
                                    ("stk_limit", ["up_limit", "down_limit"])):
                if endpoint in frames:
                    merged = merged.merge(frames[endpoint][["ts_code", "trade_date"] + cols],
                                          on=["ts_code", "trade_date"], how="left", validate="one_to_one")
                else:
                    for col in cols:
                        merged[col] = np.nan
            if not merged.empty:
                for col in ("circ_mv", "adj_factor", "up_limit", "down_limit"):
                    count = int(merged[col].isna().sum())
                    if count:
                        issues.append({"date": day, "endpoint": "merge", "problem": f"{count}只缺{col}"})
            return merged, issues

        parts, issues = [], []
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = {executor.submit(fetch, ds(day)): day for day in calendar}
            for n, future in enumerate(as_completed(pending), 1):
                frame, errors = future.result()
                if not frame.empty:
                    parts.append(frame)
                issues.extend(errors)
                self.progress(f"行情下载/读取 {n}/{len(calendar)} 日；问题记录 {len(issues)}；成功数据已缓存")
        if not parts:
            raise RuntimeError("没有可用行情；成功端点缓存保留，可重新运行补齐")
        data = pd.concat(parts, ignore_index=True)
        data["date"] = pd.to_datetime(data.trade_date)
        return data, pd.DataFrame(issues, columns=["date", "endpoint", "problem"])


def weekly_observers(g):
    """周中指标只组合上一已完成周的状态与当前已知收盘/高低，不引用未来周五。"""
    c, h, l = g.ac, g.ah, g.al
    keys = g.index.to_period("W-FRI")
    w = pd.DataFrame({"c": c, "h": h, "l": l, "key": keys}).groupby("key").agg(
        c=("c", "last"), h=("h", "max"), l=("l", "min"))
    e12 = w.c.ewm(span=12, adjust=False).mean()
    e26 = w.c.ewm(span=26, adjust=False).mean()
    dif = e12 - e26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)
    hi, lo = w.h.rolling(9).max(), w.l.rolling(9).min()
    rsv = 100 * (w.c - lo) / (hi - lo).replace(0, np.nan)
    slow = rsv.ewm(span=3, adjust=False).mean()
    k = slow.ewm(span=3, adjust=False).mean()
    def mapped(s):
        return pd.Series(keys.map(s), index=g.index, dtype=float)
    p12, p26 = mapped(e12.shift()), mapped(e26.shift())
    pdif = (2 / 13 * c + 11 / 13 * p12) - (2 / 27 * c + 25 / 27 * p26)
    pdea = 0.2 * pdif + 0.8 * mapped(dea.shift())
    phist = 2 * (pdif - pdea)
    prior_hist = mapped(hist.shift())
    group = pd.DataFrame({"h": h, "l": l, "key": keys})
    partial_h = group.groupby("key").h.cummax()
    partial_l = group.groupby("key").l.cummin()
    hh = pd.concat([partial_h, mapped(w.h.rolling(8).max().shift())], axis=1).max(axis=1)
    ll = pd.concat([partial_l, mapped(w.l.rolling(8).min().shift())], axis=1).min(axis=1)
    prsv = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    pslow = 0.5 * prsv + 0.5 * mapped(slow.shift())
    pk = 0.5 * pslow + 0.5 * mapped(k.shift())
    pdk = (pk + mapped(k.shift()) + mapped(k.shift(2))) / 3
    state = np.select([phist.le(0), phist.gt(0) & prior_hist.le(0), phist.gt(prior_hist)],
                      ["绿柱", "首红", "红柱扩张"], default="红柱缩短")
    state = pd.Series(state, index=g.index).where(phist.notna(), "历史不足")
    return pd.DataFrame({"weekly_macd": phist, "weekly_state": state,
                         "weekly_k": pk, "weekly_d": pdk}, index=g.index)


def build_features(data, basic, member, calendar, cfg, progress=lambda text: None):
    stocks, candidates, returns_for_median = {}, [], []
    pool_counts = np.zeros(len(calendar), dtype=np.int32)
    raw_counts = np.zeros(len(calendar), dtype=np.int32)
    base = basic.set_index("ts_code")
    member_groups = {code: rows for code, rows in member.groupby("ts_code")}
    raw_groups = data.groupby("ts_code", sort=True)
    for n, (code, rows) in enumerate(raw_groups, 1):
        if code not in member_groups or code not in base.index:
            continue
        g = rows.drop_duplicates("date").set_index("date").sort_index().reindex(calendar)
        for col in ("open", "high", "low", "close", "pre_close", "vol", "amount", "circ_mv",
                    "turnover_rate", "adj_factor", "up_limit", "down_limit"):
            g[col] = pd.to_numeric(g[col], errors="coerce")
        for col, dest in (("open", "ao"), ("high", "ah"), ("low", "al"), ("close", "ac")):
            g[dest] = g[col] * g.adj_factor
        c = g.ac
        previous = c.shift()
        tr = pd.concat([g.ah - g.al, (g.ah - previous).abs(), (g.al - previous).abs()], axis=1).max(axis=1)
        tr = tr.where(g.ac.notna() & previous.notna())
        g["atr"] = tr.rolling(14).mean()
        ma10, ma20, ma60 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(60).mean()
        h20, l20 = g.ah.shift().rolling(20).max(), g.al.shift().rolling(20).min()
        g["structure"] = g.al.shift().rolling(10).min()
        g["ret20"] = c / c.shift(20) - 1
        g["contraction"] = tr.shift().rolling(5).mean() / tr.shift().rolling(20).mean()
        g["risk_distance"] = (c - g.structure) / c
        a = c.gt(h20) & (h20 / l20 - 1).le(0.20)
        pulled_back = c.lt(ma20).shift().rolling(10).max().eq(1)
        b = c.gt(ma60) & ma60.gt(ma60.shift(10)) & pulled_back & c.gt(g.ah.shift().rolling(5).max())
        active = np.zeros(len(g), dtype=bool)
        for m in member_groups[code].itertuples():
            active |= (calendar >= m.in_date) & (calendar < (m.out_date if pd.notna(m.out_date) else pd.Timestamp.max))
        listed = stamp(base.loc[code, "list_date"])
        delisted = pd.to_datetime(base.loc[code, "delist_date"], errors="coerce")
        active &= calendar >= listed + pd.Timedelta(days=180)
        if pd.notna(delisted):
            active &= calendar < delisted
        # 不用今天的ST名称剔除过去正常股票；依据当时限制价格识别主板风险警示特征。
        st_like = ((g.up_limit / g.pre_close - 1).lt(0.07) if code.startswith(("60", "00"))
                   else pd.Series(False, index=g.index))
        eligible = (active & g.close.gt(cfg.min_price) & (g.circ_mv / 10000).between(cfg.min_mv, cfg.max_mv)
                    & g.vol.gt(0) & g.adj_factor.gt(0) & g.up_limit.notna() & g.down_limit.notna() & ~st_like)
        g["eligible"] = eligible
        raw = (a | b) & eligible & g.atr.notna() & g.structure.gt(0) & g.ret20.notna()
        below_twice = c.lt(ma10) & c.shift().lt(ma10.shift())
        armed, last, events = True, -100, np.zeros(len(g), dtype=bool)
        for i in range(len(g)):
            if not armed and i - last >= 5 and bool(below_twice.iloc[i]):
                armed = True
            if armed and bool(raw.iloc[i]):
                events[i] = True
                armed, last = False, i
        g["signal"] = events
        g["source"] = np.select([a & b, a, b], ["A+B", "A", "B"], default="")
        g = g.join(weekly_observers(g))
        # 只将缺失价前向传递用于估值，绝不用于信号/成交。
        g["mark"] = g.ac.ffill()
        g["last_quote"] = pd.Series(calendar, index=calendar).where(g.ac.notna()).ffill()
        g["ts_code"] = code
        g["name"] = base.loc[code, "name"]
        stocks[code] = g[["open", "ao", "ac", "vol", "adj_factor", "up_limit", "down_limit",
                          "atr", "mark", "last_quote"]].copy()
        cols = ["ts_code", "name", "source", "close", "ac", "structure", "atr", "ret20", "contraction",
                "risk_distance", "circ_mv", "turnover_rate", "weekly_state", "weekly_k", "weekly_d", "weekly_macd"]
        select = g.loc[g.signal, cols].copy()
        select["date"] = select.index
        candidates.append(select.reset_index(drop=True))
        pool_counts += eligible.to_numpy(dtype=np.int32)
        raw_counts += raw.to_numpy(dtype=np.int32)
        returns_for_median.append(g.ret20.where(eligible).to_numpy())
        if n % 25 == 0 or n == len(raw_groups):
            progress(f"向量化计算 {n}/{len(raw_groups)} 只；买点与周指标仅使用当日及此前数据")
    if not stocks:
        raise RuntimeError("没有可计算股票")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_return = np.nanmedian(np.stack(returns_for_median), axis=0)
    cov = pd.DataFrame({"pool_count": pool_counts, "raw_count": raw_counts,
                        "pool_ret20": median_return}, index=calendar)
    cov.index.name = "date"
    signals = pd.concat(candidates, ignore_index=True)
    if not signals.empty:
        signals["rs"] = signals.ret20 - signals.date.map(cov.pool_ret20)
        grouped = signals.groupby("date")
        signals["score"] = 100 * (grouped.rs.rank(pct=True) +
                                      grouped.contraction.rank(pct=True, ascending=False) +
                                      grouped.risk_distance.rank(pct=True, ascending=False)) / 3
        signals = signals.sort_values(["date", "score", "ts_code"], ascending=[True, False, True])
        signals["rank"] = signals.groupby("date").cumcount() + 1
        signals["event_id"] = signals.date.dt.strftime("%Y%m%d") + "_" + signals.ts_code
        signals = signals[(signals.date >= stamp(cfg.start)) & (signals.date <= stamp(cfg.end))].reset_index(drop=True)
    else:
        for col in ("rs", "score", "rank", "event_id"):
            signals[col] = pd.Series(dtype=float if col != "event_id" else str)
    return stocks, signals, cov


def tradable_open(row, side):
    fields = ("open", "ao", "vol", "up_limit", "down_limit")
    if any(not np.isfinite(row[x]) for x in fields) or row["open"] <= 0 or row["vol"] <= 0:
        return False
    if side == "buy":
        return row["open"] < row["up_limit"] - 0.005
    return row["open"] > row["down_limit"] + 0.005


def execution_price(row, side):
    # 滑点价格不能越过当日法定限制；不用当日高低判断开盘是否成交。
    raw = min(row["open"] * 1.001, row["up_limit"]) if side == "buy" else max(row["open"] * 0.999, row["down_limit"])
    return raw * row["adj_factor"], raw


def path_for_signal(signal, g, calendar):
    """独立事件；同一退出轨迹复用于无限资金审计及三仓账户。"""
    out = signal.copy()
    idx = calendar.get_indexer([signal["date"]])[0]
    out.update(entry_status="区间末待成交", buy_date=pd.NaT, sell_date=pd.NaT,
               buy_idx=-1, sell_idx=-1, reason="", delayed_days=0, net_return=np.nan)
    if idx + 1 >= len(calendar):
        return out
    j = idx + 1
    row = g.iloc[j]
    if not tradable_open(row, "buy"):
        out["entry_status"] = "次日停牌/开盘涨停/数据不足，取消"
        return out
    if row.ao > signal["ac"] * 1.05:
        out["entry_status"] = "高开超过5%，取消"
        return out
    buy, raw_buy = execution_price(row, "buy")
    if row.ao <= signal["structure"]:
        out["entry_status"] = "开盘跌破形态低点，取消"
        return out
    out.update(entry_status="已成交", buy_date=calendar[j], buy_idx=j,
               buy_price=buy, raw_buy=raw_buy, factor=row.adj_factor)
    stop = max(signal["structure"], buy * 0.92)
    highest = buy
    pending = ""
    deadline = calendar[j] + pd.Timedelta(weeks=8)
    out["deadline"] = deadline
    for k in range(j, len(calendar)):
        row = g.iloc[k]
        if k > j and calendar[k] >= deadline and not pending:
            pending = "满8周"
        if pending and k > j:
            if tradable_open(row, "sell"):
                sell, raw_sell = execution_price(row, "sell")
                out.update(sell_idx=k, sell_date=calendar[k], sell_price=sell, raw_sell=raw_sell,
                           reason=pending, net_return=(sell * 0.998 / (buy * 1.001) - 1) * 100)
                break
            out["delayed_days"] += 1
        if np.isfinite(row.ac):
            # 先用昨天已经确定的保护价检查收盘，再抬升明天的保护价。
            if row.ac < stop and not pending:
                pending = "收盘跌破保护价"
            highest = max(highest, row.ac)
            if np.isfinite(row.atr):
                stop = max(stop, highest - 3 * row.atr)
    if out["sell_idx"] == -1:
        out["reason"] = pending or "仍持有"
    out["final_stop"] = stop
    for week in (1, 2, 4, 8):
        # 固定5/10/20/40交易日观察，与退出的8自然周上限分开标注。
        target = j + week * 5 - 1
        out[f"W{week}_net_pct"] = np.nan
        out[f"W{week}_status"] = "观察未满"
        if out["sell_idx"] >= 0 and out["sell_idx"] <= min(target, len(calendar) - 1):
            out[f"W{week}_net_pct"] = out["net_return"]
            out[f"W{week}_status"] = "已退出，资金保持现金"
        elif target < len(calendar) and np.isfinite(g.ac.iloc[target]):
            # 假设该收盘价可估值，不假设可以收盘成交；预扣卖出费用和滑点。
            out[f"W{week}_net_pct"] = (g.ac.iloc[target] * 0.999 * 0.998 / (buy * 1.001) - 1) * 100
            out[f"W{week}_status"] = "持仓收盘估值，预扣退出成本"
        elif target < len(calendar):
            out[f"W{week}_status"] = "目标日缺行情/停牌，收益未知"
    return out


def build_events(signals, stocks, calendar, progress=lambda text: None):
    result = []
    for n, signal in enumerate(signals.to_dict("records"), 1):
        result.append(path_for_signal(signal, stocks[signal["ts_code"]], calendar))
        if n % 100 == 0 or n == len(signals):
            progress(f"独立事件审计 {n}/{len(signals)}；提前退出的失败交易保留在后续统计中")
    if result:
        return pd.DataFrame(result)
    columns = list(signals.columns) + ["entry_status", "buy_date", "sell_date", "buy_idx", "sell_idx",
                                      "net_return", "reason", "delayed_days"]
    columns += [f"W{w}_{suffix}" for w in (1, 2, 4, 8) for suffix in ("net_pct", "status")]
    return pd.DataFrame(columns=columns)


def prepare_market(stocks):
    return {code: {"mark": g.mark.to_numpy(), "last_quote": g.last_quote.to_numpy()}
            for code, g in stocks.items()}


def portfolio(events, market, calendar, cfg, random_seed=None, detail=True, cost_mult=1.0):
    """买单优先级在信号日确定；不因看到次日无法成交而改买候补。"""
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    schedule = {}
    for event in events.to_dict("records"):
        signal_idx = calendar.get_indexer([event["date"]])[0]
        schedule.setdefault(signal_idx + 1, []).append(event)
    start, end = stamp(cfg.start), stamp(cfg.end)
    cash, prior_equity = cfg.capital, cfg.capital
    positions, ledger, curve, skipped = {}, [], [], []
    for i, day in enumerate(calendar):
        if day < start or day > end:
            continue
        # 开盘先执行已排队的卖单；8周到期同样是已知安排。
        for code, pos in list(positions.items()):
            event = pos["event"]
            if event["sell_idx"] == i:
                amount = pos["units"] * event["sell_price"]
                proceeds = amount - max(5.0, amount * 0.002 * cost_mult)
                cash += proceeds
                if detail:
                    ledger.append({"event_id": event["event_id"], "ts_code": code, "name": event["name"],
                        "source": event["source"], "rank": event["rank"], "buy_date": event["buy_date"],
                        "sell_date": day, "invested": pos["invested"], "proceeds": proceeds,
                        "pnl": proceeds - pos["invested"], "net_pct": 100 * (proceeds / pos["invested"] - 1),
                        "reason": event["reason"], "delayed_days": event["delayed_days"]})
                del positions[code]
        candidates = [e for e in schedule.get(i, []) if e["ts_code"] not in positions]
        candidates.sort(key=lambda e: (e["rank"], e["ts_code"]))
        if rng is not None:
            rng.shuffle(candidates)
        # 每个空位只分配一个买单；失败保留现金，不用未来可成交状态筛选候选。
        chosen = candidates[:cfg.slots - len(positions)]
        for event in chosen:
            if event["entry_status"] != "已成交":
                if detail:
                    skipped.append({"date": day, "event_id": event["event_id"], "reason": event["entry_status"]})
                continue
            target = min(cash, prior_equity / cfg.slots)
            raw_price = event["raw_buy"]
            code = event["ts_code"]
            # 科创板最低200股、超过部分1股；其他沪深A股整手100股。
            if code.startswith("68"):
                quantity = int(max(0, target - 5) / (raw_price * (1 + 0.001 * cost_mult)))
                quantity = quantity if quantity >= 200 else 0
            else:
                quantity = int(max(0, target - 5) / (raw_price * (1 + 0.001 * cost_mult)) / 100) * 100
            if quantity <= 0:
                if detail:
                    skipped.append({"date": day, "event_id": event["event_id"], "reason": "资金不足最小申报量"})
                continue
            amount = quantity * raw_price
            invested = amount + max(5.0, amount * 0.001 * cost_mult)
            cash -= invested
            assert cash >= -1e-6, "账户现金不能为负"
            positions[code] = {"event": event, "units": quantity / event["factor"],
                               "invested": invested, "quantity": quantity}
        exposure, stale, overdue = 0.0, 0, 0
        for code, pos in positions.items():
            quote = market[code]["mark"][i]
            exposure += pos["units"] * quote
            last = pd.Timestamp(market[code]["last_quote"][i])
            stale += int((day - last).days > 7)
            overdue += int(day >= pos["event"]["deadline"])
        equity = cash + exposure
        curve.append({"date": day, "equity": equity, "cash": cash, "positions": len(positions),
                      "exposure_pct": exposure / equity * 100 if equity else 0.0,
                      "stale_positions": stale, "overdue_positions": overdue})
        prior_equity = equity
    nav = pd.DataFrame(curve)
    open_rows = []
    if detail and len(nav):
        final_i = calendar.get_indexer([nav.date.iloc[-1]])[0]
        for code, pos in positions.items():
            event = pos["event"]
            value = pos["units"] * market[code]["mark"][final_i]
            open_rows.append({"ts_code": code, "name": event["name"], "event_id": event["event_id"],
                              "buy_date": event["buy_date"], "invested": pos["invested"],
                              "market_value": value, "unrealized_pnl": value - pos["invested"],
                              "deadline": event["deadline"]})
    return nav, pd.DataFrame(ledger), pd.DataFrame(open_rows), pd.DataFrame(skipped)


def max_drawdown(values, initial):
    a = np.r_[initial, np.asarray(values, dtype=float)]
    return float(np.min(a / np.maximum.accumulate(a) - 1) * 100)


def longest_true(values):
    longest = running = 0
    for value in values:
        running = running + 1 if value else 0
        longest = max(longest, running)
    return longest


def weekly_coverage(signals, events, nav, cov, calendar, cfg, issues):
    dates = calendar[(calendar >= stamp(cfg.start)) & (calendar <= stamp(cfg.end))]
    daily = pd.DataFrame(index=dates)
    daily["new_signals"] = signals.groupby("date").size().reindex(dates, fill_value=0)
    completed = events[events.entry_status == "已成交"]
    # 对应信号周可成交的新事件，不把已有持仓算成新信号。
    daily["executable_events"] = completed.groupby("date").size().reindex(dates, fill_value=0)
    daily["pool_count"] = cov.pool_count.reindex(dates, fill_value=0)
    daily["positions"] = nav.set_index("date").positions.reindex(dates, fill_value=0)
    daily["missing"] = 0
    if not issues.empty:
        problem_days = pd.to_datetime(issues.date.unique())
        daily.loc[daily.index.isin(problem_days), "missing"] = 1
    daily["week"] = daily.index.to_period("W-SUN")
    weekly = daily.groupby("week").agg(first_date=("new_signals", lambda x: x.index.min()),
        last_date=("new_signals", lambda x: x.index.max()), trading_days=("new_signals", "size"),
        new_signals=("new_signals", "sum"), executable_events=("executable_events", "sum"),
        max_positions=("positions", "max"), mean_positions=("positions", "mean"),
        min_pool=("pool_count", "min"), missing_days=("missing", "sum")).reset_index()
    if not signals.empty:
        unique = signals.groupby(signals.date.dt.to_period("W-SUN")).ts_code.nunique()
        weekly["unique_stocks"] = weekly.week.map(unique).fillna(0).astype(int)
    else:
        weekly["unique_stocks"] = 0
    weekly["year"] = weekly.first_date.dt.year
    weekly["no_new_signal"] = weekly.new_signals.eq(0)
    weekly["no_executable_signal"] = weekly.executable_events.eq(0)
    weekly["fully_flat"] = weekly.max_positions.eq(0)
    weekly["week"] = weekly.week.astype(str)
    return weekly


def annual_report(nav, trades, weekly, cfg, issues, pool_mode):
    result, prior = [], cfg.capital
    for year, group in nav.groupby(nav.date.dt.year, sort=True):
        w = weekly[weekly.year == year]
        t = trades[pd.to_datetime(trades.sell_date).dt.year == year] if not trades.empty else trades
        start_full = stamp(cfg.start) <= pd.Timestamp(year=int(year), month=1, day=1)
        end_full = stamp(cfg.end) >= pd.Timestamp(year=int(year), month=12, day=31)
        complete_year = start_full and end_full and latest_ready_day() >= pd.Timestamp(year=int(year), month=12, day=31)
        # 任何缺失都可能影响后续信号、去重或持仓；保守地不给该次运行任何年度通过结论。
        integrity = issues.empty and "历史行业区间" in pool_mode
        gap_count = int(w.no_new_signal.sum())
        verdict = "覆盖通过，盈利能力仍须验证" if gap_count <= 5 else "覆盖未达标"
        if not complete_year:
            verdict = "部分年度，不判全年达标"
        elif not integrity or (group.stale_positions > 0).any():
            verdict = "数据/股票池受限，不判通过"
        pnl = t.pnl if not t.empty else pd.Series(dtype=float)
        result.append({"年度": int(year), "账户收益%": (group.equity.iloc[-1] / prior - 1) * 100,
            "年内最大回撤%": max_drawdown(group.equity, prior), "已平仓笔数": len(t),
            "胜率%": (pnl > 0).mean() * 100 if len(t) else np.nan,
            "盈亏比": (pnl[pnl > 0].mean() / abs(pnl[pnl < 0].mean())) if (pnl < 0).any() else np.nan,
            "无新信号周": gap_count, "无可成交信号周": int(w.no_executable_signal.sum()),
            "最长连续无新信号周": longest_true(w.no_new_signal), "整周空仓": int(w.fully_flat.sum()),
            "只有1只新股的周": int(w.unique_stocks.eq(1).sum()), "至少3只新股的周": int(w.unique_stocks.ge(3).sum()),
            "平均资金使用率%": group.exposure_pct.mean(), "陈旧估值日": int(group.stale_positions.gt(0).sum()),
            "超8周持仓日": int(group.overdue_positions.gt(0).sum()), "判定": verdict})
        prior = group.equity.iloc[-1]
    return pd.DataFrame(result)


def signal_reports(events):
    rows = []
    if events.empty:
        return pd.DataFrame()
    for label, mask in [("全部", pd.Series(True, index=events.index)),
                         ("前三名", events["rank"] <= 3), ("第1名", events["rank"] == 1),
                         ("第2名", events["rank"] == 2), ("第3名", events["rank"] == 3),
                         ("A来源（含重叠）", events.source.isin(["A", "A+B"])),
                         ("B来源（含重叠）", events.source.isin(["B", "A+B"]))]:
        selected = events[mask]
        for year in ["全部"] + sorted(events.date.dt.year.unique().tolist()):
            part = selected if year == "全部" else selected[selected.date.dt.year == year]
            for week in (1, 2, 4, 8):
                values = pd.to_numeric(part[f"W{week}_net_pct"], errors="coerce").dropna()
                rows.append({"分组": label, "年度": str(year), "观察交易日": week * 5,
                             "信号数": len(part), "可成交数": int(part.entry_status.eq("已成交").sum()),
                             "收益已知数": len(values), "均收益%": values.mean(), "中位收益%": values.median(),
                             "胜率%": values.gt(0).mean() * 100 if len(values) else np.nan,
                             "10分位收益%": values.quantile(0.1) if len(values) else np.nan})
    return pd.DataFrame(rows)


def concentration_report(trades):
    if trades.empty:
        return pd.DataFrame()
    grouped = trades.groupby("ts_code").pnl.sum().sort_values(ascending=False)
    total = grouped.sum()
    return pd.DataFrame([{"项目": f"剔除最大盈利{k}只股票", "原已实现净利润": total,
        "剔除利润": grouped.head(k).clip(lower=0).sum(),
        "剩余已实现净利润": total - grouped.head(k).clip(lower=0).sum(),
        "说明": "已实现损益归因，非删除股票后重跑账户"} for k in (1, 3)])


def random_audit(events, market, calendar, cfg, actual_nav, progress=lambda text: None):
    rows = []
    for n in range(cfg.random_runs):
        nav, _, _, _ = portfolio(events, market, calendar, cfg, random_seed=cfg.seed + n, detail=False)
        if not nav.empty:
            prior = cfg.capital
            for year, group in nav.groupby(nav.date.dt.year):
                rows.append({"seed": cfg.seed + n, "year": str(year),
                             "return_pct": (group.equity.iloc[-1] / prior - 1) * 100,
                             "drawdown_pct": max_drawdown(group.equity, prior)})
                prior = group.equity.iloc[-1]
            rows.append({"seed": cfg.seed + n, "year": "全部", "return_pct": (prior / cfg.capital - 1) * 100,
                         "drawdown_pct": max_drawdown(nav.equity, cfg.capital)})
        if (n + 1) % 10 == 0:
            progress(f"同日候选随机三仓对照 {n + 1}/{cfg.random_runs}")
    raw = pd.DataFrame(rows)
    summary = []
    if not raw.empty:
        actuals, prior = {}, cfg.capital
        for year, group in actual_nav.groupby(actual_nav.date.dt.year):
            actuals[str(year)] = (group.equity.iloc[-1] / prior - 1) * 100
            prior = group.equity.iloc[-1]
        actuals["全部"] = (prior / cfg.capital - 1) * 100
        for year, group in raw.groupby("year"):
            actual = actuals[year]
            summary.append({"年度": year, "排序账户收益%": actual, "随机中位收益%": group.return_pct.median(),
                            "随机10分位%": group.return_pct.quantile(0.1), "随机90分位%": group.return_pct.quantile(0.9),
                            "超过随机比例%": (group.return_pct < actual).mean() * 100, "模拟次数": len(group)})
    return pd.DataFrame(summary), raw


def observer_report(events):
    if events.empty:
        return pd.DataFrame()
    e = events.copy()
    e["K区间"] = pd.cut(e.weekly_k, [-np.inf, 20, 50, 80, np.inf], labels=["<20", "20—50", "50—80", ">80"])
    rows = []
    for dimension in ("weekly_state", "K区间"):
        for label, group in e.groupby(dimension, observed=True):
            values = pd.to_numeric(group.W4_net_pct, errors="coerce").dropna()
            rows.append({"观察项": dimension, "状态": str(label), "信号数": len(group), "W4收益已知数": len(values),
                         "W4均收益%": values.mean(), "W4中位数%": values.median(),
                         "W4胜率%": values.gt(0).mean() * 100 if len(values) else np.nan,
                         "用途": "探索性分组，不自动据此增加门槛"})
    return pd.DataFrame(rows)


def make_zip(tables, manifest):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            archive.writestr(name + ".csv", frame.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        archive.writestr("规则与口径.txt", "\n".join(f"{key}：{value}" for key, value in RULES.items()) +
            "\n分组收益保留提前止损交易；未成熟/目标日缺行情不按0收益填充。\n"
            "账户收益含未平仓市值；胜率仅已平仓；独立事件观察收益与账户收益不可混用。\n"
            "跨年周按该周首个纳入观察的交易日归属；仅完整自然年判断五周覆盖目标。\n"
            "固定收益观察W1/2/4/8指5/10/20/40交易日；策略最长持仓为8个自然周。\n"
            "缺数据继续运行，但相关结果不认定通过。退市/长期停牌末价估值可能高估权益。\n"
            "当前ST名称只用于显示；主板5%限制用于保守过滤，创业/科创历史风险警示未完整重建。\n"
            "同日全体候选随机排序并按空仓位下单；不能预知次日是否成交再挑股票。\n"
            "100次随机是探索性参照，不是统计显著性或实盘资格证明。")
    return buffer.getvalue()


def run_research(token, cache_root, cfg, progress):
    client = DataClient(token, cache_root, progress)
    basic, member, mode, warnings = client.universe()
    ready = latest_ready_day()
    effective_end = min(stamp(cfg.end), ready)
    if effective_end < stamp(cfg.start):
        raise RuntimeError("截至可用行情日尚未进入指定回测区间")
    # 未来观察数据只用于事后审计，不参与日期内信号；账户仍截于用户指定结束日。
    history_start = ds(stamp(cfg.start) - pd.Timedelta(days=450))
    data_end = ds(min(ready, effective_end + pd.Timedelta(days=85)))
    calendar = client.calendar(history_start, data_end)
    progress(f"科技历史候选共 {len(basic)} 只；准备 {len(calendar)} 个交易日")
    data, issues = client.download(calendar, set(basic.ts_code))
    # 行哈希排序使指纹不受并行下载完成顺序影响，避免复制整张行情大表排序。
    row_hashes = pd.util.hash_pandas_object(data, index=False).to_numpy(copy=True)
    row_hashes.sort()
    data_hash = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    del row_hashes
    stocks, signals, cov = build_features(data, basic, member, calendar, cfg, progress)
    del data
    gc.collect()
    events = build_events(signals, stocks, calendar, progress)
    market = prepare_market(stocks)
    nav, trades, open_positions, skipped = portfolio(events, market, calendar, cfg)
    if nav.empty:
        raise RuntimeError("所选区间没有交易日")
    weeks = weekly_coverage(signals, events, nav, cov, calendar, cfg, issues)
    years = annual_report(nav, trades, weeks, cfg, issues, mode)
    random_summary, random_raw = random_audit(events, market, calendar, cfg, nav, progress)
    stress, _, _, _ = portfolio(events, market, calendar, cfg, detail=False, cost_mult=2.0)
    stress_summary = pd.DataFrame([{"情景": "基准费用", "总收益%": (nav.equity.iloc[-1] / cfg.capital - 1) * 100},
                                  {"情景": "买卖综合费用翻倍（滑点不变）", "总收益%": (stress.equity.iloc[-1] / cfg.capital - 1) * 100}])
    manifest = {"version": VERSION, "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "config": asdict(cfg), "rules": RULES, "pool_mode": mode, "warnings": warnings,
                "universe_size": len(basic), "actual_account_start": nav.date.min(), "actual_account_end": nav.date.max(),
                "data_start": history_start, "data_end": data_end, "data_issues": len(issues),
                "pool_hash": hashlib.sha256(member.to_csv(index=False).encode()).hexdigest(),
                "data_hash": data_hash,
                "limitations": ["复权总收益近似公司行动，非逐笔税费账本", "历史行业为供应商区间，未经独立完整性核验",
                                "科创/创业历史ST未完整重建", "日线不能还原开盘排队，涨跌停开盘保守不成交",
                                "没有实盘合格自动结论；历史已多次观察，须冻结后前向验证"]}
    if warnings:
        years["判定"] = "股票池受限，不判通过"
    tables = {"annual": years, "weekly_coverage": weeks, "equity": nav, "trades": trades,
              "open_positions": open_positions, "orders_skipped": skipped, "signals": signals,
              "all_events": events, "signal_groups": signal_reports(events), "random_summary": random_summary,
              "random_runs": random_raw, "concentration": concentration_report(trades),
              "observers": observer_report(events), "cost_stress": stress_summary, "data_issues": issues,
              "industry_intervals": member, "universe": basic, "daily_pool": cov.reset_index()}
    zipped = make_zip(tables, manifest)
    run_id = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:12]
    result_path = Path(cache_root) / "results" / f"{VERSION}_{run_id}.zip"
    atomic_bytes(zipped, result_path)
    return tables, manifest, zipped, str(result_path)


def latest_ready_day():
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    # 18点之前不请求未完成日线；周末/节假日由交易日历排除。
    return pd.Timestamp(now.date() - timedelta(days=1 if now.hour < 18 else 0))


def show_results(st, tables, manifest, zipped):
    cfg = manifest["config"]
    nav = tables["equity"]
    st.subheader(f"{manifest['version']} · {str(manifest['actual_account_start'])[:10]} 至 {str(manifest['actual_account_end'])[:10]}")
    st.caption("当前展示的是上次完成结果；修改侧栏后需重新运行才会更新。")
    metrics = st.columns(4)
    metrics[0].metric("账户总收益", f"{(nav.equity.iloc[-1] / cfg['capital'] - 1) * 100:.2f}%")
    metrics[1].metric("最大回撤", f"{max_drawdown(nav.equity, cfg['capital']):.2f}%")
    metrics[2].metric("独立新信号", len(tables["signals"]))
    metrics[3].metric("已平仓交易", len(tables["trades"]))
    st.info("这是冻结规则的研究版。空窗目标、排序优势、风险承受能力分别评估，不自动给出实盘合格结论。")
    if manifest["warnings"] or manifest["data_issues"]:
        st.warning("数据存在限制，本轮不能认定通过：" + "；".join(manifest["warnings"]) +
                   f"；缺失/异常记录 {manifest['data_issues']} 条。成功数据已缓存，下次运行补缺。")
    st.caption("股票池：" + manifest["pool_mode"] + "。公司行动按复权总收益近似；账户权益含未平仓市值。")
    st.download_button("下载完整回测结果（ZIP）", zipped,
        file_name=f"tech_swing_T1_0_{cfg['start']}_{cfg['end']}.zip", mime="application/zip")
    tabs = st.tabs(["逐年成绩", "覆盖与空窗", "买点与排名", "随机对照", "交易与持仓", "数据与规则"])
    with tabs[0]:
        st.dataframe(tables["annual"], use_container_width=True, hide_index=True)
        st.line_chart(nav.set_index("date")[["equity"]])
        st.dataframe(tables["concentration"], use_container_width=True, hide_index=True)
        st.dataframe(tables["cost_stress"], use_container_width=True, hide_index=True)
    with tabs[1]:
        st.caption("空窗=该交易周没有去重后的新信号。已有持仓不冲抵空窗；跨年周按首个观察交易日归属。")
        st.dataframe(tables["weekly_coverage"], use_container_width=True, hide_index=True)
    with tabs[2]:
        st.caption("W1/2/4/8=5/10/20/40交易日观察。提前退出交易保留收益；未成熟和缺行情显示未知。独立事件按无限资金计算。")
        st.dataframe(tables["signal_groups"], use_container_width=True, hide_index=True)
        st.dataframe(tables["observers"], use_container_width=True, hide_index=True)
        st.dataframe(tables["signals"].tail(200), use_container_width=True, hide_index=True)
    with tabs[3]:
        st.caption("100次同日候选随机排序，沿用相同资金、空位、退出、费用和成交限制。不是统计显著性结论。")
        st.dataframe(tables["random_summary"], use_container_width=True, hide_index=True)
    with tabs[4]:
        st.write("期末仍持有")
        st.dataframe(tables["open_positions"], use_container_width=True, hide_index=True)
        st.write("已平仓")
        st.dataframe(tables["trades"], use_container_width=True, hide_index=True)
        st.write("取消或未成交的账户买单")
        st.dataframe(tables["orders_skipped"], use_container_width=True, hide_index=True)
    with tabs[5]:
        st.dataframe(pd.DataFrame(RULES.items(), columns=["项目", "冻结规则"]), use_container_width=True, hide_index=True)
        st.dataframe(tables["data_issues"], use_container_width=True, hide_index=True)
        st.json(manifest)


def self_test():
    """独立的合成行情边界测试；不下载真实行情，也不宣称策略收益。"""
    import unittest

    class EngineTests(unittest.TestCase):
        def setUp(self):
            self.cal = pd.bdate_range("2022-01-03", periods=100)
            self.g = pd.DataFrame(index=self.cal)
            for col in ("open", "close", "ao", "ac", "mark"):
                self.g[col] = 20.0
            self.g["high"], self.g["low"] = 20.5, 19.5
            self.g["ah"], self.g["al"] = 20.5, 19.5
            self.g["vol"] = 10000.0
            self.g["adj_factor"] = 1.0
            self.g["up_limit"], self.g["down_limit"] = 22.0, 18.0
            self.g["atr"] = 1.0
            self.g["last_quote"] = self.cal
            self.signal = {"date": self.cal[0], "ts_code": "600001.SH", "name": "合成股票", "source": "A",
                "rank": 1, "ac": 20.0, "structure": 18.5, "event_id": "test", "score": 100.0}

        def test_next_day_and_time_cap(self):
            event = path_for_signal(self.signal, self.g, self.cal)
            self.assertEqual(event["buy_idx"], 1)
            self.assertEqual(event["sell_date"], self.cal[1] + pd.Timedelta(weeks=8))
            self.assertLess(event["net_return"], 0)  # 平价交易仍然支付费用与滑点

        def test_t_plus_one_and_failed_trades_preserved(self):
            self.g.loc[self.cal[1], ["close", "ac"]] = 17.5
            event = path_for_signal(self.signal, self.g, self.cal)
            self.assertEqual(event["sell_idx"], 2)  # 买入当日不能卖出
            self.assertAlmostEqual(event["W8_net_pct"], event["net_return"])
            self.assertEqual(event["W8_status"], "已退出，资金保持现金")

        def test_limit_down_delays_exit(self):
            self.g.loc[self.cal[1], ["close", "ac"]] = 17.5
            self.g.loc[self.cal[2], ["open", "ao"]] = 18.0
            event = path_for_signal(self.signal, self.g, self.cal)
            self.assertEqual(event["sell_idx"], 3)
            self.assertEqual(event["delayed_days"], 1)

        def test_missing_quote_does_not_compress_time(self):
            self.g.loc[self.cal[2:10], ["open", "ao", "ac", "close"]] = np.nan
            event = path_for_signal(self.signal, self.g, self.cal)
            self.assertEqual(event["sell_date"], self.cal[1] + pd.Timedelta(weeks=8))
            self.assertTrue(np.isnan(event["W1_net_pct"]))

        def test_no_hindsight_replacement(self):
            events, market = [], {}
            for k in range(4):
                signal = dict(self.signal, ts_code=f"60000{k}.SH", event_id=str(k), rank=k+1)
                g = self.g.copy()
                if k == 0:
                    g.loc[self.cal[1], ["open", "ao"]] = 22.0
                events.append(path_for_signal(signal, g, self.cal))
                market[signal["ts_code"]] = {"mark": g.mark.to_numpy(), "last_quote": g.last_quote.to_numpy()}
            cfg = Config(start=ds(self.cal[0]), end=ds(self.cal[10]))
            nav, _, positions, skipped = portfolio(pd.DataFrame(events), market, self.cal, cfg)
            self.assertEqual(nav.positions.max(), 2)
            self.assertNotIn("600003.SH", positions.ts_code.tolist())
            self.assertEqual(len(skipped), 1)
            self.assertTrue(nav.cash.ge(0).all())

        def test_unmatured_is_unknown(self):
            event = path_for_signal(self.signal, self.g.iloc[:4], self.cal[:4])
            self.assertTrue(np.isnan(event["W8_net_pct"]))
            self.assertEqual(event["sell_idx"], -1)

        def test_account_conservation(self):
            event = path_for_signal(self.signal, self.g, self.cal)
            cfg = Config(start=ds(self.cal[0]), end=ds(self.cal[-1]))
            market = prepare_market({"600001.SH": self.g})
            nav, trades, positions, _ = portfolio(pd.DataFrame([event]), market, self.cal, cfg)
            self.assertTrue(positions.empty)
            self.assertAlmostEqual(nav.equity.iloc[-1], cfg.capital + trades.pnl.sum())
            self.assertTrue((nav.positions <= 3).all())

        def test_split_is_not_loss(self):
            baseline = path_for_signal(self.signal, self.g, self.cal)
            g = self.g.copy()
            after = self.cal >= self.cal[10]
            g.loc[after, ["open", "close", "high", "low", "up_limit", "down_limit"]] /= 2
            g.loc[after, "adj_factor"] = 2.0
            split = path_for_signal(self.signal, g, self.cal)
            self.assertAlmostEqual(split["net_return"], baseline["net_return"])

        def test_weekly_prefix_invariance(self):
            g = self.g.copy()
            g["ac"] = 20 + np.sin(np.arange(len(g))/4) + np.arange(len(g)) * .03
            g["ah"], g["al"] = g.ac + .5, g.ac - .5
            full = weekly_observers(g)
            prefix = weekly_observers(g.iloc[:73])
            pd.testing.assert_frame_equal(full.iloc[:73], prefix)

        def test_feature_and_rank_prefix_invariance(self):
            rng = np.random.default_rng(17)
            cal = pd.bdate_range("2020-01-01", periods=350)
            frames, basics, members = [], [], []
            for k in range(5):
                code = f"60000{k}.SH"
                close = 20 * np.exp(np.cumsum(rng.normal(.002, .014, len(cal))))
                frames.append(pd.DataFrame({"date": cal, "ts_code": code, "open": close*.999, "close": close,
                    "high": close*1.006, "low": close*.994, "pre_close": np.r_[close[0], close[:-1]],
                    "vol": 10000., "amount": 20000., "circ_mv": 2000000., "turnover_rate": 1.,
                    "adj_factor": 1., "up_limit": close*1.1, "down_limit": close*.9}))
                basics.append({"ts_code": code, "name": code, "list_date": "20100101", "delist_date": None})
                members.append({"ts_code": code, "in_date": pd.Timestamp("2010-01-01"), "out_date": pd.NaT})
            data, basic, member = pd.concat(frames), pd.DataFrame(basics), pd.DataFrame(members)
            cfg = Config(start=ds(cal[80]), end=ds(cal[-1]), random_runs=3)
            stocks, signals, cov = build_features(data, basic, member, cal, cfg)
            cut = cal[231]
            _, partial, _ = build_features(data[data.date <= cut], basic, member, cal[:232], cfg)
            self.assertGreater(len(signals), 0)
            pd.testing.assert_frame_equal(signals[signals.date <= cut].reset_index(drop=True), partial)
            events = build_events(signals, stocks, cal)
            market = prepare_market(stocks)
            nav, trades, opened, skipped = portfolio(events, market, cal, cfg)
            issues = pd.DataFrame(columns=["date", "endpoint", "problem"])
            weeks = weekly_coverage(signals, events, nav, cov, cal, cfg, issues)
            years = annual_report(nav, trades, weeks, cfg, issues, "历史行业区间")
            summary, raw = random_audit(events, market, cal, cfg, nav)
            self.assertEqual(raw.seed.nunique(), 3)
            self.assertGreater(len(signal_reports(events)), 0)
            self.assertFalse(summary.empty)
            zipped = make_zip({"annual": years, "events": events}, {"synthetic": True})
            self.assertTrue(zipfile.is_zipfile(io.BytesIO(zipped)))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EngineTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    import streamlit as st
    st.set_page_config(page_title="科技波段 T1.0", layout="wide")
    st.title("科技波段 T1.0 · 独立研发第一版")
    st.write("整理突破 / 回调再启动 · 统一排序和退出 · 30万元三仓 · 持仓上限8周")
    try:
        default_token = str(st.secrets.get("TUSHARE_TOKEN", st.secrets.get("tushare_token", "")))
    except Exception:
        default_token = ""
    default_token = os.environ.get("TUSHARE_TOKEN", default_token)
    with st.sidebar:
        token = st.text_input("Tushare Token", value=default_token, type="password")
        start = st.date_input("回测开始", value=date(2022, 1, 1))
        end = st.date_input("回测结束", value=latest_ready_day().date())
        price = st.number_input("最低股价（高于，元）", value=10.0, min_value=0.0, step=1.0)
        min_mv = st.number_input("最低流通市值（亿元）", value=50.0, min_value=0.0, step=10.0)
        max_mv = st.number_input("最高流通市值（亿元）", value=1000.0, min_value=1.0, step=100.0)
        cache_root = st.text_input("数据缓存目录", value="tech_swing_cache")
        st.caption("日线、每日市值、复权因子和涨跌停价首次下载较多。缓存长期保留；托管服务重启能否保留取决于磁盘。")
        st.caption("科技范围固定：电子、计算机、通信、国防军工，以及自动化/新能源设备/医疗器械等指定细分。明细随结果导出。")
        run = st.button("运行冻结规则回测", type="primary", use_container_width=True)
    with st.expander("第一版规则与数据边界"):
        st.dataframe(pd.DataFrame(RULES.items(), columns=["项目", "规则"]), use_container_width=True, hide_index=True)
        st.write("缺行情跳过，未知收益不填零；按原交易日历推进持仓年龄。历史行业权限不足会显示快照池限制。")
        st.write("不承诺每年五周空窗；结果直接报告达标与否。参数未依据本次真实回测优化。")
    if run:
        if not token.strip():
            st.error("请输入Tushare Token，或在 secrets 中设置 TUSHARE_TOKEN。")
        elif start > end or min_mv >= max_mv or not cache_root.strip():
            st.error("请检查起止日期、市值上下限和缓存目录。")
        else:
            progress_box = st.empty()
            last_update = [0.0]
            def progress(message):
                if time.monotonic() - last_update[0] > 0.25:
                    progress_box.info(message)
                    last_update[0] = time.monotonic()
            cfg = Config(start=ds(start), end=ds(end), min_price=price, min_mv=min_mv, max_mv=max_mv)
            try:
                result = run_research(token, cache_root.strip(), cfg, progress)
                st.session_state["t1_result"] = result
                progress_box.success("回测完成。结果保留在当前会话，点击下载不会重新下载行情或重跑回测。")
            except Exception as exc:
                progress_box.error("运行未完成：" + str(exc).replace(token, "[隐藏]")[:500])
                st.info("已成功下载的数据已逐端点保存。修复权限或网络后重新运行即可补缺。")
    if "t1_result" in st.session_state:
        tables, manifest, zipped, _ = st.session_state["t1_result"]
        show_results(st, tables, manifest, zipped)
    else:
        st.info("首次建议直接测试2022年至最近完整交易日，逐年查看结果。这里尚无真实行情回测结论。")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
