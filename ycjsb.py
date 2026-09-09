# -*- coding: utf-8 -*-
"""科技波段研究 T2.0 收缩转强假设 — streamlit run app.py

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

VERSION = "T2.0-CONTRACTION-TURN-20260910"
BASELINE_VERSION = "T1.0-FROZEN-20260908"
EXIT_SCHEMES = {"none": "不提前退出（满8自然周）", "initial": "仅初始保护", "full": "沿用的完整退出对照"}
DOWNLOAD_REVISION = "DL4"
DOWNLOAD_WORKERS = 4
API_MIN_INTERVAL = 0.36
CACHE_SCHEMA = "t1_data_v1"
CORE = {"电子", "计算机", "通信", "国防军工"}
EXTENDED = {"机械设备", "电力设备", "医药生物", "汽车", "基础化工", "有色金属"}
TECH_WORDS = ("自动化", "机器人", "仪器仪表", "半导体", "光伏设备", "风电设备",
              "电池", "电网设备", "医疗器械", "电子", "金属新材料")
FALLBACK_WORDS = ("半导体", "元器件", "元件", "软件", "电脑", "通信", "电器仪表",
                  "航空", "专用机械", "电气设备", "医疗保健", "新型电力", "汽车配件")
RULES = {
    "C回调": "上一已完成周收盘较截至该周的13周最高价回撤至少10%",
    "C收缩": "上一已完成周真实波幅小于此前4个已完成周的平均真实波幅",
    "C转强": "日收盘首次突破此前3个交易日最高价；只用当时已知信息",
    "去重": "同股触发后锁定；至少5交易日后且连续两日收于10日均线下才重新待机",
    "评分": "停用评分和名次筛选；全部C事件独立研究，代码顺序只用于显示",
    "成交": "次交易日开盘；高开超过信号收盘5%、开盘涨停、缺价格/涨跌停数据时取消该次买单",
    "止损": "初始保护价=max(信号日前10日最低价,买入成交价×92%)；收盘跌破→次日开盘卖出",
    "跟踪": "每日收盘将保护价提高至max(原保护价,买入后最高收盘价−3×当日ATR14)，次日生效",
    "期限": "买入满8个自然周的首个交易日开盘退出；跌停/停牌可延迟并计入超期",
    "费用": "买入比例费用0.10%、卖出0.20%；另每边滑点0.10%；独立收益不模拟最低收费",
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
        self.rate_states = {}

    def rate_state(self, endpoint):
        # 分接口限速：日线等待时不阻塞市值、复权因子、涨跌停价请求。
        with self.lock:
            if endpoint not in self.rate_states:
                self.rate_states[endpoint] = {"lock": threading.Lock(), "next_call": 0.0,
                                              "interval": API_MIN_INTERVAL}
            return self.rate_states[endpoint]

    def query(self, endpoint, **kwargs):
        error = None
        state = self.rate_state(endpoint)
        for attempt in range(3):
            with state["lock"]:
                wait = max(0.0, state["next_call"] - time.monotonic())
                if wait:
                    time.sleep(wait)
                state["next_call"] = time.monotonic() + state["interval"]
            try:
                result = self.pro.query(endpoint, **kwargs)
                return result if isinstance(result, pd.DataFrame) else pd.DataFrame()
            except Exception as exc:
                error = exc
                message = str(exc).lower()
                # 频次报错常附带“权限详情”，必须先识别限流，不能误报Token无效。
                if any(word in message for word in ("每分钟", "频次", "rate limit", "too many requests", "429")):
                    with state["lock"]:
                        state["interval"] = min(2.0, state["interval"] * 2)
                        state["next_call"] = max(state["next_call"], time.monotonic() + 60.0)
                    if attempt == 2:
                        raise RuntimeError(f"{endpoint} 频次限制，退避重试仍失败；已缓存数据保留") from None
                    continue
                # 权限/Token问题不反复消耗额度，也不在界面输出可能包含凭据的异常全文。
                if any(word in message for word in ("token", "权限", "积分")):
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
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            pending = {executor.submit(fetch, ds(day)): day for day in calendar}
            for n, future in enumerate(as_completed(pending), 1):
                frame, errors = future.result()
                if not frame.empty:
                    parts.append(frame)
                issues.extend(errors)
                self.progress(f"{DOWNLOAD_WORKERS}路并发下载/读取 {n}/{len(calendar)} 日；问题记录 {len(issues)}；成功数据已缓存")
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


def contraction_turn_setup(g):
    """周线只用前一日历周及更早数据；当前周最终值不会进入当前周信号。"""
    keys=g.index.to_period('W-FRI')
    weekly=pd.DataFrame({'c':g.ac,'h':g.ah,'l':g.al,'key':keys}).groupby('key').agg(
        c=('c','last'),h=('h','max'),l=('l','min'))
    previous=weekly.c.shift()
    tr=pd.concat([weekly.h-weekly.l,(weekly.h-previous).abs(),(weekly.l-previous).abs()],axis=1).max(axis=1)
    tr=tr.where(weekly.c.notna() & previous.notna())
    drawdown=1-weekly.c/weekly.h.rolling(13).max()
    contraction=tr/tr.shift().rolling(4).mean().replace(0,np.nan)
    dd=pd.Series(keys.map(drawdown.shift()),index=g.index,dtype=float)
    ratio=pd.Series(keys.map(contraction.shift()),index=g.index,dtype=float)
    breakout=g.ac.gt(g.ah.shift().rolling(3).max())
    # 突破由假转真；连续创新高不重复触发。
    first=breakout & ~breakout.shift(1,fill_value=False)
    signal=dd.ge(.10) & ratio.lt(1.0) & first
    return pd.DataFrame({'setup_drawdown':dd,'setup_contraction':ratio,'turn_signal':signal},index=g.index)


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
        ma10 = c.rolling(10).mean()
        g["structure"] = g.al.shift().rolling(10).min()
        g["ret20"] = c / c.shift(20) - 1
        g["contraction"] = tr.shift().rolling(5).mean() / tr.shift().rolling(20).mean()
        g["risk_distance"] = (c - g.structure) / c
        setup = contraction_turn_setup(g)
        g = g.join(setup)
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
        raw = g.turn_signal & eligible & g.atr.notna() & g.structure.gt(0) & g.ret20.notna()
        below_twice = c.lt(ma10) & c.shift().lt(ma10.shift())
        armed, last, events = True, -100, np.zeros(len(g), dtype=bool)
        for i in range(len(g)):
            if not armed and i - last >= 5 and bool(below_twice.iloc[i]):
                armed = True
            if armed and bool(raw.iloc[i]):
                events[i] = True
                armed, last = False, i
        g["signal"] = events
        g["source"] = "C"
        g = g.join(weekly_observers(g))
        # 只将缺失价前向传递用于估值，绝不用于信号/成交。
        g["mark"] = g.ac.ffill()
        g["last_quote"] = pd.Series(calendar, index=calendar).where(g.ac.notna()).ffill()
        g["ts_code"] = code
        g["name"] = base.loc[code, "name"]
        stocks[code] = g[["open", "ao", "ac", "ah", "al", "vol", "adj_factor", "up_limit", "down_limit",
                          "atr", "mark", "last_quote", "structure", "eligible", "ret20"]].copy()
        cols = ["ts_code", "name", "source", "close", "ac", "structure", "atr", "ret20", "contraction",
                "risk_distance", "circ_mv", "turnover_rate", "weekly_state", "weekly_k", "weekly_d", "weekly_macd", "setup_drawdown", "setup_contraction"]
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
        signals = signals.sort_values(["date", "ts_code"])
        signals["event_id"] = signals.date.dt.strftime("%Y%m%d") + "_" + signals.ts_code
        signals = signals[(signals.date >= stamp(cfg.start)) & (signals.date <= stamp(cfg.end))].reset_index(drop=True)
    else:
        for col in ("event_id",):
            signals[col] = pd.Series(dtype=str)
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


def path_for_signal(signal, g, calendar, exit_scheme="full"):
    """独立事件；不设资金总额、仓位数量或复投。"""
    if exit_scheme not in EXIT_SCHEMES:
        raise ValueError("未知退出方案")
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
            if exit_scheme != "none" and row.ac < stop and not pending:
                pending = "收盘跌破保护价"
            highest = max(highest, row.ac)
            if exit_scheme == "full" and np.isfinite(row.atr):
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


def build_events(signals, stocks, calendar, progress=lambda text: None, exit_scheme="full"):
    result = []
    for n, signal in enumerate(signals.to_dict("records"), 1):
        result.append(path_for_signal(signal, stocks[signal["ts_code"]], calendar, exit_scheme))
        if n % 100 == 0 or n == len(signals):
            progress(f"{EXIT_SCHEMES[exit_scheme]}：事件审计 {n}/{len(signals)}")
    if result:
        return pd.DataFrame(result)
    columns = list(signals.columns) + ["entry_status", "buy_date", "sell_date", "buy_idx", "sell_idx",
                                      "net_return", "reason", "delayed_days"]
    columns += [f"W{w}_{suffix}" for w in (1, 2, 4, 8) for suffix in ("net_pct", "status")]
    return pd.DataFrame(columns=columns)


def exit_diagnostics(event_sets, stocks, calendar, progress=lambda text: None):
    """同一买点、同一买价、同一观察时点配对；未成熟事件不因提前退出而提前纳入。"""
    full = event_sets["full"]
    indexed = {key: frame.set_index("event_id") for key, frame in event_sets.items()}
    rows = []
    for n, event in enumerate(full.to_dict("records"), 1):
        if event["entry_status"] != "已成交":
            continue
        code, event_id = event["ts_code"], event["event_id"]
        j = int(event["buy_idx"])
        g = stocks[code]
        other = {key: frame.loc[event_id] for key, frame in indexed.items()}
        for variant in other.values():
            if variant.entry_status != "已成交" or variant.buy_idx != j or not np.isclose(variant.buy_price, event["buy_price"]):
                raise RuntimeError("退出对照入场不一致，停止生成配对结果")
        for week in (1, 2, 4, 8):
            target = j + week * 5 - 1
            row = {"event_id": event_id, "date": event["date"], "year": stamp(event["date"]).year,
                "ts_code": code, "source": event["source"],
                "week": week, "buy_date": event["buy_date"], "buy_price": event["buy_price"],
                "mature": target < len(calendar), "target_date": calendar[target] if target < len(calendar) else pd.NaT,
                "path_complete": False, "fixed_net_pct": np.nan, "mfe_gross_pct": np.nan,
                "mae_gross_pct": np.nan, "full_exited_early": False}
            for key, variant in other.items():
                row[key + "_net_pct"] = variant.get(f"W{week}_net_pct", np.nan)
            if target < len(calendar):
                path = g.iloc[j:target+1]
                row["path_complete"] = bool(path[["ac", "ah", "al"]].notna().all().all())
                if np.isfinite(g.ac.iloc[target]):
                    row["fixed_net_pct"] = (g.ac.iloc[target] * .999 * .998 / (event["buy_price"] * 1.001) - 1) * 100
                if row["path_complete"]:
                    row["mfe_gross_pct"] = (path.ah.max() / event["buy_price"] - 1) * 100
                    row["mae_gross_pct"] = (path.al.min() / event["buy_price"] - 1) * 100
                row["full_exited_early"] = 0 <= event["sell_idx"] <= target and event["reason"] != "满8周"
            row["paired"] = bool(row["mature"] and row["path_complete"] and
                all(np.isfinite(row[key]) for key in ("fixed_net_pct", "none_net_pct", "initial_net_pct", "full_net_pct")))
            rows.append(row)
        if n % 200 == 0:
            progress(f"固定窗口价格路径与退出配对 {n}/{len(full)}")
    details = pd.DataFrame(rows)
    if details.empty:
        return {"exit_diagnostic": pd.DataFrame(), "exit_pairs": details}
    summaries = []
    for year in ["全部"] + sorted(details.year.unique().tolist()):
        annual = details if year == "全部" else details[details.year == year]
        for source in ("全部", "C"):
            part = annual if source == "全部" else annual[annual.source == source]
            for rank in ("全部",):
                subset = part
                for week, group in subset.groupby("week"):
                    paired = group[group.paired]
                    early = paired[paired.full_exited_early]
                    row = {"年度": str(year), "来源": source, "事件范围": rank, "观察交易日": int(week)*5,
                        "可成交事件": len(group), "未成熟": int((~group.mature).sum()),
                        "成熟但不完整": int((group.mature & ~group.paired).sum()), "配对样本": len(paired)}
                    for key, label in (("fixed", "固定观察"), ("none", "仅8周到期"), ("initial", "初始保护"), ("full", "完整退出")):
                        values = paired[key + "_net_pct"]
                        row[label + "均收益%"] = values.mean()
                        row[label + "中位数%"] = values.median()
                        row[label + "胜率%"] = values.gt(0).mean()*100 if len(values) else np.nan
                    row.update({"仅到期减固定观察_百分点": (paired.none_net_pct-paired.fixed_net_pct).mean(),
                        "初始保护减仅到期_百分点": (paired.initial_net_pct-paired.none_net_pct).mean(),
                        "完整减初始保护_百分点": (paired.full_net_pct-paired.initial_net_pct).mean(),
                        "最高浮盈中位数%": paired.mfe_gross_pct.median(), "最大浮亏中位数%": paired.mae_gross_pct.median(),
                        "完整规则提前退出数": len(early),
                        "提前退出后期末更高比例%": (early.fixed_net_pct>early.full_net_pct).mean()*100 if len(early) else np.nan,
                        "提前退出后期末差值均值_百分点": (early.fixed_net_pct-early.full_net_pct).mean()})
                    summaries.append(row)
    return {"exit_diagnostic": pd.DataFrame(summaries), "exit_pairs": details}








def longest_true(values):
    longest = running = 0
    for value in values:
        running = running + 1 if value else 0
        longest = max(longest, running)
    return longest












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






def latest_ready_day():
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    # 18点之前不请求未完成日线；周末/节假日由交易日历排除。
    return pd.Timestamp(now.date() - timedelta(days=1 if now.hour < 18 else 0))


def fixed_pool_outcomes(stocks, signals, calendar, progress=lambda text: None):
    """同一信号日的全部时点合格科技股；先确定抽样母体，再计算未来结果。"""
    dates = pd.DatetimeIndex(signals.date.unique())
    parts = []
    for n, (code, g) in enumerate(stocks.items(), 1):
        eligible = (g.eligible & g.atr.notna() & g.structure.gt(0) & g.ret20.notna()
                    & g.index.isin(dates))
        idx = np.flatnonzero(eligible.to_numpy())
        if not len(idx):
            continue
        # 入场门槛与正式信号一致，仅去除C形态要求。
        next_open, next_ao = g.open.shift(-1), g.ao.shift(-1)
        next_up, next_down = g.up_limit.shift(-1), g.down_limit.shift(-1)
        valid = (next_open.notna() & next_ao.notna() & next_up.notna() & next_down.notna()
                 & g.vol.shift(-1).gt(0) & next_open.gt(0))
        gap = next_ao.gt(g.ac*1.05)
        broken = next_ao.le(g.structure)
        limited = next_open.ge(next_up-.005)
        filled = valid & ~gap & ~broken & ~limited
        status = np.select([~valid, limited, gap, broken],
            ["缺报价/停牌，取消", "开盘涨停，取消", "高开超过5%，取消", "跌破形态低点，取消"], default="已成交")
        status = pd.Series(status, index=g.index)
        status.iloc[-1] = "区间末待成交"
        buy_price = np.minimum(next_open*1.001, next_up)*g.adj_factor.shift(-1)
        part = pd.DataFrame({"date": g.index[idx], "ts_code": code, "entry_status": status.iloc[idx].to_numpy(),
                             "filled": filled.iloc[idx].to_numpy(), "buy_price": buy_price.iloc[idx].to_numpy()})
        for week in (1, 2, 4, 8):
            # 信号日i，买入日i+1，第5*w个持有交易日为i+5*w。
            mature = idx + week*5 < len(calendar)
            target_close = g.ac.shift(-week*5).iloc[idx].to_numpy()
            returns = (target_close*.999*.998/(buy_price.iloc[idx].to_numpy()*1.001)-1)*100
            # 随机选择发生于信号日；不因次日无法成交换一只股票，取消单收益为0。
            returns = np.where(part.filled, returns, 0.0)
            returns = np.where(mature, returns, np.nan)
            part[f"W{week}_mature"] = mature
            part[f"W{week}_net_pct"] = returns
        parts.append(part)
        if n % 50 == 0:
            progress(f"全池等概率入场参照 {n}/{len(stocks)} 只；固定5/10/20/40交易日观察")
    columns = ["date", "ts_code", "entry_status", "filled", "buy_price"]
    for week in (1,2,4,8):
        columns += [f"W{week}_mature", f"W{week}_net_pct"]
    return (pd.concat(parts, ignore_index=True).sort_values(["date", "ts_code"]).reset_index(drop=True)
            if parts else pd.DataFrame(columns=columns))


def group_masks(frame):
    for year in ["全部"] + sorted(frame.date.dt.year.unique().tolist()):
        yearly = frame if year == "全部" else frame[frame.date.dt.year == year]
        for source in ("全部", "C"):
            part = yearly if source == "全部" else yearly[yearly.source == source]
            for ranking in ("全部",):
                subset = part
                yield str(year), source, ranking, subset


def entry_edge_reports(signals, pool):
    """完整日期严格配对；未知未来值不填0、不凭已知结果换股。"""
    if signals.empty:
        return {name:pd.DataFrame() for name in ("entry_outcomes","entry_summary","pool_daily_reference","edge_daily","edge_summary","edge_bound_daily","edge_bound_summary")}
    extra = pool.drop(columns=["buy_price"])
    outcomes = signals.merge(extra, on=["date", "ts_code"], how="left", validate="one_to_one")
    if outcomes.entry_status.isna().any():
        raise RuntimeError("部分正式信号不在同日对照母体，停止比较")
    pool_rows=[]
    for week in (1,2,4,8):
        val, mat = f"W{week}_net_pct", f"W{week}_mature"
        for day, g in pool.groupby('date',sort=True):
            mature=bool(g[mat].all()); unknown=int(g[val].isna().sum()) if mature else 0
            pool_rows.append({"date":day,"week":week,"pool_size":len(g),"filled":int(g.filled.sum()),
                "mature":mature,"unknown":unknown,"complete":mature and unknown==0,
                "known_subset_mean_pct":g[val].mean(),
                "expected_lower_bound_pct":(g[val].sum()-100*unknown)/len(g) if mature else np.nan,
                "uniform_expected_pct":g[val].mean() if mature and unknown==0 else np.nan})
    pool_daily=pd.DataFrame(pool_rows)
    summaries, comparisons, daily_rows=[],[],[]
    for year,source,ranking,subset in group_masks(outcomes):
        for week in (1,2,4,8):
            val,mat=f"W{week}_net_pct",f"W{week}_mature"
            mature=subset[subset[mat]]
            known=mature[val].dropna(); executed=mature.loc[mature.filled,val].dropna()
            summaries.append({"年度":year,"来源":source,"事件范围":ranking,"观察交易日":week*5,
                "发单事件数":len(subset),"可成交数":int(subset.filled.sum()),"未成熟数":int((~subset[mat]).sum()),
                "成熟但收益未知数":int(mature[val].isna().sum()),"已知结果数":len(known),
                "每次发单均收益%":known.mean(),"每次发单中位数%":known.median(),
                "实际成交均收益%":executed.mean(),"实际成交中位数%":executed.median(),
                "实际成交胜率%":executed.gt(0).mean()*100 if len(executed) else np.nan,
                "实际成交10分位收益%":executed.quantile(.1) if len(executed) else np.nan})
            ref=pool_daily[pool_daily.week==week].set_index('date')
            paired=[]; immature=0; incomplete=0
            for day,g in subset.groupby('date',sort=True):
                r=ref.loc[day]
                if not r.mature:
                    immature+=1;continue
                if not r.complete or g[val].isna().any():
                    incomplete+=1;continue
                selected=g[val].mean(); benchmark=r.uniform_expected_pct
                item={"年度":year,"来源":source,"事件范围":ranking,"date":day,"观察交易日":week*5,
                    "信号数量":len(g),"同日母体数量":r.pool_size,"信号每次发单均收益%":selected,
                    "同池等概率入场期望收益%":benchmark,"超额收益_百分点":selected-benchmark}
                paired.append(item)
                # 日明细只存真实年份，避免与“全部”分组重复。
                if year!="全部":daily_rows.append(item)
            p=pd.DataFrame(paired)
            comparisons.append({"年度":year,"来源":source,"事件范围":ranking,"观察交易日":week*5,
                "信号日期数":subset.date.nunique(),"未成熟日期":immature,"不完整日期":incomplete,
                "严格配对日期":len(p),
                "日期覆盖率%":len(p)/subset.date.nunique()*100 if subset.date.nunique() else np.nan,
                "信号日均收益%":p['信号每次发单均收益%'].mean() if len(p) else np.nan,
                "同池等概率日均收益%":p['同池等概率入场期望收益%'].mean() if len(p) else np.nan,
                "日均超额收益_百分点":p['超额收益_百分点'].mean() if len(p) else np.nan,
                "超额收益中位数_百分点":p['超额收益_百分点'].median() if len(p) else np.nan,
                "胜出日期比例%":p['超额收益_百分点'].gt(0).mean()*100 if len(p) else np.nan,
                "解释":"完整日期辅助复核，无显著性/实盘通过结论"})
    bound_rows, bound_summary=[],[]
    for year,source,ranking,subset in group_masks(outcomes):
        for week in (1,2,4,8):
            val=f'W{week}_net_pct';ref=pool_daily[pool_daily.week==week].set_index('date')
            rows=[];immature=0;signal_unknown=0
            for day,g in subset.groupby('date',sort=True):
                r=ref.loc[day]
                if not r.mature:immature+=1;continue
                if g[val].isna().any():signal_unknown+=1;continue
                selected=g[val].mean()
                rows.append({'date':day,'年度':year,'来源':source,'事件范围':ranking,'观察交易日':week*5,
                    '信号日均收益%':selected,'已知对照部分均收益%':r.known_subset_mean_pct,
                    '对照收益下界%':r.expected_lower_bound_pct,'超额收益上界_百分点':selected-r.expected_lower_bound_pct,
                    '母体未知比例%':r.unknown/r.pool_size*100,'母体数量':r.pool_size,'母体未知数':r.unknown})
            b=pd.DataFrame(rows)
            upper=b['超额收益上界_百分点'].mean() if len(b) else np.nan
            bound_summary.append({'年度':year,'来源':source,'事件范围':ranking,'观察交易日':week*5,
                '信号日期数':subset.date.nunique(),'未成熟日期':immature,'信号收益未知日期':signal_unknown,
                '边界可比较日期':len(b),'对照未知比例_日均%':b['母体未知比例%'].mean() if len(b) else np.nan,
                '信号日均收益%':b['信号日均收益%'].mean() if len(b) else np.nan,
                '对照日均收益下界%':b['对照收益下界%'].mean() if len(b) else np.nan,
                '日均超额收益上界_百分点':upper,
                '说明':'无可比较样本' if not len(b) else ('在可比较样本上，上界仍不为正' if upper<=0 else '上界为正不能证明优势，需看完整对照及绝对收益')})
            if year!='全部':bound_rows.extend(rows)
    return {"entry_outcomes":outcomes,"entry_summary":pd.DataFrame(summaries),
            "pool_daily_reference":pool_daily,"edge_daily":pd.DataFrame(daily_rows),"edge_summary":pd.DataFrame(comparisons),
            "edge_bound_daily":pd.DataFrame(bound_rows),"edge_bound_summary":pd.DataFrame(bound_summary)}


def signal_coverage(signals, events, cov, calendar, cfg, issues):
    dates=calendar[(calendar>=stamp(cfg.start))&(calendar<=stamp(cfg.end))]
    daily=pd.DataFrame(index=dates)
    daily['new_signals']=signals.groupby('date').size().reindex(dates,fill_value=0)
    filled=events[events.entry_status=='已成交']
    daily['executable_events']=filled.groupby('date').size().reindex(dates,fill_value=0)
    daily['pool_count']=cov.pool_count.reindex(dates,fill_value=0)
    daily['missing']=daily.index.isin(pd.to_datetime(issues.date.unique())) if len(issues) else False
    daily['week']=dates.to_period('W-SUN')
    weekly=daily.groupby('week').agg(first_date=('new_signals',lambda s:s.index.min()),
        last_date=('new_signals',lambda s:s.index.max()),new_signals=('new_signals','sum'),
        executable_events=('executable_events','sum'),min_pool=('pool_count','min'),missing_days=('missing','sum')).reset_index()
    unique=signals.groupby(signals.date.dt.to_period('W-SUN')).ts_code.nunique()
    weekly['unique_stocks']=weekly.week.map(unique).fillna(0).astype(int)
    weekly['year']=weekly.first_date.dt.year
    weekly['no_new_signal']=weekly.new_signals.eq(0)
    weekly['no_executable_signal']=weekly.executable_events.eq(0)
    annual=[]
    for year,w in weekly.groupby('year'):
        full=(stamp(cfg.start)<=pd.Timestamp(year=int(year),month=1,day=1)
              and stamp(cfg.end)>=pd.Timestamp(year=int(year),month=12,day=31)
              and latest_ready_day()>=pd.Timestamp(year=int(year),month=12,day=31))
        annual.append({"年度":year,"观察周数":len(w),"新信号数":w.new_signals.sum(),
            "无新信号周":int(w.no_new_signal.sum()),"无可成交信号周":int(w.no_executable_signal.sum()),
            "最长连续无新信号周":longest_true(w.no_new_signal),"仅1只新股周":int(w.unique_stocks.eq(1).sum()),
            "覆盖目标":"部分年度" if not full else ('数据受限' if len(issues) else ('达到五周目标' if w.no_new_signal.sum()<=5 else '未达到五周目标'))})
    weekly['week']=weekly.week.astype(str)
    return weekly,pd.DataFrame(annual)


STUDY_NOTES = """T2.0 收缩转强假设——尚未验证盈利能力
保留科技池、股价高于10元、流通市值50—1000亿元；全部事件独立研究，无资金与仓位模拟。
停用旧A/B买点和评分。只有C类：上一完成周收盘较该周及此前共13周最高价回撤至少10%；
该完成周真实波幅小于此前4周平均真实波幅；日收盘首次突破此前3日最高价。
参数10%、13周、4周和3日为本轮预先固定假设，不是回测寻优结果；不因覆盖不足临时放宽。
周线条件只用上一完成周，当前周最终高低收盘不会进入当前周买点。
沿用独立事件去重：触发后至少5日且连续两日收于10日均线下才重新待机。
研究目标仍为1—8周；固定观察5/10/20/40交易日，不转为3—20日持有系统。
全部C事件纳入，没有第一名或前三名筛选。MACD、SKDJ仅为记录项，不作买入门槛。
同池参照在同一日期、同一时点科技池和价格市值条件下选取，只去除C形态要求，不根据未来选股。
随机入场期望使用全池等概率均值，不进行随机种子或样本挑选。
双方统一次日开盘成交及费用，高开超过5%、开盘涨停、跌破此前10日低点或缺报价则取消，不补选。
取消单发单收益为0，但必须观察期届满才纳入；已成交目标日收益未知不填0。
固定窗口收盘仅为估值，预扣退出成本，不保证能按该价格卖出；无最低收费模拟。
主对照采用收益边界：对照收益下界=(已知收益之和-100×未知数量)/全池数量。
信号收益全部已知的日期才计算：信号日均收益-对照下界=超额收益上界；之后对日期等权。
无杠杆、按比例费用口径下-100%为持有资产归零的边界，不是未知收益估计。
上界为正不能证明优势；上界为负只描述可比较日期。信号收益未知的日期也可能造成选择偏差。
完整日期精确对照作为辅助表；少量对照未知不再从主表删除整日。
退出诊断保留三种独立路径：仅8自然周到期、固定初始保护、沿用完整退出；不做仓位配置。
最高浮盈/最大浮亏是事后极值，不是可实现收益；8自然周与40交易日不完全相同。
所有年度单列收益、超额、覆盖和缺失计数。重叠窗口与同日事件相关，不当作独立样本证明显著性。
历史已多次观察，任何正结果仍需冻结后的前向验证。沿用四路下载和已有缓存。
"""


def make_zip(tables,manifest):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,frame in tables.items():
            z.writestr(name+'.csv',frame.to_csv(index=False).encode('utf-8-sig'))
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
        z.writestr('规则与口径.txt',STUDY_NOTES+'\n'+'\n'.join(f'{k}：{v}' for k,v in RULES.items()))
    return buffer.getvalue()


def run_research(token,cache_root,cfg,progress):
    client=DataClient(token,cache_root,progress)
    basic,member,mode,pool_warnings=client.universe()
    ready=latest_ready_day();effective_end=min(stamp(cfg.end),ready)
    if effective_end<stamp(cfg.start):raise RuntimeError('尚未进入指定观察区间')
    history_start=ds(stamp(cfg.start)-pd.Timedelta(days=450))
    data_end=ds(min(ready,effective_end+pd.Timedelta(days=85)))
    calendar=client.calendar(history_start,data_end)
    study_dates=calendar[(calendar>=stamp(cfg.start))&(calendar<=effective_end)]
    if not len(study_dates):raise RuntimeError('所选区间没有交易日')
    progress(f'科技候选共{len(basic)}只，读取{len(calendar)}个交易日')
    data,issues=client.download(calendar,set(basic.ts_code))
    hashes=pd.util.hash_pandas_object(data,index=False).to_numpy(copy=True);hashes.sort()
    data_hash=hashlib.sha256(hashes.tobytes()).hexdigest();del hashes
    stocks,signals,cov=build_features(data,basic,member,calendar,cfg,progress)
    del data;gc.collect()
    pool=fixed_pool_outcomes(stocks,signals,calendar,progress)
    tables=entry_edge_reports(signals,pool)
    event_sets={s:build_events(signals,stocks,calendar,progress,exit_scheme=s) for s in EXIT_SCHEMES}
    events=event_sets['full']
    tables.update(exit_diagnostics(event_sets,stocks,calendar,progress))
    weeks,years=signal_coverage(signals,events,cov,calendar,cfg,issues)
    if pool_warnings or '历史行业区间' not in mode:years['覆盖目标']='股票池受限'
    # 验证固定窗口模块与旧事件模块的入场完全一致。
    if not signals.empty:
        check=tables['entry_outcomes'].set_index('event_id').filled
        expected=events.set_index('event_id').entry_status.eq('已成交')
        if not check.reindex(expected.index).equals(expected.rename('filled')):
            raise RuntimeError('固定窗口与退出审计的成交状态不一致')
    manifest={'version':VERSION,'baseline_version':BASELINE_VERSION,'config':asdict(cfg),'rules':RULES,
        'study_mode':'全部C类独立事件，不评分、不配置资金与仓位','hypothesis':{'drawdown':0.10,'weekly_peak_window':13,'weekly_contraction_reference':4,'daily_breakout_window':3},'control':'同日全池入场期望；对照未知收益使用下界，完整日期精确对照辅助',
        'download_revision':DOWNLOAD_REVISION,'download_workers':DOWNLOAD_WORKERS,'exit_schemes':EXIT_SCHEMES,
        'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'pool_mode':mode,'warnings':pool_warnings,
        'data_issues':len(issues),'universe_size':len(basic),'actual_signal_start':study_dates.min(),
        'actual_signal_end':study_dates.max(),'data_start':history_start,'data_end':data_end,'data_hash':data_hash,
        'pool_hash':hashlib.sha256(member.to_csv(index=False).encode()).hexdigest(),
        'limitations':['历史行业供应商记录未经独立核验','创业/科创历史风险警示未完全重建',
                       '复权总收益和按比例费用估算','完整日期子样本可能存在选择偏差','历史已多次观察，不作为未见样本证明']}
    tables.update({'signals':signals,'all_events':events,'none_events':event_sets['none'],
        'initial_events':event_sets['initial'],'weekly_coverage':weeks,'annual_coverage':years,
        'observers':observer_report(events),'data_issues':issues,
        'universe':basic,'industry_intervals':member,'daily_pool':cov.reset_index(),'pool_entry_outcomes':pool})
    zipped=make_zip(tables,manifest)
    run_id=hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()[:12]
    path=Path(cache_root)/'results'/f'{VERSION}_{run_id}.zip';atomic_bytes(zipped,path)
    return tables,manifest,zipped,str(path)


def filtered_table(st,frame,key):
    if frame.empty:
        st.info('此分组暂无可计算结果。');return
    cols=st.columns(3)
    y=cols[0].selectbox('年度',['全部']+sorted(x for x in frame['年度'].unique() if x!='全部'),key=key+'_year')
    source=cols[1].selectbox('互斥来源',['全部','C'],key=key+'_source')
    rank=cols[2].selectbox('事件范围',['全部'],key=key+'_rank')
    st.dataframe(frame[(frame['年度']==y)&(frame['来源']==source)&(frame['事件范围']==rank)],use_container_width=True,hide_index=True)


def show_results(st,tables,manifest,zipped):
    st.subheader(f"{manifest['version']} · 独立买点验证")
    st.caption('显示上次完成结果；修改设置后需重新运行。')
    events=tables['all_events'];metrics=st.columns(3)
    metrics[0].metric('独立新信号',len(tables['signals']))
    metrics[1].metric('可成交事件',int(events.entry_status.eq('已成交').sum()))
    metrics[2].metric('科技候选股票数',manifest['universe_size'])
    st.info('所有买点独立观察。先检查扣费后收益，再检查相对同日科技池的收益差；目前不自动判定策略有效。')
    if manifest['warnings'] or manifest['data_issues']:
        st.warning('数据受限：'+'；'.join(manifest['warnings'])+f"；异常记录{manifest['data_issues']}条")
    cfg=manifest['config']
    st.download_button('下载完整研究结果（ZIP）',zipped,file_name=f"tech_swing_T2_0_{cfg['start']}_{cfg['end']}.zip",mime='application/zip')
    tabs=st.tabs(['独立买点收益','同池入场对照','覆盖与信号','退出诊断','数据与规则'])
    with tabs[0]:
        st.caption('固定5/10/20/40交易日收盘估值；已扣比例费用和滑点。取消单的发单收益为0，实际成交收益另列。未成熟和未知不填0。')
        filtered_table(st,tables['entry_summary'],'entry')
    with tabs[1]:
        st.caption('主表保留对照池含未知结果的日期，将未知对照按-100%计算收益下界，得到有利于信号的超额收益上界。上界为正不等于存在优势；信号自身收益未知仍单列排除。')
        filtered_table(st,tables['edge_bound_summary'],'bound')
        with st.expander('完整日期精确对照（辅助复核）'):
            filtered_table(st,tables['edge_summary'],'edge')
        st.caption('同日平均后对日期等权，不将重叠事件当成独立统计样本；正超额也不等于绝对盈利。')
    with tabs[2]:
        st.dataframe(tables['annual_coverage'],use_container_width=True,hide_index=True)
        st.dataframe(tables['weekly_coverage'],use_container_width=True,hide_index=True)
        st.dataframe(tables['signals'].tail(200),use_container_width=True,hide_index=True)
    with tabs[3]:
        st.caption('保持同一批买点：固定价格观察、仅8自然周到期、初始保护、原完整退出。路径极值仅作事后诊断。')
        filtered_table(st,tables['exit_diagnostic'],'exit')
    with tabs[4]:
        st.text(STUDY_NOTES)
        st.dataframe(tables['data_issues'],use_container_width=True,hide_index=True)
        st.json(manifest)


def main():
    import streamlit as st
    st.set_page_config(page_title='科技波段 T2.0 收缩转强假设',layout='wide')
    st.title('科技波段 T2.0 · 收缩转强假设')
    st.write('全信号独立观察 · 同日科技池入场对照 · 四路并发下载')
    try:token_default=str(st.secrets.get('TUSHARE_TOKEN',st.secrets.get('tushare_token','')))
    except Exception:token_default=''
    with st.sidebar:
        token=st.text_input('Tushare Token',value=os.environ.get('TUSHARE_TOKEN',token_default),type='password')
        start=st.date_input('信号开始',value=date(2022,1,1))
        end=st.date_input('信号结束',value=latest_ready_day().date())
        price=st.number_input('最低股价（高于，元）',value=10.,min_value=0.,step=1.)
        low=st.number_input('最低流通市值（亿元）',value=50.,min_value=0.,step=10.)
        high=st.number_input('最高流通市值（亿元）',value=1000.,min_value=1.,step=100.)
        cache_root=st.text_input('数据缓存目录',value='tech_swing_cache')
        st.caption('沿用已有行情缓存，四路并发补缺。新假设参数冻结，不计算评分。')
        run=st.button('运行独立买点研究',type='primary',use_container_width=True)
    with st.expander('冻结规则与研究口径'):
        st.dataframe(pd.DataFrame(RULES.items(),columns=['项目','规则']),use_container_width=True,hide_index=True)
        st.text(STUDY_NOTES)
    if run:
        if not token.strip():st.error('请输入Token，或设置TUSHARE_TOKEN。')
        elif start>end or low>=high or not cache_root.strip():st.error('请检查日期、市值上下限和缓存目录。')
        else:
            box=st.empty();last=[0.]
            def progress(message):
                if time.monotonic()-last[0]>.25:box.info(message);last[0]=time.monotonic()
            cfg=Config(start=ds(start),end=ds(end),min_price=price,min_mv=low,max_mv=high)
            try:
                st.session_state['t20_result']=run_research(token,cache_root.strip(),cfg,progress)
                box.success('研究完成，结果已保留，可下载明细。')
            except Exception as exc:
                box.error('运行未完成：'+str(exc).replace(token,'[隐藏]')[:500])
                st.info('已下载成功的数据保留，修复后继续补缺。')
    if 't20_result' in st.session_state:
        tables,manifest,zipped,_=st.session_state['t20_result'];show_results(st,tables,manifest,zipped)
    else:st.info('建议先使用与上轮相同的日期区间，核对信号数量后再分析收益和同池对照。')


def self_test():
    import unittest
    class EventTests(unittest.TestCase):
        def setUp(self):
            self.cal=pd.bdate_range('2022-01-03',periods=90)
            self.g=pd.DataFrame(index=self.cal)
            for col in ('open','ao','ac','mark'):self.g[col]=20.
            self.g['ah']=20.5;self.g['al']=19.5;self.g['vol']=1000.
            self.g['adj_factor']=1.;self.g['up_limit']=22.;self.g['down_limit']=18.
            self.g['atr']=1.;self.g['structure']=18.5;self.g['ret20']=0.;self.g['eligible']=True
            self.signal={'date':self.cal[0],'ts_code':'600001.SH','name':'合成股票','ac':20.,
                'structure':18.5,'rank':1,'source':'C','event_id':'test'}

        def test_fixed_dates_and_fees(self):
            signals=pd.DataFrame([self.signal]);pool=fixed_pool_outcomes({'600001.SH':self.g},signals,self.cal)
            expected=(20*.999*.998/(20*1.001*1.001)-1)*100
            self.assertAlmostEqual(pool.W1_net_pct.iloc[0],expected)
            self.g.loc[self.cal[5],'ac']=25.
            pool=fixed_pool_outcomes({'600001.SH':self.g},signals,self.cal)
            self.assertAlmostEqual(pool.W1_net_pct.iloc[0],(25*.999*.998/(20*1.001*1.001)-1)*100)

        def test_cancel_no_replacement_and_unknown(self):
            g=self.g.copy();g.loc[self.cal[1],['open','ao']]=22.
            signals=pd.DataFrame([self.signal]);pool=fixed_pool_outcomes({'600001.SH':g},signals,self.cal)
            self.assertFalse(pool.filled.iloc[0]);self.assertEqual(pool.W8_net_pct.iloc[0],0.)
            g=self.g.copy();g.loc[self.cal[40],'ac']=np.nan
            pool=fixed_pool_outcomes({'600001.SH':g},signals,self.cal)
            self.assertTrue(np.isnan(pool.W8_net_pct.iloc[0]))
            reports=entry_edge_reports(signals,pool)
            row=reports['edge_summary'].query("年度=='全部' and 来源=='全部' and 事件范围=='全部' and 观察交易日==40").iloc[0]
            self.assertEqual(row['不完整日期'],1);self.assertEqual(row['严格配对日期'],0)

        def test_future_pool_membership_not_used(self):
            g=self.g.copy();g['eligible']=False;g.loc[self.cal[10]:,'eligible']=True
            signals=pd.DataFrame([self.signal]);pool=fixed_pool_outcomes({'600001.SH':self.g,'600002.SH':g},signals,self.cal)
            self.assertEqual(pool.ts_code.tolist(),['600001.SH'])

        def test_uniform_reference_hand_calculation(self):
            signals=pd.DataFrame([self.signal]);other=self.g.copy();other.loc[self.cal[5],'ac']=24.
            pool=fixed_pool_outcomes({'600001.SH':self.g,'600002.SH':other},signals,self.cal)
            r=entry_edge_reports(signals,pool)
            daily=r['pool_daily_reference'].query('week==1').iloc[0]
            self.assertAlmostEqual(daily.uniform_expected_pct,pool.W1_net_pct.mean())
            edge=r['edge_summary'].query("年度=='全部' and 来源=='全部' and 事件范围=='全部' and 观察交易日==5").iloc[0]
            self.assertAlmostEqual(edge['日均超额收益_百分点'],pool.W1_net_pct.iloc[0]-pool.W1_net_pct.mean())

        def test_exit_modes_and_t_plus_one(self):
            g=self.g.copy();g.loc[self.cal[1],'ac']=17.
            e=path_for_signal(self.signal,g,self.cal,'full')
            self.assertEqual(e['sell_idx'],2)
            none=path_for_signal(self.signal,g,self.cal,'none')
            self.assertEqual(none['sell_date'],self.cal[1]+pd.Timedelta(weeks=8))

        def test_unmatured_even_when_cancelled(self):
            g=self.g.iloc[:4].copy();g.loc[self.cal[1],['open','ao']]=22.
            p=fixed_pool_outcomes({'600001.SH':g},pd.DataFrame([self.signal]),self.cal[:4])
            self.assertFalse(p.W1_mature.iloc[0]);self.assertTrue(np.isnan(p.W1_net_pct.iloc[0]))

        def test_weekly_setup_has_no_future_data(self):
            cal=pd.bdate_range('2022-01-03',periods=105)
            c=np.full(len(cal),20.);c[65:]=16.
            g=pd.DataFrame({'ac':c,'ah':c+.5,'al':c-.5},index=cal)
            g.loc[cal[85:90],'ah']=16.1;g.loc[cal[85:90],'al']=15.9
            g.loc[cal[90],['ac','ah','al']]=[16.3,16.4,16.0]
            setup=contraction_turn_setup(g)
            self.assertTrue(setup.turn_signal.iloc[90])
            pd.testing.assert_frame_equal(setup.iloc[:91],contraction_turn_setup(g.iloc[:91]))
            g.loc[cal[91:],'ah']=1000.;g.loc[cal[91:],'ac']=500.
            changed=contraction_turn_setup(g)
            pd.testing.assert_frame_equal(setup.iloc[:91],changed.iloc[:91])

        def test_unknown_control_bounds_keep_date(self):
            signals=pd.DataFrame([self.signal]);other=self.g.copy();other.loc[self.cal[40],'ac']=np.nan
            p=fixed_pool_outcomes({'600001.SH':self.g,'600002.SH':other},signals,self.cal)
            reports=entry_edge_reports(signals,p)
            r=reports['edge_bound_summary'].query("年度=='全部' and 来源=='全部' and 观察交易日==40").iloc[0]
            own=p[p.ts_code=='600001.SH'].W8_net_pct.iloc[0]
            self.assertEqual(r['边界可比较日期'],1)
            self.assertAlmostEqual(r['日均超额收益上界_百分点'],own-(own-100)/2)
            self.assertEqual(r['对照未知比例_日均%'],50.)

        def test_config_and_exports_have_no_allocation(self):
            self.assertNotIn('capital',asdict(Config()));self.assertNotIn('slots',asdict(Config()))
            self.assertNotIn('portfolio',globals())
            self.assertTrue(zipfile.is_zipfile(io.BytesIO(make_zip({'signals':pd.DataFrame([self.signal])},{'test':True}))))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EventTests))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':
    if '--self-test' in sys.argv:self_test()
    else:main()



