# -*- coding: utf-8 -*-
"""科技波段研究 T3.0 日线SKDJ等待确认 — streamlit run app.py

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

VERSION = "T3.0-DAILY-SKDJ-20260910"
WEEK_FILTERS = ("不过滤", "金叉或K拐头", "仅金叉")
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
@dataclass(frozen=True)
class Config:
    start: str = "20220101"
    end: str = "20260909"
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


RULES = {
    '股票池':'历史科技行业；不复权股价>10元，流通市值50—1000亿元（侧栏可调整）',
    '日线':'N=6、9；RSV→EMA3→EMA3(K)→MA3(D)。缺报价后重新预热，避免跨缺口交叉',
    '低位金叉':'前日K≤D，当日K>D且K、D均<25；金叉当日为D0',
    '确认':'D1/D2/D3开盘入场分别使用D0/D1/D2收盘信息；观察期K一直>D，确认日K上升',
    '比较':'D2、D3为主组；D1是不等待的辅助对照。每个N×买入日×周线规则独立研究',
    '排序':'同一组、同一确认日，K−D降序；精确相同则当日流通市值降序，再按代码；取消单不补名次',
    '周线':'统一N=9、平滑3；使用已完成周线，周五收盘纳入本周，其他交易日使用上一日历周',
    '周线组':'不过滤；金叉或K拐头（K>D或K较前周上升）；仅金叉（K>D）。10周不作为门槛',
    '入场':'下一交易日开盘；停牌、缺开盘/复权/限制价、开盘涨停时取消；不沿用C形态低点或5%高开过滤',
    '退出':'对应日线N收盘死叉（K<D且前日K≥D）后下一交易日开盘卖；买入日第1天，第30交易日开盘到期',
    '延迟':'退出指令一经触发不撤销；停牌、缺报价或开盘跌停则等待，实际可超过30交易日',
    '成本':'买入费用0.10%、卖出费用0.20%，每边滑点0.10%；无最低收费模拟',
    '事件':'同一金叉每组最多一笔；各组独立，不是多次加仓；不设资金总额或三仓',
    '范围':'侧栏日期限定D0金叉日；确认、买入和退出可在区间结束之后，行情仅读取已完成日期',
    '数据':'历史行业及风险警示重建仍有供应商限制；持有期间缺指标数据使退出路径不可确定，单列不计闭合收益',
}
STUDY_NOTES = '\n'.join(f'{k}：{v}' for k,v in RULES.items()) + '''
EMA递推系数为0.5，以第一笔有效RSV初始化；每段连续数据前N+5笔不发指标，至少N+6笔才可用。
金叉与确认时均检查股票池资格。已知资格不符是未发单；数据不足是未知，不冒充零收益。
日期等权和排名超额仅使用同组当日所有已发单结果均已确定的日期，可能存在成熟度选择偏差。
同金叉D2/D3配对同时展示两组都成交、早组成交晚组未成交、早组未成交晚组成交和都未成交。
跳过的交易有独立计数；未确认、待买入、尚未退出、数据未知不作为亏损或零收益进入胜率。
基准是同日全部合格日线信号，不是整个科技池随机选股；本版不能自动证明市场超额能力。
排名第1/2/3及前三名均为独立事件统计，不是有限资金组合收益。不同N不混排。
同股重复金叉的独立事件可能重叠；分组和持有窗口相关，不把事件数当作独立统计样本。
周线死叉周龄仅诊断，按交易周数计；周线K拐头不等于价格反转，不自动挑选表现最优参数。
所有年份均为已经观察过的历史，正收益还需要冻结后的前向验证。
'''


def latest_ready_day():
    now=datetime.now(ZoneInfo('Asia/Shanghai'))
    return pd.Timestamp(now.date()-timedelta(days=1 if now.hour<18 else 0))


def skdj(frame, n):
    """RSV两次EMA3，K再MA3；每段连续有效行情独立预热。"""
    good=frame[['ac','ah','al']].notna().all(axis=1) & frame.ah.ge(frame.al) & frame.ac.gt(0)
    output=pd.DataFrame(np.nan,index=frame.index,columns=['k','d'])
    groups=(~good).cumsum()
    for _, g in frame[good].groupby(groups[good],sort=False):
        lo=g.al.rolling(n).min();hi=g.ah.rolling(n).max()
        rsv=100*(g.ac-lo)/(hi-lo).replace(0,np.nan)
        # 完整但连续N期同价，取中性50；前N-1期仍未知。
        rsv=rsv.mask(hi.eq(lo)&hi.notna(),50.)
        slow=rsv.ewm(span=3,adjust=False,min_periods=1).mean()
        k=slow.ewm(span=3,adjust=False,min_periods=1).mean()
        ready=np.arange(len(g))>=n+5
        output.loc[g.index,'k']=k.where(ready)
        output.loc[g.index,'d']=k.rolling(3).mean().where(ready)
    return output


def completed_weekly(g):
    keys=g.index.to_period('W-FRI')
    valid=g[['ac','ah','al']].notna().all(axis=1)
    w=g.assign(key=keys,valid=valid).groupby('key').agg(ac=('ac','last'),ah=('ah','max'),al=('al','min'),valid=('valid','all'))
    w.loc[~w.valid,['ac','ah','al']]=np.nan
    kd=skdj(w,9);w=w.join(kd)
    w['k_prev']=w.k.shift();w['death']=w.k.lt(w.d)&w.k.shift().ge(w.d.shift())
    age=[];last=None
    for i,row in enumerate(w.itertuples()):
        if pd.isna(row.k) or pd.isna(row.d):last=None
        elif row.death:last=i
        elif row.k>row.d:last=None
        age.append(i-last if last is not None else np.nan)
    w['death_age']=age
    cols=['k','d','k_prev','death_age']
    out=pd.DataFrame(index=g.index)
    friday=g.index.weekday==4
    for col in cols:
        current=pd.Series(keys.map(w[col]),index=g.index,dtype=float)
        previous=pd.Series(keys.map(w[col].shift()),index=g.index,dtype=float)
        out['w_'+col]=previous.where(~friday,current)
    return out


def eligibility(g, code, info, intervals, calendar, cfg):
    active=np.zeros(len(calendar),dtype=bool)
    for m in intervals.itertuples():
        active|=(calendar>=m.in_date)&(calendar<(m.out_date if pd.notna(m.out_date) else pd.Timestamp.max))
    active&=calendar>=stamp(info.list_date)+pd.Timedelta(days=180)
    if pd.notna(info.delist_date):active&=calendar<pd.to_datetime(info.delist_date)
    st_like=(g.up_limit/g.pre_close-1).lt(.07) if code.startswith(('60','00')) else pd.Series(False,index=g.index)
    known=g[['close','circ_mv','vol','adj_factor','up_limit','down_limit','pre_close']].notna().all(axis=1)
    eligible=active & known & g.close.gt(cfg.min_price)&(g.circ_mv/10000).between(cfg.min_mv,cfg.max_mv)&g.vol.gt(0)&g.adj_factor.gt(0)&~st_like
    return eligible,known


def exit_path(g, kd, buy_i, death=None):
    """一个可执行买单；各周线组复用相同路径。"""
    cal=g.index
    result=dict(filled=False,closed=False,resolved=False,entry_status='待买入',exit_reason='',
        buy_date=pd.NaT,sell_date=pd.NaT,buy_adj=np.nan,sell_adj=np.nan,net_pct=np.nan,
        order_net_pct=np.nan,hold_days=np.nan,overdue_days=0,mfe_pct=np.nan,mae_pct=np.nan)
    if buy_i>=len(g):return result
    row=g.iloc[buy_i]
    vals=[row.open,row.adj_factor,row.up_limit,row.down_limit,row.vol]
    if not np.isfinite(vals).all() or min(row.open,row.adj_factor)<=0 or row.vol<=0:
        result.update(entry_status='缺报价或停牌取消',resolved=True,order_net_pct=0.);return result
    if row.open>=row.up_limit-.005:
        result.update(entry_status='开盘涨停取消',resolved=True,order_net_pct=0.);return result
    buy=min(row.open*1.001,row.up_limit)*row.adj_factor
    result.update(filled=True,entry_status='已成交',buy_date=cal[buy_i],buy_adj=buy)
    deadline=buy_i+29
    ka,da=kd.k.to_numpy(),kd.d.to_numpy()
    if death is None:death=((kd.k<kd.d)&(kd.k.shift()>=kd.d.shift())).to_numpy()
    request=None;reason='';unknown=False
    for j in range(buy_i,min(len(g),deadline+1)):
        # 开盘到期优先于当日收盘指标；买入当日收盘死叉最早次日卖。
        if j==deadline:request=j;reason='30交易日到期';break
        if np.isnan(ka[j]) or np.isnan(da[j]):unknown=True;break
        if death[j]:request=j+1;reason='日线死叉';break
    if unknown:
        result.update(exit_reason='持有期间指标缺失，退出时点未知');return result
    if request is None:
        result.update(exit_reason='未触发退出，观察未结束');return result
    result['exit_reason']=reason
    for j in range(request,len(g)):
        row=g.iloc[j]
        if not np.isfinite([row.open,row.adj_factor,row.down_limit,row.vol]).all() or min(row.open,row.adj_factor)<=0 or row.vol<=0:continue
        if row.open<=row.down_limit+.005:continue
        sell=max(row.open*.999,row.down_limit)*row.adj_factor
        net=(sell*.998/(buy*1.001)-1)*100
        # 卖出日在开盘退出，不使用之后的盘中高低价。
        highs=list(g.ah.iloc[buy_i:j].dropna())+[sell]
        lows=list(g.al.iloc[buy_i:j].dropna())+[sell]
        complete=g[['ah','al']].iloc[buy_i:j].notna().all().all()
        result.update(closed=True,resolved=True,sell_date=cal[j],sell_adj=sell,net_pct=net,order_net_pct=net,
            hold_days=j-buy_i+1,overdue_days=max(0,j-deadline),
            mfe_pct=(max(highs)/buy-1)*100 if complete else np.nan,
            mae_pct=(min(lows)/buy-1)*100 if complete else np.nan)
        return result
    result['exit_reason']=reason+'，待可执行卖出'
    return result


def evaluate_stock(g, code, name, eligible, known, cfg):
    rows=[];cal=g.index;weekly=completed_weekly(g)
    for n in (6,9):
        kd=skdj(g,n)
        death=(kd.k.lt(kd.d)&kd.k.shift().ge(kd.d.shift())).to_numpy()
        gold=kd.k.gt(kd.d)&kd.k.shift().le(kd.d.shift())&kd.k.lt(25)&kd.d.lt(25)
        idx=np.flatnonzero((gold&eligible&(cal>=stamp(cfg.start))&(cal<=stamp(cfg.end))).to_numpy())
        for i in idx:
            cohort=f'{n}_{ds(cal[i])}_{code}'
            for delay in (1,2,3):
                confirm=i+delay-1;buy_i=i+delay
                base=dict(cohort_id=cohort,ts_code=code,name=name,n=n,delay=delay,gold_date=cal[i],year=str(cal[i].year),
                    signal_date=cal[confirm] if confirm<len(g) else pd.NaT,
                    planned_buy_date=cal[buy_i] if buy_i<len(g) else pd.NaT,
                    gold_k=kd.k.iloc[i],gold_d=kd.d.iloc[i],score=np.nan,k=np.nan,d=np.nan,k_change=np.nan,
                    gap_change=np.nan,circ_mv_yi=np.nan,w_k=np.nan,w_d=np.nan,w_k_change=np.nan,w_death_age=np.nan,
                    weekly_state='未知',qualified=False,filled=False,closed=False,resolved=False,
                    entry_status='待确认',exit_reason='',buy_date=pd.NaT,sell_date=pd.NaT,buy_adj=np.nan,
                    sell_adj=np.nan,net_pct=np.nan,order_net_pct=np.nan,hold_days=np.nan,overdue_days=0,mfe_pct=np.nan,mae_pct=np.nan)
                if confirm<len(g):
                    k,d=kd.iloc[confirm];prev=kd.iloc[confirm-1] if confirm else pd.Series({'k':np.nan,'d':np.nan})
                    w=weekly.iloc[confirm]
                    base.update(k=k,d=d,score=k-d,k_change=k-prev.k,gap_change=k-d-(prev.k-prev.d),
                        circ_mv_yi=g.circ_mv.iloc[confirm]/10000,w_k=w.w_k,w_d=w.w_d,
                        w_k_change=w.w_k-w.w_k_prev,w_death_age=w.w_death_age)
                    wok=np.isfinite([w.w_k,w.w_d,w.w_k_prev]).all()
                    if wok:
                        base['weekly_state']='金叉' if w.w_k>w.w_d else ('K拐头' if w.w_k>w.w_k_prev else 'K未拐头')
                    continuous=kd.iloc[i:confirm+1]
                    if not known.iloc[confirm] or continuous.isna().any().any():
                        base['entry_status']='确认数据不足'
                    elif not eligible.iloc[confirm]:
                        base.update(entry_status='确认时股票池不合格',resolved=True,order_net_pct=0.)
                    elif not continuous.k.gt(continuous.d).all():
                        base.update(entry_status='等待期间金叉失效',resolved=True,order_net_pct=0.)
                    elif not k>prev.k:
                        base.update(entry_status='确认日K未上升',resolved=True,order_net_pct=0.)
                    else:
                        base.update(qualified=True)
                        base.update(exit_path(g,kd,buy_i,death))
                for wf in WEEK_FILTERS:
                    r=base.copy();r['weekly_filter']=wf
                    if r['qualified'] and wf!='不过滤':
                        wk,wd,change=r['w_k'],r['w_d'],r['w_k_change']
                        valid=np.isfinite([wk,wd,change]).all()
                        allow=valid and (wk>wd or (wf=='金叉或K拐头' and change>0))
                        if not allow:
                            r.update(qualified=False,filled=False,closed=False,resolved=valid,
                                entry_status='周线过滤剔除' if valid else '周线数据不足',exit_reason='',buy_date=pd.NaT,sell_date=pd.NaT,
                                buy_adj=np.nan,sell_adj=np.nan,net_pct=np.nan,order_net_pct=0. if valid else np.nan,
                                hold_days=np.nan,overdue_days=0,mfe_pct=np.nan,mae_pct=np.nan)
                    rows.append(r)
    return rows


def calculate_events(data,basic,member,calendar,cfg,progress):
    all_rows=[];base=basic.set_index('ts_code');members={c:m for c,m in member.groupby('ts_code')}
    for number,(code,part) in enumerate(data.groupby('ts_code',sort=True),1):
        if code not in base.index or code not in members:continue
        g=part.drop_duplicates('date').set_index('date').sort_index().reindex(calendar)
        for c in ['open','high','low','close','pre_close','vol','circ_mv','turnover_rate','adj_factor','up_limit','down_limit']:
            g[c]=pd.to_numeric(g[c],errors='coerce')
        for c,a in [('open','ao'),('high','ah'),('low','al'),('close','ac')]:g[a]=g[c]*g.adj_factor
        eligible,known=eligibility(g,code,base.loc[code],members[code],calendar,cfg)
        all_rows.extend(evaluate_stock(g,code,base.loc[code,'name'],eligible,known,cfg))
        if number%25==0:progress(f'SKDJ逐股研究 {number}/{len(base)}；路径每个N和买入日只计算一次')
    events=pd.DataFrame(all_rows)
    if events.empty:return events
    group=['n','delay','weekly_filter','signal_date']
    events['rank']=np.nan
    order=events[events.qualified].sort_values(group+['score','circ_mv_yi','ts_code'],ascending=[True]*4+[False,False,True])
    events.loc[order.index,'rank']=order.groupby(group).cumcount()+1
    return events.sort_values(['gold_date','ts_code','n','delay','weekly_filter']).reset_index(drop=True)


def rank_parts(e):
    yield '全部',e
    for rank in (1,2,3):yield f'第{rank}名',e[e['rank'].eq(rank)]
    yield '前三名',e[e['rank'].le(3)]
    yield '其余名次',e[e['rank'].gt(3)]


def build_reports(events,calendar,cfg):
    keys=['n','delay','weekly_filter']
    names=['events','summary','ranking_daily','ranking_summary','delay_pairs','delay_summary','weekly_diagnostic','annual_coverage','status_counts']
    tables={name:pd.DataFrame() for name in names};tables['events']=events
    if events.empty:return tables
    summary=[];ranking_daily=[];ranking_summary=[];coverage=[];diagnostic=[]
    status=events.groupby(keys+['entry_status'],dropna=False).size().rename('count').reset_index()
    tables['status_counts']=status
    for group,e in events.groupby(keys,sort=True):
        meta=dict(zip(keys,group));issued=e[e.qualified]
        agg=issued.groupby('signal_date').order_net_pct.agg(['size','count','mean'])
        valid_dates=agg.index[agg['size'].eq(agg['count'])]
        for year in ['全部']+sorted(e.year.unique()):
            y=e if year=='全部' else e[e.year.eq(year)]
            for label,s in rank_parts(y):
                filled=s[s.filled];done=s[s.closed];q=s[s.qualified]
                day=q.groupby('signal_date').order_net_pct.agg(['size','count','mean'])
                day=day[day.index.isin(valid_dates)]
                summary.append(dict(meta,year=year,rank_group=label,cohorts=len(s),orders=len(q),filled=len(filled),closed=len(done),
                    unresolved=int((~s.resolved).sum()),mean_net_pct=done.net_pct.mean(),median_net_pct=done.net_pct.median(),
                    win_pct=done.net_pct.gt(0).mean()*100 if len(done) else np.nan,
                    mean_hold_days=done.hold_days.mean(),p10_net_pct=done.net_pct.quantile(.1),
                    mean_mfe_pct=done.mfe_pct.mean(),mean_mae_pct=done.mae_pct.mean(),
                    mean_order_net_pct=q.order_net_pct.mean(),comparable_dates=len(day),day_equal_net_pct=day['mean'].mean(),
                    timeout_exits=int(done.exit_reason.eq('30交易日到期').sum()),overdue_exits=int(done.overdue_days.gt(0).sum())))
                differences=[]
                for date_,d in day.iterrows():
                    edge=d['mean']-agg.loc[date_,'mean'];differences.append(edge)
                    if year!='全部':ranking_daily.append(dict(meta,year=year,rank_group=label,signal_date=date_,
                        order_mean_pct=d['mean'],all_signal_mean_pct=agg.loc[date_,'mean'],edge_pp=edge))
                ranking_summary.append(dict(meta,year=year,rank_group=label,dates=len(differences),
                    edge_vs_all_signals_pp=np.mean(differences) if differences else np.nan))
        # 覆盖按确认信号日期归年；开始/结束之外的确认信号不计入区间覆盖。
        obs=calendar[(calendar>=stamp(cfg.start))&(calendar<=stamp(cfg.end))]
        for year in sorted(obs.year.unique()):
            dates=obs[obs.year==year];weeks=dates.to_period('W-SUN').unique()
            q=issued[issued.signal_date.isin(dates)];f=q[q.filled]
            signal_weeks=q.signal_date.dt.to_period('W-SUN').unique();fill_weeks=f.signal_date.dt.to_period('W-SUN').unique()
            complete=stamp(cfg.start)<=pd.Timestamp(year=year,month=1,day=1) and stamp(cfg.end)>=pd.Timestamp(year=year,month=12,day=31) and latest_ready_day()>=pd.Timestamp(year=year,month=12,day=31)
            coverage.append(dict(meta,year=str(year),observed_weeks=len(weeks),no_signal_weeks=len(weeks.difference(signal_weeks)),
                no_filled_signal_weeks=len(weeks.difference(fill_weeks)),orders=len(q),filled=len(f),full_year=complete))
        if group[2]=='不过滤':
            for (year,state),part in e[e.qualified].groupby(['year','weekly_state']):
                done=part[part.closed]
                diagnostic.append(dict(meta,year=year,weekly_state=state,orders=len(part),closed=len(done),
                    mean_net_pct=done.net_pct.mean(),win_pct=done.net_pct.gt(0).mean()*100 if len(done) else np.nan,
                    median_death_age=part.w_death_age.median() if part.w_death_age.notna().any() else np.nan))
    pairs=[];pair_summary=[]
    for (n,wf),e in events.groupby(['n','weekly_filter']):
        columns=['cohort_id','gold_date','year','qualified','filled','closed','resolved','entry_status','order_net_pct','net_pct','buy_adj','rank']
        a=e[e.delay.eq(2)][columns];b=e[e.delay.eq(3)][columns]
        p=a.merge(b,on=['cohort_id','gold_date','year'],suffixes=('_D2','_D3'),validate='one_to_one')
        p['n']=n;p['weekly_filter']=wf
        p['both_closed']=p.closed_D2&p.closed_D3
        p['net_difference_pp']=(p.net_pct_D3-p.net_pct_D2).where(p.both_closed)
        p['buy_price_change_pct']=(p.buy_adj_D3/p.buy_adj_D2-1)*100
        p['both_resolved']=p.resolved_D2&p.resolved_D3
        p['policy_difference_pp']=(p.order_net_pct_D3-p.order_net_pct_D2).where(p.both_resolved)
        pairs.append(p)
        for year in ['全部']+sorted(p.year.unique()):
            s=p if year=='全部' else p[p.year.eq(year)];known=s[s.both_resolved];both=s[s.both_closed]
            early=known[known.filled_D2&~known.filled_D3];late=known[~known.filled_D2&known.filled_D3]
            pair_summary.append(dict(n=n,weekly_filter=wf,year=year,cohorts=len(s),resolved_pairs=len(known),both_closed=len(both),
                D2_mean_net_pct=both.net_pct_D2.mean(),D3_mean_net_pct=both.net_pct_D3.mean(),
                D2_win_pct=both.net_pct_D2.gt(0).mean()*100 if len(both) else np.nan,
                D3_win_pct=both.net_pct_D3.gt(0).mean()*100 if len(both) else np.nan,
                D3_minus_D2_pp=both.net_difference_pp.mean(),mean_buy_price_change_pct=both.buy_price_change_pct.mean(),
                early_only=len(early),early_only_losers=int(early.net_pct_D2.lt(0).sum()),early_only_winners=int(early.net_pct_D2.gt(0).sum()),
                late_only=len(late),neither_filled=int((~known.filled_D2&~known.filled_D3).sum()),
                resolved_policy_difference_pp=known.policy_difference_pp.mean()))
    tables.update(summary=pd.DataFrame(summary),ranking_daily=pd.DataFrame(ranking_daily),ranking_summary=pd.DataFrame(ranking_summary),
        delay_pairs=pd.concat(pairs,ignore_index=True),delay_summary=pd.DataFrame(pair_summary),
        weekly_diagnostic=pd.DataFrame(diagnostic),annual_coverage=pd.DataFrame(coverage))
    return tables


def make_zip(tables,manifest):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,frame in tables.items():z.writestr(name+'.csv',frame.to_csv(index=False).encode('utf-8-sig'))
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
        z.writestr('规则与口径.txt',STUDY_NOTES)
    return buffer.getvalue()


def run_research(token,cache_root,cfg,progress):
    client=DataClient(token,cache_root,progress)
    basic,member,mode,pool_warnings=client.universe()
    ready=latest_ready_day();effective_end=min(stamp(cfg.end),ready)
    if effective_end<stamp(cfg.start):raise RuntimeError('尚未进入指定区间')
    start=ds(stamp(cfg.start)-pd.Timedelta(days=450));end=ds(min(ready,effective_end+pd.Timedelta(days=85)))
    calendar=client.calendar(start,end)
    data,issues=client.download(calendar,set(basic.ts_code))
    hashed=pd.util.hash_pandas_object(data,index=False).to_numpy();hashed.sort();data_hash=hashlib.sha256(hashed.tobytes()).hexdigest()
    events=calculate_events(data,basic,member,calendar,cfg,progress)
    del data;gc.collect()
    progress('汇总收益、排名、同一金叉延迟配对和年度覆盖')
    tables=build_reports(events,calendar,cfg)
    tables.update(data_issues=issues,universe=basic,industry_intervals=member)
    manifest=dict(version=VERSION,config=asdict(cfg),rules=RULES,created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        data_start=start,data_end=end,data_hash=data_hash,pool_hash=hashlib.sha256(member.to_csv(index=False).encode()).hexdigest(),
        pool_mode=mode,warnings=pool_warnings,data_issues=len(issues),universe_size=len(basic),download_workers=DOWNLOAD_WORKERS,
        study='N6/N9 × D1/D2/D3 × 三种周线规则；D1仅辅助；无资金组合',
        limitations=['重复观察历史，非未见样本证明','排名对照为同日全部信号，不是整个科技池',
        '真实周线只使用完成周；周中不使用暂态周线','历史行业完整性及风险警示重建仍有限制',
        '持有路径数据缺失及尚未卖出不计闭合收益，存在完整样本选择风险','复权收益近似公司行动，未逐笔重建税费'])
    zipped=make_zip(tables,manifest)
    run_id=hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()[:12]
    path=Path(cache_root)/'results'/f'{VERSION}_{run_id}.zip';atomic_bytes(zipped,path)
    return tables,manifest,zipped,str(path)


LABELS={'n':'日线N','delay':'金叉后第几交易日买入','weekly_filter':'周线过滤','year':'金叉年度','rank_group':'名次范围',
'cohorts':'金叉事件数','orders':'发单数','filled':'成交数','closed':'已闭合数','unresolved':'结果未确定数',
'mean_net_pct':'平均净收益%','median_net_pct':'净收益中位数%','win_pct':'胜率%',
'mean_hold_days':'平均持有交易日','p10_net_pct':'收益10分位%','mean_mfe_pct':'平均最高浮盈%',
'mean_mae_pct':'平均最大浮亏%','mean_order_net_pct':'已知发单平均净收益%','comparable_dates':'完整结果日期数',
'day_equal_net_pct':'日期等权发单净收益%','timeout_exits':'到期退出数','overdue_exits':'超30日退出数',
'dates':'比较日期数','edge_vs_all_signals_pp':'相对同日全部信号收益差_百分点',
'resolved_pairs':'两组结果均确定','both_closed':'两组都成交并闭合','D2_mean_net_pct':'D2配对平均净收益%',
'D3_mean_net_pct':'D3配对平均净收益%','D2_win_pct':'D2配对胜率%','D3_win_pct':'D3配对胜率%',
'D3_minus_D2_pp':'D3减D2配对收益_百分点','mean_buy_price_change_pct':'D3买价相对D2变化%',
'early_only':'仅D2成交','early_only_losers':'D3避开的D2亏损笔数','early_only_winners':'D3错过的D2盈利笔数',
'late_only':'仅D3成交','neither_filled':'两组均未成交','resolved_policy_difference_pp':'含确定未买入的D3减D2_百分点',
'observed_weeks':'观察周数','no_signal_weeks':'无新信号周','no_filled_signal_weeks':'无可成交新信号周',
'full_year':'完整年度','weekly_state':'周线状态','median_death_age':'死叉周龄中位数','entry_status':'状态','count':'数量'}


def show_results(st,tables,manifest,zipped):
    cfg=manifest['config'];st.subheader(manifest['version'])
    st.download_button('下载完整研究结果（ZIP）',zipped,file_name=f"tech_swing_T3_0_{cfg['start']}_{cfg['end']}.zip",mime='application/zip')
    st.caption('保留上次完成结果；更改侧栏后需重新运行。D1仅为不等待对照；各测试组独立，不能累加成组合收益。')
    if manifest['warnings'] or manifest['data_issues']:st.warning('数据受限：'+'；'.join(manifest['warnings'])+f"；问题记录{manifest['data_issues']}条")
    if tables['events'].empty:st.info('该区间没有符合股票池要求的低位金叉。');return
    years=['全部']+sorted(tables['events'].year.unique())
    year=st.selectbox('金叉年度',years,key='result_year')
    rank=st.selectbox('名次范围',['全部','第1名','第2名','第3名','前三名','其余名次'],key='result_rank')
    tabs=st.tabs(['收益与胜率','同金叉D2/D3比较','排序是否有效','周线与覆盖','事件明细','规则与数据'])
    def selected(name):
        f=tables[name]
        if 'year' in f:f=f[f.year.eq(year)]
        if 'rank_group' in f:f=f[f.rank_group.eq(rank)]
        return f.rename(columns=LABELS)
    with tabs[0]:
        st.caption('先看已闭合数及未确定数，再比较收益。未卖出和路径未知不当作零收益。按金叉年度归组。')
        st.dataframe(selected('summary'),use_container_width=True,hide_index=True)
    with tabs[1]:
        st.caption('此表固定全部金叉，不按排名筛选。两组都闭合的比较之外，单列第三天避开的亏损及错过的盈利。未确定事件不填0。')
        st.dataframe(selected('delay_summary'),use_container_width=True,hide_index=True)
    with tabs[2]:
        st.caption('同组、同日全部已发单信号作为参照；只用整日结果确定的日期。不是科技全池超额收益。')
        st.dataframe(selected('ranking_summary'),use_container_width=True,hide_index=True)
    with tabs[3]:
        f=tables['annual_coverage'];f=f if year=='全部' else f[f.year.eq(year)]
        st.caption('覆盖按确认日期归年；无信号周不是资金空仓周。部分年度和数据受限时不判断全年目标。')
        st.dataframe(f.rename(columns=LABELS),use_container_width=True,hide_index=True)
        f=tables['weekly_diagnostic'];f=f if year=='全部' or f.empty else f[f.year.eq(year)]
        st.caption('仅不过滤组的周线状态诊断；不使用10周放行。')
        st.dataframe(f.rename(columns=LABELS),use_container_width=True,hide_index=True)
    with tabs[4]:
        st.dataframe(tables['status_counts'].rename(columns=LABELS),use_container_width=True,hide_index=True)
        st.dataframe(tables['events'].tail(300),use_container_width=True,hide_index=True)
    with tabs[5]:st.text(STUDY_NOTES);st.dataframe(tables['data_issues'],use_container_width=True);st.json(manifest)


def main():
    import streamlit as st
    st.set_page_config(page_title='科技日线SKDJ T3.0',layout='wide')
    st.title('科技日线SKDJ T3.0 · 等待确认研究')
    try:default=str(st.secrets.get('TUSHARE_TOKEN',st.secrets.get('tushare_token','')))
    except Exception:default=''
    with st.sidebar:
        token=st.text_input('Tushare Token',value=os.environ.get('TUSHARE_TOKEN',default),type='password')
        start=st.date_input('低位金叉开始日',value=date(2022,1,1));end=st.date_input('低位金叉结束日',value=latest_ready_day().date())
        price=st.number_input('最低股价（高于，元）',value=10.,min_value=0.,step=1.)
        low=st.number_input('最低流通市值（亿元）',value=50.,min_value=0.,step=10.)
        high=st.number_input('最高流通市值（亿元）',value=1000.,min_value=1.,step=100.)
        root=st.text_input('数据缓存目录',value='tech_swing_cache')
        st.caption('沿用T1/T2缓存，四路并发补缺；全部18组一次计算，无资金配置。')
        run=st.button('运行SKDJ验证',type='primary')
    with st.expander('固定规则'):st.text(STUDY_NOTES)
    if run:
        if not token.strip():st.error('请输入Token或设置TUSHARE_TOKEN。')
        elif start>end or low>=high or not root.strip():st.error('请检查日期、市值范围及缓存目录。')
        else:
            box=st.empty();last=[0.]
            def progress(msg):
                if time.monotonic()-last[0]>.25:box.info(msg);last[0]=time.monotonic()
            try:
                st.session_state['t30_result']=run_research(token,root.strip(),Config(ds(start),ds(end),price,low,high),progress)
                box.success('研究完成，结果已保留，可下载。')
            except Exception as exc:box.error(str(exc).replace(token,'[隐藏]')[:500]);st.info('成功下载的数据已缓存，修复后可继续补缺。')
    if 't30_result' in st.session_state:
        tables,manifest,zipped,_=st.session_state['t30_result'];show_results(st,tables,manifest,zipped)


def self_test():
    import unittest
    from unittest.mock import patch
    class Tests(unittest.TestCase):
        def setUp(self):
            self.cal=pd.bdate_range('2022-01-03',periods=150)
            self.g=pd.DataFrame(dict(open=20.,high=21.,low=19.,close=20.,ac=20.,ah=21.,al=19.,ao=20.,
                pre_close=20.,adj_factor=1.,up_limit=22.,down_limit=18.,vol=1000.,circ_mv=2000000.,turnover_rate=1.),index=self.cal)
            self.kd=pd.DataFrame(dict(k=30.,d=20.),index=self.cal)
        def test_skdj_flat_and_missing(self):
            kd=skdj(self.g,9)
            self.assertAlmostEqual(kd.k.dropna().iloc[-1],50.)
            self.assertAlmostEqual(kd.d.dropna().iloc[-1],50.)
            self.g.loc[self.cal[60],['ac','ah','al']]=np.nan
            changed=skdj(self.g,9)
            self.assertTrue(changed.iloc[60:69].isna().all().all())
            self.assertTrue(changed.iloc[-1].notna().all())
        def test_skdj_recursive_reference(self):
            self.g['ac']=20+.5*np.sin(np.arange(len(self.g))/3)
            n=6;slow=None;prior=None;ks=[];expected={}
            for i in range(n-1,len(self.g)):
                rsv=100*(self.g.ac.iloc[i]-19)/2
                slow=rsv if slow is None else .5*rsv+.5*slow
                prior=slow if prior is None else .5*slow+.5*prior
                ks.append(prior)
                if i>=n+5:expected[i]=(prior,sum(ks[-3:])/3)
            actual=skdj(self.g,n)
            for i,(k,d) in expected.items():
                self.assertAlmostEqual(actual.k.iloc[i],k)
                self.assertAlmostEqual(actual.d.iloc[i],d)
        def test_indicator_and_weekly_causality(self):
            rng=np.random.default_rng(7);c=20*np.exp(np.cumsum(rng.normal(0,.025,len(self.g))))
            self.g['ac']=c;self.g['ah']=c+1;self.g['al']=c-1
            full=completed_weekly(self.g)
            for i in (102,104,109,123):
                pd.testing.assert_frame_equal(full.iloc[:i+1],completed_weekly(self.g.iloc[:i+1]))
                pd.testing.assert_frame_equal(skdj(self.g,6).iloc[:i+1],skdj(self.g.iloc[:i+1],6))
            i=102
            changed=self.g.copy();changed.loc[self.cal[i+1]:,['ac','ah','al']]*=5
            pd.testing.assert_frame_equal(full.iloc[:i+1],completed_weekly(changed).iloc[:i+1])
        def test_exit_next_open_and_costs(self):
            self.kd.loc[self.cal[12]:,'k']=19.
            result=exit_path(self.g,self.kd,10)
            self.assertEqual(result['sell_date'],self.cal[13])
            expected=(20*.999*.998/(20*1.001*1.001)-1)*100
            self.assertAlmostEqual(result['net_pct'],expected)
            self.assertEqual(result['hold_days'],4)
        def test_same_day_death_and_day30(self):
            self.kd.loc[self.cal[10]:,'k']=19.
            self.assertEqual(exit_path(self.g,self.kd,10)['sell_date'],self.cal[11])
            self.kd['k']=30.
            self.assertEqual(exit_path(self.g,self.kd,10)['sell_date'],self.cal[39])
        def test_limit_delay_persistent_and_no_intraday_lookahead(self):
            self.kd.loc[self.cal[12],'k']=19.
            self.g.loc[self.cal[13],['open','ao']]=18.
            r=exit_path(self.g,self.kd,10)
            self.assertEqual(r['sell_date'],self.cal[14])
            self.g.loc[self.cal[14],'ah']=1000.
            self.assertEqual(exit_path(self.g,self.kd,10)['mfe_pct'],r['mfe_pct'])
        def test_cancel_missing_and_pending(self):
            self.g.loc[self.cal[10],'open']=22.
            r=exit_path(self.g,self.kd,10)
            self.assertFalse(r['filled']);self.assertEqual(r['order_net_pct'],0.)
            self.g.loc[self.cal[10],'open']=20.
            self.kd.loc[self.cal[11],'k']=np.nan
            r=exit_path(self.g,self.kd,10)
            self.assertTrue(r['filled']);self.assertFalse(r['resolved']);self.assertTrue(np.isnan(r['net_pct']))
            r=exit_path(self.g,self.kd,len(self.g));self.assertFalse(r['resolved'])
        def test_waiting_days_and_week_filter(self):
            kd=pd.DataFrame(dict(k=19.,d=20.),index=self.cal)
            kd.loc[self.cal[40:44],'k']=[21.,22.,23.,24.]
            w=pd.DataFrame(dict(w_k=15.,w_d=20.,w_k_prev=16.,w_death_age=15.),index=self.cal)
            cfg=Config(ds(self.cal[40]),ds(self.cal[40]))
            with patch(__name__+'.skdj',return_value=kd),patch(__name__+'.completed_weekly',return_value=w):
                e=pd.DataFrame(evaluate_stock(self.g,'600001.SH','test',pd.Series(True,index=self.cal),pd.Series(True,index=self.cal),cfg))
            a=e[(e.n==6)&(e.weekly_filter=='不过滤')].set_index('delay')
            self.assertEqual(a.loc[2,'buy_date'],self.cal[42])
            self.assertEqual(a.loc[3,'buy_date'],self.cal[43])
            self.assertEqual(a.loc[2,'score'],2.)
            self.assertEqual(a.loc[3,'score'],3.)
            # 15周但K仍下降也不放行。
            self.assertFalse(e[e.weekly_filter!='不过滤'].qualified.any())
            kd.loc[self.cal[42],'k']=19.
            with patch(__name__+'.skdj',return_value=kd),patch(__name__+'.completed_weekly',return_value=w):
                e=pd.DataFrame(evaluate_stock(self.g,'600001.SH','test',pd.Series(True,index=self.cal),pd.Series(True,index=self.cal),cfg))
            self.assertFalse(e[e.delay==3].qualified.any())
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':
    if '--self-test' in sys.argv:self_test()
    else:main()
