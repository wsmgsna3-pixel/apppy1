# -*- coding: utf-8 -*-
"""科技波段研究 T3.1 周线SKDJ上穿25形态 — streamlit run app.py

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

VERSION = "T3.1-WEEKLY-SHAPE-20260910"
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


# 分组边界只从先前年份的信号形态获得，不使用未来收益。
FEATURES = {
    'pre_speed':'上穿前两周平均K增量',
    'cross_speed':'上穿周K增量',
    'acceleration':'K增量的变化',
    'gap':'K减D',
    'gap_change':'K减D较前周扩大值',
}
MODES=('周收盘确认','周中逐日观察')
EXITS=('持有5日','持有10日','持有20日','持有40日','周线死叉或40日')
RULES={
    '研究问题':'周线N=6上穿25时，K斜率、K−D及其扩大速度能否区分后续收益；无资金组合、无三仓',
    '股票池':'历史科技行业；信号日不复权股价>10元，流通市值50—1000亿元；保留上市180日及风险警示近似排除',
    '指标':'沿用RSV(N=6)→EMA3→EMA3(K)→MA3(D)；不是屏幕角度；连续有效周至少12周预热',
    '基准事件':'前一完成周K≤25，本次K>25；先保留全部上穿，K>D与否单独诊断，不预先删除纠缠样本',
    '周收盘确认':'当周最后交易日收盘上穿，下一交易日开盘买；区间最后一个未确认完成周不作为周收盘信号',
    '周中逐日观察':'每天收盘以截至当日周OHLC计算暂态周线；对比上一完成周K；每股每周仅记录首次上穿，不事后要求周末仍站上25',
    '周中斜率':'本周暂态K减上一完成周K，尚非完整一周；按信号星期另作诊断；每日暂态值不重复参与周EMA',
    '特征':'上穿前两周平均增量=(K前1周−K前3周)/2；上穿周增量=K−K前1周；加速度=本次增量−前周增量；间距=K−D；间距变化=本次间距−前周间距',
    '分组':'每个观察模式、每年冻结边界：仅用该年前的合格信号形态计算1/3、2/3分位；至少100个历史事件；不足则只记基准并标记校准不足',
    '联合比较':'上穿周增量低中高 × K−D低中高，九格比较；另看间距扩大/持平/收窄，不自动选择最佳门槛或评分',
    '买入':'信号次交易日开盘；涨停或停牌取消；缺必要行情单列未知；取消不延后追买',
    '固定窗口':'第5/10/20/40个持有交易日收盘发出退出指令，下一交易日开盘卖；因此正常记录持有天数为6/11/21/41（含卖出日）',
    '形态退出':'对应观察模式周K<D且前一完成周K≥D，收盘触发后次交易日卖；周收盘模式只在完成周判断；未触发则第40日收盘发退出指令',
    '执行':'A股T+1；退出指令不撤销；已知开盘跌停或停牌等待可执行开盘；关键退出行情缺失则路径未知，不猜测可延迟成交',
    '相关性':'D为最近3周K均值，因此K−D=(2×本周K增量+前周K增量)/3；斜率和间距相关，联合表现好不能算两份独立证据',
    '成本':'买入费用0.10%、卖出费用0.20%；每边滑点0.10%，限制在涨跌停价内；不模拟最低收费',
    '统计':'净收益均值、中位数、胜率、10分位、去掉最好1%后的均值；最高浮盈为诊断，不等于可实现收益',
    '对照':'同模式同日全部合格上穿信号；日期等权差只用当日所有事件结果已确定的日期（取消单按0）；不是整个科技池超额',
    '未完成':'待买入、待卖出、缺数据均单列，不填0；同时导出逐年和逐事件结果，避免近期未完成样本冒充失败',
    '限制':'重复使用的历史不是未见样本证明；同股事件可重叠，各窗口相关；科技历史行业、风险警示、退市及公司行动数据仍有限制',
}
STUDY_NOTES='\n'.join(f'{k}：{v}' for k,v in RULES.items())


def weekly_shape(g):
    """O(日数)暂态周指标；每个日值仅使用当日及之前数据。"""
    key=g.index.to_period('W-FRI')
    good=g[['ac','ah','al']].notna().all(axis=1)&g.ac.gt(0)&g.ah.ge(g.al)
    x=g.assign(key=key,good=good)
    w=x.groupby('key').agg(ac=('ac','last'),ah=('ah','max'),al=('al','min'),good=('good','all'))
    w.loc[~w.good,['ac','ah','al']]=np.nan
    # 内部未遮罩状态用于计算当前暂态周，公开指标仍有预热门槛。
    w['slow']=np.nan;w['raw_k']=np.nan;w['count']=0
    seg=(~w.good).cumsum()
    for _,part in w[w.good].groupby(seg[w.good],sort=False):
        lo=part.al.rolling(6).min();hi=part.ah.rolling(6).max()
        r=100*(part.ac-lo)/(hi-lo).replace(0,np.nan);r=r.mask(hi.eq(lo)&hi.notna(),50.)
        slow=r.ewm(span=3,adjust=False).mean();raw=slow.ewm(span=3,adjust=False).mean()
        w.loc[part.index,'slow']=slow;w.loc[part.index,'raw_k']=raw
        w.loc[part.index,'count']=np.arange(len(part))+1
    w=w.join(skdj(w,6))
    def mapped(series):return pd.Series(key.map(series),index=g.index,dtype=float)
    prev_lo=mapped(w.al.rolling(5,min_periods=5).min().shift())
    prev_hi=mapped(w.ah.rolling(5,min_periods=5).max().shift())
    low=pd.concat([prev_lo,x.groupby('key').al.cummin()],axis=1).min(axis=1).where(prev_lo.notna())
    high=pd.concat([prev_hi,x.groupby('key').ah.cummax()],axis=1).max(axis=1).where(prev_hi.notna())
    current_good=x.groupby('key').good.cummin().astype(bool)
    r=100*(g.ac-low)/(high-low).replace(0,np.nan);r=r.mask(high.eq(low)&high.notna(),50.)
    ps=mapped(w.slow.shift());pk=mapped(w.raw_k.shift())
    slow=(r+ps)/2;slow=slow.where(ps.notna(),r)
    raw=(slow+pk)/2;raw=raw.where(pk.notna(),slow)
    d=(raw+pk+mapped(w.raw_k.shift(2)))/3
    count=mapped(w['count'].shift()).fillna(0)+1
    valid=current_good&count.ge(12)
    out=pd.DataFrame({'k':raw.where(valid),'d':d.where(valid)},index=g.index)
    out['k1']=mapped(w.k.shift());out['k2']=mapped(w.k.shift(2));out['k3']=mapped(w.k.shift(3));out['d1']=mapped(w.d.shift())
    out['pre_speed']=(out.k1-out.k3)/2
    out['cross_speed']=out.k-out.k1
    out['acceleration']=out.cross_speed-(out.k1-out.k2)
    out['gap']=out.k-out.d;out['gap_change']=out.gap-(out.k1-out.d1)
    # 下一周确实存在，或已到周五，才能把该行当成已完成周。
    last=np.r_[key[:-1]!=key[1:],g.index[-1].weekday()==4]
    out['complete']=last
    out['cross']=out.k.gt(25)&out.k1.le(25)
    out['death']=out.k.lt(out.d)&out.k1.ge(out.d1)
    out['week']=key.astype(str)
    return out


def trade_path(g,shape,buy_i,mode,exit_rule):
    result=dict(filled=False,closed=False,resolved=False,status='待买入',exit_reason='',
        buy_date=pd.NaT,sell_date=pd.NaT,buy_adj=np.nan,sell_adj=np.nan,net_pct=np.nan,
        order_net_pct=np.nan,hold_days=np.nan,delayed_days=0,mfe_pct=np.nan,mae_pct=np.nan)
    if buy_i>=len(g):return result
    row=g.iloc[buy_i]
    needed=[row.open,row.adj_factor,row.up_limit,row.down_limit,row.vol]
    if pd.notna(row.vol) and row.vol<=0:
        result.update(status='停牌或涨停取消',resolved=True,order_net_pct=0.);return result
    if not np.isfinite(needed).all():result['status']='买入数据未知';return result
    if row.vol<=0 or row.open<=0 or row.open>=row.up_limit-.005:
        result.update(status='停牌或涨停取消',resolved=True,order_net_pct=0.);return result
    if row.adj_factor<=0 or row.down_limit<=0 or row.up_limit<row.down_limit:
        result['status']='买入数据异常';return result
    buy=min(row.open*1.001,row.up_limit)*row.adj_factor
    result.update(filled=True,status='已成交未退出',buy_date=g.index[buy_i],buy_adj=buy)
    horizon=40 if exit_rule=='周线死叉或40日' else int(exit_rule[2:-1])
    deadline=buy_i+horizon # 收盘观察horizon日，再下一日开盘执行。
    request=deadline if deadline<len(g) else None;reason=f'{horizon}日窗口到期'
    if exit_rule=='周线死叉或40日':
        for j in range(buy_i,min(deadline,len(g))):
            if mode=='周收盘确认' and not shape.complete.iloc[j]:continue
            if shape[['k','d','k1','d1']].iloc[j].isna().any():
                result.update(status='形态退出路径未知',exit_reason='持有期周指标缺失');return result
            if shape.death.iloc[j]:request=j+1;reason='周线死叉';break
    if request is None:result['exit_reason']='窗口未结束';return result
    result['exit_reason']=reason
    for j in range(request,len(g)):
        row=g.iloc[j]
        if pd.notna(row.vol) and row.vol<=0:continue
        if not np.isfinite([row.open,row.adj_factor,row.down_limit,row.vol]).all():
            result['status']='退出数据未知';return result
        if row.vol<=0 or row.adj_factor<=0 or row.down_limit<=0 or row.open<=row.down_limit+.005:continue
        sell=max(row.open*.999,row.down_limit)*row.adj_factor
        net=(sell*.998/(buy*1.001)-1)*100
        held=g.iloc[buy_i:j];complete=held[['ah','al']].notna().all().all()
        result.update(closed=True,resolved=True,status='已闭合',sell_date=g.index[j],sell_adj=sell,net_pct=net,
            order_net_pct=net,hold_days=j-buy_i+1,delayed_days=j-request,
            mfe_pct=(max(held.ah.max(),sell)/buy-1)*100 if complete else np.nan,
            mae_pct=(min(held.al.min(),sell)/buy-1)*100 if complete else np.nan)
        return result
    result['status']='待可执行卖出';return result


def stock_events(g,code,name,eligible,known,cfg):
    shape=weekly_shape(g);rows=[]
    in_range=(g.index>=stamp(cfg.start))&(g.index<=stamp(cfg.end))
    for mode in MODES:
        trigger=shape.cross.copy()
        if mode=='周收盘确认':trigger&=shape.complete
        else:
            # 首次上穿先去重再检查资格，不能用同周后来成功/合格的信号替换首次。
            trigger&=trigger.groupby(shape.week).cumsum().eq(1)
        for i in np.flatnonzero(trigger&in_range):
            f=shape.iloc[i];day=g.index[i]
            base=dict(event_id=f'{code}|{mode}|{ds(day)}',ts_code=code,name=name,mode=mode,signal_date=day,
                year=str(day.year),weekday=day.weekday()+1,week=f.week,n=6,k=f.k,d=f.d,k1=f.k1,k2=f.k2,k3=f.k3,
                circ_mv_yi=g.circ_mv.iloc[i]/10000,qualified=bool(eligible.iloc[i]),qualification_known=bool(known.iloc[i]),
                kd_state='K>D' if f.k>f.d else 'K≤D',
                gap_direction='扩大' if f.gap_change>0 else ('收窄' if f.gap_change<0 else '持平'))
            base.update({c:float(f[c]) for c in FEATURES})
            for rule in EXITS:
                if eligible.iloc[i]:path=trade_path(g,shape,i+1,mode,rule)
                else:path=dict(filled=False,closed=False,resolved=False,status='资格不符' if known.iloc[i] else '资格未知',net_pct=np.nan,order_net_pct=np.nan)
                rows.append(dict(base,exit_rule=rule,**path))
    return rows


def calculate_events(data,basic,member,calendar,cfg,progress):
    rows=[];base=basic.set_index('ts_code');members={c:m for c,m in member.groupby('ts_code')}
    for number,(code,part) in enumerate(data.groupby('ts_code',sort=True),1):
        if code not in base.index or code not in members:continue
        g=part.drop_duplicates('date').set_index('date').sort_index().reindex(calendar)
        for c in ['open','high','low','close','pre_close','vol','circ_mv','turnover_rate','adj_factor','up_limit','down_limit']:
            g[c]=pd.to_numeric(g[c],errors='coerce')
        for c,a in [('open','ao'),('high','ah'),('low','al'),('close','ac')]:g[a]=g[c]*g.adj_factor
        eligible,known=eligibility(g,code,base.loc[code],members[code],calendar,cfg)
        rows.extend(stock_events(g,code,base.loc[code,'name'],eligible,known,cfg))
        if number%25==0:progress(f'周线形态研究 {number}/{len(base)}；两个观察模式、五种退出口径')
    return pd.DataFrame(rows)


def assign_bins(events):
    e=events.copy();thresholds=[]
    for c in FEATURES:e[c+'_bin']='校准不足'
    unique=e[e.qualified].drop_duplicates('event_id')
    for mode in MODES:
        for year in sorted(e.year.unique()):
            history=unique[(unique['mode']==mode)&(unique.year<year)]
            target=(e['mode']==mode)&(e.year==year)
            for c in FEATURES:
                v=history[c].dropna();lo=hi=np.nan
                if len(v)>=100:
                    lo,hi=v.quantile([1/3,2/3]);value=e.loc[target,c]
                    e.loc[target,c+'_bin']=np.select([value.isna(),value.le(lo),value.le(hi)],['特征缺失','低','中'],default='高')
                thresholds.append(dict(mode=mode,year=year,feature=c,history_events=len(v),low_edge=lo,high_edge=hi,
                    history_through=int(year)-1,status='已冻结' if len(v)>=100 else '校准不足'))
    e['joint_bin']=e.cross_speed_bin+'斜率 / '+e.gap_bin+'间距'
    return e,pd.DataFrame(thresholds)


def metrics(g):
    c=g[g.closed];v=c.net_pct.dropna();n=len(v)
    # ceil确保小样本至少移除一个最好值，另列有效样本数。
    cut=max(1,math.ceil(n*.01)) if n else 0;trim=v.sort_values().iloc[:n-cut]
    return dict(events=len(g),filled=int(g.filled.sum()),closed=n,unresolved=int((~g.resolved).sum()),
        cancelled=int((g.resolved&~g.filled).sum()),mean_net_pct=v.mean(),median_net_pct=v.median(),
        win_pct=v.gt(0).mean()*100 if n else np.nan,p10_net_pct=v.quantile(.1) if n else np.nan,
        trim_top1_mean_pct=trim.mean(),trim_remaining=len(trim),mean_hold_days=c.hold_days.mean() if n else np.nan,
        mean_mfe_pct=c.mfe_pct.mean() if n else np.nan,mean_mae_pct=c.mae_pct.mean() if n else np.nan,
        mfe_ge10_pct=c.mfe_pct.ge(10).sum()/c.mfe_pct.notna().sum()*100 if n and c.mfe_pct.notna().any() else np.nan)


def build_reports(events,calendar,cfg):
    names=['events','summary','feature_groups','joint_groups','diagnostics','thresholds','daily_comparison','annual_coverage','status_counts']
    if events.empty:return {n:pd.DataFrame() for n in names}
    e,thresholds=assign_bins(events);q=e[e.qualified].copy()
    summary=[];feature=[];joint=[];diag=[];daily=[]
    for (mode,rule),whole in q.groupby(['mode','exit_rule']):
        for year,g in [('全部',whole)]+list(whole.groupby('year')):
            identity=dict(mode=mode,exit_rule=rule,year=year)
            summary.append(dict(identity,**metrics(g)))
            # 完整日期的全信号基准，包含确定取消的零发单收益。
            dates=g.groupby('signal_date').resolved.all();good=g[g.signal_date.isin(dates[dates].index)]
            baseline=good.groupby('signal_date').order_net_pct.mean()
            for c in FEATURES:
                for bucket,part in g.groupby(c+'_bin'):
                    feature.append(dict(identity,feature=c,bucket=bucket,**metrics(part)))
                    subgroup=good[good[c+'_bin']==bucket].groupby('signal_date').order_net_pct.mean()
                    delta=subgroup-baseline.reindex(subgroup.index)
                    daily.append(dict(identity,feature=c,bucket=bucket,complete_dates=len(delta),
                        day_equal_order_pct=subgroup.mean(),same_day_all_pct=baseline.reindex(subgroup.index).mean(),
                        edge_pp=delta.mean(),positive_edge_dates_pct=delta.gt(0).mean()*100 if len(delta) else np.nan))
            for bucket,part in g.groupby('joint_bin'):
                joint.append(dict(identity,bucket=bucket,**metrics(part)))
            for c in ['gap_direction','kd_state','weekday']:
                for bucket,part in g.groupby(c):diag.append(dict(identity,diagnostic=c,bucket=str(bucket),**metrics(part)))
    cover=[];cal=calendar[(calendar>=stamp(cfg.start))&(calendar<=stamp(cfg.end))]
    for mode in MODES:
        u=q[q['mode']==mode].drop_duplicates('event_id')
        for year in sorted(set(cal.year)):
            days=cal[cal.year==year];weeks=set(days.to_period('W-FRI'))
            p=u[u.year==str(year)];sw=set(pd.DatetimeIndex(p.signal_date).to_period('W-FRI'));fw=set(pd.DatetimeIndex(p.loc[p.filled,'signal_date']).to_period('W-FRI'))
            cover.append(dict(mode=mode,year=str(year),observed_weeks=len(weeks),no_signal_weeks=len(weeks-sw),no_filled_signal_weeks=len(weeks-fw),
                full_year=stamp(cfg.start)<=pd.Timestamp(year,1,1) and min(stamp(cfg.end),calendar.max())>=pd.Timestamp(year,12,31)))
    status=e.groupby(['mode','exit_rule','year','status']).size().reset_index(name='count')
    return dict(events=e,summary=pd.DataFrame(summary),feature_groups=pd.DataFrame(feature),joint_groups=pd.DataFrame(joint),
        diagnostics=pd.DataFrame(diag),thresholds=thresholds,daily_comparison=pd.DataFrame(daily),annual_coverage=pd.DataFrame(cover),status_counts=status)


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
    start=ds(stamp(cfg.start)-pd.Timedelta(days=450));end=ds(min(ready,effective_end+pd.Timedelta(days=110)))
    calendar=client.calendar(start,end)
    data,issues=client.download(calendar,set(basic.ts_code))
    hashed=pd.util.hash_pandas_object(data,index=False).to_numpy();hashed.sort();data_hash=hashlib.sha256(hashed.tobytes()).hexdigest()
    events=calculate_events(data,basic,member,calendar,cfg,progress)
    del data;gc.collect();progress('按先前年份冻结形态分组，汇总年度及同日对照')
    tables=build_reports(events,calendar,cfg)
    tables.update(data_issues=issues,universe=basic,industry_intervals=member)
    manifest=dict(version=VERSION,config=asdict(cfg),rules=RULES,created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        data_start=start,data_end=end,data_hash=data_hash,pool_hash=hashlib.sha256(member.to_csv(index=False).encode()).hexdigest(),
        pool_mode=mode,warnings=pool_warnings,data_issues=len(issues),universe_size=len(basic),download_workers=DOWNLOAD_WORKERS,
        study='周线N6上穿25；斜率、间距、加速度；两种观察模式×五种退出；无组合',
        limitations=['历史反复观察，尚非前向验证','分位边界只用过去形态但并不消除研究选择偏差',
        '指标公式沿用项目SKDJ；未对截图软件逐点导出值做完整校验','未知事件与未完成事件单列，缺行情仍可能漏信号',
        '最高浮盈不是实际收益；日期对照为上穿信号而非全科技池','同股重复事件及各窗口相关，不提供独立样本显著性声明'])
    zipped=make_zip(tables,manifest)
    run_id=hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()[:12]
    path=Path(cache_root)/'results'/f'{VERSION}_{run_id}.zip';atomic_bytes(zipped,path)
    return tables,manifest,zipped,str(path)


LABELS={'mode':'观察模式','exit_rule':'退出规则','year':'信号年度','feature':'形态特征','bucket':'分组',
    'events':'合格事件数','filled':'成交数','closed':'闭合数','unresolved':'结果未知或未完成','cancelled':'取消数',
    'mean_net_pct':'平均净收益%','median_net_pct':'净收益中位数%','win_pct':'胜率%', 'p10_net_pct':'收益10分位%',
    'trim_top1_mean_pct':'去掉最好1%后均值%','trim_remaining':'去极值后笔数','mean_hold_days':'平均持有日（含卖出日）',
    'mean_mfe_pct':'平均最高浮盈%','mean_mae_pct':'平均最大浮亏%','mfe_ge10_pct':'最高浮盈达10%的比例%',
    'complete_dates':'完整结果日期数','day_equal_order_pct':'日期等权发单收益%','same_day_all_pct':'同日全部信号收益%',
    'edge_pp':'相对同日基准差（百分点）','positive_edge_dates_pct':'超越基准日期占比%',
    'diagnostic':'诊断项','history_events':'历史校准事件数','low_edge':'低组上界','high_edge':'中组上界',
    'history_through':'校准截止年度','status':'状态','count':'数量','observed_weeks':'观察周数',
    'no_signal_weeks':'无新信号周','no_filled_signal_weeks':'无可成交新信号周','full_year':'完整年度'}


def show_results(st,tables,manifest,zipped):
    cfg=manifest['config'];st.subheader(manifest['version'])
    st.download_button('下载完整验证结果 ZIP',zipped,file_name=f"tech_swing_T3_1_{cfg['start']}_{cfg['end']}.zip",mime='application/zip')
    if manifest['warnings'] or manifest['data_issues']:st.warning('数据受限：'+'；'.join(manifest['warnings'])+f"；问题记录{manifest['data_issues']}条")
    if tables['events'].empty:st.info('本区间没有可识别的周线上穿25事件。');return
    st.caption('先看样本量、未完成数与净收益，再看最高浮盈。第一年通常用于校准；各组不构成资金组合。')
    mode=st.selectbox('观察模式',MODES)
    rule=st.selectbox('退出口径',EXITS,index=2)
    year=st.selectbox('信号年度',['全部']+sorted(tables['events'].year.unique()))
    feat=st.selectbox('形态特征',list(FEATURES),format_func=lambda x:FEATURES[x])
    def selected(name,filter_feature=False):
        f=tables[name].copy()
        for col,val in [('mode',mode),('exit_rule',rule),('year',year)]:
            if col in f:
                if col=='year' and name in ['thresholds','annual_coverage','status_counts'] and year=='全部':continue
                f=f[f[col].eq(val)]
        if filter_feature and 'feature' in f:f=f[f.feature.eq(feat)]
        if 'feature' in f:f['feature']=f.feature.map(FEATURES)
        return f.rename(columns=LABELS)
    tabs=st.tabs(['基准与单项形态','斜率×间距','同日对照','诊断与覆盖','事件与规则'])
    with tabs[0]:
        st.dataframe(selected('summary'),hide_index=True)
        st.dataframe(selected('feature_groups',True),hide_index=True)
        st.caption('低/中/高边界按年冻结；全期汇总的各年边界可能不同。校准不足的事件只作基准，不能混称低组。')
        st.dataframe(selected('thresholds',True),hide_index=True)
    with tabs[1]:
        st.caption('比较上穿周增量与K−D的九格组合；不自动推荐最高收益格。')
        st.dataframe(selected('joint_groups'),hide_index=True)
    with tabs[2]:
        st.caption('仅比较同日全信号结果均确定的日期；含确定取消单的零收益，不含未知。用于检查是否只是碰到整体好行情。')
        st.dataframe(selected('daily_comparison',True),hide_index=True)
    with tabs[3]:
        st.dataframe(selected('diagnostics'),hide_index=True)
        st.caption('无新信号周不等于资金空仓周；本版没有仓位系统。')
        st.dataframe(selected('annual_coverage'),hide_index=True)
    with tabs[4]:
        st.dataframe(selected('status_counts'),hide_index=True)
        e=tables['events'];e=e[(e['mode']==mode)&(e.exit_rule==rule)]
        if year!='全部':e=e[e.year==year]
        st.dataframe(e.tail(300),hide_index=True)
        st.text(STUDY_NOTES);st.dataframe(tables['data_issues']);st.json(manifest)


def main():
    import streamlit as st
    st.set_page_config(page_title='周线SKDJ形态验证 T3.1',layout='wide')
    st.title('周线SKDJ T3.1 · 上穿25形态验证')
    try:default=str(st.secrets.get('TUSHARE_TOKEN',st.secrets.get('tushare_token','')))
    except Exception:default=''
    with st.sidebar:
        token=st.text_input('Tushare Token',value=os.environ.get('TUSHARE_TOKEN',default),type='password')
        start=st.date_input('上穿事件开始日',value=date(2022,1,1));end=st.date_input('上穿事件结束日',value=latest_ready_day().date())
        price=st.number_input('最低股价（高于，元）',value=10.,min_value=0.,step=1.)
        low=st.number_input('最低流通市值（亿元）',value=50.,min_value=0.,step=10.)
        high=st.number_input('最高流通市值（亿元）',value=1000.,min_value=1.,step=100.)
        root=st.text_input('数据缓存目录',value='tech_swing_cache')
        st.caption('沿用原缓存，四路并发补缺；固定周线N=6、平滑3；无三仓。')
        run=st.button('运行周线形态验证',type='primary')
    with st.expander('研究口径'):st.text(STUDY_NOTES)
    if run:
        if not token.strip():st.error('请输入Token或设置TUSHARE_TOKEN。')
        elif start>end or low>=high or not root.strip():st.error('请检查日期、市值范围及缓存目录。')
        else:
            box=st.empty();last=[0.]
            def progress(msg):
                if time.monotonic()-last[0]>.25:box.info(msg);last[0]=time.monotonic()
            try:
                st.session_state['t31_result']=run_research(token,root.strip(),Config(ds(start),ds(end),price,low,high),progress)
                box.success('验证完成，可下载完整结果。')
            except Exception as exc:box.error(str(exc).replace(token,'[隐藏]')[:500]);st.info('成功下载的数据已缓存，修复后可继续补缺。')
    if 't31_result' in st.session_state:
        tables,manifest,zipped,_=st.session_state['t31_result'];show_results(st,tables,manifest,zipped)


def self_test():
    import unittest
    from unittest.mock import patch
    class Tests(unittest.TestCase):
        def setUp(self):
            self.cal=pd.bdate_range('2021-01-04',periods=800)
            t=np.arange(len(self.cal));c=25+6*np.sin(t/18)+2*np.sin(t/5)
            self.g=pd.DataFrame(dict(open=c,high=c+1,low=c-1,close=c,ac=c,ah=c+1,al=c-1,ao=c,
                pre_close=c,adj_factor=1.,up_limit=c*1.1,down_limit=c*.9,vol=1000.,circ_mv=2000000.,turnover_rate=1.),index=self.cal)
        def test_completed_equals_original_formula(self):
            shape=weekly_shape(self.g)
            w=self.g.groupby(self.cal.to_period('W-FRI')).agg(ac=('ac','last'),ah=('ah','max'),al=('al','min'))
            expected=skdj(w,6)
            actual=shape[shape.complete]
            np.testing.assert_allclose(actual.k,expected.k,equal_nan=True)
            np.testing.assert_allclose(actual.d,expected.d,equal_nan=True)
        def test_provisional_matches_bruteforce(self):
            full=weekly_shape(self.g)
            for i in [100,101,102,103,104,127,388,620]:
                prefix=self.g.iloc[:i+1]
                w=prefix.groupby(prefix.index.to_period('W-FRI')).agg(ac=('ac','last'),ah=('ah','max'),al=('al','min'))
                expected=skdj(w,6).iloc[-1]
                self.assertAlmostEqual(full.k.iloc[i],expected.k)
                self.assertAlmostEqual(full.d.iloc[i],expected.d)
                short=weekly_shape(prefix)
                pd.testing.assert_frame_equal(full.drop(columns='complete').iloc[:i+1],short.drop(columns='complete'))
            altered=self.g.copy();altered.loc[self.cal[301]:,['ac','ah','al']]*=4
            pd.testing.assert_frame_equal(full.iloc[:301],weekly_shape(altered).iloc[:301])
        def test_missing_week_resets(self):
            self.g.loc[self.cal[302],['ac','ah','al']]=np.nan
            s=weekly_shape(self.g)
            self.assertTrue(s.k.iloc[302:360].isna().all())
            self.assertTrue(s.k.iloc[-20:].notna().all())
            self.g[['ac','ah','al']]=20.
            self.assertAlmostEqual(weekly_shape(self.g).k.iloc[-1],50.)
        def test_first_cross_per_week(self):
            shape=weekly_shape(self.g);shape['cross']=False
            shape.loc[self.cal[200:205],'cross']=[True,False,True,True,True]
            shape.loc[self.cal[200:205],['k','d']]=[30.,20.]
            eligible=pd.Series(True,index=self.cal);eligible.iloc[200]=False
            cfg=Config(ds(self.cal[200]),ds(self.cal[204]))
            with patch(__name__+'.weekly_shape',return_value=shape):
                e=pd.DataFrame(stock_events(self.g,'600001.SH','test',eligible,pd.Series(True,index=self.cal),cfg))
            p=e[e['mode']=='周中逐日观察']
            self.assertEqual(len(p),5);self.assertTrue(p.signal_date.eq(self.cal[200]).all())
            self.assertFalse(p.qualified.any())
            w=e[e['mode']=='周收盘确认'];self.assertEqual(len(w),5)
            self.assertTrue(w.signal_date.eq(self.cal[204]).all())
        def test_execution_cost_and_horizon(self):
            g=self.g.copy()
            g[['open','ac','ao','close']]=20.;g['ah']=21.;g['al']=19.;g['up_limit']=22.;g['down_limit']=18.
            s=weekly_shape(g)
            r=trade_path(g,s,100,MODES[0],'持有5日')
            self.assertEqual(r['sell_date'],self.cal[105]);self.assertEqual(r['hold_days'],6)
            self.assertAlmostEqual(r['net_pct'],(20*.999*.998/(20*1.001*1.001)-1)*100)
            g.loc[self.cal[105],'open']=18.
            r=trade_path(g,s,100,MODES[0],'持有5日');self.assertEqual(r['sell_date'],self.cal[106])
            before=r['mfe_pct'];g.loc[self.cal[106],'ah']=9999
            self.assertEqual(trade_path(g,s,100,MODES[0],'持有5日')['mfe_pct'],before)
            g.loc[self.cal[100],'open']=22.
            self.assertTrue(trade_path(g,s,100,MODES[0],'持有5日')['resolved'])
            g.loc[self.cal[100],'open']=np.nan
            self.assertFalse(trade_path(g,s,100,MODES[0],'持有5日')['resolved'])
            g.loc[self.cal[100],'open']=20.
            g.loc[self.cal[105],'open']=np.nan
            r=trade_path(g,s,100,MODES[0],'持有5日')
            self.assertEqual(r['status'],'退出数据未知');self.assertFalse(r['closed'])
        def test_weekly_death_modes_and_pending(self):
            s=weekly_shape(self.g);s['death']=False;s.loc[self.cal[101],'death']=True
            a=trade_path(self.g,s,100,MODES[1],EXITS[-1]);b=trade_path(self.g,s,100,MODES[0],EXITS[-1])
            self.assertEqual(a['sell_date'],self.cal[102]);self.assertEqual(b['sell_date'],self.cal[140])
            s.loc[self.cal[100],'death']=True
            self.assertEqual(trade_path(self.g,s,100,MODES[1],EXITS[-1])['sell_date'],self.cal[101])
            self.assertFalse(trade_path(self.g,s,799,MODES[1],'持有5日')['resolved'])
            self.assertFalse(trade_path(self.g,s,800,MODES[1],'持有5日')['filled'])
        def test_bins_use_only_prior_shapes(self):
            rows=[]
            for year in ['2022','2023','2024']:
                for i in range(120):
                    row=dict(event_id=year+str(i),mode=MODES[0],year=year,qualified=True)
                    row.update({c:float(i) for c in FEATURES});rows.append(row)
            e=pd.DataFrame(rows);a,t=assign_bins(e)
            changed=e.copy();changed.loc[changed.year.ge('2023'),list(FEATURES)]=9999.
            b,u=assign_bins(changed)
            pd.testing.assert_frame_equal(t[t.year=='2023'],u[u.year=='2023'])
            self.assertTrue(a.loc[a.year=='2022','gap_bin'].eq('校准不足').all())
            self.assertEqual(a[(a.year=='2023')&(a.gap_bin=='低')].shape[0],40)
        def test_end_to_end_reports(self):
            cfg=Config('20220101',ds(self.cal[-1]));eligible=pd.Series(True,index=self.cal)
            e=pd.DataFrame(stock_events(self.g,'600001.SH','test',eligible,eligible,cfg))
            self.assertFalse(e.empty)
            tables=build_reports(e,self.cal,cfg)
            self.assertFalse(tables['summary'].empty)
            for row in tables['summary'].itertuples():
                g=e[(e['mode']==row.mode)&(e.exit_rule==row.exit_rule)]
                if row.year!='全部':g=g[g.year==row.year]
                self.assertEqual(row.closed,int(g.closed.sum()))
                if row.closed:self.assertAlmostEqual(row.mean_net_pct,g.loc[g.closed,'net_pct'].mean())
            payload=make_zip(tables,dict(version=VERSION))
            with zipfile.ZipFile(io.BytesIO(payload)) as z:self.assertIn('thresholds.csv',z.namelist())
            self.assertTrue(all(f.empty for f in build_reports(pd.DataFrame(),self.cal,cfg).values()))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':
    if '--self-test' in sys.argv:self_test()
    else:main()
