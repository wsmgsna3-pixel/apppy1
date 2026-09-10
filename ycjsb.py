# -*- coding: utf-8 -*-
"""科技波段研究 T3.2 周线SKDJ同日排序验证 — streamlit run app.py

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

VERSION = "T3.2-SAME-DAY-RANK-20260910"
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


def build_shape_reports(events,calendar,cfg):
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


RANK_GROUPS=('第1名','第2名','第3名','前三名','后三名')
RANK_NOTES={
    '主检验':'固定K−D降序、前三名、同日至少6个合格候选、20交易日窗口；两个观察模式独立，不选择历史最优组合',
    '辅助检验':'K斜率、上穿前斜率、加速度、间距扩大速度分别排序；不合成评分，不自动推荐历史冠军',
    '名次':'信号当日收盘按指标降序，同分按流通市值降序，再按代码；先排全体候选，取消单不补位；各退出窗口共用信号时名次',
    '比较':'逐日比较第1/2/3名、前三名、后三名与同日全部候选及未选中候选；后三名是同一排序的末三名，仅作对照',
    '数量':'主表至少6个候选，保证前三名与后三名不重叠；另列全部日期。少于3只时前三名代表实际已有数量，无选满假设',
    '未知':'同日任何合格候选的对应退出结果未知，或该排序指标缺失，则该日不进入严格同日比较；报告被排除的日期数，不将未知补零',
    '胜率':'闭合事件胜率与日期平均净收益为不同统计；取消单仅在已确定发单收益中按零，不进入闭合交易胜率',
    '稳健性':'按年统计；20日、至少6候选、前三名另做8个日历周连续区块重采样2000次；仅给探索性95%区间，不宣称未见样本有效',
    '年份':'固定数值排序无需校准，从2022年起均可计算；2023—2026单独汇总便于对照T3.1分位研究，所有历史仍已被观察',
    '导入':'可以读取T3.1/T3.2结果ZIP直接重算排序，无需Token和下载；沿用原信号、成交、成本和未知状态，不能补足其尚未结束的交易',
}
RULES.update(RANK_NOTES)
STUDY_NOTES='\n'.join(f'{k}：{v}' for k,v in RULES.items())


def ranked_signals(events):
    """仅用信号已知字段生成名次；与未来收益和成交状态无关。"""
    fields=['event_id','ts_code','mode','signal_date','year','circ_mv_yi']+list(FEATURES)
    u=events.loc[events.qualified,fields].drop_duplicates('event_id').copy()
    if u.empty:return u
    keys=['mode','signal_date']
    u['candidate_count']=u.groupby(keys).event_id.transform('size')
    for c in FEATURES:
        values=pd.to_numeric(u[c],errors='coerce')
        u[c]=values.where(np.isfinite(values))
        u['rank_'+c]=np.nan
        valid=u[c].notna()&u.circ_mv_yi.notna()
        order=u[valid].sort_values(keys+[c,'circ_mv_yi','ts_code'],ascending=[True,True,False,False,True],kind='stable')
        u.loc[order.index,'rank_'+c]=order.groupby(keys).cumcount()+1
        u['known_'+c]=valid.groupby([u[k] for k in keys]).transform('all')
        u['tied_'+c]=u.groupby(keys)[c].transform('nunique').le(1)
    return u.reset_index(drop=True)


def block_interval(daily):
    """完整同日差值，连续8周区块抽样，保留无比较日期的周。"""
    v=daily[daily.comparable].copy()
    if len(v)<40:return dict(ci_dates=len(v),edge_ci_low=np.nan,edge_ci_high=np.nan,ci_status='可比日期不足40')
    key=v.signal_date.dt.to_period('W-FRI')
    w=v.groupby(key).agg(total=('edge_pp','sum'),count=('edge_pp','size'))
    axis=pd.period_range(daily.signal_date.min(),daily.signal_date.max(),freq='W-FRI')
    if len(axis)<16:return dict(ci_dates=len(v),edge_ci_low=np.nan,edge_ci_high=np.nan,ci_status='日历周不足16')
    a=w.reindex(axis,fill_value=0).to_numpy(float);rng=np.random.default_rng(3208)
    starts=rng.integers(0,len(a),size=(2000,math.ceil(len(a)/8)))
    ids=((starts[:,:,None]+np.arange(8))%len(a)).reshape(2000,-1)[:,:len(a)]
    sample=a[ids].sum(axis=1);valid=sample[:,1]>0
    lo,hi=np.quantile(sample[valid,0]/sample[valid,1],[.025,.975])
    return dict(ci_dates=len(v),edge_ci_low=lo,edge_ci_high=hi,ci_status='8周区块，2000次；探索性')


def ranking_reports(events,progress=lambda _:None):
    names=['ranked_signals','ranking_daily','ranking_summary','ranking_robustness']
    if events.empty or not events.qualified.any():return {n:pd.DataFrame() for n in names}
    u=ranked_signals(events)
    added=['event_id','candidate_count']+[prefix+c for c in FEATURES for prefix in ['rank_','known_','tied_']]
    q=events[events.qualified].drop(columns=[c for c in added if c!='event_id' and c in events],errors='ignore').merge(u[added],on='event_id',how='left',validate='many_to_one')
    # 分组累计后向量计算，避免每个日期反复遍历全量事件。
    rows=[]
    for number,((mode,rule),g) in enumerate(q.groupby(['mode','exit_rule']),1):
        by=g.groupby('signal_date',sort=True)
        base=by.agg(candidate_count=('event_id','size'),all_resolved=('resolved','all'),
            all_sum=('order_net_pct','sum'),all_closed=('closed','sum'),all_filled=('filled','sum'))
        for c in FEATURES:
            known=by['known_'+c].first();ties=by['tied_'+c].first()
            rank=g['rank_'+c]
            for group in RANK_GROUPS:
                select=(rank.eq(int(group[1])) if group in RANK_GROUPS[:3]
                        else (rank.le(3) if group=='前三名' else rank.gt(g.candidate_count-3)))
                part=g[select].copy()
                part['closed_return']=part.net_pct.where(part.closed)
                part['win']=part.closed&part.net_pct.gt(0)
                stats=part.groupby('signal_date').agg(selected_count=('event_id','size'),selected_sum=('order_net_pct','sum'),
                    selected_closed=('closed','sum'),selected_filled=('filled','sum'),selected_wins=('win','sum'),
                    closed_sum=('closed_return','sum'))
                f=base.join(stats).copy()
                countcols=['selected_count','selected_closed','selected_filled','selected_wins']
                f[countcols]=f[countcols].fillna(0).astype(int)
                f['feature_known']=known;f['all_tied']=ties
                f['comparable']=f.all_resolved&known&f.selected_count.gt(0)
                f['selected_order_pct']=(f.selected_sum/f.selected_count.replace(0,np.nan)).where(f.comparable)
                f['all_order_pct']=(f.all_sum/f.candidate_count).where(f.comparable)
                f['others_order_pct']=((f.all_sum-f.selected_sum)/(f.candidate_count-f.selected_count).replace(0,np.nan)).where(f.comparable)
                f['edge_pp']=f.selected_order_pct-f.all_order_pct
                f['edge_vs_others_pp']=f.selected_order_pct-f.others_order_pct
                f['mode']=mode;f['exit_rule']=rule;f['feature']=c;f['rank_group']=group
                f['year']=f.index.year.astype(str)
                rows.append(f.reset_index())
        progress(f'同日排序汇总 {number}/10')
    daily=pd.concat(rows,ignore_index=True);summary=[];robust=[]
    for key,whole in daily.groupby(['mode','exit_rule','feature','rank_group'],sort=True):
        identity=dict(zip(['mode','exit_rule','feature','rank_group'],key))
        periods=[('全部',whole),('2023及以后',whole[whole.year.ge('2023')])]+list(whole.groupby('year'))
        for year,period in periods:
            for scope in ['全部日期','至少6候选']:
                p=period if scope=='全部日期' else period[period.candidate_count.ge(6)]
                v=p[p.comparable];closed=int(v.selected_closed.sum());n=len(v)
                result=dict(identity,year=year,scope=scope,signal_dates=len(p),comparable_dates=n,
                    unknown_dates=int((~p.all_resolved).sum()),missing_feature_dates=int((~p.feature_known).sum()),
                    absent_rank_dates=int(p.selected_count.eq(0).sum()),excluded_dates=int((~p.comparable).sum()),
                    all_tied_dates=int(v.all_tied.sum()),selected_orders=int(v.selected_count.sum()),closed=closed,
                    cancelled=int((v.selected_count-v.selected_filled).sum()),
                    event_mean_pct=v.closed_sum.sum()/closed if closed else np.nan,
                    event_win_pct=v.selected_wins.sum()/closed*100 if closed else np.nan,
                    day_equal_order_pct=v.selected_order_pct.mean(),day_equal_all_pct=v.all_order_pct.mean(),
                    day_equal_others_pct=v.others_order_pct.mean(),edge_pp=v.edge_pp.mean(),edge_vs_others_pp=v.edge_vs_others_pp.mean(),
                    positive_edge_dates_pct=v.edge_pp.gt(0).mean()*100 if n else np.nan,
                    worst_date_pct=v.selected_order_pct.min(),p10_date_pct=v.selected_order_pct.quantile(.1) if n else np.nan)
                summary.append(result)
                if year=='全部' and scope=='至少6候选' and key[1]=='持有20日' and key[3]=='前三名':
                    robust.append(dict(identity,year=year,scope=scope,edge_pp=v.edge_pp.mean(),**block_interval(p)))
    return dict(ranked_signals=u,ranking_daily=daily,ranking_summary=pd.DataFrame(summary),ranking_robustness=pd.DataFrame(robust))


def build_reports(events,calendar,cfg):
    tables=build_shape_reports(events,calendar,cfg)
    tables.update(ranking_reports(tables['events']))
    return tables


def reanalyze_zip(source,progress=lambda _:None):
    """只读取固定成员，不解压文件，不读取或执行压缩包中的代码。"""
    with zipfile.ZipFile(source) as z:
        if sum(x.file_size for x in z.infolist())>1024**3:raise ValueError('结果解压后超过1GB，请缩小研究区间。')
        manifest=json.loads(z.read('manifest.json'))
        if not str(manifest.get('version','')).startswith(('T3.1-','T3.2-')):raise ValueError('请上传T3.1或T3.2生成的结果ZIP。')
        keep=['events','summary','feature_groups','joint_groups','diagnostics','thresholds','daily_comparison',
              'annual_coverage','status_counts','data_issues','universe','industry_intervals']
        tables={}
        for name in keep:
            if name+'.csv' not in z.namelist():continue
            try:tables[name]=pd.read_csv(z.open(name+'.csv'),dtype={'year':str,'ts_code':str,'event_id':str})
            except pd.errors.EmptyDataError:tables[name]=pd.DataFrame()
    if 'events' not in tables:raise ValueError('结果缺少events.csv。')
    e=tables['events']
    if not e.empty:
        required={'event_id','ts_code','mode','signal_date','year','circ_mv_yi','exit_rule','qualified',
            'closed','filled','resolved','order_net_pct','net_pct'}|set(FEATURES)
        if required-set(e):raise ValueError('事件明细缺少字段：'+','.join(sorted(required-set(e))))
        for c in ['qualified','closed','filled','resolved']:
            if not e[c].astype(str).isin(['True','False']).all():raise ValueError('布尔状态异常：'+c)
            e[c]=e[c].astype(str).eq('True')
        for c in ['signal_date','buy_date','sell_date']:
            if c in e:e[c]=pd.to_datetime(e[c],errors='raise')
        if e.signal_date.isna().any() or not e['mode'].isin(MODES).all() or not e.exit_rule.isin(EXITS).all():raise ValueError('事件日期、观察模式或退出规则异常。')
        if e.duplicated(['event_id','exit_rule']).any():raise ValueError('同事件同退出窗口有重复行。')
        fixed=['ts_code','mode','signal_date','year','circ_mv_yi','qualified']+list(FEATURES)
        if e.groupby('event_id')[fixed].nunique(dropna=False).gt(1).any().any():raise ValueError('同事件的信号字段跨退出窗口不一致。')
        if (e.resolved&e.qualified&e.order_net_pct.isna()).any() or (e.closed&e.net_pct.isna()).any():raise ValueError('已确定事件缺少收益。')
        if (e.closed&~e.resolved).any() or (e.closed&~e.filled).any():raise ValueError('成交或闭合状态不一致。')
    progress('复用原信号和成交结果，开始同日排序')
    tables.update(ranking_reports(e,progress))
    old_version=manifest['version'];manifest=dict(manifest)
    manifest.update(version=VERSION,source_version=old_version,source_created_at=manifest.get('created_at'),
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),rules=RULES,
        study='同日排序验证；主检验K−D降序前三名、至少6候选、持有20日；无资金组合',
        analysis_mode='导入结果重算，未新增行情，未更新未完成交易')
    manifest['limitations']=list(manifest.get('limitations',[]))+['导入结果保留原成交路径及数据截止日；不能补足未完成交易','辅助指标及窗口是探索性比较；不自动选择最优评分']
    zipped=make_zip(tables,manifest)
    return tables,manifest,zipped,''


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
        study='同日排序验证；主检验K−D降序前三名、至少6候选、持有20日；无资金组合',
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


RANK_LABELS=dict(LABELS,rank_group='名次范围',scope='候选数量范围',signal_dates='信号日期数',
    comparable_dates='可比日期数',unknown_dates='含未知结果日期',missing_feature_dates='指标缺失日期',
    absent_rank_dates='无该名次日期',excluded_dates='不可比日期（去重）',all_tied_dates='全体指标同分日期',
    selected_orders='入选订单数',event_mean_pct='可比日期闭合事件均值%',event_win_pct='可比日期闭合事件胜率%',
    day_equal_order_pct='入选组日期等权收益%',day_equal_all_pct='同日全部候选收益%',
    day_equal_others_pct='同日未选中候选收益%',edge_vs_others_pp='相对未选中候选差（百分点）',
    worst_date_pct='最差日期组均值%',p10_date_pct='日期组均值10分位%',ci_dates='区间估计日期数',
    edge_ci_low='同日收益差95%区间下限',edge_ci_high='同日收益差95%区间上限',ci_status='区间口径')


def show_results(st,tables,manifest,zipped):
    cfg=manifest['config'];st.subheader(manifest['version'])
    st.download_button('下载T3.2完整验证结果 ZIP',zipped,file_name=f"tech_swing_T3_2_{cfg['start']}_{cfg['end']}.zip",mime='application/zip')
    st.caption(f"行情截止：{manifest.get('data_end',cfg['end'])}。"+manifest.get('analysis_mode','重新计算行情与成交路径。'))
    if manifest.get('warnings') or manifest.get('data_issues',0):st.warning('数据受限：'+'；'.join(manifest.get('warnings',[]))+f"；问题记录{manifest.get('data_issues',0)}条")
    if tables['ranking_summary'].empty:st.info('没有可用于排序的合格上穿事件。');return
    st.info('主检验固定：K−D降序、前三名、同日至少6候选、20日窗口。前三名是独立事件比较，没有三仓资金配置。')
    mode=st.selectbox('观察模式',MODES)
    feature=st.selectbox('排序指标',['gap','cross_speed','pre_speed','acceleration','gap_change'],format_func=lambda c:FEATURES[c])
    rule=st.selectbox('退出口径',EXITS,index=2)
    scope=st.selectbox('候选数量范围',['至少6候选','全部日期'])
    year=st.selectbox('信号年度',['全部','2023及以后']+sorted(tables['events'].year.unique()))
    def select(name):
        f=tables[name].copy()
        for col,val in [('mode',mode),('feature',feature),('exit_rule',rule),('scope',scope)]:
            if col in f:f=f[f[col].eq(val)]
        if 'year' in f and name!='ranking_robustness':f=f[f.year.eq(year)]
        if 'feature' in f:f['feature']=f.feature.map(FEATURES)
        return f.rename(columns=RANK_LABELS)
    tabs=st.tabs(['同日排序结果','逐年与稳健性','逐日及信号明细','原形态基准与数据'])
    with tabs[0]:
        st.caption('先比较日期等权收益差，再看事件均值与胜率。两种观察模式独立，辅助指标和窗口不自动选优。')
        st.dataframe(select('ranking_summary'),hide_index=True)
        st.caption('同日所有候选结果均确定、且指标完整，才进入严格比较。排除原因可能重叠，以“不可比日期（去重）”为准。')
    with tabs[1]:
        f=tables['ranking_summary'];f=f[f['mode'].eq(mode)&f.feature.eq(feature)&f.exit_rule.eq(rule)&f.scope.eq(scope)&f.rank_group.eq('前三名')&~f.year.isin(['全部','2023及以后'])]
        st.dataframe(f.rename(columns=RANK_LABELS),hide_index=True)
        st.caption('下表固定全部年份、20日、至少6候选、前三名。8周区块保留同周股票共振及部分时间相关性；探索性区间不等于实盘有效证明。')
        f=tables['ranking_robustness'];f=f[f['mode'].eq(mode)&f.feature.eq(feature)] if not f.empty else f
        st.dataframe(f.rename(columns=RANK_LABELS),hide_index=True)
    with tabs[2]:
        f=tables['ranking_daily'];f=f[f['mode'].eq(mode)&f.feature.eq(feature)&f.exit_rule.eq(rule)&f.rank_group.eq('前三名')]
        if scope=='至少6候选':f=f[f.candidate_count.ge(6)]
        if year=='2023及以后':f=f[f.year.ge('2023')]
        elif year!='全部':f=f[f.year.eq(year)]
        st.caption('日期表保留不可比日期，可定位未知结果；完整数据见ZIP。')
        st.dataframe(f.tail(300),hide_index=True)
        u=tables['ranked_signals'];u=u[u['mode'].eq(mode)]
        if year=='2023及以后':u=u[u.year.ge('2023')]
        elif year!='全部':u=u[u.year.eq(year)]
        cols=['signal_date','ts_code','candidate_count',feature,'rank_'+feature,'circ_mv_yi','known_'+feature]
        st.dataframe(u.sort_values(['signal_date','rank_'+feature])[cols].tail(300),hide_index=True)
    with tabs[3]:
        f=tables.get('summary',pd.DataFrame())
        if not f.empty:
            f=f[f['mode'].eq(mode)&f.exit_rule.eq(rule)]
            if year!='2023及以后':f=f[f.year.eq(year)]
            st.dataframe(f.rename(columns=LABELS),hide_index=True)
        st.text(STUDY_NOTES);st.dataframe(tables.get('data_issues',pd.DataFrame()));st.json(manifest)


def main():
    import streamlit as st
    st.set_page_config(page_title='周线SKDJ同日排序 T3.2',layout='wide')
    st.title('周线SKDJ T3.2 · 同日排序验证')
    try:default=str(st.secrets.get('TUSHARE_TOKEN',st.secrets.get('tushare_token','')))
    except Exception:default=''
    token=''
    with st.sidebar:
        source=st.radio('数据来源',['导入T3.1/T3.2结果（无需下载行情）','下载行情重新回测'])
        if source.startswith('导入'):
            uploaded=st.file_uploader('上传原回测结果ZIP',type=['zip'])
            st.caption('使用ZIP内原日期与股票池设置。可直接上传刚完成的T3.1结果；未完成交易保持原状态。')
        else:
            token=st.text_input('Tushare Token',value=os.environ.get('TUSHARE_TOKEN',default),type='password')
            start=st.date_input('上穿事件开始日',value=date(2022,1,1));end=st.date_input('上穿事件结束日',value=latest_ready_day().date())
            price=st.number_input('最低股价（高于，元）',value=10.,min_value=0.,step=1.)
            low=st.number_input('最低流通市值（亿元）',value=50.,min_value=0.,step=10.)
            high=st.number_input('最高流通市值（亿元）',value=1000.,min_value=1.,step=100.)
            root=st.text_input('数据缓存目录',value='tech_swing_cache')
            st.caption('沿用原缓存，四路并发补缺；固定周线N=6、平滑3。')
        run=st.button('运行T3.2同日排序验证',type='primary')
    with st.expander('研究规则'):st.text(STUDY_NOTES)
    if run:
        box=st.empty();last=[0.]
        def progress(msg):
            if time.monotonic()-last[0]>.25:box.info(msg);last[0]=time.monotonic()
        try:
            if source.startswith('导入'):
                if uploaded is None:raise ValueError('请先上传T3.1或T3.2结果ZIP。')
                st.session_state['t32_result']=reanalyze_zip(io.BytesIO(uploaded.getvalue()),progress)
            else:
                if not token.strip():raise ValueError('请输入Token或设置TUSHARE_TOKEN。')
                if start>end or low>=high or not root.strip():raise ValueError('请检查日期、市值范围及缓存目录。')
                st.session_state['t32_result']=run_research(token,root.strip(),Config(ds(start),ds(end),price,low,high),progress)
            box.success('T3.2验证完成，结果已保留，可下载。')
        except Exception as exc:
            msg=str(exc);msg=msg.replace(token,'[隐藏]') if token else msg
            box.error(msg[:500])
    if 't32_result' in st.session_state:
        tables,manifest,zipped,_=st.session_state['t32_result'];show_results(st,tables,manifest,zipped)


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
        def ranking_fixture(self):
            records=[]
            for day in pd.to_datetime(['2023-01-06','2023-01-13']):
                for i in range(6):
                    row=dict(event_id=f'{day.date()}_{i}',ts_code=f'{600000+i}.SH',mode=MODES[0],signal_date=day,
                        year='2023',circ_mv_yi=100.+i,qualified=True,exit_rule='持有20日',closed=True,filled=True,resolved=True,
                        net_pct=float(i),order_net_pct=float(i))
                    row.update({c:float(6-i) for c in FEATURES});records.append(row)
            return pd.DataFrame(records)
        def test_rank_causal_and_ties(self):
            e=self.ranking_fixture();a=ranked_signals(e)
            changed=e.copy();changed['net_pct']=1000.;changed['filled']=False;changed['resolved']=False
            pd.testing.assert_frame_equal(a,ranked_signals(changed))
            e['gap']=5.;u=ranked_signals(e)
            self.assertTrue(u.loc[u.ts_code=='600005.SH','rank_gap'].eq(1).all())
            self.assertTrue(u.tied_gap.all())
        def test_cancel_no_replacement_and_unknown_date(self):
            e=self.ranking_fixture()
            e.loc[0,['closed','filled']]=False;e.loc[0,'net_pct']=np.nan;e.loc[0,'order_net_pct']=0.
            e.loc[11,'resolved']=False;e.loc[11,'order_net_pct']=np.nan;e.loc[11,'closed']=False;e.loc[11,'net_pct']=np.nan
            t=ranking_reports(e)
            s=t['ranking_summary'];r=s[s.feature.eq('gap')&s.rank_group.eq('前三名')&s.year.eq('全部')&s.scope.eq('至少6候选')].iloc[0]
            self.assertEqual(r.comparable_dates,1);self.assertEqual(r.unknown_dates,1)
            self.assertEqual(r.selected_orders,3);self.assertEqual(r.cancelled,1)
            self.assertAlmostEqual(r.day_equal_order_pct,1.)
            self.assertAlmostEqual(r.edge_pp,-1.5)
            self.assertAlmostEqual(r.edge_vs_others_pp,-3.)
        def test_missing_feature_and_sparse(self):
            e=self.ranking_fixture();e.loc[5,'gap']=np.nan
            t=ranking_reports(e);d=t['ranking_daily']
            a=d[d.feature.eq('gap')&d.rank_group.eq('前三名')].sort_values('signal_date')
            self.assertFalse(a.comparable.iloc[0]);self.assertTrue(a.comparable.iloc[1])
            sparse=ranking_reports(e.iloc[:2].copy())['ranking_summary']
            self.assertTrue(sparse.loc[sparse.scope.eq('至少6候选'),'comparable_dates'].eq(0).all())
            self.assertEqual(sparse.loc[sparse.scope.eq('全部日期')&sparse.feature.eq('gap')&sparse.rank_group.eq('第3名')&sparse.year.eq('全部'),'absent_rank_dates'].iloc[0],1)
        def test_import_roundtrip_and_validation(self):
            e=self.ranking_fixture();manifest=dict(version='T3.1-WEEKLY-SHAPE-20260910',config=asdict(Config()),warnings=[],data_issues=0)
            payload=make_zip({'events':e},manifest)
            tables,new,blob,_=reanalyze_zip(io.BytesIO(payload))
            self.assertEqual(new['version'],VERSION);self.assertFalse(tables['ranking_summary'].empty)
            self.assertEqual(tables['events'].net_pct.tolist(),e.net_pct.tolist())
            b,_,_,_=reanalyze_zip(io.BytesIO(blob))
            pd.testing.assert_frame_equal(tables['ranking_summary'],b['ranking_summary'])
            e.loc[0,'resolved']=False
            with self.assertRaises(ValueError):reanalyze_zip(io.BytesIO(make_zip({'events':e},manifest)))
        def test_block_interval_constant_edge(self):
            days=pd.date_range('2023-01-06',periods=60,freq='W-FRI')
            d=pd.DataFrame(dict(signal_date=days,comparable=True,edge_pp=2.))
            r=block_interval(d)
            self.assertAlmostEqual(r['edge_ci_low'],2.);self.assertAlmostEqual(r['edge_ci_high'],2.)

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':
    if '--self-test' in sys.argv:self_test()
    elif '--reanalyze' in sys.argv:
        import argparse
        parser=argparse.ArgumentParser();parser.add_argument('--reanalyze',required=True);parser.add_argument('--output',required=True)
        args=parser.parse_args();tables,manifest,zipped,_=reanalyze_zip(args.reanalyze,print);atomic_bytes(zipped,args.output)
        print('已生成',args.output)
    else:main()
