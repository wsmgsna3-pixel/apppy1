# -*- coding: utf-8 -*-
"""科技波段研究 gpt1.1 价格波段与逐周跟踪 — streamlit run app.py

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

VERSION = "gpt1.1"
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
    end: str = "20260911"
    signal_mode: str = "周中逐日"
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


RULES={
 '版本':'gpt1.1；无SKDJ、无资金组合；前五名是独立事件推荐，不模拟五仓',
 '股票池':'历史科技股；信号日不复权价>10元，流通市值50—1000亿元；保留上市180日、风险警示近似排除',
 '启动':'日收盘首次由不高于变为高于此前两个完整交易周的最高价；默认每日更新，周内最多一次；周收盘模式仅检查完整周收盘',
 '周线':'所有结构高低点只来自此前完整周；不使用本周最终高低点提前计算。缺报价的周不作有效结构',
 '上行能力':'以信号前最近完成周为终点，向前取13个互不重叠的4周收盘收益；必须13段完整且至少4段上涨，取上涨段收益中位数作为历史典型上行幅度',
 '空间筛选':'历史典型上行幅度≥10%；是待验证历史特征，不是未来目标收益或预测涨幅。保留不做空间筛选的基础启动对照',
 '风险':'初始止损为此前两个完整周最低价；信号日收盘到止损距离必须为价格的2%—10%；下一开盘按含滑点买价重新检查，超范围或已跌破止损则取消',
 '排序':'历史典型上行幅度/信号日初始风险百分比，降序；同分按流通市值降序，再按代码；先排全体合格候选再看未来成交，取消不补位',
 '重复':'同股同周仅首次启动。同股日后产生新的启动可成为独立事件，与旧事件可能重叠；不是实际加仓，收益不可累加为账户收益',
 '买入':'信号收盘确认，次交易日开盘买；开盘涨停、停牌或风险不符取消；必要行情缺失为未知',
 '止损':'初始止损不下移。次日起日内最低价触及此前已知保护线触发；跳空跌破按开盘价而非保护价；买入当天触及则受T+1限制次日开盘退出',
 '止盈':'R=含滑点买入价格−初始止损；最高收盘浮盈达到2R后启动移动止盈，保护线=最高收盘价−1R且只能上移。收盘上移的线次日生效，不用当天最高价回溯止盈',
 '执行限制':'跌停或停牌不能保证卖出，退出指令持续；日线不能恢复精确成交队列，触及跌停价时保守延后。关键路径行情缺失后不猜测成交',
 '期限':'没有固定持仓上限，无超时退出；信号区间由侧栏限定，历史持仓持续跟踪至最新已完成行情。尚未退出不当作零收益或最终盈利',
 '费用':'买入费0.10%、卖出费0.20%；每边滑点0.10%。R是价格风险，实际净亏损还受费用和跳空影响',
 '周次':'买入日起第5、10、15……交易日收盘为W1、W2……；休市不计，停牌仍占市场交易日。周中未退出收益按收盘计价并预扣卖出费用和滑点，并非已兑现',
 '样本成熟':'某周只有整批信号已获得完整5×周次交易日观察才纳入，提前退出也不能提前进入远期周统计',
 '主表':'每个已成熟周保留所有入选事件：已退出冻结实际净收益，仍持有用当周标记收益；未知单列。主收益和胜率使用当周已知且已成交事件，另列含取消0的订单收益',
 '辅助表':'仅仍持有股票单独统计，明确剩余数量；不能用该表代替主表。越远周次的成熟批次越少，不能直接解释为同一批股票随时间改善',
 '空窗':'统计每年无新合格信号的交易周，目标≤5；另列未满5只周。不等同于实际资金空仓，不强行凑满',
 '参数声明':'2周结构、13个4周窗口、至少4段上涨、10%空间及2%—10%风险均为本轮固定假设，未经过寻优；不因空窗过长自动放宽',
}
RULES['诊断']='仅事后分析，不参与选股或成交。浮盈/不利波动按复权价格相对含滑点买价，未扣手续费；收盘最大回撤按持有期间收盘及最终成交价。盘中退出当天高低价先后未知，报告上下界；开盘退出不计当天高低价。5%、10%及2R仅为固定诊断分档，不代表可成交止盈。已退出、仍持有及缺失路径分开；未成交不进入持仓诊断。'
STUDY_NOTES='\n'.join(f'{k}：{v}' for k,v in RULES.items())


def price_setup(g,mode):
    key=g.index.to_period('W-FRI')
    good=g[['ac','ah','al']].notna().all(axis=1)&g.ac.gt(0)&g.ah.ge(g.al)
    w=g.assign(key=key,good=good).groupby('key').agg(ac=('ac','last'),ah=('ah','max'),al=('al','min'),good=('good','all'))
    w.loc[~w.good,['ac','ah','al']]=np.nan
    # 将历史缺口传播到整个4周窗口，而非只检查两端价格。
    block=pd.DataFrame(index=w.index)
    for j in range(13):
        end=1+4*j
        valid=w.good.rolling(5,min_periods=5).sum().shift(end).eq(5)
        block[j]=((w.ac.shift(end)/w.ac.shift(end+4)-1)*100).where(valid)
    positive=block.where(block.gt(0));negative=(-block).where(block.lt(0))
    def mapped(x):return pd.Series(key.map(x),index=g.index,dtype=float)
    f=pd.DataFrame(index=g.index)
    f['trigger_level']=mapped(w.ah.rolling(2,min_periods=2).max().shift())
    f['initial_stop']=mapped(w.al.rolling(2,min_periods=2).min().shift())
    f['history_blocks']=mapped(block.count(axis=1));f['up_blocks']=mapped(positive.count(axis=1))
    f['typical_up_pct']=mapped(positive.median(axis=1));f['typical_down_pct']=mapped(negative.median(axis=1))
    f['history_ready']=f.history_blocks.eq(13)&f.up_blocks.ge(4)
    f['space_pass']=f.history_ready&f.typical_up_pct.ge(10)
    f['risk_pct']=(1-f.initial_stop/g.ac)*100
    f['risk_pass']=f.risk_pct.between(2,10)&f.initial_stop.gt(0)
    f['score']=f.typical_up_pct/f.risk_pct.where(f.risk_pct.gt(0))
    complete=np.r_[key[:-1]!=key[1:],g.index[-1].weekday()==4]
    above=g.ac.gt(f.trigger_level)
    if mode=='周收盘确认':
        weekly_above=w.ac.gt(w.ah.rolling(2,min_periods=2).max().shift())
        first=weekly_above&~weekly_above.shift(fill_value=False)
        f['trigger']=pd.Series(key.map(first),index=g.index).astype(bool)&complete
    else:
        trigger=above&~above.shift(fill_value=False)
        f['trigger']=trigger&trigger.groupby(key).cumsum().eq(1)
    f['week']=key.astype(str)
    return f


def lifecycle(g,buy_i,stop):
    n=len(g);cal=g.index
    out=dict(filled=False,closed=False,resolved=False,status='待买入',exit_reason='',buy_i=buy_i,sell_i=-1,
        buy_date=pd.NaT,sell_date=pd.NaT,buy_adj=np.nan,buy_raw=np.nan,sell_adj=np.nan,
        risk_pct_actual=np.nan,r_amount=np.nan,net_pct=np.nan,order_net_pct=np.nan,hold_days=np.nan,
        max_close_gain_pct=np.nan,unknown_from=-1,exit_delay_days=0)
    marks=[]
    if buy_i>=n:return out,marks
    op=g.open.to_numpy();lo=g.low.to_numpy();cl=g.close.to_numpy();ad=g.adj_factor.to_numpy()
    up=g.up_limit.to_numpy();down=g.down_limit.to_numpy();vol=g.vol.to_numpy()
    def cancel(reason):out.update(status=reason,resolved=True,order_net_pct=0.)
    i=buy_i
    if np.isfinite(vol[i]) and vol[i]<=0:cancel('停牌取消');return out,marks
    if not np.isfinite([op[i],ad[i],up[i],down[i],vol[i],stop]).all():out['status']='买入数据未知';out['unknown_from']=i;return out,marks
    if op[i]<=0 or ad[i]<=0 or down[i]<=0 or up[i]<down[i]:out['status']='买入数据异常';out['unknown_from']=i;return out,marks
    if op[i]>=up[i]-.005:cancel('开盘涨停取消');return out,marks
    buy_raw=min(op[i]*1.001,up[i]);buy=buy_raw*ad[i];r=buy-stop;risk=r/buy*100
    if not 2<=risk<=10:cancel('开盘风险不符取消');return out,marks
    out.update(filled=True,status='持有中',buy_date=cal[i],buy_adj=buy,buy_raw=buy_raw,risk_pct_actual=risk,r_amount=r)
    protect=stop;peak=buy;last_close=buy;trailing=False;pending=False;trigger_i=-1;reason=''
    def finish(j,price,at_open=False):
        sell=max(price*.999,down[j])*ad[j];net=(sell*.998/(buy*1.001)-1)*100
        out.update(closed=True,resolved=True,status='已退出',sell_i=j,sell_date=cal[j],sell_adj=sell,net_pct=net,
            exit_at_open=at_open,order_net_pct=net,hold_days=j-buy_i+1,exit_reason=reason,max_close_gain_pct=(peak/buy-1)*100,
            exit_delay_days=max(0,j-trigger_i-1) if pending else 0)
    for j in range(buy_i,n):
        suspended=np.isfinite(vol[j]) and vol[j]<=0
        if not suspended:
            if not np.isfinite([op[j],lo[j],cl[j],ad[j],down[j],vol[j]]).all() or min(op[j],cl[j],ad[j],down[j])<=0:
                out.update(status='持有路径未知',unknown_from=j,exit_reason='关键日线行情缺失',max_close_gain_pct=(peak/buy-1)*100);break
            if pending and j>buy_i:
                if op[j]>down[j]+.005:finish(j,op[j],True);break
            elif lo[j]*ad[j]<=protect:
                reason='移动止盈' if trailing else '初始止损';trigger_i=j
                target=min(op[j],protect/ad[j])
                if j==buy_i or target<=down[j]+.005 or op[j]<=down[j]+.005:pending=True
                else:finish(j,target,op[j]*ad[j]<=protect);break
            last_close=cl[j]*ad[j]
            if not pending:
                peak=max(peak,last_close)
                if peak-buy>=2*r:trailing=True;protect=max(protect,peak-r)
        if (j-buy_i+1)%5==0:
            marks.append(dict(week_no=(j-buy_i+1)//5,mark_i=j,mark_date=cal[j],
                net_pct=(last_close*.999*.998/(buy*1.001)-1)*100,mark_kind='停牌沿用最近价' if suspended else '收盘标记',
                protect_adj=protect,trail_active=trailing,pending_exit=pending))
    if not out['closed'] and out['unknown_from']<0:
        out.update(status='待可执行退出' if pending else '持有中',exit_reason=reason,max_close_gain_pct=(peak/buy-1)*100,hold_days=n-buy_i)
    return out,marks


def path_diagnostic(g,path):
    """事后持仓路径诊断；只读取已确定交易，不回流到信号或退出。"""
    out=dict(diagnostic_complete=False,diagnostic_note='未成交',mfe_low_pct=np.nan,mfe_high_pct=np.nan,
        mae_low_pct=np.nan,mae_high_pct=np.nan,close_mdd_pct=np.nan,w1_mfe_low_pct=np.nan,w2_mfe_low_pct=np.nan,
        stop_hit5='不适用',stop_hit10='不适用',stop_hit2r='不适用')
    if not path['filled']:return out
    buy=path['buy_adj'];start=path['buy_i'];closed=path['closed'];end=path['sell_i'] if closed else len(g)-1
    unknown=path['unknown_from'];complete=unknown<0
    if unknown>=0:end=min(end,int(unknown)-1)
    peak=possible_peak=buy;trough=possible_trough=buy;close_peak=buy;mdd=0.;early={};full_days=0
    for j in range(start,end+1):
        x=g.iloc[j]
        if np.isfinite(x.vol) and x.vol<=0:
            early[j-start+1]=(peak/buy-1)*100
            continue
        if not np.isfinite([x.open,x.adj_factor,x.vol]).all() or min(x.open,x.adj_factor)<=0:
            complete=False;break
        opening=x.open*x.adj_factor
        is_exit=closed and j==path['sell_i']
        if is_exit:
            sell=path['sell_adj']
            peak=max(peak,opening,sell);trough=min(trough,opening,sell)
            possible_peak=max(possible_peak,peak);possible_trough=min(possible_trough,trough)
            if not path['exit_at_open']:
                if not np.isfinite([x.high,x.low]).all() or x.low<=0 or x.high<x.low:complete=False
                else:
                    possible_peak=max(possible_peak,x.high*x.adj_factor)
                    possible_trough=min(possible_trough,x.low*x.adj_factor)
            mdd=max(mdd,(1-sell/close_peak)*100)
        else:
            if not np.isfinite([x.high,x.low,x.close]).all() or x.low<=0 or x.high<max(x.low,x.close,x.open) or x.low>min(x.close,x.open):
                complete=False;break
            peak=max(peak,x.high*x.adj_factor);trough=min(trough,x.low*x.adj_factor)
            possible_peak=max(possible_peak,peak);possible_trough=min(possible_trough,trough)
            close=x.close*x.adj_factor;close_peak=max(close_peak,close);mdd=max(mdd,(1-close/close_peak)*100)
            full_days+=1
        early[j-start+1]=(peak/buy-1)*100
    low=(peak/buy-1)*100;high=(possible_peak/buy-1)*100
    out.update(diagnostic_complete=bool(complete),diagnostic_note='完整已退出' if complete and closed else '完整截至当前' if complete else '路径不完整，仅为已知片段',
        mfe_low_pct=low,mfe_high_pct=high,mae_low_pct=(1-trough/buy)*100,
        mae_high_pct=(1-possible_trough/buy)*100,close_mdd_pct=max(0.,mdd))
    for n in [5,10]:
        if complete and (closed or end-start+1>=n):
            available=[v for k,v in early.items() if k<=n]
            out[f'w{n//5}_mfe_low_pct']=max(available) if available else np.nan
    if path['exit_reason']=='初始止损' and closed:
        for label,threshold in [('5',5.),('10',10.),('2r',2*path['r_amount']/buy*100)]:
            out['stop_hit'+label]=('未知' if not complete else '确定达到' if low>=threshold else '确定未达' if high<threshold else '退出日先后不明')
    return out


def diagnostic_reports(e):
    names=['diagnostic_events','diagnostic_summary','stop_excursion_summary']
    if e.empty:return {n:pd.DataFrame() for n in names}
    d=e[e.base_pass].copy()
    if d.empty:return {n:pd.DataFrame() for n in names}
    summary=[];bins=[]
    for group,mask in groups(d):
        whole=d[mask]
        for year,g in [('全部',whole)]+list(whole.groupby('year')):
            filled=g[g.filled];done=filled[filled.closed];valid=done[done.diagnostic_complete.eq(True)]
            stopped=done[done.exit_reason.eq('初始止损')];n=len(valid)
            summary.append(dict(group=group,year=year,events=len(g),filled=len(filled),closed=len(done),
                diagnostic_closed=n,closed_missing=len(done)-n,open_or_unknown=int((~filled.closed).sum()),
                stop_count=len(stopped),mfe_low_median_pct=valid.mfe_low_pct.median(),
                mfe_low_mean_pct=valid.mfe_low_pct.mean(),mae_high_mean_pct=valid.mae_high_pct.mean(),
                close_mdd_mean_pct=valid.close_mdd_pct.mean(),
                confirmed_ge10=int(valid.mfe_low_pct.ge(10).sum()),
                confirmed_ge10_then_loss=int((valid.mfe_low_pct.ge(10)&valid.net_pct.lt(0)).sum()),
                confirmed_ge10_then_loss_pct=(valid.mfe_low_pct.ge(10)&valid.net_pct.lt(0)).mean()*100 if n else np.nan))
            for threshold,col in [('价格浮盈5%','stop_hit5'),('价格浮盈10%','stop_hit10'),('价格浮盈2R','stop_hit2r')]:
                for category in ['确定未达','确定达到','退出日先后不明','未知']:
                    sub=stopped[stopped[col].eq(category)]
                    bins.append(dict(group=group,year=year,threshold=threshold,category=category,count=len(sub),
                        stop_total=len(stopped),share_of_stops_pct=len(sub)/len(stopped)*100 if len(stopped) else np.nan,
                        mean_net_pct=sub.net_pct.mean(),mean_hold_days=sub.hold_days.mean()))
    return dict(diagnostic_events=d,diagnostic_summary=pd.DataFrame(summary),stop_excursion_summary=pd.DataFrame(bins))


def calculate(data,basic,member,calendar,cfg,progress):
    rows=[];marks=[];base=basic.set_index('ts_code');members={c:m for c,m in member.groupby('ts_code')}
    for num,(code,part) in enumerate(data.groupby('ts_code',sort=True),1):
        if code not in base.index or code not in members:continue
        g=part.drop_duplicates('date').set_index('date').sort_index().reindex(calendar)
        for c in ['open','high','low','close','pre_close','vol','circ_mv','turnover_rate','adj_factor','up_limit','down_limit']:
            g[c]=pd.to_numeric(g[c],errors='coerce')
        for c,a in [('high','ah'),('low','al'),('close','ac')]:g[a]=g[c]*g.adj_factor
        eligible,known=eligibility(g,code,base.loc[code],members[code],calendar,cfg);f=price_setup(g,cfg.signal_mode)
        for i in np.flatnonzero(f.trigger&(calendar>=stamp(cfg.start))&(calendar<=stamp(cfg.end))):
            x=f.iloc[i];day=calendar[i];event_id=f'{code}|{ds(day)}'
            base_pass=bool(eligible.iloc[i] and x.risk_pass)
            row=dict(event_id=event_id,ts_code=code,name=base.loc[code,'name'],signal_date=day,signal_i=i,
                year=str(day.year),signal_week=x.week,signal_close=float(g.close.iloc[i]),circ_mv_yi=float(g.circ_mv.iloc[i]/10000),
                initial_stop_raw=float(x.initial_stop/g.adj_factor.iloc[i]),pool_pass=bool(eligible.iloc[i]),pool_known=bool(known.iloc[i]),base_pass=base_pass,main_pass=base_pass and bool(x.space_pass))
            row.update({c:x[c] for c in ['trigger_level','initial_stop','history_blocks','up_blocks','history_ready','typical_up_pct','typical_down_pct','risk_pct','risk_pass','space_pass','score']})
            if base_pass:
                path,weekly=lifecycle(g,i+1,float(x.initial_stop));row.update(path);row.update(path_diagnostic(g,path))
                marks.extend(dict(event_id=event_id,**w) for w in weekly)
            else:
                row.update(filled=False,closed=False,resolved=False,status='基础资格不符',buy_i=i+1,sell_i=-1,net_pct=np.nan,order_net_pct=np.nan)
            rows.append(row)
        if num%25==0:progress(f'价格启动与退出跟踪 {num}/{len(base)}')
    e=pd.DataFrame(rows);m=pd.DataFrame(marks)
    if not e.empty:
        e['rank']=np.nan
        order=e[e.main_pass].sort_values(['signal_date','score','circ_mv_yi','ts_code'],ascending=[True,False,False,True])
        e.loc[order.index,'rank']=order.groupby('signal_date').cumcount()+1
        e['selected']=e['rank'].le(5)
        e['mature_weeks']=np.maximum(0,(len(calendar)-e.buy_i)//5)
    return e,m


def groups(e):
    yield '前五名',e.selected
    for r in range(1,6):yield f'第{r}名',e['rank'].eq(r)
    yield '其余合格',e.main_pass&~e.selected
    yield '全部合格',e.main_pass
    yield '基础启动对照',e.base_pass


def weekly_view(e,marks,w):
    """只在整批达到观察年龄后纳入；提前卖出不能提前变成熟样本。"""
    v=e[e.base_pass&e.mature_weeks.ge(w)].copy()
    if v.empty:return v
    v['mark_i']=v.buy_i+5*w-1
    v['exited']=v.closed&v.sell_i.le(v.mark_i)
    v['cancelled']=v.resolved&~v.filled
    if marks.empty:mp=pd.Series(dtype=float)
    else:mp=marks[marks.week_no.eq(w)].set_index('event_id').net_pct
    v['week_net_pct']=v.event_id.map(mp)
    v.loc[v.exited,'week_net_pct']=v.loc[v.exited,'net_pct']
    v['week_order_pct']=v.week_net_pct
    v.loc[v.cancelled,'week_order_pct']=0.
    v['week_known']=v.week_order_pct.notna()
    v['holding']=v.filled&~v.exited&v.week_known
    return v


def distribution(values):
    v=pd.Series(values).dropna();n=len(v);trim=v.sort_values().iloc[:max(0,n-max(1,math.ceil(n*.01)))]
    return dict(mean_net_pct=v.mean(),median_net_pct=v.median(),win_pct=v.gt(0).mean()*100 if n else np.nan,
        net_ge10_pct=v.ge(10).mean()*100 if n else np.nan,net_ge20_pct=v.ge(20).mean()*100 if n else np.nan,
        p10_net_pct=v.quantile(.1) if n else np.nan,trim_top1_mean_pct=trim.mean())


def build_reports(e,marks,calendar,cfg):
    names=['events','weekly_marks','weekly_summary','survivor_summary','final_summary','weekly_date_comparison','coverage','filter_counts','latest_top5']
    if e.empty:
        tables={n:pd.DataFrame() for n in names}
        days=calendar[(calendar>=stamp(cfg.start))&(calendar<=stamp(cfg.end))]
        rows=[]
        for year in sorted(set(days.year)):
            count=len(set(days[days.year==year].to_period('W-FRI')))
            rows.append(dict(year=str(year),observed_weeks=count,no_signal_weeks=count,
                max_daily_below5_signal_weeks=0,selected_events=0,unique_stocks=0,
                full_year=stamp(cfg.start)<=pd.Timestamp(year,1,1) and stamp(cfg.end)>=pd.Timestamp(year,12,31) and calendar.max()>=pd.Timestamp(year,12,31)))
        tables['coverage']=pd.DataFrame(rows)
        return tables
    summary=[];survivors=[];comparison=[];final=[]
    for w in range(1,int(e.mature_weeks.max())+1):
        view=weekly_view(e,marks,w)
        if view.empty:continue
        for group,mask in groups(view):
            whole=view[mask]
            for year,g in [('全部',whole)]+list(whole.groupby('year')):
                values=g.loc[g.filled,'week_net_pct'];known=g[g.week_known]
                identity=dict(group=group,year=year,week_no=w)
                summary.append(dict(identity,mature_events=len(g),filled=int(g.filled.sum()),known_filled=int(values.notna().sum()),
                    exited=int(g.exited.sum()),holding=int(g.holding.sum()),cancelled=int(g.cancelled.sum()),unknown=int((~g.week_known).sum()),
                    mean_order_pct=known.week_order_pct.mean(),**distribution(values)))
                alive=g[g.holding]
                survivors.append(dict(identity,holding_events=len(alive),**distribution(alive.week_net_pct)))
        # 固定同一成熟日期，对照当天基础启动和同日其余合格股票。
        for day,g in view.groupby('signal_date'):
            chosen=g[g.selected]
            if chosen.empty:continue
            base_known=bool(g.week_known.all());main=g[g.main_pass];rest=main[~main.selected]
            main_known=bool(main.week_known.all())
            chosen_mean=chosen.week_order_pct.mean() if chosen.week_known.all() else np.nan
            comparison.append(dict(signal_date=day,year=str(day.year),week_no=w,selected_count=len(chosen),base_count=len(g),
                base_comparable=base_known,main_comparable=main_known,chosen_order_pct=chosen_mean,
                base_order_pct=g.week_order_pct.mean() if base_known else np.nan,
                edge_vs_base_pp=chosen_mean-g.week_order_pct.mean() if base_known else np.nan,
                rest_order_pct=rest.week_order_pct.mean() if main_known and len(rest) else np.nan))
    for group,mask in groups(e):
        whole=e[mask]
        for year,g in [('全部',whole)]+list(whole.groupby('year')):
            closed=g[g.closed];v=closed.net_pct
            final.append(dict(group=group,year=year,events=len(g),filled=int(g.filled.sum()),closed=len(closed),
                unresolved=int((~g.resolved).sum()),cancelled=int((g.resolved&~g.filled).sum()),
                mean_hold_days=closed.hold_days.mean() if len(closed) else np.nan,**distribution(v)))
    coverage=[];filters=[];end=min(stamp(cfg.end),calendar.max());days=calendar[(calendar>=stamp(cfg.start))&(calendar<=end)]
    for year in sorted(set(days.year)):
        yr=str(year);all_weeks=set(days[days.year==year].to_period('W-FRI'));p=e[e.year.eq(yr)];chosen=p[p.selected]
        count=chosen.groupby(['signal_week','signal_date']).size().groupby(level=0).max();coverage.append(dict(year=yr,observed_weeks=len(all_weeks),
            no_signal_weeks=len(all_weeks-set(pd.DatetimeIndex(chosen.signal_date).to_period('W-FRI'))),
            max_daily_below5_signal_weeks=int(count.lt(5).sum()),selected_events=len(chosen),unique_stocks=chosen.ts_code.nunique(),
            full_year=stamp(cfg.start)<=pd.Timestamp(year,1,1) and stamp(cfg.end)>=pd.Timestamp(year,12,31) and latest_ready_day()>=pd.Timestamp(year,12,31)))
        filters.append(dict(year=yr,price_triggers=len(p),pool_pass=int(p.pool_pass.sum()),base_pass=int(p.base_pass.sum()),
            history_ready=int((p.base_pass&p.history_ready).sum()),space_pass=int(p.main_pass.sum()),top5=len(chosen)))
    last=e.loc[e.selected,'signal_date'].max()
    return dict(events=e,weekly_marks=marks,weekly_summary=pd.DataFrame(summary),survivor_summary=pd.DataFrame(survivors),
        final_summary=pd.DataFrame(final),weekly_date_comparison=pd.DataFrame(comparison),coverage=pd.DataFrame(coverage),filter_counts=pd.DataFrame(filters),
        latest_top5=e[e.selected&e.signal_date.eq(last)].sort_values('rank'))


def recommendation_history(e,marks):
    if e.empty:return pd.DataFrame()
    rows=[]
    selected=e[e.selected]
    for w in range(1,int(selected.mature_weeks.max())+1 if len(selected) else 1):
        v=weekly_view(selected,marks,w)
        if v.empty:continue
        cols=['event_id','ts_code','name','signal_date','rank','filled','exited','holding','cancelled','week_known','week_net_pct','week_order_pct']
        rows.append(v[cols].assign(week_no=w))
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def make_zip(tables,manifest):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,df in tables.items():z.writestr(name+'.csv',df.to_csv(index=False).encode('utf-8-sig'))
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
        z.writestr('规则与口径.txt',STUDY_NOTES)
    return out.getvalue()


def run_research(token,cache_root,cfg,progress):
    client=DataClient(token,cache_root,progress);basic,member,mode,pool_warnings=client.universe()
    ready=latest_ready_day()
    if min(stamp(cfg.end),ready)<stamp(cfg.start):raise RuntimeError('尚未进入指定信号区间')
    start=ds(stamp(cfg.start)-pd.Timedelta(days=450));end=ds(ready)
    calendar=client.calendar(start,end);data,issues=client.download(calendar,set(basic.ts_code))
    hashed=pd.util.hash_pandas_object(data,index=False).to_numpy();hashed.sort();data_hash=hashlib.sha256(hashed.tobytes()).hexdigest()
    e,marks=calculate(data,basic,member,calendar,cfg,progress);del data;gc.collect()
    progress('汇总前五名逐周收益，保留提前退出事件')
    tables=build_reports(e,marks,calendar,cfg);tables['top5_weekly_history']=recommendation_history(e,marks);tables.update(diagnostic_reports(e))
    tables.update(data_issues=issues,universe=basic,industry_intervals=member)
    manifest=dict(version=VERSION,config=asdict(cfg),rules=RULES,created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        data_start=start,data_end=end,data_hash=data_hash,pool_hash=hashlib.sha256(member.to_csv(index=False).encode()).hexdigest(),
        pool_mode=mode,warnings=pool_warnings,data_issues=len(issues),universe_size=len(basic),download_workers=DOWNLOAD_WORKERS,
        limitations=['全部历史已反复观察；参数未证明有效','独立事件可能重叠，不是账户收益','日线无法恢复成交队列；滑点是简化假设',
            '周次越远成熟批次越少；须结合固定信号批次表解读','关键缺失路径单列，不把未知补成零','科技历史行业及风险警示重建仍有限制'])
    zipped=make_zip(tables,manifest);run_id=hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()[:12]
    path=Path(cache_root)/'results'/f'{VERSION}_{run_id}.zip';atomic_bytes(zipped,path)
    return tables,manifest,zipped,str(path)


LABELS={'group':'样本组','year':'信号年份','week_no':'持有周次','mature_events':'成熟信号数',
 'filled':'已成交','known_filled':'收益已知成交数','exited':'已退出','holding':'仍持有','cancelled':'取消',
 'unknown':'未知','mean_net_pct':'平均净收益%','median_net_pct':'中位净收益%','win_pct':'净盈利胜率%',
 'net_ge10_pct':'净收益≥10%占比','net_ge20_pct':'净收益≥20%占比','p10_net_pct':'收益10分位%',
 'trim_top1_mean_pct':'剔除最高1%后均值%','mean_order_pct':'含取消订单均值%',
 'ts_code':'代码','name':'名称','signal_date':'信号确认日','rank':'排名','score':'空间风险比',
 'typical_up_pct':'历史上涨段中位数%','risk_pct':'信号风险%','circ_mv_yi':'流通市值亿元',
 'initial_stop_raw':'信号日初始止损价','risk_pct_actual':'成交实际风险%','buy_date':'买入日','buy_raw':'买入价','sell_date':'退出日','status':'状态','exit_reason':'退出原因',
 'net_pct':'最终净收益%','hold_days':'持有交易日','observed_weeks':'已观察交易周',
 'no_signal_weeks':'无新信号周','max_daily_below5_signal_weeks':'有信号但每日均不足5只周',
 'full_year':'完整年度','selected_events':'推荐事件数','unique_stocks':'不同股票数'}

LABELS.update({'diagnostic_complete':'路径完整','mfe_low_pct':'最大浮盈下界%','mfe_high_pct':'最大浮盈上界%',
 'mae_low_pct':'最大不利波动下界%','mae_high_pct':'最大不利波动上界%','close_mdd_pct':'收盘最大回撤%',
 'w1_mfe_low_pct':'前5交易日浮盈下界%','w2_mfe_low_pct':'前10交易日浮盈下界%',
 'stop_hit5':'止损前达到5%','stop_hit10':'止损前达到10%','stop_hit2r':'止损前达到2R',
 'diagnostic_closed':'已退出且路径完整','closed_missing':'已退出但诊断缺失','open_or_unknown':'持有中或退出未知',
 'stop_count':'初始止损笔数','mfe_low_median_pct':'浮盈下界中位数%','mfe_low_mean_pct':'浮盈下界均值%',
 'mae_high_mean_pct':'不利波动上界均值%','close_mdd_mean_pct':'收盘最大回撤均值%',
 'confirmed_ge10':'确定浮盈≥10%笔数','confirmed_ge10_then_loss':'浮盈≥10%后净亏损笔数',
 'confirmed_ge10_then_loss_pct':'浮盈≥10%后亏损占完整已退出样本%',
 'threshold':'诊断门槛','category':'分类','count':'笔数','stop_total':'全部初始止损笔数','share_of_stops_pct':'占全部止损%',
 'events':'信号数','closed':'已退出','mean_hold_days':'平均持有交易日'})


def load_result(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        manifest=json.loads(z.read('manifest.json'))
        if manifest.get('version')!=VERSION:raise ValueError('只能载入 gpt1.1 结果，旧版口径不兼容')
        tables={}
        for name in z.namelist():
            if name.endswith('.csv'):
                try:tables[Path(name).stem]=pd.read_csv(z.open(name),dtype={'year':str,'ts_code':str})
                except pd.errors.EmptyDataError:tables[Path(name).stem]=pd.DataFrame()
    required={'events','weekly_summary','coverage','top5_weekly_history','diagnostic_summary','diagnostic_events'}
    if not required.issubset(tables):raise ValueError('结果文件不完整')
    return tables,manifest,payload,'已导入结果'


def main():
    import streamlit as st
    st.set_page_config(page_title='gpt1.1 科技周线选股',layout='wide')
    st.title('gpt1.1 · 科技周线选股验证')
    st.caption('价格启动＋历史上行能力筛选；前五名独立跟踪；不模拟资金组合。新规则尚未证明盈利。')
    with st.sidebar:
        st.header('研究设置')
        token=st.text_input('Tushare Token',type='password',value=os.environ.get('TUSHARE_TOKEN',''))
        start=st.date_input('信号开始',date(2022,1,1));end=st.date_input('信号结束',latest_ready_day().date())
        mode=st.selectbox('信号确认方式',['周中逐日','周收盘确认'])
        st.caption('周中逐日：使用已完成周结构，每日收盘判断。周收盘确认：仅完整周末判断。均在下一交易日开盘买入。')
        price=st.number_input('最低股价（严格大于）',min_value=0.,value=10.)
        min_mv=st.number_input('流通市值下限（亿元）',min_value=0.,value=50.)
        max_mv=st.number_input('流通市值上限（亿元）',min_value=0.,value=1000.)
        cache=st.text_input('行情缓存目录',value='tech_swing_cache')
        run=st.button('运行 gpt1.1',type='primary')
        st.caption('四路并发，复用历史行情缓存。持仓跟踪至最新已完成行情，不设固定退出期限。')
        upload=st.file_uploader('载入已完成的 gpt1.1 结果',type=['zip'])
        if st.button('载入结果',disabled=upload is None):
            try:st.session_state['gpt_result']=load_result(upload.getvalue())
            except Exception as ex:st.error(str(ex))
    if run:
        if not token:st.error('请填写 Tushare Token 后运行；也可以载入此前完成的结果。')
        elif start>end or min_mv>=max_mv:st.error('日期或市值区间不正确。')
        else:
            status=st.empty()
            try:
                cfg=Config(start=ds(start),end=ds(end),signal_mode=mode,min_price=price,min_mv=min_mv,max_mv=max_mv)
                st.session_state['gpt_result']=run_research(token,cache,cfg,status.info)
                status.success('计算完成；切换页面选项不会重新下载。')
            except Exception as ex:status.error(f'运行未完成：{ex}')
    with st.expander('固定规则与统计口径',expanded=False):st.text(STUDY_NOTES)
    if 'gpt_result' not in st.session_state:
        st.info('运行后显示每批前五名、逐周净收益与胜率、退出记录以及空窗统计。')
        return
    tables,manifest,payload,path=st.session_state['gpt_result']
    c=manifest['config']
    st.write(f"结果：{manifest['version']}｜{c['start']}—{c['end']}｜{c['signal_mode']}｜行情至 {manifest['data_end']}")
    st.download_button('下载本次完整结果',payload,file_name=f"gpt1.1_{c['start']}_{c['end']}.zip",mime='application/zip')
    for warning in manifest.get('warnings',[]):st.warning(str(warning))
    if manifest.get('data_issues',0):st.warning(f"存在 {manifest['data_issues']} 条数据问题，请查看质量表；未知结果不计为零收益。")
    def show(df):st.dataframe(df.rename(columns=LABELS),use_container_width=True,hide_index=True)
    tabs=st.tabs(['逐周主表','每批前五名','退出与对照','空窗与数据','买点与退出诊断'])
    with tabs[0]:
        st.caption('已退出事件冻结最终收益。每周先要求整批达到观察年龄，再统计；胜率分母为收益已知的已成交事件。')
        df=tables['weekly_summary']
        if df.empty:st.info('尚无成熟样本。')
        else:
            group=st.selectbox('样本组',list(dict.fromkeys(df.group)))
            year=st.selectbox('信号年份',list(dict.fromkeys(df.year.astype(str))))
            v=df[df.group.eq(group)&df.year.astype(str).eq(year)]
            show(v)
            with st.expander('辅助：仅仍持有样本（不能替代主表）'):
                aux=tables['survivor_summary'];show(aux[aux.group.eq(group)&aux.year.astype(str).eq(year)])
    with tabs[1]:
        e=tables['events']
        chosen=e[e.selected.eq(True)].copy() if not e.empty else pd.DataFrame()
        if chosen.empty:st.info('没有合格推荐，不补足五只。')
        else:
            chosen['signal_date']=pd.to_datetime(chosen.signal_date).dt.strftime('%Y-%m-%d')
            day=st.selectbox('信号批次',sorted(chosen.signal_date.unique(),reverse=True))
            batch=chosen[chosen.signal_date.eq(day)].sort_values('rank')
            st.caption('这是所选历史批次。买入、退出和各周收益属于事后跟踪，不能用于当时排名。')
            cols=['ts_code','name','rank','score','typical_up_pct','risk_pct','circ_mv_yi','initial_stop_raw','buy_date','buy_raw','risk_pct_actual','status','sell_date','exit_reason','net_pct','hold_days']
            show(batch[[x for x in cols if x in batch]])
            hist=tables['top5_weekly_history'];h=hist[hist.event_id.isin(batch.event_id)] if not hist.empty else hist
            if not h.empty:
                wide=h.pivot(index=['event_id','ts_code','name','rank'],columns='week_no',values='week_net_pct')
                wide.columns=[f'W{int(x)} 净收益%' for x in wide.columns]
                show(wide.reset_index().drop(columns='event_id'))
            st.caption('空白表示尚未成熟、未成交或路径未知，结合状态查看；已退出后的周收益保持不变。')
    with tabs[2]:
        st.caption('最终收益表仅包含已退出事件，仍持有事件另计；不能替代逐周主表。')
        show(tables['final_summary'])
        st.write('同日成熟批次对照（订单口径包含已知取消为零；未知批次不比较）')
        comp=tables['weekly_date_comparison']
        if not comp.empty:
            k=st.number_input('对照周次',min_value=1,max_value=int(comp.week_no.max()),value=1)
            show(comp[comp.week_no.eq(k)])
    with tabs[3]:
        st.caption('空窗指无新合格信号的交易周。每年≤5周是验收目标；不为达标放宽规则。非完整年度不能认定全年达标。')
        show(tables['coverage']);show(tables['filter_counts'])
        show(tables.get('data_issues',pd.DataFrame()))
        with st.expander('数据与版本记录'):st.json(manifest)

    with tabs[4]:
        st.caption('诊断不改变买卖。最大浮盈为价格涨幅，非可兑现净利润；盘中退出日按上下界处理。已退出且行情完整的交易用于主分类，持有中及未知单列。')
        diag=tables.get('diagnostic_summary',pd.DataFrame())
        if diag.empty:st.info('尚无可诊断事件。')
        else:
            dg=st.selectbox('诊断样本组',list(dict.fromkeys(diag.group)))
            dy=st.selectbox('诊断信号年份',list(dict.fromkeys(diag.year.astype(str))))
            show(diag[diag.group.eq(dg)&diag.year.astype(str).eq(dy)])
            st.write('初始止损前是否曾上涨：分母为对应组、年份的全部初始止损笔数，未知不会被归入未上涨。')
            bins=tables['stop_excursion_summary'];show(bins[bins.group.eq(dg)&bins.year.astype(str).eq(dy)])
            st.caption('净利润回吐：至少曾出现10%价格浮盈但最终净亏损；不能据此认定在最高价卖出可实现。收盘最大回撤只反映收盘路径，不等于盘中最大回撤。')
            detail=tables['diagnostic_events']
            mask=dict(groups(detail))[dg];detail=detail[mask]
            if dy!='全部':detail=detail[detail.year.astype(str).eq(dy)]
            cols=['ts_code','name','signal_date','rank','status','exit_reason','net_pct','diagnostic_complete','mfe_low_pct','mfe_high_pct','mae_low_pct','mae_high_pct','close_mdd_pct','w1_mfe_low_pct','w2_mfe_low_pct','stop_hit5','stop_hit10','stop_hit2r']
            show(detail[[x for x in cols if x in detail]])


def self_test():
    def frame(n=30):
        idx=pd.bdate_range('2024-01-01',periods=n)
        return pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,adj_factor=1.,up_limit=150.,down_limit=50.,vol=100.),index=idx)
    g=frame();g.iloc[1,g.columns.get_loc('low')]=94
    out,marks=lifecycle(g,0,95.)
    assert out['closed'] and out['sell_i']==1 and out['exit_reason']=='初始止损'
    assert abs(out['net_pct']-(95*.999*.998/(100*1.001*1.001)-1)*100)<1e-9
    g=frame();g.loc[g.index[0],'low']=94;g.loc[g.index[1],['open','low','close']]=[93,92,94]
    out,_=lifecycle(g,0,95.);assert out['sell_i']==1 and abs(out['sell_adj']-93*.999)<1e-9
    g=frame();g.loc[g.index[1],['high','close','low']]=[112,111,96]
    g.loc[g.index[2],['open','low','close']]=[110,105,108]
    out,_=lifecycle(g,0,95.);assert out['sell_i']==2 and out['exit_reason']=='移动止盈'
    assert abs(out['sell_adj']-(111-(100.1-95))*.999)<1e-9
    g=frame(125);g['high']=120
    out,marks=lifecycle(g,0,95.);assert not out['closed'] and len(marks)==25 and not any(m['trail_active'] for m in marks)
    g=frame();g.loc[g.index[1],['open','low','close','down_limit']]=[90,90,90,90]
    g.loc[g.index[2],'vol']=0;g.loc[g.index[3],['open','low','close']]=[89,88,90]
    out,_=lifecycle(g,0,95.);assert out['sell_i']==3 and out['exit_delay_days']==1
    g=frame();g.loc[g.index[7],'low']=np.nan
    out,marks=lifecycle(g,0,95.);assert out['unknown_from']==7 and len(marks)==1 and not out['closed']
    row=dict(event_id='x',base_pass=True,main_pass=True,selected=True,rank=1,mature_weeks=1,**out)
    e=pd.DataFrame([row]);m=pd.DataFrame([dict(event_id='x',**a) for a in marks])
    assert weekly_view(e,m,1).week_known.all() and weekly_view(e,m,2).empty
    row.update(closed=True,resolved=True,sell_i=1,net_pct=-5.,mature_weeks=2)
    e=pd.DataFrame([row]);assert weekly_view(e,pd.DataFrame(),2).week_net_pct.iloc[0]==-5
    row['mature_weeks']=1;e=pd.DataFrame([row]);assert weekly_view(e,pd.DataFrame(),2).empty
    g=frame(400);g['ac']=100+np.arange(400)*.1;g['ah']=g.ac+1;g['al']=g.ac-1
    f=price_setup(g,'周中逐日');prefix=price_setup(g.iloc[:387],'周中逐日')
    pd.testing.assert_frame_equal(f.iloc[:387],prefix)
    w=g.groupby(g.index.to_period('W-FRI')).ac.last();idx=70
    expected=np.median([(w.iloc[idx-1-4*j]/w.iloc[idx-5-4*j]-1)*100 for j in range(13)])
    assert abs(f.loc[g.index.to_period('W-FRI')==w.index[idx],'typical_up_pct'].iloc[0]-expected)<1e-9
    damaged=g.copy();damaged.loc[damaged.index[300],'ac']=np.nan
    assert not price_setup(damaged,'周中逐日').history_ready.iloc[-1]
    # 盘中退出当日高点不能确定发生在卖出前；开盘退出必须排除当日高点。
    g=frame();g.loc[g.index[1],['high','low']]=[130,94]
    path,_=lifecycle(g,0,95.);diag=path_diagnostic(g,path)
    assert diag['stop_hit10']=='退出日先后不明' and diag['mfe_low_pct']<1 and diag['mfe_high_pct']>29
    g.loc[g.index[1],'open']=94
    path,_=lifecycle(g,0,95.);diag=path_diagnostic(g,path)
    assert path['exit_at_open'] and diag['stop_hit10']=='确定未达'
    g=frame();g.loc[g.index[1],'high']=115;g.loc[g.index[2],'low']=94
    path,_=lifecycle(g,0,95.);diag=path_diagnostic(g,path)
    assert diag['stop_hit10']=='确定达到'
    g.loc[g.index[1],'high']=np.nan
    assert path_diagnostic(g,path)['stop_hit10']=='未知'
    print('gpt1.1 self-test PASS: stops, T+1, trailing causality, limits, missing data, maturity, frozen losses, weekly feature causality, excursion bounds and missing highs; no real-market profitability claim.')


if __name__=='__main__':
    if '--self-test' in sys.argv:self_test()
    else:main()
