# -*- coding: utf-8 -*-
"""科技波段研究 gpt1.4.1 动量、反转与历史同行比较 — streamlit run app.py

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

VERSION = "gpt1.4.1"
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
    end: str = "20260913"
    signal_mode: str = "完整周收盘"
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


def lifecycle(g,buy_i,stop=None):
    n=len(g);cal=g.index
    out=dict(initial_stop_adj=np.nan,initial_stop_raw=np.nan,filled=False,closed=False,resolved=False,status='待买入',exit_reason='',buy_i=buy_i,sell_i=-1,
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
    if not np.isfinite([op[i],ad[i],up[i],down[i],vol[i]]).all():out['status']='买入数据未知';out['unknown_from']=i;return out,marks
    if op[i]<=0 or ad[i]<=0 or down[i]<=0 or up[i]<down[i]:out['status']='买入数据异常';out['unknown_from']=i;return out,marks
    if op[i]>=up[i]-.005:cancel('开盘涨停取消');return out,marks
    buy_raw=min(op[i]*1.001,up[i]);buy=buy_raw*ad[i]
    if stop is None:stop=buy*.92
    r=buy-stop;risk=r/buy*100
    if not 2<=risk<=10:cancel('开盘风险不符取消');return out,marks
    out.update(filled=True,status='持有中',buy_date=cal[i],buy_adj=buy,buy_raw=buy_raw,risk_pct_actual=risk,r_amount=r,initial_stop_adj=stop,initial_stop_raw=stop/ad[i])
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


def make_zip(tables,manifest):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,df in tables.items():z.writestr(name+'.csv',df.to_csv(index=False).encode('utf-8-sig'))
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
        z.writestr('规则与口径.txt',STUDY_NOTES)
    return out.getvalue()



STRATEGIES={'原始动量':'mom_raw','行业调整动量':'mom_adj','短期反转':'rev_raw','行业调整反转':'rev_adj'}
RULES={
 '研究目的':'gpt1.4.1 独立比较四个单项；不混合评分，不筛选两周新高或历史上涨空间。四组共用同一可比较股票池和同一逐股交易路径，无资金组合。',
 '股票池':'历史科技股票，信号日未复权股价严格>10元，流通市值50—1000亿元，上市至少180天。价格市值可在侧栏修改。风险警示使用原有涨跌停幅度近似识别，历史行业记录仍可能不完整。',
 '周频':'每个完整交易周最后一个交易日收盘计算一次，次交易日开盘买入；不是每个工作日重复推荐。全周休市不计入交易周窗口，未完成周不发信号。',
 '动量':'C为复权周收盘价；动量=100×(C[t−2]/C[t−28]−1)，衡量26个交易周收益，跳过最近2个交易周。',
 '反转':'短期收益=100×(C[t]/C[t−2]−1)；反转分数=−短期收益，即最近2个交易周跌幅越大评分越高。单独测试，不预设强股回撤有效。',
 '行业调整':'使用信号日已知的申万二级行业归属，与同一可比较股票池、同业其他股票的窗口收益中位数比较，剔除自身。行业动量分数=个股动量−同行动量中位数；行业反转分数=同行短期收益中位数−个股短期收益。单位为百分点，不是多因子残差或行业中性持仓。',
 '共同样本':'四组共同要求29个交易周窗口内报价完整、历史二级行业唯一且可用、同业至少有10只其他股票满足这些条件。先固定共同样本，再比较四个分数，数据不足不回退为当前行业快照。',
 '推荐':'每个分数先对共同池排序，降序；同分流通市值降序、代码升序。仅分数>0者可成为该组推荐，最多前5名；不补位、不强行凑满。行业调整分数为正不代表绝对涨幅为正，更不代表未来正期望。',
 '分层':'另对共同池全部股票按评分分为5层，每周尽量等数量；分数完全相同者不拆开。第1层分数最高。分层含非正分数，不受前五名限制，用于检查排序是否有效。',
 '重复':'同股每周可再次入选，各周分别记账，可能重叠且并非统计独立；四组亦可能选中同一事件。推荐不是加仓指令，各笔收益不可相加当账户收益。',
 '止损':'四组统一在含滑点实际买价下方8%设初始止损，R=买价×8%；日内触线卖出，跳空按开盘价；买入当天触线受T+1限制延至次日可执行开盘。8%是固定研究假设，不是寻优结果。',
 '止盈':'最高收盘价相对买价达到2R后启动保护线=最高收盘价−1R，只上移，收盘上移的线次日生效；本版不使用gpt1.2两种提前退出。无固定持仓期限。',
 '成本与执行':'买入费0.10%、卖出费0.20%，每边滑点0.10%。开盘涨停或已知停牌取消买入；跌停或停牌造成卖出延迟。缺失路径单列未知，不猜测成交。日线无法恢复成交队列。',
 '周次':'W1/W2/...为买入后第5/10/...市场交易日。先要求整个信号批次达到相应观察年龄；已退出冻结实际净收益，仍持有按收盘计价并预扣卖出成本；不删除止损事件。',
 '对照':'行业调整与原始方法按同一信号日期等权比较前五名，双方该日推荐的结果都已知才比较；双方无推荐、仅一方有推荐、未知等另计。主收益使用已成交且已知事件，另列取消为0的订单口径。',
 '检验':'分层在W1/W2/W4/W8/W12观察，持有本身不限这些周数。同期全共同池作为基准。另记录行业历史强弱，判断行业调整去掉了什么；不预先承诺改善。',
 '空窗':'每年无方向合格推荐的交易周，目标≤5。每周能排序不等于有盈利机会，单靠排名覆盖率不算策略成功。',
 '历史限制':'2022—2026数据已被多次观察，本轮结果为历史研究，不能称为全新样本外。新信号和统一8%初始止损均不同于gpt1.2，不能将跨版本差异全部归因于因子。',
}
RULES['版本对照']='gpt1.4.1保留gpt1.3选股、排名、8%止损和移动保护规则。只增加事后诊断，不用诊断收益改变选股。'
RULES['无退出路径']='固定观察W1—W12（5—60个市场交易日），从同一实际含滑点买价起，忽略止损和止盈继续观察。收盘标记预扣双边费用与滑点，不是可执行卖出收益；停牌沿用最近复权收盘，末日停牌、跌停和限制数据缺失另计。'
RULES['路径幅度']='最高/最低价相对含滑点买价为毛价格幅度，含买入日；最大上涨下限为0，最大下跌上限为0。收盘最大回撤含买价起点。极值不代表可兑现利润，也不推断同日高低点顺序。报价缺失后整个后续诊断窗口记未知，不恢复成完整路径。'
RULES['退出配对']='只在已成交且无退出路径及原规则周收益均已知的同一事件比较；退出贡献=原规则收益−无退出收盘标记。分别报告原规则盈利/亏损与无退出盈利/亏损的四格计数。诊断期限不改变实际持仓期限。'
RULES['入场对照']='逐信号日期比较前五名与排除前五名后的共同池；两组全部已成交事件的诊断均完整才纳入日期均值。未成交不计收益，数量另列。同步比较最大上涨、最大下跌；不把高波动直接认定为优势。'
RULES['评分检验']='每周横截面评分与无退出标记收益、最大上涨、最大下跌的秩相关，完整已成交样本不少于10且变量有差异才计算；逐日期等权汇总。事件及观察窗重叠，不报告独立样本显著性或自动选优。'
RULES['止损先后']='gpt1.4.1新增前五名初始止损与首次上涨10%的先后分类：退出前、开盘退出当日、盘中退出当日先后未知、退出次日起才达到、观察期未达到、路径未知。只含该观察周内已实际初始止损事件；未知不从分母删除。不把止损后高点解释为持有时可获得的利润。'
STUDY_NOTES='\n'.join(f'{k}：{v}' for k,v in RULES.items())


def prepared_stock(part,calendar):
    g=part.drop_duplicates('date').set_index('date').sort_index().reindex(calendar)
    for c in ['open','high','low','close','pre_close','vol','circ_mv','turnover_rate','adj_factor','up_limit','down_limit']:
        g[c]=pd.to_numeric(g[c],errors='coerce') if c in g else np.nan
    for c,a in [('high','ah'),('low','al'),('close','ac')]:g[a]=g[c]*g.adj_factor
    return g


def complete_week_indices(calendar,full_calendar):
    """需要未来一周的已公布交易日历，只判断周是否结束，不读取未来行情。"""
    key=calendar.to_period('W-FRI');candidates=np.flatnonzero(np.r_[key[:-1]!=key[1:],True])
    full=pd.DatetimeIndex(full_calendar);out=[]
    for i in candidates:
        later=full[full>calendar[i]]
        if len(later) and later[0].to_period('W-FRI')!=key[i]:out.append(i)
        elif calendar[i].weekday()==4:out.append(i)
    return np.asarray(out,dtype=int)


def weekly_features(g,complete_i):
    key=g.index.to_period('W-FRI')
    valid=g[['ac','ah','al','vol']].notna().all(axis=1)&g.ac.gt(0)&g.al.gt(0)&g.ah.ge(g.ac)&g.al.le(g.ac)&g.vol.gt(0)
    # 停牌/缺报价不被前向填充为平盘。跨缺口的窗口不参加四组比较。
    w=g.assign(key=key,valid=valid).groupby('key').agg(close=('ac','last'),valid=('valid','all'))
    w.loc[~w.valid,'close']=np.nan
    w['history_ready']=w.valid.rolling(29,min_periods=29).sum().eq(29)
    w['mom_raw']=(w.close.shift(2)/w.close.shift(28)-1)*100
    w['short_return_pct']=(w.close/w.close.shift(2)-1)*100
    w=w.reindex(key[complete_i]).copy();w.index=g.index[complete_i]
    return w


def industry_at(intervals,dates):
    result=pd.Series('',index=dates,dtype=object);ambiguous=pd.Series(False,index=dates)
    for row in intervals.itertuples():
        label=getattr(row,'l2_name','');code=getattr(row,'l2_code','')
        if pd.isna(label) or not str(label).strip():continue
        identity=(str(code)+'|'+str(label)) if pd.notna(code) and str(code).strip() else str(label)
        active=(dates>=row.in_date)&(dates<(row.out_date if pd.notna(row.out_date) else pd.Timestamp.max))
        ambiguous|=active&result.ne('')&result.ne(identity)
        result.loc[active]=identity
    result.loc[ambiguous]=''
    return result,ambiguous


def loo_median(values):
    """剔除自己后的中位数，排序一次即可；避免小行业被自身收益污染。"""
    v=np.asarray(values,dtype=float);n=len(v)
    if n<2:return np.full(n,np.nan)
    order=np.argsort(v,kind='stable');s=v[order];r=np.empty(n);k=n//2
    if n%2==0:r[:k]=s[k];r[k:]=s[k-1]
    else:r[:k]=(s[k]+s[k+1])/2;r[k]=(s[k-1]+s[k+1])/2;r[k+1:]=(s[k-1]+s[k])/2
    out=np.empty(n);out[order]=r;return out


def score_cross_sections(features):
    f=features.copy()
    if f.empty:return f
    f['factor_ready']=f.pool_pass&f.history_ready&f.industry_key.ne('')&f[['mom_raw','short_return_pct']].notna().all(axis=1)
    ready=f[f.factor_ready]
    counts=ready.groupby(['signal_date','industry_key']).size()
    keys=pd.MultiIndex.from_frame(f[['signal_date','industry_key']])
    f['peer_count']=np.maximum(0,counts.reindex(keys).fillna(0).to_numpy()-1)
    f['common_pass']=f.factor_ready&f.peer_count.ge(10)
    for col in ['industry_mom_pct','industry_short_pct','mom_adj','rev_raw','rev_adj','industry_vs_tech_pp']:
        f[col]=np.nan
    common=f[f.common_pass]
    for _,g in common.groupby(['signal_date','industry_key']):
        f.loc[g.index,'industry_mom_pct']=loo_median(g.mom_raw)
        f.loc[g.index,'industry_short_pct']=loo_median(g.short_return_pct)
    f.loc[common.index,'mom_adj']=f.loc[common.index,'mom_raw']-f.loc[common.index,'industry_mom_pct']
    f.loc[common.index,'rev_raw']=-f.loc[common.index,'short_return_pct']
    f.loc[common.index,'rev_adj']=f.loc[common.index,'industry_short_pct']-f.loc[common.index,'short_return_pct']
    tech=f[f.common_pass].groupby('signal_date').mom_raw.median()
    f.loc[common.index,'industry_vs_tech_pp']=f.loc[common.index,'industry_mom_pct']-f.loc[common.index,'signal_date'].map(tech)
    for col in STRATEGIES.values():
        f['rank_'+col]=np.nan;f['layer_'+col]=np.nan;f['selected_'+col]=False
        order=f[f.common_pass].sort_values(['signal_date',col,'circ_mv_yi','ts_code'],ascending=[True,False,False,True])
        f.loc[order.index,'rank_'+col]=order.groupby('signal_date').cumcount()+1
        # 中位秩分层，同分不切开，避免市值成为虚假的评分信息。
        r=order.groupby('signal_date')[col].rank(ascending=False,method='average')
        size=order.groupby('signal_date')[col].transform('size')
        layer=np.minimum(5,np.floor((r-1)/size*5)+1)
        f.loc[order.index,'layer_'+col]=layer
        f['selected_'+col]=f.common_pass&f['rank_'+col].le(5)&f[col].gt(0)
    return f


def calculate(data,basic,member,calendar,cfg,progress,full_calendar=None,diagnostic_sink=None):
    full_calendar=calendar if full_calendar is None else full_calendar
    complete_i=complete_week_indices(calendar,full_calendar)
    signal_i=complete_i[(calendar[complete_i]>=stamp(cfg.start))&(calendar[complete_i]<=stamp(cfg.end))]
    if not len(signal_i):return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame({'signal_date':calendar[signal_i]})
    base=basic.set_index('ts_code');members={c:m for c,m in member.groupby('ts_code')};rows=[]
    for num,(code,part) in enumerate(data.groupby('ts_code',sort=True),1):
        if code not in base.index or code not in members:continue
        g=prepared_stock(part,calendar);eligible,known=eligibility(g,code,base.loc[code],members[code],calendar,cfg)
        w=weekly_features(g,complete_i).reindex(calendar[signal_i]);ind,amb=industry_at(members[code],calendar[signal_i])
        z=pd.DataFrame(dict(event_id=[code+'|'+ds(d) for d in calendar[signal_i]],ts_code=code,name=base.loc[code,'name'],
            signal_date=calendar[signal_i],signal_i=signal_i,year=calendar[signal_i].year.astype(str),
            signal_week=calendar[signal_i].to_period('W-FRI').astype(str),pool_pass=eligible.iloc[signal_i].to_numpy(),
            pool_known=known.iloc[signal_i].to_numpy(),industry_key=ind.to_numpy(),industry_ambiguous=amb.to_numpy(),
            circ_mv_yi=g.circ_mv.iloc[signal_i].to_numpy()/10000,signal_close=g.close.iloc[signal_i].to_numpy(),
            history_ready=w.history_ready.fillna(False).to_numpy(),mom_raw=w.mom_raw.to_numpy(),short_return_pct=w.short_return_pct.to_numpy()))
        rows.append(z)
        if num%50==0:progress(f'周收益与历史行业 {num}/{len(base)}')
    f=score_cross_sections(pd.concat(rows,ignore_index=True)) if rows else pd.DataFrame()
    if f.empty:return f,pd.DataFrame(),f,pd.DataFrame({'signal_date':calendar[signal_i]})
    common=f[f.common_pass].copy();paths=[];marks=[];bycode={c:g for c,g in common.groupby('ts_code')}
    for num,(code,part) in enumerate(data.groupby('ts_code',sort=True),1):
        if code not in bycode:continue
        g=prepared_stock(part,calendar)
        arrays=diagnostic_arrays(g) if diagnostic_sink is not None else None
        stock_diagnostics=[]
        for row in bycode[code].itertuples():
            path,weekly=lifecycle(g,int(row.signal_i)+1)
            if diagnostic_sink is not None:
                stock_diagnostics.extend(dict(event_id=row.event_id,**x) for x in unbounded_path(arrays,path))
            paths.append(dict(event_id=row.event_id,**path))
            marks.extend(dict(event_id=row.event_id,**x) for x in weekly)
        if diagnostic_sink is not None and stock_diagnostics:
            diagnostic_sink.append(pd.DataFrame(stock_diagnostics))
        if num%25==0:progress(f'共用退出路径 {num}/{len(base)}；不重复计算四组重合股票')
    if paths:
        e=common.merge(pd.DataFrame(paths),on='event_id',validate='one_to_one')
        e['base_pass']=True;e['mature_weeks']=np.maximum(0,(len(calendar)-e.buy_i)//5)
    else:e=pd.DataFrame()
    return e,pd.DataFrame(marks),f,pd.DataFrame({'signal_date':calendar[signal_i]})


def period_stat(g):
    v=g.loc[g.filled,'week_net_pct']
    return dict(mature_events=len(g),filled=int(g.filled.sum()),known_filled=int(v.notna().sum()),
        exited=int(g.exited.sum()),holding=int(g.holding.sum()),cancelled=int(g.cancelled.sum()),unknown=int((~g.week_known).sum()),
        mean_order_pct=g.loc[g.week_known,'week_order_pct'].mean(),**distribution(v))


def years(frame):
    yield '全部',frame
    yield from frame.groupby('year')


def date_comparison(view,schedule,w,family,raw,adjusted):
    out=schedule[['signal_date']].copy();out['year']=out.signal_date.dt.year.astype(str)
    for tag,col in [('raw',raw),('adjusted',adjusted)]:
        chosen=view[view['selected_'+col]]
        g=chosen.groupby('signal_date').agg(count=('event_id','size'),known=('week_known','sum'),filled=('filled','sum'),mean=('week_order_pct','mean'))
        for field in ['count','known','filled','mean']:out[tag+'_'+field]=out.signal_date.map(g[field])
        for field in ['count','known','filled']:out[tag+'_'+field]=out[tag+'_'+field].fillna(0).astype(int)
    both=out.raw_count.gt(0)&out.adjusted_count.gt(0)
    known=out.raw_count.eq(out.raw_known)&out.adjusted_count.eq(out.adjusted_known)
    out['comparison_status']=np.select([both&known,both, out.raw_count.gt(0),out.adjusted_count.gt(0)],
        ['可比较','结果未知','仅原始有推荐','仅调整有推荐'],default='双方无推荐')
    out['delta_pp']=(out.adjusted_mean-out.raw_mean).where(both&known)
    out['family']=family;out['week_no']=w
    return out


def build_reports(e,marks,f,schedule,calendar,cfg,progress=lambda _:None):
    names=['events','weekly_marks','factor_candidates','selections','weekly_summary','survivor_summary','final_summary',
        'rank_layers','industry_adjustment_dates','industry_adjustment_summary','coverage','pool_filter_counts',
        'industry_profile','top5_weekly_history','signal_calendar']
    tables={k:pd.DataFrame() for k in names};tables.update(events=e,weekly_marks=marks,factor_candidates=f,signal_calendar=schedule)
    coverage=[];filters=[]
    for yr in sorted(schedule.signal_date.dt.year.unique()):
        days=schedule[schedule.signal_date.dt.year.eq(yr)].signal_date;yr=str(yr)
        pool=f[f.year.eq(yr)] if not f.empty else f
        for strategy,col in STRATEGIES.items():
            chosen=pool[pool.get('selected_'+col,pd.Series(False,index=pool.index)).eq(True)]
            count=chosen.groupby('signal_date').size() if not chosen.empty else pd.Series(dtype=int)
            coverage.append(dict(strategy=strategy,year=yr,observed_weeks=len(days),no_signal_weeks=int((~days.isin(count.index)).sum()),
                fewer_than5_signal_weeks=int(count.lt(5).sum()),selected_events=len(chosen),
                full_year=stamp(cfg.start)<=pd.Timestamp(int(yr),1,1) and stamp(cfg.end)>=pd.Timestamp(int(yr),12,31) and latest_ready_day()>=pd.Timestamp(int(yr),12,31)))
        if not pool.empty:
            filters.append(dict(year=yr,stock_week_records=len(pool),pool_pass=int(pool.pool_pass.sum()),
                history_ready=int((pool.pool_pass&pool.history_ready).sum()),industry_ambiguous=int(pool.industry_ambiguous.sum()),
                factor_ready=int(pool.factor_ready.sum()),common_pass=int(pool.common_pass.sum())))
    tables['coverage']=pd.DataFrame(coverage);tables['pool_filter_counts']=pd.DataFrame(filters)
    if e.empty:return tables
    c=f[f.common_pass]
    tables['industry_profile']=c.groupby(['signal_date','industry_key']).agg(stocks=('ts_code','size'),
        mom_median_pct=('mom_raw','median'),short_median_pct=('short_return_pct','median'),industry_vs_tech_pp=('industry_vs_tech_pp','median')).reset_index()
    selected=[];summary=[];survivors=[];final=[];layers=[];dates=[];history=[]
    for strategy,col in STRATEGIES.items():
        chosen=e[e['selected_'+col]].copy();chosen['strategy']=strategy;chosen['rank']=chosen['rank_'+col];chosen['score']=chosen[col]
        selected.append(chosen)
        for group,g in [('前五名',chosen)]+[(f'第{r}名',chosen[chosen['rank'].eq(r)]) for r in range(1,6)]:
            for yr,p in years(g):
                done=p[p.closed]
                final.append(dict(strategy=strategy,group=group,year=yr,events=len(p),filled=int(p.filled.sum()),closed=len(done),
                    unknown_or_open=int((~p.resolved).sum()),cancelled=int((p.resolved&~p.filled).sum()),
                    mean_hold_days=done.hold_days.mean(),**distribution(done.net_pct)))
    tables['selections']=pd.concat(selected,ignore_index=True);tables['final_summary']=pd.DataFrame(final)
    for w in range(1,int(e.mature_weeks.max())+1):
        if w%10==0:progress(f'逐周固定样本与同行比较 W{w}')
        v=weekly_view(e,marks,w)
        if v.empty:continue
        for yr,p in years(v):summary.append(dict(strategy='共同池基准',group='全部',year=yr,week_no=w,**period_stat(p)))
        for strategy,col in STRATEGIES.items():
            chosen=v[v['selected_'+col]].copy()
            chosen['strategy']=strategy;chosen['rank']=chosen['rank_'+col]
            cols=['event_id','ts_code','name','signal_date','year','strategy','rank','filled','exited','holding','cancelled','week_known','week_net_pct','week_order_pct']
            history.append(chosen[cols].assign(week_no=w))
            for group,g in [('前五名',chosen)]+[(f'第{r}名',chosen[chosen['rank'].eq(r)]) for r in range(1,6)]:
                for yr,p in years(g):
                    identity=dict(strategy=strategy,group=group,year=yr,week_no=w)
                    summary.append(dict(identity,**period_stat(p)))
                    alive=p[p.holding];survivors.append(dict(identity,holding_events=len(alive),**distribution(alive.week_net_pct)))
            if w in [1,2,4,8,12]:
                for layer in range(1,6):
                    g=v[v['layer_'+col].eq(layer)]
                    for yr,p in years(g):layers.append(dict(strategy=strategy,layer=layer,year=yr,week_no=w,**period_stat(p)))
        mature=schedule[calendar.searchsorted(schedule.signal_date)+1+5*w<=len(calendar)]
        for family,raw,adj in [('动量','mom_raw','mom_adj'),('反转','rev_raw','rev_adj')]:
            dates.append(date_comparison(v,mature,w,family,raw,adj))
    tables.update(weekly_summary=pd.DataFrame(summary),survivor_summary=pd.DataFrame(survivors),rank_layers=pd.DataFrame(layers),
        top5_weekly_history=pd.concat(history,ignore_index=True) if history else pd.DataFrame())
    d=pd.concat(dates,ignore_index=True) if dates else pd.DataFrame();comparisons=[]
    if not d.empty:
        for (family,w),whole in d.groupby(['family','week_no']):
            for yr,p in years(whole):
                valid=p[p.comparison_status.eq('可比较')]
                comparisons.append(dict(family=family,year=yr,week_no=w,mature_dates=len(p),paired_dates=len(valid),
                    unknown_dates=int(p.comparison_status.eq('结果未知').sum()),raw_only_dates=int(p.comparison_status.eq('仅原始有推荐').sum()),
                    adjusted_only_dates=int(p.comparison_status.eq('仅调整有推荐').sum()),empty_dates=int(p.comparison_status.eq('双方无推荐').sum()),
                    raw_date_mean_pct=valid.raw_mean.mean(),adjusted_date_mean_pct=valid.adjusted_mean.mean(),
                    mean_delta_pp=valid.delta_pp.mean(),median_delta_pp=valid.delta_pp.median(),
                    adjusted_better_dates_pct=valid.delta_pp.gt(0).mean()*100 if len(valid) else np.nan))
    tables.update(industry_adjustment_dates=d,industry_adjustment_summary=pd.DataFrame(comparisons))
    return tables


DIAGNOSTIC_WEEKS=tuple(range(1,13))


def diagnostic_arrays(g):
    a={k:g[k].to_numpy(dtype=float) for k in ['open','high','low','close','adj_factor','vol','down_limit']}
    o,h,l,c,ad,v=[a[k] for k in ['open','high','low','close','adj_factor','vol']]
    a['suspended']=np.isfinite(v)&(v<=0)
    valid=np.isfinite(np.column_stack([o,h,l,c,ad,v])).all(axis=1)
    valid&=(np.column_stack([o,h,l,c,ad])>0).all(axis=1)
    valid&=(h>=np.maximum.reduce([o,c,l]))&(l<=np.minimum.reduce([o,c,h]))
    a['bad']=~a['suspended']&~valid
    for col in ['high','low','close']:a['adjusted_'+col]=a[col]*ad
    return a


def stop_up10_category(path,end_i,first_i,known):
    if not (path['closed'] and path['exit_reason']=='初始止损' and path['sell_i']<=end_i):
        return '观察期内未初始止损'
    if not known:return '路径未知'
    if first_i<0 or first_i>end_i:return '观察期未达到10%'
    sell_i=path['sell_i']
    if first_i<sell_i:return '退出前已达到10%'
    if first_i>sell_i:return '退出次日起才达到10%'
    return '开盘退出当日达到10%' if path.get('exit_at_open',False) else '盘中退出当日先后未知'


def unbounded_path(a,path):
    """Independent fixed-horizon observation; vectorized without future signal input."""
    b=int(path['buy_i']);n=len(a['close']);last_w=min(12,(n-b)//5)
    if last_w<1:return []
    length=last_w*5;buy=path['buy_adj'];sl=slice(b,b+length)
    suspended=a['suspended'][sl];damaged=np.maximum.accumulate(a['bad'][sl])
    up=np.full(length,np.nan);down=up.copy();mdd=up.copy();last=up.copy();first=-1
    if path['filled']:
        # A known suspension carries the previous observed adjusted close; gaps remain unknown.
        cl=a['adjusted_close'][sl];indices=np.maximum.accumulate(np.where(~suspended,np.arange(length),-1))
        last=np.r_[buy,cl][indices+1]
        hi=np.where(suspended,buy,a['adjusted_high'][sl]);lo=np.where(suspended,buy,a['adjusted_low'][sl])
        up=(np.maximum.accumulate(np.r_[buy,hi])[1:]/buy-1)*100
        down=(np.minimum.accumulate(np.r_[buy,lo])[1:]/buy-1)*100
        peaks=np.maximum.accumulate(np.r_[buy,last])[1:]
        mdd=np.minimum.accumulate(np.minimum(0.,(last/peaks-1)*100))
        hits=np.flatnonzero(~suspended&~damaged&(a['adjusted_high'][sl]>=buy*1.1))
        if len(hits):first=b+int(hits[0])
    counts=np.cumsum(suspended & ~damaged);rows=[]
    for w in range(1,last_w+1):
        k=5*w-1;j=b+k;known=bool(path['filled'] and not damaged[k])
        if not path['filled']:state='未成交'
        elif not known:state='路径未知'
        elif suspended[k]:state='停牌标记'
        elif not np.isfinite(a['down_limit'][j]) or a['down_limit'][j]<=0:state='卖出限制数据未知'
        elif a['close'][j]<=a['down_limit'][j]+.005:state='跌停标记'
        else:state='正常报价标记'
        rows.append(dict(week_no=w,path_known=known,path_status=state,
            suspended_days=int(counts[k]) if path['filled'] else 0,
            unbounded_net_pct=(last[k]*.999*.998/(buy*1.001)-1)*100 if known else np.nan,
            max_up_pct=up[k] if known else np.nan,max_down_pct=down[k] if known else np.nan,
            close_mdd_pct=mdd[k] if known else np.nan,
            stop_up10_timing=stop_up10_category(path,j,first,known)))
    return rows


def entry_stat(g):
    valid=g[g.path_known & g.filled];pair=valid[valid.week_net_pct.notna()]
    un=pair.unbounded_net_pct;rule=pair.week_net_pct;diff=rule-un
    pct=lambda v:float(v.mean()*100) if len(v) else np.nan
    return dict(mature_events=len(g),filled=int(g.filled.sum()),path_known_count=len(valid),
        path_unknown_count=int((g.filled & ~g.path_known).sum()),not_filled_count=int((~g.filled).sum()),
        restricted_marks=int(valid.path_status.ne('正常报价标记').sum()),paired_count=len(pair),
        unbounded_mean_pct=valid.unbounded_net_pct.mean(),unbounded_win_pct=pct(valid.unbounded_net_pct.gt(0)),
        unbounded_median_pct=valid.unbounded_net_pct.median(),
        unbounded_trim_top1_pct=distribution(valid.unbounded_net_pct)['trim_top1_mean_pct'],
        unbounded_ge10_pct=pct(valid.unbounded_net_pct.ge(10)),unbounded_ge20_pct=pct(valid.unbounded_net_pct.ge(20)),
        max_up_mean_pct=valid.max_up_pct.mean(),max_up_median_pct=valid.max_up_pct.median(),
        max_up_ge10_pct=pct(valid.max_up_pct.ge(10)),max_up_ge20_pct=pct(valid.max_up_pct.ge(20)),
        max_down_mean_pct=valid.max_down_pct.mean(),max_down_le8_pct=pct(valid.max_down_pct.le(-8)),
        close_mdd_mean_pct=valid.close_mdd_pct.mean(),
        paired_unbounded_pct=un.mean(),paired_rule_pct=rule.mean(),exit_contribution_pp=diff.mean(),
        exit_helped_pct=pct(diff.gt(1e-9)),exit_hurt_pct=pct(diff.lt(-1e-9)),
        rule_loss_path_profit=int((rule.lt(0)&un.gt(0)).sum()),
        rule_profit_path_loss=int((rule.gt(0)&un.lt(0)).sum()),
        both_profit=int((rule.gt(0)&un.gt(0)).sum()),both_loss=int((rule.lt(0)&un.lt(0)).sum()),
        either_zero=int((rule.eq(0)|un.eq(0)).sum()))


def entry_exit_reports(e,marks,paths,progress=lambda _:None):
    names=['path_diagnostics','entry_exit_summary','entry_rank_dates','entry_rank_summary',
           'entry_date_comparison','entry_date_summary','entry_top5_details','stop_timing_summary']
    out={k:pd.DataFrame() for k in names};out['path_diagnostics']=paths
    if paths.empty or e.empty:return out
    if paths.duplicated(['event_id','week_no']).any():raise ValueError('诊断主键重复')
    summaries=[];rank_dates=[];dates=[];details=[];stop_timing=[]
    for w in DIAGNOSTIC_WEEKS:
        progress(f'入场与退出配对诊断 W{w}/12')
        v=weekly_view(e,marks,w)
        if v.empty:continue
        v=v.merge(paths[paths.week_no.eq(w)],on='event_id',how='left',validate='one_to_one')
        if v.path_known.isna().any():raise ValueError('成熟事件缺少诊断记录，不能当作零收益')
        for yr,p in years(v):summaries.append(dict(strategy='共同池基准',group='全部',year=yr,week_no=w,**entry_stat(p)))
        for strategy,col in STRATEGIES.items():
            selected=v['selected_'+col];chosen=v[selected]
            groups=[('前五名',chosen)]+[(f'第{r}名',chosen[chosen['rank_'+col].eq(r)]) for r in range(1,6)]
            groups += [(f'评分第{layer}层',v[v['layer_'+col].eq(layer)]) for layer in range(1,6)]
            for group,g in groups:
                for yr,p in years(g):summaries.append(dict(strategy=strategy,group=group,year=yr,week_no=w,**entry_stat(p)))
            stopped=chosen[chosen.exited & chosen.exit_reason.eq('初始止损')]
            for yr,p in years(stopped):
                for category,count in p.stop_up10_timing.value_counts().items():
                    stop_timing.append(dict(strategy=strategy,year=yr,week_no=w,category=category,
                        count=int(count),stop_total=len(p),share_of_stops_pct=count/len(p)*100))
            cols=['event_id','ts_code','name','signal_date','year','buy_date','filled','exited','week_net_pct',
                  'sell_date','exit_reason','stop_up10_timing',
                  'week_no','path_known','path_status','unbounded_net_pct','max_up_pct','max_down_pct','close_mdd_pct']
            d=chosen[cols].copy();d['strategy']=strategy;d['rank']=chosen['rank_'+col];details.append(d)
            for day,g in v.groupby('signal_date'):
                q=g[g.filled & g.path_known];rec=dict(strategy=strategy,year=str(day.year),week_no=w,signal_date=day,
                    filled=int(g.filled.sum()),path_known_count=len(q))
                for field,target in [('return_ic','unbounded_net_pct'),('up_ic','max_up_pct'),('down_ic','max_down_pct')]:
                    rec[field]=q[col].rank().corr(q[target].rank()) if len(q)>=10 and q[col].nunique()>1 and q[target].nunique()>1 else np.nan
                rank_dates.append(rec)
                top=g[g['selected_'+col]];rest=g[~g['selected_'+col]]
                tf=top[top.filled];rf=rest[rest.filled]
                valid=len(tf)>0 and len(rf)>0 and tf.path_known.all() and rf.path_known.all()
                rec=dict(strategy=strategy,year=str(day.year),week_no=w,signal_date=day,top_events=len(top),
                    top_filled=len(tf),rest_filled=len(rf),top_unknown=int((~tf.path_known).sum()),rest_unknown=int((~rf.path_known).sum()),
                    comparison_status='可比较' if valid else ('无推荐' if top.empty else '未成交或路径未知'))
                for field in ['unbounded_net_pct','max_up_pct','max_down_pct']:
                    rec['top_'+field]=tf[field].mean() if valid else np.nan
                    rec['rest_'+field]=rf[field].mean() if valid else np.nan
                    rec['delta_'+field]=rec['top_'+field]-rec['rest_'+field]
                dates.append(rec)
    rank_dates=pd.DataFrame(rank_dates);dates=pd.DataFrame(dates);rs=[];dsum=[]
    if not rank_dates.empty:
        for (strategy,w),g in rank_dates.groupby(['strategy','week_no']):
            for yr,p in years(g):
                r=dict(strategy=strategy,week_no=w,year=yr,mature_dates=len(p))
                for field in ['return_ic','up_ic','down_ic']:
                    q=p[field].dropna();r[field+'_dates']=len(q);r[field+'_mean']=q.mean();r[field+'_positive_pct']=q.gt(0).mean()*100 if len(q) else np.nan
                rs.append(r)
    if not dates.empty:
        for (strategy,w),g in dates.groupby(['strategy','week_no']):
            for yr,p in years(g):
                q=p[p.comparison_status.eq('可比较')]
                r=dict(strategy=strategy,week_no=w,year=yr,mature_dates=len(p),paired_dates=len(q),excluded_dates=len(p)-len(q))
                for field in ['unbounded_net_pct','max_up_pct','max_down_pct']:
                    for prefix in ['top_','rest_','delta_']:r[prefix+field]=q[prefix+field].mean()
                r['top_better_dates_pct']=q.delta_unbounded_net_pct.gt(1e-9).mean()*100 if len(q) else np.nan
                dsum.append(r)
    out.update(stop_timing_summary=pd.DataFrame(stop_timing),entry_exit_summary=pd.DataFrame(summaries),entry_rank_dates=rank_dates,entry_rank_summary=pd.DataFrame(rs),
        entry_date_comparison=dates,entry_date_summary=pd.DataFrame(dsum),
        entry_top5_details=pd.concat(details,ignore_index=True) if details else pd.DataFrame())
    return out


def run_research(token,cache_root,cfg,progress):
    client=DataClient(token,cache_root,progress);basic,member,pool_mode,pool_warnings=client.universe()
    if '快照' in pool_mode or 'l2_name' not in member or member.l2_name.fillna('').str.strip().eq('').all():
        raise RuntimeError('本版行业对照需要历史二级行业区间，当前仅有行业快照或字段缺失。请恢复index_member_all权限/数据后重试；已缓存行情可复用。')
    ready=latest_ready_day()
    if min(stamp(cfg.end),ready)<stamp(cfg.start):raise ValueError('尚未进入指定信号区间')
    start=ds(stamp(cfg.start)-pd.Timedelta(days=450));end=ds(ready)
    full=client.calendar(start,ds(ready+pd.Timedelta(days=14)));calendar=full[full<=ready]
    data,issues=client.download(calendar,set(basic.ts_code))
    hashed=pd.util.hash_pandas_object(data,index=False).to_numpy();hashed.sort();data_hash=hashlib.sha256(hashed.tobytes()).hexdigest()
    diagnostic_rows=[]
    e,marks,f,schedule=calculate(data,basic,member,calendar,cfg,progress,full,diagnostic_rows);del data;gc.collect()
    progress('汇总四种评分的前五名、全池分层和空窗')
    tables=build_reports(e,marks,f,schedule,calendar,cfg,progress);tables.update(data_issues=issues,universe=basic,industry_intervals=member)
    diagnostic_frame=pd.concat(diagnostic_rows,ignore_index=True) if diagnostic_rows else pd.DataFrame()
    del diagnostic_rows;gc.collect()
    tables.update(entry_exit_reports(e,marks,diagnostic_frame,progress))
    manifest=dict(version=VERSION,config=asdict(cfg),rules=RULES,created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        data_start=start,data_end=ds(calendar.max()),data_hash=data_hash,pool_hash=hashlib.sha256(member.to_csv(index=False).encode()).hexdigest(),
        pool_mode=pool_mode,warnings=pool_warnings,data_issues=len(issues),universe_size=len(basic),download_workers=DOWNLOAD_WORKERS,
        last_signal_date=str(schedule.signal_date.max().date()) if not schedule.empty else None,
        limitations=['行业调整为同行收益中位数调整，不是回归残差或中性组合','方向合格不等于正期望；覆盖率不能单独证明有效',
            '共同样本要求报价完整及足够同行，可能产生数据可用性选择','独立事件可重叠，不能当账户收益',
            '行业区间和历史风险警示仍有供应商限制','四组统一8%初始止损，跨旧版比较不是单变量实验','已经反复使用的历史不是新样本外'])
    payload=make_zip(tables,manifest);run_id=hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()[:12]
    path=Path(cache_root)/'results'/f'{VERSION}_{run_id}.zip';atomic_bytes(payload,path)
    return tables,manifest,payload,str(path)

LABELS={'strategy':'方法','group':'样本组','year':'信号年份','week_no':'观察周次','mature_events':'成熟事件数',
 'filled':'已成交','known_filled':'收益已知成交数','exited':'已退出','holding':'仍持有','cancelled':'取消','unknown':'未知',
 'mean_net_pct':'平均净收益%','median_net_pct':'净收益中位数%','win_pct':'净盈利胜率%',
 'net_ge10_pct':'净收益≥10%占比','net_ge20_pct':'净收益≥20%占比','p10_net_pct':'收益10分位%',
 'trim_top1_mean_pct':'剔除最高1%后均值%','mean_order_pct':'含取消订单均值%',
 'ts_code':'代码','name':'名称','signal_date':'信号确认日','rank':'排名','score':'分数（%或百分点）',
 'industry_key':'历史二级行业','peer_count':'其他同行数','mom_raw':'26周动量%','short_return_pct':'最近2周收益%',
 'mom_adj':'行业调整动量百分点','rev_raw':'反转分数%','rev_adj':'行业调整反转百分点',
 'industry_mom_pct':'同行26周收益中位数%','industry_short_pct':'同行2周收益中位数%',
 'industry_vs_tech_pp':'同行动量减科技池中位数百分点','circ_mv_yi':'流通市值亿元','signal_close':'信号收盘价',
 'initial_stop_raw':'买入时初始止损价','buy_date':'买入日','buy_raw':'含滑点买入价','sell_date':'退出日',
 'status':'状态','exit_reason':'退出原因','net_pct':'最终净收益%','hold_days':'持有交易日',
 'observed_weeks':'已完成交易周','no_signal_weeks':'无方向合格推荐周','fewer_than5_signal_weeks':'有推荐但不足5只周',
 'full_year':'完整年度','selected_events':'推荐事件数','events':'事件数','closed':'已退出','unknown_or_open':'未结清或未知',
 'mean_hold_days':'平均持有交易日','layer':'评分层（1最高）','family':'因子类别','mature_dates':'成熟批次',
 'paired_dates':'可比较批次','unknown_dates':'结果未知批次','raw_only_dates':'仅原始有推荐','adjusted_only_dates':'仅调整有推荐',
 'empty_dates':'双方无推荐','raw_date_mean_pct':'原始方法同日订单均值%','adjusted_date_mean_pct':'调整方法同日订单均值%',
 'mean_delta_pp':'调整减原始平均差额百分点','median_delta_pp':'差额中位数百分点','adjusted_better_dates_pct':'调整方法胜出批次%',
 'stock_week_records':'股票周记录数','pool_pass':'价格市值等基础合格','history_ready':'基础合格且历史完整',
 'industry_ambiguous':'行业区间冲突','factor_ready':'基础历史行业可用','common_pass':'同行数充足的共同样本',
 'holding_events':'仍持有数量','stocks':'同行股票数','mom_median_pct':'行业26周收益中位数%','short_median_pct':'行业2周收益中位数%'}


LABELS.update({'path_known': '诊断路径完整', 'path_status': '末日标记状态', 'path_known_count': '诊断完整成交数', 'path_unknown_count': '诊断未知成交数', 'not_filled_count': '未成交数', 'restricted_marks': '末日受限或限制未知数', 'paired_count': '同事件配对数', 'unbounded_net_pct': '无退出标记净收益%', 'unbounded_mean_pct': '无退出标记均益%', 'unbounded_win_pct': '无退出标记胜率%', 'unbounded_median_pct': '无退出标记中位数%', 'unbounded_trim_top1_pct': '无退出剔除最高1%后均益%', 'unbounded_ge10_pct': '无退出净收益≥10%占比', 'unbounded_ge20_pct': '无退出净收益≥20%占比', 'max_up_pct': '最大上涨毛幅度%', 'max_down_pct': '最大下跌毛幅度%', 'close_mdd_pct': '收盘最大回撤%', 'max_up_mean_pct': '最大上涨均值%', 'max_up_median_pct': '最大上涨中位数%', 'max_up_ge10_pct': '曾上涨≥10%占比', 'max_up_ge20_pct': '曾上涨≥20%占比', 'max_down_mean_pct': '最大下跌均值%', 'max_down_le8_pct': '曾下跌≥8%占比', 'close_mdd_mean_pct': '收盘最大回撤均值%', 'paired_unbounded_pct': '配对无退出标记均益%', 'paired_rule_pct': '配对原规则均益%', 'exit_contribution_pp': '退出贡献百分点', 'exit_helped_pct': '退出改善占比%', 'exit_hurt_pct': '退出降低收益占比%', 'rule_loss_path_profit': '原规则亏损且无退出盈利数', 'rule_profit_path_loss': '原规则盈利且无退出亏损数', 'both_profit': '两者均盈利数', 'both_loss': '两者均亏损数', 'either_zero': '任一恰好零收益数', 'excluded_dates': '不可比较日期数', 'top_better_dates_pct': '前五胜出日期%', 'week_net_pct': '原规则周净收益%', 'return_ic_dates': '收益有效日期数', 'return_ic_mean': '收益平均秩相关', 'return_ic_positive_pct': '收益正相关日期%', 'up_ic_dates': '最大上涨有效日期数', 'up_ic_mean': '最大上涨平均秩相关', 'up_ic_positive_pct': '最大上涨正相关日期%', 'down_ic_dates': '最大下跌有效日期数', 'down_ic_mean': '最大下跌平均秩相关', 'down_ic_positive_pct': '最大下跌正相关日期%', 'top_unbounded_net_pct': '前五名无退出净收益%', 'rest_unbounded_net_pct': '其余共同池无退出净收益%', 'delta_unbounded_net_pct': '前五减其余无退出净收益百分点', 'top_max_up_pct': '前五名最大上涨%', 'rest_max_up_pct': '其余共同池最大上涨%', 'delta_max_up_pct': '前五减其余最大上涨百分点', 'top_max_down_pct': '前五名最大下跌%', 'rest_max_down_pct': '其余共同池最大下跌%', 'delta_max_down_pct': '前五减其余最大下跌百分点'})

LABELS.update({'stop_up10_timing':'初始止损与上涨10%先后','category':'先后分类','count':'事件数','stop_total':'当周已初始止损总数','share_of_stops_pct':'占初始止损事件%','sell_date':'实际退出日','exit_reason':'原规则退出原因'})

def load_result(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        manifest=json.loads(z.read('manifest.json'))
        if manifest.get('version') not in [VERSION,'gpt1.4']:raise ValueError('本页支持gpt1.4及gpt1.4.1结果；更早版本缺少诊断数据')
        tables={}
        for name in z.namelist():
            if name.endswith('.csv'):
                try:tables[Path(name).stem]=pd.read_csv(z.open(name),dtype={'year':str,'ts_code':str})
                except pd.errors.EmptyDataError:tables[Path(name).stem]=pd.DataFrame()
    required={'events','weekly_summary','selections','coverage','rank_layers','industry_adjustment_summary','signal_calendar','top5_weekly_history','path_diagnostics','entry_exit_summary','entry_rank_summary','entry_date_comparison'}
    required.update(['survivor_summary','final_summary','industry_adjustment_dates','pool_filter_counts',
                     'industry_profile','entry_date_summary','entry_top5_details'])
    if manifest.get('version')==VERSION:required.add('stop_timing_summary')
    if not required.issubset(tables):raise ValueError('结果文件缺少必要表格：'+', '.join(sorted(required-set(tables))))
    tables.setdefault('stop_timing_summary',pd.DataFrame())
    return tables,manifest,payload,'已载入结果'


def main():
    import streamlit as st
    st.set_page_config(page_title='gpt1.4.1 动量与反转验证',layout='wide')
    st.title('gpt1.4.1 · 入场优势与退出规则拆分诊断')
    st.caption('四个单项独立验证；周收盘选股，次交易日开盘执行。前五名是研究候选，尚未证明盈利。')
    with st.sidebar:
        st.header('研究设置')
        token=st.text_input('Tushare Token',type='password',value=os.environ.get('TUSHARE_TOKEN',''))
        start=st.date_input('信号开始',date(2022,1,1));end=st.date_input('信号结束',latest_ready_day().date())
        price=st.number_input('最低股价（严格大于）',min_value=0.,value=10.)
        min_mv=st.number_input('流通市值下限（亿元）',min_value=0.,value=50.)
        max_mv=st.number_input('流通市值上限（亿元）',min_value=0.,value=1000.)
        cache=st.text_input('行情缓存目录',value='tech_swing_cache')
        run=st.button('运行 gpt1.4.1',type='primary')
        st.caption('四路下载并复用缓存；四组重合股票共用一次退出计算。历史二级行业数据必须可用。')
        upload=st.file_uploader('载入gpt1.4或gpt1.4.1结果',type=['zip'])
        if st.button('载入结果',disabled=upload is None):
            try:st.session_state['gpt141_result']=load_result(upload.getvalue())
            except Exception as ex:st.error(str(ex))
    if run:
        if not token:st.error('请填写Tushare Token，或载入已完成的结果。')
        elif start>end or min_mv>=max_mv:st.error('请检查日期或市值范围。')
        else:
            status=st.empty()
            try:
                cfg=Config(start=ds(start),end=ds(end),min_price=price,min_mv=min_mv,max_mv=max_mv)
                st.session_state['gpt141_result']=run_research(token,cache,cfg,status.info)
                status.success('计算完成。')
            except Exception as ex:status.error(f'本次未完成：{ex}')
    with st.expander('本版固定规则与统计口径'):st.text(STUDY_NOTES)
    if 'gpt141_result' not in st.session_state:
        st.info('本版统一8%初始止损，保留2R启动及回撤1R保护；不设固定持仓期限。运行后比较四组及评分分层。');return
    tables,manifest,payload,path=st.session_state['gpt141_result'];cfg=manifest['config']
    st.write(f"程序 {VERSION}｜结果 {manifest['version']}｜信号 {cfg['start']}—{cfg['end']}｜行情至 {manifest['data_end']}")
    st.download_button('下载完整结果',payload,file_name=f"{manifest['version']}_{cfg['start']}_{cfg['end']}.zip",mime='application/zip')
    for warning in manifest.get('warnings',[]):st.warning(str(warning))
    if manifest.get('data_issues',0):st.warning('存在行情接口或合并问题，未知交易不计为零，请查看数据质量。')
    st.caption('方向合格和高覆盖率不能证明有效；行业调整也不保证绝对收益为正。')
    strategy=st.selectbox('研究方法',list(STRATEGIES))
    cv=tables['coverage'];options=['全部']+sorted(cv.year.astype(str).unique().tolist()) if not cv.empty else ['全部']
    year=st.selectbox('信号年份',options)
    def subset(df):
        if df.empty:return df
        out=df[df.strategy.eq(strategy)] if 'strategy' in df else df
        return out[out.year.astype(str).eq(year)] if 'year' in out else out
    def show(df):st.dataframe(df.rename(columns=LABELS),use_container_width=True,hide_index=True)
    tabs=st.tabs(['逐周收益','前五名明细','行业调整对照','排序分层','覆盖与数据','入场与退出诊断'])
    with tabs[0]:
        group=st.selectbox('样本组',['前五名']+[f'第{r}名' for r in range(1,6)])
        df=subset(tables['weekly_summary'])
        if df.empty:st.info('尚无成熟事件。')
        else:show(df[df.group.eq(group)])
        st.caption('已退出冻结实际收益；仍持有按收盘标记。远期周次只含已成熟批次，不能解释为同一批股票越来越赚钱。')
        with st.expander('同期共同池基准'):
            base=tables['weekly_summary']
            if not base.empty:show(base[base.strategy.eq('共同池基准')&base.year.astype(str).eq(year)])
        with st.expander('仅仍持有样本（辅助）'):
            aux=subset(tables['survivor_summary'])
            if not aux.empty:show(aux[aux.group.eq(group)])
        with st.expander('最终收益：只含已退出，不能代替逐周表'):
            final=subset(tables['final_summary'])
            if not final.empty:show(final[final.group.eq(group)])
    with tabs[1]:
        dates=pd.to_datetime(tables['signal_calendar'].signal_date)
        if year!='全部':dates=dates[dates.dt.year.astype(str).eq(year)]
        if dates.empty:st.info('所选年份无完整周批次。')
        else:
            day=st.selectbox('完整周信号批次',sorted(dates.dt.strftime('%Y-%m-%d').unique(),reverse=True))
            s=tables['selections']
            batch=s[s.strategy.eq(strategy)&pd.to_datetime(s.signal_date).dt.strftime('%Y-%m-%d').eq(day)].sort_values('rank') if not s.empty else s
            if batch.empty:st.info('本批次没有方向合格推荐，不补足五只。')
            else:
                cols=['ts_code','name','rank','score','industry_key','peer_count','mom_raw','short_return_pct','industry_mom_pct','industry_short_pct','circ_mv_yi','buy_date','buy_raw','initial_stop_raw','status','sell_date','exit_reason','net_pct','hold_days']
                show(batch[cols]);h=tables['top5_weekly_history']
                h=h[h.strategy.eq(strategy)&h.event_id.isin(batch.event_id)] if not h.empty else h
                if not h.empty:
                    cap=st.number_input('显示周次上限（只影响表格）',min_value=1,max_value=int(h.week_no.max()),value=min(12,int(h.week_no.max())))
                    wide=h[h.week_no.le(cap)].pivot(index=['ts_code','name','rank'],columns='week_no',values='week_net_pct')
                    wide.columns=[f'W{int(w)} 净收益%' for w in wide.columns];show(wide.reset_index())
            st.caption('最新批次来自已完成周。事后买入、退出及收益不参与信号排名；空白结合取消、待买入或未知状态解读。')
    with tabs[2]:
        st.caption('按同一日期比较双方前五名，日期等权。双方推荐不同，这是选股对照；各股买卖规则相同。含已知取消为0，未知批次排除。')
        family='反转' if '反转' in strategy else '动量';d=tables['industry_adjustment_summary']
        if not d.empty:show(d[d.family.eq(family)&d.year.astype(str).eq(year)])
        with st.expander('同日比较明细'):
            d=tables['industry_adjustment_dates']
            if not d.empty:
                d=d[d.family.eq(family)]
                if year!='全部':d=d[d.year.astype(str).eq(year)]
                show(d[d.week_no.isin([1,2,4,8,12])])
    with tabs[3]:
        st.caption('全共同池按分数分五层，第1层最高，含非正分数；相同分数不拆分。观察评分是否越高收益越好，避免只看五只极端股票。')
        w=st.selectbox('分层观察周次',[1,2,4,8,12]);d=subset(tables['rank_layers'])
        if not d.empty:show(d[d.week_no.eq(w)])
    with tabs[4]:
        st.caption('四组使用同一数据可用股票池。空窗目标为每年≤5个完整交易周无方向合格推荐；不是资金空仓时间。')
        cov=cv[cv.strategy.eq(strategy)] if not cv.empty else cv
        if year!='全部' and not cov.empty:cov=cov[cov.year.astype(str).eq(year)]
        show(cov);show(tables['pool_filter_counts']);show(tables.get('data_issues',pd.DataFrame()))
        with st.expander('历史行业强弱记录'):show(tables['industry_profile'])
        with st.expander('版本与数据记录'):st.json(manifest)


    with tabs[5]:
        st.caption('无退出路径只用于诊断入场，不是新交易策略。W1—W12为固定观察期；原规则仍按止损/止盈退出，没有12周强制卖出。')
        dg=st.selectbox('诊断样本组',['前五名']+[f'第{r}名' for r in range(1,6)]+[f'评分第{i}层' for i in range(1,6)])
        d=subset(tables['entry_exit_summary'])
        if d.empty:st.info('尚无成熟的诊断样本。')
        else:
            d=d[d.group.eq(dg)]
            st.write('入场后的价格路径：不执行止损止盈，仍扣设定费用与滑点')
            show(d[['week_no','mature_events','filled','path_known_count','path_unknown_count','not_filled_count','restricted_marks',
                'unbounded_mean_pct','unbounded_win_pct','unbounded_median_pct','unbounded_trim_top1_pct',
                'unbounded_ge10_pct','unbounded_ge20_pct','max_up_mean_pct','max_up_ge10_pct','max_up_ge20_pct',
                'max_down_mean_pct','max_down_le8_pct','close_mdd_mean_pct']])
            st.write('同一事件比较：原规则收益减无退出标记收益，正值表示退出规则帮助减少损失或保留收益')
            show(d[['week_no','paired_count','paired_unbounded_pct','paired_rule_pct','exit_contribution_pp',
                'exit_helped_pct','exit_hurt_pct','rule_loss_path_profit','rule_profit_path_loss','both_profit','both_loss','either_zero']])
        st.caption('最大上涨/下跌是毛价格幅度，不是可兑现利润。末日停牌、跌停及卖出限制数据未知仍是账面标记，另列数量。缺失路径不补零。')
        with st.expander('初始止损与上涨10%的先后关系',expanded=True):
            d=tables['stop_timing_summary']
            if d.empty:st.info('没有可分类的初始止损事件；gpt1.4旧结果不含此诊断，需使用本版重新运行。')
            else:show(subset(d))
            st.caption('只含各观察周内已经实际初始止损的前五名事件，未知也保留在分母。开盘退出日的高点不能算作退出前利润；盘中退出日高低点先后不明时单列。达到10%指毛价格幅度，不代表可以兑现。')
        with st.expander('共同池无退出路径基准'):
            d=tables['entry_exit_summary']
            if not d.empty:show(d[d.strategy.eq('共同池基准')&d.year.astype(str).eq(year)])
        with st.expander('前五名与其余共同池：同日、日期等权'):
            show(subset(tables['entry_date_summary']))
            st.caption('前五名从基准中剔除；只比较双方已成交且全部路径完整的日期。差额为前五减其余。最大下跌差额为正表示下跌较少。排除日期数量须同时审阅。')
        with st.expander('评分与未来价格的秩相关'):
            show(subset(tables['entry_rank_summary']))
            st.caption('收益相关为正才支持高分对应较高收益；上涨相关为正但下跌相关为负可能只代表波动更大。无显著性检验；历史已反复观察。')
        with st.expander('前五名逐股诊断'):
            dw=st.selectbox('逐股诊断周次',list(DIAGNOSTIC_WEEKS))
            d=tables['entry_top5_details']
            if not d.empty:
                d=d[d.strategy.eq(strategy)&d.week_no.eq(dw)]
                if year!='全部':d=d[d.year.astype(str).eq(year)]
                show(d)


def diagnostic_self_test():
    def frame(n=65):
     return pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,adj_factor=1.,up_limit=150.,down_limit=50.,vol=100.),index=pd.bdate_range('2024-01-01',periods=n))
    g=frame();g.loc[g.index[1],['open','low','close']]=[90,89,90];g.loc[g.index[2]:,['open','high','low','close']]=[115,121,114,120]
    path,marks=lifecycle(g,0);d=unbounded_path(diagnostic_arrays(g),path)
    assert path['closed'] and path['net_pct']<0 and len(d)==12 and d[0]['unbounded_net_pct']>19
    assert np.isclose(d[0]['max_up_pct'],(121/100.1-1)*100)
    assert np.isclose(d[0]['max_down_pct'],(89/100.1-1)*100)
    assert np.isclose(d[0]['close_mdd_pct'],(90/100.1-1)*100)
    assert np.isclose(d[0]['unbounded_net_pct'],(120*.999*.998/(100.1*1.001)-1)*100)
    # Future mutation cannot affect earlier diagnostic horizon.
    x=g.copy();x.loc[x.index[5]:,['open','high','low','close']]=[50,60,40,50]
    assert unbounded_path(diagnostic_arrays(x),path)[0]==d[0]
    # Missing data after the original stop must invalidate diagnostic horizons, not original frozen returns.
    x=g.copy();x.loc[x.index[7],'high']=np.nan;q=unbounded_path(diagnostic_arrays(x),path)
    assert q[0]['path_known'] and all(not r['path_known'] for r in q[1:])
    # Suspension endpoint carries last observed adjusted close, not a made-up sale.
    x=g.copy();x.loc[x.index[4],['vol','open','high','low','close']]=[0,np.nan,np.nan,np.nan,np.nan]
    q=unbounded_path(diagnostic_arrays(x),path);assert q[0]['path_known'] and q[0]['path_status']=='停牌标记'
    x=g.copy();x.loc[x.index[4],'down_limit']=120
    assert unbounded_path(diagnostic_arrays(x),path)[0]['path_status']=='跌停标记'
    x.loc[x.index[4],'down_limit']=np.nan
    assert unbounded_path(diagnostic_arrays(x),path)[0]['path_status']=='卖出限制数据未知'
    # Not filled is not a zero-profit diagnostic; insufficient age is not included.
    x=frame();x['up_limit']=100;p,_=lifecycle(x,0)
    q=unbounded_path(diagnostic_arrays(x),p);assert all(not r['path_known'] and np.isnan(r['unbounded_net_pct']) for r in q)
    assert not unbounded_path(diagnostic_arrays(g.iloc[:4]),path)
    # Economically identical split-adjusted prices produce identical diagnostics.
    x=g.copy();x.loc[x.index[2]:,['open','high','low','close','down_limit','up_limit']]/=2;x.loc[x.index[2]:,'adj_factor']=2
    pd.testing.assert_frame_equal(pd.DataFrame(unbounded_path(diagnostic_arrays(x),path)),pd.DataFrame(d))
    print('New diagnostic path boundary tests PASS',flush=True)

def stop_timing_self_test():
    def frame():
     return pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,adj_factor=1.,up_limit=150.,down_limit=50.,vol=100.),index=pd.bdate_range('2024-01-01',periods=65))
    def category(g):
     path,_=lifecycle(g,0);return unbounded_path(diagnostic_arrays(g),path)[0]['stop_up10_timing']
    g=frame();g.iloc[1,g.columns.get_loc('high')]=115;g.iloc[2,g.columns.get_loc('low')]=90
    assert category(g)=='退出前已达到10%'
    g=frame();g.iloc[1,g.columns.get_loc('low')]=90;g.iloc[2,g.columns.get_loc('high')]=115
    assert category(g)=='退出次日起才达到10%'
    g=frame();g.loc[g.index[1],['high','low']]=[115,90]
    assert category(g)=='盘中退出当日先后未知'
    g.loc[g.index[1],'open']=90
    assert category(g)=='开盘退出当日达到10%'
    g=frame();g.iloc[1,g.columns.get_loc('low')]=90
    assert category(g)=='观察期未达到10%'
    g.iloc[3,g.columns.get_loc('high')]=np.nan
    assert category(g)=='路径未知'
    g=frame();g.iloc[7,g.columns.get_loc('low')]=90;p,_=lifecycle(g,0);d=unbounded_path(diagnostic_arrays(g),p)
    assert d[0]['stop_up10_timing']=='观察期内未初始止损' and d[1]['stop_up10_timing']=='观察期未达到10%'

def self_test():
    stop_timing_self_test()
    diagnostic_self_test()
    rng=np.random.default_rng(7)
    for n in [1,2,3,4,11,12,31]:
        v=rng.integers(-5,6,size=n).astype(float)
        expected=np.array([np.median(np.delete(v,i)) if n>1 else np.nan for i in range(n)])
        np.testing.assert_allclose(loo_median(v),expected,equal_nan=True)
    full=pd.bdate_range('2022-01-03',periods=420);cal=full[:410]
    g=pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,adj_factor=1.,up_limit=150.,down_limit=50.,vol=100.),index=cal)
    g['ac']=100+np.arange(len(g))*.1;g['ah']=g.ac+1;g['al']=g.ac-1
    idx=complete_week_indices(cal,full);f=weekly_features(g,idx)
    cutoff=387;subidx=complete_week_indices(cal[:cutoff],full);p=weekly_features(g.iloc[:cutoff],subidx)
    pd.testing.assert_frame_equal(f.reindex(p.index),p)
    w=g.groupby(g.index.to_period('W-FRI')).ac.last();j=60
    day=cal[idx[j]]
    assert np.isclose(f.loc[day,'mom_raw'],(w.iloc[j-2]/w.iloc[j-28]-1)*100)
    assert np.isclose(f.loc[day,'short_return_pct'],(w.iloc[j]/w.iloc[j-2]-1)*100)
    g.loc[cal[300],'ac']=np.nan
    assert not weekly_features(g,idx).history_ready.loc[day]
    dates=pd.DatetimeIndex(['2023-01-20','2023-02-20','2023-03-20'])
    intervals=pd.DataFrame(dict(in_date=pd.to_datetime(['2023-01-01','2023-02-01']),out_date=pd.to_datetime(['2023-02-01',None]),l2_name=['软件','半导体'],l2_code=['A','B']))
    ind,amb=industry_at(intervals,dates);assert ind.tolist()==['A|软件','B|半导体','B|半导体'] and not amb.any()
    overlap=pd.concat([intervals,pd.DataFrame(dict(in_date=[pd.Timestamp('2023-03-01')],out_date=[pd.NaT],l2_name=['通信'],l2_code=['C']))],ignore_index=True)
    ind,amb=industry_at(overlap,dates);assert amb.iloc[-1] and ind.iloc[-1]==''
    t=pd.DataFrame(dict(signal_date=[pd.Timestamp('2023-01-20')]*12,ts_code=[str(i) for i in range(12)],
        industry_key=['A']*12,pool_pass=True,history_ready=True,mom_raw=np.arange(12.),short_return_pct=np.arange(12.)-6,circ_mv_yi=np.arange(12.)+50))
    scored=score_cross_sections(t)
    assert scored.common_pass.all() and scored.peer_count.eq(11).all()
    assert np.isclose(scored.mom_adj.iloc[0],0-np.median(np.arange(1.,12.)))
    assert scored.selected_mom_raw.sum()==5 and scored.selected_rev_raw.sum()==5
    ties=t.copy();ties['mom_raw']=0.;ties['short_return_pct']=0.
    q=score_cross_sections(ties);assert not q[[f'selected_{c}' for c in STRATEGIES.values()]].any().any()
    assert q.layer_mom_raw.nunique()==1
    assert not score_cross_sections(t.iloc[:10]).common_pass.any()
    g=pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,adj_factor=1.,up_limit=150.,down_limit=50.,vol=100.),index=full[:20])
    g.loc[g.index[1],'low']=92
    out,_=lifecycle(g,0);assert out['sell_i']==1 and np.isclose(out['risk_pct_actual'],8)
    assert np.isclose(out['initial_stop_raw'],100.1*.92)
    g.loc[g.index[0],'low']=90;g.loc[g.index[1],['open','low','close']]=[91,90,92]
    out,_=lifecycle(g,0);assert out['sell_i']==1 and np.isclose(out['sell_adj'],91*.999)
    print('gpt1.4.1 self-test PASS: leave-one-out peers, historical industry intervals, no future weekly prices, score ties, small industries, frozen 8% risk and T+1. No profitability claim.')


if __name__=='__main__':
    if '--self-test' in sys.argv:self_test()
    else:main()
