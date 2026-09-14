# -*- coding: utf-8 -*-
"""科技波段研究 gpt1.10 活跃度与收缩突破入场验证 — streamlit run app.py

单文件；依赖 pandas、numpy、streamlit、tushare。python app.py --self-test 可离线验算。
策略阈值不是回测寻优结果。历史统计不构成策略有效或实盘合格证明。
gpt1.10：活跃度与收缩突破入场条件，统一效率排名，未验证盈利。
只对A池生成交易路径，诊断子组逐个读取，行情缓存和四路下载保持兼容。
可用gpt1.9结果加同批本地行情缓存离线验证，沿用原截止日、价格与费用，不需要下载行情。
部署：用本文件内容覆盖Streamlit实际入口（例如ycjsb.py），保持原缓存目录。
依赖仅需 pandas、numpy、streamlit、tushare；不新增第三方依赖。
官方数据字段：https://tushare.pro/document/2?doc_id=32 / 183 / 335
"""
from __future__ import annotations

import gc
import ctypes
import subprocess
import gzip
import sqlite3
import shutil
import tempfile
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
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
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

VERSION = "gpt1.10"
RUN_SUFFIX = ('_'+hashlib.sha256(os.environ['TECH_RESEARCH_JOB_TAG'].encode()).hexdigest()[:12]
              if os.environ.get('TECH_RESEARCH_JOB_TAG') else '')
DOWNLOAD_REVISION = "DL4_DISK_V1"
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


# gpt1.7.1: bounded memory infrastructure; no additional package dependencies.
MARKET_COLUMNS = ['ts_code','trade_date','open','high','low','close','pre_close','vol','amount',
                  'circ_mv','turnover_rate','adj_factor','up_limit','down_limit']
ENDPOINTS = ('daily','daily_basic','adj_factor','stk_limit')


def memory_mb():
    try:
        values = {p[0].rstrip(':'): int(p[1])/1024 for line in Path('/proc/self/status').read_text().splitlines()
                  if (p := line.split()) and p[0] in ('VmRSS:', 'VmHWM:')}
        return values
    except (OSError, ValueError):
        return {}


def release_memory():
    before=memory_mb();gc.collect();trim=False
    try:
        libc=ctypes.CDLL(None);fn=libc.malloc_trim
        fn.argtypes=[ctypes.c_size_t];fn.restype=ctypes.c_int;trim=bool(fn(0))
    except (AttributeError,OSError):pass
    return dict(before=before,after=memory_mb(),malloc_trim=trim)


def new_job(root,mode,upload=None,token=None,cfg=None):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    job=Path(tempfile.mkdtemp(prefix='job110_',dir=root));os.chmod(job,0o700)
    try:
        config=dict(mode=mode,cache_root=str(root),config=asdict(cfg) if cfg is not None else None)
        if upload is not None:
            source=job/'source.zip'
            with source.open('wb') as out:upload.seek(0);shutil.copyfileobj(upload,out,length=1024*1024)
            config['source']=str(source)
        if token:config['token']=token
        atomic_bytes(json.dumps(config).encode(),job/'job.json');os.chmod(job/'job.json',0o600)
        return dict(directory=str(job),phase='staged',before_upload_release=memory_mb())
    except BaseException:shutil.rmtree(job,ignore_errors=True);raise


def start_worker(job):
    root=Path(job['directory']);job['parent_cleanup']=release_memory()
    env=os.environ.copy()
    env['TECH_RESEARCH_JOB_TAG']=root.name
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
    with (root/'worker.log').open('wb') as log:
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(root/'job.json')],
                                 stdout=log,stderr=log,env=env)
    job['process']=process;job['phase']='running';job['parent_after_launch']=memory_mb()


def worker_main(config_path):
    config_path=Path(config_path);root=config_path.parent;config={}
    try:
        config=json.loads(config_path.read_text());config_path.unlink()
        def progress(message):
            atomic_bytes(json.dumps(dict(message=message,child_memory=memory_mb()),ensure_ascii=False).encode(),root/'progress.json')
        if config['mode']=='replay':result=replay_research(config['source'],config['cache_root'],progress)
        elif config['mode']=='fresh':result=run_research(config['token'],config['cache_root'],Config(**config['config']),progress)
        else:raise ValueError('未知计算模式')
        atomic_bytes(json.dumps(dict(ok=True,path=result[3],child_after_pack=memory_mb())).encode(),root/'result.json')
        return 0
    except Exception as exc:
        message=str(exc)
        if config.get('token'):message=message.replace(config['token'],'***')
        atomic_bytes(json.dumps(dict(ok=False,error=message,child_memory=memory_mb()),ensure_ascii=False).encode(),root/'result.json')
        return 1
    finally:
        config_path.unlink(missing_ok=True);(root/'source.zip').unlink(missing_ok=True)


def finish_worker(job):
    process=job['process'];code=process.poll()
    if code is None:return None
    root=Path(job['directory']);metadata=root/'result.json'
    data=json.loads(metadata.read_text()) if metadata.exists() else dict(ok=False,error=f'计算进程退出，退出码 {code}；已完成的行情缓存可复用。')
    if not data.get('ok') or code!=0:raise RuntimeError(data.get('error',f'计算进程退出码 {code}'))
    audit={k:v for k,v in job.items() if k not in ('process','phase','directory')}
    audit.update(child_after_pack=data['child_after_pack'],parent_after_child_exit=release_memory(),child_exit_code=code)
    path=Path(data['path']);temp=path.with_suffix('.memory.tmp')
    try:
        shutil.copyfile(path,temp)
        with zipfile.ZipFile(temp,'a',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('process_memory.json',json.dumps(audit,ensure_ascii=False,indent=2))
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)
    return load_result(path)


def render_job(st):
    # Let one complete full render finish without the uploader before launching.
    @st.fragment(run_every=2)
    def monitor():
        job=st.session_state.get('gpt110_job')
        if job is None:return
        if job['phase']=='staged':
            job['phase']='ready';st.info('上传已保存，正在释放上传页面；随后自动开始计算。');return
        try:
            if job['phase']=='ready':start_worker(job)
            root=Path(job['directory']);p=root/'progress.json'
            info=json.loads(p.read_text()) if p.exists() else {'message':'计算进程正在启动'}
            st.info(info['message']);st.caption(f"页面内存 {memory_mb()}；计算进程 {info.get('child_memory',{})}")
            if st.button('停止本次计算（保留已下载行情）'):
                job['process'].terminate()
                try:job['process'].wait(timeout=3)
                except subprocess.TimeoutExpired:job['process'].kill();job['process'].wait(timeout=3)
                shutil.rmtree(root,ignore_errors=True);st.session_state.pop('gpt110_job',None);release_memory();st.rerun()
            result=finish_worker(job)
            if result is not None:
                st.session_state['gpt110_result']=result;st.session_state.pop('gpt110_job',None)
                shutil.rmtree(root,ignore_errors=True);st.rerun()
        except Exception as exc:
            st.error(f'本次未完成：{exc}。行情缓存保留，可直接重试。')
            if st.button('返回运行页面'):
                shutil.rmtree(job['directory'],ignore_errors=True);st.session_state.pop('gpt110_job',None);st.rerun()
    monitor()


def compact_frame(df):
    # Keep all price/return/score floats at float64. Only repeated text is encoded.
    for col in df.select_dtypes(include=['object']).columns:
        if len(df) > 20 and df[col].nunique(dropna=False) < len(df)/2:
            if df[col].dropna().map(lambda x: isinstance(x, str)).all():
                df[col] = df[col].astype('category')
    return df


def connect_disk(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=30)
    db.execute('PRAGMA cache_size=-2048')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA mmap_size=0')
    return db


@contextmanager
def research_lock(root):
    # Streamlit reruns/multiple browser sessions must not launch duplicate large jobs.
    import fcntl
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root/'research.lock').open('a') as handle:
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('已有任务正在运行，请等待当前任务结束；重复点击不会开启第二次下载。') from None
        try: yield
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class MarketStore:
    """One persistent tech-only copy, indexed by stock; one atomic commit per day."""
    def __init__(self, path, calendar, codes):
        self.path=Path(path); self.db=connect_disk(path)
        self.calendar=pd.DatetimeIndex(calendar); self.codes=sorted(codes)
        self.scope=hashlib.sha256('\n'.join(self.codes).encode()).hexdigest()
        self.db.execute('CREATE TABLE IF NOT EXISTS bars (ts_code TEXT, trade_date TEXT, '+
                        ','.join(c+' REAL' for c in MARKET_COLUMNS[2:])+', PRIMARY KEY(ts_code,trade_date)) WITHOUT ROWID')
        self.db.execute('CREATE INDEX IF NOT EXISTS bars_day ON bars(trade_date)')
        self.db.execute('CREATE TABLE IF NOT EXISTS days (day TEXT PRIMARY KEY, scope TEXT, complete INTEGER, issues TEXT, digest TEXT)')
        self.db.commit()

    def completed(self):
        return {d for d, in self.db.execute('SELECT day FROM days WHERE scope=? AND complete=1', (self.scope,))}

    def put(self, day, frame, issues):
        frame=frame.reindex(columns=MARKET_COLUMNS).sort_values('ts_code')
        digest=hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
        rows=frame.astype(object).where(frame.notna(),None).itertuples(index=False,name=None)
        with self.db:
            self.db.execute('DELETE FROM bars WHERE trade_date=?', (day,))
            self.db.executemany('INSERT INTO bars VALUES ('+','.join('?' for _ in MARKET_COLUMNS)+')',rows)
            # Missing joined fields are also retried. They never become silently complete.
            self.db.execute('INSERT OR REPLACE INTO days VALUES (?,?,?,?,?)',
                            (day,self.scope,int(not issues),json.dumps(issues,ensure_ascii=False),digest))

    def groupby(self, column, sort=True, observed=True):
        assert column=='ts_code'
        start,end=ds(self.calendar[0]),ds(self.calendar[-1])
        for code in self.codes:
            part=pd.read_sql_query('SELECT * FROM bars WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date',
                                   self.db,params=(code,start,end))
            if not part.empty:
                part['date']=pd.to_datetime(part.trade_date,format='%Y%m%d')
                yield code,part

    def audit(self):
        wanted={ds(d) for d in self.calendar}; issues=[]; hashes=[]
        for day,scope,problem,digest in self.db.execute('SELECT day,scope,issues,digest FROM days ORDER BY day'):
            if day in wanted and scope==self.scope:
                issues.extend(json.loads(problem)); hashes.append(day+':'+digest)
        return pd.DataFrame(issues,columns=['date','endpoint','problem']),hashlib.sha256('\n'.join(hashes).encode()).hexdigest()

    def close(self): self.db.close()


class SQLSpool:
    """Temporary report rows, fetched for only the requested observation week."""
    def __init__(self, path):
        self.db=connect_disk(path); self.schema=None; self.count=0

    @property
    def empty(self): return self.count==0

    def append(self, frame):
        if frame.empty:return
        if self.schema is None:
            self.schema={c:str(t) for c,t in frame.dtypes.items()}
        frame.to_sql('rows',self.db,if_exists='append',index=False,chunksize=1000)
        self.count+=len(frame)

    def finish(self):
        if not self.empty:
            # Enforce uniqueness without materializing millions of diagnostic rows.
            self.db.execute('CREATE UNIQUE INDEX IF NOT EXISTS rows_key ON rows(week_no,event_id)')
            self.db.commit()
        return self

    def restore(self, f):
        for c,t in (self.schema or {}).items():
            if t=='bool':f[c]=f[c].astype(bool)
            elif t.startswith('datetime64'):f[c]=pd.to_datetime(f[c])
        return compact_frame(f)

    def week(self, w):
        if self.empty:return pd.DataFrame()
        return self.restore(pd.read_sql_query('SELECT * FROM rows WHERE week_no=?',self.db,params=(int(w),)))

    def chunks(self):
        if self.empty:yield pd.DataFrame();return
        for f in pd.read_sql_query('SELECT * FROM rows',self.db,chunksize=5000):yield self.restore(f)

    def close(self):self.db.close()


def week_rows(frame,w):
    return frame.week(w) if isinstance(frame,SQLSpool) else frame[frame.week_no.eq(w)]


class CSVSpool:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.unlink(missing_ok=True);self.count=0
    def append(self,df):
        if df.empty:return
        # Appended gzip members are a valid streaming gzip file.
        df.to_csv(self.path,mode='a',index=False,header=self.count==0,compression='gzip',chunksize=2000)
        self.count+=len(df)
    def finish(self):
        if not self.count:pd.DataFrame().to_csv(self.path,index=False,compression='gzip')
        return self


class DiskTables(MutableMapping):
    """CSV report files, with no DataFrame cache."""
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.files={}
    def __len__(self):return len(self.files)
    def __iter__(self):return iter(self.files)
    def __delitem__(self,k):self.files.pop(k).unlink(missing_ok=True)
    def __getitem__(self,k):
        try:return pd.read_csv(self.files[k],dtype={'year':str,'ts_code':str})
        except pd.errors.EmptyDataError:return pd.DataFrame()
    def __setitem__(self,k,v):
        if isinstance(v,CSVSpool):self.files[k]=v.finish().path;return
        path=self.root/(k+'.csv.gz');tmp=path.with_suffix('.tmp')
        with gzip.open(tmp,'wt',encoding='utf-8-sig',newline='') as handle:
            if isinstance(v,SQLSpool):
                first=True
                for frame in v.chunks():
                    frame.to_csv(handle,index=False,header=first,chunksize=2000);first=False
            else:v.to_csv(handle,index=False,chunksize=2000)
        os.replace(tmp,path);self.files[k]=path


class ZipTables(Mapping):
    """Completed results: lazy reads from ZIP, never retain all CSVs in a session."""
    def __init__(self,path):
        self.path=str(path)
        with zipfile.ZipFile(path) as z:
            self.names={Path(n).stem:n for n in z.namelist() if n.endswith('.csv')}
    def __iter__(self):return iter(self.names)
    def __len__(self):return len(self.names)
    def __getitem__(self,key):return self.select(key)
    def select(self,key,**filters):
        chunks=[]
        with zipfile.ZipFile(self.path) as z, z.open(self.names[key]) as handle:
            try:
                for f in pd.read_csv(handle,dtype={'year':str,'ts_code':str},chunksize=5000):
                    for col,value in filters.items():
                        if isinstance(value,(list,set,tuple,np.ndarray,pd.Series)):f=f[f[col].isin(value)]
                        else:f=f[f[col].eq(value)]
                    if len(f):chunks.append(f)
            except pd.errors.EmptyDataError:return pd.DataFrame()
        return compact_frame(pd.concat(chunks,ignore_index=True)) if chunks else pd.DataFrame()


def write_zip(tables,manifest,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for name in tables:
            with z.open(name+'.csv','w',force_zip64=True) as target:
                if isinstance(tables,DiskTables):
                    with gzip.open(tables.files[name],'rb') as source:shutil.copyfileobj(source,target,length=1024*1024)
                else:
                    with io.TextIOWrapper(target,encoding='utf-8-sig',newline='') as text:tables[name].to_csv(text,index=False,chunksize=2000)
            print(f'[{VERSION}] 已打包 {name} | 内存 {memory_mb()}',flush=True)
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
        z.writestr('规则与口径.txt',STUDY_NOTES)
    os.replace(tmp,path)
    return path


class DataClient:
    """交易日批量下载，只缓存科技池；四路有界任务，已提交日期可恢复。"""
    def __init__(self, token, root, progress=lambda text: None):
        import tushare as ts
        self.pro = ts.pro_api(token.strip(), timeout=25)
        self.root = Path(root) / CACHE_SCHEMA
        self.progress = progress
        self.lock = threading.Lock()
        self.rate_states = {}
        self.active_codes = None
        self.scope = None

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
        legacy = self.root / endpoint / (day + ".csv.gz")
        path = self.root / 'pending_tech' / str(self.scope) / endpoint / (day + '.csv.gz') if self.scope else legacy
        required = set(fields[endpoint].split(","))
        for cached in dict.fromkeys([path,legacy]):
            if cached.exists():
                try:
                    frame = load_csv(cached)
                    valid=required.issubset(frame) and frame.trade_date.astype(str).eq(day).all() and not frame.ts_code.duplicated().any()
                    if valid and (len(frame) or cached!=legacy):
                        if self.active_codes is not None:frame=frame[frame.ts_code.isin(self.active_codes)].copy()
                        # Preserve endpoint successes even when another endpoint for this day fails.
                        if cached==legacy and path!=legacy:atomic_csv(frame,path)
                        return frame
                except (OSError,ValueError,pd.errors.ParserError):
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
        if self.active_codes is not None:frame=frame[frame.ts_code.isin(self.active_codes)].copy()
        atomic_csv(frame, path)
        return frame

    def download(self, calendar, codes):
        self.active_codes=set(codes)
        store=MarketStore(self.root/'tech_bars_v1.sqlite',calendar,codes);self.scope=store.scope
        completed=store.completed();todo=[ds(day) for day in calendar if ds(day) not in completed]
        self.progress(f'行情日期 {len(calendar)}；已提交 {len(calendar)-len(todo)}；需要下载/修复 {len(todo)}')
        # Only preflight an unfinished date; fully cached reruns make no market API calls.
        if todo:
            for endpoint in ENDPOINTS:
                try:self.day_endpoint(endpoint,todo[-1])
                except RuntimeError as exc:
                    if '认证' in str(exc) or '权限' in str(exc):store.close();raise

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

        iterator=iter(todo);n=len(calendar)-len(todo)
        try:
            with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
                pending={}
                def submit_one():
                    day=next(iterator,None)
                    if day is not None:pending[executor.submit(fetch,day)]=day
                for _ in range(DOWNLOAD_WORKERS):submit_one()
                while pending:
                    done,_=wait(pending,return_when=FIRST_COMPLETED)
                    for future in done:
                        day=pending.pop(future);frame,errors=future.result()
                        store.put(day,frame,errors);n+=1
                        # Remove duplicate raw caches only AFTER the replacement transaction commits.
                        for endpoint in ENDPOINTS:
                            (self.root/endpoint/(day+'.csv.gz')).unlink(missing_ok=True)
                            if not errors:
                                (self.root/'pending_tech'/self.scope/endpoint/(day+'.csv.gz')).unlink(missing_ok=True)
                        del frame
                        self.progress(f'四路下载/读取 {n}/{len(calendar)} 日；当日已落盘；内存 {memory_mb().get("VmRSS",0):.0f} MB')
                    done.clear()  # Futures retain their return values until released.
                    del future, errors
                    while len(pending)<DOWNLOAD_WORKERS:
                        before=len(pending);submit_one()
                        if len(pending)==before:break
            issues,_=store.audit()
            if not store.db.execute('SELECT 1 FROM bars LIMIT 1').fetchone():
                raise RuntimeError('没有可用行情；成功端点缓存保留，可重新运行补齐')
            return store,issues
        except BaseException:
            store.close();raise


def latest_ready_day():
    now=datetime.now(ZoneInfo('Asia/Shanghai'))
    return pd.Timestamp(now.date()-timedelta(days=1 if now.hour<18 else 0))


def eligibility(g, code, info, intervals, calendar, cfg):
    active=np.zeros(len(calendar),dtype=bool)
    for m in intervals.itertuples():
        active|=(calendar>=m.in_date)&(calendar<(m.out_date if pd.notna(m.out_date) else pd.Timestamp.max))
    active&=calendar>=pd.Timestamp(stamp(info.list_date).to_pydatetime()+timedelta(days=180))
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
    cols=['event_id','ts_code','name','signal_date','year','filled','closed','resolved','buy_i','sell_i',
          'net_pct','mature_weeks','base_pass','buy_date','sell_date','exit_reason']
    for col in STRATEGIES.values():cols += [col,'rank_'+col,'layer_'+col,'selected_'+col,'eligible_'+col]
    v=e.loc[e.base_pass&e.mature_weeks.ge(w),[c for c in cols if c in e]].copy()
    if v.empty:return v
    v['mark_i']=v.buy_i+5*w-1
    v['exited']=v.closed&v.sell_i.le(v.mark_i)
    v['cancelled']=v.resolved&~v.filled
    if marks.empty:mp=pd.Series(dtype=float)
    else:mp=week_rows(marks,w).set_index('event_id').net_pct
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



LEGACY_STRATEGIES={'原始动量':'mom_raw','行业调整动量':'mom_adj','短期反转':'rev_raw','行业调整反转':'rev_adj'}
ENTRY_STAGES={'中期趋势':'entry_trend'}
STATIC_STRATEGIES={'中期趋势':'entry_trend'}
STRATEGIES={'中期趋势':'entry_trend','同数据完整池':'feature_base','活跃度提升':'activity_entry','收缩后突破':'squeeze_entry','活跃突破':'joint_entry'}
TRIALS={k:v for k,v in STRATEGIES.items() if v in ('activity_entry','squeeze_entry','joint_entry')}

RULES={'版本': 'gpt1.7验证入场条件；三个新组分别为中期趋势、趋势回落、回落后转强。旧四种因子作为参照；新规则未证明有效。没有资金组合。', '共同股票池': '历史科技股票，信号日未复权股价严格>10元，流通市值50—1000亿元，上市至少180天。复用此前共同样本：29个交易周历史完整、历史二级行业可用且至少10只其他合格同行。风险警示仍用历史涨跌停幅度近似，供应商行业历史有局限。', '周频': 't为刚完成的交易周；完整周最后交易日收盘确认，下一交易日开盘买入。周内不使用最终周线。整周休市不计交易周。', '中期趋势A': '26周收益为C[t−2]/C[t−28]−1，必须>0；上一周收盘C[t−1]>MA13[t−1]，且MA13[t−1]>MA13[t−2]。判断趋势用上一完整周，避免本周反弹自己制造趋势合格。', '趋势回落B': '在A基础上，上一周收盘C[t−1]<C[t−3]，即截至上一周最近两周收盘净下跌。它不要求两周都下跌，不预先排除回落幅度较大的股票。', '回落后转强C': '在B基础上，本周收盘C[t]>H[t−1]（上一完整周最高价）。确认发生于本周收盘，不能用本周内提前看到的片段行情代替。', '新组统一排序': '三组都按26周上涨效率降序：100×(C[t−2]−C[t−28])/过去26个周收盘变化绝对值之和。分母窗口同样截至t−2。分母为0则不可用。越少来回波动、越接近持续向上，效率越高；这是形态分数，不是预测收益。', '幅度检验': '效率高可能对应慢涨，因此同时报告净收益≥10%/20%、最大上涨幅度和剔除最高1%后的收益。没有用本轮结果寻优涨幅门槛；高胜率低收益仍不算达标。', '推荐与分层': '每个新组在自身合格候选中按同一效率取前5，同分依次按流通市值降序、代码升序。不足5只不补位；无合格股票不退回较宽组。新组五层分别在该组候选中划分，旧组五层在共同池内划分，同分不拆开。', '对照': 'A/B/C逐层嵌套候选，但各自前五名可能不同。同日对照比较双方已知订单均值，取消订单计0，未知或单方有信号的日期另列；不是同一股票延后买入试验。新组与原始动量跨组比较同时改变了排序与条件，不能只归因于某一项。', '旧因子': '原动量=100×(C[t−2]/C[t−28]−1)；原反转=−100×(C[t]/C[t−2]−1)。行业调整继续扣除同日同业其他股票相应收益中位数，不是回归残差或中性组合。旧组仍要求评分>0并取前5。', '买卖执行': '统一沿用8%初始价格止损；最高收盘浮盈达到2R后，保护线=最高收盘价−1R，只上移且次日生效。R为含滑点买价的8%。不使用第一周或第二周早退，没有固定持仓期限。', '成交与成本': '信号后次交易日开盘买入；开盘涨停或已知停牌取消，取消不补位。T+1；跌停/停牌导致退出延迟。买入费0.10%、卖出费0.20%，每边滑点0.10%。关键路径缺失单列未知，日线不能还原真实成交队列。', '逐周收益': 'W1/W2/...按买入起5/10/...个市场交易日。整批达到观察年龄才纳入；提前退出冻结实际收益，仍持有按收盘标记并预扣退出成本。未知不补零，另报含取消0的订单均值。', '重复': '同股后续完整周仍符合条件可再次入选，作为独立假设事件记账，彼此可能重叠；不是加仓指令或账户收益，也不是统计独立样本。', '空窗': '每年无新候选推荐的交易周目标≤5；同时列未满5只及各阶段剩余候选。不能通过放宽/降级补位掩盖空窗，也不等于实际资金空仓时间。', '无退出诊断': 'W1—W12继续观察原买价之后的价格路径，忽略退出仅用于诊断。最大上涨/下跌是毛价格幅度，极值不代表可兑现利润；无退出收盘标记扣成本，末日停牌/跌停状态另列。路径缺失后后续诊断未知。', '诊断基准': '无退出同日对照要求前五已成交路径全部已知，其余共同池仅使用已知成交路径并披露缺失比例。避免因一只缺报价剔除整周；仍存在可用样本选择偏差。不得与旧版严格完整共同池的同日表直接拼接。', '止损先后': '另列8周等观察期内初始止损与首次上涨10%的先后；盘中同日先后未知单列，不把卖出后高点视为已持有利润。', '历史限制': '2022—2026已经反复研究，不能称为全新样本外。13周均线、2周回落、26周效率及上一周高点均为本次固定假设，没有寻优或收益保证。'}
for obsolete in ['趋势回落B','回落后转强C','新组统一排序','推荐与分层','对照','版本','旧因子']:
    RULES.pop(obsolete,None)
RULES.update({
 '版本':'gpt1.10，同一中期趋势池比较四种排序；这是固定比较实验，不自动选择历史收益最高的方法。',
 '共同实验池':'沿用gpt1.7中期趋势A条件；四组用完全相同的合格候选，不增加回落或转强硬门槛。共同池基准现在仅指A池，不能与旧版全科技共同池基准直接拼接。',
 '四种排序':'上涨效率（中期趋势旧组）、26周原始动量、同行调整26周动量、最近2周收益取负。行业调整仍相对原科技共同池的其他历史同行计算。四组均降序，同分按流通市值降序、代码升序。',
 '分数与名额':'排序分数可为负，四组均不另设分数门槛，不足5只不补足；评分分层在同一个A池进行。',
 '同日对照':'前五名对本组其余A池候选，先算同日已知订单均值，再对日期等权。前五须全部已知，其余未知排除并披露数量；另报其余也全已知的严格口径。取消订单计0，未知不补零。',
 '复用历史结果':'gpt1.7.1结果包含原始因子与逐笔路径，可离线重新排名；保持原数据截止日、原费用与退出规则，不补未来行情。完整基准事件用于验证A池覆盖，缺失则拒绝复用。',
 '检验目标':'同时检查前五名相对其余候选的增益、跨年表现、收益≥10%比例、未触及10%比例及剔除最高1%后的均益。历史已反复查看，不能视作新样本外。'})
VOLUME_FEATURES=['week_turnover_mean','turnover_baseline','activity_ratio','recent_range_mean',
                'earlier_range_mean','squeeze_ratio','prior_high13','breakout_pct']
FEATURE_SPEC=dict(activity_min=1.5,squeeze_max=.75,activity_baseline_weeks=13,
                  squeeze_recent_weeks=4,squeeze_previous_weeks=12,breakout_weeks=13)
for obsolete in ['逐时训练','模型固定','训练不足','四种排序','分数与名额']:RULES.pop(obsolete,None)
RULES.update({
 '版本':'gpt1.10：固定量价入场条件实验，没有训练模型或参数搜索，未证明盈利。',
 '共同实验池':'保留原中期趋势A参照；其中特征全部完整者构成同数据完整池。活跃度提升、收缩后突破、活跃突破三个条件组均来自此池。不能把缺失数据当成条件不满足。',
 '成交活跃度':'每个完整周的日均换手率（daily_basic.turnover_rate，普通换手率，非自由流通换手率）。本周日均换手率/此前13个完整周日均换手率的中位数≥1.5。用日均而非周合计减少节假日天数影响；不等同于资金净流入。',
 '收缩后突破':'每周相对真实波幅= max(周高−周低,abs(周高−前周收盘),abs(周低−前周收盘))/前周收盘。最近4周(t−4…t−1)均值/更早12周(t−16…t−5)均值≤0.75，且本周收盘严格突破此前13周最高价。不用本周上涨本身计算收缩。',
 '活跃突破':'同时满足活跃度提升与收缩后突破。阈值为本轮固定研究假设，未根据结果挑选。',
 '统一排序':'五组各自按26周上涨效率降序取前5，同分按流通市值降序、代码升序。新条件的作用先看全部满足条件者对不满足条件者的同日收益，再看前五名；不是把旧排序宣称为有效。',
 '缺失处理':'价格缺口或停牌仍沿用旧规则。换手率缺失、非正或无穷使该周活跃度无效，基线窗口要求完整；未知条件不补0。原A参照仍保留，缺失比例逐周披露。',
 '条件检验':'W1/2/4/8/12同日比较特征完整池内满足与不满足条件的全部订单，不先取前五；日期等权，取消计0，未知收益排除且披露，两侧全部已知另列严格口径。结果是观察性分组关联，不证明因果。',
 '前五对照':'条件组前五与同数据完整池前五在双方同日已知订单上比较；另报条件组前五对其余A池候选。后者同时包含条件筛选和排序作用，不称纯排序增益。',
 '复用历史结果':'gpt1.9结果缺少量价原始字段，必须同时提供原行情SQLite缓存。核对缓存范围、科技池范围及记录摘要与原结果一致后，逐股补算信号时点特征；沿用原交易路径，不下载行情。缓存不匹配时拒绝拼接，改走行情缓存计算入口。',
 '内存':'独立计算进程；上传先落盘并移除页面；行情逐股读、报告逐表写。旧缓存兼容、四路下载与断点恢复保留。',
 '历史限制':'已有年份被反复研究，不是全新样本外。价格与行业历史仍可能受供应商修订影响。新条件可能造成一年超过5周无推荐，不能靠补位掩盖。'})

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
    w=g.assign(key=key,valid=valid).groupby('key',observed=True).agg(close=('ac','last'),high=('ah','max'),low=('al','min'),valid=('valid','all'))
    w.loc[~w.valid,['close','high','low']]=np.nan
    w['history_ready']=w.valid.rolling(29,min_periods=29).sum().eq(29)
    w['mom_raw']=(w.close.shift(2)/w.close.shift(28)-1)*100
    w['short_return_pct']=(w.close/w.close.shift(2)-1)*100
    w['prev_close']=w.close.shift(1);w['prev_high']=w.high.shift(1)
    ma=w.close.rolling(13,min_periods=13).mean()
    w['prev_ma13']=ma.shift(1);w['prev_ma13_change_pct']=(ma.shift(1)/ma.shift(2)-1)*100
    w['prior2_return_pct']=(w.close.shift(1)/w.close.shift(3)-1)*100
    travel=w.close.diff().abs().shift(2).rolling(26,min_periods=26).sum()
    w['trend_efficiency']=100*(w.close.shift(2)-w.close.shift(28))/travel.where(travel.gt(0))
    w['reclaim_margin_pct']=(w.close/w.high.shift(1)-1)*100
    w['trend_ok']=w.mom_raw.gt(0)&w.prev_close.gt(w.prev_ma13)&w.prev_ma13_change_pct.gt(0)
    w['pullback_ok']=w.trend_ok&w.prior2_return_pct.lt(0)
    w['reclaim_ok']=w.pullback_ok&w.reclaim_margin_pct.gt(0)
    # New features use complete weekly prices and daily-average turnover.
    turn=pd.to_numeric(g['turnover_rate'],errors='coerce') if 'turnover_rate' in g else pd.Series(np.nan,index=g.index)
    turn_ok=np.isfinite(turn)&turn.gt(0)&valid
    tw=pd.DataFrame(dict(key=key,turnover=turn,known=turn_ok)).groupby('key',observed=True).agg(
        mean=('turnover','mean'),known=('known','all'))
    w['week_turnover_mean']=tw['mean'].where(tw.known)
    w['turnover_baseline']=w.week_turnover_mean.shift(1).rolling(13,min_periods=13).median()
    w['activity_ratio']=w.week_turnover_mean/w.turnover_baseline.where(w.turnover_baseline.gt(0))
    prev=w.close.shift(1)
    tr=pd.concat([w.high-w.low,(w.high-prev).abs(),(w.low-prev).abs()],axis=1).max(axis=1,skipna=False)/prev
    tr=tr.where(np.isfinite(tr)&prev.gt(0))
    w['recent_range_mean']=tr.shift(1).rolling(4,min_periods=4).mean()
    w['earlier_range_mean']=tr.shift(5).rolling(12,min_periods=12).mean()
    w['squeeze_ratio']=w.recent_range_mean/w.earlier_range_mean.where(w.earlier_range_mean.gt(0))
    w['prior_high13']=w.high.shift(1).rolling(13,min_periods=13).max()
    w['breakout_pct']=100*(w.close/w.prior_high13-1)

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
    ready=f.loc[f.factor_ready,['signal_date','industry_key']]
    counts=ready.groupby(['signal_date','industry_key'],observed=True).size()
    keys=pd.MultiIndex.from_frame(f[['signal_date','industry_key']])
    f['peer_count']=np.maximum(0,counts.reindex(keys).fillna(0).to_numpy()-1)
    f['common_pass']=f.factor_ready&f.peer_count.ge(10)
    for col in ['industry_mom_pct','industry_short_pct','mom_adj','rev_raw','rev_adj','industry_vs_tech_pp']:
        f[col]=np.nan
    common=f.loc[f.common_pass,['signal_date','industry_key','mom_raw','short_return_pct']]
    for _,g in common.groupby(['signal_date','industry_key'],observed=True):
        f.loc[g.index,'industry_mom_pct']=loo_median(g.mom_raw)
        f.loc[g.index,'industry_short_pct']=loo_median(g.short_return_pct)
    f.loc[common.index,'mom_adj']=f.loc[common.index,'mom_raw']-f.loc[common.index,'industry_mom_pct']
    f.loc[common.index,'rev_raw']=-f.loc[common.index,'short_return_pct']
    f.loc[common.index,'rev_adj']=f.loc[common.index,'industry_short_pct']-f.loc[common.index,'short_return_pct']
    tech=f[f.common_pass].groupby('signal_date',observed=True).mom_raw.median()
    f.loc[common.index,'industry_vs_tech_pp']=f.loc[common.index,'industry_mom_pct']-f.loc[common.index,'signal_date'].map(tech)
    eligible=f.common_pass & f.trend_ok.eq(True) & f.trend_efficiency.gt(0)
    eligible &= f[['mom_raw','mom_adj','short_return_pct','trend_efficiency']].notna().all(axis=1)
    for col,source,sign in [('entry_trend','trend_efficiency',1)]:
        f['eligible_'+col]=eligible
        f[col]=(sign*f[source]).where(eligible)
        f['rank_'+col]=np.nan;f['layer_'+col]=np.nan
        order=f.loc[eligible,['signal_date',col,'circ_mv_yi','ts_code']].sort_values(
            ['signal_date',col,'circ_mv_yi','ts_code'],ascending=[True,False,False,True])
        f.loc[order.index,'rank_'+col]=order.groupby('signal_date',observed=True).cumcount()+1
        r=order.groupby('signal_date',observed=True)[col].rank(ascending=False,method='average')
        size=order.groupby('signal_date',observed=True)[col].transform('size')
        f.loc[order.index,'layer_'+col]=np.minimum(5,np.floor((r-1)/size*5)+1)
        # Scores can be negative: no method-specific eligibility gates in this experiment.
        f['selected_'+col]=eligible & f['rank_'+col].le(5)

    return rank_feature_groups(f)


def rank_feature_groups(f):
    features=f.reindex(columns=VOLUME_FEATURES)
    ready=f.eligible_entry_trend & np.isfinite(features).all(axis=1)
    activity=features.activity_ratio.ge(FEATURE_SPEC['activity_min'])
    breakout=features.squeeze_ratio.le(FEATURE_SPEC['squeeze_max'])&features.breakout_pct.gt(0)
    for col,gate in [('feature_base',ready),('activity_entry',ready&activity),
                     ('squeeze_entry',ready&breakout),('joint_entry',ready&activity&breakout)]:
        f['eligible_'+col]=gate;f[col]=f.trend_efficiency.where(gate)
        f['rank_'+col]=np.nan;f['layer_'+col]=np.nan
        order=f.loc[gate,['signal_date',col,'circ_mv_yi','ts_code']].sort_values(
            ['signal_date',col,'circ_mv_yi','ts_code'],ascending=[True,False,False,True])
        f.loc[order.index,'rank_'+col]=order.groupby('signal_date',observed=True).cumcount()+1
        r=order.groupby('signal_date',observed=True)[col].rank(ascending=False,method='average')
        size=order.groupby('signal_date',observed=True)[col].transform('size')
        f.loc[order.index,'layer_'+col]=np.minimum(5,np.floor((r-1)/size*5)+1)
        f['selected_'+col]=f['rank_'+col].le(5)
    return f


def enrich_cached_features(f,basic,manifest,root,progress=lambda _:None):
    path=Path(root)/CACHE_SCHEMA/'tech_bars_v1.sqlite'
    if not path.is_file():raise ValueError('未找到原行情SQLite缓存。gpt1.9 ZIP不含日均换手率所需数据，请选择行情缓存计算；保持原缓存目录。')
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        db.execute('PRAGMA cache_size=-2048');db.execute('PRAGMA mmap_size=0')
        rows=db.execute('SELECT day,scope,complete,issues,digest FROM days WHERE day BETWEEN ? AND ? ORDER BY day',
                        (manifest['data_start'],manifest['data_end'])).fetchall()
        scope=hashlib.sha256('\n'.join(sorted(basic.ts_code)).encode()).hexdigest()
        if not rows or any(r[1]!=scope or r[2]!=1 or json.loads(r[3]) for r in rows):
            raise ValueError('原区间缓存不完整或科技池范围已变化，请改用行情缓存计算，不能将不同批行情与旧交易路径拼接。')
        digest=hashlib.sha256('\n'.join(r[0]+':'+r[4] for r in rows).encode()).hexdigest()
        if digest!=manifest.get('data_hash'):raise ValueError('缓存记录摘要与gpt1.9结果不一致，请改用行情缓存计算。')
        calendar=pd.DatetimeIndex(pd.to_datetime([r[0] for r in rows],format='%Y%m%d'))
        loc=calendar.get_indexer(pd.to_datetime(f.signal_date))
        if not np.array_equal(loc,f.signal_i.to_numpy()):raise ValueError('缓存交易日序号与旧信号不一致，停止复用。')
        key=calendar.to_period('W-FRI');complete=np.flatnonzero(np.r_[key[:-1]!=key[1:],True])
        for c in VOLUME_FEATURES:f[c]=np.nan
        groups=f.groupby('ts_code',observed=True,sort=True).indices
        for number,(code,index) in enumerate(groups.items(),1):
            part=pd.read_sql_query('SELECT * FROM bars WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date',
                                  db,params=(str(code),manifest['data_start'],manifest['data_end']))
            part['date']=pd.to_datetime(part.trade_date,format='%Y%m%d')
            g=prepared_stock(part,calendar);w=weekly_features(g,complete)
            signal_dates=pd.DatetimeIndex(f.loc[index,'signal_date']);new=w.reindex(signal_dates)
            # A stale/mixed price cache must not silently change the old baseline features.
            for c in ['mom_raw','short_return_pct','prev_close','prev_high','trend_efficiency']:
                if not np.allclose(f.loc[index,c].to_numpy(dtype=float),new[c].to_numpy(dtype=float),equal_nan=True,rtol=1e-8,atol=1e-8):
                    raise ValueError('缓存价格重算与旧因子不一致：'+str(code)+'；请改走行情缓存计算。')
            f.loc[index,VOLUME_FEATURES]=new[VOLUME_FEATURES].to_numpy()
            if number%50==0:progress(f'从原缓存补算量价特征 {number}/{len(groups)}；不下载行情')
        return dict(cache_days=len(calendar),stocks=len(groups),cached_data_hash=digest,hash_matches=True,
                    baseline_features_match=True,network_calls=0)
    finally:db.close()


def feature_readiness(f,schedule):
    rows=[]
    groups=f.groupby('signal_date',observed=True).indices if not f.empty else {}
    for day in schedule.signal_date:
        index=groups.get(day,[]);g=f.iloc[index];a=g[g.eligible_entry_trend] if len(g) else g
        rows.append(dict(signal_date=day,year=str(day.year),a_candidates=len(a),
            feature_complete=int(a.eligible_feature_base.sum()) if len(a) else 0,
            activity_missing=int((~np.isfinite(a.activity_ratio)).sum()) if len(a) else 0,
            shape_missing=int((~np.isfinite(a[['squeeze_ratio','breakout_pct']]).all(axis=1)).sum()) if len(a) else 0,
            activity_candidates=int(a.eligible_activity_entry.sum()) if len(a) else 0,
            squeeze_candidates=int(a.eligible_squeeze_entry.sum()) if len(a) else 0,
            joint_candidates=int(a.eligible_joint_entry.sum()) if len(a) else 0))
    return pd.DataFrame(rows)


def condition_reports(e,marks,schedule,calendar,progress=lambda _:None):
    rows=[]
    for w in [1,2,4,8,12]:
        if e.empty:break
        progress(f'全部条件候选对照（未经前五排序） W{w}')
        v=weekly_view(e,marks,w)
        for day in mature_schedule(e,schedule,calendar,w).signal_date:
            pool=v[v.signal_date.eq(day)&v.eligible_feature_base]
            for name,col in TRIALS.items():
                yes=pool[pool['eligible_'+col]];no=pool[~pool['eligible_'+col]]
                valid=yes.week_known.any() and no.week_known.any()
                y=yes.loc[yes.week_known,'week_order_pct'];n=no.loc[no.week_known,'week_order_pct']
                rows.append(dict(strategy=name,signal_date=day,year=str(day.year),week_no=w,
                    passing_events=len(yes),failing_events=len(no),passing_unknown=int((~yes.week_known).sum()),
                    failing_unknown=int((~no.week_known).sum()),paired=bool(valid),
                    strict_pair=bool(valid and yes.week_known.all() and no.week_known.all()),
                    passing_mean=y.mean() if valid else np.nan,failing_mean=n.mean() if valid else np.nan,
                    condition_delta_pp=y.mean()-n.mean() if valid else np.nan))
    d=pd.DataFrame(rows);summary=[]
    if not d.empty:
        for (name,w),g in d.groupby(['strategy','week_no'],observed=True):
            for yr,p in years(g):
                q=p[p.paired];strict=q[q.strict_pair]
                summary.append(dict(strategy=name,year=yr,week_no=w,observed_dates=len(p),paired_dates=len(q),
                    excluded_dates=len(p)-len(q),passing_mean=q.passing_mean.mean(),failing_mean=q.failing_mean.mean(),
                    condition_delta_pp=q.condition_delta_pp.mean(),strict_dates=len(strict),
                    strict_condition_delta_pp=strict.condition_delta_pp.mean(),
                    passing_events=int(p.passing_events.sum()),failing_events=int(p.failing_events.sum()),
                    passing_unknown=int(p.passing_unknown.sum()),failing_unknown=int(p.failing_unknown.sum())))
    return dict(condition_dates=d,condition_summary=pd.DataFrame(summary))


def feature_self_test():
    full=pd.bdate_range('2020-01-06',periods=205);week=np.arange(200)//5
    cl=np.full(40,100.);cl[35]=115.
    spread=np.full(40,10.);spread[31:35]=2.
    turnover=np.full(200,2.);turnover[week==35]=4.
    g=pd.DataFrame(dict(close=cl[week],high=cl[week]+spread[week],low=cl[week]-spread[week],
        ac=cl[week],ah=cl[week]+spread[week],al=cl[week]-spread[week],vol=100.,turnover_rate=turnover),index=full[:200])
    indices=complete_week_indices(g.index,full);w=weekly_features(g,indices);day=g.index[179]
    assert np.isclose(w.loc[day,'activity_ratio'],2) and np.isclose(w.loc[day,'squeeze_ratio'],.2)
    assert w.loc[day,'breakout_pct']>0
    future=g.copy();future.loc[future.index>day,['ac','ah','al','turnover_rate']]*=10
    pd.testing.assert_frame_equal(w.loc[:day],weekly_features(future,indices).loc[:day])
    altered=g.copy();altered.loc[altered.index.to_period('W-FRI')==day.to_period('W-FRI'),'ah']*=3
    assert weekly_features(altered,indices).loc[day,'squeeze_ratio']==w.loc[day,'squeeze_ratio']
    missing=g.copy();missing.loc[day,'turnover_rate']=np.nan
    assert np.isnan(weekly_features(missing,indices).loc[day,'activity_ratio'])
    print('PASS: activity and contraction arithmetic; no current-week squeeze leakage; future isolation; missing turnover remains unknown')


def calculate(data,basic,member,calendar,cfg,progress,full_calendar=None,diagnostic_sink=None,marks_sink=None):
    full_calendar=calendar if full_calendar is None else full_calendar
    complete_i=complete_week_indices(calendar,full_calendar)
    signal_i=complete_i[(calendar[complete_i]>=stamp(cfg.start))&(calendar[complete_i]<=stamp(cfg.end))]
    if not len(signal_i):return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame({'signal_date':calendar[signal_i]})
    base=basic.set_index('ts_code');members={c:m for c,m in member.groupby('ts_code',observed=True)};rows=[]
    for num,(code,part) in enumerate(data.groupby('ts_code',sort=True,observed=True),1):
        if code not in base.index or code not in members:continue
        g=prepared_stock(part,calendar);eligible,known=eligibility(g,code,base.loc[code],members[code],calendar,cfg)
        w=weekly_features(g,complete_i).reindex(calendar[signal_i]);ind,amb=industry_at(members[code],calendar[signal_i])
        z=pd.DataFrame(dict(event_id=[code+'|'+ds(d) for d in calendar[signal_i]],ts_code=code,name=base.loc[code,'name'],
            signal_date=calendar[signal_i],signal_i=signal_i,year=calendar[signal_i].year.astype(str),
            signal_week=calendar[signal_i].to_period('W-FRI').astype(str),pool_pass=eligible.iloc[signal_i].to_numpy(),
            pool_known=known.iloc[signal_i].to_numpy(),industry_key=ind.to_numpy(),industry_ambiguous=amb.to_numpy(),
            circ_mv_yi=g.circ_mv.iloc[signal_i].to_numpy()/10000,signal_close=g.close.iloc[signal_i].to_numpy(),
            history_ready=w.history_ready.fillna(False).to_numpy(),mom_raw=w.mom_raw.to_numpy(),short_return_pct=w.short_return_pct.to_numpy()))
        for col in ENTRY_FEATURES:z[col]=w[col].to_numpy()
        rows.append(compact_frame(z) if isinstance(data,MarketStore) else z)
        if num%50==0:progress(f'周收益与历史行业 {num}/{len(base)}')
    feature_frame=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    rows.clear();del rows
    if isinstance(data,MarketStore):feature_frame=compact_frame(feature_frame)
    f=score_cross_sections(feature_frame);del feature_frame;gc.collect()
    if isinstance(data,MarketStore):f=compact_frame(f)
    if f.empty:return f,pd.DataFrame(),f,pd.DataFrame({'signal_date':calendar[signal_i]})
    common=f[f.eligible_entry_trend].copy();paths=[];marks=[];bycode=common.groupby('ts_code',observed=True).indices
    for num,(code,part) in enumerate(data.groupby('ts_code',sort=True,observed=True),1):
        if code not in bycode:continue
        g=prepared_stock(part,calendar)
        arrays=diagnostic_arrays(g) if diagnostic_sink is not None else None
        stock_diagnostics=[];stock_paths=[];stock_marks=[]
        for row in common.iloc[bycode[code]].itertuples():
            path,weekly=lifecycle(g,int(row.signal_i)+1)
            if diagnostic_sink is not None:
                stock_diagnostics.extend(dict(event_id=row.event_id,**x) for x in unbounded_path(arrays,path))
            stock_paths.append(dict(event_id=row.event_id,**path))
            stock_marks.extend(dict(event_id=row.event_id,**x) for x in weekly)
        paths.append(compact_frame(pd.DataFrame(stock_paths)))
        if marks_sink is not None:marks_sink.append(pd.DataFrame(stock_marks))
        else:marks.extend(stock_marks)
        if diagnostic_sink is not None and stock_diagnostics:
            diagnostic_sink.append(pd.DataFrame(stock_diagnostics))
        if num%25==0:progress(f'共用退出路径 {num}/{len(base)}；不重复计算各组重合股票')
    if paths:
        e=common.merge(pd.concat(paths,ignore_index=True),on='event_id',validate='one_to_one')
        e['base_pass']=True;e['mature_weeks']=np.maximum(0,(len(calendar)-e.buy_i)//5)
    else:e=pd.DataFrame()
    return e,(marks_sink.finish() if marks_sink is not None else pd.DataFrame(marks)),f,pd.DataFrame({'signal_date':calendar[signal_i]})


def period_stat(g):
    v=g.loc[g.filled,'week_net_pct']
    return dict(mature_events=len(g),filled=int(g.filled.sum()),known_filled=int(v.notna().sum()),
        exited=int(g.exited.sum()),holding=int(g.holding.sum()),cancelled=int(g.cancelled.sum()),unknown=int((~g.week_known).sum()),
        mean_order_pct=g.loc[g.week_known,'week_order_pct'].mean(),**distribution(v))


def years(frame):
    yield '全部',frame
    yield from frame.groupby('year',observed=True)


def date_comparison(view,schedule,w,family,raw,adjusted):
    out=schedule[['signal_date']].copy();out['year']=out.signal_date.dt.year.astype(str)
    for tag,col in [('raw',raw),('adjusted',adjusted)]:
        chosen=view[view['selected_'+col]]
        g=chosen.groupby('signal_date',observed=True).agg(count=('event_id','size'),known=('week_known','sum'),filled=('filled','sum'),mean=('week_order_pct','mean'))
        for field in ['count','known','filled','mean']:out[tag+'_'+field]=out.signal_date.map(g[field])
        for field in ['count','known','filled']:out[tag+'_'+field]=out[tag+'_'+field].fillna(0).astype(int)
    both=out.raw_count.gt(0)&out.adjusted_count.gt(0)
    known=out.raw_count.eq(out.raw_known)&out.adjusted_count.eq(out.adjusted_known)
    out['comparison_status']=np.select([both&known,both, out.raw_count.gt(0),out.adjusted_count.gt(0)],
        ['可比较','结果未知','仅原始有推荐','仅调整有推荐'],default='双方无推荐')
    out['delta_pp']=(out.adjusted_mean-out.raw_mean).where(both&known)
    out['family']=family;out['week_no']=w
    return out


def build_reports(e,marks,f,schedule,calendar,cfg,progress=lambda _:None,output=None):
    names=['events','weekly_marks','factor_candidates','selections','weekly_summary','survivor_summary','final_summary',
        'rank_layers','industry_adjustment_dates','industry_adjustment_summary','coverage','pool_filter_counts',
        'industry_profile','top5_weekly_history','signal_calendar']
    tables={} if output is None else output
    tables.update({k:pd.DataFrame() for k in names});tables.update(events=e,weekly_marks=marks,factor_candidates=f,signal_calendar=schedule)
    tables['feature_readiness']=feature_readiness(f,schedule)
    coverage=[];filters=[]
    for yr in sorted(schedule.signal_date.dt.year.unique()):
        days=schedule[schedule.signal_date.dt.year.eq(yr)].signal_date;yr=str(yr)
        pool=f[f.year.eq(yr)] if not f.empty else f
        for strategy,col in STRATEGIES.items():
            chosen=pool[pool.get('selected_'+col,pd.Series(False,index=pool.index)).eq(True)]
            count=chosen.groupby('signal_date',observed=True).size() if not chosen.empty else pd.Series(dtype=int)
            coverage.append(dict(strategy=strategy,year=yr,observed_weeks=len(days),no_signal_weeks=int((~days.isin(count.index)).sum()),
                fewer_than5_signal_weeks=int(count.lt(5).sum()),selected_events=len(chosen),
                full_year=stamp(cfg.start)<=pd.Timestamp(int(yr),1,1) and stamp(cfg.end)>=pd.Timestamp(int(yr),12,31) and latest_ready_day()>=pd.Timestamp(int(yr),12,31)))
        if not pool.empty:
            filters.append(dict(year=yr,stock_week_records=len(pool),pool_pass=int(pool.pool_pass.sum()),
                history_ready=int((pool.pool_pass&pool.history_ready).sum()),industry_ambiguous=int(pool.industry_ambiguous.sum()),
                factor_ready=int(pool.factor_ready.sum()),common_pass=int(pool.common_pass.sum())))
    tables['coverage']=pd.DataFrame(coverage);tables['pool_filter_counts']=pd.DataFrame(filters)
    if e.empty:return tables
    c=f.loc[f.common_pass,['signal_date','industry_key','ts_code','mom_raw','short_return_pct','industry_vs_tech_pp']]
    tables['industry_profile']=c.groupby(['signal_date','industry_key'],observed=True).agg(stocks=('ts_code','size'),
        mom_median_pct=('mom_raw','median'),short_median_pct=('short_return_pct','median'),industry_vs_tech_pp=('industry_vs_tech_pp','median')).reset_index()
    del c
    selected=[];summary=[];survivors=[];final=[];layers=[];dates=[]
    history=CSVSpool(output.root/'history_stream.csv.gz') if isinstance(output,DiskTables) else []
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
        mature=mature_schedule(e,schedule,calendar,w)
        for family,raw,adj in [(name,'feature_base',col) for name,col in TRIALS.items()]:
            dates.append(date_comparison(v,mature,w,family,raw,adj))
    tables.update(weekly_summary=pd.DataFrame(summary),survivor_summary=pd.DataFrame(survivors),rank_layers=pd.DataFrame(layers),
        top5_weekly_history=history if isinstance(history,CSVSpool) else (pd.concat(history,ignore_index=True) if history else pd.DataFrame()))
    d=pd.concat(dates,ignore_index=True) if dates else pd.DataFrame();comparisons=[]
    if not d.empty:
        for (family,w),whole in d.groupby(['family','week_no'],observed=True):
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
        never_up10_count=int(valid.max_up_pct.lt(10).sum()),never_up10_pct=pct(valid.max_up_pct.lt(10)),
        max_up_ge10_pct=pct(valid.max_up_pct.ge(10)),max_up_ge20_pct=pct(valid.max_up_pct.ge(20)),
        max_down_mean_pct=valid.max_down_pct.mean(),max_down_le8_pct=pct(valid.max_down_pct.le(-8)),
        close_mdd_mean_pct=valid.close_mdd_pct.mean(),
        paired_unbounded_pct=un.mean(),paired_rule_pct=rule.mean(),exit_contribution_pp=diff.mean(),
        exit_helped_pct=pct(diff.gt(1e-9)),exit_hurt_pct=pct(diff.lt(-1e-9)),
        rule_loss_path_profit=int((rule.lt(0)&un.gt(0)).sum()),
        rule_profit_path_loss=int((rule.gt(0)&un.lt(0)).sum()),
        both_profit=int((rule.gt(0)&un.gt(0)).sum()),both_loss=int((rule.lt(0)&un.lt(0)).sum()),
        either_zero=int((rule.eq(0)|un.eq(0)).sum()))


def entry_exit_reports(e,marks,paths,progress=lambda _:None,output=None):
    names=['path_diagnostics','entry_exit_summary','entry_rank_dates','entry_rank_summary',
           'entry_date_comparison','entry_date_summary','entry_top5_details','stop_timing_summary']
    out={} if output is None else output
    out.update({k:pd.DataFrame() for k in names});out['path_diagnostics']=paths
    if paths.empty or e.empty:return out
    if not isinstance(paths,SQLSpool) and paths.duplicated(['event_id','week_no']).any():raise ValueError('诊断主键重复')
    summaries=[];rank_dates=[];dates=[];stop_timing=[]
    details=CSVSpool(output.root/'diagnostic_top_stream.csv.gz') if isinstance(output,DiskTables) else []
    for w in DIAGNOSTIC_WEEKS:
        progress(f'入场与退出配对诊断 W{w}/12')
        v=weekly_view(e,marks,w)
        if v.empty:continue
        v=v.merge(week_rows(paths,w),on='event_id',how='left',validate='one_to_one')
        if v.path_known.isna().any():raise ValueError('成熟事件缺少诊断记录，不能当作零收益')
        for yr,p in years(v):summaries.append(dict(strategy='共同池基准',group='全部',year=yr,week_no=w,**entry_stat(p)))
        for strategy,col in STRATEGIES.items():
            selected=v['selected_'+col];chosen=v[selected]
            groups=diagnostic_groups(v,chosen,col)
            for group,g in groups:
                for yr,p in years(g):summaries.append(dict(strategy=strategy,group=group,year=yr,week_no=w,**entry_stat(p)))
            stopped=chosen[chosen.exited & chosen.exit_reason.eq('初始止损')]
            for yr,p in years(stopped):
                for category,count in p.stop_up10_timing.value_counts().items():
                    if not count:continue
                    stop_timing.append(dict(strategy=strategy,year=yr,week_no=w,category=category,
                        count=int(count),stop_total=len(p),share_of_stops_pct=count/len(p)*100))
            cols=['event_id','ts_code','name','signal_date','year','buy_date','filled','exited','week_net_pct',
                  'sell_date','exit_reason','stop_up10_timing',
                  'week_no','path_known','path_status','unbounded_net_pct','max_up_pct','max_down_pct','close_mdd_pct']
            d=chosen[cols].copy();d['strategy']=strategy;d['rank']=chosen['rank_'+col];details.append(d)
            for day,g in v.groupby('signal_date',observed=True):
                q=g[g.filled & g.path_known & g[col].notna()];rec=dict(strategy=strategy,year=str(day.year),week_no=w,signal_date=day,
                    filled=int(g.filled.sum()),path_known_count=len(q))
                for field,target in [('return_ic','unbounded_net_pct'),('up_ic','max_up_pct'),('down_ic','max_down_pct')]:
                    rec[field]=q[col].rank().corr(q[target].rank()) if len(q)>=10 and q[col].nunique()>1 and q[target].nunique()>1 else np.nan
                rank_dates.append(rec)
                top=g[g['selected_'+col]];rest=g[~g['selected_'+col]]
                tf=top[top.filled];rf=rest[rest.filled]
                rf_known=rf[rf.path_known]
                valid=len(tf)>0 and len(rf_known)>0 and tf.path_known.all()
                rec=dict(strategy=strategy,year=str(day.year),week_no=w,signal_date=day,top_events=len(top),
                    top_filled=len(tf),rest_filled=len(rf),top_unknown=int((~tf.path_known).sum()),rest_unknown=int((~rf.path_known).sum()),
                    comparison_status='可比较' if valid else ('无推荐' if top.empty else '未成交或路径未知'))
                for field in ['unbounded_net_pct','max_up_pct','max_down_pct']:
                    rec['top_'+field]=tf[field].mean() if valid else np.nan
                    rec['rest_'+field]=rf_known[field].mean() if valid else np.nan
                    rec['delta_'+field]=rec['top_'+field]-rec['rest_'+field]
                dates.append(rec)
    rank_dates=pd.DataFrame(rank_dates);dates=pd.DataFrame(dates);rs=[];dsum=[]
    if not rank_dates.empty:
        for (strategy,w),g in rank_dates.groupby(['strategy','week_no'],observed=True):
            for yr,p in years(g):
                r=dict(strategy=strategy,week_no=w,year=yr,mature_dates=len(p))
                for field in ['return_ic','up_ic','down_ic']:
                    q=p[field].dropna();r[field+'_dates']=len(q);r[field+'_mean']=q.mean();r[field+'_positive_pct']=q.gt(0).mean()*100 if len(q) else np.nan
                rs.append(r)
    if not dates.empty:
        for (strategy,w),g in dates.groupby(['strategy','week_no'],observed=True):
            for yr,p in years(g):
                q=p[p.comparison_status.eq('可比较')]
                r=dict(strategy=strategy,week_no=w,year=yr,mature_dates=len(p),paired_dates=len(q),excluded_dates=len(p)-len(q))
                for field in ['unbounded_net_pct','max_up_pct','max_down_pct']:
                    for prefix in ['top_','rest_','delta_']:r[prefix+field]=q[prefix+field].mean()
                r['top_better_dates_pct']=q.delta_unbounded_net_pct.gt(1e-9).mean()*100 if len(q) else np.nan
                r['benchmark_events']=int(q.rest_filled.sum());r['benchmark_missing']=int(q.rest_unknown.sum())
                r['benchmark_missing_pct']=q.rest_unknown.sum()/q.rest_filled.sum()*100 if q.rest_filled.sum() else np.nan
                dsum.append(r)
    out.update(stop_timing_summary=pd.DataFrame(stop_timing),entry_exit_summary=pd.DataFrame(summaries),entry_rank_dates=rank_dates,entry_rank_summary=pd.DataFrame(rs),
        entry_date_comparison=dates,entry_date_summary=pd.DataFrame(dsum),
        entry_top5_details=details if isinstance(details,CSVSpool) else (pd.concat(details,ignore_index=True) if details else pd.DataFrame()))
    return out


PRICE_ENTRY_FEATURES=['prev_close','prev_high','prev_ma13','prev_ma13_change_pct','prior2_return_pct',
                'trend_efficiency','reclaim_margin_pct','trend_ok','pullback_ok','reclaim_ok']
ENTRY_FEATURES=PRICE_ENTRY_FEATURES+VOLUME_FEATURES


def mature_schedule(e,schedule,calendar,w):
    if 'mature_weeks' in schedule:return schedule[schedule.mature_weeks.ge(w)]
    if calendar is None:raise ValueError('缺少真实交易周龄，不能用自然日代替')
    return schedule[calendar.searchsorted(schedule.signal_date)+1+5*w<=len(calendar)]


def diagnostic_groups(v,chosen,col):
    # Yield one subgroup at a time. A list of five full-pool slices caused the old peak.
    yield '前五名',chosen
    for r in range(1,6):yield f'第{r}名',chosen[chosen['rank_'+col].eq(r)]
    for layer in range(1,6):yield f'评分第{layer}层',v[v['layer_'+col].eq(layer)]


def entry_stage_reports(e,marks,f,schedule,calendar,progress=lambda _:None):
    rows=[];summary=[]
    for w in [1,2,4,8,12]:
        progress(f'同一趋势池：前五名对其余候选 W{w}')
        if e.empty:continue
        v=weekly_view(e,marks,w)
        for day in mature_schedule(e,schedule,calendar,w).signal_date:
            g=v[v.signal_date.eq(day)]
            for name,col in STRATEGIES.items():
                top=g[g['selected_'+col]];rest=g[~g['selected_'+col]]
                valid=len(top)>0 and top.week_known.all() and rest.week_known.any()
                strict=valid and rest.week_known.all()
                rows.append(dict(strategy=name,signal_date=day,year=str(day.year),week_no=w,
                    top_events=len(top),rest_events=len(rest),top_unknown=int((~top.week_known).sum()),
                    rest_unknown=int((~rest.week_known).sum()),paired=valid,strict_pair=strict,
                    top_mean=top.week_order_pct.mean() if valid else np.nan,
                    rest_mean=rest.week_order_pct.mean() if valid else np.nan,
                    delta_pp=top.week_order_pct.mean()-rest.week_order_pct.mean() if valid else np.nan))
    d=pd.DataFrame(rows)
    if not d.empty:
        for (name,w),g in d.groupby(['strategy','week_no'],observed=True):
            for yr,p in years(g):
                q=p[p.paired];strict=q[q.strict_pair]
                summary.append(dict(strategy=name,year=yr,week_no=w,observed_dates=len(p),paired_dates=len(q),
                    excluded_dates=len(p)-len(q),top_mean=q.top_mean.mean(),rest_mean=q.rest_mean.mean(),
                    delta_pp=q.delta_pp.mean(),better_dates_pct=q.delta_pp.gt(0).mean()*100 if len(q) else np.nan,
                    rest_unknown=int(q.rest_unknown.sum()),rest_events=int(q.rest_events.sum()),
                    strict_dates=len(strict),strict_delta_pp=strict.delta_pp.mean()))
    return {'ranking_dates':d,'ranking_summary':pd.DataFrame(summary)}


def run_research(token,cache_root,cfg,progress):
    root=Path(cache_root);root.mkdir(parents=True,exist_ok=True)
    with research_lock(root):
        # Previous interrupted work is disposable; persistent market DB and completed results remain.
        work=root/'work_gpt110';shutil.rmtree(work,ignore_errors=True);work.mkdir()
        started=time.monotonic(); stages=[]
        def report(message):
            rss=memory_mb();print(f'[{VERSION}] {message} | {rss}',flush=True)
            progress(message)
            if not stages or time.monotonic()-stages[-1]['elapsed_seconds']-started>=30:
                stages.append(dict(stage=message,elapsed_seconds=round(time.monotonic()-started,1),**rss))
        client=DataClient(token,cache_root,report)
        basic,member,pool_mode,pool_warnings=client.universe()
        if '快照' in pool_mode or 'l2_name' not in member or member.l2_name.fillna('').str.strip().eq('').all():
            raise RuntimeError('本版需要历史二级行业区间，当前仅有快照或字段缺失。请恢复index_member_all权限/数据；已有行情保留。')
        ready=latest_ready_day()
        if min(stamp(cfg.end),ready)<stamp(cfg.start):raise ValueError('尚未进入指定信号区间')
        start=ds(pd.Timestamp(stamp(cfg.start).to_pydatetime()-timedelta(days=450)));end=ds(ready)
        full=client.calendar(start,ds(pd.Timestamp(ready.to_pydatetime()+timedelta(days=14))));calendar=full[full<=ready]
        data=None;marks=SQLSpool(work/'marks.sqlite');diagnostics=SQLSpool(work/'paths.sqlite')
        try:
            report('开始下载/迁移；每个完成日立即保存，可直接重试')
            data,issues=client.download(calendar,set(basic.ts_code));_,data_hash=data.audit()
            report('下载阶段完成；开始逐只股票计算')
            e,_,f,schedule=calculate(data,basic,member,calendar,cfg,report,full,diagnostics,marks)
            data.close();data=None;marks.finish();diagnostics.finish();gc.collect()
            report('退出路径完成；汇总结果并逐表落盘')
            tables=DiskTables(work/'reports')
            build_reports(e,marks,f,schedule,calendar,cfg,report,output=tables)
            tables.update(condition_reports(e,marks,schedule,calendar,report))
            tables.update(data_issues=issues,universe=basic,industry_intervals=member)
            del f;gc.collect()
            entry_exit_reports(e,marks,diagnostics,report,output=tables)
            tables.update(entry_stage_reports(e,marks,None,schedule,calendar,report))
            del e;gc.collect()
            manifest=dict(version=VERSION,config=asdict(cfg),rules=RULES,created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                data_start=start,data_end=ds(calendar.max()),data_hash=data_hash,pool_hash=hashlib.sha256(member.to_csv(index=False).encode()).hexdigest(),
                pool_mode=pool_mode,warnings=pool_warnings,data_issues=len(issues),universe_size=len(basic),download_workers=DOWNLOAD_WORKERS,
                last_signal_date=str(schedule.signal_date.max().date()) if not schedule.empty else None,
                limitations=['行业调整为同行收益中位数调整，不是回归残差或中性组合','方向合格不等于正期望；覆盖率不能单独证明有效',
                    '共同样本要求报价完整及足够同行，可能产生数据可用性选择','独立事件可重叠，不能当账户收益',
                    '行业区间和历史风险警示仍有供应商限制','新条件组来自特征完整A池，统一效率排序与退出；可能增加空窗','已经反复使用的历史不是新样本外'])
            manifest.update(strategy_version='gpt1.10',storage_revision=DOWNLOAD_REVISION,
                data_hash_scheme='sha256(sorted daily tech-only CSV digests)',runtime_memory_mb=memory_mb(),
                memory_notes=['四路并发、最多四个待处理日期','仅科技池一份持久行情；日事务提交，按股票读取',
                    '诊断路径按观察周读取；报告逐表落盘；页面按需读取','本地缓存不能保证跨删除应用/重新部署保留'])
            manifest.pop('learning',None)
            manifest['feature_experiment']=FEATURE_SPEC
            tables['resource_usage']=pd.DataFrame(stages)
            run_id=hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()[:12]
            path=root/'results'/f'{VERSION}_{run_id}{RUN_SUFFIX}.zip'
            report('正在流式打包，完成后才替换结果文件')
            write_zip(tables,manifest,path)
            report('计算完成；结果保存在磁盘，切换页面无需重跑')
            return ZipTables(path),manifest,None,str(path)
        finally:
            if data is not None:data.close()
            marks.close();diagnostics.close()
            shutil.rmtree(work,ignore_errors=True);gc.collect()

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

LABELS.update({'score': '排序分数（新组为效率，旧组为原因子）', 'trend_efficiency': '26周上涨效率0—100', 'prior2_return_pct': '上一周结束的2周涨跌%', 'prev_ma13_change_pct': '上一周13周均线变化%', 'reclaim_margin_pct': '本周收盘高于前周高点%', 'trend_ok': '趋势合格', 'pullback_ok': '趋势及回落合格', 'reclaim_ok': '趋势回落转强合格', 'comparison': '实验组 对 参照组', 'reference_only': '仅参照有推荐周', 'trial_only': '仅实验有推荐周', 'neither': '双方无推荐周', 'reference_mean_pct': '参照同日订单均益%', 'trial_mean_pct': '实验同日订单均益%', 'delta_pp': '实验减参照百分点', 'trial_better_pct': '实验胜出日期%', 'eligible_events': '阶段合格股票周数', 'previous_events': '前阶段股票周数', 'excluded_events': '本条件排除股票周数', 'benchmark_events': '基准成交事件数', 'benchmark_missing': '基准路径缺失事件数', 'benchmark_missing_pct': '基准路径缺失比例%', 'common_count': '共同池候选数', 'entry_trend_count': '中期趋势合格数', 'entry_trend_selected': '中期趋势推荐数', 'entry_pullback_count': '趋势回落合格数', 'entry_pullback_selected': '趋势回落推荐数', 'entry_reclaim_count': '回落后转强合格数', 'entry_reclaim_selected': '回落后转强推荐数'})

def load_result(payload,cache_root='tech_swing_cache'):
    # Upload is already memory-backed by Streamlit. Stream its buffer to disk without getvalue().
    if isinstance(payload,(str,Path)):
        path=Path(payload)
    else:
        source=io.BytesIO(payload) if isinstance(payload,(bytes,bytearray)) else payload
        source.seek(0);root=Path(cache_root)/'results';root.mkdir(parents=True,exist_ok=True)
        fd,temp=tempfile.mkstemp(prefix='import_',suffix='.tmp',dir=root)
        try:
            with os.fdopen(fd,'wb') as handle:shutil.copyfileobj(source,handle,length=1024*1024)
            path=Path(temp)
            with zipfile.ZipFile(path) as z:json.loads(z.read('manifest.json'))
            # Content hash by blocks, avoiding a second in-memory ZIP copy.
            digest=hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
            target=root/('import_'+digest.hexdigest()[:16]+'.zip');os.replace(path,target);path=target
        except BaseException:
            Path(temp).unlink(missing_ok=True);raise
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('manifest.json'))
        if 'process_memory.json' in z.namelist():manifest['process_memory']=json.loads(z.read('process_memory.json'))
    if manifest.get('version') != VERSION:
        raise ValueError('本页仅打开gpt1.10结果；gpt1.9请走旧结果加本地行情缓存入口。')
    tables=ZipTables(path)
    required={'events','weekly_summary','selections','coverage','rank_layers','industry_adjustment_summary',
        'signal_calendar','top5_weekly_history','path_diagnostics','entry_exit_summary','entry_rank_summary',
        'entry_date_comparison','survivor_summary','final_summary','industry_adjustment_dates','pool_filter_counts',
        'industry_profile','entry_date_summary','entry_top5_details','stop_timing_summary','ranking_summary','ranking_dates','condition_dates','condition_summary','feature_readiness'}
    if not required.issubset(tables):raise ValueError('结果文件缺少必要表格：'+', '.join(sorted(required-set(tables))))
    return tables,manifest,None,str(path)


def replay_research(source,cache_root,progress=lambda _:None):
    """Reuse saved paths; derive new signal-time features from the matching local cache."""
    root=Path(cache_root);root.mkdir(parents=True,exist_ok=True)
    if not (root/CACHE_SCHEMA/'tech_bars_v1.sqlite').is_file():
        raise ValueError('此目录没有原行情SQLite缓存。旧ZIP不能单独计算新量价特征，请切换行情缓存计算；已有其他目录缓存可填写其原目录。')
    with research_lock(root), tempfile.TemporaryDirectory(prefix='replay110_',dir=root) as temp:
        work=Path(temp);marks=SQLSpool(work/'marks.sqlite');paths=SQLSpool(work/'paths.sqlite')
        stages=[];started=time.monotonic()
        def report(message):
            stages.append(dict(stage=message,elapsed_seconds=round(time.monotonic()-started,1),**memory_mb()))
            progress(message)
        try:
            with zipfile.ZipFile(source) as z:
                old=json.loads(z.read('manifest.json'))
                if old.get('version')!='gpt1.9':raise ValueError('此入口需要完整gpt1.9结果与同批行情缓存；旧ZIP单独不足以计算量价特征')
                for rule in ['中期趋势A','买卖执行','成交与成本','周频']:
                    if old.get('rules',{}).get(rule)!=RULES.get(rule):raise ValueError('旧包规则不一致：'+rule)
                cfg=Config(**old['config'])
                report('复用旧结果：读取信号时点因子，不下载行情')
                inputs=['event_id','ts_code','name','signal_date','signal_i','year','signal_week','pool_pass','pool_known',
                    'industry_key','industry_ambiguous','circ_mv_yi','signal_close','history_ready','mom_raw','short_return_pct']+PRICE_ENTRY_FEATURES
                chunks=[];old_selected=set()
                for chunk in pd.read_csv(z.open('factor_candidates.csv'),usecols=inputs+['selected_entry_trend'],
                    dtype={'year':str,'ts_code':str},float_precision='round_trip',chunksize=10000):
                    old_selected.update(chunk.loc[chunk.selected_entry_trend,'event_id'])
                    chunk['signal_date']=pd.to_datetime(chunk.signal_date)
                    chunks.append(chunk[inputs])
                f=compact_frame(pd.concat(chunks,ignore_index=True));chunks.clear()
                cache_audit=enrich_cached_features(f,pd.read_csv(z.open('universe.csv')),old,root,report)
                f=compact_frame(score_cross_sections(f));gc.collect()
                if set(f.loc[f.selected_entry_trend,'event_id'])!=old_selected:
                    raise ValueError('复算的中期趋势前五名与旧版不一致，停止以防数据或排名口径变动')
                common=f[f.eligible_entry_trend].copy();ids=set(common.event_id)
                # Read all old common-pool event ages to include dates with no A candidates.
                header=pd.read_csv(z.open('events.csv'),nrows=0).columns
                path_cols=['event_id','initial_stop_adj','initial_stop_raw','filled','closed','resolved','status','exit_reason',
                    'buy_i','sell_i','buy_date','sell_date','buy_adj','buy_raw','sell_adj','risk_pct_actual','r_amount',
                    'net_pct','order_net_pct','hold_days','max_close_gain_pct','unknown_from','exit_delay_days','exit_at_open',
                    'base_pass','mature_weeks']
                path_cols=[c for c in path_cols if c in header];chunks=[];age={}
                for chunk in pd.read_csv(z.open('events.csv'),usecols=path_cols+['signal_date'],float_precision='round_trip',chunksize=10000):
                    for day,p in chunk.groupby('signal_date',sort=False):
                        if p.mature_weeks.nunique()!=1 or (day in age and age[day]!=int(p.mature_weeks.iloc[0])):
                            raise ValueError('旧事件交易周龄不一致')
                        age[day]=int(p.mature_weeks.iloc[0])
                    p=chunk[chunk.event_id.isin(ids)][path_cols].copy()
                    for c in ['buy_date','sell_date']:
                        if c in p:p[c]=pd.to_datetime(p[c],errors='raise')
                    if len(p):chunks.append(p)
                if chunks:life=pd.concat(chunks,ignore_index=True)
                else:life=pd.DataFrame(columns=path_cols)
                chunks.clear()
                if life.event_id.duplicated().any() or set(life.event_id)!=ids:raise ValueError('旧结果缺少或重复A池交易路径')
                e=compact_frame(common.merge(life,on='event_id',validate='one_to_one'));del common,life;gc.collect()
                schedule=pd.read_csv(z.open('signal_calendar.csv'))
                schedule['mature_weeks']=schedule.signal_date.map(age).combine_first(schedule.get('mature_weeks',pd.Series(dtype=float)))
                if schedule.mature_weeks.isna().any():raise ValueError('旧包缺少批次交易周龄，请走行情缓存计算入口')
                schedule['signal_date']=pd.to_datetime(schedule.signal_date)
                report('读取A池周度标记与价格路径；其余科技股票不进入本轮诊断')
                for filename,sink in [('weekly_marks.csv',marks),('path_diagnostics.csv',paths)]:
                    for chunk in pd.read_csv(z.open(filename),float_precision='round_trip',chunksize=10000):
                        p=chunk[chunk.event_id.isin(ids)].copy()
                        if 'mark_date' in p:p['mark_date']=pd.to_datetime(p.mark_date)
                        sink.append(p)
                    sink.finish()
                tables=DiskTables(work/'reports')
                tables['cache_feature_audit']=pd.DataFrame([cache_audit])
                build_reports(e,marks,f,schedule,None,cfg,report,output=tables)
                tables.update(condition_reports(e,marks,schedule,None,report))
                # Factors have been saved; diagnostic functions only need compact event columns.
                del f;gc.collect()
                entry_exit_reports(e,marks,paths,report,output=tables)
                tables.update(entry_stage_reports(e,marks,None,schedule,None,report))
                for name in ['universe','industry_intervals','data_issues']:
                    try:tables[name]=pd.read_csv(z.open(name+'.csv'))
                    except pd.errors.EmptyDataError:tables[name]=pd.DataFrame()
                del e;gc.collect()
            manifest=dict(old,version=VERSION,strategy_version=VERSION,rules=RULES,
                created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),source_version=old['version'],
                replay=True,runtime_memory_mb=memory_mb(),memory_notes=['只计算共同A池','诊断子组逐个生成','逐周视图只保留统计字段'],
                limitations=['历史已反复研究，不是新样本外','共同池改为A池，不能与旧全科技基准直接拼接',
                    '复用原数据截止日，不代表实时选股','固定量价条件实验，不自动选择赢家'])
            manifest.pop('learning',None)
            manifest['feature_experiment']=FEATURE_SPEC
            tables['resource_usage']=pd.DataFrame(stages)
            digest=hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()[:12]
            path=root/'results'/f'{VERSION}_replay_{digest}{RUN_SUFFIX}.zip'
            write_zip(tables,manifest,path)
            return ZipTables(path),manifest,None,str(path)
        finally:marks.close();paths.close();gc.collect()


def main():
    import streamlit as st
    st.set_page_config(page_title='gpt1.10 同池排序验证',layout='wide')
    st.title('gpt1.10 · 活跃度与收缩突破入场实验')
    st.caption('固定比较三个入场条件，统一上涨效率排序与退出规则；先检验条件是否有效，不自动挑选历史赢家。')
    if 'gpt110_job' in st.session_state:
        render_job(st);return
    with st.sidebar:
        cache=st.text_input('行情缓存目录',value='tech_swing_cache')
        mode=st.radio('运行方式',['旧结果＋本地行情缓存','行情缓存计算','打开gpt1.10结果'])
        upload=None;run=False
        if mode!='行情缓存计算':
            upload=st.file_uploader('上传完整结果ZIP',type=['zip'],key=f"input110_{st.session_state.get('upload_revision110',0)}")
            run=st.button('计算新入场条件' if mode.startswith('旧结果') else '打开结果',disabled=upload is None,type='primary')
        else:
            token=st.text_input('Tushare Token',type='password',value=os.environ.get('TUSHARE_TOKEN',''))
            start=st.date_input('信号开始',date(2022,1,1));end=st.date_input('信号结束',latest_ready_day().date())
            price=st.number_input('最低股价（严格大于）',min_value=0.,value=10.)
            min_mv=st.number_input('流通市值下限（亿元）',min_value=0.,value=50.)
            max_mv=st.number_input('流通市值上限（亿元）',min_value=0.,value=1000.)
            run=st.button('四路下载/复用缓存并计算',type='primary')
        st.caption('旧结果入口需gpt1.9 ZIP及原行情SQLite缓存，无需Token。缓存不存在或不匹配时，请使用行情缓存计算；成功日期不重复下载。')
    if run:
        for key in ['gpt110_result','gpt19_result','gpt18_result','gpt17_result']:st.session_state.pop(key,None)
        try:
            if mode=='行情缓存计算':
                if not token or start>end or min_mv>=max_mv:raise ValueError('请检查Token、日期和市值范围')
                cfg=Config(start=ds(start),end=ds(end),min_price=price,min_mv=min_mv,max_mv=max_mv)
                st.session_state['gpt110_job']=new_job(cache,'fresh',token=token,cfg=cfg)
            elif mode.startswith('旧结果'):
                st.session_state['gpt110_job']=new_job(cache,'replay',upload=upload)
            else:st.session_state['gpt110_result']=load_result(upload,cache)
            upload=None
            st.session_state['upload_revision110']=st.session_state.get('upload_revision110',0)+1
            st.rerun()
        except Exception as exc:st.error(f'本次未完成：{exc}')
    with st.expander('固定研究规则'):st.text(STUDY_NOTES)
    if 'gpt110_result' not in st.session_state:
        saved=sorted((Path(cache)/'results').glob('gpt1.10*.zip'),key=lambda p:p.stat().st_mtime,reverse=True)
        if saved:
            chosen=st.selectbox('已完成的本地结果',saved,format_func=lambda p:p.name)
            if st.button('恢复本地结果'):
                try:st.session_state['gpt110_result']=load_result(chosen);st.rerun()
                except Exception as exc:st.error(str(exc))
        st.info('优先用gpt1.9结果加原行情缓存验证。新条件可能减少候选，年度空窗会如实报告；仅凭旧ZIP无法补出成交量价数据。');return
    tables,manifest,_,path=st.session_state['gpt110_result']
    st.write(f"结果 {manifest['version']}｜行情截至 {manifest['data_end']}｜股票名单 {manifest['universe_size']} 只")
    st.caption('共同池基准仅包含中期趋势A候选。独立事件可重叠，统计不是账户收益。')
    for warning in manifest.get('warnings',[]):st.warning(warning)
    if st.checkbox('准备下载完整结果'):
        with Path(path).open('rb') as handle:st.download_button('下载结果',handle,file_name=f"gpt1.10_{manifest['config']['start']}_{manifest['config']['end']}.zip",mime='application/zip',on_click='ignore')
    method=st.selectbox('研究组',list(TRIALS)+['中期趋势','同数据完整池'])
    cv=tables['coverage'];year=st.selectbox('信号年份',['全部']+sorted(cv.year.astype(str).unique().tolist()) if not cv.empty else ['全部'])
    section=st.radio('查看报告',['入场条件检验','逐周收益','前五名明细','同池前五名增益','价格路径诊断','覆盖与运行记录'],horizontal=True)
    def subset(df):
        if df.empty:return df
        if 'strategy' in df:df=df[df.strategy.eq(method)]
        if 'year' in df:
            if df.year.astype(str).eq('全部').any():df=df[df.year.astype(str).eq(year)]
            elif year!='全部':df=df[df.year.astype(str).eq(year)]
        return df
    seq=0
    def show(df):
        nonlocal seq
        seq+=1
        if len(df)>500:
            page=st.number_input(f'表{seq}页码（共{len(df)}行）',min_value=1,max_value=(len(df)+499)//500,value=1,key=f'p110_{section}_{seq}')
            df=df.iloc[(page-1)*500:page*500]
        st.dataframe(df.rename(columns=LABELS),hide_index=True,use_container_width=True)
    if section=='入场条件检验':
        d=tables['condition_summary']
        show(subset(d) if method in TRIALS else (d[d.year.astype(str).eq(year)] if not d.empty else d))
        st.caption('比较特征完整池内满足条件与不满足条件的全部订单，先同日比较再对日期等权。未知收益排除并披露，严格口径要求两侧全部已知。此表不经过前五名排序。')
    elif section=='逐周收益':
        group=st.selectbox('样本组',['前五名']+[f'第{i}名' for i in range(1,6)])
        d=subset(tables['weekly_summary']);show(d[d.group.eq(group)] if not d.empty else d)
        st.caption('W1/W2…按5/10…个市场交易日，退出后冻结收益，无固定持仓期限。条件组信号日期可能不同，效果比较请看同日对照。')
        with st.expander('整个A候选池基准'):
            d=tables['weekly_summary'];show(d[d.strategy.eq('共同池基准')&d.year.astype(str).eq(year)] if not d.empty else d)
    elif section=='前五名明细':
        s=subset(tables['selections'])
        if s.empty:st.info('没有符合条件的候选。')
        else:
            day=st.selectbox('信号批次',sorted(s.signal_date.unique(),reverse=True));p=s[s.signal_date.eq(day)].sort_values('rank')
            show(p[['ts_code','name','rank','score','activity_ratio','squeeze_ratio','breakout_pct','signal_date','buy_date','buy_raw','initial_stop_raw','status','sell_date','exit_reason','net_pct']])
            h=tables.select('top5_weekly_history',strategy=method,event_id=set(p.event_id))
            if not h.empty:
                wide=h.pivot(index=['ts_code','name','rank'],columns='week_no',values='week_net_pct');wide.columns=[f'W{int(w)}净收益%' for w in wide.columns];show(wide.reset_index())
    elif section=='同池前五名增益':
        show(subset(tables['ranking_summary']))
        st.caption('同日先比较前五名与其余A候选，再对日期等权。前五须全部已知，其余未知排除并披露；严格口径要求其余也全部已知。取消订单计0。')
        w=st.selectbox('分层观察周次',[1,2,4,8,12]);d=subset(tables['rank_layers']);show(d[d.week_no.eq(w)] if not d.empty else d)
        with st.expander('与同数据完整池前五名的同日对照',expanded=True):
            d=tables['industry_adjustment_summary'];show(d[d.year.astype(str).eq(year)&d.family.eq(method)] if not d.empty else d)
    elif section=='价格路径诊断':
        d=subset(tables['entry_exit_summary']);show(d[d.group.eq('前五名')] if not d.empty else d)
        st.caption('诊断继续观察卖出后的股价。最大上涨不等于可兑现利润；路径缺失不计为零。')
        show(subset(tables['entry_rank_summary']))
        with st.expander('初始止损与触及10%的先后'):show(subset(tables['stop_timing_summary']))
    else:
        show(subset(cv));show(tables.get('feature_readiness',pd.DataFrame()));show(tables.get('cache_feature_audit',pd.DataFrame()))
        show(tables.get('data_issues',pd.DataFrame()));show(tables.get('resource_usage',pd.DataFrame()));st.json(manifest)


LABELS.update({'never_up10_count':'始终未触及上涨10%事件数','never_up10_pct':'始终未触及上涨10%比例%',
    'raw_date_mean_pct':'上涨效率参照均益%','adjusted_date_mean_pct':'当前排序均益%','family':'排序方法','mean_delta_pp':'当前减效率百分点','top_mean':'前五名同日订单均益%','rest_mean':'其余候选同日订单均益%','delta_pp':'前五减其余百分点',
    'better_dates_pct':'前五胜出日期%','rest_unknown':'其余未知事件数','rest_events':'其余事件数','strict_dates':'严格可比较日期数',
    'strict_delta_pp':'严格口径增益百分点','score':'排序分数（各方法定义不同）'})

LABELS.update({'activity_ratio':'换手活跃度倍数','squeeze_ratio':'前4周相对波幅/更早12周',
 'breakout_pct':'收盘突破前13周高点%','score':'26周上涨效率','feature_complete':'特征完整候选数',
 'a_candidates':'原A池候选数','activity_missing':'活跃度缺失数','shape_missing':'形态数据缺失数',
 'activity_candidates':'活跃度提升候选数','squeeze_candidates':'收缩后突破候选数','joint_candidates':'活跃突破候选数',
 'passing_events':'满足条件订单数','failing_events':'不满足条件订单数','passing_unknown':'满足条件收益未知数',
 'failing_unknown':'不满足条件收益未知数','passing_mean':'满足条件同日均益%',
 'failing_mean':'不满足条件同日均益%','condition_delta_pp':'满足减不满足百分点',
 'strict_condition_delta_pp':'两侧全部已知时增益百分点','family':'研究组',
 'raw_date_mean_pct':'同数据完整池前五均益%','adjusted_date_mean_pct':'条件组前五均益%',
 'mean_delta_pp':'条件组减完整池百分点','cache_days':'缓存交易日数','network_calls':'行情网络请求数',
 'hash_matches':'缓存摘要一致','baseline_features_match':'旧价格特征重算一致'})


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

def entry_self_test():
    # Engineered completed weekly pattern: trend, two-week net retreat, close above previous high.
    full=pd.bdate_range('2021-01-04',periods=380);cal=full[:375];t=np.arange(len(cal))//5
    weekly=100+np.arange(75)*.6;weekly[62:66]=[145,144,143,146]
    cl=weekly[t];g=pd.DataFrame(dict(open=cl,high=cl+.1,low=cl-.1,close=cl,adj_factor=1.,vol=100.,up_limit=cl*1.2,down_limit=cl*.8),index=cal)
    g['ac']=g.close;g['ah']=g.high;g['al']=g.low
    idx=complete_week_indices(cal,full);f=weekly_features(g,idx);day=cal[65*5+4];row=f.loc[day]
    assert row.trend_ok and row.pullback_ok and row.reclaim_ok
    expected=100*(weekly[63]-weekly[37])/np.abs(np.diff(weekly[37:64])).sum()
    assert np.isclose(row.trend_efficiency,expected)
    # C fails when current close fails to exceed prior high; B still holds.
    x=g.copy();x.loc[x.index.to_period('W-FRI')==day.to_period('W-FRI'),'ac']=143
    q=weekly_features(x,idx).loc[day];assert q.pullback_ok and not q.reclaim_ok
    # Mutating any future price leaves earlier features identical, including unfinished-week prefixes.
    x=g.copy();x.loc[x.index>day,['ac','ah','al']]*=2
    pd.testing.assert_frame_equal(f.loc[:day],weekly_features(x,idx).loc[:day])
    cut=cal.get_loc(day)+1
    pd.testing.assert_frame_equal(f.loc[:day],weekly_features(g.iloc[:cut],complete_week_indices(cal[:cut],full)))
    # Cross-sectional ranking and nested gates, using a constant common stock pool.
    feat=pd.DataFrame(dict(signal_date=[day]*12,ts_code=[str(i) for i in range(12)],industry_key='A',pool_pass=True,history_ready=True,
     mom_raw=np.arange(1.,13.),short_return_pct=np.arange(12.)-6,circ_mv_yi=np.arange(12.)+50,
     trend_efficiency=np.arange(1.,13.)*5,trend_ok=True,pullback_ok=[True]*8+[False]*4,reclaim_ok=[True]*3+[False]*9))
    q=score_cross_sections(feat)
    for col in STATIC_STRATEGIES.values():
        assert q['eligible_'+col].sum()==12 and q['selected_'+col].sum()==5
    feat['short_return_pct']=np.arange(1.,13.)
    q=score_cross_sections(feat)
    assert q.selected_entry_trend.sum()==5
    feat['trend_ok']=False
    q=score_cross_sections(feat)
    assert not q[[f'selected_{c}' for c in STRATEGIES.values()]].any().any()

def self_test():
    feature_self_test()
    entry_self_test()
    entry_self_test()
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
    w=g.groupby(g.index.to_period('W-FRI'),observed=True).ac.last();j=60
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
    t['trend_ok']=t.mom_raw.gt(0);t['trend_efficiency']=50.
    scored=score_cross_sections(t)
    assert scored.common_pass.all() and scored.peer_count.eq(11).all()
    assert np.isclose(scored.mom_adj.iloc[0],0-np.median(np.arange(1.,12.)))
    assert all(scored['selected_'+c].sum()==5 for c in STATIC_STRATEGIES.values())
    assert scored.layer_entry_trend.nunique()==1
    ties=t.copy();ties['mom_raw']=0.;ties['short_return_pct']=0.;ties['trend_ok']=False
    q=score_cross_sections(ties);assert not q[[f'selected_{c}' for c in STRATEGIES.values()]].any().any()
    assert not score_cross_sections(t.iloc[:10]).common_pass.any()
    g=pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,adj_factor=1.,up_limit=150.,down_limit=50.,vol=100.),index=full[:20])
    g.loc[g.index[1],'low']=92
    out,_=lifecycle(g,0);assert out['sell_i']==1 and np.isclose(out['risk_pct_actual'],8)
    assert np.isclose(out['initial_stop_raw'],100.1*.92)
    g.loc[g.index[0],'low']=90;g.loc[g.index[1],['open','low','close']]=[91,90,92]
    out,_=lifecycle(g,0);assert out['sell_i']==1 and np.isclose(out['sell_adj'],91*.999)
    print(f'{VERSION} self-test PASS: leave-one-out peers, historical industry intervals, no future weekly prices, score ties, small industries, frozen 8% risk and T+1. No profitability claim.')


if __name__=='__main__':
    if '--worker' in sys.argv:sys.exit(worker_main(sys.argv[sys.argv.index('--worker')+1]))
    elif '--self-test' in sys.argv:self_test()
    elif '--replay' in sys.argv:
        import argparse
        parser=argparse.ArgumentParser();parser.add_argument('--replay',required=True);parser.add_argument('--output-dir',default='tech_swing_cache')
        args=parser.parse_args();result=replay_research(args.replay,args.output_dir,lambda text:print(text,flush=True));print(result[3])
    else:main()
