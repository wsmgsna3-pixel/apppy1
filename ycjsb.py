# -*- coding: utf-8 -*-
"""
周线 SKDJ 翻转打分系统 (V14.5-S1 稳定修复版)
------------------------------------------------
1. 周线SKDJ信号、原评分权重、Top N、买入条件和生命周期与V14.5完全一致。
2. 行情改为按交易日原子分片保存，单日损坏不会毁掉全部两小时缓存。
3. 历史结果、扫描账本和任务状态均原子保存；网络重连后可从周次断点续跑。
4. 下载和报告始终从磁盘结果加载，页面重跑不会回到无结果初始状态。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import os
import re
import pickle
import json
import hashlib
import tempfile
import shutil
import gc

try:
    import fcntl
except ImportError:  # Windows 本地运行时使用后备锁；Streamlit Cloud 为 Linux
    fcntl = None

warnings.filterwarnings("ignore")

# ---------------------------
# 全局持久化缓存配置
# ---------------------------
APP_VERSION = "V14.5-S1"
CHECKPOINT_FILE = "skdj_v14_5_stable_history.csv"
SCAN_LEDGER_FILE = "skdj_v14_5_stable_scanned_dates.csv"
RUN_TASK_FILE = "skdj_v14_5_stable_running_task.json"
RUN_LOCK_FILE = "skdj_v14_5_stable_running.lock"
MARKET_CACHE_ROOT = "skdj_v14_5_stable_market_cache"

# ---------------------------
# 页面基础配置
# ---------------------------
st.set_page_config(page_title="SKDJ V14.5-S1 稳定修复版", layout="wide")
st.title("🔬 周线 SKDJ 底部脱离系统 (V14.5-S1 稳定修复版)")
st.markdown("🔒 **策略与V14.5保持一致；本版只修复缓存、断点续跑、下载和页面崩溃问题**")

# ---------------------------
# Token 清洗与安全请求模块
# ---------------------------
def clean_token_str(raw_token: str) -> str:
    if not raw_token: return ""
    return re.sub(r'[\s\u3000\ufeff\xa0\r\n]+', '', str(raw_token)).strip()

def verify_token_connection(token_str: str):
    if not token_str:
        return False, "Token 为空，请在侧边栏填入 Token。"
    try:
        ts.set_token(token_str)
        pro = ts.pro_api(token_str)
        test_df = pro.trade_cal(exchange='SSE', start_date='20260801', end_date='20260805')
        if test_df is not None and not test_df.empty:
            return True, "验证通过"
        return False, "Token 校验未返回数据，请检查网络连接。"
    except Exception as e:
        err_msg = str(e)
        if "token不对" in err_msg or "-40001" in err_msg:
            return False, "您的 Token 不正确，请检查复制内容。"
        return False, f"接口校验失败: {err_msg}"

def safe_tushare_call(func, max_retries=3, sleep_time=0.8, **kwargs):
    for attempt in range(max_retries):
        try:
            df = func(**kwargs)
            if df is not None and not df.empty:
                return df
            time.sleep(sleep_time)
        except Exception:
            time.sleep(sleep_time * (attempt + 1))
    return pd.DataFrame()


def parse_yyyymmdd(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\.0$", "", str(value)).replace("-", "")
    return text if re.fullmatch(r"\d{8}", text) else None


def atomic_write_csv(df, path):
    """完整写入临时文件后原子替换，杜绝断线留下半行CSV。"""
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        with open(tmp_path, "rb") as file_obj:
            os.fsync(file_obj.fileno())
        if os.path.exists(path):
            try:
                shutil.copy2(path, path + ".bak")
            except OSError:
                pass
        elif os.path.exists(path + ".bak"):
            os.remove(path + ".bak")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_csv_safe(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    for candidate in (path, path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            return pd.read_csv(candidate, encoding="utf-8-sig", low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError):
            continue
    return pd.DataFrame()


def atomic_write_json(value, path):
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        if os.path.exists(path):
            try:
                shutil.copy2(path, path + ".bak")
            except OSError:
                pass
        elif os.path.exists(path + ".bak"):
            os.remove(path + ".bak")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_json_safe(path):
    if not os.path.exists(path):
        return {}
    for candidate in (path, path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as file_obj:
                value = json.load(file_obj)
            if isinstance(value, dict):
                return value
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return {}


def remove_with_backup(path):
    for candidate in (path, path + ".bak"):
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except OSError:
            pass


def append_checkpoint_atomic(new_rows):
    existing = read_csv_safe(CHECKPOINT_FILE)
    combined = pd.concat([existing, new_rows], ignore_index=True, sort=False) if not existing.empty else new_rows.copy()
    if "Trade_Date" in combined.columns:
        combined["Trade_Date"] = combined["Trade_Date"].map(parse_yyyymmdd)
    keys = [col for col in ("Config_ID", "Trade_Date", "ts_code") if col in combined.columns]
    if keys:
        combined = combined.drop_duplicates(keys, keep="last")
    sort_cols = [col for col in ("Trade_Date", "Rank") if col in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols, kind="mergesort")
    atomic_write_csv(combined.reset_index(drop=True), CHECKPOINT_FILE)


def mark_scan_complete(trade_date, raw_signal_count, selected_count, config_id):
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    row = pd.DataFrame([{
        "Trade_Date": str(trade_date),
        "Raw_Signal_Count": int(raw_signal_count),
        "Selected_Count": int(selected_count),
        "Scan_Status": "COMPLETED",
        "Config_ID": config_id,
        "Updated_At": datetime.now().isoformat(timespec="seconds"),
    }])
    ledger = pd.concat([ledger, row], ignore_index=True, sort=False) if not ledger.empty else row
    ledger["Trade_Date"] = ledger["Trade_Date"].map(parse_yyyymmdd)
    ledger = ledger.drop_duplicates(["Trade_Date", "Config_ID"], keep="last")
    atomic_write_csv(ledger.sort_values("Trade_Date").reset_index(drop=True), SCAN_LEDGER_FILE)


_RUN_LOCK_HANDLE = None


def acquire_run_lock(stale_seconds=600):
    """防止双击或两个页面会话同时写同一缓存；进程终止时系统自动释放锁。"""
    global _RUN_LOCK_HANDLE
    if fcntl is not None:
        try:
            lock_handle = open(RUN_LOCK_FILE, "a+", encoding="utf-8")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_handle.seek(0)
            lock_handle.truncate()
            lock_handle.write(str(time.time()))
            lock_handle.flush()
            _RUN_LOCK_HANDLE = lock_handle
            return True
        except (OSError, BlockingIOError):
            try:
                lock_handle.close()
            except Exception:
                pass
            return False

    if os.path.exists(RUN_LOCK_FILE):
        try:
            if time.time() - os.path.getmtime(RUN_LOCK_FILE) > stale_seconds:
                os.remove(RUN_LOCK_FILE)
        except OSError:
            pass
    try:
        fd = os.open(RUN_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(time.time()).encode("utf-8"))
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_run_lock():
    global _RUN_LOCK_HANDLE
    try:
        if _RUN_LOCK_HANDLE is not None:
            if fcntl is not None:
                fcntl.flock(_RUN_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
            _RUN_LOCK_HANDLE.close()
            _RUN_LOCK_HANDLE = None
            # flock 锁文件保留在磁盘；真正的锁随文件句柄关闭而释放。
            return
        if fcntl is None and os.path.exists(RUN_LOCK_FILE):
            os.remove(RUN_LOCK_FILE)
    except OSError:
        pass

# ---------------------------
# 科技白名单池构建
# ---------------------------
@st.cache_data(ttl=3600*24*7, show_spinner=False)
def load_custom_tech_whitelist(token):
    token_c = clean_token_str(token)
    if not token_c: return set(), {}
    
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    
    stock_basic = safe_tushare_call(pro.stock_basic, list_status='L', fields='ts_code,symbol,name,industry,market,list_date')
    if stock_basic.empty: return set(), {}
        
    BOARDS = ("主板", "创业板", "科创板")
    valid_stocks = stock_basic[stock_basic['market'].isin(BOARDS)].copy()
    valid_stocks = valid_stocks[~valid_stocks['name'].str.contains('ST|退', na=False)]
    valid_stocks = valid_stocks[~valid_stocks['ts_code'].str.startswith('92')]
    
    CORE_TECH_L1 = {"电子", "计算机", "通信", "国防军工"}
    EXTENDED_TECH_L1 = {"机械设备", "电力设备", "医药生物", "汽车", "基础化工", "有色金属", "建筑材料"}
    TECH_INDUSTRY_KEYWORDS = {
        "半导体", "电子元件", "元件", "光学光电子", "消费电子", "电子化学品",
        "计算机设备", "软件开发", "IT服务", "通信设备", "军工电子", "航空装备",
        "航天装备", "自动化设备", "机器人", "激光设备", "工控设备", "仪器仪表",
        "电池", "光伏设备", "风电设备", "电网设备", "电机", "医疗器械",
        "生物制品", "汽车电子", "金属新材料", "非金属材料", "膜材料", "碳纤维",
    }
    
    sw_indices = safe_tushare_call(pro.index_classify, level='L1', src='SW2021')
    tech_l1_names = CORE_TECH_L1.union(EXTENDED_TECH_L1)
    target_sw = sw_indices[sw_indices['industry_name'].isin(tech_l1_names)] if not sw_indices.empty else pd.DataFrame()
    
    stock_sw_map = {}
    if not target_sw.empty:
        for _, s_row in target_sw.iterrows():
            idx_code = s_row['index_code']
            ind_name = s_row['industry_name']
            m_df = safe_tushare_call(pro.index_member, index_code=idx_code, is_new='Y')
            if not m_df.empty:
                for c_code in m_df['con_code']:
                    stock_sw_map[c_code] = ind_name
            time.sleep(0.03)
            
    whitelist_set = set()
    name_map = dict(zip(stock_basic['ts_code'], stock_basic['name']))
    
    for _, row in valid_stocks.iterrows():
        code = row['ts_code']
        ind_basic = str(row['industry']) if pd.notna(row['industry']) else ""
        sw_l1 = stock_sw_map.get(code, "")
        
        if sw_l1 in CORE_TECH_L1: whitelist_set.add(code); continue
        if sw_l1 in EXTENDED_TECH_L1:
            if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS) or ind_basic == "" or sw_l1 in {"机械设备", "电力设备", "医药生物"}:
                whitelist_set.add(code); continue
        if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS): whitelist_set.add(code); continue

    return whitelist_set, name_map

# ---------------------------
# 稳定版行情缓存：每个交易日独立原子分片
# ---------------------------
def _pool_cache_dir(whitelist_set):
    pool_hash = hashlib.sha1("|".join(sorted(whitelist_set)).encode("utf-8")).hexdigest()[:12]
    cache_dir = os.path.join(MARKET_CACHE_ROOT, pool_hash)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir, pool_hash


def _valid_market_partition(payload, trade_date, pool_hash):
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != 1 or payload.get("trade_date") != str(trade_date):
        return False
    if payload.get("pool_hash") != pool_hash:
        return False
    daily = payload.get("daily")
    adj = payload.get("adj")
    basic = payload.get("daily_basic")
    if not isinstance(daily, pd.DataFrame) or not isinstance(adj, pd.DataFrame):
        return False
    if not isinstance(basic, pd.DataFrame):
        return False
    if int(payload.get("raw_daily_count", 0)) < 1000 or int(payload.get("raw_adj_count", 0)) < 1000:
        return False
    required_daily = {"ts_code", "trade_date", "open", "high", "low", "close", "vol"}
    return (
        not daily.empty and not adj.empty
        and required_daily.issubset(daily.columns)
        and {"ts_code", "trade_date", "adj_factor"}.issubset(adj.columns)
    )


def _atomic_write_pickle(payload, path):
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as file_obj:
            pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _read_market_partition(path, trade_date, pool_hash):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as file_obj:
            payload = pickle.load(file_obj)
        return payload if _valid_market_partition(payload, trade_date, pool_hash) else None
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        return None


def sync_market_data_incrementally(start_date, end_date, token, whitelist_set):
    """单日下载完成即提交；进程中断后只补未完成日期。"""
    token_c = clean_token_str(token)
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    cal_raw = safe_tushare_call(
        pro.trade_cal, exchange="SSE", start_date=start_date, end_date=end_date
    )
    if cal_raw.empty:
        return [], "", ""
    today_str = datetime.now().strftime("%Y%m%d")
    valid_dates = (
        cal_raw[(cal_raw["is_open"] == 1) & (cal_raw["cal_date"].astype(str) <= today_str)]
        .sort_values("cal_date")["cal_date"].astype(str).tolist()
    )
    cache_dir, pool_hash = _pool_cache_dir(whitelist_set)
    missing_dates = [
        trade_date for trade_date in valid_dates
        if _read_market_partition(
            os.path.join(cache_dir, f"{trade_date}.pkl"), trade_date, pool_hash
        ) is None
    ]
    if missing_dates:
        bar = st.progress(0, text=f"📥 从断点补充 {len(missing_dates)} 个交易日行情...")
        failed_dates = []
        for idx, trade_date in enumerate(missing_dates):
            daily_all = safe_tushare_call(pro.daily, max_retries=3, sleep_time=0.8, trade_date=trade_date)
            adj_all = safe_tushare_call(pro.adj_factor, max_retries=3, sleep_time=0.8, trade_date=trade_date)
            basic_all = safe_tushare_call(
                pro.daily_basic, max_retries=3, sleep_time=0.8,
                trade_date=trade_date, fields="ts_code,trade_date,circ_mv",
            )
            daily = daily_all[daily_all["ts_code"].isin(whitelist_set)].copy() if not daily_all.empty else pd.DataFrame()
            adj = adj_all[adj_all["ts_code"].isin(whitelist_set)].copy() if not adj_all.empty else pd.DataFrame()
            basic = basic_all[basic_all["ts_code"].isin(whitelist_set)].copy() if not basic_all.empty else pd.DataFrame()
            payload = {
                "version": 1,
                "trade_date": trade_date,
                "pool_hash": pool_hash,
                "raw_daily_count": int(len(daily_all)),
                "raw_adj_count": int(len(adj_all)),
                "daily": daily,
                "adj": adj,
                "daily_basic": basic,
            }
            if _valid_market_partition(payload, trade_date, pool_hash):
                _atomic_write_pickle(payload, os.path.join(cache_dir, f"{trade_date}.pkl"))
            else:
                failed_dates.append(trade_date)
            if (idx + 1) % 5 == 0 or idx == len(missing_dates) - 1:
                bar.progress(
                    (idx + 1) / len(missing_dates),
                    text=f"📥 行情同步 {idx + 1}/{len(missing_dates)}：{trade_date}",
                )
            time.sleep(0.12)
        bar.empty()
        if failed_dates:
            st.warning(
                f"有 {len(failed_dates)} 个交易日行情暂未返回；成功日期已保存，"
                "本次继续使用完整分片，下次运行只补缺失日期。"
            )
    return valid_dates, cache_dir, pool_hash


@st.cache_resource(ttl=3600 * 12, show_spinner=False)
def _build_market_index_from_partitions(valid_dates_key, cache_dir, pool_hash, whitelist_key, cache_stamp):
    del cache_stamp
    whitelist_set = set(whitelist_key)
    daily_parts, adj_parts, basic_parts = [], [], []
    for trade_date in valid_dates_key:
        payload = _read_market_partition(
            os.path.join(cache_dir, f"{trade_date}.pkl"), trade_date, pool_hash
        )
        if payload is None:
            continue
        daily_parts.append(payload["daily"])
        adj_parts.append(payload["adj"])
        if not payload["daily_basic"].empty:
            basic_parts.append(payload["daily_basic"])
    daily_raw = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    adj_raw = pd.concat(adj_parts, ignore_index=True) if adj_parts else pd.DataFrame()
    basic_raw = pd.concat(basic_parts, ignore_index=True) if basic_parts else pd.DataFrame()
    if daily_raw.empty or adj_raw.empty:
        return {}, pd.DataFrame()

    merged_all = daily_raw.merge(
        adj_raw[["ts_code", "trade_date", "adj_factor"]],
        on=["ts_code", "trade_date"], how="inner",
    )
    merged_all["trade_date_str"] = merged_all["trade_date"].astype(str)
    merged_all = merged_all.drop_duplicates(["ts_code", "trade_date_str"], keep="last")
    merged_all = merged_all.sort_values(["ts_code", "trade_date_str"])
    del daily_raw, adj_raw, daily_parts, adj_parts
    gc.collect()

    stock_qfq_dict = {}
    for ts_code, group in merged_all.groupby("ts_code"):
        df_g = group.copy()
        latest_adj = df_g["adj_factor"].iloc[-1]
        if latest_adj > 0:
            for col in ["open", "high", "low", "close", "pre_close"]:
                if col in df_g.columns:
                    df_g[col] = df_g[col] * df_g["adj_factor"] / latest_adj
        df_g = df_g.set_index("trade_date_str")
        stock_qfq_dict[ts_code] = df_g
    if not basic_raw.empty:
        basic_raw["trade_date"] = basic_raw["trade_date"].astype(str)
        basic_indexed = basic_raw.drop_duplicates(
            subset=["ts_code", "trade_date"]
        ).set_index(["trade_date", "ts_code"])
    else:
        basic_indexed = pd.DataFrame()
    return stock_qfq_dict, basic_indexed


def load_optimized_market_data(start_date, end_date, token, whitelist_keys):
    whitelist_set = set(whitelist_keys)
    valid_dates, cache_dir, pool_hash = sync_market_data_incrementally(
        start_date, end_date, token, whitelist_set
    )
    if not valid_dates:
        return {}, pd.DataFrame()
    partition_paths = [os.path.join(cache_dir, f"{date}.pkl") for date in valid_dates]
    stamp = (
        len([path for path in partition_paths if os.path.exists(path)]),
        max((os.path.getmtime(path) for path in partition_paths if os.path.exists(path)), default=0),
    )
    return _build_market_index_from_partitions(
        tuple(valid_dates), cache_dir, pool_hash, tuple(sorted(whitelist_set)), stamp
    )

# ---------------------------
# 🚀 核心引擎：翻转打分模型（通用于选股与回测）
# ---------------------------
def compute_breakout_signal(ts_code, end_date, stock_qfq_dict):
    if ts_code not in stock_qfq_dict: return {}
    df_full = stock_qfq_dict[ts_code]
    
    df_daily = df_full[df_full.index <= end_date]
    res = {}
    if df_daily.empty or len(df_daily) < 100: return res

    row_friday = df_daily.iloc[-1]
    is_20cm = any(ts_code.startswith(prefix) for prefix in ['300', '301', '688', '689'])
    limit_rate = 0.195 if is_20cm else 0.095
    pre_close_val = row_friday.get('pre_close', np.nan)
    if pd.isna(pre_close_val) or pre_close_val <= 0:
        pre_close_val = df_daily.iloc[-2]['close'] if len(df_daily) >= 2 else row_friday['open']
            
    is_friday_yiziban = (row_friday['high'] == row_friday['low']) and ((row_friday['close'] - pre_close_val) / pre_close_val >= limit_rate)
    if is_friday_yiziban: return res

    df = df_daily.copy().reset_index()
    df['dt'] = pd.to_datetime(df['trade_date_str'])
    df['year_week'] = df['dt'].dt.strftime('%G_%V') 

    weekly_df = df.groupby('year_week', as_index=False).agg({
        'trade_date_str': 'last', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'
    }).sort_values('trade_date_str').reset_index(drop=True)

    n, m = 6, 3
    if len(weekly_df) < n + 15: return res

    weekly_df['lowv'] = weekly_df['low'].rolling(window=n).min()
    weekly_df['highv'] = weekly_df['high'].rolling(window=n).max()
    diff = (weekly_df['highv'] - weekly_df['lowv']).replace(0, 0.001)

    raw_rsv = (weekly_df['close'] - weekly_df['lowv']) / diff * 100
    weekly_df['rsv'] = raw_rsv.ewm(span=m, adjust=False).mean()
    weekly_df['k'] = weekly_df['rsv'].ewm(span=m, adjust=False).mean()
    weekly_df['d'] = weekly_df['k'].rolling(window=m).mean()
    weekly_df['ma5_vol'] = weekly_df['vol'].shift(1).rolling(window=5).mean()
    weekly_df['ma20'] = weekly_df['close'].rolling(window=20).mean()

    curr_w = weekly_df.iloc[-1]
    prev_w = weekly_df.iloc[-2]
    
    if pd.isna(curr_w['k']) or pd.isna(prev_w['k']) or pd.isna(curr_w['d']): return res

    is_breakout_25 = (curr_w['k'] > 25.0) and (prev_w['k'] <= 25.0)
    is_bullish = curr_w['k'] > curr_w['d']
    if not (is_breakout_25 and is_bullish): return res

    recent_15_weeks = weekly_df.tail(15)
    k_history_before_breakout = recent_15_weeks['k'].iloc[:-1] 
    recent_k_min = k_history_before_breakout.min()
    weeks_under_25 = (k_history_before_breakout < 25.0).sum()
    
    ma20_curr = curr_w['ma20'] if pd.notna(curr_w['ma20']) else curr_w['close']
    trend_type = "均线上方" if curr_w['close'] >= ma20_curr else "均线下方(超跌)"
    vol_ratio = curr_w['vol'] / curr_w['ma5_vol'] if (pd.notna(curr_w['ma5_vol']) and curr_w['ma5_vol'] > 0) else 1.0

    res['is_buy_signal'] = True
    res['k'] = round(curr_w['k'], 2)
    res['d'] = round(curr_w['d'], 2)
    res['recent_k_min'] = round(recent_k_min, 2)
    res['weeks_under_25'] = int(weeks_under_25)
    res['signal_close'] = curr_w['close'] 
    res['trend_type'] = trend_type
    res['vol_ratio'] = round(vol_ratio, 2)
    
    score = 0.0
    if curr_w['close'] >= ma20_curr: score += 20.0
    else: score -= 5.0
        
    if 22.0 <= recent_k_min <= 25.0: score += 30.0    
    elif 15.0 <= recent_k_min < 22.0: score += 15.0   
    elif 5.0 <= recent_k_min < 15.0: score -= 10.0    
    else: score -= 25.0                               
        
    if 1 <= weeks_under_25 <= 2: score += 30.0        
    elif 3 <= weeks_under_25 <= 5: score += 15.0      
    elif 6 <= weeks_under_25 <= 9: score -= 5.0       
    else: score -= 20.0                               
        
    k_val = curr_w['k']
    if 25.0 < k_val <= 32.0: score += 10.0
    elif k_val > 38.0: score -= 10.0
        
    if 1.0 <= vol_ratio <= 2.5: score += 10.0
    elif vol_ratio > 4.0: score -= 15.0

    res['Total_Score'] = round(score, 1)
    return res

# === 第一部分结束，请等待粘贴第二部分 ===
# ---------------------------
# 🚀 独立出局系统 (仅限回测模式使用)
# ---------------------------
def track_future_performance(ts_code, selection_date, signal_close, stock_qfq_dict, hold_weeks=12):
    default_res = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    default_res.update({
        'Exit_Reason': '持仓中', 'Buy_Price': np.nan, 'Gap_pct (%)': np.nan, 
        'Exit_Date': None, 'Final_Return (%)': np.nan, 'Hold_Days': 0
    })
    
    if ts_code not in stock_qfq_dict: return default_res
    df_full = stock_qfq_dict[ts_code]
    hist_future = df_full[df_full.index > selection_date]
    results = default_res.copy()
    
    if hist_future.empty: return results

    next_row = hist_future.iloc[0]
    buy_price = next_row['open']
    if pd.isna(buy_price) or buy_price <= 0: return results

    is_20cm = any(ts_code.startswith(prefix) for prefix in ['300', '301', '688', '689'])
    limit_rate_pct = 19.0 if is_20cm else 9.5
    gap_pct = (buy_price - signal_close) / signal_close * 100.0
    
    is_monday_yiziban = (next_row['open'] == next_row['high'] == next_row['low']) and (gap_pct >= limit_rate_pct)
    if is_monday_yiziban:
        results['Exit_Reason'] = f"一字板无法买入(剔除: {round(gap_pct, 1)}%)"
        results['Buy_Price'] = round(buy_price, 2)  
        return results

    if is_20cm and gap_pct > 8.0:
        results['Exit_Reason'] = f"双创高开过大(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results
    elif not is_20cm and gap_pct > 5.0:
        results['Exit_Reason'] = f"主板高开过大(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results
    if gap_pct < -4.0:
        results['Exit_Reason'] = f"恶劣低开(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results

    results['Buy_Price'] = round(buy_price, 2)
    exit_triggered = False
    tier = 0  
    peak_price = buy_price
    pending_exit_reason = None  
    hard_stop_limit = -0.10 
    
    max_days = hold_weeks * 5
    for i in range(len(hist_future)):
        if i >= max_days: break 
            
        row = hist_future.iloc[i]
        day_count = i + 1
        current_week = ((day_count - 1) // 5) + 1 
        curr_open, curr_close, curr_high, curr_low = row['open'], row['close'], row['high'], row['low']
        curr_date = hist_future.index[i]
        
        if pending_exit_reason is not None and day_count >= 2:
            if "保本" in pending_exit_reason: final_return = 2.0  
            else: final_return = (curr_open - buy_price) / buy_price * 100.0
            exit_triggered = True
            results['Exit_Reason'] = pending_exit_reason
            results['Final_Return (%)'] = round(final_return, 2)
            results['Exit_Date'] = curr_date
            results['Hold_Days'] = day_count
            results[f'Return_W{current_week} (%)'] = round(final_return, 2)
            break
        
        peak_price = max(peak_price, curr_high)
        peak_profit_pct = (peak_price - buy_price) / buy_price
        
        if day_count >= 2:
            if (curr_low - buy_price) / buy_price <= hard_stop_limit:
                final_return = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
                exit_triggered = True
                results['Exit_Reason'] = "认栽出局(破-10%)"
                results['Final_Return (%)'] = round(final_return, 2)
                results['Exit_Date'] = curr_date
                results['Hold_Days'] = day_count
                results[f'Return_W{current_week} (%)'] = round(final_return, 2)
                break
        
        if tier == 0 and peak_profit_pct >= 0.10: tier = 1  
        if tier == 1:
            if curr_close <= buy_price * 1.02: pending_exit_reason = "保本离场(+2%)"
            elif peak_profit_pct >= 0.20: tier = 2  
        if tier == 2:
            giveback = (peak_price - curr_close) / peak_price
            if giveback >= 0.15: pending_exit_reason = "移动止盈(回撤15%)"
        
        if day_count == 5 and not exit_triggered and pending_exit_reason is None:
            w1_ret = (curr_close - buy_price) / buy_price * 100.0
            if w1_ret <= -3.0:
                exit_triggered = True
                results['Exit_Reason'] = f"首周不及预期截断({round(w1_ret, 1)}%)"
                results['Final_Return (%)'] = round(w1_ret, 2)
                results['Exit_Date'] = curr_date
                results['Hold_Days'] = 5
                results['Return_W1 (%)'] = round(w1_ret, 2)
                break
            
        if day_count % 5 == 0:
            results[f'Return_W{current_week} (%)'] = round((curr_close - buy_price) / buy_price * 100.0, 2)
            
    if not exit_triggered and len(hist_future) >= max_days:
        last_price = hist_future.iloc[max_days - 1]['close']
        final_return = (last_price - buy_price) / buy_price * 100.0
        results[f'Return_W{hold_weeks} (%)'] = round(final_return, 2)
        results['Exit_Reason'] = "12周期满平仓"
        results['Final_Return (%)'] = round(final_return, 2)
        results['Exit_Date'] = hist_future.index[max_days - 1]
        results['Hold_Days'] = max_days
        
    return results

def repair_checkpoint_df(df_in):
    df_out = df_in.copy()
    w_cols = [c for c in df_out.columns if c.startswith('Return_W') and c.endswith('(%)')]
    if w_cols: w_cols = sorted(w_cols, key=lambda x: int(x.replace('Return_W', '').replace(' (%)', '')))
    
    if 'Final_Return (%)' not in df_out.columns:
        def get_final_ret(r):
            if not w_cols: return 0.0
            rets = r[w_cols].dropna()
            return rets.iloc[-1] if not rets.empty else 0.0
        df_out['Final_Return (%)'] = df_out.apply(get_final_ret, axis=1)
    if 'Exit_Date' not in df_out.columns: df_out['Exit_Date'] = None
    if 'Hold_Days' not in df_out.columns:
        def get_hold_days(r):
            if not w_cols: return 0
            rets = r[w_cols].dropna()
            return len(rets) * 5 if not rets.empty else 0
        df_out['Hold_Days'] = df_out.apply(get_hold_days, axis=1)
    return df_out

# ---------------------------
# 稳定任务与原版扫描封装
# ---------------------------
WEEKS_PER_BATCH = 4


def make_config_id(top_n, min_price, min_mv, max_mv):
    payload = {
        "strategy": "V14.5-original",
        "top_n": int(top_n),
        "min_price": float(min_price),
        "min_mv": float(min_mv),
        "max_mv": float(max_mv),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def save_task(task):
    task = dict(task)
    task["Updated_At"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(task, RUN_TASK_FILE)


def completed_scan_dates(config_id):
    completed = set()
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    if not ledger.empty and {"Trade_Date", "Config_ID", "Scan_Status"}.issubset(ledger.columns):
        match = ledger[
            (ledger["Config_ID"].astype(str) == str(config_id))
            & (ledger["Scan_Status"].astype(str) == "COMPLETED")
        ]
        completed.update(filter(None, (parse_yyyymmdd(v) for v in match["Trade_Date"])))

    # 若进程恰好在“结果写入”与“账本写入”之间退出，结果文件本身也可恢复断点。
    history = read_csv_safe(CHECKPOINT_FILE)
    if not history.empty and "Trade_Date" in history.columns:
        if "Config_ID" in history.columns:
            history = history[history["Config_ID"].astype(str) == str(config_id)]
        completed.update(filter(None, (parse_yyyymmdd(v) for v in history["Trade_Date"])))
    return completed


def scan_one_date_original(
    date, whitelist_keys, basic_name_map, stock_qfq_dict, basic_indexed,
    min_price, min_mv, max_mv, top_n, is_picking_mode,
):
    """V14.5 原扫描循环的函数化封装；条件、字段、排序与交易规则不变。"""
    records = []
    for ts_code in whitelist_keys:
        if ts_code not in stock_qfq_dict:
            continue
        df_stock = stock_qfq_dict[ts_code]
        if date not in df_stock.index:
            continue

        row_latest = df_stock.loc[date]
        if isinstance(row_latest, pd.DataFrame):
            row_latest = row_latest.iloc[-1]

        curr_close = row_latest["close"]
        if curr_close < min_price:
            continue

        circ_mv_billion = np.nan
        if not basic_indexed.empty and (date, ts_code) in basic_indexed.index:
            circ_mv_billion = basic_indexed.loc[(date, ts_code)]["circ_mv"] / 10000.0

        if pd.notna(circ_mv_billion):
            if circ_mv_billion < min_mv or circ_mv_billion > max_mv:
                continue

        ind = compute_breakout_signal(ts_code, date, stock_qfq_dict)
        if not ind or not ind.get("is_buy_signal"):
            continue

        stock_name = basic_name_map.get(ts_code, ts_code)
        record_dict = {
            "ts_code": ts_code, "name": stock_name, "Signal_Close": ind["signal_close"],
            "SKDJ_K": ind["k"], "SKDJ_D": ind["d"],
            "D_Min(10W)": ind["recent_k_min"], "Weeks_Under": ind["weeks_under_25"],
            "Trend_Type": ind["trend_type"], "vol_ratio": ind["vol_ratio"],
            "circ_mv": round(circ_mv_billion, 2) if pd.notna(circ_mv_billion) else np.nan,
            "Total_Score": ind["Total_Score"],
        }
        if not is_picking_mode:
            future_returns = track_future_performance(
                ts_code, date, ind["signal_close"], stock_qfq_dict, hold_weeks=12
            )
            record_dict.update(future_returns)
        records.append(record_dict)

    if not records:
        return pd.DataFrame(), 0
    selected = (
        pd.DataFrame(records)
        .sort_values("Total_Score", ascending=False)
        .head(int(top_n))
    )
    selected.insert(0, "Rank", range(1, len(selected) + 1))
    selected["Trade_Date"] = date
    return selected, len(records)


def build_run_dates(pro, backtest_days, end_date, is_picking_mode, config_id):
    lookback_days = max(int(backtest_days) * 3, 900)
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_cal = (end_dt - timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_cal_extended = (end_dt + timedelta(days=15)).strftime("%Y%m%d")
    cal_raw = safe_tushare_call(
        pro.trade_cal, exchange="SSE", start_date=start_cal, end_date=end_cal_extended
    )
    if cal_raw.empty:
        raise RuntimeError("无法获取交易日历。")
    cal_open = cal_raw[cal_raw["is_open"] == 1].sort_values("cal_date")
    all_trade_days = cal_open["cal_date"].astype(str).tolist()
    trade_days_list = [date for date in all_trade_days if date <= end_date]
    if not trade_days_list:
        raise RuntimeError("未获取到有效交易日。")
    if is_picking_mode:
        return [trade_days_list[-1]], [trade_days_list[-1]]

    td_df = pd.DataFrame({"cal_date": all_trade_days})
    td_df["dt"] = pd.to_datetime(td_df["cal_date"])
    td_df["year_week"] = td_df["dt"].dt.strftime("%G_%V")
    valid_scan_dates = set(td_df.groupby("year_week")["cal_date"].max().tolist())
    requested = sorted(
        date for date in trade_days_list[-int(backtest_days):]
        if date in valid_scan_dates
    )
    processed = completed_scan_dates(config_id)
    return requested, [date for date in requested if date not in processed]


# ---------------------------
# UI 控制流与输入侧边栏
# ---------------------------
with st.sidebar:
    st.header("⚙️ 模式与分析配置")
    st.info(
        "💡 **双模引擎说明**：\n将追溯天数设为 **1**，触发【盘中极速选股】(不保存)；"
        "设为 **>1**，触发【历史回测】。历史任务每完成一周即保存，可在中断后自动续跑。"
    )

    BACKTEST_DAYS = st.number_input(
        "追溯交易天数 (设为1为极速选股)", value=1, step=30, min_value=1,
        key="v145_backtest_days",
    )
    MAX_TOP_N = st.number_input(
        "每周最多展示股票数 (Top N)", value=5, min_value=1, max_value=50, step=1,
        key="v145_top_n",
    )
    backtest_date_end = st.date_input(
        "分析截止日期", value=datetime.now().date(), key="v145_end_date"
    )

    st.markdown("---")
    clear_market_clicked = st.button("🗑️ 清空行情缓存", key="clear_market")
    clear_history_clicked = st.button("🗑️ 清除历史回测记录", key="clear_history")

    st.markdown("---")
    st.subheader("💰 护城河底座")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0, key="v145_min_price")
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=50.0, key="v145_min_mv")
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0, key="v145_max_mv")

    st.markdown("---")
    try:
        secret_token = st.secrets.get("TUSHARE_TOKEN", "")
    except Exception:
        secret_token = ""
    TS_TOKEN_INPUT = st.text_input(
        "🔑 Tushare Token", value=secret_token, type="password", key="v145_token"
    )

if clear_market_clicked:
    if acquire_run_lock():
        try:
            if os.path.isdir(MARKET_CACHE_ROOT):
                shutil.rmtree(MARKET_CACHE_ROOT)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("底层行情缓存已清理！")
        finally:
            release_run_lock()
    else:
        st.warning("回测正在使用行情缓存，请先停止或等待当前批次结束。")

if clear_history_clicked:
    if acquire_run_lock():
        try:
            for file_path in (CHECKPOINT_FILE, SCAN_LEDGER_FILE, RUN_TASK_FILE):
                remove_with_backup(file_path)
            st.success("历史记录和断点任务已清理！")
        finally:
            release_run_lock()
    else:
        st.warning("回测正在写入结果，请先等待当前批次结束。")

token_clean = clean_token_str(TS_TOKEN_INPUT)
is_picking_mode = int(BACKTEST_DAYS) == 1
current_config_id = make_config_id(MAX_TOP_N, MIN_PRICE, MIN_MV, MAX_MV)
task_before = read_json_safe(RUN_TASK_FILE)

if task_before.get("State") in {"RUNNING", "PAUSED_ERROR"}:
    done_count = int(task_before.get("Completed_Weeks", 0))
    total_count = int(task_before.get("Total_Weeks", 0))
    state_text = "运行中" if task_before.get("State") == "RUNNING" else "已暂停"
    st.info(f"🔄 检测到断点回测：{state_text}，已完成 {done_count}/{total_count} 周。")

retry_clicked = False
stop_clicked = False
if task_before.get("State") == "PAUSED_ERROR":
    retry_clicked = st.button("▶️ 从断点继续", key="resume_backtest")
if task_before.get("State") in {"RUNNING", "PAUSED_ERROR"}:
    stop_clicked = st.button("⏹️ 停止断点回测", key="stop_backtest")

if stop_clicked:
    stopped_task = read_json_safe(RUN_TASK_FILE)
    stopped_task["State"] = "STOPPED"
    save_task(stopped_task)
    st.warning("断点任务已停止；已完成的行情和周结果仍保留。")

if retry_clicked:
    task_before["State"] = "RUNNING"
    task_before["Error_Count"] = 0
    task_before.pop("Last_Error", None)
    save_task(task_before)

btn_label = (
    "🚀 启动盘中极速选股 (天数=1)"
    if is_picking_mode else "🚀 启动历史回测分析 (天数>1)"
)
start_clicked = st.button(btn_label, key="start_v145")

if start_clicked:
    is_valid, msg = verify_token_connection(token_clean)
    if not is_valid:
        st.error(f"❌ **Token 预检拦截**：{msg}")
    else:
        # 新任务只保留一个大型内存索引，避免多次回测把旧索引叠加到内存中。
        st.cache_resource.clear()
    if is_valid and not is_picking_mode:
        new_task = {
            "State": "RUNNING",
            "Config_ID": current_config_id,
            "Params": {
                "Backtest_Days": int(BACKTEST_DAYS),
                "Top_N": int(MAX_TOP_N),
                "End_Date": backtest_date_end.strftime("%Y%m%d"),
                "Min_Price": float(MIN_PRICE),
                "Min_MV": float(MIN_MV),
                "Max_MV": float(MAX_MV),
            },
            "Completed_Weeks": 0,
            "Total_Weeks": 0,
            "Error_Count": 0,
        }
        save_task(new_task)


# ---------------------------
# 主流程：原策略 + 可恢复的工程执行层
# ---------------------------
active_task = read_json_safe(RUN_TASK_FILE)
run_history = active_task.get("State") == "RUNNING" and not stop_clicked
run_picking = start_clicked and is_picking_mode
rerun_needed = False

if run_history or run_picking:
    if not token_clean:
        if run_history:
            active_task["State"] = "PAUSED_ERROR"
            active_task["Last_Error"] = "Token为空；请填写Token后点击从断点继续。"
            save_task(active_task)
        st.error("❌ Token为空，回测断点已保留。")
    elif not acquire_run_lock():
        st.info("另一个页面会话正在执行同一回测，本页面不会重复启动。")
    else:
        try:
            if run_history:
                params = active_task["Params"]
                run_backtest_days = int(params["Backtest_Days"])
                run_top_n = int(params["Top_N"])
                run_end_date = str(params["End_Date"])
                run_min_price = float(params["Min_Price"])
                run_min_mv = float(params["Min_MV"])
                run_max_mv = float(params["Max_MV"])
                run_config_id = str(active_task["Config_ID"])
            else:
                run_backtest_days = 1
                run_top_n = int(MAX_TOP_N)
                run_end_date = backtest_date_end.strftime("%Y%m%d")
                run_min_price = float(MIN_PRICE)
                run_min_mv = float(MIN_MV)
                run_max_mv = float(MAX_MV)
                run_config_id = current_config_id

            ts.set_token(token_clean)
            pro = ts.pro_api(token_clean)
            with st.spinner("正在精准筛选科技池白名单标的..."):
                whitelist_set, basic_name_map = load_custom_tech_whitelist(token_clean)
                whitelist_keys = tuple(sorted(whitelist_set))
            if not whitelist_keys:
                raise RuntimeError("未能获取到科技白名单股票，请检查Token积分或网络。")
            st.info(f"💡 成功锁定科技白名单股票池：共 **{len(whitelist_keys)}** 只标的。")

            requested_dates, pending_dates = build_run_dates(
                pro, run_backtest_days, run_end_date, run_picking, run_config_id
            )
            if run_history:
                active_task["Total_Weeks"] = len(requested_dates)
                active_task["Completed_Weeks"] = len(requested_dates) - len(pending_dates)
                save_task(active_task)

            if not pending_dates:
                if run_history:
                    remove_with_backup(RUN_TASK_FILE)
                    st.success("🎉 指定区间回测数据已全部跑完！")
                else:
                    st.warning("⚠️ 未获取到可扫描日期。")
            else:
                batch_dates = pending_dates if run_picking else pending_dates[:WEEKS_PER_BATCH]
                fetch_start = (
                    datetime.strptime(min(requested_dates), "%Y%m%d") - timedelta(days=300)
                ).strftime("%Y%m%d")
                fetch_end = (
                    datetime.strptime(max(requested_dates), "%Y%m%d") + timedelta(days=200)
                ).strftime("%Y%m%d")
                stock_qfq_dict, basic_indexed = load_optimized_market_data(
                    fetch_start, fetch_end, token_clean, whitelist_keys
                )
                if not stock_qfq_dict:
                    raise RuntimeError("未能加载到完整行情数据；已下载部分仍保留在断点缓存中。")

                bar = st.progress(0, text="执行 V14.5 原版数据扫描...")
                stopped_during_batch = False
                for idx, date in enumerate(batch_dates):
                    if run_history and read_json_safe(RUN_TASK_FILE).get("State") == "STOPPED":
                        stopped_during_batch = True
                        break
                    if not any(date in stock_data.index for stock_data in stock_qfq_dict.values()):
                        raise RuntimeError(
                            f"扫描日 {date} 的完整行情尚未取得；断点已保留，下次只补该日数据。"
                        )
                    selected, raw_count = scan_one_date_original(
                        date, whitelist_keys, basic_name_map, stock_qfq_dict, basic_indexed,
                        run_min_price, run_min_mv, run_max_mv, run_top_n, run_picking,
                    )
                    if run_picking:
                        if not selected.empty:
                            st.subheader(f"🎯 盘中极速选股结果 [{date}] - Top {run_top_n}")
                            try:
                                st.dataframe(
                                    selected.style.background_gradient(
                                        subset=["Total_Score"], cmap="YlOrRd"
                                    ),
                                    width="stretch",
                                )
                            except Exception:
                                st.dataframe(selected, width="stretch")
                        else:
                            st.info(f"[{date}] 未发现符合V14.5原条件的股票。")
                    else:
                        if not selected.empty:
                            selected["Config_ID"] = run_config_id
                            append_checkpoint_atomic(selected)
                        mark_scan_complete(date, raw_count, len(selected), run_config_id)
                        active_task["Completed_Weeks"] = (
                            int(active_task.get("Completed_Weeks", 0)) + 1
                        )
                        active_task["Error_Count"] = 0
                        active_task["Last_Date"] = date
                        save_task(active_task)
                    bar.progress(
                        (idx + 1) / len(batch_dates),
                        text=f"扫描中: {date} (捕获 {raw_count} 只目标)",
                    )
                bar.empty()

                if run_picking:
                    st.success("🎉 今日极速选股完成！(当前为选股模式，数据未写入历史回测库)")
                elif stopped_during_batch:
                    st.warning("断点任务已停止；本批已完成的周结果仍已安全保存。")
                else:
                    remaining = len(pending_dates) - len(batch_dates)
                    if remaining > 0:
                        st.success(f"✅ 本批完成 {len(batch_dates)} 周，剩余 {remaining} 周将自动续跑。")
                        rerun_needed = True
                    else:
                        remove_with_backup(RUN_TASK_FILE)
                        st.success("🎉 回测数据更新完毕！请查看下方报告。")
        except Exception as error:
            if run_history:
                latest_task = read_json_safe(RUN_TASK_FILE) or active_task
                error_count = int(latest_task.get("Error_Count", 0)) + 1
                latest_task["Error_Count"] = error_count
                latest_task["Last_Error"] = str(error)
                if error_count < 3:
                    latest_task["State"] = "RUNNING"
                    rerun_needed = True
                    st.warning(f"⚠️ 临时异常，已保留断点，将自动重试 ({error_count}/3)：{error}")
                else:
                    latest_task["State"] = "PAUSED_ERROR"
                    st.error(f"❌ 连续3次失败，任务已安全暂停：{error}")
                save_task(latest_task)
            else:
                st.error(f"❌ **运行异常拦截**：{error}")
        finally:
            release_run_lock()

if rerun_needed:
    time.sleep(0.8)
    st.rerun()


# ---------------------------
# 全景分析展示区：每次页面重跑均从安全文件恢复
# ---------------------------
raw_res = read_csv_safe(CHECKPOINT_FILE)
if not raw_res.empty:
    st.markdown("---")
    try:
        raw_res["Trade_Date"] = raw_res["Trade_Date"].map(parse_yyyymmdd)
        raw_res = raw_res.dropna(subset=["Trade_Date"])

        # 默认展示当前参数；若页面刷新回默认选股模式，则展示最近一次已保存配置。
        report_config_id = current_config_id
        if "Config_ID" in raw_res.columns:
            matching = raw_res[raw_res["Config_ID"].astype(str) == report_config_id]
            if matching.empty:
                report_config_id = str(raw_res["Config_ID"].dropna().astype(str).iloc[-1])
            raw_res = raw_res[raw_res["Config_ID"].astype(str) == report_config_id].copy()

        repaired_res = repair_checkpoint_df(raw_res)
        if "Exit_Reason" not in repaired_res.columns:
            repaired_res["Exit_Reason"] = "持仓中"
        valid_signals = repaired_res[
            ~repaired_res["Exit_Reason"].astype(str).str.contains("剔除", na=False)
        ].copy()

        st.header("📈 V14.5 历史回测全景分析报告")
        if not valid_signals.empty:
            comp_trades = valid_signals[valid_signals["Exit_Reason"] != "持仓中"].copy()
            total_executed = len(comp_trades)

            if total_executed > 0 and "Final_Return (%)" in comp_trades.columns:
                comp_trades["Final_Return (%)"] = pd.to_numeric(
                    comp_trades["Final_Return (%)"], errors="coerce"
                ).fillna(0)
                win_count = (comp_trades["Final_Return (%)"] > 0).sum()
                global_win_rate = win_count / total_executed * 100.0
                global_mean_ret = comp_trades["Final_Return (%)"].mean()

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("优选截断总笔数", f"{total_executed} 笔")
                col_m2.metric("无干预绝对胜率", f"{global_win_rate:.1f}%", f"{win_count}胜")
                col_m3.metric("全样本平均单笔收益", f"{global_mean_ret:.2f}%")

                st.subheader("🗓️ 周度胜率分布 (强制有序排列 W1 - W12)")
                week_columns = st.columns(4) + st.columns(4) + st.columns(4)
                for week in range(1, 13):
                    column_name = f"Return_W{week} (%)"
                    if column_name not in valid_signals.columns:
                        continue
                    numeric_week = pd.to_numeric(valid_signals[column_name], errors="coerce")
                    valid_week = numeric_week.dropna()
                    with week_columns[week - 1]:
                        if not valid_week.empty:
                            avg = valid_week.mean()
                            win = (valid_week > 0).mean() * 100
                            st.metric(
                                f"W{week} 均益/胜率 (存活{len(valid_week)}只)",
                                f"{avg:.2f}% / {win:.1f}%",
                            )

                required_rank_cols = {
                    "Rank", "Total_Score", "SKDJ_K", "Weeks_Under",
                    "Final_Return (%)", "Exit_Reason",
                }
                if required_rank_cols.issubset(comp_trades.columns):
                    st.markdown("### 🏆 评分层级横向对比验证 (Top N)")
                    rank_stats = comp_trades.groupby("Rank", observed=False).agg(
                        样本数=("Final_Return (%)", "count"),
                        平均分=("Total_Score", "mean"),
                        平均K值=("SKDJ_K", "mean"),
                        均水下周数=("Weeks_Under", "mean"),
                        胜率=("Final_Return (%)", lambda values: (values > 0).mean() * 100),
                        均益=("Final_Return (%)", "mean"),
                        止损率=("Exit_Reason", lambda values: values.str.contains("破-10%").mean() * 100),
                        超级大牛=("Exit_Reason", lambda values: values.str.contains("移动止盈").mean() * 100),
                    ).reset_index().head(10)
                    rank_stats["胜率"] = rank_stats["胜率"].map("{:.1f}%".format)
                    rank_stats["均益"] = rank_stats["均益"].map("{:.2f}%".format)
                    rank_stats["止损率"] = rank_stats["止损率"].map("{:.1f}%".format)
                    rank_stats["超级大牛"] = rank_stats["超级大牛"].map("{:.1f}%".format)
                    rank_stats["均水下周数"] = rank_stats["均水下周数"].map("{:.1f}".format)
                    try:
                        st.dataframe(
                            rank_stats.style.background_gradient(subset=["平均分"], cmap="YlOrRd"),
                            width="stretch",
                        )
                    except Exception:
                        st.dataframe(rank_stats, width="stretch")

            st.subheader("📋 历史回测交割流水单")
            display_columns = [
                "Trade_Date", "name", "ts_code", "Rank", "Total_Score", "SKDJ_K",
                "D_Min(10W)", "Weeks_Under", "Signal_Close", "Buy_Price", "Exit_Date",
                "Hold_Days", "Exit_Reason", "Final_Return (%)",
            ]
            final_display = [column for column in display_columns if column in valid_signals.columns]
            display_frame = valid_signals[final_display].copy()
            sort_columns = [column for column in ("Trade_Date", "Rank") if column in display_frame.columns]
            if sort_columns:
                ascending = [False if column == "Trade_Date" else True for column in sort_columns]
                display_frame = display_frame.sort_values(sort_columns, ascending=ascending)

            def color_exit_reason(value):
                if isinstance(value, str):
                    if "截断" in value: return "color: white; background-color: #8B4513"
                    if "认栽" in value: return "color: white; background-color: darkred"
                    if "保本" in value: return "color: white; background-color: darkgoldenrod"
                    if "移动止盈" in value: return "color: white; background-color: darkgreen"
                    if "期满" in value: return "color: blue"
                return ""

            try:
                styled_port = display_frame.style
                if "Exit_Reason" in display_frame.columns:
                    styled_port = styled_port.map(color_exit_reason, subset=["Exit_Reason"])
                st.dataframe(styled_port, width="stretch")
            except Exception:
                st.dataframe(display_frame, width="stretch")

            export_frame = valid_signals.drop(columns=["Config_ID"], errors="ignore")
            csv_data = export_frame.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 导出回测流水单 (CSV)",
                data=csv_data,
                file_name="skdj_v14_5_history_export.csv",
                mime="text/csv",
                key="download_v145_history",
            )
        else:
            st.info("🕒 未发现符合条件的样本。")
    except Exception as report_error:
        st.error(f"回测结果文件暂时无法展示，但已保存数据不会被删除：{report_error}")
