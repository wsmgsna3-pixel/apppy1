# -*- coding: utf-8 -*-
"""
周线 SKDJ 分级补位选股系统 (V15)
------------------------------------------------
在 Gemini V14.5 基础上的改动：
1. 【分级补位】A 级 = 原信号（周K上穿25且K>D）；A 级不足时依次用
   B 级（低位金叉 K≤30）、C 级（强势股回踩后周线再金叉）补满每周名额。
2. 【漏斗诊断】选股时显示股票池逐层剩余数量，0 只时能看到卡在哪一步。
3. 【数据未更新回退】当天日线未发布（Tushare 收盘后才更新）时自动改用最近
   一个有数据的交易日，并明确提示，不再静默返回空结果。
4. 【空周统计】回测记录每一周（包括 0 只的周），按年统计空周数。
5. 【参数隔离】回测记录按参数组分文件保存，改参数后不会和旧结果混在一起。
6. 【回测偏差修正】
   - W1~W12 表改为全样本口径（已出场的单子按出场收益计入），不再只统计幸存者；
   - “保本离场”按次日真实开盘价计算，不再固定记 +2%；
   - 最低股价用未复权价格判断；
   - “持仓中”的单子每次回测时用最新行情重新结算。
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
import json
import hashlib
import pickle
import gzip
import tempfile
import shutil
import gc
import traceback
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    fcntl = None

warnings.filterwarnings("ignore")

VERSION = "V15"
MARKET_CACHE_FILE = "skdj_market_data_master.pkl"
MARKET_CACHE_DIR = "skdj_market_data_daily_cache"

SKDJ_N, SKDJ_M = 6, 3
HOLD_WEEKS = 12

TIER_ORDER = {"A": 0, "B": 1, "C": 2}
TIER_LABEL = {"A": "A 标准上穿25", "B": "B 低位金叉", "C": "C 趋势回踩金叉"}

st.set_page_config(page_title="SKDJ V15 分级补位系统", layout="wide")
st.title("🔬 周线 SKDJ 分级补位选股系统 (V15)")
st.markdown("A 级为原始信号，A 级不足时由 B、C 级补位 · 漏斗诊断 · 空周统计")


# ---------------------------
# Token 与安全请求
# ---------------------------
def clean_token_str(raw_token):
    if not raw_token:
        return ""
    return re.sub(r'[\s\u3000\ufeff\xa0\r\n]+', '', str(raw_token)).strip()


def verify_token_connection(token_str):
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


# ---------------------------
# 原子读写与文件锁
# ---------------------------
def _atomic_replace_bytes(write_callback, target_path):
    target_dir = os.path.dirname(os.path.abspath(target_path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=os.path.basename(target_path) + ".", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        write_callback(temp_path)
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _atomic_write_csv(dataframe, target_path):
    def writer(temp_path):
        dataframe.to_csv(temp_path, index=False, encoding="utf-8-sig")
        with open(temp_path, "rb") as file_obj:
            os.fsync(file_obj.fileno())

    if os.path.exists(target_path):
        try:
            shutil.copy2(target_path, target_path + ".bak")
        except OSError:
            pass
    _atomic_replace_bytes(writer, target_path)


def _read_csv_safely(target_path):
    for candidate in (target_path, target_path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            df = pd.read_csv(candidate, encoding="utf-8-sig", low_memory=False, dtype={"Trade_Date": str, "ts_code": str, "Exit_Date": str})
            for col in ("Trade_Date", "Exit_Date"):
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True).replace({"nan": "", "None": ""})
            return df
        except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
    return pd.DataFrame()


@contextmanager
def _file_lock(path):
    handle = open(path + ".lock", "a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _append_rows_safely(path, new_rows, key_cols, sort_cols):
    with _file_lock(path):
        existing = _read_csv_safely(path)
        combined = pd.concat([existing, new_rows], ignore_index=True, sort=False) if not existing.empty else new_rows.copy()
        combined["Trade_Date"] = combined["Trade_Date"].astype(str).str.replace(r"\.0$", "", regex=True)
        if set(key_cols).issubset(combined.columns):
            combined = combined.drop_duplicates(key_cols, keep="last")
        sort_cols = [c for c in sort_cols if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols, kind="mergesort")
        _atomic_write_csv(combined.reset_index(drop=True), path)


def _rewrite_file_safely(path, dataframe):
    with _file_lock(path):
        _atomic_write_csv(dataframe.reset_index(drop=True), path)


# ---------------------------
# 科技白名单池（保持原逻辑不变）
# ---------------------------
@st.cache_data(ttl=3600 * 24 * 7, show_spinner=False)
def load_custom_tech_whitelist(token):
    token_c = clean_token_str(token)
    if not token_c:
        return set(), {}
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)

    stock_basic = safe_tushare_call(pro.stock_basic, list_status='L', fields='ts_code,symbol,name,industry,market,list_date')
    if stock_basic.empty:
        return set(), {}

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
            m_df = safe_tushare_call(pro.index_member, index_code=s_row['index_code'], is_new='Y')
            if not m_df.empty:
                for c_code in m_df['con_code']:
                    stock_sw_map[c_code] = s_row['industry_name']
            time.sleep(0.03)

    whitelist_set = set()
    name_map = dict(zip(stock_basic['ts_code'], stock_basic['name']))
    for _, row in valid_stocks.iterrows():
        code = row['ts_code']
        ind_basic = str(row['industry']) if pd.notna(row['industry']) else ""
        sw_l1 = stock_sw_map.get(code, "")
        if sw_l1 in CORE_TECH_L1:
            whitelist_set.add(code)
            continue
        if sw_l1 in EXTENDED_TECH_L1:
            if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS) or ind_basic == "" or sw_l1 in {"机械设备", "电力设备", "医药生物"}:
                whitelist_set.add(code)
                continue
        if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS):
            whitelist_set.add(code)
    return whitelist_set, name_map


# ---------------------------
# 行情增量下载与缓存（沿用原缓存目录，已下载的数据可直接复用）
# ---------------------------
def _market_partition_path(trade_date):
    os.makedirs(MARKET_CACHE_DIR, exist_ok=True)
    return os.path.join(MARKET_CACHE_DIR, f"{trade_date}.pkl.gz")


def _market_partition_exists(trade_date):
    try:
        return os.path.getsize(_market_partition_path(trade_date)) >= 100
    except OSError:
        return False


def _valid_market_partition(payload, trade_date):
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != 1 or str(payload.get("trade_date")) != str(trade_date):
        return False
    daily, adj, basic = payload.get("daily"), payload.get("adj"), payload.get("daily_basic")
    if not all(isinstance(frame, pd.DataFrame) for frame in (daily, adj, basic)):
        return False
    if daily.empty or adj.empty:
        return False
    if not {"ts_code", "trade_date", "open", "high", "low", "close", "vol"}.issubset(daily.columns):
        return False
    if not {"ts_code", "trade_date", "adj_factor"}.issubset(adj.columns):
        return False
    if int(payload.get("daily_count", 0)) < 1000 or int(payload.get("adj_count", 0)) < 1000:
        return False
    return True


def _read_market_partition(trade_date):
    partition_path = _market_partition_path(trade_date)
    if not os.path.exists(partition_path):
        return None
    try:
        with gzip.open(partition_path, "rb") as file_obj:
            payload = pickle.load(file_obj)
        if _valid_market_partition(payload, trade_date):
            return payload
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        pass
    try:
        os.remove(partition_path)
    except OSError:
        pass
    return None


def _write_market_partition(payload, trade_date):
    def writer(temp_path):
        with gzip.open(temp_path, "wb", compresslevel=3) as file_obj:
            pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
        with open(temp_path, "rb") as file_obj:
            os.fsync(file_obj.fileno())
    _atomic_replace_bytes(writer, _market_partition_path(trade_date))


def sync_market_data_incrementally(start_date, end_date, token):
    token_c = clean_token_str(token)
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_date, end_date=end_date)
    if cal_raw.empty:
        return []
    all_dates = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date')['cal_date'].astype(str).tolist()
    today_str = datetime.now().strftime("%Y%m%d")
    valid_dates = [d for d in all_dates if d <= today_str]
    missing_dates = [d for d in valid_dates if not _market_partition_exists(d)]

    if missing_dates:
        my_bar = st.progress(0, text=f"📥 需要同步 {len(missing_dates)} 天行情...")
        for i, d in enumerate(missing_dates):
            df_d = safe_tushare_call(pro.daily, trade_date=d)
            df_a = safe_tushare_call(pro.adj_factor, trade_date=d)
            df_b = safe_tushare_call(pro.daily_basic, trade_date=d, fields='ts_code,trade_date,circ_mv')
            if not df_d.empty and not df_a.empty:
                payload = {
                    "version": 1, "trade_date": d,
                    "daily_count": len(df_d), "adj_count": len(df_a),
                    "daily": df_d, "adj": df_a,
                    "daily_basic": df_b if not df_b.empty else pd.DataFrame(),
                }
                if _valid_market_partition(payload, d):
                    _write_market_partition(payload, d)
            if (i + 1) % 5 == 0 or i == len(missing_dates) - 1:
                my_bar.progress((i + 1) / len(missing_dates), text=f"📥 行情同步中: {i + 1}/{len(missing_dates)}")
            time.sleep(0.25)
        my_bar.empty()
    return valid_dates


@st.cache_resource(ttl=3600 * 12, show_spinner=False)
def _build_market_index(valid_dates, whitelist_keys, cache_stamp):
    del cache_stamp
    whitelist_set = set(whitelist_keys)
    with st.spinner("正在构建前复权行情索引..."):
        daily_list, adj_list, basic_list = [], [], []
        for trade_date in valid_dates:
            payload = _read_market_partition(trade_date)
            if payload is None:
                continue
            df_d, df_a, df_b = payload['daily'], payload['adj'], payload['daily_basic']
            if whitelist_set:
                df_d = df_d[df_d['ts_code'].isin(whitelist_set)]
                df_a = df_a[df_a['ts_code'].isin(whitelist_set)]
                if not df_b.empty:
                    df_b = df_b[df_b['ts_code'].isin(whitelist_set)]
            if not df_d.empty and not df_a.empty:
                daily_list.append(df_d)
                adj_list.append(df_a)
                if not df_b.empty:
                    basic_list.append(df_b)

        if not daily_list or not adj_list:
            return {}, pd.DataFrame()
        daily_raw = pd.concat(daily_list, ignore_index=True)
        adj_raw = pd.concat(adj_list, ignore_index=True)
        basic_raw = pd.concat(basic_list, ignore_index=True) if basic_list else pd.DataFrame()

        merged_all = daily_raw.merge(adj_raw[['ts_code', 'trade_date', 'adj_factor']], on=['ts_code', 'trade_date'], how='inner')
        merged_all['trade_date_str'] = merged_all['trade_date'].astype(str)
        merged_all = merged_all.drop_duplicates(['ts_code', 'trade_date_str'], keep='last')
        merged_all = merged_all.sort_values(['ts_code', 'trade_date_str'])
        del daily_raw, adj_raw, daily_list, adj_list
        gc.collect()

        stock_qfq_dict = {}
        for ts_code, group in merged_all.groupby('ts_code'):
            df_g = group.copy()
            df_g['close_raw'] = df_g['close']  # 未复权收盘价，用于最低股价判断
            latest_adj = df_g['adj_factor'].iloc[-1]
            if latest_adj > 0:
                for col in ['open', 'high', 'low', 'close', 'pre_close']:
                    if col in df_g.columns:
                        df_g[col] = df_g[col] * df_g['adj_factor'] / latest_adj
            stock_qfq_dict[ts_code] = df_g.set_index('trade_date_str')
        del merged_all
        gc.collect()

        if not basic_raw.empty:
            basic_raw['trade_date'] = basic_raw['trade_date'].astype(str)
            basic_indexed = basic_raw.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['trade_date', 'ts_code'])
        else:
            basic_indexed = pd.DataFrame()
    return stock_qfq_dict, basic_indexed


def load_optimized_market_data(start_date, end_date, token, whitelist_keys):
    token_c = clean_token_str(token)
    valid_dates = sync_market_data_incrementally(start_date, end_date, token_c)
    if not valid_dates:
        return {}, pd.DataFrame()
    valid_paths = [_market_partition_path(d) for d in valid_dates]
    cache_stamp = (
        sum(os.path.exists(p) for p in valid_paths),
        max((os.path.getmtime(p) for p in valid_paths if os.path.exists(p)), default=0),
    )
    return _build_market_index(tuple(valid_dates), tuple(sorted(whitelist_keys)), cache_stamp)


# ---------------------------
# 周线指标
# ---------------------------
def build_weekly_arrays(df_daily):
    """把日线合成为 ISO 周线并计算 SKDJ（同花顺公式，N=6, M=3）。所有指标只用过去数据。"""
    if df_daily is None or len(df_daily) == 0:
        return None
    date_str = pd.Index(df_daily.index).astype(str)
    dt = pd.to_datetime(date_str, format='%Y%m%d')
    iso = dt.isocalendar()
    frame = pd.DataFrame({
        'yw': iso['year'].to_numpy(dtype='int64') * 100 + iso['week'].to_numpy(dtype='int64'),
        'date': np.asarray(date_str, dtype='int64'),
        'open': df_daily['open'].to_numpy(dtype=float),
        'high': df_daily['high'].to_numpy(dtype=float),
        'low': df_daily['low'].to_numpy(dtype=float),
        'close': df_daily['close'].to_numpy(dtype=float),
        'vol': pd.to_numeric(df_daily['vol'], errors='coerce').to_numpy(dtype=float),
    })
    wk = frame.groupby('yw', sort=True).agg(
        date=('date', 'last'), open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'), close=('close', 'last'), vol=('vol', 'sum'),
    ).reset_index(drop=True)

    lowv = wk['low'].rolling(SKDJ_N).min()
    highv = wk['high'].rolling(SKDJ_N).max()
    diff = (highv - lowv).replace(0, 0.001)
    rsv = ((wk['close'] - lowv) / diff * 100).ewm(span=SKDJ_M, adjust=False).mean()
    k = rsv.ewm(span=SKDJ_M, adjust=False).mean()
    d = k.rolling(SKDJ_M).mean()
    return {
        'date': wk['date'].to_numpy(dtype='int64'),
        'close': wk['close'].to_numpy(dtype=float),
        'vol': wk['vol'].to_numpy(dtype=float),
        'k': k.to_numpy(dtype=float),
        'd': d.to_numpy(dtype=float),
        'ma5_vol': wk['vol'].shift(1).rolling(5).mean().to_numpy(dtype=float),
        'ma20': wk['close'].rolling(20).mean().to_numpy(dtype=float),
    }


def get_weekly_view(ts_code, date, df_stock, weekly_cache, is_week_end):
    """周末日期用整段缓存（指标因果，不含未来）；周中日期截断后现算，避免用到本周后几天。"""
    date_int = int(date)
    if is_week_end:
        wa = weekly_cache.get(ts_code)
        if wa is None:
            wa = build_weekly_arrays(df_stock)
            weekly_cache[ts_code] = wa
        if wa is None:
            return None, -1
        pos = int(np.searchsorted(wa['date'], date_int, side='right')) - 1
        if pos < 0 or wa['date'][pos] != date_int:
            return None, -1
        return wa, pos
    wa = build_weekly_arrays(df_stock[df_stock.index <= date])
    if wa is None:
        return None, -1
    return wa, len(wa['date']) - 1


def score_oversold(close, ma20, recent_k_min, weeks_under, k, vol_ratio):
    """原 V14.5 打分规则，A/B 级沿用。"""
    score = 20.0 if close >= ma20 else -5.0
    if 22.0 <= recent_k_min <= 25.0:
        score += 30.0
    elif 15.0 <= recent_k_min < 22.0:
        score += 15.0
    elif 5.0 <= recent_k_min < 15.0:
        score -= 10.0
    else:
        score -= 25.0
    if 1 <= weeks_under <= 2:
        score += 30.0
    elif 3 <= weeks_under <= 5:
        score += 15.0
    elif 6 <= weeks_under <= 9:
        score -= 5.0
    else:
        score -= 20.0
    if 25.0 < k <= 32.0:
        score += 10.0
    elif k > 38.0:
        score -= 10.0
    if 1.0 <= vol_ratio <= 2.5:
        score += 10.0
    elif vol_ratio > 4.0:
        score -= 15.0
    return score


def score_pullback(ext, slope, k_min6, vol_ratio):
    score = 0.0
    if 0 <= ext <= 0.10:
        score += 20.0
    elif ext <= 0.18:
        score += 10.0
    if slope >= 0.03:
        score += 15.0
    elif slope > 0:
        score += 5.0
    if 35 <= k_min6 <= 50:
        score += 15.0
    elif 25 <= k_min6 <= 60:
        score += 5.0
    if 1.0 <= vol_ratio <= 2.5:
        score += 10.0
    elif vol_ratio > 4.0:
        score -= 15.0
    return score


def evaluate_signal(wa, i):
    """
    A 标准上穿25：上周K≤25，本周K>25，且K>D（原信号）
    B 低位金叉：本周K上穿D，且K≤30（未满足A）
    C 趋势回踩金叉：周收盘在20周均线上方且均线上行，近12周K曾≥70，
      近6周K回落到25~60，本周K重新上穿D，K≤70，收盘偏离均线不超过25%
    """
    if wa is None or i < SKDJ_N + 15:
        return None
    k_arr, d_arr = wa['k'], wa['d']
    k, pk, d, pdv = k_arr[i], k_arr[i - 1], d_arr[i], d_arr[i - 1]
    if not np.all(np.isfinite([k, pk, d, pdv])):
        return None

    close = wa['close'][i]
    ma20 = wa['ma20'][i] if np.isfinite(wa['ma20'][i]) else close
    ma20_prev4 = wa['ma20'][i - 4]
    hist14 = k_arr[i - 14:i]
    hist14 = hist14[np.isfinite(hist14)]
    hist6 = k_arr[i - 6:i]
    hist12 = k_arr[i - 12:i]
    if hist14.size == 0 or not np.isfinite(hist6).any() or not np.isfinite(hist12).any():
        return None
    recent_k_min = float(hist14.min())
    weeks_under = int((hist14 < 25.0).sum())
    k_min6 = float(np.nanmin(hist6))
    k_max12 = float(np.nanmax(hist12))
    mv5 = wa['ma5_vol'][i]
    vol_ratio = float(wa['vol'][i] / mv5) if (np.isfinite(mv5) and mv5 > 0) else 1.0
    ext = close / ma20 - 1 if ma20 > 0 else 0.0
    slope = ma20 / ma20_prev4 - 1 if (np.isfinite(ma20_prev4) and ma20_prev4 > 0) else np.nan
    golden = (pk <= pdv) and (k > d)

    if pk <= 25.0 < k and k > d:
        tier = "A"
        score = score_oversold(close, ma20, recent_k_min, weeks_under, k, vol_ratio)
    elif golden and k <= 30.0:
        tier = "B"
        score = score_oversold(close, ma20, recent_k_min, weeks_under, k, vol_ratio)
    elif (golden and close >= ma20 and np.isfinite(slope) and slope > 0
          and 25.0 <= k_min6 <= 60.0 and k_max12 >= 70.0 and k <= 70.0 and ext <= 0.25):
        tier = "C"
        score = score_pullback(ext, slope, k_min6, vol_ratio)
    else:
        return None

    return {
        'tier': tier, 'score': round(score, 1), 'k': round(float(k), 2), 'd': round(float(d), 2),
        'recent_k_min': round(recent_k_min, 2), 'weeks_under': weeks_under,
        'signal_close': float(close), 'vol_ratio': round(vol_ratio, 2),
        'trend_type': "均线上方" if close >= ma20 else "均线下方(超跌)",
        'ma20_ext_pct': round(ext * 100, 1),
    }


# ---------------------------
# 单日扫描（选股与回测共用）
# ---------------------------
def scan_date(date, whitelist_keys, stock_qfq_dict, basic_indexed, name_map, weekly_cache, is_week_end, cfg):
    funnel = {
        "股票池": len(whitelist_keys), "当日有行情": 0, "股价达标": 0, "市值达标": 0,
        "历史≥100天": 0, "非一字涨停": 0, "A级": 0, "B级": 0, "C级": 0, "市值缺失(未过滤)": 0,
    }
    mv_map = {}
    if basic_indexed is not None and not basic_indexed.empty:
        try:
            mv_map = basic_indexed.xs(date, level='trade_date')['circ_mv'].to_dict()
        except KeyError:
            mv_map = {}

    cands = []
    for ts_code in whitelist_keys:
        df_stock = stock_qfq_dict.get(ts_code)
        if df_stock is None or df_stock.empty:
            continue
        pos = int(df_stock.index.searchsorted(date))
        if pos >= len(df_stock) or df_stock.index[pos] != date:
            continue
        funnel["当日有行情"] += 1

        raw_col = 'close_raw' if 'close_raw' in df_stock.columns else 'close'
        if df_stock[raw_col].iat[pos] < cfg['min_price']:
            continue
        funnel["股价达标"] += 1

        circ_mv = mv_map.get(ts_code, np.nan)
        circ_mv = circ_mv / 10000.0 if pd.notna(circ_mv) else np.nan
        if pd.notna(circ_mv):
            if circ_mv < cfg['min_mv'] or circ_mv > cfg['max_mv']:
                continue
        else:
            funnel["市值缺失(未过滤)"] += 1
        funnel["市值达标"] += 1

        if pos < 99:
            continue
        funnel["历史≥100天"] += 1

        high, low, close_q = df_stock['high'].iat[pos], df_stock['low'].iat[pos], df_stock['close'].iat[pos]
        pre = df_stock['pre_close'].iat[pos] if 'pre_close' in df_stock.columns else np.nan
        if not (pd.notna(pre) and pre > 0):
            pre = df_stock['close'].iat[pos - 1]
        limit_rate = 0.195 if ts_code.startswith(('300', '301', '688', '689')) else 0.095
        if high == low and (close_q - pre) / pre >= limit_rate:
            continue
        funnel["非一字涨停"] += 1

        wa, wpos = get_weekly_view(ts_code, date, df_stock, weekly_cache, is_week_end)
        sig = evaluate_signal(wa, wpos)
        if not sig:
            continue
        funnel[f"{sig['tier']}级"] += 1
        if sig['tier'] not in cfg['tiers']:
            continue
        cands.append({
            'Tier': sig['tier'], 'Tier_Label': TIER_LABEL[sig['tier']],
            'ts_code': ts_code, 'name': name_map.get(ts_code, ts_code),
            'Total_Score': sig['score'], 'SKDJ_K': sig['k'], 'SKDJ_D': sig['d'],
            'K_Min_14W': sig['recent_k_min'], 'Weeks_Under': sig['weeks_under'],
            'Signal_Close': sig['signal_close'], 'Close_Raw': round(float(df_stock[raw_col].iat[pos]), 2),
            'Trend_Type': sig['trend_type'], 'MA20_Ext (%)': sig['ma20_ext_pct'],
            'vol_ratio': sig['vol_ratio'], 'circ_mv': round(circ_mv, 2) if pd.notna(circ_mv) else np.nan,
        })

    if not cands:
        return pd.DataFrame(), pd.DataFrame(), funnel
    all_cands = pd.DataFrame(cands)
    all_cands['_order'] = all_cands['Tier'].map(TIER_ORDER)
    all_cands = all_cands.sort_values(['_order', 'Total_Score'], ascending=[True, False], kind='mergesort').drop(columns='_order').reset_index(drop=True)
    picks = all_cands.head(int(cfg['top_n'])).copy()
    picks.insert(0, 'Rank', range(1, len(picks) + 1))
    picks['Trade_Date'] = date
    return picks, all_cands, funnel


# ---------------------------
# 回测出场模拟
# ---------------------------
def track_future_performance(ts_code, selection_date, signal_close, stock_qfq_dict, hold_weeks=HOLD_WEEKS):
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    results.update({'Exit_Reason': '持仓中', 'Buy_Price': np.nan, 'Gap_pct (%)': np.nan,
                    'Exit_Date': None, 'Final_Return (%)': np.nan, 'Hold_Days': 0})
    if ts_code not in stock_qfq_dict:
        return results
    df_full = stock_qfq_dict[ts_code]
    hist_future = df_full[df_full.index > selection_date]
    if hist_future.empty:
        return results

    next_row = hist_future.iloc[0]
    buy_price = next_row['open']
    if pd.isna(buy_price) or buy_price <= 0 or not signal_close:
        return results

    is_20cm = ts_code.startswith(('300', '301', '688', '689'))
    limit_rate_pct = 19.0 if is_20cm else 9.5
    gap_pct = (buy_price - signal_close) / signal_close * 100.0
    results['Buy_Price'] = round(buy_price, 2)
    results['Gap_pct (%)'] = round(gap_pct, 2)

    if (next_row['open'] == next_row['high'] == next_row['low']) and gap_pct >= limit_rate_pct:
        results['Exit_Reason'] = f"一字板无法买入(剔除: {round(gap_pct, 1)}%)"
        return results
    if is_20cm and gap_pct > 8.0:
        results['Exit_Reason'] = f"双创高开过大(剔除: {round(gap_pct, 2)}%)"
        return results
    if not is_20cm and gap_pct > 5.0:
        results['Exit_Reason'] = f"主板高开过大(剔除: {round(gap_pct, 2)}%)"
        return results
    if gap_pct < -4.0:
        results['Exit_Reason'] = f"恶劣低开(剔除: {round(gap_pct, 2)}%)"
        return results

    tier = 0
    peak_price = buy_price
    pending_exit_reason = None
    hard_stop_limit = -0.10
    max_days = hold_weeks * 5

    def close_out(reason, ret, date, days, week):
        results['Exit_Reason'] = reason
        results['Final_Return (%)'] = round(ret, 2)
        results['Exit_Date'] = date
        results['Hold_Days'] = days
        results[f'Return_W{week} (%)'] = round(ret, 2)

    for i in range(min(len(hist_future), max_days)):
        row = hist_future.iloc[i]
        day_count = i + 1
        current_week = (day_count - 1) // 5 + 1
        curr_open, curr_close, curr_high, curr_low = row['open'], row['close'], row['high'], row['low']
        curr_date = hist_future.index[i]

        if pending_exit_reason is not None and day_count >= 2:
            # 按次日真实开盘价离场（V14.5 对“保本”固定记 +2%，偏乐观）
            close_out(pending_exit_reason, (curr_open - buy_price) / buy_price * 100.0, curr_date, day_count, current_week)
            return results

        peak_price = max(peak_price, curr_high)
        peak_profit_pct = (peak_price - buy_price) / buy_price

        if day_count >= 2 and (curr_low - buy_price) / buy_price <= hard_stop_limit:
            ret = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
            close_out("认栽出局(破-10%)", ret, curr_date, day_count, current_week)
            return results

        if tier == 0 and peak_profit_pct >= 0.10:
            tier = 1
        if tier == 1:
            if curr_close <= buy_price * 1.02:
                pending_exit_reason = "保本离场(+2%触发)"
            elif peak_profit_pct >= 0.20:
                tier = 2
        if tier == 2 and (peak_price - curr_close) / peak_price >= 0.15:
            pending_exit_reason = "移动止盈(回撤15%)"

        if day_count == 5 and pending_exit_reason is None:
            w1_ret = (curr_close - buy_price) / buy_price * 100.0
            if w1_ret <= -3.0:
                close_out(f"首周不及预期截断({round(w1_ret, 1)}%)", w1_ret, curr_date, 5, 1)
                return results

        if day_count % 5 == 0:
            results[f'Return_W{current_week} (%)'] = round((curr_close - buy_price) / buy_price * 100.0, 2)

    if len(hist_future) >= max_days:
        last_price = hist_future.iloc[max_days - 1]['close']
        close_out(f"{hold_weeks}周期满平仓", (last_price - buy_price) / buy_price * 100.0,
                  hist_future.index[max_days - 1], max_days, hold_weeks)
    return results


# ---------------------------
# 侧边栏
# ---------------------------
with st.sidebar:
    st.header("⚙️ 设置")
    MODE = st.radio("运行模式", ["📌 今日选股", "📊 历史回测"], index=0)
    is_picking_mode = MODE.startswith("📌")

    if is_picking_mode:
        backtest_date_end = st.date_input("选股日期（默认今天）", value=datetime.now().date())
        BACKTEST_WEEKS = 0
    else:
        BACKTEST_WEEKS = int(st.number_input("回测周数（52≈1年）", value=52, min_value=4, max_value=520, step=4))
        backtest_date_end = st.date_input("回测截止日期", value=datetime.now().date())

    TOP_N = int(st.number_input("每周选股数量", value=3, min_value=1, max_value=10, step=1))

    st.markdown("---")
    st.subheader("补位层级")
    st.caption("A 级（周K上穿25）始终启用。A 级不足名额时，依次用 B、C 补位。")
    USE_B = st.checkbox("B 低位金叉（K≤30 时 K 上穿 D）", value=True)
    USE_C = st.checkbox("C 趋势回踩金叉（强势股回调后周线再金叉）", value=True)

    st.markdown("---")
    st.subheader("💰 股票池门槛")
    MIN_PRICE = float(st.number_input("最低股价 (元，未复权)", value=10.0))
    col1, col2 = st.columns(2)
    MIN_MV = float(col1.number_input("最小流通市值(亿)", value=50.0))
    MAX_MV = float(col2.number_input("最大流通市值(亿)", value=1000.0))

    st.markdown("---")
    secret_token = ""
    try:
        secret_token = st.secrets.get("TUSHARE_TOKEN", "")
    except Exception:
        secret_token = ""
    TS_TOKEN_INPUT = st.text_input("🔑 Tushare Token", value=secret_token, type="password")

    tiers_enabled = ["A"] + (["B"] if USE_B else []) + (["C"] if USE_C else [])
    CFG = {"v": VERSION, "top_n": TOP_N, "tiers": tiers_enabled,
           "min_price": MIN_PRICE, "min_mv": MIN_MV, "max_mv": MAX_MV}
    CFG_SIG = hashlib.md5(json.dumps(CFG, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    TRADES_FILE = f"skdj_v15_{CFG_SIG}_trades.csv"
    WEEKS_FILE = f"skdj_v15_{CFG_SIG}_weeks.csv"

    with st.expander("🧹 缓存与记录维护"):
        st.caption(f"当前参数组编号：{CFG_SIG}（改任何参数都会自动使用独立的回测记录）")
        if st.button("清除【当前参数组】回测记录"):
            for p in (TRADES_FILE, WEEKS_FILE):
                for suffix in ("", ".bak", ".lock"):
                    if os.path.exists(p + suffix):
                        os.remove(p + suffix)
            st.success("当前参数组的回测记录已清除。")
        if st.button("清空行情缓存（需重新下载）"):
            if os.path.isdir(MARKET_CACHE_DIR):
                shutil.rmtree(MARKET_CACHE_DIR)
            for cache_path in (MARKET_CACHE_FILE, MARKET_CACHE_FILE + ".tmp"):
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("行情缓存已清理。")

token_clean = clean_token_str(TS_TOKEN_INPUT)


# ---------------------------
# 运行流程
# ---------------------------
def load_calendar(pro, end_date_obj, lookback_days):
    end_str = end_date_obj.strftime("%Y%m%d")
    today_str = datetime.now().strftime("%Y%m%d")
    start_cal = (end_date_obj - timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_ext = (end_date_obj + timedelta(days=15)).strftime("%Y%m%d")
    cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_cal, end_date=end_ext)
    if cal_raw.empty:
        return [], set()
    all_days = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date')['cal_date'].astype(str).tolist()
    iso = pd.to_datetime(pd.Index(all_days), format='%Y%m%d').isocalendar()
    td = pd.DataFrame({'d': all_days, 'yw': iso['year'].to_numpy(dtype='int64') * 100 + iso['week'].to_numpy(dtype='int64')})
    week_end_set = set(td.groupby('yw')['d'].max().tolist())
    trade_days = [d for d in all_days if d <= end_str and d <= today_str]
    return trade_days, week_end_set


def show_df(df):
    st.dataframe(df, width='stretch', hide_index=True)


PICK_COLS_CN = {
    'Rank': '排名', 'Tier_Label': '层级', 'name': '名称', 'ts_code': '代码', 'Total_Score': '评分',
    'SKDJ_K': '周K', 'SKDJ_D': '周D', 'K_Min_14W': '前14周K最低', 'Weeks_Under': 'K<25周数',
    'Close_Raw': '收盘价', 'MA20_Ext (%)': '偏离20周线%', 'vol_ratio': '周量比', 'circ_mv': '流通市值(亿)',
    'Trend_Type': '位置',
}


def run_picking(pro, whitelist_keys, name_map):
    trade_days, week_end_set = load_calendar(pro, backtest_date_end, 60)
    if not trade_days:
        st.error("❌ 未获取到交易日历。")
        return
    last_day = trade_days[-1]
    fetch_start = (datetime.strptime(last_day, "%Y%m%d") - timedelta(days=300)).strftime("%Y%m%d")
    stock_qfq_dict, basic_indexed = load_optimized_market_data(fetch_start, last_day, token_clean, whitelist_keys)
    if not stock_qfq_dict:
        st.warning("⚠️ 未能加载到行情数据，请重试。")
        return

    scan_day = next((d for d in reversed(trade_days) if _market_partition_exists(d)), None)
    if scan_day is None:
        st.error("❌ 最近交易日都没有可用行情。")
        return
    if scan_day != last_day:
        st.warning(f"⚠️ {last_day} 的日线 Tushare 尚未发布（通常收盘后 15:30~17:00 更新），本次改用 **{scan_day}** 的数据选股。")
    is_week_end = scan_day in week_end_set
    if not is_week_end:
        st.info(f"ℹ️ {scan_day} 不是本周最后一个交易日，本周K线尚未收完，结果为临时值，周五收盘后可能变化。")

    picks, all_cands, funnel = scan_date(scan_day, whitelist_keys, stock_qfq_dict, basic_indexed,
                                         name_map, {}, is_week_end, CFG)

    st.subheader(f"🎯 选股结果 [{scan_day}]")
    if picks.empty:
        st.error("本次没有任何股票满足已启用的层级。请看下方漏斗，找出卡在哪一步。")
    else:
        cols = [c for c in PICK_COLS_CN if c in picks.columns]
        show_df(picks[cols].rename(columns=PICK_COLS_CN))
        tier_counts = picks['Tier'].value_counts().to_dict()
        if tier_counts.get("A", 0) < len(picks):
            st.caption("本周 A 级不足名额，已用补位层级填充。B/C 级的历史表现请以回测中的“分层级统计”为准。")

    st.markdown("#### 🔍 筛选漏斗")
    show_df(pd.DataFrame([{"步骤": k, "数量": v} for k, v in funnel.items()]))
    if not all_cands.empty and len(all_cands) > len(picks):
        with st.expander(f"查看全部 {len(all_cands)} 只候选"):
            cols = [c for c in PICK_COLS_CN if c in all_cands.columns and c != 'Rank']
            show_df(all_cands[cols].rename(columns=PICK_COLS_CN))
    st.success("选股完成（选股模式不写入回测记录）。")


def run_backtest(pro, whitelist_keys, name_map):
    trade_days, week_end_set = load_calendar(pro, backtest_date_end, BACKTEST_WEEKS * 7 + 60)
    if not trade_days:
        st.error("❌ 未获取到交易日历。")
        return
    trade_day_set = set(trade_days)
    week_ends = sorted(d for d in week_end_set if d in trade_day_set)
    target_dates = week_ends[-BACKTEST_WEEKS:]

    weeks_log = _read_csv_safely(WEEKS_FILE)
    scanned = set(weeks_log['Trade_Date'].astype(str)) if not weeks_log.empty else set()
    dates_to_run = [d for d in target_dates if d not in scanned]

    trades = _read_csv_safely(TRADES_FILE)
    open_dates = []
    if not trades.empty and 'Exit_Reason' in trades.columns:
        open_dates = sorted(trades.loc[trades['Exit_Reason'].astype(str) == '持仓中', 'Trade_Date'].astype(str).unique().tolist())

    if not dates_to_run and not open_dates:
        st.success("🎉 该区间已全部回测完成，且没有待结算的持仓。")
        return

    today_str = datetime.now().strftime("%Y%m%d")
    all_needed = dates_to_run + open_dates
    fetch_start = (datetime.strptime(min(all_needed), "%Y%m%d") - timedelta(days=300)).strftime("%Y%m%d")
    if open_dates:
        fetch_end = today_str
    else:
        fetch_end = min(today_str, (datetime.strptime(max(dates_to_run), "%Y%m%d") + timedelta(days=130)).strftime("%Y%m%d"))

    stock_qfq_dict, basic_indexed = load_optimized_market_data(fetch_start, fetch_end, token_clean, whitelist_keys)
    if not stock_qfq_dict:
        st.warning("⚠️ 未能加载到行情数据，请重试。")
        return

    # 1) 重新结算“持仓中”的单子
    if open_dates:
        trades = _read_csv_safely(TRADES_FILE)
        open_mask = trades['Exit_Reason'].astype(str) == '持仓中'
        rows = trades.loc[open_mask].to_dict('records')
        refreshed = []
        for r in rows:
            code, date = str(r['ts_code']), str(r['Trade_Date'])
            df_s = stock_qfq_dict.get(code)
            if df_s is not None and date in df_s.index:
                sc = float(df_s.loc[date, 'close'])
                r['Signal_Close'] = sc
                r.update(track_future_performance(code, date, sc, stock_qfq_dict))
            refreshed.append(r)
        trades = pd.concat([trades.loc[~open_mask], pd.DataFrame(refreshed)], ignore_index=True, sort=False)
        trades = trades.sort_values(['Trade_Date', 'Rank'], kind='mergesort')
        _rewrite_file_safely(TRADES_FILE, trades)

    # 2) 扫描新的周
    weekly_cache = {}
    skipped = 0
    if dates_to_run:
        bar = st.progress(0, text="回测扫描中...")
        for i, date in enumerate(dates_to_run):
            if not _market_partition_exists(date):
                skipped += 1
                continue
            picks, _, funnel = scan_date(date, whitelist_keys, stock_qfq_dict, basic_indexed,
                                         name_map, weekly_cache, True, CFG)
            if not picks.empty:
                future = [track_future_performance(r.ts_code, date, r.Signal_Close, stock_qfq_dict)
                          for r in picks.itertuples(index=False)]
                picks = pd.concat([picks.reset_index(drop=True), pd.DataFrame(future)], axis=1)
                _append_rows_safely(TRADES_FILE, picks, ["Trade_Date", "ts_code"], ["Trade_Date", "Rank"])
            week_row = pd.DataFrame([{
                'Trade_Date': date, 'A_Count': funnel["A级"], 'B_Count': funnel["B级"], 'C_Count': funnel["C级"],
                'Pool_After_Filter': funnel["非一字涨停"], 'Picks': len(picks),
                'Pick_Tiers': "".join(picks['Tier'].tolist()) if not picks.empty else "",
                'Pick_Names': "、".join(picks['name'].astype(str).tolist()) if not picks.empty else "",
            }])
            _append_rows_safely(WEEKS_FILE, week_row, ["Trade_Date"], ["Trade_Date"])
            bar.progress((i + 1) / len(dates_to_run), text=f"扫描 {date}：A{funnel['A级']} / B{funnel['B级']} / C{funnel['C级']}，入选 {len(picks)} 只")
        bar.empty()
    if skipped:
        st.warning(f"有 {skipped} 个周末交易日缺少行情数据被跳过，下次运行会自动补扫。")
    st.success("🎉 回测更新完毕，请查看下方报告。")


if st.button("🚀 开始选股" if is_picking_mode else "🚀 开始回测", type="primary"):
    is_valid, msg = verify_token_connection(token_clean)
    if not is_valid:
        st.error(f"❌ Token 预检失败：{msg}")
    else:
        try:
            st.cache_resource.clear()
            ts.set_token(token_clean)
            pro_api = ts.pro_api(token_clean)
            with st.spinner("正在加载科技股票池..."):
                wl_set, basic_name_map = load_custom_tech_whitelist(token_clean)
            wl_keys = tuple(sorted(wl_set))
            if not wl_keys:
                st.error("❌ 未能获取股票池，请检查 Token 积分或网络。")
            else:
                st.info(f"💡 股票池共 **{len(wl_keys)}** 只（筛股价、市值前）。")
                if is_picking_mode:
                    run_picking(pro_api, wl_keys, basic_name_map)
                else:
                    run_backtest(pro_api, wl_keys, basic_name_map)
        except Exception as e:
            st.error(f"❌ 运行异常：{e}")
            with st.expander("错误详情"):
                st.code(traceback.format_exc())


# ---------------------------
# 回测报告
# ---------------------------
def weekly_full_sample_table(executed):
    hold_days = pd.to_numeric(executed.get('Hold_Days'), errors='coerce').fillna(0)
    exit_week = np.ceil(hold_days / 5.0)
    is_closed = executed['Exit_Reason'].astype(str) != '持仓中'
    final = pd.to_numeric(executed.get('Final_Return (%)'), errors='coerce')
    rows = []
    for w in range(1, HOLD_WEEKS + 1):
        col_name = f'Return_W{w} (%)'
        if col_name not in executed.columns:
            continue
        held = pd.to_numeric(executed[col_name], errors='coerce')
        locked = final.where(is_closed & (exit_week <= w))
        vals = held.where(held.notna(), locked).dropna()
        if vals.empty:
            continue
        rows.append({'周': f"W{w}", '样本数': len(vals), '平均收益%': round(vals.mean(), 2),
                     '胜率%': round((vals > 0).mean() * 100, 1)})
    return pd.DataFrame(rows)


def group_stats(closed, key):
    g = closed.groupby(key, observed=True)
    out = g.agg(
        笔数=('Final_Return (%)', 'count'),
        胜率=('Final_Return (%)', lambda x: round((x > 0).mean() * 100, 1)),
        平均收益=('Final_Return (%)', lambda x: round(x.mean(), 2)),
        中位收益=('Final_Return (%)', lambda x: round(x.median(), 2)),
        止损占比=('Exit_Reason', lambda x: round(x.astype(str).str.contains('破-10%').mean() * 100, 1)),
        平均持有天数=('Hold_Days', lambda x: round(pd.to_numeric(x, errors='coerce').mean(), 1)),
    ).reset_index()
    return out


if not is_picking_mode and (os.path.exists(TRADES_FILE) or os.path.exists(WEEKS_FILE)):
    st.markdown("---")
    st.header(f"📈 回测报告（参数组 {CFG_SIG}：每周{TOP_N}只，层级 {'+'.join(tiers_enabled)}）")
    try:
        weeks_log = _read_csv_safely(WEEKS_FILE)
        trades = _read_csv_safely(TRADES_FILE)

        if not weeks_log.empty:
            weeks_log['Picks'] = pd.to_numeric(weeks_log['Picks'], errors='coerce').fillna(0).astype(int)
            n_weeks = len(weeks_log)
            n_empty = int((weeks_log['Picks'] == 0).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("已扫描周数", n_weeks)
            c2.metric("空周数（0只）", n_empty)
            c3.metric("折合每年空周", f"{n_empty / n_weeks * 52:.1f}" if n_weeks else "-")

            weeks_log['年份'] = weeks_log['Trade_Date'].astype(str).str[:4]
            weeks_log['全A级'] = weeks_log['Pick_Tiers'].fillna("").astype(str).apply(lambda s: len(s) > 0 and set(s) == {"A"})
            by_year = weeks_log.groupby('年份').agg(
                扫描周数=('Picks', 'size'),
                空周数=('Picks', lambda x: int((x == 0).sum())),
                名额全为A级的周=('全A级', 'sum'),
            ).reset_index()
            st.markdown("#### 🗓️ 按年空周统计")
            show_df(by_year)

        if not trades.empty and 'Exit_Reason' in trades.columns:
            trades['Final_Return (%)'] = pd.to_numeric(trades['Final_Return (%)'], errors='coerce')
            if 'Tier' not in trades.columns:
                trades['Tier'] = 'A'
            excluded = trades[trades['Exit_Reason'].astype(str).str.contains('剔除', na=False)]
            executed = trades[~trades.index.isin(excluded.index)].copy()
            closed = executed[executed['Exit_Reason'].astype(str) != '持仓中'].copy()

            st.markdown("#### 💼 交易总览（已出场）")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("已出场笔数", len(closed))
            if len(closed):
                c2.metric("胜率", f"{(closed['Final_Return (%)'] > 0).mean() * 100:.1f}%")
                c3.metric("平均单笔收益", f"{closed['Final_Return (%)'].mean():.2f}%")
            c4.metric("开盘剔除 / 持仓中", f"{len(excluded)} / {len(executed) - len(closed)}")

            if len(closed):
                st.markdown("#### 🧱 分层级统计（判断补位层级是否拖后腿）")
                tier_tbl = group_stats(closed, 'Tier')
                tier_tbl['Tier'] = tier_tbl['Tier'].map(TIER_LABEL).fillna(tier_tbl['Tier'])
                show_df(tier_tbl.rename(columns={'Tier': '层级'}))

                st.markdown("#### 🏅 按排名统计")
                show_df(group_stats(closed, 'Rank').rename(columns={'Rank': '排名'}))

                st.markdown("#### 🚪 出场原因分布")
                reasons = closed['Exit_Reason'].astype(str).str.replace(r'\(.*\)', '', regex=True)
                show_df(reasons.value_counts().rename_axis('出场原因').reset_index(name='笔数'))

            st.markdown("#### 📅 W1~W12 全样本收益（已出场按出场收益计入）")
            st.caption("V14.5 的周度表只统计仍在持有的股票，首周被截断的亏损单不再计入后面几周，胜率会被高估。本表修正了这一点；“样本数”下降只代表最近的信号还没走到那一周。")
            wtbl = weekly_full_sample_table(executed)
            if not wtbl.empty:
                show_df(wtbl)

        if not weeks_log.empty:
            st.markdown("#### 🗂️ 每周扫描明细")
            wl_disp = weeks_log.sort_values('Trade_Date', ascending=False)[
                ['Trade_Date', 'A_Count', 'B_Count', 'C_Count', 'Picks', 'Pick_Tiers', 'Pick_Names']
            ].rename(columns={'Trade_Date': '周末日期', 'A_Count': 'A级候选', 'B_Count': 'B级候选',
                              'C_Count': 'C级候选', 'Picks': '入选数', 'Pick_Tiers': '入选层级', 'Pick_Names': '入选股票'})
            show_df(wl_disp)

        if not trades.empty:
            st.markdown("#### 📋 交割流水")
            disp_cols = ['Trade_Date', 'Rank', 'Tier_Label', 'name', 'ts_code', 'Total_Score', 'SKDJ_K',
                         'Close_Raw', 'Buy_Price', 'Gap_pct (%)', 'Exit_Date', 'Hold_Days', 'Exit_Reason', 'Final_Return (%)']
            disp_cols = [c for c in disp_cols if c in trades.columns]
            show_df(trades[disp_cols].sort_values(['Trade_Date', 'Rank'], ascending=[False, True]))
            st.download_button(
                label="📥 导出回测流水 (CSV)",
                data=trades.to_csv(index=False).encode('utf-8-sig'),
                file_name=f"skdj_v15_{CFG_SIG}_trades.csv",
                mime="text/csv",
                key="download_v15_trades",
            )
    except Exception as report_error:
        st.warning(f"回测记录已保留，但报告暂时无法显示：{report_error}")
