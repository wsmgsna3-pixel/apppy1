# -*- coding: utf-8 -*-
"""
周线 SKDJ 翻转打分系统 (V14.5 终极实盘版)
------------------------------------------------
1. 【消除警告】：全面适配 Streamlit 最新版接口规范，将 use_container_width 替换为 width='stretch'。
2. 【真·周末锁定】：回测模式下，系统强制向未来探测15天日历，仅允许真正的周末（周五）数据进入历史库。
3. 【双模引擎】：追溯天数设为 1 时，激活“盘中极速选股”模式，秒级出票，不写入数据库，不追踪未来。
4. 【优选截断】：保留 Top N 控制阀，过滤杂波。
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
import gzip
import tempfile
import shutil
import gc
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    fcntl = None

warnings.filterwarnings("ignore")

# ---------------------------
# 全局持久化缓存配置
# ---------------------------
CHECKPOINT_FILE = "skdj_v14_analysis_checkpoint.csv"
MARKET_CACHE_FILE = "skdj_market_data_master.pkl"
MARKET_CACHE_DIR = "skdj_market_data_daily_cache"

# ---------------------------
# 页面基础配置
# ---------------------------
st.set_page_config(page_title="SKDJ V14.5 终极系统", layout="wide")
st.title("🔬 周线 SKDJ 底部脱离系统 (V14.5 终极实盘版)")
st.markdown("🔒 **天数=1为极速选股，>1为历史回测 · 纯净剥离 · 全面兼容新版UI**")

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


def _atomic_replace_bytes(write_callback, target_path):
    target_dir = os.path.dirname(os.path.abspath(target_path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(target_path) + ".", suffix=".tmp", dir=target_dir
    )
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
    if not os.path.exists(target_path):
        return pd.DataFrame()
    for candidate in (target_path, target_path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            return pd.read_csv(candidate, encoding="utf-8-sig", low_memory=False)
        except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
    return pd.DataFrame()


@contextmanager
def _checkpoint_lock():
    """只锁定数秒钟的结果提交，不锁下载、计算或页面会话。"""
    lock_path = CHECKPOINT_FILE + ".lock"
    handle = open(lock_path, "a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _append_checkpoint_safely(new_rows):
    with _checkpoint_lock():
        existing = _read_csv_safely(CHECKPOINT_FILE)
        combined = (
            pd.concat([existing, new_rows], ignore_index=True, sort=False)
            if not existing.empty else new_rows.copy()
        )
        if "Trade_Date" in combined.columns:
            combined["Trade_Date"] = (
                combined["Trade_Date"].astype(str).str.replace(r"\.0$", "", regex=True)
            )
        if {"Trade_Date", "ts_code"}.issubset(combined.columns):
            combined = combined.drop_duplicates(["Trade_Date", "ts_code"], keep="last")
        sort_columns = [col for col in ("Trade_Date", "Rank") if col in combined.columns]
        if sort_columns:
            combined = combined.sort_values(sort_columns, kind="mergesort")
        _atomic_write_csv(combined.reset_index(drop=True), CHECKPOINT_FILE)

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
# 增量下载引擎
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
    daily = payload.get("daily")
    adj = payload.get("adj")
    basic = payload.get("daily_basic")
    if not all(isinstance(frame, pd.DataFrame) for frame in (daily, adj, basic)):
        return False
    required_daily = {"ts_code", "trade_date", "open", "high", "low", "close", "vol"}
    required_adj = {"ts_code", "trade_date", "adj_factor"}
    if daily.empty or adj.empty:
        return False
    if not required_daily.issubset(daily.columns) or not required_adj.issubset(adj.columns):
        return False
    # 防止Tushare网络波动只返回一小部分股票，却被永久当成完整缓存。
    if int(payload.get("daily_count", 0)) < 1000:
        return False
    if int(payload.get("adj_count", 0)) < 1000:
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
        try:
            os.remove(partition_path)
        except OSError:
            pass
        return None
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        try:
            os.remove(partition_path)
        except OSError:
            pass
        return None


def _write_market_partition(payload, trade_date):
    partition_path = _market_partition_path(trade_date)

    def writer(temp_path):
        with gzip.open(temp_path, "wb", compresslevel=3) as file_obj:
            pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
        with open(temp_path, "rb") as file_obj:
            os.fsync(file_obj.fileno())

    _atomic_replace_bytes(writer, partition_path)


def sync_market_data_incrementally(start_date, end_date, token, whitelist_set):
    token_c = clean_token_str(token)
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    
    cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_date, end_date=end_date)
    if cal_raw.empty: return []
        
    cal_open = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date', ascending=True)
    all_dates = cal_open['cal_date'].tolist()
    
    today_str = datetime.now().strftime("%Y%m%d")
    valid_dates = [d for d in all_dates if d <= today_str]
    
    missing_dates = [
        d for d in valid_dates
        if not _market_partition_exists(d)
    ]
    
    if missing_dates:
        my_bar = st.progress(0, text=f"📥 检测到 {len(missing_dates)} 天增量行情需要同步...")
        for i, d in enumerate(missing_dates):
            df_d = safe_tushare_call(pro.daily, max_retries=3, sleep_time=0.8, trade_date=d)
            df_a = safe_tushare_call(pro.adj_factor, max_retries=3, sleep_time=0.8, trade_date=d)
            df_b = safe_tushare_call(pro.daily_basic, max_retries=3, sleep_time=0.8, trade_date=d, fields='ts_code,trade_date,circ_mv')

            if not df_d.empty and not df_a.empty:
                payload = {
                    "version": 1,
                    "trade_date": d,
                    "daily_count": len(df_d),
                    "adj_count": len(df_a),
                    "daily": df_d,
                    "adj": df_a,
                    "daily_basic": df_b if not df_b.empty else pd.DataFrame(),
                }
                if _valid_market_partition(payload, d):
                    _write_market_partition(payload, d)

            if (i + 1) % 5 == 0 or i == len(missing_dates) - 1:
                my_bar.progress((i+1)/len(missing_dates), text=f"📥 行情同步中: {i+1}/{len(missing_dates)}")
            time.sleep(0.25)
        my_bar.empty()
    return valid_dates

# ---------------------------
# 极速轻量化内存索引引擎
# ---------------------------
@st.cache_resource(ttl=3600*12, show_spinner=False)
def _build_market_index(valid_dates, whitelist_keys, cache_stamp):
    del cache_stamp
    whitelist_set = set(whitelist_keys)
    with st.spinner("正在构建全样本前复权索引..."):
        daily_list, adj_list, basic_list = [], [], []
        for trade_date in valid_dates:
            payload = _read_market_partition(trade_date)
            if payload is None:
                continue
            df_d = payload['daily']
            df_a = payload['adj']
            df_b = payload['daily_basic']
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

        daily_raw = pd.concat(daily_list, ignore_index=True) if daily_list else pd.DataFrame()
        adj_raw = pd.concat(adj_list, ignore_index=True) if adj_list else pd.DataFrame()
        basic_raw = pd.concat(basic_list, ignore_index=True) if basic_list else pd.DataFrame()

        if daily_raw.empty or adj_raw.empty: return {}, pd.DataFrame()

        merged_all = daily_raw.merge(adj_raw[['ts_code', 'trade_date', 'adj_factor']], on=['ts_code', 'trade_date'], how='inner')
        merged_all['trade_date_str'] = merged_all['trade_date'].astype(str)
        merged_all = merged_all.drop_duplicates(['ts_code', 'trade_date_str'], keep='last')
        merged_all = merged_all.sort_values(['ts_code', 'trade_date_str'])
        del daily_raw, adj_raw, daily_list, adj_list
        gc.collect()

        stock_qfq_dict = {}
        for ts_code, group in merged_all.groupby('ts_code'):
            df_g = group.copy()
            latest_adj = df_g['adj_factor'].iloc[-1]
            if latest_adj > 0:
                for col in ['open', 'high', 'low', 'close', 'pre_close']:
                    if col in df_g.columns:
                        df_g[col] = df_g[col] * df_g['adj_factor'] / latest_adj
            df_g = df_g.set_index('trade_date_str')
            stock_qfq_dict[ts_code] = df_g
        del merged_all
        gc.collect()
            
        if not basic_raw.empty:
            basic_raw['trade_date'] = basic_raw['trade_date'].astype(str)
            basic_indexed = basic_raw.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['trade_date', 'ts_code'])
        else:
            basic_indexed = pd.DataFrame()

    return stock_qfq_dict, basic_indexed


def load_optimized_market_data(start_date, end_date, token, _whitelist_keys):
    token_c = clean_token_str(token)
    whitelist_set = set(_whitelist_keys)
    valid_dates = sync_market_data_incrementally(start_date, end_date, token_c, whitelist_set)
    if not valid_dates:
        return {}, pd.DataFrame()
    valid_paths = [_market_partition_path(date) for date in valid_dates]
    cache_stamp = (
        sum(os.path.exists(path) for path in valid_paths),
        max((os.path.getmtime(path) for path in valid_paths if os.path.exists(path)), default=0),
    )
    return _build_market_index(tuple(valid_dates), tuple(sorted(whitelist_set)), cache_stamp)

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
# UI 控制流与输入侧边栏
# ---------------------------
with st.sidebar:
    st.header("⚙️ 模式与分析配置")
    
    st.info("💡 **双模引擎说明**：\n将追溯天数设为 **1**，触发【盘中极速选股】(不保存)；设为 **>1**，触发【历史回测】。系统已锁定回测模式仅在周末生效，防止污染。")
    
    BACKTEST_DAYS = st.number_input("追溯交易天数 (设为1为极速选股)", value=1, step=30, min_value=1)
    MAX_TOP_N = st.number_input("每周最多展示股票数 (Top N)", value=5, min_value=1, max_value=50, step=1)
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    
    st.markdown("---")
    if st.button("🗑️ 清空行情缓存"):
        if os.path.isdir(MARKET_CACHE_DIR): shutil.rmtree(MARKET_CACHE_DIR)
        for cache_path in (MARKET_CACHE_FILE, MARKET_CACHE_FILE + ".tmp"):
            if os.path.exists(cache_path): os.remove(cache_path)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("底层行情缓存已清理！")
            
    if st.button("🗑️ 清除历史回测记录"):
        for result_path in (CHECKPOINT_FILE, CHECKPOINT_FILE + ".bak"):
            if os.path.exists(result_path): os.remove(result_path)
        st.success("历史记录已清理！")
            
    st.markdown("---")
    st.subheader("💰 护城河底座")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=50.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)
    
    st.markdown("---")
    secret_token = st.secrets.get("TUSHARE_TOKEN", "") if hasattr(st, "secrets") else ""
    TS_TOKEN_INPUT = st.text_input("🔑 Tushare Token", value=secret_token, type="password")

token_clean = clean_token_str(TS_TOKEN_INPUT)
is_picking_mode = (int(BACKTEST_DAYS) == 1)

# ---------------------------
# 主流程：启动引擎
# ---------------------------
btn_label = "🚀 启动盘中极速选股 (天数=1)" if is_picking_mode else "🚀 启动历史回测分析 (天数>1)"

if st.button(btn_label):
    is_valid, msg = verify_token_connection(token_clean)
    if not is_valid:
        st.error(f"❌ **Token 预检拦截**：{msg}")
    else:
        try:
            # 每次新任务先释放上一轮大型内存索引，避免连续回测叠加占用内存。
            st.cache_resource.clear()
            ts.set_token(token_clean)
            pro = ts.pro_api(token_clean)
            
            with st.spinner("正在精准筛选科技池白名单标的..."):
                whitelist_set, basic_name_map = load_custom_tech_whitelist(token_clean)
                whitelist_keys = tuple(sorted(whitelist_set))
                
            if not whitelist_keys:
                st.error("❌ 未能获取到科技白名单股票，请检查 Token 积分或网络。")
            else:
                st.info(f"💡 成功锁定科技白名单股票池：共 **{len(whitelist_keys)}** 只标的。")
                
                # 🌟 修复关键点：强制系统多看 15 天的日历，确保准确识别未来的周末
                lookback_days = max(int(BACKTEST_DAYS) * 3, 900) 
                start_cal = (datetime.strptime(backtest_date_end.strftime("%Y%m%d"), "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
                end_cal_extended = (datetime.strptime(backtest_date_end.strftime("%Y%m%d"), "%Y%m%d") + timedelta(days=15)).strftime("%Y%m%d")
                
                cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_cal, end_date=end_cal_extended)
                if cal_raw.empty:
                    st.error("❌ 无法获取交易日历。")
                else:
                    cal_open = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date', ascending=True)
                    all_trade_days = cal_open['cal_date'].tolist()
                    
                    # 过滤出小于等于我们所选截止日期的真实可交易日列表
                    end_str = backtest_date_end.strftime("%Y%m%d")
                    trade_days_list = [d for d in all_trade_days if d <= end_str]
                    
                    if not trade_days_list:
                        st.error("❌ 未获取到有效交易日。")
                    else:
                        td_df = pd.DataFrame({'cal_date': all_trade_days})
                        td_df['dt'] = pd.to_datetime(td_df['cal_date'])
                        td_df['year_week'] = td_df['dt'].dt.strftime('%G_%V')
                        
                        # 🌟 核心模式分流与周末防线
                        if is_picking_mode:
                            dates_to_run = [trade_days_list[-1]] 
                        else:
                            # 找出所有包含在完整日历里的“真正的一周最后一天”
                            valid_scan_dates = set(td_df.groupby('year_week')['cal_date'].max().tolist())
                            
                            processed_dates = set()
                            if os.path.exists(CHECKPOINT_FILE):
                                try:
                                    existing_df = _read_csv_safely(CHECKPOINT_FILE)
                                    existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
                                    processed_dates = set(existing_df['Trade_Date'].unique())
                                except Exception: pass
                            recent_trade_days = trade_days_list[-int(BACKTEST_DAYS):]
                            
                            # 只有这一天既在最近回测范围内，又是真正的周末，且没被处理过，才允许进入回测
                            dates_to_run = [d for d in recent_trade_days if d not in processed_dates and d in valid_scan_dates]
                            dates_to_run.sort()
                        
                        if not dates_to_run and not is_picking_mode:
                            st.success("🎉 指定区间回测数据已全部跑完！(如果您选择了周中日期，系统已自动跳过以保护数据纯洁性)")
                        elif dates_to_run:
                            fetch_start = (datetime.strptime(min(dates_to_run), "%Y%m%d") - timedelta(days=300)).strftime("%Y%m%d")
                            fetch_end = (datetime.strptime(max(dates_to_run), "%Y%m%d") + timedelta(days=200)).strftime("%Y%m%d")
                            
                            stock_qfq_dict, basic_indexed = load_optimized_market_data(fetch_start, fetch_end, token_clean, whitelist_keys)
                            
                            if not stock_qfq_dict:
                                st.warning("⚠️ 未能加载到行情数据，请重试。")
                            else:
                                bar = st.progress(0, text="执行 V14.5 数据扫描...")
                                
                                for i, date in enumerate(dates_to_run):
                                    records = []
                                    for ts_code in whitelist_keys:
                                        if ts_code not in stock_qfq_dict: continue
                                        df_stock = stock_qfq_dict[ts_code]
                                        if date not in df_stock.index: continue
                                            
                                        row_latest = df_stock.loc[date]
                                        if isinstance(row_latest, pd.DataFrame): row_latest = row_latest.iloc[-1]
                                            
                                        curr_close = row_latest['close']
                                        if curr_close < MIN_PRICE: continue
                                            
                                        circ_mv_billion = np.nan
                                        if not basic_indexed.empty and (date, ts_code) in basic_indexed.index:
                                            circ_mv_billion = basic_indexed.loc[(date, ts_code)]['circ_mv'] / 10000.0
                                        
                                        if pd.notna(circ_mv_billion):
                                            if circ_mv_billion < MIN_MV or circ_mv_billion > MAX_MV: continue
                                        
                                        ind = compute_breakout_signal(ts_code, date, stock_qfq_dict)
                                        if not ind or not ind.get('is_buy_signal'): continue
                                            
                                        stock_name = basic_name_map.get(ts_code, ts_code)
                                        record_dict = {
                                            'ts_code': ts_code, 'name': stock_name, 'Signal_Close': ind['signal_close'], 
                                            'SKDJ_K': ind['k'], 'SKDJ_D': ind['d'], 
                                            'D_Min(10W)': ind['recent_k_min'], 'Weeks_Under': ind['weeks_under_25'],
                                            'Trend_Type': ind['trend_type'], 'vol_ratio': ind['vol_ratio'],
                                            'circ_mv': round(circ_mv_billion, 2) if pd.notna(circ_mv_billion) else np.nan, 
                                            'Total_Score': ind['Total_Score']
                                        }
                                        
                                        if not is_picking_mode:
                                            future_returns = track_future_performance(ts_code, date, ind['signal_close'], stock_qfq_dict, hold_weeks=12)
                                            record_dict.update(future_returns)
                                            
                                        records.append(record_dict)
                                            
                                    if records:
                                        fdf = pd.DataFrame(records).sort_values('Total_Score', ascending=False).head(int(MAX_TOP_N))
                                        fdf.insert(0, 'Rank', range(1, len(fdf) + 1))
                                        fdf['Trade_Date'] = date
                                        
                                        if is_picking_mode:
                                            st.subheader(f"🎯 盘中极速选股结果 [{date}] - Top {MAX_TOP_N}")
                                            try:
                                                st.dataframe(fdf.style.background_gradient(subset=['Total_Score'], cmap='YlOrRd'), width='stretch')
                                            except Exception:
                                                st.dataframe(fdf, width='stretch')
                                        else:
                                            _append_checkpoint_safely(fdf)
                                        
                                    bar.progress((i+1)/len(dates_to_run), text=f"扫描中: {date} (捕获 {len(records)} 只目标)")
                                    
                                bar.empty()
                                if is_picking_mode:
                                    st.success("🎉 今日极速选股完成！(当前为选股模式，数据未写入历史回测库)")
                                else:
                                    st.success("🎉 回测数据更新完毕！请查看下方报告。")
                                    
        except Exception as e:
            st.error(f"❌ **运行异常拦截**：{str(e)}")

# ---------------------------
# 全景分析展示区 (仅在有历史记录时展示)
# ---------------------------
if os.path.exists(CHECKPOINT_FILE) and not is_picking_mode:
    st.markdown("---")
    try:
        raw_res = _read_csv_safely(CHECKPOINT_FILE)
        if raw_res.empty:
            raise pd.errors.EmptyDataError
        raw_res['Trade_Date'] = raw_res['Trade_Date'].astype(str)
        
        repaired_res = repair_checkpoint_df(raw_res)
        valid_signals = repaired_res[~repaired_res['Exit_Reason'].astype(str).str.contains('剔除', na=False)].copy()
        
        st.header("📈 V14.5 历史回测全景分析报告")
        
        if not valid_signals.empty:
            comp_trades = valid_signals[valid_signals['Exit_Reason'] != '持仓中'].copy()
            total_executed = len(comp_trades)
            
            if total_executed > 0:
                comp_trades['Final_Return (%)'] = pd.to_numeric(comp_trades['Final_Return (%)'], errors='coerce').fillna(0)
                win_count = (comp_trades['Final_Return (%)'] > 0).sum()
                global_win_rate = (win_count / total_executed) * 100.0
                global_mean_ret = comp_trades['Final_Return (%)'].mean()
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("优选截断总笔数", f"{total_executed} 笔")
                col_m2.metric("无干预绝对胜率", f"{global_win_rate:.1f}%", f"{win_count}胜")
                col_m3.metric("全样本平均单笔收益", f"{global_mean_ret:.2f}%")
                
                st.subheader("🗓️ 周度胜率分布 (强制有序排列 W1 - W12)")
                cols_row1 = st.columns(4)
                cols_row2 = st.columns(4)
                cols_row3 = st.columns(4)
                
                for w in range(1, 13):
                    col_name = f'Return_W{w} (%)'
                    if col_name in valid_signals.columns:
                        valid = valid_signals.dropna(subset=[col_name]) 
                        if w <= 4: target_col = cols_row1[w - 1]
                        elif w <= 8: target_col = cols_row2[w - 5]
                        else: target_col = cols_row3[w - 9]
                            
                        with target_col:
                            if not valid.empty:
                                avg = valid[col_name].mean()
                                win = (valid[col_name] > 0).mean() * 100
                                st.metric(f"W{w} 均益/胜率 (存活{len(valid)}只)", f"{avg:.2f}% / {win:.1f}%")
                                
                st.markdown("### 🏆 评分层级横向对比验证 (Top N)")
                rank_stats = comp_trades.groupby('Rank', observed=False).agg(
                    样本数=('Final_Return (%)', 'count'),
                    平均分=('Total_Score', 'mean'),
                    平均K值=('SKDJ_K', 'mean'),
                    均水下周数=('Weeks_Under', 'mean'),
                    胜率=('Final_Return (%)', lambda x: (x > 0).mean() * 100),
                    均益=('Final_Return (%)', 'mean'),
                    止损率=('Exit_Reason', lambda x: x.str.contains('破-10%').mean() * 100),
                    超级大牛=('Exit_Reason', lambda x: x.str.contains('移动止盈').mean() * 100)
                ).reset_index().head(10)
                
                rank_stats['胜率'] = rank_stats['胜率'].map('{:.1f}%'.format)
                rank_stats['均益'] = rank_stats['均益'].map('{:.2f}%'.format)
                rank_stats['止损率'] = rank_stats['止损率'].map('{:.1f}%'.format)
                rank_stats['超级大牛'] = rank_stats['超级大牛'].map('{:.1f}%'.format)
                rank_stats['均水下周数'] = rank_stats['均水下周数'].map('{:.1f}'.format)
                try:
                    st.dataframe(rank_stats.style.background_gradient(subset=['平均分'], cmap='YlOrRd'), width='stretch')
                except Exception:
                    st.dataframe(rank_stats, width='stretch')

            st.subheader("📋 历史回测交割流水单")
            disp_cols = [
                'Trade_Date', 'name', 'ts_code', 'Rank', 'Total_Score', 'SKDJ_K', 'D_Min(10W)', 'Weeks_Under',
                'Signal_Close', 'Buy_Price', 'Exit_Date', 'Hold_Days', 'Exit_Reason', 'Final_Return (%)'
            ]
            final_disp = [c for c in disp_cols if c in valid_signals.columns]
            
            def color_exit_reason(val):
                if isinstance(val, str):
                    if '截断' in val: return 'color: white; background-color: #8B4513'
                    elif '认栽' in val: return 'color: white; background-color: darkred'
                    elif '保本' in val: return 'color: white; background-color: darkgoldenrod'
                    elif '移动止盈' in val: return 'color: white; background-color: darkgreen'
                    elif '期满' in val: return 'color: blue'
                return ''
                
            try:
                styled_port = valid_signals[final_disp].sort_values(['Trade_Date', 'Rank'], ascending=[False, True]).style
                if 'Exit_Reason' in valid_signals.columns:
                    styled_port = styled_port.map(color_exit_reason, subset=['Exit_Reason'])
                st.dataframe(styled_port, width='stretch')
            except Exception:
                fallback_port = valid_signals[final_disp]
                if {'Trade_Date', 'Rank'}.issubset(fallback_port.columns):
                    fallback_port = fallback_port.sort_values(['Trade_Date', 'Rank'], ascending=[False, True])
                st.dataframe(fallback_port, width='stretch')
                
            csv_data = valid_signals.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 导出回测流水单 (CSV)", 
                data=csv_data, 
                file_name="skdj_v14_5_history_export.csv", 
                mime="text/csv",
                on_click="ignore",
                key="download_v14_5_history"
            )
        else:
            st.info("🕒 未发现符合条件的样本。")
    except pd.errors.EmptyDataError:
        st.info("🕒 当前暂无满足条件的回测记录。")
    except Exception as report_error:
        st.warning(f"回测数据已保留，但报告暂时无法显示：{report_error}")
