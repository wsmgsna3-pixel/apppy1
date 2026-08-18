# -*- coding: utf-8 -*-
"""
日线 SKDJ 波段狙击系统 (V16.1 严苛防线版)
------------------------------------------------
1. 【周线多头铁律】：新增要求“当下周线必须保持金叉(K>D)”，彻底剔除周线死叉的“伪回踩”！
2. 【周线强过滤】：强制要求 8 周内探底(K<25) + 最新 K>40，锁定出坑主升浪。
3. 【日线黄金坑】：仅在日线 K 值回落至 [15, 30] 区间时，捕获金叉买点。
4. 【极速快刀流】：持仓最长 4 周，硬止损 -5%。
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

warnings.filterwarnings("ignore")

# ---------------------------
# 全局持久化缓存配置
# ---------------------------
CHECKPOINT_FILE = "skdj_v16_daily_sniper_checkpoint.csv"
MARKET_CACHE_FILE = "skdj_market_data_master.pkl"

# ---------------------------
# 页面基础配置
# ---------------------------
st.set_page_config(page_title="SKDJ 日线快刀狙击系统", layout="wide")
st.title("🎯 日线 SKDJ 波段狙击系统 (V16.1 严苛防线版)")
st.markdown("⚡ **新增周线不死叉铁律 · 过滤大级别破位飞刀 · 4 周极速出局**")

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

# ---------------------------
# 科技白名单池构建 (50亿市值底座 + 10元股价)
# ---------------------------
@st.cache_data(ttl=3600*24*7, show_spinner=False)
def load_custom_tech_whitelist(token):
    token_c = clean_token_str(token)
    if not token_c: return set(), {}
    
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
        
        if sw_l1 in CORE_TECH_L1:
            whitelist_set.add(code)
            continue
            
        if sw_l1 in EXTENDED_TECH_L1:
            if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS) or ind_basic == "" or sw_l1 in {"机械设备", "电力设备", "医药生物"}:
                whitelist_set.add(code)
                continue
                
        if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS):
            whitelist_set.add(code)
            continue

    return whitelist_set, name_map

# ---------------------------
# 增量下载引擎
# ---------------------------
def sync_market_data_incrementally(start_date, end_date, token, whitelist_set):
    token_c = clean_token_str(token)
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    
    cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_date, end_date=end_date)
    if cal_raw.empty:
        return {'daily': [], 'adj': [], 'daily_basic': [], 'fetched_dates': set()}
        
    cal_open = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date', ascending=True)
    all_dates = cal_open['cal_date'].tolist()
    
    today_str = datetime.now().strftime("%Y%m%d")
    valid_dates = [d for d in all_dates if d <= today_str]
    
    cache = {'daily': [], 'adj': [], 'daily_basic': [], 'fetched_dates': set()}
    if os.path.exists(MARKET_CACHE_FILE):
        try:
            with open(MARKET_CACHE_FILE, 'rb') as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    cache.update(loaded)
                    if 'daily_basic' not in cache: cache['daily_basic'] = []
        except Exception:
            pass 
            
    missing_dates = [d for d in valid_dates if d not in cache['fetched_dates']]
    
    if missing_dates:
        my_bar = st.progress(0, text=f"📥 检测到 {len(missing_dates)} 天增量行情需要同步...")
        
        for i, d in enumerate(missing_dates):
            df_d = safe_tushare_call(pro.daily, max_retries=3, sleep_time=0.8, trade_date=d)
            df_a = safe_tushare_call(pro.adj_factor, max_retries=3, sleep_time=0.8, trade_date=d)
            df_b = safe_tushare_call(pro.daily_basic, max_retries=3, sleep_time=0.8, trade_date=d, fields='ts_code,trade_date,circ_mv')
            
            if whitelist_set:
                if not df_d.empty: df_d = df_d[df_d['ts_code'].isin(whitelist_set)]
                if not df_a.empty: df_a = df_a[df_a['ts_code'].isin(whitelist_set)]
                if not df_b.empty: df_b = df_b[df_b['ts_code'].isin(whitelist_set)]
                
            if not df_d.empty and not df_a.empty:
                cache['daily'].append(df_d)
                cache['adj'].append(df_a)
                if not df_b.empty:
                    cache['daily_basic'].append(df_b)
                cache['fetched_dates'].add(d)
            
            if (i + 1) % 10 == 0 or i == len(missing_dates) - 1:
                my_bar.progress((i+1)/len(missing_dates), text=f"📥 行情同步中: {i+1}/{len(missing_dates)}")
                try:
                    with open(MARKET_CACHE_FILE + ".tmp", 'wb') as f:
                        pickle.dump(cache, f)
                    os.replace(MARKET_CACHE_FILE + ".tmp", MARKET_CACHE_FILE)
                except Exception:
                    pass
            
            time.sleep(0.25)
            
        my_bar.empty()
        
    return cache

# ---------------------------
# 极速轻量化内存索引引擎
# ---------------------------
@st.cache_data(ttl=3600*12, show_spinner=False)
def load_optimized_market_data(start_date, end_date, token, _whitelist_keys, _dummy_trigger):
    token_c = clean_token_str(token)
    whitelist_set = set(_whitelist_keys)
    cache = sync_market_data_incrementally(start_date, end_date, token_c, whitelist_set)
    
    with st.spinner("正在构建全样本前复权索引..."):
        daily_list = cache.get('daily', [])
        adj_list = cache.get('adj', [])
        basic_list = cache.get('daily_basic', [])
        
        daily_raw = pd.concat(daily_list, ignore_index=True) if daily_list else pd.DataFrame()
        adj_raw = pd.concat(adj_list, ignore_index=True) if adj_list else pd.DataFrame()
        basic_raw = pd.concat(basic_list, ignore_index=True) if basic_list else pd.DataFrame()
        
        if daily_raw.empty or adj_raw.empty:
            return {}, pd.DataFrame()
            
        if whitelist_set:
            daily_raw = daily_raw[daily_raw['ts_code'].isin(whitelist_set)]
            adj_raw = adj_raw[adj_raw['ts_code'].isin(whitelist_set)]
            if not basic_raw.empty:
                basic_raw = basic_raw[basic_raw['ts_code'].isin(whitelist_set)]

        merged_all = daily_raw.merge(adj_raw[['ts_code', 'trade_date', 'adj_factor']], on=['ts_code', 'trade_date'], how='inner')
        merged_all['trade_date_str'] = merged_all['trade_date'].astype(str)
        merged_all = merged_all.sort_values(['ts_code', 'trade_date_str'])
        
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
            
        if not basic_raw.empty:
            basic_indexed = basic_raw.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['trade_date', 'ts_code'])
        else:
            basic_indexed = pd.DataFrame()
            
    return stock_qfq_dict, basic_indexed

# ---------------------------
# 🚀【V16.1 核心引擎】：修复周线死叉陷阱
# ---------------------------
def compute_daily_sniper_signal(ts_code, end_date, stock_qfq_dict):
    if ts_code not in stock_qfq_dict: return {}
    df_full = stock_qfq_dict[ts_code]
    
    df_daily = df_full[df_full.index <= end_date].copy()
    res = {}
    if df_daily.empty or len(df_daily) < 100: return res
    
    # 【日线一字板涨停过滤】
    row_today = df_daily.iloc[-1]
    is_20cm = any(ts_code.startswith(prefix) for prefix in ['300', '301', '688', '689'])
    limit_rate = 0.195 if is_20cm else 0.095
    pre_close_val = row_today.get('pre_close', np.nan)
    if pd.isna(pre_close_val) or pre_close_val <= 0:
        pre_close_val = df_daily.iloc[-2]['close'] if len(df_daily) >= 2 else row_today['open']
            
    is_yiziban = (row_today['high'] == row_today['low']) and ((row_today['close'] - pre_close_val) / pre_close_val >= limit_rate)
    if is_yiziban:
        return res

    df = df_daily.copy()
    
    # -------------------------
    # 🎯 第一步：计算日线 SKDJ
    # -------------------------
    lowv_d = df['low'].rolling(window=9).min()
    highv_d = df['high'].rolling(window=9).max()
    diff_d = (highv_d - lowv_d).replace(0, 0.001)
    rsv_d = (df['close'] - lowv_d) / diff_d * 100
    df['day_k'] = rsv_d.ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()
    df['day_d'] = df['day_k'].rolling(window=3).mean()
    
    curr_d = df.iloc[-1]
    prev_d = df.iloc[-2]
    
    if pd.isna(curr_d['day_k']) or pd.isna(prev_d['day_k']): return res

    # 仅捕获 15-30 的日线金叉
    is_daily_cross = (curr_d['day_k'] > curr_d['day_d']) and (prev_d['day_k'] <= prev_d['day_d'])
    is_in_sweet_spot = (15.0 <= curr_d['day_k'] <= 30.0)
    
    if not (is_daily_cross and is_in_sweet_spot):
        return res

    # -------------------------
    # 🛡️ 第二步：周线降维过滤 (修复死叉破位漏洞)
    # -------------------------
    df['dt'] = pd.to_datetime(df.index)
    df['year_week'] = df['dt'].dt.strftime('%G_%V') 
    
    weekly_df = df.groupby('year_week', as_index=False).agg({
        'dt': 'last', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).sort_values('dt').reset_index(drop=True)
    
    if len(weekly_df) < 15: return res

    # 🌟 修复：不仅计算周线 K，同时计算周线 D 以判断多空死叉
    lowv_w = weekly_df['low'].rolling(window=9).min()
    highv_w = weekly_df['high'].rolling(window=9).max()
    diff_w = (highv_w - lowv_w).replace(0, 0.001)
    rsv_w = (weekly_df['close'] - lowv_w) / diff_w * 100
    weekly_df['week_k'] = rsv_w.ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()
    weekly_df['week_d'] = weekly_df['week_k'].rolling(window=3).mean()
    
    curr_week_str = df.iloc[-1]['year_week']
    prev_weeks = weekly_df[weekly_df['year_week'] < curr_week_str].copy()
    
    if len(prev_weeks) < 8: return res
    
    last_week_k = prev_weeks.iloc[-1]['week_k']
    
    # 过滤条件 1：上周五收盘周线 K > 40
    if pd.isna(last_week_k) or last_week_k <= 40.0:
        return res

    # 🌟🌟🌟 核心防线修复：当下的周线绝对不能死叉！
    curr_week_row = weekly_df.iloc[-1]
    if pd.isna(curr_week_row['week_k']) or pd.isna(curr_week_row['week_d']):
        return res
    if curr_week_row['week_k'] <= curr_week_row['week_d']:
        # 一旦周线死叉，说明这几天的日线下跌已经把大趋势拉崩了，坚决放弃这把飞刀！
        return res
        
    # 过滤条件 3：过去 8 周内必须有周线 K < 25 (确保是在做主升浪启动段)
    recent_8_weeks_k = prev_weeks['week_k'].tail(8)
    if not (recent_8_weeks_k < 25.0).any():
        return res

    res['is_buy_signal'] = True
    res['Daily_K'] = round(curr_d['day_k'], 2)
    res['Prev_Week_K'] = round(last_week_k, 2)
    res['Current_Week_K'] = round(curr_week_row['week_k'], 2) # 新增快照展示
    res['signal_close'] = curr_d['close']
    
    return res
# ---------------------------
# 🚀 极速出局系统：全域一字板拦截 + 4周强制平仓 + -5%铁血止损 + 移动止盈
# ---------------------------
def track_future_performance(ts_code, selection_date, signal_close, stock_qfq_dict, hold_weeks=4):
    default_res = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    default_res.update({
        'Exit_Reason': '持仓中', 'Buy_Price': np.nan, 'Gap_pct (%)': np.nan, 
        'Exit_Date': None, 'Final_Return (%)': np.nan, 'Hold_Days': 0
    })
    
    if ts_code not in stock_qfq_dict: 
        return default_res

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
    results['Gap_pct (%)'] = round(gap_pct, 2)

    # 🚀【T+1 全域一字板无法买入拦截】
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
    
    # 铁血止损：-5%
    hard_stop_limit = -0.05 
    max_days = hold_weeks * 5  
    
    for i in range(len(hist_future)):
        if i >= max_days: break 
            
        row = hist_future.iloc[i]
        day_count = i + 1
        current_week = ((day_count - 1) // 5) + 1 
        curr_open, curr_close, curr_high, curr_low = row['open'], row['close'], row['high'], row['low']
        curr_date = hist_future.index[i]
        
        # 1. 挂单止盈/保本执行 (T+1 开盘)
        if pending_exit_reason is not None and day_count >= 2:
            if "保本" in pending_exit_reason:
                final_return = 1.0  
            else:
                final_return = (curr_open - buy_price) / buy_price * 100.0
                
            exit_triggered = True
            results['Exit_Reason'] = pending_exit_reason
            results['Final_Return (%)'] = round(final_return, 2)
            results['Exit_Date'] = curr_date
            results['Hold_Days'] = day_count
            results[f'Return_W{current_week} (%)'] = round(final_return, 2)
            break
        
        peak_price = max(peak_price, curr_high)
        peak_profit_pct = (peak_price - buy_price) / buy_price
        
        # 2. T+1 日线硬止损 (-5%) 铁血执行
        if day_count >= 2:
            if (curr_low - buy_price) / buy_price <= hard_stop_limit:
                final_return = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
                exit_triggered = True
                results['Exit_Reason'] = "铁血止损(破-5%)"
                results['Final_Return (%)'] = round(final_return, 2)
                results['Exit_Date'] = curr_date
                results['Hold_Days'] = day_count
                results[f'Return_W{current_week} (%)'] = round(final_return, 2)
                break
        
        # 3. 动态阶梯状态机：波段战法
        if tier == 0 and peak_profit_pct >= 0.08:  
            tier = 1  
            
        if tier == 1:
            if curr_close <= buy_price * 1.01:  
                pending_exit_reason = "保本离场(+1%)"
            elif peak_profit_pct >= 0.15:  
                tier = 2  
                
        if tier == 2:
            giveback = (peak_price - curr_close) / peak_price
            if giveback >= 0.12:  
                pending_exit_reason = "移动止盈(回撤12%)"
            
        if day_count % 5 == 0:
            results[f'Return_W{current_week} (%)'] = round((curr_close - buy_price) / buy_price * 100.0, 2)
            
    # 4. 4周期满强制平仓（换股）
    if not exit_triggered and len(hist_future) >= max_days:
        last_price = hist_future.iloc[max_days - 1]['close']
        final_return = (last_price - buy_price) / buy_price * 100.0
        results[f'Return_W{hold_weeks} (%)'] = round(final_return, 2)
        results['Exit_Reason'] = "4周期满平仓(换股)"
        results['Final_Return (%)'] = round(final_return, 2)
        results['Exit_Date'] = hist_future.index[max_days - 1]
        results['Hold_Days'] = max_days
        
    return results

# ---------------------------
# 历史数据自动兼容与修复引擎
# ---------------------------
def repair_checkpoint_df(df_in):
    df_out = df_in.copy()
    w_cols = [c for c in df_out.columns if c.startswith('Return_W') and c.endswith('(%)')]
    if w_cols:
        w_cols = sorted(w_cols, key=lambda x: int(x.replace('Return_W', '').replace(' (%)', '')))
    
    if 'Final_Return (%)' not in df_out.columns:
        def get_final_ret(r):
            if not w_cols: return 0.0
            rets = r[w_cols].dropna()
            return rets.iloc[-1] if not rets.empty else 0.0
        df_out['Final_Return (%)'] = df_out.apply(get_final_ret, axis=1)
        
    if 'Exit_Date' not in df_out.columns:
        df_out['Exit_Date'] = None
        
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
    st.header("⚙️ 日线狙击测试配置")
    st.info("💡 终极防线版：新增周线不死叉铁律，斩断大级别破位飞刀。")
    
    st.markdown("---")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("追溯交易天数", value=250, step=30)
    
    st.markdown("---")
    if st.button("🗑️ 清空行情缓存"):
        if os.path.exists(MARKET_CACHE_FILE):
            os.remove(MARKET_CACHE_FILE)
        st.cache_data.clear()
        st.success("底层行情缓存已清理！")
            
    if st.button("🗑️ 清除日线狙击记录 (重跑前必点)"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        st.success("狙击记录已清理！")
            
    st.markdown("---")
    st.subheader("💰 护城河底座")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=50.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)
    
    st.markdown("---")
    secret_token = st.secrets.get("TUSHARE_TOKEN", "") if hasattr(st, "secrets") else ""
    TS_TOKEN_INPUT = st.text_input(
        "🔑 Tushare Token", 
        value=secret_token,
        type="password"
    )

token_clean = clean_token_str(TS_TOKEN_INPUT)

# ---------------------------
# 全景分析展示区
# ---------------------------
if os.path.exists(CHECKPOINT_FILE):
    st.markdown("---")
    try:
        raw_res = pd.read_csv(CHECKPOINT_FILE)
        raw_res['Trade_Date'] = raw_res['Trade_Date'].astype(str)
        
        repaired_res = repair_checkpoint_df(raw_res)
        valid_signals = repaired_res[~repaired_res['Exit_Reason'].astype(str).str.contains('剔除', na=False)].copy()
        
        st.header("📈 V16.1 严苛防线快刀流实战报告")
        
        if not valid_signals.empty:
            comp_trades = valid_signals[valid_signals['Exit_Reason'] != '持仓中'].copy()
            total_executed = len(comp_trades)
            
            if total_executed > 0:
                comp_trades['Final_Return (%)'] = pd.to_numeric(comp_trades['Final_Return (%)'], errors='coerce').fillna(0)
                win_count = (comp_trades['Final_Return (%)'] > 0).sum()
                global_win_rate = (win_count / total_executed) * 100.0
                global_mean_ret = comp_trades['Final_Return (%)'].mean()
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("高纯度波段捕获总数", f"{total_executed} 笔")
                col_m2.metric("4周内实战绝对胜率", f"{global_win_rate:.1f}%", f"{win_count}胜")
                col_m3.metric("波段单笔平均净收益", f"{global_mean_ret:.2f}%")
                
                st.subheader("🗓️ 极速波段周度存活状态 (最长 4 周)")
                cols_row1 = st.columns(4)
                
                for w in range(1, 5):
                    col_name = f'Return_W{w} (%)'
                    if col_name in valid_signals.columns:
                        valid = valid_signals.dropna(subset=[col_name]) 
                        with cols_row1[w - 1]:
                            if not valid.empty:
                                avg = valid[col_name].mean()
                                win = (valid[col_name] > 0).mean() * 100
                                st.metric(f"W{w} 均益/胜率 (存活{len(valid)}只)", f"{avg:.2f}% / {win:.1f}%")
                                
                st.markdown("### 🔍 日线狙击出局原因拆解")
                exit_stats = comp_trades['Exit_Reason'].value_counts().reset_index()
                exit_stats.columns = ['出局原因', '触发次数']
                exit_stats['占比'] = (exit_stats['触发次数'] / total_executed * 100).map('{:.1f}%'.format)
                st.dataframe(exit_stats.style.background_gradient(subset=['触发次数'], cmap='Purples'), use_container_width=True)

            st.subheader("📋 日线波段交割流水单")
            
            disp_cols = [
                'Trade_Date', 'name', 'ts_code', 'Current_Week_K', 'Prev_Week_K', 'Daily_K', 
                'Buy_Price', 'Exit_Date', 'Hold_Days', 'Exit_Reason', 'Final_Return (%)'
            ]
            final_disp = [c for c in disp_cols if c in valid_signals.columns]
            
            def color_exit_reason(val):
                if isinstance(val, str):
                    if '铁血' in val: return 'color: white; background-color: darkred'
                    elif '保本' in val: return 'color: white; background-color: darkgoldenrod'
                    elif '移动止盈' in val: return 'color: white; background-color: darkgreen'
                    elif '平仓' in val: return 'color: white; background-color: blue'
                return ''
                
            styled_port = valid_signals[final_disp].sort_values(['Trade_Date'], ascending=[False]).style
            if 'Exit_Reason' in valid_signals.columns:
                styled_port = styled_port.map(color_exit_reason, subset=['Exit_Reason'])
                
            try:
                st.dataframe(styled_port, width="stretch")
            except Exception:
                st.dataframe(styled_port, use_container_width=True)
                
            csv_data = valid_signals.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 导出日线防线版流水 (CSV)", 
                data=csv_data, 
                file_name="skdj_v16_1_sniper_strict.csv", 
                mime="text/csv"
            )
        else:
            st.info("🕒 未发现符合条件的样本。")
    except pd.errors.EmptyDataError:
        st.info("🕒 当前暂无满足条件的回测记录。")
