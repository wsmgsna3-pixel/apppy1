# -*- coding: utf-8 -*-
"""
第一名 1.0 版 (Rank 1 策略专用)
基于 V30.22 内存优化版架构
核心配置：
1. [默认参数] 股价 40-300元，只选 Top 1。
2. [内存保护] 支持 500 天以上长周期回测。
3. [策略逻辑] MACD修正 + 资金共振 + 严格高开突破。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import gc
import time
import os
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量
# ---------------------------
if 'pro' not in st.session_state:
    st.session_state.pro = None

GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="第一名 1.0 版", layout="wide")
st.title("🏆 第一名 1.0 版 (Rank 1 纯享策略)")
st.markdown("""
**策略核心配置：**
* **🎯 选股目标：** 每日评分 **第 1 名** (Rank 1)。
* **💰 价格区间：** **40元 - 300元** (拒绝低价杂毛)。
* **🛡️ 风控机制：** 自动剔除一字板，包含断点续传与内存保护。
""")

# ---------------------------
# 辅助函数 
# ---------------------------
def safe_get(func_name, **kwargs):
    if st.session_state.pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(st.session_state.pro, func_name) 
    try:
        for _ in range(3):
            try:
                if kwargs.get('is_index'): df = st.session_state.pro.index_daily(**kwargs)
                else: df = func(**kwargs)
                if df is not None and not df.empty:
                    return df
            except:
                time.sleep(0.5)
                continue
        return pd.DataFrame(columns=['ts_code'])
    except Exception: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3 + 100)).strftime("%Y%m%d")
    if st.session_state.pro:
        cal = st.session_state.pro.trade_cal(start_date=start_date, end_date=end_date_str)
        if cal.empty or 'is_open' not in cal.columns:
            return []
        return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()
    return []

# ----------------------------------------------------------------------
# 数据加载 (分段版)
# ----------------------------------------------------------------------
def load_data_for_batch(batch_trade_days):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    
    if not batch_trade_days: return False
    
    latest_date = max(batch_trade_days)
    earliest_date = min(batch_trade_days)
    
    data_start = (datetime.strptime(earliest_date, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
    data_end = (datetime.strptime(latest_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d")
    
    msg_slot = st.empty()
    msg_slot.info(f"📥 正在加载数据片段: {data_start} ~ {data_end} ...")
    
    cal = st.session_state.pro.trade_cal(start_date=data_start, end_date=data_end, is_open='1')
    all_dates = cal['cal_date'].tolist()
    
    adj_list, daily_list = [], []
    progress_bar = st.progress(0)
    
    total = len(all_dates)
    for i, date in enumerate(all_dates):
        try:
            df = st.session_state.pro.daily(trade_date=date)
            if not df.empty:
                df = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol']]
                for c in ['open', 'high', 'low', 'close', 'pre_close', 'vol']:
                    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
                daily_list.append(df)
            
            adj = st.session_state.pro.adj_factor(trade_date=date)
            if not adj.empty:
                adj_list.append(adj)
                
            if i % 10 == 0: progress_bar.progress((i + 1) / total)
        except: continue
        
    progress_bar.empty()
    msg_slot.empty()
    
    if not daily_list: return False
    
    GLOBAL_DAILY_RAW = pd.concat(daily_list)
    GLOBAL_DAILY_RAW = GLOBAL_DAILY_RAW.drop_duplicates(subset=['ts_code', 'trade_date'])
    GLOBAL_DAILY_RAW.set_index(['ts_code', 'trade_date'], inplace=True)
    GLOBAL_DAILY_RAW.sort_index(level=[0, 1], inplace=True)
    
    if adj_list:
        GLOBAL_ADJ_FACTOR = pd.concat(adj_list)
        GLOBAL_ADJ_FACTOR['adj_factor'] = pd.to_numeric(GLOBAL_ADJ_FACTOR['adj_factor'], errors='coerce').fillna(0)
        GLOBAL_ADJ_FACTOR = GLOBAL_ADJ_FACTOR.drop_duplicates(subset=['ts_code', 'trade_date'])
        GLOBAL_ADJ_FACTOR.set_index(['ts_code', 'trade_date'], inplace=True)
        GLOBAL_ADJ_FACTOR.sort_index(level=[0, 1], inplace=True)
        
        try:
            latest_adj = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj.droplevel(1).to_dict()
        except: GLOBAL_QFQ_BASE_FACTORS = {}
    
    return True

def clear_memory():
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR
    GLOBAL_DAILY_RAW = pd.DataFrame()
    GLOBAL_ADJ_FACTOR = pd.DataFrame()
    gc.collect()

# ----------------------------------------------------------------------
# 数据处理
# ----------------------------------------------------------------------
def get_qfq_data_batch(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    try:
        daily = GLOBAL_DAILY_RAW.loc[(ts_code, slice(start_date, end_date)), :]
        if daily.empty: return pd.DataFrame()
        
        base_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, 1.0)
        
        try:
            adj = GLOBAL_ADJ_FACTOR.loc[(ts_code, slice(start_date, end_date)), 'adj_factor']
        except:
            adj = pd.Series(index=daily.index, data=base_adj)
            
        df = daily.merge(adj.rename('adj_factor'), left_index=True, right_index=True, how='left')
        df['adj_factor'] = df['adj_factor'].fillna(method='ffill').fillna(base_adj)
        
        factor = df['adj_factor'] / base_adj
        for col in ['open', 'high', 'low', 'close', 'pre_close']:
            df[col] = df[col] * factor
            
        return df.reset_index()
    except Exception:
        return pd.DataFrame()

# ----------------------------------------------------------------------
# 核心指标与买入
# ----------------------------------------------------------------------
def compute_indicators(ts_code, current_date):
    start_date = (datetime.strptime(current_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data_batch(ts_code, start_date, current_date)
    res = {}
    if df.empty or len(df) < 30: return res
    
    df = df.sort_values('trade_date')
    close = df['close']
    vol = df['vol']
    
    # 改进版 MACD (8, 17, 5)
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd_val = (diff - dea) * 2
    
    res['macd_val'] = macd_val.iloc[-1]
    res['close'] = close.iloc[-1]
    res['ma20'] = close.rolling(20).mean().iloc[-1]
    res['vol'] = vol.iloc[-1]
    res['ma5_vol'] = vol.rolling(5).mean().iloc[-1]
    res['pct_chg'] = (close.iloc[-1] / df['pre_close'].iloc[-1] - 1) * 100
    
    return res

def check_buy_and_profit(ts_code, current_date, buy_threshold):
    d0 = datetime.strptime(current_date, "%Y%m%d")
    future_start = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    future_end = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    df = get_qfq_data_batch(ts_code, future_start, future_end)
    if df.empty: return {}
    
    df = df.sort_values('trade_date')
    d1 = df.iloc[0]
    
    res = {}
    if d1['open'] <= d1['pre_close']: return {}
    
    limit_price = d1['pre_close'] * 1.095
    if d1['open'] >= limit_price and d1['low'] >= d1['open']: return {}
    
    buy_price = d1['open'] * (1 + buy_threshold/100)
    if d1['high'] < buy_price: return {}
    
    for n in [1, 3, 5]:
        idx = n - 1
        if len(df) > idx:
            sell_price = df.iloc[idx]['close']
            res[f'Return_D{n} (%)'] = (sell_price / buy_price - 1) * 100
            
    return res

# ---------------------------
# 主逻辑
# ---------------------------
with st.sidebar:
    st.header("⚙️ 参数设置")
    backtest_days = st.number_input("回测天数", value=200, step=50)
    buy_threshold = st.number_input("买入阈值(%)", value=1.5)
    
    st.markdown("---")
    # [修改点] 默认值设为 1
    top_k = st.number_input("每日持仓(Top K)", value=1, min_value=1)
    
    # [修改点] 默认值设为 40 和 300
    min_price = st.number_input("最低股价", value=40.0)
    max_price = st.number_input("最高股价", value=300.0)
    
    ts_token = st.text_input("Tushare Token", type="password")
    
    if st.button("开始回测"):
        if not ts_token:
            st.error("请输入 Token")
            st.stop()
        ts.set_token(ts_token)
        st.session_state.pro = ts.pro_api()
        
        end_date_str = datetime.now().strftime("%Y%m%d")
        all_days = get_trade_days(end_date_str, backtest_days)
        if not all_days:
            st.error("无法获取交易日历")
            st.stop()
            
        all_days = sorted(all_days)
        
        BATCH_SIZE = 60
        results_file = "backtest_result.csv"
        if os.path.exists(results_file): os.remove(results_file)
        
        total_batches = (len(all_days) + BATCH_SIZE - 1) // BATCH_SIZE
        st.success(f"🚀 任务已启动：共 {len(all_days)} 天，将分为 {total_batches} 个批次执行")
        
        global_progress = st.progress(0)
        total_records = 0
        
        for batch_idx in range(total_batches):
            start_i = batch_idx * BATCH_SIZE
            end_i = min((batch_idx + 1) * BATCH_SIZE, len(all_days))
            batch_days = all_days[start_i:end_i]
            
            st.write(f"🔄 **正在执行批次 {batch_idx+1}/{total_batches}**: {batch_days[0]} ~ {batch_days[-1]}")
            
            if not load_data_for_batch(batch_days):
                st.warning(f"批次 {batch_idx+1} 数据加载失败，跳过")
                continue
                
            batch_res = []
            for d_idx, date in enumerate(batch_days):
                try:
                    df_basic = st.session_state.pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,volume_ratio,circ_mv,close')
                    if df_basic is None or df_basic.empty: continue
                    
                    pool = df_basic[
                        (df_basic['close'] >= min_price) & 
                        (df_basic['close'] <= max_price) &
                        (df_basic['circ_mv'] > 200000) & 
                        (df_basic['turnover_rate'] > 3.0)
                    ]
                    
                    candidates = []
                    for row in pool.itertuples():
                        ind = compute_indicators(row.ts_code, date)
                        if not ind: continue
                        
                        if ind['close'] <= ind['ma20']: continue
                        if ind['vol'] <= ind['ma5_vol'] * 1.2: continue
                        if ind['macd_val'] <= 0: continue
                        
                        macd_score = (ind['macd_val'] / ind['close']) * 1000000
                        bonus = 1.0
                        if 1.5 < getattr(row, 'volume_ratio', 0) < 5.0: bonus += 0.1
                        if 5.0 < getattr(row, 'turnover_rate', 0) < 15.0: bonus += 0.1
                        if ind['pct_chg'] > 9.5: bonus += 0.1
                        
                        score = macd_score * bonus
                        
                        candidates.append({
                            'ts_code': row.ts_code,
                            'trade_date': date,
                            'name': 'Unknown', # 节省API调用，后续可补充
                            'score': score,
                            'close': ind['close']
                        })
                    
                    if candidates:
                        day_df = pd.DataFrame(candidates).sort_values('score', ascending=False).head(top_k)
                        
                        final_res = []
                        for rec in day_df.itertuples():
                            ret = check_buy_and_profit(rec.ts_code, rec.trade_date, buy_threshold)
                            rec_dict = rec._asdict()
                            rec_dict.update(ret)
                            final_res.append(rec_dict)
                        
                        batch_res.extend(final_res)
                
                except Exception as e:
                    pass
                
                current_percent = (start_i + d_idx + 1) / len(all_days)
                global_progress.progress(current_percent)
            
            if batch_res:
                df_batch = pd.DataFrame(batch_res)
                df_batch.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False, encoding='utf-8-sig')
                total_records += len(df_batch)
                st.write(f"✅ 批次完成，新增 {len(df_batch)} 条记录 (累计: {total_records})")
            
            clear_memory()
            
        st.success("🎉 所有回测完成！")
        
        if os.path.exists(results_file):
            final_df = pd.read_csv(results_file)
            st.write("### 回测结果概览")
            
            if 'Return_D1 (%)' in final_df.columns:
                valid = final_df.dropna(subset=['Return_D1 (%)'])
                avg = valid['Return_D1 (%)'].mean()
                win = (valid['Return_D1 (%)'] > 0).mean() * 100
                st.metric("D+1 平均收益 / 胜率", f"{avg:.2f}% / {win:.1f}%", f"总交易: {len(valid)}")
            
            st.dataframe(final_df, width=None)
            
            with open(results_file, "rb") as f:
                st.download_button("下载完整回测结果", f, file_name="rank1_v1_0.csv")
