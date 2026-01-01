# -*- coding: utf-8 -*-
"""
第一名 2.0 版 (UI重构 + 真实断点续传)
核心升级：
1. [UI] Token和开始按钮移至主界面，参数收纳进折叠栏。
2. [续传] 启动前自动扫描已完成日期，崩溃后重启可无缝继续。
3. [稳健] 强化内存管理，防止500天回测崩溃。
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
# 全局变量与Session管理
# ---------------------------
if 'pro' not in st.session_state:
    st.session_state.pro = None
if 'ts_token' not in st.session_state:
    st.session_state.ts_token = ""

GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="第一名 2.0 版", layout="wide")

# ---------------------------
# UI 布局 (移出侧边栏)
# ---------------------------
st.title("🏆 第一名 2.0 版 (Rank 1 纯享 + 智能续传)")

# 主控制区
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        # Token 输入框 (带记忆)
        new_token = st.text_input("💎 请输入 Tushare Token (10000积分)", 
                                  value=st.session_state.ts_token, 
                                  type="password",
                                  help="Token 将保存在本次会话中")
        if new_token:
            st.session_state.ts_token = new_token
            ts.set_token(new_token)
            st.session_state.pro = ts.pro_api()

    with col2:
        # 显眼的开始按钮
        st.write("") # 占位对齐
        st.write("") 
        start_btn = st.button("🚀 启动/继续 回测", type="primary", use_container_width=True)

# 参数折叠栏 (默认隐藏，点击展开)
with st.expander("⚙️ 策略参数设置 (已优化默认值，无需频繁调整)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        backtest_days = st.number_input("回测天数 (N)", value=500, step=50, help="建议设置为500天以验证穿越牛熊的能力")
        buy_threshold = st.number_input("买入阈值 (%)", value=1.5, step=0.1)
    with c2:
        min_price = st.number_input("最低股价", value=40.0)
        max_price = st.number_input("最高股价", value=300.0)
    with c3:
        top_k = st.number_input("每日持仓 (Top K)", value=1, disabled=True, help="本策略核心就是只做第一名")
        
# ---------------------------
# 辅助函数 
# ---------------------------
def get_trade_days(end_date_str, num_days):
    # 增加冗余天数以确保覆盖
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3 + 200)).strftime("%Y%m%d")
    if st.session_state.pro:
        try:
            cal = st.session_state.pro.trade_cal(start_date=start_date, end_date=end_date_str)
            if cal.empty or 'is_open' not in cal.columns: return []
            return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()
        except: return []
    return []

def load_processed_dates(filepath):
    """读取已完成的日期，实现断点续传"""
    if not os.path.exists(filepath):
        return set()
    try:
        # 只读取 trade_date 列，减少内存消耗
        df = pd.read_csv(filepath, usecols=['trade_date'], dtype={'trade_date': str})
        return set(df['trade_date'].unique().tolist())
    except:
        return set()

# ----------------------------------------------------------------------
# 数据加载 (分段版 + 极速GC)
# ----------------------------------------------------------------------
def load_data_for_batch(batch_trade_days):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    
    if not batch_trade_days: return False
    
    latest_date = max(batch_trade_days)
    earliest_date = min(batch_trade_days)
    
    # 动态计算所需数据范围 (前推150天够算MACD了)
    data_start = (datetime.strptime(earliest_date, "%Y%m%d") - timedelta(days=160)).strftime("%Y%m%d")
    data_end = (datetime.strptime(latest_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d")
    
    msg_slot = st.empty()
    msg_slot.info(f"📥 正在加载数据片段: {data_start} ~ {data_end} ...")
    
    try:
        cal = st.session_state.pro.trade_cal(start_date=data_start, end_date=data_end, is_open='1')
        all_dates = cal['cal_date'].tolist()
    except:
        return False
    
    adj_list, daily_list = [], []
    
    # 进度条仅在加载数据时显示
    load_bar = st.progress(0)
    total = len(all_dates)
    
    for i, date in enumerate(all_dates):
        try:
            # 仅拉取需要的字段，大幅节省内存
            df = st.session_state.pro.daily(trade_date=date, fields='ts_code,trade_date,open,high,low,close,pre_close,vol')
            if not df.empty:
                # 强转 float32
                for c in ['open', 'high', 'low', 'close', 'pre_close', 'vol']:
                    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
                daily_list.append(df)
            
            adj = st.session_state.pro.adj_factor(trade_date=date)
            if not adj.empty:
                adj_list.append(adj)
            
            # 每20天更新一次进度条，减少刷新开销
            if i % 20 == 0: load_bar.progress((i + 1) / total)
        except: continue
        
    load_bar.empty()
    msg_slot.empty()
    
    if not daily_list: return False
    
    # 构建 DataFrame
    GLOBAL_DAILY_RAW = pd.concat(daily_list)
    GLOBAL_DAILY_RAW = GLOBAL_DAILY_RAW.drop_duplicates(subset=['ts_code', 'trade_date'])
    # 建立多级索引，这是速度的关键
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
    """强制清理全局变量"""
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR
    GLOBAL_DAILY_RAW = pd.DataFrame()
    GLOBAL_ADJ_FACTOR = pd.DataFrame()
    gc.collect()

# ----------------------------------------------------------------------
# 计算逻辑 (保持不变)
# ----------------------------------------------------------------------
def get_qfq_data_batch(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    try:
        # 极速切片
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

def compute_indicators(ts_code, current_date):
    start_date = (datetime.strptime(current_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data_batch(ts_code, start_date, current_date)
    res = {}
    if df.empty or len(df) < 20: return res # 只要有20天数据就算
    
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
    if d1['open'] <= d1['pre_close']: return {} # 必须高开
    
    limit_price = d1['pre_close'] * 1.095
    if d1['open'] >= limit_price and d1['low'] >= d1['open']: return {} # 剔除一字
    
    buy_price = d1['open'] * (1 + buy_threshold/100)
    if d1['high'] < buy_price: return {} # 必须突破
    
    for n in [1, 3, 5]:
        idx = n - 1
        if len(df) > idx:
            sell_price = df.iloc[idx]['close']
            res[f'Return_D{n} (%)'] = (sell_price / buy_price - 1) * 100
            
    return res

# ---------------------------
# 执行逻辑
# ---------------------------
if start_btn:
    if not st.session_state.ts_token:
        st.error("❌ 请先输入 Token")
        st.stop()
        
    st.info("⏳ 正在初始化...")
    
    # 1. 获取所有计划回测的日期
    end_date_str = datetime.now().strftime("%Y%m%d")
    all_target_days = get_trade_days(end_date_str, backtest_days)
    if not all_target_days:
        st.error("无法获取交易日历，请检查网络或Token")
        st.stop()
    all_target_days = sorted(all_target_days)
    
    # 2. [核心] 检查断点，剔除已完成的日期
    results_file = "backtest_result.csv"
    finished_dates = load_processed_dates(results_file)
    
    # 计算还需要跑的日期
    days_to_run = [d for d in all_target_days if d not in finished_dates]
    
    if len(finished_dates) > 0:
        st.warning(f"检测到历史记录：已完成 {len(finished_dates)} 天，自动跳过。本次仅需跑 {len(days_to_run)} 天。")
    
    if not days_to_run:
        st.success("🎉 所有日期已全部跑完！请直接下载结果。")
    else:
        # 3. 分批次执行
        BATCH_SIZE = 40 # 进一步减小Batch Size防止内存溢出
        total_batches = (len(days_to_run) + BATCH_SIZE - 1) // BATCH_SIZE
        
        status_text = st.empty()
        main_progress = st.progress(0)
        
        for batch_idx in range(total_batches):
            start_i = batch_idx * BATCH_SIZE
            end_i = min((batch_idx + 1) * BATCH_SIZE, len(days_to_run))
            batch_days = days_to_run[start_i:end_i]
            
            status_text.markdown(f"### 🔄 正在处理批次 {batch_idx+1}/{total_batches} ({batch_days[0]} ~ {batch_days[-1]})")
            
            # 这一步最耗内存，失败了直接跳过本批次，不崩溃整个程序
            if not load_data_for_batch(batch_days):
                st.error(f"⚠️ 批次 {batch_idx+1} 数据加载失败，跳过...")
                continue
                
            batch_results = []
            
            for d_idx, date in enumerate(batch_days):
                try:
                    # 每日选股
                    df_basic = st.session_state.pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,volume_ratio,circ_mv,close')
                    if df_basic is None or df_basic.empty: continue
                    
                    # 初筛
                    pool = df_basic[
                        (df_basic['close'] >= min_price) & 
                        (df_basic['close'] <= max_price) &
                        (df_basic['circ_mv'] > 200000) & # 2亿市值
                        (df_basic['turnover_rate'] > 3.0)
                    ]
                    
                    candidates = []
                    for row in pool.itertuples():
                        # 核心计算
                        ind = compute_indicators(row.ts_code, date)
                        if not ind: continue
                        
                        if ind['close'] <= ind['ma20']: continue
                        if ind['vol'] <= ind['ma5_vol'] * 1.2: continue
                        if ind['macd_val'] <= 0: continue
                        
                        macd_score = (ind['macd_val'] / ind['close']) * 1000000
                        bonus = 1.0
                        # 资金共振加分
                        if 1.5 < getattr(row, 'volume_ratio', 0) < 5.0: bonus += 0.1
                        if 5.0 < getattr(row, 'turnover_rate', 0) < 15.0: bonus += 0.1
                        if ind['pct_chg'] > 9.5: bonus += 0.1
                        
                        score = macd_score * bonus
                        
                        candidates.append({
                            'ts_code': row.ts_code,
                            'trade_date': date,
                            'score': score,
                            'close': ind['close']
                        })
                    
                    if candidates:
                        # 排序取 Top 1
                        day_df = pd.DataFrame(candidates).sort_values('score', ascending=False).head(1)
                        
                        # 回测买入
                        for rec in day_df.itertuples():
                            ret = check_buy_and_profit(rec.ts_code, rec.trade_date, buy_threshold)
                            rec_dict = rec._asdict()
                            rec_dict.update(ret)
                            batch_results.append(rec_dict)
                            
                except Exception:
                    continue
                
                # 更新进度条
                current_percent = (start_i + d_idx + 1) / len(days_to_run)
                main_progress.progress(current_percent)
            
            # 保存本批次结果
            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                # 实时写入磁盘
                header = not os.path.exists(results_file)
                df_batch.to_csv(results_file, mode='a', header=header, index=False, encoding='utf-8-sig')
                st.toast(f"✅ 已保存 {len(df_batch)} 条新记录")
            
            # 强制内存清理
            clear_memory()
    
    st.success("🎉 任务全部完成！")

# ---------------------------
# 结果展示区 (始终显示)
# ---------------------------
st.markdown("---")
if os.path.exists("backtest_result.csv"):
    try:
        final_df = pd.read_csv("backtest_result.csv")
        st.subheader("📊 回测结果分析")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总交易次数", len(final_df))
        
        if 'Return_D1 (%)' in final_df.columns:
            valid = final_df.dropna(subset=['Return_D1 (%)'])
            avg = valid['Return_D1 (%)'].mean()
            win = (valid['Return_D1 (%)'] > 0).mean() * 100
            with col2:
                st.metric("D+1 平均收益", f"{avg:.2f}%")
            with col3:
                st.metric("D+1 胜率", f"{win:.1f}%")
        
        st.dataframe(final_df, width=None)
        
        with open("backtest_result.csv", "rb") as f:
            st.download_button(
                label="📥 下载完整 CSV",
                data=f,
                file_name="rank1_final_result.csv",
                mime="text/csv",
                type="primary"
            )
    except:
        st.info("暂无结果数据")
