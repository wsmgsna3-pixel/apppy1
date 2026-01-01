# -*- coding: utf-8 -*-
"""
第一名 3.1 自由版 (解锁TopK + 防卡死优化)
核心升级：
1. [自由设置] 解除 Top 1 锁定，您可以自由回测 Top 3、Top 5 等组合。
2. [拒绝卡死] 批次大小 (Batch Size) 默认下调至 15，适配云端内存，防止进度条假死。
3. [极速内核] 保持向量化计算逻辑，500天回测依然丝滑。
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
# 全局设置
# ---------------------------
st.set_page_config(page_title="第一名 3.1 自由版", layout="wide")

if 'pro' not in st.session_state:
    st.session_state.pro = None
if 'ts_token' not in st.session_state:
    st.session_state.ts_token = ""

# ---------------------------
# 界面布局
# ---------------------------
st.title("⚡ 第一名 3.1 自由版 (Top K 解锁 + 防卡死)")
st.caption("🚀 500天长周期回测专用 | 向量化极速内核 | 内存防爆优化")

with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        new_token = st.text_input("💎 Tushare Token", value=st.session_state.ts_token, type="password")
        if new_token:
            st.session_state.ts_token = new_token
            ts.set_token(new_token)
            st.session_state.pro = ts.pro_api()
    with col2:
        st.write("")
        st.write("")
        start_btn = st.button("🚀 极速回测", type="primary", use_container_width=True)

# 参数区域 (默认展开)
with st.expander("⚙️ 策略参数设置", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        backtest_days = st.number_input("回测天数", value=500, step=50, help="建议500天以覆盖牛熊")
        stop_loss_pct = st.number_input("止损阈值 (%)", value=-4.0, step=0.5, help="盘中触及即止损，填负数")
    with c2:
        min_price = st.number_input("最低股价", value=40.0)
        max_price = st.number_input("最高股价", value=300.0)
    with c3:
        buy_threshold = st.number_input("买入阈值 (%)", value=1.5)
        # [修改点] 解锁 Top K，允许用户自由输入，默认改为 3 以测试组合效果
        top_k = st.number_input("每日持仓 (Top K)", value=3, min_value=1, max_value=20, disabled=False, help="设置为 1 即只买第一名，设置为 3 即买前三名")

# ---------------------------
# 核心引擎 (向量化)
# ---------------------------
def get_trade_days(end_date_str, num_days):
    # 扩大获取范围以应对节假日
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3 + 300)).strftime("%Y%m%d")
    if st.session_state.pro:
        try:
            cal = st.session_state.pro.trade_cal(start_date=start_date, end_date=end_date_str, is_open='1')
            return cal.sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()
        except: return []
    return []

def load_data_and_compute_vectorized(date_list):
    """
    [核心黑科技] 向量化计算引擎
    """
    if not date_list: return None
    
    # 1. 确定数据范围
    start_date = min(date_list)
    end_date = max(date_list)
    # 缓冲期需足够长以保证MACD准确
    buffer_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    
    st.info(f"📥 极速加载数据: {buffer_start} ~ {end_date} ...")
    
    # 2. 批量拉取数据
    chunk_dates = st.session_state.pro.trade_cal(start_date=buffer_start, end_date=end_date, is_open='1')['cal_date'].tolist()
    
    dfs = []
    # 简化进度显示
    bar = st.progress(0)
    total_chunks = len(chunk_dates)
    
    for i, d in enumerate(chunk_dates):
        try:
            # 只拉取必要字段
            daily = st.session_state.pro.daily(trade_date=d, fields='ts_code,trade_date,open,high,low,close,pre_close,vol')
            if not daily.empty:
                # 压缩数据类型 float32 节省内存
                for c in ['open','high','low','close','pre_close','vol']:
                    daily[c] = pd.to_numeric(daily[c], errors='coerce').astype('float32')
                dfs.append(daily)
        except: pass
        if i % 10 == 0: bar.progress((i+1)/total_chunks)
    bar.empty()
    
    if not dfs: return None
    
    # 合并大表
    df_all = pd.concat(dfs).sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    # 3. [复权处理] 
    st.caption("🔧 正在进行向量化复权与指标计算...")
    
    adj_dfs = []
    for i, d in enumerate(chunk_dates):
         try:
            adj = st.session_state.pro.adj_factor(trade_date=d, fields='ts_code,adj_factor')
            if not adj.empty:
                adj['trade_date'] = d
                adj_dfs.append(adj)
         except: pass
    
    if adj_dfs:
        adj_all = pd.concat(adj_dfs)
        df_all = pd.merge(df_all, adj_all, on=['ts_code', 'trade_date'], how='left')
        df_all['adj_factor'] = df_all['adj_factor'].fillna(method='ffill') 
        
        # 计算后复权价格用于指标计算
        df_all['hfq_close'] = df_all['close'] * df_all['adj_factor']
    else:
        df_all['hfq_close'] = df_all['close']
    
    # 4. [向量化指标计算]
    grouped = df_all.groupby('ts_code')['hfq_close']
    
    # MACD (8, 17, 5)
    ema8 = grouped.ewm(span=8, adjust=False).mean().reset_index(level=0, drop=True)
    ema17 = grouped.ewm(span=17, adjust=False).mean().reset_index(level=0, drop=True)
    
    df_all['diff'] = ema8 - ema17
    df_all['dea'] = df_all.groupby('ts_code')['diff'].ewm(span=5, adjust=False).mean().reset_index(level=0, drop=True)
    df_all['macd'] = (df_all['diff'] - df_all['dea']) * 2
    
    # MA20
    df_all['ma20'] = df_all.groupby('ts_code')['hfq_close'].rolling(20).mean().reset_index(level=0, drop=True)
    
    # MA5_Vol
    df_all['ma5_vol'] = df_all.groupby('ts_code')['vol'].rolling(5).mean().reset_index(level=0, drop=True)
    
    # 5. 过滤掉缓冲期数据，只保留目标回测期
    df_target = df_all[df_all['trade_date'].isin(date_list)].copy()
    
    # 立即清理内存
    del df_all, ema8, ema17, adj_dfs
    gc.collect()
    
    return df_target

def check_profit_with_stop_loss(ts_code, buy_date, buy_price, stop_loss_pct):
    """
    [风控引擎] 获取未来收益，包含刚性止损逻辑
    """
    try:
        d0 = datetime.strptime(buy_date, "%Y%m%d")
        f_start = (d0 + timedelta(days=1)).strftime("%Y%m%d")
        f_end = (d0 + timedelta(days=15)).strftime("%Y%m%d")
        
        # 拉取单只股票未来数据
        df = st.session_state.pro.daily(ts_code=ts_code, start_date=f_start, end_date=f_end, fields='trade_date,open,high,low,close,pre_close')
        if df.empty: return {}
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 检查买入条件 (T+1)
        d1 = df.iloc[0]
        limit_up = d1['pre_close'] * 1.095
        if d1['open'] <= d1['pre_close']: return {'status': '低开放弃'}
        if d1['open'] >= limit_up and d1['low'] >= d1['open']: return {'status': '一字板放弃'}
        if d1['high'] < buy_price: return {'status': '未突破'}
        
        # 成交
        res = {'status': '成交'}
        stop_price = buy_price * (1 + stop_loss_pct/100)
        
        for n in [1, 3, 5]:
            if len(df) >= n:
                triggered_stop = False
                for i in range(n):
                    day_low = df.iloc[i]['low']
                    if day_low <= stop_price:
                        # 触发止损，按止损价或开盘价离场
                        exit_price = min(stop_price, df.iloc[i]['open'])
                        ret = (exit_price / buy_price - 1) * 100
                        res[f'Return_D{n} (%)'] = ret
                        res[f'Stop_D{n}'] = True
                        triggered_stop = True
                        break 
                
                if not triggered_stop:
                    close_price = df.iloc[n-1]['close']
                    res[f'Return_D{n} (%)'] = (close_price / buy_price - 1) * 100
                    res[f'Stop_D{n}'] = False
                    
        return res
        
    except: return {}

# ---------------------------
# 主程序
# ---------------------------
if start_btn:
    if not st.session_state.ts_token:
        st.error("❌ 请先输入 Token")
        st.stop()
        
    # 1. 获取日期
    end_date_str = datetime.now().strftime("%Y%m%d")
    all_days = get_trade_days(end_date_str, backtest_days)
    all_days = sorted(all_days)
    
    # 2. 智能分段
    # [关键修改] 这里设置为 15，解决 500 天回测卡死问题
    BATCH_SIZE = 15
    batches = [all_days[i:i + BATCH_SIZE] for i in range(0, len(all_days), BATCH_SIZE)]
    
    results_file = "rank_free_v3_results.csv"
    if os.path.exists(results_file): os.remove(results_file) 
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_trades = 0
    
    for b_i, batch_days in enumerate(batches):
        status_text.markdown(f"### ⚡ 正在极速计算: {batch_days[0]} ~ {batch_days[-1]} ({b_i+1}/{len(batches)})")
        
        # A. 向量化准备数据
        df_batch = load_data_and_compute_vectorized(batch_days)
        if df_batch is None or df_batch.empty: continue
        
        # B. 每日选股
        batch_results = []
        
        for day in batch_days:
            try:
                # 内存筛选
                day_data = df_batch[df_batch['trade_date'] == day]
                if day_data.empty: continue
                
                # 获取 Basic (换手、量比、市值)
                basic = st.session_state.pro.daily_basic(trade_date=day, fields='ts_code,turnover_rate,volume_ratio,circ_mv')
                if basic is None or basic.empty: continue
                
                # 合并
                merged = pd.merge(day_data, basic, on='ts_code', how='inner')
                
                # V30.22 核心筛选
                mask = (
                    (merged['hfq_close'] > merged['ma20']) &
                    (merged['vol'] > merged['ma5_vol'] * 1.2) &
                    (merged['macd'] > 0) &
                    (merged['close'] >= min_price) & 
                    (merged['close'] <= max_price) &
                    (merged['turnover_rate'] > 3.0) &
                    (merged['circ_mv'] > 200000) 
                )
                candidates = merged[mask].copy()
                
                if candidates.empty: continue
                
                # 评分
                candidates['base_score'] = (candidates['macd'] / candidates['hfq_close']) * 1000000
                
                candidates['pct_chg'] = (candidates['close'] / candidates['pre_close'] - 1) * 100
                candidates['bonus'] = 1.0
                candidates.loc[(candidates['volume_ratio'] > 1.5) & (candidates['volume_ratio'] < 5.0), 'bonus'] += 0.1
                candidates.loc[(candidates['turnover_rate'] > 5.0) & (candidates['turnover_rate'] < 15.0), 'bonus'] += 0.1
                candidates.loc[candidates['pct_chg'] > 9.5, 'bonus'] += 0.1
                
                candidates['final_score'] = candidates['base_score'] * candidates['bonus']
                
                # [关键修改] 使用用户自定义的 top_k
                top_selection = candidates.sort_values('final_score', ascending=False).head(top_k)
                
                # 模拟交易
                for row in top_selection.itertuples():
                    buy_price = row.open * (1 + buy_threshold/100)
                    res = check_profit_with_stop_loss(row.ts_code, day, buy_price, stop_loss_pct)
                    
                    if res.get('status') == '成交':
                        rec = {
                            'trade_date': day,
                            'ts_code': row.ts_code,
                            'close': row.close,
                            'score': row.final_score
                        }
                        rec.update(res)
                        batch_results.append(rec)
            
            except Exception: pass
        
        # C. 写入结果
        if batch_results:
            df_res = pd.DataFrame(batch_results)
            df_res.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False, encoding='utf-8-sig')
            total_trades += len(df_res)
            st.toast(f"✅ 新增 {len(df_res)} 笔交易 (累计: {total_trades})")
            
        progress_bar.progress((b_i + 1) / len(batches))
        gc.collect() # 强制垃圾回收

    st.success("🎉 极速回测完成！")
    
    # 结果展示
    if os.path.exists(results_file):
        res_df = pd.read_csv(results_file)
        st.subheader("📊 最终回测报告")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总交易次数", len(res_df))
        
        if 'Return_D1 (%)' in res_df.columns:
            avg_d1 = res_df['Return_D1 (%)'].mean()
            win_d1 = (res_df['Return_D1 (%)'] > 0).mean() * 100
            c2.metric("D+1 均收", f"{avg_d1:.2f}%")
            c3.metric("D+1 胜率", f"{win_d1:.1f}%")
            
            # 简单回撤计算
            res_df = res_df.sort_values('trade_date')
            # 假设每日均仓
            daily_ret = res_df.groupby('trade_date')['Return_D1 (%)'].mean()
            equity = daily_ret.cumsum()
            dd = equity.cummax() - equity
            c4.metric("最大回撤", f"{dd.max():.2f}")
            
        st.dataframe(res_df, use_container_width=True)
        with open(results_file, "rb") as f:
            st.download_button("📥 下载详细战报", f, "rank_free_v3_fast.csv")
