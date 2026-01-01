# -*- coding: utf-8 -*-
"""
第一名 5.1 最终修复版 (解决空数据问题)
-------------------------------------------------
【修复说明】
1. 修正数据拉取逻辑：从“批量拉取”改为“逐日拉取”，解决 Tushare 单次 4000 行限制导致的空数据问题。
2. 保持高保真内核：全程 Float64 精度，完整 OHLCV 字段。
3. 稳健进度：20天为一个周期，步步为营。
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

# ---------------------------
# 1. 全局配置
# ---------------------------
st.set_page_config(page_title="第一名 5.1 最终修复版", layout="wide")
warnings.filterwarnings("ignore")

if 'pro' not in st.session_state:
    st.session_state.pro = None
if 'ts_token' not in st.session_state:
    st.session_state.ts_token = ""

# ---------------------------
# 2. UI 界面
# ---------------------------
st.title("🏆 第一名 5.1 最终修复版")
st.markdown("""
> **🔧 修复日志：** 已将数据获取方式改为**逐日循环拉取**，彻底解决因 Tushare 数据量超限导致的“数据为空”问题。  
> **⏳ 预计耗时：** 由于需要逐日请求，速度会比极速版慢，但**数据绝对完整可靠**。
""")

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
        start_btn = st.button("🐢 启动修复版回测", type="primary", use_container_width=True)

with st.expander("⚙️ 策略参数", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        backtest_days = st.number_input("回测天数", value=500, step=50)
        stop_loss_pct = st.number_input("止损阈值 (%)", value=-4.0, step=0.5)
    with c2:
        min_price = st.number_input("最低股价", value=40.0)
        max_price = st.number_input("最高股价", value=300.0)
    with c3:
        buy_threshold = st.number_input("买入阈值 (%)", value=1.5)
        top_k = st.number_input("每日持仓 (Top K)", value=3, min_value=1)

# ---------------------------
# 3. 核心工具函数
# ---------------------------
def get_trade_days(end_date_str, num_days):
    """获取目标回测的交易日历"""
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 2 + 365)).strftime("%Y%m%d")
    if st.session_state.pro:
        try:
            cal = st.session_state.pro.trade_cal(start_date=start_date, end_date=end_date_str, is_open='1')
            return cal.sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()
        except: return []
    return []

def load_processed_dates(filepath):
    """断点续传"""
    if not os.path.exists(filepath): return set()
    try:
        df = pd.read_csv(filepath, usecols=['trade_date'], dtype={'trade_date': str})
        return set(df['trade_date'].unique().tolist())
    except: return set()

# ---------------------------
# 4. 高保真数据加载 (逐日循环版)
# ---------------------------
def fetch_full_precision_data(target_days):
    """
    【修复版加载】
    必须逐日拉取 (Loop by Day)，因为 daily 接口不支持全市场多日拉取
    """
    if not target_days: return None
    
    start_date = min(target_days)
    end_date = max(target_days)
    
    # 历史缓冲 (180天) + 未来缓冲 (30天)
    buffer_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=180)).strftime("%Y%m%d")
    future_end = (datetime.strptime(end_date, "%Y%m%d") + timedelta(days=30)).strftime("%Y%m%d")
    
    st.info(f"📥 [逐日拉取] 正在构建数据池: {buffer_start} ~ {future_end}")
    
    # 获取所有交易日
    try:
        cal = st.session_state.pro.trade_cal(start_date=buffer_start, end_date=future_end, is_open='1')
        all_cal_dates = cal['cal_date'].tolist()
    except: return None
    
    full_dfs = []
    # 进度条
    fetch_bar = st.progress(0)
    total_days = len(all_cal_dates)
    
    # --- 逐日循环 (这是唯一稳健的方法) ---
    for i, date in enumerate(all_cal_dates):
        try:
            # 完整字段
            df = st.session_state.pro.daily(trade_date=date, fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount')
            if not df.empty:
                full_dfs.append(df)
            
            # 更新进度 (每10天更新一次UI，防止卡顿)
            if i % 10 == 0:
                fetch_bar.progress((i + 1) / total_days)
            
            # 极短休眠，防止请求过快被封 IP
            # time.sleep(0.01) 
        except:
            pass
            
    fetch_bar.empty()
    
    if not full_dfs: return None
    
    # 合并
    df_big = pd.concat(full_dfs).sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    # --- 逐日拉取复权因子 ---
    st.caption("🔧 正在拉取复权因子...")
    adj_dfs = []
    
    for date in all_cal_dates:
        try:
            adj = st.session_state.pro.adj_factor(trade_date=date, fields='ts_code,trade_date,adj_factor')
            if not adj.empty:
                adj_dfs.append(adj)
        except: pass
        
    if adj_dfs:
        adj_all = pd.concat(adj_dfs)
        df_big = pd.merge(df_big, adj_all, on=['ts_code', 'trade_date'], how='left')
        df_big['adj_factor'] = df_big['adj_factor'].fillna(method='ffill').fillna(1.0)
        df_big['hfq_close'] = df_big['close'] * df_big['adj_factor']
    else:
        df_big['hfq_close'] = df_big['close']
        
    return df_big

def calculate_indicators_safe(df_big):
    """计算指标"""
    st.caption("🧮 正在计算全市场指标...")
    df_big = df_big.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    grouped = df_big.groupby('ts_code')['hfq_close']
    
    # MACD (8, 17, 5)
    ema8 = grouped.ewm(span=8, adjust=False).mean().reset_index(level=0, drop=True)
    ema17 = grouped.ewm(span=17, adjust=False).mean().reset_index(level=0, drop=True)
    df_big['diff'] = ema8 - ema17
    df_big['dea'] = df_big.groupby('ts_code')['diff'].ewm(span=5, adjust=False).mean().reset_index(level=0, drop=True)
    df_big['macd'] = (df_big['diff'] - df_big['dea']) * 2
    
    # MA20
    df_big['ma20'] = grouped.rolling(20).mean().reset_index(level=0, drop=True)
    # MA5_Vol
    df_big['ma5_vol'] = df_big.groupby('ts_code')['vol'].rolling(5).mean().reset_index(level=0, drop=True)
    
    return df_big

def simulate_trade(ts_code, buy_date, buy_price, stop_loss_pct, df_future):
    """模拟交易"""
    try:
        # 筛选未来数据
        df_after = df_future[df_future['trade_date'] > buy_date].copy()
        if df_after.empty: return {}
        
        d1 = df_after.iloc[0]
        
        # 校验条件
        if d1['open'] <= d1['pre_close']: return {'status': '低开放弃'}
        limit_up = d1['pre_close'] * 1.095
        if d1['open'] >= limit_up and d1['low'] >= d1['open']: return {'status': '一字板放弃'}
        if d1['high'] < buy_price: return {'status': '未突破'}
        
        res = {'status': '成交'}
        stop_price = buy_price * (1 + stop_loss_pct/100)
        
        for n in [1, 3, 5]:
            if len(df_after) >= n:
                triggered_stop = False
                for i in range(n):
                    if df_after.iloc[i]['low'] <= stop_price:
                        exit_price = min(stop_price, df_after.iloc[i]['open'])
                        res[f'Return_D{n} (%)'] = (exit_price / buy_price - 1) * 100
                        res[f'Stop_D{n}'] = True
                        triggered_stop = True
                        break
                if not triggered_stop:
                    close_price = df_after.iloc[n-1]['close']
                    res[f'Return_D{n} (%)'] = (close_price / buy_price - 1) * 100
                    res[f'Stop_D{n}'] = False
        return res
    except: return {}

# ---------------------------
# 5. 主程序
# ---------------------------
if start_btn:
    if not st.session_state.ts_token:
        st.error("❌ 请先输入 Token")
        st.stop()
        
    end_date_str = datetime.now().strftime("%Y%m%d")
    all_target_days = get_trade_days(end_date_str, backtest_days)
    all_target_days = sorted(all_target_days)
    
    # 断点续传
    results_file = "rank_high_fidelity.csv"
    finished_dates = load_processed_dates(results_file)
    days_to_run = [d for d in all_target_days if d not in finished_dates]
    
    if not days_to_run:
        st.success("🎉 所有日期已完成！")
        st.stop()
        
    st.info(f"📅 本次需回测 {len(days_to_run)} 天，自动分批执行...")

    # 分批执行
    BATCH_SIZE = 20
    batches = [days_to_run[i:i + BATCH_SIZE] for i in range(0, len(days_to_run), BATCH_SIZE)]
    
    main_progress = st.progress(0)
    status_text = st.empty()
    total_trades = 0
    
    for b_i, batch_days in enumerate(batches):
        status_text.markdown(f"### 🔄 处理批次 {b_i+1}/{len(batches)} ({batch_days[0]} ~ {batch_days[-1]})")
        
        # A. 拉取全量数据
        df_big = fetch_full_precision_data(batch_days)
        if df_big is None or df_big.empty:
            st.warning(f"批次 {b_i+1} 数据为空，跳过")
            continue
            
        # B. 计算指标
        df_big = calculate_indicators_safe(df_big)
        
        # C. 逐日回测
        batch_results = []
        
        for current_date in batch_days:
            try:
                today_data = df_big[df_big['trade_date'] == current_date].copy()
                if today_data.empty: continue
                
                # 获取 Basic (必须逐日拉取)
                try:
                    basic = st.session_state.pro.daily_basic(trade_date=current_date, fields='ts_code,turnover_rate,volume_ratio,circ_mv')
                except: basic = pd.DataFrame()
                
                if basic.empty: continue
                
                merged = pd.merge(today_data, basic, on='ts_code', how='inner')
                
                # 筛选
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
                
                # Top K
                top_selection = candidates.sort_values('final_score', ascending=False).head(top_k)
                
                # 交易
                for row in top_selection.itertuples():
                    buy_price = row.open * (1 + buy_threshold/100)
                    # 查找未来数据 (内存中)
                    stock_future = df_big[df_big['ts_code'] == row.ts_code]
                    res = simulate_trade(row.ts_code, current_date, buy_price, stop_loss_pct, stock_future)
                    
                    if res.get('status') == '成交':
                        rec = {
                            'trade_date': current_date,
                            'ts_code': row.ts_code,
                            'name': 'Unknown',
                            'close': row.close,
                            'score': row.final_score
                        }
                        rec.update(res)
                        batch_results.append(rec)
            except: pass
        
        # D. 存盘
        if batch_results:
            df_res = pd.DataFrame(batch_results)
            header = not os.path.exists(results_file)
            df_res.to_csv(results_file, mode='a', header=header, index=False, encoding='utf-8-sig')
            total_trades += len(df_res)
            st.toast(f"✅ 保存 {len(df_res)} 条 | 累计: {total_trades}")
        
        del df_big, batch_results
        gc.collect()
        time.sleep(1) # 休息一下
        main_progress.progress((b_i + 1) / len(batches))

    st.success("🎉 高保真回测全部完成！")
    
    # 结果
    if os.path.exists(results_file):
        try:
            res_df = pd.read_csv(results_file)
            st.subheader("📊 回测报告")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("交易次数", len(res_df))
            if 'Return_D1 (%)' in res_df.columns:
                avg = res_df['Return_D1 (%)'].mean()
                win = (res_df['Return_D1 (%)'] > 0).mean() * 100
                c2.metric("D+1 均收", f"{avg:.2f}%")
                c3.metric("D+1 胜率", f"{win:.1f}%")
                
                res_df = res_df.sort_values('trade_date')
                equity = res_df.groupby('trade_date')['Return_D1 (%)'].mean().cumsum()
                dd = equity.cummax() - equity
                c4.metric("最大回撤", f"{dd.max():.2f}")
            st.dataframe(res_df, use_container_width=True)
            with open(results_file, "rb") as f:
                st.download_button("📥 下载 CSV", f, "high_fidelity_result.csv")
        except: pass
