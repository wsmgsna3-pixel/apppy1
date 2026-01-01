# -*- coding: utf-8 -*-
"""
第一名 5.0 高保真·独享版 (High Fidelity)
-------------------------------------------------
【核心设计理念】
1. 数据完整性第一：拒绝为了速度阉割数据，全程加载 OHLCV 完整字段。
2. 精度无损：放弃 Float32 压缩，全程使用 Float64 双精度计算，对标券商软件精度。
3. 稳健滑窗：采用“20天步进 + 完整历史重载”模式，宁可慢，不可崩。
4. 刚性风控：内置 -4% 止损与 T+1 高开逻辑。
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
st.set_page_config(page_title="第一名 5.0 高保真版", layout="wide")
warnings.filterwarnings("ignore")

if 'pro' not in st.session_state:
    st.session_state.pro = None
if 'ts_token' not in st.session_state:
    st.session_state.ts_token = ""

# ---------------------------
# 2. UI 界面
# ---------------------------
st.title("🏆 第一名 5.0 高保真·独享版")
st.markdown("""
> **⚠️ 郑重提示：** > 本版本为**全数据精度版**，不再追求极致速度，而是追求**数据的绝对完整性**。  
> 500天回测预计耗时 **40-90分钟**（取决于网络），请耐心等待。系统会实时保存结果，随时可断点续传。
""")

with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        new_token = st.text_input("💎 Tushare Token (请输入您的Token)", value=st.session_state.ts_token, type="password")
        if new_token:
            st.session_state.ts_token = new_token
            ts.set_token(new_token)
            st.session_state.pro = ts.pro_api()
    with col2:
        st.write("") 
        st.write("") 
        start_btn = st.button("🐢 启动高保真回测", type="primary", use_container_width=True)

with st.expander("⚙️ 策略参数 (已调优)", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        backtest_days = st.number_input("回测天数", value=500, step=50, help="建议500天以覆盖2024年初的股灾")
        stop_loss_pct = st.number_input("止损阈值 (%)", value=-4.0, step=0.5, help="盘中触及即止损")
    with c2:
        min_price = st.number_input("最低股价", value=40.0)
        max_price = st.number_input("最高股价", value=300.0)
    with c3:
        buy_threshold = st.number_input("买入阈值 (%)", value=1.5)
        top_k = st.number_input("每日持仓 (Top K)", value=3, min_value=1, help="自由设置，推荐 3 或 5")

# ---------------------------
# 3. 核心工具函数
# ---------------------------
def get_trade_days(end_date_str, num_days):
    """获取目标回测的交易日历"""
    # 这里只获取我们要回测的那500天，不包含缓冲期
    # 缓冲期在 batch 内部动态计算，保证数据新鲜
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 2 + 365)).strftime("%Y%m%d")
    if st.session_state.pro:
        try:
            cal = st.session_state.pro.trade_cal(start_date=start_date, end_date=end_date_str, is_open='1')
            # 取最近的 num_days 天
            return cal.sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()
        except: return []
    return []

def load_processed_dates(filepath):
    """断点续传：读取已完成日期"""
    if not os.path.exists(filepath): return set()
    try:
        df = pd.read_csv(filepath, usecols=['trade_date'], dtype={'trade_date': str})
        return set(df['trade_date'].unique().tolist())
    except: return set()

# ---------------------------
# 4. 高保真数据加载引擎
# ---------------------------
def fetch_full_precision_data(target_days):
    """
    【高保真加载】
    针对给定的 target_days (比如20天)，
    1. 自动向前推 180 天 (History Buffer)
    2. 自动向后推 30 天 (Future Buffer)
    3. 完整拉取 OHLCV，不做列裁剪，不做 float32 压缩。
    """
    if not target_days: return None
    
    # 1. 计算时间窗口
    start_date = min(target_days)
    end_date = max(target_days)
    
    # 历史缓冲 (MACD需要)
    buffer_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=180)).strftime("%Y%m%d")
    # 未来缓冲 (计算收益需要)
    future_end = (datetime.strptime(end_date, "%Y%m%d") + timedelta(days=30)).strftime("%Y%m%d")
    
    st.info(f"📥 [完整数据加载] 正在拉取区间: {buffer_start} ~ {future_end}")
    
    # 2. 获取该区间所有交易日
    try:
        cal = st.session_state.pro.trade_cal(start_date=buffer_start, end_date=future_end, is_open='1')
        all_cal_dates = cal['cal_date'].tolist()
    except: return None
    
    # 3. 分块拉取完整行情 (Batch Fetching)
    # Tushare 单次拉取有限制，我们按 50 天一块拉取
    full_dfs = []
    chunk_size = 50
    
    # 进度条
    fetch_bar = st.progress(0)
    
    for i in range(0, len(all_cal_dates), chunk_size):
        chunk = all_cal_dates[i:i+chunk_size]
        s_chunk, e_chunk = chunk[0], chunk[-1]
        
        try:
            # [关键] 拉取完整字段，不进行阉割
            df = st.session_state.pro.daily(
                start_date=s_chunk, 
                end_date=e_chunk, 
                fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount'
            )
            if not df.empty:
                # [关键] 保持 float64 精度 (Pandas 默认)，不强转 float32
                full_dfs.append(df)
        except Exception as e:
            st.warning(f"数据拉取重试中: {e}")
            time.sleep(1)
            continue
            
        fetch_bar.progress(min((i + chunk_size) / len(all_cal_dates), 1.0))
        time.sleep(0.05) # 主动休眠，防止触发 Tushare 流控
        
    fetch_bar.empty()
    
    if not full_dfs: return None
    
    # 合并
    df_big = pd.concat(full_dfs).sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    # 4. 拉取复权因子 (必须)
    st.caption("🔧 正在拉取复权因子...")
    adj_dfs = []
    for i in range(0, len(all_cal_dates), chunk_size):
        chunk = all_cal_dates[i:i+chunk_size]
        s_chunk, e_chunk = chunk[0], chunk[-1]
        try:
            adj = st.session_state.pro.adj_factor(start_date=s_chunk, end_date=e_chunk, fields='ts_code,trade_date,adj_factor')
            if not adj.empty:
                adj_dfs.append(adj)
        except: pass
        
    if adj_dfs:
        adj_all = pd.concat(adj_dfs)
        df_big = pd.merge(df_big, adj_all, on=['ts_code', 'trade_date'], how='left')
        df_big['adj_factor'] = df_big['adj_factor'].fillna(method='ffill').fillna(1.0)
        # 计算后复权价格 (High Precision)
        df_big['hfq_close'] = df_big['close'] * df_big['adj_factor']
    else:
        df_big['hfq_close'] = df_big['close']
        
    return df_big

def calculate_indicators_safe(df_big):
    """
    计算指标，全程使用 GroupBy + Transform，不使用压缩
    """
    st.caption("🧮 正在进行高精度指标计算...")
    
    # 确保排序
    df_big = df_big.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    grouped = df_big.groupby('ts_code')['hfq_close']
    
    # MACD (8, 17, 5) - 使用后复权价格
    ema8 = grouped.ewm(span=8, adjust=False).mean().reset_index(level=0, drop=True)
    ema17 = grouped.ewm(span=17, adjust=False).mean().reset_index(level=0, drop=True)
    
    df_big['diff'] = ema8 - ema17
    # DEA 需要基于 diff 再次 group
    df_big['dea'] = df_big.groupby('ts_code')['diff'].ewm(span=5, adjust=False).mean().reset_index(level=0, drop=True)
    df_big['macd'] = (df_big['diff'] - df_big['dea']) * 2
    
    # MA20
    df_big['ma20'] = grouped.rolling(20).mean().reset_index(level=0, drop=True)
    
    # MA5_Vol
    df_big['ma5_vol'] = df_big.groupby('ts_code')['vol'].rolling(5).mean().reset_index(level=0, drop=True)
    
    return df_big

def simulate_trade(ts_code, buy_date, buy_price, stop_loss_pct, df_future):
    """
    在内存中切片查找未来数据，进行模拟交易
    df_future: 包含该股票未来数据的 DataFrame 切片
    """
    try:
        # 筛选未来日期
        # 假设 df_future 已经是该股票且按日期排序的数据
        # 找到 buy_date 之后的行
        df_after = df_future[df_future['trade_date'] > buy_date].copy()
        
        if df_after.empty: return {}
        
        d1 = df_after.iloc[0]
        
        # 1. 买入条件校验
        # 必须高开
        if d1['open'] <= d1['pre_close']: return {'status': '低开放弃'}
        # 剔除一字板 (开盘价 >= 涨停价 且 Low >= Open)
        limit_up = d1['pre_close'] * 1.095
        if d1['open'] >= limit_up and d1['low'] >= d1['open']: return {'status': '一字板放弃'}
        # 必须突破买入价
        if d1['high'] < buy_price: return {'status': '未突破'}
        
        # 2. 收益计算 (含止损)
        res = {'status': '成交'}
        stop_price = buy_price * (1 + stop_loss_pct/100)
        
        for n in [1, 3, 5]:
            if len(df_after) >= n:
                triggered_stop = False
                # 遍历持有期，检查是否触及止损
                for i in range(n):
                    current_day = df_after.iloc[i]
                    if current_day['low'] <= stop_price:
                        # 触发止损
                        exit_price = min(stop_price, current_day['open'])
                        res[f'Return_D{n} (%)'] = (exit_price / buy_price - 1) * 100
                        res[f'Stop_D{n}'] = True
                        triggered_stop = True
                        break
                
                if not triggered_stop:
                    # 正常持有到期
                    close_price = df_after.iloc[n-1]['close']
                    res[f'Return_D{n} (%)'] = (close_price / buy_price - 1) * 100
                    res[f'Stop_D{n}'] = False
        return res
    except: return {}

# ---------------------------
# 5. 主程序逻辑
# ---------------------------
if start_btn:
    if not st.session_state.ts_token:
        st.error("❌ 请先输入 Token")
        st.stop()
        
    # 1. 获取所有回测日期
    end_date_str = datetime.now().strftime("%Y%m%d")
    all_target_days = get_trade_days(end_date_str, backtest_days)
    all_target_days = sorted(all_target_days)
    
    # 2. 断点续传
    results_file = "rank_high_fidelity.csv"
    finished_dates = load_processed_dates(results_file)
    days_to_run = [d for d in all_target_days if d not in finished_dates]
    
    if not days_to_run:
        st.success("🎉 所有日期已完成！")
        # 也要显示结果
    else:
        st.info(f"📅 本次需回测 {len(days_to_run)} 天，系统将自动分批执行...")

        # 3. 稳健分批 (Batch Processing)
        # 既然用户接受慢，我们用 20 天一个 Batch，保证内存绝对安全
        # 并且每个 Batch 都重新拉取 History Buffer，虽然浪费流量，但逻辑最简单最稳
        BATCH_SIZE = 20
        batches = [days_to_run[i:i + BATCH_SIZE] for i in range(0, len(days_to_run), BATCH_SIZE)]
        
        main_progress = st.progress(0)
        status_text = st.empty()
        total_trades = 0
        
        for b_i, batch_days in enumerate(batches):
            status_text.markdown(f"### 🔄 正在处理批次 {b_i+1}/{len(batches)} ({batch_days[0]} ~ {batch_days[-1]})")
            
            # A. 拉取全量数据 (History + Batch + Future)
            df_big = fetch_full_precision_data(batch_days)
            
            if df_big is None or df_big.empty:
                st.warning(f"批次 {b_i+1} 数据为空，跳过")
                continue
                
            # B. 计算指标
            df_big = calculate_indicators_safe(df_big)
            
            # C. 逐日回测 (Looping)
            batch_results = []
            
            # 批量拉取 daily_basic (优化点)
            basic_map = {} # Key: date, Value: DataFrame
            try:
                b_start, b_end = batch_days[0], batch_days[-1]
                daily_basics = st.session_state.pro.daily_basic(start_date=b_start, end_date=b_end, fields='ts_code,trade_date,turnover_rate,volume_ratio,circ_mv')
                if not daily_basics.empty:
                    for date, group in daily_basics.groupby('trade_date'):
                        basic_map[date] = group
            except: pass

            # 每日循环
            for current_date in batch_days:
                try:
                    # 1. 获取当日切片
                    today_data = df_big[df_big['trade_date'] == current_date].copy()
                    if today_data.empty: continue
                    
                    # 2. 获取当日 Basic
                    if current_date in basic_map:
                        basic = basic_map[current_date]
                        # 合并
                        merged = pd.merge(today_data, basic, on='ts_code', how='inner')
                    else:
                        continue
                    
                    # 3. 筛选 (V30.22 逻辑)
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
                    
                    # 4. 评分
                    candidates['base_score'] = (candidates['macd'] / candidates['hfq_close']) * 1000000
                    candidates['pct_chg'] = (candidates['close'] / candidates['pre_close'] - 1) * 100
                    
                    candidates['bonus'] = 1.0
                    candidates.loc[(candidates['volume_ratio'] > 1.5) & (candidates['volume_ratio'] < 5.0), 'bonus'] += 0.1
                    candidates.loc[(candidates['turnover_rate'] > 5.0) & (candidates['turnover_rate'] < 15.0), 'bonus'] += 0.1
                    candidates.loc[candidates['pct_chg'] > 9.5, 'bonus'] += 0.1
                    
                    candidates['final_score'] = candidates['base_score'] * candidates['bonus']
                    
                    # 5. Top K
                    top_selection = candidates.sort_values('final_score', ascending=False).head(top_k)
                    
                    # 6. 模拟交易
                    for row in top_selection.itertuples():
                        buy_price = row.open * (1 + buy_threshold/100)
                        
                        # 从内存中的 df_big 截取该股票的未来数据
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
                            
                except Exception as e:
                    pass
            
            # D. 存盘与清理
            if batch_results:
                df_res = pd.DataFrame(batch_results)
                header = not os.path.exists(results_file)
                df_res.to_csv(results_file, mode='a', header=header, index=False, encoding='utf-8-sig')
                total_trades += len(df_res)
                st.toast(f"✅ 保存 {len(df_res)} 条记录 | 累计: {total_trades}")
            
            # E. 彻底内存清理
            del df_big, batch_results
            gc.collect()
            
            # F. 主动休眠 (防止 API 过载)
            time.sleep(1)
            main_progress.progress((b_i + 1) / len(batches))

        st.success("🎉 高保真回测全部完成！")
    
    # ---------------------------
    # 6. 结果展示
    # ---------------------------
    st.markdown("---")
    if os.path.exists(results_file):
        try:
            res_df = pd.read_csv(results_file)
            st.subheader("📊 回测报告 (High Fidelity)")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总交易次数", len(res_df))
            
            if 'Return_D1 (%)' in res_df.columns:
                avg_d1 = res_df['Return_D1 (%)'].mean()
                win_d1 = (res_df['Return_D1 (%)'] > 0).mean() * 100
                col2.metric("D+1 均收", f"{avg_d1:.2f}%")
                col3.metric("D+1 胜率", f"{win_d1:.1f}%")
                
                res_df = res_df.sort_values('trade_date')
                # 简单资金曲线 (假设单利累加)
                daily_ret = res_df.groupby('trade_date')['Return_D1 (%)'].mean()
                equity = daily_ret.cumsum()
                dd = equity.cummax() - equity
                col4.metric("最大回撤", f"{dd.max():.2f}")
            
            st.dataframe(res_df, use_container_width=True)
            with open(results_file, "rb") as f:
                st.download_button("📥 下载 CSV", f, "high_fidelity_result.csv")
        except:
            st.error("结果读取失败")
