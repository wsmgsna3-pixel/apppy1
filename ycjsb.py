import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="三日成妖·双模实战系统", layout="wide")

st.title("🐉 三日成妖·双模实战系统 (含断点续传)")

# ==========================================
# 2. 侧边栏设置
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    # --- 模式选择 ---
    mode = st.radio("请选择功能模式", ["📡 实盘选股 (找明天买点)", "📊 历史回测 (验证胜率)"])
    
    st.subheader("🎯 严选标准")
    vol_mul = st.slider("量能倍数 (潜伏期N倍)", 2.0, 5.0, 3.0, 0.5)
    
    if mode == "📊 历史回测 (验证胜率)":
        st.subheader("📅 回测区间")
        # 默认结束日期为今天
        default_end = datetime.now().date()
        end_date_input = st.date_input("回测结束日期", default_end)
        days_back = st.slider("回测过去多少天?", 10, 100, 30, 10)
        
        st.info("✅ 已开启断点续传：结果将实时保存到 `backtest_results.csv`。重新运行会自动跳过已测日期。")
        
    else:
        st.subheader("📅 扫描设置")
        scan_date_input = st.date_input("扫描基准日 (Day 3)", datetime.now().date())
        st.caption("通常选'今天'或'昨天'收盘后")

    run_btn = st.button("🚀 开始执行")

# ==========================================
# 3. 核心工具函数
# ==========================================
@st.cache_data(persist="disk", show_spinner=False)
def get_trade_cal(token, start_date, end_date):
    """获取交易日历"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
        return df['cal_date'].tolist()
    except:
        return []

@st.cache_data(persist="disk", show_spinner=False)
def get_daily_snapshot_filtered(token, date_str):
    """获取某日全市场【严选池】股票"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # 获取行情 + 基础信息
        df_daily = pro.daily(trade_date=date_str)
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,list_date')
        
        if df_daily.empty or df_basic.empty: return pd.DataFrame()
        
        df = pd.merge(df_daily, df_basic, on='ts_code')
        
        # === 严选漏斗 ===
        # 1. 剔除 ST
        df = df[~df['name'].str.contains('ST')]
        # 2. 剔除 北交所
        df = df[~df['ts_code'].str.contains('BJ')]
        df = df[~df['market'].str.contains('北交')]
        # 3. 剔除 次新 (上市<60天)
        limit_date = (datetime.strptime(date_str, '%Y%m%d') - timedelta(days=60)).strftime('%Y%m%d')
        df = df[df['list_date'] < limit_date]
        # 4. 价格 >= 10元
        df = df[df['close'] >= 10.0]
        # 5. 成交额 > 5000万
        df = df[df['amount'] > 50000]
        
        return df
    except:
        return pd.DataFrame()

@st.cache_data(persist="disk", show_spinner=False)
def get_history_data(token, code, end_date, lookback=80):
    """获取历史数据 (含潜伏期)"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        start_dt = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=lookback * 1.5 + 20)
        df = pro.daily(ts_code=code, start_date=start_dt.strftime('%Y%m%d'), end_date=end_date)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(persist="disk", show_spinner=False)
def get_future_data(token, code, start_date, days=15):
    """获取未来数据 (用于回测验证)"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = start_dt + timedelta(days=days * 2 + 10)
        df = pro.daily(ts_code=code, start_date=start_date, end_date=end_dt.strftime('%Y%m%d'))
        return df.sort_values('trade_date').reset_index(drop=True)
    except:
        return pd.DataFrame()

# ==========================================
# 4. 核心信号逻辑 (通用)
# ==========================================
def check_signal_logic(df_hist, code, market_type, vol_multiplier):
    """判断是否三连爆"""
    if len(df_hist) < 63: return False, 0.0, 0.0, 0.0
    
    # 倒序: 0是最新(信号日)
    df_hist = df_hist.sort_values('trade_date', ascending=False).reset_index(drop=True)
    
    df_burst = df_hist.iloc[0:3]
    df_latent = df_hist.iloc[3:63]
    
    latent_vol = df_latent['vol'].mean()
    if latent_vol == 0: return False, 0.0, 0.0, 0.0
    
    burst_vol = df_burst['vol'].mean()
    
    # 1. 量能判定
    if burst_vol < latent_vol * vol_multiplier: return False, 0.0, 0.0, 0.0
    
    # 2. 涨幅判定
    is_startup = False
    if '300' in code or '688' in code or '创业' in str(market_type) or '科创' in str(market_type):
        is_startup = True
    threshold = 20 if is_startup else 12
    
    p_start = df_burst.iloc[-1]['open']
    p_end = df_burst.iloc[0]['close']
    cum_rise = (p_end - p_start) / p_start * 100
    
    if cum_rise < threshold: return False, 0.0, 0.0, 0.0
    
    # 3. 形态判定
    if df_burst.iloc[-1]['pct_chg'] < 5: return False, 0.0, 0.0, 0.0 # Day1大阳
    if p_end <= df_burst.iloc[-1]['close']: return False, 0.0, 0.0, 0.0 # 重心上移
    
    # 返回: 信号有效, 累计涨幅, 潜伏均量, 爆发均量
    return True, cum_rise, latent_vol, burst_vol

# ==========================================
# 5. 模式 A: 实盘选股 (Signal Scanner)
# ==========================================
def run_signal_scanner():
    if not user_token:
        st.error("请输入Token")
        return
        
    d_str = scan_date_input.strftime('%Y%m%d')
    st.info(f"🔎 正在扫描 {d_str} (Day 3) 的三连爆信号...")
    st.caption("符合条件的股票，建议明日 (Day 4) 关注竞价开盘情况。")
    
    # 1. 获取当日池
    df_pool = get_daily_snapshot_filtered(user_token, d_str)
    if df_pool.empty:
        st.warning(f"{d_str} 当日无数据 (可能是非交易日或未收盘)")
        return
        
    results = []
    progress = st.progress(0)
    
    total = len(df_pool)
    for i, (_, row) in enumerate(df_pool.iterrows()):
        progress.progress((i+1)/total)
        
        code = row['ts_code']
        df_hist = get_history_data(user_token, code, d_str)
        is_valid, rise, l_vol, b_vol = check_signal_logic(df_hist, code, row['market'], vol_mul)
        
        if is_valid:
            results.append({
                '代码': code,
                '名称': row['name'],
                '板块': row['market'],
                '3日涨幅(%)': round(rise, 2),
                '量能倍数': round(b_vol/l_vol, 1),
                'Day3收盘': row['close'],
                '建议': '明日关注低吸'
            })
            
    progress.empty()
    
    if results:
        st.success(f"🔥 发现 {len(results)} 只潜在妖股！")
        st.dataframe(pd.DataFrame(results).sort_values('3日涨幅(%)', ascending=False))
    else:
        st.warning("今日未发现符合严选条件的股票。")

# ==========================================
# 6. 模式 B: 断点续传回测 (Backtest)
# ==========================================
def run_backtest_resume():
    if not user_token:
        st.error("请输入Token")
        return

    # 1. 确定日期范围
    end_str = end_date_input.strftime('%Y%m%d')
    start_dt_est = end_date_input - timedelta(days=days_back * 2 + 15)
    cal_dates = get_trade_cal(user_token, start_dt_est.strftime('%Y%m%d'), end_str)
    
    # 信号日区间 (预留最后10天给 D+10)
    if len(cal_dates) < days_back + 10:
        st.error("日期范围太短")
        return
    signal_dates = cal_dates[-(days_back + 10) : -10]
    
    # === 断点续传逻辑 ===
    csv_file = 'backtest_results.csv'
    processed_dates = set()
    
    # 检查是否有历史记录
    if os.path.exists(csv_file):
        try:
            df_exist = pd.read_csv(csv_file)
            if '信号日' in df_exist.columns:
                # 记录所有已经跑出结果的日期
                # 注意：如果某天跑了但没结果，这里可能没有记录，会导致重跑(这是安全的)
                # 为了更严谨，我们可以用另一个文件记录"已扫描日期"，但这里简化处理：
                # 假设只要结果文件里有这个日期，就算跑过了。
                processed_dates = set(df_exist['信号日'].astype(str).tolist())
                st.info(f"📂 检测到历史存档，包含 {len(df_exist)} 条交易记录。将自动跳过已处理日期。")
        except:
            pass
            
    st.write(f"⏳ 计划回测区间: {signal_dates[0]} 至 {signal_dates[-1]}")
    
    # 2. 循环回测
    progress = st.progress(0)
    status = st.empty()
    
    total_dates = len(signal_dates)
    
    for i, date in enumerate(signal_dates):
        progress.progress((i+1)/total_dates)
        
        # 跳过已处理
        if str(date) in processed_dates:
            status.text(f"⏭️ 跳过已回测日期: {date}")
            continue
            
        status.text(f"⚡ 正在回测: {date} ...")
        
        # A. 获取当日池
        df_day_pool = get_daily_snapshot_filtered(user_token, date)
        if df_day_pool.empty: continue
        
        daily_results = []
        
        # B. 扫描
        for _, row in df_day_pool.iterrows():
            code = row['ts_code']
            df_hist = get_history_data(user_token, code, date)
            is_valid, rise, _, _ = check_signal_logic(df_hist, code, row['market'], vol_mul)
            
            if is_valid:
                # C. 模拟交易
                try:
                    curr_idx = cal_dates.index(date)
                    d1_date = cal_dates[curr_idx + 1]
                except:
                    continue
                    
                df_future = get_future_data(user_token, code, d1_date, days=12)
                if df_future.empty: continue
                
                # 推演
                d1 = df_future.iloc[0]
                # 风控: 低开 < -5%
                open_pct = (d1['open'] - d1['pre_close']) / d1['pre_close'] * 100
                if open_pct < -5: continue 
                
                buy_price = d1['open']
                stop_price = buy_price * 0.90
                
                trade = {
                    '信号日': date,
                    '代码': code,
                    '名称': row['name'],
                    '3日涨幅': round(rise, 2),
                    '买入价': buy_price,
                    '状态': '持有'
                }
                
                triggered = False
                for di in range(min(10, len(df_future))):
                    row_f = df_future.iloc[di]
                    key = f"D+{di+1}"
                    
                    if not triggered:
                        if row_f['low'] <= stop_price:
                            triggered = True
                            trade['状态'] = '止损'
                            ret = -10.0
                        else:
                            ret = (row_f['close'] - buy_price) / buy_price * 100
                    else:
                        ret = -10.0
                        
                    if di+1 in [1,3,5,7,10]:
                        trade[key] = round(ret, 2)
                        
                daily_results.append(trade)
        
        # D. 实时写入 CSV (断点续传核心)
        if daily_results:
            df_new = pd.DataFrame(daily_results)
            # 如果文件不存在，写表头；如果存在，不写表头，追加模式
            header = not os.path.exists(csv_file)
            df_new.to_csv(csv_file, mode='a', header=header, index=False, encoding='utf-8-sig')
            
            # 同时也添加到 processed_dates 防止本次运行重复
            processed_dates.add(str(date))
            
    progress.empty()
    status.empty()
    
    # 3. 最终展示
    if os.path.exists(csv_file):
        df_final = pd.read_csv(csv_file)
        st.success(f"🎉 回测全部完成！总计交易记录: {len(df_final)} 条")
        
        # 统计
        cols = st.columns(5)
        days = ['D+1', 'D+3', 'D+5', 'D+7', 'D+10']
        for idx, d in enumerate(days):
            if d in df_final.columns:
                win = len(df_final[df_final[d]>0]) / len(df_final) * 100
                avg = df_final[d].mean()
                cols[idx].metric(f"{d} 胜率", f"{win:.1f}%")
                cols[idx].metric(f"{d} 均收", f"{avg:.2f}%")
                
        st.dataframe(df_final.sort_values('信号日', ascending=False))
    else:
        st.warning("未发现符合条件的交易。")

# ==========================================
# 7. 入口
# ==========================================
if run_btn:
    if mode == "📡 实盘选股 (找明天买点)":
        run_signal_scanner()
    else:
        run_backtest_resume()
