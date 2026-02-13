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
st.set_page_config(page_title="三日成妖·全周期实战系统", layout="wide")

st.title("🐉 三日成妖·实战系统 (2026 稳定修复版)")
st.markdown("""
**策略核心：**
1. **严选池**：剔除北交所、<10元、<5000万成交、ST股。
2. **三连爆**：连续3天放量 (潜伏均量的N倍) + 重心上移。
3. **分板块**：主板3日涨幅>12%，双创>20%。
4. **风控**：D+1低开<-5%不买，-10%止损。
""")

# ==========================================
# 2. 侧边栏设置
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    user_token = st.text_input("Tushare Token:", type="password")
    
    # 模式选择
    mode = st.radio("功能模式", ["📡 实盘选股 (找明天买点)", "📊 历史回测 (验证胜率)"])
    
    st.subheader("🎯 筛选标准")
    vol_mul = st.slider("量能倍数", 2.0, 5.0, 3.0, 0.5, help="爆发期成交量是潜伏期的多少倍")
    
    if mode == "📊 历史回测 (验证胜率)":
        st.subheader("📅 回测设置")
        # 默认结束日期设为今天
        end_date_input = st.date_input("回测结束日期", datetime.now().date())
        days_back = st.slider("回测天数", 10, 100, 30, 5)
        st.info("✅ 已开启断点续传：结果实时保存至 `backtest_results.csv`")
    else:
        st.subheader("📅 选股设置")
        scan_date_input = st.date_input("扫描基准日", datetime.now().date())
        st.caption("选'今天'：找明天能买的。选'昨天'：复盘昨天的信号。")

    run_btn = st.button("🚀 立即运行")

# ==========================================
# 3. 核心工具函数 (增加容错)
# ==========================================
@st.cache_data(persist="disk", show_spinner=False)
def get_trade_cal(token, start_date, end_date):
    """获取交易日历 (带容错)"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # exchange='SSE' 确保只获取上交所日历，防止全空
        df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
        if df.empty: return []
        return df['cal_date'].tolist()
    except:
        return []

@st.cache_data(persist="disk", show_spinner=False)
def get_daily_snapshot_filtered(token, date_str):
    """获取某日全市场【严选池】股票"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 1. 获取行情
        df_daily = pro.daily(trade_date=date_str)
        # 2. 获取基础信息
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,list_date')
        
        if df_daily.empty or df_basic.empty: return pd.DataFrame()
        
        df = pd.merge(df_daily, df_basic, on='ts_code')
        
        # === 严选漏斗 ===
        # 剔除 ST
        df = df[~df['name'].str.contains('ST')]
        # 剔除 北交所
        df = df[~df['ts_code'].str.contains('BJ')]
        df = df[~df['market'].str.contains('北交')]
        # 剔除 次新 (上市<60天)
        limit_date = (datetime.strptime(date_str, '%Y%m%d') - timedelta(days=60)).strftime('%Y%m%d')
        df = df[df['list_date'] < limit_date]
        # 价格 >= 10元
        df = df[df['close'] >= 10.0]
        # 成交额 > 5000万
        df = df[df['amount'] > 50000]
        
        return df
    except:
        return pd.DataFrame()

@st.cache_data(persist="disk", show_spinner=False)
def get_history_data(token, code, end_date, lookback=100):
    """获取历史数据 (加大Lookback防止潜伏期不够)"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # 多取一些缓冲
        start_dt = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=lookback * 1.8 + 30)
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
# 4. 信号核心逻辑
# ==========================================
def check_signal_logic(df_hist, code, market_type, vol_multiplier):
    """判断是否三连爆"""
    # 至少需要 3(爆发) + 60(潜伏) = 63天数据
    if len(df_hist) < 63: return False, 0.0, 0.0, 0.0
    
    # 倒序: 0是最新(信号日)
    df_hist = df_hist.sort_values('trade_date', ascending=False).reset_index(drop=True)
    
    df_burst = df_hist.iloc[0:3]   # 最近3天
    df_latent = df_hist.iloc[3:63] # 前60天潜伏
    
    latent_vol = df_latent['vol'].mean()
    if latent_vol == 0: return False, 0.0, 0.0, 0.0
    
    burst_vol = df_burst['vol'].mean()
    
    # 1. 量能判定 (3倍)
    if burst_vol < latent_vol * vol_multiplier: return False, 0.0, 0.0, 0.0
    
    # 2. 涨幅判定 (分板块)
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
    
    return True, cum_rise, latent_vol, burst_vol

# ==========================================
# 5. 实盘选股模式
# ==========================================
def run_scanner():
    d_str = scan_date_input.strftime('%Y%m%d')
    st.info(f"🔎 正在扫描 {d_str} 的三连爆信号...")
    
    df_pool = get_daily_snapshot_filtered(user_token, d_str)
    if df_pool.empty:
        st.error("数据获取失败，请检查日期或Token")
        return
        
    results = []
    progress = st.progress(0)
    total = len(df_pool)
    
    for i, (_, row) in enumerate(df_pool.iterrows()):
        progress.progress((i+1)/total)
        code = row['ts_code']
        # 注意：选股模式下，历史数据截止到扫描日
        df_hist = get_history_data(user_token, code, d_str)
        is_valid, rise, l_vol, b_vol = check_signal_logic(df_hist, code, row['market'], vol_mul)
        
        if is_valid:
            results.append({
                '代码': code,
                '名称': row['name'],
                '板块': row['market'],
                '3日涨幅(%)': round(rise, 2),
                '量能倍数': round(b_vol/l_vol, 1),
                'Day3收盘': row['close']
            })
            
    progress.empty()
    if results:
        st.success(f"🔥 发现 {len(results)} 只标的！")
        st.dataframe(pd.DataFrame(results).sort_values('3日涨幅(%)', ascending=False))
    else:
        st.warning("今日无符合严选条件的股票。")

# ==========================================
# 6. 回测模式 (含修复逻辑)
# ==========================================
def run_backtest():
    # 1. 计算日期范围 (加大缓冲，防止日历为空)
    end_str = end_date_input.strftime('%Y%m%d')
    # 缓冲系数加大到 4 倍，确保覆盖长假
    start_dt_est = end_date_input - timedelta(days=days_back * 4 + 30) 
    
    cal_dates = get_trade_cal(user_token, start_dt_est.strftime('%Y%m%d'), end_str)
    
    if not cal_dates:
        st.error("无法获取交易日历，请检查Token或网络。")
        return
        
    # 逻辑修正：只要有日期就能测，不必非要满足 days_back 的数量
    # 我们取最后 days_back 天，但要预留 10 天给 D+10
    if len(cal_dates) < 12:
        st.error("日期太短，无法进行回测（至少需要12个交易日）")
        return
        
    # 截取有效区间：[倒数第 N+10 天] 到 [倒数第 10 天]
    # 比如：今天 2月1日，我们要测过去，那信号日只能截止到 1月20日，因为1月20日的D+10是今天
    valid_end_index = -10
    valid_start_index = -(days_back + 10)
    
    # 动态调整索引，防止越界
    if abs(valid_start_index) > len(cal_dates):
        valid_start_index = 0 # 如果天数不够，就从头测
        
    signal_dates = cal_dates[valid_start_index : valid_end_index]
    
    if not signal_dates:
        st.error("有效信号日为空，请调整回测结束日期（不要选未来日期）。")
        return
        
    # === 断点续传准备 ===
    csv_file = 'backtest_results.csv'
    processed_dates = set()
    if os.path.exists(csv_file):
        try:
            df_ex = pd.read_csv(csv_file)
            if '信号日' in df_ex.columns:
                processed_dates = set(df_ex['信号日'].astype(str).tolist())
                st.info(f"📂 已读取存档，跳过 {len(processed_dates)} 个已测信号日。")
        except: pass

    st.write(f"⏳ 回测区间: {signal_dates[0]} 至 {signal_dates[-1]} (共 {len(signal_dates)} 天)")
    
    progress = st.progress(0)
    status = st.empty()
    total_dates = len(signal_dates)
    
    # 2. 循环每一天
    for i, date in enumerate(signal_dates):
        progress.progress((i+1)/total_dates)
        
        if str(date) in processed_dates:
            status.text(f"⏭️ 跳过: {date}")
            continue
            
        status.text(f"⚡ 回测中: {date} ...")
        
        # A. 获取当日池
        df_day = get_daily_snapshot_filtered(user_token, date)
        if df_day.empty: continue
        
        daily_trades = []
        
        for _, row in df_day.iterrows():
            code = row['ts_code']
            # 获取历史数据判断信号
            df_hist = get_history_data(user_token, code, date)
            is_valid, rise, _, _ = check_signal_logic(df_hist, code, row['market'], vol_mul)
            
            if is_valid:
                # B. 信号触发，开始模拟买入
                # 获取 D+1 日期
                try:
                    curr_idx = cal_dates.index(date)
                    d1_date = cal_dates[curr_idx + 1]
                except: continue
                
                # 获取未来 15 天数据
                df_future = get_future_data(user_token, code, d1_date, days=15)
                if df_future.empty: continue
                
                # 风控检测
                d1 = df_future.iloc[0]
                open_pct = (d1['open'] - d1['pre_close']) / d1['pre_close'] * 100
                
                if open_pct >= -5: # 低开没超过 -5%，买入
                    buy_price = d1['open']
                    stop_price = buy_price * 0.90
                    
                    trade = {
                        '信号日': date,
                        '代码': code,
                        '名称': row['name'],
                        '3日涨幅': round(rise, 1),
                        '买入价': buy_price,
                        '状态': '持有'
                    }
                    
                    triggered = False
                    # 推演未来 10 天
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
                            
                    daily_trades.append(trade)
        
        # C. 存盘
        if daily_trades:
            df_new = pd.DataFrame(daily_trades)
            header = not os.path.exists(csv_file)
            df_new.to_csv(csv_file, mode='a', header=header, index=False, encoding='utf-8-sig')
            processed_dates.add(str(date)) # 只有真正写入了才标记跳过(可选逻辑，这里简单处理)
        
        # 即使当天没有交易，也记录一下避免死循环(可选，这里主要靠文件存在判断)
            
    progress.empty()
    status.empty()
    
    # 3. 结果展示
    if os.path.exists(csv_file):
        df_res = pd.read_csv(csv_file)
        st.success(f"🎉 回测完成！累计交易: {len(df_res)} 笔")
        
        cols = st.columns(5)
        days = ['D+1', 'D+3', 'D+5', 'D+7', 'D+10']
        for idx, d in enumerate(days):
            if d in df_res.columns:
                win = len(df_res[df_res[d]>0]) / len(df_res) * 100
                avg = df_res[d].mean()
                cols[idx].metric(f"{d} 胜率", f"{win:.1f}%")
                cols[idx].metric(f"{d} 均收", f"{avg:.2f}%")
                
        st.dataframe(df_res.sort_values('信号日', ascending=False))
    else:
        st.warning("回测区间内未发现符合条件的交易。")

# ==========================================
# 7. 入口
# ==========================================
if run_btn:
    if not user_token:
        st.error("请先输入 Token")
    else:
        if mode == "📡 实盘选股 (找明天买点)":
            run_scanner()
        else:
            run_backtest()
