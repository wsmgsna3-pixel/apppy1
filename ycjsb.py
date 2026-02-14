import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="潜龙·狙击手", layout="wide")
st.title("🎯 潜龙·狙击手 (VR>1.6 极品吸筹版)")
st.markdown("""
**本次目标：120天筛选 < 100 只极品 (日均 < 1 只)**
1.  **极度吸筹**：VR (阳量/阴量) > **1.6** (数据回测显示的赢家底线)。
2.  **攻击形态**：RSI > **55** (拒绝磨叽，只做主升)。
3.  **共振确认**：板块涨幅 > **1.2%** (确保风口)。
4.  **宁缺毋滥**：虽然显示 Top 5，但通常每天只有 0-2 只入围。
""")

# ==========================================
# 2. 核心数据引擎
# ==========================================
@st.cache_data(persist="disk", show_spinner=False)
def get_trade_cal(token, start_date, end_date):
    ts.set_token(token)
    pro = ts.pro_api()
    for attempt in range(3):
        try:
            df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
            if not df.empty:
                return sorted(df['cal_date'].tolist())
            time.sleep(0.5)
        except:
            time.sleep(1)
    return []

@st.cache_data(persist="disk", show_spinner=False)
def fetch_all_market_data_by_date(token, date_list):
    ts.set_token(token)
    pro = ts.pro_api()
    data_list = []
    total = len(date_list)
    bar = st.progress(0, text="正在同步全市场数据...")
    
    for i, date in enumerate(date_list):
        try:
            time.sleep(0.05)
            df = pro.daily(trade_date=date)
            if not df.empty:
                df = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount', 'pct_chg']]
                data_list.append(df)
        except:
            time.sleep(0.5)
        if (i+1) % 10 == 0:
            bar.progress((i+1)/total, text=f"加载进度: {i+1}/{total}")
            
    bar.empty()
    if not data_list: return pd.DataFrame()
    full_df = pd.concat(data_list)
    full_df = full_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    return full_df

@st.cache_data(persist="disk", show_spinner=False)
def get_stock_basics(token):
    ts.set_token(token)
    pro = ts.pro_api()
    for _ in range(3):
        try:
            time.sleep(0.5)
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,industry,list_date')
            if not df.empty:
                df = df[~df['name'].str.contains('ST')]
                df = df[~df['market'].str.contains('北交')]
                df = df[~df['ts_code'].str.contains('BJ')]
                return df
        except: time.sleep(1)
    return pd.DataFrame()

# ==========================================
# 3. 核心计算：狙击逻辑
# ==========================================
def calculate_sector_heat(df_daily, df_basic):
    if 'industry' not in df_daily.columns:
        df_merged = pd.merge(df_daily, df_basic[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df_merged = df_daily.copy()
    
    valid_df = df_merged[df_merged['pct_chg'] != 0]
    sector_stats = valid_df.groupby(['trade_date', 'industry'])['pct_chg'].mean().reset_index()
    sector_stats.rename(columns={'pct_chg': 'sector_pct'}, inplace=True)
    df_final = pd.merge(df_merged, sector_stats, on=['trade_date', 'industry'], how='left')
    return df_final

def calculate_strategy(df, vol_mul, box_min, box_max, vr_threshold, rsi_min, rsi_max):
    """
    计算所有信号 (狙击手核心)
    """
    # 1. 基础指标
    df['high_60'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(1).rolling(window=60).max())
    df['low_60'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(1).rolling(window=60).min())
    df['vol_60'] = df.groupby('ts_code')['vol'].transform(lambda x: x.shift(1).rolling(window=60).mean())
    df['box_amplitude'] = (df['high_60'] - df['low_60']) / df['low_60']
    
    # 2. 吸筹指标 (VR)
    df['vol_up'] = np.where(df['pct_chg'] > 0, df['vol'], 0)
    df['vol_down'] = np.where(df['pct_chg'] <= 0, df['vol'], 0)
    
    df['sum_vol_up'] = df.groupby('ts_code')['vol_up'].transform(lambda x: x.rolling(window=60).sum())
    df['sum_vol_down'] = df.groupby('ts_code')['vol_down'].transform(lambda x: x.rolling(window=60).sum())
    
    df['accumulation_ratio'] = df['sum_vol_up'] / (df['sum_vol_down'] + 1)
    
    # 3. RSI 指标 (近似算法)
    df['up_move'] = np.where(df['pct_chg'] > 0, df['pct_chg'], 0)
    df['down_move'] = np.where(df['pct_chg'] < 0, abs(df['pct_chg']), 0)
    avg_up = df.groupby('ts_code')['up_move'].transform(lambda x: x.rolling(6).mean())
    avg_down = df.groupby('ts_code')['down_move'].transform(lambda x: x.rolling(6).mean())
    df['rsi_6'] = 100 * avg_up / (avg_up + avg_down + 0.0001)
    
    # 4. 信号判定
    # A. 振幅区间 (10% ~ 40%)
    cond_box = (df['box_amplitude'] > (box_min/100)) & (df['box_amplitude'] < (box_max/100))
    
    # B. 价格突破 (创60日新高)
    cond_break = df['close'] > df['high_60']
    
    # C. 量能突破
    cond_vol = df['vol'] > (df['vol_60'] * vol_mul)
    
    # D. 流动性筛选
    cond_mv = (df['amount'] > 50000) & (df['amount'] < 5000000)
    
    # E. 狙击筛选 (极高门槛)
    cond_acc = df['accumulation_ratio'] > vr_threshold # > 1.6
    cond_rsi = (df['rsi_6'] > rsi_min) & (df['rsi_6'] < rsi_max) # 55 - 85
    
    df['is_signal_base'] = cond_box & cond_break & cond_vol & cond_mv & cond_acc & cond_rsi
    
    return df

def calculate_score(row):
    score = 60
    
    # 吸筹分 (权重最大)
    acc = row['accumulation_ratio']
    # 既然门槛已经是 1.6，这里主要区分极品
    if acc > 2.0: score += 30 # 超强吸筹
    elif acc > 1.8: score += 20
    else: score += 10
    
    # 振幅分 (偏好适中)
    amp = row['box_amplitude'] * 100
    if 20 <= amp <= 35: score += 15
    
    # 板块分
    if row['sector_pct'] > 0:
        score += min(row['sector_pct'] * 10, 30) # 放大板块权重
        
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ 狙击手参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("数据回溯天数", 60, 300, 120)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("🎯 狙击镜")
    col1, col2 = st.columns(2)
    box_min = col1.number_input("振幅下限%", 5, 20, 10)
    box_max = col2.number_input("振幅上限%", 30, 60, 40)
    
    # 默认值大幅提高
    vr_threshold = st.slider("VR 吸筹门槛 (阳/阴)", 1.0, 3.0, 1.6, 0.1, help="回测显示赢家均值>1.7")
    
    rsi_min = st.number_input("RSI下限", 0, 100, 55, help="55以上才算进入攻击区")
    rsi_max = st.number_input("RSI上限", 0, 100, 85)
    
    vol_mul = st.slider("突破量能倍数", 1.5, 5.0, 1.8, 0.1)
    sector_min_rise = st.slider("板块最低涨幅 (%)", 0.0, 3.0, 1.2, 0.1)
    
    top_n = st.number_input("每日优选 (Top N)", 1, 50, 5)
    
    run_btn = st.button("🚀 启动狙击回测")

def run_analysis():
    if not user_token:
        st.error("请先输入 Token")
        return

    # 1. 准备数据
    end_str = end_date_input.strftime('%Y%m%d')
    start_dt = end_date_input - timedelta(days=days_back * 1.5 + 80)
    
    cal_dates = get_trade_cal(user_token, start_dt.strftime('%Y%m%d'), end_str)
    if not cal_dates:
        st.error("获取日历失败")
        return
        
    df_all = fetch_all_market_data_by_date(user_token, cal_dates)
    if df_all.empty:
        st.error("数据加载失败")
        return
    st.success(f"✅ K线数据就绪: {len(df_all):,} 条")

    # 2. 基础信息
    df_basic = get_stock_basics(user_token)
    if df_basic.empty:
        st.error("无法获取行业数据。")
        return
        
    # 3. 计算板块热度
    with st.spinner("正在计算板块热度..."):
        df_sector = calculate_sector_heat(df_all, df_basic)
    
    # 4. 计算策略信号 (含吸筹逻辑)
    with st.spinner("正在执行狙击筛选..."):
        df_calc = calculate_strategy(df_sector, vol_mul, box_min, box_max, vr_threshold, rsi_min, rsi_max)
        
    # 5. 漏斗诊断
    st.markdown("### 🎯 狙击漏斗诊断")
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    st.write(f"⚪ 样本总数: {len(df_window):,} 条")
    
    # 基础筛选
    c_base = (df_window['amount'] > 50000) & (df_window['amount'] < 5000000) & \
             (df_window['box_amplitude'] > box_min/100) & (df_window['box_amplitude'] < box_max/100)
    n_base = len(df_window[c_base])
    st.write(f"1️⃣ 基础形态筛选: {n_base:,}")
    
    # 吸筹筛选
    c_acc = df_window['accumulation_ratio'] > vr_threshold
    n_acc = len(df_window[c_base & c_acc])
    st.write(f"2️⃣ 极品吸筹 (VR > {vr_threshold}): {n_acc:,} (大幅过滤)")
    
    # 最终信号
    df_window['is_signal'] = df_window['is_signal_base'] & (df_window['sector_pct'] > sector_min_rise)
    df_signals = df_window[df_window['is_signal']].copy()
    st.write(f"3️⃣ 最终狙击点 (含共振): **{len(df_signals)}** 个")
    
    if df_signals.empty:
        st.warning("无符合条件的信号。尝试降低VR门槛。")
        return

    # 6. 评分与 Top N
    df_signals['潜龙分'] = df_signals.apply(calculate_score, axis=1)
    df_signals = df_signals.sort_values(['trade_date', '潜龙分'], ascending=[True, False])
    df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
    
    df_top = df_signals[df_signals['排名'] <= top_n].copy()
    
    # 7. 收益回测
    price_lookup = df_calc[['ts_code', 'trade_date', 'open', 'close', 'low']].set_index(['ts_code', 'trade_date'])
    trades = []
    
    progress = st.progress(0)
    total_sig = len(df_top)
    
    for i, row in enumerate(df_top.itertuples()):
        progress.progress((i+1)/total_sig)
        
        signal_date = row.trade_date
        code = row.ts_code
        
        try:
            curr_idx = cal_dates.index(signal_date)
            future_dates = cal_dates[curr_idx+1 : curr_idx+11]
        except: continue
            
        if not future_dates:
            trades.append({
                '信号日': signal_date, '代码': code, '名称': row.name, '排名': row.排名,
                '行业': row.industry, '板块涨幅': f"{row.sector_pct:.1f}%",
                '吸筹比率': f"{row.accumulation_ratio:.2f}",
                '潜龙分': row.潜龙分, '状态': '等待开盘'
            })
            continue
            
        d1_date = future_dates[0]
        if (code, d1_date) not in price_lookup.index: continue
        d1_data = price_lookup.loc[(code, d1_date)]
        
        # 风控
        open_pct = (d1_data['open'] - d1_data.get('pre_close', row.close)) / row.close
        if open_pct < -0.05: continue
            
        buy_price = d1_data['open']
        stop_price = buy_price * 0.90
        
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, '排名': row.排名,
            '行业': row.industry, '板块涨幅': f"{row.sector_pct:.1f}%",
            '吸筹比率': f"{row.accumulation_ratio:.2f}",
            '潜龙分': row.潜龙分, '买入价': buy_price, '状态': '持有'
        }
        
        triggered = False
        for n, f_date in enumerate(future_dates):
            if (code, f_date) not in price_lookup.index: break
            f_data = price_lookup.loc[(code, f_date)]
            
            day_label = f"D+{n+1}"
            
            if not triggered:
                if f_data['low'] <= stop_price:
                    triggered = True
                    trade['状态'] = '止损'
                    trade[day_label] = -10.0
                else:
                    ret = (f_data['close'] - buy_price) / buy_price * 100
                    trade[day_label] = round(ret, 2)
            else:
                trade[day_label] = -10.0
        
        trades.append(trade)
        
    progress.empty()
    
    if trades:
        df_res = pd.DataFrame(trades)
        
        st.markdown(f"### 📊 狙击回测结果 (Top {top_n})")
        cols = st.columns(5)
        days = ['D+1', 'D+3', 'D+5', 'D+7', 'D+10']
        
        for idx, d in enumerate(days):
            if d in df_res.columns:
                valid_data = df_res[pd.to_numeric(df_res[d], errors='coerce').notna()]
                if not valid_data.empty:
                    wins = valid_data[valid_data[d] > 0]
                    win_rate = len(wins) / len(valid_data) * 100
                    avg_ret = valid_data[d].mean()
                    cols[idx].metric(f"{d} 胜率", f"{win_rate:.1f}%")
                    cols[idx].metric(f"{d} 均收", f"{avg_ret:.2f}%")
        
        st.markdown("### 🏆 狙击潜龙榜")
        display_cols = ['信号日', '排名', '代码', '名称', '行业', '板块涨幅', '吸筹比率', '潜龙分', '状态'] + \
                       [d for d in days if d in df_res.columns]
        
        st.dataframe(
            df_res[display_cols].sort_values(['信号日', '排名'], ascending=[False, True]),
            use_container_width=True,
            height=600
        )
    else:
        st.warning("无有效交易。")

if run_btn:
    run_analysis()
