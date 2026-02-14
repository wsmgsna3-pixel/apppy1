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
st.set_page_config(page_title="潜龙 V16·上帝指纹", layout="wide")
st.title("🐉 潜龙 V16·上帝指纹 (严选等距+低位起爆)")
st.markdown("""
**策略核心：极度苛刻的"完美图形"筛选**
1.  **绝对等距**：均线间距误差 < **1.5倍** (从 2.5 收紧到 1.5，真正的仪仗队)。
2.  **攻击角度**：MA5 必须有明显的上攻角度 (拒绝蠕动)。
3.  **贴线起爆**：股价距离 **MA10 < 5%** (拒绝追高，只做刚启动或刚回踩)。
4.  **趋势共振**：四线多头排列且全部向上。
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
            # 需要换手率过滤僵尸股
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv')
            
            if not df.empty and not df_basic.empty:
                df = pd.merge(df, df_basic, on='ts_code', how='left')
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
# 3. 核心计算
# ==========================================
def calculate_strategy(df):
    """
    V16 核心逻辑: 严选上帝指纹
    """
    # 1. 计算均线
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma30'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(30).mean())
    
    # 计算均线斜率 (归一化斜率: 每日涨幅百分比)
    # (Today - Yesterday) / Yesterday * 100
    df['ma5_slope'] = df.groupby('ts_code')['ma5'].pct_change() * 100
    df['ma10_slope'] = df.groupby('ts_code')['ma10'].pct_change() * 100
    df['ma20_slope'] = df.groupby('ts_code')['ma20'].pct_change() * 100
    df['ma30_slope'] = df.groupby('ts_code')['ma30'].pct_change() * 100
    
    # 2. 信号判定逻辑
    
    # A. 完美排列: Close > MA5 > MA10 > MA20 > MA30
    cond_order = (df['close'] > df['ma5']) & \
                 (df['ma5'] > df['ma10']) & \
                 (df['ma10'] > df['ma20']) & \
                 (df['ma20'] > df['ma30'])
    
    # B. 攻击角度: 
    # MA5 斜率 > 0.3% (约等于股价每天涨1%带动的斜率，拒绝横盘)
    # 所有均线必须向上
    cond_slope = (df['ma5_slope'] > 0.3) & \
                 (df['ma10_slope'] > 0) & \
                 (df['ma20_slope'] > 0) & \
                 (df['ma30_slope'] > 0)
    
    # C. 严选等距 (Strict Spacing)
    # 计算间距
    df['gap1'] = df['ma5'] - df['ma10']
    df['gap2'] = df['ma10'] - df['ma20']
    df['gap3'] = df['ma20'] - df['ma30']
    
    df['max_gap'] = df[['gap1', 'gap2', 'gap3']].max(axis=1)
    df['min_gap'] = df[['gap1', 'gap2', 'gap3']].min(axis=1)
    
    # 门槛：最大间距 / 最小间距 < 1.5 (极度均匀)
    cond_spacing = (df['max_gap'] / (df['min_gap'] + 0.0001)) < 1.5
    
    # D. 贴线起爆 (Low Risk)
    # 收盘价距离 MA10 不超过 5% (防止乖离过大接盘)
    # (Close - MA10) / MA10 < 0.05
    cond_low = (df['close'] - df['ma10']) / df['ma10'] < 0.05
    
    # E. 首日启动 (Yesterday NOT perfect)
    # 组合今日状态
    df['is_perfect'] = cond_order & cond_slope & cond_spacing & cond_low
    # 获取昨日状态
    df['prev_perfect'] = df.groupby('ts_code')['is_perfect'].shift(1).fillna(False)
    
    cond_start = df['is_perfect'] & (~df['prev_perfect'])
    
    # F. 基础过滤
    cond_basic = (df['turnover_rate'] > 1.0) 
    
    df['is_signal'] = cond_start & cond_basic
    
    return df

def calculate_score(row):
    # 评分逻辑：越均匀越好
    score = 60
    
    # 均匀度 (Ratio 越接近 1 越好)
    ratio = row['max_gap'] / (row['min_gap'] + 0.0001)
    if ratio < 1.2: score += 30
    elif ratio < 1.4: score += 20
    
    # 斜率越大越好 (攻击性)
    if row['ma5_slope'] > 0.8: score += 10
    
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V16 上帝指纹参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("🔥 筛选标准")
    top_n = st.number_input("每日优选 (Top N)", 1, 10, 2)
    
    run_btn = st.button("🚀 启动 V16 回测")

def run_analysis():
    if not user_token:
        st.error("请先输入 Token")
        return

    # 1. 准备数据
    end_str = end_date_input.strftime('%Y%m%d')
    start_dt = end_date_input - timedelta(days=days_back * 1.5 + 80)
    
    cal_dates = get_trade_cal(user_token, start_dt.strftime('%Y%m%d'), end_str)
    if not cal_dates: return
        
    df_all = fetch_all_market_data_by_date(user_token, cal_dates)
    if df_all.empty: return
    st.success(f"✅ K线数据就绪: {len(df_all):,} 条")

    # 2. 基础信息
    df_basic = get_stock_basics(user_token)
    if df_basic.empty: return
    
    # 合并名称行业
    if 'industry' not in df_all.columns:
        df_all = pd.merge(df_all, df_basic[['ts_code', 'industry', 'name']], on='ts_code', how='left')
        
    # 3. 计算
    with st.spinner("正在用显微镜寻找上帝指纹..."):
        df_calc = calculate_strategy(df_all)
        
    # 4. 结果
    st.markdown("### 🐉 V16 诊断 (严选版)")
    
    if df_calc.empty:
        st.warning("无信号。")
        return
        
    # 过滤时间窗
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    df_signals = df_window[df_window['is_signal']].copy()
    
    st.write(f"⚪ 捕获完美图形: **{len(df_signals)}** 个")
    
    if df_signals.empty:
        st.warning("严选标准下，近期无完美形态。")
        return

    # 5. 评分与 Top N
    df_signals['潜龙分'] = df_signals.apply(calculate_score, axis=1)
    df_signals = df_signals.sort_values(['trade_date', '潜龙分'], ascending=[True, False])
    
    # 每日取 Top N
    df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
    df_top = df_signals[df_signals['排名'] <= top_n].copy()
    
    # 6. 回测 (加入 MA10 止损逻辑)
    price_lookup = df_calc[['ts_code', 'trade_date', 'open', 'close', 'low', 'ma10']].set_index(['ts_code', 'trade_date'])
    trades = []
    
    progress = st.progress(0)
    total_sig = len(df_top)
    
    for i, row in enumerate(df_top.itertuples()):
        progress.progress((i+1)/total_sig)
        
        signal_date = row.trade_date
        code = row.ts_code
        
        try:
            curr_idx = cal_dates.index(signal_date)
            future_dates = cal_dates[curr_idx+1 : curr_idx+11] # 看10天
        except: continue
            
        if not future_dates: continue
            
        d1_date = future_dates[0]
        if (code, d1_date) not in price_lookup.index: continue
        d1_data = price_lookup.loc[(code, d1_date)]
        
        buy_price = d1_data['open']
        
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, 
            '行业': row.industry, 
            '均匀度': f"{row.max_gap / (row.min_gap+0.0001):.2f}",
            'MA5斜率': f"{row.ma5_slope:.2f}",
            '买入价': buy_price, '状态': '持有'
        }
        
        triggered = False
        
        for n, f_date in enumerate(future_dates):
            if (code, f_date) not in price_lookup.index: break
            f_data = price_lookup.loc[(code, f_date)]
            day_label = f"D+{n+1}"
            
            if not triggered:
                # 1. 硬止损
                curr_ret = (f_data['close'] - buy_price) / buy_price
                if curr_ret < -0.10:
                    triggered = True
                    trade[day_label] = -10.0
                    trade['状态'] = '止损'
                    continue
                
                # 2. 趋势止损: 收盘跌破 MA10
                if f_data['close'] < f_data['ma10']:
                    triggered = True
                    final_ret = (f_data['close'] - buy_price) / buy_price * 100
                    trade[day_label] = round(final_ret, 2)
                    trade['状态'] = '破线卖出'
                else:
                    # 继续持有
                    final_ret = (f_data['close'] - buy_price) / buy_price * 100
                    trade[day_label] = round(final_ret, 2)
            else:
                trade[day_label] = trade.get(f"D+{n}", 0)
        
        trades.append(trade)
        
    progress.empty()
    
    if trades:
        df_res = pd.DataFrame(trades)
        
        st.markdown(f"### 📊 V16 (上帝指纹·严选) 回测结果")
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
        
        st.dataframe(df_res.sort_values(['信号日'], ascending=False), use_container_width=True)
    else:
        st.warning("无交易")

if run_btn:
    run_analysis()
