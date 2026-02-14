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
st.set_page_config(page_title="潜龙 V4·MACD共振", layout="wide")
st.title("⚡ 潜龙 V4·MACD 趋势共振 (水上漂战法)")
st.markdown("""
**策略核心：拒绝垃圾股，只做“水上漂”**
1.  **水上金叉**：**DIF > 0** (必须在0轴上方，确保多头主控，过滤90%垃圾)。
2.  **趋势护体**：**MA20 > MA60** (中期趋势向上，不做下跌反弹)。
3.  **高位起爆**：收盘价 > **60日区间的80%分位** (拒绝抄底，只做主升)。
4.  **快手MACD**：10, 22, 5 (敏捷参数，快人一步)。
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
# 3. 核心计算
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

def calculate_macd(df, fast_p, slow_p, signal_p):
    """自定义 MACD"""
    df['ema_fast'] = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=fast_p, adjust=False).mean())
    df['ema_slow'] = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=slow_p, adjust=False).mean())
    df['dif'] = df['ema_fast'] - df['ema_slow']
    df['dea'] = df.groupby('ts_code')['dif'].transform(lambda x: x.ewm(span=signal_p, adjust=False).mean())
    df['macd'] = (df['dif'] - df['dea']) * 2
    return df

def calculate_strategy(df, fast_p, slow_p, signal_p, vol_min_ratio):
    """
    计算所有信号 (V4 MACD 趋势共振版)
    """
    # 1. 均线系统 (必须多头)
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma60'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(60).mean())
    
    # 2. 结构位置 (近60天高低点)
    df['high_60'] = df.groupby('ts_code')['high'].transform(lambda x: x.shift(1).rolling(60).max())
    df['low_60'] = df.groupby('ts_code')['low'].transform(lambda x: x.shift(1).rolling(60).min())
    
    # 3. MACD
    df = calculate_macd(df, fast_p, slow_p, signal_p)
    
    # 4. 量能
    df['vol_5'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5).mean())
    
    # === 信号判定 (严苛过滤) ===
    
    # A. 水上漂 (核心核心核心)
    # DIF > 0 意味着价格在长期均线之上，属于多头市场
    cond_water = df['dif'] > 0
    
    # B. MACD 攻击形态
    # 刚刚金叉 (DIF 上穿 DEA) 或者 空中加油 (DIF 不死叉反身向上)
    # 这里用简单的强势判定: DIF > DEA 且 MACD 红柱
    cond_macd = (df['dif'] > df['dea']) & (df['macd'] > 0)
    
    # C. 趋势护体
    # 中期趋势必须向上 (MA20 > MA60)
    cond_trend = df['ma20'] > df['ma60']
    
    # D. 高位起爆 (拒绝抄底)
    # 当前价必须位于过去60天震荡区间的 上方 20% 区域
    # 公式: (Close - Low60) / (High60 - Low60) > 0.8
    # 这意味着股票非常强，准备突破或已经突破
    position_ratio = (df['close'] - df['low_60']) / (df['high_60'] - df['low_60'] + 0.001)
    cond_pos = position_ratio > 0.8
    
    # E. 温和放量
    cond_vol = df['vol'] > (df['vol_5'] * vol_min_ratio)
    
    # F. 流动性
    cond_mv = (df['amount'] > 50000) & (df['amount'] < 5000000)
    
    df['is_signal'] = cond_water & cond_macd & cond_trend & cond_pos & cond_vol & cond_mv
    
    return df

def calculate_score(row):
    score = 60
    
    # MACD 越强越好
    if row['macd'] > 0: score += 10
    
    # 均线多头排列
    if row['ma5'] > row['ma10'] > row['ma20']: score += 20
    
    # 板块加分
    if row['sector_pct'] > 1.5: score += 20
    elif row['sector_pct'] > 0.8: score += 10
        
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V4 MACD共振参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("⚡ 敏捷参数 (默认 10,22,5)")
    col1, col2 = st.columns(2)
    fast_p = col1.number_input("快线", 3, 20, 10)
    slow_p = col2.number_input("慢线", 10, 60, 22)
    signal_p = st.number_input("信号线", 3, 20, 5)
    
    st.markdown("---")
    vol_min_ratio = st.slider("量能放大倍数", 1.0, 3.0, 1.2, 0.1)
    top_n = st.number_input("每日优选 (Top N)", 1, 20, 2, help="建议 Top 2")
    
    run_btn = st.button("🚀 启动水上漂回测")

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
        
    # 3. 计算
    with st.spinner("正在筛选水上漂标的..."):
        df_sector = calculate_sector_heat(df_all, df_basic)
        df_calc = calculate_strategy(df_sector, fast_p, slow_p, signal_p, vol_min_ratio)
        
    # 4. 结果
    st.markdown("### ⚡ V4 诊断")
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    df_signals = df_window[df_window['is_signal']].copy()
    st.write(f"⚪ 水上金叉 + 趋势共振标的: **{len(df_signals)}** 个")
    
    if df_signals.empty:
        st.warning("无信号。")
        return

    # 5. 评分与 Top N
    df_signals['潜龙分'] = df_signals.apply(calculate_score, axis=1)
    df_signals = df_signals.sort_values(['trade_date', '潜龙分'], ascending=[True, False])
    df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
    
    df_top = df_signals[df_signals['排名'] <= top_n].copy()
    
    # 6. 回测
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
            
        if not future_dates: continue
            
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
            'DIF': f"{row.dif:.2f}",
            '买入价': buy_price, '状态': '持有'
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
        
        st.markdown(f"### 📊 V4 (水上漂) 回测结果 (Top {top_n})")
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
