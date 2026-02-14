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
st.set_page_config(page_title="快手MACD·突击版", layout="wide")
st.title("⚡ 快手 MACD · 突击实战版")
st.markdown("""
**策略核心：捕捉“形态杂乱”中的突然启动**
1.  **快手 MACD**：使用敏捷参数 (如 10, 22, 5)，比传统 MACD 快一步发现起爆。
2.  **均线突围**：收盘价站上 5/10/20 日均线 (从混乱中确立短线优势)。
3.  **KDJ 共振**：J 线处于强势区 (情绪点火)。
4.  **温和放量**：量能 > 5日均量 (主力资金进场，无需倍量)。
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
    """自定义 MACD 计算"""
    # EMA Fast
    df['ema_fast'] = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=fast_p, adjust=False).mean())
    # EMA Slow
    df['ema_slow'] = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=slow_p, adjust=False).mean())
    # DIF
    df['dif'] = df['ema_fast'] - df['ema_slow']
    # DEA
    df['dea'] = df.groupby('ts_code')['dif'].transform(lambda x: x.ewm(span=signal_p, adjust=False).mean())
    # MACD Bar
    df['macd'] = (df['dif'] - df['dea']) * 2
    return df

def calculate_kdj(df, n=9, m1=3, m2=3):
    """计算 KDJ"""
    low_list = df['low'].rolling(window=n, min_periods=9).min()
    low_list.fillna(value=df['low'].expanding().min(), inplace=True)
    
    high_list = df['high'].rolling(window=n, min_periods=9).max()
    high_list.fillna(value=df['high'].expanding().max(), inplace=True)
    
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    
    # KDJ 需要按代码分组计算，这里简化处理，直接 transform 会有问题，需手动实现 EMA
    # 为保证速度，使用简化逻辑：当天强弱
    # RSV > 50 视为强势
    df['rsv'] = rsv
    return df

def calculate_strategy(df, fast_p, slow_p, signal_p, vol_min_ratio):
    """
    计算所有信号 (快手MACD版)
    """
    # 1. 均线 (MA5, MA10, MA20)
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    
    # 2. 量能均线
    df['vol_5'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5).mean())
    
    # 3. MACD (自定义参数)
    df = calculate_macd(df, fast_p, slow_p, signal_p)
    
    # 4. 信号判定
    
    # A. 均线突围: 价格站上所有短期均线 (解决“杂乱”问题，证明今天最强)
    cond_ma = (df['close'] > df['ma5']) & (df['close'] > df['ma10']) & (df['close'] > df['ma20'])
    
    # B. 快手 MACD: DIF > DEA (处于多头状态) 且 DIF 拐头向上
    # 或者简单点：DIF > DEA 且 MACD 红柱放大
    # 更激进：MACD 刚刚金叉 (Ref 1日 DIF < DEA, 今日 DIF > DEA)
    df['dif_shift'] = df.groupby('ts_code')['dif'].transform(lambda x: x.shift(1))
    df['dea_shift'] = df.groupby('ts_code')['dea'].transform(lambda x: x.shift(1))
    
    # 金叉 或 强势延续(红柱变长)
    # 这里我们选 "金叉" 或 "水上漂" (DIF>0 且 DIF>DEA)
    cond_macd = (df['dif'] > df['dea']) & (df['dif'] > -0.5) # 允许轻微水下，但不能太深
    
    # C. 量能: 温和放量
    cond_vol = df['vol'] > (df['vol_5'] * vol_min_ratio)
    
    # D. KDJ 模拟 (RSV > 60 表示今日收盘在近期高位，强势)
    # 在杂乱K线中，如果收盘能收在 9天内的高位，说明突破了
    df['high_9'] = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(9).max())
    df['low_9'] = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(9).min())
    df['rsv'] = (df['close'] - df['low_9']) / (df['high_9'] - df['low_9'] + 0.001) * 100
    cond_kdj = df['rsv'] > 60 # 情绪强势
    
    # E. 流动性
    cond_mv = (df['amount'] > 50000) & (df['amount'] < 5000000)
    
    df['is_signal'] = cond_ma & cond_macd & cond_vol & cond_kdj & cond_mv
    
    return df

def calculate_score(row):
    score = 60
    
    # MACD 红柱越长越好 (加速)
    if row['macd'] > 0: score += 10
    
    # 站稳均线
    if row['close'] > row['ma5'] * 1.01: score += 10
    
    # 板块加分 (共振)
    if row['sector_pct'] > 1.0: score += 20
    
    # 刚启动 (RSV 还没到 100)
    if 60 < row['rsv'] < 90: score += 10
        
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ 快手 MACD 参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("⚡ 敏捷参数 (默认 10,22,5)")
    fast_p = st.number_input("快线 (Fast EMA)", 3, 20, 10, help="越小越敏感，标准为12")
    slow_p = st.number_input("慢线 (Slow EMA)", 10, 60, 22, help="越小越敏感，标准为26")
    signal_p = st.number_input("信号线 (Signal)", 3, 20, 5, help="越小金叉越快，标准为9")
    
    st.markdown("---")
    st.subheader("📈 量能与确认")
    vol_min_ratio = st.slider("量能放大倍数 (vs 5日均量)", 1.0, 3.0, 1.2, 0.1, help="1.2表示温和放量")
    
    top_n = st.number_input("每日优选 (Top N)", 1, 20, 5)
    
    run_btn = st.button("🚀 启动快手回测")

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
    with st.spinner("正在计算快手 MACD..."):
        df_sector = calculate_sector_heat(df_all, df_basic)
        df_calc = calculate_strategy(df_sector, fast_p, slow_p, signal_p, vol_min_ratio)
        
    # 4. 结果
    st.markdown("### ⚡ 快手信号诊断")
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    df_signals = df_window[df_window['is_signal']].copy()
    st.write(f"⚪ 敏捷金叉 + 温和放量标的: **{len(df_signals)}** 个")
    
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
        
        st.markdown(f"### 📊 快手突击回测结果 (Top {top_n})")
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
