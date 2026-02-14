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
st.set_page_config(page_title="潜龙 V5·双核驱动", layout="wide")
st.title("🐉 潜龙 V5·双核驱动 (MACD空中加油 + RSI极值)")
st.markdown("""
**策略核心：集大成之作 (去粗取精)**
1.  **极值筛选**：**RSI > 80** (继承 V3.1 的暴利基因，过滤 99% 杂毛)。
2.  **空中加油**：**MACD 红柱放大** 或 **DIF 拒绝死叉** (比普通金叉更强的加速信号)。
3.  **结构突破**：**收盘价创 60日新高** (必须解放全人类)。
4.  **板块共振**：1.5% < 板块涨幅 < 4.5% (拒绝冷门，也拒绝接盘)。
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

def calculate_macd(df, fast_p=12, slow_p=26, signal_p=9):
    """标准 MACD"""
    df['ema_fast'] = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=fast_p, adjust=False).mean())
    df['ema_slow'] = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=slow_p, adjust=False).mean())
    df['dif'] = df['ema_fast'] - df['ema_slow']
    df['dea'] = df.groupby('ts_code')['dif'].transform(lambda x: x.ewm(span=signal_p, adjust=False).mean())
    df['macd'] = (df['dif'] - df['dea']) * 2
    return df

def calculate_strategy(df, vol_mul, rsi_min, sec_min, sec_max):
    """
    计算所有信号 (V5 双核驱动)
    """
    # 1. 结构: 60日新高
    df['high_60'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(1).rolling(60).max())
    df['vol_60'] = df.groupby('ts_code')['vol'].transform(lambda x: x.shift(1).rolling(60).mean())
    
    # 2. 动量: RSI (6日)
    df['up_move'] = np.where(df['pct_chg'] > 0, df['pct_chg'], 0)
    df['down_move'] = np.where(df['pct_chg'] < 0, abs(df['pct_chg']), 0)
    avg_up = df.groupby('ts_code')['up_move'].transform(lambda x: x.rolling(6).mean())
    avg_down = df.groupby('ts_code')['down_move'].transform(lambda x: x.rolling(6).mean())
    df['rsi_6'] = 100 * avg_up / (avg_up + avg_down + 0.0001)
    
    # 3. 趋势: MACD
    df = calculate_macd(df)
    # 计算 MACD 变化 (判断空中加油)
    df['macd_prev'] = df.groupby('ts_code')['macd'].transform(lambda x: x.shift(1))
    
    # === 信号判定 (严苛过滤) ===
    
    # A. 核心1: 极度亢奋 (RSI > 80)
    # 这是把 9000 个 降到 200 个的关键
    cond_rsi = (df['rsi_6'] > rsi_min) & (df['rsi_6'] < 98)
    
    # B. 核心2: MACD 加速 (空中加油)
    # 逻辑: DIF > 0 (水上) 且 (MACD刚翻红 或 MACD红柱变长)
    cond_macd_pos = df['dif'] > 0
    cond_macd_acc = (df['macd'] > 0) & (df['macd'] > df['macd_prev']) # 红柱放大
    cond_macd = cond_macd_pos & cond_macd_acc
    
    # C. 核心3: 价格新高 (确认突破)
    cond_break = df['close'] >= df['high_60']
    
    # D. 板块护航
    df['sector_pct'] = df['sector_pct'].fillna(0)
    cond_sec = (df['sector_pct'] > sec_min) & (df['sector_pct'] < sec_max)
    
    # E. 量能确认
    cond_vol = df['vol'] > (df['vol_60'] * vol_mul)
    
    # F. 流动性
    cond_mv = (df['amount'] > 50000) & (df['amount'] < 5000000)
    
    df['is_signal'] = cond_rsi & cond_macd & cond_break & cond_sec & cond_vol & cond_mv
    
    return df

def calculate_score(row):
    score = 60
    
    # 双核评分
    
    # 1. RSI 越接近 90 越好
    if 85 <= row['rsi_6'] <= 95: score += 30
    elif 80 <= row['rsi_6'] < 85: score += 15
    
    # 2. MACD 越强越好 (DIF高位加速)
    if row['dif'] > 0.5: score += 10
    
    # 3. 板块适中
    if 2.0 <= row['sector_pct'] <= 4.0: score += 10
        
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V5 双核驱动参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("🔥 核心阈值 (严苛)")
    
    # RSI
    rsi_min = st.number_input("RSI 下限", 0, 100, 80, help="V3.1验证过的暴利门槛")
    
    # 板块
    col1, col2 = st.columns(2)
    sec_min = col1.number_input("板块下限%", 0.0, 5.0, 1.5)
    sec_max = col2.number_input("板块上限%", 2.0, 10.0, 4.5)
    
    vol_mul = st.slider("突破量能倍数", 1.0, 5.0, 1.5)
    top_n = st.number_input("每日优选 (Top N)", 1, 20, 2)
    
    run_btn = st.button("🚀 启动双核回测")

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
    with st.spinner("正在筛选 RSI+MACD 双强标的..."):
        df_sector = calculate_sector_heat(df_all, df_basic)
        df_calc = calculate_strategy(df_sector, vol_mul, rsi_min, sec_min, sec_max)
        
    # 4. 结果
    st.markdown("### 🐉 V5 诊断")
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    df_signals = df_window[df_window['is_signal']].copy()
    st.write(f"⚪ RSI>80 + MACD加速 + 新高标的: **{len(df_signals)}** 个")
    
    if df_signals.empty:
        st.warning("无信号。市场情绪不足。")
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
            'RSI': f"{row.rsi_6:.1f}",
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
        
        st.markdown(f"### 📊 V5 (双核) 回测结果 (Top {top_n})")
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
