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
st.set_page_config(page_title="潜龙·狙击手 (终极版)", layout="wide")
st.title("🎯 潜龙·狙击手 (黄金区间自动锁定)")
st.markdown("""
**策略逻辑 (V5.0 终极定稿)：**
1.  **极品吸筹**：1.7 < VR < 2.0 (拒绝诱多)。
2.  **板块共振**：1.2% < 涨幅 < 3.0% (拒绝高潮接盘)。
3.  **攻击形态**：RSI > 55 (主升浪特征)。
4.  **自动风控**：已内置所有“盖子”，选出来的就是最终标的。
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

def calculate_strategy(df, vol_mul, box_min, box_max, vr_min, vr_max, rsi_min, rsi_max, sec_min, sec_max):
    """
    计算所有信号 (含自动盖子)
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
    
    # 3. RSI 指标
    df['up_move'] = np.where(df['pct_chg'] > 0, df['pct_chg'], 0)
    df['down_move'] = np.where(df['pct_chg'] < 0, abs(df['pct_chg']), 0)
    avg_up = df.groupby('ts_code')['up_move'].transform(lambda x: x.rolling(6).mean())
    avg_down = df.groupby('ts_code')['down_move'].transform(lambda x: x.rolling(6).mean())
    df['rsi_6'] = 100 * avg_up / (avg_up + avg_down + 0.0001)
    
    # 4. 信号判定
    # A. 振幅区间
    cond_box = (df['box_amplitude'] > (box_min/100)) & (df['box_amplitude'] < (box_max/100))
    
    # B. 价格突破
    cond_break = df['close'] > df['high_60']
    
    # C. 量能突破
    cond_vol = df['vol'] > (df['vol_60'] * vol_mul)
    
    # D. 流动性筛选
    cond_mv = (df['amount'] > 50000) & (df['amount'] < 5000000)
    
    # === E. 黄金区间锁定 (自动盖子) ===
    # 吸筹比率：1.7 ~ 2.0 (剔除诱多)
    cond_acc = (df['accumulation_ratio'] > vr_min) & (df['accumulation_ratio'] < vr_max)
    
    # RSI：55 ~ 85
    cond_rsi = (df['rsi_6'] > rsi_min) & (df['rsi_6'] < rsi_max)
    
    # 板块涨幅：1.2% ~ 3.0% (剔除高潮)
    # 注意：板块数据可能为NaN，需填充
    df['sector_pct'] = df['sector_pct'].fillna(0)
    cond_sec = (df['sector_pct'] > sec_min) & (df['sector_pct'] < sec_max)
    
    df['is_signal'] = cond_box & cond_break & cond_vol & cond_mv & cond_acc & cond_rsi & cond_sec
    
    return df

def calculate_score(row):
    score = 60
    # 简单的评分，主要用于每日排序
    # 因为已经筛选得很严了，这里只区分微小的优劣
    
    # 板块越接近 2.5% 越好 (风口正盛但未过热)
    if 2.0 <= row['sector_pct'] <= 2.8: score += 20
    elif 1.2 <= row['sector_pct'] < 2.0: score += 10
    
    # VR 越接近 1.9 越好
    if 1.8 <= row['accumulation_ratio'] <= 1.95: score += 20
    
    # 振幅适中
    amp = row['box_amplitude'] * 100
    if 20 <= amp <= 35: score += 10
        
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ 狙击手参数 (黄金区间)")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("数据回溯天数", 60, 300, 120)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("🎯 黄金区间控制")
    col1, col2 = st.columns(2)
    # VR 控制
    vr_min = col1.number_input("VR 下限", 1.0, 2.0, 1.7)
    vr_max = col2.number_input("VR 上限 (防诱多)", 2.0, 5.0, 2.0)
    
    # 板块控制
    sec_min = col1.number_input("板块涨幅下限%", 0.0, 2.0, 1.2)
    sec_max = col2.number_input("板块涨幅上限%", 2.0, 5.0, 3.0, help="超过3%容易接盘")
    
    # 振幅控制
    box_min = st.slider("振幅下限%", 5, 20, 10)
    box_max = st.slider("振幅上限%", 30, 60, 40)
    
    vol_mul = st.slider("突破量能倍数", 1.5, 5.0, 1.8)
    top_n = st.number_input("每日优选 (Top N)", 1, 20, 5)
    
    run_btn = st.button("🚀 启动最终回测")

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
    with st.spinner("正在执行黄金区间筛选..."):
        df_sector = calculate_sector_heat(df_all, df_basic)
        df_calc = calculate_strategy(df_sector, vol_mul, box_min, box_max, vr_min, vr_max, 55, 85, sec_min, sec_max)
        
    # 4. 结果
    st.markdown("### 🎯 狙击诊断")
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    df_signals = df_window[df_window['is_signal']].copy()
    st.write(f"⚪ 最终符合黄金区间的标的: **{len(df_signals)}** 个 (120天)")
    
    if df_signals.empty:
        st.warning("太严了，没有股票入选。")
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
            '吸筹比率': f"{row.accumulation_ratio:.2f}",
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
        
        st.markdown(f"### 📊 黄金回测结果 (Top {top_n})")
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
