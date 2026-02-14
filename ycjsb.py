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
st.set_page_config(page_title="潜龙 V14·点火者", layout="wide")
st.title("🐉 潜龙 V14·点火者 (黄金板块+硬核龙头)")
st.markdown("""
**策略核心：在"还没高潮"的板块里，抓"最硬"的龙头**
1.  **黄金板块**：**2.0% < 板块涨幅 < 3.5%** (V3.1 数据验证过的暴利温区，拒绝 >4% 的高潮板块)。
2.  **硬核龙头**：**当日涨幅 > 9.5%** (必须是涨停或 20cm 大阳，拒绝跟风杂毛)。
3.  **资金点火**：**量比 > 1.5** (主力资金大举进攻)。
4.  **趋势护航**：**RSI > 70** (姿态高昂)。
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
            # 获取日线
            df = pro.daily(trade_date=date)
            # 获取指标
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,volume_ratio,circ_mv')
            
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
def calculate_strategy(df_all, df_basic, sec_min, sec_max):
    """
    V14 核心逻辑: Golden Sector -> Top Leader
    """
    # 1. 预处理
    if 'industry' not in df_all.columns:
        df_merged = pd.merge(df_all, df_basic[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df_merged = df_all.copy()
    
    # 辅助指标: RSI
    df_merged['up_move'] = np.where(df_merged['pct_chg'] > 0, df_merged['pct_chg'], 0)
    df_merged['down_move'] = np.where(df_merged['pct_chg'] < 0, abs(df_merged['pct_chg']), 0)
    avg_up = df_merged.groupby('ts_code')['up_move'].transform(lambda x: x.rolling(6).mean())
    avg_down = df_merged.groupby('ts_code')['down_move'].transform(lambda x: x.rolling(6).mean())
    df_merged['rsi_6'] = 100 * avg_up / (avg_up + avg_down + 0.0001)

    results = []
    dates = sorted(df_merged['trade_date'].unique())
    
    for i in range(10, len(dates)):
        curr_date = dates[i]
        daily_data = df_merged[df_merged['trade_date'] == curr_date].copy()
        
        if daily_data.empty: continue
        
        # === Step 1: 筛选黄金板块 (温热区) ===
        sector_stats = daily_data.groupby('industry').agg({
            'pct_chg': 'mean',
            'ts_code': 'count',
            'amount': 'sum'
        }).reset_index()
        
        # 过滤小板块
        sector_stats = sector_stats[(sector_stats['ts_code'] > 5) & (sector_stats['amount'] > 100000)]
        
        # 关键: 2.0% < 板块涨幅 < 3.5%
        # 这个区间代表资金进场了，但还没疯狂，还可以进
        golden_sectors = sector_stats[(sector_stats['pct_chg'] > sec_min) & (sector_stats['pct_chg'] < sec_max)]
        golden_sector_names = golden_sectors['industry'].tolist()
        
        if not golden_sector_names: continue
        
        # === Step 2: 在黄金板块里找"硬核龙头" ===
        candidates = daily_data[daily_data['industry'].isin(golden_sector_names)].copy()
        
        # 1. 必须是涨停板 (或 > 9.5%) - 绝对的领头羊
        winners = candidates[candidates['pct_chg'] > 9.5]
        
        if winners.empty: continue
        
        # 2. 资金点火: 量比 > 1.5
        winners = winners[winners['volume_ratio'] > 1.5]
        
        # 3. 趋势护航: RSI > 70
        winners = winners[winners['rsi_6'] > 70]
        
        # 4. 换手率 > 3% (拒绝一字)
        winners = winners[winners['turnover_rate'] > 3.0]
        
        if winners.empty: continue
        
        # 每天取最强的 Top 2 (按量比和RSI综合)
        winners['score'] = winners['volume_ratio'] + (winners['rsi_6'] / 10)
        top_picks = winners.sort_values('score', ascending=False).head(2)
        
        for _, row in top_picks.iterrows():
            # 获取该股所在板块的实际涨幅
            sec_gain = sector_stats[sector_stats['industry'] == row['industry']]['pct_chg'].values[0]
            
            results.append({
                'ts_code': row['ts_code'],
                'trade_date': curr_date,
                'name': row['name'],
                'industry': row['industry'],
                'sector_pct': sec_gain,
                'pct_chg': row['pct_chg'],
                'vol_ratio': row['volume_ratio'],
                'close': row['close'],
                'is_signal': True
            })
            
    return pd.DataFrame(results)

def calculate_score(row):
    # 评分: 板块涨幅适中最好(2.5-3.0)，个股越强越好
    score = 60
    # 奖励黄金板块
    if 2.5 <= row['sector_pct'] <= 3.2: score += 20
    # 奖励大长腿
    if row['pct_chg'] > 15.0: score += 20
    
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V14 点火者参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("🔥 黄金温区")
    col1, col2 = st.columns(2)
    sec_min = col1.number_input("板块下限%", 0.0, 5.0, 2.0, help="确保有热度")
    sec_max = col2.number_input("板块上限%", 2.0, 10.0, 3.5, help="拒绝高潮")
    
    run_btn = st.button("🚀 启动点火回测")

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
    with st.spinner("正在寻找风口上的点火者..."):
        df_calc = calculate_strategy(df_all, df_basic, sec_min, sec_max)
        
    # 4. 结果
    st.markdown("### 🐉 V14 诊断 (黄金板块+硬核龙头)")
    
    if df_calc.empty:
        st.warning("无信号。")
        return
        
    # 过滤时间窗
    valid_dates = cal_dates[-(days_back):] 
    df_signals = df_calc[df_calc['trade_date'].isin(valid_dates)].copy()
    
    st.write(f"⚪ 捕获点火者: **{len(df_signals)}** 个")

    # 5. 评分与 Top N
    df_signals['潜龙分'] = df_signals.apply(calculate_score, axis=1)
    df_signals = df_signals.sort_values(['trade_date', '潜龙分'], ascending=[True, False])
    
    # 6. 回测
    price_lookup = df_all[['ts_code', 'trade_date', 'open', 'close', 'low', 'pre_close']].set_index(['ts_code', 'trade_date'])
    trades = []
    
    progress = st.progress(0)
    total_sig = len(df_signals)
    
    for i, row in enumerate(df_signals.itertuples()):
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
            '信号日': signal_date, '代码': code, '名称': row.name, 
            '行业': row.industry, '板块涨幅': f"{row.sector_pct:.1f}%",
            '个股涨幅': f"{row.pct_chg:.1f}%",
            '量比': f"{row.vol_ratio:.1f}",
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
        
        st.markdown(f"### 📊 V14 (点火者) 回测结果")
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
