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
st.set_page_config(page_title="潜龙·共振实战版", layout="wide")
st.title("🐉 潜龙·共振实战系统 (箱体突破 + 板块热度)")
st.markdown("""
**策略核心逻辑 (V3.1 修复版)：**
1.  **形态基石**：10% < 振幅 < 40% (拒绝死鱼与疯牛)。
2.  **身份验证**：50亿 < 流通市值 < 500亿 (锁定机构趋势票)。
3.  **爆发信号**：创 60日新高 + 放量 (突破发令枪)。
4.  **板块共振**：**移植自ZL1策略**，只做当日强势板块的成分股 (拒绝孤军深入)。
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
    """
    获取基础信息 (含行业 industry 和 名称 name)
    """
    ts.set_token(token)
    pro = ts.pro_api()
    
    for _ in range(3):
        try:
            time.sleep(0.5)
            # industry 是核心字段
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,industry,list_date')
            if not df.empty:
                df = df[~df['name'].str.contains('ST')]
                df = df[~df['market'].str.contains('北交')]
                df = df[~df['ts_code'].str.contains('BJ')]
                return df
        except: time.sleep(1)
    return pd.DataFrame()

# ==========================================
# 3. 核心计算：板块热度 + 箱体突破
# ==========================================
def calculate_sector_heat(df_daily, df_basic):
    """
    计算当日板块热度 (借鉴 ZL1)
    修复：同时合并 name 字段，防止后续报错
    """
    # 合并行业信息 和 名称信息
    if 'industry' not in df_daily.columns:
        # === 关键修复：这里加入了 'name' ===
        df_merged = pd.merge(df_daily, df_basic[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df_merged = df_daily.copy()
        
    # 按 日期 + 行业 分组，计算平均涨幅
    valid_df = df_merged[df_merged['pct_chg'] != 0]
    
    # 注意：如果某股票没有行业归属，会被过滤掉，这是预期的
    sector_stats = valid_df.groupby(['trade_date', 'industry'])['pct_chg'].mean().reset_index()
    sector_stats.rename(columns={'pct_chg': 'sector_pct'}, inplace=True)
    
    # 将板块热度合并回原数据
    df_final = pd.merge(df_merged, sector_stats, on=['trade_date', 'industry'], how='left')
    
    return df_final

def calculate_strategy(df, vol_mul, box_min, box_max, mv_min, mv_max, df_basic):
    """
    计算所有信号
    """
    # 1. 箱体指标
    df['high_60'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(1).rolling(window=60).max())
    df['low_60'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(1).rolling(window=60).min())
    df['vol_60'] = df.groupby('ts_code')['vol'].transform(lambda x: x.shift(1).rolling(window=60).mean())
    df['box_amplitude'] = (df['high_60'] - df['low_60']) / df['low_60']
    
    # 2. 信号判定
    # A. 振幅区间 (10% ~ 40%)
    cond_box = (df['box_amplitude'] > (box_min/100)) & (df['box_amplitude'] < (box_max/100))
    
    # B. 价格突破 (创60日新高)
    cond_break = df['close'] > df['high_60']
    
    # C. 量能突破
    cond_vol = df['vol'] > (df['vol_60'] * vol_mul)
    
    # D. 流动性筛选 (近似市值筛选)
    # 成交额 5000万 ~ 50亿
    cond_mv = (df['amount'] > 50000) & (df['amount'] < 5000000)
    
    # E. 板块共振
    # 要求所属板块当日平均涨幅 > 1.0% (说明板块在动)
    # 填充NaN防止报错
    df['sector_pct'] = df['sector_pct'].fillna(0)
    # 我们将在外部通过 slider 控制阈值，这里先标记，后面 filter
    
    df['is_signal_base'] = cond_box & cond_break & cond_vol & cond_mv
    
    return df

def calculate_score(row):
    """
    评分系统 (偏好活跃股)
    """
    score = 60
    
    # 1. 振幅分：偏好 20%-35% 的活跃潜伏
    amp = row['box_amplitude'] * 100
    if 20 <= amp <= 35:
        score += 20 # 满分
    elif 10 <= amp < 20:
        score += 10 # 及格
    
    # 2. 板块分：板块越热越好
    if row['sector_pct'] > 0:
        score += min(row['sector_pct'] * 5, 30) # 板块涨 2% 加 10分
        
    # 3. 突破力度
    if row['high_60'] > 0:
        brk = (row['close'] - row['high_60']) / row['high_60'] * 100
        score += min(brk * 2, 10)
        
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ 潜龙·共振版参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("数据回溯天数", 60, 300, 120)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("📦 形态与身份")
    col1, col2 = st.columns(2)
    box_min = col1.number_input("振幅下限%", 5, 20, 15)
    box_max = col2.number_input("振幅上限%", 30, 60, 45)
    
    vol_mul = st.slider("突破量能倍数", 1.5, 5.0, 1.8, 0.1)
    
    st.markdown("---")
    st.subheader("🔥 板块共振")
    sector_min_rise = st.slider("板块最低涨幅 (%)", 0.0, 3.0, 1.0, 0.1, help="所属行业当日平均涨幅需超过此值，才算共振。")
    
    top_n = st.number_input("每日优选 (Top N)", 1, 50, 20, help="放宽到20以便观察")
    
    run_btn = st.button("🚀 启动共振回测")

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
        st.error("无法获取行业数据，板块共振无法计算。")
        return
        
    # 3. 计算板块热度 (Sector Boost)
    with st.spinner("正在计算全市场板块热度 (ZL1 引擎)..."):
        # 修复：确保这里带上了 'name'
        df_sector = calculate_sector_heat(df_all, df_basic)
    
    # 4. 计算策略信号
    with st.spinner("正在扫描潜龙形态..."):
        df_calc = calculate_strategy(df_sector, vol_mul, box_min, box_max, 0, 0, df_basic)
        
    # 5. 漏斗诊断
    st.markdown("### 🕵️‍♀️ 共振漏斗")
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    st.write(f"⚪ 样本总数: {len(df_window):,} 条")
    
    # 重新应用过滤逻辑以显示漏斗
    c_mv = (df_window['amount'] > 50000) & (df_window['amount'] < 5000000)
    n_mv = len(df_window[c_mv])
    st.write(f"1️⃣ 流动性筛选 (成交额5千万-50亿): {n_mv:,}")
    
    c_box = (df_window['box_amplitude'] > (box_min/100)) & (df_window['box_amplitude'] < (box_max/100))
    n_box = len(df_window[c_mv & c_box])
    st.write(f"2️⃣ 形态筛选 ({box_min}% < 振幅 < {box_max}%): {n_box:,}")
    
    c_sec = df_window['sector_pct'] > sector_min_rise
    n_sec = len(df_window[c_mv & c_box & c_sec])
    st.write(f"3️⃣ 板块共振 (行业涨幅 > {sector_min_rise}%): {n_sec:,} (大幅过滤孤狼)")
    
    # 最终信号
    df_window['is_signal'] = df_window['is_signal_base'] & (df_window['sector_pct'] > sector_min_rise)
    df_signals = df_window[df_window['is_signal']].copy()
    st.write(f"4️⃣ 最终突破 (量价齐升): **{len(df_signals)}** 个")
    
    if df_signals.empty:
        st.warning("无符合条件的信号。尝试降低板块涨幅要求。")
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
        
        st.markdown(f"### 📊 共振回测结果 (Top {top_n})")
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
        
        st.markdown("### 🏆 潜龙榜 (含行业数据)")
        display_cols = ['信号日', '排名', '代码', '名称', '行业', '板块涨幅', '潜龙分', '状态'] + \
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
