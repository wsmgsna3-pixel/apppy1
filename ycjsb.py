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
st.set_page_config(page_title="箱体潜龙·突破实战版", layout="wide")
st.title("🐉 箱体潜龙·突破实战系统 (Box Breakout)")
st.markdown("""
**策略核心迭代：**
1.  **寻找潜伏**：锁定过去 60 天振幅 < 35% 的“死鱼股”（主力吸筹）。
2.  **捕捉惊雷**：**收盘价创60日新高** + **放量2倍** = 立即报警。
3.  **极速切入**：突破次日直接买入，不再等待三天，抢占“鱼头”。
4.  **信号冷却**：单只股票 20 天内只做第一次突破，拒绝反复挨打。
""")

# ==========================================
# 2. 核心数据引擎 (保持稳定版)
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
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,list_date')
            if not df.empty:
                df = df[~df['name'].str.contains('ST')]
                df = df[~df['market'].str.contains('北交')]
                df = df[~df['ts_code'].str.contains('BJ')]
                return df
        except: time.sleep(1)
    return pd.DataFrame()

# ==========================================
# 3. 核心计算：箱体与突破 (全新逻辑)
# ==========================================
def apply_cool_down(group, window=20):
    """
    冷却期过滤器：
    如果 Day T 触发信号，则 Day T+1 到 T+window 内的信号全部作废。
    """
    signals = group['is_signal'].values
    dates = group['trade_date'].values
    
    # 如果没有信号，直接返回全False
    if not np.any(signals):
        return pd.Series(False, index=group.index)
    
    # 找到所有信号的索引位置
    sig_indices = np.where(signals)[0]
    
    # 保留的信号掩码
    keep_mask = np.zeros_like(signals, dtype=bool)
    
    last_idx = -999
    for idx in sig_indices:
        # 如果当前信号距离上一个有效信号超过 window，则保留
        if idx - last_idx >= window:
            keep_mask[idx] = True
            last_idx = idx
            
    return pd.Series(keep_mask, index=group.index)

def calculate_box_breakout(df, vol_mul, box_limit):
    """
    向量化计算箱体突破
    """
    # 1. 计算过去 60 天的数据 (不含当天)
    # 60日最高收盘价 (箱体上沿)
    df['high_60'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(1).rolling(window=60).max())
    # 60日最低收盘价 (箱体下沿)
    df['low_60'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(1).rolling(window=60).min())
    # 60日均量
    df['vol_60'] = df.groupby('ts_code')['vol'].transform(lambda x: x.shift(1).rolling(window=60).mean())
    
    # 2. 计算箱体振幅
    # 振幅 = (上沿 - 下沿) / 下沿
    df['box_amplitude'] = (df['high_60'] - df['low_60']) / df['low_60']
    
    # 3. 判断突破信号
    # A. 潜伏条件：箱体振幅 < 阈值 (如 35%)
    cond_box = df['box_amplitude'] < (box_limit / 100)
    
    # B. 价格突破：今天收盘价 > 过去60天最高价
    cond_break = df['close'] > df['high_60']
    
    # C. 量能突破：今天成交量 > 60日均量 * 倍数
    cond_vol = df['vol'] > (df['vol_60'] * vol_mul)
    
    # D. 基础门槛 (股价>10, 成交额>5000万, 非停牌)
    cond_basic = (df['close'] >= 10) & (df['amount'] > 50000) & (df['vol'] > 0)
    
    # 初步信号
    df['is_signal'] = cond_box & cond_break & cond_vol & cond_basic
    
    # 4. 应用冷却期 (20天内不重复触发)
    # 对每个股票分组处理，这步稍微慢一点，但为了逻辑严谨必须做
    # 仅对至少有一个信号的股票处理，加速
    has_signal_codes = df[df['is_signal']]['ts_code'].unique()
    
    # 默认全为 False
    df['final_signal'] = False
    
    # 只处理有信号的股票
    if len(has_signal_codes) > 0:
        mask_codes = df['ts_code'].isin(has_signal_codes)
        df.loc[mask_codes, 'final_signal'] = df[mask_codes].groupby('ts_code').apply(
            lambda x: apply_cool_down(x, window=20)
        ).reset_index(level=0, drop=True)
        
    return df

def calculate_score(row):
    """
    潜龙分 (突破版)：
    1. 突破力度：收盘价超箱体上沿越多越好
    2. 箱体极致：箱体越扁越好 (振幅越小)
    3. 量能倍数：越大越好
    """
    score = 60
    
    # 箱体越窄加分 (基准 30%，每小 1% 加 1分)
    box_amp = row['box_amplitude'] * 100
    if box_amp < 30:
        score += (30 - box_amp) * 1.5
        
    # 突破力度 (超上沿幅度)
    break_pct = (row['close'] - row['high_60']) / row['high_60'] * 100
    score += min(break_pct * 2, 20)
    
    # 量能倍数
    vol_ratio = row['vol'] / (row['vol_60'] + 1)
    score += min(vol_ratio * 5, 20)
    
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ 突破版参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    st.info("💡 提示：'回测天数'建议设为 120天 以上，以保证有足够数据计算60日箱体。")
    days_back = st.slider("数据回溯天数", 60, 300, 120)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("📦 箱体与突破设置")
    box_limit = st.slider("箱体振幅上限 (%)", 20, 50, 35, help="过去60天震幅小于此值才算潜伏。越小越严。")
    vol_mul = st.slider("突破量能倍数", 1.5, 5.0, 2.0, 0.1, help="突破当日成交量需达到均量的多少倍")
    
    top_n = st.number_input("每日优选 (Top N)", 1, 10, 5)
    
    run_btn = st.button("🚀 启动箱体突破扫描")

def run_analysis():
    if not user_token:
        st.error("请先输入 Token")
        return

    # 1. 准备数据
    end_str = end_date_input.strftime('%Y%m%d')
    # 缓冲: 60天箱体计算 + 15天未来 + 回测天数
    start_dt = end_date_input - timedelta(days=days_back * 1.5 + 80)
    
    cal_dates = get_trade_cal(user_token, start_dt.strftime('%Y%m%d'), end_str)
    if not cal_dates:
        st.error("获取日历失败")
        return
        
    df_all = fetch_all_market_data_by_date(user_token, cal_dates)
    if df_all.empty:
        st.error("数据加载失败")
        return
    st.success(f"✅ 数据就绪: {len(df_all):,} 条 K线")

    # 2. 基础过滤
    df_basic = get_stock_basics(user_token)
    if not df_basic.empty:
        df_all = df_all[df_all['ts_code'].isin(df_basic['ts_code'])]
        df_all = pd.merge(df_all, df_basic[['ts_code', 'name', 'market']], on='ts_code', how='left')
    
    # 3. 计算指标 (核心)
    with st.spinner("正在扫描箱体形态与突破信号..."):
        df_calc = calculate_box_breakout(df_all, vol_mul, box_limit)
        
    # 4. 漏斗诊断
    st.markdown("### 🕵️‍♀️ 突破漏斗诊断")
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    st.write(f"⚪ 样本总数: {len(df_window):,} 条")
    
    # A. 基础门槛
    c_basic = (df_window['close'] >= 10) & (df_window['amount'] > 50000)
    n_basic = len(df_window[c_basic])
    st.write(f"1️⃣ 基础门槛 (价>10): {n_basic:,}")
    
    # B. 潜伏期 (箱体)
    c_box = df_window['box_amplitude'] < (box_limit / 100)
    n_box = len(df_window[c_basic & c_box])
    st.write(f"2️⃣ 潜伏期筛选 (振幅<{box_limit}%): {n_box:,} (符合箱体形态)")
    
    # C. 突破 (价+量)
    c_break = df_window['close'] > df_window['high_60']
    c_vol = df_window['vol'] > (df_window['vol_60'] * vol_mul)
    n_break = len(df_window[c_basic & c_box & c_break & c_vol])
    st.write(f"3️⃣ 突破筛选 (创60日新高+放量): {n_break:,}")
    
    # D. 冷却期
    df_signals = df_window[df_window['final_signal']].copy()
    st.write(f"4️⃣ 冷却去重 (20天不重复): 最终买点 **{len(df_signals)}** 个")
    
    if df_signals.empty:
        st.warning("无符合条件的突破信号。")
        return

    # 5. 评分与 Top N
    df_signals['潜龙分'] = df_signals.apply(calculate_score, axis=1)
    df_signals = df_signals.sort_values(['trade_date', '潜龙分'], ascending=[True, False])
    df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
    df_signals = df_signals[df_signals['排名'] <= top_n]
    
    # 6. 收益回测
    price_lookup = df_calc[['ts_code', 'trade_date', 'open', 'close', 'low']].set_index(['ts_code', 'trade_date'])
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
            
        if not future_dates:
            trades.append({
                '信号日': signal_date, '代码': code, '名称': row.name, '排名': row.排名,
                '潜龙分': row.潜龙分, '箱体振幅': f"{row.box_amplitude*100:.1f}%",
                '状态': '等待开盘'
            })
            continue
            
        d1_date = future_dates[0]
        if (code, d1_date) not in price_lookup.index: continue
        d1_data = price_lookup.loc[(code, d1_date)]
        
        # 风控: D+1 低开 < -5%
        open_pct = (d1_data['open'] - d1_data.get('pre_close', row.close)) / row.close
        if open_pct < -0.05: continue
            
        buy_price = d1_data['open']
        stop_price = buy_price * 0.90
        
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, '排名': row.排名,
            '潜龙分': row.潜龙分, '箱体振幅': f"{row.box_amplitude*100:.1f}%",
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
    
    # 7. 结果展示
    if trades:
        df_res = pd.DataFrame(trades)
        
        st.markdown(f"### 📊 突破策略回测 (Top {top_n})")
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
        
        st.markdown("### 🏆 潜龙榜 (箱体突破)")
        display_cols = ['信号日', '排名', '代码', '名称', '箱体振幅', '潜龙分', '状态'] + \
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
