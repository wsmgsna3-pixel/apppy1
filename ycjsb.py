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
st.set_page_config(page_title="潜龙 V17·狙击手", layout="wide")
st.title("🐉 潜龙 V17·狙击手 (上帝指纹+资金爆破+板块共振)")
st.markdown("""
**策略核心：从 1000 个压缩到 50 个的极致筛选**
1.  **上帝指纹**：均线等距发散 (V16 核心，保留完美图形)。
2.  **资金爆破**：**量比 > 2.5** (必须是倍量启动，拒绝缩量骗线)。
3.  **人气基础**：**换手率 > 5%** (必须有活钱接力)。
4.  **板块共振**：**板块涨幅 > 1.2%** (拒绝逆势股)。
5.  **启动力度**：**当日涨幅 > 5.0%** (首日即大阳)。
6.  **风控铁律**：**D+1 亏损坚决离场，盈利则死拿 MA10**。
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
            # 获取指标(量比、换手)
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
def calculate_strategy(df_daily, df_basic):
    """
    V17 核心逻辑: 指纹 + 爆破 + 板块
    """
    # 1. 预处理板块数据
    if 'industry' not in df_daily.columns:
        df = pd.merge(df_daily, df_basic[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df = df_daily.copy()
        
    # 计算板块涨幅
    sector_stats = df.groupby(['trade_date', 'industry'])['pct_chg'].mean().reset_index()
    sector_stats.rename(columns={'pct_chg': 'sector_pct'}, inplace=True)
    df = pd.merge(df, sector_stats, on=['trade_date', 'industry'], how='left')

    # 2. 计算均线 (上帝指纹基础)
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma30'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(30).mean())
    
    # 3. 筛选逻辑
    
    # A. 完美排列
    cond_order = (df['close'] > df['ma5']) & \
                 (df['ma5'] > df['ma10']) & \
                 (df['ma10'] > df['ma20']) & \
                 (df['ma20'] > df['ma30'])
                 
    # B. 等距发散 (保留 V16 的严格标准 1.5)
    df['gap1'] = df['ma5'] - df['ma10']
    df['gap2'] = df['ma10'] - df['ma20']
    df['gap3'] = df['ma20'] - df['ma30']
    
    df['max_gap'] = df[['gap1', 'gap2', 'gap3']].max(axis=1)
    df['min_gap'] = df[['gap1', 'gap2', 'gap3']].min(axis=1)
    
    cond_spacing = (df['max_gap'] / (df['min_gap'] + 0.0001)) < 1.5
    
    # C. 资金爆破 (新增核心)
    # 量比 > 2.5 (必须倍量)
    # 换手率 > 5% (必须活跃)
    cond_money = (df['volume_ratio'] > 2.5) & (df['turnover_rate'] > 5.0)
    
    # D. 启动力度 (新增核心)
    # 涨幅 > 5.0% (大阳线确认)
    cond_power = df['pct_chg'] > 5.0
    
    # E. 板块共振 (新增核心)
    # 板块涨幅 > 1.2% (必须在上升板块中)
    cond_sector = df['sector_pct'] > 1.2
    
    # F. 贴线起爆 (保留 V16)
    # 股价距离 MA10 < 8% (稍微放宽一点点，因为大阳线可能拉开距离，但不能太远)
    cond_low = (df['close'] - df['ma10']) / df['ma10'] < 0.08
    
    # G. 首日启动
    df['is_perfect'] = cond_order & cond_spacing & cond_money & cond_power & cond_sector & cond_low
    # 实际上由于加了 volume_ratio > 2.5，这本身就是突发事件，不需要 prev_perfect 判定，
    # 因为很难连续两天量比都 > 2.5 且都满足条件。直接用 is_perfect 即可。
    
    df['is_signal'] = df['is_perfect']
    
    return df

def calculate_score(row):
    # 评分逻辑：量比和均匀度
    score = 60
    
    # 量比越大越好 (爆发力)
    if row['volume_ratio'] > 5.0: score += 20
    elif row['volume_ratio'] > 3.0: score += 10
    
    # 均匀度
    ratio = row['max_gap'] / (row['min_gap'] + 0.0001)
    if ratio < 1.2: score += 20
    
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V17 狙击手参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    top_n = st.number_input("每日优选 (Top N)", 1, 5, 1)
    
    run_btn = st.button("🚀 启动狙击")

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
    with st.spinner("正在执行狙击任务..."):
        df_calc = calculate_strategy(df_all, df_basic)
        
    # 4. 结果
    st.markdown("### 🐉 V17 诊断 (狙击手)")
    
    if df_calc.empty:
        st.warning("无信号。")
        return
        
    # 过滤时间窗
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    df_signals = df_window[df_window['is_signal']].copy()
    
    st.write(f"⚪ 捕获狙击目标: **{len(df_signals)}** 个")
    
    if df_signals.empty:
        st.warning("筛选条件极严，近期无目标。")
        return

    # 5. 评分与 Top N
    df_signals['潜龙分'] = df_signals.apply(calculate_score, axis=1)
    df_signals = df_signals.sort_values(['trade_date', '潜龙分'], ascending=[True, False])
    
    # 每日只取 Top 1
    df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
    df_top = df_signals[df_signals['排名'] <= top_n].copy()
    
    # 6. 回测 (D+1 止损策略)
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
            '量比': f"{row.volume_ratio:.1f}",
            '均匀度': f"{row.max_gap / (row.min_gap+0.0001):.2f}",
            '当日涨幅': f"{row.pct_chg:.1f}%",
            '买入价': buy_price, '状态': '持有'
        }
        
        # D+1 止损判定
        d1_close = d1_data['close']
        d1_ret = (d1_close - buy_price) / buy_price
        
        if d1_ret < 0:
            # D+1 亏损，立刻止损，后面天数收益全部锁定为 D+1 收益
            trade['状态'] = 'D+1止损'
            trade['D+1'] = round(d1_ret * 100, 2)
            for n in range(1, 10):
                trade[f"D+{n+1}"] = round(d1_ret * 100, 2)
        else:
            # D+1 盈利，开启 MA10 跟踪止盈
            trade['D+1'] = round(d1_ret * 100, 2)
            triggered = False
            
            for n in range(1, 10): # 从 D+2 开始
                if n >= len(future_dates): break
                f_date = future_dates[n]
                if (code, f_date) not in price_lookup.index: break
                f_data = price_lookup.loc[(code, f_date)]
                day_label = f"D+{n+1}"
                
                if not triggered:
                    # 检查是否跌破 MA10
                    if f_data['close'] < f_data['ma10']:
                        triggered = True
                        trade['状态'] = '破MA10止盈'
                        curr_ret = (f_data['close'] - buy_price) / buy_price * 100
                        trade[day_label] = round(curr_ret, 2)
                    else:
                        curr_ret = (f_data['close'] - buy_price) / buy_price * 100
                        trade[day_label] = round(curr_ret, 2)
                else:
                    # 已止盈，维持收益
                    trade[day_label] = trade.get(f"D+{n}", 0)
        
        trades.append(trade)
        
    progress.empty()
    
    if trades:
        df_res = pd.DataFrame(trades)
        
        st.markdown(f"### 📊 V17 (狙击手) 回测结果")
        cols = st.columns(5)
        days = ['D+1', 'D+3', 'D+5', 'D+7', 'D+10']
        
        for idx, d in enumerate(days):
            if d in df_res.columns:
                # 统计所有交易的平均收益 (包含止损单)
                avg_ret = df_res[d].mean()
                # 胜率只看 D+1 (因为D+1定生死)
                if d == 'D+1':
                    win_rate = (df_res[d] > 0).mean() * 100
                    cols[idx].metric(f"{d} 胜率", f"{win_rate:.1f}%")
                cols[idx].metric(f"{d} 均收", f"{avg_ret:.2f}%")
        
        st.dataframe(df_res.sort_values(['信号日'], ascending=False), use_container_width=True)
    else:
        st.warning("无交易")

if run_btn:
    run_analysis()
