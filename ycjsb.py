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
st.set_page_config(page_title="潜龙 V15·天道均线", layout="wide")
st.title("🐉 潜龙 V15·天道均线 (等距发散+完美排列)")
st.markdown("""
**策略核心：寻找均线的"几何美感" (无量能干扰)**
1.  **完美排列**：**股价 > MA5 > MA10 > MA20 > MA30** (绝对多头)。
2.  **等距发散 (您的发现)**：均线之间的距离大致相等 (筹码极度稳定，如仪仗队般整齐)。
3.  **角度共振**：四根均线全部向上抬头 (合力形成)。
4.  **首日启动**：昨日未形成此形态，今日**首次**形成 (抓主升浪起点)。
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
            # 既然不看量能，只需要日线行情
            df = pro.daily(trade_date=date)
            # 为了获取更准确的均线，最好有复权因子，但Tushare每日接口通常是不复权的
            # 这里直接用原始价格计算，短期内影响不大
            # 如果需要换手率辅助过滤停牌股，可以加 daily_basic
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
    V15 核心逻辑: 均线等距发散
    """
    # 1. 计算均线 (MA5, MA10, MA20, MA30)
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma30'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(30).mean())
    
    # 计算均线斜率 (今日MA - 昨日MA) > 0
    # 为了简化，直接比较 today vs yesterday
    df['ma5_slope'] = df.groupby('ts_code')['ma5'].diff()
    df['ma10_slope'] = df.groupby('ts_code')['ma10'].diff()
    df['ma20_slope'] = df.groupby('ts_code')['ma20'].diff()
    df['ma30_slope'] = df.groupby('ts_code')['ma30'].diff()
    
    # 2. 信号判定逻辑
    
    # A. 完美排列: Close > MA5 > MA10 > MA20 > MA30
    cond_order = (df['close'] > df['ma5']) & \
                 (df['ma5'] > df['ma10']) & \
                 (df['ma10'] > df['ma20']) & \
                 (df['ma20'] > df['ma30'])
    
    # B. 角度共振: 所有均线都在上涨
    cond_slope = (df['ma5_slope'] > 0) & \
                 (df['ma10_slope'] > 0) & \
                 (df['ma20_slope'] > 0) & \
                 (df['ma30_slope'] > 0)
    
    # C. 等距发散 (核心创新)
    # 计算间距
    df['gap1'] = df['ma5'] - df['ma10']
    df['gap2'] = df['ma10'] - df['ma20']
    df['gap3'] = df['ma20'] - df['ma30']
    
    # 判断间距是否"差不多"
    # 我们用最大间距和最小间距的比值来衡量。如果比值 < 2.0 (或更严 1.5)，说明很均匀
    # 比如 gap1=0.5, gap2=0.6, gap3=0.4 -> max=0.6, min=0.4 -> ratio=1.5 (均匀)
    # 如果 gap1=2.0, gap2=0.1 -> ratio=20 (不均匀，那是乖离过大或粘合)
    
    # 为了避免除以0，加个极小值
    df['max_gap'] = df[['gap1', 'gap2', 'gap3']].max(axis=1)
    df['min_gap'] = df[['gap1', 'gap2', 'gap3']].min(axis=1)
    
    # 门槛：均匀度 (Ratio < 2.5 比较宽松，< 1.5 非常严格)
    # 另外，gap必须大于0 (已经在cond_order里隐含了，因为MA5>MA10...)
    cond_spacing = (df['max_gap'] / (df['min_gap'] + 0.001)) < 2.5
    
    # 也可以加一个绝对距离限制，防止已经发散得太大(末期)
    # 比如 (MA5 - MA30) / MA30 不能超过 15% (刚启动)
    cond_early = (df['ma5'] - df['ma30']) / df['ma30'] < 0.15
    
    # D. 首日启动 (Yesterday NOT perfect)
    # 组合今日状态
    df['is_perfect'] = cond_order & cond_slope & cond_spacing & cond_early
    # 获取昨日状态
    df['prev_perfect'] = df.groupby('ts_code')['is_perfect'].shift(1).fillna(False)
    
    cond_start = df['is_perfect'] & (~df['prev_perfect'])
    
    # E. 基础过滤 (非ST，有成交量)
    cond_basic = (df['turnover_rate'] > 1.0) # 哪怕不看量比，也要有基本换手
    
    df['is_signal'] = cond_start & cond_basic
    
    return df

def calculate_score(row):
    # 评分逻辑：越均匀越好，角度越陡越好
    score = 60
    
    # 均匀度加分 (Ratio 越接近 1 越好)
    ratio = row['max_gap'] / (row['min_gap'] + 0.001)
    if ratio < 1.5: score += 20
    elif ratio < 2.0: score += 10
    
    # 涨幅加分 (当天最好是中阳线确认，>3%)
    if row['pct_chg'] > 3.0: score += 10
    if row['pct_chg'] > 5.0: score += 10
    
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V15 天道均线参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.subheader("🔥 筛选标准")
    top_n = st.number_input("每日优选 (Top N)", 1, 10, 3)
    
    run_btn = st.button("🚀 启动 V15 回测")

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
    with st.spinner("正在测量均线的几何角度..."):
        df_calc = calculate_strategy(df_all)
        
    # 4. 结果
    st.markdown("### 🐉 V15 诊断 (等距发散)")
    
    if df_calc.empty:
        st.warning("无信号。")
        return
        
    # 过滤时间窗
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    df_signals = df_window[df_window['is_signal']].copy()
    
    st.write(f"⚪ 捕获完美图形: **{len(df_signals)}** 个")
    
    if df_signals.empty:
        st.warning("近期无完美形态。")
        return

    # 5. 评分与 Top N
    df_signals['潜龙分'] = df_signals.apply(calculate_score, axis=1)
    df_signals = df_signals.sort_values(['trade_date', '潜龙分'], ascending=[True, False])
    
    # 每日取 Top N
    df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
    df_top = df_signals[df_signals['排名'] <= top_n].copy()
    
    # 6. 回测 (加入 MA10 止损逻辑)
    # 需要 lookup 包含 MA10
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
        
        # 初始止损: 买入价 - 5% (防止当天大面)
        # 移动止损: 收盘价跌破 MA10
        
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, 
            '行业': row.industry, 
            '均匀度': f"{row.max_gap / (row.min_gap+0.001):.1f}",
            '当日涨幅': f"{row.pct_chg:.1f}%",
            '买入价': buy_price, '状态': '持有'
        }
        
        triggered = False
        hold_days = 0
        
        for n, f_date in enumerate(future_dates):
            if (code, f_date) not in price_lookup.index: break
            f_data = price_lookup.loc[(code, f_date)]
            day_label = f"D+{n+1}"
            
            if not triggered:
                # 检查止损条件
                # 1. 硬止损: 亏 10%
                curr_ret = (f_data['close'] - buy_price) / buy_price
                if curr_ret < -0.10:
                    triggered = True
                    trade[day_label] = -10.0
                    trade['状态'] = '止损'
                    continue
                
                # 2. 趋势止损: 收盘跌破 MA10
                if f_data['close'] < f_data['ma10']:
                    triggered = True
                    # 以收盘价卖出
                    final_ret = (f_data['close'] - buy_price) / buy_price * 100
                    trade[day_label] = round(final_ret, 2)
                    trade['状态'] = '破线卖出'
                else:
                    # 继续持有
                    final_ret = (f_data['close'] - buy_price) / buy_price * 100
                    trade[day_label] = round(final_ret, 2)
            else:
                # 已卖出，保持最后状态
                trade[day_label] = trade.get(f"D+{n}", 0)
        
        trades.append(trade)
        
    progress.empty()
    
    if trades:
        df_res = pd.DataFrame(trades)
        
        st.markdown(f"### 📊 V15 (天道均线) 回测结果")
        cols = st.columns(5)
        days = ['D+1', 'D+3', 'D+5', 'D+7', 'D+10']
        
        for idx, d in enumerate(days):
            if d in df_res.columns:
                valid_data = df_res[pd.to_numeric(df_res[d], errors='coerce').notna()]
                if not valid_data.empty:
                    # 计算还在持有的胜率(大于0)
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
