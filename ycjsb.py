import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 页面配置与全局设置
# ==========================================
st.set_page_config(page_title="三日成妖·诊断回测版", layout="wide")
st.title("🕵️‍♀️ 三日成妖·诊断回测系统 (含漏斗分析)")
st.markdown("""
**使用技巧：**
1. **关于缓存**：修改“回测天数”会导致重新下载数据（这是必须的）。建议一次性设为 **100天**，下载完成后，再反复调整“量能倍数”，此时不会触发下载，以此实现“秒级调参”。
2. **关于诊断**：向下滚动查看【筛选漏斗诊断】区域，观察每一步剔除了多少股票。
""")

# ==========================================
# 2. 核心数据引擎 (带重试与限流)
# ==========================================
@st.cache_data(persist="disk", show_spinner=False)
def get_trade_cal(token, start_date, end_date):
    """获取交易日历 (带重试)"""
    ts.set_token(token)
    pro = ts.pro_api()
    for attempt in range(3):
        try:
            df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
            if not df.empty:
                return df['cal_date'].tolist()
            time.sleep(0.5)
        except:
            time.sleep(1)
    return []

@st.cache_data(persist="disk", show_spinner=False)
def fetch_all_market_data_by_date(token, date_list):
    """
    批量拉取全市场数据 (核心加速环节 + 限流保护)
    """
    ts.set_token(token)
    pro = ts.pro_api()
    
    data_list = []
    total = len(date_list)
    
    # 进度条
    bar = st.progress(0, text="正在批发全市场数据 (首次运行需下载，请耐心等待)...")
    
    for i, date in enumerate(date_list):
        try:
            # === 核心修复：每次请求前暂停 0.08 秒，防止 QPS 超限 ===
            time.sleep(0.08)
            
            # 一次性拉取当天所有股票
            df = pro.daily(trade_date=date)
            
            # 只保留核心字段减小内存
            if not df.empty:
                df = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount', 'pct_chg']]
                data_list.append(df)
        except Exception as e:
            time.sleep(1)
            # print(f"日期 {date} 获取失败: {e}")
            
        if (i+1) % 5 == 0:
            bar.progress((i+1)/total, text=f"加载数据: {date} ({i+1}/{total})")
            
    bar.empty()
    
    if not data_list:
        return pd.DataFrame()
        
    # 合并为一个巨大的 DataFrame
    full_df = pd.concat(data_list)
    # 按股票和日期排序，为 rolling 计算做准备
    full_df = full_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    return full_df

@st.cache_data(persist="disk", show_spinner=False)
def get_stock_basics(token):
    """
    获取基础信息 (带重试机制，防止 API 报错)
    """
    ts.set_token(token)
    pro = ts.pro_api()
    
    for attempt in range(3):
        try:
            time.sleep(0.5) 
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,list_date')
            if not df.empty:
                # 剔除 ST / 北交所
                df = df[~df['name'].str.contains('ST')]
                df = df[~df['market'].str.contains('北交')]
                df = df[~df['ts_code'].str.contains('BJ')]
                return df
        except Exception as e:
            time.sleep(1)
            
    st.error("无法获取股票基础列表。可能是 Tushare 接口繁忙，请稍后再试。")
    return pd.DataFrame()

# ==========================================
# 3. 向量化信号计算 (含诊断逻辑)
# ==========================================
def calculate_signals_vectorized(df):
    """
    基础指标计算 (不包含参数过滤，方便后续反复调参)
    """
    # 1. 计算潜伏期均量 (Lag 3天，取过去60天)
    df['latent_vol_avg'] = df.groupby('ts_code')['vol'].transform(lambda x: x.shift(3).rolling(window=60).mean())
    
    # 2. 计算爆发期均量 (最近 3 天)
    df['burst_vol_avg'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(window=3).mean())
    
    # 3. 计算 3日 累计涨幅 (复利计算)
    df['daily_factor'] = 1 + df['pct_chg'] / 100
    df['cum_rise_3d'] = df.groupby('ts_code')['daily_factor'].transform(lambda x: x.rolling(window=3).apply(np.prod, raw=True))
    df['cum_rise_3d'] = (df['cum_rise_3d'] - 1) * 100
    
    # 4. 获取 Day 1 的涨幅
    df['day1_pct'] = df.groupby('ts_code')['pct_chg'].transform(lambda x: x.shift(2))
    
    # 5. 获取重心上移逻辑
    df['day1_close'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(2))
    
    return df

# ==========================================
# 4. 主程序逻辑
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    user_token = st.text_input("Tushare Token:", type="password")
    
    # 提示用户关于缓存的逻辑
    st.info("💡 提示：修改'回测天数'或'结束日期'会触发重新下载数据。修改'量能倍数'则是秒出结果。")
    
    days_back = st.slider("回测天数", 20, 200, 50)
    end_date_input = st.date_input("结束日期", datetime.now().date())
    
    st.markdown("---")
    vol_mul = st.slider("量能倍数", 1.5, 5.0, 2.0, 0.1, help="建议从2.0开始尝试")
    
    run_btn = st.button("🚀 启动回测 (带诊断)")

def run_diagnostic_backtest():
    if not user_token:
        st.error("请输入 Token")
        return

    # 1. 准备日期
    end_str = end_date_input.strftime('%Y%m%d')
    start_dt = end_date_input - timedelta(days=days_back * 1.5 + 100) # 宽松缓冲
    
    cal_dates = get_trade_cal(user_token, start_dt.strftime('%Y%m%d'), end_str)
    if not cal_dates:
        st.error("获取日历失败，请检查Token")
        return

    # 2. 数据加载 (Cached)
    # 只要 cal_dates 不变，这里就会直接读取磁盘缓存，不会重新下载
    df_all = fetch_all_market_data_by_date(user_token, cal_dates)
    if df_all.empty:
        st.error("数据加载失败，请检查网络或Token")
        return
        
    st.success(f"✅ 数据准备就绪！内存中共有 {len(df_all):,} 条 K线数据。")

    # 3. 基础信息匹配
    df_basic = get_stock_basics(user_token)
    if df_basic.empty:
        df_basic = pd.DataFrame(columns=['ts_code', 'name', 'market'])
    
    # 只保留基础表里有的股票
    df_all = df_all[df_all['ts_code'].isin(df_basic['ts_code'])]
    df_all = pd.merge(df_all, df_basic[['ts_code', 'name', 'market']], on='ts_code', how='left')

    # 4. 计算指标 (Vectorized)
    with st.spinner("正在计算全市场指标..."):
        df_calc = calculate_signals_vectorized(df_all)
    
    # 5. 应用板块涨幅阈值
    is_startup = df_calc['market'].str.contains('创业|科创', na=False) | df_calc['ts_code'].str.startswith(('30', '68'))
    df_calc['rise_threshold'] = np.where(is_startup, 20.0, 12.0)
    
    # ==========================================
    # 🕵️‍♀️ 核心诊断漏斗 (Funnel Analysis)
    # ==========================================
    st.markdown("### 🕵️‍♀️ 筛选漏斗诊断 (Diagnostic Funnel)")
    st.caption("👇 观察下方数据，找出哪一步筛选条件过于严苛")
    
    # 步骤 1: 基础池
    # 必须有 latent_vol_avg (说明数据足够长能算出均线)
    cond_valid_data = df_calc['latent_vol_avg'].notna()
    cond_basic = (df_calc['close'] >= 10) & (df_calc['amount'] > 50000)
    
    # 仅统计在回测区间内的日期
    valid_dates = cal_dates[-(days_back + 10) : -10]
    df_in_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    # 实时计算漏斗
    step0 = len(df_in_window)
    st.write(f"⚪ **初始样本**: {step0:,} 条 K线 (在回测区间内)")
    
    step1 = len(df_in_window[cond_basic])
    st.write(f"1️⃣ **基础门槛** (价>10, 量>5000万): 剩余 **{step1:,}** 条")
    
    # 步骤 2: 量能关
    cond_vol = df_in_window['burst_vol_avg'] > (df_in_window['latent_vol_avg'] * vol_mul)
    step2 = len(df_in_window[cond_basic & cond_vol])
    st.write(f"2️⃣ **量能筛选** (>{vol_mul}倍): 剩余 **{step2:,}** 条 (关键瓶颈)")
    
    # 步骤 3: 形态关
    cond_shape = (df_in_window['day1_pct'] > 5) & (df_in_window['close'] > df_in_window['day1_close'])
    step3 = len(df_in_window[cond_basic & cond_vol & cond_shape])
    st.write(f"3️⃣ **形态筛选** (Day1大阳+重心上移): 剩余 **{step3:,}** 条")
    
    # 步骤 4: 涨幅关
    cond_rise = df_in_window['cum_rise_3d'] > df_in_window['rise_threshold']
    final_mask = cond_basic & cond_vol & cond_shape & cond_rise
    
    df_signals = df_in_window[final_mask].copy()
    st.write(f"4️⃣ **涨幅筛选** (主板>12%/双创>20%): 最终买点 **{len(df_signals)}** 个")

    if df_signals.empty:
        st.error("❌ 结果为空！请调低【量能倍数】重试（无需重新下载数据）。")
        return

    st.success(f"⚡ 发现 {len(df_signals)} 个有效交易信号，正在计算最终收益...")

    # 6. 收益计算
    trades = []
    # 优化：只保留需要的列建立索引
    price_lookup = df_calc[['ts_code', 'trade_date', 'open', 'close', 'low']].set_index(['ts_code', 'trade_date'])
    
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
        
        # 风控：D+1 低开 < -5%
        if (d1_data['open'] - d1_data.get('pre_close', row.close)) / row.close < -0.05:
            continue
            
        buy_price = d1_data['open']
        stop_price = buy_price * 0.90
        
        trade = {
            '信号日': signal_date,
            '代码': code,
            '名称': row.name,
            '3日涨幅': round(row.cum_rise_3d, 1),
            '买入价': buy_price,
            '状态': '持有'
        }
        
        triggered = False
        for n, f_date in enumerate(future_dates):
            if (code, f_date) not in price_lookup.index: break
            f_data = price_lookup.loc[(code, f_date)]
            
            if not triggered:
                if f_data['low'] <= stop_price:
                    triggered = True
                    trade['状态'] = '止损'
                    ret = -10.0
                else:
                    ret = (f_data['close'] - buy_price) / buy_price * 100
            else:
                ret = -10.0
            
            day_label = f"D+{n+1}"
            if day_label in ['D+1', 'D+3', 'D+5', 'D+7', 'D+10']:
                trade[day_label] = round(ret, 2)
        
        trades.append(trade)
        
    progress.empty()
    
    if trades:
        df_res = pd.DataFrame(trades)
        st.markdown("### 📈 回测结果分析")
        
        cols = st.columns(5)
        days = ['D+1', 'D+3', 'D+5', 'D+7', 'D+10']
        for idx, d in enumerate(days):
            if d in df_res.columns:
                win = len(df_res[df_res[d]>0]) / len(df_res) * 100
                avg = df_res[d].mean()
                cols[idx].metric(f"{d} 胜率", f"{win:.1f}%")
                cols[idx].metric(f"{d} 均收", f"{avg:.2f}%")
        
        st.dataframe(df_res.sort_values('信号日', ascending=False))
    else:
        st.warning("所有信号均被 D+1 低开风控拦截，无实际成交。")

if run_btn:
    run_diagnostic_backtest()
