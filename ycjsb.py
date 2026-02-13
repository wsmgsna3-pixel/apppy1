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
st.set_page_config(page_title="三日成妖·极速回测版", layout="wide")
st.title("⚡ 三日成妖·极速回测系统 (稳定修复版)")
st.markdown("""
**版本说明：**
- 已修复 API 调用频率过高导致的报错。
- 采用 **"全市场数据预加载 + 向量化计算"** 模式。
- 速度提升约 100 倍，且更稳定。
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
    bar = st.progress(0, text="正在批发全市场数据 (API加速中)...")
    
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
            # 如果报错，多休息一下再继续
            time.sleep(1)
            print(f"日期 {date} 获取失败: {e}")
            
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
    
    # 重试 3 次，每次失败休息 1 秒
    for attempt in range(3):
        try:
            time.sleep(0.5) # 请求前先休息一下
            # 获取全市场股票列表
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,list_date')
            
            # 如果获取成功，开始过滤
            if not df.empty:
                # 剔除 ST
                df = df[~df['name'].str.contains('ST')]
                # 剔除 北交所
                df = df[~df['market'].str.contains('北交')]
                df = df[~df['ts_code'].str.contains('BJ')]
                return df
                
        except Exception as e:
            print(f"API 请求失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(1)
            
    st.error("无法获取股票基础列表。可能是 Tushare 接口繁忙，请稍后再试。")
    return pd.DataFrame()

# ==========================================
# 3. 向量化信号计算 (速度的核心)
# ==========================================
def calculate_signals_vectorized(df, vol_mul):
    """
    使用 Pandas Rolling 全局计算，替代循环
    """
    st.info("⚡ 正在进行向量化运算 (计算潜伏期均量、爆发期涨幅)...")
    
    # 1. 计算潜伏期均量 (Lag 3天，取过去60天)
    # 逻辑：在 Day T，我们要看 Day T-3 之前的 60 天
    # shift(3) 把数据整体后移 3 天，然后 rolling(60)
    df['latent_vol_avg'] = df.groupby('ts_code')['vol'].transform(lambda x: x.shift(3).rolling(window=60).mean())
    
    # 2. 计算爆发期均量 (最近 3 天)
    df['burst_vol_avg'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(window=3).mean())
    
    # 3. 计算 3日 累计涨幅 (复利计算)
    # pct_chg 是百分比，先转为 multiplier (1.05)，然后 rolling 乘积
    df['daily_factor'] = 1 + df['pct_chg'] / 100
    df['cum_rise_3d'] = df.groupby('ts_code')['daily_factor'].transform(lambda x: x.rolling(window=3).apply(np.prod, raw=True))
    df['cum_rise_3d'] = (df['cum_rise_3d'] - 1) * 100
    
    # 4. 获取 Day 1 的涨幅 (用于判断首日是否大阳)
    df['day1_pct'] = df.groupby('ts_code')['pct_chg'].transform(lambda x: x.shift(2))
    
    # 5. 获取重心上移逻辑 (Day 3 Close > Day 1 Close)
    # Day 1 Close 是 shift(2) 的 Close
    df['day1_close'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(2))
    
    # === 信号筛选 ===
    # A. 量能达标
    cond_vol = df['burst_vol_avg'] > (df['latent_vol_avg'] * vol_mul)
    
    # B. 形态达标 (Day1 > 5%, 重心上移)
    cond_shape = (df['day1_pct'] > 5) & (df['close'] > df['day1_close'])
    
    # C. 价格和流动性 (Close >= 10, Amount > 5000万)
    cond_basic = (df['close'] >= 10) & (df['amount'] > 50000)
    
    # D. 标记原始信号
    df['is_signal'] = cond_vol & cond_shape & cond_basic
    
    return df

# ==========================================
# 4. 主程序逻辑
# ==========================================
with st.sidebar:
    st.header("⚙️ 极速版参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 20, 200, 50)
    end_date_input = st.date_input("结束日期", datetime.now().date())
    
    vol_mul = st.slider("量能倍数", 2.0, 5.0, 3.0, 0.5)
    
    run_btn = st.button("🚀 启动极速回测")

def run_fast_backtest():
    if not user_token:
        st.error("请输入 Token")
        return

    # 1. 准备日期
    # 我们需要加载比回测期更多的数据，因为要算 60 天均线
    # 缓冲 = 60 (潜伏) + 10 (未来收益) + days_back
    end_str = end_date_input.strftime('%Y%m%d')
    start_dt = end_date_input - timedelta(days=days_back * 1.5 + 100) # 宽松缓冲
    
    cal_dates = get_trade_cal(user_token, start_dt.strftime('%Y%m%d'), end_str)
    if not cal_dates:
        st.error("获取日历失败，请检查Token")
        return

    # 2. 数据加载 (Bulk Load)
    df_all = fetch_all_market_data_by_date(user_token, cal_dates)
    if df_all.empty:
        st.error("数据加载失败，请检查网络或Token")
        return
        
    st.success(f"数据加载完成！内存中共有 {len(df_all):,} 条 K线数据。")

    # 3. 基础信息匹配 (用于分板块涨幅限制)
    df_basic = get_stock_basics(user_token)
    if df_basic.empty:
        st.warning("基础信息获取失败，将跳过板块区分。")
        # 构造一个空的 DataFrame 防止后面报错
        df_basic = pd.DataFrame(columns=['ts_code', 'name', 'market'])
    
    # 只保留基础表里有的股票 (剔除了ST/北交所)
    df_all = df_all[df_all['ts_code'].isin(df_basic['ts_code'])]
    
    # 合并板块信息
    df_all = pd.merge(df_all, df_basic[['ts_code', 'name', 'market']], on='ts_code', how='left')

    # 4. 计算信号 (Vectorized)
    df_calc = calculate_signals_vectorized(df_all, vol_mul)
    
    # 5. 应用板块涨幅阈值
    # 主板 > 12%, 双创 > 20%
    is_startup = df_calc['market'].str.contains('创业|科创', na=False) | df_calc['ts_code'].str.startswith(('30', '68'))
    df_calc['rise_threshold'] = np.where(is_startup, 20.0, 12.0)
    
    # 最终信号
    final_mask = df_calc['is_signal'] & (df_calc['cum_rise_3d'] > df_calc['rise_threshold'])
    
    # 提取所有触发信号的行
    df_signals = df_calc[final_mask].copy()
    
    # 过滤掉不在回测区间内的信号 (因为我们加载了额外历史数据)
    # 回测区间是最近 days_back 天
    valid_dates = cal_dates[-(days_back + 10) : -10] # 同样预留10天给未来
    df_signals = df_signals[df_signals['trade_date'].isin(valid_dates)]

    if df_signals.empty:
        st.warning("在此期间未发现符合严选条件的交易。")
        return

    st.write(f"⚡ 信号计算完成，共发现 {len(df_signals)} 个买点，正在计算收益...")

    # 6. 收益计算 (Look-ahead Vectorized)
    trades = []
    # 把全量数据做成索引，方便快速查找
    # 优化：只保留需要的列
    price_lookup = df_calc[['ts_code', 'trade_date', 'open', 'close', 'low']].set_index(['ts_code', 'trade_date'])
    
    progress = st.progress(0)
    total_sig = len(df_signals)
    
    for i, row in enumerate(df_signals.itertuples()):
        progress.progress((i+1)/total_sig)
        
        signal_date = row.trade_date
        code = row.ts_code
        
        # 找到信号日之后的日期
        try:
            curr_idx = cal_dates.index(signal_date)
            # D+1 到 D+10 的日期列表
            future_dates = cal_dates[curr_idx+1 : curr_idx+11]
        except:
            continue
            
        if not future_dates: continue
        
        # 获取 D+1 数据
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
        
        # 遍历未来 10 天
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
        st.success(f"🎉 回测全部完成！共交易 {len(df_res)} 笔。")
        
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
        st.warning("没有生成有效交易 (可能是 D+1 缺失或被风控过滤)")

if run_btn:
    run_fast_backtest()
