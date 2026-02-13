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
st.set_page_config(page_title="三日成妖·Top5实战版", layout="wide")
st.title("🐉 三日成妖·Top5 实战优选系统")
st.markdown("""
**本次更新特性：**
1. **每日限额**：每天仅选取**潜龙分最高的前 5 名**，模拟真实仓位管理。
2. **名次标注**：新增`排名`列，龙一龙二一目了然。
3. **实战逻辑**：解决“票太多买不过来”的痛点，集中火力打龙头。
""")

# ==========================================
# 2. 核心数据引擎
# ==========================================
@st.cache_data(persist="disk", show_spinner=False)
def get_trade_cal(token, start_date, end_date):
    """获取交易日历 (强制升序)"""
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
    """批量拉取全市场数据"""
    ts.set_token(token)
    pro = ts.pro_api()
    
    data_list = []
    total = len(date_list)
    bar = st.progress(0, text="正在同步全市场数据...")
    
    for i, date in enumerate(date_list):
        try:
            time.sleep(0.05) # 限流保护
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
    """获取基础信息"""
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
# 3. 向量化计算 (含评分系统)
# ==========================================
def calculate_signals_vectorized(df):
    # 潜伏期 (过去60天)
    df['latent_vol_avg'] = df.groupby('ts_code')['vol'].transform(lambda x: x.shift(3).rolling(window=60).mean())
    # 爆发期 (最近3天)
    df['burst_vol_avg'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(window=3).mean())
    # 3日累计涨幅
    df['daily_factor'] = 1 + df['pct_chg'] / 100
    df['cum_rise_3d'] = df.groupby('ts_code')['daily_factor'].transform(lambda x: x.rolling(window=3).apply(np.prod, raw=True))
    df['cum_rise_3d'] = (df['cum_rise_3d'] - 1) * 100
    
    # Day 1 数据
    df['day1_pct'] = df.groupby('ts_code')['pct_chg'].transform(lambda x: x.shift(2))
    df['day1_close'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(2))
    
    return df

def calculate_score(row, vol_mul):
    """
    潜龙分计算公式：
    1. 量能得分：爆发倍数越大越好 (上限 5倍)
    2. 形态得分：Day1 涨幅越大越好
    3. 稳定性：Day3 涨幅适中最好 (防止直接高潮)
    """
    # 基础分
    score = 60
    
    # 量能加分 (每多1倍加10分)
    # 防止除以0
    l_vol = row['latent_vol_avg'] if row['latent_vol_avg'] > 0 else 1
    actual_mul = row['burst_vol_avg'] / l_vol
    score += min((actual_mul - vol_mul) * 10, 30)
    
    # Day1 强度加分
    score += min(row['day1_pct'], 10)
    
    # 3日涨幅加分 (越高越好，但双创板有加成)
    score += min(row['cum_rise_3d'] / 2, 20)
    
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数控制台")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("数据回溯天数", 20, 200, 60, help="设大一点可以覆盖更多历史，避免反复下载")
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    vol_mul = st.slider("量能倍数", 1.5, 5.0, 2.0, 0.1)
    
    # 新增 Top N 参数
    top_n = st.number_input("每日优选数量 (Top N)", min_value=1, max_value=20, value=5, help="每天只选分数最高的前N名")
    
    run_btn = st.button("🚀 启动实战扫描")

def run_analysis():
    if not user_token:
        st.error("请先输入 Token")
        return

    # 1. 准备数据
    end_str = end_date_input.strftime('%Y%m%d')
    # 缓冲: 60天潜伏 + 15天未来 + 回测天数
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
    else:
        df_all['market'] = '主板' # 兜底

    # 3. 计算指标
    with st.spinner("正在扫描全市场信号..."):
        df_calc = calculate_signals_vectorized(df_all)
    
    # 4. 筛选逻辑
    # 板块阈值
    is_startup = df_calc['market'].str.contains('创业|科创', na=False) | df_calc['ts_code'].str.startswith(('30', '68'))
    df_calc['rise_threshold'] = np.where(is_startup, 20.0, 12.0)
    
    # 筛选条件
    c1 = (df_calc['close'] >= 10) & (df_calc['amount'] > 50000) & (df_calc['latent_vol_avg'] > 0)
    c2 = df_calc['burst_vol_avg'] > (df_calc['latent_vol_avg'] * vol_mul)
    c3 = (df_calc['day1_pct'] > 5) & (df_calc['close'] > df_calc['day1_close'])
    c4 = df_calc['cum_rise_3d'] > df_calc['rise_threshold']
    
    # 提取信号 (只看用户指定的回测区间)
    valid_dates = cal_dates[-(days_back):] 
    df_signals = df_calc[c1 & c2 & c3 & c4 & df_calc['trade_date'].isin(valid_dates)].copy()

    if df_signals.empty:
        st.warning("🔍 未发现符合条件的股票。请尝试降低【量能倍数】。")
        return

    st.info(f"⚡ 初步发现 {len(df_signals)} 个信号，正在进行评分与Top{top_n}截断...")
    
    # 5. 评分与 Top N 截断
    # 计算得分
    df_signals['潜龙分'] = df_signals.apply(lambda row: calculate_score(row, vol_mul), axis=1)
    
    # === 核心修改：每日排序 + 截断 ===
    # 按日期分组，组内按分数降序排列
    df_signals = df_signals.sort_values(['trade_date', '潜龙分'], ascending=[True, False])
    
    # 生成排名 (1, 2, 3...)
    df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
    
    # 只保留排名前 N 的
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
        
        # 寻找 D+1 ~ D+10
        try:
            curr_idx = cal_dates.index(signal_date)
            future_dates = cal_dates[curr_idx+1 : curr_idx+11]
        except: continue
            
        # 如果没有未来数据 (等待开盘)
        if not future_dates:
            trades.append({
                '信号日': signal_date, '代码': code, '名称': row.name, '排名': row.排名,
                '潜龙分': row.潜龙分, '3日涨幅': round(row.cum_rise_3d, 1),
                '状态': '等待开盘'
            })
            continue
            
        d1_date = future_dates[0]
        if (code, d1_date) not in price_lookup.index: continue
        d1_data = price_lookup.loc[(code, d1_date)]
        
        # 风控: D+1 低开 < -5%
        open_pct = (d1_data['open'] - d1_data.get('pre_close', row.close)) / row.close
        if open_pct < -0.05:
            continue
            
        buy_price = d1_data['open']
        stop_price = buy_price * 0.90
        
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, '排名': row.排名,
            '潜龙分': row.潜龙分, '3日涨幅': round(row.cum_rise_3d, 1),
            '买入价': buy_price, '状态': '持有'
        }
        
        # 遍历现有数据计算收益
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
        
        # A. 统计区
        st.markdown(f"### 📊 回测统计 (仅统计 Top {top_n})")
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
                else:
                    cols[idx].metric(f"{d}", "无数据")
        
        # B. 详细列表
        st.markdown("### 🏆 潜龙榜 (每日精选)")
        
        display_cols = ['信号日', '排名', '代码', '名称', '潜龙分', '3日涨幅', '状态'] + \
                       [d for d in days if d in df_res.columns]
        
        st.dataframe(
            df_res[display_cols].sort_values(['信号日', '排名'], ascending=[False, True]),
            use_container_width=True,
            height=600
        )
    else:
        st.warning("无有效交易（可能被风控拦截或无 Top 名额）。")

if run_btn:
    run_analysis()
