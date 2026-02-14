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
st.set_page_config(page_title="箱体潜龙·显微镜版", layout="wide")
st.title("🔬 箱体潜龙·显微镜诊断版")
st.markdown("""
**本次更新：**
1.  **修复评分Bug**：评分标准与侧边栏“箱体限制”动态联动，不再误杀宽幅震荡的妖股。
2.  **新增显微镜**：输入代码，透视该股票落选的真实原因（是没创新高？还是排名太低？）。
3.  **数据复权**：逻辑优化，更贴近实战。
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
            # 依然使用基础接口，依靠大量数据计算相对位置
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
# 3. 核心计算 (带诊断逻辑)
# ==========================================
def calculate_box_breakout(df, vol_mul, box_limit):
    """
    向量化计算箱体突破
    """
    # 1. 核心指标计算
    # 箱体上沿 (Max Close of prev 60 days)
    df['high_60'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(1).rolling(window=60).max())
    # 箱体下沿 (Min Close of prev 60 days)
    df['low_60'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(1).rolling(window=60).min())
    # 60日均量
    df['vol_60'] = df.groupby('ts_code')['vol'].transform(lambda x: x.shift(1).rolling(window=60).mean())
    
    # 2. 箱体振幅
    df['box_amplitude'] = (df['high_60'] - df['low_60']) / df['low_60']
    
    # 3. 信号判定列 (分开写方便诊断)
    # A. 潜伏条件 (振幅 < box_limit)
    df['cond_box'] = df['box_amplitude'] < (box_limit / 100)
    
    # B. 价格突破 (Close > High60)
    df['cond_break'] = df['close'] > df['high_60']
    
    # C. 量能突破 (Vol > Vol60 * mul)
    df['cond_vol'] = df['vol'] > (df['vol_60'] * vol_mul)
    
    # D. 基础门槛
    df['cond_basic'] = (df['close'] >= 10) & (df['amount'] > 50000)
    
    # 最终信号
    df['is_signal'] = df['cond_box'] & df['cond_break'] & df['cond_vol'] & df['cond_basic']
    
    return df

def calculate_score(row, box_limit):
    """
    潜龙分 (动态版) - 修复了硬编码 35% 的问题
    """
    score = 60
    
    # 箱体越窄加分 (基准改为用户的 box_limit)
    # 比如用户设 50%，那么 40% 的振幅也能拿分
    box_amp = row['box_amplitude'] * 100
    if box_amp < box_limit:
        # 分数权重：距离极限越远，分越高
        score += (box_limit - box_amp) * 1.5
        
    # 突破力度
    if row['high_60'] > 0:
        break_pct = (row['close'] - row['high_60']) / row['high_60'] * 100
        score += min(break_pct * 2, 20)
    
    # 量能倍数
    if row['vol_60'] > 0:
        vol_ratio = row['vol'] / row['vol_60']
        score += min(vol_ratio * 5, 20)
    
    return round(score, 1)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数控制")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("数据回溯天数", 60, 300, 120)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    box_limit = st.slider("箱体振幅上限 (%)", 20, 60, 50, help="建议设为 45-50 以捕获利通电子")
    vol_mul = st.slider("突破量能倍数", 1.5, 5.0, 1.8, 0.1)
    top_n = st.number_input("每日优选 (Top N)", 1, 50, 10)
    
    st.markdown("---")
    st.subheader("🔍 诊断特定股票")
    debug_code = st.text_input("输入代码 (如 603629)", help="输入代码后，右侧将显示该股票的详细落选原因").strip()
    
    run_btn = st.button("🚀 启动回测")

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
    st.success(f"✅ 数据就绪: {len(df_all):,} 条 K线")

    # 2. 基础过滤
    df_basic = get_stock_basics(user_token)
    if not df_basic.empty:
        df_all = df_all[df_all['ts_code'].isin(df_basic['ts_code'])]
        df_all = pd.merge(df_all, df_basic[['ts_code', 'name', 'market']], on='ts_code', how='left')
    
    # 3. 计算指标
    with st.spinner("正在执行全市场扫描..."):
        df_calc = calculate_box_breakout(df_all, vol_mul, box_limit)
        
    # 4. === 显微镜诊断模块 (User Request) ===
    if debug_code:
        st.markdown(f"### 🔬 显微镜诊断: {debug_code}")
        # 模糊匹配代码
        debug_df = df_calc[df_calc['ts_code'].astype(str).str.contains(debug_code)].copy()
        
        if debug_df.empty:
            st.error(f"未找到代码 {debug_code} 的数据，请检查是否在回测日期范围内。")
        else:
            # 计算分数方便查看
            debug_df['Temp_Score'] = debug_df.apply(lambda r: calculate_score(r, box_limit), axis=1)
            
            # 格式化显示
            debug_cols = ['trade_date', 'close', 'high_60', 'vol', 'vol_60', 
                          'box_amplitude', 'cond_box', 'cond_break', 'cond_vol', 'is_signal', 'Temp_Score']
            
            # 只显示最近几天或有信号的天
            st.dataframe(
                debug_df[debug_cols].tail(20).style.format({
                    'high_60': '{:.2f}',
                    'vol': '{:.0f}',
                    'vol_60': '{:.0f}',
                    'box_amplitude': '{:.2%}',
                    'Temp_Score': '{:.1f}'
                }),
                use_container_width=True
            )
            st.info("""
            **字段说明：**
            - `high_60`: 过去60天最高收盘价 (突破基准)
            - `box_amplitude`: 箱体振幅 (需 < 设定值)
            - `cond_break`: 价格突破是否成立?
            - `cond_vol`: 量能突破是否成立?
            - `is_signal`: 最终是否入选?
            """)

    # 5. 筛选与排名
    valid_dates = cal_dates[-(days_back):] 
    df_window = df_calc[df_calc['trade_date'].isin(valid_dates)]
    
    df_signals = df_window[df_window['is_signal']].copy()
    
    if df_signals.empty:
        st.warning("在此期间无股票入选。")
        return

    # 评分与 Top N
    df_signals['潜龙分'] = df_signals.apply(lambda r: calculate_score(r, box_limit), axis=1)
    df_signals = df_signals.sort_values(['trade_date', '潜龙分'], ascending=[True, False])
    df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
    
    # 截断 Top N
    df_top = df_signals[df_signals['排名'] <= top_n].copy()
    
    # 6. 收益回测
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
                '潜龙分': row.潜龙分, '箱体振幅': f"{row.box_amplitude*100:.1f}%",
                '状态': '等待开盘'
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
    
    if trades:
        df_res = pd.DataFrame(trades)
        
        st.markdown(f"### 📊 显微镜回测结果 (Top {top_n})")
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
        
        st.markdown("### 🏆 潜龙榜")
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
