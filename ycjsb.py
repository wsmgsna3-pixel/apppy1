import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import time
import warnings
import os

warnings.filterwarnings("ignore")

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="潜龙 V28·神射手", layout="wide")
st.title("🐉 潜龙 V28·神射手 (蓄势补漏+RSI精选)")
st.markdown("""
**策略核心：从"撒网"转为"狙击"**
1.  **双模启动 (补漏 5 只慢牛)**：
    * ⚡ **爆发模式**：单日涨幅 > 4.5% (V27逻辑)。
    * 🔋 **蓄势模式**：**3日累计涨幅 > 9%** 且 **创20日新高** (专抓利通电子/宏和科技类慢牛)。
2.  **RSI 优选 (解决信号过多)**：
    * 不再随机买入，而是按 **RSI 强度** 降序排列。
    * 优先买入 **RSI 高 (60-85)** 的品种，只取每日 **Top 3**。
3.  **趋势护航**：股价必须在 **MA5** 之上。
""")

DATA_FILE = "market_data_store.csv"

# ==========================================
# 2. 核心数据引擎 (增量更新)
# ==========================================
def get_trade_cal(pro, start_date, end_date):
    try:
        df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
        return sorted(df['cal_date'].tolist())
    except:
        return []

def sync_market_data(token, start_date, end_date):
    if not token:
        return pd.DataFrame(), "请先输入Token"
    ts.set_token(token)
    pro = ts.pro_api()
    target_dates = get_trade_cal(pro, start_date, end_date)
    if not target_dates: return pd.DataFrame(), "无法获取交易日历"
    
    existing_dates = set()
    if os.path.exists(DATA_FILE):
        try:
            df_dates = pd.read_csv(DATA_FILE, usecols=['trade_date'], dtype={'trade_date': str})
            existing_dates = set(df_dates['trade_date'].unique())
        except: pass
            
    missing_dates = sorted(list(set(target_dates) - existing_dates))
    
    if missing_dates:
        st.info(f"发现 {len(missing_dates)} 个新交易日，增量更新中...")
        progress_bar = st.progress(0)
        new_data = []
        batch_size = 5
        for i, date in enumerate(missing_dates):
            try:
                df_daily = pro.daily(trade_date=date)
                df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,volume_ratio,circ_mv')
                if not df_daily.empty and not df_basic.empty:
                    df_merged = pd.merge(df_daily, df_basic, on='ts_code', how='left')
                    df_merged['trade_date'] = str(date)
                    new_data.append(df_merged)
            except: time.sleep(1)
            progress_bar.progress((i + 1) / len(missing_dates))
            if len(new_data) >= batch_size or (i == len(missing_dates) - 1):
                if new_data:
                    df_batch = pd.concat(new_data)
                    mode = 'a' if os.path.exists(DATA_FILE) else 'w'
                    header = not os.path.exists(DATA_FILE)
                    df_batch.to_csv(DATA_FILE, mode=mode, header=header, index=False)
                    new_data = []
        progress_bar.empty()
        
    if os.path.exists(DATA_FILE):
        dtype_dict = {'ts_code': str, 'trade_date': str}
        df_all = pd.read_csv(DATA_FILE, dtype=dtype_dict)
        df_all = df_all[(df_all['trade_date'] >= start_date) & (df_all['trade_date'] <= end_date)]
        df_all = df_all.drop_duplicates(subset=['ts_code', 'trade_date'])
        
        @st.cache_data
        def get_stock_info():
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,industry')
            return df[~df['name'].str.contains('ST')]
        df_info = get_stock_info()
        return df_all, df_info
    else:
        return pd.DataFrame(), "无数据"

# ==========================================
# 3. 策略逻辑 (双模启动)
# ==========================================
def calculate_strategy(df_all, df_info):
    if 'industry' not in df_all.columns:
        df = pd.merge(df_all, df_info[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df = df_all.copy()
    
    # === 指标计算 ===
    # RSI (6日)
    def calc_rsi(x):
        delta = x.diff()
        gain = (delta.where(delta > 0, 0)).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        rs = gain / (loss + 0.001)
        return 100 - (100 / (1 + rs))
    
    df['rsi_6'] = df.groupby('ts_code')['close'].transform(calc_rsi)
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    
    # 3日累计涨幅
    df['pct_3d'] = df.groupby('ts_code')['pct_chg'].transform(lambda x: x.rolling(3).sum())
    # 20日最高价 (用于判断新高)
    df['high_20'] = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(20).max())
    
    # === 1. 启动模式 ===
    # A模式: 单日爆发 (V27)
    cond_mode_a = df['pct_chg'] > 4.5
    
    # B模式: 蓄势突破 (补漏)
    # 3天涨9% + 且今日收盘价接近20日新高(>98%)
    cond_mode_b = (df['pct_3d'] > 9.0) & (df['close'] >= df['high_20'] * 0.98)
    
    cond_start = cond_mode_a | cond_mode_b
    
    # === 2. 基础门槛 ===
    # 趋势: 站上MA5 (强势底线)
    cond_trend = df['close'] > df['ma5']
    # RSI: > 45 (非超卖)
    cond_rsi = df['rsi_6'] > 45
    # 市值: 30-800亿
    cond_mv = (df['circ_mv'] >= 30*10000) & (df['circ_mv'] <= 800*10000)
    
    # 综合信号
    df['is_signal'] = cond_start & cond_trend & cond_rsi & cond_mv
    
    # 标记模式
    df['mode'] = np.where(cond_mode_a, 'A:爆发', 'B:蓄势')
    
    return df

# ==========================================
# 4. 回测逻辑 (MA10 趋势止盈)
# ==========================================
def run_backtest_sniper(df_signals, df_all, cal_dates):
    df_lookup = df_all.copy()
    if 'ma10' not in df_lookup.columns:
         df_lookup['ma10'] = df_lookup.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    
    price_lookup = df_lookup[['ts_code', 'trade_date', 'open', 'close', 'low', 'ma10', 'pre_close']].set_index(['ts_code', 'trade_date'])
    
    trades = []
    
    for row in df_signals.itertuples():
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
        
        # 铁门槛
        open_pct = (d1_data['open'] - d1_data['pre_close']) / d1_data['pre_close'] * 100
        if open_pct < -2.0: continue
        
        buy_price = d1_data['open']
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, 
            '行业': row.industry, '买入价': buy_price, '模式': row.mode,
            '开盘涨幅': f"{open_pct:.2f}%", '状态': '持有'
        }
        
        # D+1 止损
        d1_ret = (d1_data['close'] - buy_price) / buy_price
        
        if d1_ret < -0.05:
             trade['状态'] = 'D+1止损'
             trade['D+1'] = round(d1_ret * 100, 2)
             for n in range(1, 10): trade[f"D+{n+1}"] = round(d1_ret * 100, 2)
        else:
            trade['D+1'] = round(d1_ret * 100, 2)
            triggered = False
            
            # 趋势跟踪: 破 MA10 止盈
            for n in range(1, 10):
                if n >= len(future_dates): break
                f_date = future_dates[n]
                if (code, f_date) not in price_lookup.index: break
                f_data = price_lookup.loc[(code, f_date)]
                day_key = f"D+{n+1}"
                
                if not triggered:
                    if f_data['close'] < f_data['ma10']:
                        triggered = True
                        trade['状态'] = '破MA10止盈'
                    curr_ret = (f_data['close'] - buy_price) / buy_price * 100
                    trade[day_key] = round(curr_ret, 2)
                else:
                    trade[day_key] = trade.get(f"D+{n}", 0)
        
        trades.append(trade)
        
    return pd.DataFrame(trades)

# ==========================================
# 5. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V28 神射手")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 150, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.info("🎯 优选策略")
    st.markdown("""
    * **排序**: 按 RSI(6) 降序
    * **逻辑**: 优先买入攻击性最强的品种
    * **补漏**: 包含3日累计大涨的慢牛
    """)
    top_n = st.number_input("每日优选 (Top N)", 1, 10, 3) # 收缩到 Top 3
    
    run_btn = st.button("🚀 启动狙击")

if run_btn:
    if not user_token:
        st.error("请先输入 Token")
    else:
        end_str = end_date_input.strftime('%Y%m%d')
        start_dt = end_date_input - timedelta(days=days_back * 1.5 + 80)
        start_str = start_dt.strftime('%Y%m%d')
        
        res, info = sync_market_data(user_token, start_str, end_str)
        
        if isinstance(info, pd.DataFrame):
            df_info = info
            df_all = res
            st.success(f"✅ 数据加载: {len(df_all):,} 行")
            
            with st.spinner("神射手瞄准中..."):
                df_calc = calculate_strategy(df_all, df_info)
                
            cal_dates = sorted(df_calc['trade_date'].unique())
            valid_dates = cal_dates[-(days_back):]
            
            df_signals = df_calc[(df_calc['trade_date'].isin(valid_dates)) & (df_calc['is_signal'])].copy()
            
            # === 核心排序逻辑 ===
            # 按 RSI 降序排列 (优先选 70-80 的强攻击形态)
            df_signals = df_signals.sort_values(['trade_date', 'rsi_6'], ascending=[True, False])
            
            df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
            df_top = df_signals[df_signals['排名'] <= top_n].copy()
            
            st.write(f"⚪ 狙击信号: **{len(df_top)}** 个 (每日 Top {top_n})")
            
            if not df_top.empty:
                df_res = run_backtest_sniper(df_top, df_calc, cal_dates)
                
                if not df_res.empty:
                    st.success(f"🎯 成交单数: **{len(df_res)}**")
                    
                    st.markdown(f"### 📊 V28 回测 (RSI优选 Top{top_n})")
                    cols = st.columns(5)
                    days = ['D+1', 'D+3', 'D+5', 'D+7', 'D+10']
                    for idx, d in enumerate(days):
                         if d in df_res.columns:
                             avg = df_res[d].mean()
                             if d == 'D+1':
                                 rate = (df_res[d] > 0).mean() * 100
                                 cols[idx].metric(f"{d} 胜率", f"{rate:.1f}%")
                             cols[idx].metric(f"{d} 均收", f"{avg:.2f}%")
                    
                    st.dataframe(df_res.sort_values(['信号日'], ascending=False), use_container_width=True)
                else:
                    st.warning("无成交。")
            else:
                st.warning("无信号。")
        else:
            st.error(info)
