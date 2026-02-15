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
st.set_page_config(page_title="潜龙 V29·时光倒流", layout="wide")
st.title("🐉 潜龙 V29·时光倒流 (替补策略提前版)")
st.markdown("""
**策略核心：基于"替补策略"的基因，但拒绝"完美图形"的诱惑**
1.  **均线松绑**：**取消 60日线/30日线 限制**。只要站上 **20日线** (生命线)，不管上面有没有压力，先上车。
2.  **RSI 抢跑**：**关注 RSI 50-75 区间**。抓"主升浪启动前夜"，而不是等 RSI>80 (鱼尾) 再去接盘。
3.  **获利盘下沉**：**获利盘 > 50% 即可**。不需要 90% 的人赚钱，只要主力开始赚钱就行。
4.  **力度宽容**：**涨幅 > 3.5%**。包容那些"温和启动"的慢牛。
5.  **目标**：比原版替补策略 **提前 3-5 天** 发出信号，吃掉那 30% 的利润。
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
                df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,volume_ratio,circ_mv,pe,pb')
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
# 3. 策略逻辑 (替补策略 - 抢跑版)
# ==========================================
def calculate_strategy(df_all, df_info):
    if 'industry' not in df_all.columns:
        df = pd.merge(df_all, df_info[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df = df_all.copy()
    
    # === 基础指标 ===
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    
    # RSI (6日)
    def calc_rsi(x):
        delta = x.diff()
        gain = (delta.where(delta > 0, 0)).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        rs = gain / (loss + 0.001)
        return 100 - (100 / (1 + rs))
    df['rsi_6'] = df.groupby('ts_code')['close'].transform(calc_rsi)
    
    # 模拟获利盘 (Winner Rate)
    # 原版是用 Tushare 的 pro.daily_basic 里的数据，这里我们用一个简化的算法估算
    # 算法：(当前价 - 20日最低价) / (20日最高价 - 20日最低价) * 100
    # 这虽然不精确，但在趋势判断上足够用了
    df['low_20'] = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(20).min())
    df['high_20'] = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(20).max())
    df['winner_rate'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 0.0001) * 100
    
    # === 筛选逻辑 (Time Machine) ===
    
    # 1. 均线: 只要站上 MA20 (生命线) 且 MA5 拐头向上
    cond_trend = (df['close'] > df['ma20']) & (df['ma5'] > df['ma5'].shift(1))
    
    # 2. 涨幅: > 3.5% (中阳线，比原版的 5.5% 宽松，为了抓慢牛)
    cond_up = df['pct_chg'] > 3.5
    
    # 3. RSI: 50 - 75 (黄金起步区)
    # 原版喜欢 > 80，我们这里刻意限制 < 80，拒绝鱼尾
    cond_rsi = (df['rsi_6'] > 50) & (df['rsi_6'] < 80)
    
    # 4. 获利盘: > 50% (原版要求 90%)
    cond_winner = df['winner_rate'] > 50
    
    # 5. 市值: 30-800亿 (您的自选股区间)
    cond_mv = (df['circ_mv'] >= 30*10000) & (df['circ_mv'] <= 800*10000)
    
    # 综合信号
    df['is_signal'] = cond_trend & cond_up & cond_rsi & cond_winner & cond_mv
    
    # === 评分系统 (用于 Top N) ===
    # 我们不仅要选出来，还要给它们排个序
    # 核心思想：位置越低、爆发力越强越好
    # 评分 = RSI(攻击力) * 2 - 获利盘(位置风险)
    # 逻辑：RSI 高说明强，获利盘低说明位置低，两者结合就是"低位强攻击"
    df['score'] = df['rsi_6'] * 2 - df['winner_rate']
    
    return df

# ==========================================
# 4. 回测逻辑 (MA10 拿住)
# ==========================================
def run_backtest_tm(df_signals, df_all, cal_dates):
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
        
        open_pct = (d1_data['open'] - d1_data['pre_close']) / d1_data['pre_close'] * 100
        if open_pct < -2.0: continue
        
        buy_price = d1_data['open']
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, 
            '行业': row.industry, '买入价': buy_price, 
            'RSI': f"{row.rsi_6:.1f}", '获利盘': f"{row.winner_rate:.0f}%",
            '状态': '持有'
        }
        
        d1_ret = (d1_data['close'] - buy_price) / buy_price
        
        if d1_ret < -0.05:
             trade['状态'] = 'D+1止损'
             trade['D+1'] = round(d1_ret * 100, 2)
             for n in range(1, 10): trade[f"D+{n+1}"] = round(d1_ret * 100, 2)
        else:
            trade['D+1'] = round(d1_ret * 100, 2)
            triggered = False
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
    st.header("⚙️ V29 时光倒流")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 150, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.info("🕒 抢跑参数 (VS 替补策略)")
    st.markdown("""
    * **RSI**: 50-80 (原版 >80)
    * **获利盘**: >50% (原版 >90%)
    * **均线**: 站上 MA20 (原版 MA60)
    """)
    top_n = st.number_input("每日优选 (Top N)", 1, 10, 5)
    
    run_btn = st.button("🚀 启动时光机")

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
            
            with st.spinner("穿越回鱼头时刻..."):
                df_calc = calculate_strategy(df_all, df_info)
                
            cal_dates = sorted(df_calc['trade_date'].unique())
            valid_dates = cal_dates[-(days_back):]
            
            df_signals = df_calc[(df_calc['trade_date'].isin(valid_dates)) & (df_calc['is_signal'])].copy()
            
            # 排序: 分数高的排前面 (低位强攻)
            df_signals = df_signals.sort_values(['trade_date', 'score'], ascending=[True, False])
            
            df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
            df_top = df_signals[df_signals['排名'] <= top_n].copy()
            
            st.write(f"⚪ 抢跑信号: **{len(df_top)}** 个")
            
            if not df_top.empty:
                df_res = run_backtest_tm(df_top, df_calc, cal_dates)
                
                if not df_res.empty:
                    st.success(f"🎯 成交单数: **{len(df_res)}**")
                    
                    st.markdown(f"### 📊 V29 回测 (抢跑版)")
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
