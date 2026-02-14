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
st.set_page_config(page_title="潜龙 V18·双龙戏珠", layout="wide")
st.title("🐉 潜龙 V18·双龙戏珠 (永久缓存版)")
st.markdown("""
**架构升级：数据下载与策略计算解耦。**
1.  **数据层**：首次运行会下载数据并缓存，后续调整策略**无需重新下载**。
2.  **双策略并行**：
    * **策略 A (潜伏)**：上帝指纹 (均线完美) + 温和放量。
    * **策略 B (追击)**：暴力涨停 + 巨量突破。
""")

# ==========================================
# 2. 核心数据引擎 (带持久化缓存)
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

@st.cache_data(persist="disk", show_spinner=True) # 开启spinner，让用户知道在干嘛
def fetch_and_cache_data(token, start_date, end_date):
    """
    只负责下载数据，不负责计算。
    只要日期范围不变，这个函数永远只跑一次。
    """
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 获取交易日历
    cal_dates = get_trade_cal(token, start_date, end_date)
    if not cal_dates: return pd.DataFrame(), pd.DataFrame(), []
    
    data_list = []
    total = len(cal_dates)
    bar = st.progress(0, text="正在同步全市场数据 (首次运行较慢，请耐心)...")
    
    for i, date in enumerate(cal_dates):
        try:
            time.sleep(0.02) # 稍微加速
            # 1. 行情
            df = pro.daily(trade_date=date)
            # 2. 指标
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,volume_ratio,circ_mv')
            
            if not df.empty and not df_basic.empty:
                df = pd.merge(df, df_basic, on='ts_code', how='left')
                data_list.append(df)
        except:
            time.sleep(0.5)
            
        if (i+1) % 5 == 0:
            bar.progress((i+1)/total, text=f"下载进度: {i+1}/{total}")
            
    bar.empty()
    
    if not data_list: return pd.DataFrame(), pd.DataFrame(), []
    
    full_df = pd.concat(data_list)
    full_df = full_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    # 获取基础信息 (名称行业)
    df_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,industry')
    df_info = df_info[~df_info['name'].str.contains('ST')]
    
    return full_df, df_info, cal_dates

# ==========================================
# 3. 策略逻辑 (本地计算，秒级响应)
# ==========================================
def calculate_strategy_dual(df_all, df_info, strategy_params):
    """
    双策略并行计算
    """
    # 1. 合并基础信息
    if 'industry' not in df_all.columns:
        df = pd.merge(df_all, df_info[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df = df_all.copy()
        
    # 2. 计算通用指标 (均线)
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma30'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(30).mean())
    
    # 3. 策略 A: 上帝指纹 (潜伏)
    # 逻辑: 均线等距 (Ratio < 1.5) + 多头排列 + 涨幅 > 2% (不要求涨停)
    
    # 均线间距
    df['gap1'] = df['ma5'] - df['ma10']
    df['gap2'] = df['ma10'] - df['ma20']
    df['gap3'] = df['ma20'] - df['ma30']
    df['max_gap'] = df[['gap1', 'gap2', 'gap3']].max(axis=1)
    df['min_gap'] = df[['gap1', 'gap2', 'gap3']].min(axis=1)
    
    cond_order = (df['close'] > df['ma5']) & (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20'])
    cond_spacing = (df['max_gap'] / (df['min_gap'] + 0.0001)) < strategy_params['spacing_threshold'] # 1.5
    cond_active = df['pct_chg'] > 2.0 # 只要启动就行
    cond_basic = df['turnover_rate'] > 1.0
    
    df['signal_A'] = cond_order & cond_spacing & cond_active & cond_basic
    
    # 4. 策略 B: 暴力追击 (情绪)
    # 逻辑: 涨停 (>9.5%) + 量比 > 2.0 + 站上所有均线
    
    cond_limit = df['pct_chg'] > 9.5
    cond_vol = df['volume_ratio'] > strategy_params['vol_threshold'] # 2.0
    cond_trend = df['close'] > df['ma5'] # 简单趋势
    
    df['signal_B'] = cond_limit & cond_vol & cond_trend
    
    # 5. 合并信号
    df['is_signal'] = df['signal_A'] | df['signal_B']
    df['strategy_type'] = np.where(df['signal_B'], 'B:追击', np.where(df['signal_A'], 'A:潜伏', ''))
    
    return df

def run_backtest(df_signals, df_all, cal_dates):
    # 构建价格查询表 (Open, Close, MA10)
    # MA10 需要从 df_all 里取，如果 df_all 里没算，就在这里算一下或者复用上面的 df
    # 为了简单，我们假设 df_signals 已经包含了 MA10 (因为它是 df_all 计算来的)
    # 但我们需要未来的数据，所以还是需要 lookup
    
    # 这里有点 trick: df_signals 是过去的数据，回测要看未来的数据
    # 我们需要一个全量的 lookup table
    # 重新计算一次全量 MA10 放入 lookup
    
    df_lookup = df_all.copy()
    if 'ma10' not in df_lookup.columns:
        df_lookup['ma10'] = df_lookup.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
        
    price_lookup = df_lookup[['ts_code', 'trade_date', 'open', 'close', 'low', 'ma10']].set_index(['ts_code', 'trade_date'])
    
    trades = []
    
    for i, row in enumerate(df_signals.itertuples()):
        signal_date = row.trade_date
        code = row.ts_code
        
        try:
            curr_idx = cal_dates.index(signal_date)
            future_dates = cal_dates[curr_idx+1 : curr_idx+11]
        except: continue
            
        if not future_dates: continue
        
        # D+1 开盘买入
        d1_date = future_dates[0]
        if (code, d1_date) not in price_lookup.index: continue
        d1_data = price_lookup.loc[(code, d1_date)]
        
        buy_price = d1_data['open']
        
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, '策略': row.strategy_type,
            '行业': row.industry, '买入价': buy_price, '状态': '持有'
        }
        
        # 止损逻辑: D+1 亏损即走
        d1_close = d1_data['close']
        d1_ret = (d1_close - buy_price) / buy_price
        
        if d1_ret < 0:
            trade['状态'] = 'D+1止损'
            trade['D+1'] = round(d1_ret * 100, 2)
            # 后续天数收益锁定
            for n in range(1, 10):
                trade[f"D+{n+1}"] = round(d1_ret * 100, 2)
        else:
            trade['D+1'] = round(d1_ret * 100, 2)
            triggered = False
            for n in range(1, 10):
                if n >= len(future_dates): break
                f_date = future_dates[n]
                if (code, f_date) not in price_lookup.index: break
                f_data = price_lookup.loc[(code, f_date)]
                day_label = f"D+{n+1}"
                
                if not triggered:
                    # MA10 止盈
                    if f_data['close'] < f_data['ma10']:
                        triggered = True
                        trade['状态'] = '止盈'
                        ret = (f_data['close'] - buy_price) / buy_price * 100
                        trade[day_label] = round(ret, 2)
                    else:
                        ret = (f_data['close'] - buy_price) / buy_price * 100
                        trade[day_label] = round(ret, 2)
                else:
                    trade[day_label] = trade.get(f"D+{n}", 0)
                    
        trades.append(trade)
        
    return pd.DataFrame(trades)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V18 参数配置")
    user_token = st.text_input("Tushare Token:", type="password")
    
    # 只有这里的日期改变，才会触发重新下载
    st.info("👇 修改日期会触发数据下载")
    days_back = st.slider("回测范围 (天)", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.info("👇 修改以下参数，秒级出结果")
    spacing = st.number_input("策略A: 均线均匀度 <", 1.0, 3.0, 1.5)
    vol_ratio = st.number_input("策略B: 追击量比 >", 1.0, 5.0, 2.0)
    top_n = st.number_input("每日Top N", 1, 10, 2)
    
    run_btn = st.button("🚀 启动双龙系统")

if run_btn:
    if not user_token:
        st.error("请先输入 Token")
    else:
        # 1. 下载或读取缓存数据
        end_str = end_date_input.strftime('%Y%m%d')
        start_dt = end_date_input - timedelta(days=days_back * 1.5 + 80)
        start_str = start_dt.strftime('%Y%m%d')
        
        df_all, df_info, cal_dates = fetch_and_cache_data(user_token, start_str, end_str)
        
        if not df_all.empty:
            st.success(f"✅ 数据就绪: {len(df_all):,} 行 (无需重复下载)")
            
            # 2. 计算策略
            with st.spinner("策略计算中..."):
                params = {'spacing_threshold': spacing, 'vol_threshold': vol_ratio}
                df_calc = calculate_strategy_dual(df_all, df_info, params)
                
            # 3. 筛选信号
            valid_dates = cal_dates[-(days_back):]
            df_signals = df_calc[(df_calc['trade_date'].isin(valid_dates)) & (df_calc['is_signal'])].copy()
            
            # 评分排序
            # 优先看策略B(追击)，其次策略A(潜伏)的均匀度
            # 这里简单混合排序：量比高的排前面
            df_signals = df_signals.sort_values(['trade_date', 'volume_ratio'], ascending=[True, False])
            
            # Top N
            df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
            df_top = df_signals[df_signals['排名'] <= top_n].copy()
            
            st.write(f"⚪ 捕获双龙信号: **{len(df_top)}** 个")
            
            # 4. 回测
            if not df_top.empty:
                df_res = run_backtest(df_top, df_calc, cal_dates)
                
                st.markdown(f"### 📊 V18 回测结果 (D+1止损版)")
                cols = st.columns(5)
                days = ['D+1', 'D+3', 'D+5', 'D+7', 'D+10']
                
                for idx, d in enumerate(days):
                    if d in df_res.columns:
                        avg_ret = df_res[d].mean()
                        if d == 'D+1':
                            win_rate = (df_res[d] > 0).mean() * 100
                            cols[idx].metric(f"{d} 胜率", f"{win_rate:.1f}%")
                        cols[idx].metric(f"{d} 均收", f"{avg_ret:.2f}%")
                
                st.dataframe(df_res.sort_values(['信号日'], ascending=False), use_container_width=True)
            else:
                st.warning("无满足条件的信号。")
