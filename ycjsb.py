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
st.set_page_config(page_title="潜龙 V19·铁门槛", layout="wide")
st.title("🐉 潜龙 V19·铁门槛 (竞价过滤+去弱留强)")
st.markdown("""
**策略核心：只在"对的开盘"买入**
1.  **保留 V18 双龙核心**：左侧潜伏 + 右侧追击 (包含北交所大肉)。
2.  **新增"铁门槛" (竞价过滤)**：
    * 🛑 **拒绝低开**：如果开盘价 < 昨收 (绿盘)，说明承接太弱，**放弃买入**。
    * 🛑 **拒绝过热**：如果开盘价 > 昨收 * 1.07 (高开>7%)，盈亏比差，**放弃买入**。
3.  **止损铁律**：买入当天(D+1)亏损，次日坚决走人。
""")

# ==========================================
# 2. 核心数据引擎 (复用 V18 缓存)
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

@st.cache_data(persist="disk", show_spinner=True)
def fetch_and_cache_data(token, start_date, end_date):
    ts.set_token(token)
    pro = ts.pro_api()
    cal_dates = get_trade_cal(token, start_date, end_date)
    if not cal_dates: return pd.DataFrame(), pd.DataFrame(), []
    
    data_list = []
    total = len(cal_dates)
    bar = st.progress(0, text="正在同步全市场数据...")
    
    for i, date in enumerate(cal_dates):
        try:
            time.sleep(0.02)
            df = pro.daily(trade_date=date)
            # 必须包含 pre_close 用于计算开盘涨幅
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
    
    df_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,industry')
    df_info = df_info[~df_info['name'].str.contains('ST')]
    
    return full_df, df_info, cal_dates

# ==========================================
# 3. 策略逻辑 (双龙系统)
# ==========================================
def calculate_strategy_dual(df_all, df_info, strategy_params):
    if 'industry' not in df_all.columns:
        df = pd.merge(df_all, df_info[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df = df_all.copy()
        
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma30'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(30).mean())
    
    # 策略 A: 上帝指纹
    df['gap1'] = df['ma5'] - df['ma10']
    df['gap2'] = df['ma10'] - df['ma20']
    df['gap3'] = df['ma20'] - df['ma30']
    df['max_gap'] = df[['gap1', 'gap2', 'gap3']].max(axis=1)
    df['min_gap'] = df[['gap1', 'gap2', 'gap3']].min(axis=1)
    
    cond_order = (df['close'] > df['ma5']) & (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20'])
    cond_spacing = (df['max_gap'] / (df['min_gap'] + 0.0001)) < strategy_params['spacing_threshold']
    cond_active = df['pct_chg'] > 2.0
    cond_basic = df['turnover_rate'] > 1.0
    df['signal_A'] = cond_order & cond_spacing & cond_active & cond_basic
    
    # 策略 B: 追击
    cond_limit = df['pct_chg'] > 9.5
    cond_vol = df['volume_ratio'] > strategy_params['vol_threshold']
    cond_trend = df['close'] > df['ma5']
    df['signal_B'] = cond_limit & cond_vol & cond_trend
    
    df['is_signal'] = df['signal_A'] | df['signal_B']
    df['strategy_type'] = np.where(df['signal_B'], 'B:追击', np.where(df['signal_A'], 'A:潜伏', ''))
    
    return df

# ==========================================
# 4. 回测逻辑 (新增铁门槛过滤)
# ==========================================
def run_backtest_iron(df_signals, df_all, cal_dates):
    # 构建价格查询表 (包含 pre_close 用于计算开盘涨幅)
    df_lookup = df_all.copy()
    if 'ma10' not in df_lookup.columns:
        df_lookup['ma10'] = df_lookup.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
        
    price_lookup = df_lookup[['ts_code', 'trade_date', 'open', 'close', 'low', 'ma10', 'pre_close']].set_index(['ts_code', 'trade_date'])
    
    trades = []
    
    for i, row in enumerate(df_signals.itertuples()):
        signal_date = row.trade_date
        code = row.ts_code
        
        try:
            curr_idx = cal_dates.index(signal_date)
            future_dates = cal_dates[curr_idx+1 : curr_idx+11]
        except: continue
            
        if not future_dates: continue
        
        # --- 铁门槛判定 (Day 1 开盘) ---
        d1_date = future_dates[0]
        if (code, d1_date) not in price_lookup.index: continue
        d1_data = price_lookup.loc[(code, d1_date)]
        
        d1_open = d1_data['open']
        d1_pre = d1_data['pre_close']
        
        # 计算开盘涨幅
        open_pct = (d1_open - d1_pre) / d1_pre * 100
        
        # 1. 拒绝低开 (弱势)
        if open_pct < 0:
            continue # 直接跳过，不开仓
            
        # 2. 拒绝高开 > 7% (博傻)
        if open_pct > 7.0:
            continue # 直接跳过，不开仓
            
        # 3. 满足条件，开仓
        buy_price = d1_open
        
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, '策略': row.strategy_type,
            '行业': row.industry, '买入价': buy_price, '开盘涨幅': f"{open_pct:.2f}%", '状态': '持有'
        }
        
        # D+1 止损逻辑
        d1_close = d1_data['close']
        d1_ret = (d1_close - buy_price) / buy_price
        
        if d1_ret < 0:
            trade['状态'] = 'D+1止损'
            trade['D+1'] = round(d1_ret * 100, 2)
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
# 5. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V19 铁门槛配置")
    user_token = st.text_input("Tushare Token:", type="password")
    
    st.info("👇 修改日期会触发数据下载")
    days_back = st.slider("回测范围 (天)", 30, 120, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.info("👇 修改以下参数，秒级出结果")
    spacing = st.number_input("策略A: 均线均匀度 <", 1.0, 3.0, 1.5)
    vol_ratio = st.number_input("策略B: 追击量比 >", 1.0, 5.0, 2.0)
    top_n = st.number_input("每日Top N", 1, 10, 3)
    
    run_btn = st.button("🚀 启动V19")

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
            st.success(f"✅ 数据就绪: {len(df_all):,} 行")
            
            # 2. 计算策略
            with st.spinner("双龙策略计算中..."):
                params = {'spacing_threshold': spacing, 'vol_threshold': vol_ratio}
                df_calc = calculate_strategy_dual(df_all, df_info, params)
                
            # 3. 筛选信号
            valid_dates = cal_dates[-(days_back):]
            df_signals = df_calc[(df_calc['trade_date'].isin(valid_dates)) & (df_calc['is_signal'])].copy()
            df_signals = df_signals.sort_values(['trade_date', 'volume_ratio'], ascending=[True, False])
            
            df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
            df_top = df_signals[df_signals['排名'] <= top_n].copy()
            
            st.write(f"⚪ 原始信号: **{len(df_top)}** 个")
            
            # 4. 回测 (带铁门槛)
            if not df_top.empty:
                df_res = run_backtest_iron(df_top, df_calc, cal_dates)
                
                if not df_res.empty:
                    st.success(f"🎯 铁门槛过滤后成交: **{len(df_res)}** 个 (剔除了 {len(df_top)-len(df_res)} 个坏开盘)")
                    
                    st.markdown(f"### 📊 V19 回测结果 (铁门槛+D1止损)")
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
                    st.warning("所有信号均被铁门槛拦截（低开或高开过多）。")
            else:
                st.warning("无满足条件的信号。")
