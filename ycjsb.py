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
st.set_page_config(page_title="潜龙 V24·万法归一", layout="wide")
st.title("🐉 潜龙 V24·万法归一 (全基因融合版)")
st.markdown("""
**策略核心：打破偏见，包容 12 真龙的所有形态**
1.  **市值无界**：**30亿 - 800亿** (覆盖小盘妖股与中盘悍将罗博特科)。
2.  **形态宽容**：只要 **股价 > MA20** (趋势向上)，允许均线暂时纠缠 (包容"震荡启动")。
3.  **双轨启动**：
    * **A模式 (换手)**：量比 > 1.2 + 换手 > 1.5% (常规启动)。
    * **B模式 (一字)**：涨幅 > 9.5% (缩量涨停也算，包容嘉美包装)。
4.  **趋势止盈**：坚定执行 **MA10 止盈**。
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
# 3. 策略逻辑 (V24 终极版)
# ==========================================
def calculate_strategy(df_all, df_info):
    if 'industry' not in df_all.columns:
        df = pd.merge(df_all, df_info[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df = df_all.copy()
        
    # 计算均线
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma30'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(30).mean())
    
    # === 1. 市值无界 (30-800亿) ===
    # 覆盖从田中精机到罗博特科的所有区间
    mv_min = 30 * 10000 
    mv_max = 800 * 10000
    cond_mv = (df['circ_mv'] >= mv_min) & (df['circ_mv'] <= mv_max)
    
    # === 2. 趋势底线 (包容震荡) ===
    # 只要站上 MA20 生命线，就视为趋势良好，不强求多头排列
    cond_trend = df['close'] > df['ma20']
    
    # === 3. 双轨启动 (包容一字板) ===
    # A模式: 正常换手启动
    cond_mode_a = (df['volume_ratio'] > 1.2) & (df['turnover_rate'] > 1.5) & (df['pct_chg'] > 3.0)
    
    # B模式: 缩量/一字涨停 (嘉美包装模式)
    cond_mode_b = (df['pct_chg'] > 9.5)
    
    cond_start = cond_mode_a | cond_mode_b
    
    # === 综合信号 ===
    df['is_signal'] = cond_mv & cond_trend & cond_start
    
    # 标记模式
    df['pattern'] = np.where(cond_mode_b, 'B:强力/一字', 'A:换手启动')
    
    return df

# ==========================================
# 4. 回测逻辑 (MA10 趋势止盈)
# ==========================================
def run_backtest_final(df_signals, df_all, cal_dates):
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
        
        # 铁门槛: 拒绝恶意低开
        open_pct = (d1_data['open'] - d1_data['pre_close']) / d1_data['pre_close'] * 100
        if open_pct < -2.0: continue
        
        # 注意: 如果是一字涨停(B模式)，D+1大概率买不进，或者买进就是高位
        # 但为了回测完整性，我们假设以开盘价买入 (如果D+1继续一字，open=close，还是能买/排板)
        # 这里不做过度复杂的排板模拟，统一按开盘价
        
        buy_price = d1_data['open']
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, 
            '行业': row.industry, '买入价': buy_price, '模式': row.pattern, 
            '开盘涨幅': f"{open_pct:.2f}%", '状态': '持有'
        }
        
        # D+1 止损判定
        d1_ret = (d1_data['close'] - buy_price) / buy_price
        
        # 如果买入当天大跌 > 5%，止损
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
    st.header("⚙️ V24 万法归一")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 150, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.info("🧬 策略参数 (自适应)")
    st.markdown("""
    * **市值**: 30-800亿 (全覆盖)
    * **模式**: 换手/一字双轨并行
    * **趋势**: 只要在 MA20 上方
    """)
    top_n = st.number_input("每日优选 (Top N)", 1, 10, 3)
    
    run_btn = st.button("🚀 启动终极版")

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
            
            with st.spinner("执行终极扫描..."):
                df_calc = calculate_strategy(df_all, df_info)
                
            cal_dates = sorted(df_calc['trade_date'].unique())
            valid_dates = cal_dates[-(days_back):]
            
            df_signals = df_calc[(df_calc['trade_date'].isin(valid_dates)) & (df_calc['is_signal'])].copy()
            
            # 排序: 既然是一字板和换手板混杂，我们很难用量比排序
            # 我们用"涨幅"排序，优先看涨停板 (强者恒强)
            df_signals = df_signals.sort_values(['trade_date', 'pct_chg'], ascending=[True, False])
            
            df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
            df_top = df_signals[df_signals['排名'] <= top_n].copy()
            
            st.write(f"⚪ 终极信号: **{len(df_top)}** 个")
            
            if not df_top.empty:
                df_res = run_backtest_final(df_top, df_calc, cal_dates)
                
                if not df_res.empty:
                    st.success(f"🎯 成交单数: **{len(df_res)}**")
                    
                    st.markdown(f"### 📊 V24 回测 (全基因融合)")
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
