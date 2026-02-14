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
# 1. 页面配置 & 常量
# ==========================================
st.set_page_config(page_title="潜龙 V21·快刀手", layout="wide")
st.title("🐉 潜龙 V21·快刀手 (放开高开+MA5极速止盈)")
st.markdown("""
**策略升级：适应"电风扇"行情的快进快出**
1.  **拆除天花板**：**取消"高开>7%"的限制**，放行北交所和连板妖股。
2.  **保留底线**：**拒绝大幅低开 (Open < -2%)**，回避弱势股。
3.  **极速止盈**：
    * 🛑 **破线止盈**：收盘跌破 **MA5** 即卖出 (原MA10太慢)。
    * ⏱️ **时间止盈**：**D+3** 收盘强制清仓 (3天不涨就走)。
4.  **止损铁律**：**D+1 亏损次日即走** (保持不变)。
""")

DATA_FILE = "market_data_store.csv"

# ==========================================
# 2. 核心数据引擎 (保持 V20 增量更新)
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
# 3. 策略逻辑 (双龙)
# ==========================================
def calculate_strategy(df_all, df_info, params):
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
    cond_spacing = (df['max_gap'] / (df['min_gap'] + 0.0001)) < params['spacing']
    cond_order = (df['close'] > df['ma5']) & (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20'])
    cond_active = df['pct_chg'] > 2.0
    df['signal_A'] = cond_order & cond_spacing & cond_active
    
    # 策略 B: 追击
    cond_limit = df['pct_chg'] > 9.5
    cond_vol = df['volume_ratio'] > params['vol_ratio']
    cond_trend = df['close'] > df['ma5']
    df['signal_B'] = cond_limit & cond_vol & cond_trend
    
    df['is_signal'] = df['signal_A'] | df['signal_B']
    df['strategy_type'] = np.where(df['signal_B'], 'B:追击', np.where(df['signal_A'], 'A:潜伏', ''))
    
    return df

# ==========================================
# 4. 回测逻辑 (MA5止盈 + 3天强制离场)
# ==========================================
def run_backtest_fast(df_signals, df_all, cal_dates):
    df_lookup = df_all.copy()
    if 'ma5' not in df_lookup.columns:
         df_lookup['ma5'] = df_lookup.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    
    price_lookup = df_lookup[['ts_code', 'trade_date', 'open', 'close', 'low', 'ma5', 'pre_close']].set_index(['ts_code', 'trade_date'])
    
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
        
        # === 优化的铁门槛 ===
        open_pct = (d1_data['open'] - d1_data['pre_close']) / d1_data['pre_close'] * 100
        
        # 1. 拒绝大幅低开 (<-2%) - 适度放宽
        if open_pct < -2.0: continue
        
        # 2. 放开高开限制 (为了抓妖股)
        # if open_pct > 7.0: continue  <-- 删掉这行
        
        buy_price = d1_data['open']
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, '策略': row.strategy_type,
            '行业': row.industry, '买入价': buy_price, '开盘涨幅': f"{open_pct:.2f}%", '状态': '持有'
        }
        
        # D+1 止损判定
        d1_ret = (d1_data['close'] - buy_price) / buy_price
        
        if d1_ret < 0:
            trade['状态'] = 'D+1止损'
            trade['D+1'] = round(d1_ret * 100, 2)
            # 止损后，后面全是这个收益
            for n in range(1, 10):
                 trade[f"D+{n+1}"] = round(d1_ret * 100, 2)
        else:
            trade['D+1'] = round(d1_ret * 100, 2)
            triggered = False
            
            # 从 D+2 开始
            for n in range(1, 10):
                if n >= len(future_dates): break
                f_date = future_dates[n]
                if (code, f_date) not in price_lookup.index: break
                f_data = price_lookup.loc[(code, f_date)]
                day_key = f"D+{n+1}"
                
                if not triggered:
                    # 1. 极速止盈: 收盘 < MA5
                    if f_data['close'] < f_data['ma5']:
                        triggered = True
                        trade['状态'] = '破MA5止盈'
                        curr_ret = (f_data['close'] - buy_price) / buy_price * 100
                        trade[day_key] = round(curr_ret, 2)
                    # 2. 时间止盈: D+3 强制走人 (只看前3天)
                    elif n >= 2: # 索引2对应 D+3
                        triggered = True
                        trade['状态'] = 'D+3限时卖出'
                        curr_ret = (f_data['close'] - buy_price) / buy_price * 100
                        trade[day_key] = round(curr_ret, 2)
                    else:
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
    st.header("⚙️ V21 快刀手")
    user_token = st.text_input("Tushare Token:", type="password")
    
    days_back = st.slider("回测天数", 30, 150, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.info("快进快出模式")
    spacing = st.number_input("策略A: 均匀度 <", 1.0, 3.0, 1.5)
    vol_ratio = st.number_input("策略B: 量比 >", 1.0, 5.0, 2.0)
    top_n = st.number_input("Top N", 1, 10, 3)
    
    if st.button("🗑️ 清除缓存"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
            st.success("缓存已清除")
            
    run_btn = st.button("🚀 启动快刀")

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
            
            with st.spinner("策略运算..."):
                params = {'spacing': spacing, 'vol_ratio': vol_ratio}
                df_calc = calculate_strategy(df_all, df_info, params)
                
            cal_dates = sorted(df_calc['trade_date'].unique())
            valid_dates = cal_dates[-(days_back):]
            
            df_signals = df_calc[(df_calc['trade_date'].isin(valid_dates)) & (df_calc['is_signal'])].copy()
            df_signals = df_signals.sort_values(['trade_date', 'volume_ratio'], ascending=[True, False])
            
            df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
            df_top = df_signals[df_signals['排名'] <= top_n].copy()
            
            st.write(f"⚪ 原始信号: **{len(df_top)}** 个")
            
            if not df_top.empty:
                df_res = run_backtest_fast(df_top, df_calc, cal_dates)
                
                if not df_res.empty:
                    st.success(f"🎯 成交单数: **{len(df_res)}**")
                    
                    st.markdown(f"### 📊 V21 回测 (MA5止盈+D3限时)")
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
