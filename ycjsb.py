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
st.set_page_config(page_title="潜龙 V20·永不磨灭", layout="wide")
st.title("🐉 潜龙 V20·永不磨灭 (物理缓存+增量更新)")
st.markdown("""
**架构重构：彻底解决重复下载问题**
1.  **硬盘级缓存**：数据永久保存在 `market_data_store.csv`，不怕代码覆盖。
2.  **智能增量**：自动识别缺失日期，**只下载** 没下的部分 (真正的断点续传)。
3.  **核心策略**：V19 铁门槛 (竞价过滤) + V18 双龙 (潜伏/追击)。
""")

DATA_FILE = "market_data_store.csv"

# ==========================================
# 2. 核心数据引擎 (增量更新版)
# ==========================================
def get_trade_cal(pro, start_date, end_date):
    try:
        df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
        return sorted(df['cal_date'].tolist())
    except:
        return []

def sync_market_data(token, start_date, end_date):
    """
    增量同步逻辑：
    1. 读取本地现有数据，获取已有的日期集合。
    2. 对比目标日期范围，找出缺少的日期。
    3. 只下载缺少的日期，追加写入文件。
    """
    if not token:
        return pd.DataFrame(), "请先输入Token"
        
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 1. 获取目标交易日历
    target_dates = get_trade_cal(pro, start_date, end_date)
    if not target_dates:
        return pd.DataFrame(), "无法获取交易日历"
        
    # 2. 检查本地数据
    existing_dates = set()
    if os.path.exists(DATA_FILE):
        try:
            # 只读日期列，加快速度
            df_dates = pd.read_csv(DATA_FILE, usecols=['trade_date'], dtype={'trade_date': str})
            existing_dates = set(df_dates['trade_date'].unique())
        except:
            pass # 文件可能损坏或为空
            
    # 3. 计算缺失日期
    missing_dates = sorted(list(set(target_dates) - existing_dates))
    
    # 4. 如果有缺失，进行增量下载
    if missing_dates:
        st.info(f"发现 {len(missing_dates)} 个新交易日，开始增量更新...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        new_data = []
        
        # 批量下载缓冲 (每10天存一次盘，防止内存爆)
        batch_size = 5
        
        for i, date in enumerate(missing_dates):
            try:
                status_text.text(f"正在下载: {date} ...")
                
                # 下载行情
                df_daily = pro.daily(trade_date=date)
                # 下载指标
                df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,volume_ratio,circ_mv')
                
                if not df_daily.empty and not df_basic.empty:
                    # 合并
                    df_merged = pd.merge(df_daily, df_basic, on='ts_code', how='left')
                    # 确保日期格式统一
                    df_merged['trade_date'] = str(date)
                    new_data.append(df_merged)
                
            except Exception as e:
                st.warning(f"{date} 下载失败: {e}")
                time.sleep(1)
            
            # 进度条
            progress_bar.progress((i + 1) / len(missing_dates))
            
            # 批次写入 (断点续传的关键：下一点存一点)
            if len(new_data) >= batch_size or (i == len(missing_dates) - 1):
                if new_data:
                    df_batch = pd.concat(new_data)
                    # 追加模式写入，如果文件不存在则包含header
                    mode = 'a' if os.path.exists(DATA_FILE) else 'w'
                    header = not os.path.exists(DATA_FILE)
                    df_batch.to_csv(DATA_FILE, mode=mode, header=header, index=False)
                    new_data = [] # 清空缓冲
                    
        status_text.text("增量更新完成！")
        progress_bar.empty()
        
    # 5. 读取全量数据 (为了策略计算)
    # 既然是回测，我们需要一段连续的数据
    # 这里读取文件，并根据日期过滤
    if os.path.exists(DATA_FILE):
        # 显式指定类型，防止 pandas 猜错
        dtype_dict = {'ts_code': str, 'trade_date': str}
        df_all = pd.read_csv(DATA_FILE, dtype=dtype_dict)
        
        # 过滤出需要的日期范围
        df_all = df_all[(df_all['trade_date'] >= start_date) & (df_all['trade_date'] <= end_date)]
        
        # 去重 (防止重复写入)
        df_all = df_all.drop_duplicates(subset=['ts_code', 'trade_date'])
        
        # 获取基础信息 (行业名称)
        # 这个不经常变，可以用 st.cache 缓存一下 API 调用
        @st.cache_data
        def get_stock_info():
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,industry')
            return df[~df['name'].str.contains('ST')]
            
        df_info = get_stock_info()
        
        return df_all, df_info
    else:
        return pd.DataFrame(), "无数据"

# ==========================================
# 3. 策略逻辑 (V18 双龙 + V19 铁门槛)
# ==========================================
def calculate_strategy(df_all, df_info, params):
    # 1. 关联行业
    if 'industry' not in df_all.columns:
        df = pd.merge(df_all, df_info[['ts_code', 'industry', 'name']], on='ts_code', how='left')
    else:
        df = df_all.copy()
        
    # 2. 计算均线
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma30'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(30).mean())
    
    # 3. 策略 A: 上帝指纹 (潜伏)
    df['gap1'] = df['ma5'] - df['ma10']
    df['gap2'] = df['ma10'] - df['ma20']
    df['gap3'] = df['ma20'] - df['ma30']
    df['max_gap'] = df[['gap1', 'gap2', 'gap3']].max(axis=1)
    df['min_gap'] = df[['gap1', 'gap2', 'gap3']].min(axis=1)
    
    cond_order = (df['close'] > df['ma5']) & (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20'])
    cond_spacing = (df['max_gap'] / (df['min_gap'] + 0.0001)) < params['spacing']
    cond_active = df['pct_chg'] > 2.0
    
    df['signal_A'] = cond_order & cond_spacing & cond_active
    
    # 4. 策略 B: 追击 (暴力)
    cond_limit = df['pct_chg'] > 9.5
    cond_vol = df['volume_ratio'] > params['vol_ratio']
    cond_trend = df['close'] > df['ma5']
    
    df['signal_B'] = cond_limit & cond_vol & cond_trend
    
    df['is_signal'] = df['signal_A'] | df['signal_B']
    df['strategy_type'] = np.where(df['signal_B'], 'B:追击', np.where(df['signal_A'], 'A:潜伏', ''))
    
    return df

def run_backtest_iron(df_signals, df_all, cal_dates):
    # 构建 Lookup
    df_lookup = df_all.copy()
    # 确保有 MA10 和 Pre_Close
    if 'ma10' not in df_lookup.columns:
         df_lookup['ma10'] = df_lookup.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    
    price_lookup = df_lookup[['ts_code', 'trade_date', 'open', 'close', 'low', 'ma10', 'pre_close']].set_index(['ts_code', 'trade_date'])
    
    trades = []
    
    for row in df_signals.itertuples():
        signal_date = row.trade_date
        code = row.ts_code
        
        # 找未来日期
        try:
            curr_idx = cal_dates.index(signal_date)
            future_dates = cal_dates[curr_idx+1 : curr_idx+11]
        except: continue
        
        if not future_dates: continue
        d1_date = future_dates[0]
        
        if (code, d1_date) not in price_lookup.index: continue
        d1_data = price_lookup.loc[(code, d1_date)]
        
        # === 铁门槛过滤 ===
        # 1. 计算开盘涨幅
        open_pct = (d1_data['open'] - d1_data['pre_close']) / d1_data['pre_close'] * 100
        
        # 拒绝低开 (绿盘)
        if open_pct < 0: continue
        # 拒绝高开 > 7%
        if open_pct > 7.0: continue
        
        # === 开仓 ===
        buy_price = d1_data['open']
        trade = {
            '信号日': signal_date, '代码': code, '名称': row.name, '策略': row.strategy_type,
            '行业': row.industry, '买入价': buy_price, '开盘涨幅': f"{open_pct:.2f}%", '状态': '持有'
        }
        
        # === D+1 止损判定 ===
        d1_ret = (d1_data['close'] - buy_price) / buy_price
        
        if d1_ret < 0:
            # 亏损：第二天跑路 (收益锁定为 D+1)
            trade['状态'] = 'D+1止损'
            trade['D+1'] = round(d1_ret * 100, 2)
            for n in range(1, 10):
                 trade[f"D+{n+1}"] = round(d1_ret * 100, 2)
        else:
            # 盈利：MA10 跟踪止盈
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
                        trade['状态'] = '止盈'
                    curr_ret = (f_data['close'] - buy_price) / buy_price * 100
                    trade[day_key] = round(curr_ret, 2)
                else:
                    trade[day_key] = trade.get(f"D+{n}", 0)
        
        trades.append(trade)
        
    return pd.DataFrame(trades)

# ==========================================
# 4. 主程序
# ==========================================
with st.sidebar:
    st.header("⚙️ V20 永不磨灭版")
    user_token = st.text_input("Tushare Token:", type="password")
    
    st.info("📅 数据管理")
    days_back = st.slider("回测天数", 30, 150, 60)
    end_date_input = st.date_input("截止日期", datetime.now().date())
    
    st.markdown("---")
    st.info("🎛 策略参数 (修改不触发下载)")
    spacing = st.number_input("策略A: 均匀度 <", 1.0, 3.0, 1.5)
    vol_ratio = st.number_input("策略B: 量比 >", 1.0, 5.0, 2.0)
    top_n = st.number_input("Top N", 1, 10, 3)
    
    if st.button("🗑️ 清除缓存数据 (慎点)"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
            st.success("缓存已清除，下次运行将全量下载。")
            
    run_btn = st.button("🚀 启动系统")

if run_btn:
    if not user_token:
        st.error("请先输入 Token")
    else:
        # 1. 准备日期范围
        end_str = end_date_input.strftime('%Y%m%d')
        start_dt = end_date_input - timedelta(days=days_back * 1.5 + 80) # 多下点用于算均线
        start_str = start_dt.strftime('%Y%m%d')
        
        # 2. 同步数据 (增量)
        res, info = sync_market_data(user_token, start_str, end_str)
        
        if isinstance(info, pd.DataFrame):
            df_info = info
            df_all = res
            
            st.success(f"✅ 数据加载完毕: {len(df_all):,} 行 (来自 {DATA_FILE})")
            
            # 3. 计算策略
            with st.spinner("策略引擎运行中..."):
                params = {'spacing': spacing, 'vol_ratio': vol_ratio}
                df_calc = calculate_strategy(df_all, df_info, params)
                
            # 4. 提取信号
            cal_dates = sorted(df_calc['trade_date'].unique())
            valid_dates = cal_dates[-(days_back):]
            
            df_signals = df_calc[(df_calc['trade_date'].isin(valid_dates)) & (df_calc['is_signal'])].copy()
            df_signals = df_signals.sort_values(['trade_date', 'volume_ratio'], ascending=[True, False])
            
            # Top N
            df_signals['排名'] = df_signals.groupby('trade_date').cumcount() + 1
            df_top = df_signals[df_signals['排名'] <= top_n].copy()
            
            st.write(f"⚪ 原始信号: **{len(df_top)}** 个")
            
            # 5. 回测 (铁门槛)
            if not df_top.empty:
                df_res = run_backtest_iron(df_top, df_calc, cal_dates)
                
                if not df_res.empty:
                    st.success(f"🎯 铁门槛成交: **{len(df_res)}** 单")
                    
                    st.markdown(f"### 📊 V20 最终回测")
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
                    st.warning("所有信号均被铁门槛拦截。")
            else:
                st.warning("无信号。")
        else:
            st.error(info)
