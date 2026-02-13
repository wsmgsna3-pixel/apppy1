import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# ==========================================
# 1. 页面与基础配置
# ==========================================
st.set_page_config(page_title="三日成妖·全周期回测", layout="wide")

st.title("🐉 三日成妖·实战全周期回测系统 (精英严选版)")
st.markdown("""
**策略核心逻辑：**
1. **筛选池 (严选)**：仅限沪深A股，**剔除北交所**，**股价 ≥ 10元**，**成交额 > 5000万**，非ST。
2. **信号源 (三连爆)**：
   - 潜伏期：过去 60 天均量。
   - 爆发期：连续 3 天成交量 > 潜伏均量 * N倍。
   - 涨幅：主板 > 12%，双创 > 20%。
   - 形态：Day1 涨幅 > 5%，且重心上移。
3. **交易规则**：
   - **买入**：D+1日 **开盘价** 买入 (若低开 < -5% 则放弃)。
   - **止损**：盘中跌破买入价 **-10%** 强制止损。
   - **持有**：最长持有 10 天，统计各节点胜率。
""")

# ==========================================
# 2. 数据获取函数 (利用 10000 积分权限)
# ==========================================
@st.cache_data(persist="disk", show_spinner=False)
def get_trade_cal(token, start_date, end_date):
    """获取交易日历"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
        return df['cal_date'].tolist()
    except:
        return []

@st.cache_data(persist="disk", show_spinner=False)
def get_daily_snapshot_filtered(token, date_str):
    """
    获取某日全市场符合【基础门槛】的股票
    利用 Tushare 批量获取能力，一次性通过
    """
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 1. 获取基础行情 (价格、成交量)
        df_daily = pro.daily(trade_date=date_str)
        
        # 2. 获取基础信息 (名称、板块、上市日期)
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,list_date')
        
        if df_daily.empty or df_basic.empty: return pd.DataFrame()
        
        # 合并
        df = pd.merge(df_daily, df_basic, on='ts_code')
        
        # === 核心筛选漏斗 ===
        
        # A. 剔除 ST
        df = df[~df['name'].str.contains('ST')]
        
        # B. 剔除 北交所 (BJ)
        df = df[~df['ts_code'].str.contains('BJ')]
        df = df[~df['market'].str.contains('北交')]
        
        # C. 剔除 次新股 (上市 < 60天, 否则没法算潜伏期)
        # 简单处理：只保留 list_date 早于当前日期 60天以上的
        limit_date = (datetime.strptime(date_str, '%Y%m%d') - timedelta(days=90)).strftime('%Y%m%d')
        df = df[df['list_date'] < limit_date]
        
        # D. 价格门槛: 收盘价 >= 10 元
        df = df[df['close'] >= 10.0]
        
        # E. 流动性门槛: 成交额 > 5000万 (amount单位是千元)
        # 5000万 = 50000 千元
        df = df[df['amount'] > 50000]
        
        return df
    except Exception as e:
        st.error(f"数据获取异常: {e}")
        return pd.DataFrame()

@st.cache_data(persist="disk", show_spinner=False)
def get_history_for_signal(token, code, end_date, lookback=70):
    """获取用于计算信号的历史数据 (潜伏+爆发)"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        start_dt = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=lookback*1.5 + 20)
        df = pro.daily(ts_code=code, start_date=start_dt.strftime('%Y%m%d'), end_date=end_date)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(persist="disk", show_spinner=False)
def get_future_performance(token, code, start_date, days=15):
    """获取未来N天的走势用于回测"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = start_dt + timedelta(days=days*2) # 多取防止停牌
        df = pro.daily(ts_code=code, start_date=start_date, end_date=end_dt.strftime('%Y%m%d'))
        return df.sort_values('trade_date').reset_index(drop=True)
    except:
        return pd.DataFrame()

# ==========================================
# 3. 侧边栏参数
# ==========================================
with st.sidebar:
    st.header("⚙️ 严选参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    st.subheader("📅 回测时间轴")
    days_back = st.slider("回测过去多少个交易日?", 10, 60, 30, 5)
    end_date_input = st.date_input("结束日期", datetime.now() - timedelta(days=15))
    
    st.subheader("🔥 爆发力度")
    vol_mul = st.slider("爆发期量能倍数", 2.0, 5.0, 3.0, 0.5, help="最近3天成交量是潜伏期的多少倍")
    
    run_btn = st.button("🚀 开始回测")

# ==========================================
# 4. 信号检测逻辑 (核心)
# ==========================================
def check_signal_logic(df_hist, code, market_type):
    """
    检查是否符合三日成妖
    df_hist: 包含爆发期和潜伏期的数据
    market_type: '主板' 或 '双创'
    """
    if len(df_hist) < 63: return False, 0.0, 0.0
    
    # 倒序排列，0是最新(信号日)
    df_hist = df_hist.sort_values('trade_date', ascending=False).reset_index(drop=True)
    
    # 切片
    df_burst = df_hist.iloc[0:3]   # 最近3天
    df_latent = df_hist.iloc[3:63] # 前60天潜伏
    
    # 1. 量能判定 (3倍)
    latent_vol_avg = df_latent['vol'].mean()
    if latent_vol_avg == 0: return False, 0.0, 0.0
    
    burst_vol_avg = df_burst['vol'].mean()
    
    # 条件：3天均量 > 潜伏均量 * 倍数
    if burst_vol_avg < latent_vol_avg * vol_mul: return False, 0.0, 0.0
    
    # 2. 涨幅判定 (分板块)
    # 自动识别板块 (代码头 或 market字段)
    is_startup = False
    if '300' in code or '688' in code or '创业' in str(market_type) or '科创' in str(market_type):
        is_startup = True
        
    threshold = 20 if is_startup else 12
    
    p_start = df_burst.iloc[-1]['open'] # Day1 Open
    p_end = df_burst.iloc[0]['close']   # Day3 Close
    cum_rise = (p_end - p_start) / p_start * 100
    
    if cum_rise < threshold: return False, 0.0, 0.0
    
    # 3. 形态判定
    # Day 1 必须是大阳线 (>5%)
    if df_burst.iloc[-1]['pct_chg'] < 5: return False, 0.0, 0.0
    
    # 重心上移: Day 3 收盘价 > Day 1 收盘价
    if p_end <= df_burst.iloc[-1]['close']: return False, 0.0, 0.0
    
    return True, cum_rise, latent_vol_avg

# ==========================================
# 5. 主程序执行
# ==========================================
def run_main():
    if not user_token:
        st.error("请输入 Token")
        return

    # 生成日期序列
    end_str = end_date_input.strftime('%Y%m%d')
    start_dt_est = end_date_input - timedelta(days=days_back * 2 + 20)
    cal_dates = get_trade_cal(user_token, start_dt_est.strftime('%Y%m%d'), end_str)
    
    # 截取我们要回测的区间 (留出最后10天给 D+10 计算)
    if len(cal_dates) < days_back + 10:
        st.error("日期范围太短")
        return
        
    signal_dates = cal_dates[-(days_back + 10) : -10]
    
    st.info(f"正在回测 {signal_dates[0]} 至 {signal_dates[-1]} 期间的所有交易信号...")
    
    all_trades = []
    
    progress_bar = st.progress(0)
    status_log = st.empty()
    
    total_dates = len(signal_dates)
    
    for i, date in enumerate(signal_dates):
        progress_bar.progress((i + 1) / total_dates)
        status_log.text(f"正在扫描: {date} (已发现交易: {len(all_trades)} 笔)")
        
        # 1. 获取当日【精英池】股票
        df_candidates = get_daily_snapshot_filtered(user_token, date)
        
        if df_candidates.empty: continue
        
        # 2. 遍历候选股，检查历史信号
        # 注意：这里我们只对"精英池"里的股票查历史，大大节省时间
        for _, row in df_candidates.iterrows():
            code = row['ts_code']
            market_type = row['market']
            
            df_hist = get_history_for_signal(user_token, code, date)
            is_valid, rise, _ = check_signal_logic(df_hist, code, market_type)
            
            if is_valid:
                # === 3. 模拟交易 (进入 D+1) ===
                # 寻找 D+1 日期
                try:
                    curr_idx = cal_dates.index(date)
                    d1_date = cal_dates[curr_idx + 1]
                except:
                    continue
                
                # 获取未来数据
                df_future = get_future_performance(user_token, code, d1_date, days=12)
                if df_future.empty: continue
                
                # --- 交易推演 ---
                d1 = df_future.iloc[0]
                
                # 风控 A: D+1 开盘低开幅度 < -5% -> 放弃
                open_pct = (d1['open'] - d1['pre_close']) / d1['pre_close'] * 100
                if open_pct < -5:
                    # 记录一笔被放弃的交易 (可选)
                    continue 
                
                # 执行买入
                buy_price = d1['open']
                stop_loss_price = buy_price * 0.90 # -10% 硬止损
                
                trade_record = {
                    '信号日': date,
                    '代码': code,
                    '名称': row['name'],
                    '3日涨幅(%)': round(rise, 1),
                    '买入价': buy_price,
                    '状态': '持有到期'
                }
                
                triggered_stop = False
                
                # 追踪 D+1 到 D+10
                max_days = min(10, len(df_future))
                for day_i in range(max_days):
                    row_f = df_future.iloc[day_i]
                    day_label = f"D+{day_i+1}"
                    
                    # 检查止损
                    if not triggered_stop:
                        if row_f['low'] <= stop_loss_price:
                            triggered_stop = True
                            trade_record['状态'] = '止损离场'
                            ret = -10.0 # 记为 -10%
                        else:
                            ret = (row_f['close'] - buy_price) / buy_price * 100
                    else:
                        ret = -10.0 # 止损后资金曲线躺平
                    
                    # 记录关键节点
                    if day_i+1 in [1, 3, 5, 7, 10]:
                        trade_record[day_label] = round(ret, 2)
                        
                all_trades.append(trade_record)

    progress_bar.empty()
    status_log.empty()
    
    # ==========================================
    # 6. 统计报告
    # ==========================================
    if all_trades:
        df_res = pd.DataFrame(all_trades)
        
        st.success(f"🎉 回测完成！共执行 {len(df_res)} 笔有效交易")
        
        # 1. 核心指标统计
        st.markdown("### 📊 策略表现核心指标")
        cols = st.columns(5)
        days_check = ['D+1', 'D+3', 'D+5', 'D+7', 'D+10']
        
        for idx, day in enumerate(days_check):
            if day in df_res.columns:
                win_rate = len(df_res[df_res[day] > 0]) / len(df_res) * 100
                avg_ret = df_res[day].mean()
                
                with cols[idx]:
                    st.metric(f"{day} 胜率", f"{win_rate:.1f}%")
                    st.metric(f"{day} 均收", f"{avg_ret:.2f}%", delta_color="normal")

        # 2. 资金曲线分布
        st.markdown("### 📈 收益分布 (D+5)")
        if 'D+5' in df_res.columns:
            st.scatter_chart(df_res, x='信号日', y='D+5', color='D+5')
        
        # 3. 详细交易单
        st.markdown("### 📜 交易明细 (按信号日倒序)")
        st.dataframe(df_res.sort_values('信号日', ascending=False).style.applymap(
            lambda x: 'color: red' if isinstance(x, (int, float)) and x > 0 else ('color: green' if isinstance(x, (int, float)) and x < 0 else ''), 
            subset=days_check
        ))
        
        # 4. 止损统计
        stop_count = len(df_res[df_res['状态'] == '止损离场'])
        st.warning(f"风控统计：触发 -10% 止损的交易共有 {stop_count} 笔，占比 {stop_count/len(df_res)*100:.1f}%。")
        
    else:
        st.warning("在此期间未发现符合严选条件的交易。")

if run_btn:
    run_main()
