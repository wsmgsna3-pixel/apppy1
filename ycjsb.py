# -*- coding: utf-8 -*-
"""
选股王 · V30.10 Alpha 恢复版 (冷却机制降级)
V30.10.0 更新：
1. **核心恢复**：完全移除 V30.9 引入的 RSI > 85 和 Bias > 25% 的硬性过滤。
2. **Alpha保留**：保留 V30.8 实体力度 > 0.7 的核心 Alpha 源。
3. **冷却降级**：RSI 和 Bias 降级为评分项，权重大幅降低，避免误杀强势股。
4. **目标**：恢复 D+5 收益至 > 2.0%，同时通过评分优化略微提升短周期胜率。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time  
warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量初始化
# (此处省略未修改的全局变量和 Tushare 配置部分，与 V30.9 保持一致)
# ---------------------------

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V30.10：Alpha 恢复版", layout="wide")
st.title("选股王 V30.10：Alpha 恢复版（💡 恢复惯性，冷却降级）")
st.markdown("🎯 **V30.10 策略核心：** 鉴于 V30.9 硬性过滤导致 Alpha 死亡，本版本**移除 RSI 和 Bias 的绝对过滤**，将其降级为评分辅助项。全力确保 V30.8 的实体强势股入围，并用安全指标做精选。")

# ----------------------------------------------------
# 侧边栏参数 (仅展示修改部分)
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input("回测结束日期", value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("自动回测天数 (N)", value=50, step=1))
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = int(st.number_input("入围评分数量", value=100)) 
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=5)) 
    
    st.markdown("---")
    st.header("🛡️ V30.10 核心 Alpha 参数 (V30.8 保持)")
    MAX_UPPER_SHADOW = st.number_input("最大上影线比例 (%)", value=4.0)
    MIN_BODY_POS = st.number_input("最低实体位置 (0-1)", value=0.7)
    MAX_TURNOVER_RATE = st.number_input("最大换手率 (%)", value=20.0)
    
    st.markdown("---")
    st.header("🧊 冷却因子 (V30.10 仅用于评分)")
    st.write("RSI/Bias **不再硬性过滤**，仅用于评分降权。")

    # 隐藏的固定过滤参数
    MIN_PRICE, MAX_PRICE = 10.0, 300.0
    MIN_TURNOVER = 5.0 
    MIN_CIRC_MV_BILLIONS, MAX_CIRC_MV_BILLIONS = 20.0, 200.0
    MIN_AMOUNT = 100000000

# ---------------------------
# Token 
# (此处省略未修改的 Token 配置部分)
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 


# ----------------------------------------------------------------------
# 核心回测逻辑函数 
# ----------------------------------------------------------------------
# 注意：以下函数体需包含 V30.9 的指标计算，但移除其硬性过滤逻辑。
# 为简洁，此处仅展示关键修改部分，完整代码应包含所有辅助函数（get_trade_days, get_qfq_data等）
# 
# 假设 compute_indicators 已经包含了 RSI 和 Bias 的计算。
# ----------------------------------------------------------------------

# (此处省略辅助函数 get_trade_days, fetch_and_cache_daily_data, get_all_historical_data, 
# get_qfq_data_v4_optimized_final, get_future_prices, calculate_rsi, get_market_state,
# 以及 compute_indicators, 保持 V30.9 中的逻辑以计算 rsi 和 bias)
@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    # ... (此处是 V30.9 的计算逻辑，确保计算了 macd, ma20, ma60, position_60d, rsi_12, bias_20)
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    res = {}
    if df.empty or len(df) < 26: return res # 至少需要26天计算MACD
    
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    close = df['close']
    res['last_close'] = close.iloc[-1]
    res['last_open'] = df['open'].iloc[-1]
    res['last_high'] = df['high'].iloc[-1]
    res['last_low'] = df['low'].iloc[-1]
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    res['macd_val'] = ((diff - dea) * 2).iloc[-1]
        
    # MA & Bias
    res['ma20'] = close.tail(20).mean()
    res['ma60'] = close.tail(60).mean()
    
    if pd.notna(res['ma20']) and res['ma20'] > 0:
        res['bias_20'] = (res['last_close'] - res['ma20']) / res['ma20'] * 100
    else: res['bias_20'] = 0

    # Position
    hist_60 = df.tail(60)
    res['position_60d'] = (close.iloc[-1] - hist_60['low'].min()) / (hist_60['high'].max() - hist_60['low'].min() + 1e-9) * 100
        
    # RSI (12)
    rsi_series = calculate_rsi(close, period=12)
    res['rsi_12'] = rsi_series.iloc[-1]
    
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    return res

def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS): 
    market_state = get_market_state(last_trade)
    
    # ... (此处省略数据获取与基础过滤逻辑，与 V30.9 保持一致) ...
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失 {last_trade}"
    daily_basic = safe_get('daily_basic', trade_date=last_trade, fields='ts_code,turnover_rate,circ_mv,total_mv,amount')
    mf_raw = safe_get('moneyflow', trade_date=last_trade) 
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    if not daily_basic.empty: df = df.merge(daily_basic, on='ts_code', how='left')
    else: df['turnover_rate'] = 0; df['circ_mv'] = 0; df['amount'] = 0
    if not mf_raw.empty:
        mf = mf_raw[['ts_code','net_mf_amount']].rename(columns={'net_mf_amount':'net_mf'})
        df = df.merge(mf, on='ts_code', how='left')
    df['net_mf'] = df['net_mf'].fillna(0)
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')]
    df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
    df = df[(datetime.strptime(last_trade, "%Y%m%d") - df['list_date']).dt.days > 120]
    
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS) & (df['circ_mv_billion'] <= MAX_CIRC_MV_BILLIONS)]
    df = df[df['turnover_rate'] >= MIN_TURNOVER]
    df = df[df['amount'] >= MIN_AMOUNT]
    df = df[df['turnover_rate'] <= MAX_TURNOVER_RATE] 

    if len(df) == 0: return pd.DataFrame(), "过滤后无标的"

    limit_mf = int(FINAL_POOL * 0.5)
    df_mf = df.sort_values('net_mf', ascending=False).head(limit_mf)
    limit_pct = FINAL_POOL - len(df_mf)
    df_pct = df[~df['ts_code'].isin(df_mf['ts_code'])].sort_values('pct_chg', ascending=False).head(limit_pct)
    final_candidates = pd.concat([df_mf, df_pct]).reset_index(drop=True)

    
    # 4. 深度计算
    records = []
    for row in final_candidates.itertuples():
        ts_code = row.ts_code
        ind = compute_indicators(ts_code, last_trade)
        d0_close = ind.get('last_close', np.nan)
        d0_high = ind.get('last_high', np.nan)
        d0_low = ind.get('last_low', np.nan)
        d0_ma60 = ind.get('ma60', np.nan)
        d0_ma20 = ind.get('ma20', np.nan)
        d0_pos60 = ind.get('position_60d', np.nan)
        d0_rsi = ind.get('rsi_12', 50)
        d0_bias = ind.get('bias_20', 0)
        
        # --- V30.10 过滤器核心 (恢复 V30.8 逻辑) ---
        
        # 1. 趋势保护
        if pd.isna(d0_ma60) or d0_close < d0_ma60: continue
            
        # 2. 上影线 (V30.8)
        if pd.notna(d0_high) and pd.notna(d0_close) and d0_close > 0:
            upper_shadow = (d0_high - d0_close) / d0_close * 100
            if upper_shadow > MAX_UPPER_SHADOW: continue 
        
        # 3. 实体位置 (V30.8)
        if pd.notna(d0_high) and pd.notna(d0_low) and pd.notna(d0_close):
            range_len = d0_high - d0_low
            if range_len > 0:
                body_pos = (d0_close - d0_low) / range_len
                if body_pos < MIN_BODY_POS: continue 

        # 4. V30.9 **硬性过滤移除**：允许 RSI 和 Bias 高的股票进入评分阶段。
        
        # 5. 弱市防御
        if market_state == 'Weak':
            if pd.isna(d0_ma20) or d0_close < d0_ma20: continue
            if pd.notna(d0_pos60) and d0_pos60 > 20.0: continue

        # --- 通过过滤，计算收益 ---
        if pd.notna(d0_close):
            future = get_future_prices(ts_code, last_trade, d0_close)
            rec = {
                'ts_code': ts_code, 'name': row.name,
                'Close': row.close, 'Pct_Chg': row.pct_chg,
                'Turnover': row.turnover_rate,
                'macd': ind.get('macd_val', 0),
                'rsi': d0_rsi, # 计入结果用于评分
                'bias': d0_bias, # 计入结果用于评分
                'net_mf': row.net_mf,
                'Return_D1 (%)': future.get('Return_D1', np.nan),
                'Return_D3 (%)': future.get('Return_D3', np.nan),
                'Return_D5 (%)': future.get('Return_D5', np.nan),
            }
            records.append(rec)
            
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), "深度筛选后无标的"
    
    # 5. 评分 (V30.10 核心：冷却降级为低权重评分)
    def normalize(s): 
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    
    fdf['s_mf'] = normalize(fdf['net_mf'])
    # 冷却安全因子：值越小越安全，所以用 1 - Normalize
    fdf['s_rsi_safety'] = 1 - normalize(fdf['rsi']) 
    fdf['s_bias_safety'] = 1 - normalize(fdf['bias']) 
    
    # 综合安全分 (Beta)
    fdf['s_safety'] = (fdf['s_rsi_safety'] * 0.5 + fdf['s_bias_safety'] * 0.5) 

    if market_state == 'Strong':
        fdf['策略'] = 'V30.10 Alpha 强市恢复版'
        fdf_strong = fdf[fdf['macd'] > 0].copy()
        if fdf_strong.empty: fdf['Score'] = 0
        else:
            # Alpha 权重 (MACD, MF) 80% + Beta 权重 (Safety) 20%
            fdf_strong['s_alpha'] = fdf_strong['macd'] * 10000 + fdf_strong['s_mf'] * 50
            fdf_strong['Score'] = fdf_strong['s_alpha'] * 0.8 + fdf_strong['s_safety'] * 0.2
            fdf = fdf_strong.sort_values('Score', ascending=False)
    else:
        fdf['策略'] = 'V30.10 Alpha 弱市恢复版'
        fdf['s_macd'] = normalize(fdf['macd'])
        fdf['s_alpha'] = fdf['s_macd'] * 0.6 + fdf['s_mf'] * 0.4
        fdf['Score'] = fdf['s_alpha'] * 0.8 + fdf['s_safety'] * 0.2
        fdf = fdf.sort_values('Score', ascending=False)
        
    return fdf.head(TOP_BACKTEST), None

# ---------------------------
# 主运行块
# ---------------------------
if st.button(f"🚀 运行 V30.10 Alpha 恢复版回测 ({BACKTEST_DAYS}天)"):
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not get_all_historical_data(trade_days): st.stop()
    
    results = []
    bar = st.progress(0)
    for i, date in enumerate(trade_days):
        res, err = run_backtest_for_a_day(date, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS)
        if not res.empty:
            res['Trade_Date'] = date
            results.append(res)
        bar.progress((i+1)/len(trade_days))
    bar.empty()
    
    if results:
        all_res = pd.concat(results)
        st.header("📊 V30.10 平均回测结果")
        for n in [1, 3, 5]:
            col = f'Return_D{n} (%)'
            valid = all_res.dropna(subset=[col])
            if not valid.empty:
                avg = valid[col].mean()
                win = (valid[col] > 0).mean() * 100
                st.metric(f"D+{n} 收益/胜率", f"{avg:.2f}% / {win:.1f}%")
                
        st.dataframe(all_res[['Trade_Date','name','Pct_Chg','Turnover','rsi','bias','Return_D1 (%)']].head(100))
