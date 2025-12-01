# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（V11.0 最终决战版）· 含回测功能
核心权重：
- **资金流 (w_money): 0.35**
- **MACD (w_macd): 0.20** - **60日位置 (w_position): 0.15** (防御/安全边际)
- **波动率 (w_volatility): 0.10** (风险控制)
- 当日涨幅 (w_pct): 0.10
- 换手率 (w_turn): 0.10

功能：
1. 当日选股 (旗舰模式)
2. 历史回测 (验证模式)

**【重要修正】**：已移除 pct_chg > 0 的硬性过滤，允许当日下跌但满足低位和资金流条件的股票入选。
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
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V11.0 最终决战旗舰版（修正版）", layout="wide")
st.title("选股王 · V11.0 最终决战旗舰版（修正版 - 含回测）")
st.markdown("🎯 **本版本已集成 V11.0 最终权重，并移除了当日涨幅 > 0 的硬性过滤，允许逆势低位吸筹股入选。**")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None 

# ---------------------------
# 辅助函数（回测所需）
# ---------------------------
def safe_get(func_name, **kwargs):
    """安全调用 Tushare API"""
    global pro
    if pro is None:
        return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            time.sleep(0.5) 
            return pd.DataFrame(columns=['ts_code']) 
        time.sleep(0.5) 
        return df
    except Exception as e:
        time.sleep(0.5) 
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    """获取 num_days 个交易日作为选股日"""
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 2)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历，请检查 Token 或 Tushare 权限。")
        return []
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()

# 使用非复权数据计算未来收益率 (简化版，无需再拉取复权因子)
def get_future_prices(ts_code, selection_date, days_ahead=[1, 3, 5]):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_date = (d0 - timedelta(days=1)).strftime("%Y%m%d") # 前一天
    end_date = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    
    # 拉取包含选股日和未来日期的非复权数据
    hist = safe_get('daily', ts_code=ts_code, start_date=start_date, end_date=end_date)
    if hist.empty or 'close' not in hist.columns:
        results = {}
        for n in days_ahead: results[f'Return_D{n} (%)'] = np.nan
        return results
    
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    hist = hist.dropna(subset=['close']).sort_values('trade_date').reset_index(drop=True)
    
    # 找到选股日的收盘价 (作为买入价)
    selection_price = hist[hist['trade_date'] == selection_date]['close'].iloc[-1] if not hist[hist['trade_date'] == selection_date].empty else np.nan
    
    results = {}
    
    # 找到选股日之后的所有交易日数据
    future_hist = hist[hist['trade_date'] > selection_date].reset_index(drop=True)

    if pd.isna(selection_price) or selection_price < 1e-9:
        for n in days_ahead: results[f'Return_D{n} (%)'] = np.nan
        return results
        
    for n in days_ahead:
        col_name = f'Return_D{n} (%)'
        if len(future_hist) >= n:
            future_price = future_hist.iloc[n-1]['close']
            results[col_name] = (future_price / selection_price - 1) * 100
        else:
            results[col_name] = np.nan
            
    return results


# ---------------------------
# 侧边栏参数
# ---------------------------
with st.sidebar:
    st.header("模式选择")
    mode = st.radio("选择运行模式", ["当日选股 (旗舰)", "历史回测"])
    
    st.markdown("---")
    st.header("当日选股/通用参数")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=500, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=3.0, step=0.5))
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=200_000_000.0, step=50_000_000.0))
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("10日波动 std 阈值 (%)", value=8.0, step=0.5))
    HIGH_PCT_THRESHOLD = float(st.number_input("大阳线 pct_chg (%)", value=6.0, step=0.5))
    
    if mode == "历史回测":
        st.markdown("---")
        st.header("回测参数")
        backtest_date_end = st.date_input("选择回测结束日期", value=datetime.now().date(), max_value=datetime.now().date())
        BACKTEST_DAYS = int(st.number_input("自动回测天数 (N)", value=20, step=1, min_value=1, max_value=50))
        TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=3, step=1, min_value=1))

# ---------------------------
# Token 输入与初始化 
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 


# ---------------------------
# 核心指标计算函数（V11.0 逻辑）
# ---------------------------
@st.cache_data(ttl=600)
def get_hist(ts_code, end_date, days=60):
    try:
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = safe_get('daily', ts_code=ts_code, start_date=start, end_date=end_date)
        if df.empty: return pd.DataFrame()
        df = df.sort_values('trade_date').reset_index(drop=True)
        return df
    except:
        return pd.DataFrame()

def compute_indicators(df):
    res = {}
    if df.empty or len(df) < 3: return res
    close = df['close'].astype(float); high = df['high'].astype(float); low = df['low'].astype(float)

    try: res['last_close'] = close.iloc[-1]
    except: res['last_close'] = np.nan

    # MACD (12,26,9)
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd'] = ((diff - dea) * 2).iloc[-1]
    else: res['macd'] = np.nan

    # vol ratio and metrics
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        avg_prev5 = np.mean(vols[-6:-1])
        res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9)
        res['vol_last'] = vols[-1]
        res['vol_ma5'] = avg_prev5
    else: res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan

    # 10d return
    if len(close) >= 10:
        res['10d_return'] = close.iloc[-1] / close.iloc[-10] - 1
    else: res['10d_return'] = np.nan

    # volatility (std of last 10 pct_chg)
    if 'pct_chg' in df.columns and len(df) >= 10:
        res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()
    else: res['volatility_10'] = np.nan
        
    # 60日位置计算 (防御因子)
    if len(df) >= 60:
        hist_60 = df.tail(60)
        min_low = hist_60['low'].min(); max_high = hist_60['high'].max()
        current_close = hist_60['close'].iloc[-1]
        if max_high == min_low: res['position_60d'] = 50.0 
        else: res['position_60d'] = (current_close - min_low) / (max_high - min_low) * 100
    else: res['position_60d'] = np.nan 

    return res

# ---------------------------
# 核心选股与评分逻辑（为回测/当日选股服务）
# ---------------------------
def run_selection_for_a_day(trade_date, FINAL_POOL, INITIAL_TOP_N, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, VOL_SPIKE_MULT, VOLATILITY_MAX, HIGH_PCT_THRESHOLD, is_backtest=False):
    """为单个交易日运行选股和评分逻辑"""
    
    # 1. 拉取全市场 Daily 数据
    daily_all = safe_get('daily', trade_date=trade_date) 
    if daily_all.empty or 'ts_code' not in daily_all.columns: return pd.DataFrame(), f"数据缺失或拉取失败：{trade_date}"
    
    if not is_backtest:
        st.write(f"当日记录：{len(daily_all)}，取涨幅前 {INITIAL_TOP_N} 作为初筛。")

    # 2. 初筛与数据合并
    daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
    # 【性能优化】只对涨幅靠前的股票进行初筛，以减少后续合并和拉取历史数据的量。
    pool0 = daily_all.head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)
    
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date,total_mv,circ_mv')
    daily_basic = safe_get('daily_basic', trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    mf_raw = safe_get('moneyflow', trade_date=trade_date)

    pool_merged = pool0.copy()

    # merge stock_basic
    if not stock_basic.empty:
        keep = [c for c in ['ts_code','name','total_mv','circ_mv'] if c in stock_basic.columns]
        pool_merged = pool_merged.merge(stock_basic[keep], on='ts_code', how='left')
    else: pool_merged['name'] = pool_merged['ts_code']

    # merge daily_basic
    if not daily_basic.empty:
        pool_merged = pool_merged.merge(daily_basic, on='ts_code', how='left', suffixes=('_x', ''))
    
    # merge moneyflow robustly
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    if not mf_raw.empty and 'net_mf' in mf_raw.columns:
        moneyflow = mf_raw[['ts_code', 'net_mf']].fillna(0)
    
    if not moneyflow.empty:
        pool_merged = pool_merged.merge(moneyflow, on='ts_code', how='left').fillna({'net_mf': 0.0})
    else: pool_merged['net_mf'] = 0.0
    
    df = pool_merged.copy()
    
    # 3. 执行硬性条件过滤
    df['close'] = pd.to_numeric(df['close'], errors='coerce') 
    df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    
    # 处理 amount 单位 (若为万元，转换为元)
    def normalize_amount(amt):
        if amt > 0 and amt < 1e5: return amt * 10000.0
        return amt
    df['amount'] = df['amount'].apply(normalize_amount)
    
    mask_st = df['name'].str.contains('ST|退', case=False, na=False)
    df = df[~mask_st]
    
    mask_price = (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)
    df = df[mask_price]
    
    mask_turn = df['turnover_rate'] >= MIN_TURNOVER 
    df = df[mask_turn]
    
    mask_amt = df['amount'] >= MIN_AMOUNT
    df = df[mask_amt]
    
    # 【已修正】移除 mask_pct_chg = df['pct_chg'] > 0 的硬性过滤，允许当日下跌的股票入选。
    
    if len(df) == 0: return pd.DataFrame(), f"过滤后无股票：{trade_date}"
    
    # 4. 限制评分池大小并进行深度评分
    clean_df = df.sort_values('pct_chg', ascending=False).head(min(int(FINAL_POOL), 300)).reset_index(drop=True)
    
    records = []
    
    # 使用 st.progress 显示进度条，仅在非回测模式下
    if not is_backtest:
        st.write(f"已通过硬性过滤股票 {len(clean_df)} 支（取涨幅前 {min(int(FINAL_POOL), 300)} 支进入深度评分）")
        pbar = st.progress(0)
    
    for idx, row in enumerate(clean_df.itertuples()):
        ts_code = getattr(row, 'ts_code')
        name = getattr(row, 'name', ts_code)
        pct_chg = getattr(row, 'pct_chg', 0.0)
        turnover_rate = getattr(row, 'turnover_rate', np.nan)
        net_mf = float(getattr(row, 'net_mf', 0.0))
        
        hist = get_hist(ts_code, trade_date, days=60)
        ind = compute_indicators(hist)
        
        # 资金强度代理（简化）
        vol_ratio = ind.get('vol_ratio', 0.0)
        proxy_money = (abs(pct_chg) + 1e-9) * vol_ratio * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
        
        rec = {
            'ts_code': ts_code, 'name': name, 'pct_chg': pct_chg,
            'turnover_rate': turnover_rate, 'net_mf': net_mf,
            'vol_ratio': vol_ratio,
            '10d_return': ind.get('10d_return', np.nan),
            'macd': ind.get('macd', np.nan),
            'volatility_10': ind.get('volatility_10', np.nan),
            'position_60d': ind.get('position_60d', np.nan),
            'vol_last': ind.get('vol_last', np.nan),
            'vol_ma5': ind.get('vol_ma5', np.nan),
            'proxy_money': proxy_money,
            'last_close': ind.get('last_close', np.nan)
        }
        
        # 仅在回测模式下计算未来收益
        if is_backtest:
            future_returns = get_future_prices(ts_code, trade_date)
            rec.update(future_returns)
            
        records.append(rec)
        if not is_backtest: pbar.progress((idx+1)/len(clean_df))
    
    if not is_backtest: pbar.progress(1.0)
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), f"评分列表为空：{trade_date}"
    
    # 5. 风险过滤（V11.0 逻辑）
    
    # C: 巨量放量大阳 -> vol_last > vol_ma5 * VOL_SPIKE_MULT
    mask_vol_spike = (fdf['vol_last'] > (fdf['vol_ma5'] * VOL_SPIKE_MULT))
    fdf = fdf[~mask_vol_spike]

    # D: 极端波动 -> volatility_10 > VOLATILITY_MAX
    mask_volatility = fdf['volatility_10'] > VOLATILITY_MAX
    fdf = fdf[~mask_volatility]

    # E: 高位大阳线 -> 60日位置 > 80% 且 pct_chg > HIGH_PCT_THRESHOLD (简化风险过滤)
    mask_high_big = (fdf['position_60d'] > 80.0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
    fdf = fdf[~mask_high_big]

    # 6. 归一化与 V11.0 策略精调评分 
    def norm_col(s):
        s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        mn = s.min(); mx = s.max()
        if mx - mn < 1e-9: return pd.Series([0.5]*len(s), index=s.index)
        return (s - mn) / (mx - mn)

    fdf['s_pct'] = norm_col(fdf['pct_chg'])
    fdf['s_turn'] = norm_col(fdf['turnover_rate'])
    fdf['s_volratio'] = norm_col(fdf['vol_ratio'])
    
    if 'net_mf' in fdf.columns and fdf['net_mf'].abs().sum() > 0:
        fdf['s_money'] = norm_col(fdf['net_mf']) # 优先使用 net_mf
    else:
        fdf['s_money'] = norm_col(fdf['proxy_money']) # 否则使用代理资金流
        
    fdf['s_macd'] = norm_col(fdf['macd'])
    fdf['s_10d'] = norm_col(fdf['10d_return'])
    
    fdf['s_volatility'] = 1 - norm_col(fdf['volatility_10']) # 波动率越低得分越高
    fdf['s_position'] = 1 - (fdf['position_60d'].fillna(50.0) / 100) # 60日位置越低得分越高


    # V11.0 最终权重配置 (总和 1.00)
    w_money = 0.35; w_macd = 0.20; w_position = 0.15; w_volatility = 0.10
    w_pct = 0.10; w_turn = 0.10
    w_volratio = 0.00; w_10d = 0.00; w_rsl = 0.00
    
    score = (
        fdf['s_pct'] * w_pct + fdf['s_turn'] * w_turn + 
        fdf['s_money'] * w_money + 
        fdf['s_macd'] * w_macd + 
        
        # 防御项
        fdf['s_position'] * w_position + 
        fdf['s_volatility'] * w_volatility + 
        
        # 归零项
        fdf['s_volratio'] * w_volratio + fdf['s_10d'] * w_10d
    )
    fdf['综合评分'] = score * 100
    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index += 1

    return fdf, None

# ---------------------------
# 主运行块
# ---------------------------
if mode == "当日选股 (旗舰)":
    
    def find_last_trade_day(max_days=20):
        """Helper for daily mode"""
        today = datetime.now().date()
        for i in range(max_days):
            d = today - timedelta(days=i)
            ds = d.strftime("%Y%m%d")
            df = safe_get('daily', trade_date=ds)
            if not df.empty:
                return ds
        return None
        
    last_trade = find_last_trade_day(20) # 寻找最近交易日
    if not last_trade:
        st.error("无法找到最近交易日，请检查网络或 Token 权限。")
        st.stop()
    st.info(f"🚀 **当前运行模式：当日选股 (旗舰)** | 选股基准日：{last_trade}")
    st.markdown("---")
    
    st.write(f"正在进行当日选股和评分...")
    
    scored_df, error = run_selection_for_a_day(
        last_trade, FINAL_POOL, INITIAL_TOP_N, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, VOL_SPIKE_MULT, VOLATILITY_MAX, HIGH_PCT_THRESHOLD, is_backtest=False
    )
    
    if error:
        st.error(f"选股失败：{error}")
    elif not scored_df.empty:
        st.success(f"评分完成：总候选 {len(scored_df)} 支，显示 Top {min(TOP_DISPLAY, len(scored_df))}。")
        display_cols = ['name','ts_code','综合评分','pct_chg','turnover_rate','net_mf','position_60d','volatility_10']
        
        st.dataframe(scored_df[display_cols].head(TOP_DISPLAY), use_container_width=True)
        
        out_csv = scored_df[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
        st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")
        
    st.markdown("---")
    st.markdown("### 小结与操作提示")
    st.info("当前使用 V11.0 最佳权重：资金流 $0.35$ + MACD $0.20$ + 60日低位 $0.15$。请根据 $\mathbf{9:40-10:05}$ 量价节奏择优介入。")

# ---------------------------
# 历史回测块
# ---------------------------
elif mode == "历史回测":
    st.info(f"🔬 **当前运行模式：历史回测** | 结束日期：{backtest_date_end.strftime('%Y%m%d')}，天数：{BACKTEST_DAYS}")
    st.markdown("---")

    if st.button(f"🚀 开始 {BACKTEST_DAYS} 日回测 (Top {TOP_BACKTEST})"):
        
        trade_days_str = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
        if not trade_days_str:
            st.error("无法获取交易日列表，请检查日期或 Token。")
            st.stop()
        
        st.header(f"📈 正在进行 {BACKTEST_DAYS} 个交易日的回测...")
        
        results_list = []
        total_days = len(trade_days_str)
        
        progress_text = st.empty()
        my_bar = st.progress(0)
        
        for i, trade_date in enumerate(trade_days_str):
            progress_text.text(f"🚀 正在处理第 {i+1}/{total_days} 个交易日：{trade_date}")
            
            daily_result_df, error = run_selection_for_a_day(
                trade_date, FINAL_POOL, INITIAL_TOP_N, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, VOL_SPIKE_MULT, VOLATILITY_MAX, HIGH_PCT_THRESHOLD, is_backtest=True
            )
            
            if error:
                st.warning(f"跳过 {trade_date}：{error}")
            elif not daily_result_df.empty:
                daily_result_df['Trade_Date'] = trade_date
                results_list.append(daily_result_df.head(TOP_BACKTEST))
                
            my_bar.progress((i + 1) / total_days)

        progress_text.text("✅ 回测完成，正在汇总结果...")
        my_bar.empty()
        
        if not results_list:
            st.error("所有交易日的回测均失败或无结果。")
            st.stop()
            
        all_results = pd.concat(results_list)
        
        st.header(f"📊 最终平均回测结果 (Top {TOP_BACKTEST}，共 {total_days} 个交易日)")
        
        for n in [1, 3, 5]:
            col = f'Return_D{n} (%)' 
            
            filtered_returns = all_results.copy()
            valid_returns = filtered_returns.dropna(subset=[col])

            if not valid_returns.empty:
                avg_return = valid_returns[col].mean()
                hit_rate = (valid_returns[col] > 0).sum() / len(valid_returns) * 100 if len(valid_returns) > 0 else 0.0
                total_count = len(valid_returns)
            else:
                avg_return = np.nan
                hit_rate = 0.0
                total_count = 0
                
            st.metric(f"Top {TOP_BACKTEST}：D+{n} 平均收益 / 准确率", 
                      f"{avg_return:.2f}% / {hit_rate:.1f}%", 
                      help=f"总有效样本数：{total_count}。**V11.0 策略表现。**")

        st.header("📋 每日回测详情 (Top K 明细)")
        
        display_cols = ['Trade_Date', 'name', 'ts_code', '综合评分', 
                        'pct_chg', 'position_60d',
                        'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)']
        
        st.dataframe(all_results[display_cols].sort_values('Trade_Date', ascending=False), use_container_width=True)
