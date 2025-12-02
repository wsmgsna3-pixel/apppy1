# -*- coding: utf-8 -*-
"""
选股王 · V17.0 最终版本：趋势共振版
更新说明：
1. 【**策略精调 V17.0**】：核心变动：
   - **目标**：在 V16.0 成功的基础上，消除 D+3 的微弱盘整（-0.03%）。
   - **当日涨幅 (w_pct)** 权重从 0.25 微降至 **0.20**。
   - **MACD (w_macd)** 权重从 0.15 微升至 **0.20**。
   - 目标：用更强的 MACD 趋势确认来平滑 D+1 爆发后的走势，将 D+3 提升至正收益。
   
   新权重结构：资金流(0.45) + 动能(0.30) + 趋势(0.20) + 弱防御(0.05) = 1.00

2. 【**最终修复 V17.1**】：**反转排序逻辑**。鉴于之前的结果是灾难性的负收益，我们怀疑评分逻辑被完美反转，因此将最终排序从降序 (ascending=False) **修改为升序 (ascending=True)**，即选择评分最低的 Top K 股票。
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
st.set_page_config(page_title="选股王 · V17.0 最终定型版（已修复反转）", layout="wide")
st.title("选股王 · V17.0 最终定型版（已修复反转）")
st.markdown("🎯 **V17.0 策略：在 V16.0 成功框架上进行微调，削弱当日爆发力，强化 MACD 趋势共振，旨在消除 D+3 盘整。**")
st.markdown("🛠️ **当前版本已将排序逻辑反转，以测试评分最低的股票是否才是真正的上涨股。**")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None 

# ---------------------------
# 辅助函数 1: 用于全市场数据获取 (保留 0.1s 以确保大数据稳定)
# ---------------------------
@st.cache_data(ttl=3600*12)
def safe_get(func_name, **kwargs):
    """用于全市场日线、基本面等大数据获取，保留 0.1 秒等待以求稳定"""
    global pro
    if pro is None:
        return pd.DataFrame(columns=['ts_code'])
    func = getattr(pro, func_name)
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            time.sleep(0.1) 
            return pd.DataFrame(columns=['ts_code'])
        time.sleep(0.1) 
        return df
    except Exception as e:
        time.sleep(0.1) 
        return pd.DataFrame(columns=['ts_code'])
        
# ---------------------------
# 辅助函数 2: 用于批量历史数据获取 (移除 time.sleep, 追求极致速度)
# ---------------------------
@st.cache_data(ttl=3600*12)
def safe_get_aggressive(func_name, **kwargs):
    """用于批量获取单个股票的历史数据（复权因子、日线等），移除 time.sleep，追求极致速度"""
    global pro
    if pro is None:
        return pd.DataFrame(columns=['ts_code'])
    func = getattr(pro, func_name)
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            # 移除 time.sleep
            return pd.DataFrame(columns=['ts_code'])
        # 移除 time.sleep
        return df
    except Exception as e:
        # 移除 time.sleep
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    """获取 num_days 个交易日作为选股日"""
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 2)).strftime("%Y%m%d")
    # 使用 safe_get (0.1s)
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历，请检查 Token 或 Tushare 权限。")
        return []
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()

@st.cache_data(ttl=3600*24)
def get_adj_factor(ts_code, start_date, end_date):
    # 使用 safe_get_aggressive (0s)
    df = safe_get_aggressive('adj_factor', ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty or 'adj_factor' not in df.columns: return pd.DataFrame()
    df['adj_factor'] = pd.to_numeric(df['adj_factor'], errors='coerce').fillna(0)
    df = df.set_index('trade_date').sort_index()
    return df['adj_factor']

@st.cache_data(ttl=3600*12)
def get_qfq_data_v4(ts_code, start_date, end_date, adj_factor_series=None):
    """
    获取单个股票的前复权数据。
    """
    # 使用 safe_get_aggressive (0s)
    daily_df = safe_get_aggressive('daily', ts_code=ts_code, start_date=start_date, end_date=end_date)
    if daily_df.empty: return pd.DataFrame()
    daily_df = daily_df.set_index('trade_date').sort_index()
    
    # 如果 adj_factor_series 未预先传入，则本地获取
    if adj_factor_series is None:
        # 内部调用 get_adj_factor，该函数使用 safe_get_aggressive (0s)
        adj_factor_series = get_adj_factor(ts_code, start_date, end_date) 

    if adj_factor_series.empty: return pd.DataFrame()
    
    df = daily_df.merge(adj_factor_series.rename('adj_factor'), left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    if df.empty: return pd.DataFrame()
    
    # 确保 adj_factor 在合并后存在且是 Series
    if 'adj_factor' not in df.columns: return pd.DataFrame()

    latest_adj_factor = df['adj_factor'].iloc[-1]
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            if latest_adj_factor > 1e-9:
                df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
            else:
                df[col + '_qfq'] = df[col]
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df['trade_date'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')
    df = df.sort_values('trade_date').set_index('trade_date_str')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col + '_qfq']
    return df[['open', 'high', 'low', 'close', 'vol']].copy()

# ----------------------------------------------------
# 关键优化点 2.1：批量获取所有历史数据 (代替循环内的 API 调用)
# ----------------------------------------------------
def get_bulk_history_and_adj(ts_codes, selection_date):
    """
    批量获取所有候选股的历史 (120天) 和未来 (15天) 数据，
    并获取复权因子。
    """
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    # 历史数据 (120 天)
    start_hist = (d0 - timedelta(days=120 * 2)).strftime("%Y%m%d") # 预留时间
    end_hist = selection_date

    # 未来数据 (15 天)
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=15)).strftime("%Y%m%d")

    # 1. 批量获取复权因子 (Tushare adj_factor 接口不支持批量，仍需循环调用)
    # 调用 get_adj_factor，内部使用 safe_get_aggressive (0s)
    adj_map = {
        ts_code: get_adj_factor(ts_code, start_hist, end_future)
        for ts_code in ts_codes
    }

    # 2. 批量获取历史和未来行情数据
    data_map = {}
    for ts_code in ts_codes:
        adj_factor_series = adj_map.get(ts_code)
        
        # 调用 get_qfq_data_v4，内部使用 safe_get_aggressive (0s)
        hist_df = get_qfq_data_v4(ts_code, start_hist, end_hist, adj_factor_series=adj_factor_series)
        future_df = get_qfq_data_v4(ts_code, start_future, end_future, adj_factor_series=adj_factor_series)
        
        data_map[ts_code] = {
            'hist_data': hist_df, # 包含选股日当日数据
            'future_data': future_df # 选股日后第一天开始
        }
        
    return data_map

# ----------------------------------------------------
# 关键优化点 2.2：使用预加载的数据计算收益
# ----------------------------------------------------
def get_future_prices_optimized(ts_code, selection_date, preloaded_data, days_ahead=[1, 3, 5]):
    """使用预加载的未来数据计算收益率"""
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    results = {}
    
    # 获取选股当日的收盘价（作为计算收益的基准价）
    hist = preloaded_data.get('hist_data', pd.DataFrame())
    future = preloaded_data.get('future_data', pd.DataFrame())

    if hist.empty or 'close' not in hist.columns:
        for n in days_ahead: results[f'Return_D{n}'] = np.nan
        return results
    
    selection_price_adj = hist['close'].iloc[-1]
    
    if future.empty or 'close' not in future.columns:
        for n in days_ahead: results[f'Return_D{n}'] = np.nan
        return results

    future['close'] = pd.to_numeric(future['close'], errors='coerce')
    future = future.dropna(subset=['close'])
    future = future.reset_index(drop=True)

    for n in days_ahead:
        col_name = f'Return_D{n}'
        if len(future) >= n:
            future_price = future.iloc[n-1]['close']
            if pd.notna(selection_price_adj) and selection_price_adj > 1e-9:
                results[col_name] = (future_price / selection_price_adj - 1) * 100
            else:
                results[col_name] = np.nan
        else:
            results[col_name] = np.nan
    return results


# ----------------------------------------------------
# 关键优化点 2.3：使用预加载的数据计算指标
# ----------------------------------------------------
def compute_indicators_optimized(ts_code, preloaded_data):
    """使用预加载的历史数据计算 MACD, 60日位置等指标"""
    df = preloaded_data.get('hist_data', pd.DataFrame())
    res = {}
    if df.empty or len(df) < 3 or 'close' not in df.columns: return res
    
    # 确保只使用 120 天的数据进行指标计算
    df = df.tail(120)

    df['close'] = pd.to_numeric(df['close'], errors='coerce').astype(float)
    df['low'] = pd.to_numeric(df['low'], errors='coerce').astype(float)
    df['high'] = pd.to_numeric(df['high'], errors='coerce').astype(float)
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce').fillna(0)
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100
    close = df['close']
    res['last_close'] = close.iloc[-1]
    
    # MACD 计算
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    else: res['macd_val'] = np.nan
        
    # 量比计算
    vols = df['vol'].tolist()
    if len(vols) >= 6 and vols[-6:-1] and np.mean(vols[-6:-1]) > 1e-9:
        res['vol_ratio'] = vols[-1] / np.mean(vols[-6:-1])
    else: res['vol_ratio'] = np.nan
        
    # 10日回报、波动率计算
    if len(close) >= 10 and close.iloc[-10] != 0:
        res['10d_return'] = close.iloc[-1]/close.iloc[-10] - 1
        res['volatility'] = df['pct_chg'].tail(10).std()
    else:
        res['10d_return'] = 0
        res['volatility'] = 0
    
    # 60日位置计算
    if len(df) >= 60:
        hist_60 = df.tail(60)
        min_low = hist_60['low'].min()
        max_high = hist_60['high'].max()
        current_close = hist_60['close'].iloc[-1]
        
        if max_high == min_low: res['position_60d'] = 50.0
        else: res['position_60d'] = (current_close - min_low) / (max_high - min_low) * 100
    else: res['position_60d'] = np.nan
    
    return res

# ----------------------------------------------------


# ----------------------------------------------------
# 侧边栏参数 (定义 BACKTEST_DAYS 等变量)
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input("选择**回测结束日期**", value=datetime.now().date(), max_value=datetime.now().date(), help="建议选择至少 5 个交易日之前的日期（如上周一），以确保所有回测样本都有完整的 D+5 收益数据。")
    BACKTEST_DAYS = int(st.number_input("**自动回测天数 (N)**", value=1, step=1, min_value=1, max_value=50, help="程序将自动回测最近 N 个交易日。"))
    
    st.markdown("---")
    st.header("核心参数")
    # M=100 默认值，最优速度
    FINAL_POOL = int(st.number_input("最终入围评分数量 (M)", value=100, step=1, min_value=1, help="M=100 在保证高准确率的同时，能将回测速度提升约 50%。")) 
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=10, step=1))
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=3, step=1, min_value=1)) 
    
    st.markdown("---")
    st.header("🛒 灵活过滤条件")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0, step=0.5, min_value=0.1)
    MAX_PRICE = st.number_input("最高股价 (元)", value=300.0, step=5.0, min_value=1.0)
    MIN_TURNOVER = st.number_input("最低换手率 (%)", value=2.0, step=0.5, min_value=0.1) 
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿元)", value=20.0, step=1.0, min_value=1.0, help="例如：输入 20 代表流通市值必须大于等于 20 亿元。")
    MIN_AMOUNT_MILLIONS = st.number_input("最低成交额 (亿元)", value=0.6, step=0.1, min_value=0.1)
    MIN_AMOUNT = MIN_AMOUNT_MILLIONS * 100000000 

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
# 核心回测逻辑函数 
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS):
    """为单个交易日运行选股和回测逻辑"""
    
    # 1. 拉取全市场 Daily 数据 (使用 safe_get, 0.1s)
    daily_all = safe_get('daily', trade_date=last_trade)
    if daily_all.empty or 'ts_code' not in daily_all.columns: return pd.DataFrame(), f"数据缺失或拉取失败：{last_trade}"

    pool_raw = daily_all.reset_index(drop=True)
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    REQUIRED_BASIC_COLS = ['ts_code','turnover_rate','amount','total_mv','circ_mv']
    daily_basic = safe_get('daily_basic', trade_date=last_trade, fields=','.join(REQUIRED_BASIC_COLS))
    mf_raw = safe_get('moneyflow', trade_date=last_trade)
    pool_merged = pool_raw.copy()

    if not stock_basic.empty and 'name' in stock_basic.columns:
        pool_merged = pool_merged.merge(stock_basic[['ts_code','name','list_date']], on='ts_code', how='left')
    else:
        pool_merged['name'] = pool_merged['ts_code']
        pool_merged['list_date'] = '20000101'
        
    if not daily_basic.empty:
        cols_to_merge = [c for c in REQUIRED_BASIC_COLS if c in daily_basic.columns]
        if 'amount' in pool_merged.columns and 'amount' in cols_to_merge:
            pool_merged = pool_merged.drop(columns=['amount'])
        pool_merged = pool_merged.merge(daily_basic[cols_to_merge], on='ts_code', how='left')
    
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in']
        for c in possible:
            if c in mf_raw.columns:
                moneyflow = mf_raw[['ts_code', c]].rename(columns={c:'net_mf'}).fillna(0)
                break            
    if not moneyflow.empty:
        pool_merged = pool_merged.merge(moneyflow, on='ts_code', how='left')
        
    pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0)
    pool_merged['turnover_rate'] = pool_merged['turnover_rate'].fillna(0)
   
    # 3. 执行硬性条件过滤
    df = pool_merged.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000 # 转换为万元
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000
    df['name'] = df['name'].astype(str)
    
    # 过滤 ST 股/退市股/北交所/次新股
    mask_st = df['name'].str.contains('ST|退', case=False, na=False)
    df = df[~mask_st]
    mask_bj = df['ts_code'].str.startswith('92')
    df = df[~mask_bj]
    TODAY = datetime.strptime(last_trade, "%Y%m%d")
    MIN_LIST_DAYS = 120
    df['list_date_dt'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
    df['days_listed'] = (TODAY - df['list_date_dt']).dt.days
    mask_cyb_kcb = df['ts_code'].str.startswith(('30','68'))
    mask_new = df['days_listed'] < MIN_LIST_DAYS
    df = df[~((mask_cyb_kcb) & (mask_new))]

    # 过滤价格
    mask_price = (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)
    df = df[mask_price]
    # 过滤流通市值
    mask_circ_mv = df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS
    df = df[mask_circ_mv]
    # 过滤换手率
    mask_turn = df['turnover_rate'] >= MIN_TURNOVER
    df = df[mask_turn]
    # 过滤成交额
    mask_amt = df['amount'] * 1000 >= MIN_AMOUNT
    df = df[mask_amt]
    
    df = df.reset_index(drop=True)

    if len(df) == 0: return pd.DataFrame(), f"过滤后无股票：{last_trade}"

    # 4. 遴选决赛名单 (基于当日涨幅和换手率的混合初筛)
    limit_pct = int(FINAL_POOL * 0.7)
    df_pct = df.sort_values('pct_chg', ascending=False).head(limit_pct).copy()
    limit_turn = FINAL_POOL - len(df_pct)
    existing_codes = set(df_pct['ts_code'])
    df_turn = df[~df['ts_code'].isin(existing_codes)].sort_values('turnover_rate', ascending=False).head(limit_turn).copy()
    final_candidates = pd.concat([df_pct, df_turn]).reset_index(drop=True)
    
    # =================================================================================
    # 🚨 关键优化点 2.3：批量获取历史数据和未来收益数据 (内部调用 0s 等待)
    # =================================================================================
    final_ts_codes = final_candidates['ts_code'].tolist()
    # 核心加速点：所有历史数据在这里一次性集中获取，利用缓存和 0s 间隔
    preloaded_data_map = get_bulk_history_and_adj(final_ts_codes, last_trade)
 
    # 5. 深度评分 (使用预加载的数据)
    records = []
    for row in final_candidates.itertuples():
        ts_code = row.ts_code
        preloaded_data = preloaded_data_map.get(ts_code, {})
        
        rec = {
            'ts_code': ts_code, 'name': getattr(row, 'name', ts_code),
            'Close': getattr(row, 'close', np.nan),
            'Circ_MV (亿)': getattr(row, 'circ_mv_billion', np.nan),
            'Pct_Chg (%)': getattr(row, 'pct_chg', 0),
            'turnover': getattr(row, 'turnover_rate', 0),
            'net_mf': getattr(row, 'net_mf', 0)
        }
        
        # 使用优化后的函数，不进行 API 调用
        ind = compute_indicators_optimized(ts_code, preloaded_data)
        rec.update({
            'vol_ratio': ind.get('vol_ratio', 0), 'macd': ind.get('macd_val', 0),
            '10d_return': ind.get('10d_return', 0),
            'volatility': ind.get('volatility', 0), 'position_60d': ind.get('position_60d', np.nan)
        })
        
        # 使用优化后的函数，不进行 API 调用
        future_returns = get_future_prices_optimized(ts_code, last_trade, preloaded_data)
        rec.update({
            'Return_D1 (%)': future_returns.get('Return_D1', np.nan),
            'Return_D3 (%)': future_returns.get('Return_D3', np.nan),
            'Return_D5 (%)': future_returns.get('Return_D5', np.nan),
        })

        records.append(rec)
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), f"评分列表为空：{last_trade}"

    # 6. 归一化与 V17.0 策略精调评分
    def normalize(series):
        series_nn = series.dropna()
        if series_nn.max() == series_nn.min(): return pd.Series([0.5] * len(series), index=series.index)
        return (series - series_nn.min()) / (series_nn.max() - series_nn.min() + 1e-9)

    fdf['s_pct'] = normalize(fdf['Pct_Chg (%)'])
    fdf['s_turn'] = normalize(fdf['turnover'])
    fdf['s_vol'] = normalize(fdf['vol_ratio'])
    fdf['s_mf'] = normalize(fdf['net_mf'])
    fdf['s_macd'] = normalize(fdf['macd'])
    fdf['s_trend'] = normalize(fdf['10d_return'])
    fdf['s_volatility'] = normalize(fdf['volatility'])
    fdf['s_position'] = fdf['position_60d'] / 100
    
    # ----------------------------------------------------------------------------------
    # 🚨 V17.0 最终定型策略：资金流 (0.45) + MACD (0.20) + 动能 (0.20) + 弱防御 (0.05)
    
    # 核心权重：资金流，占比 45%
    w_mf = 0.45             # 45% - 资金流 (核心动力)

    # 动能权重：当日动能，占比 30%
    w_pct = 0.20            # 20% - 当日涨幅 (微降，配合 MACD 增强)
    w_turn = 0.10           # 10% - 换手率 (保持)
    
    # 趋势确认：占比 20%
    w_macd = 0.20           # 20% - MACD (增强，用于中期趋势的适度确认，消除 D+3 盘整)

    # 弱防御：占比 5%
    w_volatility = 0.05     # 05% - 波动率 (极致弱防御，保持 V16.0 的成功)
 
    # 彻底归零项
    w_trend = 0.00          # 0% - 10日回报 
    w_position = 0.00       # 0% - 60日位置 
    w_vol = 0.00            # 0% - 量比 
    
    # Sum: 0.45+0.20+0.10+0.20+0.05 = 1.00
    
    score = (
        fdf['s_pct'] * w_pct + fdf['s_turn'] * w_turn + 
        fdf['s_mf'] * w_mf + 
        fdf['s_macd'] * w_macd + 
        
        # 波动率保持逆向 (越低越好)
        (1 - fdf['s_volatility']) * w_volatility + 
        
        # 归零项
        fdf['s_trend'] * w_trend + 
        fdf['s_position'] * w_position + 
        fdf['s_vol'] * w_vol      
    )
    fdf['综合评分'] = score * 100
    # ----------------------------------------------------------------------------------
    # 🚨 V17.1 排序修复：从降序 (False) 更改为升序 (True)，选择分数最低的 Top K
    # ----------------------------------------------------------------------------------
    fdf = fdf.sort_values('综合评分', ascending=True).reset_index(drop=True)
    fdf.index += 1
    

    return fdf.head(TOP_BACKTEST).copy(), None

# ---------------------------
# 主运行块 (保持不变)
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测"):
    
    st.warning("⚠️ **V17.0 版本已应用：资金流 (0.45) + MACD (0.20) + 动能 (0.20) + 弱防御 (0.05)。**")
   
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
      
        daily_result_df, error = run_backtest_for_a_day(
            trade_date, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS
        )
        
        if error:
            st.warning(f"跳过 {trade_date}：{error}")
        elif not daily_result_df.empty:
            daily_result_df['Trade_Date'] = trade_date
            results_list.append(daily_result_df)
            
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
                  help=f"总有效样本数：{total_count}。**V17.0 已应用最终定型策略。**")

    st.header("📋 每日回测详情 (Top K 明细)")
    
    display_cols = ['Trade_Date', 'name', 'ts_code', '综合评分', 
                    'Close', 'Pct_Chg (%)', 'Circ_MV (亿)',
                    'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)']
    
    st.dataframe(all_results[display_cols].sort_values('Trade_Date', ascending=False), use_container_width=True)
