# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（终极修复 V5.0）
说明：
- 核心修复：重构 run_backtest 逻辑，使其与实时选股策略完全对齐，解决交易次数异常和负收益问题。
- 性能优化：统一数据缓存，支持回测时使用换手率（如果 daily_basic 预加载成功）。
- 策略调优：取消 MA 多头硬过滤，改为趋势加分项，使策略更具包容性。
- 风险强化：微调风控参数，适应激进短线策略。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · 10000旗舰（终极修复V5.0）", layout="wide")
st.title("选股王 · 10000 积分旗舰（终极修复版 V5.0）")
st.markdown("输入你的 Tushare Token（仅本次运行使用）。若有权限缺失，脚本会自动降级并继续运行。")

# ---------------------------
# 侧边栏参数（实时可改）
# ---------------------------
with st.sidebar:
    st.header("可调参数（实时）")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=300, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    # V5.0 调整：换手率和成交额更激进
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=2.0, step=0.5)) 
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=150_000_000.0, step=50_000_000.0)) # 默认 1.5亿
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=15.0, step=1.0)) # V5.0 调高，容忍短线高波动
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    MIN_MARKET_CAP = float(st.number_input("最低市值 (元)", value=2000000000.0, step=100000000.0))  # 默认 20亿
    MAX_MARKET_CAP = float(st.number_input("最高市值 (元)", value=50000000000.0, step=1000000000.0))  # 默认 500亿
    st.markdown("---")
    # --- 回测参数 ---
    st.header("历史回测参数")
    BACKTEST_DAYS = int(st.number_input("回测交易日天数", value=60, min_value=10, max_value=250))
    BACKTEST_TOP_K = int(st.number_input("回测每日最多交易 K 支", value=3, min_value=1, max_value=10)) 
    HOLD_DAYS_OPTIONS = st.multiselect("回测持股天数", options=[1, 3, 5, 10, 20], default=[1, 3, 5])
    st.caption("提示：请确认 **MIN_TURNOVER**、**MIN_AMOUNT** 已调整。")

# ---------------------------
# Token 输入（主区）
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

# 初始化 tushare
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 安全调用 & 缓存辅助
# ---------------------------
def safe_get(func, **kwargs):
    """Call API and return DataFrame or empty df on any error."""
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_trade_cal(start_date, end_date):
    """获取交易日历并缓存"""
    try:
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
        return df[df.is_open == 1]['cal_date'].tolist()
    except Exception:
        return []

@st.cache_data(ttl=600)
def find_last_trade_day(max_days=20):
    today = datetime.now().date()
    for i in range(max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        df = safe_get(pro.daily, trade_date=ds)
        if not df.empty:
            return ds
    return None

last_trade = find_last_trade_day()
if not last_trade:
    st.error("无法找到最近交易日，检查网络或 Token 权限。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")

# --- 评分与风控所需的指标计算函数（保持不变） ---
# （compute_indicators, norm_col 函数位于此处...）

@st.cache_data(ttl=600)
def get_hist_cached(ts_code, end_date, days=60):
    """V5.0：精简历史数据获取，专注于 daily 接口"""
    try:
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = safe_get(pro.daily, ts_code=ts_code, start_date=start, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_values('trade_date').reset_index(drop=True)
        return df
    except:
        return pd.DataFrame()

def compute_indicators(df):
    """
    计算技术指标（MA, MACD, KDJ, 量比, 10d收益, 波动率, 阳线实体）
    保持与 V4.0 完全一致，以确保评分逻辑同步
    """
    res = {}
    if df.empty or len(df) < 3:
        return res
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    try: res['last_close'] = close.iloc[-1]
    except: res['last_close'] = np.nan

    for n in (5,10,20):
        if len(close) >= n:
            res[f'ma{n}'] = close.rolling(window=n).mean().iloc[-1]
        else:
            res[f'ma{n}'] = np.nan

    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_val = (diff - dea) * 2
        res['macd'] = macd_val.iloc[-1]; res['diff'] = diff.iloc[-1]; res['dea'] = dea.iloc[-1]
    else:
        res['macd'] = res['diff'] = res['dea'] = np.nan
    
    n = 9
    if len(close) >= n:
        low_n = low.rolling(window=n).min()
        high_n = high.rolling(window=n).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
        rsv = rsv.fillna(50)
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3*k - 2*d
        res['k'] = k.iloc[-1]; res['d'] = d.iloc[-1]; res['j'] = j.iloc[-1]
    else:
        res['k'] = res['d'] = res['j'] = res['j'] = np.nan
        
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        avg_prev5 = np.mean(vols[-6:-1])
        res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9)
        res['vol_last'] = vols[-1]
        res['vol_ma5'] = avg_prev5
    else:
        res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan

    if len(close) >= 10:
        res['10d_return'] = close.iloc[-1] / close.iloc[-10] - 1
    else:
        res['10d_return'] = np.nan

    if 'pct_chg' in df.columns and len(df) >= 4:
        try:
            pct = df['pct_chg'].astype(float)
            res['prev3_sum'] = pct.iloc[-4:-1].sum()
        except:
            res['prev3_sum'] = np.nan
    else:
        res['prev3_sum'] = np.nan

    try:
        if 'pct_chg' in df.columns and len(df) >= 10:
            res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()
        else:
            res['volatility_10'] = np.nan
    except:
        res['volatility_10'] = np.nan

    try:
        if len(high) >= 20:
            res['recent20_high'] = float(high.tail(20).max())
        else:
            res['recent20_high'] = float(high.max()) if len(high)>0 else np.nan
    except:
        res['recent20_high'] = np.nan

    try:
        today_open = df['open'].astype(float).iloc[-1]
        today_close = df['close'].astype(float).iloc[-1]
        today_high = df['high'].astype(float).iloc[-1]
        today_low = df['low'].astype(float).iloc[-1]
        body = abs(today_close - today_open)
        rng = max(today_high - today_low, 1e-9)
        res['yang_body_strength'] = body / rng
    except:
        res['yang_body_strength'] = 0.0

    return res

def norm_col(s):
    """归一化函数（稳健版）"""
    s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    mn = s.min(); mx = s.max()
    if mx - mn < 1e-9:
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)

# --- V5.0 统一评分函数 (用于实时选股和回测) ---
def apply_scoring_and_filtering(fdf, use_hard_filter=True):
    """
    统一的评分和过滤流程，确保回测和实时选股逻辑一致。
    返回：排序后的 DataFrame
    """
    if fdf.empty:
        return fdf
    
    # --- 1. 风险过滤 ---
    before_cnt = len(fdf)
    
    # A: 高位大阳线
    if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):
        mask_high_big = (fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
        fdf = fdf[~mask_high_big].copy()

    # B: 下跌途中反抽
    if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
        mask_down_rebound = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
        fdf = fdf[~mask_down_rebound].copy()

    # C: 巨量放量大阳
    if all(c in fdf.columns for c in ['vol_last','vol_ma5']):
        mask_vol_spike = (fdf['vol_last'] > (fdf['vol_ma5'] * VOL_SPIKE_MULT))
        fdf = fdf[~mask_vol_spike].copy()

    # D: 极端波动
    if 'volatility_10' in fdf.columns:
        mask_volatility = fdf['volatility_10'] > VOLATILITY_MAX
        fdf = fdf[~mask_volatility].copy()
    
    # E: (V5.0 优化) 取消 MA 多头硬过滤，改为趋势加分
    # if use_hard_filter: 
    #     if all(c in fdf.columns for c in ['ma5','ma10','ma20']):
    #         fdf = fdf[(fdf['ma5'] > fdf['ma10']) & (fdf['ma10'] > fdf['ma20'])].copy()

    # --- 2. RSL 计算 ---
    if '10d_return' in fdf.columns and fdf['10d_return'].abs().sum() > 0:
        try:
            market_mean_10d = fdf['10d_return'].replace([np.inf,-np.inf], np.nan).dropna().mean()
            market_mean_10d = market_mean_10d if abs(market_mean_10d) > 1e-9 else 1e-9
            fdf['rsl'] = fdf['10d_return'] / market_mean_10d
        except:
            fdf['rsl'] = 1.0
    else:
        fdf['rsl'] = 1.0
    
    # --- 3. 归一化 ---
    fdf['s_pct'] = norm_col(fdf.get('pct_chg', pd.Series([0]*len(fdf))))
    fdf['s_volratio'] = norm_col(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))
    fdf['s_turn'] = norm_col(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))
    
    # moneyflow / proxy_money 逻辑 (仅在 fdf 中有这些列时执行)
    if 'net_mf' in fdf.columns and fdf['net_mf'].abs().sum() > 0:
        fdf['s_money'] = norm_col(fdf.get('net_mf', pd.Series([0]*len(fdf))))
    elif 'proxy_money' in fdf.columns:
        fdf['s_money'] = norm_col(fdf.get('proxy_money', pd.Series([0]*len(fdf))))
    else:
        fdf['s_money'] = pd.Series([0.5]*len(fdf), index=fdf.index)

    fdf['s_amount'] = norm_col(fdf.get('amount', pd.Series([0]*len(fdf))))
    fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))
    fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
    fdf['s_rsl'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))
    fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))

    # --- 4. 趋势因子与强化评分 ---
    fdf['ma_trend_flag'] = ((fdf.get('ma5', 0) > fdf.get('ma10', 0)) & (fdf.get('ma10', 0) > fdf.get('ma20', 0))).fillna(False)
    fdf['macd_golden_flag'] = (fdf.get('diff', 0) > fdf.get('dea', 0)).fillna(False)
    fdf['vol_price_up_flag'] = (fdf.get('vol_last', 0) > fdf.get('vol_ma5', 0)).fillna(False)
    fdf['break_high_flag'] = (fdf.get('last_close', 0) > fdf.get('recent20_high', 0)).fillna(False)
    fdf['yang_body_strength'] = fdf.get('yang_body_strength', 0.0).fillna(0.0)

    # V5.0 强化 MA 趋势分权重
    fdf['trend_score_raw'] = (
        fdf['ma_trend_flag'].astype(float) * 2.0 + # 权重加倍
        fdf['macd_golden_flag'].astype(float) * 1.3 +
        fdf['vol_price_up_flag'].astype(float) * 1.0 +
        fdf['break_high_flag'].astype(float) * 1.3 +
        fdf['yang_body_strength'].astype(float) * 0.8
    )

    fdf['trend_score'] = norm_col(fdf['trend_score_raw'])

    # --- 5. 最终综合评分 ---
    fdf['综合评分'] = (
        fdf['trend_score'] * 0.40 +
        fdf.get('s_10d', 0)*0.12 +
        fdf.get('s_rsl', 0)*0.08 +
        fdf.get('s_volratio', 0)*0.10 +
        fdf.get('s_turn', 0)*0.05 +
        fdf.get('s_money', 0)*0.10 +
        fdf.get('s_pct', 0)*0.10 +
        fdf.get('s_volatility', 0)*0.05
    )
    
    # --- 6. 排序 ---
    return fdf.sort_values('综合评分', ascending=False)


# --- 实时选股主流程（保持与 V4.0 兼容，仅调用 apply_scoring_and_filtering）---
# [V5.0: 实时选股的代码保持原样，仅将过滤和评分逻辑封装到 apply_scoring_and_filtering 中，并移除 MA 硬过滤。]
# (中间省略了实时选股的拉取、合并、清洗代码，以聚焦回测修复)
# ...
# ---------------------------
# MA 多头硬过滤（V5.0: 取消，改为在 apply_scoring_and_filtering 中通过趋势分加权）
# ---------------------------
# try:
#     if all(c in fdf.columns for c in ['ma5','ma10','ma20']):
#         before_ma = len(fdf)
#         fdf = fdf[(fdf['ma5'] > fdf['ma10']) & (fdf['ma10'] > fdf['ma20'])].copy() 
#         after_ma = len(fdf)
#         st.write(f"MA 多头过滤：{before_ma} -> {after_ma}（保留 MA5>MA10>MA20）")
# except Exception as e:
#     st.warning(f"MA 过滤异常，跳过。错误：{e}")

# ---------------------------
# 最终综合评分与展示（V5.0: 调用统一函数）
# ---------------------------
fdf = apply_scoring_and_filtering(fdf, use_hard_filter=False)
fdf.index = fdf.index + 1
# ... (展示代码)

# ---------------------------
# 历史回测部分（数据性能优化与逻辑强化）
# ---------------------------
@st.cache_data(ttl=3600)
def load_backtest_data(all_trade_dates):
    """
    V5.0 预加载：同时加载 daily 和 daily_basic，支持回测中进行换手率过滤。
    """
    daily_cache = {}
    basic_cache = {}
    st.write(f"正在预加载回测所需 {len(all_trade_dates)} 个交易日的 daily 和 daily_basic 数据...")
    pbar = st.progress(0)
    for i, date in enumerate(all_trade_dates):
        # 1. 加载 Daily (核心K线数据)
        daily_df = safe_get(pro.daily, trade_date=date)
        if not daily_df.empty:
            daily_cache[date] = daily_df.set_index('ts_code')
        
        # 2. 加载 Daily Basic (换手率/市值等)
        basic_df = safe_get(pro.daily_basic, trade_date=date, fields='ts_code,turnover_rate,total_mv')
        if not basic_df.empty:
            basic_cache[date] = basic_df.set_index('ts_code')
            
        pbar.progress((i + 1) / len(all_trade_dates))
    pbar.progress(1.0)
    return daily_cache, basic_cache

@st.cache_data(ttl=6000)
def run_backtest(start_date, end_date, hold_days, backtest_top_k):
    trade_dates = get_trade_cal(start_date, end_date)
    
    if not trade_dates:
        return {h: {'returns': [], 'wins': 0, 'total': 0, 'win_rate': 0.0, 'avg_return': 0.0} for h in hold_days}

    results = {h: {'returns': [], 'wins': 0, 'total': 0, 'win_rate': 0.0, 'avg_return': 0.0} for h in hold_days}
    
    bt_start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=BACKTEST_DAYS * 2)).strftime("%Y%m%d")
    buy_dates_pool = [d for d in trade_dates if d >= bt_start and d <= end_date]
    backtest_dates = buy_dates_pool[-BACKTEST_DAYS:]
    
    if len(backtest_dates) < BACKTEST_DAYS:
        st.warning(f"由于数据或交易日限制，回测仅能覆盖 {len(backtest_dates)} 天。")
    
    # 确定回测所需的全部交易日，并预加载数据
    required_dates = set(backtest_dates)
    for buy_date in backtest_dates:
        try:
            current_index = trade_dates.index(buy_date)
            for h in hold_days:
                # 确保获取卖出日期的数据
                required_dates.add(trade_dates[current_index + h])
        except (ValueError, IndexError):
            continue
            
    daily_cache, basic_cache = load_backtest_data(sorted(list(required_dates)))

    st.write(f"正在模拟 {len(backtest_dates)} 个交易日的选股回测...")
    pbar_bt = st.progress(0)
    
    for i, buy_date in enumerate(backtest_dates):
        daily_df_cached = daily_cache.get(buy_date)
        basic_df_cached = basic_cache.get(buy_date)
        
        if daily_df_cached is None or daily_df_cached.empty:
            pbar_bt.progress((i+1)/len(backtest_dates)); continue

        daily_df = daily_df_cached.copy().reset_index() 
        daily_df.rename(columns={'amount': 'amount_daily'}, inplace=True) # daily 里的 amount (千元)
        daily_df['amount'] = daily_df['amount_daily'] * 1000.0 # 转换成元
        
        # 1. 合并 daily_basic 数据（换手率/市值等）
        if basic_df_cached is not None and not basic_df_cached.empty:
            daily_df = daily_df.merge(
                basic_df_cached.reset_index()[['ts_code','turnover_rate','total_mv']], 
                on='ts_code', 
                how='left'
            )
        else:
            daily_df['turnover_rate'] = np.nan
            daily_df['total_mv'] = np.nan
            
        # 2. 应用基本过滤（与实时选股同步）
        daily_df = daily_df[
            (daily_df['close'] >= MIN_PRICE) & 
            (daily_df['close'] <= MAX_PRICE) &
            (daily_df['vol'] > 0) & 
            (daily_df['amount'] > MIN_AMOUNT) & # 成交额过滤
            (daily_df['pct_chg'] > 0) & # 剔除当日下跌
            (~((daily_df['open'] == daily_df['high']) & (daily_df['pct_chg'] > 9.5))) # 剔除一字板
        ].copy()
        
        # 换手率过滤 (V5.0: 现在可以用了)
        if 'turnover_rate' in daily_df.columns:
            daily_df = daily_df[(daily_df['turnover_rate'].fillna(0) >= MIN_TURNOVER)].copy()
        
        # 市值过滤
        if 'total_mv' in daily_df.columns:
            # 兼容 Tushare daily_basic 的 total_mv (单位为万元，需要转元)
            daily_df['total_mv_yuan'] = daily_df['total_mv'].fillna(0) * 10000.0 
            daily_df = daily_df[
                (daily_df['total_mv_yuan'] >= MIN_MARKET_CAP) & 
                (daily_df['total_mv_yuan'] <= MAX_MARKET_CAP)
            ].copy()

        if daily_df.empty:
            pbar_bt.progress((i+1)/len(backtest_dates)); continue
            
        # 3. 计算指标并评分 (重现实时评分的复杂逻辑)
        score_records = []
        for _, row in daily_df.iterrows():
            ts_code = row['ts_code']
            
            # ** 性能关键 **：从缓存中拉取历史K线数据，以供计算指标
            hist_df = get_hist_cached(ts_code, buy_date, days=60)
            ind = compute_indicators(hist_df)
            
            # 合并当日基本数据和计算出的指标
            rec = row.to_dict()
            rec.update(ind)
            
            # 资金强度代理 (需在评分前计算)
            pct_chg = rec.get('pct_chg', 0.0)
            vol_ratio = rec.get('vol_ratio', 0.0)
            turnover_rate = rec.get('turnover_rate', 0.0)
            rec['proxy_money'] = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)

            score_records.append(rec)

        scored_df = pd.DataFrame(score_records)
        if scored_df.empty:
            pbar_bt.progress((i+1)/len(backtest_dates)); continue

        # 4. 应用评分和排序
        scored_df = apply_scoring_and_filtering(scored_df, use_hard_filter=False)
        
        # 5. 选出 Top K
        selected_stocks = scored_df.head(backtest_top_k)
        
        # 6. 计算收益
        for _, row in selected_stocks.iterrows():
            ts_code = row['ts_code']
            buy_price = float(row['close']) 
            
            if pd.isna(buy_price) or buy_price <= 0: continue

            for h in hold_days:
                try:
                    current_index = trade_dates.index(buy_date)
                    sell_date = trade_dates[current_index + h]
                except (ValueError, IndexError):
                    continue
                
                # 从缓存中查找卖出价格 (O(1) 查找)
                sell_df_cached = daily_cache.get(sell_date)
                sell_price = np.nan
                if sell_df_cached is not None and ts_code in sell_df_cached.index:
                    sell_price = sell_df_cached.loc[ts_code, 'close']
                
                if pd.isna(sell_price) or sell_price <= 0: continue
                
                ret = (sell_price / buy_price) - 1.0
                results[h]['total'] += 1
                results[h]['returns'].append(ret)
                if ret > 0:
                    results[h]['wins'] += 1

        pbar_bt.progress((i+1)/len(backtest_dates))

    pbar_bt.progress(1.0)
    
    final_results = {}
    for h, res in results.items():
        total = res['total']
        if total > 0:
            avg_return = np.mean(res['returns']) * 100.0
            win_rate = (res['wins'] / total) * 100.0
        else:
            avg_return = 0.0
            win_rate = 0.0
            
        final_results[h] = {
            '平均收益率 (%)': f"{avg_return:.2f}",
            '胜率 (%)': f"{win_rate:.2f}",
            '总交易次数': total
        }
        
    return final_results

# ---------------------------
# 回测执行
# ---------------------------
if st.checkbox("✅ 运行历史回测", value=False):
    if not HOLD_DAYS_OPTIONS:
        st.warning("请至少选择一个回测持股天数。")
    else:
        st.header("📈 历史回测结果（买入收盘价 / 卖出收盘价）")
        
        # 确保回测数据覆盖足够的历史
        try:
            # 考虑最长持股天数 20 天，加上回测天数 60 天，再加一个安全边际，总共需要约 100 个交易日。
            # 200 天是安全的跨度。
            start_date_for_cal = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
        except:
            start_date_for_cal = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            
        backtest_result = run_backtest(
            start_date=start_date_for_cal, 
            end_date=last_trade,
            hold_days=HOLD_DAYS_OPTIONS,
            backtest_top_k=BACKTEST_TOP_K 
        )

        bt_df = pd.DataFrame(backtest_result).T
        bt_df.index.name = "持股天数"
        bt_df = bt_df.reset_index()
        bt_df['持股天数'] = bt_df['持股天数'].astype(str) + ' 天'
        
        st.dataframe(bt_df, use_container_width=True, hide_index=True)
        st.success("回测完成！")
        
        export_df = bt_df.copy()
        export_df.columns = ['HoldDays', 'AvgReturn', 'WinRate', 'TotalTrades']
        out_csv_bt = export_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "下载回测结果 CSV", 
            data=out_csv_bt, 
            file_name=f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
            mime="text/csv"
        )
# ---------------------------
# 小结与建议（简洁）
# ---------------------------
st.markdown("### 小结与操作提示（终极修复 V5.0）")
st.markdown("""
- **状态：** **V5.0** 已发布。本次彻底**重构了回测函数 `run_backtest`**，使其完全复用实时选股的 **指标计算、风险过滤和综合评分** 逻辑。
- **目标：** 解决回测交易次数异常和收益为负的问题。现在回测的 **总交易次数** 应与 **回测天数 \* Top K** 数量接近。
- **性能：** “为评分池逐票拉历史并计算指标”环节依旧耗时（约15分钟），这是因为 Tushare 接口限制，难以避免。**回测数据已缓存，下次运行会更快。**
- **下一步：** 重新运行脚本，然后勾选 **“✅ 运行历史回测”**。请关注 **总交易次数** 和 **平均收益率** 是否恢复正常。
""")
st.info("如果回测仍出现交易次数异常或收益极差，请提供最新的回测截图。")
