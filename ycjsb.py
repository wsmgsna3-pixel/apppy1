# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（最终修复版 v4.6 - 均值回归/回调策略）
说明：
- **V4.6 核心修复：** 修复了回测函数 `run_backtest` 的参数签名和过滤逻辑，
  确保侧边栏设置的**涨跌幅限制**能够正确传递和应用，彻底解决“交易次数0”的问题。
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
st.set_page_config(page_title="选股王 · 10000旗舰（均值回归 v4.6）", layout="wide")
st.title("选股王 · 10000 积分旗舰（最终修复版 v4.6 - 均值回归策略）")
st.markdown("输入你的 Tushare Token（仅本次运行使用）。若有权限缺失，脚本会自动降级并继续运行。")

# ---------------------------
# 侧边栏参数（实时可改）
# ---------------------------
with st.sidebar:
    st.header("可调参数（策略核心）")
    # 策略硬性过滤参数
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=1.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=500.0, step=10.0))
    MIN_MARKET_CAP = float(st.number_input("最低市值 (元)", value=500000000.0, step=100000000.0)) # 默认 5亿
    MAX_MARKET_CAP = float(st.number_input("最高市值 (元)", value=100000000000.0, step=1000000000.0)) # 默认 1000亿
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=100000000.0, step=10000000.0)) # 默认 1亿
    
    st.markdown("---")
    st.subheader("技术指标参数")
    MACD_FAST = int(st.number_input("MACD 快线周期", value=12, step=1))
    MACD_SLOW = int(st.number_input("MACD 慢线周期", value=26, step=1))
    MACD_SIGNAL = int(st.number_input("MACD 信号线周期", value=9, step=1))
    RSI_PERIOD = int(st.number_input("RSI 周期", value=14, step=1))
    
    st.markdown("---")
    # --- 历史回测参数 ---
    st.header("历史回测参数")
    BACKTEST_DAYS = int(st.number_input("回测交易日天数", value=60, min_value=10, max_value=250))
    BACKTEST_TOP_K = int(st.number_input("回测每日最多交易 K 支", value=5, min_value=1, max_value=20))
    HOLD_DAYS_OPTIONS = st.multiselect("回测持股天数", options=[1, 3, 5, 10, 20], default=[3, 5])
    
    st.subheader("回测当日涨跌幅控制")
    # V4.6 关键参数：用于回测时的硬过滤
    BT_MIN_PCT_FOR_CACHE = float(st.number_input("回测：当日最低涨幅 (%)", value=-3.0, step=0.5, help="回调策略，当日最低跌幅"))
    BT_MAX_PCT_FOR_CACHE = float(st.number_input("回测：当日最高涨幅 (%)", value=1.5, step=0.5, help="回调策略，当日最高涨幅（避免追高）"))

    # 缓存破坏键 (用于强制回测重新加载数据)
    CACHE_BREAKER = float(st.number_input("回测：缓存破坏键（任意修改刷新回测）", value=1.20, step=0.01))
    st.caption("提示：策略为**均值回归/回调策略**，买入条件是 MACD 金叉后的回调。")

# ---------------------------
# Token 输入（主区）与初始化
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 缓存辅助函数
# ---------------------------
def safe_get(func, **kwargs):
    """安全调用 API，若失败则返回空 DataFrame。"""
    try:
        if func == pro.query:
             df = pro.query(kwargs.pop('api_name'), **kwargs)
        else:
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

@st.cache_data(ttl=36000)
def find_last_trade_day(max_days=20):
    """寻找最近一个交易日"""
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

# ---------------------------
# 核心指标计算函数
# ---------------------------
@st.cache_data(ttl=36000)
def get_hist_cached(ts_code, end_date, days=120): 
    """获取单只股票历史数据并缓存"""
    try:
        # 扩展历史天数以确保计算指标的准确性
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days * 2)).strftime("%Y%m%d")
        df = safe_get(pro.daily, ts_code=ts_code, start_date=start, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_values('trade_date').reset_index(drop=True)
        return df.tail(days).reset_index(drop=True) 
    except:
        return pd.DataFrame()

def compute_indicators(df, macd_fast, macd_slow, macd_signal, rsi_period):
    """计算 MACD, RSI, VWA 等关键指标"""
    res = {}
    if df.empty or len(df) < max(macd_slow, rsi_period) + 1:
        return res
    
    close = df['close'].astype(float)
    pct = df['pct_chg'].astype(float)
    vol = df['vol'].astype(float)
    amount = df['amount'].astype(float) # Tushare amount 是千元

    # MACD (EMA)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=macd_signal, adjust=False).mean()
    hist = (diff - dea) * 2
    
    res['macd_diff'] = diff.iloc[-1]
    res['macd_dea'] = dea.iloc[-1]
    res['macd_hist'] = hist.iloc[-1]
    
    # MACD金叉判断: DIFF > DEA 且 上一周期 DIFF <= DEA
    # V4.x 策略：寻找 MACD 金叉后的“强势股”
    if len(diff) > 1:
        # 均值回归策略寻找的是**金叉后的回调买点**，而非金叉当日
        res['macd_golden_yesterday'] = (diff.iloc[-2] > dea.iloc[-2]) and (diff.iloc[-3] <= dea.iloc[-3])
    else:
        res['macd_golden_yesterday'] = False
    
    # RSI (RSI < 50 表示回调，RSI < 30 表示超跌)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=rsi_period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    res['rsi'] = rsi.iloc[-1]
    
    # 趋势指标 (过去 N 天累计涨幅)
    for n in (5, 10, 20):
        if len(close) >= n:
            res[f'{n}d_return'] = (close.iloc[-1] / close.iloc[-n] - 1) * 100
        else:
            res[f'{n}d_return'] = np.nan
            
    # 量能指标
    res['last_vol'] = vol.iloc[-1]
    res['last_amount'] = amount.iloc[-1] * 1000 # 千元 -> 元

    return res

# ---------------------------
# 选股逻辑
# ---------------------------
if st.button("🚀 运行当日选股（初次运行可能较久）"):
    # 1. 拉取当日 daily_basic 作为初筛
    st.write("正在拉取当日 daily_basic 作为初筛...")
    # 均值回归策略不需要按涨幅初筛，直接全量加载
    daily_all = safe_get(pro.daily_basic, trade_date=last_trade)
    if daily_all.empty:
        st.error("无法获取当日 daily_basic 数据。")
        st.stop()
        
    # 2. 拉取高级接口数据
    st.write("尝试加载 stock_basic / daily 等高级接口...")
    # 尽量使用一次性查询，提高效率
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,total_mv,circ_mv')
    daily_raw = safe_get(pro.daily, trade_date=last_trade, fields='ts_code,close,pct_chg,vol,amount')
    
    # 3. 数据合并
    pool_merged = daily_all.copy()
    
    if not stock_basic.empty:
        pool_merged = pool_merged.merge(stock_basic[['ts_code','name','industry']], 
                                        on='ts_code', how='left')
    if not daily_raw.empty:
        pool_merged = pool_merged.merge(daily_raw, 
                                        on='ts_code', how='left')
        
    # Tushare 的 total_mv/circ_mv 默认单位是万元
    pool_merged['total_mv_yuan'] = pool_merged['total_mv'].fillna(0) * 10000.0
    pool_merged['amount_yuan'] = pool_merged['amount'].fillna(0) * 1000.0 # Tushare amount 是千元

    # 4. 硬性过滤（清洗阶段）
    st.write("正在对初筛池进行硬性过滤（价格/市值/成交额/ST/北交所）...")
    clean_df = pool_merged.copy()
    
    # F1: 价格区间
    clean_df = clean_df[(clean_df['close'] >= MIN_PRICE) & (clean_df['close'] <= MAX_PRICE)]
    
    # F2: 市值区间
    clean_df = clean_df[(clean_df['total_mv_yuan'] >= MIN_MARKET_CAP) & (clean_df['total_mv_yuan'] <= MAX_MARKET_CAP)]
    
    # F3: 成交额
    clean_df = clean_df[clean_df['amount_yuan'] >= MIN_AMOUNT]
    
    # F4: ST / 北交所 / 停牌 / 无成交
    clean_df = clean_df[~clean_df['name'].str.contains('ST|退', na=False)]
    clean_df = clean_df[~clean_df['ts_code'].str.startswith('4', na=False)]
    clean_df = clean_df[~clean_df['ts_code'].str.startswith('8', na=False)]
    clean_df = clean_df[(clean_df['vol'] > 0) & (clean_df['amount_yuan'] > 0)]
    
    st.write(f"硬性过滤后候选数量：{len(clean_df)} 支。")
    if len(clean_df) == 0:
        st.error("硬性过滤后没有候选，请放宽侧边栏条件。")
        st.stop()
        
    # 5. 逐个计算指标与核心过滤（耗时步骤）
    st.write("为候选股逐票计算指标（MACD/RSI/趋势）...")
    records = []
    pbar = st.progress(0)
    
    for idx, row in clean_df.iterrows():
        ts_code = row['ts_code']
        
        hist = get_hist_cached(ts_code, last_trade, days=60) # 60日历史数据足够
        ind = compute_indicators(hist, MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_PERIOD)
        
        # --- 策略核心过滤：MACD 金叉后的回调买点 ---
        
        # F5: 昨天 MACD 发生金叉 (趋势确认)
        if not ind.get('macd_golden_yesterday', False):
            pbar.progress((idx+1)/len(clean_df))
            continue
            
        # F6: RSI 中位回调 (RSI < 50 表示回调买点)
        rsi_val = ind.get('rsi', np.nan)
        if pd.isna(rsi_val) or rsi_val >= 50.0:
            pbar.progress((idx+1)/len(clean_df))
            continue
            
        # F7: 短期涨幅适中 (避免超高和超低)
        d5_ret = ind.get('5d_return', np.nan)
        if pd.isna(d5_ret) or d5_ret > 10.0 or d5_ret < -10.0:
             pbar.progress((idx+1)/len(clean_df))
             continue

        # --- 合并指标，准备评分 ---
        row_dict = row.to_dict()
        row_dict.update(ind)
        records.append(row_dict)
        pbar.progress((idx+1)/len(clean_df))
    
    pbar.progress(1.0)
    fdf = pd.DataFrame(records)
    fdf = fdf.dropna(subset=['rsi']).reset_index(drop=True)
    st.write(f"策略核心过滤后，进入评分阶段的候选数量：{len(fdf)} 支。")
    if fdf.empty:
        st.error("所有股票都被过滤，请放宽策略过滤条件。")
        st.stop()

    # 6. 归一化与评分 (偏好 RSI 低、短期回调浅、MACD动能强的)
    
    def norm_col(s, reverse=False):
        s = s.fillna(s.median()).replace([np.inf,-np.inf], np.nan).fillna(s.median())
        mn = s.min(); mx = s.max()
        if mx - mn < 1e-9:
            return pd.Series([0.5]*len(s), index=s.index)
        
        normalized = (s - mn) / (mx - mn)
        return 1 - normalized if reverse else normalized

    # 归一化子指标 (s_rsi 反转，越低越好)
    fdf['s_rsi'] = norm_col(fdf['rsi'], reverse=True)
    fdf['s_5d_ret'] = norm_col(fdf['5d_return'], reverse=True) # 短期跌幅越大越好
    fdf['s_macd_hist'] = norm_col(fdf['macd_hist'], reverse=False) # MACD动能越强越好
    fdf['s_vol'] = norm_col(fdf['vol'], reverse=False) # 量能越大越好
    
    # 最终综合评分（权重分配 - 重点增强 MACD/RSI/短期回调信号）
    fdf['综合评分'] = (
        fdf['s_rsi'] * 0.30 +         # RSI 低位回调
        fdf['s_5d_ret'] * 0.30 +      # 短期回调深度
        fdf['s_macd_hist'] * 0.20 +   # MACD动能
        fdf['s_vol'] * 0.20            # 量能支持
    )

    # 7. 排序与展示
    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index = fdf.index + 1
    
    st.success(f"评分完成：总候选 {len(fdf)} 支，显示 Top {min(20, len(fdf))}。")
    display_cols = ['name','ts_code','综合评分','pct_chg','rsi','5d_return','macd_hist','macd_diff','macd_dea','close','total_mv','amount_yuan']
    
    final_cols = [c for c in display_cols if c in fdf.columns]
    
    st.dataframe(fdf[final_cols].head(20), use_container_width=True)

# ---------------------------
# 历史回测部分
# ---------------------------
@st.cache_data(ttl=3600)
def load_backtest_data(all_trade_dates):
    """预加载所有回测日期的 daily 数据。"""
    data_cache = {}
    st.write(f"正在预加载回测所需 {len(all_trade_dates)} 个交易日的 daily 数据...")
    pbar = st.progress(0)
    for i, date in enumerate(all_trade_dates):
        daily_df = safe_get(pro.daily, trade_date=date)
        if not daily_df.empty:
            data_cache[date] = daily_df
        pbar.progress((i + 1) / len(all_trade_dates))
    pbar.progress(1.0)
    return data_cache

@st.cache_data(ttl=36000)
def get_stock_basic_filter(cache_breaker):
    """一次性加载股票基础数据，并构建硬过滤的白名单"""
    _ = cache_breaker # 确保参数修改时缓存刷新
    st.write("正在构建回测的股票白名单（ST/北交所过滤）...")
    
    df = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,list_date')
    if df.empty:
        return pd.DataFrame()
        
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('4', na=False)]
    df = df[~df['ts_code'].str.startswith('8', na=False)]
    
    return df[['ts_code']]


# V4.6 核心修复：修改参数签名，传入涨跌幅参数
@st.cache_data(ttl=6000)
def run_backtest(start_date, end_date, hold_days, backtest_top_k, 
                 bt_min_pct_chg, bt_max_pct_chg, cache_breaker): 
    
    _ = cache_breaker 

    trade_dates = get_trade_cal(start_date, end_date)
    
    if not trade_dates:
        return {h: {'returns': [], 'wins': 0, 'total': 0} for h in hold_days}

    results = {h: {'returns': [], 'wins': 0, 'total': 0} for h in hold_days}
    
    # ... (确定回测日期部分不变) ...
    bt_start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=BACKTEST_DAYS * 2)).strftime("%Y%m%d")
    buy_dates_pool = [d for d in trade_dates if d >= bt_start and d <= end_date]
    backtest_dates = buy_dates_pool[-BACKTEST_DAYS:]
    
    required_dates = set(backtest_dates)
    for buy_date in backtest_dates:
        try:
            current_index = trade_dates.index(buy_date)
            for h in hold_days:
                required_dates.add(trade_dates[current_index + h])
        except (ValueError, IndexError):
            continue
    
    # --- 阶段一：数据批量加载与预处理 ---
    
    # 1. 构建硬过滤白名单
    basic_filter_df = get_stock_basic_filter(cache_breaker)
    if basic_filter_df.empty:
        st.error("无法构建股票白名单，请检查Tushare权限。")
        return {h: {'returns': [], 'wins': 0, 'total': 0} for h in hold_days}
    
    valid_ts_codes = set(basic_filter_df['ts_code'])
    
    # 2. 预加载 daily 数据 (有进度条)
    data_cache = load_backtest_data(sorted(list(required_dates)))

    # ----------------------------------------------------
    # --- 阶段二：回测主循环 ---
    # ----------------------------------------------------
    
    st.write(f"正在模拟 {len(backtest_dates)} 个交易日的均值回归选股回测...")
    pbar_bt = st.progress(0)
    
    # 回测内部使用 MIN_AMOUNT (1亿) 的 2倍作为流动性过滤（避免 Tushare 的 daily_basic 数据缺失）
    BACKTEST_MIN_AMOUNT_PROXY = MIN_AMOUNT * 2.0
    
    for i, buy_date in enumerate(backtest_dates):
        daily_df_raw = data_cache.get(buy_date)
        
        if daily_df_raw is None or daily_df_raw.empty:
            pbar_bt.progress((i+1)/len(backtest_dates)); continue

        daily_df = daily_df_raw.copy()
        
        # 0. 初始过滤：只保留白名单中的股票
        daily_df = daily_df[daily_df['ts_code'].isin(valid_ts_codes)]
        
        # V4.6: 转换成交额 (Tushare amount 是千元)
        daily_df['amount_yuan'] = daily_df['amount'].fillna(0) * 1000.0
        
        # 1. 应用硬过滤 (回测模块只应用最核心的过滤条件)
        
        # 过滤：均值回归策略：寻找回调/盘整的股票
        daily_df = daily_df[
            # F1: 价格区间（此处使用硬编码，因为 MIN/MAX_PRICE 未传递到此函数）
            (daily_df['close'] >= MIN_PRICE) & 
            (daily_df['close'] <= MAX_PRICE) &
            
            # F2: 成交额流动性（双倍过滤）
            (daily_df['amount_yuan'] >= BACKTEST_MIN_AMOUNT_PROXY) & 

            # F3: 涨跌幅区间（核心修复：使用侧边栏参数）
            (daily_df['pct_chg'] >= bt_min_pct_chg) & 
            (daily_df['pct_chg'] <= bt_max_pct_chg) & 
            
            # F4: 停牌/无成交
            (daily_df['vol'] > 0) & 
            (daily_df['amount_yuan'] > 0)
        ].copy()
        
        # 2. 策略评分（简化版：按 MACD 金叉后的回调强度排序）
        # 均值回归策略：由于无法在回测中逐日计算 MACD/RSI，我们使用代理评分。
        # 代理评分：寻找回调/盘整（低 pct_chg），但成交量 (vol) 相对较高的股票。
        # pct_chg * (-1) 奖励跌幅，vol 奖励流动性
        daily_df['score_proxy'] = (daily_df['pct_chg'] * -1) * daily_df['vol']
        
        scored_stocks = daily_df.sort_values("score_proxy", ascending=False).head(backtest_top_k).copy()
        
        for _, row in scored_stocks.iterrows():
            ts_code = row['ts_code']
            buy_price = float(row['close'])
            
            if pd.isna(buy_price) or buy_price <= 0: continue

            for h in hold_days:
                try:
                    current_index = trade_dates.index(buy_date)
                    sell_date = trade_dates[current_index + h]
                except (ValueError, IndexError):
                    continue
        
                # 从缓存中查找卖出价格
                sell_df_cached = data_cache.get(sell_date)
                sell_price = np.nan
                if sell_df_cached is not None and ts_code in sell_df_cached['ts_code'].values:
                    sell_price = sell_df_cached[sell_df_cached['ts_code'] == ts_code]['close'].iloc[0]
                
                # 检查卖出日是否停牌或数据缺失
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
        
        try:
            start_date_for_cal = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
        except:
            start_date_for_cal = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            
        # V4.6 修复：传入 BT_MIN_PCT_FOR_CACHE 和 BT_MAX_PCT_FOR_CACHE
        backtest_result = run_backtest(
            start_date=start_date_for_cal,
            end_date=last_trade,
            hold_days=HOLD_DAYS_OPTIONS,
            backtest_top_k=BACKTEST_TOP_K,
            bt_min_pct_chg=BT_MIN_PCT_FOR_CACHE, # 传入最低涨幅
            bt_max_pct_chg=BT_MAX_PCT_FOR_CACHE, # 传入最高涨幅
            cache_breaker=CACHE_BREAKER # 传入缓存破坏键
        )

        bt_df = pd.DataFrame(backtest_result).T
        bt_df.index.name = "持股天数"
        bt_df = bt_df.reset_index()
        bt_df['持股天数'] = bt_df['持股天数'].astype(str) + ' 天'
        
        st.dataframe(bt_df, use_container_width=True, hide_index=True)
        st.success("回测完成！")
        
        # ... (下载按钮代码不变) ...
        export_df = bt_df.copy()
        export_df.columns = ['HoldDays', 'AvgReturn', 'WinRate', 'TotalTrades']
        out_csv_bt = export_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "下载回测结果 CSV", 
            data=out_csv_bt, 
            file_name=f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
