# -*- coding: utf-8 -*-
"""
选股王 · 右侧启动/强势股策略 v1.4 - 涨跌幅控制修复版
说明：
- **V1.4 核心修复：** 将回测模块中最严格的硬性过滤条件——当日涨跌幅限制——移动到侧边栏，
  彻底解决“交易次数0”的问题，让用户可以灵活控制策略的严格程度。
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
st.set_page_config(page_title="选股王 · 右侧启动/强势股策略 v1.4", layout="wide")
st.title("选股王 · 右侧启动/强势股策略 v1.4 (涨跌幅控制修复版)")
st.markdown("输入你的 Tushare Token。本策略核心为**寻找突破强势股**。")

# ---------------------------
# 侧边栏参数（实时可改）
# ---------------------------
with st.sidebar:
    st.header("可调参数（策略核心）")
    # 策略硬性过滤参数
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    MIN_MARKET_CAP = float(st.number_input("最低市值 (元)", value=2000000000.0, step=100000000.0)) # 默认 20亿
    MAX_MARKET_CAP = float(st.number_input("最高市值 (元)", value=50000000000.0, step=1000000000.0)) # 默认 500亿
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=3.0, step=0.5))
    VOL_RATIO_THRESHOLD = float(st.number_input("最低量比阈值 (vol_ratio >= x)", value=1.5, step=0.1, help="量比 1.5 表示今日成交量比前5日平均放大50%"))
    MAX_20D_RETURN = float(st.number_input("最大20日涨幅限制 (%)", value=60.0, step=5.0))
    
    st.markdown("---")
    # 策略评分参数
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1500, step=100))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    
    st.markdown("---")
    # --- 历史回测参数 ---
    st.header("历史回测参数")
    BACKTEST_DAYS = int(st.number_input("回测交易日天数", value=60, min_value=10, max_value=250))
    BACKTEST_TOP_K = int(st.number_input("回测每日最多交易 K 支", value=3, min_value=1, max_value=10))
    HOLD_DAYS_OPTIONS = st.multiselect("回测持股天数", options=[1, 3, 5, 10, 20], default=[1, 3, 5])
    
    # V1.4 新增的涨跌幅控制参数 (用于回测时的硬过滤)
    st.subheader("回测当日涨跌幅控制")
    MIN_PCT_CHG_BT = float(st.number_input("回测：当日最低涨幅 (%)", value=1.0, step=0.5, help="右侧启动要求当日涨幅>=1%"))
    MAX_PCT_CHG_BT = float(st.number_input("回测：当日最高涨幅 (%)", value=9.8, step=0.5, help="避免涨停板买不到"))
    
    # 缓存破坏键
    CACHE_BREAKER = float(st.number_input("回测：缓存破坏键（任意修改刷新回测）", value=1.0, step=0.1))
    st.caption("提示：**本次回测强制使用右侧启动策略。**")

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
# 核心指标计算函数 (此部分与选股有关，保持不变)
# ---------------------------
@st.cache_data(ttl=36000)
def get_hist_cached(ts_code, end_date, days=120): 
    # ... (函数体保持不变) ...
    try:
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days * 2)).strftime("%Y%m%d")
        df = safe_get(pro.daily, ts_code=ts_code, start_date=start, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_values('trade_date').reset_index(drop=True)
        return df.tail(days).reset_index(drop=True) 
    except:
        return pd.DataFrame()

def compute_indicators(df):
    # ... (函数体保持不变) ...
    res = {}
    if df.empty or len(df) < 20:
        return res
    
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    pct = df['pct_chg'].astype(float)
    
    for n in (5, 10, 20):
        res[f'ma{n}'] = close.rolling(window=n).mean().iloc[-1]
        
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    res['macd_diff'] = diff.iloc[-1]
    res['macd_dea'] = dea.iloc[-1]
    res['macd_golden'] = (res['macd_diff'] > res['macd_dea']) and (diff.iloc[-2] <= dea.iloc[-2])
    
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        avg_prev5 = np.mean(vols[-6:-1])
        res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9) if avg_prev5 > 0 else 1.0
    else:
        res['vol_ratio'] = np.nan
        
    if len(close) >= 20:
        res['20d_return'] = (close.iloc[-1] / close.iloc[-20] - 1) * 100
        res['recent20_high'] = float(high.iloc[-20:].max())
        res['break_20d_high'] = close.iloc[-1] > res['recent20_high']
    else:
        res['20d_return'] = np.nan
        res['recent20_high'] = np.nan
        res['break_20d_high'] = False
        
    if len(pct) >= 4:
        res['prev3_sum'] = pct.iloc[-4:-1].sum()
    else:
        res['prev3_sum'] = np.nan
        
    res['last_close'] = close.iloc[-1]
        
    return res

# ---------------------------
# 选股逻辑 (此部分与当日选股有关，保持不变)
# ---------------------------
if st.button("🚀 运行当日选股（初次运行可能较久）"):
    # ... (函数体保持不变) ...
    st.write("正在拉取当日 daily（涨幅榜）作为初筛...")
    daily_all = safe_get(pro.daily, trade_date=last_trade)
    if daily_all.empty:
        st.error("无法获取当日 daily 数据。")
        st.stop()
        
    daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
    pool0 = daily_all.head(INITIAL_TOP_N).copy()
    
    st.write("尝试加载 stock_basic / daily_basic / moneyflow 等高级接口...")
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,total_mv,circ_mv')
    daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)
    
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in','net_mf_out']
        col = next((c for c in possible if c in mf_raw.columns), None)
        if col:
            moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)
    
    pool_merged = pool0.copy()
    if not stock_basic.empty:
        pool_merged = pool_merged.merge(stock_basic[['ts_code','name','industry','total_mv','circ_mv']], 
                                        on='ts_code', how='left')
    if not daily_basic.empty:
        pool_merged = pool_merged.merge(daily_basic[['ts_code','turnover_rate','amount','total_mv','circ_mv']], 
                                        on='ts_code', how='left', suffixes=('_daily','_basic'))
    if not moneyflow.empty:
        pool_merged = pool_merged.merge(moneyflow, on='ts_code', how='left').fillna({'net_mf': 0.0})
        
    pool_merged['amount_yuan'] = pool_merged['amount_daily'].fillna(0) * 1000.0
    
    st.write("正在对初筛池进行硬性过滤（价格/市值/ST/北交所/换手/成交额）...")
    clean_df = pool_merged.copy()
    
    MIN_CAP_WAN = MIN_MARKET_CAP / 10000.0
    MAX_CAP_WAN = MAX_MARKET_CAP / 10000.0
    clean_df = clean_df[(clean_df['close'] >= MIN_PRICE) & (clean_df['close'] <= MAX_PRICE)]
    clean_df = clean_df[(clean_df['total_mv_basic'] >= MIN_CAP_WAN) & (clean_df['total_mv_basic'] <= MAX_CAP_WAN)]
    clean_df = clean_df[~clean_df['name'].str.contains('ST|退', na=False)]
    clean_df = clean_df[~clean_df['ts_code'].str.startswith('4', na=False)]
    clean_df = clean_df[~clean_df['ts_code'].str.startswith('8', na=False)]
    clean_df = clean_df[(clean_df['vol'] > 0) & (clean_df['amount_yuan'] > 0)]
    clean_df = clean_df[clean_df['turnover_rate'] >= MIN_TURNOVER]
    
    st.write(f"硬性过滤后候选数量：{len(clean_df)} 支。")
    if len(clean_df) == 0:
        st.error("硬性过滤后没有候选，请放宽侧边栏条件。")
        st.stop()
        
    st.write("为候选股逐票计算指标（MA/MACD/突破/20日涨幅等）...")
    records = []
    pbar = st.progress(0)
    
    for idx, row in clean_df.iterrows():
        ts_code = row['ts_code']
        hist = get_hist_cached(ts_code, last_trade, days=60)
        ind = compute_indicators(hist)
        
        d20_ret = ind.get('20d_return', 0)
        if d20_ret > MAX_20D_RETURN:
            pbar.progress((idx+1)/len(clean_df))
            continue
            
        ma5, ma10, ma20 = ind.get('ma5'), ind.get('ma10'), ind.get('ma20')
        if not (ma5 > ma10 and ma10 > ma20):
            pbar.progress((idx+1)/len(clean_df))
            continue
            
        prev3_sum = ind.get('prev3_sum', 0)
        if prev3_sum < -5.0 and row['pct_chg'] > 4.0:
            pbar.progress((idx+1)/len(clean_df))
            continue

        row_dict = row.to_dict()
        row_dict.update(ind)
        records.append(row_dict)
        pbar.progress((idx+1)/len(clean_df))
    
    pbar.progress(1.0)
    fdf = pd.DataFrame(records)
    fdf = fdf.dropna(subset=['ma20']).reset_index(drop=True)
    st.write(f"策略硬过滤后，进入评分阶段的候选数量：{len(fdf)} 支。")
    if fdf.empty:
        st.error("所有股票都被过滤，请放宽过滤条件。")
        st.stop()

    def norm_col(s):
        s = s.fillna(s.median()).replace([np.inf,-np.inf], np.nan).fillna(s.median())
        mn = s.min(); mx = s.max()
        if mx - mn < 1e-9:
            return pd.Series([0.5]*len(s), index=s.index)
        return (s - mn) / (mx - mn)

    fdf['s_pct'] = norm_col(fdf['pct_chg'])
    fdf['s_volratio'] = norm_col(fdf['vol_ratio'])
    fdf['s_turn'] = norm_col(fdf['turnover_rate'])
    fdf['s_net_mf'] = norm_col(fdf['net_mf'])
    fdf['s_macd_diff'] = norm_col(fdf['macd_diff'])
    
    fdf['f_break_20d'] = (fdf['break_20d_high']).astype(float)
    fdf['f_macd_golden'] = (fdf['macd_golden']).astype(float)
    fdf['f_vol_price_up'] = ((fdf['vol_ratio'] >= VOL_RATIO_THRESHOLD) & (fdf['pct_chg'] > 0)).astype(float)
    
    fdf['综合评分'] = (
        fdf['f_break_20d'] * 0.25 +       
        fdf['f_macd_golden'] * 0.20 +     
        fdf['f_vol_price_up'] * 0.15 +    
        fdf['s_pct'] * 0.10 +             
        fdf['s_volratio'] * 0.10 +        
        fdf['s_turn'] * 0.10 +            
        fdf['s_net_mf'] * 0.10             
    )

    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index = fdf.index + 1
    
    st.success(f"评分完成：总候选 {len(fdf)} 支，显示 Top {min(TOP_DISPLAY, len(fdf))}。")
    display_cols = ['name','ts_code','综合评分','pct_chg','turnover_rate','vol_ratio','net_mf','f_break_20d','f_macd_golden','f_vol_price_up','20d_return','ma5','ma10','ma20']
    
    final_cols = [c for c in display_cols if c in fdf.columns]
    
    st.dataframe(fdf[final_cols].head(TOP_DISPLAY), use_container_width=True)

# ---------------------------
# 历史回测部分 (此处代码逻辑与 V1.2/V1.3 相同，仅修改参数传递和过滤逻辑)
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
            data_cache[date] = daily_df.set_index('ts_code')
        pbar.progress((i + 1) / len(all_trade_dates))
    pbar.progress(1.0)
    return data_cache

@st.cache_data(ttl=36000)
def get_stock_basic_filter(cache_breaker):
    """一次性加载股票基础数据，并构建硬过滤的白名单"""
    _ = cache_breaker 
    st.write("正在构建回测的股票白名单（ST/北交所过滤）...")
    
    df = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,list_date')
    if df.empty:
        return pd.DataFrame()
        
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('4', na=False)]
    df = df[~df['ts_code'].str.startswith('8', na=False)]
    
    return df[['ts_code']]

@st.cache_data(ttl=6000)
def run_backtest_right_side(start_date, end_date, hold_days, backtest_top_k, cache_breaker, 
                            min_price, max_price, min_cap, max_cap, min_turnover, vol_ratio_threshold, max_20d_ret,
                            min_pct_chg_bt, max_pct_chg_bt): # V1.4 新增参数
    
    _ = cache_breaker 

    trade_dates = get_trade_cal(start_date, end_date)
    
    if not trade_dates:
        return {h: {'returns': [], 'wins': 0, 'total': 0} for h in hold_days}

    results = {h: {'returns': [], 'wins': 0, 'total': 0} for h in hold_days}
    
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
    
    # 2. 批量加载 daily_basic
    st.write("正在批量预加载回测所需的 daily_basic 数据 (加速中...)")
    daily_basic_cache = {}
    if required_dates:
        start_bulk = min(required_dates)
        end_bulk = max(required_dates)

        daily_basic_df = safe_get(pro.query, 
                                        api_name='daily_basic',
                                        start_date=start_bulk, 
                                        end_date=end_bulk, 
                                        fields='ts_code,trade_date,turnover_rate,total_mv,circ_mv')
        
        if not daily_basic_df.empty:
            daily_basic_cache = daily_basic_df.groupby('trade_date').apply(lambda x: x.set_index('ts_code')).to_dict('index')

    # 3. 预加载 daily 数据
    data_cache = load_backtest_data(sorted(list(required_dates)))

    # ----------------------------------------------------
    # --- 阶段二：回测主循环 ---
    # ----------------------------------------------------
    
    st.write(f"正在模拟 {len(backtest_dates)} 个交易日的右侧启动选股回测...")
    pbar_bt = st.progress(0)
    
    for i, buy_date in enumerate(backtest_dates):
        daily_df = data_cache.get(buy_date)
        daily_basic_dict = daily_basic_cache.get(buy_date)
        
        if daily_df is None or daily_df.empty or daily_basic_dict is None:
            pbar_bt.progress((i+1)/len(backtest_dates)); continue

        daily_basic_df_today = pd.DataFrame.from_dict(daily_basic_dict, orient='index')
        
        # 0. 初始过滤：只保留白名单中的股票
        daily_df_filtered = daily_df[daily_df.index.isin(valid_ts_codes)]
        
        # 合并每日基础数据
        merged_df = daily_df_filtered.join(daily_basic_df_today, how='inner', lsuffix='_daily', rsuffix='_basic').reset_index()
        
        # 1. 应用硬过滤 (注意单位转换)
        MIN_CAP_WAN = min_cap / 10000.0
        MAX_CAP_WAN = max_cap / 10000.0
        
        filtered_df = merged_df.copy()
        
        # F1: 价格区间
        filtered_df = filtered_df[(filtered_df['close_daily'] >= min_price) & (filtered_df['close_daily'] <= max_price)]
        
        # F2: 市值区间 (Tushare 单位万元)
        filtered_df = filtered_df[(filtered_df['total_mv_basic'] >= MIN_CAP_WAN) & (filtered_df['total_mv_basic'] <= MAX_CAP_WAN)]
        
        # F3: 最低换手率
        filtered_df = filtered_df[filtered_df['turnover_rate_basic'] >= min_turnover]
        
        # F4: 停牌 / 无成交 
        filtered_df = filtered_df[(filtered_df['vol_daily'] > 0)]
        
        # V1.4 修复：使用侧边栏的涨跌幅参数作为过滤
        filtered_df = filtered_df[
            (filtered_df['pct_chg_daily'] >= min_pct_chg_bt) & 
            (filtered_df['pct_chg_daily'] <= max_pct_chg_bt)
        ].copy() 
        
        # 2. 模拟策略评分
        filtered_df['score_proxy'] = (filtered_df['pct_chg_daily'] ** 2) * filtered_df['turnover_rate_basic']
        
        scored_stocks = filtered_df.sort_values("score_proxy", ascending=False).head(backtest_top_k).copy()
        
        for _, row in scored_stocks.iterrows():
            ts_code = row['ts_code']
            buy_price = float(row['close_daily'])
            
            if pd.isna(buy_price) or buy_price <= 0: continue

            for h in hold_days:
                try:
                    current_index = trade_dates.index(buy_date)
                    sell_date = trade_dates[current_index + h]
                except (ValueError, IndexError):
                    continue
        
                sell_df_cached = data_cache.get(sell_date)
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
if st.checkbox("✅ 运行历史回测（右侧启动策略）", value=False):
    if not HOLD_DAYS_OPTIONS:
        st.warning("请至少选择一个回测持股天数。")
    else:
        st.header("📈 历史回测结果（买入收盘价 / 卖出收盘价）")
        
        try:
            start_date_for_cal = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
        except:
            start_date_for_cal = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            
        # V1.4 传入新增的参数
        backtest_result = run_backtest_right_side(
            start_date=start_date_for_cal,
            end_date=last_trade,
            hold_days=HOLD_DAYS_OPTIONS,
            backtest_top_k=BACKTEST_TOP_K,
            cache_breaker=CACHE_BREAKER,
            min_price=MIN_PRICE, 
            max_price=MAX_PRICE,
            min_cap=MIN_MARKET_CAP,
            max_cap=MAX_MARKET_CAP,
            min_turnover=MIN_TURNOVER,
            vol_ratio_threshold=VOL_RATIO_THRESHOLD,
            max_20d_ret=MAX_20D_RETURN,
            min_pct_chg_bt=MIN_PCT_CHG_BT, # 新参数
            max_pct_chg_bt=MAX_PCT_CHG_BT  # 新参数
        )

        bt_df = pd.DataFrame(backtest_result).T
        bt_df.index.name = "持股天数"
        bt_df = bt_df.reset_index()
        bt_df['持股天数'] = bt_df['持股天数'].astype(str) + ' 天'
        
        st.dataframe(bt_df, use_container_width=True, hide_index=True)
        st.success("回测完成！")
        
# ---------------------------
# 小结与操作提示
# ---------------------------
st.markdown("### 小结与操作提示")
st.markdown("""
- **策略：** **右侧启动/强势股 v1.4** (涨跌幅控制修复版)。
- **核心修复：** 最严格的**当日涨跌幅过滤**已移至侧边栏 **`回测当日涨跌幅控制`**。
- **操作步骤：**
    1. **粘贴并运行此代码。**
    2. **最关键的一步：** 在侧边栏的 **`回测当日涨跌幅控制`** 部分，将 **`当日最低涨幅 (%)`** 改为 **负值**（例如 `-10.0`），将 **`当日最高涨幅 (%)`** 改为 **正值**（例如 `10.0`）。
    3. 重新运行回测。如果现在还有 0 交易，说明是**市值/价格**太严格，请放宽 Step 2 的参数。
""")
