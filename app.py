# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（BC 混合增强版）
说明：
- 目标：短线爆发 (B) + 妖股捕捉 (C)，持股 1-5 天
- 在界面输入 Tushare Token（仅本次运行使用）
- 尽可能调用 moneyflow / chip / ths_member 等高级接口，若无权限会自动降级
- 已做大量异常处理与缓存，降低因接口波动导致的报错
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
st.set_page_config(page_title="选股王 · 10000旗舰（BC增强）", layout="wide")
st.title("选股王 · 10000 积分旗舰（BC 混合增强版）")
st.markdown("输入你的 Tushare Token（仅本次运行使用）。若有权限缺失，脚本会自动降级并继续运行。")

# ---------------------------
# 侧边栏参数（实时可改）
# ---------------------------
with st.sidebar:
    st.header("可调参数（实时）")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=500, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=3.0, step=0.5))
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=200_000_000.0, step=50_000_000.0))
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=8.0, step=0.5))
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    st.markdown("---")
    st.caption("提示：保守→降低阈值；激进→提高阈值。")

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

# ---------------------------
# 拉当日涨幅榜初筛
# ---------------------------
st.write("正在拉取当日 daily（涨幅榜）作为初筛...")
daily_all = safe_get(pro.daily, trade_date=last_trade)
if daily_all.empty:
    st.error("无法获取当日 daily 数据（Tushare 返回空）。请确认 Token 权限。")
    st.stop()

daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
st.write(f"当日记录：{len(daily_all)}，取涨幅前 {INITIAL_TOP_N} 作为初筛。")
pool0 = daily_all.head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

# ---------------------------
# 尝试加载高级接口（有权限时启用）
# ---------------------------
st.write("尝试加载 stock_basic / daily_basic / moneyflow / ths_member / chip 等高级接口（若权限允许）...")
stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')
daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)

# 合并基本信息（safe）
def safe_merge_pool(pool_df, other_df, cols):
    pool = pool_df.set_index('ts_code').copy()
    if other_df is None or other_df.empty:
        for c in cols:
            pool[c] = np.nan
        return pool.reset_index()
    if 'ts_code' not in other_df.columns:
        try:
            other_df = other_df.reset_index()
        except:
            for c in cols:
                pool[c] = np.nan
            return pool.reset_index()
    for c in cols:
        if c not in other_df.columns:
            other_df[c] = np.nan
    try:
        joined = pool.join(other_df.set_index('ts_code')[cols], how='left')
    except Exception:
        for c in cols:
            pool[c] = np.nan
        return pool.reset_index()
    for c in cols:
        if c not in joined.columns:
            joined[c] = np.nan
    return joined.reset_index()

# merge stock_basic
if not stock_basic.empty:
    keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_basic.columns]
    try:
        pool0 = pool0.merge(stock_basic[keep], on='ts_code', how='left')
    except Exception:
        pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
else:
    pool0['name'] = pool0['ts_code']; pool0['industry'] = ''

# merge daily_basic
pool_merged = safe_merge_pool(pool0, daily_basic, ['turnover_rate','amount','total_mv','circ_mv'])

# ---------------------------
# 基本清洗（ST / 停牌 / 价格区间 / 一字板 / 换手 / 成交额 / 市值 / 下跌趋势）
# ---------------------------
st.write("对初筛池进行清洗（ST/停牌/价格/一字板/换手/成交额等）...")
clean_list = []
pbar = st.progress(0)
for i, r in enumerate(pool_merged.itertuples()):
    ts = getattr(r, 'ts_code')
    # try volume detection with fallback
    try:
        vol_df = safe_get(pro.daily, ts_code=ts, trade_date=last_trade)
        vol = vol_df.get('vol', pd.Series([0])).iloc[0] if not vol_df.empty else getattr(r, 'vol') if 'vol' in pool_merged.columns else 0
    except:
        vol = getattr(r, 'vol') if 'vol' in pool_merged.columns else 0

    close = getattr(r, 'close', np.nan)
    open_p = getattr(r, 'open', np.nan)
    pre_close = getattr(r, 'pre_close', np.nan)
    pct = getattr(r, 'pct_chg', np.nan)
    amount = getattr(r, 'amount', np.nan)
    turnover = getattr(r, 'turnover_rate', np.nan)
    total_mv = getattr(r, 'total_mv', np.nan)
    name = getattr(r, 'name', ts)

    # skip no trading
    if vol == 0 or (isinstance(amount,(int,float)) and amount == 0):
        pbar.progress((i+1)/len(pool_merged)); continue

    # price filter
    if pd.isna(close): pbar.progress((i+1)/len(pool_merged)); continue
    if (close < MIN_PRICE) or (close > MAX_PRICE): pbar.progress((i+1)/len(pool_merged)); continue

    # exclude ST / delist
    if isinstance(name, str) and (('ST' in name.upper()) or ('退' in name)):
        pbar.progress((i+1)/len(pool_merged)); continue

    # market cap filter (排除市值过大的大盘股)
    try:
        tv = total_mv
        if not pd.isna(tv):
            tv = float(tv)
            if tv > 1e6:
                tv_yuan = tv * 10000.0
            else:
                tv_yuan = tv
            # 排除大市值股
            if tv_yuan > 2000 * 1e8:  # 2000亿
                pbar.progress((i+1)/len(pool_merged)); continue
    except:
        pass

    # **新增：判断是否处于下跌趋势** (例如：10日均线低于当前股价，股价处于下跌趋势)
    try:
        hist_data = safe_get(pro.daily, ts_code=ts, start_date="20230101", end_date=last_trade)
        ma10 = hist_data['close'].rolling(window=10).mean().iloc[-1]  # 10日均线
        if close < ma10:  # 如果当前股价低于10日均线，认为是下跌趋势
            pbar.progress((i+1)/len(pool_merged)); continue
    except:
        pass

    # turnover
    if not pd.isna(turnover):
        try:
            if float(turnover) < MIN_TURNOVER: pbar.progress((i+1)/len(pool_merged)); continue
        except:
            pass

    clean_list.append(r)
    pbar.progress((i+1)/len(pool_merged))

pbar.progress(1.0)
clean_df = pd.DataFrame([dict(zip(r._fields, r)) for r in clean_list])

# ---------------------------
# 计算技术指标：MACD、RSI、成交量与价格配合
# ---------------------------
def compute_macd(df):
    """计算 MACD"""
    if len(df) >= 26:
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']
    return df

def compute_rsi(df, period=14):
    """计算 RSI"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi
    return df

# ---------------------------
# 量价关系分析
# ---------------------------
def volume_price_analysis(df):
    """成交量与股价的关系分析"""
    df['vol_ratio'] = df['vol'] / df['vol'].rolling(window=5).mean()  # 5日成交量比
    return df

# 合并指标并计算
for idx, row in clean_df.iterrows():
    ts_code = row['ts_code']
    df = safe_get(pro.daily, ts_code=ts_code, start_date="20230101", end_date=last_trade)
    
    if df.empty:
        continue
    
    df = compute_macd(df)
    df = compute_rsi(df)
    df = volume_price_analysis(df)
    
    clean_df.at[idx, 'macd'] = df['macd'].iloc[-1] if not df['macd'].isnull().all() else np.nan
    clean_df.at[idx, 'rsi'] = df['rsi'].iloc[-1] if not df['rsi'].isnull().all() else np.nan
    clean_df.at[idx, 'vol_ratio'] = df['vol_ratio'].iloc[-1] if not df['vol_ratio'].isnull().all() else np.nan

# 评分与回馈
st.write("技术指标计算完成，进行回馈与评分：")
clean_df['持股收益'] = clean_df['ts_code'].apply(lambda x: calculate_profit(x, hold_days=5))

# 显示最终结果
st.dataframe(clean_df[['name', 'ts_code', 'macd', 'rsi', 'vol_ratio', 'pct_chg', '持股收益']], use_container_width=True)
