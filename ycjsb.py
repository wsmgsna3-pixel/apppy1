# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（BC 混合增强版）—— 带趋势主导（MA/MACD/量价/突破）增强
说明：
- 目标：短线爆发 (B) + 妖股捕捉 (C)，持股 1-5 天
- **本次优化**：
    1. 移除回测中“风险过滤后，剩余 XX 支”的重复提示。
    2. 增强回测买入价的获取鲁棒性，修复“交易次数0”的问题。
- 采用全局 K 线数据缓存（GLOBAL_KLINE_DATA）和批量预加载机制，大幅提升回测稳定性和速度。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time 
import sys # 用于判断 Streamlit 运行环境

warnings.filterwarnings("ignore")

# ---------------------------
# 全局数据缓存（用于性能优化）
# ---------------------------
# GLOBAL_KLINE_DATA 存储预加载的全市场K线数据
GLOBAL_KLINE_DATA = {} 

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
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=300, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=3.0, step=0.5))
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=200_000_000.0, step=50_000_000.0))
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=8.0, step=0.5))
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    MIN_MARKET_CAP = float(st.number_input("最低市值 (元)", value=2000000000.0, step=100000000.0))  # 默认 20亿
    MAX_MARKET_CAP = float(st.number_input("最高市值 (元)", value=50000000000.0, step=1000000000.0))  # 默认 500亿
    st.markdown("---")
    # --- 回测新增参数 ---
    st.header("回测参数（新增）")
    BACKTEST_DAYS = int(st.number_input("回测交易日天数 (N)", value=60, step=5)) # 默认为 60
    HOLD_DAYS_LIST = st.text_input("回测持股天数（逗号分隔）", value="1, 3, 5")
    try:
        HOLD_DAYS = [int(x.strip()) for x in HOLD_DAYS_LIST.split(',') if x.strip().isdigit()]
    except:
        HOLD_DAYS = [1, 3, 5]
    if not HOLD_DAYS:
         HOLD_DAYS = [1, 3, 5]
         st.warning("持股天数解析失败，使用默认值 1, 3, 5。")
    # ------------------
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
            if len(safe_get(pro.daily, trade_date=ds)) > 100:
                return ds
    return None

@st.cache_data(ttl=3600) 
def get_all_trade_cals(start_date, end_date, max_retries=3):
    """获取指定范围内的所有交易日，带重试机制"""
    for attempt in range(max_retries):
        try:
            df = safe_get(pro.trade_cal, start_date=start_date, end_date=end_date)
            
            if not df.empty:
                return df[df['is_open']==1]['cal_date'].tolist()
            
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return []
                
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return []
    return []

last_trade = find_last_trade_day()
if not last_trade:
    st.error("无法找到最近交易日，检查网络或 Token 权限。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")

# ---------------------------
# 尝试加载高级接口
# ---------------------------
@st.cache_data(ttl=600)
def get_advanced_data(trade_date):
    """缓存并获取当日所有高级数据"""
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')
    daily_basic = safe_get(pro.daily_basic, trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    mf_raw = safe_get(pro.moneyflow, trade_date=trade_date)

    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in','net_mf_out']
        col = None
        for c in possible:
            for c in possible:
                if c in mf_raw.columns:
                    col = c; break
            if col:
                moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)
            else:
                numeric_cols = [c for c in mf_raw.columns if c != 'ts_code' and pd.api.types.is_numeric_dtype(mf_raw[c])]
                col = numeric_cols[0] if numeric_cols else None
                if col:
                     moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)
                else:
                     pass 
                 
    return stock_basic, daily_basic, moneyflow

# ---------------------------
# 合并基本信息 
# ---------------------------
def safe_merge_pool(pool_df, other_df, cols):
    """安全合并辅助函数"""
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

def merge_all_info(pool0, stock_basic, daily_basic, moneyflow):
    """统一合并流程"""
    if not stock_basic.empty:
        keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_basic.columns]
        try:
            pool0 = pool0.merge(stock_basic[keep], on='ts_code', how='left')
        except Exception:
            pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
    else:
        pool0['name'] = pool0['ts_code']; pool0['industry'] = ''

    pool_merged = safe_merge_pool(pool0, daily_basic, ['turnover_rate','amount','total_mv','circ_mv'])

    if moneyflow.empty:
        moneyflow = pd.DataFrame({'ts_code': pool_merged['ts_code'].tolist(), 'net_mf': [0.0]*len(pool_merged)})
    else:
        if 'ts_code' not in moneyflow.columns:
            moneyflow['ts_code'] = None
    try:
        pool_merged = pool_merged.set_index('ts_code').join(moneyflow.set_index('ts_code'), how='left').reset_index()
    except Exception:
        if 'net_mf' not in pool_merged.columns:
            pool_merged['net_mf'] = 0.0

    if 'net_mf' not in pool_merged.columns:
        pool_merged['net_mf'] = 0.0
    pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0.0)
    return pool_merged

# ---------------------------
# 清洗与过滤（clean_and_filter）
# ---------------------------
def clean_and_filter(pool_merged, min_price, max_price, min_turnover, min_amount, min_market_cap, max_market_cap, vol_spike_mult, volatility_max, high_pct_threshold, final_pool):
    """统一清洗和过滤流程"""
    clean_list = []
    
    # 移除内层进度条，只保留日志
    
    for i, r in enumerate(pool_merged.itertuples()):
        ts = getattr(r, 'ts_code')
        vol = getattr(r, 'vol', 0)
        close = getattr(r, 'close', np.nan)
        open_p = getattr(r, 'open', np.nan)
        pre_close = getattr(r, 'pre_close', np.nan)
        pct = getattr(r, 'pct_chg', np.nan)
        amount = getattr(r, 'amount', np.nan)
        turnover = getattr(r, 'turnover_rate', np.nan)
        total_mv = getattr(r, 'total_mv', np.nan)
        name = getattr(r, 'name', ts)
        
        if vol == 0 or (isinstance(amount,(int,float)) and amount == 0):
            continue

        if pd.isna(close) or (close < min_price) or (close > max_price): 
            continue

        if isinstance(name, str) and (('ST' in name.upper()) or ('退' in name)):
            continue

        tsck = getattr(r, 'ts_code', '')
        if isinstance(tsck, str) and (tsck.startswith('4') or tsck.startswith('8')):
            continue

        try:
            tv = getattr(r, 'total_mv', np.nan)
            if not pd.isna(tv):
                tv = float(tv)
                tv_yuan = tv * 10000.0 if tv > 1e6 else tv
                if tv_yuan < min_market_cap or tv_yuan > max_market_cap:
                    continue
        except: pass

        try:
            high = getattr(r, 'high', np.nan); low = getattr(r, 'low', np.nan)
            if (not pd.isna(open_p) and not pd.isna(high) and not pd.isna(low) and not pd.isna(pre_close)):
                if (open_p == high == low == pre_close):
                    continue
        except: pass

        if not pd.isna(turnover):
            try:
                if float(turnover) < min_turnover: 
                    continue
            except: pass

        if not pd.isna(amount):
            amt = amount
            amt = amt * 10000.0 if amt > 0 and amt < 1e5 else amt
            if amt < min_amount: 
                continue

        try:
            if float(pct) < 0: 
                continue
        except: pass

        clean_list.append(r)

    clean_df = pd.DataFrame([dict(zip(r._fields, r)) for r in clean_list])
    
    if len(clean_df) == 0:
        return pd.DataFrame()
        
    clean_df = clean_df.sort_values('pct_chg', ascending=False).head(int(final_pool)).reset_index(drop=True)
    return clean_df

# ---------------------------
# 性能优化：K 线批量加载（核心）
# ---------------------------
@st.cache_data(ttl=600, show_spinner=False)
def get_bulk_daily_data(start_date, end_date, max_retries=3):
    """
    性能优化核心函数：批量获取全市场在指定时间范围内的日线数据。
    返回一个字典: {ts_code: DataFrame(kline)}
    """
    global GLOBAL_KLINE_DATA
    if 'streamlit' in sys.modules:
        st.write(f"📈 正在批量加载全市场 {start_date} 至 {end_date} 的 K 线数据（Tushare调用密集，请耐心等待...）")
    
    for attempt in range(max_retries):
        try:
            # 使用 pro.daily 获取所有股票的日线数据
            df_all = safe_get(pro.daily, start_date=start_date, end_date=end_date)
            
            if df_all.empty:
                if attempt < max_retries - 1:
                    if 'streamlit' in sys.modules: st.warning(f"第 {attempt + 1}/{max_retries} 次尝试：批量获取 K 线数据返回空，正在重试（等待 5 秒）...")
                    time.sleep(5)
                    continue
                else:
                    if 'streamlit' in sys.modules: st.error("批量获取 K 线数据最终失败或返回空，回测无法进行。")
                    return {}
            
            # 成功后，按 ts_code 分组，并确保按 trade_date 排序
            # *** 修复：强制将 close/open/high/low/pre_close 转换为 float，避免类型问题 ***
            for col in ['close','open','high','low','pre_close','vol','amount']:
                if col in df_all.columns:
                    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

            GLOBAL_KLINE_DATA = {
                ts_code: group.sort_values('trade_date').reset_index(drop=True)
                for ts_code, group in df_all.groupby('ts_code')
            }
            if 'streamlit' in sys.modules: st.write(f"✅ K 线数据加载完成（第 {attempt + 1} 次尝试）。共获取 {len(GLOBAL_KLINE_DATA)} 支股票的历史数据。")
            return GLOBAL_KLINE_DATA
            
        except Exception as e:
            if attempt < max_retries - 1:
                if 'streamlit' in sys.modules: st.warning(f"第 {attempt + 1}/{max_retries} 次尝试：批量获取 K 线数据出错：{e}。正在重试（等待 5 秒）...")
                time.sleep(5)
            else:
                if 'streamlit' in sys.modules: st.error(f"批量获取 K 线数据最终失败，请检查网络或 Token 权限。错误：{e}")
                return {}
    
    return {} 

# ---------------------------
# 评分指标计算（从全局缓存读取）
# ---------------------------
def compute_indicators(ts_code, end_date, days=60):
    """从全局缓存中获取数据并计算指标"""
    res = {}
    
    if ts_code not in GLOBAL_KLINE_DATA:
        return res
        
    df_full = GLOBAL_KLINE_DATA[ts_code]
    
    # 筛选出当前回测日之前的数据（包括 end_date 当天）
    df = df_full[df_full['trade_date'] <= end_date].tail(days + 26) 
    
    if df.empty or len(df) < 3:
        return res
        
    # --- 指标计算逻辑 ---
    # 由于在批量加载时已转为 float，这里直接使用
    close = df['close']
    high = df['high']
    low = df['low']

    try: res['last_close'] = close.iloc[-1]
    except: res['last_close'] = np.nan

    for n in (5,10,20):
        if len(close) >= n:
            res[f'ma{n}'] = close.rolling(window=n).mean().iloc[-1]
        else:
            res[f'ma{n}'] = np.nan

    # MACD (12,26,9)
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_val = (diff - dea) * 2
        res['macd'] = macd_val.iloc[-1]; res['diff'] = diff.iloc[-1]; res['dea'] = dea.iloc[-1]
    else:
        res['macd'] = res['diff'] = res['dea'] = np.nan

    # KDJ
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
        res['k'] = res['d'] = res['j'] = np.nan

    # vol ratio and metrics
    vols = df['vol'].tolist()
    if len(vols) >= 6:
        avg_prev5 = np.mean(vols[-6:-1])
        res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9)
        res['vol_last'] = vols[-1]
        res['vol_ma5'] = avg_prev5
    else:
        res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan

    # 10d return
    if len(close) >= 10:
        res['10d_return'] = close.iloc[-1] / close.iloc[-10] - 1
    else:
        res['10d_return'] = np.nan

    # prev3_sum for down-then-bounce detection
    if 'pct_chg' in df.columns and len(df) >= 4:
        try:
            pct = df['pct_chg'].astype(float)
            res['prev3_sum'] = pct.iloc[-4:-1].sum()
        except:
            res['prev3_sum'] = np.nan
    else:
        res['prev3_sum'] = np.nan

    # volatility (std of last 10 pct_chg)
    try:
        if 'pct_chg' in df.columns and len(df) >= 10:
            res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()
        else:
            res['volatility_10'] = np.nan
    except:
        res['volatility_10'] = np.nan

    # recent 20-day high for breakout detection
    try:
        if len(high) >= 20:
            res['recent20_high'] = float(high.tail(20).max())
        else:
            res['recent20_high'] = float(high.max()) if len(high)>0 else np.nan
    except:
        res['recent20_high'] = np.nan

    # 阳线实体强度（今天）
    try:
        today_open = df['open'].iloc[-1]
        today_close = df['close'].iloc[-1]
        today_high = df['high'].iloc[-1]
        today_low = df['low'].iloc[-1]
        # 修复：防止遇到 nan 价格时计算失败
        if pd.isna(today_open) or pd.isna(today_close) or pd.isna(today_high) or pd.isna(today_low):
            res['yang_body_strength'] = 0.0
        else:
            body = abs(today_close - today_open)
            rng = max(today_high - today_low, 1e-9)
            res['yang_body_strength'] = body / rng
    except:
        res['yang_body_strength'] = 0.0
        
    return res

# ---------------------------
# 评分计算主体
# ---------------------------
def compute_scores(clean_df, current_trade_date, min_market_cap, max_market_cap, vol_spike_mult, volatility_max, high_pct_threshold):
    """统一评分和风险过滤流程"""
    records = []
        
    for idx, row in enumerate(clean_df.itertuples()):
        ts_code = getattr(row, 'ts_code')
        name = getattr(row, 'name', ts_code)
        pct_chg = getattr(row, 'pct_chg', 0.0)
        amount = getattr(row, 'amount', np.nan)
        if amount is not None and not pd.isna(amount) and amount > 0 and amount < 1e5:
            amount = amount * 10000.0

        turnover_rate = getattr(row, 'turnover_rate', np.nan)
        net_mf = float(getattr(row, 'net_mf', 0.0))
        # 修复2：获取原始 close 价格作为回测买入价的鲁棒来源
        buy_price = getattr(row, 'close', np.nan)

        ind = compute_indicators(ts_code, current_trade_date, days=60)

        vol_ratio = ind.get('vol_ratio', np.nan)
        ten_return = ind.get('10d_return', np.nan)
        ma5 = ind.get('ma5', np.nan)
        ma10 = ind.get('ma10', np.nan)
        ma20 = ind.get('ma20', np.nan)
        macd = ind.get('macd', np.nan)
        diff = ind.get('diff', np.nan)
        dea = ind.get('dea', np.nan)
        k, d, j = ind.get('k', np.nan), ind.get('d', np.nan), ind.get('j', np.nan)
        last_close = ind.get('last_close', np.nan)
        vol_last = ind.get('vol_last', np.nan)
        vol_ma5 = ind.get('vol_ma5', np.nan)
        prev3_sum = ind.get('prev3_sum', np.nan)
        volatility_10 = ind.get('volatility_10', np.nan)
        recent20_high = ind.get('recent20_high', np.nan)
        yang_body_strength = ind.get('yang_body_strength', 0.0)

        try:
            proxy_money = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
        except:
            proxy_money = 0.0

        rec = {
            'ts_code': ts_code, 'name': name, 'pct_chg': pct_chg,
            'amount': amount if not pd.isna(amount) else 0.0,
            'turnover_rate': turnover_rate if not pd.isna(turnover_rate) else np.nan,
            'net_mf': net_mf,
            'vol_ratio': vol_ratio if not pd.isna(vol_ratio) else np.nan,
            '10d_return': ten_return if not pd.isna(ten_return) else np.nan,
            'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
            'macd': macd, 'diff': diff, 'dea': dea, 'k': k, 'd': d, 'j': j,
            'last_close': last_close, # K线计算的收盘价
            'buy_price': buy_price, # 修复2：原始 daily 接口的收盘价，作为回测的鲁棒买入价
            'vol_last': vol_last, 'vol_ma5': vol_ma5, 'recent20_high': recent20_high, 'yang_body_strength': yang_body_strength,
            'prev3_sum': prev3_sum, 'volatility_10': volatility_10,
            'proxy_money': proxy_money
        }

        records.append(rec)

    fdf = pd.DataFrame(records)
    
    if fdf.empty: 
        if 'streamlit' in sys.modules: st.error("【内部错误】指标计算后 DataFrame 为空。请确认 K 线数据是否成功加载。")
        return pd.DataFrame()

    count_before_filter = len(fdf) 
    
    # 风险过滤
    try:
        # 1. 远离 MA20 且大阳线
        if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):
            mask_high_big = (fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > high_pct_threshold)
            fdf = fdf[~mask_high_big]

        # 2. 连续下跌后大阳线
        if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
            mask_down_rebound = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > high_pct_threshold)
            fdf = fdf[~mask_down_rebound] # 保留，因为我们要做反向过滤

        # 3. 巨量放量
        if all(c in fdf.columns for c in ['vol_last','vol_ma5']):
            mask_vol_spike = (fdf['vol_last'] > (fdf['vol_ma5'] * vol_spike_mult))
            fdf = fdf[~mask_vol_spike]

        # 4. 波动率过大
        if 'volatility_10' in fdf.columns:
            mask_volatility = fdf['volatility_10'] > volatility_max
            fdf = fdf[~mask_volatility]
    except: pass 

    count_after_risk_filter = len(fdf)
    if count_after_risk_filter == 0:
        if 'streamlit' in sys.modules: 
            st.error(f"【过滤失败】风险过滤机制排除了所有 {count_before_filter} 支股票。请放宽侧边栏风险参数（例如降低**放量倍数阈值**、提高**波动 std 阈值**等）。")
        return pd.DataFrame()
    
    # 修复1：移除重复的 st.write 提示
    # if 'streamlit' in sys.modules: 
    #     st.write(f"风险过滤后，剩余 {count_after_risk_filter} 支候选股进入下一阶段。")


    # ** 移除 MA 多头硬过滤，改为只靠评分来决定权重 ** if fdf.empty:
        if 'streamlit' in sys.modules: st.error("【内部错误】经过所有过滤后，评分池为空。")
        return pd.DataFrame()
        
    # RSL
    if '10d_return' in fdf.columns:
        try:
            market_mean_10d = fdf['10d_return'].replace([np.inf,-np.inf], np.nan).dropna().mean()
            if np.isnan(market_mean_10d) or abs(market_mean_10d) < 1e-9:
                market_mean_10d = 1e-9
            fdf['rsl'] = fdf['10d_return'] / market_mean_10d
        except:
            fdf['rsl'] = 1.0
    else:
        fdf['rsl'] = 1.0
        
    # 子指标归一化
    def norm_col(s):
        s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        mn = s.min(); mx = s.max()
        if mx - mn < 1e-9:
            return pd.Series([0.5]*len(s), index=s.index)
        return (s - mn) / (mx - mn)

    fdf['s_pct'] = norm_col(fdf.get('pct_chg', pd.Series([0]*len(fdf))))
    fdf['s_volratio'] = norm_col(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))
    fdf['s_turn'] = norm_col(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))
    if 'net_mf' in fdf.columns and fdf['net_mf'].abs().sum() > 0:
        fdf['s_money'] = norm_col(fdf.get('net_mf', pd.Series([0]*len(fdf))))
    else:
        fdf['s_money'] = norm_col(fdf.get('proxy_money', pd.Series([0]*len(fdf))))
    fdf['s_amount'] = norm_col(fdf.get('amount', pd.Series([0]*len(fdf))))
    fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))
    fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
    fdf['s_rsl'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))
    fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))

    # 趋势因子与强化评分 (这里保留了 MA 多头排列的加分项)
    fdf['ma_trend_flag'] = ((fdf.get('ma5', pd.Series([])) > fdf.get('ma10', pd.Series([]))) & (fdf.get('ma10', pd.Series([])) > fdf.get('ma20', pd.Series([])))).fillna(False)
    fdf['macd_golden_flag'] = (fdf.get('diff', 0) > fdf.get('dea', 0)).fillna(False)
    fdf['vol_price_up_flag'] = (fdf.get('vol_last', 0) > fdf.get('vol_ma5', 0)).fillna(False)
    fdf['break_high_flag'] = (fdf.get('last_close', 0) > fdf.get('recent20_high', 0)).fillna(False)
    fdf['yang_body_strength'] = fdf.get('yang_body_strength', 0.0).fillna(0.0)

    fdf['trend_score_raw'] = (
        fdf['ma_trend_flag'].astype(float) * 1.0 +
        fdf['macd_golden_flag'].astype(float) * 1.3 +
        fdf['vol_price_up_flag'].astype(float) * 1.0 +
        fdf['break_high_flag'].astype(float) * 1.3 +
        fdf['yang_body_strength'].astype(float) * 0.8
    )

    fdf['trend_score'] = norm_col(fdf['trend_score_raw'])

    # 最终综合评分
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
    
    return fdf

# ---------------------------
# 回测主模块
# ---------------------------
def run_backtest(trade_dates, hold_days, top_k):
    """
    运行回测。
    trade_dates: 需要回测的交易日列表 (即买入日)
    hold_days: 持有天数列表 [1, 3, 5]
    top_k: 每天选股 Top K
    """
    global GLOBAL_KLINE_DATA 
    
    if not trade_dates:
        return pd.DataFrame()
        
    start_buy_date = trade_dates[0]
    lookback_days = 60 * 2 + max(HOLD_DAYS) 
    start_kline_date = (datetime.strptime(start_buy_date, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    
    get_bulk_daily_data(start_kline_date, last_trade)
    
    if not GLOBAL_KLINE_DATA:
        if 'streamlit' in sys.modules: st.error("全局 K 线数据加载失败，回测无法进行。")
        return pd.DataFrame()


    if 'streamlit' in sys.modules: st.info(f"开始回测：{trade_dates[0]} 到 {trade_dates[-1]}，持股 {hold_days} 天，每日选择 Top {top_k}。")
    
    results = {h: {'returns': [], 'wins': 0, 'total': 0} for h in hold_days}
    
    max_lookback = BACKTEST_DAYS + max(HOLD_DAYS) + 30
    start_lookback = (datetime.strptime(trade_dates[0], "%Y%m%d") - timedelta(days=max_lookback)).strftime("%Y%m%d")
    all_trade_cals = get_all_trade_cals(start_lookback, last_trade) 
    all_trade_cals.sort() 

    if len(all_trade_cals) == 0:
        if 'streamlit' in sys.modules: st.error("无法获取交易日历（Trade Cal），回测失败。")
        return pd.DataFrame()

    pbar = None
    if 'streamlit' in sys.modules:
        pbar = st.progress(0, text=f"回测进度：0 / {len(trade_dates)} 天")

    for i, buy_date in enumerate(trade_dates):
        
        daily_all = safe_get(pro.daily, trade_date=buy_date)
        if daily_all.empty: 
            if 'streamlit' in sys.modules: st.warning(f"跳过回测日 {buy_date}: 无法获取当日 daily 数据。")
            if pbar: pbar.progress((i + 1) / len(trade_dates), text=f"回测进度：{i+1} / {len(trade_dates)} 天"); continue
            
        daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
            
        pool0 = daily_all.head(INITIAL_TOP_N).copy().reset_index(drop=True)
        
        # 使用缓存的 get_advanced_data 
        stock_basic, daily_basic, moneyflow = get_advanced_data(buy_date)
        
        pool_merged = merge_all_info(pool0, stock_basic, daily_basic, moneyflow)

        # 清洗/过滤
        clean_df = clean_and_filter(pool_merged, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_MARKET_CAP, MAX_MARKET_CAP, VOL_SPIKE_MULT, VOLATILITY_MAX, HIGH_PCT_THRESHOLD, FINAL_POOL)
        if clean_df.empty: 
            if pbar: pbar.progress((i + 1) / len(trade_dates), text=f"回测进度：{i+1} / {len(trade_dates)} 天"); continue

        # 评分/再次过滤
        fdf_scored = compute_scores(clean_df, buy_date, MIN_MARKET_CAP, MAX_MARKET_CAP, VOL_SPIKE_MULT, VOLATILITY_MAX, HIGH_PCT_THRESHOLD)
        if fdf_scored.empty: 
            if pbar: pbar.progress((i + 1) / len(trade_dates), text=f"回测进度：{i+1} / {len(trade_dates)} 天"); continue
        
        fdf_scored = fdf_scored.sort_values('综合评分', ascending=False).head(top_k)
        
        try:
            buy_date_cal_idx = all_trade_cals.index(buy_date)
        except ValueError:
            if pbar: pbar.progress((i + 1) / len(trade_dates), text=f"回测进度：{i+1} / {len(trade_dates)} 天"); continue
        
        # 核心收益计算循环
        for _, row in fdf_scored.iterrows():
            ts_code = row['ts_code']
            
            # 修复2：更鲁棒的买入价获取
            buy_close = np.nan
            try:
                # 优先使用 compute_scores 中存储的原始 daily close 价格
                buy_close = float(row.get('buy_price', row.get('last_close', np.nan)))
            except (ValueError, TypeError):
                buy_close = np.nan
            
            # 严格检查买入价
            if pd.isna(buy_close) or buy_close <= 0: 
                continue

            for h in hold_days:
                try:
                    sell_cal_idx = buy_date_cal_idx + h
                    
                    if sell_cal_idx >= len(all_trade_cals): continue 
                    
                    sell_date = all_trade_cals[sell_cal_idx]
                    
                    if ts_code not in GLOBAL_KLINE_DATA: continue
                    
                    sell_data_row = GLOBAL_KLINE_DATA[ts_code]
                    
                    # 查找 sell_date 对应的收盘价
                    sell_close_df = sell_data_row[sell_data_row['trade_date'] == sell_date]['close']
                    
                    if sell_close_df.empty: continue
                    
                    # 尝试安全转换卖出价
                    try:
                        sell_close = float(sell_close_df.iloc[0])
                    except (ValueError, TypeError):
                        sell_close = np.nan
                    
                    # 严格检查卖出价
                    if pd.isna(sell_close) or sell_close <= 0:
                        continue 
                        
                    # 统计交易次数 (通过价格检查才算有效交易)
                    results[h]['total'] += 1 

                    # 计算收益
                    ret = (sell_close / buy_close) - 1.0
                    results[h]['returns'].append(ret)
                    if ret > 0:
                        results[h]['wins'] += 1
                except Exception:
                    continue # 忽略任何其他计算错误

        
        if pbar: pbar.progress((i + 1) / len(trade_dates), text=f"回测进度：{i+1} / {len(trade_dates)} 天")

    if pbar: pbar.empty() 
    
    final_results = []
    for h in hold_days:
        r = results[h]
        # 使用 np.nanmean 确保 nan 不影响有效交易的平均值
        avg_ret = np.nanmean(r['returns']) * 100 if r['returns'] else 0.0
        
        # 防止 total 为 0 时计算胜率报错
        win_rate = (r['wins'] / r['total']) * 100 if r['total'] > 0 else 0.0
        
        final_results.append({
            '持股天数': f'{h} 天',
            '平均收益率 (%)': f'{avg_ret:.2f}',
            '胜率 (%)': f'{win_rate:.2f}',
            '总交易次数': r['total']
        })
        
    return pd.DataFrame(final_results)


# ---------------------------
# 实时选股主流程 (保持不变)
# ---------------------------
def live_stock_pick():
    global GLOBAL_KLINE_DATA 
    
    start_date_90 = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=90)).strftime("%Y%m%d")
    get_bulk_daily_data(start_date_90, last_trade)
    
    st.write("正在拉取当日 daily（涨幅榜）作为初筛...")
    daily_all = safe_get(pro.daily, trade_date=last_trade)
    if daily_all.empty:
        st.error("无法获取当日 daily 数据（Tushare 返回空）。请确认 Token 权限。")
        st.stop()

    daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
    st.write(f"当日记录：{len(daily_all)}，取涨幅前 {INITIAL_TOP_N} 作为初筛。")
    pool0 = daily_all.head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

    stock_basic, daily_basic, moneyflow = get_advanced_data(last_trade)
    
    pool_merged = merge_all_info(pool0, stock_basic, daily_basic, moneyflow)

    st.write("对初筛池进行清洗（ST/停牌/价格/一字板/换手/成交额等）...")
    # clean_and_filter 内部不显示进度条
    clean_df = clean_and_filter(pool_merged, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_MARKET_CAP, MAX_MARKET_CAP, VOL_SPIKE_MULT, VOLATILITY_MAX, HIGH_PCT_THRESHOLD, FINAL_POOL)

    if clean_df.empty:
        st.error("清洗后没有候选，建议放宽条件或检查接口权限。")
        st.stop()
    
    st.write(f"清洗后候选数量：{len(clean_df)} （将从中取涨幅前 {FINAL_POOL} 进入评分阶段）")
    
    st.write("为评分池逐票计算指标（本次已优化：从本地缓存读取 K 线数据）...")
    # compute_scores 内部不显示进度条
    fdf = compute_scores(clean_df, last_trade, MIN_MARKET_CAP, MAX_MARKET_CAP, VOL_SPIKE_MULT, VOLATILITY_MAX, HIGH_PCT_THRESHOLD)

    if fdf.empty:
        st.error("评分计算失败或无数据，请检查上面是否有【过滤失败】的警告，并放宽侧边栏参数。")
        st.stop()

    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index = fdf.index + 1

    st.success(f"评分完成：总候选 {len(fdf)} 支，显示 Top {min(TOP_DISPLAY, len(fdf))}。")
    display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','proxy_money','amount','10d_return','macd','diff','dea','k','d','j','rsl','volatility_10']
    
    for c in display_cols:
        if c not in fdf.columns:
            fdf[c] = np.nan

    st.dataframe(fdf[display_cols].head(TOP_DISPLAY), use_container_width=True)

    out_csv = fdf[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")
    
    return fdf


# ---------------------------
# 主流程控制 
# ---------------------------

if st.button('🟢 **运行当日选股**'):
    live_stock_pick()

if st.button('🟠 **启动回测** (N 天前买入, 持有 H 天, 收盘价计算)'):
    if BACKTEST_DAYS <= 0:
        st.error("回测天数必须大于 0。")
        st.stop()
        
    with st.spinner(f'正在获取过去 {BACKTEST_DAYS} 个交易日的数据并回测... (已启用批量加载)'):
        
        today = datetime.strptime(last_trade, "%Y%m%d")
        start_date = (today - timedelta(days=BACKTEST_DAYS * 3)).strftime("%Y%m%d")
        
        all_trade_cals = get_all_trade_cals(start_date, last_trade)
        all_trade_cals.sort()
        
        if len(all_trade_cals) < BACKTEST_DAYS + 1:
            st.error(f"【日历缺失】交易日历不足 {BACKTEST_DAYS} 天，或获取失败。请检查 Token 权限或降低回测天数。")
            st.stop() 
        
        try:
            last_trade_idx = all_trade_cals.index(last_trade)
        except ValueError:
            st.error(f"最近交易日 {last_trade} 不在交易日历中。")
            st.stop()
            
        start_idx = last_trade_idx - BACKTEST_DAYS
        end_idx = last_trade_idx 
        
        if start_idx < 0:
            start_idx = 0
            
        backtest_dates = all_trade_cals[start_idx:end_idx]
        
        st.warning(f"回测日期范围：{backtest_dates[0]} 到 {backtest_dates[-1]} (共 {len(backtest_dates)} 天)")

        if not backtest_dates:
             st.error(f"【内部错误】回测日期列表为空。请检查回测天数设置。")
             st.stop()

        results_df = run_backtest(backtest_dates, HOLD_DAYS, TOP_DISPLAY)
        
        if not results_df.empty:
            st.subheader("📊 历史回测结果 (买入收盘价 / 卖出收盘价)")
            st.dataframe(results_df, use_container_width=True)
            st.success("回测完成！")
        else:
            st.warning("回测未产生有效结果，**极可能是每日选股过滤过于严格**。请检查上方是否有**【过滤失败】**警告，并放宽价格、换手率、波动率等参数。")


# ---------------------------
# 小结与建议（简洁）
# ---------------------------
st.markdown("---")
st.markdown("### 小结与操作提示（简洁）")
st.markdown("""
- **当日选股**：点击 **🟢 运行当日选股**。
- **回测**：点击 **🟠 启动回测**。
- **本次优化**：
    - 移除了回测过程中重复打印的“风险过滤后，剩余 XX 支”提示。
    - 增强了回测中买入价的获取逻辑，以解决“交易次数0”的问题。
- **故障排除**：如果仍出现**【过滤失败】**警告，请重点调整侧边栏中的参数，例如降低**放量倍数阈值**（如改为 1.3）或提高**波动 std 阈值**（如改为 10.0）。
""")
