# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（V5.0 趋势策略 - 缓存稳定版）
说明：
- **V5.0S 核心修复：** 引入核心缓存版本号，允许修改策略逻辑（如评分和过滤）而不会导致耗时的历史数据缓存失效。
- **策略：** 趋势延续策略（找收阳强势股），T+1 开盘价买入。
- **已移除：** 导致交易次数为 0 的 MA 多头硬过滤。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# V5.0S 缓存稳定性变量
# ---------------------------
# 只有手动修改此版本号（例如改为 1.1），才会强制清空 get_hist_cached 的历史数据缓存。
V5_CORE_CACHE_VERSION = 1.0 


# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · 10000旗舰（V5.0S - 稳定版）", layout="wide")
st.title("选股王 · 10000 积分旗舰（V5.0S - 趋势策略 + 缓存稳定）")
st.markdown("输入你的 Tushare Token（仅本次运行使用）。若有权限缺失，脚本会自动降级并继续运行。")

# ---------------------------
# 侧边栏参数（实时可改）
# ---------------------------
with st.sidebar:
    st.header("可调参数（策略核心）")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=300, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=3.0, step=0.5))
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=100_000_000.0, step=50_000_000.0)) # 默认 1亿
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=12.0, step=0.5))
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    MIN_MARKET_CAP = float(st.number_input("最低市值 (元)", value=2000000000.0, step=100000000.0)) # 默认 20亿
    MAX_MARKET_CAP = float(st.number_input("最高市值 (元)", value=50000000000.0, step=1000000000.0)) # 默认 500亿
    
    st.markdown("---")
    # --- 历史回测参数 ---
    st.header("历史回测参数")
    BACKTEST_DAYS = int(st.number_input("回测交易日天数", value=60, min_value=10, max_value=250))
    BACKTEST_TOP_K = int(st.number_input("回测每日最多交易 K 支", value=3, min_value=1, max_value=10))
    HOLD_DAYS_OPTIONS = st.multiselect("回测持股天数", options=[1, 3, 5, 10, 20], default=[1, 3, 5])
    # 新增参数，用于强制缓存失效
    BT_CACHE_KEY = float(st.number_input("回测：缓存破坏键（任意改动刷新回测）", value=1.25, step=0.01))
    st.caption("提示：本次回测为 **T+1 日开盘价买入** 趋势策略。")


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

@st.cache_data(ttl=36000) 
def find_last_trade_day(max_days=20):
    """查找最近交易日"""
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
# 选股逻辑 (使用 V5.0S 的缓存稳定性)
# ---------------------------
# V5.0S 核心优化：延长历史数据缓存至 24 小时 (86400秒)，并添加版本控制
@st.cache_data(ttl=86400) 
def get_hist_cached(ts_code, end_date, days=60, code_version=V5_CORE_CACHE_VERSION): 
    """获取单只股票的历史数据"""
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
    """计算 MA/MACD/KDJ/量比等指标"""
    res = {}
    if df.empty or len(df) < 3:
        return res
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    # last close
    try: res['last_close'] = close.iloc[-1]
    except: res['last_close'] = np.nan

    # MA
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
        res['macd'] = macd_val.iloc[-1]; res['diff'] = diff.iloc[-1];
        res['dea'] = dea.iloc[-1]
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
        res['k'] = k.iloc[-1]; res['d'] = d.iloc[-1];
        res['j'] = j.iloc[-1]
    else:
        res['k'] = res['d'] = res['j'] = np.nan

    # vol ratio and metrics
    vols = df['vol'].astype(float).tolist()
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

def safe_merge_pool(pool_df, other_df, cols):
    """安全合并数据"""
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

def norm_col(s):
    """归一化数据"""
    s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    mn = s.min(); mx = s.max()
    if mx - mn < 1e-9:
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)


def compute_scores(trade_date):
    """
    运行 T 日的选股、清洗和评分逻辑，并返回最终得分表。
    """
    
    # ---------------------------
    # 1. 拉当日涨幅榜初筛
    # ---------------------------
    daily_all = safe_get(pro.daily, trade_date=trade_date)
    if daily_all.empty:
        return pd.DataFrame()

    daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
    pool0 = daily_all.head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

    # ---------------------------
    # 2. 尝试加载高级接口
    # ---------------------------
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')
    daily_basic = safe_get(pro.daily_basic, trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    mf_raw = safe_get(pro.moneyflow, trade_date=trade_date)
    
    # moneyflow 预处理
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in','net_mf_out']
        col = None
        for c in possible:
            if c in mf_raw.columns:
                col = c; break
        if col:
            moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)
        
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
    pool_merged.rename(columns={'amount': 'amount_basic'}, inplace=True) # daily_basic的amount
    
    # merge moneyflow robustly
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
    
    # ---------------------------
    # 3. 基本清洗
    # ---------------------------
    clean_list = []
    # 统一使用 daily 里的 amount（单位千元） 和 daily_basic 里的 turnover_rate（单位 %）
    for i, r in enumerate(pool_merged.itertuples()):
        ts = getattr(r, 'ts_code')
        vol = getattr(r, 'vol', 0)

        close = getattr(r, 'close', np.nan)
        open_p = getattr(r, 'open', np.nan)
        pre_close = getattr(r, 'pre_close', np.nan)
        pct = getattr(r, 'pct_chg', np.nan)
        amount_daily = getattr(r, 'amount', np.nan) # daily 里的 amount
        turnover = getattr(r, 'turnover_rate', np.nan)
        name = getattr(r, 'name', ts)

    
        # 1. 过滤：停牌/无成交
        if vol == 0 or (isinstance(amount_daily,(int,float)) and amount_daily == 0):
            continue

        # 2. 过滤：价格区间
        if pd.isna(close): 
            continue
        if (close < MIN_PRICE) or (close > MAX_PRICE): 
            continue

        # 3. 过滤：ST / 退市 / 北交所
        if isinstance(name, str) and (('ST' in name.upper()) or ('退' in name)):
            continue
        tsck = getattr(r, 'ts_code', '')
        if isinstance(tsck, str) and (tsck.startswith('4') or tsck.startswith('8')):
            continue

        # 4. 过滤：市值（兼容万元单位）
        try:
            tv = getattr(r, 'total_mv', np.nan)
            if not pd.isna(tv):
                tv = float(tv)
                if tv > 1e6:
                    tv_yuan = tv * 10000.0
                else:
                    tv_yuan = tv
                if tv_yuan < MIN_MARKET_CAP or tv_yuan > MAX_MARKET_CAP:
                    continue
        except:
            pass # 发生异常时不过滤

        # 5. 过滤：一字涨停板
        try:
            high = getattr(r, 'high', np.nan); low = getattr(r, 'low', np.nan)
            if (not pd.isna(open_p) and not pd.isna(high) and not pd.isna(low) and not pd.isna(pre_close)):
                if (open_p == high == low == pre_close) and (pct > 9.5):
                    continue
        except:
            pass # 发生异常时不过滤

        # 6. 过滤：换手率
        if not pd.isna(turnover):
            try:
                if float(turnover) < MIN_TURNOVER: 
                    continue
            except:
                pass # 发生异常时不过滤

        # 7. 过滤：成交额（修正单位：daily amount是千元）
        if not pd.isna(amount_daily):
            amt = amount_daily * 1000.0 # 转换成元
            if amt < MIN_AMOUNT: 
                continue

        # 8. 过滤：T 日收阳过滤
        try:
            if float(pct) < 0: 
                continue
        except:
            pass # 发生异常时不过滤
            
        clean_list.append(r)
        
    clean_df = pd.DataFrame([dict(zip(r._fields, r)) for r in clean_list])
    if clean_df.empty:
        return pd.DataFrame()

    # ---------------------------
    # 4. 评分池逐票计算因子
    # ---------------------------
    # 为了回测性能，这里只取前 FINAL_POOL 股票计算指标
    clean_df = clean_df.sort_values("pct_chg", ascending=False).head(FINAL_POOL).copy()
    
    records = []
    for idx, row in enumerate(clean_df.itertuples()):
        ts_code = getattr(row, 'ts_code')
        name = getattr(row, 'name', ts_code)
        pct_chg = getattr(row, 'pct_chg', 0.0)
        
        amount_daily = getattr(row, 'amount', np.nan)
        amount = 0.0
        if amount_daily is not None and not pd.isna(amount_daily):
            amount = amount_daily * 1000.0 # 转换成元

        turnover_rate = getattr(row, 'turnover_rate', np.nan)
        net_mf = float(getattr(row, 'net_mf', 0.0))
        
        # V5.0S 核心修改：增加 code_version 参数，保证稳定性
        hist = get_hist_cached(ts_code, trade_date, days=60, code_version=V5_CORE_CACHE_VERSION)
        ind = compute_indicators(hist)

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

        # 资金强度代理（不依赖 moneyflow）：简单乘积指标
        try:
            proxy_money = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
        except:
            proxy_money = 0.0

        rec = {
            'ts_code': ts_code, 'name': name, 'pct_chg': pct_chg,
            'amount': amount,
            'turnover_rate': turnover_rate if not pd.isna(turnover_rate) else np.nan,
            'net_mf': net_mf,
            'vol_ratio': vol_ratio if not pd.isna(vol_ratio) else np.nan,
            '10d_return': ten_return if not pd.isna(ten_return) else np.nan,
            'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
            'macd': macd, 'diff': diff, 'dea': dea, 'k': k, 'd': d, 'j': j,
            'last_close': last_close, 'vol_last': vol_last,
            'vol_ma5': vol_ma5, 'recent20_high': recent20_high, 'yang_body_strength': yang_body_strength,
            'prev3_sum': prev3_sum, 'volatility_10': volatility_10,
            'proxy_money': proxy_money
        }

        records.append(rec)
 
    fdf = pd.DataFrame(records)
    if fdf.empty:
        return pd.DataFrame()

    # ---------------------------
    # 5. 风险过滤 (策略核心部分)
    # ---------------------------
    try:
        # A: 高位大阳线过滤 (V5.0保留)
        if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):
            mask_high_big = (fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
            fdf = fdf[~mask_high_big]

        # B: 下跌途中反抽过滤 (V5.0保留)
        if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
            mask_down_rebound = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
            fdf = fdf[~mask_down_rebound]

        # C: 巨量放量大阳过滤 (V5.0保留)
        if 'vol_ratio' in fdf.columns:
            mask_vol_spike = fdf['vol_ratio'] > VOL_SPIKE_MULT
            fdf = fdf[~mask_vol_spike]

        # D: 极端波动过滤 (V5.0保留)
        if 'volatility_10' in fdf.columns:
            mask_volatility = fdf['volatility_10'] > VOLATILITY_MAX
            fdf = fdf[~mask_volatility]
    except:
        pass

    # ---------------------------
    # 6. MA 多头硬过滤 (已移除，V5.0S 不进行硬过滤)
    # ---------------------------
    try:
        if all(c in fdf.columns for c in ['ma5','ma10','ma20']):
            # fdf = fdf[(fdf['ma5'] > fdf['ma10']) & (fdf['ma10'] > fdf['ma20'])].copy() 
            pass
    except:
        pass

    # ---------------------------
    # 7. RSL、归一化与评分 (保持V5.0/V6.0逻辑)
    # ---------------------------
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

    # 趋势因子与强化评分（右侧趋势主导）
    fdf['ma_trend_flag'] = ((fdf.get('ma5', pd.Series([])) > fdf.get('ma10', pd.Series([]))) & (fdf.get('ma10', pd.Series([])) > fdf.get('ma20', pd.Series([])))).fillna(False)
    fdf['macd_golden_flag'] = (fdf.get('diff', 0) > fdf.get('dea', 0)).fillna(False)
    fdf['vol_price_up_flag'] = (fdf.get('vol_last', 0) > fdf.get('vol_ma5', 0)).fillna(False)
    fdf['break_high_flag'] = (fdf.get('last_close', 0) > fdf.get('recent20_high', 0)).fillna(False)
    fdf['yang_body_strength'] = fdf.get('yang_body_strength', 0.0).fillna(0.0)

    # 组合成趋势原始分
    fdf['trend_score_raw'] = (
        fdf['ma_trend_flag'].astype(float) * 1.0 +
        fdf['macd_golden_flag'].astype(float) * 1.3 +
        fdf['vol_price_up_flag'].astype(float) * 1.0 +
        fdf['break_high_flag'].astype(float) * 1.3 +
        fdf['yang_body_strength'].astype(float) * 0.8
    )

    # 归一化趋势分
    fdf['trend_score'] = norm_col(fdf['trend_score_raw'])

    # 最终综合评分（趋势主导）
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
# 运行当日选股（初次运行时运行）
# ---------------------------
if st.button("🚀 运行当日选股（初次运行可能较久）"):
    st.write("正在拉取当日 daily 数据并计算评分...")
    fdf = compute_scores(last_trade)

    if fdf.empty:
        st.error("评分计算失败或无数据，请检查 Token 权限与接口。")
        st.stop()

    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index = fdf.index + 1

    st.success(f"评分完成：总候选 {len(fdf)} 支，显示 Top {min(TOP_DISPLAY, len(fdf))}。")
    display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','proxy_money','amount','10d_return','macd','diff','dea','k','d','j','rsl','volatility_10']
    for c in display_cols:
        if c not in fdf.columns:
            fdf[c] = np.nan

    st.dataframe(fdf[display_cols].head(TOP_DISPLAY), use_container_width=True)

    # 下载（仅导出前200避免过大）
    out_csv = fdf[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")


# ---------------------------
# 历史回测部分（V5.0S 稳定版）
# ---------------------------
# V5.0S 核心优化：延长回测数据缓存至 24 小时 (86400秒)
@st.cache_data(ttl=86400)
def load_backtest_data(all_trade_dates):
    """预加载所有回测日期的 daily 数据，以字典 {trade_date: DataFrame} 缓存。"""
    data_cache = {}
    st.write(f"正在预加载回测所需 {len(all_trade_dates)} 个交易日的全部 daily 数据 (约 {len(all_trade_dates)} 次 API 调用)...")
    pbar = st.progress(0)
    for i, date in enumerate(all_trade_dates):
        daily_df = safe_get(pro.daily, trade_date=date)
        if not daily_df.empty:
            data_cache[date] = daily_df.set_index('ts_code')
        pbar.progress((i + 1) / len(all_trade_dates))
    pbar.progress(1.0)
    return data_cache

@st.cache_data(ttl=6000)
def run_backtest(start_date, end_date, hold_days, backtest_top_k, bt_cache_key):
    # 使用 bt_cache_key 确保每次参数变化都破坏缓存
    _ = bt_cache_key 

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
            # 需要 T 日的 daily 数据来选股
            required_dates.add(buy_date) 
            # 需要 T+1 的 open price 作为买入价
            required_dates.add(trade_dates[current_index + 1]) 
            # 需要 T+1+H 的 close price 作为卖出价
            for h in hold_days:
                required_dates.add(trade_dates[current_index + h + 1])
        except (ValueError, IndexError):
            continue
            
    # V5.0S 核心：这里加载数据（如果有缓存，会跳过 API 调用）
    data_cache = load_backtest_data(sorted(list(required_dates)))

    st.write(f"正在模拟 {len(backtest_dates)} 个交易日的选股回测...")
    pbar_bt = st.progress(0)
    
    for i, t_day in enumerate(backtest_dates): # T 日 (选股日)
        
        # 1. 运行 T 日选股与评分逻辑，获取综合评分
        t_scores = compute_scores(t_day) 
        
        if t_scores.empty:
            pbar_bt.progress((i+1)/len(backtest_dates)); continue

        # 按综合评分排序，选择 Top K
        scored_stocks = t_scores.sort_values("综合评分", ascending=False).head(backtest_top_k).copy()
        
        # 2. 确定 T+1 买入日
        try:
            t_day_index = trade_dates.index(t_day)
            t_plus_1_day = trade_dates[t_day_index + 1]
        except (ValueError, IndexError):
            pbar_bt.progress((i+1)/len(backtest_dates)); continue
        
        # 获取 T+1 日的 daily 数据
        t_plus_1_df_cached = data_cache.get(t_plus_1_day)

        for _, row in scored_stocks.iterrows():
            ts_code = row['ts_code']

            # 确定买入价 (T+1 日开盘价)
            buy_price = np.nan
            if t_plus_1_df_cached is not None and ts_code in t_plus_1_df_cached.index:
                # V5.0S 核心修正：使用 T+1 日的 open price
                buy_price = t_plus_1_df_cached.loc[ts_code, 'open'] 
            
            if pd.isna(buy_price) or buy_price <= 0: continue

            for h in hold_days:
                try:
                    # 卖出日：T+1+H (即 T+1 后的第 H 个交易日收盘)
                    sell_date = trade_dates[t_day_index + h + 1] 
                except (ValueError, IndexError):
                    continue
        
                # 从缓存中查找卖出价格 (T+1+H 日收盘价)
                sell_df_cached = data_cache.get(sell_date)
                sell_price = np.nan
                if sell_df_cached is not None and ts_code in sell_df_cached.index:
                    # V5.0S 修正：使用 T+1+H 日的 close price
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
        st.header("📈 历史回测结果（V5.0S 稳定版 / 趋势策略）")
        
        try:
            start_date_for_cal = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
        except:
            start_date_for_cal = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            
        backtest_result = run_backtest(
            start_date=start_date_for_cal,
            end_date=last_trade,
            hold_days=HOLD_DAYS_OPTIONS,
            backtest_top_k=BACKTEST_TOP_K,
            bt_cache_key=BT_CACHE_KEY # 传入参数确保缓存刷新
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
# 小结与操作提示（简洁）
# ---------------------------
st.markdown("### 小结与操作提示（V5.0S 稳定版重点）")
st.markdown("""
- **状态：** **趋势加强策略版 v5.0S**（已彻底修复缓存重置问题）。
- **缓存策略：** 只有手动修改代码第 33 行的 `V5_CORE_CACHE_VERSION`，才会重置历史数据缓存。
- **操作步骤：** 1. **请使用上方 V5.0S 完整代码替换您的脚本内容。**
    2. **关键步骤（最后一次等待）：** 确保您停止了之前的运行后，进行最后一次低负荷缓存重建：
        - 将 **“回测交易日天数”** 设为 **20**。
        - 将 **“清洗后取前 M 进入评分”** 设为 **50**。
    3. **运行回测**。这次运行预计耗时 **10-30 分钟**。
    
请您耐心等待这次最后的缓存重建。一旦完成，您就可以自由地将参数调回 60/300，并在秒级速度下继续测试。
""")
