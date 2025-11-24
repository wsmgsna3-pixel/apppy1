# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（终极修复版 v4.6 - 右侧启动/趋势强化策略）
说明：
- 整合了 BC 混合增强策略。
- 修复了性能优化后 `run_backtest` 函数中因缺少 `turnover_rate` 导致的 KeyError 错误。
- **v4.6 核心策略调整：**
    - **回测买入逻辑** 调整为“右侧启动”：寻找当日涨幅较高，且高流动性的股票。
    - **新回测条件：** 1. 股票当日涨幅必须在 4.0% 到 9.5% 之间（避免盘整和极端激进）。
        2. 在此池内，按 `pct_chg` (涨幅) 降序选择 Top K（确保捕捉动量）。
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
st.set_page_config(page_title="选股王 · 10000旗舰（右侧启动 v4.6）", layout="wide")
st.title("选股王 · 10000 积分旗舰（终极修复版 v4.6 - 右侧启动策略）")
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
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=100_000_000.0, step=50_000_000.0)) # 默认 1亿
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=12.0, step=0.5))
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    MIN_MARKET_CAP = float(st.number_input("最低市值 (元)", value=2000000000.0, step=100000000.0)) # 默认 20亿
    MAX_MARKET_CAP = float(st.number_input("最高市值 (元)", value=50000000000.0, step=1000000000.0)) # 默认 500亿
    st.markdown("---")
    # --- 新增回测参数 ---
    st.header("历史回测参数")
    BACKTEST_DAYS = int(st.number_input("回测交易日天数", value=60, min_value=10, max_value=250))
    BACKTEST_TOP_K = int(st.number_input("回测每日最多交易 K 支", value=3, min_value=1, max_value=10))
    HOLD_DAYS_OPTIONS = st.multiselect("回测持股天数", options=[1, 3, 5, 10, 20], default=[1, 3, 5])
    # 新增参数，用于强制缓存失效 (右侧策略使用涨幅上限作为缓存键)
    BT_MAX_PCT_FOR_CACHE = float(st.number_input("回测：最高涨幅 (缓存破坏键)", value=9.5, step=0.5))
    st.caption("提示：**本次回测使用右侧启动/趋势强化策略 (4.0% < 涨幅 < 9.5%)。**")

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

@st.cache_data(ttl=36000) # 延长历史数据缓存至 10 小时
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
# 优化点 1: 将选股逻辑包裹在按钮中
# ---------------------------
if st.button("🚀 运行当日选股（初次运行可能较久）"):

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
    st.write("尝试加载 stock_basic / daily_basic / moneyflow 等高级接口（若权限允许）...")
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')
    daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)

    # moneyflow 预处理
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in','net_mf_out']
        col = None
        for c in possible:
            if c in mf_raw.columns:
                col = c;
                break
        if col is None:
            numeric_cols = [c for c in mf_raw.columns if c != 'ts_code' and pd.api.types.is_numeric_dtype(mf_raw[c])]
            col = numeric_cols[0] if numeric_cols else None
        if col:
            moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)
        else:
            moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    else:
        moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
        st.warning("moneyflow 未获取到，将把主力流向因子置为 0（若有权限请确认 Token/积分）。")

    # ---------------------------
    # 合并基本信息（safe）
    # ---------------------------
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
            pool0['name'] = pool0['ts_code'];
            pool0['industry'] = ''
    else:
        pool0['name'] = pool0['ts_code'];
        pool0['industry'] = ''

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
    # 基本清洗（ST / 停牌 / 价格区间 / 一字板 / 换手 / 成交额 / 市值）
    # ---------------------------
    st.write("对初筛池进行清洗（ST/停牌/价格/一字板/换手/成交额等）...")
    clean_list = []
    pbar = st.progress(0)
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
            pbar.progress((i+1)/len(pool_merged))
            continue

        # 2. 过滤：价格区间
        if pd.isna(close): 
            pbar.progress((i+1)/len(pool_merged))
            continue
        if (close < MIN_PRICE) or (close > MAX_PRICE): 
            pbar.progress((i+1)/len(pool_merged))
            continue

        # 3. 过滤：ST / 退市 / 北交所
        if isinstance(name, str) and (('ST' in name.upper()) or ('退' in name)):
            pbar.progress((i+1)/len(pool_merged))
            continue
        tsck = getattr(r, 'ts_code', '')
        if isinstance(tsck, str) and (tsck.startswith('4') or tsck.startswith('8')):
            pbar.progress((i+1)/len(pool_merged))
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
                    pbar.progress((i+1)/len(pool_merged))
                    continue
        except:
            pass # 发生异常时不过滤

        # 5. 过滤：一字涨停板
        try:
            high = getattr(r, 'high', np.nan);
            low = getattr(r, 'low', np.nan)
            if (not pd.isna(open_p) and not pd.isna(high) and not pd.isna(low) and not pd.isna(pre_close)):
                if (open_p == high == low == pre_close) and (pct > 9.5):
                    pbar.progress((i+1)/len(pool_merged))
                    continue
        except:
            pass # 发生异常时不过滤

        # 6. 过滤：换手率
        if not pd.isna(turnover):
            try:
                if float(turnover) < MIN_TURNOVER: 
                    pbar.progress((i+1)/len(pool_merged))
                    continue
            except:
                pass # 发生异常时不过滤

        # 7. 过滤：成交额（修正单位：daily amount是千元）
        if not pd.isna(amount_daily):
            amt = amount_daily * 1000.0 # 转换成元
            if amt < MIN_AMOUNT: 
                pbar.progress((i+1)/len(pool_merged))
                continue

        # 8. 过滤：剔除当日下跌的股（只保留当日上涨的）
        try:
            if float(pct) < 0: 
                pbar.progress((i+1)/len(pool_merged))
                continue
        except:
            pass # 发生异常时不过滤
            
        clean_list.append(r)
        pbar.progress((i+1)/len(pool_merged))

    pbar.progress(1.0)
    clean_df = pd.DataFrame([dict(zip(r._fields, r)) for r in clean_list])
    st.write(f"清洗后候选数量：{len(clean_df)} （将从中取涨幅前 {FINAL_POOL} 进入评分阶段）")
    if len(clean_df) == 0:
        st.error("清洗后没有候选，建议放宽条件或检查接口权限。")
        st.stop()

    # ---------------------------
    # 辅助：获取单只历史（用于量比/10日收益等）
    # ---------------------------
    @st.cache_data(ttl=36000) # 延长历史数据缓存至 10 小时
    def get_hist_cached(ts_code, end_date, days=60):
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
            res['macd'] = macd_val.iloc[-1];
            res['diff'] = diff.iloc[-1];
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
            res['k'] = k.iloc[-1];
            res['d'] = d.iloc[-1];
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

    # ---------------------------
    # 评分池逐票计算因子（缓存 get_hist）
    # ---------------------------
    st.write("为评分池逐票拉历史并计算指标（此步骤调用历史接口，已缓存）...")
    st.warning(f"⚠️ **耗时警告：** 当前有 {len(clean_df)} 支股票需要计算指标。如果太慢，请调整侧边栏 **'清洗后取前 M'** 参数。")
    records = []
    pbar2 = st.progress(0)
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

        
        hist = get_hist_cached(ts_code, last_trade, days=60)
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
        pbar2.progress((idx+1)/len(clean_df))

    pbar2.progress(1.0)
    fdf = pd.DataFrame(records)
    if fdf.empty:
        st.error("评分计算失败或无数据，请检查 Token 权限与接口。")
        st.stop()

    # ---------------------------
    # 风险过滤（放在评分前以节省历史调用）
    # ---------------------------
    st.write("执行风险过滤：下跌途中大阳 / 巨量冲高 / 高位大阳 / 极端波动 ...")
    try:
        before_cnt = len(fdf)
        # A: 高位大阳线 -> last_close > ma20*1.10 且 pct_chg > HIGH_PCT_THRESHOLD
        if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):
            mask_high_big = (fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
            fdf = fdf[~mask_high_big]

        # B: 下跌途中反抽 -> prev3_sum < 0 且 pct_chg > HIGH_PCT_THRESHOLD
        if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
            mask_down_rebound = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
            fdf = fdf[~mask_down_rebound]

        # C: 巨量放量大阳 -> vol_ratio > VOL_SPIKE_MULT
        if 'vol_ratio' in fdf.columns:
            mask_vol_spike = fdf['vol_ratio'] > VOL_SPIKE_MULT
            fdf = fdf[~mask_vol_spike]

        # D: 极端波动 -> volatility_10 > VOLATILITY_MAX
        if 'volatility_10' in fdf.columns:
            mask_volatility = fdf['volatility_10'] > VOLATILITY_MAX
            fdf = fdf[~mask_volatility]

        after_cnt = len(fdf)
        st.write(f"风险过滤：{before_cnt} -> {after_cnt}（若过严请在侧边栏调整阈值）")
    except Exception as e:
        st.warning(f"风险过滤模块异常，跳过过滤。错误：{e}")

    # ---------------------------
    # MA 多头硬过滤（必须满足 MA5 > MA10 > MA20）
    # ---------------------------
    try:
        if all(c in fdf.columns for c in ['ma5','ma10','ma20']):
            before_ma = len(fdf)
            fdf = fdf[(fdf['ma5'] > fdf['ma10']) & (fdf['ma10'] > fdf['ma20'])].copy() 
            after_ma = len(fdf)
            st.write(f"MA 多头过滤：{before_ma} -> {after_ma}（保留 MA5>MA10>MA20，强化趋势）")
    except Exception as e:
        st.warning(f"MA 过滤异常，跳过。错误：{e}")

    # ---------------------------
    # RSL（相对强弱）：基于池内 10d_return 的相对表现
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

    # ---------------------------
    # 子指标归一化（稳健）
    # ---------------------------
    def norm_col(s):
        s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        mn = s.min();
        mx = s.max()
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

    # ---------------------------
    # 趋势因子与强化评分（右侧趋势主导）
    # ---------------------------
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

    # ---------------------------
    # 最终综合评分（趋势主导）
    # ---------------------------
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

    # ---------------------------
    # 最终排序与展示
    # ---------------------------
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
# 历史回测部分（数据性能优化与逻辑强化）
# ---------------------------
@st.cache_data(ttl=3600)
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
def run_backtest(start_date, end_date, hold_days, backtest_top_k, bt_max_pct_for_cache):
    # 使用 bt_max_pct_for_cache 确保每次参数变化都破坏缓存
    _ = bt_max_pct_for_cache # 假装使用这个参数，让它进入缓存哈希

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
                required_dates.add(trade_dates[current_index + h])
        except (ValueError, IndexError):
            continue
            
    data_cache = load_backtest_data(sorted(list(required_dates)))

    st.write(f"正在模拟 {len(backtest_dates)} 个交易日的选股回测...")
    pbar_bt = st.progress(0)
    
    for i, buy_date in enumerate(backtest_dates):
        daily_df_cached = data_cache.get(buy_date)
        
        if daily_df_cached is None or daily_df_cached.empty:
            pbar_bt.progress((i+1)/len(backtest_dates));
            continue

        daily_df = daily_df_cached.copy().reset_index()
        
        # 1. 应用基本过滤 (价格/成交额/动量/停牌/一字板)
        
        # amount 字段在 daily 接口中，单位是千元，我们要求的是元。
        BACKTEST_MIN_AMOUNT_PROXY = MIN_AMOUNT * 2.0 
        
        daily_df['amount_yuan'] = daily_df['amount'].fillna(0) * 1000.0 # 转换成元
        
        # 过滤：V4.6 右侧启动策略：寻找当日涨幅较高，且高成交额的股票
        daily_df = daily_df[
            (daily_df['close'] >= MIN_PRICE) & 
            (daily_df['close'] <= MAX_PRICE) &
            (daily_df['amount_yuan'] >= BACKTEST_MIN_AMOUNT_PROXY) & 
            (daily_df['pct_chg'] >= 4.0) & # **策略调整：当日涨幅必须达到 4.0% 以上 (右侧启动)**
            (daily_df['pct_chg'] <= BT_MAX_PCT_FOR_CACHE) & # **使用缓存破坏键的参数作为上限 (9.5%)**
            (daily_df['vol'] > 0) & 
            (daily_df['amount_yuan'] > 0)
        ].copy()
        
        # 过滤一字涨停板 (防止 BT_MAX_PCT_FOR_CACHE 被设置为 10.0 时漏网)
        daily_df['is_zt'] = (daily_df['open'] == daily_df['high']) & (daily_df['pct_chg'] > 9.5)
        daily_df = daily_df[~daily_df['is_zt']].copy()
        
        # 2. 模拟评分：v4.6 选股逻辑改为按【涨幅】排序 (右侧启动)
        scored_stocks = daily_df.sort_values("pct_chg", ascending=False).head(backtest_top_k).copy()
        
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
        
                # 从缓存中查找卖出价格 (O(1) 查找)
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
if st.checkbox("✅ 运行历史回测", value=False):
    if not HOLD_DAYS_OPTIONS:
        st.warning("请至少选择一个回测持股天数。")
    else:
        st.header("📈 历史回测结果（买入收盘价 / 卖出收盘价）")
        
        try:
            start_date_for_cal = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
        except:
            start_date_for_cal = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            
        backtest_result = run_backtest(
            start_date=start_date_for_cal,
            end_date=last_trade,
            hold_days=HOLD_DAYS_OPTIONS,
            backtest_top_k=BACKTEST_TOP_K,
            bt_max_pct_for_cache=BT_MAX_PCT_FOR_CACHE # 传入参数确保缓存刷新
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
st.markdown("### 小结与操作提示（简洁）")
st.markdown("""
- **状态：** **右侧启动/趋势强化策略版 v4.6**。
- **改动：** **回测买入策略**已调整为“右侧启动”：现在只选择当日涨幅在 **4.0% 到 9.5%** 之间的股票，并按 **涨幅** 排序。这旨在捕捉刚刚启动的趋势股。
- **操作步骤：**
    1. **使用上述完整代码替换您现有脚本的全部内容。**
    2. **更改一个回测参数** (如：侧边栏“回测交易日天数”或 **“回测：最高涨幅 (缓存破坏键)”** 参数)，以确保强制刷新缓存。
    3. **勾选 “✅ 运行历史回测”**。

这次我们专注于**高质量的右侧启动**，希望能在趋势强劲的股票中获得持续的收益。
""")
