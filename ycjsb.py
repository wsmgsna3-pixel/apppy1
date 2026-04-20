# -*- coding: utf-8 -*-
"""
选股王 · V35.1 终极全量验证版
------------------------------------------------
修复记录:
1. 恢复逐日下载逻辑，解决 Tushare 单次 5000 条限制导致的均线计算失败。
2. 增加 Buy_Triggered 标签，分别处理仪表盘和 CSV。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import os
import pickle

warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_INDUSTRY = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V35.1 修复版", layout="wide")
st.title("选股王 V35.1：抢跑修复版 (含未触发股对照记录)")

# ---------------------------
# 基础 API 函数
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: 
        return pd.DataFrame(columns=['ts_code']) 
   
    func = getattr(pro, func_name) 
    try:
        for _ in range(3):
            try:
                if kwargs.get('is_index'):
                    df = pro.index_daily(**kwargs)
                else:
                    df = func(**kwargs)
                
                if df is not None and not df.empty:
                    return df
                time.sleep(0.3)
            except:
                time.sleep(1)
                continue
        return pd.DataFrame(columns=['ts_code']) 
    except Exception as e:
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    lookback_days = max(num_days * 3, 365) 
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    
    if cal.empty or 'cal_date' not in cal.columns:
        return []
        
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()

@st.cache_data(ttl=3600*24*7) 
def load_industry_mapping():
    global pro
    if pro is None: return {}
    try:
        sw_indices = pro.index_classify(level='L1', src='SW2021')
        if sw_indices.empty: return {}
        index_codes = sw_indices['index_code'].tolist()
        all_members = []
        load_bar = st.progress(0, text="正在遍历加载行业数据...")
        for i, idx_code in enumerate(index_codes):
            df = pro.index_member(index_code=idx_code, is_new='Y')
            if not df.empty: all_members.append(df)
            time.sleep(0.02) 
            load_bar.progress((i + 1) / len(index_codes), text=f"加载行业数据: {idx_code}")
        load_bar.empty()
        if not all_members: return {}
        full_df = pd.concat(all_members)
        full_df = full_df.drop_duplicates(subset=['con_code'])
        return dict(zip(full_df['con_code'], full_df['index_code']))
    except Exception as e:
        return {}

# ---------------------------
# 数据获取核心 (本地缓存版)
# ---------------------------
CACHE_FILE_NAME = "market_data_cache_v35.pkl"

def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_INDUSTRY
    if not trade_days_list: return False
    
    with st.spinner("正在同步全市场行业数据..."):
        GLOBAL_STOCK_INDUSTRY = load_industry_mapping()

    if use_cache and os.path.exists(CACHE_FILE_NAME):
        st.success(f"⚡ 发现本地行情缓存 ({CACHE_FILE_NAME})，正在极速加载...")
        try:
            with open(CACHE_FILE_NAME, 'rb') as f:
                cache_data = pickle.load(f)
            
            cached_dates = set(cache_data.get('trade_dates', []))
            needed_dates = set(trade_days_list)
            
            if needed_dates.issubset(cached_dates):
                GLOBAL_ADJ_FACTOR = cache_data['adj']
                GLOBAL_DAILY_RAW = cache_data['daily']
                GLOBAL_QFQ_BASE_FACTORS = cache_data['qfq_base']
                st.info("✅ 缓存数据完美覆盖，无需重新下载！")
                return True
            else:
                st.warning("⚠️ 缓存数据与当前分析日期不匹配，需要重新下载全量数据。")
        except Exception as e:
            st.error("缓存读取失败，将重新下载。")

    st.info(f"正在从 Tushare 逐日下载 {len(trade_days_list)} 天的全量行情数据 (需要几分钟，请耐心等待)...")
    
    # 【修复核心】：恢复原来的逐日下载循环，绕过 5000 条限制！
    adj_list = []
    daily_list = []
    bar = st.progress(0, text="逐日下载数据中...")
    
    for i, d in enumerate(trade_days_list):
        adj_list.append(safe_get('adj_factor', trade_date=d))
        daily_list.append(safe_get('daily', trade_date=d))
        bar.progress((i + 1) / len(trade_days_list), text=f"下载进度: {d} ({i+1}/{len(trade_days_list)})")
        
    bar.empty()
    
    GLOBAL_ADJ_FACTOR = pd.concat(adj_list, ignore_index=True) if adj_list else pd.DataFrame()
    GLOBAL_DAILY_RAW = pd.concat(daily_list, ignore_index=True) if daily_list else pd.DataFrame()
        
    if GLOBAL_ADJ_FACTOR.empty or GLOBAL_DAILY_RAW.empty:
        st.error("下载失败，请检查网络或 Token 权限。")
        return False

    with st.spinner("⚙️ 预计算最新复权基准参数..."):
        latest_adj = GLOBAL_ADJ_FACTOR.sort_values('trade_date').drop_duplicates('ts_code', keep='last')
        GLOBAL_QFQ_BASE_FACTORS = dict(zip(latest_adj['ts_code'], latest_adj['adj_factor']))

    with st.spinner("💾 正在将行情数据序列化保存到本地硬盘..."):
        try:
            cache_to_save = {
                'trade_dates': trade_days_list,
                'adj': GLOBAL_ADJ_FACTOR,
                'daily': GLOBAL_DAILY_RAW,
                'qfq_base': GLOBAL_QFQ_BASE_FACTORS
            }
            with open(CACHE_FILE_NAME, 'wb') as f:
                pickle.dump(cache_to_save, f)
            st.success("✅ 数据已持久化缓存，下次分析将秒开！")
        except Exception as e:
            st.warning("缓存写入失败，但不影响本次运行。")

    return True


def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    
    if GLOBAL_ADJ_FACTOR.empty or GLOBAL_DAILY_RAW.empty:
        return pd.DataFrame()

    adj_mask = (GLOBAL_ADJ_FACTOR['ts_code'] == ts_code) & \
               (GLOBAL_ADJ_FACTOR['trade_date'] >= start_date) & \
               (GLOBAL_ADJ_FACTOR['trade_date'] <= end_date)
    adj_df = GLOBAL_ADJ_FACTOR[adj_mask]
    
    daily_mask = (GLOBAL_DAILY_RAW['ts_code'] == ts_code) & \
                 (GLOBAL_DAILY_RAW['trade_date'] >= start_date) & \
                 (GLOBAL_DAILY_RAW['trade_date'] <= end_date)
    daily_df = GLOBAL_DAILY_RAW[daily_mask]
    
    if daily_df.empty or adj_df.empty:
        return pd.DataFrame()

    df = pd.merge(daily_df, adj_df[['trade_date', 'adj_factor']], on='trade_date', how='inner')
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    latest_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, 1.0)
    
    df['adj_ratio'] = df['adj_factor'] / latest_factor
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] * df['adj_ratio']
        
    return df

def get_future_prices(ts_code, selection_date, d0_qfq_close, days_ahead=[1, 3, 5]):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    
    if hist.empty or len(hist) < 1: return results
    
    hist['open'] = pd.to_numeric(hist['open'], errors='coerce')
    hist['high'] = pd.to_numeric(hist['high'], errors='coerce')
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    
    d1_data = hist.iloc[0]
    next_open = d1_data['open']
    next_high = d1_data['high']
    
    # 判断是否符合“高开且冲高1.5%”的原买入纪律
    triggered = 'Yes'
    target_buy_price = next_open * 1.015
    
    if next_open <= d0_qfq_close or next_high < target_buy_price:
        triggered = 'No'
        # 不符合条件的，仅记录用于分析低开反包，按次日开盘价基准算收益
        target_buy_price = next_open 
        
    for n in days_ahead:
        col = f'Return_D{n}'
        if len(hist) >= n:
            sell_price = hist.iloc[n-1]['close']
            results[col] = (sell_price - target_buy_price) / target_buy_price * 100
        else:
            results[col] = np.nan
            
    results['Buy_Triggered'] = triggered
    return results

def calculate_rsi(series, period=12):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date, end_date)
    
    if df.empty or len(df) < 60: return None
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    
    df['rsi_12'] = calculate_rsi(df['close'], 12)
    
    last = df.iloc[-1]
    return {
        'macd_val': last['macd'],
        'ma20': last['ma20'],
        'ma60': last['ma60'],
        'rsi_12': last['rsi_12'],
        'last_close': last['close'],
        'last_high': last['high'],
        'last_low': last['low']
    }

def get_market_state(trade_date):
    end = trade_date
    start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
    index_df = safe_get('daily', ts_code='000001.SH', start_date=start, end_date=end, is_index=True)
    if index_df.empty or len(index_df) < 20: return 'Strong'
    index_df = index_df.sort_values('trade_date').reset_index(drop=True)
    ma20 = index_df['close'].rolling(20).mean().iloc[-1]
    return 'Strong' if index_df['close'].iloc[-1] > ma20 else 'Weak'

def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS, RSI_LIMIT, CHIP_MIN_WIN_RATE, SECTOR_THRESHOLD, MIN_MV, MAX_MV, MAX_PREV_PCT, MIN_PRICE):
    global GLOBAL_STOCK_INDUSTRY
    
    market_state = get_market_state(last_trade)
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失 {last_trade}"

    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    if stock_basic.empty or 'name' not in stock_basic.columns:
        stock_basic = safe_get('stock_basic', list_status='L')
    
    chip_dict = {}
    try:
        chip_df = safe_get('cyq_perf', trade_date=last_trade)
        if not chip_df.empty:
            chip_dict = dict(zip(chip_df['ts_code'], chip_df['winner_rate']))
    except: pass 
    
    strong_industry_codes = set()
    try:
        sw_df = safe_get('sw_daily', trade_date=last_trade)
        if not sw_df.empty:
            strong_sw = sw_df[sw_df['pct_chg'] >= SECTOR_THRESHOLD]
            strong_industry_codes = set(strong_sw['index_code'].tolist())
    except: pass 
        
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    if 'name' not in df.columns: df['name'] = ''

    daily_basic = safe_get('daily_basic', trade_date=last_trade)
    if not daily_basic.empty:
        needed_cols = ['ts_code','turnover_rate','circ_mv','amount']
        existing_cols = [c for c in needed_cols if c in daily_basic.columns]
        df = df.merge(daily_basic[existing_cols], on='ts_code', how='left')
    
    mf_raw = safe_get('moneyflow', trade_date=last_trade)
    if not mf_raw.empty:
        mf = mf_raw[['ts_code','net_mf_amount']].rename(columns={'net_mf_amount':'net_mf'})
        df = df.merge(mf, on='ts_code', how='left')
    else:
        df['net_mf'] = 0 
    
    for col in ['net_mf', 'turnover_rate', 'circ_mv', 'amount']:
        if col not in df.columns: df[col] = 0
    df['net_mf'] = df['net_mf'].fillna(0)
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')] # 排除北交所
    
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= 2000.0)]
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    df = df[df['turnover_rate'] <= MAX_TURNOVER_RATE] 

    if len(df) == 0: return pd.DataFrame(), "过滤后无标的"

    candidates = df.sort_values('pct_chg', ascending=False).head(FINAL_POOL)
    records = []
    
    for row in candidates.itertuples():
        if GLOBAL_STOCK_INDUSTRY and strong_industry_codes:
            ind_code = GLOBAL_STOCK_INDUSTRY.get(row.ts_code)
            if ind_code and (ind_code not in strong_industry_codes): continue
        
        if row.pct_chg > MAX_PREV_PCT: continue

        ind = compute_indicators(row.ts_code, last_trade)
        if not ind: continue
        d0_close = ind['last_close']
        d0_rsi = ind.get('rsi_12', 50)
        
        if d0_rsi < 50: continue 
        
        if market_state == 'Weak':
            if d0_rsi > RSI_LIMIT: continue
            if d0_close < ind['ma20']: continue 
            
        if d0_close < ind['ma60']: continue
        
        upper_shadow = (ind['last_high'] - d0_close) / d0_close * 100
        if upper_shadow > MAX_UPPER_SHADOW: continue
        range_len = ind['last_high'] - ind['last_low']
        if range_len > 0:
            body_pos = (d0_close - ind['last_low']) / range_len
            if body_pos < MIN_BODY_POS: continue

        win_rate = chip_dict.get(row.ts_code, 50) 
        if win_rate < CHIP_MIN_WIN_RATE: continue

        future = get_future_prices(row.ts_code, last_trade, d0_close)
        records.append({
            'ts_code': row.ts_code, 'name': row.name, 'Close': row.close, 'Pct_Chg': row.pct_chg,
            'rsi': d0_rsi, 'winner_rate': win_rate, 
            'macd': ind['macd_val'], 'net_mf': row.net_mf,
            'Buy_Triggered': future.get('Buy_Triggered', 'No'),
            'Return_D1 (%)': future.get('Return_D1', np.nan),
            'Return_D3 (%)': future.get('Return_D3', np.nan),
            'Return_D5 (%)': future.get('Return_D5', np.nan),
            'market_state': market_state,
            'Sector_Boost': 'Yes' if GLOBAL_STOCK_INDUSTRY else 'N/A'
        })
            
    if not records: return pd.DataFrame(), "深度筛选后无标的"
    fdf = pd.DataFrame(records)
    
    def dynamic_score(r):
        base_score = r['macd'] * 1000 + (r['net_mf'] / 10000) 
        penalty = 0 
        
        if r['winner_rate'] > 60: base_score += 1000
        if 55 < r['rsi'] < 80: base_score += 2000 
        if r['rsi'] > RSI_LIMIT: penalty += 500
        return base_score - penalty

    fdf['Score'] = fdf.apply(dynamic_score, axis=1)
    final_df = fdf.sort_values('Score', ascending=False).head(TOP_BACKTEST).copy()
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    
    return final_df, None

# ---------------------------
# UI 及 主程序
# ---------------------------
with st.sidebar:
    st.header("V35.1 修复版")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数", value=30, step=1, help="建议30-50天")
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=4, help="实盘重点看 Rank 1, 2, 4")
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 清除行情缓存"):
        if os.path.exists(CACHE_FILE_NAME):
            os.remove(CACHE_FILE_NAME)
            st.success("缓存已清除，下次运行将重新下载最新数据。")
        if os.path.exists("backtest_checkpoint_v35.csv"):
            os.remove("backtest_checkpoint_v35.csv")
            st.success("断点文件已清除！")
            
    CHECKPOINT_FILE = "backtest_checkpoint_v35.csv" 
    
    st.markdown("---")
    st.subheader("💰 基础过滤")
    col1, col2 = st.columns(2)
    MIN_PRICE = col1.number_input("最低股价", value=15.0) 
    MIN_MV = col2.number_input("最小市值(亿)", value=30.0) 
    MAX_MV = st.number_input("最大市值(亿)", value=1000.0)
    
    st.markdown("---")
    st.subheader("⚔️ 核心风控参数 (V35)")
    CHIP_MIN_WIN_RATE = st.number_input("最低获利盘 (%)", value=40.0, help="V35建议: 40-50%")
    MAX_PREV_PCT = st.number_input("昨日最大涨幅限制 (%)", value=10.0)
    RSI_LIMIT = st.number_input("RSI 拦截线 (建议100)", value=100.0)
    
    st.markdown("---")
    st.subheader("📊 形态参数")
    SECTOR_THRESHOLD = st.number_input("板块涨幅 (%)", value=1.0)
    MAX_UPPER_SHADOW = st.number_input("上影线 (%)", value=6.0) 
    MIN_BODY_POS = st.number_input("实体位置", value=0.5) 
    MAX_TURNOVER_RATE = st.number_input("换手率 (%)", value=20.0)

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🚀 启动 V35.1"):
    st.empty() 
    end_date_str = backtest_date_end.strftime("%Y%m%d")
    
    with st.spinner("获取交易日历..."):
        # 提取120天保证 MA60 计算不断层
        dates_to_run = get_trade_days(end_date_str, int(BACKTEST_DAYS) + 120) 
        if not dates_to_run:
            st.error("获取日历失败，请检查网络或 Token 额度。")
            st.stop()
        
        test_dates = dates_to_run[:int(BACKTEST_DAYS)] 
        
    st.success(f"🗓️ 将回测 {len(test_dates)} 个交易日: {test_dates[-1]} 到 {test_dates[0]}")
    
    if not get_all_historical_data(dates_to_run):
        st.stop()

    results = []
    
    if RESUME_CHECKPOINT and os.path.exists(CHECKPOINT_FILE):
        try:
            saved_df = pd.read_csv(CHECKPOINT_FILE, dtype={'Trade_Date': str})
            saved_dates = saved_df['Trade_Date'].unique()
            remaining_dates = [d for d in test_dates if d not in saved_dates]
            
            if len(remaining_dates) < len(test_dates):
                st.info(f"🔄 发现断点记录，已跳过 {len(test_dates) - len(remaining_dates)} 天，继续跑剩下的 {len(remaining_dates)} 天。")
                results.append(saved_df)
                test_dates = remaining_dates
        except Exception as e:
            st.warning("断点文件读取失败，将重新跑全量。")

    if not test_dates and results:
        st.success("🎉 所有日期均已在缓存记录中，直接出结果！")
    else:
        bar = st.progress(0, text="开始回测...")
        
        for i, date in enumerate(test_dates):
            res, err = run_backtest_for_a_day(date, int(TOP_BACKTEST), 100, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS, RSI_LIMIT, CHIP_MIN_WIN_RATE, SECTOR_THRESHOLD, MIN_MV, MAX_MV, MAX_PREV_PCT, MIN_PRICE)
            if not res.empty:
                res['Trade_Date'] = date
                is_first = not os.path.exists(CHECKPOINT_FILE)
                res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                results.append(res)
            
            bar.progress((i+1)/len(test_dates), text=f"分析中: {date}")
        
        bar.empty()
    
    if results:
        all_res = pd.concat(results)
        all_res = all_res[all_res['Rank'] <= int(TOP_BACKTEST)]
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        all_res = all_res.sort_values(['Trade_Date', 'Rank'], ascending=[False, True])
        
        st.header(f"📊 V35.1 统计仪表盘 (Top {TOP_BACKTEST})")
        cols = st.columns(3)
        
        # 【核心逻辑】：仪表盘仅统计触发买入条件 (Yes) 的股票
        valid_buys = all_res[all_res['Buy_Triggered'] == 'Yes']
        
        for idx, n in enumerate([1, 3, 5]):
            col_name = f'Return_D{n} (%)'
            valid = valid_buys.dropna(subset=[col_name]) 
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                cols[idx].metric(f"D+{n} 均益 / 胜率", f"{avg:.2f}% / {win:.1f}%")
            else:
                cols[idx].metric(f"D+{n} 均益 / 胜率", "0.00% / 0.0%")
 
        st.subheader("📋 回测清单")
        
        show_cols = ['Rank', 'Trade_Date','name','ts_code','Close','Pct_Chg',
             'Buy_Triggered', 'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)', 
             'rsi', 'winner_rate', 'Sector_Boost']
        
        exist_cols = [c for c in show_cols if c in all_res.columns]
        st.dataframe(all_res[exist_cols], use_container_width=True)
        
        csv = all_res.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="💾 导出回测明细 (CSV)",
            data=csv,
            file_name=f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_export.csv",
            mime='text/csv'
        )
    else:
        st.warning("这段时间没有选出任何符合条件的股票。")
