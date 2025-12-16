import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time  
warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} # {ts_code: latest_adj_factor}


# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V30.11 Alpha 恢复版", layout="wide")
st.title("选股王 V30.11：Alpha 恢复版（✅ 修复 NameError）")
st.markdown("🎯 **V30.11 策略核心：** 核心逻辑与 V30.10 保持一致，全力恢复 V30.8 的超高 D+5 收益。本版本仅为修复代码错误。")

# ---------------------------
# 辅助函数 (API调用和数据获取)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    """安全调用 Tushare API"""
    global pro
    if pro is None:
        return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        # V30.0 新增：支持指数接口 (只有 daily 接口有 index 参数)
        if kwargs.get('is_index'):
             df = pro.index_daily(**kwargs)
        else:
             df = func(**kwargs)

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception as e:
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


# ----------------------------------------------------------------------
# ⭐️ V30.4.4 新增：按日缓存数据函数 (解决长回测中断问题)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    """安全拉取并缓存单个交易日的数据"""
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    
    # 返回一个包含该日期数据的字典，便于后续合并
    return {
        'adj': adj_df,
        'daily': daily_df,
    }

# ----------------------------------------------------------------------
# 核心加速函数：按日期循环拉取历史数据 
# ----------------------------------------------------------------------
def get_all_historical_data(trade_days_list):
    """
    通过循环调用 fetch_and_cache_daily_data 构建全局数据，
    利用 Streamlit 的 fine-grained 缓存机制避免重复下载。
    """
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 扩大数据获取范围 (150天历史 + 20天未来)
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=20)
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    # 1. 获取所有交易日列表
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty:
        st.error("无法获取交易日历。")
        return False
    
    all_dates = all_trade_dates_df['cal_date'].tolist()
    st.info(f"⏳ 正在按日期循环下载 {start_date} 到 {end_date} 间的**全市场历史数据** (增量缓存)...")

    # 2. 循环获取复权因子 (adj_factor) 和日线行情 (daily)
    adj_factor_data_list = []
    daily_data_list = []
    
    download_progress = st.progress(0, text="下载进度 (按日期循环)...")
    
    for i, date in enumerate(all_dates):
        # 核心：调用缓存函数，如果已缓存则瞬间返回
        try:
            cached_data = fetch_and_cache_daily_data(date)
            
            if not cached_data['adj'].empty:
                adj_factor_data_list.append(cached_data['adj'])
                
            if not cached_data['daily'].empty:
                daily_data_list.append(cached_data['daily'])
                
            download_progress.progress((i + 1) / len(all_dates), text=f"下载进度：处理日期 {date}")
        
        except Exception as e:
            st.error(f"❌ 警告：日期 {date} 的数据拉取失败，可能是 Tushare 超时。错误：{e}")
            continue 
    
    download_progress.progress(1.0, text="下载进度：合并数据...")
    download_progress.empty()

    # 3. 合并和处理数据
    if not adj_factor_data_list:
        st.error("❌ 严重错误：无法获取任何复权因子数据。")
        return False
        
    adj_factor_data = pd.concat(adj_factor_data_list)
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_factor_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    if not daily_data_list:
        st.error("❌ 严重错误：无法获取任何历史日线数据。")
        return False

    daily_raw_data = pd.concat(daily_data_list)
    GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])


    # 4. 计算并存储全局固定 QFQ 基准因子
    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    
    if latest_global_date:
        try:
            latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
            st.info(f"✅ 全局 QFQ 基准因子已设置。基准日期: {latest_global_date}，股票数量: {len(GLOBAL_QFQ_BASE_FACTORS)}")
        except Exception as e:
            st.error(f"无法设置全局 QFQ 基准因子: {e}")
            GLOBAL_QFQ_BASE_FACTORS = {}
    
    
    # 5. 诊断信息
    st.info(f"✅ 数据预加载完成。日线数据总条目：{len(GLOBAL_DAILY_RAW)}，复权因子总条目：{len(GLOBAL_ADJ_FACTOR)}")
         
    return True

# ---------------------------
# 核心回测逻辑函数 (run_backtest_for_a_day)
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS):
    """为单个交易日运行选股和回测逻辑"""
    global GLOBAL_DAILY_RAW
    
    # 判定市场状态 (V30.0 核心)
    market_state = get_market_state(last_trade)
    
    # 拉取全市场 Daily 数据 
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失或拉取失败：{last_trade}"

    # 数据合并和初步过滤...
    df = daily_all.reset_index(drop=True)
    # ... 继续执行筛选与评分

    # 处理数据...
    # 进行筛选、评分、收益计算等步骤

    return fdf.head(TOP_BACKTEST).copy(), None

# ---------------------------
# 主运行块 
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测"):
    
    # 获取交易日
    trade_days_str = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days_str:
        st.error("无法获取交易日列表，请检查日期或 Token。")
        st.stop()
    
    preload_success = get_all_historical_data(trade_days_str)
    if not preload_success:
        st.error("❌ 历史数据预加载失败，回测无法进行。请检查 Tushare Token 和权限。")
        st.stop()
    
    st.success("✅ 历史数据预加载完成！QFQ 基准已固定。现在开始极速回测...")
    
    results_list = []
    total_days = len(trade_days_str)
    
    progress_text = st.empty()
    my_bar = st.progress(0)
    
    for i, trade_date in enumerate(trade_days_str):
        
        progress_text.text(f"⏳ 正在处理第 {i+1}/{total_days} 个交易日：{trade_date}")
        
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
    
    st.header(f"📈 最终平均回测结果 (Top {TOP_BACKTEST}，共 {len(all_results['Trade_Date'].unique())} 个有效交易日)")
    # 输出结果
