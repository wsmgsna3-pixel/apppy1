# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（终极修复版 v7.0 - 结构加固与参数验证）
说明：
- 整合了 BC 混合增强策略。
- **v7.0 核心改动：** 1. 修复了回测结果转换为 DataFrame 时的结构性崩溃 (ValueError)。
    2. 维持 UUID 缓存绕过机制，配合用户重启操作。
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import uuid # 引入 uuid 用于生成随机数

# 确保 tushare 在必要时被导入和配置
try:
    import tushare as ts
except ImportError:
    st.error("缺少 tushare 库，请确保环境已安装。")
    st.stop()

warnings.filterwarnings("ignore")

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · 10000旗舰（强制执行 v7.0）", layout="wide")
st.title("选股王 · 10000 积分旗舰（强制执行版 v7.0）")
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
    BACKTEST_TOP_K = int(st.number_input("回测每日最多交易 K 支", value=3, min_value=1, max_value=10)) # 默认 K=3
    HOLD_DAYS_OPTIONS = st.multiselect("回测持股天数", options=[1, 3, 5, 10, 20], default=[1, 3, 5])
    # 策略参数 (用于回测逻辑)
    BT_MAX_PCT = float(st.number_input("回测：最高涨幅 (上限)", value=9.9, step=0.5)) # 默认 9.9
    BT_MIN_PCT = float(st.number_input("回测：最低涨幅 (下限)", value=3.0, step=0.1)) # 默认 3.0
    st.caption("提示：**当前回测使用默认的涨幅区间 (3.0% < 涨幅 < 9.9%)。**")

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
# 安全调用 & 缓存辅助 (保持 V6.0 逻辑不变)
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

# --- 当日选股逻辑 (略) --- 
# (此处代码与 V6.0 相同，省略以节省篇幅，但在用户脚本中需要完整保留)

# ---------------------------
# 历史回测部分（数据性能优化与逻辑强化）
# ---------------------------

# ⚠️ V7.0 核心改动：在参数中引入随机数，强制 Streamlit 忽略缓存
def load_backtest_data(all_trade_dates, cache_buster):
    """预加载所有回测日期的 daily 数据，并使用 cache_buster 强制绕过 Streamlit 缓存。"""
    data_cache = {}
    st.write(f"正在预加载回测所需 {len(all_trade_dates)} 个交易日的全部 daily 数据 (约 {len(all_trade_dates)} 次 API 调用)...")
    st.warning("✅ 强制执行：每次回测都将重新下载数据和计算，耗时约 3 分钟。")
    pbar = st.progress(0)
    for i, date in enumerate(all_trade_dates):
        daily_df = safe_get(pro.daily, trade_date=date)
        if not daily_df.empty:
            data_cache[date] = daily_df.set_index('ts_code')
        pbar.progress((i + 1) / len(all_trade_dates))
    pbar.progress(1.0)
    return data_cache

# ⚠️ 彻底移除 @st.cache_data 装饰器
def run_backtest(start_date, end_date, hold_days, backtest_top_k, bt_max_pct, bt_min_pct):
    st.text(f"🚀 V7.0 回测逻辑强制激活中...日期范围 {start_date} 到 {end_date}。")
    
    trade_dates = get_trade_cal(start_date, end_date)
    
    if not trade_dates:
        # 如果没有交易日，返回一个带有 Hold Days 但数据为空的结构
        return {h: {'returns': [], 'wins': 0, 'total': 0, 'win_rate': 0.0, 'avg_return': 0.0} for h in hold_days}

    results = {h: {'returns': [], 'wins': 0, 'total': 0, 'win_rate': 0.0, 'avg_return': 0.0} for h in hold_days}
    
    bt_start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=BACKTEST_DAYS * 2)).strftime("%Y%m%d")
    buy_dates_pool = [d for d in trade_dates if d >= bt_start and d <= end_date]
    backtest_dates = buy_dates_pool[-BACKTEST_DAYS:]
    
    if len(backtest_dates) < BACKTEST_DAYS:
        st.warning(f"由于数据或交易日限制，回测仅能覆盖 {len(backtest_dates)} 天。")
    
    # 确定回测所需的全部交易日
    required_dates = set(backtest_dates)
    for buy_date in backtest_dates:
        try:
            current_index = trade_dates.index(buy_date)
            for h in hold_days:
                required_dates.add(trade_dates[current_index + h])
        except (ValueError, IndexError):
            continue
            
    # ** V7.0 核心：每次都重新加载数据，并使用 UUID 强制绕过缓存 **
    data_cache = load_backtest_data(sorted(list(required_dates)), cache_buster=str(uuid.uuid4()))

    st.write(f"正在模拟 {len(backtest_dates)} 个交易日的选股回测...")
    pbar_bt = st.progress(0)
    
    for i, buy_date in enumerate(backtest_dates):
        daily_df_cached = data_cache.get(buy_date)
        
        if daily_df_cached is None or daily_df_cached.empty:
            pbar_bt.progress((i+1)/len(backtest_dates));
            continue

        daily_df = daily_df_cached.copy().reset_index()
        
        # 1. 应用基本过滤 
        BACKTEST_MIN_AMOUNT_PROXY = MIN_AMOUNT * 2.0 
        
        daily_df['amount_yuan'] = daily_df['amount'].fillna(0) * 1000.0 # 转换成元
        
        daily_df = daily_df[
            (daily_df['close'] >= MIN_PRICE) & 
            (daily_df['close'] <= MAX_PRICE) &
            (daily_df['amount_yuan'] >= BACKTEST_MIN_AMOUNT_PROXY) & 
            (daily_df['pct_chg'] >= bt_min_pct) & 
            (daily_df['pct_chg'] <= bt_max_pct) & 
            (daily_df['vol'] > 0) & 
            (daily_df['amount_yuan'] > 0)
        ].copy()
        
        # 过滤一字涨停板
        daily_df['is_zt'] = (daily_df['open'] == daily_df['high']) & (daily_df['pct_chg'] > 9.5)
        daily_df = daily_df[~daily_df['is_zt']].copy()
        
        # 2. 模拟评分：按【涨幅】排序
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
        # 即使 total=0，也返回结构，但计算收益率和胜率
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
        
        st.warning("✅ V7.0 终极修复：回测正在执行中...这次预计耗时约 3 分钟！")
        
        try:
            start_date_for_cal = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
        except:
            start_date_for_cal = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            
        # 强制运行回测
        backtest_result = run_backtest(
            start_date=start_date_for_cal,
            end_date=last_trade,
            hold_days=HOLD_DAYS_OPTIONS,
            backtest_top_k=BACKTEST_TOP_K,
            bt_max_pct=BT_MAX_PCT,
            bt_min_pct=BT_MIN_PCT
        )

        # ⚠️ V7.0 核心加固：确保结果可以转换为 DataFrame
        if not backtest_result or not any(v['总交易次数'] > 0 for v in backtest_result.values()):
            # 如果字典为空或所有交易次数都为 0，则手动构造一个 DataFrame 来避免崩溃
            results_list = []
            for h in HOLD_DAYS_OPTIONS:
                results_list.append({
                    '持股天数': f"{h} 天", 
                    '平均收益率 (%)': "0.00", 
                    '胜率 (%)': "0.00", 
                    '总交易次数': 0
                })
            bt_df = pd.DataFrame(results_list)
            st.error("回测结果为 0 交易：请检查 Tushare Token 权限、回测日期范围或参数设置是否过于严格。")
        else:
            # 正常转换结果
            bt_df = pd.DataFrame(backtest_result).T
            bt_df.index.name = "持股天数"
            bt_df = bt_df.reset_index()
            bt_df['持股天数'] = bt_df['持股天数'].astype(str) + ' 天'
            
            st.success("回测完成！")
        
        # 显示结果
        st.dataframe(bt_df, use_container_width=True, hide_index=True)

        # 导出逻辑：确保列数匹配，解决 ValueError
        export_df = bt_df.copy()
        if len(export_df.columns) == 4:
            export_df.columns = ['HoldDays', 'AvgReturn', 'WinRate', 'TotalTrades']
            out_csv_bt = export_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "下载回测结果 CSV", 
                data=out_csv_bt, 
                file_name=f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("下载导出失败：回测结果结构异常，无法生成 CSV。")

