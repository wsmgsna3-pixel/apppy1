import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置页面配置
st.set_page_config(page_title="短线王 · 双模型选股王", layout="wide")
st.title("短线王 · 双模型（右侧 + 反转）")

# 输入 Tushare Token
TS_TOKEN = st.text_input("请输入你的 Tushare Token（只在本次会话使用）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 后才能运行。若已输入请回车确保激活。")
    st.stop()

# 初始化 Tushare API
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 获取最近交易日
def get_last_trade_day(pro, max_days=10):
    today = datetime.now()
    for i in range(0, max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        try:
            dd = pro.daily(trade_date=ds)
            if dd is not None and len(dd) > 0:
                return ds
        except Exception:
            continue
    return None

last_trade = get_last_trade_day(pro)
if not last_trade:
    st.error("无法获取最近交易日，请检查 Token 或网络。")
    st.stop()

st.info(f"参考最近交易日：{last_trade}")

# 获取市场数据和相关指标
def get_market_data(pro, last_trade):
    # 获取当天的市场数据
    market_df = pro.daily(trade_date=last_trade)
    # 获取额外的股票信息（如行业、换手率、成交额等）
    stock_basic_df = pro.stock_basic(list_status='L', fields='ts_code,name,industry')
    daily_basic_df = pro.daily_basic(trade_date=last_trade, fields='ts_code,turnover_rate,amount')
    moneyflow_df = pro.moneyflow(trade_date=last_trade)

    # 合并数据
    df = market_df.merge(stock_basic_df[['ts_code', 'name', 'industry']], on='ts_code', how='left')
    df = df.merge(daily_basic_df[['ts_code', 'turnover_rate', 'amount']], on='ts_code', how='left')
    df = df.merge(moneyflow_df[['ts_code', 'net_mf']], on='ts_code', how='left')

    return df

# 筛选右侧交易股票（趋势向上）
def right_side_trading_filter(df):
    # 右侧交易模型：只筛选涨幅、趋势向上的股票
    df = df[df['pct_chg'] > 2]  # 涨幅大于2%
    df = df[df['turnover_rate'] > 1]  # 换手率大于1%
    df = df[df['net_mf'] > 0]  # 主力资金流入
    df = df[df['amount'] > 1000000]  # 成交额大于100万
    return df

# 筛选反转交易股票（超卖反转）
def reversal_trading_filter(df):
    # 反转交易模型：寻找跌幅过深，资金回流的股票
    df = df[df['pct_chg'] < -2]  # 跌幅超过2%
    df = df[df['net_mf'] > -1000000]  # 主力资金净流出但不超过100万
    df = df[df['amount'] > 1000000]  # 成交额大于100万
    return df

# 动态切换模型
def select_stocks(pro, last_trade):
    # 获取市场数据
    market_df = get_market_data(pro, last_trade)

    # 判断当前市场是强势还是弱势
    if market_df['pct_chg'].mean() > 0:
        # 市场强势，使用右侧交易模型
        selected_stocks = right_side_trading_filter(market_df)
    else:
        # 市场弱势，使用反转交易模型
        selected_stocks = reversal_trading_filter(market_df)

    return selected_stocks

# 运行股票筛选过程
def run_pipeline():
    # 获取最后交易日
    last_trade = get_last_trade_day(pro)
    if not last_trade:
        st.error("无法获取最近交易日，请检查 Token 或网络。")
        return
    
    # 获取股票数据
    selected_stocks = select_stocks(pro, last_trade)
    
    if selected_stocks.empty:
        st.warning("未选出符合条件的股票，尝试放宽筛选条件或检查权限。")
        return
    
    # 显示前20名股票
    st.write("最终选出的股票（排名前 20）：")
    st.dataframe(selected_stocks.head(20))

# 执行选股操作
run_pipeline()
