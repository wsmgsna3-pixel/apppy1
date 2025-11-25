import tushare as ts
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 获取历史数据
def get_stock_data(stock_code, start_date, end_date, token):
    ts.set_token(token)  # 使用用户输入的Tushare token
    pro = ts.pro_api()
    df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
    return df

# 计算均线
def calculate_ma(df, period):
    df[f"MA_{period}"] = df['close'].rolling(window=period).mean()
    return df

# 计算20日涨幅
def calculate_20d_return(df):
    df['20d_return'] = (df['close'] / df['close'].shift(20) - 1) * 100
    return df

# 计算RSI
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    df['RSI'] = rsi
    return df

# 评分系统
def score_stock(df):
    score = 0
    
    # 突破强度评分
    df['breakout'] = df['close'] > df['close'].shift(5)  # 突破前期5天的最高价
    score += df['breakout'].sum()  # 如果突破，则给分数
    
    # 20日涨幅评分
    df['within_60'] = df['20d_return'] <= 60
    score += df['within_60'].sum()  # 如果过去20日涨幅不超过60%，给分
    
    # RSI评分
    df = calculate_rsi(df)
    df['RSI_signal'] = df['RSI'] < 30  # RSI小于30为买入信号
    score += df['RSI_signal'].sum()  # 如果RSI为买入信号，则加分
    
    # 均线交叉评分：5日均线穿越20日均线
    df['MA_cross'] = df['MA_5'] > df['MA_20']
    score += df['MA_cross'].sum()  # 如果均线交叉为买入信号，则加分
    
    return score

# 筛选股票：右侧刚启动或者20日涨幅不超过60%
def filter_stocks(df):
    # 条件1：价格突破前期高点
    df['breakout'] = df['close'] > df['close'].shift(5)  # 假设前期高点为5日前的最高点
    # 条件2：过去20个交易日涨幅不超过60%
    df['within_60'] = df['20d_return'] <= 60
    
    # 筛选符合两个条件的股票
    df = df[(df['breakout'] == True) & (df['within_60'] == True)]
    return df

# 回测逻辑
def backtest(df):
    initial_capital = 10000
    capital = initial_capital
    buy_price = 0
    for i in range(1, len(df)):
        if df['breakout'][i] == True and buy_price == 0:
            # 假设在信号出现时买入
            buy_price = df['close'][i]
        elif df['breakout'][i] == False and buy_price != 0:
            # 卖出
            sell_price = df['close'][i]
            capital += (sell_price - buy_price) * (capital // buy_price)
            buy_price = 0  # 清空买入价格，等待下次信号
    return capital

# Streamlit前端展示
def main():
    st.title("慧股 1.0 - 选股与回测系统")
    
    # 手动输入Tushare token
    token = st.text_input("请输入Tushare API Token：")
    
    if token:
        stock_code = st.text_input("股票代码", "000001.SZ")
        start_date = st.date_input("开始日期", pd.to_datetime("2022-01-01"))
        end_date = st.date_input("结束日期", pd.to_datetime("2023-01-01"))
        
        if stock_code:
            df = get_stock_data(stock_code, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), token)
            df = calculate_ma(df, 5)
            df = calculate_ma(df, 20)  # 添加20日均线
            df = calculate_20d_return(df)
            df = filter_stocks(df)
            
            # 显示选股结果
            st.write("筛选出的股票：")
            st.write(df[['trade_date', 'close', 'MA_5', 'MA_20', '20d_return']])

            # 显示评分
            score = score_stock(df)
            st.write(f"股票评分: {score}")

            # 添加回测按钮
            if st.button('开始回测'):
                final_capital = backtest(df)
                st.write(f"回测后资金: {final_capital:.2f} 元")

                # 绘制回测图表
                st.write("股票数据与均线：")
                st.line_chart(df[['close', 'MA_5', 'MA_20']])

# 启动Streamlit
if __name__ == "__main__":
    main()
