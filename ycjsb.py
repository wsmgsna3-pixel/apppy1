
import tushare as ts
import pandas as pd
import numpy as np
from tkinter import Tk, Label, Entry, Button

# 设置Tushare Token
def set_tushare_token():
    token = token_entry.get()  # 获取用户输入的token
    ts.set_token(token)
    pro = ts.pro_api()
    print("Token已设置，开始获取数据...")
    # 测试是否能成功连接到Tushare
    try:
        data = pro.stock_basic()
        print("数据获取成功")
    except Exception as e:
        print(f"获取数据失败: {e}")

# 获取股票的基本面数据
def get_stock_basic_info(stock_code, pro):
    stock_info = pro.stock_basic(ts_code=stock_code)
    return stock_info

# 获取股票的日线数据
def get_stock_data(stock_code, pro, start_date, end_date):
    df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    return df

# MA5 > MA10 > MA20 (趋势确认)
def trend_confirmation(df):
    df['short_sma'] = df['close'].rolling(window=5).mean()
    df['long_sma'] = df['close'].rolling(window=10).mean()
    df['longest_sma'] = df['close'].rolling(window=20).mean()
    
    if df['short_sma'].iloc[-1] > df['long_sma'].iloc[-1] and df['long_sma'].iloc[-1] > df['longest_sma'].iloc[-1]:
        return 1
    return 0

# MACD 金叉 (动能启动)
def macd_cross(df):
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    if macd.iloc[-1] > signal.iloc[-1]:
        return 1  # 金叉信号
    return 0

# 量价齐升 (主力资金确认)
def volume_price_up(df, volume_period=20):
    df['volume_sma'] = df['volume'].rolling(window=volume_period).mean()
    if df['volume'].iloc[-1] > df['volume_sma'].iloc[-1] and df['close'].iloc[-1] > df['close'].iloc[-2]:
        return 1  # 量价齐升
    return 0

# 突破前高 (加速上涨)
def breakout_previous_high(df):
    df['previous_high'] = df['close'].shift(1).rolling(window=20).max()  # 过去20天的最高价
    if df['close'].iloc[-1] > df['previous_high'].iloc[-1]:
        return 1  # 突破前高
    return 0

# 中阳线 / 吞没形态 (K线形态确认)
def candlestick_pattern(df):
    if df['close'].iloc[-1] > df['open'].iloc[-1] and (df['close'].iloc[-1] - df['open'].iloc[-1]) > 0.5 * (df['high'].iloc[-1] - df['low'].iloc[-1]):
        return 1  # 中阳线
    return 0

# 评分系统
def score_stock(df):
    trend_score = trend_confirmation(df)
    macd_score = macd_cross(df)
    volume_score = volume_price_up(df)
    breakout_score = breakout_previous_high(df)
    candle_score = candlestick_pattern(df)
    
    total_score = trend_score + macd_score + volume_score + breakout_score + candle_score
    return total_score

# 主选股函数
def select_stocks(stock_list, start_date, end_date, pro):
    selected_stocks = []
    
    for stock in stock_list:
        df = get_stock_data(stock, pro, start_date, end_date)
        
        # 计算股票评分
        score = score_stock(df)
        print(f"{stock}的评分: {score}")
        
        if score >= 3:  # 如果评分大于等于3则认为是合格股票
            selected_stocks.append(stock)
    
    return selected_stocks

# GUI界面设置
def on_submit():
    token = token_entry.get()
    set_tushare_token()  # 设置Tushare Token
    stock_list = ['000001.SZ', '000002.SZ', '600000.SH']  # 示例股票
    start_date = '20210101'
    end_date = '20211231'
    selected_stocks = select_stocks(stock_list, start_date, end_date, ts.pro_api())
    print("选出的股票：", selected_stocks)

# 创建GUI窗口
window = Tk()
window.title("慧股选股系统")

Label(window, text="请输入Tushare Token:").grid(row=0, column=0)
token_entry = Entry(window, width=40)
token_entry.grid(row=0, column=1)

Button(window, text="开始选股", command=on_submit).grid(row=1, columnspan=2)

window.mainloop()
