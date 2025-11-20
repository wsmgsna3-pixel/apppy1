import tushare as ts
import pandas as pd
import talib
import time
import random

# 获取Tushare token
def get_tushare_token():
    return input("请输入您的Tushare token: ")

# 设置Tushare token
def set_tushare_token(token):
    ts.set_token(token)
    return ts.pro_api()

# 获取股票数据（获取指定时间区间的日线数据）
def get_stock_data(stock_code, start_date, end_date):
    try:
        df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
        if df.empty:
            raise ValueError(f"无法获取数据: {stock_code}")
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        return df
    except Exception as e:
        print(f"获取股票数据失败: {e}")
        return pd.DataFrame()

# 计算技术指标
def calculate_technical_indicators(df):
    try:
        # 计算MACD
        df['macd'], df['signal'], df['hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        # 计算RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        # 计算均线（如5日线、20日线）
        df['ma5'] = talib.SMA(df['close'], timeperiod=5)
        df['ma20'] = talib.SMA(df['close'], timeperiod=20)
        return df
    except Exception as e:
        print(f"计算技术指标失败: {e}")
        return df

# 筛选符合右侧选股条件的股票
def stock_selection(stock_code, start_date, end_date):
    df = get_stock_data(stock_code, start_date, end_date)
    if df.empty:
        return None

    df = calculate_technical_indicators(df)
    
    # 判断是否符合右侧选股条件
    # 1. MACD金叉（短期上涨信号）
    # 2. 均线金叉（5日线突破20日线）
    # 3. RSI回升至50以上
    if (df['macd'].iloc[-1] > df['signal'].iloc[-1] and  # MACD金叉
        df['ma5'].iloc[-1] > df['ma20'].iloc[-1] and    # 均线金叉
        df['rsi'].iloc[-1] > 50):                        # RSI回升至50以上
        print(f"{stock_code} 符合右侧选股条件！")
        return stock_code
    return None

# 获取资金流向数据
def get_fund_flow(stock_code):
    try:
        fund_flow = pro.moneyflow(ts_code=stock_code, start_date='20230101', end_date='20231031')
        if fund_flow.empty:
            print(f"无法获取资金流向数据: {stock_code}")
            return None
        return fund_flow
    except Exception as e:
        print(f"获取资金流向数据失败: {e}")
        return None

# 排除ST股和北交所股票
def is_valid_stock(stock_code):
    stock_info = pro.stock_basic(ts_code=stock_code, fields='ts_code,name,market')
    if stock_info.empty:
        return False
    market_type = stock_info['market'][0]
    if 'ST' in stock_info['name'][0] or market_type in ['B', 'ST']:
        print(f"排除ST股票或北交所股票: {stock_code}")
        return False
    return True

# 筛选符合条件的股票（包括资金流向和基本面）
def filter_stocks(stock_codes, start_date, end_date):
    valid_stocks = []
    for stock_code in stock_codes:
        if is_valid_stock(stock_code):
            print(f"检查股票: {stock_code}")
            # 检查资金流向
            fund_flow_data = get_fund_flow(stock_code)
            if fund_flow_data is None or fund_flow_data['net_money'][0] <= 0:
                print(f"资金流出，忽略股票: {stock_code}")
                continue

            # 检查技术指标是否符合右侧选股条件
            selected_stock = stock_selection(stock_code, start_date, end_date)
            if selected_stock:
                valid_stocks.append(stock_code)
    
    return valid_stocks

# 主程序
def select_right_side_stocks(stock_codes, start_date, end_date):
    selected_stocks = filter_stocks(stock_codes, start_date, end_date)
    print("\n最终筛选出的符合右侧选股条件的股票：")
    for stock in selected_stocks:
        print(stock)

# 示例股票列表（你可以根据需求调整股票列表）
stock_codes = ['000001.SZ', '000002.SZ', '600000.SH', '300001.SZ', '688001.SH']
start_date = '20230101'
end_date = '20231031'

# 运行右侧选股策略
select_right_side_stocks(stock_codes, start_date, end_date)
