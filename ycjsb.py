import tushare as ts
import pandas as pd
import numpy as np

# 设置 Tushare Token
def set_token(token):
    ts.set_token(token)
    pro = ts.pro_api()
    return pro

# 获取基本面数据（如市值和股价）
def get_fundamentals(pro, start_date, end_date):
    """
    获取股票的基本面数据，包括市值和股价
    """
    fundamentals = pro.daily_basic(ts_code='', trade_date=end_date, fields='ts_code,total_mv,close')
    return fundamentals

# 获取技术面数据（如均线、RSI、MACD、布林带）
def get_technical_data(pro, stock_code, start_date, end_date):
    """
    获取股票的技术面数据，例如5日均线、20日均线、RSI、MACD和布林带
    """
    df = ts.pro_bar(ts_code=stock_code, start_date=start_date, end_date=end_date, asset='E', ma=[5, 20, 60])
    
    # 计算RSI
    df['rsi'] = pd.Series.ta.rsi(df['close'], length=14)
    
    # 计算MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = pd.Series.ta.macd(df['close'], fast=12, slow=26, signal=9)
    
    # 计算布林带
    df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = pd.Series.ta.bbands(df['close'], length=20, std=2)
    
    return df

# 计算股价涨幅
def calculate_price_change(pro, stock_code, start_date, end_date):
    """
    计算股票在一定时间区间内的涨幅，避免选择已经涨幅过大的股票
    """
    df = ts.pro_bar(ts_code=stock_code, start_date=start_date, end_date=end_date, asset='E')
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    price_change = (end_price - start_price) / start_price
    return price_change

# 排除 ST股 和 北交所股的逻辑
def filter_st_and_bse(stocks):
    """
    排除 ST股 和 北交所股
    """
    valid_stocks = []
    for stock in stocks:
        # 排除 ST 股（股票代码中包含 "ST"）
        if "ST" in stock or "st" in stock:
            continue
        # 排除 北交所股（假设代码中包含 "B" 的为北交所股，需根据实际情况调整）
        if stock[:2] == "B":
            continue
        valid_stocks.append(stock)
    return valid_stocks

# 评分逻辑
def calculate_score(tech_data, fundamentals, stock):
    score = 0
    
    # 5日均线 > 20日均线
    if tech_data['ma5'].iloc[-1] > tech_data['ma20'].iloc[-1]:
        score += 20
    
    # RSI < 70
    if tech_data['rsi'].iloc[-1] < 70:
        score += 10
    
    # MACD > 0
    if tech_data['macd_hist'].iloc[-1] > 0:
        score += 15
    
    # 股价突破布林带上轨
    if fundamentals['close'].iloc[-1] > tech_data['bollinger_upper'].iloc[-1]:
        score += 15
    
    # 股价范围在10-200元之间
    price = fundamentals['close'].iloc[-1]
    if 10 <= price <= 200:
        score += 10
    
    # 市值范围在20亿-500亿之间
    market_cap = fundamentals['total_mv'].iloc[-1]
    if 20e8 <= market_cap <= 500e8:
        score += 10
    
    return score

# 选股逻辑：筛选符合条件的股票
def select_stocks(pro, start_date, end_date):
    """
    选股策略：结合技术指标、资金流向和涨幅条件筛选符合要求的股票
    """
    fundamentals = get_fundamentals(pro, start_date, end_date)
    selected_stocks = []

    for stock in fundamentals['ts_code']:
        # 获取技术面数据
        tech_data = get_technical_data(pro, stock, start_date, end_date)

        # 计算股票得分
        score = calculate_score(tech_data, fundamentals, stock)

        # 保存股票代码和得分
        selected_stocks.append({'stock': stock, 'score': score})

    # 按得分排序，输出前20名
    selected_stocks_sorted = sorted(selected_stocks, key=lambda x: x['score'], reverse=True)[:20]
    
    return selected_stocks_sorted

# 回测逻辑：模拟持股1到5天，并设置买点卖点
def backtest(pro, start_date, end_date, initial_cash=100000, hold_days=5, fee_rate=0.001):
    selected_stocks = select_stocks(pro, start_date, end_date)
    
    cash = initial_cash  # 初始资金
    trade_history = []  # 记录交易历史
    total_profit = 0  # 总收益

    for stock_info in selected_stocks:
        stock = stock_info['stock']
        # 获取股票的历史数据
        stock_data = get_technical_data(pro, stock, start_date, end_date)
        
        if stock_data is None or len(stock_data) == 0:
            continue  # 如果没有历史数据，跳过

        # 设置买入信号：5日均线突破20日均线
        if stock_data['ma5'].iloc[-1] > stock_data['ma20'].iloc[-1]:
            buy_price = stock_data['close'].iloc[0]  # 第一天的收盘价
            buy_date = stock_data['trade_date'].iloc[0]
            
            # 持股一定天数后卖出（设定为5天）
            sell_price = stock_data['close'].iloc[hold_days-1]  # 持股5天后的收盘价
            sell_date = stock_data['trade_date'].iloc[hold_days-1]
            
            # 计算买入股票的数量
            quantity = cash // buy_price  # 用现有资金买入股票
            cash -= quantity * buy_price  # 扣除资金

            # 记录交易历史
            trade_history.append({
                'stock': stock,
                'buy_date': buy_date,
                'buy_price': buy_price,
                'sell_date': sell_date,
                'sell_price': sell_price,
                'quantity': quantity,
                'profit': quantity * (sell_price - buy_price) - (quantity * buy_price * fee_rate) - (quantity * sell_price * fee_rate)
            })

            # 卖出股票，更新资金
            cash += quantity * sell_price  # 卖出股票获得现金

    total_assets = cash
    total_profit = total_assets - initial_cash
    return total_assets, total_profit, trade_history

# 主函数
def main():
    token = input("请输入你的Tushare Token: ")
    pro = set_token(token)
    
    start_date = '20220101'
    end_date = '20221231'
    
    # 执行回测
    final_cash, total_profit, history = backtest(pro, start_date, end_date)

    print(f"最终资金：{final_cash}")
    print(f"总收益：{total_profit}")
    print("交易历史：")
    for trade in history:
        print(trade)

if __name__ == "__main__":
    main()
