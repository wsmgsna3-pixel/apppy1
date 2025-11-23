import tushare as ts
import pandas as pd

# 获取股票的历史数据
def get_stock_data(stock_code, start_date, end_date):
    """
    获取股票的历史交易数据
    """
    df = ts.pro_bar(ts_code=stock_code, start_date=start_date, end_date=end_date, asset='E')
    return df

# 执行回测
def backtest(start_date, end_date, initial_cash=100000, TOP_DISPLAY=30, hold_days=5, fee_rate=0.001):
    """
    回测选股王选出的股票
    """
    selected_stocks = get_selected_stocks(TOP_DISPLAY)  # 获取选出的股票池
    cash = initial_cash  # 初始资金
    portfolio = {}  # 当前持仓
    trade_history = []  # 记录交易历史

    # 循环遍历每只选出的股票
    for stock in selected_stocks:
        stock_data = get_stock_data(stock, start_date, end_date)
        
        if stock_data is None or len(stock_data) == 0:
            continue  # 如果没有历史数据，跳过

        # 设定买入和卖出价格：买入后持有 hold_days 天
        buy_price = stock_data['close'].iloc[0]  # 第一天的收盘价
        buy_date = stock_data['trade_date'].iloc[0]
        sell_price = stock_data['close'].iloc[hold_days-1]  # 持股 hold_days 天后的收盘价
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

    # 计算回测结束时的总资产
    total_assets = cash
    total_profit = total_assets - initial_cash
    return total_assets, total_profit, trade_history

# 执行回测并输出结果
start_date = '20210101'
end_date = '20230101'
initial_cash = 100000  # 初始资金
TOP_DISPLAY = 30  # 显示前30只股票
hold_days = 5  # 持股5天
fee_rate = 0.001  # 交易费用率（千分之一）

final_cash, total_profit, history = backtest(start_date, end_date, initial_cash, TOP_DISPLAY, hold_days, fee_rate)

# 输出回测结果
print(f"最终资金：{final_cash}")
print(f"总收益：{total_profit}")
print("交易历史：")
for trade in history:
    print(trade)
