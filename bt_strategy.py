# bt_strategy.py
# backtrader 策略：基于信号入场，次日开盘执行，支持止损/止盈/持仓天数/等权仓位

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SignalStrategy(bt.Strategy):
    params = dict(
        stake=1,
        stoploss_pct=0.08,   # 初始止损
        takeprofit_pct=0.20, # 初始止盈
        hold_days=5,
        verbose=False
    )

    def __init__(self):
        # track pending buy orders per data (we use datas list)
        self.order_dict = {}  # data._name -> order
        self.entry_price = {}
        self.entry_dt = {}

    def log(self, txt, dt=None):
        if self.p.verbose:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        dt = self.datas[0].datetime.date(0)
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: {order.data._name}, Price: {order.executed.price}, Size: {order.executed.size}', dt)
                self.entry_price[order.data._name] = order.executed.price
                self.entry_dt[order.data._name] = dt
            elif order.issell():
                self.log(f'SELL EXECUTED: {order.data._name}, Price: {order.executed.price}, Size: {order.executed.size}', dt)
                if order.data._name in self.entry_price:
                    del self.entry_price[order.data._name]
                    del self.entry_dt[order.data._name]
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected', dt)

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'OPERATION RESULT, GROSS {trade.pnl}, NET {trade.pnlcomm}')

    def next(self):
        # We implement next-day-open execution by checking if a data has "signal" flag on its last bar.
        # Here, for each data, if it has attribute 'signal' on current bar set to 1, we will create a buy order on next open.
        for d in self.datas:
            name = d._name
            sig = getattr(d, 'signal', 0)
            pos = self.getposition(d).size
            # if buy signal and no position, create buy at next open (we place market order now; engine will execute at open if next() used with "cheat-on-close")
            if sig == 1 and pos == 0:
                cash = self.broker.getcash()
                # stake calculation: use percent of portfolio or fixed stake
                size = self.p.stake
                try:
                    self.buy(data=d, size=size)
                except Exception as e:
                    self.log(f"Buy order failed: {e}")
            # manage existing position: stoploss, takeprofit, hold_days
            if pos > 0:
                cur_price = d.close[0]
                entry = self.entry_price.get(name, None)
                buy_dt = self.entry_dt.get(name, None)
                # stoploss
                if entry and cur_price <= entry * (1 - self.p.stoploss_pct):
                    self.log(f"Stoploss triggered for {name} at {cur_price}")
                    self.close(data=d)
                # takeprofit
                if entry and cur_price >= entry * (1 + self.p.takeprofit_pct):
                    self.log(f"Takeprofit triggered for {name} at {cur_price}")
                    self.close(data=d)
                # hold days
                if buy_dt:
                    if (datetime.now().date() - buy_dt).days >= self.p.hold_days:
                        self.log(f"Max hold days reached for {name}")
                        self.close(data=d)
