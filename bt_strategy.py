# bt_strategy.py
# backtrader 策略：内置 RSI/MACD 计算，自动生成买卖信号
# 修复了信号读取问题和持仓天数计算 BUG

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SignalStrategy(bt.Strategy):
    params = dict(
        stake=100,           # 默认每次买1手 (100股)，之前是1股太少了
        stoploss_pct=0.08,   # 止损 8%
        takeprofit_pct=0.20, # 止盈 20%
        hold_days=5,         # 最大持仓天数
        rsi_max=75,          # RSI 阈值
        macd_min=-0.3,       # MACD 阈值
        verbose=False
    )

    def __init__(self):
        # 记录订单状态
        self.order_dict = {} 
        self.entry_price = {}
        self.entry_dt = {}
        
        # --- 核心修复：在这里为每一只股票提前计算好指标 ---
        self.inds = {}
        for d in self.datas:
            self.inds[d] = {
                # 计算 RSI (14天)
                'rsi': bt.indicators.RSI(d.close, period=14),
                # 计算 MACD
                'macd': bt.indicators.MACD(d.close)
            }

    def log(self, txt, dt=None):
        if self.p.verbose:
            # 兼容处理：如果没有指定时间，尝试获取第一个数据流的时间
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        # 获取回测当前的日期
        dt = self.datas[0].datetime.date(0)
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🔵 买入执行: {order.data._name}, 价格: {order.executed.price:.2f}, 数量: {order.executed.size}', dt)
                self.entry_price[order.data._name] = order.executed.price
                self.entry_dt[order.data._name] = dt
            elif order.issell():
                self.log(f'🔴 卖出执行: {order.data._name}, 价格: {order.executed.price:.2f}, 收益: {order.executed.pnl:.2f}', dt)
                # 清理记录
                if order.data._name in self.entry_price:
                    del self.entry_price[order.data._name]
                    del self.entry_dt[order.data._name]
                    
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'❌ 订单被取消/拒绝/保证金不足: {order.data._name}', dt)

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'💰 交易结算: 毛利 {trade.pnl:.2f}, 净利 {trade.pnlcomm:.2f}')

    def next(self):
        # 遍历回测池中的每一只股票
        for d in self.datas:
            name = d._name
            pos = self.getposition(d).size
            
            # 获取当前回测日期
            current_date = d.datetime.date(0)
            
            # --- 1. 买入逻辑 (修复部分) ---
            if pos == 0:
                # 获取该股票预先算好的指标
                rsi_val = self.inds[d]['rsi'][0]
                macd_val = self.inds[d]['macd'].macd[0]
                
                # 定义买入条件：RSI 不超买 且 MACD 大于底限
                # 既然这些股票已经是你精选池里的，我们只要指标不坏就买入
                condition_buy = (rsi_val < self.p.rsi_max) and (macd_val > self.p.macd_min)
                
                if condition_buy:
                    # 获取当前账户现金
                    cash = self.broker.getcash()
                    # 只有现金足够买1手才下单
                    if cash > d.close[0] * 100:
                        self.buy(data=d, size=self.p.stake)
                        # self.log(f"发出买入信号: {name} (RSI={rsi_val:.1f})")

            # --- 2. 卖出逻辑 (止盈/止损/限时) ---
            elif pos > 0:
                cur_price = d.close[0]
                entry = self.entry_price.get(name, cur_price) # 防错默认值
                buy_date = self.entry_dt.get(name, None)
                
                # (A) 止损
                if cur_price <= entry * (1 - self.p.stoploss_pct):
                    self.log(f"📉 触发止损: {name} 现价 {cur_price:.2f} < 成本 {entry:.2f}")
                    self.close(data=d)
                    
                # (B) 止盈
                elif cur_price >= entry * (1 + self.p.takeprofit_pct):
                    self.log(f"🚀 触发止盈: {name} 现价 {cur_price:.2f} > 成本 {entry:.2f}")
                    self.close(data=d)
                    
                # (C) 持仓天数限制 (修复了 datetime.now 的错误)
                elif buy_date:
                    # 计算持仓了多少个“日历日”
                    days_held = (current_date - buy_date).days
                    if days_held >= self.p.hold_days:
                        self.log(f"⏰ 到期平仓: {name} 持仓 {days_held} 天")
                        self.close(data=d)
