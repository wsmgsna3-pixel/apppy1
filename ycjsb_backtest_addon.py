# bt_strategy.py
# backtrader 策略：宽松调试版
# 目的：强制触发交易，验证回测系统是否正常计算盈亏

import backtrader as bt
import math

class SignalStrategy(bt.Strategy):
    # 将默认参数设置得非常宽松，确保能买入
    params = dict(
        stake=100,           # 每次买100股
        stoploss_pct=0.10,   # 止损 10%
        takeprofit_pct=0.30, # 止盈 30%
        hold_days=10,        # 持仓 10 天
        rsi_max=100,         # 【修改】设为 100，意味着只要 RSI 算出来就买
        macd_min=-100.0,     # 【修改】设为 -100，意味着不管 MACD 是多少都买
        verbose=True
    )

    def __init__(self):
        self.entry_price = {}
        self.entry_dt = {}
        self.inds = {}
        
        # 为每只股票初始化指标
        for d in self.datas:
            self.inds[d] = {
                # RSI 周期 14
                'rsi': bt.indicators.RSI(d.close, period=14),
                # MACD 标准参数
                'macd': bt.indicators.MACD(d.close)
            }

    def log(self, txt, dt=None):
        if self.p.verbose:
            # 尝试获取当前时间
            try:
                dt = dt or self.datas[0].datetime.date(0)
                print(f'[{dt}] {txt}')
            except:
                print(f'[Unknown Date] {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        dt = self.datas[0].datetime.date(0)
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🔵 买入成功: {order.data._name}, 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}', dt)
                self.entry_price[order.data._name] = order.executed.price
                self.entry_dt[order.data._name] = dt
            elif order.issell():
                gross_pnl = order.executed.pnl
                net_pnl = order.executed.pnlcomm
                self.log(f'🔴 卖出成功: {order.data._name}, 价格: {order.executed.price:.2f}, 毛利: {gross_pnl:.2f}', dt)
                
                if order.data._name in self.entry_price:
                    del self.entry_price[order.data._name]
                    del self.entry_dt[order.data._name]
                    
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'❌ 订单被拒 (原因: 可能是资金不足): {order.data._name}', dt)

    def next(self):
        # 遍历每一只股票
        for d in self.datas:
            name = d._name
            pos = self.getposition(d).size
            current_date = d.datetime.date(0)
            
            # --- 1. 获取指标值 ---
            # 注意：在数据刚开始的几十天，RSI 和 MACD 是算不出来的（是 NaN），这时候不能交易
            rsi_val = self.inds[d]['rsi'][0]
            macd_val = self.inds[d]['macd'].macd[0]
            
            # 检查是否为有效数字
            if math.isnan(rsi_val) or math.isnan(macd_val):
                continue # 指标还没算出来（数据太少），跳过今天
            
            # --- 2. 买入逻辑 (宽松版) ---
            if pos == 0:
                # 因为 params.rsi_max 设为了 100，这里几乎永远是 True
                condition_rsi = rsi_val < self.p.rsi_max
                condition_macd = macd_val > self.p.macd_min
                
                if condition_rsi and condition_macd:
                    # 检查现金是否足够
                    cash = self.broker.getcash()
                    # 预估大概需要多少钱 (股价 * 100股)
                    cost = d.close[0] * self.p.stake
                    
                    if cash > cost * 1.1: # 留一点余量防止滑点
                        # self.log(f"触发信号 {name}: RSI={rsi_val:.1f}, MACD={macd_val:.2f} -> 买入")
                        self.buy(data=d, size=self.p.stake)

            # --- 3. 卖出逻辑 ---
            elif pos > 0:
                entry = self.entry_price.get(name, d.close[0])
                buy_date = self.entry_dt.get(name, None)
                cur_price = d.close[0]
                
                # 止损
                if cur_price <= entry * (1 - self.p.stoploss_pct):
                    self.log(f"📉 止损平仓: {name} ({cur_price} < {entry})")
                    self.close(data=d)
                
                # 止盈
                elif cur_price >= entry * (1 + self.p.takeprofit_pct):
                    self.log(f"🚀 止盈平仓: {name} ({cur_price} > {entry})")
                    self.close(data=d)
                
                # 持仓时间到了
                elif buy_date and (current_date - buy_date).days >= self.p.hold_days:
                    self.log(f"⏱️ 到期平仓: {name} 持有超过 {self.p.hold_days} 天")
                    self.close(data=d)
