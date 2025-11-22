# viz.py
# 可视化模块：绘制资金曲线、回撤、交易点

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_equity_curve(equity_ts: pd.Series, title="Equity Curve", out_path="equity.png"):
    plt.figure(figsize=(10,5))
    plt.plot(equity_ts.index, equity_ts.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def plot_drawdown(equity_ts: pd.Series, title="Drawdown", out_path="drawdown.png"):
    # drawdown series
    hwm = equity_ts.cummax()
    dd = (equity_ts - hwm) / (hwm + 1e-9)
    plt.figure(figsize=(10,4))
    plt.fill_between(dd.index, dd.values, color='red')
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def plot_trades(price_df: pd.DataFrame, trades: pd.DataFrame, out_path="trades.png", title="Trades"):
    """
    price_df: DataFrame with index date and columns ['close']
    trades: DataFrame with columns ['date', 'ts_code', 'size', 'price', 'side'] side in ['buy','sell']
    """
    plt.figure(figsize=(12,6))
    plt.plot(price_df.index, price_df['close'].values, label='price')
    buys = trades[trades['side']=='buy']
    sells = trades[trades['side']=='sell']
    if not buys.empty:
        plt.scatter(pd.to_datetime(buys['date']), buys['price'], marker='^', color='g', label='buy')
    if not sells.empty:
        plt.scatter(pd.to_datetime(sells['date']), sells['price'], marker='v', color='r', label='sell')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
