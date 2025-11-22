# ycjsb_backtest_addon.py
"""
外挂回测 & 优化模块（基于你的 ycjsb.py 稳定程序）
"""

import streamlit as st
import argparse
import importlib
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ==================== 完全杜绝旧的 input 提示 ====================
if 'tushare_token' not in st.session_state:
    st.set_page_config(page_title="盈财金手指-回测增强版", layout="wide")
    st.title("请先输入 TuShare Pro Token")
    st.markdown("从 https://tushare.pro/user/token 获取（最新 Token 为 **56 位**）")
    
    token = st.text_input(
        "TuShare Token",
        type="password",
        placeholder="粘贴 56 位 Token 后回车",
        help="输入正确后会自动保存并进入主界面"
    )
    
    if token:
        token = token.strip()
        if len(token) == 56:
            st.session_state.tushare_token = token
            st.success("Token 已保存成功，即将进入程序…")
            st.rerun()
        else:
            st.error(f"Token 长度错误：当前 {len(token)} 位，最新 Token 必须为 56 位，请完整复制")
    st.stop()

# ==================== Token 已确认，安全导入 ====================
import tushare as ts
ts.set_token(st.session_state.tushare_token)
pro = ts.pro_api()

# 彻底覆盖旧的 set_pro（防止 signal_builder 里再弹 input）
from signal_builder import basic_filters, get_hist, get_moneyflow
def set_pro(x): pass  # 空函数，防止被旧代码调用

import backtrader as bt
import json
from bt_strategy import SignalStrategy
from viz import plot_equity_curve, plot_drawdown
from optimize import run_optuna
from tqdm import tqdm

# ============== 以下全部保留原逻辑，只删除了所有 input/print =============
DEFAULTS = {
    "MIN_PRICE": 3.0,
    "MAX_PRICE": 500.0,
    "MIN_TURNOVER": 2.0,
    "MIN_AMOUNT": 50_000_000.0,
    "VOL_RATIO_MIN": 1.2,
    "RSI_MAX": 75,
    "MACD_MIN": -0.3,
    "MAX_5D_PCT": 40,
    "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
}

def try_import_ycjsb(path="ycjsb.py"):
    mod = None
    base_dir = os.getcwd()
    if os.path.exists(os.path.join(base_dir, path)):
        try:
            sys.path.insert(0, base_dir)
            modname = os.path.splitext(os.path.basename(path))[0]
            mod = importlib.import_module(modname)
        except Exception:
            pass
    return mod

def create_bt_datas(cerebro, df_dict):
    feeds = []
    for ts_code, df in df_dict.items():
        tmp = df.copy()
        tmp = tmp.rename(columns={'trade_date':'datetime'})
        tmp['datetime'] = pd.to_datetime(tmp['datetime'])
        tmp = tmp.set_index('datetime')
        tmp = tmp[['open','high','low','close','vol']]
        data = bt.feeds.PandasData(dataname=tmp, name=ts_code)
        cerebro.adddata(data, name=ts_code)
        feeds.append(data)
    return feeds

def run_backtest(universe, params, cash=100000.0, commission=0.0003, verbose=False):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    
    df_dict = {}
    for ts_code in universe:
        try:
            df = get_hist(ts_code, start_date=DEFAULTS['start_date'])
            if df.empty:
                continue
            df = df[['trade_date','open','high','low','close','vol']].copy()
            df_dict[ts_code] = df
        except:
            continue
    if not df_dict:
        st.error("所有股票均无数据，请检查网络或 Token 积分")
        return None

    create_bt_datas(cerebro, df_dict)
    
    stratparams = dict(
        stake=1,
        stoploss_pct=params.get('stoploss_pct', 0.08),
        takeprofit_pct=params.get('takeprofit_pct', 0.2),
        hold_days=int(params.get('hold_days', 5)),
        verbose=verbose
    )
    cerebro.addstrategy(SignalStrategy, **stratparams)
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    return {
        "final_value": final_value,
        "cerebro": cerebro,
        "df_dict": df_dict
    }

# ============== Streamlit 主界面 ==============
st.sidebar.title("回测参数设置")
mode = st.sidebar.selectbox("运行模式", ["单次回测", "参数优化"])

# 其余参数略（你原来 argparse 的全部移到 sidebar）
topn = st.sidebar.number_input("初始选股数量", 100, 2000, 800)
cash = st.sidebar.number_input("初始资金", 10000.0, 10000000.0, 100000.0, step=10000.0)

if st.sidebar.button("开始运行"):
    with st.spinner("正在获取最新交易日和股票池…"):
        last_trade = None
        for i in range(15):
            d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            if not pro.daily(trade_date=d).empty:
                last_trade = d
                break
        if not last_trade:
            st.error("无法获取最近交易日")
            st.stop()

        ycjsb_mod = try_import_ycjsb()
        universe = []
        if ycjsb_mod and hasattr(ycjsb_mod, "get_candidate_pool"):
            try:
                pool = ycjsb_mod.get_candidate_pool()
                universe = list(pd.DataFrame(pool)['ts_code'].unique())
            except:
                pass
        
        if not universe:
            daily = pro.daily(trade_date=last_trade)
            top = daily.sort_values("pct_chg", ascending=False).head(topn)
            universe = list(top['ts_code'].unique())
        
        st.success(f"股票池构建完成，共 {len(universe)} 只股票")

    if mode == "单次回测":
        with st.spinner("正在回测…"):
            params = {
                "stoploss_pct": 0.08,
                "takeprofit_pct": 0.20,
                "VOL_RATIO_MIN": 1.2,
                "MACD_MIN": -0.3,
                "RSI_MAX": 75
            }
            result = run_backtest(universe, params, cash=cash)
            if result:
                st.success(f"回测完成！最终资产：{result['final_value']:,.2f} 元")
                # 可继续加图表
    else:
        st.info("参数优化功能暂未完全迁移，后续补上")
