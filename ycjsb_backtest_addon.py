# ycjsb_backtest_addon.py
"""
盈财金手指 - 回测增强版（Streamlit 专属版）
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import importlib

# ==================== 1. Token 输入（56位） ====================
if 'tushare_token' not in st.session_state:
    st.set_page_config(page_title="盈财金手指 - 回测增强版", layout="wide")
    st.title("请先输入 TuShare Pro Token")
    st.markdown("最新 Token 长度为 **56 位**，请到 https://tushare.pro/user/token 获取")

    token = st.text_input(
        "粘贴你的 TuShare Token",
        type="password",
        placeholder="56 位完整 Token",
        help="输入正确后自动保存并进入主程序"
    )

    if token:
        token = token.strip()
        if len(token) == 56:
            st.session_state.tushare_token = token
            st.success("Token 保存成功！正在加载程序…")
            st.rerun()
        else:
            st.error(f"Token 长度错误：当前 {len(token)} 位，必须为 56 位")
    st.stop()

# ==================== 2. Token 已确认，安全初始化 ====================
import tushare as ts
ts.set_token(st.session_state.tushare_token)
pro = ts.pro_api()

# 防止 signal_builder 里旧的 set_pro 再出问题
from signal_builder import basic_filters, get_hist, get_moneyflow
def set_pro(*args, **kwargs): pass  # 空函数屏蔽

import backtrader as bt
from bt_strategy import SignalStrategy
from viz import plot_equity_curve, plot_drawdown
from optimize import run_optuna
from tqdm import tqdm

# ==================== 3. 基础配置 ====================
DEFAULTS = {
    "MIN_PRICE": 3.0, "MAX_PRICE": 500.0, "MIN_TURNOVER": 2.0,
    "MIN_AMOUNT": 50_000_000.0, "VOL_RATIO_MIN": 1.2, "RSI_MAX": 75,
    "MACD_MIN": -0.3, "MAX_5D_PCT": 40,
    "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
}

# ==================== 4. 辅助函数 ====================
def try_import_ycjsb():
    if os.path.exists("ycjsb.py"):
        try:
            sys.path.insert(0, os.getcwd())
            return importlib.import_module("ycjsb")
        except:
            return None
    return None

def create_bt_datas(cerebro, df_dict):
    for ts_code, df in df_dict.items():
        df = df.copy()
        df = df.rename(columns={'trade_date': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')[['open','high','low','close','vol']]
        data = bt.feeds.PandasData(dataname=df, name=ts_code)
        cerebro.adddata(data, name=ts_code)

def run_backtest(universe, params, cash=100000.0):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0003)

    df_dict = {}
    for ts_code in universe[:100]:  # 防止超时，先取前100只
        try:
            df = get_hist(ts_code, start_date=DEFAULTS['start_date'])
            if not df.empty:
                df = df[['trade_date','open','high','low','close','vol']]
                df_dict[ts_code] = df
        except:
            continue

    if not df_dict:
        st.error("所有股票均无数据（可能积分不足）")
        return None

    create_bt_datas(cerebro, df_dict)
    cerebro.addstrategy(SignalStrategy,
                        stoploss_pct=params.get("stoploss", 0.08),
                        takeprofit_pct=params.get("takeprofit", 0.20),
                        hold_days=5)

    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    return {"final_value": final_value, "cerebro": cerebro, "df_dict": df_dict}

# ==================== 5. Streamlit 主界面 ====================
st.set_page_config(page_title="盈财金手指 - 回测增强版", layout="wide")
st.title("盈财金手指 - 回测增强版")

st.sidebar.header("回测参数")
topn = st.sidebar.slider("初始选股数量", 100, 2000, 800)
cash = st.sidebar.number_input("初始资金（元）", 50000.0, 10000000.0, 100000.0, step=10000.0)

if st.sidebar.button("开始回测", type="primary"):
    with st.spinner("正在构建股票池…"):
        # 找最近交易日
        last_trade = None
        for i in range(20):
            d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            if not pro.daily(trade_date=d).empty:
                last_trade = d
                break

        # 优先用 ycjsb.py 的池子
        ycjsb_mod = try_import_ycjsb()
        universe = []
        if ycjsb_mod and hasattr(ycjsb_mod, "get_candidate_pool"):
            try:
                pool = ycjsb_mod.get_candidate_pool()
                universe = pd.DataFrame(pool)['ts_code'].unique().tolist()
            except:
                pass

        if len(universe) < 50:
            daily = pro.daily(trade_date=last_trade)
            top = daily.sort_values("pct_chg", ascending=False).head(topn)
            universe = top['ts_code'].tolist()

        st.success(f"股票池构建完成，共 {len(universe)} 只股票")

    with st.spinner("正在回测（可能需要 30-90 秒）…"):
        result = run_backtest(universe, {}, cash=cash)
        if result:
            st.success(f"回测完成！最终资产：**{result['final_value']:,.2f} 元**")
            st.balloons()

# 防止有人直接 python 运行
if __name__ == "__main__":
    sys.exit("请通过 Streamlit 访问此程序，不要直接运行此文件")
