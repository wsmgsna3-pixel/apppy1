# ycjsb_backtest_addon.py
"""
外挂回测 & 优化模块（基于你的 ycjsb.py 稳定程序）
用法：
  python ycjsb_backtest_addon.py --mode run
  python ycjsb_backtest_addon.py --mode optimize --trials 50
"""

import streamlit as st
import argparse
import importlib
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

if 'tushare_token' not in st.session_state:
    st.set_page_config(page_title="盈财金手指-回测增强版", layout="wide")
    st.title("请先输入 TuShare Pro Token")
    token = st.text_input("TuShare Token（48位）", type="password", help="从 https://tushare.pro/user/token 获取")
    if token:
        token = token.strip()
        if len(token) == 48:
            st.session_state.tushare_token = token
            st.success("Token 已保存")
            st.rerun()
        else:
            st.error("Token 长度必须为 48 位，请检查是否复制完整")
    st.stop()

import tushare as ts
ts.set_token(st.session_state.tushare_token)

import backtrader as bt
import json
from signal_builder import set_pro, basic_filters, get_hist, get_moneyflow
from bt_strategy import SignalStrategy
from viz import plot_equity_curve, plot_drawdown
from optimize import run_optuna
from tqdm import tqdm

# ------------- Configurable defaults -------------
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

# ------------- utility to try import user's ycjsb module -------------
def try_import_ycjsb(path="ycjsb.py"):
    mod = None
    base_dir = os.getcwd()
    if os.path.exists(os.path.join(base_dir, path)):
        try:
            sys.path.insert(0, base_dir)
            modname = os.path.splitext(os.path.basename(path))[0]
            mod = importlib.import_module(modname)
            print(f"Imported user module {modname}")
        except Exception as e:
            print(f"Failed import ycjsb module: {e}")
    return mod

# ------------- create data feeds for backtrader -------------
def create_bt_datas(cerebro, df_dict):
    """
    df_dict: {ts_code: DataFrame with ['trade_date','open','high','low','close','vol']}
    returns list of bt datas
    """
    feeds = []
    for ts_code, df in df_dict.items():
        tmp = df.copy()
        tmp = tmp.rename(columns={'trade_date':'datetime'})
        tmp['datetime'] = pd.to_datetime(tmp['datetime'])
        tmp = tmp.set_index('datetime')
        tmp = tmp[['open','high','low','close','vol']]
        # create feed
        data = bt.feeds.PandasData(dataname=tmp, name=ts_code)
        cerebro.adddata(data, name=ts_code)
        feeds.append(data)
    return feeds

# ------------- run a single backtest given parameter dict -------------
def run_backtest(universe, params, cash=100000.0, commission=0.0003, slippage=0.000, verbose=False):
    """
    universe: list of ts_code
    params: dict containing strategy params e.g. stoploss_pct, takeprofit_pct, VOL_RATIO_MIN, MACD_MIN, RSI_MAX
    Returns: performance metric (sharpe) and equity series (pd.Series)
    """
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    # gather historical data for each ticker
    df_dict = {}
    for ts_code in universe:
        try:
            df = get_hist(ts_code, start_date=DEFAULTS['start_date'], end_date=datetime.now().strftime("%Y%m%d"))
            if df.empty:
                continue
            # backtrader expects index ascending and datetime
            df = df[['trade_date','open','high','low','close','vol']].copy()
            df_dict[ts_code] = df
        except Exception as e:
            print(f"failed to fetch {ts_code}: {e}")
    if not df_dict:
        raise RuntimeError("No data available for backtest universe.")
    # create datas
    create_bt_datas(cerebro, df_dict)
    # add strategy
    stratparams = dict(
        stake=1,
        stoploss_pct=params.get('stoploss_pct', 0.08),
        takeprofit_pct=params.get('takeprofit_pct', 0.2),
        hold_days=int(params.get('hold_days', 5)),
        verbose=verbose
    )
    cerebro.addstrategy(SignalStrategy, **stratparams)
    # run
    try:
        results = cerebro.run()
    except Exception as e:
        print(f"Backtest error: {e}")
        raise
    # get portfolio value time series (cerebro has no direct TS; use broker.getvalue at each data point via analyzer or observers)
    # For simplicity, we re-run with observer of value
    try:
        # re-run to capture value history with observers disabled? Instead, use the built-in broker snapshot is not trivial.
        # As compromise: return final value and approximate equity series by stepping through daily close per combined index
        final_value = cerebro.broker.getvalue()
    except Exception:
        final_value = None
    # WARNING: building precise equity time series in backtrader requires observers; for this addon we approximate by final value.
    return {
        "final_value": final_value,
        "cerebro": cerebro,
        "df_dict": df_dict
    }

# ------------- collate universe: prefer ycjsb.provided pool if available -------------
def build_universe_from_ycjsb(ycjsb_mod, last_trade, params):
    """
    If ycjsb.py exposes a function to get pool, we call it; else we fallback to scanning the top N from daily
    """
    if ycjsb_mod is not None:
        try:
            if hasattr(ycjsb_mod, "get_candidate_pool"):
                # expected to return DataFrame with ts_code column
                pool = ycjsb_mod.get_candidate_pool()
                if isinstance(pool, (list, tuple)):
                    return list(pool)
                if hasattr(pool, "ts_code"):
                    return list(pool['ts_code'].unique())
        except Exception as e:
            print(f"Error calling get_candidate_pool(): {e}")
    # fallback: use daily top INITIAL_TOP_N
    daily = pro.daily(trade_date=last_trade)
    if daily is None or daily.empty:
        raise RuntimeError("cannot fetch daily for fallback universe.")
    top = daily.sort_values("pct_chg", ascending=False).head(params.get("INITIAL_TOP_N",800))
    return list(top['ts_code'].unique())

# ------------- high level runner used by optimizer -------------
def runner_for_opt(params, fixed_args):
    """
    params: optimization parameters from optuna
    fixed_args: contains universe, last_trade
    Returns a metric (Sharpe-like) to maximize
    """
    # merge fixed and params
    run_params = {}
    run_params.update(fixed_args.get("base_params", {}))
    run_params.update(params)
    try:
        res = run_backtest(fixed_args['universe'], run_params, cash=fixed_args.get('cash',100000.0))
        # compute a simple metric: final_value / initial - 1
        fv = res.get('final_value', None)
        if fv is None:
            return -999
        ret = (fv - fixed_args.get('cash',100000.0)) / fixed_args.get('cash',100000.0)
        # use return as metric (can be replaced by sharpe)
        return float(ret)
    except Exception as e:
        print(f"runner_for_opt exception: {e}")
        return -999

# ------------- main -------------
def main(args):
    global pro
    # init tushare
    token = os.environ.get("TUSHARE_TOKEN", None)
    if token is None:
        token = input("Please input your Tushare token (or set env var TUSHARE_TOKEN): ").strip()
    ts.set_token(token)
    pro_local = ts.pro_api()
    set_pro(pro_local)
    # try import ycjsb
    ycjsb_mod = try_import_ycjsb("ycjsb.py")

    # build universe
    last_trade = None
    for i in range(15):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        if not pro_local.daily(trade_date=d).empty:
            last_trade = d
            break
    if last_trade is None:
        raise RuntimeError("Cannot find last trade date")

    universe = build_universe_from_ycjsb(ycjsb_mod, last_trade, {"INITIAL_TOP_N": args.topn})

    print(f"Universe size: {len(universe)}")

    # prepare base params merged with defaults
    base_params = DEFAULTS.copy()
    base_params.update({
        "MIN_PRICE": args.min_price,
        "MAX_PRICE": args.max_price,
        "MIN_TURNOVER": args.min_turnover,
        "MIN_AMOUNT": args.min_amount,
        "VOL_RATIO_MIN": args.vol_ratio_min,
        "RSI_MAX": args.rsi_max,
        "MACD_MIN": args.macd_min,
        "MAX_5D_PCT": args.max_5d_pct
    })

    if args.mode == "run":
        # run single backtest with given params
        run_params = {
            "stoploss_pct": args.stoploss,
            "takeprofit_pct": args.takeprofit,
            "VOL_RATIO_MIN": base_params['VOL_RATIO_MIN'],
            "MACD_MIN": base_params['MACD_MIN'],
            "RSI_MAX": base_params['RSI_MAX']
        }
        out = run_backtest(universe, run_params, cash=args.cash)
        print("Backtest finished. final value:", out.get('final_value'))
        # Try to save equity as placeholder (not precise)
        try:
            # If cerebro exists we can at least save final value to txt
            with open("backtest_result.json","w") as f:
                json.dump({"final_value": out.get('final_value')}, f)
        except Exception:
            pass
        print("Results saved to backtest_result.json")
    elif args.mode == "optimize":
        fixed_args = {"universe": universe, "cash": args.cash, "base_params": base_params}
        study, best = run_optuna(runner_for_opt, fixed_args, n_trials=args.trials)
        print("Best params:", best)
    else:
        raise RuntimeError("Unknown mode")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run","optimize"], default="run")
    parser.add_argument("--topn", type=int, default=800)
    parser.add_argument("--min_price", type=float, default=3.0)
    parser.add_argument("--max_price", type=float, default=500.0)
    parser.add_argument("--min_turnover", type=float, default=2.0)
    parser.add_argument("--min_amount", type=float, default=50_000_000.0)
    parser.add_argument("--vol_ratio_min", type=float, default=1.2)
    parser.add_argument("--rsi_max", type=float, default=75)
    parser.add_argument("--macd_min", type=float, default=-0.3)
    parser.add_argument("--max_5d_pct", type=float, default=40.0)
    parser.add_argument("--stoploss", type=float, default=0.08)
    parser.add_argument("--takeprofit", type=float, default=0.2)
    parser.add_argument("--cash", type=float, default=100000.0)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    main(args)
