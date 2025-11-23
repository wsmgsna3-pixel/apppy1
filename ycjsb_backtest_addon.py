# ycjsb_backtest_addon.py
"""
外挂回测 & 优化模块（Streamlit GUI 修复版）
修复内容：
1. 增加 .sort_index() 解决 Tushare 数据时间倒序导致回测秒停的问题。
2. 映射 vol 列名为 volume，确保 Backtrader 能读到成交量。
"""

import importlib
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tushare as ts
import backtrader as bt
import json
import streamlit as st

# 尝试导入用户模块
from signal_builder import set_pro, basic_filters, get_hist, get_moneyflow
from bt_strategy import SignalStrategy
from viz import plot_equity_curve, plot_drawdown
from optimize import run_optuna
from tqdm import tqdm

# ------------- 页面基础配置 -------------
st.set_page_config(page_title="选股回测工具", layout="wide")

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
        except Exception as e:
            st.error(f"加载 ycjsb 模块失败: {e}")
    return mod

# ------------- create data feeds for backtrader -------------
def create_bt_datas(cerebro, df_dict):
    feeds = []
    for ts_code, df in df_dict.items():
        # 1. 复制数据
        tmp = df.copy()
        
        # 2. 重命名列 (trade_date -> datetime, vol -> volume)
        tmp = tmp.rename(columns={
            'trade_date': 'datetime',
            'vol': 'volume' # 关键：Backtrader 默认识别 volume
        })
        
        # 3. 设置时间索引
        tmp['datetime'] = pd.to_datetime(tmp['datetime'])
        tmp = tmp.set_index('datetime')
        
        # 4. 【核心修复】强制按时间正序排列 (Old -> New)
        tmp = tmp.sort_index(ascending=True)
        
        # 5. 确保包含所需的列
        # 如果数据中没有 openinterest，补0，防止报错
        if 'openinterest' not in tmp.columns:
            tmp['openinterest'] = 0
            
        tmp = tmp[['open', 'high', 'low', 'close', 'volume', 'openinterest']]

        # 6. 创建 Feed
        data = bt.feeds.PandasData(dataname=tmp, name=ts_code)
        cerebro.adddata(data, name=ts_code)
        feeds.append(data)
    return feeds

# ------------- run a single backtest -------------
def run_backtest(universe, params, cash=100000.0, commission=0.0003, slippage=0.000, verbose=False):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    
    df_dict = {}
    progress_text = "正在获取历史数据..."
    my_bar = st.progress(0, text=progress_text)
    
    total_stocks = len(universe)
    
    # 获取数据的逻辑
    for i, ts_code in enumerate(universe):
        try:
            # 限制进度条刷新频率，避免卡顿
            if i % 5 == 0 or i == total_stocks - 1:
                my_bar.progress((i + 1) / total_stocks, text=f"获取数据 ({i+1}/{total_stocks}): {ts_code}")
            
            # 获取最近 1.5 年的数据，确保指标有足够的计算周期
            start_dt = (datetime.now() - timedelta(days=500)).strftime("%Y%m%d")
            df = get_hist(ts_code, start_date=start_dt, end_date=datetime.now().strftime("%Y%m%d"))
            
            if df is None or df.empty:
                continue
            
            # 简单校验数据行数，太少无法计算指标 (MACD至少需要35天)
            if len(df) < 50:
                continue

            df = df[['trade_date','open','high','low','close','vol']].copy()
            df_dict[ts_code] = df
        except Exception as e:
            print(f"failed to fetch {ts_code}: {e}")
            
    my_bar.empty() 

    if not df_dict:
        st.error("没有获取到任何有效的历史数据。请检查网络或Token。")
        return {"final_value": cash}
    
    # 这里的 create_bt_datas 已经包含了 sort_index 修复
    create_bt_datas(cerebro, df_dict)
    
    # 策略参数：确保 stake 足够大，或者在策略内部处理
    # 注意：这里我们传入 stake=100，表示每次买100股
    stratparams = dict(
        stake=100, 
        stoploss_pct=params.get('stoploss_pct', 0.08),
        takeprofit_pct=params.get('takeprofit_pct', 0.2),
        hold_days=int(params.get('hold_days', 5)),
        verbose=verbose
    )
    cerebro.addstrategy(SignalStrategy, **stratparams)
    
    try:
        st.info(f"数据加载完成 (共 {len(df_dict)} 只)，开始回测计算...")
        results = cerebro.run()
    except Exception as e:
        st.error(f"Backtest error: {e}")
        raise
    
    try:
        final_value = cerebro.broker.getvalue()
    except Exception:
        final_value = cash

    return {
        "final_value": final_value,
        "cerebro": cerebro,
        "df_dict": df_dict
    }

# ------------- collate universe -------------
def build_universe_from_ycjsb(ycjsb_mod, last_trade, params, pro_api):
    if ycjsb_mod is not None:
        try:
            if hasattr(ycjsb_mod, "get_candidate_pool"):
                pool = ycjsb_mod.get_candidate_pool()
                if isinstance(pool, (list, tuple)):
                    return list(pool)
                if hasattr(pool, "ts_code"):
                    return list(pool['ts_code'].unique())
        except Exception as e:
            st.warning(f"调用 get_candidate_pool() 出错: {e}，将使用默认Top N策略")
            
    daily = pro_api.daily(trade_date=last_trade)
    if daily is None or daily.empty:
        raise RuntimeError("无法获取每日行情数据 (daily data unavailable).")
    top = daily.sort_values("pct_chg", ascending=False).head(params.get("INITIAL_TOP_N",800))
    return list(top['ts_code'].unique())

# ------------- GUI 主程序 -------------
def main_gui():
    st.title("📈 选股回测系统 (修复版)")
    
    # --- 1. 安全输入 Token ---
    with st.expander("🔐 Tushare Token 设置 (必填)", expanded=True):
        token_input = st.text_input(
            "请输入你的 Tushare Token", 
            type="password", 
            help="你的Token不会被保存，仅用于本次运行"
        )
    
    if not token_input:
        st.warning("👉 请在上框中输入 Tushare Token 以启用系统。")
        st.stop()

    try:
        ts.set_token(token_input)
        pro_local = ts.pro_api()
        set_pro(pro_local) 
    except Exception as e:
        st.error(f"Token 设置失败: {e}")
        st.stop()

    # --- 2. 侧边栏参数设置 ---
    st.sidebar.header("⚙️ 回测参数")
    
    mode = st.sidebar.selectbox("运行模式", ["单次运行 (Run)"])
    
    st.sidebar.subheader("筛选条件")
    topn = st.sidebar.number_input("每日候选池大小", value=50, help="为了速度，建议先设小一点(如50)测试")
    
    st.sidebar.subheader("交易策略")
    cash = st.sidebar.number_input("初始资金", value=100000.0)
    stoploss = st.sidebar.number_input("止损百分比 (0.08 = 8%)", value=0.08, step=0.01)
    takeprofit = st.sidebar.number_input("止盈百分比 (0.2 = 20%)", value=0.2, step=0.01)

    # --- 3. 运行逻辑 ---
    if st.button("🚀 开始运行", type="primary"):
        
        base_params = DEFAULTS.copy()
        ycjsb_mod = try_import_ycjsb("ycjsb.py")

        with st.spinner("正在获取最新交易日期..."):
            last_trade = None
            for i in range(15):
                d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                try:
                    if not pro_local.daily(trade_date=d).empty:
                        last_trade = d
                        break
                except Exception:
                    pass
            
            if last_trade is None:
                st.error("无法连接 Tushare 获取日期，请检查网络。")
                st.stop()

        with st.spinner(f"正在构建股票池..."):
            try:
                # 传入 topn 参数
                universe = build_universe_from_ycjsb(ycjsb_mod, last_trade, {"INITIAL_TOP_N": int(topn)}, pro_local)
                # 为了手机上跑得快，如果 universe 太多，可以截断，或者全跑
                st.success(f"股票池构建完成，共包含 {len(universe)} 只股票")
            except Exception as e:
                st.error(f"构建股票池失败: {e}")
                st.stop()

        # 执行回测
        run_params = {
            "stoploss_pct": stoploss,
            "takeprofit_pct": takeprofit,
            "hold_days": 5
        }
        
        try:
            out = run_backtest(universe, run_params, cash=cash, verbose=True)
            
            final_val = out.get('final_value')
            profit = final_val - cash
            ret_pct = (profit / cash) * 100
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("初始资金", f"{cash:,.0f}")
            c2.metric("最终资金", f"{final_val:,.2f}")
            c3.metric("收益率", f"{ret_pct:.2f}%", delta=f"{profit:,.2f}")
            
            if profit == 0:
                st.warning("收益率仍为 0？可能是最近行情不满足你的 RSI/MACD 开仓条件。")
            else:
                st.balloons()
            
            st.json({"Status": "Finished", "Final Value": final_val})
            
        except Exception as e:
            st.error(f"回测运行出错: {e}")

if __name__ == "__main__":
    main_gui()
