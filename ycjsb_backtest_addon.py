# ycjsb_backtest_addon.py
"""
外挂回测 & 优化模块（Streamlit GUI 版本）
用法：
  直接在终端运行: streamlit run ycjsb_backtest_addon.py
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
import streamlit as st # 引入 Streamlit

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
            # st.success(f"成功加载用户模块: {modname}") # UI提示太频繁可注释
        except Exception as e:
            st.error(f"加载 ycjsb 模块失败: {e}")
    return mod

# ------------- create data feeds for backtrader -------------
def create_bt_datas(cerebro, df_dict):
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

# ------------- run a single backtest -------------
def run_backtest(universe, params, cash=100000.0, commission=0.0003, slippage=0.000, verbose=False):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    
    df_dict = {}
    # 显示进度条
    progress_text = "正在获取历史数据..."
    my_bar = st.progress(0, text=progress_text)
    
    total_stocks = len(universe)
    for i, ts_code in enumerate(universe):
        try:
            # 更新进度条
            my_bar.progress((i + 1) / total_stocks, text=f"获取数据: {ts_code}")
            
            df = get_hist(ts_code, start_date=DEFAULTS['start_date'], end_date=datetime.now().strftime("%Y%m%d"))
            if df.empty:
                continue
            df = df[['trade_date','open','high','low','close','vol']].copy()
            df_dict[ts_code] = df
        except Exception as e:
            print(f"failed to fetch {ts_code}: {e}")
            
    my_bar.empty() # 清除进度条

    if not df_dict:
        raise RuntimeError("没有获取到任何回测数据，请检查Token或网络。")
    
    create_bt_datas(cerebro, df_dict)
    
    stratparams = dict(
        stake=1,
        stoploss_pct=params.get('stoploss_pct', 0.08),
        takeprofit_pct=params.get('takeprofit_pct', 0.2),
        hold_days=int(params.get('hold_days', 5)),
        verbose=verbose
    )
    cerebro.addstrategy(SignalStrategy, **stratparams)
    
    try:
        st.info("开始执行 Backtrader 回测...")
        results = cerebro.run()
    except Exception as e:
        st.error(f"Backtest error: {e}")
        raise
    
    try:
        final_value = cerebro.broker.getvalue()
    except Exception:
        final_value = None

    return {
        "final_value": final_value,
        "cerebro": cerebro,
        "df_dict": df_dict
    }

# ------------- collate universe -------------
def build_universe_from_ycjsb(ycjsb_mod, last_trade, params, pro_api):
    """
    构建股票池
    """
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
            
    # fallback
    daily = pro_api.daily(trade_date=last_trade)
    if daily is None or daily.empty:
        raise RuntimeError("无法获取每日行情数据 (daily data unavailable).")
    top = daily.sort_values("pct_chg", ascending=False).head(params.get("INITIAL_TOP_N",800))
    return list(top['ts_code'].unique())

# ------------- GUI 主程序 -------------
def main_gui():
    st.title("📈 选股回测系统 (Secure Mode)")
    
    st.markdown("""
    此界面允许你安全地输入 Tushare Token 进行回测。
    Token 仅保存在当前会话内存中，刷新页面即清除。
    """)

    # --- 1. 安全输入 Token ---
    with st.expander("🔐 Tushare Token 设置 (必填)", expanded=True):
        token_input = st.text_input(
            "请输入你的 Tushare Token", 
            type="password", 
            help="你的Token不会被保存，仅用于本次运行"
        )
    
    if not token_input:
        st.warning("👉 请在上框中输入 Tushare Token 以启用系统。")
        st.stop() # 停止执行后续代码，直到用户输入

    # 初始化 Tushare
    try:
        ts.set_token(token_input)
        pro_local = ts.pro_api()
        set_pro(pro_local) # 设置全局 pro
    except Exception as e:
        st.error(f"Token 设置失败: {e}")
        st.stop()

    # --- 2. 侧边栏参数设置 ---
    st.sidebar.header("⚙️ 回测参数设置")
    
    # 模式选择
    mode = st.sidebar.selectbox("运行模式", ["单次运行 (Run)", "参数优化 (Optimize)"])
    
    st.sidebar.subheader("筛选条件")
    topn = st.sidebar.number_input("每日候选池大小 (Top N)", value=800, step=50)
    min_price = st.sidebar.number_input("最低股价", value=3.0)
    max_price = st.sidebar.number_input("最高股价", value=500.0)
    min_turnover = st.sidebar.number_input("最低换手率", value=2.0)
    
    st.sidebar.subheader("技术指标")
    vol_ratio_min = st.sidebar.number_input("最小量比", value=1.2)
    rsi_max = st.sidebar.number_input("RSI 上限", value=75.0)
    
    st.sidebar.subheader("交易策略")
    cash = st.sidebar.number_input("初始资金", value=100000.0)
    stoploss = st.sidebar.number_input("止损百分比 (0.08 = 8%)", value=0.08, step=0.01)
    takeprofit = st.sidebar.number_input("止盈百分比 (0.2 = 20%)", value=0.2, step=0.01)

    # --- 3. 运行逻辑 ---
    
    # 只有点击按钮才开始运行
    if st.button("🚀 开始运行", type="primary"):
        
        # 准备参数
        base_params = DEFAULTS.copy()
        base_params.update({
            "MIN_PRICE": min_price,
            "MAX_PRICE": max_price,
            "MIN_TURNOVER": min_turnover,
            "VOL_RATIO_MIN": vol_ratio_min,
            "RSI_MAX": rsi_max,
        })

        # 尝试导入 ycjsb
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
                st.error("无法连接 Tushare 获取日期，请检查 Token 是否正确或已过期。")
                st.stop()

        with st.spinner(f"正在构建股票池 (基准日期: {last_trade})..."):
            try:
                universe = build_universe_from_ycjsb(ycjsb_mod, last_trade, {"INITIAL_TOP_N": topn}, pro_local)
                st.success(f"股票池构建完成，共包含 {len(universe)} 只股票")
            except Exception as e:
                st.error(f"构建股票池失败: {e}")
                st.stop()

        # 执行模式
        if mode == "单次运行 (Run)":
            run_params = {
                "stoploss_pct": stoploss,
                "takeprofit_pct": takeprofit,
                "VOL_RATIO_MIN": vol_ratio_min,
                "RSI_MAX": rsi_max
            }
            
            try:
                out = run_backtest(universe, run_params, cash=cash)
                
                # 结果展示
                final_val = out.get('final_value')
                profit = final_val - cash
                ret_pct = (profit / cash) * 100
                
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("初始资金", f"{cash:,.0f}")
                c2.metric("最终资金", f"{final_val:,.2f}")
                c3.metric("收益率", f"{ret_pct:.2f}%", delta=f"{profit:,.2f}")
                
                st.json({"Status": "Finished", "Final Value": final_val})
                
            except Exception as e:
                st.error(f"回测运行出错: {e}")

        elif mode == "参数优化 (Optimize)":
            st.info("参数优化功能在此 Web 模式下简化展示，建议在本地环境运行以获得最佳性能。")
            # 这里可以接入 run_optuna 逻辑，但考虑到网页超时问题，建议谨慎
            st.warning("优化功能耗时较长，请确保服务器不会超时。")

if __name__ == "__main__":
    main_gui()
