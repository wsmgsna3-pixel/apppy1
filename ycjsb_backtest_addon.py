import importlib
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tushare as ts
import backtrader as bt
import streamlit as st  # 引入 Streamlit

# ------------- 页面基础配置 -------------
st.set_page_config(page_title="选股回测工具", layout="wide")

# ------------- Configurable defaults -------------
DEFAULTS = {
    "MIN_PRICE": 3.0,
    "MAX_PRICE": 500.0,
    "MIN_TURNOVER": 2.0,
    "MIN_AMOUNT": 50_000_000.0,
    "VOL_RATIO_MIN": 1.0,  # 更宽松的量比条件
    "RSI_MAX": 90,  # 提高 RSI 上限到 90，更宽松
    "MACD_MIN": -0.1,  # 提高 MACD 最低值
    "MAX_5D_PCT": 50,  # 提高最大 5 日涨幅限制
    "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
}

# ------------- 使用 Tushare 获取股票池 -------------
def build_universe_from_ycjsb(last_trade, params, pro_api):
    """
    使用 Tushare 获取股票池，默认选择每日涨幅前 N 名的股票
    """
    daily = pro_api.daily(trade_date=last_trade)
    if daily is None or daily.empty:
        raise RuntimeError("无法获取每日行情数据 (daily data unavailable).")
    
    # 排序，选择涨幅前 N 的股票
    top = daily.sort_values("pct_chg", ascending=False).head(params.get("INITIAL_TOP_N", 800))
    return list(top['ts_code'].unique())

# ------------- run a single backtest -------------
def run_backtest(universe, params, cash=100000.0, commission=0.0003, slippage=0.000, verbose=False):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    
    df_dict = {}
    progress_text = "正在获取历史数据..."
    my_bar = st.progress(0, text=progress_text)
    
    total_stocks = len(universe)
    for i, ts_code in enumerate(universe):
        try:
            my_bar.progress((i + 1) / total_stocks, text=f"获取数据: {ts_code}")
            
            df = get_hist(ts_code, start_date=DEFAULTS['start_date'], end_date=datetime.now().strftime("%Y%m%d"))
            if df.empty:
                continue
            df = df[['trade_date','open','high','low','close','vol']].copy()
            df_dict[ts_code] = df
        except Exception as e:
            print(f"failed to fetch {ts_code}: {e}")
            
    my_bar.empty()

    if not df_dict:
        raise RuntimeError("没有获取到任何回测数据，请检查Token或网络。")
    
    create_bt_datas(cerebro, df_dict)
    
    # 宽松回测参数设置
    stratparams = dict(
        stake=1,
        stoploss_pct=params.get('stoploss_pct', 0.10),  # 放宽止损为10%
        takeprofit_pct=params.get('takeprofit_pct', 0.30),  # 放宽止盈为30%
        hold_days=int(params.get('hold_days', 10)),  # 持股天数增加到10天
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
        st.stop() 

    try:
        ts.set_token(token_input)
        pro_local = ts.pro_api()
    except Exception as e:
        st.error(f"Token 设置失败: {e}")
        st.stop()

    # --- 2. 参数设置 ---
    st.sidebar.header("⚙️ 回测参数设置")
    
    # 模式选择
    mode = st.sidebar.selectbox("运行模式", ["单次运行 (Run)", "参数优化 (Optimize)"])
    
    # 设置一些默认参数
    topn = st.sidebar.number_input("每日候选池大小 (Top N)", value=800, step=50)
    min_price = st.sidebar.number_input("最低股价", value=3.0)
    max_price = st.sidebar.number_input("最高股价", value=500.0)
    min_turnover = st.sidebar.number_input("最低换手率", value=2.0)
    
    st.sidebar.subheader("交易策略")
    cash = st.sidebar.number_input("初始资金", value=100000.0)
    stoploss = st.sidebar.number_input("止损百分比 (0.10 = 10%)", value=0.10, step=0.01)
    takeprofit = st.sidebar.number_input("止盈百分比 (0.30 = 30%)", value=0.30, step=0.01)

    # --- 3. 运行逻辑 ---
    if st.button("🚀 开始运行", type="primary"):
        
        # 准备参数
        base_params = DEFAULTS.copy()
        base_params.update({
            "MIN_PRICE": min_price,
            "MAX_PRICE": max_price,
            "MIN_TURNOVER": min_turnover,
        })

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

        # 构建股票池
        with st.spinner(f"正在构建股票池 (基准日期: {last_trade})..."):
            try:
                universe = build_universe_from_ycjsb(last_trade, {"INITIAL_TOP_N": topn}, pro_local)
                st.success(f"股票池构建完成，共包含 {len(universe)} 只股票")
            except Exception as e:
                st.error(f"构建股票池失败: {e}")
                st.stop()

        # 执行回测
        if mode == "单次运行 (Run)":
            run_params = {
                "stoploss_pct": stoploss,
                "takeprofit_pct": takeprofit,
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

if __name__ == "__main__":
    main_gui()
