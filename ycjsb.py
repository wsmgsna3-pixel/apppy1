import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 目标样本库 (您的 12 罗汉)
# ==========================================
TARGET_STOCKS = {
    '申菱环境': '301018.SZ',
    '利通电子': '603629.SH',
    '田中精机': '300461.SZ',
    '宏和科技': '603256.SH',
    '致尚科技': '301486.SZ',  # 修正名称
    '罗博特科': '300757.SZ',
    '炬光科技': '688167.SH',
    '嘉美包装': '002969.SZ',
    '横店影视': '603103.SH',
    '长飞光纤': '601869.SH',
    '博迁新材': '605376.SH',
    '振德医疗': '603301.SH'
}

# 特殊时间段设定 (振德医疗 202509-202511, 其他近40天)
# 这里的"近40天"是相对于您当前的模拟时间 (2026-02-14)
SPECIAL_PERIODS = {
    '振德医疗': ('20250901', '20251110')
}

DEFAULT_START = '20251201' # 其他股票的默认回溯起点
DEFAULT_END = '20260214'   # 当前模拟时间

# ==========================================
# 2. 测序引擎
# ==========================================
def analyze_dna(token):
    ts.set_token(token)
    pro = ts.pro_api()
    
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    total = len(TARGET_STOCKS)
    
    for i, (name, code) in enumerate(TARGET_STOCKS.items()):
        status.text(f"正在测序: {name} ({code}) ...")
        progress.progress((i)/total)
        
        # 1. 确定时间段
        start_date, end_date = SPECIAL_PERIODS.get(name, (DEFAULT_START, DEFAULT_END))
        
        # 2. 获取数据 (日线 + 指标)
        try:
            df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
            df_basic = pro.daily_basic(ts_code=code, start_date=start_date, end_date=end_date, 
                                     fields='trade_date,turnover_rate,turnover_rate_f,volume_ratio,circ_mv,pe,pb')
            
            if df.empty or df_basic.empty:
                st.warning(f"{name}: 无数据")
                continue
                
            df = pd.merge(df, df_basic, on='trade_date')
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            # 3. 寻找"启动点" (Launch Point)
            # 定义启动点：区间内涨幅最大的那一波的主升浪起点
            # 简单算法：找到区间内最低点后，第一根涨幅 > 5% 且量比 > 1.5 的K线
            
            # 计算滚动最低价
            df['min_20'] = df['low'].rolling(20, min_periods=1).min()
            
            # 启动条件: 
            # 1. 当日大涨 > 5%
            # 2. 距离近期低点不超过 20% (还在底部区域)
            # 3. 量比放大
            
            launch_candidates = df[
                (df['pct_chg'] > 5.0) & 
                (df['close'] < df['min_20'] * 1.3) # 底部起涨
            ]
            
            if launch_candidates.empty:
                # 如果没抓到，就取涨幅最大的一天作为参考
                launch_day = df.loc[df['pct_chg'].idxmax()]
                note = "最大涨幅日"
            else:
                # 取第一个满足条件的作为启动日
                launch_day = launch_candidates.iloc[0]
                note = "精准启动日"
                
            # 4. 提取基因 (T-0 启动日特征)
            # 均线计算
            idx = launch_day.name
            if idx < 5: continue # 数据不足
            
            # T-1 (启动前一天) 的状态
            prev_day = df.iloc[idx-1]
            
            # 计算 RSI
            # 简单手写 RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
            rs = gain / loss
            rsi_6 = 100 - (100 / (1 + rs))
            launch_rsi = rsi_6.iloc[idx]
            
            dna = {
                '名称': name,
                '启动日期': launch_day['trade_date'],
                '类型': note,
                '启动涨幅': f"{launch_day['pct_chg']:.1f}%",
                '启动换手(%)': f"{launch_day['turnover_rate']:.1f}",
                '启动量比': f"{launch_day['volume_ratio']:.1f}",
                '流通市值(亿)': f"{launch_day['circ_mv']/10000:.1f}",
                '启动RSI(6)': f"{launch_rsi:.1f}",
                '启动前均线': '多头' if (prev_day['close'] > prev_day['open']) else '震荡' # 简化判断
            }
            results.append(dna)
            
        except Exception as e:
            st.error(f"{name} 测序失败: {e}")
            
    progress.empty()
    status.empty()
    
    return pd.DataFrame(results)

# ==========================================
# 3. 主程序
# ==========================================
st.set_page_config(page_title="DNA 逆向测序", layout="wide")
st.title("🧬 翻倍股 DNA 逆向测序报告")

token = st.sidebar.text_input("Tushare Token", type="password")

if st.sidebar.button("开始测序"):
    if not token:
        st.error("请输入 Token")
    else:
        df_dna = analyze_dna(token)
        
        if not df_dna.empty:
            st.success("测序完成！发现翻倍基因如下：")
            st.dataframe(df_dna)
            
            # 自动总结规律
            st.markdown("### 📊 基因图谱总结")
            
            # 数值提取
            mvs = df_dna['流通市值(亿)'].astype(float)
            turns = df_dna['启动换手(%)'].astype(float)
            vols = df_dna['启动量比'].astype(float)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("流通市值中位数", f"{mvs.median():.1f} 亿", help="V23 市值参数参考")
            col2.metric("启动换手率中位数", f"{turns.median():.1f}%", help="V23 换手参数参考")
            col3.metric("启动量比中位数", f"{vols.median():.1f}", help="V23 量比参数参考")
            
            st.info(f"💡 **V23 策略建议：**\n"
                    f"1. 市值锁定在 **{mvs.min():.1f} - {mvs.max():.1f} 亿** 之间。\n"
                    f"2. 换手率门槛设为 **{turns.min():.1f}%** 以上。\n"
                    f"3. 量比门槛设为 **{vols.min():.1f}** 以上。")
            
        else:
            st.warning("未提取到有效基因。")
