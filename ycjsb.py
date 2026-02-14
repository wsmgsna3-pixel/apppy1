import streamlit as st
import pandas as pd
import tushare as ts
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 🎯 补全目标 (Missing 4)
# ==========================================
MISSING_STOCKS = {
    '致尚科技': '301486.SZ',
    '罗博特科': '300757.SZ',
    '炬光科技': '688167.SH',
    '嘉美包装': '002969.SZ'
}

DEFAULT_START = '20251101' # 放宽时间范围 (多看一个月)
DEFAULT_END = '20260214'

def fix_dna(token):
    if not token:
        st.error("请输入 Token")
        return pd.DataFrame()
        
    ts.set_token(token)
    pro = ts.pro_api()
    
    results = []
    status = st.empty()
    
    for name, code in MISSING_STOCKS.items():
        status.text(f"正在补全: {name} ({code}) ...")
        
        try:
            # 1. 获取更长周期的数据
            df = pro.daily(ts_code=code, start_date=DEFAULT_START, end_date=DEFAULT_END)
            df_basic = pro.daily_basic(ts_code=code, start_date=DEFAULT_START, end_date=DEFAULT_END, 
                                     fields='trade_date,turnover_rate,volume_ratio,circ_mv')
            
            if df.empty:
                st.error(f"❌ {name}: Tushare 返回数据为空 (可能停牌或代码错误)")
                continue
                
            df = pd.merge(df, df_basic, on='trade_date')
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            # 2. 强制提取启动点 (Force Mode)
            # 不再要求 > 5% 的硬指标，直接找区间内涨幅最大的一天
            launch_day = df.loc[df['pct_chg'].idxmax()]
            
            # 3. 提取基因
            idx = launch_day.name
            prev_day = df.iloc[idx-1] if idx > 0 else df.iloc[0]
            
            # 计算 RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            launch_rsi = rsi.iloc[idx]
            
            dna = {
                '名称': name,
                '启动日期': launch_day['trade_date'],
                '类型': '补全数据',
                '启动涨幅': f"{launch_day['pct_chg']:.1f}%",
                '启动换手(%)': f"{launch_day['turnover_rate']:.1f}",
                '启动量比': f"{launch_day['volume_ratio']:.1f}",
                '流通市值(亿)': f"{launch_day['circ_mv']/10000:.1f}",
                '启动RSI': f"{launch_rsi:.1f}",
                '前日趋势': '多头' if (prev_day['close'] > prev_day['open']) else '震荡'
            }
            results.append(dna)
            
        except Exception as e:
            st.error(f"❌ {name} 补全失败: {e}")
            
    status.empty()
    return pd.DataFrame(results)

# ==========================================
# 主程序
# ==========================================
st.title("🧬 DNA 补全计划")
token = st.sidebar.text_input("Tushare Token", type="password")

if st.button("开始补全"):
    df_fix = fix_dna(token)
    if not df_fix.empty:
        st.success("✅ 补全成功！以下是缺失的 4 个基因：")
        st.dataframe(df_fix)
        
        # 重新计算 V23 参数建议
        mvs = df_fix['流通市值(亿)'].astype(float)
        turns = df_fix['启动换手(%)'].astype(float)
        vols = df_fix['启动量比'].astype(float)
        
        st.info(f"💡 **新发现：**\n"
                f"这 4 个股票的市值范围: {mvs.min()} - {mvs.max()} 亿\n"
                f"换手率最低: {turns.min()}%\n"
                f"量比最低: {vols.min()}")
