import streamlit as st
import tushare as ts
import pandas as pd
import time
from datetime import datetime, timedelta
import os

# ==========================================
# Streamlit 页面配置
# ==========================================
st.set_page_config(page_title="鹰眼·Pro (缓存加速版)", layout="wide")

st.title("🦅 鹰眼·假摔猎杀 Pro (缓存加速+智能优选)")
st.markdown("""
**升级说明：**
1. **硬盘缓存**：数据拉取一次后自动存入硬盘。修改参数或重启后，直接读取缓存，**秒级回测且不耗积分**。
2. **智能优选**：候选股过多时，优先扫描**成交额最大**的前 N 只（主力战场），拒绝扫描垃圾股。
""")

# ==========================================
# 1. 缓存化数据获取函数 (核心升级)
# ==========================================
# 使用 persist="disk" 实现断点续传和缓存，缓存文件保存在 .streamlit/cache 中

@st.cache_data(persist="disk", show_spinner=False)
def get_cached_daily(token, date_str):
    """缓存日线数据，避免重复拉取"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.daily(trade_date=date_str)
        # 同时拉取基础信息用于过滤
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
        if df.empty: return pd.DataFrame()
        return pd.merge(df, df_basic, on='ts_code')
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(persist="disk", show_spinner=False)
def get_cached_cyq(token, code, date_str):
    """缓存单个股票的筹码数据，这是最耗时的一步"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # 注意：这里可能会因为频率限制报错，外部需要处理重试
        df = pro.cyq_perf(ts_code=code, trade_date=date_str)
        return df
    except:
        return pd.DataFrame()

# ==========================================
# 2. 侧边栏设置
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数控制台")
    
    # Token 输入
    user_token = st.text_input("Tushare Token (自动缓存):", type="password")
    
    st.subheader("🔍 筛选阈值 (随时调整，秒级生效)")
    shadow_threshold = st.slider("上影线长度 (%)", 1.0, 10.0, 3.0, 0.5)
    profit_threshold = st.slider("筹码获利盘 (%)", 0, 99, 50, 5, help="如果全军覆没，请尝试降低此值")
    scan_limit = st.slider("每日最大扫描数 (只)", 10, 200, 50, 10, help="优先扫描成交额最大的前N只")
    
    st.subheader("📅 回测区间")
    default_start = datetime.now() - timedelta(days=10)
    default_end = datetime.now()
    start_date = st.date_input("开始日期", default_start)
    end_date = st.date_input("结束日期", default_end)
    
    run_btn = st.button("🚀 启动/刷新回测")
    
    if st.button("🧹 清除所有缓存数据"):
        st.cache_data.clear()
        st.success("缓存已清除！")

# ==========================================
# 3. 策略主逻辑
# ==========================================
def run_strategy():
    # 初始化接口
    ts.set_token(user_token)
    pro = ts.pro_api()
    
    # 获取日历
    s_str = start_date.strftime('%Y%m%d')
    e_str = end_date.strftime('%Y%m%d')
    try:
        cal = pro.trade_cal(exchange='', start_date=s_str, end_date=e_str, is_open='1')
        trade_days = cal['cal_date'].tolist()
    except:
        st.error("日期获取失败，请检查Token或网络")
        return

    log_container = st.container()
    progress_bar = st.progress(0)
    
    results = []
    total_days = len(trade_days) - 1
    
    if total_days < 1:
        st.warning("回测区间太短，需要至少2个交易日")
        return

    with log_container:
        st.write("### 📜 实时扫描日志")
        
    for i in range(total_days):
        date_today = trade_days[i]
        date_next = trade_days[i+1]
        
        progress_bar.progress((i + 1) / total_days)
        
        # --- 1. 获取日线 (读取缓存) ---
        df_today = get_cached_daily(user_token, date_today)
        
        if df_today.empty:
            with log_container:
                st.write(f"⚠️ {date_today}: 无行情数据")
            continue
            
        # 简单过滤
        df_today = df_today[~df_today['name'].str.contains('ST')]
        
        # --- 2. 形态初筛 ---
        # 向量化计算，速度更快
        df_today['body_top'] = df_today[['open', 'close']].max(axis=1)
        df_today['upper_shadow'] = (df_today['high'] - df_today['body_top']) / df_today['pre_close'] * 100
        
        # 筛选符合形态的
        mask_shape = (df_today['upper_shadow'] > shadow_threshold) & \
                     (df_today['pct_chg'] > -3) & (df_today['pct_chg'] < 8)
        
        candidates_df = df_today[mask_shape].copy()
        
        count_raw = len(candidates_df)
        if count_raw == 0:
            with log_container:
                st.write(f"📅 {date_today}: 无形态符合股票")
            continue
            
        # --- 3. 智能优选 (关键修改) ---
        # 按【成交额 amount】降序排列，优先看主力资金活跃的票
        candidates_df = candidates_df.sort_values(by='amount', ascending=False)
        
        # 截取前 N 只
        target_list = candidates_df.head(scan_limit)['ts_code'].tolist()
        
        with log_container:
            st.write(f"📅 {date_today}: 发现 {count_raw} 只形态股。**智能优选成交额最大的 {len(target_list)} 只进行深度扫描...**")
        
        # --- 4. 筹码测谎 (读取缓存) ---
        passed_codes = []
        profits_list = [] # 用于统计市场情绪
        
        # 进度显示的占位符
        scan_status = st.empty()
        
        for idx, code in enumerate(target_list):
            scan_status.text(f"正在扫描: {date_today} - {code} ({idx+1}/{len(target_list)})")
            
            # 调用缓存函数
            df_cyq = get_cached_cyq(user_token, code, date_today)
            
            if not df_cyq.empty:
                profit = df_cyq.iloc[0]['profit_rate']
                profits_list.append(profit)
                
                if profit > profit_threshold:
                    passed_codes.append(code)
                    # 实时打印命中信息
                    stock_name = candidates_df[candidates_df['ts_code']==code]['name'].values[0]
                    with log_container:
                        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;✅ **命中**: {stock_name} ({code}) - 获利盘: {profit:.1f}%")
        
        scan_status.empty()
        
        # 市场情绪反馈
        if profits_list:
            avg_profit = sum(profits_list) / len(profits_list)
            if not passed_codes:
                with log_container:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;❌ **全军覆没** (该批次平均获利盘仅为: {avg_profit:.1f}%，市场环境极差)")
        
        if not passed_codes:
            continue
            
        # --- 5. 次日验证 (读取缓存) ---
        # 批量获取次日数据
        df_next = get_cached_daily(user_token, date_next)
        if df_next.empty: continue
        
        for code in passed_codes:
            row_next = df_next[df_next['ts_code'] == code]
            if row_next.empty: continue
            
            open_T1 = row_next.iloc[0]['open']
            close_T1 = row_next.iloc[0]['close']
            
            # T日收盘价
            close_T = candidates_df[candidates_df['ts_code'] == code]['close'].values[0]
            stock_name = candidates_df[candidates_df['ts_code'] == code]['name'].values[0]
            
            # 必须高开 (弱转强)
            if open_T1 > close_T:
                profit_pct = (close_T1 - open_T1) / open_T1 * 100
                
                results.append({
                    '信号日期': date_today,
                    '代码': code,
                    '名称': stock_name,
                    'T日获利盘': f"{profits_list[target_list.index(code)]:.1f}%",
                    '买入价': open_T1,
                    '当日收益': round(profit_pct, 2)
                })

    progress_bar.empty()
    
    # --- 6. 结果展示 ---
    if results:
        df_res = pd.DataFrame(results)
        st.success(f"🎉 扫描完成！共发现 {len(df_res)} 次机会")
        
        # 统计面板
        wins = df_res[df_res['当日收益'] > 0]
        win_rate = len(wins) / len(df_res) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("总胜率", f"{win_rate:.1f}%")
        c2.metric("平均收益", f"{df_res['当日收益'].mean():.2f}%")
        c3.metric("累计收益", f"{df_res['当日收益'].sum():.2f}%")
        
        st.dataframe(df_res.style.applymap(lambda x: f'color: {"red" if x>0 else "green"}', subset=['当日收益']), use_container_width=True)
    else:
        st.warning("本次扫描未发现符合条件的标的。请尝试：1. 降低获利盘阈值；2. 扩大日期范围。")

# ==========================================
# 启动入口
# ==========================================
if run_btn:
    if not user_token:
        st.error("请先输入 Token")
    else:
        run_strategy()
