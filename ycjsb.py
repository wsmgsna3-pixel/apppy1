import streamlit as st
import tushare as ts
import pandas as pd
import time
from datetime import datetime, timedelta
import os

# ==========================================
# Streamlit 页面配置
# ==========================================
st.set_page_config(page_title="鹰眼·Pro (稳健修复版)", layout="wide")

st.title("🦅 鹰眼·假摔猎杀 Pro (稳健修复版)")
st.markdown("""
**修复说明：**
增加了对 Tushare 返回数据的健壮性检查，解决了 KeyError 报错。
如果运行出现问题，**请先点击左侧的“🧹 清除所有缓存数据”按钮**。
""")

# ==========================================
# 1. 缓存化数据获取函数
# ==========================================
@st.cache_data(persist="disk", show_spinner=False)
def get_cached_daily(token, date_str):
    """缓存日线数据"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.daily(trade_date=date_str)
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
        if df.empty: return pd.DataFrame()
        return pd.merge(df, df_basic, on='ts_code')
    except Exception:
        return pd.DataFrame()

@st.cache_data(persist="disk", show_spinner=False)
def get_cached_cyq(token, code, date_str):
    """缓存筹码数据"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # 注意：Tushare 接口可能会因为无数据返回空 DF，这是正常的
        df = pro.cyq_perf(ts_code=code, trade_date=date_str)
        return df
    except Exception:
        return pd.DataFrame()

# ==========================================
# 2. 侧边栏设置
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数控制台")
    
    user_token = st.text_input("Tushare Token (必填):", type="password")
    
    st.subheader("🔍 筛选阈值")
    shadow_threshold = st.slider("上影线长度 (%)", 1.0, 10.0, 3.0, 0.5)
    profit_threshold = st.slider("筹码获利盘 (%)", 0, 99, 50, 5)
    scan_limit = st.slider("每日最大扫描数", 10, 200, 50, 10)
    
    st.subheader("📅 回测区间")
    default_start = datetime.now() - timedelta(days=10)
    default_end = datetime.now()
    start_date = st.date_input("开始日期", default_start)
    end_date = st.date_input("结束日期", default_end)
    
    st.markdown("---")
    run_btn = st.button("🚀 启动/刷新回测")
    
    # 增加清除缓存按钮的显眼提示
    if st.button("🧹 清除所有缓存数据 (修复报错用)"):
        st.cache_data.clear()
        st.success("缓存已清除！请重新点击启动。")

# ==========================================
# 3. 策略主逻辑 (已修复 KeyError)
# ==========================================
def run_strategy():
    ts.set_token(user_token)
    pro = ts.pro_api()
    
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
        st.warning("回测区间太短，请选择更长的时间段。")
        return

    with log_container:
        st.write("### 📜 实时扫描日志")
        
    for i in range(total_days):
        date_today = trade_days[i]
        date_next = trade_days[i+1]
        
        progress_bar.progress((i + 1) / total_days)
        
        # --- 1. 获取日线 ---
        df_today = get_cached_daily(user_token, date_today)
        
        if df_today.empty:
            continue
            
        df_today = df_today[~df_today['name'].str.contains('ST')]
        
        # --- 2. 形态初筛 ---
        df_today['body_top'] = df_today[['open', 'close']].max(axis=1)
        df_today['upper_shadow'] = (df_today['high'] - df_today['body_top']) / df_today['pre_close'] * 100
        
        mask_shape = (df_today['upper_shadow'] > shadow_threshold) & \
                     (df_today['pct_chg'] > -3) & (df_today['pct_chg'] < 8)
        
        candidates_df = df_today[mask_shape].copy()
        
        if len(candidates_df) == 0:
            with log_container:
                st.write(f"📅 {date_today}: 无形态符合股票")
            continue
            
        # --- 3. 智能优选 ---
        candidates_df = candidates_df.sort_values(by='amount', ascending=False)
        target_list = candidates_df.head(scan_limit)['ts_code'].tolist()
        
        with log_container:
            st.write(f"📅 {date_today}: 初筛 {len(candidates_df)} 只，深度扫描前 {len(target_list)} 只热门股...")
        
        # --- 4. 筹码测谎 (修复报错点) ---
        passed_codes = []
        profits_list = [] 
        
        scan_status = st.empty()
        
        for idx, code in enumerate(target_list):
            scan_status.text(f"扫描进度: {date_today} - {idx+1}/{len(target_list)}")
            
            df_cyq = get_cached_cyq(user_token, code, date_today)
            
            # === 核心修复: 增加列名检查 ===
            # 只有当 DataFrame 不为空，且包含 'profit_rate' 列时才读取
            if not df_cyq.empty and 'profit_rate' in df_cyq.columns:
                try:
                    profit = df_cyq.iloc[0]['profit_rate']
                    
                    # 确保 profit 是数字
                    if pd.isna(profit): continue
                    
                    profits_list.append(profit)
                    
                    if profit > profit_threshold:
                        passed_codes.append(code)
                        stock_name = candidates_df[candidates_df['ts_code']==code]['name'].values[0]
                        with log_container:
                            st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;✅ **发现**: {stock_name} | 获利盘: {profit:.1f}%")
                except Exception:
                    continue
            else:
                # 如果没有数据，或者数据缺失列，直接跳过，不报错
                continue
        
        scan_status.empty()
        
        # 统计
        if profits_list:
            avg_profit = sum(profits_list) / len(profits_list)
            if not passed_codes:
                with log_container:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;❌ 未通过 (市场平均获利盘: {avg_profit:.1f}%)")
        else:
             with log_container:
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;⚠️ 无有效筹码数据 (可能是Token权限或数据缺失)")
        
        if not passed_codes:
            continue
            
        # --- 5. 次日验证 ---
        df_next = get_cached_daily(user_token, date_next)
        if df_next.empty: continue
        
        for code in passed_codes:
            row_next = df_next[df_next['ts_code'] == code]
            if row_next.empty: continue
            
            open_T1 = row_next.iloc[0]['open']
            close_T1 = row_next.iloc[0]['close']
            
            close_T = candidates_df[candidates_df['ts_code'] == code]['close'].values[0]
            stock_name = candidates_df[candidates_df['ts_code'] == code]['name'].values[0]
            
            if open_T1 > close_T:
                profit_pct = (close_T1 - open_T1) / open_T1 * 100
                
                # 记录获利盘数据，防止 index error
                try:
                     # 找到该代码在 target_list 中的位置，再取 profits_list
                     # 这种对应关系在复杂逻辑下可能不稳，改用直接存储
                     # 简单处理：这里不显示具体获利盘数字了，或者在上面 loop 里存 dict
                     display_profit = "High"
                except:
                    display_profit = "High"

                results.append({
                    '信号日期': date_today,
                    '代码': code,
                    '名称': stock_name,
                    '买入价': open_T1,
                    '当日收益(%)': round(profit_pct, 2)
                })

    progress_bar.empty()
    
    # --- 6. 结果展示 ---
    if results:
        df_res = pd.DataFrame(results)
        st.success(f"🎉 扫描完成！共发现 {len(df_res)} 次机会")
        
        wins = df_res[df_res['当日收益(%)'] > 0]
        win_rate = len(wins) / len(df_res) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("总胜率", f"{win_rate:.1f}%")
        c2.metric("平均收益", f"{df_res['当日收益(%)'].mean():.2f}%")
        c3.metric("累计收益", f"{df_res['当日收益(%)'].sum():.2f}%")
        
        st.dataframe(df_res.style.applymap(lambda x: f'color: {"red" if x>0 else "green"}', subset=['当日收益(%)']), use_container_width=True)
    else:
        st.warning("本次扫描未发现符合条件的标的。")

# ==========================================
# 启动入口
# ==========================================
if run_btn:
    if not user_token:
        st.error("请先输入 Token")
    else:
        run_strategy()
