import streamlit as st
import tushare as ts
import pandas as pd
import time
from datetime import datetime, timedelta

# ==========================================
# Streamlit 页面配置
# ==========================================
st.set_page_config(page_title="鹰眼·诊断模式", layout="wide")

st.title("🦅 鹰眼·假摔猎杀 (全量诊断版)")
st.markdown("""
**调试模式说明：**
此版本移除了所有数量限制，并增加了实时日志。
如果仍然选不出股，请尝试降低侧边栏的【获利盘阈值】。
""")

# ==========================================
# 1. 侧边栏：参数全开放
# ==========================================
with st.sidebar:
    st.header("⚙️ 核心参数调节")
    
    user_token = st.text_input("Tushare Token (必填):", type="password")
    
    # 增加参数滑块，方便调试
    shadow_threshold = st.slider("上影线长度阈值 (%)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    profit_threshold = st.slider("筹码获利盘阈值 (%)", min_value=50, max_value=99, value=80, step=5)
    
    st.markdown("---")
    st.markdown("### 回测区间")
    default_start = datetime.now() - timedelta(days=14) # 默认只跑最近两周，太久会很慢
    default_end = datetime.now()
    
    start_date = st.date_input("开始日期", default_start)
    end_date = st.date_input("结束日期", default_end)
    
    run_btn = st.button("🚀 启动全量扫描")

# ==========================================
# 2. 策略逻辑函数 (带诊断输出)
# ==========================================
def run_diagnostic_strategy(token, s_date, e_date, shadow_limit, profit_limit):
    s_str = s_date.strftime('%Y%m%d')
    e_str = e_date.strftime('%Y%m%d')
    
    # 初始化
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # 测试连通性
        pro.daily(trade_date=s_str, limit=1)
    except Exception as e:
        st.error(f"Token 连接失败: {e}")
        return pd.DataFrame()

    # 进度与日志区
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_area = st.expander("📜 实时扫描日志 (点击展开)", expanded=True)
    
    # 获取日历
    try:
        cal = pro.trade_cal(exchange='', start_date=s_str, end_date=e_str, is_open='1')
        trade_days = cal['cal_date'].tolist()
    except:
        st.error("无法获取交易日历，请检查日期范围或网络。")
        return pd.DataFrame()

    if len(trade_days) < 2:
        st.warning("交易日不足 2 天，无法回测次日表现。")
        return pd.DataFrame()

    trade_log = []
    
    # 遍历每一天
    total_days = len(trade_days) - 1
    
    for i in range(total_days):
        date_today = trade_days[i]
        date_tomorrow = trade_days[i+1]
        
        progress = (i + 1) / total_days
        progress_bar.progress(progress)
        status_text.text(f"正在深度分析: {date_today} ...")
        
        # --- A. 形态初筛 ---
        try:
            df_today = pro.daily(trade_date=date_today)
            df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
            df_today = pd.merge(df_today, df_basic, on='ts_code')
            
            # 过滤掉 ST 和 北交所
            df_today = df_today[~df_today['name'].str.contains('ST')]
            df_today = df_today[~df_today['market'].str.contains('北交所')]
            
        except Exception as e:
            log_area.write(f"❌ {date_today} 行情数据获取失败: {e}")
            continue

        # 形态计算
        candidates = []
        for idx, row in df_today.iterrows():
            if row['close'] == 0 or row['pre_close'] == 0: continue
            
            body_top = max(row['open'], row['close'])
            upper_shadow = (row['high'] - body_top) / row['pre_close'] * 100
            pct_chg = row['pct_chg']
            
            # 使用用户设定的阈值
            if upper_shadow > shadow_limit and -3 < pct_chg < 8:
                candidates.append(row['ts_code'])
        
        # 日志输出初筛结果
        if len(candidates) == 0:
            log_area.write(f"📅 {date_today}: 无股票符合【长上影 > {shadow_limit}%】形态")
            continue
        else:
            log_area.write(f"📅 {date_today}: 初筛发现 {len(candidates)} 只形态股，开始筹码测谎...")

        # --- B. 筹码测谎 (全量检查，无 [:30] 限制) ---
        real_targets = []
        
        # 批量处理技巧：虽然cyq_perf只能单只取，但我们可以在这里加个简单的限流保护
        # 为了不让页面卡死太久，我们设定单日最大扫描数，如果太多则只取前50个（毕竟Streamlit有超时限制）
        # 但这次我们稍微放宽点
        scan_limit = 50 
        scan_list = candidates[:scan_limit] 
        
        if len(candidates) > scan_limit:
            log_area.write(f"⚠️ {date_today} 候选过多 ({len(candidates)}只)，仅扫描前 {scan_limit} 只以防超时...")

        pass_chip_count = 0
        
        for code in scan_list:
            try:
                # 核心：调用筹码接口
                # 注意：高频调用可能会偶尔失败，需要容错
                df_cyq = pro.cyq_perf(ts_code=code, trade_date=date_today)
                
                if df_cyq.empty: 
                    continue
                
                profit_rate = df_cyq.iloc[0]['profit_rate']
                
                if profit_rate > profit_limit:
                    real_targets.append(code)
                    pass_chip_count += 1
                    # 打印一条发现日志
                    log_area.write(f"  --> ✅ 发现猎物 {code}: 获利盘 {profit_rate:.1f}%")
                
                # 极速限流：10000积分每分钟300次，理论上不用sleep太久，但保险起见
                # time.sleep(0.05) 
                
            except Exception as e:
                # 可以在这里打印 API 错误，排查是不是权限问题
                # log_area.write(f"API Error on {code}: {e}")
                continue
        
        if pass_chip_count == 0:
            log_area.write(f"  --> ❌ 全军覆没：没有股票的获利盘 > {profit_limit}%")
            continue
            
        # --- C. 次日验证 ---
        try:
            df_next = pro.daily(trade_date=date_tomorrow, ts_code=','.join(real_targets))
        except:
            continue
            
        for idx, row_next in df_next.iterrows():
            code = row_next['ts_code']
            stock_name = df_basic[df_basic['ts_code'] == code]['name'].values[0] if not df_basic.empty else code
            
            # T日收盘价
            close_T = df_today[df_today['ts_code'] == code]['close'].values[0]
            
            # T+1 开盘价
            open_T1 = row_next['open']
            
            # 必须高开
            if open_T1 > close_T:
                close_T1 = row_next['close']
                profit_pct = (close_T1 - open_T1) / open_T1 * 100
                
                trade_log.append({
                    '信号日期': date_today,
                    '代码': code,
                    '名称': stock_name,
                    '买入价': open_T1,
                    '当日收益(%)': round(profit_pct, 2),
                    '触发原因': f"影线>{shadow_limit}%, 获利>{profit_limit}%"
                })

    progress_bar.empty()
    status_text.text("全量诊断完成。")
    return pd.DataFrame(trade_log)

# ==========================================
# 3. 运行入口
# ==========================================
if run_btn:
    if not user_token:
        st.error("请先输入 Token！")
    else:
        with st.spinner('正在进行全量诊断扫描...'):
            df_res = run_diagnostic_strategy(user_token, start_date, end_date, shadow_threshold, profit_threshold)
            
        if not df_res.empty:
            st.success(f"诊断完成！共选出 {len(df_res)} 次交易机会")
            
            # 指标计算
            wins = df_res[df_res['当日收益(%)'] > 0]
            win_rate = len(wins) / len(df_res) * 100
            
            col1, col2 = st.columns(2)
            col1.metric("胜率 (Win Rate)", f"{win_rate:.1f}%")
            col2.metric("总收益 (Total)", f"{df_res['当日收益(%)'].sum():.1f}%")
            
            def color_profit(val):
                return f'color: {"red" if val > 0 else "green"}'
            
            st.dataframe(df_res.style.applymap(color_profit, subset=['当日收益(%)']))
        else:
            st.warning("⚠️ 依然没有结果。")
            st.info("""
            **排查建议：**
            1. 请查看上方的【实时扫描日志】，确认 '初筛发现' 的数量是否为 0？
            2. 如果初筛有数据，但筹码全军覆没，请尝试将【获利盘阈值】调低至 60% 或 50%。
            3. 确保您测试的日期不是全市场暴跌的日期（那时大家都亏钱，获利盘自然低）。
            """)
