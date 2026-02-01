import streamlit as st
import tushare as ts
import pandas as pd
import time
from datetime import datetime, timedelta

# ==========================================
# Streamlit 页面配置
# ==========================================
st.set_page_config(page_title="鹰眼·假摔猎杀回测系统", layout="wide")

st.title("🦅 鹰眼·假摔猎杀策略 (10000积分专用)")
st.markdown("### 策略核心：寻找'射击之星'形态 + 筹码锁定 + 次日弱转强")

# ==========================================
# 1. 侧边栏：参数设置 (Token 输入)
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 获取 Token，默认留空提醒用户输入
    user_token = st.text_input("请输入 Tushare Token (必填):", type="password")
    
    # 日期选择
    default_start = datetime.now() - timedelta(days=30)
    default_end = datetime.now()
    
    start_date = st.date_input("回测开始日期", default_start)
    end_date = st.date_input("回测结束日期", default_end)
    
    run_btn = st.button("🚀 开始回测 / 选股")

# ==========================================
# 2. 策略逻辑函数
# ==========================================
def get_eagle_eye_stocks(token, s_date, e_date):
    # 转换日期格式为 Tushare 要求的 YYYYMMDD
    s_str = s_date.strftime('%Y%m%d')
    e_str = e_date.strftime('%Y%m%d')
    
    # 设置 Token
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # 测试 Token 是否有效
        pro.trade_cal(exchange='', start_date=s_str, end_date=e_str, is_open='1')
    except Exception as e:
        st.error(f"Token 无效或连接失败，错误信息: {e}")
        return pd.DataFrame()

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 获取交易日历
    try:
        cal = pro.trade_cal(exchange='', start_date=s_str, end_date=e_str, is_open='1')
        trade_days = cal['cal_date'].tolist()
    except Exception as e:
        st.error(f"获取交易日历失败: {e}")
        return pd.DataFrame()

    trade_log = []
    
    if len(trade_days) < 2:
        st.warning("选定的日期范围内交易日不足 2 天，无法进行回测。")
        return pd.DataFrame()

    # 循环遍历（保留最后一天作为选股日，前面的做回测）
    total_days = len(trade_days) - 1
    
    for i in range(total_days):
        date_today = trade_days[i]      # T日 (信号日)
        date_tomorrow = trade_days[i+1] # T+1日 (验证日)
        
        # 更新进度条
        progress = (i + 1) / total_days
        progress_bar.progress(progress)
        status_text.text(f"正在扫描: {date_today} ...")
        
        # --- A. 获取 T日 基础数据 ---
        try:
            df_today = pro.daily(trade_date=date_today)
            df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
            df_today = pd.merge(df_today, df_basic, on='ts_code')
            # 简单过滤
            df_today = df_today[~df_today['name'].str.contains('ST')]
        except:
            continue

        # --- B. 形态初筛 (射击之星) ---
        candidates = []
        for idx, row in df_today.iterrows():
            if row['close'] == 0 or row['pre_close'] == 0: continue
            
            body_top = max(row['open'], row['close'])
            upper_shadow = (row['high'] - body_top) / row['pre_close'] * 100
            pct_chg = row['pct_chg']
            
            # 条件：长上影 > 3%，涨跌幅在合理区间(-2% 到 8%)
            if upper_shadow > 3.0 and -2 < pct_chg < 8:
                candidates.append(row['ts_code'])
        
        if not candidates: continue
        
        # --- C. 筹码测谎 (10000积分核心) ---
        # 限制数量以防超时，实盘可放开
        check_list = candidates[:30] 
        real_targets = []
        
        for code in check_list:
            try:
                # 获取筹码数据 (cyq_perf)
                df_cyq = pro.cyq_perf(ts_code=code, trade_date=date_today)
                if df_cyq.empty: continue
                
                profit_rate = df_cyq.iloc[0]['profit_rate']
                
                # 核心过滤：获利盘 > 85%
                if profit_rate > 85:
                    real_targets.append(code)
            except:
                time.sleep(0.1) # 防止接口超限
                continue
        
        if not real_targets: continue
        
        # --- D. 次日验证 (T+1) ---
        try:
            df_next = pro.daily(trade_date=date_tomorrow, ts_code=','.join(real_targets))
        except:
            continue
            
        for idx, row_next in df_next.iterrows():
            code = row_next['ts_code']
            stock_name = df_basic[df_basic['ts_code'] == code]['name'].values[0] if not df_basic.empty else code
            
            # 获取 T日收盘价
            close_T = df_today[df_today['ts_code'] == code]['close'].values[0]
            
            # 验证条件：次日高开
            open_T1 = row_next['open']
            
            if open_T1 > close_T:
                close_T1 = row_next['close']
                profit_pct = (close_T1 - open_T1) / open_T1 * 100
                
                trade_log.append({
                    '信号日期': date_today,
                    '买入日期': date_tomorrow,
                    '代码': code,
                    '名称': stock_name,
                    'T日获利盘(%)': 'High (>85%)',
                    '买入价': open_T1,
                    '卖出价(收盘)': close_T1,
                    '单日收益率(%)': round(profit_pct, 2)
                })

    progress_bar.empty()
    status_text.text("扫描完成！")
    return pd.DataFrame(trade_log)

# ==========================================
# 3. 主运行区
# ==========================================

if run_btn:
    if not user_token:
        st.error("❌ 请先在左侧侧边栏输入您的 Tushare Token！")
    else:
        with st.spinner('正在连接 Tushare 数据中心进行深度扫描...这可能需要几分钟...'):
            df_result = get_eagle_eye_stocks(user_token, start_date, end_date)
            
        if not df_result.empty:
            # 展示汇总数据
            st.success(f"✅ 回测完成！共触发交易 {len(df_result)} 次")
            
            col1, col2, col3 = st.columns(3)
            win_rate = len(df_result[df_result['单日收益率(%)'] > 0]) / len(df_result) * 100
            total_return = df_result['单日收益率(%)'].sum()
            avg_return = df_result['单日收益率(%)'].mean()
            
            col1.metric("胜率 (Win Rate)", f"{win_rate:.2f}%")
            col2.metric("累计收益 (Total)", f"{total_return:.2f}%")
            col3.metric("平均单笔收益", f"{avg_return:.2f}%")
            
            st.markdown("---")
            st.markdown("### 📋 详细交易记录")
            
            # 颜色高亮显示收益
            def highlight_profit(val):
                color = 'red' if val > 0 else 'green'
                return f'color: {color}'

            st.dataframe(df_result.style.applymap(highlight_profit, subset=['单日收益率(%)']))
            
        else:
            st.info("在此时间段内未发现符合【鹰眼·假摔】形态的股票，或 Token 权限不足/错误。")

