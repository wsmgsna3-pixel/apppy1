import streamlit as st
import tushare as ts
import pandas as pd
import time
from datetime import datetime, timedelta
import os

# ==========================================
# Streamlit 页面配置
# ==========================================
st.set_page_config(page_title="鹰眼·调试版", layout="wide")

st.title("🦅 鹰眼·假摔猎杀 (深度调试版)")
st.error("⚠️ 调试重点：此版本会显示 Tushare 返回的真实错误信息，且强制降速以防封禁。")

# ==========================================
# 1. 缓存化数据获取 (带错误透传)
# ==========================================
@st.cache_data(persist="disk", show_spinner=False)
def get_cached_daily(token, date_str):
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.daily(trade_date=date_str)
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
        if df.empty: return pd.DataFrame()
        return pd.merge(df, df_basic, on='ts_code')
    except Exception:
        return pd.DataFrame()

# ⚠️ 注意：这里去掉了 silent error，为了看清真相
def get_cyq_debug(token, code, date_str):
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # 强制休眠，防止触发每分钟频次限制
        time.sleep(0.25) 
        df = pro.cyq_perf(ts_code=code, trade_date=date_str)
        return df, None # Data, Error
    except Exception as e:
        return pd.DataFrame(), str(e) # Empty, Error Message

# ==========================================
# 2. 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数控制台")
    user_token = st.text_input("Tushare Token (必填):", type="password")
    
    st.info("💡 建议：由于筹码数据计算滞后，请尽量回测 **3天前** 的数据。")
    
    # 默认回测半个月前的数据，避开滞后区
    default_start = datetime.now() - timedelta(days=20)
    default_end = datetime.now() - timedelta(days=5)
    
    start_date = st.date_input("开始日期", default_start)
    end_date = st.date_input("结束日期", default_end)
    
    profit_threshold = st.slider("筹码获利盘 (%)", 0, 99, 50, 5)
    scan_limit = st.slider("每日扫描数", 10, 50, 20, 5, help="调试期间建议设小一点，比如20")
    
    run_btn = st.button("🚀 启动调试扫描")
    
    if st.button("🧹 清除缓存"):
        st.cache_data.clear()
        st.success("缓存已清除")

# ==========================================
# 3. 主逻辑
# ==========================================
def run_debug():
    ts.set_token(user_token)
    pro = ts.pro_api()
    
    s_str = start_date.strftime('%Y%m%d')
    e_str = end_date.strftime('%Y%m%d')
    
    try:
        cal = pro.trade_cal(exchange='', start_date=s_str, end_date=e_str, is_open='1')
        trade_days = cal['cal_date'].tolist()
    except Exception as e:
        st.error(f"日历获取失败: {e}")
        return

    log_area = st.container()
    
    if len(trade_days) < 2:
        st.warning("交易日不足")
        return

    # 循环
    for i in range(len(trade_days)-1):
        date_today = trade_days[i]
        
        # 1. 获取日线
        df_today = get_cached_daily(user_token, date_today)
        if df_today.empty: continue
        df_today = df_today[~df_today['name'].str.contains('ST')]
        
        # 2. 筛选形态
        df_today['body_top'] = df_today[['open', 'close']].max(axis=1)
        df_today['upper_shadow'] = (df_today['high'] - df_today['body_top']) / df_today['pre_close'] * 100
        mask = (df_today['upper_shadow'] > 3.0) & (df_today['pct_chg'] > -3) & (df_today['pct_chg'] < 8)
        
        candidates = df_today[mask].sort_values(by='amount', ascending=False)
        targets = candidates.head(scan_limit)['ts_code'].tolist()
        
        with log_area:
            st.write(f"📅 **{date_today}**: 初筛 {len(candidates)} 只，尝试获取前 {len(targets)} 只筹码...")
            
            success_count = 0
            empty_count = 0
            error_msg = ""
            
            # 3. 逐个获取筹码 (不使用缓存函数，直接调用 debug 函数)
            debug_progress = st.empty()
            
            for idx, code in enumerate(targets):
                debug_progress.text(f"请求中: {code} ({idx+1}/{len(targets)})")
                
                # 调用接口
                df_cyq, error = get_cyq_debug(user_token, code, date_today)
                
                if error:
                    # 捕获到了真实的报错！
                    st.error(f"❌ 接口报错 ({code}): {error}")
                    error_msg = error
                    break # 报错直接停止，不用再跑了
                
                if df_cyq.empty:
                    empty_count += 1
                else:
                    # 有数据！
                    if 'profit_rate' in df_cyq.columns:
                        p = df_cyq.iloc[0]['profit_rate']
                        success_count += 1
                        if p > profit_threshold:
                            name = candidates[candidates['ts_code']==code]['name'].values[0]
                            st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;✅ {name}: 获利盘 {p:.2f}%")
                    else:
                        st.warning(f"⚠️ {code} 返回了数据但没有 profit_rate 列")
            
            debug_progress.empty()
            
            # 诊断总结
            if error_msg:
                st.stop() # 停止运行
            elif success_count > 0:
                st.info(f"✅ {date_today} 测试通过: 成功获取 {success_count} 条，空数据 {empty_count} 条")
            else:
                st.warning(f"⚠️ {date_today} 全军覆没: 请求了 {len(targets)} 次，全部返回空数据。")
                st.markdown("""
                **可能原因分析：**
                1. **数据滞后**：Tushare 后台还没计算出这一天的筹码（最可能）。
                2. **权限问题**：虽然不太像，但如果 20 天前的数据也这样，就是权限问题。
                """)

if run_btn:
    if not user_token:
        st.error("请输入 Token")
    else:
        run_debug()
