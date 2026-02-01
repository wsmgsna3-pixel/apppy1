import streamlit as st
import tushare as ts
import pandas as pd
import time
from datetime import datetime, timedelta
import os

# ==========================================
# Streamlit 页面配置
# ==========================================
st.set_page_config(page_title="鹰眼·资金背离版", layout="wide")

st.title("🦅 鹰眼·主力假摔 (资金背离版)")
st.markdown("""
**策略核心升级：**
放弃不稳定的筹码数据，改用 **10000积分专属的 `moneyflow` (个股资金流向)**。
**寻找背离：** 股价收出长上影线（看似出货），但主力资金（特大单+大单）却是**净买入**的股票。
""")

# ==========================================
# 1. 缓存化数据获取 (资金流向版)
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
    except:
        return pd.DataFrame()

@st.cache_data(persist="disk", show_spinner=False)
def get_moneyflow(token, code, date_str):
    """获取个股资金流向，替代不稳定的筹码接口"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # moneyflow 接口非常稳定
        df = pro.moneyflow(ts_code=code, trade_date=date_str)
        return df
    except:
        return pd.DataFrame()

# ==========================================
# 2. 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数控制台")
    user_token = st.text_input("Tushare Token (必填):", type="password")
    
    st.subheader("🔍 形态与资金阈值")
    shadow_threshold = st.slider("上影线长度 (%)", 1.0, 10.0, 3.0, 0.5)
    # 资金背离的强度：主力净买入额（万元）
    net_buy_threshold = st.slider("主力净买入至少 (万元)", 100, 5000, 500, 100, help="虽然K线难看，但主力必须净买入超过此金额")
    
    scan_limit = st.slider("每日扫描热门股数", 20, 200, 100, 10)
    
    st.subheader("📅 回测区间")
    # 资金流数据通常T+1早上更新，回测最近的也没问题
    default_start = datetime.now() - timedelta(days=14)
    default_end = datetime.now()
    start_date = st.date_input("开始日期", default_start)
    end_date = st.date_input("结束日期", default_end)
    
    run_btn = st.button("🚀 启动背离扫描")
    
    if st.button("🧹 清除缓存"):
        st.cache_data.clear()
        st.success("缓存已清除")

# ==========================================
# 3. 策略主逻辑
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
        st.error("日期获取失败")
        return

    log_area = st.container()
    progress_bar = st.progress(0)
    results = []
    
    total_days = len(trade_days) - 1
    if total_days < 1:
        st.warning("回测区间过短")
        return

    with log_area:
        st.write("### 📜 资金背离扫描日志")

    for i in range(total_days):
        date_today = trade_days[i]
        date_next = trade_days[i+1]
        progress_bar.progress((i+1)/total_days)
        
        # 1. 获取日线
        df_today = get_cached_daily(user_token, date_today)
        if df_today.empty: continue
        df_today = df_today[~df_today['name'].str.contains('ST')]
        
        # 2. 形态初筛 (射击之星)
        df_today['body_top'] = df_today[['open', 'close']].max(axis=1)
        df_today['upper_shadow'] = (df_today['high'] - df_today['body_top']) / df_today['pre_close'] * 100
        
        # 筛选：长上影，且成交量不能太小（资金流分析需要量）
        mask = (df_today['upper_shadow'] > shadow_threshold) & \
               (df_today['pct_chg'] > -4) & (df_today['pct_chg'] < 8)
        
        candidates = df_today[mask].copy()
        
        if len(candidates) == 0:
            with log_area:
                st.write(f"📅 {date_today}: 无形态符合股票")
            continue
            
        # 智能排序：按成交额排序
        candidates = candidates.sort_values(by='amount', ascending=False)
        targets = candidates.head(scan_limit)['ts_code'].tolist()
        
        with log_area:
            st.write(f"📅 {date_today}: 形态初筛 {len(candidates)} 只，正在透视前 {len(targets)} 只资金流向...")
        
        passed_codes = []
        
        # 3. 资金测谎 (MoneyFlow)
        for code in targets:
            df_mf = get_moneyflow(user_token, code, date_today)
            
            if not df_mf.empty:
                # 核心字段：
                # buy_lg_vol: 大单买入量
                # buy_elg_vol: 特大单买入量
                # net_mf_vol: 净流入量 (单位：手) -> 我们要转成金额近似值
                # net_mf_amount: 净流入额 (单位：万元) -> 这个最直接！
                
                row = df_mf.iloc[0]
                net_amount = row['net_mf_amount'] # 主力净流入金额(万元)
                
                # === 变态逻辑 ===
                # K线难看(上影线)，散户在跑，但主力净流入 > 500万 (或者你设定的阈值)
                if net_amount > net_buy_threshold:
                    passed_codes.append({
                        'code': code,
                        'net_amount': net_amount
                    })
                    
                    stock_name = candidates[candidates['ts_code']==code]['name'].values[0]
                    with log_area:
                        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;💰 **背离发现**: {stock_name} | 上影线: {candidates[candidates['ts_code']==code]['upper_shadow'].values[0]:.1f}% | **主力净买: {net_amount:.0f}万元**")

        if not passed_codes:
            with log_area:
                st.write("&nbsp;&nbsp;&nbsp;&nbsp;❌ 本日无资金背离标的 (主力也在跑)")
            continue
            
        # 4. 次日验证
        df_next = get_cached_daily(user_token, date_next)
        if df_next.empty: continue
        
        for item in passed_codes:
            code = item['code']
            net_amt = item['net_amount']
            
            row_next = df_next[df_next['ts_code'] == code]
            if row_next.empty: continue
            
            open_T1 = row_next.iloc[0]['open']
            close_T1 = row_next.iloc[0]['close']
            
            # T日收盘价
            close_T = candidates[candidates['ts_code'] == code]['close'].values[0]
            stock_name = candidates[candidates['ts_code'] == code]['name'].values[0]
            
            # 必须高开 (弱转强)
            if open_T1 > close_T:
                profit_pct = (close_T1 - open_T1) / open_T1 * 100
                
                results.append({
                    '日期': date_today,
                    '代码': code,
                    '名称': stock_name,
                    '主力净买(万)': int(net_amt),
                    '买入价': open_T1,
                    '当日收益(%)': round(profit_pct, 2)
                })

    progress_bar.empty()
    
    # 5. 结果展示
    if results:
        df_res = pd.DataFrame(results)
        st.success(f"🎉 扫描完成！发现 {len(df_res)} 次主力骗线机会")
        
        c1, c2, c3 = st.columns(3)
        wins = df_res[df_res['当日收益(%)'] > 0]
        c1.metric("胜率", f"{len(wins)/len(df_res)*100:.1f}%")
        c2.metric("平均收益", f"{df_res['当日收益(%)'].mean():.2f}%")
        c3.metric("总收益", f"{df_res['当日收益(%)'].sum():.2f}%")
        
        st.dataframe(df_res.style.applymap(lambda x: f'color: {"red" if x>0 else "green"}', subset=['当日收益(%)']), use_container_width=True)
    else:
        st.warning("未发现符合条件的标的。请尝试降低【主力净买入】阈值。")

if run_btn:
    if not user_token:
        st.error("请输入 Token")
    else:
        run_strategy()
