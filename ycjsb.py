import streamlit as st
import pandas as pd
import tushare as ts
import warnings
from datetime import datetime, timedelta  # 补上了这一行

warnings.filterwarnings("ignore")

st.set_page_config(page_title="V33 照妖镜", layout="wide")
st.title("🪞 V33 镜像回溯 (照妖镜)")
st.markdown("直接查看 12 只翻倍股在启动前 3 天的真实 K 线形态，不再盲猜。")

# 12 罗汉启动日 (Launch Dates)
TARGETS = {
    '申菱环境': ('301018.SZ', '20251217'),
    '利通电子': ('603629.SH', '20251210'),
    '田中精机': ('300461.SZ', '20260119'),
    '宏和科技': ('603256.SH', '20251208'),
    '致尚科技': ('301486.SZ', '20251205'),
    '罗博特科': ('300757.SZ', '20260203'),
    '炬光科技': ('688167.SH', '20260203'),
    '嘉美包装': ('002969.SZ', '20251217'),
    '横店影视': ('603103.SH', '20260108'),
    '长飞光纤': ('601869.SH', '20251208'),
    '博迁新材': ('605376.SH', '20251218'),
    '振德医疗': ('603301.SH', '20250908')
}

token = st.sidebar.text_input("Tushare Token", type="password")

if st.sidebar.button("启动照妖镜"):
    if not token:
        st.error("请输入 Token")
    else:
        ts.set_token(token)
        pro = ts.pro_api()
        
        results = []
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total = len(TARGETS)
        
        for i, (name, (code, launch_date)) in enumerate(TARGETS.items()):
            status_text.text(f"正在回溯: {name}...")
            progress_bar.progress((i + 1) / total)
            
            try:
                # 获取启动日前 25 天的数据 (多取点算指标)
                end_dt = datetime.strptime(launch_date, '%Y%m%d')
                start_dt = end_dt - timedelta(days=40) 
                
                df = pro.daily(ts_code=code, start_date=start_dt.strftime('%Y%m%d'), end_date=launch_date)
                df = df.sort_values('trade_date').reset_index(drop=True)
                
                if len(df) < 5:
                    st.warning(f"{name}: 数据不足")
                    continue
                    
                # 取最后 4 天 (T-3, T-2, T-1, T=Launch)
                # Launch Day 是最后一天
                launch_idx = len(df) - 1
                
                # --- 计算形态 (T-3 到 启动日) ---
                days_info = []
                for j in range(3, -1, -1): # 3, 2, 1, 0
                    idx = launch_idx - j
                    if idx < 0: continue
                    
                    row = df.iloc[idx]
                    pct = row['pct_chg']
                    
                    # 定义 K 线颜色和形态
                    icon = "🔴" if pct > 0 else ("🟢" if pct < 0 else "⚪")
                    if pct > 9.0: type_str = "涨停"
                    elif pct > 5.0: type_str = "大阳"
                    elif pct > 0: type_str = "小阳"
                    elif pct > -5.0: type_str = "小阴"
                    else: type_str = "大阴"
                    
                    days_info.append(f"{icon}{type_str}({pct:.1f}%)")
                
                pattern_str = " -> ".join(days_info)
                
                # --- 计算 RSI(6) at T-1 (启动前一天) ---
                # 简单模拟 RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(6).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
                rs = gain / (loss + 0.001)
                rsi = 100 - (100 / (1 + rs))
                rsi_t1 = rsi.iloc[launch_idx-1]
                
                # --- 计算换手率 at T-1 ---
                # 需要调 daily_basic，为了速度这里简化，只看涨跌幅形态
                # 如果需要换手率，可以再加一个 API 请求，但会变慢
                
                results.append({
                    '名称': name,
                    '启动日': launch_date,
                    '启动前3天走势 (T-3 -> T-2 -> T-1 -> 启动)': pattern_str,
                    '启动前RSI(6)': f"{rsi_t1:.1f}"
                })
                
            except Exception as e:
                st.error(f"{name} 回溯失败: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        st.success("回溯完成！真相如下：")
        st.table(pd.DataFrame(results))
        
        st.info("""
        💡 **看图说话**：
        1. **看连阳**：如果全是 🔴小阳，说明"蚂蚁上树"是对的。
        2. **看洗盘**：如果中间夹杂了 🟢小阴，说明主力在洗盘，V32 过滤太严了。
        3. **看力度**：启动那一下是不是都是"涨停"？如果是，说明必须做首板。
        """)
