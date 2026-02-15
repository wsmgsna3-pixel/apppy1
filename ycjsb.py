import streamlit as st
import pandas as pd
import tushare as ts
import warnings

warnings.filterwarnings("ignore")

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
        
        for name, (code, launch_date) in TARGETS.items():
            # 获取启动日前 5 天的数据
            end_dt = datetime.strptime(launch_date, '%Y%m%d')
            start_dt = end_dt - timedelta(days=20) # 多取点算指标
            
            df = pro.daily(ts_code=code, start_date=start_dt.strftime('%Y%m%d'), end_date=launch_date)
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            if len(df) < 5:
                st.warning(f"{name}: 数据不足")
                continue
                
            # 取最后 4 天 (T-3, T-2, T-1, T=Launch)
            # Launch Day 是最后一天
            launch_idx = len(df) - 1
            
            # 计算形态
            days = []
            for i in range(3, -1, -1): # 3, 2, 1, 0
                idx = launch_idx - i
                if idx < 0: continue
                
                row = df.iloc[idx]
                pct = row['pct_chg']
                color = "🔴" if pct > 0 else "Vk" # 🔴阳 🟢阴
                days.append(f"{color} {pct:.1f}%")
            
            # 组合形态字符串
            pattern_str = " -> ".join(days)
            
            # 计算 RSI(6) at T-1
            # 简单模拟
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
            rs = gain / (loss + 0.001)
            rsi = 100 - (100 / (1 + rs))
            rsi_t1 = rsi.iloc[launch_idx-1]
            
            results.append({
                '名称': name,
                '启动日': launch_date,
                '形态 (T-3 -> 启动)': pattern_str,
                '启动前RSI': f"{rsi_t1:.1f}"
            })
            
        st.table(pd.DataFrame(results))
        
        st.info("💡 **分析指南**：\n"
                "1. 看 **形态**：是不是全是红的(蚂蚁上树)？还是夹杂了绿的(洗盘)？\n"
                "2. 看 **RSI**：启动前到底是 50 还是 70？")
