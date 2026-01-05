import streamlit as st
import tushare as ts
import pandas as pd

st.set_page_config(page_title="Tushare 权限诊断", layout="centered")

st.title("🔍 Tushare 筹码接口权限诊断")
st.markdown("此工具用于测试您的 Token 是否拥有 `cyq_perf` (每日筹码及胜率) 的调用权限。")

# --- 安全输入框 ---
token = st.text_input("请在此输入您的 Tushare Token", type="password")

# --- 测试按钮 ---
if st.button("开始诊断", type="primary"):
    if not token:
        st.error("❌ 请先输入 Token")
        st.stop()
    
    # 设置 Token
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        
        st.info("正在尝试连接 Tushare 服务器...")
        
        # 测试 1: 基础连接测试 (拉取平安银行作为基准)
        st.write("1️⃣ 第一步：测试基础连接 (调用 daily 接口)...")
        df_base = pro.daily(ts_code='000001.SZ', start_date='20241220', end_date='20241220')
        if df_base.empty:
            st.error("❌ 基础连接失败！可能是 Token 错误或网络问题。")
            st.stop()
        else:
            st.success("✅ 基础连接正常，Token 有效。")
            
        # 测试 2: 筹码权限测试
        # 选取一个确定有交易的日期 (2024-11-01 是周五) 以排除节假日因素
        test_date = '20241101'
        test_code = '300750.SZ' # 宁德时代
        
        st.write(f"2️⃣ 第二步：测试筹码接口 (cyq_perf) - 日期: {test_date}...")
        
        try:
            df_cyq = pro.cyq_perf(ts_code=test_code, trade_date=test_date, fields='ts_code,trade_date,profit_pct')
            
            if df_cyq.empty:
                st.error("❌ **测试失败：接口返回为空**")
                st.warning("""
                **诊断结论：**
                您的积分可能足够 (10000分)，但 **[每日筹码及胜率]** 这个特定接口的权限可能未开通。
                
                Tushare 规则复杂，有时 10000 积分只包含通用数据，而筹码数据属于“特色数据”，可能需要单独申请或处于维护中。
                """)
            else:
                st.balloons()
                st.success(f"✅ **测试成功！您拥有筹码数据权限！**")
                st.write("⬇️ 获取到的数据样本：")
                st.dataframe(df_cyq)
                st.markdown(f"**获利盘比例:** `{df_cyq.iloc[0]['profit_pct']}%`")
                st.info("您可以放心地使用 V36.0 版本了。")
                
        except Exception as e:
            st.error(f"❌ 接口调用报错: {e}")
            st.markdown("这通常意味着权限被拒绝 (No Permission)。")

    except Exception as e:
        st.error(f"发生未知错误: {e}")
