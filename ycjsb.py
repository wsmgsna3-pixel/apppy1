import tushare as ts
import pandas as pd

# 在这里输入您的 Token
my_token = "您的TOKEN"

ts.set_token(my_token)
pro = ts.pro_api()

print("正在测试筹码接口权限...")

try:
    # 尝试拉取 宁德时代(300750.SZ) 某一天的筹码数据
    df = pro.cyq_perf(ts_code='300750.SZ', trade_date='20250105', fields='ts_code,trade_date,profit_pct')
    
    if df.empty:
        print("❌ 测试失败：接口返回为空。")
        print("原因可能是：")
        print("1. 您的积分虽然够，但该特色数据权限未开通（需单独捐赠或申请）。")
        print("2. 恰好该日无数据。")
    else:
        print("✅ 测试成功！获取到数据：")
        print(df)
        print(f"获利盘比例: {df.iloc[0]['profit_pct']}%")
        
except Exception as e:
    print(f"❌ 发生错误：{e}")
    print("结论：接口调用报错，权限受限或网络超时。")
