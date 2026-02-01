import tushare as ts
import pandas as pd
import time
import numpy as np

# ==========================================
# 1. 基础配置与 Token 输入
# ==========================================
print("请在下方输入您的 Tushare 10000 积分 Token：")
MY_TOKEN = '这里填入你的TOKEN'  # 你可以在这里直接修改，或者运行时输入
# MY_TOKEN = input("请输入 Token: ") # 如果想每次手动输，取消这行注释

ts.set_token(MY_TOKEN)
pro = ts.pro_api()

# ==========================================
# 2. 策略核心逻辑函数
# ==========================================

def check_signal_and_backtest(start_date, end_date):
    print(f"\n🚀 开始回测策略：【鹰眼·假摔猎杀】")
    print(f"📅 回测区间：{start_date} 至 {end_date}")
    print("⚠️ 注意：由于涉及筹码数据，回测速度取决于网络和接口限制，请耐心等待...")
    
    # 获取交易日历
    cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
    trade_days = cal['cal_date'].tolist()
    
    trade_log = [] # 交易记录
    
    for i in range(len(trade_days) - 1):
        date_today = trade_days[i]      # T日 (信号日)
        date_tomorrow = trade_days[i+1] # T+1日 (交易日)
        
        print(f"正在扫描: {date_today} -> 验证: {date_tomorrow}")
        
        # --- A. 获取 T日 全市场行情 ---
        try:
            df_today = pro.daily(trade_date=date_today)
            df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
            df_today = pd.merge(df_today, df_basic, on='ts_code')
            
            # 过滤科创北交所(可选，为了稳健先只测主板和创业板)
            df_today = df_today[~df_today['market'].str.contains('北交所')]
            
        except Exception as e:
            print(f"数据获取失败: {e}")
            continue

        # --- B. 形态初筛 (Shooting Star) ---
        candidates = []
        for idx, row in df_today.iterrows():
            if row['close'] == 0 or row['pre_close'] == 0: continue
            
            # 形态定义
            body_top = max(row['open'], row['close'])
            upper_shadow = (row['high'] - body_top) / row['pre_close'] * 100
            pct_chg = row['pct_chg']
            
            # 1. 长上影线 > 3%
            # 2. 涨幅不能太大（比如超过8%可能是真板炸了，波动太大），也不能大跌
            if upper_shadow > 3.0 and -2 < pct_chg < 8:
                candidates.append(row['ts_code'])
        
        if not candidates: continue
        
        # --- C. 筹码测谎 (调用 cyq_perf) ---
        # 这一步最慢，为了演示回测效率，我们只取前20个候选做示例
        # 实盘请去掉 [:20] 限制
        real_targets = []
        
        for code in candidates[:30]: 
            try:
                # 获取筹码数据
                df_cyq = pro.cyq_perf(ts_code=code, trade_date=date_today)
                if df_cyq.empty: continue
                
                profit_rate = df_cyq.iloc[0]['profit_rate']
                
                # 核心条件：虽然炸板/回落，获利盘依然 > 85%
                if profit_rate > 85:
                    real_targets.append(code)
            except:
                continue
                
        if not real_targets: continue
        
        # --- D. 次日验证 (T+1 弱转强) ---
        # 获取这些票第二天的行情
        if not real_targets: continue
        
        try:
            # 批量获取第二天行情
            df_next = pro.daily(trade_date=date_tomorrow, ts_code=','.join(real_targets))
        except:
            continue
            
        for idx, row_next in df_next.iterrows():
            code = row_next['ts_code']
            
            # 获取 T日收盘价
            close_T = df_today[df_today['ts_code'] == code]['close'].values[0]
            
            # === 买入条件：弱转强 ===
            # T+1日 开盘价 > T日收盘价 (高开)
            open_T1 = row_next['open']
            
            if open_T1 > close_T:
                # 模拟交易：开盘买入，收盘卖出 (日内超短)
                # 或者：开盘买入，持有看后续涨幅
                
                close_T1 = row_next['close']
                
                # 计算收益率
                profit_pct = (close_T1 - open_T1) / open_T1 * 100
                
                trade_log.append({
                    '信号日': date_today,
                    '交易日': date_tomorrow,
                    '代码': code,
                    'T日获利盘': 'High',
                    '买入价(T+1开盘)': open_T1,
                    '卖出价(T+1收盘)': close_T1,
                    '收益率(%)': round(profit_pct, 2)
                })
    
    # ==========================================
    # 3. 回测报告生成
    # ==========================================
    if not trade_log:
        print("\n没有触发交易。")
        return pd.DataFrame()
        
    df_result = pd.DataFrame(trade_log)
    
    print("\n" + "="*30)
    print("📊 回测总结报告")
    print("="*30)
    print(f"总交易次数: {len(df_result)}")
    print(f"胜率 (收益>0): {len(df_result[df_result['收益率(%)'] > 0]) / len(df_result) * 100:.2f}%")
    print(f"平均单笔收益: {df_result['收益率(%)'].mean():.2f}%")
    print(f"累计收益率(简单叠加): {df_result['收益率(%)'].sum():.2f}%")
    print(f"单笔最大亏损: {df_result['收益率(%)'].min():.2f}%")
    print(f"单笔最大盈利: {df_result['收益率(%)'].max():.2f}%")
    print("="*30)
    
    return df_result

# ==========================================
# 4. 运行入口
# ==========================================
# 建议先测最近半个月，因为循环取筹码数据比较慢
# 修改这里的日期进行回测
# df_backtest = check_signal_and_backtest('20240501', '20240520') 
# print(df_backtest.head(10))

if __name__ == '__main__':
    # 示例：回测最近一周
    # 获取今天的日期
    today = time.strftime("%Y%m%d")
    # 随便设个开始日期，比如10天前
    start_str = '20240501' # 你可以手动修改这个
    
    # 运行
    res = check_signal_and_backtest(start_str, '20240524')
    if not res.empty:
        print("\n详细交易单：")
        print(res[['交易日', '代码', '买入价(T+1开盘)', '收益率(%)']])
