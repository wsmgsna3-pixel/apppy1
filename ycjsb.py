import tushare as ts
import pandas as pd
import time
import datetime
from tenacity import retry, stop_after_attempt, wait_fixed

# ================= 1. 安全配置与工具函数 =================

def get_user_token():
    """
    安全获取用户Token，不保存到文件
    """
    print("="*50)
    token = input("请输入您的 Tushare Token (输入后回车): ").strip()
    if len(token) < 20:
        print("错误: Token 长度看起来不对，请重新运行程序。")
        exit()
    return token

def get_real_trade_date(pro):
    """
    自动识别最近的一个交易日
    如果是周六(今天)，会自动定位到本周五
    """
    today = datetime.datetime.now().strftime('%Y%m%d')
    try:
        # 获取包含今天在内的过去20天交易日历
        start_check = (datetime.datetime.now() - datetime.timedelta(days=20)).strftime('%Y%m%d')
        df = pro.trade_cal(exchange='', start_date=start_check, end_date=today, is_open='1')
        
        if df.empty:
            print("错误: 无法获取交易日历，请检查网络。")
            exit()
            
        # 取最后一个日期，即为最近的交易日
        real_date = df['cal_date'].values[-1]
        print(f"系统检测: 今天是 {today}，最近的有效交易日是 【{real_date}】")
        return real_date
    except Exception as e:
        print(f"获取交易日历失败: {e}")
        exit()

# 重试装饰器：用于不稳定的网络请求
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def fetch_chips_data(pro, ts_code, trade_date):
    """单独封装筹码接口，便于重试"""
    return pro.cyq_perf(ts_code=ts_code, start_date=trade_date, end_date=trade_date)

# ================= 2. 策略核心逻辑 =================

class StrategyRunner:
    def __init__(self, token):
        self.ts = ts
        self.ts.set_token(token)
        try:
            self.pro = self.ts.pro_api()
        except Exception as e:
            print(f"Token 无效或连接失败: {e}")
            exit()
            
        self.trade_date = get_real_trade_date(self.pro)
        
    def get_basic_pool(self):
        """
        第一步：初筛 (剔除ST、新股)
        """
        print(f"\n正在初始化股票池 (日期基准: {self.trade_date})...")
        try:
            # 这里的 fields 加上 list_date 用于过滤新股
            df = self.pro.stock_basic(exchange='', list_status='L', 
                                    fields='ts_code,symbol,name,area,industry,list_date')
            
            # 1. 剔除ST
            df = df[~df['name'].str.contains('ST')]
            df = df[~df['name'].str.contains('退')]
            
            # 2. 剔除上市不满 6 个月的次新股 (数据太少，技术面不稳定)
            # 将 list_date 转为 datetime 对象
            df['list_date'] = pd.to_datetime(df['list_date'])
            # 计算半年前的时间点
            limit_date = pd.to_datetime(self.trade_date) - pd.Timedelta(days=180)
            df = df[df['list_date'] < limit_date]
            
            codes = df['ts_code'].tolist()
            print(f"基础过滤完成，剩余 {len(codes)} 只标的等待扫描。")
            return codes
        except Exception as e:
            print(f"获取基础数据失败: {e}")
            return []

    def check_weekly_low(self, ts_code):
        """
        第二步：周线逻辑 (判断相对低位)
        """
        try:
            # 获取最近 60 周数据
            df = self.pro.weekly(ts_code=ts_code, end_date=self.trade_date, limit=60)
            if df is None or len(df) < 50: 
                return False
            
            # 简单有效的相对位置算法：(当前价 - 50周最低) / (50周最高 - 50周最低)
            # 这种算法不需要复权因子也能大致判断区间
            last_close = df.iloc[0]['close'] # 最近一周收盘价
            period_high = df['high'].max()
            period_low = df['low'].min()
            
            if period_high == period_low: return False # 防止除以0
            
            position = (last_close - period_low) / (period_high - period_low)
            
            # 判定标准：处于过去一年价格区间的底部 30% 以内
            if position <= 0.30:
                return True
            return False
            
        except Exception:
            # 任何报错都视为不符合，继续下一个
            return False

    def check_daily_trigger(self, ts_code):
        """
        第三步：日线买入信号
        """
        try:
            # 获取最近 10 个交易日的数据，用于判断趋势
            df = self.pro.daily(ts_code=ts_code, end_date=self.trade_date, limit=10)
            if df is None or len(df) < 5: 
                return False
            
            # 数据是按日期倒序的 (index 0 是最新一天)
            today = df.iloc[0]
            
            # 1. 涨幅过滤：最近一天涨幅 > 2% (有资金点火) 且 < 8% (不追高/不追涨停)
            if not (2.0 < today['pct_chg'] < 8.0):
                return False
            
            # 2. 量能过滤：量比 > 1.2 (简化版，今日量 > 5日均量 * 1.2)
            # 注意：DataFrame切片 [1:6] 代表过去5天
            avg_vol_5 = df.iloc[1:6]['vol'].mean()
            if avg_vol_5 == 0: return False
            
            if today['vol'] < 1.2 * avg_vol_5:
                return False
                
            return True
            
        except Exception:
            return False

    def check_chips_structure(self, ts_code):
        """
        第四步：筹码验证 (最耗时，放在最后)
        """
        try:
            # 调用带重试机制的函数
            df = fetch_chips_data(self.pro, ts_code, self.trade_date)
            
            if df is None or df.empty: 
                return False
            
            row = df.iloc[0]
            winner_rate = row['winner_rate'] # 获利比例
            
            # 逻辑：
            # 1. 极度缩量跌无可跌 (winner_rate < 5%) -> 反弹一触即发
            # 2. 或者底部吸筹结束，刚突破 (50% < winner_rate < 80%)
            # 这里为了安全，我们选获利盘比较干净的，或者刚起步的
            
            if winner_rate < 15 or (50 < winner_rate < 85):
                return True
                
            return False
            
        except Exception:
            # 网络实在不行就跳过
            return False

    def run(self):
        codes = self.get_basic_pool()
        if not codes: return
        
        candidates = []
        print("\n=== 开始执行选股策略 (按 Ctrl+C 可中止) ===")
        
        # 计数器
        checked_count = 0
        
        # 建议：为了演示速度，这里可以先切片 codes[:200] 测试
        # 实盘请去掉 [:200]
        # target_pool = codes  # 全量
        target_pool = codes[:200] # 测试用，只跑前200个
        
        total = len(target_pool)
        
        for ts_code in target_pool:
            checked_count += 1
            # 打印进度条效果 (每20个打印一次)
            if checked_count % 20 == 0:
                print(f"进度: {checked_count}/{total} ...")
                
            # --- 漏斗筛选法 ---
            
            # 1. 周线不合格，直接 pass (最快)
            if not self.check_weekly_low(ts_code):
                continue
                
            # 2. 日线没信号，直接 pass
            if not self.check_daily_trigger(ts_code):
                continue
                
            # 3. 筹码验证 (最慢，最后做)
            # 打印一下，表示进入决赛圈了
            print(f"正在验证筹码: {ts_code} ...", end="") 
            if self.check_chips_structure(ts_code):
                print("【命中！】")
                candidates.append(ts_code)
            else:
                print(" 筹码结构一般")
                
        print("\n" + "="*50)
        print(f"选股完成！日期：{self.trade_date}")
        print(f"最终入选股票 ({len(candidates)}只):")
        print(candidates)
        print("="*50)
        
        # 简单的保存结果
        if candidates:
            with open(f'result_{self.trade_date}.txt', 'w') as f:
                f.write(','.join(candidates))
            print(f"结果已保存至 result_{self.trade_date}.txt")

# ================= 3. 程序入口 =================

if __name__ == "__main__":
    try:
        # 1. 输入Token
        my_token = get_user_token()
        
        # 2. 初始化策略
        strategy = StrategyRunner(my_token)
        
        # 3. 运行
        strategy.run()
        
    except KeyboardInterrupt:
        print("\n程序已手动中止。")
    except Exception as e:
        print(f"\n程序发生未知错误: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n按回车键退出...")
