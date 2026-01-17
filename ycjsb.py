import streamlit as st
import tushare as ts
import pandas as pd
import time
import datetime
from tenacity import retry, stop_after_attempt, wait_fixed

# ================= 1. 页面基础配置 =================
st.set_page_config(page_title="A股短线狙击", page_icon="📈")

st.title("📈 A股周线潜伏+日线突击策略")
st.caption("专为 10000 积分用户优化的移动端版本")

# ================= 2. 侧边栏：安全配置 =================
with st.sidebar:
    st.header("⚙️ 参数设置")
    # 使用 type="password" 隐藏 Token，安全且不落地
    my_token = st.text_input("请输入 Tushare Token", type="password", key="token_input")
    
    # 为了防止手机端运行时间过长，增加一个测试数量限制
    scan_limit = st.slider("扫描股票数量 (测试用)", 50, 5000, 200, help="全市场约5000只，建议先用200只测试")
    
    st.info("提示：手机端运行请保持屏幕常亮，或使用 Streamlit Cloud 部署。")

# ================= 3. 核心逻辑函数 =================

# 自动获取最近交易日
def get_real_trade_date(pro):
    today = datetime.datetime.now().strftime('%Y%m%d')
    # 向前找20天
    start_check = (datetime.datetime.now() - datetime.timedelta(days=20)).strftime('%Y%m%d')
    try:
        df = pro.trade_cal(exchange='', start_date=start_check, end_date=today, is_open='1')
        return df['cal_date'].values[-1]
    except:
        return today

# 重试机制装饰器
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def fetch_chips_data(pro, ts_code, trade_date):
    return pro.cyq_perf(ts_code=ts_code, start_date=trade_date, end_date=trade_date)

# 缓存基础数据，避免每次点击按钮都重新下载
@st.cache_data(ttl=3600)
def get_basic_pool(_pro, trade_date):
    # 获取基础列表
    df = _pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,list_date')
    # 剔除ST
    df = df[~df['name'].str.contains('ST')]
    df = df[~df['name'].str.contains('退')]
    # 剔除次新股
    limit_date = pd.to_datetime(trade_date) - pd.Timedelta(days=180)
    df = df[pd.to_datetime(df['list_date']) < limit_date]
    return df['ts_code'].tolist()

# 策略逻辑封装
class MobileStrategy:
    def __init__(self, pro, trade_date):
        self.pro = pro
        self.trade_date = trade_date

    def check_weekly_low(self, ts_code):
        try:
            df = self.pro.weekly(ts_code=ts_code, end_date=self.trade_date, limit=60)
            if df is None or len(df) < 50: return False
            
            # 计算位置
            last_close = df.iloc[0]['close']
            p_high = df['high'].max()
            p_low = df['low'].min()
            
            if p_high == p_low: return False
            pos = (last_close - p_low) / (p_high - p_low)
            
            # 只要底部 35%
            return pos <= 0.35
        except:
            return False

    def check_daily_trigger(self, ts_code):
        try:
            df = self.pro.daily(ts_code=ts_code, end_date=self.trade_date, limit=10)
            if df is None or len(df) < 5: return False
            
            today = df.iloc[0]
            # 涨幅 2% - 8%
            if not (2.0 < today['pct_chg'] < 8.0): return False
            
            # 量比 > 1.2
            avg_vol = df.iloc[1:6]['vol'].mean()
            if avg_vol == 0 or today['vol'] < 1.2 * avg_vol: return False
            
            return True
        except:
            return False

    def check_chips(self, ts_code):
        try:
            df = fetch_chips_data(self.pro, ts_code, self.trade_date)
            if df is None or df.empty: return False
            
            winner_rate = df.iloc[0]['winner_rate']
            # 获利盘极少(超跌) 或 筹码密集突破(50-85)
            if winner_rate < 15 or (50 < winner_rate < 85):
                return True
            return False
        except:
            return False

# ================= 4. 主运行区 =================

if st.button("🚀 开始选股", type="primary"):
    if not my_token:
        st.error("请先在左侧侧边栏输入 Tushare Token！")
        st.stop()
    
    # 初始化连接
    status_box = st.status("正在初始化...", expanded=True)
    try:
        ts.set_token(my_token)
        pro = ts.pro_api()
        trade_date = get_real_trade_date(pro)
        status_box.write(f"📅 交易日基准: **{trade_date}**")
        
        # 获取股票池
        status_box.write("正在获取全市场股票池...")
        full_codes = get_basic_pool(pro, trade_date)
        
        # 截取用户设定的数量
        target_pool = full_codes[:scan_limit]
        status_box.write(f"🔍 目标扫描数量: {len(target_pool)} 只")
        
    except Exception as e:
        status_box.update(label="初始化失败", state="error")
        st.error(f"连接失败，请检查Token或网络: {e}")
        st.stop()

    # 开始循环
    strategy = MobileStrategy(pro, trade_date)
    candidates = []
    
    # 进度条
    progress_bar = st.progress(0)
    
    status_box.write("⏳ 正在扫描中，请稍候...")
    
    for i, code in enumerate(target_pool):
        # 更新进度条
        progress = (i + 1) / len(target_pool)
        progress_bar.progress(progress)
        
        # 漏斗筛选
        if not strategy.check_weekly_low(code): continue
        if not strategy.check_daily_trigger(code): continue
        
        # 只有前两步通过，才显示日志并查筹码
        status_box.write(f"正在验证筹码: {code} ...")
        
        if strategy.check_chips(code):
            candidates.append(code)
            st.toast(f"🎉 发现目标: {code}") # 手机弹出提示
    
    status_box.update(label="扫描完成！", state="complete", expanded=False)
    
    # 结果展示
    st.divider()
    if candidates:
        st.success(f"✅ 选股完成！共发现 {len(candidates)} 只标的")
        
        # 获取股票名称方便查看
        if len(candidates) > 0:
            df_res = pro.stock_basic(ts_code=','.join(candidates), fields='ts_code,name,industry')
            st.dataframe(df_res, use_container_width=True)
            
            st.code(','.join(candidates), language="text") # 方便复制
    else:
        st.warning("本次扫描未发现符合条件的股票，建议调整参数或扩大扫描范围。")

else:
    # 初始状态提示
    st.info("👈 请在左侧输入 Token，然后点击上方“开始选股”按钮。")
