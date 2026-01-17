import streamlit as st
import tushare as ts
import pandas as pd
import datetime
from tenacity import retry, stop_after_attempt, wait_fixed

# ================= 1. 页面基础配置 =================
st.set_page_config(page_title="A股全能狙击+回测", page_icon="📈", layout="wide")

st.title("📈 A股周线潜伏+回测验证系统")
st.markdown("### 核心策略：周线低位 + 日线启动 + 筹码集中 + 硬性门槛过滤")

# ================= 2. 侧边栏：参数与配置 =================
with st.sidebar:
    st.header("⚙️ 核心参数")
    
    # 1. Token 输入
    my_token = st.text_input("Tushare Token", type="password", key="token_input", help="请输入您的10000积分Token")
    
    # 2. 日期选择 (支持回测)
    st.divider()
    st.subheader("📅 日期设定")
    # 默认为今天，如果选过去的时间，自动触发回测
    default_date = datetime.date.today()
    selected_date_obj = st.date_input("选择选股/回测日期", default_date)
    selected_date_str = selected_date_obj.strftime('%Y%m%d')

    # 3. 硬性过滤条件
    st.divider()
    st.subheader("🚫 过滤门槛")
    
    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("最低股价(元)", value=10.0)
        min_cap = st.number_input("最小流值(亿)", value=40.0)
    with col2:
        max_price = st.number_input("最高股价(元)", value=300.0)
        max_cap = st.number_input("最大流值(亿)", value=1000.0)

    # 4. 扫描范围
    st.divider()
    scan_limit = st.slider("扫描股票数量", 100, 5500, 500, help="全市场约5300只，手机端建议先用500-1000只测试")
    st.caption("提示：选择过去的日期可查看T+1/3/5收益率")

# ================= 3. 核心功能函数 =================

# 自动获取最近交易日（如果选了周末，自动前推）
def get_real_trade_date(pro, date_str):
    try:
        # 向前找20天确保能覆盖假期
        start_check = (pd.to_datetime(date_str) - pd.Timedelta(days=20)).strftime('%Y%m%d')
        df = pro.trade_cal(exchange='', start_date=start_check, end_date=date_str, is_open='1')
        if df.empty: return date_str
        return df['cal_date'].values[-1]
    except:
        return date_str

# 获取未来N个交易日的日期（用于回测）
def get_future_dates(pro, start_date, days=10):
    try:
        start_dt = pd.to_datetime(start_date)
        # 向后找30天自然日，足够覆盖5个交易日
        end_check = (start_dt + pd.Timedelta(days=30)).strftime('%Y%m%d')
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_check, is_open='1')
        # 返回日期列表，排除start_date本身，取后5个
        future_dates = df[df['cal_date'] > start_date]['cal_date'].tolist()
        return future_dates[:5] # 返回未来5个交易日
    except:
        return []

# 重试机制：获取筹码
@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
def fetch_chips_data(pro, ts_code, trade_date):
    return pro.cyq_perf(ts_code=ts_code, start_date=trade_date, end_date=trade_date)

# 【核心优化】一次性获取并过滤基础池
# @st.cache_data(ttl=3600) # 调试期间先注释缓存，正式使用可打开
def get_filtered_pool(_pro, trade_date, _min_p, _max_p, _min_c, _max_c):
    status_text = st.empty()
    status_text.info("正在进行第一轮大数据清洗（剔除ST、北交所、价格市值不符）...")
    
    # 1. 获取基础列表 (ST, 上市日期, 板块)
    df_basic = _pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,list_date')
    
    # 2. 基础过滤
    # 剔除ST
    df_basic = df_basic[~df_basic['name'].str.contains('ST')]
    df_basic = df_basic[~df_basic['name'].str.contains('退')]
    # 剔除北交所 (market='北交所' 或 代码以 8/4 开头, Tushare中北交所后缀为.BJ)
    df_basic = df_basic[~df_basic['ts_code'].str.endswith('.BJ')]
    # 剔除上市不满半年
    limit_date = pd.to_datetime(trade_date) - pd.Timedelta(days=180)
    df_basic = df_basic[pd.to_datetime(df_basic['list_date']) < limit_date]
    
    # 3. 获取每日指标 (价格、流通市值)
    # circ_mv 单位是万，所以 40亿 = 400000万
    status_text.info(f"正在获取 {trade_date} 的全市场价格与市值数据...")
    df_daily = _pro.daily_basic(trade_date=trade_date, fields='ts_code,close,circ_mv')
    
    if df_daily.empty:
        status_text.error(f"日期 {trade_date} 没有行情数据，可能是休市日。")
        return pd.DataFrame()

    # 4. 合并数据
    df_merge = pd.merge(df_basic, df_daily, on='ts_code', how='inner')
    
    # 5. 数值过滤
    # 市值转换：用户输入是亿，数据是万 -> 1亿 = 10000万
    min_mv_val = _min_c * 10000
    max_mv_val = _max_c * 10000
    
    condition = (
        (df_merge['close'] >= _min_p) & 
        (df_merge['close'] <= _max_p) &
        (df_merge['circ_mv'] >= min_mv_val) & 
        (df_merge['circ_mv'] <= max_mv_val)
    )
    
    final_pool = df_merge[condition]
    status_text.empty() # 清除提示
    return final_pool

# 策略类
class StrategyPro:
    def __init__(self, pro, trade_date):
        self.pro = pro
        self.trade_date = trade_date

    def check_weekly_low(self, ts_code):
        try:
            # 取60周数据
            df = self.pro.weekly(ts_code=ts_code, end_date=self.trade_date, limit=60)
            if df is None or len(df) < 50: return False
            
            last_close = df.iloc[0]['close']
            p_high = df['high'].max()
            p_low = df['low'].min()
            
            if p_high == p_low: return False
            # 相对位置：(当前价-最低)/(最高-最低)
            pos = (last_close - p_low) / (p_high - p_low)
            
            # 放宽一点点周线要求到 40%，防止漏掉刚启动的
            return pos <= 0.40
        except:
            return False

    def check_daily_trigger(self, ts_code):
        try:
            df = self.pro.daily(ts_code=ts_code, end_date=self.trade_date, limit=10)
            if df is None or len(df) < 5: return False
            
            today = df.iloc[0]
            # 涨幅 1.5% - 9.5% (放宽一点下限，有些慢牛是1.5%起步)
            if not (1.5 < today['pct_chg'] < 9.5): return False
            
            # 量比 > 1.1 (放宽一点)
            avg_vol = df.iloc[1:6]['vol'].mean()
            if avg_vol == 0 or today['vol'] < 1.1 * avg_vol: return False
            
            return True
        except:
            return False

    def check_chips(self, ts_code):
        try:
            df = fetch_chips_data(self.pro, ts_code, self.trade_date)
            if df is None or df.empty: return False
            
            winner_rate = df.iloc[0]['winner_rate']
            # 获利盘极少(超跌 <15) 或 突破拉升态势(50-90)
            if winner_rate < 15 or (50 < winner_rate < 90):
                return True
            return False
        except:
            return False

# 回测计算函数
def run_backtest(pro, ts_code, start_date, future_dates):
    """
    计算 T+1, T+3, T+5 的收益率
    """
    res = {'T+1': None, 'T+3': None, 'T+5': None}
    try:
        # 一次性取未来几天的行情
        end_dt = future_dates[-1]
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_dt)
        df = df.sort_values('trade_date') # 按日期正序
        
        # 获取基准日收盘价
        base_row = df[df['trade_date'] == start_date]
        if base_row.empty: return res
        base_price = base_row.iloc[0]['close']
        
        # 寻找对应的交易日
        all_dates = df['trade_date'].tolist()
        
        # 辅助函数：计算收益率
        def calc_ret(target_date):
            if target_date in all_dates:
                curr_price = df[df['trade_date'] == target_date].iloc[0]['close']
                return round((curr_price - base_price) / base_price * 100, 2)
            return None

        if len(future_dates) >= 1: res['T+1'] = calc_ret(future_dates[0])
        if len(future_dates) >= 3: res['T+3'] = calc_ret(future_dates[2])
        if len(future_dates) >= 5: res['T+5'] = calc_ret(future_dates[4])
        
    except Exception as e:
        pass
    return res

# ================= 4. 主程序入口 =================

if st.button("🚀 开始选股/回测", type="primary"):
    if not my_token:
        st.error("请先在侧边栏输入 Tushare Token")
        st.stop()
        
    ts.set_token(my_token)
    try:
        pro = ts.pro_api()
        # 1. 确定实际交易日
        real_date = get_real_trade_date(pro, selected_date_str)
        st.info(f"🔍 正在扫描交易日: **{real_date}**")
        
        # 2. 判断是否需要回测
        future_dates = get_future_dates(pro, real_date)
        is_backtest = len(future_dates) > 0
        if is_backtest:
            st.success(f"检测到历史日期，将自动计算 T+1, T+3, T+5 收益率")
            
        # 3. 获取并清洗基础池
        pool_df = get_filtered_pool(pro, real_date, min_price, max_price, min_cap, max_cap)
        if pool_df.empty:
            st.warning("该日期无数据或所有股票均被基础条件过滤，请调整参数。")
            st.stop()
            
        # 4. 截取扫描范围
        full_codes = pool_df['ts_code'].tolist()
        # 如果池子比设定的少，就全扫
        actual_limit = min(len(full_codes), scan_limit)
        target_pool = full_codes[:actual_limit]
        
        st.write(f"📉 基础过滤后剩余 {len(full_codes)} 只，本次将扫描前 {actual_limit} 只...")
        
        # 5. 循环策略
        strategy = StrategyPro(pro, real_date)
        results = []
        
        progress_bar = st.progress(0)
        status_box = st.status("正在进行量化分析...", expanded=True)
        
        for i, code in enumerate(target_pool):
            progress_bar.progress((i + 1) / actual_limit)
            
            # 策略漏斗
            if not strategy.check_weekly_low(code): continue
            if not strategy.check_daily_trigger(code): continue
            
            status_box.write(f"正在验证筹码: {code} ...")
            if strategy.check_chips(code):
                # 命中目标
                stock_name = pool_df[pool_df['ts_code']==code]['name'].values[0]
                industry = "未知" # 简化处理，如需行业可再调接口
                
                item = {
                    "代码": code,
                    "名称": stock_name,
                    "选入日期": real_date
                }
                
                # 如果是回测模式，计算收益
                if is_backtest:
                    ret_data = run_backtest(pro, code, real_date, future_dates)
                    item['T+1收益%'] = ret_data['T+1']
                    item['T+3收益%'] = ret_data['T+3']
                    item['T+5收益%'] = ret_data['T+5']
                    
                    # 胜率标记
                    win = 0
                    if ret_data['T+1'] and ret_data['T+1'] > 0: win = 1
                    item['首日胜'] = '✅' if win else '❌'
                
                results.append(item)
                st.toast(f"✅ 选中: {stock_name}")

        status_box.update(label="分析完成", state="complete", expanded=False)
        
        # 6. 展示结果
        st.divider()
        if results:
            res_df = pd.DataFrame(results)
            
            # 高亮显示收益率 (Style)
            if is_backtest:
                st.write(f"### 📊 回测结果报告 (共 {len(res_df)} 只)")
                
                # 计算综合胜率
                valid_t1 = res_df['T+1收益%'].dropna()
                if len(valid_t1) > 0:
                    win_rate = (valid_t1 > 0).sum() / len(valid_t1) * 100
                    avg_ret = valid_t1.mean()
                    col_a, col_b = st.columns(2)
                    col_a.metric("T+1 平均胜率", f"{win_rate:.1f}%")
                    col_b.metric("T+1 平均收益", f"{avg_ret:.2f}%")
                
                st.dataframe(res_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            else:
                st.write(f"### 🎯 今日选股结果 (共 {len(res_df)} 只)")
                st.dataframe(res_df, use_container_width=True)
                st.code(','.join([r['代码'] for r in results]))
                
        else:
            st.warning("⚠️ 扫描结束，未发现符合条件的股票。")
            st.caption("建议：1. 扩大扫描数量 2. 放宽最低/最高股价限制 3. 检查Token是否支持筹码数据")

    except Exception as e:
        st.error(f"发生错误: {e}")
