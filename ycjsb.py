import streamlit as st
import tushare as ts
import pandas as pd
import datetime
import time
from tenacity import retry, stop_after_attempt, wait_fixed

# ================= 1. 页面配置 =================
st.set_page_config(page_title="A股狙击系统(公平版)", page_icon="⚖️", layout="wide")

st.title("⚖️ A股狙击系统：解决大市值权重偏差")
st.markdown("### 核心升级：增加“换手率排序”，让大小盘股公平竞技")

# ================= 2. 侧边栏：参数设置 =================
with st.sidebar:
    st.header("⚙️ 核心控制台")
    my_token = st.text_input("Tushare Token", type="password", key="token", help="请输入10000积分Token")
    
    st.divider()
    st.subheader("⚖️ 排序逻辑 (关键修改)")
    # 【新增】排序方式选择
    sort_method = st.radio(
        "优先筛选标准", 
        ["按换手率 (活跃度优先)", "按成交额 (资金流优先)"],
        index=0,
        help="【换手率】适合抓妖股，消除市值差异；【成交额】适合抓大票龙头。"
    )
    
    st.divider()
    st.subheader("🛠️ 策略开关")
    use_chips = st.checkbox("启用筹码数据过滤", value=True)
    
    st.divider()
    st.subheader("🗓️ 模式选择")
    mode = st.radio("运行模式", ["单日扫描", "区间回测"], index=0)
    
    if mode == "单日扫描":
        default_date = datetime.date.today() - datetime.timedelta(days=1)
        selected_date = st.date_input("选择日期", default_date)
        start_date_str = selected_date.strftime('%Y%m%d')
        end_date_str = start_date_str
    else:
        c1, c2 = st.columns(2)
        with c1: d1 = st.date_input("开始", datetime.date.today() - datetime.timedelta(days=20))
        with c2: d2 = st.date_input("结束", datetime.date.today())
        start_date_str = d1.strftime('%Y%m%d')
        end_date_str = d2.strftime('%Y%m%d')

    st.divider()
    st.subheader("🚫 硬性门槛")
    scan_limit = st.slider("扫描前N名", 100, 3000, 300)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        min_p = st.number_input("最低价", value=10.0)
        min_mv = st.number_input("最小市值(亿)", value=40.0)
    with col_p2:
        max_p = st.number_input("最高价", value=300.0)
        max_mv = st.number_input("最大市值(亿)", value=1000.0)

# ================= 3. 核心逻辑 =================

def get_trade_cal(pro, start, end):
    df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
    return df['cal_date'].tolist()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
def fetch_chips_safe(pro, ts_code, trade_date):
    return pro.cyq_perf(ts_code=ts_code, start_date=trade_date, end_date=trade_date)

# 【核心修改】支持动态排序字段
def get_sorted_pool(_pro, trade_date, _min_p, _max_p, _min_mv, _max_mv, _sort_method):
    
    # 1. 基础表
    df_basic = _pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,list_date')
    df_basic = df_basic[~df_basic['name'].str.contains('ST|退')]
    df_basic = df_basic[~df_basic['ts_code'].str.contains('\.BJ')] 
    limit_date = pd.to_datetime(trade_date) - pd.Timedelta(days=180)
    df_basic = df_basic[pd.to_datetime(df_basic['list_date']) < limit_date]
    
    # 2. 行情表
    # 获取成交额(amount)用于资金流排序，获取换手率(turnover_rate)用于活跃度排序
    df_daily = _pro.daily(trade_date=trade_date, fields='ts_code,close,amount')
    df_basic_daily = _pro.daily_basic(trade_date=trade_date, fields='ts_code,circ_mv,turnover_rate')
    
    if df_daily.empty or df_basic_daily.empty: return pd.DataFrame()
    
    # 合并
    df_merge = pd.merge(df_basic, df_daily, on='ts_code')
    df_merge = pd.merge(df_merge, df_basic_daily, on='ts_code')
    
    # 3. 硬性过滤
    cond = (
        (df_merge['close'] >= _min_p) & 
        (df_merge['close'] <= _max_p) &
        (df_merge['circ_mv'] >= _min_mv * 10000) & 
        (df_merge['circ_mv'] <= _max_mv * 10000)
    )
    pool = df_merge[cond]
    
    # 4. 【关键修改】动态排序
    if "换手率" in _sort_method:
        # 按换手率倒序：50亿的小票如果换手高，会排在最前面！
        pool = pool.sort_values('turnover_rate', ascending=False)
    else:
        # 按成交额倒序：大票有天然优势
        pool = pool.sort_values('amount', ascending=False)
    
    return pool

class StrategyRunner:
    def __init__(self, pro, trade_date):
        self.pro = pro
        self.trade_date = trade_date
        self.last_chips_value = None 

    def check_weekly(self, ts_code):
        try:
            df = self.pro.weekly(ts_code=ts_code, end_date=self.trade_date, limit=60)
            if df is None or len(df) < 50: return False
            last = df.iloc[0]['close']
            low = df['low'].min()
            high = df['high'].max()
            if high == low: return False
            pos = (last - low) / (high - low)
            return pos <= 0.45 
        except:
            return False

    def check_daily(self, ts_code):
        try:
            df = self.pro.daily(ts_code=ts_code, end_date=self.trade_date, limit=10)
            if df is None or len(df) < 5: return False
            today = df.iloc[0]
            if not (2.0 < today['pct_chg'] < 9.0): return False
            v_ma = df.iloc[1:6]['vol'].mean()
            if v_ma > 0 and today['vol'] > 1.2 * v_ma: return True
            return False
        except:
            return False

    def check_chips_or_alternative(self, ts_code, use_chips_api):
        self.last_chips_value = "未启用"
        if use_chips_api:
            try:
                df = fetch_chips_safe(self.pro, ts_code, self.trade_date)
                if df is None or df.empty: 
                    self.last_chips_value = "获取失败"
                    return False
                win = df.iloc[0]['winner_rate']
                self.last_chips_value = f"{win:.2f}%" 
                if win < 15 or (50 < win < 90): return True
                return False
            except:
                self.last_chips_value = "接口错误"
                return False
        else:
            try:
                df = self.pro.daily_basic(ts_code=ts_code, trade_date=self.trade_date, fields='turnover_rate')
                if df.empty: return False
                turnover = df.iloc[0]['turnover_rate']
                self.last_chips_value = f"换手{turnover}%(替代)"
                if 3.0 < turnover < 15.0: return True
                return False
            except:
                return False

def calc_returns(pro, ts_code, buy_date):
    res = {'T+1': None, 'T+3': None, 'T+5': None}
    try:
        start_dt = pd.to_datetime(buy_date)
        end_check = (start_dt + pd.Timedelta(days=20)).strftime('%Y%m%d')
        df = pro.daily(ts_code=ts_code, start_date=buy_date, end_date=end_check)
        if df.empty or len(df) < 2: return res
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        base = df.iloc[0]['close']
        
        if len(df) > 1: res['T+1'] = round((df.iloc[1]['close'] - base)/base*100, 2)
        if len(df) > 3: res['T+3'] = round((df.iloc[3]['close'] - base)/base*100, 2)
        if len(df) > 5: res['T+5'] = round((df.iloc[5]['close'] - base)/base*100, 2)
    except:
        pass
    return res

# ================= 4. 主程序 =================

if st.button("🚀 启动策略", type="primary"):
    if not my_token:
        st.error("请先输入Token")
        st.stop()
        
    ts.set_token(my_token)
    pro = ts.pro_api()
    
    trade_dates = get_trade_cal(pro, start_date_str, end_date_str)
    if not trade_dates:
        st.error("日期范围内无交易日")
        st.stop()
        
    st.info(f"📅 扫描区间: {trade_dates[0]} ~ {trade_dates[-1]} ({len(trade_dates)}天)")
    
    all_results = []
    
    main_progress = st.progress(0)
    status_box = st.status("正在初始化...", expanded=True)
    log_area = st.empty() 
    
    for i, t_date in enumerate(trade_dates):
        status_box.write(f"📂 [{i+1}/{len(trade_dates)}] 正在加载 {t_date} 数据...")
        main_progress.progress(i / len(trade_dates))
        
        # 传入新的排序参数
        pool = get_sorted_pool(pro, t_date, min_p, max_p, min_mv, max_mv, sort_method)
        if pool.empty: continue
        
        target_codes = pool['ts_code'].tolist()[:scan_limit]
        status_box.write(f"🔍 {t_date}: 初筛合格 {len(pool)} 只，扫描头部 {len(target_codes)} 只 ({sort_method})...")
        
        runner = StrategyRunner(pro, t_date)
        
        for code in target_codes:
            if not runner.check_weekly(code): continue
            if not runner.check_daily(code): continue
            
            is_match = runner.check_chips_or_alternative(code, use_chips)
            
            stock_name = pool[pool['ts_code']==code]['name'].values[0]
            log_area.text(f"正在验证: {stock_name} ({code}) -> 筹码/指标值: {runner.last_chips_value}")
            
            if is_match:
                ret = calc_returns(pro, code, t_date)
                # 获取该股票的换手率和成交额数据用于展示
                row = pool[pool['ts_code']==code].iloc[0]
                
                item = {
                    "日期": t_date,
                    "代码": code,
                    "名称": stock_name,
                    "价格": row['close'],
                    "市值(亿)": round(row['circ_mv']/10000, 2),
                    "换手率%": row.get('turnover_rate', 0), # 新增展示
                    "成交额(千)": row.get('amount', 0),    # 新增展示
                    "筹码/指标": runner.last_chips_value, 
                    "T+1": ret['T+1'],
                    "T+3": ret['T+3'],
                    "T+5": ret['T+5']
                }
                all_results.append(item)
                st.toast(f"✅ 命中: {stock_name}")
                
    main_progress.progress(100)
    status_box.update(label="扫描完成", state="complete", expanded=False)
    
    if all_results:
        res_df = pd.DataFrame(all_results)
        st.success(f"🎉 扫描结束，共发现 {len(res_df)} 个买点")
        
        win_df = res_df.dropna(subset=['T+1'])
        if not win_df.empty:
            win_rate = len(win_df[win_df['T+1']>0]) / len(win_df) * 100
            st.metric("T+1 胜率", f"{win_rate:.1f}%")
        
        st.dataframe(
            res_df.style.background_gradient(subset=['T+1'], cmap='RdYlGn', vmin=-5, vmax=5),
            column_order=["日期", "名称", "代码", "换手率%", "T+1", "T+3", "T+5", "筹码/指标"],
            use_container_width=True
        )
        
        st.download_button("📥 下载详细CSV", res_df.to_csv(index=False).encode('utf-8-sig'), "report.csv")
    else:
        st.warning("未找到符合条件的股票。")
