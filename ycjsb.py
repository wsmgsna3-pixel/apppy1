import streamlit as st
import tushare as ts
import pandas as pd
import datetime
import os
import time
from tenacity import retry, stop_after_attempt, wait_fixed

# ================= 1. 页面配置 =================
st.set_page_config(page_title="周线选股(防崩溃版)", page_icon="🛡️", layout="wide")

st.title("🛡️ A股周线选股系统 (支持断点续传)")
st.caption("每扫描一只股票都会自动存档，崩溃后重新运行即可接着跑")

# 定义缓存文件路径
CACHE_FILE = "scan_checkpoint.csv"     # 存放已完成的结果
PROGRESS_FILE = "scan_progress.txt"    # 存放进度（当前日期|当前股票代码）

# ================= 2. 侧边栏：参数设置 =================
with st.sidebar:
    st.header("⚙️ 核心控制台")
    my_token = st.text_input("Tushare Token", type="password", key="token", help="请输入10000积分Token")
    
    st.divider()
    st.subheader("⚖️ 排序逻辑")
    sort_method = st.radio("优先筛选标准", ["按换手率 (活跃度优先)", "按成交额 (资金流优先)"], index=0)
    
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

    # 清除缓存按钮
    st.divider()
    if st.button("🗑️ 清除历史缓存/重新开始"):
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
        if os.path.exists(PROGRESS_FILE): os.remove(PROGRESS_FILE)
        st.toast("已清除缓存，下次将重新开始！")

# ================= 3. 核心工具函数 =================

def get_trade_cal(pro, start, end):
    df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
    return df['cal_date'].tolist()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
def fetch_chips_safe(pro, ts_code, trade_date):
    return pro.cyq_perf(ts_code=ts_code, start_date=trade_date, end_date=trade_date)

# 保存单条结果到CSV（追加模式）
def save_result_to_csv(item):
    df = pd.DataFrame([item])
    # 如果文件不存在，写入表头；如果存在，不写表头直接追加
    if not os.path.exists(CACHE_FILE):
        df.to_csv(CACHE_FILE, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(CACHE_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

# 保存进度
def save_progress(date_str, code_str):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(f"{date_str},{code_str}")

# 读取进度
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            content = f.read().strip()
            if content:
                return content.split(',')
    return None, None

def get_sorted_pool(_pro, trade_date, _min_p, _max_p, _min_mv, _max_mv, _sort_method):
    try:
        df_basic = _pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,list_date')
        df_basic = df_basic[~df_basic['name'].str.contains('ST|退')]
        df_basic = df_basic[~df_basic['ts_code'].str.contains('\.BJ')] 
        limit_date = pd.to_datetime(trade_date) - pd.Timedelta(days=180)
        df_basic = df_basic[pd.to_datetime(df_basic['list_date']) < limit_date]
        
        df_daily = _pro.daily(trade_date=trade_date, fields='ts_code,close,amount')
        df_basic_daily = _pro.daily_basic(trade_date=trade_date, fields='ts_code,circ_mv,turnover_rate')
        
        if df_daily.empty or df_basic_daily.empty: return pd.DataFrame()
        
        df_merge = pd.merge(df_basic, df_daily, on='ts_code')
        df_merge = pd.merge(df_merge, df_basic_daily, on='ts_code')
        
        cond = (
            (df_merge['close'] >= _min_p) & 
            (df_merge['close'] <= _max_p) &
            (df_merge['circ_mv'] >= _min_mv * 10000) & 
            (df_merge['circ_mv'] <= _max_mv * 10000)
        )
        pool = df_merge[cond]
        
        if "换手率" in _sort_method:
            pool = pool.sort_values('turnover_rate', ascending=False)
        else:
            pool = pool.sort_values('amount', ascending=False)
        
        return pool
    except Exception as e:
        return pd.DataFrame()

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

# 检查是否有上次的进度
last_date, last_code = load_progress()
start_msg = "🚀 启动策略"
if last_date:
    start_msg = f"🔄 检测到异常退出 ({last_date})，点击继续"

if st.button(start_msg, type="primary"):
    if not my_token:
        st.error("请先输入Token")
        st.stop()
        
    ts.set_token(my_token)
    pro = ts.pro_api()
    
    trade_dates = get_trade_cal(pro, start_date_str, end_date_str)
    if not trade_dates:
        st.error("日期范围内无交易日")
        st.stop()
    
    # 显示结果容器
    result_container = st.container()
    
    # 如果有缓存文件，先读取展示
    if os.path.exists(CACHE_FILE):
        try:
            existing_df = pd.read_csv(CACHE_FILE)
            with result_container:
                st.success(f"📂 已加载历史缓存数据：{len(existing_df)} 条")
                st.dataframe(existing_df, height=300)
        except:
            pass

    status_box = st.status("正在初始化...", expanded=True)
    log_area = st.empty()
    
    # --- 寻找断点位置 ---
    start_date_idx = 0
    if last_date and last_date in trade_dates:
        start_date_idx = trade_dates.index(last_date)
        status_box.write(f"🔄 恢复进度：跳过 {last_date} 之前的所有日期...")

    # --- 日期循环 ---
    for i in range(start_date_idx, len(trade_dates)):
        t_date = trade_dates[i]
        
        status_box.write(f"📂 [{i+1}/{len(trade_dates)}] 正在加载 {t_date} 数据...")
        
        pool = get_sorted_pool(pro, t_date, min_p, max_p, min_mv, max_mv, sort_method)
        if pool.empty: continue
        
        target_codes = pool['ts_code'].tolist()[:scan_limit]
        
        # --- 寻找当天内的断点 ---
        start_code_idx = 0
        if last_code and t_date == last_date:
            if last_code in target_codes:
                start_code_idx = target_codes.index(last_code) + 1 # 从下一只开始
                status_box.write(f"🔄 {t_date}: 跳过已完成的 {start_code_idx} 只股票，继续扫描...")
        
        # --- 股票循环 ---
        runner = StrategyRunner(pro, t_date)
        
        for j in range(start_code_idx, len(target_codes)):
            code = target_codes[j]
            
            # 【关键】每扫描一只，就更新一下进度文件
            save_progress(t_date, code)
            
            # 策略检查
            if not runner.check_weekly(code): continue
            if not runner.check_daily(code): continue
            
            is_match = runner.check_chips_or_alternative(code, use_chips)
            
            stock_name = pool[pool['ts_code']==code]['name'].values[0]
            log_area.text(f"[{j+1}/{len(target_codes)}] 正在验证: {stock_name} ({code}) -> {runner.last_chips_value}")
            
            if is_match:
                ret = calc_returns(pro, code, t_date)
                row = pool[pool['ts_code']==code].iloc[0]
                
                item = {
                    "日期": t_date,
                    "代码": code,
                    "名称": stock_name,
                    "价格": row['close'],
                    "市值(亿)": round(row['circ_mv']/10000, 2),
                    "换手率%": row.get('turnover_rate', 0),
                    "成交额(千)": row.get('amount', 0),
                    "筹码/指标": runner.last_chips_value, 
                    "T+1": ret['T+1'],
                    "T+3": ret['T+3'],
                    "T+5": ret['T+5']
                }
                
                # 【关键】发现一只，存一只！
                save_result_to_csv(item)
                st.toast(f"✅ 命中: {stock_name}")
        
        # 当天跑完，重置code进度，防止影响下一天
        last_code = None 

    status_box.update(label="全部扫描完成！", state="complete", expanded=False)
    
    # 最终结果展示
    if os.path.exists(CACHE_FILE):
        final_df = pd.read_csv(CACHE_FILE)
        st.success(f"🎉 任务结束！累计发现 {len(final_df)} 个买点")
        st.dataframe(final_df.style.background_gradient(subset=['T+1'], cmap='RdYlGn', vmin=-5, vmax=5))
        
        # 清除进度文件，因为已经全部跑完了
        if os.path.exists(PROGRESS_FILE): os.remove(PROGRESS_FILE)
        
        with open(CACHE_FILE, "rb") as f:
            st.download_button("📥 下载最终CSV", f, "final_report.csv")
