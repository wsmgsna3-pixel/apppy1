import streamlit as st
import tushare as ts
import pandas as pd
import datetime
import os
import time
from tenacity import retry, stop_after_attempt, wait_fixed

# ================= 1. 页面配置 =================
st.set_page_config(page_title="周线选股驾驶舱", page_icon="🚀", layout="wide")

st.title("🚀 A股周线选股驾驶舱")
st.markdown("### 策略核心：周线底部(≤0.45) + 日线放量 + 筹码集中 -> 综合打分Top5")

# 文件路径
CACHE_FILE = "scan_checkpoint_dashboard.csv"
PROGRESS_FILE = "scan_progress_dashboard.txt"

# ================= 2. 侧边栏：参数设置 =================
with st.sidebar:
    st.header("⚙️ 核心控制台")
    my_token = st.text_input("Tushare Token", type="password", key="token", help="请输入10000积分Token")
    
    st.divider()
    st.subheader("🗓️ 模式选择")
    mode = st.radio("运行模式", ["单日扫描", "区间回测"], index=0)
    
    if mode == "单日扫描":
        # 默认昨天
        default_date = datetime.date.today() - datetime.timedelta(days=1)
        # 如果今天是周一，默认上周五
        if datetime.date.today().weekday() == 0:
            default_date = datetime.date.today() - datetime.timedelta(days=3)
            
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
    st.subheader("⚖️ 筛选标准")
    # 默认综合打分
    sort_method = st.radio("最终排名依据", ["按综合得分 (推荐)", "按换手率", "按成交额"], index=0)
    
    scan_limit = st.slider("初筛活跃股数量", 200, 3000, 500, help="先按成交额取前N名，再进行精细打分")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        min_p = st.number_input("最低价", value=5.0) # 放宽一点包含部分低价潜力股
        min_mv = st.number_input("最小市值(亿)", value=30.0) # 放宽到30亿包含创业板小票
    with col_p2:
        max_p = st.number_input("最高价", value=300.0)
        max_mv = st.number_input("最大市值(亿)", value=2000.0)

    st.divider()
    if st.button("🗑️ 清除缓存/重新开始"):
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
        if os.path.exists(PROGRESS_FILE): os.remove(PROGRESS_FILE)
        st.toast("已重置任务！")

# ================= 3. 核心工具函数 =================

def get_trade_cal(pro, start, end):
    df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
    return df['cal_date'].tolist()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
def fetch_chips_safe(pro, ts_code, trade_date):
    return pro.cyq_perf(ts_code=ts_code, start_date=trade_date, end_date=trade_date)

def save_result_to_csv(item):
    df = pd.DataFrame([item])
    if not os.path.exists(CACHE_FILE):
        df.to_csv(CACHE_FILE, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(CACHE_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

def save_progress(date_str, code_str):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(f"{date_str},{code_str}")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            content = f.read().strip()
            if content: return content.split(',')
    return None, None

def get_sorted_pool(_pro, trade_date, _min_p, _max_p, _min_mv, _max_mv):
    try:
        # 1. 基础信息 (含科创/创业板)
        df_basic = _pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,list_date,market')
        df_basic = df_basic[~df_basic['name'].str.contains('ST|退')]
        df_basic = df_basic[~df_basic['ts_code'].str.contains('\.BJ')] # 剔除北交所
        limit_date = pd.to_datetime(trade_date) - pd.Timedelta(days=180)
        df_basic = df_basic[pd.to_datetime(df_basic['list_date']) < limit_date]
        
        # 2. 行情数据
        df_daily = _pro.daily(trade_date=trade_date, fields='ts_code,close,amount')
        df_basic_daily = _pro.daily_basic(trade_date=trade_date, fields='ts_code,circ_mv,turnover_rate,volume_ratio')
        
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
        
        # 初筛：按成交额倒序，保证流动性
        pool = pool.sort_values('amount', ascending=False)
        return pool
    except:
        return pd.DataFrame()

class StrategyRunner:
    def __init__(self, pro, trade_date):
        self.pro = pro
        self.trade_date = trade_date
        self.chips_data = 0
        self.vol_ratio = 0

    def check_weekly(self, ts_code):
        try:
            df = self.pro.weekly(ts_code=ts_code, end_date=self.trade_date, limit=60)
            if df is None or len(df) < 50: return False
            last = df.iloc[0]['close']
            low = df['low'].min()
            high = df['high'].max()
            if high == low: return False
            pos = (last - low) / (high - low)
            # 0.45 黄金分割下沿经验值
            return pos <= 0.45 
        except:
            return False

    def check_daily(self, ts_code):
        try:
            df = self.pro.daily(ts_code=ts_code, end_date=self.trade_date, limit=10)
            if df is None or len(df) < 5: return False
            today = df.iloc[0]
            
            # 涨幅宽松一点，包含慢牛
            if not (2.0 < today['pct_chg'] < 10.5): return False
            
            # 计算量比
            v_ma = df.iloc[1:6]['vol'].mean()
            if v_ma > 0:
                self.vol_ratio = round(today['vol'] / v_ma, 2)
            else:
                self.vol_ratio = 0
                
            if self.vol_ratio < 1.2: return False
            return True
        except:
            return False

    def check_chips(self, ts_code):
        try:
            df = fetch_chips_safe(self.pro, ts_code, self.trade_date)
            if df is None or df.empty: return False
            win = df.iloc[0]['winner_rate']
            self.chips_data = win
            
            # 只要获利盘不是极度尴尬的区间(比如30-40可能在洗盘)，两头都行
            # 低位<20是超跌，高位>50是突破
            if win < 20 or win > 45: 
                return True
            return False
        except:
            return False

def calc_returns(pro, ts_code, buy_date):
    res = {'T+1': None, 'T+3': None, 'T+5': None}
    try:
        start_dt = pd.to_datetime(buy_date)
        # 向后取20天自然日足够覆盖
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

last_date, last_code = load_progress()
btn_label = "🚀 启动扫描"
if last_date:
    btn_label = f"🔄 恢复中断任务 ({last_date})"

if st.button(btn_label, type="primary"):
    if not my_token:
        st.error("请先输入Token")
        st.stop()
        
    ts.set_token(my_token)
    pro = ts.pro_api()
    trade_dates = get_trade_cal(pro, start_date_str, end_date_str)
    
    if not trade_dates:
        st.error("该时间段无交易日")
        st.stop()
        
    # --- 界面布局 ---
    # 顶部留白给仪表盘，等跑完了再填
    dashboard_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_box = st.status("系统初始化...", expanded=True)
    log_area = st.empty()

    # --- 恢复进度 ---
    start_date_idx = 0
    if last_date and last_date in trade_dates:
        start_date_idx = trade_dates.index(last_date)
        
    # --- 循环执行 ---
    for i in range(start_date_idx, len(trade_dates)):
        t_date = trade_dates[i]
        status_box.write(f"📅 [{i+1}/{len(trade_dates)}] 正在分析 {t_date} ...")
        progress_bar.progress((i)/len(trade_dates))
        
        # 获取池子
        pool = get_sorted_pool(pro, t_date, min_p, max_p, min_mv, max_mv)
        if pool.empty: continue
        target_codes = pool['ts_code'].tolist()[:scan_limit]
        
        start_code_idx = 0
        if last_code and t_date == last_date:
            if last_code in target_codes:
                start_code_idx = target_codes.index(last_code) + 1

        runner = StrategyRunner(pro, t_date)
        
        for j in range(start_code_idx, len(target_codes)):
            code = target_codes[j]
            save_progress(t_date, code)
            
            if not runner.check_weekly(code): continue
            if not runner.check_daily(code): continue
            if runner.check_chips(code):
                
                # === 打分公式 ===
                # 1. 获利盘: 权重0.4 (最高40分)
                s1 = runner.chips_data * 0.4
                # 2. 量比: 权重20 (最高100分，一般在20-60之间)
                s2 = min(runner.vol_ratio, 5.0) * 20
                # 3. 换手: 权重0.5 (最高10分)
                row = pool[pool['ts_code']==code].iloc[0]
                turn = row.get('turnover_rate', 0)
                s3 = min(turn, 20) * 0.5
                
                total_score = round(s1 + s2 + s3, 1)
                
                ret = calc_returns(pro, code, t_date)
                
                item = {
                    "日期": t_date,
                    "代码": code,
                    "名称": row['name'],
                    "综合得分": total_score,
                    "获利盘%": round(runner.chips_data, 1),
                    "量比": runner.vol_ratio,
                    "换手率%": turn,
                    "T+1": ret['T+1'],
                    "T+3": ret['T+3'],
                    "T+5": ret['T+5']
                }
                save_result_to_csv(item)
                log_area.text(f"命中: {row['name']} | 得分: {total_score} (量比{runner.vol_ratio} 获利{runner.chips_data:.0f}%)")
        
        last_code = None

    progress_bar.progress(100)
    status_box.update(label="分析完成！", state="complete", expanded=False)
    
    # ================= 仪表盘展示逻辑 =================
    if os.path.exists(CACHE_FILE):
        df_all = pd.read_csv(CACHE_FILE)
        
        # 按用户选择排序
        if "打分" in sort_method:
            df_sorted = df_all.sort_values("综合得分", ascending=False)
        elif "换手" in sort_method:
            df_sorted = df_all.sort_values("换手率%", ascending=False)
        else:
            # 默认
            df_sorted = df_all
            
        # 提取 Top 5
        top_5 = df_sorted.head(5)
        
        # 计算仪表盘指标
        t1_avg = top_5['T+1'].mean() if 'T+1' in top_5 else 0
        t3_avg = top_5['T+3'].mean() if 'T+3' in top_5 else 0
        t5_avg = top_5['T+5'].mean() if 'T+5' in top_5 else 0
        
        # 计算胜率 (T+3 > 0)
        win_count = len(top_5[top_5['T+3'] > 0])
        win_rate = win_count / len(top_5) * 100 if len(top_5) > 0 else 0
        
        # 渲染仪表盘
        with dashboard_placeholder.container():
            st.divider()
            st.markdown("## 📊 Top 5 战绩仪表盘")
            
            # 第一行：核心收益指标
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("T+1 平均收益", f"{t1_avg:.2f}%", delta_color="normal")
            k2.metric("T+3 平均收益", f"{t3_avg:.2f}%", delta="关键持仓期")
            k3.metric("T+5 平均收益", f"{t5_avg:.2f}%")
            k4.metric("T+3 胜率", f"{win_rate:.0f}%", help="持仓3天后盈利的概率")
            
            st.info(f"💡 策略建议：当前 Top 5 的平均换手率为 {top_5['换手率%'].mean():.1f}%，量比为 {top_5['量比'].mean():.1f}。")
            
            # 第二行：Top 5 详情卡片
            st.markdown("### 🏆 推荐关注 (Top 5)")
            cols = st.columns(5)
            for idx, row in enumerate(top_5.itertuples()):
                with cols[idx]:
                    st.success(f"No.{idx+1} {row.名称}")
                    st.caption(f"代码: {row.代码}")
                    st.metric("得分", f"{row.综合得分}")
                    st.text(f"获利: {row._5}%")
                    st.text(f"量比: {row.量比}")

            st.divider()
        
        # 底部：完整表格
        st.subheader("📋 完整选股数据")
        st.dataframe(
            df_sorted.style.background_gradient(subset=['综合得分'], cmap='Greens'),
            use_container_width=True
        )
        
        with open(CACHE_FILE, "rb") as f:
            st.download_button("📥 下载详细CSV", f, "final_report.csv")
