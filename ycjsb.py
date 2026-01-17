import streamlit as st
import tushare as ts
import pandas as pd
import datetime
import os
import time
from tenacity import retry, stop_after_attempt, wait_fixed

# ================= 1. 页面配置 =================
st.set_page_config(page_title="潜伏底突破策略", page_icon="🐲", layout="wide")

st.title("🐲 A股潜伏底突破系统 (Stage 2 Breakout)")
st.markdown("### 核心策略：60日潜伏平台 + 倍量创机高 + 筹码锁定 -> 只选 Top 5")

# 文件路径
CACHE_FILE = "scan_result_stage2.csv"
HISTORY_FILE = "scan_history_stage2.txt"

# ================= 2. 主界面：Token输入 =================
st.info("👇 请在下方输入您的 Tushare Token")
my_token = st.text_input("Tushare Token", type="password", key="token_main", placeholder="在此粘贴 Token...")

# ================= 3. 侧边栏：参数设置 =================
with st.sidebar:
    st.header("⚙️ 参数控制台")
    
    st.divider()
    st.subheader("🗓️ 模式选择")
    mode = st.radio("运行模式", ["单日扫描", "区间回测"], index=0)
    
    if mode == "单日扫描":
        default_date = datetime.date.today() - datetime.timedelta(days=1)
        if datetime.date.today().weekday() == 0:
            default_date = datetime.date.today() - datetime.timedelta(days=3)
        selected_date = st.date_input("选择日期", default_date)
        start_date_str = selected_date.strftime('%Y%m%d')
        end_date_str = start_date_str
    else:
        default_start = datetime.date(2025, 9, 1)
        c1, c2 = st.columns(2)
        with c1: d1 = st.date_input("开始", default_start)
        with c2: d2 = st.date_input("结束", datetime.date.today())
        start_date_str = d1.strftime('%Y%m%d')
        end_date_str = d2.strftime('%Y%m%d')

    st.divider()
    st.subheader("🎯 筛选标准")
    st.write("已锁定策略：潜伏底突破")
    
    # 增加扫描样本，因为我们要过滤掉绝大多数假突破
    scan_limit = st.slider("初筛活跃股数量", 200, 5000, 1000, step=50, help="为了找到真正突破的票，初筛范围建议大一点")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        min_p = st.number_input("最低价(元)", value=8.0)
        min_mv = st.number_input("最小市值(亿)", value=30.0)
    with col_p2:
        max_p = st.number_input("最高价(元)", value=300.0)
        max_mv = st.number_input("最大市值(亿)", value=2000.0)

    st.divider()
    if st.button("🗑️ 彻底清除所有缓存"):
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
        if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
        st.toast("缓存已清空，一切重新开始！")

# ================= 4. 核心工具函数 =================

def get_trade_cal(pro, start, end):
    df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
    return df['cal_date'].tolist()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
def fetch_chips_safe(pro, ts_code, trade_date):
    return pro.cyq_perf(ts_code=ts_code, start_date=trade_date, end_date=trade_date)

def save_result_to_csv(item_list):
    # 批量保存 Top 5
    if not item_list: return
    df = pd.DataFrame(item_list)
    if not os.path.exists(CACHE_FILE):
        df.to_csv(CACHE_FILE, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(CACHE_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

def mark_date_as_scanned(date_str):
    with open(HISTORY_FILE, 'a') as f:
        f.write(date_str + "\n")

def is_date_scanned(date_str):
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE)
            if str(date_str) in df['日期'].astype(str).values:
                return True
        except:
            pass
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            scanned_dates = f.read().splitlines()
            if str(date_str) in scanned_dates:
                return True
    return False

# --- 批量获取 ---
def batch_get_daily(pro, codes, trade_date):
    try:
        chunk_size = 50
        all_df = []
        # 获取过去90天数据，用于计算60日均线和平台新高
        start_dt = pd.to_datetime(trade_date) - pd.Timedelta(days=130)
        start_date_fmt = start_dt.strftime('%Y%m%d')
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i:i+chunk_size]
            codes_str = ",".join(chunk)
            df = pro.daily(ts_code=codes_str, start_date=start_date_fmt, end_date=trade_date)
            if not df.empty: all_df.append(df)
            time.sleep(0.1)
        if not all_df: return pd.DataFrame()
        return pd.concat(all_df)
    except:
        return pd.DataFrame()

# ================= 5. 潜伏底突破逻辑 (Stage 2) =================

def filter_stage2_batch(df_daily_all, trade_date):
    """
    逻辑严苛：
    1. 潜伏期：过去60天涨幅 < 40% (拒绝已经大涨的)
    2. 趋势好：Close > MA60
    3. 真突破：今日收盘价创60日新高 + 量比 > 2.5 + 涨幅 > 4%
    """
    results = {}
    if df_daily_all.empty: return {}
    
    grouped = df_daily_all.groupby('ts_code')
    
    for code, group in grouped:
        group = group.sort_values('trade_date')
        if group.iloc[-1]['trade_date'] != trade_date: continue
        
        # 数据长度要求
        if len(group) < 60: continue
        
        # 取最近60个交易日的数据
        recent_60 = group.tail(60)
        today = recent_60.iloc[-1]
        
        # 1. 潜伏期验证 (Platform Check)
        # 计算过去60天的区间涨幅 (不含今天)
        past_59 = recent_60.iloc[:-1]
        if past_59.empty: continue
        
        start_price = past_59.iloc[0]['close']
        end_price = past_59.iloc[-1]['close']
        
        period_chg = (end_price - start_price) / start_price
        
        # 潜伏期涨幅必须在 0% 到 40% 之间
        # <0 说明趋势向下(熊市)，>40 说明已经涨太高了
        if not (0 < period_chg < 0.40): continue
        
        # 2. 趋势验证 (Trend Check)
        ma60 = recent_60['close'].mean()
        if today['close'] < ma60: continue
        
        # 3. 突破验证 (Breakout Check)
        # 3.1 创新高：今天的收盘价 > 过去59天的最高收盘价
        # 这就过滤掉了"孤独的高柱子"(如果没有创新高)
        max_past_close = past_59['close'].max()
        if today['close'] <= max_past_close: continue
        
        # 3.2 爆发力：量比 > 2.5, 涨幅 > 4.0%
        if not (4.0 < today['pct_chg'] < 10.5): continue
        
        v_ma5 = past_59['vol'].tail(5).mean()
        if v_ma5 == 0: continue
        vol_ratio = today['vol'] / v_ma5
        
        if vol_ratio < 2.5: continue
        
        results[code] = {
            'vol_ratio': round(vol_ratio, 2), 
            'pct_chg': today['pct_chg'],
            'period_chg': round(period_chg * 100, 1), # 潜伏期涨幅
            'close': today['close']
        }
            
    return results

def get_sorted_pool(_pro, trade_date, _min_p, _max_p, _min_mv, _max_mv):
    try:
        # 基础过滤
        df_basic = _pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,list_date,market')
        df_basic = df_basic[~df_basic['name'].str.contains('ST|退')]
        df_basic = df_basic[~df_basic['ts_code'].str.contains('\.BJ')]
        limit_date = pd.to_datetime(trade_date) - pd.Timedelta(days=180)
        df_basic = df_basic[pd.to_datetime(df_basic['list_date']) < limit_date]
        
        # 成交额初筛
        df_daily = _pro.daily(trade_date=trade_date, fields='ts_code,amount')
        df_basic_daily = _pro.daily_basic(trade_date=trade_date, fields='ts_code,circ_mv')
        
        if df_daily.empty or df_basic_daily.empty: return pd.DataFrame()
        
        df_merge = pd.merge(df_basic, df_daily, on='ts_code')
        df_merge = pd.merge(df_merge, df_basic_daily, on='ts_code')
        
        cond = (
            (df_merge['circ_mv'] >= _min_mv * 10000) & 
            (df_merge['circ_mv'] <= _max_mv * 10000)
        )
        pool = df_merge[cond]
        pool = pool.sort_values('amount', ascending=False)
        return pool
    except:
        return pd.DataFrame()

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

# ================= 6. 主程序 =================

if st.button("🚀 启动潜伏底扫描", type="primary"):
    if not my_token:
        st.error("🚨 请输入 Token！")
        st.stop()
        
    ts.set_token(my_token)
    pro = ts.pro_api()
    trade_dates = get_trade_cal(pro, start_date_str, end_date_str)
    
    if not trade_dates:
        st.error("❌ 无交易日")
        st.stop()
        
    dashboard_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_box = st.status("正在雷达扫描...", expanded=True)
    log_area = st.empty()

    for i, t_date in enumerate(trade_dates):
        if is_date_scanned(t_date):
            status_box.write(f"⚡️ {t_date} 已跳过...")
            progress_bar.progress((i+1)/len(trade_dates))
            continue
            
        status_box.write(f"📆 [{i+1}/{len(trade_dates)}] 扫描 {t_date} (潜伏底+创新高) ...")
        progress_bar.progress((i)/len(trade_dates))
        
        # 1. 基础池
        pool = get_sorted_pool(pro, t_date, min_p, max_p, min_mv, max_mv)
        if pool.empty: 
            mark_date_as_scanned(t_date)
            continue
            
        target_codes = pool['ts_code'].tolist()[:scan_limit]
        
        # 2. 批量日线 (判断潜伏期和突破)
        df_daily_all = batch_get_daily(pro, target_codes, t_date)
        
        # 3. 核心筛选：Stage 2 Breakout
        valid_map = filter_stage2_batch(df_daily_all, t_date)
        
        survivors = list(valid_map.keys())
        
        # 4. 查筹码 & 打分 & 排序 (只取Top 5)
        daily_candidates = []
        
        for code in survivors:
            # 查筹码
            df_chips = fetch_chips_safe(pro, code, t_date)
            win_rate = 0
            
            if df_chips is not None and not df_chips.empty:
                win_rate = df_chips.iloc[0]['winner_rate']
            
            # 【过滤】获利盘必须 > 50% (说明套牢盘少，是真突破)
            if win_rate > 50:
                vol_ratio = valid_map[code]['vol_ratio']
                period_chg = valid_map[code]['period_chg']
                
                # 打分公式
                # 量比越大越好(爆发)，获利盘越高越好(稳)，潜伏期涨幅适中就好
                s1 = min(vol_ratio, 6.0) * 10
                s2 = min(win_rate, 100) * 0.4
                total_score = round(s1 + s2, 1)
                
                # 获取名称
                row = pool[pool['ts_code']==code].iloc[0]
                
                daily_candidates.append({
                    "日期": t_date,
                    "代码": code,
                    "名称": row['name'],
                    "综合得分": total_score,
                    "量比": vol_ratio,
                    "获利盘%": round(win_rate, 1),
                    "潜伏涨幅%": period_chg,
                    "ts_code": code # 临时用
                })
        
        # 5. 每日结算：只留 Top 5
        if daily_candidates:
            # 按得分降序
            daily_candidates.sort(key=lambda x: x["综合得分"], reverse=True)
            top_5_today = daily_candidates[:5]
            
            # 补充计算收益率 (只给这5个算，省流)
            for item in top_5_today:
                ret = calc_returns(pro, item['ts_code'], t_date)
                item['T+1'] = ret['T+1']
                item['T+3'] = ret['T+3']
                item['T+5'] = ret['T+5']
                del item['ts_code'] # 删掉临时字段
                
                log_area.text(f"🐲 {t_date} 擒龙: {item['名称']} (量比{item['量比']} 创60日新高)")
            
            # 写入 CSV
            save_result_to_csv(top_5_today)
        
        mark_date_as_scanned(t_date)

    progress_bar.progress(100)
    status_box.update(label="扫描完成！", state="complete", expanded=False)
    
    # ================= 7. 仪表盘 =================
    if os.path.exists(CACHE_FILE):
        try:
            df_all = pd.read_csv(CACHE_FILE)
            
            if mode == "单日扫描":
                df_all = df_all[df_all['日期'].astype(str) == start_date_str]
                if df_all.empty:
                    st.warning(f"{start_date_str} 未发现突破个股。")
                    st.stop()
            
            # 默认按得分展示
            df_sorted = df_all.sort_values("综合得分", ascending=False)
            
            # 全周期战报
            def get_metrics(df, col):
                valid_df = df.dropna(subset=[col])
                if valid_df.empty: return 0.0, 0.0
                avg = valid_df[col].mean()
                win = (len(valid_df[valid_df[col] > 0]) / len(valid_df) * 100)
                return avg, win

            t1_avg, t1_win = get_metrics(df_sorted, 'T+1')
            t3_avg, t3_win = get_metrics(df_sorted, 'T+3')
            t5_avg, t5_win = get_metrics(df_sorted, 'T+5')
            
            with dashboard_placeholder.container():
                st.divider()
                st.markdown(f"## 📈 潜伏底战报 (日期: {start_date_str})")
                
                k1, k2, k3 = st.columns(3)
                k1.metric("T+1 平均收益", f"{t1_avg:.2f}%", f"胜率 {t1_win:.1f}%")
                k2.metric("T+3 平均收益", f"{t3_avg:.2f}%", f"胜率 {t3_win:.1f}%", delta_color="normal")
                k3.metric("T+5 平均收益", f"{t5_avg:.2f}%", f"胜率 {t5_win:.1f}%")
                
                st.markdown("### 🏆 精英 Top 5 (每日仅选5只)")
                st.dataframe(df_sorted, use_container_width=True)
                
                with open(CACHE_FILE, "rb") as f:
                    st.download_button("📥 下载精华CSV", f, "breakout_result.csv")
        except Exception as e:
            st.error(f"读取结果出错: {e}")
