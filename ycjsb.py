import streamlit as st
import tushare as ts
import pandas as pd
import datetime
import os
import time
from tenacity import retry, stop_after_attempt, wait_fixed

# ================= 1. 页面配置 =================
st.set_page_config(page_title="暴力突破(大道至简)", page_icon="🔥", layout="wide")

st.title("🔥 A股潜伏底·暴力突破系统")
st.markdown("### 策略内核：Top500活跃股 + 涨幅优先 + 潜伏底结构")

# 文件路径
CACHE_FILE = "scan_result_impulse.csv"
HISTORY_FILE = "scan_history_impulse.txt"

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
    st.caption("✅ 逻辑回归：不搞复杂打分，谁涨得猛买谁")
    
    scan_limit = st.slider("初筛活跃股数量", 200, 2000, 500, step=50, help="默认500，核心资产池")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        min_p = st.number_input("最低价(元)", value=20.0)
        min_mv = st.number_input("最小市值(亿)", value=30.0)
    with col_p2:
        max_p = st.number_input("最高价(元)", value=300.0)
        max_mv = st.number_input("最大市值(亿)", value=1000.0)

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

# ================= 5. 极简筛选逻辑 =================

def filter_impulse_batch(df_daily_all, trade_date):
    results = {}
    if df_daily_all.empty: return {}
    
    grouped = df_daily_all.groupby('ts_code')
    
    for code, group in grouped:
        group = group.sort_values('trade_date')
        if group.iloc[-1]['trade_date'] != trade_date: continue
        
        if len(group) < 60: continue
        
        recent_60 = group.tail(60)
        today = recent_60.iloc[-1]
        past_59 = recent_60.iloc[:-1]
        if past_59.empty: continue
        
        # 1. 趋势：站上60日线
        ma60 = recent_60['close'].mean()
        if today['close'] < ma60: continue
        
        # 2. 突破：创60日收盘新高
        max_past_close = past_59['close'].max()
        if today['close'] <= max_past_close: continue
        
        # 3. 涨幅：必须大阳线 (>4.5%)
        # 既然是暴力突破，力度必须大
        if today['pct_chg'] < 4.5: continue
        
        # 4. 形态：光头阳线 (拒绝上影线)
        high = today['high']
        low = today['low']
        close = today['close']
        if high != low:
            pos = (close - low) / (high - low)
            if pos < 0.8: continue 
        
        # 5. 量能：只要倍量就行 (>2.0)
        # 不再惩罚高量比，因为有时候妖股就是天量
        v_ma5 = past_59['vol'].tail(5).mean()
        if v_ma5 == 0: continue
        vol_ratio = today['vol'] / v_ma5
        
        if vol_ratio < 2.0: continue
        
        # 潜伏期涨幅限制
        start_price = past_59.iloc[0]['close']
        end_price = past_59.iloc[-1]['close']
        period_chg = (end_price - start_price) / start_price
        if not (0 < period_chg < 0.45): continue # 稍微放宽到45%
        
        results[code] = {
            'vol_ratio': round(vol_ratio, 2), 
            'pct_chg': today['pct_chg'],
            'period_chg': round(period_chg * 100, 1),
            'close': close
        }
            
    return results

def get_sorted_pool(_pro, trade_date, _min_p, _max_p, _min_mv, _max_mv):
    try:
        df_basic = _pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,list_date,market')
        df_basic = df_basic[~df_basic['name'].str.contains('ST|退')]
        df_basic = df_basic[~df_basic['ts_code'].str.contains('\.BJ')]
        limit_date = pd.to_datetime(trade_date) - pd.Timedelta(days=180)
        df_basic = df_basic[pd.to_datetime(df_basic['list_date']) < limit_date]
        
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

if st.button("🚀 启动暴力扫描", type="primary"):
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
    status_box = st.status("正在执行暴力扫描...", expanded=True)
    log_area = st.empty()

    for i, t_date in enumerate(trade_dates):
        if is_date_scanned(t_date):
            status_box.write(f"⚡️ {t_date} 已跳过...")
            progress_bar.progress((i+1)/len(trade_dates))
            continue
            
        status_box.write(f"📆 [{i+1}/{len(trade_dates)}] 扫描 {t_date} ...")
        progress_bar.progress((i)/len(trade_dates))
        
        # 1. 基础池
        pool = get_sorted_pool(pro, t_date, min_p, max_p, min_mv, max_mv)
        if pool.empty: 
            mark_date_as_scanned(t_date)
            continue
            
        target_codes = pool['ts_code'].tolist()[:scan_limit]
        
        # 2. 批量日线
        df_daily_all = batch_get_daily(pro, target_codes, t_date)
        
        # 3. 核心筛选
        valid_map = filter_impulse_batch(df_daily_all, t_date)
        survivors = list(valid_map.keys())
        daily_candidates = []
        
        for code in survivors:
            # 查筹码
            df_chips = fetch_chips_safe(pro, code, t_date)
            win_rate = 0
            if df_chips is not None and not df_chips.empty:
                win_rate = df_chips.iloc[0]['winner_rate']
            
            # 获利盘门槛 > 70%
            if win_rate > 70:
                vol_ratio = valid_map[code]['vol_ratio']
                pct_chg = valid_map[code]['pct_chg']
                
                # === 核心修改：只看涨幅 (力度) ===
                # 谁涨得猛，谁就是主力意图最坚决的
                total_score = pct_chg 
                
                row = pool[pool['ts_code']==code].iloc[0]
                
                daily_candidates.append({
                    "日期": t_date,
                    "代码": code,
                    "名称": row['name'],
                    "综合得分": total_score, # 这里其实就是涨幅
                    "量比": vol_ratio,
                    "获利盘%": round(win_rate, 1),
                    "ts_code": code
                })
        
        # 5. Top 1 (按涨幅排序)
        if daily_candidates:
            daily_candidates.sort(key=lambda x: x["综合得分"], reverse=True)
            top_1_today = daily_candidates[:1]
            
            for item in top_1_today:
                ret = calc_returns(pro, item['ts_code'], t_date)
                item['T+1'] = ret['T+1']
                item['T+3'] = ret['T+3']
                item['T+5'] = ret['T+5']
                del item['ts_code']
                
                log_area.text(f"🔥 {t_date} 暴力王: {item['名称']} (涨幅{item['综合得分']}% 量比{item['量比']})")
            
            save_result_to_csv(top_1_today)
        
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
                    st.warning(f"{start_date_str} 未发现。")
                    st.stop()
            
            df_all = df_all.sort_values("日期", ascending=False)
            
            def get_metrics(df, col):
                valid_df = df.dropna(subset=[col])
                if valid_df.empty: return 0.0, 0.0
                avg = valid_df[col].mean()
                win = (len(valid_df[valid_df[col] > 0]) / len(valid_df) * 100)
                return avg, win

            t1_avg, t1_win = get_metrics(df_all, 'T+1')
            t3_avg, t3_win = get_metrics(df_all, 'T+3')
            t5_avg, t5_win = get_metrics(df_all, 'T+5')
            
            with dashboard_placeholder.container():
                st.divider()
                st.markdown(f"## 🔥 暴力战报 (Top500 + 涨幅优先)")
                st.caption("策略：移除D1止损，移除打分公式，纯粹看涨幅力度")
                
                k1, k2, k3 = st.columns(3)
                k1.metric("T+1 平均收益", f"{t1_avg:.2f}%", f"胜率 {t1_win:.1f}%")
                k2.metric("T+3 平均收益", f"{t3_avg:.2f}%", f"胜率 {t3_win:.1f}%", delta_color="normal")
                k3.metric("T+5 平均收益", f"{t5_avg:.2f}%", f"胜率 {t5_win:.1f}%")
                
                st.markdown("### 🏆 每日暴力王")
                st.dataframe(df_all, use_container_width=True)
                
                with open(CACHE_FILE, "rb") as f:
                    st.download_button("📥 下载战报", f, "impulse_result.csv")
        except Exception as e:
            st.error(f"读取结果出错: {e}")
