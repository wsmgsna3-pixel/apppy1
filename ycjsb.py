import streamlit as st
import tushare as ts
import pandas as pd
import datetime
import os
import time
from tenacity import retry, stop_after_attempt, wait_fixed

# ================= 1. 页面配置 =================
st.set_page_config(page_title="周线选股Turbo(稳健版)", page_icon="⚡️", layout="wide")

st.title("⚡️ A股周线选股 Turbo：智能缓存版")
st.markdown("### 核心升级：修复绘图库缺失问题，确保手机端稳定运行")

# 文件路径
CACHE_FILE = "scan_result_turbo.csv"     # 存结果
HISTORY_FILE = "scan_history_turbo.txt"  # 存已扫描过的所有日期

# ================= 2. 侧边栏：参数设置 =================
with st.sidebar:
    st.header("⚙️ 核心控制台")
    my_token = st.text_input("Tushare Token", type="password", key="token", help="请输入10000积分Token")
    
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
    st.subheader("⚖️ 筛选标准")
    sort_method = st.radio("排名依据", ["按综合得分 (推荐)", "按换手率", "按成交额"], index=0)
    scan_limit = st.slider("初筛活跃股数量", 200, 5000, 500)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        min_p = st.number_input("最低价", value=5.0)
        min_mv = st.number_input("最小市值(亿)", value=30.0)
    with col_p2:
        max_p = st.number_input("最高价", value=300.0)
        max_mv = st.number_input("最大市值(亿)", value=2000.0)

    st.divider()
    if st.button("🗑️ 彻底清除所有缓存"):
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
        if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
        st.toast("缓存已清空，一切重新开始！")

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
def batch_get_weekly(pro, codes, trade_date):
    try:
        chunk_size = 50
        all_df = []
        start_dt = pd.to_datetime(trade_date) - pd.Timedelta(days=500)
        start_date_fmt = start_dt.strftime('%Y%m%d')
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i:i+chunk_size]
            codes_str = ",".join(chunk)
            df = pro.weekly(ts_code=codes_str, start_date=start_date_fmt, end_date=trade_date)
            if not df.empty: all_df.append(df)
            time.sleep(0.1)
        if not all_df: return pd.DataFrame()
        return pd.concat(all_df)
    except:
        return pd.DataFrame()

def batch_get_daily(pro, codes, trade_date):
    try:
        chunk_size = 50
        all_df = []
        start_dt = pd.to_datetime(trade_date) - pd.Timedelta(days=30)
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

# ================= 4. 内存筛选 =================

def filter_weekly_batch(df_weekly_all, trade_date):
    valid_codes = []
    if df_weekly_all.empty: return []
    grouped = df_weekly_all.groupby('ts_code')
    for code, group in grouped:
        if len(group) < 50: continue
        group = group.sort_values('trade_date')
        group = group.tail(60)
        last = group.iloc[-1]['close']
        low = group['low'].min()
        high = group['high'].max()
        if high == low: continue
        pos = (last - low) / (high - low)
        if pos <= 0.45: valid_codes.append(code)
    return valid_codes

def filter_daily_batch(df_daily_all, valid_weekly_codes, trade_date):
    results = {}
    if df_daily_all.empty: return {}
    df_filtered = df_daily_all[df_daily_all['ts_code'].isin(valid_weekly_codes)]
    grouped = df_filtered.groupby('ts_code')
    for code, group in grouped:
        group = group.sort_values('trade_date')
        if group.iloc[-1]['trade_date'] != trade_date: continue
        if len(group) < 6: continue
        today = group.iloc[-1]
        if not (2.0 < today['pct_chg'] < 10.5): continue
        v_ma = group.iloc[-6:-1]['vol'].mean()
        vol_ratio = 0
        if v_ma > 0: vol_ratio = round(today['vol'] / v_ma, 2)
        if vol_ratio >= 1.2:
            results[code] = {'vol_ratio': vol_ratio, 'pct_chg': today['pct_chg']}
    return results

def get_sorted_pool(_pro, trade_date, _min_p, _max_p, _min_mv, _max_mv):
    try:
        df_basic = _pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,list_date,market')
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

# ================= 5. 主程序 =================

if st.button("🚀 启动/继续", type="primary"):
    if not my_token:
        st.error("请先输入Token")
        st.stop()
        
    ts.set_token(my_token)
    pro = ts.pro_api()
    trade_dates = get_trade_cal(pro, start_date_str, end_date_str)
    
    if not trade_dates:
        st.error("该时间段无交易日")
        st.stop()
        
    dashboard_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_box = st.status("正在启动...", expanded=True)
    log_area = st.empty()

    for i, t_date in enumerate(trade_dates):
        # 智能跳过逻辑
        if is_date_scanned(t_date):
            status_box.write(f"⚡️ {t_date} 已在缓存中，自动跳过...")
            progress_bar.progress((i+1)/len(trade_dates))
            continue
            
        status_box.write(f"📆 [{i+1}/{len(trade_dates)}] 正在扫描 {t_date} (批量模式) ...")
        progress_bar.progress((i)/len(trade_dates))
        
        pool = get_sorted_pool(pro, t_date, min_p, max_p, min_mv, max_mv)
        if pool.empty: 
            mark_date_as_scanned(t_date)
            continue
            
        target_codes = pool['ts_code'].tolist()[:scan_limit]
        
        df_weekly_all = batch_get_weekly(pro, target_codes, t_date)
        valid_weekly_codes = filter_weekly_batch(df_weekly_all, t_date)
        
        if not valid_weekly_codes:
            mark_date_as_scanned(t_date)
            continue
            
        df_daily_all = batch_get_daily(pro, valid_weekly_codes, t_date)
        valid_daily_map = filter_daily_batch(df_daily_all, valid_weekly_codes, t_date)
        
        final_survivors = list(valid_daily_map.keys())
        
        for code in final_survivors:
            df_chips = fetch_chips_safe(pro, code, t_date)
            win_rate = 0
            pass_chips = False
            
            if df_chips is not None and not df_chips.empty:
                win_rate = df_chips.iloc[0]['winner_rate']
                if win_rate < 20 or win_rate > 45:
                    pass_chips = True
            
            if pass_chips:
                vol_ratio = valid_daily_map[code]['vol_ratio']
                row = pool[pool['ts_code']==code].iloc[0]
                turn = row.get('turnover_rate', 0)
                
                s1 = win_rate * 0.4
                s2 = min(vol_ratio, 5.0) * 20
                s3 = min(turn, 20) * 0.5
                total_score = round(s1 + s2 + s3, 1)
                
                ret = calc_returns(pro, code, t_date)
                
                item = {
                    "日期": t_date,
                    "代码": code,
                    "名称": row['name'],
                    "综合得分": total_score,
                    "获利盘%": round(win_rate, 1),
                    "量比": vol_ratio,
                    "换手率%": turn,
                    "T+1": ret['T+1'],
                    "T+3": ret['T+3'],
                    "T+5": ret['T+5']
                }
                save_result_to_csv(item)
                log_area.text(f"✅ {t_date} 命中: {row['name']} (得分{total_score})")

        mark_date_as_scanned(t_date)

    progress_bar.progress(100)
    status_box.update(label="处理完成！", state="complete", expanded=False)
    
    # ================= 仪表盘 =================
    if os.path.exists(CACHE_FILE):
        try:
            df_all = pd.read_csv(CACHE_FILE)
            
            # 单日模式只看当天
            if mode == "单日扫描":
                df_all = df_all[df_all['日期'].astype(str) == start_date_str]
                if df_all.empty:
                    st.warning(f"{start_date_str} 未发现符合条件的股票。")
                    st.stop()

            # 排序
            if "打分" in sort_method:
                df_sorted = df_all.sort_values("综合得分", ascending=False)
            elif "换手" in sort_method:
                df_sorted = df_all.sort_values("换手率%", ascending=False)
            else:
                df_sorted = df_all
                
            top_5 = df_sorted.head(5)
            
            t3_avg = top_5['T+3'].mean() if 'T+3' in top_5 else 0
            win_count = len(top_5[top_5['T+3'] > 0]) if 'T+3' in top_5 else 0
            win_rate = win_count / len(top_5) * 100 if len(top_5) > 0 else 0
            
            with dashboard_placeholder.container():
                st.divider()
                st.markdown(f"## 📊 战报 (日期: {start_date_str} - {end_date_str})")
                k1, k2 = st.columns(2)
                k1.metric("Top5 平均T+3收益", f"{t3_avg:.2f}%")
                k2.metric("Top5 T+3胜率", f"{win_rate:.0f}%")
                
                # 【修改处】移除了 .style.background_gradient，防止报错
                st.dataframe(df_sorted, use_container_width=True)
                
                with open(CACHE_FILE, "rb") as f:
                    st.download_button("📥 下载完整CSV", f, "turbo_result.csv")
        except Exception as e:
            st.error(f"读取结果出错: {e}")
