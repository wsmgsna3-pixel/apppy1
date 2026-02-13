import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# Streamlit 页面配置
# ==========================================
st.set_page_config(page_title="三日成妖·时光机回测", layout="wide")

st.title("🐉 三日成妖·时光机回测系统")
st.markdown("""
**策略逻辑 (主板/双创分轨制)：**
1. **主板 (60/00)**: 3日累计涨幅 > **12%**
2. **双创 (68/30)**: 3日累计涨幅 > **20%**
3. **成交量**: 连续3天放量 (潜伏期均量的 3倍)
4. **风控**: 
   - D+1 开盘 < -5% **不买**
   - 亏损 > 10% **止损**
""")

# ==========================================
# 1. 核心数据获取
# ==========================================
@st.cache_data(persist="disk", show_spinner=False)
def get_stock_pool(token, date_str):
    """获取某日的全市场股票列表"""
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # 获取基础信息区分板块
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
        return df[~df['name'].str.contains('ST')]
    except:
        return pd.DataFrame()

@st.cache_data(persist="disk", show_spinner=False)
def get_backtest_data(token, code, signal_date, latent_days=60, hold_days=10):
    """
    获取单只股票的完整数据链：
    潜伏期 (Past) + 爆发期 (Signal 3 Days) + 持有期 (Future 10 Days)
    """
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 计算时间范围
        # 往后多取一些天数，防止停牌
        end_dt = datetime.strptime(signal_date, '%Y%m%d') + timedelta(days=hold_days * 2 + 10)
        start_dt = datetime.strptime(signal_date, '%Y%m%d') - timedelta(days=latent_days + 10)
        
        df = pro.daily(ts_code=code, start_date=start_dt.strftime('%Y%m%d'), end_date=end_dt.strftime('%Y%m%d'))
        return df
    except:
        return pd.DataFrame()

# ==========================================
# 2. 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 时光机参数")
    user_token = st.text_input("Tushare Token:", type="password")
    
    target_date = st.date_input("选择回测日期 (D日)", datetime.now() - timedelta(days=20))
    st.caption("系统将穿越回这一天选股，并计算随后 10 天的收益。")
    
    st.subheader("📊 爆发标准")
    vol_mul = st.slider("量能放大倍数", 2.0, 5.0, 3.0, 0.5)
    
    run_btn = st.button("🚀 启动时光机")

# ==========================================
# 3. 策略核心：信号检测 + 模拟交易
# ==========================================
def run_backtest():
    if not user_token:
        st.error("请输入 Token")
        return

    d_str = target_date.strftime('%Y%m%d')
    st.info(f"⏳ 正在穿越回 {d_str} ...")
    
    # 1. 获取股票池
    df_pool = get_stock_pool(user_token, d_str)
    if df_pool.empty:
        st.error("无法获取股票池")
        return

    # 为了演示速度，这里只扫描当天活跃的股票 (实际生产可扫全市场)
    # 我们先获取 D日的日线，只回测 D日放量的股票，节省 API 额度
    pro = ts.pro_api()
    try:
        df_daily_d = pro.daily(trade_date=d_str)
        # 初筛：成交量 > 10000手 (活跃)
        active_codes = df_daily_d[df_daily_d['vol'] > 10000]['ts_code'].tolist()
        # 过滤池子
        scan_list = df_pool[df_pool['ts_code'].isin(active_codes)]
    except:
        st.warning("日线获取失败，使用部分列表扫描")
        scan_list = df_pool.head(500)

    results = []
    progress = st.progress(0)
    log_area = st.empty()
    
    # 2. 循环扫描
    total = len(scan_list)
    for i, (_, row) in enumerate(scan_list.iterrows()):
        code = row['ts_code']
        name = row['name']
        market = row['market'] # 主板/科创/创业
        
        progress.progress((i+1)/total)
        
        # 获取数据 (潜伏+爆发+持有)
        df_all = get_backtest_data(user_token, code, d_str)
        if len(df_all) < 30: continue
        
        # 按日期正序排列
        df_all = df_all.sort_values('trade_date').reset_index(drop=True)
        
        # 定位 D日 (Signal Date) 的索引
        try:
            d_idx = df_all[df_all['trade_date'] == d_str].index[0]
        except:
            continue # 停牌或无数据
            
        # === A. 信号检测 (Signal Detection) ===
        # 需要 D, D-1, D-2 (共3天)
        if d_idx < 62: continue # 数据不够算潜伏期
        
        df_burst = df_all.iloc[d_idx-2 : d_idx+1] # 3天爆发期
        df_latent = df_all.iloc[d_idx-62 : d_idx-2] # 60天潜伏期
        
        # 1. 量能判定
        latent_vol = df_latent['vol'].mean()
        if latent_vol == 0: continue
        
        # 检查这 3 天是否每天都放量 (或者 均量达标)
        # 这里用严谨版：3天均量 > 3倍潜伏
        burst_vol = df_burst['vol'].mean()
        if burst_vol < latent_vol * vol_mul: continue
        
        # 2. 涨幅判定 (分板块)
        price_start = df_burst.iloc[0]['open'] # D-2 开盘
        price_end = df_burst.iloc[-1]['close'] # D日 收盘
        cum_rise = (price_end - price_start) / price_start * 100
        
        is_startup = ('300' in code) or ('688' in code) or (market == '创业板') or (market == '科创板')
        threshold = 20 if is_startup else 12
        
        if cum_rise < threshold: continue
        
        # 3. 形态判定 (重心上移 + 第一天大阳)
        if df_burst.iloc[0]['pct_chg'] < 5: continue # 第一天要猛
        if df_burst.iloc[-1]['close'] < df_burst.iloc[0]['close']: continue # 重心不能下沉
        
        # === B. 模拟交易 (Simulation) ===
        # D+1 买入
        if d_idx + 1 >= len(df_all):
            results.append({'代码': code, '名称': name, '状态': '信号触发，但无后续数据'})
            continue
            
        d1_row = df_all.iloc[d_idx + 1]
        open_price = d1_row['open']
        pre_close = d1_row['pre_close']
        
        # 风控1: 低开 -5% 不买
        open_pct = (open_price - pre_close) / pre_close * 100
        if open_pct < -5:
            results.append({'代码': code, '名称': name, '状态': '❌ D+1 低开超-5%，放弃买入'})
            continue
            
        # 买入成功，开始持仓推演
        buy_price = open_price
        stop_price = buy_price * 0.90 # -10% 止损线
        
        trade_res = {
            '代码': code,
            '名称': name,
            '板块': '双创' if is_startup else '主板',
            '3日涨幅(%)': round(cum_rise, 1),
            '买入价': buy_price,
            '止损价': stop_price,
            '状态': '✅ 买入持有',
            'D+1收益(%)': 0.0,
            'D+3收益(%)': 0.0,
            'D+5收益(%)': 0.0,
            'D+7收益(%)': 0.0,
            'D+10收益(%)': 0.0,
            '触发止损': '否',
            '最高触及(%)': 0.0
        }
        
        triggered_stop = False
        max_high = -999
        
        # 遍历 D+1 到 D+10 (或者数据结束)
        hold_len = min(10, len(df_all) - (d_idx + 1))
        
        for h in range(hold_len):
            day_row = df_all.iloc[d_idx + 1 + h]
            curr_close = day_row['close']
            curr_low = day_row['low']
            curr_high = day_row['high']
            
            # 计算最高触及
            high_ret = (curr_high - buy_price) / buy_price * 100
            if high_ret > max_high: max_high = high_ret
            
            # 检查止损
            if not triggered_stop and curr_low < stop_price:
                triggered_stop = True
                trade_res['触发止损'] = f"Day+{h+1}"
                # 止损按 -10% 算 (假设触价即出)
                final_ret = -10.0
                
                # 止损后，后面的收益都锁死在 -10%
                for k in [1, 3, 5, 7, 10]:
                    if k >= h+1: trade_res[f'D+{k}收益(%)'] = -10.0
                break 
            
            # 记录关键节点的收益
            ret = (curr_close - buy_price) / buy_price * 100
            day_num = h + 1
            if day_num in [1, 3, 5, 7, 10]:
                trade_res[f'D+{day_num}收益(%)'] = round(ret, 2)
        
        trade_res['最高触及(%)'] = round(max_high, 2)
        results.append(trade_res)

    progress.empty()
    
    # 4. 展示报告
    if results:
        df_res = pd.DataFrame(results)
        
        # 区分买入的和没买入的
        df_traded = df_res[df_res['状态'] == '✅ 买入持有'].copy()
        df_skipped = df_res[df_res['状态'].str.contains('放弃')]
        
        st.success(f"回测结束！共触发信号 {len(df_res)} 次。其中符合买入条件 {len(df_traded)} 次。")
        
        if not df_traded.empty:
            # 统计胜率 (以 D+5 为例)
            win_count = len(df_traded[df_traded['D+5收益(%)'] > 0])
            win_rate = win_count / len(df_traded) * 100
            
            # 统计平均收益
            avg_d5 = df_traded['D+5收益(%)'].mean()
            avg_max = df_traded['最高触及(%)'].mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("D+5 胜率", f"{win_rate:.1f}%")
            col1.caption("持仓5天盈利比例")
            
            col2.metric("D+5 平均收益", f"{avg_d5:.2f}%")
            col2.caption("持仓5天平均盈亏")
            
            col3.metric("平均最高冲高", f"{avg_max:.2f}%")
            col3.caption("持仓期间最高摸到多少")
            
            st.markdown("### 📜 交易明细表 (重点看 D+3 和 D+5)")
            # 颜色标记
            st.dataframe(df_traded.style.applymap(lambda x: 'color: red' if isinstance(x, (int, float)) and x > 0 else ('color: green' if isinstance(x, (int, float)) and x < 0 else ''), subset=['D+1收益(%)', 'D+3收益(%)', 'D+5收益(%)', 'D+10收益(%)']))
        
        if not df_skipped.empty:
            with st.expander("查看被风控拦截的股票 (D+1低开 > -5%)"):
                st.dataframe(df_skipped)
            
    else:
        st.warning(f"{d_str} 当天没有发现符合【三日成妖】形态的股票。")

if run_btn:
    run_backtest()
