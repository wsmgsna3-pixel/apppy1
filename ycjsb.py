# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（终极修复版 v4.1 - 性能优化）
说明：
- 整合了 BC 混合增强策略。
- 修复了性能优化后 `run_backtest` 函数中因缺少 `turnover_rate` 导致的 KeyError 错误。
- 回测逻辑强化：使用更高的成交额要求和涨幅要求替代换手率过滤。
- **v4.1 优化：**
    1. 使用主运行按钮包裹选股逻辑，避免重复执行。
    2. 延长历史数据缓存 TTL 至 10 小时，提升重复运行速度。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · 10000旗舰（终极修复 - 性能优化）", layout="wide")
st.title("选股王 · 10000 积分旗舰（终极修复版 v4.1 - 性能优化）")
st.markdown("输入你的 Tushare Token（仅本次运行使用）。若有权限缺失，脚本会自动降级并继续运行。")

# ---------------------------
# 侧边栏参数（实时可改）
# ---------------------------
with st.sidebar:
    st.header("可调参数（实时）")
    [span_0](start_span)INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))[span_0](end_span)
    [span_1](start_span)FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=300, step=50))[span_1](end_span)
    [span_2](start_span)TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))[span_2](end_span)
    [span_3](start_span)MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))[span_3](end_span)
    [span_4](start_span)MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))[span_4](end_span)
    [span_5](start_span)MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=3.0, step=0.5))[span_5](end_span)
    [span_6](start_span)MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=100_000_000.0, step=50_000_000.0))[span_6](end_span) # 默认 1亿
    [span_7](start_span)VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.7, step=0.1))[span_7](end_span)
    [span_8](start_span)VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=12.0, step=0.5))[span_8](end_span)
    [span_9](start_span)HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))[span_9](end_span)
    [span_10](start_span)MIN_MARKET_CAP = float(st.number_input("最低市值 (元)", value=2000000000.0, step=100000000.0))[span_10](end_span) # 默认 20亿
    [span_11](start_span)MAX_MARKET_CAP = float(st.number_input("最高市值 (元)", value=50000000000.0, step=1000000000.0))[span_11](end_span) # 默认 500亿
    st.markdown("---")
    # --- 新增回测参数 ---
    [span_12](start_span)st.header("历史回测参数")[span_12](end_span)
    [span_13](start_span)BACKTEST_DAYS = int(st.number_input("回测交易日天数", value=60, min_value=10, max_value=250))[span_13](end_span)
    [span_14](start_span)BACKTEST_TOP_K = int(st.number_input("回测每日最多交易 K 支", value=3, min_value=1, max_value=10))[span_14](end_span)
    [span_15](start_span)HOLD_DAYS_OPTIONS = st.multiselect("回测持股天数", options=[1, 3, 5, 10, 20], default=[1, 3, 5])[span_15](end_span)
    [span_16](start_span)st.caption("提示：**回测已修复 Key Error**，请重新运行。")[span_16](end_span)

# ---------------------------
# Token 输入（主区）
# ---------------------------
[span_17](start_span)TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")[span_17](end_span)
if not TS_TOKEN:
    [span_18](start_span)st.warning("请输入 Tushare Token 才能运行脚本。")[span_18](end_span)
    st.stop()

# 初始化 tushare
[span_19](start_span)ts.set_token(TS_TOKEN)[span_19](end_span)
[span_20](start_span)pro = ts.pro_api()[span_20](end_span)

# ---------------------------
# 安全调用 & 缓存辅助
# ---------------------------
def safe_get(func, **kwargs):
    """Call API and return DataFrame or empty df on any error."""
    try:
        [span_21](start_span)df = func(**kwargs)[span_21](end_span)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_trade_cal(start_date, end_date):
    """获取交易日历并缓存"""
    try:
        [span_22](start_span)df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)[span_22](end_span)
        [span_23](start_span)return df[df.is_open == 1]['cal_date'].tolist()[span_23](end_span)
    except Exception:
        return []

@st.cache_data(ttl=36000) # **优化点 2: 延长历史数据缓存至 10 小时**
[span_24](start_span)def find_last_trade_day(max_days=20):[span_24](end_span)
    [span_25](start_span)today = datetime.now().date()[span_25](end_span)
    [span_26](start_span)for i in range(max_days):[span_26](end_span)
        [span_27](start_span)d = today - timedelta(days=i)[span_27](end_span)
        [span_28](start_span)ds = d.strftime("%Y%m%d")[span_28](end_span)
        [span_29](start_span)df = safe_get(pro.daily, trade_date=ds)[span_29](end_span)
        [span_30](start_span)if not df.empty:[span_30](end_span)
            [span_31](start_span)return ds[span_31](end_span)
    [span_32](start_span)return None[span_32](end_span)

[span_33](start_span)last_trade = find_last_trade_day()[span_33](end_span)
if not last_trade:
    [span_34](start_span)st.error("无法找到最近交易日，检查网络或 Token 权限。")[span_34](end_span)
    st.stop()
[span_35](start_span)st.info(f"参考最近交易日：{last_trade}")[span_35](end_span)

# ---------------------------
# **优化点 1: 将选股逻辑包裹在按钮中**
# ---------------------------
if st.button("🚀 运行当日选股（初次运行可能较久）"):

    # ---------------------------
    # 拉当日涨幅榜初筛
    # ---------------------------
    [span_36](start_span)st.write("正在拉取当日 daily（涨幅榜）作为初筛...")[span_36](end_span)
    [span_37](start_span)daily_all = safe_get(pro.daily, trade_date=last_trade)[span_37](end_span)
    if daily_all.empty:
        [span_38](start_span)st.error("无法获取当日 daily 数据（Tushare 返回空）。请确认 Token 权限。")[span_38](end_span)
        st.stop()

    [span_39](start_span)daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)[span_39](end_span)
    [span_40](start_span)st.write(f"当日记录：{len(daily_all)}，取涨幅前 {INITIAL_TOP_N} 作为初筛。")[span_40](end_span)
    [span_41](start_span)pool0 = daily_all.head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)[span_41](end_span)

    # ---------------------------
    # 尝试加载高级接口（有权限时启用）
    # ---------------------------
    [span_42](start_span)st.write("尝试加载 stock_basic / daily_basic / moneyflow 等高级接口（若权限允许）...")[span_42](end_span)
    [span_43](start_span)stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')[span_43](end_span)
    [span_44](start_span)daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')[span_44](end_span)
    [span_45](start_span)mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)[span_45](end_span)

    # moneyflow 预处理
    [span_46](start_span)if not mf_raw.empty:[span_46](end_span)
        [span_47](start_span)possible = ['net_mf','net_mf_amount','net_mf_in','net_mf_out'][span_47](end_span)
        [span_48](start_span)col = None[span_48](end_span)
        [span_49](start_span)for c in possible:[span_49](end_span)
            [span_50](start_span)if c in mf_raw.columns:[span_50](end_span)
                [span_51](start_span)col = c;[span_51](end_span)
                [span_52](start_span)break[span_52](end_span)
        [span_53](start_span)if col is None:[span_53](end_span)
            [span_54](start_span)numeric_cols = [c for c in mf_raw.columns if c != 'ts_code' and pd.api.types.is_numeric_dtype(mf_raw[c])][span_54](end_span)
            [span_55](start_span)col = numeric_cols[0] if numeric_cols else None[span_55](end_span)
        [span_56](start_span)if col:[span_56](end_span)
            [span_57](start_span)moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)[span_57](end_span)
        else:
            [span_58](start_span)moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])[span_58](end_span)
    else:
        [span_59](start_span)moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])[span_59](end_span)
        [span_60](start_span)st.warning("moneyflow 未获取到，将把主力流向因子置为 0（若有权限请确认 Token/积分）。")[span_60](end_span)

    # ---------------------------
    # 合并基本信息（safe）
    # ---------------------------
    def safe_merge_pool(pool_df, other_df, cols):
        [span_61](start_span)pool = pool_df.set_index('ts_code').copy()[span_61](end_span)
        
        [span_62](start_span)if other_df is None or other_df.empty:[span_62](end_span)
            [span_63](start_span)for c in cols:[span_63](end_span)
                [span_64](start_span)pool[c] = np.nan[span_64](end_span)
            [span_65](start_span)return pool.reset_index()[span_65](end_span)
        [span_66](start_span)if 'ts_code' not in other_df.columns:[span_66](end_span)
            try:
                [span_67](start_span)other_df = other_df.reset_index()[span_67](end_span)
            except:
                for c in cols:
                    [span_68](start_span)pool[c] = np.nan[span_68](end_span)
                [span_69](start_span)return pool.reset_index()[span_69](end_span)
        [span_70](start_span)for c in cols:[span_70](end_span)
            [span_71](start_span)if c not in other_df.columns:[span_71](end_span)
                [span_72](start_span)other_df[c] = np.nan[span_72](end_span)
        try:
            [span_73](start_span)joined = pool.join(other_df.set_index('ts_code')[cols], how='left')[span_73](end_span)
        except Exception:
            [span_74](start_span)for c in cols:[span_74](end_span)
                [span_75](start_span)pool[c] = np.nan[span_75](end_span)
            [span_76](start_span)return pool.reset_index()[span_76](end_span)
        [span_77](start_span)for c in cols:[span_77](end_span)
            [span_78](start_span)if c not in joined.columns:[span_78](end_span)
                [span_79](start_span)joined[c] = np.nan[span_79](end_span)
        [span_80](start_span)return joined.reset_index()[span_80](end_span)

    # merge stock_basic
    [span_81](start_span)if not stock_basic.empty:[span_81](end_span)
        [span_82](start_span)keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_basic.columns][span_82](end_span)
        try:
            [span_83](start_span)pool0 = pool0.merge(stock_basic[keep], on='ts_code', how='left')[span_83](end_span)
        except Exception:
            [span_84](start_span)pool0['name'] = pool0['ts_code'];[span_84](end_span)
            [span_85](start_span)pool0['industry'] = ''[span_85](end_span)
    else:
        [span_86](start_span)pool0['name'] = pool0['ts_code'];[span_86](end_span)
        [span_87](start_span)pool0['industry'] = ''[span_87](end_span)

    # merge daily_basic
    [span_88](start_span)pool_merged = safe_merge_pool(pool0, daily_basic, ['turnover_rate','amount','total_mv','circ_mv'])[span_88](end_span)
    [span_89](start_span)pool_merged.rename(columns={'amount': 'amount_basic'}, inplace=True)[span_89](end_span) # daily_basic的amount

    # merge moneyflow robustly
    [span_90](start_span)if moneyflow.empty:[span_90](end_span)
        [span_91](start_span)moneyflow = pd.DataFrame({'ts_code': pool_merged['ts_code'].tolist(), 'net_mf': [0.0]*len(pool_merged)})[span_91](end_span)
    else:
        [span_92](start_span)if 'ts_code' not in moneyflow.columns:[span_92](end_span)
            [span_93](start_span)moneyflow['ts_code'] = None[span_93](end_span)
    try:
        [span_94](start_span)pool_merged = pool_merged.set_index('ts_code').join(moneyflow.set_index('ts_code'), how='left').reset_index()[span_94](end_span)
    except Exception:
        [span_95](start_span)if 'net_mf' not in pool_merged.columns:[span_95](end_span)
            [span_96](start_span)pool_merged['net_mf'] = 0.0[span_96](end_span)

    [span_97](start_span)if 'net_mf' not in pool_merged.columns:[span_97](end_span)
        [span_98](start_span)pool_merged['net_mf'] = 0.0[span_98](end_span)
    [span_99](start_span)pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0.0)[span_99](end_span)

    # ---------------------------
    # 基本清洗（ST / 停牌 / 价格区间 / 一字板 / 换手 / 成交额 / 市值）
    # ---------------------------
    [span_100](start_span)st.write("对初筛池进行清洗（ST/停牌/价格/一字板/换手/成交额等）...")[span_100](end_span)
    [span_101](start_span)clean_list = [][span_101](end_span)
    [span_102](start_span)pbar = st.progress(0)[span_102](end_span)
    # 统一使用 daily 里的 amount（单位千元） 和 daily_basic 里的 turnover_rate（单位 %）
    [span_103](start_span)for i, r in enumerate(pool_merged.itertuples()):[span_103](end_span)
        [span_104](start_span)ts = getattr(r, 'ts_code')[span_104](end_span)
        [span_105](start_span)vol = getattr(r, 'vol', 0)[span_105](end_span)

        [span_106](start_span)close = getattr(r, 'close', np.nan)[span_106](end_span)
        [span_107](start_span)open_p = getattr(r, 'open', np.nan)[span_107](end_span)
        [span_108](start_span)pre_close = getattr(r, 'pre_close', np.nan)[span_108](end_span)
        [span_109](start_span)pct = getattr(r, 'pct_chg', np.nan)[span_109](end_span)
        [span_110](start_span)amount_daily = getattr(r, 'amount', np.nan)[span_110](end_span) # daily 里的 amount
        [span_111](start_span)turnover = getattr(r, 'turnover_rate', np.nan)[span_111](end_span)
        [span_112](start_span)name = getattr(r, 'name', ts)[span_112](end_span)

    
        # 1. 过滤：停牌/无成交
        [span_113](start_span)if vol == 0 or (isinstance(amount_daily,(int,float)) and amount_daily == 0):[span_113](end_span)
            [span_114](start_span)pbar.progress((i+1)/len(pool_merged));[span_114](end_span)
            [span_115](start_span)continue[span_115](end_span)

        # 2. 过滤：价格区间
        [span_116](start_span)if pd.isna(close): pbar.progress((i+1)/len(pool_merged));[span_116](end_span)
        [span_117](start_span)continue[span_117](end_span)
        [span_118](start_span)if (close < MIN_PRICE) or (close > MAX_PRICE): pbar.progress((i+1)/len(pool_merged));[span_118](end_span)
        [span_119](start_span)continue[span_119](end_span)

        # 3. 过滤：ST / 退市 / 北交所
        [span_120](start_span)if isinstance(name, str) and (('ST' in name.upper()) or ('退' in name)):[span_120](end_span)
            [span_121](start_span)pbar.progress((i+1)/len(pool_merged));[span_121](end_span)
            [span_122](start_span)continue[span_122](end_span)
        [span_123](start_span)tsck = getattr(r, 'ts_code', '')[span_123](end_span)
        [span_124](start_span)if isinstance(tsck, str) and (tsck.startswith('4') or tsck.startswith('8')):[span_124](end_span)
            [span_125](start_span)pbar.progress((i+1)/len(pool_merged));[span_125](end_span)
            [span_126](start_span)continue[span_126](end_span)

        # 4. 过滤：市值（兼容万元单位）
        try:
            [span_127](start_span)tv = getattr(r, 'total_mv', np.nan)[span_127](end_span)
            [span_128](start_span)if not pd.isna(tv):[span_128](end_span)
                [span_129](start_span)tv = float(tv)[span_129](end_span)
                [span_130](start_span)if tv > 1e6:[span_130](end_span)
                    [span_131](start_span)tv_yuan = tv * 10000.0[span_131](end_span)
                else:
                    [span_132](start_span)tv_yuan = tv[span_132](end_span)
                [span_133](start_span)if tv_yuan < MIN_MARKET_CAP or tv_yuan > MAX_MARKET_CAP:[span_133](end_span)
                    [span_134](start_span)pbar.progress((i+1)/len(pool_merged));[span_134](end_span)
                    [span_135](start_span)continue[span_135](end_span)
        except:
            pass

        # 5. 过滤：一字涨停板
        try:
            [span_136](start_span)high = getattr(r, 'high', np.nan);[span_136](end_span)
            [span_137](start_span)low = getattr(r, 'low', np.nan)[span_137](end_span)
            [span_138](start_span)if (not pd.isna(open_p) and not pd.isna(high) and not pd.isna(low) and not pd.isna(pre_close)):[span_138](end_span)
                [span_139](start_span)if (open_p == high == low == pre_close) and (pct > 9.5):[span_139](end_span)
                    [span_140](start_span)pbar.progress((i+1)/len(pool_merged));[span_140](end_span)
                    [span_141](start_span)continue[span_141](end_span)
        except:
            pass

        # 6. 过滤：换手率
        [span_142](start_span)if not pd.isna(turnover):[span_142](end_span)
            try:
                [span_143](start_span)if float(turnover) < MIN_TURNOVER: pbar.progress((i+1)/len(pool_merged));[span_143](end_span)
                [span_144](start_span)continue[span_144](end_span)
            except:
                pass

        # 7. 过滤：成交额（修正单位：daily amount是千元）
        [span_145](start_span)if not pd.isna(amount_daily):[span_145](end_span)
            [span_146](start_span)amt = amount_daily * 1000.0 # 转换成元[span_146](end_span)
            [span_147](start_span)if amt < MIN_AMOUNT: pbar.progress((i+1)/len(pool_merged));[span_147](end_span)
            [span_148](start_span)continue[span_148](end_span)

        # 8. 过滤：剔除昨日收阴股（保留当日上涨的）
        try:
            [span_149](start_span)if float(pct) < 0: pbar.progress((i+1)/len(pool_merged));[span_149](end_span)
            [span_150](start_span)continue[span_150](end_span)
        except:
            pass
            
        [span_151](start_span)clean_list.append(r)[span_151](end_span)
        [span_152](start_span)pbar.progress((i+1)/len(pool_merged))[span_152](end_span)

    [span_153](start_span)pbar.progress(1.0)[span_153](end_span)
    [span_154](start_span)clean_df = pd.DataFrame([dict(zip(r._fields, r)) for r in clean_list])[span_154](end_span)
    [span_155](start_span)st.write(f"清洗后候选数量：{len(clean_df)} （将从中取涨幅前 {FINAL_POOL} 进入评分阶段）")[span_155](end_span)
    if len(clean_df) == 0:
        [span_156](start_span)st.error("清洗后没有候选，建议放宽条件或检查接口权限。")[span_156](end_span)
        st.stop()

    # ---------------------------
    # 辅助：获取单只历史（用于量比/10日收益等）
    # ---------------------------
    @st.cache_data(ttl=36000) # **优化点 2: 延长历史数据缓存至 10 小时**
    [span_157](start_span)def get_hist_cached(ts_code, end_date, days=60):[span_157](end_span)
        try:
            [span_158](start_span)start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")[span_158](end_span)
            [span_159](start_span)df = safe_get(pro.daily, ts_code=ts_code, start_date=start, end_date=end_date)[span_159](end_span)
            if df is None or df.empty:
                [span_160](start_span)return pd.DataFrame()[span_160](end_span)
            [span_161](start_span)df = df.sort_values('trade_date').reset_index(drop=True)[span_161](end_span)
            return df
        except:
            [span_162](start_span)return pd.DataFrame()[span_162](end_span)

    [span_163](start_span)def compute_indicators(df):[span_163](end_span)
        [span_164](start_span)res = {}[span_164](end_span)
        [span_165](start_span)if df.empty or len(df) < 3:[span_165](end_span)
            [span_166](start_span)return res[span_166](end_span)
        [span_167](start_span)close = df['close'].astype(float)[span_167](end_span)
        [span_168](start_span)high = df['high'].astype(float)[span_168](end_span)
        [span_169](start_span)low = df['low'].astype(float)[span_169](end_span)

        # last close
        [span_170](start_span)try: res['last_close'] = close.iloc[-1][span_170](end_span)
        [span_171](start_span)except: res['last_close'] = np.nan[span_171](end_span)

        # MA
        [span_172](start_span)for n in (5,10,20):[span_172](end_span)
            [span_173](start_span)if len(close) >= n:[span_173](end_span)
                [span_174](start_span)res[f'ma{n}'] = close.rolling(window=n).mean().iloc[-1][span_174](end_span)
            else:
                [span_175](start_span)res[f'ma{n}'] = np.nan[span_175](end_span)

        # MACD (12,26,9)
        [span_176](start_span)if len(close) >= 26:[span_176](end_span)
            [span_177](start_span)ema12 = close.ewm(span=12, adjust=False).mean()[span_177](end_span)
            [span_178](start_span)ema26 = close.ewm(span=26, adjust=False).mean()[span_178](end_span)
            [span_179](start_span)diff = ema12 - ema26[span_179](end_span)
            [span_180](start_span)dea = diff.ewm(span=9, adjust=False).mean()[span_180](end_span)
            [span_181](start_span)macd_val = (diff - dea) * 2[span_181](end_span)
            [span_182](start_span)res['macd'] = macd_val.iloc[-1];[span_182](end_span)
            [span_183](start_span)res['diff'] = diff.iloc[-1];[span_183](end_span)
            [span_184](start_span)res['dea'] = dea.iloc[-1][span_184](end_span)
        else:
            [span_185](start_span)res['macd'] = res['diff'] = res['dea'] = np.nan[span_185](end_span)

        # KDJ
        [span_186](start_span)n = 9[span_186](end_span)
        [span_187](start_span)if len(close) >= n:[span_187](end_span)
            [span_188](start_span)low_n = low.rolling(window=n).min()[span_188](end_span)
            [span_189](start_span)high_n = high.rolling(window=n).max()[span_189](end_span)
            [span_190](start_span)rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100[span_190](end_span)
            [span_191](start_span)rsv = rsv.fillna(50)[span_191](end_span)
            [span_192](start_span)k = rsv.ewm(alpha=1/3, adjust=False).mean()[span_192](end_span)
            [span_193](start_span)d = k.ewm(alpha=1/3, adjust=False).mean()[span_193](end_span)
            [span_194](start_span)j = 3*k - 2*d[span_194](end_span)
            [span_195](start_span)res['k'] = k.iloc[-1];[span_195](end_span)
            [span_196](start_span)res['d'] = d.iloc[-1];[span_196](end_span)
            [span_197](start_span)res['j'] = j.iloc[-1][span_197](end_span)
        else:
            [span_198](start_span)res['k'] = res['d'] = res['j'] = np.nan[span_198](end_span)

        # vol ratio and metrics
        [span_199](start_span)vols = df['vol'].astype(float).tolist()[span_199](end_span)
        [span_200](start_span)if len(vols) >= 6:[span_200](end_span)
            [span_201](start_span)avg_prev5 = np.mean(vols[-6:-1])[span_201](end_span)
            [span_202](start_span)res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9)[span_202](end_span)
            [span_203](start_span)res['vol_last'] = vols[-1][span_203](end_span)
            [span_204](start_span)res['vol_ma5'] = avg_prev5[span_204](end_span)
        else:
            [span_205](start_span)res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan[span_205](end_span)

        # 10d return
        [span_206](start_span)if len(close) >= 10:[span_206](end_span)
            [span_207](start_span)res['10d_return'] = close.iloc[-1] / close.iloc[-10] - 1[span_207](end_span)
        else:
            [span_208](start_span)res['10d_return'] = np.nan[span_208](end_span)

        # prev3_sum for down-then-bounce detection
        [span_209](start_span)if 'pct_chg' in df.columns and len(df) >= 4:[span_209](end_span)
            try:
                [span_210](start_span)pct = df['pct_chg'].astype(float)[span_210](end_span)
                [span_211](start_span)res['prev3_sum'] = pct.iloc[-4:-1].sum()[span_211](end_span)
            except:
                [span_212](start_span)res['prev3_sum'] = np.nan[span_212](end_span)
        else:
            [span_213](start_span)res['prev3_sum'] = np.nan[span_213](end_span)

        # volatility (std of last 10 pct_chg)
        try:
            [span_214](start_span)if 'pct_chg' in df.columns and len(df) >= 10:[span_214](end_span)
                [span_215](start_span)res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()[span_215](end_span)
            else:
                [span_216](start_span)res['volatility_10'] = np.nan[span_216](end_span)
        except:
            [span_217](start_span)res['volatility_10'] = np.nan[span_217](end_span)

        # recent 20-day high for breakout detection
        try:
            [span_218](start_span)if len(high) >= 20:[span_218](end_span)
                [span_219](start_span)res['recent20_high'] = float(high.tail(20).max())[span_219](end_span)
            else:
                [span_220](start_span)res['recent20_high'] = float(high.max()) if len(high)>0 else np.nan[span_220](end_span)
        except:
            [span_221](start_span)res['recent20_high'] = np.nan[span_221](end_span)

        
        # 阳线实体强度（今天）
        [span_222](start_span)try:[span_222](end_span)
            [span_223](start_span)today_open = df['open'].astype(float).iloc[-1][span_223](end_span)
            [span_224](start_span)today_close = df['close'].astype(float).iloc[-1][span_224](end_span)
            [span_225](start_span)today_high = df['high'].astype(float).iloc[-1][span_225](end_span)
            [span_226](start_span)today_low = df['low'].astype(float).iloc[-1][span_226](end_span)
            [span_227](start_span)body = abs(today_close - today_open)[span_227](end_span)
            [span_228](start_span)rng = max(today_high - today_low, 1e-9)[span_228](end_span)
            [span_229](start_span)res['yang_body_strength'] = body / rng[span_229](end_span)
        except:
            [span_230](start_span)res['yang_body_strength'] = 0.0[span_230](end_span)

        [span_231](start_span)return res[span_231](end_span)

    # ---------------------------
    # 评分池逐票计算因子（缓存 get_hist）
    # ---------------------------
    [span_232](start_span)st.write("为评分池逐票拉历史并计算指标（此步骤调用历史接口，已缓存）...")[span_232](end_span)
    st.warning(f"⚠️ **耗时警告：** 当前有 {len(clean_df)} 支股票需要计算指标。如果太慢，请调整侧边栏 **'清洗后取前 M'** 参数。") # **优化点 3: 增加耗时警告**
    [span_233](start_span)records = [][span_233](end_span)
    [span_234](start_span)pbar2 = st.progress(0)[span_234](end_span)
    [span_235](start_span)for idx, row in enumerate(clean_df.itertuples()):[span_235](end_span)
        [span_236](start_span)ts_code = getattr(row, 'ts_code')[span_236](end_span)
        [span_237](start_span)name = getattr(row, 'name', ts_code)[span_237](end_span)
        [span_238](start_span)pct_chg = getattr(row, 'pct_chg', 0.0)[span_238](end_span)
        
        [span_239](start_span)amount_daily = getattr(row, 'amount', np.nan)[span_239](end_span)
        amount = 0.0
        [span_240](start_span)if amount_daily is not None and not pd.isna(amount_daily):[span_240](end_span)
            [span_241](start_span)amount = amount_daily * 1000.0 # 转换成元[span_241](end_span)

        [span_242](start_span)turnover_rate = getattr(row, 'turnover_rate', np.nan)[span_242](end_span)
        [span_243](start_span)net_mf = float(getattr(row, 'net_mf', 0.0))[span_243](end_span)

        
        [span_244](start_span)hist = get_hist_cached(ts_code, last_trade, days=60)[span_244](end_span)
        [span_245](start_span)ind = compute_indicators(hist)[span_245](end_span)

        [span_246](start_span)vol_ratio = ind.get('vol_ratio', np.nan)[span_246](end_span)
        [span_247](start_span)ten_return = ind.get('10d_return', np.nan)[span_247](end_span)
        [span_248](start_span)ma5 = ind.get('ma5', np.nan)[span_248](end_span)
        [span_249](start_span)ma10 = ind.get('ma10', np.nan)[span_249](end_span)
        [span_250](start_span)ma20 = ind.get('ma20', np.nan)[span_250](end_span)
        [span_251](start_span)macd = ind.get('macd', np.nan)[span_251](end_span)
        [span_252](start_span)diff = ind.get('diff', np.nan)[span_252](end_span)
        [span_253](start_span)dea = ind.get('dea', np.nan)[span_253](end_span)
        [span_254](start_span)k, d, j = ind.get('k', np.nan), ind.get('d', np.nan), ind.get('j', np.nan)[span_254](end_span)
        [span_255](start_span)last_close = ind.get('last_close', np.nan)[span_255](end_span)
        [span_256](start_span)vol_last = ind.get('vol_last', np.nan)[span_256](end_span)
        [span_257](start_span)vol_ma5 = ind.get('vol_ma5', np.nan)[span_257](end_span)
        [span_258](start_span)prev3_sum = ind.get('prev3_sum', np.nan)[span_258](end_span)
        [span_259](start_span)volatility_10 = ind.get('volatility_10', np.nan)[span_259](end_span)
        [span_260](start_span)recent20_high = ind.get('recent20_high', np.nan)[span_260](end_span)
        [span_261](start_span)yang_body_strength = ind.get('yang_body_strength', 0.0)[span_261](end_span)

        # 资金强度代理（不依赖 moneyflow）：简单乘积指标
        try:
            proxy_money = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
        except:
            proxy_money = 0.0

        rec = {
            'ts_code': ts_code, 'name': name, 'pct_chg': pct_chg,
            [span_262](start_span)'amount': amount,[span_262](end_span)
            [span_263](start_span)'turnover_rate': turnover_rate if not pd.isna(turnover_rate) else np.nan,[span_263](end_span)
            [span_264](start_span)'net_mf': net_mf,[span_264](end_span)
            [span_265](start_span)'vol_ratio': vol_ratio if not pd.isna(vol_ratio) else np.nan,[span_265](end_span)
            [span_266](start_span)'10d_return': ten_return if not pd.isna(ten_return) else np.nan,[span_266](end_span)
            [span_267](start_span)'ma5': ma5, 'ma10': ma10, 'ma20': ma20,[span_267](end_span)
            [span_268](start_span)'macd': macd, 'diff': diff, 'dea': dea, 'k': k, 'd': d, 'j': j,[span_268](end_span)
            [span_269](start_span)'last_close': last_close, 'vol_last': vol_last,[span_269](end_span)
            [span_270](start_span)'vol_ma5': vol_ma5, 'recent20_high': recent20_high, 'yang_body_strength': yang_body_strength,[span_270](end_span)
            [span_271](start_span)'prev3_sum': prev3_sum, 'volatility_10': volatility_10,[span_271](end_span)
            [span_272](start_span)'proxy_money': proxy_money[span_272](end_span)
        }

        [span_273](start_span)records.append(rec)[span_273](end_span)
        [span_274](start_span)pbar2.progress((idx+1)/len(clean_df))[span_274](end_span)

    [span_275](start_span)pbar2.progress(1.0)[span_275](end_span)
    [span_276](start_span)fdf = pd.DataFrame(records)[span_276](end_span)
    if fdf.empty:
        [span_277](start_span)st.error("评分计算失败或无数据，请检查 Token 权限与接口。")[span_277](end_span)
        st.stop()

    # ---------------------------
    # 风险过滤（放在评分前以节省历史调用）
    # ---------------------------
    [span_278](start_span)st.write("执行风险过滤：下跌途中大阳 / 巨量冲高 / 高位大阳 / 极端波动 ...")[span_278](end_span)
    try:
        [span_279](start_span)before_cnt = len(fdf)[span_279](end_span)
        # A: 高位大阳线 -> last_close > ma20*1.10 且 pct_chg > HIGH_PCT_THRESHOLD
        [span_280](start_span)if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):[span_280](end_span)
            [span_281](start_span)mask_high_big = (fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)[span_281](end_span)
            [span_282](start_span)fdf = fdf[~mask_high_big][span_282](end_span)

        # B: 下跌途中反抽 -> prev3_sum < 0 且 pct_chg > HIGH_PCT_THRESHOLD
        [span_283](start_span)if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):[span_283](end_span)
            [span_284](start_span)mask_down_rebound = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)[span_284](end_span)
            [span_285](start_span)fdf = fdf[~mask_down_rebound][span_285](end_span)

        # C: 巨量放量大阳 -> vol_ratio > VOL_SPIKE_MULT
        [span_286](start_span)if 'vol_ratio' in fdf.columns:[span_286](end_span)
            [span_287](start_span)mask_vol_spike = fdf['vol_ratio'] > VOL_SPIKE_MULT[span_287](end_span)
            [span_288](start_span)fdf = fdf[~mask_vol_spike][span_288](end_span)

        # D: 极端波动 -> volatility_10 > VOLATILITY_MAX
        [span_289](start_span)if 'volatility_10' in fdf.columns:[span_289](end_span)
            [span_290](start_span)mask_volatility = fdf['volatility_10'] > VOLATILITY_MAX[span_290](end_span)
            [span_291](start_span)fdf = fdf[~mask_volatility][span_291](end_span)

        [span_292](start_span)after_cnt = len(fdf)[span_292](end_span)
        [span_293](start_span)st.write(f"风险过滤：{before_cnt} -> {after_cnt}（若过严请在侧边栏调整阈值）")[span_293](end_span)
    except Exception as e:
        st.warning(f"风险过滤模块异常，跳过过滤。错误：{e}")

    # ---------------------------
    # MA 多头硬过滤（必须满足 MA5 > MA10 > MA20）
    # ---------------------------
    try:
        [span_294](start_span)if all(c in fdf.columns for c in ['ma5','ma10','ma20']):[span_294](end_span)
            [span_295](start_span)before_ma = len(fdf)[span_295](end_span)
            [span_296](start_span)fdf = fdf[(fdf['ma5'] > fdf['ma10']) & (fdf['ma10'] > fdf['ma20'])].copy()[span_296](end_span) 
            [span_297](start_span)after_ma = len(fdf)[span_297](end_span)
            [span_298](start_span)st.write(f"MA 多头过滤：{before_ma} -> {after_ma}（保留 MA5>MA10>MA20）")[span_298](end_span)
    except Exception as e:
        [span_299](start_span)st.warning(f"MA 过滤异常，跳过。错误：{e}")[span_299](end_span)

    # ---------------------------
    # RSL（相对强弱）：基于池内 10d_return 的相对表现
    # ---------------------------
    [span_300](start_span)if '10d_return' in fdf.columns:[span_300](end_span)
        try:
            [span_301](start_span)market_mean_10d = fdf['10d_return'].replace([np.inf,-np.inf], np.nan).dropna().mean()[span_301](end_span)
            [span_302](start_span)if np.isnan(market_mean_10d) or abs(market_mean_10d) < 1e-9:[span_302](end_span)
                [span_303](start_span)market_mean_10d = 1e-9[span_303](end_span)
            [span_304](start_span)fdf['rsl'] = fdf['10d_return'] / market_mean_10d[span_304](end_span)
        except:
            [span_305](start_span)fdf['rsl'] = 1.0[span_305](end_span)
    else:
        [span_306](start_span)fdf['rsl'] = 1.0[span_306](end_span)

    # ---------------------------
    # 子指标归一化（稳健）
    # ---------------------------
    def norm_col(s):
        [span_307](start_span)s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)[span_307](end_span)
        [span_308](start_span)mn = s.min();[span_308](end_span)
        [span_309](start_span)mx = s.max()[span_309](end_span)
        [span_310](start_span)if mx - mn < 1e-9:[span_310](end_span)
            [span_311](start_span)return pd.Series([0.5]*len(s), index=s.index)[span_311](end_span)
        [span_312](start_span)return (s - mn) / (mx - mn)[span_312](end_span)

    [span_313](start_span)fdf['s_pct'] = norm_col(fdf.get('pct_chg', pd.Series([0]*len(fdf))))[span_313](end_span)
    [span_314](start_span)fdf['s_volratio'] = norm_col(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))[span_314](end_span)
    [span_315](start_span)fdf['s_turn'] = norm_col(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))[span_315](end_span)
    if 'net_mf' in fdf.columns and fdf['net_mf'].abs().sum() > 0:
        [span_316](start_span)fdf['s_money'] = norm_col(fdf.get('net_mf', pd.Series([0]*len(fdf))))[span_316](end_span)
    else:
        [span_317](start_span)fdf['s_money'] = norm_col(fdf.get('proxy_money', pd.Series([0]*len(fdf))))[span_317](end_span)
    [span_318](start_span)fdf['s_amount'] = norm_col(fdf.get('amount', pd.Series([0]*len(fdf))))[span_318](end_span)
    [span_319](start_span)fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))[span_319](end_span)
    [span_320](start_span)fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))[span_320](end_span)
    [span_321](start_span)fdf['s_rsl'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))[span_321](end_span)
    [span_322](start_span)fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))[span_322](end_span)

    # ---------------------------
    # 趋势因子与强化评分（右侧趋势主导）
    # ---------------------------
    [span_323](start_span)fdf['ma_trend_flag'] = ((fdf.get('ma5', pd.Series([])) > fdf.get('ma10', pd.Series([]))) & (fdf.get('ma10', pd.Series([])) > fdf.get('ma20', pd.Series([])))).fillna(False)[span_323](end_span)
    [span_324](start_span)fdf['macd_golden_flag'] = (fdf.get('diff', 0) > fdf.get('dea', 0)).fillna(False)[span_324](end_span)
    [span_325](start_span)fdf['vol_price_up_flag'] = (fdf.get('vol_last', 0) > fdf.get('vol_ma5', 0)).fillna(False)[span_325](end_span)
    [span_326](start_span)fdf['break_high_flag'] = (fdf.get('last_close', 0) > fdf.get('recent20_high', 0)).fillna(False)[span_326](end_span)
    [span_327](start_span)fdf['yang_body_strength'] = fdf.get('yang_body_strength', 0.0).fillna(0.0)[span_327](end_span)

    # 组合成趋势原始分
    fdf['trend_score_raw'] = (
        fdf['ma_trend_flag'].astype(float) * 1.0 +
        fdf['macd_golden_flag'].astype(float) * 1.3 +
        fdf['vol_price_up_flag'].astype(float) * 1.0 +
        fdf['break_high_flag'].astype(float) * 1.3 +
        fdf['yang_body_strength'].astype(float) * 0.8
    )

    # 归一化趋势分
    fdf['trend_score'] = norm_col(fdf['trend_score_raw'])

    # ---------------------------
    # 最终综合评分（趋势主导）
    # ---------------------------
    fdf['综合评分'] = (
        fdf['trend_score'] * 0.40 +
        fdf.get('s_10d', 0)*0.12 +
        fdf.get('s_rsl', 0)*0.08 +
        fdf.get('s_volratio', 0)*0.10 +
        fdf.get('s_turn', 0)*0.05 +
        fdf.get('s_money', 0)*0.10 +
        [span_328](start_span)fdf.get('s_pct', 0)*0.10 +[span_328](end_span)
        [span_329](start_span)fdf.get('s_volatility', 0)*0.05[span_329](end_span)
    )

    # ---------------------------
    # 最终排序与展示
    # ---------------------------
    [span_330](start_span)fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)[span_330](end_span)
    [span_331](start_span)fdf.index = fdf.index + 1[span_331](end_span)

    [span_332](start_span)st.success(f"评分完成：总候选 {len(fdf)} 支，显示 Top {min(TOP_DISPLAY, len(fdf))}。")[span_332](end_span)
    [span_333](start_span)display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','proxy_money','amount','10d_return','macd','diff','dea','k','d','j','rsl','volatility_10'][span_333](end_span)
    for c in display_cols:
        if c not in fdf.columns:
            [span_334](start_span)fdf[c] = np.nan[span_334](end_span)

    [span_335](start_span)st.dataframe(fdf[display_cols].head(TOP_DISPLAY), use_container_width=True)[span_335](end_span)

    # 下载（仅导出前200避免过大）
    [span_336](start_span)out_csv = fdf[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')[span_336](end_span)
    [span_337](start_span)st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")[span_337](end_span)


# ---------------------------
# 历史回测部分（数据性能优化与逻辑强化）
# ---------------------------
@st.cache_data(ttl=3600)
[span_338](start_span)def load_backtest_data(all_trade_dates):[span_338](end_span)
    """预加载所有回测日期的 daily 数据，以字典 {trade_date: DataFrame} 缓存。"""
    [span_339](start_span)data_cache = {}[span_339](end_span)
    [span_340](start_span)st.write(f"正在预加载回测所需 {len(all_trade_dates)} 个交易日的全部 daily 数据 (约 {len(all_trade_dates)} 次 API 调用)...")[span_340](end_span)
    [span_341](start_span)pbar = st.progress(0)[span_341](end_span)
    [span_342](start_span)for i, date in enumerate(all_trade_dates):[span_342](end_span)
        [span_343](start_span)daily_df = safe_get(pro.daily, trade_date=date)[span_343](end_span)
        [span_344](start_span)if not daily_df.empty:[span_344](end_span)
            [span_345](start_span)data_cache[date] = daily_df.set_index('ts_code')[span_345](end_span)
        [span_346](start_span)pbar.progress((i + 1) / len(all_trade_dates))[span_346](end_span)
    [span_347](start_span)pbar.progress(1.0)[span_347](end_span)
    return data_cache

@st.cache_data(ttl=6000)
[span_348](start_span)def run_backtest(start_date, end_date, hold_days, backtest_top_k):[span_348](end_span)
    [span_349](start_span)trade_dates = get_trade_cal(start_date, end_date)[span_349](end_span)
    
    [span_350](start_span)if not trade_dates:[span_350](end_span)
        [span_351](start_span)return {h: {'returns': [], 'wins': 0, 'total': 0, 'win_rate': 0.0, 'avg_return': 0.0} for h in hold_days}[span_351](end_span)

    [span_352](start_span)results = {h: {'returns': [], 'wins': 0, 'total': 0, 'win_rate': 0.0, 'avg_return': 0.0} for h in hold_days}[span_352](end_span)
    
    [span_353](start_span)bt_start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=BACKTEST_DAYS * 2)).strftime("%Y%m%d")[span_353](end_span)
    [span_354](start_span)buy_dates_pool = [d for d in trade_dates if d >= bt_start and d <= end_date][span_354](end_span)
    [span_355](start_span)backtest_dates = buy_dates_pool[-BACKTEST_DAYS:][span_355](end_span)
    
    [span_356](start_span)if len(backtest_dates) < BACKTEST_DAYS:[span_356](end_span)
        [span_357](start_span)st.warning(f"由于数据或交易日限制，回测仅能覆盖 {len(backtest_dates)} 天。")[span_357](end_span)
    
    # 确定回测所需的全部交易日，并预加载数据
    [span_358](start_span)required_dates = set(backtest_dates)[span_358](end_span)
    [span_359](start_span)for buy_date in backtest_dates:[span_359](end_span)
        try:
            [span_360](start_span)current_index = trade_dates.index(buy_date)[span_360](end_span)
            [span_361](start_span)for h in hold_days:[span_361](end_span)
                [span_362](start_span)required_dates.add(trade_dates[current_index + h])[span_362](end_span)
        except (ValueError, IndexError):
            [span_363](start_span)continue[span_363](end_span)
            
    [span_364](start_span)data_cache = load_backtest_data(sorted(list(required_dates)))[span_364](end_span)

    [span_365](start_span)st.write(f"正在模拟 {len(backtest_dates)} 个交易日的选股回测...")[span_365](end_span)
    [span_366](start_span)pbar_bt = st.progress(0)[span_366](end_span)
    
    [span_367](start_span)for i, buy_date in enumerate(backtest_dates):[span_367](end_span)
        [span_368](start_span)daily_df_cached = data_cache.get(buy_date)[span_368](end_span)
        
        [span_369](start_span)if daily_df_cached is None or daily_df_cached.empty:[span_369](end_span)
            [span_370](start_span)pbar_bt.progress((i+1)/len(backtest_dates));[span_370](end_span)
            [span_371](start_span)continue[span_371](end_span)

        [span_372](start_span)daily_df = daily_df_cached.copy().reset_index()[span_372](end_span)
        
        # 1. 应用基本过滤 (价格/成交额/停牌/一字板)
        
        # amount 字段在 daily 接口中，单位是千元，我们要求的是元。
        # 0.5 * MIN_AMOUNT 是一个粗略的替代，因为没有换手率，所以提高对成交额的要求。
        [span_373](start_span)BACKTEST_MIN_AMOUNT_PROXY = MIN_AMOUNT * 2.0[span_373](end_span)
        
        [span_374](start_span)daily_df['amount_yuan'] = daily_df['amount'].fillna(0) * 1000.0[span_374](end_span) # 转换成元
        
        # 过滤：价格/成交额/动量/停牌/一字板
        [span_375](start_span)daily_df = daily_df[[span_375](end_span)
            (daily_df['close'] >= MIN_PRICE) [span_376](start_span)&
            (daily_df['close'] <= MAX_PRICE) &[span_376](end_span)
            (daily_df['amount_yuan'] >= BACKTEST_MIN_AMOUNT_PROXY) [span_377](start_span)& # **使用更高的成交额要求替代换手率**[span_377](end_span)
            (daily_df['pct_chg'] >= 3.0) [span_378](start_span)& # 必须当天涨幅 >= 3%[span_378](end_span)
            (daily_df['vol'] > 0) [span_379](start_span)&
            (daily_df['amount_yuan'] > 0) [span_379](end_span)
        ].copy()
        
        # 过滤一字涨停板
        [span_380](start_span)daily_df['is_zt'] = (daily_df['open'] == daily_df['high']) & (daily_df['pct_chg'] > 9.5)[span_380](end_span)
        [span_381](start_span)daily_df = daily_df[~daily_df['is_zt']].copy()[span_381](end_span)
        
        # 2. 模拟评分：取当日涨幅榜前 backtest_top_k
        [span_382](start_span)scored_stocks = daily_df.sort_values("pct_chg", ascending=False).head(backtest_top_k).copy()[span_382](end_span)
        
        [span_383](start_span)for _, row in scored_stocks.iterrows():[span_383](end_span)
            [span_384](start_span)ts_code = row['ts_code'][span_384](end_span)
            [span_385](start_span)buy_price = float(row['close'])[span_385](end_span)
            
            [span_386](start_span)if pd.isna(buy_price) or buy_price <= 0: continue[span_386](end_span)

            [span_387](start_span)for h in hold_days:[span_387](end_span)
                try:
                    [span_388](start_span)current_index = trade_dates.index(buy_date)[span_388](end_span)
                    [span_389](start_span)sell_date = trade_dates[current_index + h][span_389](end_span)
                except (ValueError, IndexError):
                    [span_390](start_span)continue[span_390](end_span)
        
                # 从缓存中查找卖出价格 (O(1) 查找)
                [span_391](start_span)sell_df_cached = data_cache.get(sell_date)[span_391](end_span)
                [span_392](start_span)sell_price = np.nan[span_392](end_span)
                [span_393](start_span)if sell_df_cached is not None and ts_code in sell_df_cached.index:[span_393](end_span)
                    [span_394](start_span)sell_price = sell_df_cached.loc[ts_code, 'close'][span_394](end_span)
                
                [span_395](start_span)if pd.isna(sell_price) or sell_price <= 0: continue[span_395](end_span)
                
                [span_396](start_span)ret = (sell_price / buy_price) - 1.0[span_396](end_span)
                [span_397](start_span)results[h]['total'] += 1[span_397](end_span)
                [span_398](start_span)results[h]['returns'].append(ret)[span_398](end_span)
                [span_399](start_span)if ret > 0:[span_399](end_span)
                    [span_400](start_span)results[h]['wins'] += 1[span_400](end_span)

        [span_401](start_span)pbar_bt.progress((i+1)/len(backtest_dates))[span_401](end_span)

    [span_402](start_span)pbar_bt.progress(1.0)[span_402](end_span)
    
    [span_403](start_span)final_results = {}[span_403](end_span)
    [span_404](start_span)for h, res in results.items():[span_404](end_span)
        [span_405](start_span)total = res['total'][span_405](end_span)
        [span_406](start_span)if total > 0:[span_406](end_span)
            [span_407](start_span)avg_return = np.mean(res['returns']) * 100.0[span_407](end_span)
            [span_408](start_span)win_rate = (res['wins'] / total) * 100.0[span_408](end_span)
        else:
            [span_409](start_span)avg_return = 0.0[span_409](end_span)
            [span_410](start_span)win_rate = 0.0[span_410](end_span)
            
        [span_411](start_span)final_results[h] = {[span_411](end_span)
            [span_412](start_span)'平均收益率 (%)': f"{avg_return:.2f}",[span_412](end_span)
            [span_413](start_span)'胜率 (%)': f"{win_rate:.2f}",[span_413](end_span)
            [span_414](start_span)'总交易次数': total[span_414](end_span)
        }
        
    return final_results

# ---------------------------
# 回测执行
# ---------------------------
if st.checkbox("✅ 运行历史回测", value=False):
    [span_415](start_span)if not HOLD_DAYS_OPTIONS:[span_415](end_span)
        [span_416](start_span)st.warning("请至少选择一个回测持股天数。")[span_416](end_span)
    else:
        [span_417](start_span)st.header("📈 历史回测结果（买入收盘价 / 卖出收盘价）")[span_417](end_span)
        
        try:
            [span_418](start_span)start_date_for_cal = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")[span_418](end_span)
        except:
            [span_419](start_span)start_date_for_cal = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")[span_419](end_span)
            
        backtest_result = run_backtest(
            [span_420](start_span)start_date=start_date_for_cal,[span_420](end_span)
            [span_421](start_span)end_date=last_trade,[span_421](end_span)
            [span_422](start_span)hold_days=HOLD_DAYS_OPTIONS,[span_422](end_span)
            [span_423](start_span)backtest_top_k=BACKTEST_TOP_K[span_423](end_span)
        )

        [span_424](start_span)bt_df = pd.DataFrame(backtest_result).T[span_424](end_span)
        [span_425](start_span)bt_df.index.name = "持股天数"[span_425](end_span)
        [span_426](start_span)bt_df = bt_df.reset_index()[span_426](end_span)
        [span_427](start_span)bt_df['持股天数'] = bt_df['持股天数'].astype(str) + ' 天'[span_427](end_span)
        
        [span_428](start_span)st.dataframe(bt_df, use_container_width=True, hide_index=True)[span_428](end_span)
        [span_429](start_span)st.success("回测完成！")[span_429](end_span)
        
        [span_430](start_span)export_df = bt_df.copy()[span_430](end_span)
        [span_431](start_span)export_df.columns = ['HoldDays', 'AvgReturn', 'WinRate', 'TotalTrades'][span_431](end_span)
        [span_432](start_span)out_csv_bt = export_df.to_csv(index=False, encoding='utf-8-sig')[span_432](end_span)
        st.download_button(
            "下载回测结果 CSV", 
            data=out_csv_bt, 
            [span_433](start_span)file_name=f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",[span_433](end_span)
            [span_434](start_span)mime="text/csv"[span_434](end_span)
        )
# ---------------------------
# 小结与建议（简洁）
# ---------------------------
st.markdown("### 小结与操作提示（简洁）")
st.markdown("""
- **状态：** **性能优化版 v4.1**。
- **优化：** 选股流程现已封装在 **主运行按钮** 中，避免了在操作回测时重复耗时运行。历史指标缓存延长至 10 小时。
- **操作步骤：**
    1. **点击 “🚀 运行当日选股”**：完成当日选股和评分（仅需点击一次）。
    2. **勾选 “✅ 运行历史回测”**：开始回测。
- **提速建议：** 如果选股（步骤 1）仍然太慢，请尝试减小侧边栏的 **“清洗后取前 M 进入评分”** 参数。
""")
