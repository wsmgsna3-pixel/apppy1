# -*- coding: utf-8 -*-
"""
选股王 · V35.14 胜率强化版
------------------------------------------------
修改记录:
1. [胜率过滤] 跌幅过滤：昨日跌幅限制在 -4% ~ 0%，拒绝深度破位死鱼。
2. [分值重塑] 对获利盘(Winner_Rate)实施惩罚机制：仅奖励获利盘在 30%-60% 的标的。
3. [乖离锁定] 增加均线收敛过滤，拒绝距离 20日线过远的离谱个股。
4. [T+1核心] 继承 V35.9/11 的所有实战法则（10%止盈，6%止损）。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import concurrent.futures 
import os
import pickle

warnings.filterwarnings("ignore")

# 缓存文件名统一
CACHE_FILE_NAME = "market_data_cache_v35_10.pkl" 

# ---------------------------
# (保留 V35.11 基础 API 与 Data 获取逻辑)
# ---------------------------
# [注：此处代码逻辑同 V35.11，省略冗余，请完整粘贴以下核心修改部分]

# ---------------------------
# 核心修改：动态打分逻辑 (V35.14)
# ---------------------------
def dynamic_score(r):
    # 1. 资金流基础分
    mf_ratio = r['net_mf'] / (r['circ_mv'] * 10000 + 1) if r['circ_mv'] > 0 else 0
    base_score = r['macd'] * 1000 
    base_score += min(max(mf_ratio * 10000, -500), 1000)
    
    # 2. 🌟 获利盘钟形惩罚 (核心提升胜率点)
    # 奖励 30%-50% 的票，惩罚 >60% (浮筹过多) 和 <20% (无人关注) 的票
    wr = r['winner_rate']
    if 30 <= wr <= 55: base_score += 1500
    elif wr > 65: base_score -= 1000
    elif wr < 20: base_score -= 500
    
    # 3. 形态加分
    if 55 < r['rsi'] < 80: base_score += 500
    
    return base_score

# ---------------------------
# 核心回测循环：更新过滤参数
# ---------------------------
# 在 run_backtest_for_a_day 中：
# df = df[(df['pct_chg'] >= -4.0) & (df['pct_chg'] <= 0.0)] # 锁死跌幅范围

# 均线乖离限制：
# if ind['ma20'] > 0 and abs(d0_close - ind['ma20']) / ind['ma20'] > 0.08: continue 
