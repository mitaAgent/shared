"""
ETF 动量信号模块 — 差异化 R²加权动量

核心差异化 (vs 五福闹新春):
  - 双回看期 (15 + 35 日) 而非单一 25 日
  - Regime v3 叠加仓位管理
  - 持仓 2-3 只而非 1 只
  - 风格条件过滤候选池

输出:
  dict[symbol] → (combo_score, r2_short, r2_long)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


LOOKBACK_SHORT = 15
LOOKBACK_LONG = 35
MIN_R2 = 0.3
SHORT_WEIGHT = 0.4
LONG_WEIGHT = 0.6


def compute_momentum_score(
    closes: np.ndarray,
    lookback: int,
) -> tuple[float, float, float]:
    """
    R²加权线性回归动量得分。

    Returns: (momentum_score, annualized_return, r_squared)
    """
    if len(closes) < lookback + 1:
        return np.nan, np.nan, np.nan
    recent = closes[-(lookback + 1):]
    y = np.log(recent)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    ann_ret = math.exp(slope * 250) - 1
    ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
    ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
    score = ann_ret * r2
    return score, ann_ret, r2


def rank_etfs(
    etf_closes: dict[str, np.ndarray],
    *,
    lookback_short: int = LOOKBACK_SHORT,
    lookback_long: int = LOOKBACK_LONG,
    min_r2: float = MIN_R2,
    short_weight: float = SHORT_WEIGHT,
    long_weight: float = LONG_WEIGHT,
) -> list[tuple[str, float, float, float]]:
    """
    对 ETF 候选池按组合动量得分排序。

    Parameters
    ----------
    etf_closes : {symbol: np.ndarray of close prices}

    Returns
    -------
    [(symbol, combo_score, r2_short, r2_long), ...] sorted by combo_score desc
    """
    results = []
    for sym, closes in etf_closes.items():
        if not isinstance(closes, np.ndarray):
            closes = np.array(closes, dtype=np.float64)

        valid = closes[np.isfinite(closes)]
        if len(valid) < lookback_long + 5:
            continue

        s_score, _, s_r2 = compute_momentum_score(valid, lookback_short)
        l_score, _, l_r2 = compute_momentum_score(valid, lookback_long)

        if np.isnan(s_score) or np.isnan(l_score):
            continue
        if max(s_r2, l_r2) < min_r2:
            continue

        combo = short_weight * s_score + long_weight * l_score
        results.append((sym, combo, s_r2, l_r2))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# 候选池定义
LARGECAP_ETF_POOL = {
    "510300.SH": "沪深300ETF",
    "510880.SH": "红利ETF",
    "512010.SH": "医药ETF",
    "512690.SH": "酒ETF",
    "515790.SH": "光伏ETF",
    "512800.SH": "银行ETF",
    "515030.SH": "新能源ETF",
    "512660.SH": "军工ETF",
    "512170.SH": "医疗ETF",
}

SMALLCAP_ETF_POOL = {
    "512100.SH": "中证1000ETF",
    "510500.SH": "中证500ETF",
    "159915.SZ": "创业板ETF",
}

ALLWEATHER_ETF_POOL = {
    "518880.SH": "黄金ETF",
}

DEFENSIVE_ETF = "511880.SH"  # 银华日利


def get_etf_pool(style_score: float, style_threshold: float = 0.2) -> list[str]:
    """根据风格信号返回候选 ETF 池."""
    if style_score < -style_threshold:
        pool = list(LARGECAP_ETF_POOL.keys()) + list(ALLWEATHER_ETF_POOL.keys())
    elif style_score > style_threshold:
        pool = list(SMALLCAP_ETF_POOL.keys()) + list(LARGECAP_ETF_POOL.keys())
    else:
        pool = (list(LARGECAP_ETF_POOL.keys()) +
                list(ALLWEATHER_ETF_POOL.keys()) +
                list(SMALLCAP_ETF_POOL.keys()))
    return pool
