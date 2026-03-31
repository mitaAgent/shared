"""
风格轮动信号 — 大盘 vs 小盘相对强弱

输出 style_score ∈ [-1, +1]:
  +1  = 小盘极端占优 (适合 VWAP+VCV+Amihud+Chip 小盘策略)
  -1  = 大盘极端占优 (适合大盘因子策略 / ETF 动量)
   0  = 均衡 / 过渡区

信号构造 (三票加权):
  (a) 短期动量: CSI1000/CSI300 比值 20 日变化率
  (b) 中期趋势: 比值 vs 60 日均线偏离度
  (c) 长期分位: 比值 60 日动量在 250 日窗口内的分位数

三票各独立映射到 [-1, +1], 等权加总后 clip 到 [-1, +1]。
内置惯性 (hysteresis): 只有变化超过阈值才更新, 防止日频抖动。
"""
from __future__ import annotations

import numpy as np

SHORT_WINDOW = 20
TREND_MA = 60
QUANTILE_LOOKBACK = 250
HYSTERESIS = 0.12


def _ma(arr: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均, 不足窗口填 nan."""
    n = len(arr)
    out = np.full(n, np.nan)
    cumsum = np.nancumsum(arr)
    out[window - 1:] = (cumsum[window - 1:] - np.concatenate([[0], cumsum[:-window]])) / window
    return out


def _pct_change(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    valid = (arr[period:] > 0) & (arr[:-period] > 0)
    out[period:] = np.where(valid, arr[period:] / arr[:-period] - 1, np.nan)
    return out


def compute_style_signal(
    csi300: np.ndarray,
    csi1000: np.ndarray,
    dates: np.ndarray,
    *,
    short_window: int = SHORT_WINDOW,
    trend_ma: int = TREND_MA,
    quantile_lookback: int = QUANTILE_LOOKBACK,
    hysteresis: float = HYSTERESIS,
) -> dict:
    """
    计算风格轮动信号。

    Parameters
    ----------
    csi300 : 沪深300 收盘价序列
    csi1000 : 中证1000 收盘价序列
    dates : 日期数组 (int, YYYYMMDD)
    short_window : 短期动量窗口 (默认20日)
    trend_ma : 趋势均线窗口 (默认60日)
    quantile_lookback : 分位数回看窗口 (默认250日)
    hysteresis : 惯性阈值 (默认0.15)

    Returns
    -------
    dict with keys:
      "dates": int array
      "style_scores": float array, 每日 style_score ∈ [-1, +1]
      "raw_scores": float array, 惯性前的原始得分
      "ratio": float array, CSI1000/CSI300 比值
      "detail": dict[int, dict], 按日详情
    """
    n = len(csi300)
    assert len(csi1000) == n and len(dates) == n

    ratio = np.where((csi300 > 0) & (csi1000 > 0), csi1000 / csi300, np.nan)

    # (a) 短期动量: 比值 20 日变化率 → 映射到 [-1, +1]
    short_mom = _pct_change(ratio, short_window)
    vote_a = np.full(n, 0.0)
    for i in range(n):
        if np.isfinite(short_mom[i]):
            vote_a[i] = np.clip(short_mom[i] / 0.08, -1, 1)

    # (b) 趋势偏离: 比值 vs 60 日均线 → 映射到 [-1, +1]
    ratio_ma = _ma(ratio, trend_ma)
    vote_b = np.full(n, 0.0)
    for i in range(n):
        if np.isfinite(ratio[i]) and np.isfinite(ratio_ma[i]) and ratio_ma[i] > 0:
            deviation = ratio[i] / ratio_ma[i] - 1
            vote_b[i] = np.clip(deviation / 0.05, -1, 1)

    # (c) 长期分位: 60 日动量在 250 日窗口内的分位 → 映射到 [-1, +1]
    mom60 = _pct_change(ratio, trend_ma)
    vote_c = np.full(n, 0.0)
    for i in range(quantile_lookback + trend_ma, n):
        window = mom60[max(0, i - quantile_lookback):i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) >= 60:
            current = mom60[i]
            if np.isfinite(current):
                quantile = np.mean(valid <= current)
                vote_c[i] = (quantile - 0.5) * 2  # [0,1] → [-1,+1]

    # 等权加总 → raw_score ∈ [-1, +1]
    raw_scores = np.clip((vote_a + vote_b + vote_c) / 3, -1, 1)

    # 惯性 (hysteresis): 只有变化超过阈值才更新
    style_scores = np.full(n, 0.0)
    prev = 0.0
    for i in range(n):
        if not np.isfinite(raw_scores[i]):
            style_scores[i] = prev
        elif abs(raw_scores[i] - prev) > hysteresis:
            style_scores[i] = raw_scores[i]
            prev = raw_scores[i]
        else:
            style_scores[i] = prev

    detail = {}
    for i in range(n):
        d = int(dates[i])
        detail[d] = {
            "style_score": float(style_scores[i]),
            "raw_score": float(raw_scores[i]),
            "vote_a_short_mom": float(vote_a[i]),
            "vote_b_trend_dev": float(vote_b[i]),
            "vote_c_quantile": float(vote_c[i]),
            "ratio": float(ratio[i]) if np.isfinite(ratio[i]) else None,
        }

    return {
        "dates": dates,
        "style_scores": style_scores,
        "raw_scores": raw_scores,
        "ratio": ratio,
        "detail": detail,
    }


def compute_style_signal_from_mongo(
    mongo_db,
    start: str = "20060101",
    end: str = "20260401",
    **kwargs,
) -> dict:
    """从 MongoDB 加载 CSI300 + CSI1000, 计算风格轮动信号."""
    import pandas as pd

    if hasattr(mongo_db, "_db"):
        mongo_db = mongo_db._db

    symbols = ["000300.SH", "000852.SH"]
    docs = list(mongo_db["index_daily"].find(
        {"date": {"$gte": start, "$lte": end},
         "symbol": {"$in": symbols}},
        {"_id": 0, "symbol": 1, "date": 1, "close": 1},
    ))

    df = pd.DataFrame(docs)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    pivoted = df.pivot(index="date", columns="symbol", values="close").sort_index()
    pivoted = pivoted.dropna(subset=["000300.SH", "000852.SH"])

    dates_int = pivoted.index.astype(int).values
    csi300 = pivoted["000300.SH"].values.astype(np.float64)
    csi1000 = pivoted["000852.SH"].values.astype(np.float64)

    return compute_style_signal(csi300, csi1000, dates_int, **kwargs)
