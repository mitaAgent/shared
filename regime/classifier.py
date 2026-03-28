"""
R4b 双维度 Regime 分类器 — 生产 SSOT

维度 1 — 市场活跃度:
  沪深300 20日年化波动率，按过去 500 日滚动分位数分三档:
  < Q25 → 冷淡 / Q25~Q75 → 正常 / > Q75 → 活跃

维度 2 — 风格方向:
  创业板指/沪深300 比值 20日变化率:
  |变化率| ≤ DEAD_ZONE → 沿用前值 / > 0 → 成长 / < 0 → 价值

工程平滑:
  死区 (DEAD_ZONE=0.03): 避免比值微波导致风格频繁切换
  确认 (CONFIRM_DAYS=3): 新 regime 需连续 3 天才切换

来源: investment_strategy R4b 研究 + is2.0 生产实现
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

DEAD_ZONE = 0.03
CONFIRM_DAYS = 3


def compute_timeline(
    csi_close: np.ndarray,
    gem_close: np.ndarray,
    dates: np.ndarray,
    *,
    dead_zone: float = DEAD_ZONE,
    confirm_days: int = CONFIRM_DAYS,
) -> dict[int, str]:
    """
    从历史价格序列计算逐日 Regime 时间线。

    Args:
        csi_close: 沪深300 收盘价序列 (float, 按日期升序)
        gem_close: 创业板指 收盘价序列 (float, 按日期升序)
        dates:     日期序列 (int YYYYMMDD, 与价格对齐)
        dead_zone: 风格方向死区阈值
        confirm_days: 切换确认天数

    Returns:
        {date_int: regime_str}  如 {20260324: "活跃+成长"}
    """
    csi = np.asarray(csi_close, dtype=np.float64)
    gem = np.asarray(gem_close, dtype=np.float64)
    dates = np.asarray(dates)
    n = len(csi)

    log_ret = np.log(csi[1:] / csi[:-1])

    vol_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_20[i] = np.std(log_ret[i - 20:i], ddof=1) * math.sqrt(252)

    ratio = gem / csi
    gem_ratio_20 = np.full(n, np.nan)
    for i in range(20, n):
        gem_ratio_20[i] = ratio[i] / ratio[i - 20] - 1

    vol_q25 = np.full(n, np.nan)
    vol_q75 = np.full(n, np.nan)
    for i in range(21, n):
        lookback = max(0, i - 500)
        v_window = vol_20[lookback:i]
        valid = v_window[~np.isnan(v_window)]
        if len(valid) > 4:
            vol_q25[i] = np.percentile(valid, 25)
            vol_q75[i] = np.percentile(valid, 75)
        else:
            vol_q25[i] = 0.12
            vol_q75[i] = 0.20

    results: dict[int, str] = {}
    prev_regime = "正常+价值"
    prev_style = "价值"
    pending_regime = None
    pending_count = 0

    for i in range(21, n):
        if np.isnan(vol_20[i]) or np.isnan(gem_ratio_20[i]):
            continue

        v = vol_20[i]
        gr = gem_ratio_20[i]
        q25 = vol_q25[i] if not np.isnan(vol_q25[i]) else 0.12
        q75 = vol_q75[i] if not np.isnan(vol_q75[i]) else 0.20

        vol_label = "冷淡" if v < q25 else ("活跃" if v > q75 else "正常")

        if abs(gr) <= dead_zone:
            style_label = prev_style
        else:
            style_label = "成长" if gr > 0 else "价值"
        prev_style = style_label

        raw_regime = f"{vol_label}+{style_label}"

        if raw_regime != prev_regime:
            if pending_regime == raw_regime:
                pending_count += 1
            else:
                pending_regime = raw_regime
                pending_count = 1
            if pending_count >= confirm_days:
                prev_regime = raw_regime
                pending_regime = None
                pending_count = 0
        else:
            pending_regime = None
            pending_count = 0

        results[int(dates[i])] = prev_regime

    return results


def compute_timeline_from_df(idx_df, **kwargs) -> dict[int, str]:
    """
    DataFrame 包装器 (兼容 precompute_regime_scores.py 接口)。

    idx_df 需含列: csi_close, gem_close, date_int
    """
    return compute_timeline(
        idx_df["csi_close"].values,
        idx_df["gem_close"].values,
        idx_df["date_int"].values,
        **kwargs,
    )


def classify_current(mongo_db, *, confirm_days: int = CONFIRM_DAYS) -> dict:
    """
    从 MongoDB 拉取指数数据，实时计算当前 Regime + 辅助指标。

    Args:
        mongo_db: pymongo Database 对象 (e.g. client["panda"])
                  或具有 _db 属性的对象 (如 MarketDB 实例)
        confirm_days: Regime 切换需连续确认的天数。预计算/回测常用 3；
            看板展示「当日收盘后状态」时宜用 1（无多日平滑延迟）。

    Returns:
        dict with keys:
            regime, vol_label, style_label,
            vol_20, vol_pct,
            daily_ret: {hs300, gem},
            ret_20d: {hs300, gem},
            date
    """
    import pandas as pd

    if hasattr(mongo_db, "_db"):
        mongo_db = mongo_db._db

    end_s = datetime.now().strftime("%Y%m%d")
    start_s = (datetime.now() - timedelta(days=800)).strftime("%Y%m%d")

    docs = list(mongo_db["index_daily"].find(
        {"date": {"$gte": start_s, "$lte": end_s},
         "symbol": {"$in": ["000300.SH", "399006.SZ"]}},
        {"_id": 0, "symbol": 1, "date": 1, "close": 1},
    ))

    if not docs:
        return _fallback(end_s)

    df = pd.DataFrame(docs)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    csi = df[df["symbol"] == "000300.SH"].sort_values("date").reset_index(drop=True)
    gem = df[df["symbol"] == "399006.SZ"].sort_values("date").reset_index(drop=True)

    if len(csi) < 25 or len(gem) < 25:
        return _fallback(end_s)

    merged = (
        csi[["date", "close"]].rename(columns={"close": "csi"})
        .merge(gem[["date", "close"]].rename(columns={"close": "gem"}), on="date")
        .sort_values("date").reset_index(drop=True)
    )

    if len(merged) < 25:
        return _fallback(end_s)

    csi_arr = merged["csi"].values.astype(np.float64)
    gem_arr = merged["gem"].values.astype(np.float64)
    dates_int = merged["date"].apply(int).values

    timeline = compute_timeline(csi_arr, gem_arr, dates_int, confirm_days=confirm_days)
    n = len(merged)

    # --- 最新 regime ---
    if timeline:
        latest_key = max(timeline.keys())
        regime = timeline[latest_key]
    else:
        latest_key = int(dates_int[-1])
        regime = "未知"

    # --- 波动率 ---
    log_rets = np.diff(np.log(csi_arr[csi_arr > 0]))
    vol_20 = (
        float(np.std(log_rets[-20:], ddof=1) * math.sqrt(252))
        if len(log_rets) >= 20 else 0.0
    )

    vol_series = []
    for i in range(20, len(log_rets) + 1):
        vol_series.append(float(np.std(log_rets[i - 20:i], ddof=1) * math.sqrt(252)))
    lookback = vol_series[-500:] if len(vol_series) > 500 else vol_series
    vol_pct = (
        float(np.searchsorted(np.sort(lookback), vol_20) / max(len(lookback), 1) * 100)
        if lookback else 50.0
    )

    # --- 当日涨跌 ---
    hs300_daily = float(csi_arr[-1] / csi_arr[-2] - 1) * 100 if n >= 2 else 0.0
    gem_daily = float(gem_arr[-1] / gem_arr[-2] - 1) * 100 if n >= 2 else 0.0

    # --- 20 日累计收益 ---
    hs300_20d = float(csi_arr[-1] / csi_arr[-21] - 1) * 100 if n >= 21 else 0.0
    gem_20d = float(gem_arr[-1] / gem_arr[-21] - 1) * 100 if n >= 21 else 0.0

    parts = regime.split("+") if "+" in regime else [regime, "未知"]

    return {
        "regime": regime,
        "vol_label": parts[0],
        "style_label": parts[1],
        "vol_20": round(vol_20, 4),
        "vol_pct": round(vol_pct, 1),
        "daily_ret": {"hs300": round(hs300_daily, 2), "gem": round(gem_daily, 2)},
        "ret_20d": {"hs300": round(hs300_20d, 2), "gem": round(gem_20d, 2)},
        "date": str(latest_key),
    }


def _fallback(date_str: str) -> dict:
    return {
        "regime": "未知",
        "vol_label": "未知",
        "style_label": "未知",
        "vol_20": 0.0,
        "vol_pct": 50.0,
        "daily_ret": {"hs300": 0.0, "gem": 0.0},
        "ret_20d": {"hs300": 0.0, "gem": 0.0},
        "date": date_str,
    }
