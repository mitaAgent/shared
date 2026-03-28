"""
三维 Regime 分类器 v3 — 策略区推荐版本

输出三状态: 进攻 / 防守 / 空仓
输入: 沪深300 + 创业板指 + 可选 4 条辅助指数 (上证50/中证500/中证1000/上证综指)

架构: 波动率 x 方向 x 风格 → 三动作

  维度 1 — 波动率档位:
    20 日年化波动率在过去 500 日中的分位数
    < Q25 → 低 / Q25~Q75 → 中 / > Q75 → 高

  维度 2 — 市场方向 (四子信号投票, 得分 ∈ [-4, +4]):
    (a) 趋势: CSI300 close vs MA60 + MA20 斜率
    (b) 指数广度: 多指数站上各自 MA20 的比例 (≥80% 看多, ≤20% 看空)
    (c) 下行波动占比: 下行方差/总方差 (<0.45 看多, >0.55 看空)
    (d) 60 日动量 + V 型反转修正:
        - 60d > +3%  → +1
        - 60d < -3% 且 20d > +5% → +1 (V 型反转: 深跌后强反弹)
        - 60d < -3% 且 20d > +3% →  0 (温和反弹: 钝化)
        - 60d < -3% 其余           → -1
    得分 >= 2 → 上行 / <= -2 → 下行 / 其余 → 震荡

  维度 3 — 风格:
    创业板/沪深300 比值 20 日变化率, 死区 ±3%
    > 0 → 成长 / < 0 → 价值

  折叠为三动作:
    上行 → 进攻 | 震荡 → 防守 | 下行 + 低波 → 防守 | 下行 + 中/高波 → 空仓

  非对称确认 (防止频繁切换):
    升级 (空仓→进攻 3 天, 空仓→防守 2 天, 防守→进攻 2 天)
    降级 (进攻→防守/空仓 1 天, 防守→空仓 1 天)

验证 (2010-06 ~ 2026-03, 3776 天):
  12 个已知牛熊段: 11/12 正确
  单调性 (进攻 > 防守 > 空仓): 5d / 10d / 20d 全通过
  误报率 8.3%, 漏报率 7.8%
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

DEAD_ZONE = 0.03
TREND_MA_SHORT = 20
TREND_MA_LONG = 60
DOWNVOL_UP = 0.45
DOWNVOL_DOWN = 0.55
MOM60_UP = 0.03
MOM60_DOWN = -0.03
VOL_LOOKBACK = 500

ACTION_OFFENSE = "进攻"
ACTION_DEFENSE = "防守"
ACTION_CASH = "空仓"

CONFIRM_UPGRADE = {
    (ACTION_CASH, ACTION_OFFENSE): 3,
    (ACTION_CASH, ACTION_DEFENSE): 2,
    (ACTION_DEFENSE, ACTION_OFFENSE): 2,
}
CONFIRM_DOWNGRADE = {
    (ACTION_OFFENSE, ACTION_CASH): 1,
    (ACTION_OFFENSE, ACTION_DEFENSE): 1,
    (ACTION_DEFENSE, ACTION_CASH): 1,
}


def _get_confirm_days(old: str, new: str) -> int:
    if old == new:
        return 0
    pair = (old, new)
    return CONFIRM_UPGRADE.get(pair, CONFIRM_DOWNGRADE.get(pair, 2))


def _rolling_ma(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    cs = np.nancumsum(arr)
    cs = np.insert(cs, 0, 0.0)
    for i in range(window, n + 1):
        out[i - 1] = (cs[i] - cs[i - window]) / window
    return out


def compute_v3_timeline(
    csi_close: np.ndarray,
    gem_close: np.ndarray,
    dates: np.ndarray,
    extra_indices: Optional[dict] = None,
    *,
    dead_zone: float = DEAD_ZONE,
) -> dict:
    """
    Args:
        csi_close: 沪深300 收盘价 (按日期升序)
        gem_close: 创业板指 收盘价 (按日期升序)
        dates:     YYYYMMDD int
        extra_indices: {"000016.SH": np.ndarray, "000905.SH": ..., ...}
            与 dates 对齐的其他指数收盘价; None 则指数广度子信号仅用 CSI300+GEM

    Returns:
        {"timeline": {date_int: action}, "detail": {...}, "style": {...}}
    """
    csi = np.asarray(csi_close, dtype=np.float64)
    gem = np.asarray(gem_close, dtype=np.float64)
    dt = np.asarray(dates)
    n = len(csi)

    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(csi[1:] / csi[:-1])

    # --- Dim 1: Volatility ---
    vol_20 = np.full(n, np.nan)
    for i in range(20, n):
        seg = log_ret[i - 19:i + 1]
        valid = seg[np.isfinite(seg)]
        if len(valid) >= 10:
            vol_20[i] = float(np.std(valid, ddof=1)) * math.sqrt(252)

    vol_q25 = np.full(n, np.nan)
    vol_q75 = np.full(n, np.nan)
    for i in range(21, n):
        lb = max(0, i - VOL_LOOKBACK)
        window = vol_20[lb:i]
        valid = window[np.isfinite(window)]
        if len(valid) > 4:
            vol_q25[i] = np.percentile(valid, 25)
            vol_q75[i] = np.percentile(valid, 75)

    # --- Sub-signals for Dim 2: Direction ---

    # (a) Trend: CSI300 vs MA60 + MA20 slope
    ma20_csi = _rolling_ma(csi, TREND_MA_SHORT)
    ma60_csi = _rolling_ma(csi, TREND_MA_LONG)
    ma20_slope = np.full(n, np.nan)
    for i in range(5, n):
        if np.isfinite(ma20_csi[i]) and np.isfinite(ma20_csi[i - 5]) and ma20_csi[i - 5] > 0:
            ma20_slope[i] = ma20_csi[i] / ma20_csi[i - 5] - 1.0

    # (b) Index breadth: 多指数站上各自 MA20 的比例
    all_index_series = [csi, gem]
    if extra_indices:
        for sym in sorted(extra_indices.keys()):
            arr = np.asarray(extra_indices[sym], dtype=np.float64)
            if len(arr) == n:
                all_index_series.append(arr)

    index_ma20s = [_rolling_ma(s, TREND_MA_SHORT) for s in all_index_series]
    n_indices = len(all_index_series)

    index_breadth = np.full(n, np.nan)
    for i in range(TREND_MA_SHORT, n):
        above = 0
        total = 0
        for k in range(n_indices):
            if np.isfinite(all_index_series[k][i]) and np.isfinite(index_ma20s[k][i]):
                total += 1
                if all_index_series[k][i] > index_ma20s[k][i]:
                    above += 1
        if total >= 2:
            index_breadth[i] = above / total * 100.0

    # (c) Downside vol ratio
    downvol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        seg = log_ret[i - 19:i + 1]
        valid = seg[np.isfinite(seg)]
        if len(valid) >= 10:
            total_var = float(np.var(valid, ddof=1))
            down = valid[valid < 0]
            down_var = float(np.sum(down ** 2) / (len(valid) - 1)) if len(valid) > 1 else 0.0
            downvol_ratio[i] = down_var / total_var if total_var > 0 else 0.5

    # (d) 60-day momentum + 20-day for conditional dampening
    mom_60 = np.full(n, np.nan)
    for i in range(60, n):
        if csi[i - 60] > 0:
            mom_60[i] = csi[i] / csi[i - 60] - 1.0
    ret_20 = np.full(n, np.nan)
    for i in range(20, n):
        if csi[i - 20] > 0:
            ret_20[i] = csi[i] / csi[i - 20] - 1.0

    # --- Dim 3: Style ---
    ratio = gem / csi
    gem_ratio_20 = np.full(n, np.nan)
    for i in range(20, n):
        if ratio[i - 20] > 0:
            gem_ratio_20[i] = ratio[i] / ratio[i - 20] - 1.0

    # --- Day-by-day classification ---
    timeline: dict[int, str] = {}
    detail: dict[int, dict] = {}
    style_map: dict[int, str] = {}

    prev_action = ACTION_DEFENSE
    prev_style = "价值"
    pending_action: Optional[str] = None
    pending_count = 0

    start_idx = TREND_MA_LONG + 5

    for i in range(start_idx, n):
        if not np.isfinite(vol_20[i]):
            continue

        v = vol_20[i]
        q25 = vol_q25[i] if np.isfinite(vol_q25[i]) else 0.12
        q75 = vol_q75[i] if np.isfinite(vol_q75[i]) else 0.20
        vol_label = "低" if v < q25 else ("高" if v > q75 else "中")

        # (a) Trend vote
        trend_vote = 0
        if np.isfinite(ma60_csi[i]) and np.isfinite(ma20_csi[i]) and np.isfinite(ma20_slope[i]):
            above_ma60 = csi[i] > ma60_csi[i]
            ma20_above_ma60 = ma20_csi[i] > ma60_csi[i]
            slope_pos = ma20_slope[i] > 0.002
            slope_neg = ma20_slope[i] < -0.002
            if above_ma60 and ma20_above_ma60 and slope_pos:
                trend_vote = 1
            elif (not above_ma60) and (not ma20_above_ma60) and slope_neg:
                trend_vote = -1

        # (b) Index breadth vote
        breadth_vote = 0
        if np.isfinite(index_breadth[i]):
            b = index_breadth[i]
            if b >= 80.0:
                breadth_vote = 1
            elif b <= 20.0:
                breadth_vote = -1

        # (c) Downside vol vote
        downvol_vote = 0
        if np.isfinite(downvol_ratio[i]):
            dv = downvol_ratio[i]
            if dv < DOWNVOL_UP:
                downvol_vote = 1
            elif dv > DOWNVOL_DOWN:
                downvol_vote = -1

        # (d) 60-day momentum vote (with reversal override)
        mom_vote = 0
        if np.isfinite(mom_60[i]):
            if mom_60[i] > MOM60_UP:
                mom_vote = 1
            elif mom_60[i] < MOM60_DOWN:
                r20 = ret_20[i] if np.isfinite(ret_20[i]) else 0.0
                if r20 > 0.05:
                    mom_vote = 1   # V型反转: 深跌+短期强反弹 → 翻多
                elif r20 > 0.03:
                    mom_vote = 0   # 深跌后温和反弹 → 钝化为中性
                else:
                    mom_vote = -1

        direction_score = trend_vote + breadth_vote + downvol_vote + mom_vote

        if direction_score >= 2:
            direction = "上行"
        elif direction_score <= -2:
            direction = "下行"
        else:
            direction = "震荡"

        # 3-dim → 3-action fold
        if direction == "上行":
            raw_action = ACTION_OFFENSE
        elif direction == "震荡":
            raw_action = ACTION_DEFENSE
        else:
            if vol_label == "低":
                raw_action = ACTION_DEFENSE
            else:
                raw_action = ACTION_CASH

        # Asymmetric confirmation
        if raw_action != prev_action:
            need = _get_confirm_days(prev_action, raw_action)
            if pending_action == raw_action:
                pending_count += 1
            else:
                pending_action = raw_action
                pending_count = 1
            if pending_count >= need:
                prev_action = raw_action
                pending_action = None
                pending_count = 0
        else:
            pending_action = None
            pending_count = 0

        # Style (only matters within offense)
        if np.isfinite(gem_ratio_20[i]):
            gr = gem_ratio_20[i]
            if abs(gr) <= dead_zone:
                style_label = prev_style
            else:
                style_label = "成长" if gr > 0 else "价值"
            prev_style = style_label
        else:
            style_label = prev_style

        d = int(dt[i])
        timeline[d] = prev_action
        style_map[d] = style_label
        detail[d] = {
            "vol_20": round(v, 4),
            "vol_label": vol_label,
            "direction": direction,
            "trend_vote": trend_vote,
            "breadth_vote": breadth_vote,
            "downvol_vote": downvol_vote,
            "mom_vote": mom_vote,
            "direction_score": direction_score,
            "raw_action": raw_action,
            "confirmed_action": prev_action,
            "style": style_label,
        }

    return {"timeline": timeline, "detail": detail, "style": style_map}


def compute_v3_timeline_from_mongo(
    mongo_db,
    start: str = "20060101",
    end: str = "20260401",
    *,
    dead_zone: float = DEAD_ZONE,
) -> dict:
    """从 MongoDB 加载全部指数数据，计算 v3 时间线。"""
    import pandas as pd

    if hasattr(mongo_db, "_db"):
        mongo_db = mongo_db._db

    symbols = ["000300.SH", "399006.SZ", "000016.SH", "000905.SH",
               "000852.SH", "000001.SH"]
    docs = list(mongo_db["index_daily"].find(
        {"date": {"$gte": start, "$lte": end},
         "symbol": {"$in": symbols}},
        {"_id": 0, "symbol": 1, "date": 1, "close": 1},
    ))
    df = pd.DataFrame(docs)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    pivoted = df.pivot(index="date", columns="symbol", values="close").sort_index()
    pivoted = pivoted.dropna(subset=["000300.SH", "399006.SZ"])

    dates_int = pivoted.index.astype(int).values
    csi = pivoted["000300.SH"].values.astype(np.float64)
    gem = pivoted["399006.SZ"].values.astype(np.float64)

    extra = {}
    for sym in ["000016.SH", "000905.SH", "000852.SH", "000001.SH"]:
        if sym in pivoted.columns:
            arr = pivoted[sym].values.astype(np.float64)
            arr = np.where(np.isfinite(arr), arr, np.nan)
            extra[sym] = arr

    return compute_v3_timeline(csi, gem, dates_int, extra_indices=extra,
                               dead_zone=dead_zone)
