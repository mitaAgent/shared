"""
[已废弃] 三维 Regime 分类器 v2 — 请使用 classifier_v3.py

v2 的已知问题: breadth 依赖 stock_market 全 A 数据，2020 年前覆盖率仅 39%。
v3 已用多指数一致性替代，并新增 60 日动量信号，全量覆盖 2010~2026。

原始说明:
三维 Regime 分类器 v2 — 增加「市场方向」维度

v1 的缺陷: 只用波动率 + 风格两个维度，无法区分「高波动牛市」和「高波动熊市」，
  导致 2018 阴跌、2022 急跌、2015 股灾下半场等被标为"活跃" → 策略触发进攻 → 亏损。

v2 新增维度 2 — 市场方向:
  三个子信号多数投票:
    (a) 趋势信号: CSI300 close vs MA60, MA20 斜率
    (b) 市场广度: 全 A 站上 MA20 比例 (需 stock_market 数据)
    (c) 下行波动率占比: downside_vol / total_vol

三维 → 三态折叠映射:
  方向=上行 → 进攻 (风格维度决定用哪套打分)
  方向=震荡 → 防守
  方向=下行 + 高/中波动 → 空仓
  方向=下行 + 低波动 → 防守 (阴跌尾声，跌不动了)

确认机制: 非对称 — 降档快(1天)，升档慢(2~3天)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

DEAD_ZONE = 0.03
TREND_MA_SHORT = 20
TREND_MA_LONG = 60
BREADTH_UP = 55.0
BREADTH_DOWN = 35.0
DOWNVOL_UP = 0.45
DOWNVOL_DOWN = 0.55
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
_ACTION_RANK = {ACTION_CASH: 0, ACTION_DEFENSE: 1, ACTION_OFFENSE: 2}


def _get_confirm_days(old: str, new: str) -> int:
    if old == new:
        return 0
    pair = (old, new)
    if pair in CONFIRM_UPGRADE:
        return CONFIRM_UPGRADE[pair]
    if pair in CONFIRM_DOWNGRADE:
        return CONFIRM_DOWNGRADE[pair]
    return 2


def _rolling_ma(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    cs = np.nancumsum(arr)
    cs = np.insert(cs, 0, 0.0)
    for i in range(window, len(arr) + 1):
        out[i - 1] = (cs[i] - cs[i - window]) / window
    return out


def compute_v2_timeline(
    csi_close: np.ndarray,
    gem_close: np.ndarray,
    dates: np.ndarray,
    breadth_series: Optional[np.ndarray] = None,
    *,
    dead_zone: float = DEAD_ZONE,
) -> dict:
    """
    计算增强版三态 Regime 时间线。

    Args:
        csi_close: 沪深300 收盘价 (按日期升序)
        gem_close: 创业板指 收盘价 (按日期升序)
        dates:     YYYYMMDD int 日期序列
        breadth_series: 全 A 站上 MA20 比例 (0~100), 与 dates 对齐;
                        None 则该子信号缺席，由其余两票决定

    Returns:
        {
            "timeline": {date_int: action_str},  # 进攻/防守/空仓
            "detail": {date_int: {...}},          # 中间变量，供验证
            "style": {date_int: str},             # 成长/价值 (进攻态子类)
        }
    """
    csi = np.asarray(csi_close, dtype=np.float64)
    gem = np.asarray(gem_close, dtype=np.float64)
    dt = np.asarray(dates)
    n = len(csi)
    assert n == len(gem) == len(dt)

    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(csi[1:] / csi[:-1])

    # --- 维度 1: 波动水平 ---
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

    # --- 维度 2: 市场方向 (三子信号投票) ---

    ma20 = _rolling_ma(csi, TREND_MA_SHORT)
    ma60 = _rolling_ma(csi, TREND_MA_LONG)
    ma20_slope = np.full(n, np.nan)
    for i in range(5, n):
        if np.isfinite(ma20[i]) and np.isfinite(ma20[i - 5]) and ma20[i - 5] > 0:
            ma20_slope[i] = ma20[i] / ma20[i - 5] - 1.0

    downvol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        seg = log_ret[i - 19:i + 1]
        valid = seg[np.isfinite(seg)]
        if len(valid) >= 10:
            total_var = float(np.var(valid, ddof=1))
            down = valid[valid < 0]
            down_var = float(np.sum(down ** 2) / (len(valid) - 1)) if len(valid) > 1 else 0.0
            downvol_ratio[i] = down_var / total_var if total_var > 0 else 0.5

    has_breadth = breadth_series is not None and len(breadth_series) == n

    # --- 维度 3: 风格方向 (保留 v1 逻辑) ---
    ratio = gem / csi
    gem_ratio_20 = np.full(n, np.nan)
    for i in range(20, n):
        if ratio[i - 20] > 0:
            gem_ratio_20[i] = ratio[i] / ratio[i - 20] - 1.0

    # --- 逐日分类 ---
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

        # 子信号 A: 趋势
        trend_vote = 0
        if np.isfinite(ma60[i]) and np.isfinite(ma20[i]) and np.isfinite(ma20_slope[i]):
            above_ma60 = csi[i] > ma60[i]
            ma20_above_ma60 = ma20[i] > ma60[i]
            slope_pos = ma20_slope[i] > 0.002
            slope_neg = ma20_slope[i] < -0.002
            if above_ma60 and ma20_above_ma60 and slope_pos:
                trend_vote = 1
            elif (not above_ma60) and (not ma20_above_ma60) and slope_neg:
                trend_vote = -1

        # 子信号 B: 广度
        breadth_vote = 0
        if has_breadth and np.isfinite(breadth_series[i]):
            b = breadth_series[i]
            if b > BREADTH_UP:
                breadth_vote = 1
            elif b < BREADTH_DOWN:
                breadth_vote = -1

        # 子信号 C: 下行波动占比
        downvol_vote = 0
        if np.isfinite(downvol_ratio[i]):
            dv = downvol_ratio[i]
            if dv < DOWNVOL_UP:
                downvol_vote = 1
            elif dv > DOWNVOL_DOWN:
                downvol_vote = -1

        direction_score = trend_vote + breadth_vote + downvol_vote
        n_voters = (1 if trend_vote != 0 else 0) + \
                   (1 if breadth_vote != 0 else 0) + \
                   (1 if downvol_vote != 0 else 0)

        if direction_score >= 2:
            direction = "上行"
        elif direction_score <= -2:
            direction = "下行"
        elif direction_score == 1 and n_voters >= 2:
            direction = "上行"
        elif direction_score == -1 and n_voters >= 2:
            direction = "下行"
        else:
            direction = "震荡"

        # 三维 → 三态折叠
        if direction == "上行":
            raw_action = ACTION_OFFENSE
        elif direction == "震荡":
            raw_action = ACTION_DEFENSE
        else:  # 下行
            if vol_label == "低":
                raw_action = ACTION_DEFENSE
            else:
                raw_action = ACTION_CASH

        # 非对称确认
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

        # 风格维度 (仅进攻态内部使用)
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
            "direction_score": direction_score,
            "raw_action": raw_action,
            "confirmed_action": prev_action,
            "style": style_label,
        }

    return {"timeline": timeline, "detail": detail, "style": style_map}


def compute_v2_timeline_from_mongo(
    mongo_db,
    start: str = "20060101",
    end: str = "20260401",
    *,
    dead_zone: float = DEAD_ZONE,
    with_breadth: bool = True,
) -> dict:
    """
    从 MongoDB 加载指数 + stock_market 数据，计算完整 v2 时间线。

    Args:
        mongo_db: pymongo Database
        start, end: YYYYMMDD
        with_breadth: 是否计算全 A MA20 breadth（较慢，约 1~2 分钟）
    """
    if hasattr(mongo_db, "_db"):
        mongo_db = mongo_db._db

    # 加载指数
    docs = list(mongo_db["index_daily"].find(
        {"date": {"$gte": start, "$lte": end},
         "symbol": {"$in": ["000300.SH", "399006.SZ"]}},
        {"_id": 0, "symbol": 1, "date": 1, "close": 1},
    ))
    df = pd.DataFrame(docs)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    csi = df[df["symbol"] == "000300.SH"].sort_values("date").reset_index(drop=True)
    gem = df[df["symbol"] == "399006.SZ"].sort_values("date").reset_index(drop=True)
    merged = (
        csi[["date", "close"]].rename(columns={"close": "csi"})
        .merge(gem[["date", "close"]].rename(columns={"close": "gem"}), on="date")
        .sort_values("date").reset_index(drop=True)
    )

    breadth_arr = None
    if with_breadth:
        breadth_arr = _compute_breadth_series(mongo_db, merged["date"].tolist())

    return compute_v2_timeline(
        merged["csi"].values,
        merged["gem"].values,
        merged["date"].astype(int).values,
        breadth_series=breadth_arr,
        dead_zone=dead_zone,
    )


def _compute_breadth_series(mongo_db, dates: list[str]) -> np.ndarray:
    """
    批量计算全 A MA20 breadth 时间序列。
    为性能考虑，一次性加载全量 stock_market 数据做 pivot。
    """
    print("[v2 classifier] 计算全 A MA20 breadth（首次较慢）...", flush=True)

    min_d = min(dates)
    lookback_d = str(int(min_d) - 500)  # 粗略 500 天余量供 MA20 热身

    docs = list(mongo_db["stock_market"].find(
        {"date": {"$gte": lookback_d, "$lte": max(dates)}},
        {"_id": 0, "symbol": 1, "date": 1, "close": 1},
    ).batch_size(500000))

    if not docs:
        print("  breadth: 无 stock_market 数据，跳过", flush=True)
        return None

    sm = pd.DataFrame(docs)
    sm["close"] = pd.to_numeric(sm["close"], errors="coerce")

    # 排除指数（只保留股票）
    sm = sm[sm["symbol"].str.match(r"^[036]\d{5}\.\w{2}$", na=False)]

    pv = sm.pivot(index="date", columns="symbol", values="close").sort_index()
    ma20 = pv.rolling(20, min_periods=15).mean()

    above = (pv > ma20).sum(axis=1)
    total = pv.notna().sum(axis=1)
    breadth_pct = (above / total * 100).where(total > 100)

    breadth_by_date = breadth_pct.to_dict()

    result = np.full(len(dates), np.nan)
    for i, d in enumerate(dates):
        result[i] = breadth_by_date.get(d, np.nan)

    valid_count = np.sum(np.isfinite(result))
    print(f"  breadth: {valid_count}/{len(dates)} 天有效", flush=True)
    return result
