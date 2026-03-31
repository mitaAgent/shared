"""
Enhanced 7-signal direction scorer (v3.4)

Base: ECD+FG from FIX-26/27 (6 signals)
  tv_d: Adaptive Trend (MA10/MA30 in high-vol, MA20/MA60 otherwise)
  pv:   20d Price Position (>0.8 → +1, <0.2 → -1)
  dv:   Downvol Ratio (<0.45 → +1, >0.55 → -1)
  mv_c: Loosened V-reversal Momentum
  fv:   Directional Volume Surge (5d/20d vol > 1.5 + 3d return direction)
  gv:   North Flow Z-score (|z| > 1.5, 2014-11+)

New: Short-term reversal vote (sv) from FIX-28
  4 sub-votes collapsed to {-1, 0, +1}:
    ret_3d > 1% / < -1%
    ret_5d > 2% / < -2%
    ECD_FG score delta 5d > 0 / < 0
    vol_5d / vol_20d < 0.8 / > 1.2

Final score range: [-7, +7]  →  frac = (score + 7) / 14
"""
from __future__ import annotations
import math
import numpy as np
from pymongo import MongoClient


def _rolling_ma(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    cs = np.nancumsum(arr)
    cs = np.insert(cs, 0, 0.0)
    for i in range(window, n + 1):
        out[i - 1] = (cs[i] - cs[i - window]) / window
    return out


def _rolling_std(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = arr[i - window + 1:i + 1]
        valid = seg[np.isfinite(seg)]
        if len(valid) >= window // 2:
            out[i] = float(np.std(valid, ddof=1))
    return out


def compute_v34_scores_from_mongo(db, start="20100101"):
    """Compute enhanced 7-signal direction scores.

    Returns (dir_scores, regime_map) where:
        dir_scores: {date_int: score}  score ∈ [-7, +7]
        regime_map: {date_int: "进攻"/"防守"/"空仓"}
    """
    import pandas as pd

    # ── Load CSI300 close + volume ──
    docs = list(db["index_daily"].find(
        {"date": {"$gte": start, "$lte": "20270101"},
         "symbol": "000300.SH"},
        {"_id": 0, "date": 1, "close": 1, "volume": 1},
    ).sort("date", 1))
    df = pd.DataFrame(docs)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    dates = df["date"].astype(int).values
    csi = df["close"].values.astype(np.float64)
    vol_raw = df["volume"].values.astype(np.float64)
    n = len(csi)

    # ── Load north flow ──
    north_docs = list(db["north_moneyflow_hsgt"].find(
        {}, {"_id": 0, "trade_date": 1, "north_money": 1}
    ).sort("trade_date", 1))
    north_map = {}
    for r in north_docs:
        nm = r.get("north_money")
        if nm is not None:
            try:
                north_map[int(r["trade_date"])] = float(nm)
            except (ValueError, TypeError):
                pass

    # ── Derived arrays ──
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(csi[1:] / csi[:-1])

    # Volatility
    vol_20 = np.full(n, np.nan)
    vol_5d = np.full(n, np.nan)
    for i in range(5, n):
        seg = log_ret[max(1, i - 4):i + 1]
        v = seg[np.isfinite(seg)]
        if len(v) >= 3:
            vol_5d[i] = float(np.std(v, ddof=1)) * math.sqrt(252)
    for i in range(20, n):
        seg = log_ret[i - 19:i + 1]
        v = seg[np.isfinite(seg)]
        if len(v) >= 10:
            vol_20[i] = float(np.std(v, ddof=1)) * math.sqrt(252)

    vol_q25, vol_q75 = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(21, n):
        lb = max(0, i - 500)
        w = vol_20[lb:i]
        v = w[np.isfinite(w)]
        if len(v) > 4:
            vol_q25[i] = np.percentile(v, 25)
            vol_q75[i] = np.percentile(v, 75)

    # MAs
    ma10 = _rolling_ma(csi, 10)
    ma20 = _rolling_ma(csi, 20)
    ma30 = _rolling_ma(csi, 30)
    ma60 = _rolling_ma(csi, 60)

    def _slope(ma_arr, lag=5):
        out = np.full(n, np.nan)
        for i in range(lag, n):
            if np.isfinite(ma_arr[i]) and np.isfinite(ma_arr[i - lag]) and ma_arr[i - lag] > 0:
                out[i] = ma_arr[i] / ma_arr[i - lag] - 1.0
        return out

    slope_ma20 = _slope(ma20)
    slope_ma10 = _slope(ma10)

    # Returns
    mom60 = np.full(n, np.nan)
    ret20 = np.full(n, np.nan)
    ret10 = np.full(n, np.nan)
    ret5 = np.full(n, np.nan)
    ret3 = np.full(n, np.nan)
    for i in range(60, n):
        if csi[i - 60] > 0: mom60[i] = csi[i] / csi[i - 60] - 1.0
    for i in range(20, n):
        if csi[i - 20] > 0: ret20[i] = csi[i] / csi[i - 20] - 1.0
    for i in range(10, n):
        if csi[i - 10] > 0: ret10[i] = csi[i] / csi[i - 10] - 1.0
    for i in range(5, n):
        if csi[i - 5] > 0: ret5[i] = csi[i] / csi[i - 5] - 1.0
    for i in range(3, n):
        if csi[i - 3] > 0: ret3[i] = csi[i] / csi[i - 3] - 1.0

    # Price position
    high20 = np.full(n, np.nan)
    low20 = np.full(n, np.nan)
    for i in range(19, n):
        seg = csi[i - 19:i + 1]
        high20[i] = np.nanmax(seg)
        low20[i] = np.nanmin(seg)

    # Downvol
    downvol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        seg = log_ret[i - 19:i + 1]
        v = seg[np.isfinite(seg)]
        if len(v) >= 10:
            tv = float(np.var(v, ddof=1))
            dv_vals = v[v < 0]
            d_var = float(np.sum(dv_vals ** 2) / (len(v) - 1)) if len(v) > 1 else 0.0
            downvol_ratio[i] = d_var / tv if tv > 0 else 0.5

    # Volume MA
    vol_ma5 = _rolling_ma(vol_raw, 5)
    vol_ma20 = _rolling_ma(vol_raw, 20)

    # North flow aligned
    north_arr = np.full(n, np.nan)
    for i in range(n):
        v = north_map.get(int(dates[i]))
        if v is not None:
            north_arr[i] = v
    north_ma20 = _rolling_ma(north_arr, 20)
    north_std20 = _rolling_std(north_arr, 20)

    # ── Compute votes day-by-day ──
    start_idx = 65
    ecdfg_scores = np.full(n, np.nan)  # for score_delta computation
    dir_scores = {}
    regime_map = {}

    # First pass: compute ECD+FG (without sv)
    ecdfg_arr = np.full(n, 0.0)
    for i in range(start_idx, n):
        if not np.isfinite(vol_20[i]):
            continue
        vol = vol_20[i]
        q25 = vol_q25[i] if np.isfinite(vol_q25[i]) else 0.12
        q75 = vol_q75[i] if np.isfinite(vol_q75[i]) else 0.20
        vol_label = "低" if vol < q25 else ("高" if vol > q75 else "中")

        # tv_d: Adaptive Trend
        tv_d = 0
        if vol_label == "高":
            if np.isfinite(ma30[i]) and np.isfinite(ma10[i]) and np.isfinite(slope_ma10[i]):
                a30 = csi[i] > ma30[i]; m10_a30 = ma10[i] > ma30[i]
                if a30 and m10_a30 and slope_ma10[i] > 0.002: tv_d = 1
                elif (not a30) and (not m10_a30) and slope_ma10[i] < -0.002: tv_d = -1
        else:
            if np.isfinite(ma60[i]) and np.isfinite(ma20[i]) and np.isfinite(slope_ma20[i]):
                a60 = csi[i] > ma60[i]; m20_a60 = ma20[i] > ma60[i]
                if a60 and m20_a60 and slope_ma20[i] > 0.002: tv_d = 1
                elif (not a60) and (not m20_a60) and slope_ma20[i] < -0.002: tv_d = -1

        # pv: Price Position
        pv = 0
        if np.isfinite(high20[i]) and np.isfinite(low20[i]):
            rng = high20[i] - low20[i]
            if rng > 0:
                pos = (csi[i] - low20[i]) / rng
                if pos > 0.8: pv = 1
                elif pos < 0.2: pv = -1

        # dv: Downvol
        ddv = 0
        if np.isfinite(downvol_ratio[i]):
            dr = downvol_ratio[i]
            if dr < 0.45: ddv = 1
            elif dr > 0.55: ddv = -1

        # mv_c: Loosened V-reversal
        mv_c = 0
        if np.isfinite(mom60[i]):
            if mom60[i] > 0.03:
                mv_c = 1
            elif mom60[i] < -0.03:
                r20 = ret20[i] if np.isfinite(ret20[i]) else 0.0
                r10 = ret10[i] if np.isfinite(ret10[i]) else 0.0
                if r20 > 0.03 or r10 > 0.03: mv_c = 1
                else: mv_c = -1

        # fv: DirVol
        fv = 0
        if np.isfinite(vol_ma5[i]) and np.isfinite(vol_ma20[i]) and vol_ma20[i] > 0:
            vr = vol_ma5[i] / vol_ma20[i]
            r3v = ret3[i] if np.isfinite(ret3[i]) else 0.0
            if vr > 1.5 and r3v > 0.01: fv = 1
            elif vr > 1.5 and r3v < -0.01: fv = -1

        # gv: NorthZ
        gv = 0
        if (np.isfinite(north_arr[i]) and np.isfinite(north_ma20[i])
                and np.isfinite(north_std20[i]) and north_std20[i] > 0):
            z = (north_arr[i] - north_ma20[i]) / north_std20[i]
            if z > 1.5: gv = 1
            elif z < -1.5: gv = -1

        ecdfg_arr[i] = tv_d + pv + ddv + mv_c + fv + gv

    # Second pass: compute sv and final score
    for i in range(start_idx, n):
        if not np.isfinite(vol_20[i]):
            continue

        base = int(ecdfg_arr[i])

        # sv: Short-term reversal (4 sub-votes → ternary)
        sv_sub = 0
        r3 = ret3[i] if np.isfinite(ret3[i]) else 0.0
        r5 = ret5[i] if np.isfinite(ret5[i]) else 0.0
        if r3 > 0.01: sv_sub += 1
        elif r3 < -0.01: sv_sub -= 1
        if r5 > 0.02: sv_sub += 1
        elif r5 < -0.02: sv_sub -= 1

        delta_5d = ecdfg_arr[i] - ecdfg_arr[max(0, i - 5)]
        if delta_5d > 0: sv_sub += 1
        elif delta_5d < 0: sv_sub -= 1

        vc = vol_5d[i] / vol_20[i] if (np.isfinite(vol_5d[i]) and vol_20[i] > 0) else 1.0
        if vc < 0.8: sv_sub += 1
        elif vc > 1.2: sv_sub -= 1

        sv = 0
        if sv_sub >= 2: sv = 1
        elif sv_sub <= -2: sv = -1

        score = base + sv

        d = int(dates[i])
        dir_scores[d] = score

        if score >= 3:
            regime_map[d] = "进攻"
        elif score >= -1:
            regime_map[d] = "防守"
        else:
            regime_map[d] = "空仓"

    return dir_scores, regime_map
