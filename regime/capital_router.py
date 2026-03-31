"""
资本路由器 — 风格自适应资本配置

基于风格信号 (style_score) + Regime v3 (direction_score) 决定资本去向:

  小盘进攻:  style_score > 0   && direction >= 2  →  100% 小盘股策略
  小盘减仓:  style_score > 0   && 0 <= direction < 2  →  frac 小盘 + (1-frac) ETF
  大盘进攻:  style_score < -th && direction >= 0  →  ETF 动量 (+ 可选大盘辅助)
  大盘防守:  style_score < -th && direction < 0   →  防御 (货币ETF/低仓位)
  过渡区:    |style_score| <= th                   →  部分小盘 + 部分ETF (按比例)

输出 allocation dict:
  {"smallcap_frac": 0~1, "etf_frac": 0~1, "largecap_frac": 0~1, "defensive_frac": 0~1}
  四个 frac 之和 = 1.0
"""
from __future__ import annotations

STYLE_THRESHOLD = 0.2
LARGECAP_AUX_WEIGHT = 0.2  # 大盘进攻时, 给大盘因子策略的辅助权重


def compute_allocation(
    style_score: float,
    direction_score: int,
    *,
    style_threshold: float = STYLE_THRESHOLD,
    use_largecap_aux: bool = True,
) -> dict:
    """
    根据风格信号和 Regime 方向得分计算资本分配。

    Parameters
    ----------
    style_score : float, [-1, +1], 正=小盘占优
    direction_score : int, [-4, +4], 正=市场看多
    style_threshold : float, 风格判定阈值
    use_largecap_aux : bool, 是否在大盘占优时分配部分仓位给大盘因子策略

    Returns
    -------
    dict with keys: smallcap_frac, etf_frac, largecap_frac, defensive_frac
    """
    sc = float(style_score)
    ds = int(direction_score)

    if sc > style_threshold:
        # 小盘风格占优
        if ds >= 2:
            # 小盘进攻
            frac = min(1.0, (ds + 4) / 8)
            return {"smallcap_frac": frac, "etf_frac": 1 - frac,
                    "largecap_frac": 0.0, "defensive_frac": 0.0}
        elif ds >= 0:
            # 小盘温和
            sc_strength = min(1.0, (sc - style_threshold) / (1 - style_threshold))
            sm_frac = 0.5 * sc_strength
            etf_frac = 0.3
            def_frac = 1.0 - sm_frac - etf_frac
            return {"smallcap_frac": sm_frac, "etf_frac": etf_frac,
                    "largecap_frac": 0.0, "defensive_frac": max(0, def_frac)}
        else:
            # 小盘风格但市场弱
            def_frac = min(1.0, (-ds) / 4)
            return {"smallcap_frac": 0.0, "etf_frac": 1 - def_frac,
                    "largecap_frac": 0.0, "defensive_frac": def_frac}

    elif sc < -style_threshold:
        # 大盘风格占优
        lc_strength = min(1.0, (-sc - style_threshold) / (1 - style_threshold))
        if ds >= 1:
            # 大盘进攻
            if use_largecap_aux:
                lc = LARGECAP_AUX_WEIGHT * lc_strength
                etf = 1.0 - lc
            else:
                lc = 0.0
                etf = 1.0
            return {"smallcap_frac": 0.0, "etf_frac": etf,
                    "largecap_frac": lc, "defensive_frac": 0.0}
        elif ds >= -1:
            # 大盘温和
            etf = 0.6 * lc_strength
            def_frac = 1.0 - etf
            return {"smallcap_frac": 0.0, "etf_frac": etf,
                    "largecap_frac": 0.0, "defensive_frac": def_frac}
        else:
            # 大盘防守
            def_frac = min(1.0, (-ds) / 3)
            etf = (1.0 - def_frac) * 0.4
            return {"smallcap_frac": 0.0, "etf_frac": etf,
                    "largecap_frac": 0.0, "defensive_frac": 1.0 - etf}

    else:
        # 过渡区: 按 style_score 线性分配
        sm_w = max(0, (sc + style_threshold) / (2 * style_threshold))  # [0, 1]
        etf_w = 1 - sm_w

        if ds >= 2:
            return {"smallcap_frac": sm_w * 0.8, "etf_frac": etf_w * 0.8,
                    "largecap_frac": 0.0, "defensive_frac": 0.2}
        elif ds >= 0:
            return {"smallcap_frac": sm_w * 0.5, "etf_frac": etf_w * 0.5,
                    "largecap_frac": 0.0, "defensive_frac": 0.5}
        else:
            def_frac = min(0.8, (-ds) / 4)
            rem = 1 - def_frac
            return {"smallcap_frac": sm_w * rem * 0.3, "etf_frac": etf_w * rem * 0.7,
                    "largecap_frac": 0.0, "defensive_frac": def_frac}
