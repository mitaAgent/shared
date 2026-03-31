"""
Regime 独立模块 — SSOT (Single Source of Truth)

所有看板和策略共享此模块的 Regime 分类、天气映射和市场温度。

版本:
  v1 (classifier.py) — 双维度 (波动率+风格), 生产看板仍在用
  v3 (classifier_v3.py) — 推荐版本, 四子信号方向判断, 11/12 已知段正确
  v2 (classifier_v2.py) — 已废弃, 仅保留供参考
"""
from .classifier import compute_timeline, compute_timeline_from_df, classify_current
from .classifier_v3 import compute_v3_timeline, compute_v3_timeline_from_mongo
from .style_signal import compute_style_signal, compute_style_signal_from_mongo
from .capital_router import compute_allocation
from .weather import WEATHER_MAP, REGIME_COLORS, get_weather
from .market_temp import compute_breadth
from .snapshot import get_market_snapshot

__all__ = [
    # v1 (看板兼容)
    "compute_timeline",
    "compute_timeline_from_df",
    "classify_current",
    # v3 (策略推荐)
    "compute_v3_timeline",
    "compute_v3_timeline_from_mongo",
    # 风格轮动
    "compute_style_signal",
    "compute_style_signal_from_mongo",
    # 资本路由
    "compute_allocation",
    # 辅助
    "WEATHER_MAP",
    "REGIME_COLORS",
    "get_weather",
    "compute_breadth",
    "get_market_snapshot",
]
