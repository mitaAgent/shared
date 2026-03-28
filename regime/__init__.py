"""
Regime 独立模块 — SSOT (Single Source of Truth)

所有看板和策略共享此模块的 Regime 分类、天气映射和市场温度。
"""
from .classifier import compute_timeline, compute_timeline_from_df, classify_current
from .weather import WEATHER_MAP, REGIME_COLORS, get_weather
from .market_temp import compute_breadth
from .snapshot import get_market_snapshot

__all__ = [
    "compute_timeline",
    "compute_timeline_from_df",
    "classify_current",
    "WEATHER_MAP",
    "REGIME_COLORS",
    "get_weather",
    "compute_breadth",
    "get_market_snapshot",
]
