"""
6 种 Regime → 天气映射 + 看板配色

天气仅做"纯描述"，不含行动建议。
"""
from __future__ import annotations

WEATHER_MAP: dict[str, dict] = {
    "活跃+成长": {"emoji": "🔥", "label": "烈日", "desc": "市场活跃，成长风格占优"},
    "活跃+价值": {"emoji": "☀️", "label": "晴",   "desc": "市场活跃，价值风格占优"},
    "正常+成长": {"emoji": "⛅", "label": "多云转晴", "desc": "市场正常，成长略强"},
    "正常+价值": {"emoji": "☁️", "label": "多云",  "desc": "市场正常，价值略强"},
    "冷淡+成长": {"emoji": "🌥️", "label": "阴天",  "desc": "市场冷淡，成长略强"},
    "冷淡+价值": {"emoji": "🌧️", "label": "阴雨",  "desc": "市场冷淡，价值略强"},
}

REGIME_COLORS: dict[str, str] = {
    "活跃+成长": "#22c55e",
    "活跃+价值": "#16a34a",
    "正常+成长": "#60a5fa",
    "正常+价值": "#3b82f6",
    "冷淡+成长": "#f59e0b",
    "冷淡+价值": "#d97706",
}

_FALLBACK = {"emoji": "❓", "label": "未知", "desc": "数据不足"}


def get_weather(regime: str) -> dict:
    """返回 regime 对应的天气信息 {emoji, label, desc}"""
    return WEATHER_MAP.get(regime, _FALLBACK).copy()
