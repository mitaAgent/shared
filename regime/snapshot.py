"""
统一市场快照 API — 一次调用返回所有看板需要的实时市场状态。

预留 mode 参数：
  mode="current"   当前实时状态 (默认)
  mode="predicted"  未来: 接入预测模型
"""
from __future__ import annotations

import json
from pathlib import Path

from .classifier import classify_current
from .weather import get_weather
from .market_temp import compute_breadth

_DEFAULT_HISTORY = Path(__file__).resolve().parent / ".cache" / "style_history.json"


def get_market_snapshot(
    db=None,
    mongo_uri: str = "mongodb://panda:panda@127.0.0.1:27017/admin?directConnection=true",
    mongo_db: str = "panda",
    *,
    cache_dir: Path | str | None = None,
    mode: str = "current",
) -> dict:
    """
    一次调用返回所有看板需要的实时市场状态。

    Args:
        db:        MarketDB / pymongo.Database / None
        mongo_uri: 无 db 时用来创建连接
        mongo_db:  数据库名
        cache_dir: 历史缓存目录 (None=模块内 .cache/)
        mode:      "current" | "predicted" (predicted 为未来预留)

    Returns:
        {
            "regime": str,
            "weather": {emoji, label, desc},
            "daily_ret": {hs300, gem},      # 当日涨跌 (%)
            "ret_20d": {hs300, gem},         # 20日累计 (%)
            "vol_20": float,
            "vol_pct": float,                # 波动率分位 (0-100)
            "breadth": float,                # MA20 广度 (0-100)
            "breadth_detail": {above, total},
            "date": str,                     # 数据截止日
            "history": {last_switch, switches},
        }
    """
    mdb = _resolve_db(db, mongo_uri, mongo_db)

    # 看板/快照：与「最新指数 bar」当日分类一致，不做 3 日确认延迟（预计算 parquet 仍可用 confirm_days=3）
    regime_info = classify_current(mdb, confirm_days=1)
    weather = get_weather(regime_info["regime"])
    temp = compute_breadth(mdb)

    history = _update_history(regime_info, weather, cache_dir)

    return {
        "regime": regime_info["regime"],
        "weather": weather,
        "daily_ret": regime_info["daily_ret"],
        "ret_20d": regime_info["ret_20d"],
        "vol_20": regime_info["vol_20"],
        "vol_pct": regime_info["vol_pct"],
        "breadth": temp["breadth"],
        "breadth_detail": {"above": temp["above"], "total": temp["total"]},
        "date": regime_info["date"],
        "history": history,
    }


# ── 内部工具 ─────────────────────────────────────────────


def _resolve_db(db, mongo_uri, mongo_db):
    """将各种 db 输入统一为 pymongo Database 对象。"""
    if db is not None:
        if hasattr(db, "reconnect"):
            db.reconnect()
        if hasattr(db, "_db"):
            return db._db
        return db
    from pymongo import MongoClient
    client = MongoClient(
        mongo_uri, serverSelectionTimeoutMS=5000,
        socketTimeoutMS=30000, retryReads=True,
    )
    return client[mongo_db]


def _update_history(regime_info: dict, weather: dict, cache_dir) -> dict:
    """记录 regime 切换，返回 {last_switch, switches}。"""
    hist_path = (
        Path(cache_dir) / "style_history.json" if cache_dir
        else _DEFAULT_HISTORY
    )
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    if hist_path.exists():
        try:
            history = json.loads(hist_path.read_text())
        except Exception:
            history = []

    cur_regime = regime_info.get("regime", "")
    cur_date = regime_info.get("date", "")

    if not history or history[-1].get("regime") != cur_regime:
        history.append({
            "regime": cur_regime,
            "emoji": weather.get("emoji", ""),
            "label": weather.get("label", ""),
            "date": cur_date,
        })
        try:
            hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=2))
        except Exception:
            pass

    switches = []
    for i in range(len(history) - 1, 0, -1):
        switches.append({
            "from_regime": history[i - 1]["regime"],
            "from_emoji": history[i - 1].get("emoji", ""),
            "from_label": history[i - 1].get("label", ""),
            "to_regime": history[i]["regime"],
            "to_emoji": history[i].get("emoji", ""),
            "to_label": history[i].get("label", ""),
            "date": history[i]["date"],
        })
        if len(switches) >= 10:
            break

    return {
        "last_switch": switches[0] if switches else None,
        "switches": switches,
    }
