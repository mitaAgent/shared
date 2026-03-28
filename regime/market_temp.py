"""
市场温度 — 全 A 站上 MA20 比例 (实时计算)

直接查询 MongoDB stock_market，不依赖 pool JSON 快照。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def compute_breadth(
    mongo_db,
    end_date: str | None = None,
) -> dict:
    """
    计算全 A 站上 MA20 比例。

    Args:
        mongo_db: pymongo Database 或含 _db 属性的对象
        end_date: YYYYMMDD 截止日 (None=今天)

    Returns:
        {"breadth": float (0-100), "date": str, "above": int, "total": int}
    """
    if hasattr(mongo_db, "_db"):
        mongo_db = mongo_db._db

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=50)).strftime("%Y%m%d")

    docs = list(mongo_db["stock_market"].find(
        {"date": {"$gte": start, "$lte": end_date}},
        {"_id": 0, "symbol": 1, "date": 1, "close": 1},
    ).batch_size(300000))

    if not docs:
        return {"breadth": 50.0, "date": end_date, "above": 0, "total": 0}

    df = pd.DataFrame(docs)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    pv = df.pivot(index="date", columns="symbol", values="close").sort_index()
    if len(pv) < 15:
        return {"breadth": 50.0, "date": end_date, "above": 0, "total": 0}

    ma20 = pv.rolling(20, min_periods=15).mean()

    latest_close = pv.iloc[-1].dropna()
    latest_ma20 = ma20.iloc[-1].dropna()
    common = latest_close.index.intersection(latest_ma20.index)
    if common.empty:
        return {"breadth": 50.0, "date": str(pv.index[-1]), "above": 0, "total": 0}

    above = (latest_close[common] > latest_ma20[common]).sum()
    total = len(common)
    breadth = float(above / total * 100) if total > 0 else 50.0

    return {
        "breadth": round(breadth, 1),
        "date": str(pv.index[-1]),
        "above": int(above),
        "total": int(total),
    }
