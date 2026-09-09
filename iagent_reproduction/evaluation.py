"""Published leave-one-out ranking protocol (positive + 9 negatives)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def ranking_metrics(ranked: Sequence[str], target: str) -> dict[str, float]:
    rank = ranked.index(target) + 1 if target in ranked else math.inf
    return {"HR@1": float(rank <= 1), "HR@3": float(rank <= 3),
            "NDCG@3": 1 / math.log2(rank + 1) if rank <= 3 else 0.0,
            "MRR": 1 / rank if math.isfinite(rank) else 0.0}


def average(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("empty result set")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
