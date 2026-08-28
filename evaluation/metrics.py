from __future__ import annotations

import math
import statistics
from typing import Dict, List, Sequence, Tuple

Hit = Tuple[int, int, float]
Gold = Tuple[int, int]


def _positions(hits: Sequence[Hit], gold: Gold, chunk_level: bool) -> int:
    """1-basierter Rang des Goldtreffers, 0 wenn nicht enthalten."""
    for rank, (case_id, chunk_id, _score) in enumerate(hits, start=1):
        if chunk_level:
            if (case_id, chunk_id) == gold:
                return rank
        elif case_id == gold[0]:
            return rank
    return 0


def hit_at_k(hits: Sequence[Hit], gold: Gold, k: int, chunk_level: bool = True) -> float:
    rank = _positions(hits[:k], gold, chunk_level)
    return 1.0 if rank else 0.0


def reciprocal_rank(hits: Sequence[Hit], gold: Gold, chunk_level: bool = True) -> float:
    rank = _positions(hits, gold, chunk_level)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(hits: Sequence[Hit], gold: Gold, k: int, chunk_level: bool = True) -> float:
    """nDCG bei genau einem relevanten Dokument: 1/log2(rang+1), ideal ist 1.0."""
    rank = _positions(hits[:k], gold, chunk_level)
    if not rank:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def aggregate(
    per_question: List[Dict[str, float]],
) -> Dict[str, float]:
    """Mittelt die Einzelmetriken über alle Fragen."""
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {key: round(statistics.mean([q[key] for q in per_question]), 4) for key in keys}


def evaluate_hits(
    hits: Sequence[Hit],
    gold: Gold,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Alle Metriken für eine einzelne Frage, auf Chunk- und auf Fallebene."""
    out: Dict[str, float] = {}
    for k in k_values:
        out[f"hit@{k}"] = hit_at_k(hits, gold, k, chunk_level=True)
        out[f"case_hit@{k}"] = hit_at_k(hits, gold, k, chunk_level=False)
    out["mrr"] = reciprocal_rank(hits, gold, chunk_level=True)
    out["case_mrr"] = reciprocal_rank(hits, gold, chunk_level=False)
    out["ndcg@10"] = ndcg_at_k(hits, gold, 10, chunk_level=True)
    return out
