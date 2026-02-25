from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import silhouette_score


def topic_diversity(topic_top_terms: Sequence[Sequence[str]]) -> float:
    total = sum(len(t) for t in topic_top_terms)
    if total == 0:
        return 0.0
    unique = len({term for terms in topic_top_terms for term in terms})
    return float(unique) / float(total)


def sampled_silhouette(matrix, labels, max_samples: int = 2000, random_state: int = 42) -> float | None:
    n = int(matrix.shape[0])
    if n < 3:
        return None
    unique_labels = len(set(int(x) for x in labels))
    if unique_labels < 2:
        return None

    if n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(n, size=max_samples, replace=False))
        matrix = matrix[idx]
        labels = np.asarray(labels)[idx]

    try:
        return float(silhouette_score(matrix, labels, metric="cosine"))
    except Exception:
        return None
