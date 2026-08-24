"""Statistical layer — dispersion of explanations across seeds.

All overlap statistics are **chance-corrected**: observed − expected-under-random,
normalised to [·, 1] so identical explanations read 1 and random pairs read ~0
(the chance-correction bar). Kendall's τ measures importance-rank agreement;
per-concept support variance measures concept-presence dispersion across seeds.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import kendalltau


def _topk_set(scores: Sequence[float], k: int) -> set:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return set(order[:k].tolist())


def chance_corrected_overlap(a: Sequence[float], b: Sequence[float], k: int) -> float:
    """Chance-corrected top-k set overlap of two importance vectors.

    obs = |A∩B|/k ; exp = k/n (expected overlap of a random k-subset) ;
    returns (obs − exp)/(1 − exp). Identical → 1; independent-random → ~0.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("importance vectors must be the same length")
    k = min(k, n)
    if k == 0:
        return 0.0
    A, B = _topk_set(a, k), _topk_set(b, k)
    obs = len(A & B) / k
    exp = k / n
    if exp >= 1.0:
        return 1.0
    return (obs - exp) / (1.0 - exp)


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall's τ between two importance rankings (NaN → 0.0 for constant input)."""
    tau, _ = kendalltau(a, b)
    return float(tau) if tau == tau else 0.0  # NaN guard


def support_variance(concept_vectors: Sequence[Dict[str, int]]) -> Dict[str, float]:
    """Per-concept variance of presence across seeds (0 = perfectly stable)."""
    if not concept_vectors:
        return {}
    names = list(concept_vectors[0].keys())
    return {
        n: float(np.var([cv.get(n, 0) for cv in concept_vectors]))
        for n in names
    }
