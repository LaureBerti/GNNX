"""The one significance bar: paired Wilcoxon + cluster bootstrap.

Nesting is respected: the bootstrap resamples **clusters** (datasets/graphs) — NOT
a naive paired test across 3 datasets (under-powered) and NOT per-graph
pseudo-replication (anti-conservative). A comparison below the seed floor is
labelled ``under_powered`` instead of asserting a p-value (the seed floor).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon


@dataclass
class PairedResult:
    statistic: Optional[float]
    p_value: Optional[float]
    n: int
    under_powered: bool
    label: str

    def as_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n": self.n,
            "under_powered": self.under_powered,
            "label": self.label,
        }


def paired_wilcoxon(
    a: Sequence[float], b: Sequence[float], seed_floor: int = 10, alpha: float = 0.05
) -> PairedResult:
    """Paired Wilcoxon signed-rank of a vs b, with a seed-floor guard."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    n = int(a.shape[0])
    if n < seed_floor:
        return PairedResult(None, None, n, True, "under-powered")
    if np.allclose(a, b):
        return PairedResult(0.0, 1.0, n, False, "no-difference")
    stat, p = wilcoxon(a, b)
    label = "significant" if p < alpha else "n.s."
    return PairedResult(float(stat), float(p), n, False, label)


def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence,
    resamples: int = 10000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """Cluster bootstrap CI for the mean: resample *clusters* with replacement.

    Returns (point_mean, lo, hi). Datasets/graphs are the clusters so nesting is
    respected. Deterministic given ``rng_seed``.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    groups: Dict[object, np.ndarray] = {c: values[clusters == c] for c in uniq}
    rng = np.random.default_rng(rng_seed)

    means: List[float] = []
    for _ in range(resamples):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        pooled = np.concatenate([groups[c] for c in chosen])
        means.append(pooled.mean())
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return float(values.mean()), lo, hi
