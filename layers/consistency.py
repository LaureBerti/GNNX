"""Self-consistency aggregation over S explainer seeds.

Logical self-consistency = fraction of the C(S,2) seed-*pairs* whose class rules are
mutually non-contradictory, reported with a Wilson score CI (robust near 0/1 and at
small S — seed floor). A deterministic explainer must read rate == 1
(the IG-ceiling guarantee).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..aggregate.rule import Rule
from .logical import graded_agreement, non_contradiction


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% at z=1.96)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class ConsistencyResult:
    rate: float
    n_pairs: int
    n_consistent: int
    ci: Tuple[float, float]
    mean_graded_agreement: float
    under_powered: bool

    def as_dict(self) -> dict:
        return {
            "rate": self.rate,
            "n_pairs": self.n_pairs,
            "n_consistent": self.n_consistent,
            "ci95": list(self.ci),
            "mean_graded_agreement": self.mean_graded_agreement,
            "under_powered": self.under_powered,
        }


def logical_consistency(
    rules: Sequence[Rule],
    names: Sequence[str] | None = None,
    seed_floor: int = 10,
) -> ConsistencyResult:
    """Pairwise logical self-consistency over {R_s} with a Wilson CI."""
    pairs: List[Tuple[Rule, Rule]] = list(itertools.combinations(rules, 2))
    n_pairs = len(pairs)
    n_consistent = sum(1 for a, b in pairs if non_contradiction([a, b], names))
    graded = [graded_agreement(a, b, names) for a, b in pairs]
    rate = n_consistent / n_pairs if n_pairs else 1.0
    mean_graded = sum(graded) / len(graded) if graded else 1.0
    return ConsistencyResult(
        rate=rate,
        n_pairs=n_pairs,
        n_consistent=n_consistent,
        ci=wilson_ci(n_consistent, n_pairs),
        mean_graded_agreement=mean_graded,
        under_powered=len(rules) < seed_floor,
    )
