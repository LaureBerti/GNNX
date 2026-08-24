"""The self-consistency certificate (the reporting standard).

A JSON record accompanying any GNN-explanation claim: the two consistency scores
with CIs, plus the three control anchors (random floor / IG ceiling / GT oracle)
so a reader can tell "consistent-and-right" from "consistent-but-wrong" and knows
the apparatus was validated noise-free.
"""
from __future__ import annotations

import json
from typing import Optional


def certificate(
    explainer: str,
    dataset: str,
    k: int,
    stat: float,
    logic: float,
    logic_ci: tuple,
    n_seeds: int,
    under_powered: bool,
    floor: Optional[float] = None,      # random-subgraph chance floor
    ceiling: Optional[float] = None,    # IG-ceiling validator (must be 1.0)
    oracle: Optional[float] = None,     # GT-motif correctness anchor
) -> dict:
    """Build one certificate record. ``ceiling`` != 1.0 flags a leaking apparatus."""
    return {
        "explainer": explainer,
        "dataset": dataset,
        "k": k,
        "n_seeds": n_seeds,
        "statistical_consistency": stat,
        "logical_consistency": logic,
        "logical_consistency_ci95": list(logic_ci),
        "under_powered": under_powered,
        "controls": {
            "random_floor": floor,
            "ig_ceiling": ceiling,
            "gt_oracle": oracle,
            "apparatus_valid": (ceiling is None or abs(ceiling - 1.0) < 1e-9),
        },
    }


def save_certificates(records: list, path: str) -> str:
    with open(path, "w") as f:
        json.dump({"certificates": records}, f, indent=2)
    return path
