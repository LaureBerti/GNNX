"""The dissociation plane (the central result).

Each (explainer, dataset) point is placed in the (statistical × logical
self-consistency) plane and classified into one of four cells. The off-diagonal
cells are the contribution:

- ``stat_hi ∧ logic_lo`` — the **dangerous** cell: reproducible masks, contradictory
  class rules; invisible to every current overlap metric.
- ``stat_lo ∧ logic_hi`` — current metrics wrongly penalise a semantically stable
  explainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

DANGEROUS = "stat_consistent_logic_contradictory"
BENIGN_DIVERGENT = "stat_divergent_logic_equivalent"
CONSISTENT = "consistent"
INCONSISTENT = "inconsistent"


@dataclass
class PlanePoint:
    explainer: str
    dataset: str
    stat: float
    logic: float
    cell: str

    def as_dict(self) -> dict:
        return {
            "explainer": self.explainer,
            "dataset": self.dataset,
            "stat": self.stat,
            "logic": self.logic,
            "cell": self.cell,
        }


def classify(stat: float, logic: float, tau: float = 0.5) -> str:
    stat_hi, logic_hi = stat >= tau, logic >= tau
    if stat_hi and logic_hi:
        return CONSISTENT
    if stat_hi and not logic_hi:
        return DANGEROUS
    if not stat_hi and logic_hi:
        return BENIGN_DIVERGENT
    return INCONSISTENT


def plane(points: Sequence[dict], tau: float = 0.5) -> List[PlanePoint]:
    """Classify each {explainer, dataset, stat, logic} point into a plane cell."""
    out = []
    for p in points:
        out.append(
            PlanePoint(
                explainer=p["explainer"],
                dataset=p["dataset"],
                stat=float(p["stat"]),
                logic=float(p["logic"]),
                cell=classify(float(p["stat"]), float(p["logic"]), tau),
            )
        )
    return out


def off_diagonal_proportion(points: Sequence[PlanePoint]) -> float:
    """Fraction of points in an off-diagonal cell (the H2 dissociation quantity)."""
    if not points:
        return 0.0
    off = sum(1 for p in points if p.cell in (DANGEROUS, BENIGN_DIVERGENT))
    return off / len(points)


def save_plane_figure(points: Sequence[PlanePoint], path: str, tau: float = 0.5) -> str:
    """Scatter of the (stat × logic) plane → PDF. Matplotlib Agg (headless-safe)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.axhline(tau, color="grey", lw=0.6, ls="--")
    ax.axvline(tau, color="grey", lw=0.6, ls="--")
    for p in points:
        ax.scatter(p.stat, p.logic, s=40)
        ax.annotate(f"{p.explainer}\n{p.dataset}", (p.stat, p.logic), fontsize=6)
    ax.set_xlabel("statistical self-consistency")
    ax.set_ylabel("logical self-consistency")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Dissociation plane")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
