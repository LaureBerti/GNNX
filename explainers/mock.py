"""MockExplainer — the TDD seam.

Lets every downstream layer (projection → aggregate → statistical/logical) be
tested end-to-end with no real explainer and no compute, and gives the
consistency estimators a *known ground truth*:

- ``mode="deterministic"``  → identical importance for every seed (the IG-ceiling
  stand-in: logical-consistency must read 1 through the full pipeline).
- ``mode="noisy"``          → seed-varying importance. If ``motif_edges`` /
  ``decoy_edges`` are supplied, each seed highlights the decoy with probability
  ``contradiction_rate`` and the true motif otherwise — a *controllable* pairwise
  contradiction rate, so the estimator can be checked against a known value.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .base import Edge, Importance, canonical_edge


class MockExplainer:
    name = "mock"

    def __init__(
        self,
        mode: str = "deterministic",
        contradiction_rate: float = 0.0,
        motif_edges: Optional[Sequence[Tuple[int, int]]] = None,
        decoy_edges: Optional[Sequence[Tuple[int, int]]] = None,
        base_seed: int = 12345,
    ):
        if mode not in ("deterministic", "noisy"):
            raise ValueError(f"unknown mode: {mode!r}")
        if not 0.0 <= contradiction_rate <= 1.0:
            raise ValueError("contradiction_rate must be in [0, 1]")
        self.mode = mode
        self.contradiction_rate = contradiction_rate
        self.motif_edges = _canon(motif_edges) if motif_edges else None
        self.decoy_edges = _canon(decoy_edges) if decoy_edges else None
        self.base_seed = base_seed

    def explain(self, graph: nx.Graph, seed: int) -> Importance:
        edges = [canonical_edge(u, v) for u, v in graph.edges()]
        if self.mode == "deterministic":
            return {e: _fixed_score(e) for e in edges}

        rng = np.random.default_rng(self.base_seed + seed)
        imp: Importance = {e: float(rng.random()) for e in edges}

        if self.motif_edges is not None:
            highlight = self.motif_edges
            if self.decoy_edges is not None and rng.random() < self.contradiction_rate:
                highlight = self.decoy_edges
            for e in highlight:
                if e in imp:
                    imp[e] += 1.0
        return imp


def _canon(edges: Iterable[Tuple[int, int]]) -> list[Edge]:
    return [canonical_edge(u, v) for u, v in edges]


def _fixed_score(edge: Edge) -> float:
    a, b = sorted(edge)
    return float((a * 131 + b) % 1000) / 1000.0
