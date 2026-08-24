"""Explainer protocol: the single seam every adapter (real or mock) implements.

An explainer maps ``(graph, seed) -> edge-importance``. Importance is a dict
keyed by undirected edge ``frozenset({u, v})`` (canonical, order-free) to a float.
Thresholding at sparsity k turns this into the discrete subgraph.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Protocol, runtime_checkable

import networkx as nx

Edge = FrozenSet[int]
Importance = Dict[Edge, float]


def canonical_edge(u: int, v: int) -> Edge:
    """Order-free edge key so (u,v) and (v,u) collide."""
    return frozenset((u, v))


@runtime_checkable
class Explainer(Protocol):
    """Any object exposing ``explain(graph, seed) -> Importance``."""

    name: str

    def explain(self, graph: nx.Graph, seed: int) -> Importance:
        ...
