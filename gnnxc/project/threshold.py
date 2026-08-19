"""Sparsity thresholding — the k-sweep (axis d).

``threshold(importance, k)`` keeps the top-k edges by importance and returns the
discrete explanation subgraph. Ties break on the canonical edge id so the result
is deterministic and the sweep is *nested*: top-k ⊂ top-(k+1).
"""
from __future__ import annotations

import networkx as nx

from ..explainers.base import Importance


def top_k_edges(importance: Importance, k: int) -> list:
    """Return the top-k edge keys, deterministically (importance desc, id asc)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    ordered = sorted(
        importance.keys(),
        key=lambda e: (-importance[e], tuple(sorted(e))),
    )
    return ordered[:k]


def threshold(importance: Importance, k: int, graph: nx.Graph | None = None) -> nx.Graph:
    """Discrete explanation subgraph = graph on the top-k edges.

    When ``graph`` is provided, node and edge attributes (e.g. atom/bond labels) are
    carried over from it onto the subgraph. This is REQUIRED for label-aware molecular
    concept projection: without it the subgraph is unlabelled and every atom-typed
    concept (benzene, nitro, …) can never match, collapsing the logical axis to a
    constant (the Mutagenicity/BBBP degeneracy). Synthetic structural motifs are
    unaffected, so the argument is optional for backward compatibility.
    """
    g = nx.Graph()
    for e in top_k_edges(importance, k):
        u, v = tuple(sorted(e))
        g.add_edge(u, v)
        if graph is not None:
            g.nodes[u].update(graph.nodes[u])
            g.nodes[v].update(graph.nodes[v])
            if graph.has_edge(u, v):
                g.edges[u, v].update(graph.edges[u, v])
    return g
