"""Subgraph-isomorphism concept projection.

``project(subgraph, V)`` -> concept-presence vector c ∈ {0,1}^|V|. A concept is
present iff the explanation subgraph *contains* the motif template as a
(non-induced) subgraph — networkx monomorphism. Deterministic by construction:
the same input yields an identical vector, always (the IG-ceiling validator depends on this).
"""
from __future__ import annotations

from typing import Dict, Mapping

import networkx as nx

from .vocab import default_vocab


def _contains(subgraph: nx.Graph, motif: nx.Graph) -> int:
    """1 iff ``subgraph`` contains ``motif`` as a (non-induced) subgraph.

    If the motif carries node ``atom`` and/or edge ``bond`` labels (molecular
    motifs), matching is **label-aware** (categorical match on those attributes);
    otherwise it is purely structural (synthetic BA motifs). This lets the same
    projection serve synthetic graphs and real molecules.
    """
    if motif.number_of_nodes() > subgraph.number_of_nodes():
        return 0
    if motif.number_of_edges() > subgraph.number_of_edges():
        return 0
    iso = nx.algorithms.isomorphism
    node_match = None
    if any("atom" in d for _, d in motif.nodes(data=True)):
        node_match = iso.categorical_node_match("atom", None)
    edge_match = None
    if any("bond" in d for _, _, d in motif.edges(data=True)):
        edge_match = iso.categorical_edge_match("bond", None)
    gm = iso.GraphMatcher(subgraph, motif, node_match=node_match, edge_match=edge_match)
    return int(gm.subgraph_is_monomorphic())


def project(
    subgraph: nx.Graph,
    vocab: "Mapping[str, nx.Graph] | None" = None,
) -> "Dict[str, int]":
    """Return the ordered concept-presence dict for ``subgraph`` over ``vocab``.

    Deterministic: no randomness, no learned prototypes. The dict preserves the
    fixed vocabulary order so downstream vectors line up across seeds/explainers.
    """
    vocab = vocab if vocab is not None else default_vocab()
    return {name: _contains(subgraph, motif) for name, motif in vocab.items()}


def project_vector(
    subgraph: nx.Graph,
    vocab: "Mapping[str, nx.Graph] | None" = None,
) -> list[int]:
    """Same as :func:`project` but returns just the ordered 0/1 list."""
    return list(project(subgraph, vocab).values())
