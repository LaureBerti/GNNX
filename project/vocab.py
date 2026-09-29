"""Fixed motif vocabulary V.

Each concept is a structural template graph; presence is decided by (non-induced)
subgraph monomorphism — a binary, learned-prototype-free test so the projection
injects zero noise (the determinism guarantee underpinning the IG-ceiling validator).

The vocabulary is *fixed* and ordered: the concept vector index layout never changes
within a run, so rules over c(G) are comparable across seeds and explainers.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import networkx as nx


def _cycle(n: int) -> nx.Graph:
    return nx.cycle_graph(n)


def _house() -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)])
    return g


def _grid() -> nx.Graph:
    return nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))


def _wheel() -> nx.Graph:
    return nx.wheel_graph(6)


def default_vocab() -> "OrderedDict[str, nx.Graph]":
    """Return the fixed, ordered motif vocabulary used in the scope studied here.

    Structural-only templates (no node-label matching) — sufficient for the
    synthetic BA datasets and adequate for MUTAG rings in this scope. A
    later extension adds label-aware predicates (NO2, atom types).
    """
    return OrderedDict(
        [
            ("triangle", _cycle(3)),
            ("cycle", _cycle(5)),
            ("benzene", _cycle(6)),
            ("house", _house()),
            ("grid", _grid()),
            ("wheel", _wheel()),
        ]
    )


def vocab_names(vocab: "Dict[str, nx.Graph] | None" = None) -> list[str]:
    """Ordered concept names — the fixed index layout of every concept vector."""
    return list((vocab or default_vocab()).keys())


def _atom(*atoms: str) -> nx.Graph:
    """A single- or multi-node motif labelled by atom type (structural edges)."""
    g = nx.Graph()
    for i, a in enumerate(atoms):
        g.add_node(i, atom=a)
    return g


def _benzene() -> nx.Graph:
    g = nx.cycle_graph(6)
    nx.set_node_attributes(g, "C", "atom")
    return g


def _nitro() -> nx.Graph:
    g = nx.Graph()
    g.add_node(0, atom="N"); g.add_node(1, atom="O"); g.add_node(2, atom="O")
    g.add_edges_from([(0, 1), (0, 2)])
    return g


def _carbonyl() -> nx.Graph:
    g = nx.Graph(); g.add_node(0, atom="C"); g.add_node(1, atom="O"); g.add_edge(0, 1)
    return g


def _aromatic_benzene() -> nx.Graph:
    """Aromatic-carbon 6-ring: the ground-truth motif of the Benzene benchmark."""
    g = nx.cycle_graph(6)
    nx.set_node_attributes(g, "c", "atom")
    return g


def benzene_vocab() -> "OrderedDict[str, nx.Graph]":
    """Vocabulary for the Benzene benchmark: the ground-truth motif (aromatic 6-ring)
    plus distractor concepts. The GT motif is the class-defining, localized concept, so
    the induced class rule ('benzene present => class 1') is interpretable and the
    logical axis is informative (admissible) — validated in the Benzene prototype."""
    return OrderedDict(
        [
            ("benzene", _aromatic_benzene()),
            ("carbonyl", _carbonyl()),
            ("aromatic_n", _atom("n")),
            ("nitrogen", _atom("N")),
            ("oxygen", _atom("O")),
        ]
    )


def _alkane3() -> nx.Graph:
    """A short unbranched aliphatic-carbon chain (alkane proxy)."""
    g = nx.path_graph(3)
    nx.set_node_attributes(g, "C", "atom")
    return g


def fluoride_carbonyl_vocab() -> "OrderedDict[str, nx.Graph]":
    """GT-motif vocabulary for the Fluoride-Carbonyl benchmark (label = F and C=O)."""
    return OrderedDict(
        [
            ("fluoride", _atom("F")),
            ("carbonyl", _carbonyl()),
            ("benzene", _aromatic_benzene()),
            ("nitrogen", _atom("N")),
            ("oxygen", _atom("O")),
        ]
    )


def alkane_carbonyl_vocab() -> "OrderedDict[str, nx.Graph]":
    """GT-motif vocabulary for the Alkane-Carbonyl benchmark (label = alkane and C=O)."""
    return OrderedDict(
        [
            ("alkane", _alkane3()),
            ("carbonyl", _carbonyl()),
            ("benzene", _aromatic_benzene()),
            ("oxygen", _atom("O")),
        ]
    )


def molecular_vocab() -> "OrderedDict[str, nx.Graph]":
    """Label-aware motif vocabulary for real molecular graphs (heavy-atom).

    Concepts present iff the explanation subgraph contains the labelled motif
    (atom-type-aware subgraph monomorphism). |V| is kept small (8) so exact
    model-counting stays cheap; the atom strings must match the loader's
    node ``atom`` labels (the loader maps dataset atom-type ids → 'C','N','O',...).
    """
    return OrderedDict(
        [
            ("benzene", _benzene()),
            ("nitro", _nitro()),
            ("carbonyl", _carbonyl()),
            ("nitrogen", _atom("N")),
            ("oxygen", _atom("O")),
            ("fluorine", _atom("F")),
            ("chlorine", _atom("Cl")),
            ("ring6", _cycle(6)),
        ]
    )
