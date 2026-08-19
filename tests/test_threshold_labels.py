"""Regression: threshold() must carry node/edge labels so label-aware molecular
concepts can match the explanation subgraph. Without this the logical axis collapses
to a constant on molecular data (the Mutagenicity/BBBP degeneracy) — a plumbing bug,
not a metric flaw."""
import networkx as nx

from gnnxc.project.threshold import threshold
from gnnxc.project.projection import project
from gnnxc.project.vocab import molecular_vocab


def _benzene_with_n():
    g = nx.Graph()
    for i in range(6):
        g.add_node(i, atom="C")
    g.add_node(6, atom="N")
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)])
    return g


def test_threshold_carries_atom_labels_when_graph_given():
    g = _benzene_with_n()
    imp = {frozenset(e): 1.0 for e in g.edges()}
    sub = threshold(imp, 7, g)
    assert all("atom" in d for _, d in sub.nodes(data=True))
    p = project(sub, molecular_vocab())
    assert p["benzene"] == 1 and p["nitrogen"] == 1


def test_threshold_without_graph_is_unlabelled_backward_compatible():
    g = _benzene_with_n()
    imp = {frozenset(e): 1.0 for e in g.edges()}
    sub = threshold(imp, 7)  # no graph → structural only (synthetic path unchanged)
    assert all(d == {} for _, d in sub.nodes(data=True))
    # structural ring6 still fires; label-aware benzene cannot (no labels) — as before
    p = project(sub, molecular_vocab())
    assert p["ring6"] == 1 and p["benzene"] == 0
