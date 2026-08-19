"""Benzene benchmark pieces (torch-free): GT-motif vocabulary, explanation-accuracy
(GEA) against ground-truth motif edges, and vocab selection. The load_benzene loader
itself is skip-if-import (RDKit + PyG), validated on the VM."""
import networkx as nx

from gnnxc.data import vocab_for
from gnnxc.pipeline import explanation_accuracy
from gnnxc.project.vocab import benzene_vocab, molecular_vocab
from gnnxc.project.projection import project
from gnnxc.project.threshold import threshold


def test_vocab_for_benzene():
    assert list(vocab_for("benzene").keys()) == list(benzene_vocab().keys())
    assert "benzene" in vocab_for("benzene")


def test_vocab_for_fc_and_ac():
    assert "fluoride" in vocab_for("fluoride_carbonyl") and "carbonyl" in vocab_for("fluoride_carbonyl")
    assert "alkane" in vocab_for("alkane_carbonyl") and "carbonyl" in vocab_for("alkane_carbonyl")


def test_benzene_concept_fires_on_aromatic_ring_via_fixed_threshold():
    g = nx.Graph()
    for i in range(6):
        g.add_node(i, atom="c")   # aromatic carbons
    g.add_node(6, atom="O")
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)])
    imp = {frozenset(e): 1.0 for e in g.edges()}
    sub = threshold(imp, 7, g)   # label-propagating threshold
    assert project(sub, benzene_vocab())["benzene"] == 1


class _FixedExplainer:
    """Deterministic explainer that returns the same importances every seed."""
    name = "fixed"
    def __init__(self, scores): self.scores = scores
    def explain(self, graph, seed): return self.scores


def test_explanation_accuracy_matches_ground_truth():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])   # triangle motif + a tail edge
    gt = {frozenset((0, 1)), frozenset((1, 2)), frozenset((2, 0))}
    g.graph["motif_edges"] = gt
    # explainer that ranks exactly the motif edges top-3
    scores = {frozenset((0, 1)): 1.0, frozenset((1, 2)): 1.0, frozenset((2, 0)): 1.0,
              frozenset((2, 3)): 0.0}
    ex = _FixedExplainer(scores)
    gea = explanation_accuracy(ex, [(g, 1)], S=3, k=3)
    assert gea == 1.0   # perfect Jaccard
    # top-4 includes the tail edge → Jaccard 3/4
    assert explanation_accuracy(ex, [(g, 1)], S=3, k=4) == 0.75


def test_explanation_accuracy_none_without_ground_truth():
    g = nx.Graph(); g.add_edges_from([(0, 1), (1, 2)])
    ex = _FixedExplainer({frozenset((0, 1)): 1.0, frozenset((1, 2)): 0.0})
    assert explanation_accuracy(ex, [(g, 0)], S=2, k=1) is None
