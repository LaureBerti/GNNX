import networkx as nx
import numpy as np
import pytest

from gnnxc.explainers.base import Explainer, canonical_edge
from gnnxc.explainers.mock import MockExplainer


@pytest.fixture
def graph():
    return nx.cycle_graph(6)


def test_protocol_conformance():
    assert isinstance(MockExplainer(), Explainer)


def test_deterministic_identical_across_seeds(graph):
    ex = MockExplainer(mode="deterministic")
    assert ex.explain(graph, seed=0) == ex.explain(graph, seed=99)


def test_noisy_varies_across_seeds(graph):
    ex = MockExplainer(mode="noisy")
    a = ex.explain(graph, seed=0)
    b = ex.explain(graph, seed=1)
    assert a != b
    assert a.keys() == b.keys()  # same edges, different values


def test_controllable_contradiction_rate(graph):
    # With decoy planted at rate ~0.3, the fraction of seeds highlighting the
    # decoy over the true motif should track contradiction_rate.
    motif = [(0, 1), (1, 2)]
    decoy = [(3, 4), (4, 5)]
    ex = MockExplainer(
        mode="noisy", contradiction_rate=0.3, motif_edges=motif, decoy_edges=decoy
    )
    decoy_keys = [canonical_edge(u, v) for u, v in decoy]
    motif_keys = [canonical_edge(u, v) for u, v in motif]
    n, decoy_wins = 400, 0
    for s in range(n):
        imp = ex.explain(graph, seed=s)
        if np.mean([imp[e] for e in decoy_keys]) > np.mean([imp[e] for e in motif_keys]):
            decoy_wins += 1
    assert 0.2 < decoy_wins / n < 0.4


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        MockExplainer(mode="banana")
