import networkx as nx

from gnnxc.project.projection import project
from gnnxc.project.vocab import default_vocab


def test_cycle_present_benzene_absent():
    # A 5-cycle: contains `cycle` (C5, and C3 monomorphically? no — C3 needs a
    # triangle, a pure C5 has none) but NOT `benzene` (C6).
    g = nx.cycle_graph(5)
    c = project(g)
    assert c["cycle"] == 1
    assert c["benzene"] == 0
    assert c["triangle"] == 0  # pure 5-cycle has no triangle


def test_benzene_ring_present():
    g = nx.cycle_graph(6)
    c = project(g)
    assert c["benzene"] == 1


def test_determinism_same_input_twice():
    g = nx.cycle_graph(5)
    assert project(g) == project(g)


def test_vector_order_matches_vocab():
    g = nx.cycle_graph(6)
    c = project(g)
    assert list(c.keys()) == list(default_vocab().keys())
