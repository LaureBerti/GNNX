from gnnxc.explainers.base import canonical_edge
from gnnxc.project.threshold import threshold, top_k_edges


def _imp(pairs):
    return {canonical_edge(u, v): w for (u, v), w in pairs}


def test_selects_exactly_top_k():
    imp = _imp([((0, 1), 0.9), ((1, 2), 0.8), ((2, 3), 0.1), ((3, 4), 0.05)])
    g = threshold(imp, 2)
    assert g.number_of_edges() == 2
    assert set(map(frozenset, g.edges())) == {canonical_edge(0, 1), canonical_edge(1, 2)}


def test_sweep_is_nested():
    imp = _imp([((0, 1), 0.9), ((1, 2), 0.8), ((2, 3), 0.7), ((3, 4), 0.6)])
    e2 = set(top_k_edges(imp, 2))
    e3 = set(top_k_edges(imp, 3))
    assert e2 < e3  # strict subset — nested


def test_deterministic_tiebreak():
    imp = _imp([((0, 1), 0.5), ((2, 3), 0.5), ((4, 5), 0.5)])
    assert top_k_edges(imp, 2) == top_k_edges(imp, 2)
    # lowest canonical ids win ties
    assert top_k_edges(imp, 1) == [canonical_edge(0, 1)]
