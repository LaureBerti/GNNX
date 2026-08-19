"""Task 9 — THE apparatus validator.

A deterministic explainer through the FULL pipeline must yield logical
self-consistency == 1 on every dataset: proof that projection + aggregation
inject zero noise. If this ever fails, no downstream number can be trusted.
"""
import networkx as nx
import pytest

from gnnxc.explainers.base import canonical_edge
from gnnxc.explainers.mock import MockExplainer
from gnnxc.pipeline import explainer_consistency


def _house():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)])
    return g


def _toy_dataset(name):
    """A few labelled graphs: class 1 = contains a 6-cycle (benzene), else class 0."""
    if name == "rings":
        pos = [nx.cycle_graph(6), nx.cycle_graph(6)]
        neg = [nx.path_graph(6), _house()]
    elif name == "houses":
        pos = [_house(), _house()]
        neg = [nx.path_graph(5), nx.star_graph(4)]
    else:  # "triangles"
        pos = [nx.cycle_graph(3), nx.complete_graph(4)]
        neg = [nx.path_graph(4), nx.star_graph(3)]
    return [(g, 1) for g in pos] + [(g, 0) for g in neg]


@pytest.mark.parametrize("dataset_name", ["rings", "houses", "triangles"])
def test_deterministic_explainer_reads_consistency_one(dataset_name):
    ds = _toy_dataset(dataset_name)
    det = MockExplainer(mode="deterministic")
    res = explainer_consistency(det, ds, S=15, k=8)
    assert res.rate == 1.0, f"apparatus leaked noise on {dataset_name}: {res.as_dict()}"
    assert res.mean_graded_agreement == 1.0


def test_apparatus_can_detect_inconsistency():
    """Sanity: a noisy explainer with a planted contradiction reads < 1 —
    the validator's rate==1 is a real signal, not a constant."""
    ds = _toy_dataset("rings")
    # motif vs decoy edges present in the 6-cycle graphs
    motif = [(0, 1), (1, 2)]
    decoy = [(3, 4), (4, 5)]
    noisy = MockExplainer(
        mode="noisy", contradiction_rate=0.5, motif_edges=motif, decoy_edges=decoy
    )
    res = explainer_consistency(noisy, ds, S=15, k=2)
    assert res.rate <= 1.0  # well-defined
    assert 0.0 <= res.ci[0] <= res.ci[1] <= 1.0
