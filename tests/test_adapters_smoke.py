"""Task 12 — real adapter smoke tests. Torch-backed adapters skip-if-import
(validated on the L4 VM); the random control needs no torch and runs anywhere."""
import networkx as nx
import pytest

from gnnxc.explainers.adapters import build
from gnnxc.explainers.base import Explainer


def test_gt_oracle_perfect_and_deterministic():
    """GT-oracle (correctness anchor) is torch-free: importance 1 on planted-motif
    edges, 0 elsewhere; deterministic across seeds; skips graphs w/o ground truth."""
    import networkx as nx
    from gnnxc.data import load_ba2motifs
    from gnnxc.explainers.base import canonical_edge

    ex = build("gt_oracle")
    assert isinstance(ex, Explainer)
    g = load_ba2motifs(2, seed=0)[0][0]
    motif = g.graph["motif_edges"]
    imp = ex.explain(g, seed=0)
    assert all(imp[e] == 1.0 for e in motif)                    # motif edges = 1
    assert ex.explain(g, 0) == ex.explain(g, 9)                 # deterministic
    assert not ex.has_ground_truth(nx.cycle_graph(6))          # plain graph → no GT


def test_random_control_runs_anywhere():
    ex = build("random")
    assert isinstance(ex, Explainer)
    g = nx.cycle_graph(6)
    a, b = ex.explain(g, 0), ex.explain(g, 1)
    assert a.keys() == b.keys()   # same edges
    assert a != b                 # seed-varying (chance floor)


def test_deferred_adapters_skip_not_fake():
    # spec §12 roster restriction — SubgraphX still deferred, returns None (skip), never stub
    assert build("subgraphx") is None
    # pgexplainer is no longer deferred but needs model+dataset → None without them
    assert build("pgexplainer") is None


def test_pgexplainer_trains_and_explains():
    """PGExplainer smoke: trains a mask predictor per seed, then explains every graph."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from gnnxc.model import train_gin
    from gnnxc.data import load_ba2motifs

    ds = load_ba2motifs(12, seed=0)
    model, _ = train_gin(ds, epochs=30, seed=0)
    ex = build("pgexplainer", model=model, dataset=ds)
    assert ex is not None
    imp = ex.explain(ds[0][0], seed=0)
    assert len(imp) == ds[0][0].number_of_edges()
    # same seed → identical trained predictor → identical mask (caching + determinism)
    assert ex.explain(ds[0][0], seed=0) == imp


@pytest.mark.parametrize("name", ["ig", "gnnexplainer"])
def test_torch_adapters_smoke(name):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from gnnxc.model import train_gin
    from gnnxc.data import load_ba2motifs

    ds = load_ba2motifs(20, seed=0)
    model, _ = train_gin(ds, epochs=5, seed=0)
    ex = build(name, model=model)
    assert ex is not None
    imp = ex.explain(ds[0][0], seed=0)
    assert len(imp) == ds[0][0].number_of_edges()
    assert all(isinstance(v, float) for v in imp.values())


def test_ig_runs_on_all_graphs_and_is_deterministic():
    """The ceiling property on REAL explanations: IG must run on EVERY graph
    (not just graph[0] — a well-trained model leaves some edges with no gradient
    path, which crashed captum before the allow_unused fix) and be seed-invariant.
    Trains to the grid's epoch count so the failure mode is actually exercised."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from gnnxc.model import train_gin
    from gnnxc.data import load_ba2motifs

    ds = load_ba2motifs(20, seed=0)
    model, _ = train_gin(ds, epochs=100, seed=0)  # grid-matched: exercises unused-grad edges
    ex = build("ig", model=model)
    for g, _ in ds:  # every graph, not just the first
        imp = ex.explain(g, seed=0)
        assert len(imp) == g.number_of_edges()
    assert ex.explain(ds[0][0], seed=0) == ex.explain(ds[0][0], seed=7)  # deterministic
