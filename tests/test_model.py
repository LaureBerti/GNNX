"""Task 13 — GIN trainer. Skips without torch (Intel-mac dev box); on the VM the
GIN must reach >0.9 train accuracy on BA-2Motifs (motif-separable sanity)."""
import pytest


def test_gin_learns_ba2motifs():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from gnnxc.data import load_ba2motifs
    from gnnxc.model import train_gin, evaluate

    ds = load_ba2motifs(120, seed=0)
    model, acc = train_gin(ds, epochs=100, seed=0)
    assert acc > 0.9, f"GIN only reached {acc:.3f} on BA-2Motifs"


def test_nx_to_pyg_shapes():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import networkx as nx
    from gnnxc.model import nx_to_pyg

    g = nx.cycle_graph(6)
    d = nx_to_pyg(g, 1)
    assert d.x.size(0) == 6
    assert d.edge_index.size(1) == 2 * g.number_of_edges()  # undirected → both dirs
    assert int(d.y.item()) == 1
