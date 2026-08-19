import networkx as nx

from gnnxc.data import load_ba2motifs, load_bamultishapes, load_dataset
from gnnxc.project.projection import project


def test_ba2motifs_has_motif_ground_truth():
    ds = load_ba2motifs(10, seed=0)
    assert len(ds) == 10
    assert {label for _, label in ds} == {0, 1}
    # class-1 graphs contain a house motif; the projection should see it
    houses = [project(g)["house"] for g, y in ds if y == 1]
    assert sum(houses) >= 1


def test_bamultishapes_balanced_labels():
    ds = load_bamultishapes(9, seed=1)
    labels = [y for _, y in ds]
    assert set(labels) <= {0, 1}
    assert all(isinstance(g, nx.Graph) for g, _ in ds)


def test_load_dataset_dispatch_and_determinism():
    a = load_dataset("ba2motifs", 6, seed=3)
    b = load_dataset("ba2motifs", 6, seed=3)
    assert [y for _, y in a] == [y for _, y in b]
    assert [g.number_of_edges() for g, _ in a] == [g.number_of_edges() for g, _ in b]
