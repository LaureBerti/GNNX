"""Benzene loader validation (skip-if-import: RDKit + PyG). Runs in the VM smoke step
to fail fast before the 30-seed grid: confirms real molecules load, both classes are
present, and the benzene-ring ground-truth motif edges are set."""
import pytest


@pytest.mark.parametrize("loader", ["load_benzene", "load_fluoride_carbonyl", "load_alkane_carbonyl"])
def test_gt_motif_loaders_have_both_classes_and_gt(loader):
    pytest.importorskip("torch_geometric")
    pytest.importorskip("rdkit")
    import gnnxc.data as data

    ds = getattr(data, loader)(60, seed=0)
    assert len(ds) > 0
    labels = {y for _, y in ds}
    assert labels == {0, 1}, f"{loader}: expected both classes, got {labels}"
    # every positive-labelled molecule carries ground-truth motif edges
    pos = [g for g, y in ds if y == 1]
    assert pos and all(g.graph.get("motif_edges") for g in pos), f"{loader}: positives missing GT motif"
