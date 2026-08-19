"""Real molecular dataset loaders (journal Stage-A). Skip-if-import; validated on
the L4 VM (need torch-geometric + downloads). Checks atom labels are set and the
label-aware molecular projection produces non-trivial concept vectors."""
import pytest


@pytest.mark.parametrize("name,loader", [
    ("mutagenicity", "load_mutagenicity"),
    ("mutag_labeled", "load_mutag_labeled"),
    ("bbbp", "load_bbbp"),
])
def test_real_loader_sets_atom_labels(name, loader):
    pytest.importorskip("torch_geometric")
    import gnnxc.data as data
    from gnnxc.project.projection import project
    from gnnxc.data import vocab_for

    ds = getattr(data, loader)(12, seed=0)
    assert len(ds) > 0
    g0 = ds[0][0]
    atoms = {d.get("atom") for _, d in g0.nodes(data=True)}
    assert atoms and "?" not in atoms, f"{name}: atom labels missing ({atoms})"
    # label-aware projection yields a concept vector over the molecular vocab
    V = vocab_for(name)
    c = project(g0, V)
    assert set(c.keys()) == set(V.keys())
    # across a few graphs at least one molecular concept fires (sanity, not always)
    fired = any(any(project(g, V).values()) for g, _ in ds[:12])
    assert fired, f"{name}: no molecular concept fired on 12 graphs"


def test_vocab_for_selects_molecular():
    from gnnxc.data import vocab_for
    from gnnxc.project.vocab import molecular_vocab, default_vocab
    assert list(vocab_for("bbbp").keys()) == list(molecular_vocab().keys())
    assert list(vocab_for("ba2motifs").keys()) == list(default_vocab().keys())
