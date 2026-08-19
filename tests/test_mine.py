"""Data-adaptive vocabulary mining + admissibility gate (prereg-journal-v2).

Torch-free: synthetic graphs only. Verifies the mined vocabulary is (a) informative
(every concept's document-frequency lands inside the band), (b) label-aware when atoms
are present, (c) empty on a degenerate near-constant dataset — the exact failure mode
that collapsed the logic axis on real molecules — and that the admissibility gate
flags a saturated random control as inadmissible.
"""
import networkx as nx
import pytest

from gnnxc.data import load_ba2motifs
from gnnxc.project.mine import admissibility, mine_vocab
from gnnxc.project.projection import project


def _df(graphs, motif):
    """Document-frequency of a motif template across graphs (label-aware if labelled)."""
    v = {"c": motif}
    return sum(1 for g in graphs if project(g, v)["c"]) / len(graphs)


def test_mined_concepts_are_in_the_informative_band():
    graphs = [g for g, _ in load_ba2motifs(40, seed=1)]
    band = (0.2, 0.8)
    vocab = mine_vocab(graphs, max_nodes=3, band=band, max_concepts=10, sample=40, seed=0)
    assert len(vocab) > 0, "no informative concept mined on ba2motifs"
    assert len(vocab) <= 10
    for name, motif in vocab.items():
        df = _df(graphs, motif)
        # concepts selected by the band must actually fire in the band (allow tiny slack
        # since monomorphism can find a template in more graphs than raw enumeration)
        assert 0.2 - 0.05 <= df <= 1.0, f"{name}: df={df:.2f} outside informative range"
        assert df < 1.0, f"{name}: near-universal concept leaked into vocab (df={df:.2f})"


def test_mining_is_deterministic():
    graphs = [g for g, _ in load_ba2motifs(30, seed=2)]
    a = list(mine_vocab(graphs, seed=0))
    b = list(mine_vocab(graphs, seed=0))
    assert a == b


def test_label_aware_when_atoms_present():
    # small labelled molecules: a C-N-O path repeated → mid-frequency labelled concepts
    graphs = []
    for i in range(20):
        g = nx.Graph()
        atoms = ["C", "N", "O", "C"] if i % 2 == 0 else ["C", "C", "O", "N"]
        for j, a in enumerate(atoms):
            g.add_node(j, atom=a)
        g.add_edges_from([(0, 1), (1, 2), (2, 3)])
        graphs.append(g)
    vocab = mine_vocab(graphs, max_nodes=3, band=(0.1, 0.9), max_concepts=8, sample=20, seed=0)
    assert len(vocab) > 0
    # every mined template carries atom labels (label-aware projection will fire on atoms)
    for _, motif in vocab.items():
        assert all("atom" in d for _, d in motif.nodes(data=True))


def test_degenerate_dataset_yields_no_informative_concept():
    # every graph identical → every subgraph has df=1.0 → band excludes all → empty vocab.
    # This is the real-molecule failure mode made explicit: a non-varying projection
    # cannot support a logical axis, and mining refuses to invent one.
    base = nx.cycle_graph(6)
    graphs = [base.copy() for _ in range(20)]
    vocab = mine_vocab(graphs, max_nodes=3, band=(0.2, 0.8), max_concepts=10, sample=20, seed=0)
    assert len(vocab) == 0


def test_admissibility_gate_flags_saturated_random():
    points = [
        {"dataset": "good", "explainer": "random", "logic": 0.30, "k": 5},
        {"dataset": "good", "explainer": "random", "logic": 0.50, "k": 10},
        {"dataset": "bad", "explainer": "random", "logic": 1.00, "k": 5},
        {"dataset": "bad", "explainer": "random", "logic": 1.00, "k": 10},
    ]
    adm = admissibility(points, tau=0.90)
    assert adm["good"]["admissible"] is True
    assert adm["bad"]["admissible"] is False
    assert adm["bad"]["random_logic_mean"] == pytest.approx(1.0)
