"""Label-aware projection (journal Stage-A): motifs with atom labels match by
atom type, not just structure. Torch-free."""
import networkx as nx

from gnnxc.project.projection import project
from gnnxc.project.vocab import molecular_vocab


def _carbon_ring6():
    g = nx.cycle_graph(6)
    nx.set_node_attributes(g, "C", "atom")
    return g


def _nitrogen_ring6():
    g = nx.cycle_graph(6)
    nx.set_node_attributes(g, "N", "atom")
    return g


def test_benzene_matches_carbon_ring_not_nitrogen_ring():
    V = molecular_vocab()
    # a carbon 6-ring contains 'benzene'; a nitrogen 6-ring does not (label-aware)
    assert project(_carbon_ring6(), V)["benzene"] == 1
    assert project(_nitrogen_ring6(), V)["benzene"] == 0
    # but 'ring6' (structural, label-agnostic) matches both
    assert project(_carbon_ring6(), V)["ring6"] == 1
    assert project(_nitrogen_ring6(), V)["ring6"] == 1


def test_nitro_group_detected():
    V = molecular_vocab()
    g = nx.Graph()
    g.add_node(0, atom="N"); g.add_node(1, atom="O"); g.add_node(2, atom="O")
    g.add_node(3, atom="C")
    g.add_edges_from([(0, 1), (0, 2), (0, 3)])
    c = project(g, V)
    assert c["nitro"] == 1 and c["nitrogen"] == 1 and c["oxygen"] == 1


def test_single_atom_concepts():
    V = molecular_vocab()
    g = nx.Graph(); g.add_node(0, atom="Cl"); g.add_node(1, atom="C"); g.add_edge(0, 1)
    c = project(g, V)
    assert c["chlorine"] == 1 and c["fluorine"] == 0


def test_determinism_labeled():
    V = molecular_vocab()
    g = _carbon_ring6()
    assert project(g, V) == project(g, V)
