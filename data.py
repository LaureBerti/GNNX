"""Datasets with motif ground truth.

The synthetic BA-family datasets (BA-2Motifs, BAMultiShapes) are generated in pure
networkx — no torch — so the whole MockExplainer plumbing runs with zero heavy
deps. MUTAG loads via torch-geometric's TUDataset **only if available**
(skip-if-import; never faked). The real GIN training path lives in
:mod:`gnnxc.model`.
"""
from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import numpy as np

Labeled = Tuple[nx.Graph, int]


def _ba_base(n: int, rng) -> nx.Graph:
    seed = int(rng.integers(0, 2**31 - 1))
    return nx.barabasi_albert_graph(n, 1, seed=seed)


def _attach(base: nx.Graph, motif: nx.Graph, rng) -> nx.Graph:
    g = nx.disjoint_union(base, motif)  # base: 0..nb-1, motif: nb..nb+nm-1
    nb = base.number_of_nodes()
    # Record the planted-motif edges (ground truth) for the GT-oracle control.
    g.graph["motif_edges"] = {frozenset((nb + u, nb + v)) for u, v in motif.edges()}
    b = int(rng.integers(0, nb))
    m = nb + int(rng.integers(0, motif.number_of_nodes()))
    g.add_edge(b, m)
    return g


def _house() -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)])
    return g


def load_ba2motifs(n_graphs: int, seed: int = 0) -> List[Labeled]:
    """BA base + {house ⇒ class 1 | 5-cycle ⇒ class 0} (canonical BA-2Motifs)."""
    rng = np.random.default_rng(seed)
    out: List[Labeled] = []
    for i in range(n_graphs):
        motif, label = (_house(), 1) if i % 2 == 0 else (nx.cycle_graph(5), 0)
        out.append((_attach(_ba_base(15, rng), motif, rng), label))
    return out


def load_bamultishapes(n_graphs: int, seed: int = 0) -> List[Labeled]:
    """BA base + one shape; class 1 iff a house is present (simplified multi-shape)."""
    rng = np.random.default_rng(seed)
    shapes = [(_house(), 1), (nx.wheel_graph(6), 0), (_ba_grid(), 0)]
    out: List[Labeled] = []
    for i in range(n_graphs):
        motif, label = shapes[i % len(shapes)]
        out.append((_attach(_ba_base(15, rng), motif, rng), label))
    return out


def _ba_grid() -> nx.Graph:
    return nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))


def load_mutag(n_graphs: int = 150, seed: int = 0) -> List[Labeled]:
    """MUTAG via torch-geometric TUDataset (skip-if-import — never faked)."""
    try:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.utils import to_networkx
    except Exception as e:  # pragma: no cover - exercised only when PyG present
        raise ImportError(f"MUTAG requires torch-geometric: {e}")
    ds = TUDataset(root="data/TUDataset", name="MUTAG")
    out: List[Labeled] = []
    for d in list(ds)[:n_graphs]:
        g = to_networkx(d, to_undirected=True)
        out.append((nx.Graph(g), int(d.y.item())))
    return out


# Atom-type orderings for TUDataset one-hot node labels (dataset-specific).
_MUTAG_ATOMS = ["C", "N", "O", "F", "I", "Cl", "Br"]
_MUTAGENICITY_ATOMS = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I",
                        "Na", "K", "Li", "Ca"]


def _tu_labeled(name: str, atoms: List[str], n_graphs: int, seed: int) -> List[Labeled]:
    """Load a TUDataset molecular dataset as atom-labeled nx graphs.

    Sets node ``atom`` from the one-hot node label (argmax → ``atoms[idx]``), so the
    label-aware projection can match molecular motifs. VM-validated: the atom
    ordering must match the dataset's node-label encoding.
    """
    from torch_geometric.datasets import TUDataset  # noqa: F401
    import numpy as np

    ds = TUDataset(root="data/TUDataset", name=name)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_graphs]
    out: List[Labeled] = []
    for i in idx:
        d = ds[int(i)]
        g = nx.Graph()
        x = d.x.cpu().numpy() if d.x is not None else None
        for v in range(d.num_nodes):
            atom = atoms[int(x[v].argmax())] if x is not None and x.shape[1] >= len(atoms) else "?"
            g.add_node(v, atom=atom)
        ei = d.edge_index.cpu().numpy()
        for a, b in zip(ei[0], ei[1]):
            g.add_edge(int(a), int(b))
        out.append((g, int(d.y.item())))
    return out


def load_mutagenicity(n_graphs: int = 200, seed: int = 0) -> List[Labeled]:
    """Mutagenicity (TUDataset, ~4337 real molecules) as atom-labeled graphs."""
    try:
        import torch_geometric  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(f"Mutagenicity requires torch-geometric: {e}")
    return _tu_labeled("Mutagenicity", _MUTAGENICITY_ATOMS, n_graphs, seed)


def load_mutag_labeled(n_graphs: int = 150, seed: int = 0) -> List[Labeled]:
    """MUTAG with atom labels (for the label-aware projection)."""
    try:
        import torch_geometric  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(f"MUTAG requires torch-geometric: {e}")
    return _tu_labeled("MUTAG", _MUTAG_ATOMS, n_graphs, seed)


# atomic-number → symbol for the common organic set (BBBP / MoleculeNet)
_Z_SYMBOL = {6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl",
             35: "Br", 53: "I", 5: "B", 11: "Na", 1: "H"}


def load_bbbp(n_graphs: int = 200, seed: int = 0) -> List[Labeled]:
    """BBBP (MoleculeNet, real, in-the-wild) as atom-labeled graphs; NO motif oracle.

    PyG MoleculeNet featurizes atoms with x[:,0] = atomic number; we map that to a
    symbol. VM-validated (the exact x layout must be confirmed on the machine).
    """
    try:
        from torch_geometric.datasets import MoleculeNet
    except Exception as e:  # pragma: no cover
        raise ImportError(f"BBBP requires torch-geometric: {e}")
    import numpy as np

    ds = MoleculeNet(root="data/MoleculeNet", name="BBBP")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_graphs]
    out: List[Labeled] = []
    for i in idx:
        d = ds[int(i)]
        if d.num_nodes == 0 or d.edge_index is None:
            continue
        g = nx.Graph()
        x = d.x.cpu().numpy()
        for v in range(d.num_nodes):
            g.add_node(v, atom=_Z_SYMBOL.get(int(x[v, 0]), "?"))
        ei = d.edge_index.cpu().numpy()
        for a, b in zip(ei[0], ei[1]):
            g.add_edge(int(a), int(b))
        out.append((g, int(d.y.item()) if d.y is not None else 0))
    return out


def _bbbp_mols(seed: int):
    """Yield RDKit molecules from BBBP SMILES in a fixed shuffled order (skip-if-import)."""
    from torch_geometric.datasets import MoleculeNet
    from rdkit import Chem
    import numpy as np

    ds = MoleculeNet(root="data/MoleculeNet", name="BBBP")
    rng = np.random.default_rng(seed)
    for i in rng.permutation(len(ds)):
        smi = getattr(ds[int(i)], "smiles", None)
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumAtoms() == 0:
            continue
        yield m


def _mol_to_nx(m) -> nx.Graph:
    """RDKit mol -> atom-labelled nx graph (lowercase symbol = aromatic atom)."""
    g = nx.Graph()
    for a in m.GetAtoms():
        sym = a.GetSymbol()
        g.add_node(a.GetIdx(), atom=(sym.lower() if a.GetIsAromatic() else sym))
    for b in m.GetBonds():
        g.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return g


def _smarts_edges(m, patts) -> set:
    """Union of bonds whose both endpoints fall inside a match of any SMARTS in ``patts``."""
    edges = set()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    for p in patts:
        for match in m.GetSubstructMatches(p):
            s = set(match)
            for u, v in bonds:
                if u in s and v in s:
                    edges.add(frozenset((u, v)))
    return edges


def _molecular_motif_task(label_smarts, gt_smarts, n_graphs, seed) -> List[Labeled]:
    """Build a motif-classification task on real BBBP molecules.

    ``label_smarts`` = list of SMARTS ALL of which must match for the positive label;
    ``gt_smarts`` = list of SMARTS whose matched bonds form the ground-truth motif.
    Real molecules, real substructures, controlled ground truth (Sanchez-Lengeling-style).
    """
    try:
        from rdkit import Chem
    except Exception as e:  # pragma: no cover
        raise ImportError(f"molecular motif tasks require torch-geometric + rdkit: {e}")
    lab = [Chem.MolFromSmarts(s) for s in label_smarts]
    gtp = [Chem.MolFromSmarts(s) for s in gt_smarts]
    out: List[Labeled] = []
    for m in _bbbp_mols(seed):
        pos = all(m.HasSubstructMatch(p) for p in lab)
        g = _mol_to_nx(m)
        g.graph["motif_edges"] = _smarts_edges(m, gtp) if pos else set()
        out.append((g, int(pos)))
        if len(out) >= n_graphs:
            break
    return out


def load_benzene(n_graphs: int = 200, seed: int = 0) -> List[Labeled]:
    """Benzene benchmark (Sanchez-Lengeling NeurIPS 2020 style): real BBBP molecules
    labelled by benzene-ring presence, GT motif = the aromatic-carbon ring bonds."""
    return _molecular_motif_task(["c1ccccc1"], ["c1ccccc1"], n_graphs, seed)


def load_fluoride_carbonyl(n_graphs: int = 200, seed: int = 0) -> List[Labeled]:
    """Fluoride-Carbonyl benchmark: label = molecule has BOTH a fluoride and a carbonyl;
    GT motif = the C-F and C=O bonds."""
    return _molecular_motif_task(
        ["[F]", "[CX3]=[OX1]"], ["[#6][F]", "[CX3]=[OX1]"], n_graphs, seed
    )


def load_alkane_carbonyl(n_graphs: int = 200, seed: int = 0) -> List[Labeled]:
    """Alkane-Carbonyl benchmark: label = molecule has BOTH an unbranched aliphatic chain
    and a carbonyl; GT motif = the alkane chain and C=O bonds."""
    return _molecular_motif_task(
        ["[CX4;!R][CX4;!R][CX4;!R]", "[CX3]=[OX1]"],
        ["[CX4;!R][CX4;!R][CX4;!R]", "[CX3]=[OX1]"], n_graphs, seed,
    )


def load_dataset(name: str, n_graphs: int = 150, seed: int = 0) -> List[Labeled]:
    loaders = {
        "ba2motifs": load_ba2motifs,
        "bamultishapes": load_bamultishapes,
        "mutag": load_mutag,
        "mutag_labeled": load_mutag_labeled,
        "mutagenicity": load_mutagenicity,   # real molecular (TUDataset)
        "bbbp": load_bbbp,                    # real, in-the-wild (MoleculeNet), no oracle
        "benzene": load_benzene,             # real molecules, benzene-ring GT motif
        "fluoride_carbonyl": load_fluoride_carbonyl,
        "alkane_carbonyl": load_alkane_carbonyl,
    }
    if name not in loaders:
        raise ValueError(f"unknown dataset: {name!r}")
    return loaders[name](n_graphs, seed)


# datasets whose graphs carry atom labels → use the label-aware molecular vocabulary
_MOLECULAR = {"mutag_labeled", "mutagenicity", "bbbp"}


def is_molecular(name: str) -> bool:
    """True for real atom-labelled molecular datasets (where the fixed vocabulary
    degenerates and adaptive mining is applied)."""
    return name in _MOLECULAR


def vocab_for(name: str):
    """Return the concept vocabulary appropriate for a dataset: the benzene GT-motif
    vocabulary for the Benzene benchmark, label-aware molecular motifs for real
    molecules, structural motifs for the BA family."""
    from .project.vocab import (
        alkane_carbonyl_vocab,
        benzene_vocab,
        default_vocab,
        fluoride_carbonyl_vocab,
        molecular_vocab,
    )

    gt_vocabs = {
        "benzene": benzene_vocab,
        "fluoride_carbonyl": fluoride_carbonyl_vocab,
        "alkane_carbonyl": alkane_carbonyl_vocab,
    }
    if name in gt_vocabs:
        return gt_vocabs[name]()
    return molecular_vocab() if name in _MOLECULAR else default_vocab()
