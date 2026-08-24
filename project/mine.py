"""Data-adaptive concept-vocabulary mining (a data-adaptive extension).

**Why this exists.** The fixed molecular vocabulary (``molecular_vocab``) contains
near-universal concepts (single ``N``/``O`` atoms, ``carbonyl``, ``ring6``) that fire
on almost every top-k subgraph of a real molecule. The resulting concept vectors are
near-constant, so the deterministic rule-learner returns a trivial majority-class rule
for *every* seed — collapsing the logical-consistency axis to 1.0 even for the random
control (random logic 0.83 on real data, 85% of cells saturated). A measure
on which random explanations pass is **non-informative**, so the logical axis is
untestable there.

**The fix.** Mine a per-dataset vocabulary of connected labelled subgraphs whose
document-frequency lies in an informative middle band ``[lo, hi]`` — excluding both the
near-universal concepts (df→1) and the near-absent ones (df→0). Selection is
**class-agnostic** (frequency only, never class-discriminativeness): we build an
informative instrument, not one engineered to produce any particular H5 outcome.

Mining is deterministic and torch-free: same graphs + seed ⇒ identical vocabulary.
The vocabulary is mined ONCE per dataset (fixed seed) and frozen across all explainer
seeds, preserving cross-seed rule comparability. Whether the axis is actually
informative on a given dataset is then checked by the pre-registered admissibility
gate (random-control logic < ``tau``); mining makes admissibility *possible*, it does
not assume it.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np


def _connected_node_subsets(g: nx.Graph, max_nodes: int) -> List[frozenset]:
    """All connected node subsets of ``g`` of size 2..``max_nodes`` (deterministic)."""
    result: set = set()
    visited: set = set()

    def extend(cur: frozenset) -> None:
        if cur in visited:
            return
        visited.add(cur)
        if len(cur) >= 2:
            result.add(cur)
        if len(cur) >= max_nodes:
            return
        frontier: set = set()
        for n in cur:
            frontier |= set(g.neighbors(n))
        frontier -= cur
        for w in frontier:
            extend(cur | {w})

    for u in g.nodes():
        extend(frozenset([u]))
    return [s for s in result if len(s) >= 2]


def _has_atom_labels(graphs: Sequence[nx.Graph]) -> bool:
    return any("atom" in d for g in graphs for _, d in g.nodes(data=True))


def _canon_hash(sub: nx.Graph, node_attr) -> str:
    return nx.weisfeiler_lehman_graph_hash(sub, node_attr=node_attr, iterations=3)


def _template_name(t: nx.Graph, idx: int, labeled: bool) -> str:
    """A readable, deterministic concept name from a template subgraph."""
    ne = t.number_of_edges()
    nn = t.number_of_nodes()
    if labeled:
        atoms = "".join(sorted(str(d.get("atom", "?")) for _, d in t.nodes(data=True)))
        return f"m{idx}_{atoms}_{nn}n{ne}e"
    return f"m{idx}_{nn}n{ne}e"


def mine_vocab(
    graphs: Sequence[nx.Graph],
    max_nodes: int = 3,
    band: Tuple[float, float] = (0.2, 0.8),
    max_concepts: int = 10,
    sample: int = 60,
    seed: int = 0,
) -> "OrderedDict[str, nx.Graph]":
    """Mine a data-adaptive, mid-frequency concept vocabulary from ``graphs``.

    Enumerates connected (labelled) subgraphs up to ``max_nodes`` across a fixed
    ``sample`` of graphs, groups them by label-aware Weisfeiler-Lehman hash, and keeps
    the concepts whose document-frequency falls in ``band`` — ordered most-informative
    first (closest to df=0.5, i.e. maximal across-graph variance). Class labels are
    never consulted. Returns at most ``max_concepts`` templates.
    """
    graphs = list(graphs)
    if not graphs:
        return OrderedDict()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs))[: min(sample, len(graphs))]
    sample_graphs = [graphs[int(i)] for i in idx]
    labeled = _has_atom_labels(sample_graphs)
    node_attr = "atom" if labeled else None

    support: Dict[str, int] = {}
    template: Dict[str, nx.Graph] = {}
    N = len(sample_graphs)
    for g in sample_graphs:
        seen: set = set()
        for nodes in _connected_node_subsets(g, max_nodes):
            sub = g.subgraph(nodes).copy()
            h = _canon_hash(sub, node_attr)
            seen.add(h)
            if h not in template:
                t = nx.convert_node_labels_to_integers(sub)
                # keep only the atom attribute on the template (drop stray attrs)
                if labeled:
                    for _, d in t.nodes(data=True):
                        for key in list(d):
                            if key != "atom":
                                del d[key]
                else:
                    for _, d in t.nodes(data=True):
                        d.clear()
                for _, _, d in t.edges(data=True):
                    d.clear()
                template[h] = t
        for h in seen:
            support[h] = support.get(h, 0) + 1

    lo, hi = band
    cand = [(h, support[h] / N) for h in support if lo <= support[h] / N <= hi]
    # informative-first: closest to df=0.5 (max variance); tie-break higher df, then hash
    cand.sort(key=lambda x: (abs(x[1] - 0.5), -x[1], x[0]))
    vocab: "OrderedDict[str, nx.Graph]" = OrderedDict()
    for i, (h, _df) in enumerate(cand[:max_concepts]):
        vocab[_template_name(template[h], i, labeled)] = template[h]
    return vocab


def admissibility(points: Sequence[dict], tau: float = 0.90) -> Dict[str, dict]:
    """Per-dataset logic-axis admissibility from the random control (pre-registered gate).

    A dataset's logical axis is admissible for H5 only if the *random* control's mean
    logical-consistency is below ``tau`` — i.e. random explanations are contradictory
    often enough that the axis can discriminate. This is the floor analogue of the
    IG-ceiling validator. Datasets with no random control are marked ``unknown``.
    """
    from collections import defaultdict

    by_ds: "defaultdict[str, List[float]]" = defaultdict(list)
    for p in points:
        if p.get("explainer") == "random":
            by_ds[p["dataset"]].append(float(p["logic"]))
    out: Dict[str, dict] = {}
    datasets = {p["dataset"] for p in points}
    for ds in sorted(datasets):
        vals = by_ds.get(ds, [])
        if not vals:
            out[ds] = {"random_logic_mean": None, "admissible": None, "reason": "no random control"}
            continue
        mean = sum(vals) / len(vals)
        out[ds] = {"random_logic_mean": mean, "admissible": bool(mean < tau), "tau": tau}
    return out
