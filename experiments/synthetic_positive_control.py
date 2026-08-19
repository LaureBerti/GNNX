"""POSITIVE CONTROL: construct a synthetic dataset where a statistical-vs-logical dissociation
EXISTS BY DESIGN, and show the diagnostic DETECTS it (and that it survives the aggregator control).
Pairs with the specificity evidence (IG ceiling = 1.0; confounder nulls on real data): this is the
SENSITIVITY evidence.

Design. Every graph carries, on nodes 0..5, BOTH a 6-ring (benzene = {01,12,23,34,45,50}) and a
5-ring (cycle = {01,12,23,34,40}); they share 4 edges. Two groups: A (label 1), B (label 0), with
identical ring structure — the label is an independent target, not readable from the ring alone.

Two explainers, both go through the REAL pipeline (threshold -> project -> aggregate -> logic):
  DISSOCIATOR : for a group-A graph highlight benzene on even seeds, cycle on odd seeds (same
                molecule, flipped concept); for group-B always cycle. High mask overlap (rings share
                edges) but the per-seed CLASS RULE flips -> logical contradiction by construction.
  CONSISTENT  : always highlight by true class (benzene for A, cycle for B) -> should read logic 1.0
                (negative control: the diagnostic must NOT cry wolf).

Report statistical vs logical (decision-tree AND OneR) consistency + the confounder verdict.
"""
import networkx as nx
import numpy as np
from gnnxc.explainers.base import canonical_edge
from gnnxc.pipeline import CachingExplainer, explainer_consistency, statistical_consistency
from gnnxc.project.vocab import default_vocab, vocab_names
from gnnxc.report.confounders import confounder_analysis

# Shared 5-edge path 0-1-2-3-4-5; the two concepts differ by ONE closing edge:
#   benzene (6-ring) = path + (5,0);   cycle (5-ring) = path + (4,0).
# So the highlighted masks overlap 5/6 (high statistical consistency) yet the detected concept
# flips (benzene XOR cycle) -> logical contradiction driven by a single edge.
PATH = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
BENZ = [canonical_edge(u, v) for u, v in PATH + [(5, 0)]]       # -> 6-ring (benzene)
CYC = [canonical_edge(u, v) for u, v in PATH + [(4, 0)]]        # -> 5-ring (cycle)


def make_graph(gid, group, rng):
    g = nx.Graph()
    g.add_edges_from(PATH + [(5, 0), (4, 0)])  # both closing edges present -> both rings available
    # small random background so statistical overlap is non-trivial (nodes 6..10)
    for _ in range(4):
        u, v = int(rng.integers(6, 11)), int(rng.integers(0, 11))
        if u != v:
            g.add_edge(u, v)
    g.graph["group"] = group
    return g


def make_dataset(n=40, seed=0):
    rng = np.random.default_rng(seed)
    ds = []
    for i in range(n):
        group = "A" if i % 2 == 0 else "B"
        ds.append((make_graph(i, group, rng), 1 if group == "A" else 0))
    return ds


class PlantedExplainer:
    def __init__(self, mode, base_seed=999):
        self.mode = mode; self.base_seed = base_seed; self.name = f"planted-{mode}"

    def explain(self, graph, seed):
        rng = np.random.default_rng(self.base_seed + seed)
        imp = {canonical_edge(u, v): float(rng.random()) * 0.1 for u, v in graph.edges()}
        grp = graph.graph["group"]
        if self.mode == "dissociator":
            hi = (BENZ if seed % 2 == 0 else CYC) if grp == "A" else CYC
        else:  # consistent: highlight by TRUE class
            hi = BENZ if grp == "A" else CYC
        for e in hi:
            if e in imp:
                imp[e] += 1.0
        return imp


S, K = 30, 6
ds = make_dataset(40, seed=0)
vocab = default_vocab(); names = vocab_names(vocab)
print(f"synthetic positive control: n={len(ds)}, S={S}, k={K}, concepts={names}\n")

points = []
for mode in ("dissociator", "consistent"):
    ex = CachingExplainer(PlantedExplainer(mode))
    stat = statistical_consistency(ex, ds, S, K)
    lt = explainer_consistency(ex, ds, S, K, vocab=vocab, aggregator="default").rate
    lo = explainer_consistency(ex, ds, S, K, vocab=vocab, aggregator="onerule").rate
    points.append({"explainer": ex.name, "dataset": "synthetic", "k": K,
                   "stat": stat, "logic": lt, "logic_alt": lo})
    print(f"[{mode:11}] statistical={stat:.3f}  logical(tree)={lt:.3f}  logical(OneR)={lo:.3f}")

ca = confounder_analysis(points, isolated=True)
print(f"\nconfounder analysis: apparent={ca['n_apparent_dissociations']} "
      f"ROBUST={ca['n_robust_dissociations']} artifacts={ca['n_aggregator_artifacts']}")
for r in ca["rows"]:
    print(f"  {r['explainer']:18} verdict: {r['verdict']}")
print("\nExpect: dissociator -> stat HIGH, logic LOW under BOTH aggregators -> ROBUST (true positive);"
      "\n        consistent  -> logic ~1.0 -> no false alarm (specificity).")
