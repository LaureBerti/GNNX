"""The single pipeline every explainer and control shares.

    explain(graph, seed) → threshold@k → subgraph-iso projection → concept vector
                         → deterministic aggregation → class rule R_s
    {R_s over seeds}     → statistical + logical self-consistency

Because projection and aggregation are deterministic, a **deterministic explainer
must read logical-consistency == 1** through the full pipeline on every dataset —
this is the IG-ceiling validator: if it ever dropped below 1, the
apparatus would be leaking noise and every downstream number would be suspect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

import networkx as nx

from .aggregate.rule import Rule, learn_rule
from .explainers.base import Explainer
from .layers.consistency import ConsistencyResult, logical_consistency
from .layers.statistical import chance_corrected_overlap
from .project.projection import project
from .project.threshold import threshold
from .project.vocab import default_vocab, vocab_names

Labeled = Tuple[nx.Graph, int]  # (graph, class label)


class CachingExplainer:
    """Memoize ``explain(graph, seed)`` so the logical and statistical layers
    share one explanation pass instead of recomputing (a free ~2x). Keyed by
    graph identity + seed; valid within a single (explainer, dataset) run."""

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "explainer")
        self._cache = {}

    def explain(self, graph: nx.Graph, seed: int):
        key = (id(graph), seed)
        if key not in self._cache:
            self._cache[key] = self.inner.explain(graph, seed)
        return self._cache[key]


def rule_for_seed(
    explainer: Explainer,
    dataset: Sequence[Labeled],
    seed: int,
    k: int,
    vocab: Mapping[str, nx.Graph] | None = None,
    aggregator: str = "default",
) -> Rule:
    """Lift one explainer seed into a single class rule over the test set."""
    vocab = vocab if vocab is not None else default_vocab()
    names = vocab_names(vocab)
    vectors, labels = [], []
    for graph, label in dataset:
        importance = explainer.explain(graph, seed)
        sub = threshold(importance, k, graph)  # carry atom/bond labels for label-aware concepts
        vectors.append(project(sub, vocab))
        labels.append(int(label))
    return learn_rule(vectors, labels, names, kind=aggregator)


def rules_over_seeds(
    explainer: Explainer,
    dataset: Sequence[Labeled],
    S: int,
    k: int,
    vocab: Mapping[str, nx.Graph] | None = None,
    aggregator: str = "default",
) -> List[Rule]:
    return [rule_for_seed(explainer, dataset, s, k, vocab, aggregator) for s in range(S)]


def explainer_consistency(
    explainer: Explainer,
    dataset: Sequence[Labeled],
    S: int,
    k: int,
    vocab: Mapping[str, nx.Graph] | None = None,
    seed_floor: int = 10,
    aggregator: str = "default",
) -> ConsistencyResult:
    """Logical self-consistency of one explainer on one dataset at sparsity k."""
    names = vocab_names(vocab or default_vocab())
    rules = rules_over_seeds(explainer, dataset, S, k, vocab, aggregator)
    return logical_consistency(rules, names, seed_floor=seed_floor)


def explanation_accuracy(
    explainer: Explainer,
    dataset: Sequence[Labeled],
    S: int,
    k: int,
) -> float | None:
    """Mean GEA (Jaccard of top-k explanation edges vs the ground-truth motif edges),
    averaged over graphs with a known motif and over seeds (standard explanation-accuracy
    metric, GraphXAI). Returns None if no graph carries ground-truth motif edges."""
    from .project.threshold import top_k_edges

    scores = []
    for graph, _ in dataset:
        gt = graph.graph.get("motif_edges")
        if not gt:
            continue
        gt = {frozenset(e) for e in gt}
        for s in range(S):
            topk = {frozenset(e) for e in top_k_edges(explainer.explain(graph, s), k)}
            union = topk | gt
            scores.append(len(topk & gt) / len(union) if union else 1.0)
    return sum(scores) / len(scores) if scores else None


def statistical_consistency(
    explainer: Explainer,
    dataset: Sequence[Labeled],
    S: int,
    k: int,
) -> float:
    """Mean pairwise chance-corrected top-k overlap of seed importances.

    Averaged over graphs and over the C(S,2) seed pairs; 1.0 for a deterministic
    explainer, ~0 for random. This is the statistical axis of the dissociation plane.
    """
    import itertools

    per_graph = []
    for graph, _ in dataset:
        edges = list(graph.edges())
        imps = [
            [explainer.explain(graph, s)[frozenset(e)] for e in edges] for s in range(S)
        ]
        pair_scores = [
            chance_corrected_overlap(imps[i], imps[j], k)
            for i, j in itertools.combinations(range(S), 2)
        ]
        if pair_scores:
            per_graph.append(sum(pair_scores) / len(pair_scores))
    return sum(per_graph) / len(per_graph) if per_graph else 1.0
