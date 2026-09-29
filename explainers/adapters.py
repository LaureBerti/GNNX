"""Real explainer adapters — bound to a trained GIN.

STATUS: scaffold, **validation pending on a torch-capable machine** (this repo's
dev box is Intel macOS + Py3.13, for which no PyTorch wheel exists). The adapters
follow the PyG 2.x `torch_geometric.explain` API; they must be smoke-tested on a
torch-capable machine before any grid run. An adapter that fights integration past
~half a day is DROPPED from the roster and noted — never stubbed
with fake outputs.

Roster:
- ``ig``          Integrated Gradients (Captum) — deterministic → the ceiling/validator.
- ``gnnexplainer`` GNNExplainer — stochastic.
- ``random``      uniform random edge importance — chance floor.
- ``pgexplainer`` DEFERRED (needs a separate training phase over the dataset).
- ``subgraphx``   DEFERRED (DIG dependency; MCTS capped) — highest integration risk.

Every adapter exposes ``.explain(graph, seed) -> {frozenset({u,v}): float}``.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from .base import Explainer, Importance, canonical_edge

_DEFERRED = {"subgraphx"}


def build(name: str, model=None, dataset=None) -> Optional[Explainer]:
    """Return a real adapter bound to ``model``, or None if unavailable/deferred."""
    if name in _DEFERRED:
        return None
    if name == "random":
        return _RandomControl()
    if name == "gt_oracle":
        return _GTOracle()
    try:
        import torch
        from torch_geometric.explain import Explainer as PyGExplainer
    except Exception:
        return None
    if model is None:
        return None
    if name == "ig":
        return _CaptumIGAdapter(model)
    if name == "gnnexplainer":
        return _GNNExplainerAdapter(model)
    if name == "pgexplainer":
        if dataset is None:
            return None
        return _PGExplainerAdapter(model, dataset)
    return None


class _RandomControl:
    """Chance floor — seed-varying uniform random edge importance (no model needed)."""

    name = "random"

    def explain(self, graph: nx.Graph, seed: int) -> Importance:
        rng = np.random.default_rng(seed)
        return {canonical_edge(u, v): float(rng.random()) for u, v in graph.edges()}


class _GTOracle:
    """Ground-truth-motif oracle (correctness anchor). Returns importance 1
    on the planted-motif edges, 0 elsewhere — a perfect, deterministic (seed-
    invariant) explanation. Reads ``graph.graph['motif_edges']`` (tagged by the
    synthetic generators); on graphs without ground truth (e.g. MUTAG) it has no
    motif to point to and must be skipped, never faked."""

    name = "gt_oracle"

    def has_ground_truth(self, graph: nx.Graph) -> bool:
        return bool(graph.graph.get("motif_edges"))

    def explain(self, graph: nx.Graph, seed: int) -> Importance:
        motif = graph.graph.get("motif_edges", set())
        return {
            canonical_edge(u, v): (1.0 if canonical_edge(u, v) in motif else 0.0)
            for u, v in graph.edges()
        }


class _TorchAdapterBase:
    def __init__(self, model):
        self.model = model

    def _to_data(self, graph: nx.Graph):
        from ..model import nx_to_pyg

        return nx_to_pyg(graph, 0)

    def _edge_importance(self, graph, edge_index, edge_mask) -> Importance:
        """Map a PyG edge_mask (over directed edge_index) back to undirected keys."""
        imp: Importance = {}
        ei = edge_index.cpu().numpy()
        em = edge_mask.detach().cpu().numpy()
        for j in range(ei.shape[1]):
            key = canonical_edge(int(ei[0, j]), int(ei[1, j]))
            imp[key] = max(imp.get(key, 0.0), float(em[j]))
        return imp


class _allow_unused_grads:
    """Context manager: make ``torch.autograd.grad`` tolerate unused inputs.

    captum's exact-gradient IG calls ``torch.autograd.grad`` without
    ``allow_unused``; for a trained GIN some edges/features genuinely have no
    gradient path and torch raises "differentiated Tensor not used in the graph".
    An unused input's correct attribution is **zero**, so we enable allow_unused
    and substitute zeros for the ``None`` gradients — no bias, no skipped graphs.
    """

    def __enter__(self):
        import torch

        self._orig = torch.autograd.grad

        def patched(outputs, inputs, *a, **k):
            k["allow_unused"] = True
            grads = self._orig(outputs, inputs, *a, **k)
            return tuple(
                torch.zeros_like(inp) if g is None else g
                for inp, g in zip(inputs, grads)
            )

        torch.autograd.grad = patched
        return self

    def __exit__(self, *exc):
        import torch

        torch.autograd.grad = self._orig
        return False


class _CaptumIGAdapter(_TorchAdapterBase):
    """Integrated Gradients — deterministic across seeds (the ceiling/validator)."""

    name = "ig"

    def explain(self, graph: nx.Graph, seed: int) -> Importance:
        import torch
        from torch_geometric.explain import Explainer as PyGExplainer
        from torch_geometric.explain import CaptumExplainer, ModelConfig

        torch.manual_seed(seed)
        data = self._to_data(graph)
        explainer = PyGExplainer(
            model=self.model,
            algorithm=CaptumExplainer("IntegratedGradients"),
            explanation_type="model",
            edge_mask_type="object",
            node_mask_type=None,
            model_config=ModelConfig(mode="multiclass_classification", task_level="graph", return_type="raw"),
        )
        with _allow_unused_grads():
            expl = explainer(data.x, data.edge_index)
        return self._edge_importance(graph, data.edge_index, expl.edge_mask)


class _PGExplainerAdapter(_TorchAdapterBase):
    """PGExplainer — a parametric mask predictor trained over the dataset per seed.

    Because the mask predictor is *shared* across graphs, its masks are far more
    seed-stable than GNNExplainer's per-graph optimization — the prime candidate
    to populate the dangerous stat-consistent ∧ logic-contradictory cell. The
    per-seed trained explainer is cached so each seed trains once, then explains
    all graphs cheaply.
    """

    name = "pgexplainer"

    def __init__(self, model, dataset, epochs: int = 100, lr: float = 0.003):
        super().__init__(model)
        self.dataset = list(dataset)
        self.epochs = epochs
        self.lr = lr
        self._trained = {}

    def _target(self, data):
        import torch

        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long))
        return out.argmax(dim=1)

    def _explainer_for_seed(self, seed: int):
        import torch
        from torch_geometric.explain import Explainer as PyGExplainer
        from torch_geometric.explain import ModelConfig, PGExplainer

        if seed in self._trained:
            return self._trained[seed]
        torch.manual_seed(seed)
        explainer = PyGExplainer(
            model=self.model,
            algorithm=PGExplainer(epochs=self.epochs, lr=self.lr),
            explanation_type="phenomenon",
            edge_mask_type="object",
            node_mask_type=None,
            model_config=ModelConfig(mode="multiclass_classification", task_level="graph", return_type="raw"),
        )
        data_list = [self._to_data(g) for g, _ in self.dataset]
        targets = [self._target(d) for d in data_list]
        for epoch in range(self.epochs):
            for d, tgt in zip(data_list, targets):
                batch = torch.zeros(d.x.size(0), dtype=torch.long)
                explainer.algorithm.train(
                    epoch, self.model, d.x, d.edge_index, target=tgt, batch=batch
                )
        self._trained[seed] = explainer
        return explainer

    def explain(self, graph: nx.Graph, seed: int) -> Importance:
        import torch

        explainer = self._explainer_for_seed(seed)
        data = self._to_data(graph)
        batch = torch.zeros(data.x.size(0), dtype=torch.long)
        target = self._target(data)
        expl = explainer(data.x, data.edge_index, target=target, batch=batch)
        return self._edge_importance(graph, data.edge_index, expl.edge_mask)


class _GNNExplainerAdapter(_TorchAdapterBase):
    """GNNExplainer — stochastic (seed-dependent mask optimization)."""

    name = "gnnexplainer"

    def explain(self, graph: nx.Graph, seed: int) -> Importance:
        import torch
        from torch_geometric.explain import Explainer as PyGExplainer
        from torch_geometric.explain import GNNExplainer, ModelConfig

        torch.manual_seed(seed)
        data = self._to_data(graph)
        explainer = PyGExplainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            edge_mask_type="object",
            node_mask_type="attributes",
            model_config=ModelConfig(mode="multiclass_classification", task_level="graph", return_type="raw"),
        )
        expl = explainer(data.x, data.edge_index)
        return self._edge_importance(graph, data.edge_index, expl.edge_mask)
