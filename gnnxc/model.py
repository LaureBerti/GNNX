"""GIN classifier + trainer.

Single architecture for the scope studied here. Node features are one-hot capped degree
(BA datasets ship no features), so GIN discriminates motifs by structure.
Import-guarded: the module imports cleanly only when torch/PyG are present; the
mock pipeline never touches it. Deterministic given a seed (fixed torch seeds).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np

try:  # torch/PyG are the optional [explainers] extra — skip-if-import
    import torch
    import torch.nn.functional as F
    from torch.nn import Linear, ReLU, Sequential
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINConv, global_add_pool

    _TORCH = True
except Exception:  # pragma: no cover - exercised only without torch
    _TORCH = False

Labeled = Tuple[nx.Graph, int]
_MAX_DEG = 10  # one-hot degree cap → feature dim


def _degree_features(g: nx.Graph):
    n = g.number_of_nodes()
    x = np.zeros((n, _MAX_DEG), dtype=np.float32)
    nodes = sorted(g.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    for v in nodes:
        d = min(g.degree(v), _MAX_DEG - 1)
        x[idx[v], d] = 1.0
    return x, idx


def nx_to_pyg(graph: nx.Graph, label: int) -> "Data":
    """Convert an (nx.Graph, label) into a PyG Data with one-hot degree features."""
    x_np, idx = _degree_features(graph)
    edges = [[idx[u], idx[v]] for u, v in graph.edges()]
    edges += [[v, u] for u, v in edges]  # undirected → both directions
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return Data(
        x=torch.tensor(x_np),
        edge_index=edge_index,
        y=torch.tensor([int(label)], dtype=torch.long),
    )


if _TORCH:

    class GIN(torch.nn.Module):
        def __init__(self, in_dim: int = _MAX_DEG, hidden: int = 32, n_layers: int = 3, n_classes: int = 2):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            d = in_dim
            for _ in range(n_layers):
                mlp = Sequential(Linear(d, hidden), ReLU(), Linear(hidden, hidden))
                self.convs.append(GINConv(mlp))
                d = hidden
            self.lin = Linear(hidden, n_classes)

        def forward(self, x, edge_index, batch=None):
            # batch defaults to a single graph → compatible with PyG's Explainer,
            # which calls model(x, edge_index) without a batch vector.
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            x = global_add_pool(x, batch)
            return self.lin(x)


def train_gin(
    dataset: Sequence[Labeled],
    epochs: int = 100,
    hidden: int = 32,
    n_layers: int = 3,
    lr: float = 1e-3,
    seed: int = 0,
) -> Tuple["GIN", float]:
    """Train a GIN; return (model, train_accuracy). Deterministic given seed."""
    if not _TORCH:  # pragma: no cover
        raise ImportError("train_gin requires torch + torch-geometric ([explainers] extra)")
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = [nx_to_pyg(g, y) for g, y in dataset]
    loader = DataLoader(data, batch_size=32, shuffle=True)
    model = GIN(hidden=hidden, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
    return model, evaluate(model, dataset)


def evaluate(model: "GIN", dataset: Sequence[Labeled]) -> float:
    if not _TORCH:  # pragma: no cover
        raise ImportError("evaluate requires torch + torch-geometric")
    model.eval()
    correct = 0
    with torch.no_grad():
        for g, y in dataset:
            d = nx_to_pyg(g, y)
            pred = model(d.x, d.edge_index, torch.zeros(d.x.size(0), dtype=torch.long)).argmax(dim=1)
            correct += int(pred.item() == y)
    return correct / len(dataset)


def predict(model: "GIN", graph: nx.Graph) -> int:
    d = nx_to_pyg(graph, 0)
    model.eval()
    with torch.no_grad():
        logits = model(d.x, d.edge_index, torch.zeros(d.x.size(0), dtype=torch.long))
    return int(logits.argmax(dim=1).item())
