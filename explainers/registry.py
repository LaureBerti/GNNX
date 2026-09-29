"""Explainer factory. Real adapters skip-if-import (return None) — never faked.

Mock/random stand-ins drive the zero-compute plumbing path; the real roster
(GNNExplainer, PGExplainer, SubgraphX, IG) is loaded only when torch-geometric/DIG
are importable and is filled in by the real adapters.
"""
from __future__ import annotations

from typing import Optional

from .base import Explainer
from .mock import MockExplainer

_MOCKS = {
    "mock": lambda: MockExplainer(mode="deterministic"),
    "ig_ceiling": lambda: MockExplainer(mode="deterministic"),
    "random": lambda: MockExplainer(mode="noisy"),
}

_REAL = {"gnnexplainer", "pgexplainer", "subgraphx", "ig", "gt_oracle"}
_MODEL_FREE = {"gt_oracle"}


def build_explainer(name: str, model=None, dataset=None) -> Optional[Explainer]:
    """Return an explainer for ``name`` or ``None`` if unavailable.

    Real adapters (gnnexplainer/ig/...) require a trained ``model``; PGExplainer
    also needs ``dataset`` to train its mask predictor; mock/random ignore both.
    """
    if name in _MOCKS:
        return _MOCKS[name]()
    if name in _REAL:
        try:
            from . import adapters

            return adapters.build(name, model=model, dataset=dataset)
        except Exception:
            return None
    raise ValueError(f"unknown explainer: {name!r}")


def needs_model(name: str) -> bool:
    """Whether this explainer requires a trained model to be bound."""
    return name in _REAL and name not in _MODEL_FREE
