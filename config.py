"""Hydra structured config for the harness (Hydra + OmegaConf).

All tunable parameters — paths, seeds, thresholds, the k-sweep — live here with
sensible defaults (S=30).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    root: str = "data"
    # ~150 graphs/dataset in the scope studied here; test-set only for aggregation.
    n_graphs: int = 150
    test_fraction: float = 0.5
    holdout: bool = False   # train GIN on train split, run diagnostic + report test acc on held-out


@dataclass
class ModelConfig:
    arch: str = "gin"          # single architecture in the scope studied here
    hidden_dim: int = 32
    n_layers: int = 3
    epochs: int = 100
    lr: float = 1e-3


@dataclass
class SeedsConfig:
    S: int = 30                # explainer seeds (axis i)
    floor: int = 10            # below this a comparison is labelled under-powered
    model_seed: int = 0        # fixed — model-init source of variance controlled out
    # model-seed axis (ii): train one GIN per seed; separates model- from explainer-variance
    model_seeds: List[int] = field(default_factory=lambda: [0])


@dataclass
class StatsConfig:
    alpha: float = 0.05                # paired Wilcoxon significance bar
    bootstrap_resamples: int = 10000
    chance_correct: bool = True        # observed − expected-under-random


@dataclass
class ProjectionConfig:
    """Concept-projection vocabulary (a data-adaptive extension).

    ``adaptive=True`` mines a per-dataset mid-frequency vocabulary instead of the
    fixed one, so the logical axis is informative on real molecular graphs. The
    admissibility gate (random-control logic < ``admissibility_tau``) decides, per
    dataset, whether the logical axis is testable — it is fixed in advance, not
    tuned post hoc.
    """
    adaptive: bool = False        # mine adaptive vocab for MOLECULAR datasets only
    mine_max_nodes: int = 3
    mine_band_lo: float = 0.15
    mine_band_hi: float = 0.85
    mine_max_concepts: int = 12
    mine_sample: int = 60
    mine_seed: int = 0
    admissibility_tau: float = 0.90
    aggregator: str = "default"   # "default" (CORELS/tree) | "onerule" (2nd-aggregator control)
    aggregator_alt: str = ""      # if set, ALSO compute logic under this aggregator (same cached
                                  # explanations) → stored as logic_alt; the 2nd-aggregator control in one pass


@dataclass
class ProjectConfig:
    """Root config. Nested groups: data, model, explainers, stats, seeds, k_sweep."""
    datasets: List[str] = field(
        default_factory=lambda: ["ba2motifs", "bamultishapes", "mutag"]
    )
    explainers: List[str] = field(
        default_factory=lambda: ["gnnexplainer", "pgexplainer", "subgraphx", "ig"]
    )
    controls: List[str] = field(
        default_factory=lambda: ["random", "ig_ceiling", "gt_oracle"]
    )
    # swept sparsity (axis d) — number of top edges kept
    k_sweep: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seeds: SeedsConfig = field(default_factory=SeedsConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    output_dir: str = "outputs"


def default_config() -> "OmegaConf":
    """Return the default config as an OmegaConf DictConfig (attribute access)."""
    return OmegaConf.structured(ProjectConfig())


def write_config(path: str = "conf/config.yaml") -> str:
    """Dump the default config to ``path`` (--write-config). Returns the path."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(OmegaConf.to_yaml(default_config()))
    return path
