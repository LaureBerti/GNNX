"""CLI: ``gnnxc --write-config`` and ``gnnxc run <hydra-overrides>``.

Config is Hydra/OmegaConf; overrides use dot-notation with no ``--`` prefix,
e.g. ``gnnxc run explainers=[mock] datasets=[ba2motifs] seeds.S=5``.
The mock roster runs end-to-end with zero GPU compute and emits ``report.json`` —
the plumbing proof that must pass before any real run.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List

from omegaconf import OmegaConf

from .config import default_config, write_config
from .data import load_dataset, vocab_for
from .explainers.registry import build_explainer, needs_model
from .pipeline import (
    CachingExplainer,
    explainer_consistency,
    explanation_accuracy,
    statistical_consistency,
)
from .report.certificate import certificate
from .report.dissociation import off_diagonal_proportion, plane


def _apply_overrides(overrides: List[str]):
    cfg = default_config()
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def _labeled_for_model(dataset, model):
    """Relabel each graph with the MODEL's prediction (ŷ)."""
    from .model import predict

    return [(g, predict(model, g)) for g, _ in dataset]


def _fresh_model_for_explainer(pristine_model):
    """Return a FRESH model object (new parameter tensors) for one explainer, so no explainer
    sees another's residual state on a shared ``model`` instance.

    Root-caused defect: running GNNExplainer before PGExplainer in one process made
    PGExplainer's per-seed logical self-consistency execution-order dependent (isolated it reads
    1.0; after a full GNNExplainer pass over the same model object it read 0.478 — reproducibly).
    The leak is carried by the shared ``model`` OBJECT: it is NOT the weights (reloading the
    state_dict did not fix it), NOT ``requires_grad`` (frozen vs unfrozen both read 1.0), NOT PyG
    edge-mask attributes (``_edge_mask``/``_explain`` are ``None`` after the run), and NOT a
    vanilla-PyG effect (a minimal PyG loop shows no contamination). Only a brand-new module object
    reads the correct, isolated result. So we hand each explainer a deep copy of the pristine
    trained model and reset global RNG to a fixed base; per-seed stochasticity is then driven
    solely by each adapter's own ``manual_seed(seed)``. (Process isolation — one interpreter per
    explainer, as in experiments/clean_grid.py — is the strongest form of the same guarantee.)
    """
    import copy

    import numpy as _np

    try:
        import torch

        torch.manual_seed(0)
    except Exception:
        pass
    _np.random.seed(0)
    if pristine_model is None:
        return None
    m = copy.deepcopy(pristine_model)
    try:
        m.zero_grad(set_to_none=True)
        m.eval()
    except Exception:
        pass
    return m


def run(overrides: List[str]) -> dict:
    cfg = _apply_overrides(overrides)
    n_graphs = int(cfg.data.n_graphs)
    S, floor = int(cfg.seeds.S), int(cfg.seeds.floor)

    points, certs, skipped, accuracies = [], [], [], []
    from .data import is_molecular
    from .project.mine import admissibility

    adaptive = bool(cfg.projection.adaptive)
    tau = float(cfg.projection.admissibility_tau)
    out_dir = str(cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    def _flush() -> dict:
        """Assemble and write report.json from results SO FAR. Called after each dataset
        so a kill/OOM keeps all completed units (resume-safety for the whole grid)."""
        from .report.confounders import confounder_analysis

        pts = plane(points)
        for p, pp in zip(points, pts):
            p["cell"] = pp.cell
        report = {
            "config": {"S": int(cfg.seeds.S), "k_sweep": [int(k) for k in cfg.k_sweep],
                       "n_graphs": n_graphs, "model_seeds": [int(s) for s in cfg.seeds.model_seeds],
                       "datasets": list(cfg.datasets), "explainers": list(cfg.explainers),
                       "adaptive_vocab": adaptive, "admissibility_tau": tau},
            "points": points,
            "off_diagonal_proportion": off_diagonal_proportion(pts),
            "logic_admissibility": admissibility(points, tau=tau),
            "confounder_analysis": confounder_analysis(
                points, isolated=True) if bool(cfg.projection.aggregator_alt) else None,
            "certificates": certs,
            "skipped_unavailable": skipped,
            "accuracies": accuracies,
        }
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return report
    holdout = bool(cfg.data.holdout)
    for ds_name in cfg.datasets:
        full_dataset = load_dataset(str(ds_name), n_graphs, int(cfg.seeds.model_seed))
        if holdout:
            import numpy as _np

            rng = _np.random.default_rng(0)
            idx = rng.permutation(len(full_dataset))
            cut = int(round((1.0 - float(cfg.data.test_fraction)) * len(full_dataset)))
            train_ds = [full_dataset[int(i)] for i in idx[:cut]]
            eval_ds = [full_dataset[int(i)] for i in idx[cut:]]
        else:
            train_ds = eval_ds = full_dataset
        gt_dataset = eval_ds
        if adaptive and is_molecular(str(ds_name)):
            from .project.mine import mine_vocab

            vocab = mine_vocab(
                [g for g, _ in train_ds],
                max_nodes=int(cfg.projection.mine_max_nodes),
                band=(float(cfg.projection.mine_band_lo), float(cfg.projection.mine_band_hi)),
                max_concepts=int(cfg.projection.mine_max_concepts),
                sample=int(cfg.projection.mine_sample),
                seed=int(cfg.projection.mine_seed),
            )
            print(f"[{ds_name}] mined adaptive vocab ({len(vocab)}): {list(vocab)}")
            if not vocab:
                skipped.append(f"{ds_name}(empty adaptive vocab — no informative concept)")
                continue
        else:
            vocab = vocab_for(str(ds_name))
        wants_model = any(needs_model(str(e)) for e in cfg.explainers)
        model_seeds = [int(s) for s in cfg.seeds.model_seeds]
        for ms in model_seeds:
            model, model_dataset = None, gt_dataset
            if wants_model:
                from .model import evaluate, train_gin

                model, tr_acc = train_gin(
                    train_ds, epochs=int(cfg.model.epochs), hidden=int(cfg.model.hidden_dim),
                    n_layers=int(cfg.model.n_layers), lr=float(cfg.model.lr), seed=ms,
                )
                te_acc = evaluate(model, eval_ds) if holdout else tr_acc
                accuracies.append({"dataset": str(ds_name), "model_seed": ms,
                                   "train_acc": tr_acc, "test_acc": te_acc, "holdout": holdout})
                print(f"[{ds_name}] model_seed={ms} GIN — train {tr_acc:.3f} test {te_acc:.3f}")
                model_dataset = _labeled_for_model(gt_dataset, model)

            import copy as _copy

            pristine_model = (_copy.deepcopy(model) if model is not None else None)
            for expl_name in cfg.explainers:
                expl_model = _fresh_model_for_explainer(pristine_model)
                ex = build_explainer(str(expl_name), model=expl_model, dataset=model_dataset)
                if ex is None:
                    skipped.append(f"{expl_name}@{ds_name}#{ms}")
                    continue
                if str(expl_name) == "gt_oracle" and not gt_dataset[0][0].graph.get("motif_edges"):
                    skipped.append(f"gt_oracle@{ds_name}(no ground-truth motif)")
                    continue
                ex = CachingExplainer(ex)
                dataset = model_dataset if needs_model(str(expl_name)) else gt_dataset
                for k in cfg.k_sweep:
                    k = int(k)
                    logic = explainer_consistency(ex, dataset, S, k, vocab=vocab, seed_floor=floor,
                                              aggregator=str(cfg.projection.aggregator))
                    alt = str(cfg.projection.aggregator_alt)
                    logic_alt = (explainer_consistency(ex, dataset, S, k, vocab=vocab, seed_floor=floor,
                                                       aggregator=alt).rate if alt else None)
                    stat = statistical_consistency(ex, dataset, S, k)
                    gea = explanation_accuracy(ex, dataset, S, k)
                    points.append(
                        {"explainer": str(expl_name), "dataset": str(ds_name), "k": k,
                         "model_seed": ms,
                         "stat": stat, "logic": logic.rate, "logic_ci": list(logic.ci),
                         "logic_alt": logic_alt,
                         "gap": stat - logic.rate, "under_powered": logic.under_powered,
                         "gea": gea}
                    )
                    certs.append(
                        certificate(
                            explainer=str(expl_name), dataset=str(ds_name), k=k,
                            stat=stat, logic=logic.rate, logic_ci=logic.ci,
                            n_seeds=S, under_powered=logic.under_powered,
                            ceiling=(1.0 if str(expl_name) in ("ig", "ig_ceiling", "mock") else None),
                        )
                    )
            _flush()
        _flush()

    report = _flush()
    path = os.path.join(out_dir, "report.json")
    print(f"wrote {path}  ({len(points)} points, skipped={skipped})")
    return report


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 0
    if argv[0] == "--write-config":
        path = argv[1] if len(argv) > 1 else "conf/config.yaml"
        print(f"wrote {write_config(path)}")
        return 0
    if argv[0] == "run":
        run(argv[1:])
        return 0
    print(f"unknown command: {argv[0]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
