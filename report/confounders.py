"""Confounder analysis for the self-consistency diagnostic (a reporting-standard component).

A cell shows an *apparent* statistical-vs-logical dissociation when statistical consistency is
high but logical consistency is low. Such a gap can be manufactured
by two confounds that have nothing to do with the explainer:

  1. EXPLAINER-ISOLATION confound — running another explainer first, in the same process, can
     corrupt a stochastic-training explainer (e.g. PGExplainer) through residual state on the
     shared model object. Controlled by running each explainer on a fresh model (see
     cli._fresh_model_for_explainer) or in its own process (experiments/clean_grid.py). This
     analysis assumes isolated inputs; it records the assumption so the reader can check it.

  2. AGGREGATOR-CAPACITY confound — lifting explanations to logical rules with a high-capacity
     learner (decision tree) can manufacture cross-seed contradictions that a low-capacity
     learner (OneR) does not. Controlled by reporting logical consistency under BOTH aggregators
     (``logic`` = primary, ``logic_alt`` = second aggregator) and flagging cells whose apparent
     dissociation does not survive the swap.

A dissociation is reported as ROBUST only if it is present under BOTH aggregators.
"""
from __future__ import annotations

from typing import List, Mapping, Optional


def confounder_analysis(
    points: List[Mapping],
    stat_hi: float = 0.80,
    logic_lo: float = 0.85,
    isolated: bool = True,
) -> dict:
    """Per-cell aggregator-confounder analysis + a summary of robust dissociations.

    ``points`` are report points carrying ``stat``, ``logic`` (primary aggregator) and, when the
    second-aggregator control was run, ``logic_alt``. A cell is an *apparent* dissociation if
    ``stat >= stat_hi and logic <= logic_lo``; it is *aggregator-robust* if ``logic_alt`` is also
    ``<= logic_lo`` (or missing → cannot clear the confound → not robust).
    """
    rows = []
    for p in points:
        stat = p.get("stat")
        logic = p.get("logic")
        alt = p.get("logic_alt")
        if stat is None or logic is None:
            continue
        apparent = (stat >= stat_hi) and (logic <= logic_lo)
        if not apparent:
            verdict = "consistent"
        elif alt is None:
            verdict = "apparent (no aggregator control run)"
        elif alt <= logic_lo:
            verdict = "ROBUST dissociation (survives aggregator swap)"
        else:
            verdict = "aggregator-artifact (vanishes under OneR)"
        rows.append({
            "explainer": p.get("explainer"), "dataset": p.get("dataset"), "k": p.get("k"),
            "stat": stat, "logic": logic, "logic_alt": alt,
            "gap": stat - logic, "apparent": apparent, "verdict": verdict,
        })
    robust = [r for r in rows if r["verdict"].startswith("ROBUST")]
    artifact = [r for r in rows if r["verdict"].startswith("aggregator-artifact")]
    uncontrolled = [r for r in rows if r["verdict"].startswith("apparent (")]
    return {
        "isolated_inputs": isolated,
        "thresholds": {"stat_hi": stat_hi, "logic_lo": logic_lo},
        "n_cells": len(rows),
        "n_apparent_dissociations": sum(1 for r in rows if r["apparent"]),
        "n_robust_dissociations": len(robust),
        "n_aggregator_artifacts": len(artifact),
        "n_uncontrolled": len(uncontrolled),
        "robust_dissociations": robust,
        "aggregator_artifacts": artifact,
        "rows": rows,
    }
