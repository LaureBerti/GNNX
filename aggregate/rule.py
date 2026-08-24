"""Deterministic class-rule aggregator.

Lifts a set of ``(concept-vector, predicted-label)`` pairs into ONE Boolean class
rule R: c(G) -> ŷ. The aggregator is held **constant** across all explainers, so
any inconsistency in {R_s} is attributable to the explainer/seed, not the
machinery. It MUST be deterministic (the IG-ceiling validator depends on
it) and environment-independent: the canonical aggregator is a fixed-seed
depth-capped decision tree lowered to a rule list, with OneR and a greedy
conjunction as the reported capacity controls — all order-stable and pure-Python.

A :class:`Rule` is an ordered rule list (first matching clause wins) plus a default
label. This same structure is consumed by the logical layer (equivalence,
non-contradiction, model-count) in :mod:`gnnxc.layers.logical`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Clause:
    """A conjunction of concept literals ⇒ a class label. Empty antecedent = always fires."""
    antecedent: Dict[str, int]  # concept name -> required 0/1
    label: int

    def matches(self, assignment: Mapping[str, int]) -> bool:
        return all(assignment.get(k, 0) == v for k, v in self.antecedent.items())


@dataclass
class Rule:
    """Ordered rule list with a default label (``None`` = abstain, i.e. partial rule)."""
    clauses: List[Clause] = field(default_factory=list)
    default: Optional[int] = None
    names: List[str] = field(default_factory=list)

    def predict(self, assignment: Mapping[str, int]) -> Optional[int]:
        for c in self.clauses:
            if c.matches(assignment):
                return c.label
        return self.default

    def fires(self, assignment: Mapping[str, int]) -> bool:
        """Whether the rule makes a prediction on this assignment (not abstain)."""
        return self.predict(assignment) is not None


def learn_rule(
    vectors: Sequence[Mapping[str, int]],
    labels: Sequence[int],
    names: Sequence[str],
    kind: str = "default",
) -> Rule:
    """Deterministically learn a class rule from concept vectors and labels.

    Three reproducible, pure-Python aggregators span three hypothesis families and
    are the reported capacity controls:

    * ``kind="default"`` / ``kind="tree"``: a fixed-seed depth-capped **decision
      tree** lowered to a rule list (decision-list boundary). This is the canonical
      aggregator; it does **not** depend on any optional native extension, so the
      logical-consistency reading is identical in every environment.
    * ``kind="onerule"``: a **OneR** single-best-concept learner (one literal).
    * ``kind="conj"``: a greedy **conjunction/monomial** learner (AND of present
      literals) — a distinct concept-learning family from both decision lists and
      OneR.

    A dissociation is reported only if it survives all three; if it is an artefact
    of one aggregator's boundary it will not. All three are deterministic.

    ``kind="corels"`` is available for opt-in comparison only (CORELS optimal rule
    lists); it is never used by ``default`` because it is an optional native
    extension that does not build in every environment, which would make the
    reading environment-dependent. ``kind="tree2"`` (depth-2 entropy tree) is a
    further opt-in variant.
    """
    names = list(names)
    X = np.asarray([[int(v.get(n, 0)) for n in names] for v in vectors], dtype=int)
    y = np.asarray([int(l) for l in labels], dtype=int)
    if X.shape[0] == 0:
        return Rule(clauses=[], default=0, names=names)

    if kind == "onerule":
        return _onerule_rule(X, y, names)
    if kind == "conj":
        return _conj_rule(X, y, names)
    if kind == "tree2":
        return _tree_rule(X, y, names, max_depth=2, criterion="entropy")
    if kind == "corels":  # opt-in only; never the default (see docstring)
        rule = _corels_rule(X, y, names)
        if rule is not None:
            return rule
    return _tree_rule(X, y, names)


def _onerule_rule(X, y, names) -> Rule:
    """OneR: pick the single concept whose presence/absence split best predicts y,
    emit a one-literal rule (present -> majority label there; default -> majority
    among absent). Deterministic; ties break on concept index."""
    default_all = int(np.bincount(y).argmax())
    best = None  # (accuracy, -index, concept, label_present, label_absent)
    for j, name in enumerate(names):
        col = X[:, j]
        pres, absc = y[col == 1], y[col == 0]
        lab_p = int(np.bincount(pres).argmax()) if pres.size else default_all
        lab_a = int(np.bincount(absc).argmax()) if absc.size else default_all
        correct = int((pres == lab_p).sum() + (absc == lab_a).sum())
        acc = correct / len(y)
        key = (acc, -j)
        if best is None or key > best[0]:
            best = (key, name, lab_p, lab_a)
    _, name, lab_p, lab_a = best
    return Rule(clauses=[Clause(antecedent={name: 1}, label=lab_p)],
                default=lab_a, names=list(names))


def _conj_rule(X, y, names, max_lits: int = 2) -> Rule:
    """Greedy monomial: grow a single conjunction (AND of up-to ``max_lits`` present
    literals) that best predicts the minority class, emit 'conjunction -> minority
    label' with default = majority. A canonical concept-learning primitive, in a
    different hypothesis family from both decision lists/trees and OneR. Deterministic;
    ties break on concept index."""
    default_all = int(np.bincount(y).argmax())
    target = int(1 - default_all)  # minority class
    ante: Dict[str, int] = {}
    mask = np.ones(len(y), dtype=bool)
    for _ in range(max_lits):
        best = None  # (accuracy_on_covered, -index, name)
        for j, name in enumerate(names):
            if name in ante:
                continue
            cand = mask & (X[:, j] == 1)
            if cand.sum() == 0:
                continue
            acc = (y[cand] == target).mean()
            key = (acc, -j)
            if best is None or key > best[0]:
                best = (key, name, j)
        if best is None:
            break
        _, name, j = best
        cand = mask & (X[:, j] == 1)
        # stop if adding this literal no longer purifies toward the target
        if (y[cand] == target).mean() < (y[mask] == target).mean():
            break
        ante[name] = 1
        mask = cand
    if not ante:  # no useful literal → constant rule
        return Rule(clauses=[], default=default_all, names=list(names))
    label = int(np.bincount(y[mask]).argmax()) if mask.any() else target
    return Rule(clauses=[Clause(antecedent=ante, label=label)],
                default=default_all, names=list(names))


def _corels_rule(X, y, names) -> Optional[Rule]:
    try:
        from corels import CorelsClassifier  # type: ignore
    except Exception:
        return None
    try:
        clf = CorelsClassifier(verbosity=[])
        clf.fit(X, y, features=list(names))
        # CorelsClassifier.rl() gives an ordered rule list; lower it to clauses.
        rl = clf.rl()
        clauses: List[Clause] = []
        default = int(np.bincount(y).argmax())
        for r in rl.rules[:-1]:  # last is the default rule
            ante = {names[a[0]]: 1 for a in r["antecedents"] if a[0] >= 0}
            clauses.append(Clause(antecedent=ante, label=int(r["prediction"])))
        default = int(rl.rules[-1]["prediction"])
        return Rule(clauses=clauses, default=default, names=names)
    except Exception:
        return None


def _tree_rule(X, y, names, max_depth: Optional[int] = None, criterion: str = "gini") -> Rule:
    from sklearn.tree import DecisionTreeClassifier

    depth = min(len(names), 4) if max_depth is None else max_depth
    clf = DecisionTreeClassifier(random_state=0, max_depth=depth, criterion=criterion)
    clf.fit(X, y)
    t = clf.tree_
    clauses: List[Clause] = []

    def recurse(node: int, ante: Dict[str, int]):
        if t.children_left[node] == t.children_right[node]:  # leaf
            label = int(np.argmax(t.value[node][0]))
            clauses.append(Clause(antecedent=dict(ante), label=label))
            return
        feat = names[t.feature[node]]  # binary features: threshold ~0.5
        left = dict(ante); left[feat] = 0   # feature <= 0.5  → absent
        recurse(t.children_left[node], left)
        right = dict(ante); right[feat] = 1  # feature > 0.5   → present
        recurse(t.children_right[node], right)

    recurse(0, {})
    default = int(np.bincount(y).argmax())
    # Order clauses deterministically (by antecedent items) so equal data → equal rule.
    clauses.sort(key=lambda c: (sorted(c.antecedent.items()), c.label))
    return Rule(clauses=clauses, default=default, names=names)
