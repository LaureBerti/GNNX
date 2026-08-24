"""Logical layer — semantics of class rules.

Repurposes the logic layer as a *consistency-auditing instrument*: over the fixed
concept vocabulary V (|V| small), we enumerate all 2^|V| assignments exactly — an
exact model-count over all 2^|V| assignments, a sound,
deterministic decision procedure for equivalence and non-contradiction at this
scale (z3 is the drop-in for a larger-scale vocabulary; see :func:`equivalent_z3`).

Three measures on class rules R (:class:`gnnxc.aggregate.rule.Rule`):
- **equivalence**  R_i ⇔ R_j  : same predicted class on every assignment.
- **non-contradiction** : no assignment where two rules both fire with different labels.
- **graded agreement** : fraction of assignments on which they agree (model-count / 2^|V|).
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Sequence

from ..aggregate.rule import Rule


def _names(rules: Sequence[Rule], names: Sequence[str] | None) -> List[str]:
    if names is not None:
        return list(names)
    seen: List[str] = []
    for r in rules:
        for n in r.names:
            if n not in seen:
                seen.append(n)
        for c in r.clauses:
            for n in c.antecedent:
                if n not in seen:
                    seen.append(n)
    return seen


def _assignments(names: Sequence[str]):
    for bits in itertools.product((0, 1), repeat=len(names)):
        yield dict(zip(names, bits))


def equivalent(r1: Rule, r2: Rule, names: Sequence[str] | None = None) -> bool:
    """True iff r1 and r2 predict the same class on every assignment."""
    ns = _names([r1, r2], names)
    return all(r1.predict(a) == r2.predict(a) for a in _assignments(ns))


def non_contradiction(rules: Sequence[Rule], names: Sequence[str] | None = None) -> bool:
    """True iff no assignment has two rules firing with *different* labels.

    ``benzene⇒1`` vs ``benzene⇒0`` → False (both fire on {benzene:1}, differ).
    Rules that only ever abstain together, or agree, are non-contradictory.
    """
    ns = _names(rules, names)
    for a in _assignments(ns):
        preds = [r.predict(a) for r in rules]
        firing = [p for p in preds if p is not None]
        if len(set(firing)) > 1:
            return False
    return True


def non_contradiction_manifold(
    rules: Sequence[Rule],
    realized: Sequence[Mapping[str, int]],
    names: Sequence[str] | None = None,
) -> bool:
    """Manifold-restricted non-contradiction: like :func:`non_contradiction`, but
    the check ranges only over the concept vectors that real graphs actually
    realize (``realized``), not the full $2^{|V|}$ Boolean cube. This is the more
    defensible logical-consistency notion: it never charges a disagreement on a
    concept combination that no graph exhibits (pure rule-learner extrapolation).
    """
    for a in realized:
        preds = [r.predict(a) for r in rules]
        firing = [p for p in preds if p is not None]
        if len(set(firing)) > 1:
            return False
    return True


def graded_agreement(r1: Rule, r2: Rule, names: Sequence[str] | None = None) -> float:
    """Exact model-count agreement: fraction of 2^|V| assignments where preds match."""
    ns = _names([r1, r2], names)
    total = agree = 0
    for a in _assignments(ns):
        total += 1
        if r1.predict(a) == r2.predict(a):
            agree += 1
    return agree / total if total else 1.0


def equivalent_z3(r1: Rule, r2: Rule, names: Sequence[str] | None = None) -> bool:
    """z3-backed equivalence (journal-scale vocabulary). Semantics match :func:`equivalent`.

    Encodes each rule's predicted class as a nested if-then-else over Boolean
    concept vars and asserts non-equivalence is UNSAT.
    """
    import z3  # optional; enumeration is the default path

    ns = _names([r1, r2], names)
    vs = {n: z3.Bool(n) for n in ns}

    def encode(rule: Rule):
        expr = z3.IntVal(-1 if rule.default is None else rule.default)
        for c in reversed(rule.clauses):
            cond = z3.And(*[vs[k] if v else z3.Not(vs[k]) for k, v in c.antecedent.items()]) \
                if c.antecedent else z3.BoolVal(True)
            expr = z3.If(cond, z3.IntVal(c.label), expr)
        return expr

    s = z3.Solver()
    s.add(encode(r1) != encode(r2))
    return s.check() == z3.unsat
