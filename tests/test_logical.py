import itertools

import pytest

from gnnxc.aggregate.rule import Clause, Rule
from gnnxc.layers.logical import (
    equivalent,
    equivalent_z3,
    graded_agreement,
    non_contradiction,
)

NAMES = ["cycle", "benzene"]


def _brute_agreement(r1, r2, names):
    total = agree = 0
    for bits in itertools.product((0, 1), repeat=len(names)):
        a = dict(zip(names, bits))
        total += 1
        agree += int(r1.predict(a) == r2.predict(a))
    return agree / total


def test_equivalent_syntactically_different():
    # cycle ⇒ 1 (default 0)
    r1 = Rule(clauses=[Clause({"cycle": 1}, 1)], default=0, names=NAMES)
    # split on benzene but same function
    r2 = Rule(
        clauses=[Clause({"cycle": 1, "benzene": 0}, 1), Clause({"cycle": 1, "benzene": 1}, 1)],
        default=0,
        names=NAMES,
    )
    assert equivalent(r1, r2, NAMES)


def test_contradiction_benzene():
    r1 = Rule(clauses=[Clause({"benzene": 1}, 1)], default=None, names=NAMES)
    r0 = Rule(clauses=[Clause({"benzene": 1}, 0)], default=None, names=NAMES)
    assert non_contradiction([r1, r1], NAMES) is True
    assert non_contradiction([r1, r0], NAMES) is False


def test_graded_agreement_matches_brute_force():
    r1 = Rule(clauses=[Clause({"cycle": 1}, 1)], default=0, names=NAMES)
    r2 = Rule(clauses=[Clause({"benzene": 1}, 1)], default=0, names=NAMES)
    assert graded_agreement(r1, r2, NAMES) == _brute_agreement(r1, r2, NAMES)


def test_z3_agrees_with_enumeration():
    try:
        import z3  # noqa: F401
    except Exception as e:  # broken/absent native libz3 → skip (enumeration is the default)
        pytest.skip(f"z3 unavailable: {e}")
    r1 = Rule(clauses=[Clause({"cycle": 1}, 1)], default=0, names=NAMES)
    r2 = Rule(
        clauses=[Clause({"cycle": 1, "benzene": 0}, 1), Clause({"cycle": 1, "benzene": 1}, 1)],
        default=0,
        names=NAMES,
    )
    r3 = Rule(clauses=[Clause({"benzene": 1}, 1)], default=0, names=NAMES)
    assert equivalent_z3(r1, r2, NAMES) == equivalent(r1, r2, NAMES) is True
    assert equivalent_z3(r1, r3, NAMES) == equivalent(r1, r3, NAMES) is False
