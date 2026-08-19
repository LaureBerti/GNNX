from gnnxc.aggregate.rule import Clause, Rule
from gnnxc.layers.consistency import logical_consistency, wilson_ci

NAMES = ["cycle", "benzene"]


def _rule(label_on_benzene):
    # benzene ⇒ label ; else 0
    return Rule(clauses=[Clause({"benzene": 1}, label_on_benzene)], default=0, names=NAMES)


def test_deterministic_rules_rate_one():
    rules = [_rule(1) for _ in range(30)]
    res = logical_consistency(rules, NAMES)
    assert res.rate == 1.0
    assert res.n_pairs == 30 * 29 // 2
    assert not res.under_powered


def test_known_contradiction_fraction():
    # 24 rules predict benzene⇒1, 6 predict benzene⇒0. A pair is contradictory
    # iff it mixes the two groups: 24*6 mixed pairs out of C(30,2).
    rules = [_rule(1) for _ in range(24)] + [_rule(0) for _ in range(6)]
    res = logical_consistency(rules, NAMES)
    n_pairs = 30 * 29 // 2
    contradictory = 24 * 6
    expected_rate = (n_pairs - contradictory) / n_pairs
    assert abs(res.rate - expected_rate) < 1e-9
    assert res.ci[0] <= res.rate <= res.ci[1]


def test_under_powered_flag():
    rules = [_rule(1) for _ in range(4)]
    assert logical_consistency(rules, NAMES, seed_floor=10).under_powered


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(30, 30)
    assert 0.0 <= lo <= 1.0 and hi == 1.0
