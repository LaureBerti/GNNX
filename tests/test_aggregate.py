from gnnxc.aggregate.rule import Clause, Rule, learn_rule
from gnnxc.project.vocab import vocab_names


def _dataset_cycle_separable():
    names = vocab_names()
    vectors, labels = [], []
    for cyc in (0, 1):
        for benz in (0, 1):
            v = {n: 0 for n in names}
            v["cycle"], v["benzene"] = cyc, benz
            vectors.append(v)
            labels.append(cyc)  # class 1 iff cycle present
    return vectors * 5, labels * 5, names


def test_determinism_same_rule_twice():
    vectors, labels, names = _dataset_cycle_separable()
    r1 = learn_rule(vectors, labels, names)
    r2 = learn_rule(vectors, labels, names)
    assert r1 == r2


def test_recovers_cycle_implies_class1():
    vectors, labels, names = _dataset_cycle_separable()
    rule = learn_rule(vectors, labels, names)
    # every assignment with cycle=1 predicts 1, cycle=0 predicts 0
    for benz in (0, 1):
        base = {n: 0 for n in names}
        assert rule.predict({**base, "cycle": 1, "benzene": benz}) == 1
        assert rule.predict({**base, "cycle": 0, "benzene": benz}) == 0


def test_clause_matches():
    c = Clause(antecedent={"cycle": 1}, label=1)
    assert c.matches({"cycle": 1, "benzene": 0})
    assert not c.matches({"cycle": 0})
