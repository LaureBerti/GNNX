"""Second-aggregator control: the OneR aggregator (experiment #1). Torch-free."""
from gnnxc.aggregate.rule import learn_rule


def test_onerule_picks_the_predictive_concept():
    names = ["a", "b", "c"]
    vecs = [{"a": 1, "b": 0, "c": 1}, {"a": 1, "b": 1, "c": 0},
            {"a": 0, "b": 1, "c": 1}, {"a": 0, "b": 0, "c": 0}]
    y = [1, 1, 0, 0]  # concept 'a' predicts y perfectly
    r = learn_rule(vecs, y, names, kind="onerule")
    assert len(r.clauses) == 1 and r.clauses[0].antecedent == {"a": 1}
    assert r.predict({"a": 1, "b": 0, "c": 0}) == 1
    assert r.predict({"a": 0, "b": 1, "c": 1}) == 0


def test_onerule_is_deterministic_and_structurally_distinct():
    names = ["x", "y"]
    vecs = [{"x": 1, "y": 1}, {"x": 0, "y": 0}, {"x": 1, "y": 0}]
    lab = [1, 0, 1]
    a = learn_rule(vecs, lab, names, kind="onerule")
    b = learn_rule(vecs, lab, names, kind="onerule")
    # deterministic
    assert [(c.antecedent, c.label) for c in a.clauses] == [(c.antecedent, c.label) for c in b.clauses]
    # one-literal (OneR), unlike the multi-clause default decision list
    assert len(a.clauses) == 1


def test_default_and_onerule_can_differ():
    names = ["a", "b"]
    vecs = [{"a": 1, "b": 1}, {"a": 1, "b": 0}, {"a": 0, "b": 1}, {"a": 0, "b": 0}]
    y = [1, 0, 1, 0]  # b predicts y; a does not — both learners should still be deterministic
    one = learn_rule(vecs, y, names, kind="onerule")
    assert one.clauses[0].antecedent == {"b": 1}  # OneR finds b
