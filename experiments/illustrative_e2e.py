"""END-TO-END illustrative use of the diagnostic on the one confounder-robust cell:
PGExplainer / MUTAG / k=10. Shows the practitioner journey:
  (1) statistical audit says 'reproducible' (high cross-seed top-k overlap),
  (2) logical audit says 'inconsistent' (seed rules disagree on real molecules),
  (3) a CONCRETE contradiction: two seeds, two rules, one molecule, opposite predicted class,
  (4) confounder control: this survives the OneR aggregator swap => a REAL problem, not an artifact
      (contrast k=15, which vanishes under OneR => the tool correctly flags it as a false alarm).
Everything printed is extracted from the real run — no fabrication. CPU, isolated PGExplainer."""
from itertools import combinations
import numpy as np
from gnnxc.data import load_dataset, vocab_for
from gnnxc.model import train_gin, predict
from gnnxc.explainers.registry import build_explainer
from gnnxc.pipeline import (CachingExplainer, rules_over_seeds, statistical_consistency,
                            explainer_consistency)
from gnnxc.project.projection import project
from gnnxc.project.threshold import threshold
from gnnxc.project.vocab import vocab_names
from gnnxc.layers.logical import non_contradiction

S, N = 30, 60
ds = load_dataset("mutag", N, 0)
vocab = vocab_for("mutag"); names = vocab_names(vocab)
model, acc = train_gin(ds, epochs=100, seed=0)
mds = [(g, predict(model, g)) for g, _ in ds]
ex = CachingExplainer(build_explainer("pgexplainer", model=model, dataset=mds))  # isolated: pg only
CLS = {0: "non-mutagenic", 1: "mutagenic"}

def render(rule):
    parts = []
    for c in rule.clauses:
        ante = " AND ".join(f"{k}{'' if v else '=absent'}" for k, v in c.antecedent.items()) or "(any)"
        parts.append(f"IF {ante} THEN {CLS[c.label]}")
    parts.append(f"ELSE {CLS[rule.default]}")
    return "  " + "\n  ".join(parts)

print(f"=== END-TO-END: PGExplainer on MUTAG, GIN acc {acc:.2f}, S={S} seeds ===\n")

for K in (10, 15):
    stat = statistical_consistency(ex, mds, S, K)
    logic_tree = explainer_consistency(ex, mds, S, K, vocab=vocab, aggregator="default").rate
    logic_one = explainer_consistency(ex, mds, S, K, vocab=vocab, aggregator="onerule").rate
    verdict = ("ROBUST self-inconsistency (survives aggregator swap)"
               if (stat >= 0.8 and logic_tree <= 0.85 and logic_one <= 0.85)
               else "aggregator ARTIFACT (vanishes under OneR)" if (stat >= 0.8 and logic_tree <= 0.85)
               else "consistent")
    print(f"--- k={K} ---")
    print(f"  (1) statistical consistency (top-k overlap across seeds): {stat:.3f}  -> looks reproducible")
    print(f"  (2) logical consistency, decision-tree aggregator:        {logic_tree:.3f}")
    print(f"      logical consistency, OneR aggregator (2nd control):   {logic_one:.3f}")
    print(f"  (4) confounder verdict: {verdict}\n")

# (3) concrete contradiction at k=10 (the robust cell)
K = 10
rules = rules_over_seeds(ex, mds, S, K, vocab, aggregator="default")
molvecs = {}
for g, y in mds:
    cv = project(threshold(ex.explain(g, 0), K, g), vocab)
    molvecs[tuple(cv[n] for n in names)] = (g, y)
found = False
for i, j in combinations(range(S), 2):
    if non_contradiction([rules[i], rules[j]], names):
        continue
    for vec, (g, y) in molvecs.items():
        a = dict(zip(names, vec))
        pi, pj = rules[i].predict(a), rules[j].predict(a)
        if pi is not None and pj is not None and pi != pj:
            present = [n for n in names if a[n]]
            print("  (3) CONCRETE CONTRADICTION (same explainer, two seeds, one molecule):")
            print(f"      molecule concepts present: {present or '(none in vocab)'};  GIN predicts: {CLS[y]}")
            print(f"\n      seed {i} rule:\n{render(rules[i])}\n      -> predicts {CLS[pi]}")
            print(f"\n      seed {j} rule:\n{render(rules[j])}\n      -> predicts {CLS[pj]}")
            print(f"\n      => seeds {i} and {j} of the SAME explainer give OPPOSITE class logic "
                  f"for this molecule.")
            found = True
            break
    if found:
        break
if not found:
    print("  (3) no cross-seed contradiction found at k=10 (unexpected).")
