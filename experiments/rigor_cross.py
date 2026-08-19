"""RIGOR confirmation of the cross-explainer logical contradiction: >=3 model seeds, seed-clustered
bootstrap CIs, k-sweep, both aggregators, with a faithfulness (GEA) skeptical check.

PRE-DECLARED (before running): a cross-explainer pair on a (dataset,k) is a ROBUST CONTRADICTION iff
in >=2/3 model seeds the seed-clustered 95% CI UPPER bound of cross logical-consistency is < 0.50
under BOTH aggregators (tree AND OneR). Report every pair whichever way it falls (no hunting).

Focus pair: PGExplainer vs IG (both reasonably faithful, fast). Floors: pg-vs-random, ig-vs-random.
Datasets: bamultishapes, mutag (candidates) + ba2motifs (negative control — expected to vanish
under OneR). Faithfulness GEA vs ground-truth motif reported for synthetic datasets, so we can tell a
meaningful disagreement from two unfaithful explainers. CPU, fresh model per explainer.
"""
import copy, itertools
import numpy as np
from gnnxc.data import load_dataset, vocab_for
from gnnxc.model import train_gin, predict
from gnnxc.explainers.registry import build_explainer
from gnnxc.pipeline import CachingExplainer, rules_over_seeds
from gnnxc.project.threshold import top_k_edges
from gnnxc.project.vocab import vocab_names
from gnnxc.layers.logical import non_contradiction

DATASETS = ["bamultishapes", "mutag", "ba2motifs"]
MODEL_SEEDS = [0, 1, 2]
KS = [10, 15]
S, N, B = 30, 60, 1000
rng = np.random.default_rng(0)


def cross_matrix(A, B, names):
    M = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            M[i, j] = 1.0 if non_contradiction([A[i], B[j]], names) else 0.0
    return M


def boot_ci(M):
    na, nb = M.shape
    vals = []
    for _ in range(B):
        ia = rng.integers(0, na, na); ib = rng.integers(0, nb, nb) if nb > 1 else np.array([0])
        vals.append(float(M[np.ix_(ia, ib)].mean()))
    return float(np.mean(M)), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def gea_of(ex, model_ds, k):
    sc = []
    for g, _ in model_ds:
        gt = g.graph.get("motif_edges")
        if not gt:
            return None
        gt = {tuple(sorted(e)) for e in gt}
        for s in range(S):
            tk = {tuple(sorted(e)) for e in top_k_edges(ex.explain(g, s), k)}
            u = tk | gt; sc.append(len(tk & gt) / len(u) if u else 1.0)
    return float(np.mean(sc))


print("PRE-DECLARED: robust contradiction iff >=2/3 model seeds have CI-upper<0.50 under BOTH aggregators.\n")
results = {}
for ds_name in DATASETS:
    vocab = vocab_for(ds_name); names = vocab_names(vocab)
    for ms in MODEL_SEEDS:
        ds = load_dataset(ds_name, N, ms)
        model, acc = train_gin(ds, epochs=100, seed=ms)
        model_ds = [(g, predict(model, g)) for g, _ in ds]
        exs = {}
        for e in ["pgexplainer", "ig", "random"]:
            m = copy.deepcopy(model)
            for p in m.parameters():
                p.requires_grad_(False)
            exs[e] = CachingExplainer(build_explainer(e, model=m, dataset=model_ds))
        for k in KS:
            R = {e: {"tree": rules_over_seeds(exs[e], model_ds, S, k, vocab, aggregator="default"),
                     "one": rules_over_seeds(exs[e], model_ds, S, k, vocab, aggregator="onerule")}
                 for e in exs}
            gea = {e: gea_of(exs[e], model_ds, k) for e in ["pgexplainer", "ig"]}
            for a, b in [("pgexplainer", "ig"), ("pgexplainer", "random"), ("ig", "random")]:
                pt, ct = boot_ci(cross_matrix(R[a]["tree"], R[b]["tree"], names))
                po, co = boot_ci(cross_matrix(R[a]["one"], R[b]["one"], names))
                results.setdefault((ds_name, k, a, b), []).append((ms, pt, ct, po, co))
                print(f"{ds_name:13} ms{ms} k{k} {a[:8]}-{b[:6]}: tree {pt:.3f} CI[{ct[0]:.2f},{ct[1]:.2f}] | "
                      f"oneR {po:.3f} CI[{co[0]:.2f},{co[1]:.2f}]  (GEA pg={gea['pgexplainer']} ig={gea['ig']})",
                      flush=True)

print("\n=== PRE-DECLARED verdicts ===")
for (ds_name, k, a, b), rows in sorted(results.items()):
    hits = sum(1 for (ms, pt, ct, po, co) in rows if ct[1] < 0.50 and co[1] < 0.50)
    verdict = "ROBUST CONTRADICTION" if hits >= 2 else "not robust"
    print(f"{ds_name:13} k{k} {a[:8]}-{b[:6]}: robust in {hits}/{len(rows)} model seeds -> {verdict}")
