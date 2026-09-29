"""Round-2 strengthening: is the confounder-controlled NULL (no robust statistical-vs-logical
dissociation on real explainers) robust to the motif vocabulary V? We rerun the strongest real
candidate -- MUTAG / PGExplainer / k=10, 3 model seeds, both aggregators, S=30, seed-clustered CIs --
under three vocabularies: the default V (6 motifs), V without benzene (drop the 6-ring), and V plus a
4-cycle. Masks are vocab-independent, so we compute PGExplainer once per model seed and only re-project.
PRE-DECLARED (same criterion as the main grid, fixed before running): a cell is a ROBUST dissociation
iff in >=2/3 model seeds the seed-clustered 95%% CI has statistical lower bound > 0.70 AND logical
upper bound < 0.85 under BOTH aggregators. The null holds iff 0 robust cells under every vocabulary.
Report every number whichever way it falls."""
import copy, itertools, json
from collections import OrderedDict
import numpy as np
import networkx as nx
from gnnxc.data import load_dataset
from gnnxc.model import train_gin, predict
from gnnxc.explainers.registry import build_explainer
from gnnxc.pipeline import CachingExplainer
from gnnxc.project.projection import project
from gnnxc.project.threshold import threshold, top_k_edges
from gnnxc.project.vocab import default_vocab, vocab_names
from gnnxc.aggregate.rule import learn_rule
from gnnxc.layers.logical import non_contradiction
from gnnxc.layers.statistical import chance_corrected_overlap

DS, EX, K, S, N, B = "mutag", "pgexplainer", 10, 30, 60, 1000
MODEL_SEEDS = [0, 1, 2]
rng = np.random.default_rng(0)


def make_vocabs():
    base = default_vocab()
    v_drop = OrderedDict((k, v) for k, v in base.items() if k != "benzene")
    v_add = OrderedDict(base); v_add["cycle4"] = nx.cycle_graph(4)
    return {"default(6)": base, "drop-benzene(5)": v_drop, "add-4cycle(7)": v_add}


def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot(pw):
    S_ = pw.shape[0]; mask = ~np.eye(S_, dtype=bool); out = []
    for _ in range(B):
        idx = rng.integers(0, S_, S_); sub = pw[np.ix_(idx, idx)]
        out.append(float(sub[mask].mean()))
    return out


VOCABS = make_vocabs()
records = []
for ms in MODEL_SEEDS:
    ds = load_dataset(DS, N, ms)
    model, acc = train_gin(ds, epochs=100, seed=ms)
    md = [(g, predict(model, g)) for g, _ in ds]
    y = np.array([int(l) for _, l in md])
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    ex = CachingExplainer(build_explainer(EX, model=m, dataset=md))
    imps = {s: [ex.explain(g, s) for g, _ in md] for s in range(S)}
    edges = {gi: list(md[gi][0].edges()) for gi in range(len(md))}
    pw_stat = np.ones((S, S))
    for i, j in itertools.combinations(range(S), 2):
        ov = [chance_corrected_overlap([imps[i][gi][frozenset(e)] for e in edges[gi]],
                                       [imps[j][gi][frozenset(e)] for e in edges[gi]], K)
              for gi in range(len(md))]
        pw_stat[i, j] = pw_stat[j, i] = float(np.mean(ov)) if ov else 1.0
    stat = float(pw_stat[~np.eye(S, dtype=bool)].mean()); stat_ci = ci(boot(pw_stat))
    for vname, vocab in VOCABS.items():
        names = vocab_names(vocab)
        cmat = {s: np.array([[project(threshold(imps[s][gi], K, md[gi][0]), vocab)[n] for n in names]
                             for gi in range(len(md))], dtype=int) for s in range(S)}
        AGGS = ["tree", "onerule", "conj"]
        rules = {a: {s: learn_rule([dict(zip(names, r)) for r in cmat[s]], y, names, a) for s in range(S)}
                 for a in AGGS}
        off = ~np.eye(S, dtype=bool)
        logic = {}
        for a in AGGS:
            pw = np.ones((S, S))
            for i, j in itertools.combinations(range(S), 2):
                pw[i, j] = pw[j, i] = 1.0 if non_contradiction([rules[a][i], rules[a][j]], names) else 0.0
            logic[a] = (float(pw[off].mean()), ci(boot(pw)))
        robust = stat_ci[0] > 0.70 and all(logic[a][1][1] < 0.85 for a in AGGS)
        rec = {"ms": ms, "vocab": vname, "acc": round(float(acc), 3), "stat": round(stat, 3),
               "stat_ci": [round(x, 3) for x in stat_ci], "robust_cell": bool(robust)}
        for a in AGGS:
            rec[f"logic_{a}"] = round(logic[a][0], 3)
            rec[f"{a}_ci"] = [round(x, 3) for x in logic[a][1]]
        records.append(rec)
        print(f"ms{ms} {vname:16} stat {stat:.3f} | "
              f"tree {logic['tree'][0]:.3f} onerule {logic['onerule'][0]:.3f} conj {logic['conj'][0]:.3f} "
              f"| robust={robust}", flush=True)

with open("outputs/ees_vocab.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print("\n=== PRE-DECLARED: null holds iff 0 robust cells under EVERY vocabulary ===")
for vname in VOCABS:
    rows = [r for r in records if r["vocab"] == vname]
    nrob = sum(1 for r in rows if r["robust_cell"])
    print(f"  {vname:16}: robust cells = {nrob}/{len(rows)} model seeds -> {'NULL holds' if nrob == 0 else 'DISSOCIATION'}")
allnull = all(not r["robust_cell"] for r in records)
print(f"\n=> confounder-controlled null is {'VOCABULARY-ROBUST' if allnull else 'NOT vocabulary-robust'}")
