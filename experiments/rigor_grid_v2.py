"""Regenerated rigor grid (round-3): THREE reproducible aggregators + more model seeds, so every
paper number traces to one committed CPU environment. Fixes the reproducibility defect a reviewer
caught: the old 'default' aggregator was CORELS-first with a decision-tree fallback, so its logical
reading changed depending on whether the CORELS native extension was installed (0.648 with CORELS,
0.602 with the tree fallback). CORELS does not build in the pinned .venv-cpu, so we drop it and report
three pure-Python aggregators spanning three hypothesis families:
  tree   = fixed-seed depth-capped decision tree (decision-list boundary)  [canonical]
  onerule= OneR single-literal
  conj   = greedy conjunction / monomial (AND of present literals)
All three are deterministic and environment-independent.

PRE-DECLARED robust criterion (fixed before running; no HARKing): a cell is a robust dissociation iff
in >= ceil(2/3) of its model seeds the seed-clustered 95% CI has statistical LOWER bound > 0.70 AND
logical UPPER bound < 0.85 under ALL THREE aggregators. Report every cell whichever way it falls.

Seeds: PGExplainer is the only explainer that ever produced a candidate dissociation, so it is the
power-critical arm and gets --pg-seeds model seeds (default 10); IG (calibration ceiling), random
(floor) and GNNExplainer get --other-seeds (default 5). Masks are computed once per
(dataset, model_seed, explainer) and reused across k. Incremental JSONL out.
"""
import copy, itertools, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
import numpy as np
from gnnxc.data import load_dataset, vocab_for
from gnnxc.model import train_gin, predict
from gnnxc.explainers.registry import build_explainer
from gnnxc.pipeline import CachingExplainer
from gnnxc.project.projection import project
from gnnxc.project.threshold import threshold, top_k_edges
from gnnxc.project.vocab import vocab_names
from gnnxc.aggregate.rule import learn_rule
from gnnxc.layers.logical import non_contradiction
from gnnxc.layers.statistical import chance_corrected_overlap

EXPLAINERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ig", "pgexplainer", "gnnexplainer", "random"]
DATASETS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["ba2motifs", "bamultishapes", "mutag"]
KS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [5, 10, 15, 20, 25]
OUT = sys.argv[4] if len(sys.argv) > 4 else "outputs/rigor_grid_v2.jsonl"
PG_SEEDS = int(sys.argv[5]) if len(sys.argv) > 5 else 10
OTHER_SEEDS = int(sys.argv[6]) if len(sys.argv) > 6 else 5
AGGS = ["tree", "onerule", "conj"]
S, N, B = 30, 60, 1000
rng_boot = np.random.default_rng(0)
done = set()
if os.path.exists(OUT):
    for _line in open(OUT):
        try:
            _r = json.loads(_line); done.add((_r["dataset"], _r["explainer"], _r["model_seed"]))
        except Exception:
            pass


def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot_pairwise(pw):
    S_ = pw.shape[0]; mask = ~np.eye(S_, dtype=bool); out = []
    for _ in range(B):
        idx = rng_boot.integers(0, S_, S_); sub = pw[np.ix_(idx, idx)]
        out.append(float(sub[mask].mean()))
    return out


for ds_name in DATASETS:
    vocab = vocab_for(ds_name); names = vocab_names(vocab)
    for ex_name in EXPLAINERS:
        n_seeds = PG_SEEDS if ex_name == "pgexplainer" else OTHER_SEEDS
        for ms in range(n_seeds):
            if (ds_name, ex_name, ms) in done:
                continue
            ds = load_dataset(ds_name, N, ms)
            gsz = np.mean([[g.number_of_nodes(), g.number_of_edges()] for g, _ in ds], axis=0)
            model, acc = train_gin(ds, epochs=100, seed=ms)
            model_ds = [(g, predict(model, g)) for g, _ in ds]
            y = np.array([int(l) for _, l in model_ds])
            m = copy.deepcopy(model)
            for p in m.parameters():
                p.requires_grad_(False)
            ex = CachingExplainer(build_explainer(ex_name, model=m, dataset=model_ds))
            imps = {s: [ex.explain(g, s) for g, _ in model_ds] for s in range(S)}
            edges = {gi: list(model_ds[gi][0].edges()) for gi in range(len(model_ds))}
            for k in KS:
                cmat = {s: np.array([[project(threshold(imps[s][gi], k, model_ds[gi][0]), vocab)[n]
                                      for n in names] for gi in range(len(model_ds))], dtype=int)
                        for s in range(S)}
                rules = {a: {s: learn_rule([dict(zip(names, r)) for r in cmat[s]], y, names, a)
                             for s in range(S)} for a in AGGS}
                pw_stat = np.ones((S, S))
                pw_log = {a: np.ones((S, S)) for a in AGGS}
                for i, j in itertools.combinations(range(S), 2):
                    ov = [chance_corrected_overlap([imps[i][gi][frozenset(e)] for e in edges[gi]],
                                                   [imps[j][gi][frozenset(e)] for e in edges[gi]], k)
                          for gi in range(len(model_ds))]
                    pw_stat[i, j] = pw_stat[j, i] = float(np.mean(ov)) if ov else 1.0
                    for a in AGGS:
                        v = 1.0 if non_contradiction([rules[a][i], rules[a][j]], names) else 0.0
                        pw_log[a][i, j] = pw_log[a][j, i] = v
                off = ~np.eye(S, dtype=bool)
                rec = {"dataset": ds_name, "model_seed": ms, "explainer": ex_name, "k": k,
                       "n_model_seeds": n_seeds, "acc": float(acc),
                       "nodes": float(gsz[0]), "edges": float(gsz[1]),
                       "stat": float(pw_stat[off].mean()), "stat_ci": ci(boot_pairwise(pw_stat))}
                for a in AGGS:
                    rec[f"logic_{a}"] = float(pw_log[a][off].mean())
                    rec[f"logic_{a}_ci"] = ci(boot_pairwise(pw_log[a]))
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{ds_name} ms{ms} {ex_name} k{k}: stat {rec['stat']:.3f} | "
                      f"tree {rec['logic_tree']:.3f} onerule {rec['logic_onerule']:.3f} "
                      f"conj {rec['logic_conj']:.3f}", flush=True)
print("DONE")
