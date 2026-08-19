"""Round-4 (reviewer R3): report the MANIFOLD-RESTRICTED logical consistency lambda^man alongside the
full-cube reading. The full-cube non-contradiction charges disagreements on concept vectors that no
graph realizes (pure rule-learner extrapolation); lambda^man restricts the check to the concept vectors
the graphs actually realize, which is the more defensible notion. We report both for the headline cells
(MUTAG/PGExplainer all k, BAMultiShapes/PGExplainer k=10,15) over up to 10 model seeds and all three
aggregators. Expectation (stated before running): lambda^man >= full-cube reading (fewer chargeable
disagreements), so the null only gets cleaner. Committed to outputs/rigor_manifold.jsonl + .log.
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
from gnnxc.project.threshold import threshold
from gnnxc.project.vocab import vocab_names
from gnnxc.aggregate.rule import learn_rule
from gnnxc.layers.logical import non_contradiction, non_contradiction_manifold

CELLS = {"mutag": [5, 10, 15, 20, 25], "bamultishapes": [10, 15]}
EX = "pgexplainer"
AGGS = ["tree", "onerule", "conj"]
S, N, N_SEEDS = 30, 60, 10
OUT = "outputs/rigor_manifold.jsonl"
open(OUT, "w").close()

records = []
for ds_name, KS in CELLS.items():
    vocab = vocab_for(ds_name); names = vocab_names(vocab)
    for ms in range(N_SEEDS):
        ds = load_dataset(ds_name, N, ms)
        model, acc = train_gin(ds, epochs=100, seed=ms)
        md = [(g, predict(model, g)) for g, _ in ds]
        y = np.array([int(l) for _, l in md])
        m = copy.deepcopy(model)
        for p in m.parameters():
            p.requires_grad_(False)
        ex = CachingExplainer(build_explainer(EX, model=m, dataset=md))
        imps = {s: [ex.explain(g, s) for g, _ in md] for s in range(S)}
        for k in KS:
            cmat = {s: np.array([[project(threshold(imps[s][gi], k, md[gi][0]), vocab)[n] for n in names]
                                 for gi in range(len(md))], dtype=int) for s in range(S)}
            # realized concept-vector support across all seeds and graphs for this cell
            realized = {tuple(int(v) for v in cmat[s][gi]) for s in range(S) for gi in range(len(md))}
            realized_dicts = [dict(zip(names, vec)) for vec in realized]
            for agg in AGGS:
                rules = {s: learn_rule([dict(zip(names, r)) for r in cmat[s]], y, names, agg) for s in range(S)}
                full = manif = 0
                for i, j in itertools.combinations(range(S), 2):
                    full += 1.0 if non_contradiction([rules[i], rules[j]], names) else 0.0
                    manif += 1.0 if non_contradiction_manifold([rules[i], rules[j]], realized_dicts, names) else 0.0
                npairs = S * (S - 1) / 2
                rec = {"dataset": ds_name, "model_seed": ms, "k": k, "aggregator": agg,
                       "n_realized": len(realized), "full_cube": full / npairs, "manifold": manif / npairs}
                records.append(rec)
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            print(f"{ds_name} ms{ms} k{k}: realized={len(realized)}/{2**len(names)} | "
                  + " ".join(f"{a}: full {[r for r in records if r['dataset']==ds_name and r['model_seed']==ms and r['k']==k and r['aggregator']==a][0]['full_cube']:.3f}/man {[r for r in records if r['dataset']==ds_name and r['model_seed']==ms and r['k']==k and r['aggregator']==a][0]['manifold']:.3f}" for a in AGGS), flush=True)

# summary: headline cell MUTAG/PG/k=10, lambda^man mean over seeds per aggregator
print("\n=== lambda^man summary (mean over model seeds) ===")
for ds_name, KS in CELLS.items():
    for k in KS:
        for a in AGGS:
            rows = [r for r in records if r["dataset"] == ds_name and r["k"] == k and r["aggregator"] == a]
            if rows:
                fm = np.mean([r["full_cube"] for r in rows]); mm = np.mean([r["manifold"] for r in rows])
                print(f"  {ds_name}/k{k}/{a}: full-cube {fm:.3f}  manifold {mm:.3f}  (n={len(rows)} seeds)")
print("DONE")
