"""Round-2 strengthening: leave-one-explainer-out (LOEO) robustness of the C4 consensus-agreement
signal. With only three explainers, is consensus-agreement just 'IG-agreement in disguise'? For each
dropped explainer we recompute the crowd consensus over the REMAINING two and correlate each remaining
explanation's consensus-agreement with its true faithfulness (GEA vs motif). PRE-DECLARED: C4 is
crowd-robust iff Spearman rho stays positive with graph-clustered CI-lower > 0 in >=2/3 model seeds
for EVERY leave-one-out pool, on both datasets. CPU, S=6, n=40, k=6 (matches the C4 config)."""
import copy, itertools
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from gnnxc.data import load_dataset
from gnnxc.model import train_gin, predict
from gnnxc.explainers.registry import build_explainer
from gnnxc.pipeline import CachingExplainer
from gnnxc.project.threshold import top_k_edges

DATASETS = ["ba2motifs", "bamultishapes"]
MODEL_SEEDS = [0, 1, 2]
EXPL = ["gnnexplainer", "pgexplainer", "ig"]
S, N, K, Bt = 6, 40, 6, 800
rng = np.random.default_rng(0)


def tk(imp, k):
    return set(tuple(sorted(e)) for e in top_k_edges(imp, k))


def gclust_ci(x, y, groups):
    x, y, g = map(np.asarray, (x, y, groups)); uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}; out = []
    for _ in range(Bt):
        gs = rng.choice(uq, len(uq)); rows = np.concatenate([idx[u] for u in gs])
        r, _ = spearmanr(x[rows], y[rows])
        if r == r: out.append(r)
    return float(np.percentile(out, 2.5))


results = {}
for ds in DATASETS:
    for ms in MODEL_SEEDS:
        data = load_dataset(ds, N, ms)
        gts = [{tuple(sorted(e)) for e in g.graph["motif_edges"]} for g, _ in data]
        model, _ = train_gin(data, epochs=100, seed=ms)
        md = [(g, predict(model, g)) for g, _ in data]
        TK = {}
        for e in EXPL:
            m = copy.deepcopy(model)
            for p in m.parameters():
                p.requires_grad_(False)
            ex = CachingExplainer(build_explainer(e, model=m, dataset=md))
            TK[e] = [[tk(ex.explain(g, s), K) for s in range(S)] for g, _ in md]
        for dropped in [None] + EXPL:
            pool = [e for e in EXPL if e != dropped]
            agree, gea, grp = [], [], []
            for gi in range(len(md)):
                cnt = Counter()
                for e in pool:
                    for s in range(S):
                        cnt.update(TK[e][gi][s])
                C = set([ed for ed, _ in cnt.most_common(K)])
                for e in pool:
                    for s in range(S):
                        t = TK[e][gi][s]
                        agree.append(len(t & C) / len(t | C) if (t | C) else 1.0)
                        gea.append(len(t & gts[gi]) / len(t | gts[gi]) if (t | gts[gi]) else 1.0)
                        grp.append(gi)
            rho, _ = spearmanr(agree, gea); lo = gclust_ci(agree, gea, grp)
            results.setdefault((ds, dropped or "full"), []).append((ms, rho, lo))
            print(f"{ds:13} ms{ms} drop={str(dropped):12} rho={rho:+.3f} CI-lo={lo:+.3f}", flush=True)

print("\n=== PRE-DECLARED: C4 crowd-robust iff rho>0 & CI-lo>0 in >=2/3 seeds for EVERY pool ===")
allok = True
for (ds, pool), rows in sorted(results.items()):
    hits = sum(1 for (ms, rho, lo) in rows if rho > 0 and lo > 0)
    ok = hits >= 2; allok = allok and ok
    print(f"  {ds:13} pool=drop-{pool:12}: positive+CI>0 in {hits}/{len(rows)} seeds -> {'ok' if ok else 'FAILS'}")
print(f"\n=> C4 consensus signal is {'CROWD-ROBUST (not one-explainer-in-disguise)' if allok else 'NOT crowd-robust'}")
