"""Rigor confirmation of the constructive positive: is consensus-agreement a ROBUST GT-free predictor
of faithfulness across model seeds, with valid (graph-clustered) CIs? Also certified-core precision
across seeds. PRE-DECLARED: robust iff in >=2/3 model seeds the graph-clustered bootstrap Spearman
CI-lower > 0.2 on BOTH datasets. Reports fidelity alongside (expected to be non-robust)."""
import copy
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
import torch, torch.nn.functional as F
from gnnxc.data import load_dataset
from gnnxc.model import train_gin, predict, nx_to_pyg
from gnnxc.explainers.registry import build_explainer
from gnnxc.pipeline import CachingExplainer
from gnnxc.project.threshold import top_k_edges

DATASETS = ["ba2motifs", "bamultishapes"]
MODEL_SEEDS = [0, 1, 2]
EXPL = ["gnnexplainer", "pgexplainer", "ig"]
S, N, K, Bt = 6, 40, 6, 1000
rng = np.random.default_rng(0)


def tk(imp, k):
    return set(tuple(sorted(e)) for e in top_k_edges(imp, k))


def conf_of(model, g):
    if g.number_of_edges() == 0:
        return 0.5
    d = nx_to_pyg(g, 0); model.eval()
    with torch.no_grad():
        return float(F.softmax(model(d.x, d.edge_index, torch.zeros(d.x.size(0), dtype=torch.long)), 1).max())


def clustered_spearman_ci(x, y, groups):
    """graph-clustered bootstrap CI on Spearman rho (resample graphs, keep their rows)."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    uniq = np.unique(groups); idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rhos = []
    for _ in range(Bt):
        gs = rng.choice(uniq, len(uniq)); rows = np.concatenate([idx_by_g[g] for g in gs])
        r, _ = spearmanr(x[rows], y[rows])
        if r == r:
            rhos.append(r)
    return [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))]


verdict = {}
for ds_name in DATASETS:
    for ms in MODEL_SEEDS:
        ds = load_dataset(ds_name, N, ms)
        if not ds[0][0].graph.get("motif_edges"):
            continue
        model, acc = train_gin(ds, epochs=100, seed=ms)
        md = [(g, predict(model, g)) for g, _ in ds]
        gts = [{tuple(sorted(e)) for e in g.graph["motif_edges"]} for g, _ in md]
        cfull = [conf_of(model, g) for g, _ in md]
        TK = {e: [] for e in EXPL}; FC = {e: [] for e in EXPL}
        for e in EXPL:
            m = copy.deepcopy(model)
            for p in m.parameters():
                p.requires_grad_(False)
            ex = CachingExplainer(build_explainer(e, model=m, dataset=md))
            for gi, (g, _) in enumerate(md):
                tks = [tk(ex.explain(g, s), K) for s in range(S)]; TK[e].append(tks)
                fc = []
                for t in tks:
                    gr = g.copy(); gr.remove_edges_from(list(t)); fc.append(max(0.0, cfull[gi] - conf_of(m, gr)))
                FC[e].append(np.mean(fc))
        agree, gea, fid, grp = [], [], [], []
        precs = []; recs = []
        for gi in range(len(md)):
            cnt = Counter()
            for e in EXPL:
                for s in range(S):
                    cnt.update(TK[e][gi][s])
            C = set([ed for ed, _ in cnt.most_common(K)])
            allsets = [TK[e][gi][s] for e in EXPL for s in range(S)]
            core = {ed for ed, c in Counter(ed for st in allsets for ed in st).items() if c >= 0.8 * len(allsets)}
            if core:
                precs.append(len(core & gts[gi]) / len(core))
                if gts[gi]:
                    recs.append(len(core & gts[gi]) / len(gts[gi]))
            for e in EXPL:
                for s in range(S):
                    t = TK[e][gi][s]
                    agree.append(len(t & C) / len(t | C) if (t | C) else 1.0)
                    gea.append(len(t & gts[gi]) / len(t | gts[gi]) if (t | gts[gi]) else 1.0)
                    fid.append(FC[e][gi]); grp.append(gi)
        r_a, _ = spearmanr(agree, gea); ci_a = clustered_spearman_ci(agree, gea, grp)
        r_f, _ = spearmanr(fid, gea)
        verdict.setdefault(ds_name, []).append((ms, r_a, ci_a, r_f, np.mean(precs) if precs else float('nan')))
        print(f"{ds_name:13} ms{ms}: consensus rho={r_a:+.3f} CI[{ci_a[0]:+.2f},{ci_a[1]:+.2f}] | "
              f"fidelity rho={r_f:+.3f} | core_prec(tau.8)={np.mean(precs) if precs else float('nan'):.3f} "
              f"core_recall(tau.8)={np.mean(recs) if recs else float('nan'):.3f}", flush=True)

print("\n=== PRE-DECLARED: consensus-agreement robust iff >=2/3 seeds CI-lower>0.2 on BOTH datasets ===")
ok_all = True
for ds_name in DATASETS:
    hits = sum(1 for (ms, ra, ci, rf, pr) in verdict.get(ds_name, []) if ci[0] > 0.2)
    print(f"  {ds_name}: CI-lower>0.2 in {hits}/{len(verdict.get(ds_name,[]))} model seeds")
    ok_all = ok_all and hits >= 2
print(f"  => consensus-agreement is {'ROBUST (constructive positive holds)' if ok_all else 'NOT robust'}")
