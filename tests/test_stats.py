import numpy as np
from scipy.stats import wilcoxon

from gnnxc.stats.tests import cluster_bootstrap_ci, paired_wilcoxon


def test_paired_wilcoxon_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.random(20)
    b = a + rng.normal(0.1, 0.05, 20)
    res = paired_wilcoxon(a, b, seed_floor=10)
    stat, p = wilcoxon(a, b)
    assert abs(res.statistic - stat) < 1e-9
    assert abs(res.p_value - p) < 1e-9


def test_under_powered_below_floor():
    res = paired_wilcoxon([1, 2, 3], [1.1, 2.2, 2.9], seed_floor=10)
    assert res.under_powered and res.p_value is None and res.label == "under-powered"


def test_cluster_bootstrap_covers_known_mean():
    rng = np.random.default_rng(1)
    # 20 clusters, each ~N(0.5, .1); enough clusters for the grand mean ≈ 0.5
    values, clusters = [], []
    for c in range(20):
        xs = rng.normal(0.5, 0.1, 30)
        values += xs.tolist()
        clusters += [c] * 30
    mean, lo, hi = cluster_bootstrap_ci(values, clusters, resamples=2000, rng_seed=0)
    assert lo < mean < hi          # percentile CI brackets the point estimate
    assert abs(mean - 0.5) < 0.05  # data is centred where we generated it
    assert lo <= 0.5 <= hi         # true mean covered with 20 clusters
