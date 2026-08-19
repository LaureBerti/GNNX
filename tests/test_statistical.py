import numpy as np

from gnnxc.layers.statistical import (
    chance_corrected_overlap,
    kendall_tau,
    support_variance,
)


def test_identical_overlap_is_one():
    a = [0.9, 0.8, 0.7, 0.1, 0.05, 0.02]
    assert chance_corrected_overlap(a, a, k=3) == 1.0


def test_random_overlap_near_zero():
    rng = np.random.default_rng(0)
    n, k, vals = 40, 5, []
    for _ in range(300):
        a, b = rng.random(n), rng.random(n)
        vals.append(chance_corrected_overlap(a, b, k))
    assert abs(np.mean(vals)) < 0.1  # chance-corrected → centred on 0


def test_kendall_tau_sign():
    a = [1, 2, 3, 4, 5]
    assert kendall_tau(a, [1, 2, 3, 4, 5]) > 0.9   # concordant
    assert kendall_tau(a, [5, 4, 3, 2, 1]) < -0.9  # discordant


def test_support_variance_zero_when_stable():
    cvs = [{"cycle": 1, "benzene": 0}] * 5
    v = support_variance(cvs)
    assert v["cycle"] == 0.0 and v["benzene"] == 0.0
