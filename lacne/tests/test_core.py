"""Unit tests for the LAC link costs, GP kernel and EI."""
import numpy as np

from lacne.gp.acquisition import expected_improvement
from lacne.gp.kernel import kernel_matrix, l1_dist, masked_sq_dist
from lacne.link_cost import bpr_deriv, bpr_time, queue_deriv, queue_time


def test_queue_time_zero_and_limit():
    cap = 10.0
    assert queue_time(np.array([0.0]), np.array([cap]))[0] < 1e-9
    # rho = lam/cap == 1 -> stay time (cap+1)/(2*cap) hours
    t1 = queue_time(np.array([cap]), np.array([cap]))[0]
    assert abs(t1 - 11.0 / (2.0 * cap) * 60.0) < 1e-6


def test_queue_time_monotone():
    cap = 10000.0
    lams = np.linspace(0, 6000, 60)
    ts = queue_time(lams, np.full(60, cap))
    assert np.all(np.diff(ts) >= 0)


def test_queue_time_decreases_with_capacity():
    """Throughput model: larger capacity -> shorter (or equal) delay."""
    lam = 4035.0
    ts = [queue_time(np.array([lam]), np.array([c]))[0] for c in [4000.0, 6000.0, 10000.0]]
    assert ts[2] < ts[1] <= ts[0]


def test_queue_time_near_rho_one_no_blowup():
    """Regression: rho within ~1e-12 of 1 must stay bounded (~minutes)."""
    cap = 2384.0
    for rho in [1.0 - 6e-16, 1.0, 1.0 + 6e-16, 0.9999, 1.0001]:
        t = queue_time(np.array([rho * cap]), np.array([cap]))[0]
        assert np.isfinite(t) and t < 100.0, f"rho={rho} -> t={t}"


def test_queue_deriv_positive():
    d = queue_deriv(np.array([3000.0]), np.array([10000.0]))
    assert d[0] > 0


def test_bpr():
    x = np.array([4000.0]); cap = np.array([10000.0])
    fftt = np.array([1.0]); B = np.array([0.15]); power = np.array([4.0])
    t = bpr_time(x, cap, fftt, B, power)
    assert abs(t[0] - 1.0 * (1 + 0.15 * 0.4 ** 4)) < 1e-12


def test_kernel_hamming_binary():
    Z = np.array([[1, 0, 1], [1, 0, 0]], dtype=float)
    d = l1_dist(Z, Z)
    assert d[0, 1] == 1 and d[1, 0] == 1 and d[0, 0] == 0


def test_kernel_masked_dist():
    C = np.array([[5.0, 3.0, 2.0], [5.0, 9.0, 9.0]])
    Z = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    sq = masked_sq_dist(C, Z, C, Z)
    # only dim 0 is built in both: (5-5)^2 = 0
    assert sq[0, 1] == 0.0


def test_ei_zero_variance():
    mu = np.array([0.5]); sigma = np.array([0.0])
    assert expected_improvement(mu, sigma, 0.0)[0] == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
