"""Branch-and-bound for the integer-restricted EI (Zhang et al. 2025, Alg. 1)."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.optimize import minimize

from .gp.acquisition import expected_improvement
from .gp.model import MixedIntegerGP

_FRAC_TOL = 1e-5


class EIAcquisition:
    """Wrapper that maps (z, c) -> EI for the B&B relaxations."""

    def __init__(self, gp: MixedIntegerGP, f_star: float, d_z: int,
                 c_min: float, c_max: float, xi: float = 0.0) -> None:
        self.gp = gp
        self.f_star = f_star
        self.d_z = d_z
        self.c_min = c_min
        self.c_max = c_max
        self.xi = xi

    def ei(self, z: np.ndarray, c: np.ndarray) -> float:
        mu, var = self.gp.predict(z[None, :], c[None, :])
        return float(expected_improvement(mu, np.sqrt(var), self.f_star,
                                          self.xi)[0])


def _optimize_relaxed(acq: EIAcquisition, fixed: dict, n_select: int,
                      n_starts: int, rng):
    """Maximize EI over free z in [0,1] and c in [c_min,c_max], sum z = n_select.

    Returns (z_full, c, ei, integer_feasible) or None if the node is infeasible.
    """
    d_z = acq.d_z
    free_idx = np.array([i for i in range(d_z) if i not in fixed], dtype=int)
    n_free = len(free_idx)
    n_fixed_ones = sum(v for i, v in fixed.items() if v == 1)
    remaining = n_select - n_fixed_ones
    if remaining < 0 or remaining > n_free:
        return None

    def unpack(xv):
        z = np.zeros(d_z)
        z[free_idx] = xv[:n_free]
        for i, v in fixed.items():
            z[i] = v
        return z, xv[n_free:]

    def obj(xv):
        z, c = unpack(xv)
        return -acq.ei(z, c)

    cons = ({"type": "eq",
             "fun": lambda xv: xv[:n_free].sum() - remaining}) if n_free > 0 else ()
    bounds = [(0.0, 1.0)] * n_free + [(acq.c_min, acq.c_max)] * d_z

    starts = []
    if n_free > 0:
        base = np.full(n_free, remaining / n_free)
        starts.append(np.concatenate([base, np.full(d_z, acq.c_max * 0.5)]))
        starts.append(np.concatenate([base, np.full(d_z, acq.c_max)]))
    rand_zs = np.zeros((max(1, n_starts), n_free))
    for r in range(rand_zs.shape[0]):
        v = rng.random(n_free)
        if n_free > 0:
            v = v / max(v.sum(), 1e-12) * remaining
        rand_zs[r] = v
    for r in range(rand_zs.shape[0]):
        c0 = rng.uniform(acq.c_min, acq.c_max, d_z)
        starts.append(np.concatenate([rand_zs[r], c0]))

    best = None
    for x0 in starts:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-9, "disp": False})
        if best is None or res.fun < best.fun:
            best = res

    z_opt, c_opt = unpack(best.x)
    ei = -float(best.fun)
    integer_feasible = True
    if n_free > 0:
        integer_feasible = bool(np.all(
            np.abs(z_opt[free_idx] - np.round(z_opt[free_idx])) < _FRAC_TOL))
    return z_opt, c_opt, ei, integer_feasible


def branch_and_bound(gp: MixedIntegerGP, f_star: float, d_z: int,
                     c_min: float, c_max: float, n_select: int,
                     n_starts: int = 6, max_nodes: int = 5000,
                     seed: int = 0, xi: float = 0.0):
    """Solve the integer-restricted EI (Algorithm 1) and return (z, c)."""
    rng = np.random.default_rng(seed)
    acq = EIAcquisition(gp, f_star, d_z, c_min, c_max, xi)

    best_x = None
    best_ei = -np.inf
    counter = 0
    heap = [(-np.inf, counter, ())]
    counter += 1
    n_eval = 0

    while heap and n_eval < max_nodes:
        neg_ub, _, node = heapq.heappop(heap)
        if -neg_ub <= best_ei + 1e-12:
            continue
        fixed = dict(node)
        res = _optimize_relaxed(acq, fixed, n_select, n_starts, rng)
        n_eval += 1
        if res is None:
            continue
        z, c, ei, feasible = res
        if ei <= best_ei + 1e-12:
            continue
        if feasible:
            if ei > best_ei:
                best_ei = ei
                best_x = (z.copy(), c.copy())
        else:
            free_idx = [i for i in range(d_z) if i not in fixed]
            frac = {i: abs(z[i] - round(z[i])) for i in free_idx}
            branch_i = max(frac, key=frac.get)
            for val in (0, 1):
                child = tuple(sorted(node + ((branch_i, val),)))
                heapq.heappush(heap, (-ei, counter, child))
                counter += 1

    if best_x is None:
        z = np.zeros(d_z)
        z[rng.choice(d_z, n_select, replace=False)] = 1.0
        best_x = (z, rng.uniform(c_min, c_max, d_z))
    return best_x
