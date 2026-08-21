"""Mixed-integer Bayesian optimization (Zhang et al. 2025, Algorithm 2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bb import branch_and_bound
from .gp.model import MixedIntegerGP


@dataclass
class MIBOResult:
    z: np.ndarray
    c: np.ndarray
    f: float
    all_f: np.ndarray
    best_so_far: np.ndarray


class MIBO:
    def __init__(self, evaluate, d_z: int, n_select: int,
                 c_min: float, c_max: float,
                 n_init: int = 5, budget: int = 75, seed: int = 0) -> None:
        self.evaluate = evaluate
        self.d_z = d_z
        self.n_select = n_select
        self.c_min = c_min
        self.c_max = c_max
        self.n_init = n_init
        self.budget = budget
        self.seed = seed

    def _random_z(self, rng) -> np.ndarray:
        z = np.zeros(self.d_z)
        z[rng.choice(self.d_z, self.n_select, replace=False)] = 1.0
        return z

    def run(self) -> MIBOResult:
        rng = np.random.default_rng(self.seed)
        Z, C, F = [], [], []
        best_so_far = []

        for _ in range(self.n_init):
            z = self._random_z(rng)
            c = rng.uniform(self.c_min, self.c_max, self.d_z)
            f = float(self.evaluate(z, c))
            Z.append(z); C.append(c); F.append(f)
            best_so_far.append(min(F))

        Z = np.asarray(Z); C = np.asarray(C); F = np.asarray(F)

        while len(F) < self.budget:
            gp = MixedIntegerGP()
            gp.fit(Z, C, F)
            xi = 0.01 * (float(F.max()) - float(F.min()))
            z, c = branch_and_bound(gp, float(F.min()), self.d_z,
                                    self.c_min, self.c_max, self.n_select,
                                    seed=rng.integers(0, 2**31 - 1), xi=xi)
            z = np.round(z)
            f = float(self.evaluate(z, c))
            Z = np.vstack([Z, z]); C = np.vstack([C, c]); F = np.append(F, f)
            best_so_far.append(F.min())

        best_i = int(np.argmin(F))
        return MIBOResult(z=Z[best_i], c=C[best_i], f=float(F[best_i]),
                          all_f=F, best_so_far=np.asarray(best_so_far))
