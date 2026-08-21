"""Gaussian-process surrogate with maximum-likelihood hyperparameters."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

from .kernel import kernel_matrix, l1_dist, masked_sq_dist

JITTER = 1e-8


class MixedIntegerGP:
    def __init__(self, jitter: float = JITTER) -> None:
        self.jitter = jitter
        self.sigma2 = 1.0
        self.eta_z = 1.0
        self.eta_c = 1.0
        self.Z = None
        self.C = None
        self.y = None          # standardized targets
        self.y_mean = 0.0
        self.y_std = 1.0
        self._L = None
        self._alpha = None

    # ------------------------------------------------------------------ fit
    def fit(self, Z: np.ndarray, C: np.ndarray, y: np.ndarray,
            restarts: int = 5) -> None:
        Z = np.asarray(Z, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.Z, self.C = Z, C
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) if y.std() > 1e-12 else 1.0
        ys = (y - self.y_mean) / self.y_std

        m = len(ys)
        best = None
        rng = np.random.default_rng(0)
        for _ in range(restarts):
            x0 = np.array([np.log(1.0), np.log(1.0), np.log(1.0)])
            if restarts > 1:
                x0 = np.log(np.exp(rng.normal(0.0, 1.0, 3)) * 1.0)
            res = minimize(
                lambda p: self._nlml(p, ys, m),
                x0, method="L-BFGS-B",
                bounds=[(-12.0, 8.0), (-6.0, 6.0), (-6.0, 6.0)],
            )
            if best is None or res.fun < best.fun:
                best = res
        self.sigma2 = float(np.exp(best.x[0]))
        self.eta_z = float(np.exp(best.x[1]))
        self.eta_c = float(np.exp(best.x[2]))
        self.y = ys
        self._factorize()

    def _nlml(self, log_p, ys, m) -> float:
        sigma2 = float(np.exp(log_p[0]))
        eta_z = float(np.exp(log_p[1]))
        eta_c = float(np.exp(log_p[2]))
        K = kernel_matrix(self.Z, self.C, self.Z, self.C, sigma2, eta_z, eta_c)
        K = K + self.jitter * np.eye(m)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = solve_triangular(L.T, solve_triangular(L, ys, lower=True),
                                 lower=False)
        return float(0.5 * ys @ alpha + np.log(np.diag(L)).sum()
                     + 0.5 * m * np.log(2.0 * np.pi))

    def _factorize(self) -> None:
        K = kernel_matrix(self.Z, self.C, self.Z, self.C,
                          self.sigma2, self.eta_z, self.eta_c)
        K = K + self.jitter * np.eye(len(self.y))
        self._L = np.linalg.cholesky(K)
        self._alpha = solve_triangular(
            self._L.T, solve_triangular(self._L, self.y, lower=True), lower=False)

    # -------------------------------------------------------------- predict
    def predict(self, Zq: np.ndarray, Cq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Zq = np.asarray(Zq, dtype=np.float64)
        Cq = np.asarray(Cq, dtype=np.float64)
        Ks = kernel_matrix(self.Z, self.C, Zq, Cq,
                           self.sigma2, self.eta_z, self.eta_c)  # (m, nq)
        mu_s = Ks.T @ self._alpha
        v = solve_triangular(self._L, Ks, lower=True)
        # self-variance (diag of kernel at query points)
        l1 = l1_dist(Zq, Zq)
        sq = masked_sq_dist(Cq, Zq, Cq, Zq)
        kss = self.sigma2 * np.exp(-l1 / (2.0 * self.eta_z ** 2)
                                   - sq / (2.0 * self.eta_c ** 2))
        var_s = np.maximum(np.diag(kss) - (v * v).sum(axis=0), 0.0)
        mu = mu_s * self.y_std + self.y_mean
        var = var_s * self.y_std ** 2
        return mu, var
