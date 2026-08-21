"""Expected-improvement acquisition function (Zhang et al. 2025, Eq. 33-34)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_star: float,
                         xi: float = 0.0) -> np.ndarray:
    """Closed-form EI for a normal predictive distribution.

    ``xi`` is the exploration-exploitation trade-off (larger xi explores more).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    d = f_star - xi - mu
    out = np.zeros_like(mu)
    pos = sigma > 1e-12
    z = d[pos] / sigma[pos]
    out[pos] = d[pos] * norm.cdf(z) + sigma[pos] * norm.pdf(z)
    return np.maximum(out, 0.0)
