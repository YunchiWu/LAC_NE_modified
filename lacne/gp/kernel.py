"""Mixed-integer kernel for the GP surrogate (Zhang et al. 2025, Eqs. 28-30).

Discrete part uses the Hamming distance (which equals the L1 distance for
binary vectors); the continuous part uses a masked squared Euclidean distance
so that only vertiports built in *both* solutions contribute to the capacity
distance. The full kernel is the product of the two RBF-style terms:

    kappa_MI(x, x') = sigma^2 * exp( -L1(z,z')/(2 eta_z^2)
                                     - ||c-c'||^2_masked/(2 eta_c^2) )
"""
from __future__ import annotations

import numpy as np


def l1_dist(Z1: np.ndarray, Z2: np.ndarray) -> np.ndarray:
    """L1 (Hamming, for binary) distance matrix: (n1, n2)."""
    return np.abs(Z1[:, None, :] - Z2[None, :, :]).sum(axis=-1)


def masked_sq_dist(C1: np.ndarray, Z1: np.ndarray,
                   C2: np.ndarray, Z2: np.ndarray) -> np.ndarray:
    """Masked squared Euclidean distance: sum_p z1_p z2_p (c1_p - c2_p)^2."""
    mask = Z1[:, None, :] * Z2[None, :, :]
    diff = C1[:, None, :] - C2[None, :, :]
    return (mask * diff * diff).sum(axis=-1)


def kernel_matrix(Z1: np.ndarray, C1: np.ndarray,
                  Z2: np.ndarray, C2: np.ndarray,
                  sigma2: float, eta_z: float, eta_c: float) -> np.ndarray:
    l1 = l1_dist(Z1, Z2)
    sq = masked_sq_dist(C1, Z1, C2, Z2)
    return sigma2 * np.exp(-l1 / (2.0 * eta_z ** 2) - sq / (2.0 * eta_c ** 2))
