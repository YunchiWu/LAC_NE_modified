"""Impedance functions for the four link types of the LAC composite network.

Link types (integer codes):
    ROAD  = 0   BPR road link
    QUEUE = 1   vertiport incoming queue link (M/M/1/c point queue)
    FLIGHT= 2   vertiport-to-vertiport flight link (constant)
    EVAC  = 3   vertiport outgoing evacuation link (constant)

All times are returned in **minutes** (matching the TNTP free-flow times);
derivatives are d(travel time) / d(link flow) in minutes per (veh/hour).
"""
from __future__ import annotations

import numpy as np

ROAD = 0
QUEUE = 1
FLIGHT = 2
EVAC = 3


# --------------------------------------------------------------------------
# Road link: BPR
# --------------------------------------------------------------------------
def bpr_time(x: np.ndarray, cap: np.ndarray, fftt: np.ndarray,
             B: np.ndarray, power: np.ndarray) -> np.ndarray:
    ratio = x / cap
    return fftt * (1.0 + B * ratio ** power)


def bpr_deriv(x: np.ndarray, cap: np.ndarray, fftt: np.ndarray,
              B: np.ndarray, power: np.ndarray) -> np.ndarray:
    return fftt * B * power * (x ** (power - 1.0)) / (cap ** power)


# --------------------------------------------------------------------------
# Queue link: M/M/1/c (Zhang et al. 2025, Eqs. 1-5)
# --------------------------------------------------------------------------
def _queue_time_rho(rho: np.ndarray, cap: np.ndarray, srv: np.ndarray) -> np.ndarray:
    """Expected stay time in the M/M/1/c system, given rho = lambda / srv.

    ``srv`` is the queue service rate (throughput capacity), an array of the
    same shape as ``rho``/``cap``.
    """
    rho = np.asarray(rho, dtype=np.float64)
    cap = np.asarray(cap, dtype=np.float64)
    srv = np.asarray(srv, dtype=np.float64)
    # Flush sub-1e-12 rho to zero: avoids log1p(-1) from underflow in r-1.
    rho = np.where(rho < 1e-12, 0.0, rho)
    n = cap + 1.0
    t = np.zeros_like(rho)   # rho == 0 -> zero queue time

    eq = np.abs(rho - 1.0) < 1e-9
    if eq.any():
        t[eq] = n[eq] / (2.0 * srv[eq])

    lo = (~eq) & (rho > 0.0) & (rho < 1.0)
    if lo.any():
        r = rho[lo]
        rn = np.exp(n[lo] * np.log1p(r - 1.0))       # rho^(c+1)  (<= 1)
        D = 1.0 - rn
        P0 = (1.0 - r) / D
        L = r / (1.0 - r) - n[lo] * rn / D
        t[lo] = L / (srv[lo] * (1.0 - P0))

    hi = (~eq) & (rho > 1.0)
    if hi.any():
        r = rho[hi]
        q = np.exp(-n[hi] * np.log1p(r - 1.0))       # rho^-(c+1)  (<= 1)
        P0 = (r - 1.0) * q / (1.0 - q)
        L = -r / (r - 1.0) + n[hi] / (1.0 - q)
        t[hi] = L / (srv[hi] * (1.0 - P0))

    # L / (srv * (1-P0)) is in hours (srv in veh/h); convert to minutes.
    return np.maximum(t, 0.0) * 60.0


def queue_time(lam: np.ndarray, cap: np.ndarray, mu: float = 5000.0) -> np.ndarray:
    """Expected stay time (minutes) for queue-link flow ``lam``.

    The vertiport *capacity* ``cap`` is interpreted as the queue's **throughput
    (service rate)**, so rho = lam / cap: when flow approaches capacity the
    queue saturates and the delay grows, and when capacity is ample the delay is
    small. ``mu`` (V_v/d) is the physical per-vehicle take-off rate and is
    retained only for API compatibility; it is not used in the queue delay.
    """
    lam = np.asarray(lam, dtype=np.float64)
    cap = np.asarray(cap, dtype=np.float64)
    cap_safe = np.maximum(cap, 1e-9)
    rho = lam / cap_safe
    return _queue_time_rho(rho, cap, cap_safe)


def queue_deriv(lam: np.ndarray, cap: np.ndarray, mu: float = 5000.0) -> np.ndarray:
    """dt/dlam via central difference on rho (queue links are few; robust)."""
    lam = np.asarray(lam, dtype=np.float64)
    cap = np.asarray(cap, dtype=np.float64)
    cap_safe = np.maximum(cap, 1e-9)
    rho = lam / cap_safe
    h = 1e-6
    rp = rho + h
    rm = np.maximum(rho - h, 0.0)
    tp = _queue_time_rho(rp, cap, cap_safe)
    tm = _queue_time_rho(rm, cap, cap_safe)
    # forward difference where the backward point is clamped at 0
    use_fwd = (rho - h) < 0.0
    out = (tp - tm) / (2.0 * h * cap_safe)
    if use_fwd.any():
        t0 = _queue_time_rho(np.zeros_like(rho[use_fwd]), cap[use_fwd], cap_safe[use_fwd])
        out[use_fwd] = (tp[use_fwd] - t0) / (h * cap_safe[use_fwd])
    return out


# --------------------------------------------------------------------------
# Flight / evacuation links: constant travel time
# --------------------------------------------------------------------------
def flight_time(h_lp: float, d: float, V_v: float, V_h: float,
                L_hat_km: np.ndarray) -> np.ndarray:
    """Eq. (6): 2(h-d)/V_v + L/V_h, converted to minutes."""
    L_hat_km = np.asarray(L_hat_km, dtype=np.float64)
    vertical_min = 2.0 * (h_lp - d) / 1000.0 / V_v * 60.0
    return vertical_min + L_hat_km / V_h * 60.0


def evac_time(L_tilde_km: np.ndarray, V_f: float) -> np.ndarray:
    """Eq. (7): L/V_f, converted to minutes."""
    L_tilde_km = np.asarray(L_tilde_km, dtype=np.float64)
    return L_tilde_km / V_f * 60.0


# --------------------------------------------------------------------------
# Geodesic distance (haversine) in km
# --------------------------------------------------------------------------
def haversine_km(lon1: np.ndarray, lat1: np.ndarray,
                 lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    R = 6371.0088
    lon1, lat1 = np.radians(lon1), np.radians(lat1)
    lon2, lat2 = np.radians(lon2), np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))
