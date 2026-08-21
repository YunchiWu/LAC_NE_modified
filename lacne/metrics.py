"""Performance metrics for a solved LAC composite network."""
from __future__ import annotations

import numpy as np

from .composite_network import BuiltNetwork


def ttt(built: BuiltNetwork) -> float:
    return float(built.net.TST)


def auto_lac_shares(built: BuiltNetwork, solver) -> dict[str, float]:
    """Fraction of demand served by pure-road vs land-air paths."""
    total = float(solver.total_demand)
    if total <= 0:
        return {"auto": 1.0, "lac": 0.0}
    lac_flow = 0.0
    for od in solver.ods:
        for path, flow in zip(od.paths, od.flows):
            if np.any(np.asarray(path) >= built.num_road_links):
                lac_flow += flow
    lac = lac_flow / total
    return {"auto": 1.0 - lac, "lac": lac}


def vc_distribution(built: BuiltNetwork) -> np.ndarray:
    """Percentage of ROAD links in each V/C bucket.

    Buckets: [0,0.5), [0.5,1), [1,1.5), [1.5,2), [2,inf).
    """
    x = built.net.x[:built.num_road_links]
    cap = built.net.capacity[:built.num_road_links]
    vc = x / np.maximum(cap, 1e-12)
    bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, np.inf])
    counts, _ = np.histogram(vc, bins=bins)
    return counts / max(len(vc), 1) * 100.0
