"""Lower-level LAC-NE solver: evaluate (z, c) via the IGP greedy path solver."""
from __future__ import annotations

import numpy as np

from igp.assignment import PathSolver  # noqa: E402

from .composite_network import BuiltNetwork, build_composite_network
from .config import AirParams
from .road_network import RoadNetwork


class LACNESolver:
    """Adapter that turns an upper-level decision into a UE solve.

    ``evaluate(z, c)`` builds the composite network and runs the greedy
    path-based (IGP) algorithm to a tight relative gap, returning the total
    travel time (TTT) used as the upper-level objective.
    """

    def __init__(self, road: RoadNetwork, trips_path: str,
                 vertiport_nodes: np.ndarray, params: AirParams,
                 rg_target: float = 1e-12, max_iter: int = 200,
                 subproblem: str = "greedy", demand_scale: float = 1.0) -> None:
        self.road = road
        self.trips_path = trips_path
        self.vertiport_nodes = np.asarray(vertiport_nodes, dtype=np.int64)
        self.params = params
        self.rg_target = rg_target
        self.max_iter = max_iter
        self.subproblem = subproblem
        self.demand_scale = demand_scale

    def solve(self, z, c) -> tuple[BuiltNetwork, PathSolver]:
        z = np.asarray(z, dtype=bool)
        c = np.asarray(c, dtype=np.float64)
        built = build_composite_network(self.road, self.vertiport_nodes, z, c,
                                        self.params)
        solver = PathSolver(built.net, self.trips_path,
                            subproblem=self.subproblem, use_inner_loop=True)
        if self.demand_scale != 1.0:
            solver.total_demand *= self.demand_scale
            for od in solver.ods:
                od.demand *= self.demand_scale
        solver.solve(rg_target=self.rg_target, max_iter=self.max_iter,
                     verbose=False)
        return built, solver

    def evaluate(self, z, c) -> float:
        built, _ = self.solve(z, c)
        return float(built.net.TST)
