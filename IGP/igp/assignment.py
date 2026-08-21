"""Greedy path-based traffic assignment (Xie, Nie & Liu 2018).

Implements Algorithm 1 (greedy single-OD solver) and Algorithm 2 (main loop
with column generation + intelligent inner loop), plus GP / iGP / s-greedy
baselines used in the paper's Figure 1 and Figure 2.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .network import Network, parse_trips
from .shortest_path import ShortestPathEngine

EPS = 1e-12
S_FLOOR = 1e-8  # floor for s_h to avoid division-by-zero on fresh (zero-flow) paths


@dataclass
class OD:
    o: int
    d: int
    demand: float
    paths: list = field(default_factory=list)   # list of np.ndarray (link idx)
    flows: list = field(default_factory=list)   # list of float
    path_set: set = field(default_factory=set)  # set of tuple(link idx)
    drs: float = float("inf")                   # max-min path cost difference


class PathSolver:
    def __init__(
        self,
        net: Network,
        trips_path: str,
        subproblem: str = "greedy",   # 'greedy' or 'gp'
        use_inner_loop: bool = True,
        gp_alpha: float = 0.25,       # GP step-size modifier (α; 0.25 reproduces
                                       # the paper's Figure 2 ordering)
    ) -> None:
        self.net = net
        self.subproblem = subproblem
        self.use_inner_loop = use_inner_loop
        self.gp_alpha = gp_alpha
        self.sp = ShortestPathEngine(net)
        o, d, demand, total = parse_trips(trips_path)
        self.o = o
        self.d = d
        self.demand = demand
        self.total_demand = total

        # Build OD records in deterministic (o, d) order.
        order = np.lexsort((d, o))
        self.ods: list[OD] = []
        self.od_index: dict[tuple[int, int], int] = {}
        for idx in order:
            oi, di = int(o[idx]), int(d[idx])
            od = OD(oi, di, float(demand[idx]))
            self.od_index[(oi, di)] = len(self.ods)
            self.ods.append(od)

        self.origins = np.array(sorted({int(x) for x in o}), dtype=np.int64)
        self.dests_by_origin: dict[int, list[tuple[int, int]]] = {int(r): [] for r in self.origins}
        for idx in order:
            oi, di = int(o[idx]), int(d[idx])
            self.dests_by_origin[oi].append((self.od_index[(oi, di)], di))

    # ------------------------------------------------------------------ setup
    def init_aon(self) -> None:
        """All-or-nothing on free-flow shortest paths."""
        self.net.t = self.net.fftt + self.net.const  # BPR at x=0
        self.net.dt = np.zeros(self.net.num_links)
        for r in self.origins:
            _, pred = self.sp.dijkstra(self.net.t, np.array([r], dtype=np.int64))
            pr = pred[0]
            for od_idx, s in self.dests_by_origin[int(r)]:
                od = self.ods[od_idx]
                path = self.sp.recover_path(pr, int(r), s)
                if path is None:
                    continue
                links = np.asarray(path, dtype=np.int64)
                od.paths = [links]
                od.path_set = {tuple(path)}
                od.flows = [od.demand]
                self.net.x[links] += od.demand
        self.net.update_all_costs()

    # ------------------------------------------------------------- subproblems
    def _path_v_s(self, paths) -> tuple[np.ndarray, np.ndarray]:
        n = len(paths)
        v = np.empty(n)
        s = np.empty(n)
        for i, p in enumerate(paths):
            v[i] = self.net.t[p].sum()
            s[i] = self.net.dt[p].sum()
        return v, s

    def adjust_greedy(self, od: OD) -> None:
        """Algorithm 1: exact greedy solve of the quadratic subproblem."""
        paths = od.paths
        n = len(paths)
        g = np.asarray(od.flows, dtype=np.float64)
        v, s = self._path_v_s(paths)
        c = v - s * g

        if n == 1:
            f_new = np.array([od.demand])
            od.drs = 0.0
            used_indices = [0]
        else:
            order = np.argsort(c, kind="stable")
            d = od.demand
            ss = np.maximum(s, S_FLOOR)
            cs = c[order]
            ss_sorted = ss[order]

            # Equation (21): w = (d + sum c_h/s_h) / (sum 1/s_h)
            sum_inv = 1.0 / ss_sorted[0]
            sum_c_inv = cs[0] / ss_sorted[0]
            w = (d + sum_c_inv) / sum_inv
            used = [order[0]]
            for k in range(1, n):
                if cs[k] >= w:
                    break
                sum_inv += 1.0 / ss_sorted[k]
                sum_c_inv += cs[k] / ss_sorted[k]
                w = (d + sum_c_inv) / sum_inv
                used.append(order[k])

            f_new = np.zeros(n)
            for k in used:
                f_new[k] = (w - c[k]) / ss[k]
            f_new = np.maximum(f_new, 0.0)
            od.drs = float(v.max() - v.min())
            used_indices = used

        touched: set[int] = set()
        for i in range(n):
            delta = f_new[i] - g[i]
            if delta != 0.0:
                self.net.x[paths[i]] += delta
                touched.update(paths[i].tolist())
        if touched:
            self.net.update_links_costs(np.fromiter(touched, dtype=np.int64))

        od.paths = [paths[k] for k in used_indices]
        od.flows = [float(f_new[k]) for k in used_indices]
        od.path_set = {tuple(p.tolist()) for p in od.paths}

    def adjust_gp(self, od: OD) -> None:
        """One gradient-projection step exactly as in Jayakrishnan et al. (1994):

            f_k <- max(0, f_k - (d_k - d_ref)/s_k),  for k != ref
            s_k = sum of link-cost derivatives over links in (path k XOR ref)

        With 3+ paths this is only an approximate solution to the quadratic
        subproblem (unlike the greedy solve, which is exact)."""
        paths = od.paths
        f = np.asarray(od.flows, dtype=np.float64)
        n = len(paths)
        if n < 2:
            od.drs = 0.0
            return
        v, _ = self._path_v_s(paths)
        od.drs = float(v.max() - v.min())
        ref = int(np.argmin(v))
        ref_links = paths[ref].tolist()
        ref_set = set(ref_links)

        moves: list[tuple[int, float, list, list]] = []
        for k in range(n):
            if k == ref or f[k] <= 0.0:
                continue
            k_links = paths[k].tolist()
            k_set = set(k_links)
            k_only = [a for a in k_links if a not in ref_set]
            ref_only = [a for a in ref_links if a not in k_set]
            disjoint = k_only + ref_only
            s_k = float(self.net.dt[disjoint].sum())
            if s_k <= EPS:
                continue
            step = min(self.gp_alpha * (v[k] - v[ref]) / s_k, f[k])
            if step <= 0.0:
                continue
            moves.append((k, step, k_only, ref_only))

        touched: set[int] = set()
        for k, step, k_only, ref_only in moves:
            f[k] -= step
            f[ref] += step
            if k_only:
                self.net.x[k_only] -= step
                touched.update(k_only)
            if ref_only:
                self.net.x[ref_only] += step
                touched.update(ref_only)
        if touched:
            self.net.update_links_costs(np.fromiter(touched, dtype=np.int64))

        # Drop zero-flow paths (consistent with the greedy solver and the
        # paper's Table 2, where GP and greedy generate comparable path counts).
        keep = np.where(f > EPS)[0]
        od.paths = [paths[i] for i in keep]
        od.flows = [float(f[i]) for i in keep]
        od.path_set = {tuple(p.tolist()) for p in od.paths}

    def adjust(self, od: OD) -> None:
        if self.subproblem == "gp":
            self.adjust_gp(od)
        else:
            self.adjust_greedy(od)

    def compute_drs(self, od: OD) -> float:
        if not od.paths:
            return 0.0
        vs = [float(self.net.t[p].sum()) for p in od.paths]
        return max(vs) - min(vs)

    # ------------------------------------------------------------------ loops
    def main_loop(self) -> None:
        """Column generation + one flow adjustment per OD (Gauss-Seidel)."""
        for r in self.origins:
            ri = int(r)
            _, pred = self.sp.dijkstra(self.net.t, np.array([ri], dtype=np.int64))
            pr = pred[0]
            for od_idx, s in self.dests_by_origin[ri]:
                od = self.ods[od_idx]
                path = self.sp.recover_path(pr, ri, s)
                if path is None:
                    continue
                pt = tuple(path)
                if pt not in od.path_set:
                    od.paths.append(np.asarray(path, dtype=np.int64))
                    od.path_set.add(pt)
                    od.flows.append(0.0)
                self.adjust(od)

    def inner_loop(self, rg_prev: float) -> None:
        """Intelligent scheme: repeat flow adjustment on less-converged ODs."""
        if not self.use_inner_loop:
            return
        I = 0
        max_I = 1000
        while I < max_I:
            I += 1
            FC = 0
            recompute = (I % 100 == 0)
            for od in self.ods:
                if recompute:
                    od.drs = self.compute_drs(od)
                if od.drs >= rg_prev / 2.0:
                    FC += 1
                    self.adjust(od)
            if FC == 0:
                break

    def relative_gap(self) -> float:
        TST = self.net.TST
        if TST <= 0.0:
            return 1.0
        dist, _ = self.sp.dijkstra(self.net.t, self.origins)
        SPTT = 0.0
        for i, r in enumerate(self.origins):
            dr = dist[i]
            for od_idx, s in self.dests_by_origin[int(r)]:
                SPTT += dr[s] * self.ods[od_idx].demand
        return 1.0 - SPTT / TST

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        rg_target: float = 1e-8,
        max_iter: int = 200,
        verbose: bool = True,
    ) -> list[tuple[int, float, float]]:
        """Return history [(iter, cpu_seconds, relative_gap), ...]."""
        self.init_aon()
        rg = self.relative_gap()
        history = [(0, 0.0, rg)]
        if verbose:
            print(f"[{self.subproblem}{'+IL' if self.use_inner_loop else ''}] "
                  f"iter 0  RG={rg:.6e}  TST={self.net.TST:.6f}")
        k = 0
        while rg > rg_target and k < max_iter:
            t0 = time.perf_counter()
            self.main_loop()
            self.inner_loop(rg)
            rg = self.relative_gap()
            dt = time.perf_counter() - t0
            history.append((k + 1, history[-1][1] + dt, rg))
            if verbose:
                print(f"  iter {k + 1}  RG={rg:.6e}  TST={self.net.TST:.6f}  "
                      f"paths={self.count_paths()}")
            k += 1
        return history

    def count_paths(self) -> int:
        return sum(len(od.paths) for od in self.ods)

    def total_path_links(self) -> int:
        return sum(int(p.size) for od in self.ods for p in od.paths)
