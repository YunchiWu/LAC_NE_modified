"""Command-line entry point for the greedy path-based assignment solver."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .assignment import PathSolver
from .network import Network

# Network registry: name -> (net, trips, flow, dist_weight, toll_weight)
NETWORKS = {
    "Chicago-Sketch": {
        "net": "data/tn/Chicago-Sketch/ChicagoSketch_net.tntp",
        "trips": "data/tn/Chicago-Sketch/ChicagoSketch_trips.tntp",
        "flow": "data/tn/Chicago-Sketch/ChicagoSketch_flow.tntp",
        "dist_weight": 0.04,
        "toll_weight": 0.02,
        "objective": 17313018.7387477,
    },
    "Birmingham": {
        "net": "data/tn/Birmingham-England/Birmingham_Net.tntp",
        "trips": "data/tn/Birmingham-England/Birmingham_Trips.tntp",
        "flow": None,
        "dist_weight": 0.0,
        "toll_weight": 0.0,
        "objective": None,
    },
    "Chicago-Regional": {
        "net": "data/tn/chicago-regional/ChicagoRegional_net.tntp",
        "trips": "data/tn/chicago-regional/ChicagoRegional_trips.tntp",
        "flow": "data/tn/chicago-regional/ChicagoRegional_flow.tntp",
        "dist_weight": 0.25,
        "toll_weight": 0.1,
        "objective": 30792611.3864393,
    },
    "Philadelphia": {
        "net": "data/tn/Philadelphia/Philadelphia_net.tntp",
        "trips": "data/tn/Philadelphia/Philadelphia_trips.tntp",
        "flow": None,
        "dist_weight": 0.0,
        "toll_weight": 0.055,
        "objective": None,
    },
}


def load_reference_flow(net: Network, flow_path: str) -> np.ndarray:
    x = np.zeros(net.num_links)
    with open(flow_path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 3 and p[0].isdigit():
                t, h = int(p[0]) - 1, int(p[1]) - 1
                x[net.link_index[(t, h)]] = float(p[2])
    return x


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Greedy path-based traffic assignment")
    ap.add_argument("network", choices=list(NETWORKS), help="TNTP network name")
    ap.add_argument("--subproblem", choices=["greedy", "gp"], default="greedy")
    ap.add_argument("--no-inner-loop", action="store_true",
                    help="disable inner loop (s-greedy / plain GP)")
    ap.add_argument("--gp-alpha", type=float, default=0.25,
                    help="GP step-size modifier (alpha); 0.25 reproduces the paper's Fig. 2")
    ap.add_argument("--rg-target", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot", action="store_true", help="render convergence plots")
    args = ap.parse_args(argv)

    cfg = NETWORKS[args.network]
    net = Network.from_tntp(cfg["net"], cfg["dist_weight"], cfg["toll_weight"])
    solver = PathSolver(
        net,
        cfg["trips"],
        subproblem=args.subproblem,
        use_inner_loop=not args.no_inner_loop,
        gp_alpha=args.gp_alpha,
    )

    t_start = time.perf_counter()
    history = solver.solve(rg_target=args.rg_target, max_iter=args.max_iter)
    wall = time.perf_counter() - t_start

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.network}_{args.subproblem}" + ("_sil" if args.no_inner_loop else "_il")
    iters = [h[0] for h in history]
    times = [h[1] for h in history]
    rgs = [h[2] for h in history]

    np.savetxt(
        os.path.join(args.out, f"{tag}_history.csv"),
        np.column_stack([iters, times, rgs]),
        header="iter,cpu_seconds,relative_gap",
        delimiter=",",
        comments="",
    )

    summary = {
        "network": args.network,
        "subproblem": args.subproblem,
        "inner_loop": not args.no_inner_loop,
        "final_rg": float(rgs[-1]),
        "final_TST": net.TST,
        "n_paths": solver.count_paths(),
        "total_demand": solver.total_demand,
        "iterations": iters[-1],
        "wall_seconds": wall,
    }
    if cfg["flow"]:
        x_ref = load_reference_flow(net, cfg["flow"])
        rel = np.abs(net.x - x_ref) / np.maximum(x_ref, 1e-9)
        summary["flow_rel_err_max"] = float(rel.max())
        summary["flow_rel_err_mean"] = float(rel.mean())
        summary["tst_reference"] = float((x_ref * net.compute_costs(x_ref)[0]).sum())
    if cfg["objective"]:
        obj = float((net.fftt * (net.x + net.B / (net.power + 1.0) * net.x
                                 * (net.x / net.capacity) ** net.power)
                     + net.const * net.x).sum())
        summary["objective"] = obj
        summary["objective_rel_err"] = abs(obj - cfg["objective"]) / cfg["objective"]

    with open(os.path.join(args.out, f"{tag}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.plot:
        _plot(args.out, tag, history)


def _plot(out_dir: str, tag: str, history) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = [h[0] for h in history]
    times = [h[1] for h in history]
    rgs = [h[2] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].semilogy(iters, rgs, marker="o", markersize=3)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("relative gap")
    axes[0].set_title(f"{tag} (per iteration)")
    axes[0].grid(True, which="both", ls=":")

    axes[1].semilogy(times, rgs, marker="o", markersize=3)
    axes[1].set_xlabel("CPU time (s)")
    axes[1].set_ylabel("relative gap")
    axes[1].set_title(f"{tag} (per CPU time)")
    axes[1].grid(True, which="both", ls=":")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_convergence.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
