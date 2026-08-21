"""Reproduce the paper's Table 1 / Figure 1 / Figure 2 (path-based algorithms).

Reads the convergence CSVs produced by `igp.run` and renders paper-style
relative-gap vs CPU-time plots, plus a network-properties table.

Robust to missing result files: only the algorithms whose CSVs exist are
plotted / tabulated; the rest are reported as missing.
"""
from __future__ import annotations

import json
import os

import numpy as np

from igp.network import parse_net, parse_trips

RESULTS = "results"
NET_CFG = {
    "Chicago-Sketch": ("data/tn/Chicago-Sketch/ChicagoSketch_net.tntp",
                       "data/tn/Chicago-Sketch/ChicagoSketch_trips.tntp"),
    "Birmingham": ("data/tn/Birmingham-England/Birmingham_Net.tntp",
                   "data/tn/Birmingham-England/Birmingham_Trips.tntp"),
    "Chicago-Regional": ("data/tn/chicago-regional/ChicagoRegional_net.tntp",
                         "data/tn/chicago-regional/ChicagoRegional_trips.tntp"),
    "Philadelphia": ("data/tn/Philadelphia/Philadelphia_net.tntp",
                     "data/tn/Philadelphia/Philadelphia_trips.tntp"),
}

# algorithm key -> (display label, result-file tag, figure-1? , figure-2?)
ALGORITHMS = {
    "greedy":   ("greedy (greedy+IL)", "Chicago-Sketch_greedy_il", "C0", "o", True, False),
    "iGP":      ("iGP (GP+IL, α=0.25)", "Chicago-Sketch_gp_il", "C1", "s", True, False),
    "GP":       ("GP (α=0.25)", "Chicago-Sketch_gp_sil", "C2", "^", True, True),
    "s-greedy": ("s-greedy", "Chicago-Sketch_greedy_sil", "C3", "v", False, True),
}


def _csv_path(tag: str) -> str:
    return os.path.join(RESULTS, f"{tag}_history.csv")


def _json_path(tag: str) -> str:
    return os.path.join(RESULTS, f"{tag}_summary.json")


def available(tag: str) -> bool:
    return os.path.exists(_csv_path(tag)) and os.path.exists(_json_path(tag))


def load_history(tag: str):
    """Return (iter, cpu, rg) arrays, or None if the CSV is missing."""
    path = _csv_path(tag)
    if not os.path.exists(path):
        return None
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1], a[:, 2]


def table1() -> str:
    rows = [["Network", "Nodes", "Links", "Zones", "OD pairs", "Total flow"]]
    for name, (netp, tripp) in NET_CFG.items():
        tail, head, cap, length, fftt, B, power, toll, ltype, n_nodes, n_zones = parse_net(netp)
        o, d, demand, total = parse_trips(tripp)
        rows.append([name, str(n_nodes), str(len(tail)), str(n_zones),
                     f"{len(o):,}", f"{total:,.1f}"])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)


def _style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.set_yscale("log")


def _plot_algos(keys, title, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [k for k in keys if available(ALGORITHMS[k][1])]
    if not present:
        print(f"[skip] {title}: no result files found for {keys}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for k in present:
        label, tag, color, marker, _, _ = ALGORITHMS[k]
        hist = load_history(tag)
        it, t, rg = hist
        axes[0].semilogy(it, rg, color=color, marker=marker, markersize=3, lw=1.2, label=label)
        axes[1].semilogy(t, rg, color=color, marker=marker, markersize=3, lw=1.2, label=label)
    _style_axes(axes[0], "iteration", "relative gap")
    _style_axes(axes[1], "CPU time (s)", "relative gap")
    axes[0].set_title("Chicago Sketch (per iteration)")
    axes[1].set_title("Chicago Sketch (per CPU time)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, fname), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {fname}: plotted {', '.join(ALGORITHMS[k][0] for k in present)}")


def make_figures() -> None:
    _plot_algos(["greedy", "iGP", "GP"],
                "Convergence of path-based algorithms (cf. paper Figure 1)",
                "figure1_pathbased.png")
    _plot_algos(["s-greedy", "GP"],
                "s-greedy vs GP (cf. paper Figure 2)",
                "figure2_sgreedy_vs_gp.png")


def summary_table() -> str:
    rows = [["Algorithm", "final RG", "iter", "CPU s", "paths", "obj. rel. err"]]
    for key, (label, tag, *_rest) in ALGORITHMS.items():
        if not available(tag):
            rows.append([label, "—", "—", "—", "—", "—"])
            continue
        with open(_json_path(tag)) as fh:
            s = json.load(fh)
        hist = load_history(tag)
        it = hist[0]
        rows.append([label, f"{s['final_rg']:.2e}", str(int(it[-1])),
                     f"{s['wall_seconds']:.1f}", str(s['n_paths']),
                     f"{s.get('objective_rel_err', float('nan')):.2e}"])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)


if __name__ == "__main__":
    missing = [ALGORITHMS[k][0] for k in ALGORITHMS if not available(ALGORITHMS[k][1])]
    if missing:
        print("NOTE: missing result files for: " + ", ".join(missing))
        print("      run the corresponding `python3 -m igp.run ...` first.\n")

    print("=== Table 1: network properties (parsed from data) ===")
    print(table1())
    print()
    print("=== Convergence summary (Chicago Sketch) ===")
    print(summary_table())
    make_figures()
    print("\nDone. Figures (if any) written to results/.")
