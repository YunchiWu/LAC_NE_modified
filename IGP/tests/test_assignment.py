"""Unit tests (pytest-compatible but also runnable standalone).

Run either `python3 tests/test_assignment.py` or `pytest tests/`.
"""
import numpy as np

from igp.network import Network, parse_trips
from igp.assignment import PathSolver, OD


def _tiny_net() -> Network:
    """Two nodes connected by two parallel links (two candidate paths)."""
    tail = np.array([0, 0], dtype=np.int64)
    head = np.array([1, 1], dtype=np.int64)
    cap = np.array([5.0, 5.0])
    length = np.zeros(2)
    fftt = np.array([1.0, 2.0])
    B = np.full(2, 0.15)
    power = np.full(2, 4.0)
    toll = np.zeros(2)
    return Network(tail, head, cap, length, fftt, B, power, toll, num_nodes=2,
                   num_zones=1, dist_weight=0.0, toll_weight=0.0)


def _bare_solver(net):
    s = PathSolver.__new__(PathSolver)  # bypass __init__
    s.net = net
    s.subproblem = "greedy"
    return s


def test_cost_function_reference_objective():
    net = Network.from_tntp(
        "data/tn/Chicago-Sketch/ChicagoSketch_net.tntp", dist_weight=0.04, toll_weight=0.02
    )
    x_ref = np.zeros(net.num_links)
    for line in open("data/tn/Chicago-Sketch/ChicagoSketch_flow.tntp"):
        p = line.split()
        if len(p) >= 3 and p[0].isdigit():
            x_ref[net.link_index[(int(p[0]) - 1, int(p[1]) - 1)]] = float(p[2])
    obj = float((net.fftt * (x_ref + net.B / (net.power + 1.0) * x_ref
                             * (x_ref / net.capacity) ** net.power)
                 + net.const * x_ref).sum())
    assert abs(obj - 17313018.7387477) / 17313018.7387477 < 1e-12


def test_greedy_subproblem_kk():
    """Two-path greedy: verify KKT of the quadratic subproblem exactly."""
    net = _tiny_net()
    # Load path 0 so its local slope is steeper, making both paths competitive.
    net.x = np.array([4.0, 1.0])
    net.update_all_costs()
    od = OD(o=0, d=1, demand=20.0)
    od.paths = [np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)]
    od.path_set = {(0,), (1,)}
    od.flows = [4.0, 1.0]

    # Pre-update constants c_h = v_h - s_h g_h (quadratic approx at current point).
    g0 = np.array([4.0, 1.0])
    v0 = np.array([net.t[0], net.t[1]])
    s0 = np.array([net.dt[0], net.dt[1]])
    c0 = v0 - s0 * g0

    s = _bare_solver(net)
    s.adjust_greedy(od)

    # Map surviving paths back to original indices via their single link.
    f_full = np.zeros(2)
    for path, flow in zip(od.paths, od.flows):
        f_full[int(path[0])] = flow

    assert abs(f_full.sum() - 20.0) < 1e-9
    assert np.all(f_full >= -1e-12)
    assert len(od.paths) == 2, "both paths should carry flow at this demand"
    marginal = c0 + s0 * f_full
    used = f_full > 1e-9
    assert np.allclose(marginal[used], marginal[used][0], rtol=1e-8)


def test_greedy_single_path():
    net = _tiny_net()
    net.x = np.array([2.0, 2.0])
    net.update_all_costs()
    od = OD(o=0, d=1, demand=3.0)
    od.paths = [np.array([0], dtype=np.int64)]
    od.path_set = {(0,)}
    od.flows = [3.0]
    s = _bare_solver(net)
    s.adjust_greedy(od)
    assert abs(od.flows[0] - 3.0) < 1e-12
    assert od.paths[0].tolist() == [0]


def test_trips_parse():
    o, d, demand, total = parse_trips("data/tn/Chicago-Sketch/ChicagoSketch_trips.tntp")
    # `total` is the raw TOTAL OD FLOW header; demand drops intrazonal/zero entries.
    assert abs(total - 1260907.4400005303) < 1e-6
    assert 0.0 < demand.sum() <= total
    assert o.min() >= 0 and o.max() < 387


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
