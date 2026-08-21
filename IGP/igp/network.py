"""TNTP network parsing and BPR cost functions.

Implements the generalized link cost used by the Bar-Gera (TNTP) test networks:

    t_a(x_a) = fftt_a * (1 + B_a * (x_a / cap_a)^power_a)
               + dist_weight * length_a + toll_weight * toll_a

The last two terms are constant in x, so they affect path costs and the UE
solution but not the derivative used in the quadratic approximation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


def _parse_metadata(path: str) -> tuple[dict, list[str]]:
    """Return (metadata dict, list of data lines after END OF METADATA)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    meta: dict[str, str] = {}
    data_lines: list[str] = []
    in_data = False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("<"):
            m = re.match(r"<([^>]*)>\s*(.*)", s)
            if m:
                key = m.group(1).strip().upper()
                val = m.group(2).strip()
                meta[key] = val
                if key == "END OF METADATA":
                    in_data = True
            continue
        if in_data and not s.startswith("~"):
            data_lines.append(s)
    return meta, data_lines


def parse_net(path: str) -> tuple[np.ndarray, ...]:
    """Parse a TNTP ``*_net.tntp`` file.

    Returns arrays (tail, head, capacity, length, fftt, B, power, toll, link_type)
    with node indices converted to 0-based.
    """
    meta, data = _parse_metadata(path)
    n_zones = int(float(meta["NUMBER OF ZONES"]))
    n_nodes = int(float(meta["NUMBER OF NODES"]))
    n_links = int(float(meta["NUMBER OF LINKS"]))
    cols = []
    for line in data:
        line = line.replace(";", "").strip()
        if not line:
            continue
        parts = line.split()
        cols.append([float(x) for x in parts[:10]])
    a = np.asarray(cols, dtype=np.float64)
    if a.shape[0] != n_links:
        raise ValueError(f"expected {n_links} links, parsed {a.shape[0]}")
    tail = a[:, 0].astype(np.int64) - 1
    head = a[:, 1].astype(np.int64) - 1
    capacity = a[:, 2]
    length = a[:, 3]
    fftt = a[:, 4]
    B = a[:, 5]
    power = a[:, 6]
    toll = a[:, 8] if a.shape[1] > 8 else np.zeros_like(fftt)
    link_type = a[:, 9] if a.shape[1] > 9 else np.zeros_like(fftt)
    return tail, head, capacity, length, fftt, B, power, toll, link_type, n_nodes, n_zones


def parse_trips(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Parse a TNTP ``*_trips.tntp`` file.

    Returns (o, d, demand) as 0-based node (zone) indices, plus total demand.
    Zero-demand and intrazonal entries are dropped.
    """
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    parts = text.split("END OF METADATA")
    body = parts[-1]
    total_match = re.search(r"TOTAL OD FLOW[>]\s*([\d.]+)", text)
    total = float(total_match.group(1)) if total_match else 0.0
    blocks = re.split(r"Origin\s+(\d+)", body)
    o_list, d_list, v_list = [], [], []
    for i in range(1, len(blocks), 2):
        o = int(blocks[i])
        content = blocks[i + 1]
        for d_str, v_str in re.findall(r"(\d+)\s*:\s*([\d.eE+-]+)", content):
            v = float(v_str)
            if v == 0.0:
                continue
            d = int(d_str)
            if o == d:
                continue
            o_list.append(o - 1)
            d_list.append(d - 1)
            v_list.append(v)
    o = np.asarray(o_list, dtype=np.int64)
    d = np.asarray(d_list, dtype=np.int64)
    demand = np.asarray(v_list, dtype=np.float64)
    return o, d, demand, total


@dataclass
class Network:
    tail: np.ndarray
    head: np.ndarray
    capacity: np.ndarray
    length: np.ndarray
    fftt: np.ndarray
    B: np.ndarray
    power: np.ndarray
    toll: np.ndarray
    num_nodes: int
    num_zones: int
    dist_weight: float = 0.0
    toll_weight: float = 0.0
    # runtime state
    x: np.ndarray = field(init=False)
    t: np.ndarray = field(init=False)
    dt: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.num_links = len(self.tail)
        self.const = self.dist_weight * self.length + self.toll_weight * self.toll
        self.link_index = {(int(t), int(h)): i for i, (t, h) in enumerate(zip(self.tail, self.head))}
        self.x = np.zeros(self.num_links, dtype=np.float64)
        self.t = self.const.copy()
        self.dt = np.zeros(self.num_links, dtype=np.float64)

    @classmethod
    def from_tntp(
        cls,
        net_path: str,
        dist_weight: float = 0.0,
        toll_weight: float = 0.0,
    ) -> "Network":
        tail, head, cap, length, fftt, B, power, toll, ltype, n_nodes, n_zones = parse_net(net_path)
        return cls(tail, head, cap, length, fftt, B, power, toll, n_nodes, n_zones,
                   dist_weight, toll_weight)

    def compute_costs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cap = self.capacity
        ratio = x / cap
        bpr = self.fftt * (1.0 + self.B * ratio ** self.power)
        t = bpr + self.const
        dt = self.fftt * self.B * self.power * (x ** (self.power - 1.0)) / (cap ** self.power)
        return t, dt

    def update_all_costs(self) -> None:
        self.t, self.dt = self.compute_costs(self.x)

    def update_links_costs(self, links: np.ndarray) -> None:
        """Recompute cost and derivative for a subset of links (0-based)."""
        xl = self.x[links]
        cap = self.capacity[links]
        ratio = xl / cap
        self.t[links] = self.fftt[links] * (1.0 + self.B[links] * ratio ** self.power[links]) \
            + self.const[links]
        self.dt[links] = self.fftt[links] * self.B[links] * self.power[links] \
            * (xl ** (self.power[links] - 1.0)) / (cap ** self.power[links])

    @property
    def TST(self) -> float:
        """Total system travel time = sum_a x_a * t_a."""
        return float((self.x * self.t).sum())
