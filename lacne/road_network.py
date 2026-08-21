"""Load road networks (TNTP), node coordinates, and vertiport candidates."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Make the sibling IGP package importable regardless of cwd.
_IGP_DIR = str(Path(__file__).resolve().parents[1] / "IGP")
if _IGP_DIR not in sys.path:
    sys.path.insert(0, _IGP_DIR)

from igp.network import parse_net, parse_trips  # noqa: E402

from .link_cost import haversine_km


@dataclass
class RoadNetwork:
    name: str
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
    coords: np.ndarray          # (num_nodes, 2) lon/lat
    o: np.ndarray               # trip origins (0-based)
    d: np.ndarray               # trip destinations (0-based)
    demand: np.ndarray
    total_demand: float

    @property
    def num_links(self) -> int:
        return len(self.tail)


def _parse_tntp_node_coords(path: str) -> np.ndarray:
    """Parse ``*_node.tntp`` (Node X Y ; lines) -> (num_nodes, 2) lon/lat."""
    rows = {}
    for line in open(path, "r", encoding="utf-8", errors="replace"):
        s = line.strip()
        if not s or s.startswith("<") or s.startswith("~"):
            continue
        parts = s.replace(";", "").split()
        if len(parts) >= 3 and parts[0].lstrip("-").isdigit():
            rows[int(parts[0])] = (float(parts[1]), float(parts[2]))
    n = max(rows) if rows else 0
    coords = np.full((n, 2), np.nan)
    for i, (x, y) in rows.items():
        coords[i - 1] = (x, y)
    return coords


def _parse_geojson_node_coords(path: str) -> np.ndarray:
    """Parse Anaheim ``anaheim_nodes.geojson`` -> (num_nodes, 2) lon/lat."""
    data = json.load(open(path, "r", encoding="utf-8"))
    feats = data["features"]
    n = len(feats)
    coords = np.full((n, 2), np.nan)
    for f in feats:
        i = int(f["properties"]["id"]) - 1
        lon, lat = f["geometry"]["coordinates"]
        coords[i] = (lon, lat)
    return coords


def load_road_network(name: str) -> RoadNetwork:
    """Load a road network by registry name ('SiouxFalls' | 'Anaheim')."""
    from .config import NETWORKS
    cfg = NETWORKS[name]
    tail, head, cap, length, fftt, B, power, toll, _ltype, n_nodes, n_zones = \
        parse_net(cfg["net"])
    if cfg["coords_fmt"] == "tntp_node":
        coords = _parse_tntp_node_coords(cfg["nodes"])
    else:
        coords = _parse_geojson_node_coords(cfg["nodes"])
    o, d, demand, total = parse_trips(cfg["trips"])
    return RoadNetwork(name, tail, head, cap, length, fftt, B, power, toll,
                       n_nodes, n_zones, coords, o, d, demand, total)


def select_vertiport_candidates(net: RoadNetwork, n_candidates: int) -> np.ndarray:
    """Pick candidate vertiport road-nodes by spatial farthest-point sampling.

    The paper does not publish vertiport coordinates. We co-locate candidates at
    road nodes spread across the network (maximizing the minimum pairwise
    distance), preferring non-centroid nodes when they exist, so aerial mobility
    is not trivially reachable from every zone centroid.
    """
    coords = net.coords
    n = net.num_nodes
    eligible = np.arange(net.num_zones, n)      # non-centroid road nodes
    if len(eligible) < n_candidates:
        eligible = np.arange(n)                 # fallback (e.g. Sioux Falls)
    picks = [int(eligible[0])]
    while len(picks) < n_candidates:
        d = np.array([
            min(haversine_km(np.array(coords[p, 0]), np.array(coords[p, 1]),
                             np.array(coords[e, 0]), np.array(coords[e, 1]))
                for p in picks)
            for e in eligible
        ])
        picks.append(int(eligible[int(np.argmax(d))]))
    return np.array(picks, dtype=np.int64)
