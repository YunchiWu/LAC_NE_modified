"""Build the land-air composite network from an upper-level decision (z, c)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AirParams
from .link_cost import (EVAC, FLIGHT, QUEUE, ROAD, bpr_deriv, bpr_time, evac_time,
                        flight_time, haversine_km, queue_deriv, queue_time)
from .road_network import RoadNetwork

# Import the sibling IGP Network (base class).
from igp.network import Network  # noqa: E402


class CompositeNetwork(Network):
    """An ``igp.network.Network`` whose links carry heterogeneous cost models.

    Link types (``link_type``): 0=ROAD, 1=QUEUE, 2=FLIGHT, 3=EVAC.
    """

    def __init__(self, tail, head, capacity, length, fftt, B, power, toll,
                 num_nodes, num_zones, link_type, const_time, queue_cap, mu,
                 dist_weight=0.0, toll_weight=0.0):
        super().__init__(tail, head, capacity, length, fftt, B, power, toll,
                         num_nodes, num_zones, dist_weight, toll_weight)
        self.link_type = np.asarray(link_type, dtype=np.int64)
        self.const_time = np.asarray(const_time, dtype=np.float64)
        self.queue_cap = np.asarray(queue_cap, dtype=np.float64)
        self.mu = float(mu)

    # -- cost dispatch ------------------------------------------------------
    def compute_costs(self, x):
        x = np.asarray(x, dtype=np.float64)
        t = np.empty_like(x)
        dt = np.zeros_like(x)
        lt = self.link_type

        road = lt == ROAD
        if road.any():
            t[road] = bpr_time(x[road], self.capacity[road], self.fftt[road],
                               self.B[road], self.power[road])
            dt[road] = bpr_deriv(x[road], self.capacity[road], self.fftt[road],
                                 self.B[road], self.power[road])

        queue = lt == QUEUE
        if queue.any():
            t[queue] = queue_time(x[queue], self.queue_cap[queue], self.mu)
            dt[queue] = queue_deriv(x[queue], self.queue_cap[queue], self.mu)

        const = (lt == FLIGHT) | (lt == EVAC)
        if const.any():
            t[const] = self.const_time[const]
            dt[const] = 0.0

        return t, dt

    def update_links_costs(self, links):
        links = np.asarray(links, dtype=np.int64)
        lt = self.link_type[links]
        xl = self.x[links]

        road = lt == ROAD
        if road.any():
            idx = links[road]
            self.t[idx] = bpr_time(self.x[idx], self.capacity[idx], self.fftt[idx],
                                   self.B[idx], self.power[idx])
            self.dt[idx] = bpr_deriv(self.x[idx], self.capacity[idx], self.fftt[idx],
                                     self.B[idx], self.power[idx])

        queue = lt == QUEUE
        if queue.any():
            idx = links[queue]
            self.t[idx] = queue_time(self.x[idx], self.queue_cap[idx], self.mu)
            self.dt[idx] = queue_deriv(self.x[idx], self.queue_cap[idx], self.mu)

        const = (lt == FLIGHT) | (lt == EVAC)
        if const.any():
            idx = links[const]
            self.t[idx] = self.const_time[idx]
            self.dt[idx] = 0.0


@dataclass
class BuiltNetwork:
    net: CompositeNetwork
    num_road_links: int
    road_node_of_vertiport: np.ndarray   # built vertiport -> road node
    vertiport_nodes: np.ndarray          # built vertiport -> composite node id


def build_composite_network(road: RoadNetwork, vertiport_nodes: np.ndarray,
                            z: np.ndarray, c: np.ndarray,
                            params: AirParams) -> BuiltNetwork:
    """Construct the composite network for a fixed upper-level decision.

    Vertiports are co-located at their candidate road nodes (see
    ``select_vertiport_candidates``); the queue link connects road node ->
    vertiport and the evacuation link vertiport -> road node. Flight links
    (bidirectional) connect every pair of built vertiports.
    """
    z = np.asarray(z, dtype=bool)
    c = np.asarray(c, dtype=np.float64)
    built = np.where(z)[0]

    # --- road links ---------------------------------------------------------
    tail = list(road.tail.astype(np.int64))
    head = list(road.head.astype(np.int64))
    capacity = list(road.capacity.astype(np.float64))
    length = list(road.length.astype(np.float64))
    fftt = list(road.fftt.astype(np.float64))
    B = list(road.B.astype(np.float64))
    power = list(road.power.astype(np.float64))
    toll = list(road.toll.astype(np.float64))
    link_type = [ROAD] * road.num_links
    const_time = [0.0] * road.num_links
    queue_cap = [0.0] * road.num_links

    num_road_links = road.num_links
    n_road_nodes = road.num_nodes
    vnodes = n_road_nodes + np.arange(len(built), dtype=np.int64)  # vertiport ids
    road_node_of_vertiport = vertiport_nodes[built].astype(np.int64)

    # --- queue + evacuation links per built vertiport ------------------------
    for k, p in enumerate(built):
        rn = int(vertiport_nodes[p])
        vn = int(vnodes[k])
        # queue: road node -> vertiport
        tail.append(rn); head.append(vn)
        capacity.append(1.0); length.append(0.0); fftt.append(0.0)
        B.append(0.0); power.append(1.0); toll.append(0.0)
        link_type.append(QUEUE); const_time.append(0.0); queue_cap.append(c[p])
        # evacuation: vertiport -> road node (co-located, L_tilde = 0)
        tail.append(vn); head.append(rn)
        capacity.append(1.0); length.append(0.0); fftt.append(0.0)
        B.append(0.0); power.append(1.0); toll.append(0.0)
        link_type.append(EVAC); const_time.append(0.0); queue_cap.append(0.0)

    # --- flight links between every pair of built vertiports -----------------
    coords = road.coords
    for i in range(len(built)):
        for j in range(i + 1, len(built)):
            ni = int(road_node_of_vertiport[i])
            nj = int(road_node_of_vertiport[j])
            L_hat = float(haversine_km(np.array(coords[ni, 0]), np.array(coords[ni, 1]),
                                       np.array(coords[nj, 0]), np.array(coords[nj, 1])))
            tt = float(flight_time(params.h_lp, params.d, params.V_v, params.V_h,
                                   np.array(L_hat)))
            # i -> j
            tail.append(int(vnodes[i])); head.append(int(vnodes[j]))
            capacity.append(1.0); length.append(L_hat); fftt.append(tt)
            B.append(0.0); power.append(1.0); toll.append(0.0)
            link_type.append(FLIGHT); const_time.append(tt); queue_cap.append(0.0)
            # j -> i
            tail.append(int(vnodes[j])); head.append(int(vnodes[i]))
            capacity.append(1.0); length.append(L_hat); fftt.append(tt)
            B.append(0.0); power.append(1.0); toll.append(0.0)
            link_type.append(FLIGHT); const_time.append(tt); queue_cap.append(0.0)

    net = CompositeNetwork(
        np.asarray(tail, dtype=np.int64), np.asarray(head, dtype=np.int64),
        np.asarray(capacity), np.asarray(length), np.asarray(fftt),
        np.asarray(B), np.asarray(power), np.asarray(toll),
        num_nodes=int(n_road_nodes + len(built)), num_zones=road.num_zones,
        link_type=np.asarray(link_type, dtype=np.int64),
        const_time=np.asarray(const_time), queue_cap=np.asarray(queue_cap),
        mu=params.mu,
    )
    return BuiltNetwork(net, num_road_links, road_node_of_vertiport,
                        vnodes.astype(np.int64))
