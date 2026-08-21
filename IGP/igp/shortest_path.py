"""Shortest-path trees via scipy's C dijkstra, plus path recovery."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .network import Network


class ShortestPathEngine:
    """Pre-builds the CSR structure once; weights are swapped per call."""

    def __init__(self, net: Network) -> None:
        self.net = net
        n = net.num_nodes
        order = np.argsort(net.tail, kind="stable")
        self.order = order
        tail_sorted = net.tail[order]
        head_sorted = net.head[order]
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(indptr, tail_sorted + 1, 1)
        np.cumsum(indptr, out=indptr)
        self.indices = head_sorted.astype(np.int64)
        self.indptr = indptr
        self._csr = csr_matrix((net.t[order], self.indices, self.indptr), shape=(n, n))

    def dijkstra(self, t: np.ndarray, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (dist, pred). dist/pred shape (len(origins), n_nodes)."""
        self._csr.data = t[self.order]
        dist, pred = dijkstra(
            self._csr, directed=True, indices=origins, return_predecessors=True
        )
        if dist.ndim == 1:
            dist = dist[None, :]
            pred = pred[None, :]
        return dist, pred

    def recover_path(self, pred_row: np.ndarray, o: int, d: int) -> list[int] | None:
        """Recover node sequence o->d and return link-index list, or None."""
        if o == d:
            return []
        nodes = [d]
        cur = d
        while cur != o:
            p = int(pred_row[cur])
            if p < 0:
                return None
            nodes.append(p)
            cur = p
        nodes.reverse()
        li = self.net.link_index
        try:
            return [li[(int(nodes[i]), int(nodes[i + 1]))] for i in range(len(nodes) - 1)]
        except KeyError:
            return None
