"""Core data structures and shared helpers for single-source shortest paths
with negative edge weights.

All algorithms in this package return a :class:`Result`. Counters
(``relaxations`` = edge examinations, ``passes``) are machine-independent
work measures used in the experiments alongside wall-clock time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

INF = float("inf")
Edge = Tuple[int, int, float]


class Graph:
    """Directed weighted graph on vertices 0..n-1 stored as an edge list
    plus an adjacency list built on demand."""

    def __init__(self, n: int, edges: Optional[List[Edge]] = None):
        self.n = n
        self.edges: List[Edge] = list(edges) if edges else []
        self._adj: Optional[List[List[Tuple[int, float]]]] = None

    @property
    def m(self) -> int:
        return len(self.edges)

    def add_edge(self, u: int, v: int, w: float) -> None:
        self.edges.append((u, v, w))
        self._adj = None

    @property
    def adj(self) -> List[List[Tuple[int, float]]]:
        if self._adj is None:
            adj: List[List[Tuple[int, float]]] = [[] for _ in range(self.n)]
            for u, v, w in self.edges:
                adj[u].append((v, w))
            self._adj = adj
        return self._adj


@dataclass
class Result:
    dist: List[float]
    pred: List[int]
    negative_cycle: Optional[List[int]] = None  # vertices in cycle order
    relaxations: int = 0  # number of edge examinations
    passes: int = 0
    algorithm: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def has_negative_cycle(self) -> bool:
        return self.negative_cycle is not None


def find_pred_cycle(pred: List[int]) -> Optional[List[int]]:
    """Return a cycle in the predecessor graph (each vertex v has at most one
    out-pointer pred[v]) or None. Runs in O(n) using three-colour marking.

    Lemma (see report, Lemma 4): every cycle in the predecessor graph built
    by label-correcting relaxations has negative total weight, so a cycle
    found here is a certificate of a negative cycle.
    """
    n = len(pred)
    state = [0] * n  # 0 = unvisited, 1 = on current walk, 2 = finished
    for start in range(n):
        if state[start]:
            continue
        walk = []
        v = start
        while v != -1 and state[v] == 0:
            state[v] = 1
            walk.append(v)
            v = pred[v]
        if v != -1 and state[v] == 1:
            # v lies on the cycle; collect it by following pred pointers
            cycle = [v]
            u = pred[v]
            while u != v:
                cycle.append(u)
                u = pred[u]
            cycle.reverse()  # pred points backwards; reverse to edge order
            for x in walk:
                state[x] = 2
            return cycle
        for x in walk:
            state[x] = 2
    return None


def cycle_weight(g: Graph, cycle: List[int]) -> float:
    """Weight of cycle v0 -> v1 -> ... -> vk-1 -> v0 using the cheapest
    parallel edge for each hop."""
    best = {}
    for u, v, w in g.edges:
        if (u, v) not in best or w < best[(u, v)]:
            best[(u, v)] = w
    total = 0.0
    k = len(cycle)
    for i in range(k):
        total += best[(cycle[i], cycle[(i + 1) % k])]
    return total
