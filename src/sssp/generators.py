"""Graph generators for testing and experiments.

Negative edges without negative cycles are produced with the potential trick:
draw non-negative base weights c(u,v) and random potentials p(v), then set
w(u,v) = c(u,v) + p(u) - p(v). Every cycle keeps its (non-negative) base
weight, so no negative cycle exists, yet many edges become negative.
"""
from __future__ import annotations

import random
from typing import Optional

from .core import Graph


def _connect_from_source(n, rng, edges, maxw):
    # random spanning arborescence rooted at 0 so all vertices are reachable
    order = list(range(1, n))
    rng.shuffle(order)
    placed = [0]
    for v in order:
        u = rng.choice(placed)
        edges.append((u, v, rng.randint(0, maxw)))
        placed.append(v)


def random_potential_graph(n: int, m: int, seed: Optional[int] = None,
                           maxw: int = 100, pot: int = 1000) -> Graph:
    """Sparse random digraph, ~m edges, negative weights, no negative cycle,
    every vertex reachable from 0."""
    rng = random.Random(seed)
    base = []
    _connect_from_source(n, rng, base, maxw)
    while len(base) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            base.append((u, v, rng.randint(0, maxw)))
    p = [rng.randint(0, pot) for _ in range(n)]
    edges = [(u, v, w + p[u] - p[v]) for u, v, w in base]
    rng.shuffle(edges)
    return Graph(n, edges)


def grid_potential_graph(side: int, seed: Optional[int] = None,
                         maxw: int = 100, pot: int = 1000) -> Graph:
    """side x side grid with edges in both directions (long shortest paths,
    hard for label-correcting methods)."""
    rng = random.Random(seed)
    n = side * side
    idx = lambda r, c: r * side + c
    base = []
    for r in range(side):
        for c in range(side):
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    base.append((idx(r, c), idx(rr, cc), rng.randint(0, maxw)))
    p = [rng.randint(0, pot) for _ in range(n)]
    edges = [(u, v, w + p[u] - p[v]) for u, v, w in base]
    rng.shuffle(edges)
    return Graph(n, edges)


def adversarial_chain(n: int, extra: int = 3, seed: Optional[int] = None) -> Graph:
    """Path 0 -> 1 -> ... -> n-1 of negative edges listed in *reverse* order,
    plus `extra` random long-way-round edges per vertex. Textbook
    Bellman-Ford needs n-1 passes; ordering tricks help."""
    rng = random.Random(seed)
    edges = []
    for v in range(n - 1, 0, -1):
        edges.append((v - 1, v, -1))
    for _ in range(extra * n):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            # big enough to never beat the chain: chain dist(v) = -v
            edges.append((u, v, (v - u) * -1 + rng.randint(1, 50)))
    return Graph(n, edges)


def plant_negative_cycle(g: Graph, length: int, seed: Optional[int] = None) -> Graph:
    """Copy of g with a negative cycle of the given length added among
    random vertices (all reachable from 0 in our generators)."""
    rng = random.Random(seed)
    verts = rng.sample(range(1, g.n), length)
    edges = list(g.edges)
    for i in range(length):
        edges.append((verts[i], verts[(i + 1) % length], -10_000))
    return Graph(g.n, edges)


def plant_subtle_negative_cycle(g: Graph, length: int, seed: Optional[int] = None) -> Graph:
    """Copy of g (which must have no negative cycle) with a long, *barely*
    negative cycle. Edge weights are chosen relative to the exact distances
    delta of g: w(u,v) = delta(v) - delta(u) (reduced cost 0) for the first
    length-1 edges and reduced cost -1 for the closing edge. Every negative
    cycle in the result then weighs exactly -1, so labels drop by only one
    unit per trip around it, which is much harder to expose than a heavy
    cycle."""
    from .algorithms import bellman_ford_early  # local import avoids a cycle
    delta = bellman_ford_early(g, 0).dist
    rng = random.Random(seed)
    verts = rng.sample(range(1, g.n), length)
    edges = list(g.edges)
    for i in range(length):
        u, v = verts[i], verts[(i + 1) % length]
        red = -1 if i == length - 1 else 0
        edges.append((u, v, delta[v] - delta[u] + red))
    return Graph(g.n, edges)


def random_small_graph(n: int, p: float, seed: Optional[int] = None,
                       lo: int = -5, hi: int = 10) -> Graph:
    """Erdos-Renyi style digraph with arbitrary weights (may contain negative
    cycles, may be disconnected). Used for exhaustive testing."""
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(n):
            if rng.random() < p:
                edges.append((u, v, rng.randint(lo, hi)))
    return Graph(n, edges)
