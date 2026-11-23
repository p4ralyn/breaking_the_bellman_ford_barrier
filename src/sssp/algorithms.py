"""Single-source shortest-path algorithms for graphs with negative weights.

Implemented:
  * bellman_ford            - textbook Bellman-Ford (always n-1 passes)
  * bellman_ford_early      - Bellman-Ford with early termination
  * yen                     - Yen's (1970) ordering improvement, ~n/2 passes
  * yen_randomized          - Bannister-Eppstein (2012) random ordering, ~n/3 passes expected
  * spfa                    - queue-based Bellman-Ford (FIFO label-correcting)
  * spfa_amortized          - SPFA + amortized predecessor-graph cycle checks
  * hybrid                  - randomized Yen + early termination + amortized
                              cycle checks (our engineered variant)
  * dijkstra                - for non-negative weights (used inside Johnson)
  * johnson                 - all-pairs via potentials (reweighting)
  * floyd_warshall          - O(n^3) reference oracle for testing

Every SSSP routine returns a core.Result. If a negative cycle reachable from
the source exists, Result.negative_cycle holds its vertices in edge order and
distances are not meaningful.
"""
from __future__ import annotations

import heapq
import random
from collections import deque
from typing import List, Optional

from .core import INF, Graph, Result, find_pred_cycle


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _relax_pass(edges, dist, pred) -> tuple:
    """Relax every edge once in the given order. Returns (changed, count)."""
    changed = False
    for u, v, w in edges:
        du = dist[u]
        if du != INF and du + w < dist[v]:
            dist[v] = du + w
            pred[v] = u
            changed = True
    return changed, len(edges)


def _extract_cycle(g: Graph, dist, pred, relax_fn, res: Result) -> List[int]:
    """Called once a negative cycle is known to exist (a full pass still
    relaxed some edge). Keeps relaxing until the predecessor graph contains a
    cycle, which is guaranteed to happen (Cherkassky-Goldberg) and which is
    negative by Lemma 4 of the report."""
    cyc = find_pred_cycle(pred)
    while cyc is None:
        _, c = relax_fn()
        res.relaxations += c
        res.passes += 1
        cyc = find_pred_cycle(pred)
    return cyc


# ---------------------------------------------------------------------------
# Bellman-Ford family
# ---------------------------------------------------------------------------
def bellman_ford(g: Graph, s: int, early_stop: bool = False) -> Result:
    n = g.n
    dist = [INF] * n
    pred = [-1] * n
    dist[s] = 0
    res = Result(dist, pred, algorithm="bellman_ford_early" if early_stop else "bellman_ford")
    edges = g.edges
    for _ in range(n - 1):
        changed, c = _relax_pass(edges, dist, pred)
        res.relaxations += c
        res.passes += 1
        if early_stop and not changed:
            return res  # converged: no negative cycle reachable (Lemma 3)
    # n-th pass: any further improvement certifies a negative cycle (Thm 2)
    changed, c = _relax_pass(edges, dist, pred)
    res.relaxations += c
    res.passes += 1
    if changed:
        res.negative_cycle = _extract_cycle(
            g, dist, pred, lambda: _relax_pass(edges, dist, pred), res)
    return res


def bellman_ford_early(g: Graph, s: int) -> Result:
    return bellman_ford(g, s, early_stop=True)


def _yen_core(g: Graph, s: int, order: List[int], name: str,
              amortized_check: bool = False) -> Result:
    """Yen's method. `order` is a permutation of the vertices; rank[v] is the
    position of v. Edges are split into forward (rank[u] < rank[v]) and
    backward edges; one pass relaxes forward edges in increasing rank of the
    tail, then backward edges in decreasing rank of the tail. Each half-pass
    processes a DAG in topological order, so a whole monotone run of a path
    is fixed in one half-pass.
    """
    n = g.n
    rank = [0] * n
    for i, v in enumerate(order):
        rank[v] = i
    fwd = sorted((e for e in g.edges if rank[e[0]] < rank[e[1]]), key=lambda e: rank[e[0]])
    bwd = sorted((e for e in g.edges if rank[e[0]] > rank[e[1]]), key=lambda e: -rank[e[0]])
    self_loops = [e for e in g.edges if e[0] == e[1]]
    seq = fwd + bwd + self_loops

    dist = [INF] * n
    pred = [-1] * n
    dist[s] = 0
    res = Result(dist, pred, algorithm=name)

    def one_pass():
        return _relax_pass(seq, dist, pred)

    max_passes = (n + 1) // 2 + 1  # ceil(n/2) passes suffice (Theorem 5)
    for _ in range(max_passes):
        changed, c = one_pass()
        res.relaxations += c
        res.passes += 1
        if not changed:
            return res
        if amortized_check:
            cyc = find_pred_cycle(pred)
            if cyc is not None:  # early negative-cycle certificate (Lemma 4)
                res.negative_cycle = cyc
                return res
    # still changing after ceil(n/2)+1 passes  =>  negative cycle reachable
    res.negative_cycle = _extract_cycle(g, dist, pred, one_pass, res)
    return res


def yen(g: Graph, s: int) -> Result:
    return _yen_core(g, s, list(range(g.n)), "yen")


def yen_randomized(g: Graph, s: int, seed: Optional[int] = None) -> Result:
    order = list(range(g.n))
    random.Random(seed).shuffle(order)
    return _yen_core(g, s, order, "yen_randomized")


def bfs_order(g: Graph, s: int) -> List[int]:
    """Vertices in breadth-first order from s (unreachable ones appended).
    Shortest paths tend to move 'away' from s, so most of their edges become
    forward edges and long monotone runs are fixed in a single half-pass."""
    seen = [False] * g.n
    seen[s] = True
    order = [s]
    q = deque([s])
    adj = g.adj
    while q:
        u = q.popleft()
        for v, _ in adj[u]:
            if not seen[v]:
                seen[v] = True
                order.append(v)
                q.append(v)
    order.extend(v for v in range(g.n) if not seen[v])
    return order


def hybrid(g: Graph, s: int, seed: Optional[int] = None) -> Result:
    """Our engineered variant: Yen's two-DAG passes with a BFS vertex order
    (O(m) preprocessing), early termination, and a predecessor-graph cycle
    check after every pass (O(n) per pass, i.e. amortized O(1) per edge
    examination since each pass examines all m >= n-1 edges).
    Worst case is still ceil(n/2)+1 passes (Theorem 5 holds for any order)."""
    return _yen_core(g, s, bfs_order(g, s), "hybrid", amortized_check=True)


# ---------------------------------------------------------------------------
# Queue-based (SPFA) family
# ---------------------------------------------------------------------------
def spfa(g: Graph, s: int, amortized_check: bool = False) -> Result:
    """FIFO label-correcting algorithm. Without cycle checks, a negative cycle
    is detected when some vertex has been dequeued n times (i.e. its shortest
    path would need >= n edges). With amortized_check, the predecessor graph
    is searched for a cycle after every n relaxations that changed a label,
    which detects cycles far earlier in practice while adding only O(1)
    amortized work per relaxation."""
    n = g.n
    adj = g.adj
    dist = [INF] * n
    pred = [-1] * n
    dist[s] = 0
    in_q = [False] * n
    count = [0] * n
    q = deque([s])
    in_q[s] = True
    res = Result(dist, pred, algorithm="spfa_amortized" if amortized_check else "spfa")
    updates_since_check = 0
    detected = False  # a negative cycle is certain; just need a certificate
    while q:
        u = q.popleft()
        in_q[u] = False
        count[u] += 1
        if count[u] >= n and not detected:  # classic detection (Lemma 7)
            detected = True
            amortized_check = True
            updates_since_check = n
        du = dist[u]
        for v, w in adj[u]:
            res.relaxations += 1
            if du + w < dist[v]:
                dist[v] = du + w
                pred[v] = u
                updates_since_check += 1
                if not in_q[v]:
                    in_q[v] = True
                    q.append(v)
        if amortized_check and updates_since_check >= (1 if detected else n):
            updates_since_check = 0
            cyc = find_pred_cycle(pred)
            if cyc is not None:
                res.negative_cycle = cyc
                return res
    return res


def spfa_amortized(g: Graph, s: int) -> Result:
    return spfa(g, s, amortized_check=True)


# ---------------------------------------------------------------------------
# Dijkstra, Johnson, Floyd-Warshall
# ---------------------------------------------------------------------------
def dijkstra(g: Graph, s: int, weight=None) -> Result:
    n = g.n
    adj = g.adj
    dist = [INF] * n
    pred = [-1] * n
    dist[s] = 0
    res = Result(dist, pred, algorithm="dijkstra")
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if weight is not None:
                w = weight(u, v, w)
            res.relaxations += 1
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(pq, (nd, v))
    return res


def johnson(g: Graph, sssp=bellman_ford_early):
    """All-pairs shortest paths. Returns (D, negative_cycle) where D[u][v] is
    the distance matrix (None if a negative cycle exists)."""
    n = g.n
    h_graph = Graph(n + 1, g.edges + [(n, v, 0) for v in range(n)])
    r = sssp(h_graph, n)
    if r.has_negative_cycle:
        return None, r.negative_cycle
    h = r.dist[:n]
    reweight = lambda u, v, w: w + h[u] - h[v]  # >= 0 by Lemma 6
    D = []
    for u in range(n):
        du = dijkstra(g, u, weight=reweight).dist
        D.append([du[v] - h[u] + h[v] if du[v] != INF else INF for v in range(n)])
    return D, None


def floyd_warshall(g: Graph):
    n = g.n
    D = [[INF] * n for _ in range(n)]
    for i in range(n):
        D[i][i] = 0
    for u, v, w in g.edges:
        if w < D[u][v]:
            D[u][v] = w
    for k in range(n):
        Dk = D[k]
        for i in range(n):
            dik = D[i][k]
            if dik == INF:
                continue
            Di = D[i]
            for j in range(n):
                if dik + Dk[j] < Di[j]:
                    Di[j] = dik + Dk[j]
    return D


SSSP_ALGORITHMS = {
    "bellman_ford": bellman_ford,
    "bellman_ford_early": bellman_ford_early,
    "yen": yen,
    "yen_randomized": lambda g, s: yen_randomized(g, s, seed=1),
    "spfa": spfa,
    "spfa_amortized": spfa_amortized,
    "hybrid": lambda g, s: hybrid(g, s, seed=1),
}
