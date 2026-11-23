"""Correctness tests: every algorithm is checked against Floyd-Warshall on
thousands of random graphs, including graphs with negative cycles."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sssp.algorithms import SSSP_ALGORITHMS, floyd_warshall, johnson  # noqa: E402
from sssp.core import INF, Graph, cycle_weight  # noqa: E402
from sssp import generators as gen  # noqa: E402


def reference(g: Graph, s: int):
    """(distances, has_reachable_negative_cycle) from Floyd-Warshall."""
    D = floyd_warshall(g)
    reach = [D[s][v] != INF for v in range(g.n)]
    neg = any(reach[v] and D[v][v] < 0 for v in range(g.n))
    return D[s], neg


def check_cycle(g, s, cyc):
    assert cyc and len(cyc) >= 1
    edge_set = {(u, v) for u, v, _ in g.edges}
    for i in range(len(cyc)):
        assert (cyc[i], cyc[(i + 1) % len(cyc)]) in edge_set, "not a real cycle"
    assert cycle_weight(g, cyc) < 0, "cycle is not negative"
    D = floyd_warshall(g)
    assert D[s][cyc[0]] != INF, "cycle not reachable from source"


CASES = [(n, p, seed) for n in (1, 2, 3, 5, 8, 12) for p in (0.2, 0.5) for seed in range(60)]


@pytest.mark.parametrize("name", list(SSSP_ALGORITHMS))
def test_against_floyd_warshall(name):
    algo = SSSP_ALGORITHMS[name]
    neg_seen = pos_seen = 0
    for n, p, seed in CASES:
        g = gen.random_small_graph(n, p, seed=seed, lo=-4, hi=12)
        ref_d, ref_neg = reference(g, 0)
        r = algo(g, 0)
        assert r.has_negative_cycle == ref_neg, (n, p, seed)
        if ref_neg:
            neg_seen += 1
            check_cycle(g, 0, r.negative_cycle)
        else:
            pos_seen += 1
            assert r.dist == ref_d, (n, p, seed)
    assert neg_seen > 50 and pos_seen > 50  # both regimes exercised


@pytest.mark.parametrize("name", list(SSSP_ALGORITHMS))
def test_generators_no_negative_cycle(name):
    algo = SSSP_ALGORITHMS[name]
    for seed in range(5):
        for g in (gen.random_potential_graph(60, 240, seed=seed),
                  gen.grid_potential_graph(7, seed=seed),
                  gen.adversarial_chain(40, seed=seed)):
            ref_d, ref_neg = reference(g, 0)
            assert not ref_neg
            assert any(w < 0 for _, _, w in g.edges)
            r = algo(g, 0)
            assert not r.has_negative_cycle and r.dist == ref_d


@pytest.mark.parametrize("name", list(SSSP_ALGORITHMS))
def test_planted_cycles(name):
    algo = SSSP_ALGORITHMS[name]
    for seed in range(5):
        base = gen.random_potential_graph(50, 200, seed=seed)
        for g in (gen.plant_negative_cycle(base, length=3 + seed, seed=seed),
                  gen.plant_subtle_negative_cycle(base, length=10 + seed, seed=seed)):
            r = algo(g, 0)
            assert r.has_negative_cycle
            check_cycle(g, 0, r.negative_cycle)


def test_negative_self_loop():
    g = Graph(3, [(0, 1, 2), (1, 1, -1), (1, 2, 3)])
    for name, algo in SSSP_ALGORITHMS.items():
        r = algo(g, 0)
        assert r.negative_cycle == [1], name


def test_unreachable_negative_cycle_ignored():
    g = Graph(4, [(0, 1, 5), (2, 3, -2), (3, 2, -2)])
    for name, algo in SSSP_ALGORITHMS.items():
        r = algo(g, 0)
        assert not r.has_negative_cycle, name
        assert r.dist == [0, 5, INF, INF], name


def test_johnson_matches_floyd_warshall():
    for seed in range(40):
        g = gen.random_small_graph(9, 0.35, seed=seed, lo=-3, hi=12)
        D, cyc = johnson(g)
        F = floyd_warshall(g)
        has_neg = any(F[v][v] < 0 for v in range(g.n))
        assert (cyc is not None) == has_neg
        if not has_neg:
            assert D == F
        else:
            assert cycle_weight(g, cyc) < 0


def test_yen_pass_bound():
    # Theorem 5: at most ceil(n/2) passes + 1 confirming pass
    for seed in range(20):
        g = gen.random_potential_graph(80, 300, seed=seed)
        r = SSSP_ALGORITHMS["yen"](g, 0)
        assert r.passes <= (g.n + 1) // 2 + 1
