"""Benchmark suite. Writes results/*.csv and figures/*.pdf|png.

Usage:  python experiments/benchmark.py            (full run, a few minutes)
        python experiments/benchmark.py --quick    (smaller sizes)
"""
import argparse
import csv
import os
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sssp import generators as gen  # noqa: E402
from sssp.algorithms import (SSSP_ALGORITHMS, floyd_warshall,  # noqa: E402
                             johnson)

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

LABEL = {
    "bellman_ford": "Bellman-Ford (textbook)",
    "bellman_ford_early": "Bellman-Ford + early stop",
    "yen": "Yen ordering",
    "yen_randomized": "Randomized Yen",
    "spfa": "SPFA (queue)",
    "spfa_amortized": "SPFA + amortized check",
    "hybrid": "Hybrid (ours)",
}
STYLE = {
    "bellman_ford": ("#7f7f7f", "o"),
    "bellman_ford_early": ("#1f77b4", "s"),
    "yen": ("#2ca02c", "^"),
    "yen_randomized": ("#17becf", "v"),
    "spfa": ("#ff7f0e", "D"),
    "spfa_amortized": ("#d62728", "P"),
    "hybrid": ("#9467bd", "*"),
}


def run(algo, g, reps=1):
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        r = algo(g, 0)
        times.append(time.perf_counter() - t)
    return min(times), r


def sweep(family, sizes, make, algos, seeds, bf_cap, writer):
    rows = []
    for n in sizes:
        for name in algos:
            if name == "bellman_ford" and n > bf_cap:
                continue
            ts, rel, ps = [], [], []
            for sd in seeds:
                g = make(n, sd)
                t, r = run(SSSP_ALGORITHMS[name], g)
                ts.append(t); rel.append(r.relaxations); ps.append(r.passes)
                neg = r.has_negative_cycle
            row = dict(family=family, n=g.n, m=g.m, algorithm=name,
                       time_s=statistics.mean(ts), relaxations=statistics.mean(rel),
                       passes=statistics.mean(ps), negative_cycle=neg)
            writer.writerow(row); rows.append(row)
            print(f"{family:10s} n={g.n:5d} {name:20s} {row['time_s']:8.3f}s "
                  f"relax={row['relaxations']:.3g} neg={neg}", flush=True)
    return rows


def plot(rows, family, metric, ylabel, fname, title, logy=True, loc="best"):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name in LABEL:
        pts = sorted((r["n"], r[metric]) for r in rows if r["algorithm"] == name)
        if not pts:
            continue
        c, mk = STYLE[name]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=c,
                label=LABEL[name], linewidth=1.6, markersize=5)
    ax.set_xlabel("number of vertices n")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, frameon=False, loc=loc)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{fname}.{ext}"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw figures from results/sssp_results.csv")
    a = ap.parse_args()
    if a.plots_only:
        return replot()
    seeds = [0, 1, 2] if not a.quick else [0]
    sizes = [250, 500, 1000, 2000, 4000] if not a.quick else [200, 400, 800]
    sides = [15, 25, 35, 45, 60] if not a.quick else [10, 15, 20]
    all_algos = list(LABEL)
    no_cycle_algos = [x for x in all_algos if x != "spfa_amortized"]

    f = open(os.path.join(RES, "sssp_results.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=["family", "n", "m", "algorithm", "time_s",
                                      "relaxations", "passes", "negative_cycle"])
    w.writeheader()

    rnd = sweep("random", sizes, lambda n, s: gen.random_potential_graph(n, 4 * n, seed=s),
                no_cycle_algos, seeds, 2000, w)
    grid = sweep("grid", sides, lambda k, s: gen.grid_potential_graph(k, seed=s),
                 no_cycle_algos, seeds, 2000, w)
    chain = sweep("chain", sizes[:4], lambda n, s: gen.adversarial_chain(n, seed=s),
                  no_cycle_algos, seeds, 2000, w)
    negc = sweep("negcycle", sizes,
                 lambda n, s: gen.plant_negative_cycle(
                     gen.random_potential_graph(n, 4 * n, seed=s), length=10, seed=s),
                 all_algos, seeds, 2000, w)
    subtle = sweep("subtle", sizes,
                   lambda n, s: gen.plant_subtle_negative_cycle(
                       gen.random_potential_graph(n, 4 * n, seed=s), length=50, seed=s),
                   all_algos, seeds, 2000, w)
    f.close()

    plot(subtle, "subtle", "time_s", "time to certify cycle (s)", "subtle_time",
         "Subtle negative cycle (50 edges, total weight -1)", loc="upper left")
    plot(rnd, "random", "time_s", "time (s)", "random_time",
         "Random sparse graphs (m = 4n), no negative cycle")
    plot(rnd, "random", "relaxations", "edge relaxations", "random_relax",
         "Random sparse graphs: work (edge examinations)")
    plot(grid, "grid", "time_s", "time (s)", "grid_time",
         "Bidirected grid graphs, no negative cycle")
    plot(chain, "chain", "relaxations", "edge relaxations", "chain_relax",
         "Adversarial chain (textbook BF coincides with early stop)")
    plot(negc, "negcycle", "time_s", "time to certify cycle (s)", "negcycle_time",
         "Planted negative cycle: time to return a certificate", loc="upper left")

    # APSP: Johnson vs Floyd-Warshall
    apsp = []
    for n in ([50, 100, 150, 200, 300] if not a.quick else [30, 60]):
        g = gen.random_potential_graph(n, 4 * n, seed=0)
        t0 = time.perf_counter(); johnson(g); tj = time.perf_counter() - t0
        t0 = time.perf_counter(); floyd_warshall(g); tf = time.perf_counter() - t0
        apsp.append((n, g.m, tj, tf))
        print(f"apsp n={n} johnson={tj:.3f}s floyd={tf:.3f}s", flush=True)
    with open(os.path.join(RES, "apsp_results.csv"), "w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["n", "m", "johnson_s", "floyd_warshall_s"])
        cw.writerows(apsp)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot([x[0] for x in apsp], [x[2] for x in apsp], "o-", label="Johnson (BF + n Dijkstra)")
    ax.plot([x[0] for x in apsp], [x[3] for x in apsp], "s-", label="Floyd-Warshall")
    ax.set_xlabel("number of vertices n"); ax.set_ylabel("time (s)")
    ax.set_title("All-pairs shortest paths on sparse graphs (m = 4n)", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"apsp_time.{ext}"), dpi=200)
    plt.close(fig)


def replot():
    rows = []
    with open(os.path.join(RES, "sssp_results.csv")) as fh:
        for r in csv.DictReader(fh):
            for k in ("n", "m"):
                r[k] = int(r[k])
            for k in ("time_s", "relaxations", "passes"):
                r[k] = float(r[k])
            rows.append(r)
    fam = lambda f: [r for r in rows if r["family"] == f]
    plot(fam("subtle"), "subtle", "time_s", "time to certify cycle (s)", "subtle_time",
         "Subtle negative cycle (50 edges, total weight -1)", loc="upper left")
    plot(fam("random"), "random", "time_s", "time (s)", "random_time",
         "Random sparse graphs (m = 4n), no negative cycle")
    plot(fam("random"), "random", "relaxations", "edge relaxations", "random_relax",
         "Random sparse graphs: work (edge examinations)")
    plot(fam("grid"), "grid", "time_s", "time (s)", "grid_time",
         "Bidirected grid graphs, no negative cycle")
    plot(fam("chain"), "chain", "relaxations", "edge relaxations", "chain_relax",
         "Adversarial chain (textbook BF coincides with early stop)")
    plot(fam("negcycle"), "negcycle", "time_s", "time to certify cycle (s)", "negcycle_time",
         "Planted negative cycle: time to return a certificate", loc="upper left")


if __name__ == "__main__":
    main()
