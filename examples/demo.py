"""Small worked example: run every algorithm on a graph with negative edges,
then add an edge that creates a negative cycle and show the certificate."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sssp.algorithms import SSSP_ALGORITHMS, johnson  # noqa: E402
from sssp.core import Graph, cycle_weight  # noqa: E402

#   0 --4--> 1 --(-2)--> 2
#   |         ^          |
#   5       (-3)         6
#   v         |          v
#   3 ------> 4 <--------+
#        2
edges = [(0, 1, 4), (1, 2, -2), (0, 3, 5), (3, 4, 2), (4, 1, -3), (2, 4, 6)]
g = Graph(5, edges)

print("Graph without negative cycle (source 0):")
for name, algo in SSSP_ALGORITHMS.items():
    r = algo(g, 0)
    print(f"  {name:20s} dist={r.dist}  relaxations={r.relaxations}")

D, _ = johnson(g)
print("\nJohnson all-pairs distance matrix:")
for row in D:
    print("  ", ["inf" if x == float("inf") else x for x in row])

g.add_edge(2, 0, -1)  # cycle 0->1->2->0 has weight 4-2-1 = 1 (ok); add one more
g.add_edge(4, 3, -3)  # cycle 3->4->3 has weight 2-3 = -1  (negative!)
print("\nAfter adding edges 2->0 (-1) and 4->3 (-3):")
for name, algo in SSSP_ALGORITHMS.items():
    r = algo(g, 0)
    cyc = r.negative_cycle
    print(f"  {name:20s} negative cycle={cyc}  weight={cycle_weight(g, cyc)}")
