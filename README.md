# CSD319 End-Term Project: Breaking the Bellman–Ford Barrier

**Problem 2: Shortest-Path Algorithms for Graphs with Negative Edge Weights**

| Name | Roll Number |
|---|---|
| Pragun Kathuria | 2410110237 |
| Amitoj Oberoi | 2410110046 |
| Tanvi Pandey | 2410110356 |
| Vishnu Sriram | 2410110629 |

## Deliverables

| File | What it is |
|---|---|
| `CSD319_Project_Report.pdf` | Final report: problem statement, algorithms, proofs of correctness, complexity analysis, experiments, conclusion, gaps and future directions |
| `CSD319_Project_Slides.pdf` | Presentation slides (16:9) |
| `CSD319_Video_Script.pdf` | Per-member script for the 6-minute video presentation |
| `CSD319_Project_Submission_Details.pdf` | Project details for the Google Form (title, members, description) |
| `report/report.tex`, `slides/slides.tex` | LaTeX sources |
| `src/sssp/` | Python implementation of all algorithms |
| `tests/test_sssp.py` | Correctness tests (validated against Floyd–Warshall) |
| `experiments/benchmark.py` | Reproduces every figure and table |
| `results/`, `figures/` | Raw benchmark data (CSV) and plots |

## Algorithms implemented (`src/sssp/algorithms.py`)

- Bellman–Ford (textbook, plus early termination)
- Yen's two-DAG ordering (≤ ⌈n/2⌉ passes) and the Bannister–Eppstein randomized ordering
- SPFA (queue-based Bellman–Ford), classic and with amortized negative-cycle checks
- **Hybrid (ours):** BFS-ordered Yen passes + early stop + a predecessor-graph cycle check after each pass
- Dijkstra, Johnson (all pairs), Floyd–Warshall (reference)

Every routine either returns distances or a **verified negative-cycle certificate** (a cycle reachable from the source with negative total weight).

## How to run

Requires Python 3.9+, `pytest` and `matplotlib`.

```bash
pip install -r requirements.txt
python -m pytest -q tests            # 25 correctness tests
python experiments/benchmark.py      # full benchmarks (~5–10 min), writes results/ and figures/
python experiments/benchmark.py --quick
python examples/demo.py              # small worked example
```

To rebuild the PDFs: `cd report && latexmk -pdf report.tex`, and `cd slides && latexmk -pdf slides.tex`.
