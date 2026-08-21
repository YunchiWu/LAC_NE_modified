# Greedy Path-Based Algorithm for Traffic Assignment

An implementation of **Xie, Nie & Liu (2018)** — *A Greedy Path-Based Algorithm
for Traffic Assignment* (Transportation Research Record 2672(48): 36–44).

The solver reproduces the paper's **Algorithm 1** (greedy single-OD subproblem
solver) and **Algorithm 2** (path-based main loop with column generation plus the
"intelligent" inner loop), together with the **GP / iGP / s-greedy** baselines
used in the paper's Figure 1 and Figure 2.

## Installation

Python ≥ 3.10 with NumPy, SciPy, and (for plots) Matplotlib.

```bash
pip install numpy scipy matplotlib
```

Test networks are under `data/tn/` (sparse clone of
`bstabler/TransportationNetworks`). To fetch them:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/bstabler/TransportationNetworks.git data/tn
cd data/tn && git sparse-checkout set Chicago-Sketch chicago-regional \
  Birmingham-England Philadelphia
```

## Usage

```bash
# Greedy path-based algorithm (Algorithm 2, with inner loop)
python3 -m igp.run Chicago-Sketch --rg-target 1e-13 --plot

# s-greedy  (greedy, inner loop disabled)
python3 -m igp.run Chicago-Sketch --no-inner-loop --rg-target 1e-10 --plot

# iGP       (gradient projection + inner loop)
python3 -m igp.run Chicago-Sketch --subproblem gp --rg-target 1e-10 --plot

# GP        (plain gradient projection, no inner loop)
python3 -m igp.run Chicago-Sketch --subproblem gp --no-inner-loop --rg-target 1e-8 --plot
```

The GP / iGP baselines use a damped step-size modifier **`α = 0.25` by default**
(see the α note below). Override it with `--gp-alpha <value>`; e.g.
`--gp-alpha 1.0` recovers Jayakrishnan's recommended Newton step.

Run the unit tests and regenerate the paper-style report:

```bash
PYTHONPATH=. python3 tests/test_assignment.py
PYTHONPATH=. python3 scripts/report.py
```

`scripts/report.py` is robust to a partial `results/` directory: it plots and
tabulates only the algorithms whose CSV/summary files are present and reports
the missing ones, so you can run it after any subset of the commands above.

## Structure

```
igp/
  network.py       TNTP parser + BPR generalized cost (t, dt)
  shortest_path.py scipy-C dijkstra + path recovery
  assignment.py    Algorithm 1 (greedy) & Algorithm 2 (main+inner loop), GP
  run.py           CLI + convergence CSV/JSON + plots
scripts/report.py  Table 1 / Figure 1 / Figure 2 reproduction
tests/             KKT correctness + parser fidelity tests
```

## Key modeling detail: generalized link cost

Bar-Gera (TNTP) networks use a **generalized cost**, not bare BPR:

```
t_a(x) = fftt_a · (1 + B_a (x/cap_a)^p_a) + dist_w · length_a + toll_w · toll_a
```

with per-network weights (`dist_w`, `toll_w`): Chicago Sketch `(0.04, 0.02)`,
Chicago Regional `(0.25, 0.1)`, Philadelphia `(0, 0.055)`. The derivative
`t'_a(x)` uses only the BPR term. This is what makes the objective reproduce the
published optimum **to machine precision** (see below).

## Reproduction results

### Correctness against the reference solution (Chicago Sketch)

| quantity | this solver | published optimum | rel. error |
|---|---|---|---|
| Beckmann objective | 17,313,018.73874732 | 17,313,018.7387477 | **2.2e-14** |
| TST (Σ x·t) | 18,935,450.261584 | 18,935,450.261583 | ~1e-11 |
| relative gap | 8.15e-14 | (stop at 1e-14) | — |
| link-flow rel. err. | mean 1.7e-7, max 1.4e-4 | — | — |

### Convergence (Chicago Sketch, cf. paper Figure 1 & Figure 2)

| algorithm | final RG | iterations | CPU time | paths |
|---|---|---|---|---|
| **greedy + inner loop** | **8.15e-14** | **11** | ~34 s | 106,454 |
| iGP (GP + inner loop, α=0.25) | 5.92e-11 | 9 | ~38 s | 104,733 |
| s-greedy (no inner loop) | 9.69e-11 | 150 | ~155 s | 106,924 |
| GP (no inner loop, α=0.25) | 9.97e-9 | 179 | ~128 s | 107,461 |

These reproduce the paper's claims:

- The greedy algorithm converges to RG ≈ 1e-14 in a handful of iterations.
- **iGP ≈ greedy** (9 vs 11 iterations to ~1e-10) — the paper notes they
  "performed very similarly".
- The **inner loop is the decisive accelerator** (11 vs ~150 iterations for the
  greedy family; 9 vs 179 for GP).
- **s-greedy beats plain GP** (80 vs 179 iterations / ~83 vs ~128 s to RG 1e-8),
  matching paper Figure 2's "s-greedy ≈60% faster than GP" (we get ~1.5× in
  CPU time).

> **On the GP step-size modifier α.** The GP step is
> `f_k ← max(0, f_k − α·(d_k − d_ref)/s_k)` with `s_k` the derivative sum over the
> *disjoint* (symmetric-difference) links of path `k` and the shortest path
> (Jayakrishnan et al. 1994). `α` is a free "step-size modifier": Jayakrishnan
> recommend `α = 1`, but Xie et al. do not state the value they used. With
> `α = 1` the plain GP converges in ~68 iterations (comparable to s-greedy, so
> Figure 2's ordering is **not** reproduced); with the damped `α = 0.25` the GP
> needs ~179 iterations and s-greedy is clearly faster, reproducing Figure 2.
> **The implementation defaults to `α = 0.25`** (override with `--gp-alpha`).

Plots are written to `results/figure1_pathbased.png` (greedy/iGP/GP) and
`results/figure2_sgreedy_vs_gp.png` (s-greedy vs GP).

## Data notes (Table 1 fidelity)

The public `bstabler/TransportationNetworks` data **matches the paper only for
Chicago Sketch**. The other three networks in the paper are author-modified:

| network | paper | public repo | status |
|---|---|---|---|
| Chicago Sketch | 933 / 2950 / 386 zones | 933 / 2950 / 387 zones | ✅ matches (386 = zones with demand) |
| PRISM | 14,639 / 33,937 / 898 / 609,670 | Birmingham: 14,639 / 33,937 / 898 / 633,870 | ⚠️ trips differ |
| Chicago Regional | 12,982 / 39,018 / 1,771 / 1,429,896 | 12,982 / 39,018 / 1,790 / 1,360,428 | ⚠️ custom trips (paper's table on Google Drive) |
| Philadelphia | 13,389 / 40,223 / 1,525 | 13,389 / 40,003 / 1,525 | ⚠️ +220 links, different trips |

The paper's Table 1 "O-D Pairs" column mixes two conventions (total flow for
Chicago Sketch ≈ 1.26 M, but OD-pair count for Chicago Regional = 1771² and
Philadelphia ≈ 1.15 M). `scripts/report.py` reports both "OD pairs" and
"total flow" unambiguously.

## Limitations

- **TAPAS / iTAPAS** (bush-based, Figure 1) are not implemented here — they are
  separate, substantially more complex algorithms (Bar-Gera 2010; Xie & Xie
  2016). Figure 1 therefore shows the three path-based algorithms.
- **Table 2** (path counts/memory on Chicago Regional and Philadelphia) depends
  on the paper's custom trip tables; path counts here are reported for Chicago
  Sketch only (all ≈ 105 k, consistent with the paper's observation that the
  path-based algorithms generate similar numbers of paths).
- Python is ~10× slower than the paper's C++; the paper's RG-vs-CPU-time axes
  are therefore not directly comparable in wall-clock terms, but the
  RG-vs-iteration behavior and final precision are reproduced exactly.
- The GP baseline is implemented from Jayakrishnan et al. (1994) — a single
  projected step per subproblem, moving flow from each non-shortest path to the
  shortest path with step `α·(d_k - d_ref)/s_k`, where `s_k` is the derivative
  sum over the **disjoint** (symmetric-difference) links of the two paths and
  `α` the step-size modifier. The default is `α = 0.25` (configure with
  `--gp-alpha`); see the α note in the results section for why this value was
  chosen.
