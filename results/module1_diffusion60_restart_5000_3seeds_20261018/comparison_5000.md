Module 1 diffusion 60³: inputs 1–5000, seeds 1–3

For each seed, total-time reduction is 100*(1-method_total/default_total); reported means and sample standard deviations are across the three seed reductions. Total includes measured learner overhead, every attempted setup/solve, and any default fallback.

| Method | Updated reduction | Official reduction, same seeds/window | Change | Final failures |
|---|---:|---:|---:|---:|
| Default | 0.00% ± 0.00 pp | 0.00% | +0.00 pp | 0/15000 |
| LSTDQ | 8.20% ± 0.03 pp | 7.89% | +0.31 pp | 0/15000 |
| LinUCB | 27.37% ± 0.25 pp | 26.86% | +0.52 pp | 0/15000 |
| LinUCB–LSTDQ | 43.25% ± 1.48 pp | 43.38% | -0.14 pp | 0/15000 |

± denotes the sample standard deviation across seeds, not a confidence interval.

The rerun evaluates the combined algorithm updates; it does not isolate the effect of any individual change.
Execution-time variation can also change learned choices, including the unchanged setup-only branch.

Source commit: 012e78f321e7edaaf4f807ed5bc016548b4b8ed8.

Per-seed values and validation checks are in the adjacent JSON and CSV files.

All three seeds completed the requested 5,000 inputs for all four branches (60,000 method-input evaluations). The runner checks passed, and independently recalculated totals and reductions matched the runner summary.

| Seed | Joint total-time reduction | Original, same seed |
|---|---:|---:|
| 1 | 44.73% | 43.22% |
| 2 | 43.23% | 42.86% |
| 3 | 41.78% | 44.07% |

[Per-seed CSV](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/module1_diffusion60_restart_5000_3seeds_20261018/comparison_5000.csv) · [Calculation and checks](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/module1_diffusion60_restart_5000_3seeds_20261018/comparison_5000.json) · [Run report](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/module1_diffusion60_restart_5000_3seeds_20261018/run/report.md) · [Figures and captions](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/module1_diffusion60_restart_5000_3seeds_20261018/run/paper_figures/module1_paper_figures.pdf)

Cumulative-time figures show native time, excluding learner overhead. The runtime-breakdown figure and the total-time reductions above include measured learner overhead.
