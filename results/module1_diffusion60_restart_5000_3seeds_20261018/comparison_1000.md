Module 1 diffusion 60³: inputs 1–1000, seeds 1–3

For each seed, total-time reduction is 100*(1-method_total/default_total); reported means and sample standard deviations are across the three seed reductions. Total includes measured learner overhead, every attempted setup/solve, and any default fallback.

| Method | Updated reduction | Official reduction, same seeds/window | Change | Final failures |
|---|---:|---:|---:|---:|
| Default | 0.00% ± 0.00 pp | 0.00% | +0.00 pp | 0/3000 |
| LSTDQ | 7.40% ± 0.25 pp | 7.11% | +0.29 pp | 0/3000 |
| LinUCB | 4.77% ± 4.02 pp | 6.37% | -1.60 pp | 0/3000 |
| LinUCB–LSTDQ | 21.85% ± 4.84 pp | 22.15% | -0.30 pp | 0/3000 |

± denotes the sample standard deviation across seeds, not a confidence interval.

The rerun evaluates the combined algorithm updates; it does not isolate the effect of any individual change.
Execution-time variation can also change learned choices, including the unchanged setup-only branch.

Source commit: 012e78f321e7edaaf4f807ed5bc016548b4b8ed8.

Per-seed values and validation checks are in the adjacent JSON and CSV files.
