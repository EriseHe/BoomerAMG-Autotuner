Module 1 diffusion 60 updated algorithm restart, seeds 1-3

All four methods were measured independently from their first paired input. Native time includes every setup/solve attempt and fallback. Inclusive time also includes measured learner costs.

| Seed | Method | Native seconds | Native reduction | Inclusive reduction | Final failures |
|---|---|---:|---:|---:|---:|
| 1 | Default | 1438.879 | 0.00% | 0.00% | 0 |
| 1 | Default + LSTDQ | 1308.557 | 9.06% | 8.23% | 0 |
| 1 | LinUCB | 1028.629 | 28.51% | 27.46% | 0 |
| 1 | LinUCB + LSTDQ | 762.727 | 46.99% | 44.73% | 0 |
| 2 | Default | 1441.179 | 0.00% | 0.00% | 0 |
| 2 | Default + LSTDQ | 1311.235 | 9.02% | 8.20% | 0 |
| 2 | LinUCB | 1028.573 | 28.63% | 27.57% | 0 |
| 2 | LinUCB + LSTDQ | 784.640 | 45.56% | 43.23% | 0 |
| 3 | Default | 1435.889 | 0.00% | 0.00% | 0 |
| 3 | Default + LSTDQ | 1306.613 | 9.00% | 8.17% | 0 |
| 3 | LinUCB | 1031.872 | 28.14% | 27.08% | 0 |
| 3 | LinUCB + LSTDQ | 801.852 | 44.16% | 41.78% | 0 |

Per-input trajectories and final setup/controller checkpoints are in each seed directory. Source and native hashes, feature checks, activation checks and batch reconstruction of both setup regressions passed.

[Paper figures](paper_figures/module1_paper_figures.pdf) · [Joint LSTDQ heat maps](paper_figures/joint_lstdq_heatmaps_all_seeds.png) · [Plot data and provenance](paper_figures/manifest.json)
