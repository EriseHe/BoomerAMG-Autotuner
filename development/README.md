Development checks and diagnostic experiments live on `codex/development`.
Production algorithm code and the official Module 1–4 runners belong on `main`.

The tests in `tests/` check state encoding, scoring arithmetic, cost accounting,
and frozen evaluation. Official experiment runners do not import or execute
them. The scoring-equivalence check compares the controller with the direct
matrix expression; it adds no work to an official solver run.

The RL checks cover the published first-cycle time normalization and
overhead-inclusive learning cost. The earlier completed scoring optimization
is retained; the interrupted follow-up round introduced no controller changes.
The frozen-evaluation checks additionally cover pending Module 4 changes in the
local `main` working tree. `validation-source.json` records the tested revision
and pending source files. Run the checks against that working tree:

```sh
cd /path/to/RLTune-BoomerAMG
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  python -m unittest discover \
  -s /path/to/RLTune-BoomerAMG-diagnostics/development/tests -p 'test_*.py' -v
```

The selected scripts and compact reports under `results/rl_*` are development
records. Full trajectories, checkpoints, source snapshots, and candidate
schedules remain local ignored artifacts. Historical scripts retain their
original run assumptions and must be checked before another run.

The cycle-bin comparison has five completed runs out of six; the final
without-bins run for seed 3 was interrupted. It is not a completed three-seed
paired study. None of these diagnostics replace the official Module 1–4 data.
