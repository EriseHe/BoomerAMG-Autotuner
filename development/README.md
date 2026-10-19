Start algorithm changes, tests, scripts, and diagnostic experiments on
`codex/development`, and validate them there. Promote only selected, validated
algorithm changes and official Module 1–4 runner updates to `main`. Keep tests,
diagnostic scripts, and unofficial experiment records on development; do not
merge the complete development branch into main. After promotion, merge any
main changes back into development so development contains the released code.

The tests in `tests/` check state encoding, scoring arithmetic, cost accounting,
and frozen evaluation. Official experiment runners do not import or execute
them. The scoring-equivalence check compares the controller with the direct
matrix expression; it adds no work to an official solver run.

The RL checks cover the published first-cycle time normalization and
overhead-inclusive learning cost. The earlier completed scoring optimization
is retained; the interrupted follow-up round introduced no controller changes.
The frozen-evaluation checks additionally cover pending Module 4 changes now
included on development. `pending_module4_source.json` records their source
snapshot, and `validation-source.json` records the validation. The development
checkout is self-contained; run the checks there:

```sh
cd /path/to/RLTune-BoomerAMG-diagnostics
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  python -m unittest discover \
  -s development/tests -p 'test_*.py' -v

python tools/verify_data.py
```

The selected scripts and compact reports under `results/rl_*` are development
records. Full trajectories, checkpoints, source snapshots, and candidate
schedules remain local ignored artifacts. Historical scripts retain their
original run assumptions and must be checked before another run.

The cycle-bin comparison has five completed runs out of six; the final
without-bins run for seed 3 was interrupted. It is not a completed three-seed
paired study. None of these diagnostics replace the official Module 1–4 data.

The completed `results/module1_diffusion60_restart_5000_3seeds_20261018` rerun
also belongs to development. Its selected analysis scripts, configurations,
provenance, and compact reports are versioned here. Full trajectories,
checkpoints, and generated figures remain local ignored artifacts. This run
uses three seeds and 5,000 inputs per seed and is separate from the original
official Module 1 measurements. Its recorded source is `012e78f`; the later
redundant-guard cleanup preserves feature values for valid observations.
