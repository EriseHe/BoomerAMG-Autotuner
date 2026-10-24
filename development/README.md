Start algorithm changes, tests, scripts, and diagnostic experiments on
`codex/development`, and validate them there. Promote only selected, validated
algorithm changes and official Module 1–3 runner updates to `main`. Keep tests,
diagnostic scripts, and unofficial experiment records on development; do not
merge the complete development branch into main. After promotion, merge any
main changes back into development so development contains the released code.

`retry_comparison.py` retains the conservative runtime retry selector and its
comparison registry. Official runners use the shared failure-aware learner in
`experiments/paper_final/recovery/setup_feedback.py`; they do not import the
development alternatives.

`run_retry_comparison.py` runs the Module 3 comparison in three sequential
batches: fixed-coefficient failure scoring, failure scoring with the runtime
alpha schedule, and conservative runtime scoring without a failure predictor.
It uses the official runner with concurrent seed workers and frozen source
snapshots. All variants retain the same adjacent-cost targets and successful
Default-fallback feedback. For seeds 1–3:

```sh
python -m development.run_retry_comparison sequence \
  --suite configs/module3/suites/variable_flow_60_batch1.json \
  --output results/module3_retry_comparison
```

The tests in `tests/` check state encoding, scoring arithmetic, cost accounting,
and frozen evaluation. Official experiment runners do not import or execute
them. The scoring-equivalence check compares the controller with the direct
matrix expression; it adds no work to an official solver run.

`setup_normalization.py` uses fixed parameter ranges to map the five numerical
setup descriptors to [-1, 1] for both RL controllers. The categorical reference
indicators and conditional masks are unchanged; the bandit keeps its existing
encoder. `run_setup_normalization.py` applies this development experiment through
the Module 1 runner, preserving its continuous learning and frozen phases:

```sh
python -m development.run_setup_normalization sequence \
  --suite configs/module1/suites/advection_60_batch1.json \
  --suite configs/module1/suites/advection_60_batch2.json \
  --output results/setup_normalization_advection_60
```

`shared_setup_features.py` uses the shared 104-term setup-action feature structure
in both RL controllers. By default, the five numerical inputs use the earlier
19-feature experiment's fixed range mapping to [-1, 1] before forming squares
and interactions. Categories and their conditional masks are unchanged, and
all 104 resulting features lie within [-1, 1] for allowed configurations.
The bandit and the other solve features remain unchanged. The wrapper below
runs the first 5,000 paired inputs with concurrent
seeds 1–3 and all four branches. It does not execute a frozen phase:

```sh
python -m development.run_shared_setup_features prepare \
  --suite configs/module1/suites/advection_60_batch1.json \
  --cases 5000 --normalization range --output results/shared_setup_features_advection_60
```

Run `python -m development.run_shared_setup_features run --output /absolute/run/path`
from the prepared run's `source/` directory, with that directory on `PYTHONPATH`.
The runner verifies the numerical ranges and matching categorical structure
before measurement and preserves source snapshots and native/inclusive timings.
`--normalization bandit` explicitly selects the earlier unbounded scaling for a
separate comparison. Run and worker commands read the prepared normalization
from the saved protocol; they do not change an existing run's scaling.

`run_compact_context.py` uses the 19 range-normalized setup descriptors and
absolute logarithmic cycle time in both RL controllers. Each controller has
32 state entries and 288 state-action coefficients. It measures learning
inputs only with the Module 1 runner:

```sh
python -m development.run_compact_context prepare \
  --suite configs/module1/suites/advection_60_batch1.json \
  --cases 5000 --output results/normalized_setup_log_time
```

Run `python -m development.run_compact_context run --output /absolute/run/path`
from the prepared run's `source/` directory, with that directory on `PYTHONPATH`.

The compact runner accepts `--cycle-time first_cycle_ratio` during preparation
to compare the first native cycle reference with the default `log` context.
The selected formula is frozen in the protocol and encoder checkpoint metadata;
both modes use 32 state entries and differ only at the timing entry.
`run_cycle_time_comparison.py` executes an explicitly prepared sequence serially,
verifies completion and matching candidate arrays for the timing pair, and
reports native and inclusive savings. An archived original-paper baseline uses
its own verified source and original feature map; it is a full algorithm
comparison rather than an isolated timing-feature change.

`cycle_bins.py` adds the seven original cycle-range indicators and their seven
residual-gap products to the current solve context, retaining every existing
feature value. `run_joint_context.py` measures only Joint and Default with
46 state entries and 414 state-action coefficients for `--context cycle_bins`.
Select `--context log_time` to retain the original compact 32-entry state and
288 state-action coefficients, without bins or additional quadratic terms.
Its suite must explicitly
contain those two methods. It reuses the current setup learner, native episode
execution, fallback feedback, final-model audit, and overhead accounting.
Tests verify feature equality, checkpoint identity and physical execution of
exactly two branches. These are development diagnostics; earlier four-branch
runs have a different within-input measurement order.

`quadratic_context.py` completes the quadratic basis in cycle fraction and
normalized residual gap by adding only `(k/K_max)^2` and `(k/K_max)*g` to
the compact state. Select it with `run_joint_context prepare --context quadratic`.
It uses 34 state entries and 306 state-action coefficients, without cycle bins.
The configuration is frozen during preparation and recorded in checkpoints.
All existing observations, numerical setup normalization, native-cycle reward,
retry selection, fallback feedback, and overhead accounting remain the same.

`run_original_joint.py` measures Joint and Default using an isolated original
source checkout. It verifies all 151 recorded source hashes, original cold
learner checkpoints and encoder metadata before measurement. The algorithm
comes entirely from that checkout; the adapter supplies the same two-method
measurement loop and ordering used by the current context diagnostics.

The RL checks cover state normalization, scoring, and timing accounting.
The frozen-evaluation checks cover Module 1's 1,000-input frozen phase.
`frozen_evaluation_source.json` records the original source
snapshot, and `validation-source.json` records the validation. The development
checkout is self-contained; run the checks there:

```sh
cd /path/to/RLTune-BoomerAMG-diagnostics
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  python -m unittest discover \
  -s development/tests -p 'test_*.py' -v

python development/verify_data.py
```

The selected scripts and compact reports under `results/rl_*` are development
records. Full trajectories, checkpoints, source snapshots, and candidate
schedules remain local ignored artifacts. Historical scripts retain their
original run assumptions and must be checked before another run.

The cycle-bin comparison has five completed runs out of six; the final
without-bins run for seed 3 was interrupted. It is not a completed three-seed
paired study. None of these diagnostics replace the official Module 1–3 data.

The completed `results/module1_diffusion60_restart_5000_3seeds_20261018` rerun
also belongs to development. Its selected analysis scripts, configurations,
provenance, and compact reports are versioned here. Full trajectories,
checkpoints, and generated figures remain local ignored artifacts. This run
uses three seeds and 5,000 inputs per seed and is separate from the original
official Module 1 measurements. Its recorded source is `012e78f`; the later
redundant-guard cleanup preserves feature values for valid observations.
