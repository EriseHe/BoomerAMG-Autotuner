# Experiments

Run commands from the repository root after installing the package and building
HYPRE. Use fresh output directories. Each online suite uses three single-thread
workers, one per seed, with four independent cold methods from case 1.

## Module 1: online learning and frozen evaluation

Module 1 has two consecutive phases per seed: 5,000 online-learning inputs
(1–5,000), followed by 1,000 frozen-evaluation inputs (5,001–6,000).
Fresh runs use one worker process and one uninterrupted loop for all 6,000
inputs. Add the matching held-out configuration's `stream` object as the
top-level `frozen_stream` alongside the original 5,000-input `stream`. The
runner concatenates these pinned blocks without reshuffling the training
prefix, allocates the continuing candidate schedule, and disables both
learners in place after input 5,000. It preserves the live objects, RNG,
history, and caches; it neither rebuilds the learners nor reloads checkpoints
at that boundary. Epsilon exploration is disabled for the last 1,000 inputs,
while inference overhead continues to be measured. Summaries include separate
`learning` and `frozen` windows as well as the complete stream, and the final
audit checks that learned state stayed unchanged after the boundary.

The supplied archived configurations describe the original 5,000-input runs.
For each JSON file in `data/module1/suites/`, prepare and run the learning phase
sequentially. For example:

```sh
python -m experiments.paper_final.run_online prepare --suite data/module1/suites/diffusion_60_batch1.json --output results/module1/diffusion_60_batch1
python -m experiments.paper_final.run_online run --output results/module1/diffusion_60_batch1
```

The learning phase has 5,000 paired inputs. Methods are Default, fixed-setup LSTDQ,
setup-only LinUCB, and joint LinUCB–LSTDQ. The configurations specify β=2,
1,024 setup candidates, aggressive levels 0–2 with interpolation 4/6/7, cycle
weights 1–3, tolerance 10⁻⁶, and a 50-cycle cap. Both paths apply the same
cap-first stopping rule. Native time includes all setup/solve attempts, retries,
and fallbacks. Inclusive time also includes learner work. Assembly and file I/O
are excluded. Training and frozen-evaluation summaries are reported separately.

### Frozen phase: inputs 5,001–6,000

The frozen phase uses the same `run_online` runner, samplers, four branches, setup search,
confidence scores, retry budgets, smoother, tolerances, native build, and timing
definitions as Module 1. It loads each branch's final released checkpoint after
5,000 training inputs, including the LinUCB failure model, candidate statistics,
elite cache, RNG, and last training setup anchor. LSTDQ weights and covariance
are restored from the matching branch checkpoint.

All learner updates are disabled. Learned arrays are read-only, update entry
points raise an error, and the final audit compares every learned checkpoint
payload with both its initial state and the released Module 1 checkpoint.
LSTDQ epsilon exploration is disabled during evaluation; its trained confidence
score remains unchanged. LinUCB also retains its trained confidence rule. RNG
tie-breaking, per-solve execution counters, and the deterministic candidate
schedule cursor may advance; no model or candidate statistics are updated.
Setup selection, candidate scoring, and solve-controller overhead are still
measured and included in total recorded time. Both setup and solve updates are
disabled, so their update time is zero. Native time remains separate.

Each of the eight original sampling streams contributes its next 125 child
instances, at indices 625–749, shuffled with the original shuffle seed. The
training inputs are preserved; extending and reshuffling the original training
pool would change that prefix. All 1,000 evaluation inputs are pinned and
checked against all six training streams of the same PDE/grid combination.
Evaluation rows have global problem indices 5,001–6,000. Setup candidate rows
and method execution orders also continue at the 5,000-input boundary.

All six seeds are evaluated for diffusion and advection at 40³, 60³, and 80³:
36 trained runs and 144,000 independently measured method/input cases.
Run the twelve suites sequentially, with three single-thread seed workers per
suite. The existing runner can prepare and execute this queue:

```sh
# Generate configurations and pinned inputs once, when adding the experiment.
python -m experiments.paper_final.run_online prepare-frozen
python -m experiments.paper_final.run_online prepare-all --suite data/module1/frozen/suites --output results/module1/frozen
python -m experiments.paper_final.run_online run-all --output results/module1/frozen
```

The supplied frozen configurations and inputs are already prepared; the first
command is only needed when generating a fresh set. Each frozen configuration
identifies its matching 5,000-input training configuration and checkpoints.
Use checkpoints with the same algorithm and state encoding as their training
run; archived measurements retain their original configuration and source hashes.
For a fresh rerun, point `evaluation.checkpoint_directory` at the matching
completed training seed directory and `evaluation.training_configuration` at
its `experiment_config.json`. The runner verifies the completed run's model source,
native build, input/configuration hashes, and audited checkpoint/trajectory
hashes before freezing it. Released checkpoints retain their original inventory
checks.

The final frozen queue summary is saved under `results/module1/frozen` and
`data/module1/frozen`.
Each suite retains raw trajectories, source/native hashes, checkpoint copies,
held-out-input checks, and frozen-state audits. Original Module 1 measurements
and checkpoints are never overwritten.

## Module 2

The six supplied checkpoints come from Module 1 diffusion 60³.

```sh
python -m experiments.paper_final.run_policy prepare --config data/module2/configuration.json --output results/module2
python -m experiments.paper_final.run_policy run --output results/module2
```

Scan all 41 weights on 100 independent cases, freeze global/per-case choices,
then independently evaluate five policy roles with three repetitions. Identical
execution cells are shared. Continuation costs exclude the common primary setup
and include recovery costs; inclusive costs also include controller work.

## Module 3

For each JSON file in `data/module3/suites/`, use the same `prepare` and `run`
commands with a fresh Module 3 output directory. Each seed has 5,000 online
inputs. Module 3 retains failed cases and includes common-success comparisons;
the additional frozen phase belongs to Module 1.
