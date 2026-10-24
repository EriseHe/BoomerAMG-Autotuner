# Experiments

Run commands from the repository root after installing the package and building
HYPRE. Use a new output directory for each run. Active configurations are in
`configs/`; the state encoding is `shared_context`.

## Complete study

```sh
python -m experiments.run_all --output results/full_study
```

This runs all six Module 1 family/grid suites in two batches each, Module 2
using the resulting six Diffusion 60 joint models, and both Module 3 batches.
Module 1 includes both its learning and frozen parts. Three seeds run
concurrently within each online batch; batches and modules run sequentially.
The Module 1 order is Advection 60, 40, 80, then Diffusion 40, 60, 80.

Use `--dry-run` to inspect the full command plan without starting a run.
Use `--module2-source advection_60` to select another Module 1 family/grid
for Module 2. Its paths always refer to the new study's Module 1 outputs.
The wrapper uses the existing module commands, records their logs and status,
and stops on an error or a change to source/configuration files. Choose a new
output directory for each study; interrupted streams are not resumed.

## Module 1: online learning and frozen evaluation

Each seed runs one continuous stream in a single worker process:

- **Part 1 — online learning, inputs 1–5,000:** both LinUCB and LSTDQ learn from input one.
- **Part 2 — frozen evaluation, inputs 5,001–6,000:** both learners are frozen and LSTDQ epsilon exploration is disabled.

All four branches are measured independently: Default, fixed-setup LSTDQ,
setup-only LinUCB, and joint LinUCB–LSTDQ. Each suite starts three concurrent
single-thread seed workers. Run the two batches to cover all six seeds.

```sh
python -m experiments.paper_final.run_online prepare --suite configs/module1/suites/diffusion_60_batch1.json --output results/module1/diffusion_60_batch1
python -m experiments.paper_final.run_online run --output results/module1/diffusion_60_batch1
```

Repeat for `diffusion_60_batch2.json` and the other grid/family suites as needed.
Each configuration contains the pinned learning `stream` and held-out
`frozen_stream`. Their concatenation preserves the first 5,000 inputs and
keeps the last 1,000 disjoint. The worker freezes the live learners after
input 5,000 without restarting, rebuilding the models, or reloading checkpoints.
RNG, candidate schedule, history, and caches continue in the same process.
The final audit verifies that learned state remains unchanged during evaluation.

Candidate scoring, setup selection, and solve-controller overhead are measured
in both phases. Frozen inputs have no setup or RL update work. Native time
includes every setup/solve attempt, retry, and fallback. Total recorded time
also includes learner/controller work; assembly and file I/O are excluded.
Summaries report Part 1 as the `learning` window, Part 2 as the `frozen` window,
and both together as the `full` stream.

The configurations use confidence coefficient 2, 1,024 setup candidates,
relaxation weights 1–3, tolerance 10⁻⁶, and a 50-cycle cap. The solve controller
uses the shared categorical setup encoding with fixed-range numerical
descriptors, tolerance-based residual normalization, and
`log(1 + preceding_cycle_seconds / 0.001)` for the cycle-time context.

## Module 2: matched policy evaluation

After Module 1 is complete, create a configuration using its trained joint
models. Specify one seed directory per source; no run is selected automatically.
Module 2 uses the learned models from Part 1; Part 2 preserves their learned
parameters. Source paths are supplied when this study is initialized, so the
repository does not bind Module 2 to a particular Module 1 experiment.
For example, with the output layout above:

```sh
python -m experiments.paper_final.run_policy init --config results/module2/configuration.json --random-seed 20261019 \
  --training-run 1=results/module1/diffusion_60_batch1/seed_1 \
  --training-run 2=results/module1/diffusion_60_batch1/seed_2 \
  --training-run 3=results/module1/diffusion_60_batch1/seed_3 \
  --training-run 4=results/module1/diffusion_60_batch2/seed_4 \
  --training-run 5=results/module1/diffusion_60_batch2/seed_5 \
  --training-run 6=results/module1/diffusion_60_batch2/seed_6
python -m experiments.paper_final.run_policy validate --config results/module2/configuration.json
python -m experiments.paper_final.run_policy prepare --config results/module2/configuration.json --output results/module2/run
python -m experiments.paper_final.run_policy run --output results/module2/run
```

Preparation copies and verifies the models and selects frozen hierarchies; it
executes no native solves. The run scans all 41 fixed weights on 100 paired
inputs, freezes global/per-input choices, then independently evaluates five
policy roles with three repetitions: fixed weight one, best global fixed,
best per-input fixed, periodic relaxation, and frozen LSTDQ. Identical execution
cells are shared. Setup and RL learning stay disabled.

Continuation costs exclude the common primary setup and include recovery;
total costs also include the common setup. Inclusive costs add controller
work. The complete protocol, measured inputs, selections, timing records, and
checkpoint hashes are saved with the results.

## Module 3: variable-flow evaluation

Use the same online runner with the suites under `configs/module3/suites/`:

```sh
python -m experiments.paper_final.run_online prepare --suite configs/module3/suites/variable_flow_60_batch1.json --output results/module3/variable_flow_60_batch1
python -m experiments.paper_final.run_online run --output results/module3/variable_flow_60_batch1
```

Run the second batch for seeds 4–6. Each seed starts four fresh branches and has
5,000 learning inputs. Module 3 runs independently with its own variable-flow
problem stream and the shared setup encoding. Native and overhead-included
timing are both recorded. Failed
cases remain in the measurements, and common-success comparisons are reported.

## Paper figures

The runners produce the manuscript figures in each completed run's
`paper_figures/` directory. To regenerate them from saved results:

```sh
python -m experiments.paper_final.plot_online results/module1/diffusion_60_batch1
python -m experiments.paper_final.plot_policy results/module2/run
```

The online renderer produces the runtime breakdown, cumulative-time plot, and
relaxation-weight heatmap. Supply recorded individual-cycle examples with
`--cycle-examples PATH.json` to also produce the schedule/residual figure.
The matched-policy renderer produces the savings plot and selected weight panels.
These commands read saved evidence and perform no solver or learner runs.
