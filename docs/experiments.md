# Experiments

Run commands from the repository root after installing the package and building
HYPRE. Use a new output directory for each run. Active configurations are in
`configs/`; the state encoding is `shared_context`.

## Module 1: online learning and frozen evaluation

Each seed runs one continuous stream in a single worker process:

- Inputs 1–5,000: both LinUCB and LSTDQ learn from input one.
- Inputs 5,001–6,000: both learners are frozen and LSTDQ epsilon exploration is disabled.

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
LSTDQ learns from native cycle execution time only. Controller overhead remains
included in total-runtime evaluation and the setup bandit's cost feedback.
Summaries report separate `learning` and `frozen` windows and the full stream.

The configurations use confidence coefficient 2, 1,024 setup candidates,
relaxation weights 1–3, tolerance 10⁻⁶, and a 50-cycle cap. The solve controller
uses the same setup descriptors as the bandit, tolerance-based residual
normalization, and cycle time relative to the first native cycle of the attempt.

## Module 2: matched policy evaluation

After Module 1 is complete, create a configuration using its trained joint
models. Specify one seed directory per source; no run is selected automatically.
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

Run the second batch for seeds 4–6. Each seed has 5,000 learning inputs. Failed
cases remain in the measurements, and common-success comparisons are reported.
