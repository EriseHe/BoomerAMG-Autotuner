# RLTune-BoomerAMG

Online Autotuning of BoomerAMG with Contextual Bandits and Reinforcement Learning,
by Xudong He, Ruiyu Wang, Yilang Ding, Rui Peng Li, and Yuanzhe Xi.

The setup learner selects an AMG configuration. The solve controller adapts
relaxation using the **shared context**: problem information, shared setup
features, and normalized cycle observations. The encoder is named
`shared_context`.

| Study | Inputs and learning | Configuration |
|---|---|---|
| Module 1, Part 1: online learning | Inputs 1–5,000; LinUCB and LSTDQ learn from input one | [Module 1 suites](configs/module1/suites) |
| Module 1, Part 2: frozen evaluation | Inputs 5,001–6,000; retain the Part 1 models, disable updates and RL exploration | Same Module 1 suites |
| Module 2: matched policy evaluation | Use selected trained Module 1 models to compare frozen LSTDQ with fixed and periodic relaxation on matched hierarchies | [Module 2 instructions](docs/experiments.md#module-2-matched-policy-evaluation) |
| Module 3: variable flow | Six independent 5,000-input learning streams at 60³ | [Module 3 suites](configs/module3/suites) |

Module 1 runs both parts in one continuous worker process per seed. Each
diffusion/advection grid has six seeds, arranged in two batches of three.
Both Module 1 and Module 3 measure four branches: Default, fixed-setup LSTDQ,
setup-only LinUCB, and joint LinUCB–LSTDQ. Module 3 starts its own learners and
can run independently of Modules 1 and 2.

## Install

Use Python 3.10 or later, MPI, CMake, Make, and a C/C++ compiler:

```sh
python -m pip install -e .
make -C hypre JOBS=4
```

## Run

[Experiment instructions](docs/experiments.md) give the commands for each
module. Active configurations are in `configs/`. After the selected Module 1
run finishes, initialize Module 2 with explicit paths to its trained seed
directories using `run_policy init --training-run SEED=PATH`. No experiment is
selected automatically. Checkpoint loading checks the complete feature
description before loading learned parameters.

LSTDQ learns from **native cycle execution time only** (`cost_target=native_cycle`).
The setup bandit's cost feedback includes measured learner/controller overhead.

Each run writes its own trajectories, checkpoints, source hashes, summaries,
and figures under `results/`. Native solver time and total time including
learner/controller overhead are reported separately.
Overhead remains measured during Module 1 Part 2 and Module 2, when learning
updates are disabled. Saved experiment artifacts in `data/` retain their
recorded provenance; new runs write to `results/`.

HYPRE retains its [copyright](hypre/source/COPYRIGHT),
[license](hypre/source/LICENSE-MIT), and [notices](hypre/source/NOTICE).
A license for the project-owned code has not yet been selected.
