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
| Module 1, Part 2: frozen evaluation | Inputs 5,001–6,000; retain the Part 1 models, disable updates and RL exploration | [Module 1 suites](configs/module1/suites) |
| Module 2: matched policy evaluation | Use selected trained Module 1 models to compare frozen LSTDQ with fixed and periodic relaxation on matched hierarchies | [Module 2 instructions](experiments/README.md#module-2-matched-policy-evaluation) |
| Module 3: variable flow | Six independent 5,000-input learning streams at 60³ | [Module 3 suites](configs/module3/suites) |

Module 1 runs both parts in one continuous worker process per seed. Each
diffusion/advection grid has six seeds, arranged in two batches of three.
Both Module 1 and Module 3 measure four branches: Default, fixed-setup LSTDQ,
setup-only LinUCB, and joint LinUCB–LSTDQ. Module 3 starts its own learners and
can run independently of Modules 1 and 2.

## Code organization

- `setup/`: setup learners, candidate features, failure feedback, and cost targets.
- `solve/`: cycle-level solve controllers and episode execution.
- `joint/`: retry loops, fallback coordination, and learner state across attempts.
- `experiments/`: study configuration, candidate schedules, solver adapters, traces, audits, and reporting.

## Install

Use Python 3.10 or later, MPI, CMake, Make, and a C/C++ compiler:

```sh
python -m pip install -e .
make -C hypre JOBS=4
```

## Run

Run the complete study in order with a new output directory:

```sh
python -m experiments.run_all --output results/full_study
```

This runs all Module 1 suites, passes their six Diffusion 60 joint models to
Module 2, then runs both Module 3 batches. Each online batch uses three
concurrent seed workers. Use `--dry-run` to inspect the plan or
`--module2-source advection_60` to select another Module 1 family/grid.
[Experiment instructions](experiments/README.md) also give individual module
commands. Checkpoint loading checks the complete feature description before
loading learned parameters.

Each run writes its own trajectories, checkpoints, source hashes, summaries,
and figures under `results/`. Native solver time and total time including
learner/controller overhead are reported separately.
Overhead remains measured during Module 1 Part 2 and Module 2, when learning
updates are disabled. Saved experiment artifacts in `data/` retain their
recorded provenance; new runs write to `results/`.

HYPRE retains its [copyright](hypre/source/COPYRIGHT),
[license](hypre/source/LICENSE-MIT), and [notices](hypre/source/NOTICE).
A license for the project-owned code has not yet been selected.
