# RLTune-BoomerAMG

Online Autotuning of BoomerAMG with Contextual Bandits and Reinforcement Learning,
by Xudong He, Ruiyu Wang, Yilang Ding, Rui Peng Li, and Yuanzhe Xi.

The setup learner selects an AMG configuration. The solve controller adapts
relaxation using the **shared context**: problem information, shared setup
features, and normalized cycle observations. The encoder is named
`shared_context`.

| Module | Study | Configuration |
|---|---|---|
| 1 | Six seeds per diffusion/advection grid; 5,000 learning inputs and 1,000 frozen inputs in one continuous stream | [Module 1](configs/module1/suites) |
| 2 | Compare frozen LSTDQ with fixed and periodic relaxation on matched hierarchies | [Module 2](docs/experiments.md#module-2-matched-policy-evaluation) |
| 3 | Six 5,000-input variable-flow streams at 60³ | [Module 3](configs/module3/suites) |

## Install

Use Python 3.10 or later, MPI, CMake, Make, and a C/C++ compiler:

```sh
python -m pip install -e .
make -C hypre JOBS=4
```

## Run

[Experiment instructions](docs/experiments.md) describe the commands, timing,
and evaluation protocol. Active configurations are in `configs/`. After Module 1
finishes, Module 2 takes explicit paths to its trained seed directories. It is
not connected to any particular run. Checkpoint loading checks the complete
feature description before loading learned parameters.

Each run writes its own trajectories, checkpoints, source hashes, summaries,
and figures under `results/`. Native solver time and total time including
learner/controller overhead are reported separately.

HYPRE retains its [copyright](hypre/source/COPYRIGHT),
[license](hypre/source/LICENSE-MIT), and [notices](hypre/source/NOTICE).
A license for the project-owned code has not yet been selected.
