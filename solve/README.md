# Solve phase

The SISC studies use recursive LSTDQ to select relaxation weights after each
AMG cycle. `registry.py` accepts `default`, `fixed`, and `recursive_lstdq`.

- `controllers/recursive_lstdq/`: the single episode-cluster controller,
  numerical configuration, and factory.
- `controllers/common/`: state and action encoders, construction inputs,
  controller/checkpoint pairing, and linear numerical primitives.
- `core/episode.py`: the cycle loop, completion-cost targets, recovery accounting,
  and transactional commit or rollback of controller learning.
- `core/outcomes.py`: failure classification shared with the native adapter.

## LSTDQ implementation

[RecursiveLstdqController](controllers/recursive_lstdq/controller.py) contains
the recursive mean update, inverse maintenance, episode-cluster covariance,
and checkpoint operations in one implementation.
`controllers/recursive_lstdq/common.py` factorizes the shared state-action scoring.

`controllers/recursive_lstdq/config.py` defines the numerical specs; its factory
constructs the controller with the encoder and action configuration supplied by
`solve.registry`. `controllers/common/` contains the encoders, controller bundle,
TD configuration, and linear primitives. The TD configuration retains its
historical class name because checkpoint configuration fields remain unchanged.

The controller commits one cluster covariance update per successfully committed episode.
The episode runner owns recovery and rollback; the controller owns its numerical
state and NPZ checkpoint. The saved checkpoint fields and schema remain unchanged.

The state contains the intercept, normalized cycle count `k / Kmax`, remaining
residual fraction `g`, `g^2`, signed residual change `d`, preceding cycle time,
normalized preceding relaxation weight, problem context, and shared setup
descriptors. Cycle count enters through `k / Kmax` alone. For diffusion, the
three problem features and 19 setup descriptors give 29 state features and
261 state-action features with the nine action basis functions. Dimensions
are derived from the supplied encoders. The obsolete cycle-tabular mode is removed.

The state encoding is `shared_setup_one_hot_tolerance_residuals_v2`. Older
state layouts require fresh training and are rejected by encoder metadata.

## Shared components and checks

A `ControllerBundle` pairs a controller with its exact state encoder. It runs a
case, reports a summary, and preserves checkpoint metadata. Setup selection,
stream order, fallback choice, and reporting belong to the experiment layer.

Setup and solve use the same native interfaces in `hypre/interfaces/` and the
same [PDE streams](../docs/repository_layout.md#pde-problems). Build the interfaces
with `make -C hypre`; see [reproduction](../docs/reproduction.md) for prerequisites.
Algorithm implementation belongs on `main`, alongside the official Module 1–4
runners. Development tests and diagnostic experiments are maintained on
`codex/development`; the official runners do not execute those tests.
