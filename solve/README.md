# Solve phase

The solve controller uses recursive LSTDQ to select relaxation weights after each
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

Scoring caches the derived product `M = A_inverse @ Omega`, where `Omega` is
the episode-moment covariance. Each inverse rank-one correction applies the
same correction to `M`; an episode addition `g @ g.T` adds
`(A_inverse @ g) @ g.T` to the cache. This replaces the repeated dense
projection-covariance product without changing the LSTDQ estimator, confidence
score, action basis, or shared setup features. The cache is reconstructed after
checkpoint loading, rollback, and an inverse rebuild; it is not saved as learned
state. Rollback copies preserve matrix memory layout, and scoring does not build
an unused state-lift matrix. Dense-formula tests check scores and greedy actions.

`controllers/recursive_lstdq/config.py` defines the numerical specs; its factory
constructs the controller with the encoder and action configuration supplied by
`solve.registry`. `controllers/common/` contains the encoders, controller bundle,
TD configuration, and linear primitives. The TD configuration retains its
historical class name because checkpoint configuration fields remain unchanged.

The controller commits one cluster covariance update per successfully committed episode.
The episode runner owns recovery and rollback; the controller owns its numerical
state and NPZ checkpoint. Checkpoint version 3 includes the retained update-time
estimator and requires matching encoder metadata; older controllers need fresh training.

Setup descriptors use `GenericActionFeatureEncoder.encode_setup`, the same scaled
numerical values and reference-coded one-hot categories used by the setup bandit.
The bandit adds its polynomial and interaction terms. The shared descriptors are
not clipped; inactive setup parameters use the same masking in both learners.

The state contains the intercept, normalized cycle count `k / Kmax`, remaining
residual fraction `g`, `g^2`, signed residual change `d`, preceding cycle time,
normalized preceding relaxation weight, problem context, and shared setup
descriptors. Cycle count enters through `k / Kmax` alone. For diffusion, the
three problem features and 19 setup descriptors give 29 state features and
261 state-action features with the nine action basis functions. Dimensions
are derived from the supplied encoders. The obsolete cycle-tabular mode is removed.

The RL transition cost includes native cycle time, the measured encoding and
selection time for that cycle, and the most recent measured runtime of a retained
transition update (zero before the first update). The current update is measured afterward
and enters the next estimate. This avoids making an update depend on its own
unavailable runtime. A failed-input rollback also restores this estimator.
Snapshots and episode finalization remain evaluation overhead. Reported total
runtime uses actual measured overhead once, never these estimated charges.
Per-cycle traces retain both estimated and actual update times for auditing.

The preceding native cycle time is divided by the first native cycle time of
the current attempt, without clipping. Before the first cycle, this feature is
zero; after that cycle, it is one. The denominator stays fixed for the attempt
and resets on every new attempt, including retries. This state feature uses
native times only; the overhead-inclusive learning cost remains in seconds.
The residual features share the required logarithmic reduction
`G = log(r0 / tolerance)`: the remaining fraction is
`g_k = log(r_k / tolerance) / G`, and the preceding signed change is
`d_k = log(r_k / r_(k-1)) / G = g_k - g_(k-1)`. Neither feature is clipped.
Thus `g` starts at 1 and reaches 0 at the tolerance; negative `d` denotes
contraction and positive `d` denotes growth. Their logarithms require positive
residual observations and an initial residual above tolerance; terminal states
are not encoded by the episode runner. Logarithms are subtracted rather than
forming potentially overflowing residual ratios.

Wording reserved for the later Section 4 revision: **the signed fraction of the
required logarithmic reduction achieved in the preceding cycle**. Clarify that
the sign convention assigns negative values to residual reduction. The paper
has not been updated for these implementation changes yet.

Fresh experiment configurations use `state_encoding: "shared_setup_one_hot_tolerance_residuals_v3"`
and `cost_target: "native_plus_controller"`. Archived configurations, models, and
published results retain their original encoding and native-cycle objective.
The updated feature semantics require fresh training; checkpoints using the
fixed time scale or the earlier 43-feature state are rejected by the encoder
metadata check, even when their feature count matches.

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
