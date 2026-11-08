# Continuing to 2,000 systems

The user extended the target from 1,000 to 2,000 systems per application during
the first SPE10 campaign. `extend_batch.py` waits for the current batch's SPE10
and MFEM runs and audits to finish, then runs systems 1,001–2,000 sequentially
for each application. The cycle limit stays at 100. This retains one learner
seed; a continuation is not an independent statistical replicate.

`continuation.py` reuses `Study` and the physical drivers from the original
runner. It restores both setup models' precision matrices, target vectors,
failure models, observation counts, RNGs and candidate statistics. It restores
both LSTDQ controllers' weights, recursive statistics, uncertainty, counters,
exploration state and RNGs. State and counters are checked against checkpoints
before any additional solve. The preceding setup-update timing is retained for
the next cost label. The extended candidate schedule preserves all original
candidate rows and uses subsequent rows for the additional systems.

The added SPE10 trajectories use oil/water viscosity ratios 1.5, 3, 4, 7.5 and
15, each with constant and pulsed injection. The additional heat regimes
`(kappa, alpha, dt)` are `(0.25, 0.05, 0.1)`, `(0.5, 0.2, 0.1)`,
`(0.2, 0.05, 0.1)`, `(0.5, 0.1, 0.075)` and `(0.5, 0.1, 0.15)`, each with
initial temperature amplitude 1 or 4. These ten new trajectories per
application keep the mesh and context dimensions compatible with the existing
checkpoints. Their seeded order is fixed before continuation.

The original 1,000 measurements and checkpoints are retained. No prefix solve
is repeated or replaced. Native baseline hierarchy objects are rebuilt after
the process restart, and that work is included in measured times. Learner
restoration/preparation is included in continuation initialization costs.

`merge_continuation.py` creates a combined 2,000-system dataset, preserving the
original timing values and adding the new measurements and initialization
costs. It retains the exact source snapshots from both segments. The ordinary
audit then checks all 12,000 method/system measurements per application and
reconstructs the final setup statistics from the complete observation history.

```sh
python experiments/application_online/extend_batch.py \
  --prefix results/application_campaign_1000_cap100_20261104 \
  --output results/application_campaign_2000_cap100_20261104
```

`status.json` in the new output directory reports waiting/running/audit states.
Continuation progress lives under `<application>/progress.json`; complete
2,000-system results and audits live under `<application>_combined/`.
