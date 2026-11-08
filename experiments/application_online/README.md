# Application online campaigns

The current requested experiment runs **1,000 systems per application, with a
100-cycle/iteration ceiling per attempt**. Ten distinct physical simulations
contribute 100 chronological systems each. Learners start fresh and remain
active across all scenario boundaries. This is one paired learner seed, not
the paper's frozen matched-hierarchy evaluation or a six-seed final study.

The four paper branches are Default, fixed-setup LSTDQ, LinUCB with fixed
relaxation, and joint LinUCB–LSTDQ. Two additional references use PCG–AMG with
fresh hierarchies or hierarchy reuse. Every method receives the same matrix,
physical RHS and initial guess. A separate accurate reference solve advances
each physical trajectory. Warm starts use the preceding reference solution;
physical state and the initial guess reset when the next simulation begins.
Learner states and reusable hierarchies persist where applicable.

## Physical campaigns

SPE10 uses MRST layers 1–15, 198,000 pressure cells, TPFA with well elimination,
initially oil-filled cells, quadratic relative permeabilities, gravity and
capillarity off, and a corner injector/opposite-corner 100-bar producer.
Water viscosity is 1 cP. Oil viscosity is 1, 2, 5, 10 or 20 cP. Each ratio is
paired with constant injection or four 25-step rate intervals with multipliers
0.5, 1.5, 0.5, 1.5. Every full scenario injects one pore volume over ten years.

MFEM uses the existing irregular tetrahedral mesh, quadratic H1 elements and
220,881 unknowns. Lagged backward Euler solves
`(M + dt*K(u_old))*u_new = M*u_old`, with conductivity `kappa + alpha*u_old`.
The five `(kappa, alpha, dt)` regimes are `(0.5, 0.01, 0.1)`, `(0.5, 0.1, 0.1)`,
`(0.1, 0.1, 0.1)`, `(0.5, 0.1, 0.05)` and `(0.5, 0.1, 0.2)`. Each is paired
with initial hot-region amplitude 1 or 4 above background temperature 1.
The hot region remains the radius-0.5 sphere at the origin; boundaries are
insulating. Every scenario has 100 steps, so physical horizons depend on dt.

These are declared benchmark extensions to the public application examples.
`campaigns.py` fixes the ten scenarios and their seeded permutation before
timing. `campaign.json` records the exact sequence. The larger campaign does
not repeat the same recorded matrix trajectory ten times. The isolated MRST,
MFEM and Octave dependencies are described in `../application_pilot/README.md`.

## Solver protocol

The learning branches retain the current paper's setup catalog, 1,024-candidate
schedule, action weights, controller parameters and retry/recovery executors.
Each branch owns independent learners. The application contexts and matrix
interface are new; the learning algorithms are unchanged.

The fixed PCG reference uses HMIS/ext+i AMG, symmetric hybrid Gauss–Seidel
(type 6) on both legs, and a direct coarse solve. It is chosen in advance,
**not calibrated on separate development trajectories**. The reuse reference
rebuilds after ten uses or when the preceding iteration count exceeds twice
that immediately after its last build. Its hierarchy stays fixed during each
PCG solve. Outer PCG always uses the current operator, while the older hierarchy
acts only as a preconditioner.

All measured methods require the externally checked residual
`norm(b-A*x)/norm(b) <= 1e-6`. A zero RHS uses denominator one. An already
converged warm start returns without a hierarchy build. Every AMG attempt,
PCG solve and rescue attempt has a maximum of 100 cycles/iterations. Existing
retries remain separate attempts and all their costs are charged. After
unresolved failure, a common fresh PCG–AMG rescue targets 1e-7 with the same
100-iteration ceiling; its observations do not alter learning targets. If it
also fails, the workload stops and remains incomplete. The accuracy reference
also has a 100-iteration ceiling, targets 1e-10 internally and must satisfy an
external 1e-9 check. No hidden 500- or 1,000-iteration rescue is used.

## Context and timing

SPE10 context contains the bias, log viscosity ratio divided by log(20),
saturation mean/population standard deviation, four saturation half-domain
means, injected pore volumes and current injection-rate multiplier. Heat
context contains the bias, logarithms of dt/kappa/alpha, and nodal temperature
mean/population standard deviation/minimum/maximum. All are known before the
solve, with fixed scaling and no fitting to future systems.

Methods run serially in seeded orders, one MPI rank and one thread. Method
wall time includes matrix import, selection, setup, iterations, residual
checks, learning, retries, rescue and relevant context extraction. Assembly,
transport, common IPC, reference generation and diagnostic file/checkpoint
writing are excluded. Initialization is separate; cold-start totals add the
native initialization and each method's learner preparation. Installation,
compilation, Python launch and package imports are not part of those totals.
These are solver deployment costs, not full simulation wall times.

`records.jsonl` records all attempts, input hashes, scenario IDs, cycle traces
and recovery costs. `reference.jsonl` and `application.csv` hold accuracy and
physical diagnostics. Checkpoints are written every 100 systems. `progress.json`
is replaced atomically after each completed system. Each protocol records the
base Git commit, dependency versions, source/binary hashes, scenario definitions
and its producing source snapshot. The audit checks pairing, residuals, mass
conservation within each heat trajectory, iteration caps and setup-model
statistics against a direct batch calculation.

## Running

Use the existing `rl` Python environment and isolated dependencies. Output
directories must be new. Both applications are normally run sequentially.
`python experiments/application_online/batch.py --output <new-directory>` runs
both full campaigns and their audits, recording live state in `status.json`.
It stops on a failed run or audit and writes the combined summary only after
both applications finish successfully.

```sh
env OMPI_CC=/usr/bin/clang make -C hypre/interfaces MPICC=/path/to/mpicc
clang++ -O3 -std=c++11 -Iresults/application_default_100/mfem \
  experiments/application_online/heat_stream.cpp \
  results/application_default_100/mfem/libmfem.a \
  -o results/application_default_100/heat_stream
python -m unittest experiments.application_online.test_runtime
python experiments/application_online/run.py --application spe10 --cases 1000 \
  --max-cycles 100 --output results/application_online_1000_cap100_spe10
python experiments/application_online/run.py --application heat --cases 1000 \
  --max-cycles 100 --output results/application_online_1000_cap100_heat
python experiments/application_online/audit.py \
  results/application_online_1000_cap100_spe10 \
  results/application_online_1000_cap100_heat
```

For integration checks only, `--cases 4 --steps-per-scenario 2` verifies a
physical reset and persistent learning across two short trajectories. Those
checks are stored separately and never included in campaign measurements.
Completed older pilots remain historical results with their original caps.
The interrupted 500-cycle MFEM pilot and its log were removed at the user's
request; they are not resumed or included in the current campaign.
