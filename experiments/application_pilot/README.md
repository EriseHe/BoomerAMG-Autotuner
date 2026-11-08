# Default-only application timing pilot

This is a bounded integration and timing experiment. It runs no learners and
does not replace the paper's generated-family runners.

Both applications use the current paper's explicit default setup parameters
from `setup/space.py`, l1-Jacobi on the fine/intermediate levels and direct
coarse relaxation, one MPI rank and one thread. This is the paper's reference
configuration, not an assertion about the unmodified HYPRE library defaults.
Every measured system builds a fresh hierarchy. Inputs use physical right-hand
sides and a preceding reference solution as the initial guess.

The timed solves require an independently checked relative residual
`norm(b-A*x)/norm(b) <= 1e-6`. The final completion pilot has a safety ceiling
of 500 V-cycles; it stops as soon as the residual target is reached. This
ceiling is a pilot choice, not an application standard. The initial heat
diagnostic separately preserves the paper's 50-cycle limit.

## Workloads

* **SPE10:** MRST's `getSPE10setup(1:15)`, 198,000 cells, TPFA pressure with well
  equations eliminated and reconstructed through `incompTPFA(...,'reduce',true)`.
  Initially oil-filled, porosity floored at 0.01, no gravity or capillarity,
  quadratic relative permeabilities, water/oil viscosity 1/5 cP. A corner rate
  injector and opposite-corner 100 bar producer inject one pore volume over ten
  years. The pilot takes 100 pressure solves at t=0,...,9.9 years, with MRST's
  implicit upwind transport advancing each 0.1-year interval.
* **Heat:** MFEM 4.8 assembly of the nonlinear heat equation in Example 16,
  `du/dt = div((0.5 + 0.01*u)*grad(u))`, with natural boundary conditions.
  Backward Euler with lagged conductivity uses `(M + 0.1*K(u_old))*u_new=M*u_old`
  for 100 steps. This is the equivalent state form of the example's stage
  equation, with a custom irregular tetrahedral mesh, not an unchanged stock
  example invocation. The domain is `[0,8] x [0,1] x [0,1]`. Initial nodal
  temperature is 2 inside radius 0.5 about the origin, otherwise 1.
  `make_heat_mesh.py` deterministically jitters a coarse point set tangentially
  to the box boundaries and generates a Delaunay tetrahedralization. Two MFEM
  refinements and quadratic H1 elements give 220,881 unknowns, 157,056
  tetrahedra, and 6,153,729 matrix entries.

Separate AMG-preconditioned CG reference solves advance each trajectory. They
are **excluded** from default timings and are not a matched-tolerance baseline.
The reference internal tolerance is 1e-10. Heat also verifies 1e-10 externally;
SPE10 verifies 1e-9 externally to allow recursive/recomputed residual differences
with absolute pressures in pascals (still 1000 times tighter than the measured
target). A preliminary three-step SPE10 run stopped on a residual of 1.08e-10
and is preserved as `spe10_reference_preflight`; it is not part of the final
100-case results.

## Timing and validation

`native_default.c` imports arbitrary CSR matrices into the repository's
installed HYPRE 3.0 library. Native setup and solve are timed separately.
CSR transfer, external residual verification, reference solves, finite-element
assembly, reservoir transport, dependency installation, and compilation are
outside the native setup-plus-solve figure and reported separately where
available. No application timing cases are discarded as warmups.

`bridge.py` checks this adapter on a known-solution 3D SPD matrix using standalone
AMG and preconditioned CG. Application residuals are recomputed externally.
Heat additionally records the mass integral; reservoir transport records
saturation bounds and water volume. All raw case records are retained.

## Local execution

Dependencies and build outputs are isolated under
`results/application_default_100/`. MRST and MFEM sources are pinned by hashes
in the generated provenance file. Octave is installed in `octave_env`; the
launcher explicitly sets `OCTAVE_HOME` and `OCTAVE_EXEC_HOME` to work around the
Conda package's embedded-prefix startup issue on this machine.

From the repository root, with the existing `rl` Python environment:

```sh
# With mpicc available and MFEM already built as a serial library:
OMPI_CC=/usr/bin/clang sh experiments/application_pilot/build.sh
python experiments/application_pilot/bridge.py
python experiments/application_pilot/make_heat_mesh.py
python experiments/application_pilot/run_spe10.py --cases 100
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 results/application_default_100/heat_pilot_500 \
  results/application_default_100/heat.mesh \
  results/application_default_100/heat_records_500.csv 100 500
python experiments/application_pilot/summarize.py
```

The SPE10 launcher refuses to overwrite existing measurements. Archive the
previous result directory before deliberately starting another run. The heat
CSV output path must likewise be new for a new run.

Dependency setup used a shallow clone of `SINTEF-AppliedCompSci/MRST`, a clone
of `mfem/mfem` at tag `v4.8`, and an isolated Conda environment containing Octave.
MFEM was built with `make config MFEM_USE_MPI=NO MFEM_USE_METIS=NO
MFEM_USE_LAPACK=NO CXX=/usr/bin/clang++`, then `make -j4`. The official SPE10
archive came from the URL in MRST's `dataset_spe10.m` and was extracted into
`results/application_default_100/datasets/spe10`. The summary records the exact
Git revisions and data/library hashes used for the measurements.

These are one trajectory per application, not six independent campaign seeds,
a robustness study, a timestep-accuracy study, or measurements of the learned
branches. They estimate the cost of these specified default-solver workloads.
