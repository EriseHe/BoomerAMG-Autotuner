# Official paper data

These datasets support the newest official manuscript, checked on 2026-11-08.
The current numbering is **Module 2 = variable-flow online learning** and
**Module 3 = matched-hierarchy policy evaluation**.

| Study | Canonical files | Coverage |
|---|---|---|
| Module 1 Part 1 | [module1/online](module1/online) | Six diffusion/advection grid cases × six seeds × four methods; inputs 1–5000 |
| Module 1 Part 2 | [module1/frozen](module1/frozen) | Same continuous streams; frozen inputs 5001–6000 |
| Module 2 | [module2](module2) | Six variable-flow seeds, 5000 inputs and four methods each; original seeds 1–3 plus October 30 rerun seeds 4–6 |
| Module 3 | [module3](module3) | Both diffusion and variable-flow; six trained checkpoints per family, 100 matched cases and five policies |

Module 1 contains 720,000 online and 144,000 frozen method–problem records.
The online setup illustration uses diffusion 60 seed 3. The current Figure 5
examples are diffusion 80 seed 5, inputs 1563–1565, including verified Default
residual replays. Original protocol/source hashes remain intact.

Module 2 includes per-problem timing/status arrays, all seed configurations and
paired PDE inputs, summaries, retry counts and source provenance. The joint
full-stream total-time reduction is **54.1 ± 2.1%**, matching the paper.

**Module 3 reports test repetition 0**, the first prescribed independent test,
using first-attempt native solve cost and retaining failed attempts. Policy
selection scans use three timings. The raw archives retain all three test
repetitions; `source_summary.json` preserves the original three-repeat summary
as provenance. Use `arrays.npz`, `per_case.csv`, `per_seed.csv` and `summary.json`
in each family for the official paper values. Those reproduce all Figure 6
points, the comparison table, and the displayed Figure 7 trajectories.
Do not substitute the three-repeat average or the older continuation/recovery metric.

## Raw records and restoration

The complete raw measurements are published as GitHub release ZIP assets,
following the repository's existing archive layout. [archive_manifest.json](archive_manifest.json)
provides exact download URLs, sizes, SHA256 digests and coverage. Extract the
ZIPs into one directory to restore `data/official/`. The six unchanged Module 1
archives remain in the October 28 release. The new release contains all six
current Module 2 seeds and both current Module 3 families.
The October 28 release used historical Module 2/3 numbering; use the manifest
rather than interpreting those older archive filenames as current module numbers.

[manifest.json](manifest.json) checksums every compact file except itself;
[raw_files.json](raw_files.json) checksums every raw archive member.
[publication/validation.json](publication/validation.json) records the checks.
[publication/paper_source_fingerprints.json](publication/paper_source_fingerprints.json)
identifies the exact manuscript tables/text/figures used for comparison.

These are immutable measurement artifacts with historical execution protocols,
configurations and source versions. They are not active runner inputs, and old
checkpoints are not assumed compatible with later feature encoders. New scripts,
algorithm changes and unpublished work are retained on `codex/development`.
