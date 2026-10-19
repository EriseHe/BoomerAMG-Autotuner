# Study data

`module1/`, `module2/`, and `module3/` contain the published configurations,
measurements, and arrays. `manifest.json` checksums these files.

Module 1 has 5,000 online-learning inputs followed by 1,000 frozen-evaluation
inputs per seed. Its original training measurements remain in `module1/`;
the frozen-stage configurations, inputs, measurements, and audits are in
`module1/frozen/`. The two phases retain separate summaries and timing arrays.

Full local trajectories and checkpoints use the same numbering under
`official/`; `raw_files.json` records their checksums. They are excluded from Git.
