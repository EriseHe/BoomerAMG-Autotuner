# Study data

`module1/`, `module2/`, and `module3/` contain the published configurations,
measurements, and arrays. `manifest.json` checksums these files.

These files are archival paper artifacts, not active runner inputs. Use
`configs/` and `experiments/README.md` for new experiments. Historical
configurations and checkpoints retain their recorded state encoding and are
not expected to load into the current training workflow.

Module 1 Part 1 is online learning on inputs 1–5,000, stored in `module1/online/`.
Part 2 is frozen evaluation on inputs 5,001–6,000, stored in `module1/frozen/`.
Both parts are measured in one continuous worker stream, including overhead.
The two parts retain separate summaries and timing arrays. The Module 1 protocol
records the selected measurement sources and the execution-source manifest.

Full local trajectories and checkpoints use the same numbering under
`official/`; `raw_files.json` records their checksums. They are excluded from Git.
