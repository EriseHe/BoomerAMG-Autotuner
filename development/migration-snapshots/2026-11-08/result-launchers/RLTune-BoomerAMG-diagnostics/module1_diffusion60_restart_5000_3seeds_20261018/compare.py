"""Compare this Module 1 rerun with the same official seeds and input window."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LABELS = {
    "default_setup_default_solve": "Default",
    "default_setup_lstdq": "LSTDQ",
    "bandit_default": "LinUCB",
    "bandit_lstdq": "LinUCB–LSTDQ",
}
FIELDS = {
    "native_sec": "runtime",
    "total_sec": "end_to_end_runtime",
    "rl_overhead_sec": "infer_runtime",
    "setup_overhead_sec": "bandit_overhead_runtime",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_metrics(seed, method, stop):
    path = ROOT / "run" / f"seed_{seed}" / "trajectories" / f"{method}.jsonl"
    values = {key: [] for key in FIELDS}
    result = dict(cases=0, failures=0, first_failures=0, fallbacks=0, cycles=0)
    with path.open() as handle:
        for index, line in enumerate(itertools.islice(handle, stop), start=1):
            row = json.loads(line)
            assert row["problem_index"] == index, (seed, method, index)
            out = row["outcome"]
            for key, field in FIELDS.items():
                value = out[field]
                assert math.isfinite(value) and value >= 0, (seed, method, index, field)
                values[key].append(value)
            assert math.isclose(
                out["end_to_end_runtime"],
                out["runtime"] + out["infer_runtime"] + out["bandit_overhead_runtime"],
                rel_tol=1e-8, abs_tol=1e-10,
            ), (seed, method, index, "time accounting")
            result["cases"] += 1
            result["failures"] += int(out["failed"])
            result["first_failures"] += int(out["first_primary_status"] != "success")
            result["fallbacks"] += int(out["fallback_used"])
            result["cycles"] += sum(a["cycles"] for a in out["primary_attempts"]) + out["fallback_cycles"]
    assert result["cases"] == stop, (seed, method, result["cases"], stop)
    return {**result, **{key: math.fsum(items) for key, items in values.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=int, required=True)
    args = parser.parse_args()
    stop = args.inputs
    if not 1 <= stop <= 5000:
        parser.error("--inputs must be between 1 and 5000")
    protocol = json.loads((ROOT / "run/protocol.json").read_text())
    seeds = [str(seed) for seed in protocol["seeds"]]
    assert seeds == ["1", "2", "3"]
    baseline_path = REPO / "data/module1/arrays/diffusion_60.npz"
    counts_path = REPO / "data/module1/method_counts.json"
    official_counts = json.loads(counts_path.read_text())["diffusion_60"]
    old_protocol = json.loads((REPO / "data/module1/protocol.json").read_text())
    assert protocol["native_sha256"] == old_protocol["native_sha256"]
    rows = []
    with np.load(baseline_path) as data:
        old_seeds = data["seeds"].tolist()
        methods = data["methods"].tolist()
        timing_fields = data["timing_fields"].tolist()
        assert methods == list(LABELS)
        for seed in seeds:
            current = {method: current_metrics(seed, method, stop) for method in methods}
            old = data["timings_sec"][old_seeds.index(seed), :, :stop, :].sum(axis=1)
            for mi, method in enumerate(methods):
                row = dict(seed=seed, method=method, **current[method])
                for prefix, name in (("native", "native_runtime"), ("total", "end_to_end_runtime")):
                    index = timing_fields.index(name)
                    row[f"official_{prefix}_sec"] = float(old[mi, index])
                    row[f"{prefix}_reduction_pct"] = 100 * (1 - row[f"{prefix}_sec"] / current[methods[0]][f"{prefix}_sec"])
                    row[f"official_{prefix}_reduction_pct"] = float(100 * (1 - old[mi, index] / old[0, index]))
                    row[f"{prefix}_reduction_change_pp"] = row[f"{prefix}_reduction_pct"] - row[f"official_{prefix}_reduction_pct"]
                for key, name in (("rl_overhead_sec", "infer_runtime"), ("setup_overhead_sec", "bandit_overhead_runtime")):
                    row[f"official_{key}"] = float(old[mi, timing_fields.index(name)])
                if stop == 5000:
                    for key in ("failures", "first_failures", "fallbacks", "cycles"):
                        row[f"official_{key}"] = official_counts[seed][method][key]
                rows.append(row)
    aggregates = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        out = dict(method=method, label=LABELS[method], seeds=len(selected), inputs_per_seed=stop)
        for field in selected[0]:
            if field not in ("seed", "method"):
                values = [row[field] for row in selected]
                out[f"{field}_mean"] = float(np.mean(values))
                out[f"{field}_std"] = float(np.std(values, ddof=1))
                if field in ("failures", "first_failures", "fallbacks", "cycles", "official_failures", "official_first_failures", "official_fallbacks", "official_cycles"):
                    out[f"{field}_sum"] = int(sum(values))
        aggregates.append(out)
    checks = {
        "same_native_binaries": True,
        "contiguous_input_indices": True,
        "exact_requested_window_per_method_and_seed": True,
        "finite_nonnegative_times": True,
        "total_equals_native_plus_measured_learner_overheads": True,
    }
    if stop == 5000:
        summary = json.loads((ROOT / "run/summary.json").read_text())
        official_runner_rows = {(row["seed"], row["method"]): row for row in summary["rows"] if row["window"] == "full"}
        for row in rows:
            reported = official_runner_rows[(row["seed"], row["method"])]
            for field in ("cases", "failures", "first_failures", "fallbacks", "cycles", "native_sec", "total_sec"):
                assert math.isclose(row[field], reported[field], rel_tol=1e-10, abs_tol=1e-8), (row["seed"], row["method"], field)
            assert math.isclose(row["total_reduction_pct"], reported["inclusive_reduction_pct"], abs_tol=1e-8)
        checks["independent_calculation_matches_existing_runner"] = True
    result = dict(
        executed_at=datetime.now(timezone.utc).isoformat(),
        source_commit=protocol["source_commit"], seeds=seeds,
        input_window=[1, stop],
        definition="For each seed, total-time reduction is 100*(1-method_total/default_total); reported means and sample standard deviations are across the three seed reductions. Total includes measured learner overhead, every attempted setup/solve, and any default fallback.",
        baseline=dict(path=str(baseline_path.relative_to(REPO)), sha256=digest(baseline_path), counts_sha256=digest(counts_path)),
        checks=checks, rows=rows, aggregates=aggregates,
        caveats=["The rerun evaluates the combined algorithm updates; it does not isolate the effect of any individual change.", "Execution-time variation can also change learned choices, including the unchanged setup-only branch."],
    )
    prefix = ROOT / f"comparison_{stop}"
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"Module 1 diffusion 60³: inputs 1–{stop}, seeds 1–3\n",
        result["definition"] + "\n",
        "| Method | Updated reduction | Official reduction, same seeds/window | Change | Final failures |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(f"| {row['label']} | {row['total_reduction_pct_mean']:.2f}% ± {row['total_reduction_pct_std']:.2f} pp | {row['official_total_reduction_pct_mean']:.2f}% | {row['total_reduction_change_pp_mean']:+.2f} pp | {row['failures_sum']}/{3*stop} |")
    lines += ["", "± denotes the sample standard deviation across seeds, not a confidence interval.", "", *result["caveats"], "", f"Source commit: {protocol['source_commit']}.", "", "Per-seed values and validation checks are in the adjacent JSON and CSV files."]
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"checks":checks, "aggregates":aggregates}, indent=2))


if __name__ == "__main__":
    main()
