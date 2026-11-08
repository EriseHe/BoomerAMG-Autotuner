"""Recompute saved timings on identical per-seed, period-two-success subsets."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "run"
OUT = ROOT / "readout" / "common_periodic_success"
POLICIES = ("fixed_1", "periodic", "rl_frozen")
FIELDS = ("native_continuation_sec", "inclusive_continuation_sec")


def main():
    marker = json.loads((RUN / "raw/test/complete.json").read_text())
    for relative, expected in marker["raw_sha256"].items():
        assert hashlib.sha256((RUN / relative).read_bytes()).hexdigest() == expected
    rows_by_case = defaultdict(list)
    observed = set()
    for path in sorted((RUN / "raw/test").glob("worker_*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            key = (row["seed"], row["case_id"], row["policy"], row["repeat"])
            assert key not in observed
            observed.add(key)
            rows_by_case[key[:3]].append(row)
    expected = {(s, c, p, r) for s in range(1, 7) for c in range(100)
                for p in POLICIES for r in range(3)}
    assert observed == expected
    cases = {}
    for key, rows in rows_by_case.items():
        assert {r["repeat"] for r in rows} == {0, 1, 2}
        primary = {r["outcome"]["primary_status"] == "success" for r in rows}
        final = {r["success"] for r in rows}
        assert len(primary) == len(final) == 1, ("Repetition-dependent success", key)
        for r in rows:
            assert (r["outcome"]["primary_status"] == "success") == (
                not r["outcome"]["fallback_used"])
            assert math.isclose(r[FIELDS[0]], r["outcome"]["solve_runtime"] +
                                r["outcome"].get("fallback_setup_runtime", 0), abs_tol=1e-9)
            assert math.isclose(r[FIELDS[1]], r[FIELDS[0]] +
                                r["outcome"]["infer_runtime"], abs_tol=1e-9)
        cases[key] = {
            "primary_success": all(primary), "final_success": all(final),
            **{field: statistics.mean(r[field] for r in rows) for field in FIELDS},
        }
    for seed in range(1, 7):
        for case in range(100):
            paired = [r for p in POLICIES for r in rows_by_case[seed, case, p]]
            assert len({(r["input_id"], r["hierarchy_id"]) for r in paired}) == 1

    output = []
    masks = []
    for seed in range(1, 7):
        requested = [c for c in range(100) if cases[seed, c, "periodic"]["primary_success"]]
        completed = [c for c in requested if all(cases[seed, c, p]["final_success"] for p in POLICIES)]
        masks.append({
            "seed": seed, "requested_retained_case_ids_zero_based": requested,
            "excluded_periodic_primary_failure_case_ids_zero_based": sorted(set(range(100)) - set(requested)),
            "all_completed_retained_case_ids_zero_based": completed,
            "additional_final_failure_exclusions_zero_based": sorted(set(requested) - set(completed)),
        })
        for subset, keep in (("periodic_first_attempt_success", requested),
                             ("periodic_first_attempt_success_and_all_methods_complete", completed)):
            result = {"subset": subset, "seed": seed, "retained_cases": len(keep),
                      "excluded_periodic_primary_failures": 100 - len(requested),
                      "additional_final_failure_exclusions": len(requested) - len(keep)}
            totals = {}
            for policy in POLICIES:
                result[policy + "_primary_failures"] = sum(not cases[seed, c, policy]["primary_success"] for c in keep)
                result[policy + "_final_failures"] = sum(not cases[seed, c, policy]["final_success"] for c in keep)
                for field in FIELDS:
                    total = math.fsum(cases[seed, c, policy][field] for c in keep)
                    result[policy + "_" + field] = totals[policy, field] = total
            for policy in ("periodic", "rl_frozen"):
                for field, label in zip(FIELDS, ("solver", "including_controller")):
                    reduction = 100 * (1 - totals[policy, field] / totals["fixed_1", field])
                    # Independent direct sum over raw repetitions: each case has exactly 3.
                    numerator = math.fsum(r[field] for c in keep for r in rows_by_case[seed, c, policy])
                    denominator = math.fsum(r[field] for c in keep for r in rows_by_case[seed, c, "fixed_1"])
                    assert math.isclose(reduction, 100 * (1 - numerator / denominator), abs_tol=1e-10)
                    result[policy + "_" + label + "_reduction_pct"] = reduction
            output.append(result)

    OUT.mkdir(exist_ok=True)
    with (OUT / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    report = {
        "filter": "Per seed, exclude every problem for which period-two failed its first attempt, even if fallback succeeded. Use identical retained case IDs for every policy and reference.",
        "denominator": "Fixed w=1 on the same bandit-selected hierarchy and retained inputs.",
        "formula": "100 * (1 - sum(mean_over_three_repetitions(policy_time)) / sum(mean_over_three_repetitions(reference_time)))",
        "timing_scope": "Continuation time excludes the common primary hierarchy construction; includes any fallback setup and solve. The including_controller metric also includes controller overhead.",
        "additional_comparison": "The second subset also excludes cases with final failure in any compared method, without excluding successfully recovered cases.",
        "qualification": "Conditional results on period-two's first-attempt successes; excluded failures are omitted from all denominators and numerators. Not an overall performance measure. A requested-subset ratio whose reference has final failures compares recorded costs with different completion outcomes.",
        "raw_files_modified": False,
        "raw_sha256": marker["raw_sha256"],
        "validation": {"raw_checksums": True, "exact_5400_trial_coverage": True,
                       "paired_inputs_and_hierarchies": True, "identical_case_sets": True,
                       "success_stable_across_repetitions": True,
                       "independent_direct_ratio_check": True},
        "case_masks": masks, "results": output,
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    for result in output:
        print(json.dumps({k: v for k, v in result.items() if k in (
            "subset", "seed", "retained_cases", "fixed_1_final_failures",
            "periodic_including_controller_reduction_pct", "rl_frozen_including_controller_reduction_pct")}))


if __name__ == "__main__":
    main()
