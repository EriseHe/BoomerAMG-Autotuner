"""Paper figures from any completed fresh Module 3 study; no numerical work.

python -m experiments.paper_final.plot_policy RUN_DIRECTORY
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from experiments.runtime import configure_single_thread

configure_single_thread()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from experiments.paper_final.plot_online import sha
from experiments.paper_final.matched_policy.configuration import METHODS
from experiments.paper_final.matched_policy.selection import trial_key, verify_coverage

from experiments.paper_final.common.figure_style import matched_figures

FIELDS = ("native_continuation_sec", "inclusive_continuation_sec", "native_total_sec", "inclusive_total_sec")


def read_study(root):
    def read(name):
        return json.loads((root / name).read_text())
    summary = read("analysis/summary.json")
    protocol = read("protocol.json")
    assert protocol["format"] in ("matched_policy", "matched-policy-fresh-v1") and tuple(protocol["methods"]) == METHODS
    assert summary["protocol"] == protocol and summary["audit"]["exact_coverage"]
    jobs = read("jobs_test.json")
    marker = read("raw/test/complete.json")
    sources = {str(root / name): sha(root / name) for name in
               ("analysis/summary.json", "protocol.json", "jobs_test.json", "selections.json", "raw/test/complete.json")}
    rows = []
    for name, expected in marker["raw_sha256"].items():
        assert sha(root / name) == expected, "Evaluation trajectory changed"
        sources[str(root / name)] = expected
        rows.extend(json.loads(line) for line in (root / name).read_text().splitlines())
    verify_coverage(rows, jobs)
    assert len(rows) == marker["trials"] == summary["audit"]["evaluation_trials"]
    by_key = {trial_key(row): row for row in rows}
    seeds = protocol["training_seeds"]
    case_ids = sorted({job["case_id"] for job in jobs})
    n, cap = len(case_ids), protocol["max_cycles"]
    assert n == protocol["test_cases"]
    costs = np.full((len(seeds), 5, n, 4), np.nan)
    success = np.zeros((len(seeds), 5, n), dtype=bool)
    weights = np.full((len(seeds), 5, n, cap), np.nan)
    base_jobs = {(j["seed"], j["case_id"]): j for j in jobs if j["repeat"] == 0}
    assert len(base_jobs) == len(seeds) * n
    for i, seed in enumerate(seeds):
        for c, case_id in enumerate(case_ids):
            job = base_jobs[seed, case_id]
            for j, method in enumerate(METHODS):
                policy = job["method_policies"][method]
                if policy is None:
                    continue
                observations = [by_key[(seed, job["source"], case_id, rep, policy)]
                                for rep in range(protocol["repetitions_per_case"])]
                for row in observations:
                    o = row["outcome"]
                    values = np.array([row[k] for k in FIELDS])
                    assert np.isfinite(values).all() and (values >= 0).all()
                    np.testing.assert_allclose(values[0], o["solve_runtime"] + o.get("fallback_setup_runtime", 0), atol=1e-8)
                    np.testing.assert_allclose(values[1], values[0] + o.get("infer_runtime", 0), atol=1e-8)
                    np.testing.assert_allclose(values[2], o["setup_runtime"] + o["solve_runtime"], atol=1e-8)
                    np.testing.assert_allclose(values[3], o["end_to_end_runtime"], atol=1e-8)
                costs[i, j, c] = np.mean([[r[k] for k in FIELDS] for r in observations], axis=0)
                success[i, j, c] = all(r["success"] for r in observations)
                # The first prescribed repetition is fixed before looking at costs.
                o = observations[0]["outcome"]
                primary = (o.get("primary_attempts") or [o])[0]
                cycles = int(primary.get("cycles", o.get("primary_cycles", o.get("iterations", 0))))
                actions = primary.get("cycle_actions", o.get("cycle_actions", o.get("cycle_action_values", [])))
                if not actions and cycles:
                    actions = o.get("cycle_action_values", o.get("cycle_actions", []))
                assert len(actions) == cycles and 0 <= cycles <= cap, "Incomplete primary action trace"
                assert all(np.isfinite(actions))
                weights[i, j, c, :cycles] = actions
                spec = protocol["policies"][policy]
                if spec["kind"] in ("fixed", "periodic"):
                    np.testing.assert_allclose(actions, np.resize(spec["pattern"], cycles), rtol=0, atol=1e-12)
        for j, method in enumerate(METHODS):
            reported = next(r for r in summary["per_seed"] if r["seed"] == seed and r["method"] == method)
            defined = bool(np.isfinite(costs[i, j]).all())
            assert defined == reported["defined"]
            if defined:
                for k, field in enumerate(FIELDS):
                    np.testing.assert_allclose(costs[i, j, :, k].sum(), reported[field], atol=1e-8, rtol=1e-10)
                assert int((~success[i, j]).sum()) == reported["failed_cases"]
    return summary, seeds, case_ids, costs, success, weights, sources


def paired_gains(costs, success, field=0):
    """Ratios of case sums after repetition averaging; full-success seeds only."""
    total = costs[:, :, :, field].sum(axis=2)
    valid = success.all(axis=2) & np.isfinite(total)
    eligible = valid & valid[:, :1] & (total[:, :1] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        gains = 100 * (1 - total / total[:, :1])
    return np.where(eligible, gains, np.nan)




def generate(run, *, output=None):
    root = Path(run).expanduser().resolve()
    output = Path(output or root / "paper_figures").expanduser().resolve()
    summary, seeds, case_ids, costs, success, weights, sources = read_study(root)
    native = paired_gains(costs, success)
    inclusive = paired_gains(costs, success, 1)
    eligible = np.flatnonzero(np.isfinite(native[:, 4]))
    selected = int(eligible[np.argmax(native[eligible, 4])]) if len(eligible) else 0
    seed = seeds[selected]
    orders = np.array([np.lexsort((case_ids, -np.isfinite(w[0]).sum(axis=1))) for w in weights])
    protocol = summary["protocol"]
    grid = protocol["problem"]["grid"]
    kind = protocol["problem"]["kind"]
    family = "Diffusion–advection" if "advection" in kind or "convection" in kind else "Diffusion"
    title = family + " " + (rf"${grid[0]}^3$" if len(set(grid)) == 1 else " × ".join(map(str, grid)))
    figures = matched_figures(native, weights, case_ids, selected, seeds, title)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "Module 3 fresh five-policy study", "Author": "RLTune-BoomerAMG"}
    with PdfPages(output / "module3_paper_figures.pdf", metadata=metadata) as pdf:
        for name, fig in figures:
            for extension in ("png", "pdf", "svg"):
                fig.savefig(output / f"{name}.{extension}", dpi=260 if extension == "png" else 600)
            pdf.savefig(fig, dpi=600, bbox_inches="tight", pad_inches=.04)
            plt.close(fig)
    np.savez_compressed(output / "plot_data.npz", seeds=np.array(seeds), methods=np.array(METHODS),
                        case_ids=np.array(case_ids), cost_fields=np.array(FIELDS), costs_sec=costs,
                        all_repetitions_success=success, first_repetition_primary_weights=weights,
                        case_order_0_based=orders, native_reductions_pct=native, inclusive_reductions_pct=inclusive)
    captions = (
        f"Module 3: {protocol['name']}. Seeds: {seeds}; {len(case_ids)} paired inputs; {protocol['repetitions_per_case']} evaluation repetitions.\n\n"
        "matched_policy_savings: Percentage reduction against fixed weight one. Repetitions are averaged within input before summing native continuation costs. Open points are seed-level ratios; diamonds and whiskers are their mean and sample SD (no SD for a single seed). Only seeds where the compared policy and reference complete every evaluation successfully enter that comparison. Undefined comparators stay undefined. Native continuation excludes common primary setup and includes recovery setup/solve. Inclusive continuation adds controller computation. The four alternatives to fixed weight one are shown.\n\n"
        f"matched_policy_weights: Selected illustration, seed {seed}. The seed with the largest full-panel native frozen-RL reduction versus weight one among qualifying successful seeds is used; if no seed qualifies, the first listed seed is displayed without a best-run claim. This is an illustrative selection, not a typical-run estimate. Every panel uses the recorded first prescribed repetition, including exploration settings as executed; actions are neither averaged nor extrapolated. Columns use one shared order, decreasing weight-one primary cycle count, with case ID breaking ties. White cells have no recorded primary action after termination, or correspond to an undefined fixed comparator. Recovery cycles are excluded. The four displayed heatmap policies are fixed weight one, per-input best fixed, periodic, and frozen RL.\n\n"
        "Fixed references are freshly scan-selected best-observed grid choices on this evaluation panel, then independently retimed. Scan costs do not substitute for evaluation costs and are excluded from the plotted costs.\n"
    )
    (output / "captions.txt").write_text(captions)
    if Path(__file__).resolve() != (output / "plot_policy.py").resolve():
        shutil.copyfile(__file__, output / "plot_policy.py")
    def finite_list(values):
        return [[float(v) if np.isfinite(v) else None for v in row] for row in values]
    manifest = {"run": str(root), "seeds": seeds, "methods": list(METHODS), "illustrative_seed": seed,
                "selection_qualified": bool(len(eligible)), "trajectory_repetition_0_based": 0,
                "native_reductions_pct": finite_list(native), "inclusive_reductions_pct": finite_list(inclusive),
                "case_ids": case_ids, "case_order_0_based": orders.tolist(),
                "checks": {"passed": True, "raw_hashes": True, "exact_paired_coverage": True,
                           "matches_analysis": True, "timing_accounting": True, "recorded_actions": True},
                "source_sha256": sources,
                "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"},
                "reproduce": f"python -m experiments.paper_final.plot_policy {root}"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"output": str(output), "illustrative_seed": seed, "figures": [name for name, _ in figures], "checks_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.run, output=args.output), indent=2))


if __name__ == "__main__":
    main()
