"""Private paired reward pilot; reuses the official episode/recovery runners.

Nothing in the official data or manuscript is written. No existing output may
be resumed or overwritten. This is an LSTDQ objective ablation, not PPO.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.runtime import configure_single_thread, single_thread_environment, prevent_sleep, stop_sleep_prevention

configure_single_thread()
sys.dont_write_bytecode = True

import numpy as np

from experiments.paper_final.common.artifacts import dump, read, ready, file_hash, now
from experiments.paper_final.online.configuration import parse_joint_experiment_config, runtime_config_from_spec
from experiments.paper_final.online.case_loop import configure_paired_environment, report_online_outcome
from experiments.paper_final.online.methods import build_composable_solve_runtime
from experiments.paper_final.online.factorial import balanced_orders
from experiments.paper_final.online.native_evaluation import solve_no_rl_case
from experiments.paper_final.recovery.execution import EpisodeRecorder, run_fixed_case
from experiments.paper_final.run_06_recovery import request_performance_qos
from hypre.bindings import augment_setup_params
from hypre.bindings.recovery import InvalidObservationError
from setup.space import DEFAULT_SETUP_PARAMS
from solve.controllers.common import ControllerBundle
from solve.controllers.recursive_lstdq import RecursiveLstdqController
from solve.core.episode import run_td_episode

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent / "RLTune-BoomerAMG"
METHODS = ("time_original", "time_unclipped", "residual_rate")
LABELS = {"time_original": "Time cost, original clipping", "time_unclipped": "Time cost, unclipped", "residual_rate": "Residual progress/time, unclipped"}
REFERENCE_TIME = 0.01  # seconds; fixed before any pilot measurements
CASES = 500


def git_state(directory):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()
    return {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "status": git("status", "--porcelain=v1", "--untracked-files=all")}


def source_hashes():
    files = []
    for package in ("experiments", "solve", "setup", "problems", "hypre/bindings"):
        files.extend((ROOT / package).rglob("*.py"))
    files.extend([ROOT / "hypre/interfaces/libamg_runtime.dylib", ROOT / "hypre/install/lib/libHYPRE.300.dylib"])
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


def target_cost(method, cycle_time, ratio):
    if not math.isfinite(cycle_time) or cycle_time <= 0:
        raise InvalidObservationError("Reward comparison needs a positive finite cycle time")
    if method != "residual_rate":
        return float(cycle_time)
    if ratio is None or not math.isfinite(ratio) or ratio < 0:
        raise InvalidObservationError("Residual-rate reward is undefined for this residual observation")
    # Positive, predeclared scaling preserves the reward objective. It puts
    # its numerical scale near seconds for the existing covariance prior.
    return -(REFERENCE_TIME ** 2) * (1.0 - ratio) / cycle_time


class ObjectiveController(RecursiveLstdqController):
    def __init__(self, *, objective, **kwargs):
        super().__init__(**kwargs)
        self.objective = objective
        self.target_trace = []

    def start_episode(self, **kwargs):
        self.target_trace = []
        return super().start_episode(**kwargs)

    def update(self, **kwargs):
        dt, ratio = kwargs.get("native_cycle_cost"), kwargs.get("residual_ratio")
        if dt is None:
            raise InvalidObservationError("Native-error cycle has no defined reward; preserve evidence and stop")
        if not math.isclose(kwargs["cost"], dt, rel_tol=1e-12, abs_tol=1e-14):
            raise InvalidObservationError("Recovery or penalty entered the within-attempt target")
        cost = target_cost(self.objective, float(dt), ratio)
        self.target_trace.append({"cycle_time": float(dt), "residual_ratio": ratio,
                                  "learning_cost": cost, "terminal": bool(kwargs["terminal"])})
        return super().update(**dict(kwargs, cost=cost))


def bundles_for(config, destination):
    spec = parse_joint_experiment_config(config, output_dir_override=destination)
    args = runtime_config_from_spec(spec)
    configure_paired_environment(args)
    method_spec = next(m for m in spec.methods if m.name == "default_setup_lstdq")
    bundles = {}
    for name in METHODS:
        original = build_composable_solve_runtime(args, specs=[method_spec]).controller_bundles[method_spec.name]
        old = original.controller
        algorithm = replace(old.spec, lcb_lower_bound_sec=0.0 if name == "time_original" else None)
        controller = ObjectiveController(objective=name, feature_dim=old.feature_dim,
            config=old.config, spec=algorithm, seed=int(args.controller_seed + 2018 + method_spec.seed_offset))
        bundles[name] = ControllerBundle(controller=controller, encoder=original.encoder,
            _protocol_metadata=original.protocol_metadata())
    for name, b in bundles.items():
        assert b.controller.steps == b.controller.episodes == b.controller.sample_count == 0
        assert not np.any(b.controller.theta) and not np.any(b.controller.b)
        assert b.controller.config.gamma == 1 and b.controller.spec.uncertainty_beta == 2
        for other, c in bundles.items():
            if name != other:
                for key in ("theta", "b", "a_matrix", "a_inverse", "episode_moment_covariance"):
                    assert not np.shares_memory(getattr(b.controller, key), getattr(c.controller, key))
    return args, bundles


def execute(bundle, args, item, name, index, rank):
    """Same fixed-setup lifecycle as factorial.execute_fixed_rl; reward-aware audit below."""
    started = time.perf_counter()
    tick = time.perf_counter()
    recorder = EpisodeRecorder(bundle.controller)
    snapshot_time = time.perf_counter() - tick
    traces = []
    def primary():
        result = run_td_episode(mkw=dict(item["mkw"]), params=dict(DEFAULT_SETUP_PARAMS),
            controller=recorder, encoder=bundle.encoder, problem_context=item["context"],
            solve_tol=args.tol, solve_max_cycles=args.max_cycles, learn=True, explore=True,
            record_action_metadata=True, fallback_attempt=None, failure_penalty_sec=0.0)
        result["recovery_protocol_applied"] = False
        traces.append(result)
        return result
    def fallback():
        return solve_no_rl_case(params=dict(DEFAULT_SETUP_PARAMS), mkw=dict(item["mkw"]),
            solver_tol=args.tol, solver_max_iter=args.max_cycles, augment_params=augment_setup_params)
    raw = run_fixed_case(primary, recorder, snapshot_time, False,
                         fallback_solver=fallback, max_attempts=1)
    out = report_online_outcome(raw, bandit_timing={})
    out["method_wall_runtime"] = time.perf_counter() - started
    row = {"problem_index": index + 1, "method": name, "rank": rank,
           "mkw": item["mkw"], "context": item["context"], "params": dict(DEFAULT_SETUP_PARAMS),
           "outcome": out, "trace": traces[0], "learning_targets": bundle.controller.target_trace,
           "steps_after": bundle.controller.steps, "episodes_after": bundle.controller.episodes,
           "pid": os.getpid(), "started_monotonic": started, "ended_monotonic": time.perf_counter()}
    verify_row(row)
    return row


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10), (a, b)


def verify_row(row):
    name, out, tr = row["method"], row["outcome"], row["trace"]
    assert row["params"] == DEFAULT_SETUP_PARAMS
    assert out["primary_attempt_count"] == 1 and out["fixed_setup_retries"] == 0
    first = out["primary_attempts"][0]
    assert out["fallback_used"] == (first["status"] != "success")
    assert not out["replayed_transitions"] and not out["target_replay_runtime"]
    assert first.get("rl_recovery_cost") in (None, 0.0)
    assert not (out["failed"] and out["controller_update_committed"])
    dt = tr["cycle_times"]
    ratios = tr["cycle_residual_ratios"]
    targets = row["learning_targets"]
    assert len(dt) == len(ratios) == len(tr["cycle_actions"]) == len(targets) == first["cycles"]
    close(sum(dt), first["solve_runtime"])
    close(first["setup_runtime"] + first["solve_runtime"] + out["fallback_setup_runtime"] + out["fallback_solve_runtime"], out["runtime"])
    close(out["runtime"] + out["infer_runtime"], out["end_to_end_runtime"])
    for k, (duration, ratio, target) in enumerate(zip(dt, ratios, targets)):
        close(duration, target["cycle_time"])
        close(ratio, target["residual_ratio"])
        # Independently written equation, not target_cost().
        expected = duration if name != "residual_rate" else -0.0001 * (1 - ratio) / duration
        close(expected, target["learning_cost"])
        assert target["terminal"] == (k == len(dt) - 1)
    for q, u, score in zip(tr["cycle_selected_q_values"], tr["cycle_selected_uncertainties"], tr["cycle_selection_scores"]):
        assert math.isfinite(q) and math.isfinite(u) and u >= 0
        expected = q - 2 * u
        close(max(expected, 0) if name == "time_original" else expected, score)
    if first["status"] == "success":
        assert first["residual_norm"] < 1e-6 and first["cycles"] < 50
    if out["fallback_used"] and not out["failed"]:
        assert out["fallback_status"] == "success"
    if not out["failed"]:
        assert out["controller_update_committed"]


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    assert git_state(MAIN)["branch"] == "main"
    protocol = {"created": now(), "purpose": "Private LSTDQ reward-objective pilot, not a reproduction of Huergo PPO",
        "cases_per_seed": CASES, "seeds": [1, 2, 3], "methods": METHODS,
        "main_before": git_state(MAIN), "development_before": git_state(ROOT),
        "source_sha256": source_hashes(), "reference_time_sec": REFERENCE_TIME,
        "objectives": {"time_original": "dt with scores clipped at zero",
                       "time_unclipped": "dt with unclipped scores",
                       "residual_rate": "-0.01^2 * (1 - r_new/r_old) / dt with unclipped scores"},
        "controls": "Fixed exact Default setup; cold independent LSTDQ branches; same input order, state, action grid, RNG seed, gamma=1, beta=2, epsilon schedule, cap50 and strict tolerance1e-6. Existing whole-case failure rollback and one Default fallback; no recovery costs in solve target.",
        "interpretation": "Primary objective comparison is residual_rate vs time_unclipped. time_original checks the impact of removing the zero score bound. Fixed positive reward scaling is predeclared; this small cold-start pilot is not a tuned comparison or proof of general superiority.",
        "seeds_inputs": {}}
    for seed in (1, 2, 3):
        source = ROOT / f"data/official/module04/batches/batch_01/diffusion_60/seed_{seed}"
        config = read(source / "experiment_config.json")
        payload = (source / "inputs.jsonl").read_bytes().splitlines(keepends=True)
        assert len(payload) == 5000
        dest = output / f"seed_{seed}"
        dest.mkdir()
        (dest / "inputs.jsonl").write_bytes(b"".join(payload[:CASES]))
        (dest / "validation_inputs.jsonl").write_bytes(b"".join(payload[CASES:CASES + 3]))
        dump(dest / "source_config.json", config)
        protocol["seeds_inputs"][str(seed)] = {"official_inputs": str(source / "inputs.jsonl"),
            "official_sha256": file_hash(source / "inputs.jsonl"),
            "pilot_sha256": file_hash(dest / "inputs.jsonl"),
            "validation_sha256": file_hash(dest / "validation_inputs.jsonl"),
            "config_sha256": file_hash(dest / "source_config.json")}
    dump(output / "protocol.json", protocol)
    print(json.dumps({"prepared": str(output), "cases_per_seed": CASES, "seeds": [1, 2, 3]}), flush=True)


def worker(output, seed, validation):
    from mpi4py import MPI  # Initialize MPI before native solver entry.
    # Inspect the actual BLAS runtime without adding a package to the user's environment.
    blas_path = Path(sys.prefix) / "lib/libopenblas.0.dylib"
    blas = ctypes.CDLL(str(blas_path))
    blas.openblas_get_num_threads.restype = ctypes.c_int
    blas.openblas_set_num_threads.argtypes = [ctypes.c_int]
    blas.openblas_set_num_threads(1)
    threadpools = [{"library": str(blas_path.resolve()), "num_threads": int(blas.openblas_get_num_threads())}]
    qos = request_performance_qos()
    protocol = read(output / "protocol.json")
    assert source_hashes() == protocol["source_sha256"]
    directory = output / f"seed_{seed}"
    config = read(directory / "source_config.json")
    assert file_hash(directory / "source_config.json") == protocol["seeds_inputs"][str(seed)]["config_sha256"]
    phase = "validation" if validation else "run"
    if not validation:
        assert read(output / "validation_verified.json")["passed"]
    destination = directory / phase
    destination.mkdir(exist_ok=False)
    filename = "validation_inputs.jsonl" if validation else "inputs.jsonl"
    expected_hash = protocol["seeds_inputs"][str(seed)]["validation_sha256" if validation else "pilot_sha256"]
    assert file_hash(directory / filename) == expected_hash
    items = [json.loads(line) for line in (directory / filename).read_text().splitlines()]
    args, bundles = bundles_for(config, destination)
    dump(destination / "cold_start.json", {"passed": True, "states": {k: b.summary() for k, b in bundles.items()},
        "threadpools": threadpools, "qos": qos, "pid": os.getpid()})
    assert all(p["num_threads"] == 1 for p in threadpools)
    orders = balanced_orders(len(items), int(config["seeds"]["method_order"]), methods=METHODS)
    started = time.time()
    with (destination / "trajectory.jsonl").open("x") as handle:
        for index, (item, order) in enumerate(zip(items, orders)):
            for rank, name in enumerate(order):
                row = execute(bundles[name], args, item, name, index, rank)
                handle.write(json.dumps(ready(row), allow_nan=False) + "\n")
            handle.flush()
            if (index + 1) % 50 == 0 or index + 1 == len(items):
                dump(destination / "status.json", {"completed": index + 1, "total": len(items),
                    "elapsed_sec": time.time() - started, "updated": now(), "pid": os.getpid()})
    checkpoints = {}
    for name, bundle in bundles.items():
        path = destination / f"{name}.npz"
        bundle.save(path)
        checkpoints[name] = file_hash(path)
    assert source_hashes() == protocol["source_sha256"]
    dump(destination / "complete.json", {"passed": True, "cases": len(items), "rows": len(items)*3,
        "elapsed_sec": time.time() - started, "finished": now(), "checkpoint_sha256": checkpoints,
        "states": {k: b.summary() for k, b in bundles.items()}})
    print(json.dumps({"seed": seed, "phase": phase, "complete": True}), flush=True)


def verify(output, validation=False):
    protocol = read(output / "protocol.json")
    assert source_hashes() == protocol["source_sha256"]
    assert git_state(MAIN) == protocol["main_before"], "Main checkout changed during this pilot"
    phase = "validation" if validation else "run"
    results, comparisons = [], []
    for seed in (1, 2, 3):
        directory = output / f"seed_{seed}"
        manifest = protocol["seeds_inputs"][str(seed)]
        assert file_hash(Path(manifest["official_inputs"])) == manifest["official_sha256"]
        assert file_hash(directory / "inputs.jsonl") == manifest["pilot_sha256"]
        assert file_hash(directory / "source_config.json") == manifest["config_sha256"]
        source_lines = Path(manifest["official_inputs"]).read_bytes().splitlines(keepends=True)
        assert (directory / "inputs.jsonl").read_bytes() == b"".join(source_lines[:500])
        cases = 3 if validation else CASES
        records = [json.loads(line) for line in (directory / phase / "trajectory.jsonl").read_text().splitlines()]
        assert len(records) == cases * len(METHODS)
        items = [json.loads(line) for line in (directory / ("validation_inputs.jsonl" if validation else "inputs.jsonl")).read_text().splitlines()]
        config = read(directory / "source_config.json")
        orders = balanced_orders(cases, int(config["seeds"]["method_order"]), methods=METHODS)
        completion = read(directory / phase / "complete.json")
        assert completion["passed"] and completion["cases"] == cases
        assert read(directory / phase / "cold_start.json")["passed"]
        previous_end = 0
        for i, row in enumerate(records):
            index, rank = divmod(i, 3)
            assert row["method"] == orders[index][rank] and row["problem_index"] == index + 1 and row["rank"] == rank
            assert row["mkw"] == items[index]["mkw"] and row["context"] == items[index]["context"]
            assert row["started_monotonic"] >= previous_end
            previous_end = row["ended_monotonic"]
            verify_row(row)
        for name in METHODS:
            branch = [r for r in records if r["method"] == name]
            assert file_hash(directory / phase / f"{name}.npz") == completion["checkpoint_sha256"][name]
            with np.load(directory / phase / f"{name}.npz", allow_pickle=False) as cp:
                assert int(cp["sample_count"]) == branch[-1]["steps_after"]
                assert int(cp["episode_moment_count"]) == branch[-1]["episodes_after"]
                assert np.all(np.isfinite(cp["theta"]))
            for window, selected in (("full", branch), ("final100", branch[-100:])):
                outcomes = [r["outcome"] for r in selected]
                entry = {"seed": seed, "method": name, "window": window, "cases": len(selected)}
                for field in ("setup_runtime", "solve_runtime", "runtime", "infer_runtime", "end_to_end_runtime"):
                    entry[field + "_mean_ms"] = 1000 * np.mean([r[field] for r in outcomes])
                entry["cycles_mean"] = np.mean([r["primary_attempts"][0]["cycles"] for r in outcomes])
                entry["first_failures"] = sum(r["primary_attempts"][0]["status"] != "success" for r in outcomes)
                entry["final_failures"] = sum(r["failed"] for r in outcomes)
                entry["fallbacks"] = sum(r["fallback_used"] for r in outcomes)
                results.append(entry)
        for window in ("full", "final100"):
            subset = {r["method"]: r for r in results if r["seed"] == seed and r["window"] == window}
            for baseline in ("time_original", "time_unclipped"):
                comparisons.append({"seed": seed, "window": window, "baseline": baseline,
                    **{field + "_reduction_pct": 100 * (1 - subset["residual_rate"][field + "_mean_ms"] / subset[baseline][field + "_mean_ms"])
                       for field in ("solve_runtime", "runtime", "end_to_end_runtime")}})
    summary = {"passed": True, "phase": phase, "verified_method_cases": (3 if validation else CASES)*9,
               "verified_at": now(), "rows": results, "comparisons": comparisons, "main_unchanged": True}
    dump(output / ("validation_verified.json" if validation else "summary.json"), summary)
    if not validation:
        import csv
        with (output / "summary.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)
        lines = ["# Private reward comparison", "", "Three cold seeds, 500 paired diffusion 60³ inputs each; fixed Default setup. All figures include any failed attempts and Default fallback. Native total is setup + solve; inclusive adds controller computation.", "",
                 "This tests Huergo-style residual-progress reward inside our LSTDQ controller (gamma=1), not their PPO method. Primary comparison uses unclipped scores for both objectives. Reward uses the fixed positive scale 0.01², declared before measurements. Original clipped time-cost controller is an additional reference.", "",
                 "| Window | Seed | Method | Setup ms | Solve ms | Native ms | Learner ms | Inclusive ms | Cycles | First/final failures | Fallbacks |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in results:
            lines.append(f"| {r['window']} | {r['seed']} | {LABELS[r['method']]} | {r['setup_runtime_mean_ms']:.3f} | {r['solve_runtime_mean_ms']:.3f} | {r['runtime_mean_ms']:.3f} | {r['infer_runtime_mean_ms']:.3f} | {r['end_to_end_runtime_mean_ms']:.3f} | {r['cycles_mean']:.2f} | {r['first_failures']}/{r['final_failures']} | {r['fallbacks']} |")
        lines.extend(["", "Positive reductions favor residual-progress reward. These are paired per-seed ratios; no pooled ratio substitutes for a seed mean.", "", "| Window | Seed | Compared with | Solve reduction % | Native reduction % | Inclusive reduction % |", "|---|---:|---|---:|---:|---:|"])
        for r in comparisons:
            lines.append(f"| {r['window']} | {r['seed']} | {r['baseline']} | {r['solve_runtime_reduction_pct']:.2f} | {r['runtime_reduction_pct']:.2f} | {r['end_to_end_runtime_reduction_pct']:.2f} |")
        lines.extend(["", "Audited: all 4,500 method-cases; cold independent states; exact paired inputs; sequential balanced method order within each seed; native/source/config/checkpoint identities; cycle targets and score equations; timing sums and failure counts. Main Git state unchanged.", "", "Limit: short exploratory pilot on one PDE family and one prescribed reward scale. No claim of general superiority or reproduction of PPO."])
        (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def coordinate(output, validation):
    phase = "validation" if validation else "run"
    receipt = output / f"{phase}_processes.json"
    if receipt.exists():
        raise RuntimeError("This phase has already been launched; no rerun/resume is allowed")
    if not validation:
        assert read(output / "validation_verified.json")["passed"]
    protection = prevent_sleep()
    children = []
    handles = []
    try:
        for seed in (1, 2, 3):
            log = (output / f"{phase}_seed_{seed}.log").open("x")
            handles.append(log)
            cmd = [sys.executable, "-B", "-u", "-m", "experiments.reward_objective_pilot", "worker", "--output", str(output), "--seed", str(seed)]
            if validation:
                cmd.append("--validation")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=single_thread_environment(), stdout=log, stderr=subprocess.STDOUT)
            children.append(proc)
        dump(receipt, {"started": now(), "coordinator_pid": os.getpid(), "workers": [p.pid for p in children]})
        statuses = [p.wait() for p in children]
        dump(output / f"{phase}_process_completion.json", {"finished": now(), "returncodes": statuses})
        if any(statuses):
            raise RuntimeError(f"A worker failed; outputs retained: {statuses}")
        result = verify(output, validation)
        print(json.dumps({"phase": phase, "passed": result["passed"], "rows": result["verified_method_cases"]}), flush=True)
    finally:
        for handle in handles:
            handle.close()
        stop_sleep_prevention(protection)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "coordinate", "worker", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3))
    parser.add_argument("--validation", action="store_true")
    cli = parser.parse_args()
    output = cli.output.resolve()
    if cli.mode == "prepare": prepare(output)
    elif cli.mode == "worker": worker(output, cli.seed, cli.validation)
    elif cli.mode == "coordinate": coordinate(output, cli.validation)
    else: print(json.dumps(ready(verify(output, cli.validation)), indent=2))


if __name__ == "__main__":
    main()
