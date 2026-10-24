"""Run ordered Module 3 retry comparisons through the official online runner."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from unittest.mock import patch

from experiments.runtime import configure_single_thread, single_thread_environment
configure_single_thread()

import numpy as np

from development.retry_comparison import (
    AlphaFailureRetryLinUCB, ConservativeRetryLinUCB, observed_model,
)
from experiments.paper_final import run_online as runner
from experiments.paper_final.recovery.setup_feedback import FailureAwareRetryLinUCB


VARIANTS = {
    "failure_fixed": dict(label="Failure model, fixed coefficient 2",
        model=FailureAwareRetryLinUCB, rule="failure_model_adjacent", coefficient="2"),
    "failure_alpha": dict(label="Failure model, runtime alpha",
        model=AlphaFailureRetryLinUCB, rule="failure_model_alpha_adjacent", coefficient="alpha_n"),
    "runtime_conservative": dict(label="Conservative runtime, no failure predictor",
        model=ConservativeRetryLinUCB, rule="conservative_adjacent", coefficient=None),
}
SOURCE_FILES = ("development/retry_comparison.py", "development/run_retry_comparison.py")
OFFICIAL_SOURCE_HASHES = runner.source_hashes
OFFICIAL_INITIALIZE = runner.initialize


def comparison_source_hashes():
    hashes = OFFICIAL_SOURCE_HASHES()
    hashes.update({name: runner.file_hash(runner.ROOT/name) for name in SOURCE_FILES})
    return hashes


def audit_runtime_independence(runtimes, *, allow_trained=False):
    """Apply the runner's independence checks to models without failure weights."""
    methods = runner.factorial.METHODS
    models = [runtimes[name].branch.policy.model for name in methods[2:]]
    if any(hasattr(model, "failure_b") for model in models):
        raise ValueError("Runtime-only comparison unexpectedly has a failure predictor")
    for key in ("A_inv", "b"):
        if np.shares_memory(getattr(models[0], key), getattr(models[1], key)):
            raise ValueError(f"Setup models share {key}")
    if models[0].rng is models[1].rng or models[0]._candidate_schedule is models[1]._candidate_schedule:
        raise ValueError("Setup RNG or schedule cursors are shared")
    bundles = [runtimes[name].bundle for name in (methods[1], methods[3])]
    controllers = [bundle.controller for bundle in bundles]
    for key in ("a_matrix", "a_inverse", "b", "theta", "episode_moment_covariance"):
        if np.shares_memory(getattr(controllers[0], key), getattr(controllers[1], key)):
            raise ValueError(f"Solve controllers share {key}")
    if controllers[0].rng is controllers[1].rng:
        raise ValueError("Solve controllers share RNG")
    if not allow_trained and (any(c.steps or c.episodes for c in controllers) or any(m.t for m in models)):
        raise ValueError("Learners trained before measurement")
    encodings = [bundle.protocol_metadata()["state_encoder"] for bundle in bundles]
    if runner.digest(encodings[0]) != runner.digest(encodings[1]):
        raise ValueError("Solve-state encodings differ across branches")
    return dict(passed=True, independent_setup_models=2, independent_controllers=2,
                identical_solve_encoding=True, solve_state=encodings[0], failure_predictor=False)


class WorkerCommands:
    """Keep the official launcher and route its workers through this entry point."""
    def __init__(self, variant):
        self.variant = variant

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, argv, *args, **kwargs):
        command = list(argv)
        if command[1:4] == ["-m", "experiments.paper_final.run_online", "worker"]:
            command[2] = "development.run_retry_comparison"
            command.extend(["--variant", self.variant])
        return subprocess.Popen(command, *args, **kwargs)


def initialize_variant(output, label, variant):
    runtimes = OFFICIAL_INITIALIZE(output, label)
    checks = {}
    for name in runner.factorial.METHODS[2:]:
        model = runtimes[name].branch.policy.model
        has_failure = variant != "runtime_conservative"
        if hasattr(model, "failure_b") != has_failure:
            raise ValueError("Wrong predictor for the declared retry comparison")
        if not isinstance(model, VARIANTS[variant]["model"]):
            raise ValueError("Wrong retry selector")
        if model.alpha != 1. or not model.alpha_decay or model.alpha_decay_burnin_rounds:
            raise ValueError("Expected the official alpha_n = 1/sqrt(n+1) schedule")
        checks[name] = dict(model_class=type(model).__name__, failure_predictor=has_failure,
            runtime_coefficient="1/sqrt(n+1)", failure_coefficient=VARIANTS[variant]["coefficient"],
            initial_observations=model.t)
    runner.dump(output/f"seed_{label}"/"retry_selector.json", dict(variant=variant, models=checks, passed=True))
    return runtimes


@contextmanager
def configured_runner(variant):
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner, "source_hashes", comparison_source_hashes))
        stack.enter_context(patch.object(runner, "FailureAwareRetryLinUCB", VARIANTS[variant]["model"]))
        stack.enter_context(patch.object(runner, "observed_model", observed_model))
        stack.enter_context(patch.object(runner, "subprocess", WorkerCommands(variant)))
        stack.enter_context(patch.object(runner, "initialize",
            lambda output, label: initialize_variant(output, label, variant)))
        if variant == "runtime_conservative":
            stack.enter_context(patch.object(runner.factorial, "audit_independence", audit_runtime_independence))
        yield


def prepare(suite, output, variant, cases=None):
    suite_data = runner.read(suite)
    for entry in suite_data["seeds"]:
        config = Path(entry["config"])
        if not config.is_absolute():
            config = suite.parent/config
        raw = runner.read(config)
        if raw["problem"]["kind"] != "rs_convection_3d_type2" or "frozen_stream" in raw or "evaluation" in raw:
            raise ValueError("This comparison requires fresh Module 3 variable-flow learning streams")
    with configured_runner(variant):
        protocol = runner.prepare(suite, output, cases=cases)
    protocol.update(name=f"{suite_data['name']} — {VARIANTS[variant]['label']}",
        recovery=VARIANTS[variant]["rule"],
        failure_score_beta=2. if variant == "failure_fixed" else None,
        retry_comparison=dict(variant=variant, failure_predictor=variant != "runtime_conservative",
            failure_coefficient=VARIANTS[variant]["coefficient"], runtime_coefficient="1/sqrt(n+1)",
            feedback="adjacent_delayed", development_experiment=True))
    runner.dump(output/"protocol.json", protocol)
    return protocol


def check_variant(output, variant):
    if runner.read(output/"protocol.json")["retry_comparison"]["variant"] != variant:
        raise ValueError("Requested selector differs from the prepared comparison")


def compare(output):
    rows = []
    for variant in VARIANTS:
        summary = runner.read(output/variant/"summary.json")
        rows.extend(dict(variant=variant, label=VARIANTS[variant]["label"], **row)
                    for row in summary["rows"] if row["window"] == "full")
    aggregates = []
    for variant in VARIANTS:
        for method in runner.factorial.METHODS:
            selected = [row for row in rows if row["variant"] == variant and row["method"] == method]
            aggregates.append(dict(variant=variant, method=method, seeds=len(selected),
                native_reduction_pct_mean=float(np.mean([r["native_reduction_pct"] for r in selected])),
                inclusive_reduction_pct_mean=float(np.mean([r["inclusive_reduction_pct"] for r in selected])),
                first_failures=sum(r["first_failures"] for r in selected),
                final_failures=sum(r["failures"] for r in selected),
                fallbacks=sum(r["fallbacks"] for r in selected)))
    runner.write_csv(output/"comparison.csv", rows)
    runner.dump(output/"comparison.json", dict(rows=rows, aggregates=aggregates))
    lines = ["# Module 3 retry comparison", "",
        "Each batch independently measures all four methods on the same paired inputs. Reductions use that batch's Default measurement. Means are across seeds; failure and fallback counts are summed.", "",
        "| Retry rule | Method | Native reduction | Including overhead | First failures | Final failures | Fallbacks |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        lines.append(f"| {VARIANTS[row['variant']]['label']} | {runner.factorial.LABELS[row['method']]} | "
            f"{row['native_reduction_pct_mean']:.2f}% | {row['inclusive_reduction_pct_mean']:.2f}% | "
            f"{row['first_failures']} | {row['final_failures']} | {row['fallbacks']} |")
    (output/"comparison.md").write_text("\n".join(lines)+"\n")


def sequence(suite, output, cases=None):
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="preparing", pid=os.getpid(), completed=[], batch_order=list(VARIANTS), current=None)
    runner.dump(output/"status.json", state)
    child = None
    try:
        protocols = {}
        for variant in VARIANTS:
            protocols[variant] = prepare(suite, output/variant, variant, cases)
        reference = protocols["failure_fixed"]
        paired = {label: stage["inputs_sha256"] for label, stage in reference["stages"].items()}
        for protocol in protocols.values():
            if paired != {label: stage["inputs_sha256"] for label, stage in protocol["stages"].items()}:
                raise ValueError("Comparison batches use different input streams")
            if protocol["native_sha256"] != reference["native_sha256"]:
                raise ValueError("Comparison batches use different native libraries")
        runner.dump(output/"plan.json", dict(suite=str(suite), batch_order=list(VARIANTS),
            seeds=reference["seeds"], stages=reference["stages"], paired_inputs=paired,
            concurrent_seeds=len(reference["seeds"]), sequential_batches=True,
            default_fallback_feedback="successful_default_only", rl_cost_target="native_cycle",
            native_and_inclusive_costs_measured=True,
            source_manifest=runner.read(output/"failure_fixed/source_manifest.json")))
        for variant in VARIANTS:
            state.update(status="running", current=variant, at=runner.now())
            runner.dump(output/"status.json", state)
            batch = output/variant
            environment = single_thread_environment()
            environment["PYTHONPATH"] = str(batch/"source")
            with (output/f"{variant}.log").open("w") as log:
                child = subprocess.Popen([sys.executable, "-u", "-m", "development.run_retry_comparison",
                    "run", "--output", str(batch), "--variant", variant],
                    cwd=batch/"source", env=environment, stdout=log, stderr=subprocess.STDOUT)
                state["batch_pid"] = child.pid
                runner.dump(output/"status.json", state)
                code = child.wait()
                if code:
                    raise RuntimeError(f"{variant} exited with status {code}; later batches were not started")
            state["completed"].append(variant)
            state.pop("batch_pid", None)
            child = None
            runner.dump(output/"status.json", state)
        compare(output)
        state.update(status="complete", current=None, at=runner.now())
        runner.dump(output/"status.json", state)
    except BaseException:
        state.update(status="failed", traceback=traceback.format_exc(), at=runner.now())
        runner.dump(output/"status.json", state)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sequence", "prepare", "run", "worker", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--label")
    parser.add_argument("--cases", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command in ("prepare", "sequence") and args.suite is None:
        parser.error("Preparation requires --suite")
    if args.command != "sequence" and args.variant is None:
        parser.error("This command requires --variant")
    if args.command == "worker" and args.label is None:
        parser.error("A worker requires --label")
    if args.command == "sequence":
        sequence(args.suite.resolve(), output, args.cases)
    elif args.command == "prepare":
        prepare(args.suite.resolve(), output, args.variant, args.cases)
    else:
        check_variant(output, args.variant)
        with configured_runner(args.variant):
            if args.command == "worker":
                runner.worker(output, args.label)
            elif args.command == "run":
                runner.launch(output)
            else:
                runner.analyze(output)


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt("Retry comparison stopped")
    signal.signal(signal.SIGTERM, terminate)
    main()
