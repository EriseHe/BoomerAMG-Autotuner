"""Run Module 1 with fixed-range numerical setup descriptors in RL only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from unittest.mock import patch

from experiments.runtime import configure_single_thread, single_thread_environment
configure_single_thread()

from development.setup_normalization import make_range_normalized_setup_encoder
from experiments.paper_final import run_online as runner
from experiments.paper_final.online import methods


SOURCE_FILES = ("development/setup_normalization.py", "development/run_setup_normalization.py")
OFFICIAL_SOURCE_HASHES = runner.source_hashes


def normalization_source_hashes():
    hashes = OFFICIAL_SOURCE_HASHES()
    hashes.update({name: runner.file_hash(runner.ROOT/name) for name in SOURCE_FILES})
    return hashes


class WorkerCommands:
    """Reuse the official seed launcher, barrier and thread limits."""
    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, argv, *args, **kwargs):
        command = list(argv)
        if command[1:4] == ["-m", "experiments.paper_final.run_online", "worker"]:
            command[2] = "development.run_setup_normalization"
        return subprocess.Popen(command, *args, **kwargs)


@contextmanager
def configured_runner():
    with patch.object(runner, "source_hashes", normalization_source_hashes), \
         patch.object(methods, "make_setup_feature_encoder", make_range_normalized_setup_encoder), \
         patch.object(runner, "subprocess", WorkerCommands()):
        yield


def prepare(suite, output):
    with configured_runner():
        protocol = runner.prepare(suite, output)
    if not protocol.get("continuous_stream") or protocol.get("freeze_after") != 5000:
        raise ValueError("Use a continuous Module 1 suite with 5,000 learning and 1,000 frozen inputs")
    protocol["setup_descriptor_normalization"] = dict(
        development_experiment=True, applies_to="both RL controllers only",
        numerical="(value - range midpoint) / range half-width; constant ranges map to zero",
        bounds="fixed allowed parameter grid including Default", clipping=False,
        categorical="unchanged reference one-hot, including conditional masks",
        bandit="unchanged", setup_descriptor_dim=19)
    runner.dump(output/"protocol.json", protocol)
    return protocol


def check_protocol(output):
    declaration = runner.read(output/"protocol.json").get("setup_descriptor_normalization", {})
    if declaration.get("applies_to") != "both RL controllers only" or declaration.get("clipping") is not False:
        raise ValueError("Output was not prepared for fixed-range RL setup normalization")


def sequence(suites, output):
    output.mkdir(parents=True, exist_ok=False)
    names = [suite.stem for suite in suites]
    if len(set(names)) != len(names):
        raise ValueError("Suite names must be unique")
    state = dict(status="preparing", pid=os.getpid(), batch_order=names, completed=[], current=None)
    runner.dump(output/"status.json", state)
    child = None
    guard = runner.prevent_sleep()
    try:
        protocols = [prepare(suite, output/name) for suite, name in zip(suites, names)]
        manifests = [runner.read(output/name/"source_manifest.json") for name in names]
        if any(manifest != manifests[0] for manifest in manifests[1:]):
            raise ValueError("Source changed while preparing suites")
        if any(p["native_sha256"] != protocols[0]["native_sha256"] for p in protocols[1:]):
            raise ValueError("Native libraries changed while preparing suites")
        runner.dump(output/"plan.json", dict(
            development_experiment=True, suites=[str(suite) for suite in suites],
            batches=names, seeds=[p["seeds"] for p in protocols],
            methods=runner.factorial.METHODS, concurrent_seeds=3, sequential_batches=True,
            learning_inputs=[1, 5000], frozen_inputs=[5001, 6000],
            continuous_stream=True, overhead_measured_in_both_phases=True,
            rl_cost_target="native_cycle", failure_score_beta=2., bandit_changed=False,
            source_sha256=runner.digest(manifests[0]), native_sha256=protocols[0]["native_sha256"],
            normalization=protocols[0]["setup_descriptor_normalization"]))
        for name in names:
            batch = output/name
            state.update(status="running", current=name, at=runner.now())
            environment = single_thread_environment()
            environment["PYTHONPATH"] = str(batch/"source")
            with (output/f"{name}.log").open("w") as log:
                child = subprocess.Popen([sys.executable, "-u", "-m", "development.run_setup_normalization",
                    "run", "--output", str(batch)], cwd=batch/"source", env=environment,
                    stdout=log, stderr=subprocess.STDOUT)
                state["batch_pid"] = child.pid
                runner.dump(output/"status.json", state)
                code = child.wait()
                if code:
                    raise RuntimeError(f"{name} failed with status {code}; later batches were not started")
            state["completed"].append(name)
            state.pop("batch_pid", None)
            child = None
            runner.dump(output/"status.json", state)
        rows = [row for name in names for row in runner.read(output/name/"summary.json")["rows"]]
        runner.dump(output/"summary.json", dict(rows=rows))
        runner.write_csv(output/"summary.csv", rows)
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
        runner.stop_sleep_prevention(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sequence", "prepare", "run", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path, action="append", default=[])
    parser.add_argument("--label")
    args = parser.parse_args()
    output = args.output.resolve()
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    if args.command == "sequence":
        if not args.suite:
            parser.error("sequence requires --suite")
        sequence([suite.resolve() for suite in args.suite], output)
    elif args.command == "prepare":
        if len(args.suite) != 1:
            parser.error("prepare requires exactly one --suite")
        prepare(args.suite[0].resolve(), output)
    else:
        if args.command == "worker" and args.label is None:
            parser.error("worker requires --label")
        check_protocol(output)
        with configured_runner():
            if args.command == "worker":
                runner.worker(output, args.label)
            else:
                runner.launch(output)


if __name__ == "__main__":
    main()
