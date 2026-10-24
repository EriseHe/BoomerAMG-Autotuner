"""Run the four Module 1 branches with normalized setup descriptors and log time."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
import signal
import subprocess
from unittest.mock import patch

from experiments.runtime import configure_single_thread
configure_single_thread()

from development.compact_context import CompactSolveStateEncoder
from development.setup_normalization import make_range_normalized_setup_encoder
from experiments.paper_final import run_online as runner
from experiments.paper_final.online import methods
from solve.controllers.common import config as state_config


SOURCE_FILES = ("development/compact_context.py", "development/run_compact_context.py",
                "development/setup_normalization.py")
TIME_FORMULAS = {"log": "log1p(last_native_cycle_seconds / 0.001)",
                 "first_cycle_ratio": "last_native_cycle_seconds / first_native_cycle_seconds"}
OFFICIAL_SOURCE_HASHES = runner.source_hashes
OFFICIAL_INITIALIZE = runner.initialize
OFFICIAL_INDEPENDENCE = runner.factorial.audit_independence


def context_source_hashes():
    hashes = OFFICIAL_SOURCE_HASHES()
    hashes.update({name: runner.file_hash(runner.ROOT/name) for name in SOURCE_FILES})
    return hashes


def build_solve_runtime(args, *, specs, cycle_time_mode="log"):
    bundles = {}
    for method in specs:
        with patch.object(state_config, "SolveStateEncoder", partial(CompactSolveStateEncoder, cycle_time_mode=cycle_time_mode)), \
             patch.object(methods, "make_setup_feature_encoder", make_range_normalized_setup_encoder):
            built = methods.build_composable_solve_runtime(args, specs=(method,))
        for name, bundle in built.controller_bundles.items():
            metadata = bundle.protocol_metadata()
            metadata["state_encoder"].update(bundle.encoder.metadata())
            bundles[name] = replace(bundle, _protocol_metadata=metadata)
    return methods.ComposableSolveRuntime(controller_bundles=bundles)


def audit_independence(runtimes, *, allow_trained=False):
    """Verify independent learners with the same 32-entry solve encoding."""
    result = OFFICIAL_INDEPENDENCE(runtimes, allow_trained=allow_trained)
    for name in (runner.factorial.METHODS[1], runner.factorial.METHODS[3]):
        bundle = runtimes[name].bundle
        if bundle.encoder.feature_dim != 32 or bundle.controller.joint_dim != 288:
            raise ValueError("Expected 32 state entries and 288 state-action coefficients")
    return result


def initialize(output, label):
    runtimes = OFFICIAL_INITIALIZE(output, label)
    audit_independence(runtimes)
    controllers = {}
    for name in (runner.factorial.METHODS[1], runner.factorial.METHODS[3]):
        bundle = runtimes[name].bundle
        if bundle.encoder.setup_obs_encoder.descriptor_dim != 19:
            raise ValueError("Expected the 19 basic setup descriptors")
        controllers[name] = dict(setup_features=19, state_features=bundle.encoder.feature_dim,
            state_action_features=bundle.controller.joint_dim,
            metadata=bundle.protocol_metadata()["state_encoder"])
    runner.dump(output/f"seed_{label}"/"compact_context_audit.json",
                dict(passed=True, controllers=controllers, bandit_unchanged=True,
                     rl_reward="native_cycle", additional_clipping=False))
    return runtimes


class WorkerCommands:
    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, argv, *args, **kwargs):
        command = list(argv)
        if command[1:4] == ["-m", "experiments.paper_final.run_online", "worker"]:
            command[2] = "development.run_compact_context"
        return subprocess.Popen(command, *args, **kwargs)


@contextmanager
def configured_runner(*, cycle_time_mode="log"):
    with patch.object(runner, "source_hashes", context_source_hashes), \
         patch.object(runner, "build_composable_solve_runtime",
                      partial(build_solve_runtime, cycle_time_mode=cycle_time_mode)), \
         patch.object(runner, "initialize", initialize), \
         patch.object(runner.factorial, "audit_independence",
                      audit_independence), \
         patch.object(runner, "subprocess", WorkerCommands()):
        yield


def prepare(suite, output, cases=5000, *, cycle_time_mode="log"):
    with configured_runner(cycle_time_mode=cycle_time_mode):
        protocol = runner.prepare(suite, output, cases=cases)
    if protocol.get("frozen_evaluation") or any(
        stage.get("freeze_after") is not None and cases > stage["freeze_after"]
        for stage in protocol["stages"].values()
    ):
        raise ValueError("This experiment measures learning inputs only")
    protocol.update(measured_learning_inputs=[1, cases], measured_frozen_inputs=[],
        compact_context=dict(development_experiment=True,
            setup_descriptor_dim=19, numeric_scaling="fixed parameter range [-1, 1]",
            categorical_encoding="unchanged reference one-hot with conditional masks",
            joint_state_dim=32, fixed_rl_state_dim=32,
            cycle_time_mode=cycle_time_mode,
            cycle_time_formula=TIME_FORMULAS[cycle_time_mode],
            initial_cycle_time_feature=0.0, clipping=False,
            bandit_changed=False, rl_reward="native_cycle", independent_learners=True))
    runner.dump(output/"protocol.json", protocol)
    return protocol


def check_protocol(output):
    metadata = runner.read(output/"protocol.json").get("compact_context", {})
    mode = metadata.get("cycle_time_mode", "log")
    if (mode not in TIME_FORMULAS or metadata.get("cycle_time_formula") != TIME_FORMULAS.get(mode)
            or metadata.get("clipping") is not False
            or metadata.get("joint_state_dim") != 32
            or metadata.get("fixed_rl_state_dim") != 32):
        raise ValueError("Output was not prepared for 32-entry normalized setup context and absolute log time")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--cases", type=int, default=5000)
    parser.add_argument("--label")
    parser.add_argument("--cycle-time", choices=tuple(TIME_FORMULAS), default=None)
    args = parser.parse_args()
    output = args.output.resolve()
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    if args.command == "prepare":
        if args.suite is None:
            parser.error("prepare requires --suite")
        prepare(args.suite.resolve(), output, args.cases, cycle_time_mode=args.cycle_time or "log")
    else:
        if args.command == "worker" and args.label is None:
            parser.error("worker requires --label")
        if args.cycle_time is not None:
            parser.error("Cycle-time context is fixed when the run is prepared")
        check_protocol(output)
        mode = runner.read(output/"protocol.json")["compact_context"].get("cycle_time_mode", "log")
        with configured_runner(cycle_time_mode=mode):
            if args.command == "worker":
                runner.worker(output, args.label)
            else:
                runner.launch(output)


if __name__ == "__main__":
    main()
