"""Test the shared setup-action feature structure with fixed RL normalization."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import signal
import subprocess
from unittest.mock import patch

from experiments.runtime import configure_single_thread
configure_single_thread()

import numpy as np

from development.shared_setup_features import (
    make_shared_setup_action_encoder, make_range_normalized_setup_action_encoder,
)
from experiments.paper_final import run_online as runner
from experiments.paper_final.online import methods


SOURCE_FILES = ("development/shared_setup_features.py", "development/run_shared_setup_features.py",
                "development/setup_normalization.py")
ENCODERS = {"range": make_range_normalized_setup_action_encoder,
            "bandit": make_shared_setup_action_encoder}
OFFICIAL_SOURCE_HASHES = runner.source_hashes
OFFICIAL_INITIALIZE = runner.initialize


def shared_feature_source_hashes():
    hashes = OFFICIAL_SOURCE_HASHES()
    hashes.update({name: runner.file_hash(runner.ROOT/name) for name in SOURCE_FILES})
    return hashes


class WorkerCommands:
    """Reuse the official concurrent seed launcher and readiness barrier."""
    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, argv, *args, **kwargs):
        command = list(argv)
        if command[1:4] == ["-m", "experiments.paper_final.run_online", "worker"]:
            command[2] = "development.run_shared_setup_features"
        return subprocess.Popen(command, *args, **kwargs)


def initialize(output, label, *, normalization):
    runtimes = OFFICIAL_INITIALIZE(output, label)
    setup_models = [runtimes[name].branch.policy.model for name in runner.factorial.METHODS[2:]]
    reference = setup_models[0]
    arms = [reference.actions.default_arm_index, *np.random.default_rng(0).integers(len(reference.actions), size=16)]
    checks = {}
    for name in (runner.factorial.METHODS[1], runner.factorial.METHODS[3]):
        bundle = runtimes[name].bundle
        encoder = bundle.encoder.setup_obs_encoder
        for model in setup_models:
            if encoder.descriptor_dim != model.g_dim:
                raise ValueError("RL setup representation differs from the bandit's action encoder")
            for observed, original in zip(encoder.parameter_spec.parameters, model._encoder.parameter_spec.parameters):
                expected = original
                if normalization == "range" and original.kind in ("continuous", "integer"):
                    values = (*original.values, original.default)
                    low, high = min(values), max(values)
                    expected = replace(original, center=(low+high)/2., scale=(high-low)/2. if high>low else 1.)
                if observed != expected:
                    raise ValueError("RL parameters differ beyond the declared numerical scaling")
            for arm in arms:
                params = model.actions[int(arm)]
                observed = encoder.encode_setup(params)
                baseline = model._feature_for_arm(int(arm))
                if normalization == "bandit":
                    np.testing.assert_array_equal(observed, baseline)
                else:
                    numeric = []
                    for parameter in model._encoder.parameter_spec.parameters:
                        if parameter.kind in ("continuous", "integer"):
                            values = (*parameter.values, parameter.default)
                            low, high = min(values), max(values)
                            numeric.append(2*(params[parameter.name]-low)/(high-low)-1 if high>low else 0.)
                    numeric = np.asarray(numeric)
                    categories = baseline[20:34]
                    expected = np.r_[numeric, numeric**2,
                        [numeric[i]*numeric[j] for i in range(5) for j in range(i+1,5)],
                        categories, (categories[:,None]*numeric).ravel()]
                    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)
                    if np.any(np.abs(observed)>1+1e-12):
                        raise ValueError("Normalized setup features exceed [-1, 1]")
        checks[name] = dict(setup_features=encoder.descriptor_dim, state_features=bundle.encoder.feature_dim,
                            state_action_features=bundle.controller.joint_dim,
                            exact_bandit_features=normalization == "bandit",
                            normalized_range_verified=normalization == "range")
    runner.dump(output/f"seed_{label}"/"shared_setup_feature_audit.json",
                dict(passed=True, checked_configurations=len(arms), controllers=checks,
                     bandit_scaling_unchanged=True, rl_only_rescaling=normalization == "range",
                     normalization=normalization))
    return runtimes


@contextmanager
def configured_runner(normalization="range"):
    if normalization not in ENCODERS:
        raise ValueError("Unknown setup feature normalization")
    def audited_initialize(output, label):
        return initialize(output, label, normalization=normalization)
    with patch.object(runner, "source_hashes", shared_feature_source_hashes), \
         patch.object(methods, "make_setup_feature_encoder", ENCODERS[normalization]), \
         patch.object(runner, "initialize", audited_initialize), \
         patch.object(runner, "subprocess", WorkerCommands()):
        yield


def prepare(suite, output, cases=1000, *, normalization="range"):
    with configured_runner(normalization):
        protocol = runner.prepare(suite, output, cases=cases)
    if protocol.get("frozen_evaluation") or any(
        stage.get("freeze_after") is not None and cases > stage["freeze_after"]
        for stage in protocol["stages"].values()
    ):
        raise ValueError("This diagnostic requires learning inputs only")
    protocol.update(measured_learning_inputs=[1, cases], measured_frozen_inputs=[],
        shared_setup_features=dict(development_experiment=True,
            setup_block="bandit encode_action", applies_to="both RL controllers",
            normalization=normalization,
            scaling="fixed numerical ranges [-1, 1]" if normalization == "range" else "identical to bandit",
            categories="identical to bandit",
            bandit_changed=False, additional_clipping=False, independent_learners=True))
    runner.dump(output/"protocol.json", protocol)
    return protocol


def check_protocol(output):
    metadata = runner.read(output/"protocol.json").get("shared_setup_features", {})
    if metadata.get("setup_block") != "bandit encode_action":
        raise ValueError("Output was not prepared for the shared setup-action feature experiment")
    normalization = metadata.get("normalization", "bandit")
    if normalization not in ENCODERS:
        raise ValueError("Unknown setup feature normalization in saved protocol")
    return normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--normalization", choices=tuple(ENCODERS), default="range",
                        help="Scaling for prepare; run and worker use the saved protocol")
    parser.add_argument("--label")
    args = parser.parse_args()
    output = args.output.resolve()
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    if args.command == "prepare":
        if args.suite is None:
            parser.error("prepare requires --suite")
        prepare(args.suite.resolve(), output, args.cases, normalization=args.normalization)
    else:
        if args.command == "worker" and args.label is None:
            parser.error("worker requires --label")
        normalization = check_protocol(output)
        with configured_runner(normalization):
            if args.command == "worker":
                runner.worker(output, args.label)
            else:
                runner.launch(output)


if __name__ == "__main__":
    main()
