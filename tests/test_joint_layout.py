"""Diagnosis-only regression checks for the setup/joint package boundary."""
from __future__ import annotations

import ast
import importlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import ModuleType
import typing
import unittest

import numpy as np
from setuptools import find_namespace_packages

from joint import execution, recovery, protocol
from setup.learners.linucb import feedback
from experiments.paper_final.common.frozen_inputs import execution_source_paths
from test_construction_failure_feedback import (
    Harness, ROOT, SUCCESS, CONSTRUCTION, NONCONVERGENCE, SOLVE_FAILURE, NONFINITE,
)


BEFORE_COMMIT = 'e55157bf80497edd22efbc522b11803611671644'


def previous_source(relative):
    return subprocess.check_output(['git', 'show', f'{BEFORE_COMMIT}:{relative}'],
                                   cwd=ROOT, text=True)


BEFORE_FEEDBACK = previous_source('experiments/paper_final/recovery/setup_feedback.py')
BEFORE_EXECUTION = previous_source('experiments/paper_final/recovery/execution.py')


def definition(source, name):
    return next(node for node in ast.parse(source).body if getattr(node, 'name', None) == name)


class JointLayoutTests(unittest.TestCase):
    def test_moved_models_recorders_and_protocol_are_unchanged(self):
        comparisons = [
            (BEFORE_FEEDBACK, Path(feedback.__file__).read_text(),
             ('DeferredCost', 'DeferredCostLinUCB', 'FailureAwareRetryLinUCB')),
            (BEFORE_FEEDBACK, Path(recovery.__file__).read_text(), ('SetupReselectionResult',)),
            (BEFORE_EXECUTION, Path(execution.__file__).read_text(),
             ('RecordedEpisode', 'EpisodeRecorder', 'run_fixed_case')),
            (BEFORE_EXECUTION, (ROOT/'experiments/paper_final/recovery/execution.py').read_text(),
             ('CandidateRows',)),
            (previous_source('experiments/paper_final/recovery/protocol.py'),
             Path(protocol.__file__).read_text(), ('RecoveryVariant',)),
        ]
        for before, after, names in comparisons:
            for name in names:
                with self.subTest(name=name):
                    self.assertEqual(ast.dump(definition(before, name)),
                                     ast.dump(definition(after, name)))
        typing.get_type_hints(execution.RecordedEpisode)

    def test_core_packages_do_not_depend_on_experiment_code(self):
        paths = list((ROOT/'joint').glob('*.py'))+[Path(feedback.__file__)]
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                modules = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                           else [item.name for item in node.names] if isinstance(node, ast.Import)
                           else [])
                self.assertFalse(any(name == 'experiments' or name.startswith('experiments.')
                                     for name in modules), str(path))
        subprocess.run([sys.executable, '-c',
                        'import sys, joint.recovery, joint.execution; '
                        'assert not any(n == "experiments" or n.startswith("experiments.") for n in sys.modules)'],
                       cwd=ROOT, check=True, capture_output=True, text=True)

    def test_legacy_imports_are_aliases_of_the_new_implementations(self):
        old_feedback = importlib.import_module('experiments.paper_final.recovery.setup_feedback')
        old_execution = importlib.import_module('experiments.paper_final.recovery.execution')
        old_protocol = importlib.import_module('experiments.paper_final.recovery.protocol')
        for name in ('DeferredCost', 'DeferredCostLinUCB', 'FailureAwareRetryLinUCB'):
            self.assertIs(getattr(old_feedback, name), getattr(feedback, name))
        self.assertIs(old_feedback.run_deferred_case, recovery.run_deferred_case)
        self.assertIs(old_feedback.SetupReselectionResult, recovery.SetupReselectionResult)
        for name in ('RecordedEpisode', 'EpisodeRecorder', 'run_fixed_case'):
            self.assertIs(getattr(old_execution, name), getattr(execution, name))
        self.assertIs(old_protocol.RecoveryVariant, protocol.RecoveryVariant)

    def test_unchanged_attempt_sequences_match_before_move(self):
        failures = (CONSTRUCTION, NONCONVERGENCE, SOLVE_FAILURE, NONFINITE)
        cases = []
        for length in (1, 2, 3):
            cases.extend((prefix+(SUCCESS,), None)
                         for prefix in itertools.product(failures, repeat=length-1))
        cases.extend((statuses, fallback)
                     for statuses in itertools.product(failures, repeat=3)
                     for fallback in (SUCCESS, *failures))
        self.assertEqual(len(cases), 341)
        comparisons = 0
        for mode in ('adjacent_delayed', 'suffix_delayed'):
            for learn in (True, False):
                for statuses, fallback in cases:
                    # A later correction extends any failed attempt's target across
                    # subsequent construction failures. Those changed targets are
                    # checked independently in ConstructionFeedbackTests.
                    if (learn and mode == 'adjacent_delayed'
                            and any(left != CONSTRUCTION and right == CONSTRUCTION
                                    for left, right in zip(statuses, statuses[1:]))):
                        continue
                    with self.subTest(mode=mode, learn=learn, statuses=statuses, fallback=fallback):
                        old = Harness(BEFORE_FEEDBACK, learn=learn, mode=mode)
                        new = Harness(learn=learn, mode=mode)
                        before, _ = old.execute(statuses, fallback)
                        after, _ = new.execute(statuses, fallback)
                        self.assertEqual(json.dumps(before.outcome, sort_keys=True),
                                         json.dumps(after.outcome, sort_keys=True))
                        self.assertEqual(before.timing, after.timing)
                        self.assertEqual(old.trace, new.trace)
                        for left, right in zip(old.state(), new.state()):
                            if isinstance(left, np.ndarray):
                                np.testing.assert_array_equal(left, right)
                            else:
                                self.assertEqual(left, right)
                        comparisons += 1
        self.assertEqual(comparisons, 1241)

    def test_previous_checkpoint_loads_without_format_changes(self):
        old_module = ModuleType('_before_joint_layout')
        sys.modules[old_module.__name__] = old_module
        try:
            exec(compile(BEFORE_FEEDBACK, '<before-joint-layout>', 'exec'), vars(old_module))
            template = Harness()
            base = old_module.SharedLinUCB(template.model.actions, 1,
                parameter_spec=template.model.parameter_spec, context_interaction_indices=(), seed=17)
            old_model = old_module.FailureAwareRetryLinUCB.from_runtime_model(base)
            old_model.predict(template.context)
            observation = old_model.defer_cost(False)
            old_model.resolve_cost(observation, 2.5)
            with tempfile.TemporaryDirectory() as directory:
                old_path = Path(directory)/'old.npz'
                new_path = Path(directory)/'new.npz'
                old_model.save_mutable_state(old_path, metadata={'purpose': 'layout regression'})
                new_model = Harness().model
                metadata = new_model.load_mutable_state(old_path)
                new_model.save_mutable_state(new_path, metadata=metadata)
                with np.load(old_path, allow_pickle=False) as before, np.load(new_path, allow_pickle=False) as after:
                    self.assertEqual(set(before.files), set(after.files))
                    for name in before.files:
                        np.testing.assert_array_equal(before[name], after[name])
        finally:
            sys.modules.pop(old_module.__name__, None)

    def test_packaging_and_source_snapshots_include_joint(self):
        required = {'joint/__init__.py', 'joint/recovery.py', 'joint/execution.py',
                    'joint/protocol.py', 'setup/learners/linucb/feedback.py'}
        self.assertTrue(required <= execution_source_paths())
        text = (ROOT/'pyproject.toml').read_text()
        include = ast.literal_eval(re.search(r'^include = (.+)$', text, re.MULTILINE).group(1))
        self.assertIn('joint', find_namespace_packages(where=str(ROOT), include=include))


if __name__ == '__main__':
    unittest.main(verbosity=2)
