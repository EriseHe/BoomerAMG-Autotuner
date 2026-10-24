"""All online methods use the same Default fallback settings and execution."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.paper_final.online import factorial, native_case
from experiments.paper_final.recovery import default_fallback, joint_validation
from hypre.bindings import augment_setup_params
from setup.space import DEFAULT_SETUP_PARAMS


class FallbackExecuted(Exception):
    """Stop a caller after checking its actual fallback callback."""


class DefaultFallbackTests(unittest.TestCase):
    def test_all_methods_dispatch_the_same_default_fallback(self):
        inputs = dict(nx=12, ny=12, nz=12)
        args = SimpleNamespace(tol=1e-7, max_cycles=37, evaluation=None, problem="diffusion")
        controller = SimpleNamespace(snapshot_learning_state=lambda: {})
        bundle = SimpleNamespace(controller=controller)
        runtime = SimpleNamespace(args=args, stream=[(inputs, np.ones(2))], bundle=bundle,
            method=SimpleNamespace(setup_kind="linucb", setup_context="diffusion"),
            branch=SimpleNamespace(parameter_space={}, policy=SimpleNamespace(
                model=SimpleNamespace(selection_diagnostics=[]))))
        completed = []

        def execute_fallback(*unused, fallback_solver, **kwargs):
            self.assertIs(fallback_solver.func, default_fallback.run_default_fallback)
            completed.append(fallback_solver())
            raise FallbackExecuted

        def execute_episode(case):
            completed.append(case.fallback_attempt())
            raise FallbackExecuted

        result = dict(runtime=.2, failed=False)
        with patch.object(default_fallback, "solve_no_rl_case", return_value=result) as solve:
            with patch.object(factorial, "run_fixed_case", side_effect=execute_fallback):
                with self.assertRaises(FallbackExecuted):
                    factorial.execute_fixed_rl(runtime, 0)
            with patch.object(joint_validation, "run_deferred_case", side_effect=execute_fallback), \
                 patch.object(joint_validation, "context_for_setup_method", return_value=np.ones(2)):
                for rl_active in (False, True):
                    runtime.bundle = bundle if rl_active else None
                    with self.assertRaises(FallbackExecuted):
                        joint_validation.execute_learned(runtime, 0, 0.,
                            feedback_mode="adjacent_delayed", activation_case=0)
            with self.assertRaises(FallbackExecuted):
                native_case.run_case(bundle=SimpleNamespace(run_case=execute_episode),
                    setup_row=dict(mkw=inputs, params=DEFAULT_SETUP_PARAMS), args=args,
                    learn=True, explore=True)
            self.assertEqual(len(completed), 4)
            self.assertEqual(solve.call_count, 4)
            for call in solve.call_args_list:
                self.assertEqual(call.kwargs, dict(params=DEFAULT_SETUP_PARAMS, mkw=inputs,
                    solver_tol=1e-7, solver_max_iter=37, augment_params=augment_setup_params))

    def test_each_fallback_receives_fresh_parameter_and_input_dictionaries(self):
        inputs = dict(nx=12)
        expected = dict(DEFAULT_SETUP_PARAMS)
        def solve(**kwargs):
            self.assertEqual(kwargs["params"], expected)
            self.assertEqual(kwargs["mkw"], inputs)
            kwargs["params"]["strong_threshold"] = -1
            kwargs["mkw"]["nx"] = -1
            return dict(runtime=.2, failed=False)
        with patch.object(default_fallback, "solve_no_rl_case", side_effect=solve):
            for _ in range(2):
                default_fallback.run_default_fallback(mkw=inputs, solver_tol=1e-6, solver_max_iter=50)
        self.assertEqual(DEFAULT_SETUP_PARAMS, expected)
        self.assertEqual(inputs, dict(nx=12))


if __name__ == "__main__":
    unittest.main()
