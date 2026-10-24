"""Check optimized uncertainty against the full dense LSTDQ expression."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.paper_final.online.controllers import make_setup_feature_encoder
from setup.learners.common.aot_candidates import CompactActionCatalog
from solve.controllers.common import SharedActionSpec, SolveStateSpec
from solve.controllers.common.linear_lcb import _greedy_cost_index
from solve.controllers.recursive_lstdq import RecursiveLstdqController, RecursiveLstdqSpec


def controller(feature_dim):
    return RecursiveLstdqController(
        feature_dim=feature_dim, seed=4,
        config=SharedActionSpec(
            weights=tuple(np.linspace(1., 3., 41)),
            rbf_centers=tuple(np.linspace(1., 3., 9)),
        ).to_td_config(trace_lambda=.8),
        spec=RecursiveLstdqSpec(),
    )


class ScoringEquivalenceTests(unittest.TestCase):
    def setUp(self):
        shared = make_setup_feature_encoder(parameter_resolution=20)
        self.catalog = CompactActionCatalog(shared.parameter_spec)
        self.encoder = SolveStateSpec(
            tol=1e-6, max_cycles=50, c_max=1000., problem_context_mode='diffusion3d',
        ).build_encoder(setup_obs_encoder=shared, weights=(1., 3.))

    def state(self, rng, cycle):
        coefficients = 10. ** rng.uniform(0., 3., size=3)
        return self.encoder.encode(
            mkw=dict(nx=40, ny=40, nz=40, k=coefficients[0],
                     c=coefficients[1], a0=coefficients[2]),
            setup_params=self.catalog[int(rng.integers(len(self.catalog)))],
            initial_residual=1., residual=10. ** (-.5 * cycle),
            previous_residual=10. ** (-.5 * max(0, cycle - 1)),
            cycle=cycle, last_weight=float(rng.uniform(1., 3.)),
            last_cycle_time=float(rng.uniform(0., .03)),
        )

    def assert_dense_scores(self, model, features):
        joint = model.state_action_features(features)
        projected = joint @ model.a_inverse
        means = joint @ model.theta
        quadratic = np.sum(
            (projected @ model.episode_moment_covariance) * projected, axis=1)
        uncertainty = np.sqrt(np.maximum(quadratic, 0.))
        scores = means - model.spec.uncertainty_beta * uncertainty
        if model.spec.lcb_lower_bound_sec is not None:
            scores = np.maximum(scores, model.spec.lcb_lower_bound_sec)
        actual = model._values(features, cycle=0)
        for value, expected in zip(actual, (means, uncertainty, scores)):
            np.testing.assert_allclose(value, expected, rtol=2e-9, atol=2e-11)
        expected_action = _greedy_cost_index(
            scores, allowed_indices=model._all_action_indices,
            anchor_index=model.anchor_index, secondary_values=means)
        selected, _, _ = model.select_action(features, explore=False)
        self.assertEqual(selected, expected_action)

    def test_scores_match_dense_formula_through_updates_and_episode_commits(self):
        model = controller(self.encoder.feature_dim)
        rng = np.random.default_rng(13)
        for episode in range(24):
            model.start_episode()
            current = self.state(rng, 0)
            for cycle in range(9):
                following = self.state(rng, cycle + 1)
                self.assert_dense_scores(model, current)
                model.update(
                    features=current, action_index=(episode + cycle) % 41,
                    cost=.002 + .0001 * cycle, next_features=following,
                    terminal=cycle == 8, next_cycle=cycle + 1,
                    next_action_index=(episode + cycle + 1) % 41,
                )
                current = following
            model.finish_episode(learned=True)
            self.assert_dense_scores(model, current)

    def test_snapshot_restore_and_checkpoint_rebuild_derived_cache(self):
        model = controller(self.encoder.feature_dim)
        rng = np.random.default_rng(9)
        current, following = self.state(rng, 0), self.state(rng, 1)
        model.start_episode()
        model.update(features=current, action_index=5, cost=.003,
                     next_features=following, terminal=True)
        model.finish_episode(learned=True)
        snapshot = model.snapshot_learning_state()
        self.assertTrue(snapshot['a_inverse'].flags.f_contiguous)
        original = tuple(a.copy() for a in model._values(current, cycle=0))
        model.start_episode()
        model.update(features=following, action_index=7, cost=.01,
                     next_features=current, terminal=True)
        model.finish_episode(learned=True)
        model.restore_learning_state(snapshot)
        for value, expected in zip(model._values(current, cycle=0), original):
            np.testing.assert_allclose(value, expected, rtol=2e-9, atol=2e-11)
        self.assert_dense_scores(model, current)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'controller.npz'
            model.save(path)
            restored = controller(self.encoder.feature_dim)
            restored.load(path)
            self.assert_dense_scores(restored, current)
            np.testing.assert_array_equal(restored.theta, model.theta)

    def test_nonsymmetric_inverse_and_inverse_rebuild(self):
        model = controller(self.encoder.feature_dim)
        rng = np.random.default_rng(17)
        model.a_matrix[:] = np.eye(model.joint_dim) + rng.normal(
            0., .005, (model.joint_dim, model.joint_dim))
        model.a_inverse[:] = np.linalg.inv(model.a_matrix)
        moment = rng.normal(0., .001, (model.joint_dim, 20))
        model.episode_moment_covariance += moment @ moment.T
        model._refresh_inverse_moment()
        current, following = self.state(rng, 0), self.state(rng, 1)
        self.assertFalse(np.allclose(model.a_inverse, model.a_inverse.T))
        self.assert_dense_scores(model, current)
        model.start_episode()
        model.inverse_is_valid = False
        model.update(features=current, action_index=2, cost=.004,
                     next_features=following, terminal=True)
        model.finish_episode(learned=True)
        self.assertEqual(model.inverse_rebuild_count, 1)
        self.assert_dense_scores(model, following)


if __name__ == '__main__':
    unittest.main()
