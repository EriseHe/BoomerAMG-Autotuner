"""Check solve-basis isolation, checkpoint identity and two-branch native runs."""
import itertools
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from development.cycle_bins import CycleBinSolveStateEncoder, CYCLE_BOUNDS
from development.quadratic_context import QuadraticSolveStateEncoder
from development.run_joint_context import build_solve_runtime, configured_runner, prepare, runner, worker, analyze, METHODS
from development.tests.test_compact_context import encoder, observation

ROOT=Path(__file__).resolve().parents[2]


def config(problem='advection_40'):
    raw=runner.read(ROOT/f'configs/module1/{problem}/seed_1.json')
    raw.pop('frozen_stream')
    raw['methods']=[m for m in raw['methods'] if m['id'] in METHODS]
    return raw


class JointContextTests(unittest.TestCase):
    def test_bins_preserve_every_current_feature_and_uncapped_gap(self):
        base=encoder()
        binned=CycleBinSolveStateEncoder(tol=base.tol,max_cycles=base.max_cycles,c_max=base.c_max,
            mode=base.mode,setup_obs_encoder=base.setup_obs_encoder,
            problem_context_mode=base.problem_context_mode,weight_bounds=base.weight_bounds)
        for cycle,residual,seconds in itertools.product(range(51),(1e-8,.001,1.,1e8),(.001,.1)):
            kwargs=observation(cycle=cycle,residual=residual,last_cycle_time=seconds)
            a,b=base.encode(**kwargs),binned.encode(**kwargs)
            np.testing.assert_array_equal(a,np.r_[b[:13],b[27:]])
            indicators=np.zeros(7);indicators[sum(cycle>=edge for edge in CYCLE_BOUNDS)]=1.
            np.testing.assert_array_equal(b[13:20],indicators)
            np.testing.assert_array_equal(b[20:27],a[2]*indicators)
            self.assertEqual(b.size,46)
        self.assertGreater(binned.encode(**observation(residual=1e8))[2],1.5)
        self.assertAlmostEqual(binned.encode(**observation(last_cycle_time=.1))[5],math.log1p(100))
        np.testing.assert_array_equal(binned.constant_value_parameters(4.),np.r_[4.,np.zeros(45)])

    def test_exact_roster_and_checkpoint_identity(self):
        raw=config();spec=runner.parse_joint_experiment_config(raw);args=runner.runtime_config_from_spec(spec)
        bundles={}
        for variant,dimensions in (('log_time',(32,288)),('cycle_bins',(46,414)),('quadratic',(34,306))):
            built=build_solve_runtime(args,specs=spec.methods,context_variant=variant).controller_bundles
            self.assertEqual(set(built),{'bandit_lstdq'})
            bundle=built['bandit_lstdq'];bundles[variant]=bundle
            self.assertEqual((bundle.encoder.feature_dim,bundle.controller.joint_dim),dimensions)
            self.assertEqual(bundle.encoder.setup_obs_encoder.descriptor_dim,19)
        with TemporaryDirectory() as directory:
            from development.run_compact_context import build_solve_runtime as plain
            unbinned=plain(args,specs=spec.methods).controller_bundles['bandit_lstdq']
            for variant,bundle in bundles.items():
                path=Path(directory)/f'{variant}.npz'
                bundle.save(path);bundle.load(path)
                if variant=='log_time':
                    unbinned.load(path)
                    self.assertEqual(bundle.encoder.metadata(),unbinned.encoder.metadata())
                    for cycle,residual,seconds in itertools.product(range(51),(1e-8,.001,1.,1e8),(.001,.1)):
                        kwargs=observation(cycle=cycle,residual=residual,last_cycle_time=seconds)
                        np.testing.assert_array_equal(bundle.encoder.encode(**kwargs),unbinned.encoder.encode(**kwargs))
                else:
                    with self.assertRaises(ValueError):unbinned.load(path)
                for other_variant,other in bundles.items():
                    if other_variant!=variant:
                        with self.assertRaises(ValueError):other.load(path)

    def test_quadratic_preserves_base_and_completes_progress_gap_basis(self):
        base=encoder()
        quadratic=QuadraticSolveStateEncoder(tol=base.tol,max_cycles=base.max_cycles,c_max=base.c_max,
            mode=base.mode,setup_obs_encoder=base.setup_obs_encoder,
            problem_context_mode=base.problem_context_mode,weight_bounds=base.weight_bounds)
        for cycle,residual,initial,seconds in itertools.product(range(51),(1e-8,.001,1.,1e8),(.1,10.),(.001,.1)):
            kwargs=observation(cycle=cycle,residual=residual,initial_residual=initial,last_cycle_time=seconds)
            original,actual=base.encode(**kwargs),quadratic.encode(**kwargs)
            np.testing.assert_array_equal(original,np.r_[actual[:13],actual[15:]])
            u=cycle/50.
            g=math.log(residual/1e-6)/math.log(initial/1e-6)
            self.assertAlmostEqual(actual[13],u**2)
            self.assertAlmostEqual(actual[14],u*g)
            # An arbitrary quadratic in the two physical quantities must be
            # represented exactly, including its mixed term and both squares.
            reconstructed=np.dot(actual[[0,1,2,13,14,3]],[2.,3.,5.,7.,11.,13.])
            expected=2.+3.*u+5.*g+7.*u**2+11.*u*g+13.*g**2
            self.assertAlmostEqual(reconstructed,expected)
            self.assertEqual(actual.size,34)
        self.assertGreater(quadratic.encode(**observation(residual=1e8))[2],1.5)
        self.assertLess(quadratic.encode(**observation(residual=1e-8))[2],0.)
        self.assertEqual(quadratic.encode(**observation(cycle=0,first_cycle_time=None))[5],0.)
        np.testing.assert_array_equal(quadratic.constant_value_parameters(4.),np.r_[4.,np.zeros(33)])

    def test_native_two_branch_worker_and_final_audits(self):
        for variant in ('log_time','cycle_bins','quadratic'):
            with self.subTest(variant=variant):
                self._check_native_worker(variant)

    def test_diffusion_log_time_preserves_configured_context(self):
        self._check_native_worker('log_time', problem='diffusion_40')

    def _check_native_worker(self, variant, problem='advection_40'):
        raw=config(problem)
        with TemporaryDirectory(dir=ROOT/'results',prefix='joint_context_validation_') as directory:
            base=Path(directory)
            runner.dump(base/'config.json',raw)
            runner.dump(base/'suite.json',dict(name='Joint and Default native validation',
                seeds=[dict(label='1',config=str(base/'config.json'))]))
            output=base/'run'
            prepare(base/'suite.json',output,cases=5,context_variant=variant)
            runner.dump(output/'release.json',dict(validation_only=True))
            with configured_runner():
                worker(output,'1')
                analyze(output)
            self.assertEqual(runner.read(output/'completion_audit.json')['measured_method_cases'],10)
            self.assertEqual(set(runner.read(output/'seed_1/result.json')['methods']),set(METHODS))
            self.assertEqual({p.stem for p in (output/'seed_1/trajectories').glob('*.jsonl')},set(METHODS))
            joint=list(runner.read_records(output/'seed_1/trajectories/bandit_lstdq.jsonl'))
            self.assertTrue(all(row['outcome']['end_to_end_runtime']>row['outcome']['runtime'] for row in joint))
            metadata=runner.read(output/'seed_1/initial_controllers/bandit_lstdq.npz.encoder.json')
            if variant=='cycle_bins':
                self.assertEqual(metadata['feature_dim'],46)
                self.assertEqual(metadata['cycle_bin_bounds'],list(CYCLE_BOUNDS))
            elif variant=='quadratic':
                self.assertEqual(metadata['feature_dim'],34)
                self.assertEqual(metadata['quadratic_degree'],2)
                self.assertFalse(metadata['cycle_bins'])
                self.assertEqual(len(metadata['additional_state_terms']),2)
            else:
                problem_entries=3 if problem=='diffusion_40' else 6
                self.assertEqual(metadata['feature_dim'],26+problem_entries)
                self.assertEqual(metadata['cycle_time_mode'],'log')
                context=runner.read(output/'protocol.json')['solve_context']
                self.assertEqual(context['problem_entries'],problem_entries)
                self.assertEqual(context['state_action_entries'],9*(26+problem_entries))
                self.assertFalse(context['cycle_bins'])
                self.assertEqual(context['additional_state_terms'],[])
            self.assertFalse(metadata['cycle_time_clipping'])
            self.assertEqual(runner.read(output/'protocol.json')['solve_context']['variant'],variant)
            self.assertFalse((output/'seed_1/freeze_boundary.json').exists())


if __name__=='__main__':unittest.main()
