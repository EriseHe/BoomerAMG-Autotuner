"""Exercise Module 2 with synthetic source models, without choosing a real run."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.paper_final.common import artifacts, frozen_policy
from experiments.paper_final.matched_policy import planning, workflow
from experiments.paper_final.online.configuration import parse_joint_experiment_config
from experiments.paper_final.online.setup_branches import build_online_linucb_branch
from experiments.paper_final.run_policy import main as policy_main
from problems.registry import learning_context_for_setup
from setup.space import SetupConfigurationSpace


ROOT = Path(__file__).resolve().parents[2]


class ModuleTwoReadyTests(unittest.TestCase):
    def test_active_online_configs_use_shared_context(self):
        paths = list((ROOT/'configs').rglob('seed_*.json'))
        self.assertEqual(len(paths), 42)
        for path in paths:
            spec = parse_joint_experiment_config(artifacts.read(path))
            self.assertEqual(spec.solve.state.encoding, 'shared_context')
            if 'module1' in path.parts:
                self.assertEqual(spec.frozen_stream.online_cases, 1000)

    def test_prepare_scan_and_evaluate_accept_current_models(self):
        raw = artifacts.read(ROOT/'configs/module1/diffusion_60/seed_1.json')
        raw.pop('frozen_stream')
        raw['methods'] = [m for m in raw['methods'] if m['id']=='bandit_lstdq']
        spec = parse_joint_experiment_config(raw)
        method = spec.methods[0]
        context = learning_context_for_setup(spec.problem.kind, method.setup_kind, method.setup_context)
        space = SetupConfigurationSpace.from_mapping(method.setup_space,
                    raw['setup']['configuration_spaces'][method.setup_space])
        with TemporaryDirectory(dir=ROOT/'results', prefix='module2_readiness_') as directory:
            root = Path(directory)
            training = root/'synthetic_source'
            (training/'checkpoints').mkdir(parents=True)
            (training/'final_bandit_states').mkdir()
            artifacts.dump(training/'experiment_config.json', raw)
            bundle = frozen_policy.build_bundle(raw, method.name)
            bundle.save(training/'checkpoints/bandit_lstdq_final.npz')
            branch, _ = build_online_linucb_branch(seed=7,
                action_space_mode=raw['setup']['action_space'], parameter_resolution=20,
                configuration_space=space, solver_tol=1e-6, solver_max_iter=50,
                candidate_schedule_dir=training/'candidates', candidate_schedule_rounds=2,
                candidate_pool_size=spec.setup.candidate_schedule.pool_size,
                structured_quotas=spec.setup.candidate_schedule.structured_quotas,
                candidate_sampling=method.candidate_sampling, context_dim=context.dimension,
                context_interaction_indices=context.interaction_indices)
            branch.policy.model.save_mutable_state(training/'final_bandit_states/bandit_lstdq.npz')
            config = dict(schema_version=1, name='Module 2 readiness validation',
                training_runs=[dict(seed=1, run_dir=str(training))], cases=2,
                problem=dict(raw['problem'],grid=[12,12,12]), weights=[1.,2.],
                scan_repetitions=1, scan_extra_repetitions=0, scan_repeat_fraction=0.,
                evaluation_repetitions=1, workers=1, seeds=dict(inputs=31,candidates=32,order=33,repeats=34))
            artifacts.dump(root/'configuration.json',config)
            validation = StringIO()
            with patch('sys.argv', ['run_policy','validate','--config',str(root/'configuration.json')]), redirect_stdout(validation):
                policy_main()
            validated = json.loads(validation.getvalue())
            self.assertEqual(validated['native_solves_executed'],0)
            self.assertEqual(validated['controllers'][0]['state_encoding'],'shared_context')
            output = root/'run'
            protocol = planning.prepare(root/'configuration.json',output)
            self.assertEqual(protocol['module'],2)
            self.assertEqual(protocol['format'],'matched_policy')
            self.assertEqual(artifacts.read(output/'status.json')['native_solves_executed'],0)
            with patch('experiments.paper_final.plot_policy.generate'):
                workflow.run(output)
            self.assertEqual(artifacts.read(output/'status.json')['status'],'complete')
            summary = artifacts.read(output/'analysis/summary.json')
            self.assertTrue(summary['audit']['exact_coverage'])
            self.assertEqual(summary['audit']['fresh_scan_trials'],4)
            from experiments.paper_final.plot_policy import read_study
            self.assertEqual(read_study(output)[0]['protocol']['format'],'matched_policy')
            for phase in ('scan','test'):
                for path in (output/f'raw/{phase}').glob('worker_*.jsonl'):
                    for row in artifacts.read_records(path):
                        self.assertEqual(row['run_number'],2)
                        if row['kind']=='rl':
                            out=row['outcome']
                            self.assertFalse(out.get('controller_update_committed',False))
                            self.assertGreaterEqual(row['inclusive_continuation_sec'],row['native_continuation_sec'])
