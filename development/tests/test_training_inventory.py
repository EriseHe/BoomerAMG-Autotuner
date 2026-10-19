"""Fresh frozen stages must consume complete, matching training artifacts."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.paper_final import run_online as runner


class TrainingInventoryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(TemporaryDirectory())).resolve()
        self.source = self.root / 'results/training/seed_1'
        self.output = self.source.parent
        self.stack.enter_context(patch.object(runner, 'ROOT', self.root))
        self.stack.enter_context(patch.object(runner, 'source_hashes', return_value={'code': 'source'}))
        self.stack.enter_context(patch.object(runner.sweep, 'native_hashes', return_value={'native': 'library'}))
        runner.dump(self.source / 'experiment_config.json', {'setup': 'matching'})
        (self.source / 'inputs.jsonl').write_text('{"input":1}\n')
        self.protocol = dict(frozen_evaluation=False, native_sha256={'native': 'library'},
            stages={'1': dict(cases=5000, config_sha256=runner.file_hash(self.source / 'experiment_config.json'),
                inputs_sha256=runner.digest([{'input': 1}]))})
        runner.dump(self.output / 'protocol.json', self.protocol)
        runner.dump(self.output / 'source_manifest.json', {'code': 'source'})
        runner.dump(self.output / 'status.json', {'status': 'complete'})
        runner.dump(self.source / 'status.json', {'status': 'complete'})
        runner.dump(self.source / 'result.json', dict(complete=True, passed=True, cases=5000,
            checkpoint_sha256={'checkpoints/final.npz': 'checkpoint-hash'},
            audits={name: dict(cases=5000, trajectory_sha256=name+'-hash') for name in runner.factorial.METHODS}))

    def test_new_run_uses_audited_checkpoint_and_trajectory_hashes(self):
        inventory = runner.training_inventory(self.source)
        self.assertEqual(inventory['results/training/seed_1/checkpoints/final.npz']['sha256'], 'checkpoint-hash')
        for name in runner.factorial.METHODS:
            self.assertEqual(inventory[f'results/training/seed_1/trajectories/{name}.jsonl']['sha256'], name+'-hash')

    def test_incomplete_or_already_frozen_training_is_rejected(self):
        for changes in ({'frozen_evaluation': True}, {'stages': {'1': dict(self.protocol['stages']['1'], cases=1000)}}):
            runner.dump(self.output / 'protocol.json', dict(self.protocol, **changes))
            with self.assertRaisesRegex(ValueError, 'complete audited'):
                runner.training_inventory(self.source)
        runner.dump(self.output / 'protocol.json', self.protocol)
        runner.dump(self.output / 'status.json', {'status': 'running'})
        with self.assertRaisesRegex(ValueError, 'complete audited'):
            runner.training_inventory(self.source)

    def test_changed_training_source_and_input_are_rejected(self):
        runner.dump(self.output / 'source_manifest.json', {'code': 'different'})
        with self.assertRaisesRegex(ValueError, 'training source'):
            runner.training_inventory(self.source)
        runner.dump(self.output / 'source_manifest.json', {'code': 'source'})
        (self.source / 'inputs.jsonl').write_text('{"input":2}\n')
        with self.assertRaisesRegex(ValueError, 'Training inputs changed'):
            runner.training_inventory(self.source)

    def test_released_training_still_uses_original_inventory(self):
        runner.dump(self.root / 'data/raw_files.json', {'files': {'released-file': {'sha256': 'pinned'}}})
        self.assertEqual(runner.training_inventory(self.root / 'data/official/module1/seed_1'),
                         {'released-file': {'sha256': 'pinned'}})
