"""Check the online/frozen boundary and preserve the archived frozen study."""
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.paper_final import run_online


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'data/module1/frozen'


def read(path):
    return json.loads(path.read_text())


class ModuleOneFrozenTests(unittest.TestCase):
    def test_protocol_has_5000_learning_and_1000_frozen_inputs(self):
        protocol = read(ROOT / 'data/module1/online/protocol.json')
        self.assertEqual(protocol['cases_per_seed'], 6000)
        self.assertEqual(protocol['training_cases_per_seed'], 5000)
        self.assertEqual(protocol['frozen_cases_per_seed'], 1000)
        phases = protocol['phases']
        self.assertEqual(phases['online']['global_cases'], [1, 5000])
        self.assertEqual(phases['frozen']['global_cases'], [5001, 6000])
        self.assertTrue(phases['online']['model_updates'])
        self.assertFalse(phases['frozen']['model_updates'])
        self.assertFalse(phases['frozen']['rl_epsilon_exploration'])
        self.assertFalse((ROOT / 'data/module4').exists())

    def test_frozen_settings_and_checkpoints_match_each_training_run(self):
        configurations = list((FROZEN / 'configurations').rglob('*.json'))
        self.assertEqual(len(configurations), 36)
        for path in configurations:
            with self.subTest(path=str(path.relative_to(ROOT))):
                frozen = read(path)
                evaluation = frozen['evaluation']
                training = read(ROOT / evaluation['training_configuration'])
                for key in ('problem', 'setup', 'solve', 'methods', 'seeds', 'failure_feedback'):
                    self.assertEqual(frozen[key], training[key])
                self.assertEqual(evaluation['training_cases'], 5000)
                self.assertFalse(evaluation['explore'])
                self.assertEqual(frozen['stream']['online_cases'], 1000)
                self.assertEqual(frozen['stream']['instance_offset'], 625)
                self.assertEqual(frozen['stream']['cases_per_seed'], 125)
                self.assertEqual(Path(evaluation['checkpoint_directory']).parts[-2:],
                                 (path.parent.name, path.stem))

    def test_frozen_input_indices_continue_at_5001(self):
        runtime = SimpleNamespace(
            args=SimpleNamespace(evaluation=SimpleNamespace(training_cases=5000)),
            stream=[({'instance': index}, np.zeros(8)) for index in range(1000)],
        )
        rows = run_online.input_rows(runtime, 1000)
        self.assertEqual([r['problem_index'] for r in rows], list(range(5001, 6001)))
        self.assertEqual(rows[0]['mkw'], {'instance': 0})
        self.assertEqual(rows[-1]['mkw'], {'instance': 999})

    def test_frozen_inputs_and_measurements_are_unchanged_by_reorganization(self):
        record = read(ROOT / 'development/validation/module1_frozen_migration.json')
        for name, expected in record['unchanged_files'].items():
            actual = hashlib.sha256((FROZEN / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)
        summary = read(FROZEN / 'summary.json')
        self.assertEqual((summary['module'], summary['phase']), (1, 'frozen'))
        for key, expected in record['unchanged_summary_content'].items():
            payload = json.dumps(summary[key], sort_keys=True, separators=(',', ':')).encode()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected, key)

    def test_cli_exposes_frozen_phase_without_a_separate_module(self):
        output = StringIO()
        with patch('sys.argv', ['run_online', '--help']), redirect_stdout(output):
            with self.assertRaises(SystemExit) as caught:
                run_online.main()
        self.assertEqual(caught.exception.code, 0)
        self.assertIn('prepare-frozen', output.getvalue())
        self.assertNotIn('prepare-module4', output.getvalue())

    def test_new_frozen_artifacts_refresh_both_manifests(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            phase = root / 'data/module1/frozen'
            phase.mkdir(parents=True)
            report = phase / 'summary.json'
            report.write_text('{"passed":true}')
            inventory = root / 'data/manifest.json'
            inventory.write_text(json.dumps({'modules': {'module1': {}}, 'files': {
                'data/module1/protocol.json': 'preserved',
                'data/module1/frozen/removed.json': 'obsolete',
            }}))
            with patch.object(run_online, 'ROOT', root):
                run_online.write_frozen_manifest()
                report.write_text('{"passed":true,"updated":true}')
                run_online.write_frozen_manifest()
            expected = hashlib.sha256(report.read_bytes()).hexdigest()
            files = read(inventory)['files']
            self.assertEqual(files['data/module1/frozen/summary.json'], expected)
            self.assertEqual(read(phase / 'manifest.json')['data/module1/frozen/summary.json'], expected)
            self.assertEqual(files['data/module1/protocol.json'], 'preserved')
            self.assertNotIn('data/module1/frozen/removed.json', files)


if __name__ == '__main__':
    unittest.main()
