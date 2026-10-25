"""Verify actual worker scheduling and portable scheduling failure behavior."""
import errno
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments import runtime


ROOT = Path(__file__).resolve().parents[2]


class RuntimeSchedulingTests(unittest.TestCase):
    def test_other_platforms_do_not_load_macos_library(self):
        for platform in ('linux', 'win32'):
            with self.subTest(platform=platform), patch.object(sys, 'platform', platform), \
                 patch.object(runtime.ctypes, 'CDLL') as library:
                result = runtime.request_performance_qos()
                self.assertFalse(result['requested'])
                self.assertFalse(result['hard_affinity'])
                library.assert_not_called()

    def test_rejected_request_stops_execution(self):
        with patch.object(sys, 'platform', 'darwin'), patch.object(runtime.ctypes, 'CDLL') as library:
            library.return_value.pthread_set_qos_class_self_np.return_value = errno.EPERM
            with self.assertRaises(OSError) as caught:
                runtime.request_performance_qos()
            self.assertEqual(caught.exception.errno, errno.EPERM)

    @unittest.skipUnless(sys.platform == 'darwin', 'Requires the real macOS scheduler')
    def test_macos_scheduler_applies_qos_in_fresh_process(self):
        # Read the effective class from macOS independently of the returned metadata.
        program = '''
import ctypes
import json
from experiments.runtime import request_performance_qos
metadata = request_performance_qos()
query = ctypes.CDLL('/usr/lib/libSystem.B.dylib').qos_class_self
query.argtypes = []
query.restype = ctypes.c_uint
print(json.dumps(dict(metadata=metadata, effective_class=query())))
'''
        result = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                                text=True, capture_output=True, check=True)
        observed = json.loads(result.stdout)
        self.assertEqual(observed['effective_class'], 0x19)
        self.assertTrue(observed['metadata']['requested'])
        self.assertFalse(observed['metadata']['hard_affinity'])

    def test_module3_native_worker_records_shared_policy(self):
        from experiments.paper_final import run_online as runner
        raw = runner.read(ROOT/'configs/module3/variable_flow_60/seed_1.json')
        raw['problem']['grid'] = [12, 12, 12]
        raw['stream'].update(cases=2, online_cases=2, group_take=2,
                             cases_per_seed=1, expected_sha256='')
        spec = runner.parse_joint_experiment_config(raw)
        args = runner.runtime_config_from_spec(spec)
        runner.configure_paired_environment(args)
        _, manifest = runner.build_paired_instance_stream(args)
        raw['stream']['expected_sha256'] = manifest['sha256']
        with TemporaryDirectory(dir=ROOT/'results', prefix='module3_scheduling_') as directory:
            base = Path(directory)
            runner.dump(base/'config.json', raw)
            runner.dump(base/'suite.json', dict(name='Module 3 scheduling validation',
                seeds=[dict(label='1', config=str(base/'config.json'))]))
            output = base/'run'
            runner.prepare(base/'suite.json', output, cases=2)
            with patch('experiments.paper_final.plot_online.generate'):
                runner.launch(output)
            self.assertTrue(runner.read(output/'status.json')['passed'])
            environment = runner.read(output/'seed_1/environment.json')
            self.assertEqual(environment['qos']['requested'], sys.platform == 'darwin')
            self.assertFalse(environment['qos']['hard_affinity'])
            self.assertTrue(all(value == '1' for value in environment['threads'].values()))
            rows = runner.read(output/'summary.json')['rows']
            self.assertEqual({row['method'] for row in rows}, set(runner.factorial.METHODS))
            self.assertEqual(len([row for row in rows if row['window'] == 'full']), 4)


if __name__ == '__main__':
    unittest.main()
