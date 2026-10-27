"""Keep each frozen-policy worker on one seed and wait at seed-batch boundaries."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.paper_final.common import artifacts, workers


class MatchedPolicyWorkerTests(unittest.TestCase):
    def prepare(self, root, phase, seeds):
        for folder in ('progress', f'raw/{phase}'):
            (root/folder).mkdir(parents=True, exist_ok=True)
        artifacts.dump(root/'protocol.json', dict(training_seeds=seeds, max_workers=3,
                                                 policies={'fixed': {}, 'rl': {}}, tolerance=1e-6))
        # Interleave seeds in the plan; workers must preserve only their own jobs.
        jobs = [dict(seed=seed, source='joint', case_id=case, repeat=repeat,
                     policy_order=['fixed', 'rl'])
                for case in (1, 0) for seed in reversed(seeds) for repeat in range(3)]
        artifacts.dump(root/f'jobs_{phase}.json', jobs)
        return jobs

    def policy(self):
        calls = []
        loads = []
        def load(output, seed):
            loads.append(seed)
            return seed
        def run(job, name, spec, bundle):
            self.assertEqual(job['seed'], bundle)
            calls.append((bundle, job['case_id'], job['repeat'], name))
            return dict(job, policy=name, outcome={}, wall_sec=0.001)
        return SimpleNamespace(load_bundle=load, frozen_snapshot=lambda bundle: bundle,
                               run_policy=run, verify_frozen=lambda bundle, snapshot: self.assertEqual(bundle, snapshot)), calls, loads

    def test_workers_keep_one_seed_and_resume_without_duplicate_cells(self):
        for phase in ('scan', 'test'):
            with self.subTest(phase=phase), TemporaryDirectory() as directory:
                root = Path(directory)
                seeds = [4, 5, 6, 1, 2, 3]
                self.prepare(root, phase, seeds)
                policy, calls, loads = self.policy()
                with patch.object(workers, 'request_performance_qos', return_value={}):
                    for index, seed in enumerate(seeds):
                        for _ in range(2):
                            workers.run_frozen_worker(root, phase, index, 3, run_number=2,
                                                      verify=lambda output: None, policy=policy)
                        rows = list(artifacts.read_records(root/f'raw/{phase}/worker_{index}.jsonl'))
                        self.assertEqual({row['seed'] for row in rows}, {seed})
                        self.assertEqual(len(rows), 12)
                self.assertEqual(len(calls), 72)
                self.assertEqual(loads, [seed for seed in seeds for _ in range(2)])

    def test_supervisor_waits_for_each_batch_and_covers_every_seed(self):
        for phase in ('scan', 'test'):
            with self.subTest(phase=phase), TemporaryDirectory() as directory:
                root = Path(directory)
                seeds = [1, 2, 3, 4, 5, 6]
                jobs = self.prepare(root, phase, seeds)
                policy, calls, loads = self.policy()
                events = []
                def launch(command, **kwargs):
                    index = int(command[-1])
                    events.append(('launch', index))
                    workers.run_frozen_worker(root, phase, index, 3, run_number=2,
                                              verify=lambda output: None, policy=policy)
                    def poll():
                        events.append(('poll', index))
                        return 0
                    return SimpleNamespace(pid=100+index, poll=poll, wait=lambda **kw: 0)
                with patch.object(workers, 'request_performance_qos', return_value={}), \
                     patch.object(workers.subprocess, 'Popen', side_effect=launch), \
                     patch.object(workers.frozen_policy, 'audit_outcome'):
                    workers.run_frozen_phase(root, phase, runner_module='unused', run_number=2, workers=3)
                self.assertEqual(events[:6], [('launch', 0), ('launch', 1), ('launch', 2),
                                             ('poll', 0), ('poll', 1), ('poll', 2)])
                self.assertEqual(events[6:12], [('launch', 3), ('launch', 4), ('launch', 5),
                                              ('poll', 3), ('poll', 4), ('poll', 5)])
                self.assertEqual(loads, seeds)
                marker = artifacts.read(root/f'raw/{phase}/complete.json')
                self.assertEqual(marker['trials'], len(jobs)*2)
                self.assertEqual(len(marker['raw_sha256']), 6)

    def test_worker_failure_stops_before_next_batch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, 'test', list(range(1, 7)))
            launched = []
            def launch(command, **kwargs):
                launched.append(int(command[-1]))
                return SimpleNamespace(pid=100, poll=lambda: 1, wait=lambda **kw: 1)
            with patch.object(workers.subprocess, 'Popen', side_effect=launch):
                with self.assertRaisesRegex(RuntimeError, 'worker stopped'):
                    workers.run_frozen_phase(root, 'test', runner_module='unused', run_number=2, workers=3)
            self.assertEqual(launched, [0, 1, 2])
            self.assertFalse((root/'raw/test/complete.json').exists())


if __name__ == '__main__':
    unittest.main()
