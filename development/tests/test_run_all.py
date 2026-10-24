"""Check complete-study order, explicit model handoff and failure propagation."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments import run_all


class CompleteStudyTests(unittest.TestCase):
    def test_plan_connects_all_suites_and_exact_training_paths(self):
        output = Path('/tmp/full-study')
        stages = run_all.plan(output, 'diffusion_60')
        self.assertEqual(len(stages), 32)
        self.assertEqual(stages[0][0], 'module1_advection_60_batch1_prepare')
        self.assertEqual([name for name, _ in stages[24:28]],
                         ['module2_init', 'module2_validate', 'module2_prepare', 'module2_run'])
        self.assertEqual(stages[-1][0], 'module3_variable_flow_60_batch2_run')
        init = stages[24][1]
        sources = [init[i+1] for i, value in enumerate(init) if value == '--training-run']
        self.assertEqual(sources, [f'{seed}={output}/module1/diffusion_60_batch{1 if seed<=3 else 2}/seed_{seed}'
                                  for seed in range(1, 7)])
        alternative = run_all.plan(output, 'advection_40')[24][1]
        self.assertIn(f'6={output}/module1/advection_40_batch2/seed_6', alternative)
        for _, argv in stages:
            if '--suite' in argv:
                suite = Path(argv[argv.index('--suite')+1])
                self.assertTrue(suite.is_file())
                for entry in json.loads(suite.read_text())['seeds']:
                    self.assertTrue((suite.parent/entry['config']).is_file())
        self.assertFalse(any('--cases' in argv for _, argv in stages))

    def test_dry_run_creates_nothing(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)/'study'
            with patch('sys.argv', ['run_all','--output',str(output),'--dry-run']), redirect_stdout(StringIO()) as log:
                run_all.main()
            self.assertEqual(len(json.loads(log.getvalue())),32)
            self.assertFalse(output.exists())

    def test_commands_are_sequential_and_stop_on_failure(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)/'study'
            with patch('sys.argv', ['run_all','--output',str(output)]), \
                 patch.object(run_all, 'source_identity', return_value={'source':'hash'}), \
                 patch.object(run_all, 'prevent_sleep', return_value=None), \
                 patch.object(run_all, 'subprocess') as commands, redirect_stdout(StringIO()):
                commands.run.side_effect = [None, subprocess.CalledProcessError(1,'test')]
                with self.assertRaises(subprocess.CalledProcessError):run_all.main()
            status = json.loads((output/'status.json').read_text())
            self.assertEqual(status['status'],'failed')
            self.assertEqual(status['completed_commands'],1)
            self.assertEqual(commands.run.call_count,2)
            self.assertEqual(commands.run.call_args.kwargs['env']['OMP_NUM_THREADS'],'1')

    def test_existing_outputs_are_never_reused(self):
        with TemporaryDirectory() as directory:
            with patch('sys.argv', ['run_all','--output',directory]), redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit):run_all.main()
