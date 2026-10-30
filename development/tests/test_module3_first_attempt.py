from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from experiments.paper_final.common import frozen_policy as policy
from experiments.paper_final.matched_policy.selection import select_fixed
from experiments.paper_final.plot_policy import paired_gains
from hypre.bindings import AttemptOutcome, RecoveryOutcome

JOB = dict(seed=1,source='joint',case_id=0,input_id='input',hierarchy_id='hierarchy',mkw={},params={})

def native_result(failed=True):
    return dict(runtime=107.,native_runtime=102.,setup_runtime=100.,solve_runtime=7.,
                native_solve_runtime=2.,infer_runtime=5.,failed=failed,
                failure_reason='max_cycles' if failed else '',attempt_status='nonconvergence' if failed else 'success',
                residual_norm=1. if failed else 1e-8,iterations=1,cycle_actions=[1.],cycle_times=[2.])

class FirstAttemptTests(unittest.TestCase):
    def test_selection_averages_repetitions_then_sums_cases_without_success_filter(self):
        jobs=[];rows=[]
        costs={0:{'fixed_1':[1.,3.,2.],'fixed_2':[4.,4.,4.]},1:{'fixed_1':[8.,8.,8.],'fixed_2':[5.,5.,5.]}}
        for case in (0,1):
            for repeat in range(3):
                job=dict(JOB,case_id=case,repeat=repeat,policy_order=['fixed_1','fixed_2']);jobs.append(job)
                for name in job['policy_order']:
                    rows.append(dict(job,policy=name,native_solve_sec=costs[case][name][repeat],
                                     native_total_sec=1000. if name=='fixed_1' else 0.,success=name=='fixed_2'))
        result=select_fixed(rows,jobs)['1']
        self.assertEqual(result['global'],'fixed_2')
        self.assertEqual(result['per_instance'],{'0':'fixed_1','1':'fixed_2'})
        self.assertEqual(result['uncovered_case_ids'],[])
        self.assertEqual(result['global_candidates'][0]['failed_cases'],2)

    def test_fixed_policy_executes_no_fallback_and_excludes_setup_and_overhead(self):
        with patch.object(policy,'solve_schedule_case',return_value=native_result()), \
             patch.object(policy,'run_with_default_fallback',side_effect=AssertionError('fallback requested')):
            row=policy.run_first_attempt_policy(JOB,'fixed_1',dict(kind='fixed',pattern=[1.]),None)
        self.assertFalse(row['success'])
        self.assertEqual(row['native_solve_sec'],2.)
        self.assertEqual(row['inclusive_solve_sec'],7.)
        self.assertEqual(row['native_total_sec'],102.)
        self.assertFalse(row['outcome']['fallback_used'])

    def test_rl_disables_learning_exploration_and_fallback(self):
        observed=[]
        def run(case):
            observed.append(case)
            return RecoveryOutcome(primary=AttemptOutcome.from_mapping(native_result())).to_result()
        with patch.object(policy,'run_with_default_fallback',side_effect=AssertionError('fallback requested')):
            row=policy.run_first_attempt_policy(JOB,'rl_frozen',dict(kind='rl'),SimpleNamespace(run_case=run,controller=SimpleNamespace(finish_episode=lambda **kw:self.assertFalse(kw['learned']))))
        self.assertFalse(observed[0].learn)
        self.assertFalse(observed[0].explore)
        self.assertIsNone(observed[0].fallback_attempt)
        self.assertEqual(row['native_solve_sec'],2.)
        self.assertFalse(row['success'])

    def test_reductions_include_failed_first_attempts(self):
        costs=np.ones((2,5,3,4));costs[:,0]=2.
        gains=paired_gains(costs,np.zeros((2,5,3),dtype=bool))
        np.testing.assert_allclose(gains[:,1:],50.)

    def test_metric_audit_rejects_fallback_or_modified_solve_time(self):
        with patch.object(policy,'solve_schedule_case',return_value=native_result()):
            row=policy.run_first_attempt_policy(JOB,'fixed_1',dict(kind='fixed',pattern=[1.]),None)
        row['native_solve_sec']=102.
        with self.assertRaises(AssertionError):policy.audit_first_attempt_row(row)
        row['native_solve_sec']=2.;row['outcome']['fallback_used']=True
        with self.assertRaises(AssertionError):policy.audit_first_attempt_row(row)

if __name__=='__main__':unittest.main()
