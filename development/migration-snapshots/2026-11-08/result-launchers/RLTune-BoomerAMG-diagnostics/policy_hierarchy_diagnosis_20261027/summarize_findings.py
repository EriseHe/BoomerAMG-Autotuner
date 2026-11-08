"""Reproduce the diagnostic comparisons used in the accompanying discussion."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
load=lambda name:json.loads((ROOT/name).read_text())
m1=load('advection_60_module1.json')
m2=load('advection_60_module2.json')
diff=load('diffusion_60_module2.json')
cases=load('advection_60_paired_cases.json')
third=m2['seeds']['3']['methods']
other=[m2['seeds'][str(s)]['methods'] for s in (1,2,4,5,6)]
gap=third['reference']['native_sec']-np.mean([x['reference']['native_sec'] for x in other])
recovery_gap=third['reference']['recovery_sec']-np.mean([x['reference']['recovery_sec'] for x in other])
findings={
    'module1_phase':'frozen inputs 5001–6000; initial setup choices; accepted Advection 60 data',
    'module2_population':'same 100 inputs across all six checkpoints; same primary hierarchy across policies within checkpoint/input; 3 timing repetitions averaged within input',
    'module2_cost':'native continuation seconds summed over 100 inputs; excludes common primary setup; includes all recovery setup/solve',
    'seed3_module1_aggressive_joint_pct':100*m1['3']['bandit_lstdq']['frozen']['aggressive_setup_fraction'],
    'seed3_module1_aggressive_setup_only_pct':100*m1['3']['bandit_default']['frozen']['aggressive_setup_fraction'],
    'seed3_module1_primary_cycles':m1['3']['bandit_lstdq']['frozen']['first_attempt_cycles'],
    'seed3_module1_primary_ms_per_cycle':m1['3']['bandit_lstdq']['frozen']['first_attempt_cycle_ms'],
    'seed3_module2_reference_gap_vs_other_seed_mean_sec':float(gap),
    'seed3_module2_recovery_contribution_to_reference_gap_sec':float(recovery_gap),
    'seed3_module2_recovery_contribution_to_reference_gap_pct':float(100*recovery_gap/gap),
    'seed3_module2_policy_comparison':{m:{k:v[k] for k in ('native_sec','reduction_pct','primary_cycles','recovered_cases')} for m,v in third.items()},
    'seed3_rl_vs_periodic_pct':100*(1-third['rl']['native_sec']/third['periodic']['native_sec']),
    'seed3_rl_vs_best_global_fixed_pct':100*(1-third['rl']['native_sec']/third['fixed']['native_sec']),
    'seed3_rl_vs_per_problem_fixed_pct':100*(1-third['rl']['native_sec']/third['oracle']['native_sec']),
    'seed3_setup_groups':{},
    'comparison_families':{},
    'limits':['Setup parameters were not independently randomized; their individual causal effects are not identified.',
              'The recorded setup configurations are available, but realized hierarchy level sizes, operator complexity, and spectra were not recorded.',
              'The periodic policy was an evaluation comparator, not the training policy.',
              'The Module 2 w=1 reference acts on learned hierarchies and differs from the complete Default method in Module 1.'],
}
for aggressive in (False,True):
    group=[x for x in cases if x['seed']==3 and (x['params']['agg_num_levels']>0)==aggressive]
    totals={m:sum(x['methods'][m]['native_sec'] for x in group) for m in third}
    findings['seed3_setup_groups'][str(aggressive)]={
        'inputs':len(group),'native_sec':totals,
        'reference_recovered_inputs':sum(x['methods']['reference']['fallbacks']>0 for x in group),
        'rl_vs_periodic_pct':100*(1-totals['rl']/totals['periodic']),
        'rl_mean_primary_cycles':float(np.mean([x['methods']['rl']['primary_cycles'] for x in group])),
        'periodic_mean_primary_cycles':float(np.mean([x['methods']['periodic']['primary_cycles'] for x in group])),
    }
for name,data in [('advection_60',m2),('diffusion_60',diff)]:
    savings=[s['methods']['rl']['reduction_pct'] for s in data['seeds'].values()]
    counts=[100-int(s['setup_counts']['agg_num_levels'].get('0',0)) for s in data['seeds'].values()]
    findings['comparison_families'][name]={'aggressive_setup_percentages':counts,
                                         'rl_native_reduction_mean_pct':float(np.mean(savings)),
                                         'rl_native_reduction_sample_sd_pp':float(np.std(savings,ddof=1))}
# Audit every case's accounting independently of the plotting helper.
for case in cases:
    for m,v in case['methods'].items():
        assert abs(v['native_sec']-v['primary_sec']-v['recovery_sec'])<1e-7
(ROOT/'findings.json').write_text(json.dumps(findings,indent=2,allow_nan=False)+'\n')
print('Wrote findings.json; all 3,000 Advection case/policy accounting checks passed.')
print('Seed 3: RL vs global fixed, per-problem fixed, periodic:',
      *(round(findings[k],3) for k in ('seed3_rl_vs_best_global_fixed_pct','seed3_rl_vs_per_problem_fixed_pct','seed3_rl_vs_periodic_pct')))
