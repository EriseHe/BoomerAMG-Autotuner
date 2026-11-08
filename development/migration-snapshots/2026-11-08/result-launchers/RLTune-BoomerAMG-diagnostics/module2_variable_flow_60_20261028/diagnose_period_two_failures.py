"""Describe recorded period-two failures; no new solves or source changes."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import math
import statistics as stats

BASE = Path(__file__).resolve().parent
RUN = BASE / 'run'
jobs = {(j['seed'], j['case_id']): j for j in json.loads((RUN/'jobs_base.json').read_text())}
selections = json.loads((RUN/'selections.json').read_text())
trials = {}
for path in sorted((RUN/'raw/test').glob('worker_*.jsonl')):
    for line in path.open():
        r = json.loads(line)
        key = (r['seed'], r['case_id'], r['policy'], r['repeat'])
        assert key not in trials
        trials[key] = r

def summary(values):
    return dict(n=len(values), min=min(values), median=stats.median(values),
                max=max(values), mean=stats.mean(values)) if values else None

def primary_cycles(outcome):
    primary = (outcome.get('primary_attempts') or [outcome])[0]
    return primary.get('cycles', outcome.get('primary_cycles', outcome.get('iterations', 0)))

bins = [dict(n=0, unique_cases=set(), failures=Counter()) for _ in range(3)]
by_seed = {s: [Counter() for _ in range(3)] for s in range(1, 7)}
cases = []
growth = defaultdict(list)
failure_reasons = Counter()
repetition_cycle_differences = Counter()
late_trends = Counter()
fail_eps, fail_resid, failed_low_eps_resid = [], [], []
for (seed, case), job in sorted(jobs.items()):
    eps = job['mkw']['k']
    band = 0 if eps < 1e-3 else 1 if eps < 1e-2 else 2
    names = dict(periodic='periodic', rl='rl_frozen', reference='fixed_1',
                 global_fixed=selections[str(seed)]['global'],
                 per_problem_fixed=selections[str(seed)]['per_instance'][str(case)])
    rows = {role: trials[seed, case, name, 0] for role, name in names.items()}
    bucket = bins[band]
    bucket['n'] += 1
    bucket['unique_cases'].add(job['input_id'])
    by_seed[seed][band]['n'] += 1
    statuses = {}
    for role, row in rows.items():
        o = row['outcome']
        statuses[role] = dict(primary_failed=o['primary_status'] != 'success',
                              final_failed=not row['success'],
                              cycles=primary_cycles(o))
        for repeat in (1, 2):
            other = trials[seed, case, names[role], repeat]
            assert other['outcome']['primary_status'] == o['primary_status']
            assert other['success'] == row['success']
            repetition_cycle_differences[role] += primary_cycles(other['outcome']) != primary_cycles(o)
        bucket['failures'][role] += statuses[role]['primary_failed']
    o = rows['periodic']['outcome']
    residuals = o['cycle_residuals']
    actions = o['cycle_actions']
    assert len(residuals) == len(actions) == o['primary_cycles']
    assert all(abs(w - (2.85 if i % 2 == 0 else 1.1)) < 1e-12 for i, w in enumerate(actions))
    failed = statuses['periodic']['primary_failed']
    by_seed[seed][band]['periodic_failures'] += failed
    if failed:
        assert o['primary_cycles'] == 50
        failure_reasons[o['primary_failure_reason']] += 1
        fail_eps.append(eps)
        fail_resid.append(residuals[-1])
        if band == 0:
            failed_low_eps_resid.append(residuals[-1])
        for w, before, after in zip(actions, [1.] + residuals[:-1], residuals):
            growth['high_weight' if w == 2.85 else 'low_weight'].append(after > before)
        growth['end_above_initial'].append(residuals[-1] > 1.)
        growth['final_pair_expands'].append(residuals[-1] > residuals[-3])
        trajectory = [1.] + residuals
        last_pair_ratios = [trajectory[k]/trajectory[k-2] for k in (42, 44, 46, 48, 50)]
        trend = ('decreasing' if all(q < 1 for q in last_pair_ratios) else
                 'increasing' if all(q > 1 for q in last_pair_ratios) else 'mixed')
        late_trends[trend] += 1
    cases.append(dict(seed=seed, case_id_zero_based=case, epsilon=eps,
                      input_id=job['input_id'], params=job['params'], status=statuses,
                      periodic_residual_at_stop=residuals[-1],
                      periodic_peak_residual=max([1.] + residuals),
                      periodic_last_pair_ratio=residuals[-1]/residuals[-3] if len(residuals)>2 else None,
                      periodic_residuals=residuals if seed == 1 and case == 30 else None))

result = dict(
    scope='100 shared held-out PDE/RHS inputs, each tested with six frozen checkpoints; repetitions are not counted as separate cases.',
    failure_definition='Official primary nonconvergence status. The native rule gives the 50-cycle cap priority over the strict relative-residual < 1e-6 check; two periodic trials first cross tolerance at that cap.',
    late_trend_definition='Trend across each of the final five complete two-cycle periods (cycles 40 to 50). This does not establish asymptotic convergence or divergence.',
    failed_late_trends=late_trends,
    failures_first_below_tolerance_at_cap=[dict(seed=c['seed'], case_id_zero_based=c['case_id_zero_based'], epsilon=c['epsilon'], residual=c['periodic_residual_at_stop']) for c in cases if c['status']['periodic']['primary_failed'] and c['periodic_residual_at_stop'] < 1e-6],
    trajectory_repetition_zero_based=0,
    all_three_repetitions_agree_on_primary_and_final_status=True,
    repetition_cycle_counts_differ_from_first_repetition=dict(repetition_cycle_differences),
    epsilon_bins=[dict(n=b['n'], unique_pde_rhs_inputs=len(b['unique_cases']), failures=dict(b['failures'])) for b in bins],
    by_seed=by_seed, failure_reasons=failure_reasons,
    failed_epsilon=summary(fail_eps), failed_residual=summary(fail_resid),
    failed_low_epsilon_residual=summary(failed_low_eps_resid),
    failed_growth={k: dict(count=sum(v), n=len(v), fraction=stats.mean(v)) for k,v in growth.items()},
    cases=cases)
output = BASE/'readout/period_two_failure_diagnostics.json'
output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'cases'}, indent=2))
print('High-epsilon failures:', json.dumps([c for c in cases if c['epsilon']>=1e-2 and c['status']['periodic']['primary_failed']],indent=2))
print('Evidence:', output)
