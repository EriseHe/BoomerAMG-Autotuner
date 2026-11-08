"""Summarize accepted Module 1 setup choices and recorded solve trajectories."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG')
OUT = Path(__file__).resolve().parent
SUMMARY = {}
for phase in ('online','frozen'):
    with (REPO/f'data/module1/{phase}/summary.csv').open() as f:
        for r in csv.DictReader(f):
            if r['experiment']=='advection_60' and r['window']=='full':
                SUMMARY[phase,int(r['seed']),r['method']]=r

def summarize(rows):
    assert rows
    params=[r['params'] for r in rows]
    actions=np.asarray([a for r in rows for a in r['actions']])
    cycles=sum(r['first_cycles'] for r in rows)
    switches=[a!=b for r in rows for a,b in zip(r['actions'],r['actions'][1:])]
    hi_lo=[a>=2.5 and b<=1.5 for r in rows for a,b in zip(r['actions'],r['actions'][1:])]
    return {
        'n':len(rows),
        'setup_counts':{k:dict(Counter(str(p[k]) for p in params)) for k in params[0]},
        'aggressive_setup_fraction':float(np.mean([p['agg_num_levels']>0 for p in params])),
        'unique_configs':len({json.dumps(p,sort_keys=True) for p in params}),
        'setup_sec':sum(r['setup_sec'] for r in rows),
        'solve_sec':sum(r['solve_sec'] for r in rows),
        'native_sec':sum(r['native_sec'] for r in rows),
        'total_sec':sum(r['total_sec'] for r in rows),
        'first_attempt_cycles':float(np.mean([r['first_cycles'] for r in rows])),
        'first_attempt_cycle_ms':1000*sum(r['first_solve_sec'] for r in rows)/cycles if cycles else None,
        'first_attempt_failures':sum(r['first_failed'] for r in rows),
        'fallbacks':sum(r['fallback'] for r in rows),
        'action_counts':dict(Counter(str(round(a,3)) for a in actions)),
        'switch_fraction':float(np.mean(switches)) if switches else None,
        'high_low_pair_fraction':float(np.mean(hi_lo)) if hi_lo else None,
    }

result={}
for seed in range(1,7):
    batch=1 if seed<=3 else 2
    root=REPO/f'data/official/module1/batches/batch_{batch:02d}/advection_60/seed_{seed}'
    result[seed]={}
    for method in ('bandit_lstdq','bandit_default'):
        path=root/f'trajectories/{method}.jsonl'
        rows=[]; h=hashlib.sha256()
        with path.open('rb') as f:
            for i,line in enumerate(f,1):
                h.update(line)
                r=json.loads(line)
                assert r['problem_index']==i
                o=r['outcome']; first=o['primary_attempts'][0]
                if i>5000:
                    assert not o.get('bandit_update_committed',False)
                    assert not o.get('controller_update_committed',False)
                assert not o['failed']
                actions=first.get('cycle_actions',o.get('cycle_actions',[]))
                times=first.get('cycle_times',o.get('cycle_times',[]))
                rows.append({'params':first['params'], 'actions':actions,'times':times,
                             'first_cycles':first['cycles'],'first_solve_sec':first['solve_runtime'],
                             'setup_sec':o['setup_runtime'], 'solve_sec':o['solve_runtime'],
                             'native_sec':o['native_runtime'],'total_sec':o['end_to_end_runtime'],
                             'first_failed':o['first_primary_status']!='success',
                             'fallback':bool(o['fallback_used'])})
        assert len(rows)==6000
        data={'source':str(path),'sha256':h.hexdigest(),
              'blocks':{f'{a+1}-{a+1000}':summarize(rows[a:a+1000]) for a in range(0,6000,1000)}}
        for phase,lo,hi in [('online',0,5000),('frozen',5000,6000)]:
            stats=summarize(rows[lo:hi]); expected=SUMMARY[phase,seed,method]
            for key in ('native_sec','total_sec'):
                np.testing.assert_allclose(stats[key],float(expected[key]),rtol=1e-10,atol=1e-8)
            data[phase]=stats
        result[seed][method]=data
    print('Finished seed',seed,flush=True)
    for method,data in result[seed].items():
        print(method,[(k,round(v['aggressive_setup_fraction'],3),round(v['first_attempt_cycles'],2),round(v['native_sec'],2)) for k,v in data['blocks'].items()],flush=True)
(OUT/'advection_60_module1.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('All 72,000 problem records validated against accepted Module 1 summaries.',flush=True)
