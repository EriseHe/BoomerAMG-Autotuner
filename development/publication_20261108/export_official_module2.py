from pathlib import Path
import json,csv,hashlib,gzip,shutil,statistics,math
import numpy as np
BASE=Path('/Users/erisehe/Documents/Codex/2026-11-08/i-am-about-to-s')
SRC=Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG-diagnostics/results')
OLD=SRC/'modules_2_3_20261027/study/module3/variable_flow_60_batch1'
NEW=SRC/'module2_variable_flow_rerun_20261030/run'
OUT=BASE/'work/boomer-publication/data/module2'
METHODS=['default_setup_default_solve','default_setup_lstdq','bandit_default','bandit_lstdq']
FIELDS=['setup_runtime','native_solve_runtime','native_runtime','infer_runtime','bandit_overhead_runtime','end_to_end_runtime']
def read(p):return json.loads(p.read_text())
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def savecsv(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if OUT.exists():shutil.rmtree(OUT)
OUT.mkdir(parents=True)
timings=np.zeros((6,4,5000,6)); failed=np.zeros((6,4,5000),bool); firstfailed=failed.copy();fallback=failed.copy();counts=np.zeros((6,4,5000),np.int8)
rawhashes={}; inputshashes={};rows=[]; phases=[];retrycounts={};validated=0
for seed in range(1,7):
 folder=OLD if seed<=3 else NEW
 assert read(folder/'completion_audit.json')['passed'] and read(folder/'status.json')['status']=='complete'
 srcrows=[r for r in read(folder/'summary.json')['rows'] if str(r['seed'])==str(seed)];assert len(srcrows)==16
 for r in srcrows:rows.append(dict(r,seed=str(seed)))
 path=folder/f'seed_{seed}'
 cfg=path/'experiment_config.json';dest=OUT/f'configurations/seed_{seed}.json';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(cfg,dest)
 inp=(path/'inputs.jsonl').read_bytes();inputshashes[str(seed)]=hashlib.sha256(inp).hexdigest();dest=OUT/f'inputs/seed_{seed}.jsonl.gz';dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('wb') as f:
  with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as g:g.write(inp)
 baseline_inputs=None
 for mi,method in enumerate(METHODS):
  p=path/f'trajectories/{method}.jsonl';h=hashlib.sha256();seen=set();paired=[];attemptfails=[0]*4
  with p.open('rb') as f:
   for line in f:
    h.update(line);r=json.loads(line);o=r['outcome'];idx=r['problem_index'];assert idx not in seen and r['method']==method;seen.add(idx);j=idx-1;paired.append((idx,r['mkw']))
    timings[seed-1,mi,j]=[o[k] for k in FIELDS]; failed[seed-1,mi,j]=o['failed'];firstfailed[seed-1,mi,j]=o['first_primary_status']!='success';fallback[seed-1,mi,j]=o['fallback_used'];counts[seed-1,mi,j]=o['primary_attempt_count']
    ats=o.get('primary_attempts',[])
    if method==METHODS[0]:attemptfails[0]+=bool(o['failed'])
    elif method==METHODS[1]:attemptfails[0]+=o['first_primary_status']!='success';attemptfails[1]+=bool(o['failed'])
    else:
     for k in range(3):
      if k<len(ats):attemptfails[k]+=ats[k]['status']!='success'
     attemptfails[3]+=bool(o['failed'])
  assert seen==set(range(1,5001));validated+=len(seen)
  if baseline_inputs is None:baseline_inputs=paired
  else:assert paired==baseline_inputs
  rawhashes[str(p)]={'bytes':p.stat().st_size,'sha256':h.hexdigest()};retrycounts.setdefault(method,[0]*4)
  retrycounts[method]=[a+b for a,b in zip(retrycounts[method],attemptfails)]
  a=timings[seed-1,mi];np.testing.assert_allclose(a[:,0]+a[:,1],a[:,2],atol=1e-8);np.testing.assert_allclose(a[:,2]+a[:,3]+a[:,4],a[:,5],atol=1e-8)
  for window,sl in [('full',slice(None)),('first_1000',slice(0,1000)),('joint',slice(None)),('last_1000',slice(4000,5000))]:
   rr=next(r for r in srcrows if r['method']==method and r['window']==window); vals=a[sl].sum(axis=0)
   assert math.isclose(vals[2],rr['native_sec'],rel_tol=1e-9,abs_tol=1e-7)
   assert math.isclose(vals[5],rr['total_sec'],rel_tol=1e-9,abs_tol=1e-7)
   assert int(failed[seed-1,mi,sl].sum())==rr['failures']
   if window in ['full','last_1000']:phases.append({'seed':str(seed),'method':method,'window':window,'cases':len(a[sl]),'setup_sec':float(vals[0]),'solve_sec':float(vals[1]),'learner_sec':float(vals[3]+vals[4])})
 print('Verified Module2 seed',seed,flush=True)
common_success=~failed.any(axis=1)
aggs=[]
for win in ['full','first_1000','joint','last_1000']:
 for method in METHODS:
  group=[r for r in rows if r['window']==win and r['method']==method];assert len(group)==6
  a={'window':win,'method':method,'seeds':6,'final_failures':sum(r['failures'] for r in group),'first_failures':sum(r['first_failures'] for r in group),'fallbacks':sum(r['fallbacks'] for r in group)}
  for metric in ['native_reduction_pct','inclusive_reduction_pct','common_success_native_reduction_pct']:
   a[metric+'_mean']=statistics.mean(r[metric] for r in group);a[metric+'_sd']=statistics.stdev(r[metric] for r in group)
  aggs.append(a)
np.savez_compressed(OUT/'arrays.npz',seeds=np.array([str(s) for s in range(1,7)]),methods=np.array(METHODS),problem_indices=np.arange(1,5001),timing_fields=np.array(FIELDS),timings_sec=timings,failed=failed,first_primary_failed=firstfailed,fallback_used=fallback,primary_attempt_count=counts,common_success=common_success)
provenance=OUT/'provenance';provenance.mkdir()
for label,folder in [('seeds_1_3',OLD),('seeds_4_6',NEW)]:
 for name in ['protocol.json','source_manifest.json','completion_audit.json']:
  shutil.copyfile(folder/name,provenance/(label+'_'+name))
for name in ['results.json','six_seed_results.csv']:shutil.copyfile(SRC/'module2_variable_flow_rerun_20261030/comparison_older123_rerun456'/name,OUT/('comparison_'+name))
shutil.copyfile(SRC/'module2_variable_flow_rerun_20261030/launch_provenance.json',provenance/'rerun_launch.json')
protocol={'module':2,'title':'Variable-flow online recovery, official current paper','seeds':list(range(1,7)),'cases_per_seed':5000,'methods':METHODS,'metric':'All setup/solve attempts plus measured learner overhead; mean and sample SD of six paired seed-level reductions','measurement_sources':{str(s):{'run':str(OLD if s<=3 else NEW),'seed':s,'raw_archive_prefix':f'data/official/module2/seeds/seed_{s}'} for s in range(1,7)},'input_sha256':inputshashes,'source_protocols':['provenance/seeds_1_3_protocol.json','provenance/seeds_4_6_protocol.json'],'historical_numbering':'The seeds 1–3 protocol retains its original Module3 runtime label; current paper Module2 is variable-flow.'}
dump(OUT/'protocol.json',protocol);dump(OUT/'summary.json',{'module':2,'rows':rows,'aggregates':aggs});savecsv(OUT/'summary.csv',rows);savecsv(OUT/'aggregate.csv',aggs);dump(OUT/'phase_breakdown.json',{'passed':True,'phases':phases})
expected={METHODS[0]:[1677,0,0,0],METHODS[1]:[777,26,0,0],METHODS[2]:[810,62,44,28],METHODS[3]:[1152,138,69,2]}
assert retrycounts==expected,(retrycounts,expected)
comparison=read(OUT/'comparison_results.json')
for a in comparison['aggregate']:
 ours=next(x for x in aggs if x['window']=='full' and x['method']==a['method'])
 assert math.isclose(ours['inclusive_reduction_pct_mean'],a['total_reduction_mean_pct'],abs_tol=1e-8)
 assert math.isclose(ours['inclusive_reduction_pct_sd'],a['total_reduction_sample_sd_pct'],abs_tol=1e-8)
audit={'passed':True,'records_independently_read':validated,'all_5000_problem_indices_complete_and_unique':True,'paired_mkw_inputs_exactly_equal':True,'array_timings_match_source_summaries':True,'raw_trajectory_hashes':rawhashes,'paper_retry_failure_counts':retrycounts,'common_success_cases':int(common_success.sum()),'official_displayed_full_reduction_pct':{a['method']:{'mean':round(a['inclusive_reduction_pct_mean'],1),'sample_sd':round(a['inclusive_reduction_pct_sd'],1)} for a in aggs if a['window']=='full'}}
assert audit['common_success_cases']==28323
dump(OUT/'fairness_audit.json',audit);dump(BASE/'work/paper-data-audit/module2-publication-validation.json',audit)
print(json.dumps({k:v for k,v in audit.items() if k!='raw_trajectory_hashes'},indent=2))
