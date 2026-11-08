"""Combine untouched prefix measurements with a checkpoint continuation and audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.application_online.audit import audit


def merge(prefix,continuation,output):
    prefix,continuation,output=map(Path,(prefix,continuation,output))
    protocols=[json.loads((p/'protocol.json').read_text()) for p in (prefix,continuation)]
    summaries=[json.loads((p/'summary.json').read_text()) for p in (prefix,continuation)]
    first,last=protocols;offset=first['cases'];total=offset+last['cases']
    assert last['case_offset']==offset and last['prefix']==str(prefix.resolve())
    assert first['seed']==last['seed'] and first['max_cycles']==last['max_cycles']==100
    assert first['methods']==last['methods'] and first['context_fields']==last['context_fields']
    assert all(s['complete'] for s in summaries)
    output.mkdir(parents=True,exist_ok=False)
    for directory,protocol in zip((prefix,continuation),protocols):
        with zipfile.ZipFile(directory/'source_snapshot.zip') as archive:
            for name,sha in protocol['source_sha256'].items():
                assert hashlib.sha256(archive.read(name)).hexdigest()==sha
    for name in ('records.jsonl','reference.jsonl'):
        with (output/name).open('w') as target:
            for directory in (prefix,continuation):
                with (directory/name).open() as source:shutil.copyfileobj(source,target)
    arrays=[np.loadtxt(p/'application.csv',delimiter=',',ndmin=2) for p in (prefix,continuation)]
    arrays[1][:,0]+=offset
    arrays[1][:,6]+=offset//first['steps_per_scenario']
    np.savetxt(output/'application.csv',np.vstack(arrays),delimiter=',',fmt='%.17g')
    campaign=[]
    for protocol in protocols:
        count=(protocol['cases']+protocol['steps_per_scenario']-1)//protocol['steps_per_scenario']
        campaign.extend(protocol['campaign'][:count])
    campaign=[dict(item,scenario=i+1) for i,item in enumerate(campaign)]
    protocol=dict(last,cases=total,case_offset=0,target_cases=total,campaign=campaign,
        segments=[str(prefix.resolve()),str(continuation.resolve())],
        source_snapshot_semantics='continuation sources at root; both exact segment source snapshots nested under segments',
        learning='checkpoint continuation; original prefix measurements retained without recomputation')
    protocol.pop('prefix',None)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with zipfile.ZipFile(output/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as target:
        with zipfile.ZipFile(continuation/'source_snapshot.zip') as source:
            for name in last['source_sha256']:target.writestr(name,source.read(name))
        target.write(output/'protocol.json','protocol.json')
        target.write(prefix/'source_snapshot.zip','segments/prefix_source_snapshot.zip')
        target.write(continuation/'source_snapshot.zip','segments/continuation_source_snapshot.zip')
    summary=dict(complete=True,cases=total,segments=protocol['segments'],
        initialization_sec=sum(s['initialization_sec'] for s in summaries),
        elapsed_sec=sum(s['elapsed_sec'] for s in summaries),methods={})
    for method in first['methods']:
        a,b=(s['methods'][method] for s in summaries)
        result=dict(b)
        for key in ('cases','total_wall_sec','cold_start_total_sec','primary_failures',
                    'first_attempt_failures','rescues','final_failures','hierarchy_builds'):
            result[key]=a[key]+b[key]
        result['mean_total_wall_sec']=result['total_wall_sec']/result['cases']
        summary['methods'][method]=result
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name in ('linucb_setup.npz','joint_setup.npz','lstdq_controller.npz','joint_controller.npz',
                 'lstdq_controller.npz.encoder.json','joint_controller.npz.encoder.json'):
        shutil.copy2(continuation/name,output/name)
    audit(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prefix',type=Path);p.add_argument('continuation',type=Path)
    p.add_argument('output',type=Path);args=p.parse_args();merge(args.prefix,args.continuation,args.output)
