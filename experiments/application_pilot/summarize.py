"""Verify completeness and summarize the two default-only application pilots."""
import csv,hashlib,json,statistics,subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[2]
out=root/'results/application_default_100'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def summary(rows):
    assert len(rows)==100 and [int(r['case']) for r in rows]==list(range(1,101))
    result={'cases':100,'n':int(rows[0]['n']),'nnz':int(rows[0]['nnz']),
            'converged':sum(bool(r['converged']) for r in rows),
            'failed':sum(not bool(r['converged']) for r in rows)}
    for key in ['setup_sec','solve_sec','setup_solve_sec','iterations','true_residual']:
        values=[float(r[key]) for r in rows]
        result[key]={'mean':statistics.mean(values),'median':statistics.median(values),
                     'min':min(values),'max':max(values),'total':sum(values)}
    success=[r['setup_solve_sec'] for r in rows if r['converged']]
    result['successful_only_mean_setup_solve_sec']=statistics.mean(success) if success else None
    return result
def heat_rows(path):
    with path.open() as f:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    for r in rows:r['setup_solve_sec']=r['setup_sec']+r['solve_sec']
    return rows

spe=[json.loads(s) for s in (out/'spe10/records.jsonl').read_text().splitlines()]
heat=heat_rows(out/'heat_records_500.csv')
diagnostic=heat_rows(out/'heat_records.csv')
assert all(r['reference']['externally_accepted'] for r in spe)
assert max(r['reference_residual'] for r in heat)<1e-10
heat_mass=[r['mass_integral'] for r in heat]
mass_drift=(max(heat_mass)-min(heat_mass))/statistics.mean(heat_mass)
assert mass_drift<1e-8
with (out/'spe10/application.csv').open() as f:
    app=[[float(x) for x in row] for row in csv.reader(f)]
assert len(app)==100
assert min(r[3] for r in app)>=-1e-8 and max(r[4] for r in app)<=1+1e-8
assert 'implicitTransport failed' not in (out/'spe10/octave.log').read_text()

report={'protocol':{'tolerance':1e-6,'max_cycles':500,'ranks':1,'threads_per_process':1,
         'solver':'paper explicit default standalone AMG, l1 Jacobi / direct coarse',
         'hierarchy':'rebuilt every measured system','warm_start':'previous reference solution; heat starts at initial temperature',
         'measured':'HYPRE hierarchy setup + standalone AMG solve',
         'excluded':'assembly, CSR import, external residual checks, reference solves, transport, installation/build',
         'concurrency':'SPE10 and heat pilots overlapped on this machine; each solver process was single-threaded',
         'statistical_scope':'one deterministic chronological physical trajectory each; no learning or seed replication'},
        'spe10':summary(spe),'mfem_heat':summary(heat),'heat_50_cycle_diagnostic':summary(diagnostic),
        'accuracy_checks':{'heat_relative_mass_drift':mass_drift,
         'spe10_reference_max_true_residual':max(r['reference']['true_residual'] for r in spe),
         'spe10_min_saturation':min(r[3] for r in app),'spe10_max_saturation':max(r[4] for r in app)},
        'integration':{'spe10_mean_import_and_check_wall_sec':statistics.mean(r['wall_with_import_and_check_sec'] for r in spe),
         'spe10_mean_transport_sec':statistics.mean(r[2] for r in app),
         'spe10_mean_pressure_application_wall_including_reference_and_ipc_sec':statistics.mean(r[1] for r in app),
         'heat_mean_assembly_sec':statistics.mean(r['assembly_sec'] for r in heat),
         'heat_mean_csr_import_sec':statistics.mean(r['import_sec'] for r in heat)},
        'provenance':{'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
         'working_tree':subprocess.check_output(['git','status','--short','--untracked-files=all'],cwd=root,text=True),
         'mfem_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=out/'mfem',text=True).strip(),
         'mrst_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=out/'MRST',text=True).strip(),
         'hypre_version':'3.0.0','mfem_version':'4.8','octave_version':'10.3.0'}}
files=[*sorted((root/'experiments/application_pilot').glob('*.*')),
       out/'SPE10.zip',out/'heat.mesh',out/'libpilot.dylib',out/'heat_pilot',out/'heat_pilot_500',
       root/'hypre/install/lib/libHYPRE.dylib',out/'spe10/records.jsonl',out/'heat_records_500.csv',out/'heat_records.csv']
report['provenance']['sha256']={str(p.relative_to(root)):digest(p) for p in files if p.is_file()}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='provenance'},indent=2))
