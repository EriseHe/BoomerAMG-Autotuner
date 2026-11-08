"""Use the frozen measurement runner with a separate reporting-only repair."""
from pathlib import Path
import argparse,importlib.util,sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from experiments.runtime import configure_single_thread
configure_single_thread()
from experiments.paper_final.matched_policy import workflow
from experiments.paper_final import plot_policy
path=Path(__file__).with_name('plot_policy_report_fix.py')
spec=importlib.util.spec_from_file_location('module3_report_fix',path)
fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)
plot_policy.generate=fixed.generate

def reject_measurement(*args,**kwargs):
    raise RuntimeError('Report-only finalization must not rerun a measured phase')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    output=args.output.resolve()
    if args.report_only:
        assert all((output/f'raw/{phase}/complete.json').exists() for phase in ('scan','test'))
        workflow.run_frozen_phase=reject_measurement
    workflow.run(output)
    result=json.loads((output/'status.json').read_text())
    assert result['status']=='complete'
    print(json.dumps(result,indent=2))
