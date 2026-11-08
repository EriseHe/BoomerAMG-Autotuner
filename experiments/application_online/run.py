"""Serial, paired application pilot with unchanged LinUCB/LSTDQ algorithms.

Chronological physical trajectories, four paper branches and PCG-AMG references.
An independently accurate pressure/heat solve advances the physical trajectory.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.sparse import csr_matrix
from hypre.bindings import create_env, augment_setup_params
from hypre.bindings.config import configure_smoother_profile
from setup.space import DEFAULT_SETUP_PARAMS, SetupConfigurationSpace
from setup.learners.linucb.feedback import FailureAwareRetryLinUCB
from experiments.paper_final.online.setup_branches import build_online_linucb_branch
from experiments.paper_final.online.controllers import make_setup_feature_encoder
from experiments.paper_final.online.factorial import balanced_orders
from experiments.paper_final.recovery.execution import CandidateRows
from experiments.paper_final.recovery.metrics import VARIANT
from joint.recovery import run_deferred_case
from joint.execution import EpisodeRecorder, run_fixed_case
from solve.controllers.common.config import SharedActionSpec, SolveStateSpec, EpsilonScheduleSpec
from solve.controllers.common.factory import OnlineControllerFactoryRequest
from solve.controllers.recursive_lstdq.config import RecursiveLstdqSpec
from solve.controllers.recursive_lstdq.factory import build_recursive_lstdq_controller
from solve.core.episode import run_td_episode
from problems.application_context import APPLICATION_CONTEXT_FIELDS, validate_application_context
from experiments.application_online.campaigns import scenarios

METHODS = ("default", "lstdq", "linucb", "joint", "pcg_rebuild", "pcg_reuse")
TOL = 1e-6


def true_residual(A, b, x):
    return float(np.linalg.norm(b-A@x)/(np.linalg.norm(b) or 1.0))


class Pcg:
    """Fixed symmetric GS AMG; reuse maximum age 10 or >2x initial iterations."""
    def __init__(self, reuse=False):
        self.reuse, self.env, self.age = reuse, None, 0
        self.last_iterations, self.build_iterations = 0, 0

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def solve(self, A, b, x0, *, tol=TOL, cap=100):
        started = time.perf_counter()
        x = x0.copy()
        initial = true_residual(A, b, x)
        if initial <= tol:
            return x, dict(setup_sec=0., solve_sec=0., iterations=0, true_residual=initial,
                converged=True, builds=0, wall_sec=time.perf_counter()-started)
        rebuild = (not self.reuse or self.env is None or self.age >= 10 or
                   self.last_iterations > 2*max(self.build_iterations, 1))
        setup = 0.
        current = create_env(matrix=A, rhs=b, initial_guess=x0, solution_out=x)
        try:
            if rebuild:
                self.close()
                self.env = current
                # Symmetric hybrid GS on both legs, direct coarse solve.
                setup = current.prepare_rl(dict(DEFAULT_SETUP_PARAMS, relax_type=6)).setup_runtime_sec
                self.age = 0
            stats = current.solve_pcg(self.env, tol=tol, max_iter=cap)
            # Fetch the solution without destroying a retained hierarchy.
            current.copy_solution_to(x)
        finally:
            if current is not self.env:
                current.close()
        self.age += 1
        self.last_iterations = stats["iterations"]
        if rebuild:
            self.build_iterations = self.last_iterations
        residual = true_residual(A, b, x)
        stats.update(setup_sec=setup+stats["setup_sec"], true_residual=residual,
            converged=bool(np.isfinite(residual) and residual <= tol), builds=int(rebuild),
            wall_sec=time.perf_counter()-started)
        if not stats["converged"]:
            self.close()
        return x, stats


class Study:
    def __init__(self, mode, out, cases, seed, cap):
        self.mode, self.cap = mode, cap
        self.initialization_by_method = dict.fromkeys(METHODS, 0.)
        self.branches, self.rows, self.bundles, self.previous_update = {}, {}, {}, {}
        config = json.loads((ROOT/'data/module1/online/configurations/diffusion_60/seed_1.json').read_text())
        name, raw_space = next(iter(config["setup"]["configuration_spaces"].items()))
        space = SetupConfigurationSpace.from_mapping(name, raw_space)
        setup_encoder = make_setup_feature_encoder(configuration_space=space, parameter_resolution=20)
        dimension = len(APPLICATION_CONTEXT_FIELDS[mode])
        for method in ("linucb", "joint"):
            tick = time.perf_counter()
            branch, _ = build_online_linucb_branch(seed=seed, action_space_mode="full_cartesian",
                solver_tol=TOL, solver_max_iter=cap, parameter_resolution=20, configuration_space=space,
                candidate_schedule_dir=out/method/'candidates', candidate_schedule_rounds=cases*3,
                candidate_pool_size=1024, candidate_sampling="structured512", context_dim=dimension,
                context_interaction_indices=tuple(range(1, dimension)),
                structured_quotas=config["setup"]["candidate_schedule"]["structured_quotas"])
            branch.policy.model = FailureAwareRetryLinUCB.from_runtime_model(branch.policy.model)
            self.branches[method] = branch
            self.rows[method] = CandidateRows(branch.policy.model, cases=cases, attempts=3, directory=out/method)
            self.previous_update[method] = 0.
            self.initialization_by_method[method] += time.perf_counter()-tick
        for method in ("lstdq", "joint"):
            tick = time.perf_counter()
            request = OnlineControllerFactoryRequest(
                state=SolveStateSpec(tol=TOL, max_cycles=cap, c_max=1000., problem_context_mode=mode),
                actions=SharedActionSpec(weights=tuple(np.linspace(1.,3.,41)),
                    rbf_centers=tuple(np.linspace(1.,3.,9)), rbf_sigma=.2,
                    epsilon=EpsilonScheduleSpec(start=.3, final=.03, decay_steps=20000.)),
                algorithm=RecursiveLstdqSpec(ridge=1., uncertainty_beta=2., residual_floor_sec=.001,
                                             lcb_lower_bound_sec=0.),
                trace_lambda=.8, setup_obs_encoder=setup_encoder, seed=seed+2018)
            self.bundles[method] = build_recursive_lstdq_controller(request)
            self.initialization_by_method[method] += time.perf_counter()-tick
        self.pcg = {"pcg_rebuild": Pcg(), "pcg_reuse": Pcg(True)}
        self.rescue, self.reference = Pcg(), Pcg()
        self.config = config

    def solve(self, method, index, A, b, x0, context, context_sec):
        started = time.perf_counter()
        if method in self.pcg:
            x, raw = self.pcg[method].solve(A, b, x0, cap=self.cap)
            result = dict(primary_converged=raw["converged"], raw=raw,
                          setup_sec=raw["setup_sec"], solve_sec=raw["solve_sec"],
                          iterations=raw["iterations"], builds=raw["builds"],
                          first_attempt_failed=not raw["converged"], true_residual=raw["true_residual"])
        else:
            initial = true_residual(A,b,x0)
            if initial <= TOL:
                return dict(method=method, case=index+1, primary_converged=True,
                    final_converged=True, true_residual=initial, iterations=0, builds=0,
                    setup_sec=0., solve_sec=0., first_attempt_failed=False,
                    rescue_used=False, context_sec=0., total_wall_sec=time.perf_counter()-started,
                    skipped_already_converged=True)
            context = validate_application_context(context, self.mode)
            x = np.empty_like(x0)
            mkw = dict(matrix=A, rhs=b, initial_guess=x0, solution_out=x)
            bundle = self.bundles.get(method)
            tick = time.perf_counter()
            recorder = EpisodeRecorder(bundle.controller) if bundle else None
            snapshot_sec = time.perf_counter()-tick if bundle else 0.

            def attempt(params, rl=False):
                if rl:
                    raw = run_td_episode(mkw=mkw, params=dict(params), controller=recorder,
                        encoder=bundle.encoder, problem_context=context, solve_tol=TOL,
                        solve_max_cycles=self.cap, learn=True, explore=True,
                        record_action_metadata=True, fallback_attempt=None, failure_penalty_sec=0.)
                    raw["recovery_protocol_applied"] = False
                else:
                    with create_env(**mkw) as env:
                        res = env.solve(params=augment_setup_params(dict(params)), tol=TOL, max_iter=self.cap)
                    raw = dict(runtime=res.runtime_sec, setup_runtime=res.setup_runtime_sec,
                               solve_runtime=res.solve_runtime_sec, infer_runtime=0., iterations=res.iterations)
                r = true_residual(A,b,x)
                failed = not np.isfinite(r) or r > TOL
                raw.update(residual_norm=r, failed=failed,
                    failure_reason="external_residual_above_tolerance" if failed else "",
                    attempt_status="nonconvergence" if failed else "success")
                return raw

            fallback = lambda: attempt(DEFAULT_SETUP_PARAMS)
            if method in self.branches:
                branch = self.branches[method]
                outcome = run_deferred_case(policy=branch.policy, parameter_space=branch.parameter_space,
                    context=context, solver=lambda params: attempt(params, bundle is not None),
                    fallback_solver=fallback, variant=VARIANT, previous_update=self.previous_update[method],
                    before_attempt=lambda retry: self.rows[method].before_attempt(index, retry),
                    feedback_mode="adjacent_delayed", recorder=recorder, snapshot_runtime=snapshot_sec)
                self.rows[method].finish_case(index)
                self.previous_update[method] = outcome.update_runtime_sec
                raw = outcome.outcome
                raw["bandit_timing"] = outcome.timing
                raw["selected_params"] = outcome.params
            elif method == "lstdq":
                raw = run_fixed_case(lambda: attempt(DEFAULT_SETUP_PARAMS, True), recorder,
                    snapshot_sec, False, fallback_solver=fallback, max_attempts=1)
            else:
                raw = fallback()
            result = dict(primary_converged=not raw["failed"], raw=raw,
                setup_sec=raw["setup_runtime"], solve_sec=raw.get("native_solve_runtime", raw["solve_runtime"]),
                iterations=raw["iterations"],
                builds=len(raw.get("primary_attempts", [None]))+int(raw.get("fallback_used", False)),
                first_attempt_failed=(raw.get("first_primary_status",raw.get("primary_status","success"))!="success"
                                      if method != "default" else raw["failed"]),
                true_residual=true_residual(A,b,x))
        result["rescue_used"] = not result["primary_converged"]
        if result["rescue_used"]:
            x, rescue = self.rescue.solve(A, b, x0, tol=TOL*.1, cap=self.cap)
            result["rescue"] = rescue
            result["true_residual"] = rescue["true_residual"]
        result["final_converged"] = bool(np.isfinite(result["true_residual"]) and result["true_residual"]<=TOL)
        result.update(method=method, case=index+1,
            context_sec=context_sec if method in ("lstdq","linucb","joint") else 0.)
        result["total_wall_sec"] = time.perf_counter()-started+result["context_sec"]
        return result

    def close(self):
        for solver in (*self.pcg.values(), self.rescue, self.reference):
            solver.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--application',choices=['spe10','heat'],required=True)
    parser.add_argument('--cases',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20261104)
    parser.add_argument('--max-cycles',type=int,default=100)
    parser.add_argument('--steps-per-scenario',type=int,default=100,
                        help='100 for measured campaigns; smaller values only for integration checks')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.steps_per_scenario<=100 or not 1<=args.cases<=10*args.steps_per_scenario:
        parser.error('Use at most ten distinct scenarios, with at most 100 physical steps each.')
    if args.max_cycles != 100:
        parser.error('The current experiment requires the user-selected 100-cycle ceiling.')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    configure_smoother_profile('l1_jacobi_direct_coarse')
    tick=time.perf_counter()
    with create_env(matrix=csr_matrix([[1.]]),rhs=np.ones(1)):
        pass
    common_initialization=time.perf_counter()-tick
    mode='spe10_pressure' if args.application=='spe10' else 'mfem_heat'
    campaign=scenarios(args.application,args.seed)
    (out/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
    if args.application=='heat':
        (out/'campaign.tsv').write_text(''.join(' '.join(str(s[k]) for k in
            ('kappa','alpha','dt','hot_amplitude'))+'\n' for s in campaign))
    tick=time.perf_counter();study=Study(mode,out,args.cases,args.seed,args.max_cycles)
    startup=time.perf_counter()-tick
    source_paths=list((ROOT/'experiments/application_online').glob('*'))+[
        ROOT/'hypre/interfaces/amg_runtime.c',ROOT/'hypre/bindings/boomeramg.py',
        ROOT/'hypre/interfaces/libamg_runtime.dylib',ROOT/'problems/application_context.py',
        ROOT/'hypre/interfaces/amg_runtime.h',ROOT/'results/application_default_100/heat_stream',
        ROOT/'solve/controllers/common/state_encoder.py',ROOT/'solve/controllers/common/factory.py']
    protocol=dict(application=args.application,cases=args.cases,seed=args.seed,tolerance=TOL,
        max_cycles=args.max_cycles,methods=METHODS,initialization_sec=startup,
        steps_per_scenario=args.steps_per_scenario,campaign=campaign,
        rescue_max_iterations=args.max_cycles,reference_max_iterations=args.max_cycles,
        common_initialization_sec=common_initialization,
        initialization_by_method=study.initialization_by_method,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths if p.is_file()},
        context_fields=APPLICATION_CONTEXT_FIELDS[mode],
        learning='active throughout; fresh independent learners; current paper actions and recovery',
        baseline='PCG + HMIS/ext+i AMG, symmetric hybrid GS, direct coarse; fixed in advance, not calibrated',
        reuse='rebuild after age 10 or previous iterations > 2x iterations immediately after last build',
        timing='serial method wall time including import, checks, learning, retries, rescue and used context extraction; initialization separate',
        scope='ten distinct physical trajectories; one learner seed; learning persists across trajectories',
        physical_state='reset at scenario boundaries; common reference warm starts within trajectories',
        integration_check=args.steps_per_scenario!=100)
    dependency_provenance=json.loads((ROOT/'results/application_default_100/summary.json').read_text())['provenance']
    protocol['dependencies']={k:dependency_provenance[k] for k in
        ('mfem_commit','mrst_commit','hypre_version','mfem_version','octave_version')}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with zipfile.ZipFile(out/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in protocol['source_sha256']:archive.write(ROOT/relative,relative)
        archive.write(out/'protocol.json','protocol.json')
    log=(out/'application.log').open('w')
    if args.application=='spe10':
        octave=ROOT/'results/application_default_100/octave_env/bin/octave-cli'
        env=dict(os.environ,OCTAVE_HOME=str(octave.parent.parent),OCTAVE_EXEC_HOME=str(octave.parent.parent))
        # Escape paths as Octave string literals, passed as one argv argument.
        quote=lambda s: "'"+str(s).replace("'","''")+"'"
        expression=f"addpath({quote(ROOT/'experiments/application_online')});spe10_stream({quote(ROOT)},{quote(out)},{args.cases},{args.steps_per_scenario});"
        command=[str(octave),'--no-gui','--quiet','--eval',expression]
    else:
        env=os.environ.copy()
        command=[str(ROOT/'results/application_default_100/heat_stream'),
                 str(ROOT/'results/application_default_100/heat.mesh'),str(out),str(args.cases),
                 str(args.steps_per_scenario),str(out/'campaign.tsv')]
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    orders=balanced_orders(args.cases,args.seed,methods=METHODS)
    previous=None; records=[]; started=time.perf_counter()
    try:
        for index in range(args.cases):
            while not (out/'ready').exists():
                if process.poll() is not None:
                    raise RuntimeError(f'Application exited {process.returncode}; inspect {out}/application.log')
                time.sleep(.05)
            with (out/'request.bin').open('rb') as f:
                n,nnz=np.fromfile(f,np.int32,2)
                ia=np.fromfile(f,np.int32,n+1);ja=np.fromfile(f,np.int32,nnz)
                av=np.fromfile(f,np.float64,nnz);b=np.fromfile(f,np.float64,n)
                x0=np.fromfile(f,np.float64,n) if args.application=='heat' else (
                    np.zeros(n) if index % args.steps_per_scenario==0 else previous)
            A=csr_matrix((av,ja,ia),shape=(n,n))
            metadata=json.loads((out/'context.json').read_text())
            assert metadata['scenario']==index//args.steps_per_scenario+1
            assert metadata['scenario_step']==index%args.steps_per_scenario+1
            assert x0.shape==(n,) and np.isfinite(b).all()
            assert np.max(np.abs((A-A.T).data),initial=0.)<1e-11*np.max(abs(A.data))
            digest=hashlib.sha256((out/'request.bin').read_bytes()).hexdigest()
            for position,method in enumerate(orders[index]):
                row=study.solve(method,index,A,b,x0,metadata['context'],metadata['context_sec'])
                row.update(n=int(n),nnz=int(nnz),input_sha256=digest,position=position,context=metadata['context'],
                    scenario=metadata['scenario'],scenario_step=metadata['scenario_step'],max_cycles=args.max_cycles)
                with (out/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                records.append(row)
                print(json.dumps({k:row[k] for k in ('case','method','total_wall_sec','first_attempt_failed','rescue_used','final_converged')}),flush=True)
                if not row['final_converged']:raise RuntimeError('Rescue failed; workload not completed')
            reference,check=study.reference.solve(A,b,x0,tol=1e-10,cap=args.max_cycles)
            if not np.isfinite(check['true_residual']) or check['true_residual']>1e-9:
                raise RuntimeError(f'Reference solve failed: {check}')
            with (out/'reference.jsonl').open('a') as f:f.write(json.dumps(dict(case=index+1,**check))+'\n')
            previous=reference;reference.tofile(out/'response.bin')
            (out/'ready').unlink();(out/'response.ready').touch()
            progress=dict(completed_systems=index+1,total_systems=args.cases,
                method_cases=len(records),scenario=metadata['scenario'],scenario_step=metadata['scenario_step'],
                elapsed_sec=time.perf_counter()-started,
                rescue_calls=sum(row['rescue_used'] for row in records),
                final_failures=sum(not row['final_converged'] for row in records))
            (out/'progress.tmp').write_text(json.dumps(progress)+'\n')
            (out/'progress.tmp').replace(out/'progress.json')
            if (index+1)%args.steps_per_scenario==0:
                checkpoint=out/f'checkpoint_{index+1:04d}';checkpoint.mkdir()
                for name,bundle in study.bundles.items():bundle.save(checkpoint/f'{name}_controller.npz')
                for name,branch in study.branches.items():branch.policy.model.save_mutable_state(checkpoint/f'{name}_setup.npz')
        if process.wait()!=0:raise RuntimeError('Application failed after final response')
        summary=dict(complete=True,initialization_sec=startup,elapsed_sec=time.perf_counter()-started,methods={})
        for method in METHODS:
            rows=[r for r in records if r['method']==method]
            summary['methods'][method]=dict(cases=len(rows),
                mean_total_wall_sec=float(np.mean([r['total_wall_sec'] for r in rows])),
                total_wall_sec=sum(r['total_wall_sec'] for r in rows),
                primary_failures=sum(not r['primary_converged'] for r in rows),
                first_attempt_failures=sum(r['first_attempt_failed'] for r in rows),
                rescues=sum(r['rescue_used'] for r in rows),
                final_failures=sum(not r['final_converged'] for r in rows),
                hierarchy_builds=sum(r['builds'] for r in rows))
            summary['methods'][method]['cold_start_total_sec']=(
                summary['methods'][method]['total_wall_sec']+common_initialization+
                study.initialization_by_method[method])
        for method,bundle in study.bundles.items():
            bundle.save(out/f'{method}_controller.npz')
            summary['methods'][method]['controller']=bundle.summary()
        for method,branch in study.branches.items():
            summary['methods'][method]['setup_observations']=branch.policy.model.t
            branch.policy.model.save_mutable_state(out/f'{method}_setup.npz')
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    finally:
        if process.poll() is None:process.terminate();process.wait()
        study.close();log.close()


if __name__=='__main__':
    main()
