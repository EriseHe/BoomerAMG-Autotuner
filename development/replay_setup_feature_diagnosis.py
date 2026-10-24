"""Fixed-data early-trajectory ablations; no solver or policy rollout is run."""
import json

import numpy as np

from development.diagnose_shared_setup_features import CURRENT, OUTPUT
from development.tests.test_shared_setup_features import encoders
from solve.controllers.common.state_encoder import SolveStateEncoder
from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig
from solve.controllers.recursive_lstdq import RecursiveLstdqController, RecursiveLstdqSpec


def encode_states(row, attempt, encoder):
    actions=attempt['cycle_actions']
    residuals=[1.,*attempt['cycle_residuals']]
    times=[0.,*attempt['cycle_times']]
    weights=[1.,*actions]
    return [encoder.encode(mkw=row['mkw'],setup_params=attempt['params'],problem_context=row['context'],
        initial_residual=1.,residual=residuals[k],previous_residual=residuals[max(0,k-1)],cycle=k,
        last_weight=weights[k],last_cycle_time=times[k],first_cycle_time=times[1])
        for k in range(len(actions))]


class NumericCap:
    """Diagnostic only: clip five base numerical features before polynomial expansion."""
    def __init__(self, shared, *, normalize=False):
        self.shared=shared
        self.parameter_spec=shared.parameter_spec
        self.descriptor_dim=shared.descriptor_dim
        self.normalize=normalize
        numeric=[p for p in self.parameter_spec.parameters if p.kind!='categorical']
        self.numeric_bounds=np.array([max(abs((float(v)-p.center)/p.scale) for v in p.values) for p in numeric])

    def encode_setup(self, params):
        v=self.shared.encode_setup(params).copy()
        n=v[:5]/self.numeric_bounds if self.normalize else np.clip(v[:5],-2.,2.)
        v[:5]=n
        v[5:10]=n*n
        v[10:20]=[n[i]*n[j] for i in range(5) for j in range(i+1,5)]
        v[34:]=(v[20:34,None]*n[None,:]).ravel()
        return v


def run(seed):
    stage=CURRENT/f'seed_{seed}'
    with np.load(stage/'initial_controllers/bandit_lstdq.npz') as initial:
        config=ExpectedSarsaLambdaConfig(**json.loads(initial['config'].item()))
        spec=RecursiveLstdqSpec(**json.loads(initial['spec'].item()))
        assert np.count_nonzero(initial['theta'])==np.count_nonzero(initial['b'])==0
    with (stage/'trajectories/bandit_lstdq.jsonl').open() as f:
        rows=[json.loads(next(f)) for _ in range(3)]
    basic,shared=encoders()
    options=('current','numeric_setup_cap','numeric_setup_normalization','basic_setup',
             'residual_gap_cap','absolute_cycle_time','cycle_bins')
    result={}
    for option in options:
        setup=(NumericCap(shared, normalize=option=='numeric_setup_normalization')
               if option in ('numeric_setup_cap','numeric_setup_normalization') else
               basic if option=='basic_setup' else shared)
        encoder=SolveStateEncoder(tol=1e-6,max_cycles=50,c_max=1000,mode='setup_full',
            setup_obs_encoder=setup,problem_context_mode='canonical_no_c_mean',weight_bounds=(1.,3.))
        dim=encoder.feature_dim+(14 if option=='cycle_bins' else 0)
        controller=RecursiveLstdqController(feature_dim=dim,config=config,spec=spec,seed=1)
        differences=[]; td_all=[]; cases=[]
        for row in rows:
            case_errors=[]; terminal=[]
            assert not row['outcome']['unrecovered_failure']
            for attempt in row['outcome']['primary_attempts']:
                assert attempt['rl_episode_recorded']
                states=encode_states(row,attempt,encoder)
                for k,state in enumerate(states):
                    if option=='residual_gap_cap':
                        state[2]=np.clip(state[2],0.,1.5); state[3]=state[2]**2
                    elif option=='absolute_cycle_time':
                        state[5]=0. if k==0 else attempt['cycle_times'][k-1]/.01
                    elif option=='cycle_bins':
                        indicator=np.eye(7)[np.searchsorted([1,2,3,5,8,12],k,side='right')]
                        states[k]=np.r_[state[:13],indicator,state[2]*indicator,state[13:]]
                indices=[int(np.argmin(np.abs(controller.weights-w))) for w in attempt['cycle_actions']]
                controller.start_episode(initial_environment_weight=1.)
                for k,state in enumerate(states):
                    last=k==len(states)-1
                    value=controller.update(features=state,action_index=indices[k],
                        cost=attempt['cycle_times'][k],next_features=state if last else states[k+1],
                        terminal=last,next_cycle=k+1,next_action_index=None if last else indices[k+1])
                    case_errors.append(abs(value));td_all.append(abs(value))
                    differences.append(abs(value-attempt['td_errors'][k]))
                    if last:terminal.append(value)
                controller.finish_episode(learned=True)
            cases.append(dict(input=row['problem_index'],max_abs_td=max(case_errors),terminal_td=terminal))
        if option=='current':
            assert max(differences)<1e-5, max(differences)
        result[option]=dict(max_abs_td=max(td_all),td_above_1=sum(x>1 for x in td_all),
            per_input=cases,recorded_baseline_max_abs_error=max(differences) if option=='current' else None,
            inverse_rebuilds=controller.inverse_rebuild_count)
    return dict(seed=seed,variants=result)


def main():
    result=dict(scope='First three recorded problems, current joint method; fixed actions, native costs, residuals, and setup selections.',
        native_solves_executed=0,closed_loop_performance_test=False,
        normalization='Divide each centered numerical descriptor by its largest allowed absolute value before constructing the same polynomial features; default remains zero, one-hot categories unchanged, no clipping.',
        limitation='A change in fixed-data TD errors does not establish a gain in native execution time.',seeds=[])
    for seed in (1,2,3):
        entry=run(seed);result['seeds'].append(entry)
        print(json.dumps(entry),flush=True)
    (OUTPUT/'early_replay.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
