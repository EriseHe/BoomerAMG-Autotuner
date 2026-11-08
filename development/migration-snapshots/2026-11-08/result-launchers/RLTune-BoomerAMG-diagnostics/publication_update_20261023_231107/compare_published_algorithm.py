"""Export deterministic encoder and learner behavior from an isolated source tree."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from experiments.paper_final import run_online as runner
from experiments.paper_final.online import methods
from setup.learners.common.aot_candidates import CompactActionCatalog

parser = argparse.ArgumentParser()
parser.add_argument('--configs', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--reference', action='store_true')
options = parser.parse_args()
options.output.mkdir(parents=True, exist_ok=True)
if options.reference:
    from development.run_joint_context import build_solve_runtime
else:
    build_solve_runtime = methods.build_composable_solve_runtime

arrays = {}
metadata = {}
for family in ('diffusion_40', 'advection_40', 'variable_flow_60'):
    module = 'module3' if family == 'variable_flow_60' else 'module1'
    raw = runner.read(options.configs/module/family/'seed_1.json')
    raw.pop('frozen_stream', None)
    directory = options.output/family
    directory.mkdir()
    raw['output_dir'] = str(directory)
    runtime = runner.decode(raw, directory)
    built = (build_solve_runtime(runtime.args, specs=runtime.spec.methods, context_variant='log_time')
             if options.reference else build_solve_runtime(runtime.args, specs=runtime.spec.methods))
    metadata[family] = {}
    for name, bundle in built.controller_bundles.items():
        key = family+'__'+name
        model, encoder = bundle.controller, bundle.encoder
        catalog = CompactActionCatalog(encoder.setup_obs_encoder.parameter_spec)
        rng = np.random.default_rng(771)
        states = []
        for i in range(204):
            mkw, context = runtime.stream[i % 12]
            cycle = i % 51
            params = dict(catalog[int(rng.integers(len(catalog)))])
            residual = 10. ** (-9. + (i % 19))
            previous = 10. ** (-8. + (i % 17))
            seconds = (0., 1e-9, .001, .01, .1, 1., 1000.)[i % 7]
            observation = dict(mkw=mkw, problem_context=context, setup_params=params,
                initial_residual=1., residual=residual, previous_residual=previous,
                cycle=cycle, last_weight=1.+(i % 41)*.05, last_cycle_time=seconds)
            if options.reference:
                observation['first_cycle_time'] = None if cycle == 0 else .123
            states.append(encoder.encode(**observation))
        arrays[key+'__states'] = np.asarray(states)
        # Replay identical observations and costs through the actual controller,
        # including action selection, exploration, updates and episode commits.
        actions, predictions = [], []
        for episode in range(12):
            model.start_episode(initial_environment_weight=1.)
            for cycle in range(6):
                current = states[episode*6+cycle]
                following = states[episode*6+cycle+1]
                predictions.append(np.asarray(model._values(current, cycle=cycle)))
                action, _, _ = model.select_action(current, cycle=cycle, explore=True)
                actions.append(action)
                model.update(features=current, action_index=action,
                    cost=.001 + .0001*cycle, next_features=following,
                    terminal=cycle == 5, next_cycle=cycle+1,
                    next_action_index=(action+1) % len(model.config.weights))
            model.finish_episode(learned=True)
        arrays[key+'__actions'] = np.asarray(actions)
        arrays[key+'__predictions'] = np.asarray(predictions)
        for field in ('theta','a_matrix','a_inverse','b','episode_moment_covariance'):
            arrays[key+'__'+field] = getattr(model, field).copy()
        metadata[family][name] = bundle.protocol_metadata()['state_encoder']
    # The setup learner remains separately encoded. Compare its full initial
    # states, candidate rows and deterministic delayed-feedback updates too.
    args = replace(runtime.args, online_cases=12)
    study = methods.resolve_composable_study(args)
    if options.reference:
        setup = methods.build_named_setup_branches(args, study=study,
            warmup_instances=(), stream_hash=runtime.manifest['sha256']).branches
    else:
        setup = methods.build_named_setup_branches(args, study=study)
    for name, branch in setup.items():
        from experiments.paper_final.recovery.setup_feedback import FailureAwareRetryLinUCB
        from problems.registry import context_for_setup_method
        model = FailureAwareRetryLinUCB.from_runtime_model(branch.policy.model)
        method = next(m for m in runtime.spec.methods if m.name == name)
        selected = []
        candidate_rows = runner.CandidateRows(model, cases=12, attempts=3, directory=directory/'unused')
        for i in range(12):
            mkw, context = runtime.stream[i]
            x = context_for_setup_method(problem_kind=args.problem,
                setup_kind=method.setup_kind, setup_context=method.setup_context,
                matrix_kwargs=mkw, stream_context=context)
            observations = []
            successes = []
            for attempt in range(3):
                candidate_rows.before_attempt(i, attempt)
                if attempt == 0:
                    model.predict(x)
                else:
                    model.select_recovery(x, excluded_arms=selected[-attempt:])
                selected.append(int(model._last_arm))
                succeeded = i % 5 != 0 and attempt == i % 3
                observations.append(model.defer_cost(succeeded))
                successes.append(succeeded)
                if succeeded:
                    break
            costs = [.02+.01*j for j in range(len(observations))]
            sequence = costs if successes[-1] else costs+[.05]
            for j, observation in enumerate(observations):
                target = costs[j] if successes[j] else sum(sequence[j:j+2])
                model.resolve_cost(observation, target)
            if not successes[-1] and i != 5:
                model.update_default_fallback(x, runner.DEFAULT_SETUP_PARAMS, .05)
            candidate_rows.finish_case(i)
        key = family+'__'+name+'__setup'
        arrays[key+'__arms'] = np.asarray(selected)
        arrays[key+'__candidates'] = np.asarray(candidate_rows.base._ids).copy()
        arrays[key+'__count'] = np.asarray(model.t)
        for field in ('A_inv', 'b', 'failure_b'):
            arrays[key+'__'+field] = getattr(model, field).copy()
        arrays[key+'__history'] = np.asarray([row.loss for row in model.history])

np.savez_compressed(options.output/'behavior.npz', **arrays)
(options.output/'encoder_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps({'arrays':len(arrays), 'families':list(metadata), 'reference':options.reference}))
