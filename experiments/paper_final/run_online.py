"""Configurable Module 1 four-branch suite, one single-thread worker per seed.

Prepare freezes configs, paired inputs, source and native-library hashes.
Run refuses partial state and releases workers only after all are ready.
All four branches are measured independently from input one. Both setup learners
and both LSTDQ controllers start cold; no prefix, copied solves or state forks.
Module 1 continues through held-out inputs 5,001-6,000 in the same worker,
freezing both learners after input 5,000 and disabling epsilon exploration.
Controller and setup-selection overhead remain measured in both phases.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import gzip
import hashlib
from dataclasses import replace
from types import SimpleNamespace

from experiments.runtime import (configure_single_thread, single_thread_environment,
    prevent_sleep, stop_sleep_prevention, THREAD_KEYS)
configure_single_thread()
import numpy as np

from experiments.paper_final.common.artifacts import (
    ROOT, dump, read, digest, file_hash, ready, now, write_csv, read_records, checkpoint_payload_hash,
)
from experiments.paper_final.common.frozen_inputs import execution_source_paths
from experiments.paper_final.online.configuration import (parse_joint_experiment_config,
    runtime_config_from_spec, FrozenEvaluationSpec, evaluation_index_offset)
from experiments.paper_final.online.case_loop import build_paired_instance_stream, configure_paired_environment
from experiments.paper_final.online.methods import (
    build_named_setup_branches, build_composable_solve_runtime, resolve_composable_study,
    validate_setup_branch_independence,
)
from experiments.paper_final.online import factorial
from experiments.paper_final.recovery.execution import CandidateRows
from experiments.paper_final.recovery.setup_feedback import FailureAwareRetryLinUCB
from experiments.paper_final.recovery import metrics as sweep
from experiments.paper_final.recovery import model_audit as screen
from experiments.paper_final.recovery.observations import observed_model
from experiments.paper_final.recovery.controller_audit import audit_controller_counts
from setup.space import DEFAULT_SETUP_PARAMS
from solve.controllers.common.state_encoder import STATE_ENCODING


def source_hashes():
    return {name: file_hash(ROOT/name) for name in sorted(execution_source_paths())}


def write_frozen_manifest():
    destination = ROOT/"data/module1/frozen"
    files = {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(destination.rglob('*'))
             if p.is_file() and p != destination/"manifest.json"}
    dump(destination/"manifest.json", files)
    inventory_path = ROOT/"data/manifest.json"
    if inventory_path.exists():
        inventory = read(inventory_path)
        inventory["files"] = {name: sha for name, sha in inventory["files"].items()
                              if not name.startswith("data/module1/frozen/")}
        inventory["files"].update(files)
        inventory["files"]["data/module1/frozen/manifest.json"] = file_hash(destination/"manifest.json")
        dump(inventory_path, inventory)


def prepare_frozen():
    """Describe held-out blocks using the official sampler and released models."""
    from experiments.paper_final.common.online_inputs import canonical_stream_bytes, sampling_signature
    from experiments.paper_final.online.case_loop import _problem_stream_spec, _sampled_advection_range
    destination = ROOT/"data/module1/frozen"
    destination.mkdir(exist_ok=False)
    (destination/"inputs").mkdir()
    captures = dict(format_version=1, module=1, phase="frozen", streams={})
    for family in ("diffusion", "advection"):
        for grid in (40, 60, 80):
            experiment = f"{family}_{grid}"
            training_keys = set()
            configurations = []
            for seed in range(1, 7):
                batch = "batch_01" if seed <= 3 else "batch_02"
                source = Path(f"data/official/module1/batches/{batch}/{experiment}/seed_{seed}")
                training_path = Path(f"data/module1/configurations/{experiment}/seed_{seed}.json")
                for row in read_records(ROOT/source/"inputs.jsonl"):
                    training_keys.add(digest(row["mkw"]))
                raw = copy.deepcopy(read(ROOT/training_path))
                raw.update(name=f"module1_frozen_{experiment}_seed{seed}",
                    description="Frozen Module 1 deployment on new inputs 5001-6000; no learner updates.",
                    output_dir=f"results/module1/frozen/{experiment}_{batch}/seed_{seed}",
                    evaluation=dict(checkpoint_directory=str(source), training_configuration=str(training_path),
                                    training_cases=5000, explore=False))
                raw["stream"].update(cases=1000, online_cases=1000, instance_offset=625,
                    cases_per_seed=125, group_take=1000, expected_sha256="")
                raw["reporting"]["progress_every"] = 500
                spec = parse_joint_experiment_config(raw)
                args = runtime_config_from_spec(spec)
                configure_paired_environment(args)
                stream, manifest = build_paired_instance_stream(args)
                payload = canonical_stream_bytes(stream)
                sha = hashlib.sha256(payload).hexdigest()
                assert sha == manifest["sha256"] and len(stream) == 1000
                raw["stream"]["expected_sha256"] = sha
                (destination/"inputs"/f"{sha}.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))
                problem, shape, advection = _problem_stream_spec(args)
                sampled = _sampled_advection_range(args, problem=problem)
                captures["streams"][sha] = dict(file=f"{sha}.jsonl.gz", canonical_sha256=sha,
                    compressed_sha256=file_hash(destination/"inputs"/f"{sha}.jsonl.gz"), cases=1000,
                    sampling_signature=sampling_signature(args, problem=problem, grid=shape, advection=advection,
                                                           sampled_advection_range=sampled), stream_manifest=manifest)
                path = destination/f"configurations/{experiment}/seed_{seed}.json"
                dump(path, raw)
                configurations.append((seed, raw, stream))
            new_keys = set()
            for seed, raw, stream in configurations:
                keys = {digest(mkw) for mkw, _ in stream}
                assert len(keys) == 1000 and not keys.intersection(training_keys) and not keys.intersection(new_keys)
                new_keys.update(keys)
            for batch, seeds in ((1, range(1, 4)), (2, range(4, 7))):
                dump(destination/f"suites/{experiment}_batch{batch}.json", dict(name=f"Module 1: frozen {experiment}",
                     seeds=[dict(label=str(s), config=f"../configurations/{experiment}/seed_{s}.json") for s in seeds]))
            print(f"Module 1 frozen {experiment}: 6,000 held-out inputs; no overlap with any of the six training streams.", flush=True)
    dump(destination/"inputs/manifest.json", captures)
    dump(destination/"protocol.json", dict(module=1, phase="frozen", training_module=1, training_cases=5000,
         evaluation_global_cases=[5001, 6000], cases_per_seed=1000, seeds=list(range(1,7)),
         families=["diffusion", "advection"], grids=[40,60,80], methods=factorial.METHODS,
         model_updates=False, exploration=False, setup_confidence_rule="unchanged trained LinUCB LCB and failure-aware retries",
         sampling="Next 125 child instances per each of the eight original streams, window [625,750), shuffled by the original seed",
         native_sha256=sweep.native_hashes(), original_main_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()))
    write_frozen_manifest()


def decode(raw, directory):
    spec = parse_joint_experiment_config(raw, output_dir_override=directory)
    factorial.validate_methods(spec)
    args = runtime_config_from_spec(spec)
    configure_paired_environment(args)
    stream, manifest = build_paired_instance_stream(args)
    if not args.expected_stream_hash or args.expected_stream_hash != manifest["sha256"]:
        raise ValueError("A pinned, matching input stream hash is required")
    freeze_after = None
    if spec.frozen_stream is not None:
        frozen_spec = replace(spec, stream=spec.frozen_stream, frozen_stream=None)
        frozen_args = runtime_config_from_spec(frozen_spec)
        configure_paired_environment(frozen_args)
        heldout, heldout_manifest = build_paired_instance_stream(frozen_args)
        if not frozen_args.expected_stream_hash or frozen_args.expected_stream_hash != heldout_manifest["sha256"]:
            raise ValueError("A pinned, matching frozen input stream hash is required")
        old_keys = {digest(mkw) for mkw, _ in stream}
        new_keys = {digest(mkw) for mkw, _ in heldout}
        if len(stream) != 5000 or len(heldout) != 1000 or len(new_keys) != 1000 or old_keys.intersection(new_keys):
            raise ValueError("Continuous frozen inputs must be new, unique held-out problems")
        freeze_after = len(stream)
        stream = stream + heldout
        from experiments.paper_final.common.online_inputs import canonical_stream_bytes
        manifest = dict(sha256=hashlib.sha256(canonical_stream_bytes(stream)).hexdigest(),
            training_stream=manifest, frozen_stream=heldout_manifest, training_cases=freeze_after,
            cases=len(stream), continuous=True)
        args = replace(args, online_cases=len(stream), continuous_stream=True)
    return SimpleNamespace(spec=spec, args=args, stream=stream, manifest=manifest, freeze_after=freeze_after)


def input_rows(runtime, cases):
    offset = evaluation_index_offset(runtime.args)
    return [dict(problem_index=offset+i+1, mkw=mkw, context=context.tolist())
            for i, (mkw, context) in enumerate(runtime.stream[:cases])]


def training_inventory(source):
    """Read pinned released artifacts or the audited output of a fresh full run."""
    if source.is_relative_to((ROOT/"data/official").resolve()):
        return read(ROOT/"data/raw_files.json")["files"]
    output = source.parent
    protocol = read(output/"protocol.json")
    result = read(source/"result.json")
    label = source.name.removeprefix("seed_")
    checks = protocol["stages"][label]
    if (read(output/"status.json")["status"] != "complete"
            or read(source/"status.json")["status"] != "complete"
            or not result["complete"] or not result["passed"]
            or result["cases"] != 5000 or checks["cases"] != 5000
            or protocol.get("frozen_evaluation")):
        raise ValueError("Frozen evaluation requires a complete audited 5,000-input training run")
    model_prefixes = ("setup/", "solve/", "problems/", "hypre/")
    model_sources = lambda hashes: {k:v for k,v in hashes.items() if k.startswith(model_prefixes)}
    training_models = model_sources(read(output/"source_manifest.json"))
    if (not training_models or training_models != model_sources(source_hashes())
            or protocol["native_sha256"] != sweep.native_hashes()):
        raise ValueError("Frozen evaluation must use the training source for models and the native build")
    if file_hash(source/"experiment_config.json") != checks["config_sha256"]:
        raise ValueError("Training configuration changed")
    if digest(list(read_records(source/"inputs.jsonl"))) != checks["inputs_sha256"]:
        raise ValueError("Training inputs changed")
    hashes = dict(result["checkpoint_sha256"])
    for name in factorial.METHODS:
        audit = result["audits"][name]
        if audit["cases"] != 5000:
            raise ValueError("Training branch is incomplete")
        hashes[f"trajectories/{name}.jsonl"] = audit["trajectory_sha256"]
    return {str((source/name).relative_to(ROOT.resolve())): {"sha256": sha}
            for name, sha in hashes.items()}


def prepare_evaluation(runtime, stage):
    """Pin released training artifacts and prove evaluation inputs are disjoint."""
    evaluation = runtime.args.evaluation
    source = (ROOT/evaluation.checkpoint_directory).resolve()
    training = read(ROOT/evaluation.training_configuration)
    raw = read(stage/"experiment_config.json")
    for key in ("problem", "setup", "solve", "methods", "seeds", "failure_feedback"):
        if training[key] != raw[key]:
            raise ValueError(f"Frozen evaluation changed the Module 1 {key} settings")
    if sweep.native_hashes() != read(ROOT/"data/module1/protocol.json")["native_sha256"]:
        raise ValueError("Frozen evaluation requires the matching Module 1 native build")
    inventory = training_inventory(source)
    sources, histories = {}, {}
    for name in factorial.METHODS:
        relative = (f"final_bandit_states/{name}.npz" if name in factorial.METHODS[2:] else None)
        paths = ([relative] if relative else [])
        if name in (factorial.METHODS[1], factorial.METHODS[3]):
            paths += [f"checkpoints/{name}_final.npz", f"checkpoints/{name}_final.npz.encoder.json"]
        for relative in paths:
            origin = source/relative
            key = str(origin.relative_to(ROOT.resolve()))
            expected = inventory[key]["sha256"]
            if file_hash(origin) != expected:
                raise ValueError(f"Released checkpoint changed: {origin}")
            destination = stage/"training_checkpoints"/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
            sources[relative] = dict(source=key, sha256=expected)
        if name in factorial.METHODS[2:]:
            path = source/f"trajectories/{name}.jsonl"
            key = str(path.relative_to(ROOT.resolve()))
            if file_hash(path) != inventory[key]["sha256"]:
                raise ValueError("Training trajectory changed")
            with path.open('rb') as stream:
                size = stream.seek(0, 2)
                window = min(size, 1024*1024)
                while True:
                    stream.seek(size-window)
                    lines = stream.read().splitlines()
                    if window == size or len(lines) >= 2:
                        last = json.loads(lines[-1]); break
                    window = min(size, 2*window)
            if last["problem_index"] != evaluation.training_cases:
                raise ValueError("Training trajectory is incomplete")
            outcome = last["outcome"]
            last_observation = outcome["primary_attempts"][-1]
            if outcome.get("fallback_learning_cost") is not None:
                last_observation = dict(arm_index=outcome["fallback_arm_index"],
                                        learning_cost=outcome["fallback_learning_cost"])
            histories[name] = dict(last_attempt=last_observation,
                                   trajectory=key, trajectory_sha256=inventory[key]["sha256"])
    training_inputs = list(read_records(source/"inputs.jsonl"))
    old = {digest(r["mkw"]) for r in training_inputs}
    new = [digest(mkw) for mkw, _ in runtime.stream]
    if len(training_inputs) != 5000 or len(set(new)) != len(new) or old.intersection(new):
        raise ValueError("Evaluation inputs repeat training inputs")
    dump(stage/"checkpoint_sources.json", sources)
    dump(stage/"training_history.json", histories)
    dump(stage/"heldout_audit.json", dict(passed=True, training_cases=5000, evaluation_cases=len(new),
         no_training_overlap=True, no_evaluation_duplicates=True, training_inputs_sha256=file_hash(source/"inputs.jsonl")))
    return dict(frozen_evaluation=True, checkpoint_sources_sha256=file_hash(stage/"checkpoint_sources.json"),
                training_history_sha256=file_hash(stage/"training_history.json"))


def reject_frozen_update(*args, **kwargs):
    raise RuntimeError("A learner update was attempted during frozen Module 1 evaluation")


def frozen_state_audit(stage, runtimes, *, continuous=False):
    """Compare every learned checkpoint payload before and after evaluation."""
    audit = {}
    for name, runtime in runtimes.items():
        pairs = []
        if runtime.bundle is not None:
            prefix = "frozen_initial_controllers" if continuous else "initial_controllers"
            pairs.append((stage/f"{prefix}/{name}.npz", stage/f"checkpoints/{name}_final.npz",
                          ("rng_state", "default_first_action_pending", "forced_initial_action_count")))
        if hasattr(runtime, "branch"):
            prefix = "frozen_initial_bandit_states" if continuous else "initial_bandit_states"
            pairs.append((stage/f"{prefix}/{name}.npz", stage/f"final_bandit_states/{name}.npz",
                          ("rng_state", "candidate_schedule_cursor", "metadata")))
        for initial, final, excluded in pairs:
            before = checkpoint_payload_hash(initial, exclude=excluded)
            after = checkpoint_payload_hash(final, exclude=excluded)
            relative = (f"checkpoints/{name}_final.npz" if 'initial_controllers' in str(initial)
                        else f"final_bandit_states/{name}.npz")
            released = before if continuous else checkpoint_payload_hash(stage/"training_checkpoints"/relative, exclude=excluded)
            if before != released:
                raise ValueError(f"Initial learner differs from the released Module 1 checkpoint: {name}")
            if before != after:
                raise ValueError(f"Frozen learner state changed: {name}")
            audit[str(final.relative_to(stage))] = dict(passed=True, learned_state_sha256=after,
                excluded_execution_state=list(excluded))
    return dict(passed=True, updates=0, checkpoints=audit)


def freeze_live_runtimes(stage, runtimes, training_cases):
    """Disable learning in place, preserving models, RNG, history and caches."""
    identities = {}
    for name, runtime in runtimes.items():
        identities[name] = dict(runtime=id(runtime), controller=id(runtime.bundle.controller) if runtime.bundle else None,
                                setup_model=id(runtime.branch.policy.model) if hasattr(runtime, "branch") else None)
        runtime.args = replace(runtime.args, evaluation=FrozenEvaluationSpec(stage, stage/"experiment_config.json",
                               training_cases=training_cases, explore=False))
        if hasattr(runtime, "branch"):
            model = runtime.branch.policy.model
            if model._last_arm is not None or model._last_phi is not None:
                raise ValueError("Unresolved setup observation at the freeze boundary")
            directory = stage/"frozen_initial_bandit_states"
            directory.mkdir(exist_ok=True)
            model.save_mutable_state(directory/f"{name}.npz")
            for field in ("update", "defer_cost", "resolve_cost", "update_default_fallback"):
                setattr(model, field, reject_frozen_update)
            for field in ("A_inv", "b", "failure_b"):
                getattr(model, field).setflags(write=False)
        if runtime.bundle is not None:
            directory = stage/"frozen_initial_controllers"
            directory.mkdir(exist_ok=True)
            runtime.bundle.save(directory/f"{name}.npz")
            runtime.bundle.controller.update = reject_frozen_update
            for field in ("theta", "a_matrix", "a_inverse", "b", "episode_moment_covariance"):
                getattr(runtime.bundle.controller, field).setflags(write=False)
    dump(stage/"freeze_boundary.json", dict(passed=True, pid=os.getpid(), after_input=training_cases,
         next_input=training_cases+1, models_reinitialized=False, checkpoints_reloaded=False,
         model_updates=False, epsilon_exploration=False, measure_overhead=True, identities=identities))


def prepare_all(suite_directory, output, *, cases=None):
    output.mkdir(parents=True, exist_ok=False)
    frozen = suite_directory.resolve() == (ROOT/"data/module1/frozen/suites").resolve()
    if frozen:
        dump(ROOT/"data/module1/frozen/source_manifest.json", source_hashes())
    suites = sorted(suite_directory.glob('*.json'), key=lambda p:(p.name.startswith('advection'),p.name))
    if not suites:
        raise ValueError("No experiment suites found")
    for suite in suites:
        prepare(suite.resolve(), output/suite.stem, cases=cases)
    dump(output/"queue.json", dict(suites=[p.stem for p in suites], suite_directory=str(suite_directory.resolve())))
    dump(output/"status.json", dict(status="prepared", completed_suites=0, total_suites=len(suites)))
    if frozen:
        write_frozen_manifest()


def run_all(output):
    """Run the existing frozen suite entry point sequentially, three seeds at a time."""
    queue = read(output/"queue.json")["suites"]
    if read(output/"status.json")["status"] != "prepared":
        raise ValueError("Use a fresh prepared suite queue")
    guard = prevent_sleep()
    try:
        for position, name in enumerate(queue):
            stage = output/name
            dump(output/"status.json", dict(status="running", pid=os.getpid(), active_suite=name,
                 completed_suites=position, total_suites=len(queue)))
            with (stage/"launcher.log").open('x') as log:
                result = subprocess.run([sys.executable,"-m","experiments.paper_final.run_online","run","--output",str(stage)],
                    cwd=stage/"source", env=single_thread_environment(), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Suite failed: {name}; inspect its launcher.log and seed status")
            print(f"Completed {position+1}/{len(queue)}: {name}", flush=True)
        rows = []
        for name in queue:
            summary = read(output/name/"summary.json")
            for row in summary["rows"]:
                rows.append(dict(experiment=name.rsplit('_batch',1)[0], **row))
        aggregates = []
        for experiment in sorted({row["experiment"] for row in rows}):
            for window in dict.fromkeys(row["window"] for row in rows):
                for method in factorial.METHODS:
                    selected = [r for r in rows if (r["experiment"],r["window"],r["method"]) == (experiment,window,method)]
                    aggregates.append(dict(experiment=experiment,window=window,method=method,seeds=len(selected),
                        final_failures=sum(r["failures"] for r in selected),
                        **{field+"_"+stat:float(np.mean([r[field] for r in selected]) if stat=="mean"
                            else np.std([r[field] for r in selected],ddof=1))
                            for field in ("native_reduction_pct","inclusive_reduction_pct")
                            for stat in (("mean","sd") if len(selected)>1 else ("mean",))}))
        frozen = all(read(output/name/"protocol.json").get("frozen_evaluation") for name in queue)
        combined = dict(module=1, phase="frozen" if frozen else "online", passed=True,
                        rows=rows, aggregates=aggregates, frozen=frozen)
        dump(output/"summary.json", combined)
        write_csv(output/"summary.csv", rows)
        write_csv(output/"aggregate.csv", aggregates)
        if len(queue) == 12 and all(read(output/name/"protocol.json").get("frozen_evaluation")
            and all(s["cases"] == 1000 for s in read(output/name/"protocol.json")["stages"].values()) for name in queue):
            dump(ROOT/"data/module1/frozen/summary.json", combined)
            write_csv(ROOT/"data/module1/frozen/summary.csv", rows)
            write_csv(ROOT/"data/module1/frozen/aggregate.csv", aggregates)
            for experiment in sorted({row["experiment"] for row in rows}):
                values, seeds = [], []
                for name in queue:
                    if name.rsplit('_batch',1)[0] != experiment:
                        continue
                    with np.load(output/name/"paper_figures/plot_data.npz",allow_pickle=False) as payload:
                        values.append(payload['timings_sec'])
                        seeds.extend(payload['seeds'].tolist())
                        fields = payload['timing_fields'].copy()
                arrays = ROOT/f'data/module1/frozen/arrays/{experiment}.npz'
                arrays.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(arrays,seeds=np.asarray(seeds),methods=np.asarray(factorial.METHODS),
                     timing_fields=fields,timings_sec=np.concatenate(values),problem_indices=np.arange(5001,6001))
            write_frozen_manifest()
        dump(output/"status.json", dict(status="complete", passed=True, completed_suites=len(queue),total_suites=len(queue),at=now()))
    except BaseException:
        dump(output/"status.json", dict(status="failed", traceback=traceback.format_exc()))
        raise
    finally:
        stop_sleep_prevention(guard)


def prepare(suite_path, output, *, cases=None):
    suite = read(suite_path)
    if set(suite) != {"name", "seeds"} or not suite["seeds"]:
        raise ValueError("Suite needs a name and nonempty seeds [{label, config}]")
    labels = [str(item["label"]) for item in suite["seeds"]]
    if len(set(labels)) != len(labels) or any(not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label) for label in labels):
        raise ValueError("Seed labels must be unique safe directory names")
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(name=suite["name"], prepared_at=now(), methods=factorial.METHODS,
        seeds=labels, recovery="failure_model_adjacent", feedback_mode="adjacent_delayed",
        learned_attempts=3, fixed_rl_attempts=1, default_fallback_attempts=1,
        default_attempts=1, rl_recovery_cost=False, rl_whole_case_failure_rollback=True,
        rl_cost_target="native_cycle",
        rl_update_time_charge="none; controller overhead is evaluation only",
        solve_state_encoding=STATE_ENCODING,
        setup_failure_observation_retention=True, failure_score_beta=2.0,
        setup_fallback_feedback="successful_default_only",
        shared_prefix=False, copied_measurements=False, exact_default=DEFAULT_SETUP_PARAMS,
        learning_start="Both setup learners and both LSTDQ controllers start cold at input 1",
        concurrency="One single-thread worker per seed; four methods serially in balanced permutation blocks",
        native_sha256=sweep.native_hashes(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        stages={})
    for item, label in zip(suite["seeds"], labels):
        config_path = Path(item["config"])
        if not config_path.is_absolute():
            config_path = suite_path.parent / config_path
        raw = copy.deepcopy(read(config_path))
        stage = output/f"seed_{label}"
        stage.mkdir()
        raw["output_dir"] = str(stage)
        runtime = decode(raw, stage)
        declared_cases = len(runtime.stream)
        count = declared_cases if cases is None else int(cases)
        if count < 1 or count > declared_cases:
            raise ValueError("Case count exceeds the declared stream")
        frozen = runtime.args.evaluation is not None
        if protocol.setdefault("frozen_evaluation", frozen) != frozen:
            raise ValueError("Do not mix training and frozen evaluation in one suite")
        if protocol.setdefault("continuous_stream", runtime.freeze_after is not None) != (runtime.freeze_after is not None):
            raise ValueError("Do not mix continuous and separate-stage streams in one suite")
        if frozen:
            protocol["learning_start"] = "Final Module 1 checkpoints; all learning disabled"
        protocol["phase"] = "frozen" if frozen else "online"
        if runtime.freeze_after is not None:
            protocol.update(phase="continuous", continuous_stream=True, freeze_after=runtime.freeze_after,
                            learning_inputs=[1, runtime.freeze_after], frozen_inputs=[runtime.freeze_after+1, len(runtime.stream)])
        joint_method = next(m for m in runtime.spec.methods if m.name == "bandit_lstdq")
        if runtime.args.warmup_cases or runtime.args.aot_max_selections_per_case != 3:
            raise ValueError("Expected cold starts and three candidate rows per input")
        # Explicit grids protect final experiments from inherited shell settings.
        for space in runtime.spec.setup.configuration_spaces.values():
            if set(space.numeric_grids) != {"strong_threshold", "max_row_sum", "trunc_factor", "P_max_elmts", "agg_num_levels"}:
                raise ValueError("The new runner requires all five numerical grids explicitly")
        dump(stage/"experiment_config.json", raw)
        dump(stage/"stream_manifest.json", runtime.manifest)
        rows = input_rows(runtime, count)
        (stage/"inputs.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        dump(stage/"status.json", dict(status="prepared", processed=0, cases=count))
        protocol["stages"][label] = dict(cases=count, smoke=count != declared_cases,
            freeze_after=runtime.freeze_after,
            joint_activation_case=joint_method.solve_activation_case,
            config_sha256=file_hash(stage/"experiment_config.json"),
            source_config=str(config_path.resolve()), source_config_sha256=file_hash(config_path),
            inputs_sha256=digest(rows), full_stream_sha256=runtime.manifest["sha256"])
        if frozen:
            protocol["stages"][label].update(prepare_evaluation(runtime, stage))
        print(f"Prepared seed {label}: {count} cases; frozen evaluation={frozen}", flush=True)
    hashes = source_hashes()
    dump(output/"source_manifest.json", hashes)
    for name in hashes:
        destination = output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    (output/"source/data").symlink_to(ROOT/"data", target_is_directory=True)
    (output/"source/hypre/install").symlink_to(ROOT/"hypre/install", target_is_directory=True)
    (output/"source/hypre/interfaces/libamg_runtime.dylib").symlink_to(ROOT/"hypre/interfaces/libamg_runtime.dylib")
    dump(output/"protocol.json", protocol)
    dump(output/"status.json", dict(status="prepared", native_solves_executed=0))
    return protocol


def check_frozen(output):
    if source_hashes() != read(output/"source_manifest.json"):
        raise ValueError("Source changed since preparation")
    if sweep.native_hashes() != read(output/"protocol.json")["native_sha256"]:
        raise ValueError("Native libraries changed since preparation")


def initialize(output, label):
    protocol = read(output/"protocol.json")
    stage = output/f"seed_{label}"
    if read(stage/"status.json")["status"] != "prepared":
        raise ValueError("Refusing to resume partial learned state; use a fresh prepared output")
    checks = protocol["stages"][label]
    if file_hash(stage/"experiment_config.json") != checks["config_sha256"]:
        raise ValueError("Prepared configuration changed")
    if checks.get("frozen_evaluation"):
        for name, key in (("checkpoint_sources.json", "checkpoint_sources_sha256"),
                          ("training_history.json", "training_history_sha256")):
            if file_hash(stage/name) != checks[key]:
                raise ValueError("Frozen training provenance changed")
        for relative, info in read(stage/"checkpoint_sources.json").items():
            if file_hash(stage/"training_checkpoints"/relative) != info["sha256"]:
                raise ValueError("Frozen training checkpoint changed")
    root = decode(read(stage/"experiment_config.json"), stage)
    evaluation = root.args.evaluation
    if digest(input_rows(root, checks["cases"])) != checks["inputs_sha256"]:
        raise ValueError("Input pairing changed")
    study = resolve_composable_study(root.args)
    setup_args = replace(root.args, online_cases=evaluation.training_cases+root.args.online_cases) if evaluation else root.args
    setup = build_named_setup_branches(setup_args, study=study, warmup_instances=(),
                                       stream_hash=root.manifest["sha256"])
    validate_setup_branch_independence(setup.branches, named_setup_spaces=True, setup_candidate_mode="aot")
    solve = build_composable_solve_runtime(root.args, specs=root.spec.methods)
    runtimes = {}
    for method in root.spec.methods:
        runtime = SimpleNamespace(**vars(root), name=method.name, method=method,
                                  bundle=solve.controller_bundles.get(method.name))
        if method.name in setup.branches:
            runtime.branch = setup.branches[method.name]
            model = observed_model(FailureAwareRetryLinUCB.from_runtime_model(runtime.branch.policy.model))
            runtime.branch.policy.model = model
            runtime.rows = CandidateRows(model, cases=len(root.stream), attempts=3, directory=stage/"unused")
            if evaluation:
                model.load_mutable_state(stage/f"training_checkpoints/final_bandit_states/{method.name}.npz")
                from setup.learners.linucb.learner import SharedLinUCBStep
                last = read(stage/"training_history.json")[method.name]["last_attempt"]
                model.history = [SharedLinUCBStep(model.t, last["arm_index"], last["learning_cost"], 0., 0.)]
                for field in ("update", "defer_cost", "resolve_cost", "update_default_fallback"):
                    setattr(model, field, reject_frozen_update)
                for field in ("A_inv", "b", "failure_b"):
                    getattr(model, field).setflags(write=False)
            destination = stage/"initial_bandit_states"/f"{method.name}.npz"
            destination.parent.mkdir(exist_ok=True)
            model.save_mutable_state(destination)
        if runtime.bundle is not None:
            if evaluation:
                runtime.bundle.load(stage/f"training_checkpoints/checkpoints/{method.name}_final.npz")
                runtime.bundle.controller.update = reject_frozen_update
                for field in ("theta", "a_matrix", "a_inverse", "b", "episode_moment_covariance"):
                    getattr(runtime.bundle.controller, field).setflags(write=False)
            destination = stage/"initial_controllers"/f"{method.name}.npz"
            destination.parent.mkdir(exist_ok=True)
            runtime.bundle.save(destination)
        runtimes[method.name] = runtime
    features = {name: factorial.audit_features(runtimes[name]) for name in factorial.METHODS[2:]}
    independence = factorial.audit_independence(runtimes, allow_trained=evaluation is not None)
    candidate_hashes = {name: file_hash(Path(runtimes[name].rows.base._ids.filename))
                        for name in factorial.METHODS[2:]}
    if len(set(candidate_hashes.values())) != 1:
        raise ValueError("Paired setup candidate schedules differ")
    dump(stage/"initial_audit.json", dict(features=features, independence=independence,
         candidates=candidate_hashes, inputs_sha256=checks["inputs_sha256"], passed=True))
    return runtimes


def save_checkpoints(stage, runtimes, suffix):
    for name, runtime in runtimes.items():
        if hasattr(runtime, "branch"):
            directory = stage/("final_bandit_states" if suffix == "final" else "bandit_checkpoints")
            directory.mkdir(exist_ok=True)
            filename = f"{name}.npz" if suffix == "final" else f"{name}_{suffix}.npz"
            runtime.branch.policy.model.save_mutable_state(directory/filename)
        if runtime.bundle is not None:
            directory = stage/"checkpoints"
            directory.mkdir(exist_ok=True)
            runtime.bundle.save(directory/f"{name}_{suffix}.npz")


def controller_summary(runtime):
    if runtime.bundle is None:
        return dict(steps=0, episodes=0)
    controller = runtime.bundle.controller
    return dict(steps=controller.steps, episodes=controller.episodes, **controller.summary())


def worker(output, label):
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError("Exactly one MPI rank is required per seed")
    stage = output/f"seed_{label}"
    handles = {}
    try:
        check_frozen(output)
        runtimes = initialize(output, label)
        first = runtimes[factorial.METHODS[0]]
        protocol = read(output/"protocol.json")
        cases = protocol["stages"][label]["cases"]
        dump(stage/"environment.json", dict(pid=os.getpid(), executable=sys.executable,
            python=sys.version, qos=screen.request_performance_qos(), mpi=MPI.Get_library_version(),
            threads={key: os.environ.get(key) for key in THREAD_KEYS}))
        warmup = [sweep.execute_default_case(first, i) for i in range(min(3, cases))]
        dump(stage/"native_warmup.json", dict(measured=False, learner_updates=0, records=warmup))
        dump(stage/"status.json", dict(status="ready", pid=os.getpid(), processed=0, cases=cases))
        while not (output/"release.json").exists():
            time.sleep(0.2)
        rows = {name: [] for name in factorial.METHODS}
        previous = dict.fromkeys(factorial.METHODS, 0.0)
        offset = evaluation_index_offset(first.args)
        freeze_after = protocol["stages"][label].get("freeze_after")
        orders = factorial.balanced_orders(cases+offset, first.args.method_order_seed)[offset:]
        factorial.audit_independence(runtimes, allow_trained=bool(offset))
        dump(stage/"activation_audit.json", dict(passed=True, problem_index=offset+1, frozen_evaluation=bool(offset),
            joint_controller_steps=runtimes[factorial.METHODS[3]].bundle.controller.steps,
            fixed_controller_steps=runtimes[factorial.METHODS[1]].bundle.controller.steps,
            setup_observations={name: runtimes[name].branch.policy.model.t for name in factorial.METHODS[2:]}))
        (stage/"trajectories").mkdir()
        handles = {name: (stage/"trajectories"/f"{name}.jsonl").open("w") for name in rows}
        started = time.perf_counter()
        with (stage/"execution.jsonl").open("w") as schedule:
            for index, order in enumerate(orders):
                if freeze_after is not None and index == freeze_after:
                    for handle in handles.values():
                        handle.flush()
                    schedule.flush()
                    freeze_live_runtimes(stage, runtimes, freeze_after)
                for rank, name in enumerate(order):
                    begin = time.monotonic_ns()
                    row, previous[name] = factorial.execute(runtimes[name], index, previous[name])
                    end = time.monotonic_ns()
                    row.update(execution_rank=rank, execution=dict(pid=os.getpid(), position=rank))
                    rows[name].append(row)
                    handles[name].write(json.dumps(ready(row), allow_nan=False)+"\n")
                    schedule.write(json.dumps(dict(input_index=index, method=name, position=rank,
                                                   start_ns=begin, end_ns=end, pid=os.getpid()))+"\n")
                if (index+1) % first.args.progress_every == 0 or index+1 == cases:
                    for handle in handles.values():
                        handle.flush()
                    schedule.flush()
                    dump(stage/"status.json", dict(status="running", pid=os.getpid(), processed=index+1,
                        cases=cases, elapsed_sec=time.perf_counter()-started,
                        methods={name: sweep.metrics(records) for name, records in rows.items()}))
                    print(f"seed {label}: {index+1}/{cases}", flush=True)
                if (index+1) % 1000 == 0:
                    save_checkpoints(stage, runtimes, str(index+1))
        for handle in handles.values():
            handle.close()
        final_audits = {}
        for name, runtime in runtimes.items():
            measured_inputs = [dict(problem_index=row["problem_index"], mkw=row["mkw"],
                                    context=row["context"]) for row in rows[name]]
            if digest(measured_inputs) != protocol["stages"][label]["inputs_sha256"]:
                raise ValueError(f"Measured inputs differ from the frozen stream: {name}")
            controller = controller_summary(runtime)
            if runtime.bundle is not None and not offset:
                audit_controller_counts(rows[name], controller)
            model_records = rows[name][:freeze_after] if freeze_after is not None else rows[name]
            model = screen.audit_final_model(runtime, model_records) if hasattr(runtime, "branch") and not offset else None
            final_audits[name] = dict(controller=controller, model=model,
                cases=len(rows[name]), trajectory_sha256=file_hash(stage/"trajectories"/f"{name}.jsonl"))
        save_checkpoints(stage, runtimes, "final")
        if offset or (freeze_after is not None and cases > freeze_after):
            dump(stage/"frozen_state_audit.json", frozen_state_audit(stage, runtimes, continuous=not bool(offset)))
        check_frozen(output)
        checkpoint_hashes = {str(path.relative_to(stage)): file_hash(path)
            for directory in (stage/"checkpoints", stage/"final_bandit_states")
            for path in directory.iterdir() if path.is_file()}
        dump(stage/"result.json", dict(complete=True, passed=True, cases=cases,
            elapsed_sec=time.perf_counter()-started, audits=final_audits,
            checkpoint_sha256=checkpoint_hashes,
            methods={name: sweep.metrics(records) for name, records in rows.items()}))
        dump(stage/"status.json", dict(status="complete", processed=cases, cases=cases,
                                      elapsed_sec=time.perf_counter()-started))
    except BaseException:
        dump(stage/"status.json", dict(status="failed", pid=os.getpid(), traceback=traceback.format_exc()))
        raise
    finally:
        for handle in handles.values():
            handle.close()


def analyze(output):
    protocol = read(output/"protocol.json")
    table = []
    for label in protocol["seeds"]:
        stage = output/f"seed_{label}"
        result = read(stage/"result.json")
        if not result["complete"] or not result["passed"]:
            raise ValueError(f"Seed {label} incomplete or failed its audit")
        for name in factorial.METHODS:
            if file_hash(stage/"trajectories"/f"{name}.jsonl") != result["audits"][name]["trajectory_sha256"]:
                raise ValueError(f"Measured trajectory changed for seed {label}: {name}")
        records = {name: list(read_records(stage/"trajectories"/f"{name}.jsonl")) for name in factorial.METHODS}
        cases = result["cases"]
        windows = {"full": (0, cases), "first_1000": (0, min(1000, cases))}
        freeze_after = protocol["stages"][label].get("freeze_after")
        if freeze_after is not None:
            windows["learning"] = (0, min(freeze_after, cases))
            if cases > freeze_after:
                windows["frozen"] = (freeze_after, cases)
        activation = next(m.get("solve_activation_case", 0) for m in read(stage/"experiment_config.json")["methods"] if m["id"] == "bandit_lstdq")
        if activation < cases:
            windows["joint"] = (activation, cases)
        if cases >= 1000:
            windows["last_1000"] = (cases-1000, cases)
        for window, (start, stop) in windows.items():
            # Shared final-success subset is explicit and identical for all four methods.
            common = [i for i in range(start, stop) if all(not records[n][i]["outcome"]["failed"] for n in records)]
            default = sweep.metrics(records[factorial.METHODS[0]][start:stop])
            common_default = sweep.metrics([records[factorial.METHODS[0]][i] for i in common])
            for name in factorial.METHODS:
                metrics = sweep.metrics(records[name][start:stop])
                matched = sweep.metrics([records[name][i] for i in common])
                table.append(dict(seed=label, window=window, method=name, **metrics,
                    native_reduction_pct=100*(1-metrics["native_sec"]/default["native_sec"]),
                    inclusive_reduction_pct=100*(1-metrics["total_sec"]/default["total_sec"]),
                    common_success_cases=len(common),
                    common_success_native_reduction_pct=100*(1-matched["native_sec"]/common_default["native_sec"]) if common else None))
    write_csv(output/"summary.csv", table)
    aggregates = []
    for window in dict.fromkeys(row["window"] for row in table):
        for name in factorial.METHODS:
            selected = [r for r in table if r["window"] == window and r["method"] == name]
            aggregates.append(dict(window=window, method=name, seeds=len(selected),
                **{f"{field}_{stat}": float(np.mean(values) if stat == "mean" else np.std(values, ddof=1))
                   for field in ("native_reduction_pct", "inclusive_reduction_pct")
                   for values in ([r[field] for r in selected],)
                   for stat in (("mean", "std") if len(selected) > 1 else ("mean",))}))
    dump(output/"summary.json", dict(protocol=protocol, rows=table, aggregates=aggregates, passed=True))
    accounting = "All four methods were measured independently from their first paired input."
    text = [f"{protocol['name']}\n", accounting + " Native time includes every setup/solve attempt and fallback. Inclusive time also includes measured learner costs.\n",
        "| Seed | Method | Native seconds | Native reduction | Inclusive reduction | Final failures |",
        "|---|---|---:|---:|---:|---:|"]
    for row in table:
        if row["window"] == "full":
            text.append(f"| {row['seed']} | {factorial.LABELS[row['method']]} | {row['native_sec']:.3f} | {row['native_reduction_pct']:.2f}% | {row['inclusive_reduction_pct']:.2f}% | {row['failures']} |")
    text += ["", "Per-input trajectories and final setup/controller checkpoints are in each seed directory. Source and native hashes, feature checks, activation checks and batch reconstruction of both setup regressions passed."]
    (output/"report.md").write_text("\n".join(text)+"\n")
    factorial.audit_execution(output)
    from experiments.paper_final.plot_online import generate
    figures = generate(output)
    with (output/"report.md").open("a") as handle:
        kind = read(output/f"seed_{protocol['seeds'][0]}"/"experiment_config.json")["problem"]["kind"]
        module_number = 3 if kind.startswith("rs_") else 1
        handle.write(f"\n[Paper figures](paper_figures/module{module_number}_paper_figures.pdf) · [Plot data and provenance](paper_figures/manifest.json)\n")
    return table


def launch(output):
    check_frozen(output)
    if read(output/"status.json")["status"] != "prepared":
        raise ValueError("Suite is not fresh; partial experiment resume is unsupported")
    protocol = read(output/"protocol.json")
    children, logs = [], []
    sleep_guard = prevent_sleep()
    try:
        dump(output/"status.json", dict(status="initializing", pid=os.getpid()))
        for label in protocol["seeds"]:
            log = (output/f"seed_{label}"/"worker.log").open("w")
            logs.append(log)
            children.append(subprocess.Popen([sys.executable, "-m", "experiments.paper_final.run_online",
                "worker", "--output", str(output), "--label", label], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, env=single_thread_environment()))
        while True:
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A worker exited during preparation; inspect seed worker.log/status.json")
            if all(read(output/f"seed_{label}"/"status.json")["status"] == "ready" for label in protocol["seeds"]):
                break
            time.sleep(0.5)
        dump(output/"release.json", dict(at=now(), worker_pids=[child.pid for child in children]))
        dump(output/"status.json", dict(status="running", pid=os.getpid(), worker_pids=[child.pid for child in children]))
        print(f"Released {len(children)} seed workers", flush=True)
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("A worker failed; stopping the paired suite")
            time.sleep(1)
        if any(child.returncode for child in children):
            raise RuntimeError("A worker failed; inspect worker.log/status.json")
        check_frozen(output)
        analyze(output)
        dump(output/"status.json", dict(status="complete", at=now(), passed=True))
        print("All seed experiments and audits complete", flush=True)
    except BaseException:
        dump(output/"status.json", dict(status="failed", traceback=traceback.format_exc()))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()
        stop_sleep_prevention(sleep_guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker", "analyze", "prepare-frozen", "prepare-all", "run-all"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--cases", type=int)
    args = parser.parse_args()
    if args.command == "prepare-frozen":
        prepare_frozen()
        return
    if args.output is None:
        parser.error("This command requires --output")
    output = args.output.resolve()
    if args.command == "prepare-all":
        if args.suite is None:
            parser.error("prepare-all requires --suite pointing to a suite directory")
        prepare_all(args.suite.resolve(), output, cases=args.cases)
        return
    if args.command == "run-all":
        run_all(output)
        return
    if args.command == "prepare":
        if args.suite is None:
            parser.error("prepare requires --suite")
        prepare(args.suite.resolve(), output, cases=args.cases)
    elif args.command == "worker":
        worker(output, args.label)
    elif args.command == "run":
        launch(output)
    else:
        analyze(output)


if __name__ == "__main__":
    main()
