#!/usr/bin/env python3
"""Offline packaging and singleton Linux jobs. No AWS mutation API calls."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = "bf962f2d990f22d2abc94043db0ef4a84aa277fe"
METHODS = ["default_setup_default_solve", "default_setup_lstdq", "bandit_default", "bandit_lstdq"]
THREADS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]
PATCHED = ["experiments/paper_final/run_online.py", "hypre/bindings/Makefile"]
TOOLS = ["deployment/aws/" + p for p in ["campaign.py", "bootstrap.sh", "README.md", "test_campaign.py",
    "serial_pilot.py", "test_serial_pilot.py", "test_shutdown.py", "pilot-budget.template.json", "serial-plan.json"]]
PREFIXES = ("experiments/", "setup/", "solve/", "problems/", "configs/module1/",
            "hypre/source/", "hypre/interfaces/", "hypre/bindings/", "configs/module3/",
            "data/module1/inputs/", "data/module1/frozen/inputs/")
EXACT = {"pyproject.toml", "README.md", "hypre/Makefile", "hypre/__init__.py"}
SECRET_RULES = {
    "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----",
    "aws_access_key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "github_token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "slack_token": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b",
    "google_api_key": rb"\bAIza[A-Za-z0-9_-]{35}\b",
    "aws_secret_assignment": rb"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{35,}",
    "url_password": rb"https?://[^\s/:]+:[^\s/@]{8,}@",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def tracked():
    if git("rev-parse", "HEAD").decode().strip() != BASE:
        raise ValueError("Checkout is not the selected base commit")
    return [p for p in git("ls-tree", "-rz", "--name-only", BASE).decode().split("\0") if p]


def scan(files):
    """Return locations/rules only. Never return matched secret bytes."""
    findings = []
    for name, data in files.items():
        parts = Path(name).parts
        if any(p in {".aws", ".ssh", ".env", "credentials", ".netrc"} for p in parts) or name.endswith((".pem", ".p12", ".pfx", ".key")):
            findings.append({"path": name, "rule": "sensitive_filename"})
        if b"\x00" in data:
            continue
        for rule, pattern in SECRET_RULES.items():
            for match in re.finditer(pattern, data):
                findings.append({"path": name, "rule": rule, "line": data[:match.start()].count(b"\n") + 1})
    return {"files_scanned": len(files), "findings": findings, "passed": not findings,
            "scope": "Common credential patterns and sensitive filenames; not a proof of absence"}


def scan_base():
    # Worktree is a fresh clone. Read pinned blobs, so local patches cannot hide a finding.
    paths = tracked()
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        blobs = {}
        for name in paths:
            proc.stdin.write((BASE + ":" + name + "\n").encode()); proc.stdin.flush()
            header = proc.stdout.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                raise ValueError("Unexpected pinned Git object")
            blobs[name] = proc.stdout.read(int(header[2]))
            proc.stdout.read(1)
        proc.stdin.close(); proc.wait()
    return scan(blobs)


def plan_payloads():
    jobs, payloads = [], {}
    for family in ("diffusion", "advection"):
        for grid in (40, 60, 80):
            for seed in range(1, 7):
                job_id = f"{family}_{grid}_seed{seed}"
                config = f"configs/module1/{family}_{grid}/seed_{seed}.json"
                raw = read(ROOT/config)
                if (raw["stream"]["cases"] != 5000 or raw["frozen_stream"]["cases"] != 1000
                        or [m["id"] for m in raw["methods"]] != METHODS
                        or raw["problem"]["grid"] != [grid]*3):
                    raise ValueError("Unexpected selected-commit experiment design")
                suite = f"deployment/aws/plan/suites/{job_id}.json"
                payloads[suite] = {"name": f"AWS singleton {job_id}",
                    "seeds": [{"label": str(seed), "config": "../../../../" + config}]}
                jobs.append({"id": job_id, "family": family, "grid": grid, "seed": seed,
                    "suite": suite, "config": config, "config_sha256": sha((ROOT/config).read_bytes()),
                    "learning_input_sha256": raw["stream"]["expected_sha256"],
                    "frozen_input_sha256": raw["frozen_stream"]["expected_sha256"],
                    "learning_cases": 5000, "frozen_cases": 1000, "methods": METHODS,
                    "timeout_seconds": {40: 14400, 60: 21600, 80: 43200}[grid]})
    payloads["deployment/aws/plan/campaign.json"] = {"base_commit": BASE, "jobs": jobs,
        "module": 1, "launch_approval": "pending", "run_all_on_each_instance": False,
        "job_count": 36, "mpi_ranks_per_job": 1, "threads": {k: "1" for k in THREADS},
        "note": "Timeouts are provisional limits, not measured Linux runtimes"}
    payloads["deployment/aws/plan/full-study.json"] = full_study_plan()
    return payloads


def full_study_plan():
    """Reviewable dependencies only; additional modules are never launch jobs."""
    module3 = []
    for seed in range(1, 7):
        name = f"configs/module3/variable_flow_60/seed_{seed}.json"
        raw = read(ROOT/name)
        if raw["stream"]["cases"] != 5000 or raw.get("frozen_stream") or [m["id"] for m in raw["methods"]] != METHODS:
            raise ValueError("Unexpected selected-commit Module 3 design")
        module3.append({"id": f"variable_flow_60_seed{seed}", "module": 3, "seed": seed,
            "config": name, "config_sha256": sha((ROOT/name).read_bytes()), "learning_cases": 5000,
            "frozen_cases": 0, "methods": METHODS, "requires": [], "launchable_by_this_tool": False})
    return {"base_commit": BASE, "approval": "No resources approved or launched",
        "upstream_wrapper": "experiments/run_all.py",
        "wrapper_sha256": sha((ROOT/"experiments/run_all.py").read_bytes()),
        "wrapper_use_on_each_vm": "Do not invoke: it repeats the entire multi-module study",
        "module1": {"singleton_jobs": 36, "learning_inputs": 180000, "frozen_inputs": 36000,
                    "method_evaluations_before_retries": 864000},
        "module2": {"scope": "Separate approval and runtime estimate required", "automatic_launch": False,
            "default_training_source": "diffusion_60", "method": "bandit_lstdq",
            "requires": [f"diffusion_60_seed{s}" for s in range(1, 7)],
            "barrier": "All six FULL verified source jobs; no smoke/partial/Mac/mixed-revision substitutes",
            "required_files_per_seed": ["experiment_config.json", "checkpoints/bandit_lstdq_final.npz",
                "checkpoints/bandit_lstdq_final.npz.encoder.json", "final_bandit_states/bandit_lstdq.npz"],
            "collection_layout": "COLLECTION_ROOT/diffusion_60_seedN/run/seed_N",
            "random_seed": 20261019, "evaluation_cases": 100, "fixed_weights": 41,
            "scan_repetitions": 1, "extra_scan_case_fraction": 0.1, "extra_scan_repetitions": 2,
            "planned_fixed_scan_trials": 29520, "evaluation_repetitions": 3,
            "policy_roles": ["fixed_w1", "global_best_fixed", "per_case_best_fixed", "periodic", "frozen_lstdq"],
            "maximum_evaluation_cells": 9000,
            "note": "Identical policy cells are shared; native preflight/recovery add work. No time/price claim.",
            "workers_default": 3, "execution": "One separate evaluation host/controller, three single-rank workers by default"},
        "module3": {"scope": "Additional work outside the 36 Module 1 jobs; separate approval required",
            "automatic_launch": False, "requires": [], "singleton_jobs": module3,
            "learning_inputs": 30000, "method_evaluations_before_retries": 120000,
            "runtime_and_cost": "Unmeasured for selected Linux host; not included in Module 1 estimate"}}


def collect_module2(inputs_root, destination):
    """Validate downloaded Module 1 evidence; emit commands, never run policies."""
    inputs_root = Path(inputs_root).resolve()
    destination = Path(destination).resolve()
    records, source_hashes, native_hashes = [], set(), set()
    for seed in range(1, 7):
        job = get_job(f"diffusion_60_seed{seed}")
        directory = inputs_root/job["id"]
        if not directory.resolve().is_relative_to(inputs_root):
            raise ValueError("Collection path escapes its root")
        receipt = read(directory/"job_receipt.json")
        if (receipt.get("base_commit") != BASE or receipt.get("mode") != "full"
                or not receipt.get("passed") or receipt.get("cases") != 6000
                or receipt.get("job", {}).get("id") != job["id"]
                or receipt["job"].get("config_sha256") != job["config_sha256"]):
            raise ValueError(f"Not a full verified source from this revision: {job['id']}")
        for name, expected in read(directory/"OUTPUT_SHA256.json").items():
            path = directory/name
            if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                    or not path.resolve().is_relative_to(directory.resolve()) or sha(path.read_bytes()) != expected):
                raise ValueError("Collected output checksum mismatch")
        verify_result(directory, job, 6000)
        stage = directory/f"run/seed_{seed}"
        raw = read(stage/"experiment_config.json")
        pinned = read(ROOT/job["config"])
        raw["output_dir"] = pinned["output_dir"]  # Official preparation overrides only its destination.
        if raw != pinned: raise ValueError("Collected configuration differs from pinned design")
        protocol = read(directory/"run/protocol.json")
        for name, expected in read(directory/"run/source_manifest.json").items():
            path = ROOT/name
            if (Path(name).is_absolute() or ".." in Path(name).parts
                    or not path.resolve().is_relative_to(ROOT) or sha(path.read_bytes()) != expected):
                raise ValueError("Training execution source differs from this deployment")
        source_hashes.add(sha((directory/"run/source_manifest.json").read_bytes()))
        native_hashes.add(json.dumps(protocol["native_sha256"], sort_keys=True))
        files = full_study_plan()["module2"]["required_files_per_seed"]
        records.append({"seed": seed, "run_dir": str(stage), "job": job["id"],
            "files_sha256": {name: sha((stage/name).read_bytes()) for name in files}})
    if len(source_hashes) != 1 or len(native_hashes) != 1:
        raise ValueError("Mixed source/native builds; investigate before evaluation")
    if destination.exists(): raise FileExistsError("Refusing to overwrite Module 2 collection plan")
    destination.mkdir(parents=True)
    configuration, output = destination/"configuration.json", destination/"run"
    init = ["python", "-m", "experiments.paper_final.run_policy", "init", "--config", str(configuration),
            "--random-seed", "20261019"]
    for record in records: init += ["--training-run", f"{record['seed']}={record['run_dir']}"]
    commands = [init,
        ["python", "-m", "experiments.paper_final.run_policy", "validate", "--config", str(configuration)],
        ["python", "-m", "experiments.paper_final.run_policy", "prepare", "--config", str(configuration), "--output", str(output)],
        ["python", "-m", "experiments.paper_final.run_policy", "run", "--output", str(output)]]
    result = {"base_commit": BASE, "passed": True, "training_sources": records,
        "training_source_manifest_sha256": next(iter(source_hashes)),
        "training_native_sha256": json.loads(next(iter(native_hashes))),
        "execution_commands": commands, "native_solves_executed": 0, "aws_calls": 0,
        "note": "Commands only. Module 2 execution/cost approval and matching Linux environment remain required."}
    write(destination/"collection-plan.json", result)
    return result


def plan():
    for name, value in plan_payloads().items():
        write(ROOT/name, value)
    return {"job_count": 36, "manifest": str(ROOT/"deployment/aws/plan/campaign.json")}


def bundle(destination):
    report = scan_base()
    if not report["passed"]:
        return report
    paths = tracked()
    changes = set(git("diff", "--name-only", BASE).decode().splitlines())
    if changes - set(PATCHED):
        raise ValueError("Unexpected tracked modifications; refusing to package")
    selected = [p for p in paths if p in EXACT or p.startswith(PREFIXES)]
    files = {}
    for name in selected + TOOLS:
        path = ROOT/name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Refusing missing/nonregular source: {name}")
        files[name] = path.read_bytes()
    for name, value in plan_payloads().items():
        files[name] = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    files["deployment/aws/portability.patch"] = git("diff", "--binary", BASE, "--", *PATCHED)
    report = scan(files)
    if not report["passed"]:
        return report
    manifest = {"base_commit": BASE, "files": {n: sha(b) for n, b in sorted(files.items())},
        "deviations": [{"path": p, "base_sha256": sha(git("show", BASE+":"+p)),
                        "deployed_sha256": sha(files[p])} for p in PATCHED],
        "credential_scan": report, "git_history_included": False,
        "excluded": [".git", "credentials", "old results", "historical measurements", "local environment", "native binaries"]}
    files["DEPLOYMENT.json"] = (json.dumps(manifest, indent=2, sort_keys=True)+"\n").encode()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name); info.size = len(data); info.mtime = 0
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))
    destination.write_bytes(gzip.compress(buf.getvalue(), mtime=0))
    digest = sha(destination.read_bytes())
    destination.with_suffix(destination.suffix+".sha256").write_text(digest+"  "+destination.name+"\n")
    return {"bundle": str(destination), "sha256": digest, "files": len(files),
            "bytes": destination.stat().st_size, "credential_scan": report}


def verify_source():
    manifest = read(ROOT/"DEPLOYMENT.json")
    if manifest["base_commit"] != BASE or not manifest["credential_scan"]["passed"]:
        raise ValueError("Wrong or unscanned deployment")
    for name, expected in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe manifest path")
        path = ROOT/name
        if path.is_symlink() or sha(path.read_bytes()) != expected:
            raise ValueError(f"Source integrity check failed: {name}")
    return {"passed": True, "base_commit": BASE, "files": len(manifest["files"])}


def get_job(job_id):
    jobs = plan_payloads()["deployment/aws/plan/campaign.json"]["jobs"]
    return next(j for j in jobs if j["id"] == job_id)


def timed_command(command, directory, label, seconds):
    """Terminate only this job's process group, including its MPI worker."""
    if seconds <= 0:
        raise TimeoutError("Job deadline expired")
    with (directory/f"{label}.log").open("w") as log:
        proc = subprocess.Popen(["/usr/bin/time", "-v", "-o", str(directory/f"{label}.time.txt"), *command],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=seconds)
            if rc:
                raise RuntimeError(f"{label} exited {rc}; inspect its log")
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: pass
                # The group can outlive its leader. Reap any surviving worker.
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()
            except ProcessLookupError:
                pass
            raise


def verify_result(directory, job, cases):
    run = directory/"run"; stage = run/f"seed_{job['seed']}"
    for name in ["status.json", "summary.json", "completion_audit.json"]:
        result = read(run/name)
        if not result.get("passed") or (name == "status.json" and result["status"] != "complete"):
            raise ValueError(f"Missing successful completion: {name}")
    result = read(stage/"result.json")
    if not result["complete"] or not result["passed"] or result["cases"] != cases:
        raise ValueError("Unexpected result case count")
    if set(result["audits"]) != set(METHODS):
        raise ValueError("Missing method audit")
    for method, audit in result["audits"].items():
        if audit["cases"] != cases or sha((stage/f"trajectories/{method}.jsonl").read_bytes()) != audit["trajectory_sha256"]:
            raise ValueError("Trajectory integrity mismatch")
    for name, expected in result["checkpoint_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha((stage/name).read_bytes()) != expected:
            raise ValueError("Checkpoint integrity mismatch")
    if cases == 6000 and (not read(stage/"freeze_boundary.json")["passed"] or not read(stage/"frozen_state_audit.json")["passed"]):
        raise ValueError("Missing frozen-stage audit")
    return {"passed": True, "cases": cases, "methods": METHODS}


def output_hashes(directory):
    hashes = {}
    for path in sorted(directory.rglob("*")):
        rel = path.relative_to(directory)
        # The source snapshot has links back into this deployment. Ship its manifests,
        # not a second copy of source/data/libraries or anything outside the job.
        if "source" in rel.parts or path.is_symlink() or not path.is_file() or path.name == "OUTPUT_SHA256.json":
            continue
        hashes[str(rel)] = sha(path.read_bytes())
    write(directory/"OUTPUT_SHA256.json", hashes)


def run_job(args):
    if sys.platform != "linux":
        raise ValueError("Job execution is Linux-only; no Mac solver run is permitted here")
    verify_source()
    job = get_job(args.job)
    for key in THREADS: os.environ[key] = "1"
    directory = args.output.resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError("Output must be outside the immutable source deployment")
    directory.mkdir(parents=True, exist_ok=False)
    cases = 6000 if args.mode == "full" else args.smoke_cases
    if args.mode == "smoke" and not 1 <= cases <= 100:
        raise ValueError("Smoke inputs must be 1..100; never report as official data")
    started = time.time()
    limit = args.timeout_seconds or (job["timeout_seconds"] if args.mode == "full" else 1800)
    if limit <= 0: raise ValueError("Timeout must be positive")
    deadline = time.monotonic()+limit
    receipt = {"base_commit": BASE, "job": job, "mode": args.mode, "cases": cases,
        "started_unix": started, "timeout_seconds": limit, "passed": False,
        "source_manifest_sha256": sha((ROOT/"DEPLOYMENT.json").read_bytes()),
        "threads": {k: os.environ[k] for k in THREADS},
        "memory_note": "GNU time maximum RSS is a per-process high-water mark, not simultaneous whole-job RSS"}
    write(directory/"job_receipt.json", receipt)
    try:
        timed_command([sys.executable, str(ROOT/"deployment/aws/campaign.py"), "preflight"], directory,
                      "preflight", min(120, deadline-time.monotonic()))
        command = [sys.executable, "-m", "experiments.paper_final.run_online", "prepare",
                   "--suite", str(ROOT/job["suite"]), "--output", str(directory/"run")]
        if args.mode == "smoke": command += ["--cases", str(cases)]
        timed_command(command, directory, "prepare", deadline-time.monotonic())
        timed_command([sys.executable, "-m", "experiments.paper_final.run_online", "run",
                       "--output", str(directory/"run")], directory, "execute", deadline-time.monotonic())
        receipt["verification"] = verify_result(directory, job, cases)
        receipt["passed"] = True
    except BaseException as exc:
        receipt["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        receipt["elapsed_seconds"] = time.time()-started
        write(directory/"job_receipt.json", receipt)
        output_hashes(directory)
    return receipt


def preflight():
    if sys.platform != "linux": raise ValueError("Native preflight is Linux-only")
    result = verify_source()
    import ctypes
    import importlib.metadata
    from mpi4py import MPI
    from hypre.bindings import AMG_RUNTIME_LIBRARY
    if MPI.COMM_WORLD.Get_size() != 1: raise ValueError("Expected one MPI rank")
    ctypes.CDLL(str(AMG_RUNTIME_LIBRARY))  # Load/link only; no native solve.
    result.update(python=sys.version, mpi=MPI.Get_library_version(), mpi_ranks=1,
        platform=os.uname()._asdict() if hasattr(os.uname(), "_asdict") else list(os.uname()),
        dependencies={p: importlib.metadata.version(p) for p in ["numpy", "scipy", "mpi4py", "matplotlib", "mpmath"]},
        threads={k: os.environ.get(k) for k in THREADS})
    if any(result["threads"][k] != "1" for k in THREADS): raise ValueError("Thread limits missing")
    return result


def launch_request(args):
    """Write an explicit one-instance request; never call AWS."""
    for value in [args.campaign, args.instance_type, args.region]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value): raise ValueError("Unsafe launch identifier")
    if not re.fullmatch(r"[0-9a-f]{64}", args.bundle_sha256): raise ValueError("Invalid bundle SHA-256")
    for uri in [args.bundle_uri, args.output_uri]:
        if not re.fullmatch(r"s3://[a-z0-9.-]+/[A-Za-z0-9_./-]+", uri) or ".." in uri.split("/"):
            raise ValueError("Expected an explicit safe S3 object/prefix URI")
    if args.bundle_uri == args.output_uri or args.bundle_uri.startswith(args.output_uri.rstrip("/")+"/"):
        raise ValueError("Input bundle must be outside the output prefix")
    get_job(args.job)
    bootstrap = (ROOT/"deployment/aws/bootstrap.sh").read_text()
    settings = {"RLTUNE_BUNDLE_URI": args.bundle_uri, "RLTUNE_BUNDLE_SHA256": args.bundle_sha256,
        "RLTUNE_OUTPUT_URI": args.output_uri.rstrip("/")+"/"+args.campaign+"/"+args.job+"/"+args.mode+"/"+args.bundle_sha256[:12],
        "RLTUNE_JOB": args.job, "RLTUNE_MODE": args.mode, "AWS_DEFAULT_REGION": args.region,
        "RLTUNE_TIMEOUT": str(get_job(args.job)["timeout_seconds"] if args.mode == "full" else 1800)}
    user_data = "#!/bin/bash\n" + "\n".join("export "+k+"="+shlex.quote(v) for k,v in settings.items()) + "\n" + bootstrap.split("\n", 1)[1]
    if len(user_data.encode()) > 16384: raise ValueError("EC2 user-data exceeds 16 KiB")
    tags = [{"Key": "Project", "Value": "RLTune-SISC"}, {"Key": "Campaign", "Value": args.campaign},
            {"Key": "Job", "Value": args.job}, {"Key": "BaseCommit", "Value": BASE},
            {"Key": "Mode", "Value": args.mode}]
    request = {"ImageId": args.ami, "InstanceType": args.instance_type, "MinCount": 1, "MaxCount": 1,
        "ClientToken": sha((args.campaign+args.job+args.mode+args.bundle_sha256).encode()),
        "IamInstanceProfile": {"Name": args.instance_profile},
        "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": args.subnet, "Groups": [args.security_group],
                               "AssociatePublicIpAddress": args.public_ip}],
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled", "HttpPutResponseHopLimit": 1},
        "InstanceInitiatedShutdownBehavior": "stop",
        "BlockDeviceMappings": [{"DeviceName": args.root_device, "Ebs": {"VolumeSize": args.disk_gib,
            "VolumeType": "gp3", "Encrypted": True, "DeleteOnTermination": True}}],
        "TagSpecifications": [{"ResourceType": t, "Tags": tags} for t in ["instance", "volume"]],
        "UserData": base64.b64encode(user_data.encode()).decode()}
    if args.disk_gib < 20: raise ValueError("Provision at least 20 GiB for environment/build/output")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"run-instances.json", request)
    (args.output/"user-data.sh").write_text(user_data)
    return {"request": str(args.output/"run-instances.json"), "instances": 1, "aws_calls": 0,
        "shutdown": "stop; EBS remains billable until verified cleanup", "region": args.region}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("scan"); sub.add_parser("plan"); sub.add_parser("verify-source"); sub.add_parser("preflight")
    b = sub.add_parser("bundle"); b.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("run"); r.add_argument("--job", required=True); r.add_argument("--output", type=Path, required=True)
    r.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    r.add_argument("--smoke-cases", type=int, default=24); r.add_argument("--timeout-seconds", type=int)
    m = sub.add_parser("module2-collect")
    m.add_argument("--inputs-root", type=Path, required=True); m.add_argument("--output", type=Path, required=True)
    l = sub.add_parser("launch-request")
    for name in ["ami", "instance-type", "region", "subnet", "security-group", "instance-profile",
                 "root-device", "campaign", "bundle-uri", "bundle-sha256", "output-uri", "job"]:
        l.add_argument("--"+name, required=True)
    l.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    l.add_argument("--disk-gib", type=int, default=30); l.add_argument("--public-ip", action="store_true")
    l.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    actions = {"scan": scan_base, "plan": plan, "verify-source": verify_source, "preflight": preflight,
               "bundle": lambda: bundle(args.output), "run": lambda: run_job(args),
               "module2-collect": lambda: collect_module2(args.inputs_root, args.output),
               "launch-request": lambda: launch_request(args)}
    result = actions[args.command](); print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("passed") is False: sys.exit(1)


if __name__ == "__main__":
    main()
