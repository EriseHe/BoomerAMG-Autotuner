#!/usr/bin/env python3
"""One-VM serial correctness pilot. No resource creation or credential access."""
import argparse
import base64
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as c

ORDER = [f"{family}_40_seed{seed}" for seed in (1, 2, 3) for family in ("diffusion", "advection")]
ELIGIBLE = {"t3.micro", "t3.small", "t4g.micro", "t4g.small", "c7i-flex.large", "m7i-flex.large"}


def plan():
    return {"base_commit": c.BASE, "jobs": ORDER, "instances": 1, "max_seed_workers": 1,
        "numerical_threads": 1, "cases_per_job": 6000, "learning_cases": 5000, "frozen_cases": 1000,
        "total_paired_inputs": 36000, "method_evaluations_before_retries": 144000,
        "gate": "Seed-1 pair must pass full correctness before seeds 2/3; abort at any failure",
        "purpose": "Build/correctness/memory; Flex timings exploratory, not publishable official timings",
        "maximum_authorized_usage_usd": 10, "pricing_status": "Exact resource/rate/cleanup plan pending"}


def budget_check(raw):
    if raw.get("resource_plan_validated") is not True or raw.get("cleanup_plan_confirmed") is not True:
        raise ValueError("Validated resource rates AND a bounded post-stop cleanup plan are required")
    if (raw.get("instance_count") != 1 or raw.get("account_plan") != "Free"
            or raw.get("instance_type") not in ELIGIBLE or raw.get("region") != "us-east-2"):
        raise ValueError("Pilot requires exactly one approved Free-plan-eligible Ohio instance")
    if not isinstance(raw.get("public_ip"), bool) or not 20 <= raw.get("disk_gib", 0) <= 30:
        raise ValueError("Validate public-IP choice and 20–30 GiB disk size")
    limit = Decimal(str(raw["max_total_usage_usd"]))
    wall = raw["wall_seconds"]; upload = raw["upload_reserve_seconds"]
    if not 0 < limit <= 10 or not 3600 <= wall <= 28800 or not 600 <= upload < wall:
        raise ValueError("Budget/time cap exceeds approved bounds")
    rates = {name: Decimal(str(raw["hourly_usd"][name])) for name in ("ec2", "ebs", "public_ipv4")}
    reserve = Decimal(str(raw["other_costs_reserve_usd"]))
    retention = Decimal(str(raw["post_stop_storage_hours"]))
    if any(not rate.is_finite() or rate < 0 for rate in rates.values()) or rates["ec2"] == 0 or rates["ebs"] == 0:
        raise ValueError("Require conservative undiscounted EC2/EBS and IPv4 rates")
    if not reserve.is_finite() or reserve < 1 or not 0 < retention <= 24:
        raise ValueError("Require storage/network/request reserve and <=24h post-stop cleanup")
    if raw["public_ip"] and rates["public_ipv4"] <= 0:
        raise ValueError("Public IPv4 charges must be included")
    if not raw.get("pricing_sources") or not raw.get("rates_checked_utc"):
        raise ValueError("Record validated pricing sources and check time")
    checked = datetime.fromisoformat(raw["rates_checked_utc"].replace("Z", "+00:00"))
    if checked.tzinfo is None or abs((datetime.now(timezone.utc)-checked).total_seconds()) > 86400:
        raise ValueError("Refresh rate/resource validation within 24 hours of launch")
    # One additional billed hour is a conservative timing/rounding buffer.
    ceiling = (Decimal(wall)/3600+1)*sum(rates.values())
    ceiling += retention*(rates["ebs"]+rates["public_ipv4"])+reserve
    if ceiling > limit: raise ValueError("Conservative planned costs exceed the authorized allowance")
    return {"passed": True, "planned_ceiling_usd": str(ceiling), "allowance_usd": str(limit),
        "wall_seconds": wall, "conditional_on": "Validated rates/reserve and confirmed <=24h storage cleanup; credits do not reduce this usage estimate"}


def tree_memory(pid):
    """Sample this process tree only. Summed RSS double-counts shared pages."""
    pending, seen, rss, workers = [pid], set(), 0, 0
    while pending:
        current = pending.pop()
        if current in seen: continue
        seen.add(current); root = Path("/proc")/str(current)
        try:
            status = (root/"status").read_text()
            match = re.search(r"^VmRSS:\s+(\d+)", status, re.MULTILINE)
            rss += int(match.group(1)) if match else 0
            command = (root/"cmdline").read_bytes().split(b"\0")
            workers += int(b"experiments.paper_final.run_online" in command and b"worker" in command)
            pending.extend(int(p) for p in (root/f"task/{current}/children").read_text().split())
        except (FileNotFoundError, ProcessLookupError, PermissionError): pass
    return {"rss_sum_kib": rss, "processes": len(seen), "seed_workers": workers}


def run_one(job_id, output, seconds, samples):
    command = [sys.executable, str(c.ROOT/"deployment/aws/campaign.py"), "run", "--job", job_id,
        "--mode", "full", "--timeout-seconds", str(seconds), "--output", str(output)]
    with output.with_suffix(".launcher.log").open("x") as log:
        proc = subprocess.Popen(command, cwd=c.ROOT, stdout=log, stderr=subprocess.STDOUT)
        peak = 0
        deadline = time.monotonic()+seconds
        try:
            while proc.poll() is None:
                if time.monotonic() >= deadline: raise TimeoutError("Singleton exceeded remaining pilot time")
                sample = {"unix_time": time.time(), "job": job_id, **tree_memory(proc.pid)}
                samples.write(json.dumps(sample)+"\n"); samples.flush()
                peak = max(peak, sample["rss_sum_kib"])
                if sample["seed_workers"] > 1: raise RuntimeError("More than one seed worker detected")
                time.sleep(2)
            if proc.returncode: raise RuntimeError(f"{job_id} failed; preserve output and stop the pilot")
        except BaseException:
            # campaign.py handles SIGINT with its existing descendant-group cleanup.
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try: proc.wait(timeout=45)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            raise
    job = c.get_job(job_id)
    c.verify_result(output, job, 6000)
    receipt = c.read(output/"job_receipt.json")
    if not receipt.get("passed") or receipt.get("mode") != "full": raise ValueError("Full-run receipt failed")
    metrics = c.read(output/f"run/seed_{job['seed']}/result.json")["methods"]
    if set(metrics) != set(c.METHODS) or any(v["cases"] != 6000 or v["failures"] != 0 for v in metrics.values()):
        raise ValueError("Baseline correctness gate failed: missing cases or final failures")
    return {"job": job_id, "passed": True, "cases": 6000, "sampled_peak_tree_rss_kib": peak,
        "memory_note": "2-second summed descendant RSS; shared pages counted more than once, brief peaks can be missed. Per-process GNU-time high-water marks are also retained."}


def run(args):
    if sys.platform != "linux": raise ValueError("Serial pilot execution is Linux-only")
    raw = c.read(args.budget_file); checked = budget_check(raw); c.verify_source()
    for key in c.THREADS: os.environ[key] = "1"
    start = float(os.environ.get("RLTUNE_BOOT_STARTED_UNIX", time.time()))
    deadline = start+raw["wall_seconds"]-raw["upload_reserve_seconds"]
    output = args.output.resolve()
    if output.is_relative_to(c.ROOT): raise ValueError("Output must be outside immutable source")
    with Path("/tmp/rltune-serial-pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        state = {**plan(), "budget": checked, "completed": [], "seed1_pair_passed": False, "passed": False}
        c.write(output/"resource-budget.json", raw)
        try:
            with (output/"memory.jsonl").open("x") as samples:
                for index, job in enumerate(ORDER):
                    if index >= 2 and not state["seed1_pair_passed"]: raise RuntimeError("Seed-1 gate not passed")
                    seconds = int(min(14400, deadline-time.time()))
                    if seconds <= 0: raise TimeoutError("Global pilot execution window exhausted; reserve upload time")
                    state["current"] = job; c.write(output/"pilot_receipt.json", state)
                    state["completed"].append(run_one(job, output/job, seconds, samples))
                    if index == 1: state["seed1_pair_passed"] = True
                    c.write(output/"pilot_receipt.json", state)
            if ([r.get("job") for r in state["completed"]] != ORDER
                    or not all(r.get("passed") is True for r in state["completed"])):
                raise RuntimeError("Incomplete six-job pilot cannot pass")
            state.update(passed=True, current=None)
        except BaseException as exc:
            state["error"] = type(exc).__name__+": "+str(exc)
            raise
        finally:
            state["elapsed_since_boot_seconds"] = time.time()-start
            c.write(output/"pilot_receipt.json", state)
            c.output_hashes(output)
    return state


def request(args):
    raw = c.read(args.budget_file); budget_check(raw)
    for key in ["instance_type", "region", "disk_gib", "public_ip"]:
        if getattr(args, key) != raw[key]: raise ValueError("Request differs from validated resource plan")
    args.job = ORDER[0]; args.mode = "full"
    result = c.launch_request(args)
    data = (args.output/"user-data.sh").read_text()
    exports = "\n".join([
        "export RLTUNE_SERIAL_PILOT=1",
        "export RLTUNE_PILOT_WALL_SECONDS="+str(raw["wall_seconds"]),
        "export RLTUNE_PILOT_BUDGET_B64="+shlex.quote(base64.b64encode(json.dumps(raw).encode()).decode())])+"\n"
    data = data.replace("#!/bin/bash\n", "#!/bin/bash\n"+exports, 1)
    uri = args.output_uri.rstrip("/")+"/"+args.campaign+"/serial40-seeds1-3/"+args.bundle_sha256[:12]
    data = re.sub(r"^export RLTUNE_OUTPUT_URI=.*$", "export RLTUNE_OUTPUT_URI="+shlex.quote(uri), data, flags=re.MULTILINE)
    if len(data.encode()) > 16384: raise ValueError("Pilot user data exceeds EC2 limit")
    req = c.read(args.output/"run-instances.json")
    req["UserData"] = base64.b64encode(data.encode()).decode()
    req["ClientToken"] = c.sha((args.campaign+"serial40"+args.bundle_sha256+json.dumps(raw, sort_keys=True)).encode())
    for item in req["TagSpecifications"]:
        for tag in item["Tags"]:
            if tag["Key"] == "Job": tag["Value"] = "serial40-seeds1-3"
        item["Tags"].append({"Key": "Purpose", "Value": "correctness-memory-exploratory-timing"})
    c.write(args.output/"run-instances.json", req); (args.output/"user-data.sh").write_text(data)
    return {**result, "pilot": plan(), "budget": budget_check(raw), "output_uri": uri}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    b = sub.add_parser("validate-budget"); b.add_argument("--budget-file", type=Path, required=True)
    r = sub.add_parser("run"); r.add_argument("--budget-file", type=Path, required=True); r.add_argument("--output", type=Path, required=True)
    l = sub.add_parser("request")
    for key in ["ami", "instance-type", "region", "subnet", "security-group", "instance-profile", "root-device", "campaign", "bundle-uri", "bundle-sha256", "output-uri"]:
        l.add_argument("--"+key, required=True)
    l.add_argument("--budget-file", type=Path, required=True); l.add_argument("--output", type=Path, required=True)
    l.add_argument("--disk-gib", type=int, default=30); l.add_argument("--public-ip", action="store_true")
    args = p.parse_args()
    result = {"plan": plan, "validate-budget": lambda: budget_check(c.read(args.budget_file)),
              "run": lambda: run(args), "request": lambda: request(args)}[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
