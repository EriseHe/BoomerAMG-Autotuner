"""Serial gating and cost-guard tests; mocked processes, no solver or AWS calls."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("serial_pilot", Path(__file__).with_name("serial_pilot.py"))
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def budget():
    return {"resource_plan_validated": True, "cleanup_plan_confirmed": True,
        "account_plan": "Free", "region": "us-east-2", "instance_type": "m7i-flex.large", "instance_count": 1,
        "max_total_usage_usd": 10, "wall_seconds": 28800, "upload_reserve_seconds": 1800,
        "disk_gib": 30, "public_ip": True, "hourly_usd": {"ec2": "0.1", "ebs": "0.0042", "public_ipv4": "0.005"},
        "other_costs_reserve_usd": 2, "post_stop_storage_hours": 24,
        "pricing_sources": ["offline test fixture; not a price quote"], "rates_checked_utc": datetime.now(timezone.utc).isoformat()}


class SerialPilotTests(unittest.TestCase):
    def test_exact_six_full_job_order(self):
        self.assertEqual(s.ORDER, ["diffusion_40_seed1", "advection_40_seed1", "diffusion_40_seed2",
            "advection_40_seed2", "diffusion_40_seed3", "advection_40_seed3"])
        self.assertEqual(s.plan()["cases_per_job"], 6000)
        self.assertEqual(s.plan()["instances"], 1)

    def test_pending_rates_and_cleanup_cannot_render_request(self):
        raw = s.c.read(s.c.ROOT/"deployment/aws/pilot-budget.template.json")
        with self.assertRaises(ValueError): s.budget_check(raw)
        for field, value in [("resource_plan_validated", False), ("cleanup_plan_confirmed", False),
                             ("instance_type", "m7a.medium"), ("instance_count", 2), ("max_total_usage_usd", 20)]:
            raw = budget(); raw[field] = value
            with self.assertRaises(ValueError): s.budget_check(raw)

    def test_cost_guard_uses_undiscounted_rates_and_storage_reserve(self):
        result = s.budget_check(budget())
        self.assertLess(float(result["planned_ceiling_usd"]), 10)
        raw = budget(); raw["hourly_usd"]["ec2"] = 2
        with self.assertRaises(ValueError): s.budget_check(raw)
        raw = budget(); raw["hourly_usd"]["public_ipv4"] = 0
        with self.assertRaises(ValueError): s.budget_check(raw)
        raw = budget(); raw["rates_checked_utc"] = "2000-01-01T00:00:00Z"
        with self.assertRaises(ValueError): s.budget_check(raw)

    def test_success_is_serial_and_seed1_pair_passes_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); s.c.write(root/"budget.json", budget())
            seen = []
            def one(job, output, seconds, samples):
                seen.append(job)
                if len(seen) >= 3:
                    self.assertTrue(s.c.read(root/"output/pilot_receipt.json")["seed1_pair_passed"])
                return {"job": job, "passed": True}
            with mock.patch.object(s.sys, "platform", "linux"), mock.patch.object(s.c, "verify_source"), \
                 mock.patch.object(s, "run_one", side_effect=one):
                state = s.run(argparse.Namespace(budget_file=root/"budget.json", output=root/"output"))
            self.assertEqual(seen, s.ORDER); self.assertTrue(state["passed"])

    def test_seed1_pair_failure_stops_before_seeds2_and3(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); s.c.write(root/"budget.json", budget())
            with mock.patch.object(s.sys, "platform", "linux"), mock.patch.object(s.c, "verify_source"), \
                 mock.patch.object(s, "run_one", side_effect=[{"passed": True}, RuntimeError("correctness failed")]) as one:
                with self.assertRaises(RuntimeError):
                    s.run(argparse.Namespace(budget_file=root/"budget.json", output=root/"output"))
                self.assertEqual([x.args[0] for x in one.call_args_list], s.ORDER[:2])
            state = s.c.read(root/"output/pilot_receipt.json")
            self.assertFalse(state["passed"]); self.assertFalse(state["seed1_pair_passed"])
            self.assertTrue((root/"output/OUTPUT_SHA256.json").exists())

    def test_request_is_one_serial_vm_and_under_userdata_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); s.c.write(root/"budget.json", budget())
            args = argparse.Namespace(budget_file=root/"budget.json", instance_type="m7i-flex.large", region="us-east-2",
                disk_gib=30, public_ip=True, campaign="offline-test", ami="ami-placeholder", subnet="subnet-placeholder",
                security_group="sg-placeholder", instance_profile="approved-role", root_device="/dev/sda1",
                bundle_uri="s3://approved-bucket/source/pilot.tgz", bundle_sha256="b"*64,
                output_uri="s3://approved-bucket/results", output=root/"request")
            with mock.patch.object(s.subprocess, "Popen", side_effect=AssertionError("No cloud calls")):
                result = s.request(args)
            req = s.c.read(args.output/"run-instances.json")
            self.assertEqual((req["MinCount"], req["MaxCount"]), (1, 1))
            self.assertIn("serial40-seeds1-3", result["output_uri"])
            text = (args.output/"user-data.sh").read_text()
            self.assertIn("export RLTUNE_SERIAL_PILOT=1", text)
            self.assertLess(len(text.encode()), 16384)

    def test_failure_on_sixth_job_preserves_five_results_without_passing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); s.c.write(root/"budget.json", budget())
            outcomes = [{"job": job, "passed": True} for job in s.ORDER[:5]] + [TimeoutError("deadline")]
            with mock.patch.object(s.sys, "platform", "linux"), mock.patch.object(s.c, "verify_source"), \
                 mock.patch.object(s, "run_one", side_effect=outcomes):
                with self.assertRaises(TimeoutError):
                    s.run(argparse.Namespace(budget_file=root/"budget.json", output=root/"output"))
            state = s.c.read(root/"output/pilot_receipt.json")
            self.assertFalse(state["passed"])
            self.assertTrue(state["seed1_pair_passed"])
            self.assertEqual(len(state["completed"]), 5)

    def test_independent_timer_records_timeout_and_stops_without_termination(self):
        script = (s.c.ROOT/"deployment/aws/bootstrap.sh").read_text()
        self.assertIn("systemd-run --unit=rltune-hard-stop", script)
        self.assertIn('HARD_STOP_SECONDS=${RLTUNE_PILOT_WALL_SECONDS:-28800}', script)
        self.assertIn('/usr/bin/timeout -k 1s 2s', script)
        self.assertIn('exec /sbin/shutdown -h now', script)
        self.assertNotIn('; sync;', script)
        self.assertNotIn("terminate-instances", script)

    def test_mac_pilot_execution_refused(self):
        with mock.patch.object(s.sys, "platform", "darwin"):
            with self.assertRaises(ValueError): s.run(argparse.Namespace())


if __name__ == "__main__": unittest.main()
