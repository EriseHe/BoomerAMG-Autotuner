"""Offline checks; never import native solvers or contact AWS."""
import argparse
import ast
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("campaign", Path(__file__).with_name("campaign.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CampaignTests(unittest.TestCase):
    def test_all_36_configs_are_unmodified_singletons(self):
        plans = c.plan_payloads()
        jobs = plans["deployment/aws/plan/campaign.json"]["jobs"]
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len({j["id"] for j in jobs}), 36)
        for job in jobs:
            config_path = c.ROOT/job["config"]
            self.assertEqual(config_path.read_bytes(), c.git("show", c.BASE+":"+job["config"]))
            suite = plans[job["suite"]]
            self.assertEqual(len(suite["seeds"]), 1)
            self.assertEqual((c.ROOT/job["suite"]).parent.joinpath(suite["seeds"][0]["config"]).resolve(), config_path)
            self.assertEqual(job["learning_cases"]+job["frozen_cases"], 6000)
            self.assertEqual(job["methods"], c.METHODS)

    def test_scanner_reports_locations_without_secret_values(self):
        fake = b"AK"+b"IA"+b"Z"*16
        report = c.scan({"test.txt": b"first\n"+fake+b"\n", ".aws/credentials": b"placeholder"})
        self.assertFalse(report["passed"])
        self.assertNotIn(fake.decode(), json.dumps(report))
        self.assertIn({"path": "test.txt", "rule": "aws_access_key", "line": 2}, report["findings"])
        self.assertTrue(c.scan({"code.py": b"print('ordinary code')\n"})["passed"])

    def request_args(self, output):
        return argparse.Namespace(campaign="test-20261024", instance_type="approved.type", region="us-east-1",
            bundle_sha256="a"*64, bundle_uri="s3://approved-bucket/source/code.tgz",
            output_uri="s3://approved-bucket/results", job="diffusion_40_seed1", mode="smoke",
            ami="ami-placeholder", subnet="subnet-placeholder", security_group="sg-placeholder",
            instance_profile="approved-role", public_ip=False, root_device="/dev/sda1", disk_gib=30,
            output=output)

    def test_request_is_one_instance_and_stops_without_cloud_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.request_args(Path(tmp)/"request")
            with mock.patch.object(c.subprocess, "Popen", side_effect=AssertionError("Cloud/process call forbidden")):
                result = c.launch_request(args)
            req = c.read(args.output/"run-instances.json")
            self.assertEqual(result["aws_calls"], 0)
            self.assertEqual((req["MinCount"], req["MaxCount"]), (1, 1))
            self.assertEqual(req["MetadataOptions"]["HttpTokens"], "required")
            self.assertEqual(req["InstanceInitiatedShutdownBehavior"], "stop")
            self.assertTrue(req["BlockDeviceMappings"][0]["Ebs"]["Encrypted"])
            self.assertFalse(req["NetworkInterfaces"][0]["AssociatePublicIpAddress"])
            self.assertNotIn("KeyName", req)
            self.assertEqual(base64.b64decode(req["UserData"]), (args.output/"user-data.sh").read_bytes())
            with self.assertRaises(FileExistsError): c.launch_request(args)

    def test_injection_and_unsafe_s3_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for field, value in [("campaign", "bad;command"), ("bundle_uri", "s3://b/x/../secret"),
                                 ("bundle_sha256", "wrong"), ("output_uri", "s3://b/x?credential=foo")]:
                args = self.request_args(Path(tmp)/"unused")
                setattr(args, field, value)
                with self.assertRaises(ValueError): c.launch_request(args)

    def test_timeout_kills_worker_group_even_after_leader_exits(self):
        proc = mock.Mock(pid=987654)
        proc.wait.side_effect = [subprocess.TimeoutExpired("test", 0.01), 0, 0]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(c.subprocess, "Popen", return_value=proc), \
             mock.patch.object(c.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                c.timed_command(["never-executed"], Path(tmp), "test", .01)
            self.assertEqual([call.args[0] for call in kill.call_args_list], [987654, 987654])
            self.assertEqual([call.args[1] for call in kill.call_args_list], [c.signal.SIGTERM, c.signal.SIGKILL])

    def test_output_inventory_never_follows_source_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/"out"; out.mkdir()
            (root/"unrelated-secret").write_text("not a result")
            (out/"outside-link").symlink_to(root/"unrelated-secret")
            (out/"source").mkdir(); (out/"source/file").write_text("source duplicate")
            (out/"result.json").write_text("{}")
            c.output_hashes(out)
            self.assertEqual(set(c.read(out/"OUTPUT_SHA256.json")), {"result.json"})

    def test_result_verification_rejects_missing_or_corrupt_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); run = directory/"run"; stage = run/"seed_1"
            for name in ["status.json", "summary.json", "completion_audit.json"]:
                c.write(run/name, {"passed": True, "status": "complete"})
            result = {"complete": True, "passed": True, "cases": 24, "audits": {}, "checkpoint_sha256": {}}
            for method in c.METHODS:
                path = stage/f"trajectories/{method}.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
                result["audits"][method] = {"cases": 24, "trajectory_sha256": c.sha(path.read_bytes())}
            c.write(stage/"result.json", result)
            job = c.get_job("diffusion_40_seed1")
            self.assertTrue(c.verify_result(directory, job, 24)["passed"])
            (stage/f"trajectories/{c.METHODS[0]}.jsonl").write_text("corrupted")
            with self.assertRaises(ValueError): c.verify_result(directory, job, 24)

    def test_gitless_source_revision_and_both_platform_suffixes(self):
        tree = ast.parse((c.ROOT/"experiments/paper_final/run_online.py").read_text())
        revision = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "source_revision")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c.write(root/"DEPLOYMENT.json", {"base_commit": c.BASE})
            namespace = {"ROOT": root, "read": c.read}
            exec(compile(ast.Module(body=[revision], type_ignores=[]), "revision-test", "exec"), namespace)
            self.assertEqual(namespace["source_revision"](), c.BASE)
            c.write(root/"DEPLOYMENT.json", {"base_commit": "invalid"})
            with self.assertRaises(ValueError): namespace["source_revision"]()
        for platform, suffix in [("Linux", "so"), ("Darwin", "dylib")]:
            output = subprocess.check_output(["make", "-s", "-C", str(c.ROOT/"hypre/bindings"), "print-vars",
                                               "SHARED_EXT="+suffix], text=True)
            self.assertIn("libamg_runtime."+suffix, output)

    def test_mac_execution_is_refused(self):
        with mock.patch.object(c.sys, "platform", "darwin"):
            with self.assertRaises(ValueError): c.run_job(argparse.Namespace())


if __name__ == "__main__":
    unittest.main()
