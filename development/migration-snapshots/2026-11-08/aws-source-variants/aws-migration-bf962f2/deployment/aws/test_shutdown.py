"""Exercise bootstrap control flow with inert command stubs; no root/AWS/solves."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ShutdownTests(unittest.TestCase):
    def run_prefix(self, *, missing_setting=False, existing_work=False, timer_fails=False,
                   exit_code=0, uploads_fail=False):
        context = tempfile.TemporaryDirectory(prefix="rltune-shutdown-test-")
        self.addCleanup(context.cleanup)
        root = Path(context.name); bindir = root/"bin"; bindir.mkdir()
        work = root/"work"; trace = root/"trace"; timer = root/"timer.json"
        if existing_work: work.mkdir()
        def stub(name, text):
            path = bindir/name; path.write_text(text); path.chmod(0o755); return path
        stop = stub("shutdown", '#!/bin/sh\nprintf "shutdown\\n" >> "$TEST_TRACE"\n')
        stub("systemd-run", '#!/usr/bin/python3\nimport json,os,sys\nfrom pathlib import Path\n'
            'with Path(os.environ["TEST_TRACE"]).open("a") as f:f.write("timer\\n")\n'
            'Path(os.environ["TEST_TIMER"]).write_text(json.dumps(sys.argv[1:]))\n'
            'sys.exit(int(os.environ["TEST_TIMER_FAILS"]))\n')
        stub("timeout", '#!/bin/sh\nshift\nexec "$@"\n')
        stub("aws", '#!/bin/sh\nprintf "aws\\n" >> "$TEST_TRACE"\n'
            'if [ "$TEST_UPLOADS_FAIL" = 1 ]; then exit 1; fi\n'
            'case "$3" in s3://*) cp "$TEST_WORK/results.tgz" "$4";; esac\n')
        script = Path(__file__).with_name("bootstrap.sh").read_text()
        prefix = script.split("export DEBIAN_FRONTEND=noninteractive", 1)[0]
        # Remove only the platform/root precondition; all shell control flow is
        # the actual bootstrap. Replace host paths and stop command with stubs.
        prefix = prefix.replace('[[ $(uname -s) == Linux && $EUID == 0 ]] || exit 64\n', '')
        prefix = prefix.replace('/sbin/shutdown', str(stop)).replace('WORK=/opt/rltune-singleton', 'WORK='+str(work))
        # The Mac execution sandbox disallows /dev/fd process substitution.
        # Keep a real fallible log redirection, without the streaming tee.
        prefix = prefix.replace('exec > >(tee -a "$WORK/bootstrap.log") 2>&1', 'exec >> "$WORK/bootstrap.log" 2>&1')
        self.assertNotIn('/sbin/shutdown', prefix)
        path = root/"prefix.sh"; path.write_text(prefix+f"\nexit {exit_code}\n")
        env = {"PATH": str(bindir)+":/usr/bin:/bin:/usr/sbin:/sbin", "TEST_TRACE": str(trace),
            "TEST_TIMER": str(timer), "TEST_TIMER_FAILS": str(int(timer_fails)), "TEST_WORK": str(work),
            "TEST_UPLOADS_FAIL": str(int(uploads_fail)), "RLTUNE_BUNDLE_URI": "s3://test/source.tgz",
            "RLTUNE_BUNDLE_SHA256": "a"*64, "RLTUNE_OUTPUT_URI": "s3://test/results",
            "RLTUNE_JOB": "diffusion_40_seed1", "RLTUNE_MODE": "full", "RLTUNE_TIMEOUT": "14400",
            "AWS_DEFAULT_REGION": "us-east-2", "RLTUNE_SERIAL_PILOT": "1", "RLTUNE_PILOT_WALL_SECONDS": "28800"}
        if missing_setting: env.pop("RLTUNE_BUNDLE_URI")
        result = subprocess.run(["/bin/bash", str(path)], env=env, capture_output=True, text=True, timeout=15)
        return root, work, trace.read_text().splitlines(), timer, result, env

    def test_missing_setting_arms_timer_then_requests_stop(self):
        _, _, events, _, result, _ = self.run_prefix(missing_setting=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events, ["timer", "shutdown"])

    def test_failed_mkdir_still_requests_stop(self):
        _, _, events, _, result, _ = self.run_prefix(existing_work=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events, ["timer", "shutdown"])

    def test_timer_registration_failure_requests_immediate_stop(self):
        _, work, events, _, result, _ = self.run_prefix(timer_fails=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(work.exists())
        self.assertEqual(events, ["timer", "shutdown"])

    def test_normal_completion_uploads_then_stops(self):
        _, work, events, _, result, _ = self.run_prefix()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(events[0], "timer"); self.assertEqual(events[-1], "shutdown")
        self.assertEqual(events.count("aws"), 4)
        self.assertTrue(json.loads((work/"upload_receipt.json").read_text())["upload_confirmed"])

    def test_failure_and_failed_uploads_still_stop_with_evidence(self):
        _, work, events, _, result, _ = self.run_prefix(exit_code=1, uploads_fail=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(events[-1], "shutdown"); self.assertEqual(events.count("aws"), 3)
        self.assertTrue((work/"results.tgz").exists())
        self.assertFalse((work/"upload_receipt.json").exists())

    def test_deadline_stop_does_not_wait_on_marker_or_sync(self):
        root, _, events, timer, _, env = self.run_prefix(missing_setting=True)
        args = json.loads(timer.read_text())
        self.assertIn("--on-active=28800s", args)
        command = args[args.index("/bin/sh"):]
        self.assertNotIn("sync", command[-1])
        # Simulate a timed-out marker. The stop action must still run, without
        # running a real /usr/bin/timeout or touching /opt on this Mac.
        failed_marker = root/"marker-timeout"; failed_marker.write_text('#!/bin/sh\nexit 124\n'); failed_marker.chmod(0o755)
        command[-1] = command[-1].replace("/usr/bin/timeout", str(failed_marker))
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((root/"trace").read_text().splitlines(), events+["shutdown"])


if __name__ == "__main__": unittest.main()
