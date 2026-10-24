"""Exercise the three-batch launcher with concurrent native seed workers."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from development import run_retry_comparison as comparison
from experiments.paper_final import run_online as runner


class ModuleThreeRetryComparisonTests(unittest.TestCase):
    def test_three_batches_use_the_same_inputs_and_correct_models(self):
        root = runner.ROOT
        with TemporaryDirectory(dir=root/"results", prefix="module3_retry_validation_") as directory:
            base = Path(directory)
            suite = dict(name="Module 3 launcher validation", seeds=[])
            for seed in (1, 2, 3):
                raw = copy.deepcopy(runner.read(root/f"configs/module3/variable_flow_60/seed_{seed}.json"))
                raw["problem"]["grid"] = [12, 12, 12]
                raw["stream"].update(cases=2, online_cases=2, group_take=2, cases_per_seed=1,
                                     expected_sha256="")
                spec = runner.parse_joint_experiment_config(raw)
                args = runner.runtime_config_from_spec(spec)
                runner.configure_paired_environment(args)
                stream, manifest = runner.build_paired_instance_stream(args)
                self.assertEqual(len(stream), 2)
                raw["stream"]["expected_sha256"] = manifest["sha256"]
                config = base/f"seed_{seed}.json"
                runner.dump(config, raw)
                suite["seeds"].append(dict(label=str(seed), config=str(config)))
            runner.dump(base/"suite.json", suite)
            output = base/"run"
            try:
                comparison.sequence(base/"suite.json", output)
            except BaseException:
                for log in sorted(output.rglob("*.log")):
                    print(f"{log.relative_to(output)}: {log.read_text()[-6000:]}")
                raise
            status = runner.read(output/"status.json")
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["completed"], list(comparison.VARIANTS))
            previous_finished = None
            for variant in comparison.VARIANTS:
                batch = output/variant
                release = runner.read(batch/"release.json")
                self.assertEqual(len(set(release["worker_pids"])), 3)
                if previous_finished is not None:
                    self.assertGreaterEqual(release["at"], previous_finished)
                previous_finished = runner.read(batch/"status.json")["at"]
                self.assertTrue(runner.read(batch/"completion_audit.json")["passed"])
                for seed in (1, 2, 3):
                    stage = batch/f"seed_{seed}"
                    declared = runner.read(stage/"retry_selector.json")
                    self.assertEqual(declared["variant"], variant)
                    for name in runner.factorial.METHODS[2:]:
                        with np.load(stage/f"final_bandit_states/{name}.npz", allow_pickle=False) as checkpoint:
                            self.assertEqual("retry_failure_b" in checkpoint.files, variant != "runtime_conservative")
                self.assertTrue(all(row["cases"] == 2 for row in runner.read(batch/"summary.json")["rows"]))
            self.assertEqual(len(runner.read(output/"comparison.json")["rows"]), 36)


if __name__ == "__main__":
    unittest.main()
