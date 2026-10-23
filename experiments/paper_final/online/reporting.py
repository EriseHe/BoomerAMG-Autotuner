from __future__ import annotations

from typing import Any, Dict


def _empty_stream_summary() -> Dict[str, Any]:
    runtime_fields = (
        "setup_runtime",
        "native_solve_runtime",
        "native_total_runtime",
        "controller_runtime",
        "setup_bandit_overhead",
        "setup_bandit_select",
        "setup_bandit_loss_eval",
        "setup_bandit_update",
        "end_to_end_runtime",
    )
    return {
        "cases": 0,
        "failed_count": 0,
        "mean_runtime_sec": 0.0,
        "mean_runtime_with_overhead_sec": 0.0,
        "mean_setup_runtime_sec": 0.0,
        "mean_solve_runtime_sec": 0.0,
        "mean_solve_with_overhead_sec": 0.0,
        "mean_policy_overhead_sec": 0.0,
        "mean_feature_runtime_sec": 0.0,
        "mean_decision_runtime_sec": 0.0,
        "mean_update_runtime_sec": 0.0,
        "mean_iterations": 0.0,
        "totals_sec": {field: 0.0 for field in runtime_fields},
        "means_sec": {field: 0.0 for field in runtime_fields},
        "unique_setup_count": 0,
        "primary_failure_count": 0,
        "setup_fallback_count": 0,
        "recovered_failure_count": 0,
        "unrecovered_failure_count": 0,
        "bandit_update_count": 0,
        "controller_update_count": 0,
    }
