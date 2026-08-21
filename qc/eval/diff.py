"""Evaluation baseline/current comparison helpers."""

from __future__ import annotations

from typing import Any


def compare_eval_runs(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    baseline_cases = {item["caseId"]: item for item in baseline.get("caseResults", [])}
    current_cases = {item["caseId"]: item for item in current.get("caseResults", [])}
    lines = ["# 评测 Diff", "", f"- baseline: `{baseline.get('evalRunId', 'unknown')}`", f"- current: `{current.get('evalRunId', 'unknown')}`", ""]
    for case_id in sorted(set(baseline_cases) | set(current_cases)):
        old = baseline_cases.get(case_id, {}).get("status", "not_run")
        new = current_cases.get(case_id, {}).get("status", "not_run")
        if old != new:
            lines.append(f"- `{case_id}`: `{old}` -> `{new}`")
    if len(lines) == 5:
        lines.append("- 无案例状态变化")
    return "\n".join(lines) + "\n"
