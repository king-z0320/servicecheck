"""Deterministic event, rule, disposition, and RAG evaluation metrics."""

from __future__ import annotations

from typing import Any

from qc.eval.models import CaseMetrics, EvalCase
from qc.models import QualityReport


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if not denominator else numerator / denominator


def _event_metrics(case: EvalCase, report: QualityReport) -> dict[str, Any]:
    expected = list(case.expected.events)
    predicted = list(report.events)
    used: set[int] = set()
    tp = 0
    evidence_errors = 0
    for target in expected:
        match = None
        for index, event in enumerate(predicted):
            if index in used or event.type.value != target.eventType:
                continue
            if set(target.requiredTurnIds).issubset(set(event.turnIds)):
                match = index
                break
        if match is not None:
            used.add(match)
            tp += 1
        elif any(event.type.value == target.eventType for event in predicted):
            evidence_errors += 1
    fp = len(predicted) - tp
    fn = len(expected) - tp
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "evidenceErrors": evidence_errors,
        "status": "not_applicable" if not expected and not predicted else "completed",
        "passed": tp == len(expected) and fp == 0,
        "calculatorVersion": "events-v1",
    }


def _rules_metrics(case: EvalCase, report: QualityReport) -> dict[str, Any]:
    expected = sorted(set(case.expected.ruleIds))
    actual = sorted({item.ruleId for item in report.violations})
    return {
        "passed": actual == expected,
        "expected": expected,
        "actual": actual,
        "status": "completed",
        "calculatorVersion": "rules-v1",
    }


def _disposition_metrics(case: EvalCase, report: QualityReport) -> dict[str, Any]:
    actual = report.disposition.value
    return {
        "passed": actual in set(case.expected.allowedDispositions),
        "actual": actual,
        "allowed": list(case.expected.allowedDispositions),
        "status": "completed",
        "calculatorVersion": "disposition-v1",
    }


def _rag_metrics(case: EvalCase, report: QualityReport) -> dict[str, Any]:
    hits = list(report.knowledgeHits)
    ids = [item.documentId for item in hits]
    relevant = set(case.expected.relevantContextIds)
    required = set(case.expected.requiredContextIds)
    forbidden = set(case.expected.forbiddenContextIds)

    if relevant:
        precisions = []
        relevant_seen = 0
        for rank, item_id in enumerate(ids, start=1):
            if item_id in relevant:
                relevant_seen += 1
                precisions.append(relevant_seen / rank)
        context_precision: dict[str, Any] = {
            "status": "completed",
            "value": sum(precisions) / len(relevant) if relevant else 0.0,
            "relevantRetrieved": relevant_seen,
            "retrieved": len(ids),
            "calculatorVersion": "rag-v1",
        }
    else:
        context_precision = {"status": "not_run", "value": None, "reason": "missing relevantContextIds"}

    if required:
        matched = required.intersection(ids)
        context_recall: dict[str, Any] = {
            "status": "completed",
            "value": len(matched) / len(required),
            "matched": sorted(matched),
            "missing": sorted(required - matched),
            "calculatorVersion": "rag-v1",
        }
    else:
        context_recall = {"status": "not_run", "value": None, "reason": "missing requiredContextIds"}

    hard_reasons: list[str] = []
    if forbidden.intersection(ids):
        hard_reasons.append("forbidden_context_retrieved")
    hit_map = {item.documentId: item for item in hits}
    for required_id in required:
        if required_id not in hit_map:
            hard_reasons.append(f"missing_context:{required_id}")
    faithfulness_hard = {
        "passed": not hard_reasons,
        "status": "completed" if required or forbidden else "not_run",
        "reasons": hard_reasons,
        "referencedIds": ids,
        "calculatorVersion": "faithfulness-hard-v1",
    }
    return {
        "contextPrecision": context_precision,
        "contextRecall": context_recall,
        "faithfulnessHard": faithfulness_hard,
        "faithfulness": {"status": "not_run", "value": None},
        "answerRelevancy": {"status": "not_run", "value": None},
        "retrievedContextIds": ids,
    }


def calculate_case_metrics(case: EvalCase, report: QualityReport) -> CaseMetrics:
    deterministic = {
        "events": _event_metrics(case, report),
        "rules": _rules_metrics(case, report),
        "disposition": _disposition_metrics(case, report),
    }
    rag = _rag_metrics(case, report)
    failures: list[dict[str, Any]] = []
    for name, value in deterministic.items():
        if value.get("passed") is False:
            failures.append({"metric": name, "reason": "deterministic_mismatch"})
    if rag["faithfulnessHard"].get("passed") is False:
        failures.append({"metric": "faithfulness", "reason": rag["faithfulnessHard"]["reasons"]})
    return CaseMetrics(
        deterministic=deterministic,
        rag=rag,
        status="failed" if failures else "passed",
        failureReasons=failures,
    )


def aggregate_metrics(results: list[CaseMetrics]) -> dict[str, Any]:
    def average(path: tuple[str, ...]) -> float | None:
        values = []
        for result in results:
            value: Any = result.model_dump()
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                values.append(value)
        return sum(values) / len(values) if values else None

    return {
        "caseCount": len(results),
        "passedCases": sum(result.status == "passed" for result in results),
        "failedCases": sum(result.status == "failed" for result in results),
        "eventPrecision": average(("deterministic", "events", "precision")),
        "eventRecall": average(("deterministic", "events", "recall")),
        "eventF1": average(("deterministic", "events", "f1")),
        "contextPrecision": average(("rag", "contextPrecision", "value")),
        "contextRecall": average(("rag", "contextRecall", "value")),
    }
