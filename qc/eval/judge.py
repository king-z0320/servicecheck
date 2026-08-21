"""Judge request construction and deterministic/Judge result combination."""

from __future__ import annotations

from dataclasses import dataclass

from qc.eval.judge_models import JudgeRequest, JudgeResult


@dataclass(frozen=True)
class CombinedJudgement:
    status: str
    deterministicPassed: bool
    judgeStatus: str
    score: int | None = None


def combine_deterministic_and_judge(deterministic_passed: bool, judge_result: JudgeResult | None) -> CombinedJudgement:
    if not deterministic_passed:
        return CombinedJudgement("failed", False, judge_result.status if judge_result else "not_run", judge_result.score if judge_result else None)
    if judge_result is None or judge_result.status == "not_run":
        return CombinedJudgement("deterministic_completed", True, "not_run", None)
    if judge_result.status != "completed":
        return CombinedJudgement("deterministic_completed", True, judge_result.status, None)
    if judge_result.score is not None and judge_result.score < 3:
        return CombinedJudgement("NEEDS_REVIEW", True, judge_result.status, judge_result.score)
    return CombinedJudgement("passed", True, judge_result.status, judge_result.score)


def build_judge_request(case, report, dimension: str) -> JudgeRequest:
    return JudgeRequest(
        dimension=dimension,
        questionSummary=f"caseId={case.caseId}",
        reportSummary={
            "callId": report.callId,
            "eventTypes": [item.type.value for item in report.events],
            "ruleIds": [item.ruleId for item in report.violations],
            "disposition": report.disposition.value,
        },
        referenceAnswerPoints=case.expected.referenceAnswerPoints,
        evidenceIds=[item.documentId for item in report.knowledgeHits],
    )
