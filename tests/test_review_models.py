from qc.errors import AnalysisError, ErrorStage
from qc.models import QualityReport, ReviewDisposition
from qc.review_models import (
    HumanOutcome,
    ReasonCode,
    ReviewSubmitRequest,
    canonical_submit_hash,
    configured_reviewer_context,
)
from qc.review_service import compute_route_reasons, needs_review_task
from pydantic import ValidationError
import pytest


def test_submit_request_rejects_protected_fields():
    with pytest.raises(ValidationError):
        ReviewSubmitRequest.model_validate(
            {
                "expectedVersion": 1,
                "outcome": "CONFIRMED_PASS",
                "reasonCode": "PASS_CONFIRMED",
                "note": "ok",
                "score": 100,
            }
        )


def test_submit_request_rejects_client_reviewer_id():
    with pytest.raises(ValidationError):
        ReviewSubmitRequest.model_validate(
            {
                "expectedVersion": 1,
                "outcome": "CONFIRMED_VIOLATION",
                "reasonCode": "VIOLATION_CONFIRMED",
                "reviewerId": "attacker",
            }
        )


def test_reason_code_must_match_outcome():
    with pytest.raises(ValidationError):
        ReviewSubmitRequest(
            expectedVersion=1,
            outcome=HumanOutcome.CONFIRMED_PASS,
            reasonCode=ReasonCode.INSUFFICIENT_EVIDENCE,
        )


def test_unresolved_allows_insufficient_evidence():
    request = ReviewSubmitRequest(
        expectedVersion=1,
        outcome=HumanOutcome.UNRESOLVED,
        reasonCode=ReasonCode.INSUFFICIENT_EVIDENCE,
        note="听不清",
    )
    assert request.outcome is HumanOutcome.UNRESOLVED


def test_configured_reviewer_context_is_demo_and_human():
    context = configured_reviewer_context()
    assert context.reviewerId == "configured-demo-reviewer"
    assert context.contextSource.value == "CONFIGURED_DEMO"
    assert context.decisionSource.value == "HUMAN"


def test_request_hash_is_stable_for_same_payload():
    first = canonical_submit_hash("CONFIRMED_PASS", "PASS_CONFIRMED", "a")
    second = canonical_submit_hash("CONFIRMED_PASS", "PASS_CONFIRMED", "a")
    third = canonical_submit_hash("CONFIRMED_PASS", "PASS_CONFIRMED", "b")
    assert first == second
    assert first != third


def test_auto_pass_and_failed_without_report_do_not_need_review():
    assert needs_review_task("COMPLETED", QualityReport(callId="C1")) is False
    assert (
        needs_review_task(
            "COMPLETED",
            QualityReport(callId="C1", disposition=ReviewDisposition.AUTO_VIOLATION),
        )
        is False
    )
    assert needs_review_task("FAILED", None) is False


def test_partial_and_human_review_required_need_review():
    report = QualityReport(
        callId="C1",
        disposition=ReviewDisposition.HUMAN_REVIEW_REQUIRED,
    )
    assert needs_review_task("PARTIAL", report) is True


def test_route_reasons_keep_gate_rag_audit_and_budget_codes():
    report = QualityReport(
        callId="C1",
        disposition=ReviewDisposition.HUMAN_REVIEW_REQUIRED,
        summary={"pendingReviewIssues": [{"code": "NO_ACTIVE_RULE_SUPPORT", "ruleId": "R006"}]},
    )
    errors = [
        AnalysisError(
            code="RAG_WEAK_SUPPORT",
            stage=ErrorStage.RAG,
            message="weak",
            retryable=False,
        ),
        AnalysisError(
            code="LOOP_BUDGET_EXHAUSTED",
            stage=ErrorStage.AGENT_LOOP,
            message="budget",
            retryable=False,
        ),
        AnalysisError(
            code="AUDIT_TIMEOUT",
            stage=ErrorStage.AUDIT,
            message="timeout",
            retryable=True,
        ),
    ]
    reasons = compute_route_reasons("PARTIAL", report, errors)
    codes = {item.code for item in reasons}
    assert {
        "RAG_WEAK_SUPPORT",
        "LOOP_BUDGET_EXHAUSTED",
        "AUDIT_TIMEOUT",
        "NO_ACTIVE_RULE_SUPPORT",
    }.issubset(codes)
    assert any(item.ruleId == "R006" for item in reasons)
