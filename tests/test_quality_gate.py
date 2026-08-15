from datetime import datetime, timezone

from qc.errors import AnalysisError, ErrorStage
from qc.models import (
    AuditSnapshot,
    EventType,
    KnowledgeHit,
    QualityEvent,
    QualityReport,
    ReviewDisposition,
    TranscriptTurn,
    Violation,
)
from qc.quality_gate import QualityGate
from qc.rules import RuleRepository


AT = datetime(2026, 7, 27, tzinfo=timezone.utc)
TRANSCRIPT = [
    TranscriptTurn(
        turnId="T0001",
        speaker="客户",
        text="已还完",
        start=0,
        end=1,
    )
]


def event(event_id="EVT-" + "A" * 32):
    return QualityEvent(
        eventId=event_id,
        type=EventType.REPAYMENT_DISPUTE,
        statement="客户称已还款",
        turnIds=["T0001"],
        confidence=0.9,
        ambiguous=False,
    )


def hit(**overrides):
    data = {
        "documentId": "POLICY-REPAYMENT-003",
        "category": "POLICY",
        "title": "还款争议规范",
        "content": "登记核验",
        "version": "1",
        "score": 0.9,
        "metadata": {
            "eventType": "REPAYMENT_DISPUTE",
            "effectiveFrom": "2025-01-01T00:00:00Z",
            "effectiveTo": None,
        },
    }
    data.update(overrides)
    return KnowledgeHit(**data)


def violation(event_id="EVT-" + "A" * 32, **overrides):
    data = {
        "eventId": event_id,
        "ruleId": "R006",
        "ruleName": "还款争议处置",
        "penalty": 20,
        "evidenceTurnIds": ["T0001"],
        "knowledgeDocumentIds": ["POLICY-REPAYMENT-003"],
        "explanation": "未经核验直接否定",
        "suggestion": "登记核验",
    }
    data.update(overrides)
    return Violation(**data)


def valid_report():
    return QualityReport(
        callId="CALL",
        score=80,
        events=[event()],
        violations=[violation()],
        knowledgeHits=[hit()],
        disposition=ReviewDisposition.AUTO_VIOLATION,
    )


def check(report, at=AT):
    return QualityGate(
        RuleRepository("knowledge/rules/quality_rules.json"),
        min_support_score=0.7,
    ).check(report, TRANSCRIPT, at)


def codes(report, at=AT):
    return {issue.code for issue in check(report, at).issues}


def test_valid_report_passes_gate():
    assert check(valid_report()).passed is True


def test_gate_rejects_missing_transcript_and_event_references():
    report = valid_report()
    report.violations[0].evidenceTurnIds = ["T9999"]
    report.violations[0].eventId = "EVT-" + "B" * 32

    assert {"MISSING_TRANSCRIPT_EVIDENCE", "MISSING_EVENT_REFERENCE"} <= codes(report)


def test_gate_rejects_unknown_rule_without_crashing():
    report = valid_report()
    report.violations[0].ruleId = "R999"
    assert "UNKNOWN_RULE" in codes(report)


def test_rule_source_document_is_not_valid_unless_retrieved():
    report = valid_report()
    report.knowledgeHits = []
    assert "UNKNOWN_POLICY_EVIDENCE" in codes(report)


def test_gate_classifies_low_score_wrong_type_and_inactive_hits():
    report = valid_report()
    report.knowledgeHits[0].score = 0.69
    assert "RAG_BELOW_THRESHOLD" in codes(report)

    report = valid_report()
    report.knowledgeHits[0].metadata["eventType"] = "THREAT_OR_COERCION"
    assert "RAG_EVENT_TYPE_MISMATCH" in codes(report)

    report = valid_report()
    report.knowledgeHits[0].metadata["effectiveTo"] = "2026-01-01T00:00:00Z"
    assert "RAG_DOCUMENT_INACTIVE" in codes(report)


def test_gate_detects_duplicate_event_and_violation():
    report = valid_report()
    report.events.append(event())
    report.violations.append(violation())
    assert {"DUPLICATE_EVENT_ID", "DUPLICATE_VIOLATION"} <= codes(report)


def test_gate_detects_invalid_event_id_penalty_score_and_disposition():
    report = valid_report()
    report.events[0].eventId = "MODEL-ID"
    report.violations[0].eventId = "MODEL-ID"
    report.violations[0].penalty = 999
    report.score = 1
    report.disposition = ReviewDisposition.AUTO_PASS
    assert {
        "INVALID_EVENT_ID_FORMAT",
        "INVALID_PENALTY",
        "INVALID_SCORE",
        "DISPOSITION_CONFLICT",
    } <= codes(report)


def test_audit_error_requires_human_review_disposition():
    report = QualityReport(
        callId="CALL",
        auditSnapshot=AuditSnapshot(
            callId="CALL",
            errors=[
                AnalysisError(
                    code="AUDIT_TIMEOUT",
                    stage=ErrorStage.AUDIT,
                    message="审计超时",
                    retryable=True,
                    attempts=2,
                )
            ],
        ),
        disposition=ReviewDisposition.AUTO_PASS,
    )
    assert {"AUDIT_ERROR_REQUIRES_REVIEW", "DISPOSITION_CONFLICT"} <= codes(report)


def test_inactive_rule_cannot_support_or_reduce_score():
    report = valid_report()
    report.score = 100
    before_effective = datetime(2024, 12, 31, tzinfo=timezone.utc)
    assert "INACTIVE_RULE" in codes(report, before_effective)
