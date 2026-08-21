from datetime import datetime, timezone

import pytest

from qc.direct_analyzer import DirectAnalyzer
from qc.event_extractor import make_event_id
from qc.models import (
    AnalysisRequest,
    AuditSnapshot,
    EventType,
    KnowledgeHit,
    QualityEvent,
    TranscriptTurn,
)
from qc.rules import RuleRepository


AT = datetime(2026, 7, 27, tzinfo=timezone.utc)


class FakeExtractor:
    def __init__(self, event_type=EventType.REPAYMENT_DISPUTE, ambiguous=False):
        self.event_type = event_type
        self.ambiguous = ambiguous

    def extract(self, request):
        if self.event_type == EventType.THREAT_OR_COERCION:
            focus = next(t for t in request.transcript if t.speaker == "坐席")
        else:
            focus = request.transcript[0]
        event_id, ordered = make_event_id(
            request.callId,
            self.event_type,
            [focus.turnId],
            request.transcript,
        )
        return [
            QualityEvent(
                eventId=event_id,
                type=self.event_type,
                statement=focus.text,
                turnIds=ordered,
                confidence=0.65 if self.ambiguous else 0.99,
                ambiguous=self.ambiguous,
            )
        ]


class FakeKnowledge:
    def __init__(self, score=0.95):
        self.score = score

    def search(self, query, event_type, at_time, top_k=5):
        mapping = {
            EventType.REPAYMENT_DISPUTE: "POLICY-REPAYMENT-003",
            EventType.THREAT_OR_COERCION: "POLICY-COLLECTION-LANGUAGE-001",
            EventType.THIRD_PARTY_CONTACT: "POLICY-THIRD-PARTY-001",
            EventType.DEBT_DENIAL: "POLICY-DEBT-DENIAL-001",
            EventType.AMOUNT_DISPUTE: "POLICY-AMOUNT-DISPUTE-001",
            EventType.FINANCIAL_HARDSHIP: "POLICY-FINANCIAL-HARDSHIP-001",
            EventType.COMPLAINT_INTENT: "POLICY-COMPLAINT-INTENT-001",
            EventType.STOP_CONTACT_REQUEST: "POLICY-STOP-CONTACT-001",
            EventType.EMOTIONAL_ESCALATION: "POLICY-EMOTIONAL-ESCALATION-001",
        }
        return [
            KnowledgeHit(
                documentId=mapping[event_type],
                category="POLICY",
                title="规范",
                content="合规要求",
                version="1.0",
                score=self.score,
                metadata={
                    "eventType": event_type.value,
                    "effectiveFrom": "2025-01-01T00:00:00Z",
                    "effectiveTo": None,
                },
            )
        ]


class FakeAudit:
    def __init__(self, dispute_ticket_created=False):
        self.dispute_ticket_created = dispute_ticket_created

    def fetch_snapshot(self, call_id):
        return AuditSnapshot(
            callId=call_id,
            crmSummary="客户拒绝还款",
            disputeTicketCreated=self.dispute_ticket_created,
            followUpType="CONTINUE_COLLECTION",
        )


def analyzer(event_type=EventType.REPAYMENT_DISPUTE, score=0.95, ambiguous=False):
    return DirectAnalyzer(
        FakeExtractor(event_type, ambiguous),
        FakeKnowledge(score),
        RuleRepository("knowledge/rules/quality_rules.json"),
        FakeAudit(),
        min_support_score=0.7,
    )


def repayment_request(at=AT):
    return AnalysisRequest(
        caseId="CASE-001",
        callId="CALL-NONCOMPLIANT-002",
        callStartedAt=at,
        transcript=[
            TranscriptTurn(
                turnId="T0001",
                speaker="客户",
                text="我已经还完了",
                start=0,
                end=1,
            ),
            TranscriptTurn(
                turnId="T0002",
                speaker="坐席",
                text="不可能，你今天必须处理",
                start=1,
                end=2,
            ),
        ],
    )


def test_direct_path_uses_active_rule_supported_hit_and_backend_event_id():
    report = analyzer().analyze(repayment_request())

    assert report.score == 80
    assert report.violations[0].ruleId == "R006"
    assert report.violations[0].eventId == report.events[0].eventId
    assert report.violations[0].evidenceTurnIds == ["T0001", "T0002"]
    assert report.violations[0].knowledgeDocumentIds == ["POLICY-REPAYMENT-003"]
    assert report.businessFact.status.value == "NOT_CHECKED"
    assert report.disposition.value == "AUTO_VIOLATION"


def test_ambiguous_event_requests_review_instead_of_auto_penalty():
    report = analyzer(ambiguous=True).analyze(repayment_request())
    assert report.violations == []
    assert report.score == 100
    assert report.disposition.value == "HUMAN_REVIEW_REQUIRED"


def test_low_rag_support_does_not_create_scored_violation():
    report = analyzer(score=0.69).analyze(repayment_request())

    assert report.violations == []
    assert report.score == 100
    assert report.disposition.value == "HUMAN_REVIEW_REQUIRED"
    assert report.summary["pendingReviewIssues"][0]["code"] == "RAG_WEAK_SUPPORT"


def test_inactive_rule_does_not_create_scored_violation():
    report = analyzer().analyze(
        repayment_request(datetime(2024, 12, 31, tzinfo=timezone.utc))
    )

    assert report.violations == []
    assert report.score == 100
    assert report.disposition.value == "HUMAN_REVIEW_REQUIRED"
    assert report.summary["pendingReviewIssues"][0]["code"] == "NO_ACTIVE_RULE_SUPPORT"


def test_direct_path_flags_threat_language_with_active_support():
    report = analyzer(EventType.THREAT_OR_COERCION).analyze(
        AnalysisRequest(
            caseId="CASE-T",
            callId="CALL-T",
            callStartedAt=AT,
            transcript=[
                TranscriptTurn(
                    turnId="T0001", speaker="客户", text="再缓缓", start=0, end=1
                ),
                TranscriptTurn(
                    turnId="T0002",
                    speaker="坐席",
                    text="不还就法院抓人",
                    start=1,
                    end=2,
                ),
            ],
        )
    )

    assert report.score == 70
    assert report.violations[0].ruleId == "R002"
    assert report.disposition.value == "AUTO_VIOLATION"


@pytest.mark.parametrize(
    ("event_type", "customer_text", "agent_text", "rule_id", "penalty"),
    [
        (
            EventType.DEBT_DENIAL,
            "这笔借款不是我借的",
            "别装了，就是你借的，你就是恶意逃债",
            "R003",
            20,
        ),
        (
            EventType.AMOUNT_DISPUTE,
            "你们算的金额不对",
            "系统不可能算错，必须按这个金额处理",
            "R004",
            20,
        ),
        (
            EventType.FINANCIAL_HARDSHIP,
            "我失业了，确实没钱还",
            "没钱也要还，去借钱还，不接受协商",
            "R008",
            20,
        ),
        (
            EventType.COMPLAINT_INTENT,
            "你们这样我就要投诉",
            "随便你投诉，投诉也没用",
            "R009",
            30,
        ),
        (
            EventType.STOP_CONTACT_REQUEST,
            "请不要再打电话了",
            "拉黑也没用，我们还会继续打",
            "R007",
            30,
        ),
        (
            EventType.EMOTIONAL_ESCALATION,
            "你们一直打我真的很烦",
            "你吵什么，闭嘴，发什么疯",
            "R010",
            10,
        ),
    ],
)
def test_direct_path_flags_six_expanded_event_types(
    event_type, customer_text, agent_text, rule_id, penalty
):
    report = analyzer(event_type).analyze(
        AnalysisRequest(
            caseId=f"CASE-{event_type.value}",
            callId=f"CALL-{event_type.value}",
            callStartedAt=AT,
            transcript=[
                TranscriptTurn(
                    turnId="T0001", speaker="客户", text=customer_text, start=0, end=1
                ),
                TranscriptTurn(
                    turnId="T0002", speaker="坐席", text=agent_text, start=1, end=2
                ),
            ],
        )
    )

    assert report.score == 100 - penalty
    assert len(report.violations) == 1
    assert report.violations[0].ruleId == rule_id
    assert report.violations[0].evidenceTurnIds == ["T0001", "T0002"]
    assert report.violations[0].knowledgeDocumentIds == [
        {
            EventType.DEBT_DENIAL: "POLICY-DEBT-DENIAL-001",
            EventType.AMOUNT_DISPUTE: "POLICY-AMOUNT-DISPUTE-001",
            EventType.FINANCIAL_HARDSHIP: "POLICY-FINANCIAL-HARDSHIP-001",
            EventType.COMPLAINT_INTENT: "POLICY-COMPLAINT-INTENT-001",
            EventType.STOP_CONTACT_REQUEST: "POLICY-STOP-CONTACT-001",
            EventType.EMOTIONAL_ESCALATION: "POLICY-EMOTIONAL-ESCALATION-001",
        }[event_type]
    ]
    assert report.disposition.value == "AUTO_VIOLATION"


@pytest.mark.parametrize(
    ("event_type", "customer_text", "agent_text"),
    [
        (EventType.DEBT_DENIAL, "这笔借款不是我借的", "我先记录异议，为您提交核验"),
        (EventType.AMOUNT_DISPUTE, "你们算的金额不对", "我先为您核对账单明细"),
        (EventType.FINANCIAL_HARDSHIP, "我失业了，确实没钱还", "我为您登记困难并说明申请渠道"),
        (EventType.COMPLAINT_INTENT, "你们这样我就要投诉", "我记录您的诉求并提供正式投诉渠道"),
        (EventType.STOP_CONTACT_REQUEST, "请不要再打电话了", "我记录停联请求并转交相关流程"),
        (EventType.EMOTIONAL_ESCALATION, "你们一直打我真的很烦", "我先确认您的诉求，再为您升级处理"),
    ],
)
def test_expanded_event_only_creates_violation_for_agent_language(
    event_type, customer_text, agent_text
):
    report = analyzer(event_type).analyze(
        AnalysisRequest(
            caseId=f"CASE-GOOD-{event_type.value}",
            callId=f"CALL-GOOD-{event_type.value}",
            callStartedAt=AT,
            transcript=[
                TranscriptTurn(
                    turnId="T0001", speaker="客户", text=customer_text, start=0, end=1
                ),
                TranscriptTurn(
                    turnId="T0002", speaker="坐席", text=agent_text, start=1, end=2
                ),
            ],
        )
    )

    assert report.violations == []
    assert report.score == 100
    assert report.disposition.value == "AUTO_PASS"
