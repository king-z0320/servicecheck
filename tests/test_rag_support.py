from datetime import datetime, timezone

from qc.models import EventType, KnowledgeHit, QualityEvent, Violation
from qc.rag_support import supporting_hits
from qc.rules import RuleRepository


AT = datetime(2026, 7, 27, tzinfo=timezone.utc)
EVENT = QualityEvent(
    eventId="EVT-" + "A" * 32,
    type=EventType.REPAYMENT_DISPUTE,
    statement="客户称已还款",
    turnIds=["T0001"],
    confidence=0.9,
    ambiguous=False,
)
VIOLATION = Violation(
    eventId=EVENT.eventId,
    ruleId="R006",
    ruleName="还款争议处置",
    penalty=20,
    evidenceTurnIds=["T0001"],
    knowledgeDocumentIds=["POLICY-REPAYMENT-003"],
    explanation="未经核验直接否定",
    suggestion="登记核验",
)
RULE = RuleRepository("knowledge/rules/quality_rules.json").get("R006")


def hit(
    *,
    document_id="POLICY-REPAYMENT-003",
    category="POLICY",
    score=0.8,
    event_type="REPAYMENT_DISPUTE",
    effective_from="2025-01-01T00:00:00Z",
    effective_to=None,
    related_rule_ids=None,
):
    return KnowledgeHit(
        documentId=document_id,
        category=category,
        title="知识",
        content="内容",
        version="1",
        score=score,
        metadata={
            "eventType": event_type,
            "effectiveFrom": effective_from,
            "effectiveTo": effective_to,
            **(
                {"relatedRuleIds": related_rule_ids}
                if related_rule_ids is not None
                else {}
            ),
        },
    )


def supported(items, minimum=0.7):
    violation = VIOLATION.model_copy(deep=True)
    violation.knowledgeDocumentIds = [item.documentId for item in items]
    return supporting_hits(
        violation=violation,
        event=EVENT,
        rule=RULE,
        hits=items,
        at_time=AT,
        min_score=minimum,
    )


def test_policy_source_document_supports_violation():
    assert [item.documentId for item in supported([hit()])] == [
        "POLICY-REPAYMENT-003"
    ]


def test_rule_document_itself_can_support_violation():
    assert supported([hit(document_id="R006", category="RULE")])


def test_low_score_wrong_event_and_inactive_document_do_not_support():
    assert supported([hit(score=0.69)]) == []
    assert supported([hit(event_type="THREAT_OR_COERCION")]) == []
    assert supported([hit(effective_to="2026-01-01T00:00:00Z")]) == []


def test_case_category_is_not_excluded_when_rule_relation_is_explicit():
    case = hit(
        document_id="BAD-REPAYMENT-001",
        category="BAD_CASE",
        related_rule_ids=["R006"],
    )
    assert supported([case]) == [case]


def test_case_without_explicit_rule_relation_cannot_support_violation():
    case = hit(
        document_id="BAD-REPAYMENT-001",
        category="BAD_CASE",
    )
    assert supported([case]) == []
