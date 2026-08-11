from qc.direct_analyzer import DirectAnalyzer
from qc.models import (
    AnalysisRequest,
    AuditSnapshot,
    EventType,
    KnowledgeHit,
    QualityEvent,
    TranscriptTurn,
)
from qc.rules import RuleRepository


class FakeExtractor:
    def extract(self, turns):
        return [
            QualityEvent(
                eventId="E001",
                type=EventType.REPAYMENT_DISPUTE,
                statement="我已经还完了",
                turnIds=["T0001"],
                confidence=0.99,
                ambiguous=False,
            )
        ]


class FakeKnowledge:
    def search(self, query, event_type, at_time, top_k=5):
        return [
            KnowledgeHit(
                documentId="POLICY-REPAYMENT-003",
                category="POLICY",
                title="还款争议处理规范",
                content="应登记核查，不得未经核实直接否定。",
                version="1.0",
                score=0.95,
                metadata={"eventType": "REPAYMENT_DISPUTE"},
            )
        ]


class FakeAudit:
    def fetch_snapshot(self, call_id):
        return AuditSnapshot(
            callId=call_id,
            crmSummary="客户拒绝还款",
            disputeTicketCreated=False,
            followUpType="CONTINUE_COLLECTION",
        )


def test_direct_path_flags_missing_dispute_ticket_without_checking_debt():
    analyzer = DirectAnalyzer(
        FakeExtractor(),
        FakeKnowledge(),
        RuleRepository("knowledge/rules/quality_rules.json"),
        FakeAudit(),
    )
    result = analyzer.analyze(
        AnalysisRequest(
            caseId="CASE-001",
            callId="CALL-NONCOMPLIANT-002",
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
    )
    assert result.score == 80
    assert result.violations[0].ruleId == "R006"
    assert result.violations[0].evidenceTurnIds == ["T0001", "T0002"]
    assert result.businessFact.status.value == "NOT_CHECKED"
    assert result.disposition.value == "AUTO_VIOLATION"


def test_ambiguous_event_requests_loop_instead_of_auto_penalty():
    class AmbiguousExtractor(FakeExtractor):
        def extract(self, turns):
            event = super().extract(turns)[0]
            event.ambiguous = True
            return [event]

    analyzer = DirectAnalyzer(
        AmbiguousExtractor(),
        FakeKnowledge(),
        RuleRepository("knowledge/rules/quality_rules.json"),
        FakeAudit(),
    )
    report = analyzer.analyze(
        AnalysisRequest(
            caseId="CASE-001",
            callId="CALL-NONCOMPLIANT-002",
            transcript=[
                TranscriptTurn(
                    turnId="T0001",
                    speaker="客户",
                    text="我好像处理过",
                    start=0,
                    end=1,
                )
            ],
        )
    )
    assert report.violations == []
    assert report.disposition.value == "HUMAN_REVIEW_REQUIRED"


class ThreatExtractor:
    def extract(self, turns):
        return [
            QualityEvent(
                eventId="E-THREAT",
                type=EventType.THREAT_OR_COERCION,
                statement="不还就法院抓人",
                turnIds=["T0002"],
                confidence=0.95,
                ambiguous=False,
            )
        ]


class ThirdPartyExtractor:
    def extract(self, turns):
        return [
            QualityEvent(
                eventId="E-3P",
                type=EventType.THIRD_PARTY_CONTACT,
                statement="我是他家属",
                turnIds=["T0001"],
                confidence=0.95,
                ambiguous=False,
            )
        ]


class PolicyKnowledge:
    def __init__(self, document_id, event_type):
        self.document_id = document_id
        self.event_type = event_type

    def search(self, query, event_type, at_time, top_k=5):
        return [
            KnowledgeHit(
                documentId=self.document_id,
                category="POLICY",
                title="规范",
                content="合规要求",
                version="1.0",
                score=0.9,
                metadata={"eventType": self.event_type},
            )
        ]


class EmptyAudit:
    def fetch_snapshot(self, call_id):
        return AuditSnapshot(callId=call_id, disputeTicketCreated=True)


def test_direct_path_flags_threat_language():
    analyzer = DirectAnalyzer(
        ThreatExtractor(),
        PolicyKnowledge("POLICY-COLLECTION-LANGUAGE-001", "THREAT_OR_COERCION"),
        RuleRepository("knowledge/rules/quality_rules.json"),
        EmptyAudit(),
    )
    report = analyzer.analyze(
        AnalysisRequest(
            caseId="CASE-T",
            callId="CALL-T",
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
    assert "T0002" in report.violations[0].evidenceTurnIds
    assert report.businessFact.status.value == "NOT_CHECKED"


def test_direct_path_flags_third_party_privacy_leak():
    analyzer = DirectAnalyzer(
        ThirdPartyExtractor(),
        PolicyKnowledge("POLICY-THIRD-PARTY-001", "THIRD_PARTY_CONTACT"),
        RuleRepository("knowledge/rules/quality_rules.json"),
        EmptyAudit(),
    )
    report = analyzer.analyze(
        AnalysisRequest(
            caseId="CASE-3",
            callId="CALL-3",
            transcript=[
                TranscriptTurn(
                    turnId="T0001",
                    speaker="客户",
                    text="我是他家属，他不在",
                    start=0,
                    end=1,
                ),
                TranscriptTurn(
                    turnId="T0002",
                    speaker="坐席",
                    text="他在我们平台欠款已经逾期很多天了",
                    start=1,
                    end=2,
                ),
            ],
        )
    )
    assert report.score == 70
    assert report.violations[0].ruleId == "R005"
    assert report.disposition.value == "AUTO_VIOLATION"
