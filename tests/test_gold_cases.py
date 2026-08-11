"""金标回归：用确定性 Fake 抽取器验证直接路径裁决，不调用真实 LLM。"""

from __future__ import annotations

import json
from pathlib import Path

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

GOLD_PATH = Path(__file__).parent / "gold" / "cases.json"


class ScriptedExtractor:
    def __init__(self, event_types: list[str]):
        self.event_types = event_types

    def extract(self, turns):
        events = []
        for i, et in enumerate(self.event_types, 1):
            # 取客户句或坐席句作为 statement/turn
            focus = turns[min(i, len(turns)) - 1]
            for t in turns:
                if et == "REPAYMENT_DISPUTE" and t.speaker == "客户":
                    focus = t
                    break
                if et == "THREAT_OR_COERCION" and t.speaker == "坐席":
                    focus = t
                    break
                if et == "THIRD_PARTY_CONTACT" and t.speaker == "客户":
                    focus = t
                    break
            events.append(
                QualityEvent(
                    eventId=f"E{i}",
                    type=EventType(et),
                    statement=focus.text,
                    turnIds=[focus.turnId],
                    confidence=0.95,
                    ambiguous=False,
                )
            )
        return events


class GoldKnowledge:
    def search(self, query, event_type, at_time, top_k=5):
        mapping = {
            EventType.REPAYMENT_DISPUTE: "POLICY-REPAYMENT-003",
            EventType.THREAT_OR_COERCION: "POLICY-COLLECTION-LANGUAGE-001",
            EventType.THIRD_PARTY_CONTACT: "POLICY-THIRD-PARTY-001",
        }
        doc = mapping.get(event_type, "POLICY-REPAYMENT-003")
        return [
            KnowledgeHit(
                documentId=doc,
                category="POLICY",
                title="gold",
                content="gold",
                version="1.0",
                score=0.9,
                metadata={"eventType": event_type.value},
            )
        ]


class GoldAudit:
    def __init__(self, case: dict):
        self.case = case

    def fetch_snapshot(self, call_id):
        created = self.case.get("auditDisputeTicketCreated")
        if created is None:
            created = call_id != "CALL-NONCOMPLIANT-002"
        return AuditSnapshot(
            callId=call_id,
            crmSummary="demo",
            disputeTicketCreated=created,
            followUpType="CONTINUE_COLLECTION",
        )


def test_gold_direct_path_cases():
    cases = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    rules = RuleRepository("knowledge/rules/quality_rules.json")
    for case in cases:
        extractor = ScriptedExtractor(case["expectedEventTypes"])
        analyzer = DirectAnalyzer(
            extractor,
            GoldKnowledge(),
            rules,
            GoldAudit(case),
        )
        request = AnalysisRequest(
            caseId=case["id"],
            callId=case["callId"],
            transcript=[TranscriptTurn(**t) for t in case["transcript"]],
        )
        report = analyzer.analyze(request)
        rule_ids = {v.ruleId for v in report.violations}
        assert rule_ids == set(case["expectedRuleIds"]), case["id"]
        assert report.disposition.value == case["expectedDisposition"], case["id"]
        assert report.businessFact.status.value == case["businessFactMustBe"], case["id"]
