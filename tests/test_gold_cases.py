"""金标回归：用确定性 Fake 抽取器验证直接路径裁决，不调用真实 LLM。"""

from __future__ import annotations

import json
from pathlib import Path

from qc.direct_analyzer import DirectAnalyzer
from qc.event_extractor import EventExtractor
from qc.models import (
    AnalysisRequest,
    AuditSnapshot,
    EventType,
    KnowledgeHit,
    TranscriptTurn,
)
from qc.rules import RuleRepository

GOLD_PATH = Path(__file__).parent / "gold" / "cases.json"


class ScriptedCandidateGateway:
    def __init__(self, candidates: list[dict]):
        self.candidates = candidates

    def complete_json(self, *, system, user, schema, validate):
        return validate({"events": self.candidates})


class GoldKnowledge:
    def __init__(self, score=0.9):
        self.score = score

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
                score=self.score,
                metadata={
                    "eventType": event_type.value,
                    "effectiveFrom": "2025-01-01T00:00:00Z",
                    "effectiveTo": None,
                },
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
    assert len(cases) >= 10
    required_fields = {
        "expectedStatus",
        "callStartedAt",
        "expectedScore",
        "expectedEventCount",
        "expectedViolationCount",
    }
    assert all(required_fields <= set(case) for case in cases)
    rules = RuleRepository("knowledge/rules/quality_rules.json")
    for case in cases:
        extractor = EventExtractor(
            ScriptedCandidateGateway(case["candidates"])
        )
        analyzer = DirectAnalyzer(
            extractor,
            GoldKnowledge(case.get("ragScore", 0.9)),
            rules,
            GoldAudit(case),
        )
        request = AnalysisRequest(
            caseId=case["id"],
            callId=case["callId"],
            callStartedAt=case["callStartedAt"],
            transcript=[TranscriptTurn(**t) for t in case["transcript"]],
        )
        report = analyzer.analyze(request)
        rule_ids = sorted(v.ruleId for v in report.violations)
        assert rule_ids == sorted(case["expectedRuleIds"]), case["id"]
        assert len(report.events) == case["expectedEventCount"], case["id"]
        assert len({event.eventId for event in report.events}) == len(report.events)
        assert len(report.violations) == case["expectedViolationCount"], case["id"]
        assert report.score == case["expectedScore"], case["id"]
        derived_status = (
            "PARTIAL"
            if report.disposition.value == "HUMAN_REVIEW_REQUIRED"
            else "COMPLETED"
        )
        assert derived_status == case["expectedStatus"], case["id"]
        assert report.disposition.value == case["expectedDisposition"], case["id"]
        assert report.businessFact.status.value == case["businessFactMustBe"], case["id"]
