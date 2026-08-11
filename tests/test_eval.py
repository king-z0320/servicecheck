"""评测模块：金标回归指标 + RAG 命中评测。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from qc.rag import KnowledgeIndex
from qc.rules import RuleRepository

GOLD_CASES = Path(__file__).parent / "gold" / "cases.json"
RAG_CASES = Path(__file__).parent / "gold" / "rag_cases.json"


@dataclass
class EvalCounters:
    total: int = 0
    rule_exact: int = 0
    disposition_ok: int = 0
    business_fact_ok: int = 0
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "ruleExactMatchRate": self.rule_exact / self.total if self.total else 0.0,
            "dispositionAccuracy": (
                self.disposition_ok / self.total if self.total else 0.0
            ),
            "businessFactOkRate": (
                self.business_fact_ok / self.total if self.total else 0.0
            ),
            "failures": list(self.failures),
        }


class ScriptedExtractor:
    def __init__(self, event_types: list[str]):
        self.event_types = event_types

    def extract(self, turns):
        events = []
        for i, et in enumerate(self.event_types, 1):
            focus = turns[0]
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


def evaluate_direct_path_gold(
    gold_path: Path = GOLD_CASES,
) -> EvalCounters:
    cases = json.loads(gold_path.read_text(encoding="utf-8"))
    rules = RuleRepository("knowledge/rules/quality_rules.json")
    counters = EvalCounters()
    for case in cases:
        counters.total += 1
        analyzer = DirectAnalyzer(
            ScriptedExtractor(case["expectedEventTypes"]),
            GoldKnowledge(),
            rules,
            GoldAudit(case),
        )
        report = analyzer.analyze(
            AnalysisRequest(
                caseId=case["id"],
                callId=case["callId"],
                transcript=[TranscriptTurn(**t) for t in case["transcript"]],
            )
        )
        rule_ids = {v.ruleId for v in report.violations}
        expected_rules = set(case["expectedRuleIds"])
        if rule_ids == expected_rules:
            counters.rule_exact += 1
        else:
            counters.failures.append(
                f"{case['id']}: rules got {sorted(rule_ids)} expected {sorted(expected_rules)}"
            )
        if report.disposition.value == case["expectedDisposition"]:
            counters.disposition_ok += 1
        else:
            counters.failures.append(
                f"{case['id']}: disposition {report.disposition.value}"
            )
        if report.businessFact.status.value == case["businessFactMustBe"]:
            counters.business_fact_ok += 1
        else:
            counters.failures.append(f"{case['id']}: businessFact")
    return counters


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            vectors.append(
                [
                    float(
                        "还款" in text
                        or "还清" in text
                        or "REPAYMENT" in text
                        or "R006" in text
                    ),
                    float(
                        "第三方" in text
                        or "家属" in text
                        or "THIRD" in text
                        or "R005" in text
                        or "隐私" in text
                    ),
                    float(
                        "威胁" in text
                        or "抓人" in text
                        or "恐吓" in text
                        or "坐牢" in text
                        or "THREAT" in text
                        or "R002" in text
                    ),
                ]
            )
        return vectors


def evaluate_rag_gold(
    gold_path: Path = RAG_CASES,
    knowledge_root: str = "knowledge",
) -> dict:
    cases = json.loads(gold_path.read_text(encoding="utf-8"))
    index = KnowledgeIndex(knowledge_root, embedder=FakeEmbedder())
    index.build()
    hit = 0
    failures = []
    for case in cases:
        results = index.search(
            case["query"],
            EventType(case["eventType"]),
            datetime(2026, 7, 27, tzinfo=timezone.utc),
            top_k=5,
        )
        ids = {h.documentId for h in results}
        if ids.intersection(case["mustHitAny"]):
            hit += 1
        else:
            failures.append(
                f"{case['id']}: got {sorted(ids)} need any of {case['mustHitAny']}"
            )
    total = len(cases)
    return {
        "total": total,
        "hitAt5Rate": hit / total if total else 0.0,
        "indexVersion": index.index_version,
        "documentCount": len(index.documents),
        "failures": failures,
    }


def test_evaluate_direct_path_gold_metrics():
    metrics = evaluate_direct_path_gold()
    assert metrics.total >= 3
    assert metrics.rule_exact == metrics.total
    assert metrics.disposition_ok == metrics.total
    assert metrics.business_fact_ok == metrics.total
    assert metrics.as_dict()["ruleExactMatchRate"] == 1.0


def test_evaluate_rag_gold_metrics():
    metrics = evaluate_rag_gold()
    assert metrics["total"] >= 3
    assert metrics["hitAt5Rate"] == 1.0
    assert metrics["documentCount"] >= 9
    assert not metrics["failures"]
