from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_server import create_app
from qc.agent_loop import BoundedAgentLoop, QualityLoopExecutor
from qc.audit_client import AuditClient
from qc.direct_analyzer import DirectAnalyzer
from qc.event_extractor import make_event_id
from qc.models import AuditSnapshot, EventType, KnowledgeHit, QualityEvent
from qc.quality_gate import QualityGate
from qc.rules import RuleRepository
from qc.run_store import RunStore
from qc.service import QualityAnalysisService


MANUAL_MARKERS = {"rag_model", "live_llm", "live_audio"}


def pytest_collection_modifyitems(config, items):
    """Keep default pytest offline; an explicit -m marker opts into live work."""
    expression = config.option.markexpr or ""
    for item in items:
        for marker in MANUAL_MARKERS:
            if marker in item.keywords and marker not in expression:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"manual {marker} test; run with -m {marker}"
                    )
                )


@pytest.fixture
def app_factory():
    def factory(service, **dependencies):
        app = create_app(service=service, **dependencies)
        return TestClient(app)

    return factory


@pytest.fixture
def system_factory(tmp_path):
    db_path = tmp_path / "runs.db"

    class FakeExtractor:
        def __init__(self, ambiguous):
            self.ambiguous = ambiguous

        def extract(self, request):
            event_id, turn_ids = make_event_id(
                request.callId,
                EventType.REPAYMENT_DISPUTE,
                [request.transcript[0].turnId],
                request.transcript,
            )
            return [
                QualityEvent(
                    eventId=event_id,
                    type=EventType.REPAYMENT_DISPUTE,
                    statement=request.transcript[0].text,
                    turnIds=turn_ids,
                    confidence=0.65 if self.ambiguous else 0.99,
                    ambiguous=self.ambiguous,
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
                    metadata={
                        "eventType": event_type.value,
                        "effectiveFrom": "2025-01-01T00:00:00Z",
                        "effectiveTo": None,
                    },
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

    class SequencePlanner:
        def __init__(self):
            self.calls = 0

        def decide(self, context):
            self.calls += 1
            return {
                "type": "EXPAND_CONTEXT" if self.calls == 1 else "FINALIZE",
                "reason": "补充上下文" if self.calls == 1 else "证据充分",
            }

    class SequenceEvaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, context, gate_result):
            self.calls += 1
            if self.calls == 1:
                return {
                    "verdict": "NEEDS_MORE_CONTEXT",
                    "issues": ["需要扩大上下文"],
                }
            return {"verdict": "PASS", "issues": []}

    rules = RuleRepository(ROOT / "knowledge/rules/quality_rules.json")
    knowledge = FakeKnowledge()
    audit = FakeAudit()

    def factory(ambiguous=False):
        direct = DirectAnalyzer(
            FakeExtractor(ambiguous),
            knowledge,
            rules,
            audit,
        )
        loop = BoundedAgentLoop(
            SequencePlanner(),
            QualityLoopExecutor(knowledge, audit, rule_repository=rules),
            SequenceEvaluator(),
            QualityGate(rules),
        )
        gate = QualityGate(rules, min_support_score=0.7)
        return QualityAnalysisService(
            direct,
            loop,
            gate,
            RunStore(db_path),
            min_support_score=0.7,
        )

    def reload_run(run_id):
        return RunStore(db_path).get_run(run_id)

    factory.reload_run = reload_run
    return factory
