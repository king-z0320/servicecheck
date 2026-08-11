from datetime import datetime, timezone

from qc.memory import (
    PROCEDURAL_MEMORY,
    EpisodicMemory,
    MemoryFacade,
    SemanticMemory,
)
from qc.models import (
    AnalysisRequest,
    AnalysisResult,
    QualityEvent,
    QualityReport,
    EventType,
    TranscriptTurn,
    Violation,
)
from qc.run_store import RunStore


def _seed_run(store: RunStore, run_id: str, call_id: str, rule_id: str | None):
    req = AnalysisRequest(
        caseId="CASE",
        callId=call_id,
        transcript=[
            TranscriptTurn(
                turnId="T0001", speaker="客户", text="测试", start=0, end=1
            )
        ],
    )
    store.create_run(run_id, req)
    violations = []
    if rule_id:
        violations.append(
            Violation(
                ruleId=rule_id,
                ruleName="x",
                penalty=20,
                evidenceTurnIds=["T0001"],
                knowledgeDocumentIds=["POLICY-REPAYMENT-003"],
                explanation="e",
                suggestion="s",
            )
        )
    report = QualityReport(
        callId=call_id,
        score=80 if rule_id else 100,
        events=[
            QualityEvent(
                eventId="E1",
                type=EventType.REPAYMENT_DISPUTE,
                statement="还完了",
                turnIds=["T0001"],
                confidence=0.9,
                ambiguous=False,
            )
        ],
        violations=violations,
    )
    store.save_result(run_id, "COMPLETED", report)


def test_procedural_memory_lists_allowlisted_actions():
    assert "EXPAND_CONTEXT" in PROCEDURAL_MEMORY["allowedLoopActions"]
    assert PROCEDURAL_MEMORY["businessFactDefault"] == "NOT_CHECKED"


def test_episodic_memory_find_by_call_and_rule(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    _seed_run(store, "RUN-A", "CALL-1", "R006")
    _seed_run(store, "RUN-B", "CALL-2", "R002")
    epi = EpisodicMemory(store)
    by_call = epi.find_by_call_id("CALL-1")
    assert len(by_call) == 1
    assert by_call[0]["runId"] == "RUN-A"
    by_rule = epi.find_by_rule_id("R006")
    assert any(r["runId"] == "RUN-A" for r in by_rule)
    similar = epi.find_similar_by_event_types(["REPAYMENT_DISPUTE"])
    assert {r["runId"] for r in similar} >= {"RUN-A", "RUN-B"}


def test_episodic_get_episode_roundtrip(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    _seed_run(store, "RUN-C", "CALL-9", "R006")
    episode = EpisodicMemory(store).get_episode("RUN-C")
    assert episode["callId"] == "CALL-9"
    assert "R006" in episode["ruleIds"]


def test_semantic_memory_stats(tmp_path):
    from qc.rag import KnowledgeIndex

    class FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            return [[0.1, 0.2, 0.3] for _ in texts]

    index = KnowledgeIndex("knowledge", embedder=FakeEmbedder())
    index.build()
    sem = SemanticMemory(index)
    stats = sem.stats()
    assert stats["documentCount"] >= 9
    assert stats["byCategory"].get("RULE", 0) >= 3
    assert stats["indexVersion"]


def test_memory_facade_working_snapshot():
    from qc.agent_loop import LoopContext

    class DummyStore:
        pass

    class DummyIndex:
        documents = []
        index_version = "x"

    facade = MemoryFacade(DummyStore(), DummyIndex())  # type: ignore[arg-type]
    ctx = LoopContext(
        report=QualityReport(callId="C"),
        transcript=[],
        reason="AMBIGUOUS_EVENT",
        focusEventId="E1",
        focusTurnIds=["T0001"],
        gaps=["need context"],
    )
    snap = facade.working_snapshot(ctx)
    assert snap["focusEventId"] == "E1"
    assert snap["gaps"] == ["need context"]
