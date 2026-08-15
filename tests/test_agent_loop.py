import pytest

from qc.agent_loop import (
    BoundedAgentLoop,
    EvaluatorDecision,
    LLMEvaluator,
    LLMPlanner,
    LoopContext,
    PlannerDecision,
    QualityLoopExecutor,
)
from qc.errors import PipelineFailure
from qc.models import (
    AuditSnapshot,
    EventType,
    KnowledgeHit,
    QualityEvent,
    QualityReport,
    ReviewDisposition,
    TranscriptTurn,
)
from qc.quality_gate import GateResult


class PassingGate:
    def __init__(self):
        self.calls = 0

    def check(self, report, transcript, call_started_at):
        self.calls += 1
        return GateResult(passed=self.calls > 1)


class SequencePlanner:
    def __init__(self):
        self.calls = 0

    def decide(self, context):
        self.calls += 1
        return {
            "type": (
                "EXPAND_CONTEXT"
                if self.calls == 1
                else "REVISE_REPORT"
            )
        }


class FakeExecutor:
    def execute(self, decision, context):
        return {
            "decision": decision["type"],
            "evidenceAdded": decision["type"] == "EXPAND_CONTEXT",
        }


class SequenceEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, context, gate_result):
        self.calls += 1
        if self.calls == 1:
            return {
                "verdict": "NEEDS_MORE_CONTEXT",
                "issues": ["扩大上下文"],
            }
        return {"verdict": "PASS", "issues": []}


def test_complex_case_replans_then_passes():
    report = QualityReport(
        callId="CALL",
        disposition=ReviewDisposition.HUMAN_REVIEW_REQUIRED,
    )
    context = LoopContext(
        report=report,
        transcript=[
            TranscriptTurn(
                turnId="T0001",
                speaker="坐席",
                text="后续流程",
                start=0,
                end=1,
            )
        ],
        reason="AMBIGUOUS_EVENT",
    )
    result = BoundedAgentLoop(
        SequencePlanner(),
        FakeExecutor(),
        SequenceEvaluator(),
        PassingGate(),
        max_iterations=3,
        max_tool_calls=8,
    ).run(context)
    assert result.status == "COMPLETED"
    assert result.iterations == 2
    assert [item.phase for item in result.trace].count("REPLAN") == 1


def test_loop_stops_at_iteration_budget():
    class NeverPassGate:
        def check(self, report, transcript, call_started_at):
            return GateResult(passed=False)

    class NeverPassEvaluator:
        def evaluate(self, context, gate_result):
            return {
                "verdict": "NEEDS_MORE_CONTEXT",
                "issues": ["仍不明确"],
            }

    context = LoopContext(
        report=QualityReport(callId="CALL"),
        transcript=[],
        reason="AMBIGUOUS_EVENT",
    )
    result = BoundedAgentLoop(
        SequencePlanner(),
        FakeExecutor(),
        NeverPassEvaluator(),
        NeverPassGate(),
        max_iterations=2,
        max_tool_calls=8,
    ).run(context)
    assert result.status == "HUMAN_REVIEW_REQUIRED"
    assert result.iterations == 2


def test_executor_rejects_non_allowlisted_action():
    context = LoopContext(
        report=QualityReport(callId="CALL"),
        transcript=[],
        reason="AMBIGUOUS_EVENT",
    )
    executor = QualityLoopExecutor(None, None)
    with pytest.raises(PipelineFailure) as captured:
        executor.execute({"type": "DELETE_RECORD"}, context)
    assert captured.value.error.code == "LOOP_UNSUPPORTED_ACTION"


def test_executor_searches_knowledge_for_existing_event():
    class FakeKnowledge:
        def search(self, query, event_type, at_time, top_k):
            return [
                KnowledgeHit(
                    documentId="POLICY-1",
                    category="POLICY",
                    title="规范",
                    content="内容",
                    version="1.0",
                    score=0.9,
                )
            ]

    event = QualityEvent(
        eventId="E001",
        type=EventType.THREAT_OR_COERCION,
        statement="后续流程",
        turnIds=["T0001"],
        confidence=0.6,
        ambiguous=True,
    )
    context = LoopContext(
        report=QualityReport(callId="CALL", events=[event]),
        transcript=[],
        reason="AMBIGUOUS_EVENT",
    )
    observation = QualityLoopExecutor(
        FakeKnowledge(),
        None,
    ).execute(
        {"type": "SEARCH_KNOWLEDGE", "reason": "法律后果边界"},
        context,
    )
    assert observation["hits"][0]["documentId"] == "POLICY-1"
    assert context.report.knowledgeHits[0].documentId == "POLICY-1"


def test_executor_refreshes_action_audit():
    class FakeAudit:
        def fetch_snapshot(self, call_id):
            return AuditSnapshot(
                callId=call_id,
                disputeTicketCreated=True,
            )

    context = LoopContext(
        report=QualityReport(callId="CALL"),
        transcript=[],
        reason="PARTIAL_AUDIT_FAILURE",
    )
    observation = QualityLoopExecutor(
        None,
        FakeAudit(),
    ).execute({"type": "QUERY_ACTION_AUDIT"}, context)
    assert observation["audit"]["disputeTicketCreated"] is True
    assert context.report.auditSnapshot.disputeTicketCreated is True


def test_planner_and_evaluator_use_separate_structured_requests():
    class RecordingGateway:
        def __init__(self):
            self.systems = []
            self.users = []

        def complete_json(self, system, user, schema, validate, **kwargs):
            self.systems.append(system)
            self.users.append(user)
            if "规划器" in system:
                return validate({"type": "FINALIZE", "reason": "证据已充分"})
            return validate({"verdict": "PASS", "issues": []})

    gateway = RecordingGateway()
    long_turns = [
        TranscriptTurn(
            turnId=f"T{i:04d}",
            speaker="客户" if i % 2 else "坐席",
            text=f"填充句{i} " + ("还清" if i == 2 else "闲聊"),
            start=float(i),
            end=float(i) + 0.5,
        )
        for i in range(1, 40)
    ]
    event = QualityEvent(
        eventId="E001",
        type=EventType.REPAYMENT_DISPUTE,
        statement="还清",
        turnIds=["T0002"],
        confidence=0.7,
        ambiguous=True,
    )
    context = LoopContext(
        report=QualityReport(callId="CALL", events=[event]),
        transcript=long_turns,
        reason="AMBIGUOUS_EVENT",
    )
    decision = LLMPlanner(gateway).decide(context)
    evaluation = LLMEvaluator(gateway).evaluate(
        context,
        GateResult(passed=True),
    )
    assert decision["type"] == "FINALIZE"
    assert evaluation["verdict"] == "PASS"
    assert gateway.systems[0] != gateway.systems[1]
    # 上下文构建：user 侧应是有界 JSON，而不是整份 transcript dump
    assert "caseCard" in gateway.users[0]
    assert gateway.users[0].count("填充句") < 15


def test_typed_planner_and_evaluator_reject_invalid_contracts():
    with pytest.raises(Exception):
        PlannerDecision(type="FINALIZE", reason="   ")
    with pytest.raises(Exception):
        EvaluatorDecision(verdict="PASS", issues=["不应存在"])
    with pytest.raises(Exception):
        EvaluatorDecision(verdict="NEEDS_MORE_CONTEXT", issues=[])


def test_expand_context_grows_bounded_window():
    turns = [
        TranscriptTurn(turnId=f"T{i:04d}", speaker="坐席", text=str(i), start=i, end=i + 1)
        for i in range(1, 8)
    ]
    event = QualityEvent(
        eventId="E1",
        type=EventType.THREAT_OR_COERCION,
        statement="后续",
        turnIds=["T0004"],
        confidence=0.6,
        ambiguous=True,
    )
    context = LoopContext(
        report=QualityReport(callId="CALL", events=[event]),
        transcript=turns,
        reason="AMBIGUOUS_EVENT",
        focusEventId="E1",
        focusTurnIds=["T0004"],
        evidenceRadius=1,
    )
    obs = QualityLoopExecutor(None, None).execute(
        {"type": "EXPAND_CONTEXT", "reason": "扩窗"},
        context,
    )
    assert obs["action"] == "EXPAND_CONTEXT"
    assert context.evidenceRadius >= 2
    assert "T0004" in obs["windowTurnIds"]
    assert len(obs["windowTurnIds"]) < len(turns)


def test_revise_report_drops_invalid_evidence_and_rescores():
    from qc.rules import RuleRepository

    context = LoopContext(
        report=QualityReport(
            callId="CALL",
            score=1,
            violations=[
                __import__("qc.models", fromlist=["Violation"]).Violation(
                    ruleId="R006",
                    ruleName="还款争议处置",
                    penalty=999,
                    evidenceTurnIds=["T9999"],
                    knowledgeDocumentIds=["POLICY-REPAYMENT-003"],
                    explanation="x",
                    suggestion="y",
                )
            ],
        ),
        transcript=[
            TranscriptTurn(turnId="T0001", speaker="客户", text="还完了", start=0, end=1)
        ],
        reason="AMBIGUOUS_EVENT",
    )
    obs = QualityLoopExecutor(
        None,
        None,
        rule_repository=RuleRepository("knowledge/rules/quality_rules.json"),
    ).execute({"type": "REVISE_REPORT", "reason": "清无效证据"}, context)
    assert obs["revised"] is True
    assert context.report.violations == []
    assert context.report.score == 100


def test_loop_budget_exhaustion_returns_structured_error():
    class NeverPassGate:
        def check(self, report, transcript, call_started_at):
            return GateResult(passed=False)

    class NeverPassEvaluator:
        def evaluate(self, context, gate_result):
            return {
                "verdict": "NEEDS_MORE_CONTEXT",
                "issues": ["仍不明确"],
            }

    result = BoundedAgentLoop(
        SequencePlanner(),
        FakeExecutor(),
        NeverPassEvaluator(),
        NeverPassGate(),
        max_iterations=1,
    ).run(
        LoopContext(
            report=QualityReport(callId="CALL"),
            transcript=[],
            reason="AMBIGUOUS_EVENT",
        )
    )

    assert result.status == "HUMAN_REVIEW_REQUIRED"
    assert result.errors[0].code == "LOOP_BUDGET_EXHAUSTED"
