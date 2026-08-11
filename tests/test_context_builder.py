from datetime import datetime, timezone

from qc.agent_loop import LoopContext
from qc.context_builder import (
    build_evaluator_user,
    build_evidence_window,
    build_planner_user,
    select_focus_event,
    summarize_observation,
)
from qc.models import (
    EventType,
    QualityEvent,
    QualityReport,
    TranscriptTurn,
)
from qc.quality_gate import GateResult


def _turns():
    return [
        TranscriptTurn(turnId="T0001", speaker="坐席", text="您好", start=0, end=1),
        TranscriptTurn(turnId="T0002", speaker="客户", text="我已经还完了", start=1, end=2),
        TranscriptTurn(turnId="T0003", speaker="坐席", text="不可能", start=2, end=3),
        TranscriptTurn(turnId="T0004", speaker="客户", text="真的还了", start=3, end=4),
        TranscriptTurn(turnId="T0005", speaker="坐席", text="必须处理", start=4, end=5),
    ]


def test_build_evidence_window_centers_on_focus_turns():
    window = build_evidence_window(_turns(), ["T0003"], radius=1)
    assert [t.turnId for t in window] == ["T0002", "T0003", "T0004"]


def test_select_focus_event_prefers_ambiguous_then_first():
    report = QualityReport(
        callId="C",
        events=[
            QualityEvent(
                eventId="E1",
                type=EventType.REPAYMENT_DISPUTE,
                statement="a",
                turnIds=["T0002"],
                confidence=0.9,
                ambiguous=False,
            ),
            QualityEvent(
                eventId="E2",
                type=EventType.THREAT_OR_COERCION,
                statement="b",
                turnIds=["T0005"],
                confidence=0.6,
                ambiguous=True,
            ),
        ],
    )
    focus = select_focus_event(report)
    assert focus is not None
    assert focus.eventId == "E2"


def test_planner_user_is_bounded_and_omits_full_transcript_dump():
    event = QualityEvent(
        eventId="E1",
        type=EventType.REPAYMENT_DISPUTE,
        statement="我已经还完了",
        turnIds=["T0002"],
        confidence=0.9,
        ambiguous=True,
    )
    context = LoopContext(
        report=QualityReport(callId="C", events=[event]),
        transcript=_turns() * 30,  # 故意做长
        reason="AMBIGUOUS_EVENT",
        focusEventId="E1",
        focusTurnIds=["T0002"],
        evidenceWindow=build_evidence_window(_turns(), ["T0002"], radius=1),
        gaps=["需要更多上下文"],
        observations=[{"action": "EXPAND_CONTEXT", "turnCount": 3}],
    )
    user = build_planner_user(context, max_chars=2000)
    assert "我已经还完了" in user
    assert "AMBIGUOUS_EVENT" in user
    assert len(user) <= 2000
    # 不应把 150 句全文无裁剪塞进 prompt
    assert user.count("必须处理") <= 3


def test_evaluator_user_includes_gate_and_stays_bounded():
    context = LoopContext(
        report=QualityReport(callId="C"),
        transcript=_turns(),
        reason="AMBIGUOUS_EVENT",
        evidenceWindow=_turns()[:2],
    )
    user = build_evaluator_user(
        context,
        GateResult(passed=False, issues=[]),
        max_chars=1500,
    )
    assert "GATE" in user
    assert len(user) <= 1500


def test_summarize_observation_keeps_only_keys():
    summary = summarize_observation(
        {
            "action": "SEARCH_KNOWLEDGE",
            "hits": [{"documentId": "P1", "content": "很长" * 100}],
            "turns": [{"turnId": "T1", "text": "x"}],
        }
    )
    assert summary["action"] == "SEARCH_KNOWLEDGE"
    assert "hitCount" in summary
    assert "hits" not in summary or len(str(summary)) < 500
