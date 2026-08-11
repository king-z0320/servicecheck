import pytest

from qc.models import (
    AgentTraceEvent,
    AnalysisRequest,
    QualityReport,
    TranscriptTurn,
)
from qc.run_store import RunStore


def make_request():
    return AnalysisRequest(
        caseId="CASE",
        callId="CALL",
        transcript=[
            TranscriptTurn(
                turnId="T0001",
                speaker="客户",
                text="测试",
                start=0,
                end=1,
            )
        ],
    )


def test_run_survives_store_recreation(tmp_path):
    db = tmp_path / "runs.db"
    store = RunStore(db)
    store.create_run("RUN-001", make_request())
    store.append_event(
        "RUN-001",
        AgentTraceEvent(
            iteration=1,
            phase="PLAN",
            message="检索规则",
        ),
    )

    reloaded = RunStore(db).get_run("RUN-001")
    assert reloaded["status"] == "RUNNING"
    assert reloaded["events"][0]["phase"] == "PLAN"


def test_completed_run_is_not_listed_as_incomplete(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    store.create_run("RUN-001", make_request())
    store.save_result(
        "RUN-001",
        "COMPLETED",
        QualityReport(callId="CALL"),
    )
    assert store.list_incomplete() == []


def test_duplicate_run_id_does_not_overwrite_existing_state(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    request = make_request()
    store.create_run("RUN-001", request)
    with pytest.raises(ValueError, match="RUN-001"):
        store.create_run("RUN-001", request)
