import sqlite3

import pytest

from qc.errors import AnalysisError, ErrorStage, PipelineFailure
from qc.models import AgentTraceEvent, AnalysisRequest, QualityReport, TranscriptTurn
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


def error(code="INTERNAL_ERROR"):
    return AnalysisError(
        code=code,
        stage=ErrorStage.PERSISTENCE,
        message="安全错误摘要",
        retryable=True,
        attempts=0,
    )


def test_run_survives_store_recreation_with_errors_column_migration(tmp_path):
    db = tmp_path / "runs.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            """
            CREATE TABLE agent_runs (
                run_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT
            )
            """
        )
    store = RunStore(db)
    store.create_run("RUN-001", make_request())
    store.append_event(
        "RUN-001",
        AgentTraceEvent(iteration=1, phase="PLAN", message="检索规则"),
    )

    reloaded = RunStore(db).get_run("RUN-001")
    assert reloaded["status"] == "RUNNING"
    assert reloaded["errors"] == []
    assert reloaded["events"][0]["phase"] == "PLAN"


def test_finish_run_persists_completed_partial_and_failed_payloads(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    for status in ("COMPLETED", "PARTIAL", "FAILED"):
        run_id = f"RUN-{status}"
        store.create_run(run_id, make_request())
        report = None if status == "FAILED" else QualityReport(callId="CALL")
        errors = [] if status == "COMPLETED" else [error()]
        store.finish_run(run_id, status, report, errors)
        stored = store.get_run(run_id)
        assert stored["status"] == status
        assert stored["result"] is None if report is None else stored["result"]
        assert len(stored["errors"]) == len(errors)


def test_terminal_state_cannot_be_overwritten(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    store.create_run("RUN-001", make_request())
    store.finish_run("RUN-001", "COMPLETED", QualityReport(callId="CALL"), [])

    with pytest.raises(ValueError, match="terminal"):
        store.finish_run("RUN-001", "FAILED", None, [error()])
    assert store.get_run("RUN-001")["status"] == "COMPLETED"


def test_locked_writes_retry_twice_then_succeed(tmp_path):
    store = RunStore(tmp_path / "runs.db", sleeper=lambda _: None)
    real_connect = store._connect
    calls = 0

    def flaky_connect():
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise sqlite3.OperationalError("database is locked")
        return real_connect()

    store._connect = flaky_connect
    store.create_run("RUN-001", make_request())
    assert calls == 3


def test_three_locked_writes_raise_typed_failure(tmp_path):
    store = RunStore(tmp_path / "runs.db", sleeper=lambda _: None)
    calls = 0

    def locked_connect():
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("database is busy")

    store._connect = locked_connect
    with pytest.raises(PipelineFailure) as captured:
        store.create_run("RUN-001", make_request())
    assert calls == 3
    assert captured.value.error.code == "SQLITE_LOCKED"
    assert captured.value.error.attempts == 3


def test_non_lock_sqlite_error_is_not_retried(tmp_path):
    store = RunStore(tmp_path / "runs.db", sleeper=lambda _: None)
    calls = 0

    def broken_connect():
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("disk I/O error")

    store._connect = broken_connect
    with pytest.raises(PipelineFailure) as captured:
        store.create_run("RUN-001", make_request())
    assert calls == 1
    assert captured.value.error.code == "PERSISTENCE_WRITE_FAILED"


def test_stale_running_runs_are_failed_on_recovery(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    store.create_run("RUN-001", make_request())
    assert store.fail_incomplete_runs(error("PROCESS_INTERRUPTED")) == 1

    stored = store.get_run("RUN-001")
    assert stored["status"] == "FAILED"
    assert stored["errors"][0]["code"] == "PROCESS_INTERRUPTED"


def test_duplicate_run_id_does_not_overwrite_existing_state(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    request = make_request()
    store.create_run("RUN-001", request)
    with pytest.raises(ValueError, match="RUN-001"):
        store.create_run("RUN-001", request)
