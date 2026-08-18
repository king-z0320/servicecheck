from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from qc.batch.models import BatchFileStatus, BatchMeta, FileRecord, StageName
from qc.errors import AnalysisError, ErrorStage
from qc.models import AgentTraceEvent, AnalysisRequest, QualityReport, TranscriptTurn


pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def database_url() -> str:
    value = os.getenv("TEST_DATABASE_URL")
    if not value:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL store tests")
    assert value.startswith("postgresql+psycopg://")
    assert value.rsplit("/", 1)[-1] == "servicecheck_test"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", value.replace("%", "%%"))
    command.upgrade(config, "head")
    return value


@pytest.fixture(autouse=True)
def clean_database(database_url: str):
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "TRUNCATE TABLE agent_trace_events, qc_reports, stage_executions, "
                    "batch_exports, batch_items, batch_jobs, qc_runs, calls, cases "
                    "RESTART IDENTITY CASCADE"
                )
            )
    finally:
        engine.dispose()


def request(call_id: str = "CALL-1") -> AnalysisRequest:
    return AnalysisRequest(
        caseId="CASE-1",
        callId=call_id,
        callStartedAt=datetime(2025, 10, 15, tzinfo=timezone.utc),
        transcript=[
            TranscriptTurn(
                turnId="T1",
                speaker="客户",
                text="测试",
                start=0,
                end=1,
            )
        ],
    )


def error() -> AnalysisError:
    return AnalysisError(
        code="INTERNAL_ERROR",
        stage=ErrorStage.PERSISTENCE,
        message="安全错误摘要",
        retryable=False,
        attempts=1,
    )


def test_run_store_keeps_multiple_immutable_runs_for_one_call(database_url: str):
    from qc.postgres_run_store import PostgresRunStore

    store = PostgresRunStore(
        database_url,
        model="deepseek-chat",
        rule_version="rules-test",
        knowledge_version="knowledge-test",
        runtime_version="runtime-test",
    )
    for index in (1, 2):
        run_id = f"RUN-{index}"
        store.create_run(run_id, request())
        store.append_event(
            run_id,
            AgentTraceEvent(iteration=1, phase="PLAN", message=f"plan-{index}"),
        )
        store.finish_run(
            run_id,
            "COMPLETED",
            QualityReport(callId="CALL-1", score=100 - index),
            [],
        )

    first = store.get_run("RUN-1")
    second = store.get_run("RUN-2")
    history = store.list_runs_by_call("CALL-1")

    assert first["result"]["score"] == 99
    assert first["loopUsed"] is True
    assert first["model"] == "deepseek-chat"
    assert first["ruleVersion"] == "rules-test"
    assert first["knowledgeVersion"] == "knowledge-test"
    assert first["runtimeVersion"] == "runtime-test"
    assert second["result"]["score"] == 98
    assert first["events"][0]["message"] == "plan-1"
    assert [item["runId"] for item in history] == ["RUN-2", "RUN-1"]
    assert len({item["reportId"] for item in history}) == 2


def test_run_terminal_state_cannot_be_overwritten(database_url: str):
    from qc.postgres_run_store import PostgresRunStore

    store = PostgresRunStore(database_url)
    store.create_run("RUN-1", request())
    store.finish_run("RUN-1", "COMPLETED", QualityReport(callId="CALL-1"), [])

    with pytest.raises(ValueError, match="terminal"):
        store.finish_run("RUN-1", "FAILED", None, [error()])

    assert store.get_run("RUN-1")["status"] == "COMPLETED"


def test_failed_run_can_persist_without_report(database_url: str):
    from qc.postgres_run_store import PostgresRunStore

    store = PostgresRunStore(database_url)
    store.create_run("RUN-FAILED", request())
    store.finish_run("RUN-FAILED", "FAILED", None, [error()])

    stored = store.get_run("RUN-FAILED")
    assert stored["status"] == "FAILED"
    assert stored["result"] is None
    assert stored["errors"][0]["code"] == "INTERNAL_ERROR"


def test_batch_idempotency_and_claim_are_database_enforced(database_url: str):
    from qc.batch.postgres_store import PostgresBatchStore

    store = PostgresBatchStore(database_url)
    store.create_batch(BatchMeta(batch_id="B-1", source="test", total=1))
    record = FileRecord(source_uri="a.wav", idempotency_key="same", callId=None)
    assert store.add_file("B-1", record) is True
    assert store.add_file("B-1", record) is False
    item_id = store.list_files("B-1")[0]["file_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(
            pool.map(
                lambda _: store.claim_file(item_id, BatchFileStatus.PENDING),
                range(2),
            )
        )

    assert sorted(claims) == [False, True]
    assert store.get_file(item_id)["status"] == "RUNNING"


def test_batch_stage_checkpoint_round_trips_all_phase_zero_fields(database_url: str):
    from qc.batch.postgres_store import PostgresBatchStore

    store = PostgresBatchStore(database_url)
    store.create_batch(BatchMeta(batch_id="B-1", source="test", total=1))
    store.add_file("B-1", FileRecord(source_uri="a.wav", idempotency_key="one"))
    item_id = store.list_files("B-1")[0]["file_id"]

    assert store.begin_stage(item_id, StageName.ASR) == 1
    store.fail_stage(
        item_id,
        StageName.ASR,
        error_code="UPSTREAM_TIMEOUT",
        retryable=True,
        error="安全错误摘要",
        duration_ms=12.5,
    )
    failed = store.get_stage_checkpoint(item_id, StageName.ASR)
    assert failed["attempts"] == 1
    assert failed["error_code"] == "UPSTREAM_TIMEOUT"
    assert failed["retryable"] is True
    assert failed["error"] == "安全错误摘要"

    assert store.begin_stage(item_id, StageName.ASR) == 2
    store.complete_stage(
        item_id,
        StageName.ASR,
        artifact_uri="batch/B-1/1/transcript.json",
        sha256="a" * 64,
        producer_version="fake-asr-v1",
        duration_ms=9.25,
    )
    completed = store.get_stage_checkpoint(item_id, StageName.ASR)
    assert completed["status"] == "DONE"
    assert completed["attempts"] == 2
    assert completed["artifact_uri"] == "batch/B-1/1/transcript.json"
    assert completed["sha256"] == "a" * 64
    assert completed["producer_version"] == "fake-asr-v1"
    assert completed["started_at"] is not None
    assert completed["finished_at"] is not None
    assert completed["duration_ms"] == pytest.approx(9.25)


def test_batch_terminal_state_and_export_metadata_are_preserved(database_url: str):
    from qc.batch.postgres_store import PostgresBatchStore
    from qc.batch.store import StateConflictError

    store = PostgresBatchStore(database_url)
    store.create_batch(BatchMeta(batch_id="B-1", source="test", total=1))
    store.add_file("B-1", FileRecord(source_uri="a.wav", idempotency_key="one"))
    item_id = store.list_files("B-1")[0]["file_id"]
    assert store.claim_file(item_id, BatchFileStatus.PENDING) is True
    store.finalize_file(item_id, "DONE", "{}", "{}")

    with pytest.raises(StateConflictError):
        store.set_file_status(item_id, "INTERRUPTED")

    uri = "exports/B-1/result.json"
    store.begin_export("B-1", "json", uri, "batch-json-export-v1")
    store.complete_export("B-1", "json", uri, "batch-json-export-v1", "b" * 64)
    export = store.get_export_record("B-1", "json", uri)
    assert export["status"] == "DONE"
    assert export["artifact_uri"] == uri
    assert export["sha256"] == "b" * 64
    assert export["producer_version"] == "batch-json-export-v1"
    assert export["error_code"] is None
