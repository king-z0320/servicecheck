from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from qc.batch.models import BatchFileStatus, BatchMeta, FileRecord, StageName
from qc.batch.service import BatchCapacityError
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
                    "TRUNCATE TABLE review_revisions, review_tasks, batch_dead_letters, outbox_events, "
                    "batch_creation_requests, agent_trace_events, qc_reports, "
                    "stage_executions, batch_exports, batch_items, batch_jobs, "
                    "qc_runs, calls, cases "
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


def test_batch_creation_items_and_outbox_commit_in_one_transaction(database_url: str):
    from qc.batch.postgres_store import PostgresBatchStore

    store = PostgresBatchStore(database_url)
    records = [
        FileRecord(
            source_uri="incoming/a.wav",
            idempotency_key="a",
            metadata={"sha256": "a" * 64, "size": 11},
        ),
        FileRecord(
            source_uri="incoming/b.wav",
            idempotency_key="b",
            metadata={"sha256": "b" * 64, "size": 22},
        ),
    ]
    created = store.create_batch_with_outbox(
        BatchMeta(batch_id="B-OUTBOX", source="incoming", total=2),
        records,
        "request-1",
        request_hash="request-hash-1",
        max_pending=10,
    )

    assert created == {"batch_id": "B-OUTBOX", "status": "QUEUED", "total": 2}
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT COUNT(*) FROM batch_items WHERE batch_id='B-OUTBOX'")
            ).scalar_one() == 2
            payloads = connection.execute(
                text(
                    "SELECT payload FROM outbox_events "
                    "WHERE payload->>'batch_id'='B-OUTBOX' ORDER BY aggregate_id"
                )
            ).scalars().all()
            assert len(payloads) == 2
            snapshots = connection.execute(
                text(
                    "SELECT source_uri, source_sha256, source_size FROM batch_items "
                    "WHERE batch_id='B-OUTBOX' ORDER BY source_uri"
                )
            ).all()
            assert snapshots[0].source_sha256 == "a" * 64
            assert snapshots[0].source_size == 11
            serialized = str(payloads).lower()
            for forbidden in ("authorization", "api_key", "prompt", "transcript"):
                assert forbidden not in serialized
    finally:
        engine.dispose()


def test_batch_creation_rolls_back_batch_items_and_outbox_together(database_url: str):
    from qc.batch.postgres_store import PostgresBatchStore

    store = PostgresBatchStore(database_url)
    invalid = FileRecord(
        source_uri="incoming/a.wav",
        idempotency_key="a",
        callId="CALL-DOES-NOT-EXIST",
    )

    with pytest.raises(Exception):
        store.create_batch_with_outbox(
            BatchMeta(batch_id="B-ROLLBACK", source="incoming", total=1),
            [invalid],
            "request-rollback",
            request_hash="request-hash-rollback",
            max_pending=10,
        )

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            for table in ("batch_jobs", "batch_items", "outbox_events"):
                assert connection.execute(
                    text(f"SELECT COUNT(*) FROM {table} WHERE " + (
                        "batch_id='B-ROLLBACK'" if table != "outbox_events"
                        else "payload->>'batch_id'='B-ROLLBACK'"
                    ))
                ).scalar_one() == 0
    finally:
        engine.dispose()


def test_batch_creation_idempotency_replays_and_rejects_different_request(database_url: str):
    from qc.batch.postgres_store import PostgresBatchStore

    store = PostgresBatchStore(database_url)
    first = store.create_batch_with_outbox(
        BatchMeta(batch_id="B-FIRST", source="incoming", total=1),
        [FileRecord(source_uri="incoming/a.wav", idempotency_key="a")],
        "same-request",
        request_hash="same-hash",
        max_pending=10,
    )
    replay = store.create_batch_with_outbox(
        BatchMeta(batch_id="B-SECOND", source="incoming", total=1),
        [FileRecord(source_uri="incoming/a.wav", idempotency_key="a")],
        "same-request",
        request_hash="same-hash",
        max_pending=10,
    )

    assert replay == first
    with pytest.raises(ValueError, match="different request"):
        store.create_batch_with_outbox(
            BatchMeta(batch_id="B-THIRD", source="other", total=1),
            [FileRecord(source_uri="other/a.wav", idempotency_key="other")],
            "same-request",
            request_hash="different-hash",
            max_pending=10,
        )


def test_batch_creation_enforces_pending_queue_capacity(database_url: str):
    from qc.batch.postgres_store import PostgresBatchStore

    store = PostgresBatchStore(database_url)
    store.create_batch_with_outbox(
        BatchMeta(batch_id="B-CAPACITY-1", source="incoming", total=1),
        [FileRecord(source_uri="incoming/a.wav", idempotency_key="a")],
        max_pending=1,
    )

    with pytest.raises(BatchCapacityError, match="pending queue"):
        store.create_batch_with_outbox(
            BatchMeta(batch_id="B-CAPACITY-2", source="incoming", total=1),
            [FileRecord(source_uri="incoming/b.wav", idempotency_key="b")],
            max_pending=1,
        )


def test_outbox_failure_is_delayed_then_stops_at_retry_limit(database_url: str):
    from qc.batch.postgres_store import PostgresBatchStore

    store = PostgresBatchStore(database_url)
    store.create_batch_with_outbox(
        BatchMeta(batch_id="B-PUBLISH", source="incoming", total=1),
        [FileRecord(source_uri="incoming/a.wav", idempotency_key="a")],
        max_pending=10,
    )
    event = store.pending_outbox_events()[0]

    assert store.mark_outbox_failed(
        event.event_id,
        "ConnectionError",
        max_attempts=2,
        retry_delay_seconds=30,
    ) is False
    assert store.pending_outbox_events() == []
    assert store.mark_outbox_failed(
        event.event_id,
        "ConnectionError",
        max_attempts=2,
        retry_delay_seconds=30,
    ) is True

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT status, attempts, last_error FROM outbox_events "
                    "WHERE event_id=:event_id"
                ),
                {"event_id": event.event_id},
            ).one()
            assert row.status == "FAILED"
            assert row.attempts == 2
            assert row.last_error == "ConnectionError"
    finally:
        engine.dispose()


def test_failed_final_and_dead_letter_are_committed_together(database_url: str):
    from qc.batch.postgres_store import PostgresBatchStore

    store = PostgresBatchStore(database_url)
    store.create_batch_with_outbox(
        BatchMeta(batch_id="B-DEAD", source="incoming", total=1),
        [FileRecord(source_uri="incoming/a.wav", idempotency_key="a")],
        max_pending=10,
    )
    item_id = store.list_files("B-DEAD")[0]["item_id"]
    assert store.claim_file(item_id, BatchFileStatus.PENDING)
    store.begin_stage(item_id, StageName.ASR)
    store.fail_stage(
        item_id,
        StageName.ASR,
        error_code="UPSTREAM_TIMEOUT",
        retryable=True,
        error="上游调用超时",
        duration_ms=1,
    )

    store.finalize_file_with_dead_letter(
        item_id,
        request_json="{}",
        result_json="",
        failed_reason="ASR/UPSTREAM_TIMEOUT: 上游调用超时",
        message_id="1-0",
        stage="ASR",
        error_code="UPSTREAM_TIMEOUT",
        attempts=3,
        last_error="上游调用超时",
        reason="retry exhausted or non-retryable failure",
    )

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT status FROM batch_items WHERE item_id=:item_id"),
                {"item_id": item_id},
            ).scalar_one() == "FAILED_FINAL"
            dead = connection.execute(
                text(
                    "SELECT stage, error_code, attempts FROM batch_dead_letters "
                    "WHERE item_id=:item_id"
                ),
                {"item_id": item_id},
            ).one()
            assert dead == ("ASR", "UPSTREAM_TIMEOUT", 3)
    finally:
        engine.dispose()


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
