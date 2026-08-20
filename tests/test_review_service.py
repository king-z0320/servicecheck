from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from qc.errors import AnalysisError, ErrorStage
from qc.models import QualityReport, ReviewDisposition, TranscriptTurn, AnalysisRequest
from qc.review_models import HumanOutcome, ReasonCode, ReviewSubmitRequest
from qc.review_service import ReviewService, ReviewStateConflict


pytestmark = pytest.mark.postgres


def _database_url():
    value = os.getenv("TEST_DATABASE_URL")
    if not value:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL review tests")
    assert value.startswith("postgresql+psycopg://")
    assert value.rsplit("/", 1)[-1] == "servicecheck_test"
    return value


@pytest.fixture(scope="module")
def database_url():
    value = _database_url()
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", value.replace("%", "%%"))
    command.upgrade(config, "head")
    return value


@pytest.fixture(autouse=True)
def clean_database(database_url):
    if database_url is None:
        yield
        return
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "TRUNCATE TABLE review_revisions, review_tasks, batch_dead_letters, "
                    "outbox_events, batch_creation_requests, agent_trace_events, qc_reports, "
                    "stage_executions, batch_exports, batch_items, batch_jobs, "
                    "qc_runs, calls, cases RESTART IDENTITY CASCADE"
                )
            )
    finally:
        engine.dispose()
    yield


def _request(call_id="CALL-1"):
    return AnalysisRequest(
        caseId="CASE-1",
        callId=call_id,
        callStartedAt=datetime(2025, 10, 15, tzinfo=timezone.utc),
        transcript=[
            TranscriptTurn(turnId="T1", speaker="客户", text="测试", start=0, end=1)
        ],
    )


def _human_report():
    return QualityReport(
        callId="CALL-1",
        score=80,
        disposition=ReviewDisposition.HUMAN_REVIEW_REQUIRED,
        summary={"pendingReviewIssues": [{"code": "RAG_WEAK_SUPPORT"}]},
    )


def _error(code="RAG_WEAK_SUPPORT", stage=ErrorStage.RAG):
    return AnalysisError(code=code, stage=stage, message=code, retryable=False, attempts=1)


def test_partial_run_creates_single_pending_task(database_url):
    from qc.postgres_run_store import PostgresRunStore
    from sqlalchemy import text as sql_text

    store = PostgresRunStore(database_url)
    store.create_run("RUN-1", _request())
    store.finish_run(
        "RUN-1",
        "PARTIAL",
        _human_report(),
        [_error()],
    )
    store.finish_run  # uniqueness via terminal state
    with pytest.raises(ValueError, match="terminal"):
        store.finish_run("RUN-1", "PARTIAL", _human_report(), [_error()])
    payload = store.get_run("RUN-1")
    assert payload["reviewTask"]["status"] == "PENDING"
    assert payload["effectiveRevision"] is None
    assert any(item["code"] == "RAG_WEAK_SUPPORT" for item in payload["reviewTask"]["routeReasons"])
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(sql_text("SELECT COUNT(*) FROM review_tasks")).scalar_one() == 1
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "status,report,errors",
    [
        ("COMPLETED", QualityReport(callId="CALL-1"), []),
        (
            "COMPLETED",
            QualityReport(callId="CALL-1", disposition=ReviewDisposition.AUTO_VIOLATION),
            [],
        ),
        ("FAILED", None, [_error("INTERNAL_ERROR", ErrorStage.PERSISTENCE)]),
    ],
)
def test_non_review_results_do_not_create_tasks(database_url, status, report, errors):
    from qc.postgres_run_store import PostgresRunStore
    from sqlalchemy import text as sql_text

    store = PostgresRunStore(database_url)
    store.create_run("RUN-1", _request())
    store.finish_run("RUN-1", status, report, errors)
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(sql_text("SELECT COUNT(*) FROM review_tasks")).scalar_one() == 0
    finally:
        engine.dispose()
    payload = store.get_run("RUN-1")
    assert payload["reviewTask"] is None


def test_report_and_task_share_transaction_rollback(database_url, monkeypatch):
    from qc.postgres_run_store import PostgresRunStore
    from sqlalchemy import text as sql_text

    def boom(*args, **kwargs):
        raise RuntimeError("task write failed")

    monkeypatch.setattr("qc.postgres_run_store.ensure_review_task_in_session", boom)
    store = PostgresRunStore(database_url)
    store.create_run("RUN-1", _request())
    with pytest.raises(RuntimeError, match="task write failed"):
        store.finish_run("RUN-1", "PARTIAL", _human_report(), [_error()])
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(sql_text("SELECT COUNT(*) FROM qc_reports")).scalar_one() == 0
            assert connection.execute(sql_text("SELECT COUNT(*) FROM review_tasks")).scalar_one() == 0
            assert connection.execute(
                sql_text("SELECT status FROM qc_runs WHERE run_id='RUN-1'")
            ).scalar_one() == "RUNNING"
    finally:
        engine.dispose()


def test_report_write_failure_does_not_leave_task(database_url, monkeypatch):
    from qc import postgres_run_store as module
    from qc.postgres_run_store import PostgresRunStore
    from sqlalchemy import text as sql_text

    def boom_row(**kwargs):
        raise RuntimeError("report write failed")

    monkeypatch.setattr(module, "QCReportRow", boom_row)
    store = PostgresRunStore(database_url)
    store.create_run("RUN-1", _request())
    with pytest.raises(RuntimeError, match="report write failed"):
        store.finish_run("RUN-1", "PARTIAL", _human_report(), [_error()])
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(sql_text("SELECT COUNT(*) FROM review_tasks")).scalar_one() == 0
            assert connection.execute(sql_text("SELECT COUNT(*) FROM qc_reports")).scalar_one() == 0
    finally:
        engine.dispose()


def test_batch_item_association_can_be_filled_after_worker_gap(database_url):
    from qc.batch.models import BatchFileStatus, BatchMeta, FileRecord
    from qc.batch.postgres_store import PostgresBatchStore
    from qc.postgres_run_store import PostgresRunStore
    from sqlalchemy import text as sql_text

    run_store = PostgresRunStore(database_url)
    run_store.create_run("RUN-1", _request())
    run_store.finish_run("RUN-1", "PARTIAL", _human_report(), [_error()])
    batch_store = PostgresBatchStore(database_url)
    batch_store.create_batch(BatchMeta(batch_id="B-1", source="test", total=1))
    batch_store.add_file("B-1", FileRecord(source_uri="a.wav", idempotency_key="a"))
    item_id = batch_store.list_files("B-1")[0]["file_id"]
    assert batch_store.claim_file(item_id, BatchFileStatus.PENDING)
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(
                sql_text("SELECT batch_item_id FROM review_tasks WHERE run_id='RUN-1'")
            ).scalar_one() is None
    finally:
        engine.dispose()
    result_json = json.dumps(
        {
            "runId": "RUN-1",
            "status": "PARTIAL",
            "report": {"callId": "CALL-1", "disposition": "HUMAN_REVIEW_REQUIRED"},
        }
    )
    batch_store.finalize_file(item_id, BatchFileStatus.HUMAN_REVIEW, "{}", result_json)
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            row = connection.execute(
                sql_text(
                    "SELECT batch_item_id, qc_run_id FROM review_tasks t "
                    "JOIN batch_items i ON i.item_id = t.batch_item_id "
                    "WHERE t.run_id='RUN-1'"
                )
            ).one()
            assert row.batch_item_id == item_id
            assert row.qc_run_id == "RUN-1"
            assert connection.execute(sql_text("SELECT COUNT(*) FROM review_tasks")).scalar_one() == 1
    finally:
        engine.dispose()


def test_human_outcomes_keep_original_report_immutable(database_url):
    from qc.postgres_run_store import PostgresRunStore
    from qc.review_store import PostgresReviewStore
    from sqlalchemy import text as sql_text

    run_store = PostgresRunStore(database_url)
    run_store.create_run("RUN-1", _request())
    report = _human_report()
    run_store.finish_run("RUN-1", "PARTIAL", report, [_error()])
    before = run_store.get_report("RUN-1")["report"]
    service = ReviewService(PostgresReviewStore(database_url))
    task_id = run_store.get_review_summary("RUN-1")["reviewTaskId"]
    submitted = service.submit(
        task_id,
        ReviewSubmitRequest(
            expectedVersion=1,
            outcome=HumanOutcome.CONFIRMED_VIOLATION,
            reasonCode=ReasonCode.VIOLATION_CONFIRMED,
            note="确认违规",
        ),
        "idem-1",
    )
    after = run_store.get_report("RUN-1")["report"]
    assert after == before
    assert submitted["effectiveRevision"]["outcome"] == "CONFIRMED_VIOLATION"
    assert submitted["effectiveRevision"]["decisionSource"] == "HUMAN"
    assert submitted["reviewTask"]["status"] == "RESOLVED"
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            row = connection.execute(
                sql_text("SELECT score, disposition FROM qc_reports WHERE run_id='RUN-1'")
            ).one()
            assert row.score == 80
            assert row.disposition == "HUMAN_REVIEW_REQUIRED"
    finally:
        engine.dispose()
    with pytest.raises(ReviewStateConflict):
        service.submit(
            task_id,
            ReviewSubmitRequest(
                expectedVersion=2,
                outcome=HumanOutcome.CONFIRMED_PASS,
                reasonCode=ReasonCode.PASS_CONFIRMED,
            ),
            "idem-2",
        )


def test_unresolved_closes_without_effective_revision(database_url):
    from qc.postgres_run_store import PostgresRunStore
    from qc.review_store import PostgresReviewStore

    run_store = PostgresRunStore(database_url)
    run_store.create_run("RUN-1", _request())
    run_store.finish_run("RUN-1", "PARTIAL", _human_report(), [_error()])
    service = ReviewService(PostgresReviewStore(database_url))
    task_id = run_store.get_review_summary("RUN-1")["reviewTaskId"]
    submitted = service.submit(
        task_id,
        ReviewSubmitRequest(
            expectedVersion=1,
            outcome=HumanOutcome.UNRESOLVED,
            reasonCode=ReasonCode.INSUFFICIENT_EVIDENCE,
            note="仍无法判定",
        ),
        "idem-unresolved",
    )
    assert submitted["reviewTask"]["status"] == "UNRESOLVED"
    assert submitted["effectiveRevision"] is None
    assert submitted["reviewHistory"][0]["outcome"] == "UNRESOLVED"
    payload = run_store.get_run("RUN-1")
    assert payload["result"]["disposition"] == "HUMAN_REVIEW_REQUIRED"
    assert payload["effectiveRevision"] is None
