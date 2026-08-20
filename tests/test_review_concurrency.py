from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from qc.errors import AnalysisError, ErrorStage
from qc.models import AnalysisRequest, QualityReport, ReviewDisposition, TranscriptTurn
from qc.review_models import HumanOutcome, ReasonCode, ReviewSubmitRequest
from qc.review_service import ReviewService, ReviewStateConflict, ReviewVersionConflict
from qc.review_store import PostgresReviewStore


pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def database_url():
    value = os.getenv("TEST_DATABASE_URL")
    if not value:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL review tests")
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", value.replace("%", "%%"))
    command.upgrade(config, "head")
    return value


@pytest.fixture(autouse=True)
def clean_database(database_url):
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


def _prepare_task(database_url: str) -> str:
    from qc.postgres_run_store import PostgresRunStore

    store = PostgresRunStore(database_url)
    store.create_run(
        "RUN-1",
        AnalysisRequest(
            caseId="CASE-1",
            callId="CALL-1",
            callStartedAt=datetime(2025, 10, 15, tzinfo=timezone.utc),
            transcript=[TranscriptTurn(turnId="T1", speaker="客户", text="测试", start=0, end=1)],
        ),
    )
    store.finish_run(
        "RUN-1",
        "PARTIAL",
        QualityReport(
            callId="CALL-1",
            score=80,
            disposition=ReviewDisposition.HUMAN_REVIEW_REQUIRED,
        ),
        [
            AnalysisError(
                code="RAG_WEAK_SUPPORT",
                stage=ErrorStage.RAG,
                message="weak",
                retryable=False,
            )
        ],
    )
    return store.get_review_summary("RUN-1")["reviewTaskId"]


def test_concurrent_submit_only_one_succeeds(database_url):
    task_id = _prepare_task(database_url)

    def submit(note: str):
        service = ReviewService(PostgresReviewStore(database_url))
        try:
            return service.submit(
                task_id,
                ReviewSubmitRequest(
                    expectedVersion=1,
                    outcome=HumanOutcome.CONFIRMED_PASS
                    if note == "pass"
                    else HumanOutcome.CONFIRMED_VIOLATION,
                    reasonCode=ReasonCode.PASS_CONFIRMED
                    if note == "pass"
                    else ReasonCode.VIOLATION_CONFIRMED,
                    note=note,
                ),
                f"key-{note}",
            )
        except (ReviewVersionConflict, ReviewStateConflict) as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(submit, "pass")
        second = pool.submit(submit, "violation")
        results = [first.result(), second.result()]

    successes = [item for item in results if not isinstance(item, Exception)]
    failures = [item for item in results if isinstance(item, Exception)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert successes[0]["reviewHistory"].__len__() == 1
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("SELECT COUNT(*) FROM review_revisions")).scalar_one() == 1
            version = connection.execute(text("SELECT version FROM review_tasks")).scalar_one()
            assert version == 2
            score = connection.execute(text("SELECT score FROM qc_reports")).scalar_one()
            assert score == 80
    finally:
        engine.dispose()


def test_idempotent_retry_returns_same_revision(database_url):
    task_id = _prepare_task(database_url)
    service = ReviewService(PostgresReviewStore(database_url))
    request = ReviewSubmitRequest(
        expectedVersion=1,
        outcome=HumanOutcome.CONFIRMED_PASS,
        reasonCode=ReasonCode.PASS_CONFIRMED,
        note="same",
    )
    first = service.submit(task_id, request, "same-key")
    second = service.submit(task_id, request, "same-key")
    assert first["reviewHistory"][0]["revisionId"] == second["reviewHistory"][0]["revisionId"]
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("SELECT COUNT(*) FROM review_revisions")).scalar_one() == 1
    finally:
        engine.dispose()


def test_idempotency_conflict_on_different_payload(database_url):
    task_id = _prepare_task(database_url)
    service = ReviewService(PostgresReviewStore(database_url))
    service.submit(
        task_id,
        ReviewSubmitRequest(
            expectedVersion=1,
            outcome=HumanOutcome.CONFIRMED_PASS,
            reasonCode=ReasonCode.PASS_CONFIRMED,
            note="one",
        ),
        "same-key",
    )
    from qc.review_service import ReviewIdempotencyConflict

    with pytest.raises(ReviewIdempotencyConflict):
        service.submit(
            task_id,
            ReviewSubmitRequest(
                expectedVersion=1,
                outcome=HumanOutcome.CONFIRMED_PASS,
                reasonCode=ReasonCode.PASS_CONFIRMED,
                note="two",
            ),
            "same-key",
        )


def test_submit_transaction_failure_leaves_no_half_revision(database_url, monkeypatch):
    task_id = _prepare_task(database_url)
    store = PostgresReviewStore(database_url)
    original_flush = None

    from sqlalchemy.orm import Session

    real_flush = Session.flush

    def boom(self, *args, **kwargs):
        if getattr(self, "_review_boom", False):
            raise RuntimeError("transaction interrupted")
        return real_flush(self, *args, **kwargs)

    monkeypatch.setattr(Session, "flush", boom)
    service = ReviewService(store)
    from sqlalchemy.orm import Session as SessionType

    def begin_wrapper(*args, **kwargs):
        context = store.session_factory.begin(*args, **kwargs)
        return context

    original_begin = store.session_factory.begin

    class FlaggedBegin:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            session = self.inner.__enter__()
            session._review_boom = True
            return session

        def __exit__(self, exc_type, exc, tb):
            return self.inner.__exit__(exc_type, exc, tb)

    def flagged_begin(*args, **kwargs):
        return FlaggedBegin(original_begin(*args, **kwargs))

    monkeypatch.setattr(store.session_factory, "begin", flagged_begin)
    with pytest.raises(RuntimeError, match="transaction interrupted"):
        service.submit(
            task_id,
            ReviewSubmitRequest(
                expectedVersion=1,
                outcome=HumanOutcome.CONFIRMED_PASS,
                reasonCode=ReasonCode.PASS_CONFIRMED,
            ),
            "boom-key",
        )
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("SELECT COUNT(*) FROM review_revisions")).scalar_one() == 0
            row = connection.execute(
                text("SELECT status, version, effective_revision_id FROM review_tasks")
            ).one()
            assert row.status == "PENDING"
            assert row.version == 1
            assert row.effective_revision_id is None
    finally:
        engine.dispose()
