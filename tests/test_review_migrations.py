from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


pytestmark = pytest.mark.postgres
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def database_url() -> str:
    value = os.getenv("TEST_DATABASE_URL")
    if not value:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL migration tests")
    return value


@pytest.fixture()
def alembic_config(database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def test_stage3_tables_exist_after_head_upgrade(alembic_config: Config, database_url: str):
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert {"review_tasks", "review_revisions"}.issubset(inspector.get_table_names())
        uniques = {
            tuple(item["column_names"])
            for item in inspector.get_unique_constraints("review_tasks")
        }
        assert ("run_id",) in uniques
        revision_uniques = {
            tuple(item["column_names"])
            for item in inspector.get_unique_constraints("review_revisions")
        }
        assert ("task_id",) in revision_uniques
        columns = {item["name"] for item in inspector.get_columns("review_tasks")}
        assert {
            "route_reasons",
            "version",
            "effective_revision_id",
            "unresolved_reason",
            "batch_item_id",
        }.issubset(columns)
    finally:
        engine.dispose()


def test_upgrade_from_0003_does_not_backfill_historical_partial(
    alembic_config: Config,
    database_url: str,
):
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "0003")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO cases(case_id, customer_display_name, source_kind, is_demo, created_at, updated_at)
                    VALUES ('CASE-P', '历史', 'IMPORTED', false, now(), now())
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO calls(
                        call_id, case_id, call_started_at, duration_ms,
                        transcript_json, transcript_version, created_at, updated_at
                    ) VALUES (
                        'CALL-P', 'CASE-P', now(), 1000, CAST('[]' AS jsonb), 'v1', now(), now()
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO qc_runs(
                        run_id, case_id, call_id, status, request_snapshot,
                        errors_json, loop_used, runtime_version, started_at, finished_at
                    ) VALUES (
                        'RUN-P', 'CASE-P', 'CALL-P', 'PARTIAL',
                        CAST('{}' AS jsonb), CAST('[]' AS jsonb), false, 'legacy-unknown', now(), now()
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO qc_reports(
                        report_id, run_id, score, disposition, report_json, created_at
                    ) VALUES (
                        'REPORT-P', 'RUN-P', 80, 'HUMAN_REVIEW_REQUIRED',
                        CAST(:payload AS jsonb), now()
                    )
                    """
                ),
                {
                    "payload": (
                        '{"callId":"CALL-P","score":80,'
                        '"disposition":"HUMAN_REVIEW_REQUIRED"}'
                    )
                },
            )
        command.upgrade(alembic_config, "head")
        with engine.connect() as connection:
            assert connection.execute(text("SELECT COUNT(*) FROM review_tasks")).scalar_one() == 0
            from scripts.backfill_review_tasks import backfill_review_tasks

            first = backfill_review_tasks(database_url)
            second = backfill_review_tasks(database_url)
            assert first == 1
            assert second == 0
            assert connection.execute(text("SELECT COUNT(*) FROM review_tasks")).scalar_one() == 1
    finally:
        engine.dispose()
