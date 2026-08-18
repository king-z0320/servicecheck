from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TABLES = {
    "cases",
    "calls",
    "qc_runs",
    "qc_reports",
    "agent_trace_events",
    "batch_jobs",
    "batch_items",
    "stage_executions",
    "batch_exports",
}


@pytest.fixture(scope="module")
def database_url() -> str:
    value = os.getenv("TEST_DATABASE_URL")
    if not value:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL migration tests")
    assert value.startswith("postgresql+psycopg://")
    assert value.rsplit("/", 1)[-1] == "servicecheck_test"
    return value


@pytest.fixture()
def alembic_config(database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def reset_to_base(config: Config) -> None:
    command.downgrade(config, "base")


def test_empty_postgres_database_upgrades_to_head_and_has_expected_schema(
    alembic_config: Config,
    database_url: str,
):
    reset_to_base(alembic_config)
    command.upgrade(alembic_config, "head")

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert EXPECTED_TABLES.issubset(inspector.get_table_names())
        assert engine.dialect.name == "postgresql"
        run_columns = {item["name"]: item for item in inspector.get_columns("qc_runs")}
        assert run_columns["runtime_version"]["nullable"] is False
        report_uniques = {
            tuple(item["column_names"])
            for item in inspector.get_unique_constraints("qc_reports")
        }
        assert ("run_id",) in report_uniques
        item_uniques = {
            tuple(item["column_names"])
            for item in inspector.get_unique_constraints("batch_items")
        }
        assert ("batch_id", "idempotency_key") in item_uniques
    finally:
        engine.dispose()


def test_revision_0001_with_history_upgrades_to_head_without_losing_data(
    alembic_config: Config,
    database_url: str,
):
    reset_to_base(alembic_config)
    command.upgrade(alembic_config, "0001")

    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO cases(
                        case_id, customer_display_name, source_kind, is_demo,
                        created_at, updated_at
                    ) VALUES (
                        'CASE-V1', '历史案件', 'IMPORTED', false, now(), now()
                    )
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
                        'CALL-V1', 'CASE-V1', now(), 1000,
                        CAST('[]' AS jsonb), 'legacy-v1', now(), now()
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO qc_runs(
                        run_id, case_id, call_id, status, request_snapshot,
                        errors_json, loop_used, started_at, finished_at
                    ) VALUES (
                        'RUN-V1', 'CASE-V1', 'CALL-V1', 'COMPLETED',
                        CAST('{}' AS jsonb), CAST('[]' AS jsonb), false, now(), now()
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
                        'REPORT-V1', 'RUN-V1', 100, 'AUTO_PASS',
                        CAST('{"callId":"CALL-V1"}' AS jsonb), now()
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO batch_jobs(batch_id, source, status, total, created_at)
                    VALUES ('BATCH-V1', 'legacy', 'CREATED', 1, now())
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO batch_items(
                        batch_id, source_uri, idempotency_key, status,
                        created_at, updated_at
                    ) VALUES (
                        'BATCH-V1', 'legacy.wav', 'legacy-key', 'PENDING', now(), now()
                    ) RETURNING item_id
                    """
                )
            ).scalar_one()

        command.upgrade(alembic_config, "head")

        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT runtime_version FROM qc_runs WHERE run_id='RUN-V1'"
                )
            ).one()
            assert row.runtime_version == "legacy-unknown"
            assert connection.execute(
                text("SELECT report_json->>'callId' FROM qc_reports WHERE run_id='RUN-V1'")
            ).scalar_one() == "CALL-V1"
            assert connection.execute(
                text("SELECT COUNT(*) FROM batch_items WHERE batch_id='BATCH-V1'")
            ).scalar_one() == 1
    finally:
        engine.dispose()


def test_database_revision_matches_code_head(
    alembic_config: Config,
    database_url: str,
):
    command.upgrade(alembic_config, "head")
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one() == "0002"
    finally:
        engine.dispose()


def test_production_database_modules_do_not_hide_schema_creation():
    sources = []
    for relative in ("api_server.py", "qc/database.py"):
        path = ROOT / relative
        sources.append(path.read_text(encoding="utf-8") if path.exists() else "")
    combined = "\n".join(sources)

    assert "create_all(" not in combined
    assert "ALTER TABLE" not in combined.upper()
