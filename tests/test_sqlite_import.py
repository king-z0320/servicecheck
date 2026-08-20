from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from qc.artifact_store import LocalArtifactStore


pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def database_url() -> str:
    value = os.getenv("TEST_DATABASE_URL")
    if not value:
        pytest.skip("TEST_DATABASE_URL is required for SQLite import tests")
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
                    "TRUNCATE TABLE review_revisions, review_tasks, agent_trace_events, qc_reports, stage_executions, "
                    "batch_exports, batch_items, batch_jobs, qc_runs, calls, cases "
                    "RESTART IDENTITY CASCADE"
                )
            )
    finally:
        engine.dispose()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_legacy_databases(tmp_path: Path) -> tuple[Path, Path]:
    runs = tmp_path / "qc_runs.db"
    request = {
        "caseId": "CASE-OLD",
        "callId": "CALL-OLD",
        "callStartedAt": "2025-10-15T02:25:11Z",
        "transcript": [
            {"turnId": "T1", "speaker": "客户", "text": "测试", "start": 0, "end": 1}
        ],
    }
    report = {
        "callId": "CALL-OLD",
        "score": 80,
        "events": [],
        "violations": [],
        "knowledgeHits": [],
        "auditSnapshot": None,
        "businessFact": {"status": "NOT_CHECKED", "note": "legacy"},
        "disposition": "AUTO_PASS",
        "summary": {},
    }
    with sqlite3.connect(runs) as connection:
        connection.executescript(
            """
            CREATE TABLE agent_runs(
                run_id TEXT PRIMARY KEY, case_id TEXT, call_id TEXT, status TEXT,
                request_json TEXT, result_json TEXT, errors_json TEXT
            );
            CREATE TABLE agent_events(
                id INTEGER PRIMARY KEY, run_id TEXT, iteration INTEGER,
                phase TEXT, event_json TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO agent_runs VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                "RUN-OLD",
                "CASE-OLD",
                "CALL-OLD",
                "COMPLETED",
                json.dumps(request, ensure_ascii=False),
                json.dumps(report, ensure_ascii=False),
                "[]",
            ),
        )
        connection.execute(
            "INSERT INTO agent_events VALUES(1, ?, 1, 'PLAN', ?)",
            (
                "RUN-OLD",
                json.dumps(
                    {"iteration": 1, "phase": "PLAN", "message": "legacy plan", "details": {}},
                    ensure_ascii=False,
                ),
            ),
        )

    batch = tmp_path / "batch.db"
    with sqlite3.connect(batch) as connection:
        connection.executescript(
            """
            CREATE TABLE batches(batch_id TEXT PRIMARY KEY, source TEXT, created_at TEXT, total INTEGER);
            CREATE TABLE batch_files(
                file_id INTEGER PRIMARY KEY, batch_id TEXT, source_uri TEXT,
                call_id TEXT, idempotency_key TEXT, status TEXT,
                failed_reason TEXT, request_json TEXT, result_json TEXT
            );
            CREATE TABLE file_stages(
                id INTEGER PRIMARY KEY, file_id INTEGER, stage TEXT, status TEXT,
                started_at TEXT, finished_at TEXT, duration_ms REAL, attempts INTEGER,
                artifact_uri TEXT, sha256 TEXT, producer_version TEXT,
                error_code TEXT, retryable INTEGER, error TEXT
            );
            CREATE TABLE batch_exports(
                export_id INTEGER PRIMARY KEY, batch_id TEXT, format TEXT,
                artifact_uri TEXT, status TEXT, sha256 TEXT,
                producer_version TEXT, error_code TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO batches VALUES('B-OLD', 'directory', '2025-10-15T00:00:00Z', 1)"
        )
        connection.execute(
            "INSERT INTO batch_files VALUES(1, 'B-OLD', 'old.wav', NULL, 'idem', 'PENDING', NULL, NULL, NULL)"
        )
        connection.execute(
            "INSERT INTO file_stages VALUES(1, 1, 'ASR', 'DONE', NULL, NULL, 12.5, 2, "
            "'batch/B-OLD/1/missing.json', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', "
            "'legacy-asr', NULL, NULL, NULL)"
        )
        connection.execute(
            "INSERT INTO batch_exports VALUES(1, 'B-OLD', 'json', 'exports/B-OLD.json', "
            "'FAILED', NULL, 'legacy-export', 'EXPORT_FAILED')"
        )
    return runs, batch


def table_counts(database_url: str) -> dict[str, int]:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return {
                table: connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
                for table in (
                    "cases",
                    "calls",
                    "qc_runs",
                    "qc_reports",
                    "agent_trace_events",
                    "batch_jobs",
                    "batch_items",
                    "stage_executions",
                    "batch_exports",
                )
            }
    finally:
        engine.dispose()


def test_dry_run_is_read_only_and_reports_planned_rows(database_url: str, tmp_path):
    from scripts.migrate_sqlite_to_postgres import migrate

    runs, batch = create_legacy_databases(tmp_path)
    before = {runs: sha256(runs), batch: sha256(batch)}
    report = migrate(
        runs,
        batch,
        database_url,
        LocalArtifactStore(tmp_path / "artifacts"),
        dry_run=True,
    )

    assert report["sourceCounts"]["agent_runs"] == 1
    assert report["sourceCounts"]["file_stages"] == 1
    assert report["planned"]["qc_reports"] == 1
    assert table_counts(database_url) == {name: 0 for name in table_counts(database_url)}
    assert {runs: sha256(runs), batch: sha256(batch)} == before


def test_import_is_transactional_idempotent_and_preserves_sources(database_url: str, tmp_path):
    from scripts.migrate_sqlite_to_postgres import migrate

    runs, batch = create_legacy_databases(tmp_path)
    before = {runs: sha256(runs), batch: sha256(batch)}
    artifacts = LocalArtifactStore(tmp_path / "artifacts")

    first = migrate(runs, batch, database_url, artifacts)
    second = migrate(runs, batch, database_url, artifacts)

    assert first["inserted"]["qc_runs"] == 1
    assert first["inserted"]["stage_executions"] == 1
    assert first["inserted"]["batch_exports"] == 1
    assert all(value == 0 for value in second["inserted"].values())
    assert table_counts(database_url) == {
        "cases": 1,
        "calls": 1,
        "qc_runs": 1,
        "qc_reports": 1,
        "agent_trace_events": 1,
        "batch_jobs": 1,
        "batch_items": 1,
        "stage_executions": 1,
        "batch_exports": 1,
    }
    assert first["missingArtifacts"][0]["uri"] == "batch/B-OLD/1/missing.json"
    assert {runs: sha256(runs), batch: sha256(batch)} == before


def test_import_copies_valid_legacy_checkpoint_to_logical_uri_and_reuses_it(
    database_url: str,
    tmp_path,
):
    from qc.batch.checkpoints import FileCheckpointSession
    from qc.batch.models import StageName
    from qc.batch.pipeline import execute_stage
    from qc.batch.postgres_store import PostgresBatchStore
    from qc.models import TranscriptTurn
    from scripts.migrate_sqlite_to_postgres import migrate

    runs, batch = create_legacy_databases(tmp_path)
    legacy_artifact = tmp_path / "legacy-batch-artifacts" / "transcript.json"
    legacy_artifact.parent.mkdir()
    legacy_artifact.write_text(
        json.dumps(
            [
                {
                    "turnId": "T1",
                    "speaker": "客户",
                    "text": "从旧检查点恢复",
                    "start": 0,
                    "end": 1,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    legacy_export = tmp_path / "legacy-batch-artifacts" / "export.json"
    legacy_export.write_text('{"batch_id":"B-OLD"}', encoding="utf-8")
    with sqlite3.connect(batch) as connection:
        connection.execute(
            "UPDATE file_stages SET artifact_uri=?, sha256=? WHERE id=1",
            (str(legacy_artifact), sha256(legacy_artifact)),
        )
        connection.execute(
            "UPDATE batch_exports SET artifact_uri=?, status='DONE', sha256=?, error_code=NULL "
            "WHERE export_id=1",
            (str(legacy_export), sha256(legacy_export)),
        )

    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    report = migrate(runs, batch, database_url, artifacts)
    store = PostgresBatchStore(database_url)
    try:
        checkpoint = store.get_stage_checkpoint(1, StageName.ASR)
        assert checkpoint["artifact_uri"] == "batch/B-OLD/1/transcript.json"
        assert artifacts.verify_sha256(
            checkpoint["artifact_uri"], checkpoint["sha256"]
        )
        export = store.get_export_record("B-OLD", "json", "exports/B-OLD/1.json")
        assert export["status"] == "DONE"
        assert artifacts.verify_sha256(export["artifact_uri"], export["sha256"])
        assert report["artifactCopiesPlanned"] == 2

        session = FileCheckpointSession(
            store,
            "B-OLD",
            1,
            artifact_store=artifacts,
        )
        calls = 0

        def should_not_run():
            nonlocal calls
            calls += 1
            return [TranscriptTurn(turnId="NEW", speaker="客户", text="不应执行", end=1)]

        restored = execute_stage(
            session,
            StageName.ASR,
            "legacy-asr",
            should_not_run,
            max_attempts=3,
        )

        assert calls == 0
        assert restored[0].text == "从旧检查点恢复"
    finally:
        store.close()


def test_demo_seed_distinguishes_real_audio_from_simulated_cases(
    database_url: str,
    tmp_path,
):
    from scripts.migrate_sqlite_to_postgres import migrate

    runs, batch = create_legacy_databases(tmp_path)
    demo_cases = tmp_path / "demo-cases.json"
    demo_cases.write_text(
        json.dumps(
            [
                {
                    "caseId": "CASE-DEMO",
                    "callId": "CALL-DEMO",
                    "customerDisplayName": "演示客户",
                    "sourceKind": "DEMO",
                    "callStartedAt": "2025-10-15T02:25:11Z",
                    "transcript": [
                        {"speaker": "客户", "text": "演示", "start": 0, "end": 1}
                    ],
                },
                {
                    "caseId": "CASE-REAL",
                    "callId": "CALL-REAL",
                    "customerDisplayName": "真实录音客户",
                    "sourceKind": "REAL_AUDIO",
                    "callStartedAt": "2025-10-15T02:25:11Z",
                    "transcript": [
                        {"speaker": "客户", "text": "真实录音", "start": 0, "end": 1}
                    ],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    migrate(
        runs,
        batch,
        database_url,
        LocalArtifactStore(tmp_path / "artifacts"),
        demo_cases_path=demo_cases,
    )
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            flags = connection.execute(
                text(
                    "SELECT case_id, source_kind, is_demo FROM cases "
                    "WHERE case_id IN ('CASE-DEMO', 'CASE-REAL') ORDER BY case_id"
                )
            ).all()
    finally:
        engine.dispose()

    assert flags == [
        ("CASE-DEMO", "DEMO", True),
        ("CASE-REAL", "REAL_AUDIO", False),
    ]


def test_import_failure_rolls_back_all_postgres_rows(database_url: str, tmp_path):
    from sqlalchemy.exc import IntegrityError

    from scripts.migrate_sqlite_to_postgres import migrate

    runs, batch = create_legacy_databases(tmp_path)
    with sqlite3.connect(runs) as connection:
        connection.execute("UPDATE agent_events SET phase='INVALID_PHASE'")

    with pytest.raises(IntegrityError):
        migrate(
            runs,
            batch,
            database_url,
            LocalArtifactStore(tmp_path / "artifacts"),
        )

    assert table_counts(database_url) == {name: 0 for name in table_counts(database_url)}
