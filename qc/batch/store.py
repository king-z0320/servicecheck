from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from qc.batch.models import (
    PIPELINE_STAGES,
    BatchFileStatus,
    BatchMeta,
    FileRecord,
    StageName,
    StageRecord,
    VALID_FILE_TRANSITIONS,
)


class StateConflictError(RuntimeError):
    """The persisted file or stage state no longer matches the expected state."""


class BatchStore:
    """SQLite 三级任务状态：batches / batch_files / file_stages。

    幂等：batch_files.idempotency_key 唯一，重复插入返回 False。
    跨进程可读：每次操作独立连接，重启后状态可恢复。
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    total INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_files (
                    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    call_id TEXT,
                    idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    failed_reason TEXT,
                    request_json TEXT,
                    result_json TEXT,
                    FOREIGN KEY(batch_id) REFERENCES batches(batch_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_files_idem
                    ON batch_files(batch_id, idempotency_key);
                CREATE TABLE IF NOT EXISTS file_stages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    duration_ms REAL NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    artifact_uri TEXT,
                    sha256 TEXT,
                    producer_version TEXT,
                    error_code TEXT,
                    retryable INTEGER,
                    error TEXT,
                    FOREIGN KEY(file_id) REFERENCES batch_files(file_id)
                );
                CREATE TABLE IF NOT EXISTS batch_exports (
                    export_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    format TEXT NOT NULL,
                    artifact_uri TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sha256 TEXT,
                    producer_version TEXT,
                    error_code TEXT,
                    UNIQUE(batch_id, format, artifact_uri)
                );
                """
            )
            # 老 DB 的 batches 表可能没有 started_at 列；缺列则补加，已存在则跳过。
            cols = {
                row["name"]
                for row in db.execute("PRAGMA table_info(batches)").fetchall()
            }
            if "started_at" not in cols:
                db.execute("ALTER TABLE batches ADD COLUMN started_at TEXT")

            stage_cols = {
                row["name"]
                for row in db.execute("PRAGMA table_info(file_stages)").fetchall()
            }
            for name in ("artifact_uri", "sha256", "producer_version", "error_code"):
                if name not in stage_cols:
                    db.execute(f"ALTER TABLE file_stages ADD COLUMN {name} TEXT")
            if "retryable" not in stage_cols:
                db.execute("ALTER TABLE file_stages ADD COLUMN retryable INTEGER")

    def create_batch(self, meta: BatchMeta) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO batches(batch_id, source, created_at, started_at, total) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    meta.batch_id,
                    meta.source,
                    meta.created_at.isoformat(),
                    meta.created_at.isoformat(),
                    meta.total,
                ),
            )

    def create_batch_if_absent(self, meta: BatchMeta) -> bool:
        """幂等建批次；已存在则跳过。返回是否实际新建。"""
        try:
            self.create_batch(meta)
            return True
        except sqlite3.IntegrityError:
            return False

    def add_file(self, batch_id: str, record: FileRecord) -> bool:
        """幂等添加文件；重复幂等键返回 False，不抛错。"""
        with self._connect() as db:
            cursor = db.execute(
                """
                INSERT OR IGNORE INTO batch_files(
                    batch_id, source_uri, call_id, idempotency_key, status
                ) VALUES (?, ?, ?, ?, 'PENDING')
                """,
                (
                    batch_id,
                    record.source_uri,
                    record.callId,
                    record.idempotency_key,
                ),
            )
            return cursor.rowcount == 1

    def list_files(
        self, batch_id: str, statuses: list[str] | None = None
    ) -> list[dict]:
        with self._connect() as db:
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                rows = db.execute(
                    f"SELECT * FROM batch_files WHERE batch_id = ? AND status IN ({placeholders})",
                    (batch_id, *statuses),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM batch_files WHERE batch_id = ?",
                    (batch_id,),
                ).fetchall()
        return [self._file_row_with_stages(row) for row in rows]

    def get_file(self, file_id: int) -> dict:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM batch_files WHERE file_id = ?",
                (file_id,),
            ).fetchone()
            if row is None:
                raise KeyError(file_id)
        return self._file_row_with_stages(row)

    def _file_row_with_stages(self, row: sqlite3.Row) -> dict:
        with self._connect() as db:
            stage_rows = db.execute(
                "SELECT * FROM file_stages WHERE file_id = ? ORDER BY id",
                (row["file_id"],),
            ).fetchall()
        return {
            "file_id": row["file_id"],
            "batch_id": row["batch_id"],
            "source_uri": row["source_uri"],
            "call_id": row["call_id"],
            "idempotency_key": row["idempotency_key"],
            "status": row["status"],
            "failed_reason": row["failed_reason"],
            "request_json": row["request_json"],
            "result_json": row["result_json"],
            "stages": [self._stage_row(s) for s in stage_rows],
        }

    @staticmethod
    def _stage_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        if result.get("status") in {"RUNNING", "DONE", "FAILED"} and not result.get(
            "attempts"
        ):
            result["attempts"] = 1
        if result.get("retryable") is not None:
            result["retryable"] = bool(result["retryable"])
        return result

    def record_stage(self, file_id: int, stage: StageRecord) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO file_stages(
                    file_id, stage, status, started_at, finished_at,
                    duration_ms, attempts, artifact_uri, sha256,
                    producer_version, error_code, retryable, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    stage.stage.value,
                    stage.status,
                    stage.started_at.isoformat() if stage.started_at else None,
                    stage.finished_at.isoformat() if stage.finished_at else None,
                    stage.duration_ms,
                    stage.attempts,
                    stage.artifact_uri,
                    stage.sha256,
                    stage.producer_version,
                    stage.error_code,
                    None if stage.retryable is None else int(stage.retryable),
                    stage.error,
                ),
            )

    def get_stage_checkpoint(
        self, file_id: int, stage: StageName | str
    ) -> dict | None:
        stage_value = stage.value if isinstance(stage, StageName) else str(stage)
        with self._connect() as db:
            row = db.execute(
                """
                SELECT * FROM file_stages
                WHERE file_id = ? AND stage = ?
                ORDER BY id DESC LIMIT 1
                """,
                (file_id, stage_value),
            ).fetchone()
        return self._stage_row(row) if row is not None else None

    def begin_stage(self, file_id: int, stage: StageName) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            current = db.execute(
                """
                SELECT * FROM file_stages
                WHERE file_id = ? AND stage = ?
                ORDER BY id DESC LIMIT 1
                """,
                (file_id, stage.value),
            ).fetchone()
            if current is None:
                attempts = 1
                db.execute(
                    """
                    INSERT INTO file_stages(
                        file_id, stage, status, started_at, duration_ms, attempts
                    ) VALUES (?, ?, 'RUNNING', ?, 0, ?)
                    """,
                    (file_id, stage.value, now, attempts),
                )
            else:
                previous_attempts = int(current["attempts"] or 0)
                if previous_attempts == 0 and current["status"] in {
                    "RUNNING",
                    "DONE",
                    "FAILED",
                }:
                    previous_attempts = 1
                attempts = previous_attempts + 1
                db.execute(
                    """
                    UPDATE file_stages
                    SET status = 'RUNNING', started_at = ?, finished_at = NULL,
                        duration_ms = 0, attempts = ?, artifact_uri = NULL,
                        sha256 = NULL, producer_version = NULL,
                        error_code = NULL, retryable = NULL, error = NULL
                    WHERE id = ?
                    """,
                    (now, attempts, current["id"]),
                )
        return attempts

    def complete_stage(
        self,
        file_id: int,
        stage: StageName,
        *,
        artifact_uri: str | Path,
        sha256: str,
        producer_version: str,
        duration_ms: float,
    ) -> None:
        current = self.get_stage_checkpoint(file_id, stage)
        if current is None:
            raise StateConflictError(f"stage {stage.value} has not started")
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE file_stages
                SET status = 'DONE', finished_at = ?, duration_ms = ?,
                    artifact_uri = ?, sha256 = ?, producer_version = ?,
                    error_code = NULL, retryable = NULL, error = NULL
                WHERE id = ? AND status = 'RUNNING'
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    duration_ms,
                    str(artifact_uri),
                    sha256,
                    producer_version,
                    current["id"],
                ),
            )
            if cursor.rowcount != 1:
                raise StateConflictError(f"stage {stage.value} is not running")

    def fail_stage(
        self,
        file_id: int,
        stage: StageName,
        *,
        error_code: str,
        retryable: bool,
        error: str,
        duration_ms: float,
    ) -> None:
        current = self.get_stage_checkpoint(file_id, stage)
        if current is None:
            raise StateConflictError(f"stage {stage.value} has not started")
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE file_stages
                SET status = 'FAILED', finished_at = ?, duration_ms = ?,
                    error_code = ?, retryable = ?, error = ?
                WHERE id = ? AND status = 'RUNNING'
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    duration_ms,
                    error_code,
                    int(retryable),
                    error,
                    current["id"],
                ),
            )
            if cursor.rowcount != 1:
                raise StateConflictError(f"stage {stage.value} is not running")

    def invalidate_stages(self, file_id: int, stages: list[StageName]) -> None:
        with self._connect() as db:
            for stage in stages:
                row = db.execute(
                    """
                    SELECT id FROM file_stages
                    WHERE file_id = ? AND stage = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (file_id, stage.value),
                ).fetchone()
                if row is not None:
                    db.execute(
                        """
                        UPDATE file_stages
                        SET status = 'PENDING', artifact_uri = NULL, sha256 = NULL,
                            producer_version = NULL, finished_at = NULL,
                            error_code = NULL, retryable = NULL, error = NULL
                        WHERE id = ?
                        """,
                        (row["id"],),
                    )

    @staticmethod
    def _status_value(status: BatchFileStatus | str) -> BatchFileStatus:
        try:
            return status if isinstance(status, BatchFileStatus) else BatchFileStatus(str(status))
        except ValueError as exc:
            raise ValueError(f"unknown batch file status: {status!r}") from exc

    def claim_file(
        self,
        file_id: int,
        expected_status: BatchFileStatus | str,
    ) -> bool:
        expected = self._status_value(expected_status)
        if expected not in {BatchFileStatus.PENDING, BatchFileStatus.INTERRUPTED}:
            return False
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE batch_files
                SET status = 'RUNNING', failed_reason = NULL
                WHERE file_id = ? AND status = ?
                """,
                (file_id, expected.value),
            )
            return cursor.rowcount == 1

    def set_file_status(
        self,
        file_id: int,
        status: BatchFileStatus | str,
        failed_reason: str | None = None,
    ) -> None:
        target = self._status_value(status)
        if target in {
            BatchFileStatus.DONE,
            BatchFileStatus.HUMAN_REVIEW,
            BatchFileStatus.FAILED_FINAL,
            BatchFileStatus.DEAD_LETTER,
        }:
            raise StateConflictError("terminal states must be written with finalize_file")
        with self._connect() as db:
            row = db.execute(
                "SELECT status FROM batch_files WHERE file_id = ?", (file_id,)
            ).fetchone()
            if row is None:
                raise KeyError(file_id)
            current = BatchFileStatus(row["status"])
            if target not in VALID_FILE_TRANSITIONS[current]:
                raise StateConflictError(
                    f"illegal file transition: {current.value} -> {target.value}"
                )
            cursor = db.execute(
                """
                UPDATE batch_files SET status = ?, failed_reason = ?
                WHERE file_id = ? AND status = ?
                """,
                (target.value, failed_reason, file_id, current.value),
            )
            if cursor.rowcount != 1:
                raise StateConflictError(f"file {file_id} state changed concurrently")

    def finalize_file(
        self,
        file_id: int,
        status: BatchFileStatus | str,
        request_json: str,
        result_json: str,
        failed_reason: str | None = None,
    ) -> None:
        target = self._status_value(status)
        if target not in {
            BatchFileStatus.DONE,
            BatchFileStatus.HUMAN_REVIEW,
            BatchFileStatus.FAILED_FINAL,
        }:
            raise ValueError(f"not a writable final status: {target.value}")
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE batch_files
                SET status = ?, request_json = ?, result_json = ?, failed_reason = ?
                WHERE file_id = ? AND status = 'RUNNING'
                """,
                (
                    target.value,
                    request_json,
                    result_json,
                    failed_reason,
                    file_id,
                ),
            )
            if cursor.rowcount != 1:
                raise StateConflictError(f"file {file_id} is not RUNNING")

    def batch_summary(self, batch_id: str) -> dict:
        with self._connect() as db:
            total = db.execute(
                "SELECT COUNT(*) AS n FROM batch_files WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()["n"]
            by_status = db.execute(
                "SELECT status, COUNT(*) AS n FROM batch_files WHERE batch_id = ? GROUP BY status",
                (batch_id,),
            ).fetchall()
        return {
            "batch_id": batch_id,
            "total": total,
            "by_status": {row["status"]: row["n"] for row in by_status},
        }

    def batch_started_at(self, batch_id: str) -> str | None:
        """批次启动时间（ISO 字符串），用于吞吐计算。缺失或老库返回 None。"""
        with self._connect() as db:
            row = db.execute(
                "SELECT started_at AS s FROM batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        return row["s"] if row else None

    def batch_durations(self, batch_id: str) -> dict[str, float]:
        """各阶段 DONE 记录的平均耗时（毫秒）。仅统计 status='DONE'。"""
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT fs.stage AS stage, AVG(fs.duration_ms) AS avg_ms
                FROM file_stages fs
                JOIN batch_files bf ON bf.file_id = fs.file_id
                WHERE bf.batch_id = ? AND fs.status = 'DONE'
                GROUP BY fs.stage
                """,
                (batch_id,),
            ).fetchall()
        return {row["stage"]: float(row["avg_ms"]) for row in rows if row["avg_ms"] is not None}

    def resume_candidates(self, batch_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT * FROM batch_files
                WHERE batch_id = ? AND status IN ('PENDING', 'INTERRUPTED')
                ORDER BY file_id
                """,
                (batch_id,),
            ).fetchall()
        return [self._file_row_with_stages(row) for row in rows]

    def mark_interrupted_running(self, batch_id: str) -> int:
        """把目标批次遗留的 RUNNING 标记为 INTERRUPTED。"""
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE batch_files SET status = 'INTERRUPTED'
                WHERE batch_id = ? AND status = 'RUNNING'
                """,
                (batch_id,),
            )
            return cursor.rowcount

    def last_completed_stage(self, file_id: int) -> StageName | None:
        with self._connect() as db:
            rows = db.execute(
                "SELECT stage FROM file_stages WHERE file_id = ? AND status = 'DONE' ORDER BY id",
                (file_id,),
            ).fetchall()
        completed_names = {StageName(row["stage"]) for row in rows}
        # 返回 PIPELINE_STAGES 中最后完成的阶段
        for stage in reversed(PIPELINE_STAGES):
            if stage in completed_names:
                return stage
        return None

    @staticmethod
    def _artifact_uri(value: str | Path) -> str:
        return str(value)

    def begin_export(
        self,
        batch_id: str,
        format: str,
        artifact_uri: str | Path,
        producer_version: str,
    ) -> None:
        uri = self._artifact_uri(artifact_uri)
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO batch_exports(
                    batch_id, format, artifact_uri, status, sha256,
                    producer_version, error_code
                ) VALUES (?, ?, ?, 'RUNNING', NULL, ?, NULL)
                ON CONFLICT(batch_id, format, artifact_uri) DO UPDATE SET
                    status = 'RUNNING', sha256 = NULL,
                    producer_version = excluded.producer_version,
                    error_code = NULL
                """,
                (batch_id, format, uri, producer_version),
            )

    def complete_export(
        self,
        batch_id: str,
        format: str,
        artifact_uri: str | Path,
        producer_version: str,
        sha256: str,
    ) -> None:
        uri = self._artifact_uri(artifact_uri)
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE batch_exports
                SET status = 'DONE', sha256 = ?, producer_version = ?,
                    error_code = NULL
                WHERE batch_id = ? AND format = ? AND artifact_uri = ?
                  AND status = 'RUNNING'
                """,
                (sha256, producer_version, batch_id, format, uri),
            )
            if cursor.rowcount != 1:
                raise StateConflictError("export record is not RUNNING")

    def fail_export(
        self,
        batch_id: str,
        format: str,
        artifact_uri: str | Path,
        producer_version: str,
        error_code: str,
    ) -> None:
        uri = self._artifact_uri(artifact_uri)
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO batch_exports(
                    batch_id, format, artifact_uri, status, sha256,
                    producer_version, error_code
                ) VALUES (?, ?, ?, 'FAILED', NULL, ?, ?)
                ON CONFLICT(batch_id, format, artifact_uri) DO UPDATE SET
                    status = 'FAILED', sha256 = NULL,
                    producer_version = excluded.producer_version,
                    error_code = excluded.error_code
                """,
                (batch_id, format, uri, producer_version, error_code),
            )

    def get_export_record(
        self,
        batch_id: str,
        format: str,
        artifact_uri: str | Path,
    ) -> dict | None:
        uri = self._artifact_uri(artifact_uri)
        with self._connect() as db:
            row = db.execute(
                """
                SELECT * FROM batch_exports
                WHERE batch_id = ? AND format = ? AND artifact_uri = ?
                """,
                (batch_id, format, uri),
            ).fetchone()
        return dict(row) if row is not None else None
