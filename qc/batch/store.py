from __future__ import annotations

import sqlite3
from pathlib import Path

from qc.batch.models import (
    PIPELINE_STAGES,
    BatchFileStatus,
    BatchMeta,
    FileRecord,
    StageName,
    StageRecord,
)


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
                    error TEXT,
                    FOREIGN KEY(file_id) REFERENCES batch_files(file_id)
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
            "stages": [dict(s) for s in stage_rows],
        }

    def record_stage(self, file_id: int, stage: StageRecord) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO file_stages(
                    file_id, stage, status, started_at, finished_at,
                    duration_ms, attempts, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    stage.stage.value,
                    stage.status,
                    stage.started_at.isoformat() if stage.started_at else None,
                    stage.finished_at.isoformat() if stage.finished_at else None,
                    stage.duration_ms,
                    stage.attempts,
                    stage.error,
                ),
            )

    _VALID_FILE_STATUSES = frozenset(s.value for s in BatchFileStatus)

    def set_file_status(
        self,
        file_id: int,
        status: BatchFileStatus | str,
        failed_reason: str | None = None,
    ) -> None:
        # 接受 BatchFileStatus 枚举或字符串字面量（后者便于 e2e/脚本场景）。
        # 对字符串做合法性校验，防止拼写错误静默写入脏状态。
        status_value = status.value if isinstance(status, BatchFileStatus) else str(status)
        if status_value not in self._VALID_FILE_STATUSES:
            raise ValueError(f"unknown batch file status: {status_value!r}")
        with self._connect() as db:
            db.execute(
                "UPDATE batch_files SET status = ?, failed_reason = ? WHERE file_id = ?",
                (status_value, failed_reason, file_id),
            )

    def save_file_report(
        self,
        file_id: int,
        status: BatchFileStatus,
        request_json: str,
        result_json: str,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE batch_files SET status = ?, request_json = ?, result_json = ? WHERE file_id = ?",
                (status.value, request_json, result_json, file_id),
            )

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
                WHERE batch_id = ? AND status IN ('PENDING', 'RUNNING', 'INTERRUPTED')
                ORDER BY file_id
                """,
                (batch_id,),
            ).fetchall()
        return [self._file_row_with_stages(row) for row in rows]

    def mark_interrupted_running(self) -> int:
        """重启时清理脏状态：所有 RUNNING 标 INTERRUPTED。返回受影响行数。"""
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE batch_files SET status = 'INTERRUPTED' WHERE status = 'RUNNING'"
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

    def increment_stage_attempts(self, file_id: int, stage: StageName) -> int:
        """对某阶段追加一次失败尝试，返回该阶段累计 attempts。"""
        with self._connect() as db:
            row = db.execute(
                "SELECT MAX(attempts) AS m FROM file_stages WHERE file_id = ? AND stage = ?",
                (file_id, stage.value),
            ).fetchone()
        current = (row["m"] or 0)
        next_attempts = current + 1
        self.record_stage(
            file_id,
            StageRecord(stage=stage, status="FAILED", attempts=next_attempts),
        )
        return next_attempts
