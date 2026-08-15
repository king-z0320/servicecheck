from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from time import sleep
from typing import Callable, Literal

from qc.errors import AnalysisError, ErrorStage, PipelineFailure
from qc.models import AgentTraceEvent, AnalysisRequest, QualityReport


TerminalStatus = Literal["COMPLETED", "PARTIAL", "FAILED"]


class RunStore:
    def __init__(
        self,
        path: str | Path,
        *,
        sleeper: Callable[[float], None] = sleep,
        retry_delay_seconds: float = 0.05,
    ):
        self.path = str(path)
        self.sleeper = sleeper
        self.retry_delay_seconds = retry_delay_seconds
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    errors_json TEXT
                );
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id)
                );
                """
            )
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(agent_runs)")
            }
            if "errors_json" not in columns:
                db.execute("ALTER TABLE agent_runs ADD COLUMN errors_json TEXT")

    def _write(self, operation):
        for attempt in (1, 2, 3):
            try:
                with self._connect() as db:
                    return operation(db)
            except sqlite3.OperationalError as exc:
                detail = str(exc).lower()
                locked = "locked" in detail or "busy" in detail
                if locked and attempt < 3:
                    self.sleeper(self.retry_delay_seconds)
                    continue
                code = "SQLITE_LOCKED" if locked else "PERSISTENCE_WRITE_FAILED"
                message = (
                    "结果存储暂时被占用"
                    if locked
                    else "结果存储写入失败"
                )
                raise PipelineFailure(
                    AnalysisError(
                        code=code,
                        stage=ErrorStage.PERSISTENCE,
                        message=message,
                        retryable=locked,
                        attempts=attempt,
                    )
                ) from exc
            except sqlite3.IntegrityError:
                raise
            except sqlite3.DatabaseError as exc:
                raise PipelineFailure(
                    AnalysisError(
                        code="PERSISTENCE_WRITE_FAILED",
                        stage=ErrorStage.PERSISTENCE,
                        message="结果存储写入失败",
                        retryable=False,
                        attempts=attempt,
                    )
                ) from exc
        raise AssertionError("bounded SQLite retry loop did not terminate")

    def create_run(self, run_id: str, request: AnalysisRequest):
        def operation(db):
            db.execute(
                """
                INSERT INTO agent_runs(
                    run_id, case_id, call_id, status, request_json, errors_json
                ) VALUES (?, ?, ?, 'RUNNING', ?, '[]')
                """,
                (
                    run_id,
                    request.caseId,
                    request.callId,
                    request.model_dump_json(),
                ),
            )

        try:
            self._write(operation)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"run already exists: {run_id}") from exc

    def append_event(self, run_id: str, event: AgentTraceEvent):
        def operation(db):
            db.execute(
                """
                INSERT INTO agent_events(run_id, iteration, phase, event_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    event.iteration,
                    event.phase,
                    event.model_dump_json(),
                ),
            )

        self._write(operation)

    def finish_run(
        self,
        run_id: str,
        status: TerminalStatus,
        report: QualityReport | None,
        errors: list[AnalysisError],
    ) -> None:
        if status not in {"COMPLETED", "PARTIAL", "FAILED"}:
            raise ValueError(f"invalid terminal status: {status}")
        result_json = report.model_dump_json() if report is not None else None
        errors_json = json.dumps(
            [error.model_dump(mode="json") for error in errors],
            ensure_ascii=False,
        )

        def operation(db):
            cursor = db.execute(
                """
                UPDATE agent_runs
                SET status = ?, result_json = ?, errors_json = ?
                WHERE run_id = ? AND status = 'RUNNING'
                """,
                (status, result_json, errors_json, run_id),
            )
            if cursor.rowcount == 1:
                return
            existing = db.execute(
                "SELECT status FROM agent_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(run_id)
            raise ValueError(
                f"run is already terminal: {run_id} ({existing['status']})"
            )

        self._write(operation)

    def save_result(self, run_id: str, status: str, report: QualityReport):
        self.finish_run(run_id, status, report, [])

    def fail_incomplete_runs(self, error: AnalysisError) -> int:
        errors_json = json.dumps(
            [error.model_dump(mode="json")],
            ensure_ascii=False,
        )

        def operation(db):
            cursor = db.execute(
                """
                UPDATE agent_runs
                SET status = 'FAILED', result_json = NULL, errors_json = ?
                WHERE status = 'RUNNING'
                """,
                (errors_json,),
            )
            return cursor.rowcount

        return int(self._write(operation))

    def get_run(self, run_id: str) -> dict:
        with self._connect() as db:
            run = db.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(run_id)
            events = db.execute(
                """
                SELECT event_json FROM agent_events
                WHERE run_id = ? ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        return {
            "runId": run["run_id"],
            "caseId": run["case_id"],
            "callId": run["call_id"],
            "status": run["status"],
            "request": json.loads(run["request_json"]),
            "result": (
                json.loads(run["result_json"])
                if run["result_json"]
                else None
            ),
            "errors": (
                json.loads(run["errors_json"])
                if run["errors_json"]
                else []
            ),
            "events": [json.loads(row["event_json"]) for row in events],
        }

    def list_incomplete(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT run_id, case_id, call_id, status
                FROM agent_runs WHERE status = 'RUNNING'
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def find_runs(
        self,
        call_id: str | None = None,
        rule_id: str | None = None,
        event_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT run_id, case_id, call_id, status, result_json
                FROM agent_runs
                WHERE result_json IS NOT NULL
                ORDER BY rowid DESC
                LIMIT 500
                """
            ).fetchall()

        matches: list[dict] = []
        wanted_events = set(event_types or [])
        for row in rows:
            if call_id and row["call_id"] != call_id:
                continue
            try:
                report = json.loads(row["result_json"]) if row["result_json"] else {}
            except json.JSONDecodeError:
                continue
            if isinstance(report, dict) and "report" in report and "score" not in report:
                report = report.get("report") or {}
            events = report.get("events") or []
            violations = report.get("violations") or []
            event_type_values = [
                item.get("type") for item in events if isinstance(item, dict)
            ]
            rule_ids = [
                item.get("ruleId") for item in violations if isinstance(item, dict)
            ]
            if rule_id and rule_id not in rule_ids:
                continue
            if wanted_events and not wanted_events.intersection(event_type_values):
                continue
            matches.append(
                {
                    "runId": row["run_id"],
                    "caseId": row["case_id"],
                    "callId": row["call_id"],
                    "status": row["status"],
                    "score": report.get("score"),
                    "disposition": report.get("disposition"),
                    "eventTypes": event_type_values,
                    "ruleIds": rule_ids,
                }
            )
            if len(matches) >= limit:
                break
        return matches
