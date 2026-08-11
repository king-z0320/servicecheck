import json
import sqlite3
from pathlib import Path

from qc.models import AgentTraceEvent, AnalysisRequest, QualityReport


class RunStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
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
                    result_json TEXT
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

    def create_run(
        self,
        run_id: str,
        request: AnalysisRequest,
    ):
        try:
            with self._connect() as db:
                db.execute(
                    """
                    INSERT INTO agent_runs(
                        run_id, case_id, call_id, status, request_json
                    ) VALUES (?, ?, ?, 'RUNNING', ?)
                    """,
                    (
                        run_id,
                        request.caseId,
                        request.callId,
                        request.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"run already exists: {run_id}") from exc

    def append_event(
        self,
        run_id: str,
        event: AgentTraceEvent,
    ):
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO agent_events(
                    run_id, iteration, phase, event_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    event.iteration,
                    event.phase,
                    event.model_dump_json(),
                ),
            )

    def save_result(
        self,
        run_id: str,
        status: str,
        report: QualityReport,
    ):
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE agent_runs
                SET status = ?, result_json = ?
                WHERE run_id = ?
                """,
                (status, report.model_dump_json(), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(run_id)

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
            "events": [
                json.loads(row["event_json"])
                for row in events
            ],
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
        """情节记忆查询：按 callId / 规则 / 事件类型过滤已落盘结果。

        结果 JSON 为 QualityReport 结构（见 save_result）。
        """
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
            # 兼容误存整包 AnalysisResult 的情况
            if isinstance(report, dict) and "report" in report and "score" not in report:
                report = report.get("report") or {}
            events = report.get("events") or []
            violations = report.get("violations") or []
            event_type_values = [
                e.get("type") for e in events if isinstance(e, dict)
            ]
            rule_ids = [
                v.get("ruleId") for v in violations if isinstance(v, dict)
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
