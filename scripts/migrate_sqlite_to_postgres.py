from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from qc.artifact_store import ArtifactStore, LocalArtifactStore
from qc.orm_models import (
    AgentTraceEventRow,
    BatchExportRow,
    BatchItemRow,
    BatchJobRow,
    CallRow,
    CaseRow,
    QCReportRow,
    QCRunRow,
    StageExecutionRow,
)


TABLE_KEYS = (
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

_STAGE_ARTIFACT_NAMES = {
    "TRANSCODE": ("transcode.wav", "audio/wav"),
    "ASR": ("transcript.json", "application/json"),
    "EMOTION": ("emotion.json", "application/json"),
    "QC": ("analysis-result.json", "application/json"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.resolve(strict=True)
    connection = sqlite3.connect(
        f"file:{resolved.as_posix()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _rows(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    if table not in _tables(connection):
        return []
    return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]


def _json_value(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def _datetime(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_fallback(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


def _batch_status(files: list[dict[str, Any]]) -> str:
    if not files:
        return "CREATED"
    statuses = {str(item.get("status") or "PENDING") for item in files}
    if statuses <= {"DONE"}:
        return "COMPLETED"
    if statuses <= {"FAILED_FINAL", "DEAD_LETTER"}:
        return "FAILED"
    if statuses & {"DONE", "HUMAN_REVIEW", "FAILED_FINAL", "DEAD_LETTER"}:
        return "PARTIAL"
    return "RUNNING"


def _insert_rows(connection, model, rows: Iterable[dict], key: str, report: dict) -> None:
    primary_key = list(model.__table__.primary_key.columns)[0]
    for row in rows:
        statement = (
            pg_insert(model)
            .values(**row)
            .on_conflict_do_nothing()
            .returning(primary_key)
        )
        inserted = connection.execute(statement).scalar_one_or_none()
        if inserted is None:
            report["conflicts"].append(
                {"table": key, "id": str(row.get(primary_key.name))}
            )
        else:
            report["inserted"][key] += 1


def _project_source(project_root: Path, value: str) -> Path:
    candidate = (project_root / value).resolve(strict=True)
    try:
        candidate.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"demo source escapes project root: {value}") from exc
    if not candidate.is_file():
        raise ValueError(f"demo source is not a file: {value}")
    return candidate


def _legacy_artifact_source(project_root: Path, value: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else project_root / raw
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"legacy artifact escapes project root: {value}") from exc
    if not resolved.is_file():
        raise ValueError(f"legacy artifact is not a file: {value}")
    return resolved


def _plan_legacy_artifact(
    *,
    report: dict,
    artifact_store: ArtifactStore,
    project_root: Path,
    original_uri: str | None,
    target_uri: str,
    expected_sha256: str | None,
    mime_type: str,
    kind: str,
    source_id: int,
    require_valid_artifact: bool,
) -> str | None:
    if not original_uri:
        return None
    if not require_valid_artifact:
        return target_uri if Path(original_uri).is_absolute() else original_uri

    expected = expected_sha256.lower() if expected_sha256 else None
    if expected and artifact_store.verify_sha256(original_uri, expected):
        return original_uri
    if expected and artifact_store.verify_sha256(target_uri, expected):
        return target_uri

    try:
        source = _legacy_artifact_source(project_root, original_uri)
    except (FileNotFoundError, OSError, ValueError):
        source = None
    if source is not None and expected and _sha256(source) == expected:
        report["_artifactCopies"].append(
            {
                "source": source,
                "uri": target_uri,
                "sha256": expected,
                "mimeType": mime_type,
            }
        )
        return target_uri

    report["missingArtifacts"].append(
        {
            "kind": kind,
            "sourceId": source_id,
            "uri": original_uri,
            "targetUri": target_uri,
            "expectedSha256": expected_sha256,
        }
    )
    return target_uri


def _apply_demo_cases(
    plan: dict[str, list[dict]],
    report: dict,
    demo_cases_path: Path | None,
) -> None:
    if demo_cases_path is None:
        report["sourceCounts"]["demo_cases"] = 0
        report.setdefault("_artifactCopies", [])
        report["artifactCopiesPlanned"] = len(report["_artifactCopies"])
        return
    project_root = Path(__file__).resolve().parents[1]
    demo_cases_path = demo_cases_path.resolve(strict=True)
    try:
        demo_cases_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("demo cases file must be inside the project root") from exc
    entries = json.loads(demo_cases_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("demo cases JSON must contain an array")
    report["sourceCounts"]["demo_cases"] = len(entries)
    copies: list[dict[str, Any]] = report.setdefault("_artifactCopies", [])

    case_rows = {row["case_id"]: row for row in plan["cases"]}
    call_rows = {row["call_id"]: row for row in plan["calls"]}
    run_ids = {row["run_id"] for row in plan["qc_runs"]}
    for entry in entries:
        case_id = str(entry["caseId"])
        call_id = str(entry["callId"])
        started_at = _datetime(entry["callStartedAt"], _source_fallback(demo_cases_path))
        if entry.get("transcriptSource"):
            transcript_path = _project_source(project_root, entry["transcriptSource"])
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        else:
            transcript = list(entry.get("transcript") or [])
        if not isinstance(transcript, list) or not transcript:
            raise ValueError(f"demo call {call_id} has no transcript")
        normalized_transcript = []
        for index, item in enumerate(transcript, 1):
            turn = dict(item)
            turn["turnId"] = turn.get("turnId") or f"T{index:04d}"
            normalized_transcript.append(turn)

        audio_uri = entry.get("audioArtifactUri")
        audio_sha256 = None
        if entry.get("audioSource"):
            audio_source = _project_source(project_root, entry["audioSource"])
            audio_sha256 = _sha256(audio_source)
            copies.append(
                {
                    "source": audio_source,
                    "uri": audio_uri,
                    "sha256": audio_sha256,
                    "mimeType": entry.get("audioMimeType") or "application/octet-stream",
                }
            )
        case_rows[case_id] = {
            "case_id": case_id,
            "customer_display_name": entry["customerDisplayName"],
            "assigned_agent_display_name": entry.get("assignedAgentDisplayName"),
            "source_kind": entry["sourceKind"],
            "is_demo": bool(
                entry.get("isDemo", entry["sourceKind"] == "DEMO")
            ),
            "created_at": started_at,
            "updated_at": started_at,
        }
        call_rows[call_id] = {
            "call_id": call_id,
            "case_id": case_id,
            "call_started_at": started_at,
            "duration_ms": int(entry.get("durationMs") or 0),
            "audio_artifact_uri": audio_uri,
            "audio_sha256": audio_sha256,
            "audio_mime_type": entry.get("audioMimeType"),
            "transcript_json": normalized_transcript,
            "transcript_version": "demo-seed-v1",
            "asr_model": entry.get("asrModel"),
            "asr_model_version": entry.get("asrModelVersion"),
            "created_at": started_at,
            "updated_at": started_at,
        }
        initial = entry.get("initialReport")
        if initial:
            run_id = f"DEMO-RUN-{call_id}"
            if run_id in run_ids:
                raise ValueError(f"duplicate demo run id: {run_id}")
            request = {
                "caseId": case_id,
                "callId": call_id,
                "callStartedAt": started_at.isoformat(),
                "transcript": normalized_transcript,
            }
            report_json = {
                "callId": call_id,
                "score": int(initial["score"]),
                "events": list(initial.get("events") or []),
                "violations": list(initial.get("violations") or []),
                "knowledgeHits": list(initial.get("knowledgeHits") or []),
                "auditSnapshot": initial.get("auditSnapshot"),
                "businessFact": initial.get("businessFact")
                or {"status": "NOT_CHECKED", "note": "演示种子，不代表真实业务结论"},
                "disposition": initial["disposition"],
                "summary": dict(initial.get("summary") or {}),
                "emotionTimeline": dict(initial.get("emotionTimeline") or {}),
            }
            plan["qc_runs"].append(
                {
                    "run_id": run_id,
                    "case_id": case_id,
                    "call_id": call_id,
                    "status": "COMPLETED",
                    "request_snapshot": request,
                    "errors_json": [],
                    "loop_used": False,
                    "loop_reason": "demo seed",
                    "model": None,
                    "prompt_version": "demo-seed-v1",
                    "rule_version": "demo-seed-v1",
                    "knowledge_version": None,
                    "runtime_version": "demo-seed-v1",
                    "started_at": started_at,
                    "finished_at": started_at,
                }
            )
            plan["qc_reports"].append(
                {
                    "report_id": run_id,
                    "run_id": run_id,
                    "score": report_json["score"],
                    "disposition": report_json["disposition"],
                    "report_json": report_json,
                    "created_at": started_at,
                }
            )
            run_ids.add(run_id)
    plan["cases"] = list(case_rows.values())
    plan["calls"] = list(call_rows.values())
    report["_artifactCopies"] = copies
    report["artifactCopiesPlanned"] = len(copies)


def _build_plan(
    runs_path: Path,
    batch_path: Path,
    artifact_store: ArtifactStore,
    demo_cases_path: Path | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    fallback_run_time = _source_fallback(runs_path)
    fallback_batch_time = _source_fallback(batch_path)
    plan = {key: [] for key in TABLE_KEYS}
    report = {
        "sourceCounts": {},
        "planned": {},
        "inserted": {key: 0 for key in TABLE_KEYS},
        "conflicts": [],
        "missingArtifacts": [],
        "warnings": [],
        "_artifactCopies": [],
    }

    with _open_read_only(runs_path) as source:
        legacy_runs = _rows(source, "agent_runs")
        legacy_events = _rows(source, "agent_events")
    report["sourceCounts"]["agent_runs"] = len(legacy_runs)
    report["sourceCounts"]["agent_events"] = len(legacy_events)

    events_by_run: dict[str, list[dict]] = {}
    for event in legacy_events:
        events_by_run.setdefault(str(event["run_id"]), []).append(event)

    cases: dict[str, dict] = {}
    calls: dict[str, dict] = {}
    for row in legacy_runs:
        request = _json_value(row.get("request_json"), {})
        case_id = str(row["case_id"])
        call_id = str(row["call_id"])
        transcript = list(request.get("transcript") or [])
        started_at = _datetime(request.get("callStartedAt"), fallback_run_time)
        now = started_at
        cases.setdefault(
            case_id,
            {
                "case_id": case_id,
                "customer_display_name": "未知（SQLite 导入）",
                "assigned_agent_display_name": None,
                "source_kind": "IMPORTED",
                "is_demo": False,
                "created_at": now,
                "updated_at": now,
            },
        )
        calls.setdefault(
            call_id,
            {
                "call_id": call_id,
                "case_id": case_id,
                "call_started_at": started_at,
                "duration_ms": int(
                    max((float(item.get("end", 0)) for item in transcript), default=0)
                    * 1000
                ),
                "audio_artifact_uri": None,
                "audio_sha256": None,
                "audio_mime_type": None,
                "transcript_json": transcript,
                "transcript_version": "legacy-sqlite-request-v1",
                "asr_model": None,
                "asr_model_version": None,
                "created_at": now,
                "updated_at": now,
            },
        )
        status = str(row.get("status") or "FAILED")
        errors = _json_value(row.get("errors_json"), [])
        run_events = events_by_run.get(str(row["run_id"]), [])
        plan["qc_runs"].append(
            {
                "run_id": row["run_id"],
                "case_id": case_id,
                "call_id": call_id,
                "status": status,
                "request_snapshot": request,
                "errors_json": errors,
                "loop_used": bool(run_events),
                "loop_reason": "legacy import; run timestamps unavailable",
                "model": None,
                "prompt_version": None,
                "rule_version": None,
                "knowledge_version": None,
                "runtime_version": "legacy-unknown",
                "started_at": started_at,
                "finished_at": None if status == "RUNNING" else started_at,
            }
        )
        result = _json_value(row.get("result_json"), None)
        if result is not None:
            disposition = result.get("disposition")
            if disposition not in {
                "AUTO_PASS",
                "AUTO_VIOLATION",
                "HUMAN_REVIEW_REQUIRED",
            }:
                disposition = (
                    "HUMAN_REVIEW_REQUIRED" if status == "PARTIAL" else "AUTO_PASS"
                )
                report["warnings"].append(
                    f"run {row['run_id']} lacked a valid disposition; used {disposition}"
                )
            plan["qc_reports"].append(
                {
                    "report_id": row["run_id"],
                    "run_id": row["run_id"],
                    "score": int(result.get("score", 0)),
                    "disposition": disposition,
                    "report_json": result,
                    "created_at": started_at,
                }
            )

    plan["cases"] = list(cases.values())
    plan["calls"] = list(calls.values())
    for event in legacy_events:
        run = next(item for item in plan["qc_runs"] if item["run_id"] == event["run_id"])
        plan["agent_trace_events"].append(
            {
                "event_id": event["id"],
                "run_id": event["run_id"],
                "iteration": event["iteration"],
                "phase": event["phase"],
                "event_json": _json_value(event.get("event_json"), {}),
                "created_at": run["started_at"],
            }
        )

    with _open_read_only(batch_path) as source:
        legacy_batches = _rows(source, "batches")
        legacy_files = _rows(source, "batch_files")
        legacy_stages = _rows(source, "file_stages")
        legacy_exports = _rows(source, "batch_exports")
        stage_columns = _columns(source, "file_stages") if "file_stages" in _tables(source) else set()
    report["sourceCounts"].update(
        {
            "batches": len(legacy_batches),
            "batch_files": len(legacy_files),
            "file_stages": len(legacy_stages),
            "batch_exports": len(legacy_exports),
        }
    )

    files_by_batch: dict[str, list[dict]] = {}
    for item in legacy_files:
        files_by_batch.setdefault(str(item["batch_id"]), []).append(item)
    batch_times: dict[str, datetime] = {}
    for row in legacy_batches:
        created_at = _datetime(row.get("created_at"), fallback_batch_time)
        batch_times[str(row["batch_id"])] = created_at
        files = files_by_batch.get(str(row["batch_id"]), [])
        status = _batch_status(files)
        plan["batch_jobs"].append(
            {
                "batch_id": row["batch_id"],
                "source": row["source"],
                "status": status,
                "total": int(row.get("total") or len(files)),
                "created_at": created_at,
                "started_at": created_at,
                "finished_at": created_at if status in {"COMPLETED", "FAILED"} else None,
            }
        )
    for row in legacy_files:
        now = batch_times.get(str(row["batch_id"]), fallback_batch_time)
        call_id = row.get("call_id")
        if call_id and call_id not in calls:
            placeholder_case = f"LEGACY-BATCH-{row['batch_id']}"
            cases[placeholder_case] = {
                "case_id": placeholder_case,
                "customer_display_name": "未知（批量 SQLite 导入）",
                "assigned_agent_display_name": None,
                "source_kind": "IMPORTED",
                "is_demo": False,
                "created_at": now,
                "updated_at": now,
            }
            request = _json_value(row.get("request_json"), {})
            transcript = list(request.get("transcript") or [])
            calls[call_id] = {
                "call_id": call_id,
                "case_id": placeholder_case,
                "call_started_at": _datetime(request.get("callStartedAt"), now),
                "duration_ms": int(max((float(x.get("end", 0)) for x in transcript), default=0) * 1000),
                "audio_artifact_uri": None,
                "audio_sha256": None,
                "audio_mime_type": None,
                "transcript_json": transcript,
                "transcript_version": "legacy-batch-request-v1",
                "asr_model": None,
                "asr_model_version": None,
                "created_at": now,
                "updated_at": now,
            }
        plan["batch_items"].append(
            {
                "item_id": row["file_id"],
                "batch_id": row["batch_id"],
                "source_uri": row["source_uri"],
                "call_id": call_id,
                "idempotency_key": row["idempotency_key"],
                "status": row.get("status") or "PENDING",
                "failed_reason": row.get("failed_reason"),
                "request_snapshot": _json_value(row.get("request_json"), None),
                "result_snapshot": _json_value(row.get("result_json"), None),
                "qc_run_id": None,
                "created_at": now,
                "updated_at": now,
            }
        )
    plan["cases"] = list(cases.values())
    plan["calls"] = list(calls.values())

    project_root = Path(__file__).resolve().parents[1]
    batch_by_file_id = {
        int(item["file_id"]): str(item["batch_id"])
        for item in legacy_files
    }
    for row in legacy_stages:
        original_uri = row.get("artifact_uri") if "artifact_uri" in stage_columns else None
        expected = row.get("sha256") if "sha256" in stage_columns else None
        stage_name = str(row["stage"])
        artifact_name, mime_type = _STAGE_ARTIFACT_NAMES[stage_name]
        target_uri = (
            f"batch/{batch_by_file_id[int(row['file_id'])]}/"
            f"{row['file_id']}/{artifact_name}"
        )
        uri = _plan_legacy_artifact(
            report=report,
            artifact_store=artifact_store,
            project_root=project_root,
            original_uri=original_uri,
            target_uri=target_uri,
            expected_sha256=expected,
            mime_type=mime_type,
            kind="stage",
            source_id=int(row["id"]),
            require_valid_artifact=str(row["status"]) == "DONE",
        )
        plan["stage_executions"].append(
            {
                "stage_execution_id": row["id"],
                "item_id": row["file_id"],
                "stage": row["stage"],
                "status": row["status"],
                "started_at": _datetime(row.get("started_at"), fallback_batch_time) if row.get("started_at") else None,
                "finished_at": _datetime(row.get("finished_at"), fallback_batch_time) if row.get("finished_at") else None,
                "duration_ms": float(row.get("duration_ms") or 0),
                "attempts": int(row.get("attempts") or 0),
                "artifact_uri": uri,
                "sha256": expected,
                "producer_version": row.get("producer_version") if "producer_version" in stage_columns else None,
                "error_code": row.get("error_code") if "error_code" in stage_columns else None,
                "retryable": bool(row.get("retryable")) if "retryable" in stage_columns and row.get("retryable") is not None else None,
                "error_summary": row.get("error"),
            }
        )
    for row in legacy_exports:
        original_uri = row["artifact_uri"]
        suffix = "csv" if str(row["format"]).lower() == "csv" else "json"
        target_uri = f"exports/{row['batch_id']}/{row['export_id']}.{suffix}"
        uri = _plan_legacy_artifact(
            report=report,
            artifact_store=artifact_store,
            project_root=project_root,
            original_uri=original_uri,
            target_uri=target_uri,
            expected_sha256=row.get("sha256"),
            mime_type=(
                "text/csv; charset=utf-8"
                if suffix == "csv"
                else "application/json"
            ),
            kind="export",
            source_id=int(row["export_id"]),
            require_valid_artifact=str(row["status"]) == "DONE",
        )
        now = batch_times.get(str(row["batch_id"]), fallback_batch_time)
        plan["batch_exports"].append(
            {
                "export_id": row["export_id"],
                "batch_id": row["batch_id"],
                "format": row["format"],
                "artifact_uri": uri,
                "status": row["status"],
                "sha256": row.get("sha256"),
                "producer_version": row.get("producer_version") or "legacy-unknown",
                "error_code": row.get("error_code"),
                "created_at": now,
                "finished_at": now if row["status"] in {"DONE", "FAILED"} else None,
            }
        )

    report["warnings"].append(
        "legacy SQLite has no run timestamps; qc_runs timestamps use callStartedAt as a deterministic fallback"
    )
    _apply_demo_cases(plan, report, demo_cases_path)
    report["planned"] = {key: len(value) for key, value in plan.items()}
    return plan, report


def migrate(
    runs_path: str | Path,
    batch_path: str | Path,
    database_url: str,
    artifact_store: ArtifactStore,
    *,
    dry_run: bool = False,
    demo_cases_path: str | Path | None = None,
) -> dict:
    runs_path = Path(runs_path).resolve(strict=True)
    batch_path = Path(batch_path).resolve(strict=True)
    before = {str(runs_path): _sha256(runs_path), str(batch_path): _sha256(batch_path)}
    demo_path = Path(demo_cases_path) if demo_cases_path is not None else None
    plan, report = _build_plan(runs_path, batch_path, artifact_store, demo_path)
    artifact_copies = report.pop("_artifactCopies")
    report["dryRun"] = dry_run
    report["sourceSha256Before"] = before

    if not dry_run:
        for copy in artifact_copies:
            reference = artifact_store.put_bytes(
                copy["uri"],
                copy["source"].read_bytes(),
                mime_type=copy["mimeType"],
            )
            if reference.sha256 != copy["sha256"]:
                raise RuntimeError(f"artifact copy hash mismatch: {copy['uri']}")
        engine = create_engine(database_url, pool_pre_ping=True)
        try:
            with engine.begin() as connection:
                for model, key in (
                    (CaseRow, "cases"),
                    (CallRow, "calls"),
                    (QCRunRow, "qc_runs"),
                    (QCReportRow, "qc_reports"),
                    (AgentTraceEventRow, "agent_trace_events"),
                    (BatchJobRow, "batch_jobs"),
                    (BatchItemRow, "batch_items"),
                    (StageExecutionRow, "stage_executions"),
                    (BatchExportRow, "batch_exports"),
                ):
                    _insert_rows(connection, model, plan[key], key, report)
                for table, column in (
                    ("agent_trace_events", "event_id"),
                    ("batch_items", "item_id"),
                    ("stage_executions", "stage_execution_id"),
                    ("batch_exports", "export_id"),
                ):
                    connection.execute(
                        text(
                            f"SELECT setval(pg_get_serial_sequence('{table}', '{column}'), "
                            f"GREATEST(COALESCE((SELECT MAX({column}) FROM {table}), 1), 1), "
                            f"(SELECT MAX({column}) IS NOT NULL FROM {table}))"
                        )
                    )
        finally:
            engine.dispose()

    after = {str(runs_path): _sha256(runs_path), str(batch_path): _sha256(batch_path)}
    report["sourceSha256After"] = after
    if before != after:
        raise RuntimeError("a source SQLite database changed during import")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Import legacy SQLite data into PostgreSQL")
    parser.add_argument("--runs-db", default="data/qc_runs.db")
    parser.add_argument("--batch-db", default="data/batch.db")
    parser.add_argument("--demo-cases", default="data/demo_cases.json")
    parser.add_argument("--database-url")
    parser.add_argument("--database-url-env", default="DATABASE_URL")
    parser.add_argument("--artifact-root", default="data/artifacts")
    parser.add_argument("--artifact-mode", choices=["copy"], default="copy")
    parser.add_argument("--report")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    database_url = arguments.database_url or os.getenv(arguments.database_url_env)
    if not database_url:
        parser.error(
            f"provide --database-url or set {arguments.database_url_env}"
        )
    report = migrate(
        arguments.runs_db,
        arguments.batch_db,
        database_url,
        LocalArtifactStore(arguments.artifact_root),
        dry_run=arguments.dry_run,
        demo_cases_path=arguments.demo_cases,
    )
    if arguments.report:
        report_path = Path(arguments.report).resolve()
        project_root = Path(__file__).resolve().parents[1]
        try:
            report_path.relative_to(project_root)
        except ValueError:
            parser.error("--report must be inside the project root")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
