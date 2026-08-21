from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import Select, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from qc.database import create_database_engine, create_session_factory
from qc.errors import AnalysisError
from qc.models import AgentTraceEvent, AnalysisRequest, QualityReport
from qc.orm_models import (
    AgentTraceEventRow,
    CallRow,
    CaseRow,
    QCReportRow,
    QCRunRow,
    LLMUsageRecordRow,
)
from qc.review_service import compute_route_reasons, needs_review_task
from qc.review_store import ensure_review_task_in_session


TerminalStatus = Literal["COMPLETED", "PARTIAL", "FAILED"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PostgresRunStore:
    def __init__(
        self,
        database_url: str,
        *,
        model: str | None = None,
        prompt_version: str | None = None,
        rule_version: str | None = None,
        knowledge_version: str | None = None,
        runtime_version: str = "unknown",
    ):
        self.engine = create_database_engine(database_url)
        self.session_factory = create_session_factory(self.engine)
        self.model = model
        self.prompt_version = prompt_version
        self.rule_version = rule_version
        self.knowledge_version = knowledge_version
        self.runtime_version = runtime_version

    @staticmethod
    def _report_id(run_id: str) -> str:
        return run_id

    @staticmethod
    def _ensure_case_and_call(session, request: AnalysisRequest) -> None:
        now = _utcnow()
        session.execute(
            pg_insert(CaseRow)
            .values(
                case_id=request.caseId,
                customer_display_name="未知（API 提交）",
                assigned_agent_display_name=None,
                source_kind="IMPORTED",
                is_demo=False,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(index_elements=[CaseRow.case_id])
        )
        duration_ms = int(max((turn.end for turn in request.transcript), default=0) * 1000)
        session.execute(
            pg_insert(CallRow)
            .values(
                call_id=request.callId,
                case_id=request.caseId,
                call_started_at=request.callStartedAt,
                duration_ms=duration_ms,
                transcript_json=[
                    turn.model_dump(mode="json") for turn in request.transcript
                ],
                transcript_version="request-v1",
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(index_elements=[CallRow.call_id])
        )

    def create_run(self, run_id: str, request: AnalysisRequest) -> None:
        try:
            with self.session_factory.begin() as session:
                self._ensure_case_and_call(session, request)
                session.add(
                    QCRunRow(
                        run_id=run_id,
                        case_id=request.caseId,
                        call_id=request.callId,
                        status="RUNNING",
                        request_snapshot=request.model_dump(mode="json"),
                        errors_json=[],
                        loop_used=False,
                        model=self.model,
                        prompt_version=self.prompt_version,
                        rule_version=self.rule_version,
                        knowledge_version=self.knowledge_version,
                        runtime_version=self.runtime_version,
                        started_at=_utcnow(),
                    )
                )
        except IntegrityError as exc:
            raise ValueError(f"run already exists: {run_id}") from exc

    def append_event(self, run_id: str, event: AgentTraceEvent) -> None:
        with self.session_factory.begin() as session:
            if session.get(QCRunRow, run_id) is None:
                raise KeyError(run_id)
            session.add(
                AgentTraceEventRow(
                    run_id=run_id,
                    iteration=event.iteration,
                    phase=event.phase,
                    event_json=event.model_dump(mode="json"),
                    created_at=_utcnow(),
                )
            )
            run = session.get(QCRunRow, run_id)
            run.loop_used = True

    def finish_run(
        self,
        run_id: str,
        status: TerminalStatus,
        report: QualityReport | None,
        errors: list[AnalysisError],
        route_reasons=None,
    ) -> None:
        if status not in {"COMPLETED", "PARTIAL", "FAILED"}:
            raise ValueError(f"invalid terminal status: {status}")
        if status != "FAILED" and report is None:
            raise ValueError(f"{status} requires a report")
        if status == "PARTIAL" and report is not None:
            if report.disposition.value != "HUMAN_REVIEW_REQUIRED":
                raise ValueError("PARTIAL report must require human review")

        try:
            with self.session_factory.begin() as session:
                result = session.execute(
                    update(QCRunRow)
                    .where(QCRunRow.run_id == run_id, QCRunRow.status == "RUNNING")
                    .values(
                        status=status,
                        errors_json=[item.model_dump(mode="json") for item in errors],
                        finished_at=_utcnow(),
                    )
                )
                if result.rowcount != 1:
                    existing = session.get(QCRunRow, run_id)
                    if existing is None:
                        raise KeyError(run_id)
                    raise ValueError(
                        f"run is already terminal: {run_id} ({existing.status})"
                    )
                if report is not None:
                    session.add(
                        QCReportRow(
                            report_id=self._report_id(run_id),
                            run_id=run_id,
                            score=report.score,
                            disposition=report.disposition.value,
                            report_json=report.model_dump(mode="json"),
                            created_at=_utcnow(),
                        )
                    )
                if needs_review_task(status, report):
                    reasons = route_reasons or compute_route_reasons(
                        status,
                        report,
                        errors,
                    )
                    ensure_review_task_in_session(session, run_id, reasons)
        except IntegrityError as exc:
            raise ValueError(f"report already exists for run: {run_id}") from exc

    def save_result(
        self,
        run_id: str,
        status: str,
        report: QualityReport,
    ) -> None:
        self.finish_run(run_id, status, report, [])

    def fail_incomplete_runs(self, error: AnalysisError) -> int:
        with self.session_factory.begin() as session:
            result = session.execute(
                update(QCRunRow)
                .where(QCRunRow.status == "RUNNING")
                .values(
                    status="FAILED",
                    errors_json=[error.model_dump(mode="json")],
                    finished_at=_utcnow(),
                )
            )
            return int(result.rowcount or 0)

    @staticmethod
    def _review_bundle(session, run_id: str) -> dict:
        from qc.review_store import PostgresReviewStore

        helper = PostgresReviewStore.__new__(PostgresReviewStore)
        return helper.summaries_for_run_in_session(session, run_id)

    @classmethod
    def _run_payload(cls, session, run: QCRunRow) -> dict:
        report = session.scalar(
            select(QCReportRow).where(QCReportRow.run_id == run.run_id)
        )
        events = session.scalars(
            select(AgentTraceEventRow)
            .where(AgentTraceEventRow.run_id == run.run_id)
            .order_by(AgentTraceEventRow.event_id)
        ).all()
        payload = {
            "runId": run.run_id,
            "caseId": run.case_id,
            "callId": run.call_id,
            "status": run.status,
            "request": run.request_snapshot,
            "result": report.report_json if report is not None else None,
            "errors": run.errors_json or [],
            "events": [item.event_json for item in events],
            "loopUsed": run.loop_used,
            "loopReason": run.loop_reason,
            "model": run.model,
            "promptVersion": run.prompt_version,
            "ruleVersion": run.rule_version,
            "knowledgeVersion": run.knowledge_version,
            "runtimeVersion": run.runtime_version,
            "startedAt": run.started_at.isoformat(),
            "finishedAt": run.finished_at.isoformat() if run.finished_at else None,
            "reportId": report.report_id if report is not None else None,
        }
        usage_rows = session.scalars(
            select(LLMUsageRecordRow).where(LLMUsageRecordRow.run_id == run.run_id)
        ).all()
        known_input = [item.input_tokens for item in usage_rows if item.input_tokens is not None]
        known_output = [item.output_tokens for item in usage_rows if item.output_tokens is not None]
        known_cost = [item.estimated_cost for item in usage_rows if item.estimated_cost is not None]
        payload["usageSummary"] = {
            "callCount": len(usage_rows),
            "inputTokens": sum(known_input) if known_input else None,
            "outputTokens": sum(known_output) if known_output else None,
            "estimatedCost": sum(known_cost) if known_cost else None,
            "unknownTokenCount": sum(item.token_source == "unknown" for item in usage_rows),
        }
        payload.update(cls._review_bundle(session, run.run_id))
        return payload

    def get_review_summary(self, run_id: str) -> dict | None:
        with self.session_factory() as session:
            bundle = self._review_bundle(session, run_id)
            return bundle.get("reviewTask")

    def get_run(self, run_id: str) -> dict:
        with self.session_factory() as session:
            run = session.get(QCRunRow, run_id)
            if run is None:
                raise KeyError(run_id)
            return self._run_payload(session, run)

    def list_incomplete(self) -> list[dict]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(QCRunRow)
                .where(QCRunRow.status == "RUNNING")
                .order_by(QCRunRow.started_at)
            ).all()
            return [
                {
                    "run_id": row.run_id,
                    "case_id": row.case_id,
                    "call_id": row.call_id,
                    "status": row.status,
                }
                for row in rows
            ]

    def list_runs_by_call(self, call_id: str) -> list[dict]:
        with self.session_factory() as session:
            rows = session.execute(
                select(QCRunRow, QCReportRow)
                .outerjoin(QCReportRow, QCReportRow.run_id == QCRunRow.run_id)
                .where(QCRunRow.call_id == call_id)
                .order_by(QCRunRow.started_at.desc(), QCRunRow.run_id.desc())
            ).all()
            return [
                {
                    "runId": run.run_id,
                    "status": run.status,
                    "loopUsed": run.loop_used,
                    "loopReason": run.loop_reason,
                    "model": run.model,
                    "promptVersion": run.prompt_version,
                    "ruleVersion": run.rule_version,
                    "knowledgeVersion": run.knowledge_version,
                    "runtimeVersion": run.runtime_version,
                    "startedAt": run.started_at.isoformat(),
                    "finishedAt": run.finished_at.isoformat() if run.finished_at else None,
                    "reportId": report.report_id if report else None,
                    "score": report.score if report else None,
                    "disposition": report.disposition if report else None,
                    "errors": run.errors_json or [],
                    **self._review_bundle(session, run.run_id),
                }
                for run, report in rows
            ]

    def find_runs(
        self,
        call_id: str | None = None,
        rule_id: str | None = None,
        event_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        with self.session_factory() as session:
            statement: Select = (
                select(QCRunRow, QCReportRow)
                .join(QCReportRow, QCReportRow.run_id == QCRunRow.run_id)
                .order_by(QCRunRow.started_at.desc())
                .limit(500)
            )
            if call_id:
                statement = statement.where(QCRunRow.call_id == call_id)
            rows = session.execute(statement).all()

        wanted_events = set(event_types or [])
        matches = []
        for run, stored_report in rows:
            report = stored_report.report_json or {}
            events = report.get("events") or []
            violations = report.get("violations") or []
            event_values = [item.get("type") for item in events if isinstance(item, dict)]
            rule_ids = [item.get("ruleId") for item in violations if isinstance(item, dict)]
            if rule_id and rule_id not in rule_ids:
                continue
            if wanted_events and not wanted_events.intersection(event_values):
                continue
            matches.append(
                {
                    "runId": run.run_id,
                    "caseId": run.case_id,
                    "callId": run.call_id,
                    "status": run.status,
                    "score": stored_report.score,
                    "disposition": stored_report.disposition,
                    "eventTypes": event_values,
                    "ruleIds": rule_ids,
                }
            )
            if len(matches) >= limit:
                break
        return matches

    def list_cases(self, page: int = 1, page_size: int = 20) -> dict:
        page = max(1, int(page))
        page_size = min(100, max(1, int(page_size)))
        with self.session_factory() as session:
            total = session.query(CaseRow).count()
            cases = session.scalars(
                select(CaseRow)
                .order_by(CaseRow.updated_at.desc(), CaseRow.case_id)
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            items = []
            for case in cases:
                call = session.scalar(
                    select(CallRow)
                    .where(CallRow.case_id == case.case_id)
                    .order_by(CallRow.call_started_at.desc())
                    .limit(1)
                )
                items.append(self._case_summary(case, call))
        return {"items": items, "page": page, "pageSize": page_size, "total": total}

    @staticmethod
    def _call_summary(call: CallRow | None) -> dict | None:
        if call is None:
            return None
        return {
            "callId": call.call_id,
            "callStartedAt": call.call_started_at.isoformat(),
            "durationMs": call.duration_ms,
            "audioAvailable": bool(call.audio_artifact_uri),
        }

    @classmethod
    def _case_summary(cls, case: CaseRow, call: CallRow | None) -> dict:
        return {
            "caseId": case.case_id,
            "customerDisplayName": case.customer_display_name,
            "assignedAgentDisplayName": case.assigned_agent_display_name,
            "sourceKind": case.source_kind,
            "isDemo": case.is_demo,
            "latestCall": cls._call_summary(call),
        }

    def get_case(self, case_id: str) -> dict:
        with self.session_factory() as session:
            case = session.get(CaseRow, case_id)
            if case is None:
                raise KeyError(case_id)
            calls = session.scalars(
                select(CallRow)
                .where(CallRow.case_id == case_id)
                .order_by(CallRow.call_started_at.desc())
            ).all()
            payload = self._case_summary(case, calls[0] if calls else None)
            payload["calls"] = [self._call_summary(call) for call in calls]
            return payload

    def get_call(self, call_id: str) -> dict:
        with self.session_factory() as session:
            call = session.get(CallRow, call_id)
            if call is None:
                raise KeyError(call_id)
            case = session.get(CaseRow, call.case_id)
            latest = session.scalar(
                select(QCRunRow)
                .where(QCRunRow.call_id == call_id)
                .order_by(QCRunRow.started_at.desc())
                .limit(1)
            )
            return {
                "callId": call.call_id,
                "caseId": call.case_id,
                "callStartedAt": call.call_started_at.isoformat(),
                "durationMs": call.duration_ms,
                "sourceKind": case.source_kind if case else "IMPORTED",
                "transcriptVersion": call.transcript_version,
                "audioAvailable": bool(call.audio_artifact_uri),
                "audioArtifactUri": call.audio_artifact_uri,
                "audioSha256": call.audio_sha256,
                "audioMimeType": call.audio_mime_type,
                "latestRunSummary": (
                    {"runId": latest.run_id, "status": latest.status} if latest else None
                ),
            }

    def get_transcript(self, call_id: str) -> list[dict]:
        with self.session_factory() as session:
            call = session.get(CallRow, call_id)
            if call is None:
                raise KeyError(call_id)
            return list(call.transcript_json)

    def get_report(self, report_id: str) -> dict:
        with self.session_factory() as session:
            report = session.get(QCReportRow, report_id)
            if report is None:
                raise KeyError(report_id)
            run = session.get(QCRunRow, report.run_id)
            payload = {
                "reportId": report.report_id,
                "runId": report.run_id,
                "report": report.report_json,
                "model": run.model,
                "promptVersion": run.prompt_version,
                "ruleVersion": run.rule_version,
                "knowledgeVersion": run.knowledge_version,
                "runtimeVersion": run.runtime_version,
                "createdAt": report.created_at.isoformat(),
            }
            payload.update(self._review_bundle(session, report.run_id))
            return payload

    def close(self) -> None:
        self.engine.dispose()
