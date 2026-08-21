"""PostgreSQL persistence for evaluation evidence and usage records."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from qc.database import create_database_engine, create_session_factory
from qc.orm_models import EvalCaseResultRow, EvalRunRow, LLMUsageRecordRow
from qc.observability.usage import UsageRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PostgresEvalStore:
    """Repository for stage-4 evidence; no complete Prompt or transcript is accepted."""

    def __init__(self, database_url: str):
        self.engine = create_database_engine(database_url)
        self.session_factory = create_session_factory(self.engine)

    def create_run(self, result, artifact_uri: str) -> None:
        with self.session_factory.begin() as session:
            session.add(EvalRunRow(eval_run_id=result.evalRunId, split=result.manifest.split.value, status="RUNNING", dataset_hash=result.manifest.datasetHash, manifest=result.manifest.model_dump(mode="json"), artifact_uri=artifact_uri, case_count=len(result.caseResults), completed_count=0, failed_count=0, created_at=_utcnow(), started_at=_utcnow()))

    def persist(self, result, artifact_uri: str) -> None:
        self.create_run(result, artifact_uri)
        for case_result in result.caseResults:
            self.save_case_result(result.evalRunId, case_result)
        self.finish_run(result)

    def save_case_result(self, eval_run_id: str, result) -> None:
        with self.session_factory.begin() as session:
            statement = pg_insert(EvalCaseResultRow).values(eval_run_id=eval_run_id, case_id=result.caseId, case_hash=result.caseHash or "unknown", status=result.status, deterministic_metrics=result.metrics.deterministic, rag_metrics=result.metrics.rag, judge_result=result.judgeResult or None, failure_reasons=result.metrics.failureReasons, trace_id=result.traceId, run_id=result.runId, created_at=_utcnow()).on_conflict_do_update(constraint="uq_eval_case_results_run_case", set_={"status": result.status, "deterministic_metrics": result.metrics.deterministic, "rag_metrics": result.metrics.rag, "judge_result": result.judgeResult or None, "failure_reasons": result.metrics.failureReasons, "trace_id": result.traceId, "run_id": result.runId})
            session.execute(statement)

    def finish_run(self, result) -> None:
        with self.session_factory.begin() as session:
            response = session.execute(update(EvalRunRow).where(EvalRunRow.eval_run_id == result.evalRunId, EvalRunRow.status == "RUNNING").values(status=result.status, completed_count=len(result.caseResults), failed_count=sum(item.status == "failed" for item in result.caseResults), finished_at=_utcnow()))
            if response.rowcount != 1:
                raise KeyError(result.evalRunId)

    def record_usage(self, record: UsageRecord) -> None:
        with self.session_factory.begin() as session:
            statement = pg_insert(LLMUsageRecordRow).values(invocation_id=record.invocationId, run_id=record.runId, eval_run_id=record.evalRunId, operation=record.operation, provider=record.provider, model=record.model, attempt=record.attempt, token_source=record.tokenSource, input_tokens=record.inputTokens, output_tokens=record.outputTokens, estimated_cost=record.estimatedCost, latency_ms=record.latencyMs, price_config_version=record.priceConfigVersion, created_at=_utcnow()).on_conflict_do_nothing(constraint="uq_llm_usage_records_invocation")
            session.execute(statement)

    def usage_summary(self, *, run_id: str | None = None, eval_run_id: str | None = None) -> dict:
        with self.session_factory() as session:
            statement = select(LLMUsageRecordRow)
            if run_id:
                statement = statement.where(LLMUsageRecordRow.run_id == run_id)
            if eval_run_id:
                statement = statement.where(LLMUsageRecordRow.eval_run_id == eval_run_id)
            rows = session.scalars(statement).all()
        known_input = [item.input_tokens for item in rows if item.input_tokens is not None]
        known_output = [item.output_tokens for item in rows if item.output_tokens is not None]
        known_cost = [item.estimated_cost for item in rows if item.estimated_cost is not None]
        return {"callCount": len(rows), "inputTokens": sum(known_input) if known_input else None, "outputTokens": sum(known_output) if known_output else None, "estimatedCost": sum(known_cost) if known_cost else None, "unknownTokenCount": sum(item.token_source == "unknown" for item in rows)}
