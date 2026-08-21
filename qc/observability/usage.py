from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class UsageRecord:
    invocationId: str
    operation: str
    provider: str
    model: str
    attempt: int
    tokenSource: str
    inputTokens: int | None
    outputTokens: int | None
    estimatedCost: float | None
    runId: str | None = None
    evalRunId: str | None = None
    latencyMs: float | None = None
    priceConfigVersion: str | None = None


class InMemoryUsageLedger:
    def __init__(self):
        self._records: dict[str, UsageRecord] = {}
        self._lock = Lock()

    def record(self, record: UsageRecord) -> None:
        with self._lock:
            self._records.setdefault(record.invocationId, record)

    def records(self) -> list[UsageRecord]:
        return list(self._records.values())

    def summary(self) -> dict[str, Any]:
        records = self.records()
        input_values = [r.inputTokens for r in records if r.inputTokens is not None]
        output_values = [r.outputTokens for r in records if r.outputTokens is not None]
        costs = [r.estimatedCost for r in records if r.estimatedCost is not None]
        return {"callCount": len(records), "inputTokens": sum(input_values) if input_values else None, "outputTokens": sum(output_values) if output_values else None, "estimatedCost": sum(costs) if costs else None, "unknownTokenCount": sum(r.tokenSource == "unknown" for r in records)}


class PostgresUsageLedger:
    """Small adapter that keeps the Gateway independent from SQLAlchemy."""

    def __init__(self, database_url: str):
        from qc.eval.store import PostgresEvalStore

        self.store = PostgresEvalStore(database_url)

    def record(self, record: UsageRecord) -> None:
        self.store.record_usage(record)


def usage_from_response(envelope: dict[str, Any], *, operation: str, provider: str, model: str, invocation_id: str, attempt: int, run_id: str | None = None, eval_run_id: str | None = None, latency_ms: float | None = None) -> UsageRecord:
    usage = envelope.get("usage") if isinstance(envelope, dict) else None
    if not isinstance(usage, dict):
        return UsageRecord(invocation_id, operation, provider, model, attempt, "unknown", None, None, None, run_id, eval_run_id, latency_ms)
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return UsageRecord(invocation_id, operation, provider, model, attempt, "unknown", None, None, None, run_id, eval_run_id, latency_ms)
    return UsageRecord(invocation_id, operation, provider, model, attempt, "provider_reported", input_tokens, output_tokens, None, run_id, eval_run_id, latency_ms)
