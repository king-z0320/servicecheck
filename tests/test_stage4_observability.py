from __future__ import annotations

import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from qc.observability.logging import JsonFormatter
from qc.observability.metrics import MetricsRegistry
from qc.observability.redaction import redact_for_observability
from qc.observability.tracing import InMemoryTracing, JsonLinesSpanExporter, traced
from qc.observability.usage import InMemoryUsageLedger, UsageRecord
from qc.llm_gateway import DeepSeekGateway


class _UsageResponse:
    status_code = 200
    headers = {}
    def json(self):
        return {"choices": [{"message": {"content": '{"value":"ok"}'}}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


class _UsageSession:
    def post(self, *args, **kwargs):
        return _UsageResponse()


def test_redaction_never_keeps_prompt_transcript_or_phone_number():
    value = redact_for_observability(
        {
            "authorization": "Bearer secret",
            "prompt": "完整提示词",
            "transcript": "13800138000 客户原话",
            "contentHash": "safe-hash",
            "count": 2,
        }
    )
    serialized = str(value)
    assert "secret" not in serialized
    assert "完整提示词" not in serialized
    assert "13800138000" not in serialized
    assert value["contentHash"] == "safe-hash"
    assert value["count"] == 2


def test_tracing_spans_keep_ids_but_not_sensitive_attributes():
    tracing = InMemoryTracing()
    with tracing.activate():
        with traced("event_extract", run_id="RUN-1", transcript="13900000000", model="deepseek-chat"):
            pass
    spans = tracing.finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes)
    assert attributes["run.id"] == "RUN-1"
    assert "transcript" not in attributes


def test_usage_ledger_preserves_unknown_and_deduplicates_invocation():
    ledger = InMemoryUsageLedger()
    record = UsageRecord(
        invocationId="INV-1", operation="event_extract", provider="deepseek", model="deepseek-chat",
        attempt=1, tokenSource="unknown", inputTokens=None, outputTokens=None, estimatedCost=None,
    )
    ledger.record(record)
    ledger.record(record)
    assert ledger.summary()["callCount"] == 1
    assert ledger.summary()["inputTokens"] is None


def test_gateway_records_provider_usage_without_changing_legacy_result_shape():
    ledger = InMemoryUsageLedger()
    gateway = DeepSeekGateway("test", session=_UsageSession(), usage_ledger=ledger, operation="event_extract")
    assert gateway.complete_json(system="s", user="u", schema={"type": "object"}) == {"value": "ok"}
    assert ledger.summary()["inputTokens"] == 11
    assert ledger.summary()["outputTokens"] == 7


def test_gateway_allows_judge_usage_to_be_recorded_as_a_separate_operation():
    ledger = InMemoryUsageLedger()
    gateway = DeepSeekGateway("test", session=_UsageSession(), usage_ledger=ledger)
    gateway.complete_json(
        system="s",
        user="u",
        schema={"type": "object"},
        operation="judge",
    )
    assert ledger.records()[0].operation == "judge"


def test_metrics_are_low_cardinality_and_exportable():
    metrics = MetricsRegistry()
    metrics.observe_stage("event_extract", 0.2, status="completed")
    metrics.record_llm_error("LLM_TIMEOUT")
    text = metrics.render().decode("utf-8")
    assert "servicecheck_stage_duration_seconds" in text
    assert "LLM_TIMEOUT" in text
    assert "run_id" not in text


def test_json_formatter_outputs_sanitized_structured_event():
    formatter = JsonFormatter()
    record = __import__("logging").LogRecord(
        "test", 20, "", 0, "event", (), None
    )
    record.observability = {"prompt": "不应输出", "runId": "RUN-1"}
    text = formatter.format(record)
    assert '"runId": "RUN-1"' in text
    assert "不应输出" not in text


def test_json_lines_span_exporter_writes_sanitized_spans_to_project_path(tmp_path):
    destination = tmp_path / "spans.jsonl"
    exporter = JsonLinesSpanExporter(destination)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    with provider.get_tracer("test").start_as_current_span(
        "event_extract",
        attributes={
            "run.id": "RUN-001",
            "eval.run_id": "EVAL-001",
            "transcript": "13900000000 客户原话",
        },
    ):
        pass
    provider.force_flush()

    payload = json.loads(destination.read_text(encoding="utf-8").strip())
    assert payload["name"] == "event_extract"
    assert payload["traceId"]
    assert payload["spanId"]
    assert payload["attributes"]["run.id"] == "RUN-001"
    assert payload["attributes"]["eval.run_id"] == "EVAL-001"
    assert payload["attributes"]["transcript"]["redacted"] is True
    assert "13900000000" not in destination.read_text(encoding="utf-8")
