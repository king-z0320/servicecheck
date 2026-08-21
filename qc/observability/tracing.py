from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from qc.observability.redaction import redact_for_observability

_context: ContextVar[dict[str, str]] = ContextVar("servicecheck_observability_context", default={})
_tracer_override: ContextVar[Any | None] = ContextVar(
    "servicecheck_observability_tracer_override",
    default=None,
)


class SafeExporter(SpanExporter):
    def __init__(self, delegate):
        self.delegate = delegate
    def export(self, spans):
        try:
            return self.delegate.export(spans)
        except Exception:
            return SpanExportResult.FAILURE
    def shutdown(self):
        try:
            self.delegate.shutdown()
        except Exception:
            pass


class JsonLinesSpanExporter(SpanExporter):
    """Append a minimal, redacted OTel Span record for local diagnosis."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    @staticmethod
    def _hex(value: int, width: int) -> str:
        return f"{value:0{width}x}"

    def export(self, spans):
        try:
            rows = []
            for span in spans:
                context = span.get_span_context()
                parent = span.parent
                rows.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "traceId": self._hex(context.trace_id, 32),
                        "spanId": self._hex(context.span_id, 16),
                        "parentSpanId": (
                            self._hex(parent.span_id, 16)
                            if parent is not None and parent.is_valid
                            else None
                        ),
                        "name": span.name,
                        "kind": span.kind.name,
                        "startTimeUnixNano": span.start_time,
                        "endTimeUnixNano": span.end_time,
                        "status": span.status.status_code.name,
                        "attributes": redact_for_observability(dict(span.attributes or {})),
                    }
                )
            if not rows:
                return SpanExportResult.SUCCESS
            self.path.parent.mkdir(parents=True, exist_ok=True)
            content = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            )
            with self._lock:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self):
        return None


class InMemoryTracing:
    def __init__(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(SafeExporter(self.exporter)))
        self.tracer = self.provider.get_tracer("servicecheck")

    @contextmanager
    def activate(self):
        token = _tracer_override.set(self.tracer)
        try:
            yield self
        finally:
            _tracer_override.reset(token)

    def finished_spans(self) -> list[ReadableSpan]:
        return list(self.exporter.get_finished_spans())


def configure_tracing(*, exporter=None):
    provider = TracerProvider()
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(SafeExporter(exporter)))
    trace.set_tracer_provider(provider)
    return provider


def _trace_id(span) -> str | None:
    context = span.get_span_context()
    if not context.is_valid:
        return None
    return f"{context.trace_id:032x}"


@contextmanager
def traced(name: str, **attributes: Any):
    tracer = _tracer_override.get() or trace.get_tracer("servicecheck")
    safe = redact_for_observability(attributes)
    attribute_names = {
        "run_id": "run.id",
        "batch_id": "batch.id",
        "item_id": "item.id",
        "eval_run_id": "eval.run_id",
        "case_id": "eval.case_id",
    }
    flattened = {
        attribute_names.get(str(key), str(key)): value
        for key, value in safe.items()
        if not isinstance(value, (dict, list))
    }
    with tracer.start_as_current_span(name, attributes=flattened) as span:
        context = _context.get().copy()
        context_keys = {
            "run_id": "runId",
            "runId": "runId",
            "batch_id": "batchId",
            "batchId": "batchId",
            "item_id": "itemId",
            "itemId": "itemId",
            "eval_run_id": "evalRunId",
            "evalRunId": "evalRunId",
            "case_id": "caseId",
            "caseId": "caseId",
        }
        for source_key, target_key in context_keys.items():
            if source_key in attributes:
                context[target_key] = str(attributes[source_key])
        trace_id = _trace_id(span)
        if trace_id is not None:
            context["traceId"] = trace_id
        token = _context.set(context)
        started = datetime.now(timezone.utc)
        try:
            yield span
        finally:
            duration_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
            import logging

            logging.getLogger("servicecheck.trace").info(
                "span.completed",
                extra={
                    "observability": {
                        "eventName": "span.completed",
                        "spanName": name,
                        "traceId": trace_id,
                        "durationMs": round(duration_ms, 3),
                        **safe,
                    }
                },
            )
            _context.reset(token)


def current_context() -> dict[str, str]:
    value = dict(_context.get())
    trace_id = _trace_id(trace.get_current_span())
    if trace_id is not None:
        value["traceId"] = trace_id
    return value
