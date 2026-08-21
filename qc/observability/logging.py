from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from qc.observability.redaction import redact_for_observability
from qc.observability.tracing import current_context


_LOG_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)
_EXTRA_FIELDS = {
    "run_id": "runId",
    "eval_run_id": "evalRunId",
    "case_id": "caseId",
    "batch_id": "batchId",
    "item_id": "itemId",
    "event_id": "eventId",
    "error_code": "errorCode",
    "stage": "stage",
    "status": "status",
    "attempt": "attempt",
    "duration_ms": "durationMs",
    "operation": "operation",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_for_observability(record.getMessage()),
        }
        payload.update(current_context())
        for source_key, target_key in _EXTRA_FIELDS.items():
            if hasattr(record, source_key):
                payload[target_key] = getattr(record, source_key)
        payload.update(redact_for_observability(getattr(record, "observability", {})))
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_json_logging(level: int = logging.INFO, *, path: str | Path | None = None) -> None:
    root = logging.getLogger()
    if not any(getattr(item, "_servicecheck_json_stream", False) for item in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._servicecheck_json_stream = True
        root.addHandler(handler)
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        resolved = destination.resolve()
        configured = {
            getattr(item, "_servicecheck_json_path", None)
            for item in root.handlers
        }
        if str(resolved) not in configured:
            handler = logging.FileHandler(resolved, encoding="utf-8")
            handler.setFormatter(JsonFormatter())
            handler._servicecheck_json_path = str(resolved)
            root.addHandler(handler)
    root.setLevel(level)
