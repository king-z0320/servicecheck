from __future__ import annotations

from pathlib import Path

from qc.observability.logging import configure_json_logging
from qc.observability.tracing import JsonLinesSpanExporter, configure_tracing


def configure_local_observability(project_root: str | Path, *, process_name: str) -> Path:
    """Configure project-local JSON logs and OTel span export for one process."""
    root = Path(project_root).resolve()
    directory = root / ".runtime" / "observability"
    directory.mkdir(parents=True, exist_ok=True)
    configure_json_logging(path=directory / f"{process_name}.log.jsonl")
    configure_tracing(exporter=JsonLinesSpanExporter(directory / f"{process_name}.spans.jsonl"))
    return directory
