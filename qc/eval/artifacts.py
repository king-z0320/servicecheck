"""Durable, redacted evidence artifacts for evaluation runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qc.observability.redaction import redact_for_observability


class EvalArtifactWriter:
    def __init__(self, root: str | Path = "eval_runs"):
        self.root = Path(root)

    def write(self, eval_run_id: str, *, manifest: dict[str, Any], metrics: dict[str, Any], failures: list[dict[str, Any]], diff: str, traces: list[dict[str, Any]]):
        directory = (self.root / eval_run_id).resolve()
        root = self.root.resolve()
        if root not in directory.parents and directory != root:
            raise ValueError("artifact path escapes project eval_runs root")
        directory.mkdir(parents=True, exist_ok=True)
        self._atomic_json(directory / "manifest.json", manifest)
        self._atomic_json(directory / "metrics.json", metrics)
        self._atomic_json(directory / "failures.json", redact_for_observability(failures))
        self._atomic_text(directory / "diff.md", diff)
        content = "".join(json.dumps(redact_for_observability(item), ensure_ascii=False, sort_keys=True) + "\n" for item in traces)
        self._atomic_text(directory / "traces.jsonl", content)
        return directory

    @staticmethod
    def _atomic_text(path: Path, value: str):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)

    def _atomic_json(self, path: Path, value: Any):
        self._atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))
