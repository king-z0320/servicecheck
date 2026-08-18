from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from qc.artifact_store import ArtifactStore
from qc.batch.store import BatchStore


def _extract_score(result_json: str | None) -> str:
    if not result_json:
        return ""
    try:
        report = json.loads(result_json).get("report")
        return str(report.get("score", "")) if isinstance(report, dict) else ""
    except (ValueError, TypeError):
        return ""


def _extract_disposition(result_json: str | None) -> str:
    if not result_json:
        return ""
    try:
        report = json.loads(result_json).get("report")
        return str(report.get("disposition", "")) if isinstance(report, dict) else ""
    except (ValueError, TypeError):
        return ""


class Exporter:
    def __init__(self, store: BatchStore, artifact_store: ArtifactStore):
        self.store = store
        self.artifact_store = artifact_store

    def export_json(self, batch_id: str, out_path: str | Path) -> Path:
        artifact_uri = Path(out_path).as_posix()
        version = "batch-json-export-v1"
        self.store.begin_export(batch_id, "json", artifact_uri, version)
        try:
            files = self.store.list_files(batch_id)
            payload = {
                "batch_id": batch_id,
                "summary": self.store.batch_summary(batch_id),
                "files": files,
            }
            reference = self.artifact_store.put_bytes(
                artifact_uri,
                json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                mime_type="application/json",
            )
            self.store.complete_export(
                batch_id, "json", reference.uri, version, reference.sha256
            )
            return self.artifact_store.resolve_for_read(reference.uri)
        except Exception:
            self.store.fail_export(
                batch_id, "json", artifact_uri, version, "EXPORT_FAILED"
            )
            raise

    def export_csv(self, batch_id: str, out_path: str | Path) -> Path:
        artifact_uri = Path(out_path).as_posix()
        version = "batch-csv-export-v1"
        self.store.begin_export(batch_id, "csv", artifact_uri, version)
        try:
            files = self.store.list_files(batch_id)
            columns = [
                "callId",
                "source_uri",
                "status",
                "score",
                "disposition",
                "failed_reason",
            ]
            output = io.StringIO(newline="")
            with output as fh:
                writer = csv.DictWriter(fh, fieldnames=columns)
                writer.writeheader()
                for row in files:
                    has_effective_report = row["status"] in {"DONE", "HUMAN_REVIEW"}
                    result_json = row["result_json"] if has_effective_report else None
                    writer.writerow(
                        {
                            "callId": row["call_id"] or "",
                            "source_uri": row["source_uri"],
                            "status": row["status"],
                            "score": _extract_score(result_json),
                            "disposition": _extract_disposition(result_json),
                            "failed_reason": row["failed_reason"] or "",
                        }
                    )
                content = fh.getvalue().encode("utf-8")
            reference = self.artifact_store.put_bytes(
                artifact_uri,
                content,
                mime_type="text/csv; charset=utf-8",
            )
            self.store.complete_export(
                batch_id, "csv", reference.uri, version, reference.sha256
            )
            return self.artifact_store.resolve_for_read(reference.uri)
        except Exception:
            self.store.fail_export(
                batch_id, "csv", artifact_uri, version, "EXPORT_FAILED"
            )
            raise
