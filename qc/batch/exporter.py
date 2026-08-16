from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

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
    def __init__(self, store: BatchStore):
        self.store = store

    def export_json(self, batch_id: str, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        version = "batch-json-export-v1"
        self.store.begin_export(batch_id, "json", out_path, version)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            files = self.store.list_files(batch_id)
            payload = {
                "batch_id": batch_id,
                "summary": self.store.batch_summary(batch_id),
                "files": files,
            }
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
            self.store.complete_export(
                batch_id, "json", out_path, version, digest
            )
            return out_path
        except Exception:
            self.store.fail_export(
                batch_id, "json", out_path, version, "EXPORT_FAILED"
            )
            raise

    def export_csv(self, batch_id: str, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        version = "batch-csv-export-v1"
        self.store.begin_export(batch_id, "csv", out_path, version)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            files = self.store.list_files(batch_id)
            columns = [
                "callId",
                "source_uri",
                "status",
                "score",
                "disposition",
                "failed_reason",
            ]
            with out_path.open("w", encoding="utf-8", newline="") as fh:
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
            digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
            self.store.complete_export(
                batch_id, "csv", out_path, version, digest
            )
            return out_path
        except Exception:
            self.store.fail_export(
                batch_id, "csv", out_path, version, "EXPORT_FAILED"
            )
            raise
