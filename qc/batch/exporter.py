from __future__ import annotations

import csv
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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        files = self.store.list_files(batch_id)
        payload = {
            "batch_id": batch_id,
            "summary": self.store.batch_summary(batch_id),
            "files": files,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def export_csv(self, batch_id: str, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        files = self.store.list_files(batch_id)
        columns = ["callId", "source_uri", "status", "score", "disposition", "failed_reason"]
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for row in files:
                writer.writerow(
                    {
                        "callId": row["call_id"] or "",
                        "source_uri": row["source_uri"],
                        "status": row["status"],
                        "score": _extract_score(row["result_json"]),
                        "disposition": _extract_disposition(row["result_json"]),
                        "failed_reason": row["failed_reason"] or "",
                    }
                )
        return out_path
