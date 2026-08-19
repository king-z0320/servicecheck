from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

from qc.batch.models import BatchConfig, BatchMeta, FileRecord


class BatchCapacityError(RuntimeError):
    """The configured batch or outstanding-item capacity would be exceeded."""


class IdempotencyConflictError(ValueError):
    """An Idempotency-Key was reused with a different request body."""


def batch_request_hash(source_dir: str) -> str:
    return hashlib.sha256(
        source_dir.strip().encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_directory(root: Path, source_dir: str) -> Path:
    if not isinstance(source_dir, str) or not source_dir.strip():
        raise ValueError("source_dir must be a non-empty relative directory")
    if "\x00" in source_dir or "\\" in source_dir:
        raise ValueError("source_dir must be a safe relative POSIX directory")
    posix = PurePosixPath(source_dir)
    windows = PureWindowsPath(source_dir)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError("source_dir must be relative")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError("source_dir escapes the configured audio root")
    candidate = (root / Path(*posix.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("source_dir escapes the configured audio root") from exc
    if not candidate.is_dir():
        raise ValueError("source_dir must refer to an existing directory")
    return candidate


def discover_relative_audio(root: Path, source_dir: str) -> list[FileRecord]:
    directory = _safe_relative_directory(root, source_dir)
    records: list[FileRecord] = []
    allowed = {".m4a", ".wav", ".mp3", ".flac"}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
        except ValueError as exc:
            raise ValueError("audio file escapes the configured audio root") from exc
        digest = _sha256_file(resolved)
        records.append(
            FileRecord(
                source_uri=relative,
                idempotency_key=hashlib.sha256(
                    f"{relative}\0{digest}".encode("utf-8")
                ).hexdigest(),
                metadata={"size": path.stat().st_size, "sha256": digest},
            )
        )
    if not records:
        raise ValueError("source_dir contains no supported audio files")
    return records


class InMemoryBatchService:
    """Offline service used by API tests; mirrors the PostgreSQL contract."""

    def __init__(self, audio_root: str | Path, config: BatchConfig | None = None):
        self.audio_root = Path(audio_root).resolve(strict=True)
        self.config = config or BatchConfig()
        self.batches: dict[str, dict] = {}
        self.items: dict[str, list[dict]] = {}
        self.requests: dict[str, tuple[str, str]] = {}

    def create_batch(self, source_dir: str, idempotency_key: str | None = None) -> dict:
        request_hash = batch_request_hash(source_dir)
        if idempotency_key and idempotency_key in self.requests:
            existing_hash, batch_id = self.requests[idempotency_key]
            if existing_hash != request_hash:
                raise IdempotencyConflictError(
                    "Idempotency-Key was reused with a different request"
                )
            return self.batches[batch_id]
        records = discover_relative_audio(self.audio_root, source_dir)
        if len(records) > self.config.max_batch_items:
            raise BatchCapacityError(
                f"batch contains {len(records)} items; max_batch_items="
                f"{self.config.max_batch_items}"
            )
        pending = sum(
            1
            for rows in self.items.values()
            for item in rows
            if item["status"] in {"PENDING", "RUNNING", "INTERRUPTED"}
        )
        if pending + len(records) > self.config.queue_max_pending:
            raise BatchCapacityError(
                "pending queue capacity would be exceeded: "
                f"{pending}+{len(records)}>{self.config.queue_max_pending}"
            )
        batch_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        batch = {
            "batch_id": batch_id,
            "source": source_dir,
            "status": "QUEUED",
            "total": len(records),
            "created_at": now,
            "started_at": None,
            "finished_at": None,
        }
        self.batches[batch_id] = batch
        self.items[batch_id] = [
            {
                "file_id": index,
                "item_id": index,
                "batch_id": batch_id,
                "source_uri": record.source_uri,
                "metadata": record.metadata,
                "idempotency_key": record.idempotency_key,
                "call_id": record.callId,
                "status": "PENDING",
                "failed_reason": None,
                "stages": [],
            }
            for index, record in enumerate(records, 1)
        ]
        if idempotency_key:
            self.requests[idempotency_key] = (request_hash, batch_id)
        return batch

    def list_items(self, batch_id: str) -> list[dict]:
        if batch_id not in self.items:
            raise KeyError(batch_id)
        return self.items[batch_id]

    def get_batch(self, batch_id: str) -> dict:
        if batch_id not in self.batches:
            raise KeyError(batch_id)
        batch = dict(self.batches[batch_id])
        items = self.items[batch_id]
        counts: dict[str, int] = {}
        for item in items:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        batch["by_status"] = counts
        return batch


class PostgresBatchService:
    """Batch control-plane service backed by a PostgresBatchStore."""

    def __init__(
        self,
        store,
        audio_root: str | Path,
        config: BatchConfig | None = None,
    ):
        self.store = store
        self.audio_root = Path(audio_root).resolve(strict=True)
        self.config = config or BatchConfig()

    def create_batch(self, source_dir: str, idempotency_key: str | None = None) -> dict:
        request_hash = batch_request_hash(source_dir)
        if idempotency_key and hasattr(self.store, "get_idempotent_batch"):
            existing = self.store.get_idempotent_batch(idempotency_key, request_hash)
            if existing is not None:
                return existing
        records = discover_relative_audio(self.audio_root, source_dir)
        if len(records) > self.config.max_batch_items:
            raise BatchCapacityError(
                f"batch contains {len(records)} items; max_batch_items="
                f"{self.config.max_batch_items}"
            )
        batch_id = str(uuid4())
        meta = BatchMeta(
            batch_id=batch_id,
            source=source_dir,
            total=len(records),
        )
        if hasattr(self.store, "create_batch_with_outbox"):
            return self.store.create_batch_with_outbox(
                meta,
                records,
                idempotency_key,
                request_hash=request_hash,
                max_pending=self.config.queue_max_pending,
            )
        if hasattr(self.store, "pending_item_count"):
            pending = self.store.pending_item_count()
            if pending + len(records) > self.config.queue_max_pending:
                raise BatchCapacityError(
                    "pending queue capacity would be exceeded: "
                    f"{pending}+{len(records)}>{self.config.queue_max_pending}"
                )
        self.store.create_batch(meta)
        for record in records:
            self.store.add_file(batch_id, record)
        return {
            "batch_id": batch_id,
            "source": source_dir,
            "status": "QUEUED",
            "total": len(records),
        }

    def list_items(self, batch_id: str) -> list[dict]:
        self.get_batch(batch_id)
        return self.store.list_files(batch_id)

    def get_batch(self, batch_id: str) -> dict:
        summary = self.store.batch_summary(batch_id)
        if summary.get("total", 0) == 0 and not self.store.list_files(batch_id):
            raise KeyError(batch_id)
        counts = summary.get("by_status", {})
        if counts and all(status in {"DONE", "HUMAN_REVIEW", "FAILED_FINAL"} for status in counts):
            if counts.get("FAILED_FINAL"):
                status = "FAILED"
            elif counts.get("HUMAN_REVIEW"):
                status = "PARTIAL"
            else:
                status = "COMPLETED"
        elif counts.get("RUNNING") or counts.get("INTERRUPTED"):
            status = "RUNNING"
        else:
            status = "QUEUED"
        return {**summary, "status": status}
