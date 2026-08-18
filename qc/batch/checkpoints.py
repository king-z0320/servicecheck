from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from qc.artifact_store import ArtifactStore, LocalArtifactStore
from qc.batch.models import StageName
from qc.batch.store import BatchStore
from qc.models import AnalysisResult, TranscriptTurn


_ARTIFACT_NAMES = {
    StageName.TRANSCODE: "transcode.wav",
    StageName.ASR: "transcript.json",
    StageName.EMOTION: "emotion.json",
    StageName.QC: "analysis-result.json",
}

_DOWNSTREAM = {
    StageName.TRANSCODE: [StageName.ASR, StageName.EMOTION, StageName.QC],
    StageName.ASR: [StageName.QC],
    StageName.EMOTION: [],
    StageName.QC: [],
}


class FileCheckpointSession:
    """Read and write the four recoverable artifacts for one batch file."""

    def __init__(
        self,
        store: BatchStore,
        batch_id: str,
        file_id: int,
        artifact_root: str | Path | None = None,
        artifact_store: ArtifactStore | None = None,
    ):
        self.store = store
        self.batch_id = batch_id
        self.file_id = file_id
        if artifact_store is not None and artifact_root is not None:
            raise ValueError("pass artifact_store or artifact_root, not both")
        if artifact_store is None:
            if artifact_root is not None:
                root = Path(artifact_root)
            elif getattr(store, "path", None) is not None:
                root = Path(store.path).resolve().parent / "batch_artifacts"
            else:
                root = Path(__file__).resolve().parents[2] / "data" / "artifacts"
            artifact_store = LocalArtifactStore(root)
        self.artifact_store = artifact_store

    def _artifact_uri(self, stage: StageName) -> str:
        try:
            name = _ARTIFACT_NAMES[stage]
        except KeyError as exc:
            raise ValueError(f"unsupported checkpoint stage: {stage.value}") from exc
        return f"batch/{self.batch_id}/{self.file_id}/{name}"

    def load(self, stage: StageName, producer_version: str) -> Any | None:
        checkpoint = self.store.get_stage_checkpoint(self.file_id, stage)
        if checkpoint is None or checkpoint["status"] != "DONE":
            return None
        if not all(
            checkpoint.get(field)
            for field in ("artifact_uri", "sha256", "producer_version")
        ):
            return None
        if checkpoint["producer_version"] != producer_version:
            return None

        try:
            uri = str(checkpoint["artifact_uri"])
            if not self.artifact_store.verify_sha256(uri, checkpoint["sha256"]):
                return None
            path = self.artifact_store.resolve_for_read(uri)
        except (FileNotFoundError, OSError, ValueError):
            return None

        try:
            if stage == StageName.TRANSCODE:
                return path
            payload = json.loads(path.read_text(encoding="utf-8"))
            if stage == StageName.ASR:
                if not isinstance(payload, list):
                    return None
                return [TranscriptTurn.model_validate(item) for item in payload]
            if stage == StageName.EMOTION:
                return payload if isinstance(payload, dict) else None
            if stage == StageName.QC:
                if not isinstance(payload, dict):
                    return None
                analysis = AnalysisResult.model_validate(payload.get("analysisResult"))
                if payload.get("producerVersion") != producer_version:
                    return None
                if payload.get("qcRunId") != analysis.runId:
                    return None
                if payload.get("reportId") != analysis.runId:
                    return None
                return analysis
        except (OSError, TypeError, ValueError, ValidationError):
            return None
        return None

    def begin(self, stage: StageName) -> int:
        return self.store.begin_stage(self.file_id, stage)

    def complete(
        self,
        stage: StageName,
        value: Any,
        producer_version: str,
        *,
        duration_ms: float,
    ) -> Any:
        uri = self._artifact_uri(stage)

        if stage == StageName.TRANSCODE:
            source = Path(value)
            content = source.read_bytes()
            mime_type = "audio/wav"
        elif stage == StageName.ASR:
            turns = [TranscriptTurn.model_validate(item) for item in value]
            content = json.dumps(
                [turn.model_dump(mode="json") for turn in turns],
                ensure_ascii=False,
            ).encode("utf-8")
            mime_type = "application/json"
            completed_value = turns
        elif stage == StageName.EMOTION:
            if not isinstance(value, dict):
                raise ValueError("emotion artifact must be a JSON object")
            content = json.dumps(value, ensure_ascii=False).encode("utf-8")
            mime_type = "application/json"
            completed_value = value
        elif stage == StageName.QC:
            analysis = AnalysisResult.model_validate(value)
            wrapper = {
                "qcRunId": analysis.runId,
                "reportId": analysis.runId,
                "producerVersion": producer_version,
                "analysisResult": analysis.model_dump(mode="json"),
            }
            content = json.dumps(wrapper, ensure_ascii=False).encode("utf-8")
            mime_type = "application/json"
            completed_value = analysis
        else:
            raise ValueError(f"unsupported checkpoint stage: {stage.value}")

        reference = self.artifact_store.put_bytes(
            uri,
            content,
            mime_type=mime_type,
        )
        if stage == StageName.TRANSCODE:
            completed_value = self.artifact_store.resolve_for_read(reference.uri)
        self.store.complete_stage(
            self.file_id,
            stage,
            artifact_uri=reference.uri,
            sha256=reference.sha256,
            producer_version=producer_version,
            duration_ms=duration_ms,
        )
        return completed_value

    def fail(
        self,
        stage: StageName,
        error_code: str,
        retryable: bool,
        message: str,
        *,
        duration_ms: float,
    ) -> None:
        self.store.fail_stage(
            self.file_id,
            stage,
            error_code=error_code,
            retryable=retryable,
            error=message,
            duration_ms=duration_ms,
        )

    def invalidate_downstream(self, stage: StageName) -> None:
        self.store.invalidate_stages(self.file_id, _DOWNSTREAM[stage])
