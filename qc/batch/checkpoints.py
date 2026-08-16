from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

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
    ):
        self.store = store
        self.batch_id = batch_id
        self.file_id = file_id
        root = (
            Path(artifact_root)
            if artifact_root is not None
            else Path(store.path).resolve().parent / "batch_artifacts"
        )
        self.artifact_root = root.resolve()
        self.file_root = (self.artifact_root / batch_id / str(file_id)).resolve()
        try:
            self.file_root.relative_to(self.artifact_root)
        except ValueError as exc:
            raise ValueError("batch artifact path escapes artifact root") from exc

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _artifact_path(self, stage: StageName) -> Path:
        try:
            name = _ARTIFACT_NAMES[stage]
        except KeyError as exc:
            raise ValueError(f"unsupported checkpoint stage: {stage.value}") from exc
        return self.file_root / name

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

        path = Path(checkpoint["artifact_uri"])
        try:
            path.resolve().relative_to(self.artifact_root)
        except (OSError, ValueError):
            return None
        if not path.is_file() or self._sha256(path) != checkpoint["sha256"]:
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
        path = self._artifact_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")

        if stage == StageName.TRANSCODE:
            source = Path(value)
            temporary.write_bytes(source.read_bytes())
            completed_value: Any = path
        elif stage == StageName.ASR:
            turns = [TranscriptTurn.model_validate(item) for item in value]
            temporary.write_text(
                json.dumps(
                    [turn.model_dump(mode="json") for turn in turns],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed_value = turns
        elif stage == StageName.EMOTION:
            if not isinstance(value, dict):
                raise ValueError("emotion artifact must be a JSON object")
            temporary.write_text(
                json.dumps(value, ensure_ascii=False), encoding="utf-8"
            )
            completed_value = value
        elif stage == StageName.QC:
            analysis = AnalysisResult.model_validate(value)
            wrapper = {
                "qcRunId": analysis.runId,
                "reportId": analysis.runId,
                "producerVersion": producer_version,
                "analysisResult": analysis.model_dump(mode="json"),
            }
            temporary.write_text(
                json.dumps(wrapper, ensure_ascii=False), encoding="utf-8"
            )
            completed_value = analysis
        else:
            raise ValueError(f"unsupported checkpoint stage: {stage.value}")

        temporary.replace(path)
        digest = self._sha256(path)
        self.store.complete_stage(
            self.file_id,
            stage,
            artifact_uri=path,
            sha256=digest,
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
