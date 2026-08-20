from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
import hashlib
import os
import threading
from typing import Any, Callable

from qc.batch.emotion_worker import EmotionSubprocessClient
from qc.batch.models import FileRecord, StageName
from qc.models import TranscriptTurn


class RealAudioStageRunner:
    """Adapter from ``process_audio.py`` to the batch Runner protocol.

    Model loading is lazy so importing the batch package stays offline. A Worker
    constructs one runner and reuses it for all files.
    """

    _SUPPORTED = {".m4a", ".wav", ".mp3", ".flac"}

    def __init__(
        self,
        audio_root: str | Path,
        work_root: str | Path,
        *,
        asr_loader: Callable[[], Any] | None = None,
        emotion_loader: Callable[[], Any] | None = None,
        audio_module: Any | None = None,
        emotion_timeout_seconds: float | None = None,
    ):
        self.audio_root = Path(audio_root).resolve(strict=True)
        if not self.audio_root.is_dir():
            raise ValueError("audio_root must be a directory")
        self.work_root = Path(work_root).resolve(strict=False)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._audio_module = audio_module
        self._asr_loader = asr_loader
        self._emotion_loader = emotion_loader
        self._asr_model: Any | None = None
        self._emotion_model: Any | None = None
        self._emotion_subprocess = None
        if audio_module is None and emotion_loader is None:
            timeout = (
                emotion_timeout_seconds
                if emotion_timeout_seconds is not None
                else float(os.getenv("EMOTION_SUBPROCESS_TIMEOUT_SECONDS", "300"))
            )
            self._emotion_subprocess = EmotionSubprocessClient(
                timeout_seconds=timeout
            )
        self._asr_lock = threading.Lock()
        self._emotion_lock = threading.Lock()
        self._durations: dict[str, float] = {}

    @property
    def audio_module(self):
        if self._audio_module is None:
            import process_audio

            self._audio_module = process_audio
        return self._audio_module

    def resolve_source(self, file_record: FileRecord) -> Path:
        uri = str(file_record.source_uri)
        if "\x00" in uri or "\\" in uri:
            raise ValueError("source_uri must be a safe relative POSIX path")
        posix = PurePosixPath(uri)
        windows = PureWindowsPath(uri)
        if posix.is_absolute() or windows.is_absolute() or windows.drive:
            raise ValueError("source_uri must stay inside the audio root")
        if any(part in {"", ".", ".."} for part in posix.parts):
            raise ValueError("source_uri escapes the audio root")
        candidate = (self.audio_root / Path(*posix.parts)).resolve(strict=False)
        try:
            candidate.relative_to(self.audio_root)
        except ValueError as exc:
            raise ValueError("source_uri escapes the audio root") from exc
        if candidate.suffix.lower() not in self._SUPPORTED:
            raise ValueError("unsupported audio extension")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        expected_size = file_record.metadata.get("size")
        if expected_size is not None and candidate.stat().st_size != int(expected_size):
            raise ValueError("source file no longer matches the batch snapshot")
        expected_sha256 = file_record.metadata.get("sha256")
        if expected_sha256:
            digest = hashlib.sha256()
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest().lower() != str(expected_sha256).lower():
                raise ValueError("source file no longer matches the batch snapshot")
        return candidate

    def transcode(self, file_record: FileRecord) -> Path:
        source = self.resolve_source(file_record)
        safe_key = hashlib.sha256(
            file_record.idempotency_key.encode("utf-8")
        ).hexdigest()
        output = self.work_root / f"{safe_key}.wav"
        converted = self.audio_module.convert_m4a_to_wav(source, output)
        if isinstance(converted, tuple) and len(converted) >= 2:
            self._durations[str(output.resolve())] = float(converted[1])
        return output

    def _ensure_asr_model(self):
        if self._asr_model is None:
            with self._asr_lock:
                if self._asr_model is None:
                    loader = self._asr_loader or self.audio_module.load_asr_model
                    self._asr_model = loader()
        return self._asr_model

    def run_asr(self, wav_path: Path) -> list[TranscriptTurn]:
        model = self._ensure_asr_model()
        raw = self.audio_module.run_asr_with_model(Path(wav_path), model)
        transcript = self.audio_module.ensure_turn_ids(
            self.audio_module.parse_asr_result(raw)
        )
        transcript = [
            item
            for item in transcript
            if str(item.get("text", "")).strip()
            and str(item.get("speaker", "")).strip()
        ]
        return [TranscriptTurn.model_validate(item) for item in transcript]

    def _ensure_emotion_model(self):
        if self._emotion_model is None:
            with self._emotion_lock:
                if self._emotion_model is None:
                    loader = self._emotion_loader or self.audio_module.load_emotion_model
                    self._emotion_model = loader()
        return self._emotion_model

    def run_emotion(self, wav_path: Path) -> dict:
        if self._emotion_subprocess is not None:
            raw = self._emotion_subprocess.infer(Path(wav_path))
        else:
            model = self._ensure_emotion_model()
            raw = self.audio_module.run_emotion_with_model(Path(wav_path), model)
        duration = self._durations.get(str(Path(wav_path).resolve()), 0.0)
        result = self.audio_module.parse_emotion_result(raw, duration)
        if not isinstance(result, dict):
            raise ValueError("emotion output must be an object")
        return {
            **result,
            "role_mapping": {
                "mode": "heuristic",
                "version": "emotion-role-heuristic-v1",
                "disclaimer": (
                    "Speaker-to-agent/customer mapping is heuristic and not accurate "
                    "role identification."
                ),
            },
        }

    def warmup(self) -> None:
        """Load ASR and start the isolated emotion process before consuming work."""
        self._ensure_asr_model()
        if self._emotion_subprocess is not None:
            self._emotion_subprocess.start()
        else:
            self._ensure_emotion_model()

    def close(self) -> None:
        if self._emotion_subprocess is not None:
            self._emotion_subprocess.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def producer_version(self, stage: StageName) -> str:
        versions = {
            StageName.TRANSCODE: "real-transcode-v1",
            StageName.ASR: "real-funasr-v1",
            StageName.EMOTION: "real-emotion2vec-v1",
        }
        try:
            return versions[stage]
        except KeyError as exc:
            raise ValueError(f"unsupported audio stage: {stage.value}") from exc
