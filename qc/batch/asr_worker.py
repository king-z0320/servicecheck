from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from qc.batch.emotion_worker import (
    EmotionSubprocessClient,
    EmotionWorkerError,
    _child_send,
    _to_python,
)


def _child_main() -> int:
    """Load only FunASR and serve the shared JSON-line model protocol."""
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch

            thread_count = int(os.getenv("ASR_SUBPROCESS_NUM_THREADS", "2"))
            torch.set_num_threads(thread_count)
            torch.set_num_interop_threads(thread_count)

            import process_audio

            model = process_audio.load_asr_model()
        _child_send({"type": "ready"})

        for line in sys.stdin:
            if not line.strip():
                continue
            request = json.loads(line)
            request_type = request.get("type")
            if request_type == "stop":
                return 0
            if request_type != "infer":
                _child_send(
                    {
                        "type": "error",
                        "code": "ASR_WORKER_PROTOCOL",
                        "message": "unknown ASR worker request",
                        "retryable": False,
                    }
                )
                continue
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    raw = process_audio.run_asr_with_model(
                        Path(str(request["path"])), model
                    )
                _child_send({"type": "result", "value": _to_python(raw)})
            except Exception as exc:
                _child_send(
                    {
                        "type": "error",
                        "code": "ASR_INFERENCE_FAILED",
                        "message": str(exc) or exc.__class__.__name__,
                        "retryable": False,
                    }
                )
    except Exception as exc:
        try:
            _child_send(
                {
                    "type": "error",
                    "code": "ASR_WORKER_START_FAILED",
                    "message": str(exc) or exc.__class__.__name__,
                    "retryable": True,
                }
            )
        except OSError:
            pass
        return 1
    return 0


class AsrWorkerError(EmotionWorkerError):
    """Structured infrastructure failure from the isolated ASR process."""


class AsrSubprocessClient:
    """ASR-specific adapter over the reusable JSON-line process client."""

    def __init__(self, *, timeout_seconds: float = 300.0):
        self._client = EmotionSubprocessClient(
            timeout_seconds=timeout_seconds,
            command=[sys.executable, "-m", "qc.batch.asr_worker", "--child"],
            thread_count_env_var="ASR_SUBPROCESS_NUM_THREADS",
        )

    @property
    def process(self):
        return self._client.process

    @staticmethod
    def _remap(exc: EmotionWorkerError) -> AsrWorkerError:
        code = exc.code
        if code.startswith("EMOTION_WORKER_"):
            code = "ASR_WORKER_" + code.removeprefix("EMOTION_WORKER_")
        message = exc.message.replace("emotion 子进程", "ASR 子进程").replace(
            "emotion 推理", "ASR 推理"
        )
        return AsrWorkerError(code, message, retryable=exc.retryable)

    def start(self) -> None:
        try:
            self._client.start()
        except EmotionWorkerError as exc:
            raise self._remap(exc) from exc

    def infer(self, wav_path: str | Path) -> Any:
        try:
            return self._client.infer(wav_path)
        except EmotionWorkerError as exc:
            raise self._remap(exc) from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.close()


if __name__ == "__main__":
    raise SystemExit(_child_main())
