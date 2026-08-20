from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any


LOGGER = logging.getLogger(__name__)
_RESPONSE_TYPES = {"ready", "result", "error"}


@dataclass(frozen=True, slots=True)
class _ResponseRead:
    payload: dict[str, Any] | None = None
    eof: bool = False
    reader_error: BaseException | None = None
    discarded_lines: int = 0


class EmotionWorkerError(RuntimeError):
    """Structured failure raised by the isolated emotion process."""

    def __init__(self, code: str, message: str, *, retryable: bool = True):
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(f"{code}: {message}")


def _to_python(value: Any) -> Any:
    """Convert common NumPy/Torch values to JSON-safe Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_python(value.tolist())
    if hasattr(value, "item"):
        return _to_python(value.item())
    raise TypeError(f"emotion result contains unsupported value: {type(value)!r}")


def _child_send(payload: dict[str, Any]) -> None:
    # Keep the pipe ASCII-only.  On Windows the child may inherit a legacy
    # console codec even though the parent explicitly opens the pipe as UTF-8;
    # JSON escapes preserve non-ASCII paths and labels without mojibake.
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _child_main() -> int:
    """Load only emotion2vec and serve JSON-line requests on stdin/stdout."""
    import contextlib

    try:
        # Keep the protocol on stdout. Model progress and diagnostics remain
        # visible on the parent's stderr without corrupting JSON responses.
        with contextlib.redirect_stdout(sys.stderr):
            import torch

            thread_count = int(
                os.getenv("EMOTION_SUBPROCESS_NUM_THREADS", "1")
            )
            torch.set_num_threads(thread_count)
            torch.set_num_interop_threads(thread_count)

            import process_audio

            model = process_audio.load_emotion_model()
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
                        "code": "EMOTION_WORKER_PROTOCOL",
                        "message": "unknown emotion worker request",
                        "retryable": False,
                    }
                )
                continue
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    raw = process_audio.run_emotion_with_model(
                        Path(str(request["path"])), model
                    )
                _child_send({"type": "result", "value": _to_python(raw)})
            except Exception as exc:
                _child_send(
                    {
                        "type": "error",
                        "code": "EMOTION_INFERENCE_FAILED",
                        "message": str(exc) or exc.__class__.__name__,
                        "retryable": False,
                    }
                )
    except Exception as exc:
        try:
            _child_send(
                {
                    "type": "error",
                    "code": "EMOTION_WORKER_START_FAILED",
                    "message": str(exc) or exc.__class__.__name__,
                    "retryable": True,
                }
            )
        except OSError:
            pass
        return 1
    return 0


class EmotionSubprocessClient:
    """A reusable emotion2vec process started with the current interpreter."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 300.0,
        startup_timeout_seconds: float | None = None,
        command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        thread_count_env_var: str = "EMOTION_SUBPROCESS_NUM_THREADS",
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.startup_timeout_seconds = float(
            startup_timeout_seconds
            if startup_timeout_seconds is not None
            else max(self.timeout_seconds, 300.0)
        )
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")

        project_root = Path(__file__).resolve().parents[2]
        self.command = list(
            command
            or [sys.executable, "-m", "qc.batch.emotion_worker", "--child"]
        )
        self.cwd = str(cwd or project_root)
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = os.pathsep.join(
            [str(project_root), child_env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        if env:
            child_env.update({str(key): str(value) for key, value in env.items()})
        if not thread_count_env_var.strip():
            raise ValueError("thread_count_env_var must not be empty")
        thread_count_raw = child_env.get(thread_count_env_var, "1")
        try:
            thread_count = int(thread_count_raw)
        except ValueError as exc:
            raise ValueError(
                f"{thread_count_env_var} must be a positive integer"
            ) from exc
        if thread_count <= 0:
            raise ValueError(
                f"{thread_count_env_var} must be a positive integer"
            )
        normalized_thread_count = str(thread_count)
        child_env[thread_count_env_var] = normalized_thread_count
        # emotion2vec pulls in native OpenMP/BLAS runtimes through Torch,
        # NumPy and scikit-learn. Keep the isolated process deliberately small
        # and deterministic instead of inheriting machine-wide thread counts.
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            child_env[variable] = normalized_thread_count
        self.env = child_env

        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | BaseException | None] | None = None
        self._reader_thread: threading.Thread | None = None

    @property
    def process(self) -> subprocess.Popen[str] | None:
        return self._process

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._discard_process()
        try:
            process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # Model logging goes to the parent terminal; stdout is JSON only.
                stderr=None,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise EmotionWorkerError(
                "EMOTION_WORKER_START_FAILED", "emotion 子进程无法启动"
            ) from exc

        self._process = process
        self._start_reader(process)
        read = self._read_response(self.startup_timeout_seconds)
        self._log_discarded_lines(read, phase="startup")
        response = read.payload
        if response is None:
            if self._has_exited(process):
                code = "EMOTION_WORKER_START_FAILED"
                message = "emotion 子进程启动时崩溃"
            elif read.eof or read.reader_error is not None:
                code = "EMOTION_WORKER_PROTOCOL"
                message = "emotion 子进程协议管道异常"
            else:
                code = "EMOTION_WORKER_START_TIMEOUT"
                message = "emotion 子进程启动超时"
            self._discard_process()
            raise EmotionWorkerError(code, message)
        if response.get("type") != "ready":
            is_error = response.get("type") == "error"
            code = str(
                response.get("code")
                or (
                    "EMOTION_WORKER_START_FAILED"
                    if is_error
                    else "EMOTION_WORKER_PROTOCOL"
                )
            )
            message = str(
                response.get("message")
                or ("emotion 子进程启动失败" if is_error else "emotion 子进程协议异常")
            )
            retryable = bool(response.get("retryable", True)) if is_error else True
            self._discard_process()
            raise EmotionWorkerError(code, message, retryable=retryable)

    def infer(self, wav_path: str | Path) -> Any:
        self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise EmotionWorkerError("EMOTION_WORKER_CRASHED", "emotion 子进程不可用")
        try:
            process.stdin.write(
                json.dumps(
                    {"type": "infer", "path": str(wav_path)}, ensure_ascii=True
                )
                + "\n"
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._discard_process()
            raise EmotionWorkerError(
                "EMOTION_WORKER_CRASHED", "emotion 子进程已退出，未返回结果"
            ) from exc

        read = self._read_response(self.timeout_seconds)
        self._log_discarded_lines(read, phase="inference")
        response = read.payload
        if response is None:
            if self._has_exited(process):
                code = "EMOTION_WORKER_CRASHED"
                message = "emotion 子进程已退出，未返回结果"
            elif read.eof or read.reader_error is not None:
                code = "EMOTION_WORKER_PROTOCOL"
                message = "emotion 子进程协议管道异常"
            else:
                code = "EMOTION_WORKER_TIMEOUT"
                message = "emotion 推理超过时间限制"
            self._discard_process()
            raise EmotionWorkerError(code, message)
        if response.get("type") == "result":
            return response.get("value")

        if response.get("type") != "error":
            self._discard_process()
            raise EmotionWorkerError(
                "EMOTION_WORKER_PROTOCOL", "emotion 子进程返回了非法协议消息"
            )

        code = str(response.get("code") or "EMOTION_WORKER_FAILED")
        message = str(response.get("message") or "emotion 子进程执行失败")
        raise EmotionWorkerError(
            code,
            message,
            retryable=bool(response.get("retryable", False)),
        )

    def _start_reader(self, process: subprocess.Popen[str]) -> None:
        responses: queue.Queue[str | BaseException | None] = queue.Queue()
        self._responses = responses

        def read_output() -> None:
            try:
                if process.stdout is None:
                    responses.put(None)
                    return
                for line in process.stdout:
                    responses.put(line)
            except BaseException as exc:
                responses.put(exc)
            finally:
                responses.put(None)

        self._reader_thread = threading.Thread(target=read_output, daemon=True)
        self._reader_thread.start()

    def _read_response(self, timeout_seconds: float) -> _ResponseRead:
        if self._responses is None:
            return _ResponseRead(eof=True)
        deadline = time.monotonic() + timeout_seconds
        discarded_lines = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _ResponseRead(discarded_lines=discarded_lines)
            try:
                item = self._responses.get(timeout=remaining)
            except queue.Empty:
                return _ResponseRead(discarded_lines=discarded_lines)
            if item is None:
                return _ResponseRead(eof=True, discarded_lines=discarded_lines)
            if isinstance(item, BaseException):
                return _ResponseRead(
                    reader_error=item,
                    discarded_lines=discarded_lines,
                )
            try:
                payload = json.loads(item)
            except (TypeError, ValueError):
                discarded_lines += 1
                continue
            if (
                isinstance(payload, dict)
                and payload.get("type") in _RESPONSE_TYPES
            ):
                return _ResponseRead(
                    payload=payload,
                    discarded_lines=discarded_lines,
                )
            discarded_lines += 1

    @staticmethod
    def _log_discarded_lines(read: _ResponseRead, *, phase: str) -> None:
        if read.discarded_lines:
            LOGGER.warning(
                "ignored non-protocol emotion worker stdout",
                extra={
                    "phase": phase,
                    "discarded_lines": read.discarded_lines,
                },
            )

    @staticmethod
    def _has_exited(
        process: subprocess.Popen[str], *, grace_seconds: float = 0.2
    ) -> bool:
        """Confirm EOF/crash without misclassifying a poll update race as timeout."""
        if process.poll() is not None:
            return True
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            return False
        return True

    @staticmethod
    def _terminate_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
            except (OSError, subprocess.TimeoutExpired):
                # taskkill may block while Windows process-tree discovery is
                # unhealthy. Do not let child cleanup freeze the Batch Worker.
                process.kill()
        else:
            process.terminate()

    def _discard_process(self) -> None:
        process = self._process
        self._process = None
        self._responses = None
        self._reader_thread = None
        if process is None:
            return
        self._terminate_tree(process)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                stream.close()

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(json.dumps({"type": "stop"}, ensure_ascii=True) + "\n")
                process.stdin.flush()
                process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                pass
        self._discard_process()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.close()


if __name__ == "__main__":
    raise SystemExit(_child_main())
