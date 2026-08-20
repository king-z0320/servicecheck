from __future__ import annotations

import sys
import time
import subprocess

import pytest

from qc.batch.emotion_worker import EmotionSubprocessClient, EmotionWorkerError


FAKE_WORKER_CODE = r"""
import json
import os
import sys
import time

print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("type") == "stop":
        break
    path = str(request["path"])
    if path == "crash":
        os._exit(17)
    if path == "slow":
        time.sleep(0.25)
    print(json.dumps({"type": "result", "value": {"path": path}}), flush=True)
"""


NOISY_WORKER_CODE = r"""
import json
import sys

print("native model startup progress", flush=True)
print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("type") == "stop":
        break
    print("native inference progress: 50%", flush=True)
    print(
        json.dumps(
            {"type": "result", "value": {"path": str(request["path"])}}
        ),
        flush=True,
    )
"""


THREAD_ENV_WORKER_CODE = r"""
import json
import os
import sys

keys = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]
print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("type") == "stop":
        break
    print(
        json.dumps(
            {
                "type": "result",
                "value": {key: os.environ.get(key) for key in keys},
            }
        ),
        flush=True,
    )
"""


def make_client(*, timeout_seconds=1):
    return EmotionSubprocessClient(
        timeout_seconds=timeout_seconds,
        startup_timeout_seconds=2,
        command=[sys.executable, "-u", "-c", FAKE_WORKER_CODE],
    )


def test_emotion_subprocess_reuses_one_process(tmp_path):
    client = make_client()

    try:
        first = client.infer(tmp_path / "first.wav")
        process = client.process
        second = client.infer(tmp_path / "second.wav")

        assert first == {"path": str(tmp_path / "first.wav")}
        assert second == {"path": str(tmp_path / "second.wav")}
        assert client.process is process
        assert process.poll() is None
    finally:
        client.close()


def test_emotion_subprocess_restarts_lazily_after_child_crash(tmp_path):
    client = make_client()

    try:
        client.start()
        first_process = client.process
        with pytest.raises(EmotionWorkerError) as error:
            client.infer("crash")

        assert error.value.code == "EMOTION_WORKER_CRASHED"
        assert error.value.retryable is True
        assert client.process is None
        assert client.infer(tmp_path / "after-crash.wav")["path"].endswith(
            "after-crash.wav"
        )
        assert client.process is not first_process
    finally:
        client.close()


def test_emotion_subprocess_restarts_lazily_after_timeout(tmp_path):
    client = make_client(timeout_seconds=0.05)

    try:
        client.start()
        first_process = client.process
        with pytest.raises(EmotionWorkerError) as error:
            client.infer("slow")

        assert error.value.code == "EMOTION_WORKER_TIMEOUT"
        assert error.value.retryable is True
        assert client.process is None
        assert client.infer(tmp_path / "after-timeout.wav")["path"].endswith(
            "after-timeout.wav"
        )
        assert client.process is not first_process
    finally:
        client.close()


def test_emotion_subprocess_reports_start_crash_without_waiting_full_timeout():
    client = EmotionSubprocessClient(
        timeout_seconds=30,
        startup_timeout_seconds=2,
        command=[sys.executable, "-c", "import os; os._exit(19)"],
    )

    started = time.monotonic()
    with pytest.raises(EmotionWorkerError) as error:
        client.start()
    elapsed = time.monotonic() - started

    assert error.value.code == "EMOTION_WORKER_START_FAILED"
    assert elapsed < 2
    client.close()


def test_emotion_subprocess_ignores_non_protocol_stdout_lines(tmp_path):
    client = EmotionSubprocessClient(
        timeout_seconds=1,
        startup_timeout_seconds=1,
        command=[sys.executable, "-u", "-c", NOISY_WORKER_CODE],
    )

    try:
        wav = tmp_path / "中文录音.wav"
        assert client.infer(wav) == {"path": str(wav)}
        assert client.process is not None
        assert client.process.poll() is None
    finally:
        client.close()


def test_emotion_subprocess_reports_closed_protocol_pipe_as_protocol_error():
    client = EmotionSubprocessClient(
        timeout_seconds=5,
        startup_timeout_seconds=5,
        command=[
            sys.executable,
            "-c",
            "import os, sys, time; os.close(sys.stdout.fileno()); time.sleep(5)",
        ],
    )

    started = time.monotonic()
    with pytest.raises(EmotionWorkerError) as error:
        client.start()
    elapsed = time.monotonic() - started

    assert error.value.code == "EMOTION_WORKER_PROTOCOL"
    assert error.value.retryable is True
    assert elapsed < 2
    client.close()


def test_emotion_subprocess_limits_native_threads_in_child():
    client = EmotionSubprocessClient(
        timeout_seconds=1,
        startup_timeout_seconds=1,
        command=[sys.executable, "-u", "-c", THREAD_ENV_WORKER_CODE],
        env={
            "EMOTION_SUBPROCESS_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "99",
            "MKL_NUM_THREADS": "99",
            "OPENBLAS_NUM_THREADS": "99",
            "NUMEXPR_NUM_THREADS": "99",
        },
    )

    try:
        assert client.infer("audio.wav") == {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    finally:
        client.close()


def test_model_subprocess_can_use_a_stage_specific_thread_setting():
    client = EmotionSubprocessClient(
        timeout_seconds=1,
        startup_timeout_seconds=1,
        command=[sys.executable, "-u", "-c", THREAD_ENV_WORKER_CODE],
        thread_count_env_var="ASR_SUBPROCESS_NUM_THREADS",
        env={
            "ASR_SUBPROCESS_NUM_THREADS": "3",
            "OMP_NUM_THREADS": "99",
            "MKL_NUM_THREADS": "99",
            "OPENBLAS_NUM_THREADS": "99",
            "NUMEXPR_NUM_THREADS": "99",
        },
    )

    try:
        assert client.infer("audio.wav") == {
            "OMP_NUM_THREADS": "3",
            "MKL_NUM_THREADS": "3",
            "OPENBLAS_NUM_THREADS": "3",
            "NUMEXPR_NUM_THREADS": "3",
        }
    finally:
        client.close()


def test_emotion_subprocess_kills_child_if_windows_tree_cleanup_hangs(monkeypatch):
    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    process = FakeProcess()
    monkeypatch.setattr(subprocess, "run", timeout)

    EmotionSubprocessClient._terminate_tree(process)

    assert process.killed is True
