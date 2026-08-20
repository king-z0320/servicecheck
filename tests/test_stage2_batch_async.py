from __future__ import annotations

import json
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from qc.artifact_store import LocalArtifactStore
from qc.batch.models import (
    BatchConfig,
    BatchFileStatus,
    BatchMeta,
    FileRecord,
    StageName,
)
from qc.batch.pipeline import FakeAudioStageRunner
from qc.batch.real_audio_runner import RealAudioStageRunner
from qc.batch.retry_policy import RetryPolicy, classify_error
from qc.batch.service import BatchCapacityError, InMemoryBatchService
from qc.batch.outbox_publisher import OutboxEvent, OutboxPublisher
from qc.batch.store import BatchStore
from qc.batch.worker import BatchItemExecutor, BatchMessage, BatchWorker
from qc.models import AnalysisResult, QualityReport
from api_server import create_app


def test_create_batch_scans_a_controlled_relative_directory_once(tmp_path):
    audio_root = tmp_path / "audio"
    incoming = audio_root / "incoming"
    incoming.mkdir(parents=True)
    (incoming / "a.wav").write_bytes(b"a")
    (incoming / "ignore.txt").write_text("x", encoding="utf-8")
    service = InMemoryBatchService(audio_root)

    created = service.create_batch("incoming")
    (incoming / "later.wav").write_bytes(b"later")

    assert created["batch_id"]
    assert created["total"] == 1
    assert [item["source_uri"] for item in service.list_items(created["batch_id"])] == [
        "incoming/a.wav"
    ]


def test_create_batch_keeps_distinct_paths_with_identical_audio_bytes(tmp_path):
    audio_root = tmp_path / "audio"
    incoming = audio_root / "incoming"
    incoming.mkdir(parents=True)
    (incoming / "a.wav").write_bytes(b"same")
    (incoming / "b.wav").write_bytes(b"same")

    created = InMemoryBatchService(audio_root).create_batch("incoming")

    assert created["total"] == 2


def test_create_batch_rejects_directory_escape(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    service = InMemoryBatchService(audio_root)

    with pytest.raises(ValueError, match="source_dir"):
        service.create_batch("../outside")


def test_create_batch_enforces_item_and_pending_queue_limits(tmp_path):
    audio_root = tmp_path / "audio"
    incoming = audio_root / "incoming"
    incoming.mkdir(parents=True)
    (incoming / "a.wav").write_bytes(b"a")
    (incoming / "b.wav").write_bytes(b"b")

    too_many = InMemoryBatchService(audio_root, BatchConfig(max_batch_items=1))
    with pytest.raises(BatchCapacityError, match="max_batch_items"):
        too_many.create_batch("incoming")

    limited = InMemoryBatchService(
        audio_root,
        BatchConfig(max_batch_items=2, queue_max_pending=2),
    )
    limited.create_batch("incoming")
    with pytest.raises(BatchCapacityError, match="pending queue"):
        limited.create_batch("incoming")


def test_idempotency_key_replays_same_request_and_rejects_different_request(tmp_path):
    audio_root = tmp_path / "audio"
    first_dir = audio_root / "first"
    second_dir = audio_root / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir()
    (first_dir / "a.wav").write_bytes(b"a")
    (second_dir / "b.wav").write_bytes(b"b")
    service = InMemoryBatchService(audio_root)

    first = service.create_batch("first", "request-1")
    assert service.create_batch("first", "request-1") == first
    with pytest.raises(ValueError, match="different request"):
        service.create_batch("second", "request-1")


def test_batch_api_returns_202_and_uses_batch_id(tmp_path):
    audio_root = tmp_path / "audio"
    incoming = audio_root / "incoming"
    incoming.mkdir(parents=True)
    (incoming / "a.wav").write_bytes(b"a")
    app = create_app(service=object(), batch_service=InMemoryBatchService(audio_root))

    with TestClient(app) as client:
        response = client.post(
            "/batches",
            json={"source_dir": "incoming"},
            headers={"Idempotency-Key": "request-1"},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["batch_id"]
    assert "jobId" not in body
    assert body["status"] == "QUEUED"


def test_batch_api_maps_capacity_rejection_to_429(tmp_path):
    audio_root = tmp_path / "audio"
    incoming = audio_root / "incoming"
    incoming.mkdir(parents=True)
    (incoming / "a.wav").write_bytes(b"a")
    service = InMemoryBatchService(audio_root, BatchConfig(queue_max_pending=1))
    service.create_batch("incoming")
    app = create_app(service=object(), batch_service=service)

    with TestClient(app) as client:
        response = client.post("/batches", json={"source_dir": "incoming"})

    assert response.status_code == 429
    assert response.json()["error"] == "batch capacity exceeded"


def test_real_runner_rejects_source_uri_outside_audio_root(tmp_path):
    (tmp_path / "audio").mkdir()
    runner = RealAudioStageRunner(tmp_path / "audio", tmp_path / "artifacts")
    record = FileRecord(source_uri="../secret.m4a", idempotency_key="x")

    with pytest.raises(ValueError, match="audio root"):
        runner.resolve_source(record)


def test_real_runner_rejects_source_changed_after_batch_snapshot(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    source = audio_root / "a.wav"
    source.write_bytes(b"changed")
    runner = RealAudioStageRunner(audio_root, tmp_path / "work")
    record = FileRecord(
        source_uri="a.wav",
        idempotency_key="sha256:unsafe-on-windows",
        metadata={"sha256": "0" * 64, "size": 7},
    )

    with pytest.raises(ValueError, match="snapshot"):
        runner.resolve_source(record)


def test_real_runner_uses_windows_safe_transcode_filename(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    source = audio_root / "a.wav"
    source.write_bytes(b"wav")

    class Module:
        @staticmethod
        def convert_m4a_to_wav(source, target):
            target.write_bytes(b"converted")
            return object(), 1.5

    runner = RealAudioStageRunner(audio_root, tmp_path / "work", audio_module=Module())
    output = runner.transcode(
        FileRecord(source_uri="a.wav", idempotency_key="sha256:unsafe-on-windows")
    )

    assert output.is_file()
    assert ":" not in output.name


def test_postgres_batch_service_items_raises_for_unknown_batch(tmp_path):
    class Store:
        def batch_summary(self, batch_id):
            return {"batch_id": batch_id, "total": 0, "by_status": {}}

        def list_files(self, batch_id):
            return []

    from qc.batch.service import PostgresBatchService

    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    service = PostgresBatchService(Store(), audio_root)

    with pytest.raises(KeyError):
        service.list_items("missing")


def test_real_runner_reuses_loaded_models(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    wav = audio_root / "a.wav"
    wav.write_bytes(b"wav")
    loads = {"asr": 0, "emotion": 0}

    class Module:
        @staticmethod
        def load_asr_model():
            loads["asr"] += 1
            return object()

        @staticmethod
        def load_emotion_model():
            loads["emotion"] += 1
            return object()

        @staticmethod
        def run_asr_with_model(path, model):
            return [{"text": "你好", "speaker": "说话人1", "start": 0, "end": 1}]

        @staticmethod
        def parse_asr_result(raw):
            return raw

        @staticmethod
        def ensure_turn_ids(raw):
            return [{**raw[0], "turnId": "T0001"}]

        @staticmethod
        def run_emotion_with_model(path, model):
            return []

        @staticmethod
        def parse_emotion_result(raw, duration):
            return {"agent": "neutral", "customer": "neutral"}

        @staticmethod
        def convert_m4a_to_wav(source, target):
            target.write_bytes(b"wav")

    runner = RealAudioStageRunner(audio_root, tmp_path / "work", audio_module=Module())
    runner.run_asr(wav)
    runner.run_asr(wav)
    runner.run_emotion(wav)
    runner.run_emotion(wav)
    assert loads == {"asr": 1, "emotion": 1}


def test_real_runner_uses_isolated_emotion_process_for_real_module(
    tmp_path, monkeypatch
):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    wav = audio_root / "a.wav"
    wav.write_bytes(b"wav")
    calls = []

    class FakeEmotionProcess:
        def __init__(self, *, timeout_seconds):
            calls.append(("init", timeout_seconds))

        def start(self):
            calls.append(("start",))

        def infer(self, path):
            calls.append(("infer", Path(path)))
            return [{"labels": ["neutral"], "scores": [1.0]}]

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(
        "qc.batch.real_audio_runner.EmotionSubprocessClient", FakeEmotionProcess
    )
    import process_audio

    monkeypatch.setattr(
        process_audio,
        "parse_emotion_result",
        lambda raw, duration: {"agent": "neutral", "customer": "neutral"},
    )
    runner = RealAudioStageRunner(
        audio_root,
        tmp_path / "work",
        asr_loader=lambda: object(),
    )
    runner.warmup()
    result = runner.run_emotion(wav)

    assert result["role_mapping"]["mode"] == "heuristic"
    assert calls[0][0] == "init"
    assert ("start",) in calls
    assert any(item[0] == "infer" for item in calls)
    runner.close()
    assert calls[-1] == ("close",)


def test_worker_main_closes_audio_runner_when_initialization_fails(monkeypatch):
    import qc.batch.worker as worker_module

    calls = []

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            calls.append(("init",))

        def warmup(self):
            calls.append(("warmup",))
            raise RuntimeError("warmup failed")

        def close(self):
            calls.append(("close",))

    # ``main`` imports these symbols locally, so patch their defining modules.
    monkeypatch.setattr("qc.batch.real_audio_runner.RealAudioStageRunner", FakeRunner)
    monkeypatch.setattr("qc.batch.postgres_store.PostgresBatchStore", lambda *_: object())
    monkeypatch.setattr("qc.database.database_url_from_env", lambda: "postgresql://test")
    monkeypatch.setattr("api_server.build_artifact_store", lambda: object())

    with pytest.raises(RuntimeError, match="warmup failed"):
        worker_module.main()
    assert calls == [("init",), ("warmup",), ("close",)]


def test_real_runner_emotion_artifact_discloses_heuristic_role_mapping(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    wav = audio_root / "a.wav"
    wav.write_bytes(b"wav")

    class Module:
        @staticmethod
        def load_emotion_model():
            return object()

        @staticmethod
        def run_emotion_with_model(path, model):
            return [{"labels": ["neutral"], "scores": [1.0]}]

        @staticmethod
        def parse_emotion_result(raw, duration):
            return {"agent": [], "customer": [], "duration": duration}

    runner = RealAudioStageRunner(audio_root, tmp_path / "work", audio_module=Module())
    result = runner.run_emotion(wav)

    assert result["role_mapping"]["mode"] == "heuristic"
    assert result["role_mapping"]["version"]
    assert "not accurate" in result["role_mapping"]["disclaimer"].lower()


def test_retry_policy_classifies_429_and_caps_attempts():
    policy = RetryPolicy(max_attempts=3, initial_delay=0, max_delay=0, jitter=0)

    failure = classify_error(TimeoutError("429 rate limit"), StageName.QC)
    assert failure.retryable is True
    assert failure.stage is StageName.QC
    assert policy.should_retry(failure, 2) is True
    assert policy.should_retry(failure, 3) is False


class _FakeRedis:
    def __init__(self):
        self.acked: list[str] = []

    def xack(self, stream, group, message_id):
        self.acked.append(message_id)
        return 1


class _FakeStore:
    def __init__(self):
        self.claimed = False
        self.finalized = []
        self.rows = {
            1: {
                "file_id": 1,
                "batch_id": "B-1",
                "source_uri": "a.wav",
                "idempotency_key": "idem-1",
                "call_id": "CALL-1",
                "status": "PENDING",
            }
        }

    def get_file(self, file_id):
        return self.rows[file_id]

    def claim_file(self, file_id, expected):
        if self.claimed:
            return False
        self.claimed = True
        self.rows[file_id]["status"] = "RUNNING"
        return True

    def finalize_file(self, file_id, status, request_json, result_json, failed_reason=None):
        self.finalized.append((file_id, status, request_json, result_json, failed_reason))
        self.rows[file_id]["status"] = status.value if hasattr(status, "value") else status

    def set_file_status(self, file_id, status, failed_reason=None):
        self.rows[file_id]["status"] = status.value if hasattr(status, "value") else status
        self.rows[file_id]["failed_reason"] = failed_reason


class _FakeExecutor:
    def __init__(self):
        self.calls = 0

    def __call__(self, row):
        self.calls += 1
        return type(
            "Result",
            (),
            {
                "status": BatchFileStatus.DONE,
                "request_json": "{}",
                "result_json": json.dumps({"ok": True}),
                "failed_reason": None,
            },
        )()


def test_worker_acks_only_after_business_executor_returns():
    store = _FakeStore()
    redis = _FakeRedis()
    executor = _FakeExecutor()
    worker = BatchWorker(store, redis, executor, stream="s", group="g", consumer="c")

    worker.handle_message(BatchMessage(message_id="m-1", batch_id="B-1", item_id=1))

    assert executor.calls == 1
    assert store.finalized
    assert redis.acked == ["m-1"]


def test_worker_duplicate_terminal_message_is_acked_without_execution():
    store = _FakeStore()
    store.rows[1]["status"] = "DONE"
    redis = _FakeRedis()
    executor = _FakeExecutor()
    worker = BatchWorker(store, redis, executor, stream="s", group="g", consumer="c")

    assert worker.handle_message(
        BatchMessage(message_id="m-duplicate", batch_id="B-1", item_id=1)
    ) is False
    assert executor.calls == 0
    assert redis.acked == ["m-duplicate"]


def test_worker_execution_interruption_leaves_message_pending():
    store = _FakeStore()
    redis = _FakeRedis()

    def interrupted(_row):
        raise RuntimeError("simulated worker stop")

    worker = BatchWorker(store, redis, interrupted, stream="s", group="g", consumer="c")
    assert worker.handle_message(
        BatchMessage(message_id="m-pending", batch_id="B-1", item_id=1)
    ) is False
    assert redis.acked == []
    assert store.rows[1]["status"] == "INTERRUPTED"


def test_worker_running_message_is_not_acked_as_terminal_duplicate():
    store = _FakeStore()
    store.rows[1]["status"] = "RUNNING"
    redis = _FakeRedis()
    worker = BatchWorker(store, redis, _FakeExecutor(), stream="s", group="g", consumer="c")

    assert worker.handle_message(
        BatchMessage(message_id="m-running", batch_id="B-1", item_id=1)
    ) is False
    assert redis.acked == []


def test_worker_reclaims_after_asr_checkpoint_without_repeating_asr(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(BatchMeta(batch_id="B-1", source="test", total=1))
    store.add_file(
        "B-1",
        FileRecord(source_uri="a.wav", idempotency_key="idem-1", callId="CALL-1"),
    )
    item_id = store.list_files("B-1")[0]["file_id"]
    artifact_store = LocalArtifactStore(tmp_path / "artifacts")

    class CrashAfterAsr(FakeAudioStageRunner):
        def __init__(self, wav_root):
            super().__init__(wav_root)
            self.transcode_calls = 0
            self.asr_calls = 0
            self.emotion_calls = 0

        def transcode(self, record):
            self.transcode_calls += 1
            return super().transcode(record)

        def run_asr(self, wav_path):
            self.asr_calls += 1
            return super().run_asr(wav_path)

        def run_emotion(self, wav_path):
            self.emotion_calls += 1
            if self.emotion_calls == 1:
                raise SystemExit("simulated process exit after ASR")
            return super().run_emotion(wav_path)

    class Quality:
        def analyze(self, request):
            return AnalysisResult(
                runId="RUN-1",
                status="COMPLETED",
                loopUsed=False,
                report=QualityReport(callId=request.callId),
            )

    class ReclaimRedis(_FakeRedis):
        def xautoclaim(self, stream, group, consumer, **kwargs):
            return (
                "0-0",
                [("m-reclaim", {"batch_id": "B-1", "item_id": str(item_id)})],
                [],
            )

    runner = CrashAfterAsr(tmp_path / "wav")
    config = BatchConfig(
        max_attempts=1,
        llm_rpm=600_000,
        backoff_initial=0,
        backoff_max=0,
        retry_jitter=0,
    )
    executor = BatchItemExecutor(
        store,
        artifact_store,
        runner,
        Quality(),
        config,
    )
    redis = ReclaimRedis()
    first = BatchWorker(store, redis, executor, stream="s", group="g", consumer="c")

    with pytest.raises(SystemExit, match="after ASR"):
        first.handle_message(BatchMessage("m-reclaim", "B-1", item_id))
    assert redis.acked == []
    assert store.get_file(item_id)["status"] == "RUNNING"
    assert store.get_stage_checkpoint(item_id, StageName.ASR)["status"] == "DONE"

    recovered = BatchWorker(store, redis, executor, stream="s", group="g", consumer="c")
    assert recovered.reclaim_once() == 1
    assert runner.transcode_calls == 1
    assert runner.asr_calls == 1
    assert runner.emotion_calls == 2
    assert store.get_file(item_id)["status"] == "DONE"
    assert redis.acked == ["m-reclaim"]


def test_outbox_publisher_marks_event_only_after_xadd():
    event = OutboxEvent(
        event_id="e-1",
        event_type="BATCH_ITEM_READY",
        batch_id="B-1",
        item_id=1,
        idempotency_key="idem-1",
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )

    class Store:
        def __init__(self):
            self.published = []
            self.failed = []

        def pending_outbox_events(self, limit=50):
            return [event]

        def mark_outbox_published(self, event_id, message_id):
            self.published.append((event_id, message_id))

        def mark_outbox_failed(
            self, event_id, error, *, max_attempts, retry_delay_seconds
        ):
            self.failed.append((event_id, error))
            return False

    class Redis:
        def xadd(self, stream, payload):
            assert payload["batch_id"] == "B-1"
            return "1-0"

    store = Store()
    assert OutboxPublisher(store, Redis()).publish_once() == 1
    assert store.published == [("e-1", "1-0")]
    assert store.failed == []


def test_outbox_publisher_uses_bounded_backoff_and_safe_error_summary():
    event = OutboxEvent(
        event_id="e-1",
        event_type="BATCH_ITEM_READY",
        batch_id="B-1",
        item_id=1,
        idempotency_key="idem-1",
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        attempts=1,
    )

    class Store:
        def __init__(self):
            self.failed = []

        def pending_outbox_events(self, limit=50):
            return [event]

        def mark_outbox_failed(
            self, event_id, error, *, max_attempts, retry_delay_seconds
        ):
            self.failed.append(
                (event_id, error, max_attempts, retry_delay_seconds)
            )
            return False

    class Redis:
        def xadd(self, stream, payload):
            raise ConnectionError("redis://user:secret@host")

    store = Store()
    publisher = OutboxPublisher(
        store,
        Redis(),
        max_attempts=3,
        backoff_initial=2,
        backoff_max=10,
    )

    assert publisher.publish_once() == 0
    assert store.failed == [("e-1", "ConnectionError", 3, 4)]
    assert "secret" not in str(store.failed)
