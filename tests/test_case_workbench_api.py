from __future__ import annotations

import hashlib

from qc.artifact_store import LocalArtifactStore


class FakeWorkbenchStore:
    def __init__(self, audio_uri: str, audio_bytes: bytes):
        self.audio_uri = audio_uri
        self.audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
        self.transcript = [
            {
                "turnId": "T0001",
                "speaker": "客户",
                "text": "测试",
                "start": 0,
                "end": 1,
            }
        ]

    def list_cases(self, page=1, page_size=20):
        return {
            "items": [{"caseId": "CASE-1", "latestCall": {"callId": "CALL-1"}}],
            "page": page,
            "pageSize": page_size,
            "total": 1,
        }

    def get_case(self, case_id):
        if case_id != "CASE-1":
            raise KeyError(case_id)
        return {"caseId": case_id, "calls": [{"callId": "CALL-1"}]}

    def get_call(self, call_id):
        if call_id != "CALL-1":
            raise KeyError(call_id)
        return {
            "callId": call_id,
            "caseId": "CASE-1",
            "audioAvailable": True,
            "audioArtifactUri": self.audio_uri,
            "audioSha256": self.audio_sha256,
            "audioMimeType": "audio/wav",
        }

    def get_transcript(self, call_id):
        if call_id != "CALL-1":
            raise KeyError(call_id)
        return self.transcript

    def list_runs_by_call(self, call_id):
        if call_id != "CALL-1":
            raise KeyError(call_id)
        return [{"runId": "RUN-1", "reportId": "RUN-1", "status": "COMPLETED"}]

    def get_report(self, report_id):
        if report_id != "RUN-1":
            raise KeyError(report_id)
        return {"reportId": report_id, "runId": "RUN-1", "runtimeVersion": "v1"}


class FakeQualityService:
    def get_run(self, run_id):
        return {"runId": run_id}


def make_client(app_factory, tmp_path):
    audio = b"0123456789"
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    reference = artifacts.put_bytes("audio/CALL-1/source.wav", audio)
    store = FakeWorkbenchStore(reference.uri, audio)
    client = app_factory(
        FakeQualityService(),
        run_store=store,
        artifact_store=artifacts,
    )
    return client, store, artifacts, audio


def test_workbench_query_routes_return_persisted_data(app_factory, tmp_path):
    client, _, _, _ = make_client(app_factory, tmp_path)

    listing = client.get("/api/cases?page=1&pageSize=10").json()
    assert listing["total"] == 1
    assert listing["pageSize"] == 10
    assert client.get("/api/cases/CASE-1").json()["calls"][0]["callId"] == "CALL-1"
    assert client.get("/api/calls/CALL-1").json()["audioAvailable"] is True
    assert client.get("/api/calls/CALL-1/transcript").json()[0]["turnId"] == "T0001"
    assert client.get("/api/calls/CALL-1/runs").json()[0]["runId"] == "RUN-1"
    assert client.get("/api/reports/RUN-1").json()["runtimeVersion"] == "v1"


def test_workbench_query_routes_return_404_for_unknown_ids(app_factory, tmp_path):
    client, _, _, _ = make_client(app_factory, tmp_path)

    for path in (
        "/api/cases/MISSING",
        "/api/calls/MISSING",
        "/api/calls/MISSING/transcript",
        "/api/calls/MISSING/runs",
        "/api/reports/MISSING",
    ):
        response = client.get(path)
        assert response.status_code == 404


def test_audio_endpoint_supports_full_and_single_range_reads(app_factory, tmp_path):
    client, _, _, audio = make_client(app_factory, tmp_path)

    full = client.get("/api/calls/CALL-1/audio")
    assert full.status_code == 200
    assert full.content == audio
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-length"] == str(len(audio))

    partial = client.get(
        "/api/calls/CALL-1/audio",
        headers={"Range": "bytes=2-5"},
    )
    assert partial.status_code == 206
    assert partial.content == b"2345"
    assert partial.headers["content-range"] == "bytes 2-5/10"
    assert partial.headers["content-length"] == "4"


def test_audio_endpoint_rejects_unsatisfiable_or_multiple_ranges(app_factory, tmp_path):
    client, _, _, _ = make_client(app_factory, tmp_path)

    for value in ("bytes=100-200", "bytes=0-1,4-5", "items=0-1"):
        response = client.get(
            "/api/calls/CALL-1/audio",
            headers={"Range": value},
        )
        assert response.status_code == 416
        assert response.headers["content-range"] == "bytes */10"


def test_audio_endpoint_fails_closed_when_artifact_hash_changed(app_factory, tmp_path):
    client, store, artifacts, _ = make_client(app_factory, tmp_path)
    artifacts.resolve_for_read(store.audio_uri).write_bytes(b"tampered")

    response = client.get("/api/calls/CALL-1/audio")

    assert response.status_code == 409
    assert response.json()["error"] == "artifact integrity check failed"


def test_audio_endpoint_returns_404_when_registered_artifact_is_missing(
    app_factory,
    tmp_path,
):
    client, store, artifacts, _ = make_client(app_factory, tmp_path)
    artifacts.resolve_for_read(store.audio_uri).unlink()

    response = client.get("/api/calls/CALL-1/audio")

    assert response.status_code == 404
    assert response.json()["error"] == "audio not found"


def test_transcript_endpoint_rejects_corrupt_persisted_turns(app_factory, tmp_path):
    client, store, _, _ = make_client(app_factory, tmp_path)
    store.transcript = [
        {"turnId": "T0001", "speaker": "客户", "text": " ", "start": 0, "end": 1}
    ]

    response = client.get("/api/calls/CALL-1/transcript")

    assert response.status_code == 409
    assert response.json()["error"] == "transcript integrity check failed"


def test_openapi_contains_all_stage_one_query_routes(app_factory, tmp_path):
    client, _, _, _ = make_client(app_factory, tmp_path)
    paths = client.get("/openapi.json").json()["paths"]

    assert {
        "/api/cases",
        "/api/cases/{case_id}",
        "/api/calls/{call_id}",
        "/api/calls/{call_id}/audio",
        "/api/calls/{call_id}/transcript",
        "/api/calls/{call_id}/runs",
        "/api/reports/{report_id}",
    }.issubset(paths)
