import hashlib
from datetime import datetime, timezone

import pytest

from process_audio import transcribe_audio
from qc.batch.models import FileRecord
from qc.batch.real_audio_runner import RealAudioStageRunner
from qc.models import AnalysisRequest, TranscriptTurn
from qc.rag import KnowledgeIndex
from tests.live_support import (
    ROOT,
    build_live_service,
    live_settings,
    running_mock_audit_server,
)


EXPECTED_HASHES = {
    "audio1.m4a": "5DB623C054EF9611B46E56CF848DC916896AC4E8BAD099EC937078BA9462F294",
    "audio2.m4a": "6E44337CD4FFE75588854504A0AA4A6634EA29B5D7066C4E52BDEC5224D2F1A6",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@pytest.mark.live_audio
def test_real_audio_runner_completes_audio_stages_without_llm(tmp_path):
    pytest.importorskip("funasr")
    source = ROOT / "audio" / "audio2.m4a"
    before = sha256(source)
    assert before == EXPECTED_HASHES[source.name]

    runner = RealAudioStageRunner(ROOT / "audio", tmp_path / "work")
    wav_path = runner.transcode(
        FileRecord(
            source_uri=source.name,
            idempotency_key=f"sha256:{before.lower()}",
            callId="CALL-REAL-RUNNER-AUDIO2",
            metadata={"sha256": before.lower(), "size": source.stat().st_size},
        )
    )
    turns = runner.run_asr(wav_path)
    emotion = runner.run_emotion(wav_path)

    assert wav_path.is_file()
    assert turns
    assert len({turn.turnId for turn in turns}) == len(turns)
    assert all(turn.text.strip() and turn.speaker.strip() for turn in turns)
    assert emotion["role_mapping"]["mode"] == "heuristic"
    assert sha256(source) == before


@pytest.mark.live_audio
@pytest.mark.parametrize("audio_name", ["audio1.m4a", "audio2.m4a"])
def test_real_funasr_audio_to_persisted_quality_report(audio_name, tmp_path):
    pytest.importorskip("funasr")
    live_settings()
    source = ROOT / "audio" / audio_name
    before = sha256(source)
    assert before == EXPECTED_HASHES[audio_name]

    derived = tmp_path / audio_name.replace(".m4a", "")
    transcript_data = transcribe_audio(source, derived)["transcript"]
    turns = [TranscriptTurn.model_validate(item) for item in transcript_data]
    assert turns
    assert len({turn.turnId for turn in turns}) == len(turns)

    knowledge = KnowledgeIndex(ROOT / "knowledge")
    knowledge.build()
    with running_mock_audit_server() as audit_url:
        system = build_live_service(
            tmp_path / f"{audio_name}.db",
            audit_url,
            knowledge_index=knowledge,
        )
        result = system.analyze(
            AnalysisRequest(
                caseId=f"LIVE-{audio_name}",
                callId=(
                    "CALL-COMPLIANT-001"
                    if audio_name == "audio1.m4a"
                    else "CALL-NONCOMPLIANT-002"
                ),
                callStartedAt=datetime(2026, 7, 27, tzinfo=timezone.utc),
                transcript=turns,
            )
        )

    assert result.status != "FAILED", [error.code for error in result.errors]
    assert result.report is not None
    if result.status == "PARTIAL":
        assert result.report.disposition.value == "HUMAN_REVIEW_REQUIRED"
    assert system.get_run(result.runId)["status"] == result.status
    assert sha256(source) == before
