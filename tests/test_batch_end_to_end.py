from pathlib import Path

import csv

from qc.batch.exporter import Exporter
from qc.batch.models import BatchConfig, BatchMeta
from qc.batch.orchestrator import BatchOrchestrator
from qc.batch.pipeline import FakeAudioStageRunner
from qc.batch.report import render_progress
from qc.batch.sources import DirectorySource
from qc.batch.store import BatchStore
from qc.models import AnalysisResult, QualityReport


class FakeQualityService:
    def analyze(self, request):
        return AnalysisResult(
            runId="RUN-X",
            status="COMPLETED",
            loopUsed=False,
            report=QualityReport(callId=request.callId),
        )


def test_small_batch_end_to_end(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for name in ("a.m4a", "b.m4a", "c.m4a"):
        (audio_dir / name).write_bytes(b"x")

    store = BatchStore(tmp_path / "batch.db")
    orch = BatchOrchestrator(store, BatchConfig())

    # 摄取
    orch.ingest("B-1", DirectorySource(audio_dir), BatchMeta(batch_id="B-1", source="directory", total=0))
    assert len(store.list_files("B-1")) == 3

    # 处理（Fake ASR + Fake 质检，验证编排与状态流转）
    summary = orch.run_batch("B-1", FakeAudioStageRunner(tmp_path / "wav"), FakeQualityService())
    assert summary["by_status"].get("DONE") == 3

    # 导出
    out_dir = tmp_path / "out"
    Exporter(store).export_json("B-1", out_dir / "B-1.json")
    Exporter(store).export_csv("B-1", out_dir / "B-1.csv")
    rows = list(csv.DictReader((out_dir / "B-1.csv").read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 3
    assert all(r["status"] == "DONE" for r in rows)

    # 报表
    text = render_progress(store, "B-1")
    assert "DONE" in text and "B-1" in text


def test_resume_after_partial_run(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"x")
    (audio_dir / "b.m4a").write_bytes(b"y")
    store = BatchStore(tmp_path / "batch.db")
    orch = BatchOrchestrator(store, BatchConfig())
    orch.ingest("B-1", DirectorySource(audio_dir), BatchMeta(batch_id="B-1", source="directory", total=0))
    # 模拟跑了一个就中断
    store.set_file_status(store.list_files("B-1")[0]["file_id"], "RUNNING")
    store.mark_interrupted_running()
    summary = orch.resume("B-1", FakeAudioStageRunner(tmp_path / "wav"), FakeQualityService())
    assert summary["by_status"].get("DONE") == 2
