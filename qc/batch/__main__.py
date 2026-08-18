"""批量质检 CLI。

用法：
  python -m qc.batch ingest <batch_id> <audio_dir>
  python -m qc.batch run <batch_id>
  python -m qc.batch resume <batch_id>
  python -m qc.batch report <batch_id>
  python -m qc.batch export <batch_id> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

from qc.batch.exporter import Exporter
from qc.batch.models import BatchConfig, BatchMeta
from qc.batch.orchestrator import BatchOrchestrator
from qc.batch.pipeline import FakeAudioStageRunner
from qc.batch.report import render_progress
from qc.batch.sources import DirectorySource
from qc.batch.postgres_store import PostgresBatchStore
from qc.database import database_url_from_env


def _quality_service():  # 真实链路需要 DeepSeek 凭证与 mock 审计服务
    import os

    from dotenv import load_dotenv

    from api_server import build_service

    load_dotenv()
    if not os.getenv("openrouter_api_key"):
        raise SystemExit("openrouter_api_key 未配置（实为 DeepSeek key）")
    return build_service()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1

    command = argv[0]
    from api_server import build_artifact_store

    store = PostgresBatchStore(database_url_from_env())
    artifacts = build_artifact_store()

    if command == "report" and len(argv) >= 2:
        print(render_progress(store, argv[1]))
        return 0

    if command == "export" and len(argv) >= 3:
        prefix = Path(argv[2]).as_posix().strip("/")
        exporter = Exporter(store, artifacts)
        json_path = exporter.export_json(
            argv[1], f"{prefix}/{argv[1]}.json"
        )
        csv_path = exporter.export_csv(
            argv[1], f"{prefix}/{argv[1]}.csv"
        )
        print(f"已导出到 {json_path} 和 {csv_path}")
        return 0

    if command == "ingest" and len(argv) >= 3:
        orch = BatchOrchestrator(store, BatchConfig(), artifacts)
        added = orch.ingest(
            argv[1], DirectorySource(argv[2]),
            BatchMeta(batch_id=argv[1], source="directory", total=0),
        )
        print(f"批次 {argv[1]} 新增 {added} 个文件")
        return 0

    if command in ("run", "resume") and len(argv) >= 2:
        quality_service = _quality_service()
        orch = BatchOrchestrator(store, BatchConfig(), artifacts)
        # 生产应注入真实常驻 AudioStageRunner；此处 Fake 仅占位，提示用户。
        runner = FakeAudioStageRunner(Path("data/wav"))
        summary = (
            orch.resume(argv[1], runner, quality_service)
            if command == "resume"
            else orch.run_batch(argv[1], runner, quality_service)
        )
        print(render_progress(store, argv[1]))
        print(summary)
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
