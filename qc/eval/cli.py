"""Stage 4 evaluation CLI.

`replay` is safe and offline. `model` explicitly builds the configured DeepSeek
service; it therefore needs the same environment and PostgreSQL settings as a
normal analysis run. `e2e` is intentionally rejected until the audio runner is
made injectable as one end-to-end evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from qc.eval.dataset import load_dataset
from qc.eval.models import EvalExecutionResult, EvalSplit
from qc.eval.runner import EvalRunner
from qc.models import QualityReport
from qc.observability.runtime import configure_local_observability


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "eval" / "datasets"


class ReplayExecutor:
    def execute(self, case, execution_mode: str) -> EvalExecutionResult:
        if execution_mode != "replay":
            raise ValueError("ReplayExecutor only supports execution=replay")
        report = case.source.get("replayReport")
        if not isinstance(report, dict):
            raise ValueError(f"case {case.caseId} has no source.replayReport")
        return EvalExecutionResult(report=QualityReport.model_validate(report), trace=[{"caseId": case.caseId, "mode": "replay"}])


class ModelServiceExecutor:
    def __init__(self):
        from api_server import build_service

        self.service = build_service()
        self.gateway = self.service.direct_analyzer.extractor.gateway
        self.target_model = {"provider": "deepseek", "model": getattr(self.gateway, "model", "deepseek-chat")}

    def execute(self, case, execution_mode: str) -> EvalExecutionResult:
        if execution_mode != "model":
            raise ValueError("ModelServiceExecutor only supports execution=model")
        result = self.service.analyze(case.input.to_request())
        if result.report is None:
            raise RuntimeError(f"target model did not produce report: {result.status}")
        return EvalExecutionResult(report=result.report, runId=result.runId, trace=[{"runId": result.runId, "caseId": case.caseId, "status": result.status}])


def _dataset_path(split: str, root: Path = DATASET_ROOT) -> Path:
    return root / f"{split}.json"


def command_validate(args) -> int:
    dataset = load_dataset(_dataset_path(args.split, Path(args.dataset_root)), expected_split=args.split)
    print(json.dumps({"split": args.split, "caseCount": len(dataset.cases), "datasetHash": dataset.dataset_hash}, ensure_ascii=False))
    return 0


def command_run(args) -> int:
    dataset = load_dataset(_dataset_path(args.split, Path(args.dataset_root)), expected_split=args.split)
    if args.execution == "e2e":
        raise RuntimeError("execution=e2e is not implemented: inject the existing audio pipeline before enabling it")
    executor = ReplayExecutor() if args.execution == "replay" else ModelServiceExecutor()
    judge_provider = None
    judge_model = {}
    if args.judge == "live":
        if not isinstance(executor, ModelServiceExecutor):
            raise RuntimeError("judge=live requires execution=model and an explicit DeepSeek target configuration")
        from qc.eval.judge_providers import DeepSeekJudge

        judge_provider = DeepSeekJudge(executor.gateway)
        judge_model = {"provider": "deepseek", "model": executor.gateway.model, "promptVersion": "judge-v1", "rubricVersion": "v1"}
    eval_store = None
    if args.persist_db:
        from qc.database import database_url_from_env
        from qc.eval.store import PostgresEvalStore

        eval_store = PostgresEvalStore(database_url_from_env())
    runner = EvalRunner(
        executor=executor,
        judge_provider=judge_provider,
        eval_store=eval_store,
    )
    result = runner.run(cases=dataset.cases, dataset_hash=dataset.dataset_hash, split=EvalSplit(args.split), execution_mode=args.execution, judge_mode=args.judge, target_model=getattr(executor, "target_model", {}), judge_model=judge_model, change_summary=args.change_summary, expected_impact=args.expected_impact)
    print(json.dumps({"evalRunId": result.evalRunId, "status": result.status, "aggregateMetrics": result.aggregateMetrics}, ensure_ascii=False))
    return 0


def command_compare(args) -> int:
    baseline = json.loads((Path(args.artifact_root) / args.baseline / "metrics.json").read_text(encoding="utf-8"))
    current = json.loads((Path(args.artifact_root) / args.current / "metrics.json").read_text(encoding="utf-8"))
    from qc.eval.diff import compare_eval_runs

    print(compare_eval_runs(baseline, current), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    configure_local_observability(ROOT, process_name="eval")
    parser = argparse.ArgumentParser(prog="python -m qc.eval")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-dataset")
    validate.add_argument("--split", choices=[item.value for item in EvalSplit], required=True)
    validate.set_defaults(handler=command_validate)
    run = sub.add_parser("run")
    run.add_argument("--split", choices=[item.value for item in EvalSplit], required=True)
    run.add_argument("--execution", choices=["replay", "model", "e2e"], required=True)
    run.add_argument("--judge", choices=["none", "fake", "live"], default="none")
    run.add_argument("--change-summary", default="")
    run.add_argument("--expected-impact", default="")
    run.add_argument(
        "--persist-db",
        action="store_true",
        help="Persist evaluation metadata, per-case metrics, and model usage to PostgreSQL.",
    )
    run.set_defaults(handler=command_run)
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--current", required=True)
    compare.add_argument("--artifact-root", default="eval_runs")
    compare.set_defaults(handler=command_compare)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
