"""Execution orchestration for evaluation cases and their evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Protocol
from uuid import uuid4

from qc.eval.artifacts import EvalArtifactWriter
from qc.eval.diff import compare_eval_runs
from qc.eval.metrics import aggregate_metrics, calculate_case_metrics
from qc.eval.models import EvalCase, EvalCaseResult, EvalExecutionResult, EvalManifest, EvalRunResult, EvalSplit
from qc.eval.judge import build_judge_request, combine_deterministic_and_judge
from qc.eval.judge_models import JudgeResult
from qc.eval.judge_providers import FakeJudge
from qc.observability.tracing import current_context, traced


class CaseExecutor(Protocol):
    def execute(self, case: EvalCase, execution_mode: str) -> EvalExecutionResult: ...


class EvalRunner:
    def __init__(self, *, executor: CaseExecutor, artifact_writer: EvalArtifactWriter | None = None, judge_provider=None, eval_store=None, now: Callable[[], datetime] | None = None, id_factory: Callable[[], str] | None = None):
        self.executor = executor
        self.artifact_writer = artifact_writer or EvalArtifactWriter()
        self.judge_provider = judge_provider
        self.eval_store = eval_store
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.id_factory = id_factory or (lambda: f"EVAL-{uuid4().hex[:12].upper()}")

    def run(self, *, cases: list[EvalCase], dataset_hash: str, split: EvalSplit | str, execution_mode: str = "replay", judge_mode: str = "none", code_hash: str = "working-tree", target_model: dict | None = None, judge_model: dict | None = None, prompt_version: str | None = None, rule_version: str | None = None, knowledge_version: str | None = None, retrieval_config: dict | None = None, change_summary: str = "", changes: list[dict] | None = None, expected_impact: str = "") -> EvalRunResult:
        if execution_mode not in {"replay", "model", "e2e"}:
            raise ValueError("execution_mode must be replay/model/e2e")
        if judge_mode not in {"none", "fake", "live"}:
            raise ValueError("judge_mode must be none/fake/live")
        run_id = self.id_factory()
        manifest = EvalManifest(evalRunId=run_id, createdAt=self.now(), codeCommitOrSourceHash=code_hash, datasetHash=dataset_hash, split=EvalSplit(split), executionMode=execution_mode, judgeMode=judge_mode, targetModel=target_model or {}, judgeModel=judge_model or {}, promptVersion=prompt_version, ruleVersion=rule_version, knowledgeVersion=knowledge_version, retrievalConfig=retrieval_config or {}, changeSummary=change_summary, changes=changes or [], expectedImpact=expected_impact)
        case_results: list[EvalCaseResult] = []
        traces: list[dict] = []
        failures: list[dict] = []
        with traced(
            "evaluation_run",
            eval_run_id=run_id,
            split=EvalSplit(split).value,
            execution_mode=execution_mode,
            judge_mode=judge_mode,
            case_count=len(cases),
        ):
            for case in cases:
                try:
                    with traced("evaluation_case", eval_run_id=run_id, case_id=case.caseId):
                        executed = self.executor.execute(case, execution_mode)
                        metrics = calculate_case_metrics(case, executed.report)
                        judge_result: JudgeResult | None = None
                        judge_payload: dict
                        if judge_mode == "none":
                            judge_result = JudgeResult(status="not_run", dimension="answer_relevancy")
                            judge_payload = judge_result.model_dump(mode="json")
                        else:
                            provider = self.judge_provider or (FakeJudge() if judge_mode == "fake" else None)
                            if provider is None:
                                judge_result = JudgeResult(status="unavailable", dimension="answer_relevancy", reason="live Judge provider is not configured")
                                judge_payload = {"answerRelevancy": judge_result.model_dump(mode="json"), "faithfulness": judge_result.model_copy(update={"dimension": "faithfulness"}).model_dump(mode="json")}
                            else:
                                with traced("judge", eval_run_id=run_id, case_id=case.caseId, judge_mode=judge_mode):
                                    faithfulness = provider.judge(build_judge_request(case, executed.report, "faithfulness"))
                                    answer_relevancy = provider.judge(build_judge_request(case, executed.report, "answer_relevancy"))
                                judge_result = answer_relevancy
                                judge_payload = {"status": answer_relevancy.status, "faithfulness": faithfulness.model_dump(mode="json"), "answerRelevancy": answer_relevancy.model_dump(mode="json")}
                        deterministic_passed = metrics.status == "passed"
                        combined = combine_deterministic_and_judge(deterministic_passed, judge_result)
                        metrics.judgeResult = judge_payload
                        if judge_mode != "none":
                            metrics.rag["faithfulness"] = judge_payload["faithfulness"]
                            metrics.rag["answerRelevancy"] = judge_payload["answerRelevancy"]
                        metrics.status = "failed" if combined.status == "failed" else ("needs_review" if combined.status == "NEEDS_REVIEW" else metrics.status)
                        context = current_context()
                        result = EvalCaseResult(caseId=case.caseId, caseHash=case.caseHash, status=metrics.status, metrics=metrics, judgeResult=judge_payload, traceId=executed.traceId or context.get("traceId"), runId=executed.runId)
                        case_results.append(result)
                        raw_traces = executed.trace or [{}]
                        for raw_trace in raw_traces:
                            trace_summary = dict(raw_trace)
                            trace_summary.setdefault("evalRunId", run_id)
                            trace_summary.setdefault("caseId", case.caseId)
                            if result.traceId is not None:
                                trace_summary.setdefault("traceId", result.traceId)
                            if executed.runId is not None:
                                trace_summary.setdefault("runId", executed.runId)
                            traces.append(trace_summary)
                        if metrics.status != "passed":
                            failures.append({"caseId": case.caseId, "status": metrics.status, "failureReasons": metrics.failureReasons, "traceId": result.traceId, "runId": executed.runId})
                except Exception as exc:
                    result = EvalCaseResult(caseId=case.caseId, caseHash=case.caseHash, status="failed", metrics={"deterministic": {}, "rag": {}, "status": "failed", "failureReasons": [{"reason": "execution_error", "message": str(exc)}]}, judgeResult={"status": "unavailable"})
                    case_results.append(result)
                    failures.append({"caseId": case.caseId, "status": "failed", "failureReasons": [{"reason": "execution_error"}]})
        aggregate = aggregate_metrics([item.metrics for item in case_results])
        result = EvalRunResult(evalRunId=run_id, status="COMPLETED", manifest=manifest, caseResults=case_results, aggregateMetrics=aggregate, failures=failures)
        artifact_metrics = {**aggregate, "evalRunId": run_id, "caseResults": [{"caseId": item.caseId, "status": item.status, "judgeResult": item.judgeResult} for item in case_results]}
        artifact_dir = self.artifact_writer.write(run_id, manifest=manifest.model_dump(mode="json"), metrics=artifact_metrics, failures=failures, diff=compare_eval_runs({}, result.model_dump(mode="json")), traces=traces)
        if self.eval_store is not None:
            try:
                self.eval_store.persist(result, str(artifact_dir))
            except Exception as exc:
                raise RuntimeError(
                    "failed to persist evaluation evidence to PostgreSQL; "
                    "files were written under eval_runs but this run is not database-persisted"
                ) from exc
        return result
