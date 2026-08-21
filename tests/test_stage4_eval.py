from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from qc.eval.artifacts import EvalArtifactWriter
from qc.eval.dataset import DatasetValidationError, load_dataset
from qc.eval.metrics import calculate_case_metrics
from qc.eval.models import (
    EvalCase,
    EvalExecutionResult,
    EvalSplit,
    ExpectedEval,
    ExpectedEvent,
)
from qc.eval.runner import EvalRunner
from qc.eval.judge import combine_deterministic_and_judge
from qc.eval.judge_models import JudgeResult
from qc.eval.judge_providers import FakeJudge
from qc.models import (
    KnowledgeHit,
    QualityEvent,
    QualityReport,
    ReviewDisposition,
    TranscriptTurn,
)


def _case(*, case_id: str = "CASE-1") -> EvalCase:
    return EvalCase(
        caseId=case_id,
        split=EvalSplit.DEV,
        source={"kind": "manual", "reference": "unit-test"},
        labelNotes="人工复核过事件、规则与知识证据。",
        input={
            "caseId": case_id,
            "callId": "CALL-1",
            "callStartedAt": "2026-08-01T00:00:00Z",
            "transcript": [
                {"turnId": "T1", "speaker": "CUSTOMER", "text": "我要投诉", "start": 0, "end": 1},
                {"turnId": "T2", "speaker": "AGENT", "text": "投诉也没用", "start": 1, "end": 2},
            ],
        },
        expected=ExpectedEval(
            events=[ExpectedEvent(eventType="COMPLAINT_INTENT", requiredTurnIds=["T1"])],
            ruleIds=["R009"],
            allowedDispositions=["AUTO_VIOLATION"],
            requiredContextIds=["DOC-1"],
            relevantContextIds=["DOC-1"],
            forbiddenContextIds=["DOC-BAD"],
            referenceAnswerPoints=["坐席不得阻拦投诉"],
        ),
    )


def _report(*, include_event: bool = True, hit_id: str = "DOC-1") -> QualityReport:
    events = []
    if include_event:
        events = [
            QualityEvent(
                eventId="EVT-00000000000000000000000000000001",
                type="COMPLAINT_INTENT",
                statement="客户表示投诉",
                turnIds=["T1"],
                confidence=0.9,
            )
        ]
    return QualityReport(
        callId="CALL-1",
        events=events,
        knowledgeHits=[
            KnowledgeHit(
                documentId=hit_id,
                category="POLICY",
                title="投诉规则",
                content="脱敏内容",
                version="v1",
                score=0.9,
                metadata={"contentHash": "hash-1", "eventType": "COMPLAINT_INTENT"},
            )
        ],
        disposition=ReviewDisposition.AUTO_VIOLATION,
        violations=[],
    )


def test_dataset_loader_rejects_duplicate_ids_and_cross_split(tmp_path):
    payload = [_case().model_dump(mode="json"), _case().model_dump(mode="json")]
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="duplicate caseId"):
        load_dataset(path, expected_split=EvalSplit.DEV)

    payload[1]["caseId"] = "CASE-2"
    payload[1]["input"]["caseId"] = "CASE-2"
    payload[1]["split"] = "challenge"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="does not match"):
        load_dataset(path, expected_split=EvalSplit.DEV)


def test_dataset_hash_is_stable_and_case_validates_turn_references(tmp_path):
    path = tmp_path / "dev.json"
    path.write_text(json.dumps([_case().model_dump(mode="json")], ensure_ascii=False), encoding="utf-8")
    first = load_dataset(path, expected_split=EvalSplit.DEV)
    second = load_dataset(path, expected_split=EvalSplit.DEV)
    assert first.dataset_hash == second.dataset_hash
    assert first.cases[0].caseHash == second.cases[0].caseHash

    invalid = _case().model_dump(mode="json")
    invalid["expected"]["events"][0]["requiredTurnIds"] = ["MISSING"]
    with pytest.raises(ValueError, match="requiredTurnIds"):
        EvalCase.model_validate(invalid)


def test_deterministic_metrics_expose_event_rule_disposition_and_rag_failures():
    metrics = calculate_case_metrics(_case(), _report(include_event=False, hit_id="DOC-BAD"))

    assert metrics.deterministic["events"]["recall"] == 0.0
    assert metrics.deterministic["rules"]["passed"] is False
    assert metrics.deterministic["disposition"]["passed"] is True
    assert metrics.rag["contextRecall"]["value"] == 0.0
    assert metrics.rag["faithfulnessHard"]["passed"] is False


def test_judge_cannot_override_hard_failure_and_low_judge_requires_review():
    judge = FakeJudge(score=4).judge_for(
        evidence_ids={"DOC-1"}, reference_points=["point"]
    )
    assert combine_deterministic_and_judge(False, judge).status == "failed"

    low = JudgeResult(
        status="completed",
        dimension="answer_relevancy",
        score=1,
        reason="偏离任务",
        evidenceIds=[],
        confidence=0.8,
        provider="fake",
        model="fake-judge",
        promptVersion="test-v1",
        rubricVersion="v1",
        invocationId="JUDGE-1",
        tokenSource="unknown",
    )
    assert combine_deterministic_and_judge(True, low).status == "NEEDS_REVIEW"


def test_runner_writes_sanitized_artifacts_and_not_run_judge(tmp_path):
    case = _case()

    class Executor:
        def execute(self, value, execution_mode):
            assert execution_mode == "replay"
            return EvalExecutionResult(report=_report(), runId="RUN-1", traceId="trace-1")

    runner = EvalRunner(
        executor=Executor(),
        artifact_writer=EvalArtifactWriter(tmp_path),
        now=lambda: datetime(2026, 8, 1, tzinfo=timezone.utc),
        id_factory=lambda: "EVAL-1",
    )
    result = runner.run(
        cases=[case],
        dataset_hash="dataset-hash",
        split=EvalSplit.DEV,
        execution_mode="replay",
        judge_mode="none",
        change_summary="增加评测闭环",
        changes=[{"category": "EVAL", "target": "qc/eval/runner.py", "description": "test"}],
        expected_impact="可重复运行",
    )

    artifact_dir = tmp_path / "EVAL-1"
    assert result.status == "COMPLETED"
    assert result.caseResults[0].judgeResult["status"] == "not_run"
    assert {item.name for item in artifact_dir.iterdir()} >= {
        "manifest.json", "metrics.json", "failures.json", "diff.md", "traces.jsonl"
    }
    trace_text = (artifact_dir / "traces.jsonl").read_text(encoding="utf-8")
    assert "我要投诉" not in trace_text
    assert '"traceId": "trace-1"' in trace_text
    assert '"evalRunId": "EVAL-1"' in trace_text


def test_runner_passes_eval_run_id_to_persistence_store(tmp_path):
    case = _case()

    class Executor:
        def execute(self, value, execution_mode):
            return EvalExecutionResult(report=_report(), runId="RUN-1")

    class Store:
        def __init__(self):
            self.result = None
            self.artifact_uri = None

        def persist(self, result, artifact_uri):
            self.result = result
            self.artifact_uri = artifact_uri

    store = Store()
    result = EvalRunner(
        executor=Executor(),
        artifact_writer=EvalArtifactWriter(tmp_path),
        eval_store=store,
        id_factory=lambda: "EVAL-PERSIST-1",
    ).run(
        cases=[case],
        dataset_hash="dataset-hash",
        split=EvalSplit.DEV,
    )

    assert store.result.evalRunId == result.evalRunId == "EVAL-PERSIST-1"
    assert store.artifact_uri.endswith("EVAL-PERSIST-1")


@pytest.mark.parametrize("split", [EvalSplit.DEV, EvalSplit.REGRESSION, EvalSplit.CHALLENGE])
def test_stage4_datasets_have_realistic_case_volume_and_dialogue_length(split):
    from qc.eval.dataset import load_dataset

    dataset = load_dataset(f"eval/datasets/{split.value}.json", expected_split=split)
    # 5 条、8 轮只够验证数据格式，不足以暴露长上下文中的事件定位问题。
    # 阶段 4 的合成基线要至少覆盖 8 个独立场景，并让每条都具备完整
    # 的开场、信息核对、核心事件、追问/澄清和收尾语境。
    assert len(dataset.cases) >= 8
    assert min(len(case.input.transcript) for case in dataset.cases) >= 12
    assert min(sum(len(turn.text) for turn in case.input.transcript) for case in dataset.cases) >= 180
