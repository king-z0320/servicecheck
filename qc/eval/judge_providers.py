"""Fake and DeepSeek implementations of the evaluation Judge protocol."""

from __future__ import annotations

from uuid import uuid4

from qc.eval.judge_models import JudgeRequest, JudgeResult


class FakeJudge:
    def __init__(self, score: int = 4, *, status: str = "completed", reason: str = "fake rubric"):
        self.score = score
        self.status = status
        self.reason = reason

    def judge(self, request: JudgeRequest) -> JudgeResult:
        return self.judge_for(
            evidence_ids=set(request.evidenceIds),
            reference_points=request.referenceAnswerPoints,
            dimension=request.dimension,
        )

    def judge_for(self, *, evidence_ids: set[str], reference_points: list[str], dimension: str = "answer_relevancy") -> JudgeResult:
        return JudgeResult(
            status=self.status,
            dimension=dimension,
            score=self.score if self.status == "completed" else None,
            reason=self.reason,
            evidenceIds=sorted(evidence_ids),
            confidence=1.0 if self.status == "completed" else None,
            provider="fake",
            model="fake-judge",
            promptVersion="fake-v1",
            rubricVersion="v1",
            invocationId=f"FAKE-{uuid4().hex[:12]}",
            tokenSource="unknown",
        )


class DeepSeekJudge:
    """Live Judge adapter. It is never constructed by default pytest or replay."""

    def __init__(self, gateway, *, model: str | None = None, prompt_version: str = "judge-v1", rubric_version: str = "v1"):
        self.gateway = gateway
        self.model = model or getattr(gateway, "model", "deepseek-chat")
        self.prompt_version = prompt_version
        self.rubric_version = rubric_version

    def judge(self, request: JudgeRequest) -> JudgeResult:
        schema = {
            "type": "object",
            "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 4}, "reason": {"type": "string"}, "evidenceIds": {"type": "array", "items": {"type": "string"}}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["score", "reason", "evidenceIds", "confidence"],
            "additionalProperties": False,
        }
        try:
            data = self.gateway.complete_json(
                system="你是质检评测 Judge，只能依据输入证据，输出 JSON。",
                user=request.model_dump_json(ensure_ascii=False),
                schema=schema,
                validate=lambda value: value,
                operation="judge",
            )
            evidence = list(data.get("evidenceIds", []))
            if not set(evidence).issubset(set(request.evidenceIds)):
                return JudgeResult(status="invalid", dimension=request.dimension, reason="Judge 引用了输入之外的证据", provider="deepseek", model=self.model, promptVersion=self.prompt_version, rubricVersion=self.rubric_version)
            return JudgeResult(status="completed", dimension=request.dimension, score=int(data["score"]), reason=str(data["reason"]), evidenceIds=evidence, confidence=float(data["confidence"]), provider="deepseek", model=self.model, promptVersion=self.prompt_version, rubricVersion=self.rubric_version, invocationId=getattr(self.gateway, "last_invocation_id", None), tokenSource="provider_reported" if getattr(self.gateway, "last_usage", None) else "unknown")
        except Exception as exc:
            return JudgeResult(status="unavailable", dimension=request.dimension, reason="Judge 调用不可用", provider="deepseek", model=self.model, promptVersion=self.prompt_version, rubricVersion=self.rubric_version)
