from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


class MetricsRegistry:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self.stage_duration = Histogram("servicecheck_stage_duration_seconds", "Stage duration", ["stage", "status"], registry=self.registry)
        self.llm_errors = Counter("servicecheck_llm_errors_total", "LLM errors", ["error_code"], registry=self.registry)
        self.gate_results = Counter("servicecheck_gate_results_total", "Quality gate results", ["result"], registry=self.registry)
        self.eval_cases = Counter("servicecheck_eval_cases_total", "Evaluation cases", ["split", "status"], registry=self.registry)

    def observe_stage(self, stage: str, duration_seconds: float, *, status: str = "completed"):
        self.stage_duration.labels(stage=stage, status=status).observe(max(0.0, duration_seconds))
    def record_llm_error(self, error_code: str):
        self.llm_errors.labels(error_code=error_code).inc()
    def record_gate(self, result: str):
        self.gate_results.labels(result=result).inc()
    def record_eval_case(self, split: str, status: str):
        self.eval_cases.labels(split=split, status=status).inc()
    def render(self) -> bytes:
        return generate_latest(self.registry)

