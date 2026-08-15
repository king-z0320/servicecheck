from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = ROOT / "knowledge/rag_calibration.json"


def calibrated_support_score(path: Path = CALIBRATION_PATH) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    value = float(data["threshold"])
    if not 0 < value <= 1:
        raise ValueError("calibrated RAG threshold must be in (0, 1]")
    if float(data["negativeMax"]) >= value or value >= float(data["positiveMin"]):
        raise ValueError("calibrated RAG threshold does not separate recorded cases")
    return value


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str | None
    deepseek_model: str
    deepseek_base_url: str
    deepseek_timeout_seconds: float
    audit_service_url: str
    rag_min_support_score: float

    @classmethod
    def from_env(cls) -> "Settings":
        raw_threshold = os.getenv("RAG_MIN_SUPPORT_SCORE")
        threshold = (
            float(raw_threshold)
            if raw_threshold is not None
            else calibrated_support_score()
        )
        if not 0 <= threshold <= 1:
            raise ValueError("RAG_MIN_SUPPORT_SCORE must be between zero and one")
        return cls(
            deepseek_api_key=(
                os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("openrouter_api_key")
            ),
            deepseek_model=os.getenv(
                "DEEPSEEK_MODEL",
                os.getenv("model_name", "deepseek-chat"),
            ),
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL",
                "https://api.deepseek.com/v1/chat/completions",
            ),
            deepseek_timeout_seconds=float(
                os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60")
            ),
            audit_service_url=os.getenv(
                "AUDIT_SERVICE_URL",
                os.getenv("audit_service_url", "http://127.0.0.1:5002"),
            ),
            rag_min_support_score=threshold,
        )
