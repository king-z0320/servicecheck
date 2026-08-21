"""Dataset validation and deterministic content hashing for evaluations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from qc.eval.models import EvalCase, EvalSplit


class DatasetValidationError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LoadedDataset:
    cases: list[EvalCase]
    dataset_hash: str
    path: str


def _legacy_case(item: dict[str, Any], split: EvalSplit) -> EvalCase:
    """Explicit adapter for tests/gold cases; it never invents RAG labels."""
    expected_events = [
        {"eventType": value.get("type", value.get("eventType", "UNKNOWN")), "requiredTurnIds": value.get("turnIds", [])}
        for value in item.get("expectedEvents", [])
        if isinstance(value, dict)
    ]
    if not expected_events and item.get("candidates"):
        expected_events = [
            {"eventType": value.get("type", "UNKNOWN"), "requiredTurnIds": value.get("turnIds", [])}
            for value in item["candidates"]
        ]
    payload = {
        "caseId": item.get("id", item.get("caseId")),
        "split": split.value,
        "source": {"kind": "legacy-gold", "path": "tests/gold"},
        "labelNotes": "由 LegacyGoldAdapter 显式转换；未提供的 RAG 标签不运行。",
        "input": {
            "caseId": item.get("id", item.get("caseId")),
            "callId": item.get("callId", item.get("id", "legacy-call")),
            "callStartedAt": item.get("callStartedAt", "2026-01-01T00:00:00Z"),
            "transcript": item.get("transcript", []),
        },
        "expected": {
            "events": expected_events,
            "ruleIds": item.get("expectedRuleIds", []),
            "allowedDispositions": [item["expectedDisposition"]] if item.get("expectedDisposition") else ["AUTO_PASS", "AUTO_VIOLATION", "HUMAN_REVIEW_REQUIRED"],
            "requiredContextIds": [],
            "relevantContextIds": [],
            "forbiddenContextIds": [],
            "referenceAnswerPoints": [],
        },
    }
    return EvalCase.model_validate(payload)


def load_dataset(path: str | Path, *, expected_split: EvalSplit | str | None = None) -> LoadedDataset:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"cannot read dataset: {path}") from exc
    if not isinstance(raw, list):
        raise DatasetValidationError("dataset root must be an array")
    split = EvalSplit(expected_split) if expected_split is not None else None
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DatasetValidationError(f"case[{index}] must be object")
        try:
            case = EvalCase.model_validate(item)
        except ValidationError:
            if "input" not in item or "expected" not in item:
                try:
                    if split is None:
                        raise ValueError("legacy adapter requires expected split")
                    case = _legacy_case(item, split)
                except Exception as exc:
                    raise DatasetValidationError(f"case[{index}] invalid: {exc}") from exc
            else:
                raise DatasetValidationError(f"case[{index}] invalid") from None
        if case.caseId in seen:
            raise DatasetValidationError(f"duplicate caseId: {case.caseId}")
        if split is not None and case.split != split:
            raise DatasetValidationError(f"case {case.caseId} split does not match {split.value}")
        seen.add(case.caseId)
        case.caseHash = content_hash(case.model_dump(exclude={"caseHash"}, mode="json"))
        cases.append(case)
    dataset_hash = content_hash([case.model_dump(mode="json") for case in cases])
    return LoadedDataset(cases=cases, dataset_hash=dataset_hash, path=str(path))
