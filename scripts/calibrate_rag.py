from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qc.models import EventType
from qc.rag import KnowledgeIndex


DEFAULT_CASES = ROOT / "tests/gold/rag_calibration_cases.json"
DEFAULT_OUTPUT = ROOT / "knowledge/rag_calibration.json"
EMBEDDER_NAME = "BAAI/bge-small-zh-v1.5"


def calibrate(cases_path: Path = DEFAULT_CASES) -> dict:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    index = KnowledgeIndex(ROOT / "knowledge")
    index.build()
    scored = []
    calibration_cases = [
        case for case in cases if case.get("includeInThresholdCalibration", True)
    ]
    for case in calibration_cases:
        hits = index.search(
            case["query"],
            EventType(case["eventType"]),
            datetime(2026, 7, 27, tzinfo=timezone.utc),
            top_k=20,
        )
        support_ids = set(case["supportDocumentIds"])
        score = max(
            (hit.score for hit in hits if hit.documentId in support_ids),
            default=0.0,
        )
        scored.append(
            {
                "id": case["id"],
                "label": case["label"],
                "score": round(score, 6),
            }
        )
    positives = [item["score"] for item in scored if item["label"] == "positive"]
    negatives = [item["score"] for item in scored if item["label"] == "negative"]
    if not positives or not negatives:
        raise RuntimeError("calibration requires positive and negative cases")
    positive_min = min(positives)
    negative_max = max(negatives)
    if positive_min <= negative_max:
        raise RuntimeError(
            "no separating threshold: "
            f"positiveMin={positive_min}, negativeMax={negative_max}, "
            f"scores={json.dumps(scored, ensure_ascii=False)}"
        )
    threshold = round((positive_min + negative_max) / 2, 4)
    return {
        "schemaVersion": 1,
        "purpose": "rule-document retrieval relevance; not violation classification",
        "embedder": EMBEDDER_NAME,
        "indexVersion": index.index_version,
        "calibratedAt": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "positiveMin": positive_min,
        "negativeMax": negative_max,
        "positiveCount": len(positives),
        "negativeCount": len(negatives),
        "excludedCaseCount": len(cases) - len(calibration_cases),
        "excludedCaseIds": [
            case["id"]
            for case in cases
            if not case.get("includeInThresholdCalibration", True)
        ],
        "cases": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    artifact = calibrate(args.cases)
    content = json.dumps(artifact, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(content)
        return
    output = Path(args.output)
    output.write_text(content + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
