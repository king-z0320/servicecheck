"""Agent 工作上下文构建：分层、有界，避免把全文 LoopContext 塞进 LLM。"""

from __future__ import annotations

import json
from typing import Any

from qc.models import QualityEvent, QualityReport, TranscriptTurn

# 默认字符预算（中文场景用字符近似 token）
DEFAULT_PLANNER_MAX_CHARS = 6000
DEFAULT_EVALUATOR_MAX_CHARS = 8000


def select_focus_event(report: QualityReport) -> QualityEvent | None:
    """优先歧义事件，否则取第一个事件。"""
    if not report.events:
        return None
    for event in report.events:
        if event.ambiguous:
            return event
    return report.events[0]


def build_evidence_window(
    transcript: list[TranscriptTurn],
    focus_turn_ids: list[str],
    radius: int = 2,
) -> list[TranscriptTurn]:
    """以焦点 turn 为中心，取前后 radius 句作为证据窗。"""
    if not transcript:
        return []
    if not focus_turn_ids:
        return transcript[: min(5, len(transcript))]

    index_by_id = {turn.turnId: i for i, turn in enumerate(transcript)}
    indices = [index_by_id[tid] for tid in focus_turn_ids if tid in index_by_id]
    if not indices:
        return transcript[: min(5, len(transcript))]

    start = max(0, min(indices) - radius)
    end = min(len(transcript), max(indices) + radius + 1)
    return transcript[start:end]


def summarize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """工具完整结果进 SQLite/trace；进 LLM 的只留摘要。"""
    action = observation.get("action") or observation.get("decision") or "UNKNOWN"
    summary: dict[str, Any] = {"action": action}
    if "hits" in observation and isinstance(observation["hits"], list):
        summary["hitCount"] = len(observation["hits"])
        summary["hitIds"] = [
            h.get("documentId")
            for h in observation["hits"][:5]
            if isinstance(h, dict)
        ]
    if "turns" in observation and isinstance(observation["turns"], list):
        summary["turnCount"] = len(observation["turns"])
        summary["turnIds"] = [
            t.get("turnId")
            for t in observation["turns"][:12]
            if isinstance(t, dict)
        ]
    if "windowTurnIds" in observation:
        summary["windowTurnIds"] = observation["windowTurnIds"]
    if "audit" in observation and isinstance(observation["audit"], dict):
        audit = observation["audit"]
        summary["audit"] = {
            "disputeTicketCreated": audit.get("disputeTicketCreated"),
            "followUpType": audit.get("followUpType"),
            "errorCount": len(audit.get("errors") or []),
        }
    if "revised" in observation:
        summary["revised"] = observation["revised"]
    if "violationCount" in observation:
        summary["violationCount"] = observation["violationCount"]
    if observation.get("error"):
        summary["error"] = str(observation["error"])[:200]
    return summary


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n…[上下文已截断]"


def build_planner_user(
    context,
    max_chars: int = DEFAULT_PLANNER_MAX_CHARS,
) -> str:
    """Planner 只看：案件卡 + 证据窗 + gaps + 检索/审计摘要 + 决策日志摘要。"""
    report = context.report
    focus = None
    if getattr(context, "focusEventId", None):
        focus = next(
            (e for e in report.events if e.eventId == context.focusEventId),
            None,
        )
    if focus is None:
        focus = select_focus_event(report)

    window = list(getattr(context, "evidenceWindow", None) or [])
    if not window and focus:
        window = build_evidence_window(
            context.transcript,
            list(getattr(context, "focusTurnIds", None) or focus.turnIds),
        )

    obs_summaries = [
        summarize_observation(o) if isinstance(o, dict) else {"raw": str(o)[:120]}
        for o in (context.observations or [])[-5:]
    ]
    decision_log = list(getattr(context, "decisionLog", None) or [])[-8:]

    payload = {
        "caseCard": {
            "callId": report.callId,
            "loopReason": context.reason,
            "focusEvent": focus.model_dump(mode="json") if focus else None,
            "disposition": report.disposition.value,
            "score": report.score,
            "violationRuleIds": [v.ruleId for v in report.violations],
        },
        "evidenceWindow": [t.model_dump(mode="json") for t in window],
        "gaps": list(context.gaps or [])[:10],
        "knowledgeHitIds": [h.documentId for h in report.knowledgeHits[:5]],
        "auditSummary": (
            {
                "disputeTicketCreated": report.auditSnapshot.disputeTicketCreated,
                "crmSummary": (report.auditSnapshot.crmSummary or "")[:80],
                "errors": report.auditSnapshot.errors[:3],
            }
            if report.auditSnapshot
            else None
        ),
        "recentObservations": obs_summaries,
        "decisionLog": decision_log,
        "allowedActions": [
            "EXPAND_CONTEXT",
            "SEARCH_KNOWLEDGE",
            "QUERY_ACTION_AUDIT",
            "REVISE_REPORT",
            "FINALIZE",
        ],
        "notes": "不要查询客户余额；只选一个动作；依据证据窗与 gaps。",
    }
    return _clip(json.dumps(payload, ensure_ascii=False), max_chars)


def build_evaluator_user(
    context,
    gate_result,
    max_chars: int = DEFAULT_EVALUATOR_MAX_CHARS,
) -> str:
    report = context.report
    focus = select_focus_event(report)
    window = list(getattr(context, "evidenceWindow", None) or [])
    if not window and focus:
        window = build_evidence_window(
            context.transcript,
            list(getattr(context, "focusTurnIds", None) or focus.turnIds),
        )

    payload = {
        "caseCard": {
            "callId": report.callId,
            "loopReason": context.reason,
            "focusEvent": focus.model_dump(mode="json") if focus else None,
            "disposition": report.disposition.value,
            "score": report.score,
            "violations": [v.model_dump(mode="json") for v in report.violations[:5]],
            "businessFact": report.businessFact.model_dump(mode="json"),
        },
        "evidenceWindow": [t.model_dump(mode="json") for t in window],
        "knowledgeHits": [
            {
                "documentId": h.documentId,
                "title": h.title,
                "score": h.score,
                "version": h.version,
            }
            for h in report.knowledgeHits[:5]
        ],
        "auditSummary": (
            report.auditSnapshot.model_dump(mode="json")
            if report.auditSnapshot
            else None
        ),
        "gaps": list(context.gaps or [])[:10],
        "GATE": gate_result.model_dump(mode="json"),
        "notes": (
            "不得将客户主张写成已确认结清；"
            "门禁失败时优先 NEEDS_MORE_CONTEXT 或 REPORT_REVISION_REQUIRED。"
        ),
    }
    return _clip(json.dumps(payload, ensure_ascii=False), max_chars)
