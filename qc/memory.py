"""分层记忆：工作 / 情节 / 语义 / 程序。

设计原则：
- 工作记忆：当轮焦点与证据窗（已在 LoopContext + context_builder）
- 情节记忆：一次或多次 run 的可查询轨迹（RunStore 之上）
- 语义记忆：制度/规则/案例（KnowledgeIndex / RAG）
- 程序记忆：稳定流程与白名单动作（代码本身 + 本模块清单）

跨客户长期记忆默认不做；组织级仅聚合脱敏后的结构化结果。
"""

from __future__ import annotations

from typing import Any

from qc.models import AgentTraceEvent, AnalysisResult, QualityReport
from qc.run_store import RunStore


# 程序记忆：系统「会做什么」的稳定清单（不靠模型发明）
PROCEDURAL_MEMORY: dict[str, Any] = {
    "allowedLoopActions": [
        "EXPAND_CONTEXT",
        "SEARCH_KNOWLEDGE",
        "QUERY_ACTION_AUDIT",
        "REVISE_REPORT",
        "FINALIZE",
    ],
    "loopBudgets": {
        "maxIterations": 3,
        "maxToolCalls": 8,
        "timeoutSeconds": 90,
    },
    "directPathEventTypes": [
        "REPAYMENT_DISPUTE",
        "THREAT_OR_COERCION",
        "THIRD_PARTY_CONTACT",
    ],
    "businessFactDefault": "NOT_CHECKED",
    "scoring": "authoritative_rule_repository_penalties_only",
}


class EpisodicMemory:
    """情节记忆：基于 RunStore 的可查询历史。

    不是向量聊天记忆，而是「哪次质检、什么事件、什么违规、是否进 Loop」。
    """

    def __init__(self, run_store: RunStore):
        self.run_store = run_store

    def remember_run(self, result: AnalysisResult) -> dict[str, Any]:
        """从 AnalysisResult 抽取可检索情节卡片（调用方仍负责 save_result）。"""
        report = result.report
        return {
            "runId": result.runId,
            "status": result.status,
            "loopUsed": result.loopUsed,
            "loopReason": result.loopReason,
            "callId": report.callId if report else None,
            "score": report.score if report else None,
            "disposition": report.disposition.value if report else None,
            "eventTypes": [e.type.value for e in report.events] if report else [],
            "ruleIds": [v.ruleId for v in report.violations] if report else [],
            "tracePhases": [t.phase for t in result.trace],
        }

    def get_episode(self, run_id: str) -> dict[str, Any]:
        raw = self.run_store.get_run(run_id)
        result = raw.get("result") or {}
        report = result if isinstance(result, dict) else {}
        # result_json 存的是 QualityReport 或整包，兼容两种
        if "report" in report:
            report = report.get("report") or {}
        events = report.get("events") or []
        violations = report.get("violations") or []
        return {
            "runId": raw["runId"],
            "caseId": raw["caseId"],
            "callId": raw["callId"],
            "status": raw["status"],
            "eventTypes": [
                e.get("type") for e in events if isinstance(e, dict)
            ],
            "ruleIds": [
                v.get("ruleId") for v in violations if isinstance(v, dict)
            ],
            "score": report.get("score"),
            "disposition": report.get("disposition"),
            "trace": raw.get("events") or [],
            "request": raw.get("request"),
        }

    def find_by_call_id(self, call_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.run_store.find_runs(call_id=call_id, limit=limit)

    def find_by_rule_id(self, rule_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.run_store.find_runs(rule_id=rule_id, limit=limit)

    def find_similar_by_event_types(
        self,
        event_types: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """结构化相似：共享至少一个事件类型的历史 run（非向量相似）。"""
        if not event_types:
            return []
        return self.run_store.find_runs(event_types=event_types, limit=limit)


class SemanticMemory:
    """语义记忆门面：包装 KnowledgeIndex，并暴露索引版本。"""

    def __init__(self, knowledge_index):
        self.knowledge_index = knowledge_index

    @property
    def index_version(self) -> str | None:
        return getattr(self.knowledge_index, "index_version", None)

    def recall(self, query: str, event_type, at_time, top_k: int = 5):
        return self.knowledge_index.search(query, event_type, at_time, top_k=top_k)

    def stats(self) -> dict[str, Any]:
        docs = getattr(self.knowledge_index, "documents", []) or []
        by_cat: dict[str, int] = {}
        for d in docs:
            cat = d.get("category", "UNKNOWN")
            by_cat[cat] = by_cat.get(cat, 0) + 1
        return {
            "documentCount": len(docs),
            "byCategory": by_cat,
            "indexVersion": self.index_version,
        }


class MemoryFacade:
    """统一入口，便于依赖注入。"""

    def __init__(self, run_store: RunStore, knowledge_index):
        self.episodic = EpisodicMemory(run_store)
        self.semantic = SemanticMemory(knowledge_index)
        self.procedural = PROCEDURAL_MEMORY

    def working_snapshot(self, loop_context) -> dict[str, Any]:
        """从 LoopContext 抽出工作记忆快照（可落盘）。"""
        return {
            "focusEventId": getattr(loop_context, "focusEventId", None),
            "focusTurnIds": list(getattr(loop_context, "focusTurnIds", []) or []),
            "evidenceWindowTurnIds": [
                t.turnId for t in (getattr(loop_context, "evidenceWindow", None) or [])
            ],
            "gaps": list(getattr(loop_context, "gaps", []) or []),
            "decisionLog": list(getattr(loop_context, "decisionLog", []) or [])[-8:],
            "reason": getattr(loop_context, "reason", None),
        }
