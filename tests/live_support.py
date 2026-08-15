from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from threading import Thread

from werkzeug.serving import make_server

from mock_audit_server import app as mock_audit_app
from qc.agent_loop import (
    BoundedAgentLoop,
    LLMEvaluator,
    LLMPlanner,
    QualityLoopExecutor,
)
from qc.audit_client import AuditClient
from qc.config import Settings
from qc.direct_analyzer import DirectAnalyzer
from qc.event_extractor import EventExtractor
from qc.llm_gateway import DeepSeekGateway
from qc.quality_gate import QualityGate
from qc.rag import KnowledgeIndex
from qc.rules import RuleRepository
from qc.run_store import RunStore
from qc.service import QualityAnalysisService


ROOT = Path(__file__).resolve().parents[1]


def live_settings() -> Settings:
    settings = Settings.from_env()
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not available for live tests")
    return settings


@contextmanager
def running_mock_audit_server():
    server = make_server("127.0.0.1", 0, mock_audit_app)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def build_live_service(
    db_path: Path,
    audit_url: str,
    *,
    knowledge_index: KnowledgeIndex | None = None,
) -> QualityAnalysisService:
    settings = live_settings()
    gateway = DeepSeekGateway(
        settings.deepseek_api_key,
        settings.deepseek_model,
        settings.deepseek_base_url,
        settings.deepseek_timeout_seconds,
    )
    rules = RuleRepository(ROOT / "knowledge/rules/quality_rules.json")
    knowledge = knowledge_index or KnowledgeIndex(ROOT / "knowledge")
    if knowledge.vectors is None:
        knowledge.build()
    audit = AuditClient(audit_url)
    gate = QualityGate(rules, settings.rag_min_support_score)
    direct = DirectAnalyzer(
        EventExtractor(gateway),
        knowledge,
        rules,
        audit,
        min_support_score=settings.rag_min_support_score,
    )
    loop = BoundedAgentLoop(
        LLMPlanner(gateway),
        QualityLoopExecutor(knowledge, audit, rule_repository=rules),
        LLMEvaluator(gateway),
        gate,
    )
    return QualityAnalysisService(
        direct,
        loop,
        gate,
        RunStore(db_path),
        min_support_score=settings.rag_min_support_score,
    )
