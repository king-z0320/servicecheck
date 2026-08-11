#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""催收录音质检 Agent API 服务。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydantic import ValidationError

from qc.models import AnalysisRequest, TranscriptTurn

load_dotenv()

ROOT = Path(__file__).resolve().parent
OPENROUTER_API_KEY = os.getenv("openrouter_api_key")
MODEL_NAME = os.getenv("model_name", "deepseek-chat")


def create_app(service=None):
    app = Flask(__name__)
    CORS(app)
    app.config["QUALITY_SERVICE"] = service

    def quality_service():
        configured = app.config.get("QUALITY_SERVICE")
        if configured is None:
            raise RuntimeError("quality analysis service is not configured")
        return configured

    @app.get("/api/health") #健康检查接口
    def health_check():
        return jsonify({"status": "ok", "model": MODEL_NAME})

    @app.post("/api/agent/analyze") #大模型质检接口
    def agent_analyze():
        try:
            payload = AnalysisRequest.model_validate(
                request.get_json(silent=True) or {}
            )
        except ValidationError as exc:
            return jsonify(
                {"error": "invalid request", "details": exc.errors()}
            ), 400

        result = quality_service().analyze(payload)
        return jsonify(result.model_dump(mode="json"))

    @app.get("/api/agent/runs/<run_id>") #查询质检运行结果接口
    def get_agent_run(run_id):
        try:
            return jsonify(quality_service().get_run(run_id))
        except KeyError:
            return jsonify({"error": "run not found"}), 404

    @app.post("/api/analyze") #兼容旧接口，仍然可以使用 /api/analyze 来提交质检请求
    def legacy_analyze():
        data = request.get_json(silent=True) or {}
        try:
            turns = [
                TranscriptTurn(
                    turnId=item.get("turnId") or f"T{index:04d}",
                    speaker=item.get("speaker", "未知"),
                    text=item.get("text", ""),
                    start=item.get("start", 0),
                    end=item.get("end", 0),
                )
                for index, item in enumerate(data.get("transcript", []), 1)
            ]
            payload = AnalysisRequest(
                caseId=data.get("caseId", "LEGACY-CASE"),
                callId=data.get("callId", "LEGACY-CALL"),
                transcript=turns,
            )
        except ValidationError as exc:
            return jsonify(
                {"error": "invalid request", "details": exc.errors()}
            ), 400

        result = quality_service().analyze(payload)
        report = result.report
        return jsonify(
            {
                "success": True,
                "summary": report.summary,
                "qcReport": {
                    "score": report.score,
                    "violations": [
                        item.model_dump(mode="json")
                        for item in report.violations
                    ],
                },
            }
        )

    return app


def build_service(): #这个函数的作用是：构建一个质检服务对象，包含大模型分析器、规则库、知识索引、外部业务系统客户端等组件，并返回一个 QualityAnalysisService 实例。
    from qc.agent_loop import (
        BoundedAgentLoop,
        LLMEvaluator,
        LLMPlanner,
        QualityLoopExecutor,
    )
    from qc.audit_client import AuditClient
    from qc.direct_analyzer import DirectAnalyzer
    from qc.event_extractor import EventExtractor
    from qc.llm_gateway import OpenRouterGateway
    from qc.quality_gate import QualityGate
    from qc.rag import KnowledgeIndex
    from qc.rules import RuleRepository
    from qc.run_store import RunStore
    from qc.service import QualityAnalysisService

    if not OPENROUTER_API_KEY:
        raise RuntimeError("openrouter_api_key is not configured")

    gateway = OpenRouterGateway(OPENROUTER_API_KEY, MODEL_NAME)
    rules = RuleRepository(ROOT / "knowledge/rules/quality_rules.json") #这个类的作用是：从指定的 JSON 文件中加载质检规则，并提供查询和验证规则的方法。
    knowledge = KnowledgeIndex(ROOT / "knowledge") #这个类的作用是：构建一个知识索引，用于存储和检索与质检相关的知识。
    knowledge.build()
    audit = AuditClient(
        os.getenv("audit_service_url", "http://127.0.0.1:5002")
    )
    direct = DirectAnalyzer(
        EventExtractor(gateway),
        knowledge,
        rules,
        audit,
    ) #这个类的作用是：直接分析通话转写，提取质检事件，查询知识索引和规则库，并返回质检报告。
    gate = QualityGate(rules) #这个类的作用是：根据规则库对质检报告进行质量检查，判断是否需要启动大模型循环分析。
    executor = QualityLoopExecutor(knowledge, audit, rule_repository=rules) #这个类的作用是：执行大模型循环分析，调用知识索引、外部业务系统和规则库，生成质检报告，并返回给调用方。
    loop = BoundedAgentLoop(
        LLMPlanner(gateway),
        executor,
        LLMEvaluator(gateway),
        gate,
    ) #这个类的作用是：实现一个有界的 Agent 循环分析器，包含计划、执行、评估和质量检查等步骤，确保大模型循环分析在合理的范围内进行。
    return QualityAnalysisService( #这个类的作用是：提供一个统一的质检服务接口，封装直接分析器和大模型循环分析器，并提供 analyze() 方法供外部调用。
        direct,
        loop,
        RunStore(ROOT / "data/qc_runs.db"),
    )


app = create_app()


if __name__ == "__main__":
    print("=" * 60)
    print("催收录音质检 Agent API 服务")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print("端点:")
    print("   GET  /api/health")
    print("   POST /api/agent/analyze")
    print("   GET  /api/agent/runs/<run_id>")
    print("   POST /api/analyze (兼容)")
    print("=" * 60)
    app.config["QUALITY_SERVICE"] = build_service()
    app.run(host="0.0.0.0", port=5001, debug=True)
