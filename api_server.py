#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""催收录音质检 Agent API 服务。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydantic import ValidationError

from qc.errors import AnalysisError, ErrorStage
from qc.models import AnalysisRequest, TranscriptTurn

load_dotenv()

ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.getenv(
    "DEEPSEEK_MODEL",
    os.getenv("model_name", "deepseek-chat"),
)


EXTERNAL_DEPENDENCY_CODES = {
    "LLM_TIMEOUT",
    "LLM_RATE_LIMITED",
    "LLM_AUTH_FAILED",
    "LLM_UPSTREAM_ERROR",
}


def http_status_for_result(result) -> int:
    if result.status in {"COMPLETED", "PARTIAL"}:
        return 200
    if any(error.code in EXTERNAL_DEPENDENCY_CODES for error in result.errors):
        return 503
    return 500


def _validation_error(code: str, message: str) -> AnalysisError:
    return AnalysisError(
        code=code,
        stage=ErrorStage.VALIDATION,
        message=message,
        retryable=False,
        attempts=0,
    )


def _validation_response(exc: ValidationError | None = None, *, missing_time=False):
    code = "INVALID_REQUEST"
    message = "请求字段不合法"
    if missing_time:
        code = "INVALID_CALL_STARTED_AT"
        message = "callStartedAt 必须显式提供带时区的通话时间"
    elif exc is not None:
        safe_errors = exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        combined = " ".join(
            str(item.get("msg", "")) + " " + ".".join(map(str, item.get("loc", ())))
            for item in safe_errors
        ).lower()
        if "unique" in combined and "turnid" in combined:
            code, message = "DUPLICATE_TURN_ID", "transcript 中存在重复 turnId"
        elif "text" in combined and "blank" in combined:
            code, message = "EMPTY_TURN_TEXT", "transcript 中存在空文本"
        elif "callstartedat" in combined or "timezone" in combined:
            code, message = "INVALID_CALL_STARTED_AT", "callStartedAt 必须包含时区"
        elif "start" in combined or "end" in combined or "time" in combined:
            code, message = "INVALID_TURN_TIME", "transcript 中存在非法时间"
    error = _validation_error(code, message)
    return {
        "error": "invalid request",
        "status": "FAILED",
        "errors": [error.model_dump(mode="json")],
    }


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
        raw_payload = request.get_json(silent=True) or {}
        if "callStartedAt" not in raw_payload:
            return jsonify(_validation_response(missing_time=True)), 400
        try:
            payload = AnalysisRequest.model_validate(raw_payload)
        except ValidationError as exc:
            return jsonify(_validation_response(exc)), 400

        result = quality_service().analyze(payload)
        return (
            jsonify(result.model_dump(mode="json")),
            http_status_for_result(result),
        )

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
            return jsonify(_validation_response(exc)), 400

        result = quality_service().analyze(payload)
        if result.status == "FAILED":
            return (
                jsonify(result.model_dump(mode="json")),
                http_status_for_result(result),
            )
        report = result.report
        response = {
            "success": result.status == "COMPLETED",
            "summary": report.summary,
            "qcReport": {
                "score": report.score,
                "violations": [
                    item.model_dump(mode="json")
                    for item in report.violations
                ],
            },
        }
        if result.status == "PARTIAL":
            response.update(
                {
                    "status": result.status.value,
                    "disposition": report.disposition.value,
                    "errors": [
                        error.model_dump(mode="json")
                        for error in result.errors
                    ],
                }
            )
        return jsonify(response), 200

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
    from qc.llm_gateway import DeepSeekGateway
    from qc.quality_gate import QualityGate
    from qc.rag import KnowledgeIndex
    from qc.rules import RuleRepository
    from qc.run_store import RunStore
    from qc.service import QualityAnalysisService
    from qc.config import Settings

    settings = Settings.from_env()
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    gateway = DeepSeekGateway(
        settings.deepseek_api_key,
        settings.deepseek_model,
        settings.deepseek_base_url,
        settings.deepseek_timeout_seconds,
    )
    rules = RuleRepository(ROOT / "knowledge/rules/quality_rules.json") #这个类的作用是：从指定的 JSON 文件中加载质检规则，并提供查询和验证规则的方法。
    knowledge = KnowledgeIndex(ROOT / "knowledge") #这个类的作用是：构建一个知识索引，用于存储和检索与质检相关的知识。
    knowledge.build()
    audit = AuditClient(
        settings.audit_service_url
    )
    direct = DirectAnalyzer(
        EventExtractor(gateway),
        knowledge,
        rules,
        audit,
        min_support_score=settings.rag_min_support_score,
    ) #这个类的作用是：直接分析通话转写，提取质检事件，查询知识索引和规则库，并返回质检报告。
    gate = QualityGate(rules, min_support_score=settings.rag_min_support_score) #这个类的作用是：根据规则库对质检报告进行质量检查，判断是否需要启动大模型循环分析。
    executor = QualityLoopExecutor(knowledge, audit, rule_repository=rules) #这个类的作用是：执行大模型循环分析，调用知识索引、外部业务系统和规则库，生成质检报告，并返回给调用方。
    loop = BoundedAgentLoop(
        LLMPlanner(gateway),
        executor,
        LLMEvaluator(gateway),
        gate,
    ) #这个类的作用是：实现一个有界的 Agent 循环分析器，包含计划、执行、评估和质量检查等步骤，确保大模型循环分析在合理的范围内进行。
    run_store = RunStore(ROOT / "data/qc_runs.db")
    run_store.fail_incomplete_runs(
        AnalysisError(
            code="PROCESS_INTERRUPTED",
            stage=ErrorStage.PERSISTENCE,
            message="上一次进程结束前任务未完成",
            retryable=True,
            attempts=0,
        )
    )
    return QualityAnalysisService( #这个类的作用是：提供一个统一的质检服务接口，封装直接分析器和大模型循环分析器，并提供 analyze() 方法供外部调用。
        direct,
        loop,
        gate,
        run_store,
        min_support_score=settings.rag_min_support_score,
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
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
