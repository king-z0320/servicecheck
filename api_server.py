#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""催收录音质检 Agent API 服务。"""

import os
import re
from time import monotonic
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import Body, FastAPI, Header, Query, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from qc.errors import AnalysisError, ErrorStage
from qc.models import AnalysisRequest, AnalysisResult, TranscriptTurn
from qc.observability.metrics import MetricsRegistry

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


class AgentAnalysisRequest(AnalysisRequest):
    """HTTP request contract requiring an explicit call timestamp."""

    callStartedAt: datetime


class LegacyAnalysisRequest(BaseModel):
    caseId: str = "LEGACY-CASE"
    callId: str = "LEGACY-CALL"
    transcript: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: str


class ErrorResponse(BaseModel):
    error: str
    status: Literal["FAILED"] = "FAILED"
    errors: list[AnalysisError]


class RunNotFoundResponse(BaseModel):
    error: Literal["run not found"] = "run not found"


class LegacyQCReport(BaseModel):
    score: int
    violations: list[dict[str, Any]]


class LegacyAnalysisResponse(BaseModel):
    success: bool
    summary: dict[str, Any]
    qcReport: LegacyQCReport
    status: str | None = None
    disposition: str | None = None
    errors: list[dict[str, Any]] | None = None


class BatchCreateRequest(BaseModel):
    source_dir: str = Field(min_length=1, max_length=512)


class BatchAcceptedResponse(BaseModel):
    batch_id: str
    status: str
    total: int


class BatchStatusResponse(BaseModel):
    batch_id: str
    status: str
    total: int
    by_status: dict[str, int] = Field(default_factory=dict)


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


def _validation_response(
    exc: ValidationError | RequestValidationError | None = None,
    *,
    missing_time: bool = False,
):
    code = "INVALID_REQUEST"
    message = "请求字段不合法"
    if missing_time:
        code = "INVALID_CALL_STARTED_AT"
        message = "callStartedAt 必须显式提供带时区的通话时间"
    elif exc is not None:
        try:
            safe_errors = exc.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
        except TypeError:
            safe_errors = exc.errors()
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


def create_app(
    service=None,
    run_store=None,
    artifact_store=None,
    batch_service=None,
    review_service=None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        created_service = False
        if application.state.quality_service is None:
            application.state.quality_service = build_service()
            application.state.run_store = application.state.quality_service.run_store
            created_service = True
        if application.state.artifact_store is None:
            application.state.artifact_store = build_artifact_store()
        if application.state.batch_service is None:
            try:
                from qc.batch.postgres_store import PostgresBatchStore
                from qc.batch.models import BatchConfig
                from qc.batch.service import PostgresBatchService
                from qc.database import database_url_from_env

                audio_root = Path(os.getenv("BATCH_AUDIO_ROOT", ROOT / "audio"))
                application.state.batch_service = PostgresBatchService(
                    PostgresBatchStore(database_url_from_env()),
                    audio_root,
                    BatchConfig(),
                )
            except (RuntimeError, OSError, ValueError):
                application.state.batch_service = None
        if application.state.review_service is None:
            try:
                from qc.database import database_url_from_env
                from qc.review_service import ReviewService
                from qc.review_store import PostgresReviewStore

                application.state.review_service = ReviewService(
                    PostgresReviewStore(database_url_from_env())
                )
            except (RuntimeError, OSError, ValueError):
                application.state.review_service = None
        try:
            yield
        finally:
            if created_service:
                close = getattr(application.state.run_store, "close", None)
                if close is not None:
                    close()

    app = FastAPI(
        title="客服质检 Agent API",
        version="1.0.0",
        lifespan=lifespan,
    )
    origins = [
        item.strip()
        for item in os.getenv(
            "API_CORS_ORIGINS",
            "http://127.0.0.1:8080,http://localhost:8080",
        ).split(",")
        if item.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Range", "Idempotency-Key"],
        expose_headers=["Accept-Ranges", "Content-Length", "Content-Range"],
    )
    app.state.quality_service = service
    app.state.run_store = run_store or getattr(service, "run_store", None)
    app.state.artifact_store = artifact_store
    app.state.batch_service = batch_service
    app.state.review_service = review_service
    app.state.metrics = MetricsRegistry()

    def quality_service():
        configured = app.state.quality_service
        if configured is None:
            raise RuntimeError("quality analysis service is not configured")
        return configured

    def workbench_store():
        configured = app.state.run_store
        if configured is None:
            configured = getattr(quality_service(), "run_store", None)
        if configured is None:
            raise RuntimeError("workbench run store is not configured")
        return configured

    def configured_artifact_store():
        configured = app.state.artifact_store
        if configured is None:
            raise RuntimeError("artifact store is not configured")
        return configured

    def configured_review_service():
        configured = app.state.review_service
        if configured is None:
            return JSONResponse(
                status_code=503,
                content={"error": "review service is not configured"},
            )
        return configured

    def review_error_response(exc):
        from qc.review_service import (
            ReviewIdempotencyConflict,
            ReviewNotFound,
            ReviewStateConflict,
            ReviewValidationError,
            ReviewVersionConflict,
        )

        payload = {"error": exc.code, "code": exc.code}
        payload.update(exc.details)
        if isinstance(exc, ReviewNotFound):
            return JSONResponse(status_code=404, content=payload)
        if isinstance(exc, ReviewValidationError):
            return JSONResponse(status_code=400, content=payload)
        if isinstance(
            exc,
            (ReviewVersionConflict, ReviewIdempotencyConflict, ReviewStateConflict),
        ):
            return JSONResponse(status_code=409, content=payload)
        return JSONResponse(status_code=400, content=payload)

    def configured_batch_service():
        configured = app.state.batch_service
        if configured is None:
            return JSONResponse(
                status_code=503,
                content={"error": "batch service is not configured"},
            )
        return configured

    def not_found(resource: str):
        return JSONResponse(status_code=404, content={"error": f"{resource} not found"})

    def parse_byte_range(value: str, size: int) -> tuple[int, int]:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
        if match is None or size <= 0:
            raise ValueError("invalid byte range")
        first, last = match.groups()
        if not first and not last:
            raise ValueError("empty byte range")
        if not first:
            length = int(last)
            if length <= 0:
                raise ValueError("invalid suffix range")
            return max(0, size - length), size - 1
        start = int(first)
        if start >= size:
            raise ValueError("range starts beyond artifact")
        end = size - 1 if not last else min(int(last), size - 1)
        if end < start:
            raise ValueError("range ends before it starts")
        return start, end

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request, exc):
        del request
        return JSONResponse(status_code=400, content=_validation_response(exc))

    @app.get("/api/health", response_model=HealthResponse) #健康检查接口
    def health_check():
        return HealthResponse(model=MODEL_NAME)

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint():
        return Response(
            content=app.state.metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/batches", response_model=BatchAcceptedResponse, status_code=202)
    def create_batch(
        payload: BatchCreateRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        batch_service = configured_batch_service()
        if isinstance(batch_service, JSONResponse):
            return batch_service
        try:
            return batch_service.create_batch(payload.source_dir, idempotency_key)
        except Exception as exc:
            from qc.batch.service import BatchCapacityError, IdempotencyConflictError

            if isinstance(exc, BatchCapacityError):
                return JSONResponse(
                    status_code=429,
                    content={"error": "batch capacity exceeded", "detail": str(exc)},
                )
            if isinstance(exc, IdempotencyConflictError):
                return JSONResponse(
                    status_code=409,
                    content={"error": "idempotency conflict", "detail": str(exc)},
                )
            if not isinstance(exc, (FileNotFoundError, ValueError)):
                raise
            return JSONResponse(
                status_code=400,
                content={"error": "invalid batch source", "detail": str(exc)},
            )

    @app.get("/batches/{batch_id}", response_model=BatchStatusResponse)
    def get_batch(batch_id: str):
        batch_service = configured_batch_service()
        if isinstance(batch_service, JSONResponse):
            return batch_service
        try:
            return batch_service.get_batch(batch_id)
        except KeyError:
            return not_found("batch")

    @app.get("/batches/{batch_id}/items", response_model=list[dict[str, Any]])
    def get_batch_items(batch_id: str):
        batch_service = configured_batch_service()
        if isinstance(batch_service, JSONResponse):
            return batch_service
        try:
            return batch_service.list_items(batch_id)
        except KeyError:
            return not_found("batch")

    @app.post(
        "/api/agent/analyze",
        response_model=AnalysisResult,
        responses={
            400: {"model": ErrorResponse},
            500: {"model": AnalysisResult},
            503: {"model": AnalysisResult},
        },
    ) #大模型质检接口
    def agent_analyze(payload: AgentAnalysisRequest, response: Response):
        request_model = AnalysisRequest.model_validate(payload.model_dump())
        started = monotonic()
        result = quality_service().analyze(request_model)
        app.state.metrics.observe_stage("quality_analysis", monotonic() - started, status=result.status.lower())
        app.state.metrics.record_gate("passed" if result.status == "COMPLETED" else "review_or_failed")
        response.status_code = http_status_for_result(result)
        return result

    @app.get(
        "/api/agent/runs/{run_id}",
        response_model=dict[str, Any],
        responses={404: {"model": RunNotFoundResponse}},
    ) #查询质检运行结果接口
    def get_agent_run(run_id: str):
        try:
            return quality_service().get_run(run_id)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content=RunNotFoundResponse().model_dump(mode="json"),
            )

    @app.get("/api/cases", response_model=dict[str, Any])
    def list_cases(
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    ):
        return workbench_store().list_cases(page=page, page_size=page_size)

    @app.get("/api/cases/{case_id}", response_model=dict[str, Any])
    def get_case(case_id: str):
        try:
            return workbench_store().get_case(case_id)
        except KeyError:
            return not_found("case")

    @app.get("/api/calls/{call_id}", response_model=dict[str, Any])
    def get_call(call_id: str):
        try:
            return workbench_store().get_call(call_id)
        except KeyError:
            return not_found("call")

    @app.get("/api/calls/{call_id}/transcript", response_model=list[TranscriptTurn])
    def get_call_transcript(call_id: str):
        try:
            transcript = workbench_store().get_transcript(call_id)
        except KeyError:
            return not_found("call")
        try:
            return [TranscriptTurn.model_validate(item) for item in transcript]
        except ValidationError:
            return JSONResponse(
                status_code=409,
                content={"error": "transcript integrity check failed"},
            )

    @app.get("/api/calls/{call_id}/runs", response_model=list[dict[str, Any]])
    def get_call_runs(call_id: str):
        store = workbench_store()
        try:
            store.get_call(call_id)
            return store.list_runs_by_call(call_id)
        except KeyError:
            return not_found("call")

    @app.get("/api/reports/{report_id}", response_model=dict[str, Any])
    def get_report(report_id: str):
        try:
            return workbench_store().get_report(report_id)
        except KeyError:
            return not_found("report")

    @app.get("/api/review-tasks", response_model=dict[str, Any])
    def list_review_tasks(
        status: str | None = Query("PENDING"),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    ):
        from qc.review_service import ReviewError

        service = configured_review_service()
        if isinstance(service, JSONResponse):
            return service
        try:
            return service.list_tasks(status=status, page=page, page_size=page_size)
        except ReviewError as exc:
            return review_error_response(exc)

    @app.get("/api/review-tasks/{review_task_id}", response_model=dict[str, Any])
    def get_review_task(review_task_id: str):
        from qc.review_service import ReviewError

        service = configured_review_service()
        if isinstance(service, JSONResponse):
            return service
        try:
            return service.get_task(review_task_id)
        except ReviewError as exc:
            return review_error_response(exc)

    @app.post("/api/review-tasks/{review_task_id}/submit", response_model=dict[str, Any])
    def submit_review_task(
        review_task_id: str,
        payload: dict[str, Any] = Body(default_factory=dict),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        from qc.review_models import PROTECTED_SUBMIT_FIELDS, ReviewSubmitRequest
        from qc.review_service import ReviewError, configured_reviewer_context

        service = configured_review_service()
        if isinstance(service, JSONResponse):
            return service
        extra = set(payload) & PROTECTED_SUBMIT_FIELDS
        if extra:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_REQUEST",
                    "code": "INVALID_REQUEST",
                    "message": "protected fields are not writable",
                },
            )
        try:
            request = ReviewSubmitRequest.model_validate(payload)
            return service.submit(
                review_task_id,
                request,
                idempotency_key,
                configured_reviewer_context(),
            )
        except ValidationError as exc:
            return JSONResponse(status_code=400, content=_validation_response(exc))
        except ReviewError as exc:
            return review_error_response(exc)

    @app.get("/api/calls/{call_id}/audio")
    def get_call_audio(
        call_id: str,
        range_header: str | None = Header(default=None, alias="Range"),
    ):
        try:
            call = workbench_store().get_call(call_id)
        except KeyError:
            return not_found("call")
        uri = call.get("audioArtifactUri")
        if not uri:
            return not_found("audio")

        artifacts = configured_artifact_store()
        if not artifacts.exists(uri):
            return not_found("audio")
        expected_sha256 = call.get("audioSha256")
        if expected_sha256 and not artifacts.verify_sha256(uri, expected_sha256):
            return JSONResponse(
                status_code=409,
                content={"error": "artifact integrity check failed"},
            )
        try:
            size = artifacts.stat(uri).st_size
        except (FileNotFoundError, OSError, ValueError):
            return not_found("audio")

        status_code = 200
        start, end = 0, max(0, size - 1)
        headers = {"Accept-Ranges": "bytes"}
        if range_header is not None:
            try:
                start, end = parse_byte_range(range_header, size)
            except (TypeError, ValueError):
                return Response(
                    status_code=416,
                    headers={"Content-Range": f"bytes */{size}"},
                )
            status_code = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"

        content_length = 0 if size == 0 else end - start + 1
        headers["Content-Length"] = str(content_length)

        def stream_audio():
            remaining = content_length
            with artifacts.open(uri) as stream:
                stream.seek(start)
                while remaining > 0:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            stream_audio(),
            status_code=status_code,
            media_type=call.get("audioMimeType") or "application/octet-stream",
            headers=headers,
        )

    @app.post(
        "/api/analyze",
        response_model=LegacyAnalysisResponse | AnalysisResult,
        response_model_exclude_none=True,
        responses={400: {"model": ErrorResponse}},
    ) #兼容旧接口，仍然可以使用 /api/analyze 来提交质检请求
    def legacy_analyze(data: LegacyAnalysisRequest = Body(default_factory=LegacyAnalysisRequest)):
        try:
            turns = [
                TranscriptTurn(
                    turnId=item.get("turnId") or f"T{index:04d}",
                    speaker=item.get("speaker", "未知"),
                    text=item.get("text", ""),
                    start=item.get("start", 0),
                    end=item.get("end", 0),
                )
                for index, item in enumerate(data.transcript, 1)
            ]
            payload = AnalysisRequest(
                caseId=data.caseId,
                callId=data.callId,
                transcript=turns,
            )
        except ValidationError as exc:
            return JSONResponse(status_code=400, content=_validation_response(exc))

        result = quality_service().analyze(payload)
        if result.status == "FAILED":
            return JSONResponse(
                status_code=http_status_for_result(result),
                content=result.model_dump(mode="json"),
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
        return response

    def contract_openapi():
        if app.openapi_schema is None:
            schema = get_openapi(
                title=app.title,
                version=app.version,
                routes=app.routes,
            )
            for path_item in schema.get("paths", {}).values():
                for operation in path_item.values():
                    if isinstance(operation, dict):
                        operation.get("responses", {}).pop("422", None)
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = contract_openapi
    return app


def build_service(): #这个函数的作用是：构建一个质检服务对象，包含大模型分析器、规则库、知识索引、外部业务系统客户端等组件，并返回一个 QualityAnalysisService 实例。
    import hashlib

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
    from qc.database import database_url_from_env
    from qc.postgres_run_store import PostgresRunStore
    from qc.service import QualityAnalysisService
    from qc.config import Settings

    settings = Settings.from_env()
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    from qc.observability.usage import PostgresUsageLedger
    usage_ledger = PostgresUsageLedger(database_url_from_env())
    gateway = DeepSeekGateway(
        settings.deepseek_api_key,
        settings.deepseek_model,
        settings.deepseek_base_url,
        settings.deepseek_timeout_seconds,
        usage_ledger=usage_ledger,
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
    rule_version = hashlib.sha256(
        (ROOT / "knowledge/rules/quality_rules.json").read_bytes()
    ).hexdigest()[:12]
    run_store = PostgresRunStore(
        database_url_from_env(),
        model=settings.deepseek_model,
        prompt_version=os.getenv("PROMPT_VERSION") or None,
        rule_version=rule_version,
        knowledge_version=knowledge.index_version,
        runtime_version=os.getenv("RUNTIME_VERSION", "unknown"),
    )
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


def build_artifact_store():
    from qc.artifact_store import LocalArtifactStore

    configured = os.getenv("ARTIFACT_ROOT", "data/artifacts").strip()
    root = Path(configured)
    if not root.is_absolute():
        root = ROOT / root
    resolved = root.resolve(strict=False)
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError("ARTIFACT_ROOT must be inside the project root") from exc
    if resolved.drive.lower() == "c:":
        raise RuntimeError("ARTIFACT_ROOT must not be on C:")
    return LocalArtifactStore(resolved)


app = create_app()


if __name__ == "__main__":
    import uvicorn
    from qc.observability.runtime import configure_local_observability

    configure_local_observability(ROOT, process_name="api")

    print("=" * 60)
    print("催收录音质检 Agent API 服务")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print("端点:")
    print("   GET  /api/health")
    print("   POST /api/agent/analyze")
    print("   GET  /api/agent/runs/<run_id>")
    print("   GET  /api/review-tasks")
    print("   POST /api/review-tasks/<id>/submit")
    print("   POST /api/analyze (兼容)")
    print("=" * 60)
    app.state.quality_service = build_service()
    app.state.run_store = app.state.quality_service.run_store
    app.state.artifact_store = build_artifact_store()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
    )
