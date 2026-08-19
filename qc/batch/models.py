from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BatchFileStatus(str, Enum):
    PENDING = "PENDING" # 等待处理
    RUNNING = "RUNNING" # 正在处理
    INTERRUPTED = "INTERRUPTED" # 处理被中断，可能是系统异常或人工干预
    DONE = "DONE" # 处理完成
    FAILED_FINAL = "FAILED_FINAL" # 新流程的不可重试或重试耗尽失败终态
    DEAD_LETTER = "DEAD_LETTER" # 死信队列，处理失败且超过最大重试次数，需人工干预
    HUMAN_REVIEW = "HUMAN_REVIEW" # 需要人工复核，可能是模型判定不确定或触发了人工复核规则


class StageName(str, Enum): #这个是管线阶段的枚举类，表示不同的处理阶段
    TRANSCODE = "TRANSCODE"
    ASR = "ASR"
    EMOTION = "EMOTION"
    EVENT_EXTRACT = "EVENT_EXTRACT"
    RAG = "RAG"
    AUDIT = "AUDIT"
    QC = "QC"
    LOOP = "LOOP"


class FileRecord(BaseModel):
    source_uri: str
    idempotency_key: str
    callId: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchMeta(BaseModel):
    batch_id: str
    source: str
    total: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StageRecord(BaseModel):
    stage: StageName
    status: Literal["PENDING", "RUNNING", "DONE", "FAILED"] = "PENDING"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float = 0.0
    attempts: int = Field(default=0, ge=0)
    artifact_uri: str | None = None
    sha256: str | None = None
    producer_version: str | None = None
    error_code: str | None = None
    retryable: bool | None = None
    error: str | None = None


VALID_FILE_TRANSITIONS: dict[BatchFileStatus, set[BatchFileStatus]] = {
    BatchFileStatus.PENDING: {BatchFileStatus.RUNNING},
    BatchFileStatus.INTERRUPTED: {BatchFileStatus.RUNNING},
    BatchFileStatus.RUNNING: {
        BatchFileStatus.DONE,
        BatchFileStatus.HUMAN_REVIEW,
        BatchFileStatus.FAILED_FINAL,
        BatchFileStatus.INTERRUPTED,
    },
    BatchFileStatus.DONE: set(),
    BatchFileStatus.HUMAN_REVIEW: set(),
    BatchFileStatus.FAILED_FINAL: set(),
    BatchFileStatus.DEAD_LETTER: set(),
}


# 完整管线阶段顺序；LOOP 仅复杂案件触发，不纳入默认顺序计数。
PIPELINE_STAGES: list[StageName] = [
    StageName.TRANSCODE,
    StageName.ASR,
    StageName.EMOTION,
    StageName.EVENT_EXTRACT,
    StageName.RAG,
    StageName.AUDIT,
    StageName.QC,
]


class BatchConfig(BaseModel):
    """批量并发与预算参数。默认值是经验起点，待真实音频/ASR/GPU 基线压测后校准。"""

    cpu_workers: int = Field(default=4, ge=1)
    gpu_workers: int = Field(default=1, ge=1)
    llm_rpm: int = Field(default=60, ge=1)
    llm_cost_budget: float | None = None  # None = 不设批次级成本硬上限
    max_attempts: int = Field(default=3, ge=1)
    gpu_queue: int = Field(default=2, ge=0)
    cpu_queue: int = Field(default=8, ge=0)
    backoff_initial: float = Field(default=1.0, ge=0)
    backoff_max: float = Field(default=30.0, ge=0)
    retry_jitter: float = Field(default=0.1, ge=0)
    stage_timeout_seconds: float = Field(default=300.0, gt=0)
    run_deadline_seconds: float = Field(default=900.0, gt=0)
    max_batch_items: int = Field(default=1000, ge=1)
    queue_max_pending: int = Field(default=1000, ge=1)
    worker_poll_seconds: float = Field(default=1.0, gt=0)

    model_config = ConfigDict()  # 允许 from 配置文件/环境构造，字段名稳定。
