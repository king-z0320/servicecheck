from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BatchFileStatus(str, Enum):
    PENDING = "PENDING" # 等待处理
    RUNNING = "RUNNING" # 正在处理
    INTERRUPTED = "INTERRUPTED" # 处理被中断，可能是系统异常或人工干预
    DONE = "DONE" # 处理完成
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
    status: str = "PENDING"  # PENDING / RUNNING / DONE / FAILED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float = 0.0
    attempts: int = 0
    error: str | None = None


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

    cpu_workers: int = 4
    gpu_workers: int = 1
    llm_rpm: int = 60
    llm_cost_budget: float | None = None  # None = 不设批次级成本硬上限
    max_attempts: int = 3
    gpu_queue: int = 2
    cpu_queue: int = 8

    model_config = ConfigDict()  # 允许 from 配置文件/环境构造，字段名稳定。
