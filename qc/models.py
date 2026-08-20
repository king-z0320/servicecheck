from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from qc.errors import AnalysisError


class EventType(str, Enum): # 事件类型，分为以下几种：
    REPAYMENT_DISPUTE = "REPAYMENT_DISPUTE"  # 还款争议
    DEBT_DENIAL = "DEBT_DENIAL" # 债务否认
    AMOUNT_DISPUTE = "AMOUNT_DISPUTE" # 金额争议
    FINANCIAL_HARDSHIP = "FINANCIAL_HARDSHIP"# 财务困难
    COMPLAINT_INTENT = "COMPLAINT_INTENT" # 投诉意图
    STOP_CONTACT_REQUEST = "STOP_CONTACT_REQUEST" # 停止联系请求
    THIRD_PARTY_CONTACT = "THIRD_PARTY_CONTACT" # 第三方联系
    THREAT_OR_COERCION = "THREAT_OR_COERCION" # 威胁或强迫
    EMOTIONAL_ESCALATION = "EMOTIONAL_ESCALATION" # 情绪升级
"""
当前有明确违规判定代码的主要是：
REPAYMENT_DISPUTE
THREAT_OR_COERCION
THIRD_PARTY_CONTACT
"""

class ClaimFactStatus(str, Enum):  #事实状态：
    NOT_CHECKED = "NOT_CHECKED" # 未检查，没有核实业务事实
    UNVERIFIED = "UNVERIFIED"# 未验证，没有证据证明业务事实


class ReviewDisposition(str, Enum):  #质检结果 disposition
    AUTO_PASS = "AUTO_PASS" # 自动通过
    AUTO_VIOLATION = "AUTO_VIOLATION" # 自动违规
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED" # 需要人工复核


class ExecutionStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class TranscriptTurn(BaseModel):  #一轮对话
    turnId: str  #这句话的唯一编号
    speaker: str  #说话人（坐席、客户或其他人）
    text: str    #这句话的内容（转写文本）
    start: float = 0.0 #在录音中的开始时间
    end: float = 0.0 #在录音中的结束时间

    @field_validator("turnId", "speaker", "text", mode="before")
    @classmethod
    def strip_required_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("must be a string")
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("start", "end")
    @classmethod
    def require_finite_time(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("time must be finite")
        return value

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.start < 0:
            raise ValueError("start must be greater than or equal to zero")
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class EvidenceRef(BaseModel): # 证据引用
    sourceType: Literal["TRANSCRIPT", "KNOWLEDGE", "ACTION_AUDIT"] #Literal类型表示sourceType只能是这三种之一
    sourceId: str
    excerpt: str = ""


class QualityEvent(BaseModel): # 大模型提取的质检事件
    eventId: str #事件的唯一编号
    type: EventType #事件的九种类型
    statement: str #对事件的简短概括
    turnIds: list[str] # 相关的对话轮编号列表
    confidence: float = Field(ge=0.0, le=1.0) #大模型对事件提取结果的置信度，范围是 0～1。
    ambiguous: bool = False #事件是否存在歧义。为 true 时，系统通常会启动 Agent Loop 补充证据。


class KnowledgeHit(BaseModel): #RAG 检索到的知识命中结果
    documentId: str #知识文档的唯一编号，用来追溯具体制度、规则或案例。
    category: Literal["RULE", "POLICY", "GOOD_CASE", "BAD_CASE"] #知识文档的类别，分为规则、政策、正面案例和负面案例。
    title: str #知识文档的标题，用于前端展示和人工复核
    content: str #命中的知识文档的内容，用于人工复核和 Agent Loop 补充证据
    version: str #文档版本，防止以后制度发生变化时无法追溯。
    score: float = Field(ge=0.0, le=1.0) #大模型对知识命中结果的置信度，范围是 0～1。 Field是Pydantic的一个函数，用于定义模型字段的属性和验证规则。ge=0.0表示score的最小值为0.0，le=1.0表示score的最大值为1.0。
    metadata: dict[str, Any] = Field(default_factory=dict) #检索过程中的补充信息。default_factory=dict表示metadata的默认值是一个空字典。


class AuditSnapshot(BaseModel): #外部业务系统快照
    callId: str #被查询的通话编号
    crmSummary: str | None = None #坐席在 CRM 中填写的通话小结。
    disputeTicketCreated: bool | None = None #是否已经创建还款争议工单。
    followUpType: str | None = None #通话结束后创建的跟进任务类型。
    actions: list[dict[str, Any]] = Field(default_factory=list) #坐席在业务系统中执行过的操作记录。
    errors: list[AnalysisError] = Field(default_factory=list)  #查询外部系统时发生的错误信息。
"""
followUpType 的可能值：
CONTINUE_COLLECTION    继续催收
MANUAL_REVIEW          人工复核
VERIFY_REPAYMENT       核验还款
NO_FOLLOW_UP           无后续任务
"""
#存在 errors 时，系统不能把“没有查到”当成“确实没有”，通常会启动 Agent Loop 或转人工复核。
class Violation(BaseModel): #一条违规结论
    eventId: str | None = None #关联的后端生成事件编号；历史数据允许为空
    ruleId: str #违规规则的编号
    ruleName: str #规则的中文名称
    penalty: int #违规的扣分值
    evidenceTurnIds: list[str] #证明该违规的原始对话编号
    knowledgeDocumentIds: list[str] #证明该违规的知识文档编号
    explanation: str #对违规结论的简短解释
    suggestion: str #对坐席违规行为的改进建议


class BusinessFact(BaseModel): #业务事实边界
    status: ClaimFactStatus = ClaimFactStatus.NOT_CHECKED
    note: str = "本次为通话行为质检，不判断客户是否实际结清。"
#这个项目只检查坐席的沟通和操作是否合规，不负责判断客户是否真的还清欠款。

class QualityReport(BaseModel): #完整的质检报告
    callId: str #这份质检报告属于哪一份通话
    score: int = 100 #当前通话的质检分数，满分 100 分。每条违规结论会扣除相应的分数。
    events: list[QualityEvent] = Field(default_factory=list) #从通话中识别出来的质检事件列表
    violations: list[Violation] = Field(default_factory=list) #从通话中识别出来的违规结论列表
    knowledgeHits: list[KnowledgeHit] = Field(default_factory=list) #RAG检索到的知识命中结果列表
    auditSnapshot: AuditSnapshot | None = None #外部业务系统快照，包含 CRM、争议工单和坐席操作等信息
    businessFact: BusinessFact = Field(default_factory=BusinessFact) #业务事实边界，说明本次质检是否判断客户是否实际结清
    disposition: ReviewDisposition = ReviewDisposition.AUTO_PASS #质检结果 disposition，默认是自动通过
    summary: dict[str, Any] = Field(default_factory=dict) #质检报告的摘要信息，包含质检分数、违规结论数量、知识命中数量等信息。default_factory=dict表示summary的默认值是一个空字典。


class AnalysisRequest(BaseModel): #大模型的一次质检请求
    caseId: str #案件编号，用来标识一次催收案件。例如同一个客户的一笔逾期业务可以对应一个案件。
    callId: str #通话编号，用来标识某一次具体电话。查询 CRM、争议工单和坐席操作时使用这个编号。一个案件可能有多个电话
    callStartedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) #通话开始时间。RAG 和规则模块可以根据它判断某条制度在通话发生时是否已经生效。
    transcript: list[TranscriptTurn] = Field(min_length=1) #通话转写

    @field_validator("caseId", "callId", mode="before")
    @classmethod
    def strip_required_id(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("must be a string")
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("callStartedAt")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("callStartedAt must include timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_transcript_references(self):
        turn_ids = [turn.turnId for turn in self.transcript]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("transcript turnId values must be unique")
        starts = [turn.start for turn in self.transcript]
        if starts != sorted(starts):
            raise ValueError("transcript must be ordered by start time")
        return self


class AgentTraceEvent(BaseModel): #大模型质检过程中的事件追踪,Agent的一步运行轨迹
    iteration: int #大模型质检过程中的迭代次数，从 1 开始计数。每次 Agent Loop 都会增加一次迭代。
    phase: Literal["PLAN", "ACT", "OBSERVE", "EVALUATE", "REPLAN", "FINALIZE"] #大模型质检过程中的阶段，分为计划、执行、观察、评估、重新计划和最终化六个阶段。
    message: str #当前步骤的简短说明
    details: dict[str, Any] = Field(default_factory=dict) #当前步骤的完整结构化信息
"""

"""

class AnalysisResult(BaseModel): #API 返回给前端的质检结果
    runId: str #本次质检运行的唯一编号，可以用来查询 SQLite 中保存的结果。
    status: ExecutionStatus #本次质检的执行状态，与业务处置分离。
    loopUsed: bool #本次质检是否启动了agent loop
    loopReason: str | None = None #为什么要启动agent loop，通常是因为大模型在初次分析时没有足够的证据来判断质检事件是否违规。
    report: QualityReport | None = None #本次质检的最终报告，包含质检分数、违规结论、知识命中和外部业务系统快照等信息。
    trace: list[AgentTraceEvent] = Field(default_factory=list) #本次质检的事件追踪，包含大模型在每次迭代中的计划、执行、观察和评估等信息。
    errors: list[AnalysisError] = Field(default_factory=list)
    reviewTask: dict[str, Any] | None = None
