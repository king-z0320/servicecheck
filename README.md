# 催收场景 AI 质检 Agent（可信 POC）

这是一个面向催收客服通话的离线质检 POC。系统把通话转写、DeepSeek 结构化事件提取、本地 RAG、规则裁决、通话后动作审计、有界 Agent Loop、确定性质量门禁和 PostgreSQL 历史记录串成一条可复核链路。

当前定位是“求职作品集级、具有生产意识的单机 POC”，不是可直接进入金融生产环境的系统。项目不判断客户是否真实还清，也不写入 CRM、工单或账务系统。

## 1. 已实现能力

- 严格校验请求、通话时间和每轮转写，非法输入在调用模型前返回 400；
- DeepSeek 官方 API 结构化事件提取，非法输出失败关闭；
- `eventId` 由后端根据通话、事件类型和证据稳定生成，模型无权指定；
- 带版本和生效时间的结构化规则；
- `BAAI/bge-small-zh-v1.5` 本地语义检索，阈值由真实模型校准产物提供；
- 独立只读 HTTP 审计服务，分别查询 CRM 小结、争议工单、跟进任务和操作记录；
- 明确案件走确定性直接路径，歧义案件进入有限 Agent Loop；
- 所有最终报告统一通过 `QualityGate`；
- `RUNNING/COMPLETED/PARTIAL/FAILED` 状态机、结构化错误和 PostgreSQL 持久化；
- Alembic 空库初始化与真实 0001 -> 0003 数据回填升级；
- 项目内 LocalArtifactStore、音频 Range、批量检查点和导出逻辑 URI；
- 原生 HTML 坐席工作台从 API 读取案件、通话、转写、历史运行、报告和音频；
- 默认离线自动测试，以及显式运行的真实 RAG、DeepSeek、FunASR E2E。

## 2. 核心架构

```text
M4A/WAV
  -> process_audio.py：转码、FunASR、说话人分离、turnId
  -> AnalysisRequest：caseId、callId、callStartedAt、transcript
  -> QualityAnalysisService：创建 RUNNING 记录并协调完整链路
       -> EventExtractor -> DeepSeekGateway
       -> KnowledgeIndex -> 本地规则/制度/案例
       -> AuditClient -> 独立 Mock Audit HTTP
       -> DirectAnalyzer -> 初始报告
       -> 初步 QualityGate
       -> 必要时 BoundedAgentLoop + 独立 Evaluator
       -> 最终 QualityGate
  -> PostgresRunStore：终态、不可变报告、轨迹和版本信息
  -> FastAPI -> 催收质检.html（坐席真实 API；其他角色明确为演示）
```

设计原则是“工作流优先，Agent 兜底”：规则和证据明确时不为了展示 Agent 而调用 Planner；证据不足、规则冲突或审计部分失败时才进入最多 3 轮、8 次工具调用、90 秒的有限循环。

更详细的数据流见 [单通话与批量质检架构说明.md](./单通话与批量质检架构说明.md)。

## 3. 状态与业务处置

执行状态描述程序运行结果：

| 状态 | 含义 |
|---|---|
| `RUNNING` | 已创建运行，仍在执行 |
| `COMPLETED` | 链路完成且最终质量门通过 |
| `PARTIAL` | 得到可展示报告，但证据或依赖不完整，必须人工复核 |
| `FAILED` | 无法形成可信报告或发生不可恢复错误 |

业务处置描述报告能否自动落地：

| 处置 | 含义 |
|---|---|
| `AUTO_PASS` | 证据完整且无违规 |
| `AUTO_VIOLATION` | 违规证据、规则和计分均通过门禁 |
| `HUMAN_REVIEW_REQUIRED` | 不能安全自动判定 |

`PARTIAL` 不等于请求失败，但必须对应 `HUMAN_REVIEW_REQUIRED`。`FAILED` 可以没有报告。历史状态 `BLOCKED` 已从单通话结果中移除。

## 4. 证据和计分可信性

每条违规必须同时满足：

1. 指向本次报告内后端生成的 `eventId`；
2. 引用本次 transcript 中真实存在的 `turnId`；
3. 规则在 `callStartedAt` 有效，时间采用 `[effectiveFrom, effectiveTo)`；
4. 至少一个本次实际检索命中达到校准阈值；
5. 命中文档事件类型与违规事件一致；
6. 命中文档与 `ruleId` 存在明确来源关系。

知识类别可以是 `RULE`、`POLICY`、`GOOD_CASE` 或 `BAD_CASE`；案例不会仅因类别被排除，但如果没有明确规则来源关系，仍不能单独支持处罚。

扣分只从结构化规则库读取。重复的 `(eventId, ruleId)` 只计一次；不同事件即使命中同一规则，也可以分别计罚。

校准结果位于 `knowledge/rag_calibration.json`：

- embedder：`BAAI/bge-small-zh-v1.5`；
- threshold：`0.6855`；
- positiveMin：`0.706302`；
- negativeMax：`0.664665`；
- indexVersion：`8424191d7c3b`。

RAG 是“相关性检索器”，不是违规分类器。是否违规仍由事件、有效规则、审计事实和质量门共同决定。

## 5. 项目结构

```text
serviceCheck/
├── api_server.py                 FastAPI、查询/音频路由与依赖装配
├── mock_audit_server.py          独立只读审计模拟服务
├── process_audio.py              音频转码、ASR、说话人和情绪处理
├── 催收质检.html                  前端演示页
├── review.html                   独立人工复核页
├── qc/
│   ├── models.py                 输入、报告、状态等领域模型
│   ├── errors.py                 结构化错误契约
│   ├── llm_gateway.py            DeepSeek 请求、分类重试和结构校验
│   ├── event_extractor.py        事件校验、稳定 ID 和去重
│   ├── rag.py / rag_support.py   本地检索和逐违规支持验证
│   ├── rules.py                  规则有效期、去重和权威计分
│   ├── audit_client.py           四类审计资源的独立重试
│   ├── direct_analyzer.py        确定性直接分析
│   ├── agent_loop.py             有界 Planner/Executor/Evaluator
│   ├── quality_gate.py           最终可信性门禁
│   ├── postgres_run_store.py     PostgreSQL 运行、报告和工作台查询
│   ├── review_models.py          人工复核 DTO 与结果枚举
│   ├── review_service.py         复核路由、提交、冲突和幂等
│   ├── review_store.py           复核任务 PostgreSQL 存储
│   ├── database.py               SQLAlchemy Engine 与 Session
│   ├── orm_models.py             PostgreSQL ORM Schema
│   ├── artifact_store.py         LocalArtifactStore 与逻辑 URI
│   ├── run_store.py              SQLite Legacy Adapter（迁移/回归）
│   ├── service.py                状态机与总协调
│   └── batch/                    批量编排原型
├── knowledge/                    规则、制度、案例和 RAG 校准产物
├── alembic/                      PostgreSQL Schema revisions
├── tests/                        离线、PostgreSQL、金标和真实 E2E 测试
├── scripts/migrate_sqlite_to_postgres.py
├── scripts/calibrate_rag.py      真实 embedding 阈值校准脚本
├── requirements.txt              核心服务/RAG/测试依赖
├── requirements-audio.txt        FunASR 和音频可选依赖
└── pytest.ini                    marker 与默认测试配置
```

第二阶段已新增受控目录扫描、`batch_id` 异步控制面、PostgreSQL Transactional Outbox、Redis Streams 单 Worker、真实音频 Runner、检查点续跑、阶段级有限重试和死信记录。第三阶段新增 `review_tasks`/`review_revisions`、审核提交 API 和独立 `review.html`；人工决定不覆盖原始报告，也不接入金标或重跑。默认离线测试仍使用 Fake Runner；真实 FunASR/DeepSeek E2E 必须显式运行并不能由离线测试替代。

## 6. 环境与安装

本阶段实际验证环境为既有 Conda 环境 `servicecheck`，Python 3.14.6。

```powershell
conda run -n servicecheck python -m pip install -r requirements.txt
```

需要处理音频或运行真实音频 E2E 时再安装：

```powershell
conda run -n servicecheck python -m pip install -r requirements-audio.txt
```

`requirements-audio.txt` 固定配套的 `torch==2.11.0` 与 `torchaudio==2.11.0`。系统没有安装 ffmpeg 时，代码会使用 `imageio-ffmpeg` 的内置二进制直接转码，避免依赖系统 `ffprobe`。

首次运行 FunASR 会下载多个本地模型，耗时和磁盘占用取决于网络与设备。

## 7. 配置

复制 `.env.example` 为 `.env`，只在本机填写真实值；不要提交 `.env`。

```env
DEEPSEEK_API_KEY=replace-with-your-key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_TIMEOUT_SECONDS=60
AUDIT_SERVICE_URL=http://127.0.0.1:5002
DATABASE_URL=postgresql+psycopg://servicecheck@127.0.0.1:55432/servicecheck
TEST_DATABASE_URL=postgresql+psycopg://servicecheck@127.0.0.1:55432/servicecheck_test
ARTIFACT_ROOT=data/artifacts
API_CORS_ORIGINS=http://127.0.0.1:8080,http://localhost:8080
```

`RAG_MIN_SUPPORT_SCORE` 可选；未配置时读取版本化校准产物。旧的 `openrouter_api_key/model_name` 只作为过渡兼容，不是文档主路径。

## 8. 本地启动

本阶段 Redis 使用 `compose.stage2.yml` 的 Docker Compose 服务；PostgreSQL 继续使用本机实例。API、Publisher、Worker 和页面分别运行。

```powershell
# 先升级 Schema（首次和版本升级时执行）
conda run -n servicecheck alembic upgrade head

# Redis（Docker Desktop 未运行时先启动）
docker compose -f compose.stage2.yml up -d redis
docker compose -f compose.stage2.yml exec -T redis redis-cli ping

# 终端 A：只读审计模拟服务，默认 5002
conda run -n servicecheck python mock_audit_server.py

# 终端 B：质检 API，默认 5001
conda run -n servicecheck python api_server.py

# 终端 C：静态页面，默认 8080
conda run -n servicecheck python -m http.server 8080
```

浏览器访问 `http://127.0.0.1:8080/催收质检.html`。

主要接口：

- `GET /api/health`：进程级健康检查，不代表 DeepSeek 和审计服务均可用；
- `POST /api/agent/analyze`：新质检接口，必须显式携带带时区的 `callStartedAt`；
- `GET /api/agent/runs/<runId>`：读取持久化运行；
- `POST /api/analyze`：旧接口兼容层；
- `GET /api/cases`、`GET /api/cases/{caseId}`：我的案件；
- `GET /api/calls/{callId}`、`/transcript`、`/runs`、`/audio`：通话工作台；
- `GET /api/reports/{reportId}`：不可变历史报告与版本信息。
- `POST /batches`：扫描受控音频目录并异步创建批次，返回 `202 + batch_id`；不使用 `jobId`。
- `GET /batches/{batch_id}`、`GET /batches/{batch_id}/items`：轮询批次和文件状态。

## 9. 测试

默认命令完全离线，自动跳过真实模型 marker：

```powershell
conda run -n servicecheck python -m pip check
conda run -n servicecheck python -m compileall -q api_server.py mock_audit_server.py process_audio.py realtime_asr_demo.py qc tests scripts
conda run -n servicecheck python -m pytest --basetemp="E:\客服质检agent项目\serviceCheck\.runtime\pytest" -q
```

真实测试必须显式运行：

```powershell
# 真实 BGE 校准边界
$env:HF_ENDPOINT='https://hf-mirror.com'  # 官方源可用时可不设置
conda run -n servicecheck python -m pytest -o addopts= -m rag_model -q

# 真实 DeepSeek 文本 E2E
conda run -n servicecheck python -m pytest -o addopts= -m live_llm -q

# 真实 Runner：转码 -> FunASR -> 情绪识别，不调用 LLM
conda run -n servicecheck python -m pytest -o addopts= -m live_audio \
  tests/test_live_audio.py::test_real_audio_runner_completes_audio_stages_without_llm -q

# 两段真实音频 -> FunASR -> DeepSeek -> RAG -> Audit -> Gate
conda run -n servicecheck python -m pytest -o addopts= -m live_audio -q
```

真实测试缺少凭证或依赖时会明确 skip；“全部 skip”不能作为真实 E2E 通过证据。测试不会打印密钥、prompt 或模型完整原文，音频派生产物只写入 pytest 临时目录。

## 10. 音频边界

演示音频是单声道公开样本。CAM++ 只能聚类不同说话人，不能天然知道谁是坐席；当前用首位说话人等启发式映射角色。生产呼叫中心应优先使用物理分离的双轨录音。

两段原始文件受哈希保护，测试前后必须一致：

```text
audio/audio1.m4a  5DB623C054EF9611B46E56CF848DC916896AC4E8BAD099EC937078BA9462F294
audio/audio2.m4a  6E44337CD4FFE75588854504A0AA4A6634EA29B5D7066C4E52BDEC5224D2F1A6
```

## 11. 当前限制

- 单通话音频处理与质检 API 仍是两个显式步骤，不是统一异步任务；
- 审计服务是 Mock，未连接真实 CRM/工单；
- 事件枚举有九类，当前确定性违规规则主要覆盖还款争议、威胁恐吓和第三方联系；
- 情绪识别主要用于展示，不参与权威扣分；
- 默认测试不加载真实模型；真实批量 Runner 由独立 Worker 使用，真实 E2E 仍依赖本地模型、FFmpeg 和上游凭据；
- 没有鉴权、RBAC、租户隔离、限流和生产高可用部署方案；
- 主管、分析师和管理员仍是明确标注的演示视图，不代表服务端 RBAC；
- 旧批量 CLI 仍可使用 FakeAudioStageRunner 做离线开发；正式异步入口是 API + Outbox Publisher + Redis Worker；
- Redis 当前是单机 Docker 容器；尚无 CI、集中日志、指标、追踪和生产高可用部署。

第一阶段的需求、技术方案和实施证据见 `docs/第一阶段：业务后端化/`（该目录按项目约定不提交 GitHub）。

## 12. 安全声明

- 项目只用于技术演示和学习；
- 不应提交 `.env`、真实密钥、客户数据、原始通话或本地数据库；
- 客户“已还款/已结清”的口述只是待核事实，本系统不替代账务系统；
- 商业使用前必须补齐数据治理、安全审计、权限、合规和生产验收。

最后更新：2026-08-19。
