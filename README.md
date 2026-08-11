# 🎯 催收场景 AI 质检助手 - POC 演示项目

## 项目背景

### 📋 业务场景

本项目源自**银行客服中心**的实际咨询项目。在催收场景中，质检团队需要对坐席与客户的通话进行事后分析，包括：

- **合规检查**：坐席是否存在威胁、恐吓等违规话术
- **服务质量**：坐席是否使用礼貌用语、是否进行身份确认
- **业务小结**：自动提取通话中的关键信息（承诺还款、还款金额、客户情绪等）

传统质检依赖人工抽检，效率低、覆盖率不足。本项目探索使用大模型技术实现自动化质检。

### ⚠️ POC 说明

由于涉及银行客户隐私，**无法使用真实的业务录音**进行演示。项目中使用的音频样本为**公开网络资源**，仅用于验证以下技术能力：

1. **开源语音识别模型**的中文识别效果
2. **说话人分离**在单声道录音中的表现
3. **语音情感识别**的可行性
4. **大语言模型 (LLM)** 用于自动化小结和质检的效果

### 🆚 实时 vs 离线

| 特性 | 本 POC 实现 | 生产环境需求 |
|------|------------|-------------|
| 处理模式 | ⏸️ **离线处理**（录音完成后分析） | ⚡ **实时流式处理** |
| 延迟 | 秒级~分钟级 | 毫秒级~秒级 |
| 使用场景 | 事后质检、抽检分析 | 实时辅助、实时告警 |
| 技术方案 | 批量识别 | 流式识别 (Streaming ASR) |

**生产环境的实时方案**需要：
- 使用 FunASR 的 `paraformer-zh-streaming` 模型进行流式识别
- WebSocket 连接实时传输音频流
- 实时情感检测与告警机制
- 分布式部署以支持高并发

---

## 技术架构

### 📊 当前质检闭环

```text
录音处理 → 质检事件提取 → 结构化规则 / 本地 RAG → 客服后处理审计 → 直接质检路径
                                                           │
复杂案件：有限 Agent Loop（最多 3 轮 / 8 次工具 / 90 秒）+ 独立 Evaluator
                                                           │
                         确定性质量门禁 → SQLite 运行状态 → 前端证据与轨迹展示
```

- `process_audio.py`：音频转换、ASR、说话人分离、情绪识别和稳定 `turnId`。
- `api_server.py`：Flask app factory、Agent API、运行查询 API 和旧端点兼容。
- `qc/`：事件提取、规则、RAG、审计适配器、直接分析、质量门禁、有限 Loop 和持久化。
- `mock_audit_server.py`：独立只读 HTTP 服务，模拟 CRM 小结、争议工单、跟进任务和坐席操作日志。
- `催收质检.html`：展示质检事件、规则/制度/案例命中、后处理审计、违规证据和可审计 Loop 轨迹。

系统只评价坐席的通话行为和通话后动作；客户“已还款/已结清”等口述始终作为 `CUSTOMER_CLAIM`，业务事实保持 `NOT_CHECKED`，本系统不替代账务系统判断余额或结清状态。

### 📁 项目结构

```
/客服质检小结
├── README.md                 # 📖 项目说明文档（本文件）
├── 开发文档.md               # 开发过程记录
├── .env                      # 环境变量配置（API Key）
│
├── process_audio.py          # 🎵 音频处理主脚本
├── api_server.py             # 🤖 Agent API 与兼容端点
├── mock_audit_server.py      # 🔎 只读 CRM/工单模拟服务
├── 催收质检.html              # 🖥️ 前端演示页面
├── qc/                       # 🧠 规则、RAG、审计、门禁、Loop、持久化
├── knowledge/                # 📚 规则、制度和 Good/Bad Case
├── tests/                    # ✅ 单元、API、前端契约和端到端测试
│
├── audio/                    # 原始音频输入目录
│   ├── audio1.m4a
│   └── audio2.m4a
│
├── processed/                # 处理后的音频输出
│   ├── audio1.wav            
│   ├── audio2.wav
│   └── process_result.json   
│
├── data/                     # 前端演示数据
│   ├── demo_data_audio1.js   
│   └── demo_data_audio2.js   

```

---

### 6. 真实场景差异说明 (单声道 vs 双轨录音)

> **⚠️ 注意**：本 POC 项目为方便演示，采用了 **单声道 (Mono)** 音频输入方案。

在实际的呼叫中心 (Call Center) 生产环境中，坐席与客户的语音流通常是物理分离的：
*   **双轨录音 (Stereo/Dual-track)**：生产环境的最佳实践。左声道为坐席，右声道为客户。这意味着**无需**使用复杂的算法去"猜测"谁在说话，直接物理上就能 100% 区分角色，且能完美处理"抢话"(Overlapping) 的情况。
*   **本 Demo 方案**：由于公开数据集多为单声道，我们使用了 **说话人分离 (Speaker Diarization)** 算法 (CAM++)。它通过分析声纹特征来"聚类"出不同的说话人。这更适用于处理存量的单声道录音，或作为双轨录音不可用时的兜底方案。

## 音频处理流程

### 🎵 Step 1: 格式转换

**输入**：`.m4a` 格式

**输出**：`.wav` 格式（16kHz，单声道）

```python
# 使用 pydub 进行格式转换
audio = AudioSegment.from_file("audio.m4a")
audio_16k = audio.set_frame_rate(16000).set_channels(1)
audio_16k.export("audio.wav", format="wav")
```

**为什么转换为 16kHz 单声道？**
- ASR 模型的标准输入格式
- 减少计算量，提高识别速度
- 人声主要频率在 300Hz-3400Hz，16kHz 采样足够

### 🎙️ Step 2: 语音识别 + 说话人分离

#### 关于声道的说明

| 理想情况 | 实际情况 |
|---------|---------|
| **双声道录音**：左声道=坐席，右声道=客户 | **单声道录音**：坐席和客户混合在一起 |
| 直接分离左右声道即可区分 | 需要使用 **说话人分离 (Speaker Diarization)** 技术 |

**本项目使用的技术方案**：

由于实际获取的录音为**单声道**，我们采用 FunASR 的 **CAM++ 说话人分离模型**：

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",    # 中文语音识别
    vad_model="fsmn-vad",     # 语音活动检测
    punc_model="ct-punc",     # 标点恢复
    spk_model="cam++",        # 说话人分离 ⭐
    device="mps",             # Apple Silicon GPU
)

result = model.generate(input="audio.wav", batch_size_s=300)
```

**输出示例**：
```json
{
  "text": "请问是杨先生吗？",
  "start": 1750,
  "end": 3305,
  "spk": 0  // 说话人 ID
}
```

#### 🔬 技术揭秘：说话人分离是如何实现的？

我们使用的是 **CAM++ (Context-Aware Masking)** 模型，这是一个专门用于声纹识别（Speaker Verification）和分离的轻量级模型。

**处理流程 (Pipeline)：**

1.  **VAD (Voice Activity Detection)**：
    *   首先使用 `fsmn-vad` 模型把音频中的"静音段"切掉，只保留有效的人声片段。
2.  **Embedding (声纹提取)**：
    *   `CAM++` 模型对每一个切分出的小段语音进行特征提取，生成一个高维向量 (Embedding)。
    *   每个人的声纹就像指纹一样，在向量空间中是独特的。
3.  **Clustering (聚类)**：
    *   算法计算这些向量之间的距离。距离近的向量（声纹相似）被归为一类（Cluster）。
    *   系统自动识别出该段音频中有 **2 个聚类中心**（即 2 个不同的说话人），分别标记为 `spk:0` 和 `spk:1`。

#### 🏷️ 业务逻辑：如何区分“谁是坐席，谁是客户”？

模型只能告诉我们“这是 A 说的”和“这是 B 说的”，但它不知道谁是坐席。我们采用了**启发式规则**进行映射：

*   **规则 1 (首位优先)**：在电话呼出场景中，通常是坐席先开口（如"喂，您好，请问是..."）。因此，我们将时间轴上**第一个出现的说话人 ID** 标记为【坐席】。
*   **规则 2 (频次辅助)**：(可选) 坐席的话术通常更密集且标准，而客户可能回答简短。

> 💡 **生产环境建议**：虽然算法可以识别，但在生产环境中，**直接利用电话软交换系统的双声道录音**（坐席左声道、客户右声道）是区分角色的最准确方式，无需任何算法介入。

### 😊 Step 3: 情感识别

使用阿里达摩院的 **emotion2vec_plus_large** 模型进行语音情感分析：

```python
emotion_model = AutoModel(
    model="iic/emotion2vec_plus_large",
    device="mps",
)

result = emotion_model.generate(
    input="audio.wav",
    granularity="utterance",
)
```

**支持的情绪类别**：

| 情绪标签 | 中文 | 前端映射 |
|---------|------|----------|
| angry | 愤怒 | angry |
| disgusted | 厌恶 | negative |
| fearful | 恐惧 | negative |
| **sad** | **难过** | **negative** |
| happy | 快乐 | positive |
| neutral | 中性 | neutral |
| surprised | 惊讶 | neutral |

**实际识别结果**：
- audio1.m4a: 难过/sad (置信度 95%)
- audio2.m4a: 难过/sad (置信度 90%)

> 这与催收场景的客户情绪特点相符。

### 📝 Step 4: 事件、RAG、审计与有限 Agent Loop

DeepSeek V3.2 via OpenRouter 首先提取带真实 `turnId` 的结构化质检事件。代码随后：
turnid：是每一轮对话的唯一编号，可以理解成“这句话的身份证”。
1. 按事件类型和通话时间过滤结构化规则； 
2. 使用本地中文 Embedding 检索制度和 Good/Bad Case，并返回来源 ID、版本、相似度和命中内容；
3. 调用独立只读 HTTP 审计服务，核验 CRM 小结、争议工单和跟进任务；（查询外部业务数据）
4. 对明确案件走直接路径，由代码读取权威扣分并计算总分；
5. 仅对歧义、规则冲突或审计数据缺失案件启动有限 Agent Loop 和独立 Evaluator；
6. 使用确定性质量门禁检查证据 ID、规则 ID、制度依据、扣分、总分和业务事实边界；（使用代码对生成的报告进行最终质量检查）
7. 将运行、外部轨迹和最终报告写入 SQLite，供 `runId` 查询和服务重启后读取。

---

## 复现指南

### 📋 环境要求

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| Python | 3.9+ | 主运行环境 |
| ffmpeg | 最新 | 音频格式转换 |
| torch | 2.0+ | 深度学习框架 |
| funasr | 1.0+ | 语音识别+情感识别 |

### 🚀 快速开始

#### 1. 安装依赖

```bash
# macOS 安装 ffmpeg
brew install ffmpeg

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate

# 安装项目依赖（质检 Agent、RAG 与测试）
pip install -r requirements.txt

# 音频处理需要时再安装
pip install pydub funasr torch torchaudio modelscope
```

#### 2. 配置环境变量

创建 `.env` 文件：

```env
openrouter_api_key=sk-or-v1-your-api-key-here
model_name=deepseek/deepseek-v3.2
```

> 💡 可从 [OpenRouter](https://openrouter.ai/) 获取 API Key

#### 3. 准备音频文件

将待处理的音频文件（.m4a, .wav, .mp3 等）放入 `audio/` 目录：

```bash

cp your_audio.m4a audio/
```

#### 4. 运行音频处理

```bash
python process_audio.py
```

**处理过程**：
1. 🔄 格式转换 (M4A → WAV)
2. 🎤 ASR 语音识别
3. 👥 说话人分离
4. 😊 情感识别
5. 📄 生成前端数据文件

**首次运行**会自动下载模型（约 3GB）：
- `paraformer-zh` (语音识别)
- `cam++` (说话人分离)
- `emotion2vec_plus_large` (情感识别)

#### 5. 启动服务

分别启动只读审计模拟服务、Agent API 和静态页面：

```bash
# 终端 A：CRM/工单/跟进任务模拟服务，端口 5002
python mock_audit_server.py

# 终端 B：质检 Agent API，端口 5001
python api_server.py

# 终端 C：前端静态服务，端口 8080
python -m http.server 8080
```

运行自动化验证：

```bash
pytest -q
python -m compileall -q api_server.py mock_audit_server.py process_audio.py qc tests
```

主要 HTTP 接口：

- `POST /api/agent/analyze`：提交带 `caseId`、`callId` 和稳定 `turnId` 的质检请求；
- `GET /api/agent/runs/<runId>`：读取 SQLite 中的运行状态、报告和外部审计轨迹；
- `POST /api/analyze`：旧前端兼容端点；
- `GET /api/health`：健康检查。

#### 批量质检

当需要一次处理大量录音（万级规模，单机三资源通道：CPU / GPU / LLM），使用 `qc.batch` 提供的批量管线。状态写入 SQLite，支持幂等摄取、断点续跑、死信隔离与导出。

启动：

```bash
python -m qc.batch ingest <batch_id> <audio_dir>   # 摄取目录，幂等去重
python -m qc.batch run <batch_id>                   # 处理（需 DeepSeek key + mock 审计）
python -m qc.batch resume <batch_id>                # 断点续跑
python -m qc.batch report <batch_id>                # 进度报表
python -m qc.batch export <batch_id> <out_dir>      # 导出 JSON + CSV
```

并发参数（`qc/batch/models.py` `BatchConfig`，默认值待音频/ASR/GPU 基线压测后校准）：

```text
cpu_workers=4  gpu_workers=1  llm_rpm=60  max_attempts=3
```

限制：

- 系统只评价客服通话行为，不判断客户是否真实结清。
- GPU 通道当前并发=1（单卡常驻模型最稳），有显存余量可调。
- 多机分片 / K8s / Celery / Redis 属后续 Level 1/2 演进，见设计文档 §8。

#### 6. 访问演示页面

打开浏览器访问：
```
http://localhost:8080/催收质检.html
```

### 📊 处理新音频

如果你有新的音频文件想要处理：

1. 将音频放入 `audio/` 目录
2. 运行 `python process_audio.py`
3. 更新 `催收质检.html` 中的数据引用

**修改前端引入**（第 260-263 行）：

```html
<script src="./data/demo_data_your_audio.js"></script>
```

**修改 MOCK_DATA**（约第 377 行）：

```javascript
'real_your_audio': typeof GENERATED_DATA_YOUR_AUDIO !== 'undefined' && GENERATED_DATA_YOUR_AUDIO ? {
    ...GENERATED_DATA_YOUR_AUDIO,
    caseInfo: {
        ...GENERATED_DATA_YOUR_AUDIO.caseInfo,
        customerName: '你的客户名称 (真实录音)',
        agentName: '你的坐席名称'
    }
} : null,
```

**修改音频文件映射**（约第 433 行）：

```javascript
const AUDIO_FILE_MAP = {
    'real_your_audio': './processed/your_audio.wav',
    // ...
};
```

---

## 技术模型说明

### 🎯 FunASR

**开发者**：阿里达摩院  
**GitHub**：https://github.com/modelscope/FunASR

| 模型 | 功能 | 参数量 |
|------|------|--------|
| Paraformer-zh | 中文语音识别 | 220M |
| FSMN-VAD | 语音活动检测 | 0.4M |
| CT-Punc | 标点恢复 | 290M |
| CAM++ | 说话人分离 | 7M |

**性能指标**（官方数据）：
- 中文识别准确率：WER < 5% (AISHELL-1)
- 说话人分离准确率：DER < 10%
- 支持方言和口音识别

### 😊 emotion2vec

**开发者**：阿里达摩院  
**模型**：emotion2vec_plus_large

**支持的情绪**：9 类（愤怒、厌恶、恐惧、快乐、中性、其他、难过、惊讶、未知）

### 🤖 大语言模型

本项目使用 **DeepSeek V3.2** via OpenRouter API：
- 中文理解能力强
- 支持结构化输出（JSON）
- 响应速度快


---

## Agent 能力与边界

当前实现采用“**工作流优先，Agent 兜底**”：

| 能力维度 | 当前实现 |
|---------|---------|
| **明确案件** | 事件 → 规则/RAG → 审计 → 确定性报告，不启动 Loop |
| **复杂案件** | Planner → 白名单工具 → 观察 → 独立 Evaluator → 有限重规划 |
| **工具调用** | 本地知识检索；只读 CRM、工单、跟进任务和操作日志查询 |
| **证据链** | 违规同时引用真实转录 `turnId` 和规则/制度 ID |
| **状态** | SQLite 保存请求、轨迹和最终结果，可由新进程实例读取 |
| **安全边界** | 不查询或修改余额，不写 CRM，不展示隐藏思维链 |

Loop 默认最多 3 轮、8 次工具调用、90 秒。规则明确且证据完整的案件不会为了“展示 Agent”而额外调用 Planner/Evaluator；证据不足、规则无命中或依赖不可恢复时转人工复核。

---

## 规则库与知识库

知识位于 `knowledge/`：

```text
knowledge/
├── rules/       结构化、带版本和生效时间的质检规则
├── policies/    合规制度与操作规范
└── cases/       Good/Bad Case
```

检索先按事件类型和通话时间做硬过滤，再使用 `BAAI/bge-small-zh-v1.5` 做本地语义排序。报告返回文档 ID、类别、版本、相似度和命中内容。扣分值只从结构化规则库读取，模型不能自由修改；RAG 无命中时不得编造规则，案件转人工复核。

当前 POC 重点实现还款争议闭环，并提供威胁话术和第三方隐私规则/制度种子。扩充规则时需要同步增加结构化规则、制度元数据与验收样例。

---

## 实时识别支持

### ❓ 当前模型是否支持实时？

**答案：是的，FunASR 原生支持实时流式识别。**

当前使用的是**非流式模型**，但 FunASR 提供了对应的流式版本：

| 模型类型 | 模型名称 | 场景 |
|---------|---------|------|
| **非流式** | `paraformer-zh` | ✅ 当前使用，离线处理 |
| **流式** | `paraformer-zh-streaming` | ⚡ 实时处理 |

### 🔧 切换到实时识别

只需修改模型配置和代码逻辑：

```python
from funasr import AutoModel

# 流式模型初始化
model = AutoModel(model="paraformer-zh-streaming")

# 流式识别示例
chunk_size = [0, 10, 5]  # 600ms 一个 chunk
cache = {}

for audio_chunk in audio_stream():
    result = model.generate(
        input=audio_chunk,
        cache=cache,
        is_final=is_last_chunk,
        chunk_size=chunk_size
    )
    print(result)  # 实时输出识别结果
```

### 🌐 实时架构方案

```
┌─────────────┐      WebSocket      ┌─────────────┐      实时输出
│  电话系统   │ ──────────────────→ │  ASR 服务   │ ──────────────→
│ (音频流)   │                      │ (Streaming) │
└─────────────┘                     └─────────────┘
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │  实时质检   │
                                   │  告警系统   │
                                   └─────────────┘
```

---

## 🎥 实时 ASR 演示 (Demo)

本项目提供了一个基于 Gradio 的实时语音识别演示，用于验证“流式”与“离线”两种模式的区别。

### 🚀 运行演示

```bash
python realtime_asr_demo.py
```

**⚠️ 注意事项 (Mac 用户必读)**：
- 由于浏览器安全限制，`localhost` 环境下麦克风权限可能会被拒绝。
- 脚本启动后会自动生成一个 `https://xxxx.gradio.live` 的公开链接。
- **请务必使用该 HTTPS 链接访问**，以确保麦克风正常工作：`http://localhost:7860/`

### 💡 两种 ASR 方案深度对比

在演示中，你可以体验到两种不同的 ASR 模式。这两种模式对应了工业界真实落地的两类场景。

| 特性 | 🔴 实时流式 (Real-time / Pseudo-Streaming) | ⏺️ 录音后分析 (Post-recording / Batch) |
| :--- | :--- | :--- |
| **演示功能** | “实时流式” Tab 页 | “录音后分析” Tab 页 |
| **技术原理** | **滑动窗口识别**：每隔几百毫秒，截取最近 N 秒的音频送入模型重新识别，实现“字一个个蹦出来”的效果。 | **全量批处理**：录音完全结束后，上传整个文件，一次性进行识别、分离、情感分析。 |
| **核心优势** | **快、即时反馈**。用户边说边出字，延迟低（毫秒级到秒级）。 | **准、信息全**。拥有全局上下文，断句更准；可叠加说话人分离、情感分析等耗时任务。 |
| **典型场景** | 1. **实时坐席辅助**：通话中实时提示话术。<br>2. **实时风控告警**：检测到客户骂人立即弹窗。<br>3. **语音输入法/字幕**：边说边上屏。 | 1. **全量质检**：每天下班后对数万通电话进行自动化打分。<br>2. **商业洞察**：分析客户整体诉求和情绪趋势。<br>3. **合规存档**：生成完美的对话文本归档。 |
| **算力消耗** | **高** (因为要重复计算重叠的音频段) | **低** (每段音频只算一次) |

> **演示中的“伪流式”黑科技**：
> 真正的工业级流式模型（如 `paraformer-streaming`）为了速度通常会牺牲一点精度。本演示采用了一种“伪流式”方案：利用 Mac 强大的 M 系列芯片性能，**高频多次调用高精度的离线模型**。这样既保留了离线模型的高精度，又实现了实时的视觉效果，非常适合 POC 演示。

---

## 待优化方向

### 🚧 当前限制

1. **通话后质检**：当前不是实时坐席辅助，业务事实只标记为 `NOT_CHECKED`。
2. **模拟后处理系统**：CRM/工单接口是独立只读演示服务，不连接真实银行系统，也不写入任何业务状态。
3. **模型凭证**：真实事件提取和 Evaluator 使用 DeepSeek 官方直连（见上文批量质检）；无凭证时自动测试使用确定性 Fake，不代表真实模型延迟。
4. **单通话范围**：单通话质检为一次请求闭环；万级录音的批量摄取、三资源通道并发、断点续跑与死信隔离已由 `qc.batch` 实现（见「批量质检」节）。多机分片与外部队列（K8s/Celery/Redis）属后续 Level 1/2 演进，当前不实现。
5. **音频局限**：公开单声道样本的说话人分离和整段情绪识别仍可能存在误差。

### 📏 当前性能基线

服务为每次运行输出 `quality_analysis_timing` JSON，包含事件提取、知识检索、审计、直接分析、Loop、持久化、LLM 请求数、Loop 轮数和总耗时。确定性、无网络验收基线为：

| 路径 | 总耗时 | Loop | LLM 请求计数 | 说明 |
|------|-------:|-----:|-------------:|------|
| 明确直接路径 | 1.9 ms | 0 轮 | 1 | 1 次事件提取；Fake RAG/审计，主要耗时为 SQLite |
| 复杂路径 | 5.2 ms | 2 轮 | 5 | 1 次事件提取 + 2 次 Planner + 2 次 Evaluator；Fake 依赖 |

这些数字只验证计时与控制流，不代表真实模型或 HTTP 延迟。真实 BGE 首次构建在本机曾测得 51.803 秒；服务应常驻索引。真实 DeepSeek 事件提取与 Planner/Evaluator 烟雾测试已通过（DeepSeek 官方直连）。批量并发参数的最终校准仍需补齐代表性录音的 ASR、GPU/CPU、真实 RAG、审计 HTTP 和 LLM 延迟基线，再决定是否升级到外部队列或分布式组件。

### 🎯 后续方向

1. 使用双声道录音或更可靠的角色映射提升证据准确性；
2. 扩充事件类型、规则、制度和验收案例覆盖；
3. 接入只读真实 CRM/工单查询适配器并保留失败降级；
4. 用代表性录音补齐 ASR/GPU/LLM 真实基线，据此校准 `qc.batch` 并发默认值，并据实时因子判断是否触发 Level 1 多机升级；
5. 在没有测量瓶颈前不预先选定 Redis、Celery、Kafka、Ray 或 Kubernetes。

---

## 协议声明

- 本项目仅用于技术演示和学习交流
- 使用的音频样本来自公开网络，不涉及真实客户数据
- 如需商业使用，请确保符合相关法律法规

---

## 联系方式

如有问题或建议，欢迎交流讨论。

---

*最后更新：2026-07-20*
1.事件（Qualityevent），一次质检的标准输入对象（AnalysisRequest），Transcipit列表都有啥区别（都在models.py中定义）