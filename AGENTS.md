# AGENTS.md

## 1. 适用范围与优先级

本文件适用于整个仓库：

```text
E:\客服质检agent项目\serviceCheck
```

本文件中的“必须”“禁止”“不得”均为硬性约束。

若用户当前指令与本文件冲突，必须先向用户指出冲突并等待确认，不能自行放宽约束。

## 2. 工作原则

1. 回答和实施前，先检查需求是否存在错误前提、概念混淆、信息缺失或未经核实的假设。
2. 不自动迎合用户。发现错误、风险或证据不足时必须直接说明。
3. 明确区分：
   - 已从源码、命令或文件核实的事实；
   - 基于现有信息的合理推测；
   - 主观建议；
   - 当前无法核实的内容。
4. 不得伪造测试结果、运行结果、性能数据、文件状态或外部服务状态。
5. 真实测试失败、被跳过或被人工中断时，必须按实际情况记录，不能用 Fake 测试冒充真实通过。
6. 修改功能时默认采用测试先行：
   - 先新增能够暴露旧问题的失败测试；
   - 确认测试按预期失败；
   - 再修改生产代码；
   - 最后运行定向测试和相关回归测试。

## 3. 允许写入的位置

### 3.1 项目写入根目录

除第 3.2 节指定的 Conda 环境，以及第 4 节允许的 Docker、操作系统和第三方工具运行数据外，Agent 主动创建、修改、下载、解压或生成的项目资产只允许位于：

```text
E:\客服质检agent项目\serviceCheck
```

不得自行把项目代码、文档、测试产物、业务快照、密钥或长期模型写到其他磁盘或用户目录。Docker、操作系统和第三方工具确实需要的运行数据可以使用其系统默认位置，但不得借此把项目资产迁出项目根目录。

### 3.2 唯一允许使用的 Python 环境

本项目已经存在用户管理的 Conda 环境：

```text
servicecheck
```

只允许使用该 Conda 环境运行 Python、pip、pytest、FunASR、ModelScope、Hugging Face、Sentence Transformers 和项目脚本。

允许因用户要求安装项目依赖而修改这个既有 Conda 环境本身。

如果无法确认 `servicecheck` 环境的实际解释器或状态，必须停止并询问用户。

## 4. 路径与 C 盘使用规则

项目文件、测试产物、长期模型权重和可重新生成的缓存默认优先写入项目 E 盘目录。用户明确允许必要时使用 C 盘，因此不再把 C 盘作为绝对禁写区域。

路径优先级：

1. 项目代码、文档、测试产物和业务快照必须位于项目根目录：
   `E:\客服质检agent项目\serviceCheck`。
2. 长期模型权重必须位于项目内 `model_store/`，不得因方便改存到临时缓存目录。
3. 临时文件、pytest、pip、Conda 和通用缓存优先使用项目内 `.runtime/`。
4. Docker Desktop、操作系统或第三方工具确实需要写入 C 盘时可以继续执行；执行前应说明实际用途，避免把项目数据、密钥、完整客户数据或可长期保留模型主动放入 C 盘。
5. 仍然禁止 `pip install --user` 和无法确认归属的全局安装；pip 必须通过 `servicecheck` 环境调用。
6. 删除、覆盖或移动用户数据仍需遵守本文件的破坏性操作规则；允许使用 C 盘不等于允许清理 C 盘。

## 5. 临时数据与持久模型必须分离

### 6.1 可删除的临时目录

以下目录只保存可重新生成的数据：

```text
E:\客服质检agent项目\serviceCheck\.runtime
```

建议结构：

```text
.runtime/
├── tmp/
├── pytest/
├── pip-cache/
├── conda-pkgs/
└── generic-cache/
```

`.runtime/` 中可以保存：

- Python 和系统临时文件；
- pytest 临时目录；
- pip 下载缓存；
- Conda 包缓存；
- 解压目录；
- 其他可以安全重新生成的中间产物。

不得把需要长期保留的模型权重放入 `.runtime/`。

### 6.2 长期保留的模型目录

所有模型权重和模型快照必须保存到：

```text
E:\客服质检agent项目\serviceCheck\model_store
```

建议结构：

```text
model_store/
├── modelscope/
├── huggingface/
├── sentence-transformers/
├── torch/
├── README.md
└── models.lock.json
```

规则：

1. `model_store/` 是长期本地资产，不是普通临时缓存。
2. 禁止自动删除、清空、移动或覆盖 `model_store/`。
3. 删除其中任何内容前，必须列出精确绝对路径并获得用户明确授权。
4. 下载模型时应记录模型 ID、来源、revision/版本和必要的校验信息。
## 6. 运行命令前必须设置的路径

执行 pip、pytest、Conda 安装、FunASR、ModelScope、Hugging Face、Sentence Transformers、Torch 或其他可能产生缓存和临时文件的工具前，必须在同一个 PowerShell 进程中先设置：

```powershell
$projectRoot = "E:\客服质检agent项目\serviceCheck"

# 可安全删除的临时数据
$env:TEMP = "$projectRoot\.runtime\tmp"
$env:TMP = "$projectRoot\.runtime\tmp"
$env:PIP_CACHE_DIR = "$projectRoot\.runtime\pip-cache"
$env:CONDA_PKGS_DIRS = "$projectRoot\.runtime\conda-pkgs"
$env:XDG_CACHE_HOME = "$projectRoot\.runtime\generic-cache"

# 需要长期保留的模型
$env:MODELSCOPE_CACHE = "$projectRoot\model_store\modelscope"
$env:HF_HOME = "$projectRoot\model_store\huggingface"
$env:HF_HUB_CACHE = "$projectRoot\model_store\huggingface\hub"
$env:SENTENCE_TRANSFORMERS_HOME = "$projectRoot\model_store\sentence-transformers"
$env:TORCH_HOME = "$projectRoot\model_store\torch"
```

要求：

1. 所需目录只能创建在项目根目录内。
2. 环境变量必须在启动 Python/Conda/pytest 之前设置。
3. 不能只把配置写在文档里而不实际应用。
4. 设置后应优先确认这些项目缓存路径位于 E 盘；若工具仍使用 C 盘，要核实它写入的是系统/运行数据还是项目资产。
5. 如果第三方库把项目数据、密钥或长期模型写入未授权位置，必须停止并先调查它实际使用的路径。

## 7. pytest 和真实模型测试

1. pytest 必须使用项目内临时目录：

   ```powershell
   conda run -n servicecheck python -m pytest --basetemp="E:\客服质检agent项目\serviceCheck\.runtime\pytest" ...
   ```

2. 默认自动测试必须离线，不得自动下载或加载真实模型。
3. `rag_model`、`live_llm`、`live_audio` 必须由用户明确要求或为已授权验收所必需时显式运行。

## 8. pip、Conda 和模型下载规则

1. pip 只能通过 Conda 环境 `servicecheck` 的解释器调用：

   ```powershell
   conda run -n servicecheck python -m pip ...
   ```

2. 禁止直接调用无法确认归属的 `python`、`pip` 或 `pytest`。
3. 运行安装前必须确认 TEMP、TMP、PIP_CACHE_DIR 和 CONDA_PKGS_DIRS 已指向项目目录。
4. 模型下载后必须核实最终路径位于 `model_store/`，不能只相信配置变量。
5. 发现工具向 C 盘写入时，应记录用途和路径；若属于 Docker/系统运行所需可以继续，若是项目资产、密钥或长期模型则必须停止并调整路径。

## 9. 文件删除与破坏性操作

1. 没有用户明确授权时，不得删除任何非临时文件。
2. 用户明确要求删除时，只能删除用户点名的精确目标。
3. 删除前必须只读核实：
   - 绝对解析路径；
   - 文件或目录类型；
   - 是否为符号链接或重解析点；
   - 是否位于用户授权范围。
4. Windows 删除必须使用 PowerShell 原生命令和 `-LiteralPath`，不得把路径跨 shell 拼接后删除。
5. 删除后只检查目标是否仍存在，不得顺手清理相邻缓存或父目录。
6. 删除完成后必须告诉用户删除了什么、是否可恢复。

## 10. 项目文件修改规则

1. 修改前先阅读相关源码、测试和设计文档，不得只凭文件名猜测。
2. 使用 `apply_patch` 编辑源码和文档。
3. 不覆盖用户已有的无关修改。
4. 不得为了让测试通过而降低 Fail Closed、RAG 支持或 QualityGate 标准。

## 11. 遇到不确定情况时

以下情况必须暂停并询问用户：

- 无法确认 Conda `servicecheck` 的解释器路径；
- 模型下载路径不受上述变量控制；
- 需要在项目目录和既有 `servicecheck` 环境之外写入项目资产，且不属于第 4 节允许的 Docker、操作系统或第三方工具运行数据；
- 需要删除未被用户精确点名的文件；
- 继续操作可能覆盖用户数据。

不得以“之后可以清理”为理由先违反约束。
