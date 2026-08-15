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

除第 3.2 节指定的 Conda 环境外，所有由 Agent 主动创建、修改、下载、解压或生成的文件，只允许位于：

```text
E:\客服质检agent项目\serviceCheck
```

不得自行把项目文件、临时文件、测试产物、模型或缓存写到其他磁盘或用户目录。

### 3.2 唯一允许使用的 Python 环境

本项目已经存在用户管理的 Conda 环境：

```text
servicecheck
```

只允许使用该 Conda 环境运行 Python、pip、pytest、FunASR、ModelScope、Hugging Face、Sentence Transformers 和项目脚本。

允许因用户要求安装项目依赖而修改这个既有 Conda 环境本身。

如果无法确认 `servicecheck` 环境的实际解释器或状态，必须停止并询问用户。

## 4. C 盘绝对禁写规则

Agent 不得通过命令、脚本、测试、安装器或第三方库主动在 `C:\` 下创建、修改、移动、覆盖或删除任何文件和目录。

禁止写入的范围包括但不限于：

```text
C:\Users\Lenovo\AppData\Local\Temp
C:\Users\Lenovo\AppData\Local\pip
C:\Users\Lenovo\.cache
C:\Users\Lenovo\.conda
C:\Users\Lenovo\.huggingface
```

具体要求：

1. 禁止执行 `pip install --user`。
2. 禁止执行会安装到系统 Python 或用户 Python 的全局 `pip install`。
3. 禁止使用 C 盘默认 TEMP、pip cache、pytest temp、ModelScope cache 或 Hugging Face cache。
4. 即使目标是缓存或可重新下载文件，也不能自行写入 C 盘。
5. 如果某个工具无法保证不写 C 盘，必须停止执行并向用户说明，不能先运行再清理。
6. 用户要求“查看 C 盘”时，只允许只读检查；没有明确删除授权时不得清理。

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
4. 设置后应检查所有路径均不以 `C:\` 开头。
5. 如果第三方库不遵守这些环境变量，必须停止并先调查它实际使用的路径。

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
5. 发现工具向 C 盘写入时必须立即停止，不能等任务结束后再清理。

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
- 无法确认工具是否会写 C 盘；
- 模型下载路径不受上述变量控制；
- 需要在项目目录和既有 `servicecheck` 环境之外写文件；
- 需要删除未被用户精确点名的文件；
- 继续操作可能覆盖用户数据。

不得以“之后可以清理”为理由先违反约束。
