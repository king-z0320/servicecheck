from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

from qc.batch.models import FileRecord


def compute_idempotency_key(source_uri: str, call_id: str | None = None) -> str:
    """优先用 callId 做幂等键；缺失时用 source_uri 字符串哈希前 16 位。

    注意：缺失 callId 时按"路径字符串"哈希，而非文件字节内容——因此同一文件改名/
    换位置会得到不同键。批次内应以稳定 callId 为主；POC 目录场景无 callId 时
    退化为路径哈希，仅保证"同批次同一路径不重复"，不保证内容相同的文件去重。
    """
    base = f"call:{call_id}" if call_id else f"uri:{source_uri}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


@runtime_checkable
class IngestSource(Protocol):
    def discover(self) -> list[FileRecord]: ...


class DirectorySource:
    """扫描本地/挂载目录下的音频文件（POC 默认适配器）。"""

    def __init__(self, root: str | Path, pattern: str = "*.m4a"):
        self.root = Path(root)
        self.pattern = pattern

    def discover(self) -> list[FileRecord]:
        records: list[FileRecord] = []
        for path in sorted(self.root.glob(self.pattern)):
            records.append(
                FileRecord(
                    source_uri=str(path),
                    idempotency_key=compute_idempotency_key(str(path)),
                )
            )
        return records
