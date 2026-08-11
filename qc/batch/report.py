from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from qc.batch.models import PIPELINE_STAGES, StageName
from qc.batch.store import BatchStore


def render_progress(store: BatchStore, batch_id: str) -> str:
    files = store.list_files(batch_id)
    summary = store.batch_summary(batch_id)
    by_status = summary["by_status"]
    total = summary["total"]

    dead_letters = [f for f in files if f["status"] == "DEAD_LETTER"]
    reasons = Counter((f["failed_reason"] or "未知") for f in dead_letters)
    top_reasons = reasons.most_common(3)

    lines = [f"批次 {batch_id}", f"总计 {total}"]
    for status in ("DONE", "DEAD_LETTER", "HUMAN_REVIEW", "RUNNING", "PENDING", "INTERRUPTED"):
        if by_status.get(status):
            lines.append(f"  ├─ {status}: {by_status[status]}")

    if top_reasons:
        lines.append("失败 Top 原因：")
        for reason, count in top_reasons:
            lines.append(f"  ① {reason}（{count}）")

    # 阶段耗时均值（仅 DONE 阶段）
    durations = store.batch_durations(batch_id)
    if durations:
        stage_order = {s.value: i for i, s in enumerate(PIPELINE_STAGES)}
        ordered = sorted(
            durations.items(),
            key=lambda kv: stage_order.get(kv[0], len(stage_order)),
        )
        lines.append("阶段耗时均值：")
        for stage, avg_ms in ordered:
            avg_s = avg_ms / 1000.0
            lines.append(f"  ├─ {stage}: {avg_s:.2f}s")

    # 吞吐：DONE 数 / 已耗时分钟（仅当 started_at 存在且 elapsed > 0）
    started_iso = store.batch_started_at(batch_id)
    if started_iso:
        try:
            started_dt = datetime.fromisoformat(started_iso)
            now_dt = datetime.now(timezone.utc)
            if started_dt.tzinfo is None:
                started_dt = started_dt.replace(tzinfo=timezone.utc)
            elapsed_sec = (now_dt - started_dt).total_seconds()
            if elapsed_sec > 0:
                elapsed_min = elapsed_sec / 60.0
                done_count = by_status.get("DONE", 0)
                throughput = done_count / elapsed_min if elapsed_min > 0 else 0.0
                lines.append(
                    f"吞吐：{done_count} 文件 / {elapsed_min:.2f} 分钟 "
                    f"= {throughput:.2f} 文件/分钟"
                )
        except (ValueError, TypeError):
            pass
    # RTF（实时因子 = 处理时间 / 音频时长）需要音频时长持久化，暂未实现，deferred。

    return "\n".join(lines)
