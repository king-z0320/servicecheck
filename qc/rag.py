from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from qc.models import EventType, KnowledgeHit
from qc.errors import AnalysisError, ErrorStage, PipelineFailure
from qc.observability.tracing import traced


class Embedder(Protocol):
    def encode(self, texts, normalize_embeddings=True): ...


def _tokenize(text: str) -> set[str]:
    """极简中英混合分词：连续中文单字 + 英文/数字词。非生产分词器。"""
    text = text.lower()
    parts = re.findall(r"[a-z0-9_]+|[一-鿿]", text)
    return {p for p in parts if p.strip()}


class KnowledgeIndex:
    """本地混合检索：元数据过滤 + 稠密向量 + 词面重合（BM25 风格稀疏信号）。"""

    def __init__(
        self,
        root: str | Path,
        embedder: Embedder | None = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self.root = Path(root)
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.documents: list[dict] = []
        self.vectors = None
        self.index_version: str | None = None
        self._token_sets: list[set[str]] = []

    def _default_embedder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            "BAAI/bge-small-zh-v1.5",
            local_files_only=True,
        )

    def build(self) -> None:  #这个函数会从本地文件系统加载知识文档，构建知识索引，包括稠密向量和稀疏信号。
        documents = []
        for path in sorted((self.root / "policies").glob("*.md")):
            first, body = path.read_text(encoding="utf-8").split("\n", 1)
            documents.append({**json.loads(first), "content": body.strip()})
        for path in (
            self.root / "cases" / "good_cases.json",
            self.root / "cases" / "bad_cases.json",
            self.root / "cases" / "boundary_cases.json",
        ):
            if path.exists():
                documents.extend(json.loads(path.read_text(encoding="utf-8")))
        # 规则库也进入可检索知识（category=RULE）
        rules_path = self.root / "rules" / "quality_rules.json"
        source_rule_ids: dict[str, list[str]] = {}
        if rules_path.exists():
            rules = json.loads(rules_path.read_text(encoding="utf-8"))
            for rule in rules:
                source_rule_ids.setdefault(rule["sourceDocumentId"], []).append(
                    rule["ruleId"]
                )
                event_types = rule.get("eventTypes") or []
                # 无事件类型的通用规则：挂到空检索时不命中具体事件；跳过或复制
                if not event_types:
                    continue
                for et in event_types:
                    rag_body = rule.get("ragText") or rule.get("description", "")
                    documents.append(
                        {
                            "documentId": rule["ruleId"],
                            "title": f"规则{rule['ruleId']} {rule['name']}",
                            "category": "RULE",
                            "version": rule.get("version", "1.0"),
                            "effectiveFrom": rule.get(
                                "effectiveFrom", "2025-01-01T00:00:00Z"
                            ),
                            "effectiveTo": rule.get("effectiveTo"),
                            "eventType": et,
                            "content": rag_body,
                            "sourceDocumentId": rule.get("sourceDocumentId"),
                            "relatedRuleIds": [rule["ruleId"]],
                        }
                    )
        for document in documents:
            related = source_rule_ids.get(document["documentId"])
            if related:
                document["relatedRuleIds"] = related
        self.documents = documents
        self.embedder = self.embedder or self._default_embedder()
        corpus = [
            document["title"] + "\n" + document["content"]
            for document in documents
        ]
        self.vectors = np.asarray( #asarray()函数将输入的列表或数组转换为NumPy数组。这里将嵌入向量转换为NumPy数组，以便后续进行向量计算和相似度搜索。
            self.embedder.encode(corpus, normalize_embeddings=True),
            dtype=float,
        )
        self._token_sets = [_tokenize(text) for text in corpus]
        blob = json.dumps(documents, ensure_ascii=False, sort_keys=True)
        self.index_version = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def _sparse_score(self, query: str, doc_tokens: set[str]) -> float:
        q = _tokenize(query)
        if not q or not doc_tokens:
            return 0.0
        overlap = len(q & doc_tokens)
        return overlap / max(len(q), 1)

    def search(
        self,
        query: str,
        event_type: EventType,
        at_time: datetime,
        top_k: int = 5,
    ) -> list[KnowledgeHit]:
        with traced("rag", event_type=event_type.value, top_k=top_k, index_version=self.index_version or "unknown"):
            return self._search(query, event_type, at_time, top_k)

    def _search(
        self,
        query: str,
        event_type: EventType,
        at_time: datetime,
        top_k: int = 5,
    ) -> list[KnowledgeHit]:
        if self.vectors is None:
            raise PipelineFailure(
                AnalysisError(
                    code="RAG_INDEX_NOT_BUILT",
                    stage=ErrorStage.RAG,
                    message="知识索引尚未构建",
                    retryable=False,
                    attempts=0,
                )
            )
        if at_time.tzinfo is None or at_time.utcoffset() is None:
            raise ValueError("at_time must include timezone")
        at_time = at_time.astimezone(timezone.utc)
        # 查询改写：带上事件类型，提升与制度/案例对齐
        enriched_query = f"事件类型={event_type.value}; 问题={query}"
        query_vector = np.asarray(
            self.embedder.encode(
                [enriched_query],
                normalize_embeddings=True,
            )[0],
            dtype=float,
        )
        candidates = []
        for index, document in enumerate(self.documents):
            effective_from = datetime.fromisoformat(
                document["effectiveFrom"].replace("Z", "+00:00")
            )
            effective_to_raw = document.get("effectiveTo")
            effective_to = (
                datetime.fromisoformat(effective_to_raw.replace("Z", "+00:00"))
                if effective_to_raw
                else None
            )
            if (
                document["eventType"] != event_type.value
                or effective_from > at_time
                or (effective_to is not None and at_time >= effective_to)
            ):
                continue
            dense = float(np.dot(self.vectors[index], query_vector))
            sparse = self._sparse_score(query, self._token_sets[index])
            # 类别先验：制度 > 规则 > 案例
            prior = {
                "POLICY": 0.05,
                "RULE": 0.03,
                "GOOD_CASE": 0.01,
                "BAD_CASE": 0.01,
                "BOUNDARY_CASE": 0.0,
            }.get(document["category"], 0.0)
            score = (
                self.dense_weight * dense
                + self.sparse_weight * sparse
                + prior
            )
            candidates.append((score, dense, sparse, document))
        candidates.sort(
            key=lambda item: (
                item[0],
                item[3]["category"] == "POLICY",
                item[3]["documentId"],
            ),
            reverse=True,
        )
        return [
            KnowledgeHit(
                documentId=document["documentId"],
                category=document["category"],
                title=document["title"],
                content=document["content"],
                version=document["version"],
                score=max(0.0, min(1.0, score)),
                metadata={
                    "eventType": document["eventType"],
                    "effectiveFrom": document["effectiveFrom"],
                    "effectiveTo": document.get("effectiveTo"),
                    "sourceDocumentId": document.get("sourceDocumentId"),
                    "relatedRuleIds": document.get("relatedRuleIds", []),
                    "denseScore": round(dense, 4),
                    "sparseScore": round(sparse, 4),
                    "indexVersion": self.index_version,
                },
            )
            for score, dense, sparse, document in candidates[:top_k]
        ]
