from __future__ import annotations

"""Metadata-first hybrid retrieval used by the quality analysis pipeline."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from qc.errors import AnalysisError, ErrorStage, PipelineFailure
from qc.models import EventType, KnowledgeHit
from qc.observability.tracing import traced


class Embedder(Protocol):
    def encode(self, texts, normalize_embeddings=True): ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


def _tokenize(text: str) -> list[str]:
    """Tokenize Chinese text while preserving R006/R-006 and error codes."""
    text = str(text).lower()
    protected = re.findall(r"r-?\d+|[a-z]+\d+|\d+(?:\.\d+)?%?|[a-z]+", text)
    try:
        import jieba
        words = [w.strip() for w in jieba.cut(text) if w.strip()]
    except Exception:
        words = re.findall(r"[一-鿿]+", text)
    return [item for item in protected + words if item not in {"的", "了", " ", "，", "。", "、"}]


class _BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus, self.k1, self.b = corpus, k1, b
        try:
            from rank_bm25 import BM25Okapi
            self._impl = BM25Okapi(corpus, k1=k1, b=b)
        except ImportError:
            self._impl = None
        self.avgdl = sum(map(len, corpus)) / max(len(corpus), 1)
        self.df: dict[str, int] = {}
        for row in corpus:
            for token in set(row):
                self.df[token] = self.df.get(token, 0) + 1

    def get_scores(self, query: list[str]) -> np.ndarray:
        if self._impl is not None:
            return np.asarray(self._impl.get_scores(query), dtype=float)
        scores = np.zeros(len(self.corpus), dtype=float)
        for i, row in enumerate(self.corpus):
            counts: dict[str, int] = {}
            for token in row:
                counts[token] = counts.get(token, 0) + 1
            for token in query:
                tf = counts.get(token, 0)
                if not tf:
                    continue
                df = self.df.get(token, 0)
                idf = math.log(1 + (len(self.corpus) - df + 0.5) / (df + 0.5))
                dl = len(row)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9)))
        return scores


class CrossEncoderReranker:
    """Local-only sentence-transformers CrossEncoder adapter."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", model=None):
        self.model_name, self._model = model_name, model

    def _local_source(self) -> str:
        """Prefer the project-owned ModelScope snapshot over any user cache."""
        root = Path(__file__).resolve().parents[1]
        snapshot = (
            root / "model_store" / "modelscope" / "models"
            / self.model_name.replace("/", "--") / "snapshots" / "master"
        )
        return str(snapshot) if snapshot.exists() else self.model_name

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._local_source(), local_files_only=True)
        values = self._model.predict([(query, text) for text in texts], show_progress_bar=False)
        return [float(value) for value in values]


@dataclass(frozen=True)
class RagSearchConfig:
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_candidate_limit: int = 20
    rrf_k: int = 60
    final_top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    chunk_strategy_version: str = "structured-atomic-v1"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    bm25_tokenizer_version: str = "jieba-protected-v1"

    def as_dict(self) -> dict[str, Any]:
        return {"method": "dense_bm25_rrf_rerank", "denseTopK": self.dense_top_k, "bm25TopK": self.bm25_top_k, "rrfLimit": self.rrf_candidate_limit, "rrfK": self.rrf_k, "finalTopK": self.final_top_k, "rerankerModel": self.reranker_model, "chunkStrategyVersion": self.chunk_strategy_version, "embeddingModel": self.embedding_model, "bm25TokenizerVersion": self.bm25_tokenizer_version, "fallback": "rrf"}


def rrf_merge(dense: list[tuple[str, float]], bm25: list[tuple[str, float]], *, rrf_k: int = 60, limit: int = 20) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for source, ranked in (("dense", dense), ("bm25", bm25)):
        for rank, (key, score) in enumerate(ranked, 1):
            row = values.setdefault(key, {"chunkId": key, "denseRank": None, "bm25Rank": None, "denseScore": None, "bm25Score": None, "rrfScore": 0.0})
            row[f"{source}Rank"], row[f"{source}Score"] = rank, float(score)
            row["rrfScore"] += 1 / (rrf_k + rank)
    return sorted(values.values(), key=lambda item: (-item["rrfScore"], item["chunkId"]))[:limit]


def _content_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(document.get("content", "").encode("utf-8")).hexdigest()


class StructuredChunker:
    def __init__(self, max_chars: int = 800):
        self.max_chars = max_chars

    def chunk(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        content = str(document.get("content", "")).strip()
        parts = [content[i : i + self.max_chars] for i in range(0, len(content), self.max_chars)] or [""]
        result = []
        for index, text in enumerate(parts):
            row = dict(document)
            row["content"] = text
            row["chunkId"] = f"{document['documentId']}#c{index:04d}"
            row["sourceRange"] = {"chunkIndex": index, "startChar": index * self.max_chars, "endChar": index * self.max_chars + len(text)}
            row["contentHash"] = _content_hash(row)
            result.append(row)
        return result


class KnowledgeIndex:
    """Dense + BM25 + RRF + Cross-Encoder, with hard metadata filters first."""

    def __init__(self, root: str | Path, embedder: Embedder | None = None, *, reranker: Reranker | None = None, config: RagSearchConfig | None = None, knowledge_version: str | None = None, dense_weight: float = 0.7, sparse_weight: float = 0.3):
        self.root, self.embedder, self.reranker = Path(root), embedder, reranker
        self.config = config or RagSearchConfig()
        self.knowledge_version = knowledge_version
        self.dense_weight, self.sparse_weight = dense_weight, sparse_weight
        self.documents: list[dict[str, Any]] = []
        self.vectors: np.ndarray | None = None
        self.index_version: str | None = None
        self.manifest: dict[str, Any] = {}
        self._bm25: _BM25 | None = None
        self.reranker_fallback: str | None = None
        self.pgvector_store = None

    def _default_embedder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.config.embedding_model, local_files_only=True)

    def _load_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for path in sorted((self.root / "policies").glob("*.md")):
            first, body = path.read_text(encoding="utf-8").split("\n", 1)
            documents.append({**json.loads(first), "content": body.strip()})
        for path in (self.root / "cases" / "good_cases.json", self.root / "cases" / "bad_cases.json", self.root / "cases" / "boundary_cases.json"):
            if path.exists():
                documents.extend(json.loads(path.read_text(encoding="utf-8")))
        rules_path = self.root / "rules" / "quality_rules.json"
        source_rule_ids: dict[str, list[str]] = {}
        if rules_path.exists():
            for rule in json.loads(rules_path.read_text(encoding="utf-8")):
                source_rule_ids.setdefault(rule["sourceDocumentId"], []).append(rule["ruleId"])
                for event_type in rule.get("eventTypes") or []:
                    documents.append({"documentId": rule["ruleId"], "title": f"规则{rule['ruleId']} {rule['name']}", "category": "RULE", "version": rule.get("version", "1.0"), "effectiveFrom": rule.get("effectiveFrom", "2025-01-01T00:00:00Z"), "effectiveTo": rule.get("effectiveTo"), "eventType": event_type, "content": rule.get("ragText") or rule.get("description", ""), "sourceDocumentId": rule.get("sourceDocumentId"), "relatedRuleIds": [rule["ruleId"]]})
        for document in documents:
            related = source_rule_ids.get(document["documentId"])
            if related:
                document["relatedRuleIds"] = related
            document["documentStatus"] = document.get("documentStatus", "published")
        return documents

    def build(self) -> None:
        source_documents = self._load_documents()
        self.documents = [chunk for document in source_documents for chunk in StructuredChunker().chunk(document)]
        self.embedder = self.embedder or self._default_embedder()
        corpus = [document["title"] + "\n" + document["content"] for document in self.documents]
        self.vectors = np.asarray(self.embedder.encode(corpus, normalize_embeddings=True), dtype=float)
        self._bm25 = _BM25([_tokenize(text) for text in corpus])
        blob = json.dumps(source_documents, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        chunks_blob = json.dumps(self.documents, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.index_version = hashlib.sha256(blob.encode()).hexdigest()[:12]
        self.knowledge_version = self.knowledge_version or f"kv-{self.index_version}"
        self.manifest = {"knowledgeVersion": self.knowledge_version, "sourceHash": hashlib.sha256(blob.encode()).hexdigest(), "chunkStrategy": {"version": self.config.chunk_strategy_version, "maxChars": 800}, "metadataSchemaVersion": "v1", "embedding": {"model": self.config.embedding_model, "dimension": int(self.vectors.shape[1]) if self.vectors.ndim == 2 else None, "embeddingMatrixHash": hashlib.sha256(self.vectors.tobytes()).hexdigest()}, "bm25": {"tokenizerVersion": self.config.bm25_tokenizer_version, "k1": 1.5, "b": 0.75}, "retrievalConfig": self.config.as_dict(), "chunkCount": len(self.documents), "indexHash": hashlib.sha256(chunks_blob.encode()).hexdigest()}
        self.manifest["manifestHash"] = hashlib.sha256(json.dumps(self.manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _eligible(self, document: dict[str, Any], event_type: EventType, at_time: datetime, rule_relation: str | None) -> bool:
        if document.get("documentStatus") != "published" or document.get("eventType") != event_type.value:
            return False
        try:
            start = datetime.fromisoformat(str(document["effectiveFrom"]).replace("Z", "+00:00"))
            end_raw = document.get("effectiveTo")
            end = datetime.fromisoformat(str(end_raw).replace("Z", "+00:00")) if end_raw else None
        except (KeyError, TypeError, ValueError):
            return False
        if start > at_time or (end is not None and at_time >= end):
            return False
        return not rule_relation or rule_relation in set(document.get("relatedRuleIds") or [])

    def search(self, query: str, event_type: EventType, at_time: datetime, top_k: int = 5, *, rule_relation: str | None = None, knowledge_version: str | None = None) -> list[KnowledgeHit]:
        with traced("rag", event_type=event_type.value, top_k=top_k, index_version=self.index_version or "unknown"):
            return self._search(query, event_type, at_time, top_k, rule_relation=rule_relation, knowledge_version=knowledge_version)

    def _search(self, query: str, event_type: EventType, at_time: datetime, top_k: int = 5, *, rule_relation: str | None = None, knowledge_version: str | None = None) -> list[KnowledgeHit]:
        if self.vectors is None or self._bm25 is None:
            raise PipelineFailure(AnalysisError(code="RAG_INDEX_NOT_BUILT", stage=ErrorStage.RAG, message="知识索引尚未构建", retryable=False, attempts=0))
        if at_time.tzinfo is None or at_time.utcoffset() is None:
            raise ValueError("at_time must include timezone")
        at_time = at_time.astimezone(timezone.utc)
        version = knowledge_version or self.knowledge_version
        eligible = [index for index, document in enumerate(self.documents) if (not version or version == self.knowledge_version) and self._eligible(document, event_type, at_time, rule_relation)]
        query_vector = np.asarray(self.embedder.encode([f"事件类型={event_type.value}; 问题={query}"], normalize_embeddings=True)[0], dtype=float)
        dense_scores = {index: float(np.dot(self.vectors[index], query_vector)) for index in eligible}
        if self.pgvector_store is not None and version:
            rows = self.pgvector_store.search_scored(
                query_vector.tolist(),
                knowledge_version=version,
                event_type=event_type.value,
                at_time=at_time,
                top_k=self.config.dense_top_k,
                rule_relation=rule_relation,
            )
            by_chunk = {document["chunkId"]: index for index, document in enumerate(self.documents)}
            dense_order = []
            dense_scores = {}
            for row, distance in rows:
                index = by_chunk.get(row.chunk_id)
                if index is not None:
                    dense_order.append(index)
                    dense_scores[index] = 1.0 - float(distance)
        else:
            dense_order = sorted(eligible, key=lambda index: (-dense_scores[index], self.documents[index]["chunkId"]))[: self.config.dense_top_k]
        bm_scores = self._bm25.get_scores(_tokenize(query))
        bm_order = sorted(eligible, key=lambda index: (-float(bm_scores[index]), self.documents[index]["chunkId"]))[: self.config.bm25_top_k]
        merged = rrf_merge([(self.documents[index]["chunkId"], dense_scores[index]) for index in dense_order], [(self.documents[index]["chunkId"], float(bm_scores[index])) for index in bm_order], rrf_k=self.config.rrf_k, limit=self.config.rrf_candidate_limit)
        by_id = {document["chunkId"]: document for document in self.documents}
        candidates = [(by_id[row["chunkId"]], row) for row in merged]
        fallback = None
        try:
            reranker = self.reranker or CrossEncoderReranker(self.config.reranker_model)
            self.reranker = reranker
            scores = reranker.score(query, [document["title"] + "\n" + document["content"] for document, _ in candidates])
            candidates = [(document, {**row, "rerankScore": scores[index]}) for index, (document, row) in enumerate(candidates)]
            candidates.sort(key=lambda item: (-item[1]["rerankScore"], item[0]["chunkId"]))
            reranker_status = "available"
        except Exception as exc:
            fallback = f"{type(exc).__name__}:{exc}"
            self.reranker_fallback = fallback
            candidates.sort(key=lambda item: (-(item[1]["rrfScore"] + {"POLICY": 0.01, "RULE": 0.006, "GOOD_CASE": 0.002, "BAD_CASE": 0.002}.get(item[0].get("category"), 0.0)), item[0]["chunkId"]))
            reranker_status = "unavailable"
        hits = []
        for rank, (document, row) in enumerate(candidates[: min(top_k, self.config.final_top_k)], 1):
            final_score = float(row.get("rerankScore", row["rrfScore"]))
            hits.append(KnowledgeHit(documentId=document["documentId"], category=document["category"], title=document["title"], content=document["content"], version=document["version"], score=max(0.0, min(1.0, final_score)), chunkId=document["chunkId"], documentVersion=document["version"], sourceRange=document.get("sourceRange"), knowledgeVersion=version, contentHash=document["contentHash"], retrievalMethod="hybrid_rrf_rerank" if reranker_status == "available" else "hybrid_rrf", denseScore=row.get("denseScore"), bm25Score=row.get("bm25Score"), rrfScore=row.get("rrfScore"), rerankScore=row.get("rerankScore"), rank=rank, rerankerStatus=reranker_status, fallback="rrf" if fallback else None, metadata={"eventType": document["eventType"], "effectiveFrom": document["effectiveFrom"], "effectiveTo": document.get("effectiveTo"), "documentStatus": document["documentStatus"], "sourceDocumentId": document.get("sourceDocumentId"), "relatedRuleIds": document.get("relatedRuleIds", []), "denseScore": round(float(row.get("denseScore") or 0), 4), "bm25Score": round(float(row.get("bm25Score") or 0), 4), "rrfScore": round(float(row["rrfScore"]), 6), "rerankScore": row.get("rerankScore"), "indexVersion": self.index_version, "knowledgeVersion": version, "chunkId": document["chunkId"], "contentHash": document["contentHash"], "rank": rank, "rerankerStatus": reranker_status, "fallbackReason": fallback}))
        return hits


def check_pgvector_extension(database_url: str) -> dict[str, Any]:
    from sqlalchemy import create_engine, text
    try:
        engine = create_engine(database_url)
        with engine.connect() as connection:
            installed = connection.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar()
            available = connection.execute(text("SELECT 1 FROM pg_available_extensions WHERE name='vector'")).scalar() is not None
        return {"available": available, "installed": installed, "database": database_url.split("@")[0] + "@..."}
    except Exception as exc:
        return {"available": False, "installed": None, "error": f"{type(exc).__name__}: {exc}"}
