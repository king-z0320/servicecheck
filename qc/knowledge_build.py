from __future__ import annotations

"""Immutable knowledge build manifests and current-pointer management.

The filesystem implementation is intentionally small and deterministic so it
can be used by the local job and by offline tests.  PostgreSQL persistence is
provided by the stage-5 migration; callers can replace this store without
changing the build/publish contract.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from qc.rag import KnowledgeIndex, RagSearchConfig
from qc.rag import _BM25, _tokenize
import numpy as np


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BuildResult:
    knowledge_version: str
    manifest: dict[str, Any]
    status: str


class KnowledgeBuildService:
    """Build, publish and roll back immutable local knowledge versions."""

    def __init__(self, root: str | Path, *, state_dir: str | Path | None = None, embedder=None, reranker=None, config: RagSearchConfig | None = None, store=None):
        self.root = Path(root)
        self.state_dir = Path(state_dir or self.root.parent / ".runtime" / "knowledge_builds")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.embedder, self.reranker, self.config, self.store = embedder, reranker, config, store

    @property
    def pointer_path(self) -> Path:
        return self.state_dir / "current.json"

    def _manifest_path(self, version: str) -> Path:
        return self.state_dir / f"{version}.json"

    def _chunks_path(self, version: str) -> Path:
        return self.state_dir / f"{version}.chunks.json"

    def _vectors_path(self, version: str) -> Path:
        return self.state_dir / f"{version}.vectors.npy"

    def build(self) -> BuildResult:
        index = KnowledgeIndex(self.root, embedder=self.embedder, reranker=self.reranker, config=self.config)
        index.build()
        return self._persist(index)

    def _persist(self, index: KnowledgeIndex) -> BuildResult:
        version = f"kv-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        manifest = dict(index.manifest)
        manifest["knowledgeVersion"] = version
        manifest["status"] = "READY"
        manifest["builtAt"] = datetime.now(timezone.utc).isoformat()
        manifest["sourceRoot"] = str(self.root)
        # Identity/time/status are publication metadata, not build inputs;
        # excluding them keeps the manifest hash reproducible across builds.
        manifest["manifestHash"] = _hash({key: value for key, value in manifest.items() if key not in {"manifestHash", "knowledgeVersion", "builtAt", "status", "sourceRoot"}})
        path = self._manifest_path(version)
        if path.exists():
            raise FileExistsError(f"knowledge build already exists: {version}")
        path.write_text(_canonical(manifest), encoding="utf-8")
        self._chunks_path(version).write_text(_canonical(index.documents), encoding="utf-8")
        np.save(self._vectors_path(version), index.vectors)
        if self.store is not None:
            self.store.save_ready(manifest, index.documents, index.vectors)
        return BuildResult(version, manifest, "READY")

    def persist_index(self, index: KnowledgeIndex) -> BuildResult:
        """Persist an already-built index without recomputing embeddings."""
        if index.vectors is None or not index.documents:
            raise ValueError("index must be built before persistence")
        return self._persist(index)

    def _read(self, version: str) -> dict[str, Any]:
        path = self._manifest_path(version)
        if not path.exists():
            raise KeyError(f"unknown knowledge version: {version}")
        return json.loads(path.read_text(encoding="utf-8"))

    def publish(self, knowledge_version: str, *, actor: str = "system") -> dict[str, Any]:
        manifest = self._read(knowledge_version)
        if manifest.get("status") != "READY":
            raise ValueError("only READY knowledge builds can be published")
        if self.store is not None:
            return self.store.publish(knowledge_version, actor)
        pointer = {"knowledgeVersion": knowledge_version, "actor": actor, "updatedAt": datetime.now(timezone.utc).isoformat()}
        temporary = self.pointer_path.with_suffix(".tmp")
        temporary.write_text(_canonical(pointer), encoding="utf-8")
        os.replace(temporary, self.pointer_path)
        return pointer

    def rollback(self, knowledge_version: str, *, actor: str = "system") -> dict[str, Any]:
        return self.publish(knowledge_version, actor=actor)

    def current(self) -> dict[str, Any] | None:
        if self.store is not None:
            current = self.store.current()
            return {"knowledgeVersion": current} if current else None
        if not self.pointer_path.exists():
            return None
        return json.loads(self.pointer_path.read_text(encoding="utf-8"))

    def load_current_index(self, *, embedder=None, reranker=None) -> KnowledgeIndex:
        pointer = self.current()
        if not pointer:
            raise RuntimeError("knowledge current pointer is not configured")
        version = pointer["knowledgeVersion"]
        manifest = self._read(version)
        index = KnowledgeIndex(self.root, embedder=embedder, reranker=reranker, config=self.config, knowledge_version=version)
        index.documents = json.loads(self._chunks_path(version).read_text(encoding="utf-8"))
        index.vectors = np.load(self._vectors_path(version), allow_pickle=False)
        index._bm25 = _BM25([_tokenize(document["title"] + "\n" + document["content"]) for document in index.documents])
        index.index_version = manifest.get("indexHash")
        index.manifest = manifest
        # Query embeddings are still required; use the configured local model.
        index.embedder = embedder or index._default_embedder()
        return index
