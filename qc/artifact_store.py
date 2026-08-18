from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Protocol
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Stable metadata for one artifact stored behind a logical URI."""

    uri: str
    sha256: str
    size: int
    mime_type: str | None = None


class ArtifactStore(Protocol):
    def put_bytes(
        self,
        uri: str,
        content: bytes,
        *,
        mime_type: str | None = None,
    ) -> ArtifactRef: ...

    def open(self, uri: str) -> BinaryIO: ...

    def read_bytes(self, uri: str) -> bytes: ...

    def exists(self, uri: str) -> bool: ...

    def stat(self, uri: str) -> os.stat_result: ...

    def verify_sha256(self, uri: str, expected_sha256: str) -> bool: ...

    def resolve_for_read(self, uri: str) -> Path: ...


class LocalArtifactStore:
    """Store artifacts below one local root without exposing absolute paths."""

    def __init__(self, root: str | Path):
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        self.root = root_path.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("artifact root must be a directory")

    @staticmethod
    def _validate_uri(uri: str) -> str:
        if not isinstance(uri, str) or not uri or "\x00" in uri:
            raise ValueError("artifact URI must be a non-empty string")
        if "\\" in uri:
            raise ValueError("artifact URI must use forward slashes")

        posix = PurePosixPath(uri)
        windows = PureWindowsPath(uri)
        if posix.is_absolute() or windows.is_absolute() or windows.drive:
            raise ValueError("artifact URI must be relative")
        if any(part in {"", ".", ".."} for part in posix.parts):
            raise ValueError("artifact URI contains an unsafe path segment")
        return posix.as_posix()

    def _resolve(self, uri: str) -> tuple[str, Path]:
        normalized = self._validate_uri(uri)
        candidate = self.root.joinpath(*PurePosixPath(normalized).parts)
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("artifact URI escapes artifact root") from exc
        return normalized, resolved

    @staticmethod
    def _digest_bytes(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _digest_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _atomic_replace(self, source: Path, target: Path) -> None:
        os.replace(source, target)

    def put_bytes(
        self,
        uri: str,
        content: bytes,
        *,
        mime_type: str | None = None,
    ) -> ArtifactRef:
        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")

        normalized, target = self._resolve(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        _, target = self._resolve(normalized)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")

        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            self._atomic_replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

        return ArtifactRef(
            uri=normalized,
            sha256=self._digest_bytes(content),
            size=len(content),
            mime_type=mime_type,
        )

    def resolve_for_read(self, uri: str) -> Path:
        _, path = self._resolve(uri)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def open(self, uri: str) -> BinaryIO:
        return self.resolve_for_read(uri).open("rb")

    def read_bytes(self, uri: str) -> bytes:
        with self.open(uri) as stream:
            return stream.read()

    def exists(self, uri: str) -> bool:
        try:
            _, path = self._resolve(uri)
        except ValueError:
            return False
        return path.is_file()

    def stat(self, uri: str) -> os.stat_result:
        return self.resolve_for_read(uri).stat()

    def verify_sha256(self, uri: str, expected_sha256: str) -> bool:
        try:
            path = self.resolve_for_read(uri)
        except (FileNotFoundError, OSError, ValueError):
            return False
        return self._digest_file(path) == expected_sha256.lower()
