"""Content-addressed receipts for resumable, offline research cases.

Callers must hold ``app.file_lock.file_lock`` around store construction and the
load/execute/save sequence. A receipt is the completion marker: a result file on
its own never permits skipping execution. Hashes establish consistency, not
authenticity against somebody who can rewrite the entire cache.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator


SCHEMA_VERSION = 1
_CASE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_RECEIPT_KEYS = {
    "schema_version", "complete", "case_name", "identity_digest",
    "config_digest", "result_sha256", "receipt_digest",
}


def _validate_json(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError("Cache values must be portable JSON with string keys and finite numbers")


def _json_bytes(value: Any) -> bytes:
    _validate_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    """Return a SHA-256 digest of canonical, portable JSON (no coercions)."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(data: bytes) -> Any:
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs)
    _validate_json(value)
    return value


@contextmanager
def _regular_file(path: Path) -> Iterator[BinaryIO]:
    """Read current bytes, refusing links, non-files and observable mutations."""
    if path.is_symlink():
        raise ValueError(f"Cache/source path must not be a symbolic link: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Expected a regular file: {path}")
        yield stream
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    current = path.lstat()
    if any(getattr(before, key) != getattr(after, key) for key in fields) or any(
        getattr(after, key) != getattr(current, key) for key in fields
    ):
        raise ValueError(f"File changed while hashing/reading: {path}")


def _regular_file_bytes(path: Path) -> bytes:
    with _regular_file(path) as stream:
        return stream.read()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_components(root: Path, relative: Path) -> Path:
    current = root
    if root.is_symlink():
        raise ValueError(f"Cache/source directory must not become a symbolic link: {root}")
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Cache/source path must not contain symbolic links: {current}")
    return current


def file_identities(paths: Iterable[str | Path], root: str | Path) -> dict[str, str]:
    """Hash current source bytes, keyed by paths relative to ``root``.

    Relative inputs are interpreted relative to root, independent of the current
    directory. Every input must be a regular file inside root; traversal and
    symlinks are rejected. File contents are read on every call, even when size
    and modification time are unchanged.
    """
    original_root = Path(root).absolute()
    resolved_root = original_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError(f"Source root must be a directory: {root}")
    identities: dict[str, str] = {}
    for value in paths:
        path = Path(value)
        if ".." in path.parts or "\\" in str(value):
            raise ValueError(f"Unsafe source path: {value}")
        if path.is_absolute():
            try:
                relative = path.relative_to(original_root)
            except ValueError:
                try:
                    relative = path.relative_to(resolved_root)
                except ValueError as exc:
                    raise ValueError(f"Source path is outside root: {value}") from exc
        else:
            relative = path
        if not relative.parts:
            raise ValueError(f"Expected a source file inside root: {value}")
        source = _check_components(resolved_root, relative)
        identities[relative.as_posix()] = _file_digest(source)
    return dict(sorted(identities.items()))


def _atomic_write(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CaseStore:
    """An immutable run identity and independently committed JSON case results.

    Missing cases return ``None``. Damaged/incomplete receipts return ``None``
    with a warning and are never considered completed. Identity corruption,
    unsafe paths and conflicting writes raise ``ValueError``. ``None`` is not a
    valid result because it is reserved for a cache miss. Blocked outcomes can
    be saved as ordinary JSON objects and resumed identically to other outcomes.
    """

    def __init__(self, directory: str | Path, identity: Any):
        encoded_identity = _json_bytes(identity)
        self.identity_digest = hashlib.sha256(encoded_identity).hexdigest()
        # Detach caller-owned mutable values before persisting the run identity.
        self.identity = _read_json(encoded_identity)
        candidate = Path(directory).absolute()
        if candidate.is_symlink():
            raise ValueError(f"Cache directory must not be a symbolic link: {candidate}")
        candidate.mkdir(parents=True, exist_ok=True)
        self.directory = candidate.resolve(strict=True)
        self._identity_record = {
            "schema_version": SCHEMA_VERSION,
            "identity_digest": self.identity_digest,
            "identity": self.identity,
        }
        self._identity_bytes = _json_bytes(self._identity_record)
        identity_path = self._path("identity.json")
        if identity_path.exists():
            self._verify_identity()
        else:
            cases_path = self._path("cases")
            if cases_path.exists() and any(cases_path.iterdir()):
                raise ValueError("Cache identity is missing for an existing case store")
            _atomic_write(identity_path, self._identity_bytes)

    def _path(self, relative: str) -> Path:
        return _check_components(self.directory, Path(relative))

    def _verify_identity(self) -> None:
        try:
            stored = _read_json(_regular_file_bytes(self._path("identity.json")))
        except (OSError, ValueError) as exc:
            raise ValueError("Cache identity is missing or corrupt") from exc
        if _json_bytes(stored) != self._identity_bytes:
            raise ValueError("Cache identity mismatch; use a separate cache directory")

    def _case_paths(self, case_name: str) -> tuple[Path, Path]:
        if not isinstance(case_name, str) or not _CASE_NAME.fullmatch(case_name) or ".." in case_name:
            raise ValueError("Case name must be a safe filename of 1 to 128 ASCII characters")
        stem = case_name.split(".", 1)[0].upper()
        if stem in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9]", stem):
            raise ValueError("Case name must not use a reserved device filename")
        return (
            self._path(f"cases/{case_name}/result.json"),
            self._path(f"cases/{case_name}/receipt.json"),
        )

    def _receipt(self, case_name: str, config_digest: str, result_bytes: bytes) -> dict[str, Any]:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "case_name": case_name,
            "identity_digest": self.identity_digest,
            "config_digest": config_digest,
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
        return {**receipt, "receipt_digest": content_digest(receipt)}

    def load(self, case_name: str, config: Any) -> Any | None:
        """Load only a result whose detached receipt matches every identity."""
        self._verify_identity()
        config_digest = content_digest(config)
        result_path, receipt_path = self._case_paths(case_name)
        if not result_path.exists() and not receipt_path.exists():
            return None
        try:
            result_bytes = _regular_file_bytes(result_path)
            result = _read_json(result_bytes)
            if result is None:
                raise ValueError("Null is reserved for a cache miss")
            receipt = _read_json(_regular_file_bytes(receipt_path))
            expected = self._receipt(case_name, config_digest, result_bytes)
            if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
                raise ValueError("Receipt schema is invalid")
            if _json_bytes(receipt) != _json_bytes(expected):
                raise ValueError("Receipt identity, configuration, digest or completion flag differs")
            return result
        except (OSError, ValueError) as exc:
            warnings.warn(f"Ignoring incomplete or invalid cached case {case_name!r}: {exc}", RuntimeWarning, stacklevel=2)
            return None

    def save(self, case_name: str, config: Any, result: Any) -> dict[str, Any]:
        """Atomically write result then receipt; never overwrite different data.

        A retry may finish a result whose receipt was never written. An existing
        receipt must already match exactly; damaged receipts need explicit
        investigation or a new cache directory, rather than silent replacement.
        """
        self._verify_identity()
        config_digest = content_digest(config)
        if result is None:
            raise ValueError("Null is reserved for a cache miss; save a result object instead")
        result_bytes = _json_bytes(result)
        receipt = self._receipt(case_name, config_digest, result_bytes)
        result_path, receipt_path = self._case_paths(case_name)
        receipt_bytes = _json_bytes(receipt)
        for path, expected in ((result_path, result_bytes), (receipt_path, receipt_bytes)):
            if path.exists() and _regular_file_bytes(path) != expected:
                raise ValueError(f"Refusing to overwrite conflicting cached case data: {path}")
        if receipt_path.exists() and not result_path.exists():
            raise ValueError("Receipt exists without its result; refusing to repair it implicitly")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        # Recheck components after mkdir, before creating files.
        result_path, receipt_path = self._case_paths(case_name)
        if not result_path.exists():
            _atomic_write(result_path, result_bytes)
        if not receipt_path.exists():
            _atomic_write(receipt_path, receipt_bytes)
        return receipt
