"""Read-only, byte-exact inventory of existing Taiwan research evidence.

This is an integrity receipt, not a backup or a filesystem write lock. Capture
never changes a protected file. Verification checks every recorded file; new
US files are allowed and cannot replace existing paths unnoticed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

SCHEMA = "tw_baseline_guard_v1"
ROOTS = ("app", "skills", "pipelines", "scripts", "storage", "ui", "docs", "tests",
         "artifacts", ".cache", "config.yaml", "Makefile", "requirements.txt", ".env")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "mcp-venv", "tw-baseline-protection", "us-market-v1"}
EXCLUDED_SUFFIXES = (".pyc", ".lock", ".tmp")
# Newly introduced guard/US tests are not legacy Taiwan inputs. Keep this exact
# list in the receipt; never broaden exclusions to an entire source directory.
NEW_FILES = {"skills/tw_baseline_guard.py", "scripts/protect_tw_baseline.py",
             "scripts/fingerprint_tw_database.py", "tests/test_tw_baseline_guard.py",
             "tests/test_us_market.py"}


def digest_file(path: Path) -> str:
    """Reject links and concurrent modifications rather than sealing mixed bytes."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Expected regular file: {path.name}")
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    after = path.lstat()
    fields = ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError(f"File changed during inventory: {path.name}")
    return h.hexdigest()


def _files(root: Path, names):
    seen = set()
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Inventory roots must be repository-relative")
        target = root / relative
        if not target.exists():
            raise ValueError(f"Missing protected root: {name}")
        if target.is_symlink():
            raise ValueError("Protected roots cannot be symbolic links")
        candidates = [target] if target.is_file() else target.rglob("*")
        for path in candidates:
            rel = path.relative_to(root)
            if (str(rel) in NEW_FILES or set(rel.parts) & EXCLUDED_PARTS
                    or path.name.endswith(EXCLUDED_SUFFIXES) or path.name == ".DS_Store"):
                continue
            if path.is_symlink():
                # Links are recorded, never followed into another tree.
                kind = "symlink"
            elif path.is_file():
                kind = "file"
            else:
                continue
            if str(rel) not in seen:
                seen.add(str(rel))
                yield rel, path, kind


def capture(root: Path, output: Path, *, names=ROOTS):
    root, output = root.resolve(), output.absolute()
    if output.exists():
        raise ValueError("Baseline already exists; use a new directory")
    # Output must not become part of its own inventory.
    if output.is_relative_to(root) and not (set(output.relative_to(root).parts) & EXCLUDED_PARTS):
        raise ValueError("Use .cache/tw-baseline-protection/<new-name> or an external directory")
    output.mkdir(parents=True, exist_ok=False)
    inventory = output / "files.jsonl"
    count = total = 0
    with inventory.open("x", encoding="utf-8") as dest:
        for rel, path, kind in _files(root, names):
            item = {"path": str(rel), "kind": kind}
            if kind == "symlink":
                item["target"] = os.readlink(path)
            else:
                item.update(size=path.stat().st_size, sha256=digest_file(path))
                total += item["size"]
            dest.write(json.dumps(item, sort_keys=True) + "\n")
            count += 1
    receipt = dict(schema=SCHEMA, roots=list(names), files=count, bytes=total,
                   inventory_sha256=digest_file(inventory), excludes=sorted(EXCLUDED_PARTS),
                   excluded_suffixes=list(EXCLUDED_SUFFIXES), is_backup=False,
                   excluded_new_files=sorted(NEW_FILES),
                   scope="existing files only; excludes volatile locks/temp/bytecode; database separate")
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def verify(root: Path, baseline: Path):
    root, baseline = root.resolve(), baseline.resolve()
    receipt = json.loads((baseline / "receipt.json").read_text())
    inventory = baseline / "files.jsonl"
    if receipt.get("schema") != SCHEMA or digest_file(inventory) != receipt.get("inventory_sha256"):
        raise ValueError("Baseline receipt/inventory mismatch")
    differences, count, total = [], 0, 0
    with inventory.open() as source:
        for line in source:
            item = json.loads(line)
            rel = Path(item["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe baseline path")
            path = root / rel
            if any((root.joinpath(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts))):
                differences.append({"path":str(rel),"reason":"symlink_parent"})
                continue
            count += 1
            try:
                if item["kind"] == "symlink":
                    equal = path.is_symlink() and os.readlink(path) == item["target"]
                else:
                    total += item["size"]
                    equal = digest_file(path) == item["sha256"]
                if not equal:
                    differences.append({"path":str(rel),"reason":"changed"})
            except (OSError, ValueError):
                differences.append({"path":str(rel),"reason":"missing_or_unstable"})
    if count != receipt["files"] or total != receipt["bytes"]:
        differences.append({"path":"files.jsonl","reason":"inventory_totals_mismatch"})
    return dict(schema=SCHEMA, unchanged=not differences, checked_files=count,
                checked_bytes=total, differences=differences,
                inventory_sha256=receipt["inventory_sha256"])
