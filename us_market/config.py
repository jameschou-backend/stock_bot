from __future__ import annotations

import os
from pathlib import Path
import stat

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / ".cache/us-market-v1"
GIB = 1024 ** 3


def api_key(key_file: Path | None = None) -> str:
    """Read a process secret or a private file outside the repository; never .env."""
    if key_file is not None:
        path = key_file.expanduser().absolute()
        if path.is_symlink() or path.resolve().is_relative_to(REPO):
            raise ValueError("API key file must be a regular private file outside the repository")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("API key file must have mode 600")
        value = path.read_text().strip()
    else:
        value = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not value or any(c.isspace() for c in value):
        raise ValueError("Set MASSIVE_API_KEY or pass --api-key-file pointing to a private file outside the repository")
    return value


def safe_root(root: Path) -> Path:
    root = root.expanduser().absolute()
    if any(p.is_symlink() for p in [root, *root.parents]):
        raise ValueError("US data root cannot contain symbolic links")
    root = root.resolve()
    if root == REPO or REPO.is_relative_to(root):
        raise ValueError("US data root cannot contain the Taiwan repository")
    if root.is_relative_to(REPO) and not root.is_relative_to(DEFAULT_ROOT):
        raise ValueError("Inside this repository US writes are restricted to .cache/us-market-v1")
    return root
