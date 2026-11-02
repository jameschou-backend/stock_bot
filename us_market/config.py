from __future__ import annotations

import os
from pathlib import Path
import stat

from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / ".cache/us-market-v1"
GIB = 1024 ** 3


def api_key(key_file: Path | None = None) -> str:
    """Explicit key file > process environment > repo .env.us; never Taiwan .env.

    Read only MASSIVE_API_KEY without interpolation or changing os.environ, so
    US configuration cannot replace Taiwan DB/settings in the current process.
    """
    if key_file is not None:
        path = key_file.expanduser().absolute()
        if path.is_symlink() or path.resolve().is_relative_to(REPO):
            raise ValueError("API key file must be a regular private file outside the repository")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("API key file must have mode 600")
        value = path.read_text().strip()
    elif "MASSIVE_API_KEY" in os.environ:
        value = os.environ.get("MASSIVE_API_KEY", "").strip()
    else:
        path = REPO / ".env.us"
        if path.is_symlink():
            raise ValueError("US .env.us must be a regular private file, not a symbolic link")
        if path.exists():
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                raise ValueError("US .env.us must have mode 600; run chmod 600 .env.us")
            value = (dotenv_values(path, interpolate=False).get("MASSIVE_API_KEY") or "").strip()
        else:
            value = ""
    if not value or any(c.isspace() for c in value):
        raise ValueError("Set MASSIVE_API_KEY in the process environment or repository .env.us (mode 600), or pass --api-key-file for a private external file")
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
