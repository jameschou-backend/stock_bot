"""Short-lived successful responses with retrieval time. Never serves expired data."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from app.rate_limiter import state_path


def cache_path(params: dict, token: str | None) -> Path:
    # Credential isolation without writing the token or a standalone token hash.
    key = hashlib.sha256(json.dumps([params, token or ""], sort_keys=True).encode()).hexdigest()
    return state_path().parent / "responses" / f"{key}.json.gz"


def read_cache(path: Path, ttl: float) -> dict | None:
    if not path.exists():
        return None
    with gzip.open(path, "rt") as f:
        value = json.load(f)
    age = time.time() - value["retrieved_at"]
    return value if 0 <= age < ttl else None


def write_cache(path: Path, data: list, retrieved_at: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(gzip.compress(json.dumps({"data": data, "retrieved_at": retrieved_at}).encode()))
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
