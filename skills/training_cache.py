"""Reuse an identical training problem across parameter sweeps and CLI workers.

The key hashes actual ordered arrays, labels, weights, rank groups, estimator
parameters and library versions. Same dates/row counts alone never identify data.
Only models created by this application in its private local cache are loaded.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np

from app.file_lock import file_lock

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def training_key(model, X, y, sample_weight=None, group=None) -> str:
    digest = hashlib.sha256()
    metadata = {
        "schema": 1,
        "estimator": f"{type(model).__module__}.{type(model).__qualname__}",
        "params": model.get_params(deep=True),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("lightgbm", "scikit-learn", "numpy", "joblib")},
    }
    digest.update(json.dumps(metadata, sort_keys=True, default=str).encode())
    for array in (X, y, sample_weight, group):
        if array is None:
            digest.update(b"null")
            continue
        array = np.ascontiguousarray(array)
        if array.dtype.hasobject:
            raise ValueError("Training cache requires numeric arrays")
        digest.update(json.dumps([array.shape, array.dtype.str]).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def fit_cached(model, X, y, *, sample_weight=None, group=None):
    """A cache hit skips fitting entirely; miss and forced refresh use identical fit arguments."""
    enabled = os.environ.get("BACKTEST_TRAIN_CACHE", "on").lower()
    if enabled not in {"on", "off"}:
        raise ValueError("BACKTEST_TRAIN_CACHE must be on or off")
    kwargs = {"sample_weight": sample_weight}
    if group is not None:
        kwargs["group"] = group
    if enabled == "off":
        model.fit(X, y, **kwargs)
        model.stock_bot_cache_hit_ = False
        return model
    started = time.perf_counter()
    key = training_key(model, X, y, sample_weight, group)
    folder = Path(os.environ.get("BACKTEST_TRAIN_CACHE_DIR", str(ROOT / ".cache/training")))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.joblib"
    with file_lock(folder / f"{key}.lock", timeout=120):
        if path.exists():
            model = joblib.load(path)
            model.stock_bot_cache_hit_ = True
            logger.info("[training-cache] hit %s %.3fs", key[:12], time.perf_counter() - started)
            return model
        model.fit(X, y, **kwargs)
        model.stock_bot_cache_hit_ = False
        fd, temp = tempfile.mkstemp(dir=folder, suffix=".tmp")
        os.close(fd)
        try:
            joblib.dump(model, temp, compress=3)
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)
        logger.info("[training-cache] miss %s %.3fs", key[:12], time.perf_counter() - started)
        return model
