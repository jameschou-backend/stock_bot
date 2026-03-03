from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

import pandas as pd


def _parse_json_chunk(chunk: List[Any]) -> List[Dict[str, Any]]:
    """Parse a list of JSON values (picklable, for multiprocessing)."""
    out: List[Dict[str, Any]] = []
    for v in chunk:
        if v is None:
            out.append({})
        elif isinstance(v, dict):
            out.append(v)
        else:
            out.append(json.loads(v))
    return out


def parse_features_json(series: pd.Series) -> pd.DataFrame:
    """Parse a Series of features_json values into a DataFrame.

    Uses multiprocessing for large datasets (>10k rows) to speed up
    the JSON parsing step which is CPU-bound.
    """
    values = series.tolist()
    n_workers = min(os.cpu_count() or 4, 8)

    if len(values) < 10_000 or n_workers <= 1:
        parsed = _parse_json_chunk(values)
    else:
        chunk_size = max(1, len(values) // n_workers)
        chunks = [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_parse_json_chunk, chunks))
        parsed = [item for sublist in results for item in sublist]

    return pd.DataFrame(parsed)
