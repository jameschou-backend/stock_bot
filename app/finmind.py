from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

FINMIND_DATA_URL = "https://api.finmindtrade.com/api/v4/data"


class FinMindError(RuntimeError):
    pass


def _build_headers(token: str | None) -> Dict[str, str]:
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def fetch_dataset(
    dataset: str,
    start_date: date,
    end_date: date,
    token: str | None = None,
    data_id: str | None = None,
) -> pd.DataFrame:
    params: Dict[str, Any] = {
        "dataset": dataset,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    if data_id:
        params["data_id"] = data_id

    resp = requests.get(FINMIND_DATA_URL, params=params, headers=_build_headers(token), timeout=30)
    if resp.status_code != 200:
        raise FinMindError(f"FinMind HTTP {resp.status_code}: {resp.text[:200]}")

    payload = resp.json()
    status = payload.get("status")
    if status not in (200, "200", None):
        raise FinMindError(f"FinMind status={status}, msg={payload.get('msg')}")

    data = payload.get("data")
    if data is None:
        raise FinMindError(f"FinMind missing data field: {payload}")
    return pd.DataFrame(data)


def date_chunks(start_date: date, end_date: date, chunk_days: int = 30) -> Iterable[tuple[date, date]]:
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(end_date, cursor + timedelta(days=chunk_days - 1))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)
