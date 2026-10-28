"""An explicit, shared publication cutoff for a daily pipeline invocation."""
import os
from datetime import date


def resolve_ingest_end_date(today: date) -> date:
    raw = os.environ.get('DAILY_DATA_END_DATE')
    end = date.fromisoformat(raw) if raw else today
    if end > today:
        raise ValueError('DAILY_DATA_END_DATE cannot be in the future')
    return end
