"""Explicit zero activity during dated, officially announced full-session halts.

These are non-executable accounting observations, never inferred market quotes.
Unknown gaps outside a verified interval remain unknown.
"""
import hashlib
from pathlib import Path
import pandas as pd


def add_verified_halt_observations(quotes, days, halts, root):
    result = quotes.copy()
    result['date'] = pd.to_datetime(result['date'])
    evidence = []
    fields = ['open', 'high', 'low', 'close', 'volume']
    for halt in halts:
        start, end = pd.Timestamp(halt['start']), pd.Timestamp(halt['end'])
        if (halt.get('market') not in ('TWSE', 'TPEx')
                or halt['kind'] != 'trading_suspension' or start >= end
                or pd.Timestamp(halt['announcement_date']) >= start):
            raise ValueError('Halt must be announced before the full-session suspension')
        path = Path(root)/halt['source_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != halt['source_sha256']:
            raise ValueError('Official halt source changed')
        for day in days[(days >= start) & (days < end)]:
            existing = result.loc[result.stock_id.eq(halt['stock_id']) & result.date.eq(day)]
            if not existing.empty:
                if len(existing) != 1 or not existing[fields].eq(0).all().all():
                    raise ValueError('Official full-session halt conflicts with quote')
                continue
            row = dict(stock_id=halt['stock_id'], date=day, **dict.fromkeys(fields, 0.))
            result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
            evidence.append(dict(stock_id=halt['stock_id'], date=str(day.date()),
                                 source_path=halt['source_path'],
                                 basis='official_full_session_halt_nonexecutable_zero_activity'))
    return result.sort_values(['date', 'stock_id']).reset_index(drop=True), evidence
