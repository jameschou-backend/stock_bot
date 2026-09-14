"""Explicit reviewed bad observations; never infer a delisting from missing prices."""
from datetime import date
from decimal import Decimal
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / 'docs/price_quarantine_20260914.json'
FIELDS = ('open', 'high', 'low', 'close', 'volume')


def load_registry(path=REGISTRY):
    data = json.loads(Path(path).read_text())
    if data['schema'] != 'reviewed_price_quarantine_v1':
        raise ValueError('Unsupported price quarantine registry')
    rows = data['rows']
    keys = [(r['stock_id'], r['date']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate quarantine key')
    for r in rows:
        if (len(r['stock_id']) != 4 or not r['stock_id'].isdigit()
                or date.fromisoformat(r['date']) < date.fromisoformat(r['official_market_end'])
                or r['resolution'] not in ('misdated_older_copy', 'zero_placeholder')):
            raise ValueError('Invalid reviewed price evidence')
    return {key: row for key, row in zip(keys, rows)}


def same_observation(row, expected):
    return all(row.get(k) is not None and
               Decimal(str(row[k])) == Decimal(str(expected[k])) for k in FIELDS)


def reject_quarantined(records, registry=None):
    """Fail the write, even for revised values: each quarantined key needs review."""
    registry = load_registry() if registry is None else registry
    conflicts = [(str(r['stock_id']), str(r['trading_date'])[:10]) for r in records
                 if (str(r['stock_id']), str(r['trading_date'])[:10]) in registry]
    if conflicts:
        raise ValueError('Reviewed price quarantine: refusing write for '+str(conflicts[:10])+
                         '; reconcile official listing and source evidence before lifting quarantine')
    return records
