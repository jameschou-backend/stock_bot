"""One reviewed 0050 date only; original FinMind files and other days stay intact."""
from copy import deepcopy
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
from pathlib import Path

import pandas as pd


EVIDENCE_PATH = 'docs/benchmark_0050_limit_overlay_20261003.json'
EVIDENCE_SHA256 = '94d50df226fd7f144619f6c4f6f0b7babf365b40218ad1006de1718529022ee5'
STOCK, DAY, PRIOR = '0050', '2025-04-10', '2025-04-09'
PROVIDER = dict(lower=132., upper=160.5)
DERIVED = dict(lower=131.6, upper=160.8)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inside(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Benchmark limit evidence escapes repository')
    return path


def _validate_patch(evidence):
    if (evidence.get('schema') != 'benchmark_0050_dated_limit_overlay_v1'
            or evidence.get('stock_id') != STOCK or len(evidence.get('patches', [])) != 1):
        raise ValueError('Only the reviewed single-stock/single-date overlay is supported')
    p = evidence['patches'][0]
    if (p.get('date') != DAY or p.get('reference_date') != PRIOR
            or p.get('reference_price') != 146.2 or p.get('provider') != PROVIDER
            or p.get('derived') != DERIVED or p.get('price_step') != .05
            or p.get('limit_fraction') != .10 or p.get('observed_official_daily_limits') is not False):
        raise ValueError('Reviewed benchmark patch identity or values changed')
    reference, step = Decimal('146.2'), Decimal('.05')
    calculated = dict(lower=float((reference*Decimal('.9')/step).to_integral_value(rounding=ROUND_CEILING)*step),
                      upper=float((reference*Decimal('1.1')/step).to_integral_value(rounding=ROUND_FLOOR)*step))
    if calculated != DERIVED:
        raise ValueError('Fixed ETF rule calculation differs from reviewed bounds')
    return p


class BenchmarkLimitOverlay:
    """Apply to original feed mappings; the returned mapping is always a copy."""
    def __init__(self, evidence, refs):
        self.patch = deepcopy(_validate_patch(evidence))
        self.refs = dict(refs)
        self.audit_rows = []

    def apply(self, stock_id, limits):
        result = deepcopy(limits)
        if stock_id != STOCK:
            return result
        if DAY not in result or any(result[DAY].get(k) != v for k, v in PROVIDER.items()):
            raise ValueError('0050 overlay requires the exact original dated provider bounds')
        result[DAY].update(DERIVED)
        if not self.audit_rows:
            self.audit_rows.append(dict(stock_id=STOCK, **deepcopy(self.patch),
                evidence_path=EVIDENCE_PATH, evidence_sha256=self.refs.get(EVIDENCE_PATH),
                source_sha256=deepcopy(self.refs), applied=True,
                strict_data_ready=False, live_qualified=False))
        return result


def load_overlay(root):
    """Verify pinned rule/source bytes, query identity and reviewed daily rows."""
    root = Path(root).resolve()
    evidence_file = _inside(root, EVIDENCE_PATH)
    if _sha(evidence_file) != EVIDENCE_SHA256:
        raise ValueError('Benchmark limit evidence hash changed')
    evidence = json.loads(evidence_file.read_text())
    patch = _validate_patch(evidence)
    refs = dict(evidence['source_sha256'])
    for name, expected in refs.items():
        if _sha(_inside(root, name)) != expected:
            raise ValueError('Benchmark limit source hash changed: '+name)
    refs[EVIDENCE_PATH] = EVIDENCE_SHA256
    limit_name = next(n for n in refs if n.endswith('/0050-TaiwanStockPriceLimit.parquet'))
    meta = json.loads(_inside(root, limit_name).with_suffix('.json').read_text())
    if any(meta.get(k) != v for k, v in evidence['provider_query'].items()):
        raise ValueError('Benchmark provider query identity changed')
    if meta.get('sha256') != refs[limit_name]:
        raise ValueError('Benchmark provider metadata does not bind its raw file')
    raw = pd.read_parquet(_inside(root, limit_name))
    row = raw.loc[raw.stock_id.eq(STOCK) & raw.date.eq(DAY)]
    if len(row) != 1 or any(row.iloc[0][k] != v for k, v in
                            dict(reference_price=146.2, limit_down=132., limit_up=160.5).items()):
        raise ValueError('Benchmark raw reference/limits differ from reviewed row')
    observed = {}
    for name in refs:
        if '/acquisition/receipts/' not in name:
            continue
        receipt = json.loads(_inside(root, name).read_text())
        day = receipt.get('date')
        if (day not in (PRIOR, DAY) or receipt.get('market') != 'TWSE'
                or receipt.get('http_status') != 200 or receipt.get('accepted') is not True
                or receipt.get('params') != dict(date=day.replace('-', ''), response='json', type='ALLBUT0999')
                or refs.get(receipt.get('raw_path')) != receipt.get('raw_sha256')):
            raise ValueError('Official benchmark daily receipt identity changed')
        payload = json.loads(_inside(root, receipt['raw_path']).read_text())
        if payload.get('date') != day.replace('-', '') or payload.get('stat') != 'OK':
            raise ValueError('Official benchmark daily response identity changed')
        found = []
        for table in payload.get('tables', []):
            fields = table.get('fields', [])
            if '證券代號' in fields:
                found.extend(dict(zip(fields, values)) for values in table['data']
                             if values[fields.index('證券代號')] == STOCK)
        if len(found) != 1 or day in observed:
            raise ValueError('Ambiguous official benchmark daily row')
        observed[day] = found[0]
    if set(observed) != {PRIOR, DAY} or Decimal(observed[PRIOR]['收盤價']) != Decimal('146.20'):
        raise ValueError('Previous official close does not establish the reference')
    if any(Decimal(observed[DAY][field]) != Decimal('160.80')
           for field in ('開盤價','最高價','最低價','收盤價')):
        raise ValueError('Official conflict-day OHLC changed')
    if Decimal(observed[DAY]['漲跌價差']) != Decimal('14.60'):
        raise ValueError('Official price-change/reference continuity changed')
    events_name = next(n for n in refs if n.endswith('/events.parquet'))
    events = pd.read_parquet(_inside(root, events_name))
    check = patch['corporate_check']
    if (events.stock_id.eq(STOCK) & pd.to_datetime(events.event_date).between(
            check['event_window_start'], check['event_window_end'])).any():
        raise ValueError('A corporate event affects the reviewed reference window')
    dividend_name = next(n for n in refs if n.endswith('/0050-TaiwanStockDividend.parquet'))
    dividends = pd.read_parquet(_inside(root, dividend_name))
    for column in ('CashExDividendTradingDate','StockExDividendTradingDate'):
        if (dividends.stock_id.eq(STOCK) & dividends[column].eq(DAY)).any():
            raise ValueError('Ex-dividend reference requires separate evidence')
    return BenchmarkLimitOverlay(evidence, refs)
