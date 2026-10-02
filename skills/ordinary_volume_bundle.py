"""Load sealed ordinary-session matrices without confusing all-day totals."""
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def bind(root, refs, name, expected):
    root = Path(root).resolve()
    path = (root/name).resolve()
    if not path.is_relative_to(root) or digest(path) != expected:
        raise ValueError('Ordinary evidence source changed: '+str(name))
    key = str(path.relative_to(root))
    if key in refs and refs[key] != expected:
        raise ValueError('Conflicting ordinary evidence hash: '+key)
    refs[key] = expected
    return path


def load_ordinary_matrices(root, manifest_path, calendar, stock_ids, refs, *, halts=()):
    root = Path(root).resolve()
    path = (root/manifest_path).resolve()
    sidecar = path.with_suffix('.sha256')
    bind(root, refs, str(path.relative_to(root)), sidecar.read_text().strip())
    bind(root, refs, str(sidecar.relative_to(root)), digest(sidecar))
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'official_quote_repair_evidence_v1':
        raise ValueError('Unsupported normalized official-volume manifest')
    for name, expected in manifest['source_sha256'].items():
        bind(root, refs, name, expected)
    normalized = path.parent/'official-normalized.parquet'
    key = str(normalized.relative_to(root))
    bind(root, refs, key, manifest['output_sha256'][key])
    all_rows = pd.read_parquet(normalized, columns=[
        'date', 'stock_id', 'market', 'volume_scope', 'volume', 'table_category',
        'open', 'high', 'low', 'close'])
    all_rows = all_rows.loc[all_rows.stock_id.isin(stock_ids)].copy()
    all_rows['date'] = pd.to_datetime(all_rows.date)
    rows = all_rows.loc[all_rows.volume_scope.eq('ordinary_session')
                    & ~all_rows.table_category.eq('管理股票')].copy()
    if rows.duplicated(['date', 'stock_id', 'market']).any():
        raise ValueError('Duplicate ordinary-session source rows')
    calendar = pd.DatetimeIndex(calendar)
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError('Canonical market calendar required')
    result = {}
    for market in ('TWSE', 'TPEX'):
        selected = rows.loc[rows.market.eq(market)]
        result[market] = selected.pivot(index='date', columns='stock_id', values='volume').reindex(
            index=calendar, columns=stock_ids)
    # Halts must already have their dated notice metadata verified by the caller.
    # They certify no activity, not a synthetic price observation.
    for halt in halts:
        market, sid = halt['market'].upper(), halt['stock_id']
        if not halt.get('known_by', halt['announcement_date']) < halt['start'] < halt['end']:
            raise ValueError('Advance official halt evidence required')
        bind(root, refs, halt['source_path'], halt['source_sha256'])
        if sid not in stock_ids:
            continue
        dates = calendar[(calendar >= halt['start']) & (calendar < halt['end'])]
        observed_market_days = {(s['market'].upper(), s['date']) for s in manifest['sources'].values()}
        if any((market, str(d.date())) not in observed_market_days for d in dates):
            raise ValueError('Cannot certify halt activity without the official market-day table')
        observed = all_rows.loc[all_rows.market.eq(market) & all_rows.stock_id.eq(sid)
                                & all_rows.date.isin(dates)]
        if (observed[['volume', 'open', 'high', 'low', 'close']].fillna(0) != 0).any().any():
            raise ValueError('Halt conflicts with all-session or ordinary activity')
        existing = result[market].loc[dates, sid]
        if (existing.dropna() != 0).any():
            raise ValueError('Halt conflicts with ordinary-session activity')
        result[market].loc[dates, sid] = 0.
    return result
