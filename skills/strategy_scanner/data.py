"""Read one sealed daily bundle for multiple offline stock scanners.

This adapter verifies its direct inputs, not the thousands of ancestor receipts.
It does not construct candidates from a portfolio, fetch data, or change a DB.
"""
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


WARMUP_SESSIONS = 420
SOURCE_DISAGREEMENT_THRESHOLD = .005
MATRICES = ('close-official.parquet', 'close-quality.parquet', 'eligibility.parquet')
SCHEMA = 'strategy_scanner_bundle_inputs_v1'


def _date(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('Scanner dates must be ISO YYYY-MM-DD strings')
    return pd.Timestamp(value)


def _digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _fingerprint(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(path.read_text(), parse_constant=reject)


class _Inputs:
    def __init__(self, bundle):
        self.bundle = Path(bundle).resolve()
        self.hashes, self.fingerprints = {}, {}
        manifest = self.path('manifest.json')
        sidecar = self.path('manifest.sha256')
        expected = sidecar.read_text().strip()
        self.verify('manifest.json', expected)
        self.manifest = _json(manifest)
        if self.manifest.get('schema') != 'poc_latest_input_bundle_v1':
            raise ValueError('Unsupported scanner bundle schema')
        if not isinstance(self.manifest.get('files_sha256'), dict):
            raise ValueError('Scanner manifest lacks direct input hashes')
        self.hashes['manifest.sha256'] = _digest(sidecar)
        self.fingerprints['manifest.sha256'] = _fingerprint(sidecar)

    def path(self, name):
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError('Bundle input path must be relative')
        path = (self.bundle / name).resolve()
        if not path.is_relative_to(self.bundle):
            raise ValueError('Bundle input escapes bundle directory')
        return path

    def verify(self, name, expected=None):
        if expected is None:
            expected = self.manifest['files_sha256'].get(name)
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError('Missing or invalid direct input hash: ' + name)
        path = self.path(name)
        before = _fingerprint(path)
        if name in self.hashes:
            if self.hashes[name] != expected or self.fingerprints[name] != before:
                raise ValueError('Bundle input changed during loading: ' + name)
            return path
        if _digest(path) != expected or _fingerprint(path) != before:
            raise ValueError('Bundle input SHA mismatch: ' + name)
        self.hashes[name] = expected
        self.fingerprints[name] = before
        return path

    def read_json(self, name):
        return _json(self.verify(name))

    def finish(self):
        if any(_fingerprint(self.path(n)) != value for n, value in self.fingerprints.items()):
            raise ValueError('Bundle input changed during loading')


def _calendar(values):
    result = pd.DatetimeIndex(pd.to_datetime(values))
    if (not len(result) or result.hasnans or result.tz is not None
            or not result.is_unique or not result.is_monotonic_increasing
            or not result.equals(result.normalize())):
        raise ValueError('Bundle market calendar must be unique, ordered and date-only')
    return result


def _matrix(reader, name, full_calendar, ids, first, end):
    path = reader.verify(name)
    schema = pq.ParquetFile(path).schema_arrow.names
    if schema != ['date', *ids]:
        raise ValueError('Bundle matrix stock axes differ: ' + name)
    all_days = _calendar(pd.read_parquet(path, columns=['date']).date)
    if not all_days.equals(full_calendar):
        raise ValueError('Bundle matrix calendars differ: ' + name)
    frame = pd.read_parquet(path, filters=[('date', '>=', first), ('date', '<=', end)])
    frame = frame.set_index('date')
    frame.index = pd.DatetimeIndex(frame.index)
    expected = full_calendar[(full_calendar >= first) & (full_calendar <= end)]
    if not frame.index.equals(expected):
        raise ValueError('Bundle matrix date filter lost observations: ' + name)
    return frame


def _signals(reader, full_calendar, end):
    source = reader.read_json('signals.json')
    groups = source.get('entries')
    if not isinstance(groups, dict) or not isinstance(groups.get('median50m'), list):
        raise ValueError('Original candidate registry needs entries.median50m')
    name = source.get('pending_signal_file', 'pending-signals.json')
    pending = reader.read_json(name)
    if (pending.get('schema') != 'pending_last_close_signals_v1'
            or not isinstance(pending.get('entries'), list)):
        raise ValueError('Invalid terminal-close candidate registry')
    counts = {'executable_candidate_count': len(groups['median50m']),
              'pending_candidate_count': len(pending['entries'])}
    counts_complete = True
    for key, actual in counts.items():
        declared = reader.manifest.get(key)
        if declared is None:
            counts_complete = False
        elif type(declared) is not int or declared < 0 or declared != actual:
            raise ValueError('Original candidate source count mismatch: ' + key)
    calendar = full_calendar.strftime('%Y-%m-%d').tolist()
    positions = {d: i for i, d in enumerate(calendar)}
    seen, coordinates, rows = {}, {}, []
    duplicates = 0
    for original in [*groups['median50m'], *pending['entries']]:
        row = deepcopy(original)
        if not isinstance(row, dict):
            raise ValueError('Candidate must be a record')
        eid, members, signal = row.get('event_id'), row.get('members'), row.get('signal_date')
        if (not isinstance(eid, str) or not eid or not isinstance(members, list)
                or len(members) != 1 or not isinstance(members[0], str)
                or not re.fullmatch(r'[1-9]\d{3}', members[0]) or signal not in positions):
            raise ValueError('Invalid original candidate identity')
        i = positions[signal]
        next_day = calendar[i + 1] if i + 1 < len(calendar) else None
        if row.get('entry_date') != next_day:
            raise ValueError('Original candidate entry must be observed T+1 or terminal null')
        canonical = json.dumps(row, sort_keys=True, allow_nan=False)
        if eid in seen:
            if seen[eid] != canonical:
                raise ValueError('Conflicting duplicate candidate event')
            duplicates += 1
            continue
        key = (signal, members[0])
        if key in coordinates:
            raise ValueError('Duplicate candidate stock/date with different event identity')
        seen[eid], coordinates[key] = canonical, eid
        if pd.Timestamp(signal) <= end:
            rows.append(row)
    # Preserve all original event fields; only deterministic chronological ordering changes.
    rows.sort(key=lambda row: (row['signal_date'], -float(row.get('priority', 0)), row['event_id']))
    coverage = dict(original_candidates_complete=counts_complete and bool(coordinates),
        original_signal_start=min((d for d, _ in coordinates), default=None),
        original_signal_end=reader.manifest['end'],
        original_candidate_source_counts=counts,
        original_candidates_complete_scope='complete_hash_bound_ledger_not_certified_market_universe')
    return rows, duplicates, coverage


def load_bundle(bundle: Path, start: str, end: str) -> dict:
    """Load all stocks, including ineligible/missing rows and 0050 context.

    ``calendar`` includes up to 420 genuine market sessions before ``start``.
    If the sealed history is shorter, provenance reports the shortfall; no dates
    or observations are invented. Price rows stop at ``end``. ``original_signals``
    preserves complete candidate records through ``end`` regardless of holdings.
    """
    first_requested, last = _date(start), _date(end)
    if first_requested > last:
        raise ValueError('Scanner start is later than end')
    reader = _Inputs(bundle)
    manifest = reader.manifest
    source_start, source_end = _date(manifest['start']), _date(manifest['end'])
    if first_requested < source_start or last > source_end:
        raise ValueError('Requested scanner dates exceed the sealed source period')
    path = reader.verify(MATRICES[0])
    schema = pq.ParquetFile(path).schema_arrow.names
    if not schema or schema[0] != 'date' or len(set(schema)) != len(schema):
        raise ValueError('Invalid adjusted matrix schema')
    ids = schema[1:]
    if '0050' not in ids or any(not re.fullmatch(r'[1-9]\d{3}|0050', sid) for sid in ids):
        raise ValueError('Bundle must contain four-digit individual stocks and 0050 context only')
    all_days = _calendar(pd.read_parquet(path, columns=['date']).date)
    if all_days[0] != source_start or all_days[-1] != source_end:
        raise ValueError('Manifest date bounds differ from market calendar')
    if first_requested not in all_days or last not in all_days:
        raise ValueError('Requested scanner bounds must be observed market sessions')
    first_index = max(0, all_days.get_loc(first_requested) - WARMUP_SESSIONS)
    calendar = all_days[first_index:all_days.get_loc(last) + 1]
    # Read one extra source day for a causal source-return comparison at the edge.
    read_first = all_days[max(0, first_index - 1)]
    frames = {name: _matrix(reader, name, all_days, ids, read_first, last) for name in MATRICES}
    adjusted, comparison, eligible = [frames[n] for n in MATRICES]
    for frame in (adjusted, comparison):
        if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
               for dtype in frame.dtypes):
            raise ValueError('Adjusted prices must have numeric, nonboolean columns')
    # Unknown eligibility remains a nullable observation, never inferred from today's master.
    if any(not pd.api.types.is_bool_dtype(dtype) for dtype in eligible.dtypes):
        raise ValueError('Dated eligibility must have boolean or nullable-boolean columns')
    a, b = adjusted.astype(float), comparison.astype(float)
    a_prev, b_prev = a.shift(1), b.shift(1)
    return_known = np.isfinite(a) & a.gt(0) & np.isfinite(b) & b.gt(0)
    return_known &= np.isfinite(a_prev) & a_prev.gt(0) & np.isfinite(b_prev) & b_prev.gt(0)
    disagreement = (a / a_prev - b / b_prev).abs().gt(SOURCE_DISAGREEMENT_THRESHOLD)

    qpath = reader.verify('quotes-unmasked.parquet')
    required = ['stock_id', 'date', 'open', 'high', 'low', 'close', 'volume']
    if not set(required).issubset(pq.ParquetFile(qpath).schema_arrow.names):
        raise ValueError('Missing raw quote columns')
    quotes = pd.read_parquet(qpath, columns=required,
        filters=[('date', '>=', calendar[0]), ('date', '<=', last)])
    quotes['date'] = pd.to_datetime(quotes.date)
    if quotes.duplicated(['date', 'stock_id']).any():
        raise ValueError('Duplicate raw stock/date quote')
    if not quotes.date.isin(calendar).all() or not quotes.stock_id.isin(ids).all():
        raise ValueError('Raw quote identity outside matrix calendar/universe')
    if any(not pd.api.types.is_numeric_dtype(quotes[k]) or pd.api.types.is_bool_dtype(quotes[k])
           for k in required[2:]):
        raise ValueError('Raw OHLCV must be numeric and nonboolean')
    grid = pd.MultiIndex.from_product([calendar, ids], names=['date', 'stock_id'])
    bars = quotes.set_index(['date', 'stock_id']).reindex(grid)
    bars['adjusted_close'] = a.loc[calendar].to_numpy().reshape(-1)
    bars['quality_close'] = b.loc[calendar].to_numpy().reshape(-1)
    bars['eligible'] = pd.array(eligible.loc[calendar].to_numpy().reshape(-1), dtype='boolean')
    bars['source_return_known'] = return_known.loc[calendar].to_numpy().reshape(-1)
    bars['source_disagreement'] = pd.array(disagreement.loc[calendar].to_numpy().reshape(-1), dtype='boolean')
    bars.loc[~bars.source_return_known, 'source_disagreement'] = pd.NA
    ohlc = bars[['open', 'high', 'low', 'close']]
    raw_known = np.isfinite(ohlc).all(axis=1) & ohlc.gt(0).all(axis=1)
    range_ok = bars.low.le(bars[['open', 'close']].min(axis=1))
    range_ok &= bars.high.ge(bars[['open', 'close']].max(axis=1))
    volume_ok = np.isfinite(bars.volume) & bars.volume.gt(0) & bars.volume.eq(np.floor(bars.volume))
    adjusted_ok = np.isfinite(bars.adjusted_close) & bars.adjusted_close.gt(0)
    adjusted_ok &= np.isfinite(bars.quality_close) & bars.quality_close.gt(0)
    bars['quality'] = raw_known & range_ok & volume_ok & adjusted_ok
    bars['quality_reason'] = np.select(
        [~raw_known, ~range_ok, ~volume_ok, ~adjusted_ok],
        ['missing_or_invalid_raw_ohlc', 'raw_ohlc_range_conflict',
         'missing_or_invalid_share_volume', 'missing_or_invalid_adjusted_price'], default='usable_observation')
    bars['amount'] = bars.close * bars.volume
    bars = bars.reset_index()

    companies = pd.read_parquet(reader.verify('companies.parquet'))
    needed = {'stock_id', 'name', 'listed_date', 'industry', 'market'}
    if not needed.issubset(companies.columns) or companies.stock_id.duplicated().any():
        raise ValueError('Invalid company display metadata')
    names = {row.stock_id: row['name'] for _, row in companies.iterrows()}
    names.setdefault('0050', '元大台灣50')
    identity = reader.read_json('identity.json')
    if (identity.get('coverage_start') > str(calendar[0].date())
            or identity.get('coverage_end') < end):
        raise ValueError('Dated identity coverage does not include scan window')
    signals, duplicates, candidate_coverage = _signals(reader, all_days, last)
    if any(row['members'][0] not in ids for row in signals):
        raise ValueError('Original candidate stock missing from bundle universe')
    reader.finish()
    observation = identity.get('extension_observation', {})
    historical_end = manifest.get('historical_end')
    provenance = dict(schema=SCHEMA, bundle=str(reader.bundle), start=start, end=end,
        source_start=manifest['start'], source_end=manifest['end'],
        historical_end=historical_end, calendar_start=str(calendar[0].date()),
        warmup_sessions_required=WARMUP_SESSIONS,
        warmup_sessions_available=int(all_days.get_loc(first_requested) - first_index),
        universe_stock_count=len(ids)-1, benchmark_stock_id='0050',
        universe_basis='all_matrix_stock_columns_not_candidate_or_portfolio_list',
        eligibility_basis='dated_eligibility_matrix_not_current_master',
        source_hashes=dict(reader.hashes), direct_input_files_verified=True,
        upstream_source_count=len(manifest.get('source_sha256', {})),
        upstream_source_closure_reverified=False,
        source_manifest_schema=manifest['schema'],
        source_limitations=deepcopy(manifest.get('limitations', [])),
        officially_crosschecked_extension=manifest.get('officially_crosschecked_extension') is True,
        complete_historical_universe=identity.get('complete_historical_universe') is True,
        continuous_eligibility_proven=identity.get('continuous_eligibility_proven') is True,
        publication_time_archive_complete=identity.get('publication_time_archive_complete') is True,
        identity_extension_provisional=observation.get('provisional'),
        official_identity_verified_through=observation.get('official_identity_verified_through'),
        identity_extension_source=observation.get('source'),
        display_metadata_point_in_time_verified=False,
        source_return_comparison_threshold=SOURCE_DISAGREEMENT_THRESHOLD,
        quality_is_observation_validity_not_independent_source_certification=True,
        amount_basis='raw_close_times_reported_share_volume_proxy_not_actual_turnover',
        volume_unit='shares', missing_bars_preserved=True,
        original_signals_source='signals.json.entries.median50m_and_pending_signals',
        original_signal_duplicates_identical=duplicates,
        original_signal_rows=len(signals),
        selection_inputs_end=end, portfolio_state_used=False,
        live_qualified=False, network_requests=0, database_mutations=False)
    provenance.update(candidate_coverage)
    return dict(bars=bars, calendar=calendar, names=names, original_signals=signals,
                poc=None, provenance=provenance, universe=ids, companies=companies)
