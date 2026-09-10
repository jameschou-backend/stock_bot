#!/usr/bin/env python
"""New, sealed signal inputs for the integer-share replay; no portfolio returns.

FinMind retrieval is explicit (``--fetch-adjusted``), checkpointed by date, and
uses the application's shared quota. Signal construction is entirely offline.
The old diffusion research files are read and verified, never rewritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy

from app.file_lock import file_lock
from skills.diffusion_signals import build_diffusion
from skills.regime_state import build_trend, gate_events

INPUT_DIR = ROOT / '.cache/million-replay-inputs'
CACHE = ROOT / '.cache/million-replay-signals'
ADJ_DIR = CACHE / 'adjraw'
OLD_DIR = ROOT / '.cache/diffusion-research'
SPEC = ROOT / 'docs/prereg_million_replay_20260910.md'
START, SIGNAL_END, END = '2022-01-03', '2026-09-08', '2026-09-09'
FREEZE, ANCHOR_START = '2026-06-23', '2026-06-01'
FRESH_EXTRA_DATES = ('2026-05-21', '2026-05-22')
DATASET = 'TaiwanStockPriceAdj'
CODE = ('scripts/prepare_million_signals.py', 'skills/diffusion_signals.py',
        'skills/regime_state.py', 'skills/official_adj_factors.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                               default=str, indent=2), encoding='utf-8')
    temp.replace(path)


def _records_digest(records):
    value = json.dumps(records, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(value.encode()).hexdigest()


def fetch_dates():
    """At most 75 weekday requests, including two known old-calendar holes."""
    return sorted(set(FRESH_EXTRA_DATES) | {
        str(day.date()) for day in pd.bdate_range(ANCHOR_START, END)})


def _validate_fresh_rows(frame, requested_date):
    if frame.empty:
        return []
    if not {'stock_id', 'date', 'close'}.issubset(frame.columns):
        raise ValueError('Adjusted-price response missing stock_id/date/close')
    if set(pd.to_datetime(frame['date']).dt.strftime('%Y-%m-%d')) != {requested_date}:
        raise ValueError('Adjusted-price response contains another date')
    if frame.duplicated(['stock_id', 'date']).any():
        raise ValueError('Duplicate adjusted-price response rows')
    # to_json converts missing source fields to null; secrets never enter rows.
    return json.loads(frame.to_json(orient='records', date_format='iso'))


def _checkpoint(path, day):
    value = json.loads(path.read_text())
    if (value.get('dataset') != DATASET or value.get('date') != day
            or not isinstance(value.get('data'), list)
            or value.get('data_sha256') != _records_digest(value.get('data'))):
        raise ValueError('Changed or malformed adjusted-price checkpoint: ' + day)
    _validate_fresh_rows(pd.DataFrame(value['data']), day)
    return value


def verify_adjusted(out_dir=ADJ_DIR):
    out_dir = Path(out_dir)
    info = json.loads((out_dir / 'manifest.json').read_text())
    if (info.get('schema') != 1 or info.get('dataset') != DATASET
            or info.get('dates') != fetch_dates()
            or set(info.get('files_sha256', {})) != {f'{day}.json' for day in fetch_dates()}
            or info.get('calls_reserved', 101) > 100):
        raise ValueError('Incomplete or incompatible adjusted-price manifest')
    for name, digest in info['files_sha256'].items():
        if sha(out_dir / name) != digest:
            raise ValueError('Frozen adjusted checkpoint changed: ' + name)
        _checkpoint(out_dir / name, name[:-5])
    if (sha(out_dir / 'plan.json') != info['plan_sha256']
            or sha(out_dir / 'attempts.json') != info['attempts_sha256']):
        raise ValueError('Adjusted retrieval provenance changed')
    return info


def fetch_adjusted(*, out_dir=ADJ_DIR, workers=4, fetcher=None, token=None):
    """Fetch only the fixed date plan, without retrying quota/permission errors.

    A batch has at most four calls. Any error stops before the next batch, and
    successful peers are saved. Calls are counted before dispatch, including
    interrupted attempts, so the local lifetime request ceiling cannot reset.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError('workers must be an integer between 1 and 4')
    if fetcher is None:
        from app.finmind import fetch_dataset
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env')
        token = token or os.getenv('FINMIND_TOKEN')
        if not token:
            raise ValueError('FINMIND_TOKEN is required for explicit adjusted-price retrieval')
        fetcher = fetch_dataset
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(out_dir / '.fetch.lock', timeout=60):
        if (out_dir / 'manifest.json').exists():
            return verify_adjusted(out_dir)
        plan = fetch_dates()
        plan_path = out_dir / 'plan.json'
        fixed = {'schema': 1, 'dataset': DATASET, 'dates': plan,
                 'maximum_calls': 100, 'requests_per_hour': 5400, 'max_retries': 0}
        if plan_path.exists() and json.loads(plan_path.read_text()) != fixed:
            raise ValueError('Adjusted retrieval plan changed; use a new output directory')
        if not plan_path.exists():
            write_json(plan_path, fixed)
        ledger_path = out_dir / 'attempts.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'calls': []}
        todo = []
        for day in plan:
            path = out_dir / f'{day}.json'
            if path.exists():
                _checkpoint(path, day)
            else:
                todo.append(day)
        def one(day):
            frame = fetcher(DATASET, pd.Timestamp(day).date(), pd.Timestamp(day).date(),
                            token=token, requests_per_hour=5400, max_retries=0,
                            cache_ttl=86400, timeout=60)
            records = _validate_fresh_rows(frame, day)
            value = {'dataset': DATASET, 'date': day, 'data': records,
                     'data_sha256': _records_digest(records),
                     'retrieved_at': frame.attrs.get('retrieved_at'),
                     'cache_hit': bool(frame.attrs.get('cache_hit', False)),
                     'saved_at': datetime.now(timezone.utc).isoformat()}
            write_json(out_dir / f'{day}.json', value)
            return len(records)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for begin in range(0, len(todo), workers):
                batch = todo[begin:begin + workers]
                if len(ledger['calls']) + len(batch) > fixed['maximum_calls']:
                    raise ValueError('Adjusted retrieval lifetime ceiling of 100 calls reached')
                ledger['calls'].extend({'date': day, 'started_at': datetime.now(timezone.utc).isoformat()}
                                       for day in batch)
                write_json(ledger_path, ledger)
                futures = [(day, pool.submit(one, day)) for day in batch]
                errors = []
                for day, future in futures:
                    try:
                        count = future.result()
                        print(f'adjusted {day}: {count} rows checkpointed', flush=True)
                    except Exception as exc:
                        errors.append(exc)
                if errors:
                    raise errors[0]
        files = {f'{day}.json': sha(out_dir / f'{day}.json') for day in plan}
        result = {**fixed, 'files_sha256': files, 'calls_reserved': len(ledger['calls']),
                  'plan_sha256': sha(plan_path), 'attempts_sha256': sha(ledger_path)
                  if ledger_path.exists() else None,
                  'prepared_at': datetime.now(timezone.utc).isoformat()}
        if (out_dir / 'manifest.json').exists():
            old = json.loads((out_dir / 'manifest.json').read_text())
            if old['files_sha256'] != files:
                raise ValueError('Frozen adjusted checkpoints changed')
            return old
        write_json(out_dir / 'manifest.json', result)
        return result


def stitch_snapshot(old, fresh, raw, *, freeze=FREEZE, anchor_start=ANCHOR_START,
                    allowed_new_dates=FRESH_EXTRA_DATES):
    """Explicitly rebase old history, then append independently fetched prices.

    Missing old stock quotes are not filled. Only explicitly named market days
    absent from the entire old calendar may be inserted before the freeze.
    Without an overlap anchor an asset retains its old history but gets no new
    prices. The splice is not a revalidation of all historical FinMind prices.
    """
    for label, frame in [('raw', raw), ('old', old), ('fresh', fresh)]:
        if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None
                or frame.index.hasnans or not frame.index.is_unique
                or not frame.index.is_monotonic_increasing or not frame.columns.is_unique
                or not frame.index.equals(frame.index.normalize())):
            raise ValueError(label + ' requires unique ordered date-only index/columns')
    if not fresh.index.equals(raw.index) or not fresh.columns.equals(raw.columns):
        raise ValueError('Fresh prices and raw prices must be exactly aligned')
    freeze, anchor_start = pd.Timestamp(freeze), pd.Timestamp(anchor_start)
    if freeze < anchor_start:
        raise ValueError('Invalid anchor interval')
    added_days = raw.index[(raw.index <= freeze) & ~raw.index.isin(old.index)]
    permitted = {pd.Timestamp(day) for day in allowed_new_dates}
    unexpected = set(added_days) - permitted
    if unexpected:
        raise ValueError('Unapproved old-calendar holes: ' + ', '.join(str(day.date()) for day in sorted(unexpected)))
    original = old.reindex(index=raw.index, columns=raw.columns).astype(float)
    fresh = fresh.astype(float)
    valid_raw = np.isfinite(raw) & raw.gt(0)
    valid_old = np.isfinite(original) & original.gt(0)
    valid_fresh = np.isfinite(fresh) & fresh.gt(0)
    output = original.where(valid_raw & valid_old)
    overlap = (raw.index >= anchor_start) & (raw.index <= freeze)
    new_rows = (raw.index > freeze) | raw.index.isin(added_days)
    output.loc[new_rows] = np.nan
    diagnostics = []
    for sid in raw.columns:
        anchor_rows = raw.index[overlap & (valid_raw[sid] & valid_old[sid] & valid_fresh[sid]).to_numpy()]
        row = {'stock_id': sid, 'anchor_date': None, 'scale': None,
               'overlap_observations': len(anchor_rows), 'reason': 'no_common_anchor',
               'new_valid_quotes': 0, 'inserted_pre_freeze_quotes': 0}
        if len(anchor_rows):
            anchor = anchor_rows[-1]
            scale = float(fresh.at[anchor, sid] / original.at[anchor, sid])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid adjusted-price anchor ratio')
            output.loc[raw.index <= freeze, sid] *= scale
            output.loc[new_rows, sid] = fresh.loc[new_rows, sid].where(
                valid_fresh.loc[new_rows, sid] & valid_raw.loc[new_rows, sid])
            row.update(anchor_date=str(anchor.date()), scale=scale, reason='anchored',
                       new_valid_quotes=int(output.loc[new_rows, sid].notna().sum()),
                       inserted_pre_freeze_quotes=int(output.loc[added_days, sid].notna().sum()))
        diagnostics.append(row)
    return output, {'freeze_date': str(freeze.date()), 'anchor_start': str(anchor_start.date()),
                    'inserted_market_dates': [str(day.date()) for day in added_days],
                    'assets': diagnostics,
                    'no_anchor_assets': [row['stock_id'] for row in diagnostics if row['reason'] != 'anchored'],
                    'historical_revalidation': False,
                    'definition': 'old snapshot rebased by a common overlap quote; fresh tail and declared calendar holes only'}


def _liquidity(frame, raw, day, sid):
    position = frame.index.get_loc(pd.Timestamp(day))
    window = frame.iloc[max(0, position - 19):position + 1][sid]
    quotes = raw.iloc[max(0, position - 19):position + 1][sid]
    valid = np.isfinite(window) & window.gt(0) & np.isfinite(quotes) & quotes.gt(0)
    complete = len(window) == 20 and bool(valid.all())
    return {'as_of': str(pd.Timestamp(day).date()), 'observations': int(valid.sum()),
            'complete_20_sessions': complete,
            'adv20_shares': float(window.mean()) if complete else None,
            'mean_turnover20_twd': float((window * quotes).mean()) if complete else None}


def build_signals(close, other_close, raw_close, volume, companies, *,
                  start=START, signal_end=SIGNAL_END):
    """Reuse fixed original leaders and ON-at-signal gate; compute no NAV."""
    for label, frame in [('raw_close', raw_close), ('volume', volume), ('other_close', other_close)]:
        if not frame.index.equals(close.index) or not frame.columns.equals(close.columns):
            raise ValueError(label + ' must align exactly with official prices')
    turnover = raw_close * volume
    volume = volume.copy()
    volume.attrs['unit'] = 'shares'
    turnover.attrs['unit'] = 'TWD'
    result = build_diffusion(close, other_close, volume, turnover, companies,
                             start=start, signal_end=signal_end)
    trend = build_trend(close['0050'])
    accepted, rejected = gate_events(result['entries']['leader_now'], trend)
    originals = {row['event_id']: row for row in result['events']}
    for batch in (accepted, rejected):
        for row in batch:
            original = originals[row['event_id']]
            row['group_id'] = original['group_id']
            row['group_members'] = deepcopy(original['members'])
            row['group_cutoff_date'] = original['group_cutoff_date']
            row['selection_reason'] = '60-session breakout, positive excess20, volume expansion, low peer breadth; original signal trend ON required'
            row['leader_evidence'] = {key: deepcopy(original[key]) for key in (
                'leader_peer_breadth', 'leader_return20', 'benchmark_return20',
                'leader_volume_ratio', 'leader_turnover_share')}
            sid = row['members'][0]
            entry_i = close.index.get_loc(pd.Timestamp(row['entry_date']))
            row['liquidity_at_signal'] = _liquidity(volume, raw_close, row['signal_date'], sid)
            row['liquidity_before_entry'] = _liquidity(volume, raw_close, close.index[entry_i - 1], sid)
    return {'entries': accepted, 'rejections': rejected, 'diffusion': result,
            'trend': [{'date': str(day.date()), 'state': row.state,
                       'ma120': float(row.ma120) if pd.notna(row.ma120) else None,
                       'evidence_date': row.evidence_date,
                       'observation_count': int(row.observation_count)}
                      for day, row in trend.iterrows()],
            'strategy': 'fixed leader_now, ON on original signal date, entry next market session',
            'slots': 3, 'horizon': 63, 'start': start, 'signal_end': signal_end,
            'live_qualified': False, 'cohort_limitation': 'current_company_cohort_not_historical_universe'}


def official_adjusted(raw_close, events):
    """Same official reference-ratio construction as the sealed old research.

    This is for signals only. It is deliberately not a cash/share action ledger.
    The documented 0050 4-for-1 split is separate from the six equity feeds.
    """
    from skills.official_adj_factors import compute_stock_factor_series
    required = {'stock_id', 'event_date', 'ratio'}
    if not required.issubset(events.columns):
        raise ValueError('Official action table missing required fields')
    events = events.copy()
    events['event_date'] = pd.to_datetime(events['event_date'])
    if events.event_date.isna().any() or not events.event_date.eq(events.event_date.dt.normalize()).all():
        raise ValueError('Invalid official action dates')
    ratios = pd.to_numeric(events.ratio, errors='raise')
    if not (np.isfinite(ratios) & ratios.gt(0)).all():
        raise ValueError('Invalid official action ratios')
    if (events.event_date > raw_close.index[-1]).any():
        raise ValueError('Official actions extend beyond the frozen price window')
    result = raw_close.where(np.isfinite(raw_close) & raw_close.gt(0)).copy()
    for sid, group in events.groupby('stock_id'):
        if sid in result.columns:
            group = group.sort_values('event_date')
            factor = compute_stock_factor_series(group.event_date.dt.date.to_numpy(),
                                                  group.ratio.to_numpy(dtype=float),
                                                  result.index.date)
            result[sid] = result[sid] * factor
    split_date = pd.Timestamp('2025-06-18')
    if ((events.stock_id == '0050') & (events.event_date == split_date)).any():
        raise ValueError('0050 split already present in official events; prevent double adjustment')
    if '0050' in result.columns:
        result.loc[result.index < split_date, '0050'] /= 4.
    return result


def validate_benchmark_quotes(close, volume):
    """A market-wide coverage pass must not hide missing benchmark history.

    The only predeclared 0050 non-trading interval in this replay is the
    documented 2025 split suspension. Any other gap needs source investigation
    before the signal build can be sealed, even if the signal engine would
    merely report many insufficient-history months.
    """
    if '0050' not in close or not close.index.equals(volume.index):
        raise ValueError('Benchmark quote/volume calendar missing or misaligned')
    good = np.isfinite(close['0050']) & close['0050'].gt(0) & np.isfinite(volume['0050']) & volume['0050'].gt(0)
    allowed = set(pd.to_datetime(['2025-06-11', '2025-06-12', '2025-06-13', '2025-06-16', '2025-06-17']))
    if good.loc[close.index.isin(allowed)].any():
        raise ValueError('0050 has an apparent trade during its documented split suspension')
    missing = close.index[~good]
    unexpected = [str(day.date()) for day in missing if day not in allowed]
    if unexpected:
        raise ValueError('Unresolved 0050 quote/volume gaps: ' + ', '.join(unexpected[:15])
                         + (f' ({len(unexpected)} dates)' if len(unexpected) > 15 else ''))
    return {'missing_dates': [str(day.date()) for day in missing],
            'unexpected_missing_dates': [], 'validated_sessions': len(close),
            'known_split_suspension': [str(day.date()) for day in sorted(allowed)]}


def _verify_hashes(base, mapping, label):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(label + ' hash mapping is empty')
    for name, digest in mapping.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(label + ' path escapes its frozen directory')
        if sha(Path(base) / relative) != digest:
            raise ValueError(label + ' changed: ' + name)


def verify_raw_inputs(input_dir=INPUT_DIR):
    input_dir = Path(input_dir)
    info = json.loads((input_dir / 'manifest.json').read_text())
    required = {'quotes.parquet', 'companies.parquet', 'events.parquet',
                'calendar.parquet', 'market_coverage.parquet', 'benchmark_coverage.parquet'}
    if (info.get('schema') != 1 or info.get('start') != '2021-01-01'
            or info.get('end') != END
            or not required.issubset(info.get('files_sha256', {}))
            or info.get('unresolved_market_date_gaps') != []
            or info.get('benchmark_coverage', {}).get('unresolved_dates') != []):
        raise ValueError('Complete frozen raw market inputs are required')
    _verify_hashes(input_dir, info['files_sha256'], 'Raw input')
    _verify_hashes(ROOT, info['code_sha256'], 'Raw preparation code')
    _verify_hashes(ROOT, info['parent_files_sha256'], 'Raw parent')
    return info


def load_fresh(raw, *, out_dir=ADJ_DIR):
    info = verify_adjusted(out_dir)
    frames, empty = [], []
    for day in info['dates']:
        value = _checkpoint(Path(out_dir) / f'{day}.json', day)
        if value['data']:
            frames.append(pd.DataFrame(value['data'])[['stock_id', 'date', 'close']])
        elif pd.Timestamp(day) in raw.index:
            empty.append(day)
    if empty:
        raise ValueError('Fresh adjusted market data missing on open sessions: ' + ', '.join(empty))
    if not frames:
        raise ValueError('No independent adjusted observations')
    values = pd.concat(frames, ignore_index=True)
    values['date'] = pd.to_datetime(values['date'])
    values['stock_id'] = values.stock_id.astype(str)
    if values.duplicated(['stock_id', 'date']).any():
        raise ValueError('Duplicate fresh adjusted observations')
    values['close'] = pd.to_numeric(values.close, errors='raise')
    return values.pivot(index='date', columns='stock_id', values='close').reindex_like(raw)


def load_market_inputs(input_dir=INPUT_DIR):
    input_dir = Path(input_dir)
    info = verify_raw_inputs(input_dir)
    companies = pd.read_parquet(input_dir / 'companies.parquet')
    calendar = pd.read_parquet(input_dir / 'calendar.parquet')
    if (not {'date', 'is_open'}.issubset(calendar.columns)
            or calendar.date.duplicated().any() or not calendar.is_open.isin([True, False]).all()):
        raise ValueError('Invalid explicit market calendar')
    days = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, 'date'])).sort_values()
    if not len(days) or days[-1] != pd.Timestamp(END) or not days.equals(days.normalize()):
        raise ValueError('Market calendar must end on the requested replay day')
    ids = pd.Index(sorted(set(companies.stock_id) | {'0050'}))
    quotes = pd.read_parquet(input_dir / 'quotes.parquet')
    quotes['date'] = pd.to_datetime(quotes.date)
    if quotes.duplicated(['stock_id', 'date']).any():
        raise ValueError('Duplicate raw stock/day quotes')
    traded = quotes.close.gt(0) & quotes.volume.gt(0)
    if (~quotes.loc[traded, 'date'].isin(days)).any():
        raise ValueError('Valid raw trades occur outside the explicit open-session calendar')
    if not set(quotes.stock_id).issubset(ids):
        raise ValueError('Raw prices contain assets outside the frozen company cohort')
    listed_dates = pd.to_datetime(companies.set_index('stock_id').listed_date).reindex(ids)
    listed_dates.loc['0050'] = pd.Timestamp('2003-06-30')
    if listed_dates.isna().any():
        raise ValueError('Missing frozen listing dates')
    listed = pd.DataFrame(days.to_numpy()[:, None] >= listed_dates.to_numpy()[None, :], index=days, columns=ids)
    matrices = {}
    for field in ('open', 'high', 'low', 'close', 'volume'):
        frame = quotes.pivot(index='date', columns='stock_id', values=field).reindex(index=days, columns=ids)
        matrices[field] = frame.where(listed)
    matrices['volume'].attrs['unit'] = 'shares'
    validate_benchmark_quotes(matrices['close'], matrices['volume'])
    events = pd.read_parquet(input_dir / 'events.parquet')
    return matrices, companies, events, info


def historical_comparison(new, old, *, end='2025-12-31'):
    """Report historical group/leader changes without computing any returns."""
    def event_key(row):
        # Group numbers may change when another cluster is inserted. Match the
        # observed leader date/id and complete members to show this explicitly.
        return (row['leader_date'], row['leader_id'], tuple(row['members']))
    before = {event_key(row): row for row in old['events'] if row['leader_date'] <= end}
    after = {event_key(row): row for row in new['events'] if row['leader_date'] <= end}
    common = sorted(set(before) & set(after))
    old_groups = {row['month']: sorted(tuple(group['members']) for group in row['clusters'])
                  for row in old['groups'] if row['month'] <= end[:7]}
    new_groups = {row['month']: sorted(tuple(group['members']) for group in row['clusters'])
                  for row in new['groups'] if row['month'] <= end[:7]}
    changed_months = [month for month in sorted(set(old_groups) | set(new_groups))
                      if old_groups.get(month) != new_groups.get(month)]
    priority_differences = [abs(float(before[key]['priority']) - float(after[key]['priority'])) for key in common]
    return {'through': end, 'old_leaders': len(before), 'new_leaders': len(after),
            'common_leaders_by_date_stock_and_group': len(common),
            'old_only_event_ids': [before[key]['event_id'] for key in sorted(set(before) - set(after))],
            'new_only_event_ids': [after[key]['event_id'] for key in sorted(set(after) - set(before))],
            'common_leaders_with_changed_priority': sum(value > 1e-12 for value in priority_differences),
            'priority_comparison_absolute_tolerance': 1e-12,
            'maximum_common_priority_difference': max(priority_differences, default=0.),
            'changed_group_months': changed_months,
            'comparison_is_new_input_reconstruction_not_original_backtest_reproduction': True}


def _versions():
    return {'python': sys.version.split()[0], 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__}


def _code_hashes():
    return {name: sha(ROOT / name) for name in CODE}


def save_matrix(path, frame):
    temp = Path(path).with_suffix('.tmp')
    frame.rename_axis('date').reset_index().to_parquet(temp, index=False)
    temp.replace(path)


def prepare_signals(*, input_dir=INPUT_DIR, cache=CACHE, adj_dir=ADJ_DIR, old_dir=OLD_DIR, spec=SPEC):
    cache, input_dir, old_dir, spec = map(Path, (cache, input_dir, old_dir, spec))
    cache.mkdir(parents=True, exist_ok=True)
    with file_lock(cache / '.signals.lock', timeout=60):
        if (cache / 'manifest.json').exists():
            return verify_signals(cache=cache)
        if not spec.exists():
            raise ValueError('The replay protocol must exist before sealing signals')
        code, versions, spec_hash = _code_hashes(), _versions(), sha(spec)
        matrices, companies, events, raw_info = load_market_inputs(input_dir)
        raw_manifest_sha = sha(input_dir / 'manifest.json')
        old_info = json.loads((old_dir / 'inputs.json').read_text())
        old_files = {name: old_info['files_sha256'][name] for name in ('close-snapshot.parquet', 'companies.parquet')}
        _verify_hashes(old_dir, old_files, 'Old frozen input')
        old_companies = pd.read_parquet(old_dir / 'companies.parquet')
        if not companies.equals(old_companies):
            raise ValueError('The current replay must preserve the entire old frozen company cohort')
        old = pd.read_parquet(old_dir / 'close-snapshot.parquet').set_index('date')
        old.index = pd.to_datetime(old.index)
        # An exchange quote with zero shares cannot become a fresh signal price.
        observed_raw = matrices['close'].where(matrices['volume'].gt(0) & np.isfinite(matrices['volume']))
        fresh = load_fresh(observed_raw, out_dir=adj_dir)
        adjusted, splice = stitch_snapshot(old, fresh, observed_raw)
        official = official_adjusted(observed_raw, events)
        if '0050' in splice['no_anchor_assets']:
            raise ValueError('The benchmark has no independent overlap anchor')
        result = build_signals(official, adjusted, matrices['close'], matrices['volume'], companies)
        old_signal_path = old_dir / 'signals-official.json'
        old_signal_meta = json.loads((old_dir / 'signals.json').read_text())
        if sha(old_signal_path) != old_signal_meta['files_sha256']['signals-official.json']:
            raise ValueError('Old frozen signal ledger changed')
        comparison = historical_comparison(result['diffusion'], json.loads(old_signal_path.read_text()))
        prior = old.index[old.index <= pd.Timestamp(FREEZE)]
        current = official.index[official.index <= pd.Timestamp(FREEZE)]
        comparison['new_market_dates_in_old_price_window'] = [str(day.date()) for day in current.difference(prior)]
        comparison['old_dates_excluded_by_new_calendar'] = [str(day.date()) for day in prior.difference(current)]
        write_json(cache / 'signals.json', result)
        write_json(cache / 'splice-audit.json', splice)
        write_json(cache / 'historical-signal-comparison.json', comparison)
        for name, frame in {**{f'raw-{name}': frame for name, frame in matrices.items()},
                            'close-official': official, 'close-quality': adjusted}.items():
            save_matrix(cache / f'{name}.parquet', frame)
        companies.to_parquet(cache / 'companies.parquet', index=False)
        output_names = {'signals.json', 'splice-audit.json', 'historical-signal-comparison.json',
                        'companies.parquet', 'close-official.parquet', 'close-quality.parquet'}
        output_names.update(f'raw-{name}.parquet' for name in matrices)
        # Seal every source used; external input manifests retain their own
        # detailed provider/source checkpoints and transformation provenance.
        sources = {str((input_dir / 'manifest.json').relative_to(ROOT)): raw_manifest_sha,
                   str((Path(adj_dir) / 'manifest.json').relative_to(ROOT)): sha(Path(adj_dir) / 'manifest.json'),
                   str((old_dir / 'inputs.json').relative_to(ROOT)): sha(old_dir / 'inputs.json'),
                   str((old_dir / 'signals.json').relative_to(ROOT)): sha(old_dir / 'signals.json'),
                   str(old_signal_path.relative_to(ROOT)): sha(old_signal_path)}
        sources.update({str((old_dir / name).relative_to(ROOT)): digest for name, digest in old_files.items()})
        if code != _code_hashes() or versions != _versions() or spec_hash != sha(spec):
            raise ValueError('Code/runtime/protocol changed while constructing signals')
        if raw_manifest_sha != sha(input_dir / 'manifest.json'):
            raise ValueError('Raw input manifest changed while constructing signals')
        verify_raw_inputs(input_dir)
        adjusted_info = verify_adjusted(adj_dir)
        _verify_hashes(ROOT, sources, 'Signal source at completion')
        manifest = {'schema': 1, 'prepared_at': datetime.now(timezone.utc).isoformat(),
                    'start': START, 'signal_end': SIGNAL_END, 'end': END,
                    'code_sha256': code, 'runtime_versions': versions,
                    'spec_path': str(spec.relative_to(ROOT)), 'spec_sha256': spec_hash,
                    'source_files_sha256': sources,
                    'raw_input_directory': str(input_dir.relative_to(ROOT)),
                    'adjusted_directory': str(Path(adj_dir).relative_to(ROOT)),
                    'files_sha256': {name: sha(cache / name) for name in sorted(output_names)},
                    'calendar_sessions': len(official), 'assets': len(official.columns),
                    'first_input_date': str(official.index[0].date()),
                    'last_input_date': str(official.index[-1].date()),
                    'leader_count': len(result['diffusion']['events']),
                    'accepted_entry_count': len(result['entries']), 'gate_rejections': len(result['rejections']),
                    'no_anchor_assets': splice['no_anchor_assets'],
                    'finmind_calls_reserved': adjusted_info['calls_reserved'],
                    'portfolio_returns_computed': False,
                    'cohort_limitation': result['cohort_limitation'],
                    'historical_price_revalidation': False,
                    'raw_calendar_changes': raw_info.get('calendar_changes', [])}
        manifest['benchmark_quote_check'] = validate_benchmark_quotes(matrices['close'], matrices['volume'])
        write_json(cache / 'manifest.json', manifest)
        return manifest


def verify_signals(*, cache=CACHE):
    cache = Path(cache)
    info = json.loads((cache / 'manifest.json').read_text())
    expected = {'signals.json', 'splice-audit.json', 'historical-signal-comparison.json',
                'companies.parquet', 'close-official.parquet', 'close-quality.parquet',
                'raw-open.parquet', 'raw-high.parquet', 'raw-low.parquet', 'raw-close.parquet', 'raw-volume.parquet'}
    if (info.get('schema') != 1 or (info.get('start'), info.get('signal_end'), info.get('end')) != (START, SIGNAL_END, END)
            or info.get('code_sha256') != _code_hashes() or info.get('runtime_versions') != _versions()
            or set(info.get('files_sha256', {})) != expected
            or info.get('portfolio_returns_computed') is not False):
        raise ValueError('Signal schema/code/runtime/window changed')
    _verify_hashes(ROOT, {info['spec_path']: info['spec_sha256']}, 'Replay protocol')
    _verify_hashes(cache, info['files_sha256'], 'Frozen signal output')
    _verify_hashes(ROOT, info['source_files_sha256'], 'Signal source')
    verify_raw_inputs(ROOT / info['raw_input_directory'])
    verify_adjusted(ROOT / info['adjusted_directory'])
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch-adjusted', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-signals', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.fetch_adjusted:
        result = fetch_adjusted(workers=args.workers)
        print(json.dumps({'dates': len(result['dates']), 'calls_reserved': result['calls_reserved']}))
    if args.prepare_signals:
        result = prepare_signals()
        print(json.dumps({key: value for key, value in result.items() if key not in {'files_sha256', 'source_files_sha256', 'code_sha256'}}, ensure_ascii=False))
    if args.verify:
        verify_signals()
        print('Frozen million-replay signals verified')
    if not (args.fetch_adjusted or args.prepare_signals or args.verify):
        parser.error('Choose --fetch-adjusted, --prepare-signals, or --verify')


if __name__ == '__main__':
    main()
