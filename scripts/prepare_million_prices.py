#!/usr/bin/env python
"""Prepare isolated raw-price/action inputs for the NT$1m replay; never write DB.

Only the frozen diffusion company cohort plus 0050 is exported. Missing market
sessions are checked against date-validated official monthly trading records,
then bounded FinMind daily requests use the existing shared quota/cache path.
Price adjustments are reference-price ratios, NOT cash/share entitlements.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from app.file_lock import file_lock
from app.finmind import fetch_dataset, FinMindQuotaError
from app.config import load_config
from app.twse_client import TWSEClient, TWSEError, roc_date_to_west
from scripts.build_official_adj_factors import _parse_validated, month_chunks, year_chunks
from skills.official_adj_factors import OfficialAdjClient, FETCH_SPECS, events_to_dataframe
from skills import official_adj_factors as action_source

CACHE = ROOT / '.cache/million-replay-inputs'
START, END = date(2021, 1, 1), date(2026, 9, 9)
OLD_END = date(2026, 6, 23)
FINMIND_BUDGET = 30
COHORT_PATH = '.cache/diffusion-research/companies.parquet'
COHORT_MANIFEST = '.cache/diffusion-research/inputs.json'
OLD_EVENTS = '.cache/event-group-research/official-events.parquet'
OLD_EVENTS_META = '.cache/event-group-research/official-events.meta.json'
CALENDAR_URL = 'https://www.twse.com.tw/exchangeReport/FMTQIK'
QUOTE_COLUMNS = ['stock_id', 'date', 'open', 'high', 'low', 'close', 'volume']
CODE = ('scripts/prepare_million_prices.py', 'skills/official_adj_factors.py',
        'scripts/build_official_adj_factors.py', 'app/finmind.py', 'app/twse_client.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str))
    temporary.replace(path)


def save_frame(path, frame, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    atomic_json(path.with_suffix('.meta.json'), {'sha256': digest(path), 'rows': len(frame),
                                                'saved_at': now(), **metadata})


def verified_frame(path):
    path = Path(path)
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    if digest(path) != meta['sha256']:
        raise ValueError('Changed prepared snapshot: ' + path.name)
    frame = pd.read_parquet(path)
    if len(frame) != meta['rows']:
        raise ValueError('Prepared row count changed')
    return frame


def validate_companies(companies):
    needed = {'stock_id', 'name', 'listed_date', 'industry', 'market'}
    if not needed.issubset(companies) or companies.empty or companies.stock_id.duplicated().any():
        raise ValueError('Frozen company cohort is incomplete')
    if (not companies.stock_id.map(lambda x: isinstance(x, str) and bool(re.fullmatch(r'\d{4}', x))).all()
            or not companies.market.isin(['TWSE', 'TPEX']).all()
            or pd.to_datetime(companies.listed_date, errors='coerce').isna().any()):
        raise ValueError('Invalid frozen company identities/listing dates')


def normalize_quotes(frame, companies, *, requested_day=None):
    """Preserve raw zero/missing quotes, exclude prelisting rows, never ffill."""
    validate_companies(companies)
    if frame.empty:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    values = frame.rename(columns={'trading_date': 'date', 'max': 'high', 'min': 'low',
                                   'Trading_Volume': 'volume'}).copy()
    if not set(QUOTE_COLUMNS).issubset(values):
        raise ValueError('Raw quote response is missing OHLCV fields')
    values['date'] = pd.to_datetime(values['date'], errors='coerce')
    if (values.date.isna().any() or not values.date.eq(values.date.dt.normalize()).all()
            or values.date.dt.tz is not None):
        raise ValueError('Invalid raw quote dates')
    if requested_day is not None and set(values.date.dt.date) != {requested_day}:
        raise ValueError('Daily quote response does not match the requested date')
    if not values.stock_id.map(lambda x: isinstance(x, str)).all():
        raise ValueError('Stock identifiers must preserve string identity')
    ids = set(companies.stock_id) | {'0050'}
    values = values[values.stock_id.isin(ids) & values.date.between(pd.Timestamp(START), pd.Timestamp(END))].copy()
    listing = dict(zip(companies.stock_id, pd.to_datetime(companies.listed_date)))
    listing['0050'] = pd.Timestamp('2003-06-30')  # Benchmark-specific listing defense.
    values = values[values.date.ge(values.stock_id.map(listing))].copy()
    for column in QUOTE_COLUMNS[2:]:
        values[column] = pd.to_numeric(values[column], errors='coerce')
    if np.isinf(values[QUOTE_COLUMNS[2:]].to_numpy(dtype=float)).any():
        raise ValueError('Infinite raw quote values')
    if ((values.volume.dropna() < 0).any()
            or (values.volume.dropna() % 1 != 0).any()):
        raise ValueError('Volume must be integer shares, not lots or fractions')
    if values.duplicated(['stock_id', 'date']).any():
        raise ValueError('Duplicate stock/date quotes must be reconciled explicitly')
    return values[QUOTE_COLUMNS].sort_values(['date', 'stock_id']).reset_index(drop=True)


def normalize_calendar(frame):
    values = frame.rename(columns={'trading_date': 'date'}).copy()
    if not {'date', 'is_open', 'session_type', 'note'}.issubset(values):
        raise ValueError('Calendar schema incomplete')
    values['date'] = pd.to_datetime(values.date, errors='coerce')
    if (values.date.isna().any() or values.date.duplicated().any()
            or not values.is_open.isin([0, 1, False, True]).all()
            or not values.date.eq(values.date.dt.normalize()).all()):
        raise ValueError('Calendar dates/flags invalid')
    values = values[values.date.between(pd.Timestamp(START), pd.Timestamp(END))].copy()
    expected = pd.date_range(START, END)
    if set(values.date) != set(expected):
        raise ValueError('DB calendar must explicitly cover every calendar date')
    values['is_open'] = values.is_open.astype(bool)
    values['calendar_source'] = 'local_trading_calendar'
    return values.sort_values('date').reset_index(drop=True)


def market_coverage(quotes, companies, calendar):
    """Audit every open session/market; denominator retains never-quoted stocks."""
    validate_companies(companies)
    data = quotes.merge(companies[['stock_id', 'market']], on='stock_id', how='inner', validate='many_to_one')
    data['positive_close'] = data.close.gt(0) & np.isfinite(data.close)
    data['positive_volume'] = data.volume.gt(0) & np.isfinite(data.volume)
    data['valid_ohlcv'] = (data.positive_close & data.positive_volume & data.open.gt(0)
                            & data.high.ge(data[['open', 'close']].max(axis=1))
                            & data.low.le(data[['open', 'close']].min(axis=1)) & data.low.gt(0))
    aggregate = data.groupby(['date', 'market']).agg(quote_rows=('stock_id', 'size'),
        positive_close=('positive_close', 'sum'), positive_volume=('positive_volume', 'sum'),
        valid_ohlcv=('valid_ohlcv', 'sum'))
    result = []
    for day in calendar.loc[calendar.is_open, 'date']:
        for market in ('TWSE', 'TPEX'):
            expected = int((companies.market.eq(market) & pd.to_datetime(companies.listed_date).le(day)).sum())
            key = (day, market)
            observed = aggregate.loc[key].to_dict() if key in aggregate.index else dict.fromkeys(aggregate.columns, 0)
            row = {'date': day, 'market': market, 'listed_cohort_count': expected,
                   **{k: int(v) for k, v in observed.items()}}
            row['quote_coverage'] = row['quote_rows'] / expected if expected else 1.
            row['positive_close_coverage'] = row['positive_close'] / expected if expected else 1.
            row['missing_market'] = bool(expected and row['quote_rows'] == 0)
            # This identifies feed outages, not a strategy admission threshold.
            row['suspected_partial_feed'] = bool(expected and row['quote_coverage'] < .90)
            result.append(row)
    return pd.DataFrame(result)


def parse_calendar_month(payload, month):
    """FMTQIK's actual trading rows establish date evidence; absent data is unknown."""
    if not isinstance(payload, dict) or payload.get('stat') != 'OK' or not isinstance(payload.get('data'), list) or not payload['data']:
        raise ValueError('Official monthly calendar is empty or unresolved')
    days = []
    for row in payload['data']:
        if not isinstance(row, list) or not row:
            raise ValueError('Malformed official calendar row')
        observed = roc_date_to_west(str(row[0]))
        if (observed.year, observed.month) != (month.year, month.month):
            raise ValueError('Official calendar ignored the requested month')
        days.append(observed)
    if len(set(days)) != len(days):
        raise ValueError('Duplicate official calendar date')
    return set(days)


def apply_calendar_evidence(calendar, quotes, evidence):
    """Only explicitly queried months are revised; conflicting local trades block."""
    result = calendar.copy()
    observed = set(quotes.loc[quotes.close.gt(0) & quotes.volume.gt(0), 'date'].dt.date)
    changes = []
    for month, open_days in evidence.items():
        mask = (result.date.dt.year.eq(month.year) & result.date.dt.month.eq(month.month))
        for index in result.index[mask]:
            day = result.at[index, 'date'].date()
            is_open = day in open_days
            if not is_open and day in observed:
                raise ValueError('Official closed date conflicts with local trading observations: ' + str(day))
            if bool(result.at[index, 'is_open']) != is_open:
                changes.append({'date': str(day), 'old_is_open': bool(result.at[index, 'is_open']), 'is_open': is_open})
            result.at[index, 'is_open'] = is_open
            result.at[index, 'calendar_source'] = 'TWSE_FMTQIK_actual_trading_rows'
            result.at[index, 'session_type'] = 'FULL' if is_open else 'CLOSED'
    return result, changes


def patch_quotes(quotes, incoming, companies, day, missing_markets):
    """Patch missing rows in specified markets only; never overwrite existing prices."""
    incoming = normalize_quotes(incoming, companies, requested_day=day)
    market = dict(zip(companies.stock_id, companies.market))
    market['0050'] = 'TWSE'
    incoming = incoming[incoming.stock_id.map(market).isin(missing_markets)].copy()
    if incoming.empty:
        raise ValueError('FinMind returned no frozen-cohort rows for missing markets: ' + str(day))
    existing = pd.MultiIndex.from_frame(quotes[['stock_id', 'date']])
    incoming = incoming[~pd.MultiIndex.from_frame(incoming[['stock_id', 'date']]).isin(existing)]
    return pd.concat([quotes, incoming], ignore_index=True).sort_values(['date', 'stock_id']).reset_index(drop=True), len(incoming)


def load_local(folder):
    cohort_manifest = json.loads((ROOT / COHORT_MANIFEST).read_text())
    if digest(ROOT / COHORT_PATH) != cohort_manifest['files_sha256']['companies.parquet']:
        raise ValueError('Frozen diffusion company cohort changed')
    companies = pd.read_parquet(ROOT / COHORT_PATH)
    validate_companies(companies)
    if not (folder / 'companies.parquet').exists():
        save_frame(folder / 'companies.parquet', companies, source_path=COHORT_PATH, source_sha256=digest(ROOT / COHORT_PATH))
    elif not verified_frame(folder / 'companies.parquet').equals(companies):
        raise ValueError('Task company snapshot differs from the frozen cohort')
    raw_path, calendar_path = folder / 'sources/db-quotes.parquet', folder / 'sources/db-calendar.parquet'
    if not raw_path.exists() or not calendar_path.exists():
        config = load_config()
        engine = create_engine(config.db_url)
        try:
            with engine.connect() as connection:
                connection.execute(text('START TRANSACTION READ ONLY'))
                if not raw_path.exists():
                    query = "SELECT stock_id,trading_date AS date,open,high,low,close,volume FROM raw_prices WHERE trading_date BETWEEN :a AND :b AND stock_id REGEXP '^[0-9]{4}$'"
                    raw = pd.read_sql(text(query), connection, params={'a': START, 'b': END})
                    raw = normalize_quotes(raw, companies)
                    save_frame(raw_path, raw, source='local_mysql_read_only', query=query, start=str(START), end=str(END),
                               trading_money_available=False, volume_unit='shares')
                if not calendar_path.exists():
                    query = 'SELECT trading_date AS date,is_open,session_type,note FROM trading_calendar WHERE trading_date BETWEEN :a AND :b'
                    calendar = pd.read_sql(text(query), connection, params={'a': START, 'b': END})
                    save_frame(calendar_path, normalize_calendar(calendar), source='local_mysql_read_only', query=query)
                connection.rollback()
        finally:
            engine.dispose()
    return verified_frame(raw_path), companies, verified_frame(calendar_path)


def calendar_evidence(folder, months):
    client = TWSEClient(timeout=30, max_retries=0)
    evidence, records = {}, []
    for month in sorted(months):
        path = folder / f'sources/calendar-twse-{month:%Y%m}.json'
        params = {'date': month.strftime('%Y%m01'), 'response': 'json'}
        if not path.exists():
            payload = client._get_json(CALENDAR_URL, params=params)
            parse_calendar_month(payload, month)
            atomic_json(path, payload)
            atomic_json(path.with_suffix('.meta.json'), {'sha256': digest(path), 'source_url': CALENDAR_URL,
                        'query': params, 'retrieved_at': now(), 'source': 'TWSE_FMTQIK'})
        meta = json.loads(path.with_suffix('.meta.json').read_text())
        if meta['sha256'] != digest(path) or meta['query'] != params:
            raise ValueError('Calendar source changed')
        evidence[month] = parse_calendar_month(json.loads(path.read_text()), month)
        records.append({'path': str(path.relative_to(folder)), **meta})
    return evidence, records



BENCHMARK_SUSPENSIONS = {pd.Timestamp(value) for value in
                         ('2025-06-11', '2025-06-12', '2025-06-13', '2025-06-16', '2025-06-17')}


def benchmark_coverage(quotes, calendar):
    """0050 has its own mandatory audit; ordinary-stock coverage cannot replace it."""
    benchmark = quotes[quotes.stock_id.eq('0050')].set_index('date')
    rows = []
    for day in calendar.loc[calendar.is_open, 'date']:
        present = day in benchmark.index
        point = benchmark.loc[day] if present else pd.Series(dtype=float)
        close = bool(present and pd.notna(point['close']) and point['close'] > 0)
        volume = bool(present and pd.notna(point['volume']) and point['volume'] > 0)
        valid = bool(close and volume and point['open'] > 0 and point['low'] > 0
                     and point['high'] >= max(point['open'], point['close'])
                     and point['low'] <= min(point['open'], point['close']))
        if day in BENCHMARK_SUSPENSIONS:
            if close and volume:
                raise ValueError('Benchmark has an apparent fill on an official split suspension')
            status = 'official_split_suspension'
        else:
            status = 'observed' if valid else 'unresolved'
        rows.append({'date': day, 'quote_present': present, 'positive_close': close,
                     'positive_volume': volume, 'valid_ohlcv': valid, 'status': status})
    return pd.DataFrame(rows)


def ensure_benchmark(folder, quotes, companies, calendar, config):
    path = folder / 'sources/finmind-price-0050-full.parquet'
    requests = 0
    if not path.exists():
        response = fetch_dataset('TaiwanStockPrice', START, END, data_id='0050', token=config.finmind_token,
                                 requests_per_hour=5400, max_retries=0, timeout=30)
        requests = int(not response.attrs.get('cache_hit', False))
        if response.empty or set(response.stock_id) != {'0050'}:
            raise ValueError('Benchmark request returned no data or other identities')
        normalize_quotes(response, companies)
        save_frame(path, response, dataset='TaiwanStockPrice', start=str(START), end=str(END),
                   data_id='0050', provider_attrs=dict(response.attrs), source='finmind_shared_quota')
    response = verified_frame(path)
    if response.empty or set(response.stock_id) != {'0050'}:
        raise ValueError('Benchmark source identity changed')
    incoming = normalize_quotes(response, companies)
    old = quotes[quotes.stock_id.eq('0050')]
    overlap = old.merge(incoming, on=['stock_id', 'date'], suffixes=('_old', '_fresh'))
    conflicts = pd.Series(False, index=overlap.index)
    for key in QUOTE_COLUMNS[2:]:
        a, b = overlap[key + '_old'].to_numpy(dtype=float), overlap[key + '_fresh'].to_numpy(dtype=float)
        conflicts |= ~np.isclose(a, b, rtol=1e-10, atol=1e-8, equal_nan=True)
    if conflicts.any():
        save_frame(folder / 'benchmark_conflicts.parquet', overlap[conflicts])
        raise ValueError('Existing benchmark quotes conflict with fresh raw data; explicit reconciliation required')
    existing_dates = set(old.date)
    appended = incoming[~incoming.date.isin(existing_dates)]
    combined = pd.concat([quotes, appended], ignore_index=True).sort_values(['date', 'stock_id']).reset_index(drop=True)
    audit = benchmark_coverage(combined, calendar)
    save_frame(folder / 'benchmark_coverage.parquet', audit,
               official_suspension_source='https://www.twse.com.tw/staticFiles/news/news/tsecnews/8a8216d696b406fc0196ce27c2e90063.pdf')
    if audit.status.eq('unresolved').any():
        raise ValueError('Benchmark has unresolved daily prices; inputs cannot be sealed')
    return combined, audit, requests


def prepare_actions(folder):
    meta = json.loads((ROOT / OLD_EVENTS_META).read_text())
    if digest(ROOT / OLD_EVENTS) != meta['sha256']:
        raise ValueError('Frozen official events changed')
    for name, expected in meta['checkpoint_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen action checkpoint changed: ' + name)
    old = pd.read_parquet(ROOT / OLD_EVENTS)
    old['event_date'] = pd.to_datetime(old.event_date)
    if not old.event_date.between(pd.Timestamp(START), pd.Timestamp(OLD_END)).all():
        raise ValueError('Old action dates outside frozen range')
    client = OfficialAdjClient(delay=1.7, timeout=30, max_retries=0)
    events, provenance = [], []
    for kind, parser, fetcher in FETCH_SPECS:
        chunker = month_chunks if kind.endswith('ex_rights') else year_chunks
        for beginning, ending in chunker(OLD_END + timedelta(days=1), END):
            path = folder / f'sources/actions-{kind}-{beginning:%Y%m%d}-{ending:%Y%m%d}.json'
            if not path.exists():
                payload = getattr(client, fetcher)(beginning, ending)
                parsed = _parse_validated(parser, payload, beginning, ending, kind)
                atomic_json(path, payload)
                atomic_json(path.with_suffix('.meta.json'), {'sha256': digest(path), 'source_kind': kind,
                            'fetch_method': fetcher, 'source_url': getattr(action_source, kind.upper() + '_URL'), 'query_start': str(beginning), 'query_end': str(ending),
                            'retrieved_at': now(), 'parsed_events': len(parsed)})
            metadata = json.loads(path.with_suffix('.meta.json').read_text())
            if (digest(path) != metadata['sha256'] or metadata['query_start'] != str(beginning)
                    or metadata['query_end'] != str(ending) or metadata['source_kind'] != kind):
                raise ValueError('Action source changed')
            parsed = _parse_validated(parser, json.loads(path.read_text()), beginning, ending, kind)
            if len(parsed) != metadata['parsed_events']:
                raise ValueError('Action parse count changed')
            events.extend(parsed)
            provenance.append({'path': str(path.relative_to(folder)), **metadata})
            print(f'actions {kind} {beginning}..{ending}: {len(parsed)}', flush=True)
    added = events_to_dataframe(events)
    combined = pd.concat([old, added], ignore_index=True)
    combined['event_date'] = pd.to_datetime(combined.event_date)
    keys = ['stock_id', 'event_date', 'market', 'source', 'event_type']
    if combined.duplicated(keys).any():
        raise ValueError('Overlapping action records require explicit reconciliation')
    save_frame(folder / 'events.parquet', combined.sort_values(keys).reset_index(drop=True),
               previous_events_sha256=digest(ROOT / OLD_EVENTS), extension_start=str(OLD_END + timedelta(days=1)),
               extension_end=str(END), cash_payment_dates_available=False)
    return provenance


def verify(folder=CACHE):
    report = json.loads((folder / 'manifest.json').read_text())
    if report['schema'] != 1 or (report['start'], report['end']) != (str(START), str(END)):
        raise ValueError('Unsupported million-replay inputs')
    for name, expected in report['code_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Preparation code changed: ' + name)
    for name, expected in report['files_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or digest(folder / path) != expected:
            raise ValueError('Prepared input changed: ' + name)
    for name, expected in report['parent_files_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Parent input changed: ' + name)
    return report


def prepare(folder=CACHE, *, local_only=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        return verify(folder)
    started = time.perf_counter()
    quotes, companies, calendar = load_local(folder)
    before = market_coverage(quotes, companies, calendar)
    missing = before[before.missing_market | before.suspected_partial_feed]
    dates = sorted(set(missing.date.dt.date))
    print(f'local raw rows={len(quotes)}; company cohort={len(companies)}; suspect feed dates={len(dates)}', flush=True)
    if local_only:
        return {'suspect_dates': list(map(str, dates)), 'local_rows': len(quotes)}
    if len(dates) > FINMIND_BUDGET:
        raise ValueError(f'{len(dates)} suspected feed dates exceed the {FINMIND_BUDGET}-request plan; review before collection')
    months = {day.replace(day=1) for day in dates}
    evidence, calendar_sources = calendar_evidence(folder, months)
    calendar, calendar_changes = apply_calendar_evidence(calendar, quotes, evidence)
    interim = market_coverage(quotes, companies, calendar)
    missing = interim[interim.missing_market | interim.suspected_partial_feed]
    target_dates = sorted(set(missing.date.dt.date))
    config, request_count, patched = load_config(), 0, []
    for day in target_dates:
        markets = sorted(missing.loc[missing.date.eq(pd.Timestamp(day)), 'market'].tolist())
        path = folder / f'sources/finmind-price-{day}.parquet'
        if not path.exists():
            if request_count >= FINMIND_BUDGET:
                raise ValueError('FinMind request budget exhausted')
            response = fetch_dataset('TaiwanStockPrice', day, day, data_id=None, token=config.finmind_token,
                                     requests_per_hour=5400, max_retries=0, timeout=30)
            request_count += int(not response.attrs.get('cache_hit', False))
            # Preserve complete provider fields, including actual Trading_money if supplied.
            normalize_quotes(response, companies, requested_day=day)
            save_frame(path, response, dataset='TaiwanStockPrice', start=str(day), end=str(day),
                       data_id=None, provider_attrs=dict(response.attrs), source='finmind_shared_quota')
        response = verified_frame(path)
        quotes, count = patch_quotes(quotes, response, companies, day, markets)
        patched.append({'date': str(day), 'markets': markets, 'rows_added': count, 'source_path': str(path.relative_to(folder))})
        print(f'patched {day} {markets}: {count}', flush=True)
    quotes, benchmark_audit, benchmark_requests = ensure_benchmark(folder, quotes, companies, calendar, config)
    request_count += benchmark_requests
    after = market_coverage(quotes, companies, calendar)
    unresolved = after[after.missing_market | after.suspected_partial_feed]
    if not unresolved.empty:
        save_frame(folder / 'unresolved_coverage.parquet', unresolved)
        raise ValueError('Market/date feed gaps remain; prepared inputs were not sealed')
    actions = prepare_actions(folder)
    save_frame(folder / 'quotes.parquet', quotes, price_basis='unadjusted_raw', volume_unit='shares',
               full_history_trading_money_available=False)
    save_frame(folder / 'calendar.parquet', calendar, official_months=[str(v) for v in sorted(months)])
    save_frame(folder / 'market_coverage.parquet', after)
    save_frame(folder / 'market_coverage_before.parquet', before)
    output_names = ('companies', 'quotes', 'calendar', 'events', 'market_coverage', 'market_coverage_before', 'benchmark_coverage')
    files = [folder / (name + suffix) for name in output_names for suffix in ('.parquet', '.meta.json')]
    source_prefixes = ('db-', 'calendar-twse-', 'actions-', 'finmind-price-')
    files += [path for path in (folder / 'sources').iterdir() if path.is_file()
              and path.name.startswith(source_prefixes) and not path.name.endswith('.tmp')]
    # Root-owned dividends/execution feeds are deliberately separate provenance.
    # They may continue to grow without invalidating frozen signal inputs.
    finmind_sources = list((folder / 'sources').glob('finmind-price-*.meta.json'))
    request_total = sum(not json.loads(path.read_text())['provider_attrs'].get('cache_hit', False) for path in finmind_sources)
    parents = (COHORT_PATH, COHORT_MANIFEST, OLD_EVENTS, OLD_EVENTS_META)
    manifest = {'schema': 1, 'prepared_at': now(), 'start': str(START), 'end': str(END),
        'code_sha256': {name: digest(ROOT / name) for name in CODE},
        'parent_files_sha256': {name: digest(ROOT / name) for name in parents},
        'files_sha256': {str(path.relative_to(folder)): digest(path) for path in sorted(files)},
        'cohort_rows': len(companies), 'quote_rows': len(quotes), 'calendar_open_days': int(calendar.is_open.sum()),
        'market_coverage_path': 'market_coverage.parquet',
        'benchmark_coverage': {'rows': len(benchmark_audit), 'observed_days': int(benchmark_audit.status.eq('observed').sum()),
            'official_suspension_dates': [str(day.date()) for day in benchmark_audit.loc[benchmark_audit.status.eq('official_split_suspension'), 'date']],
            'unresolved_dates': []},
        'original_suspect_dates': list(map(str, dates)),
        'calendar_changes': calendar_changes, 'calendar_sources': calendar_sources,
        'gaps_resolved': patched, 'unresolved_market_date_gaps': [], 'action_sources': actions,
        'finmind_requests_this_run': request_count, 'finmind_requests': request_total,
        'finmind_request_budget': FINMIND_BUDGET, 'finmind_requests_per_hour': 5400,
        'elapsed_seconds': round(time.perf_counter() - started, 3),
        'limitations': ['Frozen current ordinary-stock cohort; historical delisted/new companies are not reconstructed.',
            'Missing individual quotes and zero-volume sessions remain visible; market coverage does not prove fillability.',
            'Raw database has no actual Trading_money. close*volume is an estimate, not traded amount.',
            'Action ratios are price references, not complete dividend payment or share-conversion entitlements.',
            'Official TWSE actual-date checks cover suspect months; other dates retain the local calendar.',
            '0050 split is separate from the six equity action feeds and must use the already documented issuer evidence.']}
    atomic_json(folder / 'manifest.json', manifest)
    return verify(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-only', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    try:
        with file_lock(CACHE / 'prepare.lock', timeout=0):
            result = verify() if args.verify else prepare(local_only=args.local_only)
        print(json.dumps({k: v for k, v in result.items() if k not in {'files_sha256', 'code_sha256', 'action_sources', 'calendar_sources'}}, ensure_ascii=False, default=str))
    except FinMindQuotaError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(75)
