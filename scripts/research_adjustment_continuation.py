#!/usr/bin/env python3
"""Offline, bounded-window adjustment and membership audit; never writes trading DBs."""
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from app.file_lock import file_lock
from scripts.build_official_adj_factors import month_chunks, year_chunks
from scripts.research_exit_scenarios import read, write, sha
from skills.official_adj_factors import (FETCH_SPECS, RATIO_LO, RATIO_HI, EVENT_COLUMNS,
    validate_events_in_range, _tpex_tables_rows, events_to_dataframe, build_factor_frame)

START, END = date(2016, 1, 1), date(2026, 9, 9)
CACHE = ROOT / '.cache/readiness-continuation-20260914'
SOURCES = (ROOT / '.cache/readiness-remediation-20260914/actions-verified/checkpoints',
           ROOT / 'artifacts/adj_official/checkpoints')
RESOLUTION = ROOT / 'docs/official_action_resolution_20260914.json'


def checked_events(kind, parser, payload, start, end, source_sha256, resolution):
    """Keep raw evidence and permit only the specifically reviewed alias duplicate."""
    events = parser(payload)
    validate_events_in_range(events, start, end, kind)
    raw = _tpex_tables_rows(payload, kind) if kind.startswith('tpex_') else payload.get('data', []) or []
    if len(events) != len(raw):
        raise ValueError('Parser omitted source rows: '+kind)
    groups = defaultdict(list)
    for event in events:
        groups[(event.stock_id, str(event.event_date), event.market, event.source)].append(event)
    retained, resolutions = [], []
    for key, group in groups.items():
        if len(group) > 1:
            expected = tuple(resolution['key'])
            economic = [{k:v for k,v in asdict(e).items() if k != 'payload'} for e in group]
            if (source_sha256 != resolution['source_sha256'] or key != expected
                    or len(group) != resolution['expected_rows']
                    or any(e != economic[0] for e in economic[1:])):
                raise ValueError('Unresolved duplicate adjustment: '+str(key))
            resolutions.append(dict(key=list(key), source_rows=len(group), retained_events=1,
                                    raw_payloads=[e.payload for e in group]))
        retained.append(group[0])
    return retained, resolutions, len(raw)


def collect_events():
    resolution = read(RESOLUTION)
    all_events, sources, resolutions = [], {}, []
    raw_count = 0
    for kind, parser, _ in FETCH_SPECS:
        chunks = year_chunks if kind.endswith(('capital_reduction', 'par_value_change')) else month_chunks
        for start, end in chunks(START, END):
            name = f'{kind}_{start:%Y%m%d}_{end:%Y%m%d}.json'
            path = next((root/name for root in SOURCES if (root/name).is_file()), None)
            if path is None:
                raise ValueError('Missing official window: '+name)
            digest = sha(path)
            events, resolved, count = checked_events(kind, parser, read(path), start, end, digest, resolution)
            sources[str(path.relative_to(ROOT))] = digest
            all_events.extend(events); resolutions.extend(resolved); raw_count += count
    eligible = [e for e in all_events if len(e.stock_id) == 4 and e.stock_id.isdigit()]
    invalid = [e for e in eligible if e.ratio is None or not RATIO_LO <= e.ratio <= RATIO_HI]
    if invalid:
        raise ValueError('Invalid action ratios require explicit review: '+str([(e.stock_id, str(e.event_date)) for e in invalid]))
    frame = events_to_dataframe(eligible)
    if len(frame) != len(eligible):
        raise ValueError('Unexpected event omission across windows')
    return frame, dict(source_sha256=sources, source_windows=len(sources), raw_events=raw_count,
        retained_four_digit_events=len(frame), excluded_non_four_digit_events=len(all_events)-len(eligible),
        duplicate_resolutions=resolutions)


def add_benchmark_split(events, evidence):
    terms = evidence['verified_terms']
    ratio = 1 / float(terms['units_after_per_unit_before'])
    if evidence['stock_id'] != '0050' or ratio != float(terms['exact_split_price_factor']):
        raise ValueError('Conflicting split terms')
    day = pd.Timestamp(terms['new_units_listing_date']).date()
    if not START <= day <= END:
        raise ValueError('Split outside audit window')
    # Never double-apply a split that a provider has since incorporated.
    if ((events.stock_id == '0050') & (pd.to_datetime(events.event_date).dt.date == day)).any():
        raise ValueError('Benchmark already has an action on split date; reconcile first')
    row = dict.fromkeys(EVENT_COLUMNS)
    row.update(stock_id='0050', event_date=day, market='TWSE', source='etf_split', event_type='split',
        ratio=ratio, ratio_opening=ratio, cash_increase_suspected=False,
        reason='Exact unit ratio from dated official disclosure', payload_json=json.dumps(evidence, ensure_ascii=False))
    return pd.concat([events, pd.DataFrame([row])], ignore_index=True)


def parse_tpex_delistings(payload, year):
    if payload.get('stat') != 'ok' or str(payload.get('date')) != str(year):
        raise ValueError('Wrong TPEx delisting response/year')
    tables = payload.get('tables', [])
    if len(tables) != 1:
        raise ValueError('Unexpected TPEx table count')
    table = tables[0]
    if table['fields'][:4] != ['股票代號', '公司名稱', '終止上櫃日期', '終止上櫃原因']:
        raise ValueError('TPEx delisting fields changed')
    if len(table['data']) != int(table['totalCount']):
        raise ValueError('Incomplete TPEx pagination')
    rows = []
    for sid, name, when, reason, _ in table['data']:
        sid = sid.strip()
        roc, month, day = map(int, when.split('-'))
        ended = date(roc+1911, month, day)
        if ended.year != year or not (len(sid) == 4 and sid.isdigit()):
            raise ValueError('Wrong date or identifier in TPEx source')
        rows.append(dict(stock_id=sid, name=name, end=str(ended), market='TPEx', reason=reason))
    if len({(r['stock_id'], r['end']) for r in rows}) != len(rows):
        raise ValueError('Duplicate TPEx delisting rows')
    return rows


def load_membership_sources():
    inputs, rows = {}, []
    for year in range(2022, 2027):
        path = CACHE / (f'tpex-delisted-year-{year}.json' if year < 2026 else 'tpex-delisted-all.json')
        source = read(path.with_suffix('.source.json'))
        if sha(path) != source['sha256']:
            raise ValueError('TPEx source hash changed')
        rows.extend(parse_tpex_delistings(read(path), year))
        inputs[str(path.relative_to(ROOT))] = sha(path)
    path = ROOT / '.cache/readiness-remediation-20260914/twse-delisted.json'
    if sha(path) != read(path.with_name('twse-delisted-source.json'))['sha256']:
        raise ValueError('TWSE delisting source hash changed')
    inputs[str(path.relative_to(ROOT))] = sha(path)
    for r in read(path):
        when = str(r['DelistingDate'])
        roc, month, day = map(int, when.split('/'))
        ended = date(roc+1911, month, day)
        if 2022 <= ended.year <= 2026:
            rows.append(dict(stock_id=r['Code'], name=r['Company'], end=str(ended), market='TWSE', reason=''))
    return [r for r in rows if r['end'] <= str(END)], inputs


def snapshot(prepare=False):
    """One repeatable-read SELECT snapshot; cache reuse requires exact recorded bytes."""
    path, manifest = CACHE/'db-prices.parquet', CACHE/'db-snapshot.json'
    if manifest.exists():
        meta = read(manifest)
        for name, digest in meta['files'].items():
            if sha(CACHE/name) != digest:
                raise ValueError('DB snapshot changed: '+name)
        return pd.read_parquet(path), pd.read_parquet(CACHE/'stocks.parquet'), meta
    if not prepare:
        raise ValueError('Missing DB snapshot; use --prepare-db for read-only SELECTs')
    if path.exists():
        raise ValueError('Interrupted snapshot requires a new cache directory')
    from sqlalchemy import text
    from app.db import get_session
    import pyarrow as pa
    import pyarrow.parquet as pq
    started = datetime.now(timezone.utc).isoformat()
    with get_session() as session:
        session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ'))
        session.execute(text('START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY'))
        writer = None
        try:
            query = text('SELECT p.stock_id,p.trading_date,p.close,f.adj_factor FROM raw_prices p '
                'LEFT JOIN price_adjust_factors f ON f.stock_id=p.stock_id AND f.trading_date=p.trading_date '
                "WHERE p.trading_date BETWEEN :start AND :end AND p.stock_id REGEXP '^[0-9]{4}$'")
            for chunk in pd.read_sql(query, session.connection(), params=dict(start=START, end=END), chunksize=100000):
                chunk['trading_date'] = pd.to_datetime(chunk['trading_date'])
                for col in ('close', 'adj_factor'): chunk[col] = chunk[col].astype(float)
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None: writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
            stocks = pd.read_sql(text('SELECT stock_id,name,market,listed_date,delisted_date FROM stocks'), session.connection())
            stocks.to_parquet(CACHE/'stocks.parquet', index=False)
        finally:
            if writer is not None: writer.close()
    meta = dict(started_at=started, completed_at=datetime.now(timezone.utc).isoformat(),
        range=[str(START), str(END)], database_mutations=0,
        files={name:sha(CACHE/name) for name in ('db-prices.parquet', 'stocks.parquet')})
    write(manifest, meta)
    return pd.read_parquet(path), stocks, meta


def compare_factors(events, quotes):
    if quotes.duplicated(['stock_id', 'trading_date']).any():
        raise ValueError('Duplicate snapshot keys')
    if not quotes.trading_date.between(pd.Timestamp(START), pd.Timestamp(END)).all():
        raise ValueError('Cannot extrapolate outside event window')
    factors = build_factor_frame(events, quotes[['stock_id', 'trading_date']])
    factors.trading_date = pd.to_datetime(factors.trading_date)
    if not np.isfinite(factors.adj_factor).all() or (factors.adj_factor <= 0).any():
        raise ValueError('Invalid cumulative factors')
    merged = quotes.merge(factors, on=['stock_id','trading_date'], how='left', suffixes=('_old','_new'))
    valid = np.isfinite(merged.adj_factor_old) & (merged.adj_factor_old > 0) & merged.adj_factor_new.notna()
    merged['relative_factor_change'] = np.where(valid, merged.adj_factor_new/merged.adj_factor_old-1, np.nan)
    changed = merged.loc[valid & (merged.relative_factor_change.abs() > 1e-6)]
    return factors, merged, dict(snapshot_rows=len(quotes), shadow_rows=len(factors), shadow_stocks=factors.stock_id.nunique(),
        common_valid_rows=int(valid.sum()), changed_rows=len(changed), changed_stocks=changed.stock_id.nunique(),
        missing_old_factor_rows=int((merged.adj_factor_old.isna() | (merged.adj_factor_old <= 0)).sum()),
        no_official_event_stocks=sorted(set(quotes.stock_id)-set(factors.stock_id)),
        unclipped_rows_below_0_1=int((factors.adj_factor < .1).sum()),
        date_range=[str(factors.trading_date.min().date()), str(factors.trading_date.max().date())])


def parse_twse_listings(payload):
    if payload.get('stat') != 'OK' or len(payload['data']) != int(payload['total']):
        raise ValueError('Incomplete TWSE listing table')
    fields = payload['fields']
    needed = ('公司代號', '公司簡稱', '股票上市買賣日期', '備註')
    if not set(needed).issubset(fields):
        raise ValueError('TWSE listing fields changed')
    rows = []
    for raw in payload['data']:
        record = dict(zip(fields, raw))
        when = str(record['股票上市買賣日期']).strip()
        if not when: continue  # An application is not a completed listing.
        roc, month, day = map(int, when.split('.'))
        sid = str(record['公司代號']).strip()
        if not (len(sid)==4 and sid.isdigit()):
            raise ValueError('Unexpected listing identifier')
        rows.append(dict(stock_id=sid, name=record['公司簡稱'], start=str(date(roc+1911,month,day)),
                         market='TWSE', note=record['備註']))
    return rows


def parse_recent_tpex_listings(payload):
    if payload.get('stat')!='ok' or len(payload.get('tables',[]))!=1:
        raise ValueError('Invalid TPEx recent-listing response')
    table=payload['tables'][0]
    if table['fields'][:4]!=['索引','股票代號','公司名稱','上櫃日期']:
        raise ValueError('TPEx recent-listing fields changed')
    rows=[]
    for raw in table['data']:
        _,sid,name,when,*_=raw
        roc,month,day=map(int,when.split('/'))
        if not (len(sid)==4 and sid.isdigit()): raise ValueError('Invalid TPEx listing identifier')
        rows.append(dict(stock_id=sid,name=name,start=str(date(roc+1911,month,day)),market='TPEx',
                         note='Recent-listing snapshot; not the complete historical listing universe'))
    return rows


def audit_membership(rows, quotes, stocks, listings=()):
    groups = {sid:g.trading_date for sid,g in quotes.groupby('stock_id')}
    master = stocks.set_index('stock_id')
    result = []
    for row in rows:
        sid = row['stock_id']; days = groups.get(sid, pd.Series([], dtype='datetime64[ns]'))
        end = pd.Timestamp(row['end'])
        transfers = [r for r in listings if row['market']=='TPEx' and r['market']=='TWSE' and r['stock_id']==sid
                     and r['start']==row['end'] and '櫃轉市' in r['note']]
        reverse = [r for r in listings if row['market']=='TWSE' and r['market']=='TPEx' and r['stock_id']==sid
                   and r['start']==row['end']]
        starts = [r for r in listings if row['market']==r['market'] and r['stock_id']==sid and r['start']<row['end']]
        result.append(dict(**row, prices_before=int((days < end).sum()), prices_on_or_after=int((days >= end).sum()),
            db_listed_date=None if sid not in master.index or pd.isna(master.at[sid,'listed_date']) else str(master.at[sid,'listed_date']),
            same_day_transfer_to_twse=transfers, same_day_transition_to_tpex=reverse, official_market_starts=starts,
            post_end_classification=('confirmed_transfer_to_twse' if transfers else 'corroborated_transition_to_tpex' if reverse else 'unresolved'),
            eligible_for_automatic_historical_cohort=False))
    return result


def candidate_impact(events, output):
    """Use frozen prices, preserve missing observations, and compare existing signals only."""
    manifest_path = ROOT/'.cache/cash-allocation-inputs/manifest.json'
    manifest = read(manifest_path)
    refs = {name:manifest['references'][name] for name in ('quotes','close_official','signals','calendar')}
    inputs = {str(manifest_path.relative_to(ROOT)):sha(manifest_path)}
    for ref in refs.values():
        if sha(ROOT/ref['path']) != ref['sha256']:
            raise ValueError('Frozen research input changed: '+ref['path'])
        inputs[ref['path']] = ref['sha256']
    signals = read(ROOT/refs['signals']['path'])['entries']
    ids = sorted({'0050'} | {sid for entry in signals for sid in entry['members']})
    quotes = pd.read_parquet(ROOT/refs['quotes']['path'], columns=['stock_id','date','close'],
                             filters=[('stock_id','in',ids)]).rename(columns={'date':'trading_date'})
    if not quotes.trading_date.between(pd.Timestamp(START),pd.Timestamp(END)).all():
        raise ValueError('Frozen quote outside official event window')
    factors = build_factor_frame(events, quotes[['stock_id','trading_date']])
    factor_ids = set(factors.stock_id)
    factors.trading_date = pd.to_datetime(factors.trading_date)
    merged = quotes.merge(factors, on=['stock_id','trading_date'], how='left', validate='one_to_one')
    merged['adjusted_close'] = merged.close.where(merged.close>0)*merged.adj_factor
    old = pd.read_parquet(ROOT/refs['close_official']['path']).set_index('date')[ids]
    calendar = pd.read_parquet(ROOT/refs['calendar']['path'])
    days = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, 'date']))
    days = days[(days>=old.index.min()) & (days<=old.index.max())]
    if not old.index.equals(days):
        raise ValueError('Frozen comparison index does not match the recorded trading calendar')
    new = merged.pivot(index='trading_date', columns='stock_id', values='adjusted_close').reindex(index=days,columns=ids)
    returns = [frame/frame.shift(20)-1 for frame in (old,new)]
    checked = []
    for entry in signals:
        day = pd.Timestamp(entry['signal_date'])
        for sid in entry['members']:
            a,b = [frame.at[day,sid] for frame in returns]
            x,y = [frame.at[day,'0050'] for frame in returns]
            valid = all(np.isfinite(v) for v in (a,b,x,y))
            checked.append(dict(stock_id=sid,signal_date=str(day.date()),
                old_return20=float(a) if np.isfinite(a) else None,
                revised_return20=float(b) if np.isfinite(b) else None,
                old_benchmark_return20=float(x) if np.isfinite(x) else None,
                revised_benchmark_return20=float(y) if np.isfinite(y) else None,
                difference_pp=float((b-a)*100) if np.isfinite(a) and np.isfinite(b) else None,
                positive_excess_changed=bool((a>x)!=(b>y)) if valid else None,
                unresolved_reason=None if valid else ('No official event-derived factor; factor 1 was not assumed' if sid not in factor_ids else 'Missing price/factor endpoint')))
    write(output/'candidate-momentum.json', checked)
    merged[merged.stock_id=='0050'].to_parquet(output/'benchmark-shadow.parquet',index=False)
    return dict(input_sha256=inputs,candidates=len(checked),
        changed_above_0_1pp=sum(r['difference_pp'] is not None and abs(r['difference_pp'])>.1 for r in checked),
        positive_excess_changed=sum(r['positive_excess_changed'] is True for r in checked),
        unresolved=sum(r['positive_excess_changed'] is None for r in checked),
        note='Existing candidates only, not full reselection or an executable return. Frozen adjusted lineage may differ from raw prices; this comparison does not prove causal attribution to one correction.')


def run(output, prepare_db=False):
    output = Path(output)
    if output.exists(): raise ValueError('Use a new output directory; preserve prior evidence')
    code_names = ('scripts/research_adjustment_continuation.py','skills/official_adj_factors.py','scripts/build_official_adj_factors.py')
    code_hashes = {name:sha(ROOT/name) for name in code_names}
    events, meta = collect_events()
    split_path = ROOT/'docs/benchmark_split_evidence_20260914.json'
    split = read(split_path)
    schedule = split['schedule_source']
    if sha(ROOT/schedule['local_path']) != schedule['sha256']:
        raise ValueError('Dated benchmark disclosure changed')
    events = add_benchmark_split(events, split)
    rows, membership_inputs = load_membership_sources()
    listings_path = CACHE/'twse-newlisting.json'
    if sha(listings_path) != read(listings_path.with_suffix('.source.json'))['sha256']:
        raise ValueError('TWSE listing source changed')
    listings = parse_twse_listings(read(listings_path))
    membership_inputs[str(listings_path.relative_to(ROOT))] = sha(listings_path)
    tpex_listing_path=CACHE/'tpex-newlisting.json'
    if sha(tpex_listing_path)!=read(tpex_listing_path.with_suffix('.source.json'))['sha256']:
        raise ValueError('TPEx recent-listing source changed')
    listings.extend(parse_recent_tpex_listings(read(tpex_listing_path)))
    membership_inputs[str(tpex_listing_path.relative_to(ROOT))]=sha(tpex_listing_path)
    other_inputs = {**membership_inputs,str(split_path.relative_to(ROOT)):sha(split_path),str(RESOLUTION.relative_to(ROOT)):sha(RESOLUTION),schedule['local_path']:schedule['sha256']}
    with file_lock(CACHE/'db-snapshot.lock', timeout=0):
        quotes, stocks, db_meta = snapshot(prepare_db)
    factors, differences, summary = compare_factors(events, quotes)
    membership = audit_membership(rows, quotes, stocks, listings)
    output.mkdir(parents=True)
    events.to_parquet(output/'events.parquet', index=False)
    factors.to_parquet(output/'factors.parquet', index=False)
    differences.loc[differences.relative_factor_change.abs() > 1e-6].to_parquet(output/'factor-differences.parquet', index=False)
    write(output/'membership.json', membership)
    impact = candidate_impact(events, output)
    for name,digest in {**code_hashes,**other_inputs,**meta['source_sha256'],**impact['input_sha256']}.items():
        if sha(ROOT/name)!=digest: raise ValueError('Input or code changed during audit: '+name)
    report = dict(observed_at=datetime.now(timezone.utc).isoformat(), actions=meta, factors=summary,
        membership_summary={market:dict(events=sum(r['market']==market for r in membership),
            with_post_end_prices=sum(r['market']==market and r['prices_on_or_after']>0 for r in membership),
            missing_listing_date=sum(r['market']==market and r['db_listed_date'] is None for r in membership),
            confirmed_transfers_to_twse=sum(r['market']==market and bool(r['same_day_transfer_to_twse']) for r in membership),
            corroborated_transitions_to_tpex=sum(r['market']==market and bool(r['same_day_transition_to_tpex']) for r in membership),
            official_start_found=sum(r['market']==market and bool(r['official_market_starts']) for r in membership)) for market in ('TWSE','TPEx')},
        candidate_impact=impact, db_snapshot=db_meta, other_input_sha256=other_inputs, code_sha256=code_hashes,
        output_sha256={p.name:sha(p) for p in output.iterdir()}, network_calls=0, database_mutations=0, live_qualified=False,
        limitations=['Only existing raw-price dates; missing historical prices and market intervals remain unresolved.',
            'Shadow adjustment series is retrospective research, not proof of historical publication timestamps or delivered cash/shares.',
            'No-event securities are explicitly excluded, not silently assigned factor 1.',
            'Delisting may be a transfer; post-end prices are review items, not automatically deleted.',
            'No production cutover, broker order, or new strategy performance claim.'])
    write(output/'report.json', report)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prepare-db', action='store_true')
    args = ap.parse_args()
    result = run(args.output, args.prepare_db)
    print(json.dumps({k:result[k] for k in ('factors','membership_summary','network_calls','database_mutations','live_qualified')}, ensure_ascii=False, indent=2))
