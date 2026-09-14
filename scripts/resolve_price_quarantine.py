#!/usr/bin/env python3
"""Verify original evidence, then reversibly remove only reviewed corrupt observations."""
from datetime import datetime, timezone
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from app.db import get_session
from app.file_lock import file_lock
from app.price_quarantine import FIELDS, load_registry, same_observation
from scripts.audit_price_chronology import normalize_source, PAIRS
from scripts.research_exit_scenarios import read, write, sha

AUDIT = ROOT/'artifacts/forward_simulation/price_chronology_20260913.json'
MEMBERSHIP = ROOT/'.cache/readiness-continuation-20260914/audit-v3/membership.json'
REGISTRY = ROOT/'docs/price_quarantine_20260914.json'


def build_registry():
    audit = read(AUDIT)
    membership = read(MEMBERSHIP)
    ends = {r['stock_id']: r for r in membership
            if r['prices_on_or_after'] and r['post_end_classification'] == 'unresolved'}
    refs = {str(AUDIT.relative_to(ROOT)): sha(AUDIT),
            str(MEMBERSHIP.relative_to(ROOT)): sha(MEMBERSHIP)}
    # Bind the membership artifact back to verified official end dates.
    from scripts.research_adjustment_continuation import load_membership_sources
    official, official_refs = load_membership_sources()
    refs.update(official_refs)
    for sid, row in ends.items():
        if not any(r['stock_id']==sid and r['end']==row['end'] and r['market']==row['market'] for r in official):
            raise ValueError('Official market end differs: '+sid)
    frames = {}
    for day, old_day in PAIRS:
        paths = [f'.cache/price-chronology-20260913/{day}.parquet',
                 f'.cache/price-chronology-20260913/old-{day}.parquet']
        for path in paths:
            if sha(ROOT/path) != audit['input_sha256'][path]:
                raise ValueError('Chronology source changed: '+path)
            refs[path] = sha(ROOT/path)
        frames[day] = (normalize_source(pd.read_parquet(ROOT/paths[0]), day),
                       pd.read_parquet(ROOT/paths[1]).set_index('stock_id'), old_day)
    rows = []
    for q in audit['quarantine']:
        sid, day = q['stock_id'], q['date']
        if sid not in ends: continue
        source, older, old_day = frames[day]
        if day < ends[sid]['end'] or sid in source.index:
            raise ValueError('Price is not an absent post-end observation')
        copy = sid in older.index and same_observation(older.loc[sid], q['original']) and q['original']['volume'] > 0
        zero = all(q['original'][k] == 0 for k in FIELDS)
        if not copy and not zero: raise ValueError('Unresolved post-end observation')
        rows.append(dict(stock_id=sid, date=day, original=q['original'],
            market=ends[sid]['market'], official_market_end=ends[sid]['end'],
            resolution='misdated_older_copy' if copy else 'zero_placeholder',
            copied_from=old_day if copy else None))
    if len(rows)!=64 or len({r['stock_id'] for r in rows})!=22:
        raise ValueError('Reviewed 22-stock scope changed; require a new audit')
    return dict(schema='reviewed_price_quarantine_v1', rows=rows, source_sha256=refs,
                scope='64 reviewed post-end rows only; other chronology conflicts remain separately quarantined',
                live_qualified=False)


def invalidated_labels(days, removed, horizon):
    """Return labels whose forward endpoint disappears/changes after removal."""
    days = sorted(days)
    clean = [d for d in days if d not in removed]
    endpoints = {d:clean[i+horizon] for i,d in enumerate(clean[:-horizon])}
    return [d for i,d in enumerate(days) if d in removed or
            (i+horizon < len(days) and endpoints.get(d) != days[i+horizon])]


def run(output, apply=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidate = build_registry()
    if REGISTRY.exists() and read(REGISTRY) != candidate:
        raise ValueError('Reviewed registry changed')
    if not REGISTRY.exists(): write(REGISTRY, candidate)
    registry = load_registry()
    backup_path = output/'backup.json'
    report_path = output/'result.json'
    from app.config import load_config
    horizon = load_config().label_horizon_days
    with file_lock(ROOT/'.cache/research-or-update.lock', timeout=0), get_session() as session:
        originals, missing, label_backup = [], [], []
        for (sid, day), item in registry.items():
            row = session.execute(text('SELECT * FROM raw_prices WHERE stock_id=:s AND trading_date=:d FOR UPDATE'),
                                  dict(s=sid,d=day)).mappings().one_or_none()
            if row is None: missing.append((sid,day));continue
            if not same_observation(row,item['original']):
                raise ValueError('Current DB differs from reviewed evidence: '+sid+'/'+day)
            originals.append({k:str(v) if v is not None else None for k,v in row.items()})
        if missing:
            if len(missing)!=len(registry) or not backup_path.exists():
                raise ValueError('Partial/missing price state without complete repair backup')
            backup = read(backup_path)
            if (backup['registry_sha256']!=sha(REGISTRY) or backup['label_horizon']!=horizon
                    or len(backup['raw_prices'])!=len(registry)
                    or {(r['stock_id'],r['trading_date']) for r in backup['raw_prices']}!=set(registry)):
                raise ValueError('Repair backup identity mismatch')
            for r in backup['raw_prices']:
                if not same_observation(r,registry[(r['stock_id'],r['trading_date'])]['original']):
                    raise ValueError('Repair backup values differ')
            for r in backup['labels']:
                if session.execute(text('SELECT 1 FROM labels WHERE stock_id=:s AND trading_date=:d'),
                                   dict(s=r['stock_id'],d=r['trading_date'])).first():
                    raise ValueError('Invalid forward label reappeared')
            return dict(applied=False, already_applied=True, remaining_reviewed_rows=0,
                        registry_sha256=sha(REGISTRY), backup_sha256=sha(backup_path))
        for sid in sorted({k[0] for k in registry}):
            days = [str(r[0]) for r in session.execute(text('SELECT trading_date FROM raw_prices WHERE stock_id=:s ORDER BY trading_date'),dict(s=sid))]
            removed = {d for s,d in registry if s==sid}
            if any(d >= min(removed) and d not in removed for d in days):
                raise ValueError('Additional post-end price requires review: '+sid)
            invalid = invalidated_labels(days,removed,horizon)
            for day in invalid:
                row = session.execute(text('SELECT * FROM labels WHERE stock_id=:s AND trading_date=:d FOR UPDATE'),dict(s=sid,d=day)).mappings().one_or_none()
                if row: label_backup.append({k:str(v) if v is not None else None for k,v in row.items()})
        backup = dict(registry_sha256=sha(REGISTRY), label_horizon=horizon,
                      raw_prices=originals, labels=label_backup)
        if backup_path.exists() and read(backup_path)!=backup:
            raise ValueError('Existing backup differs; preserve it and investigate')
        if not backup_path.exists(): write(backup_path,backup)
        if apply:
            for row in originals:
                session.execute(text('DELETE FROM raw_prices WHERE stock_id=:s AND trading_date=:d'),
                                dict(s=row['stock_id'],d=row['trading_date']))
            for row in label_backup:
                session.execute(text('DELETE FROM labels WHERE stock_id=:s AND trading_date=:d'),
                                dict(s=row['stock_id'],d=row['trading_date']))
            session.commit()
        result = dict(observed_at=datetime.now(timezone.utc).isoformat(), applied=apply,
            reviewed_stocks=22, raw_rows=len(originals), invalidated_labels=len(label_backup),
            registry_sha256=sha(REGISTRY), backup_sha256=sha(backup_path),
            features_rebuild_from='2026-06-09', derived_rebuild_completed=False, live_qualified=False)
        write(report_path,result)
        return result


def rebuild_derived(output):
    """Run from a real main module so macOS process workers can start correctly."""
    from dataclasses import replace
    from datetime import date
    import shutil
    import time
    from app.config import load_config
    from skills import build_features, data_store
    output=Path(output)
    status=run(output)
    if not status.get('already_applied'):
        raise ValueError('Apply the reviewed price repair before rebuilding derived data')
    result_path=output/'derived-rebuild.json'
    if result_path.exists():
        result=read(result_path)
        if result.get('completed') and result['registry_sha256']==sha(REGISTRY):
            return dict(already_rebuilt=True, result_sha256=sha(result_path))
        raise ValueError('Prior derived rebuild result requires review')
    with file_lock(ROOT/'.cache/research-or-update.lock',timeout=0),get_session() as session:
        old=ROOT/'artifacts/features/features_2026.parquet'
        backup=output/'features_2026-before.parquet'
        if old.exists() and not backup.exists(): shutil.copy2(old,backup)
        maxday=session.execute(text('SELECT MAX(trading_date) FROM raw_prices')).scalar()
        config=replace(load_config(),force_recompute_days=(maxday-date(2026,6,9)).days)
        started=time.perf_counter()
        result=build_features.run(config,session)
        if result.get('rows',0)<=0:
            raise ValueError('Feature rebuild produced no rows')
        data_store._ensure(session,kinds=('prices','features','labels'))
        result.update(elapsed_seconds=time.perf_counter()-started,completed=True,
            registry_sha256=sha(REGISTRY),network_calls=0,backup_features_sha256=sha(backup))
        write(result_path,result)
        return result


def verify_storage(output):
    """Verify the repaired keys in MySQL and each live derived cache after pipeline runs."""
    import duckdb
    output=Path(output)
    status=run(output)
    if not status.get('already_applied'):raise ValueError('Repair is not applied')
    backup=read(output/'backup.json')
    price_keys=pd.DataFrame([dict(stock_id=r['stock_id'],trading_date=r['trading_date']) for r in backup['raw_prices']])
    label_keys=pd.DataFrame([dict(stock_id=r['stock_id'],trading_date=r['trading_date']) for r in backup['labels']])
    checked={}
    with file_lock(ROOT/'.cache/research-or-update.lock',timeout=0),get_session() as session:
        for r in backup['raw_prices']:
            if session.execute(text('SELECT 1 FROM features WHERE stock_id=:s AND trading_date=:d'),
                               dict(s=r['stock_id'],d=r['trading_date'])).first():
                raise ValueError('Quarantined feature remains in MySQL')
        caches={'prices':'artifacts/cache/prices.parquet','features':'artifacts/cache/features.parquet',
                'labels':'artifacts/cache/labels.parquet','feature_store':'artifacts/features/features_2026.parquet'}
        with duckdb.connect() as con:
            for key,relative in caches.items():
                path=ROOT/relative
                if not path.exists():raise ValueError('Required repaired cache missing: '+relative)
                con.register('bad_keys',label_keys if key=='labels' else price_keys)
                count=con.execute('SELECT count(*) FROM read_parquet(?) p JOIN bad_keys b '
                    'ON p.stock_id=b.stock_id AND CAST(p.trading_date AS DATE)=CAST(b.trading_date AS DATE)',
                    [str(path)]).fetchone()[0]
                if count:raise ValueError('Quarantined keys remain in cache: '+relative)
                checked[key]=dict(path=relative,remaining_bad_rows=0)
        result=dict(observed_at=datetime.now(timezone.utc).isoformat(),passed=True,
            registry_sha256=sha(REGISTRY),raw_rows_absent=len(price_keys),
            invalid_label_rows_absent=len(label_keys),mysql_bad_features_absent=True,caches=checked)
        write(output/'storage-verification.json',result)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--rebuild-derived',action='store_true')
    parser.add_argument('--verify-storage',action='store_true')
    args=parser.parse_args()
    if sum([args.apply,args.rebuild_derived,args.verify_storage])>1:
        parser.error('Apply, rebuild and verify are separate checkpoints')
    import logging
    logging.basicConfig(level=logging.INFO)
    print(verify_storage(args.output) if args.verify_storage else
          rebuild_derived(args.output) if args.rebuild_derived else run(args.output,args.apply))
