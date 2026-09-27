#!/usr/bin/env python3
"""Bounded chip-source supplement; reuse and verify frozen evidence first."""
from datetime import date
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.config import load_config
from app.finmind import fetch_dataset
from app.file_lock import file_lock
from app.theme_chips_ui import load as load_holders
from app.historical_selector_ui import load as load_selector
from skills.account_source_preflight import read, write, digest
from skills.launch_flows import normalize
from skills.theme_chips import aggregate_week
from scripts.prepare_sector_account_sources import RequestBudget

OUTPUT = ROOT / '.cache/leader-chip-inputs-20260928'
SPEC = ROOT / 'docs/prereg_leader_chip_20260928.md'


def prepare():
    if (OUTPUT / 'manifest.json').exists():
        manifest = read(OUTPUT / 'manifest.json')
        for p, sha in manifest['sources'].items():
            if digest(ROOT / p) != sha:
                raise ValueError('Sealed chip input changed: ' + p)
        return manifest
    selector = load_selector()
    signal_path = (ROOT / selector['run_manifest']['path']).parent / 'combined/signals.json'
    entries = read(signal_path)['entries']
    ids = sorted({e['members'][0] for e in entries})
    refs = {}
    def remember(path, expected=None):
        path = Path(path); actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Source hash mismatch: ' + str(path))
        refs[str(path.relative_to(ROOT))] = actual
        return path
    remember(SPEC); remember(Path(__file__)); remember(signal_path)
    old_manifest_path = ROOT / '.cache/launch-flow-inputs-20260927/manifest.json'
    old = read(remember(old_manifest_path))
    stock_paths = {Path(p).stem:ROOT / p for p in old['stock_files']}
    budget = RequestBudget(OUTPUT / 'budget.json', maximum={'finmind':30, 'official':0})
    parts = []; added = []
    for sid in ids:
        p = stock_paths.get(sid, OUTPUT / 'raw' / (sid + '.parquet'))
        meta = p.with_suffix('.json')
        if not meta.exists():
            token = load_config().finmind_token
            raw = budget.call('finmind', sid, fetch_dataset, 'TaiwanStockInstitutionalInvestorsBuySell',
                date(2021,1,1), date(2026,9,9), token=token, data_id=sid,
                requests_per_hour=5400, max_retries=0, timeout=45)
            if raw.empty or not raw.stock_id.eq(sid).all() or not pd.to_datetime(raw.date).between('2021-01-01','2026-09-09').all():
                raise ValueError('Invalid supplement identity or range: ' + sid)
            normalize(raw)  # Validate before sealing raw bytes.
            p.parent.mkdir(parents=True, exist_ok=True); raw.to_parquet(p, index=False)
            write(meta, dict(sha256=digest(p), rows=len(raw), retrieved_at=raw.attrs.get('retrieved_at'),
                cache_hit=bool(raw.attrs.get('cache_hit',False))))
            print('supplemented',sid,len(raw),flush=True)
        if sid in stock_paths:
            remember(meta, old['sources'][str(meta.relative_to(ROOT))])
            remember(p, old['sources'][str(p.relative_to(ROOT))])
        else:
            remember(meta); added.append(sid)
        remember(p,read(meta)['sha256'])
        raw = pd.read_parquet(p)
        if not raw.stock_id.eq(sid).all():
            raise ValueError('Wrong stock in source')
        parts.append(normalize(raw))
    flow = pd.concat(parts, ignore_index=True).sort_values(['stock_id','date']).reset_index(drop=True)
    flow.to_parquet(OUTPUT / 'flows.parquet',index=False);remember(OUTPUT / 'flows.parquet')
    holders = load_holders()
    descriptor = holders['artifacts']['weekly']
    weekly = pd.read_csv(remember(ROOT / descriptor['path'],descriptor['sha256']), dtype={'stock_id':str})
    weekly['date'] = pd.to_datetime(weekly.date)
    absent = sorted(set(ids)-set(weekly.stock_id))
    raw_weeks = {}
    for p, sha in holders['source_sha256'].items():
        path = Path(p)
        if path.suffix == '.parquet' and path.parent.name in ('theme-chip-inputs-20260927','theme-chip-calendar-20260927'):
            if path.stem in raw_weeks:
                raise ValueError('Ambiguous weekly source')
            raw_weeks[path.stem] = (ROOT / p, sha)
    extra = []
    for day in sorted(weekly.date.unique()):
        stamp = str(pd.Timestamp(day).date())
        p, sha = raw_weeks[stamp]
        extra.append(aggregate_week(pd.read_parquet(remember(p,sha)), stamp, absent))
    weekly = pd.concat([weekly[weekly.stock_id.isin(ids)], *extra], ignore_index=True)
    weekly = weekly.sort_values(['stock_id','date']).reset_index(drop=True)
    weekly.to_parquet(OUTPUT / 'weekly.parquet',index=False);remember(OUTPUT / 'weekly.parquet')
    # Disclose revisions against the independently collected day-level source.
    other = ROOT / '.cache/absorption-inputs-20260927/normalized.parquet'
    meta = read(remember(other.parent / 'manifest.json'))
    remember(other,meta['source_sha256'][str(other.relative_to(ROOT))])
    overlap = flow.merge(pd.read_parquet(other),on=['stock_id','date'],suffixes=('_stock','_day'))
    mismatch = pd.Series(False,index=overlap.index)
    for actor in ('foreign','trust'):
        a,b = overlap[actor+'_stock'],overlap[actor+'_day']
        mismatch |= ~(a.eq(b)|(a.isna()&b.isna()))
    disagreements = overlap.loc[mismatch, ['stock_id','date','foreign_stock','foreign_day','trust_stock','trust_day']].copy()
    disagreements['date'] = disagreements.date.dt.strftime('%Y-%m-%d')
    # JSON nulls explicitly preserve unavailable values.
    write(OUTPUT / 'revision_comparison.json',dict(overlap_rows=len(overlap), changed_rows=int(mismatch.sum()),
        differences=__import__('json').loads(disagreements.to_json(orient='records')), fixed_source='per_stock'))
    remember(OUTPUT / 'revision_comparison.json')
    manifest = dict(schema='leader_chip_inputs_v1', sources=refs, signals=str(signal_path.relative_to(ROOT)),
        stocks=len(ids), events=len(entries), supplemented_stocks=added, rebuilt_weekly_stocks=absent,
        budget=budget.state, historical_first_publication_verified=False)
    write(OUTPUT / 'manifest.json',manifest)
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fetch',action='store_true')
    if not parser.parse_args().fetch:parser.error('Bounded preparation requires --fetch')
    with file_lock(ROOT / '.cache/leader-chip-prepare.lock', timeout=0):
        value=prepare()
    print('ready',value['stocks'],value['events'],'attempts',value['budget']['attempts'],flush=True)
