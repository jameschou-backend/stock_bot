#!/usr/bin/env python3
"""Bounded, resumable all-market institutional snapshots for fixed anchors."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from app.config import load_config
from app.file_lock import file_lock
from app.finmind import fetch_dataset
from scripts.audit_current_causality_20260925 import matrices, BASE, ORIGINAL, verify_hashes
from scripts.prepare_launch_flows import OUTPUT as OLD
from scripts.research_exit_scenarios import read, write, sha
from skills.launch_flows import normalize

OUTPUT = ROOT/'.cache/absorption-inputs-20260927'
SPEC = ROOT/'docs/prereg_absorption_20260927.md'
DATASET = 'TaiwanStockInstitutionalInvestorsBuySell'


def plan(days):
    anchors = np.flatnonzero((days >= '2022-01-03') & (days <= '2026-09-09'))[::21]
    dates = sorted({days[p-lag-k] for p in anchors for lag in (1, 3) for k in range(5)})
    return anchors, dates


def validate(frame, day, ids):
    required = {'date', 'stock_id', 'name', 'buy', 'sell'}
    if frame.empty or not required.issubset(frame):
        raise ValueError('Empty or incomplete institutional market page')
    if not pd.to_datetime(frame.date).eq(pd.Timestamp(day)).all():
        raise ValueError('Institutional page is not the requested single date')
    if frame.duplicated(['date', 'stock_id', 'name']).any():
        raise ValueError('Duplicate institutional category on market page')
    if not frame.stock_id.map(lambda x: isinstance(x, str)).all():
        raise ValueError('Provider stock identities must be strings')
    selected = frame[frame.stock_id.isin(ids)]
    if selected.stock_id.nunique() < 500:
        raise ValueError('Implausibly sparse all-market page; keep it unqualified')
    return normalize(selected), sorted(set(frame.stock_id)-set(ids))


def prepare():
    began = time.perf_counter(); OUTPUT.mkdir(parents=True, exist_ok=True)
    with file_lock(OUTPUT/'prepare.lock', timeout=0):
        manifest_path = OUTPUT/'manifest.json'
        if manifest_path.exists():
            m = read(manifest_path)
            verify_hashes({ROOT/p:h for p,h in m['source_sha256'].items()})
            if m['spec_sha256'] != sha(SPEC): raise ValueError('Frozen specification changed')
            print('Verified complete frozen inputs; 0 new requests', flush=True)
            return m
        days = matrices(BASE)[0].index; anchors, dates = plan(days)
        if len(dates) != 385 or len(dates) > 400: raise ValueError('Fixed date plan changed')
        ids = set(pd.read_parquet(ORIGINAL/'companies.parquet').stock_id)
        if len(ids) != 1974: raise ValueError('Fixed company cohort changed')
        identity = dict(spec_sha256=sha(SPEC), companies_sha256=sha(ORIGINAL/'companies.parquet'),
            dates=[str(d.date()) for d in dates], anchors=[str(days[p].date()) for p in anchors])
        stamp = OUTPUT/'plan.json'
        if stamp.exists() and read(stamp) != identity: raise ValueError('Cannot alter a started date plan')
        if not stamp.exists(): write(stamp, identity)
        journal_path = OUTPUT/'attempts.json'; journal = read(journal_path) if journal_path.exists() else []
        config = load_config()

        def task(day):
            key = str(day.date()); p = OUTPUT/(key+'.parquet'); meta = p.with_suffix('.json')
            if meta.exists():
                if sha(p) != read(meta)['sha256']: raise ValueError('Stored institutional page changed')
                validate(pd.read_parquet(p), day, ids)
                return
            frame = fetch_dataset(DATASET, day.date(), token=config.finmind_token,
                requests_per_hour=config.finmind_requests_per_hour, max_retries=0, timeout=45)
            normalized, excluded = validate(frame, day, ids)
            frame.to_parquet(p, index=False)
            write(meta, dict(sha256=sha(p), rows=len(frame), cohort_stocks=normalized.stock_id.nunique(),
                date=key, dataset=DATASET, retrieved_at=frame.attrs.get('retrieved_at'),
                cache_hit=frame.attrs.get('cache_hit', False), excluded_codes=excluded))

        with ThreadPoolExecutor(max_workers=4) as executor:
            for offset in range(0, len(dates), 4):
                batch = dates[offset:offset+4]
                missing = [d for d in batch if not (OUTPUT/(str(d.date())+'.json')).exists()]
                if len(journal)+len(missing) > 400: raise ValueError('Preparation attempt budget exhausted')
                journal.extend(dict(date=str(d.date()), started_at=datetime.now(timezone.utc).isoformat()) for d in missing)
                write(journal_path, journal)
                # Wait for the bounded batch before submitting more; every completed page is sealed.
                futures = [executor.submit(task, d) for d in batch]
                errors = []
                for future in futures:
                    try: future.result()
                    except Exception as exc: errors.append(exc)
                if errors: raise errors[0]
                if offset % 40 == 0 or offset+4 >= len(dates):
                    print('market dates', min(offset+4, len(dates)), '/', len(dates), flush=True)
        sources = {str(p.relative_to(ROOT)):sha(p) for p in (stamp, SPEC, ORIGINAL/'companies.parquet')}
        parts = []
        for day in dates:
            p = OUTPUT/(str(day.date())+'.parquet')
            parts.append(validate(pd.read_parquet(p), day, ids)[0])
            for path in (p, p.with_suffix('.json')): sources[str(path.relative_to(ROOT))] = sha(path)
        normalized = pd.concat(parts, ignore_index=True)
        p = OUTPUT/'normalized.parquet'; normalized.to_parquet(p, index=False)
        sources[str(p.relative_to(ROOT))] = sha(p)
        # Same provider, two collection times: disclose revisions, never mix values.
        old_meta = read(OLD/'manifest.json'); verify_hashes({ROOT/p:h for p,h in old_meta['sources'].items()})
        old_parts = []
        for name in old_meta['stock_files']:
            f = pd.read_parquet(ROOT/name); old_parts.append(f[pd.to_datetime(f.date).isin(dates)])
        old = normalize(pd.concat(old_parts, ignore_index=True))
        both = normalized.merge(old, on=['date','stock_id'], suffixes=('_new','_old'))
        audit = {}
        for actor in ('foreign','trust'):
            known = both[[actor+'_new',actor+'_old']].notna().all(axis=1)
            audit[actor] = dict(compared=int(known.sum()), different=int((known & both[actor+'_new'].ne(both[actor+'_old'])).sum()))
        audit_path = OUTPUT/'prior_source_comparison.json'; write(audit_path, audit)
        sources[str(audit_path.relative_to(ROOT))] = sha(audit_path)
        sources[str((OLD/'manifest.json').relative_to(ROOT))] = sha(OLD/'manifest.json')
        sources.update(old_meta['sources'])
        sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
        sources['skills/launch_flows.py'] = sha(ROOT/'skills/launch_flows.py')
        m = dict(schema='absorption_inputs_v1', spec_sha256=sha(SPEC), source_sha256=sources,
            normalized_rows=len(normalized), stocks=normalized.stock_id.nunique(), dates=len(dates),
            new_requests=sum(not read(OUTPUT/(str(d.date())+'.json'))['cache_hit'] for d in dates),
            api_attempts=len(journal), elapsed_seconds=round(time.perf_counter()-began, 3),
            prior_source_comparison=audit, historical_first_publication_verified=False)
        verify_hashes({ROOT/p:h for p,h in sources.items()}); write(manifest_path, m)
        print({k:m[k] for k in ('stocks','dates','normalized_rows','new_requests','elapsed_seconds')}, flush=True)
        return m


if __name__ == '__main__': prepare()
