"""Fixed 31-rule account experiment; candidates are sealed before any replay.

The scanner remains the rule authority. This module supplies only a common
ranking, a dated signal boundary and an adapter to the frozen financial runner.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.cache/all-entry-accounts-20261011'
BUNDLE = ROOT / '.cache/poc-latest-20261003/inputs-v1'
PREREG = ROOT / 'docs/prereg_all_entry_accounts_20261011.md'
START, END = '2024-01-02', '2026-10-02'
CONTEXT = ('entry_contraction_narrow', 'entry_peer_narrow')
CONTEXT_REPORT = ROOT / '.cache/rally-context-20261007/run-v2/report.json'
CONTEXT_SHA = 'fc302ce2e30e057071b14bce4ef1659e1ad617a77b2359e4e0a6b78688c5ebfc'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Refusing to replace sealed evidence: ' + str(path))
    if not path.exists():
        path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def registry():
    from skills.strategy_scanner.catalog import get_catalog
    general = [dict(x) for x in get_catalog() if x['status'] == 'active' and x['kind'] == 'entry']
    if len(general) != 29 or len({x['id'] for x in general}) != 29:
        raise ValueError('The registered scanner must contain exactly 29 entry rules')
    return general + [dict(id=CONTEXT[0], name='前期價格收斂＋市場廣度偏弱', kind='entry'),
                      dict(id=CONTEXT[1], name='價格同儕強勢＋市場廣度偏弱', kind='entry')]


def event(strategy, stock, signal, next_day, amount):
    if not np.isfinite(amount) or amount <= 0:
        raise ValueError('Candidate needs a finite positive T-known common priority')
    return dict(event_id=f'{strategy}-{signal}-{stock}', strategy_id=strategy,
                signal_date=signal, entry_date=next_day, members=[stock],
                priority=float(amount), group_id=f'{strategy}-{signal[:7]}',
                group_members=[stock], group_cutoff_date=None,
                selection_reason='registered daily entry rule; common signal-day amount20 ranking',
                leader_evidence=dict(amount20=float(amount), information_cutoff=signal))


def verify_refs(refs):
    for name, expected in refs.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError('Account experiment source changed: ' + name)


def prepare(output=BASE / 'inputs-v1', *, batch_size=128):
    from skills.strategy_scanner.data import load_bundle
    from skills.strategy_scanner.engine import _prepare, _compile_rules
    output = Path(output).resolve()
    output.relative_to(BASE)
    if output.exists():
        raise ValueError('Choose a new candidate output directory')
    definitions = registry()
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    refs = dict(manifest['source_sha256'])
    for name, sha in manifest['files_sha256'].items():
        refs[str((BUNDLE / name).relative_to(ROOT))] = sha
    refs[str((BUNDLE / 'manifest.json').relative_to(ROOT))] = digest(BUNDLE / 'manifest.json')
    refs[str(CONTEXT_REPORT.relative_to(ROOT))] = CONTEXT_SHA
    if digest(CONTEXT_REPORT) != CONTEXT_SHA:
        raise ValueError('Frozen context source changed')
    context = json.loads(CONTEXT_REPORT.read_text())
    refs.update(context['source_sha256'])
    refs[context['features']['path']] = context['features']['sha256']
    for source in (Path(__file__), PREREG, *ROOT.joinpath('skills/strategy_scanner').glob('*.py')):
        refs[str(source.relative_to(ROOT))] = digest(source)
    print('verify candidate sources', len(refs), flush=True)
    verify_refs(refs)
    data = load_bundle(BUNDLE, START, END)
    bars, days = data['bars'], pd.DatetimeIndex(data['calendar'])
    # The account runner also needs the unchanged pre-warmup market calendar.
    full_days = pd.DatetimeIndex(pd.read_parquet(BUNDLE / 'eligibility.parquet', columns=['date']).date)
    dates = days.strftime('%Y-%m-%d').tolist()
    general = [x['id'] for x in definitions if x['id'] not in CONTEXT]
    entries = {sid: [] for sid in general + list(CONTEXT)}
    pending = {sid: [] for sid in entries}
    counts = {sid: Counter() for sid in entries}
    context_rows = pd.read_parquet(ROOT / context['features']['path'])
    context_rows = context_rows.loc[context_rows.cohort.eq('legacy_course_breakout')
        & context_rows.signal_date.between(START, END)].copy()
    context_keys = {(r.stock_id, r.signal_date): r for r in context_rows.itertuples()}
    seen_context = set()
    stocks = sorted(x for x in bars.stock_id.unique() if not x.startswith('0'))
    for offset in range(0, len(stocks), batch_size):
        batch = stocks[offset:offset + batch_size]
        # Relative-strength rules require the benchmark in every stock batch.
        # It supplies context only and is explicitly excluded from candidates.
        f, actual_days, ids = _prepare(bars.loc[bars.stock_id.isin(['0050', *batch])], days, pd.Timestamp(END))
        z, masks, _, _ = _compile_rules(f, actual_days, ids,
            original_signals=data['original_signals'], provenance=data['provenance'])
        in_scope = (days >= START) & (days <= END)
        rank = z['amount20']
        rank_known = rank.notna() & np.isfinite(rank) & rank.gt(0)
        for strategy in general:
            # POC is evaluated against its full red-candidate opportunity set
            # by the necessary-evidence provider before account admission.
            source = 'original_red' if strategy == 'poc_up_red' else strategy
            matched, known, _, _ = masks[source]
            eligible = f['eligible'].eq(True).fillna(False)
            eligible.loc[:, '0050'] = False
            available = known & eligible
            chosen = (matched & available & rank_known).to_numpy(bool)
            chosen[~in_scope] = False
            counts[strategy]['unknown_stock_days'] += int((eligible & ~known).loc[in_scope].to_numpy().sum())
            counts[strategy]['unknown_common_rank'] += int((matched & available & ~rank_known).loc[in_scope].to_numpy().sum())
            for i, j in zip(*np.where(chosen)):
                row = event(strategy, ids[j], dates[i], dates[i+1] if i+1 < len(dates) else None, rank.iat[i, j])
                (entries if row['entry_date'] else pending)[strategy].append(row)
        # Reuse the frozen T-only context rows, never their forward labels.
        parent_match, parent_known, _, _ = masks['legacy_course_breakout']
        parent_known = parent_known & f['eligible'].eq(True).fillna(False)
        parent_first = parent_match & parent_known & parent_known.shift(1, fill_value=False) & ~parent_match.shift(1, fill_value=False)
        for (stock, signal), row in context_keys.items():
            if stock not in ids:
                continue
            i, j = days.get_loc(signal), ids.index(stock)
            if not parent_first.iat[i, j]:
                raise ValueError('Frozen context parent differs from financial source: ' + stock + ' ' + signal)
            ratio = z['contraction_ratio'].iat[i, j]
            if not np.isclose(ratio, row.contraction_ratio, equal_nan=True, rtol=1e-10, atol=1e-10):
                raise ValueError('Frozen context contraction differs from financial source')
            seen_context.add((stock, signal))
            for strategy, condition in zip(CONTEXT, (row.contraction, row.peer_breadth)):
                counts[strategy]['parent_events'] += 1
                if pd.isna(condition) or pd.isna(row.market_breadth):
                    counts[strategy]['unknown_context'] += 1
                elif bool(condition) and not bool(row.market_breadth):
                    if not rank_known.iat[i, j]:
                        counts[strategy]['unknown_common_rank'] += 1
                        continue
                    item = event(strategy, stock, signal, dates[i+1] if i+1 < len(dates) else None, rank.iat[i, j])
                    (entries if item['entry_date'] else pending)[strategy].append(item)
        print('candidate batch', min(offset+batch_size, len(stocks)), '/', len(stocks), flush=True)
    if seen_context != set(context_keys):
        raise ValueError('Context rows are outside financial source universe/calendar')
    files = {}
    for strategy in entries:
        key = lambda e: (e['signal_date'], -e['priority'], e['members'][0])
        entries[strategy].sort(key=key)
        pending[strategy].sort(key=key)
        counts[strategy].update(executable=len(entries[strategy]), pending=len(pending[strategy]))
        payload = dict(strategy_id=strategy, entries=entries[strategy], pending_entries=pending[strategy],
                       counts=dict(counts[strategy]), poc_gate_on_demand=strategy == 'poc_up_red')
        files[strategy+'.json'] = write(output / (strategy+'.json'), payload)
    files['calendar.json'] = write(output / 'calendar.json', full_days.strftime('%Y-%m-%d').tolist())
    verify_refs(refs)
    value = dict(schema='all_entry_account_candidates_v1', start=START, end=END,
                 definitions=definitions, files_sha256=files, source_sha256=refs,
                 counts={k: dict(v) for k,v in counts.items()}, ranking='T_amount20_desc_stock_id_asc',
                 general_signal_policy='daily_matched', context_signal_policy='known_first_parent',
                 outcomes_read=False, live_qualified=False)
    h = write(output / 'manifest.json', value)
    (output / 'manifest.sha256').write_text(h+'\n')
    return value


def load_candidates(directory):
    directory = Path(directory).resolve()
    directory.relative_to(BASE)
    if digest(directory/'manifest.json') != (directory/'manifest.sha256').read_text().strip():
        raise ValueError('Candidate manifest changed')
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest['start'], manifest['end']) != (START, END):
        raise ValueError('Candidate period changed')
    refs = dict(manifest['source_sha256'])
    for name, sha in manifest['files_sha256'].items():
        refs[str((directory/name).relative_to(ROOT))] = sha
    refs[str((directory/'manifest.json').relative_to(ROOT))] = digest(directory/'manifest.json')
    verify_refs(refs)
    by_arm = {x['id']: json.loads((directory/(x['id']+'.json')).read_text())['entries']
              for x in manifest['definitions']}
    calendar = json.loads((directory/'calendar.json').read_text())
    return by_arm, calendar, manifest, refs
