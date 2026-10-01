#!/usr/bin/env python3
"""Fixed screening associations on sealed signal outcomes, not portfolio returns."""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from scripts.diagnose_signal_losses import metrics
from skills.liquidity_diagnostics import liquidity_features, market_breadth
from skills.trial_registry import append_trial_registry

ARMS = ('all', 'median50m', 'prior50m', 'persistent50m', 'breadth_rising')


def summarize(group, whole):
    result = metrics(group)
    for name, condition in [('loss', lambda f: f.net_return.lt(0)),
                            ('winner50', lambda f: f.net_return.ge(.5)),
                            ('winner100', lambda f: f.net_return.ge(1))]:
        n = int(condition(whole).sum())
        kept = int(condition(group).sum())
        result[name+'_removed'] = n-kept
        result[name+'_removed_fraction'] = (n-kept)/n if n else None
    return result


def matched(frame, arm):
    cells = []
    for (quarter, extension), group in frame[frame.status.eq('closed')].groupby(
            ['quarter', 'extension_group'], observed=True):
        yes, no = group[group[arm]], group[~group[arm]]
        if min(len(yes), len(no)) < 5:
            continue
        cells.append(dict(quarter=quarter, extension=str(extension), yes=len(yes), no=len(no),
                          weight=len(yes)*len(no)/(len(yes)+len(no)),
                          win_delta=float(yes.net_return.gt(0).mean()-no.net_return.gt(0).mean()),
                          mean_delta=float(yes.net_return.mean()-no.net_return.mean())))
    total = sum(c['weight'] for c in cells)
    return dict(cells=cells, covered=sum(c['yes']+c['no'] for c in cells),
                weighted_win_delta=sum(c['weight']*c['win_delta'] for c in cells)/total if total else None,
                weighted_mean_delta=sum(c['weight']*c['mean_delta'] for c in cells)/total if total else None)


def run(output):
    if output.exists():
        raise ValueError('Use a new output directory')
    started = time.monotonic()
    source = ROOT/'artifacts/forward_simulation/independent_signals_20261001'
    report = read(source/'report.json')
    refs = {}
    def bind(path, expected=None):
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError('Source changed: '+str(path))
        refs[str(path.relative_to(ROOT))] = digest
    for name in ('close-official', 'eligibility', 'quotes-unmasked'):
        path = ROOT/f'.cache/partial-risk-2019-20260929/inputs-final/{name}.parquet'
        bind(path, report['source_sha256'][str(path.relative_to(ROOT))])
    bind(source/'signals.parquet', report['exports_sha256']['signals.parquet'])
    bind(source/'report.json')
    for path in ('docs/prereg_liquidity_screen_20261001.md', 'skills/liquidity_diagnostics.py',
                 'scripts/research_liquidity_screen.py', 'scripts/diagnose_signal_losses.py'):
        bind(ROOT/path)
    base = ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
    c = pd.read_parquet(base/'close-official.parquet').set_index('date')
    eligibility = pd.read_parquet(base/'eligibility.parquet').set_index('date')
    q = pd.read_parquet(base/'quotes-unmasked.parquet')
    raw, volume = [q.pivot(index='date', columns='stock_id', values=k).reindex(
        index=c.index, columns=c.columns) for k in ('close', 'volume')]
    f = liquidity_features(raw, volume)
    breadth = market_breadth(c, raw, volume, eligibility)
    # A real-data prefix check supplements the synthetic future-mutation tests.
    cut = pd.Timestamp('2025-12-31')
    prefix = liquidity_features(raw.loc[:cut], volume.loc[:cut])
    for key in f:
        pd.testing.assert_frame_equal(f[key].loc[:cut], prefix[key])
    pd.testing.assert_frame_equal(breadth.loc[:cut], market_breadth(
        c.loc[:cut], raw.loc[:cut], volume.loc[:cut], eligibility.loc[:cut]))
    p = pd.read_parquet(source/'signals.parquet')
    if not p.event_id.is_unique or len(p) != report['all_signals']['signals']:
        raise ValueError('Signal population differs')
    rows, cols = c.index.get_indexer(pd.to_datetime(p.signal_date)), c.columns.get_indexer(p.stock_id)
    entry = c.index.get_indexer(pd.to_datetime(p.entry_date))
    if min(rows.min(), cols.min(), entry.min()) < 0:
        raise ValueError('Signal date or stock outside frozen inputs')
    for key, values in f.items():
        p[key] = values.to_numpy()[rows, cols]
    p['breadth'] = breadth.fraction.to_numpy()[rows]
    p['breadth_change5'] = breadth.change5.to_numpy()[rows]
    p['breadth_denominator'] = breadth.eligible_count.to_numpy()[rows]
    p['extension'] = (c/c.rolling(20, min_periods=20).mean()-1).to_numpy()[rows, cols]
    features = ['mean20', 'median20', 'prior_mean20', 'breadth', 'breadth_change5', 'extension']
    if not np.isfinite(p[features]).all().all() or not p.mean20.ge(50_000_000).all():
        raise ValueError('Missing diagnostic features or original liquidity threshold mismatch')
    p['all'] = True
    p['median50m'] = p.median20.ge(50_000_000)
    p['prior50m'] = p.prior_mean20.ge(50_000_000)
    p['persistent50m'] = p.median50m & p.prior50m
    p['breadth_rising'] = p.breadth.ge(.5) & p.breadth_change5.ge(0)
    p['mature'] = entry+63 < len(c)
    p['year'] = p.signal_date.str[:4]
    p['quarter'] = pd.to_datetime(p.signal_date).dt.to_period('Q').astype(str)
    p['extension_group'] = pd.cut(p.extension, [-np.inf, .1, .2, np.inf], right=False).astype(str)
    kept, blocked = [], {}
    for row in p.sort_values(['entry_date', 'event_id']).to_dict('records'):
        if row['entry_date'] <= blocked.get(row['stock_id'], ''):
            continue
        kept.append(row['event_id'])
        blocked[row['stock_id']] = row['exit_date'] if row['status']=='closed' else '9999-12-31'
    p['original_nonoverlap'] = p.event_id.isin(kept)
    mature = p[p.mature]
    assert not mature.status.isin(['open', 'pending_exit']).any()
    results = {}
    output.mkdir(parents=True)
    for arm in ARMS:
        selected = mature[mature[arm]]
        results[arm] = dict(selected=summarize(selected, mature), rejected=metrics(mature[~mature[arm]]),
            years={year:summarize(g[g[arm]], g) for year,g in mature.groupby('year')},
            nonoverlap=summarize(selected[selected.original_nonoverlap], mature[mature.original_nonoverlap]),
            adjusted=matched(mature, arm) if arm!='all' else None)
        append_trial_registry(dict(source='liquidity_screen_association_20261001', arm=arm,
            output=str(output.relative_to(ROOT)), completed=True, portfolio_backtest=False,
            descriptive_only=True, source_sha256=refs, unseen_validation=False, live_qualified=False),
            registry_path=output/'trials.jsonl')
    p.to_parquet(output/'signals.parquet', index=False)
    summary = dict(source_sha256=refs, results=results, total_signals=len(p),
        all_signal_filter_counts={arm:int(p[arm].sum()) for arm in ARMS},
        data_end=str(c.index[-1].date()), mature_last_entry=str(c.index[-64].date()),
        real_data_prefix_check=str(cut.date()), descriptive_only=True,
        portfolio_backtest=False, unseen_validation=False, live_qualified=False,
        exports_sha256={'signals.parquet':sha(output/'signals.parquet')})
    for path, digest in refs.items():
        if sha(ROOT/path) != digest:
            raise ValueError('Source changed during analysis')
    write(output/'report.json', summary)
    for record in (output/'trials.jsonl').read_text().splitlines():
        append_trial_registry(json.loads(record))
    print(json.dumps(dict(seconds=time.monotonic()-started, results={
        arm:result['selected'] for arm,result in results.items()}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.relative_to(ROOT)
    run(destination)
