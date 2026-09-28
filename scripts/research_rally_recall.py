#!/usr/bin/env python3
"""Offline missed-rally attribution and three frozen entry-rule diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from skills.rally_recall import NAMED, HORIZON, signals, sample, labels, phase, summarize, legacy_barrier
from skills.verified_backtest_tool import offline_only
from scripts.research_exit_scenarios import sha, write
from scripts.research_surge_anatomy import clean

SOURCE = ROOT/'artifacts/forward_simulation/historical_selector_replay_20260925.json'
ACCOUNT = ROOT/'.cache/mixed-odd-20260928/full-a/cases/strategy_normal.json'
SPEC = ROOT/'docs/prereg_rally_recall_20260928.md'


def read(path): return json.loads(path.read_text())


def run(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.cache'):
        raise ValueError('Choose a new immutable directory under .cache')
    started = time.monotonic()
    with offline_only():
        pub = read(SOURCE)
        manifest = ROOT/pub['run_manifest']['path']
        if sha(manifest) != pub['run_manifest']['sha256']: raise ValueError('Selector manifest changed')
        folder = manifest.parent/'combined'
        refs = {SOURCE: sha(SOURCE), manifest: sha(manifest)}
        for name, digest in read(manifest)['files_sha256'].items():
            if name.startswith('combined/'):
                path = manifest.parent/name
                if sha(path) != digest: raise ValueError('Sealed input changed: '+name)
                refs[path] = digest
        # The mixed account itself must match its source-bound published descriptor.
        mixed = read(ROOT/'artifacts/forward_simulation/mixed_odd_full_20260928.json')
        descriptor = mixed['cases']['strategy_normal']['result']
        if ROOT/descriptor['path'] != ACCOUNT or sha(ACCOUNT) != descriptor['sha256']:
            raise ValueError('Mixed account identity changed')
        refs[ACCOUNT] = sha(ACCOUNT)
        refs[ROOT/'artifacts/forward_simulation/mixed_odd_full_20260928.json'] = sha(ROOT/'artifacts/forward_simulation/mixed_odd_full_20260928.json')
        for path in (SPEC, Path(__file__), ROOT/'skills/rally_recall.py', ROOT/'skills/regime_state.py',
                     ROOT/'skills/diffusion_signals.py', ROOT/'tests/test_rally_recall.py'):
            refs[path] = sha(path)
        def matrix(name):
            f = pd.read_parquet(folder/(name+'.parquet')).set_index('date')
            f.index = pd.to_datetime(f.index)
            return f
        c, q, raw, vol, eligible = [matrix(k) for k in ('close-official','close-quality','raw-close','raw-volume','eligibility')]
        eligible = eligible.reindex(columns=c.columns)
        companies = pd.read_parquet(folder/'companies.parquet')
        sealed = read(folder/'signals.json'); account = read(ACCOUNT)['account']
        computed = signals(c,q,raw,vol,eligible,companies)
        original = pd.DataFrame(False,index=c.index,columns=c.columns)
        for e in sealed['entries']: original.at[pd.Timestamp(e['signal_date']),e['members'][0]] = True
        if len(sealed['entries']) != 454 or (original & ~computed['breakout_unrestricted']).any().any():
            raise ValueError('New technical mask does not contain all original signals')
        masks = {'original': original, **{k:computed[k] for k in ('breakout_unrestricted','first_expansion')}}
        checks=[]
        # Same full-cohort causal checks at fixed calendar boundaries.
        for cutoff in ('2022-12-30','2024-12-31','2026-06-30'):
            day = c.index[c.index<=cutoff][-1]
            short = signals(*(f.loc[:day] for f in (c,q,raw,vol,eligible)),companies)
            changed = [f.copy() for f in (c,q,raw,vol,eligible)]
            for f in changed[:4]: f.loc[f.index>day] *= 1.7
            long = signals(*changed,companies)
            for k in ('eligible','breakout_unrestricted','first_expansion'):
                pd.testing.assert_frame_equal(computed[k].loc[:day],short[k])
                pd.testing.assert_frame_equal(computed[k].loc[:day],long[k].loc[:day])
            checks.append(dict(cutoff=str(day.date()),truncation=True,future_mutation=True))
        quotes_path = ROOT/'.cache/million-replay-inputs/quotes.parquet'
        refs[quotes_path]=sha(quotes_path)
        quotes=pd.read_parquet(quotes_path);quotes.date=pd.to_datetime(quotes.date)
        if quotes.duplicated(['date','stock_id']).any(): raise ValueError('Duplicate opening source')
        # Invalid raw OHLC is an unresolved label, not a silently executable open.
        quotes.loc[~(quotes.open.ge(quotes.low) & quotes.open.le(quotes.high) & quotes.low.gt(0)), 'open']=np.nan
        opening, quote_close=[quotes.pivot(index='date',columns='stock_id',values=k).reindex(index=c.index,columns=c.columns) for k in ('open','close')]
        forward, known=labels(c,q,raw,opening,quote_close)
        eligible_mean=forward.where(computed['eligible']).mean(axis=1)
        names=companies.set_index('stock_id')['name'].to_dict()
        rows=[]
        for arm, mask in masks.items():
            for i,sid in sample(mask):
                day=str(c.index[i].date());end=(str(c.index[i+HORIZON+1].date()) if i+HORIZON+1<len(c) else 'unmatured')
                rows.append(dict(arm=arm,stock_id=sid,name=names.get(sid,sid),signal_date=day,
                    entry_date=str(c.index[i+1].date()) if i+1<len(c) else None,exit_date=end,
                    phase=phase(day,end),group='named' if sid in NAMED else 'other',
                    forward_return=forward.iloc[i][sid],benchmark_return=forward.iloc[i]['0050'],
                    eligible_mean=eligible_mean.iloc[i],label_known=bool(known.iloc[i][sid]),
                    volume_ratio=computed['volume_ratio'].iloc[i][sid],relative20=computed['relative20'].iloc[i][sid]))
        observations=pd.DataFrame(rows)
        accepted={(r['arm'],r['stock_id'],r['signal_date']) for r in rows}
        cases=[]
        days=c.index
        for sid in NAMED:
            candidates=forward[sid].where(computed['eligible'][sid] & (days>='2022-01-03'))
            if not candidates.notna().any():
                cases.append(dict(stock_id=sid,name=names.get(sid,sid),reason='no_known_eligible_window'));continue
            anchor=candidates.idxmax();i=days.get_loc(anchor);end=days[i+HORIZON+1]
            lo,hi=max(0,i-10),min(len(days)-1,i+20)
            record=dict(stock_id=sid,name=names.get(sid,sid),retrospective_example=True,
                anchor_signal_date=str(anchor.date()),entry_proxy_date=str(days[i+1].date()),
                exit_proxy_date=str(end.date()),best_window_return=candidates.loc[anchor],
                benchmark_return=forward.at[anchor,'0050'],signals={},original_barriers=[],original_trades=[])
            for arm,mask in masks.items():
                matches=np.flatnonzero(mask[sid].iloc[lo:hi+1])+lo
                record['signals'][arm]=[dict(signal_date=str(days[j].date()),entry_date=str(days[j+1].date()),
                    sampled_after_cooldown=(arm,sid,str(days[j].date())) in accepted,
                    return_to_same_end=float(c.at[end,sid]/(opening.iloc[j+1][sid]*c.iloc[j+1][sid]/raw.iloc[j+1][sid])-1)
                    if known.iloc[j][sid] else None) for j in matches]
            for row in record['signals']['breakout_unrestricted']:
                day=row['signal_date']
                reason=legacy_barrier(sid,day,sealed['diffusion']['groups'],sealed['diffusion']['events'],
                    sealed['entries'],sealed['rejections'],account['tick_plans'])
                record['original_barriers'].append(dict(signal_date=day,reason=reason))
            record['original_trades']=[{k:t[k] for k in ('date','side','qty','reference_price','reason','channel')}
                for t in account['trades'] if t['stock_id']==sid and str(days[lo].date())<=t['date']<=str(end.date())]
            record['held_at_anchor']=any(h['stock_id']==sid and h['date']==str(anchor.date()) for h in account['holdings'])
            record['earlier_original_trades']=[{k:t[k] for k in ('date','side','qty','reference_price','reason','channel')}
                for t in account['trades'] if t['stock_id']==sid and t['date']<str(days[lo].date())]
            record['early_signal_failed_breakout60']=[r['signal_date'] for r in record['signals']['first_expansion']
                if not c.at[pd.Timestamp(r['signal_date']),sid]>c[sid].shift().rolling(60).max().loc[r['signal_date']]]
            cases.append(record)
        barriers=Counter()
        for (i,sid) in sample(masks['breakout_unrestricted']):
            barriers[legacy_barrier(sid,str(days[i].date()),sealed['diffusion']['groups'],sealed['diffusion']['events'],
                sealed['entries'],sealed['rejections'],account['tick_plans']).split(':')[0]]+=1
        if any(sha(p)!=h for p,h in refs.items()):raise ValueError('Research inputs changed during run')
        output.mkdir(parents=True)
        observations.to_csv(output/'observations.csv',index=False,encoding='utf-8-sig')
        report=clean(dict(schema='rally_recall_v1',completed=True,source_start=str(days[0].date()),end=str(days[-1].date()),
            statistics=summarize(observations),named_cases=cases,barriers=dict(barriers),causality_checks=checks,
            original_signals_contained=454,raw_signal_counts={k:int(v.loc['2022-01-03':'2026-09-08'].sum().sum()) for k,v in masks.items()},
            source_sha256={str(p.relative_to(ROOT)):h for p,h in refs.items()},observations_sha256=sha(output/'observations.csv'),
            network_calls=0,finmind_requests=0,database_writes=0,elapsed_seconds=time.monotonic()-started,
            live_qualified=False,unseen_validation=False,portfolio_returns_computed=False,
            limitations=['Known reconstructed cohort; incomplete historical market, not an unseen sample.',
                'Named best windows are hindsight-selected examples, never signal inputs.',
                'Adjusted next-session board opening is a price proxy, not a verified order or odd-lot fill.',
                'No costs, capital limits, exits or portfolio NAV simulated; conditional forward returns only.',
                'Matched-date eligible comparator is descriptive; overlapping stocks and dates are not independent.',
                'Labels missing raw opens or conflicting price versions remain unknown.',
                'No verified news, institutional flows or concentration features added.']))
        write(output/'report.json',report)
        return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    result=run(parser.parse_args().output)
    print(json.dumps({k:result[k] for k in ('raw_signal_counts','barriers','elapsed_seconds','statistics')},ensure_ascii=False,indent=2))
