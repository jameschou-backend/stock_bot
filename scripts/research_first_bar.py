#!/usr/bin/env python3
"""Frozen all-market first-bar timing study and existing-engine account attempts."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from app.file_lock import file_lock
from app.theme_chips_ui import load as load_chips
from app.stock_launch_ui import read_table
from scripts.audit_current_causality_20260925 import provenance, matrices, BASE, verify_hashes, QUARANTINE
from scripts.research_cash_allocation import INPUT
from scripts.research_exit_scenarios import read, write, sha
from scripts.research_surge_anatomy import clean
from scripts.research_theme_catalyst import source_closure, annotate_annual_periods, CACHE
from skills.first_bar import START, END, EXCLUDED, features, episodes, wait_for_breakout, with_chips, entries, outcomes, summarize
from skills.sector_account_replay import SectorAccountInputs, run_case
from skills.scenario_exit_replay import ExitSignals
from skills.verified_backtest_tool import offline_only
from skills.trial_registry import append_trial_registry

SPEC=ROOT/'docs/prereg_first_bar_20260927.md'


def inputs():
    expected=provenance();chip=load_chips()
    expected.update({ROOT/p:h for p,h in chip['source_sha256'].items()})
    ref=chip['artifacts']['weekly'];expected[ROOT/ref['path']]=ref['sha256']
    weekly=read_table(chip,'weekly');weekly['date']=pd.to_datetime(weekly.date)
    references=read(INPUT/'manifest.json')['references']
    data={}
    for name,meta in references.items():
        p=ROOT/meta['path'];expected[p]=meta['sha256']
        if name in ('quotes','companies','events','calendar'):
            data[name]=pd.read_parquet(p)
    verify_hashes(expected)
    quotes=data['quotes'];quotes['date']=pd.to_datetime(quotes.date)
    for r in read(QUARANTINE)['quarantine']:
        quotes=quotes[~(quotes.stock_id.eq(r['stock_id'])&quotes.date.eq(pd.Timestamp(r['date'])))]
    if quotes.duplicated(['stock_id','date']).any():raise ValueError('Duplicate OHLC date')
    ohlc={key:quotes.pivot(index='date',columns='stock_id',values=key) for key in ('open','high','low','close','volume')}
    frames=matrices(BASE);days=frames[0].index
    calendar=pd.DatetimeIndex(pd.to_datetime(data['calendar'].loc[data['calendar'].is_open,'date']))
    if not days.equals(calendar):raise ValueError('Account and signal calendars differ')
    account=SectorAccountInputs(quotes,data['companies'],days,{},data['events'],ExitSignals(frames[0],days),
        '2026-09-27',start=START,end=END)
    return frames,ohlc,weekly,account,expected


def causality(frames,ohlc,weekly,companies,base_events,base_wait,computed):
    close,quality,raw,volume=frames;checks=[]
    for cutoff in ('2023-12-29','2024-12-31','2026-03-31'):
        stop=pd.Timestamp(cutoff)
        for mode in ('truncate','mutate'):
            changed=[]
            for f in (close,raw,volume):
                x=f.loc[:stop].copy() if mode=='truncate' else f.copy()
                if mode=='mutate':x.loc[x.index>stop]*=1.6
                changed.append(x)
            bars={}
            for k,f in ohlc.items():
                x=f.loc[:stop].copy() if mode=='truncate' else f.copy()
                if mode=='mutate':x.loc[x.index>stop]*=1.6
                bars[k]=x
            weeks=weekly[weekly.date<=stop].copy() if mode=='truncate' else weekly.copy()
            if mode=='mutate':weeks.loc[weeks.date>stop,'large_pct_delta4']=.9
            f=features(*changed,bars,companies);ev=episodes(f);wait=wait_for_breakout(ev,f)
            old=base_events[base_events.signal_date<=cutoff].reset_index(drop=True)
            new=ev[ev.signal_date<=cutoff].reset_index(drop=True)
            pd.testing.assert_frame_equal(old,new)
            for lag in (8,15):
                a=with_chips(old,weekly,lag);b=with_chips(new,weeks,lag)
                pd.testing.assert_frame_equal(a,b)
                # An order requires the next market session already in the calendar;
                # compare only orders whose scheduled entry lies before the cutoff.
                for arm in ('first','wait'):
                    past=lambda values:[e for e in values if e['entry_date']<=cutoff]
                    if past(entries(a,base_wait,computed,arm=arm))!=past(entries(b,wait,f,arm=arm)):
                        raise ValueError('Future information changed past entry orders')
            checks.append(dict(cutoff=cutoff,mode=mode,passed=True,events=len(old),lags=[8,15],arms=['first','wait']))
            print('causality',cutoff,mode,flush=True)
    return checks


def run(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.cache'):raise ValueError('Use a new immutable cache directory')
    tick=time.perf_counter()
    with file_lock(ROOT/'.cache/first-bar.lock',timeout=0),offline_only():
        frames,ohlc,weekly,data,expected=inputs();close,quality,raw,volume=frames
        closure,overrides=source_closure(CACHE);expected.update(closure)
        for p in (SPEC,Path(__file__),ROOT/'skills/first_bar.py',ROOT/'tests/test_first_bar.py',
                  ROOT/'scripts/research_theme_catalyst.py',ROOT/'skills/theme_chips.py',
                  ROOT/'skills/surge_anatomy.py',ROOT/'app/theme_chips_ui.py',ROOT/'app/stock_launch_ui.py'):
            expected[p]=sha(p)
        computed=features(close,raw,volume,ohlc,data.companies)
        ev=episodes(computed);waits=wait_for_breakout(ev,computed)
        print('events',len(ev),'known waits',waits.wait_state.value_counts().to_dict(),flush=True)
        checks=causality(frames,ohlc,weekly,data.companies,ev,waits,computed)
        tables=[];orders={}
        for lag in (8,15):
            aligned=with_chips(ev,weekly,lag)
            table=outcomes(aligned,waits,close,quality);table['lag']=lag;tables.append(table)
            for arm in ('first','wait'):orders[f'{arm}_lag{lag}']=entries(aligned,waits,computed,arm=arm)
        table=pd.concat(tables,ignore_index=True);statistics=summarize(table)
        output.mkdir(parents=True)
        for name,frame in [('events',ev),('waits',waits),('outcomes',table),('statistics',pd.DataFrame(statistics))]:
            frame.to_csv(output/(name+'.csv'),index=False,encoding='utf-8-sig')
        write(output/'entries.json',orders)
        cases={};jobs=[]
        for board in (False,True):
            channel='board' if board else 'mixed'
            for stress in ('control','combined'):
                config=dict(arm='benchmark',stress=stress,benchmark=True,board_only=board,position_count=0)
                jobs.append((f'benchmark_{channel}_{stress}',data,config))
                for key,rows in orders.items():
                    selected=replace(data,entries_by_arm={'relative_strength':rows})
                    config=dict(arm='relative_strength',stress=stress,benchmark=False,board_only=board,position_count=5)
                    jobs.append((f'{key}_{channel}_{stress}',selected,config))
        for name,selected,config in jobs:
            append_trial_registry(dict(timestamp=datetime.now(timezone.utc).isoformat(),source='first_bar',case=name,
                output=str(output.relative_to(ROOT)),status='started',preregistration_sha256=sha(SPEC)))
            result=run_case(selected,config,CACHE,overrides);result['strategy_case']=name
            row={k:v for k,v in result.items() if k in ('completed','summary','reason','candidate_count','audit','execution')}
            if result['completed']:
                annotate_annual_periods(result['summary'])
                row['summary']=result['summary']
                row['average_cash_fraction']=sum(d['cash']/d['nav'] for d in result['account']['daily'])/len(result['account']['daily'])
                for key in ('trades','daily','cash_ledger'):
                    pd.DataFrame(result['account'][key]).to_csv(output/(name+'-'+key+'.csv'),index=False,encoding='utf-8-sig')
            write(output/(name+'.json'),clean(result))
            row['result']=dict(path=str((output/(name+'.json')).relative_to(ROOT)),sha256=sha(output/(name+'.json')))
            cases[name]=row;print(name,'completed' if result['completed'] else result['reason'],flush=True)
        for name,row in cases.items():
            if name.startswith('benchmark'):continue
            bm=cases['benchmark_'+name.split('_')[-2]+'_'+name.split('_')[-1]]
            row['benchmark_return']=bm['summary']['total_return'] if bm['completed'] else None
            row['excess_return']=(row['summary']['total_return']-row['benchmark_return']
                if row['completed'] and bm['completed'] else None)
        verify_hashes(expected)
        report=clean(dict(schema='first_bar_v1',completed=True,live_qualified=False,adopted=False,
            unseen_validation=False,historical_first_publication_verified=False,theme_filter_included=False,
            start=START,end=END,excluded_named_stocks=sorted(EXCLUDED),statistics=statistics,cases=cases,
            all_accounts_completed=all(r['completed'] for r in cases.values()),
            price_statistics_are_account_returns=False,causality_checks=checks,
            source_sha256={str(p.relative_to(ROOT)):h for p,h in expected.items()},
            elapsed_seconds=round(time.perf_counter()-tick,3),finmind_requests=0,network_calls=0,
            events=len(ev),signal_counts={k:len(v) for k,v in orders.items()},
            limitations=['Known history and current-company survivor cohort; no untouched validation.',
                'Chip release lags are assumptions; original historical first versions unverified.',
                'Weekly holder distributions cannot identify intraday buying intent.',
                'Paired price windows share one endpoint; waiting arms have fewer invested sessions.',
                'Price statistics exclude fees, fills and account allocation; no summing into strategy returns.',
                'Board-only is a separate diagnostic; it cannot replace missing mixed-lot evidence.',
                'Historical theme and news filter not included; this tests price/volume and holder proxies.']))
        write(output/'report.json',report)
        write(output/'manifest.json',dict(schema='first_bar_manifest_v1',source_sha256=report['source_sha256'],
            files_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=run(p.parse_args().output)
    print(json.dumps({k:r[k] for k in ('events','elapsed_seconds','signal_counts','all_accounts_completed')}))
