#!/usr/bin/env python3
"""Frozen, offline chip concentration study with audited news chronology."""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from app.stock_launch_ui import load as load_launch,read_table
from scripts.audit_current_causality_20260925 import ORIGINAL,verify_hashes
from scripts.research_exit_scenarios import sha,write
from scripts.research_surge_anatomy import clean
from skills.verified_backtest_tool import offline_only
from skills.stock_launch import TARGETS
from skills.surge_statistics import rule_statistics
from skills.theme_chips import aggregate_week,changes,align,conditions,increment,news_at,RULES

SOURCE_DIRS=['.cache/theme-chip-inputs-20260927','.cache/theme-chip-calendar-20260927']


def weekly_input(cohort,expected):
    sources={};empty=[];requests=0
    for name in SOURCE_DIRS:
        folder=ROOT/name;manifest=json.loads((folder/'manifest.json').read_text())
        for p in (folder/'manifest.json',folder/'plan.json'):expected[p]=sha(p)
        requests+=manifest['max_requests']
        for day,meta in manifest['files'].items():
            path=folder/f'{day}.parquet';mp=folder/f'{day}.json'
            if sha(path)!=meta['sha256'] or any(json.loads(mp.read_text()).get(k)!=meta[k]
                    for k in ('date','rows','sha256','retrieved_at','cache_hit')):
                raise ValueError('Holder source no longer matches its sealed metadata')
            expected[path]=meta['sha256'];expected[mp]=sha(mp)
            if not meta['rows']:empty.append(day);continue
            if day in sources:raise ValueError('Overlapping holder source dates')
            sources[day]=path
    if requests>280:raise ValueError('Input budget exceeded')
    actual_calendar=ROOT/'.cache/holder-case-2492/TaiwanStockHoldingSharesPer.parquet'
    supplement=json.loads((ROOT/SOURCE_DIRS[1]/'plan.json').read_text())['calendar_source']
    if sha(actual_calendar)!=supplement['sha256']:raise ValueError('Supplement calendar changed')
    expected[actual_calendar]=supplement['sha256']
    calendar=pd.to_datetime(pd.read_parquet(actual_calendar,columns=['date']).date).drop_duplicates()
    calendar=calendar[(calendar>=pd.Timestamp('2021-11-01'))&(calendar<=pd.Timestamp('2026-09-09'))]
    all_days=sorted(set(calendar.dt.strftime('%Y-%m-%d'))|set(sources))
    unresolved=[d for d in all_days if d not in sources]
    frames=[]
    for i,day in enumerate(all_days):
        raw=pd.read_parquet(sources[day]) if day in sources else pd.DataFrame()
        frames.append(aggregate_week(raw,day,cohort))
        if (i+1)%50==0:print(f'validated holder weeks={i+1}/{len(all_days)}',flush=True)
    return pd.concat(frames,ignore_index=True),dict(requests=requests,market_weeks=len(sources),
        empty_request_dates=empty,unresolved_observation_dates=unresolved)


def run(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.cache'):
        raise ValueError('Use a new immutable project-cache directory')
    started=time.perf_counter()
    with file_lock(ROOT/'.cache/theme-chip-research.lock',timeout=0),offline_only():
        launch=load_launch();expected={ROOT/p:h for p,h in launch['source_sha256'].items()}
        for artifact in launch['artifacts'].values():expected[ROOT/artifact['path']]=artifact['sha256']
        for name in ('docs/prereg_theme_chips_20260927.md','docs/theme_chip_sources_20260927.json',
                     'skills/theme_chips.py','tests/test_theme_chips.py','scripts/research_theme_chips.py',
                     'scripts/prepare_theme_chips.py','skills/surge_statistics.py','skills/holding_validation.py'):
            expected[ROOT/name]=sha(ROOT/name)
        verify_hashes(expected)
        companies=pd.read_parquet(ORIGINAL/'companies.parquet');cohort=set(companies.stock_id)
        weekly,audit=weekly_input(cohort,expected);weekly_changes=changes(weekly)
        observations=read_table(launch,'observations')
        for key in ('event','relative_strength'):observations[key]=observations[key].astype('boolean')
        news=json.loads((ROOT/'docs/theme_chip_sources_20260927.json').read_text())
        events=read_table(launch,'events');timeline=read_table(launch,'trajectories')
        for frame in (events,timeline):frame['relative_strength']=frame.relative_strength.astype('boolean')
        summaries=[];increments=[];coverage=[];observation_tables=[];case_tables=[];timeline_tables=[];checks=[]
        for lag in (8,15):
            aligned=align(observations,weekly_changes,lag)
            primary=conditions(aligned);primary['lag']=lag;observation_tables.append(primary)
            named=conditions(align(events,weekly_changes,lag));named['lag']=lag
            context=pd.DataFrame([news_at(news['events'],r.stock_id,r.signal_date) for r in named.itertuples()])
            named=pd.concat([named,context],axis=1);case_tables.append(named)
            tracked=conditions(align(timeline,weekly_changes,lag));tracked['lag']=lag;timeline_tables.append(tracked)
            for cutoff in ('2022-12-31','2024-12-31','2026-06-30'):
                queries=observations[pd.to_datetime(observations.signal_date)<=pd.Timestamp(cutoff)]
                original=align(queries,weekly_changes,lag)
                for mode in ('truncate','mutate'):
                    subset=weekly[weekly.date<=pd.Timestamp(cutoff)].copy() if mode=='truncate' else weekly.copy()
                    if mode=='mutate':subset.loc[subset.date>pd.Timestamp(cutoff),'large_pct']=.99
                    actual=align(queries,changes(subset),lag)
                    pd.testing.assert_frame_equal(original,actual)
                    checks.append(dict(lag=lag,cutoff=cutoff,mode=mode,passed=True,rows=len(actual)))
            for horizon in (20,60):
                base=aligned[aligned.horizon.eq(horizon)&~aligned.stock_id.isin(TARGETS)]
                for phase,part in base.groupby('phase',sort=True):
                    coverage.append(dict(lag=lag,horizon=horizon,phase=phase,rows=len(part),
                        chip_known=int(part.chip_known.sum()),outcome_known=int(part.event.notna().sum()),
                        joint_known=int((part.chip_known&part.event.notna()).sum()),stocks=part.stock_id.nunique()))
                for threshold in (0,.005,.01):
                    frame=conditions(base,threshold)
                    common=frame[frame.chip_known&frame.relative_strength.notna()]
                    stats=rule_statistics(common,RULES)
                    for row in stats:
                        scope=row['scope']
                        mask=(common.phase.eq(scope) if scope in ('discovery','replication') else
                              common.phase.isin(['discovery','replication']))
                        if scope not in ('all','discovery','replication'):
                            mask &= pd.to_datetime(common.exit_date).dt.year.eq(int(scope))
                        selected=common[mask&common[row['rule']].fillna(False)&common.event.notna()]
                        row.update(lag=lag,horizon=horizon,threshold=threshold,
                            mean_return=selected.forward_return.mean(),median_return=selected.forward_return.median(),
                            mean_excess=selected.excess_return.mean(),median_excess=selected.excess_return.median())
                        summaries.append(row)
                    for phase in ('discovery','replication'):
                        increments.append(dict(lag=lag,horizon=horizon,threshold=threshold,phase=phase,
                            **increment(common[common.phase.eq(phase)])))
            print(f'lag={lag}: statistics and causality checks completed',flush=True)
        output.mkdir(parents=True);artifacts={}
        tables=dict(weekly=weekly_changes,observations=pd.concat(observation_tables,ignore_index=True),
                    named_events=pd.concat(case_tables,ignore_index=True),trajectories=pd.concat(timeline_tables,ignore_index=True),
                    statistics=pd.DataFrame(summaries),incremental=pd.DataFrame(increments),coverage=pd.DataFrame(coverage))
        for name,frame in tables.items():
            path=output/f'{name}.csv';frame.to_csv(path,index=False,encoding='utf-8-sig')
            artifacts[name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),rows=len(frame))
        verify_hashes(expected)
        report=clean(dict(schema='theme_chips_v1',completed=True,live_qualified=False,adopted=False,
            unseen_validation=False,portfolio_returns_computed=False,strategy_net_return=None,
            theme_factorial_identifiable=False,theme_factorial_reason='Historical news dates failed audit; verified theme absence and complete coverage unavailable.',
            targets=TARGETS,source_start=launch['source_start'],source_end=launch['source_end'],
            chip_source_audit=audit,weekly_reasons=weekly.reason.value_counts().to_dict(),
            weekly_change_reasons=weekly_changes.change_reason.value_counts().to_dict(),
            coverage=coverage,statistics=summaries,incremental=increments,causality_checks=checks,
            news_sources=news,artifacts=artifacts,source_sha256={str(p.relative_to(ROOT)):h for p,h in expected.items()},
            elapsed_seconds=round(time.perf_counter()-started,3),finmind_requests=0,network_calls=0,database_writes=0,
            limitations=['Current-company survivorship and historical source revisions remain unresolved.',
                'Holder lag is an assumption, not a verified historical first-publication timestamp.',
                'Holder tiers do not reveal beneficiary intent, persistent buy orders or investor identity.',
                'Date-bootstrap intervals are descriptive; serial dependence and multiple research trials remain.',
                'News documents are selected positive examples, not a complete theme classification dataset.',
                'Price outcomes are not cost-adjusted or execution-qualified portfolio returns.']))
        write(output/'report.json',report)
        write(output/'manifest.json',dict(report_sha256=sha(output/'report.json'),files=artifacts,source_sha256=report['source_sha256']))
        return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    r=run(parser.parse_args().output)
    print(json.dumps({k:r[k] for k in ('completed','elapsed_seconds','finmind_requests','theme_factorial_identifiable')}))
