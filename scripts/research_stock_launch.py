#!/usr/bin/env python3
"""Preregistered nine-stock pre-surge anatomy; offline descriptive research only."""
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
from scripts.audit_current_causality_20260925 import provenance,matrices,BASE,ORIGINAL,verify_hashes
from scripts.research_exit_scenarios import write,sha
from scripts.research_surge_anatomy import clean
from skills.verified_backtest_tool import offline_only
from skills.stock_launch import (TARGETS,RULES,HORIZONS,features,outcomes,snapshot,episode_positions,
                                choose_controls,trajectory,paired_summary,causal_checks)
from skills.surge_statistics import rule_statistics


def run(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.cache'):
        raise ValueError('Use a new immutable directory inside the project cache')
    tick=time.perf_counter()
    with file_lock(ROOT/'.cache/stock-launch.lock',timeout=0),offline_only():
        expected=provenance()
        for name in ('docs/prereg_stock_launch_20260927.md','skills/stock_launch.py',
                     'tests/test_stock_launch.py','scripts/research_stock_launch.py',
                     'skills/surge_anatomy.py','skills/surge_statistics.py','scripts/research_surge_anatomy.py'):
            path=ROOT/name;expected[path]=sha(path)
        close,quality,raw,volume=matrices(BASE)
        companies=pd.read_parquet(ORIGINAL/'companies.parquet')
        if not set(TARGETS).issubset(companies.stock_id):raise ValueError('Named company missing from cohort')
        computed=features(close,raw,volume,companies);calendar=close.index
        positions=np.flatnonzero(calendar>=pd.Timestamp('2022-01-03'))
        tables=[];daily=[];events=[];pairs=[];timeline=[];coverage=[];statistics=[];named_coverage=[]
        for horizon in HORIZONS:
            label=outcomes(close,quality,horizon)
            chunks=[]
            for pos in positions[::21]:
                mature=pos+horizon+1<len(calendar)
                coverage.append(dict(horizon=horizon,signal_date=str(calendar[pos].date()),label_mature=mature,
                                     eligible=int(computed['eligible'].iloc[pos].sum())))
                if mature:chunks.append(snapshot(computed,label,calendar,pos,horizon))
            table=pd.concat(chunks,ignore_index=True);tables.append(table)
            for scope,frame in (('all_stocks',table),('excluding_named_nine',table[~table.stock_id.isin(TARGETS)])):
                statistics.extend(dict(horizon=horizon,cohort=scope,**row) for row in rule_statistics(frame,RULES))
            panel=pd.concat([snapshot(computed,label,calendar,p,horizon,only=list(TARGETS),eligible_only=False)
                             for p in positions],ignore_index=True);daily.append(panel)
            episodes=episode_positions(panel,calendar,horizon)
            for sid,pos in episodes:
                pool=snapshot(computed,label,calendar,pos,horizon)
                case=pool[pool.stock_id.eq(sid)].iloc[0].to_dict()
                case_id=f'{sid}_{horizon}_{case["signal_date"]}'
                controls=choose_controls(pool,case,set(TARGETS))
                case.update(case_id=case_id,matched_controls=len(controls));events.append(case)
                series=trajectory(computed,calendar,sid,pos,case_id,'case')
                for control in controls.to_dict('records'):
                    control.update(case_id=case_id,case_stock_id=sid);pairs.append(control)
                    series.extend(trajectory(computed,calendar,control['stock_id'],pos,case_id,'control'))
                timeline.extend(dict(horizon=horizon,phase=case['phase'],**row) for row in series)
            for sid,name in TARGETS.items():
                part=panel[panel.stock_id.eq(sid)]
                usable=part.eligible&part.label_mature
                named_coverage.append(dict(horizon=horizon,stock_id=sid,name=name,days=len(part),
                    eligible_days=int(part.eligible.sum()),mature_eligible_days=int(usable.sum()),
                    unknown_mature_eligible=int((usable&~part.label_known).sum()),
                    immature_eligible=int((part.eligible&~part.label_mature).sum()),
                    events=sum(s==sid for s,p in episodes)))
            print(f'horizon={horizon}: {len(table)} cohort rows, {len(episodes)} named events',flush=True)
        event_frame=pd.DataFrame(events);timeline_frame=pd.DataFrame(timeline)
        latest=[]
        for horizon in HORIZONS:
            for sid,name in TARGETS.items():
                part=event_frame[event_frame.horizon.eq(horizon)&event_frame.stock_id.eq(sid)].sort_values('signal_date')
                latest.append(dict(horizon=horizon,stock_id=sid,name=name,event_found=False) if part.empty else
                              dict(part.iloc[-1].to_dict(),name=name,event_found=True))
        summaries=[]
        for scope,frame in [('all',timeline_frame),*[(p,timeline_frame[timeline_frame.phase.eq(p)])
                                                       for p in ('discovery','replication','boundary')]]:
            summaries.extend(dict(scope=scope,**row) for row in paired_summary(frame))
        cutoffs=[calendar[calendar<=pd.Timestamp(d)][-1] for d in ('2022-12-31','2024-12-31','2026-06-30')]
        checks=causal_checks(close,raw,volume,companies,cutoffs)
        output.mkdir(parents=True);files={}
        for name,frame in (('observations',pd.concat(tables,ignore_index=True)),('named_daily',pd.concat(daily,ignore_index=True)),
                ('events',event_frame),('latest_cases',pd.DataFrame(latest)),('matched_controls',pd.DataFrame(pairs)),
                ('trajectories',timeline_frame),('paired_summary',pd.DataFrame(summaries)),
                ('rule_statistics',pd.DataFrame(statistics)),('anchor_coverage',pd.DataFrame(coverage)),
                ('named_coverage',pd.DataFrame(named_coverage))):
            path=output/f'{name}.csv';frame.to_csv(path,index=False,encoding='utf-8-sig')
            files[name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),rows=len(frame))
        verify_hashes(expected)
        report=clean(dict(schema='stock_launch_v1',completed=True,live_qualified=False,adopted=False,
            unseen_validation=False,portfolio_returns_computed=False,strategy_net_return=None,
            source_start=str(calendar[0].date()),source_end=str(calendar[-1].date()),targets=TARGETS,
            horizons={str(k):dict(absolute=v[0],excess=v[1]) for k,v in HORIZONS.items()},
            companies=len(companies),latest_cases=latest,named_coverage=named_coverage,
            named_events=len(events),matched_events=int(event_frame.matched_controls.gt(0).sum()),
            paired_summary=summaries,rule_statistics=statistics,causality_checks=checks,artifacts=files,
            source_sha256={str(p.relative_to(ROOT)):d for p,d in expected.items()},
            elapsed_seconds=round(time.perf_counter()-tick,3),finmind_requests=0,network_calls=0,database_writes=0,
            institutional_included=False,fundamentals_included=False,news_included=False,
            limitations=[
                'Named winners and launch anchors are retrospectively selected; event dates are not tradable signals.',
                'Current-company cohort and industry labels are not a complete point-in-time universe.',
                'Prices are adjusted proxies, not verified total returns; unresolved labels remain unknown.',
                'Sixty-session outcomes on 21-session anchors overlap; date bootstrap does not remove all serial dependence.',
                'Same-stock named episodes recur and industry controls use replacement; matched differences are descriptive.',
                'Every period was previously researched; replication is not untouched validation.',
                'No cost, fill or portfolio simulation; classification hit rates are not strategy returns.',
                'Historical news, financial publication times and full-universe institutional quality not validated.',
            ]))
        write(output/'report.json',report)
        write(output/'manifest.json',dict(report_sha256=sha(output/'report.json'),files=files,source_sha256=report['source_sha256']))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    result=run(parser.parse_args().output)
    print(json.dumps({k:result[k] for k in ('named_events','matched_events','elapsed_seconds','finmind_requests')},ensure_ascii=False))
