#!/usr/bin/env python3
"""Fixed first-bar and retrospective winner diagnostics, with failed cases."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from app.file_lock import file_lock
from app.first_bar_ui import load as load_first
from app.backtest_tool_ui import verified_bytes
from scripts.audit_current_causality_20260925 import matrices,BASE,ORIGINAL,verify_hashes
from scripts.research_exit_scenarios import read,write,sha
from scripts.research_surge_anatomy import clean
from scripts.prepare_launch_flows import OUTPUT as INPUT
from skills.launch_flows import normalize,price_features,flow_features,snapshots,attach_rules,statistics,RULES
from skills.stock_launch import TARGETS,HORIZONS,features,outcomes,snapshot,episode_positions
from skills.verified_backtest_tool import offline_only
from skills.trial_registry import append_trial_registry

NAMES=dict(TARGETS,**{'3491':'昇達科'})


def retrospective(frames,companies):
    selected=[*NAMES,'0050'];close,quality,raw,volume=[f[selected] for f in frames]
    computed=features(close,raw,volume,companies[companies.stock_id.isin(NAMES)])
    days=close.index;positions=np.flatnonzero(days>=pd.Timestamp('2022-01-03'));events=[]
    for horizon in HORIZONS:
        label=outcomes(close,quality,horizon)
        panel=pd.concat([snapshot(computed,label,days,p,horizon,only=list(NAMES),eligible_only=False) for p in positions],ignore_index=True)
        for sid,pos in episode_positions(panel,days,horizon):
            row=panel[panel.stock_id.eq(sid)&panel.signal_date.eq(str(days[pos].date()))].iloc[0].to_dict()
            row.update(event_id=f'retro-{horizon}-{row["signal_date"]}-{sid}',named_case=True,kind='retrospective')
            events.append(row)
    return pd.DataFrame(events)


def causal_checks(close,volume,normalized,price,flow):
    checks=[]
    for cutoff in ('2023-12-29','2024-12-31','2026-03-31'):
        day=pd.Timestamp(cutoff)
        for mode in ('truncate','mutate'):
            c=close.loc[:day].copy() if mode=='truncate' else close.copy()
            v=volume.loc[:day].copy() if mode=='truncate' else volume.copy()
            f=normalized[normalized.date<=day].copy() if mode=='truncate' else normalized.copy()
            if mode=='mutate':
                c.loc[c.index>day]*=1.8;v.loc[v.index>day]*=2.3
                f.loc[f.date>day,[k for k in f if k not in ('date','stock_id','schema_supported')]]*= -7
            p=price_features(c);n=flow_features(f,c.index,v)
            for group,old in ((p,price),(n,flow)):
                for k in old:pd.testing.assert_frame_equal(old[k].loc[:day],group[k].loc[:day])
            checks.append(dict(cutoff=cutoff,mode=mode,passed=True,matrices=len(p)+len(n)))
            print('causality',cutoff,mode,flush=True)
    return checks


def run(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.cache'):raise ValueError('New immutable cache directory required')
    tick=time.perf_counter()
    with file_lock(ROOT/'.cache/launch-flows.lock',timeout=0),offline_only():
        parent,pub=load_first();manifest=read(INPUT/'manifest.json')
        expected={ROOT/p:h for p,h in parent['source_sha256'].items()}
        expected.update({ROOT/p:h for p,h in manifest['sources'].items()})
        expected[INPUT/'manifest.json']=sha(INPUT/'manifest.json')
        expected[ROOT/pub['report']['path']]=pub['report']['sha256']
        for name in ('skills/launch_flows.py','scripts/research_launch_flows.py','scripts/prepare_launch_flows.py',
                     'tests/test_launch_flows.py','skills/stock_launch.py','skills/surge_anatomy.py'):
            expected[ROOT/name]=sha(ROOT/name)
        folder=Path(pub['report']['path']).parent
        def source_csv(name):
            p=ROOT/folder/name;h=pub['reproducibility']['files_sha256'][name]
            expected[p]=h
            from io import BytesIO
            return pd.read_csv(BytesIO(verified_bytes(dict(path=str(p.relative_to(ROOT)),sha256=h),ROOT,'.csv')),dtype={'stock_id':str})
        first=source_csv('events.csv');first['kind']='first_bar'
        result=source_csv('outcomes.csv');result=result[result.lag.eq(8)].copy()
        verify_hashes(expected)
        frames=matrices(BASE);close,quality,raw,volume=frames
        companies=pd.read_parquet(ORIGINAL/'companies.parquet')
        parts=[]
        for name in manifest['stock_files']:
            f=pd.read_parquet(ROOT/name)
            if not f.stock_id.eq(Path(name).stem).all():raise ValueError('Institution file identity mismatch')
            if not pd.to_datetime(f.date).between('2021-01-01','2026-09-09').all():raise ValueError('Institution file range mismatch')
            parts.append(f)
        normalized=normalize(pd.concat(parts,ignore_index=True))
        if normalized.stock_id.nunique()!=283:raise ValueError('Institutional scope differs from preregistration')
        price=price_features(close);flow=flow_features(normalized,close.index,volume)
        checks=causal_checks(close,volume,normalized,price,flow)
        retro=retrospective(frames,companies)
        timeline=snapshots(first,price,flow,close.index)
        retros=snapshots(retro,price,flow,close.index)
        comparison=pd.concat([attach_rules(result,timeline,lag) for lag in (1,0)],ignore_index=True)
        stats=statistics(comparison)
        latest=retro[retro.horizon.eq(20)].sort_values('signal_date').groupby('stock_id',sort=True).tail(1)
        named=timeline[timeline.stock_id.isin(NAMES)].copy()
        # Exact cases, not best performers; outcomes remain clearly future columns.
        named_outcomes=result[result.stock_id.isin(NAMES)]
        annotated=named.merge(named_outcomes[['event_id','horizon','first_return','first_excess','first_surge','false_start5','concentration']],
            on='event_id',validate='many_to_many')
        coverage=[]
        for actor in ('foreign','trust','dealer'):
            for offset in (-1,0):
                part=timeline[timeline.offset.eq(offset)]
                for scope,sub in [('all',part),('named',part[part.named_case]),('excluding_named',part[~part.named_case])]:
                    coverage.append(dict(actor=actor,offset=offset,scope=scope,events=len(sub),
                        day_known=int(sub[actor+'_net'].notna().sum()),window5_known=int(sub[actor+'_net5'].notna().sum()),
                        window20_known=int(sub[actor+'_net20'].notna().sum())))
        output.mkdir(parents=True);files={}
        for name,table in [('first_timeline',timeline),('named_first_timeline',annotated),('retrospective_events',retro),
            ('retrospective_timeline',retros),('latest_retrospective',latest),('comparisons',comparison),
            ('statistics',pd.DataFrame(stats)),('coverage',pd.DataFrame(coverage))]:
            p=output/(name+'.csv');table.to_csv(p,index=False,encoding='utf-8-sig')
            files[name]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),rows=len(table))
        verify_hashes(expected)
        report=clean(dict(schema='launch_flows_v1',completed=True,live_qualified=False,adopted=False,unseen_validation=False,
            portfolio_returns_computed=False,strategy_net_return=None,historical_first_publication_verified=False,
            targets=NAMES,start='2022-01-03',end='2026-09-09',first_events=len(first),named_first_events=int(first.named_case.sum()),
            retrospective_events=len(retro),institutional_stocks=283,institutional_selection=manifest['selection'],
            unsupported_institutional_days=int((~normalized.schema_supported).sum()),
            statistics=stats,coverage=coverage,causality_checks=checks,artifacts=files,
            source_sha256={str(p.relative_to(ROOT)):h for p,h in expected.items()},
            collection_requests=manifest['new_requests'],research_network_calls=0,database_writes=0,
            elapsed_seconds=round(time.perf_counter()-tick,3),
            limitations=['All history seen; named winners and retrospective anchors are selected using future returns.',
                'The 283-stock institutional cohort comes from prior strategy selection, not the complete market.',
                'Unknown categories and missing days remain unknown; zero net is a known no-net-flow day.',
                'T-1 flows are primary; T flows are contemporaneous and require after-close availability.',
                'Historical first publication and revised values remain unverified.',
                'Returns are gross price proxies without fills, costs or capital allocation.',
                'Same-stock and overlapping outcomes are dependent; comparisons are descriptive, not causal.']))
        write(output/'report.json',report)
        write(output/'manifest.json',dict(schema='launch_flows_manifest_v1',report_sha256=sha(output/'report.json'),
            source_sha256=report['source_sha256'],files=files))
        append_trial_registry(dict(timestamp=datetime.now(timezone.utc).isoformat(),source='launch_flows',status='completed',
            output=str(output.relative_to(ROOT)),preregistration_sha256=sha(ROOT/'docs/prereg_launch_flows_20260927.md'),
            rules=list(RULES),institutional_lags=[1,0],horizons=[20,60],comparison_rows=len(stats),portfolio_returns_computed=False))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=run(p.parse_args().output)
    print({k:r[k] for k in ('first_events','named_first_events','retrospective_events','elapsed_seconds')})
