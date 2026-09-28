#!/usr/bin/env python3
"""Publish two matching HL2 exit-policy accounts and cohort attribution."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_midpoint as study
from scripts.publish_midpoint_risk import event_profits


def compare_accounts(folder):
    cases={arm:study.old.read(folder/(arm+'.json')) for arm in ('control','midpoint_exit')}
    profits={arm:event_profits(case) for arm,case in cases.items()}
    changes=[]
    identities=sorted(set(profits['control'])|set(profits['midpoint_exit']))
    for eid in identities:
        row=dict(event_id=eid,pnl_change=profits['midpoint_exit'].get(eid,0)-profits['control'].get(eid,0))
        for arm,case in cases.items():
            cohorts=[c for c in case['account']['cohorts'] if c['event_id']==eid]
            row[arm]=dict(pnl=profits[arm].get(eid,0),cohort=cohorts[0] if cohorts else None,
                          trades=[t for t in case['account']['trades'] if t['event_id']==eid])
        changes.append(row)
    expected=cases['midpoint_exit']['summary']['profit']-cases['control']['summary']['profit']
    if abs(sum(r['pnl_change'] for r in changes)-expected)>.03:raise ValueError('Profit attribution does not reconcile')
    return dict(profit_change=expected,cohorts=sorted(changes,key=lambda r:-abs(r['pnl_change'])))


def publish(left, right, output):
    left, right, output = [Path(p).resolve() for p in (left, right, output)]
    if left == right or output.exists(): raise ValueError('Distinct runs and new output required')
    reports = [study.old.read(p/'report.json') for p in (left, right)]
    if reports[0] != reports[1]:
        # Only output paths differ between independently rerun reports.
        for arm in ('control', 'midpoint_exit'):
            a,b = [r['cases'][arm] for r in reports]
            if {k:v for k,v in a.items() if k!='path'} != {k:v for k,v in b.items() if k!='path'}:
                raise ValueError('Summary or account hash differs')
        if {k:v for k,v in reports[0].items() if k!='cases'} != {k:v for k,v in reports[1].items() if k!='cases'}:
            raise ValueError('Sources or research settings differ')
    refs = reports[0]['source_sha256']
    if study.old.file_identities([ROOT/p for p in refs],ROOT) != refs:
        raise ValueError('Source changed after run')
    if any(r['preparation'] for r in reports): raise ValueError('Offline runs required')
    cases = {}
    direct_refs = {}
    for arm in ('control','midpoint_exit'):
        paths = [folder/(arm+'.json') for folder in (left,right)]
        a,b = map(study.old.read,paths)
        if a != b or not a['completed']: raise ValueError('Complete matching accounts required')
        if study.old.sha(paths[0]) != reports[0]['cases'][arm]['sha256']:
            raise ValueError('Account changed after run')
        cases[arm] = dict(summary=a['summary'],event_pnl=event_profits(a),
            path=str(paths[0].relative_to(ROOT)),sha256=study.old.sha(paths[0]),
            exit_evidence=a['audit']['exit_evidence'])
        direct_refs.update(study.old.file_identities(paths,ROOT))
    benchmark = ROOT/'.cache/midpoint-since-2025-20260928/final-a/benchmark.json'
    if any(t['side']=='sell' for t in study.old.read(benchmark)['account']['trades']):
        raise ValueError('Benchmark contains sells; needs matching exit assumptions')
    direct_refs.update(study.old.file_identities([benchmark, Path(__file__),
        ROOT/'scripts/publish_midpoint_risk.py',*[p/f for p in (left,right) for f in ('report.json',)]],ROOT))
    result = dict(start=reports[0]['start'],end=reports[0]['end'],cases=cases,
        benchmark=study.old.read(benchmark)['summary'],source_sha256=direct_refs,comparison=compare_accounts(left),
        offline_identical=True,intraday_stop=False,
        live_qualified=False,actual_fill_verified=False,unseen_validation=False)
    study.old.write(output,result)
    output.with_suffix('.sha256').write_text(study.old.sha(output)+'\n')
    print({k:{s:v['summary'][s] for s in ('total_return','max_drawdown','final_nav')} for k,v in cases.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('left');p.add_argument('right');p.add_argument('output')
    a=p.parse_args();publish(a.left,a.right,a.output)
