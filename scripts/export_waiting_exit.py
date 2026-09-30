#!/usr/bin/env python3
"""Compare every registered waiting exit after two exact complete account replays."""
from pathlib import Path
import argparse,gzip,json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from scripts.export_midpoint_2025_report import verify_cash
from scripts.export_stock_universe_2019 import period_stats
from scripts.research_surge_capture import AccountIndex
from skills.waiting_exit import ARMS


def main():
    p=argparse.ArgumentParser()
    for key in ('left','right','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.relative_to(ROOT)
    if a.left.resolve()==a.right.resolve() or a.output.exists() or a.output.with_suffix('.json').exists():
        raise ValueError('Two independent runs and new output required')
    reports=[read(d/'report.json') for d in (a.left,a.right)]
    for report in reports:
        if not report['validated'] or not report['all_completed'] or report['preparation'] or set(report['cases'])!=set(ARMS):
            raise ValueError('Incomplete fixed waiting suite')
        if (report['start'],report['end'],report['initial_cash'])!=('2019-01-02','2026-09-09',1_000_000):
            raise ValueError('Account period or initial capital differs')
    if reports[0]['source_sha256']!=reports[1]['source_sha256']:raise ValueError('Source sets differ')
    for name,h in reports[0]['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Source changed '+name)
    cases={}
    for arm in ARMS:
        values=[]
        for d,report in zip((a.left,a.right),reports):
            f=d/(arm+'.json')
            if sha(f)!=report['cases'][arm]['sha256']:raise ValueError('Case changed '+arm)
            values.append(read(f))
        if values[0]!=values[1] or not values[0]['completed']:raise ValueError('Independent accounts differ '+arm)
        verify_cash(values[0]);cases[arm]=values[0]
    base=ROOT/'.cache/stock-universe-2019-20260929/final-a'
    if cases['control3']['account']!=read(base/'liquid_universe.json')['account']:raise ValueError('Control does not reproduce original')
    bench=base/'benchmark.json'
    if sha(bench)!=read(base/'report.json')['cases']['benchmark']['sha256']:raise ValueError('Benchmark changed')
    cases['benchmark']=read(bench)
    signal_path=ROOT/'.cache/stock-universe-2019-20260929/signals-v2.json';entries=read(signal_path)['entries']['liquid_universe']
    close=pd.read_parquet(ROOT/'.cache/partial-risk-2019-20260929/inputs-final/close-official.parquet').set_index('date')
    summary=[];annual=[];periods=[];attribution={};retained={};waiting={}
    a.output.mkdir(parents=True)
    for arm,case in cases.items():
        s=case['summary'];account=case['account'];verify_cash(case)
        durations=[int(close.index.get_loc(pd.Timestamp(c['exit_date']))-
                       close.index.get_loc(pd.Timestamp(c['entry_date']))+1)
                   for c in account['cohorts'] if c.get('exit_date')]
        if len(account['daily'])!=1867:raise ValueError('Incomplete calendar')
        if arm!='benchmark':
            settings=dict(account['settings'])
            mode=settings.pop('waiting_mode','control')
            if mode!=arm[:-1]:raise ValueError('Published arm does not match waiting policy')
            if settings!=cases['control3']['account']['settings']:raise ValueError('Unexpected execution setting change')
            ids={e['event_id'] for e in entries}
            if any(t['event_id'] not in ids for t in account['trades'] if t['side']=='buy'):raise ValueError('Buy bypassed filter')
            idx=AccountIndex(case,close)
            attribution[arm]=sorted([dict(stock_id=sid,net_profit=value) for sid,value in idx.profit.items()],key=lambda v:-v['net_profit'])
            base_ids={c['event_id'] for c in cases['control3']['account']['cohorts']};actual={c['event_id'] for c in account['cohorts']}
            retained[arm]=dict(common_entries=len(base_ids&actual),new_entries=len(actual-base_ids),old_entries_absent=len(base_ids-actual))
            triggered=[r for r in account.get('waiting_decisions',[]) if r['trigger']]
            completed={c['event_id']:c for c in account['cohorts'] if c.get('exit_date')}
            returns=[]
            for row in triggered:
                co=completed.get(row['event_id'])
                later=[c for c in account['cohorts'] if c['stock_id']==row['stock_id'] and co and c['entry_date']>co['exit_date']]
                next_date=min((c['entry_date'] for c in later),default=None)
                delay=int(close.index.get_loc(pd.Timestamp(next_date))-close.index.get_loc(pd.Timestamp(co['exit_date']))) if next_date else None
                returns.append(dict(stock_id=row['stock_id'],event_id=row['event_id'],signal_date=row['signal_date'],
                    first_sell_attempt=row['date'],exit_date=co['exit_date'] if co else None,
                    observed_closes=row['observed_closes'],reentered=bool(later),
                    next_entry_date=next_date,next_entry_delay_sessions=delay,reentered_within20=delay is not None and delay<=20))
            waiting[arm]=dict(triggered=len(triggered),fully_exited=sum(r['exit_date'] is not None for r in returns),
                later_reentered_same_stock=sum(r['reentered'] for r in returns),reentered_within20=sum(r['reentered_within20'] for r in returns),exits=returns)
        summary.append(dict(arm=arm,**{k:s[k] for k in ('total_return','max_drawdown','final_nav','profit','trade_count','stock_cohorts','costs')},
            closed_cohorts=len(durations),median_closed_holding_sessions=float(pd.Series(durations).median()) if durations else None,
            max_closed_holding_sessions=max(durations) if durations else None))
        annual.extend(dict(arm=arm,**r) for r in s['annual'])
        for start,end in [('2019-01-02','2021-12-31'),('2022-01-01','2024-12-31'),('2025-01-01','2026-09-09')]:
            periods.append(dict(arm=arm,**period_stats(account['daily'],start,end)))
        if arm!='benchmark':
            for name in ('daily','trades','orders','holdings','cash_ledger','cohorts','waiting_decisions'):
                data=pd.DataFrame(account.get(name,[])).to_csv(index=False,float_format='%.12g').encode('utf-8-sig')
                if name=='orders':(a.output/(arm+'-'+name+'.csv.gz')).write_bytes(gzip.compress(data,mtime=0))
                else:(a.output/(arm+'-'+name+'.csv')).write_bytes(data)
    for name,rows in [('comparison',summary),('annual',annual),('periods',periods)]:
        pd.DataFrame(rows).to_csv(a.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    write(a.output/'attribution.json',attribution)
    write(a.output/'waiting-exits.json',waiting)
    diagnosis=ROOT/'.cache/waiting-exit-20260930/diagnosis-v3/report.json'
    diagnosis_data=read(diagnosis)
    for name,digest in diagnosis_data['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Diagnosis source changed '+name)
    (a.output/'launch-diagnosis.json').write_bytes(diagnosis.read_bytes())
    pd.DataFrame(diagnosis_data['after_decision_statistics']).to_csv(
        a.output/'after-decision-statistics.csv',index=False,encoding='utf-8-sig')
    for name in ('signals.csv.gz','cohorts.csv.gz'):
        (a.output/('launch-'+name)).write_bytes((diagnosis.parent/name).read_bytes())
    terms=ROOT/'docs/waiting_exit_corporate_terms.json'
    (a.output/'corporate-terms.json').write_bytes(terms.read_bytes())
    for name,digest in read(terms)['source_sha256'].items():
        f=ROOT/name
        if sha(f)!=digest:raise ValueError('Corporate source changed '+name)
        (a.output/('corporate-'+f.name)).write_bytes(f.read_bytes())
    registry=ROOT/'artifacts/experiments/trial_registry.jsonl'
    trials=[json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    trials=[r for r in trials if r.get('source')=='waiting_exit_20260930']
    (a.output/'trials.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in trials))
    write(a.output.with_suffix('.json'),dict(schema='waiting_exit_20260930',comparison=summary,annual=annual,periods=periods,
        retained_entries=retained,offline_identical=True,source_reports=[dict(path=str((d/'report.json').resolve().relative_to(ROOT)),sha256=sha(d/'report.json')) for d in (a.left,a.right)],
        exports_sha256={str(f.relative_to(ROOT)):sha(f) for f in a.output.iterdir()},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),bench,signal_path,diagnosis)},
        live_qualified=False,unseen_validation=False,actual_fill_verified=False,complete_historical_universe=False,
        price_basis='channel_daily_high_low_midpoint_proxy'))
    for row in summary:print(row)


if __name__=='__main__':main()
