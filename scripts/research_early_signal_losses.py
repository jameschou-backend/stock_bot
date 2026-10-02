#!/usr/bin/env python3
"""Fixed, close-known entry screens and post-hoc fast-loss diagnostics."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from skills.independent_three_black import ThreeBlackPath, path_issue
from skills.trial_registry import append_trial_registry

FILTERS = {'close_above_open': '訊號收盤高於開盤',
           'close_location_75': '收盤位置至少0.75',
           'ma20_extension_15': '20日均線乖離不超過15%',
           'no_previous_signal_10': '前10市場交易日無同策略訊號'}
PERIODS = {'2019_2022': ('2019-01-01','2022-12-31'),
           '2023_2024': ('2023-01-01','2024-12-31'),
           '2025_2026': ('2025-01-01','2026-10-02')}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def positive(*values):
    return all(np.isfinite(v) and v > 0 for v in values)


def known_signal_features(days, rows, path):
    """One stock; all entry features read at or before each signal index.

    Repeat windows count market sessions, include all original strategy signals,
    and do not reset when the proposed repeat filter rejected an earlier event.
    """
    days = pd.DatetimeIndex(days)
    path.validate()
    if not path.days.equals(days): raise ValueError('Feature calendar differs')
    c = pd.Series(path.close, index=days)
    ma = c.where(c.gt(0)).rolling(20, min_periods=20).mean().to_numpy()
    previous_high = c.shift(1).where(c.shift(1).gt(0)).rolling(60, min_periods=60).max().to_numpy()
    meanvol = pd.Series(path.volume).where(pd.Series(path.volume).gt(0)).shift(1).rolling(20,min_periods=20).mean().to_numpy()
    results, previous_index = {}, None
    for row in sorted(rows, key=lambda r:(r['signal_date'],r['signal_id'])):
        i = int(days.get_loc(pd.Timestamp(row['signal_date'])))
        if previous_index == i: raise ValueError('Duplicate same-stock signal date')
        close, opened, high, low, volume = [a[i] for a in (path.raw_close,path.opened,path.high,path.low,path.volume)]
        raw_known = (positive(close,opened,high,low,volume) and path.eligible[i]
                     and low <= min(close,opened) <= max(close,opened) <= high)
        adjusted_known = positive(path.close[i],path.other[i]) and path.eligible[i]
        day_change = None
        if i and positive(path.close[i-1],path.other[i-1]) and adjusted_known:
            a,b = path.close[i]/path.close[i-1]-1,path.other[i]/path.other[i-1]-1
            if abs(a) <= .2 and abs(b) <= .2 and abs(a-b) <= .005: day_change=float(a)
        gap = i-previous_index if previous_index is not None else None
        extension=float(path.close[i]/ma[i]-1) if adjusted_known and positive(ma[i]) else None
        loc=float((close-low)/(high-low)) if raw_known and high > low else None
        support=float(previous_high[i]) if positive(previous_high[i]) else None
        result=dict(t0_available_at=row['signal_date']+' 收盤資料完成後',
            t0_source_scope=('FinMind延伸，未完成官方交叉核對' if row['signal_date']>'2026-09-09'
                             else '修復後封存歷史訊號資料，仍非全市場與成交認證'),
            t0_signal_open=float(opened) if raw_known else None,
            t0_signal_close=float(close) if raw_known else None,
            t0_close_location=loc, t0_ma20_extension=extension,
            t0_signal_return=day_change,
            t0_return20=float(path.close[i]/path.close[i-20]-1) if i>=20 and adjusted_known and positive(path.close[i-20]) else None,
            t0_volume_ratio=float(volume/meanvol[i]) if raw_known and positive(meanvol[i]) else None,
            t0_breakout_threshold=support,
            t0_breakout_fraction=float(path.close[i]/support-1) if adjusted_known and support is not None else None,
            t0_previous_signal_gap_sessions=gap,
            t0_raw_bar_valid=bool(raw_known),t0_adjusted_close_valid=bool(adjusted_known),
            close_above_open=bool(close>opened) if raw_known else None,
            close_location_75=bool(loc>=.75) if loc is not None else None,
            ma20_extension_15=bool(extension<=.15) if extension is not None else None,
            no_previous_signal_10=bool(gap is None or gap>10))
        results[row['signal_id']] = result
        previous_index=i
    return results


def posthoc_features(row, path, benchmark, support):
    """Never supplied to entry screens; observe complete first 3 entry sessions."""
    out=dict(posthoc_entry_vs_signal_close=None,posthoc_entry_known_at=None,
        posthoc_entry_day_close_vs_assumed_entry=None,posthoc_entry_day_close_known_at=None,
        posthoc_first3_known_at=None,posthoc_first3_issue=None,
        posthoc_first3_below_breakout=None,posthoc_first3_mean_volume_vs_signal=None,
        posthoc_first3_mean_volume_shrunk=None,posthoc_first3_benchmark_return=None,
        posthoc_first3_benchmark_down=None,posthoc_benchmark_issue=None)
    if not row.get('entry_date'):
        out['posthoc_first3_issue']='not_entered';return out
    start=int(path.days.get_loc(pd.Timestamp(row['entry_date']))); signal=start-1;end=start+2
    if path_issue(path,signal,start) is None:
        # Adjusted ratio avoids calling a split/ex-dividend price change a gap.
        entry=(path.high[start]+path.low[start])/2*path.close[start]/path.raw_close[start]
        out.update(posthoc_entry_vs_signal_close=float(entry/path.close[signal]-1),
                   posthoc_entry_day_close_vs_assumed_entry=float(path.close[start]/entry-1),
                   posthoc_entry_day_close_known_at=str(path.days[start].date())+' 收盤資料完成後',
                   posthoc_entry_known_at=str(path.days[start].date())+' 全日行情完成後')
    if end>=len(path.days):out['posthoc_first3_issue']='insufficient_followup';return out
    out['posthoc_first3_known_at']=str(path.days[end].date())+' 收盤資料完成後'
    issue=path_issue(path,signal,end)
    if issue:out['posthoc_first3_issue']=issue;return out
    out['posthoc_first3_below_breakout']=bool((path.close[start:end+1]<support).any()) if support is not None else None
    ratio=float(np.mean(path.volume[start:end+1])/path.volume[signal])
    out.update(posthoc_first3_mean_volume_vs_signal=ratio,posthoc_first3_mean_volume_shrunk=ratio<1)
    # Benchmark comparison is signal close through the third entry-session close.
    b=benchmark.close[signal:end+1];other=benchmark.other[signal:end+1]
    if not all((np.isfinite(z)&(z>0)).all() for z in (b,other)):
        out['posthoc_benchmark_issue']='missing_adjusted_benchmark'
    else:
        ar,br=b[1:]/b[:-1]-1,other[1:]/other[:-1]-1
        if ((abs(ar)>.2)|(abs(br)>.2)|(abs(ar-br)>.005)).any():
            out['posthoc_benchmark_issue']='benchmark_adjustment_conflict'
        else:
            change=float(b[-1]/b[0]-1)
            out.update(posthoc_first3_benchmark_return=change,posthoc_first3_benchmark_down=change<0)
    return out


def closed_rows(rows):
    rows=[r for r in rows if r['status']=='closed']
    if any(not isinstance(r.get('net_return'),(float,int)) or not math.isfinite(r['net_return']) for r in rows):
        raise ValueError('Closed rows require finite returns')
    return rows


def statistics(rows):
    closed=closed_rows(rows);returns=np.array([r['net_return'] for r in closed],float)
    early=[r for r in closed if r['net_return']<0 and r['holding_days']<=5]
    losses=[r for r in closed if r['net_return']<0]
    tail=max(1,int(math.ceil(len(returns)*.05)))
    return dict(total=len(rows),closed=len(closed),win=sum(r['net_return']>0 for r in closed),
        loss=len(losses),flat=sum(r['net_return']==0 for r in closed),
        win_rate=float(np.mean(returns>0)) if len(returns) else None,
        mean_net_return=float(np.mean(returns)) if len(returns) else None,
        median_net_return=float(np.median(returns)) if len(returns) else None,
        early_loss_count=len(early),early_loss_rate=len(early)/len(closed) if closed else None,
        early_loss_share_of_losses=len(early)/len(losses) if losses else None,
        worst5_mean=float(np.mean(np.sort(returns)[:tail])) if len(returns) else None,
        worst5_count=tail if len(returns) else 0,
        mean_holding_days=float(np.mean([r['holding_days'] for r in closed])) if closed else None,
        status_counts=dict(Counter(r['status'] for r in rows)),
        exit_counts=dict(Counter(r.get('exit_reason') for r in closed)))


def evaluate_filter(rows, field=None):
    yes, no, unknown=[],[],[]
    for row in rows:
        value=True if field is None else row.get(field)
        if value is True:yes.append(row)
        elif value is False:no.append(row)
        elif value is None:unknown.append(row)
        else:raise ValueError('Screen must be explicit bool or unknown')
    base=closed_rows(rows);win=[r for r in base if r['net_return']>0];big=[r for r in base if r['net_return']>=.30]
    def retained(target,keep):return sum(r['net_return']>target if target==0 else r['net_return']>=target for r in closed_rows(keep))
    return dict(arm=field or 'baseline',kept=statistics(yes),excluded=statistics(no),unknown=statistics(unknown),
        baseline=statistics(rows),winner_retention=retained(0,yes)/len(win) if win else None,
        return30_retention=retained(.30,yes)/len(big) if big else None,
        excluded_winners=retained(0,no),unknown_winners=retained(0,unknown),
        excluded_return30=retained(.30,no),unknown_return30=retained(.30,unknown),
        original_winners=len(win),original_return30=len(big))


def distributions(rows):
    keys=['t0_signal_return','t0_return20','t0_volume_ratio','t0_breakout_fraction',
          't0_close_location','t0_ma20_extension','t0_previous_signal_gap_sessions',
          *FILTERS,'posthoc_entry_vs_signal_close','posthoc_entry_day_close_vs_assumed_entry','posthoc_first3_below_breakout',
          'posthoc_first3_mean_volume_vs_signal','posthoc_first3_mean_volume_shrunk',
          'posthoc_first3_benchmark_return','posthoc_first3_benchmark_down']
    groups={'early_loss':[r for r in rows if r.get('diagnostic_group')=='early_loss'],
            'other_loss':[r for r in rows if r.get('diagnostic_group')=='other_loss'],
            'profit':[r for r in rows if r.get('diagnostic_group')=='profit'],
            'all_valid_closed':closed_rows(rows)}
    out={}
    for name,group in groups.items():
        facts={}
        for key in keys:
            values=[float(r[key]) for r in group if r.get(key) is not None]
            facts[key]=dict(known=len(values),unknown=len(group)-len(values),
                mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                q25=float(np.quantile(values,.25)) if values else None,
                q75=float(np.quantile(values,.75)) if values else None)
        out[name]=dict(summary=statistics(group),features=facts,
            gross_loss=sum(r['gross_return']<0 for r in group),
            cost_only_loss=sum(r['gross_return']>=0 and r['net_return']<0 for r in group))
    return out


def verified_sources(bundle,research):
    refs={}
    def check(path, expected):
        path=Path(path).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path)!=expected:
            raise ValueError('Missing, changed, or external evidence: '+str(path))
        refs[str(path.relative_to(ROOT))]=expected
    for folder,name in ((bundle,'manifest'),(research,'report')):
        path=folder/(name+'.json');side=folder/(name+'.sha256')
        check(path,side.read_text().strip());check(side,digest(side))
    manifest=json.loads((bundle/'manifest.json').read_text());report=json.loads((research/'report.json').read_text())
    for name,h in manifest['files_sha256'].items():check(bundle/name,h)
    # All direct sealed research/input source refs checked. Ancestor manifests
    # remain bound as files; recursively re-auditing raw official tables is not
    # performed by this feature-only study and is not claimed as new validation.
    for doc in (manifest,report):
        for name,h in doc.get('source_sha256',{}).items():check(ROOT/name,h)
    for name,h in report['output_sha256'].items():check(ROOT/name,h)
    if report['schema']!='all_independent_three_black_v1' or report['live_qualified'] is not False:
        raise ValueError('Unexpected sealed study')
    return refs,manifest,report


def run(bundle, research, output, *, record_trials=False):
    bundle,research,output=[p.resolve() for p in (bundle,research,output)]
    if output.exists() or not output.is_relative_to(ROOT):raise ValueError('Choose a new repository output')
    refs,manifest,prior=verified_sources(bundle,research)
    payload=json.loads((research/'workbook-data.json').read_text());rows=payload['rows']
    events=json.loads((bundle/'signals.json').read_text())['entries']
    if len(rows)!=30188 or {r['signal_id'] for r in rows}!={e['event_id'] for e in events} or len({r['signal_id'] for r in rows})!=len(rows):
        raise ValueError('Require all original 30,188 unique signals')
    ids=sorted({r['stock_id'] for r in rows}|{'0050'})
    frames={n:pd.read_parquet(bundle/(n+'.parquet'),columns=['date',*ids]).set_index('date') for n in ('close-official','close-quality','eligibility')}
    for f in frames.values():f.index=pd.to_datetime(f.index)
    days=frames['close-official'].index
    quotes=pd.read_parquet(bundle/'quotes-unmasked.parquet');quotes=quotes[quotes.stock_id.isin(ids)].copy();quotes.date=pd.to_datetime(quotes.date)
    if quotes.duplicated(['date','stock_id']).any():raise ValueError('Duplicate OHLC')
    fields={n:quotes.pivot(index='date',columns='stock_id',values=n).reindex(index=days,columns=ids) for n in ('close','high','low','volume','open')}
    paths={sid:ThreeBlackPath(days,*[frames[n][sid].to_numpy(bool if n=='eligibility' else float) for n in ('close-official','close-quality','eligibility')],*[fields[n][sid].to_numpy(float) for n in ('close','high','low','volume','open')]) for sid in ids}
    bystock={sid:[] for sid in ids}
    for r in rows:bystock[r['stock_id']].append(r)
    features={}
    for sid,group in bystock.items():
        if group:features.update(known_signal_features(days,group,paths[sid]))
    enriched=[]
    for original in rows:
        r=dict(original);r.update(features[r['signal_id']]);r.update(posthoc_features(r,paths[r['stock_id']],paths['0050'],r['t0_breakout_threshold']))
        group=r['status']
        if r['status']=='closed':
            group=('early_loss' if r['holding_days']<=5 else 'other_loss') if r['net_return']<0 else 'profit' if r['net_return']>0 else 'flat'
        r['diagnostic_group']=group;enriched.append(r)
    scopes={'all':enriched}
    scopes.update({str(y):[r for r in enriched if r['signal_date'].startswith(str(y))] for y in range(2019,2027)})
    scopes.update({name:[r for r in enriched if start<=r['signal_date']<=end] for name,(start,end) in PERIODS.items()})
    results={name:{arm:evaluate_filter(group,None if arm=='baseline' else arm) for arm in ['baseline',*FILTERS]} for name,group in scopes.items()}
    output.mkdir(parents=True)
    pd.DataFrame(enriched).to_parquet(output/'signal-features.parquet',index=False)
    pd.DataFrame([dict(scope=scope,arm=arm,**value['kept'],winner_retention=value['winner_retention'],return30_retention=value['return30_retention'],excluded_winners=value['excluded_winners'],excluded_return30=value['excluded_return30'],unknown_count=value['unknown']['total']) for scope,arms in results.items() for arm,value in arms.items()]).to_parquet(output/'filter-summary.parquet',index=False)
    comparisons={name:distributions(group) for name,group in scopes.items()}
    def dump(name,value):(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    dump('filter-comparisons.json',results);dump('group-diagnostics.json',comparisons)
    prereg=ROOT/'docs/prereg_early_signal_losses_20261002.md'
    for p in (Path(__file__),prereg,ROOT/'skills/independent_three_black.py',ROOT/'skills/independent_signals.py',ROOT/'skills/trial_registry.py'):
        refs[str(p.relative_to(ROOT))]=digest(p)
    trials=[]
    for arm in ['baseline',*FILTERS]:
        record=dict(timestamp=datetime.now(timezone.utc).isoformat(),command=' '.join(sys.argv),source='early_signal_losses',
            params=dict(arm=arm,start='2019-01-01',end='2026-10-02',independent_signal_research=True,cash_account=False,unseen_validation=False),
            sharpe=None,result=results['all'][arm],scope_results={scope:v[arm] for scope,v in results.items()},
            prereg_sha256=refs[str(prereg.relative_to(ROOT))])
        if record_trials:record['registry_row']=append_trial_registry(record)
        trials.append(record)
    dump('trials.json',trials)
    report=dict(schema='early_signal_loss_diagnostics_v1',created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=refs,output_sha256={str(p.relative_to(ROOT)):digest(p) for p in output.iterdir()},
        sample_count=len(enriched),baseline=results['all']['baseline'],all_filters=results['all'],
        groups=comparisons['all'],registry_recorded=record_trials,
        definitions=dict(early_loss='valid closed net_return<0 and exit_index-entry_index<=5',
            worst5='ceil(0.05*N) lowest net returns; N=valid closed in same subset',
            no_previous_signal_10='No original same-stock strategy signal in indices i-10..i-1, irrespective of earlier screen pass',
            first3='Entry day plus next2 market sessions; volume mean divided by signal-day volume; 0050 from signal close to third-session close'),
        limitations=payload['metadata']['limitations']+['Historical reused data, not unseen validation; overlapping same-stock signals are not independent observations.','Only four fixed single-factor screens; no combinations or threshold search.','Signal-time features and post-hoc path features have separate prefixes and observation dates.','Ancestor manifests bound, direct source hashes verified; no new full official-origin audit or publication-time archive.'],
        live_qualified=False,unseen_validation=False,cash_account=False)
    dump('report.json',report);(output/'report.sha256').write_text(digest(output/'report.json')+'\n')
    print(json.dumps(dict(baseline=results['all']['baseline']['kept'],filters={k:v['kept'] for k,v in results['all'].items()}),ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=ROOT/'.cache/all-signals-2019-20261002/inputs')
    p.add_argument('--research',type=Path,default=ROOT/'.cache/all-signals-2019-20261002/research-v1')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.inputs,args.research,args.output,record_trials=True)
