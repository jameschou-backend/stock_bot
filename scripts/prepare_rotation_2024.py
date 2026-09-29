#!/usr/bin/env python3
"""Build versioned 2024 relaxed and causally revalidated candidate ledgers."""
from pathlib import Path
from copy import deepcopy
import sys
import argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from skills.diffusion_signals import LOOKBACK,MIN_COMMON,MIN_TURNOVER,_returns,_rolling
from skills.regime_state import build_trend
BASE=ROOT/'.cache/partial-risk-2019-20260929/inputs-final'


def generate(frames,companies,signals,cutoff):
    close, other, raw, volume = [frames[name] for name in
        ('close-official', 'close-quality', 'raw-close', 'raw-volume')]
    companies = companies.set_index('stock_id')
    names = companies['name']
    eligibility = frames['eligibility'].reindex(columns=close.columns)
    days, ids = close.index, list(close.columns)
    listing = np.ones(close.shape, dtype=bool)
    for j, sid in enumerate(ids):
        if sid != '0050':
            listing[:, j] = days >= companies.at[sid, 'listed_date']
    listing &= eligibility.to_numpy()
    price, other_price, vol, amount = [f.to_numpy(dtype=float, copy=True)
        for f in (close, other, volume, raw*volume)]
    for values in (price, other_price, vol, amount):
        values[(values <= 0) | ~listing] = np.nan
    amount[~np.isfinite(vol)] = np.nan
    benchmark = ids.index('0050')
    ret, ret_other = _returns(price), _returns(other_price)
    common = np.isfinite(ret[:, benchmark]) & np.isfinite(ret_other[:, benchmark])
    count = _rolling(common[:, None].astype(float), LOOKBACK, 'sum')[:, 0]
    incomplete = _rolling((common[:, None] & ~(np.isfinite(ret) & np.isfinite(ret_other))).astype(float), LOOKBACK, 'sum') > 0
    anomaly = _rolling(((abs(ret) > .2) | (abs(ret_other) > .2) | (abs(ret-ret_other) > .005)).astype(float), LOOKBACK, 'sum') > 0
    mature = np.zeros(close.shape, dtype=bool)
    for j, sid in enumerate(ids):
        mature[LOOKBACK:, j] = True if sid == '0050' else days[:-LOOKBACK] >= companies.at[sid, 'listed_date']
    enough = (np.arange(len(days)) >= LOOKBACK) & (count >= MIN_COMMON)
    liquid = _rolling(amount, 20)
    quality = enough[:, None] & mature & ~incomplete & ~anomaly & np.isfinite(liquid) & (liquid >= MIN_TURNOVER)
    quality &= (enough & ~anomaly[:, benchmark])[:, None] & listing
    r5, r20 = _returns(price, 5), _returns(price, 20)
    high, meanvol = np.full_like(price, np.nan), np.full_like(price, np.nan)
    high[1:] = _rolling(price, 60, 'max')[:-1]
    meanvol[1:] = _rolling(vol, 20)[:-1]
    technical = quality & (price > high) & (r20 > 0) & (r20 > r20[:, [benchmark]]) & (vol >= meanvol*1.5)
    technical[:, benchmark] = False

    trend=build_trend(close['0050'])
    ma20=close.rolling(20,min_periods=20).mean()
    monthly={g['month']:g for g in signals['diffusion']['groups']}
    original=[deepcopy(e) for e in signals['entries'] if e['signal_date']<=cutoff]
    relaxed=[e for e in original if e['entry_date']<'2024-01-01']
    for i in np.flatnonzero((days.year==2024)&(days<=cutoff)):
        if i+1>=len(days) or days[i+1].year!=2024 or trend.state.iloc[i]!='ON':continue
        month=monthly[str(days[i].to_period('M'))]
        pool=set(month['selected_ids'])-set(month['exclusions'].get('zero_residual_variance',[]))
        for j in np.flatnonzero(technical[i]):
            sid=ids[j]
            if sid not in pool:continue
            day=str(days[i].date())
            relaxed.append(dict(event_id=f'relaxed-{day}-{sid}',signal_date=day,
                entry_date=str(days[i+1].date()),members=[sid],priority=float(r20[i,j]-r20[i,benchmark]),
                group_id='relaxed-'+str(days[i].to_period('M')),group_cutoff_date=month['cutoff_date'],
                group_members=[sid],selection_reason='same technical and monthly top300; no cluster restrictions',
                leader_evidence=dict(leader_return20=float(r20[i,j]),benchmark_return20=float(r20[i,benchmark]),
                    leader_volume_ratio=float(vol[i,j]/meanvol[i,j]))))
    def queued(entries):
        result={(e['entry_date'],e['members'][0]):deepcopy(e) for e in entries if e['entry_date']<'2024-01-01'}
        for e in sorted(entries,key=lambda e:(e['signal_date'],e['event_id'])):
            if e['entry_date']<'2024-01-01':continue
            original_i=days.get_loc(e['signal_date']);sid=e['members'][0];j=ids.index(sid)
            for offset in range(5):
                i=original_i+offset
                if i+1>=len(days) or days[i]>pd.Timestamp(cutoff) or days[i+1].year!=2024:break
                if offset and not (quality[i,j] and trend.state.iloc[i]=='ON' and price[i,j]>ma20.iloc[i][sid]
                    and r20[i,j]>r20[i,benchmark] and price[i,j]>=price[original_i,j]*.95):continue
                item=deepcopy(e)
                if offset:
                    item.update(event_id=e['event_id']+'-renew-'+str(days[i].date()),
                        origin_event_id=e['event_id'],origin_signal_date=e['signal_date'],
                        signal_date=str(days[i].date()),entry_date=str(days[i+1].date()),
                        priority=float(r20[i,j]-r20[i,benchmark]),selection_reason='five-session renewed candidate')
                result[(item['entry_date'],sid)]=item
        return sorted(result.values(),key=lambda e:(e['entry_date'],-e['priority'],e['event_id']))
    # Original entries never retimed; original prefix is shared by every arm.
    return dict(original=original,relaxed=relaxed,stagnant=queued(original),
                stronger=queued(original),combined=queued(relaxed))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cutoff',default='2024-12-30');a=parser.parse_args()
    if a.output.exists():raise ValueError('Preserve prior signal version')
    manifest=read(BASE/'manifest.json')
    for name,h in manifest['files_sha256'].items():
        if sha(BASE/name)!=h:raise ValueError('Frozen input changed '+name)
    frames={n:pd.read_parquet(BASE/(n+'.parquet')).set_index('date') for n in
        ('close-official','close-quality','raw-close','raw-volume','eligibility')}
    companies=pd.read_parquet(BASE/'companies.parquet');signals=read(BASE/'signals.json')
    entries=generate(frames,companies,signals,a.cutoff)
    last=frames['close-official'].index[frames['close-official'].index>pd.Timestamp('2024-06-28')][0]
    prefix=generate({k:v.loc[:last].copy() for k,v in frames.items()},companies,signals,'2024-06-28')
    for arm in entries:
        if [e for e in entries[arm] if e['signal_date']<='2024-06-28']!=prefix[arm]:
            raise ValueError('Future truncation changed candidates '+arm)
    refs=manifest['source_sha256']|{str((BASE/n).relative_to(ROOT)):h for n,h in manifest['files_sha256'].items()}
    for p in [BASE/'manifest.json',Path(__file__),ROOT/'skills/rotation_2024.py',ROOT/'docs/prereg_rotation_2024_20260929.md']:
        refs[str(p.relative_to(ROOT))]=sha(p)
    write(a.output,dict(entries=entries,prefix_identical=True,source_sha256=refs,unseen_validation=False))
    print({arm:sum(e['entry_date']>='2024-01-01' for e in values) for arm,values in entries.items()},flush=True)
