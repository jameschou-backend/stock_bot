"""Explicit, offline-only ETF inputs; reconstructed limits never replace raw data."""
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path
import math
import pandas as pd
from scripts.research_exit_scenarios import read,sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'.cache/index-exposure-20260927'
SPLIT='2026-03-31'
HALT=('2026-03-25','2026-03-26','2026-03-27','2026-03-30')
BENCHMARK=ROOT/'artifacts/forward_simulation/residual_slots_20260926.json'
CLOSE=ROOT/'.cache/historical-selector-replay-20260925/final-v7/combined/close-official.parquet'

def derived_limits(reference):
    value=Decimal(str(reference))
    if not value.is_finite() or value<=0:raise ValueError('Invalid ETF reference price')
    def bound(multiplier,rounding):
        price=value*Decimal(multiplier)
        step=Decimal('.01' if price<50 else '.05')
        return float((price/step).to_integral_value(rounding=rounding)*step)
    return dict(lower=bound('.8',ROUND_CEILING),upper=bound('1.2',ROUND_FLOOR),
                source='derived_domestic_2x_rule_not_observed_official_limits')

def acquisition(folder,refs):
    manifest=read(folder/'manifest.json')
    if sha(folder/'manifest.json')!=(folder/'manifest.sha256').read_text().strip():
        raise ValueError('Acquisition manifest changed')
    refs[str((folder/'manifest.json').relative_to(ROOT))]=sha(folder/'manifest.json')
    refs[str((folder/'manifest.sha256').relative_to(ROOT))]=sha(folder/'manifest.sha256')
    for name,digest in manifest['files_sha256'].items():
        path=folder/name
        if path.parent!=folder or sha(path)!=digest:raise ValueError('Acquisition bytes changed')
        refs[str(path.relative_to(ROOT))]=digest
    for name,digest in read(folder/'plan.json')['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Acquisition source changed')
        refs[name]=digest

def load():
    refs={}
    for name in ('sources-v1','warmup-v1'):acquisition(BASE/name,refs)
    expected={'fund-basic.html.json','etf-rules.html.json','split-final.pdf.json','split-last-trade.pdf.json'}
    if {p.name for p in (BASE/'official').glob('*.json')}!=expected:
        raise ValueError('Primary evidence set missing or changed')
    for meta in sorted((BASE/'official').glob('*.json')):
        record=read(meta);path=meta.with_suffix('')
        if sha(path)!=record['sha256']:raise ValueError('Primary evidence changed')
        refs[str(path.relative_to(ROOT))]=sha(path)
        refs[str(meta.relative_to(ROOT))]=sha(meta)
    def frame(folder,dataset):
        f=pd.DataFrame(read(BASE/folder/(dataset+'.json'))['data'])
        if f.duplicated('date').any() or not f.date.is_monotonic_increasing:
            raise ValueError('Dates must be unique and ordered')
        return f.set_index('date')
    raw=frame('sources-v1','TaiwanStockPrice')
    supplied=frame('sources-v1','TaiwanStockPriceLimit')
    if list(raw.index)!=list(supplied.index) or len(raw)!=1376:
        raise ValueError('ETF quote/limit coverage changed')
    if not (supplied[['limit_up','limit_down']]==0).all().all():
        raise ValueError('Expected explicit missing provider limit fields')
    quotes={}
    for day,row in raw.iterrows():
        values={k:float(row[k]) for k in ('open','close','min','max','Trading_Volume')}
        if (not all(math.isfinite(x) and x>0 for x in values.values())
                or not values['min']<=min(values['open'],values['close'])<=max(values['open'],values['close'])<=values['max']):
            raise ValueError('Invalid ETF OHLCV '+day)
        limits=derived_limits(supplied.at[day,'reference_price'])
        for key in ('open','close','min','max'):
            amount=Decimal(str(values[key]));tick=Decimal('.01' if amount<50 else '.05')
            if amount%tick:raise ValueError('ETF quote violates tick size '+day)
        if not limits['lower']<=values['min']<=values['max']<=limits['upper']:
            raise ValueError('OHLC exceeds derived limits '+day)
        quotes[day]=dict(open=values['open'],close=values['close'],low=values['min'],high=values['max'],
            volume=values['Trading_Volume'],reference=float(supplied.at[day,'reference_price']),**limits)
    for a,b in zip(raw.index,raw.index[1:]):
        expected=20.14 if b==SPLIT else quotes[a]['close']
        if abs(quotes[b]['reference']-expected)>1e-8:raise ValueError('Reference discontinuity '+b)
    if quotes['2026-03-24']['close']!=443.15 or any(d in quotes for d in HALT):
        raise ValueError('Split timeline differs from issuer evidence')
    current=pd.read_parquet(CLOSE,columns=['date','0050']).set_index('date')['0050'].dropna()
    current.index=current.index.strftime('%Y-%m-%d')
    prefix=frame('warmup-v1','TaiwanStockPriceAdj')['close']
    overlap=current.index.intersection(prefix.index)
    ratios=current.loc[overlap]/prefix.loc[overlap]
    scale=float(ratios.median())
    if len(overlap)!=21 or max(abs(ratios/scale-1))>1e-6:
        raise ValueError('Warmup adjusted-price basis does not reconcile')
    signals={d:float(v)*scale for d,v in prefix.items() if d<current.index[0]}
    signals.update({d:float(v) for d,v in current.items()})
    if sum(d<'2022-01-03' for d in signals)<250:raise ValueError('Insufficient signal warmup')
    if sha(BENCHMARK)!=BENCHMARK.with_suffix('.sha256').read_text().strip():
        raise ValueError('Benchmark publication changed')
    publication=read(BENCHMARK);benchmarks={}
    for name,digest in publication['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Sealed benchmark source changed '+name)
        refs[name]=digest
    for mode in ('control','combined'):
        desc=publication['cases']['benchmark_'+mode]['result'];path=ROOT/desc['path']
        if sha(path)!=desc['sha256']:raise ValueError('Benchmark account changed')
        benchmarks[mode]=read(path);refs[desc['path']]=desc['sha256']
    days=[d['date'] for d in benchmarks['control']['account']['daily']]
    if set(days)-set(quotes)!=set(HALT) or set(d for d in quotes if d>=days[0])!=set(days)-set(HALT):
        raise ValueError('Unexpected ETF missing or additional market session')
    calendar=sorted(set(signals)|set(quotes)|set(days))
    for path in (BENCHMARK,BENCHMARK.with_suffix('.sha256'),CLOSE):refs[str(path.relative_to(ROOT))]=sha(path)
    return dict(quotes=quotes,signals=signals,days=days,calendar=calendar,benchmarks=benchmarks,
        sources=refs,quality=dict(warmup_observations=sum(d<days[0] for d in signals),
            overlap_observations=len(overlap),warmup_scale=scale,derived_limit_rows=len(quotes),
            observed_official_limits=False,strict_data_ready=False,known_halt_dates=list(HALT)))
