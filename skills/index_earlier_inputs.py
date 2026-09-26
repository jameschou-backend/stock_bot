"""Earlier-period data adapters; fixed strategy code is reused without edits."""
from copy import deepcopy
from datetime import date
import json
import math
from pathlib import Path
import pandas as pd
from pypdf import PdfReader
from skills.index_exposure_inputs import load as load_current,acquisition,derived_limits,ROOT
from scripts.research_exit_scenarios import read,sha

BASE=ROOT/'.cache/index-earlier-20260927'
START,END='2016-01-04','2021-12-30'
# Transcribed facts from issuer's 2022-01-06 historical distribution table (page 2).
ISSUER_EX_AMOUNTS={'2015-10-26':2.,'2016-07-28':.85,'2017-02-08':1.7,'2017-07-31':.7,
    '2018-01-29':2.2,'2018-07-23':.7,'2019-01-22':2.3,'2019-07-19':.7,
    '2020-01-31':2.9,'2020-07-21':.7,'2021-01-22':3.05,'2021-07-21':.35}

class EarlierBenchmarkFeeds:
    def __init__(self,limits):self.limits=deepcopy(limits)
    def get_limits(self,sid):
        if sid!='0050':raise ValueError('Earlier benchmark only accepts 0050')
        return self.limits
    def get_odd(self,*args):raise ValueError('Earlier replication cannot execute odd lots')

class EarlierBenchmarkCorporate:
    def __init__(self,dividends):self.dividends=deepcopy(dividends)
    def prepare(self,sid):
        if sid!='0050':raise ValueError('Earlier benchmark only accepts 0050')
    def on_date(self,sid,day):
        self.prepare(sid)
        return [deepcopy(x) for x in self.dividends if x['date']==day]
    def reference_price(self,sid,day,prior):
        self.prepare(sid)
        # Announcement time is absent. Planning cannot discount tomorrow's
        # corporate reference; entitlements/payment still book on actual dates.
        return prior

def load():
    current=load_current();refs=dict(current['sources']);folder=BASE/'sources-v1'
    acquisition(folder,refs)
    pdf=BASE/'official/0050-dividend-history-20220106.pdf';meta=pdf.with_suffix('.pdf.json')
    if sha(pdf)!=read(meta)['sha256']:raise ValueError('Issuer dividend history changed')
    text=' '.join(page.extract_text() for page in PdfReader(pdf).pages)
    if '2022' not in text or '2016/7/28' not in text.replace(' ',''):
        raise ValueError('Issuer history content mismatch')
    for p in (pdf,meta):refs[str(p.relative_to(ROOT))]=sha(p)
    def frame(sid,dataset):
        value=read(folder/(sid+'-'+dataset+'.json'))
        data=pd.DataFrame(value['data'])
        if set(data.stock_id)!={sid} or data.duplicated('date').any() or not data.date.is_monotonic_increasing:
            raise ValueError('Duplicate/unordered dates or wrong instrument')
        return data.set_index('date')
    raw={sid:frame(sid,'TaiwanStockPrice') for sid in ('0050','00631L')}
    limits={sid:frame(sid,'TaiwanStockPriceLimit') for sid in raw}
    if list(raw['0050'].index)!=list(raw['00631L'].index) or any(list(raw[s].index)!=list(limits[s].index) for s in raw):
        raise ValueError('Market calendar or limit date coverage differs')
    calendar=list(raw['0050'].index);days=[d for d in calendar if START<=d<=END]
    if days[0]!=START or days[-1]!=END or sum(d<START for d in calendar)<250:
        raise ValueError('Endpoint or warmup coverage differs')
    normalized={};bounds={}
    for sid in raw:
        if sid=='00631L' and not (limits[sid][['limit_up','limit_down']]==0).all().all():
            raise ValueError('ETF missing-limit provenance changed')
        result={}
        for day,row in raw[sid].iterrows():
            q=dict(open=float(row['open']),close=float(row['close']),low=float(row['min']),
                high=float(row['max']),volume=float(row['Trading_Volume']))
            if (not all(math.isfinite(x) and x>0 for x in q.values())
                    or not q['low']<=min(q['open'],q['close'])<=max(q['open'],q['close'])<=q['high']):
                raise ValueError('Bad early ETF OHLCV')
            if sid=='0050':
                b=dict(lower=float(limits[sid].at[day,'limit_down']),upper=float(limits[sid].at[day,'limit_up']))
                if not 0<b['lower']<=q['low']<=q['high']<=b['upper']:raise ValueError('Observed 0050 bounds disagree')
                bounds[day]=b
            elif day>=START:
                b=derived_limits(limits[sid].at[day,'reference_price'])
                if not b['lower']<=q['low']<=q['high']<=b['upper']:raise ValueError('Derived ETF limits disagree')
                q.update(b)
            result[day]=q
        normalized[sid]=result
    dividends=[]
    for row in frame('0050','TaiwanStockDividend').reset_index().to_dict('records'):
        ex=row['CashExDividendTradingDate'];payment=row['CashDividendPaymentDate']
        if ex<calendar[0] or ex>END:continue
        amount=float(row['CashEarningsDistribution'])+float(row['CashStatutorySurplus'])
        if (not payment or date.fromisoformat(payment)<date.fromisoformat(ex)
                or ISSUER_EX_AMOUNTS.get(ex)!=amount or row['StockEarningsDistribution'] or row['StockStatutorySurplus']):
            raise ValueError('Incomplete or issuer-mismatched distribution')
        dividends.append(dict(action_id='0050-cash-'+ex,stock_id='0050',date=ex,kind='cash_dividend',
            cash_per_share=amount,pay_date=payment,announcement_date=None,
            source='FinMind payment date; issuer ex-date and amount independently verified'))
    if {r['date']:r['cash_per_share'] for r in dividends}!=ISSUER_EX_AMOUNTS:
        raise ValueError('Incomplete distribution set')
    for sid in raw:
        for previous,day in zip(calendar,calendar[1:]):
            diff=raw[sid].at[previous,'close']-limits[sid].at[day,'reference_price']
            expected=ISSUER_EX_AMOUNTS.get(day,0) if sid=='0050' else 0
            if abs(diff-expected)>1e-8:raise ValueError('Unresolved reference/corporate action '+sid+' '+day)
    old=pd.read_parquet(ROOT/'.cache/million-replay-inputs/quotes.parquet')
    old=old[old.stock_id=='0050'].copy();old['date']=pd.to_datetime(old.date).dt.strftime('%Y-%m-%d')
    old=old.set_index('date')
    old_maps={'0050':{d:{k:float(r[k]) for k in ('open','close','low','high','volume')} for d,r in old.iterrows()},
              '00631L':current['quotes']}
    for sid in raw:
        overlap=sorted(set(normalized[sid])&set(old_maps[sid]))
        if len(overlap)!=244 or any(normalized[sid][d][k]!=old_maps[sid][d][k] for d in overlap for k in ('open','close','low','high','volume')):
            raise ValueError('2021 raw-source overlap differs')
    adjusted=frame('0050','TaiwanStockPriceAdj')['close']
    if list(adjusted.index)!=calendar or not all(math.isfinite(v) and v>0 for v in adjusted):
        raise ValueError('Adjusted signal history incomplete')
    overlap=[d for d in adjusted.index if d in current['signals']]
    ratios=pd.Series([current['signals'][d]/adjusted.at[d] for d in overlap]);scale=float(ratios.median())
    if max(abs(ratios/scale-1))>1e-6:raise ValueError('Adjusted overlap basis mismatch')
    signals={d:float(v)*scale for d,v in adjusted.items()}
    benchmark_quotes=pd.DataFrame([dict(date=d,stock_id='0050',**q) for d,q in normalized['0050'].items()])
    return dict(days=days,calendar=calendar,quotes=normalized['00631L'],signals=signals,sources=refs,
        benchmark_quotes=benchmark_quotes,benchmark_limits=bounds,dividends=dividends,
        quality=dict(start=START,end=END,warmup_observations=sum(d<START for d in calendar),
            raw_overlap_per_asset=244,dividend_amounts_and_ex_dates_issuer_matched=True,
            dividend_payment_source='FinMind; not independently issuer-reconciled',
            announcement_times_missing=True,planning_uses_undiscounted_prior_close=True,
            official_00631L_limits=False,strict_data_ready=False))
