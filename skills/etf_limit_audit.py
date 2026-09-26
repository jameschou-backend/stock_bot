"""Explicit ETF bound derivation; preserve provider fields and report conflicts."""
from decimal import Decimal,ROUND_CEILING,ROUND_FLOOR
import math

def bounds(reference):
    value=Decimal(str(reference))
    if not value.is_finite() or value<=0:raise ValueError('Invalid ETF reference')
    def bound(factor,rounding):
        amount=value*Decimal(factor);step=Decimal('.01' if amount<50 else '.05')
        return float((amount/step).to_integral_value(rounding=rounding)*step)
    return dict(lower=bound('.9',ROUND_CEILING),upper=bound('1.1',ROUND_FLOOR))

def reconcile(quotes,raw_limits,days):
    derived={};differences=[];violations=[]
    for day in days:
        if day not in quotes:continue
        r=raw_limits[day];q=quotes[day];b=bounds(r['reference_price'])
        if (not all(math.isfinite(v) and v>0 for v in q.values())
                or not b['lower']<=q['low']<=min(q['open'],q['close'])<=max(q['open'],q['close'])<=q['high']<=b['upper']):
            raise ValueError('ETF quote exceeds derived bounds '+day)
        provider=dict(lower=float(r['limit_down']),upper=float(r['limit_up']))
        if b!=provider:differences.append(dict(date=day,reference=r['reference_price'],provider=provider,derived=b))
        if q['low']<provider['lower'] or q['high']>provider['upper']:
            violations.append(dict(date=day,low=q['low'],high=q['high'],provider=provider,derived=b))
        derived[day]=b
    return derived,dict(schema='etf_limit_reconciliation_v1',provider_field_differences=differences,
        provider_ohlc_conflicts=violations,derived_bound_quote_conflicts=0,
        source='derived_10_percent_etf_ticks_not_observed_official',strict_data_ready=False)
