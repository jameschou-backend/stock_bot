"""Keep verified halt zeroes; never turn unknown liquidity into a failed order."""
import math

import pandas as pd

from skills.residual_tick_replay import ResidualTickReplay, ResidualTickBenchmark, TickExecution
from skills.replay_market_feeds import ReplayDataUnavailable


def full_session_halts(identity):
    result=[]
    for row in identity['trading_exclusions']:
        if not row.get('start') or not row.get('end'):
            continue
        if row['kind']=='information_halt':
            source=row.get('source_row',[])
            # Intraday suspensions must not be interpreted as a zero-volume day.
            if len(source)!=7 or source[4] not in ('8:00','08:00') or source[6] not in ('8:00','08:00'):
                continue
        elif row['kind']!='trading_suspension':
            continue
        result.append(row)
    return result


def restore_halt_zeroes(quotes, raw_sources, identity):
    """Restore actual raw zero rows only when an official full-day halt agrees.

    Prices remain zero/non-executable. No forward fill, synthetic bars or
    guessed zero is allowed. The caller verifies the source file hashes.
    """
    result=quotes.copy()
    result['date']=pd.to_datetime(result['date'])
    raw=pd.concat(raw_sources,ignore_index=True).copy()
    raw['date']=pd.to_datetime(raw['date'])
    if raw.duplicated(['date','stock_id']).any() or result.duplicated(['date','stock_id']).any():
        raise ValueError('Duplicate halt-repair price source')
    additions=[];evidence=[]
    for halt in full_session_halts(identity):
        sid=halt['stock_id']
        rows=raw.loc[raw.stock_id.eq(sid) & raw.date.ge(halt['start']) & raw.date.lt(halt['end'])]
        for row in rows.to_dict('records'):
            fields=('open','high','low','close','volume')
            if any(not math.isfinite(float(row[k])) or row[k]!=0 for k in fields):
                raise ReplayDataUnavailable(f'Official halt conflicts with raw nonzero quote: {sid} {row["date"].date()}')
            existing=result.loc[result.stock_id.eq(sid) & result.date.eq(row['date'])]
            if not existing.empty:
                if any(existing.iloc[0][k]!=0 for k in fields):
                    raise ReplayDataUnavailable('Halt quote already has conflicting values')
                continue
            additions.append(row)
            evidence.append(dict(stock_id=sid,date=str(row['date'].date()),kind=halt['kind'],
                                 source_path=halt['source_path'],volume=0.,price_policy='raw_zero_not_executable'))
    if additions:
        result=pd.concat([result,pd.DataFrame(additions)],ignore_index=True).sort_values(['date','stock_id']).reset_index(drop=True)
    return result,evidence


class StrictInputs:
    def __init__(self,*args,liquidity_identity,**kwargs):
        self.full_halts=full_session_halts(liquidity_identity)
        super().__init__(*args,**kwargs)

    def official_halt(self,day,sid):
        text=str(day.date())
        return any(r['stock_id']==sid and r['start']<=text<r['end'] for r in self.full_halts)

    def require_prior_inputs(self,day,sid):
        values={'prior_price':self.prior(day,sid),
                'adv20':float(self.volume20.at[day,sid]),'amount20':float(self.amount20.at[day,sid])}
        previous=self.days[self.positions[day]-1]
        # Preserve the frozen policy: no usable immediately preceding close
        # means no order on the first resumption day. A documented suspension
        # is known unavailability, not an unexplained data hole.
        known_prior_halt=self.official_halt(previous,sid)
        unknown=[k for k,v in values.items() if (v is None or not math.isfinite(v))
                 and not (k=='prior_price' and known_prior_halt)]
        if unknown:
            raise ReplayDataUnavailable(f'Unknown execution inputs: {sid} {day.date()} {",".join(unknown)}')
        return values

    def corporate_day(self,day):
        # Ranking needs every candidate's prior liquidity, including a candidate
        # that a partial/unknown ranking might otherwise quietly discard.
        candidates=['0050'] if self.benchmark else [e['members'][0] for e in self.events.get(day,[])]
        for sid in candidates:
            if not self.official_halt(day,sid):
                self.require_prior_inputs(day,sid)
        return super().corporate_day(day)

    def _plan(self,day,sid,side,eid,signal,qty,budget,opening_cash,failure):
        if self.prior(day,sid) is None and self.official_halt(self.days[self.positions[day]-1],sid):
            qty=0
            failure='known_previous_session_halt_no_reference'
        return super()._plan(day,sid,side,eid,signal,qty,budget,opening_cash,failure)


class StrictExecution(TickExecution):
    def _execute_order(self,day,sid,side,qty,reason,event_id,signal_date=None):
        plan=self.day_plans[(event_id,side)]
        if self.official_halt(day,sid):
            self.orders.append(dict(date=str(day.date()),stock_id=sid,side=side,channel='board',
                event_id=event_id,signal_date=signal_date,reason=reason,requested_qty=plan['planned_qty'],
                filled_qty=0,failure='official_full_session_halt'))
            return 0
        if plan['planned_qty']:
            self.require_prior_inputs(day,sid)
        return super()._execute_order(day,sid,side,qty,reason,event_id,signal_date)


class StrictResidualReplay(StrictInputs,ResidualTickReplay,StrictExecution):
    pass


class StrictResidualBenchmark(StrictInputs,ResidualTickBenchmark,StrictExecution):
    pass
