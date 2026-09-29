"""Independent history-prefix audit of first close-confirmed exit instructions."""
import pandas as pd


def audit_close_stops(account, states, data, engine):
    dates = [str(d.date()) for d in data.days]
    quotes = data.quotes.set_index(['date','stock_id'])
    rows = []
    for cohort in account['cohorts']:
        eid, sid = cohort['event_id'], cohort['stock_id']
        entry = dates.index(cohort['entry_date'])
        end = dates.index(cohort['exit_date'] or data.end)
        fills = [t['reference_price'] for t in account['trades'] if t['event_id']==eid and t['side']=='buy']
        base = max(float(quotes.loc[(data.days[entry],sid),'close']),*fills)
        observed = [(entry,base)]
        factors = {}
        first = None
        for i in range(entry+1,end+1):
            j = i-1
            current = data.features.adjusted_close[sid].iloc[j]
            original = data.features.adjusted_close[sid].iloc[entry]
            if pd.notna(current) and pd.notna(original) and current/original-1 <= -.12+1e-12:
                reason = 'loss12'
            elif i-entry >= 63:
                reason = 'time63'
            else:
                reason = None
            if not reason and j > entry:
                actions = data.events.loc[data.events.stock_id.eq(sid)
                    & pd.to_datetime(data.events.event_date).eq(data.days[j])]
                factors[j] = float(actions.iloc[0].ratio) if len(actions) else 1.
                if not engine.official_halt(data.days[j],sid):
                    q = quotes.loc[(data.days[j],sid)]
                    observed.append((j,float(q.high)))
                    # Rebuild on this signal day's price basis instead of
                    # sharing the engine's incremental peak recurrence.
                    values = []
                    for k,price in observed:
                        for d,factor in factors.items():
                            if k < d <= j:
                                price *= factor
                        values.append(price)
                    peak = max(values)
                    if float(q.close)/peak-1 <= -.15+1e-12:
                        reason = 'close_confirmed_peak15'
            if reason:
                first = dict(trigger_reason=reason,signal_date=dates[j],target_date=dates[i])
                break
        actual = states.get(eid)
        if first:
            if not actual or any(actual[k]!=v for k,v in first.items()):
                raise ValueError('First close instruction differs from independent history')
        elif actual and actual['trigger_reason']:
            raise ValueError('Unexpected close instruction')
        for t in account['trades']:
            if t['event_id']==eid and t['side']=='sell':
                if (not first or t['date']<first['target_date'] or t['signal_date']!=first['signal_date']
                        or t['reason']!=first['trigger_reason'] or t['date']<=t['signal_date']):
                    raise ValueError('Close-stop sale is not after its first confirmed signal')
        rows.append(dict(event_id=eid,stock_id=sid,first_exit=first))
    return rows
