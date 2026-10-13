from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skills.poc_followup import ConfirmedBlack
from skills.poc_followup_merger_black import ExchangeBlack, independent_black_audit
from skills.three_black_exit import ThreeBlackSignals


def test_converted_holding_uses_only_new_ordinary_candles_and_original_clock():
    days = pd.bdate_range('2026-08-10', '2026-09-30')
    close = pd.DataFrame({'2867': np.linspace(40, 30, len(days)),
                          '2884': np.linspace(100, 60, len(days))}, index=days)
    quotes = pd.DataFrame([dict(date=day, stock_id=sid, open=price+1,
        high=price+2, low=price-1, close=price, volume=100000)
        for sid in close for day, price in close[sid].items()])
    source = ThreeBlackSignals(close, quotes, days)
    account = SimpleNamespace(merger_event_id='e', holdings={'2884': dict(event_id='e')})
    wrapped = ExchangeBlack(ConfirmedBlack(source), account, 4)
    entry = days.get_loc(pd.Timestamp('2026-08-17'))
    rows=[]
    for day in pd.bdate_range('2026-09-01', '2026-09-14'):
        i = days.get_loc(day)
        trigger = wrapped.exits(i, entry, '2884')
        assert trigger == (str(day.date()) == '2026-09-14')
        rows.append(dict(event_id='e', stock_id='2884', date=str(day.date()),
                         signal_date=str(days[i-1].date()), trigger=trigger))
    journal = dict(cohorts=[dict(event_id='e', stock_id='2867', current_stock_id='2884',
        share_exchange_date='2026-09-01', entry_date='2026-08-17')], black_log=rows, trades=[])
    result = independent_black_audit(4)(journal, source)
    assert result['exits'] == 1
    assert [b['date'] for b in result['triggers'][0]['bars']] == ['2026-09-08','2026-09-09','2026-09-10','2026-09-11']
    bad=deepcopy(journal);bad['black_log'][0]['trigger']=True
    with pytest.raises(ValueError, match='historical candles'):
        independent_black_audit(4)(bad, source)
    account.holdings['2884']['event_id']='separate-original-target-position'
    assert wrapped.exits(days.get_loc(pd.Timestamp('2026-09-01')), entry, '2884')
