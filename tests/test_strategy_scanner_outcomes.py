import numpy as np
import pandas as pd
import pytest

from skills.strategy_scanner.engine import _prepare
from skills.strategy_scanner.outcomes import COSTS, measure_events, study_signals
from tests.test_strategy_scanner_engine import inputs


def fixture():
    bars, days = inputs(80)
    f, days, ids = _prepare(bars, days, days[-1])
    mask = pd.DataFrame(False, index=days, columns=ids)
    mask.at[days[60], '2330'] = True
    return f, days, ids, mask


def test_next_open_and_holding_sessions_costs_and_paired_benchmark():
    f, d, ids, mask = fixture()
    f['open'].at[d[61], '2330'] = 110
    f['close'].at[d[61], '2330'] = f['c'].at[d[61], '2330'] = 112
    f['c'].at[d[65], '2330'] = 120
    r = measure_events(f, d, ids, mask, start=d[60], end=d[60], horizons=(5,)).iloc[0]
    expected = 120*.999*(1-.001425-.003)/(110*1.001*1.001425)-1
    assert r.entry_date == str(d[61].date())
    assert r.exit_date == str(d[65].date())
    assert r.status == 'evaluated'
    assert r.net_return == pytest.approx(expected)
    assert r.gross_return == pytest.approx(120/110-1)
    assert r.excess_vs0050 == pytest.approx(r.net_return-r.benchmark_net_return)
    # Signal-day price is not an execution price.
    f['open'].at[d[60], '2330'] = 1
    other = measure_events(f, d, ids, mask, start=d[60], end=d[60], horizons=(5,)).iloc[0]
    assert other.net_return == r.net_return


def test_missing_interior_path_is_not_silently_ignored_or_forward_filled():
    f, d, ids, mask = fixture()
    f['valid'].at[d[63], '2330'] = False
    r = measure_events(f, d, ids, mask, start=d[60], end=d[60], horizons=(5,)).iloc[0]
    assert r.status == 'stock_path_missing' and np.isnan(r.net_return)
    f['valid'].at[d[63], '2330'] = True
    f['valid'].at[d[63], '0050'] = False
    r = measure_events(f, d, ids, mask, start=d[60], end=d[60], horizons=(5,)).iloc[0]
    assert r.status == 'benchmark_path_missing' and np.isnan(r.excess_vs0050)


def test_ineligible_path_and_immature_events_count_without_early_exit():
    f, d, ids, mask = fixture()
    f['eligible'].at[d[62], '2330'] = False
    mask.at[d[-1], '2330'] = True
    r = measure_events(f, d, ids, mask, start=d[60], end=d[-1], horizons=(5,))
    assert r.status.tolist() == ['stock_path_missing', 'immature']
    assert r.iloc[-1].entry_date is None and r.iloc[-1].exit_date is None
    assert r.net_return.isna().all()
    with pytest.raises(ValueError):
        measure_events(f, d, ids, mask, start=d[60], end=d[-1], horizons=(0,))
    with pytest.raises(ValueError, match='known booleans'):
        measure_events(f, d, ids, mask.astype(float), start=d[60], end=d[-1], horizons=(5,))


def test_zero_volume_cannot_count_as_observed_trading_path():
    f, d, ids, mask = fixture()
    f['volume'].at[d[61], '2330'] = 0
    r = measure_events(f, d, ids, mask, start=d[60], end=d[60], horizons=(5,)).iloc[0]
    assert r.status=='stock_path_missing' and np.isnan(r.net_return)


def test_study_first_events_keeps_unknown_separate_and_rejects_filters():
    bars, days = inputs(100)
    events = [dict(signal_date=str(days[i].date()), stock_id='2330') for i in (70, 71, 99)]
    report, outcomes = study_signals(bars, days, start=days[70], end=days[-1],
        original_signals=events, provenance={'original_candidates_complete':True},
        strategies=['original_red'], horizons=(5,))
    assert report['signal_counts']['original_red']['matching_stock_days']==3
    assert report['signal_counts']['original_red']['first_events']==2
    assert outcomes.signal_date.tolist()==[str(days[70].date()), str(days[99].date())]
    assert report['summary'][0]['events']==2
    assert report['summary'][0]['evaluated']==1
    assert report['summary'][0]['immature']==1
    assert report['cumulative_return'] is None and not report['live_qualified']
    with pytest.raises(ValueError, match='entry strategies'):
        study_signals(bars, days, start=days[70], end=days[-1], strategies=['risk_momentum'])


def test_study_cannot_read_beyond_declared_end():
    bars, days = inputs(100)
    args=dict(start=days[70],end=days[80],strategies=['original_red'],horizons=(20,),
        original_signals=[dict(signal_date=str(days[75].date()),stock_id='2330')],
        provenance={'original_candidates_complete':True})
    report, frame=study_signals(bars,days,**args)
    assert frame.iloc[0].status=='immature'
    future=bars.copy();future.loc[future.date>days[80],'close']=999999
    other, second=study_signals(future,days,**args)
    assert report==other
    pd.testing.assert_frame_equal(frame,second)
