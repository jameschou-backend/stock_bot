from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from skills.theme_catalyst import COHORT, build_signals, evidence_state, validate_ledger


def event(day='2025-04-15', **changes):
    e=dict(event_id='first', stock_id='2313', theme='leo', source_date=day, observed_at='2026-09-27',
        signal_eligible=True, source_url='https://issuer.example/report.pdf', facts=['actual LEO revenue'],
        material_exposure=True, realized_growth=None, order_or_production=None, negative=None)
    return dict(e, **changes)


def prices():
    days=pd.bdate_range('2025-01-01','2025-12-31')
    columns=['0050', *COHORT, '3491']
    close=pd.DataFrame({s:100*np.exp(np.arange(len(days))*(.0001 if s=='0050' else .003)) for s in columns},index=days)
    volume=close*0+1_000_000
    return close,close.copy(),volume


def test_document_available_after_date_not_same_day_and_strict_clock():
    days=prices()[0].index
    state,refs=evidence_state([event()],days)
    assert state.loc['2025-04-15','2313'] is None
    assert state.loc['2025-04-16','2313'] is True
    delayed,_=evidence_state([event()],days,delay=5)
    assert delayed.loc['2025-04-22','2313'] is None
    assert delayed.loc['2025-04-23','2313'] is True
    observed,_=evidence_state([event()],days,mode='observed')
    assert observed.isna().all().all()


def test_unknown_supersedes_old_positive_and_adverse_overrides_positive():
    days=prices()[0].index
    unknown=event('2025-05-01',event_id='second',material_exposure=None)
    adverse=event('2025-06-01',event_id='third',negative=True)
    state,_=evidence_state([adverse,unknown,event()],days)
    assert state.loc['2025-04-30','2313'] is True
    assert state.loc['2025-05-02','2313'] is None
    assert state.loc['2025-06-02','2313'] is False


def test_later_revision_excluded_and_evidence_expires():
    days=prices()[0].index
    revised=event('2025-05-01',event_id='updated',signal_eligible=False,negative=True)
    state,_=evidence_state([event(),revised],days)
    first=days.get_loc('2025-04-16')
    assert state.iloc[first+125]['2313'] is True
    assert state.iloc[first+126]['2313'] is None


def test_old_late_backfill_cannot_override_newer_disclosure():
    days=pd.bdate_range('2025-01-01','2026-01-30')
    new=event('2025-10-30',event_id='new',negative=True,observed_at='2025-10-30')
    old=event('2025-07-31',event_id='old',observed_at='2025-11-10')
    state,_=evidence_state([new,old],days,mode='observed')
    assert state.loc['2025-11-11','2313'] is False


def test_short_calendar_and_delayed_collection_cannot_rejuvenate_old_evidence():
    with pytest.raises(ValueError,match='Calendar must cover'):
        evidence_state([event()],pd.bdate_range('2026-01-01','2026-06-01'))
    days=pd.bdate_range('2025-01-01','2026-06-01')
    state,_=evidence_state([event(observed_at='2026-01-01')],days,mode='observed')
    assert state.loc['2026-01-02','2313'] is None


def test_causal_signal_subset_shared_cooldown_and_no_etfs():
    frames=prices()
    result=build_signals(*frames,[event()])
    all_rows=result['entries_by_arm']['watchlist_breakout']
    confirmed=result['entries_by_arm']['confirmed_catalyst']
    assert confirmed and {e['event_id'] for e in confirmed}<={e['event_id'] for e in all_rows}
    assert set(e['stock_id'] for e in all_rows)==set(COHORT)
    assert set(e['stock_id'] for e in confirmed)=={'2313'}
    idx={str(d.date()):i for i,d in enumerate(frames[0].index)}
    for sid in COHORT:
        seq=[e for e in all_rows if e['stock_id']==sid]
        assert all(idx[e['entry_date']]==idx[e['signal_date']]+1 for e in seq)
        assert all(idx[b['signal_date']]-idx[a['signal_date']]>=20 for a,b in zip(seq,seq[1:]))
    assert not any(build_signals(*frames,[event()],mode='observed')['entries_by_arm'].values())


def test_future_prices_docs_and_missing_volume_cannot_change_earlier_signals():
    frames=prices();events=[event()]
    original=build_signals(*frames,events)
    changed=[f.copy() for f in frames]
    for f in changed:f.loc['2025-07-01':]*=3
    events.append(event('2025-07-01',event_id='future',negative=True))
    future=build_signals(*changed,events)
    past=lambda r:[e for e in r['decisions'] if e['signal_date']<'2025-07-01']
    assert past(original)==past(future)
    broken=[f.copy() for f in frames];broken[2]['2313']=np.nan
    assert not [e for e in build_signals(*broken,events)['entries_by_arm']['watchlist_breakout'] if e['stock_id']=='2313']


@pytest.mark.parametrize('changes',[{'stock_id':'00631L'},{'material_exposure':'yes'},{'signal_eligible':None},
    {'source_date':'2026-10-01'},{'source_date':'2025-1-1'},{'theme':'ai'}])
def test_invalid_evidence_rejected(changes):
    with pytest.raises(ValueError):validate_ledger([event(**changes)])


def test_both_boundary_years_are_partial_and_middle_year_is_complete():
    from scripts.research_theme_catalyst import annotate_annual_periods
    summary = dict(start='2024-04-16', end='2026-09-09',
        annual=[dict(year=str(y), partial_year=False) for y in (2024, 2025, 2026)])
    annotate_annual_periods(summary)
    first, middle, last = summary['annual']
    assert first['partial_year'] and first['period_start']=='2024-04-16'
    assert not middle['partial_year'] and middle['period_end']=='2025-12-31'
    assert last['partial_year'] and last['period_end']=='2026-09-09'
