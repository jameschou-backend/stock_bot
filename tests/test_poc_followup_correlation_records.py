"""Use the real reservation planner to cover its event identity contract."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from skills.poc_followup_correlation_records import CorrelationPlanner, install
from skills.volume_profile_account_adapter import ReservationPlanner


def planner():
    days = pd.date_range('2024-01-01', periods=61)
    a = np.sin(np.arange(61))*.02
    close = pd.DataFrame({'1111':100*np.cumprod(1+a), '2222':50*np.cumprod(1+a),
                          '3333':100*np.cumprod(1-a)}, index=days)
    day = days[-1]+pd.Timedelta(days=1)
    engine = SimpleNamespace(opening_limit=1000000., cash=1000000.,
        opening_members=set(), released_residuals=set(), holdings={}, slots=3,
        previous_nav=1000000., residual_budget=1000000/3, residual_block=False,
        prior=lambda day,sid:100., amount20=pd.DataFrame([[100000000.]*3], index=[day],columns=close.columns),
        exit_signals=SimpleNamespace(adjusted_close=close))
    return ReservationPlanner(engine,day), str(days[-1].date())


def test_real_planner_identity_is_logged_once_and_rejection_spends_nothing():
    source, signal = planner()
    log = []
    wrapped = CorrelationPlanner(source,log)
    def event(sid):return dict(event_id='event-'+sid,members=[sid],signal_date=signal)
    first = wrapped.consider(event('1111'))
    assert first['attempted'] and log[0]['event_id']=='event-1111'
    after = wrapped.snapshot()
    rejected = wrapped.consider(event('2222'))
    assert rejected['failure']=='correlation_above_080'
    assert wrapped.snapshot()==after
    assert wrapped.consider(event('3333'))['attempted']
    assert [r['event_id'] for r in log]==['event-1111','event-2222','event-3333']
    assert wrapped.snapshot()['cash']==333333.34


def test_conflicting_identity_fails_before_reserving_cash():
    source, signal = planner()
    original_probe = source.probe
    source.probe = lambda e: dict(original_probe(e), event_id='another-event')
    log = []
    wrapped = CorrelationPlanner(source, log)
    before = wrapped.snapshot()
    with pytest.raises(ValueError, match='event identity changed'):
        wrapped.consider(dict(event_id='event-1111', members=['1111'], signal_date=signal))
    assert wrapped.snapshot() == before and not log


def test_installer_uses_corrected_planner_and_preserves_profile_normalization():
    from skills import poc_followup, poc_followup_profile_schema
    assert install.__code__ is poc_followup.install.__code__
    assert install.__globals__['CorrelationPlanner'] is CorrelationPlanner
    assert install.__globals__['strength_order'] is poc_followup_profile_schema.strength_order
    assert poc_followup.install.__globals__['CorrelationPlanner'] is not CorrelationPlanner
    source, signal = planner()
    captured = {}
    class Engine:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    runner = SimpleNamespace(engine_types=lambda: (Engine, object), write=lambda p,v:None)
    install(runner, 'correlation60', {})
    def selector(events, *, arm, provider, planner, planner_factory):
        pristine = planner_factory(planner)
        assert isinstance(pristine, CorrelationPlanner)
        return [pristine.consider(e) for e in events]
    stock, _ = runner.engine_types()
    engine = stock(candidate_selector=selector, selection_arm='poc_up_red_available')
    events=[dict(event_id='event-'+sid, members=[sid], signal_date=signal) for sid in ('1111','2222')]
    results=captured['candidate_selector'](events, arm='poc_up_red_available', provider=None, planner=source)
    assert results[0]['attempted'] and results[1]['failure']=='correlation_above_080'
    assert len(engine.followup_log[0]['reservations'])==2


@pytest.mark.parametrize('unknown', [False, True])
def test_real_selector_and_quality_fallback_keep_correlation_gate(unknown):
    from skills.volume_profile_selection import select_candidates
    source, signal = planner()
    captured = {}
    class Engine:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    runner=SimpleNamespace(engine_types=lambda:(Engine,object),write=lambda p,v:None)
    install(runner,'correlation60',{})
    stock,_=runner.engine_types()
    engine=stock(candidate_selector=select_candidates,selection_arm='poc_filter_available')
    events=[dict(event_id='event-'+sid,members=[sid],signal_date=signal,priority=3-i)
            for i,sid in enumerate(('1111','2222','3333'))]
    def provider(event):
        if unknown and event['members']==['3333']:
            return dict(available=False,reason='ordinary_tape_conflict',recoverable=False)
        return dict(available=True,poc_up=True)
    result=captured['candidate_selector'](events,arm='poc_filter',provider=provider,planner=source)
    assert [e['members'][0] for e in result['events']]==['1111','3333']
    assert result['certificate']['final_state']['cash']==333333.34
    rejected=result['certificate']['decisions'][1]['probe']
    assert rejected['failure']=='correlation_above_080' and not rejected['attempted']
    assert engine.followup_log[0].get('original_quality_fallback',False) is unknown
