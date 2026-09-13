import numpy as np
import pandas as pd
import pytest

from skills.cash_risk_replay import risk_schedule
from skills.observed_risk import observed_risk_schedule, ObservedRiskReplay
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, ENTRY, SIZE
from skills.execution_stress import audit_stress


@pytest.mark.parametrize('mode', ['none', 'trend60', 'shock'])
def test_no_gap_has_identical_policy(mode):
    close = pd.Series(100 + np.sin(np.arange(180)/8)*15, index=pd.bdate_range('2020', periods=180))
    pd.testing.assert_frame_equal(observed_risk_schedule(close, mode), risk_schedule(close, mode))


@pytest.mark.parametrize('mode', ['trend60', 'shock'])
def test_missing_closes_block_without_sixty_day_aftereffect(mode):
    close = pd.Series(100 + np.arange(150)*.1, index=pd.bdate_range('2020', periods=150))
    close.iloc[90:95] = np.nan
    actual = observed_risk_schedule(close, mode)
    assert actual.iloc[90:95]['cap'].isna().all()
    assert actual.iloc[95]['cap'] == 1
    assert pd.isna(risk_schedule(close, mode).iloc[95]['cap'])
    mutated = close.copy(); mutated.iloc[105:] *= .1
    pd.testing.assert_frame_equal(actual.iloc[:105], observed_risk_schedule(mutated, mode).iloc[:105])
    pd.testing.assert_frame_equal(actual.iloc[:105], observed_risk_schedule(close.iloc[:105], mode))


def test_gap_resets_recovery_and_never_fills_missing_price():
    close = pd.Series(np.full(100, 100.), index=pd.bdate_range('2020', periods=100))
    close.iloc[70] = 80; close.iloc[74] = np.nan
    actual = observed_risk_schedule(close, 'trend60')
    assert actual.iloc[73]['recovery_sessions'] == 3
    assert pd.isna(actual.iloc[74]['cap'])
    assert actual.iloc[78]['cap'] == .5
    assert actual.iloc[79]['cap'] == 1
    assert pd.isna(close.iloc[74])


def test_all_missing_is_unknown():
    close = pd.Series(np.full(100, np.nan), index=pd.bdate_range('2020', periods=100))
    assert observed_risk_schedule(close, 'shock')['cap'].isna().all()


def test_full_account_never_trades_on_unavailable_signal():
    market = 100 + np.arange(SIZE)*.1; market[ENTRY-1] = np.nan
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], market=market)
    engine = ObservedRiskReplay(*args, exit_signals=ExitSignals(adjusted, days), risk_mode='shock', **kwargs)
    account = engine.run()
    audit_stress(account)
    assert not account['trades']
    assert engine.risk_decisions[0]['allowed_qty'] == 0
