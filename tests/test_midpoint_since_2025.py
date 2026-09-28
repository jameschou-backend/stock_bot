from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.research_midpoint_since_2025 import restart_inputs, exit_evidence
from scripts.export_midpoint_2025_report import verify_cash, cohort_rows
from skills.scenario_exit_replay import ExitSignals


@dataclass
class Inputs:
    days: pd.DatetimeIndex
    entries: list
    end: str
    start: str = '2022-01-03'


def test_restart_keeps_warmup_but_drops_prior_signals():
    days = pd.to_datetime(['2024-12-31', '2025-01-02', '2025-01-03'])
    rows = [dict(signal_date='2024-12-31', entry_date='2025-01-02'),
            dict(signal_date='2025-01-02', entry_date='2025-01-03')]
    data = Inputs(days, rows, '2025-01-03')
    result = restart_inputs(data)
    assert result.entries == rows[1:]
    assert result.days is days
    assert data.entries == rows
    with pytest.raises(ValueError, match='audited session'):
        restart_inputs(data, '2025-01-01')


def test_exit_reconstruction_uses_previous_close_and_latches_reason():
    days = pd.bdate_range('2025-01-02', periods=5)
    close = pd.DataFrame({'1101': [100, 87, 120, 130, 140], '0050': [100]*5}, index=days)
    data = SimpleNamespace(days=days, features=ExitSignals(close, days), end=str(days[-1].date()))
    account = dict(cohorts=[dict(stock_id='1101', event_id='e', entry_date=str(days[0].date()))],
                   trades=[dict(event_id='e', side='sell', reason='loss12', signal_date=str(days[1].date()))])
    evidence = exit_evidence(data, account)['e']
    assert evidence['target_date'] == str(days[2].date())
    assert evidence['signal_adjusted_close'] == 87
    assert evidence['stop_adjusted_close'] == 88
    account['trades'][0]['signal_date'] = str(days[2].date())
    with pytest.raises(ValueError, match='prior closes'):
        exit_evidence(data, account)


def test_cash_audit_rejects_missing_dividend_in_daily_balance():
    result = dict(account=dict(cash_ledger=[dict(date='2025-01-02', cash_change=100, cash_after=100)],
                               daily=[dict(date='2025-01-02', cash=101, market_value=0, receivable=0, nav=101)],
                               trades=[]), summary=dict(cash=100, final_nav=101))
    with pytest.raises(ValueError, match='Daily cash'):
        verify_cash(result)


def test_cohort_attribution_includes_late_dividend_after_sale():
    event = dict(stock_id='1101', name='台泥', event_id='e', signal_date='2025-01-02',
                 leader_evidence=dict(leader_return20=.3, benchmark_return20=.1,
                                      leader_volume_ratio=2, leader_peer_breadth=.2))
    result = dict(account=dict(cohorts=[event], trades=[
        dict(event_id='e', side='buy', date='2025-01-03', qty=10, gross=100, cash_change=-101),
        dict(event_id='e', side='sell', date='2025-03-01', qty=10, gross=120, cash_change=118)],
        corporate_actions=[dict(action_id='d', event_id='e')],
        cash_ledger=[dict(kind='dividend_payment', action_id='d', cash_change=5)]),
        summary=dict(final_holdings=[], final_receivables=[], profit=22),
        exit_evidence={'e': dict(reason='time63', signal_date='2025-02-28', entry_return=.2,
                                 signal_adjusted_close=12, stop_adjusted_close=8.8)})
    row = cohort_rows(result)[0]
    assert row['pnl'] == 22
    assert row['dividends'] == 5
    assert row['status'] == '已結束'
