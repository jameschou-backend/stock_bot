from copy import deepcopy
import pytest

from skills.corporate_account_audit import audit_corporate_account
from skills.execution_stress import StressReplay, audit_stress
from skills.top3_cost_audit import private_cash_audit
from skills.top3_validation import MODES, cost_function
from test_technical_replay import fixture, Signals, ENTRY


def account(mode):
    cost = cost_function(mode)
    class Account(StressReplay):
        def _costs(self, price, qty, side, sid):
            return cost(price, qty, side, sid)
    days, adjusted, args, options = fixture(entries=[ENTRY])
    value = Account(*args, technical_signals=Signals(adjusted, days), stress_mode='control', **options).run()
    value['settings'].update(commission=MODES[mode][0], slippage=MODES[mode][1])
    return value


@pytest.mark.parametrize('auditor', [audit_corporate_account, audit_stress])
def test_neutral_auditor_exactly_matches_original_and_doubled_fee_is_independent(auditor):
    base = account('baseline')
    assert private_cash_audit(auditor, 'baseline')(base) == auditor(base)
    stress = account('combined')
    with pytest.raises(ValueError, match='fee/cash calculation differs: commission'):
        auditor(stress)
    result = private_cash_audit(auditor, 'combined')(stress)
    assert result['fees_recomputed'] and result['all_daily_nav_reconciled']
    # Using the new copy does not make the original auditor accept a new policy.
    with pytest.raises(ValueError, match='fee/cash'):
        auditor(stress)


@pytest.mark.parametrize('field', ['commission', 'tax', 'slippage', 'cash_change', 'remaining_shares'])
def test_stress_auditor_retains_fee_cash_and_integer_share_checks(field):
    value = deepcopy(account('combined'))
    value['trades'][0][field] += 1
    with pytest.raises(ValueError):
        private_cash_audit(audit_corporate_account, 'combined')(value)
