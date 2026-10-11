from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.all_entry_capital_completion import (
    KEY, TERMS, CapitalCompletionRoute, load_capital_completion,
)
from skills.corporate_account_audit import audit_capital_cash
from skills.face_value_capital import FaceValueCapitalActions
from skills.million_replay import Replay, UnresolvedAction

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT/'docs/all_entry_2352_capital_20261011.json'


def account_fixture():
    action = dict(TERMS, action_id=KEY, stock_id='2352', date='2025-10-07',
                  evidence_files=['synthetic-reviewed-notice'])
    provider = SimpleNamespace(on_date=lambda sid, day: [deepcopy(action)]
        if (sid, day) == ('2352', '2025-10-07') else [])
    account = Replay.__new__(Replay)
    account.holdings = {'2352': dict(qty=8281, event_id='bollinger-2352', due_index=1945)}
    account.marks = {'2352': dict(price=30.15)}
    account.receivables = []; account.actions = []; account.cash_ledger = []
    account.cash = 1000.
    account.raw = lambda day, sid: 30.15 if day == pd.Timestamp('2025-09-24') else 35.4
    original = FaceValueCapitalActions(provider, account)
    account.corporate = CapitalCompletionRoute(original, account)
    return account, action, provider, original


def test_actual_bollinger_quantity_exchanges_only_on_date_and_cannot_spend_fraction():
    account, _, _, _ = account_fixture()
    account.corporate_day(pd.Timestamp('2025-10-03'))
    assert account.holdings['2352']['qty'] == 8281 and account.cash == 1000.
    account.corporate_day(pd.Timestamp('2025-10-07'))
    assert account.holdings['2352']['qty'] == 6790
    assert account.cash == 15905.  # floor(8281 * 1.8), separately from the fraction
    right, = account.receivables
    assert right['amount'] == 12. and right['pay_date'] is None
    assert right['net_amount_verified'] is False
    exchange = account.actions[0]
    assert exchange['fractional_right'] == .42 and exchange['fractional_reference_price'] == 30.15
    account.corporate_day(pd.Timestamp('2025-10-08'))
    assert account.cash == 15905. and account.receivables == [right]
    audit_capital_cash(dict(corporate_actions=account.actions, cash_ledger=account.cash_ledger,
        receivables=account.receivables, daily=[{'date': '2025-10-03'}, {'date': '2025-10-07'}, {'date': '2025-10-08'}]))


@pytest.mark.parametrize('change', ['duplicate', 'ratio', 'payment', 'reference'])
def test_invalid_or_duplicate_settlement_fails_before_cash_or_share_change(change):
    account, action, provider, _ = account_fixture()
    if change == 'duplicate': provider.on_date = lambda sid, day: [action, action]
    if change == 'ratio': action['multiplier'] = .8
    if change == 'payment': action['pay_date'] = '2025-10-03'
    if change == 'reference': account.raw = lambda day, sid: 31.
    with pytest.raises(UnresolvedAction): account.corporate_day(pd.Timestamp('2025-10-07'))
    assert account.cash == 1000. and account.holdings['2352']['qty'] == 8281
    assert not account.receivables and not account.actions


def test_other_events_keep_the_existing_frozen_adapter():
    account, _, provider, original = account_fixture()
    rows = [dict(kind='cash_dividend', cash_per_share=2., stock_id='1101')]
    provider.on_date = lambda sid, day: deepcopy(rows)
    assert account.corporate.on_date('1101', '2025-10-07') == original.on_date('1101', '2025-10-07')
    assert account.corporate.on_date('2352', '2025-10-08') == rows


@pytest.mark.parametrize('field,value', [('multiplier', .8), ('cash_per_share', 2.),
    ('pay_date', '2025-10-03'), ('fractional_reference_date', '2025-10-07'),
    ('fractional_cash_pay_date', '2025-10-07'), ('fractional_cash_available_for_trading', True)])
def test_document_cannot_change_reviewed_amounts_or_promote_fraction_to_cash(tmp_path, field, value):
    document = json.loads(DOCUMENT.read_text()); document['terms'][field] = value
    path = tmp_path/'capital.json'; path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='reviewed entitlement'):
        load_capital_completion(tmp_path, path)


def test_reviewed_production_sources_and_market_date_match():
    document = json.loads(DOCUMENT.read_text())
    if not all((ROOT/source['path']).is_file() for source in document['sources'].values()):
        pytest.skip('Local sealed primary evidence is not in the source-only checkout')
    key, row, refs = load_capital_completion(ROOT, DOCUMENT)
    assert key == KEY and row['multiplier'] == .82 and len(refs) == 11
