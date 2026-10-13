from copy import deepcopy
import json
import hashlib
from pathlib import Path
import pytest
from app import account_analytics as analytics


def account():
    ledger = [dict(kind='initial_deposit', cash_change=1000),
        dict(kind='buy', event_id='a', cash_change=-100),
        dict(kind='buy', event_id='a', cash_change=-50),
        dict(kind='sell', event_id='a', cash_change=70),
        dict(kind='sell', event_id='a', cash_change=75),
        dict(kind='dividend_payment', date='2026-10-02', stock_id='2330', action_id='d', cash_change=10),
        dict(kind='buy', event_id='b', cash_change=-200),
        dict(kind='buy', event_id='c', cash_change=-100),
        dict(kind='sell', event_id='c', cash_change=90)]
    return dict(cash_ledger=ledger, trades=[dict(r, side=r['kind']) for r in ledger if r['kind'] in ('buy','sell')],
        corporate_actions=[dict(kind='payment', date='2026-10-02', stock_id='2330', action_id='d', amount=10, event_id='a')],
        cohorts=[dict(event_id='a', exit_date='2026-09-30', bought_qty=15),
                 dict(event_id='b', exit_date=None, bought_qty=20),
                 dict(event_id='c', exit_date='2026-10-01', bought_qty=10)],
        receivables=[], ending_inventory=dict(positions=[dict(event_id='b',market_value=220)]),
        daily=[dict(date='2026-10-02', cash=795, nav=1015)])


def test_partial_fills_and_later_dividend_belong_to_one_entry_not_each_fill():
    m = analytics.realized_metrics(account())
    assert (m['wins'], m['losses'], m['settled'], m['open']) == (1, 1, 2, 1)
    assert m['win_rate'] == .5
    assert m['average_win'] == 5 and m['average_loss'] == 10
    assert m['profit_factor'] == .5 and m['realized_profit'] == -5


def test_unpaid_receivable_excludes_sold_event_until_resolved():
    a = account()
    a['receivables'] = [dict(event_id='c', kind='cash', amount=30)]
    a['daily'][0]['nav'] += 30
    m = analytics.realized_metrics(a)
    assert m['settled'] == 1 and m['pending_only'] == 1 and m['win_rate'] == 1
    assert m['profit_factor'] is None


@pytest.mark.parametrize('change', ['nav', 'cash', 'ambiguous', 'trades', 'exit'])
def test_bad_financial_provenance_is_rejected(change):
    a = account()
    if change in ('nav','cash'): a['daily'][0][change] += 1
    elif change == 'ambiguous': a['corporate_actions'] *= 2
    elif change == 'trades': a['trades'][0]['cash_change'] -= 1
    else: a['cohorts'][1]['exit_date'] = '2026-10-01'
    with pytest.raises(ValueError): analytics.realized_metrics(a)


def test_buy_and_hold_does_not_claim_a_hundred_percent_win_rate():
    a = account()
    a.update(cash_ledger=[dict(kind='initial_deposit',cash_change=1000),dict(kind='buy',event_id='benchmark',cash_change=-200)],
        trades=[dict(side='buy',event_id='benchmark',cash_change=-200)], corporate_actions=[], cohorts=[],
        ending_inventory=dict(positions=[dict(event_id='benchmark',market_value=300)]),daily=[dict(nav=1100,cash=800)])
    m = analytics.realized_metrics(a)
    assert m['settled'] == 0 and m['win_rate'] is None and m['not_applicable']


def test_supplement_cannot_be_attached_to_a_different_publication(tmp_path):
    path = tmp_path / analytics.SUPPLEMENT
    path.parent.mkdir(parents=True)
    report = dict(cases={'x':dict(completed=True,sha256='account-sha')})
    data = dict(schema='account_analytics_v1',live_qualified=False,
        calculator_sha256=hashlib.sha256(Path(analytics.__file__).read_bytes()).hexdigest(),
        studies={'study':dict(publication_sha256='report-sha',cases={'x':dict(source_sha256='account-sha')})})
    raw = json.dumps(data).encode();path.write_bytes(raw)
    path.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest())
    assert analytics.supplement('study',report,'report-sha',tmp_path)['x']['source_sha256']=='account-sha'
    with pytest.raises(ValueError,match='版本不同'):analytics.supplement('study',report,'other',tmp_path)
    path.write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='指紋不符'):analytics.supplement('study',report,'report-sha',tmp_path)
