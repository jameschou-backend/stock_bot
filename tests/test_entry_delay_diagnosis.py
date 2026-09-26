from copy import deepcopy

import pytest

from scripts.analyze_entry_delay import event_book, compare


def case(specs):
    account=dict(cohorts=[],trades=[],cash_ledger=[dict(kind='initial_deposit',cash_change=1_000_000.,date='2022-01-03')],
        holdings=[],corporate_actions=[],receivables=[],orders=[],settings=dict(initial_cash=1_000_000))
    stocks={}
    for event,sid,qty,buy,sell in specs:
        account['cohorts'].append(dict(event_id=event,stock_id=sid,name=sid,entry_date='2022-01-03',exit_date='2022-01-05'))
        stock=stocks.setdefault(sid,dict(cash_flow=0.,holding_value=0.,receivable_value=0.))
        for side,price,day in [('buy',buy,'2022-01-03'),('sell',sell,'2022-01-05')]:
            gross=qty*price;cash=(-gross if side=='buy' else gross)-1
            account['trades'].append(dict(event_id=event,stock_id=sid,side=side,qty=qty,gross=gross,
                total_cost=1.,cash_change=cash,date=day,reference_price=price,reason='test'))
            account['cash_ledger'].append(dict(event_id=event,stock_id=sid,kind=side,cash_change=cash,date=day))
            stock['cash_flow']+=cash
    return finish(account,stocks)


def finish(account,stocks):
    cash=sum(r['cash_change'] for r in account['cash_ledger'])
    assets=sum(s['holding_value']+s['receivable_value'] for s in stocks.values())
    account['daily']=[dict(date='2022-01-03',opening_nav=1_000_000.,nav=1_000_000.),
                      dict(date='2022-01-05',opening_nav=1_000_000.,nav=cash+assets)]
    for s in stocks.values():s['profit']=sum(s[k] for k in ('cash_flow','holding_value','receivable_value'))
    return dict(completed=True,account=account,stock_pnl=stocks,summary=dict(total_return=(cash+assets)/1_000_000-1))


def test_selection_quantity_and_per_initial_share_effects_reconcile():
    a=case([('common','1101',100,10,12),('old','1102',100,20,19)])
    b=case([('common','1101',200,11,13),('new','1103',100,10,14)])
    result=compare(a,b)
    assert result['event_counts']==dict(base=2,delayed=2,common=1,removed=1,added=1)
    assert result['final_asset_gap']==700
    assert sum(result['attribution_twd'].values())==pytest.approx(700)
    assert result['attribution_twd']['removed_events']==102
    assert result['attribution_twd']['added_events']==398
    assert result['attribution_twd']['common_quantity']==pytest.approx(198.5)
    assert result['attribution_twd']['common_entry_gross']==-150
    assert result['attribution_is_counterfactual'] is False


def test_dividend_after_disposal_belongs_to_original_event_and_rights_are_retained():
    value=case([('old','1101',100,10,12),('new','1101',100,11,13)])
    a=value['account'];stocks=value['stock_pnl']
    a['corporate_actions'].append(dict(kind='payment',event_id='old',stock_id='1101',action_id='cash-old',date='2022-01-05'))
    a['cash_ledger'].append(dict(kind='dividend_payment',stock_id='1101',action_id='cash-old',date='2022-01-05',cash_change=10))
    stocks['1101']['cash_flow']+=10
    a['receivables']=[dict(kind='cash',stock_id='1101',event_id='old',amount=3),
                       dict(kind='cash',stock_id='1101',event_id='new',amount=7)]
    stocks['1101']['receivable_value']=10
    book=event_book(finish(a,stocks))
    assert book['old']['components']['distribution_cash']==10
    assert book['new']['components']['distribution_cash']==0
    assert book['old']['components']['terminal_receivables']==3
    assert book['new']['components']['terminal_receivables']==7


def test_unknown_or_ambiguous_cash_is_not_silently_assigned():
    value=case([('a','1101',100,10,12),('b','1101',100,10,12)])
    a=value['account']
    a['cash_ledger'].append(dict(kind='dividend_payment',stock_id='1101',action_id='ambiguous',date='2022-01-05',cash_change=10))
    a['corporate_actions']=[dict(kind='payment',stock_id='1101',event_id=e,action_id='ambiguous',date='2022-01-05') for e in ['a','b']]
    with pytest.raises(ValueError,match='ambiguous'):
        event_book(value)
    a['cash_ledger'][-1].update(event_id='a',kind='external_deposit')
    with pytest.raises(ValueError,match='Unsupported cash'):
        event_book(value)


def test_cash_and_stock_reconciliations_catch_missing_data():
    value=case([('a','1101',100,10,12)])
    bad=deepcopy(value);bad['account']['cash_ledger'][-1]['cash_change']+=5
    with pytest.raises(ValueError,match='corporate cash'):
        event_book(bad)
    bad=deepcopy(value);bad['stock_pnl']['1101']['holding_value']=5
    with pytest.raises(ValueError,match='sealed stock totals'):
        event_book(bad)
    bad=deepcopy(value);bad['account']['trades'][0]['cash_change']+=5
    with pytest.raises(ValueError,match='Trade cash'):
        event_book(bad)


def test_no_cross_calendar_or_incomplete_account_comparisons():
    value=case([('a','1101',100,10,12)])
    other=deepcopy(value);other['account']['daily'][-1]['date']='2022-01-06'
    with pytest.raises(ValueError,match='dates'):
        compare(value,other)
    value['completed']=False
    with pytest.raises(ValueError,match='Incomplete'):
        event_book(value)
