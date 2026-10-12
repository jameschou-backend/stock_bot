from copy import deepcopy
from types import SimpleNamespace
import pytest
from skills.legacy_return_comparison import ARMS, apply_original_exit, audit_original_exit
from skills.top3_validation import MODES
from app.legacy_return_comparison_ui import validate
from scripts.summarize_legacy_return_comparison import check_prices


def test_shared_signal_ids_are_namespaced_without_changing_rank_or_original_data():
    from scripts.research_legacy_return_comparison_v2 import namespace_events
    rows=[dict(event_id='liquid_universe-2024-01-02-2330',priority=.3,members=['2330']),
          dict(event_id='liquid_universe-2024-01-02-2317',priority=.2,members=['2317'])]
    original=deepcopy(rows);changed=namespace_events(rows)
    assert rows==original
    assert [r['event_id'] for r in changed]==['legacy949-'+r['event_id'] for r in original]
    for before,after in zip(original,changed):
        assert after['legacy_event_id']==before['event_id']
        assert {k:after[k] for k in before if k!='event_id'}=={k:v for k,v in before.items() if k!='event_id'}


def test_original_949_exit_does_not_inherit_the_later_black_candle_rule():
    engine=SimpleNamespace(drawdown_arm='three_black')
    apply_original_exit(engine,'liquid_universe')
    assert engine.drawdown_arm=='control'
    poc=SimpleNamespace(drawdown_arm='three_black')
    apply_original_exit(poc,'poc_priority')
    assert poc.drawdown_arm=='three_black'
    audit_original_exit(dict(trades=[dict(reason='time63')]))
    with pytest.raises(ValueError):audit_original_exit(dict(trades=[dict(reason='three_black')]))
    with pytest.raises(ValueError):apply_original_exit(engine,'unknown')


def test_price_audit_rejects_same_day_lookahead_and_wrong_sell_fraction():
    account=dict(trades=[dict(side='buy',reference_price=105,source_low=100,source_high=110,
                             date='2024-01-03',signal_date='2024-01-02'),
                         dict(side='sell',reference_price=103,source_low=100,source_high=110)])
    assert check_prices(account)['trades_checked']==2
    bad=deepcopy(account);bad['trades'][0]['signal_date']='2024-01-03'
    with pytest.raises(ValueError,match='prior-session'):check_prices(bad)
    bad=deepcopy(account);bad['trades'][1]['reference_price']=105
    with pytest.raises(ValueError,match='price differs'):check_prices(bad)


def test_stock_profit_attribution_includes_rights_and_keeps_merger_lineage():
    from scripts.summarize_legacy_return_comparison import stock_contributions
    a=dict(settings=dict(initial_cash=1000),daily=[dict(date='2026-10-02',nav=1209)],
        cohorts=[dict(event_id='old-entry',stock_id='2867')],
        cash_ledger=[dict(kind='initial_deposit',cash_change=1000),
                     dict(kind='buy',stock_id='2867',event_id='old-entry',cash_change=-500)],
        holdings=[dict(date='2026-10-02',stock_id='2884',event_id='old-entry',market_value=700)],
        receivables=[dict(kind='cash',stock_id='2884',event_id='old-entry',amount=9)])
    assert stock_contributions(a)=={'2867':209}
    a['daily'][0]['nav']+=1
    with pytest.raises(ValueError,match='sum to account profit'):stock_contributions(a)


def publication():
    cases={}
    for a in ARMS:
        for m in MODES:
            r=.2 if a=='benchmark' else .4
            metrics=dict(trading_days=666,initial_cash=1_000_000,total_return=r,max_drawdown=-.3,final_nav=1_000_000*(1+r))
            cases[f'{a}:{m}']=dict(id=a,mode=m,completed=True,total_return=r,max_drawdown=-.3,
                metrics=metrics,benchmark_comparable=True,excess_return=r-.2)
    return dict(schema='legacy_return_comparison_v1',start='2024-01-02',end='2026-10-02',initial_cash=1_000_000,
        position_count=3,buy_fraction=.5,sell_fraction=.3,live_qualified=False,unseen_validation=False,
        actual_fill_verified=False,paper_enabled=False,cases=cases,completed_cases=12,all_completed=True,
        modes={k:dict(commission=v[0],slippage=v[1]) for k,v in MODES.items()},
        benchmark_parity={mode:dict(all_exact=True) for mode in MODES})


@pytest.mark.parametrize('change',['none','money','price','cost','return','excess','parity','paper','unseen','partial'])
def test_dashboard_rejects_scope_drift_and_false_qualification(change):
    report=publication()
    if change=='none':assert validate(report)==report;return
    if change=='money':report['initial_cash']=2_000_000
    if change=='price':report['sell_fraction']=.5
    if change=='cost':report['modes']['combined']['slippage']=.0045
    if change=='return':report['cases']['liquid_universe:baseline']['total_return']=9.49
    if change=='excess':report['cases']['poc_priority:baseline']['excess_return']=5.76
    if change=='parity':report['benchmark_parity']['baseline']['all_exact']=False
    if change=='paper':report['paper_enabled']=True
    if change=='unseen':report['unseen_validation']=True
    if change=='partial':report['cases']['benchmark:baseline']['completed']=False
    with pytest.raises(ValueError):validate(report)
