from copy import deepcopy
import pytest
from scripts.append_scanner_daily import combine


def scan(day):
    return dict(schema='multi_strategy_scan_v1',live_qualified=False,account_independent=True,
        start=day,end=day,source_end=day,strategies=[dict(id='poc_up_red',version='1')],
        evaluated_strategy_ids=['poc_up_red'],signal_timing='T_close_confirmed__earliest_next_market_session',
        execution_model=None,portfolio_model=None,returns_inherited=False,provenance={},
        days=[dict(date=day,stocks=[dict(stock_id='2330',results={'poc_up_red':dict(status='matched',first_signal=False)})])])


def test_append_keeps_old_unknown_reasons_metrics_and_first_flags_without_recalculation():
    old,new=scan('2026-10-14'),scan('2026-10-15')
    old['days'][0]['stocks'][0]['results']['poc_up_red']=dict(status='unknown',reasons=['source conflict'],metrics={'poc':None},first_signal=None)
    before=deepcopy(old)
    result=combine(old,new,{'sha256':'a'},{'sha256':'b'})
    assert old==before and result['days'][0]==before['days'][0]
    assert result['days'][1]==new['days'][0]
    assert result['provenance']['daily_append']['historical_days_recomputed']==0
    assert result['provenance']['daily_append']['historical_days_reused']==1


@pytest.mark.parametrize('change',['overlap','rule','universe','promote','duplicate'])
def test_append_rejects_replaced_dates_or_changed_contract(change):
    old,new=scan('2026-10-14'),scan('2026-10-15')
    if change=='overlap':new=scan('2026-10-14')
    if change=='rule':new['strategies'][0]['version']='2'
    if change=='universe':new['days'][0]['stocks'][0]['stock_id']='2317'
    if change=='promote':new['live_qualified']=True
    if change=='duplicate':new['days']*=2
    with pytest.raises(ValueError):combine(old,new,{}, {})
