import pytest
from scripts.prepare_mixed_odd import CompletionBudget,PreparedOdds
from scripts.prepare_theme_catalyst import Budget
from skills.replay_market_feeds import URLS,ReplayDataUnavailable


def test_twse_is_rejected_before_any_network(monkeypatch,tmp_path):
    monkeypatch.setattr(Budget,'official',lambda *a,**k:pytest.fail('network dispatched'))
    budget=CompletionBudget(tmp_path/'budget.json',maximum={'finmind':0,'official':20})
    with pytest.raises(ReplayDataUnavailable,match='hold preserved'):budget.official(URLS['twse'])


def test_security_reply_stops_future_calls_without_redirects(monkeypatch,tmp_path):
    calls=[]
    class Response:status_code=428;content=b'blocked'
    def reply(*args,**kwargs):calls.append(kwargs);return Response()
    monkeypatch.setattr(Budget,'official',reply)
    budget=CompletionBudget(tmp_path/'budget.json',maximum={'finmind':0,'official':20})
    with pytest.raises(ReplayDataUnavailable,match='security response'):budget.official(URLS['tpex'])
    with pytest.raises(ReplayDataUnavailable,match='stopped'):budget.official(URLS['tpex'])
    assert len(calls)==1 and calls[0]['allow_redirects'] is False


def test_missing_twse_never_reaches_provider(tmp_path):
    class Provider:
        def get_odd(self,*args):pytest.fail('TWSE dispatched')
    feed=PreparedOdds(tmp_path,tmp_path/'empty',Provider())
    with pytest.raises(ReplayDataUnavailable,match='hold preserved'):feed.get_odd('2022-08-23','0050','TWSE')


def test_odd_market_case_cannot_break_board_tick_audit():
    from scripts.research_mixed_odd import market_routes
    board=dict(stock_id='8261',date='2022-01-04',market='TPEX')
    odd=dict(board,market='tpex')
    assert market_routes([board,odd])=={('8261','2022-01-04'):'TPEX'}
    with pytest.raises(ValueError,match='Conflicting'):
        market_routes([board,dict(odd,market='twse')])
