import pytest
from scripts.prepare_mixed_odd_authorized import AuthorizedBudget
from scripts.prepare_holder_flow_accounts import ExecutionBudget
from scripts.prepare_theme_catalyst import Budget
from skills.replay_market_feeds import ReplayDataUnavailable,URLS


def test_missing_recovery_cannot_open_held_twse(monkeypatch,tmp_path):
    def held(*args,**kwargs):raise ReplayDataUnavailable('held origin')
    monkeypatch.setattr(Budget,'official',held)
    budget=AuthorizedBudget(tmp_path/'budget.json',maximum={'finmind':0,'official':119})
    with pytest.raises(ReplayDataUnavailable,match='held origin'):budget.official(URLS['twse'])


@pytest.mark.parametrize('status',[302,401,403,428])
def test_redirects_and_security_responses_stop_without_retry(monkeypatch,tmp_path,status):
    calls=[]
    class Response:status_code=status;content=b'blocked'
    def reply(*args,**kwargs):calls.append(kwargs);return Response()
    monkeypatch.setattr(ExecutionBudget,'official',reply)
    budget=AuthorizedBudget(tmp_path/'budget.json',maximum={'finmind':0,'official':119})
    with pytest.raises(ReplayDataUnavailable):budget.official(URLS['twse'])
    with pytest.raises(ReplayDataUnavailable,match='stop active'):budget.official(URLS['twse'])
    assert len(calls)==1 and calls[0]['allow_redirects'] is False


def test_scope_excludes_other_official_endpoints(monkeypatch,tmp_path):
    monkeypatch.setattr(ExecutionBudget,'official',lambda *a,**k:pytest.fail('request sent'))
    budget=AuthorizedBudget(tmp_path/'budget.json',maximum={'finmind':0,'official':119})
    with pytest.raises(ReplayDataUnavailable,match='Unapproved'):budget.official('https://www.twse.com.tw/rwd/zh/fund/T86')
