import json
from pathlib import Path

import pytest

from scripts.prepare_twse_board_reconciliation_v2 import Fetcher,query,identity


def test_dispatch_completes_days_without_mutating_plan():
    from scripts.prepare_twse_board_reconciliation_v2 import dispatch_order,DAY_KIND_ORDER
    entries=[query(kind,day) for kind in reversed(DAY_KIND_ORDER) for day in ('2022-01-04','2022-01-03')]
    before=json.dumps(entries)
    ordered=dispatch_order(entries)
    assert [(i['date'],i['kind']) for i in ordered]==[(day,kind) for day in ('2022-01-03','2022-01-04') for kind in DAY_KIND_ORDER]
    assert json.dumps(entries)==before


def test_html_discovery_is_limited_to_official_detail_document(monkeypatch,tmp_path):
    response=Response();response.content=b'<html data-api="block/BFIAUU"></html>'
    calls=patch_runtime(monkeypatch,tmp_path,[response])
    value=dict(kind='detail_documentation',date='2022-01-03',url='https://www.twse.com.tw/zh/trading/block/bfiauu-detail.html',params={},response_format='html')
    value['identity']=identity(value)
    assert Fetcher(tmp_path/'cache',3).fetch(value)['accepted']
    value['url']='https://www.twse.com.tw/unknown.html';value['identity']=identity(value)
    with pytest.raises(ValueError,match='official HTTPS'):Fetcher(tmp_path/'cache',3).fetch(value)
    assert len(calls)==1


def item(day='2022-01-03'):
    value=query('total',day);value['identity']=identity(value);return value


class Response:
    def __init__(self,status=200,payload=None):
        self.status_code=status
        self.payload=payload or dict(date='20220103',stat='OK')
        self.content=json.dumps(self.payload).encode()
    def json(self):return self.payload


def patch_runtime(monkeypatch,tmp_path,responses):
    import scripts.prepare_twse_board_reconciliation_v2 as module
    clock=[100.0];calls=[]
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module.time,'time',lambda:clock[0])
    monkeypatch.setattr(module.time,'sleep',lambda wait:clock.__setitem__(0,clock[0]+wait))
    class Session:
        headers={}
        def get(self,url,**kwargs):
            assert kwargs['allow_redirects'] is False
            calls.append(clock[0])
            # The identity is already persisted before any network request.
            events=[json.loads(x) for x in (tmp_path/'cache/requests.jsonl').read_text().splitlines()]
            assert events[-1]['event']=='start'
            response=responses.pop(0)
            if isinstance(response,Exception):raise response
            return response
    monkeypatch.setattr(module.requests,'Session',Session)
    return calls


def test_success_is_reused_and_starts_are_rate_limited(monkeypatch,tmp_path):
    calls=patch_runtime(monkeypatch,tmp_path,[Response(),Response(payload=dict(date='20220104',stat='OK'))])
    f=Fetcher(tmp_path/'cache',3)
    first=f.fetch(item());assert first['accepted']
    assert f.fetch(item())==first and len(calls)==1
    assert f.fetch(item('2022-01-04'))['accepted']
    assert calls[1]-calls[0]>=1.5
    # A fresh process also reuses the same receipt without another request.
    assert Fetcher(tmp_path/'cache',3).fetch(item())==first and len(calls)==2


def test_transient_failure_has_bounded_retries(monkeypatch,tmp_path):
    calls=patch_runtime(monkeypatch,tmp_path,[Response(503),Response(503),Response(503)])
    f=Fetcher(tmp_path/'cache',10)
    result=f.fetch(item())
    assert not result['accepted'] and len(calls)==3
    assert Fetcher(tmp_path/'cache',10).fetch(item())==result and len(calls)==3


def test_budget_blocks_before_transport(monkeypatch,tmp_path):
    calls=patch_runtime(monkeypatch,tmp_path,[Response(503)])
    with pytest.raises(ValueError,match='budget'):Fetcher(tmp_path/'cache',1).fetch(item())
    assert len(calls)==1


def test_wrong_date_and_nontransient_are_not_retried(monkeypatch,tmp_path):
    calls=patch_runtime(monkeypatch,tmp_path,[Response(payload=dict(date='20260925',stat='OK'))])
    result=Fetcher(tmp_path/'cache',3).fetch(item())
    assert not result['accepted'] and result['semantic_error']=='response_date_or_status_mismatch'
    assert len(calls)==1


def test_success_cache_digest_change_is_fatal(monkeypatch,tmp_path):
    calls=patch_runtime(monkeypatch,tmp_path,[Response()])
    f=Fetcher(tmp_path/'cache',3);result=f.fetch(item())
    (tmp_path/result['raw_path']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):f.fetch(item())
    assert len(calls)==1


def test_interrupted_attempt_still_consumes_budget(monkeypatch,tmp_path):
    calls=patch_runtime(monkeypatch,tmp_path,[])
    folder=tmp_path/'cache';folder.mkdir()
    folder.joinpath('requests.jsonl').write_text(json.dumps(dict(event='start',identity=item()['identity'],epoch=100.0))+'\n')
    with pytest.raises(ValueError,match='budget'):Fetcher(folder,1).fetch(item())
    assert not calls


def test_official_security_page_opens_circuit_before_next_identity(monkeypatch,tmp_path):
    from scripts.prepare_twse_board_reconciliation_v2 import OfficialAccessDenied
    response=Response(307);response.content=b'FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED.'
    calls=patch_runtime(monkeypatch,tmp_path,[response])
    with pytest.raises(OfficialAccessDenied,match='stop this origin'):
        Fetcher(tmp_path/'cache',3).fetch(item())
    events=[json.loads(x) for x in tmp_path.joinpath('cache/requests.jsonl').read_text().splitlines()]
    assert events[-1]['accepted'] is False and events[-1]['security_denied'] is True
    assert len(calls)==1
    # A normal restart does not silently resend a permanent rejection.
    with pytest.raises(OfficialAccessDenied,match='Shared official-origin hold'):
        Fetcher(tmp_path/'cache',3).fetch(item())
    assert len(calls)==1


def test_three_interrupted_attempts_return_explicit_blocked_receipt(monkeypatch,tmp_path):
    calls=patch_runtime(monkeypatch,tmp_path,[])
    folder=tmp_path/'cache';folder.mkdir()
    starts=[dict(event='start',identity=item()['identity'],epoch=100.0,attempt=i) for i in range(1,4)]
    folder.joinpath('requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in starts))
    result=Fetcher(folder,10).fetch(item())
    assert result['accepted'] is False and result['error_type']=='interrupted_attempt_limit_exhausted'
    assert not calls


def test_global_denial_requires_explicit_same_identity_recovery_after_cooldown(monkeypatch,tmp_path):
    from scripts.prepare_twse_board_reconciliation_v2 import OfficialAccessDenied
    calls=patch_runtime(monkeypatch,tmp_path,[Response()])
    folder=tmp_path/'cache';folder.mkdir()
    raw=folder/'denied.html';raw.write_bytes(b'FOR SECURITY REASONS')
    from scripts.prepare_twse_board_reconciliation_v2 import digest
    denied=dict(event='finish',identity=item()['identity'],accepted=False,retryable=False,security_denied=True,
        http_status=307,retrieved_at='1970-01-01T00:00:00+00:00',raw_path='cache/denied.html',raw_sha256=digest(raw))
    folder.joinpath('requests.jsonl').write_text(json.dumps(denied)+'\n')
    import scripts.prepare_twse_board_reconciliation_v2 as module
    module.time.sleep(301)
    with pytest.raises(OfficialAccessDenied,match='Origin blocked'):
        Fetcher(folder,3).fetch(item('2022-01-04'))
    with pytest.raises(OfficialAccessDenied,match='same-identity'):
        Fetcher(folder,3,allow_security_retry=True).fetch(item('2022-01-04'))
    assert Fetcher(folder,3,allow_security_retry=True).fetch(item())['accepted']
    events=[json.loads(x) for x in folder.joinpath('requests.jsonl').read_text().splitlines()]
    assert events[-1]['event']=='security_recovery' and len(calls)==1


def test_shared_origin_hold_blocks_even_explicit_local_recovery(monkeypatch,tmp_path):
    from scripts.prepare_twse_board_reconciliation_v2 import OfficialAccessDenied,digest
    calls=patch_runtime(monkeypatch,tmp_path,[])
    cache=tmp_path/'cache';cache.mkdir()
    denied=tmp_path/'another-agent-denial.html';denied.write_text('Official security challenge')
    (cache/'origin-hold.json').write_text(json.dumps(dict(evidence_sha256={'another-agent-denial.html':digest(denied)})))
    with pytest.raises(OfficialAccessDenied,match='Shared official-origin hold'):
        Fetcher(cache,3,allow_security_retry=True).fetch(item())
    assert not calls


def test_428_challenge_is_not_retried(monkeypatch,tmp_path):
    from scripts.prepare_twse_board_reconciliation_v2 import OfficialAccessDenied
    calls=patch_runtime(monkeypatch,tmp_path,[Response(428)])
    with pytest.raises(OfficialAccessDenied,match='stop this origin'):Fetcher(tmp_path/'cache',3).fetch(item())
    assert len(calls)==1


def test_308_is_not_followed_or_accepted(monkeypatch,tmp_path):
    from scripts.prepare_twse_board_reconciliation_v2 import OfficialAccessDenied
    calls=patch_runtime(monkeypatch,tmp_path,[Response(308)])
    with pytest.raises(OfficialAccessDenied,match='stop this origin'):Fetcher(tmp_path/'cache',3).fetch(item())
    assert len(calls)==1


def test_repo_origin_hold_applies_to_new_cache(monkeypatch,tmp_path):
    from scripts.prepare_twse_board_reconciliation_v2 import OfficialAccessDenied,digest,ORIGIN_HOLD_RELATIVE
    calls=patch_runtime(monkeypatch,tmp_path,[])
    denied=tmp_path/'denied';denied.write_text('428')
    hold=tmp_path/ORIGIN_HOLD_RELATIVE;hold.parent.mkdir(parents=True)
    hold.write_text(json.dumps(dict(evidence_sha256={'denied':digest(denied)})))
    with pytest.raises(OfficialAccessDenied,match='Shared official-origin hold'):
        Fetcher(tmp_path/'brand-new-cache',3,allow_security_retry=True).fetch(item())
    assert not calls


def test_fresh_denial_automatically_holds_a_second_cache(monkeypatch,tmp_path):
    from scripts.prepare_twse_board_reconciliation_v2 import OfficialAccessDenied,ORIGIN_HOLD_RELATIVE,digest
    calls=patch_runtime(monkeypatch,tmp_path,[Response(403)])
    with pytest.raises(OfficialAccessDenied):Fetcher(tmp_path/'cache',3).fetch(item())
    hold=json.loads((tmp_path/ORIGIN_HOLD_RELATIVE).read_text())
    assert hold['no_automatic_recovery'] is True
    assert len(hold['evidence_sha256'])==2
    for name,expected in hold['evidence_sha256'].items():assert digest(tmp_path/name)==expected
    with pytest.raises(OfficialAccessDenied,match='Shared official-origin hold'):
        Fetcher(tmp_path/'second-cache',3,allow_security_retry=True).fetch(item('2022-01-04'))
    assert len(calls)==1
