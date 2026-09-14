import json
from datetime import timedelta

import pytest
import requests
from streamlit.testing.v1 import AppTest
from app import official_review_packet as packet
from app import forward_journal as j, capacity_forward as cap
from tests.test_forward_portfolio import clock
from tests.test_capacity_close_preview import filled


def raw(day='1150914', time='70003', company='1560'):
    return json.dumps([{'出表日期':day,'發言日期':day,'發言時間':time,'公司代號':company,
                        '公司名稱':'fixture','主旨 ':'fixture title','說明':'fixture explanation'}]).encode()


def test_integer_time_title_whitespace_and_scope_are_explicit(tmp_path):
    path=tmp_path/'packet'; now=clock('2026-09-14',12)()
    packet.capture(path,lambda:now,lambda:raw())
    report=packet.inspect({'1560':'tse','0050':'tse','6488':'otc'},path,lambda:now)
    assert report['ready'] and not report['coverage_complete'] and not report['actions_reviewed']
    assert report['stocks'][0]['disclosures'][0]['published_at']=='2026-09-14T07:00:03+08:00'
    assert report['stocks'][1]['scope']=='未涵蓋：需另查基金公告'
    assert report['stocks'][2]['scope']=='未涵蓋：需另查上櫃公告'


@pytest.mark.parametrize('payload',[b'[]',b'<html>error</html>',raw('1150913'),raw(time='130000'),raw(company='abc')])
def test_empty_stale_malformed_future_never_become_clean_review(tmp_path,payload):
    path=tmp_path/'packet'; at=clock('2026-09-14',12)
    result=packet.capture(path,at,lambda:payload)
    assert result['status']=='error'
    assert not packet.inspect({'1560':'tse'},path,at)['ready']


def test_caching_future_cutoff_and_failure_do_not_reuse_old_success(tmp_path):
    path=tmp_path/'packet'; now=clock('2026-09-14',12)()
    first=packet.capture(path,lambda:now,lambda:raw())
    before=path.read_bytes()
    assert packet.inspect({'1560':'tse'},path,lambda:now-timedelta(seconds=1))['snapshot_at'] is None
    assert path.read_bytes()==before
    assert packet.capture(path,lambda:now+timedelta(seconds=1),lambda:pytest.fail('must reuse'))['requests']==0
    later=now+timedelta(seconds=1801)
    assert not packet.inspect({'1560':'tse'},path,lambda:later)['ready']
    def fail():
        raise requests.ConnectionError('secret must not be copied into messages')
    result=packet.capture(path,lambda:later,fail)
    assert result['status']=='error' and result['hash']!=first['hash']
    assert not packet.inspect({'1560':'tse'},path,lambda:later)['ready']
    assert packet.latest(path)['body']['error']=='ConnectionError'
    assert packet.capture(path,lambda:later,lambda:pytest.fail('failure cooldown'))['requests']==0


def test_ui_reads_official_packet_without_approving_or_fetching(tmp_path,monkeypatch):
    root,evidence,now=filled(tmp_path)
    source=tmp_path/'official'
    packet.capture(source,lambda:now,lambda:raw(company='1101'))
    report=packet.inspect({'1101':'tse'},source,lambda:now)
    monkeypatch.setattr(packet,'inspect',lambda stocks:report)
    from app import official_review_packet_ui as ui
    monkeypatch.setattr(ui.auto,'markets',lambda stocks:{sid:'tse' for sid in stocks})
    monkeypatch.setattr(packet,'fetch',lambda:pytest.fail('no UI fetching'))
    before=cap.verify(root/'strategy.sqlite3')
    app=AppTest.from_string('from pathlib import Path\nfrom app.official_review_packet_ui import render\nrender(Path('+repr(str(root))+'),"strategy")').run()
    assert not app.exception and not app.error
    assert any('fixture title' in t.value for t in app.text)
    assert not app.checkbox
    assert cap.verify(root/'strategy.sqlite3')==before
