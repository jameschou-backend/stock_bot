import json

import pytest

from app.backtest_data_followup_ui import load
from skills.publication_versions import digest,encoded


def fixture(root,monkeypatch,*,wrong_scope=False):
    def put(name,value):
        path=root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(encoded(value))
        return dict(path=name,sha256=digest(path.read_bytes()))
    base=put('base.json',{'case_count':20})
    binding={base['path']:base['sha256']}
    descriptors={
        'ordinary':put('ordinary.json',dict(input_sha256={} if wrong_scope else binding)),
        'identity':put('identity.json',{'identity':'verified'}),
        'odd_lot':put('odd/request.json',dict(input_sha256=binding)),
        'publication':put('publication.json',{'versions':[]}),
    }
    put('odd/manifest.json',dict(files_sha256={'stock_days.csv':'a'*64}))
    sources={k:json.loads((root/v['path']).read_text()) for k,v in descriptors.items()}
    seen=[]
    def checked(kind):
        def verify(*args):
            seen.append(kind)
            return sources[kind]
        return verify
    monkeypatch.setattr('skills.board_tape_reconciliation.verify_report',checked('ordinary'))
    monkeypatch.setattr('scripts.audit_historical_universe_followup.verify_report',checked('identity'))
    monkeypatch.setattr('scripts.prepare_odd_lot_evidence_request.verify_request',checked('odd_lot'))
    monkeypatch.setattr('app.backtest_data_followup_ui.verify_archive',checked('publication'))
    value=dict(schema='backtest_data_followup_v1',live_qualified=False,performance_recomputed=False,
        reports=descriptors,base_data_report=base)
    path=root/'index.json'
    path.write_bytes(encoded(value))
    path.with_suffix('.sha256').write_text(digest(path.read_bytes()))
    return path,seen


def test_every_evidence_consumer_revalidates_underlying_source_before_display(tmp_path,monkeypatch):
    path,seen=fixture(tmp_path,monkeypatch)
    _,reports=load(path,tmp_path)
    assert set(seen)==set(reports)=={'ordinary','identity','odd_lot','publication'}


def test_different_twenty_case_report_cannot_lend_coverage(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,monkeypatch,wrong_scope=True)
    with pytest.raises(ValueError,match='相同資料範圍'):
        load(path,tmp_path)


def test_source_verification_failure_reaches_consumer(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,monkeypatch)
    def changed(*args): raise ValueError('source changed')
    monkeypatch.setattr('skills.board_tape_reconciliation.verify_report',changed)
    with pytest.raises(ValueError,match='source changed'):
        load(path,tmp_path)


def test_changed_report_cannot_reuse_published_descriptor(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,monkeypatch)
    (tmp_path/'ordinary.json').write_text('{}')
    with pytest.raises(ValueError,match='已變動'):
        load(path,tmp_path)


def test_verified_result_cannot_diverge_from_displayed_bytes(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr('app.backtest_data_followup_ui.verify_archive',lambda *a:{'different':True})
    with pytest.raises(ValueError,match='核對期間'):
        load(path,tmp_path)


def test_data_evidence_cannot_promote_live_status(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,monkeypatch)
    value=json.loads(path.read_text())
    value['live_qualified']=True
    path.write_bytes(encoded(value))
    path.with_suffix('.sha256').write_text(digest(path.read_bytes()))
    with pytest.raises(ValueError,match='實盤資格'):
        load(path,tmp_path)


def test_overlapping_publications_keep_their_own_verified_evidence(tmp_path,monkeypatch):
    from scripts import publish_backtest_data_followup as publisher
    monkeypatch.setattr(publisher,'ROOT',tmp_path)
    base=tmp_path/'artifacts/forward_simulation/backtest_data_completion_20260925.json'
    base.parent.mkdir(parents=True)
    base.write_text('{}')
    first,second=tmp_path/'first.json',tmp_path/'second.json'
    first.write_text('{"evidence":1}')
    second.write_text('{"evidence":2}')
    out1,out2=tmp_path/'artifacts/one.json',tmp_path/'artifacts/two.json'
    checked=[]
    def verify(staging):
        checked.append(staging.read_bytes())
        if len(checked)==1:
            publisher.publish(out2,second,second,second,second)
    monkeypatch.setattr(publisher,'load',verify)
    publisher.publish(out1,first,first,first,first)
    assert out1.read_bytes()==checked[0]
    assert out2.read_bytes()==checked[1]
    with pytest.raises(ValueError,match='immutable'):
        publisher.publish(out1,second,second,second,second)


def test_failed_evidence_check_does_not_publish_an_artifact(tmp_path,monkeypatch):
    from scripts import publish_backtest_data_followup as publisher
    monkeypatch.setattr(publisher,'ROOT',tmp_path)
    monkeypatch.setattr(publisher,'descriptor',lambda path:dict(path='unused',sha256='a'*64))
    def reject(*args): raise ValueError('evidence is incomplete')
    monkeypatch.setattr(publisher,'load',reject)
    output=tmp_path/'artifacts/report.json'
    with pytest.raises(ValueError,match='incomplete'):
        publisher.publish(output,*[tmp_path/'input.json']*4)
    assert not output.exists()


def test_render_keeps_conflicting_and_missing_evidence_visible(tmp_path,monkeypatch):
    from app import backtest_data_followup_ui as ui
    from streamlit.testing.v1 import AppTest
    report=tmp_path/'report.json'
    report.write_text('{}')
    csv=tmp_path/'stock_days.csv'
    csv.write_text('market,date,stock_id,channel\nTWSE,2022-01-04,0050,intraday_odd\n')
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    monkeypatch.setattr(ui,'REPORT',report)
    index=dict(reports={'odd_lot':{'path':'request.json'}},
        odd_lot_exports={'stock_days.csv':digest(csv.read_bytes())})
    reports=dict(ordinary={'summary':dict(same_scope_aggregate_matched=194,
        aggregate_conflicts=42,independent_daily_source_missing=681)},
        identity=dict(previous_unknown_starts=2,remaining_unknown_starts=0,
            unconfirmed_categories=18,unresolved_current_date_discrepancies=14),
        odd_lot={'total_stock_days':1303},publication={'observations':[{}]*8})
    monkeypatch.setattr(ui,'load',lambda:(index,reports))
    app=AppTest.from_string('from app.backtest_data_followup_ui import render\nrender()').run()
    assert not app.exception and not app.error
    text='\n'.join(v.value for v in app.markdown)
    assert '194 股日一致、42 股日衝突、681 股日待補' in text
    assert '歷史零股仍缺 1,303 股日' in text
    assert len(app.warning)==1 and len(app.get('download_button'))==4
    assert any('未取得實盤資格' in v.value for v in app.caption)


def test_render_refuses_invalid_source_instead_of_displaying_counts(tmp_path,monkeypatch):
    from app import backtest_data_followup_ui as ui
    from streamlit.testing.v1 import AppTest
    report=tmp_path/'report.json'
    report.write_text('{}')
    monkeypatch.setattr(ui,'REPORT',report)
    def reject(): raise ValueError('來源已修改')
    monkeypatch.setattr(ui,'load',reject)
    app=AppTest.from_string('from app.backtest_data_followup_ui import render\nrender()').run()
    assert not app.exception and '來源已修改' in app.error[0].value
    assert not app.get('download_button')
