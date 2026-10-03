import hashlib
from scripts.research_event_stage_audit_20261003 import audit_event, available_event


def example(tmp_path):
    p=tmp_path/'document.txt';p.write_text('dated evidence')
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(event_id='event',stock_id='2330',source_local_path=p.name,source_sha256=h,
        original_document_retrieved=True,first_publication_verified=True,
        first_publication_at='2025-01-02T21:00:00+08:00',
        publication_evidence_local_path=p.name,publication_evidence_sha256=h,
        order_or_production=True,realized_growth=False)


def test_source_date_and_exploratory_flag_cannot_replace_publication(tmp_path):
    e=example(tmp_path);e.update(first_publication_verified=False,source_date='2020-01-01',signal_eligible=True)
    r=audit_event(e,tmp_path)
    assert not r['eligible'] and not available_event(r,'2026-01-01')


def test_publication_time_not_event_day(tmp_path):
    r=audit_event(example(tmp_path),tmp_path)
    assert r['eligible'] and not available_event(r,'2025-01-01') and available_event(r,'2025-01-02')


def test_missing_conflicting_or_changed_evidence_unknown(tmp_path):
    e=example(tmp_path);e['realized_growth']=True
    assert 'stage_unknown_or_conflicting' in audit_event(e,tmp_path)['issues']
    e=example(tmp_path);(tmp_path/'document.txt').write_text('changed')
    assert not audit_event(e,tmp_path)['eligible']


def test_naive_timestamp_and_escape_not_accepted(tmp_path):
    e=example(tmp_path);e['first_publication_at']='2025-01-02'
    assert 'missing_verified_publication_timestamp' in audit_event(e,tmp_path)['issues']
    e=example(tmp_path);e['source_local_path']='../document.txt'
    assert not audit_event(e,tmp_path)['eligible']
