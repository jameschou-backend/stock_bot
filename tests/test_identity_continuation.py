import hashlib
import pytest

from scripts import audit_identity_continuation as audit


def test_html_evidence_is_reextracted_and_tampering_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    file=tmp_path/'source.html';file.write_text('<p>1234 上櫃日：2010/02/03</p>')
    row=dict(stock_id='1234',source_path='source.html',source_sha256=audit.sha(file),
             source_url='https://example.org/listing',
             verification=dict(kind='html',snippets=['1234 上櫃日：2010/02/03']))
    refs={}
    assert audit.verify_primary(row,refs)=='primary_text_reextracted'
    assert refs=={'source.html':audit.sha(file)}
    file.write_text('<p>1234 上櫃日：2011/02/03</p>')
    with pytest.raises(ValueError,match='changed'): audit.verify_primary(row,{})
    row['source_sha256']=audit.sha(file)
    with pytest.raises(ValueError,match='not found'): audit.verify_primary(row,{})


def test_visual_review_is_explicit_and_requires_hashed_render(tmp_path, monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    (tmp_path/'scan.pdf').write_bytes(b'pdf fixture')
    (tmp_path/'page.png').write_bytes(b'page fixture')
    row=dict(source_path='scan.pdf',source_sha256=audit.sha(tmp_path/'scan.pdf'),
             source_url='https://example.org/listing',verification=dict(kind='pdf',page=2,
             visual_only=True,snippets=['human reviewed listing date'],render_path='page.png',
             render_sha256=audit.sha(tmp_path/'page.png')))
    assert audit.verify_primary(row,{})=='visual_review_with_hashed_page'
    (tmp_path/'page.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'): audit.verify_primary(row,{})


def test_evidence_must_stay_within_project(tmp_path, monkeypatch):
    project=tmp_path/'project';project.mkdir()
    outside=tmp_path/'outside';outside.write_bytes(b'x')
    monkeypatch.setattr(audit,'ROOT',project)
    with pytest.raises(ValueError,match='changed'):
        audit.checked_file('../outside',hashlib.sha256(b'x').hexdigest(),{})


def test_csv_dates_must_belong_to_exactly_one_matching_company(tmp_path, monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    file=tmp_path/'basic.csv'
    file.write_text('公司代號,上櫃日期\n6111,20010808\n6125,20020123\n',encoding='utf-8-sig')
    row=dict(stock_id='6111',start='2001-08-08',source_path='basic.csv',
             source_sha256=audit.sha(file),source_url='https://example.org/basic.csv',
             verification=dict(kind='csv',key_column='公司代號',key_value='6111',
                               values={'上櫃日期':'20010808'}))
    assert audit.verify_primary(row,{})=='primary_csv_row_verified'
    with pytest.raises(ValueError,match='company/date'): audit.verify_primary({**row,'stock_id':'6125'}, {})
    with pytest.raises(ValueError,match='company/date'): audit.verify_primary({**row,'start':'2002-01-23'}, {})
    file.write_text('公司代號,上櫃日期\n6111,20010808\n6111,20010808\n')
    row['source_sha256']=audit.sha(file)
    with pytest.raises(ValueError,match='company/date'): audit.verify_primary(row,{})
