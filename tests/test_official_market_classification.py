from copy import deepcopy
from pathlib import Path
import json

import pytest

from skills.market_input_validation import MarketEvidenceError
from skills import official_market_classification as classification


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def write_manifest(root, value):
    path = root/'overlay.json'
    write_json(path, value)
    path.with_suffix('.sha256').write_text(classification.sha(path)+'\n')
    return path


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # A tiny stand-in for a visually reviewed PDF keeps unit tests independent
    # of private local .cache files. The production reviewed facts stay exact.
    reviewed = deepcopy(next(iter(classification.REVIEWED_PDF_ROWS.values())))
    (tmp_path/'listing.pdf').write_bytes(b'%PDF-1.4 unit-test reviewed source')
    listing_hash = classification.sha(tmp_path/'listing.pdf')
    monkeypatch.setattr(classification, 'REVIEWED_PDF_ROWS', {listing_hash: reviewed})
    write_json(tmp_path/'listing.source.json', dict(url=classification.LISTING_URL,
        final_url=classification.LISTING_URL, status=200, sha256=listing_hash,
        retrieved_at='2026-09-24T15:15:38.906842+00:00'))
    delisting_row = dict(Code='9157', Company='陽光能源-DR', DelistingDate='108/11/12')
    write_json(tmp_path/'delisting.json', [delisting_row])
    write_json(tmp_path/'delisting.source.json', dict(url=classification.DELISTING_URL,
        sha256=classification.sha(tmp_path/'delisting.json'),
        retrieved_at='2026-09-14T04:56:33.056874+00:00'))
    names = ('listing.pdf', 'listing.source.json', 'delisting.json', 'delisting.source.json')
    sources = {name: classification.sha(tmp_path/name) for name in names}
    entry = {k: reviewed[k] for k in ('stock_id','name','market','category','start')}
    entry.update(end='2019-11-12', interval='start_inclusive_end_exclusive')
    entry['listing'] = dict(raw_path='listing.pdf', raw_sha256=sources['listing.pdf'],
        receipt_path='listing.source.json', receipt_sha256=sources['listing.source.json'],
        http_status_evidence='recorded_http_200',
        **{k: reviewed[k] for k in ('pdf_page','printed_page','table_title','source_row')})
    entry['delisting'] = dict(raw_path='delisting.json', raw_sha256=sources['delisting.json'],
        receipt_path='delisting.source.json', receipt_sha256=sources['delisting.source.json'],
        http_status_evidence='legacy_status_unknown',
        end_basis='official_DelistingDate_exclusive', source_row=delisting_row)
    value = dict(schema=classification.SCHEMA, entries=[entry], source_sha256=sources,
        live_qualified=False, complete_historical_universe=False, publication_time_archive_complete=False)
    return tmp_path, value


def load(evidence):
    root, value = evidence
    return classification.load_nonordinary_overlay(write_manifest(root, value), root)


def test_verified_tdr_is_dated_and_never_classifies_other_instruments(evidence):
    rows = load(evidence)
    for day in ('2009-12-11','2018-08-15','2019-11-11'):
        result = classification.resolve_nonordinary(rows, '9157', day, 'TWSE')
        assert result['category'] == classification.CATEGORY
    for sid, day, market in [('9157','2009-12-10','TWSE'),('9157','2019-11-12','TWSE'),
            ('9157','2026-01-01','TWSE'),('9157','2018-08-15','TPEx'),('2330','2018-08-15','TWSE')]:
        assert classification.resolve_nonordinary(rows, sid, day, market) is None
    result = classification.resolve_nonordinary(rows, '9157', '2018-08-15', 'TWSE')
    result['category'] = '股票'
    assert rows[0]['category'] == classification.CATEGORY


def test_full_hash_closure_and_no_mutation_of_original_inputs(evidence):
    root, value = evidence
    before = {name: (root/name).read_bytes() for name in value['source_sha256']}
    refs = {}
    path = write_manifest(root, value)
    classification.load_nonordinary_overlay(path, root, refs)
    assert refs == dict(value['source_sha256'], **{'overlay.json': classification.sha(path)})
    assert all((root/name).read_bytes() == raw for name, raw in before.items())
    with pytest.raises(MarketEvidenceError, match='closure conflict'):
        classification.load_nonordinary_overlay(path, root, {'listing.pdf': '0'*64})


@pytest.mark.parametrize('field,value', [('stock_id','2330'),('market','TPEX'),('category','股票'),
    ('start','2009-12-10'),('end','2019-11-13'),('interval','inclusive'),('name','陽光能源-DR')])
def test_changed_identity_or_boundary_cannot_override_official_evidence(evidence, field, value):
    evidence[1]['entries'][0][field] = value
    with pytest.raises(MarketEvidenceError):
        load(evidence)


@pytest.mark.parametrize('field,value', [('pdf_page',17),('printed_page',32),
    ('table_title','上市普通股'),('source_row',['2330','台積電','TSMC','98/12/11'])])
def test_reviewed_pdf_locator_and_row_cannot_be_reassigned(evidence, field, value):
    evidence[1]['entries'][0]['listing'][field] = value
    with pytest.raises(MarketEvidenceError, match='PDF locator differs'):
        load(evidence)


def test_unknown_pdf_bytes_do_not_become_reviewed_by_changing_manifest_hash(evidence):
    root, value = evidence
    (root/'listing.pdf').write_bytes(b'%PDF-1.4 unrelated source')
    value['source_sha256']['listing.pdf'] = classification.sha(root/'listing.pdf')
    value['entries'][0]['listing']['raw_sha256'] = value['source_sha256']['listing.pdf']
    with pytest.raises(MarketEvidenceError, match='not been visually reviewed'):
        load(evidence)


@pytest.mark.parametrize('mutation', ['manifest', 'raw', 'receipt', 'unused_source', 'escape'])
def test_every_source_hash_and_path_is_validated(evidence, mutation):
    root, value = evidence
    path = write_manifest(root, value)
    if mutation == 'manifest': path.write_text(path.read_text()+' ')
    elif mutation == 'raw': (root/'delisting.json').write_text('[]')
    elif mutation == 'receipt': (root/'listing.source.json').write_text('{}')
    elif mutation == 'unused_source':
        (root/'extra.txt').write_text('x')
        value['source_sha256']['extra.txt'] = '0'*64
        path = write_manifest(root, value)
    else:
        value['source_sha256']['../outside.json'] = '0'*64
        path = write_manifest(root, value)
    with pytest.raises(MarketEvidenceError):
        classification.load_nonordinary_overlay(path, root)


def change_receipt(evidence, name, changes):
    root, value = evidence
    filename = name+'.source.json'
    path = root/filename
    receipt = json.loads(path.read_text())
    receipt.update(changes)
    write_json(path, receipt)
    digest = classification.sha(path)
    value['source_sha256'][filename] = digest
    value['entries'][0][name]['receipt_sha256'] = digest


@pytest.mark.parametrize('name,changes', [('listing',dict(status=403)),
    ('listing',dict(url='https://example.com/annual.pdf')),
    ('listing',dict(final_url='https://example.com/annual.pdf')),
    ('listing',dict(sha256='0'*64)),('listing',dict(retrieved_at='2026-09-24T00:00:00')),
    ('delisting',dict(status=200)),('delisting',dict(url='https://example.com/list.json'))])
def test_status_provenance_and_legacy_unknown_transport_are_preserved(evidence, name, changes):
    change_receipt(evidence, name, changes)
    with pytest.raises(MarketEvidenceError):
        load(evidence)


def test_duplicate_or_overlapping_classifications_are_rejected(evidence):
    evidence[1]['entries'] *= 2
    with pytest.raises(MarketEvidenceError, match='Overlapping'):
        load(evidence)
    rows = [deepcopy(evidence[1]['entries'][0])] * 2
    with pytest.raises(MarketEvidenceError, match='Overlapping'):
        classification.resolve_nonordinary(rows, '9157', '2018-08-15', 'TWSE')


def test_delisting_row_must_match_actual_raw_table(evidence):
    evidence[1]['entries'][0]['delisting']['source_row']['Company'] = '另一家公司'
    with pytest.raises(MarketEvidenceError, match='exactly one official instrument'):
        load(evidence)


@pytest.mark.parametrize('flag', ['live_qualified','complete_historical_universe','publication_time_archive_complete'])
def test_overlay_cannot_claim_complete_or_live_qualification(evidence, flag):
    evidence[1][flag] = True
    with pytest.raises(MarketEvidenceError, match='Unsupported classification overlay'):
        load(evidence)
