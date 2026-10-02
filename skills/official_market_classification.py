"""Narrow, reviewed nonordinary classifications; unknown instruments stay unknown.

The scanned annual report requires human/agent visual review. Its exact bytes,
page and table row are pinned below, so editing an overlay cannot turn an
unrelated security into an excluded TDR. This is retrospective classification
evidence, not a complete historical universe or a trading qualification.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re

from skills.market_input_validation import require


SCHEMA = 'official_nonordinary_classification_v1'
CATEGORY = '臺灣存託憑證(TDR)'
LISTING_URL = 'https://www.twse.com.tw/downloads/zh/about/company/annual_98.pdf'
DELISTING_URL = 'https://openapi.twse.com.tw/v1/company/suspendListingCsvAndHtml'
# Visually reviewed from the original PDF, not its empty text extraction or
# a name/code-prefix heuristic. PDF page numbers here are one-based.
REVIEWED_PDF_ROWS = {
    '47a4def2d2bfe7cea08aaa52cefa113a8babce9db9870d7926c6287b4b6f4bfc': {
        'stock_id': '9157', 'name': '陽光能源', 'market': 'TWSE',
        'category': CATEGORY, 'start': '2009-12-11',
        'pdf_page': 18, 'printed_page': 33,
        'table_title': '上市臺灣存託憑證（合計10家）',
        'source_row': ['9157', '陽光能源', 'Solargiga Energy Holdings limited', '98/12/11'],
    },
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _day(value):
    require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value,
            'Canonical classification date required')
    return value


def _path(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Relative evidence path required')
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'Missing or escaped classification evidence')
    return path


def _receipt(evidence, root, source_hashes, url, *, recorded_http):
    raw = _path(root, evidence['raw_path'])
    receipt_path = _path(root, evidence['receipt_path'])
    for path, field in [(raw, 'raw_sha256'), (receipt_path, 'receipt_sha256')]:
        name = str(path.relative_to(root))
        require(sha(path) == evidence[field] == source_hashes.get(name),
                'Classification evidence hash differs')
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get('url') == url and receipt.get('sha256') == evidence['raw_sha256'],
            'Classification receipt provenance differs')
    if 'final_url' in receipt:
        require(receipt['final_url'] == url, 'Classification receipt redirected')
    stamp = datetime.fromisoformat(receipt['retrieved_at'])
    require(stamp.tzinfo is not None, 'Classification receipt timezone missing')
    statuses = [receipt[k] for k in ('status', 'status_code', 'http_status') if k in receipt]
    if recorded_http:
        require(statuses and all(type(v) is int and v == 200 for v in statuses)
                and evidence.get('http_status_evidence') == 'recorded_http_200',
                'Unsuccessful classification source')
    else:
        # The archived delisting receipt never recorded HTTP status. Preserve
        # that limitation rather than inventing a successful transport result.
        require(not statuses and evidence.get('http_status_evidence') == 'legacy_status_unknown',
                'Legacy classification receipt status differs')
    return raw


def load_nonordinary_overlay(manifest_path, root, refs=None):
    """Validate all evidence, return dated entries, and extend hash closure."""
    root = Path(root).resolve()
    path = Path(manifest_path)
    path = (path if path.is_absolute() else root / path).resolve()
    require(path.is_relative_to(root), 'Classification manifest escapes repository')
    require(sha(path) == path.with_suffix('.sha256').read_text().strip(), 'Classification manifest changed')
    manifest = json.loads(path.read_text())
    require(manifest.get('schema') == SCHEMA and manifest.get('live_qualified') is False
            and manifest.get('complete_historical_universe') is False
            and manifest.get('publication_time_archive_complete') is False,
            'Unsupported classification overlay')
    entries, sources = manifest.get('entries'), manifest.get('source_sha256')
    require(isinstance(entries, list) and entries and isinstance(sources, dict) and sources,
            'Empty classification overlay')
    verified_refs = {str(path.relative_to(root)): sha(path)}
    for name, expected in sources.items():
        require(sha(_path(root, name)) == expected, 'Classification source hash changed')
        verified_refs[name] = expected
    groups = defaultdict(list)
    for entry in entries:
        require(isinstance(entry.get('stock_id'), str)
                and re.fullmatch(r'[0-9]{4}', entry['stock_id'])
                and entry.get('market') == 'TWSE' and entry.get('category') == CATEGORY,
                'Unsupported nonordinary classification')
        start, end = _day(entry['start']), _day(entry['end'])
        require(start < end and entry.get('interval') == 'start_inclusive_end_exclusive',
                'Invalid classification interval')
        listing, delisting = entry['listing'], entry['delisting']
        reviewed = REVIEWED_PDF_ROWS.get(listing.get('raw_sha256'))
        require(reviewed is not None, 'PDF classification has not been visually reviewed')
        for key in ('stock_id', 'name', 'market', 'category', 'start'):
            require(entry.get(key) == reviewed[key], 'Classification differs from reviewed PDF row')
        for key in ('pdf_page', 'printed_page', 'table_title', 'source_row'):
            require(listing.get(key) == reviewed[key], 'Classification PDF locator differs')
        _receipt(listing, root, sources, LISTING_URL, recorded_http=True)
        raw = _receipt(delisting, root, sources, DELISTING_URL, recorded_http=False)
        rows = json.loads(raw.read_text())
        require(isinstance(rows, list), 'Unexpected official delisting table')
        matches = [r for r in rows if r.get('Code') == entry['stock_id']]
        require(len(matches) == 1 and matches[0] == delisting.get('source_row'),
                'Delisting evidence must identify exactly one official instrument')
        source_row = matches[0]
        require(source_row['Company'] == entry['name'] + '-DR', 'Official delisting name differs')
        parts = source_row['DelistingDate'].split('/')
        require(len(parts) == 3 and all(p.isdigit() for p in parts), 'Invalid ROC delisting date')
        roc_year, month, day = map(int, parts)
        require(str(date(roc_year + 1911, month, day)) == end
                and delisting.get('end_basis') == 'official_DelistingDate_exclusive',
                'Classification end differs from official delisting date')
        groups[entry['stock_id']].append(entry)
    for rows in groups.values():
        ordered = sorted(rows, key=lambda r: r['start'])
        require(all(left['end'] <= right['start'] for left, right in zip(ordered, ordered[1:])),
                'Overlapping nonordinary classifications')
    if refs is not None:
        require(all(name not in refs or refs[name] == expected for name, expected in verified_refs.items()),
                'Classification source hash closure conflict')
        refs.update(verified_refs)
    return deepcopy(entries)


def resolve_nonordinary(entries, stock_id, stamp, market):
    """Use verified entries only; never extend a classification beyond its dates."""
    _day(stamp)
    require(market in ('TWSE', 'TPEX', 'TPEx'), 'Unknown query market')
    venue = market.upper()
    matched = [r for r in entries if r['stock_id'] == stock_id and r['market'].upper() == venue
               and r['start'] <= stamp < r['end']]
    require(len(matched) <= 1, 'Overlapping nonordinary classifications')
    return deepcopy(matched[0]) if matched else None
