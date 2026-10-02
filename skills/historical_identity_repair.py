"""Evidence-bound listing repairs, separate from ordinary-retail buy access.

This is a retrospective correction, not a publication-time archive, trading
authorization, or proof of market-wide historical identity completeness.
"""
from copy import deepcopy
from datetime import date, datetime
import json
from pathlib import Path
import re

from skills.historical_universe_completion import apply_completion, resolve_completion
from skills.market_input_validation import require
from skills.official_market_classification import sha


SCHEMA = 'historical_identity_repair_v1'
TIB_REFORM = '2025-01-06'
TARGETS = {'5292', '6757', '6794', '6869', '6873', '6902', '7780'}
IPO_URL = 'https://www.twse.com.tw/company/newlisting'
REVIEWED_IPO_SOURCE = {
    'raw_sha256': 'c45549cd8c749bab1aa41d65ae81ea5415ecf1e9876da373bafef174fb56ba9b',
    'receipt_sha256': 'e14e001278e05a3aa94f2e338931847799cbca4f8e2505cb9573f6a0505ee9fe',
}
# Reviewed official PDF text, obtained through web.run; these hashes are NOT
# hashes of the source PDF bytes. Changing an extraction requires fresh review.
REVIEWED_EXTRACTIONS = {
    '6757': 'c7724d425122c1427a6a66712431babf846a8fb444d139464ba0bd958e5421b0',
    '6794': '547b98c3c859ba326b28e809052cf5af2d64beac6e157742983ffe43441ea050',
    '6869': '4818cab44921750b02dc2db4850e8603d80d18a9d691e7fd743ac4f0bdaa3c0e',
    '6873': '2d70b0cfd69196c4b62fbe67c02566afdfa2e38b91a08b6ed6a0503a0fd2d81b',
    '6902': 'a3adfd4559ed1b5abf4a3ff27045acf2f14bd25d6258ccd79b2959b4f7b014fe',
    'tib_rule_2023': 'e8ee9be00c2251d1317f6002d6ed7be3bbccb33bb90cf75f7d15e10fe4aba48d',
    'tib_rule_2025': 'e534df5482748626073db3ed5d41686d8b65b4932c6d8b724a8fbdb945dfb8ff',
}


def _day(value):
    require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value,
            'Canonical identity date required')
    return value


def _path(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Relative identity source required')
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'Identity source missing or escapes root')
    return path


def _stamp(value):
    require(datetime.fromisoformat(value.replace('Z', '+00:00')).tzinfo is not None,
            'Source retrieval timezone missing')


def _extraction(key, spec, root, sources):
    path = _path(root, spec['path'])
    require(sources.get(spec['path']) == sha(path) == REVIEWED_EXTRACTIONS[key],
            'Official PDF text extraction needs independent review')
    data = json.loads(path.read_text())
    require(data.get('schema') == 'official_pdf_web_extraction_v1'
            and data.get('source_kind') == 'official_pdf_web_extraction'
            and data.get('source_url') == spec['source_url']
            and data.get('raw_pdf_bytes_obtained') is False and data.get('http_status') is None,
            'PDF extraction provenance differs')
    _stamp(data['retrieved_at'])
    require(data.get('pdf_pages_1based') == spec['pdf_pages_1based'], 'PDF page locator differs')
    # Remove extractor line labels, preserving the literal primary source text.
    return re.sub(r'\s+', '', re.sub(r'L\d+@P[\d-]+:', '', data['extraction']))


def _validate(manifest, root):
    require(manifest.get('schema') == SCHEMA and manifest.get('live_qualified') is False
            and manifest.get('complete_historical_universe') is False
            and manifest.get('publication_time_archive_complete') is False,
            'Unsupported identity repair or qualification promotion')
    sources = manifest['source_sha256']
    require(isinstance(sources, dict) and sources, 'Missing identity source closure')
    for name, digest in sources.items():
        require(sha(_path(root, name)) == digest, 'Identity source hash changed')
    baseline = manifest['base_identity']
    require(sources.get(baseline['path']) == baseline['sha256'], 'Base identity missing from closure')
    base = json.loads(_path(root, baseline['path']).read_text())
    listing = manifest['listing_source']
    require(all(listing.get(key) == digest for key, digest in REVIEWED_IPO_SOURCE.items()),
            'IPO archive needs independent review')
    require(sources.get(listing['raw_path']) == listing['raw_sha256']
            and sources.get(listing['receipt_path']) == listing['receipt_sha256'],
            'IPO source missing from closure')
    receipt = json.loads(_path(root, listing['receipt_path']).read_text())
    require(receipt.get('url') == IPO_URL and receipt.get('sha256') == listing['raw_sha256']
            and receipt.get('params') == {'response': 'json'}, 'IPO receipt differs')
    require(not any(k in receipt for k in ('status', 'status_code', 'http_status'))
            and listing.get('http_status_evidence') == 'legacy_status_unknown',
            'Unrecorded IPO HTTP status must remain unknown')
    _stamp(receipt['retrieved_at'])
    raw = json.loads(_path(root, listing['raw_path']).read_text())
    require(raw['stat'] == 'OK' and raw['fields'] == listing['fields']
            and raw['total'] == len(raw['data']), 'Invalid IPO table shape')
    entries = manifest['entries']
    require(len(entries) == len(TARGETS) and {e['stock_id'] for e in entries} == TARGETS,
            'Duplicate or unsupported identity repair targets')
    for e in entries:
        sid = e['stock_id']
        require(e['market'] == 'TWSE' and e['category'] == '股票', 'Repair is not ordinary TWSE stock')
        require(e.get('interval') == 'start_inclusive_end_exclusive', 'Unsupported identity interval')
        matches = [r for r in raw['data'] if r[0] == sid]
        require(len(matches) == 1 and matches[0] == e['listing_source_row'], 'IPO row differs')
        row = dict(zip(raw['fields'], matches[0]))
        y, m, d = map(int, row['股票上市買賣日期'].split('.'))
        require(e['start'] == date(y + 1911, m, d).isoformat(), 'IPO date differs from official row')
        expected = [ep for ep in base['episodes'] if ep['stock_id'] == sid]
        require(expected == [e['replaces_episode']], 'Repair does not bind exact base episode')
        require(expected[0]['market'] == e['market'] and expected[0]['category'] == '股票'
                and expected[0]['start_evidence'] == 'current_official_ISIN'
                and expected[0]['end'] is None, 'Unsupported base identity')
        transfer = e.get('general_board_start')
        require(e['initial_board'] == ('innovation' if row['備註'] == '創新板' else 'general'),
                'Board differs from IPO category')
        if e['initial_board'] == 'innovation':
            require(e['start'] < _day(transfer) <= expected[0]['snapshot_date'], 'Invalid board boundary')
            text = _extraction(sid, e['transfer_source'], root, sources)
            day = date.fromisoformat(transfer)
            literal = f'{day.year-1911}年{day.month}月{day.day}日開始改列上市買賣'
            require(literal in text and sid in text and '普通股' in text and '創新板' in text,
                    'Board effective date differs from official PDF')
        else:
            require(transfer is None and 'transfer_source' not in e, 'Unexpected board transfer')
    policy = manifest['account_policy']
    require(policy['name'] == 'ordinary_retail_research'
            and policy['innovation_reform_date'] == TIB_REFORM
            and policy['pre_reform_qualified_investor_assumed'] is False
            and policy['risk_notice_assumed_by_default'] is False
            and policy['regular_lot_shares'] == 1000,
            'Unsupported retail policy')
    old = _extraction('tib_rule_2023', policy['old_rule_source'], root, sources)
    new = _extraction('tib_rule_2025', policy['reform_source'], root, sources)
    require('1,000股' in old and '不可盤中零股交易' in old and '可盤後零股交易' in old,
            'Old innovation trading rules differ')
    require('2025年1月6日' in new and '取消合格投資人' in new
            and '初次買進前須已簽署風險預告書' in new and '盤中零股交易' in new,
            'Innovation reform evidence differs')


def load_identity_repair(manifest_path, root, refs=None):
    """Read and verify source closure; source text is not a raw PDF archive."""
    root = Path(root).resolve()
    path = Path(manifest_path)
    path = (path if path.is_absolute() else root / path).resolve()
    require(path.is_relative_to(root), 'Repair manifest escapes root')
    require(sha(path) == path.with_suffix('.sha256').read_text().strip(), 'Repair manifest hash changed')
    manifest = json.loads(path.read_text())
    _validate(manifest, root)
    closure = dict(manifest['source_sha256'], **{str(path.relative_to(root)): sha(path)})
    if refs is not None:
        require(all(k not in refs or refs[k] == v for k, v in closure.items()), 'Source closure conflict')
        refs.update(closure)
    manifest['verified_manifest'] = dict(path=str(path.relative_to(root)), sha256=sha(path))
    return manifest


def apply_identity_repair(base, repair):
    """Apply a loaded repair to its exact base; retain ordinary-share board history."""
    require('verified_manifest' in repair, 'Use load_identity_repair before applying a repair')
    for e in repair['entries']:
        require([ep for ep in base['episodes'] if ep['stock_id'] == e['stock_id']]
                == [e['replaces_episode']], 'Base episode changed after repair review')
    dates = [dict(stock_id=e['stock_id'], market=e['market'], start=e['start'],
                  replaces_snapshot_start=e['replaces_episode']['start'],
                  identity_repair_evidence=deepcopy(e)) for e in repair['entries']]
    episodes = apply_completion(base, dates, [])
    mapped = {e['stock_id']: e for e in repair['entries']}
    output = []
    for ep in episodes:
        entry = mapped.get(ep['stock_id'])
        if entry is None:
            output.append(ep)
            continue
        ep.update(security_type='ordinary_share', board=entry['initial_board'])
        transfer = entry.get('general_board_start')
        if transfer:
            early = deepcopy(ep)
            early.update(end=transfer, end_evidence=deepcopy(entry['transfer_source']))
            output.append(early)
            ep.update(start=transfer, board='general', start_evidence='official_pdf_web_extraction',
                      board_transfer_evidence=deepcopy(entry['transfer_source']))
        output.append(ep)
    result = deepcopy(base)
    result.update(schema=SCHEMA, episodes=output, identity_repair=deepcopy(repair),
                  live_qualified=False, complete_historical_universe=False,
                  publication_time_archive_complete=False, continuous_eligibility_proven=False,
                  performance_recomputed=False)
    result['source_sha256'] = dict(base.get('source_sha256', {}))
    closure = dict(repair['source_sha256'])
    closure[repair['verified_manifest']['path']] = repair['verified_manifest']['sha256']
    for name, digest in closure.items():
        require(name not in result['source_sha256'] or result['source_sha256'][name] == digest,
                'Repaired report source closure conflict')
        result['source_sha256'][name] = digest
    return result


def account_entry_decision(report, stock_id, stamp, *, channel='regular',
                           research_risk_notice_assumed=False, quantity=None):
    """Ordinary-retail buy gate only; never infers qualified-investor status.

    Allowed means this identity/account filter passes, not that an order fills.
    Risk-notice assumption is explicitly research-only and never live evidence.
    This function is not a sell gate: existing owners can have sale rights even
    where new purchases are restricted.
    """
    _day(stamp)
    require(type(research_risk_notice_assumed) is bool, 'Explicit boolean assumption required')
    require(channel in ('regular', 'intraday_odd', 'afterhours_odd'), 'Unknown trading channel')
    require('identity_repair' in report, 'Validated repair report required')
    result = dict(allowed=False, reason='', board=None, assumptions=[], live_qualified=False,
                  scope='identity_and_innovation_buy_access_only')
    identity = resolve_completion(report, stock_id, stamp)
    if identity['status'] != 'identified':
        return dict(result, reason=identity['status'])
    matches = [e for e in report['episodes'] if e['stock_id'] == stock_id
               and e['start'] is not None and e['start'] <= stamp
               and (e['end'] is None or stamp < e['end'])
               and stamp <= e.get('snapshot_date', report['coverage_end'])]
    require(len(matches) <= 1, 'Overlapping identity account dates')
    if not matches:
        return dict(result, reason='outside_identity_snapshot')
    ep = matches[0]
    if ep['category'] not in ('股票', '創新板', 'ETF'):
        return dict(result, reason='outside_ordinary_strategy_scope')
    board = ep.get('board', 'innovation' if ep['category'] == '創新板' else 'general')
    require(board in ('general', 'innovation'), 'Unknown listing board')
    result['board'] = board
    if quantity is not None:
        require(type(quantity) is int and quantity > 0, 'Positive integer share quantity required')
        if ((channel == 'regular' and quantity % 1000)
                or (channel != 'regular' and quantity >= 1000)):
            return dict(result, reason='quantity_does_not_match_channel')
    if board == 'innovation':
        if stamp < TIB_REFORM:
            return dict(result, reason=('innovation_intraday_odd_not_available' if channel == 'intraday_odd'
                                       else 'qualified_investor_status_not_proven'))
        if not research_risk_notice_assumed:
            return dict(result, reason='innovation_risk_notice_not_proven')
        result['assumptions'].append('innovation_risk_notice_signed_research_assumption')
    return dict(result, allowed=True, reason='identity_and_account_scope_pass')


def apply_account_entry_policy(mask, report, *, channel='regular', research_risk_notice_assumed=False):
    """Filter an existing dated listing eligibility matrix without touching inputs.

    First build the matrix with historical_selector_replay.eligibility_matrix
    on apply_identity_repair's output, then apply this buy-access layer. Existing
    unknowns/exclusions are never turned on here. Other innovation stocks whose
    base category is not ordinary remain unchanged and require separate review.
    """
    require('identity_repair' in report, 'Validated repair report required')
    require(type(research_risk_notice_assumed) is bool, 'Explicit boolean assumption required')
    require(channel in ('regular', 'intraday_odd', 'afterhours_odd'), 'Unknown trading channel')
    output = mask.copy(deep=True)
    affected = {e['stock_id'] for e in report['episodes']
                if e.get('board') == 'innovation' or e['category'] == '創新板'}
    for sid in affected.intersection(output.columns):
        for stamp in output.index[output[sid].astype(bool)]:
            if not account_entry_decision(report, sid, str(stamp.date()), channel=channel,
                    research_risk_notice_assumed=research_risk_notice_assumed)['allowed']:
                output.loc[stamp, sid] = False
    output.attrs.update(account_policy='ordinary_retail_research', channel=channel,
                        innovation_risk_notice_research_assumption=research_risk_notice_assumed,
                        live_qualified=False)
    return output
