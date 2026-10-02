"""Offline accounting of a completed primary-market acquisition.

Completing market/date downloads is distinct from certifying a strategy, its
fills, the historical universe, or every volume definition.
"""
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path

from skills.market_input_validation import require
from skills.official_daily_acquisition import (request_item, USER_REQUEST, OfficialDailyAcquisition,
    TRANSPORT_RECOVERY_KIND, validate_transport_failure)
from skills.official_market_supplement import bound_path, sha, validate_entry


ORIGINAL_REPORT_SHA256 = '072a7c35dbc3a1cf2fc4dcec587d587fd5ee048a70e2d7a46c89aaf84a7160f2'
ORIGINAL_HOLD_SHA256 = '9a1a13f85664356abba3ef070bd05fd128a3a7b95bd2e765707e1cbea8b28c0e'
REQUIRED_DAYS = 3928
ORIGINAL_MISSING = 3745
OFFLINE_RECOVERED = 3
PLANNED_DOWNLOADS = 3742
HOLD_PATH = '.cache/official-origin-holds/www.twse.com.tw.json'


def _key(row):
    require(row.get('market') in ('TWSE','TPEX'), 'Unknown market identity')
    stamp = row.get('date')
    require(isinstance(stamp,str) and date.fromisoformat(stamp).isoformat() == stamp, 'Invalid market date')
    return row['market'], stamp


def _index(rows, label):
    require(isinstance(rows,list), label+' entries must be a list')
    result = {}
    for row in rows:
        key = _key(row)
        require(key not in result, 'Duplicate '+label+' market/date')
        result[key] = row
    return result


def _stamp(value):
    stamp = datetime.fromisoformat(value)
    require(stamp.tzinfo is not None, 'Acquisition timestamp lacks timezone')
    return stamp.timestamp()


def verify_completion(root, original_report, offline_manifest, acquisition_cache,
                      final_manifest, *, final_audit=None, output=None):
    """Verify exact sets and every new attempt/receipt; never issue HTTP requests.

    The final supplement has 3,745 days. Its union with the original 183 cached
    days covers 3,928; the supplement alone must not be labelled 3,928 days.
    A final zero-missing-day audit is required before ``complete`` becomes true.
    """
    root = Path(root).resolve()
    refs = {}

    def path(value):
        p = Path(value)
        p = (p if p.is_absolute() else root/p).resolve()
        require(p.is_relative_to(root), 'Completion source escapes repository')
        return p

    def reference(value, expected=None):
        p = path(value)
        name = str(p.relative_to(root))
        actual = sha(p)
        require(expected is None or actual == expected, 'Completion source hash differs: '+name)
        require(name not in refs or refs[name] == actual, 'Completion hash closure conflict')
        refs[name] = actual
        return p

    def bound_json(value, expected=None):
        p = reference(value, expected)
        sidecar = reference(p.with_suffix('.sha256'))
        require(refs[str(p.relative_to(root))] == sidecar.read_text().strip(), 'Changed completion input sidecar')
        return json.loads(p.read_text())

    def closure(mapping):
        require(isinstance(mapping,dict), 'Missing source hash closure')
        for name, expected in mapping.items():
            reference(bound_path(root,name),expected)

    original = bound_json(original_report, ORIGINAL_REPORT_SHA256)
    require(original['schema'] == 'market_input_validation_v1' and original['live_qualified'] is False,
            'Unsupported original published audit')
    closure(original['source_sha256'])
    required = _index(original['request_plan'],'original required')
    require(len(required) == REQUIRED_DAYS, 'Original required-day count differs')
    require(all(r['status'] in ('cached','source_missing') for r in required.values()), 'Unknown original source status')
    cached = {k for k,r in required.items() if r['status'] == 'cached'}
    missing = set(required)-cached
    require(len(missing) == ORIGINAL_MISSING == original['requests_lower_bound'], 'Original missing-day count differs')

    # Original format is authenticated through its published hash closure.
    # Do not retrofit new receipt fields or HTTP statuses onto old receipts.
    original_source_rows = _index(original['sources'],'original source')
    originals = {}
    for key,row in original_source_rows.items():
        for field in ('path','receipt'):
            name = row[field]
            require(name in original['source_sha256'], 'Original source missing from its hash closure')
            reference(name,original['source_sha256'][name])
        require(row['sha256'] == original['source_sha256'][row['path']], 'Original source descriptor hash differs')
        if key in cached:
            originals[key] = row
    require(set(originals) == cached, 'Original cached days lack sealed source descriptors')

    def manifest(value, label):
        data = bound_json(value)
        require(data.get('schema') == 'official_market_supplement_v1' and data.get('live_qualified') is False,
                'Unsupported '+label+' manifest')
        closure(data.get('source_sha256',{}))
        entries = _index(data['entries'],label)
        return data,entries

    offline, offline_rows = manifest(offline_manifest,'offline')
    require(len(offline_rows) == OFFLINE_RECOVERED and set(offline_rows) <= missing,
            'Offline recovered days differ from original missing set')
    for row in offline_rows.values():
        validate_entry(row,root)
        reference(row['raw_path'],row['raw_sha256'])
        reference(row['receipt_path'],row['receipt_sha256'])

    cache = path(acquisition_cache)
    plan_path = cache/'plan.json'
    plan = bound_json(plan_path)
    require(plan.get('schema') == 'official_daily_plan_v1' and plan.get('automatic_retries') == 0
            and plan.get('minimum_start_interval_seconds',0) >= 3.1, 'Invalid acquisition plan policy')
    closure(plan['source_sha256'])
    planned = _index(plan['entries'],'acquisition plan')
    require(len(planned) == PLANNED_DOWNLOADS == plan['max_requests'], 'Planned download count differs')
    require(not set(planned) & set(offline_rows) and set(planned) | set(offline_rows) == missing,
            'Offline and acquisition plan do not partition original missing days')
    require(all(r == request_item(*key) for key,r in planned.items()), 'Acquisition plan endpoint/query differs')
    plan_hash = sha(plan_path)

    final, final_rows = manifest(final_manifest,'final supplement')
    require(set(final_rows) == missing and len(final_rows) == ORIGINAL_MISSING,
            'Final supplement is missing days or includes unknown identities')
    for key,row in offline_rows.items():
        require(all(final_rows[key].get(k) == row[k] for k in
                    ('raw_path','raw_sha256','receipt_path','receipt_sha256')), 'Offline source replaced in final supplement')
    final_descriptors = {}
    for key,row in final_rows.items():
        _, descriptor = validate_entry(row,root)
        final_descriptors[key] = descriptor
        closure(descriptor.get('recovery_source_sha256',{}))
        reference(row['raw_path'],row['raw_sha256'])
        reference(row['receipt_path'],row['receipt_sha256'])

    ids = {row['identity'] for row in planned.values()}
    attempts = {p.stem:p for p in (cache/'attempts').glob('*.json')}
    receipts = {p.stem:p for p in (cache/'receipts').glob('*.json')}
    require(set(attempts) == ids, 'Missing or unknown acquisition attempts')
    require(set(receipts) == ids, 'Missing, unfinished or unknown acquisition receipts')
    recovery_attempts = {p.parent.name:p for p in (cache/'recoveries').glob('*/attempt.json')}
    recovery_receipts = {p.parent.name:p for p in (cache/'recoveries').glob('*/receipt.json')}
    require(set(recovery_attempts) == set(recovery_receipts) and set(recovery_attempts) <= ids,
            'Unknown or unfinished transport recovery')
    reader = OfficialDailyAcquisition(root,cache)
    probes, accepted_by_market, effective_receipts, original_failures = {}, Counter(), {}, []
    for key,item in planned.items():
        attempt_path, receipt_path = attempts[item['identity']],receipts[item['identity']]
        attempt = json.loads(reference(attempt_path).read_text())
        receipt = bound_json(receipt_path)
        require(attempt.get('schema') == 'official_daily_attempt_v1', 'Invalid acquisition attempt')
        require(all(attempt.get(k) == receipt.get(k) == v for k,v in item.items())
                and attempt.get('plan_sha256') == receipt.get('plan_sha256') == plan_hash,
                'Original attempt/receipt request differs')
        recovered = receipt.get('accepted') is not True
        if recovered:
            require(item['identity'] in recovery_receipts, 'Unresolved original transport failure')
            recovery_metadata = json.loads(recovery_receipts[item['identity']].read_text())
            validate_transport_failure(receipt,item,plan_hash,receipt_sha256=sha(receipt_path),
                legacy_failure_sha256=recovery_metadata.get('legacy_failure_sha256'))
            original_failures.append(dict(market=key[0],date=key[1],error_type=receipt['error_type'],
                receipt_path=str(receipt_path.relative_to(root)),receipt_sha256=sha(receipt_path),
                exception_response_presence=('recorded_absent' if receipt.get('exception_response_present') is False
                                             else 'not_recorded_in_original_version')))
            reader.inspect_transport_recovery(item)
            attempt_path = recovery_attempts[item['identity']]
            receipt_path = recovery_receipts[item['identity']]
            attempt = json.loads(reference(attempt_path).read_text())
            receipt = bound_json(receipt_path)
            for prefix in ('base_receipt','base_attempt','recovery_proof'):
                reference(receipt[prefix+'_path'],receipt[prefix+'_sha256'])
        else:
            require(item['identity'] not in recovery_receipts, 'Recovery exists for successful original request')
        require(receipt.get('schema') == 'official_daily_receipt_v1'
                and receipt.get('accepted') is True and receipt.get('status') == 'verified_market_day',
                'Failed acquisition receipt')
        require(all(attempt.get(k) == receipt.get(k) == v for k,v in item.items()), 'Attempt/receipt request differs')
        require(attempt.get('plan_sha256') == receipt.get('plan_sha256') == plan_hash, 'Receipt references another plan')
        require(attempt.get('request_kind') == receipt.get('request_kind')
                and attempt.get('authorization') == receipt.get('authorization'), 'Receipt request authorization differs')
        require(receipt['started_at'] == attempt['started_at']
                and _stamp(receipt['retrieved_at']) >= _stamp(attempt['started_at']), 'Acquisition timestamps differ')
        row = final_rows[key]
        require(path(row['receipt_path']) == receipt_path.resolve()
                and row['receipt_sha256'] == sha(receipt_path)
                and row['raw_path'] == receipt['raw_path']
                and row['raw_sha256'] == receipt['raw_sha256'], 'Final manifest does not match acquired receipt')
        expected_raw = (cache/'recoveries'/item['identity']/'raw.bin' if recovered
                        else cache/'raw'/(item['identity']+'.bin'))
        require(path(receipt['raw_path']) == expected_raw.resolve(), 'Unexpected acquired raw path')
        require(receipt.get('http_status') == 200 and receipt.get('security_denied') is False
                and receipt.get('automatic_redirects_disabled') is True and receipt.get('redirect_statuses') == [],
                'Unsafe acquired receipt')
        accepted_by_market[key[0]] += 1
        effective_receipts[key] = receipt
        if attempt['request_kind'] == 'single_normal_probe':
            require(key[0] not in probes, 'More than one normal probe for a market')
            authorization = attempt['authorization']
            document = json.loads(reference(authorization['path'],authorization['sha256']).read_text())
            require(document.get('schema') == 'official_daily_authorization_v1'
                    and document.get('user_request') == USER_REQUEST
                    and document.get('scope') == 'missing_official_daily_tables'
                    and document.get('security_bypass_authorized') is False
                    and document.get('plan_sha256') == plan_hash
                    and document.get('created_at') == authorization['created_at'], 'Invalid current-request authorization')
            require(0 <= _stamp(attempt['started_at'])-_stamp(document['created_at']) <= 86400,
                    'Probe was outside authorization period')
            probes[key[0]] = receipt
        elif recovered:
            require(attempt['request_kind'] == TRANSPORT_RECOVERY_KIND, 'Unrecognized reviewed transport recovery')
            reference(receipt['authorization']['path'],receipt['authorization']['sha256'])
        else:
            require(attempt['request_kind'] == 'planned_missing_day' and attempt['authorization'] is None,
                    'Unknown acquisition request kind')
    require(set(probes) == {m for m,d in planned}, 'Every acquisition market needs its successful normal probe')
    for key,item in planned.items():
        receipt = effective_receipts[key]
        if receipt['request_kind'] in ('planned_missing_day',TRANSPORT_RECOVERY_KIND):
            require(0 <= _stamp(receipt['started_at'])-_stamp(probes[key[0]]['retrieved_at']) <= 86400,
                    'Download occurred outside successful endpoint proof period')
        if receipt['request_kind'] == TRANSPORT_RECOVERY_KIND:
            require(receipt['authorization'] == probes[key[0]]['authorization'], 'Recovery changed the original authorization')

    hold = reference(HOLD_PATH,ORIGINAL_HOLD_SHA256)
    hold_data = json.loads(hold.read_text())
    closure(hold_data.get('evidence_sha256',{}))
    require(hold_data.get('status') == 'blocked' and hold_data.get('no_automatic_recovery') is True,
            'Original security hold was not preserved')
    union = cached | set(final_rows)
    require(union == set(required) and len(union) == REQUIRED_DAYS, 'Combined market coverage differs')

    audit_summary = None
    if final_audit is not None:
        audit = bound_json(final_audit)
        require(audit.get('schema') == 'market_input_validation_v2' and audit.get('live_qualified') is False,
                'Unsupported final data audit')
        closure(audit['source_sha256'])
        manifest_name = str(path(final_manifest).relative_to(root))
        require(audit['source_sha256'].get(manifest_name) == sha(path(final_manifest)),
                'Final audit does not bind final supplement')
        audited = _index(audit['request_plan'],'final audit required')
        require(set(audited) == union and audit['requests_lower_bound'] == 0
                and all(r['status'] == 'cached' for r in audited.values()), 'Final audit still has missing market days')
        audited_sources = _index(audit['sources'],'final audit source')
        require(set(audited_sources) == set(original_source_rows) | set(final_rows),
                'Final audit source coverage differs')
        for key,row in audited_sources.items():
            require(audit['source_sha256'].get(row['path']) == row['sha256']
                    and row['receipt'] in audit['source_sha256'], 'Final audit source descriptor lacks hash evidence')
            expected = final_descriptors[key] if key in final_rows else original_source_rows[key]
            fields = ['path','sha256','receipt','rows','volume_scope']
            fields.extend(k for k in ('receipt_sha256','http_status','http_status_evidence','retrieved_at','url','recovery_source_sha256')
                          if k in expected)
            require(all(row.get(k) == expected[k] for k in fields),
                    'Final audit source descriptor differs from verified source')
            if 'receipt_sha256' in row:
                require(row['receipt_sha256'] == audit['source_sha256'][row['receipt']],
                        'Final audit receipt descriptor hash differs')
        audit_summary = dict(path=str(path(final_audit).relative_to(root)), sha256=sha(path(final_audit)),
            missing_market_days=0, required_market_days=len(audited),
            complete_verified_data=audit.get('complete_verified_data'), checks=audit.get('checks'))

    for code in (Path(__file__),Path(__file__).with_name('official_daily_acquisition.py'),
                 Path(__file__).with_name('official_market_supplement.py')):
        if code.resolve().is_relative_to(root):
            reference(code)
    result = dict(schema='official_market_completion_v1', created_at=datetime.now(timezone.utc).isoformat(),
        verification_scope='daily_table_acquisition_only', complete=final_audit is not None,
        status='acquisition_and_coverage_verified' if final_audit is not None else 'acquisition_complete_pending_audit',
        original_missing_market_days=len(missing), original_cached_market_days=len(cached),
        offline_recovered_market_days=len(offline_rows), planned_downloads=len(planned),
        new_accepted_receipts=len(receipts), new_accepted_receipts_by_market=dict(accepted_by_market),
        original_successful_receipts=len(receipts)-len(original_failures),
        all_recorded_receipts=len(receipts)+len(recovery_receipts),
        new_http_requests=len(attempts)+len(recovery_attempts), single_normal_probes=len(probes),
        final_supplement_count=len(final_rows), total_required_days=len(union),
        missing_market_days=0, failed_receipts=len(original_failures), unfinished_attempts=0,
        original_failed_receipts=len(original_failures), recovered_receipts=len(recovery_receipts),
        unresolved_failures=0, reviewed_transport_recovery_requests=len(recovery_attempts),
        original_failure_history=original_failures,
        supplemental_http_status_evidence=dict(Counter(d['http_status_evidence'] for d in final_descriptors.values())),
        original_hold_preserved=True, original_hold_sha256=ORIGINAL_HOLD_SHA256,
        final_audit=audit_summary, source_sha256=refs, network_requests=0,
        finmind_requests=0, live_qualified=False, backtest_qualified=False, actual_fill_verified=False,
        return_recomputed=False,
        limitations=['Completion covers daily table acquisition and exact market/date coverage only.',
            'A legacy receipt without recorded HTTP status remains legacy_status_unknown.',
            'Ordinary-session capacity, historical eligibility, price differences, and executable returns are separate checks.'])
    if output is not None:
        target = path(output)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('x') as stream:
            stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        target.with_suffix('.sha256').write_text(sha(target)+'\n')
    return result
