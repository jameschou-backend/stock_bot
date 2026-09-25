#!/usr/bin/env python3
"""Seal the actual external-source follow-up; never fetch, purchase, or trade."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_odd_lot_daily_gaps import verify_report as verify_daily
from scripts.prepare_odd_lot_evidence_request import verify_request
from skills.publication_versions import as_of, digest, encoded, extend_versions, parse_release, stamp

DIRECTORY = ROOT / '.cache/mops-publication-channel-20260925'
DEMAND = ROOT / '.cache/odd-lot-provider-request-20260925-v3'
DAILY = ROOT / '.cache/odd-lot-daily-gaps-20260925/report.json'
OFFICIAL = ROOT / '.cache/backtest-data-completion-20260925/official'


def build(directory=DIRECTORY):
    demand = verify_request(DEMAND)
    daily = verify_daily(DAILY)
    files = {}

    def consume(path):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Source is outside this repository')
        raw = path.read_bytes()
        files[str(path.relative_to(ROOT))] = digest(raw)
        return raw

    for path in (DEMAND/'request.json', DEMAND/'manifest.json', DEMAND/'stock_days.json',
                 DEMAND/'stock_days.csv', DAILY, DAILY.with_suffix('.sha256')):
        consume(path)
    files.update(daily['input_sha256'])
    files.update(daily['code_sha256'])
    mops = json.loads(consume(directory/'entry.source.json'))
    raw = consume(directory/'entry.html')
    stamp(mops['observed_at'])
    if (mops['url'] != 'https://mops.twse.com.tw/mops/' or mops['http_status'] != 200
            or mops['sha256'] != digest(raw)
            or b'FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED' not in raw):
        raise ValueError('MOPS capture does not match the recorded security response')
    receipt = json.loads(consume(directory/'walsin-december-2022.source.json'))
    if receipt['http_status'] != 200 or receipt['path'] != 'walsin-december-2022.html':
        raise ValueError('Historical issuer receipt missing')
    issuer_raw = consume(directory/receipt['path'])
    if digest(issuer_raw) != receipt['sha256']:
        raise ValueError('Historical issuer bytes differ from their receipt')
    observation = dict(parse_release(issuer_raw, receipt['url']),
                       observed_at=receipt['observed_at'], path=receipt['path'])
    versions = extend_versions([], [observation])
    if as_of(versions, '2023-01-10T23:59:59+08:00'):
        raise ValueError('A newly observed old document leaked into a historical cutoff')
    official = json.loads(consume(OFFICIAL/'manifest.json'))
    for name in ('twse-h4.html', 'tpex-mth.html', 'finmind-technical.html', 'finmind-fundamental.html'):
        source = consume(OFFICIAL/name)
        metadata = official['files'][name]
        if metadata['http_status'] != 200 or digest(source) != metadata['sha256']:
            raise ValueError('Previously captured official product evidence changed: ' + name)
    # Verify the narrow pricing facts actually used in the explanatory report.
    h4 = (OFFICIAL/'twse-h4.html').read_text()
    mth = (OFFICIAL/'tpex-mth.html').read_text()
    if ('data-price="1500"' not in h4 or '上兩個月底' not in h4
            or '2022/11/01' not in mth or '僅提供購買一年前' not in mth
            or '外部使用' not in mth or 'data-price="10000"' not in mth):
        raise ValueError('Product pricing or availability assumptions differ from official captures')
    stock_days = json.loads((DEMAND/'stock_days.json').read_bytes())
    tpex_eligible = [r for r in stock_days if r['market'] == 'TPEX'
                     and '2022-11-01' <= r['date'] < '2025-09-25']
    tpex_months = dict(sorted(Counter(r['date'][:7] for r in tpex_eligible).items()))
    return dict(schema='external_source_followup_v1',
        as_of='2026-09-25', source_directory=str(directory.relative_to(ROOT)),
        daily_gap_report=str(DAILY.relative_to(ROOT)),
        daily_stock_days_acquired=daily['required_stock_days'],
        daily_positive_trade_stock_days=daily['statuses'].get('official_daily_positive_trade', 0),
        daily_data_requests=daily['preparation_official_requests'],
        missing_original_daily_evidence_was_not_proof_of_no_trades=True,
        historical_auction_rows_acquired=0, historical_auction_sessions_acquired=0,
        missing_historical_auction_sessions=demand['total_stock_days'],
        procurement_plan=demand['procurement_plan'],
        tpex_mth_additional_pricing_context=dict(
            public_price_category='external_use', public_twd_per_month=10000,
            individual_internal_use_quote_confirmed=False,
            eligible_requested_months=tpex_months,
            listed_external_use_sum_twd=len(tpex_months)*10000,
            not_a_personal_use_quote=True),
        mops=dict(status='official_security_response', source=mops,
                  historical_records_received=0, bypass_attempted=False),
        issuer=dict(observations=[observation], versions=versions,
                    current_observed_historical_documents_received=1,
                    point_in_time_documents_usable_on_20230110=0,
                    revision_history_received=False),
        new_publication_http_responses=2, finmind_requests=0,
        verification_network_requests=0, paid_requests=0, supplier_messages_sent=False,
        broker_authentication_attempted=False,
        source_scope='Bounded public official interfaces; no proof of absence of all commercial or custom products.',
        announcement_dependency_note='Old news/revenue version history is not automatically a dependency of price-only strategies. Verify each selector and corporate-action dependency separately.',
        strict_data_ready=False, live_qualified=False,
        input_sha256=files,
        code_sha256={name:digest((ROOT/name).read_bytes()) for name in (
            'scripts/audit_external_source_followup.py', 'skills/publication_versions.py')})


def verify_report(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    if digest(raw) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('External-source report hash mismatch')
    report = json.loads(raw)
    if report.get('schema') != 'external_source_followup_v1':
        raise ValueError('Unsupported external-source report')
    for key in ('input_sha256', 'code_sha256'):
        for name, expected in report[key].items():
            source = (ROOT/name).resolve()
            if not source.is_relative_to(ROOT) or digest(source.read_bytes()) != expected:
                raise ValueError('External-source evidence changed: ' + name)
    if report != build(path.parent):
        raise ValueError('External-source report differs from its source replay')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    path = DIRECTORY/'report.json'
    if args.verify:
        report = verify_report(path)
    else:
        if path.exists() or path.with_suffix('.sha256').exists():
            raise ValueError('Do not overwrite a sealed report')
        report = build()
        path.write_bytes(encoded(report))
        path.with_suffix('.sha256').write_text(digest(path.read_bytes())+'\n')
    print(json.dumps({key:report[key] for key in ('daily_stock_days_acquired',
        'daily_positive_trade_stock_days', 'historical_auction_sessions_acquired',
        'missing_historical_auction_sessions', 'finmind_requests')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
