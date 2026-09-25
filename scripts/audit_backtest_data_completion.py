#!/usr/bin/env python3
"""Build/reproduce a strictly offline inventory of actual case data dependencies."""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from bs4 import BeautifulSoup
from skills.backtest_data_evidence import (
    case_evidence, checked, digest, inspect_tape, parse_twse_industry_change, verify_report)
from scripts.audit_market_identity import resolve_on
from skills.verified_backtest_tool import offline_only
from scripts.audit_h4_odd_lot import audit as audit_h4
from skills.tpex_mth import inspect_mth_sample

CACHE = ROOT / '.cache/backtest-data-completion-20260925'
SEAL = ROOT / '.cache/board-only-supplement-verified-r2-20260925'
CURRENT_CASE_SETS = [
    ('corporate', ROOT / '.cache/backtest-corporate-completion-20260925/probe-v2'),
    ('sector', ROOT / '.cache/sector-accounts-20260925'),
]
TARGET = ROOT / 'artifacts/forward_simulation/backtest_data_completion_20260925.json'
CODE = (
    'skills/backtest_data_evidence.py', 'scripts/audit_backtest_data_completion.py',
    'scripts/prepare_backtest_data_evidence.py', 'tests/test_backtest_data_evidence.py',
    'scripts/prepare_backtest_board_ticks.py',
    'docs/backtest_data_completion_20260925.md', 'scripts/replay_contingent_day.py',
    'skills/contingent_replay.py', 'skills/intraday_limit_replay.py',
    'scripts/audit_market_identity.py', 'scripts/audit_identity_continuation.py',
    'skills/market_identity_overlay.py', 'scripts/crawl_revenue_announcements.py',
    'skills/ingest_quarterly_fundamental.py', 'skills/verified_backtest_tool.py',
    'scripts/research_intraday_limit.py', 'scripts/research_exit_scenarios.py',
    'skills/replay_market_feeds.py', 'app/finmind.py', 'app/finmind_cache.py',
    'app/rate_limiter.py', 'app/file_lock.py', 'app/config.py',
)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'


def build(case_sets=None, extra_tick_summaries=()):
    refs = {}
    def file(path, expected=None):
        path = ROOT / path
        return checked(path, digest(path) if expected is None else expected, refs, ROOT)
    def read(path, expected=None):
        return json.loads(file(path, expected).read_text())

    case_sets = case_sets or CURRENT_CASE_SETS
    # The caller's backtest source_context validates the 24,094-file transitive
    # account input chain. Here we validate the precise account bytes consumed.
    cases, case_sources = {}, {}
    for label, directory in case_sets:
        directory = Path(directory).resolve()
        manifest = read(directory / 'manifest.json')
        if directory == SEAL:
            offline = read(SEAL / 'offline.json')
            if (offline.get('all_cases_identical') is not True or offline.get('parent_controls_identical') is not True
                    or offline['manifest_sha256'] != digest(SEAL / 'manifest.json')):
                raise ValueError('Account manifest lacks bound offline verification')
        paths = sorted((directory / 'cases').glob('*.json'))
        if not paths:
            raise ValueError('No cases in the declared case directory')
        for path in paths:
            name = label + ':' + path.stem if label else path.stem
            if name in cases:
                raise ValueError('Duplicate case key')
            cases[name] = read(path, manifest['files_sha256'][str(path.relative_to(directory))])
            case_sources[name] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path))
    listing_path = ROOT / '.cache/listing-continuation-20260924/report.json'
    verification = read(listing_path.with_name('verification.json'))
    if verification.get('identical') is not True:
        raise ValueError('Listing identity audit was not reproduced')
    listing = read(listing_path, verification['report_sha256'])
    for name, value in listing['source_sha256'].items():
        file(name, value)
    indexed = defaultdict(list)
    for episode in listing['episodes']:
        indexed[episode['stock_id']].append(episode)

    old = ROOT / '.cache/intraday-limit-20260914-final'
    old_manifest = read(old / 'manifest.json')['files_sha256']
    new = read('.cache/contingent-ticks-20260914/summary.json')
    items = []
    for name, value in sorted(old_manifest.items()):
        if not name.startswith('ticks/') or not name.endswith('.parquet'):
            continue
        path = old / name
        sid, day = path.stem.split('-', 1)
        identity = resolve_on(indexed[sid], sid, day)
        if identity['status'] != 'identified':
            raise ValueError('Local tick lacks dated market identity: ' + path.name)
        items.append(dict(path=str(path.relative_to(ROOT)), sha256=value,
            metadata_sha256=old_manifest[name.replace('.parquet', '.json')],
            format='finmind_board', stock_id=sid, date=day, channel='board',
            market=identity['market'].upper()))
    old_count = len(items)
    for item in new['rows']:
        if item['completed']:
            items.append(dict(item, format='finmind_board', channel='board'))
    contingent_count = len(items) - old_count
    preparations = []
    for summary in extra_tick_summaries:
        summary = Path(summary).resolve()
        supplemental = read(summary)
        if supplemental.get('schema') != 'bounded_backtest_board_ticks_v1':
            raise ValueError('Unknown supplemental tick preparation receipt')
        identity = read(supplemental['identity_path'], supplemental['identity_sha256'])
        file(supplemental['plan_path'], supplemental['plan_sha256'])
        attempts = read(supplemental['attempts_path'], supplemental['attempts_sha256'])
        budget = read(summary.parent / 'ticks/budget.json')
        if (identity['maximum'] != supplemental['hard_maximum']
                or not 0 <= budget['reserved'] <= identity['maximum']
                or budget['reserved'] != supplemental['adapter_attempts_lifetime']):
            raise ValueError('Supplemental tick request budget differs from receipt')
        for name, value in identity['code_sha256'].items():
            file(name, value)
        planned = {(row['date'], row['stock_id'], row['market']) for row in identity['requests']}
        if len(planned) != supplemental['planned_stock_days']:
            raise ValueError('Supplemental request inventory changed')
        seen = set()
        for item in supplemental['rows']:
            key = (item['date'], item['stock_id'], item['market'])
            if key not in planned or key in seen:
                raise ValueError('Unknown or duplicate prepared tick')
            seen.add(key)
            if item['completed']:
                if attempts[item['stock_id']+'-'+item['date']]['status'] != 'success':
                    raise ValueError('Completed tick lacks a successful preparation attempt')
                items.append(dict(item, format='finmind_board', channel='board'))
        preparations.append({key:supplemental[key] for key in (
            'planned_stock_days', 'all_completed', 'adapter_attempts_lifetime', 'hard_maximum',
            'plan_path', 'plan_sha256', 'identity_path', 'identity_sha256')})
    catalog, duplicates = {}, 0
    for item in items:
        observed = inspect_tape(item, ROOT, refs)
        key = (observed['date'], observed['stock_id'], observed['channel'])
        if key in catalog:
            if catalog[key]['sha256'] != observed['sha256']:
                raise ValueError('Conflicting source versions for a single tape identity')
            duplicates += 1
        else:
            catalog[key] = observed

    official = {}
    for folder in ('official', 'official-execution'):
        directory = CACHE / folder
        capture = read(directory / 'manifest.json')
        for name, value in (capture['files'] | capture['derived']).items():
            file(directory / name, value['sha256'])
        official[folder] = capture
    event_html = (CACHE / 'official/twse-industry-announcement.html').read_text()
    if not all(s in event_html for s in ('112年05月22日', '112年7月3日', '1121802250')):
        raise ValueError('Industry attachment announcement identity changed')
    events = parse_twse_industry_change((CACHE / 'official/twse-industry-1121802250-1.txt').read_text())
    # An HTTP 200 block page is recorded as unavailable, never as data success.
    mops_text = (CACHE / 'official/mops-correction-query.html').read_text()
    mops_blocked = 'THIS PAGE CAN NOT BE ACCESSED' in mops_text
    members_meta = read('.cache/chain-flow-research/members.meta.json')
    members = pd.read_parquet(file('.cache/chain-flow-research/members.parquet', members_meta['sha256']))
    revenue = pd.read_parquet(file('artifacts/revenue_announcements/announcements.parquet'))
    first = revenue.announcement_date.min()
    quarterly = []
    for path in sorted((ROOT / '.cache/quarterly-observations').glob('*/*/manifest.json')):
        observed = read(path)
        for name, value in observed['files_sha256'].items():
            file(path.parent / name, value)
        quarterly.append(dict(stock_id=observed['stock_id'], observed_at=observed['observed_at'],
                              reporting_period_start=observed['start'],
                              path=str(path.relative_to(ROOT))))
    publication = dict(revenue_rows=len(revenue), revenue_stocks=int(revenue.stock_id.nunique()),
        revenue_first_observed_date=str(first.date()),
        revenue_last_observed_date=str(revenue.announcement_date.max().date()),
        revenue_first_batch_left_truncated_rows=int(revenue.announcement_date.eq(first).sum()),
        revenue_revision_observations=int(revenue.is_revision.sum()),
        revenue_timestamp_semantics='local_first_observed_date_not_official_publication_time',
        quarterly_observations=quarterly,
        quarterly_timestamp_semantics='provider_payload_observed_at_not_original_publication_time',
        complete_historical_versions=False,
        mops_official_history_query_status='blocked_by_official_security_page' if mops_blocked else 'not_validated',
        mops_history_rows_acquired=0)
    corporate_evidence = []
    for name in ('docs/backtest_corporate_completion_20260925.json', 'docs/sector_account_corporate_20260925.json'):
        document = read(name)
        for source, value in document['evidence_sha256'].items():
            file(source, value)
        events_by_date = []
        for event, record in sorted(document['overrides'].items()):
            fields = {key:record.get(key) for key in (
                'entitlement_announcement_date', 'delivery_announcement_date', 'listing_announcement_date',
                'pay_date', 'certificate_delivery_date', 'ordinary_share_available_date',
                'fractional_cash_pay_date', 'fractional_valuation_verified',
                'pending_only', 'pending_delivery_not_before', 'record_date',
                'ordinary_share_delivery_status')}
            fields.update(event=event, official_publication_clock_verified=False,
                          complete_historical_version_chain=False)
            events_by_date.append(fields)
        corporate_evidence.append(dict(path=name, source_files=len(document['evidence_sha256']),
            status=document['status'], events=events_by_date))
    publication['specific_corporate_event_evidence'] = corporate_evidence
    components = [
        dict(code='historical_universe_and_eligibility', status='blocked',
             unknown_listing_starts=listing['remaining_unknown_starts'],
             unresolved_stock_ids=sorted(r['stock_id'] for r in listing['unresolved']),
             unconfirmed_categories=listing['unconfirmed_categories'],
             current_date_discrepancies=listing['unresolved_current_date_discrepancies'],
             continuous_eligibility_proven=False),
        dict(code='historical_industry_membership', status='blocked', current_snapshot_rows=len(members),
             current_snapshot_stocks=int(members.stock_id.nunique()),
             current_snapshot_groups=int(members.industry.nunique()),
             current_snapshot_dates=sorted(members.date.astype(str).unique()),
             newly_verified_official_change_rows=len(events),
             official_industry_is_not_finmind_supply_chain=True),
        dict(code='official_announcement_and_revision_times', status='blocked',
             local_first_observation_available=True, official_clock_complete=False,
             historical_version_archive_complete=False),
    ]
    pit = dict(episodes=listing['episodes'], components=components)
    evidence = {name: case_evidence(name, value, catalog, pit) for name, value in cases.items()}
    totals = dict(case_count=len(evidence), complete_daily_cases=sum(row['case_completed_daily'] for row in evidence.values()))
    for label in ('ordinary', 'odd_lot'):
        missing = {(row['date'], row['stock_id']) for case in evidence.values() for row in case[label]['missing']}
        unverified = {(row['date'], row['stock_id']) for case in evidence.values() for row in case[label]['unverified']}
        required = missing | unverified
        totals[label] = dict(required_unique_sessions=len(required), missing_unique_sessions=len(missing),
            local_format_valid_unique_sessions=len(unverified), accepted_unique_sessions=0,
            required_first_date=min((key[0] for key in required), default=None),
            required_last_date=max((key[0] for key in required), default=None),
            required_stock_count=len({key[1] for key in required}))
    # Keep existing samples and guard findings visible, bound to their actual bytes.
    supporting = {}
    for name in (
        '.cache/readiness-completion-20260914/h4-sample-audit.json',
        '.cache/execution-sources-20260924-v3/mth-sample-inspection.json',
        '.cache/oddlot_scope_20260925/summary.json',
        '.cache/source-guard-audit-after-20260914.json',
    ):
        value = read(name)
        supporting[name] = dict(sha256=refs[name], keys=sorted(value))
    for name in (
        '.cache/readiness-completion-20260914/h4-sample.txt',
        '.cache/execution-sources-20260924-v3/tpex-mth-sample.txt',
        'scripts/audit_h4_odd_lot.py', 'skills/tpex_mth.py',
        'scripts/audit_oddlot_scope_20260925.py', 'docs/research_capacity_source_guard_20260914.md',
    ):
        file(name)
    h4_path = ROOT / '.cache/readiness-completion-20260914/h4-sample.txt'
    h4 = audit_h4(h4_path, 'legacy190')
    if h4 != read('.cache/readiness-completion-20260914/h4-sample-audit.json'):
        raise ValueError('H4 sample inspection cannot be reproduced')
    mth = inspect_mth_sample((ROOT / '.cache/execution-sources-20260924-v3/tpex-mth-sample.txt').read_bytes(),
                            format_id='tpex_mth_67_v1')
    if mth != read('.cache/execution-sources-20260924-v3/mth-sample-inspection.json'):
        raise ValueError('MTH sample inspection cannot be reproduced')
    # Pages contain prices and rules which can change; bind assertions to capture.
    facts = [
        dict(id='twse_board_trades_h2', path='official-execution/twse-h2.html',
             availability='paid_internal_use_2006_onward_excludes_latest_year', free_full_history=False,
             required_period_gap='2022-01-03 through 2026-09-09', acquisition='TWSE Data E-Shop H2 custom order; do not purchase automatically'),
        dict(id='twse_order_logs_h1', path='official-execution/twse-h1.html',
             availability='paid_internal_use_2006_onward_excludes_latest_year', free_full_history=False,
             acquisition='TWSE Data E-Shop H1; may aid queue reconstruction, separate from trade prints'),
        dict(id='twse_intraday_odd_h4', path='official/twse-h4.html',
             availability='paid_from_2020_11_01_with_recent_month_lag', free_full_history=False,
             acquisition='TWSE Data E-Shop H4; dated 190/201-byte versions; independently reconcile actual matches'),
        dict(id='tpex_mth', path='official/tpex-mth.html',
             availability='paid_from_2022_11_01_product_says_older_than_one_year', free_full_history=False,
             required_period_gap='2022-01-03 through 2022-10-31 predates advertised start; latest-year restriction also unresolved',
             acquisition='TPEx MTH custom order; provider confirmation needed for archive range and intended-use license'),
        dict(id='finmind_board_ticks', path='official/finmind-technical.html',
             availability='backer_or_sponsor_per_stock_day_2019_onward', free_full_history=False,
             acquisition='Existing shared FinMind adapter; authorized bounded preparation recorded separately; this audit is offline'),
        dict(id='finmind_revenue_create_time', path='official/finmind-fundamental.html',
             availability='ingestion_date_recording_from_2026_04_21_not_official_clock',
             supplies_original_announcement_versions=False),
        dict(id='twse_official_industry_change', path='official/twse-industry-announcement.html',
             availability='free_47_company_event_2023_07_03', full_history=False),
        dict(id='mops_correction_query', path='official/mops-correction-query.html',
             availability=publication['mops_official_history_query_status'], rows_acquired=0,
             proves_archive_impossible=False),
    ]
    for fact in facts:
        path = CACHE / fact['path']
        fact['source_sha256'] = digest(path)
        folder, name = fact['path'].split('/', 1)
        fact['url'] = official[folder]['files'][name]['requested_url']
    # Sanity-check the acquisition claims against captured page contents.
    for name, phrases in {
        'official-execution/twse-h2.html': ('95年1月1日', '不提供最近一年', '10,000'),
        'official-execution/twse-h1.html': ('95年1月1日', '不提供最近一年'),
        'official/twse-h4.html': ('2020-11-01', '1,500'),
        'official/tpex-mth.html': ('2022/11/01', '一年前'),
        'official/finmind-technical.html': ('TaiwanStockPriceTick', '2019-01-01'),
        'official/finmind-fundamental.html': ('create_time', '2026-04-21'),
    }.items():
        text = BeautifulSoup((CACHE / name).read_bytes(), 'html.parser').get_text(' ', strip=True)
        if not all(phrase in text for phrase in phrases):
            raise ValueError('Captured source does not support declared range: ' + name)
    report = dict(schema='backtest_data_completion_v1', as_of='2026-09-25',
        execution_mode='offline_evidence_audit_of_separately_prepared_sources',
        requested_start='2022-01-03', requested_end='2026-09-09',
        status='partial_evidence_completed_required_data_still_missing',
        live_qualified=False, strict_data_ready=False, network_requests=0, finmind_api_requests=0,
        preparation_public_document_requests=sum(v['attempts'] for v in official.values()),
        preparation_finmind_adapter_attempts=sum(row['adapter_attempts_lifetime'] for row in preparations),
        preparation_finmind_budget_total=sum(row['hard_maximum'] for row in preparations),
        prior_sandbox_dns_failure_attempts=1, paid_requests=0, database_mutations=0,
        account_source_scope='sealed_account_bytes_bound; full_transitive_price_chain_checked_by_backtest_source_context',
        catalog=dict(old_final_files=old_count, contingent_completed_files=contingent_count,
                     additional_completed_files=len(items)-old_count-contingent_count,
                     duplicate_stock_days=duplicates, unique_stock_days=len(catalog),
                     board_format_valid=len(catalog), odd_format_valid=0,
                     independently_authenticated_complete_sessions=0,
                     entries=[catalog[key] for key in sorted(catalog)]),
        cases=evidence, case_sources=case_sources, coverage_totals=totals,
        reproduction=dict(case_sets=[dict(label=label, directory=str(Path(directory).resolve().relative_to(ROOT)))
                                     for label, directory in case_sets],
                          extra_tick_summaries=[str(Path(path).resolve().relative_to(ROOT)) for path in extra_tick_summaries]),
        supplemental_tick_preparations=preparations,
        sample_verification=dict(h4=dict(records=h4['record_count'], actual_matches=h4['actual_match_records']),
                                 tpex_mth={k:mth[k] for k in ('raw_rows','paired_trades','by_trade_type',
                                     'historical_session_complete','execution_tape_accepted','source_authenticated')}),
        industry_change_events=events, publication_inventory=publication,
        official_source_findings=facts, supporting_prior_audits=supporting,
        required_tape_fields=['stock_id', 'market', 'date', 'channel', 'timezone', 'time_us',
            'price_cents', 'shares', 'record_type', 'source_sequence_or_documented_duplicate_policy',
            'raw_source_sha256', 'source_url', 'retrieved_at', 'format_version',
            'session_boundaries', 'session_complete_evidence', 'unit_evidence', 'daily_reconciliation'],
        required_pit_fields=['stock_id', 'market', 'security_category', 'valid_from', 'valid_to_exclusive',
            'classification_system', 'industry_id', 'membership_valid_from', 'membership_valid_to_exclusive',
            'official_published_at_with_timezone', 'version_available_at_with_timezone', 'version_id',
            'supersedes_version_id', 'original_payload_sha256', 'announcement_source_sha256', 'observed_at'],
        required_pit_period='Before first lookback/candidate cut through 2026-09-09; price input starts 2021-01-04',
        limitations=[
            'Known daily-model order paths only; sequence replay may request additional stock-days.',
            'A complete trade-print file still does not prove own queue position or execution.',
            'Industry attachment proves one official event, not FinMind supply-chain history or all intervening changes.',
            'Local publication observation ledgers cannot recover versions already overwritten before observation.',
            'Forward source_guard freshness is not evidence of historical announcement-time availability.',
            'No claim that all free official archives are impossible; checked public endpoints do not supply the full requested history.',
        ], input_sha256=refs, code_sha256={name:digest(ROOT / name) for name in CODE})
    return report


def run(output=TARGET, verify=False, case_sets=None, extra_tick_summaries=()):
    started = time.monotonic()
    with offline_only():
        output = Path(output).resolve()
        if not output.is_relative_to(ROOT) or output.suffix != '.json':
            raise ValueError('Use a JSON output within the repository')
        if verify:
            old = verify_report(output, ROOT)
            recipe = old['reproduction']
            if not case_sets:
                case_sets = [(row['label'], ROOT / row['directory']) for row in recipe['case_sets']]
            if not extra_tick_summaries:
                extra_tick_summaries = [ROOT / path for path in recipe['extra_tick_summaries']]
        report = build(case_sets, extra_tick_summaries)
        if verify:
            if old != report:
                raise ValueError('Recomputed data evidence differs from sealed report')
        else:
            if output.exists() or output.with_suffix('.sha256').exists():
                raise ValueError('Data evidence report is immutable; choose a new output')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded(report))
            output.with_suffix('.sha256').write_text(digest(output) + '\n')
    print(json.dumps(dict(output=str(output.relative_to(ROOT)), seconds=round(time.monotonic()-started, 3),
        verified=verify, cases=len(report['cases']), catalog_sessions=report['catalog']['unique_stock_days'],
        new_industry_events=len(report['industry_change_events']), strict_data_ready=False,
        network_requests=0, finmind_api_requests=0), ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=TARGET)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--case-set', action='append', default=[], metavar='LABEL=DIRECTORY')
    parser.add_argument('--extra-tick-summary', action='append', default=[], type=Path)
    args = parser.parse_args()
    case_sets = []
    for value in args.case_set:
        label, separator, path = value.partition('=')
        if not separator or not label or not path:
            parser.error('--case-set must be LABEL=DIRECTORY')
        case_sets.append((label, Path(path)))
    run(args.output, args.verify, case_sets, args.extra_tick_summary)
