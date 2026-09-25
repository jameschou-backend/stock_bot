#!/usr/bin/env python3
"""Replay immutable identity evidence, dated exclusions, and affected frozen inputs offline."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from bs4 import BeautifulSoup
from scripts.audit_identity_continuation import checked_file, compact
from scripts.audit_historical_universe_followup import verify_bound_source
from scripts.research_exit_scenarios import read, write, sha
from skills.historical_universe_completion import (
    apply_completion, boundary_examples, parse_tpex_halts, parse_twse_halts,
    resolve_completion, roc_day, validate_exclusions)

RECIPE = 'docs/evidence_historical_universe_completion_20260925.json'
TARGET = ROOT / '.cache/historical-universe-followup-v2-20260925/report-final.json'


def source(row, refs):
    """Re-extract primary evidence and preserve URL, receipt, raw bytes, and page binding."""
    path = checked_file(row['source_path'], row['source_sha256'], refs)
    meta = read(checked_file(row['source_meta_path'], row['source_meta_sha256'], refs))
    if (urlparse(row['source_url']).hostname not in ('www.twse.com.tw', 'doc.twse.com.tw', 'www.tpex.org.tw', 'mopsfin.twse.com.tw')
            or meta.get('status') != 200 or meta['sha256'] != row['source_sha256']
            or meta['url'] != row['source_url']):
        raise ValueError('Primary origin, receipt, URL, or bytes differ')
    proof = row['verification']
    leaf = {k: v for k, v in row.items() if k not in ('corroborating_evidence', 'category_evidence', 'correction_evidence')}
    if proof['kind'] == 'json':
        result = read(path)
        if not isinstance(result, dict):
            raise ValueError('Expected an official JSON object')
        if 'row_binding' in proof:
            binding = proof['row_binding']
            table = result['tables'][0]
            matching = [r for r in table['data'] if r[0] == row['stock_id']]
            if (result.get('stat') != 'ok' or len(matching) != 1 or matching[0] != binding['row']
                    or table['fields'] != binding['fields'] or result['date'] != binding['date']):
                raise ValueError('Official status snapshot company/date row differs')
    elif proof['kind'] == 'doc':
        try:
            text = subprocess.run(['textutil', '-convert', 'txt', '-stdout', str(path)],
                                  check=True, capture_output=True).stdout.decode('utf-8')
        except FileNotFoundError as exc:
            raise RuntimeError('This official DOC archive requires macOS textutil') from exc
        if not proof.get('snippets') or any(compact(s) not in compact(text) for s in proof['snippets']):
            raise ValueError('Official DOC listing row differs')
        result = text
    else:
        verify_bound_source(leaf, refs)
        result = BeautifulSoup(path.read_bytes(), 'html.parser').get_text(' ', strip=True) if proof['kind'] == 'html' else None
    for key in ('corroborating_evidence', 'category_evidence', 'correction_evidence'):
        for child in row.get(key, []):
            source(dict(stock_id=row.get('stock_id', '0000'), **child), refs)
    return result


def all_evidence_text(row):
    result = []
    if row['verification']['kind'] == 'html':
        result.append(compact(BeautifulSoup((ROOT / row['source_path']).read_bytes(), 'html.parser').get_text(' ', strip=True)))
    for key in ('corroborating_evidence', 'category_evidence', 'correction_evidence'):
        for child in row.get(key, []):
            result.extend(all_evidence_text(child))
    return result


def dated_interval(row, base):
    """Bind reviewed boundaries to literal primary dates or the sealed legal delisting date."""
    texts = all_evidence_text(row)
    dates = {roc_day(m) for text in texts for m in re.findall(r'(?:民國)?\d{2,3}年\d{1,2}月\d{1,2}日', text)}
    if row['start'] not in dates:
        raise ValueError('Exclusion start absent from primary text')
    legal_ends = {e['end'] for e in base['episodes'] if e['stock_id'] == row['stock_id'] and e['market'] == row['market']}
    if row['end'] is not None and row['end'] not in dates and row['end'] not in legal_ends:
        raise ValueError('Exclusion end lacks primary resumption or legal termination evidence')
    if row['end'] is None:
        snapshots = [child for child in row.get('corroborating_evidence', [])
                     if child['verification'].get('row_binding')]
        if len(snapshots) != 1:
            raise ValueError('Open exclusion requires a bound cutoff-date stopped-status snapshot')
        binding = snapshots[0]['verification']['row_binding']
        if binding['date'] != '20260909' or binding['row'][binding['fields'].index('停止交易')] != 'Ｙ':
            raise ValueError('Open exclusion is not confirmed stopped at the cutoff')
    return row


def audit_records(report, records):
    """Compare every frozen observation; do not infer unchanged strategy from zero fills."""
    issues = []
    indexed = defaultdict(lambda: dict(episodes=[], trading_exclusions=[], coverage_end=report['coverage_end'],
                                      coverage_start=report.get('coverage_start', '0001-01-01')))
    for episode in report['episodes']:
        indexed[episode['stock_id']]['episodes'].append(episode)
    for exclusion in report['trading_exclusions']:
        indexed[exclusion['stock_id']]['trading_exclusions'].append(exclusion)
    for index, row in enumerate(records):
        sid, stamp = str(row['stock_id']), str(row['date'])[:10]
        after = resolve_completion(indexed[sid], sid, stamp)
        if after['status'] != 'identified' or after['category'] not in ('股票', 'ETF'):
            issues.append(dict(index=index, stock_id=sid, date=stamp, status=after['status'],
                market=after['market'], category=after['category'],
                issue_type='security_category' if after['status'] == 'identified' else after['status'],
                event_id=row.get('event_id'), reason=row.get('reason')))
    return dict(rows=len(records), issue_rows=len(issues), issues=issues, passed=not issues)


def impact(report, recipe, refs, base):
    frames = {}
    for key in ('calendar', 'companies'):
        frames[key] = pd.read_parquet(checked_file(recipe[key + '_path'], recipe[key + '_sha256'], refs))
    frame = frames['calendar']
    if 'date' not in frame.columns:
        raise ValueError('Frozen wide matrix must have an explicit date column')
    stamps = pd.to_datetime(frame['date'])
    calendar = pd.DatetimeIndex(stamps).sort_values().unique()
    prior = calendar[calendar < pd.Timestamp('2022-01-03')]
    if len(prior) < 126:
        raise ValueError('Fewer than 126 warmup trading dates')
    warmup = prior[-126].date().isoformat()
    companies = frames['companies'].copy()
    if companies.stock_id.duplicated().any():
        raise ValueError('Duplicated company cohort')
    listed = {str(r.stock_id): str(r.listed_date)[:10] for r in companies.itertuples()}
    signals = read(checked_file(recipe['signals_path'], recipe['signals_sha256'], refs))
    if len(signals['entries']) != 458:
        raise ValueError('Frozen 458-candidate scope differs')
    intervals = []
    for row in report['trading_exclusions']:
        sid = row['stock_id']
        start, end = max(row['start'], warmup), min(row['end'] or '2026-09-10', '2026-09-10')
        dates = [d.date().isoformat() for d in calendar if start <= d.date().isoformat() < end]
        expected = [d for d in dates if sid in listed and listed[sid] <= d]
        positive = 0
        if sid in frame.columns and dates:
            mask = stamps.dt.strftime('%Y-%m-%d').isin(dates)
            positive = int(pd.to_numeric(frame.loc[mask, sid], errors='coerce').gt(0).sum())
        intervals.append(dict(stock_id=sid, market=row['market'], kind=row['kind'],
            start=row['start'], end=row['end'], overlaps_warmup_or_backtest=bool(dates),
            affected_trading_dates=len(dates), in_frozen_company_cohort=sid in listed,
            expected_denominator_dates=len(expected), positive_raw_close_dates=positive,
            signal_change_proven=False, requires_selector_replay=bool(expected)))
    candidate_rows = [dict(stock_id=sid, date=e['signal_date'], event_id=e['event_id'])
                      for e in signals['entries'] for sid in e['members']]
    peers = [dict(stock_id=sid, date=e[field], event_id=e['event_id'])
             for e in signals['entries'] for field in ('signal_date', 'group_cutoff_date')
             for sid in e.get('group_members', [])]
    cases = read(checked_file(recipe['case_dependency_path'], recipe['case_dependency_sha256'], refs))
    before_report = dict(episodes=base['episodes'], coverage_end=report['coverage_end'], coverage_start=report['coverage_start'],
                         trading_exclusions=[dict(r, kind='trading_suspension') for r in base['trading_exclusions']])
    case_rows = {}
    for name, digest in cases['input_sha256'].items():
        if '/cases/' not in name:
            continue
        case = read(checked_file(name, digest, refs))
        case_rows[name] = {}
        for kind in ('orders', 'trades', 'holdings'):
            before = audit_records(before_report, case['account'][kind])
            after = audit_records(report, case['account'][kind])
            previous = {(i['index'], i['issue_type'], i['market'], i['category']) for i in before['issues']}
            new = [i for i in after['issues'] if (i['index'], i['issue_type'], i['market'], i['category']) not in previous]
            case_rows[name][kind] = dict(after, before_issue_rows=before['issue_rows'], new_issue_rows=len(new), new_issues=new,
                interpretation='Suspension of a held security does not invalidate ownership; orders and price availability require review.' if kind == 'holdings' else None)
    if len(case_rows) != 20:
        raise ValueError('Expected all 20 frozen corporate/sector accounts')
    date_impacts = []
    for row in report['date_corrections']:
        lower, upper = sorted((row['start'], row['replaces_snapshot_start']))
        date_impacts.append(dict(stock_id=row['stock_id'], old_start=row['replaces_snapshot_start'],
            corrected_start=row['start'], changed_identity_interval=dict(start=lower, end=upper),
            overlaps_warmup_or_backtest=upper > warmup and lower <= '2026-09-09',
            frozen_selector_listing_date=listed.get(row['stock_id']),
            frozen_selector_already_used_corrected_date=listed.get(row['stock_id']) == row['start']))
    return dict(warmup_sessions=126, warmup_start=warmup, backtest_start='2022-01-03', backtest_end='2026-09-09',
        date_corrections=date_impacts, exclusion_intervals=intervals,
        candidate_audit=audit_records(report, candidate_rows), selected_peer_audit=audit_records(report, peers),
        case_audits=case_rows, expected_denominator_affected_stock_ids=sorted({r['stock_id'] for r in intervals if r['requires_selector_replay']}),
        calculations_require_replay=any(r['requires_selector_replay'] for r in intervals),
        signal_changes_proven=False, returns_recomputed=False)


def universe_scope(report, recipe, refs):
    """Count the observed market-day union without calling sparse dates a full universe."""
    companies = pd.read_parquet(ROOT / recipe['companies_path'])
    cohort = set(companies.stock_id.astype(str))
    active = [e for e in report['episodes'] if e['category'] == '股票' and e['start'] is not None
              and e['start'] <= report['coverage_end'] and (e['end'] is None or e['end'] > '2022-01-03')]
    omitted = [e for e in active if e['stock_id'] not in cohort]
    quotes, stamps, unmatched = set(), set(), defaultdict(list)
    ep_index = defaultdict(list)
    for e in report['episodes']:
        ep_index[e['stock_id']].append(e)
    from scripts.audit_market_identity import resolve_on
    for entry in recipe['market_day_sources']:
        path = checked_file(entry['path'], entry['sha256'], refs)
        meta = read(checked_file(entry['meta_path'], entry['meta_sha256'], refs))
        payload = read(path)
        if (meta.get('http_status') != 200 or meta['sha256'] != entry['sha256']
                or meta['path'] != entry['path'] or meta['date'] != entry['date']
                or urlparse(meta['url']).hostname != 'www.tpex.org.tw' or meta['kind'] != 'no1430'):
            raise ValueError('Official market-day union provenance differs')
        table = payload['tables'][0]
        if (payload['stat'] != 'ok' or table['title'] != '上櫃股票每日收盤行情(不含定價)'
                or len(table['data']) != table['totalCount'] or roc_day(table['date']) != entry['date']
                or table['fields'][:2] != ['代號', '名稱']):
            raise ValueError('Incomplete or wrong-day official market table')
        stamps.add(entry['date'])
        for row in table['data']:
            sid = row[0]
            if not re.fullmatch(r'[0-9]{4}', sid):
                continue
            quotes.add(sid)
            known = resolve_on(ep_index[sid], sid, entry['date'])
            if known['status'] != 'identified' or known['market'] != 'TPEx':
                unmatched[sid].append(entry['date'])
    frame = pd.read_parquet(ROOT / recipe['calendar_path'])
    days = [str(v)[:10] for v in frame['date'] if '2022-01-03' <= str(v)[:10] <= report['coverage_end']]
    return dict(known_ordinary_episodes_over_backtest=len(active), frozen_company_cohort_rows=len(cohort),
        ordinary_episodes_missing_from_frozen_cohort=len(omitted), omitted_cohort_episodes=omitted,
        missing_from_cohort_is_not_missing_from_local_database=True,
        local_price_inventory_owner='separate immutable omitted-company price supplement; verify it independently',
        local_price_supplement_reference='.cache/historical-cohort-supplement-20260925/verified-snapshot/manifest.json',
        official_observed_union=dict(market='TPEx', observed_days=len(stamps), calendar_days=len(days),
            unobserved_calendar_days=sorted(set(days) - stamps), observed_stock_ids=sorted(quotes),
            current_and_terminated_identity_unmatched={sid: dict(first_date=min(dates), last_date=max(dates), rows=len(dates))
                                                    for sid, dates in sorted(unmatched.items())},
            scope='existing execution-demand dates; no assumption that these span every historical market day'),
        remaining_requirements=[
            dict(code='twse_corporate_suspension_notice_bodies', status='external_http_307_428', rows=len(report['external_blocked_notices']),
                 evidence='external_blocked_notices contains exact URLs, timestamps, status and raw hashes'),
            dict(code='all_day_market_roster_and_lifecycle', status='not_proven_by_sparse_quote_union',
                 evidence='Known 32 discrepancies are repaired; sparse execution-demand market days cannot certify every historical constituent or board transfer.'),
            dict(code='omitted_company_selector_replay', status='local_data_work', rows=len(omitted),
                 evidence='Existing local DB prices are archived separately; this overlay does not silently append them or reuse old profits.')],
        complete_historical_universe=False)


def build():
    refs = {}
    recipe = read(checked_file(RECIPE, sha(ROOT / RECIPE), refs))
    if recipe['schema'] != 'historical_universe_completion_recipe_v2':
        raise ValueError('Unsupported identity completion recipe')
    base = read(checked_file(recipe['base_path'], recipe['base_sha256'], refs))
    for name, digest in base['source_sha256'].items():
        checked_file(name, digest, refs)
    dates, categories = recipe['date_corrections'], recipe['category_corrections']
    required_dates = {r['stock_id'] for r in base['current_isin_vs_company_basic'] if not r['individually_resolved']}
    if {r['stock_id'] for r in dates} != required_dates or len(dates) != 14 or len(categories) != 18:
        raise ValueError('Known 32-discrepancy scope changed')
    for row in dates + categories:
        source(row, refs)
    episodes = apply_completion(base, dates, categories)
    exclusions = [dict(r, kind='trading_suspension') for r in base['trading_exclusions']]
    for row in recipe['manual_exclusions']:
        source(row, refs)
        exclusions.append(dated_interval(row, base))
    tpex = []
    for row in recipe['halt_sources']:
        payload = source(row, refs)
        query = parse_qs(urlparse(row['source_url']).query)
        if row['market'] == 'TWSE':
            end = f"{row['year']}1231" if row['year'] < int(recipe['coverage_end'][:4]) else recipe['coverage_end'].replace('-', '')
            params = payload.get('params', {})
            if (query.get('startDate') != [f"{row['year']}0101"] or query.get('endDate') != [end]
                    or params.get('startDate') != query['startDate'][0] or params.get('endDate') != end):
                raise ValueError('TWSE halt source URL year differs')
            excluded_ids = {e['stock_id'] for e in episodes if e['category'] == '臺灣存託憑證(TDR)'}
            exclusions.extend(parse_twse_halts(payload, row['source_path'], recipe['coverage_end'], row['year'], excluded_ids))
        else:
            if query.get('date') != [str(row['year'])] or payload.get('date') != str(row['year']) or query.get('cate') != ['1']:
                raise ValueError('TPEx halt query year/market differs')
            tpex.append((row['source_path'], payload))
    exclusions.extend(parse_tpex_halts(tpex, recipe['coverage_end']))
    # Identical intervals can be independently corroborated by financial and halt notices.
    unique = {}
    for row in exclusions:
        key = (row['stock_id'], row['market'], row['start'], row['end'], row['kind'])
        if key not in unique:
            unique[key] = deepcopy(row)
    exclusions = validate_exclusions(sorted(unique.values(), key=lambda r: (r['stock_id'], r['start'], r['end'] or '9999')))
    catalogs = []
    for row in recipe['notice_catalogs']:
        payload = source(row, refs)
        if 'tables' in payload:
            table = payload['tables'][0]
            if payload['stat'] != 'ok' or len(table['data']) != table['totalCount']:
                raise ValueError('Incomplete TPEx event catalog')
            count = table['totalCount']
        else:
            if payload['stat'] != 'ok' or len(payload['data']) != payload['total']:
                raise ValueError('Incomplete TWSE event catalog')
            count = payload['total']
        catalogs.append(dict(source_path=row['source_path'], rows=count, query_url=row['source_url']))
    blocked = []
    for row in recipe['external_blocked_notices']:
        raw = checked_file(row['source_path'], row['source_sha256'], refs)
        meta = read(checked_file(row['source_meta_path'], row['source_meta_sha256'], refs))
        if meta['status'] != row['status'] or meta['sha256'] != sha(raw) or meta['url'] != row['url'] or row['status'] not in (307, 428):
            raise ValueError('External blocker receipt differs')
        blocked.append(deepcopy(row))
    report = dict(schema='historical_universe_completion_v2', base_path=recipe['base_path'],
        coverage_start=recipe['coverage_start'], coverage_end=recipe['coverage_end'],
        known_discrepancies_resolved=len(dates) + len(categories),
        previous_unknown_starts=base['remaining_unknown_starts'], remaining_unknown_starts=sum(e['start'] is None for e in episodes),
        previous_unconfirmed_categories=base['unconfirmed_categories'], unconfirmed_categories=sum(e['category'] == 'unconfirmed' for e in episodes),
        previous_unresolved_current_date_discrepancies=base['unresolved_current_date_discrepancies'], unresolved_current_date_discrepancies=0,
        isin_semantic_reason_unexplained_stock_ids=[r['stock_id'] for r in dates if not r['isin_event_reason_verified']],
        date_corrections=dates, category_corrections=categories, episodes=episodes, trading_exclusions=exclusions,
        exclusion_counts_by_kind=dict(Counter(r['kind'] for r in exclusions)),
        notice_catalogs=catalogs, external_blocked_notices=blocked,
        notice_intervals_without_verified_end=recipe['notice_intervals_without_verified_end'],
        notice_intervals_requiring_review=recipe['notice_intervals_requiring_review'],
        complete_historical_universe=False, continuous_eligibility_proven=False,
        historical_membership_basis='current_official_snapshot_plus_archived_termination_and_original_listing_evidence',
        publication_time_archive_complete=False, performance_recomputed=False, database_mutations=0,
        live_qualified=False, finmind_requests=0, verification_network_requests=0)
    report['impact'] = impact(report, recipe, refs, base)
    report['full_market_scope'] = universe_scope(report, recipe, refs)
    report['boundary_examples'] = boundary_examples(report, recipe['manual_exclusions'][:6])
    for name in ('scripts/audit_historical_universe_completion.py', 'skills/historical_universe_completion.py'):
        checked_file(name, sha(ROOT / name), refs)
    report['source_sha256'] = refs
    return report


def verify_report(path=TARGET):
    path = Path(path)
    if sha(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Identity completion report hash differs')
    report = read(path)
    for name, digest in report['source_sha256'].items():
        checked_file(name, digest, {})
    if report != build():
        raise ValueError('Identity completion report differs from offline reconstruction')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output:
        if args.output.exists():
            raise ValueError('Choose a new immutable report path')
        report = build()
        write(args.output, report)
        args.output.with_suffix('.sha256').write_text(sha(args.output) + '\n')
    else:
        report = verify_report()
    print(json.dumps({k: report[k] for k in ('schema', 'known_discrepancies_resolved', 'remaining_unknown_starts',
        'unconfirmed_categories', 'unresolved_current_date_discrepancies', 'exclusion_counts_by_kind', 'complete_historical_universe')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
