#!/usr/bin/env python3
"""Offline verification of a new identity overlay; leaves sealed reports and DB untouched."""
import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
import subprocess
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bs4 import BeautifulSoup
from scripts.audit_identity_continuation import checked_file, compact
from scripts.research_exit_scenarios import read, sha, write
from skills.historical_universe_followup import apply_followup, parse_industry_changes, resolve_followup
from scripts.audit_market_identity import resolve_on

RECIPE = 'docs/evidence_historical_universe_followup_20260925.json'
TARGET = ROOT / '.cache/historical-universe-followup-20260925/report-final.json'


def verify_bound_source(row, refs):
    """Require a reviewed primary host and matching archived response identity."""
    host = urlparse(row['source_url']).hostname
    if host not in ('www.twse.com.tw', 'doc.twse.com.tw', 'www.tpex.org.tw',
                    'mopsfin.twse.com.tw', 'en.hitachi-yungtay.com.tw'):
        raise ValueError('Unreviewed primary source host')
    meta = read(checked_file(row['source_meta_path'], row['source_meta_sha256'], refs))
    if (meta.get('url') != row['source_url'] or meta.get('status') != 200
            or meta.get('sha256') != row['source_sha256']):
        raise ValueError('Primary source URL differs from archived response')
    path = checked_file(row['source_path'], row['source_sha256'], refs)
    proof = row['verification']
    if proof.get('visual_only'):
        raise ValueError('This overlay requires machine-readable primary text')
    if 'render_path' in proof:
        checked_file(proof['render_path'], proof['render_sha256'], refs)
    kind = proof['kind']
    if kind == 'pdf':
        page = proof.get('page')
        if not isinstance(page, int) or page < 1 or not path.read_bytes().startswith(b'%PDF'):
            raise ValueError('Primary PDF page or format differs')
        try:
            text = subprocess.run(['pdftotext', '-f', str(page), '-l', str(page), '-layout',
                                   str(path), '-'], check=True, capture_output=True).stdout.decode('utf-8')
        except FileNotFoundError as exc:
            raise RuntimeError('Install Poppler (pdftotext) to verify primary PDF evidence') from exc
    elif kind == 'html':
        text = BeautifulSoup(path.read_bytes(), 'html.parser').get_text(' ', strip=True)
    elif kind == 'csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            selected = [r for r in csv.DictReader(stream) if r.get('公司代號') == row['stock_id']]
        values = proof['values']
        if (proof['key_column'] != '公司代號' or proof['key_value'] != row['stock_id']
                or len(selected) != 1 or not values or '上櫃日期' not in values
                or any(selected[0].get(k) != v for k, v in values.items())
                or values['上櫃日期'] != row['start'].replace('-', '')):
            raise ValueError('Official CSV company/date row differs')
        text = None
    else:
        raise ValueError('Unsupported primary source format')
    if text is not None:
        snippets = proof.get('snippets', [])
        if not snippets or any(not compact(s) or compact(s) not in compact(text) for s in snippets):
            raise ValueError('Reviewed text not found on the exact primary page')
    result = 'primary_csv_row_verified' if kind == 'csv' else 'primary_text_reextracted'
    for name in ('category_evidence', 'correction_evidence', 'corroborating_evidence'):
        children = row.get(name, [])
        if isinstance(children, dict):
            children = [children]
        for child in children:
            verify_bound_source(dict(stock_id=row.get('stock_id', '0000'), **child), refs)
    return result


def build():
    refs = {}
    recipe = read(checked_file(RECIPE, sha(ROOT / RECIPE), refs))
    if recipe['schema'] != 'historical_universe_followup_recipe_v1':
        raise ValueError('Unsupported followup recipe')
    base = read(checked_file(recipe['base_path'], recipe['base_sha256'], refs))
    for path, digest in base['source_sha256'].items():
        checked_file(path, digest, refs)
    rows = deepcopy(recipe['listing_evidence'])
    for row in rows:
        row['verification_result'] = verify_bound_source(row, refs)
        if row.get('announcement_available_at') is not None or row.get('continuous_eligibility_proven') is not False:
            raise ValueError('Retrospective listing evidence cannot certify PIT availability')
    for row in recipe['suspensions']:
        verify_bound_source(row, refs)
    episodes, exclusions = apply_followup(base, rows, recipe['suspensions'])
    industries = []
    for source in recipe['industry_sources']:
        verify_bound_source(dict(stock_id='0000', **source), refs)
        text = BeautifulSoup((ROOT / source['source_path']).read_bytes(), 'html.parser').get_text(' ', strip=True)
        parsed = parse_industry_changes(text, year=source['year'], expected_count=source['expected_count'])
        year = source['year']
        for row in parsed:
            industries.append(dict(row, market='TPEx',
                effective_date=f'{year}-06-{1 if year == 2026 else 2:02d}',
                publication_date=f'{year}-05-{19 if year == 2026 else 21:02d}',
                official_publication_time=None, prior_interval_start=None, next_change_date=None,
                source_path=source['source_path'], source_sha256=source['source_sha256'],
                source_url=source['source_url'], classification_system='TPEx_official_single_industry',
                finmind_supply_chain_mapping_proven=False))
    corrections = {r['stock_id']: r for r in rows if r.get('replaces_snapshot_start')}
    discrepancies = deepcopy(base['current_isin_vs_company_basic'])
    for discrepancy in discrepancies:
        sid = discrepancy['stock_id']
        if sid in corrections:
            row = corrections[sid]
            if (discrepancy['snapshot_start'] != row['replaces_snapshot_start']
                    or discrepancy['company_basic_listing_date'] != row['start']):
                raise ValueError('Date correction differs from the exact prior discrepancy')
            event = row['correction_evidence']
            matches = [i for i in industries if i['stock_id'] == sid and i['effective_date'] == event['event_date']]
            if len(matches) != 1:
                raise ValueError('Listing correction lacks the actual industry event')
            discrepancy.update(individually_resolved=True, event_type='industry_reclassification')
    for path in ('scripts/audit_historical_universe_followup.py', 'skills/historical_universe_followup.py',
                 'scripts/audit_identity_continuation.py', 'skills/market_identity_overlay.py',
                 'scripts/audit_market_identity.py', 'scripts/research_exit_scenarios.py'):
        checked_file(path, sha(ROOT / path), refs)
    unknown = [r for r in episodes if r['start'] is None]
    categories = [r for r in episodes if r['category'] == 'unconfirmed']
    lookup = dict(episodes=episodes, trading_exclusions=exclusions)
    examples = []
    for sid, stamp in [('1507', '1989-11-08'), ('1507', '1989-11-09'),
                       ('1507', '2022-04-13'), ('1507', '2022-04-14'),
                       ('1507', '2022-04-20'), ('1507', '2022-04-21'),
                       ('2358', '1996-12-17'), ('2358', '1996-12-18')
                       ] + [(sid, '2022-01-03') for sid in sorted(corrections)]:
        examples.append(dict(stock_id=sid, date=stamp,
            before=resolve_on(base['episodes'], sid, stamp), after=resolve_followup(lookup, sid, stamp)))
    return dict(schema='historical_universe_followup_v1', base_path=recipe['base_path'],
        previous_unknown_starts=base['remaining_unknown_starts'], remaining_unknown_starts=len(unknown),
        previous_unconfirmed_categories=base['unconfirmed_categories'], unconfirmed_categories=len(categories),
        previous_unresolved_current_date_discrepancies=base['unresolved_current_date_discrepancies'],
        unresolved_current_date_discrepancies=sum(not r['individually_resolved'] for r in discrepancies),
        listing_evidence=rows, episodes=episodes, trading_exclusions=exclusions,
        current_isin_vs_company_basic=discrepancies, unresolved=unknown,
        before_after_examples=examples,
        unconfirmed_category_rows=categories, official_industry_events=industries,
        complete_historical_universe=False, continuous_eligibility_proven=False,
        publication_time_archive_complete=False, finmind_membership_history_complete=False,
        performance_recomputed=False, database_mutations=0, live_qualified=False,
        source_acquisition_http_requests=recipe['new_http_requests'], finmind_requests=0,
        verification_network_requests=0, source_sha256=refs)


def verify_report(path=TARGET):
    path = Path(path)
    expected = path.with_suffix('.sha256').read_text().strip()
    if sha(path) != expected:
        raise ValueError('Followup report hash differs')
    result = read(path)
    for name, digest in result['source_sha256'].items():
        checked_file(name, digest, {})
    if result != build():
        raise ValueError('Followup replay differs from published report')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=TARGET)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        result = verify_report(args.output)
    else:
        if args.output.exists():
            raise ValueError('Choose a new immutable report path')
        result = build()
        write(args.output, result)
        args.output.with_suffix('.sha256').write_text(sha(args.output) + '\n')
    print(json.dumps({k: v for k, v in result.items() if isinstance(v, (int, bool))}, sort_keys=True))
