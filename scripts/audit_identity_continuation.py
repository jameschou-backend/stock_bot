#!/usr/bin/env python3
"""Verify primary listing additions and cross-check frozen accounts without rewriting them."""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bs4 import BeautifulSoup
import pandas as pd
from scripts.research_exit_scenarios import read, write, sha
from skills.market_identity_overlay import apply_starts, audit_rows, day

BASE = '.cache/readiness-completion-20260914/identity-v2/report.json'
PRIOR = '.cache/listing-sources-20260924/report.json'
RESEARCH = '.cache/conservative-diversification-20260924'
ADDITIONS = ('docs/evidence_twse_listing_continuation_20260924.json',
             'docs/evidence_tpex_listing_continuation_20260924.json')
CORRECTIONS = 'docs/evidence_isin_date_corrections_20260924.json'


def checked_file(name, digest, refs):
    path = (ROOT/name).resolve()
    if not path.is_relative_to(ROOT) or sha(path) != digest:
        raise ValueError('Evidence missing or changed: '+str(name))
    refs[str(path.relative_to(ROOT))] = digest
    return path


def compact(text):
    return ''.join(unicodedata.normalize('NFKC', text).split())


def verify_primary(row, refs):
    """Hash plus exact text on the reviewed page, or explicitly human visual evidence."""
    path = checked_file(row['source_path'], row['source_sha256'], refs)
    proof = row['verification']
    if not row['source_url'].startswith(('https://', 'http://')):
        raise ValueError('Primary evidence must retain its source URL')
    for prefix in ('source_metadata', 'source_meta'):
        if prefix+'_path' in row:
            meta = read(checked_file(row[prefix+'_path'],row[prefix+'_sha256'],refs))
            if meta.get('status') != 200 or meta['sha256'] != row['source_sha256']:
                raise ValueError('Primary source response metadata differs')
    if 'render_path' in proof:
        checked_file(proof['render_path'],proof['render_sha256'],refs)
    for extra in ('category_evidence','correction_evidence','corroborating_evidence',
                  'conflicting_retrospective_source'):
        if extra in row:
            children=row[extra] if isinstance(row[extra],list) else [row[extra]]
            if not children:
                raise ValueError('Empty supporting source list')
            for child in children:
                verify_primary(dict(stock_id=row['stock_id'],**child),refs)
    if 'catalog' in row:
        catalog=row['catalog']
        html=checked_file(catalog['path'],catalog['sha256'],refs).read_text()
        if catalog['download_path'] not in html or not row['source_url'].endswith(catalog['download_path']):
            raise ValueError('Listing table is not bound to the reviewed catalog')
    if proof.get('visual_only'):
        if proof['kind'] != 'pdf' or not isinstance(proof.get('page'), int) or proof['page'] < 1:
            raise ValueError('Visual evidence requires an exact PDF page')
        checked_file(proof['render_path'], proof['render_sha256'], refs)
        if not proof.get('snippets'):
            raise ValueError('Visual evidence requires reviewed transcription')
        # Transcription is intentionally not labelled as an automatically matched PDF string.
        return 'visual_review_with_hashed_page'
    kind = proof['kind']
    if kind in ('pdf', 'doc'):
        command = (['pdftotext','-layout',str(path),'-'] if kind == 'pdf' else
                   ['textutil','-convert','txt','-stdout',str(path)])
        try:
            text = subprocess.run(command, check=True, capture_output=True).stdout.decode('utf-8')
        except FileNotFoundError as exc:
            raise RuntimeError('Install Poppler for PDF; DOC extraction requires macOS textutil') from exc
        if kind == 'pdf':
            page = proof['page']
            if not isinstance(page, int) or page < 1:
                raise ValueError('Positive PDF page number required')
            text = text.split('\f')[page-1]
    elif kind == 'html':
        text = BeautifulSoup(path.read_bytes(),'html.parser').get_text(' ',strip=True)
    elif kind == 'csv':
        with path.open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        values=proof['values']
        selected=[r for r in rows if r.get(proof['key_column']) == proof['key_value']]
        if (proof['key_column'] != '公司代號' or proof['key_value'] != row['stock_id']
                or len(selected) != 1 or not values or '上櫃日期' not in values
                or any(selected[0].get(k) != v for k,v in values.items())
                or (row.get('start') and values['上櫃日期'] != row['start'].replace('-',''))):
            raise ValueError('Official CSV company/date row differs')
        return 'primary_csv_row_verified'
    else:
        raise ValueError('Unsupported primary document format')
    snippets = proof.get('snippets', [])
    if not snippets or any(not compact(s) or compact(s) not in compact(text) for s in snippets):
        raise ValueError('Reviewed text not found in primary document: '+row['stock_id'])
    return 'primary_text_reextracted'


def build(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new immutable identity audit path')
    refs = {}
    def load(name):
        path=ROOT/name
        checked_file(name, sha(path), refs)
        return read(path)
    for name in ('scripts/audit_identity_continuation.py','skills/market_identity_overlay.py',
                 'scripts/audit_market_identity.py'):
        checked_file(name, sha(ROOT/name), refs)
    base, prior = load(BASE), load(PRIOR)
    for report in (base, prior):
        for name, digest in report['source_sha256'].items():
            checked_file(name, digest, refs)
    new_rows, pending = [], []
    for name in ADDITIONS:
        evidence = load(name)
        for row in evidence['verified']:
            row = dict(row, verification_result=verify_primary(row, refs))
            new_rows.append(row)
        pending.extend(evidence['unresolved'])
    corrections = load(CORRECTIONS)
    corrected_rows = []
    for row in corrections['verified']:
        if not row.get('replaces_snapshot_start'):
            raise ValueError('ISIN correction must bind to the observed snapshot date')
        corrected_rows.append(dict(row,verification_result=verify_primary(row,refs)))
    filled = apply_starts(base['episodes'], prior['resolved_rows']+new_rows+corrected_rows)
    # This is a discrepancy inventory, not permission to rewrite every current date.
    # Corporate reorganizations/new codes require predecessor and event evidence.
    basic_name='.cache/isin-date-corrections-20260924/tpex-company-basic.csv'
    basic_path=checked_file(basic_name,refs[basic_name],refs)
    with basic_path.open(encoding='utf-8-sig',newline='') as stream:
        basic_rows=list(csv.DictReader(stream))
    basic={r['公司代號']:r for r in basic_rows}
    if len(basic)!=len(basic_rows):
        raise ValueError('Duplicate company in official basic table')
    corrected_ids={r['stock_id'] for r in corrected_rows}
    discrepancies=[]
    for e in base['episodes']:
        if e['market']!='TPEx' or e['end'] is not None or e['category']!='股票' or e['stock_id'] not in basic:
            continue
        raw=basic[e['stock_id']]['上櫃日期']
        if len(raw)!=8 or not raw.isascii() or not raw.isdigit():
            raise ValueError('Unreviewed company basic listing date format')
        stamp=f'{raw[:4]}-{raw[4:6]}-{raw[6:]}'
        day(stamp)
        if e['start']!=stamp:
            discrepancies.append(dict(stock_id=e['stock_id'],snapshot_start=e['start'],
                company_basic_listing_date=stamp,company_basic_publication=basic[e['stock_id']]['出表日期'],
                isin_snapshot_date=e.get('snapshot_date'),
                individually_resolved=e['stock_id'] in corrected_ids))
    # Verify the sealed research manifest against the preceding published evidence summary.
    old_name = 'artifacts/forward_simulation/completion_gaps_delivery_20260924_v2.json'
    old_path = checked_file(old_name, (ROOT/old_name).with_suffix('.sha256').read_text().strip(), refs)
    seal = read(old_path)['evidence_sha256']
    manifest_name = RESEARCH+'/manifest.json'
    manifest = read(checked_file(manifest_name, seal[manifest_name], refs))['files_sha256']
    summary_name = RESEARCH+'/summary.json'
    summary = read(checked_file(summary_name, manifest['summary.json'], refs))
    identity_name = RESEARCH+'/identity.json'
    identity = read(checked_file(identity_name, manifest['identity.json'], refs))
    companies_name = '.cache/million-replay-inputs/companies.parquet'
    companies = pd.read_parquet(checked_file(companies_name, identity[companies_name], refs))
    if companies.stock_id.duplicated().any():
        raise ValueError('Duplicate frozen company identity')
    static = dict(zip(companies.stock_id, companies.market));static['0050']='TWSE'
    signals_name = '.cache/five-axis-20260913/rebuild/signals.json'
    entries = read(checked_file(signals_name, identity[signals_name], refs))['entries']
    signals = [dict(stock_id=sid,date=entry['signal_date']) for entry in entries for sid in entry['members']]
    if len(signals) != 458:
        raise ValueError('Expected complete frozen 458-signal cohort')
    signal_audit = dict(before=audit_rows(base['episodes'],signals,static),
                        after=audit_rows(filled,signals,static))
    # Members used to form a group also require dated identity, even when never bought.
    group_audit = {}
    for stamp in ('group_cutoff_date','signal_date'):
        members = [dict(stock_id=sid,date=e[stamp],event_id=e['event_id'],
                        leader=e['members'][0],group_id=e['group_id'])
                   for e in entries for sid in e['group_members']]
        group_audit[stamp] = dict(before=audit_rows(base['episodes'],members,static),
                                 after=audit_rows(filled,members,static))
    cases = {}
    for name, item in summary['cases'].items():
        case_name = f'{RESEARCH}/cases/{name}.json'
        case = read(checked_file(case_name, manifest[f'cases/{name}.json'], refs))
        if not case['completed']:
            raise ValueError('Incomplete frozen account')
        benchmark = case['config']['benchmark']
        groups = {}
        for kind in ('orders','trades','holdings'):
            rows = case['account'][kind]
            before = audit_rows(base['episodes'],rows,static,benchmark=benchmark)
            after = audit_rows(filled,rows,static,benchmark=benchmark)
            groups[kind] = dict(before=before,after=after,unchanged=before==after)
        cases[name] = dict(groups=groups, passed=all(r['after']['passed'] for r in groups.values()),
                           summary_unchanged=item['summary']==case['summary'],
                           account_sha256=sha(ROOT/case_name))
    remaining = [e for e in filled if e['start'] is None]
    expected = {(e['stock_id'],e['market']) for e in remaining}
    if {(e['stock_id'],e['market']) for e in pending} != expected:
        raise ValueError('Remaining primary-source inventory differs from unresolved episodes')
    for name, digest in refs.items():
        checked_file(name,digest,{})
    result = dict(schema='listing_identity_continuation_v1', original_unknown_starts=len(base['missing_start_rows']),
        previously_resolved=len(prior['resolved_rows']), newly_resolved=len(new_rows),
        remaining_unknown_starts=len(remaining), new_evidence=new_rows, unresolved=remaining,
        current_isin_date_corrections=corrected_rows, unresolved_isin_corrections=corrections['unresolved'],
        current_isin_vs_company_basic=discrepancies,
        unresolved_current_date_discrepancies=sum(not r['individually_resolved'] for r in discrepancies),
        unconfirmed_categories=sum(e['category']=='unconfirmed' for e in filled),
        episodes=filled, signal_audit=signal_audit, group_member_audit=group_audit, cases=cases,
        all_selected_identities_passed=signal_audit['after']['passed'] and all(c['passed'] for c in cases.values()),
        group_member_identities_passed=all(g['after']['passed'] for g in group_audit.values()),
        selected_identities_unchanged=signal_audit['before']==signal_audit['after'] and
            all(g['unchanged'] for c in cases.values() for g in c['groups'].values()),
        complete_historical_universe=False, publication_time_archive_complete=False,
        current_isin_date_is_not_initial_ipo=True,
        continuous_eligibility_proven=False, performance_recomputed=False,
        database_mutations=0, network_requests=0, live_qualified=False, source_sha256=refs)
    write(output,result)
    return {k:v for k,v in result.items() if k not in ('episodes','source_sha256','new_evidence','unresolved','cases')}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output),ensure_ascii=False))
