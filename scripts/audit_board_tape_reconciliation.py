#!/usr/bin/env python3
"""Replay an independent daily-total audit of the 917 required board stock-days."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import requests
from skills.backtest_data_evidence import verify_report as verify_data_report
from skills.board_tape_reconciliation import (digest, parse_tpex, parse_twse, summarize_ticks,
                                              reconcile, verify_report)

CACHE = ROOT/'.cache/board-tape-reconciliation-20260925'
DATA = ROOT/'artifacts/forward_simulation/backtest_data_completion_20260925.json'
DEFAULT_OUTPUT = ROOT/'artifacts/forward_simulation/board_tape_reconciliation_20260925.json'
TPEX_URL = 'https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php'
CODE = ['skills/board_tape_reconciliation.py', 'scripts/audit_board_tape_reconciliation.py',
        'skills/intraday_limit_replay.py', 'skills/backtest_data_evidence.py',
        'scripts/replay_contingent_day.py']


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n'


def required_items(data):
    required = {(r['date'], r['stock_id']) for c in data['cases'].values()
                for r in c['ordinary']['unverified']+c['ordinary']['missing']}
    catalog = {(r['date'], r['stock_id']): r for r in data['catalog']['entries'] if r['channel'] == 'board'}
    if required-set(catalog):
        raise ValueError('A required board source is absent from the frozen catalog')
    return [catalog[k] for k in sorted(required)]


def fetch_tpex(data, cache, budget):
    """Explicit bounded preparation; no retries, repeated identities or FinMind calls."""
    if not 1 <= budget <= 250:
        raise ValueError('Request budget must be 1..250')
    cache = Path(cache)
    out = cache/'official'
    out.mkdir(parents=True, exist_ok=True)
    ledger = cache/'tpex-fetch-ledger.json'
    state = json.loads(ledger.read_text()) if ledger.exists() else dict(limit=budget, attempts=[], requests=[])
    if state['limit'] != budget:
        raise ValueError('Existing immutable request budget differs')
    days = sorted({r['date'] for r in required_items(data) if r['market'] == 'TPEX'})
    with requests.Session() as session:
        for day in days:
            name = 'tpex-no1430-'+day
            path, meta = out/(name+'.json'), out/(name+'.source.json')
            if meta.exists() or day in state['attempts']:
                continue
            if len(state['attempts']) >= budget:
                raise ValueError('Request budget exhausted')
            # Consume the identity before transport; interruption is not a free retry.
            state['attempts'].append(day)
            ledger.write_text(encoded(state))
            d = datetime.fromisoformat(day)
            params = dict(l='zh-tw', d=f'{d.year-1911}/{d.month:02d}/{d.day:02d}', o='json', se='EW')
            item = dict(date=day, market='TPEX', kind='no1430', url=TPEX_URL, params=params,
                        retrieved_at=datetime.now(timezone.utc).isoformat())
            try:
                response = session.get(TPEX_URL, params=params, timeout=30)
                path.write_bytes(response.content)
                item.update(http_status=response.status_code, sha256=digest(path),
                            path=str(path.relative_to(ROOT)), bytes=len(response.content))
                meta.write_text(encoded(item))
            except requests.RequestException as exc:
                item['error_type'] = type(exc).__name__
            state['requests'].append(item)
            ledger.write_text(encoded(state))
            print(dict(date=day, http_status=item.get('http_status'), attempts=len(state['attempts'])), flush=True)
            time.sleep(1.5)
    return state


def run(data_path=DATA, cache=CACHE, output=DEFAULT_OUTPUT, verify=False):
    data_path, cache, output = Path(data_path), Path(cache), Path(output)
    data = verify_data_report(data_path, ROOT)
    refs = {}
    def reference(path):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Source escapes repository')
        refs[str(path.relative_to(ROOT))] = digest(path)
        return path
    reference(data_path)
    reference(data_path.with_suffix('.sha256'))
    # The parent audit already authenticates its complete local hash closure.
    refs.update(data['input_sha256'])
    refs.update(data['code_sha256'])
    def source(path, meta, expected_url):
        receipt = json.loads(reference(meta).read_text())
        path = reference(path)
        if (receipt.get('url') != expected_url or receipt.get('http_status', receipt.get('status_code')) != 200
                or receipt.get('sha256') != digest(path)):
            raise ValueError('Official raw/receipt provenance mismatch')
        return json.loads(path.read_text())
    items = required_items(data)
    by_day = {}
    failures = {}
    for day in sorted({r['date'] for r in items if r['market'] == 'TPEX'}):
        name = 'tpex-no1430-'+day
        path = cache/'official'/(name+'.json')
        meta = cache/'official'/(name+'.source.json')
        if not (path.exists() and meta.exists()):
            failures[('TPEX',day)] = 'official_daily_source_missing'
            continue
        try:
            payload = source(path,meta,TPEX_URL)
            receipt = json.loads(meta.read_text())
            d = datetime.fromisoformat(day)
            if receipt.get('params') != dict(l='zh-tw',d=f'{d.year-1911}/{d.month:02d}/{d.day:02d}',o='json',se='EW'):
                raise ValueError('TPEx query identity mismatch')
            by_day[('TPEX',day)] = parse_tpex(payload,day)
        except (ValueError,KeyError,TypeError) as exc:
            failures[('TPEX',day)] = type(exc).__name__+': '+str(exc)
    # Fixed prototype selected chronologically before seeing matches; reuse all
    # required stocks covered by its complete official component bundle.
    day = '2022-01-04'
    prototype = cache/'prototype'
    names = {
        'total': ('twse-mi-index-20220104', 'afterTrading/MI_INDEX?date=20220104&type=ALLBUT0999&response=json'),
        'after_odd': ('TWT53U', 'afterTrading/TWT53U?date=20220104&response=json'),
        'fixed': ('twse-bft41u-all-20220104', 'afterTrading/BFT41U?selectType=ALL&date=20220104&response=json'),
        'block_single': ('TWT93U', 'block/BFIAUU?date=20220104&response=json'),
        'block_basket': ('twse-block-m-20220104', 'block/BFIAUU?selectType=M&date=20220104&response=json')}
    try:
        parts = {kind: source(prototype/(name+'.json'),prototype/(name+'.source.json'),
                     'https://www.twse.com.tw/rwd/zh/'+endpoint) for kind,(name,endpoint) in names.items()}
        odd_path = ROOT/'.cache/sector-account-sources-r2-20260925/inputs/execution-feeds/odd-twse-2022-01-04.raw.json'
        odd = json.loads(reference(odd_path).read_text())
        if odd.get('url') != 'https://www.twse.com.tw/rwd/zh/afterTrading/TWTC7U' or odd.get('http_status') != 200:
            raise ValueError('Existing intraday odd-lot source provenance differs')
        parts['intraday_odd'] = odd['payload']
        by_day[('TWSE',day)] = parse_twse(parts,day)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        failures[('TWSE',day)] = type(exc).__name__+': '+str(exc)
    rows = []
    for item in items:
        key = (item['market'],item['date'])
        path = reference(ROOT/item['path'])
        if digest(path) != item['sha256']:
            raise ValueError('Frozen tape digest differs')
        raw = pd.read_parquet(path)
        tape = summarize_ticks(raw,item['stock_id'],item['date'],item['market'])
        official = by_day.get(key,{}).get(item['stock_id'])
        if official is None:
            result = dict(status='official_daily_source_missing' if key not in by_day else 'official_stock_row_missing',
                reason=failures.get(key,'Complete independent same-session component bundle has not been acquired'),
                tape=tape, same_scope_aggregate_matched=False, transaction_count_verified=False,
                tick_sequence_complete=False, accepted_for_strict_replay=False, own_order_fill_proven=False, live_qualified=False)
        else:
            result = reconcile(tape,official)
            if result['status'] == 'daily_aggregate_conflict':
                result['diagnosis'] = dict(raw_rows=len(raw), normalization_dropped_rows=tape['normalization_dropped_rows'],
                    raw_all_session_shares=tape['raw_all_session_shares'], fixed_price_shares=tape['fixed_price_shares'],
                    unknown_session_rows=tape['unknown_session_rows'], raw_duplicate_rows=int(raw.duplicated().sum()),
                    duplicate_rows_removed=False, official_minus_regular_shares=official['shares']-tape['shares'],
                    official_minus_regular_amount_cents=official['amount_cents']-tape['amount_cents'],
                    classification='share_quantity_conflict' if 'shares' in result['differences'] else 'amount_or_price_conflict',
                    cause='unresolved_provider_vs_exchange_aggregate_discrepancy',
                    no_claim_of_specific_missing_transaction=True)
        rows.append(dict(date=item['date'],stock_id=item['stock_id'],market=item['market'],
                         tape_path=item['path'],tape_sha256=item['sha256'],**result))
    summary = dict(required_sessions=len(rows),same_scope_aggregate_matched=sum(r['same_scope_aggregate_matched'] for r in rows),
        aggregate_conflicts=sum(r['status']=='daily_aggregate_conflict' for r in rows),
        share_quantity_conflicts=sum('shares' in r.get('differences',{}) for r in rows),
        amount_only_conflicts=sum(set(r.get('differences',{}))=={'amount_cents'} for r in rows),
        independent_daily_source_missing=sum(r['status'].endswith('_missing') for r in rows),
        transaction_count_verified=0,tick_sequence_complete=0,accepted_for_strict_replay=0)
    markets = {m:dict(required=sum(r['market']==m for r in rows),
        status_counts=dict(Counter(r['status'] for r in rows if r['market']==m))) for m in ('TWSE','TPEX')}
    indexed={(r['date'],r['stock_id']):r for r in rows}
    cases={name:dict(required_sessions=c['ordinary']['required_sessions'],
        matched_sessions=sum(indexed[(r['date'],r['stock_id'])]['same_scope_aggregate_matched']
            for r in c['ordinary']['unverified']+c['ordinary']['missing']),strict_data_ready=False)
        for name,c in data['cases'].items()}
    ledger=cache/'tpex-fetch-ledger.json'
    prepared=json.loads(reference(ledger).read_text()) if ledger.exists() else dict(attempts=[])
    # Include successful and rejected prototype attempts in the cost accounting.
    proto_receipts=list(prototype.glob('*.source.json'))
    for p in proto_receipts:
        reference(p)
        raw_stem = p.name.removesuffix('.source.json')
        for suffix in ('.json', '.html'):
            raw = prototype/(raw_stem+suffix)
            if raw.exists():
                reference(raw)
    required_twse_days={r['date'] for r in items if r['market']=='TWSE'}
    reused_odd=sum((ROOT/f'.cache/sector-account-sources-r2-20260925/inputs/execution-feeds/odd-twse-{day}.raw.json').exists()
                   for day in required_twse_days)
    result=dict(schema='board_tape_reconciliation_v1',as_of='2026-09-25',scope='20_frozen_account_required_stock_days',
        summary=summary,markets=markets,cases=cases,rows=rows,
        quarantine=[dict(date=r['date'],stock_id=r['stock_id'],market=r['market'],
                        tape_sha256=r['tape_sha256'],reason='official_daily_aggregate_conflict',
                        differences=r['differences'],diagnosis=r['diagnosis'])
                    for r in rows if r['status']=='daily_aggregate_conflict'],input_sha256=refs,
        code_sha256={name:digest(ROOT/name) for name in CODE},
        preparation_official_http_requests=len(prepared['attempts'])+len(proto_receipts),
        preparation_finmind_requests=0,network_requests=0,database_mutations=0,
        strict_data_ready=False,live_qualified=False,own_order_fill_proven=False,
        remaining_twse_plan=dict(required_market_dates=len(required_twse_days),completed_market_dates=sum(m=='TWSE' for m,d in by_day),
            components_per_new_date=['MI_INDEX','TWT53U','BFT41U_ALL','BFIAUU_S','BFIAUU_M'],
            minimum_new_daily_component_requests=(len(required_twse_days)-sum(m=='TWSE' for m,d in by_day))*5,
            reusable_intraday_odd_dates=reused_odd,missing_intraday_odd_dates=len(required_twse_days)-reused_odd,
            extra_basket_constituent_requests='unknown_until_BFIAUU_M_is_read',batch_started=False),
        limitations=[
            'Daily aggregate equality does not prove intraday sequence, timing, missing-offsetting records, or own execution.',
            'FinMind message count is not the official execution count; count equality is deliberately not asserted.',
            '14:30 fixed-price prints are excluded. Existing intraday matcher already restricts fills to 09:01<time<13:25.',
            '09:00<=time<13:34 includes the documented delayed regular close; other positive-volume times remain flagged.',
            'TWSE totals require every odd-lot/fixed-price/block component, including basket constituent details when nonempty.',
            'No source, signal, corporate action, result or strategy setting in the sealed experiments was changed.'])
    if verify:
        if json.loads(output.read_text()) != result:
            raise ValueError('Offline daily-total reproduction differs')
        verify_report(output,ROOT)
    else:
        if output.exists() and json.loads(output.read_text()) != result:
            raise ValueError('Report is immutable; choose a new output path')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(encoded(result))
        output.with_suffix('.sha256').write_text(digest(output)+'\n')
    print(json.dumps(dict(summary=summary,markets=markets,verified=verify),ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--cache',type=Path,default=CACHE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--fetch-tpex',action='store_true')
    parser.add_argument('--request-budget',type=int,default=183)
    args=parser.parse_args()
    if args.fetch_tpex:
        if args.verify:parser.error('Fetching and offline verification cannot run together')
        fetch_tpex(verify_data_report(args.data,ROOT),args.cache,args.request_budget)
    run(args.data,args.cache,args.output,args.verify)
