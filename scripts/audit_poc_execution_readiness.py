#!/usr/bin/env python3
"""Reuse sealed accounts and tapes; report missing execution evidence without replaying."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.poc_followup_ui import validate as validate_followup
from scripts.summarize_poc_broker_account import account_metrics
from skills.top3_validation import MODES, verify_costs

PUBLICATION = 'artifacts/forward_simulation/poc_followup_20261014.json'
AUDIT = '.cache/poc-followup-20261014/execution-audit-v1.json'
REQUIREMENTS = 'docs/poc_followup_execution_requirements_20261014.json'
DISCOVERY = '.cache/poc-execution-readiness-20261014/finmind-discovery-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root=ROOT):
    root = Path(root)
    refs = {}

    def bind(name, expected=None):
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Evidence path escapes repository')
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Evidence changed: '+name)
        refs[name] = actual
        return path

    def read(name, expected=None):
        return json.loads(bind(name, expected).read_text())

    publication = read(PUBLICATION, (root/PUBLICATION).with_suffix('.sha256').read_text().strip())
    validate_followup(publication)
    requirement = read(REQUIREMENTS)
    audit = read(AUDIT, requirement['source_sha256'][AUDIT])
    for name, sha in requirement['source_sha256'].items():
        bind(name, sha)
    original = {k:r for k,r in publication['cases'].items() if r['variant']=='original'}
    if set(original) != set(audit['cases']):
        raise ValueError('Execution diagnostic and original accounts differ')

    # Verify each already completed case's bytes, without any engine/training call.
    for row in publication['cases'].values():
        bind(row['path'], row['sha256'])
    by_sha = {sha:name for name,sha in audit['source_sha256'].items() if name.endswith('.parquet')}
    ordinary, cases, odd = {}, {}, set()
    for key, row in original.items():
        case = read(row['path'], row['sha256'])
        verify_costs(case['account'], row['mode'])
        metrics = account_metrics(case)
        if metrics != row['metrics']:
            raise ValueError('Independent original account metrics differ')
        children = audit['cases'][key]['rows']
        board = [r for r in children if r['channel']=='board']
        odds = [r for r in children if r['channel']=='odd']
        planned = [r for r in case['account']['orders'] if r['channel'] in ('board','odd') and r['requested_qty']>0]
        identity = lambda r: tuple(r[k] for k in ('date','stock_id','event_id','side','channel','requested_qty'))
        if list(map(identity, children)) != list(map(identity, planned)):
            raise ValueError('Diagnostic order scope differs from original account')
        for r in board:
            coordinate = r['stock_id'], r['date']
            raw_sha = r.get('raw_sha256')
            if raw_sha not in by_sha:
                raise ValueError('Missing bound ordinary tape')
            item = dict(stock_id=coordinate[0], date=coordinate[1], raw_sha256=raw_sha,
                source_path=by_sha[raw_sha],
                daily_totals_reconciled=r['ordinary_daily_totals_reconciled'])
            if coordinate in ordinary and ordinary[coordinate] != item:
                raise ValueError('Same ordinary coordinate has conflicting evidence')
            if coordinate not in ordinary:
                bind(item['source_path'], raw_sha)
                ordinary[coordinate] = item
        odd.update((r['stock_id'],r['date']) for r in odds)
        benchmark = original['benchmark:original:'+row['mode']]['metrics']
        cases[key] = dict(arm=row['id'], name=row['name'], mode=row['mode'],
            control_path=row['path'], control_sha256=row['sha256'],
            proxy_total_return=metrics['total_return'], proxy_max_drawdown=metrics['max_drawdown'],
            proxy_excess_return=metrics['total_return']-benchmark['total_return'],
            benchmark_proxy_return=benchmark['total_return'],
            benchmark_proxy_max_drawdown=benchmark['max_drawdown'],
            historical_stress_case_pass=(metrics['total_return']>0 and metrics['total_return']>benchmark['total_return']
                and metrics['max_drawdown']>=benchmark['max_drawdown']),
            board_child_orders=len(board), missing_odd_child_orders=len(odds),
            board_total_reconciled_orders=sum(r['ordinary_daily_totals_reconciled'] for r in board),
            frozen_board_partial_or_unfilled=sum(r['estimate']['failure'] is not None for r in board),
            first_missing_odd={k:odds[0][k] for k in ('stock_id','date','side','requested_qty')},
            whole_account_replayed=False, executable_total_return=None, executable_excess_return=None,
            causal_fill_verified=False, paper_enabled=False)
    coordinates = requirement['coordinates']
    if odd != {(r['stock_id'],r['date']) for r in coordinates} or len(odd)!=len(coordinates):
        raise ValueError('Missing odd coordinate inventory differs')

    api = read(DISCOVERY+'/openapi.json')
    supported = api['components']['schemas']['DataSetInputV4']['enum']
    if any('odd' in x.lower() for x in supported):
        raise ValueError('FinMind added an odd-lot dataset; inspect before declaring a gap')
    docs = [dict(url='https://api.finmindtrade.com/openapi.json', path=DISCOVERY+'/openapi.json'),
        dict(url='https://finmind.github.io/tutor/TaiwanMarket/DataList/', path=DISCOVERY+'/datalist.html'),
        dict(url='https://finmind.github.io/tutor/TaiwanMarket/Technical/', path=DISCOVERY+'/technical.html')]
    for source in docs:
        source['sha256'] = digest(bind(source['path']))
    probes = []
    for sid,day in [('2618','2024-01-03'),('6187','2024-01-19'),('0050','2024-01-02')]:
        probe = read(DISCOVERY+'/'+sid+'-'+day+'.json')
        if (probe['status']!='received' or not probe['original_values_unchanged']
                or set(probe['columns']) != {'date','stock_id','deal_price','volume','Time','TickType'}):
            raise ValueError('FinMind response changed; investigate before publication')
        for name, sha in probe['source_sha256'].items():
            bind(name, sha)
        probes.append(probe)
    bind('.cache/poc-execution-readiness-20261014/probe_finmind.py')
    bind('scripts/audit_poc_execution_readiness.py')
    result = dict(schema='poc_execution_readiness_v1', checked_at=datetime.now(timezone.utc).isoformat(),
        start=publication['start'], end=publication['end'], initial_cash=1000000,
        status='blocked_missing_intraday_odd_sequence', audit_completed=True,
        execution_validation_completed=False, whole_account_replayed=False,
        actual_fill_verified=False, causal_fill_verified=False, paper_enabled=False,
        unseen_validation=False, live_qualified=False, rerun_accounts=0,
        preserved_completed_accounts=len(publication['cases']), source_publication=dict(path=PUBLICATION,sha256=refs[PUBLICATION]),
        source_scope='Original 12 controls and 309 bound ordinary tapes checked; the older full historical closure is reused, not rehashed.',
        modes=publication['modes'], cases=cases,
        ordinary=dict(unique_coordinates=len(ordinary), available_tapes=len(ordinary),
            totals_reconciled=sum(r['daily_totals_reconciled'] for r in ordinary.values()),
            coordinates=list(ordinary.values()), complete_exchange_sequence_verified=False),
        odd=dict(required_coordinates=len(odd), verified_chronological_coordinates=0,
            by_market=dict(Counter(r['market'].upper() for r in coordinates)), coordinates=coordinates,
            scope='Minimum missing data for frozen original orders; causal whole-account paths can require additional coordinates.'),
        finmind=dict(supported_dataset_count=len(supported), supported_odd_dataset_found=False,
            tick_schema=list(api['components']['schemas']['TaiwanStockPriceTick']['properties']),
            note='Checked documented interfaces and three actual responses. TickType is buy/sell aggressor classification, not board/odd channel. No verified intraday odd auction sequence was obtained.',
            probes=probes, actual_api_requests=sum(p['actual_api_requests'] for p in probes), documents=docs),
        remaining=[
            '取得具來源與交易通路辨識的盤中零股撮合時序，並核對普通盤逐筆完整性。',
            '固定事前可下單的時間、限價及部分成交規則，保留兩套原選股與出場；連同同成本0050重播現金、持股、公司行動及四種成本。',
            '完成訊號當時可得資料、還原價與公司行動對帳；目前原版四種成本比較均未達贏0050且回撤不較差的歷史門檻。',
            '鎖版後累積126個真正未見交易日、至少30批結清，淨報酬與超額為正且回撤不劣於0050；先核對委託與成交回報流程。'],
        source_sha256=refs)
    # Keep old successful evidence immutable even when a later stage is blocked.
    for name, sha in list(refs.items()):
        bind(name, sha)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve sealed readiness report; choose a new version')
    result = build()
    from app.poc_execution_readiness import validate
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2)+'\n')
    args.output.with_suffix('.sha256').write_text(digest(args.output)+'\n')
    print(json.dumps({k:result[k] for k in ('status','rerun_accounts','preserved_completed_accounts')}))
