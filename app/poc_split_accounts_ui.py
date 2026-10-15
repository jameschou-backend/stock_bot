"""Fail-closed publication and detail reader for independent half-capital sleeves."""
import hashlib
import json
import math
from pathlib import Path

from skills.top3_validation import MODES

IDENTIFIER = 'poc_split_accounts_20261015'
ARMS = ('poc_up_red', 'poc_priority')


def validate(report):
    expected = {f'{a}:{v}:{m}' for a in ARMS for v in ('original', 'split50') for m in MODES}
    expected |= {f'{a}:{v}:{m}' for a, v in [('benchmark', 'original'), ('combined', 'combined')] for m in MODES}
    if (report.get('schema') != 'poc_split_accounts_publication_v1'
            or (report.get('start'), report.get('end'), report.get('initial_cash'), report.get('sleeve_initial_cash'))
                != ('2024-01-02', '2026-10-02', 1000000, 500000)
            or (report.get('position_count'), report.get('slots_per_sleeve')) != (6, 3)
            or any(report.get(k) is not False for k in ('rebalance', 'cash_transfers', 'live_qualified', 'paper_enabled', 'actual_fill_verified', 'unseen_validation'))
            or report.get('modes') != {k: dict(commission=v[0], slippage=v[1]) for k, v in MODES.items()}
            or set(report.get('cases', {})) != expected
            or (report.get('expected_new_accounts'), report.get('completed_new_accounts'), report.get('reused_accounts')) != (8, 8, 12)):
        raise ValueError('分帳比較範圍、成本或資格標示不符')
    cases = report['cases']
    for key, row in cases.items():
        initial = 500000 if row['variant'] == 'split50' else 1000000
        m = row['metrics']
        if (key != f"{row['id']}:{row['variant']}:{row['mode']}" or row.get('completed') is not True
                or (m['initial_cash'], m['trading_days']) != (initial, 666)
                or any(type(m[k]) not in (int, float) or not math.isfinite(m[k]) for k in ('total_return', 'max_drawdown', 'final_nav'))
                or any(abs(m[k]-row['summary'][k]) > 1e-7 for k in ('total_return', 'max_drawdown', 'final_nav'))
                or abs(m['final_nav']/initial-1-m['total_return']) > 1e-10):
            raise ValueError('分帳績效或本金不一致')
        if row['variant'] == 'split50':
            if row.get('excess_return') is not None or row.get('rolling252') is not None:
                raise ValueError('50萬子帳戶不能冒充與100萬0050同本金比較')
        elif row['excess_return'] != m['total_return']-cases[f"benchmark:original:{row['mode']}"]['metrics']['total_return']:
            raise ValueError('分帳比較未使用同成本0050')
        if row['variant'] == 'split50':
            audit = row.get('funding_audit', {})
            if (audit.get('initial_cash') != 500000 or audit.get('independent_cash') is not True
                    or audit.get('rebalance') is not False or audit.get('scaled_original_result') is not False):
                raise ValueError('分帳缺少獨立本金核對')
        if row['variant'] == 'combined':
            audit = row.get('aggregation_audit', {})
            cap = row.get('shared_capacity', {})
            if (audit.get('passed') is not True or audit.get('days') != 666
                    or audit.get('cash_transfers') is not False or audit.get('rebalance') is not False
                    or audit.get('method') != 'sum_independent_nav_recompute_peak'
                    or cap.get('capacity_collision_count') != len(cap.get('collisions', []))
                    or cap.get('joint_capacity_pass') is not (cap.get('capacity_collision_count') == 0)
                    or cap.get('actual_fill_verified') is not False):
                raise ValueError('合併淨值或共用成交容量核對不完整')
            children = [cases[f'{a}:split50:{row["mode"]}']['metrics'] for a in ARMS]
            if abs(m['final_nav']-sum(c['final_nav'] for c in children)) > .011:
                raise ValueError('合併資產不等於兩分帳總和')
    for mode in MODES:
        new = cases[f'combined:combined:{mode}']['metrics']
        for arm in (*ARMS, 'benchmark'):
            old = cases[f'{arm}:original:{mode}']['metrics']
            wanted = dict(return_change=new['total_return']-old['total_return'],
                return_sacrifice=old['total_return']-new['total_return'],
                drawdown_improvement=new['max_drawdown']-old['max_drawdown'],
                drawdown_relative_reduction=(new['max_drawdown']-old['max_drawdown'])/abs(old['max_drawdown']),
                profit_change=new['profit']-old['profit'])
            if report['comparisons'][mode][arm] != wanted:
                raise ValueError('分帳報酬或回撤差額錯誤')
    return report


def case_detail(publication, case_id, root):
    validate(publication)
    row = next((r for k, r in publication['cases'].items() if k.replace(':', '__') == case_id), None)
    if row is None: raise ValueError('分帳案例不存在')
    root = Path(root).resolve()
    path = (root/row['path']).resolve()
    if not path.is_relative_to(root) or path.suffix != '.json': raise ValueError('分帳路徑錯誤')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha256']: raise ValueError('分帳指紋不符')
    case = json.loads(raw)
    if case.get('completed') is not True or case.get('summary') != row['summary']:
        raise ValueError('分帳摘要不符')
    account = case['account']
    if row['variant'] == 'split50':
        meta = case.get('split_accounts', {})
        if (meta.get('arm'), meta.get('mode'), meta.get('preregistration')) != (row['id'], row['mode'], publication['preregistration']):
            raise ValueError('分帳與事前登記不同')
    if row['variant'] == 'combined':
        expected = {a: {k: publication['cases'][f'{a}:split50:{row["mode"]}'][k] for k in ('path', 'sha256')} for a in ARMS}
        if case.get('sleeve_sources') != expected: raise ValueError('合併分帳來源不同')
    if (len(account['daily']) != 666 or account['settings']['initial_cash'] != row['metrics']['initial_cash']
            or (account['settings']['commission'], account['settings']['slippage']) != MODES[row['mode']]
            or (account['daily'][0]['date'], account['daily'][-1]['date']) != (publication['start'], publication['end'])):
        raise ValueError('分帳期間、本金或成本不符')
    combined = row['variant'] == 'combined'
    labels = dict(poc_up_red='硬篩帳戶', poc_priority='優先帳戶')
    return dict(id=case_id, name=row['name']+('・50萬分帳' if row['variant'] == 'split50' else ''),
        initial_cash=row['metrics']['initial_cash'], position_count=6 if combined else 1 if row['id']=='benchmark' else 3,
        position_scope='active_stock_slots',
        daily=[{k:r.get(k) for k in ('date','nav','cash','market_value','receivable','drawdown','holdings')} for r in account['daily']],
        trades=[dict({k:r.get(k) for k in ('date','signal_date','stock_id','side','qty','reason','total_cost')},
            name=(labels[r['sleeve']]+'｜' if combined else '')+r.get('name',r['stock_id']),
            price=r['reference_price'], cash_after=r.get('sleeve_cash_after') if combined else r.get('cash_after')) for r in account['trades']],
        live_qualified=False, actual_fill_verified=False, source_sha256=row['sha256'],
        note=('各分帳獨立運作；明細現金為該分帳成交後餘額。' if combined else '')+'日區間成交價估算，未驗證實際成交。')
