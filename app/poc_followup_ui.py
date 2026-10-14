"""Fail-closed reader for the fixed POC optimization matrix."""
import hashlib
import json
import math
from pathlib import Path

from skills.top3_validation import MODES

IDENTIFIER = 'poc_followup_20261014'
ARMS = ('poc_up_red', 'poc_priority')
VARIANTS = ('exit_confirm', 'poc_strength', 'correlation60')
NAMES = dict(original='原版', exit_confirm='出場確認', poc_strength='POC 強度', correlation60='持股相關性')


def validate(report):
    expected = {f'{a}:{v}:{m}' for a in ARMS for v in ('original', *VARIANTS) for m in MODES}
    expected |= {f'benchmark:original:{m}' for m in MODES}
    if (report.get('schema') != 'poc_followup_publication_v1'
            or (report.get('start'), report.get('end'), report.get('initial_cash'), report.get('position_count'))
                != ('2024-01-02', '2026-10-02', 1000000, 3)
            or (report.get('buy_fraction'), report.get('sell_fraction')) != (.5, .3)
            or any(report.get(k) is not False for k in ('live_qualified', 'paper_enabled', 'actual_fill_verified', 'unseen_validation'))
            or report.get('modes') != {k: dict(commission=v[0], slippage=v[1]) for k, v in MODES.items()}
            or set(report.get('cases', {})) != expected):
        raise ValueError('POC 後續驗證的比較範圍或資格標示不符')
    cases = report['cases']
    for key, row in cases.items():
        if key != f"{row['id']}:{row['variant']}:{row['mode']}" or type(row.get('completed')) is not bool:
            raise ValueError('POC 後續驗證案例識別錯誤')
        if not row['completed']:
            if row['variant'] == 'original':
                raise ValueError('原版與0050基準必須是完整封存帳戶')
            if any(row.get(k) is not None for k in ('metrics', 'summary', 'analytics')):
                raise ValueError('未完成案例不能顯示績效')
            continue
        if row['variant'] in VARIANTS and (row.get('followup_audit', {}).get('passed') is not True
                or row['followup_audit'].get('variant') != row['variant']):
            raise ValueError('新變因缺少獨立決策核對')
        metrics, analytics = row['metrics'], row['analytics']
        if (metrics['trading_days'] != 666 or any(type(metrics[k]) not in (int, float) or not math.isfinite(metrics[k])
                for k in ('total_return', 'max_drawdown', 'final_nav'))
                or any(abs(metrics[k]-row['summary'][k]) > 1e-7 for k in ('total_return', 'max_drawdown', 'final_nav'))):
            raise ValueError('POC 後續驗證帳戶績效不一致')
        if (analytics['settled'] != analytics['wins']+analytics['losses']+analytics['breakeven']
                or analytics['win_rate'] != (analytics['wins']/analytics['settled'] if analytics['settled'] else None)):
            raise ValueError('POC 後續驗證勝率分母錯誤')
        if row['variant'] in VARIANTS:
            unchanged = row.get('same_funded_path_as_original')
            original = cases[f"{row['id']}:original:{row['mode']}"]
            if (type(unchanged) is not bool or (unchanged and any(
                    metrics[k] != original['metrics'][k]
                    for k in ('total_return', 'max_drawdown', 'final_nav')))):
                raise ValueError('POC 後續驗證的相同帳戶路徑標示不符')
        base = cases[f"benchmark:original:{row['mode']}"]['metrics']
        if row['excess_return'] != metrics['total_return']-base['total_return']:
            raise ValueError('POC 後續驗證使用了不同成本的0050')
    completed = sum(r['completed'] for r in cases.values() if r['variant'] in VARIANTS)
    if report.get('completed_new_accounts') != completed or report.get('expected_new_accounts') != 24:
        raise ValueError('POC 後續驗證完成數錯誤')
    for arm in ARMS:
        for variant in VARIANTS:
            decision = report['decisions'][f'{arm}:{variant}']
            rows = [cases[f'{arm}:{variant}:{m}'] for m in MODES]
            improved = all(r['completed'] and r['metrics']['total_return'] > cases[f"{arm}:original:{r['mode']}"]['metrics']['total_return']
                and r['metrics']['max_drawdown'] >= cases[f"{arm}:original:{r['mode']}"]['metrics']['max_drawdown'] for r in rows)
            stress = all(r['completed'] and r['metrics']['total_return'] > 0 and r['excess_return'] > 0
                and r['metrics']['max_drawdown'] >= cases[f"benchmark:original:{r['mode']}"]['metrics']['max_drawdown'] for r in rows)
            if (decision.get('improvement_pass') is not improved or decision.get('historical_stress_pass') is not stress
                    or any(decision.get(k) is not False for k in ('unseen_validation', 'causal_fill_verified', 'paper_enabled', 'live_qualified'))):
                raise ValueError('POC 後續驗證通過狀態錯誤')
    audit = report.get('execution_audit', {})
    if (audit.get('completed') is not True or audit.get('passed') is not False
            or audit.get('whole_account_replayed') is not False or not audit.get('blockers')
            or report.get('variants') != list(VARIANTS)):
        raise ValueError('凍結委託診斷不能冒充可執行帳戶驗證')
    return report


def case_detail(publication, case_id, root):
    validate(publication)
    row = next((r for k, r in publication['cases'].items() if k.replace(':', '__') == case_id), None)
    if not row or not row['completed']:
        raise ValueError('此優化案例未完成或不存在')
    root = Path(root).resolve()
    path = (root/row['path']).resolve()
    if not path.is_relative_to(root) or path.suffix != '.json':
        raise ValueError('優化帳戶路徑錯誤')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('優化帳戶指紋不符')
    case = json.loads(raw)
    if case.get('completed') is not True or case.get('summary') != row['summary']:
        raise ValueError('優化帳戶摘要不符')
    if row['variant'] in VARIANTS:
        metadata = case.get('followup', {})
        if (any(metadata.get(k) != row[k] for k in ('variant', 'mode'))
                or metadata.get('arm') != row['id']
                or metadata.get('preregistration') != publication['preregistration']):
            raise ValueError('優化帳戶與事前登記版本不符')
    account = case['account']
    if (len(account['daily']) != 666 or (account['settings']['commission'], account['settings']['slippage']) != MODES[row['mode']]
            or (account['daily'][0]['date'], account['daily'][-1]['date']) != (publication['start'], publication['end'])):
        raise ValueError('優化帳戶期間或成本不符')
    return dict(id=case_id, name=row['name']+'・'+NAMES[row['variant']],
        initial_cash=1000000, position_count=1 if row['id']=='benchmark' else 3, position_scope='active_stock_slots',
        daily=[{k:r.get(k) for k in ('date','nav','cash','market_value','receivable','drawdown','holdings')} for r in account['daily']],
        trades=[dict({k:r.get(k) for k in ('date','signal_date','stock_id','name','side','qty','reason','total_cost','cash_after')},
            price=r['reference_price']) for r in account['trades']],
        live_qualified=False, actual_fill_verified=False, source_sha256=row['sha256'],
        note='買HL2／賣30%區間價代理；此版本為已見歷史研究，未驗證實際成交。')
