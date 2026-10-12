"""Read-only sealed top-three stress results. Never creates paper fills."""
from pathlib import Path
import hashlib
import json
import math

from skills.top3_validation import ARMS, MODES, qualification

IDENTIFIER = 'top3_validation_20261012'


def validate(report):
    if (report.get('schema') != 'top3_validation_publication_v1'
            or (report.get('start'), report.get('end')) != ('2024-01-02','2026-10-02')
            or report.get('buy_fraction') != .5 or report.get('sell_fraction') != .3
            or any(report.get(k) is not False for k in ('live_qualified','unseen_validation','actual_fill_verified'))
            or set(report.get('cases', {})) != {f'{arm}:{mode}' for arm in ARMS for mode in MODES}):
        raise ValueError('前三名驗證格式或資格標示不符')
    complete = 0
    if report.get('modes') != {k:dict(commission=v[0], slippage=v[1]) for k,v in MODES.items()}:
        raise ValueError('成本情境與固定規格不同')
    for key, row in report['cases'].items():
        if key != f"{row['id']}:{row['mode']}" or type(row.get('completed')) is not bool:
            raise ValueError('驗證案例識別或完成狀態錯誤')
        if row['completed']:
            complete += 1
            metrics = row['metrics']
            if metrics['trading_days'] != 666 or any(not math.isfinite(metrics[k]) for k in ('total_return','max_drawdown','final_nav')):
                raise ValueError('驗證案例缺少完整期間績效')
            if any(row[k] != metrics[k] for k in ('total_return','max_drawdown')):
                raise ValueError('表格與帳戶績效不一致')
        elif row.get('metrics') is not None or row.get('summary') is not None:
            raise ValueError('未完成案例不可列報酬')
    if report['completed_cases'] != complete or report['all_completed'] is not (complete == 16):
        raise ValueError('完成數不符')
    for arm in ARMS[:-1]:
        for mode in MODES:
            row, base = report['cases'][f'{arm}:{mode}'], report['cases'][f'benchmark:{mode}']
            comparable = row['completed'] and base['completed']
            if row['benchmark_comparable'] is not comparable or comparable and (
                row['excess_return'] != row['total_return']-base['total_return']
                or row['benchmark_max_drawdown'] != base['max_drawdown']):
                raise ValueError('0050比較口徑不一致')
        rows = {mode: dict(report['cases'][f'{arm}:{mode}'],
            completed=report['cases'][f'{arm}:{mode}']['benchmark_comparable']) for mode in MODES}
        if report['qualification'][arm] != qualification(rows):
            raise ValueError('樣本外或紙上關卡不符')
    if (report['paper']['enabled_strategies'] or report['paper']['simulated_fills_written'] != 0
            or report['out_of_sample']['sessions'] != 0):
        raise ValueError('此歷史版本不得虛構前瞻紀錄或啟用紙上成交')
    return report


def case_detail(publication, case_id, root):
    validate(publication)
    public_ids = {f"{r['id']}_{r['mode']}": r for r in publication['cases'].values()}
    row = public_ids.get(case_id)
    if not row or not row['completed']:
        raise ValueError('此案例尚未完成')
    root = Path(root).resolve()
    path = (root/row['path']).resolve()
    if not path.is_relative_to(root) or path.suffix != '.json':
        raise ValueError('帳戶路徑錯誤')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('帳戶指紋不符')
    case = json.loads(raw)
    if case.get('completed') is not True or case['summary'] != row['summary']:
        raise ValueError('帳戶摘要不符')
    account = case['account']
    settings, model = account['settings'], case['top3_validation']
    if (len(account['daily']) != 666
            or (account['daily'][0]['date'], account['daily'][-1]['date']) != ('2024-01-02','2026-10-02')
            or model['mode'] != row['mode'] or (model['buy_fraction'],model['sell_fraction']) != (.5,.3)
            or (settings['commission'],settings['slippage']) != MODES[row['mode']]):
        raise ValueError('帳戶期間或成本情境不符')
    return dict(id=case_id, name=row['name']+'・'+row['mode'], initial_cash=1_000_000,
        position_count=1 if row['id']=='benchmark' else 3, position_scope='active_stock_slots',
        daily=[{k:r.get(k) for k in ('date','nav','cash','market_value','receivable','drawdown','holdings')} for r in account['daily']],
        trades=[dict({k:r.get(k) for k in ('date','signal_date','stock_id','name','side','qty','reason','total_cost','cash_after')},
                     price=r['reference_price']) for r in account['trades']],
        live_qualified=False, actual_fill_verified=False, source_sha256=row['sha256'],
        note='訊號隔日買 (H+L)/2、賣 L+30%×(H−L)；含該組成本，日區間代理。')
