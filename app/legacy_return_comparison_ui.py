"""Read-only verified legacy return comparison and bound account details."""
import hashlib
import json
import math
from pathlib import Path
from skills.legacy_return_comparison import ARMS
from skills.top3_validation import MODES


def validate(report):
    if (report.get('schema')!='legacy_return_comparison_v1'
        or (report.get('start'),report.get('end'),report.get('initial_cash'),report.get('position_count'))!=('2024-01-02','2026-10-02',1_000_000,3)
        or (report.get('buy_fraction'),report.get('sell_fraction'))!=(.5,.3)
        or any(report.get(k) is not False for k in ('live_qualified','unseen_validation','actual_fill_verified','paper_enabled'))
        or set(report.get('cases',{}))!={f'{a}:{m}' for a in ARMS for m in MODES}
        or report.get('modes')!={k:dict(commission=v[0],slippage=v[1]) for k,v in MODES.items()}):
        raise ValueError('舊策略比較格式或研究資格不符')
    complete=0
    for key,row in report['cases'].items():
        if key!=f"{row['id']}:{row['mode']}" or type(row.get('completed')) is not bool:
            raise ValueError('比較案例識別不符')
        if row['completed']:
            complete+=1;metrics=row['metrics']
            if (metrics['trading_days']!=666 or metrics['initial_cash']!=1_000_000
                or any(not math.isfinite(metrics[k]) for k in ('total_return','max_drawdown','final_nav'))
                or any(row[k]!=metrics[k] for k in ('total_return','max_drawdown'))):
                raise ValueError('比較帳戶期間、資金或績效不符')
        elif row.get('metrics') is not None or row.get('summary') is not None:
            raise ValueError('未完成帳戶不得列完整報酬')
        base=report['cases'][f"benchmark:{row['mode']}"]
        comparable=row['completed'] and base['completed']
        if row['benchmark_comparable'] is not comparable or comparable and row['excess_return']!=row['total_return']-base['total_return']:
            raise ValueError('0050比較不一致')
    if report['completed_cases']!=complete or report['all_completed'] is not (complete==12):
        raise ValueError('比較完成數不符')
    for mode in MODES:
        if report['cases'][f'benchmark:{mode}']['completed'] and report.get('benchmark_parity',{}).get(mode,{}).get('all_exact') is not True:
            raise ValueError('缺少0050同成本帳本核對')
    return report


def case_detail(report,case_id,root):
    validate(report)
    rows={f"{r['id']}_{r['mode']}":r for r in report['cases'].values()}
    row=rows.get(case_id)
    if not row or not row['completed']:raise ValueError('此帳戶尚未完成')
    root=Path(root).resolve();path=(root/row['path']).resolve()
    if not path.is_relative_to(root) or path.suffix!='.json':raise ValueError('帳戶路徑錯誤')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('帳戶指紋不符')
    case=json.loads(raw);account=case['account'];cfg=case['legacy_comparison'];settings=account['settings']
    if (case['completed'] is not True or case['summary']!=row['summary'] or len(account['daily'])!=666
        or (account['daily'][0]['date'],account['daily'][-1]['date'])!=('2024-01-02','2026-10-02')
        or cfg['mode']!=row['mode'] or (cfg['buy_fraction'],cfg['sell_fraction'])!=(.5,.3)
        or (settings['commission'],settings['slippage'])!=MODES[row['mode']]):
        raise ValueError('帳戶設定不符')
    return dict(id=case_id,name=row['name']+'・'+row['mode'],initial_cash=1_000_000,
        position_count=1 if row['id']=='benchmark' else 3,position_scope='active_stock_slots',
        daily=[{k:r.get(k) for k in ('date','nav','cash','market_value','receivable','drawdown','holdings')} for r in account['daily']],
        trades=[dict({k:r.get(k) for k in ('date','signal_date','stock_id','name','side','qty','reason','total_cost','cash_after')},price=r['reference_price']) for r in account['trades']],
        source_sha256=row['sha256'],live_qualified=False,actual_fill_verified=False,
        note='買HL2／賣30%；各自原選股與出場規則，已見歷史日區間代理。')
