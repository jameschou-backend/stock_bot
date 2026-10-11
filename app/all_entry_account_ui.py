"""Read-only presentation of a hash-bound, common-model account comparison."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = 'all_entry_accounts_20261011'


def validate(report):
    from skills.all_entry_accounts import registry
    expected = {r['id'] for r in registry()} | {'benchmark'}
    if (report.get('schema') != 'all_entry_account_comparison_publication_v1'
            or report.get('strategy_count') != 31 or set(report.get('cases',{})) != expected
            or any(report.get(k) is not False for k in ('live_qualified','actual_fill_verified','unseen_validation'))
            or (report.get('start'),report.get('end'),report.get('initial_cash'),report.get('position_count'))
                != ('2024-01-02','2026-10-02',1_000_000,3)):
        raise ValueError('31組帳戶比較格式或資格標示不符')
    for key,row in report['cases'].items():
        if row.get('id') != key or type(row.get('completed')) is not bool:
            raise ValueError('帳戶比較缺少明確完成狀態')
        if not row['completed'] and (row.get('metrics') is not None or row.get('rank') is not None):
            raise ValueError('未完成帳戶不得列報完整績效或名次')
        if row['completed'] and (not isinstance(row.get('metrics'),dict)
                or row.get('completed_sessions') != 666):
            raise ValueError('完整帳戶缺少共同期間績效')
    complete = [k for k,r in report['cases'].items() if k != 'benchmark' and r['completed']]
    if report['completed_strategies'] != len(complete) or set(report['ranking']) != set(complete):
        raise ValueError('帳戶完成數與排名不符')
    ranking = sorted(complete,key=lambda k:(-report['cases'][k]['metrics']['total_return'],k))
    all_complete = all(r['completed'] for r in report['cases'].values())
    if (report['ranking'] != ranking or report.get('all_completed') is not all_complete
            or report.get('best_is_global') is not all_complete
            or report.get('best_completed_strategy') != (ranking[0] if ranking else None)
            or any(report['cases'][k]['rank'] != i for i,k in enumerate(ranking,1))):
        raise ValueError('歷史名次或全體完成標示不符')
    return report


def case_detail(publication, case_id, root=ROOT):
    validate(publication)
    row = publication['cases'].get(case_id)
    if not row or row['completed'] is not True:
        raise ValueError('此帳戶尚未完成，無完整期間曲線')
    root = Path(root).resolve()
    path = (root/row['path']).resolve()
    if not path.is_relative_to(root) or path.suffix != '.json':
        raise ValueError('帳戶來源路徑錯誤')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('帳戶指紋不符，停止顯示績效')
    case = json.loads(raw)
    if (case.get('completed') is not True or case.get('summary') != row.get('summary')
            or any(case.get(k) is not False for k in ('live_qualified','actual_fill_verified','unseen_validation'))):
        raise ValueError('帳戶與發布摘要不一致')
    if case.get('settlement_evidence_notes', []) != row.get('settlement_evidence_notes', []):
        raise ValueError('公告來源限制與封存帳戶不一致')
    account = case['account']
    restrictions = [dict(stock_id=r['stock_id'], ex_date=r['date'],
        entitled_qty=r.get('entitled_qty'), whole_new_shares=r.get('whole_new_shares'),
        **r['certificate_restriction']) for r in account.get('corporate_actions', [])
        if r.get('kind') == 'stock_dividend' and r.get('certificate_restriction')]
    return dict(id=case_id,name=row['name'],summary=case['summary'],
        daily=[{k:r.get(k) for k in ('date','nav','cash','market_value','receivable','drawdown','holdings')} for r in account['daily']],
        trades=[{k:r.get(k) for k in ('date','signal_date','stock_id','name','side','qty','price','proxy_price',
            'reason','commission','tax','slippage','gross','cash_change','total_cost','cash_after','channel')} for r in account['trades']],
        annual=row['annual'],cohorts=row['cohorts'],candidate_coverage=row.get('candidate_coverage'),
        certificate_restrictions=restrictions,
        settlement_evidence_notes=case.get('settlement_evidence_notes', []),
        initial_cash=publication['initial_cash'],position_count=1 if case_id=='benchmark' else 3,
        position_scope='single_benchmark' if case_id=='benchmark' else 'active_stock_slots',
        actual_fill_verified=False,live_qualified=False,source_sha256=row['sha256'],
        note='共同資金帳戶；T+1買70%／賣30%日區間價代理，含持股與應收權益。尚未實戰認證。')
