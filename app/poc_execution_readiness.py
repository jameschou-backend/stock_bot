"""Execution preflight is not an executable return or paper-trading approval."""
import math
from skills.top3_validation import MODES

IDENTIFIER = 'poc_execution_readiness_20261014'


def validate(report, publication_sha=None):
    if (report.get('schema') != 'poc_execution_readiness_v1'
            or report.get('status') != 'blocked_missing_intraday_odd_sequence'
            or report.get('audit_completed') is not True
            or report.get('rerun_accounts') != 0
            or report.get('preserved_completed_accounts') != 36
            or (report.get('start'),report.get('end'),report.get('initial_cash')) != ('2024-01-02','2026-10-02',1000000)
            or any(report.get(k) is not False for k in ('execution_validation_completed','whole_account_replayed',
                'causal_fill_verified','actual_fill_verified','paper_enabled','unseen_validation','live_qualified'))
            or report.get('modes') != {k:dict(commission=v[0],slippage=v[1]) for k,v in MODES.items()}
            or (publication_sha is not None and report['source_publication']['sha256'] != publication_sha)):
        raise ValueError('成交缺件檢查不可冒充已完成可執行驗證')
    expected = {f'{a}:original:{m}' for a in ('poc_up_red','poc_priority','benchmark') for m in MODES}
    cases = report.get('cases',{})
    if set(cases) != expected:
        raise ValueError('兩套原版與同成本0050必須完整配對')
    for key,row in cases.items():
        base = cases[f"benchmark:original:{row['mode']}"]
        if (key != f"{row['arm']}:original:{row['mode']}"
                or any(row.get(k) is not False for k in ('whole_account_replayed','causal_fill_verified','paper_enabled'))
                or any(row.get(k) is not None for k in ('executable_total_return','executable_excess_return'))
                or any(type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) for k in
                    ('proxy_total_return','proxy_max_drawdown','proxy_excess_return','benchmark_proxy_return','benchmark_proxy_max_drawdown'))
                or row['benchmark_proxy_return'] != base['proxy_total_return']
                or row['benchmark_proxy_max_drawdown'] != base['proxy_max_drawdown']
                or row['proxy_excess_return'] != row['proxy_total_return']-base['proxy_total_return']):
            raise ValueError('缺資料時不得提供可執行報酬或混用成本基準')
    odd,board,finmind=report['odd'],report['ordinary'],report['finmind']
    if (odd['required_coordinates'] != len(odd['coordinates'])
            or odd['required_coordinates'] != len({(r['stock_id'],r['date']) for r in odd['coordinates']})
            or odd['verified_chronological_coordinates'] != 0
            or board['available_tapes'] != board['unique_coordinates']
            or board['unique_coordinates'] != len(board['coordinates'])
            or board['totals_reconciled'] != sum(r['daily_totals_reconciled'] for r in board['coordinates'])
            or board['complete_exchange_sequence_verified'] is not False
            or finmind['supported_odd_dataset_found'] is not False):
        raise ValueError('成交資料覆蓋標示不符')
    return report
