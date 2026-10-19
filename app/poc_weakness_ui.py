"""Fail closed when a descriptive warning study looks like a new backtest."""
from collections import Counter
import math


def validate(report):
    if (report.get('schema'), report.get('strategy_id'), report.get('mode'), report.get('start'),
            report.get('end'), report.get('total_cohorts'), report.get('initial_cash')) != (
            'poc_weakness_v1', 'poc_up_red', 'baseline', '2024-01-02', '2026-10-02', 73, 1000000):
        raise ValueError('轉弱診斷的原版範圍不符')
    if (report.get('backtests_run'), report.get('new_accounts'), report.get('reused_accounts'),
            report.get('finmind_requests')) != (0, 0, 1, 0):
        raise ValueError('轉弱診斷不可重跑帳戶')
    if report.get('descriptive_only') is not True or any(report.get(k) is not False for k in (
            'counterfactual_backtest', 'strategy_changed', 'actual_fill_verified',
            'unseen_validation', 'paper_enabled', 'live_qualified')):
        raise ValueError('轉弱診斷不可變更交易資格')
    keys = {'two_black', 'below_ma20_two', 'two_black_below_ma20'}
    cohorts = report['cohorts']
    if len(cohorts) != 73 or len({c['event_id'] for c in cohorts}) != 73 or set(report['rules']) != keys:
        raise ValueError('轉弱診斷批次或規則不完整')
    for c in cohorts:
        for key, w in c['warnings'].items():
            if key not in keys or not c['entry_date'] < w['date'] < w['next_session'] <= (c['original_decision'] or report['end']):
                raise ValueError('警訊使用原出場決策當日或未來資訊')
            if c['original_decision'] and (w['lead_sessions'] < 1 or w['outcome_censored']):
                raise ValueError('提前時間或設限狀態錯誤')
            if not math.isfinite(w['close_return_from_entry']) or w['close_return_from_entry'] <= 0:
                raise ValueError('減碼候選不符合既定價格獲利篩選')
    for key, s in report['rule_summaries'].items():
        fired = [c for c in cohorts if key in c['warnings']]
        resolved = [c for c in fired if c['original_decision']]
        if (s['cohorts'], s['triggered'], s['resolved'], s['censored']) != (73, len(fired), len(resolved), len(fired)-len(resolved)):
            raise ValueError('警訊統計與批次不符')
        if s['exit_reasons'] != dict(Counter(c['original_reason'] for c in resolved)):
            raise ValueError('警訊出場原因統計不符')
        for field, predicate in (
            ('subsequent_higher_close', lambda w:w['subsequent_max_return'] > 1e-12),
            ('subsequent_new_high', lambda w:w['later_new_holding_high']),
            ('original_decision_higher', lambda w:w['return_to_original_decision'] > 1e-12),
            ('original_decision_lower', lambda w:w['return_to_original_decision'] < -1e-12)):
            if s[field] != sum(predicate(c['warnings'][key]) for c in resolved):
                raise ValueError('警訊後走勢統計不符')
    if [e['rank'] for e in report['episodes']] != list(range(1, 6)):
        raise ValueError('五段回撤未全部完成')
    return report
