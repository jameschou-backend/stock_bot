"""Validate the scope and additive identities of the ledger-only diagnosis."""
import math

IDENTIFIER = 'poc_drawdown_attribution_20261019'
COMPONENTS = ('winner_decline', 'other_existing_decline', 'new_entry_decline', 'gain_offset', 'cost')


def validate(report):
    def same(a, b):
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (a, b)) or abs(a-b) > .011:
            raise ValueError('回撤拆解加總或數字不符')
    if (report.get('schema'), report.get('strategy_id'), report.get('mode'), report.get('start'),
            report.get('end'), report.get('initial_cash'), report.get('trading_days')) != (
            'poc_drawdown_attribution_v1', 'poc_up_red', 'baseline', '2024-01-02', '2026-10-02', 1000000, 666):
        raise ValueError('回撤拆解的原版帳戶範圍不符')
    if any(report.get(k) is not False for k in ('actual_fill_verified', 'unseen_validation', 'paper_enabled',
            'live_qualified', 'strategy_changed', 'counterfactual_backtest')) or report.get('descriptive_only') is not True:
        raise ValueError('回撤拆解不可更動策略或交易資格')
    if (report.get('backtests_run'), report.get('new_accounts'), report.get('reused_accounts'),
            report.get('finmind_requests')) != (0, 0, 2, 0):
        raise ValueError('回撤拆解必須只重用既有帳本')
    if not report.get('daily_identity_reconciled') or not report.get('all_selected_boundaries_reconciled'):
        raise ValueError('回撤拆解缺少對帳')
    if report.get('method', {}).get('selection') != 'five_deepest_disjoint_high_watermark_episodes':
        raise ValueError('回撤選取方式不同')
    episodes = report.get('episodes', [])
    if len(episodes) != 5 or [e['rank'] for e in episodes] != list(range(1, 6)):
        raise ValueError('回撤段落不完整')
    for i, e in enumerate(episodes):
        if not report['start'] <= e['peak_date'] < e['trough_date'] <= report['end']:
            raise ValueError('回撤日期範圍不同')
        if e['recovery_date'] is not None and not e['trough_date'] < e['recovery_date'] <= report['end']:
            raise ValueError('回復高點日期錯誤')
        if i and e['drawdown'] < episodes[i-1]['drawdown']:
            raise ValueError('回撤排名不同')
        if set(e['components']) != set(COMPONENTS):
            raise ValueError('回撤分類缺漏')
        for key, value in e['components'].items():
            if (key == 'gain_offset' and value < 0) or (key != 'gain_offset' and value > 0):
                raise ValueError('回撤分類正負號錯誤')
            same(e['component_loss_shares'][key], value/e['net_change'])
        same(e['trough_nav']-e['peak_nav'], e['net_change'])
        same(e['drawdown'], e['net_change']/e['peak_nav'])
        same(sum(e['components'].values()), e['net_change'])
        same(sum(e['costs'].values()), -e['components']['cost'])
        same(e['winner_giveback']+e['winner_beyond_peak_profit'], -e['components']['winner_decline'])
        same(e['reconciliation_error'], 0)
        same(sum(c['net_pnl'] for c in e['cohorts']), e['net_change'])
        same(sum(c['cost'] for c in e['cohorts']), -e['components']['cost'])
        j = e['journal_components']
        same(j['market_pnl']+j['dividend_entitlement']+j['execution_basis_pnl']-j['cost'], e['net_change'])
        for c in e['cohorts']:
            same(c['gross_pnl']-c['cost'], c['net_pnl'])
            same(c['market_value_change']+c['cash_change']+c['receivable_change'], c['net_pnl'])
    chronological = sorted(episodes, key=lambda e:e['peak_date'])
    for left, right in zip(chronological, chronological[1:]):
        if left['recovery_date'] is None or left['recovery_date'] > right['peak_date']:
            raise ValueError('回撤段落重疊')
    same(episodes[0]['drawdown'], report['summary']['max_drawdown'])
    return report
