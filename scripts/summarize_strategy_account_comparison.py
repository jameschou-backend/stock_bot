#!/usr/bin/env python3
"""Independently recount the registered comparison, including failed arms."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_poc_broker_account import Sources, account_metrics, read, sha
from skills.account_cohort_attribution import cohort_outcomes

REPORTING_SOURCES = {name: sha(ROOT/name) for name in (
    'scripts/summarize_strategy_account_comparison.py',
    'scripts/summarize_poc_broker_account.py', 'skills/account_cohort_attribution.py')}


def verify_reporting_sources():
    if any(sha(ROOT/name) != digest for name, digest in REPORTING_SOURCES.items()):
        raise ValueError('Reporting code changed after import')


LABELS = {
    'red_original': '突破＋紅 K，原排序',
    'poc_priority': '原 POC 優先＋紅 K',
    'poc_filter': '只買 POC 上移＋紅 K',
    'red_known': 'POC 已知子集，原排序',
    'poc_priority_known': 'POC 已知子集，上移優先',
    'rsi_shared_exit': 'RSI 收復 30，共同出場',
    'rsi_time20': 'RSI 收復 30，持有 20 日',
    'benchmark': '0050 含息基準',
}
OLD_REPORT = '.cache/poc-range-first-20261005/online-v2/report.json'
OLD_SHA = '637071e14cf08bf06123d919bcc97b43b929f03bd12fece22f655fc9acc35594'
PARITY_FIELDS = ('daily', 'trades', 'cash_ledger', 'corporate_actions', 'holdings',
                 'receivables', 'cohorts', 'resource_plans', 'ending_inventory')


def describe_cohorts(case):
    rows = list(cohort_outcomes(case).values())
    settled = [r for r in rows if r['settled']]
    wins = [r for r in settled if r['pnl'] > .005]
    losses = [r for r in settled if r['pnl'] < -.005]
    mean = lambda values: sum(values) / len(values) if values else None
    returns = [r['return_on_cost'] for r in settled]
    profit, loss = sum(r['pnl'] for r in wins), -sum(r['pnl'] for r in losses)
    return dict(funded=len(rows), settled=len(settled), wins=len(wins), losses=len(losses),
                flat=len(settled)-len(wins)-len(losses), unresolved=len(rows)-len(settled),
                win_rate=len(wins)/len(settled) if settled else None,
                average_return_on_cost=mean(returns),
                average_winner_return=mean([r['return_on_cost'] for r in wins]),
                average_loser_return=mean([r['return_on_cost'] for r in losses]),
                profit_factor=profit/loss if loss else None,
                cohort_profit_reconciled=True)


def describe_benchmark(case):
    """Buy-and-hold has no settled trade sample to compare with stock exits."""
    events = {t['event_id'] for t in case['account']['trades'] if t['side'] == 'buy'}
    return dict(funded=len(events), settled=0, wins=0, losses=0, flat=0,
                unresolved=len(events), win_rate=None, average_return_on_cost=None,
                average_winner_return=None, average_loser_return=None, profit_factor=None,
                cohort_profit_reconciled=False, reason='buy_and_hold_win_rate_not_applicable')


def pending_cash_claims(case):
    """Disclose unconfirmed payments without changing the sealed account ledger."""
    rights = [r for r in case['summary']['final_receivables']
              if r['kind'] == 'cash' and r.get('pay_date') is None]
    gross = sum(r['amount'] for r in rights)
    nav, initial = case['summary']['final_nav'], case['summary']['initial_cash']
    return dict(undated_cash_rights=rights, gross_unavailable_cash=gross,
                available_cash_increment=0, has_unconfirmed_payment_claims=bool(rights),
                nav_if_these_claims_pay_zero=nav-gross,
                nav_using_recorded_gross_claims=nav,
                total_return_if_these_claims_pay_zero=(nav-gross)/initial-1,
                total_return_using_recorded_gross_claims=nav/initial-1,
                interpretation='Endpoint valuation sensitivity only; not a confidence interval, '
                               'not an alternative execution replay, and not certification of net NAV.')


def parity(old, current):
    if not old.get('completed') or not current.get('completed'):
        return dict(complete=False, all_exact=False, reason='incomplete_account')
    fields = {k: old['account'][k] == current['account'][k] for k in PARITY_FIELDS}
    return dict(complete=True, all_exact=all(fields.values()) and old['summary'] == current['summary'],
                fields=fields, summary_exact=old['summary'] == current['summary'])


def summarize(report_path, *, root=ROOT):
    verify_reporting_sources()
    root = Path(root).resolve()
    report_path = Path(report_path).resolve()
    sources = Sources(root)
    name = str(report_path.relative_to(root))
    expected = (report_path.parent/'report.sha256').read_text().strip()
    report = read(sources.bind(name, expected))
    if (report['start'], report['end'], report['initial_cash']) != ('2024-01-02', '2026-10-02', 1_000_000):
        raise ValueError('Comparison period or capital differs')
    if set(report['registered_arms']) != set(LABELS):
        raise ValueError('Registered comparison arms differ')
    if any(report.get(flag) is not False for flag in ('live_qualified', 'actual_fill_verified', 'unseen_validation')):
        raise ValueError('Research qualification labels differ')
    sources.closure(report, name)
    cases, rows, calendar = {}, {}, None
    for arm in LABELS:
        item = report['cases'].get(arm)
        if item is None:
            rows[arm] = dict(label=LABELS[arm], completed=False, reason='not_run', metrics=None)
            continue
        case = read(sources.bind(item['path'], item['sha256']))
        if case['completed'] != item['completed']:
            raise ValueError('Case completion differs from report')
        cases[arm] = case
        row = dict(label=LABELS[arm], completed=case['completed'], path=item['path'], sha256=item['sha256'])
        rows[arm] = row
        if not case['completed']:
            if case.get('summary') is not None:
                raise ValueError('Incomplete account must not publish a full-period return')
            row.update(metrics=None, reason=case.get('reason'), last_date=case.get('last_date'),
                       completed_sessions=case.get('completed_sessions'))
            continue
        metrics = account_metrics(case)
        if metrics['initial_cash'] != report['initial_cash']:
            raise ValueError('Account capital differs from the comparison report')
        dates = [r['date'] for r in case['account']['daily']]
        if len(dates) != 666 or (dates[0], dates[-1]) != ('2024-01-02', '2026-10-02'):
            raise ValueError('Incomplete comparison calendar')
        if calendar is not None and calendar != dates:
            raise ValueError('Arms use different market calendars')
        calendar = dates
        metrics['mean_daily_cash_weight'] = sum(r['cash']/r['nav'] for r in case['account']['daily'])/len(dates)
        row.update(metrics=metrics, cohorts=describe_benchmark(case) if arm == 'benchmark' else describe_cohorts(case),
                   pending_cash_valuation=pending_cash_claims(case),
                   profile_queries=len(case.get('profile_queries', [])),
                   profile_quality_exclusions=len(case.get('profile_data_exclusions', [])),
                   selection_fallback_days=case.get('selection_fallback_days', 0),
                   execution_exclusion_reasons=dict(Counter(r['failure_reason'] for r in case.get('data_gap_exclusions', []))))
    benchmark = rows['benchmark'].get('metrics')
    for row in rows.values():
        if row.get('metrics') and benchmark:
            row['metrics']['excess_total_return_pp'] = 100*(row['metrics']['total_return']-benchmark['total_return'])
    baseline = read(sources.bind(OLD_REPORT, OLD_SHA))
    sources.closure(baseline, OLD_REPORT)
    old = baseline['cases']['poc_range70_30_all']
    old_case = read(sources.bind(old['path'], old['sha256']))
    original_parity = parity(old_case, cases.get('poc_priority', {}))
    # Read boundaries independently from the engine and then check verified
    # file fingerprints again; an input changing mid-report cannot be sealed.
    for source, digest in tuple(sources.used.items()):
        sources.bind(source, digest)
    verify_reporting_sources()
    return dict(schema='strategy_account_comparison_summary_v1', start=report['start'], end=report['end'],
                initial_cash=1_000_000, cases=rows,
                all_registered_completed=all(row['completed'] for row in rows.values()),
                original_poc_parity=original_parity,
                main_execution='buy L+0.7*(H-L); sell L+0.3*(H-L); channel-specific range proxies',
                live_qualified=False, actual_fill_verified=False, unseen_validation=False,
                historical_period_already_researched=True,
                historical_universe_certified=False, full_market_poc_signal_study=False,
                source_report=dict(path=name, sha256=expected),
                sealed_original_report=dict(path=OLD_REPORT, sha256=OLD_SHA),
                checked_source_count=len(sources.used),
                reporting_sources=REPORTING_SOURCES,
                network_requests=report.get('network_requests'), finmind_requests=report.get('finmind_requests'))


def markdown(result):
    lines = ['# POC 與 RSI：同條件完整帳戶比較', '',
        '固定期間：2024/1/2～2026/10/2，666 個交易日。本金 100 萬，收益再投入，三個活躍個股名額；閒置資金留現金。', '',
        '全部組別使用同一手續費、稅、滑價、整零股容量、公司行動及 T 收盤→T+1 規則。買價為日低＋70% 日振幅，賣價為日低＋30% 日振幅；這是事後日區間成交代理，不代表能按此價實際成交。', '',
        '| 組別 | 完成 | 累積報酬估值 | 期末資產估值 | 最大回撤 | 已結清勝率 | 結清／進場批次 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm, row in result['cases'].items():
        if not row['completed']:
            lines.append(f"| {row['label']} | 未完成 | — | — | — | — | — |")
            continue
        m, c = row['metrics'], row['cohorts']
        win = f"{100*c['win_rate']:.2f}%" if c['win_rate'] is not None else '—'
        lines.append(f"| {row['label']} | 是 | {100*m['total_return']:+.2f}% | {m['final_nav']:,.2f} | {100*m['max_drawdown']:.2f}% | {win} | {c['settled']}／{c['funded']} |")
    for row in result['cases'].values():
        pending = row.get('pending_cash_valuation', {})
        if pending.get('gross_unavailable_cash', 0):
            lines += ['', f"估值待核：{row['label']} 仍有 {pending['gross_unavailable_cash']:,.2f} 元無確認支付日的應收，未用於買進，相關批次不算已結清。若該項淨額為零，期末資產為 {pending['nav_if_these_claims_pay_zero']:,.2f} 元；按帳載毛額估值為 {pending['nav_using_recorded_gross_claims']:,.2f} 元。這只是該款項的期末敏感度，並非完整淨資產認證。"]
    lines += ['', '勝率以扣成本後已結清的訊號批次計算；整股、零股或部分成交不重複計筆。期末持股與未到帳應收仍納入資產，不混入已結清勝率。0050 為同一個買進持有計畫，包含分批成交及股息再投入，勝率沒有與多筆已結清個股交易可比的統計意義。', '',
              '| 組別 | 2024 | 2025 | 2026 至 10/2 | 平均現金占比 |', '|---|---:|---:|---:|---:|']
    for row in result['cases'].values():
        if row['completed']:
            m = row['metrics']
            values = ' | '.join(f"{100*a['total_return']:+.2f}%" for a in m['annual'])
            lines.append(f"| {row['label']} | {values} | {100*m['mean_daily_cash_weight']:.2f}% |")
    lines += ['', f"原 POC 重現：`{result['original_poc_parity']['all_exact']}`；逐日資產、交易、現金帳、公司行動與持股均逐項比較。", '',
        'POC 優先允許未上移候選遞補；POC 硬篩只買已知上移者。已知子集兩組使用相同的可用性規則，分辨資料排除和排序效果。必要資料尚未取得不能當成不買，該組必須列未完成。', '',
        '| 組別 | POC 查核 | 品質未知排除 | 原排序回退日 |', '|---|---:|---:|---:|']
    for row in result['cases'].values():
        if row['completed']:
            lines.append(f"| {row['label']} | {row['profile_queries']} | {row['profile_quality_exclusions']} | {row['selection_fallback_days']} |")
    for row in result['cases'].values():
        if not row['completed']:
            lines += ['', f"未完成：{row['label']}；最後完成日 {row.get('last_date')}；{row.get('reason')}。"]
    lines += ['', '此區間已反覆研究，不是全新樣本外測試；歷史股票名單與實際成交未完整認證。全部組別仍為研究版本，`live_qualified=false`。全訊號平均報酬研究和三檔帳戶結果不能互換。', '',
              f"完整封存報告：`{result['source_report']['path']}`；SHA256 `{result['source_report']['sha256']}`。", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--markdown', required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    args.output.with_suffix(args.output.suffix+'.sha256').write_text(sha(args.output)+'\n')
    args.markdown.write_text(markdown(result))
    print(json.dumps(dict(completed=result['all_registered_completed'],
                         original_poc_exact=result['original_poc_parity']['all_exact'],
                         source_count=result['checked_source_count']), ensure_ascii=False))


if __name__ == '__main__':
    main()
