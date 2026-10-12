#!/usr/bin/env python3
"""Verify every replay and expose contaminated-history diagnostics as such."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_poc_broker_account import Sources, account_metrics
from scripts.summarize_all_entry_accounts import guard_certificate_inventory
from skills.all_entry_accounts import digest, registry, write
from skills.top3_validation import ARMS, MODES, qualification, verify_costs


def windows(daily, benchmark, horizon=252):
    if [d['date'] for d in daily] != [d['date'] for d in benchmark]:
        raise ValueError('Benchmark calendar differs')
    nav = [daily[0]['opening_nav']] + [d['nav'] for d in daily]
    base = [benchmark[0]['opening_nav']] + [d['nav'] for d in benchmark]
    excess = [(nav[i]/nav[i-horizon]-1)-(base[i]/base[i-horizon]-1)
              for i in range(horizon, len(nav))]
    return dict(horizon=horizon, windows=len(excess), overlapping=True, unseen=False,
        win_fraction=sum(x>0 for x in excess)/len(excess) if excess else None,
        worst_excess=min(excess) if excess else None, best_excess=max(excess) if excess else None)


def prior_publication(sources, name, expected=None):
    """Bind a historical published number, without re-certifying its old model."""
    path = ROOT/'artifacts/forward_simulation'/f'{name}.json'
    if expected is None:
        expected = path.with_suffix('.sha256').read_text().split()[0]
    data = json.loads(sources.bind(str(path.relative_to(ROOT)), expected).read_text())
    return data, dict(path=str(path.relative_to(ROOT)), sha256=expected)


def trial_accounting(sources, reports):
    refs, attempted = [], []
    for report in reports:
        path = ROOT/report['path']
        trial_path = path.parent/'trials.jsonl'
        expected = digest(trial_path)
        bound = sources.bind(str(trial_path.relative_to(ROOT)), expected)
        rows = [json.loads(line) for line in bound.read_text().splitlines() if line]
        if (len(rows) != 4 or {r['params']['arm'] for r in rows} != set(ARMS)
                or any(r.get('top3_mode') != report['mode'] for r in rows)):
            raise ValueError('Each fixed cost mode must preserve all four trial records')
        attempted.extend(rows)
        refs.append(dict(path=str(trial_path.relative_to(ROOT)), sha256=expected))
    failures = []
    for name in ('development-trials.jsonl', 'combined-v2/trials.jsonl'):
        path = ROOT/'.cache/top3-validation-20261012'/name
        expected = digest(path)
        rows = [json.loads(line) for line in sources.bind(str(path.relative_to(ROOT)), expected).read_text().splitlines() if line]
        if any(row.get('completed') is not False for row in rows):
            raise ValueError('Development attempt classification differs')
        failures.extend(rows)
        refs.append(dict(path=str(path.relative_to(ROOT)), sha256=expected))
    return dict(published_attempts=len(attempted), development_incomplete_attempts=len(failures),
        total_recorded_attempts=len(attempted)+len(failures),
        development_attempts=failures, sources=refs,
        note='含核帳失敗與中止試次；不是19個獨立策略，亦未重設既有全域試驗紀錄。')


def summarize(paths, output):
    sources = Sources(ROOT)
    names = {r['id']: r['name'] for r in registry()} | {'benchmark':'0050 含息基準'}
    cases, accounts, report_refs = {}, {}, []
    for path in map(lambda p: Path(p).resolve(), paths):
        expected = path.with_suffix('.sha256').read_text().strip()
        report = json.loads(sources.bind(str(path.relative_to(ROOT)), expected).read_text())
        sources.closure(report, str(path.relative_to(ROOT)))
        mode = report['top3_validation']['mode']
        if mode not in MODES or (report['start'], report['end'], report['initial_cash']) != ('2024-01-02','2026-10-02',1_000_000):
            raise ValueError('Study scope differs')
        report_refs.append(dict(path=str(path.relative_to(ROOT)), sha256=expected, mode=mode))
        for arm, descriptor in report['cases'].items():
            if arm not in ARMS or f'{arm}:{mode}' in cases:
                raise ValueError('Unexpected or duplicate arm/stress; select attempts explicitly')
            case = json.loads(sources.bind(descriptor['path'], descriptor['sha256']).read_text())
            if case.get('top3_validation') != report['top3_validation']:
                raise ValueError('Case cost/price configuration differs')
            row = dict(id=arm, mode=mode, name=names[arm], completed=case['completed'],
                path=descriptor['path'], sha256=descriptor['sha256'], summary=case.get('summary'),
                reason=case.get('reason'), last_date=case.get('last_date'), metrics=None,
                settlement_evidence_notes=case.get('settlement_evidence_notes', []),
                data_gap_exclusions=len(case.get('data_gap_exclusions', [])))
            if case['completed']:
                guard_certificate_inventory(case)
                verify_costs(case['account'], mode)
                metrics = account_metrics(case)
                if metrics['trading_days'] != 666 or (metrics['start'], metrics['end']) != ('2024-01-02','2026-10-02'):
                    raise ValueError('Truncated replay cannot be published as complete')
                row.update(metrics=metrics, **{k:metrics[k] for k in ('total_return','max_drawdown')})
                accounts[f'{arm}:{mode}'] = case['account']['daily']
            cases[f'{arm}:{mode}'] = row
    if set(cases) != {f'{arm}:{mode}' for arm in ARMS for mode in MODES}:
        raise ValueError('Every fixed strategy and stress needs an attempted result')
    gates, diagnostics = {}, {}
    for arm in ARMS:
        rows = {}
        for mode in MODES:
            row, benchmark = cases[f'{arm}:{mode}'], cases[f'benchmark:{mode}']
            comparable = row['completed'] and benchmark['completed']
            row['benchmark_comparable'] = comparable
            if comparable:
                row['excess_return'] = row['total_return'] - benchmark['total_return']
                row['benchmark_max_drawdown'] = benchmark['max_drawdown']
                row['rolling252'] = windows(accounts[f'{arm}:{mode}'], accounts[f'benchmark:{mode}'])
                row['annual_excess'] = [dict(year=a['year'], total_return=a['total_return'],
                    benchmark_return=b['total_return'], excess=a['total_return']-b['total_return'], unseen=False)
                    for a,b in zip(row['metrics']['annual'], benchmark['metrics']['annual'])]
            rows[mode] = dict(row, completed=comparable)
        if arm != 'benchmark':
            gates[arm] = qualification(rows)
            completed = [r for r in rows.values() if r['completed']]
            diagnostics[arm] = dict(name=names[arm], compared_stresses=len(completed),
                positive_stresses=sum(r['total_return']>0 for r in completed),
                benchmark_wins=sum(r['excess_return']>0 for r in completed),
                drawdown_not_worse=sum(r['max_drawdown']>=r['benchmark_max_drawdown'] for r in completed),
                worst_return=min((r['total_return'] for r in completed), default=None),
                worst_drawdown=min((r['max_drawdown'] for r in completed), default=None))
    old, old_ref = prior_publication(sources, 'all_entry_accounts_20261011')
    # The 2019 publication predates sidecar receipts; pin its existing bytes.
    universe, universe_ref = prior_publication(sources, 'stock_universe_2019_20260929',
        '17bd7f2f0ff6ebb23fbd8b21ebad48be64ee346f21929d8650f61ee666c9b8a7')
    broker, broker_ref = prior_publication(sources, 'poc_broker_account_20261004')
    ranges, range_ref = prior_publication(sources, 'poc_range_first_account_20261005')
    universe_row = next(r for r in universe['comparison'] if r['arm'] == 'liquid_universe')
    reconciliation = [
        dict(name='擴大流動股池（非 POC＋紅 K）', start=universe['start'], end=universe['end'],
             return_pct=universe_row['total_return']*100, publication=universe_ref,
             source='docs/research_stock_universe_2019_20260929.md',
             difference='較長期間、60日突破與大盤條件；不是目前 POC 硬篩選；舊 HL2 帳戶。'),
        dict(name='舊 POC 優先＋紅 K', start=broker['start'], end=broker['end'],
             return_pct=broker['arms']['poc_red']['metrics']['total_return']*100, publication=broker_ref,
             source='docs/research_poc_broker_account_20261004.md',
             difference='POC 優先排序、品質未知有明示原排序政策，舊 HL2 與成交容量。'),
        dict(name='POC 優先＋紅 K・重算日容量／中間價', start=ranges['start'], end=ranges['end'],
             return_pct=ranges['cases']['poc_range50_all']['summary']['total_return']*100, publication=range_ref,
             source='docs/research_poc_range_first_20261005.md',
             difference='完整普通盤日容量、整單資料排除政策；買賣均為 HL2。'),
        dict(name='31組比較・POC 硬篩選', start=old['start'], end=old['end'],
             return_pct=old['cases']['poc_up_red']['metrics']['total_return']*100, publication=old_ref,
             source='docs/research_all_entry_accounts_20261011.md',
             difference='POC 必須成立、同日均成交值排序；買70／賣30。'),
    ]
    for row in reconciliation:
        sources.bind(row['source'], digest(ROOT/row['source']))
    status_path = ROOT/'.cache/top3-validation-20261012/data-status-start.json'
    status_hash = digest(status_path)
    status = json.loads(sources.bind(str(status_path.relative_to(ROOT)), status_hash).read_text())
    trials = trial_accounting(sources, report_refs)
    result = dict(schema='top3_validation_publication_v1', start='2024-01-02', end='2026-10-02',
        initial_cash=1_000_000, position_count=3, created_at=datetime.now(timezone.utc).isoformat(),
        cases=cases, qualification=gates, diagnostics=diagnostics,
        modes={k:dict(commission=v[0], slippage=v[1]) for k,v in MODES.items()},
        buy_fraction=.5, sell_fraction=.3, live_qualified=False, unseen_validation=False,
        actual_fill_verified=False, all_completed=all(r['completed'] for r in cases.values()),
        completed_cases=sum(r['completed'] for r in cases.values()), expected_cases=16,
        out_of_sample=dict(status='blocked_no_unseen_period', sessions=0, settled_cohorts=0,
            strategy_selection_date='2026-10-12', earliest_forward_signal_date='2026-10-12',
            minimum_sessions=126, minimum_settled_cohorts=30,
            data_status=dict(path=str(status_path.relative_to(ROOT)), sha256=status_hash,
                latest_price_date=status['price_date'], observed_at=status['observed_at']),
            explanation='歷史期間已用來選出前三名；年切分與滾動窗口皆已見資料。版本固定後尚無合格未見帳戶紀錄。'),
        paper=dict(enabled_strategies=[arm for arm,g in gates.items() if g['paper_enabled']],
            status='blocked_pending_unseen_and_causal_execution', simulated_fills_written=0,
            explanation='未通過者不接紙上成交；HL2 只能事後計算。原 A 線訊號價 NAV 與本研究分開。'),
        poc_reconciliation=reconciliation,
        trial_accounting=trials,
        poc_note='目前找到的接近900%封存結果為擴大流動股池 +949.76%；未找到同期間、同定義 POC 硬篩選的900%結果。各版本同時改動規則／期間／成交模型，不能把差額全部歸因於買價。',
        historical_selection=dict(publication=old_ref['path'], sha256=old_ref['sha256'],
            old_returns={arm:old['cases'][arm]['metrics']['total_return'] for arm in ARMS}),
        limitations=['HL2 為事後日區間代理，不能保證指定價成交；賣價仍為30%區間價。',
            '歷史全市場名單、首次發布與修訂時點未完整認證；資料排除與公告來源限制仍存在。',
            '重疊窗口不是獨立樣本；選出前三名使用的全部歷史不能再稱未見樣本外。'],
        reports=report_refs, checked_source_count=len(sources.used),
        reporting_sources={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'skills/top3_validation.py')})
    write(Path(output), result)
    Path(output).with_suffix('.sha256').write_text(digest(Path(output))+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    result=summarize(args.report,args.output)
    print(json.dumps({key:result[key] for key in ('completed_cases','checked_source_count','qualification')}))
