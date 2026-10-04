#!/usr/bin/env python3
"""Read-only, hash-bound comparison of the fixed POC/broker cash accounts.

No source data is fetched and no strategy is replayed. Independent diagnostic
signal outcomes are labelled separately from funded account performance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('poc_red', 'poc_persist_guard', 'poc_combined_guard',
        'poc_known5_control', 'poc_known5_filter', 'poc_known5_combined')
START, END = '2024-01-02', '2026-10-02'
BASE_CASE = '.cache/poc-latest-20261003/run-v3/poc_red.json'
BASE_SHA = '7657f95c7633a322582fd53cc3aeddb6edb9ab07c06aae142ae4a8cd18375973'
DIAG = '.cache/broker-branch-research-20261004/final-b'
DIAG_REPORT_SHA = 'c1720205a452938c902130d325acf19775b7d6d002b3415c257c9c8d0a271010'
LABELS = dict(poc_red='原 POC＋紅K', poc_persist_guard='五日持續守門（未知保留）',
              poc_combined_guard='持續＋集中守門（未知保留）',
              poc_known5_control='完整五日子集對照', poc_known5_filter='完整子集＋持續',
              poc_known5_combined='完整子集＋持續＋集中', benchmark='0050')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Invalid finite account number')
    return value


def same(actual, expected, label, tolerance=1e-8):
    if not math.isclose(num(actual), num(expected), rel_tol=0, abs_tol=tolerance):
        raise ValueError('Accounting summary mismatch: ' + label)


class Sources:
    """Verify within one invocation; explicit bound snapshots only for code/docs."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache, self.used, self.snapshot_resolutions = {}, {}, []

    def path(self, name):
        path = (self.root / name).resolve()
        path.relative_to(self.root)
        return path

    def bind(self, name, expected):
        path = self.path(name)
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        prior = self.cache.get(path)
        if prior and prior[0] == fingerprint:
            actual = prior[1]
        else:
            actual = sha(path)
            after = path.stat()
            if fingerprint != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('Source changed while reading ' + name)
            self.cache[path] = (fingerprint, actual)
        if actual != expected:
            raise ValueError('Source hash mismatch ' + name)
        previous = self.used.get(name)
        if previous is not None and previous != expected:
            raise ValueError('Conflicting source version ' + name)
        self.used[name] = expected
        return path

    def closure(self, report, report_name):
        refs = report.get('source_sha256')
        if not refs:
            raise ValueError('Missing source closure ' + report_name)
        snapshots = report.get('source_snapshots', {})
        for name, digest in refs.items():
            snapshot = snapshots.get(name)
            # Choose a matching, explicit frozen source snapshot first. No data
            # file may silently substitute another source when its bytes change.
            if snapshot and Path(name).suffix in ('.py', '.md'):
                if snapshot['sha256'] != digest:
                    raise ValueError('Source snapshot differs from original hash')
                self.bind(snapshot['path'], digest)
                self.snapshot_resolutions.append(dict(report=report_name, original=name,
                                                      path=snapshot['path'], sha256=digest))
            else:
                self.bind(name, digest)


def account_metrics(case):
    """Recount metrics without calling the financial runner's summarize()."""
    if case.get('completed') is not True or not isinstance(case.get('summary'), dict):
        raise ValueError('Incomplete accounts have no full-period performance')
    account, summary = case['account'], case['summary']
    daily, trades = account['daily'], account['trades']
    dates = [r['date'] for r in daily]
    if not dates or dates != sorted(set(dates)):
        raise ValueError('Daily dates must be strictly increasing')
    for day in dates:
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError('Invalid account date')
    initial = num(account['settings']['initial_cash'])
    if initial <= 0:
        raise ValueError('Initial capital must be positive')
    allowed, previous, peak, mdd = set(dates), initial, initial, 0.
    costs = Counter({k: 0. for k in ('commission', 'tax', 'slippage', 'total_cost')})
    daily_cost = Counter()
    for row in trades:
        if row['date'] not in allowed or row['side'] not in ('buy', 'sell'):
            raise ValueError('Trade outside observed account dates')
        if type(row['qty']) is not int or row['qty'] <= 0:
            raise ValueError('Noninteger/nonpositive filled quantity')
        for key in costs:
            value = num(row[key])
            if value < 0:
                raise ValueError('Negative trade cost')
            costs[key] += value
        same(row['total_cost'], sum(row[k] for k in ('commission', 'tax', 'slippage')),
             'trade costs', .011)
        daily_cost[row['date']] += row['total_cost']
    annual = []
    for year in sorted({d[:4] for d in dates}):
        rows = [r for r in daily if r['date'].startswith(year)]
        year_start, year_peak, year_mdd = previous, previous, 0.
        for row in rows:
            nav = num(row['nav'])
            if nav <= 0:
                raise ValueError('Nonpositive NAV unsupported by this fixed study')
            same(row['opening_nav'], previous, 'opening NAV', .011)
            same(nav, sum(num(row[k]) for k in ('cash', 'market_value', 'receivable')), 'NAV identity', .011)
            same(row['daily_return'], nav / previous - 1, 'daily return')
            same(row['total_return'], nav / initial - 1, 'cumulative return')
            same(row['cost'], daily_cost[row['date']], 'daily trade costs', .011)
            peak = max(peak, nav)
            drawdown = nav / peak - 1
            same(row['drawdown'], drawdown, 'daily drawdown')
            mdd = min(mdd, drawdown)
            year_peak = max(year_peak, nav)
            year_mdd = min(year_mdd, nav / year_peak - 1)
            previous = nav
        annual.append(dict(year=year, start_nav=year_start, end_nav=previous,
                           profit=previous-year_start, total_return=previous/year_start-1,
                           max_drawdown=year_mdd, partial_year=year == END[:4]))
    last = daily[-1]
    metrics = dict(start=dates[0], end=dates[-1], initial_cash=initial,
                   final_nav=last['nav'], profit=last['nav']-initial,
                   total_return=last['nav']/initial-1, max_drawdown=mdd,
                   cash=last['cash'], market_value=last['market_value'], receivable=last['receivable'],
                   trading_days=len(dates), trade_count=len(trades),
                   buy_count=sum(t['side'] == 'buy' for t in trades),
                   sell_count=sum(t['side'] == 'sell' for t in trades),
                   stock_cohorts=len(account['cohorts']), costs=dict(costs), annual=annual,
                   minimum_cash=min(r['cash'] for r in daily))
    for key, value in metrics.items():
        if key == 'annual':
            if len(summary[key]) != len(value):
                raise ValueError('Annual report count mismatch')
            for expected, actual in zip(summary[key], value):
                for k, v in actual.items():
                    if isinstance(v, (str, bool)):
                        if expected[k] != v:
                            raise ValueError('Annual label mismatch')
                    else:
                        same(expected[k], v, 'annual ' + k)
        elif key == 'costs':
            for k, v in value.items():
                same(summary[key][k], v, 'cost ' + k)
        elif isinstance(value, str):
            if summary[key] != value:
                raise ValueError('Account scope mismatch')
        else:
            same(summary[key], value, key)
    funded = {t['event_id'] for t in trades if t['side'] == 'buy'}
    metrics.update(funded_events=len(funded), child_fills_by_channel=dict(sorted(Counter(t['channel'] for t in trades).items())),
                   final_holding_count=last['holdings'], final_stale_holdings=last['stale_holdings'])
    return metrics


def gate_analysis(arm, case, diagnostic_rows, baseline_funded):
    """Coverage and post-hoc outcomes; never create an entry decision from P&L."""
    rows = case['broker_gate_decisions']
    source = {r['event_id']: r for r in diagnostic_rows}
    by_pair = {(r['stock_id'], r['signal_date']): r for r in diagnostic_rows}
    if len(source) != len(diagnostic_rows) or len(by_pair) != len(diagnostic_rows):
        raise ValueError('Duplicate diagnostic identity')
    funded = {t['event_id'] for t in case['account']['trades'] if t['side'] == 'buy'}
    seen, outcomes, excluded_big = set(), [], []
    for r in rows:
        eid = r['event_id']
        if eid in seen or not r['signal_date'] < r['entry_date']:
            raise ValueError('Invalid gate identity/time')
        seen.add(eid)
        if type(r['known5']) is not bool or type(r['kept']) is not bool:
            raise ValueError('Gate availability must be explicit')
        original = by_pair.get((r['stock_id'], r['signal_date']))
        if r['source_event_id'] != (original['event_id'] if original else None):
            raise ValueError('Gate evidence join differs from frozen stock/date')
        known = bool(original and original['branch']['known'] and original['persistence5']['known'])
        if r['known5'] != known:
            raise ValueError('Gate known5 differs from frozen evidence')
        persist = original['persistence5']['passed'] if known else None
        combined = persist and original['branch']['concentrated_directional'] if known else None
        if r['persistent5'] != persist or r['combined'] != combined:
            raise ValueError('Gate condition differs from frozen evidence')
        expected = (True if arm == 'poc_red' else arm.endswith('_guard') if not known else
                    True if arm == 'poc_known5_control' else combined if 'combined' in arm else persist)
        if r['kept'] != expected:
            raise ValueError('Gate violates preregistered unknown/condition policy')
        if eid in funded and not r['kept']:
            raise ValueError('Funded candidate was rejected by broker gate')
        if original is not None:
            outcome = original['outcome']
            item = dict(event_id=eid, source_event_id=original['event_id'], stock_id=r['stock_id'],
                        name=original['name'], signal_date=r['signal_date'], entry_date=r['entry_date'],
                        kept=r['kept'], rejection_reason=None if r['kept'] else r['reason'],
                        known5=r['known5'], status=outcome['status'],
                        peak_close_return=outcome.get('peak_close_return'),
                        diagnostic_net_return=outcome.get('net_return'),
                        diagnostic_exit_date=outcome.get('exit_date'),
                        baseline_funded=eid in baseline_funded, arm_funded=eid in funded)
            outcomes.append(item)
            # Closed only, matching the prior published peak20 comparison.
            if not r['kept'] and outcome['status'] == 'closed' and num(outcome['peak_close_return']) >= .20:
                excluded_big.append(item)
    if not funded <= seen:
        raise ValueError('Funded cohort lacks branch gate decision')

    def scope(selected):
        return dict(candidates=len(selected), known5=sum(r['known5'] for r in selected),
                    unknown5=sum(not r['known5'] for r in selected),
                    kept=sum(r['kept'] for r in selected), excluded=sum(not r['kept'] for r in selected),
                    unknown_kept=sum(not r['known5'] and r['kept'] for r in selected),
                    known_condition_failed=sum(r['known5'] and not r['kept'] for r in selected),
                    reasons=dict(sorted(Counter(r['reason'] for r in selected).items())),
                    unknown_reasons=dict(sorted(Counter(r['unknown_reason'] for r in selected if not r['known5']).items())))

    return dict(all=scope(rows), annual_by_entry_year={year: scope([r for r in rows if r['entry_date'].startswith(year)])
                for year in ('2024', '2025', '2026')},
                funded=scope([r for r in rows if r['event_id'] in funded]),
                diagnostic_population=dict(matched=len(outcomes),
                    statuses=dict(sorted(Counter(r['status'] for r in outcomes).items())),
                    closed=sum(r['status'] == 'closed' for r in outcomes),
                    excluded_closed_peak20=len(excluded_big),
                    excluded_closed_peak20_by_reason=dict(sorted(Counter(r['rejection_reason'] for r in excluded_big).items())),
                    excluded_closed_peak20_baseline_funded=sum(r['baseline_funded'] for r in excluded_big),
                    definition='Prior diagnostic closed paths with peak adjusted close / assumed entry - 1 >= 20%; not account P&L or market-wide recall'),
                excluded_big_winners=excluded_big)


def compare_metrics(metrics, baseline, benchmark, subset_control=None):
    result = {}
    for label, other in [('poc_red', baseline), ('0050', benchmark), ('known5_control', subset_control)]:
        if other is None:
            result[label] = None
            continue
        years = {r['year']: r for r in other['annual']}
        result[label] = dict(total_return_percentage_points=100*(metrics['total_return']-other['total_return']),
                             final_nav_difference=metrics['final_nav']-other['final_nav'],
                             max_drawdown_percentage_points=100*(metrics['max_drawdown']-other['max_drawdown']),
                             annual_percentage_points={r['year']: 100*(r['total_return']-years[r['year']]['total_return']) for r in metrics['annual']})
    return result


def build_report(report_paths, root=ROOT):
    sources = Sources(root)
    cases, benchmarks, report_refs = {}, [], {}
    for path in report_paths:
        path = Path(path).resolve()
        name = str(path.relative_to(sources.root))
        expected = path.with_suffix('.sha256').read_text().split()[0]
        report = read(sources.bind(name, expected))
        report_refs[name] = expected
        if (report['start'], report['end'], report['initial_cash']) != (START, END, 1_000_000):
            raise ValueError('Unexpected broker study scope')
        sources.closure(report, name)
        for arm, descriptor in report['cases'].items():
            if arm not in ARMS:
                raise ValueError('Unregistered account arm')
            value = read(sources.bind(descriptor['path'], descriptor['sha256']))
            if value['completed'] != descriptor['completed'] or value['summary'] != descriptor['summary']:
                raise ValueError('Case and parent completion/summary differ')
            if arm in cases and value != cases[arm]:
                raise ValueError('Conflicting duplicate arm: provide one result per arm')
            cases[arm] = value
        benchmarks.append(report['benchmark'])
    if not benchmarks or any(b != benchmarks[0] for b in benchmarks):
        raise ValueError('Inconsistent sealed benchmark')
    benchmark = read(sources.bind(benchmarks[0]['path'], benchmarks[0]['sha256']))
    if benchmark['summary'] != benchmarks[0]['summary']:
        raise ValueError('Sealed benchmark descriptor changed')
    base = read(sources.bind(BASE_CASE, BASE_SHA))
    diagnostic = read(sources.bind(DIAG+'/report.json', DIAG_REPORT_SHA))
    sources.closure(diagnostic, DIAG+'/report.json')
    diag_rows = read(sources.bind(DIAG+'/rows.json', diagnostic['rows_sha256']))
    baseline_funded = {t['event_id'] for t in base['account']['trades'] if t['side'] == 'buy'}
    baseline, bm = account_metrics(base), account_metrics(benchmark)
    reference_dates = [r['date'] for r in base['account']['daily']]
    if [r['date'] for r in benchmark['account']['daily']] != reference_dates:
        raise ValueError('Benchmark trading calendar differs')
    if (baseline['start'], baseline['end'], baseline['initial_cash']) != (START, END, 1_000_000):
        raise ValueError('Baseline scope mismatch')
    results = {}
    for arm in ARMS:
        case = cases.get(arm)
        if case is None:
            results[arm] = dict(completed=False, reason='not_in_supplied_reports', metrics=None)
            continue
        if not case['completed']:
            if case.get('summary') is not None:
                raise ValueError('Incomplete account has published summary')
            results[arm] = dict(completed=False, reason=case.get('reason'), metrics=None)
            continue
        if [r['date'] for r in case['account']['daily']] != reference_dates:
            raise ValueError('Completed account is truncated or has different calendar')
        if arm == 'poc_red':
            for key in ('account', 'summary', 'entry_gate_decisions', 'profile_queries'):
                if case[key] != base[key]:
                    raise ValueError('Baseline differs from sealed ' + key)
        results[arm] = dict(completed=True, metrics=account_metrics(case),
                            coverage=gate_analysis(arm, case, diag_rows, baseline_funded))
    complete_gates = [case['broker_gate_decisions'] for case in cases.values() if case['completed']]
    if complete_gates:
        identities = [(r['event_id'], r['stock_id'], r['signal_date'], r['entry_date']) for r in complete_gates[0]]
        for rows in complete_gates[1:]:
            if [(r['event_id'], r['stock_id'], r['signal_date'], r['entry_date']) for r in rows] != identities:
                raise ValueError('Comparison arms do not share the same pre-broker population')
    control = results['poc_known5_control'].get('metrics')
    for value in results.values():
        if value['completed']:
            value['comparison'] = compare_metrics(value['metrics'], baseline, bm, control)
    for file in ('scripts/summarize_poc_broker_account.py', 'tests/test_summarize_poc_broker_account.py'):
        sources.bind(file, sha(sources.path(file)))
    # Publication rechecks metadata and rehashes changed files within this run.
    # No persistent verification cache or unverified active-file pointer is used.
    for name, digest in list(sources.used.items()):
        sources.bind(name, digest)
    return dict(schema='poc_broker_account_summary_v1', start=START, end=END, initial_cash=1_000_000,
                reports=report_refs, all_six_completed=all(v['completed'] for v in results.values()),
                arms=results, reference_baseline=baseline, benchmark=bm,
                source_sha256=dict(sorted(sources.used.items())),
                source_snapshot_resolutions=sorted(sources.snapshot_resolutions, key=lambda r: (r['report'], r['original'])),
                live_qualified=False, unseen_validation=False, actual_fill_verified=False,
                definitions=dict(annual='First opening NAV to last closing NAV of each calendar year; 2026 YTD ends 10/2',
                    trade_count='Child fills including board, odd-lot and partial fills; not round trips',
                    costs='Commission, sale tax and assumed slippage, all already deducted from NAV',
                    final_assets='Cash + marked holdings + receivables; no forced liquidation',
                    gate_year='Entry year, including 2023-12-29 signals entering 2024-01-02',
                    diagnostic='Sparse prior independent loss12/time63 signal paths; no three-black/account capacity/minimum fee; unavailable outcomes separate',
                    interpretation='Matched-subset filters must first compare with known5_control; guard arms retain unknown by policy; none is unseen validation'))


def table_rows(report):
    rows = []
    for arm in (*ARMS, 'benchmark'):
        item = report['arms'].get(arm, dict(completed=True, metrics=report['benchmark']))
        metrics = item['metrics']
        row = dict(arm=arm, label=LABELS[arm], completed=item['completed'], reason=item.get('reason', ''))
        if metrics:
            years = {r['year']: r['total_return']*100 for r in metrics['annual']}
            row.update(final_nav=metrics['final_nav'], total_return_pct=100*metrics['total_return'],
                       max_drawdown_pct=100*metrics['max_drawdown'], return_2024_pct=years['2024'],
                       return_2025_pct=years['2025'], return_2026_ytd_pct=years['2026'],
                       child_fills=metrics['trade_count'], buy_fills=metrics['buy_count'], sell_fills=metrics['sell_count'],
                       funded_events=metrics['funded_events'], costs=metrics['costs']['total_cost'],
                       commission=metrics['costs']['commission'], tax=metrics['costs']['tax'], slippage=metrics['costs']['slippage'])
            comparison = item.get('comparison')
            if comparison:
                row.update(vs_poc_red_pp=comparison['poc_red']['total_return_percentage_points'],
                           vs_0050_pp=comparison['0050']['total_return_percentage_points'],
                           vs_known5_control_pp=(comparison['known5_control'] or {}).get('total_return_percentage_points'))
            coverage = item.get('coverage')
            if coverage:
                row.update(known5_candidates=coverage['all']['known5'], unknown5_candidates=coverage['all']['unknown5'],
                           excluded_candidates=coverage['all']['excluded'], funded_known5=coverage['funded']['known5'],
                           excluded_diagnostic_closed_peak20=coverage['diagnostic_population']['excluded_closed_peak20'])
        rows.append(row)
    return rows


def save_report(value, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    path = output/'summary.json'
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')
    rows = table_rows(value)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output/'table.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    text = ['# POC 紅K＋分點帳戶比較', '', f'2024-01-02～2026-10-02；本金100萬元。2026為截至10/2；期末持股按市值估價。', '',
            '| 策略 | 累積報酬 | 最大回撤 | 2024 | 2025 | 2026截至10/2 | 期末資產 | 成本 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if not row['completed']:
            text.append(f"| {row['label']} | 未完成：{row['reason']} | — | — | — | — | — | — |")
        else:
            values = [f"{row[k]:.2f}%" for k in ('total_return_pct','max_drawdown_pct','return_2024_pct','return_2025_pct','return_2026_ytd_pct')]
            text.append('| '+row['label']+' | '+' | '.join(values)+f" | {row['final_nav']:,.2f} | {row['costs']:,.0f} |")
    text.extend(['', '費稅與滑價已扣除；成交數是普通盤／零股／部分成交子單數，不是完整買賣回合。', '',
                 '完整子集的篩選先與「完整五日子集對照」比較；未知保留兩組代表資料不足時維持原選股。', '',
                 '被排除的20%上漲訊號來自稀疏的既有獨立訊號診斷；不同出場／成本／資金限制，不能視為帳戶實際少賺或全市場召回率。', '',
                 '本表為歷史研究；高低價中點是成交假設，不是已證實可成交價格。live_qualified=false。'])
    (output/'summary.md').write_text('\n'.join(text)+'\n')
    (output/'summary.sha256').write_text(sha(path)+'\n')
    receipt = {p.name: sha(p) for p in (path, output/'table.csv', output/'summary.md')}
    (output/'outputs.json').write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    value = build_report(args.report)
    receipt = save_report(value, args.output)
    print(json.dumps(dict(all_six_completed=value['all_six_completed'], outputs=receipt), sort_keys=True))


if __name__ == '__main__':
    main()
