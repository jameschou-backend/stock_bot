#!/usr/bin/env python3
"""Verify independent restart runs and export a reader-facing ledger dataset."""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(left, right, label):
    if not math.isclose(left, right, abs_tol=.02, rel_tol=0):
        raise ValueError(f'{label}: {left} != {right}')


def explain_entry(cohort):
    e = cohort['leader_evidence']
    return (f"突破60日高點；20日漲幅 {e['leader_return20']:.1%}，"
            f"超越0050 {e['leader_return20']-e['benchmark_return20']:.1%}；"
            f"量比 {e['leader_volume_ratio']:.2f} 倍；同群已突破比例 {e['leader_peer_breadth']:.1%}；"
            "0050在120日均線之上。同日按20日超額報酬排序，依剩餘名額買入。")


REASONS = {'leader_entry': '領先股突破訊號', 'loss12': '還原收盤價較進場日還原收盤價下跌至少12%',
           'time63': '持有達63個交易日', 'benchmark_buy': '0050基準投入或股息再投入'}


def cohort_rows(result):
    account, summary = result['account'], result['summary']
    trades = defaultdict(list)
    for t in account['trades']:
        trades[t['event_id']].append(t)
    action_events = {a['action_id']: a['event_id'] for a in account['corporate_actions']}
    paid = defaultdict(float)
    for row in account['cash_ledger']:
        if row['kind'] not in ('initial_deposit', 'buy', 'sell'):
            paid[action_events[row['action_id']]] += row['cash_change']
    held = defaultdict(float)
    marks = {}
    for h in summary['final_holdings']:
        held[h['event_id']] += h['market_value']
        marks[h['stock_id']] = h['price']
    rights = defaultdict(float)
    for r in summary['final_receivables']:
        if r['kind']=='cash':
            value = r['amount']
        else:
            value = r['qty']*marks[r['stock_id']] + r['fraction']*(r['fractional_cash_per_share'] or 0)
        rights[r['event_id']] += value
    rows = []
    for c in account['cohorts']:
        event = c['event_id']
        buy = [t for t in trades[event] if t['side']=='buy']
        sell = [t for t in trades[event] if t['side']=='sell']
        cash_out = -sum(t['cash_change'] for t in buy)
        cash_in = sum(t['cash_change'] for t in sell)
        qty_buy, qty_sell = sum(t['qty'] for t in buy), sum(t['qty'] for t in sell)
        pnl = cash_in+paid[event]+held[event]+rights[event]-cash_out
        evidence = result['exit_evidence'][event]
        rows.append(dict(stock_id=c['stock_id'], name=c['name'], event_id=event,
            signal_date=c['signal_date'], entry_date=buy[0]['date'],
            first_sale=sell[0]['date'] if sell else None,
            last_sale=sell[-1]['date'] if sell else None,
            exit_date=c.get('exit_date'), buy_qty=qty_buy, sell_qty=qty_sell,
            buy_price=sum(t['gross'] for t in buy)/qty_buy,
            sell_price=sum(t['gross'] for t in sell)/qty_sell if qty_sell else None,
            cash_out=cash_out, cash_in=cash_in, dividends=paid[event],
            market_value=held[event], receivable=rights[event], pnl=pnl,
            return_on_cost=pnl/cash_out, buy_reason=explain_entry(c),
            sell_reason=REASONS[evidence['reason']] if evidence else '尚未觸發出場',
            exit_signal_date=evidence['signal_date'] if evidence else None,
            exit_signal_return=evidence['entry_return'] if evidence else None,
            exit_signal_close=evidence['signal_adjusted_close'] if evidence else None,
            stop_adjusted_close=evidence['stop_adjusted_close'] if evidence else None,
            status='期末仍持有或有應收權利' if held[event]+rights[event] else '已結束'))
    close(sum(r['pnl'] for r in rows), summary['profit'], 'Cohort P&L reconciliation')
    return rows


def verify_cash(result):
    a, s = result['account'], result['summary']
    cash = 0.
    by_date = defaultdict(list)
    for row in a['cash_ledger']:
        cash += row['cash_change']
        close(cash, row['cash_after'], 'Cash journal running balance')
        by_date[row['date']].append(row)
    close(cash, s['cash'], 'Ending cash')
    cash = 0.
    for d in a['daily']:
        cash += sum(r['cash_change'] for r in by_date[d['date']])
        close(cash, d['cash'], 'Daily cash')
        close(d['cash']+d['market_value']+d['receivable'], d['nav'], 'Daily NAV')
    close(a['daily'][-1]['nav'], s['final_nav'], 'Final NAV')
    for cost in ('commission', 'tax', 'slippage', 'total_cost'):
        close(sum(t[cost] for t in a['trades']), s['costs'][cost], 'Cost '+cost)


def export(left, right, output):
    left, right, output = (Path(p).resolve() for p in (left, right, output))
    if left.resolve()==right.resolve():
        raise ValueError('Independent runs required')
    reports = [read(p/'report.json') for p in (left,right)]
    if any(r['preparation'] or not r['all_completed'] for r in reports):
        raise ValueError('Complete offline accounts required')
    if reports[0] != reports[1]:
        raise ValueError('Run metadata changed')
    root = Path(__file__).resolve().parents[1]
    for path, digest in reports[0]['source_sha256'].items():
        if sha(root/path) != digest:
            raise ValueError('Source changed: '+path)
    accounts = {}
    for name in ('original', 'benchmark'):
        a, b = read(left/(name+'.json')), read(right/(name+'.json'))
        if a != b or not a['completed'] or a['network_calls']:
            raise ValueError('Independent account mismatch: '+name)
        verify_cash(a)
        accounts[name] = a
    rows = cohort_rows(accounts['original'])
    report = {k:v for k,v in reports[0].items() if k!='source_sha256'}
    report.update(offline_identical=True, cash_nav_and_cohort_reconciled=True,
        audited_data_end=report['end'], missing_period=['2026-09-10','2026-09-24'],
        source_runs=[dict(path=str(p.relative_to(root)), report_sha256=sha(p/'report.json'),
                         original_sha256=sha(p/'original.json'), benchmark_sha256=sha(p/'benchmark.json'))
                     for p in (left,right)],
        summaries={k:v['summary'] for k,v in accounts.items()})
    payload = dict(report=report, cohorts=rows, accounts={k:v['account'] for k,v in accounts.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    print(json.dumps(dict(summary=report['summaries'], cohorts=len(rows), output=str(output)), ensure_ascii=False))
    return payload


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.left, args.right, args.output)
