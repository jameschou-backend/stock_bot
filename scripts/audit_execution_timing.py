#!/usr/bin/env python3
"""Read-only timing audit of the sealed loss12 history; never qualifies execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_timings(case, entries, days):
    positions = {day: i for i, day in enumerate(days)}
    violations = []

    def require(ok, kind, identity):
        if not ok:
            violations.append({'check': kind, 'identity': identity})

    def next_day(signal, execution):
        return (signal in positions and execution in positions
                and positions[execution] == positions[signal] + 1)

    by_id = {row['event_id']: row for row in entries}
    require(len(by_id) == len(entries), 'unique_candidate_ids', 'candidate_pool')
    for row in entries:
        identity = row['event_id']
        require(next_day(row['signal_date'], row['entry_date']), 'entry_next_session', identity)
        require(row['group_cutoff_date'] < row['signal_date'], 'group_fitted_before_signal', identity)
        for field in ('liquidity_at_signal', 'liquidity_before_entry'):
            require(row[field]['as_of'] <= row['signal_date'], field, identity)

    account = case['account']
    buys = [t for t in account['trades'] if t['reason'] == 'leader_entry']
    for trade in buys:
        entry = by_id.get(trade['event_id'])
        require(bool(entry) and trade['date'] == entry['entry_date']
                and trade['signal_date'] == entry['signal_date']
                and trade['stock_id'] in entry['members'], 'fill_matches_frozen_entry', trade['sequence'])
    for row in case['exit_decisions']:
        require(next_day(row['signal_date'], row['date']), 'exit_reads_previous_session', row['event_id'])

    # These are account sequence links, not observed wall-clock sequencing.
    funding = {}
    links = []
    for trade in account['trades']:
        key = (trade['date'], trade['event_id'])
        if trade['side'] == 'sell' and trade['stock_id'] == '0050' and trade['reason'] == 'fund_stock':
            funding.setdefault(key, []).append(trade['sequence'])
        elif trade['reason'] == 'leader_entry' and key in funding:
            links.append(dict(date=trade['date'], event_id=trade['event_id'],
                              stock_id=trade['stock_id'], buy_sequence=trade['sequence'],
                              preceding_etf_sale_sequences=list(funding[key])))
    missing_clock = [t['sequence'] for t in account['trades'] if not t.get('executed_at')]
    return dict(recorded_date_checks_pass=not violations, violations=violations,
                candidates=len(entries), entry_fills=len(buys), exit_decisions=len(case['exit_decisions']),
                total_fills=len(account['trades']), fills_without_execution_timestamp=len(missing_clock),
                same_day_etf_funding_buy_fills=len(links),
                same_day_etf_funding_events=len({(x['date'], x['event_id']) for x in links}),
                funding_links=links, live_qualified=False,
                verdict='date_order_only_execution_unverified',
                limits=['Date checks do not verify original publication or retrieval timestamps.',
                        'Daily close, final odd-lot quote and full-day volume cannot establish intraday fills.',
                        'Account sequence does not prove an ETF sale filled before a stock purchase.',
                        'Current-company universe, revised prices and repeated historical selection remain unresolved.'])


def run():
    base = ROOT / '.cache/exit-research'
    manifest_path = base / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    refs_path = ROOT / '.cache/exit-research-inputs/manifest.json'
    refs = json.loads(refs_path.read_text())['references']
    inputs = {'report': (base / 'report.json', manifest['files_sha256']['report.json'])}
    for name in ('signals', 'calendar'):
        inputs[name] = (ROOT / refs[name]['path'], refs[name]['sha256'])
    for name, (path, expected) in inputs.items():
        if sha(path) != expected:
            raise ValueError('Sealed input changed: ' + name)
    report = json.loads(inputs['report'][0].read_text())
    signals = json.loads(inputs['signals'][0].read_text())
    calendar = pd.read_parquet(inputs['calendar'][0])
    days = pd.to_datetime(calendar.loc[calendar.is_open, 'date']).dt.strftime('%Y-%m-%d').tolist()
    if len(set(days)) != len(days) or days != sorted(days):
        raise ValueError('Calendar must have unique ascending sessions')
    result = audit_timings(report['cases']['loss12'], signals['entries'], days)
    result['input_sha256'] = {str(path.relative_to(ROOT)): sha(path) for path, _ in inputs.values()}
    result['manifest_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in (manifest_path, refs_path)}
    result['audit_code_sha256'] = sha(__file__)
    # A changed dependency must remain visible even though the saved report is intact.
    result['historical_code_drift'] = [name for name, expected in manifest['context']['code_sha256'].items()
                                       if not (ROOT / name).is_file() or sha(ROOT / name) != expected]
    result['verification_scope'] = 'Selected saved report, signals and calendar hashes; not full source replay or original-time certification.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(json.dumps({k: v for k, v in result.items() if k != 'funding_links'}, ensure_ascii=False, indent=2))
    sys.exit(0 if result['recorded_date_checks_pass'] else 1)
