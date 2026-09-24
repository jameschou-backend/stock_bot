#!/usr/bin/env python3
"""Audit completed prefixes of blocked board-only cases without reporting truncated returns."""
from copy import deepcopy
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_board_only import OUTPUT, inventory
from scripts.research_exit_scenarios import read, write, sha
from skills.slot_reuse_replay import audit_slots
from skills.board_only_replay import audit_board_only


def run(output=OUTPUT, check=False):
    if inventory() != read(output / 'identity.json'):
        raise ValueError('Sealed board-only source/code identity changed')
    for name, digest in read(output / 'manifest.json')['files_sha256'].items():
        if sha(output / name) != digest:
            raise ValueError('Sealed board-only result changed: ' + name)
    rows = {}
    for stress in ('control', 'combined'):
        path = output / 'cases' / f'capacity_{stress}_board_only.json'
        case = read(path)
        if case['completed']:
            continue
        account = deepcopy(case['partial_account'])
        if not account['daily']:
            rows[stress] = dict(completed_days=0, audit=None, case_sha256=sha(path))
            continue
        last = account['daily'][-1]['date']
        # The failure day may contain partial actions/fills; it is not an audited
        # daily balance sheet. Only already completed market dates are included.
        for key in ('daily', 'trades', 'orders', 'corporate_actions', 'cash_ledger', 'holdings'):
            account[key] = [row for row in account[key] if row['date'] <= last]
        account['cohorts'] = [row for row in account['cohorts'] if row['entry_date'] <= last]
        account['settings'] = dict(read(output / 'cases' / f'capacity_{stress}_mixed.json')['account']['settings'],
                                   execution_policy='board_only')
        plans = [row for row in case['resource_plans'] if row['date'] <= last]
        slots = [row for row in case['slot_decisions'] if row['date'] <= last]
        decisions = [row for row in case['board_decisions'] if row['date'] <= last]
        audit = audit_slots(account, plans, slots, opening_cash_only=True, lock_unused=True,
                            lock_opening_slots=True, lock_failed_slots=True)
        audit.update(audit_board_only(account, decisions, plans))
        residual = [row for row in account['holdings'] if 0 < row['qty'] < 1000]
        rows[stress] = dict(last_complete_date=last, case_sha256=sha(path), audit=audit,
            completed_days=len(account['daily']), trade_count=len(account['trades']),
            odd_trade_count=sum(row['channel'] == 'odd' for row in account['trades']),
            below_lot_buy_attempts=sum(row['side'] == 'buy' and row['failure'] == 'board_only_below_one_lot' for row in decisions),
            below_lot_sell_attempts=sum(row['side'] == 'sell' and row['failure'] == 'board_only_below_one_lot' for row in decisions),
            residual_only_slot_days=len(residual),
            last_complete_residual_holdings=[row for row in residual if row['date'] == last],
            cutoff_note='Audit only journals through the last fully completed market day; no full-period performance claimed')
    result = dict(cases=rows, manifest_sha256=sha(output / 'manifest.json'), auditor_sha256=sha(Path(__file__)))
    path = output / 'partial-audits.json'
    if check:
        if read(path) != result:
            raise ValueError('Partial audit differs from saved result')
    else:
        write(path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    options = parser.parse_args()
    result = run(options.output, options.check)
    print('audited blocked prefixes:', len(result['cases']), flush=True)
