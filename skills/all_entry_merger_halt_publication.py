"""Preserve historical zero-budget rejections when auditing merger exposure."""
from pathlib import Path

from skills.all_entry_merger_halt import DOCUMENT, ROOT, START, END, load_halt, verify_halt_proof


def verify_published_halt(case, root=ROOT):
    if 'merger_halt_audit' in case:
        return verify_halt_proof(case, root)
    account = case['account']
    plans = [p for p in account.get('tick_plans', [])
             if p['stock_id'] == '2888' and START <= p['date'] < END]
    held = any(r['stock_id'] == '2888' and r['date'] >= START and r['qty'] > 0
               for r in [*account.get('holdings', []), *account.get('trades', [])])
    if held or not plans:
        return verify_halt_proof(case, root)
    load_halt(root, Path(root) / DOCUMENT)
    for p in plans:
        if (p['side'] != 'buy' or p['rejection'] != 'opening_slots_locked'
                or any(p[k] != 0 for k in ('planned_qty', 'board_qty', 'odd_qty',
                                          'sizing_budget', 'reserved_cash'))
                or p['signal_date'] != p['reference_date']
                or not '2025-06-27' < p['signal_date'] < p['date']):
            raise ValueError('Old suspension plan was not a zero-budget slot rejection')
        orders = [r for r in account['orders']
                  if (r['date'], r['event_id'], r['side']) == (p['date'], p['event_id'], p['side'])]
        if (len(orders) > 1 or any(r['filled_qty'] != 0 or r['channel'] != 'event' for r in orders)
                or any(r.get('date') == p['date'] and r.get('event_id') == p['event_id']
                       and r.get('cash_change', 0) != 0 for r in account['cash_ledger'])):
            raise ValueError('Old slot rejection moved shares or cash')
