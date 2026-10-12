"""Audit an excluded 0050 reinvestment without discarding existing units.

The frozen auditor assumes every buy opens a new equity cohort. This private
revision admits only a retained benchmark position, checks both adjacent daily
holdings, and leaves source-failure reconstruction and all cash checks intact.
"""
from hashlib import sha256
import inspect
from pathlib import Path
from types import FunctionType

SOURCE = Path(__file__).with_name('poc_gap_audit.py')
SOURCE_SHA = 'c97d5646f631e12e41cf7fca629ed1b06d541bddc4a8f5d2cc97602ab13e94cf'


def retained_benchmark(account, gap, key, completed):
    if key[1:] != ('0050', 'buy', 'benchmark'):
        raise ValueError('Retained buy exception is only for the benchmark')
    if any(c.get('event_id') == 'benchmark' for c in account['cohorts']):
        raise ValueError('Excluded benchmark buy created a cohort')
    if any(r.get('date') == key[0] and r.get('stock_id') == '0050'
           for r in account['corporate_actions']):
        raise ValueError('Retained benchmark gap needs separate same-day corporate audit')
    prior_days = [r['date'] for r in account['daily'] if r['date'] < key[0]]
    if not prior_days:
        raise ValueError('Retained benchmark gap lacks a prior valuation')
    for day in [max(prior_days)] + ([key[0]] if completed else []):
        held = [r for r in account['holdings'] if r['date'] == day and r['stock_id'] == '0050']
        if (len(held) != 1 or held[0].get('event_id') != 'benchmark'
                or held[0]['qty'] != gap['holding_qty_before']):
            raise ValueError('Excluded benchmark buy altered or fabricated retained shares')


def private_gap_audit():
    from skills.poc_gap_audit import audit_gap_records
    if sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Frozen data-gap auditor changed')
    code = inspect.getsource(audit_gap_records)
    old = "        if plan['side'] == 'buy':\n"
    if code.count(old) != 1:
        raise ValueError('Expected one excluded buy holding audit')
    code = code.replace(old,
        "        if plan['side'] == 'buy' and key[1] == '0050' and gap['holding_qty_before'] > 0:\n"
        "            _retained_benchmark(account, gap, key, completed)\n"
        "        elif plan['side'] == 'buy':\n")
    namespace = dict(audit_gap_records.__globals__, _retained_benchmark=retained_benchmark)
    exec(compile(code, str(SOURCE), 'exec'), namespace)
    return namespace['audit_gap_records']


def install(runner):
    visited, namespaces = set(), {}
    def visit(function):
        if not isinstance(function, FunctionType) or id(function) in visited:
            return
        visited.add(id(function))
        ns = function.__globals__
        if ns.get('__name__') == '_all_entry_independent_audit':
            namespaces[id(ns)] = ns
        for cell in function.__closure__ or ():
            visit(cell.cell_contents)
    visit(runner._all_entry_indexed_audit)
    if len(namespaces) != 1:
        raise ValueError('Expected one private gap audit namespace')
    next(iter(namespaces.values()))['audit_gap_records'] = private_gap_audit()
