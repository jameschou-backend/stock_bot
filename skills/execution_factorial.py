"""Explicit combinations of the three active board-lot execution stresses."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
from decimal import Decimal, ROUND_CEILING
from itertools import permutations
import json
import math
from pathlib import Path

from skills.backtest_case_cache import file_identities
from skills.historical_selector_replay import HistoricalBoardReplay

FACTORS = ('slippage', 'entry_delay', 'exit_delay')
LABELS = ('正常', '僅滑價加倍', '僅進場晚一天', '滑價加倍＋進場晚一天',
          '僅出場晚一天', '滑價加倍＋出場晚一天', '進出場都晚一天', '全部壓力')
CORPORATE_DOCUMENT = 'docs/execution_factorial_corporate_20260926.json'


def load_capital_terms(root):
    root = Path(root).resolve()
    doc = json.loads((root / CORPORATE_DOCUMENT).read_text())
    evidence = doc['evidence_sha256']
    if (doc.get('schema') != 1 or doc.get('finmind_requests') != 0
            or doc.get('secondary_sources_used_in_overrides') is not False or not evidence):
        raise ValueError('Capital return requires primary evidence')
    if file_identities([root / p for p in evidence], root) != evidence:
        raise ValueError('Capital return evidence changed')
    for key, row in doc['overrides'].items():
        if (row['kind'] != 'capital_reduction' or not key[:4].isdigit()
                or len(key[:4]) != 4 or key[4] != '-'
                or not date.fromisoformat(row['known_date']) < date.fromisoformat(key[5:])
                <= date.fromisoformat(row['pay_date'])
                or row.get('cash_rounding') != 'floor_ntd'
                or row.get('fractional_policy') != 'block_noninteger_conversion'
                or not row.get('evidence_files')
                or any(p not in evidence for p in row['evidence_files'])):
            raise ValueError('Capital return settlement terms are invalid')
        if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
               for k in ('multiplier', 'cash_per_share')) or not 0 < row['multiplier'] < 1 or row['cash_per_share'] < 0:
            raise ValueError('Capital return amounts are invalid')
    return deepcopy(doc['overrides']), dict(evidence)


class CapitalReturnActions:
    """Adapt exact-share cash reductions to the sealed ledger's two operations.

    Both legs retain their economic nature. Cash is entitled on OLD shares;
    the following exchange changes shares. Noninteger exchange still blocks in
    the sealed engine instead of silently assuming fractional settlement.
    """
    def __init__(self, provider):
        self.provider = provider

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def on_date(self, sid, day):
        result = []
        for row in self.provider.on_date(sid, day):
            if row['kind'] != 'capital_reduction':
                result.append(row)
                continue
            result.append(dict(row, kind='cash_dividend', distribution_type='capital_return',
                               action_id=row['action_id']+'-capital-cash'))
            result.append(dict(row, kind='split', distribution_type='capital_reduction',
                               action_id=row['action_id']+'-capital-shares'))
        return result


def flags(mask):
    if type(mask) is not int or mask not in range(8):
        raise ValueError('Factor mask must be an integer from 0 to 7')
    return {name: bool(mask & (1 << i)) for i, name in enumerate(FACTORS)}


class FactorialReplay(HistoricalBoardReplay):
    def __init__(self, *args, factor_mask, **kwargs):
        if 'stress_mode' in kwargs:
            raise ValueError('Use factor_mask instead of an implicit stress bundle')
        active = flags(factor_mask)
        # Parent initialization shifts entries once. During run(), stress_mode
        # controls only the exit delay; prices and depth are separate attributes.
        super().__init__(*args, stress_mode='entry_delay' if active['entry_delay'] else 'control', **kwargs)
        self.stress_mode = 'exit_delay' if active['exit_delay'] else 'control'
        self.stress_slippage = .009 if active['slippage'] else .0045
        self.stress_depth = self.stress_quote = False  # No odd-lot fills in this study.
        self.execution_factors = active
        self.corporate = CapitalReturnActions(self.corporate)

    def cash_move(self, day, kind, change, **extra):
        if extra.get('action_id', '').endswith('-capital-cash'):
            extra['cash_flow_nature'] = 'capital_return'
        return super().cash_move(day, kind, change, **extra)


def shapley(values):
    """Average all six orders of adding factors; outputs use the input units."""
    if set(values) != set(range(8)) or any(not math.isfinite(v) for v in values.values()):
        raise ValueError('All eight finite outcomes are required for attribution')
    contributions = dict.fromkeys(FACTORS, 0.)
    for order in permutations(range(3)):
        mask = 0
        for i in order:
            added = mask | (1 << i)
            contributions[FACTORS[i]] += (values[added] - values[mask]) / 6
            mask = added
    if not math.isclose(sum(contributions.values()), values[7] - values[0], abs_tol=1e-7):
        raise ValueError('Factor attribution does not reconcile')
    return contributions


def stock_pnl(account, marks):
    """Reconcile stock cash flows plus terminal assets to total account profit."""
    rows = defaultdict(lambda: dict(cash_flow=0., holding_value=0., receivable_value=0.))
    for row in account['cash_ledger']:
        if row['kind'] == 'initial_deposit':
            continue
        rows[row['stock_id']]['cash_flow'] += row['cash_change']
    final = account['daily'][-1]
    for row in account['holdings']:
        if row['date'] == final['date']:
            rows[row['stock_id']]['holding_value'] += row['market_value']
    for row in account['receivables']:
        if row['kind'] != 'cash' and row['fraction'] and row.get('fractional_cash_per_share') is None:
            raise ValueError('Fractional receivable has no settlement terms')
        value = row['amount'] if row['kind'] == 'cash' else (
            row.get('qty', 0) * marks[row['stock_id']]['price']
            + row['fraction'] * (row['fractional_cash_per_share'] or 0))
        rows[row['stock_id']]['receivable_value'] += value
    for row in rows.values():
        row['profit'] = sum(row.values())
    if not math.isclose(sum(r['profit'] for r in rows.values()),
            final['nav'] - account['settings']['initial_cash'], abs_tol=.02, rel_tol=0):
        raise ValueError('Stock P&L does not reconcile to final NAV')
    if not math.isclose(sum(r['receivable_value'] for r in rows.values()), final['receivable'], abs_tol=.02, rel_tol=0):
        raise ValueError('Stock receivable valuation does not reconcile')
    return dict(sorted(rows.items()))


def fills(account):
    result = {}
    for trade in account['trades']:
        if trade['side'] != 'buy':
            continue
        key = trade['event_id']
        row = result.setdefault(key, dict(event_id=key, stock_id=trade['stock_id'], name=trade['name'],
                                         date=trade['date'], quantity=0))
        row['quantity'] += trade['qty']
    return result


def path_comparison(base, other):
    left, right = fills(base), fills(other)
    common = set(left) & set(right)
    first_nav = next((dict(date=a['date'], base_nav=a['nav'], other_nav=b['nav'])
        for a, b in zip(base['daily'], other['daily']) if abs(a['nav']-b['nav']) > .01), None)
    return dict(first_nav_difference=first_nav, common_entry_events=len(common),
        only_base=[left[k] for k in sorted(set(left)-set(right))],
        only_other=[right[k] for k in sorted(set(right)-set(left))],
        changed_quantity=[dict(base=left[k], other=right[k]) for k in sorted(common)
                          if left[k]['quantity'] != right[k]['quantity']],
        changed_entry_date=[dict(base=left[k], other=right[k]) for k in sorted(common)
                            if left[k]['date'] != right[k]['date']],
        order_failure_rows=dict(Counter(r['failure'] for r in other['orders'] if r.get('failure'))))


def fixed_trade_slippage_delta(account):
    # Arithmetic on a frozen fill list, not another executable account.
    return sum(float((Decimal(str(t['reference_price'])) * t['qty'] * Decimal('.009')).quantize(
        Decimal(1), rounding=ROUND_CEILING)) - t['slippage'] for t in account['trades'])
