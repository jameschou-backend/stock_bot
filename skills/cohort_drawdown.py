"""Attribute an existing ledger's losses; never simulate orders or change holdings.

Each disjoint underwater episode starts at the last high-water mark and ends on
recovery (or the last observation). Attribution covers (peak, trough], so peak-day
fees are not charged twice. A cohort's winner status uses only its peak-day P&L.
"""
from collections import defaultdict
from decimal import Decimal


def money(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError('Non-finite ledger value')
    return result


def equal(a, b, label):
    if abs(money(a) - money(b)) > Decimal('.011'):
        raise ValueError(label)


def episodes(daily):
    dates = [r['date'] for r in daily]
    if not daily or dates != sorted(set(dates)):
        raise ValueError('Daily dates must be unique and chronological')
    if any(money(r['nav']) <= 0 for r in daily):
        raise ValueError('NAV must be positive')
    result, peak, trough = [], 0, None
    for i, row in enumerate(daily):
        if money(row['nav']) >= money(daily[peak]['nav']):
            if trough is not None:
                result.append((peak, trough, i))
                trough = None
            peak = i
        elif trough is None or money(row['nav']) < money(daily[trough]['nav']):
            trough = i
    if trough is not None:
        result.append((peak, trough, None))
    return sorted(result, key=lambda x: (money(daily[x[1]]['nav']) / money(daily[x[0]]['nav']), x[0]))


def attributed_cash(account):
    """Resolve dividend payments back to their original funded entry cohort."""
    cohorts = {r['event_id']: r for r in account['cohorts']}
    if len(cohorts) != len(account['cohorts']):
        raise ValueError('Duplicate cohort')
    rows = []
    for row in account['cash_ledger']:
        if row['kind'] == 'initial_deposit':
            continue
        event = row.get('event_id')
        if not event:
            matches = [a for a in account['corporate_actions']
                if a['date'] == row['date'] and a['stock_id'] == row['stock_id']
                and ((row['kind'] == 'dividend_payment' and a['kind'] == 'payment'
                      and a['action_id'] == row['action_id']
                      and money(a['amount']) == money(row['cash_change']))
                     or (row['kind'] == 'fractional_share_payment' and a['kind'] == 'share_delivery'))]
            if len(matches) != 1:
                raise ValueError('Cash payment has no unique cohort')
            event = matches[0]['event_id']
        if event not in cohorts or cohorts[event]['stock_id'] != row['stock_id']:
            raise ValueError('Cash cohort/stock mismatch')
        rows.append(dict(row, event_id=event))
    trade_cash, ledger_cash = defaultdict(Decimal), defaultdict(Decimal)
    for row in account['trades']:
        key = (row['date'], row['event_id'], row['side'], row['channel'])
        trade_cash[key] += money(row['cash_change'])
        equal(row['total_cost'], sum(money(row[k]) for k in ('commission', 'tax', 'slippage')), 'Trade cost mismatch')
        gross = money(row['gross']) * (1 if row['side'] == 'sell' else -1)
        equal(row['cash_change'], gross - money(row['total_cost']), 'Trade gross/cash mismatch')
    for row in rows:
        if row['kind'] in ('buy', 'sell'):
            ledger_cash[(row['date'], row['event_id'], row['kind'], row['channel'])] += money(row['cash_change'])
    if trade_cash != ledger_cash:
        raise ValueError('Trade and cash ledger disagree')
    return rows


def boundary(account, cash_rows, day):
    """Value cash rights explicitly. Reject unsupported unpriced share rights."""
    daily = next(r for r in account['daily'] if r['date'] == day)
    state = {r['event_id']: dict(cash=Decimal(0), market_value=Decimal(0), receivable=Decimal(0))
             for r in account['cohorts']}
    actions = [r for r in account['corporate_actions'] if r['date'] <= day]
    paid = {(r['event_id'], r['action_id']) for r in actions if r['kind'] == 'payment'}
    delivered = {(r['event_id'], r['action_id']) for r in actions if r['kind'] == 'share_delivery'}
    for row in actions:
        kind = row['kind']
        if kind not in ('cash_dividend', 'payment', 'stock_dividend', 'share_delivery'):
            raise ValueError('Corporate rights need an explicit attribution adapter: ' + kind)
        key = (row['event_id'], row['action_id'])
        if kind == 'stock_dividend' and key not in delivered:
            raise ValueError('Boundary contains unpriced share rights')
        if kind == 'cash_dividend' and key not in paid:
            state[row['event_id']]['receivable'] += money(row['entitlement_value'])
    for row in cash_rows:
        if row['date'] <= day:
            state[row['event_id']]['cash'] += money(row['cash_change'])
    for row in account['holdings']:
        if row['date'] == day:
            state[row['event_id']]['market_value'] += money(row['market_value'])
    initial = money(account['settings']['initial_cash'])
    for field in ('market_value', 'receivable'):
        equal(sum(s[field] for s in state.values()), daily[field], 'Boundary ' + field + ' does not reconcile')
    equal(initial + sum(s['cash'] for s in state.values()), daily['cash'], 'Boundary cash does not reconcile')
    for row in state.values():
        row['net_pnl'] = row['cash'] + row['market_value'] + row['receivable']
    equal(initial + sum(s['net_pnl'] for s in state.values()), daily['nav'], 'Boundary NAV does not reconcile')
    return state


def analyze(account, benchmark_daily, top=5):
    daily = account['daily']
    cash_rows = attributed_cash(account)
    benchmark = {r['date']: r for r in benchmark_daily}
    if list(benchmark) != [r['date'] for r in daily]:
        raise ValueError('Benchmark calendar differs')
    # Independent daily identity check uses only the sealed journal components.
    for i, row in enumerate(daily):
        if i:
            equal(row['opening_nav'], daily[i-1]['nav'], 'Opening NAV discontinuity')
        equal(row['nav'], money(row['opening_nav']) + money(row['market_pnl'])
              + money(row['dividend_entitlement']) + money(row['execution_basis_pnl']) - money(row['cost']),
              'Daily P&L does not reconcile')
        equal(row['cost'], sum(money(t['total_cost']) for t in account['trades'] if t['date'] == row['date']),
              'Daily cost does not reconcile')
    spans = episodes(daily)
    result = []
    for rank, (p, t, recovery) in enumerate(spans[:top], 1):
        peak, trough = daily[p], daily[t]
        start, end = peak['date'], trough['date']
        before, after = boundary(account, cash_rows, start), boundary(account, cash_rows, end)
        costs, sells = defaultdict(Decimal), defaultdict(list)
        cost_split = {k: Decimal(0) for k in ('commission', 'tax', 'slippage')}
        window_trades = [r for r in account['trades'] if start < r['date'] <= end]
        for row in window_trades:
            costs[row['event_id']] += money(row['total_cost'])
            for key in cost_split:
                cost_split[key] += money(row[key])
            if row['side'] == 'sell':
                sells[row['event_id']].append({k: row[k] for k in ('date', 'qty', 'reason')})
        rows = []
        components = {k: Decimal(0) for k in ('winner_decline', 'other_existing_decline', 'new_entry_decline', 'gain_offset', 'cost')}
        for cohort in account['cohorts']:
            event = cohort['event_id']; a, b = before[event], after[event]
            net = b['net_pnl'] - a['net_pnl']
            gross = net + costs[event]
            new = cohort['entry_date'] > start
            group = 'new_entry' if new else 'existing_winner' if a['net_pnl'] > 0 else 'existing_other'
            if not (net or costs[event] or a['market_value'] or b['market_value']):
                continue
            if gross < 0:
                key = {'new_entry': 'new_entry_decline', 'existing_winner': 'winner_decline',
                       'existing_other': 'other_existing_decline'}[group]
                components[key] += gross
            else:
                components['gain_offset'] += gross
            components['cost'] -= costs[event]
            rows.append(dict(event_id=event, stock_id=cohort['stock_id'], name=cohort['name'],
                entry_date=cohort['entry_date'], group=group,
                peak_net_pnl=float(a['net_pnl']), peak_market_value=float(a['market_value']),
                peak_weight=float(a['market_value']/money(peak['nav'])), trough_market_value=float(b['market_value']),
                market_value_change=float(b['market_value']-a['market_value']),
                cash_change=float(b['cash']-a['cash']), receivable_change=float(b['receivable']-a['receivable']),
                gross_pnl=float(gross), cost=float(costs[event]), net_pnl=float(net),
                giveback_within_peak_profit=float(min(-gross, a['net_pnl'])) if group == 'existing_winner' and gross < 0 else 0.,
                decline_beyond_peak_profit=float(max(-gross-a['net_pnl'], 0)) if group == 'existing_winner' and gross < 0 else 0.,
                sales=sells[event]))
        net_change = money(trough['nav']) - money(peak['nav'])
        equal(sum(components.values()), net_change, 'Episode components do not reconcile')
        equal(sum(money(r['net_pnl']) for r in rows), net_change, 'Cohort P&L does not reconcile')
        equal(sum(cost_split.values()), -components['cost'], 'Episode costs do not reconcile')
        days = daily[p+1:t+1]
        journal = {key: float(sum(money(r[key]) for r in days)) for key in
                   ('market_pnl', 'dividend_entitlement', 'execution_basis_pnl', 'cost')}
        equal(journal['market_pnl'] + journal['dividend_entitlement'] + journal['execution_basis_pnl'] - journal['cost'],
              net_change, 'Independent journal decomposition disagrees')
        result.append(dict(rank=rank, peak_date=start, trough_date=end,
            recovery_date=daily[recovery]['date'] if recovery is not None else None,
            peak_nav=peak['nav'], trough_nav=trough['nav'], net_change=float(net_change),
            drawdown=float(net_change/money(peak['nav'])), decline_sessions=t-p,
            recovery_sessions=recovery-p if recovery is not None else None,
            observed_underwater_sessions=(recovery if recovery is not None else len(daily)-1)-p,
            components={k:float(v) for k,v in components.items()}, costs={k:float(v) for k,v in cost_split.items()},
            component_loss_shares={k:float(v/net_change) for k,v in components.items()},
            winner_giveback=float(sum(money(r['giveback_within_peak_profit']) for r in rows)),
            winner_beyond_peak_profit=float(sum(money(r['decline_beyond_peak_profit']) for r in rows)),
            journal_components=journal, cohorts=sorted(rows, key=lambda r:r['net_pnl']),
            child_trade_count=len(window_trades), new_cohorts=len({r['event_id'] for r in window_trades
                if r['side']=='buy' and next(c for c in account['cohorts'] if c['event_id']==r['event_id'])['entry_date']>start}),
            benchmark_same_window_return=benchmark[end]['nav']/benchmark[start]['nav']-1,
            reconciliation_error=float(sum(components.values())-net_change)))
    return dict(episodes=result, total_underwater_episodes=len(spans), trading_days=len(daily),
        daily_identity_reconciled=True, all_selected_boundaries_reconciled=True)
