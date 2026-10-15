"""Combine sealed, independently funded accounts without replay or rebalancing."""
from collections import defaultdict
from decimal import Decimal

from skills.poc_split_accounts import audit_funding
from skills.top3_validation import verify_costs

ARMS = ('poc_up_red', 'poc_priority')
SUM_FIELDS = ('cash', 'market_value', 'receivable', 'nav', 'cost', 'holdings',
              'stale_holdings', 'market_pnl', 'execution_basis_pnl', 'dividend_entitlement')


def total(values):
    return float(sum((Decimal(str(x)) for x in values), Decimal(0)))


def namespaced(value, sleeve):
    if isinstance(value, list):
        return [namespaced(x, sleeve) for x in value]
    if isinstance(value, dict):
        result = {k: namespaced(v, sleeve) for k, v in value.items()}
        for k in ('event_id', 'action_id'):
            if result.get(k): result[k] = sleeve+'::'+result[k]
        return result
    return value


def capacity_check(accounts):
    groups = defaultdict(list)
    for sleeve, account in accounts.items():
        for order in account['orders']:
            if order.get('filled_qty', 0):
                groups[(order['date'], order['stock_id'], order['channel'])].append((sleeve, order))
    overlapping, conflicts = 0, []
    for (day, sid, channel), rows in sorted(groups.items()):
        if len({s for s, _ in rows}) < 2:
            continue
        overlapping += 1
        caps = {r['daily_capacity_qty'] for _, r in rows}
        if len(caps) != 1:
            raise ValueError('Sleeves disagree on the shared market capacity')
        cap = next(iter(caps))
        filled = sum(r['filled_qty'] for _, r in rows)
        if filled > cap:
            conflicts.append(dict(date=day, stock_id=sid, channel=channel,
                combined_filled_qty=filled, single_market_capacity_qty=cap,
                excess_qty=filled-cap, sleeves=[dict(sleeve=s, side=r['side'], qty=r['filled_qty']) for s, r in rows]))
    return dict(overlapping_security_channel_days=overlapping, capacity_collision_count=len(conflicts),
        collisions=conflicts, joint_capacity_pass=not conflicts, actual_fill_verified=False,
        policy='independent_original_1pct_per_sleeve_no_order_netting',
        note='兩帳戶沿用原版各自1%容量；重疊訂單另核對合計容量，未重播共用成交佇列。')


def combine(accounts, mode):
    if set(accounts) != set(ARMS):
        raise ValueError('Exactly the two original sleeves are required')
    dates = None
    for account in accounts.values():
        audit_funding(account)
        verify_costs(account, mode)
        observed = [r['date'] for r in account['daily']]
        if not observed or observed != sorted(set(observed)) or (dates is not None and dates != observed):
            raise ValueError('Sleeve calendars differ; no filling or truncation allowed')
        dates = observed
    result = dict(settings=dict(initial_cash=1000000, slots=6, slots_per_sleeve=3,
        rebalance=False, cash_transfers=False, top3_cost_mode=mode,
        **{k: accounts[ARMS[0]]['settings'][k] for k in ('commission', 'slippage')}), daily=[])
    previous = peak = 1000000
    for i, day in enumerate(dates):
        rows = [accounts[a]['daily'][i] for a in ARMS]
        r = {k: total(x[k] for x in rows) for k in SUM_FIELDS}
        for k in ('holdings', 'stale_holdings'): r[k] = int(r[k])
        peak = max(peak, r['nav'])
        r.update(date=day, opening_nav=previous, daily_return=r['nav']/previous-1,
            total_return=r['nav']/1000000-1, drawdown=r['nav']/peak-1,
            sleeve_nav={a: accounts[a]['daily'][i]['nav'] for a in ARMS})
        previous = r['nav']
        result['daily'].append(r)
    for section in ('trades', 'orders', 'cash_ledger', 'holdings', 'cohorts', 'corporate_actions', 'receivables'):
        result[section] = []
        for a in ARMS:
            for original in accounts[a][section]:
                row = namespaced(original, a)
                row['sleeve'] = a
                # cash_after belongs to the originating account, not an invented
                # cross-account intraday sequence. Preserve its meaning explicitly.
                if 'cash_after' in row: row['sleeve_cash_after'] = row.pop('cash_after')
                result[section].append(row)
        result[section].sort(key=lambda r: r.get('date', r.get('entry_date', '')))
    positions = [r for r in result['holdings'] if r['date'] == dates[-1]]
    result['ending_inventory'] = dict(automatically_liquidated=False, positions=positions,
        remaining_positions=len(positions), stale_positions=sum(r['stale'] for r in positions), valuation='mark_to_market')
    result['shared_capacity'] = capacity_check(accounts)
    result['aggregation_audit'] = audit_combination(result, accounts)
    return result


def audit_combination(combined, accounts):
    """Independent scalar NAV identities and chain; never average drawdowns."""
    previous = peak = Decimal(1000000)
    for i, row in enumerate(combined['daily']):
        left, right = [accounts[a]['daily'][i] for a in ARMS]
        if row['date'] != left['date'] or row['date'] != right['date']:
            raise ValueError('Combined dates differ')
        for key in SUM_FIELDS:
            if abs(Decimal(str(row[key]))-Decimal(str(left[key]))-Decimal(str(right[key]))) > Decimal('.000001'):
                raise ValueError('Combined amounts differ: '+key)
        nav = Decimal(str(row['nav']))
        if abs(nav-sum(Decimal(str(row[k])) for k in ('cash', 'market_value', 'receivable'))) > Decimal('.011'):
            raise ValueError('Combined NAV identity differs')
        peak = max(peak, nav)
        for key, expected in [('opening_nav', previous), ('daily_return', nav/previous-1),
                              ('total_return', nav/1000000-1), ('drawdown', nav/peak-1)]:
            if abs(Decimal(str(row[key]))-expected) > Decimal('.000000001'):
                raise ValueError('Combined return chain differs: '+key)
        previous = nav
    if len(combined['daily']) != len(accounts[ARMS[0]]['daily']) or len(combined['daily']) != len(accounts[ARMS[1]]['daily']):
        raise ValueError('Combined calendar truncated')
    for section in ('trades', 'cash_ledger', 'cohorts'):
        if len(combined[section]) != sum(len(a[section]) for a in accounts.values()):
            raise ValueError('Combined journal drops sleeve records')
    return dict(passed=True, days=len(combined['daily']), method='sum_independent_nav_recompute_peak',
        independent_initial_deposits=[500000, 500000], cash_transfers=False, rebalance=False)
