"""Time and comparison boundaries shared by audited historical account runners."""
from datetime import date
import math


def iso_day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('Expected an ISO market date')
    return value


def validate_signals(entries, calendar):
    days = [iso_day(day) for day in calendar]
    if not days or days != sorted(set(days)):
        raise ValueError('Market calendar must be unique and increasing')
    positions = {day: i for i, day in enumerate(days)}
    identities = set()
    for row in entries:
        signal, entry = iso_day(row['signal_date']), iso_day(row['entry_date'])
        if signal not in positions or entry not in positions or positions[entry] != positions[signal] + 1:
            raise ValueError('Entry must follow the signal on the next market day')
        if row['event_id'] in identities:
            raise ValueError('Duplicate signal identity')
        identities.add(row['event_id'])
        for field in ('group_cutoff_date', 'trend_decision_date'):
            if iso_day(row[field]) > signal:
                raise ValueError('Signal contains future information: ' + field)
        for field in ('liquidity_at_signal', 'liquidity_before_entry'):
            evidence = row[field]
            if iso_day(evidence['as_of']) > signal:
                raise ValueError('Liquidity sizing contains future information')
            if evidence['complete_20_sessions'] is not True or evidence['observations'] != 20:
                raise ValueError('Liquidity history is incomplete')
            for key in ('adv20_shares', 'mean_turnover20_twd'):
                value = evidence[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Invalid prior liquidity value')
    return dict(signal_count=len(entries), next_market_day=True,
                dated_liquidity_checked=True, feature_values_recomputed=False)


def validate_completed_account(account, calendar, start, end):
    expected = [day for day in calendar if start <= day <= end]
    if not expected or expected[0] != start or expected[-1] != end:
        raise ValueError('Requested interval does not match the market calendar')
    if [row['date'] for row in account['daily']] != expected:
        raise ValueError('Incomplete account: full requested calendar is required')
    for row in account['trades']:
        signal = row.get('signal_date')
        if row['side'] == 'buy' and not signal:
            raise ValueError('Buy execution lacks its prior signal date')
        if signal and iso_day(signal) >= iso_day(row['date']):
            raise ValueError('Trade uses a same-day or future signal')
    return dict(full_calendar=True, trading_days=len(expected), prior_buy_signals=True)


def validate_comparison(strategy, benchmark):
    if strategy['config']['benchmark'] or not benchmark['config']['benchmark']:
        raise ValueError('Comparison must pair a strategy with its benchmark')
    for key in ('stress', 'board_only'):
        if strategy['config'][key] != benchmark['config'][key]:
            raise ValueError('Comparison execution policy differs: ' + key)
    left, right = strategy['account'], benchmark['account']
    if [r['date'] for r in left['daily']] != [r['date'] for r in right['daily']]:
        raise ValueError('Comparison market dates differ')
    for key in ('initial_cash', 'commission', 'minimum_fee', 'participation', 'odd_participation', 'slippage'):
        if key not in left['settings'] or key not in right['settings'] or left['settings'][key] != right['settings'][key]:
            raise ValueError('Comparison capital/cost assumptions differ: ' + key)
    return dict(same_dates=True, same_initial_cash=True, same_cost_and_execution_policy=True)
