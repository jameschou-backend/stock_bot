"""Validate dated execution routing before the expensive account replay."""
from collections import defaultdict
from copy import deepcopy

from skills.historical_universe_completion import resolve_completion


def benchmark_exclusion_with_market(exclusion):
    """The sealed benchmark loader returns only the TWSE 0050 split schedule."""
    if exclusion.get('stock_id') != '0050' or exclusion.get('kind') != 'trading_suspension':
        raise ValueError('Unexpected benchmark suspension contract')
    if exclusion.get('market', 'TWSE').upper() != 'TWSE':
        raise ValueError('0050 benchmark venue changed')
    return dict(exclusion, market='TWSE')


def dated_market_resolver(identity):
    by_sid = defaultdict(lambda: dict(episodes=[], trading_exclusions=[],
        coverage_start=identity['coverage_start'], coverage_end=identity['coverage_end']))
    for key in ('episodes', 'trading_exclusions'):
        for row in identity[key]:
            if row.get('market', '').upper() not in ('TWSE', 'TPEX'):
                raise ValueError('Missing explicit historical market: '+row.get('stock_id', '?'))
            by_sid[row['stock_id']][key].append(deepcopy(row))
    def resolve(day, sid):
        stamp = str(day.date()) if hasattr(day, 'date') else str(day)
        value = resolve_completion(by_sid[sid], sid, stamp)
        return value.get('market') if value.get('status') in ('identified', 'official_trading_suspension') else None
    return resolve
