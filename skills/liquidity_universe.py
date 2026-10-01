"""The fixed expanded-universe liquidity experiment, distinct from grouped signals."""
from skills.liquidity_candidates import FILTERS

ARMS = ('liquid_universe', *FILTERS, 'benchmark')
NAMES = dict(liquid_universe='擴大股池原版', median50m='20 日中位數至少五千萬',
             prior50m='前 20 日均值至少五千萬', persistent50m='兩條件同時符合',
             benchmark='0050 股息再投入')


def expanded_entries(filtered):
    """Preserve event identity/order; never substitute the 667 grouped candidates."""
    entries = dict(liquid_universe=filtered['original'])
    if len(entries['liquid_universe']) == 0:
        raise ValueError('Expanded signal population cannot be empty')
    if any(not e['event_id'].startswith('liquid_universe-') for e in entries['liquid_universe']):
        raise ValueError('Expected expanded-universe event identity')
    entries.update({arm: filtered[arm] for arm in FILTERS})
    return entries
