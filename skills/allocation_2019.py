"""Immutable long-history comparison; no score or threshold search."""
POLICIES = {
    'original': ('original', 3),
    'original5': ('original', 5),
    'capacity3': ('capacity', 3),
    'capacity5': ('capacity', 5),
    'capacity_vol3': ('capacity_vol', 3),
    'capacity_vol5': ('capacity_vol', 5),
}
ARMS = (*POLICIES, 'benchmark')


def allocation_options(arm):
    if arm not in POLICIES:
        raise ValueError('Use a registered stock allocation policy')
    ordering, count = POLICIES[arm]
    return dict(ordering=ordering, position_count=count)
