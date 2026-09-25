"""Reviewed input dependencies for the twenty frozen daily-account cases.

This inventory narrows questions to the inputs a case actually uses. It does
not certify publication times, historical universes, fills, or performance.
Adding a signal family requires a new review rather than an inferred default.
"""
from itertools import product


def configurations():
    result = {}
    for family in ('corporate', 'sector'):
        arms = ('benchmark', 'capacity') if family == 'corporate' else (
            'benchmark', 'relative_strength', 'strength_with_turnover')
        for arm, stress, board in product(arms, ('control', 'combined'), (False, True)):
            name = f'{family}:{arm}_{stress}_{"board_only" if board else "mixed"}'
            result[name] = dict(benchmark=arm == 'benchmark', board_only=board,
                position_count=0 if arm == 'benchmark' else 5, stress=stress)
            if family == 'sector':
                result[name]['arm'] = arm
    return result


def dependencies(name, config, evidence):
    if configurations().get(name) != config:
        raise ValueError('Unreviewed case or changed case configuration')
    if evidence['name'] != name:
        raise ValueError('Case evidence identity differs')
    benchmark, board = config['benchmark'], config['board_only']
    sector = name.startswith('sector:') and not benchmark
    if board and evidence['odd_lot']['required_sessions']:
        raise ValueError('Board-only case unexpectedly requires executable odd-lot orders')
    # Relative-strength sector controls use the same observable-peer pool as
    # the turnover arm. They therefore also depend on the membership snapshot.
    rules = [
        ('daily_price_volume_and_adjustments', True,
         'dated prices, traded volume and corporate adjustment factors; also 0050 trend and relative strength'),
        ('historical_universe_and_eligibility', not benchmark,
         'all candidates and peer denominators, including stocks never purchased'),
        ('historical_industry_membership', sector,
         'sector common-peer pool in both arms; diffusion groups use past price correlation'),
        ('financial_and_news_publication_versions', False,
         'these frozen selectors use prices, turnover and listing/peer identity; no revenue, earnings or news factor'),
        ('corporate_event_terms_and_delivery', True,
         'ex-date entitlements, opening references, share availability and cash payment; separate from news factors'),
        ('case_dated_market_identity', True,
         'own instrument identity and eligibility remain required for the explicit 0050 benchmark'),
        ('ordinary_complete_authenticated_sessions', evidence['ordinary']['required_sessions'] > 0,
         'every attempted executable board order on the observed account path'),
        ('odd_lot_complete_authenticated_sessions', evidence['odd_lot']['required_sessions'] > 0,
         'every attempted executable odd-lot order; policy-forbidden remainders are excluded'),
    ]
    rows = [dict(code=code, required=required, reason=reason) for code, required, reason in rules]
    return dict(name=name, config=dict(config), dependencies=rows,
        universe_required=not benchmark, historical_industry_required=sector,
        financial_news_archive_required=False, corporate_event_evidence_required=True,
        ordinary_required_sessions=evidence['ordinary']['required_sessions'],
        odd_lot_required_sessions=evidence['odd_lot']['required_sessions'],
        prior_generic_publication_gate_split=True,
        strict_data_ready=False, live_qualified=False)
