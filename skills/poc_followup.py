"""Isolated, preregistered exit, POC ordering and correlation experiments."""
from copy import deepcopy
import inspect
import math

import numpy as np
import pandas as pd

VARIANTS = ('exit_confirm', 'poc_strength', 'correlation60')


class ConfirmedBlack:
    """Two consecutive completed three-black observations; no stop change."""
    def __init__(self, source):
        self.source = source
        self.days = source.days

    def __getattr__(self, name):
        return getattr(self.source, name)

    def exits(self, execution_index, entry_index, sid):
        return (execution_index - entry_index >= 4
                and self.source.exits(execution_index - 1, entry_index, sid)
                and self.source.exits(execution_index, entry_index, sid))


def confirmed_black_auditor():
    """Independent raw-candle oracle; executor uses two three-candle signals."""
    from skills.three_black_exit import audit_three_black
    code = inspect.getsource(audit_three_black)
    for before, after in (
        ('i >= 4 and i-entry >= 3', 'i >= 5 and i-entry >= 4'),
        ('range(i-3, i)', 'range(i-4, i)'),
    ):
        if code.count(before) != 1:
            raise ValueError('Three-black independent audit seam changed')
        code = code.replace(before, after)
    namespace = dict(audit_three_black.__globals__)
    exec(compile(code, inspect.getsourcefile(audit_three_black), 'exec'), namespace)
    return namespace['audit_three_black']


def pair_correlation(close, sid, other, signal):
    """Exactly 60 common returns through the signal close, with no filling."""
    if sid not in close or other not in close:
        return None
    values = close.loc[:pd.Timestamp(signal), [sid, other]].tail(61)
    if len(values) != 61 or not np.isfinite(values.to_numpy()).all() or not values.gt(0).all().all():
        return None
    returns = values.pct_change(fill_method=None).iloc[1:]
    if returns.std(ddof=0).le(1e-12).any():
        return None
    return float(returns.iloc[:, 0].corr(returns.iloc[:, 1]))


class CorrelationPlanner:
    def __init__(self, source, log):
        self.source, self.log = source, log
        self.engine, self.day = source.engine, source.day

    def snapshot(self):
        return self.source.snapshot()

    def can_continue(self):
        return self.source.can_continue()

    def probe(self, event):
        result = self.source.probe(event)
        if not result['attempted']:
            return result
        sid = event['members'][0]
        # Occupied includes newly reserved candidates; held includes residuals.
        peers = sorted((set(self.snapshot()['held']) | set(self.snapshot()['occupied'])) - {sid})
        pairs = {other: pair_correlation(self.engine.exit_signals.adjusted_close,
            sid, other, event['signal_date']) for other in peers}
        failure = ('correlation_history_unknown' if any(v is None for v in pairs.values())
                   else 'correlation_above_080' if any(v > .80 for v in pairs.values()) else None)
        return dict(result, attempted=failure is None, failure=failure,
                    correlation_pairs=pairs, correlation_cutoff=event['signal_date'])

    def consider(self, event):
        expected = self.probe(event)
        if expected['attempted']:
            actual = self.source.consider(event)
            if any(actual[k] != expected[k] for k in actual):
                raise ValueError('Correlation reservation changed original cash/quantity')
        self.log.append(dict(event_id=event['event_id'], **expected))
        return expected


def strength_order(events, provider, planner):
    """Resolve all potentially reservable scores before ranking; never use fills."""
    scores, profiles = {}, {}
    for event in events:
        if not planner.probe(event)['attempted']:
            continue
        value = provider(deepcopy(event))
        profiles[event['event_id']] = value
        if not value.get('available'):
            return deepcopy(events), dict(applied=False, reason='profile_quality_unknown',
                                          unresolved=event['event_id']), profiles
        before, after = value.get('poc_before'), value.get('poc_after')
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                   and math.isfinite(x) and x > 0 for x in (before, after)):
            raise ValueError('Available POC profile lacks finite strength inputs')
        scores[event['event_id']] = after / before - 1
    rows = sorted(events, key=lambda e: (-scores.get(e['event_id'], -math.inf),
                                        -e['priority'], e['event_id']))
    ranked = [dict(deepcopy(e), priority=float(len(rows)-i)) for i, e in enumerate(rows)]
    return ranked, dict(applied=True, scores=scores,
                        original_priorities={e['event_id']: e['priority'] for e in events},
                        ordered_ids=[e['event_id'] for e in rows]), profiles


def correlation_fallback(events, planner):
    """Original order after an explicitly allowed quality failure, still gated."""
    rows, decisions = [], []
    initial = planner.snapshot()
    for event in events:
        result = planner.probe(event)
        decision = dict(event_id=event['event_id'], profile_status='original_quality_fallback',
                        selection_status='not_reserved', probe=deepcopy(result))
        if planner.can_continue() and result['attempted']:
            applied = planner.consider(event)
            if applied != result:
                raise ValueError('Fallback reservation changed between probe and apply')
            decision.update(selection_status='reserved', reservation=applied)
            rows.append(event)
        decisions.append(decision)
    return dict(events=rows, certificate=dict(complete=True, decisions=decisions,
        initial_state=initial, final_state=planner.snapshot(),
        original_ordered_ids=[e['event_id'] for e in events],
        reserved_event_ids=[e['event_id'] for e in rows],
        correlation_original_quality_fallback=True, profile_ranking_complete=False))


def install(runner, variant, metadata):
    if variant not in VARIANTS:
        raise ValueError('Unregistered POC follow-up variant')
    original = runner.engine_types
    if variant == 'exit_confirm':
        runner.audit_three_black = confirmed_black_auditor()

    class Experiment:
        def __init__(self, *args, **kwargs):
            self.followup_log = []
            if variant == 'exit_confirm':
                kwargs['black_signals'] = ConfirmedBlack(kwargs['black_signals'])
            else:
                selector = kwargs['candidate_selector']
                available_fallback = kwargs['selection_arm'].endswith('_available')
                def select(events, *, arm, provider, planner):
                    record = dict(date=str(planner.day.date()), variant=variant,
                                  original_ids=[e['event_id'] for e in events])
                    self.followup_log.append(record)
                    if not planner.can_continue():
                        record.update(applied=False, reason='no_opening_resources')
                        return selector(events, arm=arm, provider=provider, planner=planner)
                    if variant == 'poc_strength':
                        ranked, evidence, profiles = strength_order(events, provider, planner)
                        record.update(evidence)
                        originals = {e['event_id']: e for e in events}
                        def original_profile(event):
                            eid = event['event_id']
                            return deepcopy(profiles[eid]) if eid in profiles else provider(originals[eid])
                        return selector(ranked, arm=arm, provider=original_profile, planner=planner)
                    from skills.volume_profile_account_adapter import ReservationPlanner
                    record.update(applied=True, reservations=[])
                    wrapped = CorrelationPlanner(planner, record['reservations'])
                    options = {}
                    if 'planner_factory' in inspect.signature(selector).parameters:
                        options['planner_factory'] = lambda source: CorrelationPlanner(
                            ReservationPlanner(source.engine, source.day), record['reservations'])
                    from skills.volume_profile_selection import UnresolvedProfile
                    from skills.volume_profile_account_adapter import QUALITY_REASONS
                    try:
                        return selector(events, arm=arm, provider=provider, planner=wrapped, **options)
                    except UnresolvedProfile as exc:
                        if not (available_fallback and exc.recoverable is False and exc.reason in QUALITY_REASONS):
                            raise
                        record.update(original_quality_fallback=True, fallback_reason=exc.reason)
                        pristine = CorrelationPlanner(ReservationPlanner(planner.engine, planner.day),
                                                      record['reservations'])
                        return correlation_fallback(events, pristine)
                kwargs['candidate_selector'] = select
            super().__init__(*args, **kwargs)

        def run(self):
            account = super().run()
            account['followup_log'] = deepcopy(self.followup_log)
            account['settings']['followup_variant'] = variant
            account['settings']['protective_loss12_unchanged'] = True
            account['settings']['time63_unchanged'] = True
            return account

    def engines(*args, **kwargs):
        stock, benchmark = original(*args, **kwargs)
        return type('Followup'+stock.__name__, (Experiment, stock), {}), benchmark
    runner.engine_types = engines
    writer = runner.write
    def save(path, value):
        if 'family_rules' in value or path.name == 'report.json':
            value['followup'] = deepcopy(metadata)
        return writer(path, value)
    runner.write = save
