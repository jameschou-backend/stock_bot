"""Independent arithmetic checks for the new ordering and allocation decisions."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _correlation(close, sid, peer, signal):
    if sid not in close or peer not in close:
        return None
    frame = close.loc[close.index <= pd.Timestamp(signal), [sid, peer]].iloc[-61:]
    values = frame.to_numpy(dtype=float)
    if len(values) != 61 or not np.isfinite(values).all() or (values <= 0).any():
        return None
    returns = values[1:] / values[:-1] - 1
    deviations = returns - returns.mean(axis=0)
    if (returns.std(axis=0) <= 1e-12).any():
        return None
    return float(np.dot(deviations[:, 0], deviations[:, 1]) /
                 np.sqrt((deviations[:, 0]**2).sum()*(deviations[:, 1]**2).sum()))


def audit(case, report, root):
    root = Path(root)
    arm, variant = case['followup']['arm'], case['followup']['variant']
    account = case['account']
    settings = account['settings']
    if (settings.get('followup_variant') != variant or settings.get('horizon') != 63
            or settings.get('protective_loss12_unchanged') is not True
            or settings.get('time63_unchanged') is not True
            or (settings['buy_fraction'], settings['sell_fraction']) != (.5, .3)):
        raise ValueError('Follow-up changed protected account settings')
    if variant == 'exit_confirm':
        result = case['audit']['three_black']
        if any(len(t['bars']) != 4 for t in result['triggers']):
            raise ValueError('Exit confirmation lacks independent four-candle evidence')
        return dict(passed=True, variant=variant, decisions_checked=result['decisions_rebuilt'],
                    independent_raw_candle_audit=True)
    path = (root/'.cache/all-entry-accounts-20261011/inputs-v3/poc_up_red.json' if arm == 'poc_up_red'
            else root/'.cache/legacy-return-comparison-20261012/candidates-v1/candidates.json')
    events = {e['event_id']: e for e in json.loads(path.read_text())['entries']}
    if str(path.relative_to(root)) not in report['source_sha256']:
        raise ValueError('Candidate audit input is not bound by the replay')
    contexts = {c['date']: c for c in account['selection_decisions']}
    logs = account['followup_log']
    if [r['date'] for r in logs] != list(contexts):
        raise ValueError('Follow-up decision log is incomplete')
    profiles = report['shared_profile_results']
    close = None
    if variant == 'correlation60':
        close = pd.read_parquet(root/report['input_bundle']/'close-official.parquet').set_index('date')
        close.index = pd.to_datetime(close.index)
    count, pairs, fallbacks = 0, 0, 0
    for row in logs:
        context = contexts[row['date']]
        if row['original_ids'] != context['original_event_ids']:
            raise ValueError('Follow-up altered original candidate identity')
        if not row['applied']:
            if row['reason'] not in ('no_opening_resources', 'profile_quality_unknown'):
                raise ValueError('Unknown follow-up nonapplication reason')
            if row['reason'] == 'profile_quality_unknown':
                if profiles[row['unresolved']]['available'] is not False:
                    raise ValueError('Strength fallback was not backed by unavailable evidence')
                fallbacks += 1
            continue
        cert = context['certificate']
        if variant == 'poc_strength':
            scores = {}
            for eid, claimed in row['scores'].items():
                value = profiles[eid]
                if (value['available'] is not True or value['stock_id'] != events[eid]['members'][0]
                        or value['signal_date'] != events[eid]['signal_date']
                        or not value['prior_dates'] or max(value['prior_dates']) >= value['signal_date']):
                    raise ValueError('Strength uses unavailable or noncausal POC evidence')
                if 'poc_before' in value:
                    before, after = value['poc_before'], value['poc_after']
                else:
                    before = value['profile']['first_half']['poc_price']
                    after = value['profile']['second_half']['poc_price']
                scores[eid] = (after-before)/before
                if not math.isclose(scores[eid], claimed, rel_tol=0, abs_tol=1e-12):
                    raise ValueError('Strength score differs from sealed half profiles')
            priorities = {eid: events[eid]['priority'] for eid in row['original_ids']}
            if priorities != row['original_priorities']:
                raise ValueError('Strength tie-break priorities changed')
            expected = sorted(row['original_ids'], key=lambda e: (-scores.get(e, -math.inf), -priorities[e], e))
            if row['ordered_ids'] != expected or cert['original_ordered_ids'] != expected:
                raise ValueError('Strength candidate ordering differs')
            count += len(scores)
        elif variant == 'correlation60':
            if row.get('original_quality_fallback'):
                fallbacks += 1
                if cert.get('correlation_original_quality_fallback') is not True:
                    raise ValueError('Correlation quality fallback lost its gate')
            initial = cert['initial_state']
            held, occupied = set(initial['held']), set(initial['occupied'])
            fallback_occupied = occupied | {events[d['event_id']]['members'][0]
                for d in cert['decisions'] if d.get('profile_status') == 'known_true'
                and d.get('selection_status') == 'reserved'}
            for decision in cert['decisions']:
                eid = decision['event_id']
                sid, signal = events[eid]['members'][0], events[eid]['signal_date']
                for kind in ('probe', 'fallback_probe', 'reservation'):
                    evidence = decision.get(kind, {})
                    if 'correlation_pairs' not in evidence:
                        if kind == 'reservation' and decision.get('selection_status') == 'reserved':
                            raise ValueError('Reserved candidate lacks correlation evidence')
                        continue
                    state = decision.get('state_before') if kind == 'reservation' else None
                    peers = ((set(state['held']) | set(state['occupied'])) if state else
                             held | (fallback_occupied if kind == 'fallback_probe' else occupied)) - {sid}
                    expected = {other: _correlation(close, sid, other, signal) for other in sorted(peers)}
                    actual = evidence['correlation_pairs']
                    if set(actual) != set(expected) or evidence.get('correlation_cutoff') != signal:
                        raise ValueError('Correlation peer set or observation cutoff differs')
                    for other, value in expected.items():
                        if value is None:
                            if actual[other] is not None: raise ValueError('Missing correlation was fabricated')
                        elif actual[other] is None or not math.isclose(actual[other], value, abs_tol=1e-10):
                            raise ValueError('Correlation differs from independent 60-return arithmetic')
                    pairs += len(expected)
                    blocked = any(v is None or v > .80 for v in expected.values())
                    if evidence['attempted'] == blocked:
                        raise ValueError('Correlation admission differs from independent gate')
                if decision.get('selection_status') == 'reserved':
                    count += 1
                    if decision.get('profile_status') == 'known_false':
                        fallback_occupied.add(sid)
                    else:
                        occupied.add(sid)
        if not context.get('fallback_to_original') and context.get('actual_reservations_verified') is not True:
            raise ValueError('Candidate certificate was not checked against actual account reservations')
    return dict(passed=True, variant=variant, decisions_checked=count, correlation_pairs_checked=pairs,
                explicit_quality_fallback_days=fallbacks, source_cutoff_verified=True)
