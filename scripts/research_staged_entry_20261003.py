#!/usr/bin/env python3
"""Preregistered half-first entry, separate from funded-account execution.

This module's decision function only reads through the third holding close.
The later second fill and original exit are outcome observations, not entry
features. No historical experiment runs at import or without an explicit CLI.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.export_signal_explorer import verify_evidence
from scripts.research_early_signal_losses import PERIODS, digest, statistics
from skills.exit_policy import decide_exit
from skills.independent_signals import net_unit_return
from skills.independent_three_black import ThreeBlackPath, black_at_close, path_issue
from skills.trial_registry import append_trial_registry

PREREG_SHA256 = '7a8247ef6304e684e085d6360d9e2bf2a4649c1b1e84da48f7afebac0576efa1'
INPUT_MANIFEST_SHA256 = '6cd6ef3cbf9ebfced4741b2e60fea34212a2dc5605c6b6b5895c003a34bb65bf'
RANK_REPORT_SHA256 = 'deead8ba56001627867a0af188fa0ad4a2201fbbbac881b2a8f83dae0e6f68bb'
RANK_SIGNALS_SHA256 = 'e9c3e1826d012bf5c960f53d9ab1fb7a172d39c7a92e3c24d9fc054abd36752a'
ARMS = ('original_full_unit', 'half_then_confirm_once')


def _date(path, index):
    return str(path.days[index].date())


def _positive(value):
    return bool(np.isfinite(value) and value > 0)


def _hl2(path, index):
    """Research proxy in the same adjusted scale as the original signal path."""
    raw = float((path.high[index] + path.low[index]) / 2)
    adjusted = float(raw * path.close[index] / path.raw_close[index])
    if not _positive(raw) or not _positive(adjusted):
        raise ValueError('Positive finite HL2 prices required')
    return raw, adjusted


def _original_reason(path, entry, index):
    anchor, current = path.close[entry], path.close[index]
    known = _positive(anchor) and _positive(current)
    decision = decide_exit(dict(
        held_sessions=index + 1 - entry, has_signal=known,
        entry_return=float(current / anchor - 1) if known else None,
        peak_return=None, peak_drawdown=None, relative20=None,
        below_ma20_two=False, market_off_two=False, strong_trend=False,
    ), 'loss12')
    if decision['exit']:
        return decision['reason']
    return 'three_black' if black_at_close(path, entry, index) else None


def staged_decision(path, entry):
    """T+1 half; T+3 close decides one T+4 addition, without reading T+4.

    Entry is the first holding market-day index. A preexisting original exit
    wins over confirmation, including when both occur at the third close.
    Lack of later calendar rows is right-censoring, not invented no-add data.
    """
    path.validate()
    if type(entry) is not int or not 1 <= entry < len(path.days):
        raise ValueError('Entry must follow a completed signal session')
    result = dict(
        add_decision='unresolved', add_decision_reason=None,
        add_decision_issue=None, add_decision_known_at=None,
        confirmation_holding_day=3, add_holding_day=4,
        fixed_support=None, confirmation_adjusted_close=None,
        first_entry_raw_hl2=None, first_entry_adjusted_hl2=None,
        above_first_entry=None, three_closes_hold_support=None,
        original_exit_before_add_reason=None, original_exit_before_add_date=None,
    )
    if not _positive(path.close[entry - 1]):
        return dict(result, add_decision_issue='missing_pre_entry_close')

    # Validate only the prefix that is actually needed. Once an original exit
    # is known, an unused later confirmation-day gap cannot erase no-add.
    confirmation = entry + 2
    end = min(confirmation, len(path.days) - 1)
    for index in range(entry, end + 1):
        issue = path_issue(path, entry, index)
        if issue:
            return dict(result, add_decision_issue=issue,
                        add_decision_known_at=_date(path, index))
        if index == entry:
            raw, adjusted = _hl2(path, entry)
            result.update(first_entry_raw_hl2=raw, first_entry_adjusted_hl2=adjusted)
        reason = _original_reason(path, entry, index)
        if reason:
            return dict(result, add_decision='no_add',
                        add_decision_reason='original_exit_priority',
                        add_decision_known_at=_date(path, index),
                        original_exit_before_add_reason=reason,
                        original_exit_before_add_date=_date(path, index))

    if confirmation >= len(path.days):
        return dict(result, add_decision='pending_confirmation',
                    add_decision_reason='third_holding_close_not_yet_observed',
                    add_decision_known_at=_date(path, end))
    signal = entry - 1
    prior = path.close[max(0, signal - 60):signal]
    if len(prior) != 60 or not (np.isfinite(prior) & (prior > 0)).all():
        return dict(result, add_decision_issue='missing_signal_prior60_support',
                    add_decision_known_at=_date(path, confirmation))
    support = float(max(prior))
    recent = path.close[entry:confirmation + 1]
    above_entry = bool(path.close[confirmation] > result['first_entry_adjusted_hl2'])
    hold_support = bool((recent >= support).all())
    should_add = above_entry and hold_support
    return dict(result, add_decision='add' if should_add else 'no_add',
                add_decision_reason='confirmed_strength' if should_add else 'confirmation_failed',
                add_decision_known_at=_date(path, confirmation),
                fixed_support=support,
                confirmation_adjusted_close=float(path.close[confirmation]),
                above_first_entry=above_entry, three_closes_hold_support=hold_support)


def observe_second_fill(path, entry, decision):
    """Observe the one precommitted next-session HL2; never retry or re-time."""
    result = dict(add_fill_status='not_requested', second_entry_date=None,
                  second_entry_raw_hl2=None, second_entry_adjusted_hl2=None,
                  add_fill_issue=None, second_entry_single_price=None,
                  second_entry_volume=None)
    if decision['add_decision'] != 'add':
        return result
    index = entry + 3
    if index >= len(path.days):
        return dict(result, add_fill_status='pending_add',
                    add_fill_issue='next_session_not_yet_observed')
    issue = path_issue(path, entry, index)
    if issue is None and not path.volume[index] > 0:
        issue = 'no_volume_on_assumed_second_entry'
    if issue:
        return dict(result, add_fill_status='unresolved', add_fill_issue=issue)
    raw, adjusted = _hl2(path, index)
    return dict(result, add_fill_status='filled_hl2_proxy',
                second_entry_date=_date(path, index),
                second_entry_raw_hl2=raw, second_entry_adjusted_hl2=adjusted,
                second_entry_single_price=bool(path.high[index] == path.low[index]),
                second_entry_volume=float(path.volume[index]))


def staged_row(original, path, entry, decision=None):
    """Keep the sealed original exit and anchor; value two budget halves.

    Each half is a separate proportional, fee-inclusive budget. Inactive cash
    earns zero. Neither average entry nor the added day resets any exit rule.
    Daily highs/HL2 cannot establish queue position, integer shares, minimum
    ticket fees, corporate cash timing, or executable volume capacity.
    """
    if original['status'] == 'not_entered':
        if entry is not None:
            raise ValueError('Not-entered signal cannot have an entry index')
        return dict(original, staged_arm=ARMS[1], original_status='not_entered',
                    add_decision='not_entered', add_fill_status='not_requested',
                    invested_budget_weight=0., idle_cash_weight=1.,
                    first_leg_net_return=None, second_leg_net_return=None,
                    first_leg_return_component=None, second_leg_return_component=None)
    if original['status'] not in ('closed', 'open', 'pending_exit', 'unresolved'):
        raise ValueError('Unexpected original signal status')
    if (type(entry) is not int or not 1 <= entry < len(path.days)
            or original['entry_date'] != _date(path, entry)
            or original['signal_date'] != _date(path, entry - 1)):
        raise ValueError('Original signal and entry dates must remain T then T+1')
    decision = staged_decision(path, entry) if decision is None else decision
    # A caller may cache the causal decision, but cannot substitute another
    # event's dates or a post-confirmation instruction.
    if decision['add_decision'] == 'add' and decision['add_decision_known_at'] != _date(path, entry + 2):
        raise ValueError('Addition instruction must come from third holding close')
    fill = observe_second_fill(path, entry, decision)
    uncertain_allocation = (decision['add_decision'] == 'unresolved'
                            or fill['add_fill_status'] == 'unresolved')
    result = dict(original, staged_arm=ARMS[1], original_status=original['status'],
                  original_net_return=original.get('net_return'),
                  original_unrealized_net_return=original.get('unrealized_net_return'),
                  **decision, **fill,
                  planned_first_budget_weight=.5,
                  requested_second_budget_weight=.5 if decision['add_decision'] == 'add' else 0.
                  if decision['add_decision'] != 'unresolved' else None,
                  invested_budget_weight=None if uncertain_allocation else
                  1. if fill['add_fill_status'] == 'filled_hl2_proxy' else .5,
                  idle_cash_weight=None if uncertain_allocation else
                  0. if fill['add_fill_status'] == 'filled_hl2_proxy' else .5,
                  first_leg_net_return=None, second_leg_net_return=None,
                  first_leg_return_component=None, second_leg_return_component=None,
                  staged_data_issue=None, minimum_ticket_fees_included=False,
                  volume_capacity_verified=False, cash_account=False, live_qualified=False)
    # Original single-entry price excursions are not staged equity excursions.
    for name in ('mfe', 'mfe_date', 'peak_close_return', 'peak_close_date',
                 'confirmed_high_return', 'confirmed_high_date'):
        result['original_' + name] = original.get(name)
        result[name] = None
    issue = (decision['add_decision_issue'] if decision['add_decision'] == 'unresolved'
             else fill['add_fill_issue'] if fill['add_fill_status'] == 'unresolved' else None)
    if issue or original['status'] == 'unresolved':
        result.update(status='unresolved', outcome='unknown', net_return=None,
                      unrealized_net_return=None, gross_return=None,
                      staged_data_issue=issue or original.get('data_issue_code')
                      or original.get('data_issue') or 'original_path_unresolved')
        return result
    if original['status'] == 'closed' and (
            decision['add_decision'] == 'pending_confirmation' or fill['add_fill_status'] == 'pending_add'):
        raise ValueError('A closed original cannot have unobserved earlier confirmation/fill')

    first = decision['first_entry_adjusted_hl2']
    end = original.get('adjusted_end_price')
    if not _positive(first) or end is None or not _positive(end):
        raise ValueError('Valid original entry/end prices required for unit valuation')
    sealed_entry = original.get('adjusted_entry_price')
    if sealed_entry is None or not math.isclose(first, sealed_entry, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('First half must use the exact original entry HL2')
    first_net = float(net_unit_return(end / first))
    first_gross = float(end / first - 1)
    second_net = second_gross = None
    if fill['add_fill_status'] == 'filled_hl2_proxy':
        if original['exit_trigger_date'] and original['exit_trigger_date'] < fill['second_entry_date']:
            raise ValueError('A prior original exit cannot coexist with an addition')
        if original['observed_end_date'] < fill['second_entry_date']:
            raise ValueError('Second entry cannot follow the observed original end')
        second_net = float(net_unit_return(end / fill['second_entry_adjusted_hl2']))
        second_gross = float(end / fill['second_entry_adjusted_hl2'] - 1)
    net = .5 * first_net + (.5 * second_net if second_net is not None else 0.)
    gross = .5 * first_gross + (.5 * second_gross if second_gross is not None else 0.)
    result.update(first_leg_net_return=first_net, second_leg_net_return=second_net,
                  first_leg_return_component=.5 * first_net,
                  second_leg_return_component=.5 * second_net if second_net is not None else 0.,
                  gross_return=gross,
                  net_return=net if original['status'] == 'closed' else None,
                  unrealized_net_return=net if original['status'] != 'closed' else None,
                  outcome=('profit' if net > 1e-12 else 'loss' if net < -1e-12 else 'flat')
                  if original['status'] == 'closed' else 'unrealized')
    return result


def comparison(baseline, staged):
    """All original opportunities stay visible, including a half left in cash."""
    if [r['signal_id'] for r in baseline] != [r['signal_id'] for r in staged]:
        raise ValueError('Staged research must preserve original signal order and population')
    pairs = [(a, b) for a, b in zip(baseline, staged) if a['status'] == b['status'] == 'closed']
    original_closed = [a for a in baseline if a['status'] == 'closed']
    change = [b['net_return'] - a['net_return'] for a, b in pairs]
    groups = {}
    for label, predicate in (('original_winners', lambda r: r['net_return'] > 0),
                             ('original_return30', lambda r: r['net_return'] >= .30)):
        all_winners = [(a, b) for a, b in zip(baseline, staged) if a['status'] == 'closed' and predicate(a)]
        known = [(a, b) for a, b in all_winners if b['status'] == 'closed']
        groups[label] = dict(
            original=len(all_winners), staged_closed=len(known),
            staged_unknown_or_unfinished=len(all_winners) - len(known),
            first_half_exposure_retention=1. if all_winners else None,
            full_budget_added=sum(b.get('add_fill_status') == 'filled_hl2_proxy' for a, b in known),
            remains_profitable=sum(b['net_return'] > 0 for a, b in known),
            remains_return30=sum(b['net_return'] >= .30 for a, b in known),
            remains_profitable_fraction_of_original=sum(b['net_return'] > 0 for a, b in known) / len(all_winners) if all_winners else None,
            remains_return30_fraction_of_original=sum(b['net_return'] >= .30 for a, b in known) / len(all_winners) if all_winners else None,
            original_mean_on_known=float(np.mean([a['net_return'] for a, b in known])) if known else None,
            staged_mean_on_known=float(np.mean([b['net_return'] for a, b in known])) if known else None)
    return dict(
        baseline=statistics(baseline), staged=statistics(staged),
        add_decisions=dict(Counter(r.get('add_decision') for r in staged)),
        add_fills=dict(Counter(r.get('add_fill_status') for r in staged)),
        original_closed_opportunities=len(original_closed), common_closed_opportunities=len(pairs),
        original_closed_to_unknown_or_unfinished=len(original_closed) - len(pairs),
        observed_fraction_of_original_closed=len(pairs) / len(original_closed) if original_closed else None,
        baseline_on_common_opportunities=statistics([a for a, b in pairs]),
        staged_on_common_opportunities=statistics([b for a, b in pairs]),
        equal_unit_mean_baseline=float(np.mean([a['net_return'] for a, b in pairs])) if pairs else None,
        equal_unit_mean_staged_including_idle_cash=float(np.mean([b['net_return'] for a, b in pairs])) if pairs else None,
        equal_unit_mean_difference=float(np.mean(change)) if change else None,
        full_original_denominator_mean_staged=float(np.mean([b['net_return'] for a, b in pairs]))
        if pairs and len(pairs) == len(original_closed) else None,
        unknown_prevents_full_original_denominator=len(pairs) != len(original_closed),
        improved=sum(v > 1e-12 for v in change), worsened=sum(v < -1e-12 for v in change),
        unchanged=sum(abs(v) <= 1e-12 for v in change), winner_retention=groups,
        status_transitions=dict(Counter(a['status'] + '->' + b['status'] for a, b in zip(baseline, staged))))


def scope_comparisons(baseline, staged):
    """Fixed full/year/period scopes; a successful subgroup cannot replace all."""
    comparison(baseline, staged)  # Verify paired identities before selecting.
    scopes = {'all': lambda row: True}
    scopes.update({str(year): lambda row, y=year: row['signal_date'].startswith(str(y))
                   for year in range(2019, 2027)})
    scopes.update({name: lambda row, lo=lo, hi=hi: lo <= row['signal_date'] <= hi
                   for name, (lo, hi) in PERIODS.items()})
    return {name: comparison([r for r in baseline if keep(r)], [r for r in staged if keep(r)])
            for name, keep in scopes.items()}


def verify_fixed_support(decision, feature):
    """Bind the cached decision to its original signal-time breakout level."""
    support = decision.get('fixed_support')
    if support is not None:
        sealed = feature['previous60_high_adjusted']
        if not _positive(sealed) or not math.isclose(support, sealed, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Staged support differs from the sealed original signal feature')


def _load_paths(bundle, ids):
    frames = {name: pd.read_parquet(bundle / (name + '.parquet'), columns=['date', *ids]).set_index('date')
              for name in ('close-official', 'close-quality', 'eligibility')}
    for frame in frames.values():
        frame.index = pd.to_datetime(frame.index)
    days = frames['close-official'].index
    quotes = pd.read_parquet(bundle / 'quotes-unmasked.parquet')
    quotes = quotes.loc[quotes.stock_id.isin(ids)].copy()
    quotes.date = pd.to_datetime(quotes.date)
    if quotes.duplicated(['date', 'stock_id']).any():
        raise ValueError('Duplicate raw stock/date quote')
    fields = {name: quotes.pivot(index='date', columns='stock_id', values=name).reindex(index=days, columns=ids)
              for name in ('close', 'high', 'low', 'volume', 'open')}
    return {sid: ThreeBlackPath(days,
        *[frames[name][sid].to_numpy(bool if name == 'eligibility' else float)
          for name in ('close-official', 'close-quality', 'eligibility')],
        *[fields[name][sid].to_numpy(float) for name in ('close', 'high', 'low', 'volume', 'open')])
        for sid in ids}


def _trial(arm, result, prereg, *, status='completed', record=False):
    row = dict(timestamp=datetime.now(timezone.utc).isoformat(), command=' '.join(sys.argv),
        source='staged_entry_20261003', status=status, sharpe=None,
        params=dict(arm=arm, start='2019-01-01', end='2026-10-02',
            first_budget=.5, second_budget=.5, confirmation_holding_day=3, add_holding_day=4,
            one_confirmation_only=True, original_exit_unchanged=True, independent_path_research=True,
            cash_account=False, unseen_validation=False),
        prereg_sha256=digest(prereg) if prereg.is_file() else None, result=result)
    if record:
        row['registry_row'] = append_trial_registry(row)
    return row


def run(bundle, rank_dir, research, output, prereg, *, record_trials=False):
    bundle, rank_dir, research, output, prereg = [Path(p).resolve() for p in (bundle, rank_dir, research, output, prereg)]
    if output.exists() or not all(p.is_relative_to(ROOT) for p in (bundle, rank_dir, research, output, prereg)):
        raise ValueError('Require repository-local evidence and a new output directory')
    if digest(prereg) != PREREG_SHA256:
        raise ValueError('Changed sequential preregistration')
    refs, manifest = verify_evidence(bundle, rank_dir)
    events = json.loads((bundle / 'signals.json').read_text())['entries']
    if len(events) != 30188 or len({e['event_id'] for e in events}) != len(events):
        raise ValueError('Require all original 30188 unique signal opportunities')
    features = pd.read_parquet(bundle / 'signal-features.parquet').set_index('event_id')
    if not features.index.is_unique or set(features.index) != {e['event_id'] for e in events}:
        raise ValueError('Signal feature universe differs from frozen original opportunities')
    ids = sorted({e['members'][0] for e in events})
    paths = _load_paths(bundle, ids)
    decisions, entries = {}, {}
    for event in events:
        key, sid = event['event_id'], event['members'][0]
        feature = features.loc[key]
        if feature.stock_id != sid or feature.signal_date != event['signal_date'] or feature.entry_date != event['entry_date']:
            raise ValueError('Frozen event and signal feature identity differ')
        index = int(paths[sid].days.get_loc(pd.Timestamp(event['entry_date']))) if event['entry_date'] else None
        decision = staged_decision(paths[sid], index) if index is not None else {'add_decision': 'not_entered'}
        verify_fixed_support(decision, feature)
        decisions[key], entries[key] = decision, index
    # Every confirmation and fixed-support check precedes outcome deserialization.
    workbook = research / 'workbook-data.json'
    if str(workbook.relative_to(ROOT)) not in refs:
        raise ValueError('Unbound original outcome workbook')
    payload = json.loads(workbook.read_text())
    originals = payload['rows']
    if len(originals) != len(events) or len({r['signal_id'] for r in originals}) != len(events) or {r['signal_id'] for r in originals} != set(decisions):
        raise ValueError('Outcome and decision populations differ')
    staged = [staged_row(row, paths[row['stock_id']], entries[row['signal_id']], decisions[row['signal_id']]) for row in originals]
    results = scope_comparisons(originals, staged)
    for path in (Path(__file__), prereg, ROOT / 'tests/test_staged_entry_research.py',
                 ROOT / 'scripts/export_signal_explorer.py', ROOT / 'scripts/research_early_signal_losses.py',
                 ROOT / 'skills/independent_three_black.py', ROOT / 'skills/independent_signals.py', ROOT / 'skills/trial_registry.py'):
        refs[str(path.relative_to(ROOT))] = digest(path)
    trials = [_trial(arm, {name: value['baseline'] if arm == ARMS[0] else value for name, value in results.items()},
                     prereg, record=record_trials) for arm in ARMS]
    output.mkdir(parents=True)
    def dump(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pd.DataFrame([dict(signal_id=key, **value) for key, value in decisions.items()]).to_parquet(output / 'decisions.parquet', index=False)
    pd.DataFrame(staged).to_parquet(output / 'staged-paths.parquet', index=False)
    dump('comparisons.json', results)
    dump('trials.json', trials)
    report = dict(schema='independent_staged_entry_v1', created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=refs, output_sha256={str(p.relative_to(ROOT)): digest(p) for p in output.iterdir()},
        sample_count=len(staged), latest_data=manifest['end'], all_results=results['all'], registry_recorded=record_trials,
        all_three_period_paired_means_improve=all(results[p]['equal_unit_mean_difference'] is not None and results[p]['equal_unit_mean_difference'] > 0 for p in PERIODS),
        three_period_paired_coverage={p: results[p]['observed_fraction_of_original_closed'] for p in PERIODS},
        live_qualified=False, cash_account=False, unseen_validation=False,
        definitions=dict(holding_day='entry session is1; day3 close decides one possible day4 addition',
            support='frozen maximum of 60 adjusted closes strictly before original signal date',
            budget='0.5*fee-inclusive first-leg return + 0.5*fee-inclusive second-leg return, or idle cash at zero',
            original_exit='original stop anchor, threeblack/time rules and original observed exit day unchanged',
            promotion='only if all3 fixed historical periods improve common original-opportunity mean; requires further account validation'),
        limitations=payload['metadata']['limitations'] + [
            'Overlapping independent one-unit observations; not a compounded or volume-qualified account.',
            'HL2 fills use full-day prices after the decision; no known limit order or guaranteed execution.',
            'Two proportional half budgets include separate buy/sell fees; integer shares and minimum ticket fees are not modeled.',
            'Unknown allocation or original outcomes remain unknown, never replaced with a free-cash zero return.',
            'All original opportunities retained; unfilled/unfinished states remain separate from closed comparisons.',
            'Reused historical data, not new unseen validation or live qualification.'])
    dump('report.json', report)
    (output / 'report.sha256').write_text(digest(output / 'report.json') + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / '.cache/all-signals-2019-20261002/inputs')
    parser.add_argument('--rank-dir', type=Path, default=ROOT / '.cache/signal-rank-20261003/rank-v2')
    parser.add_argument('--research', type=Path, default=ROOT / '.cache/all-signals-2019-20261002/research-v1')
    parser.add_argument('--prereg', type=Path, default=ROOT / 'docs/research_sequential_prereg_20261003.md')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.inputs, args.rank_dir, args.research, args.output, args.prereg, record_trials=True)
        print(json.dumps({'sample_count': report['sample_count'], 'results': report['all_results'],
                          'all_three_period_paired_means_improve': report['all_three_period_paired_means_improve']}, ensure_ascii=False))
    except Exception as exc:
        rows = [_trial(arm, {'error': type(exc).__name__ + ': ' + str(exc)}, args.prereg, status='failed', record=True) for arm in ARMS]
        if args.output.resolve().is_relative_to(ROOT):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_name(args.output.name + '-failure.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
