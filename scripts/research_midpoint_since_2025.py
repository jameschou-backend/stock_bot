#!/usr/bin/env python3
"""Restart the frozen HL2 stock account in 2025 without inherited positions."""
from dataclasses import replace
from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.exit_policy import decide_exit
from scripts import research_midpoint as study


def restart_inputs(data, start='2025-01-02'):
    # Retain pre-start history for rolling indicators and the 63-session clock.
    # Only a signal observed after the chosen start can create a new position.
    if pd.Timestamp(start) not in data.days or start > data.end:
        raise ValueError('Restart must be an audited session within coverage')
    entries = [e for e in data.entries if start <= e['signal_date']
               and start <= e['entry_date'] <= data.end]
    return replace(data, start=start, entries=entries)


def exit_evidence(data, account):
    """Independently rebuild the first latched prior-close exit for each cohort."""
    result = {}
    for cohort in account['cohorts']:
        sid, event = cohort['stock_id'], cohort['event_id']
        entry = data.days.get_loc(pd.Timestamp(cohort['entry_date']))
        price = data.features.price(entry, sid)
        state = dict(entry_index=entry, entry_price=price, peak_price=price)
        result[event] = None
        for i in range(entry+1, data.days.get_loc(pd.Timestamp(data.end))+1):
            context = data.features.context(i, sid, state)
            decision = decide_exit(context, 'loss12')
            if decision['exit']:
                result[event] = dict(reason=decision['reason'],
                    signal_date=str(data.days[i-1].date()), target_date=str(data.days[i].date()),
                    signal_adjusted_close=data.features.price(i-1, sid),
                    entry_adjusted_close=price, stop_adjusted_close=price*.88 if price else None,
                    **context)
                break
        sells = [t for t in account['trades'] if t['event_id']==event and t['side']=='sell']
        if any(result[event] is None or t['reason'] != result[event]['reason']
               or t['signal_date'] != result[event]['signal_date'] for t in sells):
            raise ValueError('Exit reason does not reproduce from prior closes: '+event)
    return result


def run(output, prepare=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory')
    pub = study.old.load_selector()
    data, inputs, identity, repairs, repair_refs = study.strict.repaired_data(pub)
    data = restart_inputs(data)
    extra, extra_refs = study.load_exit_completion(ROOT)
    additions = (study.old.parent.parent.parent.load_corporate_completion(ROOT)
                 | study.old.load_capital_terms(ROOT)[0] | extra)
    provider = None
    if prepare:
        from scripts.prepare_mixed_odd_authorized import AuthorizedBudget, AuthorizedOdds, CACHE
        from skills.replay_market_feeds import ReplayMarketFeeds
        budget = AuthorizedBudget(CACHE/'budget.json', maximum={'finmind': 0, 'official': 119})
        provider = ReplayMarketFeeds(ROOT/'.cache/midpoint-since-2025-20260928/prepared-feeds',
                                    offline=False, http_get=budget.official, official_min_interval=5)
    refs = dict(pub['source_sha256']) | repair_refs | extra_refs
    refs.update(study.old.file_identities([Path(__file__), *study.CODE], ROOT))
    results = {}

    def replay():
        for name in ('original', 'benchmark'):
            print('start', name, flush=True)
            odds = (AuthorizedOdds(ROOT, inputs/'execution-feeds', provider) if prepare
                    else study.OddDailyCache(ROOT, inputs/'execution-feeds'))
            result = study.case(data, inputs, identity, additions, benchmark=name=='benchmark',
                                ordering='original', odds=odds, ticks=study.strict.AdditionalTicks())
            if result['network_calls']:
                raise ValueError('HL2 must not fetch ticks')
            if result['completed'] and name == 'original':
                result['exit_evidence'] = exit_evidence(data, result['account'])
            results[name] = result
            refs.update(result['source_sha256'])
            study.old.write(output/(name+'.json'), result)
            print(name, {k: result.get(k) for k in ('completed', 'summary', 'reason')}, flush=True)

    if prepare:
        replay()
    else:
        with study.old.offline_only():
            replay()
    study.old.write(output/'entries.json', data.entries)
    report = dict(start=data.start, end=data.end, requested_end='2026-09-24',
                  requested_period_complete=False, initial_cash=1_000_000,
                  start_policy='empty account; signals from 2025-01-02 onward',
                  candidate_count=len(data.entries), preparation=prepare,
                  all_completed=all(r['completed'] for r in results.values()),
                  live_qualified=False, actual_fill_verified=False, unseen_validation=False,
                  source_sha256=refs,
                  limitations=['Frozen selector and verified market identity end at 2026-09-09.',
                               'Full-session high-low midpoint is an ex-post execution estimate.'])
    if not prepare and study.old.file_identities([ROOT/p for p in refs], ROOT) != refs:
        raise ValueError('Research sources changed during replay')
    study.old.write(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    run(args.output, args.prepare)
