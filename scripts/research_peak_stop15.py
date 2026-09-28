#!/usr/bin/env python3
"""Fixed 15% stop: intraday evidence gaps and a separately labelled close proxy."""
from pathlib import Path
import argparse
from datetime import datetime
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from scripts import research_midpoint as study
from scripts.research_midpoint_since_2025 import restart_inputs
from scripts.export_midpoint_2025_report import verify_cash
from skills.peak_stop_research import ClosePeakStopReplay, bar_status
from skills.midpoint_replay import MidpointStock
from skills.trial_registry import append_trial_registry

SPEC = ROOT/'docs/prereg_peak_stop15_20260929.md'
CODE = [Path(__file__), ROOT/'skills/peak_stop_research.py', SPEC]


def audit_close(account, states, data):
    """Independent prefix scan includes every completed and open cohort."""
    dates = [str(d.date()) for d in data.days]
    output = []
    for c in account['cohorts']:
        sid, eid = c['stock_id'], c['event_id']
        entry = dates.index(c['entry_date'])
        end = dates.index(c['exit_date'] or data.end)
        first = None
        for i in range(entry+1, end+1):
            prices = data.features.adjusted_close[sid].iloc[entry:i]
            current, peak = prices.iloc[-1], prices.max()
            # Parent expiry has precedence on session 63.
            reason = 'time63' if i-entry >= 63 else (
                'close_peak_stop15' if pd.notna(current) and current <= peak*.85+1e-10 else None)
            if reason:
                first = dict(reason=reason, signal_date=dates[i-1], target_date=dates[i]); break
        state = states.get(eid)
        if first:
            if not state or any(state[k] != first[v] for k,v in
                    (('trigger_reason','reason'),('signal_date','signal_date'),('target_date','target_date'))):
                raise ValueError('First close stop differs from independent history scan')
        elif state and state['trigger_reason']:
            raise ValueError('Unexpected latched stop')
        for trade in account['trades']:
            if trade['event_id'] == eid and trade['side'] == 'sell':
                if not first or trade['date'] < first['target_date'] or trade['reason'] != first['reason'] or trade['signal_date'] != first['signal_date']:
                    raise ValueError('Sale precedes or contradicts stop')
        output.append(dict(event_id=eid,stock_id=sid,entry_date=c['entry_date'],first_exit=first))
    return output


def intraday_diagnostic(data, account):
    quotes = data.quotes.copy(); quotes['date'] = pd.to_datetime(quotes.date)
    quotes = quotes.set_index(['stock_id','date'])
    action_dates = {(r.stock_id, pd.Timestamp(r.event_date)) for r in data.events.itertuples()}
    cache = study.strict.AdditionalTicks()
    directories = [cache.root, *[d.root for d in cache.donors]]
    needed, available = set(), set()
    rows = []
    for c in account['cohorts']:
        sid, eid = c['stock_id'], c['event_id']
        entry, last = pd.Timestamp(c['entry_date']), pd.Timestamp(c['exit_date'] or data.end)
        buys = [t for t in account['trades'] if t['event_id']==eid and t['side']=='buy']
        fills = [float(t['reference_price']) for t in buys]
        start = quotes.loc[(sid, entry)]
        # Entry timestamp is unknown. Closing mark is after all entries;
        # entry-session high is only an upper bound on a post-entry peak.
        low_peak = max([float(start.close), *fills])
        high_peak = max(float(start.high), low_peak)
        row = dict(stock_id=sid, event_id=eid, entry_date=c['entry_date'],
            original_exit_date=c['exit_date'], entry_fill_timestamp_verified=False,
            entry_session_trigger_unknown=True, first_possible=None, first_certain=None,
            diagnostic_scope='conditional on no entry-session exit; fixed original entry; no hypothetical fills')
        for day in data.days[(data.days >= entry) & (data.days <= last)]:
            key = (sid, str(day.date())); needed.add(key)
            if any((p/f'{sid}-{key[1]}.parquet').exists() and (p/f'{sid}-{key[1]}.json').exists() for p in directories):
                available.add(key)
        for day in data.days[(data.days > entry) & (data.days <= last)]:
            if (sid, day) in action_dates:
                row['stopped_at_corporate_action'] = str(day.date()); break
            if (sid, day) not in quotes.index:
                row['stopped_at_missing_quote'] = str(day.date()); break
            q = quotes.loc[(sid, day)]
            try:
                bounds = [bar_status(p, float(q.open), float(q.high), float(q.low), float(q.close)) for p in (low_peak, high_peak)]
            except ValueError:
                row['stopped_at_invalid_or_halted_quote'] = str(day.date()); break
            if row['first_possible'] is None and bounds[1]['status'] != 'no_trigger':
                row['first_possible'] = dict(date=str(day.date()),low_peak=low_peak,high_peak=high_peak,
                    threshold_min=bounds[0]['threshold'],threshold_max=bounds[1]['threshold'],
                    open=float(q.open),high=float(q.high),low=float(q.low),close=float(q.close),
                    status=bounds[1]['status'])
            if bounds[0]['status'] == 'certain_trigger':
                row['first_certain'] = dict(date=str(day.date()),threshold=bounds[0]['threshold'],
                    open=float(q.open),high=float(q.high),low=float(q.low),close=float(q.close)); break
            low_peak, high_peak = (b['next_peak'] for b in bounds)
        rows.append(row)
    return dict(completed=False,summary=None,intraday_performance_available=False,
        reasons=['HL2 entries have no fill timestamps; entry-session high may precede ownership',
                 'Board daily OHLC does not order high/low crossings',
                 'Odd daily OHLC/volume is not post-trigger odd-lot executable volume'],
        cohorts=rows,original_path_stock_days=len(needed),
        board_tick_file_pairs_present=len(available),board_tick_missing=len(needed-available),
        cache_inventory_is_validated_tape=False,
        first_missing_board_days=[dict(stock_id=s,date=d) for s,d in sorted(needed-available,key=lambda x:(x[1],x[0]))[:20]],
        covers_changed_strategy_path=False,network_requests=0)


def run(output, prepare=False):
    output = Path(output).resolve()
    if output.exists(): raise ValueError('Use a new run directory')
    pub = study.old.load_selector()
    data, inputs, identity, repairs, refs = study.strict.repaired_data(pub)
    data = restart_inputs(data)
    extra, extra_refs = study.load_exit_completion(ROOT)
    additions = study.old.parent.parent.parent.load_corporate_completion(ROOT)|study.old.load_capital_terms(ROOT)[0]|extra
    overrides = (study.old.read(study.old.parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        study.old.read(study.old.parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        study.old.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    sources = dict(pub['source_sha256'])|refs|extra_refs|study.old.file_identities([
        *CODE, ROOT/'scripts/research_midpoint_since_2025.py',*study.CODE], ROOT)
    provider = None
    if prepare:
        from scripts.prepare_mixed_odd_authorized import AuthorizedBudget,AuthorizedOdds,CACHE
        from skills.replay_market_feeds import ReplayMarketFeeds
        budget = AuthorizedBudget(CACHE/'budget.json',maximum={'finmind':0,'official':119})
        provider = ReplayMarketFeeds(ROOT/'.cache/peak-stop15-20260929/prepared-feeds',offline=False,
            http_get=budget.official,official_min_interval=5)
    cases = {}
    def replay():
        for arm, cls in [('control',MidpointStock),('close_proxy15',ClosePeakStopReplay)]:
            print('start',arm,flush=True)
            odds = AuthorizedOdds(ROOT,inputs/'execution-feeds',provider) if prepare else study.OddDailyCache(ROOT,inputs/'execution-feeds')
            ticks = study.strict.AdditionalTicks()
            feeds = study.old.ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
            corp = study.old.TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
            engine = cls(data.quotes,data.companies,data.days,data.entries,feeds,corp,start=data.start,end=data.end,
                ticks=ticks,participation=.01,liquidity_identity=identity,odd_feeds=odds,ordering='original',
                position_count=3,factor_mask=0,residual_policy='release',identity_report=identity,
                exit_signals=data.features,action_dates=list(zip(data.events.stock_id,data.events.event_date)))
            try:
                account = engine.run()
                study.old.validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
                audit = study.audit_high_return_resources(account,engine.resource_plans,engine.slot_decisions,
                    engine.board_decisions,engine.residual_days,data.quotes)
                audit.update(study.audit_midpoint(account,ticks,odds,study.market_routes([*ticks.queries,*odds.queries]),data.quotes,data.days,corp,feeds))
                if arm == 'control':
                    expected_path = ROOT/'.cache/midpoint-since-2025-20260928/final-a/original.json'
                    if account != study.old.read(expected_path)['account']: raise ValueError('Control changed')
                    sources.update(study.old.file_identities([expected_path],ROOT))
                    study.old.write(output/'intraday_diagnostic.json',intraday_diagnostic(data,account))
                else:
                    audit['independent_first_exit_scan'] = audit_close(account,engine.exit_states,data)
                result = dict(completed=True,summary=study.old.summarize(account),account=account,audit=audit,
                              peak_decisions=getattr(engine,'peak_decisions',[]))
                verify_cash(result)
            except (study.old.ReplayDataUnavailable,study.old.UnresolvedAction) as exc:
                result = dict(completed=False,summary=None,reason=str(exc),completed_sessions=len(engine.daily))
            if ticks.calls: raise ValueError('Unexpected tick request')
            sources.update(ticks.files|odds.files)
            path = output/(arm+'.json'); study.old.write(path,result)
            cases[arm] = dict(completed=result['completed'],summary=result['summary'],reason=result.get('reason'),
                path=str(path.relative_to(ROOT)),sha256=study.old.sha(path))
            append_trial_registry(dict(timestamp=datetime.now().isoformat(timespec='seconds'),source='peak_stop15_20260929',
                params=dict(arm=arm,threshold=.15),result_path=str(path.relative_to(ROOT)),
                completed=result['completed'],preparation=prepare,live_qualified=False))
            print(arm,{k:result['summary'][k] for k in ('total_return','max_drawdown','final_nav')} if result['completed'] else result,flush=True)
    if prepare: replay()
    else:
        with study.old.offline_only(): replay()
    if not prepare and study.old.file_identities([ROOT/p for p in sources],ROOT)!=sources:
        raise ValueError('Sources changed')
    report = dict(start=data.start,end=data.end,cases=cases,source_sha256=sources,preparation=prepare,
        intraday_requested_completed=False,actual_fill_verified=False,unseen_validation=False,live_qualified=False)
    study.old.write(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();run(a.output,a.prepare)
