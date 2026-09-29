#!/usr/bin/env python3
"""Fixed-rule full-account comparison; preparation never publishes performance."""
import argparse
from datetime import date, datetime
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from app.finmind import fetch_dataset, FinMindError
from scripts import research_midpoint as study
from scripts.research_midpoint_since_2025 import restart_inputs
from scripts.export_midpoint_2025_report import verify_cash
from skills.portfolio_intraday_exit import PortfolioIntradayExit
from skills.midpoint_exit_replay import MidpointExitReplay
from skills.midpoint_exit_audit import audit_midpoint_exit
from skills.execution_resources import audit_resources
from skills.portfolio_intraday_audit import audit_sources
from skills.intraday_limit_replay import normalize_ticks
from skills.trial_registry import append_trial_registry

old = study.old
CACHE = ROOT/'.cache/portfolio-intraday-20260929'
SPEC = ROOT/'docs/prereg_portfolio_intraday_20260929.md'
CODE = [Path(__file__),ROOT/'skills/portfolio_intraday_exit.py',ROOT/'skills/intraday_peak_exit.py',
        ROOT/'skills/portfolio_intraday_audit.py',SPEC]


class StudyTicks(old.TickCache):
    def __init__(self,prepare=False):
        super().__init__(CACHE/'ticks',online=prepare,maximum=150)
        previous = study.strict.AdditionalTicks()
        self.donors = [previous.root,*[d.root for d in previous.donors]]
        self.files,self.queries = {},[]

    def fetch(self,dataset,sid,day,end=None):
        if not self.online:
            return super().fetch(dataset,sid,day,end)
        with file_lock(self.root/'budget.lock'):
            path = self.root/'budget.json'
            budget = old.read(path) if path.exists() else dict(reserved=0)
            if budget['reserved'] >= self.maximum:
                raise old.ReplayDataUnavailable('Shared intraday study 150-request ceiling reached')
            budget['reserved'] += 1; old.write(path,budget)
        self.calls += 1
        print('fetch tick',sid,day,'reserved',budget['reserved'],flush=True)
        try:
            return fetch_dataset(dataset,date.fromisoformat(day),data_id=sid,token=self.token,
                requests_per_hour=4860,max_retries=0,timeout=30)
        except FinMindError as exc:
            raise old.ReplayDataUnavailable(f'Tick request failed {sid} {day}: {type(exc).__name__}') from exc

    def get(self,sid,day,market):
        self.queries.append(dict(stock_id=sid,date=day,market=market))
        path = self.root/f'{sid}-{day}.parquet'
        for directory in self.donors:
            candidate = directory/path.name
            if candidate.exists() and candidate.with_suffix('.json').exists():
                result = old.TickCache(directory,online=False).get(sid,day,market)
                path = candidate
                break
        else:
            jpc = ROOT/'.cache/jpc-intraday-20260929'/path.name
            if sid=='6197' and day=='2026-07-13' and jpc.exists():
                meta = old.read(jpc.with_suffix('.json'))
                if old.sha(jpc) != meta['sha256']:
                    raise ValueError('JPC tick source changed')
                result = normalize_ticks(pd.read_parquet(jpc),sid,day,market),meta['sha256']
                path = jpc
            else:
                result = super().get(sid,day,market)
        self.files.update(old.file_identities([path,path.with_suffix('.json')],ROOT))
        return result


def audit_intraday(account):
    checked = 0
    for trade in account['trades']:
        if trade['reason'] != 'intraday_peak_stop15':
            continue
        if trade['channel']=='odd':
            if trade['date'] <= trade['signal_date']:
                raise ValueError('Odd exit borrowed same-day hindsight')
        else:
            if trade['fill_time'] <= trade['order_time']:
                raise ValueError('Board exit uses pre-order volume')
            if trade['reference_price'] <= trade['limit_price']:
                raise ValueError('Locked-down queue was counted')
        checked += 1
    return dict(intraday_exit_rows_checked=checked,entry_session_stop_excluded=True,
        odd_exits_strictly_after_trigger_day=True)


def validate_intraday_account(account, calendar, start, end):
    expected = [d for d in calendar if start <= d <= end]
    if [d['date'] for d in account['daily']] != expected:
        raise ValueError('Incomplete intraday account calendar')
    for trade in account['trades']:
        signal = trade.get('signal_date')
        same_day_board_stop = (trade['side']=='sell' and trade['channel']=='board'
            and trade['reason']=='intraday_peak_stop15')
        if not signal or signal > trade['date'] or (signal==trade['date'] and not same_day_board_stop):
            raise ValueError('Future or unpermitted same-day signal')
    return audit_intraday(account)


def run(output,prepare=False):
    output = Path(output).resolve()
    if output.exists(): raise ValueError('Use a new run directory')
    pub = old.load_selector()
    data,inputs,identity,_,quote_refs = study.strict.repaired_data(pub)
    data = restart_inputs(data)
    extra,extra_refs = study.load_exit_completion(ROOT)
    additions = old.parent.parent.parent.load_corporate_completion(ROOT)|old.load_capital_terms(ROOT)[0]|extra
    overrides = (old.read(old.parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        old.read(old.parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        old.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    refs = quote_refs|extra_refs|old.file_identities([*CODE,
        ROOT/'artifacts/forward_simulation/historical_selector_replay_20260925.json'],ROOT)
    old.write(output/'identity.json',refs)
    cases = {}
    provider = None
    if prepare:
        from scripts.prepare_mixed_odd_authorized import AuthorizedBudget,AuthorizedOdds,CACHE as APPROVED
        proof = CACHE/'official-recovery.json'
        if not proof.exists():
            old.write(proof,old.read(APPROVED/'official-recovery.json'))
        # The same exact-endpoint, 24-hour proof check and global origin holds
        # still apply. Preserve prior studies' mutable budget/index files.
        budget = AuthorizedBudget(CACHE/'odd-budget.json',maximum={'finmind':0,'official':60})
        provider = old.ReplayMarketFeeds(CACHE/'odd-feeds',offline=False,
            http_get=budget.official,official_min_interval=5)
    def execute():
        for arm,cls in ([('intraday15',PortfolioIntradayExit)] if prepare else
                        [('control',MidpointExitReplay),('intraday15',PortfolioIntradayExit)]):
            print('start',arm,flush=True)
            ticks = StudyTicks(prepare)
            odds = (AuthorizedOdds(ROOT,inputs/'execution-feeds',provider) if prepare
                    else study.OddDailyCache(ROOT,inputs/'execution-feeds'))
            feeds = old.ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
            corp = old.TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
            engine = cls(data.quotes,data.companies,data.days,data.entries,feeds,corp,
                start=data.start,end=data.end,ticks=ticks,participation=.01,liquidity_identity=identity,
                odd_feeds=odds,ordering='original',position_count=3,factor_mask=0,residual_policy='release',
                identity_report=identity,exit_signals=data.features,
                action_dates=list(zip(data.events.stock_id,data.events.event_date)),
                **(dict(stop_events=data.events) if arm=='intraday15' else {}))
            try:
                account = engine.run()
                validator = old.validate_completed_account if arm=='control' else validate_intraday_account
                validator(account,[str(d.date()) for d in data.days],data.start,data.end)
                audit = study.audit_high_return_resources(account,engine.resource_plans,
                    engine.slot_decisions,engine.board_decisions,engine.residual_days,data.quotes)
                if arm=='control':
                    expected = ROOT/'.cache/midpoint-exit-20260929/final-a/midpoint_exit.json'
                    if account != old.read(expected)['account']: raise ValueError('Control changed')
                    audit.update(audit_midpoint_exit(account,ticks,odds,study.market_routes([*ticks.queries,*odds.queries]),data.quotes,data.days,corp,feeds))
                    refs.update(old.file_identities([expected],ROOT))
                else:
                    audit.update(audit_intraday(account))
                    audit.update(audit_sources(account,data.quotes,data.days,data.events,
                        ticks,odds,feeds,engine.markets))
                result = dict(completed=True,summary=old.summarize(account),account=account,audit=audit)
                verify_cash(result)
            except (old.ReplayDataUnavailable,old.UnresolvedAction) as exc:
                result = dict(completed=False,summary=None,reason=str(exc),
                    completed_sessions=len(engine.daily),partial_trades=engine.trades,
                    intraday_evidence=engine.intraday_evidence if arm=='intraday15' else [])
            refs.update(ticks.files|odds.files)
            result.update(network_calls=ticks.calls,live_qualified=False,unseen_validation=False,actual_fill_verified=False)
            if prepare:
                result = {k:v for k,v in result.items() if k not in ('account','summary','audit','partial_trades')}
            old.write(output/(arm+'.json'),result)
            cases[arm] = dict(completed=result['completed'],reason=result.get('reason'),
                summary=result.get('summary'),network_calls=ticks.calls,
                path=str((output/(arm+'.json')).relative_to(ROOT)),sha256=old.sha(output/(arm+'.json')))
            append_trial_registry(dict(timestamp=datetime.now().isoformat(timespec='seconds'),
                source='portfolio_intraday15_20260929',params=dict(arm=arm,drawdown=.15),
                completed=result['completed'],preparation=prepare,result_path=cases[arm]['path'],live_qualified=False))
            print(arm,{k:cases[arm][k] for k in ('completed','reason','network_calls')},flush=True)
    if prepare: execute()
    else:
        with old.offline_only(): execute()
    old.write(output/'report.json',dict(start=data.start,end=data.end,cases=cases,source_sha256=refs,
        preparation=prepare,all_completed=all(c['completed'] for c in cases.values()),
        live_qualified=False,unseen_validation=False,actual_fill_verified=False))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();run(a.output,a.prepare)
