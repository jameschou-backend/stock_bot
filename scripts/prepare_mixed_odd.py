#!/usr/bin/env python3
"""Complete only unrestricted TPEx odd daily evidence; TWSE stays held."""
from pathlib import Path
import sys
import json
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_mixed_odd as study
from scripts.prepare_theme_catalyst import Budget
from skills.odd_daily_cache import OddDailyCache
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable,URLS
from app.file_lock import file_lock

CACHE=ROOT/'.cache/mixed-odd-completion-20260928'
old=study.old


class CompletionBudget(Budget):
    def official(self,url,**kwargs):
        if url!=URLS['tpex']:
            raise ReplayDataUnavailable('TWSE origin hold preserved; only TPEx acquisition allowed')
        stopped=self.path.parent/'security-stopped.json'
        if stopped.exists():raise ReplayDataUnavailable('Source completion stopped after security response')
        response=super().official(url,allow_redirects=False,**kwargs)
        if response.status_code in (401,403,428) or 300<=response.status_code<400 or b'FOR SECURITY REASONS' in response.content:
            old.write(stopped,dict(url=url,status=response.status_code))
            raise ReplayDataUnavailable('Official source security response; no retry')
        return response


class PreparedOdds(OddDailyCache):
    def __init__(self,root,primary,provider):
        super().__init__(root,primary);self.provider=provider
    def get_odd(self,day,sid,market):
        key=f'odd:{market.lower()}:{day}'
        # Existing conflicts, invalid files and missing stock rows never trigger
        # a replacement download. Only absent market-date tables may be fetched.
        if key not in self.sources:
            if market.lower()!='tpex':
                raise ReplayDataUnavailable('TWSE origin hold preserved; missing official daily odd source: '+key)
            self.provider.get_odd(day,sid,market)
            state=self.provider._state()
            if key not in state['entries']:raise ReplayDataUnavailable('Requested official table was not stored')
            self.sources[key]=[(self.provider.cache_dir,state,state['entries'][key])]
        return super().get_odd(day,sid,market)


def prepare(fetch_ticks=False):
    CACHE.mkdir(parents=True,exist_ok=True)
    with file_lock(CACHE/'prepare.lock',timeout=0):
        previous=CACHE/'receipt.json'
        if previous.exists():old.write(CACHE/'history'/(old.sha(previous)+'.json'),old.read(previous))
        before=old.read(study.strict.PREP/'ticks/budget.json')['reserved']
        budget=CompletionBudget(CACHE/'budget.json',maximum={'finmind':0,'official':20})
        pub=old.load_selector();data,inputs,identity,repairs,refs=study.strict.repaired_data(pub)
        additions=old.parent.parent.parent.load_corporate_completion(ROOT)|old.load_capital_terms(ROOT)[0]|study.load_exit_completion(ROOT)[0]
        provider=ReplayMarketFeeds(CACHE/'feeds',offline=False,http_get=budget.official,official_min_interval=5)
        odds=PreparedOdds(ROOT,inputs/'execution-feeds',provider)
        results={}
        for name,config in study.CASES.items():
            print('prepare',name,flush=True)
            result=study.case(data,inputs,identity,additions,**config,ticks=study.strict.AdditionalTicks(prepare=fetch_ticks),odds=odds)
            results[name]=dict(completed=result['completed'],reason=result.get('reason'),network_calls=result['network_calls'])
            old.write(CACHE/'progress.json',results)
            print(name,results[name],flush=True)
        sources=old.file_identities([Path(__file__),ROOT/'scripts/prepare_theme_catalyst.py',
            ROOT/'scripts/prepare_sector_account_sources.py',CACHE/'budget.json',
            *sorted((CACHE/'feeds').glob('*.json'))],ROOT)
        value=dict(schema='mixed_odd_preparation_v1',cases=results,all_completed=all(r['completed'] for r in results.values()),
            official_requests=budget.state['attempts']['official'],finmind_requests=sum(r['network_calls'] for r in results.values()),
            finmind_reserved_before=before,finmind_reserved_after=old.read(study.strict.PREP/'ticks/budget.json')['reserved'],
            source_sha256=sources,live_qualified=False,performance_report=False,twse_hold_preserved=True)
        old.write(CACHE/'receipt.json',value)
        return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch-ticks',action='store_true',help='Allow necessary board ticks within the existing shared 300-request ceiling')
    print(json.dumps(prepare(parser.parse_args().fetch_ticks),ensure_ascii=False,indent=2))
