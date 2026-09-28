#!/usr/bin/env python3
"""User-authorized exact-endpoint odd-lot completion after one normal probe."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_mixed_odd as study
from scripts.prepare_holder_flow_accounts import ExecutionBudget
from skills.odd_daily_cache import OddDailyCache
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable,URLS
from app.file_lock import file_lock

CACHE=ROOT/'.cache/mixed-odd-authorized-20260928'
old=study.old


class AuthorizedBudget(ExecutionBudget):
    def official(self,url,**kwargs):
        if url not in URLS.values():raise ReplayDataUnavailable('Unapproved official endpoint')
        stopped=self.path.parent/'security-stopped.json'
        if stopped.exists():raise ReplayDataUnavailable('Security stop active; no request sent')
        # ExecutionBudget verifies the exact endpoint, raw proof hash and its
        # 24-hour freshness. All other origins retain their existing holds.
        response=super().official(url,allow_redirects=False,**kwargs)
        if response.status_code in (401,403,428) or 300<=response.status_code<400 or b'FOR SECURITY REASONS' in response.content:
            old.write(stopped,dict(url=url,status=response.status_code,automatic_retry=False))
            raise ReplayDataUnavailable('Security or redirect response; no retry')
        return response


class AuthorizedOdds(OddDailyCache):
    def __init__(self,root,primary,provider):
        super().__init__(root,primary);self.provider=provider
    def get_odd(self,day,sid,market):
        key=f'odd:{market.lower()}:{day}'
        if key not in self.sources:
            self.provider.get_odd(day,sid,market)
            state=self.provider._state()
            if key not in state['entries']:raise ReplayDataUnavailable('Official table was not stored')
            self.sources[key]=[(self.provider.cache_dir,state,state['entries'][key])]
        return super().get_odd(day,sid,market)


def prepare():
    with file_lock(CACHE/'prepare.lock',timeout=0):
        authorization=old.read(CACHE/'probe-attempt.json')
        if authorization.get('authorized_by')!='user go after explicit connection permission explanation':
            raise ValueError('Missing scoped user authorization record')
        proof=old.read(CACHE/'official-recovery.json')
        if old.sha(ROOT/proof['path'])!=proof['sha256']:raise ValueError('Probe evidence changed')
        if (CACHE/'security-stopped.json').exists():raise ReplayDataUnavailable('Probe stopped; no acquisition allowed')
        previous=CACHE/'receipt.json'
        if previous.exists():old.write(CACHE/'history'/(old.sha(previous)+'.json'),old.read(previous))
        before=old.read(study.strict.PREP/'ticks/budget.json')['reserved']
        # Probe is counted separately; 119 subsequent calls => 120 total max.
        budget=AuthorizedBudget(CACHE/'budget.json',maximum={'finmind':0,'official':119})
        hold=ROOT/'.cache/official-origin-holds/www.twse.com.tw.json';hold_hash=old.sha(hold)
        pub=old.load_selector();data,inputs,identity,repairs,refs=study.strict.repaired_data(pub)
        additions=old.parent.parent.parent.load_corporate_completion(ROOT)|old.load_capital_terms(ROOT)[0]|study.load_exit_completion(ROOT)[0]
        provider=ReplayMarketFeeds(CACHE/'feeds',offline=False,http_get=budget.official,official_min_interval=5)
        odds=AuthorizedOdds(ROOT,inputs/'execution-feeds',provider)
        results={}
        for name,config in study.CASES.items():
            print('prepare',name,flush=True)
            result=study.case(data,inputs,identity,additions,**config,ticks=study.strict.AdditionalTicks(prepare=True),odds=odds)
            results[name]=dict(completed=result['completed'],reason=result.get('reason'),network_calls=result['network_calls'])
            old.write(CACHE/'progress.json',results)
            print(name,results[name],flush=True)
        if old.sha(hold)!=hold_hash:raise ValueError('Global hold changed during authorized acquisition')
        sources=old.file_identities([Path(__file__),ROOT/'scripts/prepare_holder_flow_accounts.py',
            ROOT/'scripts/prepare_theme_catalyst.py',ROOT/'scripts/prepare_sector_account_sources.py',
            hold,CACHE/'budget.json',CACHE/'probe-attempt.json',CACHE/'probe-response.json',
            CACHE/'probe-response.bin',CACHE/'official-recovery.json',*sorted((CACHE/'feeds').glob('*.json'))],ROOT)
        value=dict(schema='mixed_odd_authorized_preparation_v1',cases=results,
            all_completed=all(r['completed'] for r in results.values()),
            official_requests=1+budget.state['attempts']['official'],
            finmind_requests=sum(r['network_calls'] for r in results.values()),
            finmind_reserved_before=before,finmind_reserved_after=old.read(study.strict.PREP/'ticks/budget.json')['reserved'],
            source_sha256=sources,live_qualified=False,performance_report=False,global_hold_unchanged=True)
        old.write(CACHE/'receipt.json',value)
        return value


if __name__=='__main__':
    print(json.dumps(prepare(),ensure_ascii=False,indent=2))
