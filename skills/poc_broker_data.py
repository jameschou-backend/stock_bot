"""Isolated data and explicit missing-evidence policy for POC/broker accounts."""
from copy import copy, deepcopy
from datetime import date, datetime, timezone
import json
from pathlib import Path

import pandas as pd

from scripts.prepare_volume_profile import verify_saved, write
from scripts.research_exit_scenarios import read, sha
from skills.frozen_dividend_copy import ensure_dividend_copy
from skills.poc_latest_data import (
    ROOT, LatestAccountData, LatestProfiles, BUNDLE, OFFICIAL, FrozenOddSources,
    merge_refs,
)
from skills.poc_latest_execution import LatestExecutionData, validate_frame
from skills.replay_market_feeds import ReplayDataUnavailable

BASE = ROOT/'.cache/poc-broker-account-20261004'
PREREG = ROOT/'docs/prereg_poc_broker_account_20261004.md'
DIAGNOSTIC = ROOT/'.cache/broker-branch-research-20261004/final-b'
LATEST = ROOT/'.cache/poc-latest-20261003/run-v3/report.json'
ARMS = ('poc_red', 'poc_persist_guard', 'poc_combined_guard',
        'poc_known5_control', 'poc_known5_filter', 'poc_known5_combined')


def fetch_broker_execution(*args, **kwargs):
    """Use the shared Sponsor limit once; the limiter keeps its 10% reserve."""
    from app.config import load_config
    from app.finmind import fetch_dataset
    kwargs['requests_per_hour']=min(6000,load_config().finmind_requests_per_hour)
    return fetch_dataset(*args,**kwargs)


def apply_broker_gate(arm, entries, evidence):
    """Stable gate using only source-matched signal-day branch features.

    Unknown retains its own state. Only the explicitly named guard arms retain
    unknown candidates; matched-subset arms exclude them from *both* sides.
    """
    if arm not in ARMS:
        raise ValueError('Unregistered POC/broker account arm')
    selected, decisions, seen = [], [], set()
    for event in entries:
        if len(event.get('members', [])) != 1:
            raise ValueError('Exactly one stock per branch candidate required')
        eid, sid, day = event['event_id'], event['members'][0], event['signal_date']
        if eid in seen or not day < event['entry_date']:
            raise ValueError('Unique T+1 candidate identities required')
        seen.add(eid)
        item = evidence.get(eid)
        if item is None:
            raise ValueError('Every candidate needs an explicit branch evidence state')
        if item['stock_id'] != sid or item['signal_date'] != day:
            raise ValueError('Branch evidence stock/signal mismatch')
        known = item.get('known5')
        if type(known) is not bool:
            raise ValueError('Explicit five-day availability required')
        if known and any(type(item.get(key)) is not bool for key in ('persistent5', 'concentrated')):
            raise ValueError('Known branch evidence needs explicit conditions')
        persistent = item.get('persistent5') if known else None
        combined = persistent and item['concentrated'] if known else None
        if arm == 'poc_red':
            keep, reason = True, 'original_unchanged'
        elif not known:
            keep = arm.endswith('_guard')
            reason = 'unknown_kept_by_guard_policy' if keep else 'outside_matched_coverage'
        elif arm == 'poc_known5_control':
            keep, reason = True, 'known5_control'
        else:
            keep = combined if 'combined' in arm else persistent
            reason = 'branch_condition_passed' if keep else 'branch_condition_failed'
        decisions.append(dict(event_id=eid,stock_id=sid,signal_date=day,
            entry_date=event['entry_date'],known5=known,persistent5=persistent,
            combined=combined,kept=keep,reason=reason,
            unknown_reason=item.get('unknown_reason'),source_event_id=item.get('source_event_id')))
        if keep:
            selected.append(deepcopy(event))
    return selected, decisions


class BrokerProfiles(LatestProfiles):
    """Same POC algorithm, new request budget and reusable authentic tick files."""
    def __init__(self, *, online=False):
        super().__init__(BUNDLE, OFFICIAL, online=False)
        self.directory = BASE/'profiles-v1'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.online = online
        self.maximum = 2400
        self._mark(PREREG); self._mark(Path(__file__))

    def _raw(self, coordinate):
        sid, day = coordinate
        key = sid+'-'+day
        query = dict(dataset='TaiwanStockPriceTick',data_id=sid,start_date=day)
        dest = self.directory/'receipts'/(key+'.json')
        if not dest.exists():
            candidates = [ROOT/f'.cache/poc-latest-20261003/profiles-v1/receipts/{key}.json',
                          ROOT/f'.cache/volume-profile-account-20261003/profiles-v1/receipts/{key}.json']
            from skills.volume_profile_data import PILOT
            pilot=PILOT/'tapes-v1/receipts'/(key+'.json')
            candidates.append(pilot)
            # Bind completed immutable attempts, not an active acquisition pointer.
            folder=ROOT/'.cache/poc-daily-opportunities-20261004/data-v1/attempts'
            candidates += sorted(folder.glob(key+'-*.json'),reverse=True)
            for source in candidates:
                if not source.exists():
                    continue
                item = read(source)
                if item.get('query') != query:
                    raise ValueError('Reusable POC receipt identity changed')
                if item.get('status') not in ('received','cached'):
                    continue
                verify_saved(item, query)
                self._mark(source,self.receipt_hashes[str(source.relative_to(ROOT))]
                           if source==pilot else None)
                self._mark(ROOT/item['raw_path'],item['raw_sha256'])
                write(dest,dict(item,reused_receipt=str(source.relative_to(ROOT)),
                                reused_receipt_sha256=sha(source)))
                break
        if dest.exists():
            copied=read(dest)
            if copied.get('reused_receipt'):
                source=(ROOT/copied['reused_receipt']).resolve()
                source.relative_to(ROOT)
                expected=copied.get('reused_receipt_sha256')
                if not expected:
                    raise ValueError('Reused POC receipt lacks source hash')
                self._mark(source,expected)
                original=read(source)
                verify_saved(original,query)
                if original.get('query') != query or any(
                        original.get(k) != copied.get(k) for k in ('status','raw_path','raw_sha256')):
                    raise ValueError('Reused POC receipt identity or raw evidence changed')
        # A separate shallow view avoids toggling online on a shared object;
        # the parent evaluates four independent stock-days concurrently.
        cache_view=copy(self);cache_view.online=False
        cached=super(BrokerProfiles,cache_view)._raw(coordinate)
        if cached['status'] != 'not_requested' or not self.online:
            return cached
        return self._fetch_raw(coordinate,query,dest)

    def _fetch_raw(self,coordinate,query,dest):
        """New bounded requests use the same 6000 -> 5400 shared budget."""
        from app.config import load_config
        from app.finmind import fetch_dataset, FinMindError, FinMindQuotaError
        from app.rate_limiter import get_rate_limiter
        sid,day=coordinate;key=sid+'-'+day
        reservation=self.directory/'attempts'/(key+'.json')
        if self._config is None:
            self._config=load_config()
        limit=min(6000,self._config.finmind_requests_per_hour)
        with self._lock:
            if dest.exists():
                return verify_saved(read(dest),query)
            if reservation.exists():
                self._mark(reservation)
                if read(reservation)['query'] != query:
                    raise ValueError('Reserved request identity changed')
                return dict(query=query,status='orphaned_started_attempt')
            stats=get_rate_limiter(limit).get_stats()
            if (self._stop.is_set() or stats.remaining_requests < 4 or stats.retry_after_seconds > 0
                    or len(list(reservation.parent.glob('*.json'))) >= self.maximum):
                return dict(query=query,status='request_budget_or_quota_paused')
            item=dict(query=query,status='started',started_at=datetime.now(timezone.utc).isoformat())
            write(reservation,item);write(dest,item)
        try:
            frame=fetch_dataset('TaiwanStockPriceTick',date.fromisoformat(day),data_id=sid,
                token=self._config.finmind_token,requests_per_hour=limit,timeout=40,max_retries=0)
        except FinMindQuotaError as exc:
            self._stop.set()
            item.update(status='quota_paused',retry_after_seconds=exc.retry_after_seconds)
        except FinMindError:
            item.update(status='provider_error',error_type='FinMindError')
        else:
            path=self.directory/'raw'/(key+'.parquet');path.parent.mkdir(exist_ok=True)
            temporary=path.with_suffix('.tmp.parquet');frame.to_parquet(temporary,index=False);temporary.replace(path)
            item.update(status='received' if len(frame) else 'empty',raw_path=str(path.relative_to(ROOT)),
                raw_sha256=sha(path),rows=len(frame),retrieved_at=frame.attrs.get('retrieved_at'),
                cache_hit=bool(frame.attrs.get('cache_hit',False)))
        write(dest,item);self._mark(dest);self._mark(reservation)
        if item.get('raw_path'):
            self._mark(ROOT/item['raw_path'],item['raw_sha256'])
        return item


class BranchAccountData(LatestAccountData):
    def __init__(self, root=ROOT, *, online=False):
        super().__init__(root,online=False)
        self.online=online
        self.prereg_path=PREREG;self.prereg_sha256=sha(PREREG)
        self.refs[str(PREREG.relative_to(ROOT))]=self.prereg_sha256
        self.refs[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
        published=ROOT/'artifacts/forward_simulation/broker_branch_diagnostics_20261004.json'
        expected=published.with_suffix('.sha256').read_text().split()[0]
        self._bind(published,expected)
        publication=read(published)
        report_path=DIAGNOSTIC/'report.json';rows_path=DIAGNOSTIC/'rows.json'
        self._bind(report_path,publication['evidence'][str(report_path.relative_to(ROOT))])
        self._bind(rows_path,publication['evidence'][str(rows_path.relative_to(ROOT))])
        diagnostic=read(report_path)
        for name,digest in diagnostic['source_sha256'].items():
            self._bind(ROOT/name,digest)
        lookup={}
        for row in read(rows_path):
            key=(row['stock_id'],row['signal_date'])
            if key in lookup:
                raise ValueError('Duplicate diagnostic stock/signal identity')
            branch, p=row['branch'],row['persistence5']
            if type(branch['known']) is not bool or type(p['known']) is not bool:
                raise ValueError('Diagnostic availability must be explicit boolean')
            known=branch['known'] and p['known']
            lookup[key]=dict(stock_id=key[0],signal_date=key[1],known5=known,
                persistent5=p['passed'] if known else None,
                concentrated=branch['concentrated_directional'] if known else None,
                source_event_id=row['event_id'],unknown_reason=None if known else
                branch.get('reason') or p.get('reason'))
        from scripts.research_poc_latest_account import load_candidate_bundle
        active,pending=load_candidate_bundle(BUNDLE)
        self.broker_evidence={e['event_id']:deepcopy(lookup.get((e['members'][0],e['signal_date']),
            dict(stock_id=e['members'][0],signal_date=e['signal_date'],known5=False,
                 persistent5=None,concentrated=None,unknown_reason='no_cached_branch_snapshot')))
            for e in active+pending}
        expected=LATEST.with_suffix('.sha256').read_text().strip()
        self._bind(LATEST,expected)
        latest=read(LATEST)
        self.latest_refs=dict(latest['source_sha256'])
        # Existing frozen profiles reproduce the original path byte for byte.
        self.latest_profiles=self.profiles
        self.profiles=BrokerProfiles(online=online)
        self.used={arm:{} for arm in ARMS}
        self.execution=LatestExecutionData(ROOT,online=online,
            source_refs=self.latest_refs,maximum_requests=300)
        self.execution.fetcher=fetch_broker_execution
        self.execution.directory=BASE/'execution-v1'
        self.execution.dividend_directory=self.execution.directory/'dividends'
        self.execution.directory.mkdir(parents=True,exist_ok=True)
        self.dividend_directory=self.execution.dividend_directory
        self.execution_loaded={}
        self.original_odds=self.frozen_odds
        self.supplemental_odds=None
        self.broker_odds=None

    def _bind(self,path,expected=None):
        path=Path(path).resolve();path.relative_to(ROOT)
        actual=sha(path)
        if expected is not None and actual!=expected:
            raise ValueError('Broker-account source hash changed: '+str(path))
        merge_refs(self.refs,{str(path.relative_to(ROOT)):actual})

    def finmind(self,sid,dataset):
        key=(sid,dataset)
        if key not in self.execution_loaded:
            previous=ROOT/f'.cache/poc-latest-20261003/execution-v1/{sid}-{dataset}.parquet'
            name=str(previous.relative_to(ROOT))
            if name in self.latest_refs:
                self._bind(previous,self.latest_refs[name])
                meta=previous.with_suffix('.json')
                self._bind(meta,self.latest_refs[str(meta.relative_to(ROOT))])
                record=read(meta)
                if (record['stock_id']!=sid or record['dataset']!=dataset or record['end']!='2026-10-02'
                        or record['sha256']!=sha(previous)):
                    raise ValueError('Frozen execution identity changed')
                frame=pd.read_parquet(previous)
                validate_frame(frame,sid,dataset,'2018-01-01','2026-10-02')
            else:
                receipt=self.execution.directory/'receipts'/(sid+'-'+dataset+'.json')
                if self.online and not receipt.exists():
                    from app.config import load_config
                    from app.rate_limiter import get_rate_limiter
                    config=load_config()
                    stats=get_rate_limiter(min(6000,config.finmind_requests_per_hour)).get_stats()
                    if stats.remaining_requests < 4 or stats.retry_after_seconds > 0:
                        raise ReplayDataUnavailable('Broker execution shared quota paused before request')
                frame=self.execution.finmind(sid,dataset)
                merge_refs(self.refs,self.execution.refs)
            if dataset=='TaiwanStockDividend':
                target=self.dividend_directory/(sid+'.parquet')
                ensure_dividend_copy(frame,target,prepare=True)
                self._bind(target)
            self.execution_loaded[key]=frame.copy(deep=True)
        return self.execution_loaded[key].copy(deep=True)

    def profile(self,arm,event):
        if arm not in ARMS:
            raise ValueError('Unregistered broker account profile arm')
        item=self.broker_evidence.get(event['event_id'])
        if (len(event.get('members',[])) != 1 or item is None
                or item['stock_id'] != event['members'][0]
                or item['signal_date'] != event['signal_date']):
            raise ValueError('Profile candidate stock/signal identity mismatch')
        eid=event['event_id']
        if event['entry_date']<='2026-09-09' and eid in self.profile_index['poc_red']:
            result=deepcopy(self.profile_index['poc_red'][eid])
        elif arm=='poc_red':
            if event['entry_date']<='2026-09-09':
                raise ValueError('Original baseline profile query changed')
            result=self.latest_profiles(event)
            merge_refs(self.refs,self.latest_profiles.refs)
        else:
            result=self.profiles(event)
        self.used[arm][eid]=result
        return result

    def get_odd(self,day,sid,market,engine=None):
        try:
            result=super().get_odd(day,sid,market,engine)
            return result
        except ReplayDataUnavailable as exc:
            # Supplement only an absent market-day, never conflicting or invalid evidence.
            if not str(exc).startswith(('Frozen odd source absent:', 'Latest official odd day missing:')):
                raise
        if self.supplemental_odds is None:
            from skills.poc_broker_odd import SupplementaryOddSources
            inventory=BASE/'odd-source-inventory.json'
            saved=read(inventory) if inventory.exists() else None
            self.supplemental_odds=SupplementaryOddSources(ROOT,source_refs=saved)
            if saved is None:
                write(inventory,self.supplemental_odds.source_refs)
            self._bind(inventory)
            self._bind(ROOT/'skills/poc_broker_odd.py')
        from skills.poc_broker_odd import OddMarketDayMissing
        try:
            result=self.supplemental_odds.get(day,sid,market)
            return result
        except OddMarketDayMissing:
            if self.broker_odds is None:
                from skills.poc_broker_odd_acquisition import BrokerOddData
                self.broker_odds=BrokerOddData(ROOT,online=self.online)
                self._bind(ROOT/'skills/poc_broker_odd_acquisition.py')
            return self.broker_odds.get(day,sid,market,engine)
        finally:
            merge_refs(self.refs,self.supplemental_odds.refs)
            if self.broker_odds is not None:
                merge_refs(self.refs,self.broker_odds.refs)

    def profile_snapshot(self,output):
        result=super().profile_snapshot(output)
        if self.broker_odds is not None:
            result['broker_odd_acquisition']=self.broker_odds.snapshot()
            merge_refs(self.refs,self.broker_odds.refs)
        return result

    def verify_sources(self):
        merge_refs(self.refs,self.latest_profiles.refs)
        if self.supplemental_odds is not None:
            merge_refs(self.refs,self.supplemental_odds.refs)
        if self.broker_odds is not None:
            merge_refs(self.refs,self.broker_odds.refs)
        super().verify_sources()
