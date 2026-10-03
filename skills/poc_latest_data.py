"""Bounded data adapters for the unchanged POC strategy's October continuation."""
from copy import deepcopy
from datetime import date
from pathlib import Path
import json
import re
import pandas as pd
from scripts.export_poc_signal_explorer import load_evidence, ROOT, REPORTS, BASE, CASES
from scripts.prepare_volume_profile import verify_saved, write
from skills.volume_profile_data import AccountProfileData
from skills.replay_market_feeds import parse_odd, ReplayDataUnavailable
from skills.official_daily_acquisition import digest
from skills.poc_latest_odd import LatestOddData
from skills.poc_latest_execution import CACHE_ORDER

ANCHOR='2026-09-09'
PREREG=ROOT/'docs/prereg_poc_latest_20261003.md'
BUNDLE=ROOT/'.cache/poc-latest-20261003/inputs-v1'
OFFICIAL=ROOT/'.cache/poc-latest-20261003/official-v1'
ADAPTER_SOURCES=(
    'skills/poc_latest_data.py','skills/poc_latest_execution.py','skills/poc_latest_odd.py',
    'tests/test_poc_latest_data.py','tests/test_poc_latest_execution.py','tests/test_poc_latest_odd.py',
)


def checked_json(path,refs,root=ROOT):
    path=Path(path).resolve();key=str(path.relative_to(root));h=digest(path)
    if key in refs and refs[key]!=h:raise ValueError('Source changed: '+key)
    refs[key]=h;return json.loads(path.read_text())


def bind_closure(report,refs,root=ROOT):
    identities={}
    merge_refs(identities,report.get('source_sha256',{}))
    merge_refs(identities,report.get('output_sha256',{}))
    for k,h in identities.items():
        path=(root/k).resolve();path.relative_to(root)
        if k in refs and refs[k]!=h or digest(path)!=h:raise ValueError('Source closure changed: '+k)
        refs[k]=h


def merge_refs(refs,additions):
    for name,h in additions.items():
        if name in refs and refs[name]!=h:raise ValueError('Conflicting source hashes: '+name)
        refs[name]=h


def sealed_json(path,refs,root=ROOT):
    path=Path(path).resolve();sidecar=path.with_suffix('.sha256')
    if digest(path)!=sidecar.read_text().strip():raise ValueError('Sealed report hash differs: '+str(path))
    name=str(sidecar.relative_to(root));merge_refs(refs,{name:digest(sidecar)})
    return checked_json(path,refs,root)


def bind_adapter_sources(root,refs):
    root=Path(root).resolve()
    for name in ADAPTER_SOURCES:
        path=(root/name).resolve();path.relative_to(root)
        merge_refs(refs,{name:digest(path)})


class FrozenOddSources:
    """Only sealed old references, with the original raw-first per-stock semantics."""
    def __init__(self,root,source_refs):
        self.root=Path(root).resolve();self.source_refs=dict(source_refs)
        self.refs={};self.sources={};self.simple={};self.cached={}
        from skills import replay_market_feeds
        parser_hash=digest(Path(replay_market_feeds.__file__))
        directories=sorted({str(Path(name).parent) for name in self.source_refs
            if re.fullmatch(r'odd-(twse|tpex)-\d{4}-\d{2}-\d{2}\.raw\.json',Path(name).name)})
        for directory in directories:
            index_name=directory+'/index.json'
            if index_name not in self.source_refs:continue
            state=self._read(index_name)
            # Match the sealed OddDailyCache's handling of older parser indexes.
            if state.get('schema')!=1 or state.get('parser_sha256')!=parser_hash:continue
            for key,entry in state.get('entries',{}).items():
                if not re.fullmatch(r'odd:(twse|tpex):\d{4}-\d{2}-\d{2}',key):continue
                raw_name=self._child(directory,entry['raw_file'])
                if raw_name not in self.source_refs:continue
                self.sources.setdefault(key,[]).append((directory,index_name,deepcopy(entry)))
        for folder in CACHE_ORDER:
            for name in sorted(self.source_refs):
                path=Path(name)
                match=re.fullmatch(r'odd-(twse|tpex)-(\d{4}-\d{2}-\d{2})\.json',path.name)
                if str(path.parent)==folder and match:
                    self.simple.setdefault('odd:'+match[1]+':'+match[2],name)

    def _child(self,directory,name):
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ReplayDataUnavailable('Frozen odd index contains an unsafe filename')
        path=(self.root/directory/name).resolve();path.relative_to(self.root)
        return str(path.relative_to(self.root))

    def _read(self,name,expected=None):
        path=(self.root/name).resolve();path.relative_to(self.root)
        frozen=self.source_refs.get(name)
        if frozen is None or expected is not None and frozen!=expected:
            raise ReplayDataUnavailable('Frozen odd source is outside the sealed closure: '+name)
        if digest(path)!=frozen:raise ReplayDataUnavailable('Frozen odd source changed: '+name)
        merge_refs(self.refs,{name:frozen})
        return json.loads(path.read_text())

    def _raw_rows(self,key,market,day):
        unique={}
        for directory,index_name,entry in self.sources[key]:
            state=self._read(index_name)
            if state['entries'].get(key)!=entry:raise ReplayDataUnavailable('Frozen odd index entry changed')
            raw_name=self._child(directory,entry['raw_file'])
            rows_name=self._child(directory,entry['rows_file'])
            files=state['files_sha256']
            if entry['raw_file'] not in files or entry['rows_file'] not in files:
                raise ReplayDataUnavailable('Frozen odd index lacks source hashes')
            raw=self._read(raw_name,files[entry['raw_file']])
            normalized=self._read(rows_name,files[entry['rows_file']])
            raw_hash=files[entry['raw_file']]
            if normalized.get('schema')!=1 or normalized.get('raw_sha256')!=raw_hash:
                raise ReplayDataUnavailable('Frozen odd raw/normalized provenance disagrees')
            parsed=parse_odd(raw,market,date.fromisoformat(day))
            if parsed!=normalized.get('rows'):
                raise ReplayDataUnavailable('Frozen odd raw and normalized rows disagree')
            if entry.get('row_count') is not None and entry['row_count']!=len(parsed):
                raise ReplayDataUnavailable('Frozen odd row count differs from index')
            if raw_hash in unique and unique[raw_hash]!=parsed:
                raise ReplayDataUnavailable('Frozen identical raw source has conflicting normalized rows')
            unique.setdefault(raw_hash,parsed)
        return list(unique.values())

    def _simple_rows(self,key,market,day):
        name=self.simple.get(key)
        if name is None:raise ReplayDataUnavailable('Frozen odd source absent: '+key)
        record=self._read(name)
        if 'source_receipt' in record:
            for field in ('source_receipt','source_raw'):
                linked=self._read(record[field],record[field+'_sha256'])
                if field=='source_raw' and linked!=record.get('payload'):
                    raise ReplayDataUnavailable('Frozen odd wrapper differs from original raw payload')
        return [parse_odd(record,market,date.fromisoformat(day))]

    def get(self,day,sid,market):
        day=date.fromisoformat(day).isoformat();market=market.lower()
        if day>ANCHOR or market not in ('twse','tpex'):
            raise ValueError('Frozen odd query outside the original account scope')
        key='odd:'+market+':'+day
        if key not in self.cached:
            self.cached[key]=(self._raw_rows(key,market,day) if key in self.sources
                              else self._simple_rows(key,market,day))
        values=[rows[sid] for rows in self.cached[key] if sid in rows]
        if not values:raise ReplayDataUnavailable('Frozen odd stock absent: '+sid+' '+day)
        if any(value!=values[0] for value in values[1:]):
            raise ReplayDataUnavailable('Conflicting frozen odd rows: '+sid+' '+day)
        return deepcopy(values[0])


class LatestProfiles(AccountProfileData):
    def __init__(self,bundle,official,*,online=False):
        # Preserve the old, hash-bound normalizer and exact historical references.
        super().__init__(ROOT/'.cache/market-input-repair-20261002/inputs-v2',online=online,
            maximum_requests=2400,directory=ROOT/'.cache/poc-latest-20261003/profiles-v1')
        self.bundle=Path(bundle).resolve();official=Path(official).resolve()
        manifest=sealed_json(self.bundle/'manifest.json',self.refs,ROOT)
        if manifest['end']!='2026-10-02':raise ValueError('Profile extension has wrong endpoint')
        if manifest.get('events_extension_complete') is not True:
            raise ValueError('Profile extension lacks complete corporate events')
        for name,h in manifest['files_sha256'].items():self._mark(self.bundle/name,h)
        bind_closure(manifest,self.refs,ROOT);self.manifest=manifest
        self.days=pd.DatetimeIndex(pd.read_parquet(self.bundle/'close-official.parquet',columns=['date']).date)
        self.calendar=self.days.strftime('%Y-%m-%d').tolist()
        self.events=pd.read_parquet(self.bundle/'events.parquet');self.events.event_date=pd.to_datetime(self.events.event_date)
        report=sealed_json(official/'report.json',self.refs,ROOT)
        if (report.get('schema')!='poc_latest_official_extension_v1'
            or report.get('start')!='2026-09-10' or report.get('end')!='2026-10-02'
            or report.get('daily_tables_extension_complete') is not True
            or report.get('corporate_events_extension_complete') is not True):
            raise ValueError('Official extension report is incomplete or has wrong scope')
        bind_closure(report,self.refs,ROOT)
        normalized_name=report.get('normalized_path');sources_name=report.get('sources_path')
        outputs=report.get('output_sha256',{})
        if not normalized_name or not sources_name or any(name not in outputs for name in (normalized_name,sources_name)):
            raise ValueError('Official profile tables are not bound by the report')
        self._mark(ROOT/normalized_name,outputs[normalized_name])
        self._mark(ROOT/sources_name,outputs[sources_name])
        metadata=self._json(ROOT/sources_name)
        meta=metadata.get('sources',metadata)
        extension=pd.read_parquet(ROOT/normalized_name)
        extension['date']=pd.to_datetime(extension.date).dt.strftime('%Y-%m-%d')
        extension['market']=extension.market.str.upper()
        if not len(extension) or extension.date.min()<=ANCHOR or extension.date.max()!='2026-10-02':
            raise ValueError('Need complete new official market period')
        actual=set(zip(extension.market,extension.date))
        expected={(m,d) for m in ('TWSE','TPEX') for d in self.calendar if d>ANCHOR}
        if (actual!=expected or report.get('required_market_days')!=len(expected)
            or report.get('accepted_market_days')!=len(expected) or report.get('missing_market_days')!=[]):
            raise ValueError('Missing official market-day coverage')
        if not set(extension.source_id).issubset(meta):raise ValueError('Official table has an unbound source id')
        existing=self.official.reset_index();combined=pd.concat([existing,extension],ignore_index=True)
        if combined.duplicated(['stock_id','date']).any():raise ValueError('Official profile identity duplicate')
        self.official=combined.set_index(['stock_id','date']).sort_index()
        if set(meta)&set(self.official_meta['sources']):raise ValueError('Official source-id collision')
        self.official_meta['sources'].update(meta)
        self._mark(Path(__file__));self._mark(PREREG)

    def _raw(self,coordinate):
        sid,day=coordinate;key=sid+'-'+day;dest=self.directory/'receipts'/(key+'.json')
        if not dest.exists():
            prior=ROOT/'.cache/volume-profile-account-20261003/profiles-v1/receipts'/(key+'.json')
            if prior.exists():
                query=dict(dataset='TaiwanStockPriceTick',data_id=sid,start_date=day)
                self._mark(prior);old=verify_saved(json.loads(prior.read_text()),query)
                if old['status'] in ('received','cached'):
                    item=dict(old,reused_receipt=str(prior.relative_to(ROOT)))
                    write(dest,item)
        return super()._raw(coordinate)


class LatestAccountData:
    def __init__(self,root=ROOT,*,online=False):
        self.root=Path(root).resolve()
        if self.root!=ROOT:raise ValueError('Latest account uses this repository only')
        self.bundle=BUNDLE;self.online=online;self.refs={}
        self.prereg_path=PREREG;self.prereg_sha256=digest(PREREG)
        self.refs[str(PREREG.relative_to(ROOT))]=self.prereg_sha256
        bind_adapter_sources(ROOT,self.refs)
        self.cases,self.frozen_profiles,old_refs=load_evidence();merge_refs(self.refs,old_refs)
        anchor_report=checked_json(ROOT/BASE/'anchors-v2/report.json',self.refs)
        benchmark=anchor_report['cases']['benchmark'];path=ROOT/benchmark['path']
        if digest(path)!=benchmark['sha256']:raise ValueError('Benchmark anchor changed')
        self.cases['benchmark']=checked_json(path,self.refs)
        self.query_ids={arm:{q['event_id'] for q in c['profile_queries']} for arm,c in self.cases.items()}
        self.profile_index={arm:{p['event_id']:p for p in self.frozen_profiles.get(arm,[])} for arm in self.cases}
        self.used={arm:{} for arm in self.cases}
        manifest=sealed_json(self.bundle/'manifest.json',self.refs)
        for name,h in manifest['files_sha256'].items():
            path=self.bundle/name
            if digest(path)!=h:raise ValueError('Extended input changed: '+name)
            merge_refs(self.refs,{str(path.relative_to(ROOT)):h})
        if manifest.get('events_extension_complete') is not True:
            raise ReplayDataUnavailable('Latest account requires a verified corporate-event extension')
        bind_closure(manifest,self.refs)
        self.latest_supported_account_end='2026-10-02'
        self.corporate_overrides={};self.cash_supplements=[]
        from skills.poc_latest_execution import LatestExecutionData
        self.execution=LatestExecutionData(ROOT,online=online)
        self.dividend_directory=self.execution.dividend_directory
        self.profiles=LatestProfiles(self.bundle,OFFICIAL,online=online)
        self.odds=LatestOddData(ROOT,online=online)
        self.frozen_odds=FrozenOddSources(ROOT,old_refs)
        merge_refs(self.refs,self.frozen_odds.refs)

    def finmind(self,sid,dataset):
        frame=self.execution.finmind(sid,dataset);merge_refs(self.refs,self.execution.refs);return frame

    def get_odd(self,day,sid,market,engine=None):
        market=market.lower()
        if day>ANCHOR:
            result=self.odds.get(day,sid,market,engine);merge_refs(self.refs,self.odds.refs);return result
        result=self.frozen_odds.get(day,sid,market)
        merge_refs(self.refs,self.frozen_odds.refs)
        return result

    def profile(self,arm,event):
        eid=event['event_id']
        if event['entry_date']<=ANCHOR:
            if eid not in self.query_ids[arm] or eid not in self.profile_index[arm]:
                raise ValueError('Historical POC query path changed: '+arm+' '+eid)
            result=deepcopy(self.profile_index[arm][eid])
        else:
            result=self.profiles(event)
        self.used[arm][eid]=result
        return result

    def profile_snapshot(self,output):
        output=Path(output);arms={}
        for arm,items in self.used.items():
            path=output/arm/'profile-features.json';write(path,list(items.values()))
            merge_refs(self.refs,{str(path.relative_to(ROOT)):digest(path)})
            arms[arm]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),profiles=len(items))
        new=self.profiles.snapshot(output/'extension')
        merge_refs(self.refs,self.profiles.refs);merge_refs(self.refs,self.execution.refs);merge_refs(self.refs,self.odds.refs)
        return dict(schema='poc_latest_profiles_v1',arms=arms,extension=new,maximum_adapter_requests=2400,
                    adapter_requests=new['adapter_requests'])

    def verify_sources(self):
        merge_refs(self.refs,self.profiles.refs);merge_refs(self.refs,self.execution.refs);merge_refs(self.refs,self.odds.refs)
        for name,h in self.refs.items():
            if digest(ROOT/name)!=h:raise ValueError('Latest data source changed: '+name)
