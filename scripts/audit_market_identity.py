#!/usr/bin/env python3
"""Join dated official market episodes; retain unknown starts and security classes."""
from datetime import date
from pathlib import Path
import argparse
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bs4 import BeautifulSoup
from scripts.research_exit_scenarios import read,write,sha
from scripts.research_adjustment_continuation import (
    load_membership_sources,parse_twse_listings,parse_recent_tpex_listings,CACHE as OLD)

CACHE=ROOT/'.cache/readiness-completion-20260914'


def parse_isin(blob,market):
    if market not in ('TWSE','TPEx'):raise ValueError('Unsupported official market')
    soup=BeautifulSoup(blob,'html.parser')
    text=soup.get_text(' ',strip=True)
    expected='本國上市證券' if market=='TWSE' else '本國上櫃證券'
    if expected not in text or '上市日' not in text or 'CFICode' not in text:
        raise ValueError('Unexpected ISIN table identity')
    stamp=re.search(r'最近更新日期:(\d{4}/\d{2}/\d{2})',text)
    if not stamp:raise ValueError('Missing official snapshot date')
    snapshot=stamp.group(1).replace('/','-')
    rows=[];category=''
    for tr in soup.find_all('tr'):
        cells=[c.get_text(' ',strip=True) for c in tr.find_all('td',recursive=False)]
        if len(cells)==1:category=cells[0]
        if not cells or not re.match(r'^\d{4}\s',cells[0]):continue
        if len(cells)!=7:raise ValueError('Incomplete four-digit ISIN record')
        sid,name=cells[0].split(maxsplit=1)
        start=cells[2].replace('/','-');date.fromisoformat(start)
        expected_market = '上市臺灣創新板' if market=='TWSE' and category=='創新板' else ('上市' if market=='TWSE' else '上櫃')
        if cells[3] != expected_market:
            raise ValueError('ISIN market does not match source')
        if category not in ('股票','ETF','臺灣存託憑證(TDR)','創新板'):
            raise ValueError('Unreviewed security category: '+category)
        rows.append(dict(stock_id=sid,name=name,isin=cells[1],start=start,market=market,
                         category=category,cfi=cells[5],snapshot_date=snapshot))
    if not rows or len({r['stock_id'] for r in rows})!=len(rows):
        raise ValueError('Empty/duplicate official four-digit market snapshot')
    return rows


def resolve_on(episodes,sid,day):
    date.fromisoformat(day)
    known=[r for r in episodes if r['stock_id']==sid and r['start'] is not None
           and r['start']<=day and (r['end'] is None or day<r['end'])
           and (not r.get('snapshot_date') or day<=r['snapshot_date'])]
    unknown=[r for r in episodes if r['stock_id']==sid and r['start'] is None
             and (r['end'] is None or day<r['end'])]
    if len(known)>1 or (known and unknown):raise ValueError('Overlapping or uncertain market episodes')
    if known:return dict(status='identified',market=known[0]['market'],category=known[0]['category'])
    return dict(status='unknown' if unknown else 'outside_verified_intervals',market=None,category=None)


def build(output):
    output=Path(output)
    if output.exists():raise ValueError('Choose a new immutable identity audit output')
    refs={};current=[]
    for name,market in [('isin-twse.html','TWSE'),('isin-tpex.html','TPEx')]:
        path=CACHE/name;meta=read(path.with_name(name+'.source.json'))
        if sha(path)!=meta['sha256']:raise ValueError('Official snapshot hash changed')
        refs[str(path.relative_to(ROOT))]=sha(path)
        current.extend(parse_isin(path.read_bytes(),market))
    if len({r['stock_id'] for r in current})!=len(current):raise ValueError('Current dual-market conflict')
    ended,end_refs=load_membership_sources();refs.update(end_refs)
    listings=[]
    for name,parser in [('twse-newlisting.json',parse_twse_listings),('tpex-newlisting.json',parse_recent_tpex_listings)]:
        path=OLD/name
        if sha(path)!=read(path.with_suffix('.source.json'))['sha256']:raise ValueError('Listing source changed')
        refs[str(path.relative_to(ROOT))]=sha(path);listings.extend(parser(read(path)))
    episodes=[dict(**r,end=None,start_evidence='current_official_ISIN') for r in current]
    for r in ended:
        if any(c['stock_id']==r['stock_id'] and c['market']==r['market'] and c['start']<r['end'] for c in current):
            raise ValueError('Current and ended official market sources conflict')
        starts=[s for s in listings if s['stock_id']==r['stock_id'] and s['market']==r['market'] and s['start']<r['end']]
        distinct=sorted({s['start'] for s in starts})
        if len(distinct)>1:raise ValueError('Multiple historical listing starts need review')
        # No use of the first price date, current public-issue date or unknown date as IPO.
        start=distinct[0] if distinct else None
        category=('臺灣存託憑證(TDR)' if r['stock_id']=='9188' else
                  '創新板' if any('創新板' in s['note'] for s in starts) else 'unconfirmed')
        episodes.append(dict(**r,start=start,category=category,
            start_evidence='official_listing_record' if start else None))
    missing=[r for r in episodes if r['start'] is None]
    # Explicitly audit the original fixed signal set without presenting it as the whole market.
    manifest=read(ROOT/'.cache/cash-allocation-inputs/manifest.json')
    signal_ref=manifest['references']['signals'];signal_path=ROOT/signal_ref['path']
    if sha(signal_path)!=signal_ref['sha256']:raise ValueError('Frozen signal source changed')
    refs[signal_ref['path']]=sha(signal_path)
    entries=read(signal_path)['entries'];signals=[]
    for entry in entries:
        day=str(entry.get('signal_date',entry.get('date')))[:10]
        for sid in entry['members']:
            signals.append(dict(stock_id=sid,date=day,**resolve_on(episodes,sid,day)))
    # The current snapshot is retrospective evidence, not archived publication-time evidence.
    result=dict(schema='official_market_identity_audit_v1',current_rows=len(current),
        current_ordinary_stocks=sum(r['category']=='股票' for r in current),
        ended_episodes=len(ended),missing_historical_starts=len(missing),
        missing_start_rows=missing,episodes=episodes,signal_identity=signals,
        signal_rows=len(signals),unidentified_signals=sum(s['status']!='identified' for s in signals),
        complete_historical_universe=False,publication_time_archive_complete=False,
        database_mutations=0,live_qualified=False,source_sha256=refs)
    output.mkdir(parents=True);write(output/'report.json',result)
    return {k:v for k,v in result.items() if k not in ('episodes','missing_start_rows','source_sha256','signal_identity')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(build(a.output))
