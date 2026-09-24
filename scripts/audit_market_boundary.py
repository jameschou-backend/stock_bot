#!/usr/bin/env python3
"""Dated market presence is evidence for one day, never an inferred IPO date."""
from datetime import date
from pathlib import Path
import argparse
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bs4 import BeautifulSoup
from scripts.research_exit_scenarios import read,write,sha

CACHE=ROOT/'.cache/completion-gaps-20260924'
OLD=ROOT/'.cache/readiness-completion-20260914'


def parse_presence(payload,market,day):
    date.fromisoformat(day)
    if market not in ('TWSE','TPEx'):
        raise ValueError('Unknown market')
    if payload.get('date')!=day.replace('-','') or payload.get('stat','').lower()!='ok':
        raise ValueError('Wrong day or invalid market response')
    key='證券代號' if market=='TWSE' else '代號'
    tables=[t for t in payload['tables'] if key in t.get('fields',[]) and
        ('每日收盤行情' in t.get('title','') if market=='TWSE' else t.get('title')=='上櫃股票行情')]
    if len(tables)!=1:
        raise ValueError('Missing/ambiguous ordinary market table')
    table=tables[0];fields=table['fields'];rows=table['data']
    if market=='TPEx' and table.get('totalCount')!=len(rows):
        raise ValueError('Incomplete TPEx table')
    result={}
    for row in rows:
        if len(row)!=len(fields):
            raise ValueError('Market row width differs')
        r=dict(zip(fields,row));sid=r[key].strip()
        if not re.fullmatch(r'[0-9]{4}',sid):
            continue
        if sid in result:
            raise ValueError('Duplicate four-digit market identifier')
        result[sid]=dict(stock_id=sid,name=r['證券名稱' if market=='TWSE' else '名稱'],
            market=market,observed_session=day,listing_date=None,
            category='unconfirmed',evidence_kind='official_daily_presence_only')
    if not result:
        raise ValueError('No four-digit market records')
    return result


def build(output):
    if output.exists():
        raise ValueError('Choose a new immutable report path')
    refs={};identity_path=OLD/'identity-v2/report.json'
    identity=read(identity_path);refs[str(identity_path.relative_to(ROOT))]=sha(identity_path)
    present={}
    for market,path,meta in [('TWSE',OLD/'twse-20220103.json',OLD/'twse-20220103.json.source.json'),
                             ('TPEx',CACHE/'tpex-20220103.raw',CACHE/'tpex-20220103.source.json')]:
        if sha(path)!=read(meta)['sha256']:
            raise ValueError('Official source hash differs')
        for f in (path,meta):refs[str(f.relative_to(ROOT))]=sha(f)
        present[market]=parse_presence(read(path),market,'2022-01-03')
    # Source text is reviewed for this specific historical exception. Neither
    # the 2024 restatement nor first quote is backdated into an original event.
    snippets={
        '4712-original-halt': ['110年8月11日','4712','110年8月13日','停止在證券商營業處所買賣'],
        '4712-halt-history': ['113年8月15日','4712','110年8月13日','111年2月14日','113年9月25日'],
        '1258-original-listing': ['1258','普通股','100年12月12日'],
    }
    for name,need in snippets.items():
        path=CACHE/(name+'.html');meta=CACHE/(name+'.source.json')
        m=read(meta)
        if m.get('status')!=200 or sha(path)!=m['sha256']:
            raise ValueError('Announcement fetch/hash invalid')
        body=BeautifulSoup(path.read_bytes(),'html.parser').get_text('',strip=True)
        if not all(s in body for s in need):
            raise ValueError('Reviewed announcement differs: '+name)
        for f in (path,meta):refs[str(f.relative_to(ROOT))]=sha(f)
    rows=[]
    for missing in identity['missing_start_rows']:
        sid=missing['stock_id'];record=present[missing['market']].get(sid)
        rows.append(dict(stock_id=sid,market=missing['market'],end=missing['end'],
            original_missing_start=True,verified_present_on_boundary=record is not None,
            boundary_record=record,
            absence_explained_by_halt=sid=='4712' and missing['market']=='TPEx' and record is None,
            verified_ipo_date='2011-12-12' if sid=='1258' else None,
            verified_ipo_source='1258-original-listing' if sid=='1258' else None,
            safe_to_infer_continuous_eligibility=False))
    result=dict(schema='market_boundary_audit_v1',session='2022-01-03',rows=rows,
        original_missing_starts=len(rows),verified_boundary_presence=sum(r['verified_present_on_boundary'] for r in rows),
        explained_absences=sum(r['absence_explained_by_halt'] for r in rows),
        newly_verified_original_listings=sum(r['verified_ipo_date'] is not None for r in rows),
        missing_original_listing_dates=sum(r['verified_ipo_date'] is None for r in rows),
        boundary_audit_complete=all(r['verified_present_on_boundary'] or r['absence_explained_by_halt'] for r in rows),
        continuous_historical_universe_complete=False,publication_revision_archive_complete=False,
        database_mutations=0,live_qualified=False,source_sha256=refs)
    write(output,result)
    return {k:v for k,v in result.items() if k not in ('rows','source_sha256')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(build(p.parse_args().output))
