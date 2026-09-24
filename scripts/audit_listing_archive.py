#!/usr/bin/env python3
"""Verify reviewed primary yearbook tables without inventing announcement dates."""
from datetime import date
from pathlib import Path
import argparse
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha

CACHE=ROOT/'.cache/listing-sources-20260924'
IDENTITY=ROOT/'.cache/readiness-completion-20260914/identity-v2/report.json'
# PDF page numbers are one-based, including covers; all are reviewed main-board
# new-listing tables, not the later Emerging Stock Market tables.
TABLES={92:[40,41],94:[38],98:[28],99:[25],100:[27,28]}
DOCS=(101,103,104,106,108,109)


def parse_table(text,year):
    if not all(s in text for s in ('Company','Code','IPO Price','Capital Stock')):
        raise ValueError('Not a reviewed new-listing table')
    if 'Emerging' in text or '興櫃' in text:
        raise ValueError('Emerging-stock registration is not main-board listing')
    rows={}
    pattern=r'(?<!\d)(20\d{6}|(?:20\d{2}|\d{3})[/.]\d{1,2}[/.]\d{1,2})\s+(\d{4})(?!\d)'
    for m in re.finditer(pattern,text):
        raw,sid=m.groups()
        if raw.isdigit(): y,mo,dy=int(raw[:4]),int(raw[4:6]),int(raw[6:])
        else:y,mo,dy=map(int,re.split(r'[/.]',raw))
        if y<1911:y+=1911
        if y!=year:
            raise ValueError('Listing year differs from reviewed yearbook')
        day=date(y,mo,dy).isoformat()
        if sid in rows:
            raise ValueError('Duplicate code in reviewed new-listing table')
        rows[sid]=day
    if not rows:
        raise ValueError('No listing dates extracted')
    return rows


def verified_text(stem,suffix,refs):
    path=CACHE/(stem+suffix);meta=path.with_suffix('.source.json');text_path=path.with_suffix('.txt')
    record=read(meta)
    if record.get('status')!=200 or sha(path)!=record['sha256']:
        raise ValueError('Primary archive file hash/status differs')
    command=(['pdftotext','-layout',str(path),'-'] if suffix=='.pdf'
             else ['textutil','-convert','txt','-stdout',str(path)])
    try:generated=subprocess.run(command,check=True,capture_output=True).stdout.decode('utf-8')
    except FileNotFoundError as exc:
        raise RuntimeError('Install Poppler (pdftotext); DOC conversion requires macOS textutil') from exc
    if generated!=text_path.read_text():
        raise ValueError('Extracted table does not match primary file')
    for f in (path,meta,text_path):refs[str(f.relative_to(ROOT))]=sha(f)
    return generated,record['url']


def build(output):
    if output.exists():raise ValueError('Choose a new immutable report path')
    original=read(IDENTITY);refs={str(IDENTITY.relative_to(ROOT)):sha(IDENTITY),
        str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    missing={(r['stock_id'],r['market']):r for r in original['missing_start_rows']}
    resolved={}
    def accept(rows,stem,location,url):
        for sid,day in rows.items():
            key=(sid,'TPEx')
            if key not in missing:continue
            if key in resolved:raise ValueError('Repeated listing evidence needs explicit review')
            if day>=missing[key]['end']:raise ValueError('Listing is after termination')
            resolved[key]=dict(stock_id=sid,market='TPEx',start=day,end=missing[key]['end'],
                category='股票',evidence_kind='primary_yearbook_listing_record',
                source=stem,location=location,url=url,announcement_available_at=None,
                continuous_eligibility_proven=False)
    for year,pages in TABLES.items():
        stem=f'factbook{year}';text,url=verified_text(stem,'.pdf',refs);split=text.split('\f')
        for p in pages:accept(parse_table(split[p-1],year+1911),stem,dict(pdf_page=p),url)
    for year in DOCS:
        stem=f'new{year}';text,url=verified_text(stem,'.doc',refs)
        if str(year) not in text[:100] or '新上櫃' not in text[:100]:
            raise ValueError('Unexpected yearbook DOC heading')
        accept(parse_table(text,year+1911),stem,dict(table='main_board_new_listings'),url)
        # The official catalog binds the selected download to the correct section.
        catalog=CACHE/f'catalog{year}.html';meta=catalog.with_suffix('.source.json')
        if read(meta)['status']!=200 or sha(catalog)!=read(meta)['sha256']:
            raise ValueError('Catalog hash/status differs')
        if url.split('www.tpex.org.tw',1)[-1] not in catalog.read_text():
            raise ValueError('Downloaded file not in official catalog')
        for f in (catalog,meta):refs[str(f.relative_to(ROOT))]=sha(f)
    if resolved.get(('1258','TPEx'),{}).get('start')!='2011-12-12':
        raise ValueError('Prior original 1258 announcement control differs')
    result=dict(schema='listing_archive_audit_v1',original_unknown_starts=len(missing),
        verified_listing_dates=len(resolved),newly_resolved_since_boundary=len(resolved)-1,
        remaining_unknown_starts=len(missing)-len(resolved),resolved_rows=list(resolved.values()),
        unresolved_rows=[v for k,v in missing.items() if k not in resolved],
        complete_historical_universe=False,publication_revision_archive_complete=False,
        database_mutations=0,live_qualified=False,source_sha256=refs)
    write(output,result)
    return {k:v for k,v in result.items() if k not in ('source_sha256','resolved_rows','unresolved_rows')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(build(p.parse_args().output))
