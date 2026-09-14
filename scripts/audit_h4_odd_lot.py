#!/usr/bin/env python3
"""Parse official H4 byte layouts without inventing units or session completeness."""
from datetime import datetime
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import write,sha


def parse_h4(blob, version):
    if version not in ('legacy190','20260401_201'):
        raise ValueError('H4 layout version must be explicit')
    width, price_end, volume_end, date_start = ((190,28,36,180) if version=='legacy190' else (201,29,37,191))
    # Record separators are permitted; arbitrary whitespace stripping corrupts fixed widths.
    records=blob.splitlines() if b'\n' in blob else [blob[i:i+width] for i in range(0,len(blob),width)]
    if not records: raise ValueError('Empty H4 file')
    result=[]
    for line_no,raw in enumerate(records,1):
        if len(raw)!=width: raise ValueError(f'H4 record {line_no}: expected {width} bytes')
        row=raw.decode('ascii',errors='strict')
        sid=row[:6].strip();stamp=row[6:18];day=row[date_start:date_start+8]
        if not (sid.isalnum() and len(sid)<=6 and stamp.isdigit() and day.isdigit()):
            raise ValueError('Malformed H4 identifier or timestamp')
        when=datetime.strptime(day+stamp,'%Y%m%d%H%M%S%f')
        if version=='20260401_201' and when.date().isoformat()<'2026-04-01':
            raise ValueError('H4 new layout predates its documented effective date')
        if version=='legacy190' and when.date().isoformat()>='2026-04-01':
            raise ValueError('H4 legacy layout crosses the documented version boundary')
        remark,match=row[18],row[20]
        if remark not in (' ','T','S') or match not in (' ','Y','S'):
            raise ValueError('Unknown H4 trial/match flag')
        p,v=row[22:price_end],row[price_end:volume_end]
        if not p.isdigit() or not v.isdigit(): raise ValueError('Malformed H4 raw price/volume')
        actual=remark==' ' and match=='Y'
        if actual and (int(p)<=0 or int(v)<=0): raise ValueError('Non-positive actual H4 match')
        result.append(dict(stock_id=sid,time=when.isoformat(timespec='microseconds'),
            remark=remark,match_flag=match,actual_match=actual,trial=remark=='T',
            raw_price=int(p),raw_volume=int(v)))
    return result


def audit(path,version):
    rows=parse_h4(Path(path).read_bytes(),version)
    return dict(source_sha256=sha(path),layout=version,record_count=len(rows),
        actual_match_records=sum(r['actual_match'] for r in rows),
        trial_records=sum(r['trial'] for r in rows),rows=rows,
        source_unit_confirmation_required=True,session_complete=False,usable_for_fills=False,
        reason='Format parsing alone proves neither price/share units nor complete session coverage',
        live_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--version',choices=['legacy190','20260401_201'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.input,a.version);write(a.output,r)
    print({k:v for k,v in r.items() if k!='rows'})
