#!/usr/bin/env python3
"""Bounded official-action repair in a new cache; never updates production factors."""
from datetime import date, datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from app.twse_client import TWSEError
from scripts.build_official_adj_factors import month_chunks, year_chunks, _to_parquet
from skills.official_adj_factors import (FETCH_SPECS, OfficialAdjClient, events_to_dataframe,
                                       validate_events_in_range, _tpex_tables_rows, RATIO_LO, RATIO_HI)

START, END = date(2022, 1, 1), date(2026, 9, 9)
SOURCE = ROOT / 'artifacts/adj_official/checkpoints'
OUTPUT = ROOT / '.cache/readiness-remediation-20260914/actions'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def strict_parse(kind, parser, payload, start, end):
    events = parser(payload)
    validate_events_in_range(events, start, end, kind)
    rows = _tpex_tables_rows(payload, kind) if kind.startswith('tpex_') else payload.get('data', []) or []
    if len(events) != len(rows):
        raise ValueError(f'{kind}: parsed {len(events)} of {len(rows)} rows; omitted rows need review')
    keys = [(e.stock_id, e.event_date, e.source) for e in events]
    if len(keys) != len(set(keys)):
        raise ValueError(kind+': duplicate action keys need review')
    return events


def run(prepare=False, source=SOURCE, output=OUTPUT, client=None):
    output=Path(output).resolve(); source=Path(source).resolve(); output.mkdir(parents=True,exist_ok=True)
    with file_lock(output/'run.lock',timeout=0):
        ck=output/'checkpoints'; ck.mkdir(exist_ok=True)
        identity={str(Path(__file__).relative_to(ROOT)):digest(__file__),
                  'skills/official_adj_factors.py':digest(ROOT/'skills/official_adj_factors.py')}
        manifest_path=output/'summary.json'
        previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if previous and previous['code_sha256']!=identity:
            raise ValueError('Audit code changed; use a new output directory')
        budget_path=output/'requests.json'
        requests=json.loads(budget_path.read_text()) if budget_path.exists() else []
        rows=[]; all_events=[]; calls=0
        if prepare and client is None:
            client=OfficialAdjClient(delay=2,timeout=20,max_retries=0)
            client.session.max_redirects=0
        for kind,parser,method in FETCH_SPECS:
            chunks=year_chunks if kind.endswith(('capital_reduction','par_value_change')) else month_chunks
            for start,end in chunks(START,END):
                name=f'{kind}_{start:%Y%m%d}_{end:%Y%m%d}.json'
                target=ck/name; old=Path(source)/name
                row=dict(kind=kind,start=str(start),end=str(end),status='missing',path=str(target.relative_to(ROOT)))
                try:
                    if target.exists():
                        if previous:
                            expected=next((r.get('sha256') for r in previous['chunks'] if r['path']==row['path']),None)
                            if expected and digest(target)!=expected:
                                raise ValueError('Cached source bytes changed')
                        payload=json.loads(target.read_text())
                    elif old.exists():
                        payload=json.loads(old.read_text())
                        strict_parse(kind,parser,payload,start,end)
                        target.write_bytes(old.read_bytes())
                        row['inherited_from']=str(old.relative_to(ROOT))
                    elif prepare and len(requests)<10 and name not in {r['name'] for r in requests}:
                        # Reserve durably before the only request. A failed or
                        # interrupted attempt is not silently repeated on resume.
                        requests.append(dict(name=name,requested_at=datetime.now(timezone.utc).isoformat()))
                        save(budget_path,requests); calls+=1
                        payload=getattr(client,method)(start,end)
                        save(target,payload)
                    else:
                        rows.append(row); continue
                    row['sha256']=digest(target)
                    events=strict_parse(kind,parser,payload,start,end)
                    all_events.extend(events)
                    row.update(status='parsed',events=len(events),missing_ratio=sum(e.ratio is None for e in events))
                except (ValueError,KeyError,TypeError,TWSEError) as exc:
                    row.update(status='blocked',reason=str(exc)[:350])
                rows.append(row)
        complete=all(r['status']=='parsed' for r in rows)
        # Partial inputs never publish an apparently complete factor table.
        eligible=[e for e in all_events if len(e.stock_id)==4 and e.stock_id.isdigit()]
        invalid_ratio=[dict(stock_id=e.stock_id,date=str(e.event_date),source=e.source)
                       for e in eligible if e.ratio is None or not RATIO_LO<=e.ratio<=RATIO_HI]
        if complete:
            _to_parquet(events_to_dataframe(all_events),output/'events.parquet')
        report=dict(start=str(START),end=str(END),observed_at=datetime.now(timezone.utc).isoformat(),
                    chunks=rows,parsed_chunks=sum(r['status']=='parsed' for r in rows),
                    required_chunks=len(rows),source_windows_complete=complete,events=len(all_events),
                    requests_this_run=calls,requests_lifetime=len(requests),finmind_requests=0,
                    database_mutations=0,code_sha256=identity,live_qualified=False,
                    excluded_non_four_digit_events=len(all_events)-len(eligible),
                    invalid_ratio_events=invalid_ratio,
                    note='Parsed dates and row counts only; does not prove provider completeness, action delivery, or production factor correctness.')
        save(manifest_path,report)
        return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--source',type=Path,default=SOURCE)
    args=parser.parse_args(); report=run(args.prepare,source=args.source,output=args.output)
    print(json.dumps({k:v for k,v in report.items() if k!='chunks'},ensure_ascii=False,indent=2))
    print(json.dumps([r for r in report['chunks'] if r['status']!='parsed'],ensure_ascii=False,indent=2))
    sys.exit(0 if report['source_windows_complete'] else 1)
