#!/usr/bin/env python3
"""Read a normalized broker CSV and emit a separate, immutable comparison file."""
from pathlib import Path
import argparse,csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import capacity_forward as policy, capacity_execution_review as review, forward_journal as j
from app.file_lock import file_lock

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=policy.ROOT);parser.add_argument('--role',choices=['strategy','control','benchmark'],default='strategy')
    args=parser.parse_args()
    with file_lock(args.root/'.run.lock',timeout=0):
        rows=policy.verify(args.root/(args.role+'.sqlite3'),args.role)
        with args.csv.open(encoding='utf-8-sig',newline='') as source:records=list(csv.DictReader(source))
        result=review.compare(rows,records)
        result.update(source_csv_sha256=__import__('hashlib').sha256(args.csv.read_bytes()).hexdigest(),account_head=rows[-1]['hash'],observed_at=j.now().isoformat())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as target:json.dump(result,target,ensure_ascii=False,indent=2)
        print('比對回報',result['report_count'],'筆；未改寫模擬帳本')
