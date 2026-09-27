#!/usr/bin/env python3
"""Publish two independently identical institutional/MA descriptions."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from scripts.audit_current_causality_20260925 import verify_hashes
from app.launch_flows_ui import PUBLICATION,load


def publish(first,second):
    folders=[Path(first).resolve(),Path(second).resolve()]
    if folders[0]==folders[1] or any(not f.is_relative_to(ROOT/'.cache') for f in folders):
        raise ValueError('Two distinct local output folders required')
    reports=[];runs=[]
    for f in folders:
        r=read(f/'report.json');m=read(f/'manifest.json')
        if m['report_sha256']!=sha(f/'report.json') or m['files']!=r['artifacts'] or m['source_sha256']!=r['source_sha256']:
            raise ValueError('Detached report or source')
        verify_hashes({ROOT/p:h for p,h in r['source_sha256'].items()})
        for a in r['artifacts'].values():
            p=ROOT/a['path']
            if p.parent!=f or sha(p)!=a['sha256']:raise ValueError('Changed evidence table')
        reports.append(r);runs.append(dict(path=str((f/'manifest.json').relative_to(ROOT)),sha256=sha(f/'manifest.json')))
    def comparable(r):
        v=dict(r);v.pop('elapsed_seconds');v['artifacts']={k:{f:x for f,x in a.items() if f!='path'} for k,a in r['artifacts'].items()}
        return v
    if comparable(reports[0])!=comparable(reports[1]):raise ValueError('Independent institutional studies differ')
    p=ROOT/PUBLICATION
    write(p,dict(schema='launch_flows_publication_v1',report=dict(path=str((folders[0]/'report.json').relative_to(ROOT)),sha256=sha(folders[0]/'report.json')),
        reproducibility=dict(passed=True,runs=runs,csv_sha256={k:a['sha256'] for k,a in reports[0]['artifacts'].items()},
            elapsed_seconds=[r['elapsed_seconds'] for r in reports]),live_qualified=False,adopted=False))
    p.with_suffix('.sha256').write_text(sha(p)+'\n');load(ROOT)
    print('Published two identical institutional/MA studies')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('first');p.add_argument('second')
    a=p.parse_args();publish(a.first,a.second)
