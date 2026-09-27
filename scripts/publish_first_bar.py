#!/usr/bin/env python3
"""Publish independently reproduced first-bar evidence without changing status."""
import argparse
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from scripts.audit_current_causality_20260925 import verify_hashes
from app.first_bar_ui import PUBLICATION, load


def publish(first,second):
    folders=[Path(first).resolve(),Path(second).resolve()]
    if folders[0]==folders[1] or any(not p.is_relative_to(ROOT/'.cache') for p in folders):
        raise ValueError('Two different immutable cache outputs required')
    reports=[];manifests=[]
    for f in folders:
        m=read(f/'manifest.json');r=read(f/'report.json')
        if m['schema']!='first_bar_manifest_v1' or m['source_sha256']!=r['source_sha256']:
            raise ValueError('Unbound result or source manifest')
        verify_hashes({ROOT/p:h for p,h in m['source_sha256'].items()})
        for n,h in m['files_sha256'].items():
            if Path(n).name!=n or sha(f/n)!=h:raise ValueError('Changed study output')
        reports.append(r);manifests.append(m)
    files=lambda m:{k:v for k,v in m['files_sha256'].items() if k!='report.json'}
    def normalized(r):
        r=copy.deepcopy(r);r.pop('elapsed_seconds')
        for c in r['cases'].values():c['result']['path']=Path(c['result']['path']).name
        return r
    if files(manifests[0])!=files(manifests[1]) or normalized(reports[0])!=normalized(reports[1]):
        raise ValueError('Independent first-bar results differ')
    path=ROOT/PUBLICATION
    write(path,dict(schema='first_bar_publication_v1',report=dict(path=str((folders[0]/'report.json').relative_to(ROOT)),sha256=sha(folders[0]/'report.json')),
        reproducibility=dict(passed=True,files_sha256=files(manifests[0]),
            runs=[dict(path=str((f/'manifest.json').relative_to(ROOT)),sha256=sha(f/'manifest.json')) for f in folders]),
        live_qualified=False,adopted=False))
    path.with_suffix('.sha256').write_text(sha(path)+'\n');load(ROOT)
    print('Published two identical first-bar studies')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('first');p.add_argument('second')
    a=p.parse_args();publish(a.first,a.second)
