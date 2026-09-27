#!/usr/bin/env python3
"""Publish two byte-identical sets of accounts, retaining blocked cases."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from scripts.audit_current_causality_20260925 import verify_hashes
from app.theme_catalyst_ui import PUBLICATION,load


def publish(first,second):
    folders=[Path(first).resolve(),Path(second).resolve()]
    if folders[0]==folders[1] or any(not f.is_relative_to(ROOT/'.cache') for f in folders):
        raise ValueError('Two distinct project cache runs required')
    manifests=[];reports=[]
    for folder in folders:
        manifest=read(folder/'manifest.json')
        verify_hashes({ROOT/p:h for p,h in manifest['source_sha256'].items()})
        for name,digest in manifest['files_sha256'].items():
            if Path(name).name!=name or sha(folder/name)!=digest:raise ValueError('Changed output: '+name)
        report=read(folder/'report.json')
        if report['source_sha256']!=manifest['source_sha256']:raise ValueError('Detached report sources')
        manifests.append(manifest);reports.append(report)
    comparable=lambda m:{k:v for k,v in m['files_sha256'].items() if k!='report.json'}
    if (comparable(manifests[0])!=comparable(manifests[1]) or
            manifests[0]['source_sha256']!=manifests[1]['source_sha256']):raise ValueError('Independent replay differs')
    def normalized(report):
        import copy
        r=copy.deepcopy(report);r.pop('elapsed_seconds')
        for c in r['cases'].values():c['result']['path']=Path(c['result']['path']).name
        return r
    if normalized(reports[0])!=normalized(reports[1]):raise ValueError('Summary results differ')
    path=ROOT/PUBLICATION
    write(path,dict(schema='theme_catalyst_publication_v1',report=dict(path=str((folders[0]/'report.json').relative_to(ROOT)),
        sha256=sha(folders[0]/'report.json')),reproducibility=dict(passed=True,
        files_sha256=comparable(manifests[0]),runs=[dict(path=str((f/'manifest.json').relative_to(ROOT)),sha256=sha(f/'manifest.json')) for f in folders]),
        live_qualified=False,adopted=False))
    path.with_suffix('.sha256').write_text(sha(path)+'\n')
    load(ROOT)
    print('Verified and published',len(reports[0]['cases']),'cases')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('first');p.add_argument('second')
    a=p.parse_args();publish(a.first,a.second)
