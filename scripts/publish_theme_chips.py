#!/usr/bin/env python3
"""Publish only identical complete offline chip studies."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import sha,write
from scripts.audit_current_causality_20260925 import verify_hashes
from app.theme_chips_ui import load,PUBLICATION


def publish(first,second):
    folders=[Path(first).resolve(),Path(second).resolve()]
    if folders[0]==folders[1] or any(not p.is_relative_to(ROOT/'.cache') for p in folders):
        raise ValueError('Two distinct local cache runs required')
    reports=[];manifests=[]
    for folder in folders:
        manifest=json.loads((folder/'manifest.json').read_text())
        if sha(folder/'report.json')!=manifest['report_sha256']:raise ValueError('Changed report')
        report=json.loads((folder/'report.json').read_text())
        if report['source_sha256']!=manifest['source_sha256'] or report['artifacts']!=manifest['files']:
            raise ValueError('Report detached from manifest')
        verify_hashes({ROOT/p:v for p,v in report['source_sha256'].items()})
        for artifact in report['artifacts'].values():
            path=ROOT/artifact['path']
            if path.parent!=folder or sha(path)!=artifact['sha256']:raise ValueError('Changed study table')
        reports.append(report);manifests.append(dict(path=str((folder/'manifest.json').relative_to(ROOT)),sha256=sha(folder/'manifest.json')))
    def comparable(report):
        value=dict(report);value.pop('elapsed_seconds')
        value['artifacts']={k:{f:v for f,v in d.items() if f!='path'} for k,d in value['artifacts'].items()}
        return value
    if comparable(reports[0])!=comparable(reports[1]):raise ValueError('Study results differ')
    report=reports[0];path=ROOT/PUBLICATION
    value=dict(schema='theme_chips_publication_v1',report=dict(path=str((folders[0]/'report.json').relative_to(ROOT)),
        sha256=sha(folders[0]/'report.json')),reproducibility=dict(passed=True,runs=manifests,
        csv_sha256={k:d['sha256'] for k,d in report['artifacts'].items()},
        elapsed_seconds=[r['elapsed_seconds'] for r in reports]),live_qualified=False,adopted=False)
    write(path,value);path.with_suffix('.sha256').write_text(sha(path)+'\n')
    load(ROOT)
    print(json.dumps(dict(passed=True,compared_csvs=len(report['artifacts']),elapsed_seconds=value['reproducibility']['elapsed_seconds'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('first');p.add_argument('second')
    args=p.parse_args();publish(args.first,args.second)
