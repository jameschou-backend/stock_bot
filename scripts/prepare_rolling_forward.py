#!/usr/bin/env python3
"""Incremental snapshots for simulation; retains every prior frozen input directory."""
from datetime import datetime
from pathlib import Path
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import forward_journal as j,forward_portfolio as p
from app.file_lock import file_lock

NAMES=('close-official.parquet','close-quality.parquet','raw-close.parquet','raw-volume.parquet','companies.parquet')


def prepare(root):
    import pandas as pd
    from scripts import prepare_forward_signals as original
    root=Path(root);today=str(datetime.now(p.TZ).date());output=root/'signals'
    with file_lock(root/'.rolling.lock',timeout=0):
        baseline=original.BASE
        old_base,old_output=original.BASE,original.OUTPUT
        try:
            previous=sorted(d for d in output.glob('????-??-??') if d.name<today and (d/'manifest.json').exists())
            if previous:
                source=previous[-1];meta=json.loads((source/'manifest.json').read_text())
                for name,sha in meta['sha256'].items():
                    if original.sha(source/name)!=sha:raise ValueError('前一日封存輸入已變更：'+name)
                for name,sha in meta['code_sha256'].items():
                    if original.sha(ROOT/name)!=sha:raise ValueError('訊號程式已變更，需另開驗證版本')
                base_meta=json.loads((baseline/'manifest.json').read_text())
                if original.sha(baseline/'companies.parquet')!=base_meta['files_sha256']['companies.parquet']:raise ValueError('原始股票名冊已變更')
                bridge=root/'rolling-base'/source.name;bridge.mkdir(parents=True,exist_ok=True)
                manifest=bridge/'manifest.json'
                if not manifest.exists():
                    for name in NAMES[:-1]:
                        frame=pd.read_parquet(source/name)
                        frame=frame.loc[pd.to_datetime(frame['date'])<=pd.Timestamp(source.name)]
                        frame.to_parquet(bridge/name,index=False)
                    shutil.copyfile(baseline/'companies.parquet',bridge/'companies.parquet')
                    original.write(manifest,dict(source_manifest_sha256=original.sha(source/'manifest.json'),
                        files_sha256={name:original.sha(bridge/name) for name in NAMES}))
                bridge_meta=json.loads(manifest.read_text())
                if bridge_meta['source_manifest_sha256']!=original.sha(source/'manifest.json'):raise ValueError('增量來源鏈已變更')
                original.BASE=bridge
            original.OUTPUT=output
            return original.prepare()
        finally:original.BASE,original.OUTPUT=old_base,old_output

if __name__=='__main__':
    from app.forward_simulation import ROOT as DEFAULT
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    print(prepare(parser.parse_args().root))
