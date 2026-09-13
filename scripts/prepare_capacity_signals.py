#!/usr/bin/env python3
"""Regenerate only the new day's signals after explicit chronology quarantine."""
from pathlib import Path
from datetime import datetime
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from app import forward_journal as j, forward_portfolio as p
from app.file_lock import file_lock
from scripts import prepare_forward_signals as original
from scripts.prepare_rolling_forward import prepare as extend
from scripts.prepare_million_signals import build_signals

AUDIT=ROOT/'artifacts/forward_simulation/price_chronology_20260913.json'


def prepare(root):
    root=Path(root);today=str(datetime.now(p.TZ).date())
    with file_lock(root/'.capacity-signals.lock',timeout=0):
        source=Path(extend(root)).parent
        parent=json.loads((source/'manifest.json').read_text())
        for name,sha in parent['sha256'].items():
            if original.sha(source/name)!=sha: raise ValueError('前向來源變更：'+name)
        for name,sha in parent['code_sha256'].items():
            if original.sha(ROOT/name)!=sha: raise ValueError('前向程式變更：'+name)
        target=root/'capacity-signals'/today;target.mkdir(parents=True,exist_ok=True)
        context=dict(parent_sha256=original.sha(source/'manifest.json'),audit_sha256=original.sha(AUDIT),
            code_sha256={**parent['code_sha256'],'scripts/prepare_capacity_signals.py':original.sha(Path(__file__))})
        if (target/'manifest.json').exists():
            manifest=json.loads((target/'manifest.json').read_text())
            if any(manifest[k]!=v for k,v in context.items()):raise ValueError('已封存的新版本來源改變')
            for name,sha in manifest['sha256'].items():
                if original.sha(target/name)!=sha:raise ValueError('已封存的新版本檔案改變')
            return target/'signals.json'
        names=('close-official.parquet','close-quality.parquet','raw-close.parquet','raw-volume.parquet')
        frames={name:pd.read_parquet(source/name).set_index('date') for name in names}
        for frame in frames.values():frame.index=pd.to_datetime(frame.index)
        applied=[]
        for row in json.loads(AUDIT.read_text())['quarantine']:
            sid,day=row['stock_id'],pd.Timestamp(row['date'])
            if sid not in frames[names[0]] or day not in frames[names[0]].index: continue
            for frame in frames.values():frame.at[day,sid]=float('nan')
            applied.append(dict(stock_id=sid,date=row['date']))
        base_meta=json.loads((original.BASE/'manifest.json').read_text())
        company_path=original.BASE/'companies.parquet'
        if original.sha(company_path)!=base_meta['files_sha256']['companies.parquet']:raise ValueError('名冊來源變更')
        companies=pd.read_parquet(company_path)
        result=build_signals(*[frames[name] for name in names],companies,
                             start=today[:8]+'01',signal_end=today)
        result.update(prepared_at=j.now().isoformat(),next_session=parent['next_session'],
            source_kind='original_rule_forward_extension',chronology_quarantine=applied,
            historical_membership_verified=False)
        for name,frame in frames.items():frame.rename_axis('date').reset_index().to_parquet(target/name,index=False)
        original.write(target/'signals.json',result)
        original.write(target/'manifest.json',dict(**context,next_session=parent['next_session'],live_qualified=False,
            sha256={name:original.sha(target/name) for name in (*names,'signals.json')}))
        return target/'signals.json'

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    print(prepare(parser.parse_args().root))
