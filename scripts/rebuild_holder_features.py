#!/usr/bin/env python3
"""Replace only two holder columns, preserving every other feature and backup."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select
from app.db import get_session
from app.models import ValidatedHoldingDist
from skills.feature_store import FeatureStore


def run():
    fs = FeatureStore()
    paths = sorted(fs.store_dir.glob('features_*.parquet'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    keys = pd.concat([pd.read_parquet(p, columns=['stock_id', 'trading_date']) for p in paths], ignore_index=True)
    keys['trading_date'] = pd.to_datetime(keys.trading_date)
    keys['_ordinal'] = range(len(keys))
    with get_session() as session:
        holders = pd.read_sql(select(ValidatedHoldingDist.stock_id,
            ValidatedHoldingDist.available_date.label('trading_date'), ValidatedHoldingDist.large_holder_pct), session.get_bind())
    holders['trading_date'] = pd.to_datetime(holders.trading_date)
    holders['large_holder_pct'] = pd.to_numeric(holders.large_holder_pct)
    aligned = pd.merge_asof(keys.sort_values('trading_date'), holders.sort_values('trading_date'),
        on='trading_date', by='stock_id', direction='backward', tolerance=pd.Timedelta(days=21))
    aligned = aligned.sort_values(['stock_id', 'trading_date'])
    aligned['large_holder_chg_4w'] = aligned.large_holder_pct-aligned.groupby('stock_id').large_holder_pct.shift(20)
    aligned = aligned.sort_values('_ordinal')
    backup = ROOT/'.cache/holder-feature-backups'
    backup.mkdir(exist_ok=True)
    offset, report = 0, []
    for path in paths:
        table = pq.read_table(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]:
            raise ValueError('Feature store changed during repair')
        saved = backup/(path.stem+'-'+hashes[str(path)]+'.parquet')
        if not saved.exists(): shutil.copy2(path, saved)
        values = aligned.iloc[offset:offset+len(table)]
        for col in ('large_holder_pct', 'large_holder_chg_4w'):
            if col not in table.column_names: raise ValueError('Expected holder feature missing')
            idx = table.column_names.index(col)
            table = table.set_column(idx, col, pa.array(values[col], type=table.field(col).type, from_pandas=True))
        temp = path.with_suffix('.holder.tmp')
        pq.write_table(table, temp)
        temp.replace(path)
        offset += len(table)
        report.append(dict(path=str(path), rows=len(table), observed=int(values.large_holder_pct.notna().sum()),
            before=hashes[str(path)], after=hashlib.sha256(path.read_bytes()).hexdigest(), backup=str(saved)))
    (ROOT/'.cache/holder-feature-rebuild.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(rows=offset, years=len(paths), observed=int(aligned.large_holder_pct.notna().sum()))))

if __name__ == '__main__': run()
