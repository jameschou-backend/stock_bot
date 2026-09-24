"""Bounded, archived FinMind financial observations with no guessed release date."""
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import re

from sqlalchemy import select
from app.file_lock import file_lock
from app.finmind import FinMindError, fetch_dataset
from app.job_utils import finish_job, start_job
from app.models import QuarterlyFundamentalSnapshot, QuarterlyIngestState, Stock
from skills.quarterly_validation import calculate

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('TaiwanStockBalanceSheet','TaiwanStockFinancialStatements','TaiwanStockCashFlowsStatement')


def persist_snapshots(session, records):
    """Do not replace old revisions or move unchanged observations forward."""
    inserted = 0
    for row in records:
        latest = session.execute(select(QuarterlyFundamentalSnapshot).where(
            QuarterlyFundamentalSnapshot.stock_id == row['stock_id'],
            QuarterlyFundamentalSnapshot.report_date == row['report_date']
        ).order_by(QuarterlyFundamentalSnapshot.observed_at.desc()).limit(1)).scalar_one_or_none()
        if latest and latest.source_sha256 == row['source_sha256']:
            continue
        if latest and row['observed_at'] <= latest.observed_at:
            raise FinMindError('Financial revision must have a later observation timestamp')
        session.add(QuarterlyFundamentalSnapshot(**row))
        session.flush()
        inserted += 1
    return inserted


def run(config, db_session, *, stock_ids=None, max_requests=180, **kwargs):
    with file_lock(ROOT/'.cache/quarterly-observations.lock', timeout=0):
        return _run(config, db_session, stock_ids=stock_ids, max_requests=max_requests)


def _run(config, session, *, stock_ids, max_requests):
    if not isinstance(max_requests,int) or max_requests < 3 or max_requests > 180:
        raise ValueError('Quarterly batch max_requests must be between 3 and 180')
    job_id = start_job(session, 'ingest_quarterly_fundamental', commit=True)
    logs = dict(rows=0, requests_reserved=0, timing_basis='first_observed_next_day', legacy_table_used=False)
    try:
        allowed = set(session.execute(select(Stock.stock_id).where(
            Stock.is_listed.is_(True),Stock.security_type=='stock')).scalars())
        ids = sorted(allowed if stock_ids is None else set(stock_ids))
        if any(not re.fullmatch(r'\d{4}',s) or s not in allowed for s in ids):
            raise ValueError('Financial batch contains a nonordinary/unlisted/invalid stock ID')
        seen = dict(session.execute(select(QuarterlyIngestState.stock_id,QuarterlyIngestState.checked_at)).all())
        today = datetime.now(ZoneInfo(config.tz)).date()
        # Fetch progress cannot move financial first-observed timestamps forward.
        ids.sort(key=lambda sid:(seen.get(sid,datetime.min),sid))
        logs['requested_stocks'] = len(ids)
        if stock_ids is not None and len(ids)*3 > max_requests:
            raise ValueError('Explicit financial batch exceeds request budget; reduce stock_ids')
        selected = ids[:max_requests//3]
        logs['stocks'] = len(selected); logs['deferred_stocks'] = len(ids)-len(selected)
        start = date(today.year-3,1,1)  # TTM income, prior-year equity and YTD cash flow warmup
        for sid in selected:
            frames = []
            for dataset in DATASETS:
                logs['requests_reserved'] += 1
                frames.append(fetch_dataset(dataset,start,today,data_id=sid,token=config.finmind_token,
                    requests_per_hour=config.finmind_requests_per_hour,max_retries=0,timeout=30))
            if any(f.empty for f in frames):
                raise FinMindError(f'Incomplete three-statement source: {sid}')
            if any(set(f.stock_id.astype(str)) != {sid} for f in frames):
                raise FinMindError(f'Wrong stock in financial source: {sid}')
            observed = datetime.now(timezone.utc)
            folder = ROOT/'.cache/quarterly-observations'/sid/observed.strftime('%Y%m%dT%H%M%S%fZ')
            folder.mkdir(parents=True)
            manifest = dict(stock_id=sid,start=start.isoformat(),end=today.isoformat(),
                observed_at=observed.isoformat(),amount_unit='TWD',income_basis='single_quarter',
                cashflow_basis='year_to_date',files_sha256={})
            for dataset,frame in zip(DATASETS,frames):
                path = folder/(dataset+'.parquet');frame.to_parquet(path,index=False)
                manifest['files_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            path = folder/'manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True))
            records = calculate(*frames,observed_at=observed)
            for row in records:
                row['source_manifest'] = str(path.relative_to(ROOT))
            logs['rows'] += persist_snapshots(session,records)
            session.merge(QuarterlyIngestState(stock_id=sid,checked_at=observed.replace(tzinfo=None)))
            session.commit()
        logs['missing_share_denominator'] = True
        finish_job(session,job_id,'success',logs=logs)
        return logs
    except Exception as exc:
        session.rollback()
        finish_job(session,job_id,'failed',error_text=str(exc),logs=logs)
        raise
