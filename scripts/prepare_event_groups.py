#!/usr/bin/env python
"""Freeze local news and validate cached official actions; bounded gap collection."""
from __future__ import annotations
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text

from app.file_lock import file_lock
from scripts.research_flow import atomic_json, save_frame, verify_inputs as verify_prices
from scripts.research_rules import digest

CACHE = ROOT/'.cache/event-group-research'
START, END = date(2021, 7, 1), date(2025, 11, 30)
# Broad local prefilter only; final event rules and explicit identity checks follow.
WORDS = ('記憶體|DRAM|NAND|HBM|DDR|被動|MLCC|電容|電阻|低軌|衛星|Starlink|SpaceX|散熱|液冷|水冷|'
         '重電|變壓器|電網|HVDC|封裝|CoWoS|玻璃基板|FOPLP|伺服器|GB200|GB300|Blackwell|Rubin|'
         '矽光子|光通訊|CPO|機器人|機械手臂|無人機|儲能|BBU|備援|報價|調漲|漲價|售價|訂單|'
         '獲單|接單|認證|量產|投產|出貨|拉貨|轉盈|虧損|毛利')


def verified_frame(path):
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    if digest(path) != meta['sha256']:
        raise ValueError('Snapshot changed: '+str(path))
    return pd.read_parquet(path)


def save_snapshot(path, frame, **meta):
    save_frame(path, frame)
    atomic_json(path.with_suffix('.meta.json'), {'sha256': digest(path), 'rows': len(frame),
                'saved_at': datetime.now(timezone.utc).isoformat(), **meta})


def prepare():
    from app.db import get_session
    from app.config import load_config
    from app.finmind import fetch_dataset
    from scripts.build_official_adj_factors import _parse_validated
    from skills.official_adj_factors import FETCH_SPECS, events_to_dataframe

    started = time.perf_counter()
    verify_prices()
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE/'inputs.json').exists():
        return verify()
    path = CACHE/'news-day-counts.parquet'
    if not path.exists():
        with get_session() as session:
            counts = pd.read_sql(text('SELECT DATE(news_datetime) AS day, COUNT(*) AS rows_count '
                'FROM raw_stock_news WHERE news_datetime>=:a AND news_datetime<:b GROUP BY DATE(news_datetime)'),
                session.get_bind(), params={'a': START, 'b': END+timedelta(days=1)})
        save_snapshot(path, counts)
    counts = verified_frame(path)
    observed = set(pd.to_datetime(counts.day).dt.date)
    gaps = [d.date() for d in pd.date_range(START, END) if d.date() not in observed]
    if len(gaps) > 20:
        raise ValueError(f'{len(gaps)} missing news dates exceed the preregistered 20-request budget')
    print(f'本機新聞 {int(counts.rows_count.sum()):,} 列；整日缺口 {len(gaps)} 天', flush=True)
    for month in pd.date_range(START, END, freq='MS'):
        path = CACHE/f'news-{month:%Y%m}.parquet'
        if not path.exists():
            with get_session() as session:
                data = pd.read_sql(text('SELECT stock_id,news_datetime,created_at,title,source,link FROM raw_stock_news '
                    'WHERE news_datetime>=:a AND news_datetime<:b LIMIT 200001'), session.get_bind(),
                    params={'a': month.date(), 'b': (month+pd.offsets.MonthBegin(1)).date()})
            if len(data) > 200000:
                raise ValueError('Monthly news exceeds 200,000 rows; split the input explicitly')
            data = data[data.title.str.contains(WORDS, case=False, regex=True, na=False)].copy()
            save_snapshot(path, data, local_prefilter=WORDS)
        data = verified_frame(path)
        print(f'{month:%Y-%m}: {len(data):,} 相關列', flush=True)
    config = load_config()
    for day in gaps:
        path = CACHE/f'gap-{day}.parquet'
        if not path.exists():
            frame = fetch_dataset('TaiwanStockNews', day, data_id=None, token=config.finmind_token,
                requests_per_hour=config.finmind_requests_per_hour, max_retries=0, timeout=30)
            if not frame.empty and set(pd.to_datetime(frame['date']).dt.date) != {day}:
                raise ValueError('News gap response contains out-of-date rows')
            save_snapshot(path, frame, provider_attrs=dict(frame.attrs))
        verified_frame(path)
        print(f'已保存缺口 {day}', flush=True)
    # Read-only re-parse of existing checkpoints. Never mutate/replace user's artifacts.
    events, provenance = [], {}
    window_start, window_end = date(2021, 1, 1), date(2026, 6, 23)
    expected_days = set(pd.date_range(window_start, window_end).date)
    for kind, parser, _ in FETCH_SPECS:
        covered = set()
        for path in sorted((ROOT/'artifacts/adj_official/checkpoints').glob(kind+'_*.json')):
            match = re.fullmatch(re.escape(kind)+r'_(\d{8})_(\d{8})\.json', path.name)
            if not match:
                continue
            a, b = (datetime.strptime(s, '%Y%m%d').date() for s in match.groups())
            if b < window_start or a > window_end:
                continue
            before = digest(path)
            events.extend(_parse_validated(parser, json.loads(path.read_text()), a, b, path.name))
            if before != digest(path):
                raise ValueError('Official checkpoint changed during reading')
            provenance[str(path.relative_to(ROOT))] = before
            covered.update(pd.date_range(max(a, window_start), min(b, window_end)).date)
        if covered != expected_days:
            raise ValueError('Official action checkpoint coverage incomplete: '+kind)
    ev = events_to_dataframe(events)
    ev = ev[pd.to_datetime(ev.event_date).dt.date.between(window_start, window_end)].copy()
    save_snapshot(CACHE/'official-events.parquet', ev, checkpoint_sha256=provenance)
    files = sorted([*CACHE.glob('*.parquet'), *CACHE.glob('*.meta.json')])
    empty_gap_dates = [str(d) for d in gaps if verified_frame(CACHE/f'gap-{d}.parquet').empty]
    result = {'schema': 1, 'prepared_at': datetime.now(timezone.utc).isoformat(),
              'news_start': str(START), 'news_end': str(END), 'db_news_rows': int(counts.rows_count.sum()),
              'gap_dates': list(map(str, gaps)), 'empty_gap_dates': empty_gap_dates,
              'gap_request_budget': len(gaps), 'official_event_rows': len(ev),
              'official_checkpoint_count': len(provenance),
              'files_sha256': {p.name: digest(p) for p in files},
              'elapsed_seconds': round(time.perf_counter()-started, 3)}
    atomic_json(CACHE/'inputs.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'files_sha256'}, ensure_ascii=False), flush=True)
    return result


def verify():
    d = json.loads((CACHE/'inputs.json').read_text())
    for name, expected in d['files_sha256'].items():
        if Path(name).name != name or digest(CACHE/name) != expected:
            raise ValueError('Event research input changed: '+name)
    return d


if __name__ == '__main__':
    from app.finmind import FinMindQuotaError
    try:
        with file_lock(ROOT/'.cache/research-or-update.lock', timeout=0):
            prepare()
    except FinMindQuotaError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(75)
