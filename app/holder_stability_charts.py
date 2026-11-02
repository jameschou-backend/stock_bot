"""Bounded display projections; never reads live prices, holdings or signals."""
from datetime import date, timedelta
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / 'artifacts/forward_simulation/holder_stability_charts_20261102.json.gz'
EXPECTED_SHA256 = 'e9b230f77b09076a0ae0b660550c752980f63169e93b1d1531ea35338feacca0'


class NotWatchlistMember(ValueError):
    pass


@lru_cache(maxsize=1)
def _load(path, fingerprint, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Chart source integrity mismatch')
    result = json.loads(gzip.decompress(raw))
    if result['schema'] != 'holder_stability_charts_v1' or result['end'] != '2026-10-30':
        raise ValueError('Chart identity mismatch')
    return result


def align_holdings(candles, observations, lag):
    """Backward as-of, retaining invalid snapshots and gaps instead of skipping."""
    if lag not in (8, 15):
        raise ValueError('Unsupported holder lag')
    ordered = sorted(observations, key=lambda row: row['observed_date'])
    if len({row['observed_date'] for row in ordered}) != len(ordered):
        raise ValueError('Duplicate holder week')
    result, index, current = [], 0, None
    for candle in candles:
        day = date.fromisoformat(candle['date'])
        while index < len(ordered):
            row = ordered[index]
            available = date.fromisoformat(row['observed_date']) + timedelta(days=lag)
            if available > day:
                break
            current = dict(row, available_date=available.isoformat())
            index += 1
        point = dict(date=candle['date'], ratio=None, delta4=None, observed_date=None,
                     available_date=None, status='unavailable')
        if current:
            fresh = (day - date.fromisoformat(current['available_date'])).days <= 14
            known = fresh and current['valid']
            point.update(observed_date=current['observed_date'], available_date=current['available_date'],
                         ratio=current['ratio'] if known else None, delta4=current['delta4'] if known else None,
                         status='known' if known else 'stale' if not fresh else 'invalid_snapshot')
        result.append(point)
    return result


def chart_view(stock_id, sessions=120, holder_lag=8):
    from app.holder_stability_web import study_view, EXPECTED_SHA256 as STUDY_SHA
    if sessions not in (60, 120, 240) or holder_lag not in (8, 15):
        raise ValueError('Unsupported chart options')
    candidates = study_view()['candidates']
    if stock_id not in {row['stock_id'] for row in candidates}:
        raise NotWatchlistMember('Not a watchlist member')
    stat = CHARTS.stat()
    data = _load(CHARTS, (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns), EXPECTED_SHA256)
    if data['study_sha256'] != STUDY_SHA or set(data['stocks']) != {row['stock_id'] for row in candidates}:
        raise ValueError('Chart/watchlist source mismatch')
    try:
        stock = data['stocks'][stock_id]
        candles = stock['candles'][-sessions:]
        if len(candles) != sessions or candles[-1]['date'] != data['end']:
            raise ValueError('Incomplete chart calendar')
        holdings = align_holdings(candles, stock['observations'], holder_lag)
    except KeyError as exc:
        raise ValueError('Incomplete chart data') from exc
    return dict(stock_id=stock_id, name=stock['name'], end=data['end'], start=candles[0]['date'],
                sessions=sessions, holder_lag=holder_lag, price_basis='adjusted', volume_unit='shares',
                candles=candles, holdings=holdings, latest_holding=holdings[-1],
                missing_price_sessions=sum(not row['valid'] for row in candles),
                unknown_holding_sessions=sum(row['ratio'] is None for row in holdings),
                historical_publication_verified=False, live_qualified=False)
