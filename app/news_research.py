"""Bounded news collection and offline research, isolated from production features."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func, select

from app.db import get_session
from app.finmind import fetch_dataset
from app.models import RawStockNews, Stock
from skills.news_radar import analyze, clean_title, title_key

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/news-research'
ROW_LIMIT = 200_000


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str))
    tmp.replace(path)


def company_names():
    with get_session() as session:
        rows = session.execute(select(Stock.stock_id, Stock.name).where(
            Stock.security_type == 'stock', Stock.market.in_(['TWSE', 'TPEX']))).all()
    return {sid: name for sid, name in rows if len(sid) == 4 and sid.isascii() and sid.isdigit() and name}


def normalize(rows, *, observed_at, day=None, legacy=False):
    result, invalid = [], 0
    for row in rows:
        stamp = row.get('news_datetime') if legacy else row.get('date')
        title, sid = row.get('title'), str(row.get('stock_id', ''))
        parsed = pd.to_datetime(stamp, errors='coerce')
        if (pd.isna(parsed) or not isinstance(title, str) or not title.strip()
                or not (sid.isascii() and sid.isdigit() and len(sid) == 4)
                or (day is not None and parsed.date() != day)):
            invalid += 1
            continue
        recorded = row.get('created_at') if legacy else observed_at
        result.append({'stock_id': sid, 'provider_datetime': parsed.isoformat(),
                       'first_recorded_at': str(recorded or observed_at),
                       'recorded_time_basis': 'legacy_db_timezone_unverified' if legacy else 'UTC',
                       'title': title, 'source': str(row.get('source') or ''), 'link': str(row.get('link') or '')})
    return result, invalid


def store_day(day, frame, now):
    """Preserve first observation across refreshes, including empty-success provenance."""
    path = CACHE / 'days' / f'{day.isoformat()}.json'
    old = json.loads(path.read_text()) if path.exists() else {'rows': []}
    rows, invalid = normalize(frame.to_dict('records'), observed_at=now.isoformat(), day=day)
    def key(r): return (r['stock_id'], r['provider_datetime'], title_key(clean_title(r['title'], r['source'])))
    prior = {key(r): r for r in old['rows']}
    for row in rows:
        k = key(row)
        if k in prior:
            row['first_recorded_at'] = prior[k]['first_recorded_at']
        prior[k] = row
    payload = {'schema': 1, 'day': day.isoformat(), 'fetched_at': now.isoformat(),
               'network_cache_hit': bool(frame.attrs.get('cache_hit', False)),
               'latest_response_rows': len(frame), 'invalid_or_out_of_day_rows': invalid,
               'coverage': 'provider_response_only_not_proof_of_all_market_news', 'rows': list(prior.values())}
    atomic_json(path, payload)
    return payload


def collect_recent(end, days, config, *, now=None):
    if not 1 <= days <= 14:
        raise ValueError('新聞更新每次限 1～14 天')
    fixed_clock = now is not None
    now = now or datetime.now(timezone.utc)
    if end > now.astimezone(ZoneInfo('Asia/Taipei')).date():
        raise ValueError('不能抓取未來新聞')
    stats = {'network_requests': 0, 'day_cache_hits': 0, 'gateway_cache_hits': 0, 'days': []}
    for offset in range(days-1, -1, -1):
        day = end - timedelta(days=offset)
        path = CACHE / 'days' / f'{day.isoformat()}.json'
        cached = json.loads(path.read_text()) if path.exists() else None
        ttl = 3600 if (now.date()-day).days <= 2 else 86400
        age = (now-datetime.fromisoformat(cached['fetched_at'])).total_seconds() if cached else None
        if cached and 0 <= age < ttl:
            stats['day_cache_hits'] += 1
        else:
            # No custom retry loop: FinMindQuotaError stops this job and keeps completed days.
            frame = fetch_dataset('TaiwanStockNews', day, token=config.finmind_token,
                                  requests_per_hour=config.finmind_requests_per_hour,
                                  max_retries=0, timeout=25)
            if frame.attrs.get('cache_hit'): stats['gateway_cache_hits'] += 1
            else: stats['network_requests'] += 1
            cached = store_day(day, frame, now if fixed_clock else datetime.now(timezone.utc))
        stats['days'].append({'date': day.isoformat(), 'rows': len(cached['rows']),
                              'fetched_at': cached['fetched_at']})
        print(f"news day {day}: {len(cached['rows'])} rows", flush=True)
    return stats


def load_window(start, end, stock_id=None):
    if end < start or (end-start).days > 365:
        raise ValueError('新聞判讀視窗最多 366 天')
    with get_session() as session:
        latest = session.execute(select(func.max(RawStockNews.news_datetime))).scalar_one_or_none()
        q = select(RawStockNews.stock_id, RawStockNews.news_datetime, RawStockNews.source,
                   RawStockNews.title, RawStockNews.link, RawStockNews.created_at).where(
            RawStockNews.news_datetime >= datetime.combine(start, datetime.min.time()),
            RawStockNews.news_datetime < datetime.combine(end+timedelta(days=1), datetime.min.time()))
        if stock_id: q = q.where(RawStockNews.stock_id == stock_id)
        records = session.execute(q.order_by(RawStockNews.news_datetime).limit(ROW_LIMIT+1)).mappings().all()
    if len(records) > ROW_LIMIT:
        raise ValueError('新聞超過 200,000 列，請縮小日期或指定股票；不截斷後假裝完整')
    rows, invalid = normalize(records, observed_at=datetime.now(timezone.utc).isoformat(), legacy=True)
    local_days = []
    for i in range((end-start).days+1):
        path = CACHE / 'days' / f'{start+timedelta(days=i)}.json'
        if path.exists():
            payload = json.loads(path.read_text())
            local_days.append({'date': payload['day'], 'fetched_at': payload['fetched_at']})
            rows.extend(r for r in payload['rows'] if stock_id is None or r['stock_id'] == stock_id)
    if len(rows) > ROW_LIMIT:
        raise ValueError('合併新聞超過 200,000 列，請縮小研究視窗')
    return rows, {'legacy_latest': str(latest) if latest else None, 'legacy_rows': len(records),
                  'legacy_invalid_rows': invalid, 'local_days': local_days,
                  'coverage_note': '未取得完整新聞覆蓋證明；無資料的日期不等於沒有新聞。'}


def price_context(stories, stock_id, cutoff):
    """Only past closes; never uses future returns to choose or rank a headline."""
    import duckdb
    from scripts.research_flow import verify_inputs, INPUT_DIR
    from skills.flow_research import listing_mask
    manifest = verify_inputs()
    with duckdb.connect() as con:
        data = con.execute('SELECT stock_id,trading_date,adj_close FROM read_parquet(?) '
                           'WHERE stock_id IN (?,?) AND trading_date<=?',
                           [str(INPUT_DIR/'quotes.parquet'), stock_id, '0050', cutoff]).df()
    data.trading_date = pd.to_datetime(data.trading_date)
    close = data.pivot(index='trading_date', columns='stock_id', values='adj_close').sort_index()
    if stock_id not in close or '0050' not in close:
        return {'available': False, 'note': '固定快照缺少個股或 0050；不以其他來源補值'}
    companies = pd.read_parquet(INPUT_DIR/'companies.parquet')
    close = close.where(listing_mask(close.index, close.columns, companies))
    latest = close.index[-1]
    for story in stories:
        date_ = pd.Timestamp(story['source_date'])
        pos = int(close.index.searchsorted(date_, side='left')) - 1
        story['price_context'] = None
        if pos < 20 or date_ > latest + pd.Timedelta(days=5):
            continue
        prices = close.iloc[[pos-20, pos]][[stock_id, '0050']]
        if prices.isna().any().any() or (prices <= 0).any().any(): continue
        returns = prices.iloc[1] / prices.iloc[0] - 1
        story['price_context'] = {'as_of': str(close.index[pos].date()),
            'stock_return_20d': float(returns[stock_id]), 'benchmark_return_20d': float(returns['0050']),
            'excess_20d': float(returns[stock_id]-returns['0050']),
            'already_outperforming_10pp': bool(returns[stock_id]-returns['0050'] >= .1)}
    return {'available': True, 'latest_price': str(latest.date()),
            'snapshot_sha256': manifest['files_sha256']['quotes.parquet'],
            'note': '供應商日期前一交易日的 20 日價格背景；不含事件之後報酬，不代表因果或買入時點。'}


def run(mode, end, days, *, stock_id=None, fetch=False, config=None):
    started = time.perf_counter()
    if mode not in ('scan', 'review') or not 1 <= days <= 366:
        raise ValueError('研究模式或天數錯誤')
    if fetch and (mode != 'scan' or config is None):
        raise ValueError('只有近期掃描可更新 FinMind；歷史判讀使用本機資料')
    names = company_names()
    if mode == 'review' and stock_id not in names:
        raise ValueError('請輸入主檔可識別的四碼上市櫃普通股')
    collection = collect_recent(end, days, config) if fetch else {'network_requests': 0, 'offline': True}
    # Date-only historical cutoff excludes the selected day itself (timezone unknown).
    data_end = end-timedelta(days=1) if mode == 'review' else end
    start = data_end-timedelta(days=days-1)
    rows, source = load_window(start, data_end, stock_id if mode == 'review' else None)
    report = analyze(rows, names)
    report.update(schema=1, experiment='news_radar_v1', mode=mode, research_only=True, live_qualified=False,
                  rules_sha256=hashlib.sha256((ROOT/'skills/news_radar.py').read_bytes()).hexdigest(),
                  analyzed_at=datetime.now(timezone.utc).isoformat(), start=str(start), end=str(data_end),
                  cutoff=str(end), stock_id=stock_id if mode == 'review' else None,
                  names={sid: names[sid] for story in report['stories'] for sid in story['stock_ids']},
                  source=source, collection=collection,
                  evidence_note='標題解析，未核對全文；提及題材或股票不等於已證實受惠。',
                  time_note='保留供應商原始時間，時區未確認。舊新聞多為事後回補；本解析器現在才建立，不能當成當年收到的訊號。')
    if mode == 'review':
        report['stock_name'] = names[stock_id]
        report['price_source'] = price_context(report['stories'], stock_id, data_end)
        first = {}
        for story in report['stories']:
            for event in story['events']:
                first.setdefault(event, story['id'])
        report['first_event_in_window'] = first
    report['input_sha256'] = hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
    report['elapsed_seconds'] = round(time.perf_counter()-started, 3)
    report['summary'] = {k: report[k] for k in ('unique_articles', 'duplicates_collapsed', 'elapsed_seconds')}
    atomic_json(CACHE/f'{mode}.json', report)
    return report


def overview(mode='scan'):
    if mode not in ('scan', 'review'): raise ValueError('Unknown news mode')
    path = CACHE/f'{mode}.json'
    if not path.exists(): return {'available': False, 'note': '尚未分析，請先啟動下方新聞工作'}
    try:
        report = json.loads(path.read_text())
        if (report['schema'] != 1 or report['experiment'] != 'news_radar_v1' or report['mode'] != mode
                or report['research_only'] is not True or report['live_qualified'] is not False):
            raise ValueError('Invalid news report')
        if report['rules_sha256'] != hashlib.sha256((ROOT/'skills/news_radar.py').read_bytes()).hexdigest():
            raise ValueError('Headline rules changed; explicit re-analysis required')
        for key in ('stories', 'themes', 'source', 'summary', 'collection'): report[key]
        return {**report, 'available': True}
    except (ValueError, KeyError, TypeError, OSError):
        return {'available': False, 'note': '新聞結果不完整，請重新分析；不沿用舊結論'}
