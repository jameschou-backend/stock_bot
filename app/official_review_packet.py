"""Official disclosure snapshots for human review, never trading authorization."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import requests
from app import forward_journal as j, forward_portfolio as p
from app.file_lock import file_lock

PATH = j.ROOT / '.cache/forward-validation/official-review.sqlite3'
URL = 'https://openapi.twse.com.tw/v1/opendata/t187ap04_L'
TTL = 1800
LIMIT = 2_000_000


def fetch():
    # Requests retains TLS verification and the configured CA bundle. No retries,
    # redirects, credentials, or per-stock requests.
    with requests.get(URL, timeout=20, stream=True, allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('官方來源回應不是 200')
        chunks = []; size = 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > LIMIT:
                raise ValueError('官方公告超過單次讀取上限')
            chunks.append(chunk)
        return b''.join(chunks)


def roc_date(value):
    value = str(value).strip().replace('/', '').replace('-', '')
    if len(value) != 7 or not value.isdigit():
        raise ValueError('官方公告民國日期格式不符')
    return datetime(int(value[:3]) + 1911, int(value[3:5]), int(value[5:])).date()


def parse(raw, retrieved):
    if len(raw) > LIMIT:
        raise ValueError('官方公告超過單次讀取上限')
    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise ValueError('官方公告沒有可驗證出表日期的資料')
    out = []
    today = retrieved.astimezone(p.TZ).date()
    for original in data:
        row = {k.strip(): v for k, v in original.items()}
        if roc_date(row['出表日期']) != today:
            raise ValueError('官方公告出表日不是今日')
        day = roc_date(row['發言日期'])
        raw_time = str(row['發言時間']).strip()
        if ':' not in raw_time:
            raw_time = raw_time.zfill(6)
            fmt = '%H%M%S'
        else:
            fmt = '%H:%M:%S'
        published = datetime.combine(day, datetime.strptime(raw_time, fmt).time(), p.TZ)
        if published > retrieved:
            raise ValueError('官方公告發言時間在取得時間之後')
        sid = str(row['公司代號']).strip()
        if len(sid) != 4 or not sid.isdigit():
            raise ValueError('官方公告公司代號格式不符')
        out.append(dict(stock_id=sid, company=str(row['公司名稱']), title=str(row['主旨']),
                        explanation=str(row['說明']), published_at=published.isoformat()))
    return out


def latest(path=PATH, at=None):
    if not Path(path).exists():
        return None
    with j.connection(path) as con:
        rows = j.read_events(con)
    return next((r for r in reversed(rows) if r['kind'] == 'official_snapshot'
                 and (at is None or j.timestamp(r['recorded_at']) <= at)), None)


def capture(path=PATH, clock=j.now, fetcher=fetch):
    path = Path(path)
    with file_lock(path.with_suffix('.lock'), timeout=0):
        started = clock(); prior = latest(path, started)
        if prior:
            age = (started - j.timestamp(prior['recorded_at'])).total_seconds()
            same_day = j.timestamp(prior['recorded_at']).astimezone(p.TZ).date() == started.astimezone(p.TZ).date()
            if same_day and 0 <= age < TTL:
                return dict(requests=0, reused=True, status=prior['body']['status'], hash=prior['hash'])
        body = dict(url=URL, status='error', rows=[], raw_sha256=None, error=None)
        try:
            raw = fetcher(); retrieved = clock()
            if retrieved < started:
                raise ValueError('取得公告期間時鐘倒退')
            body.update(raw_sha256=hashlib.sha256(raw).hexdigest(), raw_text=raw.decode('utf-8-sig'))
            body['rows'] = parse(raw, retrieved)
            body['status'] = 'ok'
        except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
            retrieved = clock()
            # Keep provider bodies / exception strings out of user error messages.
            body['error'] = type(exc).__name__
        body['retrieved_at'] = retrieved.isoformat()
        with j.connection(path) as con:
            result = j.append(con, 'official:'+j.digest(body), 'official_snapshot', body, lambda: retrieved)
        return dict(requests=1, reused=False, status=body['status'], hash=result['hash'])


def inspect(stocks, path=PATH, clock=j.now):
    """stocks maps the actual held code to tse/otc; ETFs are separate sources."""
    at = clock(); snapshot = latest(path, at)
    ready = False; note = '尚未取得官方公告快照'
    if snapshot:
        body = snapshot['body']; retrieved = j.timestamp(body['retrieved_at'])
        age = (at - retrieved).total_seconds()
        ready = body['status'] == 'ok' and 0 <= age <= TTL and retrieved.astimezone(p.TZ).date() == at.astimezone(p.TZ).date()
        note = '快照可供核對；未命中不代表沒有事件' if ready else '快照過期或取得失敗；不可沿用為今日核對結果'
    return dict(observed_at=at.isoformat(), snapshot_at=snapshot['body']['retrieved_at'] if snapshot else None,
                snapshot_hash=snapshot['hash'] if snapshot else None, ready=ready, note=note,
                coverage_complete=False, actions_reviewed=False, finmind_requests=0,
                stocks=[dict(stock_id=sid, scope='上市公司重大訊息' if market=='tse' and sid!='0050'
                             else '未涵蓋：需另查基金公告' if sid=='0050' else '未涵蓋：需另查上櫃公告',
                             disclosures=[r for r in snapshot['body']['rows'] if r['stock_id']==sid]
                             if ready and market=='tse' and sid!='0050' else []) for sid, market in stocks.items()])
