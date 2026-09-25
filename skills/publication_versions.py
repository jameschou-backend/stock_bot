"""Keep issuer dates separate from the first observation of exact document bytes."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

TAIPEI = ZoneInfo('Asia/Taipei')
HOST = 'www.passivecomponent.com'
INDEX_URL = 'https://' + HOST + '/about/news/'


def digest(raw):
    return sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'


def stamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('A timezone-aware timestamp is required')
    return parsed


def release_url(url):
    parsed = urlparse(url)
    if (parsed.scheme != 'https' or parsed.netloc != HOST or parsed.query or parsed.fragment
            or not re.fullmatch(r'/\d{4}/\d{2}/\d{2}/walsin-technology-global-consolidated-net-sales-for-[a-z0-9-]+/', parsed.path)):
        raise ValueError('Only observed official Walsin revenue release URLs are supported')
    return parsed


def index_links(raw):
    soup = BeautifulSoup(raw, 'html.parser')
    links = sorted({node['href'] for node in soup.select('article h2.entry-title a[href]')
                    if 'global consolidated net sales' in node.get_text(' ', strip=True).lower()})
    if not links or len(links) > 30:
        raise ValueError('Official index missing or exceeds the 30-release capture budget')
    for url in links:
        release_url(url)
    return links


def parse_release(raw, url):
    parsed = release_url(url)
    soup = BeautifulSoup(raw, 'html.parser')
    titles, contents = soup.select('h1.entry-title'), soup.select('div.post-content')
    if len(titles) != 1 or len(contents) != 1:
        raise ValueError('Official release title/body is ambiguous or missing')
    title, body = (nodes[0].get_text(' ', strip=True) for nodes in (titles, contents))
    if not body or 'global consolidated net sales for ' not in title.lower():
        raise ValueError('Not an official revenue release')
    dates = []
    # The hidden .updated span is a CMS modification time, not publication time.
    for node in soup.select('.fusion-meta-info span'):
        lineage = [node, *node.parents]
        hidden = any(
            bool(set(parent.get('class', [])) & {'updated', 'rich-snippet-hidden', 'vcard'})
            or parent.has_attr('hidden') or parent.get('aria-hidden') == 'true'
            or re.search(r'(?:display\s*:\s*none|visibility\s*:\s*hidden)', parent.get('style', ''), re.I)
            for parent in lineage)
        if hidden:
            continue
        value = node.get_text(' ', strip=True)
        if re.fullmatch(r'[A-Z][a-z]+ \d{1,2}(?:st|nd|rd|th)?, \d{4}', value):
            dates.append(datetime.strptime(re.sub(r'(\d)(st|nd|rd|th)', r'\1', value), '%B %d, %Y').date())
    if len(set(dates)) != 1:
        raise ValueError('Visible official publication date missing or conflicting')
    day = dates[0]
    if parsed.path.split('/')[1:4] != [f'{day.year:04}', f'{day.month:02}', f'{day.day:02}']:
        raise ValueError('Visible publication date differs from the official URL date')
    month_text = title.lower().split('global consolidated net sales for ', 1)[1].strip()
    period = None
    for pattern in ('%B %Y', '%b %Y'):
        try:
            period = datetime.strptime(month_text, pattern).strftime('%Y-%m')
            break
        except ValueError:
            pass
    if period is None:
        raise ValueError('Revenue period missing from the official title')
    if period >= day.strftime('%Y-%m'):
        raise ValueError('Revenue period must precede the visible publication month')
    semantic = dict(title=title, body=body, claimed_publication_date=day.isoformat(), revenue_period=period)
    return dict(stock_id='2492', url=url, title=title, revenue_period=period,
        claimed_publication_date=day.isoformat(), publication_precision='date',
        official_published_at=None, publication_timezone_verified=False,
        # No exact intraday timing is inferred from a date or CMS timestamp.
        publication_date_upper_bound=(datetime.combine(day, datetime.min.time(), TAIPEI)
                                      + timedelta(days=1)).isoformat(),
        content_sha256=digest(encoded(semantic)), raw_sha256=digest(raw),
        historical_revision_chain_complete=False)


def extend_versions(previous, observations):
    versions = [dict(row) for row in previous]
    for row in observations:
        observed = stamp(row['observed_at'])
        prior = [r for r in versions if r['url'] == row['url']]
        if prior and observed <= stamp(prior[-1]['observed_at']):
            raise ValueError('New document observations must follow prior observations')
        changed = not prior or prior[-1]['content_sha256'] != row['content_sha256']
        versions.append(dict(row, content_changed=changed,
            version_first_observed_at=row['observed_at'] if changed else prior[-1]['version_first_observed_at'],
            version_id=digest(encoded([row['url'], row['content_sha256'],
                row['observed_at'] if changed else prior[-1]['version_first_observed_at']])),
            previous_observation_id=prior[-1]['observation_id'] if prior else None,
            observation_id=digest(encoded([row['url'], row['raw_sha256'], row['observed_at']]))))
    return versions


def as_of(versions, cutoff):
    """Only return versions whose exact content was actually observed by cutoff."""
    when = stamp(cutoff)
    selected = {}
    for row in versions:
        observed = stamp(row['observed_at'])
        if observed <= when and stamp(row['publication_date_upper_bound']) <= when:
            if row['url'] not in selected or observed > stamp(selected[row['url']]['observed_at']):
                selected[row['url']] = row
    return [selected[url] for url in sorted(selected)]


def verify_archive(path, _visited=None):
    path = Path(path).resolve()
    visited = set() if _visited is None else set(_visited)
    if path in visited:
        raise ValueError('Publication archive ancestry contains a cycle')
    visited.add(path)
    raw = path.read_bytes()
    if digest(raw) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Publication archive hash mismatch')
    result = json.loads(raw)
    if (result.get('format') != 'issuer_publication_versions_v1' or result.get('historical_complete') is not False
            or result.get('live_qualified') is not False or result.get('finmind_requests') != 0):
        raise ValueError('Unsupported publication archive or historical completeness claim')
    for name, expected in result['files_sha256'].items():
        source = (path.parent / name).resolve()
        if not source.is_relative_to(path.parent) or digest(source.read_bytes()) != expected:
            raise ValueError('Publication source changed: ' + name)
    if result['parent'] is not None:
        parent = result['parent']
        if digest(Path(parent['path']).read_bytes()) != parent['sha256']:
            raise ValueError('Publication archive parent changed')
        old = verify_archive(parent['path'], visited)['versions']
    else:
        old = []
    observations = []
    index = result['index']
    def source_receipt(name):
        if name not in result['files_sha256'] or name+'.source.json' not in result['files_sha256']:
            raise ValueError('Consumed publication file lacks bound source receipt')
        receipt = json.loads((path.parent / (name+'.source.json')).read_bytes())
        if (receipt['path'] != name or receipt['sha256'] != result['files_sha256'][name]
                or receipt['http_status'] != 200):
            raise ValueError('Publication source receipt does not match its bytes')
        stamp(receipt['observed_at'])
        return receipt
    if source_receipt(index['path']) != index or index['url'] != INDEX_URL:
        raise ValueError('Publication index receipt differs from the archive')
    links = index_links((path.parent / index['path']).read_bytes())
    if set(links) != {row['url'] for row in result['observations']}:
        raise ValueError('Publication capture does not cover its declared index')
    if len(links) != len(result['observations']):
        raise ValueError('Duplicate publication observations in a capture')
    attempts = json.loads((path.parent / 'attempts.json').read_bytes())
    if ('attempts.json' not in result['files_sha256'] or result['network_requests'] != len(attempts)
            or [row['url'] for row in attempts] != [INDEX_URL, *links]
            or any(row.get('status') != 'received' or row.get('http_status') != 200 for row in attempts)):
        raise ValueError('Publication attempts differ from successfully captured source coverage')
    for row in result['observations']:
        receipt = source_receipt(row['path'])
        if (receipt['url'] != row['url'] or receipt['observed_at'] != row['observed_at']
                or receipt['sha256'] != row['raw_sha256']):
            raise ValueError('Publication observation differs from its source receipt')
        source = path.parent / row['path']
        parsed = parse_release(source.read_bytes(), row['url'])
        if any(row.get(k) != v for k, v in parsed.items()):
            raise ValueError('Publication parsed values differ from raw source')
        stamp(row['observed_at'])
        observations.append(row)
    if result['versions'] != extend_versions(old, observations):
        raise ValueError('Publication versions were rewritten or backdated')
    return result
