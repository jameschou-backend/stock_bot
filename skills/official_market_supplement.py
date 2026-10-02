"""Hash-bound primary market supplements, without changing sealed study inputs.

Receipts establish provenance, not a trading guarantee. In particular, an
unclassified daily volume must never pass an ordinary-session capacity check.
"""
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

from skills.market_input_validation import (compare_quote as original_compare_quote,
    excluded, numeric, ordinary_capacity, parse_market_day, require, resolve_episode)


SCHEMA = 'official_market_supplement_v1'
# This immutable 2026-09-14 receipt did not record HTTP status. It is not
# retroactively assigned HTTP 200; only its exact archived bytes are admitted.
LEGACY_RECEIPT_SHA256 = frozenset({
    'e6acb4e79c4f1af2df2381edfd59da050a3c63fde9eb47845e6bc01c86e81a9e',
})
ENDPOINTS = {
    ('www.twse.com.tw', '/rwd/zh/afterTrading/MI_INDEX'): ('TWSE', 'twse_mi_index'),
    ('www.tpex.org.tw', '/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php'):
        ('TPEX', 'tpex_no1430'),
    ('www.tpex.org.tw', '/www/zh-tw/afterTrading/dailyQuotes'): ('TPEX', 'tpex_daily_quotes'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bound_path(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Expected repository-relative source path')
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), 'Missing or escaped supplement source')
    return path


def parse_tpex_daily_quotes(payload, stamp):
    """Full modern TPEx quote table; its volume scope remains unclassified."""
    day = date.fromisoformat(stamp)
    roc = f'{day.year - 1911}/{day.month:02d}/{day.day:02d}'
    require(payload.get('date') == stamp.replace('-', '') and payload.get('stat') == 'ok',
            'Official date/status mismatch')
    tables = payload.get('tables')
    require(isinstance(tables, list) and len(tables) == 2
            and all(isinstance(t, dict) for t in tables), 'Incomplete modern TPEx tables')
    require([t.get('title') for t in tables] == ['上櫃股票行情', '管理股票'], 'Incomplete modern TPEx market scope')
    require(tables[0].get('date') == roc, 'Official table date mismatch')
    require(numeric(tables[0].get('listedCompanies'), integral=True) > 0, 'Listed-company count missing')
    required = {'代號', '名稱', '收盤', '開盤', '最高', '最低', '成交股數'}
    result = {}
    all_ids = set()
    for table in tables:
        if 'date' in table:
            require(table['date'] == roc, 'Official table date mismatch')
        headers, rows = table.get('fields'), table.get('data')
        require(isinstance(headers, list) and all(isinstance(h, str) for h in headers), 'Invalid modern TPEx headers')
        headers = [h.strip() for h in headers]
        require(len(headers) == len(set(headers)) and required.issubset(headers), 'Missing or duplicate headers')
        require(isinstance(rows, list) and numeric(table.get('totalCount'), integral=True) == len(rows),
                'Incomplete official table')
        for values in rows:
            require(isinstance(values, list) and len(values) == len(headers), 'Official row width mismatch')
            row = dict(zip(headers, values))
            sid = str(row['代號']).strip()
            require(bool(sid) and sid not in all_ids, 'Duplicate or blank official stock')
            all_ids.add(sid)
            if not re.fullmatch(r'[0-9]{4}', sid):
                continue  # Four-digit strategy namespace; full raw table stays hash-bound.
            item = dict(date=stamp, stock_id=sid, name=row['名稱'], market='TPEX',
                        table_category=table['title'], volume_scope='unclassified_daily',
                        volume=numeric(row['成交股數'], integral=True))
            for key, field in [('open', '開盤'), ('high', '最高'), ('low', '最低'), ('close', '收盤')]:
                value = numeric(row[field], missing=True)
                item[key] = value if value and value > 0 else None
            if all(item[k] is not None for k in ('open', 'high', 'low', 'close')):
                require(item['low'] <= min(item['open'], item['close'])
                        <= max(item['open'], item['close']) <= item['high'], 'Impossible official OHLC')
            result[sid] = item
    require(result, 'Empty modern TPEx market table')
    return result


def compare_quote(local, official):
    scope = official.get('volume_scope')
    require(scope in ('ordinary_session', 'all_daily_sessions', 'unclassified_daily'), 'Unknown volume scope')
    result = original_compare_quote(local, official)
    if scope == 'unclassified_daily':
        result['total_volume'] = 'unverified_scope'
        result['ordinary_volume'] = 'unverified'
    return result


def recovery_sources(receipt_path, receipt, raw_path, root, market, stamp):
    """Verify reviewed recovery lineage even for a hand-built supplement manifest."""
    # This import is local to keep the ordinary archived-receipt path independent
    # of acquisition. Inspection never dispatches transport or writes evidence.
    from skills.official_daily_acquisition import (OfficialDailyAcquisition,
        TRANSPORT_RECOVERY_KIND, request_item)

    if (receipt.get('request_kind') != TRANSPORT_RECOVERY_KIND
            and not any(k in receipt for k in ('base_receipt_path', 'base_attempt_path',
                                               'recovery_proof_path', 'recovery_reason'))
            and receipt_path.parent.parent.name != 'recoveries'
            and raw_path.parent.parent.name != 'recoveries'):
        return {}
    require(receipt.get('request_kind') == TRANSPORT_RECOVERY_KIND,
            'Recovery receipt request kind differs')
    item = request_item(market, stamp)
    folder = receipt_path.parent
    require(receipt_path.name == 'receipt.json' and folder.name == item['identity']
            and folder.parent.name == 'recoveries', 'Recovery receipt location differs')
    try:
        client = OfficialDailyAcquisition(root, folder.parent.parent, session=object())
        linked = client.inspect_transport_recovery(item)
        require(linked['receipt_path'] == str(receipt_path.relative_to(root)),
                'Recovery receipt location differs')
        proof = json.loads(bound_path(root, linked['recovery_proof_path']).read_text())
        sources = dict(client.plan['source_sha256'])
        sources[str(client.plan_path.relative_to(root))] = client.plan_hash
        for prefix in ('base_receipt', 'base_attempt', 'recovery_proof'):
            sources[linked[prefix+'_path']] = linked[prefix+'_sha256']
        sources[str((folder/'attempt.json').relative_to(root))] = sha(folder/'attempt.json')
        sources[proof['raw_path']] = proof['raw_sha256']
        sources[linked['authorization']['path']] = linked['authorization']['sha256']
        for name, expected in sources.items():
            require(sha(bound_path(root, name)) == expected, 'Changed recovery lineage source')
        return sources
    except (OSError, ValueError, KeyError, TypeError) as exc:
        require(False, 'Invalid recovery receipt lineage: '+str(exc))


def validate_entry(entry, root):
    """Derive scope from a verified receipt and raw response, never entry labels."""
    root = Path(root).resolve()
    stamp, market = entry.get('date'), entry.get('market')
    require(isinstance(stamp, str) and date.fromisoformat(stamp).isoformat() == stamp, 'Invalid supplement date')
    require(market in ('TWSE', 'TPEX'), 'Unsupported supplement market')
    raw = bound_path(root, entry.get('raw_path'))
    receipt_path = bound_path(root, entry.get('receipt_path'))
    require(sha(raw) == entry.get('raw_sha256'), 'Changed supplement raw bytes')
    require(sha(receipt_path) == entry.get('receipt_sha256'), 'Changed supplement receipt')
    receipt = json.loads(receipt_path.read_text())
    recovery_refs = recovery_sources(receipt_path, receipt, raw, root, market, stamp)
    if receipt.get('schema') == 'official_daily_receipt_v1':
        require(receipt.get('accepted') is True and receipt.get('status') == 'verified_market_day'
                and receipt.get('automatic_redirects_disabled') is True
                and receipt.get('redirect_statuses') == [] and receipt.get('security_denied') is False,
                'Official download receipt was not safely accepted')
    elif 'status' in receipt:
        require(type(receipt['status']) is int, 'Unrecognized official receipt status')
    digest_fields = [receipt[k] for k in ('sha256', 'raw_sha256') if k in receipt]
    require(digest_fields and all(v == entry['raw_sha256'] for v in digest_fields), 'Receipt raw hash differs')
    for field in ('path', 'raw_path'):
        if field in receipt:
            require(bound_path(root, receipt[field]) == raw, 'Receipt raw path differs')
    statuses = [receipt[k] for k in ('http_status', 'status_code') if k in receipt]
    if 'status' in receipt and type(receipt['status']) is int:
        statuses.append(receipt['status'])
    if statuses:
        require(all(type(s) is int and s == 200 for s in statuses), 'Unsuccessful official receipt')
        http_status, status_evidence = 200, 'recorded_http_200'
    else:
        require(entry['receipt_sha256'] in LEGACY_RECEIPT_SHA256, 'HTTP status missing from unrecognized receipt')
        http_status, status_evidence = None, 'legacy_status_unknown'
    url = urlparse(receipt.get('url', ''))
    require(url.scheme == 'https' and url.username is None and url.password is None
            and url.port in (None, 443) and not url.fragment, 'Invalid official source URL')
    kind = ENDPOINTS.get((url.hostname, url.path))
    require(kind is not None and kind[0] == market, 'Official endpoint/market mismatch')
    query = parse_qs(url.query, keep_blank_values=True)
    require(all(len(v) == 1 for v in query.values()), 'Repeated official query parameter')
    params = {k: v[0] for k, v in query.items()}
    declared = receipt.get('params', {})
    require(isinstance(declared, dict) and all(isinstance(k, str) and isinstance(v, str)
            for k, v in declared.items()), 'Invalid receipt parameters')
    require(all(k not in params or params[k] == v for k, v in declared.items()), 'Receipt parameters conflict')
    params.update(declared)
    if kind[1] == 'twse_mi_index':
        expected = dict(date=stamp.replace('-', ''), type='ALLBUT0999', response='json')
    elif kind[1] == 'tpex_no1430':
        expected = dict(l='zh-tw', d=f'{int(stamp[:4])-1911}{stamp[4:]}'.replace('-', '/'), o='json', se='EW')
    else:
        expected = dict(date=stamp.replace('-', '/'), response='json')
    require(params == expected, 'Official request date/full-market scope differs')
    for field, expected_value in [('market', market), ('date', stamp)]:
        if field in receipt:
            require(receipt[field] == expected_value, 'Receipt identity differs')
    payload = json.loads(raw.read_text())
    rows = parse_tpex_daily_quotes(payload, stamp) if kind[1] == 'tpex_daily_quotes' else parse_market_day(payload, market, stamp)
    require(rows, 'Empty official market table')
    scope = next(iter(rows.values()))['volume_scope']
    if 'volume_scope' in receipt:
        require(receipt['volume_scope'] == scope, 'Receipt volume scope differs from endpoint')
    descriptor = dict(market=market, date=stamp, rows=len(rows), path=entry['raw_path'],
        sha256=entry['raw_sha256'], receipt=entry['receipt_path'], receipt_sha256=entry['receipt_sha256'],
        volume_scope=scope, http_status=http_status, http_status_evidence=status_evidence,
        retrieved_at=receipt.get('retrieved_at', receipt.get('observed_at')), url=receipt['url'])
    if recovery_refs:
        descriptor['recovery_source_sha256'] = recovery_refs
    return rows, descriptor


def merge_source_rows(official, rows, market, stamp):
    """Require agreeing observed prices; retain the strongest known volume scope."""
    rank = {'unclassified_daily': 0, 'all_daily_sessions': 1, 'ordinary_session': 2}
    for sid, row in rows.items():
        key = market, stamp, sid
        old = official.get(key)
        if old:
            require(all(old[f] == row[f] for f in ('open', 'high', 'low', 'close')), 'Conflicting primary daily prices')
            if old['volume_scope'] == row['volume_scope']:
                require(old['volume'] == row['volume'], 'Conflicting same-scope primary volume')
            categories = {r['table_category'] for r in (old, row) if r.get('table_category')}
            require(len(categories) <= 1, 'Conflicting primary market classifications')
            selected = old if rank[old['volume_scope']] >= rank[row['volume_scope']] else row
            # Older no1430 rows carry no table classification. Stronger volume
            # evidence must not erase a managed-board fact from another source.
            official[key] = dict(selected, table_category=next(iter(categories))) if categories else selected
        else:
            official[key] = row


def collect_supplement(manifest_path, root, refs, official=None):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    require(manifest_path.is_relative_to(root), 'Supplement manifest escapes repository')
    require(sha(manifest_path) == manifest_path.with_suffix('.sha256').read_text().strip(), 'Changed supplement manifest')
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('schema') == SCHEMA and manifest.get('live_qualified') is False
            and isinstance(manifest.get('entries'), list), 'Invalid supplement manifest')
    refs[str(manifest_path.relative_to(root))] = sha(manifest_path)
    require(isinstance(manifest.get('source_sha256', {}), dict), 'Invalid manifest source hashes')
    for name, digest in manifest.get('source_sha256', {}).items():
        source_path = bound_path(root, name)
        require(sha(source_path) == digest, 'Changed manifest source')
        require(name not in refs or refs[name] == digest, 'Source hash closure conflict')
        refs[name] = digest
    official = {} if official is None else official
    descriptors = []
    seen = set()
    for entry in manifest['entries']:
        rows, descriptor = validate_entry(entry, root)
        source_key = descriptor['market'], descriptor['date'], descriptor['volume_scope']
        require(source_key not in seen, 'Duplicate supplement market/day/scope')
        seen.add(source_key)
        source_refs = {entry['raw_path']: entry['raw_sha256'], entry['receipt_path']: entry['receipt_sha256'],
                       **descriptor.get('recovery_source_sha256', {})}
        for name, digest in source_refs.items():
            require(name not in refs or refs[name] == digest, 'Source hash closure conflict')
            refs[name] = digest
        merge_source_rows(official, rows, descriptor['market'], descriptor['date'])
        descriptors.append(descriptor)
    return official, descriptors


def execution_scope(account, official, calendar, episodes, exclusions):
    """Same independent capacity check, with an explicit unknown-volume branch."""
    positions = {d: i for i, d in enumerate(calendar)}
    scope = defaultdict(Counter)
    rows, identity_issues, used = [], [], Counter()
    for trade in account['trades']:
        stamp, sid = trade['date'], trade['stock_id']
        episode = resolve_episode(episodes, sid, stamp)
        venue = episode['market'].upper() if episode else 'UNKNOWN'
        if episode is None or episode['category'] != '股票' or excluded(exclusions, sid, stamp, venue):
            identity_issues.append(dict(stock_id=sid, date=stamp, sequence=trade['sequence']))
        if trade['channel'] != 'board':
            continue
        used[(stamp, sid)] += trade['qty']
        source = official.get((venue, stamp, sid))
        item = dict(date=stamp, stock_id=sid, sequence=trade['sequence'], market=venue,
                    qty=trade['qty'], cumulative_day_qty=used[(stamp, sid)],
                    recorded_capacity=trade['capacity_qty'], status='official_day_missing')
        if source:
            item['price_fields'] = compare_quote(dict(open=None, close=None, high=trade['source_high'],
                low=trade['source_low'], volume=trade['source_volume']), source)
            if source['volume_scope'] != 'ordinary_session':
                item['status'] = 'ordinary_volume_missing'
            else:
                cap = int(min(Decimal(source['volume']), Decimal(str(trade['prior_avg_volume20']))) * Decimal('.01')) // 1000 * 1000
                item.update(official_day_volume=source['volume'], recorded_total_volume=trade['source_volume'],
                    same_scope_day_upper_bound=cap, observed_fill_exceeds_day_bound=used[(stamp, sid)] > cap)
                i = positions[stamp]
                prior = [official.get((venue, d, sid)) for d in calendar[max(0, i-20):i]]
                item['prior_ordinary_days_observed'] = sum(r is not None and r['volume_scope'] == 'ordinary_session' for r in prior)
                if len(prior) == 20 and all(r is not None and r['volume_scope'] == 'ordinary_session' for r in prior):
                    cap = ordinary_capacity(source, prior)
                    item.update(verified_capacity=cap, status='capacity_conflict' if used[(stamp, sid)] > cap else 'capacity_matched')
                else:
                    item['status'] = 'day_bound_conflict' if used[(stamp, sid)] > cap else 'prior_ordinary_volume_incomplete'
        scope[venue][item['status']] += 1
        rows.append(item)
    return dict(markets={m: dict(v) for m, v in scope.items()}, rows=rows,
        identity_issues=identity_issues, all_capacity_verified=bool(rows)
        and all(r['status'] == 'capacity_matched' for r in rows))
