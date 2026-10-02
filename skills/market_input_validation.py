"""Independent daily-market checks with explicit field and coverage boundaries.

Total daily shares, ordinary-session shares and missing observations must never
be substituted for each other. A matched price does not certify execution.
"""
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
import math
import re


class MarketEvidenceError(ValueError):
    pass


CHECK_NAMES = ('all_execution_prices_verified','all_holding_marks_verified',
               'all_ordinary_capacity_verified','all_signal_histories_verified',
               'all_historical_market_days_observed','known_identity_checks_passed',
               'all_observed_positive_quotes_present','complete_historical_universe',
               'no_observed_price_conflicts')


def require(ok, message):
    if not ok:
        raise MarketEvidenceError(message)


def numeric(value, *, integral=False, missing=False):
    text = str(value).strip().replace(',', '')
    if missing and text in ('', '-', '--', '---', '----', 'None'):
        return None
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise MarketEvidenceError('Invalid official number') from exc
    require(not isinstance(value, bool) and result.is_finite() and result >= 0, 'Invalid official number')
    require(not integral or result == result.to_integral_value(), 'Non-integer share volume')
    return int(result) if integral else float(result)


def parse_market_day(payload, market, stamp):
    """Normalize full official tables, retaining their different volume scopes."""
    day = date.fromisoformat(stamp)
    require(payload.get('date') == stamp.replace('-', '') and str(payload.get('stat')).lower() == 'ok',
            'Official date/status mismatch')
    require(market in ('TWSE', 'TPEX'), 'Unsupported market')
    if market == 'TWSE':
        tables = [t for t in payload.get('tables', []) if '證券代號' in t.get('fields', [])]
        require(payload.get('type') == 'ALLBUT0999' and len(tables) == 1, 'Incomplete TWSE market scope')
        table = tables[0]
        require(any('含一般、零股、盤後定價、鉅額交易' in n for n in table.get('notes', [])),
                'TWSE volume scope missing')
        fields = dict(stock_id='證券代號', name='證券名稱', open='開盤價', high='最高價',
                      low='最低價', close='收盤價', volume='成交股數')
        scope = 'all_daily_sessions'
    else:
        require(len(payload.get('tables', [])) == 1, 'Ambiguous TPEx table')
        table = payload['tables'][0]
        require(table.get('title') == '上櫃股票每日收盤行情(不含定價)'
                and table.get('date') == f'{day.year-1911}/{day.month:02d}/{day.day:02d}'
                and table.get('category') == '所有證券(不含權證、牛熊證)', 'Incomplete/wrong TPEx market scope')
        fields = dict(stock_id='代號', name='名稱', open='開盤', high='最高', low='最低', close='收盤', volume='成交股數')
        scope = 'ordinary_session'
    headers = [h.strip() for h in table['fields']]
    require(len(headers) == len(set(headers)) and set(fields.values()).issubset(headers), 'Missing or duplicate headers')
    # MI_INDEX's full-class table has no count field; its requested/echoed
    # ALLBUT0999 scope and complete JSON are retained as the coverage evidence.
    require(market == 'TWSE' or 'totalCount' in table or 'total' in table, 'Official row count absent')
    for key in ('totalCount', 'total'):
        if key in table:
            require(numeric(table[key], integral=True) == len(table['data']), 'Incomplete official table')
    result = {}
    for values in table['data']:
        require(len(values) == len(headers), 'Official row width mismatch')
        source = dict(zip(headers, values))
        sid = str(source[fields['stock_id']]).strip()
        if not re.fullmatch(r'[0-9]{4}', sid):
            continue  # Strategy ordinary-share/0050 namespace; no warrants inferred from names.
        require(sid not in result, 'Duplicate official stock')
        item = dict(date=stamp, stock_id=sid, name=source[fields['name']], market=market,
                    volume_scope=scope, volume=numeric(source[fields['volume']], integral=True))
        for name in ('open', 'high', 'low', 'close'):
            value = numeric(source[fields[name]], missing=True)
            item[name] = value if value and value > 0 else None
        if all(item[k] is not None for k in ('open', 'high', 'low', 'close')):
            require(item['low'] <= min(item['open'], item['close'])
                    <= max(item['open'], item['close']) <= item['high'], 'Impossible official OHLC')
        result[sid] = item
    return result


def resolve_episode(episodes, sid, stamp):
    matched = [e for e in episodes if e['stock_id'] == sid and e.get('start')
               and e['start'] <= stamp and (not e.get('end') or stamp < e['end'])]
    require(len(matched) <= 1, 'Overlapping dated market identities')
    return matched[0] if matched else None


def excluded(exclusions, sid, stamp, market):
    """Only evidenced full-session restrictions can exclude daily execution."""
    for row in exclusions:
        if row['stock_id'] != sid or row['market'].upper() != market or not row.get('start'):
            continue
        if not (row['start'] <= stamp and (not row.get('end') or stamp < row['end'])):
            continue
        if row['kind'] == 'information_halt':
            source = row.get('source_row', [])
            # TWSE and TPEx have different offsets, both exact 08:00 boundaries.
            ix = (4, 6) if len(source) == 7 else ((5, 7) if len(source) == 8 else None)
            if ix is None or any(source[i] not in ('8:00', '08:00') for i in ix):
                continue
        if row['kind'] in ('information_halt', 'trading_suspension', 'share_exchange', 'managed_board'):
            return row['kind']
    return None


def compare_quote(local, official):
    """Missing or incomparable fields remain unknown, never counted as matched."""
    fields = {}
    for field in ('open', 'high', 'low', 'close'):
        left, right = local.get(field), official[field]
        if left is None or not math.isfinite(float(left)) or float(left) <= 0 or right is None:
            fields[field] = 'unavailable'
        else:
            fields[field] = 'matched' if abs(float(left)-right) < 1e-8 else 'conflict'
    if official['volume_scope'] == 'all_daily_sessions':
        fields['total_volume'] = 'matched' if local.get('volume') == official['volume'] else 'conflict'
        fields['ordinary_volume'] = 'unverified'
    else:
        fields['total_volume'] = 'different_scope'
        fields['ordinary_volume'] = 'official_observed'
    return fields


def ordinary_capacity(current, prior):
    """Only twenty complete prior same-scope rows certify ordinary capacity."""
    require(current['volume_scope'] == 'ordinary_session', 'Total daily volume is not ordinary volume')
    require(len(prior) == 20 and all(r is not None and r['volume_scope'] == 'ordinary_session' for r in prior),
            'Twenty independently observed prior ordinary volumes are required')
    require(all(r['date'] < current['date'] and r['stock_id'] == current['stock_id']
                and r['market'] == current['market'] for r in prior), 'Prior-volume timing or identity differs')
    require(len({r['date'] for r in prior}) == 20, 'Duplicate prior volume observation')
    limit = min(Decimal(current['volume']), sum(Decimal(r['volume']) for r in prior)/20)
    return int(limit*Decimal('.01'))//1000*1000


def execution_scope(account, official, calendar, episodes, exclusions):
    positions = {d:i for i,d in enumerate(calendar)}
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
        used[(stamp,sid)] += trade['qty']
        source = official.get((venue, stamp, sid))
        item = dict(date=stamp, stock_id=sid, sequence=trade['sequence'], market=venue,
                    qty=trade['qty'],cumulative_day_qty=used[(stamp,sid)],
                    recorded_capacity=trade['capacity_qty'],status='official_day_missing')
        if source:
            fields = compare_quote(dict(open=None, close=None, high=trade['source_high'],
                                        low=trade['source_low'], volume=trade['source_volume']), source)
            item['price_fields'] = fields
            if source['volume_scope'] != 'ordinary_session':
                item['status'] = 'ordinary_volume_missing'
            else:
                current_cap = int(min(Decimal(source['volume']), Decimal(str(trade['prior_avg_volume20'])))*Decimal('.01'))//1000*1000
                item.update(official_day_volume=source['volume'], recorded_total_volume=trade['source_volume'],
                            same_scope_day_upper_bound=current_cap,
                            observed_fill_exceeds_day_bound=used[(stamp,sid)] > current_cap)
                i = positions[stamp]
                prior = [official.get((venue, d, sid)) for d in calendar[max(0,i-20):i]]
                item['prior_ordinary_days_observed'] = sum(r is not None and r['volume_scope']=='ordinary_session' for r in prior)
                if len(prior) == 20 and all(r is not None and r['volume_scope']=='ordinary_session' for r in prior):
                    cap = ordinary_capacity(source, prior)
                    item.update(verified_capacity=cap, status='capacity_conflict' if used[(stamp,sid)] > cap else 'capacity_matched')
                else:
                    item['status'] = 'day_bound_conflict' if used[(stamp,sid)] > current_cap else 'prior_ordinary_volume_incomplete'
        scope[venue][item['status']] += 1
        rows.append(item)
    return dict(markets={m:dict(v) for m,v in scope.items()}, rows=rows,
                identity_issues=identity_issues, all_capacity_verified=bool(rows)
                and all(r['status']=='capacity_matched' for r in rows))


def require_complete(report):
    """Research can display incomplete data; verified-data mode must stop."""
    checks = report['checks']
    failures = [name for name in CHECK_NAMES if checks.get(name) is not True]
    require(not failures, 'Verified-data replay unavailable: '+', '.join(failures))
    require(report.get('live_qualified') is False, 'Data validation must not qualify live trading')
