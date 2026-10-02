"""Offline, session-specific execution volume evidence.

Inputs are the normalized, hash-validated official rows from the market-table
readers. This module never promotes provider prints or all-session totals into
ordinary-session volume. Halt evidence describes absent trading, not a price.
"""
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from skills.market_input_validation import numeric, require


def _stamp(value):
    require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value,
            'Canonical ISO date required')
    return value


def _market(value):
    value = value.upper()
    require(value in ('TWSE', 'TPEX'), 'Unsupported market')
    return value


def _reference(root, name, expected, refs):
    root = Path(root).resolve()
    path = (root/name).resolve()
    require(path.is_relative_to(root), 'Evidence escapes repository')
    actual = sha256(path.read_bytes()).hexdigest()
    require(actual == expected, 'Official evidence hash mismatch')
    key = str(path.relative_to(root))
    require(key not in refs or refs[key] == actual, 'Conflicting source hash')
    refs[key] = actual
    return path


def verify_halts(halts, root, refs):
    """Validate reviewed notice records; interval ends are exclusive.

    The caller must also bind the metadata document's hash. This validates the
    referenced notice bytes and point-in-time dates, not the notice's language.
    """
    result = []
    for original in halts:
        row = dict(original)
        row['market'] = _market(row['market'])
        require(re.fullmatch(r'\d{4}', row['stock_id']) is not None,
                'Invalid halt stock identity')
        require(row.get('kind') == 'trading_suspension', 'Only full-session suspension is supported')
        start, end = _stamp(row['start']), _stamp(row['end'])
        published = _stamp(row['announcement_date'])
        known = _stamp(row.get('known_by', row['announcement_date']))
        require(published <= known < start < end, 'Halt must be known before its first full session')
        _reference(root, row['source_path'], row['source_sha256'], refs)
        row['known_by'] = known
        for prior in result:
            if (prior['market'], prior['stock_id']) == (row['market'], row['stock_id']):
                require(end <= prior['start'] or start >= prior['end'], 'Overlapping halt evidence')
        result.append(row)
    return result


def benchmark_split_halt(evidence_path, root, refs, *, expected_metadata_sha256=None):
    """Use the advance schedule, never the later split-completion announcement."""
    path = Path(evidence_path)
    if not path.is_absolute():
        path = Path(root)/path
    root, path = Path(root).resolve(), path.resolve()
    require(path.is_relative_to(root), 'Evidence escapes repository')
    relative = str(path.relative_to(root))
    expected = expected_metadata_sha256 or refs.get(relative)
    require(expected is not None, 'Reviewed benchmark metadata hash must be pinned before loading')
    path = _reference(root, relative, expected, refs)
    evidence = json.loads(path.read_text())
    require(evidence.get('stock_id') == '0050'
            and evidence.get('suspension_known_before_start_verified') is True,
            'Verified benchmark suspension schedule required')
    schedule, terms = evidence['schedule_source'], evidence['verified_terms']
    url = urlparse(schedule['url'])
    require(url.scheme == 'https' and url.hostname == 'www.twse.com.tw', 'Official TWSE schedule required')
    start, last = _stamp(terms['suspension_start']), _stamp(terms['suspension_end'])
    end = (date.fromisoformat(last)+timedelta(days=1)).isoformat()
    require(_stamp(schedule['published_on']) <= _stamp(schedule['conservative_known_by']) < start
            and start <= last and end == terms['new_units_listing_date']
            and _stamp(terms['last_trading_date_before_split']) < start,
            'Inconsistent benchmark halt schedule')
    halt = dict(stock_id='0050', market='TWSE', kind='trading_suspension',
                start=start, end=end, announcement_date=schedule['published_on'],
                known_by=schedule['conservative_known_by'],
                source_path=schedule['local_path'], source_sha256=schedule['sha256'])
    return verify_halts([halt], root, refs)[0]


class OrdinaryVolumeEvidence:
    """Read exact ordinary volume while retaining total-volume gaps explicitly.

    ``official`` is keyed by (market, ISO date, stock_id); ``market_days`` is the
    set of full market/day tables accepted by their source readers. ``halts``
    must come from verify_halts/benchmark_split_halt. No method mutates inputs.
    """
    def __init__(self, official, market_days, *, halts=()):
        self.official = official
        self.market_days = {(_market(m), _stamp(d)) for m, d in market_days}
        self.halts = {}
        for halt in halts:
            require(halt.get('known_by', '9999') < halt['start'] < halt['end'],
                    'Verified advance halt evidence required')
            self.halts.setdefault((_market(halt['market']), halt['stock_id']), []).append(halt)

    def observation(self, market, stamp, stock_id):
        market, stamp = _market(market), _stamp(stamp)
        require(re.fullmatch(r'\d{4}', stock_id) is not None, 'Invalid stock identity')
        row = self.official.get((market, stamp, stock_id))
        item = dict(market=market, date=stamp, stock_id=stock_id, ordinary_volume=None,
                    total_volume=None, lower_bound=0, upper_bound=None,
                    status='official_market_day_missing', executable=False)
        if (market, stamp) not in self.market_days:
            return item
        if row is not None:
            require((row['market'].upper(), row['date'], row['stock_id']) == (market, stamp, stock_id),
                    'Official row identity differs from its key')
        matched_halts = [h for h in self.halts.get((market, stock_id), [])
                         if h['start'] <= stamp < h['end']]
        require(len(matched_halts) <= 1, 'Ambiguous halt evidence')
        if matched_halts:
            if row is not None:
                require(numeric(row['volume'], integral=True) == 0
                        and all(row.get(k) in (None, 0) for k in ('open', 'high', 'low', 'close')),
                        'Full-session halt conflicts with an official quote')
            return item | dict(status='verified_full_session_halt', ordinary_volume=0,
                               total_volume=0, lower_bound=0, upper_bound=0,
                               source_path=matched_halts[0]['source_path'],
                               source_sha256=matched_halts[0]['source_sha256'],
                               price_observation=None)
        if row is None:
            return item | dict(status='official_stock_row_missing')
        if row.get('table_category') == '管理股票':
            return item | dict(status='managed_board_not_ordinary')
        volume = numeric(row['volume'], integral=True)
        scope = row['volume_scope']
        if scope == 'ordinary_session':
            return item | dict(status='official_ordinary_volume', ordinary_volume=volume,
                               lower_bound=volume, upper_bound=volume, executable=volume > 0)
        if scope == 'all_daily_sessions':
            return item | dict(status='only_all_session_volume', total_volume=volume, upper_bound=volume)
        return item | dict(status='unclassified_volume_scope')

    def capacity(self, market, stamp, stock_id, calendar, *, cumulative_qty=None):
        """Exact ordinary cap or mathematically valid bounds, never fallback.

        Bounds do not certify missing sources or execution. A current zero
        volume (including an official halt) can establish zero capacity without
        inventing the missing preceding history.
        """
        calendar = [_stamp(d) for d in calendar]
        require(calendar == sorted(set(calendar)) and stamp in calendar, 'Invalid trading calendar')
        index = calendar.index(stamp)
        window = calendar[max(0, index-20):index+1]
        observations = [self.observation(market, d, stock_id) for d in window]
        current, prior = observations[-1], observations[:-1]
        gaps = [r for r in observations if r['ordinary_volume'] is None]
        result = dict(market=_market(market), date=stamp, stock_id=stock_id,
                      current_ordinary_volume=current['ordinary_volume'],
                      prior_ordinary_days_observed=sum(r['ordinary_volume'] is not None for r in prior),
                      prior_required_dates=window[:-1], gaps=gaps, verified_capacity=None,
                      lower_capacity=None, upper_capacity=None, status='ordinary_volume_incomplete')
        if len(prior) != 20:
            return result | dict(status='calendar_warmup_incomplete')
        def cap(now, before):
            return int(min(Decimal(now), sum(Decimal(v) for v in before)/20)*Decimal('.01'))//1000*1000
        lower = cap(current['lower_bound'], [r['lower_bound'] for r in prior])
        upper = (cap(current['upper_bound'], [r['upper_bound'] for r in prior])
                 if all(r['upper_bound'] is not None for r in observations) else None)
        result.update(lower_capacity=lower, upper_capacity=upper)
        if not gaps:
            result.update(verified_capacity=lower, status='ordinary_capacity_verified',
                          prior_ordinary_average=float(sum(Decimal(r['ordinary_volume']) for r in prior)/20))
        if cumulative_qty is not None:
            qty = numeric(cumulative_qty, integral=True)
            require(qty % 1000 == 0, 'Board quantity must be whole lots')
            result['cumulative_qty'] = qty
            result['fill_check'] = ('capacity_matched' if not gaps and qty <= lower else
                'capacity_conflict' if not gaps or (upper is not None and qty > upper) else
                'bounded_capacity_sufficient' if qty <= lower else 'capacity_unverified')
        return result

    def matrix(self, market, calendar, stock_ids):
        """Exact ordinary shares only; NaN is a gap, never silently zero-filled."""
        import pandas as pd
        calendar = [_stamp(d) for d in calendar]
        require(calendar == sorted(set(calendar)) and len(stock_ids) == len(set(stock_ids)),
                'Matrix axes must be unique and dates sorted')
        return pd.DataFrame({sid: [self.observation(market, day, sid)['ordinary_volume']
                                  for day in calendar] for sid in stock_ids},
                            index=pd.DatetimeIndex(calendar, name='date'), dtype=float)


def execution_capacity_report(trades, calendar, evidence, resolve_market):
    """Recheck the existing account's exact footprint, including prior 20 days."""
    rows, required, gaps, used = [], set(), {}, Counter()
    for trade in trades:
        if trade['channel'] != 'board':
            continue
        market, stamp, sid = _market(resolve_market(trade)), trade['date'], trade['stock_id']
        used[(market, stamp, sid)] += numeric(trade['qty'], integral=True)
        result = evidence.capacity(market, stamp, sid, calendar,
                                   cumulative_qty=used[(market, stamp, sid)])
        result.update(sequence=trade['sequence'], original_capacity=trade['capacity_qty'])
        rows.append(result)
        required.update((market, d, sid) for d in result['prior_required_dates']+[stamp])
        for gap in result['gaps']:
            gaps[(market, gap['date'], sid)] = gap
    return dict(schema='ordinary_capacity_evidence_v1', rows=rows,
                counts=dict(Counter(r.get('fill_check', r['status']) for r in rows)),
                required_stock_days=len(required),
                required_market_days=[dict(market=m, date=d) for m, d in sorted({(m,d) for m,d,s in required})],
                missing_stock_days=[gaps[k] for k in sorted(gaps)],
                missing_market_days=[dict(market=m, date=d) for m,d in sorted({(m,d) for m,d,s in gaps})],
                all_capacity_verified=bool(rows) and all(r.get('fill_check') == 'capacity_matched' for r in rows),
                provider_tick_volume_used=False, return_recomputed=False,
                live_qualified=False, actual_fill_verified=False)
