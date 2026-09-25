"""Versioned, evidence-bound identity corrections and retrospective trading exclusions."""
from collections import defaultdict
from copy import deepcopy
from datetime import date, timedelta
import re

from skills.market_identity_overlay import day, market
from scripts.audit_market_identity import resolve_on


def roc_day(value):
    """Only official date strings, never first-price or guessed calendar dates."""
    match = re.fullmatch(r'(?:民國|中華民國)?(\d{2,3})[年/](\d{1,2})[月/](\d{1,2})日?', value.strip())
    if not match:
        raise ValueError('Unsupported official ROC date: ' + str(value))
    year, month, number = map(int, match.groups())
    return date(year + 1911, month, number).isoformat()


def apply_completion(base, date_rows, category_rows):
    """Replace only the exact reviewed current-ISIN dates; IPO may move either way."""
    episodes = deepcopy(base['episodes'])
    seen = set()
    for row in date_rows:
        sid, venue = row['stock_id'], market(row['market'])
        if not re.fullmatch(r'[0-9]{4}', sid) or sid in seen:
            raise ValueError('Invalid or duplicate date correction')
        seen.add(sid)
        selected = [e for e in episodes if e['stock_id'] == sid and market(e['market']) == venue
                    and e['end'] is None]
        if (len(selected) != 1 or selected[0]['start'] != row['replaces_snapshot_start']
                or selected[0].get('start_evidence') != 'current_official_ISIN'):
            raise ValueError('Date correction does not bind to the exact uncorrected ISIN episode')
        selected[0].update(start=day(row['start']), start_evidence='reviewed_primary_archive_v2',
                           listing_evidence=deepcopy(row))
    seen = set()
    for row in category_rows:
        key = (row['stock_id'], market(row['market']), day(row['end']))
        if key in seen or row['category'] != '股票':
            raise ValueError('Invalid or duplicate category correction')
        seen.add(key)
        selected = [e for e in episodes if (e['stock_id'], market(e['market']), e['end']) == key]
        if len(selected) != 1 or selected[0]['category'] != 'unconfirmed':
            raise ValueError('Category correction must identify one unconfirmed ending episode')
        selected[0].update(category='股票', category_evidence=deepcopy(row))
    grouped = defaultdict(list)
    for e in episodes:
        if e['start'] is not None:
            day(e['start'])
            if e['end'] is not None and day(e['end']) <= e['start']:
                raise ValueError('Invalid listing interval')
            grouped[e['stock_id']].append(e)
    for rows in grouped.values():
        rows.sort(key=lambda e: e['start'])
        for left, right in zip(rows, rows[1:]):
            if left['end'] is None or left['end'] > right['start']:
                raise ValueError('Overlapping market identities')
    return episodes


def validate_exclusions(rows):
    seen = set()
    for row in rows:
        sid, venue, start = row['stock_id'], market(row['market']), day(row['start'])
        end = day(row['end']) if row['end'] is not None else None
        if not re.fullmatch(r'[0-9]{4}', sid) or (end and end <= start):
            raise ValueError('Invalid trading exclusion')
        if row['kind'] not in ('information_halt', 'trading_suspension', 'share_exchange', 'managed_board'):
            raise ValueError('Unreviewed exclusion type')
        key = (sid, venue, start, end, row['kind'])
        if key in seen:
            raise ValueError('Duplicate trading exclusion')
        seen.add(key)
    return deepcopy(rows)


def resolve_completion(report, stock_id, stamp):
    """An identified legal listing is not proof of continuous execution eligibility."""
    day(stamp)
    result = resolve_on(report['episodes'], stock_id, stamp)
    if stamp < report.get('coverage_start', '0001-01-01') or stamp > report['coverage_end']:
        return dict(result, status='outside_verified_coverage', tradable=None,
                    continuous_eligibility_proven=False)
    matched = [e for e in report['trading_exclusions'] if result['status'] == 'identified'
               and market(e['market']) == result['market'] and e['stock_id'] == stock_id
               and e['start'] <= stamp and (e['end'] is None or stamp < e['end'])
               and stamp <= report['coverage_end']]
    if matched:
        return dict(result, status=('not_general_board' if any(e['kind'] == 'managed_board'
                    for e in matched) else 'official_trading_suspension'), tradable=False,
                    exclusions=deepcopy(matched), continuous_eligibility_proven=False)
    return dict(result, tradable=None, continuous_eligibility_proven=False)


def parse_twse_halts(payload, source, coverage_end, year, excluded_stock_ids=()):
    expected = ['編號', '證券代號', '證券名稱', '暫停交易日期', '暫停交易時間', '恢復交易日期', '恢復交易時間']
    if payload.get('stat') != 'OK' or payload.get('fields') != expected or '全部上市證券' not in payload.get('title', ''):
        raise ValueError('Unexpected TWSE historical suspension table')
    expected_end = f'{year - 1911}/12/31' if year < int(coverage_end[:4]) else str(year - 1911) + coverage_end[4:].replace('-', '/')
    if f'期間：{year - 1911}/01/01 到 {expected_end}' not in payload['title']:
        raise ValueError('TWSE suspension query year/range differs')
    for count_field in ('total', 'totalCount'):
        if count_field in payload and payload[count_field] != len(payload['data']):
            raise ValueError('Incomplete TWSE suspension table')
    result = []
    for row in payload['data']:
        if len(row) != 7:
            raise ValueError('Incomplete TWSE suspension row')
        if not re.fullmatch(r'[0-9]{4}', row[1]):
            continue  # Explicit ordinary four-digit strategy scope; warrants are not equity candidates.
        if row[1] in excluded_stock_ids:
            continue  # The verified identity archive, not a name heuristic, excludes TDRs.
        start, end = roc_day(row[3]), roc_day(row[5])
        if not f'{year}-01-01' <= start <= min(f'{year}-12-31', coverage_end):
            raise ValueError('TWSE halt event is outside the queried source year/range')
        if row[4] != '8:00' or row[6] != '8:00':
            raise ValueError('Intraday or unknown suspension requires an intraday resolver')
        result.append(dict(stock_id=row[1], market='TWSE', start=start, end=end,
                           kind='information_halt', source_path=source, source_row=row))
    return validate_exclusions(result)


def parse_tpex_halts(sources, coverage_end):
    expected = ['編號', '有價證券類別', '有價證券代號', '有價證券名稱', '暫停交易日期', '暫停交易時間', '恢復交易日期', '恢復交易時間']
    grouped = defaultdict(list)
    for source, payload in sources:
        if payload.get('stat') != 'ok' or len(payload.get('tables', [])) != 1:
            raise ValueError('Unexpected TPEx suspension response')
        table = payload['tables'][0]
        year = int(payload['date'])
        if table['date'] != str(year) or not 2021 <= year <= int(coverage_end[:4]):
            raise ValueError('TPEx suspension year differs')
        if table['fields'] != expected or len(table['data']) != table['totalCount']:
            raise ValueError('Incomplete TPEx suspension table')
        for row in table['data']:
            if len(row) != 8 or row[1] != '上櫃股票' or not re.fullmatch(r'[0-9]{4}', row[2]):
                raise ValueError('Non-mainboard or malformed TPEx suspension row')
            for index, kind in ((4, 'halt'), (6, 'resume')):
                if row[index] != '-':
                    if row[index + 1] != '8:00':
                        raise ValueError('Intraday TPEx suspension requires explicit treatment')
                    stamp = roc_day(row[index])
                    if stamp[:4] != str(year):
                        raise ValueError('TPEx event is outside its source year')
                    grouped[row[2]].append((stamp, kind, source, row))
    result = []
    for sid, events in grouped.items():
        opened = None
        for stamp, kind, source, raw in sorted(events):
            if kind == 'halt':
                if opened is not None:
                    raise ValueError('Duplicate unmatched TPEx halt')
                opened = (stamp, source, raw)
            else:
                if opened is None or stamp <= opened[0]:
                    raise ValueError('TPEx resume lacks preceding halt')
                if opened[0] <= coverage_end:
                    result.append(dict(stock_id=sid, market='TPEx', start=opened[0], end=stamp,
                        kind='information_halt', source_path=opened[1], source_row=opened[2],
                        resume_source_path=source, resume_source_row=raw))
                opened = None
        if opened is not None and opened[0] <= coverage_end:
            raise ValueError('Open TPEx information halt needs explicit evidence review')
    return validate_exclusions(result)


def boundary_examples(report, intervals):
    result = []
    for interval in intervals:
        stamps = {interval['start'], (date.fromisoformat(interval['start']) - timedelta(days=1)).isoformat()}
        if interval['end']:
            stamps.update([interval['end'], (date.fromisoformat(interval['end']) - timedelta(days=1)).isoformat()])
        result.append(dict(stock_id=interval['stock_id'], start=interval['start'], end=interval['end'],
            results=[dict(date=stamp, **resolve_completion(report, interval['stock_id'], stamp))
                     for stamp in sorted(stamps)]))
    return result
