"""Bounded primary-source identity corrections, without claiming a full PIT universe."""
from copy import deepcopy
import re

from skills.market_identity_overlay import apply_starts, day, market
from scripts.audit_market_identity import resolve_on
from scripts.audit_identity_continuation import compact


def parse_industry_changes(text, *, year, expected_count):
    """Parse reviewed TPEx announcements, including two original 2025 typos."""
    text = compact(text)
    effective = f'{year - 1911}年6月{1 if year == 2026 else 2}日'
    if effective not in text or '證券代號不予變更' not in text:
        raise ValueError('Industry announcement effective date or unchanged-code clause differs')
    pattern = r'股票代號[:：](\d{4})[）)]?由「([^」]+)」+調整為「([^」]+)」'
    rows = [dict(stock_id=sid, prior_industry=old, new_industry=new)
            for sid, old, new in re.findall(pattern, text)]
    if len(rows) != expected_count or len({r['stock_id'] for r in rows}) != expected_count:
        raise ValueError('Incomplete or duplicated reviewed industry announcement')
    return rows


def apply_followup(base, rows, suspensions):
    """Return a copy and keep suspension exclusions separate from legal listing episodes."""
    episodes = apply_starts(base['episodes'], rows)
    seen = set()
    for row in suspensions:
        key = (row['stock_id'], market(row['market']))
        start, end = day(row['start']), day(row['end'])
        matches = [r for r in episodes if (r['stock_id'], market(r['market'])) == key
                   and r['end'] == end]
        if key in seen or len(matches) != 1 or start >= end:
            raise ValueError('Invalid or duplicate suspension interval')
        seen.add(key)
        episode = matches[0]
        if episode['start'] is None or start < episode['start'] or end != episode['end']:
            raise ValueError('Suspension must bind to the reviewed ending episode')
    return episodes, deepcopy(suspensions)


def resolve_followup(report, stock_id, stamp):
    """Retrospective identity lookup that also rejects verified trading suspensions."""
    day(stamp)
    result = resolve_on(report['episodes'], stock_id, stamp)
    for interval in report['trading_exclusions']:
        if (interval['stock_id'] == stock_id and interval['start'] <= stamp < interval['end']):
            return dict(result, status='official_trading_suspension', tradable=False,
                        exclusion=deepcopy(interval))
    # Identified is deliberately not a complete historical tradability permit.
    return dict(result, continuous_eligibility_proven=False)
