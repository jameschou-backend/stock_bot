"""Retrospective listing evidence and dated identity checks, never a trading permit."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
import re

from scripts.audit_market_identity import resolve_on


def market(value):
    if value not in ('TWSE', 'TPEx', 'TPEX'):
        raise ValueError('Unsupported market identity')
    return 'TWSE' if value == 'TWSE' else 'TPEx'


def day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('Canonical ISO date required')
    return value


def apply_starts(episodes, evidence):
    """Fill unknown starts or explicitly reviewed current-ISIN date corrections."""
    result = deepcopy(episodes)
    seen = set()
    for row in evidence:
        sid = row['stock_id']
        if not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid):
            raise ValueError('Four-digit stock identity required')
        venue, start = market(row['market']), day(row['start'])
        key = (sid, venue)
        if key in seen:
            raise ValueError('Repeated listing evidence requires review')
        seen.add(key)
        matches = [e for e in result if e['stock_id'] == sid and market(e['market']) == venue
                   and (row.get('end') is None or e['end'] == row['end'])]
        if len(matches) != 1:
            raise ValueError('Listing evidence must identify exactly one episode')
        episode = matches[0]
        if episode['start'] is not None:
            # ISIN's current date can mean a rename/share-exchange or industry update.
            # A correction must bind to the precise prior observation, not overwrite
            # a verified original listing or silently replace any non-null date.
            if (episode.get('start_evidence') != 'current_official_ISIN'
                    or row.get('replaces_snapshot_start') != episode['start']
                    or start >= episode['start'] or episode['end'] is not None):
                raise ValueError('Refusing to replace an already known listing date')
        elif row.get('replaces_snapshot_start') is not None:
            raise ValueError('Snapshot correction does not match an existing start')
        if episode['end'] is not None and start >= day(episode['end']):
            raise ValueError('Listing start must precede exclusive end')
        category = row['category']
        if category not in ('股票', '臺灣存託憑證(TDR)', '創新板', 'ETF', 'unconfirmed'):
            raise ValueError('Unsupported security category')
        if episode['category'] not in ('unconfirmed', category):
            raise ValueError('Conflicting security category')
        episode.update(start=start, category=category, start_evidence='reviewed_primary_archive',
                       listing_evidence=deepcopy(row))
    # Transfers with end == next start are valid. A date fill cannot hide an overlap.
    groups = defaultdict(list)
    for row in result:
        if row['start'] is not None:
            day(row['start'])
            if row['end'] is not None and day(row['end']) <= row['start']:
                raise ValueError('Invalid episode boundary')
            groups[row['stock_id']].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: r['start'])
        for left, right in zip(rows, rows[1:]):
            if left['end'] is None or left['end'] > right['start']:
                raise ValueError('Overlapping verified market episodes')
    return result


def audit_rows(episodes, rows, static_markets, *, benchmark=False):
    """One result per record, also checking the venue used by the frozen engine."""
    indexed = defaultdict(list)
    for episode in episodes:
        indexed[episode['stock_id']].append(episode)
    counts, issues = Counter(), []
    for index, row in enumerate(rows):
        sid, stamp = row['stock_id'], day(row['date'])
        identity = resolve_on(indexed[sid], sid, stamp)
        counts[identity['status']] += 1
        reasons = []
        if identity['status'] != 'identified':
            reasons.append(identity['status'])
        else:
            expected_category = 'ETF' if benchmark and sid == '0050' else '股票'
            if identity['category'] != expected_category:
                reasons.append('security_category')
            static = static_markets.get(sid)
            if static is None or market(static) != identity['market']:
                reasons.append('execution_market')
        if reasons:
            issues.append(dict(index=index, stock_id=sid, date=stamp, reasons=reasons,
                               identity=identity, engine_market=static_markets.get(sid),
                               context={k:row[k] for k in ('event_id','leader','group_id') if k in row}))
    return dict(rows=sum(counts.values()), status_counts=dict(counts), issues=issues,
                issue_rows=len(issues), passed=not issues,
                continuous_eligibility_proven=False, publication_time_archive_complete=False)
