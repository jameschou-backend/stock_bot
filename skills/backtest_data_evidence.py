"""Offline, case-specific data evidence; no trading or performance qualification.

The catalog reuses the existing tape importer. Format validation and matching a
local source hash do not independently certify a complete exchange session.
"""
from collections import Counter, defaultdict
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
import json
import re

from scripts.replay_contingent_day import load_tape
from scripts.audit_market_identity import resolve_on


def digest(path):
    result = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def checked(path, expected, refs, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Evidence path escapes the repository')
    if not re.fullmatch(r'[0-9a-f]{64}', expected) or digest(path) != expected:
        raise ValueError('Evidence hash mismatch: ' + str(path))
    refs[str(path.relative_to(root))] = expected
    return path


def requirements(result):
    """Include attempted/unfilled orders, not just successful historical fills."""
    account = result.get('account') if result.get('completed') else result.get('partial_account')
    if not isinstance(account, dict) or 'orders' not in account:
        raise ValueError('Case must contain an explicit account or partial account')
    board_only = result['config']['board_only']
    required, excluded = {}, Counter()
    for row in account['orders']:
        channel, qty = row['channel'], row['requested_qty']
        if type(qty) is not int or qty < 0:
            raise ValueError('Non-negative integer order quantity required')
        if channel == 'event':
            if row.get('filled_qty') != 0 or not row.get('failure'):
                raise ValueError('Non-executable event must state a rejection and have no fills')
            # Resource/slot checks can reject a calculated positive quantity
            # before it becomes a market order. No market tape can change that.
            excluded['no_order_event'] += 1
            continue
        if channel not in ('board', 'odd'):
            raise ValueError('Unknown execution channel')
        if not qty:
            excluded['zero_quantity'] += 1
            continue
        if board_only and channel == 'odd':
            if row.get('filled_qty') != 0 or not str(row.get('failure')).startswith('board_only_'):
                raise ValueError('Board-only policy contains an actual odd-lot order/fill')
            excluded['policy_forbidden_odd_remainder'] += 1
            continue
        sid, stamp, side = row['stock_id'], row['date'], row['side']
        if not re.fullmatch(r'\d{4}', sid) or date.fromisoformat(stamp).isoformat() != stamp or side not in ('buy', 'sell'):
            raise ValueError('Invalid order identity')
        key = (stamp, sid, channel)
        item = required.setdefault(key, dict(date=stamp, stock_id=sid, channel=channel,
                                             sides=set(), order_count=0, unfilled_order_count=0))
        item['sides'].add(side)
        item['order_count'] += 1
        item['unfilled_order_count'] += int(row.get('filled_qty', 0) == 0)
    return [dict(required[k], sides=sorted(required[k]['sides'])) for k in sorted(required)], dict(excluded)


def inspect_tape(item, root, refs):
    """Re-use the production importer and bind every file read to its digest."""
    root = Path(root).resolve()
    path = checked(root / item['path'], item['sha256'], refs, root)
    if item['format'] == 'finmind_board':
        checked(path.with_suffix('.json'), item['metadata_sha256'], refs, root)
    tape = load_tape(item, root)
    if tape.channel != item['channel']:
        raise ValueError('Tape channel differs from declared identity')
    if tape.synthetic:
        raise ValueError('Synthetic tape cannot satisfy historical data evidence')
    return dict(date=tape.day, stock_id=tape.stock_id, channel=tape.channel,
        market=tape.market, path=str(path.relative_to(root)), sha256=tape.sha256,
        rows=len(tape.rows), actual_trade_rows=sum(row[3] for row in tape.rows),
        format_valid=True, status='local_format_valid_unverified',
        provider_query_identity_verified=item['format'] == 'finmind_board',
        raw_payload_and_metadata_verified=item['format'] == 'finmind_board',
        source_authenticated=False, independently_verified_session_complete=False,
        accepted_for_strict_replay=False, own_order_fill_proven=False)


def channel_evidence(required, catalog, channel):
    rows = []
    for request in required:
        if request['channel'] != channel:
            continue
        item = catalog.get((request['date'], request['stock_id'], channel))
        if item is not None and (item.get('date'), item.get('stock_id'), item.get('channel')) != (
                request['date'], request['stock_id'], channel):
            raise ValueError('Catalog key differs from source identity')
        # This version has no independent authenticated-session audit adapter.
        # Do not allow a caller-provided Boolean to promote a format-only receipt.
        if item is not None and item.get('accepted_for_strict_replay') is not False:
            raise ValueError('Unsupported source certification; use a reviewed audit adapter')
        if item is not None and item.get('format_valid') is not True:
            raise ValueError('Catalog contains an unvalidated source')
        status = 'missing' if item is None else item['status']
        if status not in ('missing', 'local_format_valid_unverified'):
            raise ValueError('Unknown source evidence status')
        rows.append(dict(request, status=status, source_path=item['path'] if item else None))
    counts = Counter(row['status'] for row in rows)
    return dict(required_sessions=len(rows),
        local_format_valid_sessions=counts['local_format_valid_unverified'], accepted_sessions=0,
        missing_sessions=counts['missing'], unverified_sessions=counts['local_format_valid_unverified'],
        status='not_required_by_policy' if not rows else 'blocked',
        missing=[r for r in rows if r['status'] == 'missing'],
        unverified=[r for r in rows if r['status'] != 'missing'])


def case_evidence(name, result, catalog, pit):
    required, excluded = requirements(result)
    account = result.get('account') or result.get('partial_account')
    components = []
    for component in pit['components']:
        component = dict(component)
        # Buying the explicit 0050 ETF does not use an all-stock industry or
        # selection universe. Preserve that global gap without inventing a
        # benchmark input dependency. Its own identity/actions still matter.
        required_by_case = not (result['config']['benchmark'] and component['code'] in (
            'historical_universe_and_eligibility', 'historical_industry_membership'))
        component['required_by_case'] = required_by_case
        if not required_by_case:
            component['global_evidence_status'] = component['status']
            component['status'] = 'not_required_by_case'
        components.append(component)
    # Resolve these exact cases, instead of borrowing the pass of an older account.
    indexed = defaultdict(list)
    for episode in pit['episodes']:
        indexed[episode['stock_id']].append(episode)
    identities, issues = set(), []
    for group in ('orders', 'holdings', 'trades'):
        for row in account.get(group, []):
            identities.add((row['date'], row['stock_id']))
    for stamp, sid in sorted(identities):
        identity = resolve_on(indexed[sid], sid, stamp)
        expected = 'ETF' if result['config']['benchmark'] and sid == '0050' else '股票'
        if identity['status'] != 'identified' or identity['category'] != expected:
            issues.append(dict(date=stamp, stock_id=sid, identity=identity))
    components = [dict(code='case_dated_market_identity', status='verified_for_observed_rows' if not issues else 'blocked',
                       rows=len(identities), issue_count=len(issues), issues=issues)] + components
    ordinary = channel_evidence(required, catalog, 'board')
    odd = channel_evidence(required, catalog, 'odd')
    codes = [row['code'] for row in components if row['status'] == 'blocked']
    if ordinary['required_sessions']:
        codes.append('ordinary_complete_authenticated_sessions')
    if odd['required_sessions']:
        codes.append('odd_lot_complete_authenticated_sessions')
    complete = result.get('completed') is True
    if not complete:
        codes.append('unobserved_path_after_blocked_session')
    return dict(name=name, case_completed_daily=complete, complete_path=complete,
        all_possible_paths_covered=False, known_path_only=True,
        last_account_date=account['daily'][-1]['date'] if account.get('daily') else None,
        source_block_reason=result.get('reason'), ordinary=ordinary, odd_lot=odd,
        excluded_events=excluded, pit=dict(components=components,
            complete_historical_universe=False, publication_time_archive_complete=False,
            complete_historical_industry_membership=False),
        missing_codes=codes, strict_data_ready=False, live_qualified=False,
        own_order_fill_proven=False)


def historical_value_available(row, decision_time):
    """An official publication timestamp must be bound to this particular version.

    A provider ingestion/first-observation date or reporting period is insufficient.
    This helper validates a record contract; it does not authenticate its document.
    """
    required = ('official_published_at', 'version_available_at', 'payload_sha256',
                'publication_source_sha256', 'version_id')
    if any(not row.get(key) for key in required):
        return False
    if any(not re.fullmatch(r'[0-9a-f]{64}', row[k]) for k in ('payload_sha256', 'publication_source_sha256')):
        raise ValueError('Publication/version source digest required')
    stamps = [datetime.fromisoformat(str(x)) for x in (
        row['official_published_at'], row['version_available_at'], decision_time)]
    if any(stamp.tzinfo is None or stamp.utcoffset() is None for stamp in stamps):
        raise ValueError('Timezone-aware publication, version and decision timestamps required')
    published, available, decision = stamps
    if available < published:
        raise ValueError('Version cannot precede its official publication')
    return available <= decision


def parse_twse_industry_change(text):
    """Extract one 47-company official event; never extrapolate full intervals."""
    rows, group, expected, counts = [], None, {}, Counter()
    lines = text.replace('\f', '\n').splitlines()
    for index, line in enumerate(lines):
        title = re.search(r'調整至「([^」]+)」：共計(\d+)家', line)
        if title:
            group = title[1]
            expected[group] = int(title[2])
            continue
        match = re.match(r'^\s*\d+\s+(\d{4})\s+(.+?)\s*$', line)
        if not match:
            continue
        if group is None:
            raise ValueError('Industry row outside a declared category')
        fields = match[2].split()
        if len(fields) == 1:
            # PDF cells for the two computer/peripheral categories wrap above and below.
            before, after = lines[index - 1].strip(), lines[index + 1].strip()
            if before != '電腦及週邊' or after != '設備業':
                raise ValueError('Unrecognized wrapped industry cell')
            company, old = fields[0], before + after
        else:
            company, old = fields[:2]
        rows.append(dict(stock_id=match[1], company_name=company, market='TWSE',
            old_industry=old, new_industry=group, announcement_date='2023-05-22',
            effective_date='2023-07-03', official_publication_time=None,
            prior_interval_start=None, next_change_date=None,
            classification_system='TWSE_official_industry',
            finmind_supply_chain_membership_equivalent=False))
        counts[group] += 1
    if len(rows) != 47 or len({r['stock_id'] for r in rows}) != 47 or dict(counts) != expected:
        raise ValueError('Official 47-company attachment is incomplete or duplicated')
    return rows


def verify_report(path, root):
    """Offline consumer boundary: verify report and every bound input/code file."""
    path, root = Path(path), Path(root)
    expected = path.with_suffix('.sha256').read_text().strip()
    checked(path, expected, {}, root)
    report = json.loads(path.read_text())
    if (report.get('schema') != 'backtest_data_completion_v1'
            or report.get('live_qualified') is not False or report.get('strict_data_ready') is not False):
        raise ValueError('Unsupported or promoted evidence report')
    refs = {}
    for name, value in report['input_sha256'].items():
        checked(root / name, value, refs, root)
    for name, value in report['code_sha256'].items():
        checked(root / name, value, refs, root)
    return report
