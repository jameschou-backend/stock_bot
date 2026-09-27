"""Verified source reuse and a fail-closed gate for fixed account replays.

Preparation may discover the data dependencies of an account. It never publishes
performance. A replay is allowed only after every selected preparation path has
completed and the exact source, signal and engine identities still match.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd

from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable, parse_limits, parse_odd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def merge_sources(output, donors, stock_ids):
    """Copy checked evidence into a new workspace, never rewrite old studies.

    Normalized content conflicts fail rather than silently choosing a donor.
    Byte-different, semantically identical dividend frames may be reused.
    """
    output = Path(output)
    if output.exists():
        raise ValueError('Source merge requires a new output directory')
    feeds = ReplayMarketFeeds(output / 'execution-feeds')
    state = feeds._state()
    (output / 'dividends').mkdir()
    origins, conflicts, rejected = {}, [], []
    canonical, raw_checked, dividend_frames = {}, {}, {}
    for directory in donors:
        directory = Path(directory)
        donor = ReplayMarketFeeds(directory / 'execution-feeds', offline=True)
        try:
            source = donor._state()
        except ReplayDataUnavailable as exc:
            rejected.append(dict(path=str(directory), reason=str(exc)))
            continue
        index_hash = digest(directory / 'execution-feeds/index.json')
        for key, entry in sorted(source['entries'].items()):
            if key.startswith('limits:') and key[7:] not in stock_ids:
                continue
            signature = (source['files_sha256'].get(entry['raw_file']),
                         source['files_sha256'].get(entry['rows_file']))
            # Always check this donor's actual bytes; equal hashes in a mutable
            # index alone do not prove that a file exists or is unchanged.
            raw_path = donor._verify_file(source, entry['raw_file'])
            rows_path = donor._verify_file(source, entry['rows_file'])
            if signature not in raw_checked:
                normalized = read(rows_path)
                if normalized['raw_sha256'] != digest(raw_path):
                    raise ReplayDataUnavailable('Donor raw/normalized link changed: ' + key)
                raw = read(raw_path)
                if key.startswith('limits:'):
                    parsed = parse_limits(raw, key[7:])
                else:
                    _, market, day = key.split(':')
                    parsed = parse_odd(raw, market, day)
                if parsed != normalized['rows']:
                    raise ReplayDataUnavailable('Donor normalized content differs from parser: ' + key)
                # Keep only content fingerprints, not every stock of every
                # market-day in memory. This also avoids parsing donor copies.
                raw_checked[signature] = hashlib.sha256(json.dumps(parsed, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            content_hash = raw_checked[signature]
            if key in canonical:
                if canonical[key] != content_hash:
                    conflicts.append(dict(kind='execution', key=key, donor=str(directory)))
                continue
            canonical[key] = content_hash
            for name in (entry['raw_file'], entry['rows_file']):
                target = output / 'execution-feeds' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(directory / 'execution-feeds' / name, target)
                state['files_sha256'][name] = digest(target)
            state['entries'][key] = entry
            origins[key] = dict(donor=str(directory), index_sha256=index_hash,
                                raw_sha256=signature[0], rows_sha256=signature[1])
        for path in sorted((directory / 'dividends').glob('*.parquet')):
            sid = path.stem
            if sid not in stock_ids:
                continue
            frame = pd.read_parquet(path)
            if not frame.empty and ('stock_id' not in frame or not frame.stock_id.astype(str).eq(sid).all()):
                raise ValueError('Wrong stock in donated dividends: ' + sid)
            if sid in dividend_frames:
                if not dividend_frames[sid].equals(frame):
                    conflicts.append(dict(kind='dividend', key=sid, donor=str(directory)))
                continue
            dividend_frames[sid] = frame
            shutil.copyfile(path, output / 'dividends' / path.name)
            origins['dividend:' + sid] = dict(donor=str(path), sha256=digest(path))
    write(output / 'execution-feeds/index.json', state)
    receipt = dict(origins=origins, conflicts=conflicts, rejected_donors=rejected,
                   requests=0, policy='identical normalized content; conflicts block preparation')
    write(output.parent / 'reuse.json', receipt)
    return receipt


class CachedPreparationFeeds(ReplayMarketFeeds):
    """Reuse already verified responses in one run; boundary hashes remain required."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memo = {}

    def get_limits(self, sid):
        key = 'limits:' + sid
        if key not in self.memo:
            self.memo[key] = super().get_limits(sid)
        return deepcopy(self.memo[key])

    def get_odd(self, day, sid, market):
        key = (str(day), sid, market.lower())
        if key not in self.memo:
            try:
                self.memo[key] = super().get_odd(day, sid, market)
            except ReplayDataUnavailable as exc:
                raise ReplayDataUnavailable(f'{market} odd-lot evidence missing: {sid} {day}; {exc}') from None
        return deepcopy(self.memo[key])


def inventory(inputs, stock_ids):
    """Catalog coverage is deliberately separate from executable-path readiness."""
    inputs = Path(inputs)
    state = ReplayMarketFeeds(inputs / 'execution-feeds', offline=True)._state()
    limits = {key[7:] for key in state['entries'] if key.startswith('limits:')}
    dividends = {p.stem for p in (inputs / 'dividends').glob('*.parquet')}
    return dict(candidate_stock_count=len(stock_ids),
        missing_limit_stocks=sorted(stock_ids - limits),
        missing_dividend_stocks=sorted(stock_ids - dividends),
        odd_market_dates={m: sorted(k.split(':')[2] for k in state['entries'] if k.startswith('odd:' + m + ':'))
                          for m in ('twse', 'tpex')},
        scope='Candidate catalog only. A cached stock does not prove all dates or corporate terms are complete.')


def source_identity(inputs):
    inputs = Path(inputs)
    ReplayMarketFeeds(inputs / 'execution-feeds', offline=True).manifest()
    return {str(p.relative_to(inputs)): digest(p) for p in sorted(inputs.rglob('*'))
            if p.is_file() and p.suffix != '.lock'}


def gate(receipt, identity, expected_cases):
    issues = []
    if receipt.get('identity') != identity:
        issues.append(dict(code='identity_changed', message='Signals, engine or prepared sources changed; prepare again.'))
    if set(receipt.get('cases', {})) != set(expected_cases):
        issues.append(dict(code='case_set_changed', message='Every requested account must have preparation evidence.'))
    for name in expected_cases:
        row = receipt.get('cases', {}).get(name, {})
        if row.get('completed') is not True:
            issues.append(dict(code='incomplete_path', case=name, message=row.get('reason') or 'No complete preparation path'))
    return dict(ready=not issues, issues=issues, live_qualified=False,
                scope='Exact fixed signals, engine, sources and scenarios; daily execution model only.')


def suspension_terms(document, root):
    """Positive official suspension evidence, never absence-of-quote inference."""
    from datetime import date
    root = Path(root).resolve()
    evidence = document['evidence_sha256']
    for name, checksum in evidence.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or digest(path) != checksum:
            raise ValueError('Suspension evidence changed')
    terms = {}
    for row in document['suspensions']:
        sid = row['stock_id']
        known, start, end = [date.fromisoformat(row[k]) for k in ('known_date', 'start', 'end')]
        if (len(sid) != 4 or not sid.isdigit() or not known <= start <= end
                or not row.get('evidence_files') or any(p not in evidence for p in row['evidence_files'])):
            raise ValueError('Invalid official suspension terms')
        for day in pd.date_range(start, end):
            key = (sid, str(day.date()))
            if key in terms:
                raise ValueError('Overlapping official suspension terms')
            terms[key] = row
    return terms


class SuspensionOrders:
    """Reject suspended orders explicitly; preserve missing OHLC as missing."""
    verified_suspensions = {}

    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        term = self.verified_suspensions.get((sid, str(day.date())))
        if term is None:
            return super()._execute_order(day, sid, side, qty, reason, event_id, signal_date)
        if type(qty) is not int or qty < 0:
            raise ValueError('Order shares must be integer and nonnegative')
        if self.raw(day, sid, 'volume'):
            raise ReplayDataUnavailable('Positive traded volume conflicts with official suspension')
        if side == 'sell':
            qty = min(qty, self.holdings.get(sid, {}).get('qty', 0))
        for channel, requested in [('board', qty//1000*1000), ('odd', qty%1000)]:
            if requested:
                self.orders.append(dict(date=str(day.date()), stock_id=sid, name=self.names.get(sid, sid),
                    side=side, channel=channel, requested_qty=requested, filled_qty=0, reason=reason,
                    event_id=event_id, signal_date=signal_date, failure='official_trading_suspension',
                    suspension_known_date=term['known_date'], evidence_files=term['evidence_files']))
        return 0


def prepared_case(data, config, inputs, overrides, **kwargs):
    from unittest.mock import patch
    from skills import sector_account_replay as engine
    from skills import execution_resources, pending_share_entitlements
    from skills.corporate_account_audit import audit_corporate_account
    from skills.prepared_corporate_settlement import CapitalSettlementActions, validate_delivery_terms
    document = Path(inputs)/'trading-status.json'
    terms = suspension_terms(read(document), Path(__file__).resolve().parents[1]) if document.exists() else {}

    class PreparedMixed(SuspensionOrders, engine.SectorMixedReplay):
        verified_suspensions = terms

        def __init__(self, *args, **options):
            super().__init__(*args, **options)
            self.corporate = CapitalSettlementActions(self.corporate, self)

        def cash_move(self, day, kind, change, **extra):
            if extra.get('action_id', '').endswith('-capital-cash'):
                # The sealed daily bridge groups payments by its legacy kind.
                # Preserve that algebra; explicitly identify the economic nature.
                extra['cash_flow_nature'] = 'capital_return'
            return super().cash_move(day, kind, change, **extra)

    # The existing engine, rules, accounting and audits remain in use. This
    # adapter is explicit and separately included in the new source identity.
    with patch.object(engine, 'SectorMixedReplay', PreparedMixed), \
            patch.object(execution_resources, 'audit_stress', audit_corporate_account), \
            patch.object(pending_share_entitlements, 'validate_pending_terms', validate_delivery_terms):
        return engine.run_case(data, config, inputs, overrides, **kwargs)
